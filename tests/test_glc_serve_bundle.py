"""CPU tests: serving-bundle format, shard hashing, remote fetch, loader, gate.

Every test runs on a tiny random Qwen3.5 VL checkpoint (``glc_serve_tiny``):
the same architecture class as Qwen3.8-27B, with a vision tower, an untied
head and an MTP block, so the full multimodal load path is exercised.
"""
from __future__ import annotations

import functools
import http.server
import json
import os
import shutil
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "release", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from glc_serve import bundle as B  # noqa: E402
from glc_serve.gate import run_weight_gate  # noqa: E402
from glc_serve.loader import (  # noqa: E402
    LoadError,
    ServeOptions,
    _disable_cuda_only_conv_kernels,
    load_bundle_model,
    load_dense_parent,
    plan_offload,
)
from glc_serve.pack import pack_from_container, pack_from_dense  # noqa: E402

IDS = torch.tensor([[5, 17, 33, 90, 200, 7, 8, 9, 41, 3]])


def test_cpu_kernel_fallback_is_scoped_to_that_model_instance():
    class OptionalKernelModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.causal_conv1d_fn = object()
            self.causal_conv1d_update = object()
            self.chunk_gated_delta_rule = object()
            self.recurrent_gated_delta_rule = object()

    cpu_model = OptionalKernelModule()
    cuda_model = OptionalKernelModule()
    cuda_kernels = tuple(getattr(cuda_model, name) for name in (
        "causal_conv1d_fn", "causal_conv1d_update",
        "chunk_gated_delta_rule", "recurrent_gated_delta_rule",
    ))

    _disable_cuda_only_conv_kernels(cpu_model, torch.device("cpu"))
    _disable_cuda_only_conv_kernels(cuda_model, torch.device("cuda:0"))

    assert cpu_model.causal_conv1d_fn is None
    assert cuda_model.causal_conv1d_fn is cuda_kernels[0]
    assert cuda_model.causal_conv1d_update is cuda_kernels[1]
    assert cuda_model.chunk_gated_delta_rule is cuda_kernels[2]
    assert cuda_model.recurrent_gated_delta_rule is cuda_kernels[3]


@pytest.mark.parametrize("backend", ["tbe", "fwp1"])
def test_tbe_loader_derives_arch_before_first_gate_decode(monkeypatch, backend):
    import transformers

    monkeypatch.delenv("GLC_TBE_MMA_ARCH", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    seen = []
    monkeypatch.setattr(
        torch.cuda, "get_device_capability",
        lambda device=None: seen.append(torch.device(device)) or (12, 0),
    )

    class StopBeforeModelConstruction(Exception):
        pass

    def inspect_arch_before_load(path):
        assert path == "unused"
        assert os.environ["GLC_TBE_MMA_ARCH"] == "12.0"
        raise StopBeforeModelConstruction

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        staticmethod(inspect_arch_before_load),
    )
    bundle = SimpleNamespace(local_dir=Path("unused"))

    with pytest.raises(StopBeforeModelConstruction):
        load_bundle_model(bundle, ServeOptions(device="cuda:0", backend=backend))

    assert seen == [torch.device("cuda:0")]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from glc_serve_tiny import build_checkpoint

    root = tmp_path_factory.mktemp("glc_serve_tiny")
    ck = root / "ckpt"
    build_checkpoint(ck)
    bundle_dir = root / "bundle"
    manifest = pack_from_dense(ck, bundle_dir, min_numel=0, code_lm_head=True,
                               target_shard_bytes=300_000, log=lambda *a: None)
    parent = load_dense_parent(str(ck), ServeOptions(device="cpu"))
    with torch.no_grad():
        ref = parent.model(input_ids=IDS).logits
    return {"root": root, "ckpt": ck, "bundle": bundle_dir, "manifest": manifest,
            "parent": parent, "ref_logits": ref}


def _logits(loaded):
    with torch.no_grad():
        return loaded.model(input_ids=IDS).logits


def test_manifest_shape(tiny):
    m = tiny["manifest"]
    assert m["format"] == B.BUNDLE_FORMAT
    assert len(m["shards"]) > 1, "target_shard_bytes should force several shards"
    for s in m["shards"]:
        assert len(s["sha256"]) == 64
        assert B.sha256_file(tiny["bundle"] / s["file"]) == s["sha256"]
    comps = {t["component"] for t in m["tensors"]}
    assert {"text", "vision", "mtp", "lm_head", "embed"} <= comps
    head = [t for t in m["tensors"] if t["component"] == "lm_head"][0]
    assert head["kind"] == "tbe"
    # a layer group never splits across shards
    by_group = {}
    for t in m["tensors"]:
        by_group.setdefault(t["group"], set()).add(t["shard"])
    assert all(len(v) == 1 for v in by_group.values())
    sidecars = {s["file"] for s in m["sidecars"]}
    assert {"config.json", "tokenizer.json", "preprocessor_config.json",
            "video_preprocessor_config.json", "chat_template.jinja"} <= sidecars


def test_bundle_logits_bitwise_equal_parent(tiny):
    b = B.open_bundle(str(tiny["bundle"]))
    loaded = load_bundle_model(b, ServeOptions(backend="materialize", device="cpu"))
    assert loaded.model.generation_config.eos_token_id == \
        tiny["parent"].model.generation_config.eos_token_id
    assert loaded.mtp is not None
    assert torch.equal(_logits(loaded), tiny["ref_logits"])
    g = run_weight_gate(loaded, scope="full")
    assert g["status"] == "ok" and g["n_mismatched"] == 0
    assert g["n_checked"] == len(tiny["manifest"]["tensors"])
    assert g["all_shards_sha256_verified"]


def test_host_embedding_is_bit_exact(tiny):
    b = B.open_bundle(str(tiny["bundle"]))
    loaded = load_bundle_model(b, ServeOptions(backend="materialize", device="cpu",
                                               embed_on_host=True))
    emb = loaded.model.get_input_embeddings()
    assert type(emb).__name__ == "HostEmbedding"
    assert all(not n.endswith("embed_tokens.weight") for n, _ in loaded.model.named_parameters())
    assert torch.equal(_logits(loaded), tiny["ref_logits"])
    assert run_weight_gate(loaded)["status"] == "ok"


def test_text_only_debug_flag_skips_towers(tiny):
    b = B.open_bundle(str(tiny["bundle"]))
    loaded = load_bundle_model(b, ServeOptions(backend="materialize", device="cpu",
                                               text_only=True))
    assert loaded.mtp is None
    assert loaded.receipt["counts"]["ignored"] > 0
    assert loaded.receipt["text_only"] is True


def test_gate_catches_a_changed_weight(tiny):
    b = B.open_bundle(str(tiny["bundle"]))
    loaded = load_bundle_model(b, ServeOptions(backend="materialize", device="cpu"))
    name = next(t["name"] for t in tiny["manifest"]["tensors"]
                if t["component"] == "vision" and t["kind"] == "tbe")
    mod = loaded.name_to_module[name]
    with torch.no_grad():
        mod.weight.view(-1)[0] = mod.weight.view(-1)[0] + 1
    g = run_weight_gate(loaded, scope="full")
    assert g["status"] == "failed"
    assert [x["name"] for x in g["mismatched"]] == [name]


def test_tampered_shard_is_refused(tiny, tmp_path):
    copy = tmp_path / "bundle"
    shutil.copytree(tiny["bundle"], copy)
    shard = copy / tiny["manifest"]["shards"][1]["file"]
    raw = bytearray(shard.read_bytes())
    raw[-1] ^= 0x01
    shard.write_bytes(bytes(raw))
    b = B.open_bundle(str(copy))
    with pytest.raises(B.BundleError) as exc:
        load_bundle_model(b, ServeOptions(backend="materialize", device="cpu"))
    assert exc.value.reason == "shard_hash_mismatch"


def test_tampered_sidecar_and_pinned_manifest(tiny, tmp_path):
    copy = tmp_path / "bundle"
    shutil.copytree(tiny["bundle"], copy)
    with pytest.raises(B.BundleError) as exc:
        B.open_bundle(str(copy), expected_manifest_sha256="0" * 64)
    assert exc.value.reason == "manifest_hash_mismatch"
    cfg = copy / "config.json"
    cfg.write_text(cfg.read_text() + " ")
    with pytest.raises(B.BundleError) as exc:
        B.open_bundle(str(copy))
    assert exc.value.reason == "shard_hash_mismatch"


def test_manifest_validation_rejects_bad_input(tiny):
    m = json.loads((tiny["bundle"] / B.MANIFEST_FILENAME).read_text())
    bad = dict(m, shards=[dict(m["shards"][0], file="../escape.safetensors")] + m["shards"][1:])
    with pytest.raises(B.BundleError):
        B.validate_manifest(bad)
    t0 = dict(m["tensors"][0])
    t0.pop("blake2b_source")
    with pytest.raises(B.BundleError) as exc:
        B.validate_manifest(dict(m, tensors=[t0] + m["tensors"][1:]))
    assert exc.value.reason == "missing_source_hash"
    with pytest.raises(B.BundleError):
        B.validate_manifest(dict(m, format="something.else"))


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def _serve_dir(directory: Path):
    handler = functools.partial(_Quiet, directory=str(directory))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    return httpd


def test_remote_http_streaming_load(tiny, tmp_path):
    httpd = _serve_dir(tiny["bundle"])
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        cache = tmp_path / "cache"
        b = B.open_bundle(url, cache_dir=cache)
        loaded = load_bundle_model(b, ServeOptions(backend="materialize", device="cpu",
                                                   evict_shards=True))
        assert torch.equal(_logits(loaded), tiny["ref_logits"])
        fetch = loaded.receipt["shard_fetch"]
        assert len(fetch) == len(tiny["manifest"]["shards"])
        assert all(f["source"] == "remote" and f["sha256_ok"] for f in fetch)
        assert not list((cache / B.SHARD_DIR).glob("*.safetensors")), "evict should delete"
    finally:
        httpd.shutdown()


def test_remote_corruption_never_lands_in_cache(tiny, tmp_path):
    served = tmp_path / "served"
    shutil.copytree(tiny["bundle"], served)
    shard = served / tiny["manifest"]["shards"][0]["file"]
    raw = bytearray(shard.read_bytes())
    raw[100] ^= 0xFF
    shard.write_bytes(bytes(raw))
    httpd = _serve_dir(served)
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        cache = tmp_path / "cache"
        b = B.open_bundle(url, cache_dir=cache)
        with pytest.raises(B.BundleError) as exc:
            load_bundle_model(b, ServeOptions(backend="materialize", device="cpu"))
        assert exc.value.reason == "shard_hash_mismatch"
        left = list((cache / B.SHARD_DIR).glob("*"))
        assert not [p for p in left if p.name == Path(tiny["manifest"]["shards"][0]["file"]).name]
        assert not [p for p in left if p.suffix == ".tmp"]
    finally:
        httpd.shutdown()


def test_gcs_url_resolution():
    url, _headers = B.resolve_url("gs://my-bucket/some/prefix", "shards/shard-00001.safetensors")
    assert url == ("https://storage.googleapis.com/download/storage/v1/b/my-bucket/o/"
                   "some%2Fprefix%2Fshards%2Fshard-00001.safetensors?alt=media")


def test_offload_plan_reads_manifest_bytes(tiny):
    b = B.open_bundle(str(tiny["bundle"]))
    full = plan_offload(b, ServeOptions(device="cpu"))
    total = full["projected_resident_bytes_all_gpu"]
    budget = total - 1
    plan = plan_offload(b, ServeOptions(device="cpu", gpu_weight_budget_bytes=budget))
    assert plan["offload_layers"], "a budget below the total must offload something"
    assert plan["projected_resident_bytes"] <= budget
    assert plan["offload_layers"] == sorted(plan["offload_layers"])
    assert max(plan["offload_layers"]) == 3, "the tail of the trunk goes first"
    with pytest.raises(LoadError):
        plan_offload(b, ServeOptions(device="cpu", gpu_weight_budget_bytes=1))


def test_pack_from_real_v1_container(tiny, tmp_path):
    """Re-pack the output of the certified transcoder (v1) with a coded head."""
    sys.path.insert(0, str(REPO / "scripts"))
    tc = pytest.importorskip("glc_tbe_transcode")
    v1 = tmp_path / "v1"
    tc.transcode(tiny["ckpt"], v1, min_numel=0, container_version=1,
                 escape_band_pct=100.0, progress=False)
    out = tmp_path / "bundle"
    m = pack_from_container(v1, out, config_dir=tiny["ckpt"], code_lm_head=True,
                            verify_decode=True, target_shard_bytes=500_000,
                            log=lambda *a: None)
    assert m["pack"]["n_newly_coded"] == 1
    assert m["pack"]["n_decode_verified"] > 0
    b = B.open_bundle(str(out))
    loaded = load_bundle_model(b, ServeOptions(backend="materialize", device="cpu"))
    assert torch.equal(_logits(loaded), tiny["ref_logits"])
    assert run_weight_gate(loaded, scope="full")["status"] == "ok"
