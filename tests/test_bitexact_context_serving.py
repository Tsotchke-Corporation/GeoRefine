import hashlib
import io
import json
import tarfile

import numpy as np
import pytest
import torch
from torch import nn

from scripts.bitexact_context_serving import ContextServingError, load_bctx_model, prepare_metadata


class TinyModel(nn.Module):
    def __init__(self, config=None, attention_implementation=None, *, bad_buffer=False):
        super().__init__()
        self.emb = nn.Embedding(8, 4)
        self.lin = nn.Linear(4, 3)
        self.norm = nn.LayerNorm(4)
        self.fp32_exception = nn.Parameter(torch.empty(2, dtype=torch.float32))
        self.config = config
        if isinstance(config, dict) and config.get("strict_fp32_exception"):
            self._keep_in_fp32_modules_strict = {"fp32_exception"}
        self.attention_implementation = attention_implementation
        if bad_buffer:
            self.register_buffer("stuck", torch.empty(1, device="meta"), persistent=False)

    def forward(self, ids):
        return self.lin(self.norm(self.emb(ids)))


class FakeContext:
    def __init__(self, frame, *, stride, device, values, metadata):
        del stride
        self.values = values[frame]
        entry = metadata[frame]
        self.shape = tuple(entry["frame_shape"])
        self.source_sha256 = entry["source_sha256"]
        self.frame_sha256 = hashlib.sha256(frame).hexdigest()
        self.resident_bytes = 37
        self.device = torch.device(device)

    def decode(self):
        return self.values.to(self.device).reshape(self.shape).clone()

    def gather_rows(self, ids):
        return self.decode()[ids.to(self.device)]


def _write_package(tmp_path, *, missing=(), bad_sha=None, bad_source_sha=None,
                   bad_archive=False, bad_asset_pin=False, physical_sidecars=False):
    package = tmp_path / "package"
    package.mkdir()
    (package / "frames").mkdir()
    sidecars = {
        "sidecars/config.json": b'{"eos_token_id":248044,"pad_token_id":null}',
        "sidecars/generation_config.json": b'{"eos_token_id":[248046,248044],"pad_token_id":248044,"do_sample":true,"top_k":20,"top_p":0.95}',
        "sidecars/model.safetensors.index.json": b'{"weight_map":{}}',
    }
    for i in range(10):
        sidecars[f"sidecars/asset-{i:02d}.json"] = f'{{"asset":{i}}}'.encode()
    metadata_assets = {
        name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in sidecars.items()
    }
    if bad_asset_pin:
        metadata_assets["sidecars/config.json"]["sha256"] = "0" * 64
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:xz") as tf:
        for name, payload in sidecars.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    archive = archive_buffer.getvalue()
    archive_pin_sha = hashlib.sha256(archive).hexdigest()
    if bad_archive:
        archive = archive[:12] + bytes([archive[12] ^ 1]) + archive[13:]
    (package / "model-assets.tar.xz").write_bytes(archive)
    if physical_sidecars:
        for name, payload in sidecars.items():
            target = package / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    tensors = {
        "emb.weight": torch.arange(32, dtype=torch.float32).reshape(8, 4).to(torch.bfloat16),
        "lin.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4).to(torch.bfloat16),
        "lin.bias": torch.tensor([0.25, -0.5, 1.0], dtype=torch.bfloat16),
        "norm.weight": torch.tensor([1.0, 0.75, 1.25, 0.5], dtype=torch.bfloat16),
        "norm.bias": torch.zeros(4, dtype=torch.bfloat16),
        "fp32_exception": torch.tensor([2.0, -3.0], dtype=torch.bfloat16),
        "mtp.layers.0.weight": torch.ones(2, 2, dtype=torch.bfloat16),
    }
    values, metadata, rows = {}, {}, []
    for i, (name, tensor) in enumerate(tensors.items()):
        if name in missing:
            continue
        frame = f"fake-frame-{i}".encode()
        frame_sha = hashlib.sha256(frame).hexdigest()
        source_sha = hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        frame_shape = (tensor.shape[0], tensor.numel() // tensor.shape[0]) if tensor.ndim > 1 else (1, tensor.numel())
        values[frame] = tensor
        metadata[frame] = {"frame_shape": frame_shape, "source_sha256": source_sha}
        if name == bad_sha:
            frame_sha = "0" * 64
        if name == bad_source_sha:
            source_sha = "0" * 64
        rel = f"frames/{i}.bctx"
        (package / rel).write_bytes(frame)
        rows.append({"name": name, "dtype": "BF16", "shape": list(tensor.shape),
                     "source_bytes": tensor.numel() * 2,
                     "frame": rel, "frame_bytes": len(frame), "frame_sha256": frame_sha,
                     "source_sha256": source_sha, "complete": True})
    (package / "manifest.json").write_text(json.dumps({
        "schema": "bitexact-context-package-v1", "complete": True, "tensors": rows,
        "metadata_assets": metadata_assets,
        "source": {"index_sha256": hashlib.sha256(sidecars["sidecars/model.safetensors.index.json"]).hexdigest()},
        "files": [{"path": "model-assets.tar.xz", "bytes": len(archive),
                   "sha256": archive_pin_sha}],
    }))
    return package, values, metadata, tensors


def _load(package, tmp_path, values, metadata, *, loader_workers=1, progress_callback=None,
          config_options=None, **kwargs):
    return load_bctx_model(
        package,
        device="cpu",
        tensor_factory=lambda frame, **opts: FakeContext(frame, values=values, metadata=metadata, **opts),
        model_factory=lambda config, attention: TinyModel(config, attention, **kwargs),
        config_loader=lambda path: {"path": path, "eos_token_id": 248044,
                                    "pad_token_id": None, "text_config": {},
                                    "vision_config": {}, **(config_options or {})},
        empty_weights_factory=lambda: torch.device("meta"),
        host_decoder=lambda frame: values[frame].view(torch.uint16).numpy().reshape(metadata[frame]["frame_shape"]),
        metadata_cache_dir=tmp_path / "metadata-cache",
        loader_workers=loader_workers,
        progress_callback=progress_callback,
    )


def test_loader_retains_compressed_linear_and_embedding_and_materializes_dense_params(tmp_path):
    package, values, metadata, expected = _write_package(tmp_path)
    model, receipt = _load(package, tmp_path, values, metadata)

    assert isinstance(model.emb, __import__("scripts.bitexact_context_serving", fromlist=["BctxEmbedding"]).BctxEmbedding)
    assert isinstance(model.lin, __import__("scripts.bitexact_context_serving", fromlist=["BctxLinear"]).BctxLinear)
    assert model.emb.context_weight.resident_bytes == 37
    assert model.lin.context_weight.resident_bytes == 37
    assert model.norm.weight.dtype == torch.bfloat16
    assert model.fp32_exception.dtype == torch.bfloat16
    torch.testing.assert_close(model.emb(torch.tensor([2])), expected["emb.weight"][2:3])
    torch.testing.assert_close(model(torch.tensor([2])),
                               model.lin(model.norm(expected["emb.weight"][2:3])))
    assert receipt["unused_package_tensors"] == ["mtp.layers.0.weight"]
    assert receipt["compressed_tensor_count"] == 2
    assert receipt["resident_context_bytes"] == 74
    assert receipt["restored_bf16_control"] is False
    assert receipt["transcoded"] is False
    metadata_dir = tmp_path / "metadata-cache" / hashlib.sha256(
        (package / "model-assets.tar.xz").read_bytes()).hexdigest() / "sidecars"
    assert receipt["metadata_dir"] == str(metadata_dir)
    assert json.loads((metadata_dir / "config.json").read_text())["eos_token_id"] == 248044
    assert len(list(metadata_dir.iterdir())) == 13


def test_direct_embedding_weight_access_is_transient_and_not_registered(tmp_path):
    package, values, metadata, expected = _write_package(tmp_path)
    model, _ = _load(package, tmp_path, values, metadata)
    first = model.emb.weight
    second = model.emb.weight
    torch.testing.assert_close(first, expected["emb.weight"])
    assert first.data_ptr() != second.data_ptr()
    assert model.emb.weight_materializations == 2
    assert model.emb.forward_calls == 0
    assert "weight" not in dict(model.emb.named_parameters())
    assert "emb.weight" not in model.state_dict()


def test_loader_propagates_attention_and_parallel_progress_callback(tmp_path):
    package, values, metadata, _ = _write_package(tmp_path)
    events = []
    model, _ = _load(package, tmp_path, values, metadata, loader_workers=4,
                     progress_callback=lambda completed, total, name: events.append((completed, total, name)))
    assert model.config["_attn_implementation"] == "eager"
    assert model.config["text_config"]["_attn_implementation"] == "eager"
    assert model.config["vision_config"]["_attn_implementation"] == "eager"
    assert len(events) == 1 and events[0][:2] == (2, 2)


def test_loader_uses_verified_generation_config_sidecar_not_config_eos(tmp_path):
    package, values, metadata, _ = _write_package(tmp_path)
    model, receipt = _load(package, tmp_path, values, metadata)
    assert model.config["eos_token_id"] == 248044
    assert model.generation_config.eos_token_id == [248046, 248044]
    assert model.generation_config.pad_token_id == 248044
    assert model.generation_config.do_sample is True
    assert model.generation_config.top_k == 20
    assert receipt["generation_config_path"].endswith("/sidecars/generation_config.json")
    manifest = json.loads((package / "manifest.json").read_text())
    assert receipt["generation_config_sha256"] == manifest["metadata_assets"]["sidecars/generation_config.json"]["sha256"]
    assert receipt["generation_config"]["eos_token_id"] == [248046, 248044]


def test_loader_preserves_explicit_transformers_fp32_parameter_exception(tmp_path):
    package, values, metadata, _ = _write_package(tmp_path)
    model, _ = _load(package, tmp_path, values, metadata,
                     config_options={"strict_fp32_exception": True})
    assert model.fp32_exception.dtype == torch.float32


def test_loader_fails_closed_when_required_parameter_is_missing(tmp_path):
    package, values, metadata, _ = _write_package(tmp_path, missing={"norm.weight"})
    with pytest.raises(ContextServingError, match="missing required model parameters"):
        _load(package, tmp_path, values, metadata)


def test_loader_checks_frame_digest_before_backend_construction(tmp_path):
    package, values, metadata, _ = _write_package(tmp_path, bad_sha="emb.weight")
    with pytest.raises(ContextServingError, match="frame identity mismatch"):
        _load(package, tmp_path, values, metadata)


def test_loader_checks_host_decoded_source_digest_before_dtype_cast(tmp_path):
    package, values, metadata, _ = _write_package(tmp_path, bad_source_sha="norm.weight")
    with pytest.raises(ContextServingError, match="decoded-source identity mismatch: norm.weight"):
        _load(package, tmp_path, values, metadata)


def test_loader_rejects_remaining_meta_buffers(tmp_path):
    package, values, metadata, _ = _write_package(tmp_path)
    with pytest.raises(ContextServingError, match="unfilled meta tensors"):
        _load(package, tmp_path, values, metadata, bad_buffer=True)


def test_loader_rejects_metadata_archive_hash_mismatch(tmp_path):
    package, values, metadata, _ = _write_package(tmp_path, bad_archive=True)
    with pytest.raises(ContextServingError, match="metadata archive size/SHA mismatch"):
        _load(package, tmp_path, values, metadata)


def test_loader_rejects_metadata_member_hash_mismatch(tmp_path):
    package, values, metadata, _ = _write_package(tmp_path, bad_asset_pin=True)
    with pytest.raises(ContextServingError, match="metadata archive asset integrity mismatch: sidecars/config.json"):
        _load(package, tmp_path, values, metadata)


def test_loader_accepts_physical_sidecars_only_when_they_match_manifest(tmp_path):
    package, values, metadata, _ = _write_package(tmp_path, physical_sidecars=True)
    (package / "sidecars" / "rogue.json").write_text("{}")
    with pytest.raises(ContextServingError, match="physical sidecars differ"):
        _load(package, tmp_path, values, metadata)


def test_matching_physical_sidecars_are_accepted_after_archive_verification(tmp_path):
    package, values, metadata, _ = _write_package(tmp_path, physical_sidecars=True)
    model, receipt = _load(package, tmp_path, values, metadata)
    assert receipt["metadata_dir"] == str(package / "sidecars")
    assert model is not None


def test_metadata_cache_must_be_outside_immutable_package(tmp_path):
    package, _, _, _ = _write_package(tmp_path)
    with pytest.raises(ContextServingError, match="outside the immutable package"):
        prepare_metadata(package, package / ".cache")


@pytest.mark.parametrize("steps", [1, 3])
def test_strided_batched_linear_matches_parameter_matmul_dispatch(steps):
    from scripts.bitexact_context_serving import BctxLinear
    from torch._decomp.decompositions import should_fold
    torch.manual_seed(43)
    reference = torch.nn.Linear(7, 11, bias=False, dtype=torch.bfloat16)
    class Descriptor:
        shape = (11, 7)
        def decode(self):
            return reference.weight.detach().clone()
    adapter = BctxLinear(Descriptor())
    x = torch.randn(4, 9, 7, dtype=torch.bfloat16)[:, -steps:, :]
    assert not x.is_contiguous()
    with torch.inference_mode():
        assert should_fold(x, reference.weight.t(), False)
        assert not should_fold(x, reference.weight.detach().t(), False)
        expected = reference(x)
        actual = adapter(x)
    assert torch.equal(actual, expected)
    assert not list(adapter.parameters())
