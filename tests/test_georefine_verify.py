"""georefine-verify: the CPU decoder, the byte comparison, and every failure it must catch.

The package lives at release/georefine-verify/ and depends on numpy only.  These
tests build tiny parent checkpoints and TBE bundles on disk (hand-written
safetensors, no library), then run the real Verifier over them.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "release" / "georefine-verify"))

from georefine_verify import cli, export, sources, tbe, verify  # noqa: E402

rng = np.random.default_rng(20260925)


# --------------------------------------------------------------------------- helpers
def bf16_like(n, k, *, special=True):
    """uint16 bf16 patterns shaped like trained weights, plus every awkward class."""
    x = (rng.standard_normal((n, k)) * 0.02).astype(np.float32)
    w = (x.view(np.uint32) >> 16).astype(np.uint16)
    if special:
        flat = w.reshape(-1)
        pats = [0x0000, 0x8000, 0x0001, 0x807F, 0x7F80, 0xFF80, 0x7FC1, 0xFFFF, 0x7F7F, 0x3F80]
        idx = rng.choice(flat.size, size=min(flat.size, 64 * len(pats)), replace=False)
        flat[idx] = np.resize(np.array(pats, np.uint16), idx.size)
    return w


def write_safetensors(path: Path, tensors: dict):
    """{name: (dtype_tag, shape, bytes)} -> a safetensors file (spec: 8-byte LE header len)."""
    hdr, off, blobs = {}, 0, []
    for name, (dt, shape, b) in tensors.items():
        hdr[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_fixture(tmp: Path, *, mutate=None):
    """Parent: 2 shards, 5 tensors.  Bundle: 2 shards, 3 TBE + 2 raw."""
    parent = {
        "a.weight": ("BF16", (16, 128), bf16_like(16, 128)),
        "b.weight": ("BF16", (24, 192), bf16_like(24, 192)),
        "c.weight": ("BF16", (40, 64 * 37), bf16_like(40, 64 * 37)),   # > one decode chunk
        "a.norm": ("BF16", (128,), bf16_like(1, 128).reshape(-1)),
        "d.bias": ("F32", (7,), rng.standard_normal(7).astype(np.float32)),
    }
    pshard = {"a.weight": 1, "a.norm": 1, "d.bias": 1, "b.weight": 2, "c.weight": 2}
    ref = tmp / "parent"
    for s in (1, 2):
        write_safetensors(ref / f"model-0000{s}-of-00002.safetensors",
                          {n: (dt, sh, a.tobytes()) for n, (dt, sh, a) in parent.items()
                           if pshard[n] == s})
    (ref / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": {n: f"model-0000{s}-of-00002.safetensors"
                                        for n, s in pshard.items()}}))
    config = json.dumps({"model_type": "toy", "hidden_size": 128}).encode()
    (ref / "config.json").write_bytes(config)
    entries, shard_tensors = [], {0: {}, 1: {}}
    modes = {"a.weight": (tbe.MODE_W7, "mma16"), "b.weight": (tbe.MODE_W6Z, "mma16"),
             "c.weight": (None, "flat64")}
    for i, (name, (dt, shape, arr)) in enumerate(sorted(parent.items())):
        bs = i % 2
        if name in modes:
            mode, layout = modes[name]
            base = None
            if mode is not None:
                _, base = tbe.choose_window((arr.astype(np.int64) >> 7) & 0xFF)
            p, planes, smb, esc, sbbase = tbe.encode(arr, layout=layout, mode=mode, base=base)
            e = {"name": name, "kind": "tbe", "shape": list(shape), "dtype": "torch.bfloat16",
                 "shard": bs, "mode": p.mode, "base": p.base, "layout": layout,
                 "escapes": int(esc.size), "superblock": 32, "tiles": p.tiles,
                 "coded_shape": list(shape), "original_bytes": arr.nbytes}
            arrays = {"planes": ("I32", planes.shape, planes.tobytes()),
                      "smb": ("U8", smb.shape, smb.tobytes()),
                      "esc": ("U8", esc.shape, esc.tobytes()),
                      "sbbase": ("I32", sbbase.shape, sbbase.tobytes())}
            if mutate:
                mutate(name, e, arrays)
            for k, v in arrays.items():
                shard_tensors[bs][f"{name}.{k}"] = v
        else:
            e = {"name": name, "kind": "raw", "shape": list(shape),
                 "dtype": "torch.bfloat16" if dt == "BF16" else "torch.float32", "shard": bs,
                 "original_bytes": arr.nbytes}
            raw = (dt, shape, arr.tobytes())
            if mutate:
                box = {"raw": raw}
                mutate(name, e, box)
                raw = box["raw"]
            shard_tensors[bs][name] = raw
        entries.append(e)
    bundle = tmp / "bundle"
    shards = []
    for bs in (0, 1):
        f = bundle / "shards" / f"shard-0000{bs}.safetensors"
        write_safetensors(f, shard_tensors[bs])
        shards.append({"file": f"shards/{f.name}", "bytes": f.stat().st_size, "sha256": sha(f)})
    sidecar = {"config.json": config}
    if mutate:
        mutate("__manifest__", entries, shards)
        mutate("__sidecars__", sidecar, None)
    for f, b in sidecar.items():
        (bundle / f).write_bytes(b)
    sidecars = [{"file": f, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
                for f, b in sidecar.items()]
    (bundle / "serve_manifest.json").write_text(json.dumps(
        {"format": "georefine.tbe.serve.v1", "tensors": entries, "shards": shards,
         "sidecars": sidecars}))
    return ref, bundle, parent


def test_export_is_source_free_and_transformers_sharded(tmp_path):
    ref, bundle, parent = make_fixture(tmp_path)
    out = tmp_path / "exported"
    result = export.export(str(bundle), out)
    assert result["tensors"] == 5 and result["shards"] == 2
    assert (out / "config.json").read_bytes() == (ref / "config.json").read_bytes()
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(parent)
    for name, (dtype, shape, array) in parent.items():
        shard = out / index["weight_map"][name]
        src = sources.LocalFile(shard)
        try:
            entry = sources.read_safetensors_header(src)[0][name]
            assert entry.dtype == dtype and entry.shape == shape
            assert src.read(entry.offset, entry.nbytes) == array.tobytes()
        finally:
            src.close()
    # The export has no dependency on the parent; a repeat uses checked output shards.
    assert export.export(str(bundle), out)["tensors"] == 5


def test_exported_shards_open_in_safetensors(tmp_path):
    sf = pytest.importorskip("safetensors")
    torch = pytest.importorskip("torch")
    _, bundle, parent = make_fixture(tmp_path)
    out = tmp_path / "exported"
    export.export(str(bundle), out)
    index = json.loads((out / "model.safetensors.index.json").read_text())
    for name, (dtype, _, array) in parent.items():
        with sf.safe_open(out / index["weight_map"][name], framework="pt") as handle:
            tensor = handle.get_tensor(name)
            actual = tensor.view(torch.int16).numpy().tobytes() if dtype == "BF16" else tensor.numpy().tobytes()
            assert actual == array.tobytes()


def test_export_rejects_corrupt_codec_shard(tmp_path):
    _, bundle, _ = make_fixture(tmp_path)
    shard = bundle / "shards" / "shard-00000.safetensors"
    with shard.open("r+b") as f:
        f.seek(-1, 2)
        byte = f.read(1)
        f.seek(-1, 2)
        f.write(bytes([byte[0] ^ 1]))
    with pytest.raises(sources.SourceError, match="codec shard hash differs"):
        export.export(str(bundle), tmp_path / "exported")


def test_local_file_read_without_pread(tmp_path, monkeypatch):
    path = tmp_path / "bytes.bin"
    path.write_bytes(bytes(range(64)))
    monkeypatch.delattr(sources.os, "pread", raising=False)
    src = sources.LocalFile(path)
    try:
        assert src.read(17, 13) == bytes(range(17, 30))
        assert src.read(0, 4) == bytes(range(4))
    finally:
        src.close()


def test_portable_model_loader_retains_encoded_weights(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(REPO / "release"))
    from glc_loader.tbe_serve_portable import PortableTBELinear, load_compressed_transformers

    _, bundle, parent = make_fixture(tmp_path)

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(128, 16, bias=False)
            self.a.register_parameter("norm", torch.nn.Parameter(torch.empty(128)))
            self.b = torch.nn.Linear(192, 24, bias=False)
            self.c = torch.nn.Linear(64 * 37, 40, bias=False)
            self.d = torch.nn.Module()
            self.d.register_parameter("bias", torch.nn.Parameter(torch.empty(7)))

    fake = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
        AutoModelForMultimodalLM=SimpleNamespace(from_config=lambda config: Toy()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)
    model = load_compressed_transformers(bundle)
    assert model.georefine_tbe_receipt["coded_linears"] == 3
    assert model.georefine_tbe_receipt["base_tensors"] == 5
    assert isinstance(model.a, PortableTBELinear)
    assert "weight" not in dict(model.a.named_parameters())
    assert torch.equal(model.a.norm.view(torch.int16),
                       torch.from_numpy(parent["a.norm"][2].view(np.int16)))
    x = torch.randn(2, 128).to(torch.bfloat16)
    weight = torch.from_numpy(parent["a.weight"][2].view(np.int16)).view(torch.bfloat16)
    actual = model.a(x)
    expected = torch.nn.functional.linear(x, weight)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.fixture
def fake_hf(monkeypatch):
    """Point the HF metadata lookups at the local parent (LFS sha256 = real file sha256)."""
    state = {}

    def info(repo, rev):
        files = {p.name: {"size": p.stat().st_size, "sha256": sha(p)}
                 for p in state["ref"].iterdir() if p.is_file()}
        files.update(state.get("override", {}))
        return {"sha": "f" * 40, "files": files, "license": "apache-2.0"}

    monkeypatch.setattr(verify, "hf_revision_info", info)
    real = sources.Location.read_small

    def read_small(self, rel):
        if self.kind == "hf":
            return (state["ref"] / rel).read_bytes()
        return real(self, rel)

    monkeypatch.setattr(sources.Location, "read_small", read_small)
    return state


def run(tmp, ref, bundle, fake_hf, **kw):
    fake_hf["ref"] = ref
    opt = verify.Options(bundle=str(bundle), reference="Org/Parent@ffff",
                         reference_dir=str(ref), work_dir=str(tmp / "work"), chunk_tiles=32,
                         log_every_s=1e9, **kw)
    return verify.Verifier(opt, log=lambda *a: None).run()


# --------------------------------------------------------------------------- decoder
@pytest.mark.parametrize("layout", ["mma16", "flat64"])
@pytest.mark.parametrize("mode", [tbe.MODE_W7, tbe.MODE_W6Z])
@pytest.mark.parametrize("shape", [(8, 64), (13, 320), (5, 7)])   # (5,7): numel % 64 != 0
def test_decoder_round_trips_every_bit_pattern(layout, mode, shape):
    w = bf16_like(*shape)
    exp = (w.astype(np.int64) >> 7) & 0xFF
    _, base = tbe.choose_window(exp)
    p, planes, smb, esc, sbbase = tbe.encode(w, layout=layout, mode=mode, base=base)
    for chunk in (32, 64, 1 << 16):
        out = np.concatenate([wd for _, wd in tbe.iter_decode(
            p, lambda o, n: planes.tobytes()[o:o + n], lambda o, n: smb.tobytes()[o:o + n],
            lambda o, n: esc.tobytes()[o:o + n], sbbase, esc.size, chunk_tiles=chunk)])
        assert out.dtype == np.uint16 and np.array_equal(out, w.reshape(-1))


def test_all_65536_bf16_patterns_survive():
    w = np.arange(1 << 16, dtype=np.uint16).reshape(256, 256)
    for mode in (tbe.MODE_W7, tbe.MODE_W6Z):
        p, planes, smb, esc, sbbase = tbe.encode(w, mode=mode, base=120)
        out = np.concatenate([wd for _, wd in tbe.iter_decode(
            p, lambda o, n: planes.tobytes()[o:o + n], lambda o, n: smb.tobytes()[o:o + n],
            lambda o, n: esc.tobytes()[o:o + n], sbbase, esc.size, chunk_tiles=64)])
        assert np.array_equal(out, w.reshape(-1))


def test_mma16_permutation_matches_the_kernel_formula():
    for i in range(8):
        for r in range(8):
            assert tbe.MMA16_INV[8 * i + r] == 16 * (r >> 1) + 2 * i + (r & 1)


def test_decoder_reads_containers_written_by_the_production_encoder():
    """Format compatibility with glc_loader.tbe_container (the producer), not just our own encoder."""
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(REPO / "release"))
    from glc_loader import tbe_container as tc

    for shape in ((16, 128), (40, 256)):
        w16 = bf16_like(*shape)
        wt = torch.from_numpy(w16.view(np.int16).copy()).view(torch.bfloat16)
        mode, base, _ = tc.choose_window(tc.exponent_histogram(wt))
        c = tc.encode_tbe(wt, layout="mma16", mode=mode, base=base)
        planes = c.planes.to(torch.int64)
        planes = torch.where(planes >= 2 ** 31, planes - 2 ** 32, planes).to(torch.int32).numpy()
        smb, esc = c.smb.numpy(), c.esc.numpy()
        sbbase = c.sbbase.to(torch.int32).numpy()
        p = tbe.TBEParams(n=shape[0], k=shape[1], layout="mma16", mode=int(c.mode),
                          base=int(c.base), superblock=int(c.superblock))
        out = np.concatenate([wd for _, wd in tbe.iter_decode(
            p, lambda o, n: planes.tobytes()[o:o + n], lambda o, n: smb.tobytes()[o:o + n],
            lambda o, n: esc.tobytes()[o:o + n], sbbase, esc.size, chunk_tiles=32)])
        assert np.array_equal(out, w16.reshape(-1))


# --------------------------------------------------------------------------- end to end
def test_a_faithful_bundle_passes_with_every_digest_checked(tmp_path, fake_hf):
    ref, bundle, parent = make_fixture(tmp_path)
    rec = run(tmp_path, ref, bundle, fake_hf)
    s = rec["summary"]
    assert rec["verdict"] == "PASS", rec["caveats"]
    assert s["bit_identical"] == s["verified"] == s["parent_tensors"] == 5
    assert s["by_kind"]["tbe"] == {"n": 3, "match": 3, "bytes": sum(
        parent[n][2].nbytes for n in ("a.weight", "b.weight", "c.weight"))}
    assert s["parent_shards_checked_against_hf_lfs"] and s["bundle_shards_checked_against_manifest"]
    for t in rec["tensors"]:
        want = hashlib.sha256(parent[t["name"]][2].tobytes()).hexdigest()
        assert t["parent_sha256"] == t["decoded_sha256"] == want
    assert Path(rec["receipt_path"]).is_file()
    assert s["sidecars_identical"] == s["sidecars"] == 1


def test_a_sidecar_that_differs_from_the_parent_fails(tmp_path, fake_hf):
    def m(name, sidecar, _):
        if name == "__sidecars__":
            sidecar["config.json"] = sidecar["config.json"].replace(b"128", b"129")
            sidecar["extra.json"] = b"{}"
    ref, bundle, _ = make_fixture(tmp_path, mutate=m)
    rec = run(tmp_path, ref, bundle, fake_hf)
    assert rec["verdict"] == "FAIL" and rec["summary"]["bit_identical"] == 5
    by = {r["file"]: r for r in rec["sidecars"]}
    assert by["config.json"]["identical"] is False and by["extra.json"]["identical"] is None


def _flip(part, byte=5, bit=0):
    def m(name, e, arrays):
        if name == "b.weight" and part in arrays:
            dt, sh, b = arrays[part]
            b = bytearray(b)
            b[byte] ^= 1 << bit
            arrays[part] = (dt, sh, bytes(b))
    return m


@pytest.mark.parametrize("part", ["smb", "planes", "esc"])
def test_a_single_flipped_bit_in_any_coded_array_fails(tmp_path, fake_hf, part):
    ref, bundle, _ = make_fixture(tmp_path, mutate=_flip(part))
    rec = run(tmp_path, ref, bundle, fake_hf)
    assert rec["verdict"] == "FAIL"
    bad = [t for t in rec["tensors"] if not t["match"]]
    assert [t["name"] for t in bad] == ["b.weight"]
    assert bad[0]["mismatched"] >= 1 or bad[0]["problems"]


def test_a_wrong_decode_parameter_in_our_manifest_cannot_pass(tmp_path, fake_hf):
    def m(name, e, arrays):
        if name == "a.weight":
            e["base"] += 1
    ref, bundle, _ = make_fixture(tmp_path, mutate=m)
    rec = run(tmp_path, ref, bundle, fake_hf)
    assert rec["verdict"] == "FAIL"
    assert not next(t for t in rec["tensors"] if t["name"] == "a.weight")["match"]


def test_a_corrupt_superblock_index_is_reported(tmp_path, fake_hf):
    def m(name, e, arrays):
        if name == "c.weight":
            dt, sh, b = arrays["sbbase"]
            v = np.frombuffer(b, np.int32).copy()
            v[-1] += 1
            arrays["sbbase"] = (dt, sh, v.tobytes())
    ref, bundle, _ = make_fixture(tmp_path, mutate=m)
    rec = run(tmp_path, ref, bundle, fake_hf)
    t = next(t for t in rec["tensors"] if t["name"] == "c.weight")
    assert rec["verdict"] == "FAIL" and "sbbase" in " ".join(t["problems"])


def test_a_changed_raw_tensor_fails(tmp_path, fake_hf):
    def m(name, e, box):
        if name == "d.bias" and "raw" in box:
            dt, sh, b = box["raw"]
            box["raw"] = (dt, sh, b[:-1] + bytes([b[-1] ^ 0x80]))
    ref, bundle, _ = make_fixture(tmp_path, mutate=m)
    rec = run(tmp_path, ref, bundle, fake_hf)
    t = next(t for t in rec["tensors"] if t["name"] == "d.bias")
    assert rec["verdict"] == "FAIL" and t["mismatched"] == 1 and t["first_mismatch"] == 27


def test_coverage_is_counted_from_the_parent_not_from_our_manifest(tmp_path, fake_hf):
    def m(name, entries, shards):
        if name == "__manifest__":
            entries[:] = [e for e in entries if e["name"] != "a.norm"]
    ref, bundle, _ = make_fixture(tmp_path, mutate=m)
    rec = run(tmp_path, ref, bundle, fake_hf)
    assert rec["verdict"] == "FAIL"
    assert rec["coverage"]["missing_from_bundle"] == ["a.norm"]
    assert rec["summary"]["parent_tensors"] == 5


def test_a_parent_shard_that_differs_from_the_published_digest_is_refused(tmp_path, fake_hf):
    ref, bundle, _ = make_fixture(tmp_path)
    fake_hf["override"] = {"model-00002-of-00002.safetensors": {"size": None, "sha256": "0" * 64}}
    fake_hf["ref"] = ref
    opt = verify.Options(bundle=str(bundle), reference="Org/Parent@ffff", reference_dir=str(ref),
                         work_dir=str(tmp_path / "work"), log_every_s=1e9)
    with pytest.raises(sources.SourceError, match="published"):
        verify.Verifier(opt, log=lambda *a: None).run()


def test_a_bundle_shard_that_differs_from_its_manifest_fails(tmp_path, fake_hf):
    def m(name, entries, shards):
        if name == "__manifest__":
            shards[1]["sha256"] = "0" * 64
    ref, bundle, _ = make_fixture(tmp_path, mutate=m)
    rec = run(tmp_path, ref, bundle, fake_hf)
    assert rec["verdict"] == "FAIL"


def test_resume_skips_tensors_already_proven(tmp_path, fake_hf, monkeypatch):
    ref, bundle, _ = make_fixture(tmp_path)
    first = run(tmp_path, ref, bundle, fake_hf, limit=2)
    assert first["verdict"] == "PASS_PARTIAL"
    calls = []
    orig = verify.Verifier.verify_tensor
    monkeypatch.setattr(verify.Verifier, "verify_tensor",
                        lambda self, n: calls.append(n) or orig(self, n))
    rec = run(tmp_path, ref, bundle, fake_hf)
    assert rec["verdict"] == "PASS" and rec["summary"]["bit_identical"] == 5
    assert len(calls) == 3 and not set(calls) & {t["name"] for t in first["tensors"]}


def test_local_parent_without_published_digests_is_only_a_partial_pass(tmp_path):
    ref, bundle, _ = make_fixture(tmp_path)
    opt = verify.Options(bundle=str(bundle), reference=str(ref), work_dir=str(tmp_path / "w"),
                         log_every_s=1e9)
    rec = verify.Verifier(opt, log=lambda *a: None).run()
    assert rec["verdict"] == "PASS_PARTIAL"
    assert rec["summary"]["bit_identical"] == 5
    assert any("not checked against Hugging Face" in c for c in rec["caveats"])


def test_cli_exit_codes(tmp_path, fake_hf, capsys):
    ref, bundle, _ = make_fixture(tmp_path)
    fake_hf["ref"] = ref
    base = ["--bundle", str(bundle), "--reference", "Org/Parent@ffff", "--reference-dir", str(ref)]
    assert cli.main(base + ["--work-dir", str(tmp_path / "w1")]) == 0
    assert "PASS: 5/5 tensors bit-identical" in capsys.readouterr().out
    assert cli.main(base + ["--work-dir", str(tmp_path / "w2"), "--limit", "1"]) == 3
    assert cli.main(["--bundle", str(bundle), "--reference", "Org/Parent"]) == 2


def test_safetensors_header_rejects_a_lying_size(tmp_path):
    f = tmp_path / "x.safetensors"
    h = json.dumps({"t": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 6]}}).encode()
    f.write_bytes(struct.pack("<Q", len(h)) + h + b"\0" * 6)
    with pytest.raises(sources.SourceError):
        sources.read_safetensors_header(sources.LocalFile(f))


def test_the_package_imports_nothing_but_numpy_and_the_standard_library():
    import ast

    allowed = set(sys.stdlib_module_names) | {"numpy", "georefine_verify", "__future__"}
    pkg = REPO / "release" / "georefine-verify" / "georefine_verify"
    for py in pkg.glob("*.py"):
        for node in ast.walk(ast.parse(py.read_text())):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                mods = [node.module]
            else:
                continue
            for m in mods:
                assert m.split(".")[0] in allowed, f"{py.name} imports {m}"


@pytest.fixture
def http_bundle(tmp_path):
    """Serve a directory over HTTP with Range support (http.server has none)."""
    import functools
    import http.server
    import threading

    class H(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            path = Path(self.translate_path(self.path))
            if not path.is_file():
                self.send_error(404)
                return
            data = path.read_bytes()
            rng_h = self.headers.get("Range")
            if rng_h:
                a, _, b = rng_h.split("=")[1].partition("-")
                a, b = int(a), (int(b) if b else len(data) - 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
                body = data[a:b + 1]
            else:
                self.send_response(200)
                body = data
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    servers = []

    def serve(directory):
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                              functools.partial(H, directory=str(directory)))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    yield serve
    for s in servers:
        s.shutdown()


@pytest.mark.parametrize("bundle_mode", ["download", "range"])
def test_a_remote_bundle_is_downloaded_and_checked_or_range_read(tmp_path, fake_hf, http_bundle,
                                                                  bundle_mode):
    ref, bundle, _ = make_fixture(tmp_path)
    url = http_bundle(bundle)
    rec = run(tmp_path, ref, url, fake_hf, bundle_mode=bundle_mode)
    assert rec["summary"]["bit_identical"] == 5
    assert rec["bundle"]["location"] == url
    if bundle_mode == "download":
        assert rec["verdict"] == "PASS"
        assert all(s["checked"] and s["ok"] for s in rec["bundle"]["shards"])
    else:
        assert rec["verdict"] == "PASS_PARTIAL"
        assert any("bundle shard" in c for c in rec["caveats"])


def test_download_resumes_a_partial_file_and_checks_its_digest(tmp_path, http_bundle):
    src = tmp_path / "srv" / "blob.bin"
    src.parent.mkdir()
    payload = rng.integers(0, 256, 3 << 20, dtype=np.uint8).tobytes()
    src.write_bytes(payload)
    url = http_bundle(src.parent) + "/blob.bin"
    dest = tmp_path / "dl" / "blob.bin"
    dest.parent.mkdir()
    dest.with_name("blob.bin.part").write_bytes(payload[:1000])
    got = sources.download(url, dest, size=len(payload),
                           sha256=hashlib.sha256(payload).hexdigest(), log=lambda *a: None)
    assert got == hashlib.sha256(payload).hexdigest() and dest.read_bytes() == payload
    with pytest.raises(sources.SourceError, match="sha256"):
        sources.download(url, tmp_path / "dl" / "again.bin", size=len(payload), sha256="0" * 64,
                         log=lambda *a: None)
