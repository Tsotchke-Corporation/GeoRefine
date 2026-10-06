import hashlib
import importlib.util
import io
import json
import struct
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bitexact_context_package", ROOT / "scripts" / "bitexact_context_package.py"
)
pkg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pkg)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_fixture(tmp_path, shape=(8, 12)):
    model = tmp_path / "source"
    model.mkdir()
    name = "model.language_model.layers.0.weight"
    words = np.arange(int(np.prod(shape)), dtype="<u2").reshape(shape)
    raw = words.tobytes()
    header = {name: {"dtype": "BF16", "shape": list(shape), "data_offsets": [0, len(raw)]}}
    header_raw = json.dumps(header, separators=(",", ":")).encode()
    shard = "model-00001-of-00018.safetensors"
    shard_raw = struct.pack("<Q", len(header_raw)) + header_raw + raw
    (model / shard).write_bytes(shard_raw)
    index = {"metadata": {"total_size": len(raw)}, "weight_map": {name: shard}}
    (model / "model.safetensors.index.json").write_text(json.dumps(index))
    census = {
        "tensors": [
            {"name": name, "dtype": "BF16", "shape": list(shape), "raw_bytes": len(raw), "raw_sha256": digest(raw)}
        ]
    }
    census_path = tmp_path / "census.json"
    census_path.write_text(json.dumps(census))
    archive = tmp_path / "assets.tar.xz"
    sidecars = [("sidecars/model.safetensors.index.json", (model / "model.safetensors.index.json").read_bytes())]
    sidecars.extend((f"sidecars/asset-{i}.txt", f"asset-{i}".encode()) for i in range(12))
    with tarfile.open(archive, "w:xz") as tf:
        for name, payload in sidecars:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    return model, census_path, archive, shard, shard_raw


def test_build_verify_restore_exact_source_and_assets(tmp_path):
    model, census, archive, shard, shard_raw = source_fixture(tmp_path)
    package = tmp_path / "package"
    result = pkg.build(model, package, census, archive, jobs=2, strict=False)
    assert result["source"]["tensor_count"] == 1
    assert result["package_bytes"] == sum(x["bytes"] for x in result["files"])
    assert not any("libbitexact_context_ans" in x["path"] or "__pycache__" in x["path"] for x in result["files"])
    verified = pkg.verify(package)
    assert verified["source"]["census_sha256"]
    package_inventory = pkg.inventory(package)
    pkg.verify(package)
    assert pkg.inventory(package) == package_inventory  # decode cache stays outside immutable package
    (package / "manifest.json").unlink()
    resumed = pkg.build(model, package, census, archive, jobs=1, strict=False)
    assert resumed["complete"] is True
    pkg.verify(package)
    restored = tmp_path / "restored"
    model.rename(tmp_path / "original_source_unavailable")
    standalone = package / "decoder" / "bitexact_context_package.py"
    check = subprocess.run(
        [sys.executable, str(standalone), "verify", str(package)], check=True, capture_output=True, text=True
    )
    assert json.loads(check.stdout)["verified"] is True
    restore_run = subprocess.run(
        [sys.executable, str(standalone), "restore", str(package), str(restored)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(restore_run.stdout) == {"tensors": 1, "shards": 1, "assets": 13}
    assert (restored / shard).read_bytes() == shard_raw
    assert len(list(restored.iterdir())) == 14
    assert (restored / "model.safetensors.index.json").is_file()
    assert not (restored / "sidecars").exists()


def test_rank3_restores_original_shape_and_shard(tmp_path):
    model, census, archive, shard, shard_raw = source_fixture(tmp_path, shape=(2, 3, 4))
    package = tmp_path / "package"
    pkg.build(model, package, census, archive, strict=False)
    row = pkg.verify(package)["tensors"][0]
    assert row["shape"] == [2, 3, 4]
    restored = tmp_path / "restored"
    pkg.restore(package, restored)
    assert (restored / shard).read_bytes() == shard_raw


def test_reuse_inventory_skips_all_six_encodes(tmp_path, monkeypatch):
    model, census, archive, _, _ = source_fixture(tmp_path)
    original = tmp_path / "original"
    pkg.build(model, original, census, archive, strict=False)
    row = pkg.verify(original)["tensors"][0]
    reuse_root = tmp_path / "reuse"
    reuse_root.mkdir()
    reuse_frame = reuse_root / "best.bctx"
    reuse_frame.write_bytes((original / row["frame"]).read_bytes())
    normalized = {
        key: row[key]
        for key in (
            "name",
            "source_bytes",
            "source_sha256",
            "source_shard",
            "dtype",
            "shape",
            "best_config",
            "frame_bytes",
            "frame_sha256",
            "candidates",
        )
    }
    normalized["frame"] = "best.bctx"
    inventory = tmp_path / "reuse.json"
    inventory.write_text(json.dumps({"frames": [normalized]}))
    original_loader = pkg.decode_codec

    def decode_without_encode(package, cache_dir=None):
        codec = original_loader(package, cache_dir)
        codec.encode = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reused tensor was re-encoded"))
        return codec

    monkeypatch.setattr(pkg, "decode_codec", decode_without_encode)
    reused_out = tmp_path / "reused-package"
    result = pkg.build(
        model, reused_out, census, archive, strict=False, reuse_frames=reuse_root, reuse_inventory=inventory
    )
    assert result["tensors"][0]["reused"] is True


def test_verify_fails_closed_on_extra_file(tmp_path):
    model, census, archive, _, _ = source_fixture(tmp_path)
    package = tmp_path / "package"
    pkg.build(model, package, census, archive, strict=False)
    (package / "unexpected").write_text("x")
    with pytest.raises(ValueError, match="inventory"):
        pkg.verify(package)


def test_verify_fails_closed_on_missing_frame(tmp_path):
    model, census, archive, _, _ = source_fixture(tmp_path)
    package = tmp_path / "package"
    pkg.build(model, package, census, archive, strict=False)
    row = pkg.verify(package)["tensors"][0]
    (package / row["frame"]).unlink()
    with pytest.raises(ValueError, match="inventory"):
        pkg.verify(package)


def test_census_mismatch_rejected(tmp_path):
    model, census, archive, _, _ = source_fixture(tmp_path)
    doc = json.loads(census.read_text())
    doc["tensors"][0]["raw_sha256"] = "0" * 64
    census.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="census"):
        pkg.build(model, tmp_path / "package", census, archive, strict=False)


def test_corrupt_frame_and_changed_resume_receipt_fail_closed(tmp_path):
    model, census, archive, _, _ = source_fixture(tmp_path)
    package = tmp_path / "package"
    pkg.build(model, package, census, archive, strict=False)
    row = pkg.verify(package)["tensors"][0]
    (package / "manifest.json").unlink()
    receipt_path = package / "receipts" / (pkg.key_for(row["name"]) + ".json")
    receipt = json.loads(receipt_path.read_text())
    receipt["candidates"].pop()
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="corrupt|incomplete"):
        pkg.build(model, package, census, archive, strict=False)


def test_metadata_asset_corruption_fails_closed(tmp_path):
    model, census, archive, _, _ = source_fixture(tmp_path)
    package = tmp_path / "package"
    pkg.build(model, package, census, archive, strict=False)
    with (package / "model-assets.tar.xz").open("r+b") as f:
        f.seek(0)
        f.write(b"bad")
    with pytest.raises((ValueError, OSError, tarfile.TarError)):
        pkg.verify(package)


def test_metadata_archive_path_traversal_rejected(tmp_path):
    model, census, archive, _, _ = source_fixture(tmp_path)
    unsafe = tmp_path / "unsafe.tar.xz"
    with tarfile.open(archive, "r:xz") as src, tarfile.open(unsafe, "w:xz") as dst:
        members = src.getmembers()
        for i, member in enumerate(members):
            raw = src.extractfile(member).read()
            if i == 0:
                member.name = "sidecars/../escape"
            dst.addfile(member, io.BytesIO(raw))
    with pytest.raises(ValueError, match="unsafe"):
        pkg.build(model, tmp_path / "package", census, unsafe, strict=False)


def test_package_cannot_depend_on_external_symlink(tmp_path):
    model, census, archive, _, _ = source_fixture(tmp_path)
    package = tmp_path / "package"
    doc = pkg.build(model, package, census, archive, strict=False)
    path = package / doc["tensors"][0]["frame"]
    external = tmp_path / "external_frame"
    path.rename(external)
    path.symlink_to(external)
    with pytest.raises(ValueError, match="symlink"):
        pkg.verify(package)


def test_only_manifest_may_omit_digest(tmp_path):
    model, census, archive, _, _ = source_fixture(tmp_path)
    package = tmp_path / "package"
    doc = pkg.build(model, package, census, archive, strict=False)
    next(row for row in doc["files"] if row["path"].startswith("frames/"))["sha256"] = None
    (package / "manifest.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="omit"):
        pkg.verify(package)


def test_decoder_cache_cannot_modify_package(tmp_path):
    model, census, archive, _, _ = source_fixture(tmp_path)
    package = tmp_path / "package"
    pkg.build(model, package, census, archive, strict=False)
    before = pkg.inventory(package)
    with pytest.raises(ValueError, match="outside"):
        pkg.verify(package, cache_dir=package / "cache")
    assert pkg.inventory(package) == before
