#!/usr/bin/env python3
"""Build, verify, and restore a complete byte-exact BF16 Qwen38 package."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
import struct
import tarfile
import threading

import numpy as np

INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
SOURCE_BYTES = 55_562_855_904
TENSOR_COUNT = 1199
SHARD_COUNT = 18
CONFIGS = ((1, 1), (16, 1), (16, 4), (16, 16), (64, 4), (64, 16))
SHARD_NAMES = tuple(f"model-{i:05d}-of-00018.safetensors" for i in range(1, 19))
SHARD_SHA256 = {
    "model-00001-of-00018.safetensors": "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c",
    "model-00002-of-00018.safetensors": "06a148c01bfbe3faa14a5f184a7ff29a706f7ae1c8b2705d2058e26d17a001fb",
    "model-00003-of-00018.safetensors": "2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968",
    "model-00004-of-00018.safetensors": "511e34063187882659753c4d93f3859f93c019fd438d8813071921c81d9a3f1a",
    "model-00005-of-00018.safetensors": "635cb53446dc74f219740fc59e18b774f877b803b9722e289ca62575a6efa701",
    "model-00006-of-00018.safetensors": "0bc5214fac607f0e6cc92eec3789d4b8559410ef9fce66621ba8158e8410dae0",
    "model-00007-of-00018.safetensors": "80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892",
    "model-00008-of-00018.safetensors": "7192c5b66185d3592927daabee1cc19e6f6e0ce75988ee20e824b624765fda79",
    "model-00009-of-00018.safetensors": "af3c48cc37af44f3db6ae0579baf019180d48d9c527caa0a1f03ff85813a56d8",
    "model-00010-of-00018.safetensors": "163490a76f3bea3a40855b7efc04ce6d27afaf1a34f0bbde495b9491f76457c9",
    "model-00011-of-00018.safetensors": "5f3ae1b948aeee39da77aec558e8236cd65fe4d7cb7686a76bb007acc563c6d8",
    "model-00012-of-00018.safetensors": "a3de1c7114677a8f5ac5c4892c90e8238ea5c1e2038c80e757dfc87c3902ca55",
    "model-00013-of-00018.safetensors": "06ab79a41f74c9c5cb734816feb0c7fc364104b227165ee7391231e1155aa02a",
    "model-00014-of-00018.safetensors": "4138ed94603065ba884bbcadedb04d7718bb40117e85e6f5c6fc5b9c05b7a85b",
    "model-00015-of-00018.safetensors": "69224e27b9de4e7dbf6fc936c6eaae08447bda3b80a6c31a871ab451173afd22",
    "model-00016-of-00018.safetensors": "73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe",
    "model-00017-of-00018.safetensors": "beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8",
    "model-00018-of-00018.safetensors": "1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path: Path, value) -> None:
    atomic_write(path, json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n")


def key_for(name: str) -> str:
    return sha(name.encode())


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_census(path: Path):
    raw = path.read_bytes()
    doc = json.loads(raw, object_pairs_hook=unique_object)
    rows = doc.get("tensors")
    if not isinstance(rows, list):
        raise ValueError("census must contain a tensor list")
    out = {}
    for row in rows:
        name = row["name"]
        if name in out:
            raise ValueError("duplicate census tensor")
        out[name] = {
            "shape": row["shape"],
            "dtype": row["dtype"],
            "source_bytes": row.get("raw_bytes", row.get("src_bytes", row.get("source_bytes"))),
            "sha256": row.get("raw_sha256", row.get("sha256", row.get("source_sha256"))),
        }
        if not out[name]["sha256"] or out[name]["source_bytes"] is None:
            raise ValueError(f"incomplete census entry: {name}")
    census_sha = sha(raw)
    canonical = [{"name": n, **out[n]} for n in sorted(out)]
    return out, census_sha, sha(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode())


def tensor_descriptors(model: Path, index: dict):
    descriptors = {}
    headers = {}
    for shard in sorted(set(index.values())):
        if Path(shard).name != shard or shard not in SHARD_NAMES:
            raise ValueError("unexpected source shard path")
        path = model / shard
        with path.open("rb") as f:
            prefix = f.read(8)
            if len(prefix) != 8:
                raise ValueError("truncated safetensors prefix")
            n = struct.unpack("<Q", prefix)[0]
            if not 0 < n <= 100_000_000:
                raise ValueError("invalid safetensors header size")
            raw_header = f.read(n)
            header = json.loads(raw_header, object_pairs_hook=unique_object)
            headers[shard] = prefix + raw_header
            data_start = 8 + n
            file_size = path.stat().st_size
            intervals = []
            for name, desc in header.items():
                if name == "__metadata__":
                    continue
                if index.get(name) != shard or desc.get("dtype") != "BF16":
                    raise ValueError(f"unexpected indexed tensor or dtype: {name}")
                shape = desc["shape"]
                a, b = desc["data_offsets"]
                if not isinstance(shape, list) or not shape or any(type(d) is not int or d <= 0 for d in shape):
                    raise ValueError(f"invalid tensor shape: {name}")
                if type(a) is not int or type(b) is not int or a < 0 or b < a:
                    raise ValueError(f"invalid tensor offsets: {name}")
                expect = math.prod(shape) * 2
                if b - a != expect or data_start + b > file_size:
                    raise ValueError(f"invalid source tensor range: {name}")
                intervals.append((a, b, name))
                descriptors[name] = {
                    "name": name,
                    "source_shard": shard,
                    "dtype": "BF16",
                    "shape": shape,
                    "offsets": [a, b],
                    "source_bytes": expect,
                }
            intervals.sort()
            if intervals and intervals[0][0] != 0:
                raise ValueError(f"{shard}: leading unreferenced bytes")
            if any(left[1] != right[0] for left, right in zip(intervals, intervals[1:])):
                raise ValueError(f"{shard}: overlapping or gapped tensor ranges")
            if intervals and intervals[-1][1] != file_size - data_start:
                raise ValueError(f"{shard}: trailing unreferenced bytes")
    if set(descriptors) != set(index):
        raise ValueError("source headers and index differ")
    return descriptors, headers


def read_raw(model: Path, desc: dict) -> bytes:
    with (model / desc["source_shard"]).open("rb") as f:
        header_n = struct.unpack("<Q", f.read(8))[0]
        f.seek(8 + header_n + desc["offsets"][0])
        raw = f.read(desc["source_bytes"])
    if len(raw) != desc["source_bytes"]:
        raise ValueError("short source tensor")
    return raw


def decode_codec(package: Path, cache_dir: Path | None = None):
    import types

    path = package / "decoder" / "bitexact_context_codec.py"
    source_hash = sha((sha_file(path) + sha_file(path.with_name("bitexact_context_ans.cpp"))).encode())
    cache_root = cache_dir or (package.parent / ".scratch" / "bitexact-context-codec")
    if cache_root.resolve().is_relative_to(package.resolve()):
        raise ValueError("decoder cache must be outside the immutable package")
    cache = cache_root / (source_hash + "-" + os.uname().sysname.lower() + "-" + os.uname().machine.lower())
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["BITEXACT_CONTEXT_CACHE_DIR"] = str(cache.resolve())
    import sys

    module = types.ModuleType("package_bitexact_codec")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def validate_candidates(row: dict) -> dict:
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(CONFIGS):
        raise ValueError("receipt must carry all six frozen candidates")
    by_config = {}
    for item in candidates:
        config = (item.get("row_classes"), item.get("col_classes"))
        if config not in CONFIGS or config in by_config:
            raise ValueError("candidate configurations are incomplete or duplicated")
        if item.get("exact") is not True or item.get("decoded_sha256") != row.get("source_sha256"):
            raise ValueError("candidate lacks exact decoded-source evidence")
        if type(item.get("frame_bytes")) is not int or item["frame_bytes"] <= 0:
            raise ValueError("invalid candidate frame size")
        if not re.fullmatch(r"[0-9a-f]{64}", item.get("frame_sha256", "")):
            raise ValueError("invalid candidate frame SHA")
        by_config[config] = item
    if set(by_config) != set(CONFIGS):
        raise ValueError("receipt does not cover the frozen configurations")
    best = min(candidates, key=lambda x: x["frame_bytes"])
    if (
        row.get("best_config") != [best["row_classes"], best["col_classes"]]
        or row.get("frame_bytes") != best["frame_bytes"]
        or row.get("frame_sha256") != best["frame_sha256"]
    ):
        raise ValueError("receipt frame is not the minimum of its six candidates")
    return best


def load_reuse_inventory(root: Path, path: Path):
    doc = json.loads(path.read_text(), object_pairs_hook=unique_object)
    entries = doc.get("frames")
    if not isinstance(entries, list):
        raise ValueError("reuse inventory requires a frames list")
    result = {}
    for row in entries:
        name = row.get("name")
        if not isinstance(name, str) or name in result:
            raise ValueError("reuse inventory has a missing or duplicate tensor name")
        rel = PurePosixPath(row.get("frame", ""))
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise ValueError("unsafe reused frame path")
        source = root / Path(*rel.parts)
        data = source.read_bytes()
        if row.get("frame_bytes", len(data)) != len(data) or sha(data) != row.get("frame_sha256"):
            raise ValueError(f"reuse frame integrity mismatch: {name}")
        validate_candidates(row)
        result[name] = {**row, "_source_path": str(source)}
    return result


def validate_reuse_row(row: dict, desc: dict):
    for key in ("source_bytes", "source_sha256", "source_shard", "dtype", "shape"):
        want = desc[key] if key != "source_sha256" else desc["source_sha256"]
        if row.get(key) != want:
            raise ValueError(f"reused frame source identity mismatch: {desc['name']} ({key})")
    validate_candidates(row)
    return row


def verify_receipt(package: Path, row: dict, codec, *, decode: bool = True) -> bool:
    try:
        rel = PurePosixPath(row["frame"])
        if rel.is_absolute() or ".." in rel.parts or not rel.parts or rel.parts[0] != "frames":
            return False
        if rel.name != key_for(row["name"]) + ".bctx":
            return False
        frame_path = package / row["frame"]
        frame = frame_path.read_bytes()
        if len(frame) != row["frame_bytes"] or sha(frame) != row["frame_sha256"]:
            return False
        best = validate_candidates(row)
        if (
            best["frame_sha256"] != row["frame_sha256"]
            or best["frame_bytes"] != row["frame_bytes"]
            or row["complete"] is not True
        ):
            return False
        if not decode:
            return True
        decoded_words = codec.decode(frame)
        shape = row["shape"]
        expected_shape = (shape[0], math.prod(shape[1:])) if len(shape) > 1 else (1, math.prod(shape))
        return (
            tuple(decoded_words.shape) == expected_shape
            and decoded_words.nbytes == row["source_bytes"]
            and sha(decoded_words.tobytes()) == row["source_sha256"]
        )
    except (OSError, ValueError, KeyError):
        return False


def fresh_decode_check(package: Path, frame_path: Path, cache_dir: Path | None = None):
    tool = package / "decoder" / "bitexact_context_package.py"
    cmd = [sys.executable, str(tool), "decode-check", str(package), str(frame_path)]
    if cache_dir is not None:
        cmd.extend(["--cache-dir", str(cache_dir)])
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def build(
    model_dir: Path,
    output: Path,
    census_path: Path,
    metadata_archive: Path,
    jobs: int = 2,
    *,
    strict: bool = True,
    deployment_pin: Path | None = None,
    reuse_frames: Path | None = None,
    reuse_inventory: Path | None = None,
    cache_dir: Path | None = None,
):
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if output.exists() and (output.is_file() or not output.is_dir()):
        raise ValueError("output must be a directory")
    completed_package = (output / "manifest.json").exists()
    output.mkdir(parents=True, exist_ok=True)
    index_raw = (model_dir / "model.safetensors.index.json").read_bytes()
    index_sha = sha(index_raw)
    if strict and index_sha != INDEX_SHA256:
        raise ValueError("pinned source index SHA mismatch")
    index_doc = json.loads(index_raw, object_pairs_hook=unique_object)
    index = index_doc.get("weight_map")
    census, census_file_sha, census_sha = read_census(census_path)
    if set(index) != set(census):
        raise ValueError("index and frozen census tensor names differ")
    if strict and (len(index) != TENSOR_COUNT or index_doc.get("metadata", {}).get("total_size") != SOURCE_BYTES):
        raise ValueError("pinned source tensor count/size mismatch")
    if strict and len(set(index.values())) != SHARD_COUNT:
        raise ValueError("pinned source shard count mismatch")
    if reuse_frames is not None and reuse_inventory is None:
        raise ValueError("--reuse-frames requires --reuse-inventory")
    if reuse_inventory is not None and reuse_frames is None:
        raise ValueError("--reuse-inventory requires --reuse-frames")
    reused = load_reuse_inventory(reuse_frames, reuse_inventory) if reuse_inventory else {}
    if strict and reuse_inventory is not None and len(reused) != 32:
        raise ValueError("strict reuse input must contain all 32 designated prior frames")
    descriptors, headers = tensor_descriptors(model_dir, index)
    physical_source_bytes = 0
    shard_hashes = {}
    for shard in sorted(set(index.values())):
        n = (model_dir / shard).stat().st_size
        physical_source_bytes += n
        shard_hashes[shard] = {"bytes": n, "sha256": sha_file(model_dir / shard)}
        if strict and shard_hashes[shard]["sha256"] != SHARD_SHA256.get(shard):
            raise ValueError(f"pinned source shard SHA mismatch: {shard}")
    if strict and sum(d["source_bytes"] for d in descriptors.values()) != SOURCE_BYTES:
        raise ValueError("pinned source payload byte count mismatch")
    rows = []
    for name, d in descriptors.items():
        frozen = census[name]
        if (
            d["shape"] != frozen["shape"]
            or frozen["dtype"] not in ("BF16", "BF16_LE")
            or d["source_bytes"] != frozen["source_bytes"]
        ):
            raise ValueError(f"source geometry differs from census: {name}")
        d["source_sha256"] = frozen["sha256"]
        if name in reused:
            validate_reuse_row(reused[name], d)
        rows.append(d)
    if set(reused) - set(index):
        raise ValueError("reuse inventory includes tensors outside the pinned index")
    codec_src = Path(__file__).with_name("bitexact_context_codec.py")
    cpp_src = Path(__file__).with_name("bitexact_context_ans.cpp")
    builder_src = Path(__file__).resolve()
    deployment = None
    if strict:
        if deployment_pin is None:
            raise ValueError("strict full-model build requires --baseline-deployment")
        deployment = json.loads(deployment_pin.read_text(), object_pairs_hook=unique_object)
        if sha(metadata_archive.read_bytes()) != deployment.get("metadata_archive_sha256"):
            raise ValueError("metadata archive differs from baseline deployment pin")
        if deployment.get("metadata_assets") is None:
            raise ValueError("baseline deployment pin lacks metadata assets")
    build_pin = {
        "index_sha256": index_sha,
        "census_file_sha256": census_file_sha,
        "census_sha256": census_sha,
        "metadata_archive_sha256": sha(metadata_archive.read_bytes()),
        "codec_source_sha256": sha(codec_src.read_bytes()),
        "codec_cpp_sha256": sha(cpp_src.read_bytes()),
        "package_tool_sha256": sha(builder_src.read_bytes()),
        "deployment_pin_sha256": sha(deployment_pin.read_bytes()) if deployment_pin else None,
        "reuse_inventory_sha256": sha(reuse_inventory.read_bytes()) if reuse_inventory else None,
    }
    existing_pin = output / "build_inputs.json"
    if existing_pin.exists() and json.loads(existing_pin.read_text()) != build_pin:
        raise ValueError("partial package inputs differ from requested build")
    if completed_package:
        existing = verify(output, jobs=8, cache_dir=cache_dir)
        if existing_pin.exists() and json.loads(existing_pin.read_text()) == build_pin:
            return existing
        raise ValueError("completed package requested input pins differ")
    if not existing_pin.exists():
        write_json(existing_pin, build_pin)
    codec_bytes = codec_src.read_bytes()
    atomic_write(output / "decoder" / codec_src.name, codec_bytes)
    atomic_write(output / "decoder" / cpp_src.name, cpp_src.read_bytes())
    atomic_write(output / "decoder" / builder_src.name, builder_src.read_bytes())
    for shard, raw_header in headers.items():
        atomic_write(output / "headers" / (shard + ".header"), raw_header)
    # Retain the original 13 members and their exact payloads inside the provided transport archive.
    assets = {}
    with tarfile.open(metadata_archive, "r:xz") as tf:
        members = tf.getmembers()
        if len(members) != 13:
            raise ValueError("metadata archive must contain exactly 13 assets")
        if len({m.name for m in members}) != 13:
            raise ValueError("metadata archive has duplicate assets")
        archived_index_sha = None
        for member in members:
            rel = PurePosixPath(member.name)
            if not member.isfile() or rel.is_absolute() or ".." in rel.parts or rel.parts[0] != "sidecars":
                raise ValueError("unsafe metadata archive entry")
            payload = tf.extractfile(member).read()
            assets[member.name] = {"bytes": len(payload), "sha256": sha(payload)}
            if member.name == "sidecars/model.safetensors.index.json":
                archived_index_sha = sha(payload)
        if archived_index_sha != index_sha:
            raise ValueError("metadata archive index does not match source index")
        if deployment is not None and assets != deployment["metadata_assets"]:
            raise ValueError("metadata archive contents differ from baseline deployment assets")
    atomic_write(output / "model-assets.tar.xz", metadata_archive.read_bytes())
    allowed = {"build_inputs.json", "model-assets.tar.xz"}
    allowed.update(f"decoder/{n}" for n in (codec_src.name, cpp_src.name, builder_src.name))
    allowed.update(f"headers/{s}.header" for s in headers)
    for path in output.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(output).as_posix()
        if (
            rel not in allowed
            and not (rel.startswith("frames/") and rel.endswith(".bctx"))
            and not (rel.startswith("receipts/") and rel.endswith(".json"))
        ):
            raise ValueError(f"unexpected partial package file: {rel}")
    codec = decode_codec(output, cache_dir)
    frame_root = output / "frames"
    receipt_root = output / "receipts"
    frame_root.mkdir(exist_ok=True)
    receipt_root.mkdir(exist_ok=True)

    def process(d):
        name = d["name"]
        frozen = census[name]
        raw = read_raw(model_dir, d)
        source_sha = sha(raw)
        if source_sha != frozen["sha256"]:
            raise ValueError(f"source tensor SHA differs from frozen census: {name}")
        k = key_for(name)
        frame_rel = f"frames/{k}.bctx"
        receipt_path = receipt_root / f"{k}.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            expected_identity = {
                key: d[key]
                for key in ("name", "source_shard", "dtype", "shape", "offsets", "source_bytes", "source_sha256")
            }
            if any(receipt.get(key) != value for key, value in expected_identity.items()):
                raise ValueError(f"resumed receipt source identity mismatch: {name}")
            if not verify_receipt(output, receipt, codec):
                raise ValueError(f"present resumed receipt/frame is corrupt or incomplete: {name}")
            return receipt
        if name in reused:
            old = validate_reuse_row(reused[name], d)
            frame = Path(old["_source_path"]).read_bytes()
            if sha(frame) != old["frame_sha256"] or len(frame) != old["frame_bytes"]:
                raise ValueError(f"reused frame changed after inventory validation: {name}")
            atomic_write(output / frame_rel, frame)
            receipt = {
                **d,
                "source_sha256": source_sha,
                "frame": frame_rel,
                "frame_bytes": old["frame_bytes"],
                "frame_sha256": old["frame_sha256"],
                "best_config": old["best_config"],
                "candidates": old["candidates"],
                "complete": True,
                "reused": True,
            }
            if not verify_receipt(output, receipt, codec):
                raise ValueError(f"reused stored frame failed fresh decode: {name}")
            check = fresh_decode_check(output, output / frame_rel, cache_dir)
            if check.get("sha256") != source_sha or check.get("bytes") != d["source_bytes"]:
                raise ValueError(f"reused frame failed independent decoder process: {name}")
            write_json(receipt_path, receipt)
            return receipt
        shape = d["shape"]
        rows_n = shape[0] if len(shape) > 1 else 1
        cols_n = math.prod(shape) // rows_n
        words = np.frombuffer(raw, dtype="<u2").reshape(rows_n, cols_n)
        best = None
        candidates = []
        for nr, nc in CONFIGS:
            frame = codec.encode(words, row_classes=nr, col_classes=nc)
            decoded = codec.decode(frame)
            if decoded.tobytes() != raw:
                raise ValueError(f"candidate roundtrip mismatch: {name} {(nr, nc)}")
            decoded_sha = sha(decoded.tobytes())
            candidate = {
                "row_classes": nr,
                "col_classes": nc,
                "frame_bytes": len(frame),
                "frame_sha256": sha(frame),
                "decoded_sha256": decoded_sha,
                "exact": decoded_sha == source_sha,
            }
            candidates.append(candidate)
            if best is None or len(frame) < best["frame_bytes"]:
                best = candidate | {"frame": frame}
        atomic_write(output / frame_rel, best.pop("frame"))
        receipt = {
            **d,
            "source_sha256": source_sha,
            "frame": frame_rel,
            "frame_bytes": best["frame_bytes"],
            "frame_sha256": best["frame_sha256"],
            "best_config": [best["row_classes"], best["col_classes"]],
            "candidates": candidates,
            "complete": True,
        }
        if not verify_receipt(output, receipt, codec):
            raise ValueError(f"stored frame failed fresh decoder verification: {name}")
        write_json(receipt_path, receipt)
        return receipt

    # Honor requested concurrency; tensors above 512 MiB have their own two-slot cap.
    large_slots = threading.Semaphore(2)

    def bounded_process(d):
        if d["source_bytes"] <= 512 * 1024 * 1024:
            return process(d)
        with large_slots:
            return process(d)

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        receipts = list(pool.map(bounded_process, sorted(rows, key=lambda x: x["name"])))
    if len(receipts) != len(index):
        raise ValueError("incomplete tensor receipt set")
    total_payload = sum(x["source_bytes"] for x in receipts)
    if strict and total_payload != SOURCE_BYTES:
        raise ValueError("source tensor byte total mismatch")
    manifest = {
        "schema": "bitexact-context-package-v1",
        "complete": True,
        "source": {
            "index_sha256": index_sha,
            "census_file_sha256": census_file_sha,
            "census_sha256": census_sha,
            "tensor_count": len(index),
            "tensor_payload_bytes": total_payload,
            "physical_source_shard_bytes": physical_source_bytes,
            "shards": shard_hashes,
            "pinned_full_model": strict,
        },
        "configs": [list(x) for x in CONFIGS],
        "tensors": receipts,
        "metadata_assets": assets,
        "headers": {s: {"path": f"headers/{s}.header", "bytes": len(b), "sha256": sha(b)} for s, b in headers.items()},
        "decoder_files": [
            "decoder/bitexact_context_package.py",
            "decoder/bitexact_context_codec.py",
            "decoder/bitexact_context_ans.cpp",
        ],
    }
    # Full physical package accounting includes every file and the self-sized manifest.
    base_rows = inventory(output, exclude={"manifest.json"})
    manifest["files"] = base_rows + [{"path": "manifest.json", "bytes": 0, "sha256": None}]
    for _ in range(30):
        size = len(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        manifest["files"][-1]["bytes"] = size
        manifest["package_bytes"] = sum(x["bytes"] for x in manifest["files"])
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        if len(encoded) == size:
            break
    atomic_write(output / "manifest.json", encoded)
    return manifest


def inventory(root: Path, exclude=frozenset()):
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("package must contain regular files, not external symlinks")
    rows = []
    for p in sorted(x for x in root.rglob("*") if x.is_file() and x.relative_to(root).as_posix() not in exclude):
        rows.append({"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": sha_file(p)})
    return rows


def verify(package: Path, jobs: int = 8, cache_dir: Path | None = None, *, decode_frames: bool = True):
    if not 1 <= jobs <= 8:
        raise ValueError("verify jobs must be between 1 and 8")
    doc = json.loads((package / "manifest.json").read_text(), object_pairs_hook=unique_object)
    if doc.get("schema") != "bitexact-context-package-v1" or not doc.get("complete"):
        raise ValueError("package manifest incomplete or unsupported")
    expected = {x["path"]: x for x in doc["files"]}
    if len(expected) != len(doc["files"]) or any(p.is_symlink() for p in package.rglob("*")):
        raise ValueError("duplicate inventory paths or external package symlinks")
    for rel, row in expected.items():
        parts = PurePosixPath(rel)
        if parts.is_absolute() or ".." in parts.parts or not parts.parts:
            raise ValueError("unsafe package inventory path")
        if (row.get("sha256") is None) != (rel == "manifest.json"):
            raise ValueError("only the self-sized manifest may omit its own digest")
    actual_paths = {p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()}
    if actual_paths != set(expected):
        raise ValueError("package file inventory mismatch")
    for rel, row in expected.items():
        p = package / rel
        if p.stat().st_size != row["bytes"] or (row["sha256"] is not None and sha_file(p) != row["sha256"]):
            raise ValueError(f"package file integrity mismatch: {rel}")
    if doc.get("package_bytes") != sum(x["bytes"] for x in doc["files"]):
        raise ValueError("whole-package byte accounting mismatch")
    if doc["source"].get("pinned_full_model") and (
        doc["source"]["tensor_count"] != TENSOR_COUNT
        or doc["source"]["tensor_payload_bytes"] != SOURCE_BYTES
        or len(doc["source"]["shards"]) != SHARD_COUNT
        or set(doc["source"]["shards"]) != set(SHARD_SHA256)
        or {k: v["sha256"] for k, v in doc["source"]["shards"].items()} != SHARD_SHA256
    ):
        raise ValueError("pinned full-model source census mismatch")
    for shard, header in doc["headers"].items():
        p = package / header["path"]
        if p.stat().st_size != header["bytes"] or sha_file(p) != header["sha256"]:
            raise ValueError(f"safetensors header identity mismatch: {shard}")
    archived_index = None
    with tarfile.open(package / "model-assets.tar.xz", "r:xz") as tf:
        members = tf.getmembers()
        if len(members) != 13 or {m.name for m in members} != set(doc["metadata_assets"]):
            raise ValueError("metadata asset inventory mismatch")
        for m in members:
            if not m.isfile():
                raise ValueError("metadata archive contains a non-file")
            raw = tf.extractfile(m).read()
            pin = doc["metadata_assets"][m.name]
            if len(raw) != pin["bytes"] or sha(raw) != pin["sha256"]:
                raise ValueError(f"metadata asset integrity mismatch: {m.name}")
            if m.name == "sidecars/model.safetensors.index.json":
                archived_index = raw
    if archived_index is None or sha(archived_index) != doc["source"]["index_sha256"]:
        raise ValueError("archived source index identity mismatch")
    index_doc = json.loads(archived_index, object_pairs_hook=unique_object)
    expected_map = index_doc.get("weight_map", {})
    codec = decode_codec(package, cache_dir)
    tensors = doc["tensors"]
    if len(tensors) != doc["source"]["tensor_count"] or len({x["name"] for x in tensors}) != len(tensors):
        raise ValueError("tensor receipt coverage mismatch")
    if set(x["name"] for x in tensors) != set(expected_map):
        raise ValueError("tensor receipts differ from archived index")
    if set(doc["headers"]) != set(doc["source"]["shards"]):
        raise ValueError("original safetensors header coverage differs from shard set")
    rows_by_shard = {}
    for row in tensors:
        rows_by_shard.setdefault(row["source_shard"], {})[row["name"]] = row
    for shard, header_entry in doc["headers"].items():
        raw_header = (package / header_entry["path"]).read_bytes()
        if len(raw_header) < 8:
            raise ValueError(f"short original shard header: {shard}")
        header_n = struct.unpack("<Q", raw_header[:8])[0]
        if header_n != len(raw_header) - 8:
            raise ValueError(f"original header length mismatch: {shard}")
        header_doc = json.loads(raw_header[8:], object_pairs_hook=unique_object)
        shard_rows = rows_by_shard.get(shard, {})
        header_names = set(header_doc) - {"__metadata__"}
        if header_names != set(shard_rows):
            raise ValueError(f"header tensor set differs from receipts: {shard}")
        intervals = []
        for name, row in shard_rows.items():
            desc = header_doc[name]
            if (
                desc.get("dtype") != row["dtype"]
                or desc.get("shape") != row["shape"]
                or desc.get("data_offsets") != row["offsets"]
            ):
                raise ValueError(f"receipt differs from original header: {name}")
            intervals.append((row["offsets"][0], row["offsets"][1]))
        intervals.sort()
        if intervals and (
            intervals[0][0] != 0
            or any(a[1] != b[0] for a, b in zip(intervals, intervals[1:]))
            or intervals[-1][1] + len(raw_header) != doc["source"]["shards"][shard]["bytes"]
        ):
            raise ValueError(f"original shard ranges are incomplete: {shard}")

    def check_row(row):
        if (
            row.get("source_shard") != expected_map[row["name"]]
            or not row.get("complete")
            or not verify_receipt(package, row, codec, decode=decode_frames)
        ):
            raise ValueError(f"incomplete or invalid tensor receipt: {row.get('name')}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(check_row, tensors))
    if sum(x["source_bytes"] for x in tensors) != doc["source"]["tensor_payload_bytes"]:
        raise ValueError("tensor byte accounting mismatch")
    return doc


def restore(package: Path, destination: Path, jobs: int = 8, cache_dir: Path | None = None):
    if not 1 <= jobs <= 8:
        raise ValueError("restore jobs must be between 1 and 8")
    # Frame hashes and complete receipts are checked here; each frame is decoded exactly once below.
    doc = verify(package, jobs=jobs, cache_dir=cache_dir, decode_frames=False)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("restore destination must be new or empty")
    destination.mkdir(parents=True, exist_ok=True)
    codec = decode_codec(package, cache_dir)
    by_shard = {}
    for row in doc["tensors"]:
        by_shard.setdefault(row["source_shard"], []).append(row)

    def restore_shard(item):
        shard, rows = item
        header_entry = doc["headers"].get(shard)
        if not header_entry:
            raise ValueError("missing original safetensors header")
        header = (package / header_entry["path"]).read_bytes()
        if len(header) != header_entry["bytes"] or sha(header) != header_entry["sha256"]:
            raise ValueError("original header integrity mismatch")
        header_size = struct.unpack("<Q", header[:8])[0]
        hdr = json.loads(header[8 : 8 + header_size], object_pairs_hook=unique_object)
        out = destination / shard
        with out.open("wb") as f:
            f.write(header)
            end = max(r["offsets"][1] for r in rows)
            f.truncate(len(header) + end)
            for row in rows:
                tensor = codec.decode((package / row["frame"]).read_bytes())
                raw = tensor.tobytes()
                if len(raw) != row["source_bytes"] or sha(raw) != row["source_sha256"]:
                    raise ValueError("restored tensor does not match receipt")
                a, b = row["offsets"]
                if b - a != len(raw) or row["name"] not in hdr:
                    raise ValueError("restored tensor offset/header mismatch")
                f.seek(len(header) + a)
                f.write(raw)
        if sha_file(out) != doc["source"]["shards"][shard]["sha256"]:
            raise ValueError(f"restored shard differs byte-for-byte: {shard}")
        return shard

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(restore_shard, by_shard.items()))
    # Restore and verify all original sidecars from the archived 13-asset bundle.
    with tarfile.open(package / "model-assets.tar.xz", "r:xz") as tf:
        members = tf.getmembers()
        if {m.name for m in members} != set(doc["metadata_assets"]) or len(members) != 13:
            raise ValueError("metadata asset set mismatch")
        for m in members:
            rel = PurePosixPath(m.name)
            if not m.isfile() or rel.is_absolute() or ".." in rel.parts or rel.parts[0] != "sidecars":
                raise ValueError("unsafe metadata asset")
            raw = tf.extractfile(m).read()
            pin = doc["metadata_assets"][m.name]
            if len(raw) != pin["bytes"] or sha(raw) != pin["sha256"]:
                raise ValueError("metadata asset SHA mismatch")
            if rel.parts[0] != "sidecars" or len(rel.parts) < 2:
                raise ValueError("metadata asset lacks sidecars/ root")
            atomic_write(destination.joinpath(*rel.parts[1:]), raw)
    # Recreate the exact pinned 18-shard population, rejecting missing/extra shards.
    restored_shards = {p.name for p in destination.glob("model-*-of-00018.safetensors")}
    if restored_shards != set(doc["source"]["shards"]):
        raise ValueError("restored shard population mismatch")
    return {"tensors": len(doc["tensors"]), "shards": len(restored_shards), "assets": len(doc["metadata_assets"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model-dir", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--census", type=Path, required=True)
    b.add_argument("--metadata-archive", type=Path, required=True)
    b.add_argument("--jobs", type=int, default=2)
    b.add_argument("--baseline-deployment", type=Path)
    b.add_argument("--reuse-frames", type=Path)
    b.add_argument("--reuse-inventory", type=Path)
    b.add_argument("--cache-dir", type=Path)
    v = sub.add_parser("verify")
    v.add_argument("package", type=Path)
    v.add_argument("--jobs", type=int, default=8)
    v.add_argument("--cache-dir", type=Path)
    r = sub.add_parser("restore")
    r.add_argument("package", type=Path)
    r.add_argument("destination", type=Path)
    r.add_argument("--jobs", type=int, default=8)
    r.add_argument("--cache-dir", type=Path)
    c = sub.add_parser("decode-check")
    c.add_argument("package", type=Path)
    c.add_argument("frame", type=Path)
    c.add_argument("--cache-dir", type=Path)
    a = ap.parse_args()
    if a.command == "build":
        result = build(
            a.model_dir,
            a.out,
            a.census,
            a.metadata_archive,
            a.jobs,
            deployment_pin=a.baseline_deployment,
            reuse_frames=a.reuse_frames,
            reuse_inventory=a.reuse_inventory,
            cache_dir=a.cache_dir,
        )
        print(
            json.dumps(
                {
                    "complete": result["complete"],
                    "tensors": result["source"]["tensor_count"],
                    "package_bytes": result["package_bytes"],
                }
            )
        )
    elif a.command == "verify":
        result = verify(a.package, jobs=a.jobs, cache_dir=a.cache_dir)
        print(
            json.dumps(
                {
                    "verified": True,
                    "tensors": result["source"]["tensor_count"],
                    "package_bytes": result["package_bytes"],
                }
            )
        )
    elif a.command == "restore":
        print(json.dumps(restore(a.package, a.destination, jobs=a.jobs, cache_dir=a.cache_dir)))
    else:
        codec = decode_codec(a.package, a.cache_dir)
        frame = a.frame.read_bytes()
        decoded = codec.decode(frame)
        print(
            json.dumps(
                {
                    "sha256": sha(decoded.tobytes()),
                    "bytes": decoded.nbytes,
                    "shape": list(decoded.shape),
                    "frame_sha256": sha(frame),
                }
            )
        )


if __name__ == "__main__":
    main()
