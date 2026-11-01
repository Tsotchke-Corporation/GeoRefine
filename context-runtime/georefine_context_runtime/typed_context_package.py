"""Lossless, dtype opaque safetensors packages.

The typed package format is deliberately separate from the historical BCTX
package.  It records the bytes in every source tensor exactly, while keeping
the safetensors dtype token opaque so a future or packed storage token can be
packaged before this runtime knows how to materialise it.

The module is copied into ``decoder/`` in a package.  That copy, together
with the generic codec, is sufficient for the ``verify`` and ``restore`` CLI
commands without importing the build tree.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import struct
import sys
import tempfile
import threading
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any, Mapping, Optional


SCHEMA = "georefine-typed-context-package-v1"
PACKAGE_VERSION = 1
INDEX_NAME = "model.safetensors.index.json"
DEFAULT_MAX_SOURCE_BYTES = 256 * 1024 * 1024 * 1024
DEFAULT_MAX_FRAME_BYTES = 512 * 1024 * 1024
# BCTX is useful for large BF16 matrices, but small and unusual tensors stay
# in the dtype-opaque generic frame.  The threshold also keeps package builds
# independent of a C++ compiler for ordinary fixtures.
BF16_CONTEXT_THRESHOLD = 8 * 1024 * 1024
_U64 = struct.Struct("<Q")
_BCTX_CONFIGS = ((1, 1), (16, 1), (16, 4), (16, 16), (64, 4), (64, 16))
_KNOWN_BITS = {"BOOL": 8, "U8": 8, "I8": 8, "F8_E4M3": 8, "F8_E5M2": 8,
              "U16": 16, "I16": 16, "F16": 16, "BF16": 16,
              "U32": 32, "I32": 32, "F32": 32, "U64": 64, "I64": 64, "F64": 64}


class TypedContextPackageError(ValueError):
    """Raised for malformed input, package corruption, or unsafe paths."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _budget(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TypedContextPackageError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> tuple[bytes, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=_unique_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TypedContextPackageError(f"invalid JSON: {path}") from exc
    return raw, value


def _safe_rel(value: str, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise TypedContextPackageError(f"{label} must be a non-empty relative path")
    rel = PurePosixPath(value)
    if rel.is_absolute() or ".." in rel.parts or "" in rel.parts:
        raise TypedContextPackageError(f"unsafe {label}: {value!r}")
    if any(part in (".",) for part in rel.parts):
        raise TypedContextPackageError(f"unsafe {label}: {value!r}")
    return rel


def _inside(root: Path, rel: PurePosixPath, *, label: str) -> Path:
    _safe_rel(rel.as_posix(), label=label)
    path = root.joinpath(*rel.parts)
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise TypedContextPackageError(f"{label} escapes source root: {rel}") from exc
    if path.is_symlink() or not path.is_file():
        raise TypedContextPackageError(f"{label} is not a regular file: {rel}")
    return path


def _assert_no_symlinks(root: Path) -> None:
    if root.is_symlink():
        raise TypedContextPackageError("source/package root may not be a symlink")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise TypedContextPackageError(f"symlinks are not allowed: {path.relative_to(root)}")


def _overlap(left: Path, right: Path) -> bool:
    a = left.resolve(strict=False)
    b = right.resolve(strict=False)
    return a == b or a in b.parents or b in a.parents


def _shape_numel(shape: list[int]) -> int:
    if not isinstance(shape, list) or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in shape):
        raise TypedContextPackageError("safetensors shape must contain non-negative integers")
    value = 1
    for dimension in shape:
        value *= dimension
    return value


def _source_descriptor(dtype: Any, shape: list[int], offsets: Any, body_bytes: int, name: str) -> dict[str, Any]:
    if not isinstance(dtype, str) or not dtype:
        raise TypedContextPackageError(f"{name}: dtype token must be a non-empty string")
    if not isinstance(offsets, list) or len(offsets) != 2 or any(isinstance(x, bool) or not isinstance(x, int) for x in offsets):
        raise TypedContextPackageError(f"{name}: invalid data_offsets")
    start, end = offsets
    if start < 0 or end < start or end > body_bytes:
        raise TypedContextPackageError(f"{name}: data range is outside the shard")
    element_count = _shape_numel(shape)
    if dtype in _KNOWN_BITS and (element_count * _KNOWN_BITS[dtype] + 7) // 8 != end - start:
        raise TypedContextPackageError(f"{name}: dtype {dtype} shape/byte count mismatch")
    return {
        "dtype": dtype,
        "shape": shape,
        "offsets": [start, end],
        "actual_source_bytes": end - start,
        "source_bytes": end - start,
        "element_count": element_count,
    }


def _parse_source(root: Path, index_rel: PurePosixPath, roles: Optional[Mapping[str, str]]) -> tuple[dict[str, Any], bytes, dict[str, bytes], dict[str, Any]]:
    """Read and validate the index and every referenced safetensors header."""
    index_path = _inside(root, index_rel, label="index path")
    index_raw, index_doc = _read_json(index_path)
    if not isinstance(index_doc, dict) or not isinstance(index_doc.get("weight_map"), dict):
        raise TypedContextPackageError("index must contain a weight_map object")
    raw_weight_map = index_doc["weight_map"]
    if not raw_weight_map:
        raise TypedContextPackageError("index weight_map is empty")
    if any(not isinstance(name, str) or not name for name in raw_weight_map):
        raise TypedContextPackageError("index tensor names must be strings")
    if any(not isinstance(shard, str) for shard in raw_weight_map.values()):
        raise TypedContextPackageError("index shard names must be strings")
    weight_map = {name: _safe_rel(shard, label="source shard").as_posix() for name, shard in raw_weight_map.items()}

    role_map: dict[str, str] = {}
    if roles is not None:
        if not isinstance(roles, Mapping):
            raise TypeError("roles must be a mapping")
        for name, role in roles.items():
            if name not in weight_map:
                raise TypedContextPackageError(f"role supplied for unindexed tensor: {name}")
            if role not in ("parameter", "buffer"):
                raise TypedContextPackageError(f"invalid tensor role for {name}: {role!r}")
            role_map[name] = role

    shard_names = sorted(set(weight_map.values()))
    descriptors: dict[str, Any] = {}
    headers: dict[str, bytes] = {}
    shard_info: dict[str, Any] = {}
    for shard in shard_names:
        shard_rel = PurePosixPath(shard)
        shard_path = _inside(root, shard_rel, label="source shard")
        file_size = shard_path.stat().st_size
        with shard_path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise TypedContextPackageError(f"truncated safetensors prefix: {shard}")
            header_size = _U64.unpack(prefix)[0]
            if header_size > file_size - 8 or header_size > 256 * 1024 * 1024:
                raise TypedContextPackageError(f"invalid safetensors header size: {shard}")
            header_json = stream.read(header_size)
            if len(header_json) != header_size:
                raise TypedContextPackageError(f"truncated safetensors header: {shard}")
        try:
            header_doc = json.loads(header_json, object_pairs_hook=_unique_pairs)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TypedContextPackageError(f"invalid safetensors header: {shard}") from exc
        if not isinstance(header_doc, dict):
            raise TypedContextPackageError(f"safetensors header is not an object: {shard}")
        headers[shard] = prefix + header_json
        shard_rows: dict[str, Any] = {}
        intervals: list[tuple[int, int, str]] = []
        body_size = file_size - 8 - header_size
        for name, desc in header_doc.items():
            if name == "__metadata__":
                if not isinstance(desc, dict):
                    raise TypedContextPackageError(f"invalid __metadata__ in {shard}")
                continue
            if name not in weight_map or weight_map[name] != shard:
                raise TypedContextPackageError(f"header/index coverage mismatch for {name}")
            if not isinstance(desc, dict):
                raise TypedContextPackageError(f"invalid tensor descriptor: {name}")
            shape = desc.get("shape")
            if not isinstance(shape, list):
                raise TypedContextPackageError(f"{name}: missing shape")
            row = _source_descriptor(desc.get("dtype"), shape, desc.get("data_offsets"), body_size, name)
            row.update({"name": name, "source_shard": shard})
            if name in descriptors:
                raise TypedContextPackageError(f"duplicate tensor across shards: {name}")
            descriptors[name] = row
            shard_rows[name] = row
            intervals.append((row["offsets"][0], row["offsets"][1], name))
        expected_names = {name for name, mapped in weight_map.items() if mapped == shard}
        if expected_names != set(shard_rows):
            raise TypedContextPackageError(f"index/header tensor coverage mismatch: {shard}")
        intervals.sort()
        cursor = 0
        for start, end, name in intervals:
            if start != cursor:
                raise TypedContextPackageError(f"gapped or overlapping tensor ranges in {shard} at {name}")
            cursor = end
        if cursor != body_size:
            raise TypedContextPackageError(f"tensor ranges do not cover shard data: {shard}")
        shard_info[shard] = {
            "bytes": file_size,
            "sha256": sha256_file(shard_path),
            "header_bytes": len(headers[shard]),
            "header_sha256": sha256(headers[shard]),
        }

    if set(descriptors) != set(weight_map):
        raise TypedContextPackageError("index/header tensor coverage differs")
    for name, row in descriptors.items():
        if name in role_map:
            row["role"] = role_map[name]
    return descriptors, index_raw, headers, {"weight_map": weight_map, "shards": shard_info}


def _read_tensor(root: Path, row: Mapping[str, Any]) -> bytes:
    path = _inside(root, PurePosixPath(row["source_shard"]), label="source shard")
    with path.open("rb") as stream:
        header_size_raw = stream.read(8)
        if len(header_size_raw) != 8:
            raise TypedContextPackageError(f"truncated source shard: {row['source_shard']}")
        header_size = _U64.unpack(header_size_raw)[0]
        stream.seek(8 + header_size + row["offsets"][0])
        raw = stream.read(row["actual_source_bytes"])
    if len(raw) != row["actual_source_bytes"]:
        raise TypedContextPackageError(f"short source tensor: {row['name']}")
    return raw


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise TypedContextPackageError(f"cannot load decoder module: {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before execution (dataclasses and other decorators consult
    # sys.modules while a module is being initialised) and never leave a
    # generated __pycache__ inside an immutable standalone package.
    previous = sys.modules.get(name)
    sys.modules[name] = module
    old_dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    finally:
        sys.dont_write_bytecode = old_dont_write
    return module


def _generic_module(package: Optional[Path] = None):
    try:
        from . import generic_tensor_codec

        return generic_tensor_codec
    except ImportError:
        if package is None:
            raise
        return _load_module(package / "decoder" / "generic_tensor_codec.py", "typed_context_generic_codec")


def _bctx_module(package: Path):
    # The legacy BCTX codec caches its native helper beside its source.  A
    # standalone package is immutable, so redirect that cache outside the
    # package before loading its copied decoder.
    cache = package.parent / ".typed-context-bctx-cache"
    os.environ["BITEXACT_CONTEXT_CACHE_DIR"] = str(cache)
    decoder = package / "decoder" / "bitexact_context_codec.py"
    if not decoder.is_file():
        decoder = Path(__file__).with_name("bitexact_context_codec.py")
    return _load_module(decoder, "typed_context_bctx_codec")


def _encode_row(raw: bytes, row: Mapping[str, Any], output: Path, *, max_source_bytes: int,
                max_frame_bytes: int, bf16_context_min_bytes: int = BF16_CONTEXT_THRESHOLD) -> tuple[bytes, dict[str, Any]]:
    _positive_int(bf16_context_min_bytes, "bf16_context_min_bytes")
    generic = _generic_module()
    dtype = row["dtype"]
    shape = row["shape"]
    bits = None
    if len(raw) > max_source_bytes:
        raise TypedContextPackageError(f"source tensor exceeds max_source_bytes: {row['name']}")
    if dtype in _KNOWN_BITS:
        bits = _KNOWN_BITS[dtype]
    if dtype == "BF16" and len(raw) >= bf16_context_min_bytes and len(shape) >= 2 and all(shape):
        try:
            import numpy as np

            cache_parent = Path(output).parent if output is not None else Path.cwd()
            os.environ["BITEXACT_CONTEXT_CACHE_DIR"] = str(cache_parent / ".typed-context-bctx-cache")
            bctx = _load_module(Path(__file__).with_name("bitexact_context_codec.py"), "typed_context_bctx_builder")
            words = np.frombuffer(raw, dtype="<u2").reshape(shape[0], math.prod(shape[1:]))
            candidates: list[dict[str, Any]] = []
            best: Optional[tuple[bytes, dict[str, Any]]] = None
            for nr, nc in _BCTX_CONFIGS:
                frame = bctx.encode(words, row_classes=nr, col_classes=nc)
                decoded = bctx.decode(frame)
                if decoded.tobytes() != raw:
                    raise TypedContextPackageError(f"BF16 codec roundtrip failed: {row['name']}")
                candidate = {"row_classes": nr, "col_classes": nc, "frame_bytes": len(frame), "frame_sha256": sha256(frame)}
                candidates.append(candidate)
                if best is None or len(frame) < len(best[0]):
                    best = (frame, candidate)
            assert best is not None
            frame, selected = best
            # Keep the historical default selection byte-for-byte stable.  A
            # lower opt-in threshold may expose small matrices where BCTX's
            # framing overhead exceeds the generic frame, so compare both
            # valid lossless candidates before admitting the BCTX frame.
            if bf16_context_min_bytes < BF16_CONTEXT_THRESHOLD:
                generic_frame = generic.encode_tensor(raw, dtype, shape, element_bits=bits,
                                                       max_source_bytes=max_source_bytes)
                if len(generic_frame) <= max_frame_bytes and (len(frame) > max_frame_bytes or len(generic_frame) < len(frame)):
                    return generic_frame, {"codec": "generic", "element_bits": bits}
            if len(frame) > max_frame_bytes:
                raise TypedContextPackageError(f"encoded frame exceeds max_frame_bytes: {row['name']}")
            return frame, {"codec": "bctx", "best_config": [selected["row_classes"], selected["col_classes"]], "candidates": candidates}
        except (ImportError, OSError):
            # Generic is always a valid lossless fallback if the optional
            # native BCTX module is unavailable. Codec/integrity errors are
            # ValueError subclasses and intentionally propagate.
            pass
        except RuntimeError as exc:
            # The existing codec uses RuntimeError specifically for a failed
            # native C++ build. Other runtime failures must remain visible.
            if "C++ coder build failed" not in str(exc):
                raise
    frame = generic.encode_tensor(raw, dtype, shape, element_bits=bits, max_source_bytes=max_source_bytes)
    if len(frame) > max_frame_bytes:
        raise TypedContextPackageError(f"encoded frame exceeds max_frame_bytes: {row['name']}")
    return frame, {"codec": "generic", "element_bits": bits}


def _decode_row(package: Path, row: Mapping[str, Any], *, max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
                max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES) -> bytes:
    _budget(max_source_bytes, "max_source_bytes")
    _budget(max_frame_bytes, "max_frame_bytes")
    if row["actual_source_bytes"] > max_source_bytes:
        raise TypedContextPackageError(f"decoded source exceeds max_source_bytes: {row['name']}")
    frame_path = package / row["frame"]
    try:
        frame_size = frame_path.stat().st_size
    except OSError as exc:
        raise TypedContextPackageError(f"missing tensor frame: {row['name']}") from exc
    if frame_size > max_frame_bytes:
        raise TypedContextPackageError(f"frame exceeds max_frame_bytes: {row['name']}")
    frame = frame_path.read_bytes()
    if len(frame) != row["frame_bytes"] or sha256(frame) != row["frame_sha256"]:
        raise TypedContextPackageError(f"frame integrity mismatch: {row['name']}")
    if row["codec"] == "generic":
        decoded = _generic_module(package).decode_tensor(frame, max_decoded_bytes=row["actual_source_bytes"], max_frame_bytes=max_frame_bytes)
        if decoded.dtype != row["dtype"] or decoded.shape != tuple(row["shape"]):
            raise TypedContextPackageError(f"generic frame metadata mismatch: {row['name']}")
        raw = decoded.payload
    elif row["codec"] == "bctx":
        decoded = _bctx_module(package).decode(frame)
        raw = decoded.tobytes()
    else:
        raise TypedContextPackageError(f"unsupported tensor frame codec: {row['codec']}")
    if len(raw) != row["actual_source_bytes"] or sha256(raw) != row["source_sha256"]:
        raise TypedContextPackageError(f"decoded source SHA mismatch: {row['name']}")
    return raw


def _asset_row_by_source(manifest: Mapping[str, Any], source_path: str) -> Mapping[str, Any]:
    if not isinstance(source_path, str):
        raise TypedContextPackageError("asset source path must be a string")
    matches = [row for row in manifest.get("assets", ())
               if isinstance(row, Mapping) and row.get("source_path") == source_path]
    if len(matches) != 1:
        raise TypedContextPackageError(f"unknown or duplicate asset: {source_path}")
    return matches[0]


def _decode_asset_row(package: Path, row: Mapping[str, Any], *, max_source_bytes: int,
                      max_frame_bytes: int) -> bytes:
    """Decode one logical asset and verify its logical and stored identity."""
    package_path = _inside(package.resolve(), _safe_rel(row.get("package_path"), label="asset package path"),
                           label="asset package path")
    try:
        stored = package_path.read_bytes()
    except OSError as exc:
        raise TypedContextPackageError(f"missing asset: {row.get('source_path')}") from exc
    if len(stored) > max_frame_bytes:
        raise TypedContextPackageError(f"asset frame exceeds max_frame_bytes: {row.get('source_path')}")
    stored_codec = row.get("stored_codec", "raw")
    if stored_codec == "raw":
        raw = stored
    elif stored_codec == "generic":
        shape = row.get("stored_shape")
        if row.get("stored_dtype") != "U8" or not isinstance(shape, (list, tuple)) or len(shape) != 1 \
                or not isinstance(shape[0], int) or shape[0] < 0:
            raise TypedContextPackageError(f"invalid asset storage metadata: {row.get('source_path')}")
        try:
            decoded = _generic_module(package).decode_tensor(
                stored, max_decoded_bytes=max_source_bytes, max_frame_bytes=max_frame_bytes)
        except Exception as exc:
            raise TypedContextPackageError(f"asset frame decode failed: {row.get('source_path')}") from exc
        if decoded.dtype != "U8" or decoded.shape != (shape[0],) or decoded.element_bits != 8:
            raise TypedContextPackageError(f"asset frame metadata mismatch: {row.get('source_path')}")
        raw = decoded.payload
    else:
        raise TypedContextPackageError(f"unsupported asset codec: {stored_codec!r}")
    if len(raw) > max_source_bytes:
        raise TypedContextPackageError(f"asset exceeds max_source_bytes: {row.get('source_path')}")
    if row.get("stored_bytes", len(stored)) != len(stored) or row.get("stored_sha256", sha256(stored)) != sha256(stored):
        raise TypedContextPackageError(f"stored asset integrity mismatch: {row.get('source_path')}")
    if row.get("bytes") != len(raw) or row.get("sha256") != sha256(raw):
        raise TypedContextPackageError(f"asset integrity mismatch: {row.get('source_path')}")
    return raw


def decode_asset(package: Path, source_path: str, *, max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
                 max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES) -> bytes:
    """Read and verify an original asset from a typed package."""
    _budget(max_source_bytes, "max_source_bytes")
    _budget(max_frame_bytes, "max_frame_bytes")
    package = Path(package)
    manifest = _load_manifest(package)
    return _decode_asset_row(package, _asset_row_by_source(manifest, source_path),
                             max_source_bytes=max_source_bytes, max_frame_bytes=max_frame_bytes)


def _copy_decoder(output: Path, *, include_bctx: bool) -> list[dict[str, Any]]:
    decoder = output / "decoder"
    decoder.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__).with_name("generic_tensor_codec.py")]
    if include_bctx:
        files += [Path(__file__).with_name("bitexact_context_codec.py"), Path(__file__).with_name("bitexact_context_ans.cpp")]
    files += [Path(__file__)]
    for source in files:
        _atomic_write(decoder / source.name, source.read_bytes())
    return [{"path": f"decoder/{source.name}", "bytes": (decoder / source.name).stat().st_size, "sha256": sha256_file(decoder / source.name)} for source in files]


def _inventory(root: Path, *, exclude: set[str]) -> list[dict[str, Any]]:
    _assert_no_symlinks(root)
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel in exclude:
            continue
        _safe_rel(rel, label="package inventory path")
        rows.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def build(model_dir: Path, output: Path, *, index_path: Optional[Path] = None, roles: Optional[Mapping[str, str]] = None,
          max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES, max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
          bf16_context_min_bytes: int = BF16_CONTEXT_THRESHOLD, compress_assets: bool = False) -> dict[str, Any]:
    """Build a typed package from an indexed safetensors directory."""
    _budget(max_source_bytes, "max_source_bytes")
    _budget(max_frame_bytes, "max_frame_bytes")
    _positive_int(bf16_context_min_bytes, "bf16_context_min_bytes")
    root = Path(model_dir)
    output = Path(output)
    if not root.is_dir() or root.is_symlink():
        raise TypedContextPackageError("model_dir must be a regular directory")
    if output.is_symlink():
        raise TypedContextPackageError("package output may not be a symlink")
    if _overlap(root, output):
        raise TypedContextPackageError("model_dir and package output may not overlap")
    _assert_no_symlinks(root)
    if output.exists() and any(output.iterdir()):
        raise TypedContextPackageError("output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    index_rel = PurePosixPath(INDEX_NAME if index_path is None else Path(index_path).resolve().relative_to(root.resolve()).as_posix())
    descriptors, index_raw, headers, source_info = _parse_source(root, index_rel, roles)
    for row in descriptors.values():
        if row["actual_source_bytes"] > max_source_bytes:
            raise TypedContextPackageError(f"source tensor exceeds max_source_bytes: {row['name']}")

    frames_dir = output / "frames"
    headers_dir = output / "headers"
    assets_dir = output / "assets"
    frames_dir.mkdir(parents=True, exist_ok=True)
    headers_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    for shard, raw_header in headers.items():
        _atomic_write(headers_dir / (sha256(shard.encode()) + ".header"), raw_header)

    shard_set = set(source_info["weight_map"].values())
    assets: list[dict[str, Any]] = []
    for source in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = source.relative_to(root).as_posix()
        if rel in shard_set:
            continue
        _safe_rel(rel, label="source asset path")
        raw = source.read_bytes()
        if len(raw) > max_source_bytes:
            raise TypedContextPackageError(f"asset exceeds max_source_bytes: {rel}")
        target = assets_dir / rel
        if compress_assets:
            frame = _generic_module(output).encode_tensor(raw, "U8", (len(raw),), element_bits=8,
                                                          max_source_bytes=max_source_bytes)
            if len(frame) > max_frame_bytes:
                raise TypedContextPackageError(f"encoded asset frame exceeds max_frame_bytes: {rel}")
            stored_codec = "generic"
            stored = frame
        else:
            stored_codec = "raw"
            stored = raw
        _atomic_write(target, stored)
        assets.append({"source_path": rel, "package_path": f"assets/{rel}",
                       "bytes": len(raw), "sha256": sha256(raw),
                       "stored_codec": stored_codec, "stored_bytes": len(stored),
                       "stored_sha256": sha256(stored),
                       **({"stored_dtype": "U8", "stored_shape": [len(raw)]} if compress_assets else {})})

    receipts = []
    include_bctx = False
    for name in sorted(descriptors):
        row = descriptors[name]
        raw = _read_tensor(root, row)
        source_digest = sha256(raw)
        frame, codec_info = _encode_row(raw, row, output, max_source_bytes=max_source_bytes,
                                        max_frame_bytes=max_frame_bytes,
                                        bf16_context_min_bytes=bf16_context_min_bytes)
        frame_rel = f"frames/{sha256(name.encode())}.frame"
        _atomic_write(output / frame_rel, frame)
        include_bctx |= codec_info["codec"] == "bctx"
        receipt = {
            **row,
            "source_sha256": source_digest,
            "frame": frame_rel,
            "frame_bytes": len(frame),
            "frame_sha256": sha256(frame),
            "complete": True,
            **codec_info,
        }
        # The frame itself is the source of truth; validate each write before
        # allowing it into the manifest.
        if _decode_row(output, receipt, max_source_bytes=max_source_bytes, max_frame_bytes=max_frame_bytes) != raw:
            raise TypedContextPackageError(f"stored frame failed verification: {name}")
        receipts.append(receipt)

    # A source file changing during the build must never produce a package
    # whose frame receipts and archived shard identity disagree.
    for shard, info in source_info["shards"].items():
        if sha256_file(_inside(root, PurePosixPath(shard), label="source shard")) != info["sha256"]:
            raise TypedContextPackageError(f"source shard changed during build: {shard}")

    decoder_rows = _copy_decoder(output, include_bctx=include_bctx)
    headers_manifest = {
        shard: {"path": f"headers/{sha256(shard.encode())}.header", "bytes": len(raw), "sha256": sha256(raw)}
        for shard, raw in headers.items()
    }
    source_shards = {
        shard: {"bytes": info["bytes"], "sha256": info["sha256"]}
        for shard, info in source_info["shards"].items()
    }
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "package_version": PACKAGE_VERSION,
        "complete": True,
        "provenance": {
            "builder": "georefine_context_runtime.typed_context_package",
            "source_index": index_rel.as_posix(),
            "source_index_sha256": sha256(index_raw),
            "source_tensor_count": len(receipts),
            "source_tensor_bytes": sum(row["actual_source_bytes"] for row in receipts),
        },
        "source": {"index_path": index_rel.as_posix(), "index_sha256": sha256(index_raw), "shards": source_shards},
        "headers": headers_manifest,
        "assets": assets,
        "tensors": receipts,
        "decoder_files": decoder_rows,
    }
    # The manifest digest is defined over the same canonical manifest with its
    # own inventory digest temporarily set to null.  This avoids a recursive
    # hash while still giving every inventory row a verifiable SHA-256.
    manifest["files"] = _inventory(output, exclude={"manifest.json"}) + [{"path": "manifest.json", "bytes": 0, "sha256": None}]
    encoded = b""
    for _ in range(32):
        manifest["files"][-1]["sha256"] = None
        manifest["package_bytes"] = sum(row["bytes"] for row in manifest["files"])
        unsigned = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        manifest["files"][-1]["sha256"] = sha256(unsigned)
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        old_bytes = manifest["files"][-1]["bytes"]
        manifest["files"][-1]["bytes"] = len(encoded)
        if len(encoded) == old_bytes:
            break
    manifest["package_bytes"] = sum(row["bytes"] for row in manifest["files"])
    # Recompute once with the converged byte accounting and retain the final
    # signed form.  The bounded loop above normally converges in two passes.
    for _ in range(4):
        manifest["files"][-1]["sha256"] = None
        unsigned = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        manifest["files"][-1]["sha256"] = sha256(unsigned)
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        if len(encoded) == manifest["files"][-1]["bytes"]:
            break
        manifest["files"][-1]["bytes"] = len(encoded)
        manifest["package_bytes"] = sum(row["bytes"] for row in manifest["files"])
    _atomic_write(output / "manifest.json", encoded)
    return manifest


def _load_manifest(package: Path) -> dict[str, Any]:
    package = Path(package)
    if not package.is_dir() or package.is_symlink():
        raise TypedContextPackageError("package must be a regular directory")
    _assert_no_symlinks(package)
    _, manifest = _read_json(package / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA or manifest.get("complete") is not True:
        raise TypedContextPackageError("unsupported or incomplete typed package")
    return manifest


def verify(package: Path, *, decode_frames: bool = True, max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
           max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES) -> dict[str, Any]:
    """Verify package inventory, source metadata, and every decoded source hash."""
    package = Path(package)
    _budget(max_source_bytes, "max_source_bytes")
    _budget(max_frame_bytes, "max_frame_bytes")
    manifest = _load_manifest(package)
    files = manifest.get("files")
    if not isinstance(files, list) or len({row.get("path") for row in files}) != len(files):
        raise TypedContextPackageError("invalid or duplicate package inventory")
    expected = {}
    for row in files:
        if not isinstance(row, dict):
            raise TypedContextPackageError("invalid package inventory row")
        rel = _safe_rel(row.get("path"), label="package inventory path")
        if not isinstance(row.get("sha256"), str) or len(row["sha256"]) != 64:
            raise TypedContextPackageError(f"missing inventory SHA-256: {rel}")
        expected[rel.as_posix()] = row
    assets_value = manifest.get("assets")
    headers_value = manifest.get("headers")
    tensors_value = manifest.get("tensors")
    if not isinstance(assets_value, list) or not isinstance(headers_value, dict) or not isinstance(tensors_value, list):
        raise TypedContextPackageError("invalid typed package manifest collections")
    # Validate all manifest-controlled paths before touching package files;
    # this makes path traversal fail closed even if the manifest was edited.
    for asset in assets_value:
        if not isinstance(asset, dict):
            raise TypedContextPackageError("invalid asset receipt")
        _safe_rel(asset.get("source_path"), label="asset source path")
        _safe_rel(asset.get("package_path"), label="asset package path")
        stored_codec = asset.get("stored_codec", "raw")
        if stored_codec not in {"raw", "generic"}:
            raise TypedContextPackageError(f"unsupported asset codec: {stored_codec!r}")
        if not isinstance(asset.get("bytes"), int) or asset["bytes"] < 0 or not isinstance(asset.get("sha256"), str):
            raise TypedContextPackageError(f"invalid logical asset identity: {asset.get('source_path')}")
        if stored_codec == "generic":
            shape = asset.get("stored_shape")
            if asset.get("stored_dtype") != "U8" or not isinstance(shape, list) or len(shape) != 1 \
                    or shape[0] != asset["bytes"]:
                raise TypedContextPackageError(f"invalid asset storage metadata: {asset.get('source_path')}")
        if not isinstance(asset.get("stored_bytes", asset["bytes"]), int) or asset.get("stored_bytes", asset["bytes"]) < 0:
            raise TypedContextPackageError(f"invalid stored asset size: {asset.get('source_path')}")
    for shard, header in headers_value.items():
        _safe_rel(shard, label="source shard")
        if not isinstance(header, dict):
            raise TypedContextPackageError(f"invalid header receipt: {shard}")
        _safe_rel(header.get("path"), label="header path")
    # Stat declared frame/source sizes before inventory hashing or decoding so
    # caller budgets reject oversized data without reading its payload.
    for row in tensors_value:
        if not isinstance(row, dict):
            raise TypedContextPackageError("invalid tensor receipt")
        _budget(row.get("actual_source_bytes"), "tensor source bytes")
        if row["actual_source_bytes"] > max_source_bytes:
            raise TypedContextPackageError(f"decoded source exceeds max_source_bytes: {row.get('name')}")
        frame_rel = _safe_rel(row.get("frame"), label="tensor frame path")
        try:
            frame_size = (package / frame_rel).stat().st_size
        except OSError as exc:
            raise TypedContextPackageError(f"missing tensor frame: {row.get('name')}") from exc
        if frame_size > max_frame_bytes:
            raise TypedContextPackageError(f"frame exceeds max_frame_bytes: {row.get('name')}")
    actual = {p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()}
    if actual != set(expected):
        raise TypedContextPackageError("package file inventory mismatch")
    for rel, row in expected.items():
        path = package / rel
        if path.stat().st_size != row.get("bytes"):
            raise TypedContextPackageError(f"package file integrity mismatch: {rel}")
        if rel == "manifest.json":
            unsigned_manifest = json.loads(path.read_text(), object_pairs_hook=_unique_pairs)
            unsigned_manifest["files"][-1]["sha256"] = None
            unsigned = json.dumps(unsigned_manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            if sha256(unsigned) != row["sha256"]:
                raise TypedContextPackageError("manifest integrity mismatch")
        elif sha256_file(path) != row["sha256"]:
            raise TypedContextPackageError(f"package file integrity mismatch: {rel}")
    if manifest.get("package_bytes") != sum(row["bytes"] for row in files):
        raise TypedContextPackageError("package byte accounting mismatch")
    for asset in assets_value:
        if not isinstance(asset, dict):
            raise TypedContextPackageError("invalid asset receipt")
        _safe_rel(asset.get("source_path"), label="asset source path")
        package_path = _safe_rel(asset.get("package_path"), label="asset package path")
        if package_path.as_posix() not in expected:
            raise TypedContextPackageError("asset package path is absent from inventory")
        inventory_row = expected[package_path.as_posix()]
        if asset.get("stored_bytes", asset.get("bytes")) != inventory_row.get("bytes") \
                or asset.get("stored_sha256", asset.get("sha256")) != inventory_row.get("sha256"):
            raise TypedContextPackageError(f"asset inventory identity mismatch: {asset.get('source_path')}")
        if decode_frames:
            _decode_asset_row(package, asset, max_source_bytes=max_source_bytes, max_frame_bytes=max_frame_bytes)
    headers = headers_value
    source = manifest.get("source")
    source_shards = source.get("shards") if isinstance(source, dict) else None
    if not isinstance(headers, dict) or not isinstance(source_shards, dict) or set(headers) != set(source_shards):
        raise TypedContextPackageError("source shard/header coverage mismatch")
    for shard, info in source_shards.items():
        _safe_rel(shard, label="source shard")
        if not isinstance(info, dict) or not isinstance(info.get("bytes"), int) or info["bytes"] < 0 or not isinstance(info.get("sha256"), str):
            raise TypedContextPackageError(f"invalid source shard receipt: {shard}")
    for shard, header in headers.items():
        _safe_rel(shard, label="source shard")
        if not isinstance(header, dict) or _safe_rel(header.get("path"), label="header path").as_posix() not in expected:
            raise TypedContextPackageError(f"invalid header receipt: {shard}")
    if not isinstance(source, dict):
        raise TypedContextPackageError("missing source manifest")
    index_assets = [a for a in manifest.get("assets", []) if a.get("source_path") == source.get("index_path")]
    if len(index_assets) != 1 or index_assets[0].get("sha256") != source.get("index_sha256"):
        raise TypedContextPackageError("source index is not preserved as an asset")
    index_asset = index_assets[0]
    index_raw = _decode_asset_row(package, index_asset, max_source_bytes=max(max_source_bytes, index_asset["bytes"]),
                                  max_frame_bytes=max_frame_bytes)
    try:
        index_doc = json.loads(index_raw, object_pairs_hook=_unique_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TypedContextPackageError("preserved source index is invalid JSON") from exc
    if not isinstance(index_doc, dict) or not isinstance(index_doc.get("weight_map"), dict):
        raise TypedContextPackageError("preserved source index lacks weight_map")
    actual_map = {}
    for name, shard in index_doc["weight_map"].items():
        actual_map[name] = _safe_rel(shard, label="index shard").as_posix()
    tensors = tensors_value
    if not isinstance(tensors, list) or len({row.get("name") for row in tensors}) != len(tensors):
        raise TypedContextPackageError("invalid tensor receipt coverage")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise TypedContextPackageError("missing package provenance")
    if len(tensors) != provenance.get("source_tensor_count"):
        raise TypedContextPackageError("tensor count provenance mismatch")
    if sum(row.get("actual_source_bytes", -1) for row in tensors) != provenance.get("source_tensor_bytes"):
        raise TypedContextPackageError("tensor byte provenance mismatch")
    if {row.get("source_shard") for row in tensors} != set(source_shards):
        raise TypedContextPackageError("tensor/source shard coverage mismatch")
    if actual_map != {row.get("name"): row.get("source_shard") for row in tensors}:
        raise TypedContextPackageError("preserved index weight_map differs from tensor receipts")
    rows_by_shard: dict[str, dict[str, dict[str, Any]]] = {}
    for row in tensors:
        if not row.get("complete") or row.get("actual_source_bytes") != row.get("source_bytes"):
            raise TypedContextPackageError(f"incomplete tensor receipt: {row.get('name')}")
        _safe_rel(row.get("source_shard"), label="tensor source shard")
        frame_path = _safe_rel(row.get("frame"), label="tensor frame path")
        if frame_path.as_posix() not in expected:
            raise TypedContextPackageError(f"tensor frame is absent from inventory: {row.get('name')}")
        frame_inventory = expected[frame_path.as_posix()]
        if row.get("frame_bytes") != frame_inventory.get("bytes") or row.get("frame_sha256") != frame_inventory.get("sha256"):
            raise TypedContextPackageError(f"tensor frame inventory identity mismatch: {row.get('name')}")
        rows_by_shard.setdefault(row["source_shard"], {})[row["name"]] = row
        if decode_frames:
            _decode_row(package, row, max_source_bytes=max_source_bytes, max_frame_bytes=max_frame_bytes)
    for shard, header_row in headers.items():
        header = (package / header_row["path"]).read_bytes()
        if len(header) < 8:
            raise TypedContextPackageError(f"short preserved header: {shard}")
        header_size = _U64.unpack(header[:8])[0]
        if header_size != len(header) - 8 or header_row.get("bytes") != len(header) or header_row.get("sha256") != sha256(header):
            raise TypedContextPackageError(f"preserved header size/hash mismatch: {shard}")
        shard_bytes = source_shards[shard].get("bytes")
        if not isinstance(shard_bytes, int) or shard_bytes < len(header):
            raise TypedContextPackageError(f"invalid source shard byte count: {shard}")
        try:
            header_doc = json.loads(header[8 : 8 + header_size], object_pairs_hook=_unique_pairs)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TypedContextPackageError(f"invalid preserved header: {shard}") from exc
        names = set(header_doc) - {"__metadata__"}
        rows = rows_by_shard.get(shard, {})
        if names != set(rows):
            raise TypedContextPackageError(f"header/receipt tensor coverage mismatch: {shard}")
        cursor = 0
        ordered_rows = sorted(rows.values(), key=lambda item: (item["offsets"][0], item["offsets"][1], item["name"]))
        for name, row in rows.items():
            desc = header_doc[name]
            if desc.get("dtype") != row.get("dtype") or desc.get("shape") != row.get("shape") or desc.get("data_offsets") != row.get("offsets"):
                raise TypedContextPackageError(f"receipt differs from preserved header: {name}")
        digest = hashlib.sha256()
        digest.update(header)
        for row in ordered_rows:
            start, end = row["offsets"]
            if start != cursor or end < start or end - start != row.get("actual_source_bytes"):
                raise TypedContextPackageError(f"source tensor ranges are incomplete or overlapping: {shard}")
            cursor = end
            if decode_frames:
                digest.update(_decode_row(package, row, max_source_bytes=max_source_bytes, max_frame_bytes=max_frame_bytes))
        if cursor != shard_bytes - len(header):
            raise TypedContextPackageError(f"source tensor ranges do not cover shard: {shard}")
        if decode_frames and digest.hexdigest() != source_shards[shard].get("sha256"):
            raise TypedContextPackageError(f"reconstructed shard SHA mismatch: {shard}")
    return manifest


def _exclusive_write(path: Path, data: bytes, *, root: Optional[Path] = None) -> None:
    _prepare_restore_parent(path.parent, root or path.parent)
    if path.exists() or path.is_symlink():
        raise TypedContextPackageError(f"refusing to overwrite restore destination: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o644)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _prepare_restore_parent(parent: Path, root: Path) -> None:
    root = root.resolve(strict=False)
    if root.is_symlink():
        raise TypedContextPackageError(f"refusing symlink restore directory: {root}")
    try:
        relative = parent.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise TypedContextPackageError("restore path escapes destination") from exc
    current = root
    if current.exists() and (current.is_symlink() or not current.is_dir()):
        raise TypedContextPackageError(f"invalid restore destination directory: {current}")
    current.mkdir(parents=True, exist_ok=True)
    for part in relative.parts:
        current = current / part
        if current.exists():
            if current.is_symlink() or not current.is_dir():
                raise TypedContextPackageError(f"refusing symlink restore directory: {current}")
        else:
            current.mkdir()


def _exclusive_open(path: Path, *, root: Path):
    _prepare_restore_parent(path.parent, root)
    if path.exists() or path.is_symlink():
        raise TypedContextPackageError(f"refusing to overwrite restore destination: {path}")
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644), "wb")


def restore(package: Path, destination: Path, *, max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
            max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES) -> dict[str, int]:
    """Restore every original shard and asset byte-for-byte into a new tree."""
    package = Path(package)
    destination = Path(destination)
    if _overlap(package, destination):
        raise TypedContextPackageError("package and restore destination may not overlap")
    manifest = verify(package, decode_frames=False, max_source_bytes=max_source_bytes, max_frame_bytes=max_frame_bytes)
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir() or any(destination.iterdir()):
            raise TypedContextPackageError("refusing to overwrite restore destination; it must be a new or empty directory")
    else:
        destination.mkdir(parents=True)
    headers = manifest.get("headers", {})
    tensors_by_shard: dict[str, list[dict[str, Any]]] = {}
    for row in manifest["tensors"]:
        tensors_by_shard.setdefault(row["source_shard"], []).append(row)
    restored = 0
    for shard, rows in tensors_by_shard.items():
        header_row = headers.get(shard)
        if not isinstance(header_row, dict):
            raise TypedContextPackageError(f"missing original header: {shard}")
        header = (package / header_row["path"]).read_bytes()
        if len(header) != header_row["bytes"] or sha256(header) != header_row["sha256"]:
            raise TypedContextPackageError(f"original header integrity mismatch: {shard}")
        expected = manifest["source"]["shards"][shard]
        output_path = destination / shard
        digest = hashlib.sha256()
        cursor = 0
        try:
            with _exclusive_open(output_path, root=destination) as stream:
                stream.write(header)
                digest.update(header)
                for row in sorted(rows, key=lambda item: (item["offsets"][0], item["offsets"][1], item["name"])):
                    start, end = row["offsets"]
                    if start != cursor or end < start or end - start != row["actual_source_bytes"]:
                        raise TypedContextPackageError(f"restore ranges are incomplete or overlapping: {shard}")
                    raw = _decode_row(package, row, max_source_bytes=max_source_bytes, max_frame_bytes=max_frame_bytes)
                    if len(raw) != end - start:
                        raise TypedContextPackageError(f"restore range mismatch: {row['name']}")
                    stream.write(raw)
                    digest.update(raw)
                    cursor = end
                if cursor != expected["bytes"] - len(header):
                    raise TypedContextPackageError(f"restore ranges do not cover shard: {shard}")
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            try:
                output_path.unlink()
            except FileNotFoundError:
                pass
            raise
        if digest.hexdigest() != expected["sha256"]:
            output_path.unlink()
            raise TypedContextPackageError(f"restored shard differs byte-for-byte: {shard}")
        restored += 1
    for asset in manifest.get("assets", []):
        raw = _decode_asset_row(package, asset, max_source_bytes=max_source_bytes, max_frame_bytes=max_frame_bytes)
        _exclusive_write(destination / asset["source_path"], raw, root=destination)
    return {"tensors": len(manifest["tensors"]), "shards": restored, "assets": len(manifest.get("assets", []))}


class TypedContextResolver:
    """Lazy resolver over a verified typed package."""

    def __init__(self, package: Path, *, verify_package: bool = True,
                 max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
                 max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES):
        self.package = Path(package)
        self._max_source_bytes = _budget(max_source_bytes, "max_source_bytes")
        self._max_frame_bytes = _budget(max_frame_bytes, "max_frame_bytes")
        # Even the opt-out skips only frame decoding. Inventory and manifest
        # hashes remain mandatory before exposing resolver state.
        manifest = verify(self.package, decode_frames=False,
                          max_source_bytes=self._max_source_bytes,
                          max_frame_bytes=self._max_frame_bytes)
        self.manifest = _freeze(copy.deepcopy(manifest))
        self._entries = MappingProxyType({row["name"]: row for row in self.manifest["tensors"]})
        self.manifest_sha256 = sha256_file(self.package / "manifest.json")
        self._condition = threading.Condition()
        self._active_readers = 0
        self._closing = False
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise TypedContextPackageError("typed context resolver is closed")

    @contextmanager
    def _reader(self):
        with self._condition:
            if self._closed or self._closing:
                raise TypedContextPackageError("typed context resolver is closed")
            self._active_readers += 1
        try:
            current_hash = sha256_file(self.package / "manifest.json")
            if current_hash != self.manifest_sha256:
                raise TypedContextPackageError("manifest changed after resolver construction")
            yield
        finally:
            with self._condition:
                self._active_readers -= 1
                self._condition.notify_all()

    def _assert_manifest_pin(self) -> None:
        if sha256_file(self.package / "manifest.json") != self.manifest_sha256:
            raise TypedContextPackageError("manifest changed after resolver construction")

    def _verify_integrity_unlocked(self) -> dict[str, Any]:
        current_hash = sha256_file(self.package / "manifest.json")
        if current_hash != self.manifest_sha256:
            raise TypedContextPackageError("manifest changed after resolver construction")
        result = verify(self.package, decode_frames=False,
                        max_source_bytes=self._max_source_bytes,
                        max_frame_bytes=self._max_frame_bytes)
        if sha256_file(self.package / "manifest.json") != self.manifest_sha256:
            raise TypedContextPackageError("manifest changed during resolver integrity verification")
        return result

    def verify_integrity(self) -> dict[str, Any]:
        """Recheck the pinned manifest hash and complete package inventory."""
        with self._reader():
            return self._verify_integrity_unlocked()

    def close(self) -> None:
        with self._condition:
            while self._closing and not self._closed:
                self._condition.wait()
            if self._closed:
                return
            self._closing = True
            while self._active_readers:
                self._condition.wait()
        try:
            self._verify_integrity_unlocked()
        except BaseException:
            with self._condition:
                self._closing = False
                self._condition.notify_all()
            raise
        with self._condition:
            self._closed = True
            self._closing = False
            self._condition.notify_all()

    def __enter__(self) -> "TypedContextResolver":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @property
    def tensor_names(self) -> tuple[str, ...]:
        self._ensure_open()
        return tuple(sorted(self._entries))

    def entry(self, name: str) -> dict[str, Any]:
        self._ensure_open()
        try:
            return _thaw(self._entries[name])
        except KeyError as exc:
            raise KeyError(f"unknown tensor: {name}") from exc

    def resolve(self, name: str) -> dict[str, Any]:
        """Return a detached source entry for adapters such as MLX."""
        return self.entry(name)

    def decode_bytes(self, name: str) -> bytes:
        with self._reader():
            raw = _decode_row(self.package, self._entries[name], max_source_bytes=self._max_source_bytes,
                              max_frame_bytes=self._max_frame_bytes)
            self._assert_manifest_pin()
            return raw

    def read_asset(self, source_path: str) -> bytes:
        """Read and verify an original package asset by logical source path."""
        with self._reader():
            row = _asset_row_by_source(self.manifest, source_path)
            raw = _decode_asset_row(self.package, row, max_source_bytes=max(self._max_source_bytes, row["bytes"]),
                                    max_frame_bytes=self._max_frame_bytes)
            self._assert_manifest_pin()
            return raw

    def numpy_view(self, name: str):
        """Return a native NumPy view; BF16 intentionally remains uint16 bits."""
        import numpy as np

        self._ensure_open()
        row = self._entries[name]
        raw = self.decode_bytes(name)
        dtypes = {"F32": "<f4", "F64": "<f8", "F16": "<f2", "BF16": "<u2",
                  "I8": "i1", "U8": "u1", "I16": "<i2", "U16": "<u2",
                  "I32": "<i4", "U32": "<u4", "I64": "<i8", "U64": "<u8", "BOOL": "?"}
        if row["dtype"] in ("F8_E4M3", "F8_E5M2"):
            raise TypedContextPackageError(f"FP8 storage-only dtype cannot be materialized as a NumPy view: {row['dtype']}")
        dtype = dtypes.get(row["dtype"])
        if dtype is None:
            raise TypedContextPackageError(f"no NumPy native view for storage dtype {row['dtype']!r}")
        return np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(tuple(row["shape"]))

    native_numpy = numpy_view

    def materialize_torch(self, name: str, *, device: Optional[str] = None):
        """Materialise through the installed safetensors dtype registry.

        The installed Rust/Torch binding decides whether a dtype token is
        supported.  No storage-token whitelist is maintained here, so a valid
        future token can work after upgrading safetensors.
        """
        self._ensure_open()
        row = self._entries[name]
        if not isinstance(row["dtype"], str):
            raise TypedContextPackageError(f"Torch materialization requires a safetensors dtype token: {row['dtype']!r}")
        try:
            from safetensors.torch import load
        except ImportError as exc:
            raise TypedContextPackageError("Torch materialization requires safetensors.torch") from exc
        raw = self.decode_bytes(name)
        header = json.dumps({name: {"dtype": row["dtype"], "shape": list(row["shape"]), "data_offsets": [0, len(raw)]}}, separators=(",", ":")).encode()
        payload = _U64.pack(len(header)) + header + raw
        try:
            tensor = load(payload)[name]
        except Exception as exc:
            raise TypedContextPackageError(f"installed safetensors dtype registry cannot materialize {row['dtype']!r}") from exc
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    to_torch = materialize_torch

    def materialize_mlx(self, name: str):
        """Return a NumPy native view for MLX callers without importing Torch."""
        self._ensure_open()
        return self.numpy_view(name)


Resolver = TypedContextResolver
TypedTensorResolver = TypedContextResolver
build_package = build
open_package = TypedContextResolver


def read_asset(resolver: TypedContextResolver, source_path: str) -> bytes:
    """Shared asset reader for typed Torch/MLX adapters."""
    return resolver.read_asset(source_path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("model_dir", type=Path)
    build_parser.add_argument("output", type=Path)
    build_parser.add_argument("--index-path", type=Path)
    build_parser.add_argument("--max-source-bytes", type=int, default=DEFAULT_MAX_SOURCE_BYTES)
    build_parser.add_argument("--max-frame-bytes", type=int, default=DEFAULT_MAX_FRAME_BYTES)
    build_parser.add_argument("--bf16-context-min-bytes", "--bf16_context_min_bytes", type=int,
                              default=BF16_CONTEXT_THRESHOLD)
    build_parser.add_argument("--compress-assets", action="store_true",
                              help="store original non-shard assets in bounded generic U8 frames")
    check = sub.add_parser("verify")
    check.add_argument("package", type=Path)
    check.add_argument("--no-decode", action="store_true")
    check.add_argument("--max-source-bytes", type=int, default=DEFAULT_MAX_SOURCE_BYTES)
    check.add_argument("--max-frame-bytes", type=int, default=DEFAULT_MAX_FRAME_BYTES)
    restore_parser = sub.add_parser("restore")
    restore_parser.add_argument("package", type=Path)
    restore_parser.add_argument("destination", type=Path)
    restore_parser.add_argument("--max-source-bytes", type=int, default=DEFAULT_MAX_SOURCE_BYTES)
    restore_parser.add_argument("--max-frame-bytes", type=int, default=DEFAULT_MAX_FRAME_BYTES)
    args = parser.parse_args(argv)
    if args.command == "build":
        manifest = build(args.model_dir, args.output, index_path=args.index_path,
                         max_source_bytes=args.max_source_bytes,
                         max_frame_bytes=args.max_frame_bytes,
                         bf16_context_min_bytes=args.bf16_context_min_bytes,
                         compress_assets=args.compress_assets)
        print(json.dumps({"built": True, "tensors": len(manifest["tensors"]),
                          "schema": manifest["schema"]}, sort_keys=True))
    elif args.command == "verify":
        manifest = verify(args.package, decode_frames=not args.no_decode,
                          max_source_bytes=args.max_source_bytes, max_frame_bytes=args.max_frame_bytes)
        print(json.dumps({"verified": True, "tensors": len(manifest["tensors"]), "schema": manifest["schema"]}, sort_keys=True))
    else:
        print(json.dumps(restore(args.package, args.destination,
                                 max_source_bytes=args.max_source_bytes,
                                 max_frame_bytes=args.max_frame_bytes), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BF16_CONTEXT_THRESHOLD", "Resolver", "SCHEMA", "TypedContextPackageError",
    "TypedContextResolver", "TypedTensorResolver", "build", "build_package", "decode_asset", "main", "open_package", "read_asset", "restore", "sha256", "verify",
]
