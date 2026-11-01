"""Reader for the immutable qwen38-v21-full-pack-v1 tensor package."""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Dict, Iterator, List

import numpy as np

from . import codec_v21 as cv21


class V21FullPackError(RuntimeError):
    """Malformed or corrupt full-pack artifact."""


_SOURCE_DTYPES = {
    "BOOL": (1, np.bool_), "U8": (1, np.uint8), "I8": (1, np.int8),
    "U16": (2, np.uint16), "I16": (2, np.int16), "F16": (2, np.uint16),
    "BF16": (2, np.uint16), "U32": (4, np.uint32), "I32": (4, np.int32),
    "F32": (4, np.float32), "U64": (8, np.uint64), "I64": (8, np.int64),
    "F64": (8, np.float64), "F8_E4M3": (1, np.uint8),
    "F8_E4M3FN": (1, np.uint8), "F8_E5M2": (1, np.uint8), "F8_E8M0": (1, np.uint8),
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise V21FullPackError(f"unsafe artifact path {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise V21FullPackError(f"unsafe artifact path {value!r}")
    if path.as_posix() != value:
        raise V21FullPackError(f"non-canonical artifact path {value!r}")
    return value


def _regular_file(root: Path, relative: str) -> Path:
    try:
        root_mode = root.lstat().st_mode
    except OSError as exc:
        raise V21FullPackError(f"missing artifact root {root}: {exc}") from exc
    if not stat.S_ISDIR(root_mode):
        raise V21FullPackError(f"artifact root is not a real directory: {root}")
    path = root
    parts = PurePosixPath(relative).parts
    for part in parts[:-1]:
        path = path / part
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise V21FullPackError(f"missing artifact directory for {relative}: {exc}") from exc
        if not stat.S_ISDIR(mode):
            raise V21FullPackError(f"artifact parent is not a real directory: {path.relative_to(root)}")
    path = path / parts[-1]
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise V21FullPackError(f"missing artifact file {relative}: {exc}") from exc
    if not stat.S_ISREG(mode):
        raise V21FullPackError(f"artifact path is not a regular file: {relative}")
    return path


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


class V21FullPackArtifact:
    """Read full-pack records without changing or repacking the package."""

    def __init__(self, path: str):
        self.path = os.path.abspath(os.path.expanduser(str(path)))
        self.root = Path(self.path)
        manifest_path = _regular_file(self.root, "manifest.json")
        try:
            with manifest_path.open("r", encoding="utf-8") as stream:
                self.manifest = json.load(stream, object_pairs_hook=_unique_object)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise V21FullPackError(f"invalid full-pack manifest: {exc}") from exc
        manifest = self.manifest
        if not isinstance(manifest, dict) or manifest.get("schema") != "qwen38-v21-full-pack-v1":
            raise V21FullPackError("unsupported full-pack schema")
        if manifest.get("complete") is not True or manifest.get("serving_qualification") is not False:
            raise V21FullPackError("manifest is incomplete or claims an unsupported qualification state")
        if manifest.get("ratio_claim") is not None:
            raise V21FullPackError("full-pack manifest must not claim a ratio")
        tensors = manifest.get("tensors")
        if not isinstance(tensors, list) or type(manifest.get("tensor_count")) is not int:
            raise V21FullPackError("invalid tensor inventory")
        if manifest["tensor_count"] != len(tensors) or not tensors:
            raise V21FullPackError("tensor_count does not match nonempty tensor inventory")
        self.codec = cv21
        self.format = cv21.FORMAT
        self.records: Dict[str, dict] = {}
        blob_paths = set()
        for row in tensors:
            if not isinstance(row, dict):
                raise V21FullPackError("tensor inventory row must be an object")
            name = row.get("name")
            if not isinstance(name, str) or not name or name in self.records:
                raise V21FullPackError(f"invalid or duplicate tensor name {name!r}")
            header = row.get("header")
            if not isinstance(header, dict):
                raise V21FullPackError(f"{name}: missing codec header")
            for key in ("name", "dtype", "shape"):
                if header.get(key) != row.get(key):
                    raise V21FullPackError(f"{name}: row/header {key} mismatch")
            if (header.get("format") != cv21.FORMAT
                    or type(header.get("format_version")) is not int
                    or header["format_version"] != cv21.FORMAT_VERSION):
                raise V21FullPackError(f"{name}: unexpected codec format")
            source_dtype = row.get("source_safetensors_dtype")
            if source_dtype not in _SOURCE_DTYPES:
                raise V21FullPackError(f"{name}: unsupported source dtype {source_dtype!r}")
            expected_codec_dtype = ("BF16" if source_dtype == "BF16" else
                                    "F16" if source_dtype == "F16" else
                                    f"RAW_{source_dtype}")
            if row.get("dtype") != expected_codec_dtype:
                raise V21FullPackError(f"{name}: codec dtype disagrees with source dtype")
            shape = row.get("shape")
            if (not isinstance(shape, list) or
                    any(type(dim) is not int or dim < 0 for dim in shape)):
                raise V21FullPackError(f"{name}: invalid tensor shape")
            numel = math.prod(shape) if shape else 1
            if numel * _SOURCE_DTYPES[source_dtype][0] != row.get("source_bytes"):
                raise V21FullPackError(f"{name}: source byte count disagrees with shape and dtype")
            source_sha = row.get("source_sha256")
            if (not isinstance(source_sha, str) or len(source_sha) != 64 or
                    any(char not in "0123456789abcdef" for char in source_sha) or
                    row.get("decoded_sha256") != source_sha or header.get("sha256") != source_sha):
                raise V21FullPackError(f"{name}: source hashes disagree")
            if row.get("roundtrip") != "exact":
                raise V21FullPackError(f"{name}: roundtrip is not exact")
            blob = _safe_relative(row.get("serialized_blob"))
            if blob in blob_paths:
                raise V21FullPackError(f"duplicate serialized blob path {blob}")
            blob_paths.add(blob)
            if type(row.get("serialized_blob_bytes")) is not int or row["serialized_blob_bytes"] < 0:
                raise V21FullPackError(f"{name}: invalid serialized blob size")
            digest = row.get("serialized_blob_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise V21FullPackError(f"{name}: invalid serialized blob digest")
            # Store a flattened record for the native dispatch code, retaining the
            # immutable nested header for codec deserialization.
            self.records[name] = {**header, **row, "header": header}

        self.deployment = self._load_deployment()
        self._validate_files_inventory()
        config = self.root / "sidecars" / "config.json"
        self.config_path = str(config.parent) if config.is_file() else None

    def _load_deployment(self):
        deployment_path = self.root / "DEPLOYMENT.json"
        if not deployment_path.exists():
            return None
        deployment_path = _regular_file(self.root, "DEPLOYMENT.json")
        try:
            wrapper = json.loads(deployment_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise V21FullPackError(f"invalid DEPLOYMENT.json: {exc}") from exc
        if not isinstance(wrapper, dict) or wrapper.get("format") != "qwen38-v21-complete-assets-v1":
            raise V21FullPackError("unsupported deployment wrapper")
        if wrapper.get("weights_manifest") != "manifest.json":
            raise V21FullPackError("deployment wrapper references an unexpected weights manifest")
        weights_sha, _ = _sha256(self.root / "manifest.json")
        if wrapper.get("weights_manifest_sha256") != weights_sha:
            raise V21FullPackError("deployment weights manifest sha256 mismatch")
        assets = wrapper.get("metadata_assets")
        if not isinstance(assets, dict) or len(assets) != 13 or wrapper.get("all_13_original_nonweight_assets_present") is not True:
            raise V21FullPackError("deployment wrapper must bind all 13 metadata assets")
        archive_rel = _safe_relative(wrapper.get("metadata_archive"))
        archive_sha = wrapper.get("metadata_archive_sha256")
        additional = wrapper.get("additional_files")
        if not isinstance(additional, list):
            raise V21FullPackError("deployment wrapper is missing additional_files")
        additional_by_path = {}
        for row in additional:
            if not isinstance(row, dict):
                raise V21FullPackError("deployment additional_files row must be an object")
            rel = _safe_relative(row.get("path"))
            if rel in additional_by_path:
                raise V21FullPackError(f"duplicate deployment additional file {rel}")
            if type(row.get("bytes")) is not int or row["bytes"] < 0:
                raise V21FullPackError(f"invalid deployment additional file size for {rel}")
            digest = row.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise V21FullPackError(f"invalid deployment additional file hash for {rel}")
            additional_by_path[rel] = row
        if archive_rel not in additional_by_path or additional_by_path[archive_rel]["sha256"] != archive_sha:
            raise V21FullPackError("metadata archive is not bound by additional_files")
        if set(additional_by_path) != {archive_rel, "unpack_model_assets.py"}:
            raise V21FullPackError("unexpected deployment additional files")
        for rel, row in additional_by_path.items():
            path = _regular_file(self.root, rel)
            digest, size = _sha256(path)
            if size != row["bytes"] or digest != row["sha256"]:
                raise V21FullPackError(f"deployment additional file integrity mismatch: {rel}")
        if wrapper.get("metadata_uncompressed_bytes") != sum(
                row.get("bytes", -1) for row in assets.values() if isinstance(row, dict)):
            raise V21FullPackError("metadata asset byte total mismatch")
        for rel, asset in assets.items():
            rel = _safe_relative(rel)
            if not rel.startswith("sidecars/") or not isinstance(asset, dict):
                raise V21FullPackError(f"invalid restored metadata asset {rel}")
            if type(asset.get("bytes")) is not int or asset["bytes"] < 0:
                raise V21FullPackError(f"invalid metadata asset size for {rel}")
            digest = asset.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise V21FullPackError(f"invalid metadata asset hash for {rel}")
        return {"document": wrapper, "additional_files": additional_by_path, "metadata_assets": assets}

    def _validate_files_inventory(self) -> None:
        rows = self.manifest.get("files")
        if not isinstance(rows, list) or not rows:
            raise V21FullPackError("missing package file inventory")
        listed = {}
        blob_paths = {record["serialized_blob"] for record in self.records.values()}
        for row in rows:
            if not isinstance(row, dict):
                raise V21FullPackError("package inventory row must be an object")
            relative = _safe_relative(row.get("path"))
            if relative in listed:
                raise V21FullPackError(f"duplicate package inventory path {relative}")
            if type(row.get("bytes")) is not int or row["bytes"] < 0:
                raise V21FullPackError(f"invalid inventory size for {relative}")
            path = _regular_file(self.root, relative)
            if path.stat().st_size != row["bytes"]:
                raise V21FullPackError(f"inventory size mismatch for {relative}")
            digest = row.get("sha256")
            if relative == "manifest.json":
                if digest is not None:
                    raise V21FullPackError("manifest self-entry digest must be null")
            elif (not isinstance(digest, str) or len(digest) != 64 or
                  any(char not in "0123456789abcdef" for char in digest)):
                raise V21FullPackError(f"invalid inventory digest for {relative}")
            elif relative not in blob_paths:
                actual_digest, _ = _sha256(path)
                if actual_digest != digest:
                    raise V21FullPackError(f"inventory sha256 mismatch for {relative}")
            listed[relative] = row
        if "manifest.json" not in listed:
            raise V21FullPackError("file inventory omits manifest.json")
        for name, record in self.records.items():
            inventory_row = listed.get(record["serialized_blob"])
            if inventory_row is None:
                raise V21FullPackError(f"{name}: serialized blob is absent from package inventory")
            if (inventory_row["bytes"] != record["serialized_blob_bytes"] or
                    inventory_row["sha256"] != record["serialized_blob_sha256"]):
                raise V21FullPackError(f"{name}: blob inventory disagrees with tensor record")
        actual = set()
        for current, dirs, files in os.walk(self.root, followlinks=False):
            base = Path(current)
            for dirname in list(dirs):
                child = base / dirname
                if child.is_symlink():
                    raise V21FullPackError(f"symlink in artifact tree: {child.relative_to(self.root)}")
            for filename in files:
                relative = (base / filename).relative_to(self.root).as_posix()
                if relative in actual:
                    raise V21FullPackError(f"duplicate file in artifact tree: {relative}")
                actual.add(relative)
                _regular_file(self.root, relative)
        if self.deployment is None:
            expected = set(listed)
        else:
            wrapper = self.deployment["document"]
            assets = self.deployment["metadata_assets"]
            for rel, asset in assets.items():
                path = _regular_file(self.root, rel)
                digest, size = _sha256(path)
                if size != asset["bytes"] or digest != asset["sha256"]:
                    raise V21FullPackError(f"restored metadata asset integrity mismatch: {rel}")
                if rel in listed and (listed[rel]["bytes"] != size or listed[rel]["sha256"] != digest):
                    raise V21FullPackError(f"restored asset conflicts with inner inventory: {rel}")
            core_bytes = sum(row["bytes"] for rel, row in listed.items() if not rel.startswith("sidecars/"))
            if wrapper.get("core_artifact_bytes") != core_bytes:
                raise V21FullPackError("deployment core artifact byte count mismatch")
            packed_bytes = core_bytes + sum(row["bytes"] for row in self.deployment["additional_files"].values())
            packed_bytes += _regular_file(self.root, "DEPLOYMENT.json").stat().st_size
            if wrapper.get("complete_packed_bytes") != packed_bytes:
                raise V21FullPackError("deployment complete package byte count mismatch")
            expected = set(listed) | set(self.deployment["additional_files"]) | {"DEPLOYMENT.json"} | set(assets)
        if actual != expected:
            missing = sorted(actual - expected)
            absent = sorted(expected - actual)
            raise V21FullPackError(f"package file inventory mismatch; unlisted={missing[:3]}, missing={absent[:3]}")

    def names(self) -> List[str]:
        return [row["name"] for row in self.manifest["tensors"]]

    def items(self) -> Iterator[dict]:
        yield from (self.records[row["name"]] for row in self.manifest["tensors"])

    def stored_record(self, name: str) -> dict:
        return self.records[name]

    def is_alias(self, name: str) -> bool:
        if name not in self.records:
            raise KeyError(name)
        return False

    def has_config(self) -> bool:
        return self.config_path is not None

    def read(self, name: str, *, check_crc: bool = True):
        from safetensors.numpy import load

        record = self.stored_record(name)
        path = _regular_file(self.root, record["serialized_blob"])
        digest, size = _sha256(path)
        if size != record["serialized_blob_bytes"]:
            raise V21FullPackError(f"{name}: serialized blob size mismatch")
        if digest != record["serialized_blob_sha256"]:
            raise V21FullPackError(f"{name}: serialized blob sha256 mismatch")
        try:
            arrays = load(path.read_bytes())
            tensor = self.codec.from_named_arrays(record["header"], arrays)
            # Keep the v2 reader-compatible parameter, but never expose a path
            # that returns unverified coded streams.
            self.codec.verify_crc(tensor)
        except Exception as exc:
            raise V21FullPackError(f"{name}: serialized tensor is invalid: {exc}") from exc
        return tensor

    def decode_host(self, name: str) -> np.ndarray:
        record = self.stored_record(name)
        tensor = self.read(name, check_crc=True)
        try:
            decoded = self.codec.decode_tensor(tensor, check_crc=True, check_sha=True)
        except Exception as exc:
            raise V21FullPackError(f"{name}: source decode failed: {exc}") from exc
        raw = np.ascontiguousarray(decoded).view(np.uint8).reshape(-1).tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != record["source_sha256"] or digest != record["decoded_sha256"]:
            raise V21FullPackError(f"{name}: decoded source sha256 mismatch")
        source_dtype = record["source_safetensors_dtype"]
        itemsize, numpy_dtype = _SOURCE_DTYPES[source_dtype]
        if len(raw) != record["source_bytes"] or len(raw) % itemsize:
            raise V21FullPackError(f"{name}: decoded source byte count mismatch")
        if record["dtype"].startswith("RAW_"):
            decoded = np.frombuffer(raw, dtype=numpy_dtype).reshape(record["shape"])
        elif list(decoded.shape) != record["shape"]:
            raise V21FullPackError(f"{name}: decoded source shape mismatch")
        return decoded


__all__ = ["V21FullPackArtifact", "V21FullPackError"]
