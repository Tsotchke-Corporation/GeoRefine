"""Strict, read-only resolver for an original Safetensors model directory."""
from __future__ import annotations

import copy
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import threading

from .typed_context_package import _assert_no_symlinks, _inside, _parse_source, _read_tensor, _safe_rel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _filesystem_pin(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _json_config(path: Path) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid original model config") from exc
    if not isinstance(value, dict):
        raise ValueError("original model config must be an object")
    return raw, value


class SafetensorsResolver:
    """Lazy resolver with lifecycle checks over an original source directory.

    The index and every referenced shard are parsed by the typed package's
    strict Safetensors parser.  The original files are never copied or changed.
    """

    def __init__(self, package: Path, *, verify_package: bool = True,
                 max_source_bytes: int = 256 * 1024 * 1024 * 1024, **_ignored):
        del verify_package
        self.package = Path(package)
        if self.package.is_symlink() or not self.package.is_dir():
            raise ValueError("original Safetensors root must be a regular directory")
        self.package = self.package.resolve()
        _assert_no_symlinks(self.package)
        self.index_path = _inside(self.package, _safe_rel("model.safetensors.index.json", label="index"), label="index")
        self.config_path = _inside(self.package, _safe_rel("config.json", label="config"), label="config")
        descriptors, index_raw, _headers, source_info = _parse_source(
            self.package, PurePosixPath("model.safetensors.index.json"), None)
        self._max_source_bytes = max_source_bytes
        self.index_sha256 = hashlib.sha256(index_raw).hexdigest()
        self.manifest_sha256 = self.index_sha256
        config_raw, config = _json_config(self.config_path)
        self.config_sha256 = hashlib.sha256(config_raw).hexdigest()
        self.config = copy.deepcopy(config)
        self._source_info = copy.deepcopy(source_info)
        # The shared MLX loader uses this field to reject candidate frames in
        # the typed/predictive formats.  Original Safetensors has no candidates.
        self.manifest = {"selected_candidates": {}}
        self._entries = {}
        for name, row in descriptors.items():
            value = dict(row)
            value["source_sha256"] = None
            self._entries[name] = value
        self._tensor_sha256, self._filesystem_pins = self._stream_source_pins()
        for name, digest in self._tensor_sha256.items():
            self._entries[name]["source_sha256"] = digest
        self._condition = threading.Condition()
        self._active_readers = 0
        self._closing = False
        self._closed = False

    def _ensure_open(self):
        if self._closed:
            raise ValueError("Safetensors resolver is closed")

    def _stream_source_pins(self) -> tuple[dict[str, str], dict[str, tuple[int, int, int, int, int]]]:
        """Hash each shard once, deriving whole-shard and tensor pins together."""
        tensor_pins = {}
        filesystem_pins = {
            str(self.index_path): _filesystem_pin(self.index_path),
            str(self.config_path): _filesystem_pin(self.config_path),
        }
        for shard, pin in self._source_info["shards"].items():
            path = _inside(self.package, _safe_rel(shard, label="source shard"), label="source shard")
            filesystem_pins[str(path)] = _filesystem_pin(path)
            rows = sorted((row for row in self._entries.values() if row["source_shard"] == shard),
                          key=lambda row: row["offsets"][0])
            hashers = [(row, hashlib.sha256()) for row in rows]
            whole = hashlib.sha256()
            header_size = self._header_size(shard)
            with path.open("rb") as stream:
                absolute = 0
                while True:
                    block = stream.read(8 * 1024 * 1024)
                    if not block:
                        break
                    whole.update(block)
                    block_end = absolute + len(block)
                    for row, hasher in hashers:
                        start = 8 + header_size + row["offsets"][0]
                        end = 8 + header_size + row["offsets"][1]
                        lo = max(start, absolute)
                        hi = min(end, block_end)
                        if lo < hi:
                            hasher.update(block[lo - absolute:hi - absolute])
                    absolute = block_end
            if whole.hexdigest() != pin["sha256"]:
                raise ValueError(f"original Safetensors shard changed: {shard}")
            tensor_pins.update({row["name"]: hasher.hexdigest() for row, hasher in hashers})
        for path_name, pin in filesystem_pins.items():
            if _filesystem_pin(Path(path_name)) != pin:
                raise ValueError(f"original Safetensors file changed during hashing: {Path(path_name).name}")
        return tensor_pins, filesystem_pins

    def _header_size(self, shard: str) -> int:
        path = _inside(self.package, _safe_rel(shard, label="source shard"), label="source shard")
        with path.open("rb") as stream:
            prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated source shard: {shard}")
        expected = self._source_info["shards"][shard]["header_bytes"] - 8
        actual = int.from_bytes(prefix, "little")
        if actual != expected:
            raise ValueError(f"original Safetensors header changed: {shard}")
        return expected

    def _assert_pins(self):
        _assert_no_symlinks(self.package)
        paths = [self.index_path, self.config_path]
        paths.extend(_inside(self.package, _safe_rel(shard, label="source shard"), label="source shard")
                     for shard in self._source_info["shards"])
        for path in paths:
            key = str(path)
            if _filesystem_pin(path) != self._filesystem_pins[key]:
                raise ValueError(f"original Safetensors file changed: {path.name}")

    @contextmanager
    def _reader(self):
        with self._condition:
            if self._closed or self._closing:
                raise ValueError("Safetensors resolver is closed")
            self._active_readers += 1
        try:
            self._assert_pins()
            yield
            self._assert_pins()
        finally:
            with self._condition:
                self._active_readers -= 1
                self._condition.notify_all()

    @property
    def tensor_names(self):
        self._ensure_open()
        return tuple(sorted(self._entries))

    def entry(self, name: str) -> dict:
        with self._reader():
            try:
                row = self._entries[name]
                if row["actual_source_bytes"] > self._max_source_bytes:
                    raise MemoryError("source tensor exceeds source working budget")
                raw = _read_tensor(self.package, row)
                digest = hashlib.sha256(raw).hexdigest()
                if digest != self._tensor_sha256[name]:
                    raise ValueError(f"original Safetensors tensor changed: {name}")
                row["source_sha256"] = self._tensor_sha256[name]
                return copy.deepcopy(row)
            except KeyError as exc:
                raise KeyError(f"unknown tensor: {name}") from exc

    resolve = entry

    def decode_bytes(self, name: str) -> bytes:
        with self._reader():
            row = self._entries[name]
            if row["actual_source_bytes"] > self._max_source_bytes:
                raise MemoryError("source tensor exceeds source working budget")
            raw = _read_tensor(self.package, row)
            digest = hashlib.sha256(raw).hexdigest()
            if digest != self._tensor_sha256[name]:
                raise ValueError(f"source tensor SHA mismatch: {name}")
            row["source_sha256"] = self._tensor_sha256[name]
            return raw

    def numpy_view(self, name: str):
        import numpy as np
        row = self.entry(name)
        raw = self.decode_bytes(name)
        dtypes = {"F32": "<f4", "F64": "<f8", "F16": "<f2", "BF16": "<u2",
                  "I8": "i1", "U8": "u1", "I16": "<i2", "U16": "<u2",
                  "I32": "<i4", "U32": "<u4", "I64": "<i8", "U64": "<u8", "BOOL": "?"}
        dtype = dtypes.get(row["dtype"])
        if dtype is None:
            raise ValueError(f"no native NumPy view for storage dtype {row['dtype']!r}")
        return np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(tuple(row["shape"]))

    native_numpy = numpy_view

    def pinned_config(self) -> dict:
        """Return the original JSON config after rechecking its pinned bytes."""
        with self._reader():
            raw, value = _json_config(self.config_path)
            if hashlib.sha256(raw).hexdigest() != self.config_sha256:
                raise ValueError("original model config changed")
            return copy.deepcopy(value)

    read_config = pinned_config

    def materialize_torch(self, name: str, *, device: str | None = None):
        """Materialize exact source bytes through Torch's native storage dtype."""
        import torch

        self._ensure_open()
        row = self._entries[name]
        actual = row["actual_source_bytes"]
        if actual > self._max_source_bytes:
            raise MemoryError("source tensor exceeds source working budget")
        raw = self.decode_bytes(name)
        try:
            from safetensors.torch import _getdtype
            native_dtype = _getdtype(row["dtype"])
        except (ImportError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"unsupported Torch storage dtype {row['dtype']!r}") from exc
        if actual != len(raw):
            raise ValueError(f"source tensor byte count mismatch: {name}")
        element_size = torch.empty((), dtype=native_dtype).element_size()
        if actual % element_size != 0:
            raise ValueError(f"source tensor byte alignment mismatch: {name}")
        # The uint8 view preserves every bit, including signed zero and NaN
        # payloads, before changing only the Torch dtype metadata.
        if actual == 0:
            tensor = torch.empty(tuple(row["shape"]), dtype=native_dtype)
        else:
            storage = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            tensor = storage.view(native_dtype).reshape(tuple(row["shape"]))
        if tensor.numel() * tensor.element_size() != actual:
            raise ValueError(f"source tensor shape/dtype byte mismatch: {name}")
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    to_torch = materialize_torch

    def _verify_integrity_unlocked(self):
        self._assert_pins()
        descriptors, index_raw, _headers, source_info = _parse_source(
            self.package, PurePosixPath("model.safetensors.index.json"), None)
        config_raw, _config = _json_config(self.config_path)
        if hashlib.sha256(config_raw).hexdigest() != self.config_sha256:
            raise ValueError("original model config changed")
        if hashlib.sha256(index_raw).hexdigest() != self.index_sha256 or source_info != self._source_info:
            raise ValueError("original Safetensors source inventory changed")
        if set(descriptors) != set(self._entries):
            raise ValueError("original Safetensors tensor coverage changed")
        tensor_pins, filesystem_pins = self._stream_source_pins()
        if tensor_pins != self._tensor_sha256 or filesystem_pins != self._filesystem_pins:
            raise ValueError("original Safetensors tensor identity changed")
        _json_config(self.config_path)
        return {"tensors": len(self._entries), "index_sha256": self.index_sha256}

    def verify_integrity(self):
        with self._reader():
            return self._verify_integrity_unlocked()

    def close(self):
        with self._condition:
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

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
