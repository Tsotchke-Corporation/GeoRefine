"""Opt-in adapter from a verified typed Context package to the TBE21 reader API."""
from __future__ import annotations

import hashlib
import math
from types import MappingProxyType

import numpy as np

from georefine_context_runtime.typed_context_package import (
    TypedContextPackageError,
    TypedContextResolver,
)
from glc_loader import codec_v21 as cv21


class TypedContextV21Error(ValueError):
    """A typed Context package could not be represented or verified as TBE21."""


_HOST_DTYPES = {
    "F32": "<f4", "F64": "<f8", "I8": "i1", "U8": "u1",
    "I16": "<i2", "U16": "<u2", "I32": "<i4", "U32": "<u4",
    "I64": "<i8", "U64": "<u8", "BOOL": "?",
}


class TypedContextV21Artifact:
    """The V21 serving artifact surface backed by a typed-context resolver.

    Package identity and all packaged files are verified before this reader is
    exposed. Tensor payloads are decoded and source-checked before encoding;
    BF16 is eligible for the normal TBE21 codec and every other dtype stays raw.
    """

    def __init__(self, package_path, *, expected_manifest_sha256: str,
                 resolver: TypedContextResolver | None = None):
        if (not isinstance(expected_manifest_sha256, str) or len(expected_manifest_sha256) != 64
                or any(c not in "0123456789abcdef" for c in expected_manifest_sha256)):
            raise TypedContextV21Error("expected package manifest SHA-256 is required")
        self._owns_resolver = resolver is None
        try:
            self.resolver = resolver or TypedContextResolver(package_path, verify_package=True)
            if self.resolver.manifest_sha256 != expected_manifest_sha256:
                raise TypedContextV21Error("typed Context package manifest SHA-256 mismatch")
        except (TypedContextPackageError, OSError) as exc:
            if self._owns_resolver and getattr(self, "resolver", None) is not None:
                self.resolver.close()
            raise TypedContextV21Error(f"typed Context package verification failed: {exc}") from exc
        except BaseException:
            if self._owns_resolver and getattr(self, "resolver", None) is not None:
                self.resolver.close()
            raise
        self.path = str(self.resolver.package)
        self.manifest = self.resolver.manifest
        self.manifest_sha256 = self.resolver.manifest_sha256
        self.format = cv21.FORMAT
        self.codec = cv21
        # Typed package assets are frames, not files at stable filesystem
        # paths. Callers that need a config must use read_asset (or an explicit
        # parent directory); has_config only advertises an actual config_path.
        self.config_path = None
        self._names = list(self.resolver.tensor_names)
        records = {}
        for name in self._names:
            record = self.resolver.entry(name)
            record["source_bytes"] = record["actual_source_bytes"]
            records[name] = MappingProxyType(record)
        self.records = MappingProxyType(records)
        self._records = self.records
        self._cached_name = None
        self._cached_tensor = None
        self._closed = False

    def _ensure_open(self):
        if self._closed:
            raise TypedContextV21Error("typed Context TBE21 artifact is closed")

    def names(self):
        self._ensure_open()
        return list(self._names)

    def stored_record(self, name):
        self._ensure_open()
        return self._records[name]

    def is_alias(self, name):
        self.stored_record(name)
        return False

    def has_config(self):
        self._ensure_open()
        return self.config_path is not None and self.config_path.is_file()

    def release(self, name=None):
        self._ensure_open()
        if name is None or self._cached_name == name:
            self._cached_name = None
            self._cached_tensor = None

    def close(self):
        """Retire the adapter, verifying but never closing a borrowed resolver."""
        if self._closed:
            return
        if self._owns_resolver:
            self.resolver.close()
        else:
            self.resolver.verify_integrity()
        self._closed = True

    def read_asset(self, source_path):
        """Read an exact verified asset without materializing it on disk."""
        self._ensure_open()
        try:
            return self.resolver.read_asset(source_path)
        except (TypedContextPackageError, KeyError) as exc:
            raise TypedContextV21Error(f"typed asset read failed for {source_path!r}: {exc}") from exc

    def decode_source_bytes(self, name):
        """Return exact source bytes, including storage dtypes NumPy cannot view."""
        self._ensure_open()
        row = self.stored_record(name)
        try:
            raw = self.resolver.decode_bytes(name)
        except (TypedContextPackageError, KeyError) as exc:
            raise TypedContextV21Error(f"{name}: typed source decode failed: {exc}") from exc
        if (len(raw) != row["actual_source_bytes"]
                or hashlib.sha256(raw).hexdigest() != row["source_sha256"]):
            raise TypedContextV21Error(f"{name}: typed source byte identity mismatch")
        return raw

    def read(self, name, *, check_crc=True):
        self._ensure_open()
        row = self.stored_record(name)
        raw = self.decode_source_bytes(name)
        shape = tuple(row["shape"])
        dtype = row["dtype"]
        if math.prod(shape) * (2 if dtype in ("BF16", "F16") else 0) != len(raw) and dtype in ("BF16", "F16"):
            raise TypedContextV21Error(f"{name}: source byte count disagrees with shape")
        if dtype == "BF16":
            source = np.frombuffer(raw, dtype="<u2").reshape(shape)
            tensor = cv21.encode_tensor(name, source, dtype="BF16")
        else:
            # V21 raw streams are byte-exact. Construct the declared geometry
            # explicitly so future/opaque dtype tokens retain their true shape.
            geom = cv21.geometry_for(shape, dtype)
            if geom.mode != cv21.MODE_RAW:
                raise TypedContextV21Error(f"{name}: non-BF16 dtype unexpectedly eligible for coded mode")
            tensor = cv21.V21Tensor(name, dtype, geom, hashlib.sha256(raw).hexdigest(),
                                    raw=np.frombuffer(raw, dtype=np.uint8).copy())
            tensor.crc = {"raw": cv21._crc(tensor.raw)}
        if tensor.sha256 != row["source_sha256"]:
            raise TypedContextV21Error(f"{name}: TBE21 source identity changed during encode")
        decoded = cv21.decode_tensor(tensor, check_crc=True, check_sha=True)
        decoded_bytes = np.ascontiguousarray(decoded).view(np.uint8).tobytes()
        if len(decoded_bytes) != len(raw) or hashlib.sha256(decoded_bytes).hexdigest() != row["source_sha256"]:
            raise TypedContextV21Error(f"{name}: TBE21 raw decode source SHA-256 mismatch")
        self._cached_name, self._cached_tensor = name, tensor
        return tensor

    def decode_host(self, name):
        """Decode to the loader's native NumPy representation where supported."""
        self._ensure_open()
        row = self.stored_record(name)
        raw = self.decode_source_bytes(name)
        dtype = row["dtype"]
        if not isinstance(dtype, str):
            raise TypedContextV21Error(f"{name}: unsupported numeric host dtype {dtype!r}; use decode_source_bytes")
        if dtype in ("BF16", "F16"):
            native = np.dtype("<u2")
        elif dtype in _HOST_DTYPES:
            native = np.dtype(_HOST_DTYPES[dtype])
        else:
            raise TypedContextV21Error(f"{name}: unsupported numeric host dtype {dtype!r}; use decode_source_bytes")
        expected = math.prod(row["shape"]) * native.itemsize
        if len(raw) != expected:
            raise TypedContextV21Error(f"{name}: source byte count disagrees with dtype and shape")
        result = np.frombuffer(raw, dtype=native).reshape(tuple(row["shape"]))
        if hashlib.sha256(result.tobytes()).hexdigest() != row["source_sha256"]:
            raise TypedContextV21Error(f"{name}: host source SHA-256 mismatch")
        return result


def typed_context_artifact_factory(*, expected_manifest_sha256: str):
    """Return the ``factory(package_path)`` form used by the serving source bridge."""
    def build(package_path):
        return TypedContextV21Artifact(package_path,
                                       expected_manifest_sha256=expected_manifest_sha256)
    return build


__all__ = ["TypedContextV21Artifact", "TypedContextV21Error", "typed_context_artifact_factory"]
