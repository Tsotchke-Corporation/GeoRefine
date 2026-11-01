"""Bit-preserving storage for arbitrary tensor/data payloads.

This module deliberately does not interpret a dtype descriptor.  A descriptor
may be a known numeric dtype, a structured description, or a future/packed
token.  The payload bytes are the source of truth and are restored exactly.
The frame is a small, versioned container suitable for data that is not yet
supported by an accelerated operator.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
import struct
import zlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterator, Optional, Sequence, Union


MAGIC = b"GTFC"
VERSION = 1

# The limits apply to the envelope itself and to decoded body bytes.  Callers
# can use a smaller max_decoded_bytes budget for an individual frame.
DEFAULT_MAX_DECODED_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_SOURCE_BYTES = DEFAULT_MAX_DECODED_BYTES
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_ENVELOPE_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_FRAME_BYTES = MAX_ENVELOPE_BYTES
MAX_DIMS = 1024
MAX_DIMENSION = (1 << 63) - 1
MAX_JSON_DEPTH = 128

_HEADER = struct.Struct(">4sBBHQQ")
_TRAILER = struct.Struct(">32s32s")
_FLAGS = 0
_RESERVED = 0
JSONScalar = Union[None, bool, int, float, str]
JSONValue = Union[JSONScalar, list["JSONValue"], dict[str, "JSONValue"]]


class TensorCodecError(ValueError):
    """Raised for malformed frames, invalid metadata, or violated limits."""


class _FrozenList(tuple):
    """Tuple-backed JSON list with value equality against normal lists."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (list, tuple)):
            return tuple(self) == tuple(other)
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        equal = self.__eq__(other)
        return NotImplemented if equal is NotImplemented else not equal


@dataclass(frozen=True, slots=True)
class DecodedTensor:
    """Immutable result of :func:`decode_tensor`.

    ``payload`` is always an immutable ``bytes`` object.  JSON descriptor and
    layout containers are recursively frozen, while ``shape`` is a tuple.
    Iteration provides the convenient ``payload, dtype, shape, layout`` form.
    """

    payload: bytes
    dtype: Any
    shape: tuple[int, ...]
    layout: Any = None
    element_bits: Optional[int] = None

    @property
    def data(self) -> bytes:
        """Alias for payload for callers that use data-oriented terminology."""

        return self.payload

    @property
    def bytes(self) -> bytes:
        """Alias for payload."""

        return self.payload

    @property
    def descriptor(self) -> Any:
        """Alias for the original dtype descriptor."""

        return self.dtype

    @property
    def dtype_descriptor(self) -> Any:
        """Descriptive alias for callers that distinguish dtype metadata."""

        return self.dtype

    def __iter__(self) -> Iterator[Any]:
        yield self.payload
        yield self.dtype
        yield self.shape
        yield self.layout


def _duplicate_key_error(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TensorCodecError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise TensorCodecError(f"non-finite JSON number: {value}")


def _thaw_json(value: Any, *, _active: Optional[set[int]] = None, _depth: int = 0) -> Any:
    """Turn our immutable decoded JSON containers back into JSON values."""

    if not isinstance(value, (Mapping, list, tuple)):
        return value
    if _depth >= MAX_JSON_DEPTH:
        raise TensorCodecError(f"JSON metadata exceeds depth {MAX_JSON_DEPTH}")
    if _active is None:
        _active = set()
    identity = id(value)
    if identity in _active:
        raise TensorCodecError("cyclic JSON metadata")
    _active.add(identity)
    try:
        if isinstance(value, Mapping):
            return {key: _thaw_json(item, _active=_active, _depth=_depth + 1) for key, item in value.items()}
        return [_thaw_json(item, _active=_active, _depth=_depth + 1) for item in value]
    finally:
        _active.remove(identity)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(_freeze_json(item) for item in value)
    return value


def _validate_json_value(
    value: Any,
    *,
    path: str = "value",
    _active: Optional[set[int]] = None,
    _depth: int = 0,
) -> None:
    """Validate a JSON-safe value without silently coercing its type."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise TensorCodecError(f"non-finite JSON value at {path}")
        return
    if isinstance(value, (Mapping, list, tuple)):
        if _depth >= MAX_JSON_DEPTH:
            raise TensorCodecError(f"JSON metadata exceeds depth {MAX_JSON_DEPTH}")
        if _active is None:
            _active = set()
        identity = id(value)
        if identity in _active:
            raise TensorCodecError(f"cyclic JSON metadata at {path}")
        _active.add(identity)
        try:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise TensorCodecError(f"JSON object key at {path} is not a string")
                    _validate_json_value(item, path=f"{path}.{key}", _active=_active, _depth=_depth + 1)
            else:
                for index, item in enumerate(value):
                    _validate_json_value(item, path=f"{path}[{index}]", _active=_active, _depth=_depth + 1)
        finally:
            _active.remove(identity)
        return
    raise TensorCodecError(f"value at {path} is not JSON-safe: {type(value).__name__}")


def _canonical_json(value: Any) -> bytes:
    value = _thaw_json(value)
    _validate_json_value(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, UnicodeEncodeError, ValueError, RecursionError) as exc:
        raise TensorCodecError("value cannot be encoded as canonical JSON") from exc


def _parse_metadata(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_key_error,
            parse_constant=_reject_constant,
        )
    except TensorCodecError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise TensorCodecError("invalid metadata JSON") from exc
    if not isinstance(value, dict):
        raise TensorCodecError("metadata must be a JSON object")
    if _canonical_json(value) != raw:
        raise TensorCodecError("metadata is not canonical JSON")
    return value


def _shape_and_numel(shape: Sequence[int]) -> tuple[tuple[int, ...], int]:
    if isinstance(shape, (str, bytes, bytearray)):
        raise TensorCodecError("shape must be a sequence of dimensions")
    try:
        dimensions = tuple(shape)
    except TypeError as exc:
        raise TensorCodecError("shape must be a sequence of dimensions") from exc
    if len(dimensions) > MAX_DIMS:
        raise TensorCodecError(f"shape has too many dimensions (maximum {MAX_DIMS})")
    numel = 1
    for index, dimension in enumerate(dimensions):
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TensorCodecError(f"shape dimension {index} is not an integer")
        if dimension < 0 or dimension > MAX_DIMENSION:
            raise TensorCodecError(f"shape dimension {index} is out of range")
        if dimension == 0:
            numel = 0
        elif numel:
            numel *= dimension
            if numel > MAX_DIMENSION:
                raise TensorCodecError("shape element count is too large")
    return dimensions, numel


def _validate_element_bits(element_bits: Optional[int], numel: int, payload_len: int) -> Optional[int]:
    if element_bits is None:
        return None
    if isinstance(element_bits, bool) or not isinstance(element_bits, int) or element_bits <= 0:
        raise TensorCodecError("element_bits must be a positive integer")
    # Avoid imposing a numeric dtype whitelist: the declared bit width only
    # supplies an optional physical-size consistency check.
    expected = (numel * element_bits + 7) // 8
    if expected != payload_len:
        raise TensorCodecError(
            f"payload length {payload_len} does not match ceil(numel*element_bits/8)={expected}"
        )
    return element_bits


def _as_payload(
    payload: Union[bytes, bytearray, memoryview],
    *,
    max_bytes: Optional[int] = None,
    label: str = "payload",
) -> bytes:
    try:
        view = memoryview(payload)
    except (TypeError, ValueError) as exc:
        raise TypeError("payload must be bytes-like") from exc
    if max_bytes is not None and view.nbytes > max_bytes:
        raise TensorCodecError(f"{label} exceeds its byte budget")
    try:
        return view.tobytes()
    except (TypeError, ValueError) as exc:
        raise TypeError("payload must be bytes-like") from exc


def _resolve_limit(max_decoded_bytes: Optional[int]) -> int:
    if max_decoded_bytes is None:
        return DEFAULT_MAX_DECODED_BYTES
    if isinstance(max_decoded_bytes, bool) or not isinstance(max_decoded_bytes, int):
        raise TypeError("max_decoded_bytes must be an integer")
    if max_decoded_bytes < 0:
        raise ValueError("max_decoded_bytes must be non-negative")
    return max_decoded_bytes


def _resolve_budget(value: Optional[int], default: int, name: str) -> int:
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def encode_tensor(
    payload: Union[bytes, bytearray, memoryview],
    dtype: JSONValue,
    shape: Sequence[int] = (),
    *,
    layout: Optional[JSONValue] = None,
    element_bits: Optional[int] = None,
    max_source_bytes: Optional[int] = None,
) -> bytes:
    """Encode arbitrary source bytes with lossless JSON-safe metadata.

    ``dtype`` is intentionally opaque.  ``element_bits`` is optional; when
    supplied it checks only the byte length implied by the shape and bit width
    and does not inspect or modify padding bits.
    """

    source_budget = _resolve_budget(max_source_bytes, DEFAULT_MAX_SOURCE_BYTES, "max_source_bytes")
    source = _as_payload(payload, max_bytes=source_budget, label="source payload")
    shape_tuple, numel = _shape_and_numel(shape)
    bits = _validate_element_bits(element_bits, numel, len(source))

    compressed = zlib.compress(source, level=9)
    if len(compressed) < len(source):
        body_codec = "deflate"
        body = compressed
    else:
        body_codec = "raw"
        body = source

    metadata = {
        "body_codec": body_codec,
        "body_bytes": len(body),
        "dtype": dtype,
        "element_bits": bits,
        "layout": layout,
        "shape": list(shape_tuple),
        "source_bytes": len(source),
        "source_sha256": hashlib.sha256(source).hexdigest(),
    }
    metadata_bytes = _canonical_json(metadata)
    if len(metadata_bytes) > MAX_METADATA_BYTES:
        raise TensorCodecError("metadata exceeds the envelope limit")
    if len(body) > DEFAULT_MAX_FRAME_BYTES:
        raise TensorCodecError("body exceeds the default envelope limit")
    header = _HEADER.pack(
        MAGIC,
        VERSION,
        _FLAGS,
        _RESERVED,
        len(metadata_bytes),
        len(body),
    )
    trailer = _TRAILER.pack(hashlib.sha256(metadata_bytes).digest(), hashlib.sha256(body).digest())
    frame = header + metadata_bytes + body + trailer
    if len(frame) > DEFAULT_MAX_FRAME_BYTES:
        raise TensorCodecError("envelope exceeds the default limit")
    return frame


def _bounded_deflate(body: bytes, expected: int, budget: int) -> bytes:
    if expected > budget:
        raise TensorCodecError("decoded payload exceeds max_decoded_bytes")
    decoder = zlib.decompressobj()
    try:
        output = decoder.decompress(body, expected + 1)
        if len(output) > expected or decoder.unconsumed_tail:
            raise TensorCodecError("deflate payload exceeds declared source length")
        # Bound flush too: a stream can defer output until flush, and an
        # unbounded flush would undermine the explicit decoded-byte budget.
        output += decoder.flush(expected + 1 - len(output))
    except zlib.error as exc:
        raise TensorCodecError("invalid deflate body") from exc
    if not decoder.eof or decoder.unused_data:
        raise TensorCodecError("deflate body is truncated or has trailing bytes")
    if len(output) != expected:
        raise TensorCodecError("decoded payload length does not match metadata")
    return output


def decode_tensor(
    frame: Union[bytes, bytearray, memoryview],
    *,
    max_decoded_bytes: Optional[int] = None,
    max_frame_bytes: Optional[int] = None,
) -> DecodedTensor:
    """Decode and validate a generic tensor frame.

    Parsing is strict: the envelope must be complete with no trailing bytes,
    metadata must be canonical JSON with unique keys, and all source/body/
    metadata hashes and declared lengths must agree.
    """

    frame_budget = _resolve_budget(max_frame_bytes, DEFAULT_MAX_FRAME_BYTES, "max_frame_bytes")
    raw = _as_payload(frame, max_bytes=frame_budget, label="frame")
    budget = _resolve_limit(max_decoded_bytes)
    if len(raw) < _HEADER.size + _TRAILER.size:
        raise TensorCodecError("truncated envelope")
    magic, version, flags, reserved, metadata_len, body_len = _HEADER.unpack_from(raw)
    if magic != MAGIC:
        raise TensorCodecError("invalid envelope magic")
    if version != VERSION:
        raise TensorCodecError(f"unsupported envelope version: {version}")
    if flags != _FLAGS or reserved != _RESERVED:
        raise TensorCodecError("unsupported envelope flags or reserved bits")
    if metadata_len > MAX_METADATA_BYTES:
        raise TensorCodecError("metadata exceeds the limit")
    if body_len > frame_budget:
        raise TensorCodecError("body exceeds the frame budget")
    expected_total = _HEADER.size + metadata_len + body_len + _TRAILER.size
    if expected_total > frame_budget:
        raise TensorCodecError("envelope exceeds the frame budget")
    if len(raw) < expected_total:
        raise TensorCodecError("truncated envelope")
    if len(raw) > expected_total:
        raise TensorCodecError("trailing bytes after envelope")

    metadata_start = _HEADER.size
    metadata_end = metadata_start + metadata_len
    body_end = metadata_end + body_len
    metadata_bytes = raw[metadata_start:metadata_end]
    body = raw[metadata_end:body_end]
    metadata_sha, body_sha = _TRAILER.unpack_from(raw, body_end)
    if hashlib.sha256(metadata_bytes).digest() != metadata_sha:
        raise TensorCodecError("metadata integrity check failed")
    if hashlib.sha256(body).digest() != body_sha:
        raise TensorCodecError("body integrity check failed")
    metadata = _parse_metadata(metadata_bytes)

    required = {
        "body_codec",
        "body_bytes",
        "dtype",
        "element_bits",
        "layout",
        "shape",
        "source_bytes",
        "source_sha256",
    }
    if set(metadata) != required:
        missing = sorted(required - set(metadata))
        extra = sorted(set(metadata) - required)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"unknown={extra}")
        raise TensorCodecError("invalid metadata fields (" + ", ".join(details) + ")")
    body_bytes = metadata["body_bytes"]
    if isinstance(body_bytes, bool) or not isinstance(body_bytes, int) or body_bytes != body_len:
        raise TensorCodecError("body length does not match metadata")
    if not isinstance(metadata["body_codec"], str) or metadata["body_codec"] not in {"raw", "deflate"}:
        raise TensorCodecError("unsupported body codec")
    source_len = metadata["source_bytes"]
    if isinstance(source_len, bool) or not isinstance(source_len, int) or source_len < 0:
        raise TensorCodecError("invalid source byte length")
    if source_len > budget:
        raise TensorCodecError("decoded payload exceeds max_decoded_bytes")
    source_sha = metadata["source_sha256"]
    if not isinstance(source_sha, str) or len(source_sha) != 64 or source_sha != source_sha.lower():
        raise TensorCodecError("invalid source SHA-256")
    try:
        int(source_sha, 16)
    except ValueError as exc:
        raise TensorCodecError("invalid source SHA-256") from exc

    if metadata["body_codec"] == "raw":
        if len(body) != source_len:
            raise TensorCodecError("raw body length does not match source length")
        source = body
    else:
        source = _bounded_deflate(body, source_len, budget)
    if hashlib.sha256(source).hexdigest() != source_sha:
        raise TensorCodecError("source integrity check failed")

    shape_value = metadata["shape"]
    if not isinstance(shape_value, list):
        raise TensorCodecError("shape must be a JSON array")
    shape, numel = _shape_and_numel(shape_value)
    bits_value = metadata["element_bits"]
    bits = _validate_element_bits(bits_value, numel, len(source))
    return DecodedTensor(
        payload=bytes(source),
        dtype=_freeze_json(metadata["dtype"]),
        shape=shape,
        layout=_freeze_json(metadata["layout"]),
        element_bits=bits,
    )


# Explicit aliases make the frame API discoverable without introducing a
# second implementation or implying a numeric tensor conversion.
encode_frame = encode_tensor
decode_frame = decode_tensor
encode_tensor_frame = encode_tensor
decode_tensor_frame = decode_tensor
encode = encode_tensor
decode = decode_tensor


__all__ = [
    "DEFAULT_MAX_DECODED_BYTES",
    "DEFAULT_MAX_FRAME_BYTES",
    "DEFAULT_MAX_SOURCE_BYTES",
    "DecodedTensor",
    "MAX_ENVELOPE_BYTES",
    "MAX_JSON_DEPTH",
    "MAX_METADATA_BYTES",
    "MAGIC",
    "TensorCodecError",
    "VERSION",
    "decode",
    "decode_frame",
    "decode_tensor",
    "decode_tensor_frame",
    "encode",
    "encode_frame",
    "encode_tensor",
    "encode_tensor_frame",
]
