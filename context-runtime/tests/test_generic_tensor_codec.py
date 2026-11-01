"""Focused coverage for the generic bit-preserving tensor frame."""

from __future__ import annotations

import hashlib
import json
import struct
import zlib

import pytest

from georefine_context_runtime import generic_tensor_codec as codec


def round_trip(payload, dtype, shape, **kwargs):
    result = codec.decode_tensor(codec.encode_tensor(payload, dtype, shape, **kwargs))
    assert result.payload == bytes(payload)
    assert result.shape == tuple(shape)
    return result


def test_float32_and_float64_special_bits_are_unchanged():
    f32 = struct.pack(
        "<7I",
        0x00000000,
        0x80000000,
        0x7F800000,
        0xFF800000,
        0x7FC00001,
        0x7FC12345,
        0xFFC00001,
    )
    f64 = struct.pack(
        "<7Q",
        0x0000000000000000,
        0x8000000000000000,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x7FF8000000000001,
        0x7FF8123456789ABC,
        0xFFF8000000000001,
    )
    for payload, dtype in ((f32, "F32"), (f64, "F64")):
        result = round_trip(payload, dtype, (7,), element_bits=len(payload) * 8 // 7)
        assert result.dtype == dtype
        assert hashlib.sha256(result.payload).hexdigest() == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("dtype", ["F16", "BF16"])
def test_exhaustive_16_bit_words_round_trip(dtype):
    payload = struct.pack("<65536H", *range(65536))
    result = round_trip(payload, dtype, (65536,), element_bits=16)
    assert result.element_bits == 16


def test_all_fp8_byte_values_round_trip():
    payload = bytes(range(256))
    result = round_trip(payload, "F8_E4M3", (256,), element_bits=8)
    assert result.payload == bytes(range(256))


@pytest.mark.parametrize(
    ("dtype", "payload"),
    [
        ("I8", bytes([0, 1, 127, 128, 255])),
        ("U16", struct.pack("<4H", 0, 1, 32768, 65535)),
        ("I64", struct.pack("<3q", -(1 << 63), -1, (1 << 63) - 1)),
        ("BOOL", bytes([0, 1, 2, 255])),
    ],
)
def test_integer_and_bool_payloads_are_raw_bytes(dtype, payload):
    result = round_trip(payload, dtype, (len(payload),))
    assert result.payload == payload


def test_structured_and_unknown_dtype_descriptors_are_opaque_and_immutable():
    descriptor = {
        "family": "packed",
        "storage": {"code": "future4", "lanes": [1, 3]},
    }
    result = round_trip(b"opaque", descriptor, (2, 3), layout={"order": "blocked"})
    assert result.dtype == descriptor
    assert result.dtype["family"] == "packed"
    assert result.dtype["storage"]["lanes"] == (1, 3)
    assert result.layout["order"] == "blocked"
    with pytest.raises(TypeError):
        result.dtype["family"] = "changed"
    with pytest.raises(TypeError):
        result.layout["order"] = "changed"
    with pytest.raises(AttributeError):
        result.payload = b"changed"
    reencoded = codec.encode_tensor(
        result.payload,
        result.dtype,
        result.shape,
        layout=result.layout,
    )
    assert codec.decode_tensor(reencoded).dtype == descriptor


def test_packed_four_bit_odd_count_checks_bytes_and_preserves_padding():
    payload = bytes([0x12, 0x34, 0xAF])  # five nibbles plus an untouched pad nibble
    result = round_trip(payload, {"kind": "packed4"}, (5,), element_bits=4)
    assert result.payload == payload
    assert result.element_bits == 4
    with pytest.raises(codec.TensorCodecError, match="ceil"):
        codec.encode_tensor(b"\x12", {"kind": "packed4"}, (5,), element_bits=4)


def test_scalar_and_empty_payloads_are_valid():
    scalar = round_trip(b"\x01\x02", "U16", (), element_bits=16)
    empty = round_trip(b"", "F32", (0, 8), element_bits=32)
    assert scalar.shape == ()
    assert empty.shape == (0, 8)
    assert empty.payload == b""


def test_deflate_selection_and_bytes_like_inputs():
    source = memoryview(b"repeatable payload " * 200)
    frame = codec.encode_tensor(source, "bytes", (len(source),), layout="row-major")
    header = codec._HEADER.unpack_from(frame)
    metadata_start = codec._HEADER.size
    metadata_end = metadata_start + header[4]
    metadata = json.loads(frame[metadata_start:metadata_end])
    assert metadata["body_codec"] == "deflate"
    assert codec.decode_frame(bytearray(frame)).payload == source.tobytes()


def test_corruption_truncation_and_trailing_bytes_fail_closed():
    frame = codec.encode_tensor(b"source bytes", "U8", (12,))
    corrupted_body = bytearray(frame)
    corrupted_body[codec._HEADER.size + 1] ^= 0x01
    with pytest.raises(codec.TensorCodecError, match="integrity"):
        codec.decode_tensor(corrupted_body)
    with pytest.raises(codec.TensorCodecError, match="truncated"):
        codec.decode_tensor(frame[:-1])
    with pytest.raises(codec.TensorCodecError, match="trailing"):
        codec.decode_tensor(frame + b"trailing")


def _rebuild_frame(frame, *, body=None, metadata_updates=None, body_sha=None):
    magic, version, flags, reserved, metadata_len, body_len = codec._HEADER.unpack_from(frame)
    metadata_start = codec._HEADER.size
    metadata_end = metadata_start + metadata_len
    old_metadata = json.loads(frame[metadata_start:metadata_end])
    if metadata_updates:
        old_metadata.update(metadata_updates)
    old_body = frame[metadata_end : metadata_end + body_len]
    if body is None:
        body = old_body
    old_metadata["body_bytes"] = len(body)
    metadata = codec._canonical_json(old_metadata)
    header = codec._HEADER.pack(magic, version, flags, reserved, len(metadata), len(body))
    trailer = codec._TRAILER.pack(
        hashlib.sha256(metadata).digest(),
        hashlib.sha256(body).digest() if body_sha is None else body_sha,
    )
    return header + metadata + body + trailer


def test_actual_body_corruption_and_forged_source_or_body_hashes_fail():
    payload = bytes(range(256))
    frame = codec.encode_tensor(payload, "U8", (256,))
    _, _, _, _, metadata_len, _ = codec._HEADER.unpack_from(frame)
    body_start = codec._HEADER.size + metadata_len
    corrupted = bytearray(frame)
    corrupted[body_start] ^= 0x01
    with pytest.raises(codec.TensorCodecError, match="body integrity"):
        codec.decode_tensor(corrupted)

    forged_body_sha = bytearray(hashlib.sha256(payload).digest())
    forged_body_sha[0] ^= 0x01
    with pytest.raises(codec.TensorCodecError, match="body integrity"):
        codec.decode_tensor(_rebuild_frame(frame, body_sha=bytes(forged_body_sha)))

    forged_source = _rebuild_frame(frame, metadata_updates={"source_sha256": "0" * 64})
    with pytest.raises(codec.TensorCodecError, match="source integrity"):
        codec.decode_tensor(forged_source)


@pytest.mark.parametrize("variant", ["overrun", "truncated", "trailing", "multistream"])
def test_deflate_inner_stream_checks_reject_forged_frames(variant):
    frame = codec.encode_tensor(b"A" * 4096, "U8", (4096,))
    metadata_start = codec._HEADER.size
    _, _, _, _, metadata_len, body_len = codec._HEADER.unpack_from(frame)
    metadata = json.loads(frame[metadata_start : metadata_start + metadata_len])
    assert metadata["body_codec"] == "deflate"
    body_start = metadata_start + metadata_len
    body = frame[body_start : body_start + body_len]
    if variant == "overrun":
        forged = _rebuild_frame(frame, metadata_updates={"source_bytes": 1})
    elif variant == "truncated":
        forged = _rebuild_frame(frame, body=body[:-1])
    elif variant == "trailing":
        forged = _rebuild_frame(frame, body=body + b"trailing")
    else:
        forged = _rebuild_frame(frame, body=body + zlib.compress(b"tail"))
    with pytest.raises(codec.TensorCodecError, match="deflate|decoded payload"):
        codec.decode_tensor(forged)


def test_explicit_source_and_decoded_budgets_override_small_defaults(monkeypatch):
    monkeypatch.setattr(codec, "DEFAULT_MAX_SOURCE_BYTES", 4)
    with pytest.raises(codec.TensorCodecError, match="source payload"):
        codec.encode_tensor(b"12345", "U8", (5,))
    frame = codec.encode_tensor(b"12345", "U8", (5,), max_source_bytes=5)

    monkeypatch.setattr(codec, "DEFAULT_MAX_DECODED_BYTES", 4)
    with pytest.raises(codec.TensorCodecError, match="max_decoded_bytes"):
        codec.decode_tensor(frame)
    assert codec.decode_tensor(frame, max_decoded_bytes=5).payload == b"12345"


def test_frame_budget_is_checked_before_memoryview_copy(monkeypatch):
    frame = codec.encode_tensor(b"payload", "U8", (7,))
    monkeypatch.setattr(codec, "DEFAULT_MAX_FRAME_BYTES", len(frame) - 1)
    with pytest.raises(codec.TensorCodecError, match="frame"):
        codec.decode_tensor(memoryview(frame))
    assert codec.decode_tensor(memoryview(frame), max_frame_bytes=len(frame)).payload == b"payload"
    with pytest.raises(codec.TensorCodecError, match="frame"):
        codec.decode_tensor(memoryview(frame), max_frame_bytes=len(frame) - 1)
    with pytest.raises(codec.TensorCodecError, match="frame"):
        codec.decode_tensor(memoryview(frame), max_frame_bytes=0)


def test_cycles_and_excessive_metadata_depth_fail_closed():
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(codec.TensorCodecError, match="cyclic"):
        codec.encode_tensor(b"x", {"cycle": cyclic}, (1,))
    nested = "x"
    for _ in range(codec.MAX_JSON_DEPTH + 1):
        nested = [nested]
    with pytest.raises(codec.TensorCodecError, match="depth"):
        codec.encode_tensor(b"x", nested, (1,))


def test_duplicate_metadata_keys_are_rejected():
    payload = b"x"
    metadata = (
        b'{"body_bytes":1,"body_codec":"raw","dtype":"U8",'
        b'"dtype":"U8","element_bits":null,"layout":null,"shape":[1],'
        b'"source_bytes":1,"source_sha256":"' + hashlib.sha256(payload).hexdigest().encode() + b'"}'
    )
    header = codec._HEADER.pack(codec.MAGIC, codec.VERSION, 0, 0, len(metadata), len(payload))
    trailer = codec._TRAILER.pack(hashlib.sha256(metadata).digest(), hashlib.sha256(payload).digest())
    with pytest.raises(codec.TensorCodecError, match="duplicate"):
        codec.decode_tensor(header + metadata + payload + trailer)


def test_metadata_and_source_limits_prevent_deflate_bombs():
    frame = codec.encode_tensor(b"A" * 100_000, "U8", (100_000,))
    with pytest.raises(codec.TensorCodecError, match="max_decoded_bytes"):
        codec.decode_tensor(frame, max_decoded_bytes=64)
    with pytest.raises(codec.TensorCodecError, match="max_decoded_bytes"):
        codec.decode_tensor(frame, max_decoded_bytes=0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: codec.encode_tensor(b"x", "U8", (-1,)),
        lambda: codec.encode_tensor(b"x", "U8", (1,), element_bits=0),
        lambda: codec.encode_tensor(b"x", {"bad": float("nan")}, (1,)),
        lambda: codec.encode_tensor(b"x", "U8", (1,), layout={1: "bad"}),
    ],
)
def test_invalid_metadata_is_rejected(call):
    with pytest.raises((TypeError, codec.TensorCodecError)):
        call()
