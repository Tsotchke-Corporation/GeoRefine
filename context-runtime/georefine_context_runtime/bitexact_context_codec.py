"""Standalone, exact BF16 codec using integer static rANS context models."""

from __future__ import annotations
import ctypes
import fcntl
import hashlib
import os
from pathlib import Path
import struct
import threading
import tempfile
import shutil
import sys
import zlib
import numpy as np

MAGIC = b"BCTX"
VERSION = 1
HEADER = struct.Struct("<4sBBBBIIQQQQ32s")
MAX_WORDS = 0xFFFFFFFF
ROWS = {1, 4, 16, 64}
COLS = {1, 4, 16}
_LOCK = threading.Lock()


class CodecError(ValueError):
    """Invalid input or malformed frame."""


def _lib():
    src = Path(__file__).with_name("bitexact_context_ans.cpp")
    default_cache = Path(__file__).resolve().parents[1] / ".scratch" / "bitexact-context-codec-20261006"
    path = Path(os.environ.get("BITEXACT_CONTEXT_CACHE_DIR", str(default_cache))) / "libbitexact_context_ans.dylib"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, path.with_suffix(path.suffix + '.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists() or path.stat().st_size == 0 or path.stat().st_mtime_ns < src.stat().st_mtime_ns:
            p = os.environ.get("CXX", "c++")
            import subprocess

            fd, temporary = tempfile.mkstemp(prefix='.' + path.name, suffix='.partial', dir=path.parent)
            os.close(fd)
            try:
                run = subprocess.run(
                    [shutil.which(p) or p, "-O3", "-std=c++17", "-shared", "-fPIC", "-pipe", str(src), "-o", temporary], capture_output=True,
                    close_fds=sys.platform != 'darwin'
                )
                if run.returncode:
                    raise RuntimeError("C++ coder build failed: " + run.stderr.decode(errors="replace"))
                if Path(temporary).stat().st_size == 0:
                    raise RuntimeError("C++ coder build produced an empty library")
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        lib = ctypes.CDLL(str(path))
        U8 = ctypes.POINTER(ctypes.c_uint8)
        U16 = ctypes.POINTER(ctypes.c_uint16)
        U32 = ctypes.POINTER(ctypes.c_uint32)
        lib.bcx_encode.argtypes = [
            U16,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
            U8,
            U8,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            U32,
            ctypes.POINTER(U8),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.bcx_encode.restype = ctypes.c_int
        lib.bcx_decode.argtypes = [
            U8,
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
            U8,
            U8,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            U16,
            U32,
            U16,
        ]
        lib.bcx_decode.restype = ctypes.c_int
        lib.bcx_free.argtypes = [ctypes.c_void_p]
        lib.bcx_pack6.argtypes = [U16, ctypes.c_uint64, ctypes.POINTER(U8), ctypes.POINTER(ctypes.c_size_t)]
        lib.bcx_pack6.restype = ctypes.c_int
        lib.bcx_unpack6.argtypes = [U8, ctypes.c_size_t, ctypes.c_uint64, U16]
        lib.bcx_unpack6.restype = ctypes.c_int
        return lib


def _u16(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))


def _u32(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))


def _input(words):
    a = np.asarray(words)
    if a.ndim != 2 or a.dtype.kind not in "ui" or a.dtype.itemsize != 2:
        raise CodecError("expected a 2D uint16 word array")
    if not all(a.shape) or a.size > MAX_WORDS:
        raise CodecError("invalid or oversized geometry")
    if a.dtype.kind == "i" and (np.any(a < 0) or np.any(a > 65535)):
        raise CodecError("word outside uint16 range")
    return np.ascontiguousarray(a, dtype="<u2")


def _labels(a, nr, nc):
    rows, cols = a.shape
    rs = np.zeros(rows, dtype=np.uint64)
    cs = np.zeros(cols, dtype=np.uint64)
    # Stream over rows so 1.27B-word tensors do not create full-size float/magnitude arrays.
    step = max(1, min(rows, 1_000_000 // cols))
    for lo in range(0, rows, step):
        w = a[lo : lo + step].astype(np.uint32)
        mag = ((w >> 7) & 255) * 128 + (w & 127)
        rs[lo : lo + len(w)] = mag.sum(axis=1, dtype=np.uint64)
        cs += mag.sum(axis=0, dtype=np.uint64)

    def quant(v, k):
        order = np.argsort(v, kind="stable")
        out = np.empty(len(v), dtype=np.uint8)
        out[order] = (np.arange(len(v), dtype=np.uint64) * k // len(v)).astype(np.uint8)
        return out

    return quant(rs, nr), quant(cs, nc)


def _encode_kind(lib, a, rc, cc, nr, nc, kind):
    vocab = 64 if kind else 1024
    nctx = 1024 if kind else nr * nc
    freq = np.zeros(nctx * vocab, dtype="<u4")
    ptr = ctypes.POINTER(ctypes.c_uint8)()
    length = ctypes.c_size_t()
    rc = np.ascontiguousarray(rc, dtype=np.uint8)
    cc = np.ascontiguousarray(cc, dtype=np.uint8)
    rc_buf = (ctypes.c_uint8 * len(rc)).from_buffer(rc)
    cc_buf = (ctypes.c_uint8 * len(cc)).from_buffer(cc)
    status = lib.bcx_encode(
        _u16(a),
        a.size,
        a.shape[0],
        a.shape[1],
        rc_buf,
        cc_buf,
        nr,
        nc,
        kind,
        _u32(freq),
        ctypes.byref(ptr),
        ctypes.byref(length),
    )
    if status:
        raise CodecError(f"rANS encode failed ({status})")
    try:
        stream = ctypes.string_at(ptr, length.value)
    finally:
        lib.bcx_free(ptr)
    return freq.tobytes(), stream


def _decode_kind(lib, stream, freq, rc, cc, rows, cols, nr, nc, kind, main=None):
    n = rows * cols
    vocab = 64 if kind else 1024
    nctx = 1024 if kind else nr * nc
    expected = nctx * vocab * 4
    if len(freq) != expected:
        raise CodecError("frequency table length mismatch")
    table = np.frombuffer(freq, dtype="<u4").copy()
    out = np.empty(n, dtype="<u2")
    rc = np.ascontiguousarray(rc, dtype=np.uint8)
    cc = np.ascontiguousarray(cc, dtype=np.uint8)
    rb = (ctypes.c_uint8 * len(rc)).from_buffer(rc)
    cb = (ctypes.c_uint8 * len(cc)).from_buffer(cc)
    sb = ctypes.create_string_buffer(stream)
    main_arr = None if main is None else np.ascontiguousarray(main, dtype="<u2").ravel()
    status = lib.bcx_decode(
        ctypes.cast(sb, ctypes.POINTER(ctypes.c_uint8)),
        len(stream),
        n,
        rows,
        cols,
        rb,
        cb,
        nr,
        nc,
        kind,
        _u16(main_arr) if kind else ctypes.POINTER(ctypes.c_uint16)(),
        _u32(table),
        _u16(out),
    )
    if status:
        raise CodecError(f"rANS decode failed ({status})")
    return out if not kind else out.astype(np.uint8)


def _pack6(lib, a):
    ptr = ctypes.POINTER(ctypes.c_uint8)()
    length = ctypes.c_size_t()
    status = lib.bcx_pack6(_u16(a), a.size, ctypes.byref(ptr), ctypes.byref(length))
    if status:
        raise CodecError("raw residual packing failed")
    try:
        return ctypes.string_at(ptr, length.value)
    finally:
        lib.bcx_free(ptr)


def _unpack6(lib, raw, n):
    out = np.empty(n, dtype="<u2")
    b = ctypes.create_string_buffer(raw)
    if lib.bcx_unpack6(ctypes.cast(b, ctypes.POINTER(ctypes.c_uint8)), len(raw), n, _u16(out)):
        raise CodecError("invalid raw residual stream")
    return out


def _inflate_exact(data, expected):
    try:
        dec = zlib.decompressobj()
        raw = dec.decompress(data, expected + 1)
        if len(raw) > expected or dec.unconsumed_tail:
            raise CodecError("compressed section exceeds declared size")
        raw += dec.flush()
    except zlib.error as e:
        raise CodecError("invalid compressed section") from e
    if len(raw) != expected or not dec.eof or dec.unused_data:
        raise CodecError("compressed section length/trailing data mismatch")
    return raw


def _frame(mode, nr, nc, r, c, meta, table, sy, rs, raw, dig):
    body = (
        HEADER.pack(MAGIC, VERSION, mode, nr, nc, r, c, len(meta), len(table), len(sy), len(rs), dig)
        + meta
        + table
        + sy
        + rs
        + raw
    )
    return body + hashlib.sha256(body).digest()


def encode(words, *, row_classes=16, col_classes=4):
    """Encode a 2D BF16-bit-pattern array into a self-contained frame."""
    a = _input(words)
    if row_classes not in ROWS or col_classes not in COLS:
        raise CodecError("unsupported class count")
    r, c = a.shape
    raw = a.tobytes()
    dig = hashlib.sha256(raw).digest()
    fallback = _frame(1, row_classes, col_classes, r, c, b"", b"", b"", b"", raw, dig)
    lib = _lib()
    rc, cc = _labels(a, row_classes, col_classes)
    meta = zlib.compress(rc.tobytes() + cc.tobytes(), 6)
    sf, sy = _encode_kind(lib, a, rc, cc, row_classes, col_classes, 0)
    st = zlib.compress(sf, 6)
    # Always retain the best complete residual representation, including its table cost.
    crf, cr = _encode_kind(lib, a, rc, cc, row_classes, col_classes, 1)
    packed = _pack6(lib, a)
    raw6 = _frame(0, row_classes, col_classes, r, c, meta, st, sy, packed, b"", dig)
    cond = _frame(128, row_classes, col_classes, r, c, meta, zlib.compress(sf + crf, 6), sy, cr, b"", dig)
    coded = min((raw6, cond), key=len)
    return coded if len(coded) < len(fallback) else fallback


def decode(frame):
    """Decode a complete frame and verify source checksum; trailing data rejects."""
    if not isinstance(frame, (bytes, bytearray, memoryview)) or len(frame) < HEADER.size:
        raise CodecError("short frame")
    frame = bytes(frame)
    if hashlib.sha256(frame[:-32]).digest() != frame[-32:]:
        raise CodecError("frame checksum mismatch")
    try:
        magic, ver, mode, nr, nc, r, c, ml, tl, sl, dl, dig = HEADER.unpack_from(frame)
    except struct.error as e:
        raise CodecError("bad frame header") from e
    if magic != MAGIC or ver != VERSION or nr not in ROWS or nc not in COLS:
        raise CodecError("unsupported frame header")
    n = r * c
    if not r or not c or n > MAX_WORDS:
        raise CodecError("invalid geometry")
    end = HEADER.size + ml + tl + sl + dl
    if mode == 1:
        if ml or tl or sl or dl or len(frame) != end + 2 * n + 32:
            raise CodecError("invalid raw frame length")
        raw = frame[end:-32]
        if hashlib.sha256(raw).digest() != dig:
            raise CodecError("raw checksum mismatch")
        return np.frombuffer(raw, dtype="<u2").copy().reshape(r, c)
    if mode not in (0, 128) or len(frame) != end + 32:
        raise CodecError("invalid frame mode/length")
    p = HEADER.size
    meta = _inflate_exact(frame[p : p + ml], r + c)
    p += ml
    nf = (nr * nc * 1024 + 1024 * 64) * 4 if mode == 128 else nr * nc * 1024 * 4
    table = _inflate_exact(frame[p : p + tl], nf)
    p += tl
    rc = np.frombuffer(meta[:r], dtype=np.uint8).copy()
    cc = np.frombuffer(meta[r:], dtype=np.uint8).copy()
    if rc.max(initial=0) >= nr or cc.max(initial=0) >= nc:
        raise CodecError("class outside declared range")
    sfbytes = nr * nc * 1024 * 4
    sf = table[:sfbytes]
    rf = table[sfbytes:] if mode == 128 else None
    syraw = frame[p : p + sl]
    p += sl
    rsraw = frame[p : p + dl]
    lib = _lib()
    main = _decode_kind(lib, syraw, sf, rc, cc, r, c, nr, nc, 0).reshape(r, c)
    if mode == 128:
        residual = _decode_kind(lib, rsraw, rf, rc, cc, r, c, nr, nc, 1, main).reshape(r, c).astype(np.uint32)
    else:
        residual = _unpack6(lib, rsraw, n).reshape(r, c).astype(np.uint32)
    s = main.astype(np.uint32)
    out = (((s >> 2) << 7) | ((s & 3) << 5) | (residual & 31) | ((residual >> 5) << 15)).astype("<u2")
    out = np.ascontiguousarray(out)
    if hashlib.sha256(out.tobytes()).digest() != dig:
        raise CodecError("decoded checksum mismatch")
    return out
