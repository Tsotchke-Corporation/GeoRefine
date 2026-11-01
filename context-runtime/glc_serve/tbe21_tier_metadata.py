"""CPU reference for the redundant per-chunk TBE21 tier counters."""
from __future__ import annotations

import numpy as np

TAIL_WORDS = 16


def _array(value, name, dtypes, shape=None):
    if isinstance(value, (bytes, bytearray, memoryview)):
        if not value:
            raise ValueError(f"{name}: invalid byte length")
        raw = bytes(value)
        if name == "l1" and len(raw) % 2:
            raise ValueError("l1: byte length must be even")
        dt = "<u2" if name == "l1" else "<u4"
        a = np.frombuffer(raw, dtype=dt)
        if name == "l1" and shape is not None and a.size == int(np.prod(shape)):
            a = a.reshape(shape)
    elif isinstance(value, np.ndarray):
        a = value
        if name == "l1" and a.dtype == np.dtype("uint8"):
            if not a.flags.c_contiguous or a.shape != (shape[0], shape[1], 32):
                raise ValueError(f"l1: uint8 input must have shape {(shape[0], shape[1], 32)}")
            a = a.view("<u2").reshape(shape)
        elif a.dtype not in dtypes:
            raise ValueError(f"{name}: unsupported dtype {a.dtype}")
        if not a.flags.c_contiguous:
            raise ValueError(f"{name}: must be C-contiguous")
        if a.dtype.itemsize > 1 and a.dtype.byteorder == ">":
            raise ValueError(f"{name}: must use little-endian storage")
    else:
        raise ValueError(f"{name}: expected ndarray or bytes-like input")
    if isinstance(value, np.ndarray) and a.dtype not in dtypes and not (name == "l1" and value.dtype == np.dtype("uint8")):
        raise ValueError(f"{name}: unsupported dtype {a.dtype}")
    if shape is not None and a.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {a.shape}")
    return a


def _read_bits(words, bit, width):
    out = 0
    for j in range(width):
        p = bit + j
        out |= ((int(words[p >> 5]) >> (p & 31)) & 1) << j
    return out


def _dimensions(N, K):
    if (type(N) is not int or type(K) is not int or N <= 0 or K <= 0 or K % 128
            or N >= 2 ** 31 or K >= 2 ** 31):
        raise ValueError("N and K must be positive integers and K must be divisible by 128")


def build_headers(l1, ovf, rowoff, N, K):
    """Return uint32[N,K//128], each word n1|n2<<8|n3<<16|n4<<24."""
    _dimensions(N, K)
    nch = K // 128
    a = _array(l1, "l1", (np.dtype("uint16"), np.dtype("<u2")), (N, nch, 16))
    words = _array(ovf, "ovf", (np.dtype("uint32"), np.dtype("<u4"), np.dtype("int32"), np.dtype("<i4")))
    offsets = _array(rowoff, "rowoff", (np.dtype("uint32"), np.dtype("<u4"), np.dtype("int32"), np.dtype("<i4")), (N,))
    if words.ndim != 1:
        raise ValueError("ovf must be a one-dimensional word stream")
    offs = [int(x) & 0xffffffff for x in offsets]
    if offs[0] != 0:
        raise ValueError("first row must start at overflow word zero")
    if any(x > y for x, y in zip(offs, offs[1:])):
        raise ValueError("rowoff must be nondecreasing")
    if any(x > len(words) - TAIL_WORDS for x in offs) or len(words) < TAIL_WORDS:
        raise ValueError("rowoff exceeds overflow payload before required tail")
    if np.any(words[-TAIL_WORDS:] != 0):
        raise ValueError("overflow stream must end in sixteen zero words")
    result = np.zeros((N, nch), dtype=np.uint32)
    for r in range(N):
        row_start = offs[r] * 32
        row_end = (offs[r + 1] if r + 1 < N else len(words) - TAIL_WORDS) * 32
        cur = row_start
        for c in range(nch):
            n1 = sum(((int(a[r, c, lane]) >> shift) & 3) == 3
                     for lane in range(16) for shift in range(0, 16, 2))
            n2 = 0
            for _ in range(n1):
                if cur + 2 > row_end: raise ValueError("truncated L2 tier digits")
                n2 += _read_bits(words, cur, 2) == 3
                cur += 2
            n3 = 0
            for _ in range(n2):
                if cur + 2 > row_end: raise ValueError("truncated L3 tier digits")
                n3 += _read_bits(words, cur, 2) == 3
                cur += 2
            n4 = 0
            for _ in range(n3):
                if cur + 3 > row_end: raise ValueError("truncated L4 tier digits")
                n4 += _read_bits(words, cur, 3) == 7
                cur += 3
            cur += 8 * n4
            if cur > row_end: raise ValueError("truncated raw exponent payload")
            result[r, c] = n1 | (n2 << 8) | (n3 << 16) | (n4 << 24)
        if row_end - cur >= 32:
            raise ValueError("row payload has excess words")
        for p in range(cur, row_end):
            if (int(words[p >> 5]) >> (p & 31)) & 1:
                raise ValueError("nonzero row alignment padding")
    return result


def validate_headers(headers, l1, ovf, rowoff, N, K):
    expected_shape = (N, K // 128) if type(K) is int and K > 0 and K % 128 == 0 else None
    h = _array(headers, "headers", (np.dtype("uint32"), np.dtype("<u4")), expected_shape)
    expected = build_headers(l1, ovf, rowoff, N, K)
    if not np.array_equal(h, expected):
        raise ValueError("tier metadata header mismatch")
    return True
