import ctypes
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / ".scratch" / "context-gpu-index-tests"
LIBRARY = SCRATCH / ("libcontext_gpu_index.dylib" if sys.platform == "darwin" else "libcontext_gpu_index.so")
U8P = ctypes.POINTER(ctypes.c_uint8)
U16P = ctypes.POINTER(ctypes.c_uint16)
U32P = ctypes.POINTER(ctypes.c_uint32)


def _pointer(array, ctype):
    return array.ctypes.data_as(ctypes.POINTER(ctype))


@pytest.fixture(scope="session")
def lib():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    cmd = [os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-fPIC"]
    cmd += ["-dynamiclib" if sys.platform == "darwin" else "-shared"]
    cmd += [str(ROOT / "scripts" / "bitexact_context_ans.cpp"),
            str(ROOT / "scripts" / "bitexact_context_gpu_index.cpp"), "-o", str(LIBRARY)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    dll = ctypes.CDLL(str(LIBRARY))
    dll.bcx_encode.argtypes = [U16P, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32,
                               U8P, U8P, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                               U32P, ctypes.POINTER(U8P), ctypes.POINTER(ctypes.c_size_t)]
    dll.bcx_encode.restype = ctypes.c_int
    dll.bcx_free.argtypes = [ctypes.c_void_p]
    dll.bcx_checkpoint.argtypes = [U8P, ctypes.c_size_t, ctypes.c_uint64,
                                   ctypes.c_uint32, ctypes.c_uint32, U8P, U8P,
                                   ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                   U16P, U32P, ctypes.c_uint32, U32P, U32P, U16P]
    dll.bcx_checkpoint.restype = ctypes.c_int
    return dll


def symbol(word):
    return (((int(word) >> 7) & 255) << 2) | ((int(word) & 127) >> 5)


def residual(word):
    return ((int(word) >> 15) << 5) | (int(word) & 31)


def independent_decode_chunk(stream, state, offset, start, end, freq, rows, cols,
                              nc, kind, resctx):
    decoded = []
    for i in range(start, end):
        cx = int(rows[i // cols.size]) * nc + int(cols[i % cols.size]) if kind == 0 else int(resctx[i])
        slot = state & 65535
        total = 0
        found = None
        for s, f0 in enumerate(freq[cx]):
            f = int(f0)
            if total <= slot < total + f:
                found = (s, f, total)
                break
            total += f
        assert found is not None
        s, f, cumulative = found
        decoded.append(s)
        state = f * (state >> 16) + slot - cumulative
        while state < (1 << 23):
            assert offset < len(stream)
            state = (state << 8) | stream[offset]
            offset += 1
    return decoded, state, offset


def setup_case(lib, kind, stride=19):
    rng = np.random.default_rng(701 + kind)
    rows_n, cols_n, nr, nc = 9, 17, 4, 3
    n = rows_n * cols_n
    vals = rng.integers(0, 65536, size=n, dtype=np.uint16)
    rc = (np.arange(rows_n, dtype=np.uint8) % nr).copy()
    cc = (np.arange(cols_n, dtype=np.uint8) % nc).copy()
    vocab, nctx = (1024, nr * nc) if kind == 0 else (64, 1024)
    freq = np.zeros((nctx, vocab), dtype=np.uint32)
    stream_ptr = U8P()
    stream_size = ctypes.c_size_t()
    status = lib.bcx_encode(_pointer(vals, ctypes.c_uint16), n, rows_n, cols_n,
                            _pointer(rc, ctypes.c_uint8), _pointer(cc, ctypes.c_uint8),
                            nr, nc, kind, _pointer(freq, ctypes.c_uint32),
                            ctypes.byref(stream_ptr), ctypes.byref(stream_size))
    assert status == 0
    stream = ctypes.string_at(stream_ptr, stream_size.value)
    lib.bcx_free(stream_ptr)
    stream_array = np.frombuffer(stream, dtype=np.uint8).copy()
    resctx = np.array([symbol(x) for x in vals], dtype=np.uint16) if kind == 1 else None
    count = (n - 1) // stride + 1
    states = np.zeros(count, dtype=np.uint32)
    offsets = np.zeros(count, dtype=np.uint32)
    out = np.full(n, 0xFFFF, dtype=np.uint16)
    return {"kind": kind, "stride": stride, "rows_n": rows_n, "cols_n": cols_n,
            "nr": nr, "nc": nc, "n": n, "vals": vals, "rc": rc, "cc": cc,
            "freq": freq, "stream": stream, "stream_array": stream_array,
            "resctx": resctx, "states": states, "offsets": offsets, "out": out}


def call_checkpoint(lib, x, stream=None, freq=None, rc=None):
    stream_array = x["stream_array"] if stream is None else np.frombuffer(stream, dtype=np.uint8).copy()
    freq = x["freq"] if freq is None else freq
    rc = x["rc"] if rc is None else rc
    resctx_ptr = _pointer(x["resctx"], ctypes.c_uint16) if x["resctx"] is not None else None
    return lib.bcx_checkpoint(_pointer(stream_array, ctypes.c_uint8), stream_array.size,
                              x["n"], x["rows_n"], x["cols_n"],
                              _pointer(rc, ctypes.c_uint8), _pointer(x["cc"], ctypes.c_uint8),
                              x["nr"], x["nc"], x["kind"], resctx_ptr,
                              _pointer(freq, ctypes.c_uint32), x["stride"],
                              _pointer(x["states"], ctypes.c_uint32),
                              _pointer(x["offsets"], ctypes.c_uint32),
                              _pointer(x["out"], ctypes.c_uint16))


@pytest.mark.parametrize("kind", [0, 1])
@pytest.mark.parametrize("stride", [1, 7, 19, 128])
def test_every_checkpoint_resumes_independent_rans_decode(lib, kind, stride):
    x = setup_case(lib, kind, stride)
    assert call_checkpoint(lib, x) == 0
    expected = [symbol(v) if kind == 0 else residual(v) for v in x["vals"]]
    assert x["out"].tolist() == expected
    for cp, start in enumerate(range(0, x["n"], stride)):
        end = min(start + stride, x["n"])
        chunk, state, offset = independent_decode_chunk(
            x["stream"], int(x["states"][cp]), int(x["offsets"][cp]),
            start, end, x["freq"], x["rc"], x["cc"], x["nc"], x["kind"], x["resctx"])
        assert chunk == expected[start:end]
        if cp + 1 < len(x["states"]):
            assert (state, offset) == (int(x["states"][cp + 1]), int(x["offsets"][cp + 1]))
        else:
            assert state == 1 << 23
            assert offset == len(x["stream"])


@pytest.mark.parametrize("kind", [0, 1])
def test_truncated_stream_and_malformed_frequency_tables_rejected(lib, kind):
    x = setup_case(lib, kind)
    before = x["out"].copy()
    assert call_checkpoint(lib, x, stream=x["stream"][:-1]) != 0
    assert np.array_equal(x["out"], before)
    bad = x["freq"].copy()
    used_context = int(x["rc"][0]) * x["nc"] + int(x["cc"][0]) if kind == 0 else int(x["resctx"][0])
    used_symbol = symbol(x["vals"][0]) if kind == 0 else residual(x["vals"][0])
    bad[used_context, used_symbol] += 1
    assert call_checkpoint(lib, x, freq=bad) != 0
    assert np.array_equal(x["out"], before)
    assert call_checkpoint(lib, x, stream=x["stream"] + b"\x00") != 0
    assert np.array_equal(x["out"], before)


def test_invalid_geometry_or_labels_rejected_without_output(lib):
    x = setup_case(lib, kind=0)
    sentinel = x["out"].copy()
    stream = x["stream_array"]
    call = lambda n, rc: lib.bcx_checkpoint(
        _pointer(stream, ctypes.c_uint8), stream.size, n, x["rows_n"], x["cols_n"],
        _pointer(rc, ctypes.c_uint8), _pointer(x["cc"], ctypes.c_uint8), x["nr"], x["nc"], 0, None,
        _pointer(x["freq"], ctypes.c_uint32), x["stride"], _pointer(x["states"], ctypes.c_uint32),
        _pointer(x["offsets"], ctypes.c_uint32), _pointer(x["out"], ctypes.c_uint16))
    assert call(x["n"] - 1, x["rc"]) != 0
    bad_rc = x["rc"].copy(); bad_rc[0] = x["nr"]
    assert call(x["n"], bad_rc) != 0
    assert np.array_equal(x["out"], sentinel)
    assert lib.bcx_checkpoint(None, stream.size, x["n"], x["rows_n"], x["cols_n"],
                              _pointer(x["rc"], ctypes.c_uint8), _pointer(x["cc"], ctypes.c_uint8),
                              x["nr"], x["nc"], 0, None, _pointer(x["freq"], ctypes.c_uint32),
                              x["stride"], _pointer(x["states"], ctypes.c_uint32),
                              _pointer(x["offsets"], ctypes.c_uint32),
                              _pointer(x["out"], ctypes.c_uint16)) != 0
    assert np.array_equal(x["out"], sentinel)
