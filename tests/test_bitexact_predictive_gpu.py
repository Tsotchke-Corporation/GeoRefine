import hashlib
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bitexact_predictive_codec as codec
import bitexact_predictive_gpu as gpu


def _repack(frame, **sections):
    fields = list(codec.HEADER.unpack_from(frame))
    magic, version, mode, flags, rows, cols, meta_len, table_len, stream_len, ref_hash, target_hash = fields
    offset = codec.HEADER.size
    meta = frame[offset:offset + meta_len]
    offset += meta_len
    table = frame[offset:offset + table_len]
    offset += table_len
    stream = frame[offset:offset + stream_len]
    offset += stream_len
    data = frame[offset:-32]
    mode = sections.get("mode", mode)
    meta = sections.get("meta", meta)
    table = sections.get("table", table)
    stream = sections.get("stream", stream)
    data = sections.get("data", data)
    if mode == 2:
        meta = table = stream = b""
        meta_len = table_len = stream_len = 0
    else:
        meta_len, table_len, stream_len = len(meta), len(table), len(stream)
    head = codec.HEADER.pack(magic, version, mode, flags, rows, cols, meta_len, table_len, stream_len, ref_hash, target_hash)
    body = head + meta + table + stream + data
    return body + hashlib.sha256(body).digest()


def _sample(seed=8, shape=(5, 71)):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 65536, shape, dtype=np.uint16),
            rng.integers(0, 65536, shape, dtype=np.uint16))


def _frame_for_mode(target, ref, mode):
    if mode == 2:
        return _repack(codec.encode(target, ref), mode=2, data=target.astype("<u2").tobytes())
    labels, flips, thresholds = codec._fit(target, ref)
    keys = codec._aligned_keys(ref, flips)
    bins = np.searchsorted(thresholds, keys, side="right").astype(np.uint16)
    main = ((target >> 15) & 1) * 1024 + ((target >> 7) & 255) * 4 + ((target & 127) >> 5)
    residual = target & 31
    contexts = (labels[:, None].astype(np.uint16) * 16 + bins).ravel()
    main_freq, main_stream = codec._rans_encode(main.ravel(), contexts, 256, 2048)
    metadata = zlib.compress(labels.tobytes() + flips.tobytes() + thresholds.tobytes(), 6)
    if mode == 0:
        data = codec._packbits(residual.ravel(), 5)
    else:
        residual_freq, residual_stream = codec._rans_encode(residual.ravel(), main.ravel(), 2048, 32)
        data = struct.pack("<I", len(residual_freq)) + residual_freq + residual_stream
    body = codec.HEADER.pack(
        b"PPCX", 1, mode, 0, *target.shape, len(metadata), len(main_freq), len(main_stream),
        hashlib.sha256(ref.astype("<u2").tobytes()).digest(),
        hashlib.sha256(target.astype("<u2").tobytes()).digest(),
    ) + metadata + main_freq + main_stream + data
    return body + hashlib.sha256(body).digest()


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_cpu_checkpoint_index_accepts_each_frame_mode(mode):
    target, ref = _sample()
    frame = _frame_for_mode(target, ref, mode)
    assert np.array_equal(codec.decode(frame, ref), target)
    indexed = gpu._read_frame(frame, ref, stride=17)
    assert indexed["mode"] == mode and indexed["shape"] == target.shape
    assert indexed["source_sha256"] == hashlib.sha256(target.astype("<u2").tobytes()).hexdigest()
    if mode != 2:
        assert indexed["main_states"][0] >= 1 << 23
    if mode == 1:
        assert len(indexed["residual_states"]) == (target.size + 16) // 17


def test_cpu_checkpoint_rejects_reference_frame_and_target_corruption():
    target, ref = _sample()
    frame = codec.encode(target, ref)
    with pytest.raises(codec.CodecError, match="reference mismatch"):
        gpu._read_frame(frame, np.zeros_like(ref), 13)
    corrupt = bytearray(frame)
    corrupt[-33] ^= 1
    with pytest.raises(codec.CodecError, match="checksum"):
        gpu._read_frame(corrupt, ref, 13)
    body = bytearray(frame[:-32])
    body[codec.HEADER.size - 32] ^= 1
    malformed = bytes(body) + hashlib.sha256(body).digest()
    with pytest.raises(codec.CodecError, match="reference mismatch|target checksum"):
        gpu._read_frame(malformed, ref, 13)


def test_cpu_checkpoint_rejects_bad_context_tables_streams_and_thresholds():
    target, ref = _sample()
    frame = _frame_for_mode(target, ref, 0)
    fields = list(codec.HEADER.unpack_from(frame))
    offset = codec.HEADER.size
    meta_len, table_len, stream_len = fields[6:9]
    metadata = bytearray(zlib.decompress(frame[offset:offset + meta_len]))
    offset += meta_len
    table = bytearray(zlib.decompress(frame[offset:offset + table_len]))
    offset += table_len + stream_len
    metadata[0] = 16
    with pytest.raises(codec.CodecError, match="metadata"):
        gpu._read_frame(_repack(frame, meta=zlib.compress(metadata)), ref, 19)

    metadata = zlib.decompress(frame[codec.HEADER.size:codec.HEADER.size + meta_len])
    bad_table = bytearray(table)
    bad_table[:4] = b"\x01\x00\x00\x00"
    with pytest.raises(codec.CodecError, match="checkpoint"):
        gpu._read_frame(_repack(frame, table=zlib.compress(bad_table)), ref, 19)
    with pytest.raises(codec.CodecError, match="checkpoint"):
        gpu._read_frame(_repack(frame, stream=b"\x00\x00\x00\x00"), ref, 19)

    bad_meta = bytearray(metadata)
    threshold_offset = 2 * target.shape[0]
    bad_meta[threshold_offset:threshold_offset + 2] = b"\xff\xff"
    bad_meta[threshold_offset + 2:threshold_offset + 4] = b"\x00\x00"
    with pytest.raises(codec.CodecError, match="monotone"):
        gpu._read_frame(_repack(frame, meta=zlib.compress(bad_meta)), ref, 19)


def test_cold_native_cache_compiles_safely_for_concurrent_processes():
    repo = Path(__file__).resolve().parents[1]
    scratch = repo / ".scratch"
    scratch.mkdir(exist_ok=True)
    cache_dir = Path(tempfile.mkdtemp(prefix="predictive-cold-cache-", dir=scratch))
    env = os.environ.copy()
    env["BITEXACT_PREDICTIVE_CACHE_DIR"] = str(cache_dir)
    code = """
import hashlib, numpy as np
import bitexact_predictive_codec as codec
import bitexact_predictive_gpu as gpu
ref=np.full((128,512),0x3f80,dtype=np.uint16)
target=np.full_like(ref,0x3f83)
frame=codec.encode(target,ref)
decoded=codec.decode(frame,ref)
indexed=gpu._read_frame(frame,ref,17)
assert np.array_equal(decoded,target)
assert indexed['mode'] in (0,1)
assert indexed['source_sha256']==hashlib.sha256(target.astype('<u2').tobytes()).hexdigest()
print(hashlib.sha256(decoded.tobytes()).hexdigest())
"""
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "scripts") + os.pathsep + env.get("PYTHONPATH", "")
    expected = hashlib.sha256(np.full((128, 512), 0x3F83, dtype="<u2").tobytes()).hexdigest()
    processes = []
    try:
        processes = [subprocess.Popen([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for _ in range(4)]
        for process in processes:
            stdout, stderr = process.communicate(timeout=90)
            assert process.returncode == 0, stderr
            assert stdout.strip() == expected
        libraries = list((cache_dir / "build").glob("*.dylib" if sys.platform == "darwin" else "*.so"))
        assert len(libraries) == 2
        expected_source_hashes = {
            hashlib.sha256((repo / "scripts" / name).read_bytes()).hexdigest()
            for name in ("bitexact_predictive_ans.cpp", "bitexact_predictive_gpu_index.cpp")
        }
        assert all(any(source_hash in library.name for library in libraries) for source_hash in expected_source_hashes)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        shutil.rmtree(cache_dir, ignore_errors=True)
