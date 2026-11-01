"""Compact lookup retains the exact existing BCTX streams and terminal checks."""
import numpy as np
import pytest
from georefine_context_runtime import bitexact_context_codec as codec


@pytest.mark.parametrize('nr,nc', [(1, 1), (4, 4), (16, 4), (64, 16)])
@pytest.mark.parametrize('distribution', ['all_words', 'constant', 'sparse'])
def test_compact_inverse_matches_both_original_symbol_streams(monkeypatch, nr, nc, distribution):
    words = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    if distribution == 'constant':
        words = np.full((64, 128), 0x3f80, dtype=np.uint16)
    elif distribution == 'sparse':
        words = words.copy()
        words[:240] = 0x3f80
    lib = codec._lib()
    rc, cc = codec._labels(words, nr, nc)
    streams = [codec._encode_kind(lib, words, rc, cc, nr, nc, kind) for kind in (0, 1)]
    outputs = []
    for flag in ('0', '1'):
        monkeypatch.setenv('GEOREFINE_CONTEXT_COMPACT_RANS', flag)
        main = codec._decode_kind(lib, streams[0][1], streams[0][0], rc, cc,
                                  *words.shape, nr, nc, 0)
        residual = codec._decode_kind(lib, streams[1][1], streams[1][0], rc, cc,
                                      *words.shape, nr, nc, 1, main)
        m, r = main.astype(np.uint32), residual.astype(np.uint32)
        restored = (((m >> 2) << 7) | ((m & 3) << 5) | ((r >> 5) << 15) | (r & 31)).astype(np.uint16)
        assert np.array_equal(restored.reshape(words.shape), words)
        outputs.append((main, residual))
    assert all(np.array_equal(a, b) for a, b in zip(*outputs))


@pytest.mark.parametrize('fault', ['state', 'truncated', 'trailing', 'zero_frequencies', 'overflow'])
def test_compact_inverse_preserves_native_rejection(monkeypatch, fault):
    words = np.arange(64 * 128, dtype=np.uint16).reshape(64, 128)
    lib = codec._lib(); nr = nc = 4
    rc, cc = codec._labels(words, nr, nc)
    freq, stream = codec._encode_kind(lib, words, rc, cc, nr, nc, 0)
    if fault == 'state': stream = b'\0\0\0\0' + stream[4:]
    elif fault == 'truncated': stream = stream[:-1]
    elif fault == 'trailing': stream += b'\0'
    elif fault == 'zero_frequencies': freq = bytes(len(freq))
    else:
        f = np.frombuffer(freq, dtype='<u4').copy()
        f[np.flatnonzero(f)[0]] = 65537
        freq = f.tobytes()
    for flag in ('0', '1'):
        monkeypatch.setenv('GEOREFINE_CONTEXT_COMPACT_RANS', flag)
        with pytest.raises(codec.CodecError):
            codec._decode_kind(lib, stream, freq, rc, cc, *words.shape, nr, nc, 0)


def test_compact_lookup_is_explicit_and_encoding_unchanged(monkeypatch):
    words = np.full((64, 128), 0x3f80, dtype=np.uint16)
    monkeypatch.delenv('GEOREFINE_CONTEXT_COMPACT_RANS', raising=False)
    baseline = codec.encode(words)
    monkeypatch.setenv('GEOREFINE_CONTEXT_COMPACT_RANS', '1')
    assert codec.encode(words) == baseline
    assert np.array_equal(codec.decode(baseline), words)
    monkeypatch.setenv('GEOREFINE_CONTEXT_COMPACT_RANS', 'invalid')
    with pytest.raises(codec.CodecError, match='must be'):
        codec.decode(baseline)
