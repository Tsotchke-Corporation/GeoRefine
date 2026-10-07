import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import bitexact_predictive_codec as codec
from bitexact_predictive_mix import PredictiveMixError, fit_mixed_coefficients, predict


def _words(values):
    values = np.asarray(values, dtype=np.float32)
    return (values.view(np.uint32) >> 16).astype(np.uint16)


def _sample():
    rng = np.random.default_rng(41)
    ref = _words(rng.normal(size=(8, 24)))
    target_float = (ref.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    target = _words(target_float * np.float32(0.75) + rng.normal(size=(8, 24)).astype(np.float32) * 0.03)
    return target, ref


def test_ppcx_roundtrip_wire_header_and_target_reference_hashes():
    target, ref = _sample()
    frame = codec.encode(target, ref)
    magic, version, _mode, flags, rows, cols, *_ = codec.HEADER.unpack_from(frame)
    assert (magic, version, flags, rows, cols) == (b"PPCX", 1, 0, *target.shape)
    assert codec.decode(frame, ref).tobytes() == target.astype("<u2").tobytes()
    assert frame[codec.HEADER.size - 64:codec.HEADER.size - 32] == hashlib.sha256(ref.astype("<u2").tobytes()).digest()
    assert frame[codec.HEADER.size - 32:codec.HEADER.size] == hashlib.sha256(target.astype("<u2").tobytes()).digest()


@pytest.mark.parametrize("row_profile", ["quartile", "residual", "tail"])
@pytest.mark.parametrize("bin_profile", ["quantile", "lloyd"])
def test_all_profiles_encode_losslessly_and_default_is_explicit_default(row_profile, bin_profile):
    target, ref = _sample()
    frame = codec.encode(target, ref, row_profile=row_profile, bin_profile=bin_profile)
    assert np.array_equal(codec.decode(frame, ref), target)
    if row_profile == "quartile" and bin_profile == "quantile":
        assert frame == codec.encode(target, ref)


def test_profile_fit_labels_and_thresholds_match_small_golden_fixture():
    target, ref = _sample()
    expected_labels = {
        "quartile": [8, 1, 0, 9, 6, 15, 6, 15],
        "residual": [14, 10, 12, 6, 2, 0, 4, 8],
        "tail": [11, 3, 3, 11, 7, 15, 7, 15],
    }
    expected_quantile = [16447, 16485, 16515, 16596, 16628, 16712, 16805, 17239, 48549, 48730, 48865, 48914, 49004, 49045, 49061]
    expected_lloyd = [16365, 16414, 16466, 16489, 16537, 16626, 16730, 17168, 48670, 48872, 48986, 49042, 49053, 49079, 49142]
    for row_profile, labels_expected in expected_labels.items():
        for bin_profile, thresholds_expected in (("quantile", expected_quantile), ("lloyd", expected_lloyd)):
            labels, flips, thresholds = codec._fit(target, ref, row_profile, bin_profile)
            assert labels.tolist() == labels_expected
            assert flips.shape == (8,)
            assert thresholds.tolist() == thresholds_expected


def test_mixed_fit_and_predict_return_exact_bf16_words():
    rng = np.random.default_rng(3)
    up = _words(rng.normal(size=(4, 19)))
    gate = _words(rng.normal(size=(4, 19)))
    uf = (up.astype(np.uint32) << 16).view(np.float32)
    gf = (gate.astype(np.uint32) << 16).view(np.float32)
    target = _words(uf * np.float32(0.4) + gf * np.float32(-0.2) + np.float32(0.1))
    coeff = fit_mixed_coefficients(target, up, gate)
    assert coeff.dtype == np.uint16 and coeff.shape == (4, 3)
    result = predict(up, gate, coeff)
    assert result.dtype == np.uint16 and result.shape == target.shape
    assert np.all(((result.astype(np.uint32) & 0x7F80) != 0x7F80))


def test_mixed_fit_uses_constant_mean_for_singular_rows_and_rejects_nonfinite():
    up = _words(np.ones((2, 8)))
    gate = _words(np.ones((2, 8)) * 2)
    target = _words(np.stack((np.linspace(-1, 1, 8), np.ones(8) * 0.25)))
    coeff = fit_mixed_coefficients(target, up, gate)
    assert coeff[1, 0] == 0 and coeff[1, 1] == 0
    assert np.array_equal(predict(up, gate, coeff)[1], np.full(8, coeff[1, 2], dtype=np.uint16))
    bad = target.copy()
    bad[0, 0] = np.uint16(0x7F80)
    with pytest.raises(PredictiveMixError, match="nonfinite"):
        fit_mixed_coefficients(bad, up, gate)


def test_gpu_frame_index_reports_target_as_source_and_reference_separately():
    import bitexact_predictive_gpu as gpu

    target, ref = _sample()
    frame = codec.encode(target, ref)
    indexed = gpu._read_frame(frame, ref, stride=16)
    assert indexed["source_sha256"] == hashlib.sha256(target.astype("<u2").tobytes()).hexdigest()
    assert indexed["reference_sha256"] == hashlib.sha256(ref.astype("<u2").tobytes()).hexdigest()


def test_every_bf16_word_roundtrips_exactly():
    target = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    ref = np.roll(target, 7, axis=1)
    decoded = codec.decode(codec.encode(target, ref), ref)
    assert decoded.shape == target.shape and decoded.dtype == np.dtype("<u2")
    assert np.array_equal(decoded, target)


def test_correlated_zero_tiny_random_and_nonfinite_word_roundtrips():
    rng = np.random.default_rng(4431)
    ref = rng.integers(0, 65536, (32, 48), dtype=np.uint16)
    targets = (ref.copy(), ref ^ np.uint16(0x8000), np.zeros_like(ref),
               np.ones_like(ref), rng.integers(0, 65536, ref.shape, dtype=np.uint16))
    for target in targets:
        assert np.array_equal(codec.decode(codec.encode(target, ref), ref), target)
    target = np.array([[0x7F80, 0xFF80, 0x7FC1, 0xFFC1, 0x0001, 0x8001]], dtype=np.uint16)
    ref = np.array([[0x3F80, 0xBF80, 0x7F81, 0xFF81, 0x0000, 0x8000]], dtype=np.uint16)
    assert np.array_equal(codec.decode(codec.encode(target, ref), ref), target)


def test_conditional_and_raw_word_forms_are_selected_and_decode():
    ref = np.full((128, 512), 0x3F80, dtype=np.uint16)
    target = np.full_like(ref, 0x3F83)
    frame = codec.encode(target, ref)
    assert codec.HEADER.unpack_from(frame)[2] == 1
    assert np.array_equal(codec.decode(frame, ref), target)
    target = np.array([[0x8001]], dtype=np.uint16)
    ref = np.array([[0x7F80]], dtype=np.uint16)
    frame = codec.encode(target, ref)
    assert codec.HEADER.unpack_from(frame)[2] == 2
    assert np.array_equal(codec.decode(frame, ref), target)


def test_wrong_reference_checksum_and_truncated_sections_are_rejected():
    rng = np.random.default_rng(4)
    target = rng.integers(0, 65536, (8, 16), dtype=np.uint16)
    ref = rng.integers(0, 65536, target.shape, dtype=np.uint16)
    frame = codec.encode(target, ref)
    with pytest.raises(codec.CodecError, match="reference"):
        codec.decode(frame, np.roll(ref, 1, axis=0))
    with pytest.raises(codec.CodecError, match="checksum"):
        codec.decode(frame[:-1], ref)
    corrupt = bytearray(frame)
    corrupt[-33] ^= 1
    with pytest.raises(codec.CodecError, match="checksum"):
        codec.decode(corrupt, ref)
    truncated = bytearray(frame[:-1])
    truncated[-32:] = hashlib.sha256(truncated[:-32]).digest()
    with pytest.raises(codec.CodecError):
        codec.decode(truncated, ref)


def test_ordinal_transform_chunked_bit_order_and_raw5_padding():
    ref = np.array([[0x3F80, 0xBF80], [0x4000, 0xC000]], dtype=np.uint16)
    flips = np.array([True, False])
    expected = ref.copy()
    expected[0] ^= np.uint16(0x8000)
    assert np.array_equal(codec._aligned_keys(ref, flips), codec._ord(expected))
    values = np.array([0, 1, 17, 31, 3, 29, 8, 24, 5, 12, 27], dtype=np.uint8)
    bits = ((values[:, None] >> np.arange(4, -1, -1)) & 1).astype(np.uint8).ravel()
    assert codec._packbits(values, 5) == np.packbits(bits, bitorder="big").tobytes()
    large = np.resize(values, (1 << 20) + 24)
    packed_bits = ((large[:, None] >> np.arange(4, -1, -1)) & 1).astype(np.uint8).ravel()
    assert codec._packbits(large, 5) == np.packbits(packed_bits, bitorder="big").tobytes()

    target = np.array([[0x3F81, 0x3F82, 0x3F83]], dtype=np.uint16)
    ref = np.array([[0x3F80, 0x3F80, 0x3F80]], dtype=np.uint16)
    labels, flips, thresholds = codec._fit(target, ref)
    keys = codec._aligned_keys(ref, flips)
    bins = np.searchsorted(thresholds, keys, side="right").astype(np.uint16)
    main = ((target >> 15) & 1) * 1024 + ((target >> 7) & 255) * 4 + ((target & 127) >> 5)
    main_freq, main_stream = codec._rans_encode(main.ravel(), (labels[:, None] * 16 + bins).ravel(), 256, 2048)
    metadata = __import__("zlib").compress(labels.tobytes() + flips.tobytes() + thresholds.tobytes(), 6)
    raw = codec.HEADER.pack(b"PPCX", 1, 0, 0, *target.shape, len(metadata), len(main_freq), len(main_stream),
                            hashlib.sha256(ref.astype("<u2").tobytes()).digest(),
                            hashlib.sha256(target.astype("<u2").tobytes()).digest())
    raw += metadata + main_freq + main_stream + codec._packbits((target & 31).ravel(), 5)
    damaged = bytearray(raw)
    damaged[-1] |= 1
    damaged = bytes(damaged) + hashlib.sha256(damaged).digest()
    with pytest.raises(codec.CodecError, match="padding"):
        codec.decode(damaged, ref)


def test_nonmonotone_predictor_thresholds_are_rejected():
    target = np.full((128, 512), 0x3F83, dtype=np.uint16)
    ref = np.full_like(target, 0x3F80)
    frame = codec.encode(target, ref)
    fields = list(codec.HEADER.unpack_from(frame))
    payload = frame[codec.HEADER.size:-32]
    meta_len = fields[6]
    metadata = bytearray(codec._inflate(payload[:meta_len], target.shape[0] * 2 + 30))
    thresholds = np.frombuffer(metadata, dtype="<u2", offset=target.shape[0] * 2).copy()
    thresholds[0], thresholds[1] = 100, 0
    metadata[target.shape[0] * 2:] = thresholds.astype("<u2").tobytes()
    packed = __import__("zlib").compress(metadata, 6)
    fields[6] = len(packed)
    head = codec.HEADER.pack(*fields)
    body = head + packed + payload[meta_len:]
    malformed = body + hashlib.sha256(body).digest()
    with pytest.raises(codec.CodecError, match="monotone"):
        codec.decode(malformed, ref)
