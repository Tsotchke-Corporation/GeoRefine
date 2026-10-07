import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bitexact_predictive_mix import PredictiveMixError, predict


def _words(values):
    return (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def test_separate_ieee_operations_truncate_instead_of_rounding():
    up = _words([[1.0, 1.0]])
    gate = _words([[0.01171875, 0.01171875]])
    coeff = _words([[1.0, 1.0, 0.0]])
    assert predict(up, gate, coeff).tolist() == [[0x3F81, 0x3F81]]


def test_negative_values_and_rowwise_coefficients():
    up = _words([[1.5], [-1.5]])
    gate = _words([[-2.0], [2.0]])
    coeff = _words([[0.5, -0.25, 0.125], [0.5, -0.25, -0.125]])
    assert predict(up, gate, coeff).tolist() == [[0x3FB0], [0xBFB0]]


def test_shape_and_uint16_dtype_validation():
    values = _words([[1.0], [2.0]])
    coeff = _words([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    with pytest.raises(PredictiveMixError, match="shape mismatch"):
        predict(values, values[:1], coeff)
    with pytest.raises(PredictiveMixError, match="shape"):
        predict(values, values, coeff[:1])
    with pytest.raises(PredictiveMixError, match="uint16"):
        predict(values.astype(np.int16), values, coeff)


def test_nonfinite_inputs_coefficients_and_output_are_rejected():
    up = _words([[1.0]])
    gate = _words([[1.0]])
    coeff = _words([[1.0, 1.0, 0.0]])
    with pytest.raises(PredictiveMixError, match="nonfinite"):
        predict(_words([[np.nan]]), gate, coeff)
    with pytest.raises(PredictiveMixError, match="nonfinite"):
        predict(up, gate, _words([[np.inf, 1.0, 0.0]]))
    huge = np.array([[0x7F7F]], dtype=np.uint16)
    with pytest.raises(PredictiveMixError, match="nonfinite"):
        predict(huge, up, np.array([[0x7F7F, 0, 0]], dtype=np.uint16))
