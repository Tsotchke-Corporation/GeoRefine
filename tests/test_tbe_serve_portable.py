"""Compressed resident weights with a portable PyTorch forward path."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "release"))

from glc_loader.tbe_container import encode_tbe  # noqa: E402
from glc_loader.tbe_serve_portable import PortableTBELinear  # noqa: E402


@pytest.mark.parametrize("layout", ["mma16", "flat64"])
def test_compressed_linear_matches_dense_without_resident_weight(layout):
    torch.manual_seed(27)
    weight = (torch.randn(32, 128) * 0.02).to(torch.bfloat16)
    bias = (torch.randn(32) * 0.01).to(torch.bfloat16)
    x = (torch.randn(3, 128) * 0.02).to(torch.bfloat16)
    encoded = encode_tbe(weight, layout=layout)
    entry = {"name": "linear.weight", "shape": [32, 128],
             "coded_shape": [32, 128], "layout": layout, "mode": encoded.mode,
             "base": encoded.base, "tiles": encoded.tiles,
             "escapes": encoded.escapes, "superblock": encoded.superblock}
    arrays = {p: getattr(encoded, p) for p in ("planes", "smb", "esc", "sbbase")}
    layer = PortableTBELinear(entry, arrays, bias=True)
    layer.bias = torch.nn.Parameter(bias, requires_grad=False)
    before = layer.resident_weight_bytes
    assert before < weight.numel() * weight.element_size()
    assert "weight" not in dict(layer.named_parameters())
    expected = torch.nn.functional.linear(x, weight, bias)
    for _ in range(2):
        actual = layer(x)
        assert torch.equal(actual, expected)
        assert layer.resident_weight_bytes == before
        assert "weight" not in dict(layer.named_parameters())
