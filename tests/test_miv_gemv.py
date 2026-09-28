"""CUDA tests for MIV-GEMV (``glc_serve.miv_gemv``): M-invariance, accuracy, bias, dispatch.

Skipped without CUDA (the kernel is CUDA-only); the full 10k-input falsifier on
the real 27B weights is ``scripts/miv_gemv_bench.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("MIV-GEMV is a CUDA kernel", allow_module_level=True)

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "release"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch.nn as nn  # noqa: E402

from glc_serve import miv_gemv as mg  # noqa: E402

SHAPES = [(48, 5120), (96, 5120), (1024, 5120), (5120, 6144), (777, 1032), (64, 17408)]
CONFIGS = [(1, 2), (2, 4), (4, 8), (8, 4)]


def _bits(t):
    return t.view(torch.int16)


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("cfg", CONFIGS)
def test_rows_of_m_calls_equal_m1_calls(n, k, cfg):
    g = torch.Generator(device="cpu").manual_seed(n * 7 + k)
    w = (torch.randn(n, k, generator=g) * 0.05).to(torch.bfloat16).cuda()
    x = (torch.randn(64, k, generator=g) * 3).to(torch.bfloat16).cuda()
    y1 = torch.cat([mg.miv_gemv(x[i:i + 1], w, config=cfg) for i in range(64)])
    for m in range(2, 9):
        for c in range(0, 64, m):
            idx = [(c + j) % 64 for j in range(m)]
            ym = mg.miv_gemv(x[idx], w, config=cfg)
            assert torch.equal(_bits(ym), _bits(y1[idx])), (m, c)


@pytest.mark.parametrize("m", [1, 3, 8])
def test_accuracy_and_bias_single_rounding(m):
    g = torch.Generator(device="cpu").manual_seed(m)
    n, k = 1000, 5120
    w = (torch.randn(n, k, generator=g) * 0.05).to(torch.bfloat16).cuda()
    b = torch.randn(n, generator=g).to(torch.bfloat16).cuda()
    x = torch.randn(m, k, generator=g).to(torch.bfloat16).cuda()
    for cfg in CONFIGS:
        y = mg.miv_gemv(x, w, b, config=cfg)
        ref = x.double() @ w.double().T + b.double()
        # one bf16 rounding of an fp32 sum: within 1 bf16 ulp of the exact value (+ fp32 noise)
        ulp = torch.finfo(torch.bfloat16).eps * ref.abs().clamp_min(1e-3)
        assert bool(((y.double() - ref).abs() <= ulp + 1e-3).all()), cfg
        # bias path is also M-invariant
        y1 = torch.cat([mg.miv_gemv(x[i:i + 1], w, b, config=cfg) for i in range(m)])
        assert torch.equal(_bits(y), _bits(y1))


def test_linear_dispatch_and_fallback():
    lin = nn.Linear(512, 256, bias=True).to(torch.bfloat16).cuda()
    x8 = torch.randn(2, 4, 512, dtype=torch.bfloat16, device="cuda")     # M = 8
    x9 = torch.randn(9, 512, dtype=torch.bfloat16, device="cuda")        # M = 9 -> F.linear
    mg.STATS.miv_by_m.clear()
    mg.STATS.fallback_by_reason.clear()
    ref9 = lin(x9)
    assert mg.install_dense(lin) == 1
    y8 = lin(x8)
    assert y8.shape == (2, 4, 256)
    assert torch.equal(_bits(lin(x9)), _bits(ref9))
    assert mg.STATS.miv_by_m.get(8) == 1
    assert mg.STATS.fallback_by_reason.get("m_gt8") == 1
    mg.set_enabled(False)
    try:
        assert torch.equal(_bits(lin(x8)), _bits(torch.nn.functional.linear(x8, lin.weight, lin.bias)))
    finally:
        mg.set_enabled(True)


def test_exact_hook_default_off():
    from glc_serve import modules

    assert modules._EXACT_LINEAR is None
    mg.install_tbe_exact()
    try:
        assert modules._EXACT_LINEAR is mg.linear
    finally:
        mg.uninstall_tbe_exact()
    assert modules._EXACT_LINEAR is None
