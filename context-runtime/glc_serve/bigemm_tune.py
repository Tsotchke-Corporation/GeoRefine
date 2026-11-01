#!/usr/bin/env python3
"""The BI-GEMM row-tile TUNE TABLE: which weight shapes launch the 128-row tile, and at which M.

Pure data and pure arithmetic -- no torch -- so the dispatch rule (``glc_serve.bigemm``) and the
step-time model that projects it (``glc_serve.bidec_policy``) read ONE table and cannot drift.

Why a table and not the rc7 rule
--------------------------------
rc7's ``bigemm.tile_for`` launched the 128-row tile for EVERY weight whenever ``m > 64`` and the
engine cap allowed it, on the strength of one measured shape (gate|up).  Run 6a measured all
five shapes the tile gate carries (T2, ``receipts/tile_gate/t2_timing.json``) and the rule is
wrong for some of them: at M=128 the GDN input projection (``wpr_16480x5120``) is **0.931x**
under the 128 tile -- the 128-row tile makes that GEMM SLOWER -- and at M=32/64 every shape is
slower under 128 (0.53x .. 1.01x).  So rc8 launches 128 only where a measured row says it wins,
keyed by the exact (N, K) of the weight and by the row bucket the step runs at (the engine pads
M up to a bucket in ``bidec.BUCKETS``, and the buckets above the knee -- 96, 128, 192, 256 -- are
exactly the M values T2 measured).  A shape or bucket that is not in the table keeps the 64-row
tile, i.e. rc6/rc7-at-tile-64 behaviour, so **the 128 tile can never make a shape slower than
the measurement says it is at 64**.

Bits: T1 of the same run compared BI(X, tile=64) against BI(X, tile=128) for all five shapes,
both formats (bf16, tbe) and M in {1,31,32,33,63,64,65,97,127,128,129}: 332,689,280 elements,
0 differing.  So per-shape dispatch is a speed decision only -- whichever tile a shape launches,
the output bits are the same.

Provenance (every number below is copied, not rounded further, from that file):
  run 6a, campaign jack-bench-20261003, NVIDIA RTX PRO 6000 Blackwell Server Edition,
  driver 580.173.02, torch 2.14.1+cu130, glc_loader 1.2.0rc7 (wheel sha256 5ae7d618...c8e657),
  ``.icc/evidence/sixth-gpu-run-rc7-20261004/receipts/tile_gate/t2_timing.json``
  (sha256 ef43abcabfafc802a22f101ef84a8feb697cdaeb6c488dbfe2aa6705e9594434, branch
  claude/first-gpu-run-20261003), 30 reps per point after 3 warm-up calls, CUDA events.

Caveat, stated because the table cannot: T2 timed the **bf16** descriptor only (``Lin(W)``);
the served weights are TBE.  The decision is applied to every format of the same (N, K),
which is exact for the bits (T1 covered tbe) and an assumption for the speed.  A TBE T2 row
is the measurement that would retire the assumption.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

Arch = Tuple[int, int, int]

#: Measured us per call, ``{(name, N, K): {M: (tile64_us, tile128_us)}}``.
_T2_RTXPRO6000_ROWS: Dict[Tuple[str, int, int], Dict[int, Tuple[float, float]]] = {
    ("gate_up", 34816, 5120): {32: (214.1, 237.7), 64: (246.9, 243.7), 96: (434.7, 253.5),
                               128: (448.7, 274.9), 192: (693.0, 494.8), 256: (950.2, 520.1)},
    ("wpr_16480x5120", 16480, 5120): {32: (89.9, 140.0), 64: (103.7, 145.1), 96: (160.8, 151.8),
                                      128: (154.6, 166.0), 192: (248.3, 231.6),
                                      256: (325.7, 256.0)},
    ("down", 5120, 17408): {32: (96.7, 114.8), 64: (107.7, 123.1), 96: (192.4, 131.2),
                            128: (206.8, 143.7), 192: (345.0, 272.8), 256: (457.3, 287.3)},
    ("qkv", 7680, 5120): {32: (24.8, 47.2), 64: (41.0, 55.3), 96: (65.9, 59.5),
                          128: (76.0, 67.7), 192: (131.2, 116.9), 256: (201.2, 128.9)},
    ("lm_head", 151936, 5120): {32: (1005.7, 1059.5), 64: (1027.7, 1061.8),
                                96: (2059.4, 1073.2), 128: (2066.5, 1103.9),
                                192: (3174.0, 2208.9), 256: (4245.8, 2251.5)},
}
# The measurements are specific to this exact GPU architecture and SM count.  Keeping the
# architecture in the outer key prevents an unrelated device from inheriting Blackwell timings.
T2_RTXPRO6000_US: Dict[Arch, Dict[Tuple[str, int, int], Dict[int, Tuple[float, float]]]] = {
    (12, 0, 188): _T2_RTXPRO6000_ROWS,
}
T2_PROVENANCE = ("run6a rc7 RTX PRO 6000 Blackwell SE, receipts/tile_gate/t2_timing.json "
                 "sha256 ef43abcabfafc802a22f101ef84a8feb697cdaeb6c488dbfe2aa6705e9594434, bf16")

#: How many times each measured shape runs in one forward of the served 27B (64 decoder layers:
#: 48 Gated-DeltaNet + 16 attention, from the server receipt's geometry ``n_gdn``/``n_att``).
#: Used by the step-time model to weight the table into "weight passes per step"; shapes the
#: table does not carry (GDN out_proj, attention o_proj) are outside the weighting, which is
#: why the model treats the result as a RELATIVE pass count, not as microseconds.
STACK_COUNT: Dict[str, int] = {"gate_up": 64, "down": 64, "wpr_16480x5120": 48, "qkv": 16,
                               "lm_head": 1}

#: Row buckets above the 64-row knee the table decides.  Must be a subset of bidec.BUCKETS.
DECIDED_BUCKETS = (96, 128, 192, 256)


def table_for(arch: Optional[Arch]):
    """Return measurements for an exact ``(compute major, minor, SM count)`` device."""
    if arch is None:
        return {}
    return T2_RTXPRO6000_US.get(tuple(map(int, arch)), {})


def _bucket_at_least(m: int) -> Optional[int]:
    for b in DECIDED_BUCKETS:
        if m <= b:
            return b
    return None


def wins_128(N: int, K: int, m: int,
             table: Dict[Tuple[str, int, int], Dict[int, Tuple[float, float]]] = _T2_RTXPRO6000_ROWS
             ) -> bool:
    """True iff a MEASURED row says the 128-row tile is strictly faster for this (N, K) at the
    bucket an ``m``-row GEMM runs at.  Unmeasured shape, unmeasured bucket, ``m <= 64``: False."""
    m = int(m)
    if m <= 64:
        return False
    b = _bucket_at_least(m)
    if b is None:
        return False
    for (_name, n_, k_), rows in table.items():
        if n_ == int(N) and k_ == int(K):
            t = rows.get(b)
            return bool(t) and t[1] < t[0]
    return False


def per_shape_choice(table=_T2_RTXPRO6000_ROWS) -> Dict[str, Dict[int, int]]:
    """``{shape: {bucket: tile}}`` -- the decision the dispatch makes, for /health and docs."""
    out: Dict[str, Dict[int, int]] = {}
    for (name, n_, k_), rows in table.items():
        out[name] = {b: (128 if wins_128(n_, k_, b, table) else 64) for b in sorted(rows) if b > 64}
    return out


def stack_us(m_bucket: int, mode: str = "64", table=_T2_RTXPRO6000_ROWS) -> float:
    """Measured GEMM-stack microseconds for one forward at row bucket ``m_bucket``.

    ``mode`` "64": every shape at the 64-row tile (rc6, and rc7 at --gemm-tile 64).
    ``mode`` "rc7-128": every shape at 128 above the knee (rc7's --gemm-tile 128 rule).
    ``mode`` "per-shape": this module's rule (rc8 --gemm-tile 128).
    """
    tot = 0.0
    for (name, n_, k_), rows in table.items():
        t64, t128 = rows[int(m_bucket)]
        if mode == "64" or int(m_bucket) <= 64:
            t = t64
        elif mode == "rc7-128":
            t = t128
        elif mode == "per-shape":
            t = t128 if wins_128(n_, k_, m_bucket, table) else t64
        else:
            raise ValueError(f"mode must be 64 / rc7-128 / per-shape, got {mode!r}")
        tot += STACK_COUNT[name] * t
    return tot


def passes(m_bucket: int, mode: str = "64") -> float:
    """Weight passes a step at row bucket ``m_bucket`` pays, RELATIVE to one 64-row pass at
    M=64, from the measured table.  1.0 at or below the knee by definition (the chat column the
    step-time model is fitted on lives there); above it, e.g. 1.79 at 128 rows on the 64 tile
    and 1.24 under the per-shape rule."""
    if int(m_bucket) <= 64:
        return 1.0
    return stack_us(m_bucket, mode) / stack_us(64, "64")


__all__ = ["T2_RTXPRO6000_US", "T2_PROVENANCE", "STACK_COUNT", "DECIDED_BUCKETS", "table_for", "wins_128",
           "per_shape_choice", "stack_us", "passes"]
