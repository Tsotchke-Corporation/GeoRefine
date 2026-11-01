#!/usr/bin/env python3
"""Prefill-aware scheduling policy, and the CPU simulation that scores it.

Not to be confused with ``bidec_admission.py``, which is the ``bidec_iface.AdmissionPolicy``
side (which waiting request is admitted next: fairness, priority classes, preemption).  This
module answers a different question -- how many ALREADY admitted requests may prefill at once,
and how large a chunk each may take.  Both landed as ``bidec_policy.py`` on separate branches;
the 2026-10-04 serving merge kept both under distinct names.

Why this module exists
----------------------
The third GPU run (rc4, campaign ``jack-bench-20261003``) measured the ctx8k class for the
first time.  GLC's long-context **decode** holds flat where llama.cpp collapses -- 5.04x
aggregate throughput at N=32 -- but GLC's long-context **prefill** is 4.5-5.3x WORSE: TTFT
p50 109.5 s at N=16 and 209.2 s at N=32, against llama.cpp's 20.8 s and 46.1 s.  Receipts:
``.icc/evidence/third-gpu-run-rc4-20261003/bench/glc_ctx8k/summary_c{16,32}.json``.

What the receipts say the cause is
----------------------------------
Three candidates were on the table.  The arithmetic below is the engine's own, and it picks
one:

(a) *the row budget starves prompts*.  ``Batcher.step`` fills decode rows first
    (``max_decode_rows`` = 64, measured 64 in ``health.json``) out of ``max_rows_step`` = 256,
    leaving >= 192 rows for prompt chunks.  192 rows/step at ~76 ms/step would prefill
    16 x 7432 = 118,909 prompt tokens in ~47 s of steps, not 109 s.  The row budget alone
    does NOT predict the measurement.

(b) *the work-list capacity binds*.  The attention work list holds one entry per (row, KV
    page) pair, so a prompt row at position p costs ``p // 256 + 1`` entries while a decode
    row costs the same but there are at most ``pages_total`` of those.  Capacity is
    ``g_cap = pages_total + max_rows + 8`` = 800 + 256 + 8 = **1064** entries for the run-3
    server (``--pages 800 --max-rows 256``).  At position 7,400 one prompt token costs ~30
    entries, so ONE STEP can carry ~35 prompt tokens, falling to ~1,064/32 = 33 at 8k --
    against a 256-row budget it never gets to use.  ``_fit_work_list`` halves chunks until
    they fit and counts the trims.  Total entries to deliver N prompts of length L is
    ``N * (L^2 / (2 * PAGE) + L)``; at N=16, L=7432 that is 1.845M entries, 1,734 steps at
    1,064 entries/step, and at ~76 ms/step (the measured N=64 chat step) **132 s** -- which
    brackets the measured 109 s p50 / 197 s p99.  At N=32 the same arithmetic gives 264 s
    against a measured 209 s p50 / 390 s p99, and the measured ratio 209/109 = 1.92 matches
    the predicted 2.00: aggregate prefill rate is CONSTANT in N, which is the signature of a
    per-step cap, not of per-sequence cost.

(c) *admission admits all N at once*.  It does not, and cannot, at N=32: ``Batcher.add``
    reserves ``(len(prompt) + max_new + PAGE) // PAGE + 1`` ~ 33 pages per ctx8k request and
    ``_admit`` refuses once ``reserved + need > pages_total`` = 800, so at most ~24 of the 32
    are resident.  (c) is refuted as the primary cause by the engine's own admission rule;
    it does, however, mean the N=32 column is also page-pool limited.

So: **(b), amplified by (a)**.  And the reason a small step is so expensive is in the same
receipts: ``streamed_bytes_per_step`` = 35.9 GB at N=64, i.e. a step streams the weight set
whatever it carries, so step time is nearly independent of how many tokens are in the step.
A cap that limits a step to ~33 prompt tokens therefore throws away almost all of the step.

What run 5 then measured, and what it corrects above (2026-10-04)
-----------------------------------------------------------------
Everything above survives except the last sentence of it, and the last sentence is the one
the projection rested on.  A step streams the weight set **once per 64-row BI-GEMM tile**, not
once per step.  So step time is independent of the row count only WITHIN a tile, and
``--prefill-priority 0.25`` -- which raised the cap 1,064 -> 5,096 and the step from ~44 rows
to ~176 -- bought three weight re-decodes instead of one.  Run 5 measured the consequence:
ctx8k TTFT p50 improved 2.03x / 1.63x at N=16/32 (not the projected 8.2x / 6.2x), the prefill
TOKEN rate improved only 606 -> 808 tok/s = 1.33x against a 4.79x cap increase, and per-user
decode fell 13.25 -> 8.12 and 13.23 -> 4.82 tok/s, because the decode rows riding in those
steps were stretched by the same factor.  Receipts:
``.icc/evidence/fifth-gpu-run-rc6-20261004/bench/glc_ctx8k{,_pp0.25}/summary_c{16,32}.json``.

The 1.33x is exactly the tile-occupancy ratio (176/192) / (44/64), which says the gain was
entirely filling a partial tile and none of it was new throughput.  Total row throughput is
~850 rows/s measured and nearly constant above the knee: **decode and prefill share a fixed
row budget, and only the tile HEIGHT can raise it.**  Hence ``StepTimeModel.tile_rows``,
``PrefillPolicy.row_budget_tiles`` and ``rc7_policy`` below, and the full write-up in
``docs/serving/PREFILL_BUDGET_20261004.md``.  Lever 1 above is downgraded: the work-list cap
is necessary (without it ``_fit_work_list`` trims every chunk to a sliver) but it is NOT
sufficient and on its own it is harmful.

What run 6a then measured, and what it corrects above (rc8, 2026-10-04)
-----------------------------------------------------------------------
Two of the paragraphs above do not survive.  (1) "total row throughput is nearly a constant":
it followed from charging the WHOLE fixed step cost once per tile, which run 6a refutes -- its
steps never left one tile and cost the UNSCALED chat-column time, which rc7's 0.8934 scale
contradicts.  ``StepTimeModel.tile_share`` (0.6348, calibrated on one row) is the share of the
fixed cost a weight pass carries; the rest is per step, so rows/s RISES above the knee and the
row budget is a decode-LATENCY budget.  (2) rc7 scored every row on run 5's prompt pool; run 6a
served a different pool (glc-bench drew it from the package's own files), which is the whole of
its measured TTFT "regression".  ``simulate(arrivals=...)`` now replays each measured row's own
requests (``bidec_traces``).  Worst error over ten measured rows: 27 % -> 11 % (TTFT p50),
13 % -> 2 % (per-user decode).  ``docs/serving/RC8_FIXES_20261004.md``.

The policy
----------
1. **A prefill-sized work list.**  ``g_cap`` is sized for the decode case
   (``pages_total + max_rows + 8``).  The buffers it sizes (``pacc``, ``pml``) cost
   ``NQ * 256 * 4`` and ``NQ * 2 * 4`` bytes per entry -- 26.4 MB total at 1,064 entries on
   the run-3 server, so ~24.8 KB/entry.  Raising the cap to 8x costs ~185 MB of the 36 GB
   headroom and multiplies the prompt tokens a step can carry by the same factor.  This is
   the whole of the gain; it is a BUFFER SIZE, not a hardware limit.
2. **Prefill-aware admission.**  Cap the number of sequences prefilling at once so each gets
   at least one whole aligned block (``PAGE`` tokens) of chunk, instead of all of them
   getting a 1-token sliver.  Fewer, bigger chunks cost the same total entries and finish
   individual prompts sooner, which is what TTFT measures.
3. **A decode floor.**  Reserve entries for decode rows first, so raising prefill pressure can
   never stall the decode column that the 5.04x result rests on.
4. **A bigger chunk when decode rows are idle**, bounded by the entry budget.

5. **A total row budget of whole BI-GEMM tiles** (``row_budget_tiles``, run 5).  Decode rows
   first, prompt chunks filling the remainder, total capped at one tile: per-user decode is
   pinned at the single-tile step time and every row decode does not want goes to prefill
   instead of being wasted.

6. **rc8: a row budget in ROWS with the work list sized** (``rc8_ctx8k_policy``: priority
   0.25, 56 rows).  PROJECTED on identical arrivals of every measured ctx8k workload: TTFT p50
   0.76-0.90x of the control with per-user decode 1.003-1.09x.

Everything here is pure arithmetic: no torch, no CUDA, importable on any machine.  The
projected TTFT numbers it produces are PROJECTED.  ``reproduce_measured()`` scores the model
against every ctx8k row runs 3, 4, 5 and 6a measured, each replayed on its own requests, with ONE
parameter (``tile_share``) calibrated on ONE of them; worst error 11 % TTFT p50, 2 % per-user
decode.  Running the module prints that table, rc7's beside it, and the policy projections.
"""
from __future__ import annotations

import math
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

PAGE = 256                      # must equal glc_serve.bidec.PAGE
#: must equal glc_serve.bidec.BUCKETS: the row counts a step is padded up to (one CUDA graph each)
BUCKETS: Tuple[int, ...] = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)


def bucket_for(m: int, buckets: Sequence[int] = BUCKETS) -> int:
    """The padded row count a step of ``m`` rows runs at (``glc_serve.bidec.bucket_for``)."""
    for b in buckets:
        if m <= b:
            return int(b)
    return int(buckets[-1])
BYTES_PER_WORK_ITEM = 24_800    # pacc + pml, measured from run-3 /health (26,353,152 B / 1,064)


# --------------------------------------------------------------------------- work-list maths
def items_for_span(pos0: int, n: int) -> int:
    """Work-list entries for ``n`` rows starting at position ``pos0`` -- the engine's formula
    (``glc_serve.bidec.BatchDecoder.items_for_span``), repeated here so the policy can be
    reasoned about without a GPU in the room."""
    pos0, n = int(pos0), int(n)
    if n <= 0:
        return 0

    def g(m: int) -> int:                 # sum_{k=0}^{m-1} floor(k / PAGE), in closed form
        q, r = divmod(m, PAGE)
        return PAGE * q * (q - 1) // 2 + r * q

    return n + g(pos0 + n) - g(pos0)


def items_to_prefill(length: int) -> int:
    """Total work-list entries a whole prompt of ``length`` tokens costs, however chunked.
    Chunking never changes it: entry count is per (token, page) pair."""
    return items_for_span(0, int(length))


def rc4_item_cap(pages_total: int, max_rows: int) -> int:
    """The shipped cap: ``pages_total + max_rows + 8`` (bidec.py)."""
    return int(pages_total) + int(max_rows) + 8


def plan_item_cap(pages_total: int, max_rows: int, max_ctx: int, *,
                  prefill_priority: float = 1.0,
                  byte_budget: Optional[int] = None,
                  bytes_per_item: int = BYTES_PER_WORK_ITEM) -> Dict[str, int]:
    """How large the step work list should be for prefill, and what it costs in bytes.

    ``prefill_priority`` 0 reproduces rc4 exactly.  1 gives the cap that lets one prompt
    chunk of ``max_rows`` rows sit at the deepest supported context -- the point past which a
    bigger buffer buys nothing because the row budget binds instead.
    """
    base = rc4_item_cap(pages_total, max_rows)
    pages_at_max_ctx = (int(max_ctx) + PAGE - 1) // PAGE
    ideal = int(pages_total) + int(max_rows) * pages_at_max_ctx + 8
    pr = max(0.0, min(1.0, float(prefill_priority)))
    want = int(round(base + pr * (ideal - base)))
    if byte_budget is not None:
        want = min(want, max(base, int(byte_budget) // int(bytes_per_item)))
    cap = max(base, want)
    return {"item_cap": cap, "rc4_item_cap": base, "ideal_item_cap": ideal,
            "bytes": cap * int(bytes_per_item),
            "extra_bytes_vs_rc4": (cap - base) * int(bytes_per_item)}


@dataclass
class PrefillPolicy:
    """The knobs.  Defaults reproduce rc4 byte for byte (``prefill_priority = 0``)."""
    prefill_priority: float = 0.0
    item_cap: Optional[int] = None          # None -> rc4 formula
    max_prefill_seqs: int = 0               # 0 -> unlimited (rc4)
    decode_floor_items: int = 0             # entries reserved for decode rows
    prefill_chunk: int = 256
    chunk_align: int = PAGE
    max_rows_step: int = 256
    max_decode_rows: int = 64
    #: BI-GEMM tile height.  Must equal the kernel's weight re-decode granularity.
    row_tile: int = 64
    #: How many whole tiles a step may fill.  0 = off, i.e. the shipped rc6 behaviour, where
    #: the row budget is ``max_rows_step`` (256 = 4 tiles) and a prefill-heavy step therefore
    #: pays the weight set up to four times while the decode rows riding in it are stretched by
    #: the same factor.  1 is the budgeted policy: TOTAL rows per step (decode first, prefill
    #: filling the remainder) are capped at one tile, so per-user decode latency is the
    #: single-tile step time no matter how much prefill pressure there is, and every row of the
    #: tile that decode does not want goes to prefill instead of being wasted.
    row_budget_tiles: int = 0
    #: rc8: the budget in ROWS; when > 0 it takes precedence over ``row_budget_tiles``
    #: (``bidec.Batcher.row_budget_rows``, ``--row-budget-rows``).
    row_budget_rows: int = 0

    def row_budget(self) -> int:
        """Total rows (decode + prefill) this policy allows in one step."""
        if self.row_budget_rows > 0:
            return max(1, min(int(self.max_rows_step), int(self.row_budget_rows)))
        if self.row_budget_tiles <= 0:
            return int(self.max_rows_step)
        return max(1, min(int(self.max_rows_step),
                          int(self.row_tile) * int(self.row_budget_tiles)))

    def effective_item_cap(self, pages_total: int, max_rows: int, max_ctx: int) -> int:
        if self.item_cap:
            return int(self.item_cap)
        return plan_item_cap(pages_total, max_rows, max_ctx,
                             prefill_priority=self.prefill_priority)["item_cap"]

    def prefill_seq_limit(self, item_cap: int, mean_pos: float) -> int:
        """How many sequences should prefill at once so each gets a whole aligned block.

        With ``item_cap`` entries and a token costing ``mean_pos / PAGE + 1`` entries deep in
        a context, a block of ``chunk_align`` tokens costs ``align * (mean_pos / PAGE + 1)``.
        Admitting more prefills than that is how everyone ends up with a sliver.
        """
        if self.max_prefill_seqs:
            return int(self.max_prefill_seqs)
        per_token = mean_pos / PAGE + 1.0
        block = max(1.0, self.chunk_align * per_token)
        return max(1, int(item_cap // block))


# --------------------------------------------------------------------------- step-time model
@dataclass
class StepTimeModel:
    """``step_s = fixed * tiles + per_row * rows + per_item * items``, ``tiles = ceil(rows/64)``.

    ``fixed`` and ``per_row`` come from the six GLC chat rows (where a step is one row per
    decoding user and the aggregate tok/s is the step rate).  ``per_item`` is a residual term,
    0 by default.

    **The ``tiles`` factor is the run-5 correction and it is the whole of the fix.**  Up to rc6
    this class was ``fixed + per_row * rows`` -- linear.  It was fitted on the chat column,
    where the row count is 1..64 and therefore ``tiles`` is 1 at every point, so the fit could
    not see the tile term and the extrapolation to a 256-row prefill step was free to be
    arbitrarily optimistic.  It was: with ``tile_rows = 0`` this model projects the
    ``--prefill-priority 0.25`` arm at ctx8k N=32 as 91 aggregate decode tok/s where run 5
    measured 35.4, and TTFT p50 30.0 s where run 5 measured 118.2 s
    (``.icc/evidence/fifth-gpu-run-rc6-20261004/bench/glc_ctx8k_pp0.25/summary_c32.json``).

    The physics the tile term encodes was already written down in ``bidec.Batcher.__init__``
    and in the 128-row patch spec: the BI-GEMM weight set is re-decoded **once per 64-row
    tile** (measured gate|up 237.9 us for M=1..32, 261.2 us at M=64, 479.0 us at M=128), so a
    step's weight traffic -- the 35.9 GB/step the server reports as
    ``streamed_bytes_per_step`` -- is paid ``ceil(rows / 64)`` times, not once.  Step cost is a
    STAIRCASE in rows, which makes total row throughput very nearly a CONSTANT (~850 rows/s on
    the RTX PRO 6000) that decode and prefill must share.  No scheduling policy can raise it;
    a policy can only stop wasting it (run 5's control arm ran a 44-row step inside a 64-row
    tile, i.e. 31% of the tile thrown away) and choose the split.

    ``tile_rows = 0`` restores the refuted linear form, and is kept only so a test can assert
    the defect.
    """
    fixed_s: float
    per_row_s: float
    per_item_s: float = 0.0
    tile_rows: int = 64
    #: Global multiplier, calibrated on ONE measured deep-context row.  The chat column fixes
    #: the model's SHAPE (a tile term and a row term) but not its absolute scale at ctx8k
    #: depth: a deep step also reads ~1,000 work-list entries of paged KV that a 40-token chat
    #: step does not, and the chat column cannot separate that term from the row term because
    #: there items are ~2x rows at every point.  One scalar absorbs it.
    scale: float = 1.0
    #: **rc8 (run 6a).**  The fraction of ``fixed_s`` -- the 1-row step cost the chat column
    #: measures -- that is paid once per WEIGHT PASS; the rest (``1 - tile_share``) is paid once
    #: per STEP whatever its row count (norms, GDN state traffic, attention launch, host).  rc7
    #: assumed 1.0: the whole fixed cost re-paid per 64-row tile.  Run 6a ran every step inside
    #: one tile and measured the single-tile step at the UNSCALED chat-column cost (per-user
    #: decode -2% against the model with no free parameter), which rc7's 0.8934 scale
    #: contradicts; the scale had been absorbing an over-charged multi-tile step.  Calibrated on
    #: ONE row (``TILE_SHARE_CALIBRATION_ROW``); every other measured row is held out.
    tile_share: float = 1.0
    #: How a step's weight passes are counted.  "structural": ``ceil(rows / tile_rows)``, rc7's
    #: rule.  "t2-64" / "t2-per-shape" / "t2-rc7-128": the MEASURED GEMM-stack cost from the
    #: run-6a T2 tune table (``glc_serve.bigemm_tune.passes``), relative to one 64-row pass, at
    #: the row bucket the step runs at -- the 64 tile, rc8's per-shape 128 dispatch, or rc7's
    #: shape-blind 128 rule.  The T2 modes make the tile-64 and tile-128 projections read ONE
    #: measured table.
    pass_mode: str = "structural"
    #: Charge ``per_row_s`` on the PADDED row bucket (what the CUDA graph runs) instead of the
    #: live rows.  The ctx8k receipts cannot tell the two apart once a scale is free (both fit
    #: every measured row within 10 %), so a policy is only claimed if it wins under BOTH.
    padded_rows: bool = False

    def tiles(self, rows: int) -> int:
        """Weight re-decode passes a step of ``rows`` rows pays (structural count)."""
        rows = max(0, int(rows))
        if self.tile_rows <= 0:
            return 1
        return max(1, (rows + self.tile_rows - 1) // self.tile_rows)

    def passes(self, rows: int) -> float:
        """Weight passes for a ``rows``-row step under ``pass_mode`` (1.0 at or below 64 rows
        in the T2 modes, by the definition of the unit)."""
        if self.pass_mode == "structural":
            return float(self.tiles(rows))
        from glc_serve import bigemm_tune as _tune

        mode = {"t2-64": "64", "t2-per-shape": "per-shape", "t2-rc7-128": "rc7-128"}[self.pass_mode]
        b = bucket_for(max(1, int(rows)))
        if b <= 256:
            return _tune.passes(b, mode)
        return _tune.passes(256, mode) * b / 256.0           # past the measured range: linear

    def step_s(self, rows: int, items: int) -> float:
        r = bucket_for(max(1, int(rows))) if (self.padded_rows and rows > 0) else max(0, rows)
        phi = float(self.tile_share)
        return self.scale * (self.fixed_s * ((1.0 - phi) + phi * self.passes(rows))
                             + self.per_row_s * r
                             + self.per_item_s * max(0, items))

    def with_tile(self, tile_rows: int) -> "StepTimeModel":
        """The same card, the same weights, a different BI-GEMM row tile.

        This is how the 128-row kernel is PROJECTED: the tile term is ``fixed_s`` once per
        weight re-decode pass, and the kernel change makes one pass serve ``tile_rows`` rows
        instead of 64.  Nothing else in the model moves -- ``per_row_s`` is MMA plus the
        activation traffic, which is per row either way, and ``per_item_s`` is paged-KV work
        the GEMM tile does not touch.  So a one-line substitution is the honest projection,
        and the only thing it assumes is the thing the GPU gate T2 measures.

        Caveat, stated because the simulator cannot see it: a WG=2 CTA always runs a FULL
        128-row MMA, so a partially filled 128 tile pays ``per_row_s`` for its padding rows.
        ``step_s`` below is therefore a LOWER bound for a partial tile; it is exact for a full
        one, and the budgeted policy (``row_budget_tiles``) fills whole tiles by construction.
        """
        if self.pass_mode != "structural":
            return dataclasses.replace(self, tile_rows=int(tile_rows),
                                       pass_mode="t2-per-shape" if int(tile_rows) >= 128
                                       else "t2-64")
        return dataclasses.replace(self, tile_rows=int(tile_rows))

    def rows_per_s(self, rows: int) -> float:
        """Total rows (decode + prefill) the engine retires per second at this step size."""
        dt = self.step_s(rows, 0)
        return (rows / dt) if dt > 0 else 0.0

    @classmethod
    def fit_chat(cls, rows_tok_s: Sequence[Tuple[int, float]]) -> "StepTimeModel":
        """Least squares on (decode rows, aggregate tok/s): one step emits one token per row,
        so ``step_s = rows / agg_tok_s``."""
        xs = [float(n) for n, _ in rows_tok_s]
        ys = [float(n) / float(t) for n, t in rows_tok_s]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        var = sum((x - mx) ** 2 for x in xs)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var if var else 0.0
        return cls(fixed_s=my - b * mx, per_row_s=b)


#: The six GLC chat rows of run 3 (N, aggregate tok/s) -- THIRD_GPU_RUN_RC4_20261003.md.
RUN3_CHAT_ROWS: Tuple[Tuple[int, float], ...] = ((1, 24.8), (4, 94.7), (8, 181.5),
                                                 (16, 334.9), (32, 565.4), (64, 837.3))
#: The two GLC ctx8k rows of run 3: N -> (mean prompt tokens, TTFT p50 s, decode tok/s/user).
RUN3_CTX8K: Dict[int, Tuple[float, float, float]] = {16: (7431.8, 109.465, 13.59),
                                                     32: (7221.0, 209.154, 13.71)}


# --------------------------------------------------------------------------- the simulator
@dataclass
class _Seq:
    arrive: float
    length: int
    max_new: int
    fed: int = 0
    out: int = 0
    ttft: Optional[float] = None
    pages: int = 0
    dec_s: float = 0.0                # wall time this sequence spent in decode steps
    dec_n: int = 0                    # decode tokens it emitted in them
    t_first: Optional[float] = None   # first-token time (end of the step that completed prefill)
    t_last: Optional[float] = None    # last-token time


@dataclass
class SimResult:
    ttft_p50: float
    ttft_p99: float
    requests_done: int
    first_tokens: int
    decode_tokens: int
    steps: int
    wall_s: float
    decode_tok_s: float
    prefill_tok_s: float
    trims: int
    item_cap: int
    ttfts: List[float] = field(default_factory=list)
    #: tokens/s a single decoding stream sees, computed EXACTLY as glc-bench computes
    #: ``per_stream_decode_tok_s.p50``: per request ``(n - 1) / (t_last - t_first)`` over
    #: requests with n > 8 tokens, then the p50 across requests (bench_serve.pct).
    per_user_decode_tok_s: float = 0.0
    rows_per_step_mean: float = 0.0
    row_budget: int = 0
    #: the row-weighted mean over steps (rc7's per-user statistic), kept for comparison.
    stream_decode_tok_s_p50: float = 0.0
    ttft_mean: float = 0.0
    #: glc-bench ``steady_tok_s``: tokens emitted inside [warm_s, send_window_s] / that window.
    steady_tok_s: float = 0.0
    #: glc-bench ``aggregate_tok_s``: completion tokens / (last finish - first send).
    aggregate_tok_s: float = 0.0
    multi_pass_steps: int = 0         # steps whose row count crossed one 64-row tile
    align_shortened: int = 0          # prompt chunks _chunk_len shortened to a block boundary


def _pct(xs: Sequence[float], q: float) -> float:
    """glc-bench's percentile (``_bench/bench_serve.pct``): index ``round(q * (n - 1))``.

    rc7's simulator used ``ceil(q * n) - 1``, which for an even sample picks the rank BELOW
    the bench's median -- on run 6a's N=16 row that is 111.1 s against the bench's 128.3 s,
    a 13 % "model error" that is only a definition."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def simulate(*, n_users: int, policy: PrefillPolicy, model: StepTimeModel,
             prompt_len: int = 0, max_new: int = 600, pages_total: int = 800,
             max_rows: int = 256, max_ctx: int = 16384, horizon_s: float = 240.0,
             closed_loop: bool = True, send_window_s: Optional[float] = None,
             prompt_pool: Optional[Sequence[int]] = None,
             trace: Optional[Sequence[Tuple[int, int]]] = None,
             arrivals: Optional[Sequence[Tuple]] = None,
             warm_s: float = 20.0, max_active: int = 64,
             buckets: Sequence[int] = BUCKETS) -> SimResult:
    """A transcription of ``bidec.Batcher.step`` (speculation off) driven by a workload.

    Per step, in the engine's order: ``_admit`` (FIFO, page reservation, the prefill-
    concurrency cap at priority > 0); decode rows first (one per decoding sequence, capped by
    ``max_decode_rows`` and the row budget); prompt chunks by ``_chunk_len`` (the whole
    remaining budget to the first prefilling sequence, ``chunk_align`` snapping, never to
    nothing); ``_fit_work_list`` (newest chunk halved, block-snapped, until the work list --
    padding rows included, as ``BatchDecoder.items_for`` counts them -- fits the cap); then
    one step of ``model.step_s`` time.

    The workload, most faithful first:
      ``arrivals`` -- ``(t_send_s, prompt_tokens, max_tokens, ...)`` per request, replayed
        open-loop at the measured send times (``bidec_traces.TRACES``).  Used to SCORE the
        model: same requests, same order, same send times as the card saw.
      ``trace`` -- ``(prompt_tokens, max_tokens)`` in admission order, closed loop: ``n_users``
        at t=0, then each completion inside the send window pulls the next item -- glc-bench's
        own worker loop.  Used to PROJECT a policy on a measured workload.
      ``prompt_pool`` / ``prompt_len`` -- rc7's form (one pool, one ``max_new``), kept.
    """
    cap = policy.effective_item_cap(pages_total, max_rows, max_ctx)
    row_budget = min(policy.row_budget(), int(max_rows))
    pp = float(policy.prefill_priority)
    align = int(policy.chunk_align)
    items_pool: List[Tuple[int, int]]
    if trace:
        items_pool = [(int(L), int(m)) for L, m in trace]
    else:
        pool = [int(x) for x in (prompt_pool or (int(prompt_len),)) if int(x) > 0] or [int(prompt_len)]
        items_pool = [(L, int(max_new)) for L in pool]
    _next = [0]

    def _mk(t: float, L: Optional[int] = None, m: Optional[int] = None) -> _Seq:
        if L is None:
            L, m = items_pool[_next[0] % len(items_pool)]
            _next[0] += 1
        q = _Seq(arrive=float(t), length=int(L), max_new=int(m))
        q.pages = (q.length + q.max_new + PAGE) // PAGE + 1
        return q

    if arrivals is not None:
        waiting = [_mk(a[0], a[1], a[2]) for a in sorted(arrivals, key=lambda a: a[0])]
        resend = False
    else:
        waiting = [_mk(0.0) for _ in range(n_users)]
        resend = bool(closed_loop)
    active: List[_Seq] = []
    finished: List[_Seq] = []
    reserved = 0
    now = 0.0
    steps = trims = first_tokens = decode_tokens = 0
    rows_sum = 0
    dec_weighted_s = 0.0
    in_window = 0
    multi = shortened = 0
    win_hi = send_window_s if send_window_s is not None else float("inf")

    def _items(chunks, dec_items: int, rows: int) -> int:
        return (dec_items + sum(items_for_span(s.fed, c) for s, c in chunks)
                + (bucket_for(rows, buckets) - rows if rows else 0))

    while now < horizon_s and (waiting or active):
        # ---- Batcher._admit
        pref = [s for s in active if s.fed < s.length]
        mean_pos = (sum(s.fed for s in pref) / len(pref)) if pref else 0.0
        limit = policy.prefill_seq_limit(cap, mean_pos) if pp else 10 ** 9
        prefilling = len(pref)
        while (waiting and waiting[0].arrive <= now and len(active) < max_active
               and prefilling < limit and reserved + waiting[0].pages <= pages_total):
            s = waiting.pop(0)
            reserved += s.pages
            active.append(s)
            prefilling += 1
        if not active:
            if waiting:
                now = max(now, waiting[0].arrive)    # the server idles until the next request
                continue
            break

        # ---- decode rows first
        budget = row_budget
        dec_budget = policy.max_decode_rows
        dec_rows: List[_Seq] = []
        dec_items = 0
        for s in active:
            if s.fed >= s.length and budget > 0 and dec_budget > 0:
                dec_rows.append(s)
                dec_items += items_for_span(s.length + s.out - 1, 1)
                budget -= 1
                dec_budget -= 1
        # ---- then prompt chunks: Batcher._chunk_len
        chunks: List[List] = []
        for s in active:
            if s.fed >= s.length or budget <= 0:
                continue
            left = s.length - s.fed
            want = policy.prefill_chunk
            if pp and not dec_rows:
                want = max(want, budget)
            c = min(want, budget, left)
            if align > 1 and c < left:
                end = s.fed + c
                snapped = end - (end % align)
                if snapped > s.fed:                   # never snap a chunk away to nothing
                    if snapped - s.fed < c:
                        shortened += 1
                    c = snapped - s.fed
            chunks.append([s, c])
            budget -= c
        rows = len(dec_rows) + sum(c for _, c in chunks)
        # ---- Batcher._fit_work_list: halve the newest chunk until the list fits
        while chunks and _items(chunks, dec_items, rows) > cap:
            s, c = chunks[-1]
            if c > 1:
                n = max(1, c // 2)
                if align > 1:
                    end = s.fed + n
                    snapped = end - (end % align)
                    if snapped > s.fed:
                        n = snapped - s.fed
                chunks[-1][1] = n
                rows -= (c - n)
            else:
                chunks.pop()
                rows -= c
            trims += 1
        if rows == 0:
            if waiting:
                now = max(now, waiting[0].arrive)
                continue
            break

        # ---- one step
        items = _items(chunks, dec_items, rows)
        dt = model.step_s(rows, items)
        now += dt
        steps += 1
        rows_sum += rows
        if rows > 64:
            multi += 1
        dec_weighted_s += dt * len(dec_rows)
        emitted = 0
        for s, c in chunks:
            s.fed += c
            if s.fed >= s.length:                 # prefill complete -> first token now
                s.out = 1
                s.t_first = s.t_last = now
                s.ttft = now - s.arrive
                first_tokens += 1
                emitted += 1
        for s in dec_rows:
            s.out += 1
            s.t_last = now
            s.dec_s += dt
            s.dec_n += 1
            decode_tokens += 1
            emitted += 1
        if warm_s <= now <= win_hi:
            in_window += emitted
        for s in list(active):
            if s.out >= s.max_new:
                active.remove(s)
                reserved -= s.pages
                finished.append(s)
                if resend and (send_window_s is None or now < send_window_s):
                    waiting.append(_mk(now))

    ttfts = [s.ttft for s in finished if s.ttft is not None]
    rates = [(s.max_new - 1) / (s.t_last - s.t_first) for s in finished
             if s.max_new > 8 and s.t_last is not None and s.t_first is not None
             and s.t_last > s.t_first]
    row_w = [s.dec_n / s.dec_s for s in finished if s.dec_n >= 2 and s.dec_s > 0]
    prompt_tokens = sum(s.length for s in finished) + sum(s.length for s in active
                                                          if s.fed >= s.length)
    t_first_send = min([s.arrive for s in finished] or [0.0])
    window = (win_hi - warm_s) if win_hi != float("inf") else 0.0
    return SimResult(ttft_p50=_pct(ttfts, 0.50), ttft_p99=_pct(ttfts, 0.99),
                     requests_done=len(finished), first_tokens=first_tokens,
                     decode_tokens=decode_tokens, steps=steps, wall_s=now,
                     decode_tok_s=(decode_tokens + first_tokens) / now if now else 0.0,
                     prefill_tok_s=prompt_tokens / now if now else 0.0,
                     trims=trims, item_cap=cap, ttfts=ttfts,
                     per_user_decode_tok_s=_pct(rates, 0.50) if rates else 0.0,
                     rows_per_step_mean=(rows_sum / steps) if steps else 0.0,
                     row_budget=row_budget,
                     stream_decode_tok_s_p50=(decode_tokens / dec_weighted_s
                                              if dec_weighted_s else 0.0),
                     ttft_mean=(sum(ttfts) / len(ttfts)) if ttfts else float("nan"),
                     steady_tok_s=(in_window / window) if window > 0 else 0.0,
                     aggregate_tok_s=(sum(s.max_new for s in finished)
                                      / max(1e-9, now - t_first_send)),
                     multi_pass_steps=multi, align_shortened=shortened)


def run3_model(per_item_s: float = 0.0) -> StepTimeModel:
    """The step-time model fitted to run 3's chat column."""
    m = StepTimeModel.fit_chat(RUN3_CHAT_ROWS)
    m.per_item_s = per_item_s
    return m


def rc4_policy(**kw) -> PrefillPolicy:
    """rc4's shipped behaviour: unlimited prefill concurrency, decode-sized work list."""
    return PrefillPolicy(prefill_priority=0.0, **kw)


def rc5_policy(pages_total: int = 800, max_rows: int = 256, max_ctx: int = 16384,
               prefill_priority: float = 1.0, byte_budget: Optional[int] = None,
               **kw) -> PrefillPolicy:
    """The policy this module proposes, sized from the same shapes the server reports."""
    plan = plan_item_cap(pages_total, max_rows, max_ctx, prefill_priority=prefill_priority,
                         byte_budget=byte_budget)
    return PrefillPolicy(prefill_priority=prefill_priority, item_cap=plan["item_cap"],
                         decode_floor_items=pages_total, **kw)


def rc8_policy(pages_total: int = 800, max_rows: int = 256, max_ctx: int = 16384,
               prefill_priority: float = 0.25, row_budget_tiles: int = 1,
               row_tile: int = 128, byte_budget: Optional[int] = None,
               **kw) -> PrefillPolicy:
    """rc7's budgeted policy on the **128-row** BI-GEMM kernel.

    The policy arithmetic is rc7's unchanged; the only input that moves is ``row_tile``, which
    must equal the kernel's weight re-decode granularity -- that is the whole contract between
    this module and ``bi_gemm.cuh``.  One tile is then 128 rows, so a one-tile step carries
    twice the rows for (projected) one weight pass, and the ~850 rows/s ceiling that runs 3-5
    measured is the thing that is supposed to move.  Pair with
    ``StepTimeModel.with_tile(128)``; a simulation that raises ``row_tile`` and leaves
    ``tile_rows`` at 64 is modelling a two-pass step and will show no gain.

    PROJECTED, not measured: the speedup and the CUDA compile are both gated on the card
    (scripts/batchserve/bi_gate_tile128.py, T2 and T4).
    """
    plan = plan_item_cap(pages_total, max_rows, max_ctx, prefill_priority=prefill_priority,
                         byte_budget=byte_budget)
    return PrefillPolicy(prefill_priority=prefill_priority, item_cap=plan["item_cap"],
                         decode_floor_items=pages_total, row_tile=row_tile,
                         row_budget_tiles=row_budget_tiles, **kw)


def rc7_policy(pages_total: int = 800, max_rows: int = 256, max_ctx: int = 16384,
               prefill_priority: float = 0.25, row_budget_tiles: int = 1,
               row_tile: int = 64, byte_budget: Optional[int] = None,
               **kw) -> PrefillPolicy:
    """**The budgeted chunked-prefill policy.**  rc5's prefill-sized work list, plus a TOTAL
    row budget of ``row_budget_tiles`` whole BI-GEMM tiles.

    Why this and not rc5's: the work list is not the only thing a prefill-heavy step spends.
    Run 5 raised the work-list cap 1,064 -> 5,096 (4.79x) and measured the prefill token rate
    rise only 606 -> 808 tok/s (1.33x), because the step that carried those entries also
    carried 176 rows = 3 tiles and therefore paid the 35.9 GB weight set three times.  The
    1.33x is exactly the tile-occupancy ratio: 44/64 = 0.69 of a tile in control against
    176/192 = 0.917 in the treatment, 0.917/0.69 = 1.33.  The decode rows riding in that step
    were stretched by the same 3x, which is the measured halving.

    Under a one-tile budget the step time is the single-tile step time (~75 ms measured, stable
    to +-3% across runs 3, 4 and 5 control), so per-user decode is pinned at its control value,
    and the rows decode does not take go to prefill rather than being wasted -- which is the
    only free prefill throughput there is on a 64-row kernel.  The remaining lever is the tile
    height itself, which is a CUDA change (see the 128-row acceptance spec), not a policy.
    """
    plan = plan_item_cap(pages_total, max_rows, max_ctx, prefill_priority=prefill_priority,
                         byte_budget=byte_budget)
    return PrefillPolicy(prefill_priority=prefill_priority, item_cap=plan["item_cap"],
                         decode_floor_items=pages_total, row_tile=row_tile,
                         row_budget_tiles=row_budget_tiles, **kw)


# --------------------------------------------------------------- measured GPU receipts
#: Every GLC ctx8k row the campaign has measured on the RTX PRO 6000, as
#: ``name -> (N, mean prompt tokens, TTFT p50 s, per-stream decode tok/s p50, prefill priority)``.
#: Sources, all in ``.icc/evidence/<run>/bench/glc_ctx8k*/summary_c{16,32}.json`` on branch
#: ``claude/first-gpu-run-20261003``: run 3 = ``third-gpu-run-rc4-20261003``, run 4 =
#: ``fourth-gpu-run-rc5-20261003``, run 5 = ``fifth-gpu-run-rc6-20261004`` (``glc_ctx8k``
#: control, ``glc_ctx8k_pp0.25`` treatment), run 6a = ``sixth-gpu-run-rc7-20261004``
#: (``--row-budget-tiles 1`` at tile 64, ``--prefill-priority`` 0.0).
MEASURED_CTX8K: Dict[str, Tuple[int, float, float, float, float]] = {
    "run3/N16":          (16, 7431.8, 109.465, 13.59, 0.0),
    "run3/N32":          (32, 7547.1, 209.154, 13.71, 0.0),
    "run4/N16":          (16, 7292.6, 107.380, 13.34, 0.0),
    "run4/N32":          (32, 7394.7, 219.481, 13.54, 0.0),
    "run5/N16":          (16, 6891.9, 103.762, 13.25, 0.0),
    "run5/N32":          (32, 7167.1, 192.376, 13.23, 0.0),
    "run5/N16/pp0.25":   (16, 6822.8,  51.131,  8.12, 0.25),
    "run5/N32/pp0.25":   (32, 7140.4, 118.207,  4.82, 0.25),
    "run6a/N16/rb1":     (16, 7381.7, 128.259, 15.70, 0.0),
    "run6a/N32/rb1":     (32, 7203.2, 219.107, 15.50, 0.0),
}
#: The row budget (in 64-row tiles) each measured row ran with; absent = 0 (none).
MEASURED_ROW_BUDGET_TILES: Dict[str, int] = {"run6a/N16/rb1": 1, "run6a/N32/rb1": 1}

#: Mean completion tokens per request across the ctx8k rows (the bench's own prompt pool).
CTX8K_MAX_NEW = 600
#: glc-bench ``--duration`` for the ctx8k rows: workers keep sending for this long, then drain.
CTX8K_SEND_WINDOW_S = 90.0
#: glc-bench ``--warm``: the steady-tok/s window opens this long after the first send.
CTX8K_WARM_S = 20.0
#: The ctx8k prompt pool, as the four run-5 per-request JSONLs report it (prompt_tokens, with
#: multiplicity): ``.icc/evidence/fifth-gpu-run-rc6-20261004/bench/glc_ctx8k*/requests_c*.jsonl``.
#: rc7 cycled THIS pool for every measured row.  It is run 5's pool only: glc-bench builds the
#: pool from the installed package's own source files, so every release that adds or edits a
#: file draws a different pool (rc6 and rc7 share 1 of 40 ctx8k prompts).  Kept for the rc7
#: (legacy) scoring; the rc8 scoring replays ``bidec_traces.TRACES`` instead.
CTX8K_PROMPT_POOL: Tuple[int, ...] = (
    5212, 5212, 5655, 5655, 5658, 5828, 6130, 6134, 6184, 6493, 6496, 6497, 6604, 6613,
    6994, 7030, 7046, 7116, 7269, 7302, 7489, 8064, 8132, 8135, 8246, 8246, 8249, 8250,
    9424, 10605, 10607, 10669,
)


def trace_from_requests_jsonl(path) -> Tuple[Tuple[float, int, int, float], ...]:
    """A glc-bench ``requests_c<N>.jsonl`` -> ``(t_send_s, prompt_tokens, max_tokens, ttft_s)``
    per request in SERVER ADMISSION order: the t=0 burst ordered by first-token time (prefill is
    serial under every policy measured so far, so first-token order is admission order), then
    the re-sends by send time.  This is how ``bidec_traces`` was generated."""
    import json as _json

    rs = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                r = _json.loads(line)
                if r.get("usage") and r.get("t_first_chunk"):
                    rs.append(r)
    t0 = min(r["t_send"] for r in rs)
    init = sorted([r for r in rs if r["t_send"] - t0 < 1.0], key=lambda r: r["t_first_chunk"])
    later = sorted([r for r in rs if r["t_send"] - t0 >= 1.0], key=lambda r: r["t_send"])
    out = [(0.0, r["usage"]["prompt_tokens"], r["max_tokens"],
            round(r["t_first_chunk"] - r["t_send"], 3)) for r in init]
    out += [(round(r["t_send"] - t0, 3), r["usage"]["prompt_tokens"], r["max_tokens"],
             round(r["t_first_chunk"] - r["t_send"], 3)) for r in later]
    return tuple(out)


# ------------------------------------------------------------- legacy (rc7) scoring, kept
#: rc7's ONE scalar, calibrated on run 3 / N=16 with the run-5 pool cycled for every row and the
#: whole fixed cost re-paid per tile (``tile_share`` 1.0).  Run 6a refutes it: a single-tile
#: step costs the UNSCALED chat-column time.  Kept so the before/after table can be recomputed.
SCALE_CALIBRATION_ROW = "run3/N16"
SCALE = 0.8934


def calibrate_scale(*, tile_rows: int = 64, row: str = SCALE_CALIBRATION_ROW,
                    lo: float = 0.2, hi: float = 3.0, iters: int = 48, **kw) -> float:
    """rc7's calibration of ``StepTimeModel.scale`` (legacy; see ``calibrate_tile_share``)."""
    _n, _plen, _ttft, per_user, _pp = MEASURED_CTX8K[row]
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _sim_row(row, scale=mid, tile_rows=tile_rows, **kw)["per_user_sim"] > per_user:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _sim_row(name: str, *, scale: float, tile_rows: int, horizon_s: float = 600.0,
             pages_total: int = 800, max_rows: int = 256, max_ctx: int = 16384,
             row_budget_tiles: Optional[int] = None, prefill_priority: Optional[float] = None,
             ) -> Dict[str, float]:
    """rc7's scoring of one measured row: run 5's prompt pool cycled, ``CTX8K_MAX_NEW`` tokens
    for every request, the structural tile staircase with ``tile_share`` 1.0 (legacy)."""
    model = run3_model()
    model.tile_rows = int(tile_rows)
    model.scale = float(scale)
    n, plen, ttft, per_user, pp = MEASURED_CTX8K[name]
    if prefill_priority is not None:
        pp = float(prefill_priority)
    rbt = MEASURED_ROW_BUDGET_TILES.get(name, 0) if row_budget_tiles is None else row_budget_tiles
    pol = _row_policy(pp, rbt, 0, pages_total=pages_total, max_rows=max_rows, max_ctx=max_ctx)
    r = simulate(n_users=n, prompt_len=int(round(plen)), max_new=CTX8K_MAX_NEW, policy=pol,
                 model=model, pages_total=pages_total, max_rows=max_rows, max_ctx=max_ctx,
                 horizon_s=horizon_s, send_window_s=CTX8K_SEND_WINDOW_S,
                 prompt_pool=CTX8K_PROMPT_POOL)
    return {"ttft_p50_measured": ttft, "ttft_p50_sim": r.ttft_p50,
            "ttft_err": (r.ttft_p50 - ttft) / ttft,
            "per_user_measured": per_user, "per_user_sim": r.per_user_decode_tok_s,
            "per_user_err": (r.per_user_decode_tok_s - per_user) / per_user,
            "rows_per_step_mean": r.rows_per_step_mean, "prefill_tok_s": r.prefill_tok_s,
            "row_budget": r.row_budget, "ttft_p99_sim": r.ttft_p99,
            "stream_p50_sim": r.stream_decode_tok_s_p50}


def reproduce_measured_rc7(*, tile_rows: int = 64, scale: float = SCALE,
                           horizon_s: float = 600.0) -> Dict[str, Dict[str, float]]:
    """The rc7 model (legacy) scored against every measured row, on rc7's pooled workload."""
    return {name: _sim_row(name, scale=scale, tile_rows=tile_rows, horizon_s=horizon_s)
            for name in MEASURED_CTX8K}


# ---------------------------------------------------------------- the rc8 model (run 6a)
#: The one parameter calibrated on the ctx8k class -- ``StepTimeModel.tile_share`` -- and the
#: one row it is calibrated on (per-user decode p50, replayed on that row's own trace).  The
#: other nine measured rows, including both run-6a rows and both arms of run 5's A/B, are held
#: out.  ``calibrate_tile_share()`` recomputes it; a test asserts it has not drifted.
TILE_SHARE_CALIBRATION_ROW = "run5/N16"
TILE_SHARE = 0.6348
#: The padded-row alternative (``StepTimeModel.padded_rows``) needs a scale as well; it is
#: calibrated on the same row with ``tile_share`` held at ``TILE_SHARE``.
PADDED_SCALE = 0.9266


def _row_policy(pp: float, row_budget_tiles: int = 0, row_budget_rows: int = 0, *,
                pages_total: int = 800, max_rows: int = 256, max_ctx: int = 16384,
                row_tile: int = 64, chunk_align: int = PAGE) -> PrefillPolicy:
    """The policy a measured row ran under, as ``bidec_serve`` builds it: ``chunk_align`` =
    PAGE unconditionally, the item cap from ``plan_item_cap(prefill_priority)``.

    rc7's ``_sim_row`` built a budget-only row (priority 0) through ``rc7_policy`` with a
    priority of 1e-9 to dodge a falsy check, which switched ON the priority>0 admission cap and
    chunk widening the engine leaves OFF at 0.0.  Here the priority is passed as measured."""
    cap = plan_item_cap(pages_total, max_rows, max_ctx, prefill_priority=pp)["item_cap"]
    return PrefillPolicy(prefill_priority=float(pp), item_cap=cap, max_rows_step=max_rows,
                         chunk_align=chunk_align, row_tile=row_tile,
                         row_budget_tiles=int(row_budget_tiles),
                         row_budget_rows=int(row_budget_rows))


def rc8_model(*, tile_share: Optional[float] = None, padded: bool = False,
              scale: Optional[float] = None, tile: int = 64) -> StepTimeModel:
    """The corrected step-time model.

    ``step_s = scale * (fixed * ((1 - tile_share) + tile_share * passes(rows))
                        + per_row * rows)``

    ``fixed_s`` and ``per_row_s`` are run 3's chat column, UNSCALED (rc7 scaled them by 0.8934;
    run 6a's single-tile steps measure the unscaled cost).  ``passes`` is the measured T2
    GEMM-stack cost at the step's row bucket relative to one 64-row pass
    (``glc_serve.bigemm_tune``), at the 64 tile or -- ``tile=128`` -- under rc8's per-shape
    dispatch.  ``tile_share`` is the single ctx8k-calibrated parameter."""
    m = run3_model()
    m.tile_share = TILE_SHARE if tile_share is None else float(tile_share)
    m.pass_mode = "t2-per-shape" if int(tile) >= 128 else "t2-64"
    m.tile_rows = int(tile)
    m.padded_rows = bool(padded)
    m.scale = (PADDED_SCALE if padded else 1.0) if scale is None else float(scale)
    return m


def measured_trace(name: str):
    from glc_serve.bidec_traces import TRACES

    return TRACES[name]


def score_row(name: str, model: StepTimeModel, *, policy: Optional[PrefillPolicy] = None,
              arrivals=None, horizon_s: float = 900.0) -> Dict[str, float]:
    """Replay a measured row's own requests at their measured send times under ``model`` and
    score TTFT (p50 and mean, glc-bench's definitions) and per-user decode p50."""
    n, _plen, ttft_p50, per_user, pp = MEASURED_CTX8K[name]
    tr = measured_trace(name)
    if policy is None:
        policy = _row_policy(pp, MEASURED_ROW_BUDGET_TILES.get(name, 0))
    r = simulate(n_users=n, policy=policy, model=model, arrivals=arrivals or tr,
                 horizon_s=horizon_s, warm_s=CTX8K_WARM_S)
    meas_mean = sum(x[3] for x in tr) / len(tr)
    return {"ttft_p50_measured": ttft_p50, "ttft_p50_sim": r.ttft_p50,
            "ttft_err": (r.ttft_p50 - ttft_p50) / ttft_p50,
            "ttft_mean_measured": meas_mean, "ttft_mean_sim": r.ttft_mean,
            "ttft_mean_err": (r.ttft_mean - meas_mean) / meas_mean,
            "per_user_measured": per_user, "per_user_sim": r.per_user_decode_tok_s,
            "per_user_err": (r.per_user_decode_tok_s - per_user) / per_user,
            "rows_per_step_mean": r.rows_per_step_mean, "multi_pass_steps": r.multi_pass_steps,
            "steps": r.steps, "requests": r.requests_done, "row_budget": r.row_budget}


def calibrate_tile_share(*, row: str = TILE_SHARE_CALIBRATION_ROW, lo: float = 0.0,
                         hi: float = 1.0, iters: int = 30) -> float:
    """Bisect ``tile_share`` so the replay of ``row`` reproduces its measured per-user decode.
    Monotone: a larger share makes every multi-pass step slower, so fewer tok/s."""
    want = MEASURED_CTX8K[row][3]
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if score_row(row, rc8_model(tile_share=mid))["per_user_sim"] > want:
            lo = mid
        else:
            hi = mid
    return round(0.5 * (lo + hi), 4)


def calibrate_padded_scale(*, row: str = TILE_SHARE_CALIBRATION_ROW, lo: float = 0.5,
                           hi: float = 1.5, iters: int = 30) -> float:
    """The padded-row alternative's scale, on the same one row, ``tile_share`` held fixed."""
    want = MEASURED_CTX8K[row][3]
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if score_row(row, rc8_model(padded=True, scale=mid))["per_user_sim"] > want:
            lo = mid
        else:
            hi = mid
    return round(0.5 * (lo + hi), 4)


def reproduce_measured(*, model: Optional[StepTimeModel] = None,
                       horizon_s: float = 900.0) -> Dict[str, Dict[str, float]]:
    """Score the rc8 model against every measured ctx8k row, each replayed on its OWN requests.

    One parameter (``TILE_SHARE``) is calibrated, on ``TILE_SHARE_CALIBRATION_ROW``; the chat
    column supplies ``fixed_s`` / ``per_row_s`` and T2 supplies the pass counts.  Returns, per
    row, simulated and measured TTFT p50 / mean and per-user decode, with relative errors."""
    model = model or rc8_model()
    return {name: score_row(name, model, horizon_s=horizon_s) for name in MEASURED_CTX8K}


# ------------------------------------------------------------------ rc8 policy + projections
#: The rc8 ctx8k policy (docs/serving/RC8_FIXES_20261004.md section 4): the work list sized for
#: prefill (``--prefill-priority 0.25``: 1,064 -> 5,096 entries, +100 MB) so the prompt chunk is
#: no longer trimmed to ~33 rows at 7k depth, and the TOTAL rows per step capped at 56
#: (``--row-budget-rows 56``) so the step -- which every decode row rides in -- stays below the
#: control's mean step time.  PROJECTED; the GPU arm that measures it is in the doc.
RC8_PREFILL_PRIORITY = 0.25
RC8_ROW_BUDGET_ROWS = 56


def rc8_ctx8k_policy(*, row_budget_rows: int = RC8_ROW_BUDGET_ROWS,
                     prefill_priority: float = RC8_PREFILL_PRIORITY, pages_total: int = 800,
                     max_rows: int = 256, max_ctx: int = 16384) -> PrefillPolicy:
    return _row_policy(prefill_priority, 0, row_budget_rows, pages_total=pages_total,
                       max_rows=max_rows, max_ctx=max_ctx)


#: The candidates the rc8 doc projects, as (label, prefill_priority, row_budget_rows, tile).
#: ``row_budget_rows`` 0 = no budget (rc6 control).  The tile only matters for steps above 64
#: rows; at or below 64 every shape launches the 64 tile under either dispatch.
PROJECTION_CANDIDATES: Tuple[Tuple[str, float, int, int], ...] = (
    ("control (rc6, pp 0, no budget)", 0.0, 0, 64),
    ("run 6a as run (pp 0, 64-row budget)", 0.0, 64, 64),
    ("rc6 pp 0.25, no budget", 0.25, 0, 64),
    ("pp 0.25 + 48 rows", 0.25, 48, 64),
    ("pp 0.25 + 56 rows (rc8)", 0.25, 56, 64),
    ("pp 0.25 + 64 rows (rc7 projected)", 0.25, 64, 64),
    ("control at tile 128 (per-shape)", 0.0, 0, 128),
    ("pp 0.25 + 128 rows at tile 128", 0.25, 128, 128),
    ("pp 0.25, no budget, tile 128", 0.25, 0, 128),
)


def project(label_pp_rows_tile: Tuple[str, float, int, int], workload: str, *,
            padded: bool = False, closed_loop: bool = False,
            horizon_s: float = 900.0) -> SimResult:
    """PROJECT one candidate on one measured workload (a ``bidec_traces`` row name).

    ``closed_loop`` False (default) replays the measured send times: an A/B on IDENTICAL
    arrivals, which is what a policy decision should rest on.  True reruns glc-bench's worker
    loop on that run's request order (what a new card run would do) -- its p50 also moves with
    how many re-sends land inside the 90 s window, which is a knife edge (run 6a's own N=16
    re-sends went out at 80.2, 88.8 and 89.3 s), so it is printed beside, never decided on."""
    _label, pp, rows, tile = label_pp_rows_tile
    n = MEASURED_CTX8K[workload][0]
    tr = measured_trace(workload)
    pol = _row_policy(pp, 0, rows)
    model = rc8_model(padded=padded, tile=tile)
    if closed_loop:
        return simulate(n_users=n, policy=pol, model=model, trace=[(x[1], x[2]) for x in tr],
                        send_window_s=CTX8K_SEND_WINDOW_S, warm_s=CTX8K_WARM_S,
                        horizon_s=horizon_s)
    return simulate(n_users=n, policy=pol, model=model, arrivals=tr, horizon_s=horizon_s,
                    warm_s=CTX8K_WARM_S)


def _cli() -> None:                                   # pragma: no cover - operator entry point
    print("tile_share calibrates to %.4f on %s (stored %.4f); padded scale %.4f (stored %.4f)"
          % (calibrate_tile_share(), TILE_SHARE_CALIBRATION_ROW, TILE_SHARE,
             calibrate_padded_scale(), PADDED_SCALE))
    legacy = reproduce_measured_rc7()
    new = reproduce_measured()
    pad = reproduce_measured(model=rc8_model(padded=True))
    print("%-17s | %-22s | %-22s | %-30s | %-22s" % (
        "row", "rc7 model+pool t/u", "rc8 ttft p50/user", "rc8 ttft mean", "rc8-padded p50/user"))
    for k in MEASURED_CTX8K:
        a, b, c = legacy[k], new[k], pad[k]
        print("%-17s | %+5.0f%% / %+5.0f%%       | %+5.0f%% / %+5.0f%%       | "
              "%6.1f vs %6.1f (%+4.0f%%)    | %+5.0f%% / %+5.0f%%" % (
                  k, a["ttft_err"] * 100, a["per_user_err"] * 100,
                  b["ttft_err"] * 100, b["per_user_err"] * 100,
                  b["ttft_mean_sim"], b["ttft_mean_measured"], b["ttft_mean_err"] * 100,
                  c["ttft_err"] * 100, c["per_user_err"] * 100))
    for wl in ("run5/N16", "run5/N32", "run6a/N16/rb1", "run6a/N32/rb1"):
        print("== PROJECTED on the %s workload: replayed arrivals (closed-loop p50 beside)" % wl)
        for padded in (False, True):
            base = basec = None
            for cand in PROJECTION_CANDIDATES:
                r = project(cand, wl, padded=padded)
                rc = project(cand, wl, padded=padded, closed_loop=True)
                base, basec = base or r, basec or rc
                print("   %-6s %-38s ttft p50 %6.1f s (%.3fx) mean %6.1f (%.3fx)  per-user %5.2f "
                      "(%.3fx) | closed-loop p50 %6.1f (%.3fx)" % (
                          "padded" if padded else "live", cand[0], r.ttft_p50,
                          r.ttft_p50 / base.ttft_p50, r.ttft_mean, r.ttft_mean / base.ttft_mean,
                          r.per_user_decode_tok_s,
                          r.per_user_decode_tok_s / base.per_user_decode_tok_s,
                          rc.ttft_p50, rc.ttft_p50 / basec.ttft_p50))


if __name__ == "__main__":                            # pragma: no cover
    _cli()
