"""Batched exact speculation for the batch-invariant decode engine (E-SPEC in a continuous batch).

This module owns the *host-side* speculation logic only: per-slot drafter state, the per-step
allocation of verify rows across the batch, accept/rewind per slot, and the measured-acceptance
policy.  It is torch-free and numpy-free so it runs (and is gated) on CPU against a reference
engine, and so the GPU scheduler can import it without pulling in the drafters' cost of a device
sync.  The engine side is an adapter with four methods (``EngineAdapter`` below).

Exactness contract (inherits G3a-BI from ``bidec``)
--------------------------------------------------
1. Every *emitted* token and every *emitted* logits row is a **trunk** row of the one batched
   forward.  Drafts never reach the output: they only decide which tokens occupy the verify rows.
2. BI-GEMM is M-invariant and attention is split at a fixed page boundary, so trunk row r is
   bitwise the M=1 step for its own sequence.  Rows therefore do not interact, and a slot may
   carry a different number of verify rows (``k_i + 1``) than its neighbours; padding rows go to
   the dummy slot.
3. Acceptance is the longest prefix of the draft that equals the trunk's own argmax rows
   (greedy).  Emission is ``argmax[r0 .. r0 + a]``, i.e. ``a`` accepted drafts plus the bonus
   token, which is exactly what the non-speculative loop would have emitted next.
4. All per-slot state (KV pages, GDN conv ring, GDN recurrent state, MTP hidden) is indexed by
   absolute position, so "rewind" is "set the position back to pos0 + a + 1".  There is no
   roll-back of a shared buffer, and hence no vLLM #39273 class of bug.
5. Because of (1)-(4), the accepted token stream of a request is identical to its single-user
   non-speculative stream for *any* k schedule, *any* batch composition and *any* arrival order.
   ``scripts/batchserve/bi_gate_spec_batch.py`` gates exactly that statement.

Why k must be capped when the batch is large
--------------------------------------------
The step cost is a function of the *total* row count, shared by every request in the batch, while
the benefit of a draft row accrues to one slot.  Speculation is therefore a *budget allocation*
problem, not a per-request choice.  ``CostModel`` holds the engine's measured t(M) curve and
``SpecPolicy`` spends rows where the marginal expected accepted tokens per marginal millisecond
is highest, stopping when that ratio falls below the batch's own tokens-per-ms.  With the measured
BI-GEMM curve (flat M=1..64, ~2x at M=128 because the kernel re-decodes weights per 64-row tile)
the crossover is explicit: see ``CostModel.crossover_batch``.
"""
from __future__ import annotations

import bisect
import heapq
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------- n-gram drafting

# Suffix-match orders, longest first.  A longer match is a stronger signal, so the index is probed
# from the longest order down and the first hit wins.
ORDERS: Tuple[int, ...] = (32, 24, 16, 12, 8, 6, 5, 4, 3, 2, 1)
_MASK = (1 << 61) - 1
_B = 1000003


def _hash(toks: Sequence[int], i: int, n: int) -> int:
    """Polynomial hash of toks[i:i+n] (collisions are harmless: matches are verified by value)."""
    h = n
    for t in toks[i:i + n]:
        h = (h * _B + (t + 1)) & _MASK
    return h


class NgramIndex:
    """Per-request suffix index: order -> most recent end position with a continuation.

    Hash buckets are keyed by a 61-bit polynomial hash, but every candidate match is then verified
    token by token, so a collision can never produce a false match (it can only cost a probe).  The
    continuation is an LZ77-style copy and may run into the request's own output, so periodic text
    keeps drafting.  Memory is O(len(history)) per order.
    """

    __slots__ = ("h", "tab", "max_extend")

    def __init__(self, history: Iterable[int] = (), max_extend: int = 64):
        self.h: List[int] = list(history)
        self.tab: Dict[Tuple[int, int], int] = {}
        self.max_extend = int(max_extend)
        if self.h:
            self._index_from(0)

    def _index_from(self, start: int) -> None:
        h, tab = self.h, self.tab
        L = len(h)
        for n in ORDERS:
            # an n-gram ending at position e (inclusive) has a continuation iff e + 1 < L
            lo = max(n - 1, start - 1)
            for e in range(lo, L - 1):
                tab[(n, _hash(h, e - n + 1, n))] = e

    def append(self, toks: Sequence[int]) -> None:
        if not toks:
            return
        start = len(self.h)
        self.h.extend(int(t) for t in toks)
        self._index_from(start)

    def match_len(self) -> int:
        """Length of the longest suffix order that has a recorded continuation (0 = no draft)."""
        h, L = self.h, len(self.h)
        for n in ORDERS:
            if n > L:
                continue
            e = self.tab.get((n, _hash(h, L - n, n)))
            if e is None or e + 1 >= L:
                continue
            if h[e - n + 1:e + 1] == h[L - n:]:
                return n
        return 0

    def draft(self, k: int) -> List[int]:
        """Up to k continuation tokens copied from the most recent occurrence of the longest
        matching suffix.  Returns [] when nothing matches."""
        if k <= 0:
            return []
        h, L = self.h, len(self.h)
        for n in ORDERS:
            if n > L:
                continue
            e = self.tab.get((n, _hash(h, L - n, n)))
            if e is None or e + 1 >= L:
                continue
            if h[e - n + 1:e + 1] != h[L - n:]:
                continue                                  # hash collision: try a shorter order
            out: List[int] = []
            p = e + 1
            while len(out) < k and p < L + len(out):
                out.append(h[p] if p < L else out[p - L])
                p += 1
            return out
        return []


# ---------------------------------------------------------------------------- acceptance stats


class AcceptStats:
    """Beta-smoothed per-depth acceptance, bucketed by a context key, updated with forgetting.

    ``key`` buckets the estimate by what the drafter knew at draft time (for n-gram: the suffix
    match length, clamped; for MTP: a single bucket).  ``p(key, d)`` is the probability that the
    draft at depth d is accepted *given depths < d were*, so the expected accepted count for a
    draft of length k is sum_{d<k} prod_{j<=d} p(key, j).
    """

    __slots__ = ("a", "b", "decay", "prior")

    def __init__(self, prior: float = 0.5, strength: float = 2.0, decay: float = 0.995):
        self.a: Dict[Tuple[Any, int], float] = {}
        self.b: Dict[Tuple[Any, int], float] = {}
        self.decay = float(decay)
        self.prior = (float(prior) * strength, (1.0 - float(prior)) * strength)

    def p(self, key: Any, d: int) -> float:
        a = self.a.get((key, d), self.prior[0])
        b = self.b.get((key, d), self.prior[1])
        return a / (a + b)

    def expected(self, key: Any, k: int) -> float:
        """E[accepted drafts | draft length k] under the chain model."""
        e, q = 0.0, 1.0
        for d in range(k):
            q *= self.p(key, d)
            e += q
        return e

    def update(self, key: Any, k: int, accepted: int) -> None:
        """One verify outcome: depths 0..accepted-1 accepted, depth `accepted` rejected (if drafted)."""
        for d in range(min(accepted + 1, k)):
            ka, kb = (key, d), (key, d)
            a = self.a.get(ka, self.prior[0]) * self.decay + (1.0 - self.decay) * self.prior[0]
            b = self.b.get(kb, self.prior[1]) * self.decay + (1.0 - self.decay) * self.prior[1]
            if d < accepted:
                a += 1.0
            else:
                b += 1.0
            self.a[ka], self.b[kb] = a, b

    def rate(self, key: Any) -> float:
        return self.p(key, 0)


# ---------------------------------------------------------------------------- cost model

# Measured BI-GEMM / engine step shape (SERVING_COST_20260925.md §3: bf16 gate|up 238 us at M=1,
# 261 us at M=64; the kernel re-decodes weights once per 64-row tile, so M=128 is about 2x).
# Normalised to "cost units per step" and scaled by the operator's measured single-row step time.
DEFAULT_SHAPE: Tuple[Tuple[int, float], ...] = (
    (1, 1.000), (2, 1.002), (4, 1.005), (8, 1.010), (16, 1.019), (24, 1.028),
    (32, 1.038), (48, 1.067), (64, 1.097), (96, 1.60), (128, 2.05), (192, 2.75),
    (256, 3.40), (384, 4.90), (512, 6.40),
)


@dataclass
class CostModel:
    """t(M): milliseconds for one batched trunk step of M rows.

    ``step_ms`` is the measured M=1 step (weight-read bound).  ``shape`` is the measured relative
    curve; pass the engine's own ``mcost`` table when one exists.  ``mtp_ms`` is one batched MTP
    draft pass (also M-invariant, so it is charged once per depth, not per slot).
    """

    step_ms: float = 20.0
    mtp_ms: float = 1.44
    flat_rows: int = 64         # rows over which the step is MEASURED flat; the binding cap
    shape: Sequence[Tuple[int, float]] = DEFAULT_SHAPE
    bucket_rows: Sequence[int] = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
    mtp_shape: Optional[Sequence[Tuple[int, float]]] = None

    def __post_init__(self) -> None:
        self._m = [int(m) for m, _ in self.shape]
        self._f = [float(f) for _, f in self.shape]
        if self.mtp_shape is not None:
            previous = 0
            for bucket, value in self.mtp_shape:
                if (type(bucket) is not int or bucket <= previous
                        or type(value) not in (int, float)
                        or not math.isfinite(value) or value <= 0):
                    raise ValueError("MTP cost table requires increasing rows and positive finite timings")
                previous = bucket
            ceiling = self.bucket(self.flat_rows)
            required = {row for row in self.bucket_rows if row <= ceiling}
            if not required.issubset({row for row, _ in self.mtp_shape}):
                raise ValueError("MTP cost table must cover every usable graph bucket")

    def bucket(self, rows: int) -> int:
        i = bisect.bisect_left(self.bucket_rows, max(1, int(rows)))
        if i >= len(self.bucket_rows):
            return int(self.bucket_rows[-1])
        return int(self.bucket_rows[i])

    def verify_ms(self, rows: int, *, bucketed: bool = True) -> float:
        """Trunk-step cost for `rows` rows.  CUDA graphs are captured per row bucket, so the engine
        actually pays the bucket's cost; set bucketed=False for the raw curve."""
        m = self.bucket(rows) if bucketed else max(1, int(rows))
        if m <= self._m[0]:
            return self.step_ms * self._f[0]
        if m >= self._m[-1]:
            return self.step_ms * self._f[-1]
        i = bisect.bisect_right(self._m, m) - 1
        m0, m1, f0, f1 = self._m[i], self._m[i + 1], self._f[i], self._f[i + 1]
        return self.step_ms * (f0 + (f1 - f0) * (m - m0) / (m1 - m0))

    def step_cost_ms(self, rows: int, depths: int) -> float:
        """One speculative cycle: `depths` batched MTP draft passes plus the trunk verify, plus the
        single batched MTP confirm pass."""
        return self.verify_ms(rows) + (max(0, depths) + 1) * self.mtp_pass_ms(rows)

    def mtp_pass_ms(self, rows: int) -> float:
        """Conservative calibrated head cost for the candidate verify-row bound.

        Every draft pass and confirm has at most the candidate's verify rows.
        Retain the maximum across all usable smaller buckets, including noise or
        nonmonotone tables. With no explicit table the historic scalar is unchanged.
        """
        if self.mtp_shape is None:
            return self.mtp_ms
        ceiling = self.bucket(rows)
        if ceiling > self.mtp_shape[-1][0]:
            raise ValueError("MTP cost table does not cover the candidate row bound")
        values = [float(value) for bucket, value in self.mtp_shape if bucket <= ceiling]
        if not values:
            raise ValueError("MTP cost table does not cover the candidate row bound")
        return max(values)

    def marginal_ms(self, rows: int) -> float:
        """Cost of adding one more verify row at a total of `rows` rows (bucket-aware: zero inside
        a bucket, a jump at its edge).  This is what makes k cheap at small batches and expensive
        at large ones."""
        return max(0.0, self.verify_ms(rows + 1) - self.verify_ms(rows))

    def crossover_rows(self) -> int:
        """The knee is a ROW BUDGET, not a batch size (Synergy, arXiv 2310.18813; vLLM's 2.8x at
        QPS 1 -> 1.8x slowdown at high QPS is the same effect).  BI-GEMM is measured flat only to
        ``flat_rows`` rows, so ``sum_slots (1 + k_i) <= flat_rows`` is the binding constraint: 64
        verify rows at one user, 8 at eight users, and k = 0 at 64 users until a 128-row tile is
        gated.  ``SpecPolicy`` clamps to this."""
        return int(self.flat_rows)

    def crossover_batch(self, accept_rate: float = 0.7, k: int = 4) -> int:
        """The batch size (decoding sequences) at which adding k drafts per sequence stops paying.

        For n sequences, non-spec costs t(n) per step and delivers n tokens; spec costs
        step_cost_ms(n*(k+1), k-1) and delivers n*(1 + E[accepted]).  Returns the smallest n at
        which tokens/ms with speculation is no better than without (or the largest bucket if it
        always pays)."""
        st = AcceptStats(prior=accept_rate, strength=1e6)
        e = st.expected("x", k)
        for n in range(1, int(self.bucket_rows[-1]) // (k + 1) + 1):
            if n * (k + 1) > self.crossover_rows():
                return n                                  # the row budget binds first (see above)
            base = n / self.verify_ms(n)
            spec = n * (1.0 + e) / self.step_cost_ms(n * (k + 1), k - 1)
            if spec <= base:
                return n
        return int(self.bucket_rows[-1]) // (k + 1) + 1


# ---------------------------------------------------------------------------- per-slot state

SOURCES = ("none", "ngram", "mtp", "hybrid")


@dataclass
class SlotSpec:
    """Per-request drafter state.  One of these per active slot; it outlives a step."""

    rid: Any
    ngram: NgramIndex = field(default_factory=NgramIndex)
    mtp_chain: List[int] = field(default_factory=list)   # confirmed MTP drafts, chain order
    source: str = "none"
    drafts: List[int] = field(default_factory=list)      # this step's proposal
    k: int = 0                                           # len(drafts) at verify time
    k_plan: int = 0                                      # draft length the policy asked for
    chain_n: int = 0                                     # how many of `drafts` came from the MTP head
    key: Any = None                                      # stats bucket used for this proposal
    proposed: int = 0
    accepted: int = 0
    cycles: int = 0
    emitted: int = 0
    by_source: Dict[str, List[int]] = field(default_factory=lambda: {s: [0, 0, 0] for s in SOURCES})

    def note_prompt(self, prompt: Sequence[int]) -> None:
        self.ngram.append(prompt)

    def mean_accepted(self) -> float:
        return self.accepted / self.cycles if self.cycles else 0.0


def ngram_key(match: int) -> Tuple[str, int]:
    """Bucket the n-gram estimate by suffix match length (0, 1-2, 3-5, 6-11, 12+)."""
    for hi, lab in ((0, 0), (2, 1), (5, 2), (11, 3)):
        if match <= hi:
            return ("ng", lab)
    return ("ng", 4)


# ---------------------------------------------------------------------------- policy


@dataclass
class SpecPolicy:
    """Acceptance-adaptive, budget-aware draft planner.

    ``plan`` is called once per step with the decoding slots and the row budget left after prompt
    chunks.  It (a) picks a source per slot from measured acceptance, (b) allocates draft rows by
    marginal gain against the *shared* marginal row cost, and (c) returns the per-slot draft
    length.  Nothing here can affect output bits; a wrong decision only costs throughput.
    """

    cost: CostModel = field(default_factory=CostModel)
    max_k: int = 7
    min_gain: float = 1.0        # require the marginal row to beat the batch's own tokens/ms by this
    allow_ngram: bool = False    # MTP is the default drafter; n-gram sources are opt-in (see below)
    stats_ng: AcceptStats = field(default_factory=lambda: AcceptStats(prior=0.4))
    stats_mtp: AcceptStats = field(default_factory=lambda: AcceptStats(prior=0.7))

    def _cand(self, sp: SlotSpec, max_k: int) -> List[Tuple[str, Any, int]]:
        """(source, stats key, max usable k) candidates for this slot.

        ``allow_ngram`` defaults to False: MTP self-drafting is the default because it is the arm
        whose batching behaviour is exonerated in the literature, while n-gram / prompt-lookup
        drafts are where other engines corrupted output under batching (vLLM #39273, #40875 -- in
        both cases a *verifier* bug, not a drafter bug, but the risk concentrates there).  Our
        verifier makes draft quality irrelevant to the bits, and the gate proves it for both arms,
        so n-gram is available -- opt in per request or per deployment, with
        ``bi_gate_spec_batch.py --ngram`` as its own exactness control.
        """
        out: List[Tuple[str, Any, int]] = [("none", None, 0)]
        m = sp.ngram.match_len() if self.allow_ngram else 0
        if m:
            out.append(("ngram", ngram_key(m), max_k))
        if sp.mtp_chain:
            # A seed makes drafting available; it is not the maximum chain length.
            # The controller extends short chains with batched MTP calls after planning.
            out.append(("mtp", ("mtp", 0), max_k))
            if m:
                out.append(("hybrid", ("hy", min(ngram_key(m)[1], 4)), max_k))
        return out

    def _stats(self, source: str) -> AcceptStats:
        return self.stats_ng if source == "ngram" else self.stats_mtp

    def expected(self, source: str, key: Any, k: int, chain: int = 0) -> float:
        """E[accepted drafts | source, k].

        ``hybrid`` is an MTP chain of up to ``chain`` tokens followed by an n-gram extension, so its
        chain probability is the MTP profile for the first ``chain`` depths and the n-gram profile
        after that.  A hopeless MTP head therefore kills hybrid too, which is what we want: depth 0
        of a hybrid proposal is always an MTP token."""
        if k <= 0:
            return 0.0
        if source == "ngram":
            return self.stats_ng.expected(key, k)
        if source == "mtp":
            return self.stats_mtp.expected(key, k)
        e, q, c = 0.0, 1.0, max(0, min(int(chain), k))
        for d in range(c):
            q *= self.stats_mtp.p(("mtp", 0), d)
            e += q
        for d in range(c, k):
            q *= self.stats_ng.p(key, d)
            e += q
        return e

    def plan(self, slots: Sequence[SlotSpec], row_budget: int) -> Dict[Any, Tuple[str, int]]:
        """Choose (source, k) per slot under a total verify-row budget.

        Every decoding slot needs 1 row unconditionally (the non-speculative row), so the draft
        budget is ``row_budget - len(slots)``.  Rows are then handed out one at a time to the slot
        with the largest marginal expected gain, as long as that gain beats the marginal row cost
        measured at the current total.  ``min_gain`` makes the comparison conservative: a marginal
        row must earn at least ``min_gain`` x (batch tokens per ms) to be taken.

        ``row_budget`` should be the scheduler's remaining rows for this step (ask ``SchedulerLike``
        for ``remaining_rows``; do not recount here).  It is additionally clamped to
        ``cost.crossover_rows()``, the measured flat region of the step, which alone yields 64
        verify rows at one user, 8 at eight users and k = 0 at 64 users.
        """
        n = len(slots)
        plan: Dict[Any, Tuple[str, int]] = {s.rid: ("none", 0) for s in slots}
        if n == 0:
            return plan
        spare = min(int(row_budget), self.cost.crossover_rows()) - n
        if spare <= 0:
            return plan
        best: Dict[Any, Tuple[str, Any, int, int]] = {}
        for s in slots:
            cands = self._cand(s, min(self.max_k, spare))
            ch = len(s.mtp_chain)
            # pick the source with the best expected accepted tokens at its own best k
            pick = None
            for src, key, mk in cands:
                if src == "none" or mk <= 0:
                    continue
                e = self.expected(src, key, mk, ch)
                if pick is None or e > pick[0] + 1e-12:
                    pick = (e, src, key, mk)
            if pick is not None and pick[0] > 0.0:
                best[s.rid] = (pick[1], pick[2], pick[3], ch)
        if not best:
            return plan
        # Explore complete allocation prefixes across graph-bucket jumps. A locally
        # expensive first row can open a profitable batch, so do not stop at that row.
        k = {rid: 0 for rid in best}
        order = {slot.rid: i for i, slot in enumerate(slots)}
        heap = []
        for rid, (src, key, mk, ch) in best.items():
            gain = self.expected(src, key, 1, ch)
            if gain > 0:
                heapq.heappush(heap, (-gain, order[rid], rid))
        rows, expected_tokens, max_depth = n, float(n), 0
        has_mtp = any(slot.mtp_chain for slot in slots)
        baseline_ms = self.cost.verify_ms(n)
        # The controller confirms even a no-draft cycle to seed its next MTP
        # chain. Compare like-for-like costs in the opt-in calibrated mode;
        # the legacy scalar policy remains byte-for-byte unchanged when off.
        if self.cost.mtp_shape is not None and has_mtp:
            baseline_ms += self.cost.mtp_pass_ms(n)
        best_rate = self.min_gain * n / baseline_ms
        while spare > 0 and heap:
            negative_gain, _, rid = heapq.heappop(heap)
            src, key, mk, ch = best[rid]
            k[rid] += 1
            rows += 1
            spare -= 1
            expected_tokens -= negative_gain
            if src != "ngram":
                max_depth = max(max_depth, max(0, k[rid] - ch))
            # One shared head pass per extension depth and one batched confirm,
            # never one head pass for every request's extra verification row.
            head_passes = max_depth + int(has_mtp)
            cost_ms = self.cost.verify_ms(rows) + head_passes * self.cost.mtp_pass_ms(rows)
            rate = expected_tokens / cost_ms
            if rate > best_rate + 1e-12:
                best_rate = rate
                plan = {slot.rid: ((best[slot.rid][0], k[slot.rid])
                                  if slot.rid in k and k[slot.rid] else ("none", 0))
                        for slot in slots}
            if k[rid] < mk:
                gain = (self.expected(src, key, k[rid] + 1, ch)
                        - self.expected(src, key, k[rid], ch))
                if gain > 0:
                    heapq.heappush(heap, (-gain, order[rid], rid))
        return plan

    def feedback(self, sp: SlotSpec, accepted: int, chain: int = 0) -> None:
        """Credit the outcome to the profile(s) that produced the proposal."""
        if sp.k <= 0 or sp.key is None:
            return
        if sp.source in ("ngram", "mtp"):
            self._stats(sp.source).update(sp.key, sp.k, accepted)
            return
        c = max(0, min(int(chain), sp.k))                  # hybrid: MTP head then n-gram tail
        if c:
            self.stats_mtp.update(("mtp", 0), c, min(accepted, c))
        if sp.k > c and accepted >= c:
            self.stats_ng.update(sp.key, sp.k, accepted)


# ---------------------------------------------------------------------------- engine adapter


class EngineAdapter(ABC):
    """What ``BatchSpecController`` needs from an engine.  ``bidec.BatchDecoder``/``Batcher``
    satisfies it (see ``docs/serving/BIDEC_SPEC_BATCH_20261002.md`` §Wiring); the CPU reference in
    ``tests/batchserve_spec/ref_engine.py`` satisfies it without a GPU."""

    @abstractmethod
    def mtp_extend(self, reqs: Sequence[Tuple[Any, int, int]]) -> Dict[Any, int]:
        """One batched MTP draft pass.  reqs: [(rid, abs_pos, token)] -> {rid: next draft token}."""

    @abstractmethod
    def trunk_verify(self, reqs: Sequence[Tuple[Any, int, Sequence[int]]]) -> Dict[Any, List[int]]:
        """ONE batched trunk forward.  reqs: [(rid, pos0, tokens)] -> {rid: argmax per row}.

        Row i of a request's block is the model's own argmax given that request's prefix plus
        tokens[:i], and must be bitwise what the M=1 step would give."""

    @abstractmethod
    def mtp_confirm(self, reqs: Sequence[Tuple[Any, int, int, bool]]) -> Dict[Any, int]:
        """One batched MTP confirm pass over the accepted prefix.
        reqs: [(rid, abs_pos, token, is_last)] -> {rid: first draft token} for the is_last rows."""

    def logits_digest(self, rid: Any, row: int) -> Optional[str]:
        """Optional: digest of an emitted logits row, for the gate."""
        return None


class MtpLike(ABC):
    """The MTP head, as the controller needs it.  ``bidec.BatchDecoder.run_mtp`` is wrapped by
    ``DecoderMtp``; the CPU gate injects a mock, because speculation's *exactness* does not depend
    on the drafter at all (only its throughput does)."""

    @abstractmethod
    def rows(self, rows: Sequence[Tuple[int, int, int, int, int]]) -> List[int]:
        """rows = [(slot, mtp_pos, token, hidx, flag)] -> one draft token per row, in order."""


class DecoderMtp(MtpLike):
    """``BatchDecoder.run_mtp`` + ``bd.mam``."""

    def __init__(self, bd: Any):
        self.bd = bd

    def rows(self, rows):
        self.bd.run_mtp(list(rows))
        return [int(t) for t in self.bd.mam[:len(rows)].tolist()]


class DecoderAdapter(EngineAdapter):
    """``EngineAdapter`` over anything satisfying ``bidec_iface.DecoderLike``.

    This is the single place that knows the engine's row layout, so the same controller drives the
    CUDA ``BatchDecoder`` and the CPU ``bidec_ref.RefDecoder``.  ``slot_of`` maps a request id to
    its device slot (the scheduler owns slots); ``mtp`` supplies drafts (``DecoderMtp`` on the real
    engine, a mock on CPU).  ``extra`` rows are prompt chunks the scheduler wants in the same
    forward: they are appended after the verify blocks and their argmax is returned separately, so
    prefill and verify share one step exactly as in ``Batcher._step_spec``.
    """

    def __init__(self, decoder: Any, slot_of: Any, mtp: Optional[MtpLike] = None, *, digest: bool = False,
                 mtp_confirm_kv_only: bool = False):
        if type(mtp_confirm_kv_only) is not bool:
            raise ValueError("mtp_confirm_kv_only must be a bool")
        if mtp_confirm_kv_only:
            buckets = getattr(decoder, "buckets", ())
            if (mtp is None or not callable(getattr(decoder, "run_mtp_kv", None))
                    or not buckets or any(type(x) is not int or x <= 0 for x in buckets)
                    or list(buckets) != sorted(set(buckets))):
                raise ValueError("KV-only confirmation requires an MTP head, KV updater and ordered buckets")
        self.d = decoder
        self.slot_of = slot_of
        self.mtp = mtp
        self.digest = bool(digest)
        self.mtp_confirm_kv_only = mtp_confirm_kv_only
        if mtp_confirm_kv_only:
            self.mtp_confirm_counters = {"optimized_calls": 0, "legacy_calls": 0,
                                         "prefix_kv_rows": 0, "full_head_rows_avoided": 0}
        self.last_rows: Dict[Any, Tuple[int, int]] = {}
        self.last_pos: Dict[Any, int] = {}
        self.extra: List[Tuple[int, int, List[int]]] = []
        self.extra_am: List[List[int]] = []
        self.steps = 0
        self.rows_total = 0

    def mtp_extend(self, reqs):
        if self.mtp is None:
            return {}
        rows = [(self.slot_of(rid), pos, int(tok), self.slot_of(rid), 1) for rid, pos, tok in reqs]
        got = self.mtp.rows(rows)
        return {rid: got[i] for i, (rid, _, _) in enumerate(reqs)}

    def trunk_verify(self, reqs):
        # Hidden rows are replaced by every forward, so old request mappings expire.
        self.last_rows.clear()
        self.last_pos.clear()
        seqs = [(self.slot_of(rid), pos0, [int(t) for t in toks]) for rid, pos0, toks in reqs]
        seqs += list(self.extra)
        Mb, ranges = self.d.run(seqs)
        am = [int(x) for x in self.d.am[:Mb].tolist()]
        self.steps += 1
        self.rows_total += sum(n for _, n in ranges)
        out: Dict[Any, List[int]] = {}
        for i, (rid, pos0, toks) in enumerate(reqs):
            r0, n = ranges[i]
            self.last_rows[rid] = (r0, n)
            self.last_pos[rid] = pos0
            out[rid] = am[r0:r0 + n]
        self.extra_am = [am[r0:r0 + n] for r0, n in ranges[len(reqs):]]
        return out

    def mtp_confirm(self, reqs):
        if self.mtp is None:
            return {}
        rows = []
        for rid, pos, tok, last in reqs:
            if rid not in self.last_rows:
                raise ValueError("MTP confirm has no row in the last trunk verify")
            r0, n = self.last_rows[rid]
            offset = pos - self.last_pos[rid]
            if not 0 <= offset < n:
                raise ValueError("MTP confirm position is outside the last trunk verify")
            s = self.slot_of(rid)
            # Confirm consumes trunk xf, not the per-slot hidden state used to extend drafts.
            hidx = self.d.hsrc_trunk0 + r0 + offset
            rows.append((s, pos, int(tok), hidx, 1 if last else 0))
        if self.mtp_confirm_kv_only:
            # Intermediate accepted tokens need their MTP K/V, but their
            # attention/MLP/logits and hidden output are never consumed. Complete
            # those K/V writes before evaluating the final token in each slot.
            last = [(request, row) for request, row in zip(reqs, rows) if request[3]]
            prefix = [row[:4] for request, row in zip(reqs, rows) if not request[3]]
            by_request = {}
            for rid, pos, _, is_last in reqs:
                if type(is_last) is not bool:
                    raise ValueError("MTP confirmation last marker must be a bool")
                by_request.setdefault(rid, []).append((pos, is_last))
            for positions in by_request.values():
                if (sum(int(flag) for _, flag in positions) != 1 or not positions[-1][1]
                        or any(b[0] != a[0] + 1 for a, b in zip(positions, positions[1:]))):
                    raise ValueError("MTP confirmation must contain an ordered prefix and one final row")
            def bucket(n):
                i = bisect.bisect_left(self.d.buckets, max(1, n))
                if i >= len(self.d.buckets):
                    raise ValueError("MTP confirmation exceeds available row buckets")
                return self.d.buckets[i]
            # Avoid an extra pass if no GPU rows disappear after bucket padding.
            # Small batches have too little measured head-cost separation; keep
            # their original path. This is a workload restriction, not a gain claim.
            if prefix and len(last) >= 64 and bucket(len(last)) < bucket(len(rows)):
                self.d.run_mtp_kv(prefix)
                got = self.mtp.rows([row for _, row in last])
                if len(got) != len(last):
                    raise RuntimeError("MTP final-row result count mismatch")
                counts = self.mtp_confirm_counters
                counts["optimized_calls"] += 1
                counts["prefix_kv_rows"] += len(prefix)
                counts["full_head_rows_avoided"] += len(prefix)
                return {request[0]: got[i] for i, (request, _) in enumerate(last)}
            self.mtp_confirm_counters["legacy_calls"] += 1
        got = self.mtp.rows(rows)
        return {rid: got[i] for i, (rid, _, _, last) in enumerate(reqs) if last}

    def logits_digest(self, rid: Any, row: int) -> Optional[str]:
        if not self.digest or rid not in self.last_rows:
            return None
        r0, _ = self.last_rows[rid]
        return self.logits_digest_row(r0 + row)

    def logits_digest_row(self, row: int) -> Optional[str]:
        if not self.digest:
            return None
        import hashlib

        lg = self.d.logits[row]
        try:
            import torch

            b = lg.view(torch.int16).cpu().numpy().tobytes()
        except Exception:
            b = bytes(memoryview(lg.cpu().numpy()).cast("B"))
        return hashlib.sha256(b).hexdigest()[:16]


# ---------------------------------------------------------------------------- controller


@dataclass
class StepReport:
    rows: int
    depths: int
    emitted: Dict[Any, List[int]]
    accepted: Dict[Any, int]
    proposed: Dict[Any, int]
    sources: Dict[Any, str]
    est_ms: float


class BatchSpecController:
    """Drives one speculative cycle over a batch.

    A cycle is: plan -> (k-1 batched MTP draft passes, one per depth) -> ONE batched trunk forward
    with a variable, padded number of verify rows per slot -> per-slot accept/rewind -> one batched
    MTP confirm pass over every accepted row.  The controller holds no device state.
    """

    def __init__(self, engine: EngineAdapter, policy: Optional[SpecPolicy] = None):
        self.engine = engine
        self.policy = policy or SpecPolicy()
        self.slots: Dict[Any, SlotSpec] = {}
        self.cycles = 0
        self.rows_total = 0
        self.tokens_total = 0
        self.proposed_total = 0
        self.accepted_total = 0

    # -- slot lifecycle -----------------------------------------------------
    def admit(self, rid: Any, prompt: Sequence[int], *, corpus: Optional[Sequence[int]] = None) -> SlotSpec:
        sp = SlotSpec(rid=rid)
        if corpus:
            sp.ngram.append(corpus)
        sp.note_prompt(prompt)
        self.slots[rid] = sp
        return sp

    def release(self, rid: Any) -> None:
        self.slots.pop(rid, None)

    def seed_mtp(self, rid: Any, token: int) -> None:
        """The engine's prefill/confirm pass produced the slot's first MTP draft."""
        sp = self.slots.get(rid)
        if sp is not None:
            sp.mtp_chain = [int(token)]

    def observe_emitted(self, rid: Any, toks: Sequence[int]) -> None:
        sp = self.slots.get(rid)
        if sp is not None and toks:
            sp.ngram.append(toks)
            sp.emitted += len(toks)

    # -- one cycle ----------------------------------------------------------
    def step(self, active: Sequence[Tuple[Any, int, int]], row_budget: int, *,
             draft_limits: Optional[Dict[Any, int]] = None) -> StepReport:
        """active: [(rid, pos_of_last_emitted, last_emitted_token)] for decoding requests.

        Returns the tokens emitted per request this cycle, in order.  The caller (the scheduler)
        owns stop/length handling: it truncates the returned list and must then call
        ``release`` or ``observe_emitted`` accordingly.
        """
        # Optional per-request scratch bounds apply before even the first draft pass.
        # A zero bound still runs the ordinary trunk row and MTP confirmation.
        slots = [self.slots[rid] for rid, _, _ in active if rid in self.slots]
        plan = self.policy.plan(slots, row_budget)
        for sp in slots:
            src, k = plan[sp.rid]
            if draft_limits is not None:
                k = min(k, max(0, int(draft_limits[sp.rid])))
            sp.source, sp.k, sp.k_plan, sp.drafts, sp.key, sp.chain_n = src, 0, max(0, k), [], None, 0
            if src == "none" or k <= 0:
                sp.source, sp.k_plan = "none", 0
                continue
            if src == "ngram":
                sp.drafts = sp.ngram.draft(k)
                sp.key = ngram_key(sp.ngram.match_len())
            else:                                           # mtp / hybrid start from the MTP chain
                sp.drafts = list(sp.mtp_chain[:k])
                sp.key = (("mtp", 0) if src == "mtp"
                          else ("hy", min(ngram_key(sp.ngram.match_len())[1], 4)))
            if not sp.drafts:
                sp.source, sp.k_plan = "none", 0

        # -- batched MTP draft passes: each depth extends every slot whose chain is still short.
        # The MTP head is M-invariant too, so one pass serves the whole batch at that depth.
        pos = {rid: p for rid, p, _ in active}
        depths = 0
        for _ in range(int(self.policy.max_k)):
            reqs = [(sp.rid, pos[sp.rid] + len(sp.drafts) - 1, sp.drafts[-1]) for sp in slots
                    if sp.source in ("mtp", "hybrid") and 0 < len(sp.drafts) < sp.k_plan]
            if not reqs:
                break
            depths += 1
            got = self.engine.mtp_extend(reqs)
            for rid, _, _ in reqs:
                t = got.get(rid)
                if t is None:
                    continue
                sp = self.slots[rid]
                sp.drafts.append(int(t))
                sp.mtp_chain = list(sp.drafts)
        for sp in slots:
            sp.chain_n = len(sp.drafts) if sp.source in ("mtp", "hybrid") else 0
            if sp.source == "hybrid" and len(sp.drafts) < sp.k_plan:
                # n-gram extension of the MTP chain's tail (only when the tail matches history)
                probe = NgramIndex(sp.ngram.h + sp.drafts)
                sp.drafts.extend(probe.draft(sp.k_plan - len(sp.drafts)))
            sp.k = len(sp.drafts)
            if sp.k == 0:
                sp.source = "none"

        # -- ONE batched trunk forward; variable k per slot, padded by the engine
        reqs = [(rid, p, [t] + self.slots[rid].drafts) for rid, p, t in active if rid in self.slots]
        rows = sum(len(x[2]) for x in reqs)
        am = self.engine.trunk_verify(reqs)

        emitted: Dict[Any, List[int]] = {}
        acc: Dict[Any, int] = {}
        prop: Dict[Any, int] = {}
        srcs: Dict[Any, str] = {}
        confirm: List[Tuple[Any, int, int, bool]] = []
        for rid, p, toks in reqs:
            sp = self.slots[rid]
            rowam = am[rid]
            if len(rowam) != len(toks):
                raise RuntimeError(f"engine returned {len(rowam)} argmax rows for {len(toks)} verify rows")
            a = 0
            while a < sp.k and sp.drafts[a] == rowam[a]:
                a += 1
            out = [int(rowam[i]) for i in range(a + 1)]
            emitted[rid] = out
            acc[rid], prop[rid], srcs[rid] = a, sp.k, sp.source
            sp.proposed += sp.k
            sp.accepted += a
            sp.cycles += 1
            bs = sp.by_source[sp.source]
            bs[0] += 1
            bs[1] += sp.k
            bs[2] += a
            self.policy.feedback(sp, a, sp.chain_n)
            sp.mtp_chain = []                               # rewound: the chain past a is stale
            sp.drafts = []
            for i, t in enumerate(out):
                confirm.append((rid, p + i, t, i == len(out) - 1))

        if confirm:
            first = self.engine.mtp_confirm(confirm)
            for rid, t in first.items():
                sp = self.slots.get(rid)
                if sp is not None and t is not None:
                    sp.mtp_chain = [int(t)]

        self.cycles += 1
        self.rows_total += rows
        self.tokens_total += sum(len(v) for v in emitted.values())
        self.proposed_total += sum(prop.values())
        self.accepted_total += sum(acc.values())
        return StepReport(rows=rows, depths=depths, emitted=emitted, accepted=acc, proposed=prop,
                          sources=srcs, est_ms=self.policy.cost.step_cost_ms(rows, depths))

    # -- reporting ----------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        per = {}
        for rid, sp in self.slots.items():
            per[str(rid)] = {"proposed": sp.proposed, "accepted": sp.accepted, "cycles": sp.cycles,
                             "emitted": sp.emitted, "mean_accepted": round(sp.mean_accepted(), 3),
                             "by_source": {k: v for k, v in sp.by_source.items() if v[0]}}
        return {"cycles": self.cycles, "rows_total": self.rows_total, "tokens_total": self.tokens_total,
                "proposed_total": self.proposed_total, "accepted_total": self.accepted_total,
                "rows_per_token": round(self.rows_total / self.tokens_total, 3) if self.tokens_total else None,
                "slots": per}


class SpecDriver:
    """Slot / page / position bookkeeping around ``BatchSpecController``, over a ``DecoderLike``.

    This is the wiring: a scheduler's speculative step becomes

        for s in newly_admitted:  drv.admit(s.rid, s.prompt)        # prefill + first token
        out = drv.step(remaining_rows, truncate=my_stop_rule)       # one speculative cycle
        for rid, toks in out.items(): ...                           # stream them

    and the scheduler keeps owning admission, stop rules, cancellation and accounting.  Only ONE
    implementation of the cycle exists (``BatchSpecController.step``), and it is the one the gate
    runs, so "gated" and "served" are the same code path.  ``prefill_chunk`` is applied here so
    that chunking is exercised; the decoder's contract makes it output-neutral.
    """

    def __init__(self, decoder: Any, mtp: Optional["MtpLike"] = None,
                 policy: Optional[SpecPolicy] = None, *, prefill_chunk: int = 256,
                 digest: bool = False, adapter_cls: Any = None):
        self.d = decoder
        self.mtp = mtp
        self.slots: Dict[Any, int] = {}
        self.pos: Dict[Any, int] = {}
        self.last: Dict[Any, int] = {}
        self.prefill_chunk = int(prefill_chunk)
        self.adapter = (adapter_cls or DecoderAdapter)(decoder, lambda rid: self.slots[rid], mtp,
                                                        digest=digest)
        self.ctl = BatchSpecController(self.adapter, policy or SpecPolicy())

    def admit(self, rid: Any, prompt: Sequence[int], *, corpus: Optional[Sequence[int]] = None,
              adopted: int = 0, reuse_slot: Optional[int] = None, seed_from_suffix: bool = False
              ) -> Tuple[int, Optional[str]]:
        """Allocate a slot, prefill through the decode step in chunks, return the first token.

        ``adopted`` is how many leading prompt positions a prefix cache already materialised in
        ``reuse_slot``: only positions ``adopted ..`` are fed.  **The drafter state is still seeded
        from the FULL prompt**, which is the hazard vLLM #58894 is: DFlash2 acceptance on the
        Qwen3.5 family fell from 36.9% to exactly 0.0% once prefix-cache hits began, because the
        drafter's state was aligned to the fed suffix instead of the adopted prefix.  Exactness is
        not at risk either way (drafts never reach the output) — acceptance is, silently, which is
        worse.  ``seed_from_suffix`` reproduces the bug on purpose, for the gate's control.
        """
        slot = self.d.alloc_slot() if reuse_slot is None else int(reuse_slot)
        self.slots[rid] = slot
        if reuse_slot is None:
            self.d.reset_slot_state(slot)
        p = list(prompt)
        adopted = max(0, min(int(adopted), len(p)))
        self.ctl.admit(rid, p[adopted:] if seed_from_suffix else p, corpus=corpus)
        # the last prompt position must always be fed: its row is the one that produces the first
        # token, so a prefix cache may adopt at most len(prompt) - 1 positions
        fed = min(adopted, len(p) - 1)
        rg = None
        while fed < len(p):
            c = min(self.prefill_chunk, len(p) - fed)
            _, rg = self.d.run([(slot, fed, p[fed:fed + c])])
            if self.mtp is not None and hasattr(self.d, "run_mtp_kv"):
                r0, n = rg[0]
                kv = [(slot, pos, p[pos + 1], self.d.hsrc_trunk0 + r0 + pos - fed)
                      for pos in range(fed, fed + n) if pos + 1 < len(p)]
                if kv:
                    self.d.run_mtp_kv(kv)
            fed += c
        r0, n = rg[0]
        dig = self.adapter.logits_digest_row(r0 + n - 1)
        t = int(self.d.am[r0 + n - 1])
        # the prompt occupies positions 0 .. len(p)-1, so the first generated token sits at
        # position len(p); ``pos`` counts committed tokens, hence pos - 1 is the last one's position
        self.pos[rid] = len(p) + 1
        self.last[rid] = t
        self.ctl.observe_emitted(rid, [t])
        if self.mtp is not None:
            hidx = self.d.hsrc_trunk0 + r0 + n - 1
            self.ctl.seed_mtp(rid, self.mtp.rows([(slot, len(p) - 1, t, hidx, 1)])[0])
        return t, dig

    def release(self, rid: Any) -> None:
        slot = self.slots.pop(rid, None)
        if slot is not None:
            self.d.free_slot(slot)
        self.pos.pop(rid, None)
        self.last.pop(rid, None)
        self.ctl.release(rid)

    def step(self, remaining_rows: int, truncate: Any = None
             ) -> Tuple[Dict[Any, List[Tuple[int, Optional[str]]]], StepReport]:
        """One speculative cycle over every admitted request.

        ``truncate(rid, toks) -> n`` lets the caller stop a request mid-block (stop token, length
        cap); only the first n tokens are committed, and the rest are discarded -- which is exact,
        because a discarded token was never streamed and the slot is about to be freed.
        """
        live = [rid for rid in self.slots if rid in self.pos]
        if not live:
            return {}, StepReport(0, 0, {}, {}, {}, {}, 0.0)
        active = [(rid, self.pos[rid] - 1, self.last[rid]) for rid in live]
        rep = self.ctl.step(active, remaining_rows)
        out: Dict[Any, List[Tuple[int, Optional[str]]]] = {}
        for rid in live:
            toks = rep.emitted[rid]
            n = len(toks) if truncate is None else max(0, min(len(toks), int(truncate(rid, toks))))
            keep = toks[:n]
            out[rid] = [(t, self.adapter.logits_digest(rid, i)) for i, t in enumerate(keep)]
            if keep:
                self.pos[rid] += len(keep)
                self.last[rid] = keep[-1]
                self.ctl.observe_emitted(rid, keep)
        return out, rep

    def stats(self) -> Dict[str, Any]:
        return {**self.ctl.stats(), "forwards": self.adapter.steps, "engine_rows": self.adapter.rows_total}


__all__ = ["ORDERS", "NgramIndex", "AcceptStats", "CostModel", "DEFAULT_SHAPE", "SOURCES", "SlotSpec",
           "ngram_key", "SpecPolicy", "EngineAdapter", "MtpLike", "DecoderMtp", "DecoderAdapter",
           "StepReport", "BatchSpecController", "SpecDriver"]
