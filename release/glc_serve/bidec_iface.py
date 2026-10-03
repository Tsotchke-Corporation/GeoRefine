"""Stable interfaces between the bidec scheduler and the modules layered on it.

Bind to these Protocols, not to ``bidec.Batcher`` / ``bidec.BatchDecoder`` internals.  Both the
real engine (``bidec.BatchDecoder``, CUDA) and the CPU reference (``bidec_ref.RefDecoder``)
satisfy ``DecoderLike``, so anything written against it is testable without a GPU.

Owner of this file and of the scheduler it describes: the multi-user serving branch
(the multi-user serving work, 2026-10-02).  Additions are append-only; existing names and
signatures do not change without a note on the ICC bus.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

PAGE = 256                     # KV page = attention split; a position's page is pos // PAGE


@runtime_checkable
class DecoderLike(Protocol):
    """One batched forward step over slots, with explicit slot / page / state ownership."""

    max_slots: int             # device slots (one per concurrent sequence)
    max_rows: int              # largest padded batch width in rows
    max_pages: int             # pages addressable by one slot (context cap)
    pages_total: int           # pages in the shared pool
    R: int                     # per-slot recurrent-state ring depth (spec needs R >= k + 1)
    spec: bool                 # an MTP head is loaded
    free_slots: List[int]
    logits: Any                # [max_rows, V] -- row-indexed, valid after run()
    am: Any                    # [max_rows] int32 argmax, valid after run()

    def alloc_slot(self) -> int: ...
    def free_slot(self, slot: int) -> None: ...
    def ensure_pages(self, slot: int, upto_pos: int) -> None: ...
    def reset_slot_state(self, slot: int) -> None: ...

    def run(self, seqs: List[Tuple[int, int, List[int]]]) -> Tuple[int, List[Tuple[int, int]]]:
        """seqs = [(slot, pos0, token_ids)]; one row per token, groups in the given order.

        Returns (Mb, ranges): Mb is the padded row count, ranges[i] = (row0, n) for seqs[i].
        A row's value must depend only on its own sequence's tokens at positions 0..pos and on
        that slot's state -- never on Mb, on neighbouring rows, or on padding.
        """


@runtime_checkable
class SchedulerLike(Protocol):
    """What ``bidec.Batcher`` offers to a server, a policy layer, or a gate."""

    max_active: int
    max_rows_step: int
    prefill_chunk: int
    spec_k: int
    waiting: List[Any]
    active: List[Any]
    reserved: int
    lock: Any

    def add(self, s: Any) -> None: ...
    def step(self) -> int: ...
    def idle(self) -> bool: ...
    def cancel(self, rid: Any) -> str: ...
    def stats(self) -> Dict[str, Any]: ...
    def admissible(self, prompt_len: int, max_new: int) -> bool: ...

    # The speculation crossover is a ROW budget, not a batch size: the BI-GEMM knee is measured
    # at 64 rows, so a spec adapter must keep sum(1 + k_i) over decoding sequences within
    # max_rows_step and degrade k toward 0 under load.  spec_k_budget() returns the largest
    # uniform k that fits, or 0.
    def remaining_rows(self) -> int: ...
    def spec_k_budget(self, k_wanted: Optional[int] = None) -> int: ...


@runtime_checkable
class AdmissionPolicy(Protocol):
    """Optional policy hook (bidec_policy.py).  Called at a step boundary, inside no lock.

    ``choose`` returns the indices of ``waiting`` to admit now, in admission order, given the
    capacity the scheduler has this step.  Returning [] defers.  A policy must not reorder or
    mutate ``active``, must not touch slots or pages, and must not depend on batch composition:
    admission order may change a request's latency, never its tokens.
    """

    def choose(self, waiting: Sequence[Any], active: Sequence[Any], *, free_slots: int,
               free_pages: int, max_active: int) -> List[int]: ...

    def on_finish(self, seq: Any, why: str) -> None: ...


@runtime_checkable
class PrefixCacheLike(Protocol):
    """Optional shared-prefix KV reuse (bidec_prefix.py).

    ``lookup`` reports how many leading prompt tokens of ``tokens`` are already materialised and
    may be adopted into ``slot`` without recomputation; ``adopt`` installs them (page table only)
    and returns the number of positions now resident.  Correctness bar: adopting a prefix must
    not change any emitted token -- the gate is bi_gate_batchexact.py with the cache enabled,
    plus a cache-miss / cache-hit A/B on identical prompts.
    """

    def lookup(self, tokens: Sequence[int]) -> int: ...
    def adopt(self, slot: int, tokens: Sequence[int], n: int) -> int: ...
    def publish(self, slot: int, tokens: Sequence[int], upto: int) -> None: ...
    def release(self, slot: int) -> None: ...
    def stats(self) -> Dict[str, Any]: ...


__all__ = ["PAGE", "DecoderLike", "SchedulerLike", "AdmissionPolicy", "PrefixCacheLike"]
