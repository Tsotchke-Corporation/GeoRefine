"""RefDecoder: a CPU reference BatchDecoder for gating the bidec *scheduler* without a GPU.

Why this exists
---------------
The batch-invariance of the arithmetic is already gated on the GPU at the kernel level
(``scripts/batchserve/bi_gate_kernel.py``, receipt ``receipts_kernel_gate_v1.json``: 0 rows
differ over prefixes M=1..200, permutation, composition and repeat).  What was never gated is
the layer above it -- the scheduler: slot allocation and reuse, KV page reservation and
recycling, per-slot recurrent-state reset, prompt chunking at arbitrary boundaries, admission
order, requests joining and leaving mid-flight, and padding rows.  Those are exactly the places
a multi-user server leaks one user's conversation into another's logits, and none of them need
a tensor core to test.

RefDecoder exposes the same surface ``Batcher`` uses from ``BatchDecoder``
(``alloc_slot`` / ``free_slot`` / ``ensure_pages`` / ``reset_slot_state`` / ``run`` /
``logits`` / ``am`` / ``free_slots`` / ``pages_total`` / ``max_pages`` / ``max_rows`` /
``max_slots`` / ``R`` / ``spec``), and stands in for the real engine in
``scripts/batchserve/bi_gate_batchexact.py`` and ``tests/test_bidec_scheduler.py``.

It is not a model.  It is a deterministic function whose value for a row is defined to depend
on **exactly** what a correct engine's row depends on, and on nothing else:

* the tokens of that row's own sequence at positions 0..p, **read back out of the KV pages
  through the slot's page table** (not out of a per-slot Python list) -- so a page that was
  never reserved, recycled while still live, or resolved through a stale page-table entry
  changes the answer;
* a per-slot recurrent state advanced one row at a time in position order -- so a slot reused
  by a second user without ``reset_slot_state`` changes the answer.

Padding rows are computed on the dummy slot and must not perturb any real row.

``leak=`` turns RefDecoder into a **positive control**: a decoder that really does violate
batch invariance, so the gate can be shown to fail when it should.

    batch      the row's value also depends on the padded batch width Mb
    neighbour  the row's value also depends on the token of the preceding row in the batch
    noreset    ``reset_slot_state`` is a no-op (state carries across users on a reused slot)
    nopages    ``ensure_pages`` is a no-op (reads fall through to whatever the page holds)
    noring     the recurrent state is folded monotonically instead of being indexed by position,
               so re-feeding a position (a rejected speculative draft, a re-run prompt chunk)
               contaminates every later row -- the real bug this ring replaced
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .bidec import BUCKETS, PAGE, bucket_for

_MUL = 6364136223846793005
_M64 = (1 << 64) - 1
LEAK_MODES = ("batch", "neighbour", "noreset", "nopages", "noring")


def _mix(x: int) -> int:
    """splitmix64 finaliser."""
    x &= _M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M64
    return (x ^ (x >> 31)) & _M64


class RefDecoder:
    """A CPU stand-in for ``BatchDecoder`` with the same slot / page / state bookkeeping."""

    def __init__(self, *, max_slots: int = 8, max_rows: int = 64, pages_total: int = 64,
                 max_ctx: int = 4096, R: int = 1, V: int = 128, leak: Optional[str] = None,
                 buckets: Sequence[int] = BUCKETS, spec: bool = False) -> None:
        if leak is not None and leak not in LEAK_MODES:
            raise ValueError(f"leak must be one of {LEAK_MODES}")
        self.leak = leak
        self.max_slots = int(max_slots)
        self.R = int(R)
        # `spec` only declares that a Batcher may be built with spec_k > 0 against this
        # reference; the MTP draft pass itself is not modelled here (see run_mtp).
        self.spec = bool(spec)
        self.buckets = tuple(b for b in buckets if b <= max_rows) or (max_rows,)
        self.max_rows = self.buckets[-1]
        self.max_pages = (int(max_ctx) + PAGE - 1) // PAGE
        self.pages_total = int(pages_total)
        self.V = int(V)
        self.dummy = self.max_slots            # padding rows' slot; page 0 is its page
        S1 = self.max_slots + 1

        # page 0 belongs to the dummy slot.  Deliberately NOT cleared on free, so that a page
        # handed to a new sequence without being written still carries the old user's token --
        # a correct engine never reads it, a broken one does and the gate sees it.
        self.pagemem = np.full((self.pages_total + 1, PAGE), -1, dtype=np.int64)
        self.pt = np.zeros((S1, self.max_pages), dtype=np.int64)
        self.free_pages: List[int] = list(range(self.pages_total, 0, -1))
        self.free_slots: List[int] = list(range(self.max_slots - 1, -1, -1))
        # The recurrent state is indexed BY POSITION, in a ring of depth R, exactly as the
        # engine keeps only the last R states of a sequence.  A row at position p is computed
        # from the state after p-1, so re-feeding a position -- which is what a rejected
        # speculative draft and a re-run prompt chunk both do -- is idempotent instead of
        # contaminating every later row.  `gpos` records which position each ring slot holds,
        # so a stale entry cannot be mistaken for the one asked for.
        self.gring = np.zeros((S1, self.R), dtype=np.uint64)
        self.gpos = np.full((S1, self.R), -1, dtype=np.int64)
        self.gmono = np.zeros(S1, dtype=np.uint64)       # only the `noring` control uses this
        # What a sequence on this slot starts from.  Zero after a reset -- which is the whole
        # point of the reset: the engine zeroes the recurrent buffer so a reused slot does not
        # start from the previous user's state.  The `noreset` control leaves it behind.
        self.gcarry = np.zeros(S1, dtype=np.uint64)

        self.logits = torch.zeros(self.max_rows, self.V, dtype=torch.float32)
        self.am = torch.zeros(self.max_rows, dtype=torch.int32)
        self.state_bytes: Dict[str, int] = {"pages": int(self.pagemem.nbytes)}
        self.steps = 0
        self.rows_run = 0
        self.max_rows_seen = 0

    # ------------------------------------------------------------------ slots and pages
    def alloc_slot(self) -> int:
        return self.free_slots.pop()

    def free_slot(self, slot: int) -> None:
        for i, p in enumerate(self.pt[slot]):
            if p:
                self.free_pages.append(int(p))
                self.pt[slot][i] = 0
        self.free_slots.append(int(slot))

    def ensure_pages(self, slot: int, upto_pos: int) -> None:
        if self.leak == "nopages":
            return
        need = upto_pos // PAGE + 1
        if need > self.max_pages:
            raise ValueError(f"position {upto_pos} exceeds max_ctx {self.max_pages * PAGE}")
        row = self.pt[slot]
        for i in range(need):
            if row[i] == 0:
                if not self.free_pages:
                    raise RuntimeError("KV page pool exhausted")
                row[i] = self.free_pages.pop()

    def reset_slot_state(self, slot: int) -> None:
        if self.leak == "noreset":
            return
        self.gring[slot] = 0
        self.gpos[slot] = -1
        self.gmono[slot] = 0
        self.gcarry[slot] = 0

    # ------------------------------------------------------------------ recurrent state
    def state_after(self, slot: int, pos: int) -> int:
        """The state after position `pos` of this slot, or 0 before the sequence starts.

        Raises if `pos` is no longer in the ring: a sequence may only be advanced within the
        last R positions, which is the same constraint the engine's state ring imposes
        (speculation at depth k needs R >= k + 1).
        """
        if pos < 0:
            return int(self.gcarry[slot])
        i = pos % self.R
        if int(self.gpos[slot][i]) != pos:
            raise RuntimeError(f"state for slot {slot} position {pos} is not in the ring of "
                               f"depth {self.R} (holds position {int(self.gpos[slot][i])}); "
                               f"a sequence cannot be advanced from a position already evicted")
        return int(self.gring[slot][i])

    def _advance(self, slot: int, pos: int, tok: int) -> int:
        if self.leak == "noring":
            self.gmono[slot] = np.uint64((int(self.gmono[slot]) * _MUL + tok + 1) & _M64)
            return int(self.gmono[slot])
        nxt = (self.state_after(slot, pos - 1) * _MUL + tok + 1) & _M64
        i = pos % self.R
        self.gring[slot][i] = np.uint64(nxt)
        self.gpos[slot][i] = pos
        self.gcarry[slot] = np.uint64(nxt) if pos == 0 and self.leak == "noreset" \
            else self.gcarry[slot]
        if self.leak == "noreset":
            self.gcarry[slot] = np.uint64(nxt)           # the slot stays dirty for the next user
        return nxt

    # ------------------------------------------------------------------ page-backed "KV"
    def _write(self, slot: int, pos: int, tok: int) -> None:
        page = int(self.pt[slot][pos // PAGE])
        self.pagemem[page][pos % PAGE] = int(tok)

    def _history_digest(self, slot: int, pos: int) -> int:
        """FNV-1a over the slot's tokens at positions 0..pos, read back through the page table."""
        h = 0xCBF29CE484222325
        for p in range(pos + 1):
            page = int(self.pt[slot][p // PAGE])
            t = int(self.pagemem[page][p % PAGE])
            h = ((h ^ (t & 0xFFFFFFFF)) * 0x100000001B3) & _M64
        return h

    def _row_logits(self, row: int, key: int) -> None:
        """Fill one logits row deterministically from `key` alone."""
        x = _mix(key)
        buf = np.empty(self.V, dtype=np.float32)
        for i in range(self.V):
            x = (x * _MUL + 1442695040888963407) & _M64
            buf[i] = np.float32((_mix(x) >> 40) / float(1 << 24))
        self.logits[row] = torch.from_numpy(buf)
        self.am[row] = int(buf.argmax())

    # ------------------------------------------------------------------ one step
    def run(self, seqs: List[Tuple[int, int, List[int]]]) -> Tuple[int, List[Tuple[int, int]]]:
        """Same contract as ``BatchDecoder.run``: seqs = [(slot, pos0, token_ids)]."""
        r = 0
        ranges: List[Tuple[int, int]] = []
        plan: List[Tuple[int, int, int, int]] = []       # (row, slot, pos, tok)
        for slot, pos0, toks in seqs:
            n = len(toks)
            if n == 0:
                raise ValueError("empty sequence row group")
            self.ensure_pages(slot, pos0 + n - 1)
            ranges.append((r, n))
            for t, tok in enumerate(toks):
                plan.append((r, int(slot), pos0 + t, int(tok)))
                r += 1
        M = r
        if M > self.max_rows:
            raise ValueError(f"{M} rows exceed max_rows {self.max_rows}")
        Mb = bucket_for(M, self.buckets)
        for rr in range(M, Mb):                          # padding rows: dummy slot, position 0
            plan.append((rr, self.dummy, 0, 0))

        for i, (row, slot, pos, tok) in enumerate(plan):
            self._write(slot, pos, tok)
            st = self._advance(slot, pos, tok)
            key = _mix(self._history_digest(slot, pos) ^ _mix(st ^ (pos << 1)))
            if self.leak == "batch":
                key ^= _mix(Mb)
            elif self.leak == "neighbour" and i:
                key ^= _mix(plan[i - 1][3] + 1)
            self._row_logits(row, key)

        self.steps += 1
        self.rows_run += M
        self.max_rows_seen = max(self.max_rows_seen, Mb)
        return Mb, ranges

    def run_mtp(self, rows):
        """Not modelled.  The reference covers the trunk pass and the state ring, which is what
        makes a rejected draft safe; the draft pass itself is gated by the speculation lane's
        own fixtures.  Refusing is deliberate -- a stub returning zeros would make a speculation
        gate pass without testing anything."""
        raise NotImplementedError(
            "RefDecoder models the trunk pass and the position-indexed state ring, not the MTP "
            "draft pass; gate speculation with the speculation lane's own reference")

    # ------------------------------------------------------------------ invariants
    def check_accounting(self) -> None:
        """Raise unless every page is either free exactly once or mapped by exactly one slot."""
        mapped: Dict[int, int] = {}
        for s in range(self.max_slots + 1):
            for p in self.pt[s]:
                p = int(p)
                if not p:
                    continue
                if p in mapped:
                    raise AssertionError(f"page {p} mapped by slots {mapped[p]} and {s}")
                mapped[p] = s
        free = list(self.free_pages)
        if len(set(free)) != len(free):
            raise AssertionError("duplicate page in the free pool")
        both = set(free) & set(mapped)
        if both:
            raise AssertionError(f"pages both free and mapped: {sorted(both)}")
        if len(free) + len(mapped) != self.pages_total:
            raise AssertionError(f"page leak: {len(free)} free + {len(mapped)} mapped "
                                 f"!= {self.pages_total}")
        slots = list(self.free_slots)
        if len(set(slots)) != len(slots):
            raise AssertionError("duplicate slot in the free list")
