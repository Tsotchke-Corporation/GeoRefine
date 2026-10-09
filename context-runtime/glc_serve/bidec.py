"""BatchDecoder: a batch-invariant continuous-batching decode engine for the Qwen3.5/3.8 hybrid
(48 Gated-DeltaNet + 16 gated full-attention layers).

Exactness contract (G3a-BI).  A request's logits -- every row, every bit -- are the same
whether it runs alone or with any number of other requests, whenever they join or leave, and
however its prompt is chunked.  That holds because every row of every step is computed from
its own sequence only, by a fixed instruction sequence:

* projections: BI-GEMM (``bigemm``; fixed K order, fixed split count per (N, K), no atomics,
  mma rows independent), with the weights decoded in the kernel by the MIV decoders (TBE
  codec -> the parent's bf16 bits; Q8_0 / K-quants -> bf16_rn(gguf-py dequantize));
* small ops: the fastdec kernels (per-row) and ``bidec_kernels.cu`` (per-row / per-sequence
  copies of the fastdec kernels with batched indexing);
* attention: paged KV, page = 256 positions = the fixed attention split, so a row's split
  partials and their split-order combine are the single-stream schedule.  The attention shape
  (head_dim 256 or 128, output gate, q/k norm) is the model's (fastdec.attn_layout,
  ATTN_HEADDIM_20261004); the page stays 256 POSITIONS at every head dim;
* Gated-DeltaNet: per-slot conv ring + fp32 recurrent state; a sequence's rows run
  sequentially with the single-step op order.

Prefill runs through the SAME step as decode (a prompt chunk is a multi-row sequence), so the
prompt's KV / state carry decode bits and chunking cannot change a request's output.

Greedy argmax on device; temperature sampling is done on the host from the row's logits with
a per-request generator (row-local, so batch-invariant too).
"""
from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import bidec_mm as bm
from . import bigemm as bg
from . import fastdec as fdm

_EXT = None
_LOCK = threading.Lock()
_FUNCS = ["b_gdn_conv", "b_gdn_recur", "b_attn_prep", "b_attn_split", "b_attn_combine", "b_kv_import",
          "b_scatter_rows"]
_CPP = r"""
#include <torch/extension.h>
void b_gdn_conv(at::Tensor in, at::Tensor hist, at::Tensor cw, at::Tensor seq_slot, at::Tensor seq_row0,
                at::Tensor seq_len, at::Tensor n_seq, at::Tensor row_pos, at::Tensor out, int64_t maxseq);
void b_gdn_recur(at::Tensor conv, at::Tensor zba, int64_t z_off, int64_t b_off, int64_t a_off, at::Tensor A_log,
                 at::Tensor dt_bias, at::Tensor nw, at::Tensor state, int64_t R, at::Tensor seq_slot,
                 at::Tensor seq_row0, at::Tensor seq_len, at::Tensor n_seq, at::Tensor row_pos, at::Tensor out,
                 int64_t maxseq, double eps, int64_t nkh_rep);
void b_attn_prep(at::Tensor qkv, at::Tensor qnw, at::Tensor knw, at::Tensor cosT, at::Tensor sinT,
                 at::Tensor row_slot, at::Tensor row_pos, at::Tensor n_rows, at::Tensor ropeoff, at::Tensor pagetab,
                 int64_t maxpages, at::Tensor qout, at::Tensor kc, at::Tensor vc, int64_t maxrows, int64_t NQ,
                 int64_t NKV, double eps, int64_t gated, int64_t qk_norm);
void b_attn_split(at::Tensor q, at::Tensor kc, at::Tensor vc, at::Tensor row_slot, at::Tensor row_pos,
                  at::Tensor item_row, at::Tensor item_split, at::Tensor n_items, at::Tensor pagetab,
                  int64_t maxpages, at::Tensor pacc, at::Tensor pml, int64_t grid_items, int64_t NQ, int64_t NKV,
                  double scaling);
void b_attn_combine(at::Tensor pacc, at::Tensor pml, at::Tensor qkv, at::Tensor row_pos, at::Tensor row_item0,
                    at::Tensor n_rows, at::Tensor out, int64_t maxrows, int64_t NQ, int64_t gated);
void b_kv_import(at::Tensor k, at::Tensor v, at::Tensor pages, int64_t n, at::Tensor kc, at::Tensor vc);
void b_scatter_rows(at::Tensor src, at::Tensor flag, at::Tensor idx, at::Tensor n_rows, at::Tensor dst, int64_t maxrows);
"""
PAGE = 256
BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)

# How many hex characters of the row digest are carried.  64 bits: the gate compares
# per-token digests of rows it has independently produced, so this is a collision bound on
# an adversary-free comparison, not a hash-security parameter.
DIGEST_HEX = 16


def logits_row_digest(row: "torch.Tensor") -> str:
    """The exactness digest of ONE logits row: sha256 over the row's raw bytes.

    Every exactness claim about the served path -- in-process (bi_gate_engine.py,
    bi_gate_batchexact.py) and over HTTP (bi_gate_stream.py --exact-digest) -- is a
    comparison of this string, so there is exactly one definition of it and both sides of
    every comparison call THIS function.  ``view(torch.int16)`` is a byte-preserving
    reinterpretation that makes ``.numpy()`` work for bf16 as well as fp32; the digest is
    therefore over all ``row.numel()`` output elements exactly as the kernel produced them,
    before any sampling or argmax.
    """
    import hashlib

    return hashlib.sha256(row.view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:DIGEST_HEX]


def ext():
    global _EXT
    if _EXT is None:
        with _LOCK:
            if _EXT is None:
                # G1: a console script invoked by absolute path does not put the venv's bin/ on
                # PATH, and cpp_extension resolves ninja with shutil.which.  Put the declared,
                # already-installed tool where the build will actually look for it.
                from glc_loader._jitenv import ensure_build_tools_on_path

                ensure_build_tools_on_path()
                from torch.utils.cpp_extension import load_inline

                arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
                if not arch:
                    major, minor = torch.cuda.get_device_capability()
                    arch = f"{major}.{minor}"
                    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
                tag = arch.split(";")[0].replace("+PTX", "").replace(".", "")
                d = bg._build_dir().parent / "bidec_build"
                d.mkdir(parents=True, exist_ok=True)
                src = Path(__file__).with_name("bidec_kernels.cu").read_text()
                _EXT = load_inline(name=f"bidec_sm{tag}", cpp_sources=[_CPP], cuda_sources=[src], functions=_FUNCS,
                                   extra_cuda_cflags=["-O3", "-lineinfo"], build_directory=str(d), verbose=False)
    return _EXT


def format_state_bytes(d: Dict[str, int]) -> str:
    """``"kv 12.52 GiB, gdn_state 9.14 GiB, ..."`` -- byte counts only.

    rc2 finding F4: this formatting lived inline and divided every value of
    ``state_bytes`` by 2**30, while ``state_bytes`` also carried the GDN ring's dtype
    *name*.  ``TypeError: unsupported operand type(s) for /: 'str' and 'int'`` killed
    startup on the happy path, for everyone, every time -- in a log statement.  A
    non-numeric value now raises here, by name, in a unit test instead of on a GPU box.
    """
    bad = {k: type(v).__name__ for k, v in d.items() if not isinstance(v, (int, float))}
    if bad:
        raise TypeError(
            f"state_bytes must hold byte counts only; non-numeric entries: {bad}. "
            f"Carry names (e.g. the GDN ring dtype) beside it, not inside it.")
    return ", ".join(f"{k} {v / 2**30:.2f} GiB" for k, v in d.items())


class WorkListOverflow(RuntimeError):
    """One step's (row, KV page) work list does not fit the engine's ``g_cap``.

    Raised BEFORE anything is written to the metadata buffer, so the caller can feed less
    and retry.  rc2 had no such check on the write path and surfaced the same condition as
    a numpy broadcast error from inside ``run`` (finding F5).
    """


def bucket_for(m: int, buckets: Sequence[int] = BUCKETS) -> int:
    for b in buckets:
        if b >= m:
            return b
    raise ValueError(f"{m} rows exceed the largest bucket {buckets[-1]}")


def needs_gdn_state_ring(bd: Any) -> bool:
    """Whether ``bd``'s rejected-draft rollback needs a GDN recurrent-state ring of depth
    ``R >= spec_k + 1`` (``Batcher.__init__``).  R only bounds anything for a model that
    actually has a GDN recurrent-state ring (``BatchDecoder.state``, indexed by R) -- an
    all-attention dense model's (Qwen3-14B/32B, Llama, Mistral, ...) rejected-draft rollback
    is plain KV-page cropping, which needs no ring at all (BIDEC_DENSE_20261004).

    ``bd.fd.n_gdn`` is the authority on a real ``BatchDecoder``.  ``RefDecoder`` (the CPU
    scheduler reference) has no ``.fd`` and no model behind it either way, so it reports
    True unconditionally here -- unchanged scheduler-gate coverage, not a per-model fact.
    """
    return getattr(getattr(bd, "fd", None), "n_gdn", 1) > 0


class _Meta:
    """All per-step metadata in ONE int32 device tensor (one H2D copy per step)."""

    def __init__(self, dev, max_rows: int, max_slots: int, max_pages: int, g_cap: int):
        spec = [("vtok", max_rows), ("row_slot", max_rows), ("row_pos", max_rows), ("row_item0", max_rows),
                ("seq_slot", max_rows), ("seq_row0", max_rows), ("seq_len", max_rows),
                ("n_rows", 1), ("n_seq", 1), ("n_items", 1),
                ("item_row", g_cap), ("item_split", g_cap), ("ropeoff", max_slots + 1),
                ("hidx", max_rows), ("hflag", max_rows), ("mropeoff", max_slots + 1),
                # image rows of this step (docs/serving/BIDEC_MULTIMODAL_20261004.md); all
                # zero on a text-only step.  Before `pagetab`, which must stay last (upload()).
                ("mmflag", max_rows),
                ("pagetab", (max_slots + 1) * max_pages)]
        n = sum(s for _, s in spec)
        self.dev = torch.zeros(n, dtype=torch.int32, device=dev)
        # Pinned on a card (rc7, unchanged); a CPU device (glc_serve.kernel_emu) has no pinned
        # allocator and needs none.
        self.host = torch.zeros(n, dtype=torch.int32, pin_memory=torch.device(dev).type == "cuda")
        self.np = self.host.numpy()
        self.off: Dict[str, Tuple[int, int]] = {}
        o = 0
        for name, s in spec:
            self.off[name] = (o, s)
            setattr(self, name, self.dev[o:o + s])
            o += s

        self._pt_off, _pt_n = self.off["pagetab"]
        self._pt_cache = np.zeros(_pt_n, dtype=np.int32)
        self._uploads = 0
        self._pt_uploads = 0

    def h(self, name: str) -> np.ndarray:
        o, s = self.off[name]
        return self.np[o:o + s]

    def upload(self) -> None:
        """One H2D copy per step.  `pagetab` is the last field and 62.4% of the buffer, and it
        changes only when a slot gains or loses a page -- so when it is unchanged, copy the
        contiguous prefix and leave it alone.  The device copy stays correct because nothing
        else writes that range."""
        self._uploads += 1
        pt = self.np[self._pt_off:]
        # BatchDecoder.run() callers may reuse this pinned host buffer as soon as upload()
        # returns.  Complete the small metadata copy before returning so DMA cannot observe
        # a later host mutation; this is metadata-only synchronization, not weight traffic.
        if np.array_equal(pt, self._pt_cache):
            self.dev[:self._pt_off].copy_(self.host[:self._pt_off], non_blocking=False)
            return
        self._pt_cache[:] = pt
        self._pt_uploads += 1
        self.dev.copy_(self.host, non_blocking=False)


class BatchDecoder:
    #: FastDecoder attributes that are single-stream working buffers (fastdec.py:369-436) and
    #: that no BatchDecoder code path reads.  Released at construction.
    FD_DEAD_BUFFERS = ("h", "x", "xf", "proj", "dn", "zba", "conv", "go", "aqkv", "q", "ao",
                       "gu", "act", "logits", "pacc", "pml", "vtok", "pos", "ropeoff", "am",
                       "log", "mkc", "mvc", "mtok", "mkv", "mropeoff", "slot", "mam", "me",
                       "mcat", "mh", "mhid", "mx", "mg", "mlogits")

    def __init__(self, fd: "fdm.FastDecoder", *, max_slots: int = 64, max_rows: int = 256,
                 pages_total: int = 1024, max_ctx: int = 32768, R: int = 1, grid_items: int = 512,
                 buckets: Sequence[int] = BUCKETS, gdn_ring_dtype: str = "fp32",
                 item_cap: Optional[int] = None, gemm_tile: Optional[int] = None,
                 conv_ring: int = 16, log=print, mm_rope_rows: int = 0, selective_head: bool = False):
        from . import state_budget as _sb


        self.selective_head = bool(selective_head)
        self.head_rows_total = 0
        self.graphs_trunk = {}
        self.fd, self.dev, self.log = fd, fd.dev, log
        E, B = fdm.ext(), ext()
        bg.extension()
        self.max_slots, self.R = int(max_slots), int(R)
        # GDN conv-history ring depth (positions kept per slot).  rc7 hard-coded 16 (`& 15` in
        # b_gdn_conv_kernel); the kernel now takes the depth from hist.size(1).  The exact
        # minimum is a power of two >= max(4, spec_k + 3) -- state_budget.conv_ring_for has the
        # derivation -- so 16 costs 12/16 of the conv history for nothing when speculation is
        # off, and is silently WRONG for spec_k >= 14, which rc7 accepted.  0 = derive from
        # spec_k = R - 1; an explicit value too shallow for R is refused here.
        self.conv_ring = _sb.check_conv_ring(conv_ring, self.R - 1)
        # The per-slot Gated-DeltaNet recurrent ring is the fixed part of the per-user bill, so
        # its storage dtype is a real lever: measured on Qwen3.5-2B, an fp16 ring is
        # behaviour-lossless (greedy 1.000000 at 8k and over 64 round-trips, top-5 0.995,
        # rare-tail gap <= 0.17 pt, error growing only as sqrt(round-trips)) and halves the ring
        # traffic.  It is NOT enabled by flipping this flag: b_gdn_recur takes the ring as
        # `float* state_all` (bidec_kernels.cu:84, :99, :371 `P<float>(state)`), so handing it a
        # half tensor would reinterpret the bits and silently corrupt every sequence.  The flag
        # is plumbed and recorded in every receipt; fp16 is REFUSED until the kernel carries a
        # half-storage path (fp32 math in registers, fp16 in memory).
        self.gdn_ring_dtype = str(gdn_ring_dtype).lower()
        if self.gdn_ring_dtype not in ("fp32", "fp16"):
            raise ValueError(f"gdn_ring_dtype must be fp32 or fp16, got {gdn_ring_dtype!r}")
        if self.gdn_ring_dtype == "fp16":
            raise NotImplementedError(
                "an fp16 GDN recurrent ring needs a half-storage path in b_gdn_recur: the kernel "
                "types the ring as `float* state_all` (bidec_kernels.cu:84, :99) and is called "
                "with P<float>(state) (:371), so an fp16 buffer would be reinterpreted as fp32 "
                "and corrupt every sequence. The measurement that motivates it is in "
                "docs/serving/KV_STATE_DESCENT_20261003.md; until the kernel lands, keep fp32 "
                "and quote the saving as projected.")
        self.gdn_ring_bytes = 4 if self.gdn_ring_dtype == "fp32" else 2
        self.buckets = tuple(b for b in buckets if b <= max_rows)
        self.max_rows = self.buckets[-1]
        self.max_pages = (int(max_ctx) + PAGE - 1) // PAGE
        self.pages_total = int(pages_total)
        # The step's attention work list holds one (row, KV page) entry.  The formula below is
        # sized for the DECODE case: decode rows cost at most ``pages_total`` entries in total
        # (pages are reserved per slot at admission), so that cap is exactly right for decode
        # and far too small for prefill, where one prompt row at position p costs p//PAGE + 1.
        # Run 3 measured what that costs: at ctx8k a step could carry ~33 prompt tokens against
        # a 256-row budget, and TTFT p50 was 109 s (N=16) / 209 s (N=32) while decode held flat.
        # ``item_cap`` raises the buffer; it is a BUFFER SIZE, not a hardware limit, and it is
        # priced in glc_serve.bidec_policy (~24.8 KB/entry in pacc+pml).  None = rc4 behaviour.
        self.g_cap = self.pages_total + self.max_rows + 8
        if item_cap:
            self.g_cap = max(self.g_cap, int(item_cap))
        self.grid_items = int(grid_items)
        self.dummy = self.max_slots                   # padding rows' slot (page 0 is its page)
        self.H, self.eps, self.V = fd.H, fd.eps, fd.V
        fd.kc = fd.vc = fd.hist = fd.state = None          # the single-stream buffers are not used
        # ...and neither is the rest of FastDecoder's single-stream working set (fastdec.py
        # 369-436: the M=8 activation rows, an 8 x vocab logits buffer, split partials, and with
        # MTP a 512-position MTP KV and its rows).  rc7 released only the four above and kept
        # ~6.6 MB (8.8 MB with MTP) of device memory alive that nothing in this engine reads:
        # every bidec read of `fd` is a weight, a norm, the embedding, the rotary tables or the
        # MTP head's weights (fd.m).  Released, like kc/vc/hist/state.
        for _n in self.FD_DEAD_BUFFERS:
            if getattr(fd, _n, None) is not None:
                setattr(fd, _n, None)
        torch.cuda.empty_cache()
        self.cosT, self.sinT = self._rope_tables(int(max_ctx))
        self.NQ, self.NKV = fd.NQ, fd.NKV
        # Attention shape, from the model (fastdec.attn_layout): head dim 256 for the hybrid,
        # 128 for Qwen3 dense / Llama / Mistral.  rc7 sized every attention buffer below for
        # 256; bidec_capacity.geometry_of reads `head_dim` off this object.
        self.head_dim = D = int(fd.D)
        self.attn_gated, self.qk_norm, self.norm_mode = fd.attn_gated, fd.qk_norm, fd.norm_mode
        ws = self.ws = bg.Workspace(self.dev)
        # BI-GEMM row-tile CAP (the weight re-decode granularity).  64 = rc6 byte for byte;
        # 128 halves the weight traffic of every step that carries more than 64 rows.  The
        # launch height is still chosen per call from the row count (bigemm.tile_for), so a
        # 1-row decode step is unaffected by raising the cap.
        self.gemm_tile = bg.tile_cap() if gemm_tile is None else int(gemm_tile)
        if self.gemm_tile not in bg.TILES:
            raise ValueError(f"gemm_tile must be one of {bg.TILES}, got {gemm_tile}")
        wrap = lambda d: bg.wrap(d, ws, self.gemm_tile)  # noqa: E731
        self.L: List[Any] = []
        for o in fd.layers:
            b = fdm._Layer()
            b.src = o
            b.gu, b.down = wrap(o.gu), wrap(o.down)
            if o.kind == "gdn":
                b.qkvzba, b.out = wrap(o.qkvzba), wrap(o.out)
            else:
                b.qkv, b.o = wrap(o.qkv), wrap(o.o)
            self.L.append(b)
        self.lm = wrap(fd.lm)
        for d in self._descs():
            d.reserve(self.max_rows)
        dev, bf = self.dev, torch.bfloat16
        z = lambda *s, dt=bf: torch.zeros(*s, dtype=dt, device=dev)  # noqa: E731
        # BIDEC_DENSE_20261004: see fastdec.split_attn_gdn / fastdec.gdn_template_dims.  An
        # all-attention dense model (Qwen3-14B/32B, Llama, Mistral, ...) has fd.n_gdn == 0
        # and no "gdn"-kind layer at all, so gd0 is None and every GDN buffer below is
        # allocated zero-sized: zero GDN state bytes, and the step loop's
        # `if o.kind == "gdn":` branch (this class's `_step`) is never taken for any layer.
        att0, gd0 = fdm.split_attn_gdn(fd.layers)
        gdn_C, gdn_N, gdn_V = fdm.gdn_template_dims(gd0)
        M, H = self.max_rows, self.H
        S1 = self.max_slots + 1
        t0 = time.time()
        self.kc = z(fd.n_att, self.pages_total + 1, PAGE, self.NKV, D)
        self.vc = z(fd.n_att, self.pages_total + 1, PAGE, self.NKV, D)
        # GDN conv history and recurrent ring: one per REAL slot.  rc7 sized both [S1] = slots + 1
        # for symmetry with the page table, but the dummy slot (index max_slots) only ever
        # carries PADDING rows, and padding rows are never a sequence: b_gdn_conv / b_gdn_recur
        # iterate b < n_seq over seq_slot[], which run() fills from real slots only and
        # _set_padding_only / run_mtp leave at n_seq = 0.  So rc7's dummy row of `state`
        # (151 MB at R = 1 on the 27B) and of `hist` was allocated, zeroed and never read or
        # written.  run() now refuses a sequence on a slot outside [0, max_slots), which is the
        # guard that makes the smaller allocation safe rather than merely sufficient.
        self.hist = z(fd.n_gdn, self.max_slots, self.conv_ring, gdn_C)
        self.state = z(fd.n_gdn, self.max_slots, self.R, 48, 128, 128,
                       dt=torch.float32 if self.gdn_ring_dtype == "fp32" else torch.float16)
        self.h, self.x, self.proj, self.dn = (z(M, H) for _ in range(4))
        # MTP hidden sources in ONE buffer: rows [0, S1) = per-slot MTP hidden store, rows [S1, S1 + M) =
        # the trunk's final-norm output of the last step (xf) -- one index_select serves draft + confirm rows
        self.hsrc = z(S1 + M, H)
        self.hsrc_trunk0 = S1
        self.mhid, self.xf = self.hsrc[:S1], self.hsrc[S1:]
        self.zba = z(M, gdn_N)
        self.conv = z(M, gdn_C)
        self.go = z(M, gdn_V)
        self.aqkv = z(M, att0.qkv.N)
        self.q = z(M, self.NQ, D)
        self.ao = z(M, self.NQ * D)
        F = fd.layers[0].F
        self.gu = z(M, 2 * F)
        self.act = z(M, F)
        self.logits = z(M, self.V)
        self.am = z(M, dt=torch.int32)
        self.pacc = z(self.g_cap, self.NQ, D, dt=torch.float32)
        self.pml = z(self.g_cap, self.NQ, 2, dt=torch.float32)
        self.spec = bool(fd.mtp_ready)
        if self.spec:
            fm = fd.m
            m = self.m = fdm._Layer()
            m.pre_e, m.pre_h, m.in_ln, m.post_ln, m.norm = fm.pre_e, fm.pre_h, fm.in_ln, fm.post_ln, fm.norm
            m.qnw, m.knw, m.scaling = fm.qnw, fm.knw, fm.scaling
            m.fc, m.qkv, m.o, m.gu, m.down = (wrap(d) for d in (fm.fc, fm.qkv, fm.o, fm.gu, fm.down))
            for d in (m.fc, m.qkv, m.o, m.gu, m.down):
                d.reserve(M)
            self.mkc = z(self.pages_total + 1, PAGE, self.NKV, D)
            self.mvc = z(self.pages_total + 1, PAGE, self.NKV, D)
            self.me, self.mhin, self.mh, self.mx, self.mg = (z(M, H) for _ in range(5))
            self.mcat = z(M, 2 * H)
            self.mlogits = z(M, self.V)
            self.mam = z(M, dt=torch.int32)
        self.graphs_m: Dict[int, torch.cuda.CUDAGraph] = {}
        self.graphs_mkv: Dict[int, torch.cuda.CUDAGraph] = {}
        # Native image path (docs/serving/BIDEC_MULTIMODAL_20261004.md).  0 = text-only: no
        # scratch rows, no staging buffer, and _step is the rc7 instruction sequence exactly.
        # Above 0 the rotary table is EXTENDED by that many scratch rows (an image request's
        # own M-RoPE rows live there while it prefills) and image-placeholder rows take their
        # embedding from a staging buffer.  Both are allocated HERE, before capture(), because
        # the CUDA graphs capture these pointers.
        self.mm: Optional[bm.MMState] = None
        if int(mm_rope_rows) > 0:
            n_text = int(self.cosT.shape[0])
            rd = int(self.cosT.shape[1])
            self.cosT = torch.cat([self.cosT, z(int(mm_rope_rows), rd)]).contiguous()
            self.sinT = torch.cat([self.sinT, z(int(mm_rope_rows), rd)]).contiguous()
            self.mm = bm.MMState(self.cosT, self.sinT, n_text, int(mm_rope_rows), stage=z(M, H))
        self.meta = _Meta(dev, M, self.max_slots, self.max_pages, self.g_cap)
        self.meta.h("mropeoff")[:] = 1              # MTP cache row p <-> rope position p + delta + 1 (text: delta 0)
        self.pt = self.meta.h("pagetab").reshape(S1, self.max_pages)
        self.pt[:] = 0
        self.free_pages = list(range(self.pages_total, 0, -1))   # page 0: the dummy slot's
        self.free_slots = list(range(self.max_slots - 1, -1, -1))
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.dev.type == "cuda":
            torch.cuda.synchronize(dev)
        # EVERY value in state_bytes is a byte count, because the next line divides all of
        # them by 2**30.  rc2 carried the ring's dtype *name* in here as well and the log
        # statement died on `str / int` for everyone, every time, on the happy path right
        # after the buffers allocated (finding F4).  The dtype is already in /health as
        # `gdn_ring_dtype`, so it is kept beside the counts, never among them.
        self.state_bytes: Dict[str, int] = {
            "kv": self.kc.numel() * 4,
            "gdn_state": self.state.numel() * self.gdn_ring_bytes,
            "conv_hist": self.hist.numel() * 2,
            "pacc": self.pacc.numel() * 4 + self.pml.numel() * 4}
        if self.spec:
            # rc7 omitted the MTP head's own KV pages (1 MiB per page on the 27B) from the
            # receipt, so state_bytes under-reported the device bill with speculation on.
            self.state_bytes["mtp_kv"] = self.mkc.numel() * 4
        # The M-sized working set: logits, gate|up, the hidden rows -- all sized by max_rows,
        # so --max-rows is a memory knob as well as a scheduling one (state_budget.py).
        self.state_bytes["rows"] = sum(
            int(t.numel()) * t.element_size()
            for t in (self.h, self.x, self.proj, self.dn, self.zba, self.conv, self.go, self.aqkv,
                      self.q, self.ao, self.gu, self.act, self.logits))
        log(f"[bidec] buffers in {time.time() - t0:.1f}s: "
            + format_state_bytes(self.state_bytes)
            + f", gdn_ring_dtype {self.gdn_ring_dtype}, R {self.R}, conv_ring {self.conv_ring}"
            + f", max_rows {self.max_rows}")

    def _rope_tables(self, n: int):
        """HF's own rotary module at text positions t = h = w (fastdec's construction, longer).
        ``bidec_mm.text_rope_table`` -> ``fastdec.rope_tables`` -- one copy, which the CPU
        emulator of the image path also builds its text table with, and which reads the rotary
        layout (multimodal 3-axis or plain, rope_theta, partial rotary, static scaling) off the
        model's own module (ATTN_HEADDIM_20261004)."""
        rot = fdm._text(self.fd.model).rotary_emb
        cosT, sinT = bm.text_rope_table(rot, n, self.dev)
        n0 = self.fd.cosT.shape[0]
        if not (torch.equal(cosT[:n0], self.fd.cosT) and torch.equal(sinT[:n0], self.fd.sinT)):
            raise RuntimeError("rotary tables disagree with fastdec's")
        return cosT, sinT

    # ------------------------------------------------------------------ descriptors
    def _descs(self):
        ds = [self.lm]
        for b in self.L:
            ds += [b.gu, b.down] + ([b.qkvzba, b.out] if b.src.kind == "gdn" else [b.qkv, b.o])
        return ds

    def streamed_bytes(self) -> int:
        return sum(d.coded_bytes for d in self._descs())

    def gemm_backend_census(self) -> dict:
        """Native receipt identity, including MTP projections when present."""
        descriptors = self._descs()
        if self.spec:
            descriptors += [self.m.fc, self.m.qkv, self.m.o, self.m.gu, self.m.down]
        return bg.backend_census(descriptors)

    # ------------------------------------------------------------------ the step (graph body)
    def _step(self, Mb: int, *, project_head: bool = True) -> None:
        E, B, fd, eps, mt, nm = fdm.ext(), ext(), self.fd, self.eps, self.meta, self.norm_mode
        fd._embed(mt.vtok, self.h, Mb)
        if self.mm is not None:
            # image-placeholder rows take the vision tower's embedding (masked_scatter's bytes)
            bm.apply_embed_override(self.h, mt.mmflag, self.mm.stage, Mb)
        delta = None
        for b in self.L:
            o = b.src
            E.rmsnorm(self.h, delta, o.in_ln, self.x, Mb, eps, nm)
            if o.kind == "gdn":
                b.qkvzba(self.x[:Mb], self.zba[:Mb])
                B.b_gdn_conv(self.zba, self.hist[o.gi], o.convw, mt.seq_slot, mt.seq_row0, mt.seq_len, mt.n_seq,
                             mt.row_pos, self.conv, Mb)
                B.b_gdn_recur(self.conv, self.zba, o.z_off, o.b_off, o.a_off, o.A_log, o.dt_bias, o.nw,
                              self.state[o.gi], self.R, mt.seq_slot, mt.seq_row0, mt.seq_len, mt.n_seq,
                              mt.row_pos, self.go, Mb, eps, o.nkh_rep)
                b.out(self.go[:Mb], self.proj[:Mb])
            else:
                b.qkv(self.x[:Mb], self.aqkv[:Mb])
                B.b_attn_prep(self.aqkv, o.qnw, o.knw, self.cosT, self.sinT, mt.row_slot, mt.row_pos, mt.n_rows,
                              mt.ropeoff, mt.pagetab, self.max_pages, self.q, self.kc[o.ai], self.vc[o.ai], Mb,
                              o.NQ, o.NKV, eps, o.gated, o.qk_norm)
                B.b_attn_split(self.q, self.kc[o.ai], self.vc[o.ai], mt.row_slot, mt.row_pos, mt.item_row,
                               mt.item_split, mt.n_items, mt.pagetab, self.max_pages, self.pacc, self.pml,
                               self.grid_items, o.NQ, o.NKV, o.scaling)
                B.b_attn_combine(self.pacc, self.pml, self.aqkv, mt.row_pos, mt.row_item0, mt.n_rows, self.ao,
                                 Mb, o.NQ, o.gated)
                b.o(self.ao[:Mb], self.proj[:Mb])
            E.rmsnorm(self.h, self.proj, o.post_ln, self.x, Mb, eps, nm)
            b.gu(self.x[:Mb], self.gu[:Mb])
            E.silu_mul(self.gu, self.act, Mb)
            b.down(self.act[:Mb], self.dn[:Mb])
            delta = self.dn
        E.rmsnorm(self.h, delta, fd.final_ln, self.xf, Mb, eps, nm)
        if project_head:
            self.lm(self.xf[:Mb], self.logits[:Mb])
            E.argmax_rows(self.logits, self.am, Mb)

    def _mtp_step(self, Mb: int) -> None:
        """The MTP head over Mb rows: row r pairs hidden hsrc[hidx[r]] with token vtok[r] at MTP cache row
        row_pos[r]; flagged rows store their output hidden into the slot's MTP hidden store.  (Drafts only
        steer acceptance; they never change an emitted token or logit.)"""
        E, B, mt, m, eps, H = fdm.ext(), ext(), self.meta, self.m, self.eps, self.H
        nm, g, qk = self.norm_mode, self.attn_gated, self.qk_norm
        self.fd._embed(mt.vtok, self.me, Mb)
        torch.index_select(self.hsrc, 0, mt.hidx[:Mb], out=self.mhin[:Mb])
        E.rmsnorm(self.me, None, m.pre_e, self.mcat[:, :H], Mb, eps, nm)
        E.rmsnorm(self.mhin, None, m.pre_h, self.mcat[:, H:], Mb, eps, nm)
        m.fc(self.mcat[:Mb], self.mh[:Mb])
        E.rmsnorm(self.mh, None, m.in_ln, self.mx, Mb, eps, nm)
        m.qkv(self.mx[:Mb], self.aqkv[:Mb])
        B.b_attn_prep(self.aqkv, m.qnw, m.knw, self.cosT, self.sinT, mt.row_slot, mt.row_pos, mt.n_rows, mt.mropeoff,
                      mt.pagetab, self.max_pages, self.q, self.mkc, self.mvc, Mb, self.NQ, self.NKV, eps, g, qk)
        B.b_attn_split(self.q, self.mkc, self.mvc, mt.row_slot, mt.row_pos, mt.item_row, mt.item_split, mt.n_items,
                       mt.pagetab, self.max_pages, self.pacc, self.pml, self.grid_items, self.NQ, self.NKV, m.scaling)
        B.b_attn_combine(self.pacc, self.pml, self.aqkv, mt.row_pos, mt.row_item0, mt.n_rows, self.ao, Mb, self.NQ, g)
        m.o(self.ao[:Mb], self.proj[:Mb])
        E.rmsnorm(self.mh, self.proj, m.post_ln, self.mx, Mb, eps, nm)
        m.gu(self.mx[:Mb], self.gu[:Mb])
        E.silu_mul(self.gu, self.act, Mb)
        m.down(self.act[:Mb], self.dn[:Mb])
        E.rmsnorm(self.mh, self.dn, m.norm, self.mg, Mb, eps, nm)
        self.lm(self.mg[:Mb], self.mlogits[:Mb])
        E.argmax_rows(self.mlogits, self.mam, Mb)
        B.b_scatter_rows(self.mg, mt.hflag, mt.row_slot, mt.n_rows, self.mhid, Mb)

    def _mtp_kv_step(self, Mb: int) -> None:
        """Fill shifted prompt KV from trunk hidden rows, without MTP attention or logits."""
        E, B, mt, m, eps, H = fdm.ext(), ext(), self.meta, self.m, self.eps, self.H
        nm = self.norm_mode
        self.fd._embed(mt.vtok, self.me, Mb)
        torch.index_select(self.hsrc, 0, mt.hidx[:Mb], out=self.mhin[:Mb])
        E.rmsnorm(self.me, None, m.pre_e, self.mcat[:, :H], Mb, eps, nm)
        E.rmsnorm(self.mhin, None, m.pre_h, self.mcat[:, H:], Mb, eps, nm)
        m.fc(self.mcat[:Mb], self.mh[:Mb])
        E.rmsnorm(self.mh, None, m.in_ln, self.mx, Mb, eps, nm)
        m.qkv(self.mx[:Mb], self.aqkv[:Mb])
        B.b_attn_prep(self.aqkv, m.qnw, m.knw, self.cosT, self.sinT, mt.row_slot, mt.row_pos, mt.n_rows,
                      mt.mropeoff, mt.pagetab, self.max_pages, self.q, self.mkc, self.mvc, Mb,
                      self.NQ, self.NKV, eps, self.attn_gated, self.qk_norm)

    def capture(self, buckets: Optional[Sequence[int]] = None) -> Dict[int, float]:
        """One CUDA graph per row bucket.  Metadata of a harmless all-padding step is uploaded
        first (every row -> the dummy slot at position 0)."""
        out = {}
        self._set_padding_only(max(buckets or self.buckets))
        s = torch.cuda.Stream(self.dev)
        s.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(s):
            for Mb in (buckets or self.buckets):
                if Mb in self.graphs:
                    continue
                t0 = time.perf_counter()
                self._set_padding_only(Mb)
                self._step(Mb)
                torch.cuda.synchronize(self.dev)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s):
                    self._step(Mb)
                torch.cuda.synchronize(self.dev)
                self.graphs[Mb] = g
                if self.selective_head:
                    self._step(Mb, project_head=False)
                    torch.cuda.synchronize(self.dev)
                    gt = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gt, stream=s):
                        self._step(Mb, project_head=False)
                    torch.cuda.synchronize(self.dev)
                    self.graphs_trunk[Mb] = gt
                if self.spec:
                    self._mtp_step(Mb)
                    torch.cuda.synchronize(self.dev)
                    gm = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gm, stream=s):
                        self._mtp_step(Mb)
                    torch.cuda.synchronize(self.dev)
                    self.graphs_m[Mb] = gm
                    self._mtp_kv_step(Mb)
                    torch.cuda.synchronize(self.dev)
                    gkv = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gkv, stream=s):
                        self._mtp_kv_step(Mb)
                    torch.cuda.synchronize(self.dev)
                    self.graphs_mkv[Mb] = gkv
                out[Mb] = round(time.perf_counter() - t0, 2)
        torch.cuda.current_stream(self.dev).wait_stream(s)
        return out

    def _set_padding_only(self, Mb: int) -> None:
        m = self.meta
        m.h("vtok")[:] = 0
        m.h("row_slot")[:] = self.dummy
        m.h("row_pos")[:] = 0
        m.h("row_item0")[:] = np.arange(len(m.h("row_item0")))
        m.h("item_row")[:Mb] = np.arange(Mb)
        m.h("item_split")[:] = 0
        m.h("hidx")[:] = 0
        m.h("hflag")[:] = 0
        m.h("mmflag")[:] = 0
        m.h("n_rows")[0] = Mb
        m.h("n_seq")[0] = 0
        m.h("n_items")[0] = Mb
        m.upload()
        torch.cuda.current_stream(self.dev).synchronize()

    # ------------------------------------------------------------------ slots and pages
    def alloc_slot(self) -> int:
        return self.free_slots.pop()

    def free_slot(self, slot: int) -> None:
        for p in self.pt[slot]:
            if p:
                self.free_pages.append(int(p))
        self.pt[slot] = 0
        if self.mm is not None:
            self.mm.detach(slot)
        self.free_slots.append(slot)

    # ------------------------------------------------------------------ image requests
    def mm_can_attach(self, prompt_len: int) -> bool:
        return self.mm is not None and self.mm.can_attach(prompt_len)

    def attach_mm(self, slot: int, plan: "bm.MMPlan") -> None:
        """Bind an image request's plan (embeddings + M-RoPE rows) to its slot, before its
        first prompt row runs.  Released by ``free_slot``."""
        if self.mm is None:
            raise RuntimeError("this BatchDecoder was built without --mm-rope-rows; it cannot "
                               "run an image request")
        self.mm.attach(slot, plan)

    def ensure_pages(self, slot: int, upto_pos: int) -> None:
        need = upto_pos // PAGE + 1
        if need > self.max_pages:
            raise ValueError(f"position {upto_pos} exceeds max_ctx {self.max_pages * PAGE}")
        row = self.pt[slot]
        for i in range(need):
            if row[i] == 0:
                if not self.free_pages:
                    raise RuntimeError("KV page pool exhausted")
                row[i] = self.free_pages.pop()

    # ------------------------------------------------------------------ work-list capacity
    @staticmethod
    def items_for_span(pos0: int, n: int) -> int:
        """Attention work-list entries for ``n`` rows starting at position ``pos0``.

        One entry per (row, KV page) pair: row at position p enumerates ``p // PAGE + 1``
        pages.  For a decode row that is a handful; for a prompt chunk deep into a long
        context it is ~pos/PAGE per row, so the work list grows with context while the ROW
        budget does not notice.  That is why this is computed, not assumed.
        """
        return sum((pos0 + t) // PAGE + 1 for t in range(int(n)))

    def items_for(self, seqs: Sequence[Tuple[int, int, Sequence[int]]]) -> int:
        """Work-list entries one ``run(seqs)`` would write, padding rows included.

        rc2 finding F5: ``run`` wrote the list first and checked ``items > g_cap``
        afterwards, so an overflowing step died inside numpy as
        ``could not broadcast input array from shape (5,) into shape (4,)`` -- which is
        what killed the engine on the ctx8k class and cost both long-context rows.  The
        capacity question is now answerable BEFORE anything is written.
        """
        M = sum(len(t) for _, _, t in seqs)
        items = sum(self.items_for_span(p0, len(t)) for _, p0, t in seqs)
        return items + (bucket_for(M, self.buckets) - M)       # padding rows: 1 entry each

    def reset_slot_state(self, slot: int) -> None:
        """A fresh sequence starts from zero conv history and zero recurrent state."""
        self.hist[:, slot].zero_()
        self.state[:, slot].zero_()

    # ------------------------------------------------------------------ one step
    def run(self, seqs: List[Tuple[int, int, List[int]]], *,
            output_rows: Optional[Sequence[int]] = None) -> Tuple[int, List[Tuple[int, int]]]:
        """seqs: [(slot, pos0, token_ids)] -- each sequence feeds its tokens at positions
        pos0 .. pos0+len-1 (len 1 = decode, >1 = prefill chunk / verify).  Returns
        (Mb, row ranges) after the graph replay; logits / argmax rows are in self.logits / self.am."""
        real_rows = sum(len(t) for _, _, t in seqs)
        selected = None if output_rows is None else [int(r) for r in output_rows]
        if selected is not None and (len(set(selected)) != len(selected) or
                                     any(r < 0 or r >= real_rows for r in selected)):
            raise ValueError("output_rows must be unique real row indices")
        need_items = self.items_for(seqs)
        if need_items > self.g_cap:
            raise WorkListOverflow(
                f"attention work list {need_items} > capacity {self.g_cap} for this step "
                f"({sum(len(t) for _, _, t in seqs)} rows, max position "
                f"{max((p0 + len(t) - 1) for _, p0, t in seqs) if seqs else 0}); feed fewer "
                f"prompt tokens per step (Batcher does this automatically) or raise --pages")
        m = self.meta
        vt, rs, rp, ri0 = m.h("vtok"), m.h("row_slot"), m.h("row_pos"), m.h("row_item0")
        ss, sr0, sl = m.h("seq_slot"), m.h("seq_row0"), m.h("seq_len")
        itr, its = m.h("item_row"), m.h("item_split")
        r = 0
        items = 0
        ranges = []
        for slot, _p0, _t in seqs:
            # The GDN state / conv history exist for real slots only (see __init__): a sequence
            # on the dummy slot or beyond would index past them.  Checked before any write.
            if not 0 <= int(slot) < self.max_slots:
                raise ValueError(f"sequence on slot {slot}: real slots are 0..{self.max_slots - 1}")
        for b, (slot, pos0, toks) in enumerate(seqs):
            n = len(toks)
            self.ensure_pages(slot, pos0 + n - 1)
            ss[b], sr0[b], sl[b] = slot, r, n
            ranges.append((r, n))
            for t in range(n):
                p = pos0 + t
                vt[r], rs[r], rp[r], ri0[r] = toks[t], slot, p, items
                ns = p // PAGE + 1
                itr[items:items + ns] = r
                its[items:items + ns] = np.arange(ns)
                items += ns
                r += 1
        M = r
        Mb = bucket_for(M, self.buckets)
        for rr in range(M, Mb):                           # padding rows: dummy slot, position 0
            vt[rr], rs[rr], rp[rr], ri0[rr] = 0, self.dummy, 0, items
            itr[items], its[items] = rr, 0
            items += 1
        if items > self.g_cap:
            raise RuntimeError(f"attention work list {items} > capacity {self.g_cap}")
        m.h("n_rows")[0] = Mb
        m.h("n_seq")[0] = len(seqs)
        m.h("n_items")[0] = items
        if self.mm is not None:
            # per-slot ropeoff (scratch range while an image prompt prefills, rope_delta after),
            # image-row flags, and the staged image embeddings for this step's image rows
            self.mm.prepare_step(seqs, ranges, m.h("ropeoff"), m.h("mmflag"))
        m.upload()
        selective = (self.selective_head and selected is not None and
                     set(selected) != set(range(real_rows)))
        graphs = self.graphs_trunk if selective else self.graphs
        g = graphs.get(Mb)
        if g is None:
            self._step(Mb, project_head=not selective)
        else:
            g.replay()
        if selective:
            if selected:
                ix = torch.tensor(selected, dtype=torch.long, device=self.dev)
                x = torch.index_select(self.xf, 0, ix)
                y = torch.empty((len(selected), self.V), dtype=self.logits.dtype, device=self.dev)
                a = torch.empty(len(selected), dtype=self.am.dtype, device=self.dev)
                # S is fixed on the descriptor. Preserve the baseline launch tile too:
                # changing M otherwise changes tile_for at its dispatch knee.
                tile = bg.tile_for(Mb, getattr(self.lm, "tile_cap", 64), self.lm.N, self.lm.K)
                self.lm(x, y, tile=tile)
                fdm.ext().argmax_rows(y, a, len(selected))
                self.logits.index_copy_(0, ix, y)
                self.am.index_copy_(0, ix, a)
                self.head_rows_total += len(selected)
        else:
            self.head_rows_total += Mb
        return Mb, ranges


def _run_mtp(self, rows: List[Tuple[int, int, int, int, int]]) -> int:
    """rows: [(slot, mtp_pos, token, hidx, flag)] -> replay the MTP graph; drafts in self.mam."""
    need_items = (sum(pos // PAGE + 1 for _, pos, _, _, _ in rows)
                  + (bucket_for(len(rows), self.buckets) - len(rows)))
    if need_items > self.g_cap:                        # checked before any write (F5)
        raise WorkListOverflow(
            f"MTP work list {need_items} > capacity {self.g_cap} for {len(rows)} rows")
    m = self.meta
    vt, rs, rp, ri0 = m.h("vtok"), m.h("row_slot"), m.h("row_pos"), m.h("row_item0")
    itr, its, hi, hf = m.h("item_row"), m.h("item_split"), m.h("hidx"), m.h("hflag")
    items = 0
    for r, (slot, pos, tok, hidx, flag) in enumerate(rows):
        self.ensure_pages(slot, pos)
        vt[r], rs[r], rp[r], ri0[r], hi[r], hf[r] = tok, slot, pos, items, hidx, flag
        ns = pos // PAGE + 1
        itr[items:items + ns] = r
        its[items:items + ns] = np.arange(ns)
        items += ns
    M = len(rows)
    Mb = bucket_for(M, self.buckets)
    for rr in range(M, Mb):
        vt[rr], rs[rr], rp[rr], ri0[rr], hi[rr], hf[rr] = 0, self.dummy, 0, items, 0, 0
        itr[items], its[items] = rr, 0
        items += 1
    if items > self.g_cap:
        raise RuntimeError(f"attention work list {items} > capacity {self.g_cap}")
    m.h("n_rows")[0] = Mb
    m.h("n_seq")[0] = 0
    m.h("n_items")[0] = items
    m.upload()
    g = self.graphs_m.get(Mb)
    if g is None:
        self._mtp_step(Mb)
    else:
        g.replay()
    return Mb


BatchDecoder.run_mtp = _run_mtp


def _run_mtp_kv(self, rows: List[Tuple[int, int, int, int]]) -> int:
    """Store prompt MTP KV: (slot, prompt position, next prompt token, trunk hidx)."""
    if not rows:
        return 0
    Mb = bucket_for(len(rows), self.buckets)
    m = self.meta
    vt, rs, rp, hi = (m.h(name) for name in ("vtok", "row_slot", "row_pos", "hidx"))
    for r, (slot, pos, tok, hidx) in enumerate(rows):
        if not 0 <= int(slot) < self.max_slots or not 0 <= int(hidx) < self.hsrc.shape[0]:
            raise ValueError("MTP prompt row has an invalid slot or hidden index")
        self.ensure_pages(slot, pos)
        vt[r], rs[r], rp[r], hi[r] = tok, slot, pos, hidx
    for r in range(len(rows), Mb):
        vt[r], rs[r], rp[r], hi[r] = 0, self.dummy, 0, 0
    m.h("n_rows")[0] = Mb
    m.h("n_seq")[0] = 0
    m.h("n_items")[0] = 0
    m.upload()
    g = self.graphs_mkv.get(Mb)
    if g is None:
        self._mtp_kv_step(Mb)
    else:
        g.replay()
    return Mb


BatchDecoder.run_mtp_kv = _run_mtp_kv


# ---------------------------------------------------------------------------- scheduler
# ---------------------------------------------------------------------------- speculation glue
_BATCHER_SPEC_ADAPTER = None


def _batcher_spec_adapter_cls():
    """``bidec_spec.DecoderAdapter`` that puts the SCHEDULER's rows in the forward.

    Built on first use so that a server with speculation off keeps rc6's import graph exactly.

    Two things the plain ``DecoderAdapter`` cannot do, both of which the gate on CPU would
    otherwise not exercise:

    * the prompt chunks of still-prefilling sequences go into the SAME forward as the verify
      blocks, after them, which is the row order plain ``Batcher.step`` produces; and
    * the whole work list goes through ``Batcher._fit_work_list`` before ``run``.  The
      attention work list has one entry per (row, KV page) pair, so a 256-row chunk at
      position 8000 wants ~8,000 entries against a capacity of ``pages_total + max_rows + 8``;
      rc2 had no item budget and the overflow landed as a numpy broadcast error that stopped
      the engine on the ctx8k class (finding F5).  ``DecoderAdapter.trunk_verify`` calls
      ``d.run(seqs)`` directly, so routing speculation through it unchanged would have put
      that failure back on the speculative path only -- visible on a long context, under load,
      and nowhere in the CPU gate.
    """
    global _BATCHER_SPEC_ADAPTER
    if _BATCHER_SPEC_ADAPTER is None:
        from glc_serve.bidec_spec import DecoderAdapter

        class _BatcherSpecAdapter(DecoderAdapter):
            def __init__(self, batcher: "Batcher", mtp=None):
                super().__init__(batcher.bd, lambda rid: self.slots[rid], mtp,
                                 digest=False)
                self.B = batcher
                self.slots: Dict[Any, int] = {}
                self.owner: Dict[Any, Seq] = {}
                self.prefill_src: List[Seq] = []
                self.prefill: List[Tuple[Tuple[Seq, int], Tuple[int, int]]] = []
                self.am: List[int] = []
                self.last_Mb = 0
                self.last_rows_run = 0

            def trunk_verify(self, reqs):
                self.last_rows.clear()
                self.last_pos.clear()
                B = self.B
                seqs: List[Tuple[int, int, List[int]]] = []
                owners: List[Tuple[Seq, Any]] = []
                for rid, pos0, toks in reqs:
                    seqs.append((self.slots[rid], int(pos0), [int(t) for t in toks]))
                    owners.append((self.owner[rid], "dec"))
                n_verify = len(seqs)
                # The row budget is the SAME budget plain ``Batcher.step`` spends, and
                # speculation spends from it too: the verify rows are already in ``seqs``, so
                # what is left over is what prompt chunks may use.  rc6 budgeted prefill
                # against ``max_rows_step`` (256) here even when ``--row-budget-tiles`` had
                # capped the step at one 64-row tile, which would have put a 256-row chunk in
                # a forward the row-budget arm says is one tile wide -- i.e. the budget would
                # have held for the non-speculative path and not for the speculative one.
                budget = B.row_budget() - sum(len(x[2]) for x in seqs)
                for s in self.prefill_src:
                    if budget <= 0:
                        break
                    c = B._chunk_len(s, budget)
                    if c <= 0:
                        continue
                    seqs.append((s.slot, s.fed, list(s.prompt[s.fed:s.fed + c])))
                    owners.append((s, c))
                    budget -= c
                self.prefill, self.am, self.last_Mb, self.last_rows_run = [], [], 0, 0
                if not seqs:
                    return {}
                B._fit_work_list(seqs, owners)
                if len(seqs) < n_verify:
                    # _fit_work_list only ever trims or drops PROMPT chunks (it refuses to
                    # touch a "dec" owner and raises WorkListOverflow when decode rows alone
                    # are over capacity), so the verify blocks are still seqs[:n_verify].
                    # Asserting it rather than assuming it, because the whole exactness
                    # argument rests on a verify block arriving whole.
                    raise WorkListOverflow(
                        "the work-list budget removed a verify block; speculation needs the "
                        "whole block in one forward")
                if not seqs:
                    return {}
                Mb, ranges = B._run_selected(seqs, owners)
                self.am = [int(x) for x in self.d.am[:Mb].tolist()]
                self.last_Mb = Mb
                self.last_rows_run = sum(n for _, n in ranges)
                self.steps += 1
                self.rows_total += self.last_rows_run
                out: Dict[Any, List[int]] = {}
                for i, (rid, pos0, toks) in enumerate(reqs):
                    r0, n = ranges[i]
                    self.last_rows[rid] = (r0, n)
                    self.last_pos[rid] = int(pos0)
                    out[rid] = self.am[r0:r0 + n]
                self.prefill = list(zip(owners[n_verify:], ranges[n_verify:]))
                if self.mtp is not None and hasattr(self.d, "run_mtp_kv"):
                    # An MTP cache row at prompt position p uses the trunk hidden from p
                    # and the token at p+1.  The last prompt row is handled by a full MTP
                    # pass after its first token is emitted.  Do this now: the next trunk
                    # forward replaces xf, even when another slot is decoding meanwhile.
                    kv = []
                    for (s, c), (r0, n) in self.prefill:
                        for j in range(n):
                            p = s.fed + j
                            if p + 1 < len(s.prompt):
                                kv.append((s.slot, p, int(s.prompt[p + 1]),
                                           self.d.hsrc_trunk0 + r0 + j))
                    if kv:
                        self.d.run_mtp_kv(kv)
                return out

        _BATCHER_SPEC_ADAPTER = _BatcherSpecAdapter
    return _BATCHER_SPEC_ADAPTER


@dataclass
class Seq:
    rid: Any
    prompt: List[int]
    max_new: int
    stop_ids: Sequence[int] = ()
    temperature: float = 0.0
    top_p: float = 1.0
    seed: Optional[int] = None
    digest: bool = False
    slot: int = -1
    fed: int = 0                      # prompt tokens fed so far
    adopted: int = 0                  # prompt positions adopted from the prefix cache (never fed)
    out: List[int] = field(default_factory=list)
    digests: List[str] = field(default_factory=list)
    done: bool = False
    finish: str = ""
    reserved_pages: int = 0
    t_admit: float = 0.0
    t_first: float = 0.0
    gen: Any = None
    on_token: Any = None              # callback(seq, token) from the engine thread
    drafts: List[int] = field(default_factory=list)
    prop: int = 0                     # speculative drafts proposed / accepted
    acc: int = 0
    cancelled: bool = False           # client gone / aborted; reaped at the next step boundary
    mm: Any = None                    # bidec_mm.MMInputs of an image request (None = text)
    mm_plan: Any = None               # its bidec_mm.MMPlan once encoded at admission
    error: str = ""                   # why an admission-time failure ended the request


class Batcher:
    """Continuous batching over a BatchDecoder.  Each step: one row per decoding sequence plus
    prompt chunks of prefilling sequences (row budget ``max_rows_step``, chunk ``prefill_chunk``).
    Pages are reserved at admission for prompt + max_new, so a running request can never run
    out of KV."""

    def __init__(self, bd: BatchDecoder, *, max_rows_step: int = 256, prefill_chunk: int = 256,
                 max_active: Optional[int] = None, spec_k: int = 0, chunk_align: int = 1,
                 max_decode_rows: int = 64, prefill_priority: float = 0.0,
                 row_tile: int = 64, row_budget_tiles: int = 0,
                 row_budget_rows: int = 0,
                 gemm_tile: Optional[int] = None,
                 prefix_cache: Optional[Any] = None, admission_policy: Optional[Any] = None,
                 spec_ngram: bool = False, mm_encoder: Optional[Any] = None):
        self.bd = bd
        # Image requests (docs/serving/BIDEC_MULTIMODAL_20261004.md): ``mm_encoder(MMInputs) ->
        # MMPlan`` runs the vision tower for ONE request, in this (the engine) thread, at
        # admission.  None = this scheduler cannot admit an image request.
        self.mm_encoder = mm_encoder
        self.mm_errors = 0
        # Optional shared-prefix KV reuse (glc_serve.bidec_prefix.PrefixCacheAdapter, or anything
        # satisfying bidec_iface.PrefixCacheLike).  None = rc6 behaviour exactly: every prompt is
        # prefilled in full.  The cache is driven from THIS class -- lookup/adopt at admission,
        # publish at a block boundary, release at finish -- rather than by a test harness reaching
        # into Batcher._admit from outside, which is how the prefix branch's gate drove it and is
        # the reason the knob counted as unreleased (rc6 rule: a knob no shipped entry point can
        # set has not been released).
        self.prefix_cache = prefix_cache
        # Optional bidec_iface.AdmissionPolicy (glc_serve.bidec_admission.MaxMinAdmission).  None =
        # FIFO, which is what rc6 shipped.
        self.admission_policy = admission_policy
        self.prefix_adopted = 0        # prompt positions served from cache, total
        self.prefix_requests = 0       # requests that adopted at least one block
        self.prefix_errors = 0         # adoptions that refused/raised and fell back to full prefill
        # A prompt chunk may end anywhere as far as exactness goes (gated: bi_gate_batchexact.py
        # runs chunk 97).  It may NOT end anywhere as far as a prefix cache goes: the GDN
        # recurrent state can only be snapshotted on a PAGE-token block boundary, so a chunk that
        # ends mid-block can never publish a cacheable prefix.  chunk_align makes every
        # non-final chunk end on a multiple of it; 1 disables the alignment.
        self.chunk_align = max(1, int(chunk_align))
        self.spec_k = int(spec_k)
        # Which DRAFTER is available, and whether speculation can run at all, are separate
        # questions.  rc6 refused spec_k > 0 without an MTP head, which also refused the only
        # drafter that exists on the CPU reference (n-gram / prompt-lookup) and so made every
        # speculation statement a GPU-only statement.  Exactness does not depend on the
        # drafter -- drafts never reach the output, they only decide which tokens occupy the
        # verify rows -- so the requirement is "a drafter", not "the MTP head".
        self.spec_ngram = bool(spec_ngram)
        # True when the engine turned n-gram drafting on although it was not asked for, which
        # is the one case /health's `spec_ngram` may differ from the argv without that being a
        # dropped knob.  It is a separate key so the difference stays ATTRIBUTABLE.
        self.spec_ngram_forced = False
        _has_mtp = bool(bd.spec) and bool(getattr(bd, "mtp_modelled", True))
        self._mtp_prefix_ready = True
        if self.prefix_cache is not None:
            cache = getattr(self.prefix_cache, "cache", None)
            store = getattr(cache, "store", None)
            mtp_prefix = bool(self.spec_k and _has_mtp)
            if store is not None and getattr(store, "bd", None) is bd and hasattr(store, "configure_mtp"):
                old_mode = bool(getattr(store, "mtp_enabled", False))
                self._mtp_prefix_ready = bool(store.configure_mtp(mtp_prefix))
                if self._mtp_prefix_ready and old_mode != mtp_prefix:
                    cache.invalidate_all("MTP prefix layout changed")
            elif mtp_prefix:
                self._mtp_prefix_ready = False
        if self.spec_k and not _has_mtp and not self.spec_ngram:
            if not hasattr(bd, "run_mtp"):
                raise ValueError("speculation needs the MTP head (load with spec=True) or "
                                 "n-gram drafting (spec_ngram=True)")
            self.spec_ngram = True
            self.spec_ngram_forced = True
        self._specctl = None           # bidec_spec.BatchSpecController, built on first step
        self._specad = None            # its engine adapter
        if self.spec_k and needs_gdn_state_ring(bd) and bd.R < self.spec_k + 1:
            raise ValueError(f"spec_k={spec_k} needs a GDN state ring R >= {spec_k + 1} (have {bd.R})")
        # The conv-history ring has the same rollback hazard as the recurrent ring, with a
        # different bound: a rejected draft overwrites slot q & (N-1) and the next verify block
        # reads three positions back, so N >= spec_k + 3 (state_budget.conv_ring_for).  rc7's
        # fixed 16 broke at spec_k >= 14 and nothing refused it.
        _cr = getattr(bd, "conv_ring", None)
        if self.spec_k and _cr is not None and int(_cr) < self.spec_k + 3:
            raise ValueError(f"spec_k={spec_k} needs a GDN conv ring >= {spec_k + 3} "
                             f"(have {_cr}); pass --gdn-conv-ring 0 to derive it")
        self.max_rows_step = min(int(max_rows_step), bd.max_rows)
        # Decode rows and prefill rows do not cost the same.  Measured gate|up: 237.9 us for
        # M=1..32, 261.2 at 64, 479.0 at 128 -- the weight traffic is shared up to the knee and
        # then is not, so a decode step past 64 rows pays nearly double for rows that share no
        # traffic, and a bucket beyond the live row count pads at pure loss.  Decode rows are
        # therefore capped at the knee; the 96..512 buckets stay available for prefill chunks,
        # where the rows do share weight traffic.
        self.max_decode_rows = max(1, min(int(max_decode_rows), self.max_rows_step))
        self.prefill_chunk = int(prefill_chunk)
        # Prefill-aware admission (run-3 finding: ctx8k TTFT).  0 = rc4 behaviour exactly: every
        # admissible request starts prefilling at once and they all share the work-list budget,
        # so deep in a long context each gets a sliver and NONE finishes early.  Above 0, the
        # number of sequences prefilling at once is capped so each gets at least one whole
        # aligned block, and a prompt chunk may grow into the entry budget when no decode row
        # wants it.  Total prefill work is unchanged -- what changes is when each prompt FINISHES,
        # which is what TTFT measures.  The projection is in glc_serve.bidec_policy.simulate.
        self.prefill_priority = max(0.0, min(1.0, float(prefill_priority)))
        # Budgeted chunked prefill (run-5 finding).  rc6 shipped a row budget of
        # `max_rows_step` (256 = four 64-row BI-GEMM tiles) and an item budget raised to 5,096
        # by --prefill-priority, and run 5 measured the consequence on the card: the step that
        # carried the extra prompt rows also crossed from one tile to three, the weight set is
        # re-decoded PER TILE, so every decode row riding in that step was stretched by the
        # same factor.  ctx8k N=32 per-user decode fell 13.23 -> 4.82 tok/s while the prefill
        # token rate rose only 606 -> 808 (1.33x -- exactly the tile-occupancy ratio
        # 0.917/0.688, i.e. all of the gain was filling the partial tile and none of it was new
        # throughput).  Receipts: .icc/evidence/fifth-gpu-run-rc6-20261004/bench/glc_ctx8k*/.
        #
        # Total row throughput is therefore ~CONSTANT (one tile per step: ~850 rows/s measured
        # on the RTX PRO 6000) and decode and prefill SHARE it.  `row_budget_tiles` makes that
        # sharing explicit: the TOTAL rows in a step -- decode rows first, prompt chunks
        # filling what is left, which is already the order `step()` uses -- are capped at whole
        # tiles, so per-user decode latency is pinned at the single-tile step time and prefill
        # still advances every step with whatever rows decode did not want.  0 = off, i.e. rc6
        # behaviour byte for byte.  Raising the tile HEIGHT is the only lever that raises the
        # total, and that is a CUDA change (the 128-row kernel), not a policy.
        # The row budget is counted in BI-GEMM tiles, so `row_tile` must track the KERNEL's
        # tile height.  With the 128-row kernel selected (`--gemm-tile 128`) a tile is 128
        # rows, and a budget of one tile is then 128 rows for one weight pass.  The server
        # defaults `--row-tile` to the kernel tile for exactly this reason; passing them
        # inconsistently is allowed (a test may want it) and reported on /health.
        self.row_tile = max(1, int(row_tile))
        self.gemm_tile = int(gemm_tile) if gemm_tile else int(getattr(bd, "gemm_tile", 64) or 64)
        self.row_budget_tiles = max(0, int(row_budget_tiles))
        # rc8: the row budget in ROWS (0 = use row_budget_tiles).  Run 6a and the corrected
        # step-time model (glc_serve.bidec_policy, docs/serving/RC8_FIXES_20261004.md) show the
        # step's marginal cost is dominated by a PER-ROW term (~0.57 ms/row measured), not by
        # the per-tile weight pass, so the budget is a decode-LATENCY budget and its right value
        # need not be a multiple of the tile.  When set it takes precedence over the tile count.
        self.row_budget_rows = max(0, int(row_budget_rows))
        self.max_active = int(max_active or bd.max_slots)
        self.waiting: List[Seq] = []
        self.active: List[Seq] = []
        self.reserved = 0
        self.steps = 0
        self.rows_total = 0
        self.trims = 0                 # prompt chunks shrunk to fit the work list (F5)
        self.lock = threading.Lock()

    def row_budget(self) -> int:
        """Total rows (decode + prompt chunks) one step may carry.

        ``max_rows_step`` when ``row_budget_tiles`` is 0, which is the shipped rc6 behaviour.
        """
        if getattr(self, "row_budget_rows", 0):
            return max(1, min(self.max_rows_step, self.row_budget_rows))
        if not self.row_budget_tiles:
            return self.max_rows_step
        return max(1, min(self.max_rows_step, self.row_tile * self.row_budget_tiles))

    def add(self, s: Seq) -> None:
        need = (len(s.prompt) + s.max_new + PAGE) // PAGE + 1
        if need > self.bd.pages_total or len(s.prompt) + s.max_new + 1 > self.bd.max_pages * PAGE:
            raise ValueError(f"request needs {len(s.prompt) + s.max_new} tokens of context; "
                             f"cap is {min(self.bd.pages_total, self.bd.max_pages) * PAGE}")
        s.reserved_pages = need
        with self.lock:
            self.waiting.append(s)

    def _prefill_seq_limit(self) -> int:
        """How many sequences may prefill at once.  ``10**9`` (i.e. no limit) at priority 0."""
        if not self.prefill_priority:
            return 10 ** 9
        from glc_serve.bidec_policy import PrefillPolicy

        pref = [s for s in self.active if s.fed < len(s.prompt) and not s.done]
        mean_pos = (sum(s.fed for s in pref) / len(pref)) if pref else 0.0
        pol = PrefillPolicy(prefill_priority=self.prefill_priority,
                            prefill_chunk=self.prefill_chunk, chunk_align=self.chunk_align,
                            max_rows_step=self.max_rows_step,
                            max_decode_rows=self.max_decode_rows)
        return pol.prefill_seq_limit(self.bd.g_cap, mean_pos)

    def _policy_reorder(self) -> None:
        """Let an optional ``bidec_iface.AdmissionPolicy`` pick the admission ORDER.

        Called with ``self.lock`` held, before the admission loop, which then pops from the
        front as always.  The protocol's own contract is that admission order may change a
        request's latency and never its tokens, so the reordering is a scheduling decision
        only: rows are still produced one per sequence, in ``self.active`` order, and a row's
        value depends on its own sequence alone (``DecoderLike.run``).  A policy that raises,
        or returns indices that are not a permutation prefix of ``waiting``, is ignored for
        that step rather than allowed to drop a request on the floor.
        """
        pol = self.admission_policy
        if pol is None or len(self.waiting) < 2:
            return
        try:
            idx = pol.choose(list(self.waiting), list(self.active),
                             free_slots=len(self.bd.free_slots),
                             free_pages=max(0, self.bd.pages_total - self.reserved),
                             max_active=self.max_active)
            order = [int(i) for i in idx]
        except Exception:  # noqa: BLE001
            # CLASSIFICATION: production_fallback.  A policy is advisory; a broken one must
            # degrade to FIFO, which is a correct schedule, not stop the engine.
            self.policy_errors = getattr(self, "policy_errors", 0) + 1
            return
        if not order or len(set(order)) != len(order) or any(not 0 <= i < len(self.waiting) for i in order):
            return
        chosen = [self.waiting[i] for i in order]
        rest = [s for j, s in enumerate(self.waiting) if j not in set(order)]
        self.waiting = chosen + rest

    def _adopt_prefix(self, s: Seq) -> None:
        """Serve the leading whole blocks of ``s.prompt`` from the shared prefix cache.

        Runs immediately after ``alloc_slot`` + ``reset_slot_state``, which is the only point
        where it is safe: the slot's pages are owned by this sequence and nothing has been fed
        yet.  ``s.fed`` is advanced to the number of adopted positions, so the ordinary chunked
        prefill resumes from there and every later row is produced by the normal path.

        At most ``len(prompt) - 1`` positions may be adopted: the row that produces the first
        token has to actually run.  Adoption is refused rather than approximated -- a cache
        that cannot hand back a recurrent snapshot at the boundary returns 0 and the prompt is
        prefilled in full, because the GDN recurrent state cannot be rewound.
        """
        c = self.prefix_cache
        if c is None or s.slot < 0 or s.mm is not None or not self._mtp_prefix_ready:
            # An image request's placeholder TOKENS are the same for every image, so a
            # token-keyed prefix cache would hand it another image's KV.  Never adopt.
            return
        try:
            n = int(c.lookup(s.prompt))
            if n <= 0:
                return
            block = int(getattr(c, "block_size", PAGE))
            keep = min(n, (len(s.prompt) - 1) // block * block)
            if keep < block:
                return
            got = int(c.adopt(s.slot, s.prompt, keep))
            if got <= 0:
                return
            got = min(got, len(s.prompt) - 1)
            s.fed = got
            s.adopted = got
            self.prefix_adopted += got
            self.prefix_requests += 1
        except Exception:  # noqa: BLE001
            # CLASSIFICATION: production_fallback.  A refused or raising adoption costs
            # prefill work, never correctness: s.fed stays where it was, so the prompt is fed
            # in full.  Counted so a cache that silently never works is visible on /health
            # instead of looking like a cache that is simply cold.
            self.prefix_errors += 1
            s.fed, s.adopted = 0, 0

    def _prefix_publish(self) -> None:
        """Offer every active slot's completed whole blocks to the cache.

        Called at a step boundary.  Only a ``fed`` that is an exact multiple of the cache's
        block size may publish: the recurrent snapshot is valid only when the slot has
        processed exactly ``n_blocks * block_size`` tokens, which is why ``chunk_align`` exists
        and why ``bidec_serve`` sets it to ``PAGE``.
        """
        c = self.prefix_cache
        if c is None or not self._mtp_prefix_ready:
            return
        block = int(getattr(c, "block_size", PAGE))
        with self.lock:
            # image requests never publish: their prompt tokens do not identify their KV
            live = [s for s in self.active if not s.done and s.slot >= 0 and s.mm is None]
        for s in live:
            if s.fed >= block and s.fed % block == 0 and s.fed > s.adopted:
                try:
                    c.publish(s.slot, s.prompt, s.fed)
                except Exception:  # noqa: BLE001
                    # CLASSIFICATION: production_fallback.  Failing to CACHE something is a
                    # missed optimisation; it must not end the request.
                    self.prefix_errors += 1

    def _admit(self) -> None:
        limit = self._prefill_seq_limit()
        with self.lock:
            self._policy_reorder()
            prefilling = sum(1 for s in self.active if s.fed < len(s.prompt) and not s.done)
            while (self.waiting and len(self.active) < self.max_active and self.bd.free_slots
                   and prefilling < limit
                   and self.reserved + self.waiting[0].reserved_pages <= self.bd.pages_total):
                if self.waiting[0].mm is not None and not self.bd.mm_can_attach(
                        len(self.waiting[0].prompt)):
                    break                      # rotary scratch full: wait for an image prompt to finish
                prefilling += 1
                s = self.waiting.pop(0)
                s.slot = self.bd.alloc_slot()
                self.bd.reset_slot_state(s.slot)
                self.reserved += s.reserved_pages
                s.t_admit = time.time()
                if s.temperature > 0:
                    s.gen = torch.Generator(device="cpu")
                    s.gen.manual_seed(int(s.seed) if s.seed is not None else int(time.time_ns() & 0x7fffffff))
                if s.mm is not None:
                    if not self._attach_mm(s):     # finished with "error"; never enters active
                        prefilling -= 1
                        continue
                else:
                    self._adopt_prefix(s)
                self.active.append(s)

    def _attach_mm(self, s: Seq) -> bool:
        """Encode an image request (vision forward for THIS request only) and bind its plan to
        its slot.  A failure ends that request alone ("error"), never the engine."""
        try:
            if self.mm_encoder is None:
                raise RuntimeError("this server has no vision tower")
            s.mm_plan = self.mm_encoder(s.mm)
            self.bd.attach_mm(s.slot, s.mm_plan)
            return True
        except Exception as e:  # noqa: BLE001
            # CLASSIFICATION: production_fallback.  One request's bad image (or a refused
            # position cross-check) must not stop the engine: that request ends with "error"
            # and its reason, its slot and pages go back to the pool.
            s.error = f"{type(e).__name__}: {e}"
            self.mm_errors += 1
            self._finish(s, "error")
            return False

    def _finish(self, s: Seq, why: str) -> None:
        s.done, s.finish = True, why
        if self._specctl is not None:
            # Here and not at the end of _step_spec: a CANCELLED request is reaped by
            # _reap_cancelled, which runs in step() before _step_spec is ever called, so it
            # has already left self.active by then.  Releasing from the one place every
            # terminal path goes through -- stop, length, cancel, engine error -- is the only
            # way the controller's per-request SlotSpec cannot accumulate for the life of the
            # process.  Found by test_stop_and_cancel_during_speculation.
            self._specctl.release(s.rid)
        if self.prefix_cache is not None and s.slot >= 0:
            # Unpin before the slot is handed back: the cache holds a refcount on every block
            # this sequence adopted, and a slot reused by the next request would otherwise pin
            # entries for a sequence that no longer exists.
            try:
                self.prefix_cache.release(s.slot)
            except Exception:  # noqa: BLE001
                # CLASSIFICATION: production_fallback.  A failed unpin leaks cache budget (the
                # entry stays pinned and unevictable); it must not leak the device SLOT, which
                # is what returning here instead of freeing would do.
                self.prefix_errors += 1
        self.bd.free_slot(s.slot)
        self.reserved -= s.reserved_pages
        if s.on_token is not None:
            s.on_token(s, None)                            # end-of-stream sentinel

    def idle(self) -> bool:
        with self.lock:
            return not self.active and not self.waiting

    # ------------------------------------------------------------------ lifecycle
    def cancel(self, rid: Any) -> str:
        """Abort a request by id.  Waiting -> dropped now; active -> reaped at the next step
        boundary (never mid-step, so no other request's rows are disturbed).  Returns
        "waiting" / "active" / "unknown"."""
        with self.lock:
            for i, s in enumerate(self.waiting):
                if s.rid == rid:
                    self.waiting.pop(i)
                    s.done, s.finish, s.cancelled = True, "cancelled", True
                    if s.on_token is not None:
                        s.on_token(s, None)
                    return "waiting"
            for s in self.active:
                if s.rid == rid:
                    s.cancelled = True
                    return "active"
        return "unknown"

    def _reap_cancelled(self) -> int:
        """Free the slots and pages of cancelled active requests.  Called at a step boundary."""
        n = 0
        for s in list(self.active):
            if s.cancelled and not s.done:
                self._finish(s, "cancelled")
                n += 1
        if n:
            with self.lock:
                self.active = [s for s in self.active if not s.done]
        return n

    def stats(self) -> Dict[str, Any]:
        """Capacity snapshot: what a /health endpoint should report."""
        with self.lock:
            return {"active": len(self.active), "waiting": len(self.waiting),
                    "max_users": self.max_active, "slots_free": len(self.bd.free_slots),
                    # rc6: every server knob glc-bench documents must be READABLE BACK off
                    # /health, so a flag that is accepted and then dropped is caught in the
                    # run (run-4 finding H4) instead of after it.
                    "selective_head": int(getattr(self.bd, "selective_head", False)),
                    "head_rows_total": int(getattr(self.bd, "head_rows_total", 0)),
                    "logits_buffer_bytes": self.bd.logits.numel() * self.bd.logits.element_size(),
                    "max_slots": self.bd.max_slots,
                    "max_ctx": self.bd.max_pages * PAGE,
                    # state-budget knobs (STATE_BUDGET_20261004), read off the live decoder:
                    # the recurrent ring depth R (= spec_k + 1, derived, not a knob), the conv
                    # history ring, and the row capacity every M-sized buffer is sized by.
                    "gdn_ring_depth": int(self.bd.R),
                    "gdn_conv_ring": int(getattr(self.bd, "conv_ring", 16)),
                    "max_rows": int(self.bd.max_rows),
                    "pages_reserved": self.reserved, "pages_total": self.bd.pages_total,
                    "steps": self.steps, "rows_total": self.rows_total,
                    "work_list_capacity": self.bd.g_cap, "prompt_chunk_trims": self.trims,
                    "max_decode_rows": self.max_decode_rows,
                    "prefill_priority": self.prefill_priority,
                    "prefill_chunk": self.prefill_chunk,
                    "row_tile": self.row_tile,
                    "gemm_tile": self.gemm_tile,
                    # rc8: at --gemm-tile 128 the 128 tile is launched per SHAPE, only where
                    # the measured tune table says it wins (glc_serve.bigemm_tune).
                    "gemm_tile_dispatch": ("per-shape (bigemm_tune T2 table)"
                                           if self.gemm_tile >= 128 else "64 everywhere"),
                    "row_budget_tiles": self.row_budget_tiles,
                    "row_budget_rows": self.row_budget_rows,
                    "row_budget": self.row_budget(),
                    "prefilling": sum(1 for s in self.active if s.fed < len(s.prompt)),
                    "spec_k": self.spec_k, "max_rows_step": self.max_rows_step,
                    # rc6 rule again: the prefix-cache and admission knobs must be READABLE
                    # BACK, and read off the live object rather than echoed from argv -- a
                    # budget reported from the cache's own CacheStats cannot disagree with the
                    # cache that is actually running.
                    "prefix_cache_mb": self._prefix_cache_mb(),
                    "prefix_cache": self._prefix_cache_stats(),
                    "prefix_adopted_tokens": self.prefix_adopted,
                    "prefix_adopted_requests": self.prefix_requests,
                    "prefix_cache_errors": self.prefix_errors,
                    "admission_policy": (type(self.admission_policy).__name__
                                         if self.admission_policy is not None else "fifo"),
                    "spec_ngram": int(self.spec_ngram),
                    "spec_ngram_forced": int(self.spec_ngram_forced),
                    "mm": self._mm_stats(),
                    "spec": self.spec_stats()}

    def _mm_stats(self) -> Optional[Dict[str, Any]]:
        mm = getattr(self.bd, "mm", None)
        if mm is None:
            return None
        return {**mm.stats(), "mm_errors": self.mm_errors,
                "mm_active": sum(1 for s in self.active if s.mm is not None)}

    def _prefix_cache_stats(self) -> Optional[Dict[str, Any]]:
        """The cache's own metrics, or an error marker.  Never raises: /health is the endpoint
        an operator reaches for when something is already wrong, so a sick cache must not be
        able to turn a 200 with bad news into a 500 with none."""
        if self.prefix_cache is None:
            return None
        try:
            return dict(self.prefix_cache.stats())
        except Exception as e:  # noqa: BLE001
            # CLASSIFICATION: production_fallback.
            return {"error": f"{type(e).__name__}: {e}"}

    def _prefix_cache_mb(self) -> int:
        """The prefix cache's byte budget in MiB, 0 when no cache is attached."""
        if self.prefix_cache is None:
            return 0
        try:
            return int(self.prefix_cache.stats().get("bytes_budget", 0)) // (1 << 20)
        except Exception:  # noqa: BLE001
            # CLASSIFICATION: production_fallback.  /health must answer; -1 says "a cache is
            # attached but would not report its budget", which is distinguishable from 0.
            return -1

    def admissible(self, prompt_len: int, max_new: int) -> bool:
        """True if a request of this size could ever be admitted (page pool / context cap)."""
        need = (prompt_len + max_new + PAGE) // PAGE + 1
        return (need <= self.bd.pages_total
                and prompt_len + max_new + 1 <= self.bd.max_pages * PAGE)

    def _chunk_len(self, s: Seq, budget: int) -> int:
        """Rows to feed from this sequence's prompt now, honouring chunk_align."""
        left = len(s.prompt) - s.fed
        chunk = self.prefill_chunk
        if self.prefill_priority and not self._scheduled_decoders():
            # No decode row wants this step: a chunk capped at --prefill-chunk leaves the rest
            # of the row budget unused, and the step costs the same weight traffic either way
            # (run-3 /health: 35.9 GB streamed per step).  _fit_work_list still shrinks the
            # chunk to the entry budget, so this can only ever take rows nothing else wanted.
            chunk = max(chunk, budget)
        c = min(chunk, budget, left)
        if self.chunk_align > 1 and c < left:
            end = s.fed + c
            snapped = end - (end % self.chunk_align)
            if snapped > s.fed:                       # never snap a chunk away to nothing
                c = snapped - s.fed
        return c

    # ------------------------------------------------------------------ work-list budget
    def _fit_work_list(self, seqs: List[Tuple[int, int, List[int]]],
                       owners: List[Tuple[Seq, Any]]) -> None:
        """Shrink prompt chunks in place until the step's work list fits ``bd.g_cap``.

        The row budget is NOT the binding constraint on a long context: the attention work
        list has one entry per (row, KV page) pair, so a 256-row chunk at position 8000
        wants ~8,000 entries against a capacity of ``pages_total + max_rows + 8``.  rc2 had
        no item budget at all and the overflow landed as a numpy broadcast error that
        stopped the engine on the ctx8k class (finding F5).

        Decode rows are never trimmed and never need to be: each contributes its slot's
        page count, pages are reserved per slot at admission, and the live pages across all
        slots cannot exceed ``pages_total`` -- which is exactly what ``g_cap`` budgets for.
        Only prompt chunks can overflow, so only prompt chunks are shrunk, newest first.  A
        chunk shrunk to nothing is dropped and retried next step; the sequence makes
        progress because any single prompt row fits by the same page-pool invariant.
        """
        bd = self.bd
        self.trims = getattr(self, "trims", 0)
        while seqs and bd.items_for(seqs) > bd.g_cap:
            for i in range(len(seqs) - 1, -1, -1):
                if owners[i][1] == "dec":
                    continue
                slot, pos0, toks = seqs[i]
                if len(toks) > 1:
                    n = max(1, len(toks) // 2)
                    # Keep the trimmed chunk on a block boundary where one is reachable, so a
                    # trim does not silently cost the prefix cache: the GDN recurrent state is
                    # only snapshottable on a PAGE-token boundary.  Where NO aligned chunk fits
                    # the work list -- pos0 already past the boundary and the budget below one
                    # whole block -- the budget wins, exactly as _chunk_len refuses to snap a
                    # chunk away to nothing.  This made itself visible only once RefDecoder
                    # carried g_cap (rc3 finding G2): on the engine, which has carried it since
                    # rc3, the behaviour was live and untested.
                    align = getattr(self, "chunk_align", 1)
                    if align > 1:
                        end = pos0 + n
                        snapped = end - (end % align)
                        if snapped > pos0:
                            n = snapped - pos0
                    cut = toks[:n]
                    seqs[i] = (slot, pos0, cut)
                    owners[i] = (owners[i][0], len(cut))
                else:
                    seqs.pop(i)
                    owners.pop(i)
                self.trims += 1
                break
            else:
                # Decode rows alone over capacity: the page-pool invariant above is broken,
                # which is a configuration error (pages_total too small for max_ctx x slots),
                # not something to trim away silently.
                raise WorkListOverflow(
                    f"{len(seqs)} decode rows need {bd.items_for(seqs)} work-list entries "
                    f"but capacity is {bd.g_cap}; raise --pages or lower --max-ctx/--slots")

    # ------------------------------------------------------------------ row budget
    def _scheduled_decoders(self) -> int:
        """Decoding sequences that will actually get a row in the next step: the decode-row cap
        is what bounds them, not how many are admitted."""
        with self.lock:
            dec = sum(1 for s in self.active if s.fed == len(s.prompt) and not s.done)
        return min(dec, self.max_decode_rows)

    def remaining_rows(self) -> int:
        """Rows left in this step's budget after one row per decoding sequence.

        The speculation crossover is a ROW budget, not a batch size: the BI-GEMM knee is at 64
        rows, so a speculation adapter must keep sum(1 + k_i) over decoding sequences within
        ``max_rows_step`` and degrade k toward 0 under load rather than overfilling the tile.
        """
        return max(0, min(self.max_decode_rows, self.row_budget()) - self._scheduled_decoders())

    def spec_k_budget(self, k_wanted: Optional[int] = None) -> int:
        """The largest uniform spec depth k with sum(1 + k) over decoding sequences <= the row
        budget.  Returns 0 when even k=1 would not fit, which is the correct degradation."""
        k = self.spec_k if k_wanted is None else int(k_wanted)
        if k <= 0:
            return 0
        dec = self._scheduled_decoders()
        if dec <= 0:
            return k
        return max(0, min(k, self.max_decode_rows // dec - 1))

    def _emit(self, s: Seq, t: int, row: int) -> bool:
        """Record one emitted token; True if the sequence finished."""
        if s.digest:
            s.digests.append(logits_row_digest(self.bd.logits[row]))
        if not s.out:
            s.t_first = time.time()
        s.out.append(t)
        if s.on_token is not None:
            s.on_token(s, t)
        if t in s.stop_ids:
            self._finish(s, "stop")
            return True
        if len(s.out) >= s.max_new:
            self._finish(s, "length")
            return True
        return False

    # ------------------------------------------------------------------ speculation
    def _spec(self):
        """The speculation driver for this scheduler, built on first use.

        ``bidec_spec.BatchSpecController.step`` is the ONE implementation of a speculative
        cycle -- plan, (k-1) batched MTP draft passes, one batched trunk forward with a
        variable number of verify rows per slot, per-slot accept, one batched MTP confirm --
        and it is the implementation ``bidec_spec.SpecDriver`` runs, which is what
        ``scripts/batchserve/bi_gate_spec_batch.py`` and ``tests/batchserve_spec`` gate.
        Routing ``_step_spec`` through it is what makes "gated" and "served" the same code.

        What is NOT reused is ``SpecDriver`` itself.  SpecDriver owns slots, pages, prompt
        prefill and the request lifecycle (``SpecDriver.admit`` allocates a slot and prefills
        the whole prompt in one synchronous loop); ``Batcher`` owns exactly those things, for
        continuous batching, chunked prefill, page reservation at admission, cancellation and
        the work-list budget.  Two owners of a slot is not a wiring question, it is a second
        scheduler.  So the cycle is shared and the bookkeeping is not, which is the split
        ``bidec_spec`` documents under "This is the wiring".

        The row budget is the decode-row cap, not ``max_rows_step``.  ``SpecPolicy.plan``
        clamps to ``CostModel.crossover_rows()`` -- ``flat_rows``, the measured BI-GEMM knee --
        so sum over decoding slots of (1 + k_i) <= max_decode_rows holds by construction, and
        k degrades to 0 as the batch grows rather than overfilling the tile.  rc6's hand-rolled
        ``_step_spec`` budgeted against ``max_rows_step`` (256) instead, so at 8 users with
        spec_k 4 it asked for 40 verify rows and at 48 users it would have asked for 240 --
        four tiles past the knee, which is the regime the Synergy result says loses.
        """
        if self._specctl is None:
            from glc_serve import bidec_spec as _bs

            # `spec` declares that speculation is PERMITTED against this decoder;
            # `mtp_modelled` declares that the draft pass exists.  bidec_ref.RefDecoder sets
            # spec=True and mtp_modelled=False -- its run_mtp raises on purpose, so that a stub
            # returning zeros cannot make a speculation gate pass vacuously -- and reading only
            # `spec` stopped the reference server at its first speculative step.
            has_mtp = bool(getattr(self.bd, "spec", False)) and bool(
                getattr(self.bd, "mtp_modelled", True))
            mtp = _bs.DecoderMtp(self.bd) if has_mtp else None
            # No MTP head (the CPU reference) means n-gram drafting or no drafting at all, so
            # it is enabled there by construction.  Exactness does not depend on the drafter --
            # drafts never reach the output -- only throughput does.
            pol = _bs.SpecPolicy(cost=_bs.CostModel(flat_rows=self.max_decode_rows),
                                 max_k=self.spec_k,
                                 allow_ngram=bool(self.spec_ngram) or mtp is None)
            self._specad = _batcher_spec_adapter_cls()(self, mtp)
            self._specctl = _bs.BatchSpecController(self._specad, pol)
        return self._specctl

    def spec_stats(self) -> Optional[Dict[str, Any]]:
        """Acceptance and row accounting for the speculation lane, or None when it is off."""
        if self._specctl is None:
            return None
        try:
            d = dict(self._specctl.stats())
            d["row_budget"] = max(1, min(self.max_decode_rows, self.row_budget()))
            return d
        except Exception as e:  # noqa: BLE001
            # CLASSIFICATION: production_fallback.  /health must answer (see
            # _prefix_cache_stats for the same reasoning).
            return {"error": f"{type(e).__name__}: {e}"}

    @torch.no_grad()
    def _step_spec(self) -> int:
        """One scheduler step with speculation on.

        Three populations share the one batched forward, in the same order plain ``step``
        uses -- decode rows first, then prompt chunks:

          (a) decoding sequences      -> a verify block of (1 + k_i) rows, from the controller
          (b) prefilling sequences    -> a prompt chunk, through the adapter's ``extra`` rows
          (c) a chunk that COMPLETES a prompt -> its last row emits that request's first token,
                                                 and the request joins the controller

        Order is what batch-invariance needs here: every emitted token is a trunk row, trunk
        rows do not interact (``DecoderLike.run``), and the rows of a request accumulate in the
        same sequence as they would with speculation off -- so the emitted stream is bitwise
        the non-speculative stream for any k schedule and any batch composition.  That is the
        statement ``bi_gate_batchexact.py`` and ``tests/batchserve_spec`` assert, and the
        statement ``tests/batchserve_spec/test_engine_spec_wiring.py`` asserts against the CPU
        ``--reference-engine`` path with speculation ON versus OFF.
        """
        ctl, ad = self._spec(), self._specad
        with self.lock:
            active = list(self.active)
        dec = [s for s in active if s.fed == len(s.prompt) and not s.done and s.out]
        # (a) decoding sequences join the controller the step after their first token.  The
        # n-gram index is seeded from the FULL prompt plus everything already emitted, never
        # from a fed suffix: that is the vLLM #58894 hazard the lane documents, where a
        # prefix-cache hit silently took DFlash2 acceptance to exactly 0.0%.
        for s in dec:
            if s.rid not in ctl.slots:
                ctl.admit(s.rid, list(s.prompt) + list(s.out[:-1]))
                ctl.observe_emitted(s.rid, s.out[-1:])
        # The decode-row cap bounds how many sequences get a verify block at all; the rest
        # wait a step, exactly as they do with speculation off.  ``--row-budget-tiles`` can cap
        # the step BELOW max_decode_rows, and when it does it is the binding cap here too:
        # ``spec_cap`` is the same min() ``remaining_rows`` takes, so the verify blocks and the
        # plain decode rows are budgeted against one number.
        spec_cap = max(1, min(self.max_decode_rows, self.row_budget()))
        dec = dec[:spec_cap]
        ad.slots = {s.rid: s.slot for s in dec}
        ad.owner = {s.rid: s for s in dec}
        ad.prefill_src = [s for s in active if s.fed < len(s.prompt) and not s.done]
        ad.verify_rows = sum(1 for _ in dec)

        out, rep = None, None
        if dec:
            live = [(s.rid, s.fed + len(s.out) - 1, s.out[-1]) for s in dec]
            rep = ctl.step(live, spec_cap)
            out = rep.emitted
        else:
            # No decoding sequence: the forward is prompt chunks only.  The adapter still runs
            # it, so prefill makes progress on a step where nothing decodes.
            ad.trunk_verify([])

        Mb = ad.last_Mb
        if Mb == 0:
            return 0

        # (a) commit the accepted tokens, in the controller's own order
        if out:
            for s in dec:
                toks = out.get(s.rid) or []
                # credited once per cycle, before the emit loop: a cycle whose accepted tokens
                # are all discarded by a stop rule still PROPOSED them, and an acceptance rate
                # computed from only the cycles that emitted is not an acceptance rate.
                s.prop += int(rep.proposed.get(s.rid, 0))
                s.acc += int(rep.accepted.get(s.rid, 0))
                r0, _n = ad.last_rows[s.rid]
                for i, tk in enumerate(toks):
                    if self._emit(s, int(tk), r0 + i):
                        # stop / length: the remaining accepted tokens were never streamed and
                        # the slot is about to be freed, so discarding them is exact.
                        ctl.observe_emitted(s.rid, toks[:i + 1])
                        ctl.release(s.rid)
                        break
                else:
                    ctl.observe_emitted(s.rid, toks)

        # (b) / (c) prompt chunks: advance fed, and emit the first token of any prompt that
        # completed on its last row
        for (s, c), (r0, n) in ad.prefill:
            p_end = s.fed + c
            s.fed = p_end
            if p_end == len(s.prompt):
                first = int(ad.am[r0 + n - 1])
                finished = self._emit(s, first, r0 + n - 1)
                if not finished and ad.mtp is not None:
                    ctl.admit(s.rid, s.prompt)
                    ctl.observe_emitted(s.rid, [first])
                    draft = ad.mtp.rows([(s.slot, p_end - 1, first,
                                          self.bd.hsrc_trunk0 + r0 + n - 1, 1)])[0]
                    ctl.seed_mtp(s.rid, draft)

        self._prefix_publish()
        with self.lock:
            self.active = [s for s in self.active if not s.done]
        self.steps += 1
        self.rows_total += ad.last_rows_run
        return Mb

    @torch.no_grad()
    def _run_selected(self, seqs, owners):
        """Verify/decode rows and completing prompt tails are observable outputs."""
        if not getattr(self.bd, "selective_head", False):
            return self.bd.run(seqs)
        rows, r0 = [], 0
        for (_, _, toks), (s, kind) in zip(seqs, owners):
            n = len(toks)
            if kind == "dec":
                rows.extend(range(r0, r0 + n))
            elif s.fed + kind == len(s.prompt):
                rows.append(r0 + n - 1)
            r0 += n
        return self.bd.run(seqs, output_rows=rows)

    def step(self) -> int:
        if self._reap_cancelled():
            self._admit()                                  # cancelled slots are reusable now
        self._admit()
        if not self.active:
            return 0
        if self.spec_k:
            return self._step_spec()
        seqs, owners = [], []
        budget = self.row_budget()
        dec_budget = self.max_decode_rows
        for s in self.active:                              # decoding sequences first
            if s.fed == len(s.prompt) and budget > 0 and dec_budget > 0:
                dec_budget -= 1
                seqs.append((s.slot, s.fed + len(s.out) - 1, [s.out[-1]]))
                owners.append((s, "dec"))
                budget -= 1
        for s in self.active:                              # then prompt chunks
            if s.fed < len(s.prompt) and budget > 0:
                c = self._chunk_len(s, budget)
                seqs.append((s.slot, s.fed, s.prompt[s.fed:s.fed + c]))
                owners.append((s, c))
                budget -= c
        if not seqs:
            return 0
        self._fit_work_list(seqs, owners)
        if not seqs:
            return 0
        Mb, ranges = self._run_selected(seqs, owners)
        need = [(s, r0 + n - 1) for (s, kind), (r0, n) in zip(owners, ranges)
                if kind == "dec" or s.fed + kind == len(s.prompt)]
        am = self.bd.am[:Mb].tolist()                      # host sync
        for (s, kind), (r0, n) in zip(owners, ranges):
            if kind != "dec":
                s.fed += kind
        for s, row in need:
            if s.temperature > 0:
                lg = self.bd.logits[row].float()
                p = torch.softmax(lg / float(s.temperature), dim=-1).cpu()
                if s.top_p < 1.0:
                    sp, si = torch.sort(p, descending=True)
                    keep = (torch.cumsum(sp, 0) - sp) <= s.top_p
                    q = torch.zeros_like(p).scatter_(0, si[keep], sp[keep])
                    p = q / q.sum()
                t = int(torch.multinomial(p, 1, generator=s.gen).item())
            else:
                t = int(am[row])
            if s.digest:
                s.digests.append(logits_row_digest(self.bd.logits[row]))
            if not s.out:
                s.t_first = time.time()
            s.out.append(t)
            if s.on_token is not None:
                s.on_token(s, t)
            if t in s.stop_ids:
                self._finish(s, "stop")
            elif len(s.out) >= s.max_new:
                self._finish(s, "length")
        self._prefix_publish()
        with self.lock:
            self.active = [s for s in self.active if not s.done]
        self.steps += 1
        self.rows_total += sum(n for _, n in ranges)
        return Mb


#: ``--vision-placement`` values accepted by ``load_model`` (docs/serving/BIDEC_MULTIMODAL_20261004.md;
#: the 2026-10-04 ENGINE_PLACEMENT values, now backed by a real image path):
#:   device    the native vision tower (``bidec_mm.VisionTower``: HF's vision module class on
#:             exact bf16 weights) resident on the accelerator.  Bit-exact by construction.
#:   ondemand  (default) its bf16 weights in pinned host RAM, uploaded for each image encode and
#:             released after.  Bit-exact by construction (the same module, weights and kernels
#:             as ``device``; only residency differs).  0 bytes of VRAM between image requests.
#:   host      weights in host RAM and the tower EXECUTED ON THE CPU.  NOT bit-exact against a
#:             GPU-run parent (CPU kernels are not GPU kernels); reported as such everywhere.
#:   off       no vision tower at all; image requests are refused (400).
#: In EVERY placement the engine's own skeleton is text-only: the loader's vision modules would
#: be TBE-serving linears (their own fused kernel, not the parent's F.linear arithmetic), so the
#: tower the engine runs is never the loader's.  (Before 2026-10-04 this flag only chose whether
#: the loader's tower was materialised -- and the bundle argv was built with the boolean
#: inverted, so ``host`` loaded it and ``device`` did not; see ``_bundle_argv``.)
VISION_PLACEMENTS = bm.VISION_PLACEMENTS


def _vision_text_only(vision_placement: str) -> bool:
    """Whether the bundle-path ENGINE skeleton is built text-only, given ``vision_placement``.

    Always True for a valid placement: the native image path runs its own
    ``bidec_mm.VisionTower`` (exact bf16 weights, HF's module class), never the loader's
    tower, so materialising the loader's tower would only be a second copy of ~0.92 GB that no
    code path reads.  Kept as a function so the mapping stays directly testable.
    """
    if vision_placement not in VISION_PLACEMENTS:
        raise ValueError(f"vision_placement must be one of {VISION_PLACEMENTS}, got {vision_placement!r}")
    return True


def _bundle_argv(bundle: str, device: str, *, text_only: bool, load_mtp: bool) -> List[str]:
    """``server._parse`` argv for the engine's bundle load.

    2026-10-04 regression: this was built inline as ``if not text_only: argv.append("--text-only")``
    -- the boolean inverted, so ``--vision-placement host`` (text_only=True) loaded the loader's
    vision tower and ``device`` skipped it.  ``--text-only`` is a store_true flag meaning
    "skip the vision tower": it is passed exactly when ``text_only`` is True.
    """
    argv = ["--bundle", bundle, "--backend", "tbe", "--exec-mode", "exact", "--device", device,
            "--gate", os.environ.get("BIDEC_GATE", "sample"), "--port", "0"]
    if text_only:
        argv.append("--text-only")
    if not load_mtp:
        argv.append("--no-mtp")
    return argv


# ---------------------------------------------------------------------------- loading
def load_model(*, bundle: Optional[str] = None, parent: Optional[str] = None, gguf: Optional[str] = None,
               tune: str, device: str = "cuda:0", spec: bool = False, weight_format: str = "tbe",
               log=print, embed_placement: str = "device", vision_placement: str = "ondemand",
               mtp_placement: str = "device", mm_options: Optional[Dict[str, Any]] = None,
               context_build_root: Optional[str] = None) -> Dict[str, Any]:
    """The model for the batch engine, plus (unless ``vision_placement == "off"``) the native
    image frontend.
    bundle=<TBE bundle dir>                   -> exact TBE codec (G1-gated at load)
    bundle=<codec v2 artifact>, weight_format="tbe2"
                                              -> exact codec v2 streams through BI-GEMM Ld<F_TBE2>
                                                 (glc_serve.tbe2_load; config + tokenizer from the
                                                 artifact, else from parent=)
    parent=<HF dir>, gguf=<file>              -> the GGUF's linears (Q8_0 / K-quants fused) on the
                                                 parent's non-linear parameters; the parent is
                                                 read on the HOST, only the GGUF bytes go to the GPU
    parent=<HF dir>                           -> dense bf16 parent

    embed_placement / mtp_placement: docs/serving/ENGINE_PLACEMENT_20261004.md.
    vision_placement / mm_options: docs/serving/BIDEC_MULTIMODAL_20261004.md (``mm_options`` is
    passed to ``bidec_mm.MMFrontend``: max_images, max_image_bytes, fetch_urls, check_hf_rope).
    ``mtp_placement == "off"`` means the MTP head is not loaded at all (speculation with an
    MTP drafter is then unavailable -- ``bidec_serve`` refuses ``--spec-k`` > 0 without
    ``--spec-ngram`` before this function is ever called, so that refusal is loud and early).
    """
    from .context_source import context_artifact_factory, validate_context_selection

    validate_context_selection(build_root=context_build_root, bundle=bundle, parent=parent,
                               gguf=gguf, weight_format=weight_format)
    from transformers import AutoConfig, AutoTokenizer

    load_mtp = bool(spec) and mtp_placement != "off"
    t0 = time.time()
    rec: Dict[str, Any] = {}
    # Every path-like argument is made a str HERE, at the one place they are spliced into an
    # argv.  rc7 run 6a: the tile gate's T3 passed `bundle` as a pathlib.Path, argparse's
    # _parse_optional indexes `arg_string[0]`, and the gate died with "TypeError: 'PosixPath'
    # object is not subscriptable" before it compared a single digest.  Callers may hand in a
    # Path (glc-bench's fetch_bundle returns one); the argv must not care.
    bundle = os.fspath(bundle) if bundle else bundle
    parent = os.fspath(parent) if parent else parent
    gguf = os.fspath(gguf) if gguf else gguf
    tune = os.fspath(tune) if tune else tune
    device = str(device)
    tn = fdm.load_tune(tune)
    if weight_format not in ("tbe", "tbe2", "tbe21"):
        raise ValueError(f"weight_format must be tbe, tbe2 or tbe21, got {weight_format!r}")
    text_only = _vision_text_only(vision_placement)
    mm_bundle = None
    if weight_format in ("tbe2", "tbe21"):
        if not bundle:
            raise SystemExit("--weight-format tbe2 needs --bundle <codec v2 artifact dir>")
        from .tbe2_load import load_v2_model

        # The codec v2 loader builds the text skeleton only (its streams carry no vision
        # tower the native image path could read), so the image frontend takes the tower's
        # exact bf16 weights from --parent; without one, a multimodal checkpoint must be served
        # with --vision-placement off rather than fail on the first image.
        v2_kwargs = ({"weight_format": "tbe21", "embed_placement": embed_placement}
                     if weight_format == "tbe21" else {})
        if context_build_root is not None:
            v2_kwargs["artifact_factory"] = context_artifact_factory(context_build_root)
        L2 = load_v2_model(bundle, config_dir=parent, device=device, text_only=True, spec=load_mtp,
                           log=log, **v2_kwargs)
        model, mtp, tok = L2["model"], L2["mtp"], L2["tok"]
        rec[weight_format] = L2["rec"]
        local_dir = L2["rec"]["config"]
        config = AutoConfig.from_pretrained(local_dir)   # the full config (vision_config incl.)
        if (vision_placement != "off" and getattr(config, "vision_config", None) is not None
                and not parent):
            raise SystemExit("--weight-format tbe2 with a multimodal checkpoint needs --parent "
                             "<HF snapshot> for the vision tower's weights, or --vision-placement off")
        src = bundle
    elif bundle:
        from .server import _parse, build_state

        # text_only selects the SKELETON (whether the loader's vision tower exists to receive
        # weights at all); load_mtp is independent of it (loader.py 2026-10-04 fix).
        st = build_state(_parse(_bundle_argv(bundle, device, text_only=text_only, load_mtp=load_mtp)),
                         log=log)
        model = st.engine.model
        mtp = st.engine.loaded.mtp if load_mtp else None
        tok = st.engine.tok
        rec["weight_gate"] = (getattr(st, "receipts", None) or {}).get("weight_gate")
        src = bundle
        config = st.engine.loaded.config
        local_dir = st.engine.loaded.local_dir
        mm_bundle = st.engine.loaded.bundle
    else:
        from .loader import _disable_cuda_only_conv_kernels, model_class_for

        config = AutoConfig.from_pretrained(parent)
        local_dir = parent
        cls = model_class_for(config, text_only=True)
        dev_map = {"": "cpu"} if gguf else {"": device}
        model = cls.from_pretrained(parent, dtype=torch.bfloat16, device_map=dev_map)
        model.eval()
        if gguf:
            from . import miv_kq
            from .q8serve import load_gguf

            miv_kq.register_with_engine()
            rec["gguf"] = load_gguf(model, gguf, tn, device=device, check_every=int(os.environ.get("BIDEC_Q8_CHECK", "16")),
                                    log=log)
            if rec["gguf"].get("check_fail"):
                raise SystemExit(f"GGUF Q8_0 blocks differ from gguf-py: {rec['gguf']['check_fail']}")
            model.to(device)
        _disable_cuda_only_conv_kernels(model, torch.device(device))
        tok = AutoTokenizer.from_pretrained(parent)
        src = gguf or parent
        mtp = None
        if load_mtp:
            from .loader import load_dense_mtp

            mtp = load_dense_mtp(parent, config, torch.device(device))
    fd_mtp_placement = mtp_placement if mtp_placement in fdm.MTP_PLACEMENTS else "device"
    fd = fdm.FastDecoder(model, mtp, tune=tn, max_len=512, R=1, device=device,
                         embed_placement=embed_placement, mtp_placement=fd_mtp_placement)
    # The native image frontend.  The rotary module is the engine's OWN text model's (the one
    # BatchDecoder builds its text table from), so an image prompt's scratch rows and the text
    # table come from one module.
    mm = bm.build_frontend(config=config, local_dir=local_dir, placement=vision_placement,
                           device=device, bundle=mm_bundle, parent=parent if mm_bundle is None else None,
                           rotary=fdm._text(model).rotary_emb, tokenizer=tok, log=log,
                           **(mm_options or {}))
    rec["placement"] = {
        "embed_placement": embed_placement, "vision_placement": vision_placement,
        "mtp_placement": mtp_placement,
        "vision_resident_bytes": mm.tower.device_bytes if mm is not None else 0,
        "vision": mm.stats() if mm is not None else None,
        **fd.placement_bytes(),
    }
    torch.cuda.synchronize()
    rec["load_s"] = round(time.time() - t0, 1)
    rec["descriptors"] = fd.descriptor_kinds()
    rec["source"] = src
    rec["weight_format"] = weight_format
    log(f"[bidec] model loaded in {rec['load_s']} s; {rec['descriptors']}")
    return {"model": model, "tok": tok, "fd": fd, "rec": rec, "mm": mm}


def stop_ids_for(tok, source_dir: Optional[str]) -> List[int]:
    import json

    ids = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    for p in (Path(source_dir) / "generation_config.json",) if source_dir and Path(source_dir).is_dir() else ():
        if p.is_file():
            e = json.loads(p.read_text()).get("eos_token_id")
            ids |= set(int(v) for v in (e if isinstance(e, list) else [e]) if v is not None)
    im = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im, int) and im >= 0:
        ids.add(im)
    return sorted(ids)


__all__ = ["Batcher", "Seq", "BatchDecoder", "BUCKETS", "PAGE", "bucket_for", "ext", "load_model",
           "needs_gdn_state_ring", "stop_ids_for"]
