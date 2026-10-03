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
  partials and their split-order combine are the single-stream schedule;
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
                 int64_t NKV, double eps);
void b_attn_split(at::Tensor q, at::Tensor kc, at::Tensor vc, at::Tensor row_slot, at::Tensor row_pos,
                  at::Tensor item_row, at::Tensor item_split, at::Tensor n_items, at::Tensor pagetab,
                  int64_t maxpages, at::Tensor pacc, at::Tensor pml, int64_t grid_items, int64_t NQ, int64_t NKV,
                  double scaling);
void b_attn_combine(at::Tensor pacc, at::Tensor pml, at::Tensor qkv, at::Tensor row_pos, at::Tensor row_item0,
                    at::Tensor n_rows, at::Tensor out, int64_t maxrows, int64_t NQ);
void b_kv_import(at::Tensor k, at::Tensor v, at::Tensor pages, int64_t n, at::Tensor kc, at::Tensor vc);
void b_scatter_rows(at::Tensor src, at::Tensor flag, at::Tensor idx, at::Tensor n_rows, at::Tensor dst, int64_t maxrows);
"""
PAGE = 256
BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)


def ext():
    global _EXT
    if _EXT is None:
        with _LOCK:
            if _EXT is None:
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


def bucket_for(m: int, buckets: Sequence[int] = BUCKETS) -> int:
    for b in buckets:
        if b >= m:
            return b
    raise ValueError(f"{m} rows exceed the largest bucket {buckets[-1]}")


class _Meta:
    """All per-step metadata in ONE int32 device tensor (one H2D copy per step)."""

    def __init__(self, dev, max_rows: int, max_slots: int, max_pages: int, g_cap: int):
        spec = [("vtok", max_rows), ("row_slot", max_rows), ("row_pos", max_rows), ("row_item0", max_rows),
                ("seq_slot", max_rows), ("seq_row0", max_rows), ("seq_len", max_rows),
                ("n_rows", 1), ("n_seq", 1), ("n_items", 1),
                ("item_row", g_cap), ("item_split", g_cap), ("ropeoff", max_slots + 1),
                ("hidx", max_rows), ("hflag", max_rows), ("mropeoff", max_slots + 1),
                ("pagetab", (max_slots + 1) * max_pages)]
        n = sum(s for _, s in spec)
        self.dev = torch.zeros(n, dtype=torch.int32, device=dev)
        self.host = torch.zeros(n, dtype=torch.int32, pin_memory=True)
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
        if np.array_equal(pt, self._pt_cache):
            self.dev[:self._pt_off].copy_(self.host[:self._pt_off], non_blocking=True)
            return
        self._pt_cache[:] = pt
        self._pt_uploads += 1
        self.dev.copy_(self.host, non_blocking=True)


class BatchDecoder:
    def __init__(self, fd: "fdm.FastDecoder", *, max_slots: int = 64, max_rows: int = 256,
                 pages_total: int = 1024, max_ctx: int = 32768, R: int = 1, grid_items: int = 512,
                 buckets: Sequence[int] = BUCKETS, gdn_ring_dtype: str = "fp32", log=print):
        self.fd, self.dev, self.log = fd, fd.dev, log
        E, B = fdm.ext(), ext()
        bg.extension()
        self.max_slots, self.R = int(max_slots), int(R)
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
        self.g_cap = self.pages_total + self.max_rows + 8
        self.grid_items = int(grid_items)
        self.dummy = self.max_slots                   # padding rows' slot (page 0 is its page)
        self.H, self.eps, self.V = fd.H, fd.eps, fd.V
        fd.kc = fd.vc = fd.hist = fd.state = None          # the single-stream buffers are not used
        torch.cuda.empty_cache()
        self.cosT, self.sinT = self._rope_tables(int(max_ctx))
        self.NQ, self.NKV = fd.NQ, fd.NKV
        ws = self.ws = bg.Workspace(self.dev)
        wrap = lambda d: bg.wrap(d, ws)  # noqa: E731
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
        att0 = next(o for o in fd.layers if o.kind == "attn")
        gd0 = next(o for o in fd.layers if o.kind == "gdn")
        M, H = self.max_rows, self.H
        S1 = self.max_slots + 1
        t0 = time.time()
        self.kc = z(fd.n_att, self.pages_total + 1, PAGE, self.NKV, 256)
        self.vc = z(fd.n_att, self.pages_total + 1, PAGE, self.NKV, 256)
        self.hist = z(fd.n_gdn, S1, 16, gd0.C)
        self.state = z(fd.n_gdn, S1, self.R, 48, 128, 128,
                       dt=torch.float32 if self.gdn_ring_dtype == "fp32" else torch.float16)
        self.h, self.x, self.proj, self.dn = (z(M, H) for _ in range(4))
        # MTP hidden sources in ONE buffer: rows [0, S1) = per-slot MTP hidden store, rows [S1, S1 + M) =
        # the trunk's final-norm output of the last step (xf) -- one index_select serves draft + confirm rows
        self.hsrc = z(S1 + M, H)
        self.mhid, self.xf = self.hsrc[:S1], self.hsrc[S1:]
        self.zba = z(M, gd0.qkvzba.N)
        self.conv = z(M, gd0.C)
        self.go = z(M, gd0.vdim)
        self.aqkv = z(M, att0.qkv.N)
        self.q = z(M, self.NQ, 256)
        self.ao = z(M, self.NQ * 256)
        F = fd.layers[0].F
        self.gu = z(M, 2 * F)
        self.act = z(M, F)
        self.logits = z(M, self.V)
        self.am = z(M, dt=torch.int32)
        self.pacc = z(self.g_cap, self.NQ, 256, dt=torch.float32)
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
            self.mkc = z(self.pages_total + 1, PAGE, self.NKV, 256)
            self.mvc = z(self.pages_total + 1, PAGE, self.NKV, 256)
            self.me, self.mhin, self.mh, self.mx, self.mg = (z(M, H) for _ in range(5))
            self.mcat = z(M, 2 * H)
            self.mlogits = z(M, self.V)
            self.mam = z(M, dt=torch.int32)
        self.graphs_m: Dict[int, torch.cuda.CUDAGraph] = {}
        self.meta = _Meta(dev, M, self.max_slots, self.max_pages, self.g_cap)
        self.meta.h("mropeoff")[:] = 1              # MTP cache row p <-> rope position p + delta + 1 (text: delta 0)
        self.pt = self.meta.h("pagetab").reshape(S1, self.max_pages)
        self.pt[:] = 0
        self.free_pages = list(range(self.pages_total, 0, -1))   # page 0: the dummy slot's
        self.free_slots = list(range(self.max_slots - 1, -1, -1))
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        torch.cuda.synchronize(dev)
        self.state_bytes = {"kv": self.kc.numel() * 4,
                            "gdn_state": self.state.numel() * self.gdn_ring_bytes,
                            "gdn_ring_dtype": self.gdn_ring_dtype,
                            "conv_hist": self.hist.numel() * 2, "pacc": self.pacc.numel() * 4 + self.pml.numel() * 4}
        log(f"[bidec] buffers in {time.time() - t0:.1f}s: " +
            ", ".join(f"{k} {v / 2**30:.2f} GiB" for k, v in self.state_bytes.items()))

    def _rope_tables(self, n: int):
        """HF's own rotary module at text positions t = h = w (fastdec's construction, longer)."""
        rot = fdm._text(self.fd.model).rotary_emb
        cs, sn = [], []
        for p0 in range(0, n + 64, 1024):
            k = min(1024, n + 64 - p0)
            pid = torch.arange(p0, p0 + k, device=self.dev).view(1, 1, -1).expand(3, 1, -1)
            c, s = rot(torch.zeros(1, device=self.dev, dtype=torch.bfloat16), pid)
            cs.append(c[0])
            sn.append(s[0])
        cosT, sinT = torch.cat(cs).contiguous(), torch.cat(sn).contiguous()
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

    # ------------------------------------------------------------------ the step (graph body)
    def _step(self, Mb: int) -> None:
        E, B, fd, eps, mt = fdm.ext(), ext(), self.fd, self.eps, self.meta
        fd._embed(mt.vtok, self.h, Mb)
        delta = None
        for b in self.L:
            o = b.src
            E.rmsnorm(self.h, delta, o.in_ln, self.x, Mb, eps, 1)
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
                              o.NQ, o.NKV, eps)
                B.b_attn_split(self.q, self.kc[o.ai], self.vc[o.ai], mt.row_slot, mt.row_pos, mt.item_row,
                               mt.item_split, mt.n_items, mt.pagetab, self.max_pages, self.pacc, self.pml,
                               self.grid_items, o.NQ, o.NKV, o.scaling)
                B.b_attn_combine(self.pacc, self.pml, self.aqkv, mt.row_pos, mt.row_item0, mt.n_rows, self.ao,
                                 Mb, o.NQ)
                b.o(self.ao[:Mb], self.proj[:Mb])
            E.rmsnorm(self.h, self.proj, o.post_ln, self.x, Mb, eps, 1)
            b.gu(self.x[:Mb], self.gu[:Mb])
            E.silu_mul(self.gu, self.act, Mb)
            b.down(self.act[:Mb], self.dn[:Mb])
            delta = self.dn
        E.rmsnorm(self.h, delta, fd.final_ln, self.xf, Mb, eps, 1)
        self.lm(self.xf[:Mb], self.logits[:Mb])
        E.argmax_rows(self.logits, self.am, Mb)

    def _mtp_step(self, Mb: int) -> None:
        """The MTP head over Mb rows: row r pairs hidden hsrc[hidx[r]] with token vtok[r] at MTP cache row
        row_pos[r]; flagged rows store their output hidden into the slot's MTP hidden store.  (Drafts only
        steer acceptance; they never change an emitted token or logit.)"""
        E, B, mt, m, eps, H = fdm.ext(), ext(), self.meta, self.m, self.eps, self.H
        self.fd._embed(mt.vtok, self.me, Mb)
        torch.index_select(self.hsrc, 0, mt.hidx[:Mb], out=self.mhin[:Mb])
        E.rmsnorm(self.me, None, m.pre_e, self.mcat[:, :H], Mb, eps, 1)
        E.rmsnorm(self.mhin, None, m.pre_h, self.mcat[:, H:], Mb, eps, 1)
        m.fc(self.mcat[:Mb], self.mh[:Mb])
        E.rmsnorm(self.mh, None, m.in_ln, self.mx, Mb, eps, 1)
        m.qkv(self.mx[:Mb], self.aqkv[:Mb])
        B.b_attn_prep(self.aqkv, m.qnw, m.knw, self.cosT, self.sinT, mt.row_slot, mt.row_pos, mt.n_rows, mt.mropeoff,
                      mt.pagetab, self.max_pages, self.q, self.mkc, self.mvc, Mb, self.NQ, self.NKV, eps)
        B.b_attn_split(self.q, self.mkc, self.mvc, mt.row_slot, mt.row_pos, mt.item_row, mt.item_split, mt.n_items,
                       mt.pagetab, self.max_pages, self.pacc, self.pml, self.grid_items, self.NQ, self.NKV, m.scaling)
        B.b_attn_combine(self.pacc, self.pml, self.aqkv, mt.row_pos, mt.row_item0, mt.n_rows, self.ao, Mb, self.NQ)
        m.o(self.ao[:Mb], self.proj[:Mb])
        E.rmsnorm(self.mh, self.proj, m.post_ln, self.mx, Mb, eps, 1)
        m.gu(self.mx[:Mb], self.gu[:Mb])
        E.silu_mul(self.gu, self.act, Mb)
        m.down(self.act[:Mb], self.dn[:Mb])
        E.rmsnorm(self.mh, self.dn, m.norm, self.mg, Mb, eps, 1)
        self.lm(self.mg[:Mb], self.mlogits[:Mb])
        E.argmax_rows(self.mlogits, self.mam, Mb)
        B.b_scatter_rows(self.mg, mt.hflag, mt.row_slot, mt.n_rows, self.mhid, Mb)

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
                if self.spec:
                    self._mtp_step(Mb)
                    torch.cuda.synchronize(self.dev)
                    gm = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gm, stream=s):
                        self._mtp_step(Mb)
                    torch.cuda.synchronize(self.dev)
                    self.graphs_m[Mb] = gm
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
        self.free_slots.append(slot)

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

    def reset_slot_state(self, slot: int) -> None:
        """A fresh sequence starts from zero conv history and zero recurrent state."""
        self.hist[:, slot].zero_()
        self.state[:, slot].zero_()

    # ------------------------------------------------------------------ one step
    def run(self, seqs: List[Tuple[int, int, List[int]]]) -> Tuple[int, List[Tuple[int, int]]]:
        """seqs: [(slot, pos0, token_ids)] -- each sequence feeds its tokens at positions
        pos0 .. pos0+len-1 (len 1 = decode, >1 = prefill chunk / verify).  Returns
        (Mb, row ranges) after the graph replay; logits / argmax rows are in self.logits / self.am."""
        m = self.meta
        vt, rs, rp, ri0 = m.h("vtok"), m.h("row_slot"), m.h("row_pos"), m.h("row_item0")
        ss, sr0, sl = m.h("seq_slot"), m.h("seq_row0"), m.h("seq_len")
        itr, its = m.h("item_row"), m.h("item_split")
        r = 0
        items = 0
        ranges = []
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
        m.upload()
        g = self.graphs.get(Mb)
        if g is None:
            self._step(Mb)
        else:
            g.replay()
        return Mb, ranges


def _run_mtp(self, rows: List[Tuple[int, int, int, int, int]]) -> int:
    """rows: [(slot, mtp_pos, token, hidx, flag)] -> replay the MTP graph; drafts in self.mam."""
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


# ---------------------------------------------------------------------------- scheduler
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


class Batcher:
    """Continuous batching over a BatchDecoder.  Each step: one row per decoding sequence plus
    prompt chunks of prefilling sequences (row budget ``max_rows_step``, chunk ``prefill_chunk``).
    Pages are reserved at admission for prompt + max_new, so a running request can never run
    out of KV."""

    def __init__(self, bd: BatchDecoder, *, max_rows_step: int = 256, prefill_chunk: int = 256,
                 max_active: Optional[int] = None, spec_k: int = 0, chunk_align: int = 1,
                 max_decode_rows: int = 64):
        self.bd = bd
        # A prompt chunk may end anywhere as far as exactness goes (gated: bi_gate_batchexact.py
        # runs chunk 97).  It may NOT end anywhere as far as a prefix cache goes: the GDN
        # recurrent state can only be snapshotted on a PAGE-token block boundary, so a chunk that
        # ends mid-block can never publish a cacheable prefix.  chunk_align makes every
        # non-final chunk end on a multiple of it; 1 disables the alignment.
        self.chunk_align = max(1, int(chunk_align))
        self.spec_k = int(spec_k)
        if self.spec_k and not bd.spec:
            raise ValueError("speculation needs the MTP head (load with spec=True)")
        if self.spec_k and bd.R < self.spec_k + 1:
            raise ValueError(f"spec_k={spec_k} needs a GDN state ring R >= {spec_k + 1} (have {bd.R})")
        self.max_rows_step = min(int(max_rows_step), bd.max_rows)
        # Decode rows and prefill rows do not cost the same.  Measured gate|up: 237.9 us for
        # M=1..32, 261.2 at 64, 479.0 at 128 -- the weight traffic is shared up to the knee and
        # then is not, so a decode step past 64 rows pays nearly double for rows that share no
        # traffic, and a bucket beyond the live row count pads at pure loss.  Decode rows are
        # therefore capped at the knee; the 96..512 buckets stay available for prefill chunks,
        # where the rows do share weight traffic.
        self.max_decode_rows = max(1, min(int(max_decode_rows), self.max_rows_step))
        self.prefill_chunk = int(prefill_chunk)
        self.max_active = int(max_active or bd.max_slots)
        self.waiting: List[Seq] = []
        self.active: List[Seq] = []
        self.reserved = 0
        self.steps = 0
        self.rows_total = 0
        self.lock = threading.Lock()

    def add(self, s: Seq) -> None:
        need = (len(s.prompt) + s.max_new + PAGE) // PAGE + 1
        if need > self.bd.pages_total or len(s.prompt) + s.max_new + 1 > self.bd.max_pages * PAGE:
            raise ValueError(f"request needs {len(s.prompt) + s.max_new} tokens of context; "
                             f"cap is {min(self.bd.pages_total, self.bd.max_pages) * PAGE}")
        s.reserved_pages = need
        with self.lock:
            self.waiting.append(s)

    def _admit(self) -> None:
        with self.lock:
            while (self.waiting and len(self.active) < self.max_active and self.bd.free_slots
                   and self.reserved + self.waiting[0].reserved_pages <= self.bd.pages_total):
                s = self.waiting.pop(0)
                s.slot = self.bd.alloc_slot()
                self.bd.reset_slot_state(s.slot)
                self.reserved += s.reserved_pages
                s.t_admit = time.time()
                if s.temperature > 0:
                    s.gen = torch.Generator(device="cpu")
                    s.gen.manual_seed(int(s.seed) if s.seed is not None else int(time.time_ns() & 0x7fffffff))
                self.active.append(s)

    def _finish(self, s: Seq, why: str) -> None:
        s.done, s.finish = True, why
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
                    "pages_reserved": self.reserved, "pages_total": self.bd.pages_total,
                    "steps": self.steps, "rows_total": self.rows_total,
                    "max_decode_rows": self.max_decode_rows,
                    "spec_k": self.spec_k, "max_rows_step": self.max_rows_step}

    def admissible(self, prompt_len: int, max_new: int) -> bool:
        """True if a request of this size could ever be admitted (page pool / context cap)."""
        need = (prompt_len + max_new + PAGE) // PAGE + 1
        return (need <= self.bd.pages_total
                and prompt_len + max_new + 1 <= self.bd.max_pages * PAGE)

    def _chunk_len(self, s: Seq, budget: int) -> int:
        """Rows to feed from this sequence's prompt now, honouring chunk_align."""
        left = len(s.prompt) - s.fed
        c = min(self.prefill_chunk, budget, left)
        if self.chunk_align > 1 and c < left:
            end = s.fed + c
            snapped = end - (end % self.chunk_align)
            if snapped > s.fed:                       # never snap a chunk away to nothing
                c = snapped - s.fed
        return c

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
        return max(0, self.max_decode_rows - self._scheduled_decoders())

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
            import hashlib

            s.digests.append(hashlib.sha256(self.bd.logits[row].view(torch.int16).cpu().numpy().tobytes())
                             .hexdigest()[:16])
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

    @torch.no_grad()
    def _step_spec(self) -> int:
        """E-SPEC in the batch: (k-1) batched MTP draft passes, ONE trunk pass verifying k+1 rows per
        decoding sequence (+ prompt chunks), then one MTP confirm/prefill pass.  Greedy only.  Every
        emitted token and logits row is a trunk row, and trunk rows are batch-invariant, so the output is
        bitwise the non-speculative output (gated by bi_gate_engine.py --spec-k)."""
        bd, k = self.bd, self.spec_k
        S1 = bd.max_slots + 1
        dec = [s for s in self.active if s.fed == len(s.prompt) and s.drafts]
        for j in range(1, k):
            if not dec:
                break
            bd.run_mtp([(s.slot, s.fed + len(s.out) - 1 + j - 1, s.drafts[j - 1], s.slot, 1) for s in dec])
            mam = bd.mam[:len(dec)].tolist()
            for s, t in zip(dec, mam):
                s.drafts.append(int(t))
        seqs, owners = [], []
        budget = self.max_rows_step
        for s in dec:
            if budget >= k + 1:
                seqs.append((s.slot, s.fed + len(s.out) - 1, [s.out[-1]] + s.drafts[:k]))
                owners.append((s, "dec"))
                budget -= k + 1
        for s in self.active:
            if s.fed < len(s.prompt) and budget > 0:
                c = self._chunk_len(s, budget)
                seqs.append((s.slot, s.fed, s.prompt[s.fed:s.fed + c]))
                owners.append((s, c))
                budget -= c
        if not seqs:
            return 0
        Mb, ranges = bd.run(seqs)
        am = bd.am[:Mb].tolist()
        mrows = []
        for (s, kind), (r0, n) in zip(owners, ranges):
            if kind == "dec":
                L = s.fed + len(s.out) - 1
                a = 0
                while a < k and s.drafts[a] == am[r0 + a]:
                    a += 1
                s.prop += k
                s.acc += a
                done = False
                for i in range(a + 1):
                    if self._emit(s, int(am[r0 + i]), r0 + i):
                        done = True
                        break
                s.drafts = []
                if not done:
                    for i in range(a + 1):
                        mrows.append((s, (s.slot, L + i, int(am[r0 + i]), S1 + r0 + i, 1 if i == a else 0)))
            else:
                p0, c = s.fed, kind
                s.fed += c
                done = False
                t0 = None
                if s.fed == len(s.prompt):
                    t0 = int(am[r0 + c - 1])
                    done = self._emit(s, t0, r0 + c - 1)
                if not done:
                    for i in range(c):
                        p = p0 + i
                        nxt = s.prompt[p + 1] if p + 1 < len(s.prompt) else t0
                        if nxt is None:            # chunk ends mid-prompt only when p + 1 < len(prompt)
                            continue
                        last = 1 if (t0 is not None and p + 1 == len(s.prompt)) else 0
                        mrows.append((s, (s.slot, p, int(nxt), S1 + r0 + i, last)))
        if mrows:
            bd.run_mtp([r for _, r in mrows])
            mam = bd.mam[:len(mrows)].tolist()
            for (s, r), t in zip(mrows, mam):
                if r[4]:
                    s.drafts = [int(t)]
        with self.lock:
            self.active = [s for s in self.active if not s.done]
        self.steps += 1
        self.rows_total += sum(n for _, n in ranges)
        return Mb

    @torch.no_grad()
    def step(self) -> int:
        if self._reap_cancelled():
            self._admit()                                  # cancelled slots are reusable now
        self._admit()
        if not self.active:
            return 0
        if self.spec_k:
            return self._step_spec()
        seqs, owners = [], []
        budget = self.max_rows_step
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
        Mb, ranges = self.bd.run(seqs)
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
                import hashlib

                s.digests.append(hashlib.sha256(self.bd.logits[row].view(torch.int16).cpu().numpy().tobytes())
                                 .hexdigest()[:16])
            if not s.out:
                s.t_first = time.time()
            s.out.append(t)
            if s.on_token is not None:
                s.on_token(s, t)
            if t in s.stop_ids:
                self._finish(s, "stop")
            elif len(s.out) >= s.max_new:
                self._finish(s, "length")
        with self.lock:
            self.active = [s for s in self.active if not s.done]
        self.steps += 1
        self.rows_total += sum(n for _, n in ranges)
        return Mb


# ---------------------------------------------------------------------------- loading
def load_model(*, bundle: Optional[str] = None, parent: Optional[str] = None, gguf: Optional[str] = None,
               tune: str, device: str = "cuda:0", spec: bool = False, log=print) -> Dict[str, Any]:
    """Text-only model for the batch engine.
    bundle=<TBE bundle dir>                   -> exact TBE codec (G1-gated at load)
    parent=<HF dir>, gguf=<file>              -> the GGUF's linears (Q8_0 / K-quants fused) on the
                                                 parent's non-linear parameters; the parent is
                                                 read on the HOST, only the GGUF bytes go to the GPU
    parent=<HF dir>                           -> dense bf16 parent"""
    from transformers import AutoConfig, AutoTokenizer

    t0 = time.time()
    rec: Dict[str, Any] = {}
    tn = fdm.load_tune(tune)
    if bundle:
        from .server import _parse, build_state

        st = build_state(_parse(["--bundle", bundle, "--backend", "tbe", "--exec-mode", "exact", "--device", device]
                                + ([] if spec else ["--text-only"])
                                + ["--gate", os.environ.get("BIDEC_GATE", "sample"), "--port", "0"]), log=log)
        model = st.engine.model
        mtp = st.engine.loaded.mtp if spec else None
        tok = st.engine.tok
        rec["weight_gate"] = (getattr(st, "receipts", None) or {}).get("weight_gate")
        src = bundle
    else:
        from .loader import _disable_cuda_only_conv_kernels, model_class_for

        config = AutoConfig.from_pretrained(parent)
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
        if spec:
            from .loader import load_dense_mtp

            mtp = load_dense_mtp(parent, config, torch.device(device))
    fd = fdm.FastDecoder(model, mtp, tune=tn, max_len=512, R=1, device=device)
    torch.cuda.synchronize()
    rec["load_s"] = round(time.time() - t0, 1)
    rec["descriptors"] = fd.descriptor_kinds()
    rec["source"] = src
    log(f"[bidec] model loaded in {rec['load_s']} s; {rec['descriptors']}")
    return {"model": model, "tok": tok, "fd": fd, "rec": rec}


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


__all__ = ["Batcher", "Seq", "BatchDecoder", "BUCKETS", "PAGE", "bucket_for", "ext", "load_model", "stop_ids_for"]
