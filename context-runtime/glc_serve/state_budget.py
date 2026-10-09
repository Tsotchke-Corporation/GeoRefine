"""Per-user STATE LEDGER of the batch engine, and the one-card budget it implies.

Every byte the ``bidec.BatchDecoder`` allocates on the device, as a function of the knobs
that size it -- ``slots``, ``pages``, ``max_ctx``, ``R`` (the GDN recurrent ring, which
``bidec_serve`` sets to ``spec_k + 1``), ``spec`` (MTP head loaded), ``max_rows``,
``item_cap``, ``conv_ring`` -- written down line by line from the allocation code
(``bidec.py`` ``BatchDecoder.__init__``, ``_Meta``, ``bigemm.Workspace``/``BILin.reserve``,
``fastdec.FastDecoder.__init__`` for what rc7 left alive) and then CHECKED against that code:
``meta_engine`` runs the real ``BatchDecoder.__init__`` with every tensor on PyTorch's
``meta`` device (shapes and dtypes, no storage), and ``tests/test_state_budget.py`` asserts
the ledger equals the bytes of the tensors the constructor actually created.  So a change to
the allocation code that is not reflected here turns a test red.

What it does NOT know, and says so in every table: the CUDA context, the CUDA-graph exec
objects, cuBLAS/allocator slack and fragmentation.  Those are measured items; pass them with
``--measured`` (a run receipt) and the table uses them, otherwise the documented assumption
is used and labelled as one.

    python -m release.glc_serve.state_budget                  # A100-40, 27B, rc7 vs cuts
    python -m release.glc_serve.state_budget --measured run6c.json --json out.json

Import is torch-free (``glcbench`` resolves knob values through it on the bench driver);
``meta_engine`` imports torch lazily.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

PAGE = 256
GIB = 1 << 30
MIB = 1 << 20
#: bidec.BUCKETS, repeated here so this module stays torch-free; a test pins the equality.
BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
#: rc7 conv-history ring (``bidec_kernels.cu`` ``& 15``).
RC7_CONV_RING = 16
#: the GDN causal conv kernel width (``b_gdn_conv_kernel`` reads p-3 .. p).
CONV_TAPS = 4

#: A100-40GB, as the owner states it: usable device bytes = 40,440.375 MiB.
A100_40_USABLE_BYTES = 42_404_806_656
#: Weights at the entropy floor with embed / vision / MTP on the host (owner's figure).
WEIGHTS_DEVICE_BYTES_27B = 33_600_000_000
#: CUDA context + driver: ASSUMED until a card run reports it (``--measured``).
CUDA_CONTEXT_BYTES_ASSUMED = int(0.6 * GIB)
#: CUDA-graph exec objects + caching-allocator slack: ASSUMED (no rc7 receipt splits it out).
GRAPH_AND_SLACK_BYTES_ASSUMED = 256 * MIB
#: Codex's codec-v2 weight projection (FULL_CONTEXT_BUDGET.json), for the sensitivity row.
WEIGHTS_CODEC_V2_PROJECTION = 37_041_903_936
#: KVP1 ALLOCATED ratio (fixed per-kind slot caps, header and stream padding included) measured
#: on real Qwen3.5-2B pages -- a PROXY: .icc/evidence/state-budget-20261004/kvp_census_*.json
KVP1_ALLOCATED_RATIO_PROXY = 1.4362
#: raw hi frames a user holds: the page being written + the next (a step may straddle a page)
KVP1_HOT_PAGES = 2


# =============================================================================== knobs
def conv_ring_for(spec_k: int) -> int:
    """The smallest conv-history ring that is exact at speculation depth ``spec_k``.

    ``b_gdn_conv_kernel`` writes the pre-conv input of position p to ring slot ``p & (N-1)``
    and then reads positions p-3, p-2, p-1.  Within one sequence call the reads hit slots the
    same thread wrote at t-3..t-1, so N >= CONV_TAPS (4) is exact without speculation.  With
    speculation a verify block feeds positions P .. P+k; if a drafts are accepted the next
    block starts at P+a+1 and reads P+a-2 .. P+a, while the REJECTED rows P+a+1 .. P+k have
    already overwritten their slots.  A rejected position q aliases a needed position r iff
    q - r is a non-zero multiple of N, and q - r ranges over 1 .. k+2, so exactness needs
    N >= k + 3 (a power of two, because the kernel masks).  rc7's 16 is therefore exact for
    k <= 13 and silently WRONG for k >= 14, which rc7 does not refuse.
    """
    k = max(0, int(spec_k))
    need = max(CONV_TAPS, k + CONV_TAPS - 1)
    n = 1
    while n < need:
        n <<= 1
    return n


def check_conv_ring(conv_ring: int, spec_k: int) -> int:
    """Validate an explicit ring; 0 means "derive from spec_k".  Returns the ring to use."""
    n = int(conv_ring)
    if n == 0:
        return conv_ring_for(spec_k)
    if n < CONV_TAPS or n & (n - 1):
        raise ValueError(f"gdn conv ring must be a power of two >= {CONV_TAPS}, got {n}")
    if n < int(spec_k) + CONV_TAPS - 1:
        raise ValueError(
            f"gdn conv ring {n} is too shallow for spec_k={spec_k}: a rejected draft would "
            f"overwrite a history slot the next verify block reads (needs >= "
            f"{int(spec_k) + CONV_TAPS - 1}, i.e. {conv_ring_for(spec_k)}). rc7 accepted this "
            f"configuration for spec_k >= 14 and served wrong bits.")
    return n


def resolve_max_rows(max_rows: int, *, row_tile: int = 64, row_budget_tiles: int = 0) -> int:
    """``--max-rows 0`` = the row budget the scheduler will actually use.

    ``Batcher.step`` never puts more than ``row_budget()`` rows in a step, and with
    ``--row-budget-tiles T`` that is ``row_tile * T``.  Every M-sized buffer (logits, gate|up,
    the hidden rows, the split-K workspace) and every CUDA graph above that bucket is then
    dead weight.  Without a tile budget the budget is ``max_rows_step`` and rc7's 256 stands.
    """
    m = int(max_rows)
    if m > 0:
        return m
    if int(row_budget_tiles) > 0:
        return max(1, int(row_tile) * int(row_budget_tiles))
    return 256


def buckets_for(max_rows: int) -> Tuple[int, ...]:
    """``BatchDecoder.buckets``: the BUCKETS at or below max_rows (bidec.py:226)."""
    b = tuple(x for x in BUCKETS if x <= int(max_rows))
    return b or (int(max_rows),)


# =============================================================================== shapes
@dataclass(frozen=True)
class Shapes:
    """Only the model shapes the engine's allocations depend on."""
    name: str
    H: int                  # hidden
    V: int                  # vocab (lm head rows)
    F: int                  # MLP intermediate
    n_att: int              # full-attention layers
    n_gdn: int              # Gated-DeltaNet layers
    NQ: int                 # query heads
    NKV: int                # KV heads
    D: int = 256            # head dim (256 hybrid; 128 dense since ATTN_HEADDIM_20261004)
    gdn_hv: int = 48        # GDN value heads (bidec hard-codes 48)
    gdn_dk: int = 128
    gdn_dv: int = 128
    conv_C: int = 10240     # GDN conv channels = 2*16*128 + 48*128
    gdn_N: int = 16480      # qkv|z|b|a output rows of the fused GDN projection
    gdn_vdim: int = 6144    # GDN value width = 48 * 128
    rope_dim: int = 64      # partial rotary: 0.25 * 256
    # Attention layout (fastdec.attn_layout, ATTN_HEADDIM_20261004): the hybrid's q_proj emits
    # q|gate, and its q/k norm and hidden RMSNorm are unit-offset (fastdec.NORM_UNIT_OFFSET = 1).
    gated: bool = True
    qk_norm: int = 1
    norm_mode: int = 1

    @property
    def attn_N(self) -> int:     # q|gate (2*NQ*D, gated) or q (NQ*D) + k + v
        return (2 if self.gated else 1) * self.NQ * self.D + 2 * self.NKV * self.D


#: Qwen3.8-27B, from the tuned shapes in miv_tbe_configs/miv_wpr_rtxpro6000.json
#: (248320x5120 lm head, 34816x5120 gate|up, 16480x5120 GDN, 14336x5120 attention q|k|v,
#: 5120x17408 down, 5120x6144 o / GDN out) and the kernels' constants.
QWEN38_27B = Shapes(name="Qwen3.8-27B", H=5120, V=248320, F=17408, n_att=16, n_gdn=48,
                    NQ=24, NKV=4)


@dataclass
class EngineConfig:
    """The BatchDecoder / Batcher knobs that size device memory."""
    slots: int
    pages: int
    max_ctx: int
    spec_k: int = 0                  # R = spec_k + 1 (bidec_serve.py:126); MTP loaded iff > 0
    max_rows: int = 256
    item_cap: Optional[int] = None
    conv_ring: int = RC7_CONV_RING
    dummy_state: bool = True         # rc7 allocates GDN state + conv history for the dummy slot
    fd_buffers_live: bool = True     # rc7 leaves FastDecoder's single-stream buffers allocated
    sms: int = 108                   # A100 SM count; sets the BI-GEMM split-K workspace
    prefix_cache_mb: int = 0
    fd_max_len: int = 512            # bidec.load_model builds FastDecoder(max_len=512)

    @property
    def R(self) -> int:
        return int(self.spec_k) + 1

    @property
    def spec(self) -> bool:
        return int(self.spec_k) > 0

    @property
    def max_pages(self) -> int:
        return (int(self.max_ctx) + PAGE - 1) // PAGE

    @property
    def g_cap(self) -> int:
        g = int(self.pages) + int(self.max_rows) + 8
        return max(g, int(self.item_cap)) if self.item_cap else g


def rc7(slots: int, pages: int, max_ctx: int, **kw) -> EngineConfig:
    """rc7 as shipped: R = spec_k + 1, max_rows 256, conv ring 16, dummy-slot state, fd alive."""
    return EngineConfig(slots=slots, pages=pages, max_ctx=max_ctx, **kw)


def rc7_knobs(slots: int, pages: int, max_ctx: int, **kw) -> EngineConfig:
    """THIS source at rc7's knob defaults (--max-rows 256, --gdn-conv-ring 16): the
    unconditional cuts (no dummy-slot state, fd buffers released) are in the code, not knobs."""
    kw.setdefault("dummy_state", False)
    kw.setdefault("fd_buffers_live", False)
    return EngineConfig(slots=slots, pages=pages, max_ctx=max_ctx, **kw)


def with_cuts(slots: int, pages: int, max_ctx: int, *, spec_k: int = 0, row_tile: int = 64,
              row_budget_tiles: int = 1, **kw) -> EngineConfig:
    """The free bit-exact cuts of STATE_BUDGET_20261004 applied."""
    return EngineConfig(slots=slots, pages=pages, max_ctx=max_ctx, spec_k=spec_k,
                        max_rows=resolve_max_rows(0, row_tile=row_tile,
                                                  row_budget_tiles=row_budget_tiles),
                        conv_ring=conv_ring_for(spec_k), dummy_state=False,
                        fd_buffers_live=False, **kw)


# =============================================================================== ledger
@dataclass
class Item:
    name: str
    bytes: int
    scales_with: str                 # "slot" | "page" | "row" | "work_item" | "ctx" | "fixed"
    per_unit: int                    # bytes per one unit of scales_with
    units: int
    source: str                      # where the allocation is
    note: str = ""


def _split_for(N: int, K: int, sms: int) -> int:
    """bigemm.split_for, verbatim (BN = 64, CH = 128); a test pins the equality."""
    BN, CH = 64, 128
    ctas = (N + BN - 1) // BN
    s = max(1, min(8, (2 * sms) // max(ctas, 1)))
    nch = K // CH
    while s > 1 and nch // s < 4:
        s -= 1
    return s


def gemm_shapes(sh: Shapes, spec: bool) -> List[Tuple[str, int, int]]:
    """(name, N, K) of every BI-GEMM the engine wraps (one fused descriptor per projection)."""
    out = [("lm_head", sh.V, sh.H), ("gate_up", 2 * sh.F, sh.H), ("down", sh.H, sh.F),
           ("gdn_qkvzba", sh.gdn_N, sh.H), ("gdn_out", sh.H, sh.gdn_vdim),
           ("attn_qkv", sh.attn_N, sh.H), ("attn_o", sh.H, sh.NQ * sh.D)]
    if spec:
        out += [("mtp_fc", sh.H, 2 * sh.H), ("mtp_qkv", sh.attn_N, sh.H),
                ("mtp_o", sh.H, sh.NQ * sh.D), ("mtp_gate_up", 2 * sh.F, sh.H),
                ("mtp_down", sh.H, sh.F)]
    return out


def workspace_bytes(sh: Shapes, cfg: EngineConfig) -> int:
    """bigemm.Workspace after ``reserve(max_rows)`` of every descriptor: max S*M*N fp32."""
    n = 0
    for _nm, N, K in gemm_shapes(sh, cfg.spec):
        S = _split_for(N, K, cfg.sms)
        if S > 1:
            n = max(n, S * cfg.max_rows * N)
    return 4 * n


def fd_dead_bytes(sh: Shapes, cfg: EngineConfig, *, max_m: int = 8) -> Dict[str, int]:
    """FastDecoder's single-stream buffers (fastdec.py:369-436) that rc7 leaves allocated.

    BatchDecoder releases fd.kc / vc / hist / state (bidec.py:244) and uses NONE of the rest:
    every bidec read of ``fd`` is a weight, a norm, the embedding, the rotary tables or the
    MTP head's weights.  ``max_m`` is fastdec.MAX_M; ``max_len`` is load_model's 512.
    """
    L, H, bf = cfg.fd_max_len, sh.H, 2
    NS = (L + PAGE - 1) // PAGE
    d = {"h_x_xf_proj_dn": 5 * max_m * H * bf, "zba": max_m * sh.gdn_N * bf,
         "conv": max_m * sh.conv_C * bf, "go": max_m * sh.gdn_vdim * bf,
         "aqkv": max_m * sh.attn_N * bf, "q_ao": 2 * max_m * sh.NQ * sh.D * bf,
         "gu_act": 3 * max_m * sh.F * bf, "logits": max_m * sh.V * bf,
         "pacc_pml": max_m * sh.NQ * NS * (sh.D + 2) * 4,
         "ints": (3 * max_m + 2 + L + 16) * 4}
    if cfg.spec:
        d.update({"mtp_mkc_mvc": 2 * L * sh.NKV * sh.D * bf,
                  "mtp_rows": 7 * max_m * H * bf, "mtp_logits": sh.V * bf,
                  "mtp_ints": (2 * max_m + 4) * 4})
    return d


def ledger(sh: Shapes, cfg: EngineConfig) -> List[Item]:
    """Every device allocation of BatchDecoder.__init__ (bidec.py:196-336), sized."""
    bf, f32, i32 = 2, 4, 4
    S1 = cfg.slots + 1
    st_slots = S1 if cfg.dummy_state else cfg.slots
    M, H = cfg.max_rows, sh.H
    it: List[Item] = []

    def add(name, scales, per, units, src, note=""):
        it.append(Item(name, int(per) * int(units), scales, int(per), int(units), src, note))

    page_kv = sh.n_att * PAGE * sh.NKV * sh.D * bf * 2
    add("kv_pages", "page", page_kv, cfg.pages + 1, "bidec.py:277-278 kc,vc",
        "bf16 K+V for the 16 attention layers; page 0 is the dummy slot's")
    if cfg.spec:
        add("mtp_kv_pages", "page", PAGE * sh.NKV * sh.D * bf * 2, cfg.pages + 1,
            "bidec.py:309-310 mkc,mvc", "MTP head's own attention layer; NOT in rc7 state_bytes")
    gdn_slot = sh.n_gdn * cfg.R * sh.gdn_hv * sh.gdn_dk * sh.gdn_dv * f32
    add("gdn_state", "slot", gdn_slot, st_slots, "bidec.py:280 state",
        f"fp32 recurrent ring, R={cfg.R} = spec_k + 1"
        + ("; includes the never-touched dummy slot" if cfg.dummy_state else ""))
    add("gdn_conv_hist", "slot", sh.n_gdn * cfg.conv_ring * sh.conv_C * bf, st_slots,
        "bidec.py:279 hist", f"bf16 pre-conv ring of {cfg.conv_ring} positions")
    add("hsrc_mhid", "slot", H * bf, S1, "bidec.py:285 hsrc[:S1]", "per-slot MTP hidden store")
    add("hsrc_xf", "row", H * bf, M, "bidec.py:285 hsrc[S1:]")
    add("h_x_proj_dn", "row", 4 * H * bf, M, "bidec.py:282")
    add("zba", "row", sh.gdn_N * bf, M, "bidec.py:287")
    add("conv_out", "row", sh.conv_C * bf, M, "bidec.py:288")
    add("go", "row", sh.gdn_vdim * bf, M, "bidec.py:289")
    add("aqkv", "row", sh.attn_N * bf, M, "bidec.py:290")
    add("q_ao", "row", 2 * sh.NQ * sh.D * bf, M, "bidec.py:291-292")
    add("gu_act", "row", 3 * sh.F * bf, M, "bidec.py:294-295")
    add("logits", "row", sh.V * bf, M, "bidec.py:296", "bf16 [max_rows, vocab]")
    add("am", "row", i32, M, "bidec.py:297")
    if cfg.spec:
        add("mtp_rows", "row", 7 * H * bf, M, "bidec.py:311-312 me,mhin,mh,mx,mg,mcat")
        add("mtp_logits", "row", sh.V * bf, M, "bidec.py:313")
        add("mam", "row", i32, M, "bidec.py:314")
    add("pacc_pml", "work_item", sh.NQ * sh.D * f32 + sh.NQ * 2 * f32, cfg.g_cap,
        "bidec.py:298-299", f"fp32 split partials; g_cap = {cfg.g_cap}")
    add("bigemm_workspace", "fixed", workspace_bytes(sh, cfg), 1, "bigemm.py:140-155 + BILin.reserve",
        f"split-K fp32 workspace at max_rows={M}, {cfg.sms} SMs (scales with max_rows)")
    # 10 row-sized fields: vtok, row_slot, row_pos, row_item0, seq_slot, seq_row0, seq_len,
    # hidx, hflag, and the multimodal lane's mmflag (bidec_mm; zero on a text-only step but
    # always allocated).
    meta_n = 10 * M + 3 + 2 * cfg.g_cap + 2 * S1 + S1 * cfg.max_pages
    add("step_metadata", "fixed", meta_n * i32, 1, "bidec.py:152-175 _Meta.dev",
        "device int32 metadata incl. pagetab (+ the same bytes pinned on the host)")
    add("rope_tables", "ctx", 2 * sh.rope_dim * bf, cfg.max_ctx + 64, "bidec.py:338-352",
        "cos+sin, bf16, (max_ctx + 64) x rope_dim")
    if cfg.fd_buffers_live:
        add("fastdec_dead_buffers", "fixed", sum(fd_dead_bytes(sh, cfg).values()), 1,
            "fastdec.py:369-436", "single-stream buffers bidec never reads")
    if cfg.prefix_cache_mb:
        add("prefix_cache", "fixed", cfg.prefix_cache_mb * MIB, 1, "bidec_prefix.py BidecSlotStore",
            "device clones of KV blocks + recurrent snapshots, up to the budget")
    return it


def summarize(items: Sequence[Item]) -> Dict[str, Any]:
    by = {}
    for i in items:
        by[i.scales_with] = by.get(i.scales_with, 0) + i.bytes
    return {"total_bytes": sum(i.bytes for i in items), "by_scaling": by,
            "items": [asdict(i) for i in items]}


def per_slot_bytes(sh: Shapes, cfg: EngineConfig) -> int:
    """The fixed per-user tax: GDN ring + conv history + MTP hidden row."""
    return (sh.n_gdn * cfg.R * sh.gdn_hv * sh.gdn_dk * sh.gdn_dv * 4
            + sh.n_gdn * cfg.conv_ring * sh.conv_C * 2 + sh.H * 2)


def per_page_bytes(sh: Shapes, cfg: EngineConfig, *, kv_ratio: float = 1.0) -> int:
    kv = sh.n_att * PAGE * sh.NKV * sh.D * 2 * 2
    if cfg.spec:
        kv += PAGE * sh.NKV * sh.D * 2 * 2
    return int(math.ceil(kv / float(kv_ratio)))


def pages_reserved(tokens: int) -> int:
    """Batcher.add's reservation for prompt + max_new = tokens (bidec.py:866)."""
    return (int(tokens) + PAGE) // PAGE + 1


# =============================================================================== meta engine
@contextlib.contextmanager
def _meta_patches(sms: int):
    """Run BatchDecoder.__init__ with no GPU: every extension is a stub, every tensor `meta`."""
    import functools

    import torch

    from glc_serve import bidec, bigemm as bg, fastdec as fdm

    saved = [(bidec, "ext", bidec.ext), (fdm, "ext", fdm.ext), (bg, "extension", bg.extension),
             (bg, "split_for", bg.split_for), (torch.cuda, "synchronize", torch.cuda.synchronize),
             (torch.cuda, "empty_cache", torch.cuda.empty_cache), (torch, "zeros", torch.zeros),
             (bidec.BatchDecoder, "_rope_tables", bidec.BatchDecoder._rope_tables)]
    zeros = torch.zeros

    def _zeros(*a, **k):
        k.pop("pin_memory", None)              # the host half of _Meta; not device memory
        return zeros(*a, **k)

    def _rope(self, n):
        t = torch.empty(n + 64, self.fd.rope_dim, dtype=torch.bfloat16, device=self.dev)
        return t, torch.empty_like(t)

    try:
        bidec.ext = fdm.ext = bg.extension = (lambda: None)
        bg.split_for = functools.partial(saved[3][2], sms=sms)
        torch.cuda.synchronize = lambda *a, **k: None
        torch.cuda.empty_cache = lambda *a, **k: None
        torch.zeros = _zeros
        bidec.BatchDecoder._rope_tables = _rope
        yield
    finally:
        for mod, name, val in saved:
            setattr(mod, name, val)


def _fake_fd(sh: Shapes, cfg: EngineConfig):
    """A FastDecoder-shaped object whose descriptors are meta bf16 weights."""
    import types

    import torch

    dev = torch.device("meta")

    def desc(N, K):
        return types.SimpleNamespace(kind="bf16", N=N, K=K, bytes=0, coded_bytes=0,
                                     w=torch.empty(N, K, dtype=torch.bfloat16, device=dev))

    layers = []
    ai = gi = 0
    for i in range(sh.n_att + sh.n_gdn):
        is_attn = (i % 4) == 3
        o = types.SimpleNamespace(kind="attn" if is_attn else "gdn", F=sh.F,
                                  gu=desc(2 * sh.F, sh.H), down=desc(sh.H, sh.F))
        if is_attn:
            o.qkv, o.o, o.ai, o.NQ, o.NKV = desc(sh.attn_N, sh.H), desc(sh.H, sh.NQ * sh.D), ai, sh.NQ, sh.NKV
            o.D, o.gated, o.qk_norm = sh.D, int(sh.gated), sh.qk_norm
            ai += 1
        else:
            o.qkvzba, o.out, o.gi = desc(sh.gdn_N, sh.H), desc(sh.H, sh.gdn_vdim), gi
            o.C, o.vdim = sh.conv_C, sh.gdn_vdim
            gi += 1
        layers.append(o)
    fd = types.SimpleNamespace(dev=dev, H=sh.H, eps=1e-6, V=sh.V, NQ=sh.NQ, NKV=sh.NKV,
                               n_att=sh.n_att, n_gdn=sh.n_gdn, layers=layers,
                               lm=desc(sh.V, sh.H), mtp_ready=cfg.spec, model=None,
                               rope_dim=sh.rope_dim, kc=None, vc=None, hist=None, state=None,
                               # attention shape (BatchDecoder reads these off fd since
                               # ATTN_HEADDIM_20261004)
                               D=sh.D, attn_gated=int(sh.gated), qk_norm=sh.qk_norm,
                               norm_mode=sh.norm_mode)
    if cfg.spec:
        fd.m = types.SimpleNamespace(pre_e=None, pre_h=None, in_ln=None, post_ln=None, norm=None,
                                     qnw=None, knw=None, scaling=1.0, fc=desc(sh.H, 2 * sh.H),
                                     qkv=desc(sh.attn_N, sh.H), o=desc(sh.H, sh.NQ * sh.D),
                                     gu=desc(2 * sh.F, sh.H), down=desc(sh.H, sh.F))
    return fd


def meta_engine(sh: Shapes, cfg: EngineConfig):
    """The REAL BatchDecoder.__init__, on the meta device.  Returns the decoder."""
    from glc_serve import bidec

    fd = _fake_fd(sh, cfg)
    with _meta_patches(cfg.sms):
        kw = dict(max_slots=cfg.slots, max_rows=cfg.max_rows, pages_total=cfg.pages,
                  max_ctx=cfg.max_ctx, R=cfg.R, item_cap=cfg.item_cap, log=lambda *a: None)
        import inspect
        if "conv_ring" in inspect.signature(bidec.BatchDecoder.__init__).parameters:
            kw["conv_ring"] = cfg.conv_ring
        bd = bidec.BatchDecoder(fd, **kw)
    return bd


def device_tensors(bd) -> Dict[str, int]:
    """Bytes of every base (non-view) tensor the decoder owns, by attribute name."""
    import torch

    out: Dict[str, int] = {}
    seen = set()

    def visit(name, t):
        if isinstance(t, torch.Tensor) and t._base is None and id(t) not in seen:
            seen.add(id(t))
            out[name] = out.get(name, 0) + t.numel() * t.element_size()

    for k, v in vars(bd).items():
        if k in ("fd", "L", "lm", "m"):
            continue
        visit(k, v)
    visit("bigemm_workspace", bd.ws.buf)
    visit("step_metadata", bd.meta.dev)
    return out


def ledger_vs_meta(sh: Shapes, cfg: EngineConfig) -> Dict[str, Any]:
    """The ledger's device bytes (minus the fd/prefix items the decoder does not own) and the
    bytes of the tensors the real constructor created.  Equal, or the ledger is wrong."""
    items = [i for i in ledger(sh, cfg) if i.name not in ("fastdec_dead_buffers", "prefix_cache")]
    want = sum(i.bytes for i in items)
    bd = meta_engine(sh, cfg)
    got = device_tensors(bd)
    return {"ledger_bytes": want, "meta_bytes": sum(got.values()), "meta_by_attr": got}


# =============================================================================== budget
@dataclass
class Budget:
    usable: int = A100_40_USABLE_BYTES
    weights: int = WEIGHTS_DEVICE_BYTES_27B
    cuda_context: int = CUDA_CONTEXT_BYTES_ASSUMED
    graphs_and_slack: int = GRAPH_AND_SLACK_BYTES_ASSUMED
    provenance: Dict[str, str] = field(default_factory=lambda: {
        "usable": "owner statement (40,440.375 MiB)",
        "weights": "owner statement: entropy-floor weights, embed/vision/MTP on host",
        "cuda_context": "ASSUMED 0.6 GiB (no card receipt in this tree)",
        "graphs_and_slack": "ASSUMED 256 MiB (graph exec + allocator slack; unmeasured)"})

    def apply_measured(self, m: Dict[str, Any]) -> "Budget":
        b = replace(self, provenance=dict(self.provenance))
        for key, attr in (("cuda_context_bytes", "cuda_context"),
                          ("graphs_and_slack_bytes", "graphs_and_slack"),
                          ("weights_device_bytes", "weights"), ("usable_bytes", "usable")):
            if m.get(key) is not None:
                setattr(b, attr, int(m[key]))
                b.provenance[attr] = f"MEASURED: {m.get('source', 'receipt')}"
        return b

    @property
    def for_engine(self) -> int:
        return self.usable - self.weights - self.cuda_context - self.graphs_and_slack


def _fixed_bytes(sh: Shapes, cfg: EngineConfig, kv_ratio: float = 1.0) -> int:
    """Everything that does not scale with users or pages (rows, work list, ctx, fixed)."""
    tot = 0
    for i in ledger(sh, cfg):
        if i.name in ("kv_pages", "mtp_kv_pages", "gdn_state", "gdn_conv_hist", "hsrc_mhid"):
            continue
        tot += i.bytes
    # the dummy slot's page and, in rc7, its GDN state + conv history
    tot += per_page_bytes(sh, cfg)
    if cfg.dummy_state:
        tot += per_slot_bytes(sh, cfg) - sh.H * 2
    tot += sh.H * 2                                   # dummy row of hsrc_mhid
    return tot


def _kv_bytes(sh: Shapes, cfg: EngineConfig, pages: int, kv_ratio: float, hot_pages: int) -> int:
    """KV bytes of ``pages`` pages of one user.  Raw (ratio 1): pages x raw.  KVP1: every page
    holds its lo plane + its reserved coded slots (raw / ratio); the hot pages also hold a raw
    hi frame (raw / 2)."""
    raw = per_page_bytes(sh, cfg)
    if kv_ratio <= 1.0:
        return pages * raw
    return pages * per_page_bytes(sh, cfg, kv_ratio=kv_ratio) + min(pages, hot_pages) * (raw // 2)


def max_users(sh: Shapes, mk, budget: Budget, ctx_tokens: int, *, kv_ratio: float = 1.0,
              hot_pages: int = 1) -> Dict[str, Any]:
    """Largest N such that N users of ``ctx_tokens`` (prompt + max_new) fit.

    ``mk(slots, pages, max_ctx)`` builds the EngineConfig (rc7 or with cuts).  With
    ``kv_ratio`` > 1 every page except ``hot_pages`` per user is stored coded (PROJECTED).
    """
    need = pages_reserved(ctx_tokens)
    best = None
    lo, hi = 0, 4096
    while lo < hi:
        n = (lo + hi + 1) // 2
        cfg = mk(n, n * need, max(ctx_tokens + 1, PAGE))
        kv = n * _kv_bytes(sh, cfg, need, kv_ratio, hot_pages)
        total = _fixed_bytes(sh, cfg) + n * per_slot_bytes(sh, cfg) + kv
        if total <= budget.for_engine:
            lo, best = n, {"users": n, "engine_bytes": total, "pages": n * need}
        else:
            hi = n - 1
    return best or {"users": 0, "engine_bytes": 0, "pages": 0}


def max_context(sh: Shapes, mk, budget: Budget, *, kv_ratio: float = 1.0,
                hot_pages: int = 1) -> Dict[str, Any]:
    """Longest single-user context (prompt + max_new) that fits."""
    lo, hi, best = PAGE, 1 << 21, None
    while lo < hi:
        L = (lo + hi + 1) // 2
        need = pages_reserved(L)
        cfg = mk(1, need, L + 1)
        kv = _kv_bytes(sh, cfg, need, kv_ratio, hot_pages)
        total = _fixed_bytes(sh, cfg) + per_slot_bytes(sh, cfg) + kv
        if total <= budget.for_engine:
            lo, best = L, {"tokens": L, "pages": need, "engine_bytes": total}
        else:
            hi = L - 1
    return best or {"tokens": 0, "pages": 0, "engine_bytes": 0}


def budget_table(sh: Shapes = QWEN38_27B, budget: Optional[Budget] = None, *,
                 ctxs: Sequence[int] = (4096, 8192, 32768), spec_k: int = 0,
                 kv_ratio: float = KVP1_ALLOCATED_RATIO_PROXY) -> Dict[str, Any]:
    budget = budget or Budget()
    arms = {
        "rc7": (lambda s, p, c: rc7(s, p, c, spec_k=spec_k), 1.0,
                "rc7 as shipped (max_rows 256, conv ring 16, dummy-slot state, fd buffers alive)"),
        "cuts": (lambda s, p, c: with_cuts(s, p, c, spec_k=spec_k), 1.0,
                 "free bit-exact cuts (max_rows = 1 tile, minimal conv ring, no dummy state, "
                 "fd buffers released)"),
        "cuts+kvp": (lambda s, p, c: with_cuts(s, p, c, spec_k=spec_k), kv_ratio,
                     f"cuts + KVP1 on-device lossless KV: cold pages at the ALLOCATED ratio "
                     f"{kv_ratio}x, {KVP1_HOT_PAGES} raw pages per user (PROJECTED: kernel "
                     "GPU-unmeasured, ratio measured on Qwen3.5-2B proxy pages)"),
    }
    rows = []
    for name, (mk, ratio, desc) in arms.items():
        hp = KVP1_HOT_PAGES if ratio > 1.0 else 1
        r = {"arm": name, "description": desc}
        for c in ctxs:
            r[f"users@{c // 1024}k"] = max_users(sh, mk, budget, c, kv_ratio=ratio, hot_pages=hp)
        r["max_single_context"] = max_context(sh, mk, budget, kv_ratio=ratio, hot_pages=hp)
        cfg = mk(1, 1, PAGE)
        r["per_slot_bytes"] = per_slot_bytes(sh, cfg)
        r["per_page_bytes"] = per_page_bytes(sh, cfg, kv_ratio=ratio)     # cold page, allocated
        r["per_token_bytes_raw"] = per_page_bytes(sh, cfg) // PAGE
        rows.append(r)
    return {"schema": "glc.state_budget.v1", "model": sh.name, "spec_k": spec_k,
            "budget": {**{k: v for k, v in asdict(budget).items() if k != "provenance"},
                       "for_engine": budget.for_engine, "provenance": budget.provenance},
            "status": "PROJECTED until the A100-40 card run: every engine byte is the allocation "
                      "code's own arithmetic (checked against the constructor on the meta "
                      "device); CUDA context / graphs / slack are assumptions unless --measured",
            "arms": rows}


def long_context_tiers(sh: Shapes = QWEN38_27B, budget: Optional[Budget] = None, *,
                       ctx: int = 262144, pcie_bytes_s: float = 25e9, hbm_bytes_s: float = 1.555e12,
                       engine_prefill_rows_s: float = 740.0, a100_bf16_flops: float = 312e12,
                       active_params: float = 26.9e9, kv_ratio_wire: float = 1.4624,
                       fp32_flops: float = 19.5e12) -> Dict[str, Any]:
    """One user at ``ctx`` tokens on the card: what stays resident, what pages over PCIe, and
    what exact recomputation would cost instead.  ARITHMETIC, not measurement:

    * PCIe 4.0 x16: 31.5 GB/s theoretical per direction, ``pcie_bytes_s`` (25 GB/s) is the
      pinned-H2D figure assumed achievable; HBM2 1,555 GB/s (A100-40 spec).
    * decode reads EVERY KV page every token (no sparsity: exactness), so the bytes that are
      not resident cross PCIe once per generated token, at the wire ratio of KVP1 pages
      (variable 1.4624, proxy) when the host copy is kept coded.
    * the weight stream is the other floor: weights / HBM per token.
    * exact recomputation re-prefills token ids through the unchanged engine: chunk-invariant
      (G3a-BI), so the KV/state it produces are the original bits; cost per context token is
      the engine's prefill rate (``engine_prefill_rows_s``, RTX PRO 6000 ~850 rows/s scaled by
      the HBM ratio) or, as a floor, 2 x active params FLOPs at A100 peak.
    """
    budget = budget or Budget()
    per_tok = per_page_bytes(sh, with_cuts(1, 1, PAGE)) // PAGE
    out: Dict[str, Any] = {"ctx": ctx, "kv_bytes_raw": ctx * per_tok}
    t_w = budget.weights / hbm_bytes_s
    for name, ratio, hp in (("raw_resident", 1.0, 1), ("kvp_resident", KVP1_ALLOCATED_RATIO_PROXY, KVP1_HOT_PAGES)):
        mc = max_context(sh, lambda s, p, c: with_cuts(s, p, c), budget, kv_ratio=ratio, hot_pages=hp)
        resident = min(ctx, mc["tokens"])
        off = ctx - resident
        # the host copy is coded only when the device can consume coded pages (the KVP arm)
        wire = off * per_tok / (kv_ratio_wire if ratio > 1.0 else 1.0)
        t_pcie = wire / pcie_bytes_s
        t_kv_dev = resident * per_tok / ratio / hbm_bytes_s
        step = max(t_w + t_kv_dev, t_pcie)
        out[name] = {"resident_tokens": resident, "offloaded_tokens": off,
                     "pcie_bytes_per_token": int(wire), "t_pcie_ms": round(1e3 * t_pcie, 2),
                     "t_weights_ms": round(1e3 * t_w, 2), "t_resident_kv_ms": round(1e3 * t_kv_dev, 2),
                     "tok_s_upper_bound": round(1.0 / step, 2) if step else None,
                     "all_on_device": off == 0}
    out["recompute"] = {
        "engine_s_per_context_token": round(1.0 / engine_prefill_rows_s, 6),
        "flops_floor_s_per_context_token": round(2 * active_params / a100_bf16_flops, 6),
        "pcie_s_per_context_token_coded": round(per_tok / kv_ratio_wire / pcie_bytes_s, 9),
        "steady_state_note": "a dropped KV prefix of X tokens must be re-prefilled for EVERY "
                             "generated token (later positions attend to it): X / prefill rate",
        "steady_state_s_per_token_at_kvp_offload": round(
            out["kvp_resident"]["offloaded_tokens"] / engine_prefill_rows_s, 1),
        "preemption_resume_s_full_ctx": round(ctx / engine_prefill_rows_s, 1),
        "preemption_resume_s_flops_floor": round(ctx * 2 * active_params / a100_bf16_flops, 1)}
    # Layerwise full-prefix recomputation (Codex, LAYERWISE_RECOMPUTATION_HYPOTHESIS.json): run
    # the whole prefix one LAYER at a time, keeping one bf16 residual array [ctx][H] and one
    # attention layer's KV, discarding that KV before the next layer.  The residual between
    # layers is ONE bf16 array because rmsnorm's fused add is h = bf16(h + delta) written back
    # (fastdec_kernels.cu:80-83): folding the pending delta in eagerly stores the same bits.
    hidden = ctx * sh.H * 2
    layer_kv = ctx * 2 * sh.NKV * sh.D * 2
    attn_flops = sh.n_att * sh.NQ * sh.D * 4 * ctx * ctx / 2.0      # QK + PV, causal
    gemm_s_engine = ctx / engine_prefill_rows_s
    gemm_s_floor = ctx * 2 * active_params / a100_bf16_flops
    attn_s_fp32 = attn_flops / fp32_flops                            # b_attn_split is fp32 FMA
    out["layerwise_recompute"] = {
        "hidden_bytes": hidden, "one_layer_kv_bytes": layer_kv,
        "gdn_state_one_layer_bytes": sh.gdn_hv * sh.gdn_dk * sh.gdn_dv * 4,
        "left_for_workspace_bytes": budget.for_engine - hidden - layer_kv,
        "left_for_workspace_bytes_at_codex_weights": (budget.for_engine + budget.weights
                                                      - WEIGHTS_CODEC_V2_PROJECTION - hidden - layer_kv),
        "all_on_device": True,
        "pass_s_gemm_engine": round(gemm_s_engine, 1), "pass_s_gemm_flop_floor": round(gemm_s_floor, 1),
        "pass_s_attention_fp32_floor": round(attn_s_fp32, 1),
        "s_per_generated_token": round(gemm_s_engine + attn_s_fp32, 1),
        "note": "nothing of the prefix survives a pass, so EVERY generated token is one full pass; "
                "the work list must be chunked (a 256-row chunk at 262k wants 262k items = 6.5 GB "
                "of fp32 partials, so deep chunks are a few rows)"}
    out["assumptions"] = {"pcie_bytes_s": pcie_bytes_s, "hbm_bytes_s": hbm_bytes_s, "fp32_flops": fp32_flops,
                          "engine_prefill_rows_s": engine_prefill_rows_s,
                          "a100_bf16_flops": a100_bf16_flops, "active_params": active_params,
                          "kv_ratio_wire": kv_ratio_wire}
    return out


def _fmt_table(t: Dict[str, Any]) -> str:
    b = t["budget"]
    lines = [f"{t['model']}  spec_k={t['spec_k']}  usable {b['usable']:,} B  weights {b['weights']:,} B  "
             f"ctx {b['cuda_context']:,} B  graphs+slack {b['graphs_and_slack']:,} B  "
             f"=> engine {b['for_engine']:,} B ({b['for_engine'] / GIB:.3f} GiB)", ""]
    cols = [k for k in t["arms"][0] if k.startswith("users@")]
    head = f"{'arm':10s} {'slot MiB':>9s} {'page MiB':>9s} " + " ".join(f"{c:>10s}" for c in cols) \
        + f" {'max ctx (1 user)':>18s}"
    lines += [head, "-" * len(head)]
    for r in t["arms"]:
        lines.append(f"{r['arm']:10s} {r['per_slot_bytes'] / MIB:9.2f} {r['per_page_bytes'] / MIB:9.2f} "
                     + " ".join(f"{r[c]['users']:>10d}" for c in cols)
                     + f" {r['max_single_context']['tokens']:>18,d}")
    lines += ["", t["status"]]
    for k, v in b["provenance"].items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measured", type=Path,
                    help="JSON with any of cuda_context_bytes, graphs_and_slack_bytes, "
                         "weights_device_bytes, usable_bytes, source (e.g. a run-6c receipt)")
    ap.add_argument("--spec-k", type=int, default=0)
    ap.add_argument("--kv-ratio", type=float, default=KVP1_ALLOCATED_RATIO_PROXY)
    ap.add_argument("--ledger", action="store_true", help="also print the per-item ledger")
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--pages", type=int, default=288)
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    b = Budget()
    if a.measured:
        b = b.apply_measured(json.loads(a.measured.read_text()))
    t = budget_table(QWEN38_27B, b, spec_k=a.spec_k, kv_ratio=a.kv_ratio)
    t["sensitivity"] = {
        "weights_codec_v2_projection_37.04e9": budget_table(
            QWEN38_27B, replace(b, weights=WEIGHTS_CODEC_V2_PROJECTION), spec_k=a.spec_k,
            kv_ratio=a.kv_ratio)["arms"],
        "spec_k_2": budget_table(QWEN38_27B, b, spec_k=2, kv_ratio=a.kv_ratio)["arms"]}
    t["long_context_262k"] = long_context_tiers(QWEN38_27B, b)
    if a.ledger:
        for arm, cfg in (("rc7", rc7(a.slots, a.pages, a.max_ctx, spec_k=a.spec_k)),
                         ("cuts", with_cuts(a.slots, a.pages, a.max_ctx, spec_k=a.spec_k))):
            s = summarize(ledger(QWEN38_27B, cfg))
            t.setdefault("ledgers", {})[arm] = {"config": asdict(cfg), **s}
            print(f"== ledger {arm}: slots {a.slots} pages {a.pages} max_ctx {a.max_ctx} spec_k {a.spec_k}")
            for i in s["items"]:
                print(f"  {i['name']:22s} {i['bytes']:>15,d} B  = {i['per_unit']:>12,d} x {i['units']:<6d}"
                      f" per {i['scales_with']:9s} {i['source']}")
            print(f"  {'TOTAL':22s} {s['total_bytes']:>15,d} B")
    print(_fmt_table(t))
    for name, arms in t["sensitivity"].items():
        print(f"\nsensitivity {name}: " + "; ".join(
            f"{r['arm']} " + " ".join(f"{k}={r[k]['users']}" for k in r if k.startswith("users@"))
            + f" maxctx={r['max_single_context']['tokens']:,}" for r in arms))
    lc = t["long_context_262k"]
    print("\n262k single user: " + "; ".join(
        f"{k}: resident {lc[k]['resident_tokens']:,} offloaded {lc[k]['offloaded_tokens']:,} "
        f"pcie {lc[k]['t_pcie_ms']} ms -> <= {lc[k]['tok_s_upper_bound']} tok/s"
        for k in ("raw_resident", "kvp_resident")))
    if a.json:
        a.json.write_text(json.dumps(t, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
