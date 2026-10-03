"""GLC-TBE flat64 **fused decode-in-GEMV**, Metal (MLX) -- phase 5.

WHY THIS FILE EXISTS
--------------------
Phase 4 established that the decode kernel is *at* the harness ceiling
(``v3_t4_g4``: 279.0 / 289.4 GB/s against a ``copy8`` control that reproduced
at 284.0 / 286.3) and that the remaining 0.048x-of-dense gap on the certified
Qwen3-4B artifact is **traffic**, not instruction cost
(``docs/research/TBE_METAL_ACCESS_PATTERN_20260903.md`` sec. 6):

===========================  ====================================
path                         bytes moved per token (Qwen3-4B)
===========================  ====================================
dense bf16                   8.0 GB  (read the weights)
TBE decode-then-matmul       6.2 GB  (read the container)
                            + 8.0 GB (write the decoded bf16)
                            + 8.0 GB (read it back into the GEMV)
                            = **22.2 GB**, i.e. 2.8x dense
===========================  ====================================

No decode kernel, however fast, can beat ~0.36x of dense from inside that
architecture.  Fusing the decode into the GEMV deletes both 8 GB legs: the
decoded bf16 never reaches global memory, it is consumed in registers by the
dot product that needed it.  TBE traffic becomes ~0.72-0.77x of dense -- BELOW
dense, which is the point of a compressed weight format at batch 1.

Phase 3 declined to build this, recording ``fused_gemv_prototype.attempted =
false`` with the reason *"best decode throughput 60.7 GB/s is below the 150
GB/s internal threshold"* (``.icc/evidence/tbe-metal-decode-20260903/
bench_phase3.json``).  That threshold is met by its own terms at 279-289 GB/s,
so the gate is open.

THE DECODE ROUTINE IS SHARED, NOT COPIED
----------------------------------------
``_decode_header`` emits ONE Metal function, ``tbe_decode_pair``, holding the
whole of the flat64 bit semantics.  Both kernels in this module are compiled
from that identical text:

* :func:`tbe_gemv_fused` -- the fused GEMV, which never stores a decoded
  weight to global memory;
* :func:`fused_decode_probe` -- a verification-only kernel that runs
  ``tbe_decode_pair`` over every tile and *writes* the result, so it can be
  compared byte for byte with the CPU reference ``tbe_container.decode_tbe``.

That is what makes "bit-exact weights" a meaningful claim for a path that
materialises no weights (see BIT-EXACTNESS below).

KERNEL GEOMETRY
---------------
``W`` is ``[O, I]`` row-major and flat64 tiles run over the flattened
row-major array, so a tile never straddles a row **iff** ``I % 64 == 0``.
That is a hard precondition, not a fallback: :func:`tbe_gemv_fused` raises
:class:`TBEFusedGemvError` on any other shape rather than silently routing to
decode-then-matmul, because a silent fallback would make a receipt say "fused"
while measuring the unfused path.

One SIMD group (32 lanes) owns one 64-element tile at a time, lane ``l``
owning columns ``2l`` and ``2l+1`` -- the same lane->element map as v3, so the
``simd_prefix_exclusive_sum`` escape-rank scan is still the tile's
element-ascending rank order.  Two work decompositions, both compile-time:

* ``row`` (default): SIMD group ``g`` owns output row ``g``; ``SPT`` groups
  per threadgroup cover ``SPT`` consecutive rows.  No cross-group reduction.
* ``splitk``: one threadgroup owns one row and its ``SPT`` groups split that
  row's tiles ``SPT``-strided; the partials are reduced through threadgroup
  memory in a FIXED group order (0..SPT-1), so the result is deterministic.
  Exists for narrow outputs (``k_proj`` at O=1024 launches only 1024 groups in
  ``row`` mode), where it buys occupancy.

``TPG`` unrolls that many tiles per iteration.  In v3's decode-only kernel
that was the single biggest win (``t4_g4`` 279 GB/s vs ``t1_g1`` 92).  Here it
is measured NEGATIVE and ``TPG = 1`` is the default: at TPG=4 the kernel runs
**4.7-8x slower** (511 / 781 us against 109 / 99 us at TPG=1 on 4096x2560),
and the effect is identical whether the escape rank comes from
``simd_prefix_exclusive_sum`` or from v1-style popcounts, so it is register
pressure from N inlined decode bodies sharing one accumulator chain, not the
cross-lane op.  The variant grid keeps TPG 2/4/8 so the measurement stays
reproducible, not because any of them is a candidate.

Per 64-element tile the fused kernel reads 24 B of plane words + 4 B of tile
base + 64 B of ``smb`` = **92 B** (plus the escape stream, ~3.1% of elements
at 1 B) and writes **nothing** but the final ``O`` bf16 outputs.  Against
dense's 128 B of weight read per 64 elements that is **0.72x**, before the
activation vector -- which is read once per row from cache and is ~5 KB.

BIT-EXACTNESS (two parts, both reported)
----------------------------------------
(a) *Numeric agreement with the reference path.*  The reference is
``mx.matmul(x, decode_tbe_metal_v4(c).T)``.  Exact equality is NOT achieved and
is not claimable: MLX's GEMV reduction order is opaque and this kernel's is not
(per-lane serial over tiles, then one ``simd_sum``).  Two different orders of
the same fp32 additions differ in the last fp32 bits, and the bf16 rounding of
the result agrees except when the fp32 sum lands on a rounding boundary.
Measured over 4 shapes x 2 weight draws x 12 activation vectors:

* **0.004%-0.024% of elements differ at all**;
* **<= 1 bf16 ulp** on every element carrying signal
  (``max_ulp_above_cancellation``, where "signal" is ``|y| >= max|y| / 256``);
* raw ULP reaches **10** only on catastrophic-cancellation residue, whose
  absolute delta is still within one ulp of the largest output, and where the
  fused value is nearer the fp64 truth about half the time -- neither order is
  "right" there;
* **max absolute delta <= 1 bf16 ulp of the largest output** on every shape.

:func:`ulp_report` computes all of these, and reports the raw max ULP and the
signal-restricted one as SEPARATE fields rather than picking whichever reads
better.

(b) *"246/246 bit-exact weights" for a path with no weights.*  Defined as:
every decoded element produced by ``tbe_decode_pair`` -- the routine the fused
GEMV itself uses -- equals ``tbe_container.decode_tbe`` bit for bit, checked
through :func:`fused_decode_probe`.  The fused GEMV cannot decode differently
from the probe because they are compiled from the same function text.

WHAT THE MEASUREMENT SAID (phase-5 receipts, M2 Ultra, M=1)
----------------------------------------------------------
The kernel works and is the fastest TBE path on record -- **1.2x to 9.4x** the
phase-4 decode-then-matmul -- but it does **not** reach dense:

=================  ==========  ==============  =============  ==============
shape (O x I)      dense (us)  best fused      x dense        unfused v4
=================  ==========  ==============  =============  ==============
1024 x 2560          25.8/27.6  splitk_t1_g4    0.66 / 0.72    0.46 / 0.53
4096 x 2560          60.5/54.2  splitk_t1_g8    0.66 / 0.60    0.41 / 0.37
9728 x 2560        115.0/107.0  splitk_t1_g8    0.65 / 0.65    0.36 / 0.35
16384 x 8192       474.2/501.2  splitk_t1_g16   0.56 / 0.57    0.056/ 0.061
=================  ==========  ==============  =============  ==============

The traffic statement in the table at the top is CONFIRMED (the container is
0.731x of dense's weight bytes and the fused path writes no weights) but the
performance prediction drawn from it is REFUTED, for a reason the bench
measures directly with the ``ctl_byte_gemv`` control: **bandwidth elasticity at
these sizes is not 1**.  A control GEMV reading ONE byte per weight -- half of
dense's traffic, zero decode work -- runs at only 1.15-1.29x of dense on the
three large shapes and *slower* than dense (0.86-0.91x) on the launch-bound
1024x2560.  A 50%% traffic cut is worth ~15-29%% of time, so this path's 27%%
cut is worth ~8-15%% -- less than the decode instructions cost now that the
bf16 write no longer hides them.

The phase-5 gate for wiring this into ``tbe_mlx_model.DECODERS`` was
**>= 0.9x of dense**.  It is not met, so ``TBELinearFusedMLX`` is deliberately
NOT registered and no Qwen3-4B certify was run through it.  The class is here,
tested, and ready for the day the gate is met.

NEW FILE.  No existing decoder, container, or receipt is touched.
"""
from __future__ import annotations

from typing import Optional, Tuple

import mlx.core as mx

from .tbe_decode_mlx import (  # noqa: F401  (re-exported for fused callers)
    METAL_MEMORY_LIMIT_BYTES,
    TBEDeviceMLX,
    bits_view,
    mlx_metal_available,
    upload_tbe_mlx,
)
from ..tbe_container import MODE_W6Z, TILE


class TBEFusedGemvError(RuntimeError):
    """The fused decode-in-GEMV cannot serve this container or this call."""


# ---------------------------------------------------------------------------
# the shared decode routine -- the ONLY place flat64 bit semantics appear here
# ---------------------------------------------------------------------------
_LOAD_HALF = r"""
    // A lane needs only ONE of the two 32-bit words of each bit-plane: lanes
    // 0-15 own elements 0-31 (the low word), lanes 16-31 own 32-63 (the high
    // word).  Loading only the needed half is 3 loads per lane per tile
    // instead of 6, and the group still touches all 24 header bytes exactly
    // once per cache line.
    uint hsel = hi ? 1u : 0u;
    uint w0 = planes[off + 0u + hsel];
    uint w1 = planes[off + 2u + hsel];
    uint w2 = planes[off + 4u + hsel];
"""

_LOAD_REDUNDANT = r"""
    uint p0l = planes[off + 0u], p0h = planes[off + 1u];
    uint p1l = planes[off + 2u], p1h = planes[off + 3u];
    uint p2l = planes[off + 4u], p2h = planes[off + 5u];
    uint w0 = hi ? p0h : p0l;
    uint w1 = hi ? p1h : p1l;
    uint w2 = hi ? p2h : p2l;
"""

_LOAD_BROADCAST = r"""
    uint q0l = 0u, q0h = 0u, q1l = 0u, q1h = 0u, q2l = 0u, q2h = 0u;
    if (lane == 0u) {
        q0l = planes[off + 0u]; q0h = planes[off + 1u];
        q1l = planes[off + 2u]; q1h = planes[off + 3u];
        q2l = planes[off + 4u]; q2h = planes[off + 5u];
    }
    uint p0l = simd_broadcast(q0l, 0u), p0h = simd_broadcast(q0h, 0u);
    uint p1l = simd_broadcast(q1l, 0u), p1h = simd_broadcast(q1h, 0u);
    uint p2l = simd_broadcast(q2l, 0u), p2h = simd_broadcast(q2h, 0u);
    uint w0 = hi ? p0h : p0l;
    uint w1 = hi ? p1h : p1l;
    uint w2 = hi ? p2h : p2l;
"""

#: Tile-header load strategies, measured on M2 Ultra (see the phase-5
#: receipt): ``half`` reads only the 32-bit word its lane needs, ``redundant``
#: reads all six words per lane and selects, ``broadcast`` has lane 0 read
#: them and ``simd_broadcast``s.  v3's decode-only kernel defaults to
#: ``broadcast``; inside the fused GEMV that is the SLOWEST of the three,
#: because a cross-lane op sits on the critical path of a loop that already
#: carries one (the escape-rank prefix sum).
LOAD_MODES = ("half", "redundant", "broadcast")

_LOAD_BLOCKS = {
    "half": _LOAD_HALF,
    "redundant": _LOAD_REDUNDANT,
    "broadcast": _LOAD_BROADCAST,
}

_DECODE_FN = r"""
constant int TBE_DELTA[8] = {0, 0, 1, 2, 3, 4, 5, 6};

// One SIMD group, one 64-element flat64 tile.  Lane `lane` returns the raw
// bf16 BIT PATTERNS of elements 2*lane and 2*lane+1 of tile `tile`.
// Identical semantics to tbe_container.decode_tbe:
//   code == 0            -> escape, exponent read from esc[] by popcount rank
//   code in [1, 6|7]     -> in-window, exponent = base + code - 1
//   mode W6Z, code == 7  -> exact zero
//   word = (sign << 15) | (exponent << 7) | mantissa
// Templated on the four buffer POINTER TYPES, not written against
// `const device T*`: MLX promotes a small input array to the `constant`
// address space, so a container with (say) a single escape byte hands this
// routine a `constant uchar*` while a large one hands it `device uchar*`.
// Address space is part of the type in Metal, so a non-template signature
// fails to compile on exactly the small containers -- deduce it instead.
template <typename PLANES_T, typename SMB_T, typename ESC_T, typename TB_T>
inline void tbe_decode_pair(
        PLANES_T planes, SMB_T smb, ESC_T esc, TB_T tile_base,
        uint tile, uint lane, int b, uint is_w6z, uint esc_len,
        thread ushort& word_a, thread ushort& word_b) {
    uint off = tile * 6u;
    bool hi = lane >= 16u;
%(load_block)s
    uint bit_a = 2u * lane;
    uint bit_b = bit_a + 1u;
    uint shA = hi ? (bit_a - 32u) : bit_a;
    uint shB = hi ? (bit_b - 32u) : bit_b;

    uint code_a = ((w0 >> shA) & 1u) | (((w1 >> shA) & 1u) << 1) | (((w2 >> shA) & 1u) << 2);
    uint code_b = ((w0 >> shB) & 1u) | (((w1 >> shB) & 1u) << 1) | (((w2 >> shB) & 1u) << 2);

    bool esc_a = (code_a == 0u);
    bool esc_b = (code_b == 0u);
    uint lane_esc = (esc_a ? 1u : 0u) + (esc_b ? 1u : 0u);
    // Lane order 0..31 -> element order (2*lane, 2*lane+1) is monotonic, so
    // this lane-ascending scan IS the tile's element-ascending escape rank.
    uint prefix = simd_prefix_exclusive_sum(lane_esc);
    uint rank_before = tile_base[tile] + prefix;
    uint rank_a = rank_before;
    uint rank_b = rank_before + (esc_a ? 1u : 0u);

    uint inwin_a = uint(b + TBE_DELTA[code_a]);
    uint inwin_b = uint(b + TBE_DELTA[code_b]);
    inwin_a = (is_w6z != 0u && code_a == 7u) ? 0u : inwin_a;
    inwin_b = (is_w6z != 0u && code_b == 7u) ? 0u : inwin_b;

    // ESCAPE READ: PREDICATED, NOT SPECULATIVE -- and that is a measured
    // reversal of v1/v2/v3, which all read esc[] unconditionally with a
    // clamped index and threw the value away on non-escape lanes.  That was
    // right for a decode-only kernel whose cost was dominated by the bf16
    // WRITE.  With the write fused away, two unconditional data-dependent
    // byte loads per lane per tile become the single largest term left:
    // ablated on M2 Ultra at 4096x2560, M=1 (see the phase-5 receipt's
    // `ablation` block), speculative 108.9 us vs predicated 102.3 us vs
    // removing the escape path entirely 81.2 us.  A uniform
    // `simd_any(esc_a || esc_b)` guard on top of the predicate was measured
    // and is WORSE (119.4 us): at a 3.1%% escape rate only ~13%% of tiles are
    // escape-free, so the extra cross-lane reduction costs more than the
    // loads it skips.  The clamp is kept as a bounds guard on the taken
    // branch, so an out-of-range rank still cannot read out of bounds.
    uint mexp_a = inwin_a;
    uint mexp_b = inwin_b;
    if (esc_a) { mexp_a = uint(esc[rank_a < esc_len ? rank_a : (esc_len - 1u)]); }
    if (esc_b) { mexp_b = uint(esc[rank_b < esc_len ? rank_b : (esc_len - 1u)]); }

    uint elem_a = tile * 64u + bit_a;
    // Address-space-agnostic 2-byte read of the adjacent pair.
    uint smb_a_ = uint(smb[elem_a]);
    uint smb_b_ = uint(smb[elem_a + 1u]);
    uint smb_a = smb_a_;
    uint smb_b = smb_b_;
    word_a = (ushort(smb_a & 0x80u) << 8) | (ushort(mexp_a & 0xFFu) << 7) | ushort(smb_a & 0x7Fu);
    word_b = (ushort(smb_b & 0x80u) << 8) | (ushort(mexp_b & 0xFFu) << 7) | ushort(smb_b & 0x7Fu);
}
"""


def _decode_header(load: str) -> str:
    try:
        block = _LOAD_BLOCKS[load]
    except KeyError:
        raise TBEFusedGemvError(
            f"unknown tile-header load mode {load!r}; expected one of {LOAD_MODES}"
        ) from None
    return _DECODE_FN % {"load_block": block}


# ---------------------------------------------------------------------------
# the fused GEMV kernel source
# ---------------------------------------------------------------------------
#: One decoded tile's contribution to the M accumulators.  ``JJ`` is the
#: tile index within the row (a C expression), emitted once per unroll slot.
_TILE_MAC = r"""
        {
            ushort wa_, wb_;
            tbe_decode_pair(planes, smb, esc, tile_base,
                            tile0 + (%(JJ)s), lane, b, is_w6z, esc_len, wa_, wb_);
            float fa_ = float(as_type<bfloat16_t>(wa_));
            float fb_ = float(as_type<bfloat16_t>(wb_));
            uint col_ = (%(JJ)s) * 64u + 2u * lane;
%(xmac)s
        }
"""

_X_MAC = r"""
            {
                ushort2 xv_ = *((const device ushort2*)(x + %(M)du * n_cols + col_));
                acc%(M)d += fa_ * float(as_type<bfloat16_t>(xv_.x))
                          + fb_ * float(as_type<bfloat16_t>(xv_.y));
            }
"""


def _emit_tile_mac(jj: str, batch: int) -> str:
    xmac = "".join(_X_MAC % {"M": m} for m in range(batch))
    return _TILE_MAC % {"JJ": jj, "xmac": xmac}


def _build_gemv_source(mode: str, tiles_per_group: int, simdgroups_per_tg: int,
                       batch: int) -> str:
    tpg = int(tiles_per_group)
    spt = int(simdgroups_per_tg)
    decls = "".join(f"    float acc{m} = 0.0f;\n" for m in range(batch))

    unrolled = "".join(_emit_tile_mac(f"j + {u}u", batch) for u in range(tpg))
    tail = _emit_tile_mac("j", batch)

    if mode == "row":
        prologue = r"""
    uint lane = thread_index_in_simdgroup;
    uint sgid = simdgroup_index_in_threadgroup;
    uint row = threadgroup_position_in_grid.x * %(spt)du + sgid;
    uint n_rows = n_rows_a[0];
    if (row >= n_rows) { return; }
    uint tpr = tiles_per_row[0];
    uint n_cols = tpr * 64u;
    int  b = base[0];
    uint is_w6z = w6z[0];
    uint esc_len = esc_len_arr[0];
    uint tile0 = row * tpr;
    uint j = 0u;
    uint jend = tpr;
""" % {"spt": spt}
        loop = r"""
    for (; j + %(tpg)du <= jend; j += %(tpg)du) {
%(unrolled)s
    }
    for (; j < jend; j += 1u) {
%(tail)s
    }
""" % {"tpg": tpg, "unrolled": unrolled, "tail": tail}
        epilogue = "".join(
            r"""
    {
        float s%(M)d = simd_sum(acc%(M)d);
        if (lane == 0u) { out[%(M)du * n_rows + row] = bfloat16_t(s%(M)d); }
    }
""" % {"M": m} for m in range(batch)
        )
        return prologue + decls + loop + epilogue

    if mode == "splitk":
        prologue = r"""
    uint lane = thread_index_in_simdgroup;
    uint sgid = simdgroup_index_in_threadgroup;
    uint row = threadgroup_position_in_grid.x;
    uint n_rows = n_rows_a[0];
    uint tpr = tiles_per_row[0];
    uint n_cols = tpr * 64u;
    int  b = base[0];
    uint is_w6z = w6z[0];
    uint esc_len = esc_len_arr[0];
    uint tile0 = row * tpr;
    threadgroup float part[%(spt)du * %(batch)du];
    uint j = sgid;
    uint jend = tpr;
""" % {"spt": spt, "batch": batch}
        # SPT-strided over tiles; TPG unrolled slots are SPT apart.
        unrolled_sk = "".join(
            _emit_tile_mac(f"j + {u * spt}u", batch) for u in range(tpg)
        )
        loop = r"""
    for (; j + %(stride)du <= jend; j += %(stride)du) {
%(unrolled)s
    }
    for (; j < jend; j += %(spt)du) {
%(tail)s
    }
""" % {"stride": tpg * spt, "spt": spt, "unrolled": unrolled_sk, "tail": tail}
        # simd_sum is a whole-group reduction: every lane must reach it, so
        # it is called unconditionally and only the STORE is lane-0 guarded.
        store = "".join(
            "    { float p%d = simd_sum(acc%d);"
            " if (lane == 0u) { part[sgid * %du + %du] = p%d; } }\n"
            % (m, m, batch, m, m) for m in range(batch)
        )
        reduce = r"""
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u && lane == 0u) {
""" + "".join(
            r"""
        {
            float s%(M)d = 0.0f;
            // FIXED group order: the reduction is deterministic run to run.
            for (uint g = 0u; g < %(spt)du; g++) { s%(M)d += part[g * %(batch)du + %(M)du]; }
            out[%(M)du * n_rows + row] = bfloat16_t(s%(M)d);
        }
""" % {"M": m, "spt": spt, "batch": batch} for m in range(batch)
        ) + "    }\n"
        return prologue + decls + loop + store + reduce

    raise TBEFusedGemvError(f"unknown fused mode {mode!r}")


# ---------------------------------------------------------------------------
# the verification probe -- same decode function, but it WRITES the weights
# ---------------------------------------------------------------------------
_PROBE_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint sgid = simdgroup_index_in_threadgroup;
    uint tile = threadgroup_position_in_grid.x * %(spt)du + sgid;
    uint n_tiles_v = n_tiles[0];
    if (tile >= n_tiles_v) { return; }
    uint n = n_elem[0];
    int  b = base[0];
    uint is_w6z = w6z[0];
    uint esc_len = esc_len_arr[0];

    ushort wa_, wb_;
    tbe_decode_pair(planes, smb, esc, tile_base, tile, lane, b, is_w6z,
                    esc_len, wa_, wb_);
    uint elem_a = tile * 64u + 2u * lane;
    if (elem_a + 1u < n) {
        *((device ushort2*)(out + elem_a)) = ushort2(wa_, wb_);
    } else {
        if (elem_a < n)      { out[elem_a]      = as_type<bfloat16_t>(wa_); }
        if (elem_a + 1u < n) { out[elem_a + 1u] = as_type<bfloat16_t>(wb_); }
    }
"""

#: ``(mode, tiles_per_group, simdgroups_per_threadgroup)`` grid.
_VARIANT_GRID = {}
for _mode in ("row", "splitk"):
    for _tpg in (1, 2, 4, 8):
        for _spt in (1, 2, 4, 8, 16, 32):
            _VARIANT_GRID[f"{_mode}_t{_tpg}_g{_spt}"] = (_mode, _tpg, _spt)
VARIANTS = tuple(_VARIANT_GRID.keys())

#: Serving default.  ``row`` needs no cross-group reduction and every Qwen3-4B
#: linear has O >= 1024, i.e. >= 1024 SIMD groups in flight.
DEFAULT_VARIANT = "splitk_t1_g8"

#: Tile-header load strategy default.  See LOAD_MODES.
DEFAULT_LOAD = "half"

#: Largest M the fused kernel will compile for.  Beyond this the decode is
#: amortised enough that a tiled GEMM (decode-then-matmul) is the right shape.
MAX_FUSED_BATCH = 8

_GEMV_CACHE: dict = {}
_PROBE_CACHE: dict = {}


def _gemv_kernel(mode: str, tpg: int, spt: int, batch: int, load: str):
    key = (mode, tpg, spt, batch, load)
    if key not in _GEMV_CACHE:
        tag = f"{mode}_t{tpg}_g{spt}_m{batch}_{load}"
        _GEMV_CACHE[key] = mx.fast.metal_kernel(
            name=f"tbe_gemv_fused_{tag}",
            input_names=[
                "planes", "smb", "esc", "tile_base", "x",
                "base", "w6z", "n_rows_a", "tiles_per_row", "esc_len_arr",
            ],
            output_names=["out"],
            source=_build_gemv_source(mode, tpg, spt, batch),
            header=_decode_header(load),
        )
    return _GEMV_CACHE[key]


def _probe_kernel(spt: int, load: str):
    key = (spt, load)
    if key not in _PROBE_CACHE:
        tag = f"g{spt}_{load}"
        _PROBE_CACHE[key] = mx.fast.metal_kernel(
            name=f"tbe_fused_decode_probe_{tag}",
            input_names=[
                "planes", "smb", "esc", "tile_base",
                "base", "w6z", "n_elem", "n_tiles", "esc_len_arr",
            ],
            output_names=["out"],
            source=_PROBE_SOURCE % {"spt": int(spt)},
            header=_decode_header(load),
        )
    return _PROBE_CACHE[key]


# ---------------------------------------------------------------------------
# uniforms, cached on the container (same discipline as tbe_decode_mlx_v4)
# ---------------------------------------------------------------------------
_UNIFORMS_ATTR = "_glc_tbe_fused_uniforms"


def _uniforms(c: TBEDeviceMLX):
    got = getattr(c, _UNIFORMS_ATTR, None)
    if got is None:
        rows, cols = int(c.shape[0]), int(c.shape[1])
        got = (
            mx.array([int(c.base)], dtype=mx.int32),
            mx.array([1 if int(c.mode) == MODE_W6Z else 0], dtype=mx.uint32),
            mx.array([rows], dtype=mx.uint32),
            mx.array([cols // TILE], dtype=mx.uint32),
            mx.array([int(c.esc.size)], dtype=mx.uint32),
        )
        try:
            setattr(c, _UNIFORMS_ATTR, got)
        except Exception:
            pass
    return got


def fused_tileable(c: TBEDeviceMLX) -> bool:
    """True iff flat64 tiles of ``c`` never straddle a row of ``W``."""
    return int(c.shape[1]) % TILE == 0


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def tbe_gemv_fused(
    c: TBEDeviceMLX,
    x: mx.array,
    variant: str = DEFAULT_VARIANT,
    load: str = DEFAULT_LOAD,
    eval_now: bool = False,
) -> mx.array:
    """``y = x @ W.T`` straight from the TBE container -- no bf16 ever stored.

    ``x`` is ``[M, I]`` bf16 (or anything castable); the result is ``[M, O]``
    bf16.  ``M`` is baked into the kernel as a compile-time constant, so each
    distinct ``M`` compiles once and is cached.  The design target is
    ``M = 1``; small ``M`` (2-8) generalises without a redesign because the
    decoded tile is already in registers and is simply reused across the ``M``
    accumulators, but ``M`` beyond that wants a tiled GEMM, not this kernel.

    Refuses (never falls back) when the container cannot be tiled per row.
    """
    if variant not in _VARIANT_GRID:
        raise TBEFusedGemvError(
            f"unknown variant {variant!r}; expected one of {VARIANTS}"
        )
    mode, tpg, spt = _VARIANT_GRID[variant]

    rows, cols = int(c.shape[0]), int(c.shape[1])
    if cols % TILE != 0:
        raise TBEFusedGemvError(
            f"fused decode-in-GEMV needs in_features divisible by {TILE} so a "
            f"flat64 tile never straddles a row of W; got shape {c.shape} "
            f"(in_features={cols}). Decode-then-matmul "
            "(tbe_linear_mlx_v4) serves this shape; this path refuses rather "
            "than falling back, so a receipt cannot say 'fused' and measure "
            "something else."
        )
    if int(c.tiles) * TILE != rows * cols:
        raise TBEFusedGemvError(
            f"container tile count {c.tiles} does not cover {rows}x{cols} "
            "exactly; the fused kernel indexes tiles by row and cannot use a "
            "padded final tile"
        )

    if x.ndim == 1:
        x2 = x.reshape(1, -1)
        squeeze = True
    else:
        x2 = x.reshape(-1, x.shape[-1])
        squeeze = False
    batch = int(x2.shape[0])
    if int(x2.shape[-1]) != cols:
        raise TBEFusedGemvError(
            f"activation has {int(x2.shape[-1])} features, weight expects {cols}"
        )
    if batch < 1 or batch > MAX_FUSED_BATCH:
        raise TBEFusedGemvError(
            f"fused GEMV is built for M in [1, {MAX_FUSED_BATCH}]; got M={batch}. "
            "Larger M is a tiled GEMM problem, not this kernel."
        )
    xb = x2.astype(mx.bfloat16)

    if mode == "row":
        n_tg = max(1, -(-rows // spt))
    else:
        n_tg = rows
    tg_width = spt * 32

    kernel = _gemv_kernel(mode, tpg, spt, batch, load)
    out = kernel(
        inputs=[c.planes, c.smb, c.esc, c.tile_base, xb, *_uniforms(c)],
        template=[],
        grid=(n_tg * tg_width, 1, 1),
        threadgroup=(tg_width, 1, 1),
        output_shapes=[(batch * rows,)],
        output_dtypes=[mx.bfloat16],
    )[0].reshape(batch, rows)
    if squeeze:
        out = out.reshape(rows)
    if eval_now:
        mx.eval(out)
    return out


def fused_decode_probe(
    c: TBEDeviceMLX,
    simdgroups_per_tg: int = 4,
    load: str = DEFAULT_LOAD,
    eval_now: bool = True,
) -> mx.array:
    """Decode ``c`` **through the fused GEMV's own decode routine**.

    This is the operational definition of "bit-exact weights" for a path that
    materialises no weights: the fused kernel and this probe are compiled from
    the same ``tbe_decode_pair`` text, so an element-for-element match between
    this and ``tbe_container.decode_tbe`` pins the fused GEMV's view of every
    weight, tile by tile, without the GEMV ever writing one.
    """
    n = c.numel
    if n == 0:
        return mx.zeros(c.shape, dtype=mx.bfloat16)
    n_tiles = max(1, int(c.tiles))
    spt = int(simdgroups_per_tg)
    n_tg = max(1, -(-n_tiles // spt))
    tg_width = spt * 32

    kernel = _probe_kernel(spt, load)
    out = kernel(
        inputs=[
            c.planes, c.smb, c.esc, c.tile_base,
            mx.array([int(c.base)], dtype=mx.int32),
            mx.array([1 if int(c.mode) == MODE_W6Z else 0], dtype=mx.uint32),
            mx.array([n], dtype=mx.uint32),
            mx.array([int(c.tiles)], dtype=mx.uint32),
            mx.array([int(c.esc.size)], dtype=mx.uint32),
        ],
        template=[],
        grid=(n_tg * tg_width, 1, 1),
        threadgroup=(tg_width, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.bfloat16],
    )[0].reshape(c.shape)
    if eval_now:
        mx.eval(out)
    return out


# ---------------------------------------------------------------------------
# bit-exactness accounting for (a) -- ulp distance against the reference path
# ---------------------------------------------------------------------------
def ulp_report(got: mx.array, ref: mx.array) -> dict:
    """Max bf16-ulp distance and non-identical count between two bf16 arrays.

    bf16 has a monotone bit pattern within a sign, so the ulp distance is the
    difference of the *sign-magnitude-to-ordinal* mapped bit patterns -- the
    same trick as a float ulp compare, at 16 bits.  NaN/Inf inputs are not
    expected on this path and are reported as a separate count rather than
    folded into the distance.
    """
    import numpy as np

    if got.shape != ref.shape:
        raise TBEFusedGemvError(f"shape mismatch {got.shape} vs {ref.shape}")
    a = np.array(mx.view(got.reshape(-1), mx.uint16), copy=True).astype(np.int64)
    b = np.array(mx.view(ref.reshape(-1), mx.uint16), copy=True).astype(np.int64)

    def ordinal(u: "np.ndarray") -> "np.ndarray":
        # 0x8000 (-0.0) maps onto 0 (+0.0); negatives mirror below zero.
        neg = (u & 0x8000) != 0
        mag = u & 0x7FFF
        return np.where(neg, -mag, mag)

    oa, ob = ordinal(a), ordinal(b)
    d = np.abs(oa - ob)
    exp_a = (a >> 7) & 0xFF
    exp_b = (b >> 7) & 0xFF
    nonfinite = int(((exp_a == 0xFF) | (exp_b == 0xFF)).sum())

    fa = ((a.astype(np.uint32) & 0xFFFF) << 16).astype(np.uint32).view(np.float32)
    fb = ((b.astype(np.uint32) & 0xFFFF) << 16).astype(np.uint32).view(np.float32)
    abs_delta = np.abs(fa.astype(np.float64) - fb.astype(np.float64))
    scale = float(np.abs(fb.astype(np.float64)).max()) if d.size else 0.0
    # An element whose own magnitude is far below the largest output is pure
    # cancellation residue: its ULP distance is large while its ABSOLUTE
    # distance is negligible, and neither summation order is "right" there.
    # Report the ULP bound on the elements that carry signal separately from
    # the raw max, rather than picking whichever number reads better.
    signal = np.abs(fb.astype(np.float64)) >= (scale / 256.0)
    return {
        "n_elements": int(d.size),
        "n_identical": int((d == 0).sum()),
        "n_non_identical": int((d != 0).sum()),
        "max_ulp": int(d.max()) if d.size else 0,
        "max_ulp_above_cancellation": int(d[signal].max()) if signal.any() else 0,
        "n_below_cancellation_threshold": int((~signal).sum()),
        "mean_ulp": float(d.mean()) if d.size else 0.0,
        "max_abs_delta": float(abs_delta.max()) if d.size else 0.0,
        "max_abs_reference": scale,
        "frac_non_identical": float((d != 0).mean()) if d.size else 0.0,
        "n_non_finite": nonfinite,
        "exact": bool(d.size and int(d.max()) == 0),
    }


# ---------------------------------------------------------------------------
# the serving module
# ---------------------------------------------------------------------------
class TBELinearFusedMLX:
    """Linear layer served by the fused decode-in-GEMV at small M.

    Same constructor signature as ``TBELinearMLX`` / ``V4`` so it drops into
    ``tbe_mlx_model.DECODERS``.  For ``M > MAX_FUSED_BATCH`` (prefill) it
    delegates to the phase-4 decode-then-matmul, which is the right shape
    there: the decode cost is amortised over M rows and the fused kernel's
    per-tile decode would be repeated for every one of them.  That delegation
    is explicit and reported (:attr:`n_fused_calls` / :attr:`n_fallback_calls`),
    not silent.
    """

    def __init__(
        self,
        container: TBEDeviceMLX,
        bias: Optional[mx.array] = None,
        variant: str = DEFAULT_VARIANT,
        load: str = DEFAULT_LOAD,
    ) -> None:
        if not fused_tileable(container):
            raise TBEFusedGemvError(
                f"container {container.shape} has in_features not divisible by "
                f"{TILE}; the fused GEMV refuses it"
            )
        self.container = container
        self.bias = bias
        self.variant = variant
        self.load = load
        self.out_features, self.in_features = container.shape
        self.n_fused_calls = 0
        self.n_fallback_calls = 0

    @property
    def resident_bytes(self) -> int:
        return self.container.resident_bytes

    @property
    def dense_bytes(self) -> int:
        return self.container.dense_bytes

    def decode(self, eval_now: bool = False) -> mx.array:
        from .tbe_decode_mlx_v4 import decode_tbe_metal_v4

        return decode_tbe_metal_v4(self.container, eval_now=eval_now)

    def __call__(self, x: mx.array) -> mx.array:
        flat = x.reshape(-1, x.shape[-1])
        if int(flat.shape[-1]) != self.in_features:
            raise TBEFusedGemvError(
                f"input has {int(flat.shape[-1])} features, weight expects "
                f"{self.in_features}"
            )
        m = int(flat.shape[0])
        if m <= MAX_FUSED_BATCH:
            self.n_fused_calls += 1
            y = tbe_gemv_fused(
                self.container, flat, variant=self.variant,
                load=self.load,
            )
        else:
            self.n_fallback_calls += 1
            y = mx.matmul(flat.astype(mx.bfloat16), self.decode().T)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*x.shape[:-1], self.out_features)


__all__ = [
    "DEFAULT_VARIANT",
    "DEFAULT_LOAD",
    "LOAD_MODES",
    "MAX_FUSED_BATCH",
    "METAL_MEMORY_LIMIT_BYTES",
    "TBEDeviceMLX",
    "TBEFusedGemvError",
    "TBELinearFusedMLX",
    "VARIANTS",
    "bits_view",
    "fused_decode_probe",
    "fused_tileable",
    "mlx_metal_available",
    "tbe_gemv_fused",
    "ulp_report",
    "upload_tbe_mlx",
]
