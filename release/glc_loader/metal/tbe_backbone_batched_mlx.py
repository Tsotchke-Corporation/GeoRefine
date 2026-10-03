"""Phase 8 -- ONE Metal launch over the DIFFERENT-shaped linears of one backbone layer.

WHY THIS EXISTS
---------------
Phase 7 (``docs/research/MOE_BATCHED_KERNEL_20260904.md``) removed the dispatch +
occupancy tax from the routed expert block -- 30 launches -> 3, 934.0 -> 548.6
us/layer, 1.703x -- and by doing so made the BACKBONE the larger half of the
flagship token: 33.16 of 59.54 ms, 56%.  That backbone is 773 separate GEMV
launches per token (``docs/research/FLASHNEXT_COMPUTE_MODEL_20260904.md`` sec.1a),
five per transformer layer and three per hyper-connection block, each one a
per-linear boundary that imposes a launch tax AND a small-grid occupancy floor --
the two terms phase 7 sec.4 separated.

The phase-7 bundle cannot be pointed at a backbone layer: it is SAME-SHAPE by
construction.  ``build_expert_bundle`` refuses a mixed-shape set precisely because
planes / smb / tile_base carry no per-member offset table, and a MoE layer never
needs one.  A backbone layer is nothing but mixed shapes: GDN's
``{qkv 10240x2560, z 6144x2560, a 48x2560, b 48x2560}`` all read the same
post-norm activation and share nothing else.

This module is that bundle made RAGGED.  Every per-tensor quantity the phase-5
kernel held as a scalar uniform -- tile offset, escape slice, base, W6Z mode,
tiles_per_row, activation offset, row_start -- moves into a per-member ``meta``
row, and a ``row_member`` lookup maps a global grid row to the member that owns
it.  That is the whole of the new mechanism.

WHAT IS AND IS NOT NEW HERE
---------------------------
The decode routine is imported, twice removed and never re-typed:
``_decode_header`` (and through it ``_DECODE_FN``, ``TBE_DELTA``, the W6Z select,
``simd_prefix_exclusive_sum`` escape ranking and bf16 word assembly) comes from
``tbe_gemv_fused_mlx``, and the two mechanical substitutions that make it
addressable per member (``esc`` -> ``esc_e``, ``x`` -> ``xp``) are NOT re-derived:
this module imports ``tbe_moe_batched_mlx._emit_tile_mac``, which already applies
them behind assertions on the phase-5 call-site counts.  A refactor of either base
file therefore RAISES here rather than silently emitting a kernel built against a
stale template.  The only Metal text this module authors is the ragged grid
mapping, the per-member uniform fetch and the epilogue.

BIT-IDENTITY
------------
For a given (member m, row r) the batched kernel walks the SAME tiles in the SAME
order with the SAME per-lane decode and the SAME ``simd_sum`` as
``tbe_gemv_fused(containers[m], x_m, variant=V)``, differing only in the tile INDEX
(``tile_offset[m]`` is added) and in where the result is stored.  The float
accumulation order is identical, so the output is **bit-identical**, not merely
within 1 ulp.  There is no combine epilogue in this module and therefore no
exception to that statement -- unlike the MoE path, a backbone group's members
produce DIFFERENT outputs that are consumed separately, so there is nothing to sum.

BUNDLE LAYOUT (members may differ in BOTH dimensions)
-----------------------------------------------------
  planes      uint32 [sum_m T_m * 6]   member m's planes at [6*tile_off_m, ...)
  smb         uint8  [sum_m T_m * 64]
  esc         uint8  [sum_m E_m]       ragged
  tile_base   uint32 [sum_m T_m]       per-member escape ranks, unchanged
  meta        int32  [N * 8]           per member:
                                       (tile_off, esc_off, esc_len, base,
                                        is_w6z, tiles_per_row, x_off, row_start)
  row_member  uint32 [sum_m O_m]       global grid row -> member index
  dims        uint32 [8]               (n_out_total, n_members, 0 ...)

``row_member`` is the price of raggedness and it is small: the widest group in the
flagship backbone is 16,480 rows, i.e. 66 KB read once per launch and resident in
cache for the whole grid.  It replaces the phase-7 kernel's ``e = gr / n_rows``
runtime divide, which cannot work when members have different row counts.

REFUSALS
--------
Every refusal is a :class:`TBEBackboneBatchedError` and there is NO fallback to the
per-linear loop.  A receipt that says "batched" must have measured a batched
launch; silently degrading to N launches is exactly how a dispatch measurement
stops measuring dispatch.  Non-tileable shapes (in_features not divisible by 64)
are refused rather than padded, because a padded final tile changes the tile-index
arithmetic the whole kernel rests on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

try:
    import mlx.core as mx
except Exception:  # pragma: no cover - exercised only on a non-MLX host
    mx = None  # type: ignore

from .tbe_decode_mlx import TBEDeviceMLX
from .tbe_gemv_fused_mlx import (
    MAX_FUSED_BATCH,
    TILE,
    _decode_header,
    _VARIANT_GRID,
    DEFAULT_LOAD,
    VARIANTS,
)
# The two per-member substitutions (esc -> esc_e, x -> xp) are the phase-7
# module's, applied there behind assertions on the phase-5 call-site counts.
# Importing the emitter is the whole of the reuse: this file never re-types the
# decode, the MAC, or the substitutions.
from .tbe_moe_batched_mlx import _emit_tile_mac

#: Number of int32 fields per member in ``meta``.
META_STRIDE = 8


class TBEBackboneBatchedError(RuntimeError):
    """The ragged batched path refuses this bundle or this call."""


# ---------------------------------------------------------------------------
# the ragged kernel source
# ---------------------------------------------------------------------------
#: Read once per SIMD group (row mode) or per threadgroup (splitk mode).  Every
#: name this block binds is one the imported phase-5 MAC template expects to find
#: in scope: planes / smb / tile_base come in as buffers, and b, is_w6z, esc_len,
#: esc_e, xp, n_cols, tile0 are bound here from the member's meta row.
_PER_MEMBER_PROLOGUE = r"""
    uint mi  = row_member[gr];
    uint mo  = %(stride)du * mi;
    uint tile_off = uint(meta[mo + 0u]);
    uint esc_off  = uint(meta[mo + 1u]);
    uint esc_len  = uint(meta[mo + 2u]);
    int  b        = meta[mo + 3u];
    uint is_w6z   = uint(meta[mo + 4u]);
    uint tpr      = uint(meta[mo + 5u]);
    uint x_off    = uint(meta[mo + 6u]);
    uint row      = gr - uint(meta[mo + 7u]);
    uint n_cols   = tpr * 64u;
    // Address-space-preserving pointer shifts.  MLX may promote a small buffer
    // to `constant`, and address space is part of the type in Metal, so `auto`
    // is load-bearing: naming `device const uchar*` here would fail to compile
    // on exactly the small-escape containers this kernel is full of.
    auto esc_e = esc + esc_off;
    auto xp    = x + x_off;
    uint tile0 = tile_off + row * tpr;
""" % {"stride": META_STRIDE}


def _build_backbone_source(mode: str, tiles_per_group: int,
                           simdgroups_per_tg: int, batch: int) -> str:
    tpg = int(tiles_per_group)
    spt = int(simdgroups_per_tg)
    decls = "".join(f"    float acc{m} = 0.0f;\n" for m in range(batch))

    if mode == "row":
        prologue = r"""
    uint lane = thread_index_in_simdgroup;
    uint sgid = simdgroup_index_in_threadgroup;
    uint gr = threadgroup_position_in_grid.x * %(spt)du + sgid;
    uint n_out = dims[0];
    if (gr >= n_out) { return; }
""" % {"spt": spt} + _PER_MEMBER_PROLOGUE
        loop = r"""
    uint j = 0u;
    uint jend = tpr;
    for (; j + %(tpg)du <= jend; j += %(tpg)du) {
%(unrolled)s
    }
    for (; j < jend; j += 1u) {
%(tail)s
    }
""" % {"tpg": tpg,
       "unrolled": "".join(_emit_tile_mac(f"j + {u}u", batch) for u in range(tpg)),
       "tail": _emit_tile_mac("j", batch)}
        epilogue = "".join(
            r"""
    {
        float s%(M)d = simd_sum(acc%(M)d);
        if (lane == 0u) { out[%(M)du * n_out + gr] = bfloat16_t(s%(M)d); }
    }
""" % {"M": m} for m in range(batch)
        )
        return prologue + decls + loop + epilogue

    if mode == "splitk":
        prologue = r"""
    uint lane = thread_index_in_simdgroup;
    uint sgid = simdgroup_index_in_threadgroup;
    uint gr = threadgroup_position_in_grid.x;
    uint n_out = dims[0];
    threadgroup float part[%(spt)du * %(batch)du];
""" % {"spt": spt, "batch": batch} + _PER_MEMBER_PROLOGUE
        loop = r"""
    uint j = sgid;
    uint jend = tpr;
    for (; j + %(stride)du <= jend; j += %(stride)du) {
%(unrolled)s
    }
    for (; j < jend; j += %(spt)du) {
%(tail)s
    }
""" % {"stride": tpg * spt, "spt": spt,
       "unrolled": "".join(
           _emit_tile_mac(f"j + {u * spt}u", batch) for u in range(tpg)),
       "tail": _emit_tile_mac("j", batch)}
        store = "".join(
            "    { float p%d = simd_sum(acc%d);"
            " if (lane == 0u) { part[sgid * %du + %du] = p%d; } }\n"
            % (m, m, batch, m, m) for m in range(batch))
        reduce = r"""
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u && lane == 0u) {
""" + "".join(
            r"""
        {
            float s%(M)d = 0.0f;
            for (uint g = 0u; g < %(spt)du; g++) { s%(M)d += part[g * %(batch)du + %(M)du]; }
            out[%(M)du * n_out + gr] = bfloat16_t(s%(M)d);
        }
""" % {"M": m, "spt": spt, "batch": batch} for m in range(batch)
        ) + "    }\n"
        return prologue + decls + loop + store + reduce

    raise TBEBackboneBatchedError(f"unknown batched mode {mode!r}")


_KERNEL_CACHE: dict = {}


def _backbone_kernel(mode: str, tpg: int, spt: int, batch: int, load: str):
    key = (mode, tpg, spt, batch, load)
    if key not in _KERNEL_CACHE:
        tag = f"{mode}_t{tpg}_g{spt}_m{batch}_{load}"
        _KERNEL_CACHE[key] = mx.fast.metal_kernel(
            name=f"tbe_backbone_batched_{tag}",
            input_names=["planes", "smb", "esc", "tile_base", "x", "meta",
                         "row_member", "dims"],
            output_names=["out"],
            source=_build_backbone_source(mode, tpg, spt, batch),
            header=_decode_header(load),
        )
    return _KERNEL_CACHE[key]


# ---------------------------------------------------------------------------
# the ragged bundle
# ---------------------------------------------------------------------------
@dataclass
class TBEBackboneBundle:
    """N differently-shaped TBE containers concatenated for one launch.

    Built ONCE at load time.  It must never be built inside a timed region: the
    dense arm's row-concatenated weight matrix is not built inside one either,
    and timing one side's preparation and not the other's is how a dispatch
    comparison stops being one.
    """

    planes: "mx.array"       # uint32 [sum_m T_m * 6]
    smb: "mx.array"          # uint8  [sum_m T_m * 64]
    esc: "mx.array"          # uint8  [sum_m E_m]
    tile_base: "mx.array"    # uint32 [sum_m T_m]
    meta: "mx.array"         # int32  [N * META_STRIDE]
    row_member: "mx.array"   # uint32 [sum_m O_m]
    shapes: List[tuple]      # per member (O_m, I_m)
    x_slot: List[int]        # per member: which activation block it reads
    row_start: List[int]     # per member: first global grid row
    n_out: int               # sum_m O_m
    slot_cols: List[int]     # in_features of each activation slot
    #: Host-side meta with ``x_off`` still unset.  ``x_off`` depends only on M
    #: and the slot layout, so the device meta and dims are built ONCE per M and
    #: cached -- never inside the timed region, where an mx.array upload or an
    #: mx.concatenate would add launches to the arm being advocated.
    meta_np: "np.ndarray" = field(repr=False, default=None)
    _uniform_cache: dict = field(repr=False, default_factory=dict)

    def uniforms(self, batch: int, offs: Sequence[int]):
        key = (int(batch),) + tuple(int(o) for o in offs)
        hit = self._uniform_cache.get(key)
        if hit is None:
            m = self.meta_np.copy()
            for i, s in enumerate(self.x_slot):
                m[i, 6] = int(offs[s])
            dev_meta = mx.array(m.reshape(-1))
            dev_dims = mx.array(np.array(
                [self.n_out, self.n_members, 0, 0, 0, 0, 0, 0], dtype=np.uint32))
            mx.eval(dev_meta, dev_dims)
            hit = (dev_meta, dev_dims)
            self._uniform_cache[key] = hit
        return hit

    @property
    def n_members(self) -> int:
        return len(self.shapes)

    @property
    def numel(self) -> int:
        return sum(int(o) * int(i) for o, i in self.shapes)

    @property
    def resident_bytes(self) -> int:
        def _b(a) -> int:
            return int(a.size) * int(a.itemsize)
        return (_b(self.planes) + _b(self.smb) + _b(self.esc)
                + _b(self.tile_base) + _b(self.meta) + _b(self.row_member))

    @property
    def dense_bytes(self) -> int:
        return self.numel * 2

    def ratio(self) -> float:
        return self.dense_bytes / max(1, self.resident_bytes)

    def slice_of(self, m: int) -> slice:
        """Where member ``m``'s outputs live in the ``[M, n_out]`` result."""
        o = int(self.shapes[m][0])
        s = int(self.row_start[m])
        return slice(s, s + o)


def build_backbone_bundle(containers: Sequence[TBEDeviceMLX],
                          x_slot: Sequence[int] | None = None
                          ) -> TBEBackboneBundle:
    """Concatenate N flat64 containers of ANY shapes into one launchable bundle.

    ``x_slot[m]`` names which activation block member ``m`` reads.  The default is
    all zeros -- every member reads the same vector, which is the case that makes
    a backbone layer batchable at all (GDN's q/k/v/z/a/b and attention's q/k/v all
    consume one post-norm hidden).  Members sharing a slot MUST share
    ``in_features``; that is checked, not assumed.
    """
    if mx is None:  # pragma: no cover
        raise TBEBackboneBatchedError("MLX is not importable on this host")
    cs = list(containers)
    if not cs:
        raise TBEBackboneBatchedError("a backbone bundle needs at least one container")
    slots = [0] * len(cs) if x_slot is None else [int(s) for s in x_slot]
    if len(slots) != len(cs):
        raise TBEBackboneBatchedError(
            f"x_slot has {len(slots)} entries for {len(cs)} containers")
    if min(slots) < 0:
        raise TBEBackboneBatchedError("x_slot entries must be non-negative")

    slot_cols: dict = {}
    tile_off = 0
    esc_off = 0
    row_start = 0
    shapes: List[tuple] = []
    starts: List[int] = []
    meta = np.zeros((len(cs), META_STRIDE), dtype=np.int32)
    row_member = []

    for i, c in enumerate(cs):
        rows, cols = int(c.shape[0]), int(c.shape[1])
        if cols % TILE != 0:
            raise TBEBackboneBatchedError(
                f"member {i} has in_features {cols}, which is not divisible by "
                f"{TILE}; a flat64 tile would straddle a row of W. This path "
                "refuses rather than padding or falling back.")
        if rows < 1:
            raise TBEBackboneBatchedError(
                f"member {i} has {rows} output rows")
        tpr = cols // TILE
        if int(c.tiles) != rows * tpr:
            raise TBEBackboneBatchedError(
                f"member {i} has {c.tiles} tiles, {rows}x{cols} needs exactly "
                f"{rows * tpr}; the batched kernel indexes tiles by "
                "(member, row) and cannot use a padded final tile")
        s = slots[i]
        if s in slot_cols and slot_cols[s] != cols:
            raise TBEBackboneBatchedError(
                f"members sharing activation slot {s} disagree on in_features: "
                f"{slot_cols[s]} vs {cols}. A slot is ONE activation block; two "
                "widths cannot share it.")
        slot_cols[s] = cols

        meta[i, 0] = tile_off
        meta[i, 1] = esc_off
        meta[i, 2] = int(c.esc.size)
        meta[i, 3] = int(c.base)
        meta[i, 4] = 1 if int(c.mode) != 0 else 0
        meta[i, 5] = tpr
        meta[i, 6] = 0          # filled at call time: depends on M
        meta[i, 7] = row_start
        shapes.append((rows, cols))
        starts.append(row_start)
        row_member.append(np.full(rows, i, dtype=np.uint32))

        tile_off += rows * tpr
        esc_off += int(c.esc.size)
        row_start += rows

    missing = sorted(set(range(max(slots) + 1)) - set(slot_cols))
    if missing:
        raise TBEBackboneBatchedError(
            f"activation slots {missing} are named by no member; slots must be "
            "dense from 0 so the caller's x list and the bundle agree")

    return TBEBackboneBundle(
        planes=mx.concatenate([c.planes for c in cs], axis=0),
        smb=mx.concatenate([c.smb for c in cs], axis=0),
        esc=mx.concatenate([c.esc for c in cs], axis=0),
        tile_base=mx.concatenate([c.tile_base for c in cs], axis=0),
        meta=mx.array(meta.reshape(-1)),
        row_member=mx.array(np.concatenate(row_member)),
        shapes=shapes,
        x_slot=slots,
        row_start=starts,
        n_out=row_start,
        slot_cols=[slot_cols[s] for s in range(max(slots) + 1)],
        meta_np=meta,
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def tbe_backbone_batched_gemv(
    bundle: TBEBackboneBundle,
    x,
    *,
    variant: str = "row_t1_g4",
    load: str = DEFAULT_LOAD,
    eval_now: bool = False,
):
    """One launch: ``y_m = x[slot_m] @ W_m.T`` for every member of ``bundle``.

    ``x`` is one ``[M, I]`` array when the bundle has a single activation slot,
    or a list of one ``[M, I_s]`` array per slot.  Returns ``[M, n_out]``;
    ``out[:, bundle.slice_of(m)]`` is exactly what
    ``tbe_gemv_fused(containers[m], x[slot_m], variant=variant)`` returns, bit for
    bit.

    A multi-slot call concatenates the activation blocks, which costs one extra
    launch; :func:`launches_for_call` reports it so a receipt can never
    under-count.  Every group measured in the phase-8 receipt is single-slot,
    because that is what makes a backbone layer batchable in the first place.
    """
    if variant not in _VARIANT_GRID:
        raise TBEBackboneBatchedError(
            f"unknown variant {variant!r}; expected one of {VARIANTS}")
    mode, tpg, spt = _VARIANT_GRID[variant]
    n_slots = len(bundle.slot_cols)

    xs = [x] if not isinstance(x, (list, tuple)) else list(x)
    if len(xs) != n_slots:
        raise TBEBackboneBatchedError(
            f"bundle has {n_slots} activation slot(s); got {len(xs)} array(s)")

    blocks = []
    batch = None
    for s, xv in enumerate(xs):
        x2 = xv.reshape(1, -1) if xv.ndim == 1 else xv.reshape(-1, xv.shape[-1])
        if int(x2.shape[-1]) != bundle.slot_cols[s]:
            raise TBEBackboneBatchedError(
                f"activation slot {s} has {int(x2.shape[-1])} features, its "
                f"members expect {bundle.slot_cols[s]}")
        if batch is None:
            batch = int(x2.shape[0])
        elif int(x2.shape[0]) != batch:
            raise TBEBackboneBatchedError(
                f"activation slot {s} carries M={int(x2.shape[0])}, slot 0 "
                f"carries M={batch}; one launch is one M")
        blocks.append(x2.astype(mx.bfloat16))
    if batch is None or batch < 1 or batch > MAX_FUSED_BATCH:
        raise TBEBackboneBatchedError(
            f"the batched fused GEMV is built for M in [1, {MAX_FUSED_BATCH}]; "
            f"got M={batch}. Larger M is a tiled GEMM problem, not this kernel.")

    # Per-slot element offset into the flat activation buffer.  Member m reads
    # xp + M*n_cols_m + col_, so a slot's block must be laid out [M, I_s]
    # contiguously -- which is what a reshape(-1) of an [M, I_s] array is.
    offs, acc = [], 0
    for s in range(n_slots):
        offs.append(acc)
        acc += batch * bundle.slot_cols[s]
    if n_slots == 1:
        xflat = blocks[0].reshape(-1)
    else:
        xflat = mx.concatenate([b.reshape(-1) for b in blocks], axis=0)

    meta, dims = bundle.uniforms(batch, offs)

    if mode == "row":
        n_tg = max(1, -(-bundle.n_out // spt))
    else:
        n_tg = bundle.n_out
    tg_width = spt * 32

    kernel = _backbone_kernel(mode, tpg, spt, batch, load)
    out = kernel(
        inputs=[bundle.planes, bundle.smb, bundle.esc, bundle.tile_base,
                xflat, meta, bundle.row_member, dims],
        template=[],
        grid=(n_tg * tg_width, 1, 1),
        threadgroup=(tg_width, 1, 1),
        output_shapes=[(batch * bundle.n_out,)],
        output_dtypes=[mx.bfloat16],
    )[0].reshape(batch, bundle.n_out)
    if eval_now:
        mx.eval(out)
    return out


def launches_for_call(bundle: TBEBackboneBundle) -> int:
    """Launches one :func:`tbe_backbone_batched_gemv` call issues.

    One GEMV kernel, plus one concatenate when the bundle has more than one
    activation slot.  Reported rather than assumed so a receipt cannot
    under-count the arm it is advocating.
    """
    return 1 + (1 if len(bundle.slot_cols) > 1 else 0)


__all__ = [
    "META_STRIDE",
    "TBEBackboneBatchedError",
    "TBEBackboneBundle",
    "build_backbone_bundle",
    "tbe_backbone_batched_gemv",
    "launches_for_call",
]
