"""GLC-TBE flat64 decode, Metal (MLX) -- Phase 3: one SIMD group per tile,
cross-lane escape rank via ``simd_prefix_exclusive_sum``.

Phase 2 (``tbe_decode_mlx_v2.py``) moved from one thread per element (phase
1, 43 GB/s) to one thread per TILE (55 GB/s) and diagnosed the ceiling as
"ALU-bound at 55 GB/s".  That diagnosis is wrong.  At 55 GB/s of bf16 the GPU
produces 27.5 G elements/s; an M2 Ultra GPU (76 cores x 4 SIMD units x 32
lanes, ~1.4 GHz, ~13.6 M SIMD-cycles/s of *32-wide* issue) would have to
spend on the order of 470 scalar instructions per element to be
throughput-bound at that rate, against a CUDA kernel for the SAME container
format that runs the same job at 13.6 instructions/element.  Occupancy
sweeps were flat and a single-thread-ILP variant was flat too -- that
signature is per-lane SERIAL LATENCY, not ALU throughput.

Both phase-1 and phase-2 kernels are serial in the escape rank specifically:

  * v1 (element-per-thread) recomputes the rank from scratch per element,
    with a popcount over up to 64 packed bits (two 32-bit halves) every
    single thread redoes independently.
  * v2 (tile-per-thread) removes the popcount but replaces it with 64
    DEPENDENT loop iterations inside one lane (the running-counter design) --
    serial latency, just a cheaper op per step.

Neither has the cross-lane parallel rank that took the CUDA kernel from 22.5
to 13.6 instructions/element (its branch-free escape scatter, built on
``__shfl``/warp ballot).  This module builds that for Metal: one SIMD group
(32 lanes) owns one 64-element tile, 2 adjacent elements per lane, and the
escape rank is a single ``simd_prefix_exclusive_sum`` hardware instruction
across the group instead of a 64-step dependency chain or a per-element
popcount.  Lane order (0..31) maps monotonically onto element order
(elem = 2*lane, 2*lane+1), so the prefix sum's lane-ascending accumulation
order IS the tile's element-ascending escape-rank order -- no reshuffling
needed to make the parallel scan match the format's rank definition.

GEOMETRY (see the phase-3 owner brief for the full sweep):

  * ``TPG`` (tiles per SIMD group, in flight) -- 1, 2, 4, 8: independent
    tiles processed by ONE simdgroup in an unrolled loop, for memory-level
    parallelism (more outstanding loads before the group blocks on the first
    one), not more lanes per tile.
  * ``SPT`` (simdgroups per threadgroup) -- 1, 2, 4: occupancy/threadgroup
    shape, independent of TPG.
  * tile-header load strategy -- every lane loads its own (redundant, cache-
    absorbed) copy of the tile's 6 plane words (default), OR lane 0 loads
    once and ``simd_broadcast``s to the rest (``broadcast_load=True``,
    exposed only on variant ``t1_g1`` / its broadcast twin for the
    internal "measure both" comparison -- see
    ``scripts/tbe_metal_decode_bench_v3.py``).

Both ``TPG`` and ``SPT`` are baked into the kernel source as compile-time
integer literals (Python string formatting, not a runtime branch), same
discipline as v2's ``tiles_per_thread``, so the Metal compiler sees fixed
trip counts and can unroll/schedule accordingly.

Global-simdgroup indexing uses ``threadgroup_position_in_grid`` and
``simdgroup_index_in_threadgroup`` (not
``thread_position_in_grid.x / 32``) so the grid can always be dispatched as
an exact multiple of the threadgroup width -- no dependence on how MLX's
underlying ``dispatchThreads`` numbers a non-uniform last threadgroup.

Escape-value / in-window select is branch-free (ternary/``select``-style, as
in v1/v2): every lane always computes an in-window delta from a tiny 8-entry
constant table (``TBE_DELTA``) AND a speculative (always-in-bounds-clamped)
escape-table read, then picks between them with a boolean select.  The only
two branches per tile iteration are (a) the trip-count guard
(``tile >= n_tiles``) and (b) the fast/slow store-path selector
(``remain >= 64``) -- both UNIFORM across the whole simdgroup (every lane in
a group agrees, since ``elem0``/``tile`` don't vary by lane), so neither is a
divergent branch on Apple's SIMD hardware.

NEW FILE.  v1/v2/their tests/bench are untouched, matching the phase-1/2
precedent of not editing a module that was mid-commit when a new phase
starts.
"""
from __future__ import annotations

import mlx.core as mx

from .tbe_decode_mlx import (  # noqa: F401  (re-exported for v3 callers)
    METAL_MEMORY_LIMIT_BYTES,
    TBEDeviceMLX,
    _enforce_memory_limit,
    bits_view,
    mlx_metal_available,
    upload_tbe_mlx,
)
from ..tbe_container import MODE_W6Z

# ---------------------------------------------------------------------------
# header: the "tiny constant table" the owner brief asks for -- code (0..7)
# -> in-window exponent delta from base.  code 0 (escape) maps to 0 but is
# never selected for escapes (the select picks the escape-table read
# instead); code 7 maps to 6 (mode W7) and is overridden to the exact-zero
# exponent (0) for mode W6Z by a separate select, same as v1/v2.
# ---------------------------------------------------------------------------
_HEADER = r"""
constant int TBE_DELTA[8] = {0, 0, 1, 2, 3, 4, 5, 6};
"""

# ---------------------------------------------------------------------------
# per-tile body, unrolled TPG times inside one simdgroup's kernel body.
# ---------------------------------------------------------------------------
_TILE_BODY_LOAD_REDUNDANT = r"""
        p0l = planes[off + 0u]; p0h = planes[off + 1u];
        p1l = planes[off + 2u]; p1h = planes[off + 3u];
        p2l = planes[off + 4u]; p2h = planes[off + 5u];
"""

_TILE_BODY_LOAD_BROADCAST = r"""
        uint w0l = 0u, w0h = 0u, w1l = 0u, w1h = 0u, w2l = 0u, w2h = 0u;
        if (lane == 0u) {
            w0l = planes[off + 0u]; w0h = planes[off + 1u];
            w1l = planes[off + 2u]; w1h = planes[off + 3u];
            w2l = planes[off + 4u]; w2h = planes[off + 5u];
        }
        p0l = simd_broadcast(w0l, 0u); p0h = simd_broadcast(w0h, 0u);
        p1l = simd_broadcast(w1l, 0u); p1h = simd_broadcast(w1h, 0u);
        p2l = simd_broadcast(w2l, 0u); p2h = simd_broadcast(w2h, 0u);
"""

_TILE_BODY = r"""
        uint elem0 = tile * 64u;
        uint off = tile * 6u;
        uint p0l, p0h, p1l, p1h, p2l, p2h;
%(load_block)s
        // Lane order 0..31 -> element order (2*lane, 2*lane+1) is
        // monotonic in elem, so a lane-ascending simd_prefix_exclusive_sum
        // IS the tile's element-ascending escape-rank scan.
        uint bit_a = 2u * lane;
        uint bit_b = bit_a + 1u;
        bool hi = lane >= 16u;
        uint shA = hi ? (bit_a - 32u) : bit_a;
        uint shB = hi ? (bit_b - 32u) : bit_b;
        uint w0 = hi ? p0h : p0l;
        uint w1 = hi ? p1h : p1l;
        uint w2 = hi ? p2h : p2l;

        uint code_a = ((w0 >> shA) & 1u) | (((w1 >> shA) & 1u) << 1) | (((w2 >> shA) & 1u) << 2);
        uint code_b = ((w0 >> shB) & 1u) | (((w1 >> shB) & 1u) << 1) | (((w2 >> shB) & 1u) << 2);

        bool esc_a = (code_a == 0u);
        bool esc_b = (code_b == 0u);
        uint lane_esc = (esc_a ? 1u : 0u) + (esc_b ? 1u : 0u);
        // The one cross-lane instruction replacing v1's per-element popcount
        // and v2's 64-deep dependent running counter: O(log 32) hardware
        // scan, not a serial chain.
        uint prefix = simd_prefix_exclusive_sum(lane_esc);
        uint rank_before = tile_base[tile] + prefix;
        uint rank_a = rank_before;
        uint rank_b = rank_before + (esc_a ? 1u : 0u);

        int delta_a = TBE_DELTA[code_a];
        int delta_b = TBE_DELTA[code_b];
        uint inwin_a = uint(b + delta_a);
        uint inwin_b = uint(b + delta_b);
        inwin_a = (is_w6z != 0u && code_a == 7u) ? 0u : inwin_a;
        inwin_b = (is_w6z != 0u && code_b == 7u) ? 0u : inwin_b;

        // Speculative, always-in-bounds-clamped escape reads (same
        // discipline as v1/v2): the select below discards them on every
        // non-escape lane rather than branching around the load.
        uint esc_idx_a = rank_a < esc_len ? rank_a : (esc_len - 1u);
        uint esc_idx_b = rank_b < esc_len ? rank_b : (esc_len - 1u);
        uint esc_val_a = uint(esc[esc_idx_a]);
        uint esc_val_b = uint(esc[esc_idx_b]);

        uint mexp_a = esc_a ? esc_val_a : inwin_a;
        uint mexp_b = esc_b ? esc_val_b : inwin_b;

        uint elem_a = elem0 + bit_a;
        uint elem_b = elem0 + bit_b;
        uchar smb_a = smb[elem_a];
        uchar smb_b = smb[elem_b];
        ushort word_a = (ushort(smb_a & 0x80) << 8) | (ushort(mexp_a & 0xFFu) << 7) | ushort(smb_a & 0x7Fu);
        ushort word_b = (ushort(smb_b & 0x80) << 8) | (ushort(mexp_b & 0xFFu) << 7) | ushort(smb_b & 0x7Fu);

        uint remain = n - elem0;
        if (remain >= 64u) {
            // Coalesced: 32 lanes each store one ushort2 (4 B), 128 B total,
            // contiguous across the group -- the tile's whole bf16 output.
            ((device ushort2*)(out + elem0))[lane] = ushort2(word_a, word_b);
        } else {
            // Partial final tile only (at most one per decode call).
            if (elem_a < n) { out[elem_a] = as_type<bfloat16_t>(word_a); }
            if (elem_b < n) { out[elem_b] = as_type<bfloat16_t>(word_b); }
        }
"""


def _build_source(tiles_per_group: int, simdgroups_per_tg: int, broadcast_load: bool) -> str:
    load_block = _TILE_BODY_LOAD_BROADCAST if broadcast_load else _TILE_BODY_LOAD_REDUNDANT
    body = _TILE_BODY % {"load_block": load_block}
    header = r"""
    uint lane = thread_index_in_simdgroup;
    uint sgid = simdgroup_index_in_threadgroup;
    uint tgid = threadgroup_position_in_grid.x;
    uint global_sg = tgid * %(spt)du + sgid;

    uint n_tiles_v = n_tiles[0];
    uint n = n_elem[0];
    uint esc_len = esc_len_arr[0];
    int b = base[0];
    uint is_w6z = w6z[0];

    uint tile_group_base = global_sg * %(tpg)du;

    for (uint tt = 0u; tt < %(tpg)du; tt++) {
        uint tile = tile_group_base + tt;
        if (tile >= n_tiles_v) { break; }
%(body)s
    }
""" % {"tpg": int(tiles_per_group), "spt": int(simdgroups_per_tg), "body": body}
    return header


_KERNEL_CACHE: dict = {}


def _kernel_for(tiles_per_group: int, simdgroups_per_tg: int, broadcast_load: bool):
    key = (tiles_per_group, simdgroups_per_tg, broadcast_load)
    if key not in _KERNEL_CACHE:
        tag = f"t{tiles_per_group}_g{simdgroups_per_tg}" + ("_bcast" if broadcast_load else "")
        _KERNEL_CACHE[key] = mx.fast.metal_kernel(
            name=f"tbe_decode_flat64_v3_{tag}",
            input_names=[
                "planes", "smb", "esc", "tile_base",
                "base", "w6z", "n_elem", "n_tiles", "esc_len_arr",
            ],
            output_names=["out"],
            source=_build_source(tiles_per_group, simdgroups_per_tg, broadcast_load),
            header=_HEADER,
        )
    return _KERNEL_CACHE[key]


#: (tiles_per_group, simdgroups_per_threadgroup) grid the owner brief asks
#: for: TPG in {1,2,4,8} x SPT in {1,2,4}.
_VARIANT_GRID = {
    f"t{tpg}_g{spt}": (tpg, spt)
    for tpg in (1, 2, 4, 8)
    for spt in (1, 2, 4)
}
VARIANTS = tuple(_VARIANT_GRID.keys())

#: The one variant the owner brief's broadcast-vs-redundant-load comparison
#: is measured on (the baseline geometry, 1 tile/group, 1 group/threadgroup).
BROADCAST_COMPARISON_VARIANT = "t1_g1"


def decode_tbe_metal_v3(
    c: TBEDeviceMLX,
    variant: str = "t1_g1",
    broadcast_load: bool = True,
) -> mx.array:
    """MLX-resident TBE container -> bf16 ``mx.array`` of ``c.shape``, phase-3.

    ``variant`` selects ``(tiles_per_group, simdgroups_per_threadgroup)`` --
    see module docstring. ``broadcast_load`` toggles the tile-header load
    strategy: lane-0-loads-then-``simd_broadcast`` (the default -- measured
    ~2x the redundant per-lane load at the baseline geometry, see
    ``scripts/tbe_metal_decode_bench_v3.py``'s header-load-strategy rows) vs
    every lane loading its own redundant copy (``broadcast_load=False``, kept
    for the comparison and as a fallback if a future MLX/Metal version
    restricts ``simd_broadcast``).
    """
    if variant not in _VARIANT_GRID:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    tiles_per_group, simdgroups_per_tg = _VARIANT_GRID[variant]

    _enforce_memory_limit()
    n = c.numel
    if n == 0:
        return mx.zeros(c.shape, dtype=mx.bfloat16)

    n_tiles = max(1, int(c.tiles))
    total_sg = max(1, -(-n_tiles // tiles_per_group))  # ceil div
    n_tg = max(1, -(-total_sg // simdgroups_per_tg))  # ceil div
    tg_width = simdgroups_per_tg * 32
    grid_x = n_tg * tg_width

    base_arr = mx.array([int(c.base)], dtype=mx.int32)
    w6z_arr = mx.array([1 if int(c.mode) == MODE_W6Z else 0], dtype=mx.uint32)
    n_arr = mx.array([n], dtype=mx.uint32)
    n_tiles_arr = mx.array([int(c.tiles)], dtype=mx.uint32)
    esc_len_arr = mx.array([int(c.esc.size)], dtype=mx.uint32)

    kernel = _kernel_for(tiles_per_group, simdgroups_per_tg, broadcast_load)
    outputs = kernel(
        inputs=[
            c.planes, c.smb, c.esc, c.tile_base,
            base_arr, w6z_arr, n_arr, n_tiles_arr, esc_len_arr,
        ],
        template=[],
        grid=(grid_x, 1, 1),
        threadgroup=(tg_width, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.bfloat16],
    )
    out = outputs[0].reshape(c.shape)
    mx.eval(out)
    return out


__all__ = [
    "BROADCAST_COMPARISON_VARIANT",
    "METAL_MEMORY_LIMIT_BYTES",
    "TBEDeviceMLX",
    "VARIANTS",
    "bits_view",
    "decode_tbe_metal_v3",
    "mlx_metal_available",
    "upload_tbe_mlx",
]
