"""GLC-TBE flat64 decode, Metal (MLX) -- Phase 2: thread-per-TILE kernels.

Phase 1 (``tbe_decode_mlx.py``) is one thread per OUTPUT ELEMENT: 43-47 GB/s
of bf16 produced, ~9% of a 505 GB/s Metal bandwidth probe -- instruction-bound,
not bandwidth-bound.  Every thread there redoes the same tile-relative address
math (``elem/64``, ``elem%64``) and the same two-32-bit-half popcount for
escape rank that its 63 tile-mates also compute, for the exact same tile.

This module is one thread per TILE instead.  A single thread:

  * loads the tile's 6 plane words ONCE (v1's 64 threads reload them 64 times,
    modulo whatever the L1 cache absorbs);
  * derives ``elem0 = tile * 64`` directly from ``thread_position_in_grid``
    instead of a division/modulo per element;
  * carries the escape rank as a RUNNING COUNTER seeded from
    ``tile_base[tile]`` and incremented by 0 or 1 per element (a compare and
    an add) instead of v1's per-element popcount over a masked 32-bit half --
    the free lunch a tile-per-thread kernel gets that an element-per-thread
    kernel structurally cannot (threads don't execute in tile order there);
  * writes output in ``ushort4`` groups (4 elements/store) on the fast path
    (a full 64-element tile), not one ``bfloat16_t`` store per element.

Escape-rank correctness note: the per-element ``mexp`` select reads
``esc[esc_idx]`` on EVERY element, escape or not, to keep the hot loop
branch-light (a ``select``/ternary instead of an ``if``/``else`` around the
memory read).  ``esc_idx`` is clamped to ``esc_len - 1`` so this speculative
read is always in-bounds; the value is discarded by the select whenever the
element is not an escape, so an out-of-range clamp never contaminates a real
result -- only ``rank`` (which advances only on real escapes) determines
which ``esc[]`` byte counts.

TWO VARIANTS, chosen by ``tiles_per_thread`` (baked into the kernel source as
a compile-time-constant loop trip count, not a runtime branch):

  * ``tile1`` -- 1 tile / thread (64 elements/thread), grid = tile count.
  * ``tile2`` -- 2 tiles / thread (128 elements/thread), grid = ceil(tile
    count / 2).  Half the thread count, more serial work per thread; measures
    the occupancy trade the phase-2 brief asks for (64 vs 128 elements/thread).

Both variants keep the module-level Metal-allocation safety hold
(``METAL_MEMORY_LIMIT_BYTES`` from ``tbe_decode_mlx``, re-enforced here at
import time and before every launch) and reuse ``TBEDeviceMLX`` /
``upload_tbe_mlx`` / ``bits_view`` unchanged -- only the kernel and its
launch geometry differ from phase 1.

NEW FILE, phase-1 file untouched: phase 1's module was mid-commit in this
repo's git history when this file was written (see the phase-2 owner brief);
touching it in place would have raced that commit, so this is an addition,
not an edit.  ``scripts/tbe_metal_decode_bench_v2.py`` selects among v1/v2
variants at the CLI rather than the phase-1 bench script being changed.
"""
from __future__ import annotations

import mlx.core as mx

from .tbe_decode_mlx import (  # noqa: F401  (re-exported for v2 callers)
    METAL_MEMORY_LIMIT_BYTES,
    TBEDeviceMLX,
    _enforce_memory_limit,
    bits_view,
    mlx_metal_available,
    upload_tbe_mlx,
)
from ..tbe_container import MODE_W6Z

# ---------------------------------------------------------------------------
# kernel source, templated on tiles-per-thread (a Python-side compile-time
# constant baked into the source text as an integer literal, so the Metal
# compiler sees a fixed-trip-count outer loop and can unroll it)
# ---------------------------------------------------------------------------
_TILE_BODY = r"""
        uint elem0 = tile * 64u;
        if (elem0 >= n) { break; }

        uint off = tile * 6u;
        uint p0l = planes[off + 0]; uint p0h = planes[off + 1];
        uint p1l = planes[off + 2]; uint p1h = planes[off + 3];
        uint p2l = planes[off + 4]; uint p2h = planes[off + 5];
        uint rank = tile_base[tile];
        uint remain = n - elem0;

        if (remain >= 64u) {
            // Fast path: full tile, no per-lane bounds check, 16 vectorized
            // ushort4 stores (4x fewer store instructions than v1's 64
            // scalar bfloat16_t stores for this tile).
            for (uint g = 0u; g < 16u; g++) {
                ushort4 words;
                for (uint j = 0u; j < 4u; j++) {
                    uint lane = g * 4u + j;
                    bool hi = lane >= 32u;
                    uint sh = hi ? (lane - 32u) : lane;
                    uint w0 = hi ? p0h : p0l;
                    uint w1 = hi ? p1h : p1l;
                    uint w2 = hi ? p2h : p2l;
                    uint code = ((w0 >> sh) & 1u)
                              | (((w1 >> sh) & 1u) << 1)
                              | (((w2 >> sh) & 1u) << 2);
                    bool is_escape = (code == 0u);
                    uint inwin = uint(b + int(code) - 1);
                    if (is_w6z != 0u && code == 7u) { inwin = 0u; }
                    uint esc_idx = rank < esc_len ? rank : (esc_len - 1u);
                    uint esc_val = uint(esc[esc_idx]);
                    uint mexp = is_escape ? esc_val : inwin;
                    rank = rank + (is_escape ? 1u : 0u);
                    uint idx = elem0 + lane;
                    uchar smb_v = smb[idx];
                    ushort word = (ushort(smb_v & 0x80) << 8)
                                | (ushort(mexp & 0xFFu) << 7)
                                | ushort(smb_v & 0x7Fu);
                    if (j == 0u) { words.x = word; }
                    else if (j == 1u) { words.y = word; }
                    else if (j == 2u) { words.z = word; }
                    else { words.w = word; }
                }
                ((device ushort4*)(out + elem0 + g * 4u))[0] = words;
            }
        } else {
            // Slow path: the tensor's final, partial tile only (at most one
            // per whole decode call) -- scalar, per-lane bounds-checked.
            for (uint lane = 0u; lane < remain; lane++) {
                bool hi = lane >= 32u;
                uint sh = hi ? (lane - 32u) : lane;
                uint w0 = hi ? p0h : p0l;
                uint w1 = hi ? p1h : p1l;
                uint w2 = hi ? p2h : p2l;
                uint code = ((w0 >> sh) & 1u)
                          | (((w1 >> sh) & 1u) << 1)
                          | (((w2 >> sh) & 1u) << 2);
                bool is_escape = (code == 0u);
                uint inwin = uint(b + int(code) - 1);
                if (is_w6z != 0u && code == 7u) { inwin = 0u; }
                uint esc_idx = rank < esc_len ? rank : (esc_len - 1u);
                uint esc_val = uint(esc[esc_idx]);
                uint mexp = is_escape ? esc_val : inwin;
                rank = rank + (is_escape ? 1u : 0u);
                uint idx = elem0 + lane;
                uchar smb_v = smb[idx];
                ushort word = (ushort(smb_v & 0x80) << 8)
                            | (ushort(mexp & 0xFFu) << 7)
                            | ushort(smb_v & 0x7Fu);
                out[idx] = as_type<bfloat16_t>(word);
            }
        }
"""


def _build_source(tiles_per_thread: int) -> str:
    header = r"""
    uint group_tile = thread_position_in_grid.x;
    uint n_tiles_v = n_tiles[0];
    uint n = n_elem[0];
    uint esc_len = esc_len_arr[0];
    int b = base[0];
    uint is_w6z = w6z[0];

    for (uint tt = 0u; tt < %(k)du; tt++) {
        uint tile = group_tile * %(k)du + tt;
        if (tile >= n_tiles_v) { break; }
%(body)s
    }
""" % {"k": int(tiles_per_thread), "body": _TILE_BODY}
    return header


_KERNEL_CACHE: dict = {}


def _kernel_for(tiles_per_thread: int):
    if tiles_per_thread not in _KERNEL_CACHE:
        _KERNEL_CACHE[tiles_per_thread] = mx.fast.metal_kernel(
            name=f"tbe_decode_flat64_tile{tiles_per_thread}",
            input_names=[
                "planes", "smb", "esc", "tile_base",
                "base", "w6z", "n_elem", "n_tiles", "esc_len_arr",
            ],
            output_names=["out"],
            source=_build_source(tiles_per_thread),
        )
    return _KERNEL_CACHE[tiles_per_thread]


#: Default threadgroup size for the tile-per-thread kernels.  Grids here are
#: over TILES (1/64th the element count of v1's grid), so the same 256
#: nominal width covers 16384 elements/threadgroup at tiles_per_thread=1.
DEFAULT_THREADGROUP = 256

VARIANTS = ("tile1", "tile2")
_VARIANT_TILES_PER_THREAD = {"tile1": 1, "tile2": 2}


def decode_tbe_metal_v2(
    c: TBEDeviceMLX,
    variant: str = "tile1",
    threadgroup: int = DEFAULT_THREADGROUP,
) -> mx.array:
    """MLX-resident TBE container -> bf16 ``mx.array`` of ``c.shape``, tile-per-thread.

    ``variant`` selects tiles-per-thread (see module docstring): ``"tile1"``
    (1 tile, 64 elements/thread) or ``"tile2"`` (2 tiles, 128 elements/thread).
    """
    if variant not in _VARIANT_TILES_PER_THREAD:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    tiles_per_thread = _VARIANT_TILES_PER_THREAD[variant]

    _enforce_memory_limit()
    n = c.numel
    if n == 0:
        return mx.zeros(c.shape, dtype=mx.bfloat16)

    n_threads = max(1, -(-c.tiles // tiles_per_thread))  # ceil div
    tg = max(1, min(int(threadgroup), n_threads))

    base_arr = mx.array([int(c.base)], dtype=mx.int32)
    w6z_arr = mx.array([1 if int(c.mode) == MODE_W6Z else 0], dtype=mx.uint32)
    n_arr = mx.array([n], dtype=mx.uint32)
    n_tiles_arr = mx.array([int(c.tiles)], dtype=mx.uint32)
    esc_len_arr = mx.array([int(c.esc.size)], dtype=mx.uint32)

    kernel = _kernel_for(tiles_per_thread)
    outputs = kernel(
        inputs=[
            c.planes, c.smb, c.esc, c.tile_base,
            base_arr, w6z_arr, n_arr, n_tiles_arr, esc_len_arr,
        ],
        template=[],
        grid=(n_threads, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.bfloat16],
    )
    out = outputs[0].reshape(c.shape)
    mx.eval(out)
    return out


__all__ = [
    "DEFAULT_THREADGROUP",
    "METAL_MEMORY_LIMIT_BYTES",
    "TBEDeviceMLX",
    "VARIANTS",
    "bits_view",
    "decode_tbe_metal_v2",
    "mlx_metal_available",
    "upload_tbe_mlx",
]
