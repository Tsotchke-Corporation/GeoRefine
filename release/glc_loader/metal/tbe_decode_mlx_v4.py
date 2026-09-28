"""GLC-TBE flat64 decode, Metal (MLX) -- Phase 4: BATCHABLE dispatch.

Phase 4 changes no Metal source at all.  It imports ``_kernel_for`` from
``tbe_decode_mlx_v3`` and dispatches the identical compiled kernel, so every
phase-1..3 receipt stays comparable element for element.  What it changes is
the *dispatch policy*, which the phase-4 measurement showed to be the whole
of the gap the phase-1..3 receipts attributed, in turn, to ALU throughput,
per-lane latency, and memory access pattern.

WHY THERE IS NO NEW KERNEL HERE
-------------------------------
The open hypothesis on record was an interleaved tile layout to fix an
assumed coalescing deficit.  Measured on this machine (receipts:
``.icc/evidence/tbe-metal-decode-20260903/phase4/``, analysis:
``docs/research/TBE_METAL_ACCESS_PATTERN_20260903.md``):

  * ``v3`` already reads and writes fully coalesced -- 64 contiguous ``smb``
    bytes and 128 contiguous output bytes per SIMD group -- and moves
    **1.72 B per output byte** against the ``copy8`` ceiling control's
    **2.0 B**;
  * ``decode_tbe_metal_v3(variant="t4_g4")`` on a 256 MB container in ONE
    launch measures **279.0 / 289.4 GB/s**, i.e. **1.008 / 1.136x a pure
    vectorised copy of the same size**, against a harness ceiling that
    reproduced at 284.0 / 286.3 GB/s.

A kernel already beating a same-shape copy has no coalescing left to
recover.  There is no v4 *kernel*; the interleaved-tile item is measured out.

WHAT THE GAP ACTUALLY WAS
-------------------------
Every decoder in this package calls ``mx.eval(out)`` on itself before
returning (``tbe_decode_mlx.py:326``, ``_v2.py:232``, ``_v3.py:300``).  An
``mx.eval`` is a CPU<->GPU round trip costing **~167 us** on this machine,
against **~4.6 us** for a dispatch that is left in the graph.  For a 5.24 MB
k/v-proj -- 18 us of kernel work -- that is 90% overhead; the certified
Qwen3-4B artifact pays it 252 times per token.  Measured on 16 x 16 MB
containers, the identical kernel runs at **57.3 GB/s** with an eval per
chunk and **173.1 GB/s** with one eval for the batch.

So :func:`decode_tbe_metal_v4` returns a LAZY array.  The caller -- a
forward pass, a benchmark rotation -- evaluates once for the whole batch.
``eval_now=True`` restores the old eager contract for callers that genuinely
need the value in hand (a bit-exactness check, a one-shot decode).

TWO SMALLER FIXES, BOTH MEASURED
--------------------------------
* **Default variant ``t4_g4``, not ``t1_g1``.**  ``TBELinearMLXV3`` defaults
  to ``t1_g1``, which measures 92.5 GB/s at 256 MB against ``t4_g4``'s 279.0
  -- and the certified serving path
  (``tbe_mlx_model.py:237``) uses ``TBELinearMLX``, the v1 element-per-thread
  kernel, at 106.6 GB/s.  The default here is the fastest measured geometry.
* **Hoisted uniform scalars.**  ``decode_tbe_metal_v3`` builds five
  single-element ``mx.array`` uniforms on every call (``_v3.py:285-289``).
  They are properties of the container, not of the call, so this module
  builds them once and caches them on the container.  Worth ~15 us/decode at
  20.97 MB in lazy mode (nothing in eager mode, where the 167 us eval buries
  it) -- second order, but free.

Bit-exactness is unchanged by construction: same kernel, same inputs, same
output dtype.  ``scripts/tbe_metal_decode_bench_phase4.py`` checks every
variant against ``tbe_container.decode_tbe`` before it is allowed to report
a timing.
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx

from .tbe_decode_mlx import (  # noqa: F401  (re-exported for v4 callers)
    METAL_MEMORY_LIMIT_BYTES,
    TBEDeviceMLX,
    _enforce_memory_limit,
    bits_view,
    mlx_metal_available,
    upload_tbe_mlx,
)
from .tbe_decode_mlx_v3 import VARIANTS, _VARIANT_GRID, _kernel_for
from ..tbe_container import MODE_W6Z

#: Fastest geometry measured on Apple M2 Ultra across 5.24 / 20.97 / 49.81 /
#: 256 MB tiers, both runs (see the phase-4 receipts).  ``t8_g2`` is within
#: noise of it; ``t1_g1`` -- the ``TBELinearMLXV3`` default -- is ~3x slower.
DEFAULT_VARIANT = "t4_g4"

#: Attribute name under which the per-container uniform arrays are cached.
_UNIFORMS_ATTR = "_glc_tbe_v4_uniforms"


def _uniforms(c: TBEDeviceMLX):
    """The five call-invariant uniform arrays, built once per container.

    Cached on the container object rather than in a module-level dict so the
    arrays are released when the container is, with no id-reuse hazard and no
    process-lifetime leak in a loader that swaps hundreds of tensors.
    """
    got = getattr(c, _UNIFORMS_ATTR, None)
    if got is None:
        got = (
            mx.array([int(c.base)], dtype=mx.int32),
            mx.array([1 if int(c.mode) == MODE_W6Z else 0], dtype=mx.uint32),
            mx.array([c.numel], dtype=mx.uint32),
            mx.array([int(c.tiles)], dtype=mx.uint32),
            mx.array([int(c.esc.size)], dtype=mx.uint32),
        )
        # Deliberately NOT evaluated here: these are five single-element
        # arrays, and forcing them would put a CPU<->GPU round trip back into
        # the decode path that this module exists to remove.  They are cached
        # objects, so MLX materialises them once, folded into whichever graph
        # first uses them.
        try:
            setattr(c, _UNIFORMS_ATTR, got)
        except Exception:
            # A frozen/slotted container still works, it just rebuilds them.
            pass
    return got


def decode_tbe_metal_v4(
    c: TBEDeviceMLX,
    variant: str = DEFAULT_VARIANT,
    broadcast_load: bool = True,
    eval_now: bool = False,
) -> mx.array:
    """MLX-resident TBE container -> bf16 ``mx.array`` of ``c.shape``, LAZY.

    Identical kernel to :func:`decode_tbe_metal_v3`; the returned array is
    left unevaluated so a caller can batch many decodes behind one
    ``mx.eval``.  Pass ``eval_now=True`` for the eager v3 contract.

    A caller that forgets to evaluate gets MLX's ordinary lazy semantics --
    the value materialises the first time it is used -- so this is not a
    correctness hazard, only a performance one, and the whole point is that
    the natural use (feeding the array straight into ``mx.matmul``) is now
    also the fast one.
    """
    if variant not in _VARIANT_GRID:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    tiles_per_group, simdgroups_per_tg = _VARIANT_GRID[variant]

    _enforce_memory_limit()
    n = c.numel
    if n == 0:
        return mx.zeros(c.shape, dtype=mx.bfloat16)

    n_tiles = max(1, int(c.tiles))
    total_sg = max(1, -(-n_tiles // tiles_per_group))
    n_tg = max(1, -(-total_sg // simdgroups_per_tg))
    tg_width = simdgroups_per_tg * 32

    kernel = _kernel_for(tiles_per_group, simdgroups_per_tg, broadcast_load)
    out = kernel(
        inputs=[c.planes, c.smb, c.esc, c.tile_base, *_uniforms(c)],
        template=[],
        grid=(n_tg * tg_width, 1, 1),
        threadgroup=(tg_width, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.bfloat16],
    )[0].reshape(c.shape)
    if eval_now:
        mx.eval(out)
    return out


__all__ = [
    "DEFAULT_VARIANT",
    "METAL_MEMORY_LIMIT_BYTES",
    "TBEDeviceMLX",
    "VARIANTS",
    "bits_view",
    "decode_tbe_metal_v4",
    "mlx_metal_available",
    "upload_tbe_mlx",
]
