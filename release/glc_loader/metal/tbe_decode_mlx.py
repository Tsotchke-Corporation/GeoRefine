"""GLC-TBE flat64 decode as an MLX custom Metal kernel.

Ports the ``transient-pool`` arm of ``tbe_modules.py`` (M > 16: decode a
weight to a bf16 tile, then ``torch.matmul`` / here ``mx.matmul``) to Apple
Silicon.  The fused fragment kernel (``tbe_mma.py``, M <= 16, CUDA
``mma.sync``) is NOT ported -- it is a CUDA tensor-core-fragment layout
(``mma16``) with no Metal analogue, and the owner's spec for this piece
explicitly scopes it out.  What is ported is architecture-independent: read
the compressed bytes, reconstruct the bf16 bit pattern exactly, matmul.

BIT SEMANTICS, unchanged from ``tbe_container.py`` (the source of truth is
``experiments/georefine/_glc_tbe.py``; this file re-derives nothing, it just
targets a different device):

    code = 3-bit value packed across three 1-bit planes, tile of 64 elements
    code == 0           -> escape: raw exponent stored in ``esc[]``, addressed
                            by popcount RANK (not an offset table)
    code in [1, 6 or 7]  -> in-window: exponent = base + code - 1
    mode W6Z, code == 7  -> exact zero (exponent 0), the pruned-tensor amendment
    word = (sign<<15) | (exponent<<7) | mantissa    -- exactly decode_tbe's line

KERNEL DESIGN (read before changing anything below)
----------------------------------------------------
**One thread per OUTPUT ELEMENT**, not one thread per tile.  A tile-per-thread
kernel would serialize 64 elements behind one Apple-GPU thread and leave the
SIMD width unused; element-per-thread keeps every thread doing O(1) work
(a handful of shifts, one popcount, one or two byte loads) and saturates
occupancy the way the CUDA kernel's own "the same address for all 64 lanes"
design note says the format was built for -- the 6 plane words for a tile are
read by all 64 threads of that tile at an address that does not depend on the
thread, so on Apple's SIMD-32 hardware every warp-equivalent group that
shares a tile reads the same 24 bytes twice (once per 32-lane half), not 64
times.

The one piece of state a tile-per-thread kernel gets for free -- a running
escape-rank counter -- an element-per-thread kernel cannot have, because
threads don't execute in tile order.  This is solved the same way
``tbe_row_escape_base`` solves it for rows in the CUDA host code: a
**precomputed per-tile escape base** (``tile_base[T]``, uint32, "escapes
strictly before tile t"), built once on the host from the full plane bitmaps
in :func:`_tile_escape_base` and uploaded alongside the container.  With that
base in hand, a thread's own rank is thread-local: popcount the tile's escape
mask, masked to the bits before its own lane -- exactly the CUDA note's
``popcount(esc_mask & ((1 << lane) - 1))``, done here in two 32-bit halves
because Metal's ``popcount`` is defined over 32-bit (and smaller) integer
types, not a 64-bit lane mask.

THREADGROUP SHAPE: ``(256, 1, 1)`` (or ``min(256, n)`` for tiny inputs), a flat
1-D grid over elements.  256 is 8 tiles' worth of elements per threadgroup,
big enough to hide the two scalar-uniform loads (``base``, ``w6z``) behind
real work and small enough that a threadgroup never needs more than 6 tiles'
plane words (144 B) resident to serve it, which is irrelevant for Apple's
unified memory model but keeps the launch a single simple dispatch with no
threadgroup-memory declaration at all.

SAFETY: every entry point in this module calls ``mx.set_memory_limit`` at
import time to enforce the standing GPU-panic hold
at the MLX allocator level, not just by
convention in caller code -- so an over-budget call fails as an MLX
allocation error rather than risking the driver.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

import mlx.core as mx

from ..tbe_container import (
    LANES_PER_WORD,
    MODE_W6Z,
    MODE_W7,
    PLANES,
    TILE,
    TBEError,
    TBETensor,
)

#: Hard ceiling on total Metal allocation for every call in this module,
#: enforcing the standing safety hold against a driver-level GPU panic under large MLX allocations.  Set well
#: under the 1 GB budget so kernel scratch + MLX's own bookkeeping never
#: pushes the process over it.
METAL_MEMORY_LIMIT_BYTES = 900 * 1024 * 1024


def mlx_metal_available() -> bool:
    """True iff MLX can see a Metal GPU device on this machine.

    Never raises -- callers (tests, benchmark) use this to skip cleanly
    rather than fail when there is no GPU or the Metal backend errors out.
    """
    try:
        import mlx.core as _mx

        return bool(_mx.metal.is_available())
    except Exception:
        return False


def _enforce_memory_limit() -> None:
    try:
        mx.set_memory_limit(METAL_MEMORY_LIMIT_BYTES)
    except Exception:
        # Older/newer MLX may rename this; never let the safety call itself
        # be the thing that crashes an otherwise-working decode.
        pass


_enforce_memory_limit()


# ---------------------------------------------------------------------------
# host-side precompute: per-tile escape base (derived, not stored on disk)
# ---------------------------------------------------------------------------
def _unpack_planes_np(planes: np.ndarray) -> np.ndarray:
    """``[T, 3, 2]`` uint32 plane words -> ``[T, 64]`` int code (0..7).

    Numpy mirror of ``tbe_container._unpack_planes`` (torch).  Kept
    independent rather than imported so this module's host precompute has no
    torch-tensor-vs-numpy-array ambiguity at the call site.
    """
    p = planes.astype(np.int64)
    t = p.shape[0]
    lane_bit = np.arange(LANES_PER_WORD, dtype=np.int64).reshape(1, 1, LANES_PER_WORD)
    code = np.zeros((t, TILE // LANES_PER_WORD, LANES_PER_WORD), dtype=np.int64)
    for b in range(PLANES):
        code |= ((p[:, b, :][:, :, None] >> lane_bit) & 1) << b
    return code.reshape(t, TILE)


def _tile_escape_base(planes: np.ndarray) -> np.ndarray:
    """Escapes strictly before tile ``t``, for every ``t`` -- exclusive cumsum.

    A DERIVED array like ``tbe_row_escape_base`` in the research module: it
    changes no stored byte and is reproducible from ``planes`` alone.  Global
    (not per-superblock) because the host has every tile's planes in hand
    when building an :class:`TBEDeviceMLX`, so there is no reason to make the
    kernel walk a superblock the way the CUDA host code does for a
    streaming GEMV that never sees all tiles at once.
    """
    code = _unpack_planes_np(planes)
    esc_per_tile = (code == 0).sum(axis=1).astype(np.int64)
    base = np.zeros_like(esc_per_tile)
    if esc_per_tile.size > 1:
        base[1:] = np.cumsum(esc_per_tile)[:-1]
    return base.astype(np.uint32)


# ---------------------------------------------------------------------------
# device container
# ---------------------------------------------------------------------------
@dataclass
class TBEDeviceMLX:
    """One flat64 TBE tensor resident as MLX arrays, byte for byte.

    ``planes`` is flattened to ``[T * 6]`` uint32 (tile ``t``'s six words at
    ``[6t, 6t+6)``) so the kernel indexes it with one multiply-add instead of
    a 3-D stride computation. ``esc`` always has at least one byte (a zero
    pad) so a zero-escape container never hands the kernel a zero-length
    buffer; the pad is never read because the pad is never a valid rank for
    any tile.
    """

    planes: mx.array   # uint32 [T * 6]
    smb: mx.array      # uint8  [T * 64]
    esc: mx.array      # uint8  [max(E, 1)]
    tile_base: mx.array  # uint32 [T]
    shape: tuple
    mode: int
    base: int
    tiles: int
    escapes: int
    #: GLC dtype code of the ORIGINAL tensor (1 = bf16).  The plane geometry is
    #: identical for every 2-byte dtype -- one coded byte, one raw byte -- so
    #: the only thing that changes is how the kernel puts the word back together
    #: and what it bitcasts to.  Defaults to bf16, which is what every container
    #: uploaded before 2026-09-05 is.
    dtype_code: int = 1

    @property
    def numel(self) -> int:
        return int(self.shape[0]) * int(self.shape[1])

    @property
    def resident_bytes(self) -> int:
        def _b(a: mx.array) -> int:
            return int(a.size) * int(a.itemsize)

        return _b(self.planes) + _b(self.smb) + _b(self.esc) + _b(self.tile_base)

    @property
    def dense_bytes(self) -> int:
        return self.numel * 2  # every dtype this container holds is 2 bytes

    def ratio(self) -> float:
        return self.dense_bytes / max(1, self.resident_bytes)


def upload_tbe_mlx(c: TBETensor) -> TBEDeviceMLX:
    """CPU ``TBETensor`` (torch, ``layout='flat64'``) -> MLX-resident container.

    Host-only work (numpy plane unpack for :func:`_tile_escape_base`,
    dtype narrowing); the actual byte transfer is ``mx.array`` on numpy views,
    which is the only interaction with the Metal allocator this function has.
    """
    if c.layout != "flat64":
        raise TBEError(
            f"the Metal decode targets layout='flat64'; got {c.layout!r}. "
            "tile8x8 encodes byte-identically but its element->lane map is "
            "not this kernel's -- re-encode with layout='flat64' to use it."
        )

    planes_np = c.planes.numpy().astype(np.uint32).reshape(c.tiles, PLANES, 2)
    smb_np = c.smb.numpy().astype(np.uint8)
    esc_src = c.esc.numpy().astype(np.uint8)
    esc_np = esc_src if esc_src.size > 0 else np.zeros(1, dtype=np.uint8)
    tile_base_np = _tile_escape_base(planes_np)

    return TBEDeviceMLX(
        planes=mx.array(planes_np.reshape(-1)),
        smb=mx.array(smb_np),
        esc=mx.array(esc_np),
        tile_base=mx.array(tile_base_np),
        shape=tuple(int(s) for s in c.shape),
        mode=int(c.mode),
        base=int(c.base),
        tiles=int(c.tiles),
        escapes=int(c.escapes),
        # Carried, not defaulted: a container whose dtype was dropped on upload
        # would be decoded as bf16 by a kernel that cannot tell the difference.
        dtype_code=int(getattr(c, "dtype_code", 1)),
    )


# ---------------------------------------------------------------------------
# the kernel
# ---------------------------------------------------------------------------
_DECODE_SOURCE = r"""
    uint elem = thread_position_in_grid.x;
    uint n = n_elem[0];
    if (elem >= n) {
        return;
    }
    uint tile = elem / 64;
    uint lane = elem % 64;
    uint off = tile * 6;

    uint p0l = planes[off + 0];
    uint p0h = planes[off + 1];
    uint p1l = planes[off + 2];
    uint p1h = planes[off + 3];
    uint p2l = planes[off + 4];
    uint p2h = planes[off + 5];

    bool hi = lane >= 32;
    uint sh = hi ? (lane - 32) : lane;
    uint w0 = hi ? p0h : p0l;
    uint w1 = hi ? p1h : p1l;
    uint w2 = hi ? p2h : p2l;
    uint code = ((w0 >> sh) & 1u)
              | (((w1 >> sh) & 1u) << 1)
              | (((w2 >> sh) & 1u) << 2);

    int b = base[0];
    uint is_w6z = w6z[0];
    uint mexp = 0u;
    if (code != 0u) {
        mexp = uint(b + int(code) - 1);
        if (is_w6z != 0u && code == 7u) {
            mexp = 0u;
        }
    } else {
        // escape: rank = tile base + popcount of the escape mask before
        // this lane, split into the two 32-bit halves the planes are
        // stored as (Metal popcount is defined over <=32-bit integers).
        uint esc_lo = ~(p0l | p1l | p2l);
        uint esc_hi = ~(p0h | p1h | p2h);
        uint rank_in_tile;
        if (!hi) {
            uint mask = (1u << lane) - 1u;
            rank_in_tile = uint(popcount(esc_lo & mask));
        } else {
            uint mask = (1u << sh) - 1u;
            rank_in_tile = uint(popcount(esc_lo)) + uint(popcount(esc_hi & mask));
        }
        uint rank = tile_base[tile] + rank_in_tile;
        mexp = uint(esc[rank]);
    }

    uchar smb_v = smb[elem];
    ushort word = (ushort(smb_v & 0x80) << 8)
                | (ushort(mexp & 0xFFu) << 7)
                | ushort(smb_v & 0x7F);
    out[elem] = as_type<bfloat16_t>(word);
"""

_DECODE_KERNEL = mx.fast.metal_kernel(
    name="tbe_decode_flat64",
    input_names=["planes", "smb", "esc", "tile_base", "base", "w6z", "n_elem"],
    output_names=["out"],
    source=_DECODE_SOURCE,
)

# The SAME kernel body for every other 2-byte dtype.  Only the last three lines
# differ, and they differ for one reason: bf16 uses the FIELD-EXACT split (code
# the 8-bit exponent, store sign|mantissa[6:0]) while every other 2-byte dtype
# uses the ``hi`` split (code the high byte, store the low byte).  Reassembly is
# therefore a plain byte concatenation and the bitcast target changes.
#
# This is what closes the hazard on the device side: before 2026-09-05 there was
# ONE kernel that bitcast to bfloat16_t unconditionally, so an fp16 container
# decoded to bf16 numbers WITHOUT raising.
_DECODE_SOURCE_HI = _DECODE_SOURCE.replace(
    """    uchar smb_v = smb[elem];
    ushort word = (ushort(smb_v & 0x80) << 8)
                | (ushort(mexp & 0xFFu) << 7)
                | ushort(smb_v & 0x7F);
    out[elem] = as_type<bfloat16_t>(word);""",
    """    uchar smb_v = smb[elem];
    ushort word = (ushort(mexp & 0xFFu) << 8) | ushort(smb_v);
    out[elem] = as_type<half>(word);""",
)
assert _DECODE_SOURCE_HI != _DECODE_SOURCE, (
    "the bf16 word-reassembly block moved; the fp16 kernel would silently be a "
    "copy of the bf16 one, which is exactly the defect this pair exists to fix"
)

_DECODE_KERNEL_HI = mx.fast.metal_kernel(
    name="tbe_decode_flat64_hi",
    input_names=["planes", "smb", "esc", "tile_base", "base", "w6z", "n_elem"],
    output_names=["out"],
    source=_DECODE_SOURCE_HI,
)

#: dtype code -> (kernel, mlx output dtype).  A code that is not here is
#: refused by name rather than decoded by the bf16 kernel.
_KERNEL_BY_DTYPE = {
    1: (_DECODE_KERNEL, mx.bfloat16),      # DT_BF16, field-exact split
    3: (_DECODE_KERNEL_HI, mx.float16),    # DT_FP16, hi split
}

#: See module docstring "THREADGROUP SHAPE".
DEFAULT_THREADGROUP = 256


def decode_tbe_metal(c: TBEDeviceMLX, threadgroup: int = DEFAULT_THREADGROUP) -> mx.array:
    """MLX-resident TBE container -> ``mx.array`` of ``c.shape`` in ITS dtype.

    One kernel launch, one thread per element.  See the module docstring for
    the escape-rank derivation and the threadgroup-size rationale.
    """
    _enforce_memory_limit()
    code = int(getattr(c, "dtype_code", 1))
    if code not in _KERNEL_BY_DTYPE:
        raise TBEError(
            f"no Metal decode kernel for GLC dtype code {code}; this container "
            "would otherwise be decoded by the bf16 kernel and return the wrong "
            "numbers without raising. Decode on CPU with "
            "experiments.georefine._glc_tbe.decode_tbe, which handles every "
            "2-byte dtype, or add a kernel to _KERNEL_BY_DTYPE."
        )
    kernel, out_dtype = _KERNEL_BY_DTYPE[code]
    n = c.numel
    if n == 0:
        return mx.zeros(c.shape, dtype=out_dtype)
    tg = max(1, min(int(threadgroup), n))

    base_arr = mx.array([int(c.base)], dtype=mx.int32)
    w6z_arr = mx.array([1 if int(c.mode) == MODE_W6Z else 0], dtype=mx.uint32)
    n_arr = mx.array([n], dtype=mx.uint32)

    outputs = kernel(
        inputs=[c.planes, c.smb, c.esc, c.tile_base, base_arr, w6z_arr, n_arr],
        template=[],
        grid=(n, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[out_dtype],
    )
    out = outputs[0].reshape(c.shape)
    mx.eval(out)
    return out


def bits_view(a: mx.array) -> mx.array:
    """Bitcast ``a`` to its same-width unsigned-integer type, for exact compares.

    ``mx.array`` has no numpy bridge for ``bfloat16`` (no ``ml_dtypes`` on
    this machine), so bit-exact assertions go through ``mx.view`` to an
    integer type and compare THOSE -- never ``allclose``, per the spec.
    """
    itemsize = a.itemsize
    target = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}.get(itemsize)
    if target is None:
        raise TBEError(f"no unsigned integer type of width {itemsize} bytes")
    return mx.view(a, target)


__all__ = [
    "DEFAULT_THREADGROUP",
    "METAL_MEMORY_LIMIT_BYTES",
    "TBEDeviceMLX",
    "bits_view",
    "decode_tbe_metal",
    "mlx_metal_available",
    "upload_tbe_mlx",
]
