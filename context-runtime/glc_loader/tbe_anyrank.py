"""Code a tensor of any rank: the 2-D requirement is the caller's, not the codec's.

``encode_tbe`` raises on ``weight.dim() != 2``, and that single check is the
only thing between the TBE codec and every 1-D and rank >= 3 tensor in a
checkpoint.  The codec itself has no opinion about shape: with layout
``flat64`` it tiles the FLATTENED buffer 64 elements at a time
(``n_tiles_for(n, k, "flat64") == ceil(n * k / 64)``, no shape precondition),
splits each bf16 word into sign, exponent and mantissa, and reassembles it.
Reinterpreting the same contiguous row-major buffer as ``[rows, last_dim]``
does not move a byte, so the round trip stays bit-exact by construction --
this is the same argument the transcoder's ``--nd-policy code`` already makes
for the fused MoE expert stacks, generalised.

WHAT THIS IS NOT.  It is not a dtype widening.  TBE is a bf16 bit-field
codec: it stores the 8-bit exponent through a 3-bit window and the sign plus
7-bit mantissa verbatim.  float32, float16 and the integer types have
different field widths and are NOT in scope; ``encode_any`` refuses them by
name rather than casting, because a cast would silently ship a lossy
container from a codec whose entire claim is that it is lossless.

Standard library plus ``torch`` plus this package only.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from .tbe_container import TBEError, TBETensor, decode_tbe, encode_tbe


def coded_2d_view(shape: Sequence[int]) -> Optional[Tuple[int, int]]:
    """The 2-D shape the codec sees for ``shape``, or ``None`` if there is none.

    ``[L]`` -> ``(1, L)``; ``[N, K]`` -> itself; ``[a, b, ..., z]`` ->
    ``(a * b * ... , z)``.  A rank-0 tensor holds one element and is viewed as
    ``(1, 1)``.  An empty tensor (any dimension zero) has no coded view: there
    is nothing to tile, and a zero-tile container would only be a way to
    record that fact twice.
    """
    dims = [int(d) for d in tuple(shape)]
    if any(d == 0 for d in dims):
        return None
    if not dims:
        return 1, 1
    if len(dims) == 1:
        return 1, dims[0]
    rows = 1
    for d in dims[:-1]:
        rows *= d
    return rows, dims[-1]


def encode_any(t: torch.Tensor, layout: str = "flat64", **kw) -> TBETensor:
    """Encode a bf16 tensor of any rank through :func:`coded_2d_view`.

    ``layout`` defaults to ``flat64`` and not to the artifact's ``mma16``:
    a tensor that needed this function is by definition not a fragment-kernel
    linear weight, so there is nothing to be gained from the kernel's in-tile
    permutation and the container will be decoded to dense at load.
    """
    if not isinstance(t, torch.Tensor):
        raise TBEError(f"expected a torch.Tensor, got {type(t).__name__}")
    if t.dtype != torch.bfloat16:
        raise TBEError(
            f"TBE is a bf16 bit-field codec and {t.dtype} is not in scope; "
            "casting would make a lossless container lossy, so this refuses "
            "rather than converting"
        )
    cs = coded_2d_view(t.shape)
    if cs is None:
        raise TBEError(
            f"shape {tuple(t.shape)} has no coded 2-D view (it holds no "
            "elements)"
        )
    flat = t.contiguous()
    if tuple(flat.shape) != cs:
        flat = flat.reshape(cs)
    return encode_tbe(flat, layout=layout, **kw)


def decode_any(c: TBETensor, original_shape: Sequence[int]) -> torch.Tensor:
    """Decode and restore ``original_shape``.  Bitwise exact, by construction."""
    dims = [int(d) for d in tuple(original_shape)]
    want = 1
    for d in dims:
        want *= d
    if want != c.numel:
        raise TBEError(
            f"shape {tuple(dims)} holds {want} element(s), the container holds "
            f"{c.numel}"
        )
    return decode_tbe(c).reshape(dims)


def certify_any(c: TBETensor, original: torch.Tensor) -> dict:
    """Bitwise round-trip check at the ORIGINAL rank, never a tolerance.

    ``tbe_certify`` compares at the container's 2-D shape and ``torch.equal``
    is False for equal bytes in different shapes, so an any-rank container
    needs its comparison done at the shape the caller handed in.
    """
    decoded = decode_any(c, original.shape)
    a = original.detach().contiguous().view(torch.int16).reshape(-1)
    b = decoded.contiguous().view(torch.int16).reshape(-1)
    mismatched = int((a != b).sum().item())
    return {
        "elements_checked": int(a.numel()),
        "elements_total": int(a.numel()),
        "mismatched_elements": mismatched,
        "bitwise_ok": mismatched == 0,
        "fully_checked": True,
    }


__all__ = ["certify_any", "coded_2d_view", "decode_any", "encode_any"]
