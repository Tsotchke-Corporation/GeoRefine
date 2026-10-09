"""GLC-FWP1 container: the decode side, with no dependency on the build repo.

This file is vendored verbatim into every GLC-RELEASE artifact.  It must import
nothing but the standard library, ``torch`` and (optionally) ``triton``.  A
client machine has neither this repository nor the original checkpoint, so any
import that reaches back into ``experiments.georefine`` is a shipping defect,
not a convenience -- ``tests/test_glc_release.py`` asserts the absence.

CONTAINER LAYOUT (GLC-FWP1, per 2-D bf16 tensor ``W[N, K]``, row-major)
-----------------------------------------------------------------------
Groups are contiguous runs of ``G`` elements along ``K`` within one row, so one
group is exactly one inner-loop chunk of a GEMV over the reduction axis.
``K % G == 0`` is required.

  smb  : uint8[N, K]            (sign << 7) | mantissa        -- 8 bits/elem
  eidx : uint8[N, K // 2]       two 4-bit palette indices     -- 4 bits/elem
                                low nibble = even k, high = odd k
  pal  : uint8[N * K // G, 16]  per-group exponent palette,
                                ascending, tail-padded        -- 128/G bits/elem

Total = 8 + 4 + 128/G bits/element.  G=128 -> 13.0 bits/elem (1.2308x vs bf16);
G=256 -> 12.5 bits/elem (1.28x).

Reconstruction is exact for every bf16 bit pattern -- zero, denormals, Inf and
NaN payloads included -- because the container stores the three fields of the
word and reassembles them, rather than storing an approximation of its value.
There is no tolerance anywhere in this file.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

PALETTE_ENTRIES = 16
DEFAULT_GROUP = 128
CONTAINER_MAGIC = "GLC-FWP1"

_BYTEORDER = sys.byteorder


class FWP1Error(RuntimeError):
    """A tensor cannot be represented exactly by the container."""


@dataclass
class FWP1Tensor:
    """A bitwise-lossless, randomly-addressable bf16 weight."""

    smb: torch.Tensor    # uint8 [N, K]
    eidx: torch.Tensor   # uint8 [N, K//2]
    pal: torch.Tensor    # uint8 [N*K//G, 16]
    shape: Tuple[int, int]
    group: int

    @property
    def resident_bytes(self) -> int:
        return (
            self.smb.numel() * self.smb.element_size()
            + self.eidx.numel() * self.eidx.element_size()
            + self.pal.numel() * self.pal.element_size()
        )

    @property
    def original_bytes(self) -> int:
        return int(self.shape[0]) * int(self.shape[1]) * 2

    @property
    def ratio(self) -> float:
        return self.original_bytes / self.resident_bytes

    @property
    def bits_per_element(self) -> float:
        return 8.0 * self.resident_bytes / (int(self.shape[0]) * int(self.shape[1]))

    def to(self, device) -> "FWP1Tensor":
        return FWP1Tensor(
            smb=self.smb.to(device, non_blocking=False),
            eidx=self.eidx.to(device, non_blocking=False),
            pal=self.pal.to(device, non_blocking=False),
            shape=self.shape,
            group=self.group,
        )

    def row_slice(self, r0: int, r1: int) -> "FWP1Tensor":
        """Rows are independent, so a row range is itself a container.

        ``smb`` and ``eidx`` are row-major ``[N, *]``; ``pal`` carries ``K // G``
        consecutive groups per row.  Slicing all three on the same row range
        keeps the group ids self-consistent, which is what lets a whole-tensor
        round-trip check run in bounded memory at any size.
        """
        gk = int(self.shape[1]) // int(self.group)
        return FWP1Tensor(
            smb=self.smb[r0:r1],
            eidx=self.eidx[r0:r1],
            pal=self.pal[r0 * gk:r1 * gk],
            shape=(int(r1) - int(r0), int(self.shape[1])),
            group=int(self.group),
        )


def _split_bf16(weight: torch.Tensor):
    """bf16 tensor -> (sign, exponent, mantissa) fields as int32 codes."""
    if weight.dtype != torch.bfloat16:
        raise FWP1Error(f"expected bfloat16, got {weight.dtype}")
    bits = weight.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    return (bits >> 15) & 0x1, (bits >> 7) & 0xFF, bits & 0x7F


def encode_fwp1(weight: torch.Tensor, group: int = DEFAULT_GROUP) -> FWP1Tensor:
    """Encode a 2-D bf16 weight into the FWP1 container.  Exact, or it raises.

    Carried in the loader as well as the builder so a client can re-encode a
    tensor they decoded and confirm byte-identity against the shipped planes
    without trusting our recorded digests.  ``verify --deep`` does exactly that.
    """
    if weight.dim() != 2:
        raise FWP1Error(f"expected a 2-D weight, got shape {tuple(weight.shape)}")
    n, k = int(weight.shape[0]), int(weight.shape[1])
    if k % group != 0:
        raise FWP1Error(
            f"in_features {k} is not a multiple of group {group}; FWP1 groups "
            "run along the reduction axis and must tile it exactly"
        )
    dev = weight.device
    sign, exp, mant = _split_bf16(weight)

    eg = exp.reshape(-1, group)
    num_groups = int(eg.shape[0])
    present = torch.zeros((num_groups, 256), dtype=torch.bool, device=dev)
    present.scatter_(1, eg.to(torch.int64), True)
    counts = present.sum(dim=1)
    worst = int(counts.max().item()) if num_groups else 0
    if worst > PALETTE_ENTRIES:
        raise FWP1Error(
            f"group holds {worst} distinct exponents (> {PALETTE_ENTRIES}); "
            f"FWP1 cannot represent this tensor exactly at group={group}"
        )

    ranks = torch.cumsum(present.to(torch.int32), dim=1) - 1
    idx = torch.gather(ranks, 1, eg.to(torch.int64)).to(torch.uint8)

    pal = torch.zeros((num_groups, PALETTE_ENTRIES), dtype=torch.uint8, device=dev)
    grow, gval = torch.nonzero(present, as_tuple=True)
    pal[grow, ranks[grow, gval].to(torch.int64)] = gval.to(torch.uint8)
    last = (counts - 1).clamp(min=0).to(torch.int64)
    slot = torch.arange(PALETTE_ENTRIES, device=dev).unsqueeze(0)
    padmask = slot >= counts.unsqueeze(1)
    pal = torch.where(padmask, pal.gather(1, last.unsqueeze(1)).expand_as(pal), pal)

    smb = ((sign << 7) | mant).to(torch.uint8).reshape(n, k).contiguous()
    idx_flat = idx.reshape(n, k)
    eidx = (idx_flat[:, 0::2] | (idx_flat[:, 1::2] << 4)).contiguous()
    return FWP1Tensor(smb=smb, eidx=eidx, pal=pal.contiguous(),
                      shape=(n, k), group=int(group))


#: Bytes of scratch the decode may hold live per chunk, per element decoded.
#:
#: Counted, not estimated.  At the peak of :func:`_decode_block` the live set is
#: the unpacked nibble plane (1), the int32 gather index (4) and the gathered
#: exponent byte (1) = 6 bytes; afterwards it is the exponent (1), the two
#: assembled bytes (2, written in place into the output) and nothing else.  Six
#: is therefore the number a chunk size has to be divided by, and it is written
#: here rather than inlined so that a change to the decode which invalidates it
#: is a change to this constant.
_SCRATCH_BYTES_PER_ELEMENT = 6

#: Default ceiling on that live scratch, per decode call.
#:
#: This used to be a 2**23 ELEMENT budget, which is a byte budget only if you
#: know how many bytes an element costs -- and the old decode cost about 24, so
#: the real ceiling was ~200 MB and nobody had written that down.  Measured on
#: an RTX PRO 6000 with Llama-3.2-1B: the `resident` backend held 710.0 MiB of
#: transient to save 514.6 MiB of weights, so its peak came out 154.8 MiB ABOVE
#: the dense model it was meant to shrink.  209 MiB of that was this constant.
#:
#: 32 MiB is a throughput/footprint trade and nothing else: the decode is
#: exactly as exact at any chunk size, because rows of this container are
#: independent, so a smaller chunk only costs kernel launches.
DEFAULT_DECODE_SCRATCH_BYTES = 32 << 20


def _decode_block(t: FWP1Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Decode one whole container into ``out`` (allocated if not supplied).

    Written to hold as little live scratch as the operation admits, because the
    caller's peak memory is this function's live set plus the output:

    * the palette gather is an ``index_select`` with an **int32** index rather
      than a ``gather`` with an int64 one.  ``torch.gather`` requires int64, so
      it costs 8 bytes per element of pure addressing; ``index_select`` accepts
      int32 and costs 4.  Same values, same order, half the plane.
    * the bf16 word is assembled as its two BYTES rather than widened to int32.
      ``(sign<<15)|(exp<<7)|mant`` splits cleanly on the byte boundary --
      ``lo = ((exp & 1) << 7) | mant`` and ``hi = (sign << 7) | (exp >> 1)`` --
      and both halves are uint8-valued at every step, so nothing ever widens
      and no shift can overflow.  The previous form went through five int32
      temporaries the size of the tensor.
    * the two bytes are written straight into a uint8 view of the output, so
      the assembled word is never a separate allocation.

    Nothing here is a numerical choice.  Every operation is integer bit
    manipulation on fields that were stored, not computed, which is why the
    container reproduces signed zero, every denormal, both infinities and all
    256 NaN payloads rather than merely comparing equal to them.
    """
    if _BYTEORDER != "little":  # pragma: no cover - no big-endian target exists
        raise FWP1Error(
            "the byte-assembled decode writes a bf16 word low byte first and "
            f"this machine is {_BYTEORDER}-endian; refusing to produce weights "
            "whose bytes would be transposed rather than fail"
        )
    n, k = int(t.shape[0]), int(t.shape[1])
    g = int(t.group)
    gk = k // g
    dev = t.smb.device
    if out is None:
        out = torch.empty((n, k), dtype=torch.bfloat16, device=dev)

    # 4-bit palette indices, one byte each.
    idx = torch.empty((n, k), dtype=torch.uint8, device=dev)
    idx[:, 0::2] = t.eidx & 0x0F
    idx[:, 1::2] = t.eidx >> 4

    # The group id is implicit in the [n, gk, g] shape, so the only thing that
    # has to be materialised per element is the flat palette offset.  The base
    # is [n, gk, 1] -- one int32 per GROUP, not per element -- and broadcasts.
    base = (torch.arange(n * gk, device=dev, dtype=torch.int32) * PALETTE_ENTRIES)
    flat_index = base.reshape(n, gk, 1) + idx.reshape(n, gk, g).to(torch.int32)
    del idx
    exp = torch.index_select(
        t.pal.reshape(-1), 0, flat_index.reshape(-1),
    ).reshape(n, k)
    del flat_index, base

    # bf16 word = sign<<15 | exp<<7 | mant, assembled as two uint8 lanes.
    ob = out.view(torch.uint8).reshape(n, k, 2)
    torch.bitwise_or(
        (exp & 0x01) << 7, t.smb & 0x7F, out=ob[..., 0],
    )
    torch.bitwise_or(t.smb & 0x80, exp >> 1, out=ob[..., 1])
    return out


def decode_fwp1(
    t: FWP1Tensor,
    *,
    scratch_bytes: int = DEFAULT_DECODE_SCRATCH_BYTES,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Exact decode back to bf16.  Pure torch: runs on CPU, MPS, CUDA alike.

    Decoding is chunked over rows so live scratch stays under ``scratch_bytes``
    regardless of tensor size.  The result is independent of the chunk size --
    rows are independent in this container -- so chunking is a memory control
    and never a source of numerical difference.  ``test_container.py`` asserts
    that directly rather than leaving it as a claim.

    ``out`` lets a caller supply the destination, which is what keeps the
    output out of the transient budget when the same buffer is reused.
    """
    n, k = int(t.shape[0]), int(t.shape[1])
    dev = t.smb.device
    if out is None:
        out = torch.empty((n, k), dtype=torch.bfloat16, device=dev)
    elif tuple(out.shape) != (n, k) or out.dtype != torch.bfloat16:
        raise FWP1Error(
            f"out has shape {tuple(out.shape)} dtype {out.dtype}; the container "
            f"decodes to {(n, k)} bfloat16"
        )
    if n == 0:
        return out
    per_row = max(1, k * _SCRATCH_BYTES_PER_ELEMENT)
    rows = max(1, min(n, int(scratch_bytes) // per_row))
    if rows >= n:
        return _decode_block(t, out=out)
    for r0 in range(0, n, rows):
        r1 = min(n, r0 + rows)
        _decode_block(t.row_slice(r0, r1), out=out[r0:r1])
    return out


def decode_fwp1_rows(t: FWP1Tensor, row_ids: torch.Tensor) -> torch.Tensor:
    """Gather and decode an arbitrary set of rows without touching the rest.

    This is what makes the container usable for an embedding: a token lookup
    decodes ``len(row_ids)`` rows, not the vocabulary.
    """
    ids = row_ids.reshape(-1).to(device=t.smb.device, dtype=torch.int64)
    gk = int(t.shape[1]) // int(t.group)
    pal = t.pal.reshape(-1, gk, PALETTE_ENTRIES)[ids].reshape(-1, PALETTE_ENTRIES)
    sub = FWP1Tensor(
        smb=t.smb[ids], eidx=t.eidx[ids], pal=pal,
        shape=(int(ids.numel()), int(t.shape[1])), group=int(t.group),
    )
    return decode_fwp1(sub)


def fwp1_certify(
    t: FWP1Tensor,
    original: torch.Tensor,
    chunk_elements: int = 1 << 22,
) -> dict:
    """Bitwise round-trip over EVERY element of ``original``.  Never skips.

    A size threshold on a correctness check is indistinguishable, in the
    receipt, from the check having passed; this project shipped exactly that
    once.  So this verifies in row chunks sized to bound peak memory rather
    than declining to verify, and returns ``elements_checked`` for the caller
    to compare against ``N * K``.  ``fully_checked`` is that comparison made
    explicit, so a partial check can never read as a complete one.

    Compares raw 16-bit patterns, so NaN payloads, signed zero and denormals
    must match exactly rather than merely compare equal.
    """
    n, k = int(t.shape[0]), int(t.shape[1])
    if tuple(original.shape) != (n, k):
        raise FWP1Error(
            f"original has shape {tuple(original.shape)}, container has {(n, k)}"
        )
    ref = original.detach().to(t.smb.device).contiguous()
    rows_per_chunk = max(1, min(n, int(chunk_elements) // max(1, k)))
    mismatched = 0
    checked = 0
    for r0 in range(0, n, rows_per_chunk):
        r1 = min(n, r0 + rows_per_chunk)
        decoded = decode_fwp1(t.row_slice(r0, r1))
        a = ref[r0:r1].contiguous().view(torch.int16)
        b = decoded.contiguous().view(torch.int16)
        mismatched += int((a != b).sum().item())
        checked += (r1 - r0) * k
    return {
        "elements_checked": int(checked),
        "elements_total": n * k,
        "mismatched_elements": int(mismatched),
        "bitwise_ok": mismatched == 0,
        "fully_checked": checked == n * k,
    }


# ---------------------------------------------------------------------------
# Optional compute-in-domain kernels
# ---------------------------------------------------------------------------
def load_kernels():
    """Return the vendored Triton kernel module, or ``None``.

    The kernels are a speed/VRAM optimisation, never a correctness dependency:
    every path in this package produces the same bits with or without them.
    """
    try:
        import triton  # noqa: F401
    except Exception:
        return None
    if not (torch.cuda.is_available()):
        return None
    try:
        from . import fwp1_kernels  # type: ignore
    except Exception:
        return None
    return fwp1_kernels


__all__ = [
    "CONTAINER_MAGIC",
    "DEFAULT_GROUP",
    "PALETTE_ENTRIES",
    "FWP1Error",
    "FWP1Tensor",
    "decode_fwp1",
    "decode_fwp1_rows",
    "encode_fwp1",
    "fwp1_certify",
    "load_kernels",
]
