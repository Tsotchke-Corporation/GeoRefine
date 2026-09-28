"""Random access into a TBE container: decode the tiles a gather touches.

An embedding is a GATHER, not a matmul.  The fragment kernel path in
``tbe_mma.py`` exists because a linear weight is multiplied in full on every
token, so the whole tensor has to be resident and decodable at kernel speed.
An embedding table is the opposite: a forward pass reads ``len(unique(ids))``
rows out of a table with tens or hundreds of thousands of them, and every
other row is dead weight for that step.  That is why the transcoder's
``_NON_LINEAR_NAME_HINTS`` exclusion -- embeddings kept raw at full bf16 --
costs the artifact real bytes for nothing: the codec can serve those rows, it
just needs to be asked for a range instead of for the whole tensor.

This module adds no container format and changes none.  ``decode_tbe`` in
``tbe_container.py`` already decodes a chunk that STARTS AT AN ARBITRARY TILE:
it seeds the escape-rank cursor from ``sbbase[t0 // superblock]`` and then
unpacks the at most ``superblock - 1`` lead tiles of that superblock to finish
the count.  Random access is therefore already implied by the shipped layout;
what was missing was an entry point that asks for it.  Everything here is that
entry point, plus the coalescing that turns a batch of token ids into one
range decode per contiguous run of rows rather than one per row.

SCOPE.  ``flat64`` and ``mma16`` only.  Both tile the FLATTENED tensor 64
elements at a time (``n_tiles_for`` is identical for the two, and ``mma16``
differs only by an in-tile permutation), so tile ``t`` holds flat elements
``[64t, 64t + 64)`` and a row of a ``[N, K]`` table is a flat range.
``tile8x8`` interleaves eight rows into every tile, so a row is not a
contiguous tile range there and this module refuses it rather than silently
decoding eight times the data.

Standard library plus ``torch`` plus this package only: it ships inside the
artifact like the rest of ``glc_loader``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .tbe_container import (
    MODE_W6Z,
    TILE,
    TBEError,
    TBETensor,
    _mma16_index,
    _unpack_planes,
)

_FLAT_LAYOUTS = ("flat64", "mma16")


def _require_flat(c: TBETensor) -> None:
    if c.layout not in _FLAT_LAYOUTS:
        raise TBEError(
            f"layout {c.layout!r} is not flat-contiguous; range decode needs "
            f"one of {_FLAT_LAYOUTS} (a tile8x8 tile interleaves eight rows, "
            "so a row range is not a tile range)"
        )


def _decode_tile_span(
    c: TBETensor,
    t0: int,
    t1: int,
    *,
    target_elems: int = 1 << 25,
    stats: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    """Raw 16-bit words for tiles ``[t0, t1)``, in flat element order.

    A transcription of ``decode_tbe``'s per-chunk body with the chunk bounds
    taken from the caller instead of from a full sweep.  The escape cursor is
    seeded exactly as it is there -- superblock base plus a lead scan -- which
    is the whole reason an arbitrary ``t0`` is legal.
    """
    _require_flat(c)
    if not (0 <= t0 <= t1 <= c.tiles):
        raise TBEError(f"tile span [{t0}, {t1}) outside [0, {c.tiles})")
    device = c.planes.device
    out = torch.empty((t1 - t0) * TILE, dtype=torch.int16, device=device)
    per_chunk = max(1, int(target_elems) // TILE)
    for a in range(t0, t1, per_chunk):
        b = min(t1, a + per_chunk)
        code = _unpack_planes(c.planes[a:b])
        exp = torch.where(code > 0, code - 1 + c.base, torch.zeros_like(code))
        if c.mode == MODE_W6Z:
            exp = torch.where(code == 7, torch.zeros_like(exp), exp)

        em = code == 0
        if bool(em.any()):
            before = int(c.sbbase[a // c.superblock].item())
            sb_a = (a // c.superblock) * c.superblock
            if sb_a < a:
                lead = _unpack_planes(c.planes[sb_a:a])
                before += int((lead == 0).sum().item())
                if stats is not None:
                    stats["lead_tiles_scanned"] += a - sb_a
            flat = em.reshape(-1)
            rank = torch.cumsum(flat.to(torch.int64), dim=0) - 1 + before
            expf = exp.reshape(-1)
            expf[flat] = c.esc[rank[flat]].to(torch.int64)
            exp = expf.reshape(code.shape)

        smb = c.smb[a * TILE:b * TILE].to(torch.int64).reshape(code.shape)
        word = ((smb & 0x80) << 8) | ((exp & 0xFF) << 7) | (smb & 0x7F)
        if c.layout == "mma16":
            word = word[:, _mma16_index(word.device, inverse=True)]
        word = word.reshape(-1)
        out[(a - t0) * TILE:(b - t0) * TILE] = (
            (word & 0xFFFF).to(torch.int32).to(torch.int16)
        )
        if stats is not None:
            stats["n_tiles_decoded"] += b - a
    return out


def decode_tbe_range(
    c: TBETensor,
    elem0: int,
    elem1: int,
    *,
    target_elems: int = 1 << 25,
    stats: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    """Elements ``[elem0, elem1)`` of the flattened ``[N, K]`` tensor, bf16.

    Decodes exactly the tiles ``[elem0 // 64, ceil(elem1 / 64))`` plus the
    escape-rank lead scan of the first tile's superblock -- never the tensor.
    """
    _require_flat(c)
    elem0, elem1 = int(elem0), int(elem1)
    if elem0 < 0:
        raise TBEError(f"elem0 must be non-negative, got {elem0}")
    if elem0 > elem1:
        raise TBEError(f"empty or reversed range [{elem0}, {elem1})")
    if elem1 > c.numel:
        raise TBEError(
            f"range [{elem0}, {elem1}) runs past the tensor's {c.numel} elements"
        )
    if elem0 == elem1:
        return torch.empty(0, dtype=torch.bfloat16, device=c.planes.device)
    t0 = elem0 // TILE
    t1 = (elem1 + TILE - 1) // TILE
    words = _decode_tile_span(
        c, t0, t1, target_elems=target_elems, stats=stats,
    )
    lo = elem0 - t0 * TILE
    return words.view(torch.bfloat16)[lo:lo + (elem1 - elem0)]


def _runs(sorted_rows: torch.Tensor):
    """Ascending contiguous runs of a sorted, unique 1-D int64 tensor.

    Yields ``(i, j)`` index pairs into ``sorted_rows`` such that
    ``sorted_rows[i:j]`` is ``range(sorted_rows[i], sorted_rows[i] + j - i)``.
    One run is one range decode; without this a batch of 512 consecutive token
    ids would pay 512 lead scans instead of one.
    """
    n = int(sorted_rows.numel())
    if n == 0:
        return
    if n == 1:
        yield 0, 1
        return
    step = sorted_rows[1:] - sorted_rows[:-1]
    breaks = torch.nonzero(step != 1, as_tuple=False).reshape(-1)
    start = 0
    for b in breaks.tolist():
        yield start, b + 1
        start = b + 1
    yield start, n


def decode_tbe_rows(
    c: TBETensor,
    rows: torch.Tensor,
    *,
    target_elems: int = 1 << 25,
    stats: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    """Rows ``rows`` of the coded ``[N, K]`` tensor, in the caller's order.

    Duplicates and arbitrary order are allowed: the rows are made unique and
    sorted, coalesced into ascending contiguous runs, decoded one run at a
    time, and scattered back onto the caller's ordering.
    """
    _require_flat(c)
    n, k = int(c.shape[0]), int(c.shape[1])
    idx = torch.as_tensor(rows)
    if idx.dim() != 1:
        raise TBEError(f"rows must be 1-D, got shape {tuple(idx.shape)}")
    if idx.dtype not in (torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TBEError(f"rows must be an integer tensor, got {idx.dtype}")
    idx = idx.to(torch.int64)
    device = c.planes.device
    if int(idx.numel()) == 0:
        return torch.empty((0, k), dtype=torch.bfloat16, device=device)
    lo = int(idx.min().item())
    hi = int(idx.max().item())
    if lo < 0 or hi >= n:
        raise TBEError(f"row index out of range: [{lo}, {hi}] not within [0, {n})")

    uniq, inverse = torch.unique(idx, sorted=True, return_inverse=True)
    gathered = torch.empty((int(uniq.numel()), k), dtype=torch.bfloat16,
                           device=device)
    for i, j in _runs(uniq):
        r0 = int(uniq[i].item())
        r1 = int(uniq[j - 1].item()) + 1
        span = decode_tbe_range(
            c, r0 * k, r1 * k, target_elems=target_elems, stats=stats,
        )
        gathered[i:j] = span.reshape(j - i, k)
    return gathered[inverse.to(device)]


def tbe_embedding_lookup(
    c: TBETensor,
    token_ids: torch.Tensor,
    *,
    target_elems: int = 1 << 25,
    stats: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    """``F.embedding(token_ids, table)`` with the table left coded.

    Returns ``token_ids.shape + (K,)`` bf16, bitwise identical to the dense
    gather: only the rows this call touches are ever decoded, and each is
    decoded once however many times it appears.
    """
    ids = torch.as_tensor(token_ids)
    flat = ids.reshape(-1)
    uniq, inverse = torch.unique(flat.to(torch.int64), sorted=True,
                                 return_inverse=True)
    table = decode_tbe_rows(c, uniq, target_elems=target_elems, stats=stats)
    out = table[inverse.to(table.device)]
    return out.reshape(*tuple(ids.shape), int(c.shape[1]))


def tiles_touched(
    c: TBETensor,
    rows: torch.Tensor,
    *,
    target_elems: int = 1 << 25,
) -> Dict[str, Any]:
    """What a row gather actually costs, in tiles.

    The counts are MEASURED by running the gather with a stats collector, not
    projected: ``lead_tiles_scanned`` in particular depends on whether the
    decoded span holds an escape at all, which is only knowable by unpacking
    the planes.  ``fraction`` is the decoded tiles over the container's tiles;
    ``fraction_including_lead`` charges the lead scan as well.
    """
    stats = {"n_tiles_decoded": 0, "lead_tiles_scanned": 0}
    decode_tbe_rows(c, rows, target_elems=target_elems, stats=stats)
    total = int(c.tiles)
    dec = int(stats["n_tiles_decoded"])
    lead = int(stats["lead_tiles_scanned"])
    return {
        "n_tiles_decoded": dec,
        "n_tiles_total": total,
        "lead_tiles_scanned": lead,
        "fraction": (dec / total) if total else 0.0,
        "fraction_including_lead": ((dec + lead) / total) if total else 0.0,
    }


__all__ = [
    "decode_tbe_range",
    "decode_tbe_rows",
    "tbe_embedding_lookup",
    "tiles_touched",
]
