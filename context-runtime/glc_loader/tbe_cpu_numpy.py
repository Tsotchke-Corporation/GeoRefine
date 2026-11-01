"""CPU NumPy encoder and decoder for the mma16 GLC-TBE layout.

This keeps the public ``TBETensor`` representation while moving the per-lane
plane packing/unpacking work out of Torch's int64 broadcast reductions.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .tbe_container import (
    LANES_PER_WORD,
    MODE_W6Z,
    MODE_W7,
    PLANES,
    SUPERBLOCK,
    TILE,
    TBEError,
    TBETensor,
    _mma16_index,
    _row_chunks,
    choose_window,
)

_BYTE_POPCOUNT = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def _check_weight(weight: torch.Tensor) -> tuple[int, int]:
    if not isinstance(weight, torch.Tensor):
        raise TBEError("expected a torch.Tensor")
    if weight.device.type != "cpu":
        raise TBEError(f"expected a CPU tensor, got {weight.device}")
    if weight.layout != torch.strided:
        raise TBEError(f"expected a strided tensor, got {weight.layout}")
    if weight.dtype != torch.bfloat16:
        raise TBEError(f"expected bfloat16, got {weight.dtype}")
    if weight.ndim != 2:
        raise TBEError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")
    n, k = map(int, weight.shape)
    if n <= 0 or k <= 0 or n % 8 or k % 64:
        raise TBEError(f"mma16 CPU codec needs positive N%8=0 and K%64=0, got {(n, k)}")
    return n, k


def _pack_planes(code_tiles: np.ndarray) -> np.ndarray:
    """Pack [tiles,64] codes into uint32 plane words, lane zero in bit zero."""
    t = code_tiles.shape[0]
    # packbits consumes the lanes directly; each output byte is one 8-lane
    # group.  View the 8 bytes as little-endian uint64s and split into words.
    out = np.empty((t, PLANES, TILE // LANES_PER_WORD), dtype="<u4")
    for b in range(PLANES):
        lane_bits = ((code_tiles >> b) & 1).astype(np.uint8, copy=False)
        packed8 = np.packbits(lane_bits, axis=1, bitorder="little")
        # Each packed byte covers 8 adjacent lanes. Reassemble four bytes
        # per word without an int64 [tiles,32,32] broadcast intermediate.
        words = packed8.reshape(t, 2, 4).astype(np.uint32)
        out[:, b, 0] = (words[:, 0, 0] | (words[:, 0, 1] << 8) |
                        (words[:, 0, 2] << 16) | (words[:, 0, 3] << 24))
        out[:, b, 1] = (words[:, 1, 0] | (words[:, 1, 1] << 8) |
                        (words[:, 1, 2] << 16) | (words[:, 1, 3] << 24))
    return out


def _unpack_planes(planes: np.ndarray) -> np.ndarray:
    t = planes.shape[0]
    out = np.zeros((t, TILE), dtype=np.uint8)
    for b in range(PLANES):
        words = np.asarray(planes[:, b, :], dtype="<u4")
        bytes8 = np.empty((t, 8), dtype=np.uint8)
        for wi in range(2):
            w = words[:, wi]
            bytes8[:, wi * 4 + 0] = w & 0xFF
            bytes8[:, wi * 4 + 1] = (w >> 8) & 0xFF
            bytes8[:, wi * 4 + 2] = (w >> 16) & 0xFF
            bytes8[:, wi * 4 + 3] = (w >> 24) & 0xFF
        out |= np.unpackbits(bytes8, axis=1, bitorder="little") << b
    return out


def _zero_code_counts(planes: np.ndarray) -> np.ndarray:
    """Count zero codes per tile directly from packed words, without unpacking."""
    p = np.asarray(planes, dtype=np.uint32)
    nonzero_mask = p[:, 0, :] | p[:, 1, :] | p[:, 2, :]
    zero_mask = np.bitwise_not(nonzero_mask)
    octets = zero_mask.view(np.uint8).reshape(p.shape[0], 2, 4)
    return _BYTE_POPCOUNT[octets].sum(axis=(1, 2), dtype=np.uint64)


def _raw_uint32(value: torch.Tensor, name: str) -> np.ndarray:
    """Read packed int32 values as raw bits; range-check int64 containers."""
    signed = value.detach().numpy()
    if value.dtype == torch.int32:
        return signed.view(np.uint32)
    if np.any(signed < 0) or np.any(signed > 0xFFFFFFFF):
        raise TBEError(f"{name} word is outside uint32 range")
    return signed.astype(np.uint32, copy=False)


def _exponent_histogram_numpy(weight: torch.Tensor, target_elems: int) -> torch.Tensor:
    """Count raw exponent bits in bounded CPU chunks without Torch int64 expansion."""
    n, k = int(weight.shape[0]), int(weight.shape[1])
    histogram = np.zeros(256, dtype=np.int64)
    for r0, r1 in _row_chunks(n, k, target_elems):
        words = weight[r0:r1].detach().view(torch.int16).numpy().view(np.uint16)
        exponents = ((words >> 7) & 255).reshape(-1)
        histogram += np.bincount(exponents, minlength=256)
    return torch.from_numpy(histogram)


def encode_tbe_numpy(
    weight: torch.Tensor,
    layout: str = "mma16",
    superblock: int = SUPERBLOCK,
    mode: Optional[int] = None,
    base: Optional[int] = None,
    target_elems: int = 1 << 25,
) -> TBETensor:
    """Encode a CPU BF16 ``[N,K]`` tensor using NumPy plane bit packing."""
    n, k = _check_weight(weight)
    if layout != "mma16":
        raise TBEError(f"CPU NumPy codec supports layout='mma16', got {layout!r}")
    if isinstance(superblock, bool) or not isinstance(superblock, int) or superblock <= 0:
        raise TBEError(f"superblock must be a positive integer, got {superblock!r}")
    if isinstance(target_elems, bool) or not isinstance(target_elems, int) or target_elems <= 0:
        raise TBEError(f"target_elems must be a positive integer, got {target_elems!r}")
    tiles = n * k // TILE
    if mode is None or base is None:
        mode, base, _ = choose_window(_exponent_histogram_numpy(weight, target_elems))
    if isinstance(mode, bool) or not isinstance(mode, int) or mode not in (MODE_W7, MODE_W6Z):
        raise TBEError(f"unsupported TBE mode {mode!r}")
    if isinstance(base, bool) or not isinstance(base, int) or not 0 <= base <= 255:
        raise TBEError(f"base must be an integer in [0,255], got {base!r}")
    span = 6 if mode == MODE_W7 else 5

    planes_out = np.empty((tiles, PLANES, TILE // LANES_PER_WORD), dtype="<u4")
    smb_out = np.empty(tiles * TILE, dtype=np.uint8)
    per_tile = np.empty(tiles, dtype=np.uint32)
    esc_parts: list[np.ndarray] = []
    perm = np.asarray(_mma16_index("cpu", inverse=False).numpy())

    chunks = list(_row_chunks(n, k, target_elems))
    t_at = 0
    for r0, r1 in chunks:
        # Viewing as int16 preserves every BF16 bit, including NaN payloads.
        raw = weight[r0:r1].contiguous().view(torch.int16).numpy().view(np.uint16)
        exp = ((raw >> 7) & 0xFF).astype(np.uint8, copy=False).reshape(-1)
        smb = (((raw >> 15) << 7) | (raw & 0x7F)).astype(np.uint8, copy=False).reshape(-1)
        e = exp.reshape(-1, TILE)[:, perm]
        s = smb.reshape(-1, TILE)[:, perm]
        ei = e.astype(np.int16)
        in_win = (ei >= base) & (ei <= base + span)
        codes = np.where(in_win, ei - base + 1, 0).astype(np.uint8)
        if mode == MODE_W6Z:
            codes[e == 0] = 7
        nt = codes.shape[0]
        planes_out[t_at:t_at + nt] = _pack_planes(codes)
        smb_out[t_at * TILE:(t_at + nt) * TILE] = s.reshape(-1)
        escapes_mask = codes == 0
        per_tile[t_at:t_at + nt] = escapes_mask.sum(axis=1)
        if escapes_mask.any():
            esc_parts.append(e[escapes_mask].copy())
        t_at += nt
    if t_at != tiles:
        raise TBEError(f"chunking produced {t_at} tiles, expected {tiles}")

    esc = np.concatenate(esc_parts) if esc_parts else np.zeros(0, dtype=np.uint8)
    if esc.size > 0xFFFFFFFF:
        raise TBEError("more than 2^32 escapes; a uint32 base cannot address them")
    nsb = (tiles + superblock - 1) // superblock
    sbbase = np.zeros(nsb, dtype=np.uint32)
    if nsb > 1:
        sb_counts = np.zeros(nsb, dtype=np.uint64)
        for si in range(nsb):
            lo, hi = si * superblock, min(tiles, (si + 1) * superblock)
            sb_counts[si] = per_tile[lo:hi].sum(dtype=np.uint64)
        sbbase[1:] = np.cumsum(sb_counts[:-1], dtype=np.uint64).astype(np.uint32)

    return TBETensor(
        shape=(n, k), layout="mma16", mode=int(mode), base=int(base),
        tiles=int(tiles), escapes=int(esc.size),
        planes=torch.from_numpy(planes_out.astype(np.int64)),
        smb=torch.from_numpy(smb_out), esc=torch.from_numpy(esc),
        sbbase=torch.from_numpy(sbbase.astype(np.int64)),
        superblock=int(superblock),
    )


def _check_container(c: TBETensor) -> tuple[int, int]:
    if not isinstance(c, TBETensor):
        raise TBEError("expected a TBETensor")
    try:
        n, k = map(int, c.shape)
    except Exception as exc:
        raise TBEError(f"invalid tensor shape {c.shape!r}") from exc
    if n <= 0 or k <= 0 or n % 8 or k % 64 or c.layout != "mma16":
        raise TBEError(f"CPU NumPy decoder needs mma16 with positive N%8=0, K%64=0, got {(c.shape, c.layout)}")
    if isinstance(c.mode, bool) or not isinstance(c.mode, int) or c.mode not in (MODE_W7, MODE_W6Z):
        raise TBEError(f"unsupported TBE mode {c.mode!r}")
    if isinstance(c.base, bool) or not isinstance(c.base, int) or not 0 <= c.base <= 255:
        raise TBEError(f"invalid base {c.base!r}")
    if isinstance(c.superblock, bool) or not isinstance(c.superblock, int) or c.superblock <= 0:
        raise TBEError(f"invalid superblock {c.superblock!r}")
    if isinstance(c.tiles, bool) or not isinstance(c.tiles, int):
        raise TBEError(f"invalid tile count {c.tiles!r}")
    if isinstance(c.escapes, bool) or not isinstance(c.escapes, int):
        raise TBEError(f"invalid escape count {c.escapes!r}")
    tiles = n * k // TILE
    nsb = (tiles + c.superblock - 1) // c.superblock
    fields = (("planes", c.planes, (tiles, PLANES, 2)),
              ("smb", c.smb, (tiles * TILE,)),
              ("esc", c.esc, (c.escapes,)),
              ("sbbase", c.sbbase, (nsb,)))
    for name, value, shape in fields:
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise TBEError(f"{name} must be a CPU tensor")
        if value.layout != torch.strided:
            raise TBEError(f"{name} must be strided")
        if tuple(value.shape) != shape:
            raise TBEError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if c.planes.dtype not in (torch.int32, torch.int64):
        raise TBEError(f"planes must be int32 or int64, got {c.planes.dtype}")
    if c.smb.dtype != torch.uint8 or c.esc.dtype != torch.uint8:
        raise TBEError("smb and esc must be uint8")
    if c.sbbase.dtype not in (torch.int32, torch.int64):
        raise TBEError(f"sbbase must be int32 or int64, got {c.sbbase.dtype}")
    if c.escapes < 0 or c.escapes > 0xFFFFFFFF or c.tiles != tiles:
        raise TBEError("container escape/tile metadata is inconsistent")
    p = _raw_uint32(c.planes, "plane")
    sb = _raw_uint32(c.sbbase, "sbbase")
    running = 0
    validation_chunk_tiles = 65536
    for t0 in range(0, tiles, validation_chunk_tiles):
        t1 = min(tiles, t0 + validation_chunk_tiles)
        zero_counts = _zero_code_counts(p[t0:t1])
        prefix = np.concatenate((np.zeros(1, dtype=np.uint64), np.cumsum(zero_counts, dtype=np.uint64)))
        s0 = (t0 + c.superblock - 1) // c.superblock
        s1 = min(nsb, (t1 + c.superblock - 1) // c.superblock)
        if t0 == 0:
            s0 = 0
        offsets = np.fromiter(
            (si * c.superblock - t0 for si in range(s0, s1)),
            dtype=np.int64, count=max(0, s1 - s0),
        )
        expected = running + prefix[offsets]
        if not np.array_equal(sb[s0:s1], expected):
            raise TBEError("sbbase does not match escape prefixes")
        running += int(zero_counts.sum(dtype=np.uint64))
    if running != c.escapes:
        raise TBEError("escape count does not match zero-code population")
    return n, k


def decode_tbe_numpy(c: TBETensor, target_elems: int = 1 << 25) -> torch.Tensor:
    """Decode an mma16 container to CPU BF16, preserving raw bit patterns."""
    n, k = _check_container(c)
    if isinstance(target_elems, bool) or not isinstance(target_elems, int) or target_elems <= 0:
        raise TBEError(f"target_elems must be a positive integer, got {target_elems!r}")
    out = np.empty(n * k, dtype=np.uint16)
    tiles_per_chunk = max(1, target_elems // TILE)
    planes_all = _raw_uint32(c.planes, "plane")
    smb_all = c.smb.detach().numpy()
    esc_all = c.esc.detach().numpy()
    escapes_before = 0
    inv = np.asarray(_mma16_index("cpu", inverse=True).numpy())
    for t0 in range(0, c.tiles, tiles_per_chunk):
        t1 = min(c.tiles, t0 + tiles_per_chunk)
        codes = _unpack_planes(planes_all[t0:t1])
        exp = np.where(codes > 0, codes.astype(np.int16) - 1 + c.base, 0).astype(np.uint8)
        if c.mode == MODE_W6Z:
            exp[codes == 7] = 0
        escaped = codes == 0
        if escaped.any():
            escaped_flat = escaped.reshape(-1)
            ranks = np.cumsum(escaped_flat, dtype=np.int64) - 1 + escapes_before
            exp.reshape(-1)[escaped_flat] = esc_all[ranks[escaped_flat]]
        escapes_before += int(escaped.sum())
        smb = smb_all[t0 * TILE:t1 * TILE].reshape(-1, TILE).astype(np.uint16)
        word = ((smb & 0x80) << 8) | (exp.astype(np.uint16) << 7) | (smb & 0x7F)
        word = word[:, inv].reshape(-1)
        lo, hi = t0 * TILE, min(t1 * TILE, n * k)
        out[lo:hi] = word[:hi - lo]
    return torch.from_numpy(out.view(np.int16)).view(torch.bfloat16).reshape(n, k)


__all__ = ["decode_tbe_numpy", "encode_tbe_numpy"]
