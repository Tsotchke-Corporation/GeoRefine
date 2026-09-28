"""Reference CPU decoder for the GLC-TBE exact codec, numpy only.

This is an independent, streaming re-statement of the decode side of
``glc_loader.tbe_container.decode_tbe`` and of the Metal reference pack's
``scripts/metal_ref/tbe.py`` (which reproduces the CUDA receipts).  It imports
nothing from either, so an outsider can read it on its own.

FORMAT (per 2-D bf16 tensor W[N, K], elements taken in row-major order)
-----------------------------------------------------------------------
* A tile is 64 consecutive elements of the flattened tensor; there are
  T = ceil(N*K / 64) tiles (the last one zero-padded if N*K % 64 != 0).
* Layout ``mma16`` permutes positions inside a tile: stored position p holds
  original in-tile element PERM[p] with
  PERM[p] = 16*((p >> 2) & 3) + 2*(p >> 4) + (0, 1, 8, 9)[p & 3].
  Layout ``flat64`` stores the tile in order.
* ``planes`` int32 [T, 3, 2] (little-endian): bit (p & 31) of word (p >> 5) of
  plane b is bit b of the 3-bit code of stored position p.
* Code c in 1..7 means exponent ``base + c - 1`` (mode W7).  In mode W6Z codes
  1..6 mean ``base + c - 1`` and code 7 means exponent 0.  Code 0 is an escape:
  the raw 8-bit exponent is the next byte of ``esc`` (tile-major, stored
  position ascending).
* ``smb`` uint8 [T*64] = (sign << 7) | 7-bit mantissa, in stored order.
* ``sbbase`` int32 [ceil(T / superblock)] = escapes before each superblock (an
  index used by the GPU kernels; the decoder here recomputes it and checks it).
* bf16 word = (sign << 15) | (exponent << 7) | mantissa.

Nothing is approximated: every 16-bit pattern (zeros, subnormals, inf, NaN
payloads) is reassembled from its three stored fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Tuple

import numpy as np

TILE = 64
MODE_W7 = 0
MODE_W6Z = 1
LAYOUTS = ("flat64", "mma16")

_OFF = (0, 1, 8, 9)
MMA16_PERM = np.array([16 * ((p >> 2) & 3) + 2 * (p >> 4) + _OFF[p & 3] for p in range(TILE)],
                      dtype=np.intp)
MMA16_INV = np.argsort(MMA16_PERM).astype(np.intp)   # original q -> stored p


class TBEFormatError(ValueError):
    """The stored arrays are not a well-formed TBE tensor."""


@dataclass
class TBEParams:
    n: int
    k: int
    layout: str
    mode: int
    base: int
    superblock: int = 32

    @property
    def numel(self) -> int:
        return self.n * self.k

    @property
    def tiles(self) -> int:
        return -(-self.numel // TILE)

    @property
    def n_superblocks(self) -> int:
        return -(-self.tiles // self.superblock)


def unpack_codes(planes_bytes: bytes | np.ndarray, n_tiles: int) -> np.ndarray:
    """planes (little-endian int32 [T,3,2] as bytes) -> codes uint8 [T, 64]."""
    pl = np.frombuffer(planes_bytes, dtype=np.uint8) if not isinstance(planes_bytes, np.ndarray) \
        else planes_bytes.view(np.uint8)
    if pl.size != n_tiles * 24:
        raise TBEFormatError(f"planes: {pl.size} bytes for {n_tiles} tiles (want {n_tiles * 24})")
    bits = np.unpackbits(pl.reshape(n_tiles, 3, 8), axis=2, bitorder="little")  # [T,3,64]
    return bits[:, 0, :] | (bits[:, 1, :] << 1) | (bits[:, 2, :] << 2)


def decode_tiles(code: np.ndarray, smb: np.ndarray, esc: np.ndarray, p: TBEParams) -> np.ndarray:
    """Decode a run of whole tiles -> bf16 bit patterns uint16 [t, 64] in ORIGINAL order.

    ``esc`` must hold exactly the escapes of these tiles, in order."""
    if p.mode not in (MODE_W7, MODE_W6Z):
        raise TBEFormatError(f"unknown mode {p.mode}")
    if p.layout not in LAYOUTS:
        raise TBEFormatError(f"unsupported layout {p.layout!r}")
    em = code == 0
    if int(em.sum()) != esc.size:
        raise TBEFormatError(f"{int(em.sum())} escape codes but {esc.size} escape bytes")
    e = code.astype(np.int16) + np.int16(p.base - 1)
    if p.mode == MODE_W6Z:
        e[code == 7] = 0
    e[em] = esc.astype(np.int16)
    s = smb.reshape(code.shape).astype(np.uint16)
    word = ((s & 0x80) << 8) | ((e.astype(np.uint16) & 0xFF) << 7) | (s & 0x7F)
    if p.layout == "mma16":
        word = word[:, MMA16_INV]
    return word


ReadFn = Callable[[int, int], bytes]  # (offset_in_array, nbytes) -> bytes


def iter_decode(p: TBEParams, read_planes: ReadFn, read_smb: ReadFn, read_esc: ReadFn,
                sbbase: Optional[np.ndarray], n_esc_total: int,
                chunk_tiles: int = 1 << 16) -> Iterator[Tuple[int, np.ndarray]]:
    """Stream-decode a TBE tensor.  Yields (first_element_index, uint16 words).

    Reads each array piecewise, so memory is bounded by ``chunk_tiles`` whatever
    the tensor size.  Checks, and raises TBEFormatError on: an escape count that
    disagrees with the ``esc`` length, and any ``sbbase`` entry that disagrees
    with the running escape count (when ``sbbase`` is given)."""
    sb = p.superblock
    chunk_tiles = max(sb, (chunk_tiles // sb) * sb)
    T, numel = p.tiles, p.numel
    if sbbase is not None and sbbase.size != p.n_superblocks:
        raise TBEFormatError(f"sbbase has {sbbase.size} entries, want {p.n_superblocks}")
    esc_pos = 0
    for t0 in range(0, T, chunk_tiles):
        t1 = min(T, t0 + chunk_tiles)
        nt = t1 - t0
        code = unpack_codes(read_planes(t0 * 24, nt * 24), nt)
        if sbbase is not None:
            per_tile = (code == 0).sum(axis=1, dtype=np.int64)
            starts = np.concatenate([[0], np.cumsum(per_tile)])[::sb][: -(-nt // sb)] + esc_pos
            want = sbbase[t0 // sb: t0 // sb + starts.size].astype(np.int64)
            if not np.array_equal(starts, want):
                bad = int(np.flatnonzero(starts != want)[0])
                raise TBEFormatError(
                    f"sbbase[{t0 // sb + bad}] = {int(want[bad])}, escapes before it = {int(starts[bad])}")
            n_esc = int(per_tile.sum())
        else:
            n_esc = int((code == 0).sum())
        if esc_pos + n_esc > n_esc_total:
            raise TBEFormatError(f"escape codes exceed the {n_esc_total} stored escape bytes")
        esc = np.frombuffer(read_esc(esc_pos, n_esc), dtype=np.uint8) if n_esc else \
            np.zeros(0, np.uint8)
        esc_pos += n_esc
        smb = np.frombuffer(read_smb(t0 * TILE, nt * TILE), dtype=np.uint8)
        words = decode_tiles(code, smb, esc, p).reshape(-1)
        e0 = t0 * TILE
        yield e0, words[: min(numel, t1 * TILE) - e0]
    if esc_pos != n_esc_total:
        raise TBEFormatError(f"{n_esc_total} escape bytes stored, {esc_pos} used")


# ------------------------------------------------------------------ encoder (tests only)
def choose_window(exp: np.ndarray) -> Tuple[int, int]:
    hist = np.bincount(exp.reshape(-1).astype(np.int64), minlength=256)[:256]
    total, zero_ct = int(hist.sum()), int(hist[0])

    def best(width, exclude_zero):
        h = hist.copy()
        if exclude_zero:
            h[0] = 0
        c = np.cumsum(h)
        lo = np.concatenate([[0], c[:-1]])
        starts = 256 - width + 1
        sums = c[width - 1: width - 1 + starts] - lo[:starts]
        b = int(np.argmax(sums))
        return b, int(sums[b])

    b7, cov7 = best(7, False)
    b6, cov6 = best(6, True)
    if total - cov6 - zero_ct < total - cov7:
        return MODE_W6Z, b6
    return MODE_W7, b7


def encode(w_bits: np.ndarray, layout: str = "mma16", mode: Optional[int] = None,
           base: Optional[int] = None, superblock: int = 32):
    """bf16 bits uint16 [N, K] -> (params, planes int32 [T,3,2], smb, esc, sbbase).
    Used by the tests to build containers; the verifier never encodes."""
    w = np.asarray(w_bits, dtype=np.uint16)
    n, k = w.shape
    flat = w.reshape(-1).astype(np.int64)
    T = -(-flat.size // TILE)
    flat = np.concatenate([flat, np.zeros(T * TILE - flat.size, np.int64)])
    exp = (flat >> 7) & 0xFF
    smb = (((flat >> 15) & 1) << 7) | (flat & 0x7F)
    if mode is None or base is None:
        mode, base = choose_window(exp)
    span = 6 if mode == MODE_W7 else 5
    e = exp.reshape(T, TILE)
    s = smb.reshape(T, TILE)
    if layout == "mma16":
        e, s = e[:, MMA16_PERM], s[:, MMA16_PERM]
    code = np.where((e >= base) & (e <= base + span), e - base + 1, 0)
    if mode == MODE_W6Z:
        code = np.where(e == 0, 7, code)
    planes = np.zeros((T, 3, 2), dtype=np.uint64)
    for b in range(3):
        bitv = ((code >> b) & 1).astype(np.uint64).reshape(T, 2, 32)
        planes[:, b, :] = (bitv << np.arange(32, dtype=np.uint64)).sum(axis=2)
    esc = e[code == 0].astype(np.uint8)
    per_tile = (code == 0).sum(axis=1)
    csum = np.concatenate([[0], np.cumsum(per_tile)])
    sbbase = csum[::superblock][: -(-T // superblock)].astype(np.int32)
    p = TBEParams(n=n, k=k, layout=layout, mode=int(mode), base=int(base), superblock=superblock)
    return p, planes.astype(np.uint32).view(np.int32), s.astype(np.uint8).reshape(-1), esc, sbbase
