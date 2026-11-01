"""GLC-TBE container: the decode side, with no dependency on the build repo.

This file is the static, source-controlled twin of
``experiments.georefine._glc_tbe`` -- vendored verbatim into every GLC-RELEASE
artifact that ships the lossless TBE line, the same way ``container.py`` is
vendored for FWP1.  It must import nothing but the standard library and
``torch``.  A client machine has neither this repository nor the original
checkpoint, so any import that reaches back into ``experiments.georefine`` is a
shipping defect, not a convenience -- ``tests/test_glc_release.py`` asserts the
absence for every ``*.py`` in this package, and this file is one of them.

CONTAINER LAYOUT (GLC-TBE, per 2-D bf16 tensor ``W[N, K]``, row-major)
------------------------------------------------------------------------
A 3-bit fixed-length code over a contiguous exponent window, tiled 64
elements at a time (one machine word per bitmap plane).  Codes ``1..7`` (or
``1..6`` in mode ``W6Z``) address ``[base, base + span)``; code ``0`` escapes
to a raw 8-bit exponent stored on the side, addressed by a popcount rank
rather than an offset table.  See ``experiments.georefine._glc_tbe`` for the
full derivation; this file reproduces only what a client needs to decode.

  planes  int32[T, 3, 2]   plane b of tile t, lane j -> word j//32, bit j%32
  smb     uint8[T * 64]    (sign << 7) | mantissa, in TILE order
  esc     uint8[E]         raw exponents, tile-major then lane-ascending
  sbbase  int32[ceil(T/superblock)]   escapes before each superblock

Layout ``mma16`` permutes the 64 positions of a tile so a lane's 16 elements
are contiguous -- the property the fragment kernel needs.  It is a pure
in-tile permutation: identical tile count, identical plane width, identical
escape population, so the byte count is identical to ``flat64`` by
construction.  This module supports all three published layouts
(``flat64``, ``tile8x8``, ``mma16``) so its round-trip can be checked against
the research container element for element; the fragment-kernel serving path
in ``tbe_mma.py`` accepts only ``mma16``.

There is no tolerance anywhere in this file: reconstruction is exact for
every bf16 bit pattern, zero, denormals, both infinities and every NaN
payload, because the container stores the three fields of the word and
reassembles them rather than storing an approximation of the value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

TILE = 64
LANES_PER_WORD = 32
PLANES = 3
SUPERBLOCK = 32
HEADER_BYTES = 24

MODE_W7 = 0
MODE_W6Z = 1

LAYOUTS = ("flat64", "tile8x8", "mma16")

_MMA16_OFFSETS = (0, 1, 8, 9)
MMA16_PERM = tuple(
    16 * ((p >> 2) & 3) + 2 * (p >> 4) + _MMA16_OFFSETS[p & 3]
    for p in range(TILE)
)
MMA16_INVERSE = tuple(sorted(range(TILE), key=lambda p: MMA16_PERM[p]))

_MMA16_CACHE: dict = {}


def _mma16_index(device, inverse: bool) -> torch.Tensor:
    key = (str(device), bool(inverse))
    got = _MMA16_CACHE.get(key)
    if got is None:
        got = torch.tensor(
            MMA16_INVERSE if inverse else MMA16_PERM,
            dtype=torch.int64, device=device,
        )
        _MMA16_CACHE[key] = got
    return got


class TBEError(RuntimeError):
    """Raised only for malformed input, never for un-encodable data.

    The container has no representability ceiling: every exponent the window
    misses takes the escape.  If this is raised it is a caller bug, not a
    property of the weights.
    """


@dataclass
class TBETensor:
    """An encoded tensor.  Every field is needed to decode; nothing derived."""

    shape: Tuple[int, int]
    layout: str
    mode: int
    base: int
    tiles: int
    escapes: int
    planes: torch.Tensor   # int64 or int32 [T, 3, 2]
    smb: torch.Tensor      # uint8 [T * 64]
    esc: torch.Tensor      # uint8 [E]
    sbbase: torch.Tensor   # int64 or int32 [ceil(T / superblock)]
    superblock: int = SUPERBLOCK

    @property
    def numel(self) -> int:
        return int(self.shape[0]) * int(self.shape[1])

    def byte_size(self) -> dict:
        """Serialized bytes, counted from the arrays that exist.

        Includes the padding of the final tile and the tensor header.  A
        projection that omits its own metadata is how a container ends up
        quoting a ratio it cannot deliver.
        """
        pl = int(self.planes.numel()) * 4
        smb = int(self.smb.numel())
        esc = int(self.esc.numel())
        sb = int(self.sbbase.numel()) * 4
        return {
            "planes": pl, "smb": smb, "escapes": esc, "sbbase": sb,
            "header": HEADER_BYTES,
            "total": pl + smb + esc + sb + HEADER_BYTES,
        }

    def bits_per_element(self) -> float:
        return self.byte_size()["total"] * 8.0 / self.numel

    def ratio(self) -> float:
        return 16.0 / self.bits_per_element()

    def escape_rate(self) -> float:
        return float(self.escapes) / float(max(1, self.numel))

    def to(self, device) -> "TBETensor":
        return TBETensor(
            shape=self.shape, layout=self.layout, mode=self.mode,
            base=self.base, tiles=self.tiles, escapes=self.escapes,
            planes=self.planes.to(device), smb=self.smb.to(device),
            esc=self.esc.to(device), sbbase=self.sbbase.to(device),
            superblock=self.superblock,
        )


# ---------------------------------------------------------------------------
# tile geometry
# ---------------------------------------------------------------------------
def n_tiles_for(n: int, k: int, layout: str) -> int:
    if layout in ("flat64", "mma16"):
        return (n * k + TILE - 1) // TILE
    if layout == "tile8x8":
        if n % 8 or k % 8:
            raise TBEError(
                f"layout tile8x8 needs both dims divisible by 8, got {(n, k)}; "
                "use layout='flat64', which has no shape precondition"
            )
        return (n // 8) * (k // 8)
    raise TBEError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")


def _row_chunks(n: int, k: int, target_elems: int):
    per = max(64, (max(target_elems // max(k, 1), 1) // 64) * 64)
    for r0 in range(0, n, per):
        yield r0, min(n, r0 + per)


def _split_bf16(weight: torch.Tensor):
    if weight.dtype != torch.bfloat16:
        raise TBEError(f"expected bfloat16, got {weight.dtype}")
    bits = weight.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    exp = (bits >> 7) & 0xFF
    smb = (((bits >> 15) & 0x1) << 7) | (bits & 0x7F)
    return exp, smb


def _chunk_to_tiles(sub: torch.Tensor, k: int, layout: str, is_last: bool):
    exp, smb = _split_bf16(sub)
    if layout == "tile8x8":
        rows = sub.shape[0]
        shape = (rows // 8, 8, k // 8, 8)
        e = exp.reshape(shape).permute(0, 2, 1, 3).reshape(-1, TILE)
        s = smb.reshape(shape).permute(0, 2, 1, 3).reshape(-1, TILE)
        pad = torch.zeros_like(e, dtype=torch.bool)
        return (e.to(torch.uint8).contiguous(), s.to(torch.uint8).contiguous(),
                pad.contiguous())

    e = exp.reshape(-1)
    s = smb.reshape(-1)
    m = e.numel()
    tiles = (m + TILE - 1) // TILE
    short = tiles * TILE - m
    if short:
        if not is_last:
            raise TBEError("only the final chunk may hold a partial tile")
        e = torch.cat([e, torch.zeros(short, dtype=e.dtype, device=e.device)])
        s = torch.cat([s, torch.zeros(short, dtype=s.dtype, device=s.device)])
    pad = torch.zeros(tiles * TILE, dtype=torch.bool, device=e.device)
    if short:
        pad[m:] = True
    e = e.to(torch.uint8).reshape(tiles, TILE)
    s = s.to(torch.uint8).reshape(tiles, TILE)
    pad = pad.reshape(tiles, TILE)
    if layout == "mma16":
        idx = _mma16_index(e.device, inverse=False)
        e, s, pad = e[:, idx], s[:, idx], pad[:, idx]
    return (e.contiguous(), s.contiguous(), pad.contiguous())


def _pack_planes(code_tiles: torch.Tensor) -> torch.Tensor:
    t = code_tiles.shape[0]
    cw = code_tiles.to(torch.int64).reshape(t, TILE // LANES_PER_WORD,
                                            LANES_PER_WORD)
    lane_bit = torch.arange(LANES_PER_WORD, dtype=torch.int64,
                            device=code_tiles.device).view(1, 1, LANES_PER_WORD)
    out = torch.empty((t, PLANES, TILE // LANES_PER_WORD), dtype=torch.int64,
                      device=code_tiles.device)
    for b in range(PLANES):
        out[:, b, :] = (((cw >> b) & 1) << lane_bit).sum(dim=2)
    return out


def _unpack_planes(planes: torch.Tensor) -> torch.Tensor:
    p = planes.to(torch.int64)
    t = p.shape[0]
    lane_bit = torch.arange(LANES_PER_WORD, dtype=torch.int64,
                            device=p.device).view(1, 1, LANES_PER_WORD)
    code = torch.zeros((t, TILE // LANES_PER_WORD, LANES_PER_WORD),
                       dtype=torch.int64, device=p.device)
    for b in range(PLANES):
        code |= ((p[:, b, :].unsqueeze(2) >> lane_bit) & 1) << b
    return code.reshape(t, TILE)


# ---------------------------------------------------------------------------
# window selection
# ---------------------------------------------------------------------------
def _best_window(hist: torch.Tensor, width: int, exclude_zero: bool):
    h = hist.clone()
    if exclude_zero:
        h[0] = 0
    c = torch.cumsum(h, dim=0)
    lo = torch.cat([torch.zeros(1, dtype=c.dtype, device=c.device), c[:-1]])
    starts = 256 - width + 1
    sums = c[width - 1 : width - 1 + starts] - lo[:starts]
    b = int(torch.argmax(sums).item())
    return b, int(sums[b].item())


def exponent_histogram(weight: torch.Tensor, target_elems: int = 1 << 25) -> torch.Tensor:
    n, k = int(weight.shape[0]), int(weight.shape[1])
    hist = torch.zeros(256, dtype=torch.int64)
    for r0, r1 in _row_chunks(n, k, target_elems):
        exp, _ = _split_bf16(weight[r0:r1])
        hist += torch.bincount(exp.reshape(-1).to(torch.int64),
                               minlength=256)[:256].cpu()
    return hist


def choose_window(exp_or_hist: torch.Tensor):
    """Pick ``(mode, base, escape_rate)`` by exact cost, not by heuristic."""
    if exp_or_hist.numel() == 256 and exp_or_hist.dtype == torch.int64:
        hist = exp_or_hist
    else:
        hist = torch.bincount(exp_or_hist.reshape(-1).to(torch.int64),
                              minlength=256)[:256]
    total = int(hist.sum().item())
    zero_ct = int(hist[0].item())

    b7, cov7 = _best_window(hist, 7, exclude_zero=False)
    b6, cov6 = _best_window(hist, 6, exclude_zero=True)
    esc7 = total - cov7
    esc6z = total - cov6 - zero_ct

    if esc6z < esc7:
        return MODE_W6Z, b6, esc6z / max(1, total)
    return MODE_W7, b7, esc7 / max(1, total)


# ---------------------------------------------------------------------------
# encode / decode
# ---------------------------------------------------------------------------
def encode_tbe(
    weight: torch.Tensor,
    layout: str = "mma16",
    superblock: int = SUPERBLOCK,
    mode: Optional[int] = None,
    base: Optional[int] = None,
    target_elems: int = 1 << 25,
) -> TBETensor:
    """bf16 ``[N, K]`` -> TBE container.  Exact for every input bit pattern.

    Carried in the loader as well as the builder so a client can re-encode a
    tensor they decoded and confirm byte-identity against the shipped planes
    without trusting a recorded digest.
    """
    if weight.dim() != 2:
        raise TBEError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")
    n, k = int(weight.shape[0]), int(weight.shape[1])
    tiles = n_tiles_for(n, k, layout)

    if mode is None or base is None:
        mode, base, _ = choose_window(exponent_histogram(weight, target_elems))
    span = 6 if mode == MODE_W7 else 5

    planes_out = torch.empty((tiles, PLANES, TILE // LANES_PER_WORD),
                             dtype=torch.int64)
    smb_out = torch.empty(tiles * TILE, dtype=torch.uint8)
    per_tile = torch.empty(tiles, dtype=torch.int64)
    esc_parts = []
    t_at = 0

    chunks = list(_row_chunks(n, k, target_elems))
    for ci, (r0, r1) in enumerate(chunks):
        e, s, pad = _chunk_to_tiles(weight[r0:r1], k, layout, ci == len(chunks) - 1)
        ei = e.to(torch.int16)
        in_win = (ei >= base) & (ei <= base + span)
        code = torch.where(in_win, ei - base + 1, torch.zeros_like(ei))
        if mode == MODE_W6Z:
            code = torch.where(ei == 0, torch.full_like(code, 7), code)
        code = torch.where(pad, torch.ones_like(code), code)

        nt = code.shape[0]
        planes_out[t_at:t_at + nt] = _pack_planes(code)
        smb_out[t_at * TILE:(t_at + nt) * TILE] = s.reshape(-1)
        em = code == 0
        per_tile[t_at:t_at + nt] = em.sum(dim=1).to(torch.int64)
        if bool(em.any()):
            esc_parts.append(e[em].contiguous())
        t_at += nt

    if t_at != tiles:
        raise TBEError(f"chunking produced {t_at} tiles, expected {tiles}")

    esc = (torch.cat(esc_parts) if esc_parts
           else torch.zeros(0, dtype=torch.uint8))
    if int(esc.numel()) > 0xFFFFFFFF:
        raise TBEError("more than 2^32 escapes; a uint32 base cannot address them")
    if planes_out.numel() and int(planes_out.max().item()) > 0xFFFFFFFF:
        raise TBEError("plane word overflowed 32 bits -- lane packing is wrong")

    nsb = (tiles + superblock - 1) // superblock
    if nsb:
        padded = torch.zeros(nsb * superblock, dtype=torch.int64)
        padded[:tiles] = per_tile
        sb_tot = padded.reshape(nsb, superblock).sum(dim=1)
        sbbase = torch.cat([torch.zeros(1, dtype=torch.int64),
                            torch.cumsum(sb_tot, dim=0)[:-1]])
    else:
        sbbase = torch.zeros(0, dtype=torch.int64)

    return TBETensor(
        shape=(n, k), layout=layout, mode=int(mode), base=int(base),
        tiles=int(tiles), escapes=int(esc.numel()),
        planes=planes_out, smb=smb_out, esc=esc, sbbase=sbbase,
        superblock=int(superblock),
    )


def decode_tbe(c: TBETensor, target_elems: int = 1 << 25) -> torch.Tensor:
    """TBE container -> bf16 ``[N, K]``.  The inverse, by construction."""
    n, k = c.shape
    out = torch.empty(n * k, dtype=torch.int16)
    tiles_per_chunk = max(1, (target_elems // TILE))
    if c.layout == "tile8x8":
        row_tiles = k // 8
        tiles_per_chunk = max(row_tiles, (tiles_per_chunk // row_tiles) * row_tiles)
    for t0 in range(0, c.tiles, tiles_per_chunk):
        t1 = min(c.tiles, t0 + tiles_per_chunk)
        code = _unpack_planes(c.planes[t0:t1])
        exp = torch.where(code > 0, code - 1 + c.base, torch.zeros_like(code))
        if c.mode == MODE_W6Z:
            exp = torch.where(code == 7, torch.zeros_like(exp), exp)

        em = code == 0
        if bool(em.any()):
            before = int(c.sbbase[t0 // c.superblock].item())
            sb_t0 = (t0 // c.superblock) * c.superblock
            if sb_t0 < t0:
                lead = _unpack_planes(c.planes[sb_t0:t0])
                before += int((lead == 0).sum().item())
            flat = em.reshape(-1)
            rank = torch.cumsum(flat.to(torch.int64), dim=0) - 1 + before
            expf = exp.reshape(-1)
            expf[flat] = c.esc[rank[flat]].to(torch.int64)
            exp = expf.reshape(code.shape)

        smb = c.smb[t0 * TILE:t1 * TILE].to(torch.int64).reshape(code.shape)
        word = ((smb & 0x80) << 8) | ((exp & 0xFF) << 7) | (smb & 0x7F)
        if c.layout == "mma16":
            word = word[:, _mma16_index(word.device, inverse=True)]
        word = word.reshape(-1)

        if c.layout in ("flat64", "mma16"):
            lo, hi = t0 * TILE, min(t1 * TILE, n * k)
            out[lo:hi] = (word[: hi - lo] & 0xFFFF).to(torch.int32).to(torch.int16)
        else:
            rb0, rb1 = t0 // (k // 8), t1 // (k // 8)
            if t0 % (k // 8) or t1 % (k // 8):
                raise TBEError("tile8x8 decode chunk must cover whole row-blocks")
            sub = word.reshape(rb1 - rb0, k // 8, 8, 8).permute(0, 2, 1, 3)
            out[rb0 * 8 * k:rb1 * 8 * k] = (
                sub.reshape(-1) & 0xFFFF).to(torch.int32).to(torch.int16)

    return out.view(torch.bfloat16).reshape(n, k)


def tbe_certify(t: TBETensor, original: torch.Tensor, chunk_elements: int = 1 << 22) -> dict:
    """Bitwise round-trip over EVERY element of ``original``.  Never skips.

    Compares raw 16-bit patterns, so NaN payloads, signed zero and denormals
    must match exactly rather than merely compare equal.
    """
    n, k = int(t.shape[0]), int(t.shape[1])
    if tuple(original.shape) != (n, k):
        raise TBEError(
            f"original has shape {tuple(original.shape)}, container has {(n, k)}"
        )
    decoded = decode_tbe(t, target_elems=chunk_elements)
    a = original.detach().contiguous().view(torch.int16)
    b = decoded.contiguous().view(torch.int16)
    mismatched = int((a != b).sum().item())
    return {
        "elements_checked": n * k,
        "elements_total": n * k,
        "mismatched_elements": mismatched,
        "bitwise_ok": mismatched == 0,
        "fully_checked": True,
    }


__all__ = [
    "HEADER_BYTES",
    "LAYOUTS",
    "MODE_W6Z",
    "MODE_W7",
    "SUPERBLOCK",
    "TILE",
    "TBEError",
    "TBETensor",
    "choose_window",
    "decode_tbe",
    "encode_tbe",
    "exponent_histogram",
    "n_tiles_for",
    "tbe_certify",
]
