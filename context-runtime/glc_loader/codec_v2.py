"""GLC codec v2 ("TBE2"): bit-exact bf16 container, decodable per 128-column chunk.

Frozen format spec: ``docs/research/CODEC_V2_FORMAT_20261004.md``.  This file is the
encoder, the pure-numpy reference decoder, the container I/O, the exact size
accounting, and a lane-model decoder that executes the chunk decode the way the
GPU kernel will (16 lanes, 32-bit words, popcount / funnel-shift / shuffle only).

Dependencies: the standard library and numpy.  ``safetensors`` only for the
on-disk container helpers.  Importable by file path (no package ``__init__``
side effects), so the streaming size ledger runs on a bare CPU VM.

THE CODE (per tensor, per element)
----------------------------------
Every bf16 word is split into ``smb = sign<<7 | mantissa7`` (stored raw, one byte)
and the 8-bit exponent ``e``.  The exponent is mapped through a per-tensor
codebook ``cb[0..12]`` (the 13 most frequent exponents, by descending count) to a
symbol ``s``; exponents outside the codebook take ``s = RAW``.  The symbol is
written as a three-level escape code:

    level 1  2-bit code, fixed position (two bit-planes):  s in 0..2   -> code s
                                                           otherwise   -> code 3
    level 2  2-bit digit, chunk overflow, rank-addressed:  s in 3..5   -> digit s-3
                                                           otherwise   -> digit 3
    level 3  3-bit digit, chunk overflow, rank-addressed:  s in 6..12  -> digit s-6
                                                           s == RAW    -> digit 7
    raw      8-bit exponent byte, chunk overflow, rank-addressed.

Rank-addressed: the k-th element of a chunk (column order) that reaches level d
owns the k-th entry of that chunk's level-d region.  So the position of every
entry is a popcount/prefix-sum over fixed-position data or over the level
above -- never over the chunk's previous variable-length entries.

There is no tolerance anywhere: every bf16 bit pattern (+-0, denormals, +-Inf,
every NaN payload) is the three stored fields reassembled.
"""
from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FORMAT = "glc-codec-v2"
FORMAT_VERSION = 1

CH = 128                  # columns per chunk (= BI-GEMM K chunk)
GROUP = 16                # chunks per index group (one u32 word offset per group)
LEN_UNITS = (2, 4, 8)     # per-tensor: chunk regions padded to a multiple of u bits, len = bits/u (u8)
LEN_MAX = 0xFF            # len is ALWAYS one byte; u = smallest unit with max region <= 255*u bits
W1, W2, W3, WRAW = 2, 2, 3, 8
N1, N2, N3 = 3, 3, 7      # direct symbols per level
NSYM = N1 + N2 + N3       # 13 codebook entries
RAW = NSYM                # symbol id of a raw-escaped exponent
ESC1, ESC2, ESC3 = 3, 3, 7
WINDOW_BITS = 16 * 32     # the kernel's 16-lane distributed overflow window
FAST_PATH_MAX_BITS = WINDOW_BITS - 32   # chunk overflow decodable from one window

MODE_ROWS = "rows"        # [N, K] with K = prod(shape[1:]), rows padded to x128
MODE_FLAT = "flat"        # numel padded to x128, viewed [numel/128, 128]
MODE_RAW = "raw"          # stored verbatim (non-bf16 dtypes, tiny tensors)

MIN_CODED_NUMEL = 4096    # below this a tensor is stored raw (header > payload win)
LOADER_TAIL_PAD_BYTES = 16 + 16 * 4   # resident only: zero tail after len (16 B) and ovf (16 words)


class CodecV2Error(RuntimeError):
    """Malformed input or a container that fails an integrity check."""


# ---------------------------------------------------------------------------
# codebook
# ---------------------------------------------------------------------------
def exponent_histogram_u16(bits: np.ndarray) -> np.ndarray:
    return np.bincount(((bits >> 7) & 0xFF).ravel(), minlength=256)[:256].astype(np.int64)


def build_codebook(hist: np.ndarray) -> np.ndarray:
    """13 exponent values by descending count (ties -> smaller exponent first).

    Slots the tensor cannot fill repeat cb[0]; the encoder maps an exponent to its
    FIRST slot, so a repeated slot is never emitted and decode is unaffected.
    """
    hist = np.asarray(hist, dtype=np.int64)
    order = np.lexsort((np.arange(256), -hist))      # primary: -count, secondary: value
    present = [int(v) for v in order if hist[v] > 0][:NSYM]
    if not present:
        present = [0]
    cb = present + [present[0]] * (NSYM - len(present))
    return np.asarray(cb, dtype=np.uint8)


def symbol_lut(cb: np.ndarray) -> np.ndarray:
    lut = np.full(256, RAW, dtype=np.uint8)
    for s in range(NSYM - 1, -1, -1):          # reverse so the first slot wins
        lut[int(cb[s])] = s
    return lut


# ---------------------------------------------------------------------------
# tensor geometry
# ---------------------------------------------------------------------------
@dataclass
class Geometry:
    mode: str
    shape: Tuple[int, ...]
    N: int            # rows of the coded view
    K: int            # true columns of the coded view
    Kp: int           # columns padded to x128
    nch: int          # chunks per row
    ngrp: int         # index groups per row
    nchp: int         # len-row stride: nch rounded up to x4 (u32-aligned len rows)

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1


def geometry_for(shape: Sequence[int], dtype: str = "BF16") -> Geometry:
    shape = tuple(int(x) for x in shape)
    numel = int(np.prod(shape)) if shape else 1
    if dtype != "BF16" or numel < MIN_CODED_NUMEL:
        return Geometry(MODE_RAW, shape, 0, 0, 0, 0, 0, 0)
    if len(shape) >= 2 and int(np.prod(shape[1:])) >= CH:
        N, K = shape[0], int(np.prod(shape[1:]))
        mode = MODE_ROWS
    else:
        N, K = (numel + CH - 1) // CH, CH
        mode = MODE_FLAT
    Kp = ((K + CH - 1) // CH) * CH
    nch = Kp // CH
    ngrp = (nch + GROUP - 1) // GROUP
    return Geometry(mode, shape, N, K, Kp, nch, ngrp, ((nch + 3) // 4) * 4)


def _as_rows(bits: np.ndarray, g: Geometry, pad_word: int = 0) -> np.ndarray:
    """uint16 words of the source -> [N, Kp] padded view.  The encoder pads with
    ``cb[0] << 7`` (the cheapest symbol, no overflow); the decoder drops padding."""
    flat = np.ascontiguousarray(bits).reshape(-1)
    if g.mode == MODE_ROWS:
        rows = flat.reshape(g.N, g.K)
        if g.Kp == g.K:
            return rows
        out = np.full((g.N, g.Kp), pad_word, dtype=np.uint16)
        out[:, :g.K] = rows
        return out
    out = np.full(g.N * g.Kp, pad_word, dtype=np.uint16)
    out[:flat.size] = flat
    return out.reshape(g.N, g.Kp)


def _source_hist(bits: np.ndarray) -> np.ndarray:
    return exponent_histogram_u16(np.ascontiguousarray(bits).reshape(-1))


# ---------------------------------------------------------------------------
# encoded tensor
# ---------------------------------------------------------------------------
@dataclass
class V2Tensor:
    name: str
    dtype: str
    geom: Geometry
    sha256: str                         # of the source tensor bytes (identity)
    cb: Optional[np.ndarray] = None     # uint8[13]
    smb: Optional[np.ndarray] = None    # uint8[N, Kp]
    planes: Optional[np.ndarray] = None  # uint32[N, nch, 8]
    ovf: Optional[np.ndarray] = None    # uint32[n_words]
    len_: Optional[np.ndarray] = None   # uint8[N, nchp], chunk region = len * len_unit bits
    grp: Optional[np.ndarray] = None    # uint32[N, ngrp]
    raw: Optional[np.ndarray] = None    # uint8[nbytes] (MODE_RAW)
    len_unit: int = 2
    counts: Dict[str, int] = field(default_factory=dict)
    crc: Dict[str, int] = field(default_factory=dict)

    # -- accounting -------------------------------------------------------
    def stream_bytes(self) -> Dict[str, int]:
        if self.geom.mode == MODE_RAW:
            return {"raw": int(self.raw.nbytes)}
        return {
            "smb": int(self.smb.nbytes), "planes": int(self.planes.nbytes),
            "ovf": int(self.ovf.nbytes), "len": int(self.len_.nbytes),
            "grp": int(self.grp.nbytes), "codebook": NSYM,
        }

    def total_bytes(self) -> int:
        return sum(self.stream_bytes().values())

    def header(self) -> dict:
        g = self.geom
        h = {"name": self.name, "dtype": self.dtype, "shape": list(g.shape), "mode": g.mode,
             "sha256": self.sha256, "crc32": dict(self.crc)}
        if g.mode != MODE_RAW:
            h.update({"N": g.N, "K": g.K, "Kp": g.Kp, "nch": g.nch, "ngrp": g.ngrp,
                      "len_unit": int(self.len_unit),
                      "codebook": [int(x) for x in self.cb], "counts": dict(self.counts)})
        return h

    def streams(self) -> Dict[str, np.ndarray]:
        if self.geom.mode == MODE_RAW:
            return {"raw": self.raw}
        return {"smb": self.smb, "planes": self.planes, "ovf": self.ovf, "len": self.len_,
                "grp": self.grp, "codebook": self.cb}


def _crc(a: np.ndarray) -> int:
    return zlib.crc32(np.ascontiguousarray(a).view(np.uint8).reshape(-1).data) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# per-block symbol analysis (shared by encoder and size ledger)
# ---------------------------------------------------------------------------
def _block_symbols(rows: np.ndarray, lut: np.ndarray):
    """rows uint16[n, Kp] -> (smb u8, s u8 [n, nch, 128], e u8 [n, nch, 128])."""
    n, Kp = rows.shape
    e = ((rows >> 7) & 0xFF).astype(np.uint8)
    smb = (((rows >> 8) & 0x80) | (rows & 0x7F)).astype(np.uint8)
    s = lut[e]
    return smb, s.reshape(n, Kp // CH, CH), e.reshape(n, Kp // CH, CH)


def _chunk_bits(s3: np.ndarray):
    esc1 = s3 >= N1
    esc2 = s3 >= N1 + N2
    esc3 = s3 == RAW
    n1 = esc1.sum(-1, dtype=np.int64)
    n2 = esc2.sum(-1, dtype=np.int64)
    n3 = esc3.sum(-1, dtype=np.int64)
    L = W2 * n1 + W3 * n2 + WRAW * n3
    return esc1, esc2, esc3, n1, n2, n3, L


def _units(L: np.ndarray, unit: int) -> np.ndarray:
    """payload bits -> stored length in ``unit``-bit units (region = units * unit bits)."""
    return (L + (unit - 1)) // unit


def choose_len_unit(max_payload_bits: int) -> int:
    for u in LEN_UNITS:
        if (max_payload_bits + u - 1) // u <= LEN_MAX:
            return u
    raise CodecV2Error(f"chunk payload {max_payload_bits} bits exceeds 255*8")  # impossible: <= 1664


def _group_layout(L: np.ndarray):
    """L int64[n, nch] chunk REGION bits (len * unit) -> (group bits [n, ngrp],
    within-group exclusive bit offset [n, nch]).  Groups = 16 consecutive chunks of a row."""
    n, nch = L.shape
    gp = ((nch + GROUP - 1) // GROUP) * GROUP
    Lp = np.zeros((n, gp), dtype=np.int64)
    Lp[:, :nch] = L
    G = Lp.reshape(n, gp // GROUP, GROUP)
    excl = np.cumsum(G, axis=-1) - G
    return G.sum(-1), excl.reshape(n, gp)[:, :nch]


# ---------------------------------------------------------------------------
# encoder
# ---------------------------------------------------------------------------
def _block_rows_for(g: Geometry, target_elems: int) -> int:
    return max(1, target_elems // max(1, g.Kp))


def encode_tensor(name: str, arr: np.ndarray, dtype: str = "BF16",
                  target_elems: int = 1 << 23) -> V2Tensor:
    """Encode one tensor.  ``arr`` holds the source words: uint16 for BF16 (raw bit
    patterns, NOT values), any array for other dtypes (stored verbatim)."""
    shape = tuple(int(x) for x in arr.shape)
    src = np.ascontiguousarray(arr)
    sha = hashlib.sha256(src.view(np.uint8).reshape(-1).data).hexdigest()
    g = geometry_for(shape, dtype)
    if g.mode == MODE_RAW:
        raw = src.view(np.uint8).reshape(-1).copy()
        t = V2Tensor(name, dtype, g, sha, raw=raw)
        t.crc = {"raw": _crc(raw)}
        return t
    if src.dtype != np.uint16:
        raise CodecV2Error(f"{name}: BF16 source must be passed as uint16 bit patterns")
    cb = build_codebook(_source_hist(src))
    lut = symbol_lut(cb)
    rows_all = _as_rows(src, g, int(cb[0]) << 7)
    step = _block_rows_for(g, target_elems)
    max_payload = 0
    for r0 in range(0, g.N, step):                      # pre-pass: the length unit
        s3 = lut[((rows_all[r0:r0 + step] >> 7) & 0xFF).astype(np.uint8)].reshape(-1, g.nch, CH)
        max_payload = max(max_payload, int(_chunk_bits(s3)[-1].max()))
    unit = choose_len_unit(max_payload)

    smb_out = np.empty((g.N, g.Kp), dtype=np.uint8)
    planes_out = np.empty((g.N, g.nch, 8), dtype=np.uint32)
    len_out = np.zeros((g.N, g.nchp), dtype=np.uint8)
    grp_out = np.empty((g.N, g.ngrp), dtype=np.uint32)
    ovf_parts: List[np.ndarray] = []
    word_base = 0
    tot = {"n1": 0, "n2": 0, "n3": 0, "pad_bits": 0}
    for r0 in range(0, g.N, step):
        r1 = min(g.N, r0 + step)
        n = r1 - r0
        smb, s3, e3 = _block_symbols(rows_all[r0:r1], lut)
        smb_out[r0:r1] = smb
        # level-1 planes: plane0 = bit0 of code, plane1 = bit1 (code = min(s, 3))
        code = np.minimum(s3, ESC1)
        p0 = np.packbits((code & 1).astype(np.uint8), axis=-1, bitorder="little")   # [n,nch,16]
        p1 = np.packbits((code >> 1).astype(np.uint8), axis=-1, bitorder="little")
        planes_out[r0:r1, :, 0:4] = p0.view("<u4")
        planes_out[r0:r1, :, 4:8] = p1.view("<u4")
        esc1, esc2, esc3, n1, n2, n3, Lp = _chunk_bits(s3)
        U = _units(Lp, unit)
        len_out[r0:r1, :g.nch] = U
        tot["pad_bits"] += int((U * unit - Lp).sum())
        gbits, excl = _group_layout(U * unit)
        gwords = (gbits + 31) // 32
        tot["pad_bits"] += int((gwords * 32 - gbits).sum())
        flat_gw = gwords.reshape(-1)
        gstart = np.cumsum(flat_gw) - flat_gw            # block-relative word offsets
        if word_base + int(flat_gw.sum()) >= 2 ** 32:
            raise CodecV2Error(f"{name}: overflow stream exceeds 2^32 words")
        grp_out[r0:r1] = (gstart + word_base).reshape(n, g.ngrp).astype(np.uint32)
        nbits = int(flat_gw.sum()) * 32
        O = (gstart.reshape(n, g.ngrp)[:, np.arange(g.nch) // GROUP] * 32 + excl)  # [n, nch]
        bitarr = np.zeros(nbits, dtype=np.uint8)
        # level-2 digits at O + 2*rank1
        r1k = np.cumsum(esc1, axis=-1) - 1
        pos = (O[..., None] + W2 * r1k)[esc1]
        d2 = np.minimum(s3[esc1].astype(np.int64) - N1, ESC2)
        for b in range(W2):
            bitarr[pos + b] = (d2 >> b) & 1
        # level-3 digits at O + 2*n1 + 3*rank2
        r2k = np.cumsum(esc2, axis=-1) - 1
        pos = (O[..., None] + W2 * n1[..., None] + W3 * r2k)[esc2]
        d3 = np.minimum(s3[esc2].astype(np.int64) - (N1 + N2), ESC3)
        for b in range(W3):
            bitarr[pos + b] = (d3 >> b) & 1
        # raw exponent bytes at O + 2*n1 + 3*n2 + 8*rank3
        if esc3.any():
            r3k = np.cumsum(esc3, axis=-1) - 1
            pos = (O[..., None] + W2 * n1[..., None] + W3 * n2[..., None] + WRAW * r3k)[esc3]
            ev = e3[esc3].astype(np.int64)
            for b in range(WRAW):
                bitarr[pos + b] = (ev >> b) & 1
        ovf_parts.append(np.packbits(bitarr, bitorder="little").view("<u4"))
        word_base += int(flat_gw.sum())
        tot["n1"] += int(n1.sum()); tot["n2"] += int(n2.sum()); tot["n3"] += int(n3.sum())
    ovf = np.concatenate(ovf_parts) if ovf_parts else np.zeros(0, dtype=np.uint32)
    t = V2Tensor(name, dtype, g, sha, cb=cb, smb=smb_out, planes=planes_out,
                 ovf=ovf.astype(np.uint32, copy=False), len_=len_out, grp=grp_out, len_unit=unit)
    tot["max_chunk_payload_bits"] = max_payload
    tot["len_unit"] = unit
    tot["slow_path_chunks"] = int((len_out[:, :g.nch].astype(np.int64) * unit > FAST_PATH_MAX_BITS).sum())
    t.counts = tot
    t.crc = {k: _crc(v) for k, v in t.streams().items()}
    return t


# ---------------------------------------------------------------------------
# reference decoder (vectorised numpy)
# ---------------------------------------------------------------------------
def _read_bits(bitarr: np.ndarray, pos: np.ndarray, width: int) -> np.ndarray:
    v = np.zeros(pos.shape, dtype=np.int64)
    for b in range(width):
        v |= bitarr[pos + b].astype(np.int64) << b
    return v


def chunk_offsets(t: V2Tensor, r0: int = 0, r1: Optional[int] = None) -> np.ndarray:
    """Absolute bit offset of every chunk's overflow region, rows r0..r1 (int64[n, nch])."""
    g = t.geom
    r1 = g.N if r1 is None else r1
    L = t.len_[r0:r1, :g.nch].astype(np.int64) * int(t.len_unit)
    _, excl = _group_layout(L)
    base = t.grp[r0:r1].astype(np.int64)[:, np.arange(g.nch) // GROUP] * 32
    return base + excl


def kernel_chunk_offset(t: V2Tensor, row: int, c: int) -> int:
    """Random access exactly as the kernel does it at a split start / gather:
    32*grp[row, c>>4] + unit * (sum of the len bytes of the group's chunks before c),
    the len bytes read as 4 aligned u32 words with a byte mask (dp4a form)."""
    g = c >> 4
    words = np.ascontiguousarray(t.len_[row]).view(np.uint8)
    base = 16 * g
    tot = 0
    for k in range(4):                              # 4 x u32 at len + row*nchp + 16g (+ zero tail)
        w = 0
        for b in range(4):
            i = base + 4 * k + b
            byte = int(words[i]) if i < words.size else 0
            if 4 * k + b < (c & 15):                # mask: only chunks 16g .. c-1
                w |= byte << (8 * b)
        tot += sum((w >> (8 * b)) & 0xFF for b in range(4))   # __dp4a(w, 0x01010101, 0)
    return 32 * int(t.grp[row, g]) + int(t.len_unit) * tot


def kernel_next_offset(o: int, region_bits: int, c: int) -> int:
    """Cursor step inside a split, from chunk c to c+1, with NO index load:
    o + region(c); when c+1 opens a new 16-chunk group the group starts on the next
    32-bit word (groups are packed word-aligned), so round up to a multiple of 32."""
    o2 = o + region_bits
    if (c + 1) % GROUP == 0:
        o2 = (o2 + 31) & ~31
    return o2


def kernel_split_walk(t: V2Tensor, row: int, c0: int, c1: int) -> List[int]:
    """Offsets of chunks c0..c1-1 of ``row`` as one CTA computes them: random access
    at c0, then the cursor.  region(c) is what dec(c) knows: units(2n1+3n2+8n3)*unit,
    recomputed here from the decoded symbols, not read from len."""
    g = t.geom
    ref = decode_tensor(t, check_crc=False, check_sha=False) if not hasattr(t, "_dec_cache") else t._dec_cache
    rows = _as_rows(ref, g, int(t.cb[0]) << 7)
    s3 = symbol_lut(t.cb)[((rows[row:row + 1] >> 7) & 0xFF).astype(np.uint8)].reshape(1, g.nch, CH)
    _, _, _, n1, n2, n3, Lp = _chunk_bits(s3)
    region = (_units(Lp, int(t.len_unit)) * int(t.len_unit))[0]
    out = []
    o = kernel_chunk_offset(t, row, c0)
    for c in range(c0, c1):
        out.append(o)
        o = kernel_next_offset(o, int(region[c]), c)
    return out


def verify_crc(t: V2Tensor) -> None:
    for k, v in t.streams().items():
        want = t.crc.get(k)
        if want is not None and _crc(v) != want:
            raise CodecV2Error(f"{t.name}: crc32 mismatch on stream {k!r}")


def decode_tensor(t: V2Tensor, target_elems: int = 1 << 23, check_crc: bool = True,
                  check_sha: bool = True) -> np.ndarray:
    """V2Tensor -> source words (uint16 bit patterns for BF16; raw bytes otherwise),
    reshaped to the source shape.  Raises CodecV2Error on any integrity failure."""
    if check_crc:
        verify_crc(t)
    g = t.geom
    if g.mode == MODE_RAW:
        out = t.raw
        if check_sha and hashlib.sha256(out.data).hexdigest() != t.sha256:
            raise CodecV2Error(f"{t.name}: sha256 mismatch")
        if t.dtype in ("BF16", "F16"):                 # 16-bit words keep their shape
            return out.view(np.uint16).reshape(g.shape)
        return out
    out = np.empty((g.N, g.Kp), dtype=np.uint16)
    cb = t.cb.astype(np.int64)
    ovf_bytes = np.ascontiguousarray(t.ovf).view(np.uint8)
    step = _block_rows_for(g, target_elems)
    for r0 in range(0, g.N, step):
        r1 = min(g.N, r0 + step)
        n = r1 - r0
        O = chunk_offsets(t, r0, r1)
        w0 = int(t.grp[r0, 0])
        w1 = int(t.grp[r1, 0]) if r1 < g.N else int(t.ovf.size)
        bitarr = np.unpackbits(ovf_bytes[w0 * 4:w1 * 4], bitorder="little")
        Ob = O - w0 * 32
        pl = np.ascontiguousarray(t.planes[r0:r1]).view(np.uint8).reshape(n, g.nch, 2, 16)
        b0 = np.unpackbits(pl[:, :, 0], axis=-1, bitorder="little")
        b1 = np.unpackbits(pl[:, :, 1], axis=-1, bitorder="little")
        code = (b0 | (b1 << 1)).astype(np.int64)                       # [n, nch, 128]
        s = code.copy()
        esc1 = code == ESC1
        r1k = np.cumsum(esc1, axis=-1) - 1
        n1 = esc1.sum(-1)
        d2 = _read_bits(bitarr, (Ob[..., None] + W2 * r1k)[esc1], W2)
        s[esc1] = N1 + d2
        esc2 = np.zeros_like(esc1)
        esc2[esc1] = d2 == ESC2
        r2k = np.cumsum(esc2, axis=-1) - 1
        n2 = esc2.sum(-1)
        d3 = _read_bits(bitarr, (Ob[..., None] + W2 * n1[..., None] + W3 * r2k)[esc2], W3)
        s[esc2] = N1 + N2 + d3
        esc3 = np.zeros_like(esc1)
        esc3[esc2] = d3 == ESC3
        n3 = esc3.sum(-1)
        Lchk = _units(W2 * n1 + W3 * n2 + WRAW * n3, int(t.len_unit))
        if not np.array_equal(Lchk, t.len_[r0:r1, :g.nch].astype(np.int64)):
            raise CodecV2Error(f"{t.name}: chunk length field disagrees with payload")
        ex = cb[np.minimum(s, NSYM - 1)]
        if esc3.any():
            r3k = np.cumsum(esc3, axis=-1) - 1
            pos = (Ob[..., None] + W2 * n1[..., None] + W3 * n2[..., None] + WRAW * r3k)[esc3]
            ex[esc3] = _read_bits(bitarr, pos, WRAW)
        smb = t.smb[r0:r1].astype(np.int64).reshape(n, g.nch, CH)
        word = ((smb & 0x80) << 8) | (ex << 7) | (smb & 0x7F)
        out[r0:r1] = word.reshape(n, g.Kp).astype(np.uint16)
    flat = out[:, :g.K].reshape(-1) if g.mode == MODE_ROWS else out.reshape(-1)[:g.numel]
    res = np.ascontiguousarray(flat).reshape(g.shape)
    if check_sha and hashlib.sha256(res.view(np.uint8).reshape(-1).data).hexdigest() != t.sha256:
        raise CodecV2Error(f"{t.name}: sha256 mismatch after decode")
    return res


def decode_rows(t: V2Tensor, rows: Sequence[int]) -> np.ndarray:
    """Gather-time row decode (the embedding path): uint16[len(rows), K]; each row is
    decoded from its own group bases -- no other row is touched."""
    g = t.geom
    if g.mode != MODE_ROWS:
        raise CodecV2Error("row gather needs a rows-mode tensor")
    out = np.empty((len(rows), g.K), dtype=np.uint16)
    for i, r in enumerate(rows):
        sub = _row_subtensor(t, int(r))
        out[i] = decode_tensor(sub, check_crc=False, check_sha=False).reshape(-1)[:g.K]
    return out


def _row_subtensor(t: V2Tensor, r: int) -> V2Tensor:
    g = t.geom
    w0 = int(t.grp[r, 0])
    w1 = int(t.grp[r + 1, 0]) if r + 1 < g.N else int(t.ovf.size)
    sg = Geometry(MODE_ROWS, (1, g.Kp), 1, g.Kp, g.Kp, g.nch, g.ngrp, g.nchp)
    return V2Tensor(t.name, t.dtype, sg, "", cb=t.cb, smb=t.smb[r:r + 1],
                    planes=t.planes[r:r + 1], ovf=t.ovf[w0:w1],
                    len_=t.len_[r:r + 1], len_unit=t.len_unit, grp=(t.grp[r:r + 1].astype(np.int64) - w0).astype(np.uint32))


# ---------------------------------------------------------------------------
# lane model: the chunk decode exactly as the kernel executes it
# ---------------------------------------------------------------------------
M32 = 0xFFFFFFFF


def _popc(x: int) -> int:
    return bin(x & M32).count("1")


def _funnel_r(lo: int, hi: int, sh: int) -> int:
    """__funnelshift_r(lo, hi, sh): low 32 bits of ((hi:lo) >> (sh & 31))."""
    return ((((hi & M32) << 32) | (lo & M32)) >> (sh & 31)) & M32


def _scan_excl(vals: List[int]) -> Tuple[List[int], int]:
    """Exclusive prefix sum over the 16 lanes of a half-warp (shfl_up ladder)."""
    inc = list(vals)
    off = 1
    while off < 16:
        inc = [inc[i] + (inc[i - off] if i >= off else 0) for i in range(16)]
        off <<= 1
    return [inc[i] - vals[i] for i in range(16)], inc[15]


def lane_model_decode_chunk(planes8: Sequence[int], window: Sequence[int], bit0: int,
                            smb128: Sequence[int], cb: Sequence[int]) -> List[int]:
    """Decode one 128-column chunk the way 16 lanes of a half-warp do.

    planes8  the chunk's 8 plane words (plane0 w0..w3, plane1 w0..w3)
    window   16 overflow words loaded by lanes 0..15 (word i by lane i) starting at
             word floor(O/32); a lane reads word q with __shfl_sync(.., q, 16)
    bit0     O & 31, the chunk's first overflow bit inside window word 0
    Returns the 128 bf16 words in column order.  Only 32-bit ops are used: and/or/
    shift, popc, funnel-shift, half-warp shuffles and two exclusive scans.
    """
    win = list(window) + [0]

    def bits(pos: int, width: int) -> int:          # width <= 32, pos window-relative
        q = bit0 + pos
        v = _funnel_r(win[q >> 5], win[(q >> 5) + 1], q & 31)
        return v & ((1 << width) - 1) if width < 32 else v

    out = [0] * 128
    # level 1: lane j owns columns 8j..8j+7 = byte (j & 3) of plane word (j >> 2)
    b0 = [(planes8[j >> 2] >> (8 * (j & 3))) & 0xFF for j in range(16)]
    b1 = [(planes8[4 + (j >> 2)] >> (8 * (j & 3))) & 0xFF for j in range(16)]
    e1 = [b0[j] & b1[j] for j in range(16)]                       # code 3 = escape
    rho1, n1 = _scan_excl([_popc(x) for x in e1])
    # level 2: the lane's escaped elements own consecutive entries rho1 .. rho1+k1-1
    d2 = [[0] * 8 for _ in range(16)]
    e2 = [0] * 16
    for j in range(16):
        k1 = _popc(e1[j])
        f2 = bits(W2 * rho1[j], W2 * k1) if k1 else 0             # <= 16 bits, one extract
        for r in range(8):
            if (e1[j] >> r) & 1:
                lr = _popc(e1[j] & ((1 << r) - 1))
                d2[j][r] = (f2 >> (W2 * lr)) & 3
                if d2[j][r] == ESC2:
                    e2[j] |= 1 << r
    rho2, n2 = _scan_excl([_popc(x) for x in e2])
    d3 = [[0] * 8 for _ in range(16)]
    e3 = [0] * 16
    for j in range(16):
        k2 = _popc(e2[j])
        f3 = bits(W2 * n1 + W3 * rho2[j], W3 * k2) if k2 else 0   # <= 24 bits, one extract
        for r in range(8):
            if (e2[j] >> r) & 1:
                lr = _popc(e2[j] & ((1 << r) - 1))
                d3[j][r] = (f3 >> (W3 * lr)) & 7
                if d3[j][r] == ESC3:
                    e3[j] |= 1 << r
    rho3, n3 = _scan_excl([_popc(x) for x in e3])
    raw_base = W2 * n1 + W3 * n2
    for j in range(16):
        for r in range(8):
            col = 8 * j + r
            code = ((b0[j] >> r) & 1) | (((b1[j] >> r) & 1) << 1)
            if code < ESC1:
                ex = cb[code]
            elif d2[j][r] < ESC2:
                ex = cb[N1 + d2[j][r]]
            elif d3[j][r] < ESC3:
                ex = cb[N1 + N2 + d3[j][r]]
            else:
                lr = _popc(e3[j] & ((1 << r) - 1))
                ex = bits(raw_base + WRAW * (rho3[j] + lr), WRAW)
            s = smb128[col]
            out[col] = ((s & 0x80) << 8) | ((ex & 0xFF) << 7) | (s & 0x7F)
    return out


def lane_model_check(t: V2Tensor, rows: Sequence[int], chunks: Sequence[int],
                     ref: Optional[np.ndarray] = None) -> int:
    """Decode (row, chunk) pairs with the lane model and compare with the reference
    decoder (``ref`` = its output, recomputed if None).  Fast-path chunks use the
    16-word window; slow-path chunks (region > FAST_PATH_MAX_BITS) the direct-load
    variant.  Returns the number of chunks checked; raises on any mismatch."""
    if ref is None:
        ref = decode_tensor(t, check_crc=False, check_sha=False)
    g = t.geom
    refrows = _as_rows(ref, g, int(t.cb[0]) << 7)
    O = chunk_offsets(t)
    ovf = np.concatenate([t.ovf, np.zeros(17, dtype=np.uint32)])
    n = 0
    for r in rows:
        for c in chunks:
            o = int(O[r, c])
            region = int(t.len_[r, c]) * int(t.len_unit)
            # fast path: the 16-lane window; slow path (region > FAST_PATH_MAX_BITS):
            # the same arithmetic on words fetched straight from global memory
            nw = 16 if region <= FAST_PATH_MAX_BITS else ((o & 31) + region + 31) // 32 + 1
            win = [int(x) for x in ovf[o >> 5:(o >> 5) + nw]]
            got = lane_model_decode_chunk([int(x) for x in t.planes[r, c]], win, o & 31,
                                          [int(x) for x in t.smb[r, c * CH:(c + 1) * CH]],
                                          [int(x) for x in t.cb])
            want = [int(x) for x in refrows[r, c * CH:(c + 1) * CH]]
            if got != want:
                raise CodecV2Error(f"lane model disagrees with reference at row {r} chunk {c}")
            n += 1
    return n


# ---------------------------------------------------------------------------
# exact size from symbols (the ledger path: no bit packing)
# ---------------------------------------------------------------------------
class SizeAccumulator:
    """Exact v2 byte count for one BF16 tensor, fed row blocks in order.  Needs the
    codebook first (pass 1 = histogram).  Identical arithmetic to encode_tensor."""

    def __init__(self, g: Geometry, cb: np.ndarray):
        self.g, self.cb, self.lut = g, cb, symbol_lut(cb)
        self.words = {u: 0 for u in LEN_UNITS}
        self.pad = {u: 0 for u in LEN_UNITS}
        self.slow = {u: 0 for u in LEN_UNITS}
        self.max_payload = 0
        self.n = {"n1": 0, "n2": 0, "n3": 0}
        self.rows_seen = 0

    def feed(self, rows: np.ndarray) -> None:
        """rows: uint16 words [n, Kp] (already padded)."""
        self.feed_exponents(((rows >> 7) & 0xFF).astype(np.uint8))

    def feed_exponents(self, e_rows: np.ndarray) -> None:
        """e_rows: uint8 exponents [n, Kp] (already padded with cb[0])."""
        n = e_rows.shape[0]
        s3 = self.lut[e_rows].reshape(n, self.g.nch, CH)
        _, _, _, n1, n2, n3, Lp = _chunk_bits(s3)
        self.max_payload = max(self.max_payload, int(Lp.max()) if Lp.size else 0)
        for u in LEN_UNITS:
            L = _units(Lp, u) * u
            gbits, _ = _group_layout(L)
            gw = (gbits + 31) // 32
            self.words[u] += int(gw.sum())
            self.pad[u] += int((gw * 32 - gbits).sum()) + int((L - Lp).sum())
            self.slow[u] += int((L > FAST_PATH_MAX_BITS).sum())
        self.n["n1"] += int(n1.sum()); self.n["n2"] += int(n2.sum()); self.n["n3"] += int(n3.sum())
        self.rows_seen += n

    def result(self) -> Dict[str, int]:
        g = self.g
        if self.rows_seen != g.N:
            raise CodecV2Error(f"size accumulator saw {self.rows_seen} of {g.N} rows")
        u = choose_len_unit(self.max_payload)
        d = {"smb": g.N * g.Kp, "planes": g.N * g.nch * 32, "ovf": self.words[u] * 4,
             "len": g.N * g.nchp, "grp": g.N * g.ngrp * 4, "codebook": NSYM}
        d["total"] = sum(d.values())
        d.update({f"count_{k}": v for k, v in self.n.items()})
        d.update({"count_pad_bits": self.pad[u], "count_slow_path_chunks": self.slow[u],
                  "max_chunk_payload_bits": self.max_payload, "len_unit": u})
        return d


def size_of(arr_u16: np.ndarray, shape: Sequence[int], target_elems: int = 1 << 23) -> Dict[str, int]:
    """Exact v2 bytes of one BF16 tensor without packing bits."""
    g = geometry_for(shape, "BF16")
    if g.mode == MODE_RAW:
        return {"raw": int(arr_u16.nbytes), "total": int(arr_u16.nbytes)}
    e = ((np.ascontiguousarray(arr_u16).reshape(-1) >> 7) & 0xFF).astype(np.uint8)
    return size_from_exponents(e, shape, target_elems=target_elems)


def size_from_exponents(e_flat: np.ndarray, shape: Sequence[int],
                        hist: Optional[np.ndarray] = None,
                        target_elems: int = 1 << 23) -> Dict[str, int]:
    """Exact v2 bytes of one BF16 tensor from its exponent bytes alone (uint8, source
    order).  The sign/mantissa plane is raw, so the exponents determine every byte.
    Memory: one u8 per element plus one block -- the streaming ledger's path."""
    g = geometry_for(shape, "BF16")
    if g.mode == MODE_RAW:
        n = int(np.prod(shape)) if shape else 1
        return {"raw": 2 * n, "total": 2 * n}
    e_flat = np.ascontiguousarray(e_flat).reshape(-1)
    if hist is None:
        hist = np.bincount(e_flat, minlength=256)[:256]
    cb = build_codebook(hist)
    acc = SizeAccumulator(g, cb)
    pad = int(cb[0])
    step = _block_rows_for(g, target_elems)
    for r0 in range(0, g.N, step):
        r1 = min(g.N, r0 + step)
        if g.mode == MODE_ROWS and g.Kp == g.K:
            blk = e_flat[r0 * g.K:r1 * g.K].reshape(r1 - r0, g.K)
        elif g.mode == MODE_ROWS:
            blk = np.full((r1 - r0, g.Kp), pad, dtype=np.uint8)
            blk[:, :g.K] = e_flat[r0 * g.K:r1 * g.K].reshape(r1 - r0, g.K)
        else:
            blk = np.full((r1 - r0) * g.Kp, pad, dtype=np.uint8)
            src = e_flat[r0 * g.Kp:r1 * g.Kp]
            blk[:src.size] = src
            blk = blk.reshape(r1 - r0, g.Kp)
        acc.feed_exponents(blk)
    out = acc.result()
    out["codebook"] = NSYM
    out["codebook_values"] = [int(x) for x in cb]
    return out


# ---------------------------------------------------------------------------
# container I/O (manifest + safetensors shards)
# ---------------------------------------------------------------------------
def to_named_arrays(t: V2Tensor) -> Dict[str, np.ndarray]:
    return {f"{t.name}::{k}": np.ascontiguousarray(v) for k, v in t.streams().items()}


def from_named_arrays(header: dict, arrays: Dict[str, np.ndarray]) -> V2Tensor:
    name = header["name"]
    shape = tuple(header["shape"])
    g = geometry_for(shape, header["dtype"])
    if g.mode != header["mode"]:
        raise CodecV2Error(f"{name}: geometry mode {g.mode} != header {header['mode']}")
    t = V2Tensor(name, header["dtype"], g, header["sha256"])
    t.crc = {k: int(v) for k, v in header.get("crc32", {}).items()}
    if g.mode == MODE_RAW:
        t.raw = arrays[f"{name}::raw"]
        return t
    for k in ("N", "K", "Kp", "nch", "ngrp"):
        if int(header[k]) != int(getattr(g, k)):
            raise CodecV2Error(f"{name}: header {k} disagrees with geometry")
    t.cb = arrays[f"{name}::codebook"]
    t.smb = arrays[f"{name}::smb"]
    t.planes = arrays[f"{name}::planes"]
    t.ovf = arrays[f"{name}::ovf"]
    t.len_ = arrays[f"{name}::len"]
    t.grp = arrays[f"{name}::grp"]
    t.counts = dict(header.get("counts", {}))
    exp_shapes = {"smb": (g.N, g.Kp), "planes": (g.N, g.nch, 8), "len": (g.N, g.nchp),
                  "grp": (g.N, g.ngrp), "codebook": (NSYM,)}
    for k, shp in exp_shapes.items():
        a = {"smb": t.smb, "planes": t.planes, "len": t.len_, "grp": t.grp, "codebook": t.cb}[k]
        if tuple(a.shape) != shp:
            raise CodecV2Error(f"{name}: stream {k} shape {a.shape} != {shp}")
    t.len_unit = int(header["len_unit"])
    if t.len_unit not in LEN_UNITS or t.len_.dtype != np.uint8:
        raise CodecV2Error(f"{name}: bad length field (unit {t.len_unit}, dtype {t.len_.dtype})")
    return t


def serialize(t: V2Tensor) -> Tuple[dict, bytes]:
    """In-memory container record: (header, safetensors bytes of the streams)."""
    from safetensors.numpy import save
    return t.header(), save(to_named_arrays(t))


def deserialize(header: dict, blob: bytes) -> V2Tensor:
    from safetensors.numpy import load
    return from_named_arrays(header, load(blob))


__all__ = [
    "CH", "GROUP", "LEN_UNITS", "LOADER_TAIL_PAD_BYTES", "FORMAT", "FORMAT_VERSION", "NSYM", "RAW", "MODE_ROWS", "MODE_FLAT",
    "MODE_RAW", "FAST_PATH_MAX_BITS", "CodecV2Error", "Geometry", "V2Tensor",
    "SizeAccumulator", "build_codebook", "chunk_offsets", "decode_rows", "decode_tensor",
    "deserialize", "encode_tensor", "size_from_exponents", "kernel_chunk_offset",
    "kernel_next_offset", "kernel_split_walk", "exponent_histogram_u16", "from_named_arrays",
    "geometry_for", "lane_model_check", "lane_model_decode_chunk", "serialize", "size_of",
    "symbol_lut", "to_named_arrays", "verify_crc",
]
