"""GLC codec v2.1 ("TBE21"): bit-exact bf16 container decoded inside the BI-GEMM K-chunk loop.

Format spec: ``docs/research/CODEC_V21_FORMAT_20261004.md``.  Codec v2 (``codec_v2.py``,
format id ``glc-codec-v2``) is frozen and stays readable; this is a NEW format id
(``glc-codec-v2.1``, ``FORMAT_VERSION = 2``).  What changes vs v2, and why:

  1. Exponent code (2,2,2,3) + raw-8 (v2: (2,2,3) + raw-8), 16-symbol codebook.  Measured the
     best rank-addressed profile on every model tested (deeper profiles add <= 0.0008x).
  2. No per-chunk index.  v2 stored a u8 chunk length (0.0625 b/weight), a u32 per 16-chunk
     group, rounded every chunk region to a length unit and every group to a word.  v2.1 stores
     ONE u32 word offset per row; a row's chunk regions are bit-contiguous; only the row is
     word-padded.  Random access is per ROW (the embedding gather); the GEMM's split-K starts
     are load-time derived checkpoints (``split_checkpoints``): resident bytes proportional to
     the split count S - 1 per row, zero when S == 1.
  3. Up to 2 codebooks per tensor (row clusters, Lloyd on the exact code cost), selected per row
     by a 1-bit map -- free in the kernel, which already picks a codebook per row.
  4. Level-1 codes stored lane-interleaved (u16 per lane per chunk: code of element m at bits
     4m, code of element 4+m at bits 4m+2) so the kernel's nibble spread is one shift + one LOP3.

Streams per coded tensor (all little-endian):
  smb    u8  [N, Kp]       sign << 7 | mantissa7, natural column order (as v2)
  l1     u16 [N, nch, 16]  level-1 2-bit codes, lane-interleaved (above)
  ovf    u32 [W]           per row: chunk regions back to back, then pad to a word
  rowoff u32 [N]           word index of each row's first chunk region
  cbsel  u8  [ceil(N/8)]   bit r = codebook of row r (all zero when ncb == 1)
  codebook u8 [ncb, 16]    exponent values of symbols 0..15

Chunk region (bits from the chunk's start, rank-addressed exactly like v2):
  [n1 level-2 digits x 2][n2 level-3 digits x 2][n3 level-4 digits x 3][n4 raw bytes x 8]
  symbol s: 0..2 level-1 code s; 3..5 L1 escape + L2 digit s-3; 6..8 + L3 digit s-6;
  9..15 + L4 digit s-9; RAW = 16: L4 digit 7 + raw exponent byte.

Dependencies: the standard library and numpy (``safetensors`` only for container helpers).
"""
from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FORMAT = "glc-codec-v2.1"
FORMAT_VERSION = 2

CH = 128
PROFILE = (2, 2, 2, 3)
W = PROFILE                       # digit width per level (level 1 = the l1 plane)
D = tuple((1 << w) - 1 for w in W)   # direct symbols per level: 3, 3, 3, 7
NSYM = sum(D)                     # 16 codebook entries
RAW = NSYM                        # symbol id of a raw exponent
WRAW = 8
FIRST = (0, 3, 6, 9)              # first symbol of each level
MAX_CB = 2
WINDOW_BITS = 16 * 32
FAST_PATH_MAX_BITS = WINDOW_BITS - 32

MODE_ROWS, MODE_FLAT, MODE_RAW = "rows", "flat", "raw"
MIN_CODED_NUMEL = 4096
LOADER_OVF_TAIL_WORDS = 16


class CodecV21Error(RuntimeError):
    """Malformed input or a container that fails an integrity check."""


# ---------------------------------------------------------------------------
# geometry (identical rules to v2)
# ---------------------------------------------------------------------------
@dataclass
class Geometry:
    mode: str
    shape: Tuple[int, ...]
    N: int
    K: int
    Kp: int
    nch: int

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1


def geometry_for(shape: Sequence[int], dtype: str = "BF16") -> Geometry:
    shape = tuple(int(x) for x in shape)
    numel = int(np.prod(shape)) if shape else 1
    if dtype != "BF16" or numel < MIN_CODED_NUMEL:
        return Geometry(MODE_RAW, shape, 0, 0, 0, 0)
    if len(shape) >= 2 and int(np.prod(shape[1:])) >= CH:
        N, K, mode = shape[0], int(np.prod(shape[1:])), MODE_ROWS
    else:
        N, K, mode = (numel + CH - 1) // CH, CH, MODE_FLAT
    Kp = ((K + CH - 1) // CH) * CH
    return Geometry(mode, shape, N, K, Kp, Kp // CH)


def _as_rows(bits: np.ndarray, g: Geometry, pad_word: int) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# codebooks and the code
# ---------------------------------------------------------------------------
def build_codebook(hist: np.ndarray) -> np.ndarray:
    """16 exponent values by descending count (ties -> smaller exponent); unused slots repeat
    cb[0] (the encoder maps an exponent to its FIRST slot, so a repeat is never emitted)."""
    hist = np.asarray(hist, dtype=np.int64)
    order = np.lexsort((np.arange(256), -hist))
    present = [int(v) for v in order if hist[v] > 0][:NSYM]
    if not present:
        present = [0]
    return np.asarray(present + [present[0]] * (NSYM - len(present)), dtype=np.uint8)


def symbol_lut(cb: np.ndarray) -> np.ndarray:
    lut = np.full(256, RAW, dtype=np.uint8)
    for s in range(NSYM - 1, -1, -1):
        lut[int(cb[s])] = s
    return lut


def symbol_cost_bits() -> np.ndarray:
    """Overflow bits of symbol s (0..RAW): level-1 field excluded."""
    c = np.zeros(RAW + 1, np.int64)
    for s in range(RAW + 1):
        if s < FIRST[1]:
            c[s] = 0
        elif s < FIRST[2]:
            c[s] = W[1]
        elif s < FIRST[3]:
            c[s] = W[1] + W[2]
        elif s < RAW:
            c[s] = W[1] + W[2] + W[3]
        else:
            c[s] = W[1] + W[2] + W[3] + WRAW
    return c


SYM_COST = symbol_cost_bits()


def exp_cost_lut(cb: np.ndarray) -> np.ndarray:
    """Overflow bits per exponent value under codebook cb."""
    return SYM_COST[symbol_lut(cb)]


def _row_exp_hist(e_rows: np.ndarray) -> np.ndarray:
    n = e_rows.shape[0]
    idx = (np.arange(n, dtype=np.int64)[:, None] * 256 + e_rows.astype(np.int64)).ravel()
    return np.bincount(idx, minlength=n * 256).reshape(n, 256)


def choose_codebooks(e_rows: np.ndarray, iters: int = 8, block: int = 1 << 22) -> Tuple[np.ndarray, np.ndarray]:
    """(codebooks uint8[ncb, 16], row selector uint8[N]).  ncb = 2 when two row clusters with
    their own frequency-ordered codebooks (Lloyd on the exact overflow cost, deterministic
    init: rows split at the median of the mean exponent) save more than the extra table and
    the selector map; else 1."""
    N, Kp = e_rows.shape
    step = max(1, block // max(1, Kp))
    RH = np.zeros((N, 256), np.int64)
    for r0 in range(0, N, step):
        RH[r0:r0 + step] = _row_exp_hist(e_rows[r0:r0 + step])
    tot = RH.sum(0)
    cb1 = build_codebook(tot)
    cost1 = RH @ exp_cost_lut(cb1)
    one = int(cost1.sum())
    if N < 2:
        return cb1[None, :], np.zeros(N, np.uint8)
    key = (RH * np.arange(256)[None, :]).sum(1) / np.maximum(RH.sum(1), 1)
    a = key >= np.median(key)
    if a.all() or (~a).all():
        return cb1[None, :], np.zeros(N, np.uint8)
    cbs = None
    for _ in range(iters):
        cbs = [build_codebook(RH[~a].sum(0)), build_codebook(RH[a].sum(0))]
        ca, cb_ = RH @ exp_cost_lut(cbs[0]), RH @ exp_cost_lut(cbs[1])
        na = cb_ < ca                                   # tie -> codebook 0
        if np.array_equal(na, a):
            break
        a = na
    sel = a.astype(np.uint8)
    two = int(np.where(a, RH @ exp_cost_lut(cbs[1]), RH @ exp_cost_lut(cbs[0])).sum())
    # pay: 16-byte table + the selector map (bits) + word-padding differences ignored here
    if two + 8 * (NSYM + (N + 7) // 8) < one and 0 < sel.sum() < N:
        return np.stack(cbs).astype(np.uint8), sel
    return cb1[None, :], np.zeros(N, np.uint8)


# ---------------------------------------------------------------------------
# encoded tensor
# ---------------------------------------------------------------------------
@dataclass
class V21Tensor:
    name: str
    dtype: str
    geom: Geometry
    sha256: str
    cb: Optional[np.ndarray] = None       # uint8[ncb, 16]
    cbsel: Optional[np.ndarray] = None    # uint8[ceil(N/8)] bitmap
    smb: Optional[np.ndarray] = None      # uint8[N, Kp]
    l1: Optional[np.ndarray] = None       # uint16[N, nch, 16]
    ovf: Optional[np.ndarray] = None      # uint32[W]
    rowoff: Optional[np.ndarray] = None   # uint32[N]
    raw: Optional[np.ndarray] = None
    counts: Dict[str, int] = field(default_factory=dict)
    crc: Dict[str, int] = field(default_factory=dict)

    @property
    def ncb(self) -> int:
        return int(self.cb.shape[0])

    def row_codebook(self) -> np.ndarray:
        """uint8[N]: codebook index of every row."""
        N = self.geom.N
        return np.unpackbits(self.cbsel, bitorder="little")[:N]

    def stream_bytes(self) -> Dict[str, int]:
        if self.geom.mode == MODE_RAW:
            return {"raw": int(self.raw.nbytes)}
        return {"smb": int(self.smb.nbytes), "l1": int(self.l1.nbytes), "ovf": int(self.ovf.nbytes),
                "rowoff": int(self.rowoff.nbytes), "cbsel": int(self.cbsel.nbytes),
                "codebook": int(self.cb.nbytes)}

    def total_bytes(self) -> int:
        return sum(self.stream_bytes().values())

    def streams(self) -> Dict[str, np.ndarray]:
        if self.geom.mode == MODE_RAW:
            return {"raw": self.raw}
        return {"smb": self.smb, "l1": self.l1, "ovf": self.ovf, "rowoff": self.rowoff,
                "cbsel": self.cbsel, "codebook": self.cb}

    def header(self) -> dict:
        g = self.geom
        h = {"name": self.name, "dtype": self.dtype, "shape": list(g.shape), "mode": g.mode,
             "format": FORMAT, "format_version": FORMAT_VERSION,
             "sha256": self.sha256, "crc32": dict(self.crc)}
        if g.mode != MODE_RAW:
            h.update({"N": g.N, "K": g.K, "Kp": g.Kp, "nch": g.nch, "profile": list(PROFILE),
                      "ncb": self.ncb, "codebook": [[int(x) for x in c] for c in self.cb],
                      "counts": dict(self.counts)})
        return h


def _crc(a: np.ndarray) -> int:
    return zlib.crc32(np.ascontiguousarray(a).view(np.uint8).reshape(-1).data) & 0xFFFFFFFF


def _block_rows(g: Geometry, target: int) -> int:
    return max(1, target // max(1, g.Kp))


def _levels(s3: np.ndarray):
    """symbols [n, nch, 128] -> escape masks and per-chunk counts of every level."""
    e1 = s3 >= FIRST[1]
    e2 = s3 >= FIRST[2]
    e3 = s3 >= FIRST[3]
    e4 = s3 == RAW
    n = [m.sum(-1, dtype=np.int64) for m in (e1, e2, e3, e4)]
    L = W[1] * n[0] + W[2] * n[1] + W[3] * n[2] + WRAW * n[3]
    return (e1, e2, e3, e4), n, L


def l1_pack(code: np.ndarray) -> np.ndarray:
    """level-1 codes [n, nch, 128] (0..3) -> u16 [n, nch, 16] lane-interleaved."""
    c = code.reshape(code.shape[:-1] + (16, 8)).astype(np.uint16)
    v = np.zeros(c.shape[:-1], dtype=np.uint16)
    for m in range(4):
        v |= c[..., m] << (4 * m)
        v |= c[..., 4 + m] << (4 * m + 2)
    return v


def l1_unpack(v: np.ndarray) -> np.ndarray:
    """u16 [..., 16] -> codes [..., 128]."""
    v = v.astype(np.int64)
    out = np.empty(v.shape + (8,), dtype=np.int64)
    for m in range(4):
        out[..., m] = (v >> (4 * m)) & 3
        out[..., 4 + m] = (v >> (4 * m + 2)) & 3
    return out.reshape(v.shape[:-1] + (v.shape[-1] * 8,))


def encode_tensor(name: str, arr: np.ndarray, dtype: str = "BF16", target_elems: int = 1 << 23,
                  max_cb: int = MAX_CB) -> V21Tensor:
    shape = tuple(int(x) for x in arr.shape)
    src = np.ascontiguousarray(arr)
    sha = hashlib.sha256(src.view(np.uint8).reshape(-1).data).hexdigest()
    g = geometry_for(shape, dtype)
    if g.mode == MODE_RAW:
        raw = src.view(np.uint8).reshape(-1).copy()
        t = V21Tensor(name, dtype, g, sha, raw=raw)
        t.crc = {"raw": _crc(raw)}
        return t
    if src.dtype != np.uint16:
        raise CodecV21Error(f"{name}: BF16 source must be passed as uint16 bit patterns")
    hist = np.bincount(((src.reshape(-1) >> 7) & 0xFF).astype(np.uint8), minlength=256)[:256]
    pad_e = int(build_codebook(hist)[0])
    rows_all = _as_rows(src, g, pad_e << 7)
    e_all = ((rows_all >> 7) & 0xFF).astype(np.uint8)
    if max_cb >= 2:
        cbs, sel = choose_codebooks(e_all)
    else:
        cbs, sel = build_codebook(hist)[None, :], np.zeros(g.N, np.uint8)
    luts = np.stack([symbol_lut(c) for c in cbs])           # [ncb, 256]
    smb_out = np.empty((g.N, g.Kp), dtype=np.uint8)
    l1_out = np.empty((g.N, g.nch, 16), dtype=np.uint16)
    rowoff = np.empty(g.N, dtype=np.uint32)
    parts: List[np.ndarray] = []
    word_base = 0
    tot = {"n1": 0, "n2": 0, "n3": 0, "n4": 0, "pad_bits": 0}
    step = _block_rows(g, target_elems)
    max_region = 0
    for r0 in range(0, g.N, step):
        r1 = min(g.N, r0 + step)
        n = r1 - r0
        rows = rows_all[r0:r1]
        smb_out[r0:r1] = (((rows >> 8) & 0x80) | (rows & 0x7F)).astype(np.uint8)
        s3 = luts[sel[r0:r1].astype(np.int64)[:, None], e_all[r0:r1].astype(np.int64)].reshape(n, g.nch, CH)
        e3v = e_all[r0:r1].reshape(n, g.nch, CH)
        l1_out[r0:r1] = l1_pack(np.minimum(s3, 3))
        (m1, m2, m3, m4), cnt, L = _levels(s3)
        max_region = max(max_region, int(L.max()) if L.size else 0)
        rbits = L.sum(1)
        rwords = (rbits + 31) // 32
        tot["pad_bits"] += int((rwords * 32 - rbits).sum())
        start = np.cumsum(rwords) - rwords                   # block-relative row word offsets
        if word_base + int(rwords.sum()) >= 2 ** 32:
            raise CodecV21Error(f"{name}: overflow stream exceeds 2^32 words")
        rowoff[r0:r1] = (start + word_base).astype(np.uint32)
        O = start[:, None] * 32 + (np.cumsum(L, 1) - L)        # [n, nch] block-relative bit offsets
        bitarr = np.zeros(int(rwords.sum()) * 32, dtype=np.uint8)
        base = O[..., None]
        offs = [np.zeros_like(cnt[0])]
        for q in range(3):
            offs.append(offs[-1] + W[q + 1] * cnt[q])
        # level-2 digits (rank among level-1 escapes)
        rk = np.cumsum(m1, -1) - 1
        pos = (base + W[1] * rk)[m1]
        dg = np.minimum(s3[m1].astype(np.int64) - FIRST[1], D[1])
        for b in range(W[1]):
            bitarr[pos + b] = (dg >> b) & 1
        rk = np.cumsum(m2, -1) - 1
        pos = (base + offs[1][..., None] + W[2] * rk)[m2]
        dg = np.minimum(s3[m2].astype(np.int64) - FIRST[2], D[2])
        for b in range(W[2]):
            bitarr[pos + b] = (dg >> b) & 1
        rk = np.cumsum(m3, -1) - 1
        pos = (base + offs[2][..., None] + W[3] * rk)[m3]
        dg = np.minimum(s3[m3].astype(np.int64) - FIRST[3], D[3])
        for b in range(W[3]):
            bitarr[pos + b] = (dg >> b) & 1
        if m4.any():
            rk = np.cumsum(m4, -1) - 1
            pos = (base + offs[3][..., None] + WRAW * rk)[m4]
            ev = e3v[m4].astype(np.int64)
            for b in range(WRAW):
                bitarr[pos + b] = (ev >> b) & 1
        parts.append(np.packbits(bitarr, bitorder="little").view("<u4"))
        word_base += int(rwords.sum())
        for q, k in enumerate(("n1", "n2", "n3", "n4")):
            tot[k] += int(cnt[q].sum())
    ovf = np.concatenate(parts) if parts else np.zeros(0, np.uint32)
    t = V21Tensor(name, dtype, g, sha, cb=cbs.astype(np.uint8),
                  cbsel=np.packbits(sel.astype(np.uint8), bitorder="little"),
                  smb=smb_out, l1=l1_out, ovf=ovf.astype(np.uint32, copy=False), rowoff=rowoff)
    tot["max_chunk_payload_bits"] = max_region
    tot["slow_path_chunks_upper"] = None
    tot["ncb"] = int(cbs.shape[0])
    t.counts = tot
    t.crc = {k: _crc(v) for k, v in t.streams().items()}
    return t


# ---------------------------------------------------------------------------
# reference decoder (vectorised over rows; chunks walked in order -- there is no chunk index)
# ---------------------------------------------------------------------------
def _read_bits(bitarr: np.ndarray, pos: np.ndarray, width: int) -> np.ndarray:
    v = np.zeros(pos.shape, dtype=np.int64)
    for b in range(width):
        v |= bitarr[pos + b].astype(np.int64) << b
    return v


def verify_crc(t: V21Tensor) -> None:
    for k, v in t.streams().items():
        want = t.crc.get(k)
        if want is not None and _crc(v) != want:
            raise CodecV21Error(f"{t.name}: crc32 mismatch on stream {k!r}")


def _decode_block(t: V21Tensor, r0: int, r1: int, want_offsets: bool = False):
    """rows r0..r1 -> (words uint16 [n, Kp], chunk start bit offsets [n, nch] or None).
    Offsets are absolute (bit 32*rowoff[row] = the row's first chunk)."""
    g = t.geom
    n = r1 - r0
    cbi = t.row_codebook()[r0:r1].astype(np.int64)
    cb = t.cb.astype(np.int64)
    w0 = int(t.rowoff[r0])
    w1 = int(t.rowoff[r1]) if r1 < g.N else int(t.ovf.size)
    bitarr = np.unpackbits(np.ascontiguousarray(t.ovf[w0:w1]).view(np.uint8), bitorder="little")
    bitarr = np.concatenate([bitarr, np.zeros(64, np.uint8)])
    cur = (t.rowoff[r0:r1].astype(np.int64) - w0) * 32
    codes = l1_unpack(t.l1[r0:r1]).reshape(n, g.Kp)          # [n, nch*128]
    out = np.empty((n, g.Kp), dtype=np.uint16)
    offs = np.empty((n, g.nch), dtype=np.int64) if want_offsets else None
    smb = t.smb[r0:r1].astype(np.int64)
    for c in range(g.nch):
        if offs is not None:
            offs[:, c] = cur + w0 * 32
        code = codes[:, c * CH:(c + 1) * CH]
        s = code.copy()
        m1 = code == 3
        n1 = m1.sum(1)
        rk = np.cumsum(m1, 1) - 1
        d = _read_bits(bitarr, (cur[:, None] + W[1] * rk)[m1], W[1])
        s[m1] = FIRST[1] + d
        m2 = np.zeros_like(m1)
        m2[m1] = d == D[1]
        n2 = m2.sum(1)
        rk = np.cumsum(m2, 1) - 1
        d = _read_bits(bitarr, (cur[:, None] + W[1] * n1[:, None] + W[2] * rk)[m2], W[2])
        s[m2] = FIRST[2] + d
        m3 = np.zeros_like(m1)
        m3[m2] = d == D[2]
        n3 = m3.sum(1)
        rk = np.cumsum(m3, 1) - 1
        b3 = cur + W[1] * n1 + W[2] * n2
        d = _read_bits(bitarr, (b3[:, None] + W[3] * rk)[m3], W[3])
        s[m3] = FIRST[3] + d
        m4 = np.zeros_like(m1)
        m4[m3] = d == D[3]
        n4 = m4.sum(1)
        ex = cb[cbi[:, None], np.minimum(s, NSYM - 1)]
        if m4.any():
            rk = np.cumsum(m4, 1) - 1
            b4 = b3 + W[3] * n3
            ex[m4] = _read_bits(bitarr, (b4[:, None] + WRAW * rk)[m4], WRAW)
        sm = smb[:, c * CH:(c + 1) * CH]
        out[:, c * CH:(c + 1) * CH] = (((sm & 0x80) << 8) | (ex << 7) | (sm & 0x7F)).astype(np.uint16)
        cur = cur + W[1] * n1 + W[2] * n2 + W[3] * n3 + WRAW * n4
    # every row must end exactly within its own words (the next row starts at rowoff[r+1])
    nxt = np.empty(n, np.int64)
    nxt[:-1] = (t.rowoff[r0 + 1:r1].astype(np.int64) - w0) * 32
    nxt[-1] = (w1 - w0) * 32
    if np.any(cur > nxt) or np.any(nxt - cur >= 32):
        raise CodecV21Error(f"{t.name}: row payload disagrees with the row offsets")
    return out, offs


def decode_tensor(t: V21Tensor, target_elems: int = 1 << 23, check_crc: bool = True,
                  check_sha: bool = True) -> np.ndarray:
    if check_crc:
        verify_crc(t)
    g = t.geom
    if g.mode == MODE_RAW:
        out = t.raw
        if check_sha and hashlib.sha256(out.data).hexdigest() != t.sha256:
            raise CodecV21Error(f"{t.name}: sha256 mismatch")
        if t.dtype in ("BF16", "F16"):
            return out.view(np.uint16).reshape(g.shape)
        return out
    out = np.empty((g.N, g.Kp), dtype=np.uint16)
    step = _block_rows(g, target_elems)
    for r0 in range(0, g.N, step):
        r1 = min(g.N, r0 + step)
        out[r0:r1], _ = _decode_block(t, r0, r1)
    flat = out[:, :g.K].reshape(-1) if g.mode == MODE_ROWS else out.reshape(-1)[:g.numel]
    res = np.ascontiguousarray(flat).reshape(g.shape)
    if check_sha and hashlib.sha256(res.view(np.uint8).reshape(-1).data).hexdigest() != t.sha256:
        raise CodecV21Error(f"{t.name}: sha256 mismatch after decode")
    return res


def chunk_offsets(t: V21Tensor, target_elems: int = 1 << 23) -> np.ndarray:
    """Absolute bit offset of every chunk's region (int64 [N, nch]) -- by walking each row."""
    g = t.geom
    O = np.empty((g.N, g.nch), dtype=np.int64)
    step = _block_rows(g, target_elems)
    for r0 in range(0, g.N, step):
        r1 = min(g.N, r0 + step)
        _, O[r0:r1] = _decode_block(t, r0, r1, want_offsets=True)
    return O


def split_starts(nch: int, S: int) -> List[int]:
    """First chunk of every split, exactly as bi_gemm_kernel: c0 = s * nch // S."""
    return [(s * nch) // S for s in range(S)]


def split_checkpoints(t: V21Tensor, S: int, offsets: Optional[np.ndarray] = None) -> np.ndarray:
    """RESIDENT (load-time) split-K checkpoints: uint32 [N, S-1], entry s-1 = the bit offset,
    relative to 32 * rowoff[row], of split s's first chunk (s = 1..S-1).  Empty when S == 1."""
    g = t.geom
    if S <= 1:
        return np.zeros((g.N, 0), dtype=np.uint32)
    O = chunk_offsets(t) if offsets is None else offsets
    cs = split_starts(g.nch, S)[1:]
    rel = O[:, cs] - t.rowoff.astype(np.int64)[:, None] * 32
    if rel.size and (rel.min() < 0 or rel.max() >= 2 ** 32):
        raise CodecV21Error("checkpoint out of range")
    return rel.astype(np.uint32)


def decode_rows(t: V21Tensor, rows: Sequence[int]) -> np.ndarray:
    """Gather-time row decode (the embedding path): each row from its own offset only."""
    g = t.geom
    if g.mode != MODE_ROWS:
        raise CodecV21Error("row gather needs a rows-mode tensor")
    out = np.empty((len(rows), g.K), dtype=np.uint16)
    for i, r in enumerate(rows):
        sub = _row_subtensor(t, int(r))
        out[i] = _decode_block(sub, 0, 1)[0].reshape(-1)[:g.K]
    return out


def _row_subtensor(t: V21Tensor, r: int) -> V21Tensor:
    g = t.geom
    w0 = int(t.rowoff[r])
    w1 = int(t.rowoff[r + 1]) if r + 1 < g.N else int(t.ovf.size)
    sg = Geometry(MODE_ROWS, (1, g.Kp), 1, g.Kp, g.Kp, g.nch)
    sel = np.packbits(t.row_codebook()[r:r + 1], bitorder="little")
    return V21Tensor(t.name, t.dtype, sg, "", cb=t.cb, cbsel=sel, smb=t.smb[r:r + 1], l1=t.l1[r:r + 1],
                     ovf=t.ovf[w0:w1], rowoff=np.zeros(1, np.uint32))


# ---------------------------------------------------------------------------
# lane model: one chunk, 16 lanes, 32-bit words only (the kernel's algorithm, scalar form)
# ---------------------------------------------------------------------------
M32 = 0xFFFFFFFF


def _popc(x: int) -> int:
    return bin(x & M32).count("1")


def _funnel_r(lo: int, hi: int, sh: int) -> int:
    return ((((hi & M32) << 32) | (lo & M32)) >> (sh & 31)) & M32


def _scan_excl(vals: List[int]) -> Tuple[List[int], int]:
    inc = list(vals)
    off = 1
    while off < 16:
        inc = [inc[i] + (inc[i - off] if i >= off else 0) for i in range(16)]
        off <<= 1
    return [inc[i] - vals[i] for i in range(16)], inc[15]


def lane_model_decode_chunk(l1_16: Sequence[int], window: Sequence[int], bit0: int,
                            smb128: Sequence[int], cb: Sequence[int]) -> Tuple[List[int], int]:
    """Decode one chunk as 16 lanes do.  ``window`` = overflow words from floor(O/32) (as many
    as the region needs), ``bit0`` = O & 31.  Returns (128 bf16 words, region bits)."""
    win = list(window) + [0, 0]

    def bits(pos: int, width: int) -> int:
        q = bit0 + pos
        v = _funnel_r(win[q >> 5], win[(q >> 5) + 1], q & 31)
        return v & ((1 << width) - 1) if width < 32 else v

    codes = [[(l1_16[j] >> (4 * m + (2 if i >= 4 else 0))) & 3 for i, m in
              ((i, i % 4) for i in range(8))] for j in range(16)]
    esc = [[codes[j][i] == 3 for i in range(8)] for j in range(16)]
    sym = [[codes[j][i] for i in range(8)] for j in range(16)]
    base = 0
    for lev in (1, 2, 3):
        k = [sum(esc[j]) for j in range(16)]
        rho, tot = _scan_excl(k)
        nxt = [[False] * 8 for _ in range(16)]
        for j in range(16):
            f = bits(base + W[lev] * rho[j], W[lev] * k[j]) if k[j] else 0
            r = 0
            for i in range(8):
                if esc[j][i]:
                    dg = (f >> (W[lev] * r)) & ((1 << W[lev]) - 1)
                    r += 1
                    if dg < D[lev]:
                        sym[j][i] = FIRST[lev] + dg
                    else:
                        nxt[j][i] = True
                        sym[j][i] = RAW if lev == 3 else -1
        base += W[lev] * tot
        esc = nxt
    k = [sum(esc[j]) for j in range(16)]
    rho, tot = _scan_excl(k)
    out = [0] * 128
    for j in range(16):
        r = 0
        for i in range(8):
            col = 8 * j + i
            if esc[j][i]:
                ex = bits(base + WRAW * (rho[j] + r), WRAW)
                r += 1
            else:
                ex = int(cb[sym[j][i]])
            s = smb128[col]
            out[col] = ((s & 0x80) << 8) | ((ex & 0xFF) << 7) | (s & 0x7F)
    return out, base + WRAW * tot


def lane_model_check(t: V21Tensor, rows: Sequence[int], chunks: Sequence[int],
                     ref: Optional[np.ndarray] = None) -> int:
    if ref is None:
        ref = decode_tensor(t, check_crc=False, check_sha=False)
    g = t.geom
    refrows = _as_rows(ref, g, 0)
    O = chunk_offsets(t)
    ovf = np.concatenate([t.ovf, np.zeros(72, dtype=np.uint32)])
    cbi = t.row_codebook()
    n = 0
    for r in rows:
        for c in chunks:
            o = int(O[r, c])
            win = [int(x) for x in ovf[o >> 5:(o >> 5) + 64]]   # max region 128*15 bits = 60 words
            got, _ = lane_model_decode_chunk([int(x) for x in t.l1[r, c]], win, o & 31,
                                             [int(x) for x in t.smb[r, c * CH:(c + 1) * CH]],
                                             [int(x) for x in t.cb[cbi[r]]])
            want = [int(x) for x in refrows[r, c * CH:(c + 1) * CH]]
            valid = (min(CH, g.K - c * CH) if g.mode == MODE_ROWS
                     else min(CH, max(0, g.numel - r * g.Kp - c * CH)))   # padding is not data
            got, want = got[:valid], want[:valid]
            if got != want:
                raise CodecV21Error(f"lane model disagrees with reference at row {r} chunk {c}")
            n += 1
    return n


# ---------------------------------------------------------------------------
# exact size without packing (the streaming ledger's path)
# ---------------------------------------------------------------------------
def size_from_exponents(e_flat: np.ndarray, shape: Sequence[int], max_cb: int = MAX_CB,
                        target_elems: int = 1 << 23) -> Dict[str, int]:
    """Exact v2.1 stored bytes of one BF16 tensor from its exponent bytes (source order)."""
    g = geometry_for(shape, "BF16")
    if g.mode == MODE_RAW:
        n = int(np.prod(shape)) if shape else 1
        return {"raw": 2 * n, "total": 2 * n}
    e_flat = np.ascontiguousarray(e_flat).reshape(-1)
    hist = np.bincount(e_flat, minlength=256)[:256]
    pad = int(build_codebook(hist)[0])
    if g.mode == MODE_ROWS and g.Kp == g.K:
        e_rows = e_flat.reshape(g.N, g.K)
    else:
        e_rows = np.full(g.N * g.Kp, pad, np.uint8)
        if g.mode == MODE_ROWS:
            e_rows = e_rows.reshape(g.N, g.Kp)
            e_rows[:, :g.K] = e_flat.reshape(g.N, g.K)
        else:
            e_rows[:e_flat.size] = e_flat
            e_rows = e_rows.reshape(g.N, g.Kp)
    cbs, sel = choose_codebooks(e_rows) if max_cb >= 2 else (build_codebook(hist)[None, :], np.zeros(g.N, np.uint8))
    costs = np.stack([exp_cost_lut(c) for c in cbs])
    words = 0
    step = _block_rows(g, target_elems)
    for r0 in range(0, g.N, step):
        r1 = min(g.N, r0 + step)
        rb = costs[sel[r0:r1].astype(np.int64)[:, None], e_rows[r0:r1].astype(np.int64)].sum(1)
        words += int(((rb + 31) // 32).sum())
    d = {"smb": g.N * g.Kp, "l1": g.N * g.nch * 32, "ovf": 4 * words, "rowoff": 4 * g.N,
         "cbsel": (g.N + 7) // 8, "codebook": NSYM * int(cbs.shape[0])}
    d["total"] = sum(d.values())
    d["ncb"] = int(cbs.shape[0])
    return d


# ---------------------------------------------------------------------------
# container I/O
# ---------------------------------------------------------------------------
def to_named_arrays(t: V21Tensor) -> Dict[str, np.ndarray]:
    return {f"{t.name}::{k}": np.ascontiguousarray(v) for k, v in t.streams().items()}


def from_named_arrays(header: dict, arrays: Dict[str, np.ndarray]) -> V21Tensor:
    if header.get("format", FORMAT) != FORMAT:
        raise CodecV21Error(f"{header.get('name')}: format {header.get('format')} is not {FORMAT}")
    name = header["name"]
    g = geometry_for(tuple(header["shape"]), header["dtype"])
    if g.mode != header["mode"]:
        raise CodecV21Error(f"{name}: geometry mode {g.mode} != header {header['mode']}")
    t = V21Tensor(name, header["dtype"], g, header["sha256"])
    t.crc = {k: int(v) for k, v in header.get("crc32", {}).items()}
    if g.mode == MODE_RAW:
        t.raw = arrays[f"{name}::raw"]
        return t
    if tuple(header.get("profile", PROFILE)) != PROFILE:
        raise CodecV21Error(f"{name}: profile {header.get('profile')} unsupported")
    for k in ("N", "K", "Kp", "nch"):
        if int(header[k]) != int(getattr(g, k)):
            raise CodecV21Error(f"{name}: header {k} disagrees with geometry")
    t.smb = arrays[f"{name}::smb"]
    t.l1 = arrays[f"{name}::l1"]
    t.ovf = arrays[f"{name}::ovf"]
    t.rowoff = arrays[f"{name}::rowoff"]
    t.cbsel = arrays[f"{name}::cbsel"]
    t.cb = arrays[f"{name}::codebook"].reshape(-1, NSYM)
    t.counts = dict(header.get("counts", {}))
    exp = {"smb": (g.N, g.Kp), "l1": (g.N, g.nch, 16), "rowoff": (g.N,), "cbsel": ((g.N + 7) // 8,)}
    for k, shp in exp.items():
        a = getattr(t, k)
        if tuple(a.shape) != shp:
            raise CodecV21Error(f"{name}: stream {k} shape {a.shape} != {shp}")
    if not 1 <= t.ncb <= MAX_CB:
        raise CodecV21Error(f"{name}: {t.ncb} codebooks")
    if t.ncb == 1 and t.row_codebook().any():
        raise CodecV21Error(f"{name}: row selects a codebook that does not exist")
    return t


def serialize(t: V21Tensor) -> Tuple[dict, bytes]:
    from safetensors.numpy import save
    return t.header(), save(to_named_arrays(t))


def deserialize(header: dict, blob: bytes) -> V21Tensor:
    from safetensors.numpy import load
    return from_named_arrays(header, load(blob))


__all__ = ["CH", "FORMAT", "FORMAT_VERSION", "PROFILE", "NSYM", "RAW", "MODE_ROWS", "MODE_FLAT",
           "MODE_RAW", "CodecV21Error", "Geometry", "V21Tensor", "build_codebook", "choose_codebooks",
           "chunk_offsets", "decode_rows", "decode_tensor", "deserialize", "encode_tensor",
           "geometry_for", "l1_pack", "l1_unpack", "lane_model_check", "lane_model_decode_chunk",
           "serialize", "size_from_exponents", "split_checkpoints", "split_starts", "symbol_lut",
           "to_named_arrays", "verify_crc"]
