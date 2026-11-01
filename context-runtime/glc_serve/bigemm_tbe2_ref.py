"""CPU bitwise emulation of the BI-GEMM TBE2 (codec v2) weight decode, as the kernel runs it.

What this module is
-------------------
``bi_csrc/bi_gemm.cuh`` decodes codec-v2 weights inside the GEMM with ``Ld<F_TBE2>`` (driven
through ``Drv<Ld<F_TBE2>>``: ``start`` / ``load`` / ``dec``).  No CUDA card is available where
this was written, so the kernel's decode is proved here: every function below is a line-for-line
numpy transcription of the device code, vectorised over (row, lane) instead of run by threads,
using ONLY the operations the device code uses (32-bit and/or/shift, ``__popc``,
``__funnelshift_r``, ``__shfl_sync(.., width 16)``, ``__shfl_up_sync`` ladders, ``__dp4a``,
``__byte_perm``, ``__any_sync``).  The proof is ``decode_tensor_as_kernel(t, S) == source bits``
for every element, no tolerance, and a cursor equality at every chunk of every split.

Device-code correspondence (bi_gemm.cuh)          numpy here
  hscan16                                          _hscan16
  t2_get (window shuffle | slow-path global load)  _get
  Ld<F_TBE2>::start   (split start, random access) _start
  Ld<F_TBE2>::load    (one chunk ahead)            _load
  Ld<F_TBE2>::dec     (levels 1-3, raw, assembly,  _dec
                       cursor step)
  bi_gemm_kernel / bi_decode_kernel loop nest      decode_tensor_as_kernel

Kernel facts the emulation encodes (and therefore proves are sufficient):
  * a CTA owns 64 weight rows ``n0 .. n0+63``; tail rows are CLAMPED to ``N-1`` (duplicate decode,
    never stored); local rows ``2k`` / ``2k+1`` share a warp (half 0 / half 1) -- for WG = 1 AND
    WG = 2 (``rowl = warp*2*NR + 2*i + half``), so the only cross-row coupling, the warp-uniform
    ``__any_sync`` raw branch, pairs the same two rows at either tile height;
  * split ``s`` of ``S`` runs chunks ``c0 = s*nch//S .. c1 = (s+1)*nch//S`` in increasing order;
    ``start`` does the O(1) random access at ``c0`` (4 x u32 ``len`` words, byte-masked, ``__dp4a``)
    and every later chunk uses the cursor carried in registers as (word, bit) -- never an index load;
  * the overflow window is the 16 words ``ovf[ow .. ow+15]``, one per lane; a bit range that
    reaches past word 15 (a region > 480 bits, or its tail) is read from global memory instead.
    The loader pads ``ovf`` with 16 zero words and ``len`` with 16 zero bytes, so neither the
    window load nor the masked ``len`` read can fault.

``chunk_offsets`` / ``kernel_chunk_offset`` / ``kernel_next_offset`` in ``glc_loader.codec_v2``
are the frozen cursor spec; ``lane_model_decode_chunk`` the frozen decode spec.  The tests compare
this emulation with the source words (the identity), with ``chunk_offsets`` at every chunk, and
with ``lane_model_decode_chunk`` on sampled chunks.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---- kernel constants (bi_gemm.cuh: bi::BN, bi::CH, bi::T2_MAXCB) ----------------------------
BN = 64                 # weight rows per CTA
CH = 128                # K columns per chunk
GROUP = 16              # chunks per index group (codec v2)
T2_MAXCB = 8            # codebooks per launch (fused row groups: q|k|v, gate|up, qkv|z|b|a)
T2_TAB_BYTES = 16 * T2_MAXCB
OVF_TAIL_WORDS = 16     # loader tail after ovf (codec_v2.LOADER_TAIL_PAD_BYTES = 16 + 16*4)
LEN_TAIL_BYTES = 16     # loader tail after len
M32 = (1 << 32) - 1
#: lanes that took the slow path (a bit range past window word 15), summed over every _get call
COUNTERS = {"slow_gets": 0, "gets": 0}


def codec_v2():
    """``glc_loader.codec_v2`` (the frozen format), importable without the package __init__."""
    mod = sys.modules.get("glc_loader.codec_v2") or sys.modules.get("codec_v2_for_bigemm")
    if mod is not None:
        return mod
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "glc_loader", "codec_v2.py")
    spec = importlib.util.spec_from_file_location("codec_v2_for_bigemm", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["codec_v2_for_bigemm"] = mod
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------- device streams
class T2Streams:
    """The arrays the kernel reads, exactly as the loader lays them out on the device.

    One launch = one ``T2Streams``: a single v2 tensor, or several fused along N (q|k|v ...).
    ``rowp[row] = cb_index | unit_shift << 8`` (``unit_shift`` = log2 len_unit, 1..3); the
    codebook table is ``ncb`` x 16 bytes (cb[0..12] then three zero bytes).
    """

    def __init__(self, N: int, K: int, planes: np.ndarray, smb: np.ndarray, ovf: np.ndarray,
                 len_: np.ndarray, grp: np.ndarray, cbtab: np.ndarray, rowp: np.ndarray):
        self.N, self.K = int(N), int(K)
        self.nch = self.K // CH
        self.nchp = ((self.nch + 3) // 4) * 4
        self.ngrp = (self.nch + GROUP - 1) // GROUP
        self.planes = np.ascontiguousarray(planes, dtype=np.uint32).reshape(self.N, self.nch, 8)
        self.smb = np.ascontiguousarray(smb, dtype=np.uint8).reshape(self.N, self.K)
        self.ovf = np.ascontiguousarray(ovf, dtype=np.uint32).reshape(-1)       # incl. tail
        self.len = np.ascontiguousarray(len_, dtype=np.uint8).reshape(-1)       # incl. tail
        self.grp = np.ascontiguousarray(grp, dtype=np.uint32).reshape(self.N, self.ngrp)
        self.cbtab = np.ascontiguousarray(cbtab, dtype=np.uint8).reshape(-1, 16)
        self.rowp = np.ascontiguousarray(rowp, dtype=np.int32).reshape(self.N)
        if self.K % CH:
            raise ValueError(f"TBE2 GEMM needs K % {CH} == 0 (got {self.K}); padding columns are "
                             "not zero weights, so a padded tensor cannot enter a GEMM")
        if self.cbtab.shape[0] > T2_MAXCB:
            raise ValueError(f"{self.cbtab.shape[0]} codebooks > T2_MAXCB={T2_MAXCB}")
        if self.len.size < self.N * self.nchp + LEN_TAIL_BYTES:
            raise ValueError("len stream lacks the loader's 16-byte tail")
        if self.ovf.size < OVF_TAIL_WORDS:
            raise ValueError("ovf stream lacks the loader's 16-word tail")
        self.planes_b = self.planes.view(np.uint8).reshape(self.N, self.nch, 32)
        self.len_w = self.len[: (self.len.size // 4) * 4].view("<u4")

    @property
    def nbytes(self) -> int:
        return int(self.planes.nbytes + self.smb.nbytes + self.ovf.nbytes + self.len.nbytes
                   + self.grp.nbytes + self.cbtab.nbytes + self.rowp.nbytes)


def _unit_shift(u: int) -> int:
    return {2: 1, 4: 2, 8: 3}[int(u)]


def streams_for(tensors: Sequence) -> T2Streams:
    """Fuse one or more ``codec_v2.V2Tensor`` (rows mode, equal K) along N, as the loader does:
    rows concatenated, ``ovf`` concatenated, every member's ``grp`` re-based by the words before
    it, one codebook per member, ``rowp`` = (member index, member's unit shift)."""
    cv = codec_v2()
    tensors = list(tensors)
    if not tensors:
        raise ValueError("no tensors")
    K = int(tensors[0].geom.Kp)
    planes, smb, ovf, lens, grp, cbs, rowp = [], [], [], [], [], [], []
    wbase = 0
    for i, t in enumerate(tensors):
        g = t.geom
        if g.mode != cv.MODE_ROWS:
            raise ValueError(f"{t.name}: mode {g.mode} is not a GEMM weight")
        if int(g.Kp) != K or int(g.K) != K:
            raise ValueError(f"{t.name}: K={g.K} Kp={g.Kp}; fused members need K == Kp == {K}")
        planes.append(np.asarray(t.planes, dtype=np.uint32).reshape(-1))
        smb.append(np.asarray(t.smb, dtype=np.uint8).reshape(-1))
        ovf.append(np.asarray(t.ovf, dtype=np.uint32).reshape(-1))
        lens.append(np.asarray(t.len_, dtype=np.uint8).reshape(-1))
        grp.append(np.asarray(t.grp, dtype=np.uint64).reshape(-1) + wbase)
        wbase += int(np.asarray(t.ovf).size)
        cb = np.zeros(16, dtype=np.uint8)
        cb[:cv.NSYM] = np.asarray(t.cb, dtype=np.uint8)
        cbs.append(cb)
        rowp.append(np.full(int(g.N), i | (_unit_shift(t.len_unit) << 8), dtype=np.int32))
    if wbase + OVF_TAIL_WORDS >= 2 ** 32:
        raise ValueError("fused overflow stream exceeds 2^32 words")
    N = sum(int(t.geom.N) for t in tensors)
    return T2Streams(
        N, K, np.concatenate(planes), np.concatenate(smb),
        np.concatenate(ovf + [np.zeros(OVF_TAIL_WORDS, np.uint32)]),
        np.concatenate(lens + [np.zeros(LEN_TAIL_BYTES, np.uint8)]),
        np.concatenate(grp).astype(np.uint32), np.stack(cbs), np.concatenate(rowp))


# ----------------------------------------------------------------------------- warp primitives
def _popc8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.int64)
    c = np.zeros_like(x)
    for b in range(8):
        c += (x >> b) & 1
    return c


def _hscan16(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``hscan16``: exclusive prefix over the 16 lanes (last axis) by the __shfl_up_sync ladder
    (off = 1, 2, 4, 8; a lane below ``off`` keeps its own value), total by __shfl_sync(.., 15, 16)."""
    inc = v.astype(np.int64).copy()
    off = 1
    while off < 16:
        t = np.concatenate([inc[..., :off], inc[..., :-off]], axis=-1)   # shfl_up: own value if j < off
        j = np.arange(16)
        inc = np.where(j >= off, inc + t, inc)
        off <<= 1
    return inc - v, inc[..., 15]


def _shfl16(win: np.ndarray, src: np.ndarray) -> np.ndarray:
    """``__shfl_sync(full, win, src, 16)``: srcLane taken modulo the 16-lane width."""
    return np.take_along_axis(win, (src & 15).astype(np.int64), axis=-1)


def _get(win: np.ndarray, ovf: np.ndarray, ow: np.ndarray, q: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``t2_get``: bits [q, q+w) past word ``ow`` of the overflow stream (w <= 32).

    Both shuffles always run (they are warp-collective); a lane whose range ends past window
    word 15 replaces the two words with global loads (the slow path).  w == 0 -> 0.
    """
    lo = q >> 5
    a = _shfl16(win, lo)
    b = _shfl16(win, lo + 1)
    slow = (w != 0) & (((q + w - 1) >> 5) > 15)
    COUNTERS["gets"] += int((w != 0).sum())
    COUNTERS["slow_gets"] += int(slow.sum())
    if slow.any():
        owb = np.broadcast_to(ow[..., None], q.shape)
        ia = (owb + lo)[slow]
        a = a.copy(); b = b.copy()
        a[slow] = ovf[ia]
        b[slow] = ovf[ia + 1]
    v = (((b.astype(np.uint64) << np.uint64(32)) | a.astype(np.uint64))
         >> (q & 31).astype(np.uint64)) & np.uint64(M32)
    v = v.astype(np.int64)
    mask = np.where(w >= 32, M32, (np.int64(1) << np.minimum(w, 31)) - 1)
    return v & mask


def _byte_perm_lo(x: np.ndarray, y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """``__byte_perm(x, y, s & 7) & 0xff`` -- byte ``s & 7`` of the 8-byte value y:x."""
    s = s & 7
    return np.where(s < 4, (x >> (8 * s)) & 0xFF, (y >> (8 * (s - 4))) & 0xFF)


# ----------------------------------------------------------------------------- Ld<F_TBE2>
def _start(st: T2Streams, rows: np.ndarray, c: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``Ld<F_TBE2>::start``: (ow, ob, rp) at chunk ``c`` -- grp word + unit * (masked len sum)."""
    g = c >> 4
    r = c & 15
    base_w = (rows.astype(np.int64) * st.nchp + 16 * g) // 4           # u32-aligned (nchp % 4 == 0)
    tot = np.zeros(rows.shape, dtype=np.int64)
    for q in range(4):
        keep = min(max(r - 4 * q, 0), 4)
        m = M32 if keep >= 4 else (1 << (8 * keep)) - 1
        wv = st.len_w[base_w + q].astype(np.int64) & m
        tot += (wv & 0xFF) + ((wv >> 8) & 0xFF) + ((wv >> 16) & 0xFF) + ((wv >> 24) & 0xFF)  # __dp4a
    rp = st.rowp[rows].astype(np.int64)
    ush = (rp >> 8) & 0xFF
    bits = tot << ush
    gw = st.grp[rows, g].astype(np.int64)
    return gw + (bits >> 5), bits & 31, rp


def _load(st: T2Streams, rows: np.ndarray, c: int, ow: np.ndarray):
    """``Ld<F_TBE2>::load``: plane bytes j / 16+j, the 8 smb bytes, window word ow+j."""
    pl = st.planes_b[rows, c].astype(np.int64)                 # [R, 32]
    pb = pl[:, :16] | (pl[:, 16:32] << 8)                       # [R, 16]
    s8 = st.smb[rows, c * CH:(c + 1) * CH].astype(np.int64).reshape(-1, 16, 8)
    win = st.ovf[ow[:, None] + np.arange(16)[None, :]].astype(np.int64)
    return pb, s8, win


def _dec(st: T2Streams, pb, s8, win, ow, ob, rp, c: int, pair_any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """``Ld<F_TBE2>::dec`` for every (row, lane): returns (words [R,128] uint16, ow', ob', stats).

    ``pair_any(e3)`` = the warp's ``__any_sync`` (rows of one warp OR-ed together)."""
    obl = ob[:, None]
    b0 = pb & 0xFF
    b1 = (pb >> 8) & 0xFF
    # level 1: code 3 = escape
    e1 = b0 & b1
    k1 = _popc8(e1)
    rho1, n1 = _hscan16(k1)
    # level 2: the lane's escaped elements own entries rho1 .. rho1+k1-1 (2 bits each)
    f2 = _get(win, st.ovf, ow, obl + 2 * rho1, 2 * k1)
    sym = np.zeros_like(pb)
    e2 = np.zeros_like(pb)
    cnt = np.zeros_like(pb)
    for i in range(8):
        code = ((b0 >> i) & 1) | (((b1 >> i) & 1) << 1)
        d2 = (f2 >> (2 * cnt)) & 3
        esc = (e1 >> i) & 1
        s = np.where(esc == 1, 3 + d2, code)
        e2 |= (esc & (d2 == 3)) << i
        cnt += esc
        sym |= s << (4 * i)
    # level 3: 3 bits each, after the n1 level-2 digits
    k2 = _popc8(e2)
    rho2, n2 = _hscan16(k2)
    f3 = _get(win, st.ovf, ow, obl + 2 * n1[:, None] + 3 * rho2, 3 * k2)
    e3 = np.zeros_like(pb)
    cnt = np.zeros_like(pb)
    for i in range(8):
        d3 = (f3 >> (3 * cnt)) & 7
        esc = (e2 >> i) & 1
        nib = 0xF << (4 * i)
        sym = np.where(esc == 1, (sym & ~nib) | ((6 + d3) << (4 * i)), sym)
        e3 |= (esc & (d3 == 7)) << i
        cnt += esc
    # raw: warp-uniform branch; 8 bits each after the level-3 digits; <= 8 bytes per lane as
    # two 32-bit gets
    n3 = np.zeros_like(n1)
    raw = np.zeros(pb.shape + (8,), dtype=np.int64)
    any_raw = pair_any(e3)                                      # [R] bool, warp-uniform
    took = int(any_raw.sum())
    if took:
        k3 = _popc8(e3)
        rho3, n3_all = _hscan16(k3)
        n3 = np.where(any_raw, n3_all, 0)
        base = obl + 2 * n1[:, None] + 3 * n2[:, None] + 8 * rho3
        g0 = _get(win, st.ovf, ow, base, np.minimum(8 * k3, 32))
        g1 = _get(win, st.ovf, ow, base + 32, np.maximum(8 * k3 - 32, 0))
        cnt = np.zeros_like(pb)
        for i in range(8):
            esc = (e3 >> i) & 1
            sh = 8 * cnt
            byte = np.where(sh < 32, (g0 >> np.minimum(sh, 31)) & 0xFF, (g1 >> np.maximum(sh - 32, 0)) & 0xFF)
            raw[..., i] = np.where(esc == 1, byte, 0)
            cnt += esc
        raw = np.where(any_raw[:, None, None], raw, 0)
        e3 = np.where(any_raw[:, None], e3, 0)
    # symbol -> exponent: the row's 16-byte codebook from the smem table, PRMT
    cbo = rp & 0xFF
    tab = st.cbtab[cbo].astype(np.int64)                         # [R, 16]
    cx = tab[:, 0] | (tab[:, 1] << 8) | (tab[:, 2] << 16) | (tab[:, 3] << 24)
    cy = tab[:, 4] | (tab[:, 5] << 8) | (tab[:, 6] << 16) | (tab[:, 7] << 24)
    cz = tab[:, 8] | (tab[:, 9] << 8) | (tab[:, 10] << 16) | (tab[:, 11] << 24)
    cw = tab[:, 12] | (tab[:, 13] << 8) | (tab[:, 14] << 16) | (tab[:, 15] << 24)
    out = np.empty(pb.shape + (8,), dtype=np.int64)
    for i in range(8):
        s = (sym >> (4 * i)) & 15
        ex = np.where(s < 8, _byte_perm_lo(cx[:, None], cy[:, None], s),
                      _byte_perm_lo(cz[:, None], cw[:, None], s))
        ex = np.where(((e3 >> i) & 1) == 1, raw[..., i], ex)
        sm = s8[..., i]
        out[..., i] = ((sm & 0x80) << 8) | (ex << 7) | (sm & 0x7F)
    # cursor: region rounded up to the length unit, word-aligned when chunk c+1 opens a group
    L = 2 * n1 + 3 * n2 + 8 * n3
    ush = (rp >> 8) & 0xFF
    reg = ((L + (np.int64(1) << ush) - 1) >> ush) << ush
    t = ob + reg
    ow2 = ow + (t >> 5)
    ob2 = t & 31
    if ((c + 1) & 15) == 0:
        bump = ob2 != 0
        ow2 = ow2 + bump
        ob2 = np.where(bump, 0, ob2)
    stats = {"raw_warp_iters": took, "region_bits": reg}
    return out.reshape(-1, CH).astype(np.uint16), ow2, ob2, stats


# ----------------------------------------------------------------------------- the loop nest
def cta_rows(N: int) -> np.ndarray:
    """rowg of every (CTA, local row): [nCTA, 64], clamped to N-1 like the kernel."""
    nct = (N + BN - 1) // BN
    return np.minimum(np.arange(nct)[:, None] * BN + np.arange(BN)[None, :], N - 1)


def splits(nch: int, S: int) -> List[Tuple[int, int]]:
    return [((s * nch) // S, ((s + 1) * nch) // S) for s in range(S)]


def decode_tensor_as_kernel(st: T2Streams, S: int, *, cta_block: int = 256,
                            record_offsets: bool = False):
    """Run the decode loop nest of ``bi_gemm_kernel`` / ``bi_decode_kernel`` over the whole
    weight: every CTA (x), every split (y), every chunk of the split in order.  Returns
    ``(words uint16[N, K], info)``; ``info["offsets"]`` (if asked) is the cursor bit offset
    (32*ow + ob) at the START of every (row, chunk), as the kernel holds it."""
    N, K, nch = st.N, st.K, st.nch
    if not 1 <= S <= nch:
        raise ValueError(f"S={S} outside 1..{nch}")
    out = np.empty((N, K), dtype=np.uint16)
    offs = np.full((N, nch), -1, dtype=np.int64) if record_offsets else None
    rows_all = cta_rows(N)                                       # [nCTA, 64]
    info = {"chunks": 0, "raw_warp_iters": 0, "warp_iters": 0, "S": S}
    for b0 in range(0, rows_all.shape[0], cta_block):
        blk = rows_all[b0:b0 + cta_block]                       # [B, 64]
        rows = blk.reshape(-1)
        local = (np.arange(blk.shape[0])[:, None] * BN + b0 * BN + np.arange(BN)[None, :]).reshape(-1)
        store = local < N                                        # tail rows: decoded, not stored

        def pair_any(e3):                                        # warp = local rows 2k, 2k+1
            a = (e3 != 0).any(axis=1).reshape(-1, 2).any(axis=1)
            return np.repeat(a, 2)

        for c0, c1 in splits(nch, S):
            if c0 >= c1:
                continue
            ow, ob, rp = _start(st, rows, c0)
            for c in range(c0, c1):
                if offs is not None:
                    offs[rows[store], c] = (32 * ow + ob)[store]
                pb, s8, win = _load(st, rows, c, ow)
                w, ow, ob, sts = _dec(st, pb, s8, win, ow, ob, rp, c, pair_any)
                out[rows[store], c * CH:(c + 1) * CH] = w[store]
                info["chunks"] += int(store.sum())
                info["raw_warp_iters"] += sts["raw_warp_iters"] // 2
                info["warp_iters"] += rows.size // 2
    if offs is not None:
        info["offsets"] = offs
    return out, info


def reference_offsets(st: T2Streams) -> np.ndarray:
    """Frozen-spec chunk offsets for a (possibly fused) stream set: 32*grp[row, c>>4] + unit *
    (sum of the group's earlier len bytes) -- ``codec_v2.chunk_offsets`` on the fused arrays."""
    L = st.len[: st.N * st.nchp].reshape(st.N, st.nchp)[:, :st.nch].astype(np.int64)
    unit = np.int64(1) << ((st.rowp.astype(np.int64) >> 8) & 0xFF)
    L = L * unit[:, None]
    gp = st.ngrp * GROUP
    Lp = np.zeros((st.N, gp), np.int64)
    Lp[:, :st.nch] = L
    G = Lp.reshape(st.N, st.ngrp, GROUP)
    excl = (np.cumsum(G, -1) - G).reshape(st.N, gp)[:, :st.nch]
    return st.grp.astype(np.int64)[:, np.arange(st.nch) // GROUP] * 32 + excl


def source_rows(words: np.ndarray, K: int) -> np.ndarray:
    return np.ascontiguousarray(words).reshape(-1, K)


# ----------------------------------------------------------------------------- split policy
def split_for(N: int, K: int, sms: int) -> int:
    """``glc_serve.bigemm.split_for`` (copied: it imports torch); a function of (N, K, card)."""
    ctas = (N + BN - 1) // BN
    s = max(1, min(8, (2 * sms) // max(ctas, 1)))
    nch = K // CH
    while s > 1 and nch // s < 4:
        s -= 1
    return s


#: SM counts of the two acceptance cards (A100-SXM4/PCIe-40GB: 108; RTX PRO 6000 Blackwell: 188).
CARD_SMS = {"a100_40": 108, "rtxpro6000": 188}


# ----------------------------------------------------------------------------- cost model
def decode_ops_per_lane(raw_rate: float = 0.192) -> Dict[str, float]:
    """Instruction-class count of ``Ld<F_TBE2>::dec`` + ``load`` per lane per chunk, read off the
    device code (not measured).  ``raw_rate`` = fraction of warp-iterations taking the raw branch
    (0.192 on the whole 2B, CODEC_V2_FORMAT section 6)."""
    scans = 2 * (4 * 2 + 1)            # 2 always-on scans: 4 x (shfl_up + predicated add) + total shfl
    gets = 2 * (2 + 6)                 # 2 window gets: 2 shfl + q, lo, slow test, funnel, mask
    lvl = 8 * 7 + 8 * 7                # level-2 / level-3 deposit loops, 7 ALU each, unrolled
    sym2exp = 8 * 4 + 1                # 8 x (PRMT x2 + select + raw select) + one LDS.128
    assemble = 8 * 4 + 4               # sign / exponent / mantissa splice + 4 packs
    cursor = 8
    level1 = 4
    loads = 4                          # 2 x u8 planes, 1 x u2 smb, 1 x u32 window
    always = scans + gets + lvl + sym2exp + assemble + cursor + level1 + loads
    raw_branch = (4 * 2 + 1) + 2 * 8 + 8 * 6 + 2   # third scan + 2 gets + 8 byte extracts + vote
    return {"always": float(always), "raw_branch": float(raw_branch),
            "expected": float(always + raw_rate * raw_branch)}


__all__ = ["BN", "CH", "CARD_SMS", "T2Streams", "T2_MAXCB", "codec_v2", "cta_rows",
           "decode_ops_per_lane", "decode_tensor_as_kernel", "reference_offsets", "source_rows",
           "split_for", "splits", "streams_for"]


# Shared bit-field helpers used by the independent v2.1 reference.
def _get_win(win: np.ndarray, q: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``t2_get_win``: a read that must end inside the 16-word window (asserted, not assumed):
    two shuffles, funnel shift, mask to w < 32 bits."""
    lo = q >> 5
    if np.any((w > 0) & (((q + w - 1) >> 5) > 15)):
        raise AssertionError("t2_get_win read past the window")
    a = _shfl16(win, lo)
    b = _shfl16(win, lo + 1)
    v = (((b.astype(np.uint64) << np.uint64(32)) | a.astype(np.uint64)) >> (q & 31).astype(np.uint64)) & np.uint64(M32)
    return v.astype(np.int64) & ((np.int64(1) << w) - 1)

def _u32(x):
    return np.asarray(x, dtype=np.int64) & M32

def _prmt(x, y, s):
    """``__byte_perm(x, y, s)`` (default mode): result byte n = byte ((s >> 4n) & 7) of {y, x}.
    Selectors here never set the sign-replicate bit (every nibble value is <= 7)."""
    x = _u32(x); y = _u32(y); s = _u32(s)
    v = (y << 32) | x
    r = np.zeros(np.broadcast(x, y, s).shape, dtype=np.int64)
    for n in range(4):
        sel = (s >> (4 * n)) & 7
        r |= ((v >> (8 * sel)) & 0xFF) << (8 * n)
    return r

def _spread8(x):
    """``t2_spread8``: nibble i (bit 4i) <- bit i of x."""
    x = _u32(x)
    x = (x | (x << 12)) & 0x000F000F
    x = (x | (x << 6)) & 0x03030303
    return (x | (x << 3)) & 0x11111111

def _spread2(x):
    """``t2_spread2``: nibble i <- 2-bit field i of x."""
    x = _prmt(x, 0, 0x4140)
    x = (x | (x << 4)) & 0x0F0F0F0F
    return (x | (x << 2)) & 0x33333333

def _spread3(x):
    """``t2_spread3``: nibble i <- 3-bit field i of x."""
    x = _u32(x)
    x = (x & 0x00000FFF) | ((x << 4) & 0x0FFF0000)
    x = (x & 0x003F003F) | ((x << 2) & 0x3F003F00)
    return (x & 0x07070707) | ((x << 1) & 0x70707070)

def _rotr8x4(x):
    """``t2_rotr8x4``: every byte rotated right by one bit."""
    x = _u32(x)
    return ((x >> 1) & 0x7F7F7F7F) | ((x << 7) & 0x80808080)

def _pair(s, e, selx, sely):
    """``t2_pair``: two bf16 words from sign|mantissa bytes and rotated exponent bytes."""
    return (_prmt(s, e, selx) & 0x7F7F7F7F) | (_prmt(s, e, sely) & 0x80808080)
