"""CPU bitwise emulation of the BI-GEMM TBE21 (codec v2.1) weight decode, as the kernel runs it.

Line-for-line numpy transcription of ``Ld<F_TBE21>`` / ``Drv<Ld<F_TBE21>>`` in
``bi_csrc/bi_gemm.cuh`` (start / load / dec), vectorised over (row, lane), using only the
operations the device code uses -- the warp primitives and SWAR helpers are the ones already
proved for TBE2 in ``bigemm_tbe2_ref`` (``_prmt``, ``_spread2``, ``_spread3``, ``_rotr8x4``,
``_pair``, ``_hscan16``, ``_get``, ``_get_win``).  The proof is ``decode_tensor_as_kernel(st, S)
== source words`` at every split count, plus the cursor at the start of every (row, chunk) ==
``codec_v21.chunk_offsets``.

Device-code correspondence (bi_gemm.cuh)         numpy here
  Ld<F_TBE21>::prologue (resident codebook table) resident_table21
  Ld<F_TBE21>::start    (row offset + checkpoint) _start
  Ld<F_TBE21>::load                                _load
  Ld<F_TBE21>::dec                                 _dec
  bi_gemm_kernel / bi_decode_kernel loop nest      decode_tensor_as_kernel
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import bigemm_tbe2_ref as T2   # noqa: F401  (package import when available)

BN, CH, T2_MAXCB = T2.BN, T2.CH, T2.T2_MAXCB
M32 = T2.M32
OVF_TAIL_WORDS = 16
COUNTERS = T2.COUNTERS
cta_rows, splits, split_for, CARD_SMS = T2.cta_rows, T2.splits, T2.split_for, T2.CARD_SMS


def codec_v21():
    mod = sys.modules.get("glc_loader.codec_v21") or sys.modules.get("codec_v21_for_bigemm")
    if mod is not None:
        return mod
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "glc_loader", "codec_v21.py")
    spec = importlib.util.spec_from_file_location("codec_v21_for_bigemm", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["codec_v21_for_bigemm"] = mod
    spec.loader.exec_module(mod)
    return mod


def resident_table21(cbtab: np.ndarray) -> np.ndarray:
    """``Ld<F_TBE21>::prologue``: stored codebooks (ncb x 16 B, symbols 0..15) -> ncb x 8 u32:
    w0 = R(cb0..3), w1 = R(cb4 cb5 0 0), w2 = R(cb6 cb7 cb8 0), w3 = R(cb9..12),
    w4 = R(cb13 cb14 cb15 0), w5..w7 = 0; R = byte-wise rotate right by one."""
    src = np.ascontiguousarray(cbtab, dtype=np.uint8).reshape(-1, 16).view("<u4").astype(np.int64)
    c0, c1, c2, c3 = (src[:, q] for q in range(4))
    z = np.zeros_like(c0)
    w = np.stack([c0, c1 & 0x0000FFFF, T2._prmt(c1, c2, 0x0432) & 0x00FFFFFF, T2._prmt(c2, c3, 0x4321),
                  c3 >> 8, z, z, z], 1)
    return T2._rotr8x4(w)


class T21Streams:
    """The arrays a TBE21 launch reads, as the loader lays them out (one tensor or a fused
    group along N), with the RESIDENT checkpoints for split count ``S``."""

    def __init__(self, N, K, l1, smb, ovf, rowoff, cbtab, rowp, ckpt, S):
        self.N, self.K, self.S = int(N), int(K), int(S)
        self.nch = self.K // CH
        self.l1 = np.ascontiguousarray(l1, dtype=np.uint16).reshape(self.N, self.nch, 16)
        self.smb = np.ascontiguousarray(smb, dtype=np.uint8).reshape(self.N, self.K)
        self.ovf = np.ascontiguousarray(ovf, dtype=np.uint32).reshape(-1)
        self.rowoff = np.ascontiguousarray(rowoff, dtype=np.uint32).reshape(self.N)
        self.cbtab = np.ascontiguousarray(cbtab, dtype=np.uint8).reshape(-1, 16)
        self.rowp = np.ascontiguousarray(rowp, dtype=np.int32).reshape(self.N)
        self.ckpt = np.ascontiguousarray(ckpt, dtype=np.uint32).reshape(self.N, max(0, self.S - 1))
        if self.K % CH:
            raise ValueError("TBE21 GEMM needs K % 128 == 0")
        if self.cbtab.shape[0] > T2_MAXCB:
            raise ValueError(f"{self.cbtab.shape[0]} codebooks > T2_MAXCB={T2_MAXCB}")
        if self.ovf.size < OVF_TAIL_WORDS:
            raise ValueError("ovf stream lacks the loader's 16-word tail")
        self.restab = resident_table21(self.cbtab)

    @property
    def nbytes(self) -> int:
        return int(self.l1.nbytes + self.smb.nbytes + self.ovf.nbytes + self.rowoff.nbytes
                   + self.cbtab.nbytes + self.rowp.nbytes + self.ckpt.nbytes)


def streams_for(tensors: Sequence, S: int, offsets: Sequence[np.ndarray] = None) -> T21Streams:
    """Fuse ``codec_v21.V21Tensor`` members (rows mode, equal K == Kp) along N as the loader does:
    rows / ovf concatenated, every member's rowoff re-based by the words before it, codebooks
    stacked (rowp = member's codebook base + the row's codebook), checkpoints for ``S``."""
    cv = codec_v21()
    tensors = list(tensors)
    K = int(tensors[0].geom.Kp)
    l1, smb, ovf, roff, cbs, rowp, ck = [], [], [], [], [], [], []
    wbase = 0
    cbase = 0
    for i, t in enumerate(tensors):
        g = t.geom
        if g.mode != cv.MODE_ROWS or int(g.K) != K or int(g.Kp) != K:
            raise ValueError(f"{t.name}: not a fusable GEMM weight (mode {g.mode}, K {g.K})")
        l1.append(t.l1.reshape(-1))
        smb.append(t.smb.reshape(-1))
        ovf.append(t.ovf.reshape(-1))
        roff.append(t.rowoff.astype(np.uint64) + wbase)
        wbase += int(t.ovf.size)
        cbs.append(t.cb.reshape(-1, 16))
        rowp.append((cbase + t.row_codebook().astype(np.int32)))
        cbase += t.ncb
        off = None if offsets is None else offsets[i]
        ck.append(cv.split_checkpoints(t, S, off))
    if wbase + OVF_TAIL_WORDS >= 2 ** 32:
        raise ValueError("fused overflow stream exceeds 2^32 words")
    N = sum(int(t.geom.N) for t in tensors)
    return T21Streams(N, K, np.concatenate(l1), np.concatenate(smb),
                      np.concatenate(ovf + [np.zeros(OVF_TAIL_WORDS, np.uint32)]),
                      np.concatenate(roff).astype(np.uint32), np.concatenate(cbs),
                      np.concatenate(rowp), np.concatenate(ck) if S > 1 else np.zeros((N, 0), np.uint32), S)


def _start(st: T21Streams, rows: np.ndarray, s: int, S: int):
    """``Ld<F_TBE21>::start``: (ow, ob, rp) = rowoff + checkpoint of split s (0 for s == 0)."""
    if s == 0:
        bits = np.zeros(rows.shape, np.int64)
    else:
        if S != st.S:
            raise ValueError(f"checkpoints built for S={st.S}, launched with S={S}")
        bits = st.ckpt[rows, s - 1].astype(np.int64)
    rp = st.rowp[rows].astype(np.int64)
    return st.rowoff[rows].astype(np.int64) + (bits >> 5), bits & 31, rp


def _load(st: T21Streams, rows: np.ndarray, c: int, ow: np.ndarray):
    c16 = st.l1[rows, c].astype(np.int64)                     # [R, 16]
    s8 = st.smb[rows, c * CH:(c + 1) * CH].astype(np.int64).reshape(-1, 16, 8)
    win = st.ovf[ow[:, None] + np.arange(16)[None, :]].astype(np.int64)
    return c16, s8, win


def _merge(x, y, z, hi):
    """``__byte_perm(x, y, 0x3210 | flags << 2)`` for the lo (nibbles 0..3) or hi (4..7) half."""
    sel = ((z >> 14) + 0x3210) if hi else ((z * 4 + 0x3210) & M32)     # t2_msel_hi / t2_msel_lo
    return T2._prmt(x, y, sel)


def _dec(st: T21Streams, c16, s8, win, ow, ob, rp, pair_any):
    """``Ld<F_TBE21>::dec`` for every (row, lane) -> (words [R,128] u16, ow', ob', stats)."""
    P = T2._prmt
    obl = ob[:, None]
    C1 = (c16 | (c16 << 14)) & 0x33333333
    T1 = C1 & (C1 >> 1) & 0x11111111
    P1 = (T1 * 0x11111111) & M32
    k1 = P1 >> 28
    R1 = (P1 - T1) & M32
    rho1, n1 = T2._hscan16(k1)
    tab = st.restab[rp]                                        # [R, 8]
    tx, ty, tz, tw, t4 = (tab[:, q][:, None] for q in range(5))
    E1lo, E1hi = P(tx, ty, C1), P(tx, ty, C1 >> 16)
    f2 = T2._get_win(win, obl + 2 * rho1, 2 * k1)
    N2 = T2._spread2(f2)
    S2 = (N2 + 0x33333333) & M32
    X2lo, X2hi = P(tx, ty, S2), P(tx, ty, S2 >> 16)
    Z2 = N2 & (N2 >> 1) & 0x11111111
    P2 = (Z2 * 0x11111111) & M32
    k2 = P2 >> 28
    R2 = (P2 - Z2) & M32
    rho2, n2 = T2._hscan16(k2)
    f3 = T2._get(win, st.ovf, ow, obl + 2 * n1[:, None] + 2 * rho2, 2 * k2)
    N3 = T2._spread2(f3)
    X3lo, X3hi = P(tz, tz, N3), P(tz, tz, N3 >> 16)
    Z3 = N3 & (N3 >> 1) & 0x11111111
    P3 = (Z3 * 0x11111111) & M32
    k3 = P3 >> 28
    R3 = (P3 - Z3) & M32
    rho3, n3 = T2._hscan16(k3)
    b4 = ob + 2 * n1 + 2 * n2
    f4 = T2._get(win, st.ovf, ow, b4[:, None] + 3 * rho3, 3 * k3)
    N4 = T2._spread3(f4)
    X4lo, X4hi = P(tw, t4, N4), P(tw, t4, N4 >> 16)
    Z4 = N4 & (N4 >> 1) & (N4 >> 2) & 0x11111111
    n4 = np.zeros_like(n1)
    any_raw = pair_any(Z4)
    took = int(any_raw.sum())
    if took:
        P4 = (Z4 * 0x11111111) & M32
        k4 = P4 >> 28
        R4 = (P4 - Z4) & M32
        rho4, n4_all = T2._hscan16(k4)
        n4 = np.where(any_raw, n4_all, 0)
        base = b4[:, None] + 3 * n3[:, None] + 8 * rho4
        g0 = T2._rotr8x4(T2._get(win, st.ovf, ow, base, np.minimum(8 * k4, 32)))
        g1 = T2._rotr8x4(T2._get(win, st.ovf, ow, base + 32, np.maximum(8 * k4 - 32, 0)))
        nlo = _merge(X4lo, P(g0, g1, R4), Z4, False)
        nhi = _merge(X4hi, P(g0, g1, R4 >> 16), Z4, True)
        X4lo = np.where(any_raw[:, None], nlo, X4lo)
        X4hi = np.where(any_raw[:, None], nhi, X4hi)
    X3lo, X3hi = _merge(X3lo, P(X4lo, X4hi, R3), Z3, False), _merge(X3hi, P(X4lo, X4hi, R3 >> 16), Z3, True)
    X2lo, X2hi = _merge(X2lo, P(X3lo, X3hi, R2), Z2, False), _merge(X2hi, P(X3lo, X3hi, R2 >> 16), Z2, True)
    Elo = _merge(E1lo, P(X2lo, X2hi, R1), T1, False)
    Ehi = _merge(E1hi, P(X2lo, X2hi, R1 >> 16), T1, True)
    t = b4 + 3 * n3 + 8 * n4
    ow2, ob2 = ow + (t >> 5), t & 31
    sx = s8[..., 0] | (s8[..., 1] << 8) | (s8[..., 2] << 16) | (s8[..., 3] << 24)
    sy = s8[..., 4] | (s8[..., 5] << 8) | (s8[..., 6] << 16) | (s8[..., 7] << 24)
    wv = np.stack([T2._pair(sx, Elo, 0x5140, 0x1504), T2._pair(sx, Elo, 0x7362, 0x3726),
                   T2._pair(sy, Ehi, 0x5140, 0x1504), T2._pair(sy, Ehi, 0x7362, 0x3726)], -1)
    out = np.stack([wv & 0xFFFF, wv >> 16], -1).reshape(c16.shape + (8,))
    return out.reshape(-1, CH).astype(np.uint16), ow2, ob2, {"raw_warp_iters": took, "region_bits": t - ob}


def decode_tensor_as_kernel(st: T21Streams, S: int, *, cta_block: int = 256, record_offsets: bool = False):
    """The decode loop nest of bi_gemm_kernel / bi_decode_kernel for TBE21: every CTA, every
    split (start = row offset + resident checkpoint), every chunk in order."""
    N, K, nch = st.N, st.K, st.nch
    if not 1 <= S <= nch:
        raise ValueError(f"S={S} outside 1..{nch}")
    out = np.empty((N, K), dtype=np.uint16)
    offs = np.full((N, nch), -1, dtype=np.int64) if record_offsets else None
    rows_all = cta_rows(N)
    info = {"chunks": 0, "raw_warp_iters": 0, "warp_iters": 0, "S": S}
    for b0 in range(0, rows_all.shape[0], cta_block):
        blk = rows_all[b0:b0 + cta_block]
        rows = blk.reshape(-1)
        local = (np.arange(blk.shape[0])[:, None] * BN + b0 * BN + np.arange(BN)[None, :]).reshape(-1)
        store = local < N

        def pair_any(z):
            a = (z != 0).any(axis=1).reshape(-1, 2).any(axis=1)
            return np.repeat(a, 2)

        for s, (c0, c1) in enumerate(splits(nch, S)):
            if c0 >= c1:
                continue
            ow, ob, rp = _start(st, rows, s, S)
            for c in range(c0, c1):
                if offs is not None:
                    offs[rows[store], c] = (32 * ow + ob)[store]
                c16, s8, win = _load(st, rows, c, ow)
                w, ow, ob, sts = _dec(st, c16, s8, win, ow, ob, rp, pair_any)
                out[rows[store], c * CH:(c + 1) * CH] = w[store]
                info["chunks"] += int(store.sum())
                info["raw_warp_iters"] += sts["raw_warp_iters"] // 2
                info["warp_iters"] += rows.size // 2
    if offs is not None:
        info["offsets"] = offs
    return out, info


def reference_offsets(tensors: Sequence, offsets: Sequence[np.ndarray] = None) -> np.ndarray:
    """Frozen-spec chunk offsets of a fused stream set: each member's codec_v21.chunk_offsets,
    shifted by the overflow words of the members before it."""
    cv = codec_v21()
    out, wbase = [], 0
    for i, t in enumerate(tensors):
        O = cv.chunk_offsets(t) if offsets is None else offsets[i]
        out.append(O + 32 * wbase)
        wbase += int(t.ovf.size)
    return np.concatenate(out)


# ----------------------------------------------------------------------------- cost
def decode_ops_per_lane(raw_rate: float = 0.0296) -> Dict[str, float]:
    """MEASURED static sm_80 SASS of one row-chunk of ``Ld<F_TBE21>`` (load + dec + store),
    ``scripts/batchserve/bigemm_opcount.py``: common path 207, + 108 on the raw branch, taken by
    2.96 % of warp-iterations on the whole 2B (CPU proof).  v1 TBE 115, TBE2 174."""
    always, raw_branch = 207.0, 108.0
    return {"always": always, "raw_branch": raw_branch, "expected": always + raw_rate * raw_branch}


# ----------------------------------------------------------------------------- resident bytes
def resident_checkpoint_bytes(N: int, K: int, sms: int) -> int:
    """Bytes of the load-time split checkpoints for an (N, K) weight on a card with ``sms`` SMs:
    4 * N * (S - 1) with S = split_for(N, K, sms) -- zero whenever the launch does not split."""
    return 4 * N * (split_for(N, K, sms) - 1)


__all__ = ["T21Streams", "codec_v21", "decode_ops_per_lane", "decode_tensor_as_kernel", "reference_offsets",
           "resident_checkpoint_bytes", "resident_table21", "streams_for"]


def checkpoints_by_walk(st: T21Streams, S: int) -> np.ndarray:
    """``bi_t21_ckpt_kernel`` transcribed: the S = 1 decode walk, cursor recorded at every split
    start of S, relative to 32 * rowoff[row] -> u32 [N, S-1]."""
    if S <= 1:
        return np.zeros((st.N, 0), np.uint32)
    _, info = decode_tensor_as_kernel(st, 1, record_offsets=True)
    cs = [(s * st.nch) // S for s in range(1, S)]
    rel = info["offsets"][:, cs] - st.rowoff.astype(np.int64)[:, None] * 32
    return rel.astype(np.uint32)


__all__ += ["checkpoints_by_walk"]
