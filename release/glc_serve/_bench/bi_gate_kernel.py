#!/usr/bin/env python3
"""BI-GEMM kernel gates (bitwise; no tolerance anywhere).

G-BI (batch invariance), per (format, shape):
  reference Y = BI(X[:256]) (256 rows).  Fails if ANY row differs, bit for bit, in
    (a) prefixes: BI(X[:M]) vs Y[:M] for M in MS,
    (b) permutation: BI(X[perm]) vs Y[perm],
    (c) composition: the probe rows X[:4] placed at slots {0, 7, 15, 16, 31, 63, 64, 100, 127}
        of a batch whose other rows are fresh random rows, M = slot + 1 + tail,
    (d) repeat: BI(X) twice (run-to-run determinism).
G-EXACT (weight contract through the same GEMM):
  Q8_0 : BI-q8(pack)          == BI-bf16(bf16_rn(d*q))              (dequant = miv_gemv q8_dequant)
  KQ   : BI-kq(desc)          == BI-bf16(miv_kq.dequant_bf16(desc))  (decode twin, G1-q-verified)
  TBE  : BI-tbe(encode_tbe(W)) == BI-bf16(W)                          (codec exact: G1)
Speed (reported, not gated): us per call at M in {1, 8, 16, 32, 64, 128} and weight GB/s.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

MS = [1, 2, 3, 5, 8, 15, 16, 17, 31, 33, 48, 63, 64, 65, 100, 127, 128, 200]
SLOTS = [0, 7, 15, 16, 31, 63, 64, 100, 127]
SHAPES = {"gdn_qkvzba": (16480, 5120), "out_o": (5120, 6144), "attn_qkv": (14336, 5120),
          "gate_up": (34816, 5120), "down": (5120, 17408), "small48": (48, 5120), "kv1024": (1024, 5120)}


def bits(t):
    return t.view(torch.int16)


def rows_diff(a, b):
    return int((bits(a) != bits(b)).any(dim=1).sum().item())


def invariance(bl, K, dev, seed=1):
    g = torch.Generator(device=dev).manual_seed(seed)
    X = torch.randn(256, K, generator=g, device=dev).to(torch.bfloat16)
    N = bl.N
    Y = torch.empty(256, N, dtype=torch.bfloat16, device=dev)
    bl.reserve(256)
    bl(X, Y)
    out = {"prefix": 0, "perm": 0, "composition": 0, "repeat": 0, "rows_checked": 0}
    for M in MS:
        if M > 256:
            continue
        Ym = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
        bl(X[:M].contiguous(), Ym)
        out["prefix"] += rows_diff(Ym, Y[:M])
        out["rows_checked"] += M
    perm = torch.randperm(256, generator=torch.Generator().manual_seed(seed)).to(dev)
    Yp = torch.empty_like(Y)
    bl(X[perm].contiguous(), Yp)
    out["perm"] += rows_diff(Yp, Y[perm])
    out["rows_checked"] += 256
    for slot in SLOTS:
        for tail in (0, 9):
            M = slot + 4 + tail
            Z = torch.randn(M, K, generator=g, device=dev).to(torch.bfloat16)
            Z[slot:slot + 4] = X[:4]
            Yz = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
            bl(Z, Yz)
            out["composition"] += rows_diff(Yz[slot:slot + 4], Y[:4])
            out["rows_checked"] += 4
    Y2 = torch.empty_like(Y)
    bl(X, Y2)
    out["repeat"] += rows_diff(Y2, Y)
    out["fail_rows"] = out["prefix"] + out["perm"] + out["composition"] + out["repeat"]
    return out, X, Y


def exact(bl, ref, K, dev, seed=2):
    g = torch.Generator(device=dev).manual_seed(seed)
    X = torch.randn(96, K, generator=g, device=dev).to(torch.bfloat16)
    a = torch.empty(96, bl.N, dtype=torch.bfloat16, device=dev)
    b = torch.empty_like(a)
    bl.reserve(96)
    ref.reserve(96)
    bl(X, a)
    ref(X, b)
    return rows_diff(a, b)


def sanity(bl, W, K, dev):
    """Not a gate: BI-bf16 vs an fp32 matmul (relative error), to catch a broken kernel."""
    g = torch.Generator(device=dev).manual_seed(3)
    X = torch.randn(16, K, generator=g, device=dev).to(torch.bfloat16)
    y = torch.empty(16, bl.N, dtype=torch.bfloat16, device=dev)
    bl(X, y)
    r = X.float() @ W.float().t()
    return float(((y.float() - r).norm() / r.norm()).item())


def timing(bl, K, dev, ms=(1, 8, 16, 32, 64, 128), reps=20):
    res = {}
    for M in ms:
        X = torch.randn(M, K, device=dev).to(torch.bfloat16)
        Y = torch.empty(M, bl.N, dtype=torch.bfloat16, device=dev)
        bl.reserve(M)
        for _ in range(3):
            bl(X, Y)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(reps):
            bl(X, Y)
        e1.record()
        torch.cuda.synchronize()
        us = e0.elapsed_time(e1) * 1000 / reps
        res[M] = {"us": round(us, 1), "wGBps": round(bl.coded_bytes / us / 1e3, 1) if bl.coded_bytes else None}
    return res


class _Shim(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.weight = torch.nn.Parameter(w, requires_grad=False)
        self.out_features, self.in_features = int(w.shape[0]), int(w.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--formats", default="bf16,q8,tbe,kq")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--gguf", default="", help="GGUF with K-quant tensors for the KQ gate")
    ap.add_argument("--kq-types", default="IQ4_XS,IQ3_S,Q4_K,Q5_K,Q6_K,IQ3_XXS,Q2_K,Q3_K")
    ap.add_argument("--timing", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    from glc_serve import bigemm as bg
    from glc_serve.fastdec import Lin

    ws = bg.Workspace(dev)
    rep = {"device": torch.cuda.get_device_name(0), "results": [], "ms": MS, "slots": SLOTS}
    fails = 0
    fm = a.formats.split(",")
    for sname in a.shapes.split(","):
        N, K = SHAPES[sname]
        torch.manual_seed(hash(sname) & 0xffff)
        W = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
        ref = bg.wrap(Lin(W, (1, 1)), ws)
        if "bf16" in fm:
            inv, _, _ = invariance(ref, K, dev)
            r = {"format": "bf16", "shape": sname, "N": N, "K": K, "S": ref.S, "invariance": inv,
                 "rel_err_vs_fp32": sanity(ref, W, K, dev)}
            if a.timing:
                r["timing"] = timing(ref, K, dev)
            fails += inv["fail_rows"]
            rep["results"].append(r)
            print(json.dumps(r), flush=True)
        if "q8" in fm:
            from glc_serve.q8serve import Q8Lin, Q8Pack

            g = torch.Generator(device=dev).manual_seed(5)
            qs = torch.randint(-127, 128, (N, K), generator=g, device=dev, dtype=torch.int8)
            sc = (torch.rand(N, K // 32, generator=g, device=dev) * 0.002 + 1e-4).to(torch.float16)
            pk = Q8Pack(qs, sc)
            bl = bg.wrap(Q8Lin(pk, (1, 1)), ws)
            refq = bg.wrap(Lin(pk.dequant().contiguous(), (1, 1)), ws)
            inv, _, _ = invariance(bl, K, dev)
            ex = exact(bl, refq, K, dev)
            r = {"format": "q8_0", "shape": sname, "N": N, "K": K, "S": bl.S, "invariance": inv, "exact_rows_diff": ex}
            if a.timing:
                r["timing"] = timing(bl, K, dev)
            fails += inv["fail_rows"] + ex
            rep["results"].append(r)
            print(json.dumps(r), flush=True)
            del qs, sc, pk, bl, refq
        if "tbe" in fm:
            from glc_serve.tbe_desc import TBELin

            t0 = time.time()
            tl = TBELin([_Shim(W)], (1, 1))
            bl = bg.wrap(tl, ws)
            inv, _, _ = invariance(bl, K, dev)
            ex = exact(bl, ref, K, dev)
            r = {"format": "tbe", "shape": sname, "N": N, "K": K, "S": bl.S, "invariance": inv, "exact_rows_diff": ex,
                 "encode_s": round(time.time() - t0, 1), "coded_bytes": tl.coded_bytes}
            if a.timing:
                r["timing"] = timing(bl, K, dev)
            fails += inv["fail_rows"] + ex
            rep["results"].append(r)
            print(json.dumps(r), flush=True)
            del tl, bl
        del W, ref
        torch.cuda.empty_cache()
    if "kq" in fm and a.gguf:
        import gguf
        from glc_serve import miv_kq

        rd = gguf.GGUFReader(a.gguf)
        want = set(a.kq_types.split(","))
        seen = {}
        for t in rd.tensors:
            tn = t.tensor_type.name
            if tn not in want or tn in seen or len(t.shape) != 2:
                continue
            K = int(t.shape[0])
            if K % 256:
                continue
            raw = np.asarray(t.data)
            desc = miv_kq.kq_desc_from_blocks(tn, raw.reshape(int(t.shape[1]), -1), dev, name=t.name)
            bl = bg.wrap(desc, ws)
            refk = bg.wrap(Lin(miv_kq.dequant_bf16(desc).contiguous(), (1, 1)), ws)
            inv, _, _ = invariance(bl, desc.K, dev)
            ex = exact(bl, refk, desc.K, dev)
            r = {"format": f"kq:{tn}", "tensor": t.name, "N": desc.N, "K": desc.K, "S": bl.S, "invariance": inv,
                 "exact_rows_diff": ex}
            if a.timing:
                r["timing"] = timing(bl, desc.K, dev)
            fails += inv["fail_rows"] + ex
            rep["results"].append(r)
            seen[tn] = t.name
            print(json.dumps(r), flush=True)
            del desc, bl, refk
            torch.cuda.empty_cache()
        rep["kq_missing"] = sorted(want - set(seen))
    rep["fail_rows_total"] = fails
    rep["verdict"] = "PASS" if fails == 0 else "FAIL"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print("VERDICT", rep["verdict"], "fail_rows", fails)
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
