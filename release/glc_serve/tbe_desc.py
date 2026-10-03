"""TBE-coded weights as FastDecoder descriptors (codec read coded, decoded in registers).

``TBELin`` wraps one or more ``TBEServeLinear`` modules with the same K (a fused
row concatenation) as ONE weight for ``miv_gemv.miv_tbe_out``:

* the containers' arrays are concatenated in row order (planes, smb, esc with
  each container's own zero pad kept), and every module is re-pointed at views
  of the concatenation, so the bytes exist once and the HF exact path
  (prefill) keeps reading the very same bytes;
* ``rowparam[row] = base | mode << 8`` keeps every tensor's own exponent window;
* ``escbase[row * WPR + s]`` = index in the concatenated ``esc`` of the first
  escape of K-slice ``s`` of ``row``: a load-time prefix sum over the stored
  planes, checked against every container's own ``sbbase`` (superblock bases)
  before use.

Nothing about the stored format changes: the planes, sign/mantissa bytes and
escape bytes the kernel reads are the bundle's, byte for byte.  The kernel's
FMA order is the bf16 MIV kernel's (same ``fma_vec`` / butterfly / slice sum),
so for G1-identical weights the logits are bitwise the parent's through the
same engine and WPR (G3a; tested, not assumed).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from . import miv_gemv as mg

TILE = 64


class TBELin:
    kind = "tbe"

    def __init__(self, mods: Sequence, cfg: Tuple[int, int], *, verify: bool = True):
        devs, own = [], []
        for m in mods:
            if hasattr(m, "device_container"):
                devs.append(m.device_container())
                own.append(True)
            else:                  # a dense member of a fused group: encode it exactly (G1-checked)
                from glc_loader.tbe_container import decode_tbe, encode_tbe
                from glc_loader.tbe_mma import upload_tbe

                w = m.weight.detach()
                c = encode_tbe(w.cpu(), layout="mma16")
                if not torch.equal(decode_tbe(c).view(torch.int16), w.cpu().view(torch.int16)):
                    raise ValueError("TBE round trip of a dense member failed")
                devs.append(upload_tbe(c, w.device))
                own.append(False)
        self.encoded_dense_members = own.count(False)
        K = int(devs[0].shape[1])
        if any(int(d.shape[1]) != K for d in devs):
            raise ValueError("fused TBE tensors must share K")
        self.wpr, self.unroll = int(cfg[0]), int(cfg[1])
        if K % (TILE * self.wpr):
            raise ValueError(f"K={K} not a multiple of 64*WPR={64 * self.wpr}")
        self.K = K
        self.N = sum(int(d.shape[0]) for d in devs)
        dev = devs[0].planes.device
        E = mg.extension()
        tpr = K // TILE
        if len(devs) == 1:
            d = devs[0]
            planes, smb, esc = d.planes.reshape(-1), d.smb.reshape(-1), d.esc.reshape(-1)
        else:
            planes = torch.cat([d.planes.reshape(-1) for d in devs])
            smb = torch.cat([d.smb.reshape(-1) for d in devs])
            # every segment starts 16-byte aligned: the exact-mode decode kernel reads a
            # module's esc with aligned vector loads from its own base
            parts = []
            for d in devs:
                e = d.esc.reshape(-1)
                pad = (-int(e.numel())) % 16
                parts.append(e)
                if pad:
                    parts.append(torch.zeros(pad, dtype=e.dtype, device=e.device))
            esc = torch.cat(parts)
        rowparam, escbase = [], []
        t_off = e_off = 0
        for d in devs:
            n = int(d.shape[0])
            T = n * tpr
            if int(d.tiles) != T:
                raise ValueError(f"container tiles {d.tiles} != {T}")
            cnt = E.tbe_tile_escapes(planes[t_off * 6:(t_off + T) * 6].contiguous()).to(torch.int64)
            excl = torch.cumsum(cnt, 0) - cnt
            if int(cnt.sum().item()) != int(d.escapes):
                raise ValueError(f"escape count {int(cnt.sum())} != container {d.escapes}")
            if verify:
                sb = int(d.superblock)
                starts = excl[::sb]
                if not torch.equal(starts.to(torch.int64), d.sbbase.reshape(-1)[:starts.numel()]
                                   .to(torch.int64)):
                    raise ValueError("per-tile escape prefix disagrees with the container sbbase")
            rows = torch.arange(n, device=dev, dtype=torch.int64)
            sl = torch.arange(self.wpr, device=dev, dtype=torch.int64)
            tidx = rows[:, None] * tpr + sl[None, :] * (tpr // self.wpr)
            escbase.append((excl[tidx.reshape(-1)] + e_off).to(torch.int32))
            rowparam.append(torch.full((n,), int(d.base) | (int(d.mode) << 8), dtype=torch.int32,
                                       device=dev))
            t_off += T
            e_off += int(d.esc.numel()) + ((-int(d.esc.numel())) % 16 if len(devs) > 1 else 0)
        self.planes, self.smb, self.esc = planes.contiguous(), smb.contiguous(), esc.contiguous()
        self.rowparam = torch.cat(rowparam).contiguous()
        self.escbase = torch.cat(escbase).contiguous()
        if int(self.esc.numel()) >= 2 ** 31:
            raise ValueError("esc index exceeds int32")
        if len(devs) > 1:                         # re-point the modules at views (one copy)
            t_off = e_off = 0
            for d, o in zip(devs, own):
                T = int(d.tiles)
                ne = int(d.esc.numel())
                step = ne + ((-ne) % 16)
                if not o:
                    t_off += T
                    e_off += step
                    continue
                d.planes = self.planes[t_off * 6:(t_off + T) * 6].view(d.planes.shape)
                d.smb = self.smb[t_off * TILE:(t_off + T) * TILE].view(d.smb.shape)
                d.esc = self.esc[e_off:e_off + ne].view(d.esc.shape)
                assert self.esc[e_off:].data_ptr() % 16 == 0
                t_off += T
                e_off += step

    @property
    def bytes(self) -> int:
        return (self.planes.numel() * 4 + self.smb.numel() + self.esc.numel()
                + self.rowparam.numel() * 4 + self.escbase.numel() * 4)

    @property
    def coded_bytes(self) -> int:
        """Bytes the kernel streams per call (planes + sign/mantissa + escapes)."""
        return self.planes.numel() * 4 + self.smb.numel() + self.esc.numel()

    impl = "engine"            # "engine": miv_gemv.miv_tbe_out; "v2": glc_serve.miv_tbe (PRMT decode)

    by_m = None                # optional {M: (impl, unroll)} from tune_tbe(per_m=True)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if self.by_m is not None:
            impl, u = self.by_m.get(int(x.shape[0]), (self.impl, self.unroll))
            self._run(impl, u, x, y)
        else:
            self._run(self.impl, self.unroll, x, y)

    def _run(self, impl, unroll, x, y) -> None:
        if impl == "v2":
            from . import miv_tbe

            miv_tbe.extension().miv_tbe_out(x, self.planes, self.smb, self.esc, self.rowparam,
                                            self.escbase, None, y, self.N, self.K, self.wpr,
                                            unroll, 2)
        else:
            mg.extension().miv_tbe_out(x, self.planes, self.smb, self.esc, self.rowparam,
                                       self.escbase, y, self.N, self.K, self.wpr, unroll)


def _time(ds, impl, u, m, key):
    dev = ds[0].planes.device
    x = torch.randn(m, key[1], device=dev, dtype=torch.bfloat16)
    y = torch.empty(m, key[0], device=dev, dtype=torch.bfloat16)
    for d in ds[:2]:
        d._run(impl, u, x, y)
    torch.cuda.synchronize()
    reps = max(1, int(2e9 // (len(ds) * ds[0].coded_bytes)))
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(reps):
        for d in ds:
            d._run(impl, u, x, y)
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / (reps * len(ds))


def tune_tbe_per_m(descs, *, impls=("engine", "v2"), unrolls=(1, 2, 4), log=print):
    """Per shape AND per M = 1..8, the fastest (impl, unroll).  Identical arithmetic
    (both kernels are bitwise the bf16 MIV chain), so a pure speed choice."""
    groups = {}
    for d in descs:
        if getattr(d, "kind", None) == "tbe":
            groups.setdefault((d.N, d.K), []).append(d)
    avail = []
    for impl in impls:
        try:
            if impl == "v2":
                from . import miv_tbe

                miv_tbe.extension()
            avail.append(impl)
        except Exception:  # noqa: BLE001
            pass
    out = {}
    for key, ds in sorted(groups.items()):
        table, us = {}, {}
        for m in range(1, 9):
            best = None
            for impl in avail:
                for u in unrolls:
                    t = _time(ds, impl, u, m, key)
                    us[f"{impl},U{u},M{m}"] = round(t * 1e3, 2)
                    if best is None or t < best[0]:
                        best = (t, impl, u)
            table[m] = (best[1], best[2])
        for d in ds:
            d.by_m = dict(table)
            d.impl, d.unroll = table[1]
        m1 = us[f"{table[1][0]},U{table[1][1]},M1"]
        out[f"{key[0]}x{key[1]}"] = {"by_m": {str(m): list(v) for m, v in table.items()}, "us": us,
                                     "coded_gbps_m1": round(ds[0].coded_bytes / (m1 * 1e-6) / 1e9, 1)}
        log(f"[tbe-tune] {key[0]}x{key[1]} x{len(ds)}: " +
            " ".join(f"M{m}:{v[0][0]}{v[1]}:{us[f'{v[0]},U{v[1]},M{m}']}us" for m, v in table.items()))
    return out


def tune_tbe(descs, *, ms=(1, 6), impls=("engine", "v2"), unrolls=(1, 2, 4), log=print):
    """Pick (impl, unroll) per shape by measured time over all same-shape descriptors
    (rotation defeats L2).  Both impls and every unroll run the identical arithmetic
    (bitwise-tested), so this is a speed choice only."""
    groups = {}
    for d in descs:
        if getattr(d, "kind", None) == "tbe":
            groups.setdefault((d.N, d.K), []).append(d)
    out = {}
    for key, ds in sorted(groups.items()):
        dev = ds[0].planes.device
        best = None
        rows = {}
        for impl in impls:
            try:
                if impl == "v2":
                    from . import miv_tbe

                    miv_tbe.extension()
            except Exception as e:  # noqa: BLE001
                rows[impl] = f"unavailable: {e!r}"[:200]
                continue
            for u in unrolls:
                tot = 0.0
                for m in ms:
                    x = torch.randn(m, key[1], device=dev, dtype=torch.bfloat16)
                    y = torch.empty(m, key[0], device=dev, dtype=torch.bfloat16)
                    for d in ds:
                        d.impl, d.unroll = impl, u
                    for d in ds[:2]:
                        d(x, y)
                    torch.cuda.synchronize()
                    reps = max(1, int(3e9 // (len(ds) * ds[0].coded_bytes)))
                    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    e0.record()
                    for _ in range(reps):
                        for d in ds:
                            d(x, y)
                    e1.record()
                    torch.cuda.synchronize()
                    t = e0.elapsed_time(e1) / (reps * len(ds))
                    rows[f"{impl},U{u},M{m}"] = round(t * 1e3, 2)
                    tot += t
                if best is None or tot < best[0]:
                    best = (tot, impl, u)
        for d in ds:
            d.impl, d.unroll = best[1], best[2]
        m1 = rows.get(f"{best[1]},U{best[2]},M1")
        out[f"{key[0]}x{key[1]}"] = {"impl": best[1], "unroll": best[2], "us": rows,
                                     "coded_gbps_m1": round(ds[0].coded_bytes / (m1 * 1e-6) / 1e9, 1)
                                     if m1 else None}
        log(f"[tbe-tune] {key[0]}x{key[1]} x{len(ds)}: {best[1]} U={best[2]} "
            f"M1 {m1} us ({out[f'{key[0]}x{key[1]}']['coded_gbps_m1']} coded GB/s)")
    return out


def is_tbe(mod) -> bool:
    return hasattr(mod, "device_container") and hasattr(mod, "exec_mode")


__all__ = ["TBELin", "is_tbe", "tune_tbe", "tune_tbe_per_m"]
