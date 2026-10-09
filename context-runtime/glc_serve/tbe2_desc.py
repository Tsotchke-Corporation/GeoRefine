"""TBE2 (GLC codec v2) descriptors for FastDecoder / BI-GEMM.

``TBE2Linear`` is the ``nn.Module`` the v2 loader (``glc_serve.tbe2_load``) puts where an
``nn.Linear`` was: it holds the tensor's coded streams on the device (``codec_v2_artifact.V2Device``)
and nothing dense.  ``TBE2Lin`` is the descriptor FastDecoder builds from one module or from a fused
group (gate|up, q|k|v, qkv|z|b|a): the members' streams concatenated along N, every member's group
words re-based by the overflow words before it, one 16-byte codebook per member (smem table in the
kernel, <= ``T2_MAXCB`` = 8) and a per-row parameter ``codebook index | log2(len_unit) << 8``.
This is exactly ``glc_serve.bigemm_tbe2_ref.streams_for`` -- the layout the CPU proof decodes
bitwise (``scripts/batchserve/bigemm_v2_cpu_proof.py``, fused groups included).

The descriptor's only device path is BI-GEMM (``bigemm.wrap`` -> ``BILin("tbe2", ...)``); calling
it directly runs the same kernel at the 64-row tile.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T2_MAXCB = 8
_U32 = 1 << 32


def _u32_to_i32(x64: torch.Tensor) -> torch.Tensor:
    """int64 values in [0, 2^32) -> the int32 tensor with the same 32-bit patterns."""
    if bool((x64 < 0).any()) or bool((x64 >= _U32).any()):
        raise ValueError("value outside u32")
    return torch.where(x64 >= (1 << 31), x64 - _U32, x64).to(torch.int32)


def _i32_to_u32(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.int64) & 0xFFFFFFFF


class TBE2Linear(nn.Module):
    """A linear layer whose weight lives on the device as codec-v2 streams only."""

    def __init__(self, dev, bias: Optional[torch.Tensor] = None, name: str = ""):
        super().__init__()
        self.codec_v2 = dev
        self.name = name or dev.name
        self.out_features, self.in_features = int(dev.N), int(dev.K)
        if bias is not None:
            self.bias = nn.Parameter(bias.detach(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def dense_weight(self) -> torch.Tensor:
        """The exact bf16 weight, decoded by the kernel's own decode path (bi_decode_kernel)."""
        from . import bigemm as bg

        d = self.codec_v2
        return bg.decode_weight("tbe2", d.N, d.K, arrays_of([d], [0]), S=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # non-BI fallback paths only
        return F.linear(x, self.dense_weight(), self.bias)


def arrays_of(devs: Sequence, rowp_parts: Sequence) -> List[torch.Tensor]:
    """bi_torch.cpp argument order for a single tensor (rowp built from the member index)."""
    d = devs[0]
    unit_shift = {2: 1, 4: 2, 8: 3}[int(d.len_unit)]
    rowp = torch.full((d.N,), int(rowp_parts[0]) | (unit_shift << 8), dtype=torch.int32,
                      device=d.smb.device)
    return [d.planes, d.smb, d.ovf, d.len, d.grp, d.cb16, rowp]


def _encode_dense(m: nn.Module, device):
    """A dense member of a fused group: encode it exactly (round trip checked) and upload."""
    from glc_loader import codec_v2 as cv
    from glc_loader.codec_v2_artifact import upload

    w = m.weight.detach().to("cpu").contiguous()
    words = w.view(torch.int16).numpy().view(np.uint16)
    t = cv.encode_tensor(getattr(m, "name", "dense_member"), words)
    if not np.array_equal(cv.decode_tensor(t, check_sha=True), words):
        raise ValueError("TBE2 round trip of a dense member failed")
    return upload(t, device)


class TBE2Lin:
    kind = "tbe2"

    def __init__(self, mods: Sequence[nn.Module], *, repoint: bool = True, bi: bool = True):
        if not 1 <= len(mods) <= T2_MAXCB:
            raise ValueError(f"TBE2 fused group of {len(mods)} members (1..{T2_MAXCB})")
        dev0 = next((m.codec_v2.smb.device for m in mods if hasattr(m, "codec_v2")), None)
        if dev0 is None:
            raise ValueError("no TBE2 member in the group")
        devs, own = [], []
        for m in mods:
            if hasattr(m, "codec_v2"):
                devs.append(m.codec_v2)
                own.append(True)
            else:
                devs.append(_encode_dense(m, dev0))
                own.append(False)
        self.encoded_dense_members = own.count(False)
        K = int(devs[0].K)
        if any(int(d.K) != K for d in devs):
            raise ValueError("fused TBE2 tensors must share K")
        self.K, self.N = K, sum(int(d.N) for d in devs)
        self.members = [(getattr(m, "name", ""), int(d.N)) for m, d in zip(mods, devs)]
        if len(devs) == 1:
            d = devs[0]
            self.planes, self.smb, self.ovf, self.len, self.grp = d.planes, d.smb, d.ovf, d.len, d.grp
            self.cb = d.cb16
            self.rowp = arrays_of([d], [0])[6]
        else:
            self.planes = torch.cat([d.planes for d in devs])
            self.smb = torch.cat([d.smb for d in devs])
            words = [d.ovf_words for d in devs]
            self.ovf = torch.cat([d.ovf[:w] for d, w in zip(devs, words)]
                                 + [torch.zeros(16, dtype=torch.int32, device=dev0)])
            nl = [int(d.len.numel()) - 16 for d in devs]
            self.len = torch.cat([d.len[:n] for d, n in zip(devs, nl)]
                                 + [torch.zeros(16, dtype=torch.uint8, device=dev0)])
            grp, base = [], 0
            for d, w in zip(devs, words):
                grp.append(_u32_to_i32(_i32_to_u32(d.grp) + base))
                base += w
            if base + 16 >= _U32:
                raise ValueError("fused overflow stream exceeds 2^32 words")
            self.grp = torch.cat(grp)
            self.cb = torch.cat([d.cb16 for d in devs])
            rp = []
            for i, d in enumerate(devs):
                us = {2: 1, 4: 2, 8: 3}[int(d.len_unit)]
                rp.append(torch.full((int(d.N),), i | (us << 8), dtype=torch.int32, device=dev0))
            self.rowp = torch.cat(rp)
            if repoint:                     # one copy: members become views of the fused buffers
                p_off = s_off = o_off = l_off = 0
                for d, w, n, o in zip(devs, words, nl, own):
                    npl, nsm = int(d.planes.numel()), int(d.smb.numel())
                    if o:
                        d.planes = self.planes[p_off:p_off + npl]
                        d.smb = self.smb[s_off:s_off + nsm]
                        d.ovf = self.ovf[o_off:o_off + w + 16]   # tail = next member / zero pad:
                        d.len = self.len[l_off:l_off + n + 16]   # addressable, never consumed
                    p_off += npl; s_off += nsm; o_off += w; l_off += n
        self.arrays = [self.planes, self.smb, self.ovf, self.len, self.grp, self.cb, self.rowp]
        # The direct-call path, built now (not lazily) so nothing allocates inside a CUDA-graph
        # capture; BatchDecoder wraps the descriptor itself (bigemm.wrap) and never uses this.
        self._bi = None
        if bi:
            from . import bigemm as bg

            self._bi = bg.wrap(self, bg.Workspace(dev0))

    def reserve(self, max_m: int) -> None:
        self._bi.reserve(max_m)

    @property
    def bytes(self) -> int:
        return int(sum(int(a.numel()) * int(a.element_size()) for a in self.arrays))

    @property
    def coded_bytes(self) -> int:
        """Bytes the kernel streams per call: planes + sign/mantissa + overflow."""
        return int(self.planes.numel() * 4 + self.smb.numel() + (self.ovf.numel() - 16) * 4)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self._bi(x, y)


__all__ = ["T2_MAXCB", "TBE2Lin", "TBE2Linear", "arrays_of"]


# ----------------------------------------------------------------------------- TBE21 (codec v2.1)
class TBE21Linear(nn.Module):
    """A linear layer whose weight lives on the device as codec-v2.1 streams only
    (``codec_v2_artifact.V21Device``; its own checkpoints are not used -- a launch builds them
    for its split count, see ``TBE21Lin``)."""

    def __init__(self, dev, bias: Optional[torch.Tensor] = None, name: str = ""):
        super().__init__()
        self.codec_v21 = dev
        self.name = name or dev.name
        self.out_features, self.in_features = int(dev.N), int(dev.K)
        if bias is not None:
            self.bias = nn.Parameter(bias.detach(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def dense_weight(self) -> torch.Tensor:
        from . import bigemm as bg

        d = TBE21Lin([self], bi=False, S=1)
        return bg.decode_weight("tbe21", d.N, d.K, list(d.arrays), S=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # non-BI fallback paths only
        return F.linear(x, self.dense_weight(), self.bias)


def tbe21_checkpoints(arrays: Sequence[torch.Tensor], N: int, K: int, S: int) -> torch.Tensor:
    """RESIDENT split checkpoints int32 [N*(S-1)] for a TBE21 stream set: on CUDA the kernel's own
    decode walk (``bi_t21_ckpt_kernel``); on the CPU the proved numpy transcription."""
    dev = arrays[0].device
    if S <= 1:
        return torch.zeros(0, dtype=torch.int32, device=dev)
    if dev.type == "cuda":
        from . import bigemm as bg

        out = torch.empty(N * (S - 1), dtype=torch.int32, device=dev)
        bg.extension().bi_t21_ckpt_out(int(N), int(K), list(arrays[:6]), int(S), out)
        return out
    from . import bigemm_tbe21_ref as R21

    l1, smb, ovf, rowoff, cb, rowp = (a.cpu().numpy() for a in arrays[:6])
    st = R21.T21Streams(N, K, l1.view(np.uint16), smb, ovf.view(np.uint32), rowoff.view(np.uint32),
                        cb, rowp, np.zeros((N, 0), np.uint32), 1)
    ck = R21.checkpoints_by_walk(st, S)
    return torch.from_numpy(np.ascontiguousarray(ck).reshape(-1).view(np.int32))


class TBE21Lin:
    """FastDecoder / BI-GEMM descriptor over one TBE21 module or a fused group: members' streams
    concatenated along N, row offsets re-based by the overflow words before them, codebooks
    stacked (<= 8; rowp = member codebook base + the row's codebook), and the RESIDENT checkpoints
    for the launch split count S = split_for(N_fused, K) built once, here."""
    kind = "tbe21"

    def __init__(self, mods: Sequence[nn.Module], *, bi: bool = True, S: Optional[int] = None,
                 sms: Optional[int] = None, repoint: bool = True):
        devs = [m.codec_v21 for m in mods]
        if sum(int(d.ncb) for d in devs) > T2_MAXCB:
            raise ValueError(f"TBE21 fused group needs {sum(int(d.ncb) for d in devs)} codebooks > {T2_MAXCB}")
        K = int(devs[0].K)
        if any(int(d.K) != K for d in devs):
            raise ValueError("fused TBE21 tensors must share K")
        dev0 = devs[0].smb.device
        self.K, self.N = K, sum(int(d.N) for d in devs)
        self.members = [(getattr(m, "name", ""), int(d.N)) for m, d in zip(mods, devs)]
        if len(devs) == 1:
            # Reuse stored streams: copying the LM head here raises the load peak
            # by an entire compressed tensor, even for a row-bounded G1 decode.
            d = devs[0]
            self.l1, self.smb, self.ovf = d.l1, d.smb, d.ovf
            self.rowoff, self.cb, self.rowp = d.rowoff, d.cb16, d.rowcb
            if S is None:
                from . import bigemm as bg

                S = bg.split_for(self.N, self.K, sms)
            self.S = int(S)
            base_arrays = [self.l1, self.smb, self.ovf, self.rowoff, self.cb, self.rowp]
            self.ckpt = tbe21_checkpoints(base_arrays, self.N, self.K, self.S)
            self.arrays = base_arrays + [self.ckpt]
            self._bi = None
            if bi:
                from . import bigemm as bg

                self._bi = bg.wrap(self, bg.Workspace(dev0))
            return
        words = [d.ovf_words for d in devs]
        self.l1 = torch.cat([d.l1 for d in devs])
        self.smb = torch.cat([d.smb for d in devs])
        self.ovf = torch.cat([d.ovf[:w] for d, w in zip(devs, words)]
                             + [torch.zeros(16, dtype=torch.int32, device=dev0)])
        ro, base = [], 0
        for d, w in zip(devs, words):
            ro.append(_u32_to_i32(_i32_to_u32(d.rowoff) + base))
            base += w
        if base + 16 >= _U32:
            raise ValueError("fused overflow stream exceeds 2^32 words")
        self.rowoff = torch.cat(ro)
        self.cb = torch.cat([d.cb16 for d in devs])
        rp, cbase = [], 0
        for d in devs:
            rp.append(d.rowcb.to(torch.int32) + cbase)
            cbase += int(d.ncb)
        self.rowp = torch.cat(rp)
        if S is None:
            from . import bigemm as bg

            S = bg.split_for(self.N, self.K, sms)
        self.S = int(S)
        base_arrays = [self.l1, self.smb, self.ovf, self.rowoff, self.cb, self.rowp]
        self.ckpt = tbe21_checkpoints(base_arrays, self.N, self.K, self.S)
        self.arrays = base_arrays + [self.ckpt]
        if repoint:
            # Keep each source module usable while releasing its original large
            # allocation. Its local row offsets still address its own ovf slice;
            # the following member supplies the addressable masked-load tail.
            l_off = s_off = o_off = 0
            for d, w in zip(devs, words):
                nl, ns = int(d.l1.numel()), int(d.smb.numel())
                d.l1 = self.l1[l_off:l_off + nl]
                d.smb = self.smb[s_off:s_off + ns]
                d.ovf = self.ovf[o_off:o_off + w + 16]
                l_off += nl
                s_off += ns
                o_off += w
        self._bi = None
        if bi:
            from . import bigemm as bg

            self._bi = bg.wrap(self, bg.Workspace(dev0))

    def reserve(self, max_m: int) -> None:
        self._bi.reserve(max_m)

    @property
    def bytes(self) -> int:
        return int(sum(int(a.numel()) * int(a.element_size()) for a in self.arrays))

    @property
    def coded_bytes(self) -> int:
        return int(self.l1.numel() * 2 + self.smb.numel() + (self.ovf.numel() - 16) * 4)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self._bi(x, y)


__all__ += ["TBE21Lin", "TBE21Linear", "tbe21_checkpoints"]
