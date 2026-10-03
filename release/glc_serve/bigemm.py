"""BI-GEMM descriptors: batch-invariant tensor-core GEMM over every engine weight format.

``bi_csrc/bi_gemm.cuh`` holds the contract.  In one sentence: row m of ``y = x W^T`` is a
function of row m of x and of W only -- never of M, of the row's slot in the batch, or of
the other rows -- because the K reduction is one fixed sequence (128-wide chunks, a split
count S that depends on (N, K) alone, mma.m16n8k16 in increasing k, split partials added
in split order, one bf16 rounding).  Weights are decoded in the kernel by the MIV decode
functions, so TBE-coded weights give the bits of the bf16 parent and K-quant weights the
bits of gguf-py's dequantize, through the same GEMM.

``wrap(desc)`` turns a FastDecoder descriptor (``fastdec.Lin``, ``tbe_desc.TBELin``,
``q8serve.Q8Lin``, ``miv_kq.KQDesc``, ``q8serve.GroupLin``, ``q8serve.PermIn``) into a
``BILin`` with ``__call__(x, y)`` for ANY number of rows.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import List, Optional

import torch

_CSRC = Path(__file__).resolve().parent / "bi_csrc"
_EXT = None
_LOCK = threading.Lock()
BN, CH = 64, 128
FMT = {"bf16": 0, "q8": 1, "tbe": 2, "kq": 3}


def _build_dir() -> Path:
    d = os.environ.get("BI_GEMM_BUILD_DIR")
    p = Path(d) if d else Path(__file__).resolve().parents[2] / ".scratch" / "bi_gemm_build"
    if str(p.resolve()).startswith(("/tmp", "/private/tmp")):
        raise RuntimeError(f"refusing a volatile build dir {p}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def extension():
    global _EXT
    if _EXT is not None:
        return _EXT
    with _LOCK:
        if _EXT is None:
            from torch.utils.cpp_extension import load

            arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
            if not arch:
                major, minor = torch.cuda.get_device_capability()
                arch = f"{major}.{minor}"
                os.environ["TORCH_CUDA_ARCH_LIST"] = arch
            major, minor = arch.split(";")[0].replace("+PTX", "").split(".")
            os.environ.setdefault("MAX_JOBS", "8")
            here = Path(__file__).resolve().parent
            _EXT = load(name=f"bi_gemm_sm{major}{minor}",
                        sources=[str(_CSRC / s) for s in ("bi_gemm_dense.cu", "bi_gemm_kq.cu", "bi_torch.cpp")],
                        extra_include_paths=[str(_CSRC), str(here / "miv_kq_csrc"), str(here / "miv_tbe_csrc")],
                        extra_cflags=["-O3"],
                        extra_cuda_cflags=["-O3", "--fmad=true", "-lineinfo", "-std=c++17"],
                        build_directory=str(_build_dir()),
                        verbose=bool(int(os.environ.get("BI_GEMM_VERBOSE", "0"))))
    return _EXT


def split_for(N: int, K: int, sms: Optional[int] = None) -> int:
    """Split-K count: a function of (N, K) and the card ONLY -- never of M (batch invariance)."""
    if sms is None:
        sms = torch.cuda.get_device_properties(0).multi_processor_count
    ctas = (N + BN - 1) // BN
    s = max(1, min(8, (2 * sms) // max(ctas, 1)))
    nch = K // CH
    while s > 1 and nch // s < 4:
        s -= 1
    return s


class Workspace:
    """One fp32 split-K workspace shared by every BILin of an engine (calls are serialised on
    one stream)."""

    def __init__(self, device):
        self.device = device
        self.buf = torch.empty(0, dtype=torch.float32, device=device)

    def get(self, n: int) -> torch.Tensor:
        if self.buf.numel() < n:
            self.buf = torch.empty(n, dtype=torch.float32, device=self.device)
        return self.buf

    def reserve(self, n: int) -> None:
        self.get(n)


_EMPTY = {}


def _empty(dev):
    if dev not in _EMPTY:
        _EMPTY[dev] = torch.empty(0, dtype=torch.uint8, device=dev)
    return _EMPTY[dev]


class BILin:
    """One weight of one format through BI-GEMM."""

    def __init__(self, fmt: str, N: int, K: int, arrs: List[torch.Tensor], ws: Workspace, *,
                 kqt: int = 0, bias: Optional[torch.Tensor] = None, src=None, S: Optional[int] = None):
        self.fmt, self.N, self.K, self.arrs, self.ws, self.kqt, self.bias = fmt, int(N), int(K), arrs, ws, kqt, bias
        self.S = int(S) if S else split_for(self.N, self.K)
        self.src = src
        self.kind = f"bi-{fmt}"

    @property
    def bytes(self) -> int:
        return int(getattr(self.src, "bytes", 0))

    @property
    def coded_bytes(self) -> int:
        return int(getattr(self.src, "coded_bytes", self.bytes))

    def reserve(self, max_m: int) -> None:
        if self.S > 1:
            self.ws.reserve(self.S * max_m * self.N)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        m = int(x.shape[0])
        ws = self.ws.get(self.S * m * self.N) if self.S > 1 else self.ws.buf
        extension().bi_gemm_out(x, m, FMT[self.fmt], self.kqt, self.N, self.K, self.arrs, self.bias, y,
                                self.S, ws)


class BIGroup:
    kind = "bi-group"

    def __init__(self, members):
        self.members = list(members)            # [(BILin|BIPerm, n)]
        self.N = sum(n for _, n in self.members)
        self.K = self.members[0][0].K

    @property
    def bytes(self):
        return sum(d.bytes for d, _ in self.members)

    @property
    def coded_bytes(self):
        return sum(d.coded_bytes for d, _ in self.members)

    def reserve(self, max_m):
        for d, _ in self.members:
            d.reserve(max_m)

    def __call__(self, x, y):
        r = 0
        for d, n in self.members:
            d(x, y[:, r:r + n])
            r += n


class BIPerm:
    kind = "bi-perm"

    def __init__(self, d, perm: torch.Tensor):
        self.d, self.perm = d, perm
        self.N, self.K = d.N, d.K
        self.buf = torch.empty(0, self.K, dtype=torch.bfloat16, device=perm.device)

    @property
    def bytes(self):
        return self.d.bytes

    @property
    def coded_bytes(self):
        return self.d.coded_bytes

    def reserve(self, max_m):
        if self.buf.shape[0] < max_m:
            self.buf = torch.empty(max_m, self.K, dtype=torch.bfloat16, device=self.perm.device)
        self.d.reserve(max_m)

    def __call__(self, x, y):
        m = int(x.shape[0])
        torch.index_select(x, 1, self.perm, out=self.buf[:m])     # per-row gather: M-invariant
        self.d(self.buf[:m], y)


def tbe_escidx(t) -> torch.Tensor:
    """[N][K/128] index in ``esc`` of each 128-column chunk's first escape, from the per-tile
    escape counts, anchored at TBELin's per-row slice-0 base; cross-checked against every
    other slice base TBELin computed (a load-time consistency gate)."""
    from . import miv_gemv as mg

    N, K, wpr = t.N, t.K, t.wpr
    tpr = K // 64
    cnt = mg.extension().tbe_tile_escapes(t.planes).to(torch.int64).reshape(N, tpr)
    excl = torch.cumsum(cnt, 1) - cnt                                  # within-row exclusive prefix
    base = t.escbase.to(torch.int64).reshape(N, wpr)
    for s in range(1, wpr):
        if not torch.equal(base[:, 0] + excl[:, s * (tpr // wpr)], base[:, s]):
            raise ValueError("TBE escape prefix disagrees with TBELin.escbase")
    idx = base[:, :1] + excl[:, ::2]
    if int(idx.max()) >= 2 ** 31:
        raise ValueError("escape index exceeds int32")
    return idx.to(torch.int32).contiguous()


def wrap(desc, ws: Workspace):
    """FastDecoder descriptor -> BI descriptor (same weight bytes; views, no copies)."""
    kind = getattr(desc, "kind", None)
    if kind == "group":
        return BIGroup([(wrap(d, ws), n) for d, n in desc.members])
    if kind == "perm":
        return BIPerm(wrap(desc.d, ws), desc.perm)
    if kind == "bf16":
        return BILin("bf16", desc.N, desc.K, [desc.w], ws, src=desc)
    if kind == "q8_0":
        pk = desc.p
        return BILin("q8", pk.N, pk.K, [pk.qs, pk.sc], ws, src=desc)
    if kind == "tbe":
        return BILin("tbe", desc.N, desc.K, [desc.planes, desc.smb, desc.esc, desc.rowparam, tbe_escidx(desc)],
                     ws, src=desc)
    if kind == "kq":
        dev = desc.arrays[0].device
        arrs = list(desc.arrays) + [_empty(dev)] * (5 - len(desc.arrays))
        arrs += [desc.grid if desc.grid is not None else _empty(dev),
                 desc.ksigns if desc.ksigns is not None else _empty(dev)]
        return BILin("kq", desc.N, desc.K, arrs, ws, kqt=desc.type_id, bias=desc.bias, src=desc)
    raise TypeError(f"no BI-GEMM path for descriptor {type(desc).__name__} kind={kind}")


__all__ = ["BILin", "BIGroup", "BIPerm", "Workspace", "extension", "split_for", "tbe_escidx", "wrap"]
