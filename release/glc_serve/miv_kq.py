"""MIV-KQ: llama.cpp K-quant / i-quant weights dequantized in registers inside MIV-GEMV.

Types: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS (Q8_0 lives in the engine branch's
``miv_gemv.miv_q8_out``; this module follows its descriptor pattern).

Each weight is ``bf16_rn(v)`` where ``v`` is the FP32 value gguf-py's reference
``gguf.quants.dequantize`` computes -- reproduced operation by operation -- and the
FMA chain / butterfly / slice sum / rounding are the verbatim bf16 MIV code.  With
the SAME WPR as bf16 MIV for (N, K), the output is bitwise bf16 MIV run on
``torch.from_numpy(dequantize(...)).to(torch.bfloat16)``.  Contract and array
layout: ``miv_kq_csrc/miv_kq.h``.

Engine hook (same as MIV-TBE / Q8):

    d = kq_desc_from_blocks("Q4_K", raw_u8[N, row_bytes], device)    # SoA split once
    y = linear(x, d, out=None)          # 1 <= M <= 8
    w32 = dequant_f32(d)                # decode-only twin (G1-q)

``register_with_engine()`` registers every type with
``glc_serve.q8serve.register_gguf_type`` (engine branch): builder(t, rest, cfg,
device, tune) -> ``KQLinear`` (HF forward + ``fast_desc()``).
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_M = 8
QK_K = 256
UNROLL_CHOICES = (1, 2, 4)

# name -> (ggml id, bytes per 256-block, [(field, start, end)])
TYPES: Dict[str, Tuple[int, int, List[Tuple[str, int, int]]]] = {
    "Q4_K": (12, 144, [("dm", 0, 4), ("scales", 4, 16), ("qs", 16, 144)]),
    "Q5_K": (13, 176, [("dm", 0, 4), ("scales", 4, 16), ("qh", 16, 48), ("qs", 48, 176)]),
    "Q6_K": (14, 210, [("ql", 0, 128), ("qh", 128, 192), ("scales", 192, 208), ("d", 208, 210)]),
    "IQ4_XS": (23, 136, [("d", 0, 2), ("scales_h", 2, 4), ("scales_l", 4, 8), ("qs", 8, 136)]),
    "IQ3_XXS": (18, 98, [("d", 0, 2), ("qs", 2, 66), ("scales", 66, 98)]),
    "IQ3_S": (21, 110, [("d", 0, 2), ("qs", 2, 66), ("qh", 66, 74), ("signs", 74, 106),
                        ("scales", 106, 110)]),
    "Q2_K": (10, 84, [("scales", 0, 16), ("qs", 16, 80), ("dm", 80, 84)]),
    "Q3_K": (11, 110, [("hmask", 0, 32), ("qs", 32, 96), ("scales", 96, 108), ("d", 108, 110)]),
    "IQ2_XS": (17, 74, [("d", 0, 2), ("qs", 2, 66), ("scales", 66, 74)]),
    "IQ2_S": (22, 82, [("d", 0, 2), ("qs", 2, 34), ("signs", 34, 66), ("qh", 66, 74),
                       ("scales", 74, 82)]),
}
GRID_TYPES = ("IQ3_XXS", "IQ3_S", "IQ2_XS", "IQ2_S")
KSIGN_TYPES = ("IQ3_XXS", "IQ2_XS")
ID2NAME = {v[0]: k for k, v in TYPES.items()}

# (N, K, M, type) -> unroll or (unroll, rows_per_warp); speed only (all bit-identical)
KQ_UNROLL: Dict[Tuple[int, int, int, str], Any] = {}

_CSRC = Path(__file__).resolve().parent / "miv_kq_csrc"
_SOURCES = ["miv_kq_api.cu"] + [f"miv_kq_m{m}.cu" for m in range(1, 9)] + ["miv_kq_torch.cpp"]
_EXT = None
_EXT_LOCK = threading.Lock()


def _build_dir() -> Path:
    d = os.environ.get("MIV_KQ_BUILD_DIR")
    p = Path(d) if d else Path(__file__).resolve().parents[2] / ".scratch" / "miv_kq_build"
    if str(p.resolve()).startswith(("/tmp", "/private/tmp")):
        raise RuntimeError(f"refusing a volatile build dir {p}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def extension():
    global _EXT
    if _EXT is not None:
        return _EXT
    with _EXT_LOCK:
        if _EXT is None:
            from torch.utils.cpp_extension import load

            major, minor = torch.cuda.get_device_capability()
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
            os.environ.setdefault("MAX_JOBS", "16")
            _EXT = load(name=f"miv_kq_sm{major}{minor}",
                        sources=[str(_CSRC / s) for s in _SOURCES],
                        extra_include_paths=[str(_CSRC)], extra_cflags=["-O3"],
                        extra_cuda_cflags=["-O3", "--fmad=true", "-lineinfo"],
                        build_directory=str(_build_dir()),
                        verbose=bool(int(os.environ.get("MIV_KQ_VERBOSE", "0"))))
    return _EXT


# ------------------------------------------------------------------ IQ3 tables
_TABLES: Dict[Tuple[str, str], Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {}


def _tables(name: str, device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Grid (one byte per value, 4 values per uint32; 8-value entries take two) and
    ksigns, both taken from gguf-py itself."""
    if name not in GRID_TYPES:
        return None, None
    key = (name, str(device))
    if key not in _TABLES:
        from gguf import quants as q

        cls = getattr(q, name)
        cls.init_grid()
        g = np.asarray(cls.grid, dtype=np.float32).reshape(cls.grid_shape)
        gb = g.astype(np.int64)
        if not np.array_equal(gb.astype(np.float32), g) or gb.min() < 0 or gb.max() > 255:
            raise ValueError(f"{name}: grid values are not bytes")
        packed = np.ascontiguousarray(gb.astype(np.uint8)).view(np.uint32).reshape(-1)
        grid = torch.from_numpy(packed.view(np.int32).copy()).to(device)
        ks = None
        if name in KSIGN_TYPES:
            ks = torch.from_numpy(np.frombuffer(q.IQ2_XXS.ksigns, dtype=np.uint8).copy()).to(device)
        _TABLES[key] = (grid, ks)
    return _TABLES[key]


# ------------------------------------------------------------------ descriptor
@dataclass
class KQDesc:
    """Descriptor over one K/I-quant weight [N, K] (rows in GGUF order)."""

    type: str
    N: int
    K: int
    wpr: int
    unroll: int
    arrays: List[torch.Tensor]              # SoA planes, see miv_kq.h
    grid: Optional[torch.Tensor] = None
    ksigns: Optional[torch.Tensor] = None
    bias: Optional[torch.Tensor] = None
    parts: List[str] = field(default_factory=list)
    kind: str = "kq"
    rpw: int = 1                            # rows per warp (1 or 2); identical bits

    @property
    def n(self) -> int:
        return self.N

    @property
    def k(self) -> int:
        return self.K

    @property
    def type_id(self) -> int:
        return TYPES[self.type][0]

    @property
    def bytes(self) -> int:
        """Weight bytes one call reads (the GGUF block payload, byte for byte)."""
        return sum(int(a.numel()) * a.element_size() for a in self.arrays)

    coded_bytes = bytes

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """FastDecoder descriptor protocol: y[:M, :N] = x @ W^T, y's row stride honoured."""
        m = int(x.shape[0])
        cfg = KQ_UNROLL.get((self.N, self.K, m, self.type), (self.unroll, self.rpw))
        u, r = (cfg, self.rpw) if isinstance(cfg, int) else cfg
        extension().miv_kq_out(x, self.type_id, self.N, self.K, self.wpr, int(u), int(r),
                               self.arrays, self.grid, self.ksigns, self.bias, y)

    def with_(self, **kw) -> "KQDesc":
        d = KQDesc(**{**self.__dict__})
        for key, val in kw.items():
            setattr(d, key, val)
        return d

    def clone(self) -> "KQDesc":
        return self.with_(arrays=[a.clone() for a in self.arrays])


def _dense_config(n: int, k: int) -> Tuple[int, int]:
    from . import miv_gemv as mg

    wpr, unroll = mg.config_for(n, k)
    return int(wpr), int(unroll)


def split_blocks(type_name: str, raw: np.ndarray) -> List[np.ndarray]:
    """raw uint8 [N, K/256*type_size] (GGUF row bytes) -> SoA planes (copies)."""
    _, ts, fields = TYPES[type_name]
    n = int(raw.shape[0])
    blocks = np.ascontiguousarray(raw).reshape(n, -1, ts)
    return [np.ascontiguousarray(blocks[:, :, a:b]).reshape(-1) for _, a, b in fields]


def kq_desc_from_blocks(type_name: str, raw: np.ndarray, device, *, k: Optional[int] = None,
                        wpr: Optional[int] = None, unroll: int = 1,
                        bias: Optional[torch.Tensor] = None,
                        row_perm: Optional[np.ndarray] = None,
                        name: str = "") -> KQDesc:
    """Build a descriptor from GGUF block bytes.  ``row_perm`` (optional) relabels
    rows (new row i = GGUF row row_perm[i]) -- block-safe for every type."""
    if type_name not in TYPES:
        raise ValueError(f"unsupported type {type_name}")
    _, ts, _ = TYPES[type_name]
    raw = np.asarray(raw, dtype=np.uint8)
    n = int(raw.shape[0])
    nb_row = raw.reshape(n, -1).shape[1] // ts
    kk = nb_row * QK_K
    if k is not None and int(k) != kk:
        raise ValueError(f"K mismatch {k} vs {kk}")
    if row_perm is not None:
        raw = raw.reshape(n, -1)[np.asarray(row_perm)]
    planes = split_blocks(type_name, raw)
    dev = torch.device(device)
    arrays = [torch.from_numpy(p).to(dev) for p in planes]
    dw, _ = _dense_config(n, kk)
    grid, ks = _tables(type_name, dev)
    if bias is not None:
        bias = bias.to(device=dev, dtype=torch.bfloat16).contiguous()
    return KQDesc(type=type_name, N=n, K=kk, wpr=int(wpr or dw), unroll=int(unroll),
                  arrays=arrays, grid=grid, ksigns=ks, bias=bias, parts=[name] if name else [])


def fuse_kq(descs: Sequence[KQDesc]) -> KQDesc:
    """Row-concatenate same-type descriptors (SoA planes are per-block, row-major)."""
    t = descs[0].type
    if any(d.type != t for d in descs) or any(d.K != descs[0].K for d in descs):
        raise ValueError("fuse_kq: members must share type and K (use GroupLin otherwise)")
    arrays = [torch.cat([d.arrays[i] for d in descs]) for i in range(len(descs[0].arrays))]
    n = sum(d.N for d in descs)
    wpr, _ = _dense_config(n, descs[0].K)
    bias = None
    if any(d.bias is not None for d in descs):
        bias = torch.cat([d.bias if d.bias is not None else
                          torch.zeros(d.N, dtype=torch.bfloat16, device=arrays[0].device)
                          for d in descs])
    return KQDesc(type=t, N=n, K=descs[0].K, wpr=wpr, unroll=descs[0].unroll, arrays=arrays,
                  grid=descs[0].grid, ksigns=descs[0].ksigns, bias=bias,
                  parts=[p for d in descs for p in d.parts])


def row_slice(desc: KQDesc, r0: int, r1: int) -> KQDesc:
    """Descriptor over rows [r0, r1) -- views of the SoA planes, no copy."""
    nb_row = desc.K // QK_K
    arrays = []
    for arr, (_, a0, a1) in zip(desc.arrays, TYPES[desc.type][2]):
        per_row = nb_row * (a1 - a0) // arr.element_size()
        arrays.append(arr[r0 * per_row:r1 * per_row])
    bias = desc.bias[r0:r1] if desc.bias is not None else None
    return desc.with_(N=r1 - r0, arrays=arrays, bias=bias)


def linear(x: torch.Tensor, desc: KQDesc, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if x2.dtype != torch.bfloat16:
        x2 = x2.to(torch.bfloat16)
    if x2.stride(-1) != 1 or x2.stride(0) % 8 or x2.data_ptr() % 16:
        x2 = x2.contiguous()
        if x2.data_ptr() % 16:
            x2 = x2.clone()
    m = int(x2.shape[0])
    if not 1 <= m <= MAX_M:
        raise ValueError(f"miv_kq.linear: 1 <= M <= 8, got {m}")
    y = out if out is not None else torch.empty(m, desc.N, dtype=torch.bfloat16, device=x2.device)
    desc(x2, y)
    return out if out is not None else y.reshape(*lead, desc.N)


def dequant_f32(desc: KQDesc, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    w = out if out is not None else torch.empty(desc.N, desc.K, dtype=torch.float32,
                                                device=desc.arrays[0].device)
    extension().miv_kq_dequant(desc.type_id, desc.N, desc.K, desc.arrays, desc.grid, desc.ksigns, w)
    return w


def dequant_bf16(desc: KQDesc) -> torch.Tensor:
    return dequant_f32(desc).to(torch.bfloat16)


# ------------------------------------------------------------------ engine plug-in
class KQLinear(nn.Module):
    """HF-side linear over a KQDesc: M <= 8 fused; larger M dequantizes then F.linear.
    ``in_perm`` (ssm_out under K-quants): input gathered into GGUF column order."""

    def __init__(self, desc: KQDesc, in_perm: Optional[torch.Tensor] = None):
        super().__init__()
        self.desc = desc
        self.in_perm = in_perm
        self.in_features, self.out_features = desc.K, desc.N
        self.bias = None

    @property
    def weight(self):                 # shape/dtype/device probes only; never materialized
        return torch.empty(0, dtype=torch.bfloat16, device=self.desc.arrays[0].device)

    @property
    def bytes(self) -> int:
        return self.desc.bytes

    def fast_desc(self) -> KQDesc:
        return self.desc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_perm is not None:
            x = x.index_select(-1, self.in_perm)
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1]).to(torch.bfloat16)
        m = int(x2.shape[0])
        if 1 <= m <= MAX_M:
            y = linear(x2.contiguous(), self.desc)
        else:                           # prefill: the same fused kernel, 8 rows at a time
            x2 = x2.contiguous()        # (no dequant scratch; M-invariant -> same bits as decode)
            y = torch.empty(m, self.desc.N, dtype=torch.bfloat16, device=x2.device)
            for c0 in range(0, m, MAX_M):
                self.desc(x2[c0:c0 + MAX_M], y[c0:c0 + MAX_M])
        return y.reshape(*lead, self.desc.N).to(x.dtype)


def _row_perm(rest: str, cfg, n: int) -> Optional[np.ndarray]:
    from .q8serve import _fixer

    fix = _fixer(rest, cfg)
    if fix is None or rest == "linear_attn.out_proj":
        return None
    idx = torch.arange(n, dtype=torch.int64).reshape(n, 1)
    p, _ = fix(idx, idx.clone())
    return p.reshape(-1).numpy()


def gguf_builder(t, rest: str, cfg, device, tune) -> KQLinear:
    """``q8serve.register_gguf_type`` builder for every KQ type."""
    name = t.tensor_type.name
    ne = [int(v) for v in t.shape]
    k, n = ne[0], int(np.prod(ne[1:]))
    raw = np.asarray(t.data).reshape(n, -1)
    wpr = None
    if tune is not None and (n, k) in tune:
        wpr = int(tune[(n, k)][0])
    desc = kq_desc_from_blocks(name, raw, device, k=k, wpr=wpr,
                               row_perm=_row_perm(rest, cfg, n), name=t.name)
    in_perm = None
    if rest == "linear_attn.out_proj":
        from .q8serve import _col_perm

        in_perm = _col_perm(cfg, k, device)
    return KQLinear(desc, in_perm)


def load_table(path) -> int:
    """Load a ``kq_autotune`` table: "NxKxMxTYPE" -> [unroll, rows_per_warp]."""
    import json

    tab = json.loads(Path(path).read_text())["table"]
    for key, v in tab.items():
        n, k, m, t = key.split("x", 3)
        KQ_UNROLL[(int(n), int(k), int(m), t)] = (int(v[0]), int(v[1])) if isinstance(v, list) else int(v)
    return len(tab)


def register_with_engine() -> List[str]:
    from .q8serve import register_gguf_type

    for name in TYPES:
        register_gguf_type(name, gguf_builder)
    return list(TYPES)


__all__ = ["KQDesc", "KQLinear", "KQ_UNROLL", "TYPES", "dequant_bf16", "dequant_f32", "extension",
           "fuse_kq", "gguf_builder", "kq_desc_from_blocks", "linear", "load_table", "row_slice", "register_with_engine",
           "split_blocks"]
