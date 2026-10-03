"""MIV-TBE: bf16 MIV-GEMV fused with in-register GLC-TBE decode (M = 1..8).

Weights are read CODED (planes + smb + esc, ~11.2 bits/weight) and each bf16
weight is rebuilt in registers; nothing dense is ever materialised.  The FMA
chain, butterfly, slice sum and rounding are the verbatim bf16 MIV code
(``miv_gemv.py`` @ c155643be), so with the SAME WPR per shape the output is
bitwise the bf16-MIV output on the parent weights (G3a by construction, G1
holds because the container is exact).  Contract and descriptor layout:
``miv_tbe_csrc/miv_tbe.h``.

Engine-facing API (one hook, per-tensor descriptors):

    desc = build_tbe_desc([tbe_tensor_or_bf16, ...], device)   # row concat
    desc = bf16_desc(weight)                                    # dense twin
    y = linear(x, desc, out=None)          # x [..., K], 1 <= M <= 8

``desc.wpr`` is taken from ``glc_serve.miv_gemv.config_for(N, K)`` -- the
SAME table the bf16 MIV path uses -- so dense and coded weights of one shape
always share the reduction split.  Default-off: nothing imports this module
unless a caller does.
"""
from __future__ import annotations

import json
import os
import threading

from glc_serve import _winenv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

MAX_M = 8
ESC_PAD = 16
TILE = 64
UNROLL_CHOICES = (1, 2, 4)
DEFAULT_VERSION = 2

# (N, K) -> TBE unroll (load batching only; never changes arithmetic).  Tuned
# on RTX PRO 6000 Blackwell Server at fixed WPR (scripts/miv_tbe_bench.py).
TBE_UNROLL: Dict[Tuple[int, int], int] = {}

# (N, K, M) -> (version, unroll): launch-shape autotune.  Every version and
# unroll produces the SAME bits (same lane order, same decode, same epilogue;
# tested), so choosing them per M changes speed only -- never the output.
TBE_TABLE: Dict[Tuple[int, int, int], Tuple[int, int]] = {}


_CONFIGS = Path(__file__).resolve().parent / "miv_tbe_configs"


def load_default_configs(device_tag: str = "rtxpro6000") -> Tuple[int, int]:
    """Load the gated RTX PRO 6000 tables: bf16-MIV WPR (shared by dense and
    coded descriptors -- REQUIRED for bitwise equality) and the TBE
    (version, unroll) autotune.  Call BEFORE building descriptors."""
    from . import miv_gemv as mg

    n1 = mg.load_config(_CONFIGS / f"miv_wpr_{device_tag}.json")
    n2 = load_table(_CONFIGS / f"tbe_autotune_{device_tag}.json")
    return n1, n2


def load_table(path) -> int:
    """Load ``tbe_autotune.json`` (``scripts/miv_tbe_bench.py --phases autotune``)."""
    tab = json.loads(Path(path).read_text())["table"]
    for key, (ver, u) in tab.items():
        n, k, m = (int(v) for v in key.split("x"))
        TBE_TABLE[(n, k, m)] = (int(ver), int(u))
    return len(tab)

_CSRC = Path(__file__).resolve().parent / "miv_tbe_csrc"
_SOURCES = (["miv_tbe_api.cu"] + [f"miv_tbe_m{m}.cu" for m in range(1, 9)] +
            ["miv_tbe_torch.cpp"])

_EXT = None
_EXT_LOCK = threading.Lock()


def _build_dir() -> Path:
    d = os.environ.get("MIV_TBE_BUILD_DIR")
    p = Path(d) if d else Path(__file__).resolve().parents[2] / ".scratch" / "miv_tbe_build"
    if str(p.resolve()).startswith(("/tmp", "/private/tmp")):
        raise RuntimeError(f"refusing a volatile build dir {p}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def extension():
    """JIT-build (once, in parallel TUs) and return the CUDA extension."""
    global _EXT
    if _EXT is not None:
        return _EXT
    with _EXT_LOCK:
        if _EXT is None:
            from torch.utils.cpp_extension import load

            major, minor = torch.cuda.get_device_capability()
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
            os.environ.setdefault("MAX_JOBS", "16")
            _EXT = load(
                name=f"miv_tbe_sm{major}{minor}",
                sources=[str(_CSRC / s) for s in _SOURCES],
                extra_include_paths=[str(_CSRC)],
                extra_cflags=_winenv.host_cflags(),
                extra_cuda_cflags=["-O3", "--fmad=true", "-lineinfo"],
                build_directory=str(_build_dir()),
                verbose=bool(int(os.environ.get("MIV_TBE_VERBOSE", "0"))),
            )
    return _EXT


# ---------------------------------------------------------------------------
# descriptors
# ---------------------------------------------------------------------------
@dataclass
class BF16Desc:
    """Dense bf16 weight on the bf16 MIV path (same hook, same WPR table)."""

    weight: torch.Tensor
    bias: Optional[torch.Tensor] = None
    wpr: int = 1
    unroll: int = 2
    kind: str = "bf16"

    @property
    def n(self) -> int:
        return int(self.weight.shape[0])

    @property
    def k(self) -> int:
        return int(self.weight.shape[1])

    @property
    def stream_bytes(self) -> int:
        return int(self.weight.numel()) * 2


@dataclass
class TBEDesc:
    """Coded weight [n, k]: the row concatenation of one or more TBE tensors."""

    n: int
    k: int
    wpr: int
    unroll: int
    planes: torch.Tensor     # int32 [n*k/64*6]
    smb: torch.Tensor        # uint8 [n*k]
    esc: torch.Tensor        # uint8 [E + ESC_PAD]
    rowparam: torch.Tensor   # int32 [n]   base | mode << 8
    escbase: torch.Tensor    # int32 [n*wpr]
    escapes: int
    bias: Optional[torch.Tensor] = None
    version: int = DEFAULT_VERSION
    parts: List[Dict[str, Any]] = field(default_factory=list)
    kind: str = "tbe"
    autotune: bool = True    # take (version, unroll) per M from TBE_TABLE when present

    @property
    def stream_bytes(self) -> int:
        """Bytes one call reads from memory (weights side): the coded stream."""
        return (int(self.planes.numel()) * 4 + int(self.smb.numel()) + int(self.escapes) +
                int(self.rowparam.numel()) * 4 + int(self.escbase.numel()) * 4)

    @property
    def bits_per_weight(self) -> float:
        return self.stream_bytes * 8.0 / (self.n * self.k)

    def clone(self) -> "TBEDesc":
        return TBEDesc(n=self.n, k=self.k, wpr=self.wpr, unroll=self.unroll,
                       planes=self.planes.clone(), smb=self.smb.clone(), esc=self.esc.clone(),
                       rowparam=self.rowparam.clone(), escbase=self.escbase.clone(),
                       escapes=self.escapes, bias=self.bias, version=self.version,
                       parts=list(self.parts), autotune=self.autotune)

    def with_(self, **kw) -> "TBEDesc":
        d = TBEDesc(**{**self.__dict__})
        for key, val in kw.items():
            setattr(d, key, val)
        return d


def _dense_config(n: int, k: int) -> Tuple[int, int]:
    from . import miv_gemv as mg

    wpr, unroll = mg.config_for(n, k)
    return int(wpr), int(unroll)


def default_unroll(n: int, k: int) -> int:
    return int(TBE_UNROLL.get((int(n), int(k)), 2))


def bf16_desc(weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> BF16Desc:
    if weight.dtype != torch.bfloat16 or weight.dim() != 2 or not weight.is_contiguous():
        raise ValueError("bf16_desc: contiguous 2-D bf16 weight required")
    wpr, unroll = _dense_config(*weight.shape)
    return BF16Desc(weight=weight, bias=bias, wpr=wpr, unroll=unroll)


def _as_int32_words(values: torch.Tensor) -> torch.Tensor:
    v = values.reshape(-1)
    if v.dtype == torch.int32:
        return v.contiguous()
    v = v.to(torch.int64)
    v = torch.where(v >= 0x80000000, v - 0x100000000, v)
    return v.to(torch.int32).contiguous()


def _encode(w: torch.Tensor):
    from glc_loader.tbe_container import encode_tbe

    return encode_tbe(w.detach().to("cpu").contiguous(), layout="mma16")


def _part(p, nm: str, dev) -> Dict[str, Any]:
    """Normalise one part to flat arrays.  Accepts a bf16 tensor (encoded exactly
    at load), a CPU ``TBETensor`` (bundle payload), or a device-resident
    ``glc_loader.tbe_mma.TBEDevice`` (the serving loader's upload, esc padded)."""
    if isinstance(p, torch.Tensor):
        c = _encode(p)
        src = "bf16_encoded_at_load"
    else:
        c = p
        src = type(p).__name__
    layout = getattr(c, "layout", "mma16")          # TBEDevice is mma16 by construction
    if layout != "mma16":
        raise ValueError(f"{nm}: layout {layout!r}, mma16 required")
    n, k = int(c.shape[0]), int(c.shape[1])
    escapes = int(c.escapes)
    esc = c.esc.reshape(-1)
    pad = int(getattr(c, "esc_pad", 0))
    if int(esc.numel()) != escapes + pad:
        raise ValueError(f"{nm}: esc holds {int(esc.numel())} bytes, expected {escapes}+{pad}")
    if int(c.tiles) != n * k // TILE:
        raise ValueError(f"{nm}: tile count does not match shape (partial tile?)")
    return {"name": nm, "source": src, "n": n, "k": k, "mode": int(c.mode), "base": int(c.base),
            "escapes": escapes, "planes": _as_int32_words(c.planes), "smb": c.smb.reshape(-1),
            "esc": esc, "esc_pad": pad}


def build_tbe_desc(parts: Sequence[Any], device, *, wpr: Optional[int] = None,
                   unroll: Optional[int] = None, version: int = DEFAULT_VERSION,
                   bias: Optional[torch.Tensor] = None,
                   names: Optional[Sequence[str]] = None) -> TBEDesc:
    """Row-concatenate TBE parts into one descriptor on ``device``.

    Parts: ``TBETensor`` (CPU bundle payload), ``TBEDevice`` (already on the
    GPU as the serving loader uploads it), or a bf16 tensor (encoded exactly
    with the shipped encoder).  Tiles are row-local (K % 64 == 0), so
    concatenation is plain array concatenation; each row keeps its own
    exponent window via ``rowparam``.  A single ``TBEDevice`` on ``device`` is
    used IN PLACE (no copy): the descriptor only adds rowparam + escbase.
    """
    dev = torch.device(device)
    ps = [_part(p, names[j] if names else f"part{j}", dev) for j, p in enumerate(parts)]
    k = ps[0]["k"]
    if any(p["k"] != k for p in ps):
        raise ValueError("build_tbe_desc: parts disagree on K")
    if k % TILE:
        raise ValueError(f"K={k} is not a multiple of {TILE}")
    n = sum(p["n"] for p in ps)
    dwpr, _ = _dense_config(n, k)
    wpr = int(wpr or dwpr)
    unroll = int(unroll or default_unroll(n, k))
    if k % (TILE * wpr):
        raise ValueError(f"K={k} not a multiple of 64*WPR={64 * wpr}")
    escapes = sum(p["escapes"] for p in ps)
    one = ps[0]
    if (len(ps) == 1 and one["planes"].device == dev and one["esc_pad"] >= ESC_PAD
            and one["esc"].data_ptr() % 4 == 0):
        planes, smb, esc = one["planes"], one["smb"].contiguous(), one["esc"]
    else:
        planes = torch.cat([p["planes"].to(dev) for p in ps])
        smb = torch.cat([p["smb"].to(dev, torch.uint8) for p in ps])
        esc = torch.cat([p["esc"][:p["escapes"]].to(dev, torch.uint8) for p in ps] +
                        [torch.zeros(ESC_PAD, dtype=torch.uint8, device=dev)])
    rowparam = torch.cat([torch.full((p["n"],), p["base"] | (p["mode"] << 8), dtype=torch.int32)
                          for p in ps]).to(dev)
    escbase = compute_escbase(planes, n, k, wpr, expect_escapes=escapes)
    if bias is not None:
        bias = bias.to(device=dev, dtype=torch.bfloat16).contiguous()
    meta = [{key: p[key] for key in ("name", "source", "n", "mode", "base", "escapes")} for p in ps]
    return TBEDesc(n=n, k=k, wpr=wpr, unroll=unroll, planes=planes, smb=smb, esc=esc,
                   rowparam=rowparam, escbase=escbase, escapes=escapes, bias=bias,
                   version=int(version), parts=meta)


def compute_escbase(planes: torch.Tensor, n: int, k: int, wpr: int,
                    expect_escapes: Optional[int] = None) -> torch.Tensor:
    """escbase[row*wpr + s] = escapes before K-slice s of row (tile order)."""
    tiles_per_row = k // TILE
    if planes.is_cuda:
        cnt = extension().tile_escapes(planes)
    else:
        cnt = tile_escapes_cpu(planes)
    per_slice = cnt.to(torch.int64).reshape(n, wpr, tiles_per_row // wpr).sum(dim=2).reshape(-1)
    total = int(per_slice.sum().item())
    if expect_escapes is not None and total != int(expect_escapes):
        raise ValueError(f"plane escape count {total} != esc length {expect_escapes}")
    if total >= (1 << 31):
        raise ValueError("more than 2^31 escapes in one descriptor")
    base = torch.cumsum(per_slice, 0) - per_slice
    return base.to(torch.int32).contiguous()


_POP8 = None


def tile_escapes_cpu(planes: torch.Tensor) -> torch.Tensor:
    global _POP8
    if _POP8 is None:
        _POP8 = torch.tensor([bin(i).count("1") for i in range(256)], dtype=torch.int32)
    p = planes.reshape(-1, 3, 2).to(torch.int32)
    m = ~(p[:, 0] | p[:, 1] | p[:, 2])                    # int32 [T, 2]
    b = m.contiguous().view(torch.uint8).to(torch.int64)  # [T, 8]
    return _POP8[b].sum(dim=1).to(torch.int32)


Desc = Union[BF16Desc, TBEDesc]


# ---------------------------------------------------------------------------
# the hook
# ---------------------------------------------------------------------------
def _x2(x: torch.Tensor, k: int) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1])
    if x2.shape[-1] != k:
        raise ValueError(f"x has K={x2.shape[-1]}, weight K={k}")
    if x2.dtype != torch.bfloat16:
        x2 = x2.to(torch.bfloat16)
    if x2.stride(-1) != 1 or x2.stride(0) % 8 or x2.data_ptr() % 16:
        x2 = x2.contiguous()
        if x2.data_ptr() % 16:
            x2 = x2.clone()
    return x2


def linear(x: torch.Tensor, desc: Desc, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``F.linear(x, W, bias)`` for 1 <= M <= 8 through MIV (bf16) or MIV-TBE.

    ``out`` (optional) is a preallocated bf16 [M, N] (row stride may exceed N,
    unit column stride) -- no allocation in the call, CUDA-graph safe.
    """
    lead = x.shape[:-1]
    x2 = _x2(x, desc.k)
    m = int(x2.shape[0])
    if not 1 <= m <= MAX_M:
        raise ValueError(f"miv_tbe.linear: 1 <= M <= {MAX_M}, got {m}")
    y = out if out is not None else torch.empty(m, desc.n, dtype=torch.bfloat16,
                                                device=x2.device)
    if desc.kind == "bf16":
        from . import miv_gemv as mg

        ext = mg.extension()
        if hasattr(ext, "miv_gemv_out"):       # newer miv_gemv (engine branch)
            ext.miv_gemv_out(x2, desc.weight, desc.bias, y, int(desc.wpr), int(desc.unroll))
        else:                                  # c155643be: allocating entry point
            y.copy_(ext.miv_gemv(x2, desc.weight, desc.bias, int(desc.wpr), int(desc.unroll)))
    elif desc.kind == "tbe":
        ver, u = desc.version, desc.unroll
        if desc.autotune:
            ver, u = TBE_TABLE.get((desc.n, desc.k, m), (ver, u))
        extension().miv_tbe_out(x2, desc.planes, desc.smb, desc.esc, desc.rowparam,
                                desc.escbase, desc.bias, y, int(desc.n), int(desc.k),
                                int(desc.wpr), int(u), int(ver))
    else:
        raise ValueError(f"unknown descriptor kind {desc.kind!r}")
    if out is not None:
        return out
    return y.reshape(*lead, desc.n)


def decode(desc: TBEDesc, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Decode-only twin: same traversal and decode function as the fused kernel."""
    w = out if out is not None else torch.empty(desc.n, desc.k, dtype=torch.bfloat16,
                                                device=desc.planes.device)
    extension().miv_tbe_decode_out(desc.planes, desc.smb, desc.esc, desc.rowparam,
                                   desc.escbase, w, int(desc.n), int(desc.k), int(desc.wpr),
                                   int(desc.unroll))
    return w


# ---------------------------------------------------------------------------
# model integration (explicit, default-off): glc_serve TBEServeLinear modules
# ---------------------------------------------------------------------------
class Stats:
    def __init__(self):
        self.fused_by_m: Dict[int, int] = {}
        self.fallback_by_m: Dict[int, int] = {}

    def as_dict(self) -> Dict[str, Any]:
        return {"fused_calls_by_m": dict(sorted(self.fused_by_m.items())),
                "fallback_calls_by_m": dict(sorted(self.fallback_by_m.items()))}


STATS = Stats()


def install_tbe_fused(root: torch.nn.Module) -> int:
    """Route every resident ``glc_serve.modules.TBEServeLinear`` under ``root``
    through the fused kernel for 1 <= M <= 8 (descriptor built once, in place
    on the module's own device container: no copy).  M > 8 keeps the module's
    original forward.  Returns the number of modules patched."""
    from .modules import TBEServeLinear

    count = 0
    for _name, mod in root.named_modules():
        if not isinstance(mod, TBEServeLinear) or getattr(mod, "_miv_tbe_desc", None) is not None:
            continue
        if mod.offloaded:
            continue
        dc = mod.device_container()
        desc = build_tbe_desc([dc], dc.device, bias=mod.bias.detach() if mod.bias is not None
                              else None, names=[mod.name])
        mod._miv_tbe_desc = desc
        orig = mod.forward

        def fwd(x, _m=mod, _orig=orig):
            m = int(x.numel() // max(1, x.shape[-1]))
            if 1 <= m <= MAX_M:
                STATS.fused_by_m[m] = STATS.fused_by_m.get(m, 0) + 1
                return linear(x, _m._miv_tbe_desc).to(x.dtype)
            STATS.fallback_by_m[m] = STATS.fallback_by_m.get(m, 0) + 1
            return _orig(x)

        mod.forward = fwd
        count += 1
    return count


# ---------------------------------------------------------------------------
# bundle loading (georefine.tbe.serve.v1)
# ---------------------------------------------------------------------------
class Bundle:
    """Minimal reader for a glc_serve bundle directory (manifest + shards)."""

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "serve_manifest.json").read_text())
        self.by_name = {t["name"]: t for t in self.manifest["tensors"]}
        self._handles: Dict[int, Any] = {}

    def _shard_path(self, idx: int) -> Path:
        rec = self.manifest["shards"][idx]
        rel = rec.get("path") or rec.get("file") or f"shards/shard-{idx:05d}.safetensors"
        return self.root / rel

    def _handle(self, idx: int):
        if idx not in self._handles:
            from safetensors import safe_open

            self._handles[idx] = safe_open(str(self._shard_path(idx)), framework="pt",
                                           device="cpu")
        return self._handles[idx]

    def entry(self, name: str) -> Dict[str, Any]:
        return self.by_name[name]

    def read(self, name: str):
        """TBETensor (arrays exactly as stored) for ``tbe`` entries, else the raw tensor."""
        from glc_loader.tbe_container import TBETensor

        e = self.by_name[name]
        h = self._handle(int(e["shard"]))
        if e["kind"] == "raw":
            return h.get_tensor(name)
        coded = e.get("coded_shape") or e["shape"]
        return TBETensor(
            shape=(int(coded[0]), int(coded[1])), layout=str(e["layout"]), mode=int(e["mode"]),
            base=int(e["base"]), tiles=int(e["tiles"]), escapes=int(e["escapes"]),
            planes=h.get_tensor(f"{name}.planes"), smb=h.get_tensor(f"{name}.smb"),
            esc=h.get_tensor(f"{name}.esc"), sbbase=h.get_tensor(f"{name}.sbbase"),
            superblock=int(e["superblock"]),
        )

    def desc(self, names: Sequence[str], device, **kw) -> Desc:
        """One descriptor for the row concatenation of ``names``.

        All-raw -> ``BF16Desc`` (bf16 MIV).  Any TBE part -> ``TBEDesc``; raw
        parts are then encoded exactly at load (G1 checks the decode).
        """
        payloads = [self.read(nm) for nm in names]
        if all(isinstance(p, torch.Tensor) for p in payloads):
            w = payloads[0] if len(payloads) == 1 else torch.cat(payloads, 0)
            return bf16_desc(w.to(torch.device(device)).contiguous())
        return build_tbe_desc(payloads, device, names=list(names), **kw)


__all__ = ["BF16Desc", "Bundle", "DEFAULT_VERSION", "Desc", "ESC_PAD", "MAX_M", "TBEDesc",
           "STATS", "TBE_TABLE", "TBE_UNROLL", "install_tbe_fused", "load_default_configs", "load_table", "bf16_desc", "build_tbe_desc", "compute_escbase", "decode",
           "default_unroll", "extension", "linear", "tile_escapes_cpu"]
