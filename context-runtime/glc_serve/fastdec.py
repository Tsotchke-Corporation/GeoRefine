"""FastDecoder: a static-buffer, whole-step CUDA-graph decode engine for the
Qwen3.5/3.8 hybrid (48 Gated-DeltaNet + 16 gated full-attention layers), with
fast exact speculative verification (design E1 + E4, F18 lever L2).

Every projection goes through MIV-GEMV (``miv_gemv.miv_gemv_out``: M-invariant,
row t of an M-row call is bitwise the M=1 call), every small op through the
M-invariant kernels in ``fastdec_kernels.cu``, and the Gated-DeltaNet
recurrence runs the k+1 verify tokens SEQUENTIALLY inside one kernel with the
single-step FP32 op order.  So a verify step of M = k+1 rows reproduces k+1
plain M=1 decode steps bit for bit, and every piece of decode state is indexed
by ABSOLUTE position (attention KV rows, a 16-deep conv-input ring, an R-deep
recurrent-state ring): rejecting drafts is "set the position", nothing is
restored or replayed.

Weights are reached through a per-tensor descriptor (``Lin``): ``bf16`` today
(the parent, or a codec decoded once), ``tbe`` (the in-register TBE decode in
``miv_gemv.miv_tbe_out``) or any later format with an ``__call__(x, y)``.
Fused row-concatenations (GDN qkv|z|b|a, attention q|k|v, MLP gate|up) are one
descriptor each; the per-shape (WPR, unroll) comes from a tune JSON that every
run of a comparison MUST share (WPR fixes the reduction order).

The prompt prefill runs through the HF model (unchanged path, cuBLAS for M>8);
its caches are imported once.  Both arms of every gate (plain vs speculative,
codec vs parent) share that prefill.

Greedy, batch 1.  Default-off: nothing imports this module unless asked.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from . import miv_gemv as mg

_LOG = logging.getLogger(__name__)
# TBE descriptors split K into 64-element tiles per warp slice (glc_serve.tbe_desc.TILE).
_TBE_TILE = 64


def _has_tbe21_codec(module: Any) -> bool:
    """Whether a linear owns TBE21 streams rather than a dense ``weight`` tensor."""
    return hasattr(module, "codec_v21")


def _make_tbe21_descriptor(modules: Sequence[Any], factory: Callable[[Sequence[Any]], Any]) -> Any:
    """Build the fused TBE21 descriptor only for an all-TBE21 projection group."""
    if not modules or not all(_has_tbe21_codec(module) for module in modules):
        raise ValueError("TBE21 descriptor requires one or more TBE21 linear modules")
    descriptor = factory(modules)
    if getattr(descriptor, "kind", None) != "tbe21":
        raise TypeError("TBE21 descriptor factory returned an unexpected descriptor kind")
    return descriptor


def untuned_config(key: Tuple[int, int], name: str, *, tbe: bool = False) -> Tuple[int, int]:
    """The (WPR, unroll) used for a shape missing from the tune file.

    This is ``miv_gemv.default_config``, the same shape-only rule ``FastDecoder`` already
    applies to single dense modules. It is exactness-safe: every MIV/TBE kernel fixes its
    reduction order per shape from (WPR, unroll), never from M, so a fallback config is
    still M-invariant and deterministic for a given tune file, and the TBE and dense BF16
    descriptors of one shape get the same config (G3a). Only speed is untuned. A TBE shape
    whose K is not a multiple of 64*WPR cannot use it and is refused by name.
    """
    n, k = int(key[0]), int(key[1])
    wpr, unroll = mg.default_config(n, k)
    if tbe and k % (_TBE_TILE * wpr):
        raise KeyError(
            f"no tuned (WPR, unroll) for shape {(n, k)} ({name}) and the shape-only default "
            f"WPR={wpr} does not split K={k} into {_TBE_TILE}-element tiles; pass --tune with a "
            "tune file that covers this shape")
    return int(wpr), int(unroll)

_EXT = None
_LOCK = threading.Lock()
_FUNCS = ["embed", "embed_q8", "rmsnorm", "gdn_conv", "gdn_recur", "gdn_recur_hv16",
          "gdn_recur_native", "gdn_recur_native_hv16", "attn_prep", "attn_split", "attn_combine",
          "silu_mul", "argmax_rows", "ar_advance", "draft_advance"]
_CPP = r"""
#include <torch/extension.h>
void embed(at::Tensor ids, at::Tensor tab, at::Tensor out, int64_t M);
void embed_q8(at::Tensor ids, at::Tensor qs, at::Tensor sc, at::Tensor out, int64_t M);
void rmsnorm(at::Tensor h, c10::optional<at::Tensor> delta, at::Tensor w, at::Tensor out, int64_t M,
             double eps, int64_t unit_offset);
void gdn_conv(at::Tensor in, at::Tensor hist, at::Tensor cw, at::Tensor pos, at::Tensor out, int64_t M);
void gdn_recur(at::Tensor conv, at::Tensor zba, int64_t z_off, int64_t b_off, int64_t a_off,
               at::Tensor A_log, at::Tensor dt_bias, at::Tensor nw, at::Tensor state, int64_t R,
               at::Tensor pos, at::Tensor out, int64_t M, double eps, int64_t nkh_rep);
void gdn_recur_hv16(at::Tensor conv, at::Tensor zba, int64_t z_off, int64_t b_off, int64_t a_off,
                    at::Tensor A_log, at::Tensor dt_bias, at::Tensor nw, at::Tensor state, int64_t R,
                    at::Tensor pos, at::Tensor out, int64_t M, double eps, int64_t nkh_rep);
void gdn_recur_native(at::Tensor conv, at::Tensor zba, int64_t z_off, int64_t b_off, int64_t a_off,
                      at::Tensor A_log, at::Tensor dt_bias, at::Tensor nw, at::Tensor state, int64_t R,
                      at::Tensor pos, at::Tensor out, int64_t M, double eps, int64_t nkh_rep,
                      int64_t A_log_fp32, int64_t dt_bias_fp32, int64_t nw_fp32);
void gdn_recur_native_hv16(at::Tensor conv, at::Tensor zba, int64_t z_off, int64_t b_off, int64_t a_off,
                           at::Tensor A_log, at::Tensor dt_bias, at::Tensor nw, at::Tensor state, int64_t R,
                           at::Tensor pos, at::Tensor out, int64_t M, double eps, int64_t nkh_rep,
                           int64_t A_log_fp32, int64_t dt_bias_fp32, int64_t nw_fp32);
void attn_prep(at::Tensor qkv, at::Tensor qnw, at::Tensor knw, at::Tensor cosT, at::Tensor sinT,
               at::Tensor kvpos, at::Tensor ropeoff, at::Tensor qout, at::Tensor kc, at::Tensor vc,
               int64_t M, int64_t NQ, int64_t NKV, double eps);
void attn_split(at::Tensor q, at::Tensor kc, at::Tensor vc, at::Tensor kvpos, at::Tensor pacc,
                at::Tensor pml, int64_t M, int64_t NQ, int64_t NKV, int64_t NS, double scaling);
void attn_combine(at::Tensor pacc, at::Tensor pml, at::Tensor qkv, at::Tensor kvpos, at::Tensor out,
                  int64_t M, int64_t NQ, int64_t NS);
void silu_mul(at::Tensor gu, at::Tensor out, int64_t M);
void argmax_rows(at::Tensor lg, at::Tensor out, int64_t M);
void ar_advance(at::Tensor am, at::Tensor vtok, at::Tensor log, at::Tensor pos);
void draft_advance(at::Tensor am, at::Tensor vtok, at::Tensor slot, at::Tensor mtok, at::Tensor mkv,
                   int64_t M);
"""
SPLIT = 256
MAX_M = 8


def _own_build_dir() -> Path:
    """A directory of its own: two load_inline extensions sharing one build dir rewrite
    each other's sources and rebuild (and race) on every load."""
    d = mg._build_dir() / "fastdec"
    d.mkdir(parents=True, exist_ok=True)
    return d


def rope_tables(rotary, n: int, device, dtype=torch.bfloat16):
    """Build the Qwen text-position rows used by the native batch adapter.

    This is the existing FastDecoder's 1024-row, three-axis construction, with
    the requested row count supplied by BatchDecoder. The single-session path
    keeps its established construction unchanged.
    """
    if type(n) is not int or n <= 0:
        raise ValueError("rotary table row count must be a positive integer")
    cs, sn = [], []
    for p0 in range(0, n, 1024):
        count = min(1024, n - p0)
        pid = torch.arange(p0, p0 + count, device=device).view(1, 1, -1).expand(3, 1, -1)
        c, s = rotary(torch.zeros(1, device=device, dtype=dtype), pid)
        cs.append(c[0])
        sn.append(s[0])
    return torch.cat(cs).contiguous(), torch.cat(sn).contiguous()


def ext():
    global _EXT
    if _EXT is None:
        with _LOCK:
            if _EXT is None:
                from torch.utils.cpp_extension import load_inline

                major, minor = torch.cuda.get_device_capability()
                os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
                src = (Path(__file__).with_name("fastdec_kernels.cu")).read_text()
                _EXT = load_inline(name=f"fastdec_sm{major}{minor}", cpp_sources=[_CPP],
                                   cuda_sources=[src], functions=_FUNCS,
                                   extra_cuda_cflags=["-O3", "-lineinfo"],
                                   build_directory=str(_own_build_dir()), verbose=False)
    return _EXT


# ---------------------------------------------------------------------------
# weight descriptors
# ---------------------------------------------------------------------------
class Lin:
    """bf16 weight [N, K] through MIV-GEMV with a fixed per-shape (WPR, unroll)."""

    kind = "bf16"

    def __init__(self, weight: torch.Tensor, cfg: Tuple[int, int]):
        assert weight.dtype == torch.bfloat16 and weight.is_contiguous() and weight.dim() == 2
        self.w = weight
        self.N, self.K = int(weight.shape[0]), int(weight.shape[1])
        self.wpr, self.unroll = int(cfg[0]), int(cfg[1])

    @property
    def bytes(self) -> int:
        return self.w.numel() * 2

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        mg.extension().miv_gemv_out(x, self.w, None, y, self.wpr, self.unroll)


class DenseTemporaryLin:
    """Opt-in BHE decode descriptor using the canonical dense MIV-GEMV path."""

    kind = 'bf16_dense_temporary'

    def __init__(self, descriptor, cfg, *, max_bytes):
        self.descriptor = descriptor
        self.wpr, self.unroll = map(int, cfg)
        members = getattr(descriptor, 'members', None)
        if members is None:
            members = [(descriptor, int(descriptor.N))]
        self.members = list(members)
        self.N = sum(int(n) for _d, n in self.members)
        self.K = int(self.members[0][0].K)
        if any(int(d.K) != self.K for d, _n in self.members):
            raise ValueError('dense temporary BHE group members have different K')
        self.requirements = self._requirements()
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('dense decode mode requires a positive max_decode_dense_bytes')
        if self.requirements['total_bytes'] > max_bytes:
            raise MemoryError(
                f'dense decode temporary budget exceeded: required {self.requirements["total_bytes"]} > {max_bytes}')
        self.max_bytes = max_bytes

    def _requirements(self):
        dense = 0
        status = 0
        for desc, _n in self.members:
            padded_n = int(getattr(desc, 'padded_N', 0))
            logical_n = int(getattr(desc, 'N', 0))
            if logical_n <= 0 or padded_n < logical_n:
                raise ValueError('dense decode descriptor has invalid logical/padded N')
            tiles = getattr(desc, 'provenance', {}).get('tiles')
            if padded_n <= 0 or type(tiles) is not int or tiles <= 0:
                raise ValueError('dense decode descriptor lacks validated padded shape/status metadata')
            dense += padded_n * self.K * 2
            status += tiles * 256 * 4
        concat = self.N * self.K * 2 if len(self.members) > 1 else 0
        return {'decoded_weight_bytes': dense, 'status_bytes': status,
                'concatenated_weight_bytes': concat,
                'total_bytes': dense + status + concat}

    @property
    def bytes(self):
        return sum(int(getattr(d, 'bytes', 0)) for d, _n in self.members)

    @property
    def coded_bytes(self):
        return sum(getattr(d, 'coded_bytes', d.bytes) for d, _n in self.members)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('dense decode temporary mode is not graph-capturable')
        if x.dtype != torch.bfloat16 or y.dtype != torch.bfloat16 or x.device != y.device:
            raise RuntimeError('dense decode temporary activation/output dtype or device mismatch')
        if x.ndim != 2 or x.shape[1] != self.K or tuple(y.shape) != (x.shape[0], self.N):
            raise RuntimeError('dense decode temporary activation/output shape mismatch')
        decoded = []
        try:
            for desc, n in self.members:
                dense, status = desc._backend.decode(desc.plan)
                padded_n = int(desc.padded_N)
                tiles = desc.provenance.get('tiles')
                expected_status = tiles * 256 if type(tiles) is int and tiles > 0 else None
                if (not isinstance(dense, torch.Tensor) or dense.dtype != torch.bfloat16 or
                        dense.device != x.device or tuple(dense.shape) != (padded_n, self.K)):
                    raise RuntimeError('dense decode temporary weight shape/dtype/device mismatch')
                if (not isinstance(status, torch.Tensor) or status.dtype != torch.int32 or
                        status.device != x.device or status.numel() != expected_status):
                    raise RuntimeError('dense decode temporary status shape/dtype/device mismatch')
                if bool(torch.any(status != 0).item()):
                    raise RuntimeError('dense decode temporary packet status failure')
                decoded.append(dense[:int(n), :self.K])
            weight = decoded[0] if len(decoded) == 1 else torch.cat(decoded, dim=0)
            if tuple(weight.shape) != (self.N, self.K):
                raise RuntimeError('dense decode temporary logical weight shape mismatch')
            mg.extension().miv_gemv_out(x, weight, None, y, self.wpr, self.unroll)
        finally:
            del decoded


def fuse_rows(mods: Sequence[nn.Linear]) -> torch.Tensor:
    """Concatenate linears' weights by rows; rebind each module to a view (no 2nd copy)."""
    w = torch.cat([m.weight.detach() for m in mods], dim=0).contiguous()
    r = 0
    for m in mods:
        n = int(m.weight.shape[0])
        m.weight.data = w[r:r + n]
        r += n
    return w


def load_tune(path) -> Dict[Tuple[int, int], Tuple[int, int]]:
    cfg = json.loads(Path(path).read_text())["config"]
    return {tuple(int(v) for v in k.split("x")): (int(a), int(b)) for k, (a, b) in cfg.items()}


# ---------------------------------------------------------------------------
# model plumbing
# ---------------------------------------------------------------------------
def _text(model):
    for p in ("model.language_model", "model"):
        try:
            return model.get_submodule(p)
        except AttributeError:
            continue
    raise AttributeError("no text model")


class _Layer:
    """Attribute bag for one decoder layer's descriptors and constants.

    ``FastDecoder.__init__`` fills it (``kind`` = ``"gdn"`` or ``"attn"``, the fused weight
    descriptors, norm weights and shape constants) and ``bidec.BatchDecoder`` builds its own
    instances the same way. It has no behaviour of its own by design.
    """


def split_attn_gdn(layers: Sequence[Any]) -> Tuple[Any, Optional[Any]]:
    """``(att0, gd0)``: the first full-attention and the first Gated-DeltaNet ``_Layer``.

    Read only for per-kind buffer dims (``att0.NQ``/``.NKV``/``.qkv.N``, ``gd0.C``/
    ``.qkvzba.N``/``.vdim``) by ``bidec.BatchDecoder``. Pure bookkeeping over already-built
    layer objects: no CUDA and no tensor op. ``gd0`` is ``None`` when the model has no GDN
    layer; a model without a full-attention layer is refused by name.
    """
    rows = list(layers)
    att0 = next((row for row in rows if row.kind == "attn"), None)
    if att0 is None:
        kinds = sorted({row.kind for row in rows})
        raise ValueError(
            f"native batch decode requires at least one full-attention layer; found kinds {kinds}")
    gd0 = next((row for row in rows if row.kind == "gdn"), None)
    return att0, gd0


def gdn_template_dims(gd0: Optional[Any]) -> Tuple[int, int, int]:
    """``(conv_dim, fused qkv|z|b|a width, value_dim)`` of the template GDN layer.

    ``(0, 0, 0)`` when ``gd0`` is ``None`` (no GDN layer), so GDN-shaped batch buffers are
    allocated zero-sized.
    """
    if gd0 is None:
        return 0, 0, 0
    return int(gd0.C), int(gd0.qkvzba.N), int(gd0.vdim)


def gdn_template_geometry(gd0: Optional[Any]) -> Tuple[int, int, int, int, int, int]:
    """Return ``(HK, HV, DK, DV, conv_kernel, conv_dim)`` for a GDN template."""
    if gd0 is None:
        return 0, 0, 0, 0, 0, 0
    dv = int(getattr(gd0, "gdn_dv", 128))
    hv_attr = getattr(gd0, "gdn_hv", None)
    hv = int(hv_attr if hv_attr is not None else int(gd0.vdim) // dv)
    dk = int(getattr(gd0, "gdn_dk", 128))
    hk = int(getattr(gd0, "gdn_hk", 16))
    kernel = int(getattr(gd0, "conv_kernel", 4))
    return hk, hv, dk, dv, kernel, int(gd0.C)


def validate_gdn_geometry(*, head_k_dim: int, head_v_dim: int, num_v_heads: int,
                          num_k_heads: int, conv_kernel: int, hv16: bool = False) -> None:
    """Fail closed on the two native GDN layouts, with the small layout opt-in."""
    expected_hv = 16 if bool(hv16) else 48
    expected = (128, 128, expected_hv, 16, 4)
    actual = (int(head_k_dim), int(head_v_dim), int(num_v_heads), int(num_k_heads), int(conv_kernel))
    if actual != expected:
        mode = "gdn_hv16" if hv16 else "default GDN"
        raise ValueError(f"{mode} requires GDN geometry {expected}, got {actual}")


def validate_gdn_native_dtypes(layer: Any, device: torch.device) -> Tuple[int, int, int]:
    """Validate native GDN parameter storage before selecting a typed CUDA entrypoint."""
    expected = (("A_log", int(getattr(layer, "gdn_hv", 48))),
                ("dt_bias", int(getattr(layer, "gdn_hv", 48))),
                ("nw", int(getattr(layer, "gdn_dv", 128))))
    codes = []
    for name, size in expected:
        value = getattr(layer, name, None)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"GDN native {name} must be a torch.Tensor")
        if value.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError(f"GDN native {name} must be bfloat16 or float32, got {value.dtype}")
        if value.device != device:
            raise ValueError(f"GDN native {name} must be on {device}, got {value.device}")
        if value.dim() != 1 or value.numel() != size or not value.is_contiguous():
            raise ValueError(f"GDN native {name} must be contiguous [{size}], got shape={tuple(value.shape)}")
        codes.append(int(value.dtype == torch.float32))
    return tuple(codes)


def gdn_recur_entrypoint(extension: Any, *, hv16: bool, native: bool) -> Any:
    """Select the legacy or explicitly opted-in native-dtype GDN entrypoint."""
    if type(hv16) is not bool or type(native) is not bool:
        raise ValueError("hv16 and native must be bools")
    if native:
        return extension.gdn_recur_native_hv16 if hv16 else extension.gdn_recur_native
    return extension.gdn_recur_hv16 if hv16 else extension.gdn_recur


class FastDecoder:
    def __init__(self, model: nn.Module, mtp: Optional[nn.Module], *, tune: Dict, max_len: int = 4096,
                 R: int = 16, device="cuda:0", lin_factory=None, fuse: bool = True,
                 tbe_unroll: int = 0, gdn_hv16: bool = False,
                 gdn_native_dtypes: bool = False, decode_mode: str = 'fused',
                 max_decode_dense_bytes: int | None = None):
        if type(gdn_hv16) is not bool:
            raise ValueError("gdn_hv16 must be a bool")
        if type(gdn_native_dtypes) is not bool:
            raise ValueError("gdn_native_dtypes must be a bool")
        if decode_mode not in ('fused', 'dense_temporary'):
            raise ValueError("decode_mode must be 'fused' or 'dense_temporary'")
        if decode_mode == 'dense_temporary' and (type(max_decode_dense_bytes) is not int or max_decode_dense_bytes <= 0):
            raise ValueError('dense_temporary decode requires a positive max_decode_dense_bytes')
        E = ext()
        mg.extension()
        self.model, self.mtp = model, mtp
        self.dev = torch.device(device)
        self.tune = dict(tune)
        tm = _text(model)
        cfg = tm.config
        self.H = int(cfg.hidden_size)
        self.eps = float(cfg.rms_norm_eps)
        self.max_len = int(max_len)
        self.R = int(R)
        self.NS = (self.max_len + SPLIT - 1) // SPLIT
        self.lin_factory = lin_factory or (lambda w, c: Lin(w, c))
        self.tbe_unroll = int(tbe_unroll)
        self.gdn_hv16 = bool(gdn_hv16)
        self.gdn_native_dtypes = bool(gdn_native_dtypes)
        self.decode_mode = decode_mode
        self.max_decode_dense_bytes = max_decode_dense_bytes
        self.decode_dense_requirements = {'peak_bytes': 0, 'aggregate_bytes': 0, 'descriptors': []}
        self.shapes_used: Dict[str, List[int]] = {}
        # Shapes absent from the tune file, with the shape-only config used instead.
        self.untuned_shapes: Dict[str, Dict[str, Any]] = {}
        dev = self.dev
        bf = torch.bfloat16

        def one(m, name):
            """Descriptor for a single module of any supported format."""
            if hasattr(m, "fast_desc"):                       # registered GGUF kernel
                d = m.fast_desc()
                if self.decode_mode == 'dense_temporary' and hasattr(m, 'blobs'):
                    key = (int(m.out_features), int(m.in_features))
                    value = DenseTemporaryLin(d, tuned(key, name), max_bytes=max_decode_dense_bytes)
                    self.decode_dense_requirements['peak_bytes'] = max(self.decode_dense_requirements['peak_bytes'], value.requirements['total_bytes'])
                    self.decode_dense_requirements['aggregate_bytes'] += value.requirements['total_bytes']
                    self.decode_dense_requirements['descriptors'].append(dict(value.requirements, name=name))
                    return value
            elif (hasattr(m, "device_container") or hasattr(m, "pack")
                  or _has_tbe21_codec(m)):
                return L([m], name)
            else:
                w = m.weight.detach().contiguous()
                key = (int(w.shape[0]), int(w.shape[1]))
                d = self.lin_factory(w, self.tune.get(key) or mg.default_config(*key))
            perm = getattr(m, "in_perm", None)
            if perm is not None:
                from .q8serve import PermIn

                d = PermIn(d, perm)
            return d

        def kind_of(m):
            if _has_tbe21_codec(m):
                return "tbe21"
            if hasattr(m, "device_container"):
                return "tbe"
            if hasattr(m, "pack"):
                return "q8"
            if hasattr(m, "fast_desc") or getattr(m, "in_perm", None) is not None:
                return "other"
            return "bf16"

        def tuned(key, name, *, tbe=False):
            cfg = self.tune.get(key)
            if cfg is not None:
                return int(cfg[0]), int(cfg[1])
            cfg = untuned_config(key, name, tbe=tbe)
            label = f"{key[0]}x{key[1]}"
            if label not in self.untuned_shapes:
                _LOG.warning("FastDecoder: shape %s (%s) is not in the tune file; using the "
                             "shape-only default (WPR, unroll)=%s. Results stay exact and "
                             "M-invariant; speed is untuned. Pass --tune with a file covering "
                             "it to tune.", key, name, cfg)
            self.untuned_shapes[label] = {"name": name, "config": list(cfg)}
            return cfg

        def L(mods, name):
            if any(hasattr(m, 'blobs') and hasattr(m, 'fast_desc') for m in mods):
                from .bhe_desc import group_descriptor
                group = group_descriptor(mods)
                if self.decode_mode == 'dense_temporary':
                    key = (int(group.N), int(group.K))
                    value = DenseTemporaryLin(group, tuned(key, name), max_bytes=max_decode_dense_bytes)
                    self.decode_dense_requirements['peak_bytes'] = max(self.decode_dense_requirements['peak_bytes'], value.requirements['total_bytes'])
                    self.decode_dense_requirements['aggregate_bytes'] += value.requirements['total_bytes']
                    self.decode_dense_requirements['descriptors'].append(dict(value.requirements, name=name))
                    return value
                return group
            kinds = {kind_of(m) for m in mods}
            if len(mods) > 1 and (len(kinds) > 1 or "other" in kinds) and not kinds <= {"tbe", "bf16"}:
                from .q8serve import GroupLin                  # mixed formats in one fused group

                g = GroupLin([(one(m, name), int(m.out_features)) for m in mods])
                self.shapes_used[f"{g.N}x{g.K}:group"] = [k for k in sorted(kinds)]
                return g
            if len(mods) == 1 and kinds == {"other"}:
                return one(mods[0], name)
            if kinds == {"tbe21"}:
                from .tbe2_desc import TBE21Lin

                d21 = _make_tbe21_descriptor(mods, TBE21Lin)
                self.shapes_used[f"{d21.N}x{d21.K}:tbe21"] = [len(mods), d21.S]
                return d21
            if any(hasattr(m, "device_container") for m in mods):   # TBE codec (glc_serve.tbe_desc)
                from .tbe_desc import TBELin

                key = (sum(int(m.out_features) for m in mods), int(mods[0].in_features))
                wpr, u = tuned(key, name, tbe=True)
                cfgt = (wpr, self.tbe_unroll or min(int(u), 4))
                self.shapes_used[f"{key[0]}x{key[1]}"] = list(cfgt)
                return TBELin(mods, cfgt)
            if hasattr(mods[0], "pack"):                      # Q8_0 (glc_serve.q8serve)
                from .q8serve import Q8Lin, fuse_q8

                pk = fuse_q8(mods) if len(mods) > 1 else mods[0].pack
                key = (pk.N, pk.K)
                if key not in self.tune:
                    # Q8_0 kernels have no validated shape-only default.
                    raise KeyError(f"no tuned (WPR, unroll) for Q8_0 shape {key} ({name}); pass "
                                   "--tune with a tune file that covers this shape")
                self.shapes_used[f"{key[0]}x{key[1]}"] = list(self.tune[key])
                return Q8Lin(pk, self.tune[key])
            w = fuse_rows(mods) if (fuse and len(mods) > 1) else mods[0].weight.detach()
            if len(mods) == 1 and not w.is_contiguous():
                w = w.contiguous()
            key = (int(w.shape[0]), int(w.shape[1]))
            cfg = tuned(key, name)
            self.shapes_used[f"{key[0]}x{key[1]}"] = list(cfg)
            return self.lin_factory(w, cfg)

        self.layers: List[_Layer] = []
        n_gdn = n_att = 0
        for li, layer in enumerate(tm.layers):
            o = _Layer()
            o.in_ln = layer.input_layernorm.weight.detach()
            o.post_ln = layer.post_attention_layernorm.weight.detach()
            mlp = layer.mlp
            o.gu = L([mlp.gate_proj, mlp.up_proj], "gate|up")
            o.down = L([mlp.down_proj], "down")
            o.F = int(mlp.down_proj.in_features)
            if hasattr(layer, "linear_attn") and layer.linear_attn is not None and \
                    getattr(layer, "block_type", "linear_attention") == "linear_attention":
                la = layer.linear_attn
                o.kind = "gdn"
                o.gi = n_gdn
                n_gdn += 1
                o.qkvzba = L([la.in_proj_qkv, la.in_proj_z, la.in_proj_b, la.in_proj_a], "qkv|z|b|a")
                o.out = L([la.out_proj], "out_proj")
                o.convw = la.conv1d.weight.detach().reshape(la.conv1d.weight.shape[0], -1).contiguous()
                o.A_log = la.A_log.detach()
                o.dt_bias = la.dt_bias.detach()
                o.nw = la.norm.weight.detach()
                o.C = int(la.conv_dim)
                o.conv_kernel = int(getattr(la, "conv_kernel_size", 4))
                o.gdn_hk = int(la.num_k_heads)
                o.gdn_hv = int(la.num_v_heads)
                o.gdn_dk = int(la.head_k_dim)
                o.gdn_dv = int(la.head_v_dim)
                o.z_off = int(la.conv_dim)
                o.b_off = o.z_off + int(la.value_dim)
                o.a_off = o.b_off + int(la.num_v_heads)
                o.nkh_rep = int(la.num_v_heads) // int(la.num_k_heads)
                o.vdim = int(la.value_dim)
                actual = (o.gdn_dk, o.gdn_dv, o.gdn_hv, o.gdn_hk, o.conv_kernel)
                if self.gdn_hv16:
                    validate_gdn_geometry(head_k_dim=o.gdn_dk, head_v_dim=o.gdn_dv,
                                          num_v_heads=o.gdn_hv, num_k_heads=o.gdn_hk,
                                          conv_kernel=o.conv_kernel, hv16=True)
                else:
                    # Preserve the release guard and its assertion contract when opt-in is unset.
                    assert actual[:4] == (128, 128, 48, 16)
                if self.gdn_hv16 and o.C != 2 * o.gdn_hk * o.gdn_dk + o.gdn_hv * o.gdn_dv:
                    raise ValueError(f"GDN conv width {o.C} disagrees with geometry {actual}")
            else:
                sa = layer.self_attn
                o.kind = "attn"
                o.ai = n_att
                n_att += 1
                o.qkv = L([sa.q_proj, sa.k_proj, sa.v_proj], "q|k|v")
                o.o = L([sa.o_proj], "o_proj")
                o.qnw = sa.q_norm.weight.detach()
                o.knw = sa.k_norm.weight.detach()
                o.NQ = int(cfg.num_attention_heads)
                o.NKV = int(cfg.num_key_value_heads)
                o.D = int(sa.head_dim)
                o.scaling = float(sa.scaling)
                assert o.D == 256
            self.layers.append(o)
        if self.gdn_native_dtypes:
            for native_layer in (row for row in self.layers if row.kind == "gdn"):
                native_layer.gdn_dtype_codes = validate_gdn_native_dtypes(native_layer, dev)
        self.n_gdn, self.n_att = n_gdn, n_att
        self.gdn_hk = int(next(o for o in self.layers if o.kind == "gdn").gdn_hk)
        self.gdn_hv = int(next(o for o in self.layers if o.kind == "gdn").gdn_hv)
        self.gdn_dk = int(next(o for o in self.layers if o.kind == "gdn").gdn_dk)
        self.gdn_dv = int(next(o for o in self.layers if o.kind == "gdn").gdn_dv)
        self.gdn_conv_kernel = int(next(o for o in self.layers if o.kind == "gdn").conv_kernel)
        self.gdn_conv_dim = int(next(o for o in self.layers if o.kind == "gdn").C)
        self.final_ln = tm.norm.weight.detach()
        self.embed_q8 = getattr(tm.embed_tokens, "pack", None)
        if self.embed_q8 is None:
            self.embed_w = tm.embed_tokens.weight.detach()
            if self.embed_w.device != dev:
                self.embed_w = self.embed_w.to(dev)
        self.lm = L([model.get_output_embeddings()], "lm_head")
        self.V = self.lm.N
        att0 = next(o for o in self.layers if o.kind == "attn")
        gd0 = next(o for o in self.layers if o.kind == "gdn")
        self.NQ, self.NKV = att0.NQ, att0.NKV

        # rotary tables (HF's own rotary module, text positions t = h = w)
        rot = tm.rotary_emb
        cs, sn = [], []
        for p0 in range(0, self.max_len + 64, 1024):
            n = min(1024, self.max_len + 64 - p0)
            pid = torch.arange(p0, p0 + n, device=dev).view(1, 1, -1).expand(3, 1, -1)
            c, s = rot(torch.zeros(1, device=dev, dtype=bf), pid)
            cs.append(c[0])
            sn.append(s[0])
        self.cosT = torch.cat(cs).contiguous()
        self.sinT = torch.cat(sn).contiguous()
        self.rope_dim = int(self.cosT.shape[1])

        # ---------------- state + buffers ----------------
        z = lambda *s, dt=bf: torch.zeros(*s, dtype=dt, device=dev)  # noqa: E731
        M, H = MAX_M, self.H
        self.kc = z(n_att, self.max_len, self.NKV, 256)
        self.vc = z(n_att, self.max_len, self.NKV, 256)
        self.hist = z(n_gdn, 16, gd0.C)
        self.state = z(n_gdn, self.R, self.gdn_hv, self.gdn_dk, self.gdn_dv, dt=torch.float32)
        i32 = torch.int32
        self.vtok = z(M, dt=i32)
        self.pos = z(1, dt=i32)
        self.ropeoff = z(1, dt=i32)
        self.am = z(M, dt=i32)
        self.log = z(self.max_len + 16, dt=i32)
        self.h = z(M, H)
        self.x = z(M, H)
        self.xf = z(M, H)
        self.proj = z(M, H)
        self.dn = z(M, H)
        self.zba = z(M, gd0.qkvzba.N)
        self.conv = z(M, gd0.C)
        self.go = z(M, gd0.vdim)
        self.aqkv = z(M, att0.qkv.N)
        self.q = z(M, self.NQ, 256)
        self.ao = z(M, self.NQ * 256)
        F = self.layers[0].F
        self.gu = z(M, 2 * F)
        self.act = z(M, F)
        self.logits = z(M, self.V)
        self.pacc = z(M, self.NQ, self.NS, 256, dt=torch.float32)
        self.pml = z(M, self.NQ, self.NS, 2, dt=torch.float32)
        self.graphs: Dict[Tuple[str, int], torch.cuda.CUDAGraph] = {}
        self.mtp_ready = False
        if mtp is not None:
            self._build_mtp(mtp, L)

    # ------------------------------------------------------------------ MTP head
    def _build_mtp(self, mtp, L):
        dev, bf, H, M = self.dev, torch.bfloat16, self.H, MAX_M
        z = lambda *s, dt=bf: torch.zeros(*s, dtype=dt, device=dev)  # noqa: E731
        lay = mtp.layers[0]
        sa, mlp = lay.self_attn, lay.mlp
        m = _Layer()
        m.pre_e = mtp.pre_fc_norm_embedding.weight.detach()
        m.pre_h = mtp.pre_fc_norm_hidden.weight.detach()
        m.fc = L([mtp.fc], "mtp.fc")
        m.in_ln = lay.input_layernorm.weight.detach()
        m.post_ln = lay.post_attention_layernorm.weight.detach()
        m.qkv = L([sa.q_proj, sa.k_proj, sa.v_proj], "mtp q|k|v")
        m.o = L([sa.o_proj], "mtp o")
        m.qnw, m.knw = sa.q_norm.weight.detach(), sa.k_norm.weight.detach()
        m.gu = L([mlp.gate_proj, mlp.up_proj], "mtp gate|up")
        m.down = L([mlp.down_proj], "mtp down")
        m.norm = mtp.norm.weight.detach()
        m.scaling = float(sa.scaling)
        self.m = m
        self.mkc = z(self.max_len, self.NKV, 256)
        self.mvc = z(self.max_len, self.NKV, 256)
        i32 = torch.int32
        self.mtok = z(M, dt=i32)
        self.mkv = z(1, dt=i32)
        self.mropeoff = z(1, dt=i32)
        self.slot = z(1, dt=i32)
        self.mam = z(M, dt=i32)
        self.me = z(M, H)
        self.mcat = z(M, 2 * H)
        self.mh = z(M, H)
        self.mhid = z(M, H)
        self.mx = z(M, H)
        self.mg = z(M, H)
        self.mlogits = z(1, self.V)
        self.mtp_ready = True

    def _embed(self, ids, out, M):
        if self.embed_q8 is not None:
            ext().embed_q8(ids, self.embed_q8.qs, self.embed_q8.sc, out, M)
        else:
            ext().embed(ids, self.embed_w, out, M)

    # ------------------------------------------------------------------ the step
    def _trunk(self, M: int, advance: bool) -> None:
        E, eps = ext(), self.eps
        self._embed(self.vtok, self.h, M)
        delta = None
        for o in self.layers:
            E.rmsnorm(self.h, delta, o.in_ln, self.x, M, eps, 1)
            if o.kind == "gdn":
                o.qkvzba(self.x[:M], self.zba[:M])
                E.gdn_conv(self.zba, self.hist[o.gi], o.convw, self.pos, self.conv, M)
                if self.gdn_native_dtypes:
                    recur = gdn_recur_entrypoint(E, hv16=self.gdn_hv16, native=True)
                    recur(self.conv, self.zba, o.z_off, o.b_off, o.a_off, o.A_log, o.dt_bias, o.nw,
                          self.state[o.gi], self.R, self.pos, self.go, M, eps, o.nkh_rep,
                          *o.gdn_dtype_codes)
                else:
                    recur = gdn_recur_entrypoint(E, hv16=self.gdn_hv16, native=False)
                    recur(self.conv, self.zba, o.z_off, o.b_off, o.a_off, o.A_log, o.dt_bias, o.nw,
                          self.state[o.gi], self.R, self.pos, self.go, M, eps, o.nkh_rep)
                o.out(self.go[:M], self.proj[:M])
            else:
                o.qkv(self.x[:M], self.aqkv[:M])
                E.attn_prep(self.aqkv, o.qnw, o.knw, self.cosT, self.sinT, self.pos, self.ropeoff,
                            self.q, self.kc[o.ai], self.vc[o.ai], M, o.NQ, o.NKV, eps)
                E.attn_split(self.q, self.kc[o.ai], self.vc[o.ai], self.pos, self.pacc, self.pml, M,
                             o.NQ, o.NKV, self.NS, o.scaling)
                E.attn_combine(self.pacc, self.pml, self.aqkv, self.pos, self.ao, M, o.NQ, self.NS)
                o.o(self.ao[:M], self.proj[:M])
            E.rmsnorm(self.h, self.proj, o.post_ln, self.x, M, eps, 1)
            o.gu(self.x[:M], self.gu[:M])
            E.silu_mul(self.gu, self.act, M)
            o.down(self.act[:M], self.dn[:M])
            delta = self.dn
        E.rmsnorm(self.h, delta, self.final_ln, self.xf, M, eps, 1)
        self.lm(self.xf[:M], self.logits[:M])
        E.argmax_rows(self.logits, self.am, M)
        if advance:
            E.ar_advance(self.am, self.vtok, self.log, self.pos)

    def _mtp_step(self, M: int) -> None:
        E, eps, m = ext(), self.eps, self.m
        self._embed(self.mtok, self.me, M)
        E.rmsnorm(self.me, None, m.pre_e, self.mcat[:, :self.H], M, eps, 1)
        E.rmsnorm(self.mhid, None, m.pre_h, self.mcat[:, self.H:], M, eps, 1)
        m.fc(self.mcat[:M], self.mh[:M])
        E.rmsnorm(self.mh, None, m.in_ln, self.mx, M, eps, 1)
        m.qkv(self.mx[:M], self.aqkv[:M])
        E.attn_prep(self.aqkv, m.qnw, m.knw, self.cosT, self.sinT, self.mkv, self.mropeoff, self.q,
                    self.mkc, self.mvc, M, self.NQ, self.NKV, eps)
        E.attn_split(self.q, self.mkc, self.mvc, self.mkv, self.pacc, self.pml, M, self.NQ, self.NKV,
                     self.NS, m.scaling)
        E.attn_combine(self.pacc, self.pml, self.aqkv, self.mkv, self.ao, M, self.NQ, self.NS)
        m.o(self.ao[:M], self.proj[:M])
        E.rmsnorm(self.mh, self.proj, m.post_ln, self.mx, M, eps, 1)
        m.gu(self.mx[:M], self.gu[:M])
        E.silu_mul(self.gu, self.act, M)
        m.down(self.act[:M], self.dn[:M])
        E.rmsnorm(self.mh, self.dn, m.norm, self.mg, M, eps, 1)
        self.lm(self.mg[M - 1:M], self.mlogits)
        E.argmax_rows(self.mlogits, self.mam, 1)
        self.mhid[0:1].copy_(self.mg[M - 1:M])
        E.draft_advance(self.mam, self.vtok, self.slot, self.mtok, self.mkv, M)

    # ------------------------------------------------------------------ graphs
    def capture(self, kinds: Sequence[Tuple[str, int]]) -> Dict[str, float]:
        """Capture CUDA graphs for ('ar', 1), ('verify', M), ('mtp', M).  Returns capture seconds."""
        fns = {"ar": lambda M: self._trunk(M, True), "verify": lambda M: self._trunk(M, False),
               "mtp": self._mtp_step}
        out = {}
        snap = self._snap_scalars()
        s = torch.cuda.Stream(self.dev)
        s.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(s):
            for kind, M in kinds:
                if (kind, M) in self.graphs:
                    continue
                t0 = time.perf_counter()
                fns[kind](M)                      # warm-up (JIT, allocator)
                torch.cuda.synchronize(self.dev)
                self._restore_scalars(snap)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s):
                    fns[kind](M)
                torch.cuda.synchronize(self.dev)
                self._restore_scalars(snap)
                self.graphs[(kind, M)] = g
                out[f"{kind}{M}"] = round(time.perf_counter() - t0, 3)
        torch.cuda.current_stream(self.dev).wait_stream(s)
        return out

    def _snap_scalars(self):
        t = [self.pos, self.vtok]
        if self.mtp_ready:
            t += [self.mkv, self.slot, self.mtok, self.mhid]
        return [(x, x.clone()) for x in t]

    @staticmethod
    def _restore_scalars(snap):
        for x, c in snap:
            x.copy_(c)

    def replay(self, kind: str, M: int) -> None:
        g = self.graphs.get((kind, M))
        if g is None:
            {"ar": lambda: self._trunk(M, True), "verify": lambda: self._trunk(M, False),
             "mtp": lambda: self._mtp_step(M)}[kind]()
        else:
            g.replay()

    # ------------------------------------------------------------------ prefill import
    @torch.no_grad()
    def import_prefill(self, cache, plen: int, rope_delta: int, first_token: int,
                       mtp_cache=None, mtp_hidden=None, first_draft: Optional[int] = None) -> None:
        if plen + 16 > self.max_len:
            raise ValueError(f"prompt {plen} too long for max_len {self.max_len}")
        R = self.R
        for li, o in enumerate(self.layers):
            lay = cache.layers[li]
            if o.kind == "attn":
                self.kc[o.ai, :plen].copy_(lay.keys[0].transpose(0, 1))
                self.vc[o.ai, :plen].copy_(lay.values[0].transpose(0, 1))
            else:
                conv = lay.conv_states[0]            # [C, 4]: inputs at positions plen-4 .. plen-1
                w = int(conv.shape[-1])
                self.hist[o.gi].zero_()
                for j in range(min(w, plen)):
                    self.hist[o.gi, (plen - 1 - j) & 15].copy_(conv[:, w - 1 - j])
                self.state[o.gi, (plen - 1) % R].copy_(lay.recurrent_states[0].float())
        self.pos.fill_(plen)
        self.ropeoff.fill_(int(rope_delta))
        self.vtok.zero_()
        self.vtok[0] = int(first_token)
        self.log.zero_()
        self.log[plen] = int(first_token)
        if self.mtp_ready and mtp_cache is not None:
            ml = mtp_cache.layers[0]
            n = int(ml.keys.shape[2])
            self.mkc[:n].copy_(ml.keys[0].transpose(0, 1))
            self.mvc[:n].copy_(ml.values[0].transpose(0, 1))
            self.mkv.fill_(n)
            self.mropeoff.fill_(int(rope_delta) + 1)
            self.mhid[0].copy_(mtp_hidden.reshape(-1))
            self.vtok[1] = int(first_draft)
            self.mtok[0] = int(first_draft)
            self.slot.fill_(2)

    # ------------------------------------------------------------------ decode loops
    @torch.no_grad()
    def plain(self, n_tokens: int, eos: Sequence[int], *, first_token: int, first_digest=None,
              digest_fn=None, plen: int) -> Dict[str, Any]:
        """AR greedy: n_tokens INCLUDING the first (prefill) token.  Host sync per token only
        when digests are requested; otherwise graphs are chained on device."""
        eos = set(int(e) for e in eos)
        toks = [int(first_token)]
        dig = [first_digest] if digest_fn else []
        torch.cuda.synchronize(self.dev)
        t0 = time.perf_counter()
        if digest_fn is not None:
            while toks[-1] not in eos and len(toks) < n_tokens:
                self.replay("ar", 1)
                t = int(self.am[0].item())
                toks.append(t)
                dig.append(digest_fn(self.logits[0]))
        else:
            steps = n_tokens - 1
            for _ in range(steps):
                self.replay("ar", 1)
            torch.cuda.synchronize(self.dev)
            seq = self.log[plen:plen + n_tokens].tolist()
            toks = []
            for t in seq:
                toks.append(int(t))
                if t in eos:
                    break
        torch.cuda.synchronize(self.dev)
        steps = (len(toks) - 1) if digest_fn is not None else (n_tokens - 1)
        return {"tokens": toks, "digests": dig, "steps": steps,
                "seconds": time.perf_counter() - t0}

    @torch.no_grad()
    def speculative(self, k: int, n_tokens: int, eos: Sequence[int], *, first_token: int, plen: int,
                    first_digest=None, digest_fn=None) -> Dict[str, Any]:
        if not self.mtp_ready:
            raise RuntimeError("no MTP head")
        if not 1 <= k <= MAX_M - 1:
            raise ValueError("1 <= k <= 7")
        eos = set(int(e) for e in eos)
        toks = [int(first_token)]
        dig = [first_digest] if digest_fn else []
        st = {"k": k, "cycles": 0, "drafts_proposed": 0, "drafts_accepted": 0,
              "accept_len_hist": [0] * (k + 1)}
        pin = torch.empty(2 * MAX_M, dtype=torch.int32, pin_memory=True)
        L = plen
        done = toks[-1] in eos or len(toks) >= n_tokens
        torch.cuda.synchronize(self.dev)
        t0 = time.perf_counter()
        while not done:
            st["cycles"] += 1
            for _ in range(k - 1):
                self.replay("mtp", 1)
            self.replay("verify", k + 1)
            pin[:k + 1].copy_(self.am[:k + 1], non_blocking=True)
            pin[MAX_M:MAX_M + k].copy_(self.vtok[1:k + 1], non_blocking=True)
            torch.cuda.current_stream(self.dev).synchronize()
            outs = pin[:k + 1].tolist()
            drafts = pin[MAX_M:MAX_M + k].tolist()
            a = 0
            while a < k and drafts[a] == outs[a]:
                a += 1
            st["drafts_proposed"] += k
            st["drafts_accepted"] += a
            st["accept_len_hist"][a] += 1
            for i in range(a + 1):
                toks.append(int(outs[i]))
                if digest_fn is not None:
                    dig.append(digest_fn(self.logits[i]))
                if outs[i] in eos or len(toks) >= n_tokens:
                    done = True
                    break
            if done:
                break
            # confirm: MTP on (h_L..h_{L+a}, t_1..t_{a+1}); next x_last = t_{a+1}
            self.mhid[:a + 1].copy_(self.xf[:a + 1])
            self.mtok[:a + 1].copy_(self.am[:a + 1])
            self.vtok[0:1].copy_(self.am[a:a + 1])
            self.mkv.fill_(L)
            self.slot.fill_(1)
            self.replay("mtp", a + 1)
            L += a + 1
            self.pos.fill_(L)
        torch.cuda.synchronize(self.dev)
        n = st["drafts_proposed"]
        st["draft_acceptance"] = st["drafts_accepted"] / n if n else None
        st["mean_tokens_per_cycle"] = (st["drafts_accepted"] + st["cycles"]) / st["cycles"] \
            if st["cycles"] else None
        return {"tokens": toks, "digests": dig, "stats": st, "seconds": time.perf_counter() - t0}

    def descriptor_kinds(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for d in self._descs():
            out[d.kind] = out.get(d.kind, 0) + 1
        return out

    def _descs(self):
        ds = [self.lm]
        for o in self.layers:
            ds += [o.gu, o.down] + ([o.qkvzba, o.out] if o.kind == "gdn" else [o.qkv, o.o])
        return ds

    def streamed_bytes(self) -> int:
        """Weight bytes one trunk step reads (coded bytes for coded descriptors)."""
        return sum(getattr(d, "coded_bytes", d.bytes) for d in self._descs())

    def weight_bytes(self) -> int:
        tot = self.lm.bytes
        for o in self.layers:
            tot += o.gu.bytes + o.down.bytes
            tot += (o.qkvzba.bytes + o.out.bytes) if o.kind == "gdn" else (o.qkv.bytes + o.o.bytes)
        return tot


__all__ = ["FastDecoder", "Lin", "ext", "fuse_rows", "gdn_template_dims", "gdn_template_geometry",
           "load_tune", "validate_gdn_geometry", "validate_gdn_native_dtypes", "gdn_recur_entrypoint",
           "split_attn_gdn"]
