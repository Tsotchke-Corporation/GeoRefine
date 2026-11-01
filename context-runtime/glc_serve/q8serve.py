"""Certified Q8_0 GGUF weights served through MIV-GEMV with the dequant fused in registers.

The certified tier-(ii) artifact is the stock llama.cpp Q8_0 GGUF of
Qwen3.8-27B (SAE L1-63 0.9823, readout ``gguf_Q8_0``: every decoder linear,
the embedding and the LM head read from the SHIPPED file with gguf-py's
reference ``dequantize`` and mapped to HF layout by the heal arm's Bridge;
float tensors stay the parent's).  This module loads exactly those blocks:

* a Q8_0 block is ``{fp16 d; int8 q[32]}``; its value is ``d * q`` (exact in
  FP32), and the model runs it as ``bf16_rn(d * q)`` -- the same bits as
  ``torch.from_numpy(gguf.quants.dequantize(...)).to(torch.bfloat16)``;
* the blocks are split once into SoA arrays (``qs`` int8 [N, K], ``sc`` fp16
  [N, K/32]) -- 34 bytes per 32 weights, byte-for-byte the file's payload;
* the llama.cpp converter's value-head reorder of the GDN tensors is undone
  by permuting whole rows (qkv, z, a, b) or whole 128-column head chunks
  (out_proj) of both arrays -- a relabelling, no arithmetic.

``miv_gemv.miv_q8_out`` dequantizes each 16-byte weight vector in registers
and feeds it to the identical ``fma_vec`` chain as the bf16 kernel, so for
the same (WPR) the Q8_0 engine is bitwise the bf16 engine run on the
dequantized weights (tested by ``scripts/fastdec_run.py --q8-check``).

Default-off: nothing imports this unless asked.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import miv_gemv as mg

QK = 32


# conversion/qwen.py _LinearAttentionVReorderBase._reorder_v_heads (as in
# scripts/qwen38_27b_heal.py on feat/friday-cert-20260923, verbatim)
def reorder_v(t, dim: int, k: int, vper: int, hd: int):
    shape = list(t.shape)
    if dim < 0:
        dim += len(shape)
    new = shape[:dim] + [k, vper, hd] + shape[dim + 1:]
    t = t.reshape(*new)
    perm = list(range(len(new)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return t.permute(*perm).contiguous().reshape(*shape)


def unreorder_v(t, dim: int, k: int, vper: int, hd: int):
    return reorder_v(t, dim, vper, k, hd)


LIN = {"linear_attn.in_proj_qkv": "attn_qkv", "linear_attn.in_proj_z": "attn_gate",
       "linear_attn.in_proj_a": "ssm_alpha", "linear_attn.in_proj_b": "ssm_beta",
       "linear_attn.out_proj": "ssm_out", "self_attn.q_proj": "attn_q",
       "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
       "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate",
       "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}


class Q8Pack:
    def __init__(self, qs: torch.Tensor, sc: torch.Tensor):
        assert qs.dtype == torch.int8 and sc.dtype == torch.float16
        self.qs, self.sc = qs.contiguous(), sc.contiguous()
        self.N, self.K = int(qs.shape[0]), int(qs.shape[1])
        assert self.sc.shape == (self.N, self.K // QK)

    @property
    def bytes(self) -> int:
        return self.qs.numel() + self.sc.numel() * 2

    def dequant(self, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        if out is None:
            out = torch.empty(self.N, self.K, dtype=torch.bfloat16, device=self.qs.device)
        mg.extension().q8_dequant(self.qs, self.sc, out)
        return out


class _Scratch:
    buf: Optional[torch.Tensor] = None

    @classmethod
    def get(cls, n: int, device) -> torch.Tensor:
        if cls.buf is None or cls.buf.numel() < n:
            cls.buf = None
            cls.buf = torch.empty(n, dtype=torch.bfloat16, device=device)
        return cls.buf[:n]


class Q8Linear(nn.Module):
    """HF-side linear over a Q8Pack: M <= 8 through MIV-Q8, else dequant -> F.linear."""

    def __init__(self, pack: Q8Pack, cfg: Tuple[int, int]):
        super().__init__()
        self.pack = pack
        self.cfg = cfg
        self.in_features, self.out_features = pack.K, pack.N
        self.bias = None

    @property
    def weight(self):                 # shape/dtype/device probes only; never materialized
        return torch.empty(0, dtype=torch.bfloat16, device=self.pack.qs.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1]).to(torch.bfloat16)
        m = int(x2.shape[0])
        if 1 <= m <= 8:
            x2 = x2.contiguous()
            y = torch.empty(m, self.pack.N, dtype=torch.bfloat16, device=x.device)
            mg.extension().miv_q8_out(x2, self.pack.qs, self.pack.sc, y, self.cfg[0], self.cfg[1])
        else:
            w = self.pack.dequant(_Scratch.get(self.pack.N * self.pack.K, x.device)
                                  .view(self.pack.N, self.pack.K))
            y = F.linear(x2, w)
        return y.reshape(*lead, self.pack.N).to(x.dtype)


class Q8Embedding(nn.Module):
    def __init__(self, pack: Q8Pack, padding_idx=None):
        super().__init__()
        self.pack = pack
        self.padding_idx = padding_idx
        self.num_embeddings, self.embedding_dim = pack.N, pack.K

    @property
    def weight(self):
        return torch.empty(0, dtype=torch.bfloat16, device=self.pack.qs.device)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1)
        q = self.pack.qs.index_select(0, flat).float()
        s = self.pack.sc.index_select(0, flat).float().repeat_interleave(QK, dim=1)
        return (s * q).to(torch.bfloat16).reshape(*ids.shape, self.pack.K)


def _raw_blocks(t) -> Tuple[np.ndarray, int, int]:
    ne = [int(v) for v in t.shape]            # gguf order: ne0 = K (inner)
    k, n = ne[0], int(np.prod(ne[1:]))
    raw = np.asarray(t.data).reshape(n, k // QK, 2 + QK)
    return raw, n, k


def pack_from_gguf(t, device, fix=None) -> Q8Pack:
    from gguf import GGMLQuantizationType

    if t.tensor_type != GGMLQuantizationType.Q8_0:
        raise TypeError(f"{t.name}: {t.tensor_type.name}, not Q8_0")
    raw, n, k = _raw_blocks(t)
    sc = torch.from_numpy(np.ascontiguousarray(raw[:, :, :2]).view(np.float16).reshape(n, k // QK))
    qs = torch.from_numpy(np.ascontiguousarray(raw[:, :, 2:]).view(np.int8).reshape(n, k))
    if fix is not None:
        qs, sc = fix(qs, sc)
    return Q8Pack(qs.contiguous().to(device), sc.contiguous().to(device))


def _fixer(rest: str, cfg) -> Optional[Any]:
    K = int(cfg.linear_num_key_heads)
    vper = int(cfg.linear_num_value_heads) // K
    hk, hv = int(cfg.linear_key_head_dim), int(cfg.linear_value_head_dim)
    if rest == "linear_attn.in_proj_qkv":
        qk = hk * K * 2
        return lambda q, s: (torch.cat([q[:qk], unreorder_v(q[qk:], 0, K, vper, hv)], 0),
                             torch.cat([s[:qk], unreorder_v(s[qk:], 0, K, vper, hv)], 0))
    if rest == "linear_attn.in_proj_z":
        return lambda q, s: (unreorder_v(q, 0, K, vper, hv), unreorder_v(s, 0, K, vper, hv))
    if rest in ("linear_attn.in_proj_a", "linear_attn.in_proj_b"):
        return lambda q, s: (unreorder_v(q, 0, K, vper, 1), unreorder_v(s, 0, K, vper, 1))
    if rest == "linear_attn.out_proj":
        assert hv % QK == 0
        return lambda q, s: (unreorder_v(q, 1, K, vper, hv), unreorder_v(s, 1, K, vper, hv // QK))
    return None


def load_q8_gguf(model: nn.Module, gguf_path: str, tune: Dict, *, device, check_every: int = 0,
                 log=print) -> Dict[str, Any]:
    """Replace every decoder linear, the embedding and the LM head of ``model`` (HF, bf16)
    by the GGUF's Q8_0 blocks.  Returns a receipt.  ``check_every`` > 0: every Nth tensor
    is checked bitwise against gguf-py ``dequantize`` -> bf16 (+ inverse reorder)."""
    from gguf import GGUFReader
    from gguf.quants import dequantize

    from .fastdec import _text

    tm = _text(model)
    cfg = tm.config
    reader = GGUFReader(gguf_path)
    g = {t.name: t for t in reader.tensors}
    rec = {"gguf": gguf_path, "replaced": 0, "q8_bytes": 0, "checked": [], "check_fail": []}
    todo = []
    for li, layer in enumerate(tm.layers):
        for rest, gn in LIN.items():
            try:
                mod = layer.get_submodule(rest)
            except AttributeError:
                continue
            todo.append((layer, rest, f"blk.{li}.{gn}.weight", mod))
    todo.append((model, "lm_head", "output.weight", model.get_output_embeddings()))
    n_done = 0
    for i in range(len(todo)):
        parent, rest, gname, mod = todo[i]
        todo[i] = None                       # drop the list's reference: the bf16 weight must die
        t = g[gname]
        fix = _fixer(rest, cfg)
        pack = pack_from_gguf(t, device, fix)
        if (pack.N, pack.K) != tuple(mod.weight.shape):
            raise ValueError(f"{gname}: {pack.N}x{pack.K} vs HF {tuple(mod.weight.shape)}")
        n_done += 1
        if check_every and (n_done % check_every == 1 or gname == "output.weight"):
            ref = torch.from_numpy(np.asarray(dequantize(np.asarray(t.data), t.tensor_type),
                                              dtype=np.float32).reshape(pack.N, pack.K))
            ref = ref.to(torch.bfloat16)
            if fix is not None:
                ref, _ = fix(ref, torch.zeros(pack.N, pack.K // QK, dtype=torch.float16))
            got = pack.dequant().cpu()
            ok = torch.equal(got.view(torch.int16), ref.contiguous().view(torch.int16))
            w = mod.weight.detach().float().cpu()
            rel = float((got.float() - w).norm() / w.norm())
            rec["checked"].append({"tensor": gname, "bitwise_vs_gguf_py": ok, "rel_err_vs_parent": rel})
            if not ok:
                rec["check_fail"].append(gname)
        key = (pack.N, pack.K)
        cfgt = tune.get(key) or mg.default_config(*key)
        new = Q8Linear(pack, cfgt)
        owner = parent
        name = rest
        if "." in rest:
            head, name = rest.rsplit(".", 1)
            owner = parent.get_submodule(head)
        setattr(owner, name, new)
        rec["replaced"] += 1
        rec["q8_bytes"] += pack.bytes
        del mod
    emb_t = g["token_embd.weight"]
    ep = pack_from_gguf(emb_t, device)
    old = tm.embed_tokens
    tm.embed_tokens = Q8Embedding(ep, getattr(old, "padding_idx", None))
    del old
    rec["embed_q8_bytes"] = ep.bytes
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    rec["cuda_allocated_after"] = int(torch.cuda.memory_allocated())
    rec["cuda_reserved_after"] = int(torch.cuda.memory_reserved())
    log(f"[q8] replaced {rec['replaced']} linears + embedding; q8 bytes {rec['q8_bytes']/1e9:.3f} GB; "
        f"checked {len(rec['checked'])} fail {len(rec['check_fail'])}")
    return rec


class Q8Lin:
    """FastDecoder descriptor over a Q8Pack (fused by row concatenation)."""

    kind = "q8_0"

    def __init__(self, pack: Q8Pack, cfg: Tuple[int, int]):
        self.p = pack
        self.N, self.K = pack.N, pack.K
        self.wpr, self.unroll = int(cfg[0]), int(cfg[1])

    @property
    def bytes(self) -> int:
        return self.p.bytes

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        mg.extension().miv_q8_out(x, self.p.qs, self.p.sc, y, self.wpr, self.unroll)


def fuse_q8(mods) -> Q8Pack:
    qs = torch.cat([m.pack.qs for m in mods], 0).contiguous()
    sc = torch.cat([m.pack.sc for m in mods], 0).contiguous()
    r = 0
    for m in mods:
        n = m.pack.N
        m.pack = Q8Pack(qs[r:r + n], sc[r:r + n])
        r += n
    return Q8Pack(qs, sc)


__all__ = ["Q8Embedding", "Q8Lin", "Q8Linear", "Q8Pack", "fuse_q8", "load_q8_gguf",
           "pack_from_gguf", "unreorder_v"]


# ===========================================================================
# Mixed GGUF: per-tensor quant type -> descriptor (plug-in registry)
# ===========================================================================
# A certified mixed-precision GGUF assigns a quant type per tensor.  ``load_gguf``
# maps every decoder linear / LM head / embedding to a module chosen by that
# tensor's type:
#
#   Q8_0                 -> Q8Linear (fused in-register dequant, glc_serve.miv_gemv)
#   REGISTRY[type_name]  -> whatever a kernel lane registers (``register_gguf_type``):
#                           builder(t, rest, cfg, device, tune) -> nn.Module with
#                           .forward (HF prefill) and .fast_desc() (engine descriptor
#                           with __call__(x2d, y2d), .N, .K, .bytes)
#   anything else        -> dense bf16 from gguf-py's reference dequantize (exact
#                           decoder semantics; bf16 MIV in the engine) -- runs any file
#                           today, at bf16 bytes for those tensors.
#
# The converter's V-head reorder is undone on rows for qkv/z/a/b (row permutation is
# block-safe for every type).  For ssm_out the reorder permutes 128-wide COLUMN chunks;
# when the type's block size divides 128 (Q8_0: 32) the blocks are permuted, otherwise
# (K-quants: 256) the weight stays in GGUF column order and the module carries
# ``in_perm`` -- its input is gathered into GGUF order instead (y = W_gguf x[perm]).
REGISTRY: Dict[str, Any] = {}


def register_gguf_type(type_name: str, builder) -> None:
    """Plug in a fused kernel for a GGUF quant type (e.g. "Q4_K", "IQ4_XS")."""
    REGISTRY[str(type_name)] = builder


class DenseGGUFLinear(nn.Module):
    """bf16 weight = bf16(gguf-py dequantize(block data)); optional input permutation."""

    def __init__(self, weight: torch.Tensor, in_perm: Optional[torch.Tensor] = None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = None
        self.in_perm = in_perm
        self.out_features, self.in_features = int(weight.shape[0]), int(weight.shape[1])

    def forward(self, x):
        if self.in_perm is not None:
            x = x.index_select(-1, self.in_perm)
        return F.linear(x, self.weight)


def _col_perm(cfg, k: int, device) -> torch.Tensor:
    """perm with x_gguf = x_hf[perm] for ssm_out's input (value heads reordered)."""
    K = int(cfg.linear_num_key_heads)
    vper = int(cfg.linear_num_value_heads) // K
    hv = int(cfg.linear_value_head_dim)
    idx = torch.arange(k).reshape(1, k)
    return reorder_v(idx, 1, K, vper, hv).reshape(-1).to(device)


def load_gguf(model: nn.Module, gguf_path: str, tune: Dict, *, device, check_every: int = 0,
              log=print) -> Dict[str, Any]:
    """Load a (possibly mixed-type) GGUF's linears, LM head and embedding into ``model``."""
    from gguf import GGMLQuantizationType, GGUFReader
    from gguf.constants import GGML_QUANT_SIZES
    from gguf.quants import dequantize

    from .fastdec import _text

    tm = _text(model)
    cfg = tm.config
    reader = GGUFReader(gguf_path)
    g = {t.name: t for t in reader.tensors}
    rec: Dict[str, Any] = {"gguf": gguf_path, "by_type": {}, "bytes_by_type": {}, "checked": [],
                           "check_fail": [], "dense_fallback": [], "in_perm": []}
    todo = []
    for li, layer in enumerate(tm.layers):
        for rest, gn in LIN.items():
            try:
                layer.get_submodule(rest)
            except AttributeError:
                continue
            todo.append((layer, rest, f"blk.{li}.{gn}.weight"))
    todo.append((model, "lm_head", "output.weight"))
    for n_done, (parent, rest, gname) in enumerate(todo, 1):
        t = g[gname]
        tname = t.tensor_type.name
        mod = parent.get_submodule(rest)
        shape = tuple(int(v) for v in mod.weight.shape) if hasattr(mod, "weight") else None
        blk = GGML_QUANT_SIZES[t.tensor_type][0]
        if t.tensor_type == GGMLQuantizationType.Q8_0:
            pack = pack_from_gguf(t, device, _fixer(rest, cfg))
            new = Q8Linear(pack, tune.get((pack.N, pack.K)) or mg.default_config(pack.N, pack.K))
            nbytes = pack.bytes
        elif tname in REGISTRY:
            new = REGISTRY[tname](t, rest, cfg, device, tune)
            nbytes = int(getattr(new, "bytes", 0))
        else:
            raw = np.asarray(t.data)
            w = torch.from_numpy(np.asarray(dequantize(raw, t.tensor_type), dtype=np.float32))
            n, k = shape
            w = w.reshape(n, k).to(torch.bfloat16)
            in_perm = None
            if rest == "linear_attn.out_proj":
                in_perm = _col_perm(cfg, k, device)       # keep GGUF column order
                rec["in_perm"].append(gname)
            else:
                fix = _fixer(rest, cfg)
                if fix is not None:
                    w, _ = fix(w, torch.zeros(n, max(1, k // QK), dtype=torch.float16))
            new = DenseGGUFLinear(w.contiguous().to(device), in_perm)
            nbytes = n * k * 2
            rec["dense_fallback"].append(f"{gname}:{tname}")
        if shape is not None and (new.out_features, new.in_features) != shape:
            raise ValueError(f"{gname}: {new.out_features}x{new.in_features} vs HF {shape}")
        if check_every and n_done % check_every == 1 and isinstance(new, Q8Linear):
            ref = torch.from_numpy(np.asarray(dequantize(np.asarray(t.data), t.tensor_type),
                                              dtype=np.float32).reshape(new.pack.N, new.pack.K))
            ref = ref.to(torch.bfloat16)
            fix = _fixer(rest, cfg)
            if fix is not None:
                ref, _ = fix(ref, torch.zeros(new.pack.N, new.pack.K // QK, dtype=torch.float16))
            ok = torch.equal(new.pack.dequant().cpu().view(torch.int16),
                             ref.contiguous().view(torch.int16))
            rec["checked"].append({"tensor": gname, "bitwise_vs_gguf_py": ok})
            if not ok:
                rec["check_fail"].append(gname)
        owner, name = parent, rest
        if "." in rest:
            head, name = rest.rsplit(".", 1)
            owner = parent.get_submodule(head)
        setattr(owner, name, new)
        del mod
        rec["by_type"][tname] = rec["by_type"].get(tname, 0) + 1
        rec["bytes_by_type"][tname] = rec["bytes_by_type"].get(tname, 0) + nbytes
        _ = blk
    et = g["token_embd.weight"]
    old = tm.embed_tokens
    if et.tensor_type == GGMLQuantizationType.Q8_0:
        tm.embed_tokens = Q8Embedding(pack_from_gguf(et, device), getattr(old, "padding_idx", None))
    else:
        w = torch.from_numpy(np.asarray(dequantize(np.asarray(et.data), et.tensor_type),
                                        dtype=np.float32)).reshape(old.num_embeddings, -1)
        emb = nn.Embedding(old.num_embeddings, w.shape[1], padding_idx=old.padding_idx,
                           device="meta")
        emb.weight = nn.Parameter(w.to(torch.bfloat16).to(device), requires_grad=False)
        tm.embed_tokens = emb
        rec["dense_fallback"].append(f"token_embd.weight:{et.tensor_type.name}")
    rec["embed_type"] = et.tensor_type.name
    del old
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    rec["cuda_allocated_after"] = int(torch.cuda.memory_allocated())
    log(f"[gguf] {rec['by_type']} dense_fallback={len(rec['dense_fallback'])} "
        f"in_perm={len(rec['in_perm'])} checked={len(rec['checked'])} fail={len(rec['check_fail'])}")
    return rec


# ---------------------------------------------------------------- engine-side composition
class GroupLin:
    """A fused row group whose members use DIFFERENT formats: each member writes its own
    column slice of y (every kernel honours y's row stride)."""

    kind = "group"

    def __init__(self, members):
        self.members = list(members)            # [(desc, n)]
        self.N = sum(n for _, n in self.members)
        self.K = self.members[0][0].K

    @property
    def bytes(self) -> int:
        return sum(d.bytes for d, _ in self.members)

    @property
    def coded_bytes(self) -> int:
        return sum(getattr(d, "coded_bytes", d.bytes) for d, _ in self.members)

    def __call__(self, x, y):
        r = 0
        for d, n in self.members:
            d(x, y[:, r:r + n])
            r += n


class PermIn:
    """Gather the input columns into the weight's (GGUF) order, then run the descriptor."""

    kind = "perm"

    def __init__(self, desc, perm: torch.Tensor, max_m: int = 8):
        self.d, self.perm = desc, perm
        self.N, self.K = desc.N, desc.K
        self.buf = torch.empty(max_m, desc.K, dtype=torch.bfloat16, device=perm.device)

    @property
    def bytes(self) -> int:
        return self.d.bytes

    @property
    def coded_bytes(self) -> int:
        return getattr(self.d, "coded_bytes", self.d.bytes)

    def __call__(self, x, y):
        m = int(x.shape[0])
        torch.index_select(x, 1, self.perm, out=self.buf[:m])
        self.d(self.buf[:m], y)


__all__ += ["DenseGGUFLinear", "GroupLin", "PermIn", "REGISTRY", "load_gguf",
            "register_gguf_type"]
