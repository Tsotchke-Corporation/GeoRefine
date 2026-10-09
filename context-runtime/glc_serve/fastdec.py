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
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from . import miv_gemv as mg

_EXT = None
_LOCK = threading.Lock()
_FUNCS = ["embed", "embed_q8", "rmsnorm", "gdn_conv", "gdn_recur", "attn_prep", "attn_split", "attn_combine",
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
    pass


class FastDecoder:
    def __init__(self, model: nn.Module, mtp: Optional[nn.Module], *, tune: Dict, max_len: int = 4096,
                 R: int = 16, device="cuda:0", lin_factory=None, fuse: bool = True,
                 tbe_unroll: int = 0):
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
        self.shapes_used: Dict[str, List[int]] = {}
        dev = self.dev
        bf = torch.bfloat16

        def one(m, name):
            """Descriptor for a single module of any supported format."""
            if hasattr(m, "fast_desc"):                       # registered GGUF kernel
                d = m.fast_desc()
            elif hasattr(m, "device_container") or hasattr(m, "pack"):
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
            if hasattr(m, "device_container"):
                return "tbe"
            if hasattr(m, "pack"):
                return "q8"
            if hasattr(m, "fast_desc") or getattr(m, "in_perm", None) is not None:
                return "other"
            return "bf16"

        def L(mods, name):
            kinds = {kind_of(m) for m in mods}
            if len(mods) > 1 and (len(kinds) > 1 or "other" in kinds) and not kinds <= {"tbe", "bf16"}:
                from .q8serve import GroupLin                  # mixed formats in one fused group

                g = GroupLin([(one(m, name), int(m.out_features)) for m in mods])
                self.shapes_used[f"{g.N}x{g.K}:group"] = [k for k in sorted(kinds)]
                return g
            if len(mods) == 1 and kinds == {"other"}:
                return one(mods[0], name)
            if any(hasattr(m, "device_container") for m in mods):   # TBE codec (glc_serve.tbe_desc)
                from .tbe_desc import TBELin

                key = (sum(int(m.out_features) for m in mods), int(mods[0].in_features))
                if key not in self.tune:
                    raise KeyError(f"no tuned (WPR, unroll) for shape {key} ({name})")
                wpr, u = self.tune[key]
                cfgt = (wpr, self.tbe_unroll or min(int(u), 4))
                self.shapes_used[f"{key[0]}x{key[1]}"] = list(cfgt)
                return TBELin(mods, cfgt)
            if hasattr(mods[0], "pack"):                      # Q8_0 (glc_serve.q8serve)
                from .q8serve import Q8Lin, fuse_q8

                pk = fuse_q8(mods) if len(mods) > 1 else mods[0].pack
                key = (pk.N, pk.K)
                if key not in self.tune:
                    raise KeyError(f"no tuned (WPR, unroll) for shape {key} ({name})")
                self.shapes_used[f"{key[0]}x{key[1]}"] = list(self.tune[key])
                return Q8Lin(pk, self.tune[key])
            w = fuse_rows(mods) if (fuse and len(mods) > 1) else mods[0].weight.detach()
            if len(mods) == 1 and not w.is_contiguous():
                w = w.contiguous()
            key = (int(w.shape[0]), int(w.shape[1]))
            if key not in self.tune:
                raise KeyError(f"no tuned (WPR, unroll) for shape {key} ({name}); run fastdec tune")
            self.shapes_used[f"{key[0]}x{key[1]}"] = list(self.tune[key])
            return self.lin_factory(w, self.tune[key])

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
                o.z_off = int(la.conv_dim)
                o.b_off = o.z_off + int(la.value_dim)
                o.a_off = o.b_off + int(la.num_v_heads)
                o.nkh_rep = int(la.num_v_heads) // int(la.num_k_heads)
                o.vdim = int(la.value_dim)
                assert (la.head_k_dim, la.head_v_dim, la.num_v_heads, la.num_k_heads) == (128, 128, 48, 16)
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
        self.n_gdn, self.n_att = n_gdn, n_att
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
        self.state = z(n_gdn, self.R, 48, 128, 128, dt=torch.float32)
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
                E.gdn_recur(self.conv, self.zba, o.z_off, o.b_off, o.a_off, o.A_log, o.dt_bias, o.nw,
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


__all__ = ["FastDecoder", "Lin", "ext", "fuse_rows", "load_tune"]
