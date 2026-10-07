"""The checkpoint's own MTP head, and self-speculative greedy decoding with it.

Qwen3.5/3.8 checkpoints ship one multi-token-prediction block (``mtp.*``, 15
tensors, 0.849 GB bf16 on the 27B) that transformers does not build
(``_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]``).  Its wiring, verified
against llama.cpp ``src/models/qwen35.cpp`` ``graph_mtp`` at f46bc30 and
vLLM's Qwen3-Next MTP:

    e = pre_fc_norm_embedding(embed(x_{p+1}))
    h = pre_fc_norm_hidden(h_p)            # h_p = trunk output AFTER final norm
    y = fc(concat(e, h))                   # [2H] -> [H], embeddings first
    y = full-attention decoder layer(y)    # its own KV cache
    logits = lm_head(norm(y))              # shared head

and the pair ``(h_p, x_{p+1})`` sits at position ``p + 1`` (llama.cpp
``common/speculative.cpp``: "pair (h_p, x_{p+1}) at MTP pos p+1").

SPECULATION IS LOSSLESS BY CONSTRUCTION.  The head only PROPOSES the token
after next; the full model verifies it.  A draft is kept only when it equals
the full model's own argmax; on a rejection the hybrid cache is rolled back
(Gated-DeltaNet conv/recurrent states restored from a snapshot, attention KV
cropped) and the rejected position is recomputed with a single-token step, so
every emitted token is the full model's greedy choice.  What speculation can
change is only numerics at bf16 near-ties: an accepted token was argmaxed out
of a 2-token verify step rather than a 1-token step, and those kernels round
differently.  The engine reports the measured token agreement against plain
greedy; it never assumes it.

Greedy only (temperature 0), batch 1.  Sampling needs rejection sampling
against the draft distribution, which this module does not implement.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from functools import wraps
import hashlib
import json
import time
from types import MethodType
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cached_gdn_chunk(module, hidden, cache, mask, kwargs):
    """Run GDN projections once and update its existing conv/recurrent states."""
    if mask is not None:
        if not isinstance(mask, torch.Tensor) or mask.ndim != 2 or tuple(mask.shape) != tuple(hidden.shape[:2]):
            raise RuntimeError("cached GDN chunk mode supports only [batch, sequence] masks")
        if not bool(torch.all(mask == 1)):
            raise RuntimeError("cached GDN chunk mode cannot skip masked/padded tokens safely")
    required = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "conv1d",
                "chunk_gated_delta_rule", "norm", "out_proj")
    if any(not callable(getattr(module, name, None)) for name in required):
        raise RuntimeError("GDN module lacks the projections/kernels required for chunk mode")
    layer_idx = getattr(module, "layer_idx", None)
    layers = getattr(cache, "layers", None)
    if type(layer_idx) is not int or layers is None or not 0 <= layer_idx < len(layers):
        raise RuntimeError("GDN chunk mode cannot locate the cache layer")
    layer = layers[layer_idx]
    conv_state = getattr(layer, "conv_states", None)
    recurrent_state = getattr(layer, "recurrent_states", None)
    if not isinstance(conv_state, torch.Tensor) or not isinstance(recurrent_state, torch.Tensor):
        raise RuntimeError("GDN chunk mode requires tensor convolution and recurrent cache states")
    batch, seq_len, _ = hidden.shape
    if conv_state.shape[0] != batch or conv_state.device != hidden.device:
        raise RuntimeError("GDN chunk cache geometry/device does not match hidden states")
    qkv = module.in_proj_qkv(hidden).transpose(1, 2)
    z = module.in_proj_z(hidden).reshape(batch, seq_len, -1, module.head_v_dim)
    b = module.in_proj_b(hidden)
    a = module.in_proj_a(hidden)
    kernel = int(module.conv_kernel_size)
    if conv_state.shape[-1] != kernel:
        raise RuntimeError("GDN convolution cache width differs from configured kernel size")
    context = conv_state[:, :, -(kernel - 1):] if kernel > 1 else conv_state[:, :, :0]
    joined = torch.cat((context, qkv), dim=-1)
    conv_fn = getattr(module, "causal_conv1d_fn", None)
    if conv_fn is None:
        conv = module.conv1d
        if conv.stride != (1,) or conv.dilation != (1,):
            raise RuntimeError("cached GDN chunk mode requires stride-one dilation-one convolution")
        mixed = F.conv1d(joined, conv.weight, conv.bias, stride=1, padding=0,
                         dilation=1, groups=conv.groups)
        if mixed.shape[-1] != seq_len:
            raise RuntimeError("cached GDN valid convolution did not return one output per token")
        mixed = F.silu(mixed)
    else:
        mixed = conv_fn(x=joined, weight=module.conv1d.weight.squeeze(1),
                        bias=module.conv1d.bias, activation=module.activation,
                        seq_idx=kwargs.get("seq_idx"))
        if mixed.shape[-1] != seq_len:
            raise RuntimeError("cached GDN causal convolution did not return one output per token")
    mixed = mixed.transpose(1, 2)
    query, key, value = torch.split(mixed, (module.key_dim, module.key_dim, module.value_dim), dim=-1)
    query = query.reshape(batch, seq_len, -1, module.head_k_dim)
    key = key.reshape(batch, seq_len, -1, module.head_k_dim)
    value = value.reshape(batch, seq_len, -1, module.head_v_dim)
    beta = b.sigmoid()
    g = -module.A_log.float().exp() * F.softplus(a.float() + module.dt_bias)
    if module.num_v_heads // module.num_k_heads > 1:
        repeat = module.num_v_heads // module.num_k_heads
        query, key = query.repeat_interleave(repeat, dim=2), key.repeat_interleave(repeat, dim=2)
    core, last_state = module.chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta, initial_state=recurrent_state,
        output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    if not isinstance(core, torch.Tensor) or not isinstance(last_state, torch.Tensor):
        raise RuntimeError("GDN chunk kernel returned unsupported state/output")
    cache.update_conv_state(F.pad(joined, (kernel - joined.shape[-1], 0)), layer_idx)
    cache.update_recurrent_state(last_state, layer_idx)
    core = core.reshape(-1, module.head_v_dim)
    z = z.reshape(-1, module.head_v_dim)
    return module.out_proj(module.norm(core, z).reshape(batch, seq_len, -1))


@contextmanager
def sequential_qwen35_gdn_cache(model: nn.Module, *, mode: str = "sequential"):
    """Preserve prior Gated-DeltaNet state for cached multi-token forwards.

    Transformers releases whose Qwen3.5 GDN resets state on cached chunks get
    a per-token fallback for that narrow case. Initial prefill and one-token
    decode keep the model's original implementation. The patch is instance
    scoped, restored even after exceptions, and rejects masks/kwargs whose
    sequence semantics this fallback cannot preserve safely. Sequential work
    is intentionally correctness-first and can be slow for long chunks.
    ``mode='chunk'`` uses the cached convolution and chunk recurrence kernels.
    """
    if mode not in {"sequential", "chunk"}:
        raise ValueError("GDN cache mode must be 'sequential' or 'chunk'")
    sentinel = object()
    patched = []

    def previous_state(cache, layer_idx):
        layers = getattr(cache, "layers", None)
        layer = layers[layer_idx] if layers is not None and 0 <= layer_idx < len(layers) else None
        if layer is not None and hasattr(layer, "has_previous_state"):
            return bool(layer.has_previous_state)
        method = getattr(cache, "has_previous_state", None)
        if callable(method):
            return bool(method(layer_idx))
        raise RuntimeError("Qwen3.5 GDN cache has no inspectable previous-state flag")

    def call_parts(args, kwargs):
        # Qwen3_5GatedDeltaNet's public contract is hidden_states,
        # cache_params, attention_mask, then optional named runtime kwargs.
        args = list(args)
        if args:
            hidden = args[0]
            hidden_set = lambda value: args.__setitem__(0, value)
        elif "hidden_states" in kwargs:
            hidden = kwargs["hidden_states"]
            hidden_set = lambda value: kwargs.__setitem__("hidden_states", value)
        else:
            raise RuntimeError("GDN forward omitted hidden_states")
        if len(args) > 1:
            cache = args[1]
        else:
            cache = kwargs.get("cache_params", kwargs.get("past_key_values"))
        if len(args) > 2:
            mask = args[2]
            mask_set = lambda value: args.__setitem__(2, value)
        else:
            mask = kwargs.get("attention_mask")
            mask_set = lambda value: kwargs.__setitem__("attention_mask", value)
        return args, kwargs, hidden, hidden_set, cache, mask, mask_set

    try:
        for module in model.modules():
            if type(module).__name__ != "Qwen3_5GatedDeltaNet":
                continue
            original_instance_forward = module.__dict__.get("forward", sentinel)
            original = module.forward

            def wrapped(this, *args, __original=original, **kwargs):
                args_list, kw, hidden, set_hidden, cache, mask, set_mask = call_parts(args, kwargs)
                if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                    raise RuntimeError("GDN cached fallback expects [batch, sequence, hidden] tensors")
                seq_len = hidden.shape[1]
                if cache is None or seq_len <= 1:
                    return __original(*args_list, **kw)
                layer_idx = getattr(this, "layer_idx", None)
                if type(layer_idx) is not int or not previous_state(cache, layer_idx) or seq_len <= 1:
                    return __original(*args_list, **kw)
                if mask is not None:
                    if not isinstance(mask, torch.Tensor) or mask.ndim != 2 or tuple(mask.shape) != tuple(hidden.shape[:2]):
                        raise RuntimeError("cached GDN sequential fallback supports only [batch, sequence] attention masks")
                    if not bool(torch.all(mask == 1)):
                        raise RuntimeError("cached GDN sequential fallback cannot skip masked/padded tokens safely")
                sequence_kwargs = {"seq_idx", "position_ids", "cache_position"}
                unknown = set(kw) - {"hidden_states", "cache_params", "past_key_values", "attention_mask"} - sequence_kwargs
                if unknown:
                    raise RuntimeError(f"cached GDN fallback does not understand kwargs: {sorted(unknown)}")
                for key in sequence_kwargs & set(kw):
                    value = kw[key]
                    if not isinstance(value, torch.Tensor) or value.ndim not in (1, 2) or value.shape[-1] != seq_len:
                        raise RuntimeError(f"cached GDN fallback cannot safely slice {key}")
                if mode == "chunk":
                    return _cached_gdn_chunk(this, hidden, cache, mask, kw)
                outputs = []
                for i in range(seq_len):
                    one_args, one_kwargs = list(args_list), dict(kw)
                    one_hidden = hidden[:, i:i + 1, :]
                    if args:
                        one_args[0] = one_hidden
                    else:
                        one_kwargs["hidden_states"] = one_hidden
                    if mask is not None:
                        one_mask = mask[:, i:i + 1]
                        if len(args) > 2:
                            one_args[2] = one_mask
                        else:
                            one_kwargs["attention_mask"] = one_mask
                    for key in sequence_kwargs & set(one_kwargs):
                        one_kwargs[key] = one_kwargs[key][..., i:i + 1]
                    output = __original(*one_args, **one_kwargs)
                    if not isinstance(output, torch.Tensor) or output.ndim < 2 or output.shape[1] != 1:
                        raise RuntimeError("GDN per-token forward returned an unsupported output")
                    outputs.append(output)
                return torch.cat(outputs, dim=1)

            module.forward = MethodType(wrapped, module)
            patched.append((module, original_instance_forward))
        yield model
    finally:
        for module, old in reversed(patched):
            if old is sentinel:
                del module.__dict__["forward"]
            else:
                module.forward = old


def _scope_gdn_cache(fn):
    @wraps(fn)
    def scoped(self, *args, **kwargs):
        with sequential_qwen35_gdn_cache(self.model):
            return fn(self, *args, **kwargs)
    return scoped


MTP_SHAPES = {
    "mtp.fc.weight": (5120, 10240),
    "mtp.layers.0.input_layernorm.weight": (5120,),
    "mtp.layers.0.mlp.down_proj.weight": (5120, 17408),
    "mtp.layers.0.mlp.gate_proj.weight": (17408, 5120),
    "mtp.layers.0.mlp.up_proj.weight": (17408, 5120),
    "mtp.layers.0.post_attention_layernorm.weight": (5120,),
    "mtp.layers.0.self_attn.k_norm.weight": (256,),
    "mtp.layers.0.self_attn.k_proj.weight": (1024, 5120),
    "mtp.layers.0.self_attn.o_proj.weight": (5120, 6144),
    "mtp.layers.0.self_attn.q_norm.weight": (256,),
    "mtp.layers.0.self_attn.q_proj.weight": (12288, 5120),
    "mtp.layers.0.self_attn.v_proj.weight": (1024, 5120),
    "mtp.norm.weight": (5120,),
    "mtp.pre_fc_norm_embedding.weight": (5120,),
    "mtp.pre_fc_norm_hidden.weight": (5120,),
}
MTP_LINEAR_PATHS = {
    "mtp.fc.weight": "fc",
    "mtp.layers.0.self_attn.q_proj.weight": "layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.k_proj.weight": "layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.v_proj.weight": "layers.0.self_attn.v_proj",
    "mtp.layers.0.self_attn.o_proj.weight": "layers.0.self_attn.o_proj",
    "mtp.layers.0.mlp.gate_proj.weight": "layers.0.mlp.gate_proj",
    "mtp.layers.0.mlp.up_proj.weight": "layers.0.mlp.up_proj",
    "mtp.layers.0.mlp.down_proj.weight": "layers.0.mlp.down_proj",
}


def _meta_mtp(text_config):
    # Avoid allocating a temporary full dense MTP head while descriptors load.
    with torch.device("meta"):
        return QwenMTP(text_config).eval()


class BCTXLinear(nn.Module):
    """BCTX-backed MTP linear; dense BF16 weights exist only for this call."""

    def __init__(self, tensor, name: str):
        super().__init__()
        self.tensor = tensor
        self.name = str(name)
        self.calls = 0

    def forward(self, x):
        weight = self.tensor.decode()
        self.calls += 1
        if tuple(weight.shape) != tuple(self.tensor.shape):
            raise ValueError(f"decoded shape changed for {self.name}")
        if weight.dtype != x.dtype:
            weight = weight.to(dtype=x.dtype)
        return F.linear(x, weight)


def _set_parameter(module: nn.Module, attr: str, value: torch.Tensor) -> None:
    old = getattr(module, attr)
    p = nn.Parameter(value.detach(), requires_grad=False)
    if attr in module._parameters:
        module._parameters[attr] = p
    else:
        setattr(module, attr, p)


def _frame_geometry(shape):
    """BCTX stores matrices; rank-one source vectors use a single-row frame."""
    shape = tuple(int(dim) for dim in shape)
    if len(shape) == 1:
        return (1, shape[0])
    if len(shape) == 2:
        return shape
    raise ValueError(f"unsupported MTP tensor rank {len(shape)} for shape {shape}")


def _safe_frame_path(frame_root: Path, relative_name: str) -> Path:
    root = frame_root.resolve()
    candidate = (root / relative_name).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"BCTX frame path escapes frame_root: {relative_name!r}") from exc
    return candidate


def load_bctx_mtp(manifest_path, frame_root, text_config, *, device="cuda:0",
                  tensor_cls=None, checkpoint_stride=1024):
    """Load all 15 original BCTX MTP frames; never reads restored checkpoint weights.

    ``frame_root`` contains the BCTX paths named by the manifest. Linear matrices
    remain indexed on-device and are decoded transiently by BCTXLinear. Seven
    norm scales are decoded once and recorded as resident dense scale bytes.
    """
    if tensor_cls is None:
        try:
            from scripts.bitexact_context_gpu import CudaContextTensor
        except ImportError:
            from bitexact_context_gpu import CudaContextTensor
        tensor_cls = CudaContextTensor
    manifest_path = Path(manifest_path)
    frame_root = Path(frame_root)
    manifest = json.loads(manifest_path.read_text())
    entries = {t["name"]: t for t in manifest.get("tensors", []) if t.get("name", "").startswith("mtp.")}
    if set(entries) != set(MTP_SHAPES):
        raise ValueError(f"MTP manifest keys differ: missing={sorted(set(MTP_SHAPES)-set(entries))}, extra={sorted(set(entries)-set(MTP_SHAPES))}")
    mtp = _meta_mtp(text_config)
    descriptors, dense_scales = {}, {}
    for name, expected_shape in MTP_SHAPES.items():
        entry = entries[name]
        if tuple(entry.get("shape", ())) != expected_shape or entry.get("dtype") != "BF16":
            raise ValueError(f"manifest shape/dtype mismatch for {name}")
        frame_path = _safe_frame_path(frame_root, entry["frame"])
        frame = frame_path.read_bytes()
        if len(frame) != int(entry["frame_bytes"]) or hashlib.sha256(frame).hexdigest() != entry["frame_sha256"]:
            raise ValueError(f"BCTX frame integrity mismatch for {name}")
        tensor = tensor_cls(frame, stride=checkpoint_stride, device=device)
        frame_shape = _frame_geometry(expected_shape)
        if tuple(tensor.shape) != frame_shape or tensor.source_sha256 != entry["source_sha256"]:
            raise ValueError(f"BCTX tensor identity mismatch for {name}")
        descriptors[name] = tensor
        if name in MTP_LINEAR_PATHS:
            parent_path, _, attr = MTP_LINEAR_PATHS[name].rpartition(".")
            parent = mtp.get_submodule(parent_path) if parent_path else mtp
            dense_module = getattr(parent, attr)
            if tuple(dense_module.weight.shape) != expected_shape or dense_module.bias is not None:
                raise ValueError(f"QwenMTP linear skeleton incompatible with {name}")
            setattr(parent, attr, BCTXLinear(tensor, name))
        else:
            value = tensor.decode()
            if tuple(value.shape) != frame_shape or value.dtype != torch.bfloat16:
                raise ValueError(f"decoded scale mismatch for {name}")
            # CUDA decoding returns a stride-padded view. Own an exact-size dense
            # vector so the scale residency receipt does not count its backing block.
            value = value.reshape(expected_shape).contiguous().clone()
            module_path, _, attr = name.removeprefix("mtp.").rpartition(".")
            parent = mtp.get_submodule(module_path) if module_path else mtp
            _set_parameter(parent, attr, value)
            dense_scales[name] = value.untyped_storage().nbytes()
    mtp.to(device)
    if any(p.is_meta for p in mtp.parameters()):
        raise RuntimeError("incomplete BCTX MTP skeleton contains meta parameters")
    receipt = {
        "format": "BCTX",
        "manifest": str(manifest_path),
        "tensor_count": len(descriptors),
        "tensor_names": sorted(descriptors),
        "linear_names": sorted(MTP_LINEAR_PATHS),
        "scale_names": sorted(dense_scales),
        "linear_resident_bytes": sum(d.resident_bytes for n, d in descriptors.items() if n in MTP_LINEAR_PATHS),
        "scale_resident_bytes": sum(dense_scales.values()),
        "total_resident_bytes": sum(d.resident_bytes for d in descriptors.values()) + sum(dense_scales.values()),
        "frame_sha256": {n: d.frame_sha256 for n, d in descriptors.items()},
        "source_sha256": {n: d.source_sha256 for n, d in descriptors.items()},
        "linear_calls": {n: 0 for n in MTP_LINEAR_PATHS},
        "loaded_from_bctx_only": True,
    }
    mtp._bctx_descriptors = descriptors
    mtp._bctx_receipt = receipt
    return mtp


def bctx_mtp_receipt(mtp, *, require_consumed=False):
    """Return a strict receipt; optionally fail unless every matrix ran."""
    receipt = dict(mtp._bctx_receipt)
    calls = {name: int(mtp.get_submodule(path).calls)
             for name, path in MTP_LINEAR_PATHS.items()}
    receipt["linear_calls"] = calls
    receipt["consumed_linear_count"] = sum(v > 0 for v in calls.values())
    if require_consumed and any(v == 0 for v in calls.values()):
        missing = sorted(n for n, v in calls.items() if v == 0)
        raise RuntimeError(f"BCTX MTP linears not consumed: {missing}")
    return receipt


def load_dense_mtp_control(checkpoint_dir, text_config, *, device="cuda:0"):
    """Load an independently archived dense BF16 MTP control from safetensors."""
    from safetensors import safe_open

    root = Path(checkpoint_dir)
    index = root / "model.safetensors.index.json"
    shards = {}
    if index.is_file():
        weight_map = json.loads(index.read_text()).get("weight_map", {})
        for key, filename in weight_map.items():
            if key.startswith("mtp."):
                shards.setdefault(filename, []).append(key)
    else:
        for path in sorted(root.glob("*.safetensors")):
            with safe_open(str(path), framework="pt", device="cpu") as f:
                keys = [key for key in f.keys() if key.startswith("mtp.")]
            if keys:
                shards[path.name] = keys
    present = {key for keys in shards.values() for key in keys}
    if present != set(MTP_SHAPES):
        raise ValueError(f"dense MTP control key mismatch: missing={sorted(set(MTP_SHAPES)-present)}, extra={sorted(present-set(MTP_SHAPES))}")
    mtp = _meta_mtp(text_config)
    digests = {}
    for filename, keys in shards.items():
        with safe_open(str(root / filename), framework="pt", device="cpu") as f:
            for name in keys:
                value = f.get_tensor(name)
                if tuple(value.shape) != MTP_SHAPES[name] or value.dtype != torch.bfloat16:
                    raise ValueError(f"dense BF16 control shape/dtype mismatch for {name}")
                raw = value.contiguous().view(torch.uint16).numpy().tobytes()
                digests[name] = hashlib.sha256(raw).hexdigest()
                module_path, _, attr = name.removeprefix("mtp.").rpartition(".")
                parent = mtp.get_submodule(module_path) if module_path else mtp
                _set_parameter(parent, attr, value.to(device))
    mtp.to(device)
    if any(p.is_meta for p in mtp.parameters()):
        raise RuntimeError("incomplete dense MTP control contains meta parameters")
    mtp._dense_control_receipt = {
        "format": "dense-BF16-control",
        "checkpoint_dir": str(root),
        "tensor_count": len(digests),
        "tensor_names": sorted(digests),
        "source_sha256": digests,
        "control_only": True,
    }
    return mtp


class QwenMTP(nn.Module):
    def __init__(self, text_config):
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer,
            Qwen3_5RMSNorm,
        )

        cfg = copy.deepcopy(text_config)
        cfg.layer_types = ["full_attention"]
        cfg.num_hidden_layers = 1
        h = int(cfg.hidden_size)
        eps = float(cfg.rms_norm_eps)
        self.config = cfg
        self.fc = nn.Linear(2 * h, h, bias=False)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(h, eps=eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(h, eps=eps)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(cfg, 0)])
        self.norm = Qwen3_5RMSNorm(h, eps=eps)

    def forward(self, hidden: torch.Tensor, embeds: torch.Tensor,
                position_embeddings, attention_mask, cache) -> torch.Tensor:
        e = self.pre_fc_norm_embedding(embeds)
        h = self.pre_fc_norm_hidden(hidden.to(e.dtype))
        y = self.fc(torch.cat([e, h], dim=-1))
        y = self.layers[0](y, position_embeddings=position_embeddings,
                           attention_mask=attention_mask, past_key_values=cache,
                           use_cache=True)
        if isinstance(y, tuple):
            y = y[0]
        return self.norm(y)


# ---------------------------------------------------------------------------
# model plumbing
# ---------------------------------------------------------------------------
def _first_submodule(model: nn.Module, paths: Sequence[str]) -> nn.Module:
    for p in paths:
        try:
            return model.get_submodule(p)
        except AttributeError:
            continue
    raise AttributeError(f"none of {paths} in {type(model).__name__}")


def final_norm(model):
    return _first_submodule(model, ("model.language_model.norm", "model.norm"))


def rotary(model):
    return _first_submodule(model, ("model.language_model.rotary_emb", "model.rotary_emb"))


def text_model(model):
    return _first_submodule(model, ("model.language_model", "model"))


def new_cache(config):
    from transformers import DynamicCache

    try:
        return DynamicCache(config=config)
    except TypeError:
        return DynamicCache()


def cache_len(cache) -> int:
    return int(cache.get_seq_length())


def snapshot_recurrent(cache) -> List[Any]:
    """Clone every linear-attention layer's conv/recurrent state (small)."""
    snap = []
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for i, layer in enumerate(layers):
            if hasattr(layer, "recurrent_states") or hasattr(layer, "conv_states"):
                conv = getattr(layer, "conv_states", None)
                rec = getattr(layer, "recurrent_states", None)
                snap.append((i,
                             None if conv is None else conv.clone(),
                             None if rec is None else rec.clone(),
                             getattr(layer, "has_previous_state", None)))
        return snap
    for attr in ("conv_states", "recurrent_states", "ssm_states"):
        lst = getattr(cache, attr, None)
        if isinstance(lst, list):
            snap.append((attr, [None if t is None else t.clone() for t in lst]))
    return snap


def restore_recurrent(cache, snap) -> None:
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for i, conv, rec, has_prev in snap:
            layer = layers[i]
            if conv is not None:
                layer.conv_states.copy_(conv)
            if rec is not None:
                layer.recurrent_states.copy_(rec)
            if has_prev is not None:
                layer.has_previous_state = has_prev
        return
    for attr, lst in snap:
        cur = getattr(cache, attr)
        for j, t in enumerate(lst):
            if t is not None:
                cur[j].copy_(t)


def crop_attention(cache, length: int) -> None:
    if hasattr(cache, "crop"):
        cache.crop(int(length))
        return
    for attr in ("key_cache", "value_cache"):
        lst = getattr(cache, attr, None)
        if isinstance(lst, list):
            for j, t in enumerate(lst):
                if t is not None and t.dim() >= 3 and t.shape[-2] > length:
                    lst[j] = t[..., :length, :]


class _Capture:
    """Forward hooks: final-norm output (h_p) and the trunk's position ids."""

    def __init__(self, model):
        self.hidden: Optional[torch.Tensor] = None
        self.position_ids: Optional[torch.Tensor] = None
        self._h1 = final_norm(model).register_forward_hook(self._on_norm)
        self._h2 = text_model(model).register_forward_pre_hook(self._on_text, with_kwargs=True)

    def _on_norm(self, _m, _a, out):
        self.hidden = out

    def _on_text(self, _m, _args, kwargs):
        p = kwargs.get("position_ids")
        if isinstance(p, torch.Tensor):
            if p.dim() == 3 and p.shape[0] == 4:
                p = p[1:]
            self.position_ids = p
        else:
            self.position_ids = None

    def remove(self):
        self._h1.remove()
        self._h2.remove()


def _mrope_positions(pos_1d: torch.Tensor, delta: torch.Tensor | int) -> torch.Tensor:
    """[T] text positions -> [3, 1, T] M-RoPE positions (+ rope delta)."""
    p = pos_1d.view(1, 1, -1) + (delta if isinstance(delta, int) else delta.view(1, -1, 1))
    return p.expand(3, 1, -1)


def _rope_delta(model) -> Any:
    base = getattr(model, "model", None)
    d = getattr(base, "rope_deltas", None)
    return 0 if d is None else d.to(torch.long)


def _causal_mask(t: int, past: int, device) -> torch.Tensor:
    q = torch.arange(t, device=device).view(t, 1) + past
    k = torch.arange(past + t, device=device).view(1, past + t)
    return (k <= q).view(1, 1, t, past + t)


def _causal_attention_bias(t: int, past: int, device, dtype) -> torch.Tensor:
    allowed = _causal_mask(t, past, device)
    bias = torch.zeros(allowed.shape, device=device, dtype=dtype)
    return bias.masked_fill(~allowed, float("-inf"))


# ---------------------------------------------------------------------------
# the speculative loop
# ---------------------------------------------------------------------------
def _reset_request_rope(model):
    """A new speculative request must not inherit another request's M-RoPE delta."""
    base = getattr(model, "model", None)
    if base is not None and hasattr(base, "rope_deltas"):
        base.rope_deltas = None


class MTPSpeculator:
    def __init__(self, model: nn.Module, mtp: QwenMTP, *, position_offset: int = 1):
        self.model = model
        self.mtp = mtp
        self.offset = int(position_offset)
        self.embed = model.get_input_embeddings()
        self.lm_head = model.get_output_embeddings()
        self.rot = rotary(model)

    def _draft(self, hidden, tokens, positions3, mtp_cache) -> int:
        emb = self.embed(tokens)
        dev = emb.device
        past = cache_len(mtp_cache) if len(getattr(mtp_cache, "layers", [])) else 0
        cos_sin = self.rot(emb, positions3.to(dev))
        # Qwen eager attention adds attention_mask to logits; a bool True/False
        # mask would instead add +1/0 and leave future positions visible.
        mask = _causal_attention_bias(tokens.shape[1], past, dev, emb.dtype)
        y = self.mtp(hidden, emb, cos_sin, mask, mtp_cache)
        logits = self.lm_head(y[:, -1:, :])
        return int(logits[0, -1].argmax().item())

    @_scope_gdn_cache
    @torch.no_grad()
    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        *,
        max_new_tokens: int,
        eos_ids: Sequence[int],
        on_token: Callable[[int], bool],
    ) -> Dict[str, Any]:
        """Greedy decode with MTP drafts.  ``on_token`` returns False to stop."""
        model = self.model
        if inputs.get("past_key_values") is not None:
            raise ValueError("MTP generate starts a new request; external caches are unsupported")
        _reset_request_rope(model)
        eos = set(int(e) for e in eos_ids)
        cap = _Capture(model)
        stats = {"verify_steps": 0, "accepted": 0, "rejected": 0, "rerun_steps": 0,
                 "emitted": 0, "mtp_position_offset": self.offset}
        t0 = time.perf_counter()
        try:
            out = model(**inputs, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            ids = inputs["input_ids"]
            dev = ids.device
            plen = ids.shape[1]
            nxt = int(out.logits[0, -1].argmax().item())
            hid = cap.hidden                                 # [1, P, H]
            pos = cap.position_ids
            delta = _rope_delta(model)
            if pos is None:
                pos3 = _mrope_positions(torch.arange(plen, device=dev), 0)
            else:
                pos3 = pos.to(dev)
            if self.offset == 1:
                last = pos3[:, :, -1:] + 1
                mpos = torch.cat([pos3[:, :, 1:], last], dim=-1)
            else:
                mpos = pos3
            mtp_cache = new_cache(self.mtp.config)
            toks = torch.cat([ids[:, 1:], torch.tensor([[nxt]], device=dev)], dim=1)
            draft = self._draft(hid, toks, mpos, mtp_cache)
            done = False

            def emit(tok: int) -> bool:
                stats["emitted"] += 1
                if tok in eos:
                    return False
                keep = on_token(tok)
                return bool(keep) and stats["emitted"] < int(max_new_tokens)

            if not emit(nxt):
                done = True
            x_last = nxt
            while not done:
                L = cache_len(cache)
                snap = snapshot_recurrent(cache)
                step = torch.tensor([[x_last, draft]], device=dev)
                out = model(input_ids=step, past_key_values=cache, use_cache=True,
                            logits_to_keep=2)
                stats["verify_steps"] += 1
                h2 = cap.hidden
                y1 = int(out.logits[0, 0].argmax().item())
                if y1 == draft:
                    stats["accepted"] += 1
                    y2 = int(out.logits[0, 1].argmax().item())
                    if not emit(draft):
                        break
                    if not emit(y2):
                        break
                    p = torch.tensor([L + 1, L + 2], device=dev) if self.offset == 1 \
                        else torch.tensor([L, L + 1], device=dev)
                    draft = self._draft(h2, torch.tensor([[draft, y2]], device=dev),
                                        _mrope_positions(p, delta), mtp_cache)
                    x_last = y2
                else:
                    stats["rejected"] += 1
                    restore_recurrent(cache, snap)
                    crop_attention(cache, L)
                    out1 = model(input_ids=torch.tensor([[x_last]], device=dev),
                                 past_key_values=cache, use_cache=True, logits_to_keep=1)
                    stats["rerun_steps"] += 1
                    y = int(out1.logits[0, -1].argmax().item())
                    stats["rerun_disagreed_with_verify"] = stats.get(
                        "rerun_disagreed_with_verify", 0) + int(y != y1)
                    if not emit(y):
                        break
                    p = torch.tensor([L + 1], device=dev) if self.offset == 1 \
                        else torch.tensor([L], device=dev)
                    draft = self._draft(cap.hidden, torch.tensor([[y]], device=dev),
                                        _mrope_positions(p, delta), mtp_cache)
                    x_last = y
        finally:
            cap.remove()
        n = stats["accepted"] + stats["rejected"]
        stats["acceptance_rate"] = (stats["accepted"] / n) if n else None
        stats["tokens_per_target_forward"] = (
            stats["emitted"] / max(1, 1 + stats["verify_steps"] + stats["rerun_steps"]))
        stats["seconds"] = round(time.perf_counter() - t0, 4)
        return stats


__all__ = [
    "MTPSpeculator",
    "QwenMTP",
    "cache_len",
    "crop_attention",
    "final_norm",
    "new_cache",
    "restore_recurrent",
    "snapshot_recurrent",
    "sequential_qwen35_gdn_cache",
]
