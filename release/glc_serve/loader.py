"""Stream a serving bundle onto the GPU; or load the uncompressed parent.

``load_bundle_model`` never materialises the dense model:

  * the skeleton is built with every PARAMETER on ``meta`` (buffers such as
    rotary ``inv_freq`` are computed for real, which a plain meta build would
    leave unfilled);
  * shards arrive hash-verified (``bundle.ShardStream``) and are walked one
    tensor at a time;
  * a coded tensor is uploaded AS STORED into a ``TBEServeLinear`` (backend
    ``tbe``), or decoded on the GPU and re-coded into GLC-FWP1 with a full
    bit-exact round trip (backend ``fwp1``; tensors FWP1 cannot represent
    exactly stay TBE and are counted);
  * raw tensors go straight to their device, except the embedding when
    ``embed_on_host`` (pinned host RAM, ``HostEmbedding``);
  * with a GPU weight budget, the tail of the text trunk is kept in pinned
    host memory and streamed layer by layer (``LayerStreamer``).

``load_dense_parent`` loads the original bf16 checkpoint with plain
``from_pretrained`` (plus its MTP head) so the head-to-head runs the
uncompressed parent through the SAME engine and server code.

The full multimodal model (text + vision tower + MTP) is the default;
``text_only`` exists for debugging and is refused in product configs by the
server unless explicitly requested.
"""
from __future__ import annotations

import contextlib
import importlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .bundle import (
    Bundle,
    BundleError,
    ShardStream,
    component_of,
    entry_dense_bytes,
    layer_of,
    read_entry,
    tbe_resident_bytes,
)
from .modules import (
    DEFAULT_FUSED_MAX_M,
    HostEmbedding,
    LayerStreamer,
    PoolRegistry,
    TBEServeLinear,
)

#: ``materialize`` decodes every coded tensor exactly at load into a plain
#: ``nn.Linear`` -- no VRAM saving; the CPU/dev path, and a third arm for the
#: head-to-head (container weights, dense compute).
BACKENDS = ("tbe", "fwp1", "materialize")
LOAD_RECEIPT_SCHEMA = "georefine.tbe_serve_load_receipt.v1"


@dataclass
class ServeOptions:
    backend: str = "tbe"                  # tbe | fwp1 (bundle); dense handled separately
    exec_mode: str = "fused"              # fused | exact
    device: str = "cuda:0"
    embed_on_host: bool = False
    text_only: bool = False               # DEBUG ONLY -- not a product config
    load_mtp: bool = True
    gpu_weight_budget_bytes: Optional[int] = None
    prefetch_decode: bool = False
    fused_max_m: int = DEFAULT_FUSED_MAX_M
    fetch_depth: int = 2
    evict_shards: bool = False
    attn_implementation: str = "sdpa"
    fwp1_group: int = 128
    strict: bool = True


@dataclass
class LoadedModel:
    model: nn.Module
    mtp: Optional[nn.Module]
    config: Any
    local_dir: Path
    receipt: Dict[str, Any]
    options: ServeOptions
    streamer: Optional[LayerStreamer] = None
    pools: Optional[PoolRegistry] = None
    name_to_module: Dict[str, nn.Module] = field(default_factory=dict)
    bundle: Optional[Bundle] = None
    backend_label: str = "tbe"


class LoadError(RuntimeError):
    def __init__(self, reason: str, detail: str = ""):
        self.reason, self.detail = str(reason), str(detail)
        super().__init__(f"{reason}: {detail}" if detail else reason)


# ---------------------------------------------------------------------------
# skeletons
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def params_on_meta():
    """Every parameter registered inside the block lands on ``meta``.

    Buffers are untouched, so derived buffers (rotary ``inv_freq`` and the
    like) are computed for real on the CPU and simply moved to the device
    afterwards -- the same contract as accelerate's
    ``init_empty_weights(include_buffers=False)``.
    """
    old = nn.Module.register_parameter

    def register(self, name, param):
        old(self, name, param)
        if param is not None and not param.is_meta:
            p = self._parameters[name]
            self._parameters[name] = nn.Parameter(
                torch.empty_like(p, device="meta"), requires_grad=p.requires_grad,
            )

    nn.Module.register_parameter = register
    try:
        yield
    finally:
        nn.Module.register_parameter = old


def model_class_for(config, *, text_only: bool):
    import transformers

    if text_only:
        return transformers.AutoModelForCausalLM
    for arch in getattr(config, "architectures", None) or []:
        cls = getattr(transformers, arch, None)
        if cls is not None:
            return cls
    for auto in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        cls = getattr(transformers, auto, None)
        if cls is not None:
            return cls
    raise LoadError("no_model_class", str(getattr(config, "architectures", None)))


def build_skeleton(config, *, text_only: bool, attn_implementation: str):
    cls = model_class_for(config, text_only=text_only)
    kwargs = {"attn_implementation": attn_implementation}
    with params_on_meta():
        if cls.__name__.startswith("Auto"):
            try:
                model = cls.from_config(config, dtype=torch.bfloat16, **kwargs)
            except TypeError:
                model = cls.from_config(config, torch_dtype=torch.bfloat16, **kwargs)
        else:
            try:
                model = cls._from_config(config, dtype=torch.bfloat16, **kwargs)
            except TypeError:
                model = cls._from_config(config, torch_dtype=torch.bfloat16, **kwargs)
    model.eval()
    return model


def _disable_cuda_only_conv_kernels(model: nn.Module, device: torch.device) -> None:
    """Use Transformers' reference convolution on CPU-only model placements.

    ``causal-conv1d`` registers a callable even when it was built for CUDA;
    invoking it with CPU tensors raises instead of selecting the model's
    existing PyTorch fallback. Keep the fast path intact for CUDA placement.
    """
    if device.type == "cuda":
        return
    for module in model.modules():
        source = importlib.import_module(type(module).__module__)
        for name in ("causal_conv1d_fn", "causal_conv1d_update"):
            if getattr(module, name, None) is not None:
                fallback = getattr(source, "torch_causal_conv1d_update", None)
                setattr(module, name, fallback if name.endswith("update") else None)
        for name, fallback_name in (
            ("chunk_gated_delta_rule", "torch_chunk_gated_delta_rule"),
            ("recurrent_gated_delta_rule", "torch_recurrent_gated_delta_rule"),
        ):
            if getattr(module, name, None) is not None:
                fallback = getattr(source, fallback_name, None)
                if fallback is not None:
                    setattr(module, name, fallback)
        norm = getattr(module, "norm", None) if hasattr(module, "head_v_dim") else None
        reference_norm = getattr(source, "Qwen3_5RMSNormGated", None)
        if norm is not None and reference_norm is not None and type(norm) is not reference_norm:
            weight = getattr(norm, "weight", None)
            if weight is not None:
                replacement = reference_norm(
                    int(weight.numel()),
                    eps=float(getattr(module, "layer_norm_epsilon", 1e-6)),
                ).to(device=weight.device, dtype=weight.dtype)
                with torch.no_grad():
                    replacement.weight.copy_(weight)
                module.norm = replacement


def _initialize_tbe_mma_arch(device: torch.device, backend: str) -> None:
    """Pin the TBE kernel target from the selected CUDA device before use."""
    if device.type != "cuda" or backend not in ("tbe", "fwp1"):
        return
    from glc_loader.tbe_mma import resolve_tbe_mma_arch

    resolve_tbe_mma_arch(device)


def text_config_of(config):
    return getattr(config, "text_config", None) or config


def has_mtp_entries(names) -> bool:
    return any(str(n).startswith("mtp.") for n in names)


def build_mtp_skeleton(config):
    from .mtp import QwenMTP

    with params_on_meta():
        mtp = QwenMTP(text_config_of(config))
    return mtp.eval()


# ---------------------------------------------------------------------------
# name resolution (checkpoint key -> module path)
# ---------------------------------------------------------------------------
def _resolver(model: nn.Module):
    from glc_loader.tbe_stream_loader import (
        _ignore_unexpected_patterns,
        _real_tensor_names,
        resolve_real_name,
    )

    real = _real_tensor_names(model)
    ignore = _ignore_unexpected_patterns(model)

    def resolve(name: str) -> Optional[str]:
        r = resolve_real_name(name, real)
        if r is not None:
            return r
        if any(re.match(p, name) for p in ignore):
            return None
        raise LoadError(
            "unknown_module",
            f"{name!r} addresses no parameter of the {type(model).__name__} "
            "skeleton and is not declared ignorable by it",
        )

    return resolve


def split_parent(root: nn.Module, name: str) -> Tuple[nn.Module, str]:
    parent_name, _, attr = name.rpartition(".")
    return (root.get_submodule(parent_name) if parent_name else root), attr


def _set_module(root: nn.Module, path: str, module: nn.Module) -> None:
    parent, attr = split_parent(root, path)
    setattr(parent, attr, module)


def _place_param(parent: nn.Module, attr: str, tensor: torch.Tensor, device) -> None:
    placed = tensor.detach().to(device)
    if attr in parent._parameters:
        parent._parameters[attr] = nn.Parameter(placed, requires_grad=False)
    elif attr in parent._buffers:
        parent._buffers[attr] = placed
    else:
        # a TBEServeLinear's bias slot registered as None
        setattr(parent, attr, nn.Parameter(placed, requires_grad=False))


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------
def plan_offload(bundle: Bundle, opts: ServeOptions) -> Dict[str, Any]:
    """Which text layers stay on the host, from MANIFEST bytes (no guessing).

    Projected resident GPU weight bytes = every coded tensor's stored bytes +
    every raw tensor's bytes, minus the embedding when it lives on the host,
    minus the vision tower / MTP when they are not loaded.  Layers are
    offloaded from the END of the trunk until the projection plus the two
    streaming slots fits the budget.
    """
    tensors = bundle.tensors
    skip = set()
    if opts.text_only:
        skip.update({"vision", "mtp"})
    if not opts.load_mtp:
        skip.add("mtp")
    per_layer: Dict[int, int] = {}
    total = 0
    for t in tensors:
        comp = component_of(t["name"])
        if comp in skip:
            continue
        if comp == "embed" and opts.embed_on_host:
            continue
        b = tbe_resident_bytes(t) if t["kind"] == "tbe" else entry_dense_bytes(t)
        total += b
        layer = layer_of(t["name"])
        if layer is not None and t["kind"] == "tbe":
            per_layer[layer] = per_layer.get(layer, 0) + b
    out = {"projected_resident_bytes_all_gpu": total, "offload_layers": [],
           "budget_bytes": opts.gpu_weight_budget_bytes}
    if not opts.gpu_weight_budget_bytes:
        out["projected_resident_bytes"] = total
        return out
    budget = int(opts.gpu_weight_budget_bytes)
    resident = total
    off: List[int] = []
    slot = 0
    for layer in sorted(per_layer, reverse=True):
        if resident + 2 * slot <= budget:
            break
        off.append(layer)
        resident -= per_layer[layer]
        slot = max(slot, per_layer[layer])
    if resident + 2 * slot > budget:
        raise LoadError(
            "budget_unreachable",
            f"even with every text layer offloaded the resident projection is "
            f"{resident + 2 * slot} bytes > budget {budget}",
        )
    out.update({
        "offload_layers": sorted(off),
        "offloaded_coded_bytes": sum(per_layer[l] for l in off),
        "stream_slot_bytes": 2 * slot,
        "projected_resident_bytes": resident + 2 * slot,
    })
    return out


def pool_capacity(bundle: Bundle, opts: ServeOptions) -> int:
    """Largest coded linear served through the pool (heads use the fused path)."""
    cap = 0
    for t in bundle.tensors:
        if t["kind"] != "tbe":
            continue
        comp = component_of(t["name"])
        if comp in ("lm_head", "embed"):
            continue
        if opts.text_only and comp in ("vision", "mtp"):
            continue
        shp = t.get("coded_shape") or t["shape"]
        cap = max(cap, int(shp[0]) * int(shp[1]))
    return cap


# ---------------------------------------------------------------------------
# FWP1 re-coding
# ---------------------------------------------------------------------------
def to_fwp1_linear(container, bias, device, *, group: int, name: str):
    """Decode the stored TBE container on the GPU, re-code FWP1, certify."""
    from glc_loader.container import encode_fwp1, fwp1_certify, load_kernels
    from glc_loader.modules import GLCLinear
    from glc_loader.tbe_mma import tbe_mma_decode, upload_tbe

    if load_kernels() is None:
        raise LoadError(
            "fwp1_kernels_unavailable",
            "glc_loader/fwp1_kernels.py is not vendored or triton/CUDA is absent; "
            "run `python -m glc_serve.vendor` first",
        )
    dev = upload_tbe(container, device)
    w = tbe_mma_decode(dev)
    del dev
    t = encode_fwp1(w, group=group)
    check = fwp1_certify(t, w)
    del w
    if not (check["bitwise_ok"] and check.get("fully_checked", True)):
        raise LoadError("fwp1_not_bit_exact", f"{name}: {check}")
    b = None if bias is None else bias.detach().to(device)
    return GLCLinear(t, b, "triton")


# ---------------------------------------------------------------------------
# the bundle loader
# ---------------------------------------------------------------------------
def _cuda_mem(device) -> Dict[str, int]:
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        return {}
    d = torch.device(device)
    return {
        "allocated": int(torch.cuda.memory_allocated(d)),
        "reserved": int(torch.cuda.memory_reserved(d)),
        "max_allocated": int(torch.cuda.max_memory_allocated(d)),
        "max_reserved": int(torch.cuda.max_memory_reserved(d)),
    }


def load_bundle_model(bundle: Bundle, opts: ServeOptions, *, log=print) -> LoadedModel:
    from transformers import AutoConfig

    if opts.backend not in BACKENDS:
        raise LoadError("unknown_backend", opts.backend)
    t_start = time.perf_counter()
    device = torch.device(opts.device)
    _initialize_tbe_mma_arch(device, opts.backend)
    config = AutoConfig.from_pretrained(str(bundle.local_dir))
    model = build_skeleton(config, text_only=opts.text_only,
                           attn_implementation=opts.attn_implementation)
    _disable_cuda_only_conv_kernels(model, device)
    names = [t["name"] for t in bundle.tensors]
    mtp = None
    if opts.load_mtp and not opts.text_only and has_mtp_entries(names):
        mtp = build_mtp_skeleton(config)
        _disable_cuda_only_conv_kernels(mtp, device)
    t_skeleton = time.perf_counter()
    resolve = _resolver(model)
    resolve_mtp = _resolver(mtp) if mtp is not None else None
    plan = plan_offload(bundle, opts)
    offload_layers = set(plan.get("offload_layers") or [])
    pools = PoolRegistry()
    pool = None
    if device.type == "cuda" and opts.backend != "materialize":
        pool = pools.get(device, pool_capacity(bundle, opts),
                         slots=2 if opts.prefetch_decode else 1)
    stream = ShardStream(bundle, depth=opts.fetch_depth, evict=opts.evict_shards)
    counts = {"tbe_linear": 0, "fwp1_linear": 0, "fwp1_fallback_tbe": 0,
              "decoded_dense": 0, "raw": 0, "host_embed": 0, "ignored": 0,
              "offloaded_linears": 0}
    fwp1_fallbacks: List[str] = []
    name_to_module: Dict[str, nn.Module] = {}
    offloaded: Dict[int, List[TBEServeLinear]] = {}
    host_embed_bytes = 0
    t_first_shard = None

    from safetensors import safe_open

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    mem_before = _cuda_mem(device)
    for idx, res in stream:
        if t_first_shard is None:
            t_first_shard = time.perf_counter()
        with safe_open(str(res.path), framework="pt", device="cpu") as h:
            for entry in bundle.entries_for_shard(idx):
                name = entry["name"]
                comp = component_of(name)
                if comp == "mtp":
                    if mtp is None:
                        counts["ignored"] += 1
                        continue
                    root, local = mtp, resolve_mtp(name[len("mtp."):])
                else:
                    if opts.text_only and comp == "vision":
                        counts["ignored"] += 1
                        continue
                    root, local = model, resolve(name)
                if local is None:
                    counts["ignored"] += 1
                    continue
                parent, attr = split_parent(root, local)
                payload = read_entry(h, entry)
                owner = local.rpartition(".")[0]
                if entry["kind"] == "tbe" and isinstance(parent, nn.Linear) and attr == "weight":
                    bias = parent.bias if (parent.bias is not None and not parent.bias.is_meta) else None
                    layer = layer_of(name)
                    module = None
                    if opts.backend == "fwp1" and layer not in offload_layers:
                        try:
                            module = to_fwp1_linear(payload, bias, device,
                                                    group=opts.fwp1_group, name=name)
                            counts["fwp1_linear"] += 1
                        except LoadError as exc:
                            if exc.reason == "fwp1_kernels_unavailable":
                                raise
                            fwp1_fallbacks.append(name)
                            counts["fwp1_fallback_tbe"] += 1
                        except Exception as exc:  # palette overflow etc.
                            fwp1_fallbacks.append(f"{name}: {type(exc).__name__}")
                            counts["fwp1_fallback_tbe"] += 1
                    if module is None and opts.backend == "materialize":
                        from glc_loader.tbe_container import decode_tbe

                        dense = decode_tbe(payload).reshape(
                            [int(d) for d in entry.get("coded_shape") or entry["shape"]])
                        lin = nn.Linear(dense.shape[1], dense.shape[0],
                                        bias=bias is not None, device="meta")
                        lin.weight = nn.Parameter(dense.to(device), requires_grad=False)
                        if bias is not None:
                            lin.bias = nn.Parameter(bias.detach().to(device), requires_grad=False)
                        module = lin
                        counts["decoded_dense"] += 1
                    if module is None:
                        is_off = layer is not None and layer in offload_layers
                        module = TBEServeLinear(
                            payload, bias, device, pool=pool, name=name,
                            exec_mode=opts.exec_mode, fused_max_m=opts.fused_max_m,
                            offload=is_off,
                        )
                        counts["tbe_linear"] += 1
                        if is_off:
                            offloaded.setdefault(layer, []).append(module)
                            counts["offloaded_linears"] += 1
                    _set_module(root, owner, module)
                    name_to_module[name] = module
                    continue
                if entry["kind"] == "tbe":
                    # coded at rest, not a linear weight here (e.g. a coded
                    # embedding): decode once, exactly, and place dense.
                    from glc_loader.tbe_container import decode_tbe

                    dense = decode_tbe(payload).reshape([int(d) for d in entry["shape"]])
                    counts["decoded_dense"] += 1
                    payload = dense
                if comp == "embed" and opts.embed_on_host and isinstance(parent, nn.Embedding):
                    emb = HostEmbedding(payload, device, padding_idx=parent.padding_idx)
                    _set_module(root, owner, emb)
                    host_embed_bytes += emb.host_bytes
                    counts["host_embed"] += 1
                    name_to_module[name] = emb
                    continue
                _place_param(parent, attr, payload, device)
                counts["raw"] += 1
                name_to_module[name] = parent
                del payload
    t_placed = time.perf_counter()

    for root in [model] + ([mtp] if mtp is not None else []):
        for mod in root.modules():
            for key, buf in list(mod._buffers.items()):
                if buf is not None and not buf.is_meta and buf.device != device:
                    mod._buffers[key] = buf.to(device)
    tie = bool(getattr(text_config_of(config), "tie_word_embeddings", False)) or bool(
        getattr(config, "tie_word_embeddings", False))
    lm = getattr(model, "lm_head", None)
    if tie and isinstance(lm, nn.Linear) and lm.weight.is_meta:
        emb = model.get_input_embeddings()
        if isinstance(emb, HostEmbedding):
            raise LoadError("tied_head_with_host_embed",
                            "a tied lm_head shares the embedding; keep the embedding on the GPU")
        lm.weight = emb.weight
    residual = []
    for root in [model] + ([mtp] if mtp is not None else []):
        residual += [n for n, p in root.named_parameters() if p is not None and p.is_meta]
        residual += [n for n, b in root.named_buffers() if b is not None and b.is_meta]
    if residual and opts.strict:
        raise LoadError("unfilled_parameters",
                        f"{len(residual)} tensors never supplied: {sorted(residual)[:8]}")

    streamer = None
    if offloaded:
        layer_modules = _decoder_layers(model)
        pairs = [(layer_modules[l], offloaded[l]) for l in sorted(offloaded)]
        streamer = LayerStreamer(device, pairs)
    mem_after = _cuda_mem(device)
    acct = bundle.manifest.get("accounting", {})
    receipt = {
        "schema": LOAD_RECEIPT_SCHEMA,
        "status": "ok",
        "backend": opts.backend,
        "options": asdict(opts),
        "bundle": {
            "manifest_sha256": bundle.manifest_sha256,
            "format": bundle.manifest.get("format"),
            "n_shards": len(bundle.shards),
            "bytes": bundle.total_bytes(),
            "location": bundle.source.location,
        },
        "counts": counts,
        "fwp1_fallbacks": fwp1_fallbacks[:32],
        "offload_plan": plan,
        "streamer": None if streamer is None else {
            "n_layers": len(streamer.layers), "host_bytes": streamer.host_bytes,
            "gpu_slot_bytes": streamer.gpu_slot_bytes,
        },
        "host_embed_bytes": host_embed_bytes,
        "pool_capacity_bytes": pools.capacity_bytes,
        "manifest_accounting": acct,
        "shard_fetch": stream.fetch_log,
        "timing_s": {
            "skeleton": round(t_skeleton - t_start, 3),
            "first_shard_ready": round((t_first_shard or t_placed) - t_start, 3),
            "stream_and_place": round(t_placed - t_skeleton, 3),
            "total_load": round(time.perf_counter() - t_start, 3),
        },
        "cuda_memory_before": mem_before,
        "cuda_memory_after_load": mem_after,
        "residual_meta": residual[:16],
        "text_only": bool(opts.text_only),
        "mtp_loaded": mtp is not None,
    }
    label = opts.backend if opts.exec_mode == "fused" else f"{opts.backend}-{opts.exec_mode}"
    return LoadedModel(
        model=model, mtp=mtp, config=config, local_dir=bundle.local_dir,
        receipt=receipt, options=opts, streamer=streamer, pools=pools,
        name_to_module=name_to_module, bundle=bundle, backend_label=label,
    )


def _decoder_layers(model: nn.Module) -> List[nn.Module]:
    for path in ("model.language_model.layers", "model.layers", "language_model.layers"):
        try:
            return list(model.get_submodule(path))
        except AttributeError:
            continue
    raise LoadError("no_decoder_layers", type(model).__name__)


# ---------------------------------------------------------------------------
# the uncompressed parent
# ---------------------------------------------------------------------------
def load_dense_parent(model_dir: os.PathLike | str, opts: ServeOptions, *, log=print) -> LoadedModel:
    """``from_pretrained`` on the original checkpoint, plus its MTP head."""
    from transformers import AutoConfig

    t0 = time.perf_counter()
    device = torch.device(opts.device)
    config = AutoConfig.from_pretrained(str(model_dir))
    cls = model_class_for(config, text_only=opts.text_only)
    kwargs = dict(attn_implementation=opts.attn_implementation,
                  device_map={"": str(device)})
    try:
        model = cls.from_pretrained(str(model_dir), dtype=torch.bfloat16, **kwargs)
    except TypeError:
        model = cls.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16, **kwargs)
    model.eval()
    _disable_cuda_only_conv_kernels(model, device)
    if opts.embed_on_host:
        emb = model.get_input_embeddings()
        host = HostEmbedding(emb.weight.detach().cpu(), device, padding_idx=emb.padding_idx)
        model.set_input_embeddings(host)
        del emb
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    mtp = None
    if opts.load_mtp and not opts.text_only:
        mtp = load_dense_mtp(model_dir, config, device)
    receipt = {
        "schema": LOAD_RECEIPT_SCHEMA, "status": "ok", "backend": "dense",
        "options": asdict(opts), "model_dir": str(model_dir),
        "timing_s": {"total_load": round(time.perf_counter() - t0, 3)},
        "cuda_memory_after_load": _cuda_mem(device),
        "mtp_loaded": mtp is not None, "text_only": bool(opts.text_only),
    }
    return LoadedModel(model=model, mtp=mtp, config=config, local_dir=Path(model_dir),
                       receipt=receipt, options=opts, backend_label="dense")


def load_dense_mtp(model_dir, config, device) -> Optional[nn.Module]:
    """The parent's MTP head, from its own safetensors (dense ``nn.Linear``)."""
    from safetensors import safe_open

    index = Path(model_dir) / "model.safetensors.index.json"
    files: Dict[str, List[str]] = {}
    if index.is_file():
        wm = json.loads(index.read_text())["weight_map"]
        for k, f in wm.items():
            if k.startswith("mtp."):
                files.setdefault(f, []).append(k)
    else:
        for p in Path(model_dir).glob("*.safetensors"):
            with safe_open(str(p), framework="pt") as h:
                ks = [k for k in h.keys() if k.startswith("mtp.")]
            if ks:
                files[p.name] = ks
    if not files:
        return None
    mtp = build_mtp_skeleton(config)
    _disable_cuda_only_conv_kernels(mtp, torch.device(device))
    for f, keys in files.items():
        with safe_open(str(Path(model_dir) / f), framework="pt", device="cpu") as h:
            for k in keys:
                parent, attr = split_parent(mtp, k[len("mtp."):])
                _place_param(parent, attr, h.get_tensor(k), device)
    for mod in mtp.modules():
        for key, buf in list(mod._buffers.items()):
            if buf is not None and not buf.is_meta:
                mod._buffers[key] = buf.to(device)
    left = [n for n, p in mtp.named_parameters() if p.is_meta]
    if left:
        raise LoadError("mtp_incomplete", str(left[:8]))
    return mtp.eval()


__all__ = [
    "BACKENDS",
    "LoadError",
    "LoadedModel",
    "ServeOptions",
    "build_skeleton",
    "load_bundle_model",
    "load_dense_mtp",
    "load_dense_parent",
    "params_on_meta",
    "plan_offload",
    "pool_capacity",
]
