"""Streaming, multi-device TBE load: shard by shard, tensor by tensor.

THE CONSTRAINT.  ``AutoModelForCausalLM.from_pretrained(...).to(device)``
materialises the whole bf16 checkpoint before anything is coded.  For
Qwen3.8-Flash-Next that is ~360 GB on one allocator -- it does not fit a card
and it does not fit most hosts' RAM.  Even the single-device TBE path in
``tbe_serving.enable_hf_tbe_serving`` pays a peak of dense + coded, because
it converts a model that is already fully resident (it discloses that peak as
``tbe_build_peak_bytes``; see ``scripts/glc_tbe_certify.py``).

THIS module never holds the dense model.  The skeleton is built on ``meta``
(no storage at all), the checkpoint is walked one tensor at a time, and each
tensor is either encoded into a container and installed as a coded module, or
copied raw, straight onto ITS OWN mapped device.  The high-water mark is one
tensor's dense bytes plus one tensor's container -- for Flash-Next, one
expert slice or one attention projection, not 360 GB.

SEAMS.  ``convert_linear`` / ``convert_expert_bank`` / ``place_raw`` are
injected, defaulting to the real CUDA-only implementations.  A CPU test
substitutes fakes and exercises the whole placement, ordering, bias-pairing,
tie-weights and residual-meta-parameter logic on a two-"device" map with no
GPU present -- the same testability contract ``_run_tbe_swap_linears`` gives
the single-device path.

FAIL CLOSED.  A tensor whose name maps to no device, a parameter the
checkpoint never supplied (still on ``meta`` at the end), a coded round-trip
that is not bit-exact: each raises :class:`TBEStreamLoadError` with a typed
reason.  Nothing here falls back to loading dense, and nothing here leaves a
half-populated model in the caller's hands.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from .tbe_container import tbe_certify
from .tbe_device_map import TBEDeviceMapError, resolve_device
from .tbe_mma import MMA_N, TILE
from .tbe_modules import GLCTBEExpertBank, GLCTBELinear, TBEPoolRegistry
from .tbe_serving import (
    DEFAULT_MIN_NUMEL,
    TBE_SERVING_RECEIPT_SCHEMA,
    _SKIP_NAME_SUBSTRINGS,
    _assert_container_on_device,
    _encode_on_device,
    _normalize_device_map,
)

TBE_STREAM_RECEIPT_SCHEMA = "georefine.tbe_stream_load_receipt.v1"


class TBEStreamLoadError(RuntimeError):
    """A streaming TBE load could not be completed as specified."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = str(reason)
        self.detail = str(detail)
        super().__init__(f"{reason}: {detail}" if detail else str(reason))


# ---------------------------------------------------------------------------
# module addressing
# ---------------------------------------------------------------------------
def split_parent(model: nn.Module, name: str) -> Tuple[nn.Module, str]:
    """``(owning_module, attribute)`` for a dotted parameter name."""
    parent_name, _, attr = str(name).rpartition(".")
    if not parent_name:
        return model, attr
    try:
        return model.get_submodule(parent_name), attr
    except AttributeError as exc:
        raise TBEStreamLoadError(
            "unknown_module",
            f"{name!r}: the model has no submodule {parent_name!r}",
        ) from exc


# ---------------------------------------------------------------------------
# checkpoint-name -> real-module-tree resolution
#
# A raw checkpoint tensor name is the SAFETENSORS KEY, not necessarily the
# skeleton's own attribute path.  For an ordinary model the two coincide and
# ``split_parent`` above is the whole story.  They diverge for a composite
# HF class: Qwen3.5's "VLM compatibility" ``AutoModelForCausalLM`` entry
# (``Qwen3_5ForCausalLM``) builds a FLAT skeleton -- ``model.embed_tokens``,
# ``model.layers.N.*`` -- while the checkpoint's safetensors keys are
# ``model.language_model.embed_tokens.weight``, ``model.language_model.
# layers.N.*`` (written for the full ``Qwen3_5ForConditionalGeneration``
# class, which DOES carry a ``language_model`` submodule).  ``from_pretrained``
# reconciles this with a private, config-keyed rename table
# (``transformers.conversion_mapping``); this module never materialises a
# dense model and cannot pay for that machinery (some of its rules fuse or
# split MULTIPLE checkpoint tensors, incompatible with one-tensor-at-a-time
# streaming), so it resolves names the other direction instead: against the
# skeleton's OWN ``named_parameters()``/``named_buffers()``, which is ground
# truth for what the instantiated class actually has.
# ---------------------------------------------------------------------------
def _real_tensor_names(model: nn.Module) -> "set":
    """Every parameter and buffer name the skeleton's own tree owns.

    Cheap even on a 360 GB architecture: the skeleton is on ``meta``, so this
    walks references, never data.
    """
    names = {n for n, _ in model.named_parameters()}
    names.update(n for n, _ in model.named_buffers())
    return names


def _candidates_dropping_segments(parts: List[str], depth: int) -> Iterator[str]:
    """Every name formed by dropping ``depth`` INTERIOR, NON-NUMERIC segments.

    Never the root (index 0) and never the leaf attribute (last index) --
    those are never the wrapper being unwound -- and never a purely numeric
    segment, so a layer index can never be dropped and silently collapse two
    layers onto one name.
    """
    from itertools import combinations

    n = len(parts)
    interior = [i for i in range(1, n - 1) if not parts[i].isdigit()]
    if depth > len(interior):
        return
    for combo in combinations(interior, depth):
        drop = set(combo)
        yield ".".join(p for i, p in enumerate(parts) if i not in drop)


def resolve_real_name(
    name: str, real_names: "set", *, max_drop: int = 2,
) -> Optional[str]:
    """The skeleton's real parameter/buffer name a checkpoint ``name`` is for.

    A direct hit -- every non-wrapped model, and the overwhelming majority of
    tensors even for a wrapped one -- returns immediately.  Failing that,
    this tries removing up to ``max_drop`` interior path segments (see
    :func:`_candidates_dropping_segments`) and accepts a match only if it is
    UNIQUE at the shallowest depth that produces any: this is not a special
    case for one prefix string, it is a bounded search over the skeleton's
    OWN names, so it resolves a ``model.language_model.*`` checkpoint key,
    a ``model.visual.*`` one, or an ``mtp.*`` one identically -- and returns
    ``None`` for a tensor that genuinely has no home in this skeleton (a
    tower the skeleton never built, or an invented name), which the caller
    turns into the same typed refusal ``split_parent`` has always raised for
    an unknown module.
    """
    name = str(name)
    if name in real_names:
        return name
    parts = name.split(".")
    if len(parts) < 3:
        return None
    for depth in range(1, max_drop + 1):
        matches = {
            candidate for candidate in _candidates_dropping_segments(parts, depth)
            if candidate in real_names
        }
        if len(matches) == 1:
            return next(iter(matches))
        if len(matches) > 1:
            return None  # ambiguous at the shallowest matching depth: refuse
    return None


def _ignore_unexpected_patterns(model: nn.Module) -> "set":
    """Regex patterns THIS class declares as expected to have no home.

    Standard ``transformers.PreTrainedModel`` machinery
    (``_keys_to_ignore_on_load_unexpected``, collected onto the instance from
    every submodule at ``__init__``): a class built for a narrower scope than
    its checkpoint declares in advance which checkpoint key prefixes it knows
    will not resolve.  Qwen3.5's flat, text-only ``AutoModelForCausalLM``
    compatibility class (``Qwen3_5ForCausalLM``) sets
    ``{'^model.visual.*', '^mtp.*'}`` for exactly this reason: its checkpoint
    is written for the full VLM class and also carries a vision tower and an
    MTP head this class never builds.  ``from_pretrained`` drops those keys
    silently rather than treating them as an error; this stream loader does
    the same, reading the same standard attribute -- nothing here names
    ``visual`` or ``mtp`` itself, so a different class's own declared
    patterns (Flash-Next's, say) are honoured identically.
    """
    patterns = getattr(model, "_keys_to_ignore_on_load_unexpected", None)
    return set(patterns) if patterns else set()


def _is_linear_weight(parent: nn.Module, attr: str) -> bool:
    return isinstance(parent, nn.Linear) and attr == "weight"


def linear_is_codable(name: str, tensor: torch.Tensor, min_numel: int) -> bool:
    """The same eligibility rule ``_eligible_tbe_linears`` applies, by name."""
    if any(s in str(name).lower() for s in _SKIP_NAME_SUBSTRINGS):
        return False
    if tensor.dim() != 2 or tensor.dtype != torch.bfloat16:
        return False
    if int(tensor.numel()) < int(min_numel):
        return False
    out_f, in_f = int(tensor.shape[0]), int(tensor.shape[1])
    return not (in_f % TILE or out_f % MMA_N)


def any_tensor_is_codable(
    name: str,
    tensor: torch.Tensor,
    min_numel: int,
    *,
    embedding: bool = False,
    onedim: bool = False,
    shape_fallback: bool = False,
) -> bool:
    """``linear_is_codable`` widened by the transcoder's coverage policies.

    Mirrors ``scripts/glc_tbe_transcode.classify_tensor``'s extended policy on
    the loader side, so a caller can ask "would this artifact's build have
    coded this tensor?" without the build repo present.  With all three flags
    off it is ``linear_is_codable`` exactly -- that equivalence is pinned by
    ``tests/test_tbe_coverage.py``, because the moment the two rules disagree
    the loader starts refusing tensors the artifact contains.

    ``embedding``      drop the ``_SKIP_NAME_SUBSTRINGS`` exclusion.
    ``onedim``         admit rank < 2 through the ``(1, L)`` view.  Such a
                       shape can never satisfy ``out_f % MMA_N``, so it is
                       coded ``flat64`` and decoded dense -- the mma gate is
                       bypassed for it, not passed by it.
    ``shape_fallback`` admit a 2-D (or flattened) shape the fragment kernel
                       cannot tile, likewise coded ``flat64``.
    """
    if not embedding and any(s in str(name).lower() for s in _SKIP_NAME_SUBSTRINGS):
        return False
    if tensor.dtype != torch.bfloat16:
        return False
    rank = int(tensor.dim())
    if rank < 2:
        if not onedim:
            return False
        if int(tensor.numel()) < int(min_numel):
            return False
        return int(tensor.numel()) > 0
    if rank != 2:
        return False
    if int(tensor.numel()) < int(min_numel):
        return False
    out_f, in_f = int(tensor.shape[0]), int(tensor.shape[1])
    if in_f % TILE or out_f % MMA_N:
        return bool(shape_fallback)
    return True


def expert_bank_is_codable(
    name: str, tensor: torch.Tensor, min_numel: int,
) -> bool:
    """Eligibility for a fused 3-D MoE expert parameter, per expert slice."""
    if any(s in str(name).lower() for s in _SKIP_NAME_SUBSTRINGS):
        return False
    if tensor.dim() != 3 or tensor.dtype != torch.bfloat16:
        return False
    _e, out_f, in_f = (int(v) for v in tensor.shape)
    if int(out_f) * int(in_f) < int(min_numel):
        return False
    return not (in_f % TILE or out_f % MMA_N)


# ---------------------------------------------------------------------------
# default (real) converters -- CUDA only, certified, no dense fallback
# ---------------------------------------------------------------------------
def default_convert_linear(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    device: torch.device,
    *,
    name: str,
    layout: str,
    pool,
) -> GLCTBELinear:
    container = _encode_on_device(weight, layout=layout, device=device)
    _assert_container_on_device(container, device, name)
    with torch.device(device):
        check = tbe_certify(container, weight)
    if not check["bitwise_ok"]:
        raise TBEStreamLoadError(
            "not_bit_exact",
            f"{name}: {check['mismatched_elements']} element(s) differ after "
            "round-trip; the container is not installed",
        )
    with torch.device(device):
        return GLCTBELinear(container, bias, device, pool)


def default_convert_expert_bank(
    param: torch.Tensor,
    device: torch.device,
    *,
    name: str,
    layout: str,
    pool,
) -> GLCTBEExpertBank:
    containers = []
    for e in range(int(param.shape[0])):
        w = param[e].contiguous()
        container = _encode_on_device(w, layout=layout, device=device)
        _assert_container_on_device(container, device, f"{name}[{e}]")
        with torch.device(device):
            check = tbe_certify(container, w)
        if not check["bitwise_ok"]:
            raise TBEStreamLoadError(
                "not_bit_exact",
                f"{name}[{e}]: {check['mismatched_elements']} element(s) "
                "differ after round-trip; the bank is not installed",
            )
        containers.append(container)
    with torch.device(device):
        return GLCTBEExpertBank(containers, device, pool)


def default_place_raw(
    parent: nn.Module, attr: str, tensor: torch.Tensor, device: torch.device,
) -> None:
    """Install an uncoded tensor directly on its mapped device."""
    placed = tensor.detach().to(device)
    if attr in parent._parameters:
        parent._parameters[attr] = nn.Parameter(placed, requires_grad=False)
    elif attr in parent._buffers:
        parent._buffers[attr] = placed
    else:
        raise TBEStreamLoadError(
            "unknown_tensor_slot",
            f"{type(parent).__name__}.{attr} is neither a parameter nor a "
            "buffer of that module",
        )


# ---------------------------------------------------------------------------
# the stream
# ---------------------------------------------------------------------------
def stream_place_tensors(
    model: nn.Module,
    tensors: Iterable[Tuple[str, torch.Tensor]],
    device_map: Any,
    *,
    min_numel: int = DEFAULT_MIN_NUMEL,
    layout: str = "mma16",
    experts: bool = True,
    registry: Optional[TBEPoolRegistry] = None,
    convert_linear: Callable[..., nn.Module] = default_convert_linear,
    convert_expert_bank: Callable[..., nn.Module] = default_convert_expert_bank,
    place_raw: Callable[..., None] = default_place_raw,
    strict_meta: bool = True,
    max_conversion_source_bytes: Any = None,
) -> Dict[str, Any]:
    """Place every tensor of a checkpoint stream onto its mapped device.

    ``model`` is expected to be a meta-device skeleton.  ``tensors`` is any
    iterable of ``(name, cpu_tensor)`` -- a safetensors shard walk, in
    whatever order the shards happen to hold, which is why biases are stashed
    and applied at the end rather than assumed to follow their weight.

    Returns a receipt with per-device byte counts and the coded/raw census.
    """
    if layout != "mma16":
        raise ValueError(
            f"the TBE fragment kernel requires layout='mma16', got {layout!r}"
        )
    dmap = _normalize_device_map(device_map, model)
    if max_conversion_source_bytes is not None:
        value = max_conversion_source_bytes
        if (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise TBEStreamLoadError(
                "invalid_conversion_source_cap",
                f"max_conversion_source_bytes must be a positive integer, got {value!r}",
            )
        max_conversion_source_bytes = int(value)
    registry = registry if registry is not None else TBEPoolRegistry()
    real_names = _real_tensor_names(model)
    ignore_patterns = _ignore_unexpected_patterns(model)

    pending_bias: Dict[str, torch.Tensor] = {}
    coded_linears: Dict[str, nn.Module] = {}
    seen: List[str] = []
    ignored_unexpected: List[str] = []
    n_coded = n_raw = n_banks = n_experts = 0
    coded_bytes = raw_bytes = dense_bytes = 0
    per_device: Dict[str, Dict[str, int]] = {}

    def _bump(dev: str, field: str, value: int) -> None:
        slot = per_device.setdefault(
            dev, {"coded_bytes": 0, "raw_bytes": 0, "dense_bytes": 0, "n_tensors": 0},
        )
        slot[field] += int(value)

    for raw_name, tensor in tensors:
        raw_name = str(raw_name)
        seen.append(raw_name)

        # The module tree is walked against the RESOLVED real name (the
        # skeleton's own names), which is where a wrapped/rewritten
        # checkpoint diverges from its raw safetensors keys.  A name that
        # resolves to nothing AND matches no pattern the class itself
        # declared as an expected absence (a tower this skeleton was never
        # built to hold, e.g. Qwen3.5's vision/MTP prefixes over a
        # text-only skeleton) is a genuine refusal, not a skip.
        name = resolve_real_name(raw_name, real_names)
        if name is None:
            if any(re.match(p, raw_name) for p in ignore_patterns):
                ignored_unexpected.append(raw_name)
                continue
            raise TBEStreamLoadError(
                "unknown_module",
                f"{raw_name!r}: does not address any parameter or buffer of "
                "the model -- checked the literal path, every way of "
                "dropping up to 2 interior, non-numeric path segments, and "
                "the model's own _keys_to_ignore_on_load_unexpected "
                "patterns; this checkpoint tensor has no home in this "
                "skeleton",
            )

        # The device map is planned and resolved against the RAW checkpoint
        # name -- both come from the same safetensors keys.
        try:
            device = resolve_device(raw_name, dmap)
        except TBEDeviceMapError as exc:
            raise TBEStreamLoadError("unmapped_tensor", str(exc)) from exc
        dev_name = str(device)
        parent, attr = split_parent(model, name)
        dense_tensor_bytes = int(tensor.numel()) * int(tensor.element_size())
        is_expert = experts and attr in parent._parameters and expert_bank_is_codable(
            name, tensor, min_numel,
        )
        source_bytes = (dense_tensor_bytes // int(tensor.shape[0])) if is_expert else dense_tensor_bytes
        if max_conversion_source_bytes is not None and source_bytes > max_conversion_source_bytes:
            raise TBEStreamLoadError(
                "conversion_source_cap_exceeded",
                f"{raw_name!r} requires {source_bytes} source bytes against "
                f"max_conversion_source_bytes={max_conversion_source_bytes}",
            )
        _bump(dev_name, "n_tensors", 1)
        _bump(dev_name, "dense_bytes", dense_tensor_bytes)
        dense_bytes += dense_tensor_bytes

        if _is_linear_weight(parent, attr) and linear_is_codable(
            name, tensor, min_numel,
        ):
            owner = name.rpartition(".")[0]
            module = convert_linear(
                tensor, pending_bias.pop(owner + ".bias", None),
                device, name=name, layout=layout,
                pool=registry.pool_for(device, "linear"),
            )
            grandparent, leaf = split_parent(model, owner)
            setattr(grandparent, leaf, module)
            coded_linears[owner] = module
            n_coded += 1
            coded_bytes += int(getattr(module, "resident_bytes", 0))
            _bump(dev_name, "coded_bytes", int(getattr(module, "resident_bytes", 0)))
            continue

        if experts and attr in parent._parameters and expert_bank_is_codable(
            name, tensor, min_numel,
        ):
            bank = convert_expert_bank(
                tensor, device, name=name, layout=layout,
                pool=registry.pool_for(device, "experts"),
            )
            del parent._parameters[attr]
            setattr(parent, attr, bank)
            n_banks += 1
            n_experts += int(tensor.shape[0])
            coded_bytes += int(getattr(bank, "resident_bytes", 0))
            _bump(dev_name, "coded_bytes", int(getattr(bank, "resident_bytes", 0)))
            continue

        owner = name.rpartition(".")[0]
        if attr == "bias" and (
            owner in coded_linears or isinstance(parent, nn.Linear)
        ):
            # A bias can arrive on either side of its weight, and shard order
            # is not the caller's to choose.  If the weight is already coded
            # the bias goes straight onto the coded module; otherwise it is
            # stashed and settled once the walk is done.  Biases are vectors
            # -- holding them all is kilobytes, not a second copy of anything.
            placed = tensor.detach().to(device)
            coded = coded_linears.get(owner)
            if coded is not None:
                coded.bias = nn.Parameter(placed, requires_grad=False)
            else:
                pending_bias[name] = placed
            continue

        place_raw(parent, attr, tensor, device)
        n_raw += 1
        raw_bytes += int(tensor.numel()) * int(tensor.element_size())
        _bump(dev_name, "raw_bytes", tensor.numel() * tensor.element_size())

    for bias_name, bias in list(pending_bias.items()):
        owner = bias_name.rpartition(".")[0]
        module = coded_linears.get(owner)
        if module is not None:
            module.bias = nn.Parameter(bias, requires_grad=False)
            del pending_bias[bias_name]
            continue
        parent, attr = split_parent(model, bias_name)
        place_raw(parent, attr, bias, bias.device)
        n_raw += 1
        del pending_bias[bias_name]

    if hasattr(model, "tie_weights"):
        try:
            model.tie_weights()
        except Exception:  # pragma: no cover - model-specific, never fatal here
            pass

    residual_meta = [
        n for n, p in model.named_parameters() if p is not None and p.is_meta
    ] + [
        n for n, b in model.named_buffers() if b is not None and b.is_meta
    ]
    if residual_meta and strict_meta:
        raise TBEStreamLoadError(
            "unfilled_parameters",
            f"{len(residual_meta)} parameter(s)/buffer(s) were never supplied "
            f"by the checkpoint stream and are still on meta: "
            f"{sorted(residual_meta)[:8]}",
        )

    return {
        "schema": TBE_STREAM_RECEIPT_SCHEMA,
        "status": "ok",
        "layout": layout,
        "min_numel": int(min_numel),
        "experts": bool(experts),
        "n_tensors_seen": len(seen),
        "n_ignored_unexpected_tensors": len(ignored_unexpected),
        "ignored_unexpected_tensor_names": sorted(ignored_unexpected)[:8],
        "n_coded_linears": n_coded,
        "n_expert_banks": n_banks,
        "n_experts_coded": n_experts,
        "n_raw_tensors": n_raw,
        "coded_resident_bytes": int(coded_bytes),
        "raw_resident_bytes": int(raw_bytes),
        "dense_equivalent_bytes": int(dense_bytes),
        "resident_bytes": int(coded_bytes + raw_bytes),
        "ratio": (
            dense_bytes / (coded_bytes + raw_bytes)
            if (coded_bytes + raw_bytes) else 1.0
        ),
        "per_device": per_device,
        "device_map": dict(dmap),
        "residual_meta_parameters": sorted(residual_meta),
        "transient_pool_capacity_bytes_by_device": (
            registry.capacity_bytes_by_device()
        ),
    }


def iter_safetensors_shards(
    model_dir, *, device: str = "cpu",
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Walk ``*.safetensors`` shards one tensor at a time.

    One shard handle is open at a time and one tensor is materialised at a
    time: the host high-water mark is a single tensor, not a shard and never
    the checkpoint.  Duplicate names across shards are yielded once.
    """
    from pathlib import Path

    from safetensors import safe_open

    model_dir = Path(model_dir)
    shards = (
        [model_dir] if model_dir.is_file() and model_dir.suffix == ".safetensors"
        else sorted(model_dir.glob("*.safetensors"))
    )
    if not shards:
        raise TBEStreamLoadError(
            "no_shards", f"no .safetensors shards found under {model_dir}",
        )
    seen = set()
    for shard in shards:
        with safe_open(str(shard), framework="pt", device=device) as handle:
            for name in handle.keys():
                if name in seen:
                    continue
                seen.add(name)
                yield name, handle.get_tensor(name)


def build_meta_skeleton(reference_dir, *, dtype=torch.bfloat16):
    """A parameter-free model skeleton on ``meta``, plus its tokenizer.

    ``from_config`` under ``torch.device("meta")`` allocates no storage at
    all, so a 360 GB architecture costs nothing to instantiate; the streaming
    walk then fills it tensor by tensor.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(str(reference_dir))
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    model.eval()
    return model


def stream_load_tbe_model(
    reference_dir,
    device_map: Any,
    *,
    min_numel: int = DEFAULT_MIN_NUMEL,
    layout: str = "mma16",
    experts: bool = True,
    dtype=torch.bfloat16,
    **kwargs,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Build, stream, place and certify a TBE model without ever going dense.

    Returns ``(model, receipt)``.  The receipt is attached to the model as
    ``georefine_tbe_serving_receipt`` in the same shape
    ``enable_hf_tbe_serving`` attaches, so every downstream consumer (the
    certify harness included) reads one schema whichever path built the
    model.
    """
    if device_map is None:
        raise TBEStreamLoadError(
            "no_device_map",
            "the streaming loader places per tensor and needs an explicit "
            "device map; plan one with "
            "tbe_device_map.plan_device_map_from_manifest()",
        )
    model = build_meta_skeleton(reference_dir, dtype=dtype)
    registry = TBEPoolRegistry()
    receipt = stream_place_tensors(
        model,
        iter_safetensors_shards(reference_dir),
        device_map,
        min_numel=min_numel,
        layout=layout,
        experts=experts,
        registry=registry,
        **kwargs,
    )
    receipt = {
        **receipt,
        "schema": TBE_SERVING_RECEIPT_SCHEMA,
        "stream_schema": TBE_STREAM_RECEIPT_SCHEMA,
        "loader": "stream",
        "n_swapped": receipt["n_coded_linears"],
        "n_expert_banks_swapped": receipt["n_expert_banks"],
        "all_bitwise_ok": True,
        "n_bitwise_failed": 0,
        "transient_pool_capacity_bytes": registry.total_capacity_bytes,
        "status": "enabled",
        "failure": None,
    }
    model.georefine_tbe_serving_receipt = receipt
    model.georefine_tbe_pool_registry = registry
    return model, receipt


def stream_load_tbe_model_standalone(
    artifact,
    device_map: Any,
    *,
    min_numel: int = DEFAULT_MIN_NUMEL,
    experts: bool = True,
    dtype=None,
    **kwargs,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Stream a ``georefine.tbe.v2`` artifact with NO dense checkpoint.

    The v1 sibling above, :func:`stream_load_tbe_model`, takes a DENSE
    reference directory and uses it twice: ``build_meta_skeleton`` reads its
    ``config.json`` for the architecture, and ``iter_safetensors_shards``
    walks its shards for the weights -- the stored containers are never
    opened at all, so a v1 artifact is unloadable without the checkpoint it
    was made from.  This function takes the ARTIFACT: the skeleton is built
    from the container's own ``config.json`` and the weights come out of the
    container's own two blobs.  There is no parameter here through which a
    dense checkpoint could be supplied.

    ``artifact`` is a ``tbe_artifact.TBEArtifact`` or a path to a v2
    directory.  Returns ``(model, receipt)`` in exactly the shape
    :func:`stream_load_tbe_model` returns, so every downstream consumer --
    the certify harness included -- reads one schema whichever path built the
    model.

    PEAK.  One dense tensor plus its container, the same high-water mark the
    v1 streaming path has.  Two full weight sets are never resident: the
    skeleton holds no storage until a tensor is placed, and each decoded
    tensor is dropped as soon as it has been encoded onto its device.
    """
    from .tbe_artifact import (
        TBEArtifactError,
        build_meta_skeleton_from_artifact,
        is_tbe_artifact,
        iter_artifact_tensors,
        open_tbe_artifact,
    )

    if not is_tbe_artifact(artifact):
        artifact = open_tbe_artifact(artifact)

    if device_map is None:
        raise TBEStreamLoadError(
            "no_device_map",
            "the streaming loader places per tensor and needs an explicit "
            "device map; plan one with "
            "tbe_device_map.plan_device_map_from_manifest()",
        )

    layout = artifact.layout
    if layout != "mma16":
        raise TBEArtifactError(
            "layout_not_servable_here",
            f"{artifact.path} was transcoded with layout {layout!r}; the "
            "torch/CUDA fragment-kernel serving path requires 'mma16' (see "
            "stream_place_tensors' own guard). A 'flat64' artifact is served "
            "by the Metal arm: "
            "glc_loader.metal.tbe_mlx_model.load_tbe_model_standalone.",
        )

    if dtype is None:
        dtype = torch.bfloat16
    model = build_meta_skeleton_from_artifact(artifact, dtype=dtype)
    registry = TBEPoolRegistry()
    receipt = stream_place_tensors(
        model,
        iter_artifact_tensors(artifact),
        device_map,
        min_numel=min_numel,
        layout=layout,
        experts=experts,
        registry=registry,
        **kwargs,
    )
    receipt = {
        **receipt,
        "schema": TBE_SERVING_RECEIPT_SCHEMA,
        "stream_schema": TBE_STREAM_RECEIPT_SCHEMA,
        "loader": "stream_standalone",
        "source": "container",
        "artifact_dir": str(artifact.path),
        "artifact_format": artifact.artifact_format,
        "dense_reference_read": False,
        "n_swapped": receipt["n_coded_linears"],
        "n_expert_banks_swapped": receipt["n_expert_banks"],
        "all_bitwise_ok": True,
        "n_bitwise_failed": 0,
        "transient_pool_capacity_bytes": registry.total_capacity_bytes,
        "status": "enabled",
        "failure": None,
    }
    model.georefine_tbe_serving_receipt = receipt
    model.georefine_tbe_pool_registry = registry
    return model, receipt


__all__ = [
    "TBE_STREAM_RECEIPT_SCHEMA",
    "TBEStreamLoadError",
    "build_meta_skeleton",
    "default_convert_expert_bank",
    "default_convert_linear",
    "default_place_raw",
    "expert_bank_is_codable",
    "iter_safetensors_shards",
    "linear_is_codable",
    "resolve_real_name",
    "split_parent",
    "stream_load_tbe_model",
    "stream_load_tbe_model_standalone",
    "stream_place_tensors",
]
