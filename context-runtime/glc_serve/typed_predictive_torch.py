"""Explicit opt-in Torch loader for verified typed Qwen3.5 packages.

This adapter is deliberately outside the frozen HTTP servers.  It builds the
Hugging Face skeleton on ``meta`` (with real, config-derived buffers), then
hands installation to :func:`load_verified_torch`.  No dense state dict is
created and source tensors are installed in their recorded storage dtype.
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Mapping
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn

from georefine_context_runtime.typed_context_package import TypedContextResolver
from georefine_context_runtime.typed_context_serving import (
    TypedTorchServingError,
    _framework_dtype,
    load_verified_torch,
)


GENERATED_QWEN_BUFFERS = (
    "model.visual.rotary_pos_emb.inv_freq",
    "model.language_model.rotary_emb.inv_freq",
    "model.language_model.rotary_emb.original_inv_freq",
)


def _named(module: Any, kind: str) -> dict[str, Any]:
    fn = getattr(module, kind)
    try:
        return dict(fn(remove_duplicate=False))
    except TypeError:
        return dict(fn())


def _parent_and_leaf(model: Any, name: str):
    parent_name, _, leaf = name.rpartition(".")
    parent = model
    for part in parent_name.split(".") if parent_name else ():
        parent = parent[int(part)] if isinstance(parent, (list, tuple)) else getattr(parent, part)
    return parent, leaf


def _set_meta_dtypes(model: Any, entries: Mapping[str, Mapping[str, Any]]) -> None:
    """Apply source dtypes to meta parameters, never to materialized tensors."""
    params = _named(model, "named_parameters")
    groups: dict[int, list[tuple[str, Any]]] = {}
    for name, parameter in params.items():
        groups.setdefault(id(parameter), []).append((name, parameter))
    for members in groups.values():
        rows = [entries[name] for name, _ in members if name in entries]
        if not rows:
            continue
        dtypes = {_framework_dtype(row["dtype"], torch) for row in rows}
        if len(dtypes) != 1:
            raise TypedTorchServingError(f"tied parameter sources have conflicting dtypes: {[n for n, _ in members]!r}")
        replacement_dtype = next(iter(dtypes))
        original = members[0][1]
        if not original.is_meta:
            raise TypedTorchServingError(f"Qwen skeleton parameter is materialized: {members[0][0]}")
        replacement = nn.Parameter(torch.empty(tuple(original.shape), dtype=replacement_dtype, device="meta"), requires_grad=original.requires_grad)
        for name, _ in members:
            parent, leaf = _parent_and_leaf(model, name)
            parent._parameters[leaf] = replacement


def _call_factory(factory, config, entries):
    """Dispatch an explicit one or two argument seam without masking errors."""
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(config)
    try:
        signature.bind(config, entries)
    except TypeError:
        signature.bind(config)
        return factory(config)
    return factory(config, entries)


def _hf_model_factory(config: Any, entries: Mapping[str, Mapping[str, Any]], *, model_factory=None):
    if model_factory is not None:
        model = _call_factory(model_factory, copy.deepcopy(config), entries)
        _set_meta_dtypes(model, entries)
        return model
    try:
        from accelerate import init_empty_weights
        from transformers import Qwen3_5Config
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    except ImportError as exc:
        raise TypedTorchServingError("Qwen3.5 Torch adapter requires transformers and accelerate") from exc
    if isinstance(config, Mapping):
        config = Qwen3_5Config.from_dict(dict(config))
    with init_empty_weights(include_buffers=False):
        model = Qwen3_5ForConditionalGeneration._from_config(copy.deepcopy(config), dtype=torch.bfloat16)
    # Tying must happen after the meta context. Transformers' meta tie helper
    # does not always preserve identity, so enforce the known upstream alias.
    model.tie_weights()
    _set_meta_dtypes(model, entries)
    if getattr(getattr(config, "text_config", config), "tie_word_embeddings", False):
        try:
            model.lm_head.weight = model.model.language_model.embed_tokens.weight
        except AttributeError as exc:
            raise TypedTorchServingError("Qwen config requests tied embeddings but skeleton paths differ") from exc
    return model


def _mtp_factory(config: Any, entries: Mapping[str, Mapping[str, Any]], *, mtp_factory=None):
    if not any(name.startswith("mtp.") for name in entries):
        return None
    if mtp_factory is not None:
        mtp = _call_factory(mtp_factory, copy.deepcopy(config), entries)
    else:
        from .loader import build_mtp_skeleton
        if isinstance(config, Mapping):
            from transformers import Qwen3_5Config
            config = Qwen3_5Config.from_dict(dict(config))
        mtp = build_mtp_skeleton(copy.deepcopy(config))
    local = _named(mtp, "named_parameters")
    for name, parameter in local.items():
        row = entries.get("mtp." + name)
        if row is None:
            raise TypedTorchServingError(f"MTP source coverage missing: mtp.{name}")
        if not parameter.is_meta:
            raise TypedTorchServingError(f"MTP skeleton parameter is materialized: {name}")
        dtype = _framework_dtype(row["dtype"], torch)
        parent_name, _, leaf = name.rpartition(".")
        parent = mtp
        for part in parent_name.split(".") if parent_name else ():
            parent = parent[int(part)] if isinstance(parent, (list, tuple)) else getattr(parent, part)
        parent._parameters[leaf] = nn.Parameter(torch.empty(tuple(parameter.shape), dtype=dtype, device="meta"), requires_grad=parameter.requires_grad)
    return mtp


def load_verified_qwen_torch(
    package: Any,
    *,
    model_factory: Callable | None = None,
    mtp_factory: Callable | None = None,
    resolver_factory: Callable = TypedContextResolver,
    max_tensor_bytes: int = 256 * 1024 * 1024,
    max_resident_bytes: int = 8 * 1024**3,
    max_source_bytes: int | None = None,
    max_frame_bytes: int = 512 * 1024 * 1024,
    allowed_generated_buffers: Iterable[str] | None = None,
    load_mtp: bool = True,
):
    """Load a verified Qwen3.5 model and its optional, verified MTP head.

    The helper is explicit opt-in and returns ``(model, receipt)``.
    The returned HF model exposes ``.mtp`` alongside its normal
    ``model.language_model``/``lm_head`` paths. ``load_mtp=False`` refuses a
    package containing MTP tensors, so source coverage cannot be silently
    discarded.
    """
    resolver_limit = max_tensor_bytes if max_source_bytes is None else min(max_tensor_bytes, max_source_bytes)
    resolver = resolver_factory(package, max_source_bytes=resolver_limit, max_frame_bytes=max_frame_bytes)
    try:
        entries = {name: resolver.entry(name) for name in resolver.tensor_names}
        mtp_names = {name for name in entries if name.startswith("mtp.")}
        if mtp_names and not load_mtp:
            raise TypedTorchServingError("MTP tensors are present; load_mtp=True is required for coverage")
        config = None
        from georefine_context_runtime.typed_context_serving import _read_config
        config = _read_config(resolver)
        # The generic loader passes a detached config; preserve that behavior
        # while allowing fixture factories to use a plain mapping.
        model_config = config

        def factory(cfg):
            model = _hf_model_factory(cfg, entries, model_factory=model_factory)
            mtp = _mtp_factory(cfg, entries, mtp_factory=mtp_factory) if load_mtp else None
            if mtp is not None:
                if hasattr(model, "mtp"):
                    raise TypedTorchServingError("model already has an mtp attribute")
                model.add_module("mtp", mtp)
            return model

        requested_generated = GENERATED_QWEN_BUFFERS if allowed_generated_buffers is None else tuple(allowed_generated_buffers)
        # Factory seams may intentionally use a small fixture. Only config
        # generated buffers actually owned by that skeleton are allowlisted.
        probe = factory(model_config)
        owned = set(_named(probe, "named_buffers"))
        generated = tuple(name for name in requested_generated if name in owned) if allowed_generated_buffers is None else requested_generated
        aliases = {}
        probe_params = _named(probe, "named_parameters")
        model_names = set(probe_params)
        if "lm_head.weight" in model_names and "model.language_model.embed_tokens.weight" in model_names:
            if id(probe_params["lm_head.weight"]) == id(probe_params["model.language_model.embed_tokens.weight"]):
                aliases["lm_head.weight"] = "model.language_model.embed_tokens.weight"
        # Rebuild through the generic loader so all checks happen before decode.
        result, receipt = load_verified_torch(
            resolver,
            model_factory=factory,
            config=model_config,
            source_aliases=aliases,
            allowed_generated_buffers=generated,
            max_tensor_bytes=max_tensor_bytes,
            max_resident_bytes=max_resident_bytes,
            max_source_bytes=max_source_bytes,
            max_frame_bytes=max_frame_bytes,
        )
        return result, receipt
    finally:
        resolver.close()


load_typed_predictive_torch = load_verified_qwen_torch

__all__ = ["GENERATED_QWEN_BUFFERS", "load_verified_qwen_torch", "load_typed_predictive_torch"]
