"""Native Torch loading for verified typed context packages.

This module is intentionally separate from the legacy BF16 serving path.  It
constructs a fresh model, performs all reconciliation and resource checks
before decoding, and installs each source tensor with its original storage
dtype.  It never creates a dense state-dict checkpoint.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any, Iterable, Mapping

from .safetensors_source import SafetensorsResolver
from .typed_context_package import TypedContextResolver


class TypedTorchServingError(ValueError):
    """The package cannot be installed into the requested Torch model."""


def _named(module: Any, kind: str) -> dict[str, Any]:
    fn = getattr(module, kind)
    try:
        return dict(fn(remove_duplicate=False))
    except TypeError:
        return dict(fn())


def _parent_and_leaf(model: Any, name: str) -> tuple[Any, str]:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if isinstance(parent, (list, tuple)) else getattr(parent, part)
    return parent, parts[-1]


def _source_bytes(tensor: Any) -> bytes:
    """Read the exact CPU storage bytes, including NaNs and signed zero."""
    import torch

    value = tensor.detach().contiguous()
    if value.device.type != "cpu":
        value = value.cpu()
    return value.reshape(-1).view(torch.uint8).numpy().tobytes()


def _framework_dtype(token: str, torch: Any) -> Any:
    try:
        from safetensors.torch import _getdtype
        return _getdtype(token)
    except (ImportError, KeyError, TypeError, ValueError):
        raise TypedTorchServingError(f"unsupported Torch storage dtype {token!r}")


def _read_config(resolver: Any) -> Any:
    if isinstance(resolver, SafetensorsResolver):
        return resolver.pinned_config()
    from .typed_context_package import read_asset
    assets = resolver.manifest.get("assets", ())
    pin = next((row for row in assets if row.get("source_path") == "config.json"), None)
    if pin is None:
        raise TypedTorchServingError("typed package has no original model config")
    try:
        raw = read_asset(resolver, "config.json")
    except (OSError, ValueError) as exc:
        raise TypedTorchServingError("cannot read verified model config") from exc
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TypedTorchServingError("verified model config is invalid JSON") from exc


def _alias_groups(named: Mapping[str, Any]) -> list[tuple[str, ...]]:
    groups: dict[int, list[str]] = {}
    for name, value in named.items():
        groups.setdefault(id(value), []).append(name)
    return [tuple(names) for names in groups.values() if len(names) > 1]


def load_verified_torch(
    package: Any,
    *,
    model_factory,
    config: Any = None,
    source_aliases: Mapping[str, str] | None = None,
    allowed_unused: Iterable[str] = (),
    allowed_generated_buffers: Iterable[str] = (),
    max_tensor_bytes: int = 256 * 1024 * 1024,
    max_resident_bytes: int = 8 * 1024**3,
    max_source_bytes: int | None = None,
    max_frame_bytes: int = 512 * 1024 * 1024,
):
    """Load a typed package into a fresh native Torch model.

    ``model_factory`` is always called with a new config object.  Package
    entries and model parameters/buffers must reconcile exactly, except for
    source names explicitly listed in ``allowed_unused``.  All expensive
    materialization occurs only after preflight has passed. Config-derived,
    non-persistent buffers absent from the source require an explicit
    ``allowed_generated_buffers`` list; their bytes are budgeted and must stay
    unchanged during installation.
    """
    if not callable(model_factory):
        raise TypeError("model_factory must be callable")
    if type(max_tensor_bytes) is not int or max_tensor_bytes < 0:
        raise ValueError("max_tensor_bytes must be a non-negative integer")
    if type(max_resident_bytes) is not int or max_resident_bytes < 0:
        raise ValueError("max_resident_bytes must be a non-negative integer")
    if max_source_bytes is not None and (type(max_source_bytes) is not int or max_source_bytes < 0):
        raise ValueError("max_source_bytes must be a non-negative integer")
    if type(max_frame_bytes) is not int or max_frame_bytes < 0:
        raise ValueError("max_frame_bytes must be a non-negative integer")
    if max_source_bytes is not None:
        max_tensor_bytes = min(max_tensor_bytes, max_source_bytes)
    alias_map = dict(source_aliases or {})
    if any(not isinstance(k, str) or not isinstance(v, str) or not k or not v for k, v in alias_map.items()):
        raise TypeError("source_aliases must map non-empty target names to source names")
    allowed = set(allowed_unused)
    if any(not isinstance(name, str) for name in allowed):
        raise TypeError("allowed_unused must contain tensor names")
    generated = set(allowed_generated_buffers)
    if any(not isinstance(name, str) or not name for name in generated):
        raise TypeError("allowed_generated_buffers must contain nonempty buffer names")

    trusted_resolvers = (TypedContextResolver, SafetensorsResolver)
    if not isinstance(package, trusted_resolvers) and not isinstance(package, (str, bytes, os.PathLike)):
        # Path-like inputs retain the existing typed-package default.  Any
        # resolver-like object supplied by a caller must be one of the two
        # audited implementations above; duck-typed objects are rejected.
        raise TypeError("package must be a path, TypedContextResolver, or SafetensorsResolver")
    owns_resolver = not isinstance(package, trusted_resolvers)
    if owns_resolver:
        try:
            resolver = TypedContextResolver(package, max_source_bytes=max_tensor_bytes,
                                            max_frame_bytes=max_frame_bytes)
        except ValueError as exc:
            if "exceeds max_source_bytes" in str(exc):
                raise MemoryError(str(exc)) from exc
            raise
    else:
        resolver = package
    try:
        entries = {name: resolver.entry(name) for name in resolver.tensor_names}
        model_config = copy.deepcopy(config if config is not None else _read_config(resolver))
        model = model_factory(model_config)
        if model is None:
            raise TypedTorchServingError("model_factory returned None")
        params = _named(model, "named_parameters")
        buffers = _named(model, "named_buffers")
        generated_pins = {}
        generated_total = 0
        generated_objects = set()
        if generated:
            modules = dict(model.named_modules())
            for name in sorted(generated):
                if name not in buffers or name in entries or name in params:
                    raise TypedTorchServingError(f"generated buffer must be absent from source parameters: {name}")
                parent_name, _, leaf = name.rpartition(".")
                parent = modules.get(parent_name)
                if parent is None or leaf not in parent._non_persistent_buffers_set:
                    raise TypedTorchServingError(f"generated buffer must be explicitly non-persistent: {name}")
                value = buffers[name]
                if value.is_meta:
                    raise TypedTorchServingError(f"generated buffer must be constructed from config: {name}")
                raw = _source_bytes(value)
                generated_pins[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                if id(value) not in generated_objects:
                    generated_total += len(raw)
                    generated_objects.add(id(value))
            buffers = {name: value for name, value in buffers.items() if name not in generated}
        named = dict(params)
        overlap = set(params) & set(buffers)
        if overlap:
            raise TypedTorchServingError(f"parameter/buffer name collision: {sorted(overlap)!r}")
        named.update(buffers)

        names = set(entries)
        model_names = set(named)
        missing = sorted(model_names - names - set(alias_map))
        if missing:
            raise TypedTorchServingError(f"model tensors missing from package: {missing!r}")
        if not allowed.issubset(names):
            raise TypedTorchServingError("allowed_unused names must be package tensor names")
        if set(alias_map) & allowed or set(alias_map.values()) & allowed:
            raise TypedTorchServingError("allowed_unused overlaps consumed source aliases")
        for target, source in alias_map.items():
            if target not in model_names or source not in model_names:
                raise TypedTorchServingError(f"source alias is not a model tensor: {target} -> {source}")
            if target == source or id(named[target]) != id(named[source]):
                raise TypedTorchServingError(f"source alias must name a tied skeleton: {target} -> {source}")
        source_for = {name: alias_map.get(name, name) for name in model_names}
        consumed = set(source_for.values())
        if not consumed.issubset(names):
            raise TypedTorchServingError(f"aliased source tensors missing from package: {sorted(consumed - names)!r}")
        if consumed & allowed:
            raise TypedTorchServingError("allowed_unused overlaps consumed source tensors")
        if set(names) - consumed - allowed:
            raise TypedTorchServingError(f"package tensors are unused without declaration: {sorted(set(names) - consumed - allowed)!r}")

        import torch
        rows = []
        total = generated_total
        if total > max_resident_bytes:
            raise MemoryError("config-derived buffers exceed resident byte budget")
        budget_objects: set[int] = set()
        for name in sorted(model_names):
            source_name = source_for[name]
            row = entries[source_name]
            actual = row.get("actual_source_bytes", row.get("source_bytes"))
            if type(actual) is not int or actual < 0:
                raise TypedTorchServingError(f"invalid source byte count for {name}")
            if actual > max_tensor_bytes:
                raise MemoryError(f"tensor {name} exceeds per-tensor byte budget")
            object_id = id(named[name])
            if object_id not in budget_objects:
                total += actual
                budget_objects.add(object_id)
                if total > max_resident_bytes:
                    raise MemoryError("resident typed tensors exceed byte budget")
            expected = _framework_dtype(row.get("dtype"), torch)
            shape = tuple(row.get("shape", ()))
            if any(type(dim) is not int or dim < 0 for dim in shape):
                raise TypedTorchServingError(f"invalid source shape for {name}")
            target = named[name]
            if target.dtype != expected or tuple(target.shape) != shape:
                raise TypedTorchServingError(f"model dtype/shape mismatch before decode: {name}")
            rows.append((name, source_name, row, expected, shape, actual))

        # Check tied aliases before decoding. Every package entry named by an
        # alias must carry identical source metadata and bytes.
        for tied_names in _alias_groups(named):
            alias_rows = [entries[name] for name in tied_names if name in entries]
            if any(name not in entries and name not in alias_map for name in tied_names):
                raise TypedTorchServingError(f"tied alias coverage requires explicit source_aliases: {tied_names!r}")
            fingerprints = {(r.get("source_sha256"), r.get("dtype"), tuple(r.get("shape", ())), r.get("actual_source_bytes")) for r in alias_rows}
            if len(fingerprints) > 1:
                raise TypedTorchServingError(f"tied aliases do not have identical source bytes: {tied_names!r}")

        receipts = []
        installed: dict[int, Any] = {}
        materialized: dict[str, Any] = {}
        for name, source_name, row, expected, shape, actual in rows:
            tensor = materialized.get(source_name)
            if tensor is None:
                tensor = resolver.materialize_torch(source_name)
                materialized[source_name] = tensor
            if tensor.dtype != expected or tuple(tensor.shape) != shape:
                raise TypedTorchServingError(f"materialized dtype/shape mismatch: {name}")
            if tensor.device.type != "cpu":
                tensor = tensor.cpu()
            if tensor.numel() * tensor.element_size() != actual:
                raise TypedTorchServingError(f"materialized byte count mismatch: {name}")
            raw = _source_bytes(tensor)
            digest = hashlib.sha256(raw).hexdigest()
            if digest != row.get("source_sha256"):
                raise TypedTorchServingError(f"source SHA mismatch: {name}")
            parent, leaf = _parent_and_leaf(model, name)
            if name in params:
                old = params[name]
                replacement = installed.get(id(old))
                if replacement is None:
                    replacement = torch.nn.Parameter(tensor, requires_grad=old.requires_grad)
                    installed[id(old)] = replacement
                parent._parameters[leaf] = replacement
            else:
                old = buffers[name]
                replacement = installed.get(id(old))
                if replacement is None:
                    replacement = tensor
                    installed[id(old)] = replacement
                parent._buffers[leaf] = replacement
            receipts.append({"name": name, "source_name": source_name, "source_sha256": digest,
                             "source_dtype": row["dtype"], "torch_dtype": str(tensor.dtype),
                             "shape": list(shape), "actual_source_bytes": actual})
        resolver.verify_integrity()
        if generated:
            current_buffers = _named(model, "named_buffers")
            for name, pin in generated_pins.items():
                raw = _source_bytes(current_buffers[name])
                if len(raw) != pin['bytes'] or hashlib.sha256(raw).hexdigest() != pin['sha256']:
                    raise TypedTorchServingError(f"config-derived buffer changed during installation: {name}")
        result = {"package_format": "safetensors" if isinstance(resolver, SafetensorsResolver) else "typed", "resident_dense_bytes": total,
                  "tensors": receipts, "allowed_unused": sorted(allowed)}
        if generated:
            result['generated_buffers'] = generated_pins
        if owns_resolver:
            resolver.close()
        return model, result
    except BaseException:
        if owns_resolver:
            try:
                resolver.close()
            except BaseException:
                pass
        raise


__all__ = ["TypedTorchServingError", "load_verified_torch"]
