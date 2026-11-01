"""Explicit typed Torch -> CUDA FastSession adapter.

The adapter deliberately requires a tokenizer and an explicit list of BHE
projection names.  It never invents a tokenizer, casts model parameters, or
selects projections by heuristic.  The legacy server/router does not call it.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Iterable

import torch

from .fastdec import FastDecoder, load_tune
from .fastserve import FastSession
from .engine import Engine
from .typed_predictive_torch import load_verified_qwen_torch
from .bhe_desc import BHELinear, encode_matrix


class TypedCudaFastSessionError(ValueError):
    """Typed model cannot be admitted to the CUDA FastSession path."""


def _named_modules(model: Any) -> dict[str, Any]:
    return dict(model.named_modules())


def _parent_and_leaf(model: Any, name: str) -> tuple[Any, str]:
    parent_name, _, leaf = name.rpartition(".")
    parent = model
    for part in parent_name.split(".") if parent_name else ():
        parent = parent[int(part)] if isinstance(parent, (list, tuple)) else getattr(parent, part)
    return parent, leaf


def _replace_selected_bf16_linears(model: Any, names: Iterable[str], *, device: str,
                                   receipt: dict[str, Any], max_resident_bytes: int | None = None,
                                   resident_dense_bytes: int | None = None) -> list[tuple[Any, ...]]:
    selected = tuple(names)
    if not selected or len(set(selected)) != len(selected):
        raise TypedCudaFastSessionError("bhe_linear_names must contain unique explicit names")
    rows = {row.get("name"): row for row in receipt.get("tensors", ())
            if isinstance(row, dict) and isinstance(row.get("name"), str)}
    modules = _named_modules(model)
    plan = []
    for name in selected:
        module = modules.get(name)
        row = rows.get(f"{name}.weight")
        if module is None or not isinstance(module, torch.nn.Linear):
            raise TypedCudaFastSessionError(f"selected BHE module is not a named torch Linear: {name}")
        if row is None:
            raise TypedCudaFastSessionError(f"selected BHE source is absent from typed inventory: {name}")
        if row.get("source_dtype") not in ("BF16", "bfloat16"):
            raise TypedCudaFastSessionError(f"selected BHE source is not BF16: {name}")
        if module.weight.dtype != torch.bfloat16:
            raise TypedCudaFastSessionError(f"selected BHE module dtype is not BF16: {name}")
        if module.bias is not None:
            raise TypedCudaFastSessionError(f"selected BHE projection has unsupported bias: {name}")
        actual = row.get("actual_source_bytes")
        if type(actual) is not int or actual != module.weight.numel() * 2:
            raise TypedCudaFastSessionError(f"selected BHE source byte count disagrees: {name}")
        plan.append((name, module, row, actual))
    if max_resident_bytes is not None and resident_dense_bytes is not None:
        source_scratch = sum(item[3] for item in plan)
        if resident_dense_bytes + source_scratch > max_resident_bytes:
            raise MemoryError("typed BHE source scratch and resident bytes exceed budget")
    return plan


def _install_selected_bf16_linears(model: Any, plan: list[tuple[Any, ...]], *, device: str,
                                    max_resident_bytes: int, resident_dense_bytes: int,
                                    prefill_mode: str = 'fused',
                                    max_prefill_dense_bytes: int | None = None) -> dict[str, Any]:
    encoded = []
    scratch = 0
    wire = 0
    for name, module, row, actual in plan:
        scratch += actual
        blobs, descriptor = encode_matrix(module.weight.detach().cpu())
        encoded_wire = sum(len(blob) for blob in blobs)
        wire += encoded_wire
        if resident_dense_bytes + scratch + wire > max_resident_bytes:
            raise MemoryError("typed BHE conversion scratch and resident bytes exceed budget")
        encoded.append((name, module, blobs, descriptor, encoded_wire))
    converted = []
    for name, module, blobs, descriptor, encoded_wire in encoded:
        replacement = BHELinear(blobs, descriptor, device=device)
        parent, leaf = _parent_and_leaf(model, name)
        setattr(parent, leaf, replacement)
        converted_row = {"name": name, "shape": list(module.weight.shape),
                          "source_dtype": "BF16", "runtime_kind": "bhe1_linear",
                          "wire_bytes": encoded_wire}
        converted.append(converted_row)
    result = {"runtime_format": "typed_bhe1", "linear_count": len(converted),
            "tensors": converted, "gdn_native_dtypes": True,
            "resident_dense_bytes": resident_dense_bytes,
            "conversion_scratch_bytes": scratch, "bhe_wire_bytes": wire}
    return result


def _configure_prefill_modules(model: Any, names: Iterable[str], *, mode: str,
                               max_dense_bytes: int | None) -> dict[str, Any] | None:
    if mode == 'fused':
        return None
    requirements = []
    modules = _named_modules(model)
    for name in names:
        module = modules.get(name)
        if module is None or not hasattr(module, 'configure_prefill'):
            raise TypedCudaFastSessionError(f'BHE module lacks prefill configuration seam: {name}')
        requirements.append(module.configure_prefill(mode, max_dense_bytes=max_dense_bytes))
    peak = max(row['total_bytes'] for row in requirements)
    aggregate = sum(row['total_bytes'] for row in requirements)
    return {'prefill_mode': mode, 'max_prefill_dense_bytes': max_dense_bytes,
            'prefill_dense_temporary_bytes': peak,
            'prefill_dense_temporary_aggregate_bytes': aggregate,
            'prefill_dense_requirements': requirements}


def load_typed_cuda_fast_session(
    package: Any,
    *,
    tune: str,
    tokenizer: Any,
    processor: Any = None,
    device: str = "cuda:0",
    max_len: int = 32768,
    gdn_hv16: bool = False,
    model_factory: Callable | None = None,
    mtp_factory: Callable | None = None,
    bhe_linear_names: Iterable[str],
    max_tensor_bytes: int = 256 * 1024 * 1024,
    max_resident_bytes: int = 8 * 1024**3,
    max_source_bytes: int | None = None,
    max_frame_bytes: int = 512 * 1024 * 1024,
    prefill_mode: str = 'fused',
    max_prefill_dense_bytes: int | None = None,
    decode_mode: str = 'fused',
    max_decode_dense_bytes: int | None = None,
    log=print,
):
    """Load a verified typed package into the opt-in CUDA FastSession path.

    ``tokenizer`` and ``bhe_linear_names`` are required to prevent implicit
    asset selection and heuristic compression.  FP32/control tensors remain
    exactly as installed by the typed loader.  Images, MTP, and graph capture
    remain supported only where the supplied model/processor/FastDecoder
    implementation supports them; unsupported cases fail during construction.
    """
    if type(gdn_hv16) is not bool:
        raise TypedCudaFastSessionError("gdn_hv16 must be boolean")
    if torch.device(device).type != "cuda":
        raise TypedCudaFastSessionError("typed CUDA FastSession requires a CUDA device")
    if tokenizer is None:
        raise TypedCudaFastSessionError("typed CUDA FastSession requires an explicit tokenizer")
    if not isinstance(tune, str) or not tune:
        raise TypedCudaFastSessionError("typed CUDA FastSession requires an explicit tune path")
    if prefill_mode not in ('fused', 'dense_temporary'):
        raise TypedCudaFastSessionError("prefill_mode must be 'fused' or 'dense_temporary'")
    if prefill_mode == 'dense_temporary' and (type(max_prefill_dense_bytes) is not int or max_prefill_dense_bytes <= 0):
        raise TypedCudaFastSessionError("dense_temporary requires a positive max_prefill_dense_bytes")
    if decode_mode not in ('fused', 'dense_temporary'):
        raise TypedCudaFastSessionError("decode_mode must be 'fused' or 'dense_temporary'")
    if decode_mode == 'dense_temporary' and (type(max_decode_dense_bytes) is not int or max_decode_dense_bytes <= 0):
        raise TypedCudaFastSessionError("dense_temporary decode requires a positive max_decode_dense_bytes")
    model, receipt = load_verified_qwen_torch(
        package, model_factory=model_factory, mtp_factory=mtp_factory,
        max_tensor_bytes=max_tensor_bytes, max_resident_bytes=max_resident_bytes,
        max_source_bytes=max_source_bytes, max_frame_bytes=max_frame_bytes,
        load_mtp=True,
    )
    try:
        model.eval()
        resident_dense_bytes = receipt.get("resident_dense_bytes")
        if type(resident_dense_bytes) is not int or resident_dense_bytes < 0:
            raise TypedCudaFastSessionError("typed loader receipt lacks resident_dense_bytes")
        plan = _replace_selected_bf16_linears(
            model, bhe_linear_names, device=device, receipt=receipt,
            max_resident_bytes=max_resident_bytes, resident_dense_bytes=resident_dense_bytes)
        if getattr(model, "mtp", None) is None:
            raise TypedCudaFastSessionError("typed package MTP coverage did not produce an MTP head")
        bhe_receipt = _install_selected_bf16_linears(
            model, plan, device=device, max_resident_bytes=max_resident_bytes,
            resident_dense_bytes=resident_dense_bytes)
        model = model.to(device=device)
        loaded = SimpleNamespace(model=model, mtp=model.mtp, options=SimpleNamespace(),
                                 eos_ids=[int(tokenizer.eos_token_id)] if tokenizer.eos_token_id is not None else [],
                                 local_dir=None)
        engine = Engine(loaded, processor, tokenizer, max_batch=1, enable_mtp=False, log=log)
        engine.tokenizer = tokenizer
        tune_config = load_tune(tune)
        fd = FastDecoder(model, model.mtp, tune=tune_config, max_len=max_len, device=device,
                         gdn_hv16=gdn_hv16, gdn_native_dtypes=True,
                         decode_mode=decode_mode, max_decode_dense_bytes=max_decode_dense_bytes)
        prefill_receipt = _configure_prefill_modules(
            model, [name for name, _module, _row, _actual in plan],
            mode=prefill_mode, max_dense_bytes=max_prefill_dense_bytes)
        if prefill_receipt is not None:
            bhe_receipt.update(prefill_receipt)
        meta = {"source_kind": "typed_torch", "runtime_format": "typed_bhe1",
                "typed_load_receipt": receipt, "bhe_conversion": bhe_receipt,
                "gdn_native_dtypes": True, "gdn_hv16": gdn_hv16}
        if prefill_mode != 'fused':
            meta.update(prefill_receipt)
        if decode_mode != 'fused':
            meta['decode_mode'] = decode_mode
            meta['max_decode_dense_bytes'] = max_decode_dense_bytes
            meta['decode_dense_temporary_bytes'] = fd.decode_dense_requirements['peak_bytes']
            meta['decode_dense_temporary_aggregate_bytes'] = fd.decode_dense_requirements['aggregate_bytes']
            meta['decode_dense_requirements'] = fd.decode_dense_requirements['descriptors']
        return FastSession(engine, fd, meta)
    except Exception as exc:
        raise TypedCudaFastSessionError("typed CUDA FastSession construction failed") from exc


__all__ = ["TypedCudaFastSessionError", "load_typed_cuda_fast_session"]
