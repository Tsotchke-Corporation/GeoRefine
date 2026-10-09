"""Private in-memory bridge from predictive Context weights to FastSession/TBE v1.

This is an opt-in startup adapter for private speed evaluation. It resolves the
current BCTX+PPCX package, converts exact BF16 weights one tensor at a time into
TBE v1 containers, and runs the existing FastDecoder path. It does not write a
dense checkpoint or publish an artifact. Runtime weights use TBE after startup;
this is not direct PPCX serving.
"""
from __future__ import annotations

import gc
import hashlib
import importlib
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict

import torch

from ._startup_limits import MAX_CPU_ENCODE_WORKERS, cpu_resolver_cache_limit


class PredictiveTBEError(RuntimeError):
    """A verified predictive tensor could not be prepared for the TBE engine."""


def _predictive_runtime():
    """Use the installed codec namespace, or the explicit source-checkout runtime."""
    try:
        return importlib.import_module("georefine_context_runtime.bitexact_predictive_serving")
    except ModuleNotFoundError as exc:
        if exc.name != "georefine_context_runtime":
            # A present but incomplete installation must not borrow modules from
            # an unrelated checkout and appear to be a qualified wheel.
            raise
    return importlib.import_module("scripts.bitexact_predictive_serving")


def _text_manifest_names(tensor_names):
    """Select manifest entries for text blocks, token embeddings, and output head."""
    markers = ("language_model", "embed_tokens", "lm_head", "decoder.layers", "model.layers")
    return tuple(name for name in tensor_names if any(part in name.lower() for part in markers))


def _load_and_convert_predictive(
    load_model, *, package, device, attention_implementation, stride, loader_workers,
    metadata_cache_dir, log, verified_cpu_stream, cpu_encode_workers,
    source_inflight_bytes, cpu_vision_dense=False,
):
    """Load and convert with resolver lifetime spanning model construction and replacement."""
    if type(cpu_encode_workers) is not int or not 1 <= cpu_encode_workers <= MAX_CPU_ENCODE_WORKERS:
        raise PredictiveTBEError(f"cpu_encode_workers must be an integer between 1 and {MAX_CPU_ENCODE_WORKERS}")
    if not verified_cpu_stream and cpu_encode_workers != 1:
        raise PredictiveTBEError("cpu_encode_workers > 1 requires verified_cpu_stream")
    if type(cpu_vision_dense) is not bool:
        raise PredictiveTBEError("cpu_vision_dense must be a bool")
    if cpu_vision_dense and not verified_cpu_stream:
        raise PredictiveTBEError("cpu_vision_dense requires verified_cpu_stream")
    resolver = None
    cpu_names = None
    if verified_cpu_stream:
        if type(source_inflight_bytes) is not int or not 0 < source_inflight_bytes <= 8 * 1024**3:
            raise PredictiveTBEError("source_inflight_bytes must be between 1 byte and 8 GiB")
        try:
            package_api = importlib.import_module("georefine_context_runtime.bitexact_predictive_package")
            resolver_type = package_api.VerifiedTensorResolver
        except (ImportError, AttributeError) as exc:
            raise PredictiveTBEError("installed predictive runtime lacks VerifiedTensorResolver") from exc
        resolver_started = time.perf_counter()
        log("[verified-cpu-stream] resolver start")
        resolver = resolver_type(package, cache_bytes=cpu_resolver_cache_limit(cpu_encode_workers))
        log(f"[verified-cpu-stream] resolver end seconds={time.perf_counter() - resolver_started:.3f}")
        cpu_names = (tuple(resolver.tensor_names) if cpu_vision_dense
                     else _text_manifest_names(resolver.tensor_names))
        if not cpu_names:
            resolver.close()
            raise PredictiveTBEError("verified manifest contains no recognized text/embedding/head tensors")

    try:
        model_started = time.perf_counter()
        if verified_cpu_stream:
            log("[verified-cpu-stream] model_construct start")
        try:
            model, load_receipt = load_model(
                package, device=device, attention_implementation=attention_implementation,
                stride=stride, loader_workers=loader_workers,
                metadata_cache_dir=metadata_cache_dir, log=log,
                **({"verified_cpu_source": resolver, "cpu_source_names": cpu_names}
                   if verified_cpu_stream else {}),
            )
        finally:
            if verified_cpu_stream:
                log(f"[verified-cpu-stream] model_construct end seconds={time.perf_counter() - model_started:.3f}")
        gc.collect()
        model.eval()
        conversion_started = time.perf_counter()
        if verified_cpu_stream:
            log("[verified-cpu-stream] conversion start")
        try:
            conversion = convert_predictive_modules_to_tbe(
                model, device=device, log=log,
                **({"verified_cpu_stream": True, "cpu_encode_workers": cpu_encode_workers,
                    "source_inflight_bytes": source_inflight_bytes,
                    **({"cpu_vision_dense": True} if cpu_vision_dense else {})}
                   if verified_cpu_stream else {}),
            )
        finally:
            if verified_cpu_stream:
                log(f"[verified-cpu-stream] conversion end seconds={time.perf_counter() - conversion_started:.3f}")
        if resolver is not None and cpu_vision_dense:
            conversion["cpu_resolver_cache_bytes"] = int(resolver.cache_bytes)
        if resolver is not None:
            resolver.close()
    except BaseException:
        if resolver is not None:
            resolver.close()
        raise
    return model, load_receipt, conversion, cpu_names


def _sha256_bf16(tensor: torch.Tensor) -> str:
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.bfloat16:
        raise PredictiveTBEError("predictive linear decode must return a BF16 tensor")
    raw = tensor.detach().contiguous().view(torch.int16).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def encode_predictive_weight(context_weight: Any, expected_shape, *, name: str,
                             require_cpu: bool = False, cpu_numpy_tbe: bool = False):
    """Resolve and exact-encode one Context weight; CPU-only and independently testable."""
    from glc_loader.tbe_container import decode_tbe, encode_tbe
    if type(cpu_numpy_tbe) is not bool:
        raise PredictiveTBEError("cpu_numpy_tbe must be a bool")
    if cpu_numpy_tbe:
        from glc_loader.tbe_cpu_numpy import decode_tbe_numpy, encode_tbe_numpy
        encode_tbe, decode_tbe = encode_tbe_numpy, decode_tbe_numpy

    expected_shape = tuple(int(x) for x in expected_shape)
    shape = tuple(int(x) for x in getattr(context_weight, "shape", ()))
    if shape != expected_shape or len(shape) != 2:
        raise PredictiveTBEError(f"{name}: Context shape {shape} != expected linear shape {expected_shape}")
    if shape[0] % 8 or shape[1] % 64:
        raise PredictiveTBEError(
            f"{name}: FastDecoder TBE v1 needs N divisible by 8 and K divisible by 64; got {shape}"
        )
    source_sha = getattr(context_weight, "source_sha256", None)
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise PredictiveTBEError(f"{name}: missing Context source SHA-256")

    decoded = context_weight.decode(check=True)
    if not isinstance(decoded, torch.Tensor) or decoded.dtype != torch.bfloat16:
        raise PredictiveTBEError(f"{name}: Context decode did not return a BF16 tensor")
    if (require_cpu or cpu_numpy_tbe) and decoded.device.type != "cpu":
        raise PredictiveTBEError(f"{name}: streamed predictive decode must remain on CPU")
    if tuple(decoded.shape) != expected_shape:
        raise PredictiveTBEError(f"{name}: decoded shape {tuple(decoded.shape)} != {expected_shape}")
    decoded = decoded.detach().contiguous()
    actual_sha = _sha256_bf16(decoded)
    if actual_sha != source_sha:
        raise PredictiveTBEError(f"{name}: decoded source SHA mismatch")

    # Encode on CPU so temporary TBE arrays do not compete with source Context
    # buffers on the accelerator while the serving representation is uploaded.
    cpu_weight = decoded.to(device="cpu").contiguous()
    container = encode_tbe(cpu_weight, layout="mma16")
    if tuple(container.shape) != expected_shape:
        raise PredictiveTBEError(f"{name}: TBE encoder changed the source shape")
    roundtrip = decode_tbe(container)
    if not torch.equal(roundtrip.view(torch.int16), cpu_weight.view(torch.int16)):
        raise PredictiveTBEError(f"{name}: TBE v1 round-trip differs from predictive BF16 source")
    stored_bytes = int(container.byte_size()["total"])
    source_bytes = int(cpu_weight.numel() * cpu_weight.element_size())
    del roundtrip, cpu_weight, decoded
    return container, {
        "name": name,
        "shape": list(expected_shape),
        "source_sha256": actual_sha,
        "source_bf16_bytes": source_bytes,
        "tbe_v1_stored_bytes": stored_bytes,
    }


def _predictive_module_names(model, linear_type, embedding_type):
    rows = []
    context_name_by_id = {}
    for module_name, module in model.named_modules():
        if isinstance(module, (linear_type, embedding_type)):
            weight_name = f"{module_name}.weight" if module_name else "weight"
            rows.append((module_name, weight_name, id(module), id(module.context_weight)))
            context_name_by_id[id(module.context_weight)] = module_name
    return rows, context_name_by_id


def _dependency_order(model, rows, context_name_by_id):
    """Return dependent modules before their Context references to release buffers early."""
    module_names = {name for name, _weight, _module_id, _context_id in rows}
    visited, visiting, postorder = set(), set(), []

    def visit(name):
        if name in visited or name not in module_names:
            return
        if name in visiting:
            raise PredictiveTBEError("predictive Context reference cycle in loaded module graph")
        visiting.add(name)
        context = model.get_submodule(name).context_weight if name else model.context_weight
        for reference in getattr(context, "references", ()):
            dep_name = context_name_by_id.get(id(reference))
            if dep_name is not None:
                visit(dep_name)
        visiting.remove(name)
        visited.add(name)
        postorder.append(name)

    for name, _weight, _module_id, _context_id in rows:
        visit(name)
    # postorder puts references first; reverse it so dependent Context weights
    # are converted and released before the tensors they reference.
    return list(reversed(postorder))


def _parent_and_attr(model, dotted):
    parent_name, _, attr = dotted.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    return parent, attr


def _context_resident_bytes(contexts):
    """Count the unique Context/PPCX device allocations reachable from retained wrappers."""
    seen = set()
    total = 0

    def visit(context):
        nonlocal total
        key = id(context)
        if key in seen:
            return
        seen.add(key)
        resident = getattr(context, "resident_bytes", 0)
        if type(resident) is int and resident >= 0:
            total += resident
        for reference in getattr(context, "references", ()):
            visit(reference)

    for context in contexts:
        visit(context)
    return total


def convert_predictive_modules_to_tbe(
    model, *, device: str, log: Callable[[str], None] = print,
    verified_cpu_stream: bool = False, cpu_encode_workers: int = 1,
    source_inflight_bytes: int = 8 * 1024**3,
    cpu_vision_dense: bool = False,
):
    """Replace predictive Linear/Embedding wrappers with TBE linears/dense embedding.

    Embeddings remain BF16 because FastDecoder's current input-embedding path
    expects a dense table. Linear weights become uploaded TBE v1 device containers.
    """
    runtime = _predictive_runtime()
    BctxEmbedding, BctxLinear = runtime.BctxEmbedding, runtime.BctxLinear
    from glc_serve.modules import TBEServeLinear

    from .fastdec import _text

    # FastDecoder consumes the text model's projections/input embedding plus the
    # model output head. Vision-tower modules are used by prefill and stay on
    # their existing Context wrappers, including any real linear biases.
    text_model = _text(model)
    all_rows, _ = _predictive_module_names(model, BctxLinear, BctxEmbedding)
    selected_module_ids = {
        id(module) for module in text_model.modules()
        if isinstance(module, (BctxLinear, BctxEmbedding))
    }
    output_head = model.get_output_embeddings()
    if isinstance(output_head, (BctxLinear, BctxEmbedding)):
        selected_module_ids.add(id(output_head))
    text_rows = [
        row for row in all_rows if row[2] in selected_module_ids
    ]
    context_name_by_id = {
        context_id: name for name, _weight, _module_id, context_id in text_rows
    }
    preserved_rows = [
        row for row in all_rows if row[2] not in selected_module_ids
    ]
    order = _dependency_order(model, text_rows, context_name_by_id)
    if verified_cpu_stream:
        # CPU contexts carry no native device allocations to release. Resolve
        # references first so the bounded CPU cache can reuse them, rather than
        # starting with the deepest prediction chain as the native path does.
        order.reverse()
    weight_name_by_module = {
        name: weight_name for name, weight_name, _module_id, _ctx_id in text_rows
    }
    receipts = []
    vision_dense = []
    vision_dense_source_bytes = 0
    vision_dense_runtime_bytes = 0
    t0 = time.perf_counter()
    dev = torch.device(device)
    cuda_memory = None
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(dev)
        allocated_before = int(torch.cuda.memory_allocated(dev))
        reserved_before = int(torch.cuda.memory_reserved(dev))
        torch.cuda.reset_peak_memory_stats(dev)
        cuda_memory = {
            "allocated_before_bytes": allocated_before,
            "reserved_before_bytes": reserved_before,
        }
    if type(cpu_encode_workers) is not int or not 1 <= cpu_encode_workers <= MAX_CPU_ENCODE_WORKERS:
        raise PredictiveTBEError(f"cpu_encode_workers must be between 1 and {MAX_CPU_ENCODE_WORKERS}")
    if not verified_cpu_stream and cpu_encode_workers != 1:
        raise PredictiveTBEError("cpu_encode_workers > 1 requires verified_cpu_stream")
    if type(cpu_vision_dense) is not bool:
        raise PredictiveTBEError("cpu_vision_dense must be a bool")
    if cpu_vision_dense and not verified_cpu_stream:
        raise PredictiveTBEError("cpu_vision_dense requires verified_cpu_stream")
    if verified_cpu_stream and (
        type(source_inflight_bytes) is not int or not 0 < source_inflight_bytes <= 8 * 1024**3
    ):
        raise PredictiveTBEError("source_inflight_bytes must be between 1 byte and 8 GiB")
    first_tensor_trace = {"started": False}
    first_tensor_trace_lock = threading.Lock()

    def encode_one(module_name):
        parent, attr = _parent_and_attr(model, module_name)
        original = getattr(parent, attr)
        if not isinstance(original, BctxLinear):
            return module_name, None, None
        if original.bias is not None:
            raise PredictiveTBEError(
                f"{weight_name_by_module[module_name]}: FastDecoder's projection descriptor path does not apply linear bias"
            )
        expected = (int(original.out_features), int(original.in_features))
        nbytes = math.prod(expected) * 2
        if verified_cpu_stream and nbytes > source_inflight_bytes:
            raise PredictiveTBEError(f"{weight_name_by_module[module_name]}: tensor exceeds CPU source in-flight budget")
        first_trace = False
        if verified_cpu_stream:
            with first_tensor_trace_lock:
                if not first_tensor_trace["started"]:
                    first_tensor_trace["started"] = True
                    first_trace = True
            if first_trace:
                log(f"[verified-cpu-stream] first_tensor start name={weight_name_by_module[module_name]}")
        tensor_started = time.perf_counter()
        try:
            if verified_cpu_stream and cpu_encode_workers > 1:
                result = encode_predictive_weight(
                    original.context_weight, expected, name=weight_name_by_module[module_name],
                    require_cpu=True,
                    cpu_numpy_tbe=True,
                )
            else:
                result = encode_predictive_weight(
                    original.context_weight, expected, name=weight_name_by_module[module_name],
                    require_cpu=cpu_vision_dense or verified_cpu_stream,
                    **({"cpu_numpy_tbe": True} if verified_cpu_stream else {}),
                )
        finally:
            if first_trace:
                log(f"[verified-cpu-stream] first_tensor end seconds={time.perf_counter() - tensor_started:.3f}")
        return module_name, result, nbytes

    def batches():
        current, used = [], 0
        for module_name in order:
            parent, attr = _parent_and_attr(model, module_name)
            module = getattr(parent, attr)
            if not isinstance(module, BctxLinear):
                yield [module_name]
                continue
            size = math.prod((int(module.out_features), int(module.in_features))) * 2
            if current and (
                (verified_cpu_stream and used + size > source_inflight_bytes)
                or len(current) >= cpu_encode_workers
            ):
                yield current
                current, used = [], 0
            if verified_cpu_stream and size > source_inflight_bytes:
                raise PredictiveTBEError(f"{weight_name_by_module[module_name]}: tensor exceeds CPU source in-flight budget")
            current.append(module_name)
            used += size
        if current:
            yield current

    def encoded_in_order():
        for batch in batches():
            if cpu_encode_workers == 1 or len(batch) == 1:
                for name in batch:
                    yield encode_one(name)
            else:
                # Future count and source bytes are both bounded by the batch builder.
                with ThreadPoolExecutor(max_workers=cpu_encode_workers) as pool:
                    futures = [pool.submit(encode_one, name) for name in batch]
                    for future in futures:
                        yield future.result()

    for module_name, encoded, _source_bytes in encoded_in_order():
        weight_name = weight_name_by_module[module_name]
        # Fetch the live module instead of retaining all original wrappers in a list.
        parent, attr = _parent_and_attr(model, module_name)
        original = getattr(parent, attr)
        if isinstance(original, BctxLinear):
            container, record = encoded
            replacement = TBEServeLinear(
                container, original.bias, torch.device(device), pool=None,
                name=weight_name, exec_mode="exact",
            )
            record["runtime_kind"] = "tbe_v1_linear"
            record["tbe_v1_resident_bytes"] = int(replacement.resident_bytes)
        else:
            # FastDecoder reads the input embedding table directly. Keep this one
            # table dense and drop its original Context/PPCX references promptly.
            decoded = original.context_weight.decode(check=True).detach().contiguous()
            expected = (int(original.num_embeddings), int(original.embedding_dim))
            source_bytes = math.prod(expected) * 2
            if verified_cpu_stream and source_bytes > source_inflight_bytes:
                raise PredictiveTBEError(f"{weight_name}: tensor exceeds CPU source in-flight budget")
            if verified_cpu_stream and cpu_encode_workers > 1 and decoded.device.type != "cpu":
                raise PredictiveTBEError(f"{weight_name}: streamed predictive decode must remain on CPU")
            if tuple(decoded.shape) != expected:
                raise PredictiveTBEError(f"{weight_name}: decoded embedding shape mismatch")
            source_sha = getattr(original.context_weight, "source_sha256", None)
            actual_sha = _sha256_bf16(decoded)
            if actual_sha != source_sha:
                raise PredictiveTBEError(f"{weight_name}: embedding source SHA mismatch")
            replacement = torch.nn.Embedding.from_pretrained(
                decoded.to(device=device).contiguous(), freeze=True,
                padding_idx=original.padding_idx,
            )
            replacement.train(original.training)
            record = {
                "name": weight_name, "shape": list(expected),
                "source_sha256": actual_sha,
                "source_bf16_bytes": int(decoded.numel() * decoded.element_size()),
                "tbe_v1_stored_bytes": 0,
            }
            record["runtime_kind"] = "bf16_input_embedding"
            record["runtime_dense_bytes"] = int(decoded.numel() * decoded.element_size())
            del decoded
        if cpu_vision_dense:
            record.update({
                "component": "text", "source_kind": "verified_cpu_resolver",
                "decoded_bytes": record["source_bf16_bytes"],
                "source_bytes": record["source_bf16_bytes"],
                "runtime_bytes": record.get("tbe_v1_resident_bytes", record.get("runtime_dense_bytes", 0)),
            })
            log(f"[cpu-vision-dense] tensor={weight_name} component=text decoded_bytes={record['decoded_bytes']} source_bytes={record['source_bf16_bytes']} runtime_bytes={record['runtime_bytes']} source_kind=verified_cpu_resolver runtime_kind={record['runtime_kind']}")
        setattr(parent, attr, replacement)
        receipts.append(record)
        if isinstance(original, BctxLinear):
            del container
        del original, replacement
        if len(receipts) % 25 == 0 or len(receipts) == len(order):
            log(f"[predictive-tbe-private] converted {len(receipts)}/{len(order)} tensors")
    preserved = []
    preserved_contexts = []
    for module_name, weight_name, _module_id, _context_id in preserved_rows:
        parent, attr = _parent_and_attr(model, module_name)
        module = getattr(parent, attr)
        context = module.context_weight
        is_vision = any(part.lower() in {
            "vision", "visual", "vision_tower", "vision_model", "vision_encoder",
        } for part in module_name.split("."))
        if cpu_vision_dense and is_vision:
            expected = tuple(int(x) for x in context.shape)
            source_bytes = math.prod(expected) * 2
            if source_bytes > source_inflight_bytes:
                raise PredictiveTBEError(f"{weight_name}: tensor exceeds CPU source in-flight budget")
            decoded = context.decode(check=True)
            if not isinstance(decoded, torch.Tensor) or decoded.dtype != torch.bfloat16:
                raise PredictiveTBEError(f"{weight_name}: verified vision decode did not return BF16")
            if tuple(decoded.shape) != expected:
                raise PredictiveTBEError(f"{weight_name}: decoded vision shape mismatch")
            decoded = decoded.detach().contiguous()
            actual_sha = _sha256_bf16(decoded)
            if actual_sha != getattr(context, "source_sha256", None):
                raise PredictiveTBEError(f"{weight_name}: vision source SHA mismatch")
            if decoded.device.type != "cpu":
                raise PredictiveTBEError(f"{weight_name}: verified vision decode must remain on CPU")
            if decoded.numel() * decoded.element_size() != source_bytes:
                raise PredictiveTBEError(f"{weight_name}: decoded vision byte count mismatch")
            if isinstance(module, BctxLinear):
                out_features, in_features = expected
                replacement = torch.nn.Linear(in_features, out_features, bias=False, device="meta", dtype=decoded.dtype)
                replacement.weight = torch.nn.Parameter(decoded.to(device=device).contiguous(), requires_grad=False)
                runtime_kind = "dense_bf16_linear"
            elif isinstance(module, BctxEmbedding):
                replacement = torch.nn.Embedding.from_pretrained(
                    decoded.to(device=device).contiguous(), freeze=True,
                    padding_idx=module.padding_idx,
                )
                runtime_kind = "dense_bf16_embedding"
            else:
                raise PredictiveTBEError(f"{weight_name}: unsupported vision Context module")
            bias_receipt = None
            bias = getattr(module, "bias", None)
            runtime_bytes = int(replacement.weight.numel() * replacement.weight.element_size())
            if bias is not None:
                if not isinstance(module, BctxLinear) or tuple(bias.shape) != (expected[0],):
                    raise PredictiveTBEError(f"{weight_name}: vision linear bias shape mismatch")
                copied_bias = bias.detach().to(device=device).contiguous()
                replacement.bias = torch.nn.Parameter(copied_bias, requires_grad=False)
                bias_bytes = int(bias.numel() * bias.element_size())
                bias_receipt = {
                    "name": weight_name.rsplit(".", 1)[0] + ".bias", "component": "vision",
                    "source_kind": "verified_cpu_resolver_dense_bias",
                    "source_bytes": bias_bytes, "decoded_bytes": bias_bytes,
                    "runtime_bytes": int(copied_bias.numel() * copied_bias.element_size()),
                    "runtime_kind": "dense_bias", "shape": list(bias.shape),
                    "runtime_dtype": str(copied_bias.dtype),
                }
                log(f"[cpu-vision-dense] tensor={bias_receipt['name']} component=vision decoded_bytes={bias_bytes} source_bytes={bias_bytes} runtime_bytes={bias_receipt['runtime_bytes']} source_kind=verified_cpu_resolver_dense_bias runtime_kind=dense_bias")
            replacement.train(module.training)
            row = {
                "name": weight_name, "component": "vision",
                "source_kind": "verified_cpu_resolver",
                "shape": list(expected), "source_sha256": actual_sha,
                "source_bytes": source_bytes, "decoded_bytes": source_bytes,
                "runtime_bytes": int(replacement.weight.numel() * replacement.weight.element_size()),
                "runtime_kind": runtime_kind,
            }
            vision_dense.append(row)
            vision_dense_source_bytes += source_bytes
            vision_dense_runtime_bytes += runtime_bytes
            if bias_receipt is not None:
                vision_dense.append(bias_receipt)
                vision_dense_source_bytes += bias_receipt["source_bytes"]
                vision_dense_runtime_bytes += bias_receipt["runtime_bytes"]
            log(f"[cpu-vision-dense] tensor={weight_name} component=vision decoded_bytes={source_bytes} source_bytes={source_bytes} runtime_bytes={row['runtime_bytes']} source_kind=verified_cpu_resolver")
            setattr(parent, attr, replacement)
            del decoded, replacement, module
            continue
        preserved_contexts.append(context)
        visual = "visual" in module_name.lower()
        preserved.append({
            "name": weight_name,
            "kind": "predictive_reference_vision" if visual else "predictive_reference_nontext",
            "shape": [int(x) for x in context.shape],
            "source_sha256": getattr(context, "source_sha256", None),
            "own_resident_context_bytes": int(getattr(context, "resident_bytes", 0)),
            "bias_present": bool(getattr(module, "bias", None) is not None),
        })
    if cuda_memory is not None:
        torch.cuda.synchronize(dev)
        peak = int(torch.cuda.max_memory_allocated(dev))
        cuda_memory.update({
            "allocated_after_bytes": int(torch.cuda.memory_allocated(dev)),
            "reserved_after_bytes": int(torch.cuda.memory_reserved(dev)),
            "peak_allocated_bytes": peak,
            "peak_incremental_bytes": max(0, peak - cuda_memory["allocated_before_bytes"]),
        })
    receipt = {
        "tensor_count": len(receipts),
        "linear_count": sum(r["runtime_kind"] == "tbe_v1_linear" for r in receipts),
        "embedding_count": sum(r["runtime_kind"] == "bf16_input_embedding" for r in receipts),
        "source_bf16_bytes": sum(r["source_bf16_bytes"] for r in receipts),
        "tbe_v1_linear_source_bytes": sum(
            r["source_bf16_bytes"] for r in receipts if r["runtime_kind"] == "tbe_v1_linear"
        ),
        "tbe_v1_stored_bytes": sum(r["tbe_v1_stored_bytes"] for r in receipts),
        "tbe_v1_linear_storage_ratio": (
            sum(r["source_bf16_bytes"] for r in receipts if r["runtime_kind"] == "tbe_v1_linear")
            / max(1, sum(r["tbe_v1_stored_bytes"] for r in receipts))
        ),
        "tbe_v1_resident_bytes": sum(r.get("tbe_v1_resident_bytes", 0) for r in receipts),
        "preserved_context_module_count": len(preserved),
        "preserved_context_resident_bytes": _context_resident_bytes(preserved_contexts),
        "preserved_contexts": preserved,
        "cuda_memory": cuda_memory,
        "conversion_seconds": time.perf_counter() - t0,
        "tensors": receipts,
    }
    if cpu_vision_dense:
        receipt.update({
            "cpu_vision_dense": True,
            "vision_dense_tensor_count": len(vision_dense),
            "vision_dense_source_bytes": vision_dense_source_bytes,
            "vision_dense_runtime_bytes": vision_dense_runtime_bytes,
            "vision_dense_tensors": vision_dense,
        })
    return receipt


def load_predictive_fast_session(
    *, package: str | Path, tune: str | Path, device: str = "cuda:0",
    max_len: int = 32768, stride: int = 1024, loader_workers: int = 1,
    history_slots: int = 16,
    metadata_cache_dir: str | Path | None = None,
    attention_implementation: str = "eager", log: Callable[[str], None] = print,
    verified_cpu_stream: bool = False, cpu_encode_workers: int = 1,
    source_inflight_bytes: int = 8 * 1024**3,
    cpu_vision_dense: bool = False,
):
    """Load the current predictive package into the private TBE-v1 FastSession path.

    MTP is disabled in this first version. No dense checkpoint is created; each
    selected predictive linear is validated, encoded, uploaded, then its original
    Context wrapper is unlinked from the model. The opt-in ``cpu_vision_dense``
    experiment requires ``verified_cpu_stream`` and materializes verified vision
    Context weights as dense runtime modules after text TBE conversion.
    """
    if type(cpu_vision_dense) is not bool:
        raise PredictiveTBEError("cpu_vision_dense must be a bool")
    if cpu_vision_dense and not verified_cpu_stream:
        raise PredictiveTBEError("cpu_vision_dense requires verified_cpu_stream")
    from .fastserve import FastSession, _resolve_tune
    from .fastdec import MAX_M

    history_slots = FastSession._validate_history_slots(history_slots, mtp_disabled=True)
    from transformers import AutoProcessor, AutoTokenizer
    from .engine import Engine
    from .fastdec import FastDecoder, load_tune
    from .tbe_desc import tune_tbe_per_m
    from glc_loader.tbe_mma import resolve_tbe_mma_arch
    runtime = _predictive_runtime()
    load_predictive_model = runtime.load_predictive_model

    package = Path(package).expanduser().resolve()
    device = str(device)
    # Reject unsupported devices before opening the CPU resolver or model package.
    tbe_arch = resolve_tbe_mma_arch(device)
    # The normal bundle loader initializes this before any TBE prefill. This
    # private path bypasses that loader, so derive the arch here as well.
    tune_path = _resolve_tune(str(tune) if tune is not None else None, device)
    tune_file = Path(tune_path).expanduser().resolve()
    try:
        tune_sha256 = hashlib.sha256()
        with tune_file.open("rb") as tune_stream:
            for chunk in iter(lambda: tune_stream.read(1024 * 1024), b""):
                tune_sha256.update(chunk)
    except OSError as exc:
        raise PredictiveTBEError(f"cannot read FastDecoder tune file {tune_file}: {exc}") from exc
    tune_sha256 = tune_sha256.hexdigest()
    start = time.perf_counter()
    # Collect the loader's recursive tensor table before replacing wrappers.
    model, load_receipt, conversion, cpu_names = _load_and_convert_predictive(
        load_predictive_model, package=package, device=device,
        attention_implementation=attention_implementation, stride=stride,
        loader_workers=loader_workers, metadata_cache_dir=metadata_cache_dir,
        log=log, verified_cpu_stream=verified_cpu_stream,
        cpu_encode_workers=cpu_encode_workers,
        source_inflight_bytes=source_inflight_bytes,
        cpu_vision_dense=cpu_vision_dense,
    )
    gc.collect()
    metadata_dir = Path(load_receipt["metadata_dir"])
    processor = AutoProcessor.from_pretrained(str(metadata_dir), local_files_only=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(str(metadata_dir), local_files_only=True)

    # Engine supplies the Request preparation/tokenizer behavior as in FastSession's
    # established serving path. This first prototype deliberately disables MTP.
    loaded = SimpleNamespace(
        model=model, mtp=None, config=getattr(model, "config", None),
        local_dir=metadata_dir, receipt=load_receipt, options=None,
        streamer=None, pools=None, name_to_module={}, bundle=None, backend_label="tbe",
    )
    engine = Engine(
        loaded, processor, tokenizer, max_batch=1, enable_mtp=False,
        default_chat_template_kwargs={"enable_thinking": False}, log=log,
    )
    engine.tokenizer = tokenizer
    tune_config = load_tune(tune_path)
    fd = FastDecoder(model, None, tune=tune_config, max_len=max_len,
                     R=history_slots, device=device)
    meta: Dict[str, Any] = {
        "server_argv": ["--predictive-package", str(package), "--device", device,
                        "--exec-mode", "exact", "--mtp", "off"],
        "load_s": round(time.perf_counter() - start, 1),
        "source_dir": str(metadata_dir),
        "tune_path": str(tune_file),
        "tune_sha256": tune_sha256,
        "predictive_runtime_module": runtime.__name__,
        "tbe_mma_arch": tbe_arch,
        "predictive_load_receipt": load_receipt,
        "predictive_to_tbe_conversion": conversion,
    }
    if verified_cpu_stream:
        meta["verified_cpu_stream"] = {
            "enabled": True, "selected_tensor_count": len(cpu_names),
            "cpu_encode_workers": cpu_encode_workers,
            "source_inflight_bytes": source_inflight_bytes,
            "resolver_cache_bytes": cpu_resolver_cache_limit(cpu_encode_workers),
            "phases": {"load_and_convert_seconds": round(time.perf_counter() - start, 3)},
        }
        if cpu_vision_dense:
            meta["verified_cpu_stream"].update({
                "cpu_vision_dense": True,
                "resolver_cache_bytes_before_drain": conversion.get("cpu_resolver_cache_bytes"),
            })
    if fd.descriptor_kinds().get("tbe"):
        meta["tbe_tune"] = tune_tbe_per_m(fd._descs(), log=log)
    dense_param_bytes = sum(int(p.numel() * p.element_size()) for p in model.parameters())
    dense_buffer_bytes = sum(int(b.numel() * b.element_size()) for b in model.buffers())
    descs = fd._descs()
    descriptor_bytes = sum(int(d.bytes) for d in descs)
    dense_descriptor_bytes = sum(int(d.bytes) for d in descs if getattr(d, "kind", None) == "bf16")
    dense_parameter_remainder = max(0, dense_param_bytes - dense_descriptor_bytes)
    preserved_context_bytes = int(conversion["preserved_context_resident_bytes"])
    source_model_bytes = sum(
        math.prod(int(d) for d in row["shape"]) * 2
        for row in load_receipt.get("consumed_tensors", {}).values()
        if isinstance(row, dict) and isinstance(row.get("shape"), list)
    )
    total_runtime_weight_bytes = descriptor_bytes + dense_parameter_remainder + preserved_context_bytes
    meta["runtime_representation"] = {
        "tbe_fastdec_descriptor_bytes": descriptor_bytes,
        "dense_parameter_bytes": dense_param_bytes,
        "dense_descriptor_backing_bytes": dense_descriptor_bytes,
        "unrepresented_dense_parameter_bytes": dense_parameter_remainder,
        "preserved_predictive_context_bytes": preserved_context_bytes,
        "dense_buffer_bytes": dense_buffer_bytes,
        "total_weight_bytes": total_runtime_weight_bytes,
        "total_weight_and_buffer_bytes": total_runtime_weight_bytes + dense_buffer_bytes,
        "source_model_bf16_bytes": source_model_bytes,
        "gpu_representation_ratio": (
            source_model_bytes / max(1, total_runtime_weight_bytes)
        ),
        "gpu_representation_ratio_definition": (
            "BF16 source parameter bytes / (FastDecoder descriptor bytes + remaining dense "
            "parameter bytes + preserved predictive Context bytes); excludes workspace and allocator reserve"
        ),
    }
    if cpu_vision_dense:
        meta["runtime_representation"].pop("gpu_representation_ratio", None)
        meta["runtime_representation"].pop("gpu_representation_ratio_definition", None)
        dense_consumed = load_receipt.get("consumed_tensors", {})
        runtime_parameters = dict(model.named_parameters())
        dense_norm_rows = []
        dense_vision_rows = []
        dense_other_rows = []
        converted_vision_names = {
            row["name"] for row in conversion.get("vision_dense_tensors", [])
            if isinstance(row, dict) and isinstance(row.get("name"), str)
        }
        for name, item in dense_consumed.items():
            if not isinstance(item, dict) or item.get("kind") not in ("dense_parameter", "dense_bias"):
                continue
            if name in converted_vision_names:
                continue
            shape = item.get("shape")
            if not isinstance(shape, list):
                continue
            source_bytes = math.prod(int(dim) for dim in shape) * 2
            runtime = runtime_parameters.get(name)
            runtime_bytes = int(runtime.numel() * runtime.element_size()) if runtime is not None else 0
            row = {"name": name, "source_kind": "verified_cpu_resolver",
                   "source_bytes": source_bytes, "decoded_bytes": source_bytes,
                   "runtime_bytes": runtime_bytes, "shape": shape,
                   "runtime_dtype": item.get("runtime_dtype")}
            lower = name.lower()
            if any(token in lower for token in ("norm", "layer_norm", "rmsnorm", "ln_")):
                row["component"] = "dense_norm"
                dense_norm_rows.append(row)
            elif any(part in {"visual", "vision", "vision_tower", "vision_model", "vision_encoder"}
                     for part in name.split(".")[:-1]):
                row["component"] = "dense_vision_parameter"
                dense_vision_rows.append(row)
            else:
                row["component"] = "dense_other_parameter"
                dense_other_rows.append(row)
            log(f"[cpu-vision-dense] tensor={name} component={row['component']} decoded_bytes={source_bytes} source_bytes={source_bytes} runtime_bytes={runtime_bytes} source_kind=verified_cpu_resolver runtime_kind=dense_parameter")
        meta["cpu_source_weight_ledger"] = {
            "cpu_vision_dense": True,
            "source_kind": "verified_cpu_resolver",
            "cpu_resolver_cache_bytes": conversion.get("cpu_resolver_cache_bytes"),
            "text_runtime": conversion.get("tensors", []),
            "vision_dense": conversion.get("vision_dense_tensors", []),
            "dense_norm": dense_norm_rows,
            "dense_vision_parameters": dense_vision_rows,
            "dense_other_parameters": dense_other_rows,
        }
    if history_slots < MAX_M + 1:
        # The B1 greedy session replays verify1; capture it for reduced history.
        kinds = [("ar", 1), ("verify", 1)]
    else:
        kinds = [("ar", 1)] + [("verify", m) for m in range(1, MAX_M + 1)]
    meta["history_slots"] = int(fd.R)
    meta["mtp_enabled"] = bool(fd.mtp_ready)
    meta["captured_kinds"] = [list(kind) for kind in kinds]
    meta["capture_s"] = fd.capture(kinds)
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        meta["cuda_after_capture"] = {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "startup_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
    meta["descriptors"] = fd.descriptor_kinds()
    meta["streamed_bytes_per_step"] = fd.streamed_bytes()
    log(f"[predictive-tbe-private] ready in {meta['load_s']} s; {meta['descriptors']}")
    return FastSession(engine, fd, meta)


__all__ = [
    "PredictiveTBEError", "convert_predictive_modules_to_tbe",
    "encode_predictive_weight", "load_predictive_fast_session",
]
