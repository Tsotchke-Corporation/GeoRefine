"""Load a BCTX model without restoring or transcoding its checkpoint.

Linear and embedding weights retain their BCTX device descriptors.  The simple
linear adapter decodes a transient BF16 matrix per call; a backend with a fused
GEMM can replace it while retaining the same ``CudaContextTensor`` contract.
"""
from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from fnmatch import fnmatchcase
import hashlib
import io
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import tarfile
from typing import Any, Callable

import torch
from torch import nn


class ContextServingError(RuntimeError):
    """Malformed or incomplete BCTX serving package."""


def _param_names(module):
    try:
        return dict(module.named_parameters(remove_duplicate=False))
    except TypeError:  # older torch
        return dict(module.named_parameters())


def _default_factory(config, attention_implementation):
    try:
        from transformers import Qwen3_5ForConditionalGeneration
    except ImportError as exc:
        raise ContextServingError("transformers lacks Qwen3_5ForConditionalGeneration") from exc
    _apply_attention_implementation(config, attention_implementation)
    return Qwen3_5ForConditionalGeneration(config)


def _apply_attention_implementation(config, implementation):
    targets = [config]
    for attr in ("text_config", "vision_config"):
        child = config.get(attr) if isinstance(config, dict) else getattr(config, attr, None)
        if child is not None:
            targets.append(child)
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        try:
            child = getter(decoder=True)
        except TypeError:
            child = getter()
        if child is not None:
            targets.append(child)
    seen = set()
    for target in targets:
        if id(target) in seen:
            continue
        seen.add(id(target))
        if isinstance(target, dict):
            target["_attn_implementation"] = implementation
        else:
            setattr(target, "_attn_implementation", implementation)


def _runtime_parameter_dtype(model, name, parameter, requested_dtype):
    """Mirror HF requested dtype, honoring the model's explicit FP32 keep rules."""
    if not parameter.is_floating_point():
        return parameter.dtype
    patterns = []
    for _, module in model.named_modules():
        strict = getattr(module, "_keep_in_fp32_modules_strict", ()) or ()
        legacy = (getattr(module, "_keep_in_fp32_modules", ()) or ()
                  if requested_dtype == torch.float16 else ())
        patterns.extend(str(pattern) for pattern in (*strict, *legacy))
    if any(fnmatchcase(name, pattern) for pattern in patterns):
        return torch.float32
    return requested_dtype


def _empty_weights():
    try:
        from accelerate import init_empty_weights
    except ImportError as exc:
        raise ContextServingError("accelerate is required for meta-model construction") from exc
    return init_empty_weights(include_buffers=False)


@contextmanager
def _default_dtype(dtype):
    import torch
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def _load_host_decoder(package: Path):
    del package
    try:
        from georefine_context_runtime.bitexact_context_codec import decode
    except ImportError:
        try:
            from georefine_context_runtime.bitexact_context_codec import decode
        except ImportError as exc:
            raise ContextServingError("installed BCTX host decoder unavailable") from exc
    return decode


def _safe_asset_name(name: Any) -> str:
    if not isinstance(name, str):
        raise ContextServingError("metadata asset name must be a string")
    rel = PurePosixPath(name)
    if (rel.is_absolute() or not rel.parts or rel.parts[0] != "sidecars"
            or ".." in rel.parts or "." in rel.parts or rel.as_posix() != name):
        raise ContextServingError(f"unsafe metadata asset name: {name!r}")
    return rel.as_posix()


def _asset_pins(manifest):
    assets = manifest.get("metadata_assets")
    if not isinstance(assets, dict) or not assets:
        raise ContextServingError("manifest metadata_assets must be a nonempty object")
    pins = {}
    for name, pin in assets.items():
        name = _safe_asset_name(name)
        if name in pins or not isinstance(pin, dict):
            raise ContextServingError(f"duplicate or malformed metadata pin: {name}")
        if (type(pin.get("bytes")) is not int or pin["bytes"] < 0
                or not _hex_sha(pin.get("sha256"))):
            raise ContextServingError(f"invalid metadata asset pin: {name}")
        pins[name] = {"bytes": pin["bytes"], "sha256": pin["sha256"]}
    return pins


def _physical_sidecars(package: Path, pins):
    source = package / "sidecars"
    if not source.exists():
        return None
    if source.is_symlink() or not source.is_dir():
        raise ContextServingError("physical sidecars must be a regular directory")
    found = {}
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ContextServingError("physical sidecars contain a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ContextServingError("physical sidecars contain a non-file")
        name = path.relative_to(package).as_posix()
        if name in found:
            raise ContextServingError(f"duplicate physical sidecar: {name}")
        found[name] = path
    if set(found) != set(pins):
        raise ContextServingError("physical sidecars differ from manifest metadata_assets")
    for name, path in found.items():
        payload = path.read_bytes()
        if len(payload) != pins[name]["bytes"] or hashlib.sha256(payload).hexdigest() != pins[name]["sha256"]:
            raise ContextServingError(f"physical sidecar integrity mismatch: {name}")
    return source


def prepare_metadata(package: str | Path, cache_dir: str | Path) -> Path:
    """Verify metadata archive/assets, then return verified sidecars outside package."""
    package, cache_dir = Path(package).resolve(), Path(cache_dir).resolve()
    if cache_dir == package or package in cache_dir.parents:
        raise ContextServingError("metadata cache must be outside the immutable package")
    manifest_path = package / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContextServingError(f"cannot read BCTX manifest: {exc}") from exc
    pins = _asset_pins(manifest)
    if len(pins) != 13:
        raise ContextServingError("BCTX v1 metadata inventory must contain 13 sidecars")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ContextServingError("manifest files must be a list")
    file_pins = {}
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ContextServingError("malformed package file inventory")
        path = item["path"]
        if path in file_pins:
            raise ContextServingError(f"duplicate package file inventory path: {path}")
        file_pins[path] = item
    archive_pin = file_pins.get("model-assets.tar.xz")
    if (not isinstance(archive_pin, dict) or type(archive_pin.get("bytes")) is not int
            or not _hex_sha(archive_pin.get("sha256"))):
        raise ContextServingError("manifest lacks model-assets.tar.xz byte/SHA pin")
    archive_path = package / "model-assets.tar.xz"
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ContextServingError("metadata archive must be a regular file")
    try:
        archive = archive_path.read_bytes()
    except OSError as exc:
        raise ContextServingError(f"cannot read metadata archive: {exc}") from exc
    archive_sha = hashlib.sha256(archive).hexdigest()
    if len(archive) != archive_pin["bytes"] or archive_sha != archive_pin["sha256"]:
        raise ContextServingError("metadata archive size/SHA mismatch")

    members: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:xz") as tf:
            for member in tf.getmembers():
                name = _safe_asset_name(member.name)
                if name in members:
                    raise ContextServingError(f"duplicate metadata archive member: {name}")
                if not member.isfile():
                    raise ContextServingError(f"metadata archive member is not a regular file: {name}")
                src = tf.extractfile(member)
                if src is None:
                    raise ContextServingError(f"cannot read metadata archive member: {name}")
                members[name] = src.read()
    except (tarfile.TarError, lzma.LZMAError) as exc:
        raise ContextServingError("invalid xz metadata archive") from exc
    if set(members) != set(pins):
        raise ContextServingError("metadata archive members differ from manifest metadata_assets")
    for name, payload in members.items():
        if len(payload) != pins[name]["bytes"] or hashlib.sha256(payload).hexdigest() != pins[name]["sha256"]:
            raise ContextServingError(f"metadata archive asset integrity mismatch: {name}")
    index_sha = manifest.get("source", {}).get("index_sha256")
    index_payload = members.get("sidecars/model.safetensors.index.json")
    if index_sha is not None and (index_payload is None or hashlib.sha256(index_payload).hexdigest() != index_sha):
        raise ContextServingError("archived safetensors index differs from manifest source pin")

    physical = _physical_sidecars(package, pins)
    if physical is not None:
        return physical

    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / archive_sha
    if destination.is_symlink():
        raise ContextServingError("metadata cache destination must not be a symlink")
    if destination.exists():
        if not destination.is_dir():
            raise ContextServingError("metadata cache destination is not a directory")
        present = {}
        for path in destination.rglob("*"):
            if path.is_symlink():
                raise ContextServingError("metadata cache contains a symlink")
            if path.is_file():
                present[path.relative_to(cache_dir).as_posix()] = path
        expected_rel = {f"{archive_sha}/{name}" for name in pins}
        if set(present) != expected_rel:
            raise ContextServingError("existing metadata cache inventory mismatch")
        for name, pin in pins.items():
            payload = present[f"{archive_sha}/{name}"].read_bytes()
            if len(payload) != pin["bytes"] or hashlib.sha256(payload).hexdigest() != pin["sha256"]:
                raise ContextServingError(f"existing metadata cache integrity mismatch: {name}")
    else:
        staging = Path(tempfile.mkdtemp(prefix=archive_sha + ".partial-", dir=cache_dir))
        try:
            for name, payload in members.items():
                target = staging.joinpath(*PurePosixPath(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            os.replace(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return destination / "sidecars"


def _safe_frame_path(package: Path, value: Any) -> Path:
    if not isinstance(value, str):
        raise ContextServingError("tensor frame path must be a string")
    rel = PurePosixPath(value)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts:
        raise ContextServingError("unsafe tensor frame path")
    return package.joinpath(*rel.parts)


def _hex_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContextServingError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _tensor_from_host(decoded, expected_dtype, expected_shape):
    import numpy as np
    import torch
    if isinstance(decoded, torch.Tensor):
        tensor = decoded.detach().cpu()
    else:
        array = np.ascontiguousarray(decoded)
        if array.dtype.kind == "u" and array.dtype.itemsize == 2:
            tensor = torch.from_numpy(array).view(torch.bfloat16)
        else:
            tensor = torch.from_numpy(array)
    if tensor.numel() != __import__("math").prod(expected_shape):
        raise ContextServingError(f"decoded tensor shape {tuple(tensor.shape)} != {tuple(expected_shape)}")
    return tensor.reshape(expected_shape).to(dtype=expected_dtype).contiguous()


def _decoded_bf16_source_sha(decoded) -> str:
    """Hash the exact little-endian BF16 words before any runtime dtype cast."""
    import numpy as np
    import torch
    if isinstance(decoded, torch.Tensor):
        source = decoded.detach().cpu().contiguous()
        if source.dtype != torch.bfloat16:
            raise ContextServingError("host BCTX decoder did not return BF16")
        raw = source.view(torch.uint8).numpy().tobytes()
    else:
        source = np.ascontiguousarray(decoded)
        if source.dtype.kind != "u" or source.dtype.itemsize != 2:
            raise ContextServingError("host BCTX decoder did not return BF16 words")
        raw = source.astype("<u2", copy=False).tobytes()
    return hashlib.sha256(raw).hexdigest()


def _frame_shape(source_shape):
    """BCTX frames store rank-N tensors as rows x flattened remaining axes."""
    shape = tuple(int(d) for d in source_shape)
    rows = shape[0] if len(shape) > 1 else 1
    return rows, __import__("math").prod(shape) // rows


def _validate_host_frame(decoded, row, name):
    if tuple(decoded.shape) != _frame_shape(row["shape"]):
        raise ContextServingError(f"BCTX decoded frame geometry mismatch: {name}")
    if _decoded_bf16_source_sha(decoded) != row["source_sha256"]:
        raise ContextServingError(f"BCTX decoded-source identity mismatch: {name}")


class BctxLinear(nn.Module):
    """Linear weight stays compressed; only its per-call decoded result is dense."""
    def __init__(self, tensor, bias=None):
        import torch
        super().__init__()
        if len(tensor.shape) != 2:
            raise ContextServingError("linear BCTX weight must be rank two")
        self.context_weight = tensor
        self.out_features, self.in_features = map(int, tensor.shape)
        self.forward_calls = 0
        self.weight_materializations = 0
        self.register_parameter("bias", bias)

    @property
    def weight(self):
        """Decode for direct HF helpers; the result is never retained by this module."""
        self.weight_materializations += 1
        return self.context_weight.decode()

    def forward(self, x):
        import torch.nn.functional as F
        self.forward_calls += 1
        weight = self.weight
        # PyTorch folds strided batched inputs for an ordinary Linear Parameter
        # that requires grad, even during inference. A temporary decoded tensor
        # lacks that flag and otherwise selects bmm with different BF16 rounding.
        # Preserve the parent's mm dispatch without retaining a dense Parameter.
        if self.bias is None and x.ndim > 2 and not x.is_contiguous():
            output = F.linear(x.reshape(-1, x.shape[-1]), weight)
            return output.reshape(*x.shape[:-1], self.out_features)
        return F.linear(x, weight, self.bias)


class BctxEmbedding(nn.Module):
    """Embedding weight remains compressed and is decoded by indexed row gather."""
    def __init__(self, tensor, *, padding_idx=None):
        import torch
        super().__init__()
        if len(tensor.shape) != 2:
            raise ContextServingError("embedding BCTX weight must be rank two")
        self.context_weight = tensor
        self.num_embeddings, self.embedding_dim = map(int, tensor.shape)
        self.padding_idx = padding_idx
        self.forward_calls = 0
        self.weight_materializations = 0

    @property
    def weight(self):
        """Decode on direct access without retaining a dense embedding table."""
        self.weight_materializations += 1
        return self.context_weight.decode()

    def forward(self, input):
        self.forward_calls += 1
        return self.context_weight.gather_rows(input)


def _parent_and_attr(model, dotted):
    parent_name, _, attr = dotted.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    return parent, attr


def load_bctx_model(
    package: str | Path,
    *,
    device: str = "cuda:0",
    dtype=None,
    attention_implementation: str = "eager",
    stride: int = 1024,
    tensor_factory: Callable[..., Any] | None = None,
    model_factory: Callable[..., Any] | None = None,
    config_loader: Callable[[Path], Any] | None = None,
    generation_config_loader: Callable[[Path], Any] | None = None,
    empty_weights_factory: Callable[[], Any] | None = None,
    host_decoder: Callable[[bytes], Any] | None = None,
    metadata_cache_dir: str | Path | None = None,
    log: Callable[[str], None] = print,
    progress_callback: Callable[[int, int, str], None] | None = None,
    loader_workers: int = 1,
):
    """Construct Qwen3.5 from BCTX frames only, returning ``(model, receipt)``.

    Injection points exist for CPU tests. Production defaults require the CUDA
    BCTX tensor implementation and Transformers/Accelerate; there is no dense
    checkpoint fallback.
    """
    import torch
    from torch import nn

    if isinstance(loader_workers, bool) or not isinstance(loader_workers, int) or loader_workers < 1:
        raise ValueError("loader_workers must be a positive integer")
    package = Path(package)
    manifest_path = package / "manifest.json"
    config_path = package / "sidecars" / "config.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContextServingError(f"cannot read BCTX manifest: {exc}") from exc
    if manifest.get("schema") != "bitexact-context-package-v1" or manifest.get("complete") is not True:
        raise ContextServingError("unsupported or incomplete BCTX package manifest")
    cache_dir = (Path(metadata_cache_dir) if metadata_cache_dir is not None
                 else package.parent / ".scratch" / "bctx-serving-metadata")
    metadata_dir = prepare_metadata(package, cache_dir)
    config_path = metadata_dir / "config.json"
    if not config_path.is_file():
        raise ContextServingError(f"model config missing: {config_path}")
    rows = manifest.get("tensors")
    if not isinstance(rows, list):
        raise ContextServingError("manifest tensors must be a list")
    entries = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ContextServingError("malformed tensor receipt")
        name = row["name"]
        if name in entries:
            raise ContextServingError(f"duplicate tensor receipt: {name}")
        if row.get("complete") is not True or row.get("dtype") != "BF16":
            raise ContextServingError(f"incomplete or unsupported tensor receipt: {name}")
        if not _hex_sha(row.get("source_sha256")) or not _hex_sha(row.get("frame_sha256")):
            raise ContextServingError(f"invalid source/frame SHA: {name}")
        shape = row.get("shape")
        if not isinstance(shape, list) or not shape or any(type(d) is not int or d <= 0 for d in shape):
            raise ContextServingError(f"invalid tensor shape: {name}")
        if row.get("source_bytes") != 2 * __import__("math").prod(shape):
            raise ContextServingError(f"invalid BF16 source byte count: {name}")
        if type(row.get("frame_bytes")) is not int or row["frame_bytes"] <= 0:
            raise ContextServingError(f"invalid BCTX frame byte count: {name}")
        entries[name] = row

    if config_loader is None:
        try:
            from transformers import AutoConfig
        except ImportError as exc:
            raise ContextServingError("transformers is required") from exc
        config_loader = lambda path: AutoConfig.from_pretrained(str(path.parent), local_files_only=True)
    config = config_loader(config_path)
    default_model_factory = model_factory is None
    if model_factory is None:
        model_factory = _default_factory
    if dtype is None:
        dtype = torch.bfloat16
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    _apply_attention_implementation(config, attention_implementation)
    empty_factory = empty_weights_factory or _empty_weights
    with empty_factory():
        with _default_dtype(dtype):
            model = model_factory(config, attention_implementation)

    generation_config_path = metadata_dir / "generation_config.json"
    generation_config = None
    generation_config_sha256 = None
    if generation_config_path.is_file():
        generation_config_bytes = generation_config_path.read_bytes()
        generation_asset = manifest.get("metadata_assets", {}).get("sidecars/generation_config.json")
        if (not isinstance(generation_asset, dict)
                or len(generation_config_bytes) != generation_asset.get("bytes")
                or hashlib.sha256(generation_config_bytes).hexdigest() != generation_asset.get("sha256")):
            raise ContextServingError("generation_config.json differs from verified metadata pin")
        generation_config_sha256 = hashlib.sha256(generation_config_bytes).hexdigest()
        if generation_config_loader is not None:
            generation_config = generation_config_loader(metadata_dir)
        else:
            try:
                from transformers import GenerationConfig
            except ImportError as exc:
                raise ContextServingError("transformers is required to load generation_config.json") from exc
            generation_config = GenerationConfig.from_pretrained(
                str(metadata_dir), local_files_only=True,
            )
    elif generation_config_loader is not None:
        generation_config = generation_config_loader(metadata_dir)
    elif default_model_factory:
        raise ContextServingError("package lacks required generation_config.json")
    else:
        generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        model.generation_config = generation_config

    params = _param_names(model)
    if not params:
        raise ContextServingError("model has no named parameters")
    missing = sorted(set(params) - set(entries))
    if missing:
        raise ContextServingError(f"package missing required model parameters: {missing[:12]}")
    unexpected_shapes = []
    for name, p in params.items():
        row = entries[name]
        if tuple(row["shape"]) != tuple(p.shape):
            unexpected_shapes.append((name, tuple(row["shape"]), tuple(p.shape)))
    if unexpected_shapes:
        raise ContextServingError(f"manifest/model shape mismatch: {unexpected_shapes[:8]}")

    if tensor_factory is None:
        try:
            from georefine_context_runtime.bitexact_context_gpu import CudaContextTensor
        except ImportError as exc:
            raise ContextServingError("CUDA BCTX tensor backend unavailable") from exc
        tensor_factory = CudaContextTensor
    host_decode = host_decoder or _load_host_decoder(package)
    consumed: dict[str, dict[str, Any]] = {}
    resident_context_bytes = 0
    replacements: dict[str, nn.Module] = {}
    compressed_names = set()
    def create_context(name):
        row = entries[name]
        frame_path = _safe_frame_path(package, row.get("frame"))
        try:
            frame = frame_path.read_bytes()
        except OSError as exc:
            raise ContextServingError(f"cannot read BCTX frame for {name}: {exc}") from exc
        if len(frame) != row.get("frame_bytes") or hashlib.sha256(frame).hexdigest() != row["frame_sha256"]:
            raise ContextServingError(f"BCTX frame identity mismatch: {name}")
        context = tensor_factory(frame, stride=stride, device=device)
        if tuple(context.shape) != _frame_shape(row["shape"]):
            raise ContextServingError(f"indexed BCTX shape mismatch: {name}")
        if context.source_sha256 != row["source_sha256"]:
            raise ContextServingError(f"BCTX decoded-source identity mismatch: {name}")
        if getattr(context, "frame_sha256", None) != row["frame_sha256"]:
            raise ContextServingError(f"BCTX backend frame identity mismatch: {name}")
        resident = getattr(context, "resident_bytes", None)
        if type(resident) is not int or resident < 0:
            raise ContextServingError(f"BCTX backend lacks valid resident_bytes: {name}")
        record = {
            "kind": None,
            "shape": list(row["shape"]),
            "frame_shape": list(_frame_shape(row["shape"])),
            "source_sha256": row["source_sha256"],
            "frame_sha256": row["frame_sha256"],
            "frame_bytes": row["frame_bytes"],
            "resident_context_bytes": resident,
        }
        return context, record, resident

    module_names = dict(model.named_modules())
    module_rows = list(module_names.items())
    compressed_module_names = []
    for module_name, module in module_rows:
        weight_name = f"{module_name}.weight" if module_name else "weight"
        if weight_name in entries and isinstance(module, (nn.Linear, nn.Embedding)):
            compressed_module_names.append((module_name, module, weight_name))
    context_names = [row[2] for row in compressed_module_names]
    context_by_name: dict[str, Any] = {}
    context_records: dict[str, dict[str, Any]] = {}
    resident_context_bytes = 0
    completed_contexts = 0

    def accept_context(name, result):
        nonlocal resident_context_bytes, completed_contexts
        context, record, resident = result
        context_by_name[name] = context
        context_records[name] = record
        resident_context_bytes += resident
        completed_contexts += 1
        if completed_contexts % 25 == 0 or completed_contexts == len(context_names):
            log(f"[bctx] indexed {completed_contexts}/{len(context_names)} compressed tensors")
            if progress_callback is not None:
                progress_callback(completed_contexts, len(context_names), name)

    if loader_workers == 1:
        for name in context_names:
            accept_context(name, create_context(name))
    elif context_names:
        with ThreadPoolExecutor(max_workers=loader_workers, thread_name_prefix="bctx-index") as pool:
            futures = {pool.submit(create_context, name): name for name in context_names}
            for future in as_completed(futures):
                accept_context(futures[future], future.result())

    for module_name, module, weight_name in compressed_module_names:
        if isinstance(module, nn.Linear):
            context = context_by_name[weight_name]
            consumed[weight_name] = context_records[weight_name]
            bias = module.bias
            if bias is not None:
                bias_name = f"{module_name}.bias"
                if bias_name not in entries:
                    raise ContextServingError(f"package missing required linear bias: {bias_name}")
                br = entries[bias_name]
                bframe = _safe_frame_path(package, br.get("frame")).read_bytes()
                if len(bframe) != br.get("frame_bytes") or hashlib.sha256(bframe).hexdigest() != br["frame_sha256"]:
                    raise ContextServingError(f"BCTX frame identity mismatch: {bias_name}")
                decoded_bias = host_decode(bframe)
                _validate_host_frame(decoded_bias, br, bias_name)
                bias_dtype = _runtime_parameter_dtype(model, bias_name, bias, dtype)
                b = _tensor_from_host(decoded_bias, bias_dtype, bias.shape)
                module.bias = nn.Parameter(b, requires_grad=False)
                consumed[bias_name] = {"kind": "dense_bias", "shape": list(br["shape"]), "runtime_dtype": str(bias_dtype), "source_sha256": br["source_sha256"], "frame_sha256": br["frame_sha256"], "frame_bytes": br["frame_bytes"], "resident_context_bytes": 0}
            replacements[module_name] = BctxLinear(context, module.bias)
            compressed_names.add(weight_name)
            consumed[weight_name]["kind"] = "compressed_linear"
        else:
            context = context_by_name[weight_name]
            consumed[weight_name] = context_records[weight_name]
            replacements[module_name] = BctxEmbedding(context, padding_idx=module.padding_idx)
            compressed_names.add(weight_name)
            consumed[weight_name]["kind"] = "compressed_embedding"
            consumed[weight_name]["kind"] = "compressed_embedding"

    # Install all replacements after the scan so parent and child module names are stable.
    for name, replacement in sorted(replacements.items(), key=lambda item: item[0].count(".")):
        parent, attr = _parent_and_attr(model, name)
        setattr(parent, attr, replacement)

    # Every other model parameter is a small/nonlinear package tensor (norms,
    # recurrent state parameters, projection biases, etc.); decode it from BCTX.
    for name, p in params.items():
        if name in compressed_names or name in consumed:
            continue
        row = entries[name]
        frame = _safe_frame_path(package, row.get("frame")).read_bytes()
        if len(frame) != row.get("frame_bytes") or hashlib.sha256(frame).hexdigest() != row["frame_sha256"]:
            raise ContextServingError(f"BCTX frame identity mismatch: {name}")
        decoded_value = host_decode(frame)
        _validate_host_frame(decoded_value, row, name)
        runtime_dtype = _runtime_parameter_dtype(model, name, p, dtype)
        value = _tensor_from_host(decoded_value, runtime_dtype, p.shape)
        parent, attr = _parent_and_attr(model, name)
        parent._parameters[attr] = nn.Parameter(value, requires_grad=False)
        consumed[name] = {"kind": "dense_parameter", "shape": list(row["shape"]), "runtime_dtype": str(runtime_dtype), "source_sha256": row["source_sha256"], "frame_sha256": row["frame_sha256"], "frame_bytes": row["frame_bytes"], "resident_context_bytes": 0}

    remaining_meta = [f"parameter:{n}" for n, p in model.named_parameters() if p.is_meta]
    remaining_meta += [f"buffer:{n}" for n, b in model.named_buffers() if b.is_meta]
    if remaining_meta:
        raise ContextServingError(f"model contains unfilled meta tensors: {remaining_meta[:12]}")
    model.to(device)
    unused = sorted(set(entries) - set(consumed))
    dense_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    dense_buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    if generation_config is None:
        generation_config_runtime = None
    elif callable(getattr(generation_config, "to_dict", None)):
        generation_config_runtime = generation_config.to_dict()
    elif isinstance(generation_config, dict):
        generation_config_runtime = dict(generation_config)
    else:
        generation_config_runtime = {
            key: getattr(generation_config, key)
            for key in ("eos_token_id", "pad_token_id", "bos_token_id", "do_sample", "top_k",
                        "top_p", "temperature", "min_new_tokens", "max_new_tokens")
            if hasattr(generation_config, key)
        }
    receipt = {
        "schema": "georefine.bctx_serving_load.v1",
        "status": "ok",
        "format": "BCTX v1",
        "device": str(device),
        "dtype_default": str(dtype),
        "attention_implementation": attention_implementation,
        "package_manifest": str(manifest_path),
        "model_config": str(config_path),
        "metadata_dir": str(metadata_dir),
        "generation_config_path": str(generation_config_path) if generation_config is not None else None,
        "generation_config_sha256": generation_config_sha256,
        "generation_config": generation_config_runtime,
        "tensor_count": len(entries),
        "consumed_count": len(consumed),
        "unused_package_tensors": unused,
        "compressed_tensor_count": len(compressed_names),
        "resident_context_bytes": resident_context_bytes,
        "dense_parameter_bytes": dense_param_bytes,
        "buffer_bytes": dense_buffer_bytes,
        "consumed_tensors": consumed,
        "restored_bf16_control": False,
        "transcoded": False,
    }
    return model, receipt


__all__ = ["BctxEmbedding", "BctxLinear", "ContextServingError", "load_bctx_model"]
