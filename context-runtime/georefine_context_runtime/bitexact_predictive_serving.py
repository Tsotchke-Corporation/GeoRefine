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



import gc
import importlib.util
import threading
import sys

PREDICTIVE_SCHEMA = "bitexact-predictive-context-package-v1"
LEGACY_PREDICTIVE_SCHEMA = "private-predictive-context-package-v1"
_packaged_backend_lock = threading.RLock()


def _load_module(path: Path, name: str):
    import types
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition('.')[0]
    sys.modules[name] = module
    try:
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _candidate_frame(package: Path, row, selected=None):
    rel = selected.get("frame") if selected else row.get("frame")
    path = _safe_frame_path(package, rel)
    resolved_root = package.resolve()
    resolved = path.resolve()
    if resolved_root not in resolved.parents or path.is_symlink() or not path.is_file():
        raise ContextServingError("tensor frame must be a regular file inside candidate package")
    size = selected.get("frame_bytes") if selected else row.get("frame_bytes")
    digest = selected.get("frame_sha256") if selected else row.get("frame_sha256")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ContextServingError(f"cannot read candidate-package frame: {exc}") from exc
    if type(size) is not int or len(payload) != size or not _hex_sha(digest) or hashlib.sha256(payload).hexdigest() != digest:
        raise ContextServingError(f"candidate-package frame identity mismatch: {row.get('name')}")
    return payload


def _row_frame_shape(row):
    return _frame_shape(row["shape"])


def _words_2d(value, row, name):
    import numpy as np
    arr = np.ascontiguousarray(value)
    if arr.dtype.kind != "u" or arr.dtype.itemsize != 2 or arr.nbytes != row["source_bytes"]:
        raise ContextServingError(f"decoded BF16 words have invalid dtype/size: {name}")
    return arr.reshape(_row_frame_shape(row))


def _sha_words(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<u2").tobytes()).hexdigest()


class _LockedContext:
    def __init__(self, native, *, shape, source_sha256, frame_sha256, resident_bytes):
        self.native = native
        self.shape = tuple(shape)
        self.source_sha256 = source_sha256
        self.frame_sha256 = frame_sha256
        self.resident_bytes = resident_bytes
        self._decode_lock = threading.RLock()

    def decode(self, *, check=True):
        with self._decode_lock:
            return self.native.decode(check=check)

    def gather_rows(self, ids, *, check=True):
        with self._decode_lock:
            gather = getattr(self.native, "gather_rows", None)
            if callable(gather):
                return gather(ids, check=check)
            return self.decode(check=check)[ids]


class PredictiveContextTensor:
    """PPCX weight that resolves named runtime contexts without retaining dense weights."""
    def __init__(self, native, references, *, predictor, coefficients, transform,
                 shape, source_sha256, frame_sha256, resident_bytes):
        self.native = native
        self.references = tuple(references)
        self.predictor = predictor
        self.coefficients = coefficients
        self.transform = transform
        self.shape = tuple(shape)
        self.source_sha256 = source_sha256
        self.frame_sha256 = frame_sha256
        self.resident_bytes = resident_bytes
        self._decode_lock = threading.RLock()

    def decode(self, *, check=True):
        with self._decode_lock:
            decoded_refs = [ctx.decode(check=check) for ctx in self.references]
            if len(decoded_refs) == 1:
                reference = decoded_refs[0]
            else:
                reference = self.predictor(decoded_refs[0], decoded_refs[1], self.coefficients)
            value = self.native.decode(reference, check=check)
            if self.transform == "target transpose":
                if isinstance(value, torch.Tensor):
                    value = value.transpose(-2, -1).contiguous()
                else:
                    import numpy as np
                    value = np.ascontiguousarray(value.T)
            return value

    def gather_rows(self, ids, *, check=True):
        value = self.decode(check=check)
        return value[ids]


class _NullContext:
    def __enter__(self): return None
    def __exit__(self, *exc): return False


def _default_predictive_factories(package: Path):
    # The decoder backend is shipped with the immutable package.
    package = package.resolve()
    gpu_root = package / "decoder"
    cache_root = package.parent / ".scratch" / "bitexact-predictive-native-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    configured_cache = Path(os.environ.get("BITEXACT_PREDICTIVE_CACHE_DIR", str(cache_root))).resolve()
    if configured_cache == package or package in configured_cache.parents:
        raise ContextServingError("predictive native cache must be outside the immutable package")
    os.environ["BITEXACT_PREDICTIVE_CACHE_DIR"] = str(configured_cache)
    # Import from the selected package without bytecode writes or shared module names.
    # Bind both the path and source bytes so different packages cannot reuse a backend.
    sources = [gpu_root / name for name in ('bitexact_predictive_codec.py',
               'bitexact_predictive_gpu.py', 'bitexact_predictive_gpu_kernels.py')]
    identity = hashlib.sha256(str(gpu_root).encode())
    for source in sources:
        identity.update(source.read_bytes())
    namespace = '_predictive_package_' + identity.hexdigest()
    try:
        import types
        with _packaged_backend_lock:
            root = sys.modules.get(namespace)
            if root is None:
                root = types.ModuleType(namespace)
                root.__path__ = [str(gpu_root)]
                root.__package__ = namespace
                sys.modules[namespace] = root
                try:
                    codec = _load_module(sources[0], namespace + '.bitexact_predictive_codec')
                    root.bitexact_predictive_codec = codec
                    gpu_backend = _load_module(sources[1], namespace + '.bitexact_predictive_gpu')
                    root.bitexact_predictive_gpu = gpu_backend
                except BaseException:
                    for name in (namespace, namespace + '.bitexact_predictive_codec',
                                 namespace + '.bitexact_predictive_gpu'):
                        sys.modules.pop(name, None)
                    raise
            else:
                gpu_backend = root.bitexact_predictive_gpu
    except (OSError, ImportError) as exc:
        raise ContextServingError("CUDA PPCX tensor backend unavailable") from exc
    gpu_backend.CACHE_ROOT = configured_cache
    codec = getattr(gpu_backend, "codec", None)
    if codec is not None:
        codec.CACHE_ROOT = configured_cache
        codec.BUILD = configured_cache / "build"
    return gpu_backend.CudaPredictiveTensor


def _default_context_factory():
    try:
        from georefine_context_runtime.bitexact_context_gpu import CudaContextTensor
        return CudaContextTensor
    except ImportError:
        pass
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        from georefine_context_runtime.bitexact_context_gpu import CudaContextTensor
    except ImportError as exc:
        raise ContextServingError("CUDA BCTX tensor backend unavailable") from exc
    return CudaContextTensor


def load_predictive_model(
    package: str | Path,
    *,
    device: str = "cuda:0",
    dtype=None,
    attention_implementation: str = "eager",
    stride: int = 1024,
    tensor_factory: Callable[..., Any] | None = None,
    context_factory: Callable[..., Any] | None = None,
    predictive_factory: Callable[..., Any] | None = None,
    model_factory: Callable[..., Any] | None = None,
    config_loader: Callable[[Path], Any] | None = None,
    generation_config_loader: Callable[[Path], Any] | None = None,
    empty_weights_factory: Callable[[], Any] | None = None,
    host_decoder: Callable[[bytes], Any] | None = None,
    mix_predictor: Callable[..., Any] | None = None,
    package_verifier: Callable[..., Any] | None = None,
    metadata_cache_dir: str | Path | None = None,
    log: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    loader_workers: int = 1,
    verified_cpu_source=None,
    cpu_source_names=None,
):
    """Load a predictive package using only verified frames inside that package."""
    import numpy as np
    import torch
    from torch import nn
    from concurrent.futures import ThreadPoolExecutor, as_completed

    log = log or (lambda message: None)
    if isinstance(loader_workers, bool) or not isinstance(loader_workers, int) or loader_workers < 1:
        raise ValueError("loader_workers must be a positive integer")
    package = Path(package).resolve()
    if verified_cpu_source is None and cpu_source_names is not None:
        raise ContextServingError('cpu_source_names requires a verified CPU source')
    manifest_path = package / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContextServingError(f"cannot read predictive package manifest: {exc}") from exc
    if manifest.get("schema") not in (PREDICTIVE_SCHEMA, LEGACY_PREDICTIVE_SCHEMA) or manifest.get("complete") is not True:
        raise ContextServingError("unsupported or incomplete predictive package manifest")
    if manifest.get("schema") == PREDICTIVE_SCHEMA and manifest.get("verify_on_load") is not True:
        raise ContextServingError("public predictive package lacks stable verification metadata")

    cache_dir = (Path(metadata_cache_dir) if metadata_cache_dir is not None
                 else package.parent / ".scratch" / "predictive-serving-metadata")
    metadata_dir = prepare_metadata(package, cache_dir)
    config_path = metadata_dir / "config.json"
    if not config_path.is_file():
        raise ContextServingError("verified package config.json is missing")

    package_tools = _load_module(package / "decoder" / "predictor_package.py", "serving_predictor_package")
    try:
        rows_doc = package_tools.read_baseline_manifest(package)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ContextServingError(f"cannot read verified packaged baseline manifest: {exc}") from exc
    raw_rows = rows_doc.get("tensors")
    if not isinstance(raw_rows, list):
        raise ContextServingError("baseline-manifest tensors must be a list")
    entries = {}
    for row in raw_rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ContextServingError("malformed baseline tensor receipt")
        name = row["name"]
        if name in entries:
            raise ContextServingError(f"duplicate baseline tensor receipt: {name}")
        if row.get("dtype") != "BF16" or not _hex_sha(row.get("source_sha256")):
            raise ContextServingError(f"unsupported dtype/source SHA: {name}")
        shape = row.get("shape")
        if not isinstance(shape, list) or not shape or any(type(d) is not int or d <= 0 for d in shape):
            raise ContextServingError(f"invalid tensor shape: {name}")
        if type(row.get("source_bytes")) is not int or row["source_bytes"] != 2 * __import__("math").prod(shape):
            raise ContextServingError(f"invalid BF16 source byte count: {name}")
        entries[name] = dict(row)

    selected = manifest.get("selected_candidates")
    if not isinstance(selected, dict):
        raise ContextServingError("selected_candidates must be an object")
    for name, candidate in selected.items():
        if name not in entries or not isinstance(candidate, dict) or candidate.get("name") != name:
            raise ContextServingError(f"malformed selected candidate: {name}")
        row = entries[name]
        if candidate.get("schema") != manifest.get("experimental_codec"):
            raise ContextServingError(f"candidate codec schema mismatch: {name}")
        if candidate.get("source_sha256") != row["source_sha256"] or candidate.get("shape") != row["shape"]:
            raise ContextServingError(f"candidate source identity mismatch: {name}")
        if candidate.get("mode") not in ("single", "mixbf16"):
            raise ContextServingError(f"unsupported candidate mode: {name}")
        if candidate.get("transform") not in ("identity", "target transpose"):
            raise ContextServingError(f"unsupported candidate transform: {name}")
        refs = candidate.get("references")
        need = 1 if candidate["mode"] == "single" else 2
        if not isinstance(refs, list) or len(refs) != need:
            raise ContextServingError(f"invalid candidate references: {name}")
        for ref in refs:
            if (not isinstance(ref, dict) or ref.get("name") not in entries
                    or ref["name"] == name or ref.get("sha256") != entries[ref["name"]]["source_sha256"]):
                raise ContextServingError(f"candidate reference SHA/name mismatch: {name}")
        row["frame"] = candidate.get("frame")
        row["frame_bytes"] = candidate.get("frame_bytes")
        row["frame_sha256"] = candidate.get("frame_sha256")

    # Validate every reference graph before starting worker threads.
    visiting, visited = set(), set()
    def visit(name):
        if name in visiting:
            raise ContextServingError("candidate reference cycle")
        if name in visited:
            return
        visiting.add(name)
        c = selected.get(name)
        if c:
            for ref in c["references"]:
                visit(ref["name"])
        visiting.remove(name); visited.add(name)
    for name in entries:
        visit(name)
    cpu_names=set()
    if verified_cpu_source is not None:
        if Path(verified_cpu_source.package).resolve()!=package or verified_cpu_source.manifest_sha256!=hashlib.sha256(manifest_path.read_bytes()).hexdigest():
            raise ContextServingError('CPU resolver package/manifest identity mismatch')
        cpu_names=set(entries) if cpu_source_names is None else set(cpu_source_names)
        if not cpu_names.issubset(entries):raise ContextServingError('CPU resolver requested unknown tensors')
        if any(name not in cpu_names and any(r['name'] in cpu_names for r in candidate['references'])
               for name,candidate in selected.items()):
            raise ContextServingError('CPU source selection must include dependent predictive tensors')
        if cpu_source_names is None and torch.device(device).type!='cpu':
            raise ContextServingError('all-CPU lazy model requires device=cpu; select explicit tensors for conversion')

    # Reuse the package's own verifier and decoders; all source bytes come from this package.
    if package_verifier is not None:
        package_verifier(package, decode_frames=False, check_inventory=True)
    else:
        package_tools.verify(package, decode_frames=False, check_inventory=True)
    base_tools = package_tools.baseline_module(package=package)
    base_codec = base_tools.decode_codec(package, cache_dir=package.parent / ".scratch" / "predictive-base-codec")
    pair_codec = package_tools.load_pair_codec(package, package.parent / ".scratch" / "predictive-pair-codec")
    mix_module = package_tools.load_mix_helper(package) if any(c["mode"] == "mixbf16" for c in selected.values()) else None
    if mix_predictor is None:
        mix_predictor = mix_module.predict if mix_module is not None else None

    if config_loader is None:
        try:
            from transformers import AutoConfig
        except ImportError as exc:
            raise ContextServingError("transformers is required") from exc
        config_loader = lambda path: AutoConfig.from_pretrained(str(path.parent), local_files_only=True)
    default_model_factory = model_factory is None
    if model_factory is None:
        model_factory = _default_factory
    if dtype is None:
        dtype = torch.bfloat16
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    config = config_loader(config_path)
    _apply_attention_implementation(config, attention_implementation)
    empty_factory = empty_weights_factory or _empty_weights
    with empty_factory():
        with _default_dtype(dtype):
            model = model_factory(config, attention_implementation)
    generation_config_path = metadata_dir / "generation_config.json"
    generation_config = None
    generation_config_sha256 = None
    if generation_config_path.is_file():
        raw = generation_config_path.read_bytes()
        pin = manifest.get("metadata_assets", {}).get("sidecars/generation_config.json")
        if not isinstance(pin, dict) or len(raw) != pin.get("bytes") or hashlib.sha256(raw).hexdigest() != pin.get("sha256"):
            raise ContextServingError("generation_config differs from verified metadata pin")
        generation_config_sha256 = hashlib.sha256(raw).hexdigest()
        if generation_config_loader is not None:
            generation_config = generation_config_loader(metadata_dir)
        else:
            from transformers import GenerationConfig
            generation_config = GenerationConfig.from_pretrained(str(metadata_dir), local_files_only=True)
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
    shape_errors = [(n, tuple(entries[n]["shape"]), tuple(p.shape)) for n,p in params.items()
                    if tuple(entries[n]["shape"]) != tuple(p.shape)]
    if shape_errors:
        raise ContextServingError(f"manifest/model shape mismatch: {shape_errors[:8]}")

    if verified_cpu_source is not None and cpu_names==set(entries):
        def unavailable_native(*args,**kwargs):
            raise ContextServingError('CPU-only lazy model attempted native checkpoint construction')
        context_factory=context_factory or unavailable_native
        predictive_factory=predictive_factory or unavailable_native
    elif tensor_factory is not None:
        context_factory = context_factory or tensor_factory
        predictive_factory = predictive_factory or tensor_factory
    else:
        context_factory = context_factory or _default_context_factory()
        predictive_factory = predictive_factory or _default_predictive_factories(package)
    if host_decoder is None:
        host_decoder = base_codec.decode

    # Every frame read is rooted in the candidate package; selected rows already contain overrides.
    def frame_for(name):
        return _candidate_frame(package, entries[name], selected.get(name))

    cpu_cache = {}
    cpu_lock = threading.RLock()
    coeff_cpu = {}
    coeff_cuda = {}
    def coeff_words(name):
        if name not in coeff_cpu:
            candidate = selected[name]
            pin = candidate.get("coefficients")
            if not isinstance(pin, dict):
                raise ContextServingError(f"mixed candidate coefficient metadata missing: {name}")
            try:
                words = package_tools.read_coefficients(package, candidate)
            except (OSError, ValueError) as exc:
                raise ContextServingError(f"mixed coefficient identity/shape mismatch: {name}: {exc}") from exc
            expected_shape = [_row_frame_shape(entries[candidate["references"][0]["name"]])[0], 3]
            if (not isinstance(words, np.ndarray) or words.dtype != np.uint16
                    or tuple(words.shape) != tuple(expected_shape)
                    or tuple(pin.get("shape", ())) != tuple(expected_shape)):
                raise ContextServingError(f"mixed coefficient identity/shape mismatch: {name}")
            words = np.ascontiguousarray(words, dtype="<u2")
            if np.any((words.astype(np.uint32)&0x7f80)==0x7f80):
                raise ContextServingError(f"nonfinite BF16 mixed coefficient: {name}")
            coeff_cpu[name] = words
        return coeff_cpu[name]

    def decode_cpu(name, stack=()):
        if verified_cpu_source is not None:
            words=verified_cpu_source.resolve(name)
            if words.shape!=tuple(entries[name]['shape']) or _sha_words(words)!=entries[name]['source_sha256']:
                raise ContextServingError(f'CPU resolver source identity mismatch: {name}')
            # Dense parameters must own writable storage, separate from the verified cache.
            return words.copy()
        with cpu_lock:
            if name in cpu_cache:
                return cpu_cache[name]
            if name in stack:
                raise ContextServingError("candidate reference cycle")
            row = entries[name]
            c = selected.get(name)
            if c is None:
                decoded = host_decoder(frame_for(name))
                words = _words_2d(decoded, row, name).reshape(row["shape"])
            else:
                ref_arrays = [decode_cpu(ref["name"], stack+(name,)) for ref in c["references"]]
                ref_words = [_words_2d(value, entries[ref["name"]], ref["name"]) for value,ref in zip(ref_arrays,c["references"])]
                if c["mode"] == "mixbf16":
                    coeff = coeff_words(name)
                    derived = np.ascontiguousarray(mix_predictor(ref_words[0], ref_words[1], coeff), dtype="<u2")
                    if _sha_words(derived) != c.get("derived_reference_sha256"):
                        raise ContextServingError(f"derived CPU reference checksum mismatch: {name}")
                    reference = derived
                else:
                    reference = ref_words[0]
                decoded = pair_codec.decode(frame_for(name), reference)
                frame_words = np.ascontiguousarray(decoded, dtype="<u2")
                words = np.ascontiguousarray(frame_words.T if c["transform"] == "target transpose" else frame_words).reshape(row["shape"])
            words = np.ascontiguousarray(words, dtype="<u2")
            if words.nbytes != row["source_bytes"] or _sha_words(words.reshape(_row_frame_shape(row))) != row["source_sha256"]:
                raise ContextServingError(f"decoded tensor source SHA mismatch: {name}")
            cpu_cache[name] = words
            return words

    module_names = dict(model.named_modules())
    compressed_modules = []
    for module_name,module in module_names.items():
        weight_name = f"{module_name}.weight" if module_name else "weight"
        if weight_name in entries and isinstance(module,(nn.Linear,nn.Embedding)):
            compressed_modules.append((module_name,module,weight_name))
    needed = {name for _,_,name in compressed_modules}
    def add_dependencies(name):
        c=selected.get(name)
        if c:
            for ref in c["references"]:
                needed.add(ref["name"]); add_dependencies(ref["name"])
    for name in tuple(needed): add_dependencies(name)
    locks={name:threading.RLock() for name in needed}
    contexts={}
    def create_context(name,stack=()):
        with locks[name]:
            if name in contexts: return contexts[name]
            if name in stack: raise ContextServingError("candidate reference cycle")
            if name in cpu_names:
                row=entries[name];source=verified_cpu_source.tensor(name)
                if (tuple(source.shape)!=tuple(row['shape']) or source.source_sha256!=row['source_sha256']
                    or source.frame_sha256!=row['frame_sha256'] or source.resident_bytes!=0 or source.device!='cpu'):
                    raise ContextServingError(f'CPU lazy tensor identity mismatch: {name}')
                context=_LockedContext(source,shape=row['shape'],source_sha256=row['source_sha256'],frame_sha256=row['frame_sha256'],resident_bytes=0)
                context.references=tuple(create_context(r['name'],stack+(name,)) for r in selected.get(name,{}).get('references',()))
                contexts[name]=context
                return context
            row=entries[name]; frame=frame_for(name); candidate=selected.get(name)
            if candidate is None:
                native=context_factory(frame,stride=stride,device=device)
                if tuple(native.shape)!=_row_frame_shape(row) or native.source_sha256!=row["source_sha256"] or getattr(native,"frame_sha256",None)!=row["frame_sha256"]:
                    raise ContextServingError(f"BCTX context identity mismatch: {name}")
                resident=getattr(native,"resident_bytes",None)
                if type(resident) is not int or resident<0: raise ContextServingError(f"BCTX resident byte accounting missing: {name}")
                context=_LockedContext(native,shape=row["shape"],source_sha256=row["source_sha256"],frame_sha256=row["frame_sha256"],resident_bytes=resident)
            else:
                refs=[create_context(ref["name"],stack+(name,)) for ref in candidate["references"]]
                cpu_refs=[decode_cpu(ref["name"]) for ref in candidate["references"]]
                ref_frame=[_words_2d(value,entries[ref["name"]],ref["name"]) for value,ref in zip(cpu_refs,candidate["references"])]
                if candidate["mode"]=="mixbf16":
                    derived=np.ascontiguousarray(mix_predictor(ref_frame[0],ref_frame[1],coeff_words(name)),dtype="<u2")
                    if _sha_words(derived)!=candidate.get("derived_reference_sha256"):
                        raise ContextServingError(f"derived reference checksum mismatch: {name}")
                    reference=derived
                else: reference=ref_frame[0]
                native=predictive_factory(frame,reference,stride=stride,device=device)
                frame_shape=_row_frame_shape(row)
                if candidate["transform"]=="target transpose": frame_shape=(frame_shape[1],frame_shape[0])
                if tuple(native.shape)!=frame_shape: raise ContextServingError(f"PPCX frame geometry mismatch: {name}")
                if getattr(native,"frame_sha256",None)!=candidate["frame_sha256"]: raise ContextServingError(f"PPCX frame identity mismatch: {name}")
                own=getattr(native,"resident_bytes",None)
                if type(own) is not int or own<0: raise ContextServingError(f"PPCX resident byte accounting missing: {name}")
                coeff_tensor=None
                if candidate["mode"]=="mixbf16":
                    if name not in coeff_cuda:
                        coeff_cuda[name]=torch.from_numpy(coeff_words(name).copy()).view(torch.bfloat16).to(device)
                    coeff_tensor=coeff_cuda[name]; own+=coeff_tensor.numel()*coeff_tensor.element_size()
                refs_context=refs
                if candidate["mode"]=="single": predictor=None
                else: predictor=mix_predictor
                context=PredictiveContextTensor(native,refs_context,predictor=predictor,coefficients=coeff_tensor,transform=candidate["transform"],shape=row["shape"],source_sha256=row["source_sha256"],frame_sha256=candidate["frame_sha256"],resident_bytes=own)
            contexts[name]=context
            return context

    completed=set()
    if loader_workers==1:
        for name in sorted(needed): create_context(name); completed.add(name)
    else:
        with ThreadPoolExecutor(max_workers=loader_workers,thread_name_prefix="predictive-context") as pool:
            futures={pool.submit(create_context,name):name for name in needed}
            for fut in as_completed(futures):
                fut.result(); completed.add(futures[fut])
                if progress_callback is not None: progress_callback(len(completed),len(needed),futures[fut])
    # No host dense references remain reachable from the model or loader after context indexing.
    cpu_cache.clear(); coeff_cpu.clear(); gc.collect()

    consumed={}; replacements={}; compressed=set()
    for module_name,module,weight_name in compressed_modules:
        context=contexts[weight_name]
        row=entries[weight_name]; candidate=selected.get(weight_name)
        record={"kind":None,"shape":list(row["shape"]),"source_sha256":row["source_sha256"],"frame_sha256":context.frame_sha256,"frame_bytes":row["frame_bytes"],"resident_context_bytes":context.resident_bytes}
        consumed[weight_name]=record
        if isinstance(module,nn.Linear):
            bias=module.bias
            if bias is not None:
                bias_name=f"{module_name}.bias"; br=entries.get(bias_name)
                if br is None: raise ContextServingError(f"package missing required linear bias: {bias_name}")
                bw=decode_cpu(bias_name)
                cpu_cache.pop(bias_name, None)
                bt=_tensor_from_host(bw, _runtime_parameter_dtype(model,bias_name,bias,dtype), bias.shape)
                module.bias=nn.Parameter(bt,requires_grad=False)
                consumed[bias_name]={"kind":"dense_bias","shape":list(br["shape"]),"runtime_dtype":str(bt.dtype),"source_sha256":br["source_sha256"],"frame_sha256":br["frame_sha256"],"resident_context_bytes":0}
            replacements[module_name]=BctxLinear(context,module.bias); record["kind"]="compressed_linear"
        else:
            replacements[module_name]=BctxEmbedding(context,padding_idx=module.padding_idx); record["kind"]="compressed_embedding"
        compressed.add(weight_name)
    for name,replacement in sorted(replacements.items(),key=lambda x:x[0].count(".")):
        parent,attr=_parent_and_attr(model,name); setattr(parent,attr,replacement)

    for name,p in params.items():
        if name in compressed or name in consumed: continue
        row=entries[name]; words=decode_cpu(name)
        cpu_cache.pop(name, None)
        _validate_host_frame(words.reshape(_row_frame_shape(row)),row,name)
        value=_tensor_from_host(words,_runtime_parameter_dtype(model,name,p,dtype),p.shape)
        parent,attr=_parent_and_attr(model,name); parent._parameters[attr]=nn.Parameter(value,requires_grad=False)
        c=selected.get(name)
        consumed[name]={"kind":"dense_parameter","shape":list(row["shape"]),"runtime_dtype":str(value.dtype),"source_sha256":row["source_sha256"],"frame_sha256":row["frame_sha256"],"frame_bytes":row["frame_bytes"],"resident_context_bytes":0}
    remaining=[f"parameter:{n}" for n,p in model.named_parameters() if p.is_meta]
    remaining += [f"buffer:{n}" for n,b in model.named_buffers() if b.is_meta]
    if remaining: raise ContextServingError(f"model contains unfilled meta tensors: {remaining[:12]}")
    model.to(device)
    if generation_config is None: generation_runtime=None
    elif callable(getattr(generation_config,"to_dict",None)): generation_runtime=generation_config.to_dict()
    elif isinstance(generation_config,dict): generation_runtime=dict(generation_config)
    else: generation_runtime={k:getattr(generation_config,k) for k in ("eos_token_id","pad_token_id","bos_token_id","do_sample","top_k","top_p","temperature","min_new_tokens","max_new_tokens") if hasattr(generation_config,k)}
    receipt={"schema":"georefine.predictive_serving_load.v1","status":"ok","format":PREDICTIVE_SCHEMA,"device":str(device),"dtype_default":str(dtype),"attention_implementation":attention_implementation,"package_manifest":str(manifest_path),"model_config":str(config_path),"metadata_dir":str(metadata_dir),"generation_config_sha256":generation_config_sha256,"generation_config":generation_runtime,"consumed_tensors":consumed,"unused_tensors":sorted(set(entries)-set(consumed)),"resident_context_bytes":sum(ctx.resident_bytes for ctx in contexts.values()),"dense_parameter_bytes":sum(p.numel()*p.element_size() for p in model.parameters()),"dense_buffer_bytes":sum(b.numel()*b.element_size() for b in model.buffers()),"loader_workers":loader_workers,"context_count":len(contexts),"candidate_count":len(selected)}
    if verified_cpu_source is not None:
        receipt['cpu_lazy_context_count']=len(set(contexts)&cpu_names)
        receipt['cpu_resolver_manifest_sha256']=verified_cpu_source.manifest_sha256
        receipt['cpu_resolver_cache_bytes']=verified_cpu_source.cache_bytes
    return model,receipt


def load_predictive_mtp(package, model, *, device="cuda:0", checkpoint_stride=1024,
                        cache_dir=None, tensor_cls=None):
    """Load the unchanged BCTX MTP head and return it with its MTP 1.5 speculator.

    The tensor manifest is the hash-checked, decompressed baseline manifest stored
    outside the immutable predictive package. MTP tensors must remain baseline
    BCTX frames; a predictive candidate replacing any MTP tensor is rejected.
    """
    import hashlib

    package = Path(package).resolve()
    manifest_path = package / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContextServingError(f"cannot read predictive package manifest: {exc}") from exc
    if (manifest.get("schema") not in (PREDICTIVE_SCHEMA, LEGACY_PREDICTIVE_SCHEMA)
            or manifest.get("complete") is not True):
        raise ContextServingError("unsupported or incomplete predictive package manifest")
    if manifest.get("schema") == PREDICTIVE_SCHEMA and manifest.get("verify_on_load") is not True:
        raise ContextServingError("public predictive package lacks stable verification metadata")
    package_tools = _load_module(package / "decoder" / "predictor_package.py", "predictive_mtp_package")
    try:
        package_tools.verify(package, decode_frames=False, check_inventory=True)
        baseline_doc = package_tools.read_baseline_manifest(package)
        raw = lzma.decompress((package / "baseline-manifest.json.xz").read_bytes())
    except (OSError, ValueError, json.JSONDecodeError, lzma.LZMAError) as exc:
        raise ContextServingError(f"predictive baseline manifest verification failed: {exc}") from exc
    if manifest.get("baseline_manifest_sha256") != hashlib.sha256(raw).hexdigest():
        raise ContextServingError("predictive baseline manifest checksum mismatch")
    if manifest.get("baseline_manifest_bytes") != len(raw):
        raise ContextServingError("predictive baseline manifest length mismatch")

    try:
        from georefine_context_runtime.bitexact_context_mtp import MTP_SHAPES, MTPSpeculator, load_bctx_mtp
    except ImportError:
        scripts_dir = Path(__file__).resolve().parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from georefine_context_runtime.bitexact_context_mtp import MTP_SHAPES, MTPSpeculator, load_bctx_mtp
    rows = {row.get("name"): row for row in baseline_doc.get("tensors", [])}
    selected = manifest.get("selected_candidates", {})
    if set(MTP_SHAPES) - set(rows):
        raise ContextServingError("predictive baseline manifest lacks unchanged MTP tensors")
    for name in MTP_SHAPES:
        row = rows[name]
        if name in selected:
            raise ContextServingError(f"MTP tensor is not an unchanged BCTX frame: {name}")
        try:
            rel = _safe_frame_name(row.get("frame"))
            frame_path = package.joinpath(*PurePosixPath(rel).parts)
            if (frame_path.is_symlink() or not frame_path.is_file()
                    or frame_path.stat().st_size != row.get("frame_bytes")
                    or hashlib.sha256(frame_path.read_bytes()).hexdigest() != row.get("frame_sha256")):
                raise ValueError("MTP baseline frame integrity mismatch")
        except (OSError, ValueError) as exc:
            raise ContextServingError(f"invalid unchanged MTP frame {name}: {exc}") from exc

    cache_root = Path(cache_dir) if cache_dir is not None else package.parent / ".scratch" / "predictive-mtp-manifest"
    cache_root = cache_root.resolve()
    if cache_root == package or package in cache_root.parents:
        raise ContextServingError("MTP manifest cache must be outside the immutable package")
    cache_root.mkdir(parents=True, exist_ok=True)
    cached_manifest = cache_root / (hashlib.sha256(raw).hexdigest() + ".json")
    if not cached_manifest.exists() or cached_manifest.read_bytes() != raw:
        tmp = cached_manifest.with_name(cached_manifest.name + ".partial")
        with tmp.open("wb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, cached_manifest)
    text_config = getattr(getattr(model, "config", None), "text_config", None)
    if text_config is None:
        text_config = getattr(model, "config", None)
    if text_config is None:
        raise ContextServingError("predictive model has no text configuration for MTP")
    try:
        head = load_bctx_mtp(cached_manifest, package, text_config, device=device,
                             tensor_cls=tensor_cls, checkpoint_stride=checkpoint_stride)
        speculator = MTPSpeculator(model, head)
    except Exception as exc:
        raise ContextServingError(f"cannot load unchanged BCTX MTP 1.5 head: {exc}") from exc
    return head, speculator, dict(getattr(head, "_bctx_receipt", {}))


def _safe_frame_name(value):
    if not isinstance(value, str):
        raise ValueError("frame name must be a string")
    rel = PurePosixPath(value)
    if (rel.is_absolute() or not rel.parts or rel.parts[0] != "frames"
            or ".." in rel.parts or "." in rel.parts):
        raise ValueError("unsafe baseline frame name")
    return rel.as_posix()
