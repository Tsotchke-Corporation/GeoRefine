"""Opt-in CPU-process prototype for verified Context-to-TBE encoding.

This module is intentionally standalone and is not called by any loader. Each
spawned worker owns its own verified package resolver, and only NumPy arrays
and plain metadata cross the process boundary.
"""
from __future__ import annotations

import hashlib
import importlib
import multiprocessing as mp
from multiprocessing import util as mp_util
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
from typing import Iterator, Sequence


_RESOLVER_CACHE_BYTES = 512 * 1024**2
_MAX_WORKERS = 8
_MAX_SOURCE_INFLIGHT_BYTES = 8 * 1024**3
_WORKER_RESOLVER = None
_WORKER_FINALIZER = None
_WORKER_PACKAGE = None
_WORKER_MANIFEST_SHA256 = None


class TBEProcessError(RuntimeError):
    """Raised when verified CPU process encoding cannot be completed safely."""


@dataclass(frozen=True)
class TensorRequest:
    """One manifest tensor selected for exact CPU encoding."""

    name: str
    shape: tuple[int, int]
    source_sha256: str


def _valid_sha256(value: str) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def _read_manifest_sha256(package: Path) -> str:
    manifest = package / "manifest.json"
    try:
        raw = manifest.read_bytes()
    except OSError as exc:
        raise TBEProcessError(f"cannot read package manifest {manifest}: {exc}") from exc
    return hashlib.sha256(raw).hexdigest()


def _source_tensor_sha256(tensor) -> str:
    import torch

    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.bfloat16:
        raise TBEProcessError("verified Context tensor must decode to BF16")
    if tensor.device.type != "cpu":
        raise TBEProcessError("verified Context tensor must remain on CPU")
    raw = tensor.detach().contiguous().view(torch.int16).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _close_worker_resolver() -> None:
    global _WORKER_RESOLVER
    resolver, _WORKER_RESOLVER = _WORKER_RESOLVER, None
    if resolver is not None:
        resolver.close()


def _worker_init(package: str, manifest_sha256: str, child_import_paths: tuple[str, ...]) -> None:
    global _WORKER_RESOLVER, _WORKER_FINALIZER, _WORKER_PACKAGE, _WORKER_MANIFEST_SHA256
    for import_path in reversed(child_import_paths):
        if import_path not in sys.path:
            sys.path.insert(0, import_path)

    # Set CPU math limits before importing Torch or NumPy in a spawned child.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import torch

    if torch.get_num_threads() != 1:
        raise TBEProcessError(f"worker Torch intra-op threads must be 1, got {torch.get_num_threads()}")

    package_path = Path(package)
    actual_manifest_sha256 = _read_manifest_sha256(package_path)
    if actual_manifest_sha256 != manifest_sha256:
        raise TBEProcessError(
            f"package manifest SHA-256 changed before worker load: {actual_manifest_sha256} != {manifest_sha256}"
        )
    try:
        package_api = importlib.import_module("georefine_context_runtime.bitexact_predictive_package")
        resolver_type = package_api.VerifiedTensorResolver
    except (ImportError, AttributeError) as exc:
        raise TBEProcessError("installed predictive runtime lacks VerifiedTensorResolver") from exc

    _WORKER_RESOLVER = resolver_type(package_path, cache_bytes=_RESOLVER_CACHE_BYTES)
    _WORKER_PACKAGE = str(package_path)
    _WORKER_MANIFEST_SHA256 = manifest_sha256
    _WORKER_FINALIZER = mp_util.Finalize(None, _close_worker_resolver, exitpriority=10)


def _encode_one(request: TensorRequest) -> dict:
    import numpy as np
    import torch

    resolver = _WORKER_RESOLVER
    if resolver is None:
        raise TBEProcessError("worker resolver was not initialized")
    if request.name not in resolver.tensor_names:
        raise TBEProcessError(f"tensor is absent from verified manifest: {request.name}")
    expected_shape = tuple(int(dim) for dim in request.shape)
    if len(expected_shape) != 2 or min(expected_shape) <= 0:
        raise TBEProcessError(f"invalid expected tensor shape {expected_shape!r}")
    if expected_shape[0] % 8 or expected_shape[1] % 64:
        raise TBEProcessError(f"TBE mma16 requires N%8=0 and K%64=0, got {expected_shape}")
    if not _valid_sha256(request.source_sha256):
        raise TBEProcessError(f"invalid expected source SHA-256 for {request.name}")

    try:
        source = resolver.tensor(request.name).decode(check=True)
    except BaseException as exc:
        raise TBEProcessError(f"verified source decode failed for {request.name}: {exc}") from exc
    if tuple(getattr(source, "shape", ())) != expected_shape:
        raise TBEProcessError(
            f"verified source shape mismatch for {request.name}: {tuple(getattr(source, 'shape', ()))} != {expected_shape}"
        )
    source = source.detach().contiguous()
    actual_sha256 = _source_tensor_sha256(source)
    expected_sha256 = request.source_sha256.lower()
    if actual_sha256 != expected_sha256:
        raise TBEProcessError(
            f"verified source SHA-256 mismatch for {request.name}: {actual_sha256} != {expected_sha256}"
        )

    from glc_loader.tbe_cpu_numpy import decode_tbe_numpy, encode_tbe_numpy

    container = encode_tbe_numpy(source, layout="mma16")
    roundtrip = decode_tbe_numpy(container)
    if not torch.equal(roundtrip.view(torch.int16), source.view(torch.int16)):
        raise TBEProcessError(f"NumPy TBE round-trip mismatch for {request.name}")

    # Copies detach result buffers from the worker's Torch tensors before pickle.
    fields = {
        "planes": np.array(container.planes.detach().numpy(), copy=True),
        "smb": np.array(container.smb.detach().numpy(), copy=True),
        "esc": np.array(container.esc.detach().numpy(), copy=True),
        "sbbase": np.array(container.sbbase.detach().numpy(), copy=True),
    }
    return {
        "name": request.name,
        "shape": list(expected_shape),
        "manifest_sha256": _WORKER_MANIFEST_SHA256,
        "source_sha256": actual_sha256,
        "source_bf16_bytes": int(source.numel() * 2),
        "mode": int(container.mode),
        "base": int(container.base),
        "tiles": int(container.tiles),
        "escapes": int(container.escapes),
        "superblock": int(container.superblock),
        "layout": str(container.layout),
        "tbe_v1_stored_bytes": int(container.byte_size()["total"]),
        "roundtrip_exact": True,
        "worker_pid": int(mp.current_process().pid),
        "torch_threads": int(torch.get_num_threads()),
        **fields,
    }


def iter_encode_tensors_processes(
    *,
    package: str | Path,
    manifest_sha256: str,
    tensors: Sequence[TensorRequest],
    workers: int,
    source_inflight_bytes: int,
    child_import_paths: Sequence[str | Path],
    process_start_method: str = "spawn",
) -> Iterator[dict]:
    """Encode verified BF16 tensors with isolated CPU workers.

    ``source_inflight_bytes`` bounds the sum of BF16 payload bytes assigned to
    running tasks. Results are yielded in request order with at most ``workers``
    NumPy payloads buffered. Closing the iterator shuts down workers and closes
    their resolvers. This does not alter or select any serving loader path.
    """
    if type(workers) is not int or not 1 <= workers <= _MAX_WORKERS:
        raise TBEProcessError(f"workers must be an integer in [1, {_MAX_WORKERS}]")
    if type(source_inflight_bytes) is not int or not 0 < source_inflight_bytes <= _MAX_SOURCE_INFLIGHT_BYTES:
        raise TBEProcessError("source_inflight_bytes must be between 1 byte and 8 GiB")
    if not _valid_sha256(manifest_sha256):
        raise TBEProcessError("manifest_sha256 must be a 64-character hex digest")
    if process_start_method not in mp.get_all_start_methods():
        raise TBEProcessError(f"unsupported process_start_method {process_start_method!r}")
    if process_start_method == "fork" and "torch" in sys.modules:
        torch_module = sys.modules["torch"]
        if torch_module.cuda.is_initialized():
            raise TBEProcessError("fork workers are refused after parent CUDA initialization")
        if torch_module.get_num_threads() != 1:
            raise TBEProcessError("fork workers require parent Torch intra-op threads=1")
    if not tensors:
        return

    package_path = Path(package).expanduser().resolve()
    expected_manifest_sha256 = manifest_sha256.lower()
    actual_manifest_sha256 = _read_manifest_sha256(package_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise TBEProcessError(
            f"package manifest SHA-256 mismatch: {actual_manifest_sha256} != {expected_manifest_sha256}"
        )
    import_paths = tuple(str(Path(path).expanduser().resolve()) for path in child_import_paths)
    if not import_paths or any(not Path(path).is_dir() for path in import_paths):
        raise TBEProcessError("child_import_paths must name existing directories")

    requests = tuple(tensors)
    sizes = []
    for request in requests:
        if not isinstance(request, TensorRequest):
            raise TBEProcessError("tensors must contain TensorRequest values")
        if type(request.shape) not in (tuple, list) or len(request.shape) != 2:
            raise TBEProcessError(f"{request.name}: shape must contain two dimensions")
        n, k = request.shape
        if type(n) is not int or type(k) is not int or n <= 0 or k <= 0:
            raise TBEProcessError(f"{request.name}: shape dimensions must be positive integers")
        source_bytes = n * k * 2
        if source_bytes > source_inflight_bytes:
            raise TBEProcessError(
                f"{request.name}: source tensor {source_bytes} bytes exceeds in-flight budget {source_inflight_bytes}"
            )
        if not _valid_sha256(request.source_sha256):
            raise TBEProcessError(f"{request.name}: expected source SHA-256 must be a 64-character hex digest")
        sizes.append(source_bytes)

    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context(process_start_method),
        initializer=_worker_init,
        initargs=(str(package_path), expected_manifest_sha256, import_paths),
    )
    completed: dict[int, dict] = {}
    pending = {}
    next_index = 0
    next_yield = 0
    in_flight_bytes = 0
    try:
        while next_yield < len(requests):
            while (next_index < len(requests) and len(pending) + len(completed) < workers
                   and in_flight_bytes + sizes[next_index] <= source_inflight_bytes):
                future = executor.submit(_encode_one, requests[next_index])
                pending[future] = (next_index, sizes[next_index])
                in_flight_bytes += sizes[next_index]
                next_index += 1
            if next_yield in completed:
                yield completed.pop(next_yield)
                next_yield += 1
                continue
            if not pending:
                raise TBEProcessError("scheduler could not admit a tensor within the in-flight byte budget")
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                index, size = pending.pop(future)
                in_flight_bytes -= size
                try:
                    result = future.result()
                except BaseException as exc:
                    raise TBEProcessError(f"worker failed for {requests[index].name}: {exc}") from exc
                if result.get("manifest_sha256") != expected_manifest_sha256:
                    raise TBEProcessError(f"worker returned wrong manifest identity for {requests[index].name}")
                if result.get("source_sha256") != requests[index].source_sha256.lower():
                    raise TBEProcessError(f"worker returned wrong source identity for {requests[index].name}")
                completed[index] = result
            while next_yield in completed:
                yield completed.pop(next_yield)
                next_yield += 1
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=next_yield < len(requests))


def encode_tensors_processes(
    *,
    package: str | Path,
    manifest_sha256: str,
    tensors: Sequence[TensorRequest],
    workers: int,
    source_inflight_bytes: int,
    child_import_paths: Sequence[str | Path],
    process_start_method: str = "spawn",
) -> list[dict]:
    """Collect every result from the streaming process API."""
    return list(iter_encode_tensors_processes(
        package=package,
        manifest_sha256=manifest_sha256,
        tensors=tensors,
        workers=workers,
        source_inflight_bytes=source_inflight_bytes,
        child_import_paths=child_import_paths,
        process_start_method=process_start_method,
    ))


__all__ = [
    "TBEProcessError", "TensorRequest", "encode_tensors_processes",
    "iter_encode_tensors_processes",
]
