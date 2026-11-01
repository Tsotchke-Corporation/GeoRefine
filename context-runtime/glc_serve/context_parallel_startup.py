"""Bounded in-memory parallel Context -> TBE21 startup.

This explicitly selected startup path keeps streams in memory. Independent
Context frames are assigned to spawned CPU workers; each worker transcodes a
same-layer batch serially so its verified predictive-reference memo can be
reused. Encoded V21Tensor objects return to the parent and remain memory-only.
"""
from __future__ import annotations

import multiprocessing
import math
import os
import re
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing.connection import wait as wait_for_processes

_WORKER_ARTIFACT = None
_WORKER_RECORDS = None


def _worker_init(release_dir, package_path, build_root, manifest_sha256):
    import sys

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    if release_dir not in sys.path:
        sys.path.insert(0, release_dir)
    from glc_serve.bctx_v21_transcode import ContextV21TranscodeArtifact

    global _WORKER_ARTIFACT, _WORKER_RECORDS
    _WORKER_ARTIFACT = ContextV21TranscodeArtifact(
        package_path, build_root=build_root,
        expected_manifest_sha256=manifest_sha256)
    _WORKER_RECORDS = _WORKER_ARTIFACT.records
    from glc_loader import codec_v21_native
    codec_v21_native.prewarm(build_root)


def _worker_batch(names):
    results = []
    for name in names:
        tensor = _WORKER_ARTIFACT.read(name)
        row = _WORKER_RECORDS[name]
        results.append((name, tensor, _WORKER_ARTIFACT.transcode_receipts[-1],
                        row["source_sha256"], os.getpid(),
                        tuple(os.environ[k] for k in (
                            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                            "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))))
    return results


def _worker_streaming_batch(names):
    try:
        return _worker_batch(names)
    finally:
        # Idle workers retain only their separately budgeted predictive memo.
        # Clear tensor/source caches before the parent retires task workspace.
        for name in names:
            _WORKER_ARTIFACT.release(name)


def _layer_key(name):
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    return (0, int(match.group(1))) if match else (1, name)


def _stream_reserve(record, codec):
    """Codec geometry upper bound for stored arrays, plus Python object slack."""
    shape = tuple(int(n) for n in record["shape"])
    numel = 1
    for n in shape:
        numel *= n
    if (record.get("dtype") != "BF16" or numel < codec.MIN_CODED_NUMEL):
        arrays = int(record["source_bytes"])
    else:
        if len(shape) >= 2 and math.prod(shape[1:]) >= codec.CH:
            nrows, width = shape[0], math.prod(shape[1:])
        else:
            nrows, width = (numel + codec.CH - 1) // codec.CH, codec.CH
        padded = ((width + codec.CH - 1) // codec.CH) * codec.CH
        chunks = padded // codec.CH
        # At most 17 overflow bits per padded BF16 value, rounded per row to
        # uint32 words. These bounds mirror the public V21 geometry/array APIs.
        arrays = (nrows * padded + nrows * chunks * 32
                  + nrows * ((padded * 17 + 31) // 32) * 4
                  + nrows * 4 + (nrows + 7) // 8 + codec.NSYM * codec.MAX_CB)
    return arrays + (64 << 10)


def _work_reserve(record, codec, candidate=None):
    """Temporary source/frame decode, encode, verify, pickle and IPC allowance."""
    source = int(record["source_bytes"])
    frame = int(candidate["frame_bytes"] if candidate is not None
                else record["frame_bytes"])
    coeff = candidate.get("coefficients", {}) if candidate else {}
    coeff_bytes = int(coeff.get("bytes", 0)) + 2 * int(
        coeff.get("decoded_bytes", coeff.get("bytes", 0)))
    return (source * 4 + frame * 2 + coeff_bytes
            + _stream_reserve(record, codec) * 2)


def _batches(source, names, max_task_bytes):
    """Group same-layer matrices in first-seen source order under task RAM bound."""
    runs = []
    for name in names:
        key = _layer_key(name)
        if not runs or runs[-1][0] != key:
            runs.append((key, []))
        runs[-1][1].append(name)
    batches = []
    for _, run_names in runs:
        current, size = [], 0
        for name in run_names:
            cost = _work_reserve(source.records[name], source.codec,
                                 source._selected.get(name))
            if cost > max_task_bytes:
                raise ValueError(f"{name}: estimated worker memory {cost} exceeds {max_task_bytes}")
            if current and size + cost > max_task_bytes:
                batches.append((current, size))
                current, size = [], 0
            current.append(name)
            size += cost
        if current:
            batches.append((current, size))
    return batches


def _embedding_name(name):
    lower = name.lower()
    return any(token in lower for token in (
        "embed_tokens", "word_embeddings", ".wte.", ".embedding.",
    ))


def _mandatory_dense_name(name):
    lower = name.lower()
    return _embedding_name(name) or any(token in lower for token in (
        "vision", "visual", "image_encoder", "image_tower", "mm_projector",
    ))


def _default_coded(name, record):
    shape = record.get("shape")
    return (record.get("dtype") == "BF16"
            and isinstance(shape, list) and len(shape) == 2
            and not _mandatory_dense_name(name))


def _available_memory_bytes():
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:  # optional and advisory only
        pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_AVPHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 8 << 30


def _abort_executor(executor, *, timeout=2.0):
    """Cancel and join only this executor's processes, with bounded escalation."""
    processes = list((getattr(executor, "_processes", None) or {}).values())
    pids = {p.pid for p in processes if p.pid is not None}
    # shutdown(wait=False) clears this reference. Keep the manager as the sole
    # reaper: concurrent Process.join/is_alive calls race in Popen.poll when
    # one thread consumes waitpid's status before publishing returncode.
    manager = getattr(executor, "_executor_manager_thread", None)
    executor.shutdown(wait=False, cancel_futures=True)
    pending = {p.sentinel: p for p in processes}
    for action in ("terminate", "kill"):
        # Sentinels observe exit without consuming waitpid's status. Do not
        # signal a worker whose sentinel already records its exit.
        if pending:
            for sentinel in wait_for_processes(list(pending), timeout=0):
                pending.pop(sentinel)
        for proc in pending.values():
            getattr(proc, action)()
        deadline = time.monotonic() + timeout
        while pending:
            ready = wait_for_processes(
                list(pending), timeout=max(0.0, deadline - time.monotonic()))
            if not ready:
                break
            for sentinel in ready:
                pending.pop(sentinel)
    if pending:
        alive = [p.pid for p in pending.values()]
        raise RuntimeError(f"owned worker processes survived bounded cancellation: {alive}")
    if manager is not None:
        manager.join(max(0.0, deadline - time.monotonic()))
        if manager.is_alive():
            raise RuntimeError("owned executor manager survived bounded cancellation")
    # The manager has finished reaping; joining now cannot race with it.
    for proc in processes:
        proc.join(max(0.0, deadline - time.monotonic()))
    alive = [p.pid for p in processes if p.is_alive()]
    if alive:
        raise RuntimeError(f"owned worker processes survived bounded cancellation: {alive}")
    return pids


def _validate_result_names(expected_names, results):
    from glc_serve.bctx_v21_transcode import ContextTranscodeError

    try:
        actual = [result[0] for result in results]
    except (IndexError, TypeError) as exc:
        raise ContextTranscodeError("worker returned a malformed batch result") from exc
    if actual != list(expected_names):
        raise ContextTranscodeError(
            f"worker batch names differ: expected {list(expected_names)!r}, got {actual!r}")


class ParallelContextV21Artifact:
    """Artifact-reader-compatible in-memory startup wrapper.

    Construction eagerly prepares eligible coded tensors unless streaming=True.
    The opt-in streaming path returns with bounded CPU work pending. Dense embeddings,
    non-matrices, and vision tensors stay on the source path. Worker count is
    capped at twelve and by both the configured RAM budget and active work.
    """

    def __init__(self, package_path, *, build_root, max_workers=12,
                 max_inflight_bytes=None, expected_manifest_sha256=None,
                 coded_names=None, cancel_timeout=2.0, streaming=False):
        from glc_serve.bctx_v21_transcode import ContextV21TranscodeArtifact, _startup_detail

        if type(streaming) is not bool:
            raise ValueError("streaming must be a bool")
        self.streaming = streaming
        if type(max_workers) is not int or not 1 <= max_workers <= 12:
            raise ValueError("max_workers must be an integer from 1 through 12")
        if (max_inflight_bytes is not None
                and (type(max_inflight_bytes) is not int or max_inflight_bytes <= 0)):
            raise ValueError("max_inflight_bytes must be a positive integer or None")
        if (type(cancel_timeout) not in (int, float)
                or not math.isfinite(cancel_timeout)
                or not 1 <= cancel_timeout <= 10):
            raise ValueError("cancel_timeout must be a finite number from one through ten")

        self._source = ContextV21TranscodeArtifact(
            package_path, build_root=build_root,
            expected_manifest_sha256=expected_manifest_sha256)
        self.path = self._source.path
        self.root = self._source.root
        self.manifest = self._source.manifest
        self.manifest_sha256 = self._source.manifest_sha256
        self.records = self._source.records
        self.codec = self._source.codec
        self.format = self._source.format
        self.config_path = self._source.config_path
        self._names = self._source.names()
        if coded_names is None:
            coded_set = {n for n in self._names if _default_coded(n, self.records[n])}
        else:
            coded_set = set(coded_names)
            unknown = coded_set.difference(self.records)
            if unknown:
                raise ValueError(f"unknown coded tensor names: {sorted(unknown)[:3]}")
            unsafe = [n for n in coded_set
                      if not _default_coded(n, self.records[n])]
            if unsafe:
                raise ValueError(f"coded_names includes dense/ineligible tensors: {unsafe[:3]}")
        coded = [n for n in self._names if n in coded_set]
        self._coded_names = tuple(coded)
        if max_inflight_bytes is None:
            max_inflight_bytes = int(_available_memory_bytes() * 0.75)
        self.max_inflight_bytes = int(max_inflight_bytes)
        self._cancel_timeout = cancel_timeout
        if streaming:
            # Workers are submitted before the reader returns. Let the consumer
            # reconstruct dense embeddings while those first coded batches run,
            # rather than waiting for lm_head and only then starting the parent
            # decode. Keep the coded batch order and every lifetime RAM charge.
            self._names.sort(key=lambda name: not (
                _embedding_name(name) and len(self.records[name].get("shape", ())) == 2))
            self._init_streaming(coded, max_workers, build_root)
            return
        self._stream_reserved_bytes = sum(
            _stream_reserve(self.records[n], self.codec) for n in coded)
        self._resident_bytes = 0
        self._reference_reserve_per_worker = 1_400_000_000 if self._source._selected else 0
        requested_workers = min(12, max(1, int(max_workers)), max(1, len(coded)))
        if self._stream_reserved_bytes > self.max_inflight_bytes:
            raise ValueError("estimated retained streams exceed max_inflight_bytes")
        available = self.max_inflight_bytes - self._stream_reserved_bytes
        workers = requested_workers
        batches = None
        while workers:
            worker_reserve = workers * self._reference_reserve_per_worker
            max_task_bytes = available - worker_reserve
            if max_task_bytes > 0:
                try:
                    batches = _batches(self._source, coded, max_task_bytes)
                    break
                except ValueError:
                    pass
            workers -= 1
        if batches is None:
            raise ValueError("memory budget cannot fit retained streams, reference reserve, and largest task")
        self._scheduled_names = tuple(n for batch, _ in batches for n in batch)
        self.max_workers = workers
        self._streams = {}
        self._cached_name = self._cached_tensor = None
        self.transcode_receipts = []
        self._worker_pids = set()
        self._worker_env = set()
        self._max_accounted_memory_bytes = self._stream_reserved_bytes + workers * self._reference_reserve_per_worker
        self._executor = None
        if not batches:
            return

        # Compile/load the package-verified native libraries in the parent
        # before spawning. This initializes no frame data and avoids duplicate
        # baseline/predictive library builds in workers.
        _startup_detail(step='parallel_prewarm',done=0,total=len(coded),workers=workers)
        self._source._decoder._lib()
        from glc_loader import codec_v21_native
        codec_v21_native.prewarm(build_root)
        if self._source._pair is not None:
            self._source._pair._lib()
        # Derive the runtime path from the imported implementation so this
        # implementation remains relocatable with the release worktree.
        import glc_serve.bctx_v21_transcode as transcode
        from pathlib import Path
        release_dir = str(Path(transcode.__file__).resolve().parents[1])
        mp = multiprocessing.get_context("spawn")
        executor = ProcessPoolExecutor(
            max_workers=workers, mp_context=mp, initializer=_worker_init,
            initargs=(release_dir, self.path, str(build_root), self.manifest_sha256))
        self._executor = executor
        pending = {}
        waiting = deque(batches)
        inflight = 0
        reserved_by_name = {n: _stream_reserve(self.records[n], self.codec)
                            for n in coded}
        try:
            while pending or waiting:
                while waiting:
                    batch, cost = waiting[0]
                    current = (self._resident_bytes + self._stream_reserved_bytes
                               + workers * self._reference_reserve_per_worker + inflight)
                    if pending and current + cost > self.max_inflight_bytes:
                        break
                    if current + cost > self.max_inflight_bytes:
                        raise ValueError("in-flight work exceeds max_inflight_bytes")
                    waiting.popleft()
                    future = executor.submit(_worker_batch, batch)
                    pending[future] = (batch, cost)
                    inflight += cost
                    self._max_accounted_memory_bytes = max(
                        self._max_accounted_memory_bytes, current + cost)
                if not pending:
                    raise ValueError("in-flight RAM budget cannot schedule a batch")
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    batch, cost = pending.pop(future)
                    results = future.result()
                    _validate_result_names(batch, results)
                    for (name, tensor, receipt, source_sha, worker_pid,
                         worker_env) in results:
                        row = self.records[name]
                        expected_frame_sha = (self._source._selected[name]["frame_sha256"]
                                              if name in self._source._selected
                                              else row["frame_sha256"])
                        if (tensor.name != name or tensor.sha256 != row["source_sha256"]
                                or tensor.dtype != row["dtype"]
                                or tuple(tensor.geom.shape) != tuple(row["shape"])
                                or source_sha != row["source_sha256"]
                                or receipt.get("source_sha256") != row["source_sha256"]
                                or receipt.get("v21_stream_bytes") != tensor.total_bytes()
                                or receipt.get("frame_sha256") != expected_frame_sha):
                            from glc_serve.bctx_v21_transcode import ContextTranscodeError
                            raise ContextTranscodeError(f"{name}: worker transfer source identity mismatch")
                        self.codec.verify_crc(tensor)
                        actual = tensor.total_bytes()
                        reserve = reserved_by_name.pop(name)
                        if actual > reserve:
                            raise ValueError(f"{name}: stream exceeded reserved in-memory size")
                        self._stream_reserved_bytes -= reserve
                        self._streams[name] = tensor
                        self._resident_bytes += actual
                        self.transcode_receipts.append(receipt)
                        self._worker_pids.add(worker_pid)
                        self._worker_env.add(worker_env)
                        _startup_detail(step='parallel_tensor_ready',done=len(self._streams),
                            total=len(coded),tensor=name,workers=workers,worker_pid=worker_pid)
                    inflight -= cost
                    self._max_accounted_memory_bytes = max(
                        self._max_accounted_memory_bytes,
                        self._resident_bytes + self._stream_reserved_bytes
                        + workers * self._reference_reserve_per_worker + inflight)
        except BaseException:
            for future in pending:
                future.cancel()
            self._worker_pids.update(_abort_executor(executor, timeout=self._cancel_timeout))
            self._executor = None
            self._streams.clear()
            self._resident_bytes = 0
            self._source.release()
            raise
        else:
            executor.shutdown(wait=True)
            self._executor = None

    def _init_streaming(self, coded, max_workers, build_root):
        """Reserve active work, not worst-case streams for the whole model.

        A consumer may retain its host arrays (including CPU upload21 views).
        Admitted stream bytes stay charged after release(name). Parent dense
        outputs are similarly charged for the lifetime of this reader.
        """
        from glc_serve.bctx_v21_transcode import _startup_detail
        from glc_loader import codec_v21_native
        from pathlib import Path
        import glc_serve.bctx_v21_transcode as transcode

        self._streams = {}
        self._cached_name = self._cached_tensor = None
        self.transcode_receipts = []
        self._worker_pids, self._worker_env = set(), set()
        self._resident_bytes = self._stream_reserved_bytes = 0
        self._retained_bytes = self._inflight_bytes = 0
        self._executor = None
        self._pending = {}
        self._closed = False
        self._reference_reserve_per_worker = 1_400_000_000 if self._source._selected else 0
        # The parent can decode a dense tensor while every worker is busy.
        # Include its predictive memo and the largest decode/encode/verification
        # workspace, plus copies used to materialize a dense host tensor.
        self._parent_reserve_bytes = self._reference_reserve_per_worker + max(
            (_work_reserve(row, self.codec, self._source._selected.get(name))
             + _stream_reserve(row, self.codec) + 2 * int(row["source_bytes"])
             for name, row in self.records.items()), default=0)
        largest = max((_work_reserve(
            self.records[name], self.codec, self._source._selected.get(name))
            for name in coded), default=0)
        workers = min(max_workers, max(1, len(coded)))
        while workers > 1 and (self._parent_reserve_bytes + largest
                + workers * self._reference_reserve_per_worker > self.max_inflight_bytes):
            workers -= 1
        self.max_workers = workers
        self._fixed_reserve_bytes = (self._parent_reserve_bytes
                                    + workers * self._reference_reserve_per_worker)
        if self._fixed_reserve_bytes + largest > self.max_inflight_bytes:
            raise ValueError("memory budget cannot fit parent workspace, references, and largest task")
        # Keep same-layer tensors together so predictive references decoded by
        # one tensor remain in that worker's memo for the rest of the layer.
        # As in eager startup, split only when the task RAM bound requires it.
        self._waiting = deque(_batches(
            self._source, coded, self.max_inflight_bytes - self._fixed_reserve_bytes))
        self._waiting_by_name = {name: item for item in self._waiting for name in item[0]}
        self._pending_by_name = {}
        self._scheduled_names = tuple(name for batch, _ in self._waiting for name in batch)
        self._max_accounted_memory_bytes = self._fixed_reserve_bytes
        if not coded:
            return
        _startup_detail(step='parallel_prewarm', done=0, total=len(coded), workers=workers)
        self._source._decoder._lib()
        codec_v21_native.prewarm(build_root)
        if self._source._pair is not None:
            self._source._pair._lib()
        self._executor = ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_init,
            initargs=(str(Path(transcode.__file__).resolve().parents[1]),
                      self.path, str(build_root), self.manifest_sha256))
        try:
            self._schedule_streaming()
        except BaseException:
            self.close()
            raise

    def _accounted_bytes(self):
        return self._fixed_reserve_bytes + self._retained_bytes + self._inflight_bytes

    def _schedule_streaming(self):
        while self._waiting and len(self._pending) < self.max_workers:
            batch, cost = self._waiting[0]
            if self._accounted_bytes() + cost > self.max_inflight_bytes:
                break
            self._waiting.popleft()
            future = self._executor.submit(_worker_streaming_batch, batch)
            self._pending[future] = (batch, cost)
            for name in batch:
                self._waiting_by_name.pop(name)
                self._pending_by_name[name] = future
            self._inflight_bytes += cost
            self._max_accounted_memory_bytes = max(
                self._max_accounted_memory_bytes, self._accounted_bytes())

    def _drain_streaming(self):
        from glc_serve.bctx_v21_transcode import ContextTranscodeError, _startup_detail

        if not self._pending:
            raise ContextTranscodeError("retained host streams and parent workspace prevent startup progress")
        done, _ = wait(self._pending, return_when=FIRST_COMPLETED)
        for future in done:
            batch, cost = self._pending.pop(future)
            for name in batch:
                self._pending_by_name.pop(name)
            results = future.result()
            _validate_result_names(batch, results)
            for result in results:
                if not isinstance(result, (tuple, list)) or len(result) != 6:
                    raise ContextTranscodeError("worker returned a malformed tensor result")
                name, tensor, receipt, source_sha, worker_pid, worker_env = result
                row = self.records[name]
                expected_frame_sha = self._source._selected.get(name, row)["frame_sha256"]
                if (tensor.name != name or tensor.sha256 != row["source_sha256"]
                        or tensor.dtype != row["dtype"]
                        or tuple(tensor.geom.shape) != tuple(row["shape"])
                        or source_sha != row["source_sha256"]
                        or not isinstance(receipt, dict)
                        or receipt.get("name") != name
                        or receipt.get("source_sha256") != row["source_sha256"]
                        or receipt.get("v21_stream_bytes") != tensor.total_bytes()
                        or receipt.get("frame_sha256") != expected_frame_sha
                        or tuple(worker_env) != ("", "1", "1", "1")
                        or set(tensor.crc) != set(tensor.streams())):
                    raise ContextTranscodeError(f"{name}: worker transfer source identity mismatch")
                self.codec.verify_crc(tensor)
                actual = tensor.total_bytes()
                if actual > _stream_reserve(row, self.codec):
                    raise ValueError(f"{name}: stream exceeded reserved in-memory size")
                # Keep object/array-header slack even after the loader releases
                # this reference: a consumer may own the same host arrays.
                retained = actual + (64 << 10)
                self._retained_bytes += retained
                self._resident_bytes += actual
                self._streams[name] = tensor
                self.transcode_receipts.append(receipt)
                self._worker_pids.add(worker_pid)
                self._worker_env.add(tuple(worker_env))
                _startup_detail(step='parallel_tensor_ready', done=len(self.transcode_receipts),
                    total=len(self._coded_names), tensor=name, workers=self.max_workers,
                    worker_pid=worker_pid)
            self._inflight_bytes -= cost
        self._finish_streaming_if_done()

    def _finish_streaming_if_done(self):
        if not self._waiting and not self._pending and self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
            # Parent reference memo remains reserved after workers exit.
            self._fixed_reserve_bytes = self._parent_reserve_bytes

    def _admit_parent_result(self, size):
        # Parent operations use their permanently reserved temporary workspace;
        # resulting host storage is a separate lifetime charge.
        while self._accounted_bytes() + size > self.max_inflight_bytes and self._pending:
            self._drain_streaming()
        if self._accounted_bytes() + size > self.max_inflight_bytes:
            from glc_serve.bctx_v21_transcode import ContextTranscodeError
            raise ContextTranscodeError("parent materialization exceeds max_inflight_bytes")
        self._retained_bytes += size
        self._max_accounted_memory_bytes = max(
            self._max_accounted_memory_bytes, self._accounted_bytes())

    def _read_streaming(self, name, check_crc):
        if self._closed:
            raise RuntimeError("parallel startup artifact is closed")
        if self._cached_name == name and self._cached_tensor is not None:
            return self._cached_tensor
        try:
            # A loader can request names in a different order from the package.
            # Prioritize its next tensor instead of queuing the whole model.
            item = self._waiting_by_name.get(name)
            if item is not None:
                self._waiting.remove(item)
                self._waiting.appendleft(item)
            while name not in self._streams and (
                    name in self._waiting_by_name or name in self._pending_by_name):
                self._schedule_streaming()
                self._drain_streaming()
            tensor = self._streams.get(name)
            if tensor is None:
                self._admit_parent_result(_stream_reserve(self.records[name], self.codec))
                tensor = self._source.read(name, check_crc=check_crc)
                self._verify_transfer(name, tensor)
            self._cached_name, self._cached_tensor = name, tensor
            self._schedule_streaming()
            self._finish_streaming_if_done()
            return tensor
        except BaseException:
            self.close()
            raise

    def __getattr__(self, name):
        return getattr(self._source, name)

    def names(self):
        return list(self._names)

    def parallel_startup_stats(self):
        """Report scheduler estimates separately from observed process memory."""
        return {"workers": self.max_workers,
                **({"streaming": True,
                    "parent_workspace_bytes_estimate": self._parent_reserve_bytes,
                    "retained_host_bytes_estimate": self._retained_bytes}
                   if self.streaming else {}),
                "ram_budget_bytes": self.max_inflight_bytes,
                "max_accounted_memory_bytes_estimate": self._max_accounted_memory_bytes,
                "prepared_tensors": len(self.transcode_receipts),
                "worker_pids": sorted(self._worker_pids),
                "worker_gpu_visible": any(env[0] for env in self._worker_env)}

    def stored_record(self, name):
        return self._source.stored_record(name)

    def is_alias(self, name):
        return self._source.is_alias(name)

    def has_config(self):
        return self._source.has_config()

    def release(self, name=None):
        if name is None or self._cached_name == name:
            self._cached_name = self._cached_tensor = None
        if name is None:
            self._streams.clear()
            self._resident_bytes = 0
        else:
            tensor = self._streams.pop(name, None)
            if tensor is not None:
                self._resident_bytes -= tensor.total_bytes()
        self._source.release(name)

    def read(self, name, *, check_crc=True):
        if self.streaming:
            return self._read_streaming(name, check_crc)
        if self._cached_name == name and self._cached_tensor is not None:
            return self._cached_tensor
        tensor = self._streams.get(name)
        if tensor is None:
            # Explicit release frees the retained stream. A later read takes
            # the exact original decode/encode/verify path synchronously.
            row = self.records[name]
            estimate = (_work_reserve(row, self.codec,
                                      self._source._selected.get(name))
                        + _stream_reserve(row, self.codec))
            if self._resident_bytes + estimate > self.max_inflight_bytes:
                from glc_serve.bctx_v21_transcode import ContextTranscodeError
                raise ContextTranscodeError(
                    f"{name}: exact reread exceeds max_inflight_bytes")
            tensor = self._source.read(name, check_crc=check_crc)
            self._verify_transfer(name, tensor)
            self._cached_name, self._cached_tensor = name, tensor
            return tensor
        self._cached_name, self._cached_tensor = name, tensor
        return tensor

    def _verify_transfer(self, name, tensor):
        row = self.records[name]
        if tensor.name != name or tensor.sha256 != row["source_sha256"]:
            from glc_serve.bctx_v21_transcode import ContextTranscodeError
            raise ContextTranscodeError(f"{name}: source stream identity mismatch")
        self.codec.verify_crc(tensor)

    def decode_host(self, name):
        if not self.streaming:
            return self._source.decode_host(name)
        if self._closed:
            raise RuntimeError("parallel startup artifact is closed")
        try:
            self._admit_parent_result(int(self.records[name]["source_bytes"]) + (64 << 10))
            return self._source.decode_host(name)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.streaming:
            self._closed = True
        executor, self._executor = self._executor, None
        if executor is not None:
            if self.streaming:
                for future in self._pending:
                    future.cancel()
                self._worker_pids.update(_abort_executor(executor, timeout=self._cancel_timeout))
                self._pending.clear()
                self._waiting.clear()
                self._pending_by_name.clear()
                self._waiting_by_name.clear()
                self._inflight_bytes = 0
            else:
                executor.shutdown(wait=True, cancel_futures=True)
        self.release()
        self._streams.clear()


__all__ = ["ParallelContextV21Artifact"]
