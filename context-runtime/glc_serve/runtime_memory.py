"""Cross-platform, process-attributed memory receipts for serving runs.

CPU sampling is stdlib-only. Optional MLX and nvidia-smi probes run only when
their backend is explicitly selected. All byte metrics describe their source;
unavailable measurements are represented by ``None`` and an error receipt.
"""
from __future__ import annotations

import os
import importlib
import _thread
from contextlib import contextmanager
import math
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Optional

SCHEMA_VERSION = 1
MAX_HISTORY = 256
WINDOWS_MEMORY_SOURCE = "GetProcessMemoryInfo.WorkingSetSize/PeakWorkingSetSize"

def register_cuda_tensor_for_memory_attribution(tensor: Any) -> tuple[int, int]:
    """Return an index/PID handle evidenced by an existing CUDA tensor.

    This function never imports Torch or initializes CUDA and retains no tensor
    or storage reference. The returned handle can be stored by an observer.
    """
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    is_tensor = getattr(torch, "is_tensor", None) if torch is not None else None
    if not callable(is_tensor) or is_tensor(tensor) is not True:
        raise ValueError("registration requires an actual Torch tensor")
    initialized = getattr(cuda, "is_initialized", None)
    if not callable(initialized) or initialized() is not True:
        raise RuntimeError("CUDA attribution registration requires initialized Torch CUDA")
    try:
        is_cuda = tensor.is_cuda
        device = tensor.device
        device_type = device.type
        index = device.index
    except Exception as exc:
        raise ValueError("registration requires an existing tensor with a concrete device") from exc
    if is_cuda is not True or device_type != "cuda":
        raise ValueError("registration requires an existing CUDA tensor")
    if type(index) is not int or index < 0:
        raise ValueError("registration requires an indexed CUDA tensor")
    return index, os.getpid()


def _cuda_allocator_metrics(registrations: dict[int, int]) -> tuple[dict[str, Optional[int]], dict[str, str],
                                                                  list[dict[str, str]], dict[str, Any]]:
    """Read Torch allocator counters only for explicitly registered indices."""
    registered = sorted(registrations.items())
    if not registered:
        return {}, {}, [], {}

    metrics: dict[str, Optional[int]] = {}
    sources: dict[str, str] = {}
    errors: list[dict[str, str]] = []
    details: dict[str, Any] = {
        "cuda_allocator_scope": "registered_devices_only",
        "cuda_allocator_devices": [],
        "cuda_allocator_peak_caveat":
            "Per-device peaks are cumulative since initialization and are not simultaneous; do not sum or subtract independent peaks.",
    }
    source = "torch.cuda.memory_stats(explicit registered device index)"
    current_pid = os.getpid()
    local_registrations = [(index, creator_pid) for index, creator_pid in registered
                           if creator_pid == current_pid]
    # A forked observer must fail the inherited evidence before touching any
    # CUDA state or allocator counters.
    for index, creator_pid in registered:
        if creator_pid != current_pid:
            device_detail = {"device_index": index, "creator_pid": creator_pid,
                             "status": "creator_pid_mismatch"}
            details["cuda_allocator_devices"].append(device_detail)
            errors.append({"source": source, "error": f"registered device {index} creator PID {creator_pid} differs from current PID {current_pid}"})
    if not local_registrations:
        return metrics, sources, errors, details
    # A missing import is not a reason to import Torch in a sampler thread.
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    initialized_fn = getattr(cuda, "is_initialized", None)
    if not callable(initialized_fn):
        for index, creator_pid in local_registrations:
            errors.append({"source": source, "error": f"CUDA allocator unavailable for registered device {index}: Torch CUDA state is unavailable"})
        return metrics, sources, errors, details
    try:
        initialized = initialized_fn()
    except Exception as exc:
        initialized = None
        state_error = f"{type(exc).__name__}: {exc}"
    else:
        state_error = ""
    if initialized is not True:
        reason = state_error or ("Torch CUDA is not initialized" if initialized is False else "Torch CUDA initialization state is unknown")
        for index, creator_pid in local_registrations:
            errors.append({"source": source, "error": f"CUDA allocator unavailable for registered device {index}: {reason}"})
        return metrics, sources, errors, details

    try:
        with _cuda_probe_guard(details) as guarded:
            if not guarded:
                reason = details.get("cuda_context_guard_status", "unavailable")
                for index, _creator_pid in local_registrations:
                    errors.append({"source": source, "error": f"CUDA allocator unavailable for registered device {index}: context guard {reason}"})
                return metrics, sources, errors, details
            # Recheck while holding the same initialization guard as the query.
            if initialized_fn() is not True:
                for index, _creator_pid in local_registrations:
                    errors.append({"source": source, "error": f"CUDA allocator unavailable for registered device {index}: initialization state changed or is unknown"})
                return metrics, sources, errors, details
            memory_stats = getattr(cuda, "memory_stats", None)
            if not callable(memory_stats):
                for index, _creator_pid in local_registrations:
                    errors.append({"source": source, "error": f"CUDA allocator unavailable for registered device {index}: memory_stats is unavailable"})
                return metrics, sources, errors, details
            for index, creator_pid in local_registrations:
                device_detail: dict[str, Any] = {"device_index": index, "creator_pid": creator_pid}
                details["cuda_allocator_devices"].append(device_detail)
                if creator_pid != os.getpid():
                    device_detail["status"] = "creator_pid_mismatch"
                    errors.append({"source": source, "error": f"registered device {index} creator PID {creator_pid} differs from current PID {os.getpid()}"})
                    continue
                try:
                    stats = memory_stats(device=index)
                    if not isinstance(stats, dict):
                        raise ValueError("memory_stats did not return a mapping")
                except Exception as exc:
                    device_detail["status"] = "query_failed"
                    errors.append({"source": source, "error": f"device {index}: {type(exc).__name__}: {exc}"})
                    continue
                field_map = {
                    "allocated_bytes.all.current": "allocated_current_bytes",
                    "reserved_bytes.all.current": "reserved_current_bytes",
                    "allocated_bytes.all.peak": "allocated_peak_bytes",
                    "reserved_bytes.all.peak": "reserved_peak_bytes",
                    "requested_bytes.all.current": "requested_current_bytes",
                    "active_bytes.all.current": "active_current_bytes",
                    "inactive_split_bytes.all.current": "inactive_split_current_bytes",
                }
                count_map = {
                    "allocation.all.current": "allocation",
                    "active.all.current": "active",
                    "inactive_split.all.current": "inactive_split",
                    "segment.all.current": "segment",
                }
                device_detail["status"] = "available"
                for stats_key, label in field_map.items():
                    metric_key = f"cuda_allocator_device_{index}_{label}"
                    sources[metric_key] = source
                    value = stats.get(stats_key)
                    if type(value) is int and value >= 0:
                        metrics[metric_key] = value
                    else:
                        metrics[metric_key] = None
                        errors.append({"source": source, "error": f"device {index}: missing or malformed counter {stats_key!r}"})
                current_counts = {}
                for stats_key, label in count_map.items():
                    value = stats.get(stats_key)
                    if type(value) is int and value >= 0:
                        current_counts[label] = {"value": value, "unit": "count", "source": source}
                    else:
                        current_counts[label] = {"value": None, "unit": "count", "source": source}
                        errors.append({"source": source, "error": f"device {index}: missing or malformed counter {stats_key!r}"})
                device_detail["current_counts"] = current_counts
    except Exception as exc:
        errors.append({"source": source, "error": f"{type(exc).__name__}: {exc}"})
    return metrics, sources, errors, details


def _cuda_runtime() -> Any:
    """Import the explicit CUDA runtime without initializing its CUDA context.

    The explicit CUDA probe may precede the model loader's first Torch import.
    Importing Torch establishes a queryable runtime state; absence from
    ``sys.modules`` alone is not evidence that the process has no CUDA context.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        try:
            torch = importlib.import_module("torch")
        except Exception:
            return None
    return getattr(torch, "cuda", None) if torch is not None else None


def _cuda_context_initialized() -> Optional[bool]:
    cuda = _cuda_runtime()
    query = getattr(cuda, "is_initialized", None)
    if not callable(query):
        return None
    try:
        value = query()
    except Exception:
        return None
    return value if type(value) is bool else None


@contextmanager
def _cuda_probe_guard(details: dict[str, Any]):
    # Torch 2.8 _lazy_init holds this primitive lock through _cuda_init and
    # publication of _initialized. Import first: Torch import itself uses it.
    cuda = _cuda_runtime()
    lock = getattr(cuda, "_initialization_lock", None)
    details["cuda_context_guard"] = None
    if not isinstance(lock, _thread.LockType):
        details["cuda_context_guard_status"] = "unavailable"
        yield False
        return
    if getattr(getattr(cuda, "_tls", None), "is_initializing", False):
        # A callback within _lazy_init already owns this non-reentrant lock.
        details["cuda_context_guard_status"] = "reentrant_initialization"
        yield False
        return
    if not lock.acquire(timeout=3):
        details["cuda_context_guard_status"] = "acquire_timeout"
        raise RuntimeError("CUDA initialization guard acquisition timed out")
    try:
        details["cuda_context_guard"] = "torch.cuda._initialization_lock"
        details["cuda_context_guard_status"] = "held"
        yield True
    finally:
        lock.release()


def _windows_process_memory(pid: int, *, api_loader: Optional[Callable] = None
                           ) -> tuple[Optional[int], Optional[int], Optional[str]]:
    """Read current/peak working set with pointer-sized WinAPI handles."""
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    load = api_loader or (lambda name: ctypes.WinDLL(name, use_last_error=True))
    kernel32, psapi = load("kernel32"), load("psapi")
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x0400 | 0x1000, False, int(pid))
    if not handle:
        return None, None, f"OpenProcess failed: {ctypes.get_last_error()}"
    try:
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None, None, f"GetProcessMemoryInfo failed: {ctypes.get_last_error()}"
        return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize), None
    finally:
        kernel32.CloseHandle(handle)


def _current_rss_bytes(pid: int) -> tuple[Optional[int], str, Optional[str]]:
    if sys.platform.startswith("linux"):
        try:
            with open(f"/proc/{pid}/statm", "r", encoding="ascii") as f:
                resident_pages = int(f.read().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE"), f"/proc/{pid}/statm", None
        except Exception as exc:
            return None, f"/proc/{pid}/statm", f"{type(exc).__name__}: {exc}"
    if sys.platform == "darwin":
        source = "mach task_info(MACH_TASK_BASIC_INFO).resident_size"
        try:
            import ctypes
            class TimeValue(ctypes.Structure):
                _fields_ = [("seconds", ctypes.c_int), ("microseconds", ctypes.c_int)]
            class TaskBasicInfo(ctypes.Structure):
                _fields_ = [("virtual_size", ctypes.c_uint64), ("resident_size", ctypes.c_uint64),
                            ("resident_size_max", ctypes.c_uint64), ("user_time", TimeValue),
                            ("system_time", TimeValue), ("policy", ctypes.c_int),
                            ("suspend_count", ctypes.c_int)]
            lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            task_self = ctypes.c_uint.in_dll(lib, "mach_task_self_").value
            info = TaskBasicInfo()
            count = ctypes.c_uint(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_int))
            result = lib.task_info(task_self, 20, ctypes.byref(info), ctypes.byref(count))
            if result != 0:
                raise OSError(result, "mach task_info failed")
            return int(info.resident_size), source, None
        except Exception as exc:
            return None, source, f"{type(exc).__name__}: {exc}"
    if os.name == "nt":
        source = "GetProcessMemoryInfo.WorkingSetSize"
        try:
            current, _peak, error = _windows_process_memory(pid)
            return current, source, error
        except Exception as exc:
            return None, source, f"{type(exc).__name__}: {exc}"
    return None, "unsupported OS process RSS probe", f"unsupported platform: {sys.platform}"


def _os_peak_rss_bytes() -> tuple[Optional[int], str, Optional[str]]:
    if os.name == "nt":
        try:
            _current, peak, error = _windows_process_memory(os.getpid())
            return peak, "GetProcessMemoryInfo.PeakWorkingSetSize", error
        except Exception as exc:
            return None, "GetProcessMemoryInfo.PeakWorkingSetSize", f"{type(exc).__name__}: {exc}"
    try:
        import resource
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Darwin reports bytes; Linux and the BSDs commonly report KiB.
        return value if sys.platform == "darwin" else value * 1024, "resource.getrusage(RUSAGE_SELF).ru_maxrss", None
    except Exception as exc:
        return None, "resource.getrusage(RUSAGE_SELF).ru_maxrss", f"{type(exc).__name__}: {exc}"


def _device_metrics(backend: str, pid: int, *, cuda_allocator_devices: Optional[dict[int, int]] = None,
                    cuda_allocator_attribution: bool = False) -> tuple[dict[str, Optional[int]], dict[str, str], list[dict[str, str]], dict[str, Any]]:
    metrics: dict[str, Optional[int]] = {}
    sources: dict[str, str] = {}
    errors: list[dict[str, str]] = []
    details: dict[str, Any] = {}
    if backend == "metal":
        keys = ("mlx_current_allocated_bytes", "mlx_peak_allocated_bytes", "mlx_cache_bytes")
        for key in keys:
            metrics[key] = None
        names = ("get_active_memory", "get_peak_memory", "get_cache_memory")
        try:
            import mlx.core as mx  # Explicit backend only; never imported by the CPU path.
            legacy = getattr(mx, "metal", None)
            for key, name in zip(keys, names):
                func = getattr(mx, name, None)
                source = f"mlx.core.{name}"
                if func is None and legacy is not None and getattr(legacy, name, None) is not None:
                    func = getattr(legacy, name)
                    source = f"mlx.core.metal.{name} (legacy fallback)"
                sources[key] = source
                if func is None:
                    errors.append({"source": f"mlx.core.{name}", "error": "counter unavailable in this MLX version"})
                else:
                    try:
                        value = func()
                        if type(value) is not int or value < 0:
                            raise ValueError("counter did not return a nonnegative integer byte count")
                        metrics[key] = value
                    except Exception as exc:
                        errors.append({"source": sources[key], "error": f"{type(exc).__name__}: {exc}"})
        except Exception as exc:
            errors.append({"source": "import mlx.core", "error": f"{type(exc).__name__}: {exc}"})
        for key in keys:
            sources.setdefault(key, f"mlx.core.{names[keys.index(key)]}")
        return metrics, sources, errors, details
    if backend == "cuda":
        key = "cuda_process_device_bytes"
        sources[key] = "nvidia-smi compute-apps gpu_uuid,pid,used_memory; unique PID/device rows summed"
        metrics[key] = None
        if cuda_allocator_attribution and "torch" not in sys.modules:
            errors.append({"source": sources[key],
                           "error": "CUDA device probe skipped: Torch is not already imported; opt-in attribution never imports Torch"})
            details["cuda_context_initialized"] = None
            details["cuda_context_state_source"] = "Torch absent from sys.modules; CUDA state was not queried"
        else:
            try:
                with _cuda_probe_guard(details) as guarded:
                    initialized_before = _cuda_context_initialized()
                    details["cuda_context_state_source"] = "torch.cuda.is_initialized()"
                    details["cuda_context_initialized_before_query"] = initialized_before
                    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                         "--format=csv,noheader,nounits"], check=True,
                                        capture_output=True, text=True, timeout=3).stdout
                    initialized = _cuda_context_initialized()
                    details["cuda_context_initialized"] = initialized
                matches: list[tuple[str, int]] = []
                for line in out.splitlines():
                    if not line.strip():
                        continue
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) != 3:
                        raise ValueError(f"malformed nvidia-smi compute-app row: {line!r}")
                    if not parts[0]:
                        raise ValueError("nvidia-smi returned an empty GPU UUID")
                    if not parts[1].isdigit():
                        raise ValueError(f"invalid nvidia-smi PID value: {parts[1]!r}")
                    if not parts[2].isdigit():
                        raise ValueError(f"invalid nvidia-smi used_memory value: {parts[2]!r}")
                    if int(parts[1]) == pid:
                        matches.append((parts[0], int(parts[2])))
                if matches:
                    uuids = [uuid for uuid, _ in matches]
                    if len(uuids) != len(set(uuids)):
                        errors.append({"source": sources[key], "error": f"duplicate PID/device rows for PID {pid}"})
                    else:
                        details["cuda_devices"] = [{"gpu_uuid": uuid, "used_memory_bytes": mib * (1 << 20)}
                                                    for uuid, mib in matches]
                        metrics[key] = sum(mib for _, mib in matches) * (1 << 20)
                else:
                    if guarded and initialized_before is False and initialized is False:
                        details["phase"] = "before_cuda_context"
                    else:
                        errors.append({"source": sources[key], "error": f"no nvidia-smi row for PID {pid}"})
            except Exception as exc:
                errors.append({"source": sources[key], "error": f"{type(exc).__name__}: {exc}"})
    if backend == "cuda" and cuda_allocator_devices:
        alloc_metrics, alloc_sources, alloc_errors, alloc_details = _cuda_allocator_metrics(cuda_allocator_devices)
        metrics.update(alloc_metrics)
        sources.update(alloc_sources)
        errors.extend(alloc_errors)
        if alloc_details:
            details["cuda_allocator"] = alloc_details
    return metrics, sources, errors, details


def _sample(backend: str, timestamp: float, *, cuda_allocator_devices: Optional[dict[int, int]] = None,
            cuda_allocator_attribution: bool = False) -> dict[str, Any]:
    pid = os.getpid()
    current, current_source, current_error = _current_rss_bytes(pid)
    peak, peak_source, peak_error = _os_peak_rss_bytes()
    metrics: dict[str, Optional[int]] = {"rss_current_bytes": current, "rss_os_peak_bytes": peak}
    sources = {"rss_current_bytes": current_source, "rss_os_peak_bytes": peak_source}
    errors = []
    if current_error:
        errors.append({"source": current_source, "error": current_error})
    if peak_error:
        errors.append({"source": peak_source, "error": peak_error})
    if cuda_allocator_devices or cuda_allocator_attribution:
        dev_metrics, dev_sources, dev_errors, details = _device_metrics(
            "metal" if backend == "mlx" else backend, pid,
            cuda_allocator_devices=cuda_allocator_devices,
            cuda_allocator_attribution=cuda_allocator_attribution)
    else:
        # Keep the pre-attribution call signature intact for default-off paths.
        dev_metrics, dev_sources, dev_errors, details = _device_metrics(
            "metal" if backend == "mlx" else backend, pid)
    metrics.update(dev_metrics)
    sources.update(dev_sources)
    errors.extend(dev_errors)
    return {"schema_version": SCHEMA_VERSION, "pid": pid, "backend": backend,
            "timestamp_unix_s": float(timestamp), "sample_count": 1,
            "metrics": metrics, "units": {key: "bytes" for key in metrics},
            "sources": sources, "errors": errors, "device_details": details,
            "observed_maxima": {k: v for k, v in metrics.items()
                                if v is not None and k != "rss_os_peak_bytes"}}


def _validate_receipt(receipt: Any) -> dict[str, Any]:
    if not isinstance(receipt, dict) or not isinstance(receipt.get("metrics"), dict):
        raise ValueError("memory probe must return a receipt containing a metrics object")
    timestamp = receipt.get("timestamp_unix_s")
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
        raise ValueError("memory receipt timestamp must be finite")
    for name, value in receipt["metrics"].items():
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"memory metric {name!r} must be a nonnegative integer byte count or None")
    return receipt


def sample_current_process(backend: str = "cpu", *, probe: Optional[Callable] = None,
                           clock: Optional[Callable[[], float]] = None,
                           cuda_allocator_devices: Optional[dict[int, int]] = None,
                           cuda_allocator_attribution: bool = False) -> dict[str, Any]:
    """Return one JSON-safe memory receipt for this process.

    ``probe`` and ``clock`` are injection points for tests. A probe receives
    ``(backend, timestamp)`` and returns a receipt with the same shape.
    """
    backend = str(backend).lower()
    if backend not in {"cpu", "metal", "mlx", "cuda"}:
        raise ValueError("backend must be one of: cpu, metal, mlx, cuda")
    now = (clock or time.time)()
    if probe is not None:
        receipt = probe(backend, now)
    elif cuda_allocator_devices or cuda_allocator_attribution:
        receipt = _sample(backend, now, cuda_allocator_devices=cuda_allocator_devices,
                          cuda_allocator_attribution=cuda_allocator_attribution)
    else:
        # Do not pass new default-valued keywords to legacy monkeypatches.
        receipt = _sample(backend, now)
    return _validate_receipt(receipt)


class MemoryMonitor:
    """Periodically sample a process with bounded in-memory receipt history."""
    def __init__(self, backend: str = "cpu", interval_s: float = 0.1, *,
                 probe: Optional[Callable] = None, clock: Optional[Callable[[], float]] = None,
                 cuda_allocator_attribution: bool = False):
        try:
            interval_s = float(interval_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("interval_s must be finite and positive") from exc
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("interval_s must be finite and positive")
        self.backend = backend.lower()
        if self.backend not in {"cpu", "metal", "mlx", "cuda"}:
            raise ValueError("backend must be one of: cpu, metal, mlx, cuda")
        self.interval_s = float(interval_s)
        self.cuda_allocator_attribution = bool(cuda_allocator_attribution)
        self._creator_pid = os.getpid()
        # Per-monitor scope. A plain dict avoids an inherited process lock
        # becoming permanently held if the service forks during registration.
        self._cuda_allocator_devices: dict[int, int] = {}
        self._probe = probe
        self._clock = clock or time.time
        self._samples: list[dict[str, Any]] = []
        self._total_samples = 0
        self._maxima: dict[str, int] = {}
        self._failure_count = 0
        self._recent_errors: list[dict[str, str]] = []
        self._failure_samples: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._probe_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stop_complete = True

    def _take(self) -> dict[str, Any]:
        if self.cuda_allocator_attribution and os.getpid() != self._creator_pid:
            raise RuntimeError("forked MemoryMonitor cannot sample in a different process")
        if self._probe is not None:
            return sample_current_process(self.backend, probe=self._probe, clock=self._clock)
        if self.cuda_allocator_attribution:
            return sample_current_process(self.backend, clock=self._clock,
                                          cuda_allocator_devices=dict(self._cuda_allocator_devices),
                                          cuda_allocator_attribution=True)
        return sample_current_process(self.backend, clock=self._clock)

    def register_cuda_allocator_tensor(self, tensor: Any) -> int:
        """Register a resident tensor as evidence for allocator device scope."""
        if not self.cuda_allocator_attribution:
            raise RuntimeError("CUDA allocator attribution is not enabled for this monitor")
        if os.getpid() != self._creator_pid:
            raise RuntimeError("forked MemoryMonitor cannot register tensors in a different process")
        if self.backend != "cuda":
            raise RuntimeError("CUDA allocator attribution requires a CUDA monitor")
        index, creator_pid = register_cuda_tensor_for_memory_attribution(tensor)
        self._cuda_allocator_devices[index] = creator_pid
        return index

    def _run(self) -> None:
        if self.cuda_allocator_attribution and os.getpid() != self._creator_pid:
            return
        while not self._stop.wait(self.interval_s):
            self._record()

    def _fork_refusal_receipt(self) -> dict[str, Any]:
        current_pid = os.getpid()
        error = (f"MemoryMonitor created by PID {self._creator_pid} cannot be used in PID {current_pid}; "
                 "create a new monitor in the child process")
        return {"schema_version": SCHEMA_VERSION, "pid": current_pid, "backend": self.backend,
                "timestamp_unix_s": float(time.time()), "sample_count": 0,
                "metrics": {}, "units": {}, "sources": {},
                "device_details": {"memory_monitor_status": "forked_monitor_refused",
                                   "creator_pid": self._creator_pid},
                "errors": [{"source": "MemoryMonitor process ownership", "error": error}],
                "failure_count": 1, "failure_samples": [], "observed_maxima": {},
                "samples": [], "stop_complete": False}

    def _record(self) -> dict[str, Any]:
        if self.cuda_allocator_attribution and os.getpid() != self._creator_pid:
            return self._fork_refusal_receipt()
        try:
            with self._probe_lock:
                sample = self._take()
        except Exception as exc:
            sample = {"schema_version": SCHEMA_VERSION, "pid": os.getpid(), "backend": self.backend,
                      "timestamp_unix_s": float(self._clock()), "sample_count": 1, "metrics": {},
                      "units": {},
                      "sources": {}, "errors": [{"source": "injected/default memory probe",
                      "error": f"{type(exc).__name__}: {exc}"}], "observed_maxima": {}}
        with self._lock:
            self._samples.append(sample)
            del self._samples[:-MAX_HISTORY]
            self._total_samples += 1
            self._failure_count += len(sample.get("errors", []))
            # Retain the earliest failures independently of the rolling history:
            # startup attribution failures otherwise disappear in long loads.
            if sample.get("errors") and len(self._failure_samples) < MAX_HISTORY:
                self._failure_samples.append(sample)
            self._recent_errors.extend(sample.get("errors", []))
            del self._recent_errors[:-MAX_HISTORY]
            for key, value in sample.get("metrics", {}).items():
                if (key != "rss_os_peak_bytes" and isinstance(value, (int, float))
                        and not isinstance(value, bool)):
                    self._maxima[key] = max(self._maxima.get(key, int(value)), int(value))
        return sample

    def start(self) -> "MemoryMonitor":
        if self.cuda_allocator_attribution and os.getpid() != self._creator_pid:
            raise RuntimeError("forked MemoryMonitor cannot start in a different process")
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._stop_complete = False
        self._record()
        self._thread = threading.Thread(target=self._run, name="glc-memory-monitor", daemon=True)
        self._thread.start()
        return self

    def _receipt(self) -> dict[str, Any]:
        if self.cuda_allocator_attribution and os.getpid() != self._creator_pid:
            return self._fork_refusal_receipt()
        with self._lock:
            samples = list(self._samples)
            maxima = dict(self._maxima)
            total_samples = self._total_samples
            failure_count = self._failure_count
            errors = list(self._recent_errors)
            failure_samples = list(self._failure_samples)
        units = samples[-1].get("units", {}) if samples else {}
        return {"schema_version": SCHEMA_VERSION, "pid": os.getpid(), "backend": self.backend,
                "timestamp_unix_s": float(self._clock()), "sample_count": total_samples,
                "metrics": samples[-1].get("metrics", {}) if samples else {},
                "units": units, "sources": samples[-1].get("sources", {}) if samples else {},
                "device_details": samples[-1].get("device_details", {}) if samples else {},
                "errors": errors, "failure_count": failure_count,
                "failure_samples": failure_samples,
                "observed_maxima": maxima, "samples": samples,
                "stop_complete": self._stop_complete}

    def snapshot(self) -> dict[str, Any]:
        if self.cuda_allocator_attribution and os.getpid() != self._creator_pid:
            return self._fork_refusal_receipt()
        self._record()
        return self._receipt()

    def stop(self) -> dict[str, Any]:
        if self.cuda_allocator_attribution and os.getpid() != self._creator_pid:
            return self._fork_refusal_receipt()
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(1.0, self.interval_s * 2))
        alive = bool(self._thread and self._thread.is_alive())
        self._stop_complete = not alive
        receipt = self._receipt()
        if alive:
            receipt["errors"] = list(receipt["errors"]) + [{
                "source": "MemoryMonitor.stop", "error": "sampler thread did not stop before join timeout"}]
            receipt["failure_count"] += 1
        return receipt


__all__ = ["MemoryMonitor", "sample_current_process", "register_cuda_tensor_for_memory_attribution"]
