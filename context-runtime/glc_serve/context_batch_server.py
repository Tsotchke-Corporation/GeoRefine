"""Multi-user streaming OpenAI-compatible server over the native continuous-batching engine.

    georefine-context-batch-serve --package ./context-package [--tune TUNE.json] --port 8000
    georefine-context-batch-serve --package ./context-package --runtime-format tbe21 ...
    georefine-context-batch-serve --package ./context-package --runtime-format tbe21 \
        --context-build-root .scratch/tbe21-build --parent ./hf-parent ...
    georefine-context-batch-serve --dense ./bf16-parent [--tune TUNE.json]   # same engine, BF16

The default Context package path is loaded through ``predictive_batch.load_predictive_session``
(Context -> TBE v1 startup transcode); explicit ``--runtime-format tbe21`` uses the public
``bidec_load`` loader and the same native batch session adapter. The frozen Context source needs
``--context-build-root``; typed packages need ``--context-typed``, an exact manifest pin and
``--parent`` for config/tokenizer. ``--dense`` loads the BF16 parent through the same engine and
is labelled ``dense_bf16``. Either session is wrapped in ``bidec.BatchDecoder`` + ``Batcher``
(continuous batching; a request's logits are bit-identical alone or beside any other
requests -- the ``bidec`` module docstring states the contract).

Threads: ONE engine thread drives ``Batcher.step()``. HTTP threads only validate, tokenize,
enqueue a ``bidec.Seq`` and stream from that request's own queue. The HTTP listener starts
BEFORE the model loads: ``GET /health`` answers 503 ``{"status": "loading", "phase": ...}``
with the loader's progress until the engine is ready (200 ``"ready"``). A startup failure is
printed as ``{"status": "startup_failed", "phase": ..., "error": ...}`` on stdout, served on
``/health`` (503) for ``--startup-failure-grace-s`` seconds, and the process exits 2.

Endpoints: ``GET /health`` (never needs a key), ``GET /v1/models``, ``GET /v1/receipt``,
``POST /v1/chat/completions`` (stream or not; images with ``--enable-images``).

Request extensions (non-OpenAI): ``"ignore_eos": true`` (generate exactly ``max_tokens``
tokens; for fixed-length benchmarks), ``"chat_template_kwargs"`` (default
``{"enable_thinking": false}``). Response extensions: ``"timings"`` (``queue_s``,
``ttft_s``, ``decode_s``, ``decode_tok_s``, ``e2e_s``) and, with ``--exact-digest``,
``"glc_exact"``: one SSE chunk per generated token carrying ``{"i", "token", "d"}`` where
``d`` is ``bidec.logits_row_digest`` of that token's full logits row (a list of the same
objects in a non-stream response), so a client can prove over HTTP that a request's rows are
bit-identical served alone or beside other requests.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import inspect
import json
import math
import os
import queue
import select
import socket
import sys
import threading
import time
import uuid
from .tbe21_tier_profile import CLI_PROFILES
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import startup_progress
from . import transcode_cache
from ._startup_limits import MAX_CPU_ENCODE_WORKERS

RECEIPT_SCHEMA = "georefine.context_batch_server_receipt.v1"
LOOPBACK = {"127.0.0.1", "::1", "localhost"}
DEFAULT_MODEL_ID = "georefine-context"
MAX_TEMPERATURE = 2.0
MAX_STOP_STRINGS = 16
_POLL_S = 0.25
_ALLOWED_FIELDS = {
    "model", "messages", "stream", "stream_options", "max_tokens", "max_completion_tokens",
    "temperature", "top_p", "stop", "n", "seed", "top_k", "logprobs", "top_logprobs",
    "tools", "tool_choice", "presence_penalty", "frequency_penalty", "user",
    "ignore_eos", "chat_template_kwargs",
}
_MESSAGE_FIELDS = {"role", "content", "name"}


def _memory_backend(device: Any) -> str:
    name = str(device).lower().split(":", 1)[0]
    if name in {"cuda", "gpu"}:
        return "cuda"
    if name in {"metal", "mlx"}:
        return "mlx"
    return "cpu"


def _memory_monitor_factory(**kwargs):
    # Imported only for explicit opt-in so the default server startup is unchanged.
    from .runtime_memory import MemoryMonitor
    return MemoryMonitor(**kwargs)


class RequestRejected(ValueError):
    """A request the server refuses before it reaches the engine (HTTP ``status``)."""

    def __init__(self, message: str, status: int = 400, error_type: str = "invalid_request_error",
                 headers: Optional[Dict[str, str]] = None):
        super().__init__(message)
        self.status = int(status)
        self.error_type = error_type
        self.headers = dict(headers or {})


# ============================================================================ request parsing
@dataclass
class ChatRequest:
    messages: List[Dict[str, Any]]
    stream: bool
    include_usage: bool
    max_tokens: int
    max_tokens_requested: Optional[int]
    temperature: float
    top_p: float
    seed: Optional[int]
    stop: List[str]
    ignore_eos: bool
    chat_template_kwargs: Dict[str, Any]


def _number(body: Dict[str, Any], name: str, default: float) -> float:
    value = body.get(name, default)
    if value is None:
        return float(default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestRejected(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise RequestRejected(f"{name} must be finite")
    return value


def _positive_int_field(body: Dict[str, Any], name: str) -> Optional[int]:
    value = body.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RequestRejected(f"{name} must be a positive integer")
    return int(value)


def _validate_messages(messages: Any) -> List[Dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise RequestRejected("messages must be a non-empty array")
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise RequestRejected(f"messages[{i}] must be an object")
        extra = sorted(set(message) - _MESSAGE_FIELDS)
        if extra:
            raise RequestRejected(f"messages[{i}] has unsupported field(s): {', '.join(extra)} "
                                  "(tool calls are not supported)")
        if not isinstance(message.get("role"), str) or not message["role"]:
            raise RequestRejected(f"messages[{i}].role must be a non-empty string")
        content = message.get("content")
        if content is not None and not isinstance(content, (str, list)):
            raise RequestRejected(f"messages[{i}].content must be a string or an array of parts")
        if isinstance(content, list):
            for j, part in enumerate(content):
                if not isinstance(part, dict) or not isinstance(part.get("type"), str):
                    raise RequestRejected(f"messages[{i}].content[{j}] must be an object with a type")
    return messages


def parse_chat_request(body: Dict[str, Any], *, model_id: str, default_max_new: int,
                       max_new: int) -> ChatRequest:
    """Validate one ``/v1/chat/completions`` body; unsupported fields are a 400."""
    if not isinstance(body, dict):
        raise RequestRejected("body must be a JSON object")
    unknown = sorted(set(body) - _ALLOWED_FIELDS)
    if unknown:
        raise RequestRejected(f"unsupported request field(s): {', '.join(unknown)}")
    model = body.get("model")
    if model is not None:
        if not isinstance(model, str):
            raise RequestRejected("model must be a string")
        if model != model_id:
            raise RequestRejected(f"model {model!r} is not served here (serving {model_id!r})",
                                  404, "model_not_found")
    messages = _validate_messages(body.get("messages"))
    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise RequestRejected("stream must be boolean")
    stream_options = body.get("stream_options") or {}
    if not isinstance(stream_options, dict) or set(stream_options) - {"include_usage"}:
        raise RequestRejected("only stream_options.include_usage is supported")
    include_usage = stream_options.get("include_usage", False)
    if not isinstance(include_usage, bool):
        raise RequestRejected("stream_options.include_usage must be boolean")
    if include_usage and not stream:
        raise RequestRejected("stream_options.include_usage requires stream=true")

    temperature = _number(body, "temperature", 0.0)
    if not 0.0 <= temperature <= MAX_TEMPERATURE:
        raise RequestRejected(f"temperature must be in [0, {MAX_TEMPERATURE}]")
    top_p = _number(body, "top_p", 1.0)
    if not 0.0 < top_p <= 1.0:
        raise RequestRejected("top_p must be in (0, 1]")
    seed = body.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise RequestRejected("seed must be an integer")
    if body.get("top_k") not in (None, 0, -1):
        raise RequestRejected("top_k is not supported (the engine samples with temperature/top_p)")
    for name in ("presence_penalty", "frequency_penalty"):
        if _number(body, name, 0.0) != 0.0:
            raise RequestRejected(f"{name} is not supported")
    n = body.get("n", 1)
    if isinstance(n, bool) or not isinstance(n, int) or n != 1:
        raise RequestRejected("only n=1 is supported")
    if body.get("logprobs") not in (None, False):
        raise RequestRejected("logprobs are not supported")
    if body.get("top_logprobs") not in (None, 0):
        raise RequestRejected("top_logprobs are not supported")
    if body.get("tools") not in (None, []):
        raise RequestRejected("tools are not supported")
    if body.get("tool_choice") not in (None, "none"):
        raise RequestRejected("tool_choice is not supported")
    if body.get("user") is not None and not isinstance(body.get("user"), str):
        raise RequestRejected("user must be a string")

    requested = _positive_int_field(body, "max_completion_tokens")
    if requested is None:
        requested = _positive_int_field(body, "max_tokens")
    effective = min(requested if requested is not None else int(default_max_new), int(max_new))

    stop = body.get("stop")
    if stop is None:
        stop = []
    elif isinstance(stop, str):
        stop = [stop]
    if not isinstance(stop, list) or any(not isinstance(s, str) or not s for s in stop):
        raise RequestRejected("stop must be a non-empty string or an array of non-empty strings")
    if len(stop) > MAX_STOP_STRINGS:
        raise RequestRejected(f"at most {MAX_STOP_STRINGS} stop strings are supported")

    ignore_eos = body.get("ignore_eos", False)
    if not isinstance(ignore_eos, bool):
        raise RequestRejected("ignore_eos must be boolean")
    ctk = body.get("chat_template_kwargs")
    if ctk is None:
        ctk = {}
    if not isinstance(ctk, dict) or any(not isinstance(k, str) for k in ctk):
        raise RequestRejected("chat_template_kwargs must be an object")
    template_kwargs = {"enable_thinking": False, **ctk}
    return ChatRequest(messages=messages, stream=stream, include_usage=include_usage,
                       max_tokens=effective, max_tokens_requested=requested,
                       temperature=temperature, top_p=top_p, seed=seed, stop=list(stop),
                       ignore_eos=ignore_eos, chat_template_kwargs=template_kwargs)


# =================================================================================== text
class _Detokenizer:
    """Incremental detokenizer (prefix/read offsets) that withholds incomplete UTF-8."""

    def __init__(self, decode: Callable[[List[int]], str], skip_ids: Sequence[int]):
        self._decode = decode
        self._skip = set(int(t) for t in skip_ids)
        self.ids: List[int] = []
        self._prefix = 0
        self._read = 0

    def push(self, token: int) -> str:
        if int(token) in self._skip:
            return ""
        self.ids.append(int(token))
        prefix_text = self._decode(self.ids[self._prefix:self._read])
        new_text = self._decode(self.ids[self._prefix:])
        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            self._prefix, self._read = self._read, len(self.ids)
            return new_text[len(prefix_text):]
        return ""

    def full_text(self) -> str:
        return self._decode(self.ids)


class _TextStream:
    """Detokenized deltas with stop-string truncation (holds back a possible stop prefix)."""

    def __init__(self, detok: _Detokenizer, stop: Sequence[str]):
        self.detok = detok
        self.stop = [s for s in stop if s]
        self.hold = max((len(s) for s in self.stop), default=1) - 1
        self.emitted = ""
        self.pending = ""
        self.stopped = False

    def _scan(self, final: bool) -> str:
        cut = None
        for s in self.stop:
            idx = self.pending.find(s)
            if idx >= 0 and (cut is None or idx < cut):
                cut = idx
        if cut is not None:
            out, self.pending, self.stopped = self.pending[:cut], "", True
        elif final or not self.hold:
            out, self.pending = self.pending, ""
        else:
            keep = min(self.hold, len(self.pending))
            out = self.pending[:len(self.pending) - keep]
            self.pending = self.pending[len(self.pending) - keep:]
        self.emitted += out
        return out

    def push(self, token: int) -> str:
        if self.stopped:
            return ""
        self.pending += self.detok.push(token)
        return self._scan(final=False)

    def finish(self) -> str:
        if self.stopped:
            return ""
        full = self.detok.full_text()
        seen = self.emitted + self.pending
        if len(full) > len(seen) and full.startswith(seen):
            self.pending += full[len(seen):]
        return self._scan(final=True)


# ================================================================================ service
@dataclass
class Ticket:
    seq: Any
    events: "queue.Queue"
    request: ChatRequest
    t_submit: float
    prompt_tokens: int
    mm_info: Optional[Dict[str, Any]] = None
    tokens: List[int] = field(default_factory=list)
    digests: List[Dict[str, Any]] = field(default_factory=list)
    t_last: float = 0.0
    t_checked: float = 0.0
    finish: str = ""
    error: str = ""


def token_callback(events: "queue.Queue"):
    """``Seq.on_token`` for one request: ``(token, row_digest)``; ``(None, None)`` ends it.

    Called from the engine thread. The digest is read here because the engine appends it to
    ``seq.digests`` immediately before calling back; reading it later would race the next step.
    """
    def on_token(seq, token):
        digest = None
        if token is not None and getattr(seq, "digest", False) and seq.digests:
            digest = seq.digests[-1]
        events.put((token, digest))
    return on_token


@dataclass
class Limits:
    max_prompt: int = 12288
    max_new: int = 2048
    default_max_new: int = 1024
    max_queue: int = 256
    exact_digest: bool = False
    enable_images: bool = False


class BatchService:
    """Engine thread + admission. HTTP threads call :meth:`submit` and read the ticket queue."""

    def __init__(self, *, model_id: str, source_kind: str, source: str, limits: Limits,
                 settings: Dict[str, Any], api_keys: Sequence[str] = (),
                 startup_failure_grace_s: float = 10.0, memory_metrics: bool = False,
                 memory_interval_s: float = 0.1,
                 memory_allocator_attribution: bool = False,
                 memory_monitor_factory: Optional[Callable[..., Any]] = None):
        self.model_id = model_id
        self.source_kind = source_kind
        self.source = source
        self.limits = limits
        self.settings = dict(settings)
        self.api_keys = [k for k in api_keys if k]
        self.startup_failure_grace_s = float(startup_failure_grace_s)
        self.memory_metrics_enabled = bool(memory_metrics)
        self.memory_interval_s = float(memory_interval_s)
        self.memory_allocator_attribution = bool(memory_allocator_attribution)
        if self.memory_allocator_attribution and (
                not self.memory_metrics_enabled or _memory_backend(settings.get("device", "cpu")) != "cuda"):
            raise ValueError("allocator attribution requires CUDA and memory metrics")
        self._allocator_phase_snapshots: Dict[str, Any] = {}
        self.memory_monitor: Any = None
        self._memory_monitor_factory: Optional[Callable[..., Any]] = None
        self._memory_monitor_started = False
        self._memory_monitor_stopped = False
        self._memory_measurement_scope = "process_lifetime"
        self._startup_memory_receipt: Optional[Dict[str, Any]] = None
        if self.memory_metrics_enabled:
            factory = memory_monitor_factory or _memory_monitor_factory
            self._memory_monitor_factory = factory
            monitor_options = {"cuda_allocator_attribution": True} if self.memory_allocator_attribution else {}
            self.memory_monitor = factory(backend=_memory_backend(settings.get("device", "cpu")),
                                         interval_s=self.memory_interval_s, **monitor_options)
        self.status = "loading"
        self.failed_phase: Optional[str] = None
        self.error: Optional[str] = None
        self.fatal = ""
        self.progress: Dict[str, Any] = {"phase": "init"}
        self.startup_timings: Dict[str, float] = {}
        self.t_start = time.time()
        self.runtime: Any = None
        self.batcher: Any = None
        self.tokenizer: Any = None
        self.processor: Any = None
        self.mm_frontend: Any = None
        self.eos_ids: List[int] = []
        self.vocab: Optional[int] = None
        self.qualification_identity: Optional[Dict[str, Any]] = None
        self.served = 0
        self.cancelled = 0
        self.rejected_queue_full = 0
        self.failed_requests = 0
        self._counter_lock = threading.Lock()
        self._submit_lock = threading.Lock()
        self._tok_lock = threading.RLock()
        self.wake = threading.Event()
        self._stop = threading.Event()
        self._engine_thread: Optional[threading.Thread] = None
        self._last_print: Tuple[str, float] = ("", 0.0)

    # ----------------------------------------------------------------- startup lifecycle
    def on_progress(self, event: Dict[str, Any]) -> None:
        """``startup_progress`` sink: keep the latest event; print phase changes as JSON."""
        self.progress = dict(event)
        phase = str(event.get("phase"))
        now = time.time()
        last_phase, last_t = self._last_print
        if phase != last_phase or now - last_t >= 10.0:
            self._last_print = (phase, now)
            line = {"status": "loading", "phase": phase}
            for key in ("done", "total", "fraction", "eta_s", "unit"):
                if key in event:
                    line[key] = event[key]
            _emit_line(line)

    def attach(self, runtime: Any, *, start_engine: bool = True) -> None:
        if getattr(self, '_resident_startup_enabled', False):
            import uuid
            self._resident_model_instance = uuid.uuid4().hex
            self._resident_load_generation = getattr(self, '_resident_load_generation', 0) + 1
        self.runtime = runtime
        self.batcher = runtime.batcher
        self.tokenizer = runtime.tokenizer
        self.processor = getattr(runtime, "processor", None)
        self.mm_frontend = getattr(runtime, "mm_frontend", None)
        session = getattr(runtime, "session", None)
        eos = list(getattr(session, "eos_ids", None) or [])
        if not eos:
            from .bidec import stop_ids_for

            eos = stop_ids_for(self.tokenizer, None)
        self.eos_ids = sorted({int(e) for e in eos})
        logits = getattr(getattr(runtime, "bd", None), "logits", None)
        self.vocab = int(logits.shape[-1]) if logits is not None else None
        if start_engine:
            self._engine_thread = threading.Thread(target=self.loop, name="context-batch-engine",
                                                   daemon=True)
            self._engine_thread.start()
        self._start_ready_memory_window()
        self.status = "ready"
        startup_progress.report("ready")

    def stop(self) -> None:
        """Stop the engine thread after its current step (used at shutdown and in tests)."""
        self._stop.set()
        self.wake.set()
        if self._engine_thread is not None:
            self._engine_thread.join(timeout=5)
        self.stop_memory_monitor()

    def start_memory_monitor(self) -> None:
        if self.memory_monitor is not None and not self._memory_monitor_started:
            self.memory_monitor.start()
            self._memory_monitor_started = True
            self._memory_monitor_stopped = False

    def stop_memory_monitor(self) -> None:
        if self.memory_monitor is not None and self._memory_monitor_started and not self._memory_monitor_stopped:
            self.memory_monitor.stop()
            self._memory_monitor_stopped = True

    def _register_allocator_device(self, loaded: Any, monitor: Any) -> None:
        """Use a resident engine tensor; never resolve CUDA through a device query."""
        session = getattr(loaded, "session", None)
        if session is None:
            session = loaded
        fd = getattr(session, "fd", None)
        bd = getattr(loaded, "bd", None)
        device = str(self.settings.get("device", "cuda"))
        expected_index = int(device.split(":", 1)[1]) if ":" in device else None
        for owner, name in ((fd, "embed_w"), (fd, "x"), (fd, "final_ln"),
                            (bd, "x"), (bd, "logits")):
            tensor = getattr(owner, name, None)
            resident = getattr(tensor, "device", None)
            index = getattr(resident, "index", None)
            if (getattr(resident, "type", None) == "cuda" and type(index) is int
                    and (expected_index is None or index == expected_index)):
                monitor.register_cuda_allocator_tensor(tensor)
                return
        raise ValueError("allocator attribution needs a resident tensor on the serving device")

    def record_allocator_phase(self, phase: str, loaded: Any) -> None:
        if not self.memory_allocator_attribution:
            return
        if phase not in {"weights_loaded", "page_plan_complete", "batch_capture_complete", "ready"}:
            raise ValueError("unknown allocator attribution phase")
        self._register_allocator_device(loaded, self.memory_monitor)
        sample = self.memory_monitor.snapshot()
        # Phase summaries are bounded and retain no tensor, storage or graph references.
        summary = {
            key: sample.get(key) for key in
            ("timestamp_unix_s", "sample_count", "metrics", "sources", "device_details", "errors")}
        with self._counter_lock:
            self._allocator_phase_snapshots[phase] = summary

    def _start_ready_memory_window(self) -> None:
        # nvidia-smi often cannot attribute this PID until the CUDA context exists.
        # Preserve those startup readings, then begin an independent ready-window monitor.
        if (self.memory_monitor is None or not self._memory_monitor_started or
                _memory_backend(self.settings.get("device", "cpu")) != "cuda"):
            return
        self._startup_memory_receipt = self.memory_monitor.stop()
        self._memory_monitor_stopped = True
        monitor_options = {"cuda_allocator_attribution": True} if self.memory_allocator_attribution else {}
        replacement = self._memory_monitor_factory(backend="cuda", interval_s=self.memory_interval_s,
                                                    **monitor_options)
        if self.memory_allocator_attribution:
            self._register_allocator_device(self.runtime, replacement)
        replacement.start()
        self.memory_monitor = replacement
        self._memory_monitor_started = True
        self._memory_monitor_stopped = False
        self._memory_measurement_scope = "post_ready"
        if self.memory_allocator_attribution:
            self.record_allocator_phase("ready", self.runtime)

    def memory(self) -> Dict[str, Any]:
        """Current-process memory receipt; disabled state is explicit and contains no measurements."""
        backend = _memory_backend(self.settings.get("device", "cpu"))
        if self.memory_monitor is None:
            return self._qualification_memory_fields({"schema_version": 1, "pid": os.getpid(), "backend": backend,
                    "timestamp_unix_s": time.time(), "sample_count": 0,
                    "metrics": {"rss_current_bytes": None, "rss_os_peak_bytes": None},
                    "units": {"rss_current_bytes": "bytes", "rss_os_peak_bytes": "bytes"},
                    "sources": {"rss_current_bytes": "memory metrics disabled",
                                "rss_os_peak_bytes": "memory metrics disabled"},
                    "errors": [], "failure_count": 0, "observed_maxima": {}, "samples": [],
                    "enabled": False, "monitor_status": "disabled", "server_status": self.status,
                    "measurement_scope": "disabled"})
        if not self._memory_monitor_started:
            return self._qualification_memory_fields({"schema_version": 1, "pid": os.getpid(), "backend": backend,
                    "timestamp_unix_s": time.time(), "sample_count": 0,
                    "metrics": {"rss_current_bytes": None, "rss_os_peak_bytes": None},
                    "units": {"rss_current_bytes": "bytes", "rss_os_peak_bytes": "bytes"},
                    "sources": {"rss_current_bytes": "memory monitor not started",
                                "rss_os_peak_bytes": "memory monitor not started"},
                    "errors": [], "failure_count": 0, "observed_maxima": {}, "samples": [],
                    "enabled": True, "monitor_status": "starting", "server_status": self.status,
                    "measurement_scope": self._memory_measurement_scope})
        receipt = dict(self.memory_monitor.snapshot())
        receipt.update(enabled=True,
                       monitor_status="stopped" if self._memory_monitor_stopped else "running",
                       server_status=self.status,
                       measurement_scope=self._memory_measurement_scope)
        if self._startup_memory_receipt is not None:
            receipt["startup_memory_receipt"] = self._startup_memory_receipt
        if self.memory_allocator_attribution:
            from copy import deepcopy
            with self._counter_lock:
                phases = deepcopy(self._allocator_phase_snapshots)
            receipt["allocator_attribution"] = {
                "enabled": True, "phase_snapshots": phases,
                "interpretation": "Registered-device allocator counters are distinct from whole-process "
                                  "device residency; reserved-minus-allocated is not guaranteed reclaimable headroom."}
        return self._qualification_memory_fields(receipt)

    def _qualification_memory_fields(self, receipt: Dict[str, Any]) -> Dict[str, Any]:
        if self.qualification_identity is None:
            return receipt
        identity = self.qualification_identity
        receipt.update({k: identity.get(k) for k in
                                    ("frame_id", "protocol_sha256", "qualification_role")})
        stats = self.batcher.spec_stats() if self.batcher is not None else None
        receipt["serving_counters"] = {
            "drafts_proposed": (stats or {}).get('proposed_total',0),
            "drafts_accepted": (stats or {}).get('accepted_total',0),
        }
        if getattr(self.batcher, "mtp_confirm_kv_only", False):
            detail = self.batcher.stats()
            receipt["mtp_confirmation_counters"] = detail["mtp_confirmation_counters"]
            receipt["speculative_trunk_bucket_counts"] = detail["speculative_trunk_bucket_counts"]
        if isinstance(stats,dict) and stats.get('error'):
            receipt.setdefault('errors',[]).append({'source':'speculation counters','error':stats['error']})
            receipt['failure_count']=receipt.get('failure_count',0)+1
        return receipt

    def fail_startup(self, phase: str, error: str) -> None:
        self.failed_phase = phase
        self.error = error
        self.status = "startup_failed"

    # ------------------------------------------------------------------------ engine
    def loop(self) -> None:
        batcher = self.batcher
        while not self.fatal and not self._stop.is_set():
            if batcher.idle():
                self.wake.wait(0.05)
                self.wake.clear()
                continue
            try:
                batcher.step()
            except Exception as exc:  # noqa: BLE001 - the engine must never die silently
                self._engine_failed(f"{type(exc).__name__}: {exc}")
                return

    def _engine_failed(self, message: str) -> None:
        """Device buffers are in an unknown state: stop, end every stream, answer 503."""
        self.fatal = message
        self.error = message
        self.status = "error"
        sys.stderr.write(f"[context-batch] engine step failed, engine stopped: {message}\n")
        batcher = self.batcher
        with batcher.lock:
            seqs = list(batcher.waiting) + list(batcher.active)
            batcher.waiting, batcher.active = [], []
        for seq in seqs:
            seq.done, seq.finish = True, "error"
            if not getattr(seq, "error", ""):
                seq.error = message
            if seq.on_token is not None:
                try:
                    seq.on_token(seq, None)
                except Exception:  # noqa: BLE001 - one bad callback must not strand others
                    pass

    # -------------------------------------------------------------------- admission
    def _apply_template(self, hf_messages, template_kwargs) -> List[int]:
        processor = self.processor
        templ = processor if hasattr(processor, "apply_chat_template") else self.tokenizer
        with self._tok_lock:
            text = templ.apply_chat_template(hf_messages, tokenize=False,
                                             add_generation_prompt=True, **template_kwargs)
            if processor is not None and getattr(processor, "image_processor", None) is not None:
                enc = processor(text=[text], return_tensors="pt")
            else:
                enc = self.tokenizer([text], add_special_tokens=False)
        ids = enc["input_ids"][0]
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return [int(t) for t in ids]

    def build_prompt(self, req: ChatRequest) -> Tuple[List[int], Any]:
        from .bidec_mm import MMRequestError, has_image_parts, split_openai_messages

        messages = req.messages
        if has_image_parts(messages):
            text_only = all(part.get("type") == "text"
                            for m in messages if isinstance(m.get("content"), list)
                            for part in m["content"])
            if not text_only:
                if self.mm_frontend is None:
                    raise RequestRejected("image input is not enabled on this server "
                                          "(start it with --enable-images)")
                try:
                    with self._tok_lock:
                        ids, mm_inputs = self.mm_frontend.prepare(messages,
                                                                  req.chat_template_kwargs)
                except MMRequestError as exc:
                    raise RequestRejected(str(exc)) from exc
                return [int(t) for t in ids], mm_inputs
            try:
                hf_messages, _images, _digests = split_openai_messages(messages)
            except MMRequestError as exc:
                raise RequestRejected(str(exc)) from exc
        else:
            hf_messages = [{**m, "content": m.get("content") or ""} for m in messages]
        return self._apply_template(hf_messages, req.chat_template_kwargs), None

    def _unavailable(self) -> Optional[RequestRejected]:
        if self.status == "loading":
            return RequestRejected("the model is still loading", 503, "server_loading",
                                   {"Retry-After": "5"})
        if self.status == "startup_failed":
            return RequestRejected(f"startup failed: {self.error}", 503, "server_error")
        if self.fatal:
            return RequestRejected(f"engine stopped: {self.fatal}", 503, "server_error")
        return None

    def submit(self, req: ChatRequest) -> Ticket:
        unavailable = self._unavailable()
        if unavailable is not None:
            raise unavailable
        if self.settings.get("spec_k", 0) and req.temperature != 0:
            raise RequestRejected("speculative decoding currently requires temperature=0")
        t_submit = time.time()
        ids, mm_inputs = self.build_prompt(req)
        if not ids:
            raise RequestRejected("the rendered prompt is empty")
        if len(ids) > self.limits.max_prompt:
            raise RequestRejected(f"prompt has {len(ids)} tokens > --max-prompt "
                                  f"{self.limits.max_prompt}")
        from .bidec import Seq

        events: "queue.Queue" = queue.Queue()
        seq = Seq(rid=uuid.uuid4().hex, prompt=list(ids), max_new=int(req.max_tokens),
                  stop_ids=() if req.ignore_eos else tuple(self.eos_ids),
                  temperature=float(req.temperature), top_p=float(req.top_p), seed=req.seed,
                  digest=bool(self.limits.exact_digest), mm=mm_inputs)
        seq.on_token = token_callback(events)
        batcher = self.batcher
        with self._submit_lock:
            with batcher.lock:
                waiting = len(batcher.waiting)
            if waiting >= self.limits.max_queue:
                with self._counter_lock:
                    self.rejected_queue_full += 1
                raise RequestRejected(f"server busy: {waiting} requests already queued "
                                      f"(--max-queue {self.limits.max_queue})", 429,
                                      "rate_limit_exceeded", {"Retry-After": "2"})
            try:
                batcher.add(seq)
            except ValueError as exc:
                raise RequestRejected(f"context overflow: {exc}") from exc
        if self.fatal:
            batcher.cancel(seq.rid)
            raise RequestRejected(f"engine stopped: {self.fatal}", 503, "server_error")
        self.wake.set()
        return Ticket(seq=seq, events=events, request=req, t_submit=t_submit,
                      prompt_tokens=len(ids))

    def cancel(self, ticket: Ticket, *, client: bool = True) -> None:
        """Abort a request in the engine (slot and pages return to the pool)."""
        try:
            self.batcher.cancel(ticket.seq.rid)
        finally:
            if client:
                with self._counter_lock:
                    self.cancelled += 1

    def decode(self, ids: List[int]) -> str:
        with self._tok_lock:
            return self.tokenizer.decode(ids, skip_special_tokens=False)

    # ---------------------------------------------------------------------- reports
    def check_key(self, header: str) -> bool:
        if not self.api_keys:
            return True
        token = header[7:].strip() if header[:7].lower() == "bearer " else ""
        return bool(token) and any(hmac.compare_digest(token.encode(), k.encode())
                                   for k in self.api_keys)

    def health(self) -> Tuple[int, Dict[str, Any]]:
        snap = startup_progress.snapshot()
        out: Dict[str, Any] = {"status": self.status, "model": self.model_id,
                               "source_kind": self.source_kind,
                               "uptime_s": round(time.time() - self.t_start, 3)}
        if self.status == "loading":
            out.update({k: v for k, v in self.progress.items()
                        if k not in ("t", "status", "model", "source_kind", "uptime_s")})
            if "status" in self.progress:
                out["progress_status"] = self.progress["status"]
            out["phase"] = self.progress.get("phase", "init")
            out["phases"] = snap["phases"]
            return 503, out
        if self.status == "startup_failed":
            out.update({"phase": self.failed_phase, "error": self.error,
                        "phases": snap["phases"]})
            return 503, out
        stats: Dict[str, Any] = {}
        try:
            stats = dict(self.batcher.stats())
        except Exception as exc:  # noqa: BLE001 - health must answer even then
            stats = {"stats_error": f"{type(exc).__name__}: {exc}"}
        page_size = 256
        out.update({
            "error": self.fatal or None,
            "batcher": stats,
            "capacity": {
                "max_users": stats.get("max_users", self.settings.get("max_slots")),
                "active": stats.get("active"), "waiting": stats.get("waiting"),
                "max_queue": self.limits.max_queue,
                "max_context": stats.get("max_ctx", self.settings.get("max_ctx")),
                "pages_total": stats.get("pages_total", self.settings.get("pages_total")),
                "pages_reserved": stats.get("pages_reserved"),
                "page_size": page_size,
                "full_slot_context_capacity": (self.settings.get("capacity") or {}).get(
                    "full_slot_context_capacity"),
            },
            "settings": self.public_settings(),
            "served": self.served, "cancelled": self.cancelled,
            "rejected_queue_full": self.rejected_queue_full,
            "failed_requests": self.failed_requests,
            "exact_digest": bool(self.limits.exact_digest), "vocab": self.vocab,
        })
        if self.fatal:
            out["status"] = "error"
            return 503, out
        return 200, out

    def public_settings(self) -> Dict[str, Any]:
        keep = ("max_slots", "max_rows", "max_rows_step", "pages_total", "max_ctx",
                "prefill_chunk", "prefill_priority")
        out = {k: self.settings.get(k) for k in keep}
        out.update({"max_prompt": self.limits.max_prompt, "max_new": self.limits.max_new,
                    "default_max_new": self.limits.default_max_new,
                    "max_queue": self.limits.max_queue,
                    "exact_digest": bool(self.limits.exact_digest),
                    "enable_images": bool(self.limits.enable_images),
                    "source_kind": self.source_kind, "source": self.source,
                    "sampling": "greedy default; temperature/top_p/seed per request"})
        if self.settings.get("spec_k", 0):
            out.update(spec_k=self.settings["spec_k"],
                       spec_ngram=self.settings.get("spec_ngram", False),
                       max_decode_rows=self.settings.get("max_decode_rows", 64),
                       sampling="greedy only with speculative decoding")
        for key in ("device", "vram_headroom_gib", "pages_total_requested", "optimisation_profile",
                    "spec_row_aware", "mtp_confirm_kv_only"):
            if key in self.settings:
                out[key] = self.settings[key]
        return out

    def receipt(self) -> Dict[str, Any]:
        meta = dict(getattr(self.runtime, "metadata", {}) or {})
        source_session = meta.get("source_session") or {}
        result = {
            "schema": RECEIPT_SCHEMA,
            "model": self.model_id,
            "status": self.status,
            "source_kind": self.source_kind,
            "source": self.source,
            "settings": self.public_settings(),
            "dynamic_batching": True,
            "exactness_contract": ("a request's logits rows are bit-identical served alone or "
                                   "beside any other requests (glc_serve.bidec)"),
            "startup_timings_s": dict(self.startup_timings),
            "startup_progress": startup_progress.snapshot(),
            "transcode_cache": source_session.get("transcode_cache",
                                                  self.settings.get("transcode_cache")),
            "page_plan": meta.get("page_plan"),
            "vocab": self.vocab,
            "eos_ids": list(self.eos_ids),
            "runtime": meta,
        }
        if self.qualification_identity is not None:
            result.update(self.qualification_identity)
        if getattr(self, '_resident_startup_enabled', False) and self.runtime is not None:
            # Linux incarnation comes from the kernel, not a caller-supplied PID label.
            tail = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()
            with self._counter_lock:
                result['resident_instance'] = {
                    'process': {'pid': os.getpid(),
                        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                        'start_ticks': int(tail[19])},
                    'model_instance_id': self._resident_model_instance,
                    'load_generation': self._resident_load_generation,
                    'served': self.served, 'failed_requests': self.failed_requests}
        if self.settings.get("spec_k", 0) and self.batcher is not None:
            result["speculation"] = self.batcher.spec_stats()
        if getattr(self.batcher, "mtp_confirm_kv_only", False):
            result["mtp_confirm_kv_only"] = True
        return result


def _emit_line(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, default=str) + "\n")
    sys.stdout.flush()


# ================================================================================== HTTP
class Server(ThreadingHTTPServer):
    """ThreadingHTTPServer whose listen backlog a multi-user benchmark cannot overflow."""

    request_queue_size = 512
    daemon_threads = True
    allow_reuse_address = True


def _timings(ticket: Ticket) -> Dict[str, Any]:
    seq = ticket.seq
    t_admit = float(getattr(seq, "t_admit", 0.0) or 0.0)
    t_first = float(getattr(seq, "t_first", 0.0) or 0.0)
    t_last = ticket.t_last or time.time()
    n = len(ticket.tokens)
    out: Dict[str, Any] = {
        "queue_s": round(t_admit - ticket.t_submit, 6) if t_admit else None,
        "ttft_s": round(t_first - ticket.t_submit, 6) if t_first else None,
        "decode_s": round(t_last - t_first, 6) if t_first else None,
        "e2e_s": round(t_last - ticket.t_submit, 6),
        "decode_tok_s": (round((n - 1) / (t_last - t_first), 3)
                         if t_first and n > 1 and t_last > t_first else None),
        "max_tokens_applied": int(ticket.request.max_tokens),
    }
    return out


def _mm_info(service: BatchService, seq: Any) -> Dict[str, Any]:
    if getattr(seq, "mm", None) is None:
        return {}
    plan = getattr(seq, "mm_plan", None)
    mm = service.mm_frontend
    info = {"native": True, "n_images": int(seq.mm.image_grid_thw.shape[0]),
            "image_sha256": list(seq.mm.image_sha256),
            "vision_bitexact": bool(mm.bitexact) if mm is not None else None}
    if plan is not None:
        info.update({"n_image_tokens": int(plan.embeds.shape[0]),
                     "rope_delta": int(plan.rope_delta)})
    return info


def make_handler(service: BatchService):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "context-batch/0.1.1"

        def log_message(self, fmt, *args):  # noqa: D401 - stdlib signature
            if os.environ.get("CONTEXT_BATCH_ACCESS_LOG"):
                sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        # ---------------------------------------------------------------- plumbing
        def _json(self, status: int, value: Any, headers: Optional[Dict[str, str]] = None):
            data = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            for key, val in (headers or {}).items():
                self.send_header(key, val)
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: int, message: str, error_type: str = "invalid_request_error",
                   headers: Optional[Dict[str, str]] = None):
            self._json(status, {"error": {"message": message, "type": error_type,
                                          "code": status}}, headers)

        def _authorized(self) -> bool:
            if service.check_key(self.headers.get("Authorization", "")):
                return True
            self._error(401, "invalid or missing API key", "authentication_error")
            return False

        def _body(self) -> Dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError as exc:
                raise RequestRejected("invalid Content-Length") from exc
            limit = int(getattr(self.server, "max_body_bytes", 32 << 20))
            if length < 0 or length > limit:
                raise RequestRejected(f"request body exceeds {limit} bytes", 413)
            raw = self.rfile.read(length) if length else b""
            if not raw:
                raise RequestRejected("request body is required")
            try:
                body = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise RequestRejected(f"body is not valid UTF-8 JSON: {exc}") from exc
            if not isinstance(body, dict):
                raise RequestRejected("body must be a JSON object")
            return body

        def _client_gone(self) -> bool:
            """True when the client closed its side (readable socket with EOF)."""
            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                if not readable:
                    return False
                return self.connection.recv(1, socket.MSG_PEEK) == b""
            except (OSError, ValueError):
                return True

        # ------------------------------------------------------------------- GET
        def do_GET(self):
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path == "/health":
                status, body = service.health()
                return self._json(status, body)
            if not self._authorized():
                return None
            if path == "/v1/memory":
                return self._json(200, service.memory())
            if path == "/v1/models":
                return self._json(200, {"object": "list", "data": [{
                    "id": service.model_id, "object": "model", "created": int(service.t_start),
                    "owned_by": "georefine", "source_kind": service.source_kind,
                    "capabilities": (["chat", "stream", "images"] if service.limits.enable_images
                                     else ["chat", "stream"]),
                    "dynamic_batching": True,
                    "max_context": service.settings.get("max_ctx"),
                }]})
            if path == "/v1/receipt":
                return self._json(200, service.receipt())
            return self._error(404, f"no route {self.path}", "not_found")

        # ------------------------------------------------------------------ POST
        def do_POST(self):
            path = self.path.split("?", 1)[0].rstrip("/")
            try:
                body = self._body()
            except RequestRejected as exc:
                return self._error(exc.status, str(exc), exc.error_type)
            if not self._authorized():
                return None
            if path != "/v1/chat/completions":
                return self._error(404, f"no route {self.path}", "not_found")
            try:
                req = parse_chat_request(body, model_id=service.model_id,
                                         default_max_new=service.limits.default_max_new,
                                         max_new=service.limits.max_new)
                ticket = service.submit(req)
            except RequestRejected as exc:
                return self._error(exc.status, str(exc), exc.error_type, exc.headers)
            except Exception as exc:  # noqa: BLE001 - tokenizer/template failure
                return self._error(400, f"{type(exc).__name__}: {exc}")
            if req.stream:
                return self._stream(ticket)
            return self._complete(ticket)

        # -------------------------------------------------------------- generation
        def _next_event(self, ticket: Ticket):
            """Next ``(token, digest)``; ``None`` at the end. Raises ConnectionAbortedError
            when the client went away."""
            while True:
                now = time.monotonic()
                if now - ticket.t_checked >= _POLL_S:
                    # Checked on a clock, not only when idle: a request whose tokens keep
                    # arriving would otherwise never notice that nobody is reading them.
                    ticket.t_checked = now
                    if self._client_gone():
                        raise ConnectionAbortedError("client disconnected")
                try:
                    return ticket.events.get(timeout=_POLL_S)
                except queue.Empty:
                    pass
                if service.fatal and getattr(ticket.seq, "done", False) is False:
                    # The engine died; it ended every stream it knew. A request that was not
                    # in its lists by then would otherwise wait forever.
                    ticket.seq.finish = "error"
                    ticket.seq.error = service.fatal
                    return (None, None)

        def _run(self, ticket: Ticket, text: _TextStream,
                 on_token: Callable[[int, Optional[str], str], None]) -> str:
            """Drain the ticket; ``on_token(token, digest, text_delta)`` per token. Returns the
            finish reason. A stop string ends the request early and cancels it in the engine."""
            while True:
                token, digest = self._next_event(ticket) or (None, None)
                if token is None:
                    break
                ticket.tokens.append(int(token))
                ticket.t_last = time.time()
                if digest is not None:
                    ticket.digests.append({"i": len(ticket.tokens) - 1, "token": int(token),
                                           "d": digest})
                delta = text.push(int(token))
                on_token(int(token), digest, delta)
                if text.stopped:
                    service.cancel(ticket, client=False)   # a stop string, not a disconnect
                    return "stop"
            seq = ticket.seq
            reason = getattr(seq, "finish", "") or "stop"
            if reason == "error":
                ticket.error = getattr(seq, "error", "") or service.fatal or "request failed"
            return reason

        def _complete(self, ticket: Ticket):
            text = _TextStream(_Detokenizer(service.decode, service.eos_ids), ticket.request.stop)
            try:
                reason = self._run(ticket, text, lambda t, d, s: None)
            except ConnectionAbortedError:
                service.cancel(ticket)
                return None
            tail = text.finish()
            del tail
            ticket.finish = reason
            mm = _mm_info(service, ticket.seq)
            if reason == "error":
                with service._counter_lock:
                    service.failed_requests += 1
                status = 503 if service.fatal else 400
                payload = {"error": {"message": ticket.error, "type": (
                    "server_error" if service.fatal else "request_error"), "code": status}}
                if mm:
                    payload["glc_mm"] = mm
                return self._json(status, payload)
            with service._counter_lock:
                service.served += 1
            n = len(ticket.tokens)
            response = {
                "id": "chatcmpl-" + ticket.seq.rid[:24], "object": "chat.completion",
                "created": int(ticket.t_submit), "model": service.model_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text.emitted},
                             "finish_reason": "length" if reason == "length" else "stop"}],
                "usage": {"prompt_tokens": ticket.prompt_tokens, "completion_tokens": n,
                          "total_tokens": ticket.prompt_tokens + n},
                "timings": _timings(ticket),
            }
            if service.limits.exact_digest:
                response["glc_exact"] = list(ticket.digests)
            if mm:
                response["glc_mm"] = mm
            return self._json(200, response)

        def _stream(self, ticket: Ticket):
            cid = "chatcmpl-" + ticket.seq.rid[:24]
            base = {"id": cid, "object": "chat.completion.chunk", "created": int(ticket.t_submit),
                    "model": service.model_id}

            def sse(obj: Any) -> None:
                payload = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()

            def chunk(delta: Dict[str, Any], finish=None, **extra) -> Dict[str, Any]:
                return {**base, "choices": [{"index": 0, "delta": delta,
                                             "finish_reason": finish}], **extra}

            text = _TextStream(_Detokenizer(service.decode, service.eos_ids), ticket.request.stop)

            def on_token(token: int, digest: Optional[str], delta: str) -> None:
                if digest is not None:
                    sse(chunk({}, glc_exact={"i": len(ticket.tokens) - 1, "token": token,
                                             "d": digest}))
                if delta:
                    sse(chunk({"content": delta}))

            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                sse(chunk({"role": "assistant", "content": ""}))
                reason = self._run(ticket, text, on_token)
                tail = text.finish()
                if tail:
                    sse(chunk({"content": tail}))
                ticket.finish = reason
                n = len(ticket.tokens)
                usage = {"prompt_tokens": ticket.prompt_tokens, "completion_tokens": n,
                         "total_tokens": ticket.prompt_tokens + n}
                extra: Dict[str, Any] = {"timings": _timings(ticket)}
                mm = _mm_info(service, ticket.seq)
                if mm:
                    extra["glc_mm"] = mm
                if reason == "error":
                    with service._counter_lock:
                        service.failed_requests += 1
                    status = 503 if service.fatal else 400
                    extra["error"] = {"message": ticket.error, "code": status,
                                      "type": "server_error" if service.fatal else "request_error"}
                    sse(chunk({}, "error", usage=usage, **extra))
                else:
                    with service._counter_lock:
                        service.served += 1
                    finish = "length" if reason == "length" else "stop"
                    if ticket.request.include_usage:
                        sse(chunk({}, finish))
                        sse({**base, "choices": [], "usage": usage, **extra})
                    else:
                        sse(chunk({}, finish, usage=usage, **extra))
                sse("[DONE]")
            except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError, OSError):
                # Client gone: free its slot and KV pages instead of generating for nobody.
                if not getattr(ticket.seq, "done", False):
                    service.cancel(ticket)
                self.close_connection = True
            return None

    return Handler


def make_server(service: BatchService, host: str, port: int, *,
                max_body_bytes: int = 32 << 20) -> Server:
    httpd = Server((host, int(port)), make_handler(service))
    httpd.max_body_bytes = int(max_body_bytes)
    return httpd


# =============================================================================== startup
def _read_api_keys(path: Optional[str]) -> List[str]:
    if not path:
        return []
    keys = [line.strip() for line in Path(path).expanduser().read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")]
    if not keys:
        raise ValueError(f"--api-key-file {path} contains no keys")
    return keys


def _loader_accepts(fn: Callable[..., Any], names: Sequence[str]) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return all(name in params for name in names)


def transcode_cache_loader_kwargs(args, loader_fn: Callable[..., Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Transcode-cache loader keywords for ``--package`` and a receipt note.

    The keywords come from ``transcode_cache.kwargs_from_args``. If this build's loader does
    not take them, an explicit ``--transcode-cache DIR`` is refused; the default is dropped and
    recorded as unsupported.
    """
    kwargs = transcode_cache.kwargs_from_args(args)
    if _loader_accepts(loader_fn, list(kwargs)):
        return kwargs, {"requested": kwargs, "supported_by_loader": True}
    if transcode_cache.explicit_cache_dir(args):
        raise ValueError("--transcode-cache was given but this runtime's Context loader does "
                         "not support a transcode cache")
    return {}, {"requested": kwargs, "supported_by_loader": False}


def _validate_tbe21_server_selection(args) -> None:
    """Validate the explicit TBE21 source route before kernel/model startup."""
    from .context_source import validate_context_startup_selection

    from .tbe21_v2_backend import validate_overlap_selection
    validate_overlap_selection(args)
    startup_workers = getattr(args, "context_startup_workers", 0)
    validate_context_startup_selection(workers=startup_workers,
        ram_bytes=getattr(args, "context_startup_ram_bytes", None),
        build_root=getattr(args, "context_build_root", None),
        context_typed=getattr(args, "context_typed", False), weight_format=args.runtime_format)
    if args.runtime_format != "tbe21":
        if (getattr(args, "context_typed", False)
                or getattr(args, "context_manifest_sha256", None) is not None
                or getattr(args, "context_build_root", None)
                or getattr(args, "parent", None)):
            raise ValueError("TBE21 Context options require --runtime-format tbe21")
        return
    if not args.package:
        raise ValueError("--runtime-format tbe21 requires --package")
    active_cache = transcode_cache.kwargs_from_args(args).get("transcode_cache")
    from .context_source import validate_context_selection, validate_typed_context_selection

    validate_typed_context_selection(
        context_typed=getattr(args, "context_typed", False),
        context_manifest_sha256=getattr(args, "context_manifest_sha256", None),
        build_root=getattr(args, "context_build_root", None), cache_root=active_cache,
        bundle=args.package,
        parent=getattr(args, "parent", None), gguf=None, weight_format="tbe21")
    validate_context_selection(
        build_root=getattr(args, "context_build_root", None), bundle=args.package,
        parent=getattr(args, "parent", None), gguf=None, weight_format="tbe21")
    unsupported = (
        ("verified_cpu_stream", "--verified-cpu-stream"),
        ("cpu_vision_dense", "--cpu-vision-dense"),
        ("cpu_encode_pipeline", "--cpu-encode-pipeline"),
        ("reuse_bhe_validation", "--reuse-bhe-validation"),
        ("reuse_package_validation", "--reuse-package-validation"),
        ("cpu_ready_install", "--cpu-ready-install"),
        ("prefetch_mtp", "--prefetch-mtp"),
        ("skip_single_session_capture", "--skip-single-session-capture"),
    )
    selected = [flag for name, flag in unsupported if getattr(args, name, False)]
    if selected:
        raise ValueError(f"TBE21 loader does not support {', '.join(selected)}")


def _tbe21_load_model_kwargs(args, tune_path: str, *, log: Callable[[str], None],
                            before_tbe21_device_use: Optional[Callable[[], None]] = None) -> Dict[str, Any]:
    """Translate the batch server's TBE21 selection to the public bidec loader API."""
    kwargs = {
        "bundle": args.package,
        "parent": getattr(args, "parent", None),
        "tune": tune_path,
        "device": args.device,
        "spec": bool(args.spec_k and not args.spec_ngram),
        "weight_format": "tbe21",
        "log": log,
        "embed_placement": "device",
        "vision_placement": "device" if args.enable_images else "off",
        "mtp_placement": "device",
        "mm_options": {"max_images": args.mm_max_images,
                       "max_image_bytes": args.mm_max_image_bytes,
                       "max_image_tokens": args.mm_max_image_tokens},
        "context_typed": getattr(args, "context_typed", False),
        "context_manifest_sha256": getattr(args, "context_manifest_sha256", None),
        "context_build_root": getattr(args, "context_build_root", None),
        "vision_from_package": bool(args.enable_images),
        "gdn_hv16": bool(getattr(args, "gdn_hv16", False)),
    }
    if getattr(args, "context_startup_workers", 0):
        kwargs.update(context_startup_workers=args.context_startup_workers,
                      context_startup_ram_bytes=args.context_startup_ram_bytes)
    if before_tbe21_device_use is not None:
        kwargs["before_tbe21_device_use"] = before_tbe21_device_use
    return kwargs


def _tbe21_source_kind(args) -> str:
    if getattr(args, "context_typed", False):
        return "typed_context_tbe21"
    if getattr(args, "context_build_root", None):
        return "context_v21_transcode"
    return "tbe21_fullpack"


def _tbe21_package_visual(model: Any) -> Any:
    """Find the dense vision module installed from the verified TBE21 package."""
    get_submodule = getattr(model, "get_submodule", None)
    if callable(get_submodule):
        for path in ("model.visual", "visual"):
            try:
                visual = get_submodule(path)
            except (AttributeError, KeyError):
                continue
            if visual is not None:
                return visual
    return None


def _load_tbe21_fast_session(args, *, log: Callable[[str], None],
                             before_tbe21_device_use: Optional[Callable[[], None]] = None) -> Any:
    """Adapt the completed bidec loader result to the shared FastDecoder batch envelope."""
    _validate_tbe21_server_selection(args)
    from .bidec_load import load_model, stop_ids_for
    from .fastserve import _resolve_tune

    tune_path = _resolve_tune(args.tune, args.device)
    loaded = load_model(**_tbe21_load_model_kwargs(
        args, tune_path, log=log, before_tbe21_device_use=before_tbe21_device_use))
    model, tok, fd = loaded["model"], loaded["tok"], loaded["fd"]
    rec = dict(loaded["rec"])
    manifest_sha256 = rec.get("manifest_sha256")
    if (not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64
            or any(char not in "0123456789abcdef" for char in manifest_sha256)):
        raise RuntimeError("TBE21 loader receipt lacks the opened package manifest SHA-256")
    if (getattr(args, "context_typed", False)
            and manifest_sha256 != args.context_manifest_sha256):
        raise RuntimeError("typed Context manifest pin differs from the loaded package receipt")
    local_dir = rec.get("config")
    if args.enable_images:
        if rec.get("vision_source") != "tbe21_package":
            raise RuntimeError("TBE21 image admission requires package-resident vision tensors")
        if _tbe21_package_visual(model) is None:
            raise RuntimeError("TBE21 package vision receipt has no installed visual module")
        processor = loaded.get("processor")
        if getattr(processor, "image_processor", None) is None:
            raise RuntimeError("TBE21 sealed package metadata has no image processor")
        vision_source = "tbe21_package"
    else:
        processor = None
        vision_source = None
    config = loaded.get("config", getattr(model, "config", None))
    if args.enable_images and (config is None or getattr(config, "vision_config", None) is None):
        raise RuntimeError("TBE21 package model has no vision config")
    loaded_view = SimpleNamespace(model=model, mtp=getattr(fd, "mtp", None), config=config,
                                  local_dir=local_dir)
    engine = SimpleNamespace(model=model, loaded=loaded_view, processor=processor,
                             tok=tok, tokenizer=tok, device=args.device)
    meta = {
        "source_kind": _tbe21_source_kind(args),
        "source": str(args.package), "source_dir": local_dir,
        "runtime_format": "tbe21", "weight_format": "tbe21",
        "tune_path": tune_path,
        "tune_sha256": hashlib.sha256(Path(tune_path).read_bytes()).hexdigest(),
        "artifact_manifest_sha256": manifest_sha256,
        "tbe21_load_receipt": rec,
    }
    if vision_source is not None:
        meta["vision_source"] = vision_source
    if getattr(args, "context_typed", False):
        meta["context_manifest_sha256"] = args.context_manifest_sha256
    return SimpleNamespace(
        engine=engine, fd=fd, model=model, tokenizer=tok, meta=meta,
        eos_ids=stop_ids_for(tok, local_dir, loaded.get("generation_config")),
        mtp=getattr(fd, "mtp", None))



def _load_tbe21_session_with_overlap(args, *, log):
    if not getattr(args, "context_overlap_tbe21_build", False):
        return _load_tbe21_fast_session(args, log=log)
    from . import tbe21_v2_backend as backend
    build = backend.start_overlap(args)
    try:
        session = _load_tbe21_fast_session(args, log=log,
                                           before_tbe21_device_use=build.result)
        build.result()
        receipt = dict(build.receipt)
        receipt["artifact_manifest_sha256"] = session.meta["artifact_manifest_sha256"]
        receipt["source_kind"] = session.meta["source_kind"]
        session.meta["tbe21_load_receipt"]["overlap_tbe21_build"] = receipt
        return session
    except BaseException as exc:
        try:
            build.cancel()
        except BaseException as cleanup:
            backend.record_cleanup_failure(exc, cleanup)
        raise

def load_runtime(args, settings: Dict[str, Any], *, service: BatchService,
                 log: Callable[[str], None]) -> Any:
    """Default runtime factory: session load -> page sizing -> batch engine build."""
    from . import predictive_batch as pb
    from .context_fast_server import _configure_exact_runtime, _stream_options

    if args.runtime_format == "tbe21":
        _validate_tbe21_server_selection(args)
    _configure_exact_runtime()
    from .kernel_cache import ensure_serving_kernels

    defer_bhe_kernel_prewarm = (
        args.package and args.runtime_format == "bhe1"
        and args.verified_cpu_stream and args.cpu_encode_pipeline
        and str(args.device).lower().split(":", 1)[0] == "cuda"
    )
    defer_overlap_kernel_prewarm = (
        args.package and args.runtime_format == "tbe21"
        and getattr(args, "context_overlap_tbe21_build", False)
    )
    if (args.package and args.runtime_format in ("bhe1", "tbe21")
            and not defer_bhe_kernel_prewarm and not defer_overlap_kernel_prewarm):
        # These descriptors use BI-GEMM, not the TBE-v1 MMA or MIV-TBE
        # decoder. Keep their generic batch/FastDecoder dependencies prewarmed,
        # except when the opted-in BHE pipeline or TBE21 overlap child builds them
        # alongside the corresponding CPU startup work.
        ensure_serving_kernels("batch", device=args.device,
                               only=("miv_gemv", "fastdec", "bidec", "bi_gemm"), log=log)
    elif not defer_bhe_kernel_prewarm and not defer_overlap_kernel_prewarm:
        ensure_serving_kernels("batch", device=args.device, log=log)
    stream_options = _stream_options(args)
    # Batch forwards this loader-only option through loader_kwargs below.
    stream_options.pop('reuse_package_validation', None)
    spec_options = {"spec": True} if args.spec_k and not args.spec_ngram else {}
    timings = service.startup_timings
    t0 = time.perf_counter()
    if args.package and args.runtime_format == "tbe21":
        if getattr(args, "context_overlap_tbe21_build", False):
            session = _load_tbe21_session_with_overlap(args, log=log)
        else:
            session = _load_tbe21_fast_session(args, log=log)
        source_kind = _tbe21_source_kind(args)
    elif args.package:
        from .predictive_tbe import load_predictive_fast_session

        cache_kwargs, cache_note = transcode_cache_loader_kwargs(args, load_predictive_fast_session)
        cache_kwargs.update(transcode_cache.package_verify_kwargs(args))
        if args.runtime_format != 'tbe':
            cache_kwargs['runtime_format'] = args.runtime_format
        if getattr(args, 'reuse_bhe_validation', False):
            cache_kwargs['reuse_bhe_validation'] = True
        if getattr(args, 'reuse_package_validation', False):
            cache_kwargs['reuse_package_validation'] = True
        if getattr(args, 'cpu_ready_install', False):
            cache_kwargs['cpu_ready_install'] = True
        if getattr(args, 'prefetch_mtp', False):
            cache_kwargs['prefetch_mtp'] = True
        if getattr(args, 'skip_single_session_capture', False):
            cache_kwargs['capture_single_session'] = False
        log(transcode_cache.describe(args))
        service.settings["transcode_cache"] = cache_note
        session = pb.load_predictive_session(
            package=args.package, tune=args.tune, device=args.device,
            max_ctx=settings["max_ctx"], loader_workers=args.loader_workers,
            attention_implementation="sdpa", log=log, loader_kwargs=cache_kwargs,
            gdn_hv16=bool(getattr(args, "gdn_hv16", False)),
            **stream_options, **spec_options)
        source_kind = "predictive_tbe"
    else:
        session = pb.load_dense_session(dense=args.dense, tune=args.tune, device=args.device,
                                        max_ctx=settings["max_ctx"],
                                        attention_implementation="sdpa", log=log,
                                        gdn_hv16=bool(getattr(args, "gdn_hv16", False)), **spec_options)
        source_kind = "dense_bf16"
    timings["session_load_s"] = round(time.perf_counter() - t0, 3)
    if getattr(args, "memory_allocator_attribution", False):
        service.record_allocator_phase("weights_loaded", session)
    t1 = time.perf_counter()
    settings, plan = pb.resolve_pages_and_settings(
        session, settings, device=args.device, enable_images=bool(args.enable_images),
        headroom_bytes=int(round(float(args.vram_headroom_gib) * (1 << 30))), log=log)
    timings["page_plan_s"] = round(time.perf_counter() - t1, 3)
    if getattr(args, "memory_allocator_attribution", False):
        service.record_allocator_phase("page_plan_complete", session)
    service.settings.update(settings)
    if plan is not None:
        _emit_line({"status": "loading", "phase": "batch_build", "pages_total": plan["pages_total"],
                    "pages_fit": plan["pages_fit"], "per_page_bytes": plan["per_page_bytes"],
                    "available_bytes": plan["available_bytes"]})
    t2 = time.perf_counter()
    mm_options = {"max_images": args.mm_max_images, "max_image_bytes": args.mm_max_image_bytes,
                  "max_image_tokens": args.mm_max_image_tokens}
    runtime = pb.build_native_batch_from_fast_session(
        session, source_kind=source_kind, attention_implementation="sdpa", settings=settings,
        device=args.device, enable_images=bool(args.enable_images), log=log,
        mm_options=mm_options, page_plan=plan,
        **({"gdn_hv16": True} if bool(getattr(args, "gdn_hv16", False)) else {}),
        **({"vision_source": session.meta.get("vision_source")}
           if args.runtime_format == "tbe21" and args.enable_images else {}))
    if args.runtime_format == "tbe21":
        if runtime.metadata.get("source_kind") != source_kind:
            raise RuntimeError("TBE21 batch envelope lost its source kind")
        if runtime.metadata.get("weight_format") != "tbe21":
            raise RuntimeError("TBE21 batch envelope lost its runtime format")
    timings["batch_build_and_capture_s"] = round(time.perf_counter() - t2, 3)
    if getattr(args, "memory_allocator_attribution", False):
        service.record_allocator_phase("batch_capture_complete", runtime)
    return runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="georefine-context-batch-serve", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--package", help="Context package directory (the release serving path)")
    source.add_argument("--dense", help="BF16 parent checkpoint served through the same engine "
                                        "(labelled dense_bf16)")
    parser.add_argument("--tune", default=None,
                        help="FastDecoder tune JSON. Omitted: the bundled "
                             "glc_serve/tunes/fastdec_sm120.json on the RTX PRO 6000 it was "
                             "measured on; any other GPU must pass --tune")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address; a non-loopback host requires --api-key-file")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--api-key-file", default=None,
                        help="file with one API key per line; requests then need "
                             "'Authorization: Bearer KEY' (/health stays open)")
    batch = parser.add_argument_group("batch engine")
    batch.add_argument("--optimise", action=argparse.BooleanOptionalAction, default=False,
                       help="enable the source-verified native engine profile: workload graph "
                            "capture, full admitted decode rows, and paired-scan TBE21 kernels")
    batch.add_argument("--optimisation-kernel", choices=("baseline", "quad", "pair-vote", *CLI_PROFILES), default="baseline",
                       help="explicit experimental TBE21 kernel; requires --optimise and a TBE21 package. Tier profiles require full enabled-geometry G1")
    batch.add_argument("--spec-k", type=int, default=0,
                       help="maximum speculative draft depth, 0 disables speculation (0..7)")
    batch.add_argument("--spec-ngram", action="store_true",
                       help="use prompt/history n-gram drafting instead of the MTP head")
    batch.add_argument("--mtp-confirm-kv-only", action="store_true",
                       help="experimental: preserve accepted-prefix MTP KV and evaluate only final head rows")
    batch.add_argument("--max-decode-rows", type=int, default=None,
                       help="decode/verify row budget (default: full row capacity with speculation, otherwise 64)")
    batch.add_argument("--max-users", type=int, default=64,
                       help="concurrent sequences in the batch (engine slots)")
    batch.add_argument("--adaptive-serving", choices=("capacity", "prefill"), default=None,
                       help="opt in to memory-ledger admission: max-users/max-rows become upper "
                            "bounds; preserve full context and precision. capacity favors users; "
                            "prefill favors row capacity. Requires automatic pages and CUDA")
    batch.add_argument("--adaptive-min-users", type=int, default=1,
                       help="minimum full-context users admitted by --adaptive-serving")
    batch.add_argument("--max-context", type=int, default=16384,
                       help="per-request prompt + generation capacity in tokens")
    batch.add_argument("--max-rows", type=int, default=None,
                       help="row capacity of one engine step (a bidec bucket)")
    batch.add_argument("--max-rows-step", type=int, default=None,
                       help="rows actually scheduled per step (default --max-rows)")
    batch.add_argument("--capture-workload-only", action="store_true",
                       help="opt in to eager graph capture through the workload row ceiling; "
                            "larger buckets use one live eager body")
    batch.add_argument("--capture-workload-ceiling", type=int, default=None,
                       help="row ceiling for --capture-workload-only (default: "
                            "max-users * (spec-k + 1))")
    batch.add_argument("--spec-calibration-repeats", type=int, default=None,
                       help="CUDA-event samples per speculative cost calibration (3..20; default 5)")
    batch.add_argument("--prefill-chunk", type=int, default=256)
    batch.add_argument("--prefill-priority", type=float, default=0.0)
    batch.add_argument("--pages-total", type=int, default=0,
                       help="KV pages of 256 tokens; 0 (default) sizes the pool from free "
                            "device memory after the weights load")
    batch.add_argument("--vram-headroom-gib", type=float, default=6.0,
                       help="device memory left free by automatic page sizing")
    batch.add_argument("--gdn-hv16", action="store_true",
                       help="opt in to the Qwen0.8B GDN geometry (16 value heads, conv4)")
    limits = parser.add_argument_group("request limits")
    limits.add_argument("--max-prompt", type=int, default=12288)
    limits.add_argument("--max-new", type=int, default=2048,
                        help="upper bound on max_tokens; larger requests are clamped")
    limits.add_argument("--default-max-new", type=int, default=1024)
    limits.add_argument("--max-queue", type=int, default=256,
                        help="waiting requests before new ones get 429")
    limits.add_argument("--max-body-bytes", type=int, default=32 << 20)
    limits.add_argument("--exact-digest", action="store_true",
                        help="per-token logits-row digests in responses (glc_exact)")
    images = parser.add_argument_group("images")
    images.add_argument("--enable-images", action="store_true")
    images.add_argument("--mm-max-images", type=int, default=4)
    images.add_argument("--mm-max-image-bytes", type=int, default=20 << 20)
    images.add_argument("--mm-max-image-tokens", type=int, default=16384)
    loader = parser.add_argument_group("Context loader")
    loader.add_argument('--runtime-format', choices=('tbe', 'bhe1', 'tbe21'), default='tbe',
                        help='resident compressed backend; BHE1 is experimental, TBE21 is explicit')
    loader.add_argument('--parent', default=None,
                        help='explicit config/tokenizer and optional vision source for TBE21')
    loader.add_argument('--context-typed', action='store_true',
                        help='interpret --package as a typed Context package (TBE21 only)')
    loader.add_argument('--context-manifest-sha256', default=None,
                        help='required exact manifest pin with --context-typed')
    loader.add_argument('--context-build-root', default=None,
                        help='durable .scratch build root for the frozen Context-to-TBE21 bridge')
    loader.add_argument('--context-startup-workers', type=int, default=0,
                        help='opt-in spawned CPU workers for frozen Context startup (1..12)')
    loader.add_argument('--context-startup-ram-bytes', type=int, default=None,
                        help='required explicit RAM estimate budget for parallel Context startup')
    loader.add_argument("--context-overlap-tbe21-build", action="store_true",
                        help="overlap the unchanged TBE21 v2 build with Context CPU startup (Linux, opt-in)")
    loader.add_argument("--loader-workers", type=int, default=4)
    loader.add_argument("--verified-cpu-stream", action="store_true")
    loader.add_argument("--cpu-encode-workers", type=int,
                        choices=range(1, MAX_CPU_ENCODE_WORKERS + 1), default=1)
    loader.add_argument("--cpu-encode-pipeline", action="store_true",
                        help="use the ordered bounded CPU encoder pipeline (requires verified stream and >1 worker)")
    loader.add_argument("--reuse-bhe-validation", action="store_true",
                        help="reuse completed source validation for the same immutable BHE panels")
    loader.add_argument("--reuse-package-validation", action="store_true",
                        help="reuse the live full-integrity resolver for middle package validation")
    loader.add_argument("--cpu-ready-install", action="store_true",
                        help="install completed BHE CPU tasks while earlier tasks finish")
    loader.add_argument("--prefetch-mtp", action="store_true",
                        help="load verified MTP weights during the BHE CPU pipeline")
    loader.add_argument("--skip-single-session-capture", action="store_true",
                        help="use the batch engine's full graph capture without single-session graphs")
    loader.add_argument("--source-inflight-bytes", type=int, default=8 * 1024**3)
    loader.add_argument("--cpu-vision-dense", action="store_true")
    transcode_cache.add_cli_args(parser)
    parser.add_argument("--startup-failure-grace-s", type=float, default=10.0,
                        help="seconds /health keeps reporting startup_failed before exit 2")
    parser.add_argument("--memory-metrics", action="store_true",
                        help="enable bounded current-process memory monitoring and /v1/memory")
    parser.add_argument("--memory-allocator-attribution", action="store_true",
                        help="add allocator counters and load/capture phase snapshots for the resident "
                             "CUDA serving device; requires --memory-metrics")
    parser.add_argument("--memory-interval-s", type=float, default=0.1,
                        help="memory sample interval in seconds (finite and positive)")
    parser.add_argument("--qualification-protocol", default=None,
                        help="opt in to protocol-bound serving identity")
    parser.add_argument("--qualification-role", choices=("candidate", "reference"), default=None)
    parser.add_argument("--qualification-wheel", default=None,
                        help="wheel to verify against installed runtime source")
    return parser


def _enable_bound_resident_startup(service, qualification_identity):
    """Use only metadata from bind_protocol; never reopen a protocol after loading."""
    if 'resident_startup_selection' in qualification_identity:
        service._resident_startup_enabled = True


def _freeze_adaptive_qualification_settings(settings, identity):
    """Handoff only the already wheel/source-verified identity before model load."""
    import copy
    frozen = copy.deepcopy(identity['adaptive_serving'])
    settings['adaptive_expectation'] = frozen
    settings['adaptive_trusted_sources'] = dict(frozen['source_bindings'])


def _bind_qualification(args, *, max_requests: int) -> Optional[Dict[str, Any]]:
    if not args.qualification_protocol:
        return None
    from .qualification_identity import bind_protocol, tbe21_source_binding
    operator=args.runtime_format if args.package else "dense"
    startup_workers=getattr(args, "context_startup_workers", 0)
    parallel_startup=({"workers": startup_workers,
                       "ram_budget_bytes": args.context_startup_ram_bytes}
                      if startup_workers else None)
    tbe21_source=(tbe21_source_binding(package=args.package,parent=args.parent,
        context_typed=bool(getattr(args,'context_typed',False)),
        context_manifest_sha256=getattr(args,'context_manifest_sha256',None),
        context_build_root=getattr(args,'context_build_root',None),
        backend_expectation='tbe21-v2', parallel_startup=parallel_startup,
        overlap_tbe21_build=getattr(args, 'context_overlap_tbe21_build', False))
        if operator=='tbe21' else None)
    identity = bind_protocol(
        args.qualification_protocol, role=args.qualification_role,
        package=args.package or args.dense, wheel=args.qualification_wheel,
        exact_digest=bool(args.exact_digest),
        max_requests=max_requests, device=args.device, backend="cuda", spec_k=args.spec_k,
        fused_batching=True, max_batch_rows=args.max_rows, panels=None,
        metal_operator=operator,tune=args.tune,tbe21_source=tbe21_source,
        mtp_confirm_kv_only=getattr(args, 'mtp_confirm_kv_only', False),
        **({'adaptive_serving': args.adaptive_serving} if getattr(args, 'adaptive_serving', None) else {}))

    if 'adaptive_serving' in identity:
        from .qualification_identity import validate_adaptive_worklist_launch
        validate_adaptive_worklist_launch(identity['adaptive_serving'], objective=args.adaptive_serving,
            max_users=args.max_users, max_rows=args.max_rows, max_ctx=args.max_context,
            spec_k=args.spec_k, prefill_priority=args.prefill_priority,
            reserve_bytes=int(round(args.vram_headroom_gib * (1 << 30))))
    return identity


def _reject_unbound_runtime_compact_qualification(args, *, environ=None) -> None:
    """Reject ambient or forged compact selection before starting native work."""
    if not args.qualification_protocol:
        return
    from .qualification_identity import validate_cpu_decoder_environment, validate_cpu_encoder_environment, installed_cpu_encoder
    env = os.environ if environ is None else environ
    if args.qualification_role == "reference":
        # TBE21 startup is candidate-only. Keep reference argument validation
        # independent of protocol I/O, while rejecting ambient native selectors.
        # bind_protocol still validates the actual protocol before model loading.
        validate_cpu_decoder_environment({}, env)
        validate_cpu_encoder_environment({}, env)
        return
    protocol = json.loads(Path(args.qualification_protocol).read_text())
    phase = "exactness" if args.exact_digest else "performance"
    identity = protocol["engines"][args.qualification_role][phase]["expected_identity"]
    source = identity.get("tbe21_source", {})
    validate_cpu_decoder_environment(source, env)
    encoder = validate_cpu_encoder_environment(source, env)
    if encoder is not None and encoder != installed_cpu_encoder():
        raise ValueError("cpu_encoder differs from installed source manifest")


def _cuda_accelerator(device: str) -> Dict[str, Any]:
    import torch
    props = torch.cuda.get_device_properties(device)
    return {"name": str(props.name), "total_memory_bytes": int(props.total_memory),
            "compute_capability": [int(props.major), int(props.minor)]}


def _verify_qualification_runtime(identity: Dict[str, Any], runtime: Any, args) -> None:
    batcher = runtime.batcher
    if runtime.bd.max_slots<max(identity['concurrency']):
        raise ValueError('qualification runtime lacks required concurrent slots')
    for key, actual in (("spec_k", getattr(batcher, "spec_k", None)),
                        ("max_batch_rows", getattr(runtime.bd, "max_rows", None))):
        if actual != identity[key]:
            raise ValueError(f"qualification runtime drift: {key}={actual!r}, expected {identity[key]!r}")
    metadata = dict(getattr(runtime, "metadata", {}) or {})
    if identity['spec_k'] and metadata.get('mtp_enabled') is not True:
        raise ValueError('qualification runtime did not load the MTP head')
    source_session = metadata.get("source_session") or {}
    fmt = source_session.get("runtime_format") or metadata.get("runtime_format")
    if fmt is None:
        fmt={'tbe-v1-predictive-startup':'tbe','bhe1-predictive-startup':'bhe1','bf16-dense':'dense'}.get(metadata.get('weight_format'))
    expected = args.runtime_format if args.package else "dense"
    if args.package:
        if fmt != expected:
            raise ValueError(f"qualification runtime drift: runtime_format={fmt!r}, expected {expected!r}")
        if args.runtime_format == "tbe21":
            opened_manifest = source_session.get("artifact_manifest_sha256")
            if opened_manifest != identity.get("package_manifest_sha256"):
                raise ValueError("qualification runtime drift: TBE21 loaded package manifest differs")
            if (getattr(args, "context_typed", False)
                    and source_session.get("context_manifest_sha256") != opened_manifest):
                raise ValueError("qualification runtime drift: typed Context manifest pin differs")
    else:
        weight_format = metadata.get("weight_format")
        if weight_format not in ("bf16-dense", "dense"):
            raise ValueError(f"qualification runtime drift: dense source format={weight_format!r}")
    from .qualification_identity import bind_protocol, tbe21_source_binding, validate_tbe21_runtime_metadata
    startup_workers=getattr(args, "context_startup_workers", 0)
    parallel_startup=({"workers": startup_workers,
                       "ram_budget_bytes": args.context_startup_ram_bytes}
                      if startup_workers else None)
    tbe21_source=(tbe21_source_binding(package=args.package,parent=args.parent,
        context_typed=bool(getattr(args,'context_typed',False)),
        context_manifest_sha256=getattr(args,'context_manifest_sha256',None),
        context_build_root=getattr(args,'context_build_root',None),
        backend_expectation='tbe21-v2', parallel_startup=parallel_startup,
        overlap_tbe21_build=getattr(args, 'context_overlap_tbe21_build', False))
        if args.package and fmt=='tbe21' else None)
    current = bind_protocol(
        args.qualification_protocol, role=args.qualification_role,
        package=args.package or args.dense, wheel=args.qualification_wheel,
        max_requests=runtime.bd.max_slots, device=args.device,exact_digest=bool(args.exact_digest),
        backend="cuda", spec_k=int(batcher.spec_k), fused_batching=True,
        max_batch_rows=int(runtime.bd.max_rows), panels=None,
        metal_operator=(fmt if args.package else "dense"),tune=args.tune,
        tbe21_source=tbe21_source,
        mtp_confirm_kv_only=getattr(batcher, 'mtp_confirm_kv_only', False),
        **({'adaptive_serving': args.adaptive_serving} if getattr(args, 'adaptive_serving', None) else {}))
    if current != identity:
        raise ValueError("qualification identity changed during runtime load")
    if 'adaptive_serving' in identity or metadata.get('page_plan', {}).get('mode') == 'adaptive':
        from .qualification_identity import (validate_adaptive_worklist_metadata,
            adaptive_worklist_sources_from_manifest, MANIFEST)
        manifest = json.loads((Path(__file__).resolve().parents[1] / MANIFEST).read_text())
        trusted = adaptive_worklist_sources_from_manifest(manifest)
        if batcher.prefill_priority != metadata.get('prefill_priority'):
            raise ValueError('adaptive work-list actual prefill priority differs')
        validate_adaptive_worklist_metadata(metadata, identity.get('adaptive_serving'), trusted,
                                            actual=runtime.bd)
    if expected=='tbe21':
        identity['compressed_descriptor_count']=validate_tbe21_runtime_metadata(metadata,
            identity.get('tbe21_source'))
        from .qualification_identity import validate_tbe21_parallel_startup_metadata
        startup_metrics=validate_tbe21_parallel_startup_metadata(
            source_session, identity.get('tbe21_source'))
        if startup_metrics is not None:
            identity['parallel_context_startup']=startup_metrics
    if expected=='bhe1':
        def count(desc):
            if getattr(desc,'kind',None)=='bhe1':
                return 1
            if hasattr(desc,'members'):
                return sum(count(member) for member,_ in desc.members)
            if hasattr(desc,'d'):
                return count(desc.d)
            return 0
        descriptors=runtime.bd.fd._descs()
        identity['compressed_descriptor_count']=sum(count(d) for d in descriptors)
        if identity['compressed_descriptor_count']<=0:
            raise ValueError('BHE1 qualification runtime has no BHE1 descriptors')
    identity["accelerator"] = _cuda_accelerator(args.device)
    if expected=='tbe21':
        from .qualification_identity import source_parameter_count
        identity['served_parameter_count']=source_parameter_count(args.package)
    else:
        source_bytes=source_session.get('runtime_representation',{}).get('source_model_bf16_bytes')
        identity['served_parameter_count']=(source_bytes//2 if type(source_bytes) is int else None)
        if identity['served_parameter_count'] is None:
            model=getattr(getattr(runtime,'session',None),'model',None)
            if model is not None:
                identity['served_parameter_count']=sum(p.numel() for p in model.parameters())


def _apply_optimisation(args, *, environ=None) -> None:
    """Select the declared native kernels and existing batch-graph machinery.

    This changes execution selection, never numerical or qualification thresholds.
    PIPE128 is deliberately excluded: its GPU screen regressed fused gate/up.
    """
    if not getattr(args, "optimise", False):
        if getattr(args, "optimisation_kernel", "baseline") != "baseline":
            raise ValueError("--optimisation-kernel requires --optimise")
        return
    env = os.environ if environ is None else environ
    kernel = getattr(args, "optimisation_kernel", "baseline")
    from . import tbe21_tier_profile as tier
    profile = tier.CLI_PROFILES.get(kernel)
    quad = kernel == "quad"
    pair_vote = kernel == "pair-vote"
    if (quad or pair_vote or profile is not None) and not args.package:
        raise ValueError(f"{kernel} optimisation kernel requires a TBE21 package")
    if args.package:
        if args.runtime_format != "tbe21":
            raise ValueError("--optimise with --package requires --runtime-format tbe21")
        controls = {
            "BI_TBE21_V2": "1", "BI_TBE21_KERNEL_VARIANT": "1",
            "BI_T21_PACKSCAN2": "1", "BI_T21_TILE128": "1",
            "BI_T21_TABREG": "1", "BI_T21_SCAN": "0",
            "BI_T21_V1_MINBLOCKS": "2", "BI_T21_ASYNC_X": "0",
            "BI_T21_ROW_GEOMETRY": "0", "BI_T21_SCAN4": "0",
            "BI_T21_WIDE128": "0", "BI_T21_PIPE128": "0",
            "BI_T21_PACKSCAN4": "0", "BI_T21_PAIR_VOTE": "0",
            "BI_T21_EXEC_LAYOUT": "0",
            **{name: "0" for name in tier.ENV_FIELDS},
        }
        if quad:
            controls.update(BI_T21_PACKSCAN4="1", BI_T21_TABREG="0")
        if pair_vote:
            controls.update(BI_T21_PAIR_VOTE="1", BI_T21_PACKSCAN4="0", BI_T21_TABREG="1")
        if profile is not None:
            row_geometry = env.get('BI_T21_ROW_GEOMETRY', '0')
            if row_geometry not in ('0', '1'):
                raise ValueError('BI_T21_ROW_GEOMETRY must be exactly 0 or 1')
            controls['BI_T21_ROW_GEOMETRY'] = row_geometry
            controls.update(dict(zip(tier.ENV_FIELDS, ('1', str(int(profile['prefetch'])),
                                                       str(int(profile['wg1'])), str(int(profile['prefetch_wg1']))))))
            controls['BIDEC_TBE2_G1'] = 'full'
        conflicts = [name for name, value in controls.items()
                     if name in env and env[name] != value]
        if conflicts:
            raise ValueError("--optimise conflicts with explicit controls: " + ", ".join(conflicts))
        env.update(controls)
    # BatchDecoder already captures and replays native graphs. Limit eager capture
    # to the declared workload instead of capturing unused buckets to 512 rows.
    args.capture_workload_only = True
    if args.max_rows is None:
        from .bidec import BUCKETS
        required = (max(BUCKETS) if getattr(args, 'adaptive_serving', None)
                    else args.max_users * (args.spec_k + 1))
        eligible = [rows for rows in BUCKETS if rows >= required]
        if not eligible:
            raise ValueError("--optimise workload exceeds supported graph row capacity")
        args.max_rows = eligible[0]
    if args.max_decode_rows is None and not args.spec_k:
        args.max_decode_rows = args.max_users


def validate_args(args) -> Dict[str, Any]:
    """Flag checks that need no model; returns batch settings (pages may be 0 = auto)."""
    from .context_fast_server import _stream_options
    from .predictive_batch import _validate_batch_settings

    _apply_optimisation(args)
    adaptive = getattr(args, 'adaptive_serving', None)
    if adaptive:
        if not str(args.device).startswith('cuda'):
            raise ValueError('--adaptive-serving currently requires CUDA native serving')
        if args.pages_total:
            raise ValueError('--adaptive-serving requires --pages-total 0')
        if getattr(args, 'qualification_protocol', None):
            try:
                protocol = json.loads(Path(args.qualification_protocol).read_text())
                phase = 'exactness' if args.exact_digest else 'performance'
                frozen = protocol['engines'][args.qualification_role][phase]['expected_identity']['adaptive_serving']
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ValueError('adaptive qualification requires a frozen resolved resource plan') from exc
            from .qualification_identity import validate_adaptive_worklist_expectation
            validate_adaptive_worklist_expectation(frozen, frozen.get('source_bindings') if isinstance(frozen, dict) else None)
        if not 1 <= args.adaptive_min_users <= args.max_users:
            raise ValueError('--adaptive-min-users must be within 1..max-users')
        if adaptive == 'prefill' and args.prefill_priority <= 0:
            raise ValueError('adaptive prefill requires positive --prefill-priority')
    elif getattr(args, 'adaptive_min_users', 1) != 1:
        raise ValueError('--adaptive-min-users requires --adaptive-serving')
    if getattr(args, "qualification_protocol", None):
        _reject_unbound_runtime_compact_qualification(args)
    _validate_tbe21_server_selection(args)
    from . import tbe21_v2_backend as tbe21_backend
    tbe21_backend.selected_row_geometry()
    tbe21_backend.selected_scan4()
    tbe21_backend.selected_wide128()
    tbe21_backend.selected_packed_scan4()
    if tbe21_backend.selected_tile128() and (not tbe21_backend.enabled()
            or not args.package or args.runtime_format != "tbe21"):
        raise ValueError("BI_T21_TILE128 requires BI_TBE21_V2=1 and package runtime-format=tbe21")
    if args.runtime_format == 'bhe1':
        if not args.package:
            raise ValueError('--runtime-format bhe1 requires --package')
        if transcode_cache.kwargs_from_args(args)['transcode_cache'] is not None or args.cpu_vision_dense:
            raise ValueError('BHE1 requires --transcode-cache off and no --cpu-vision-dense')
    if args.max_rows is None:
        from .bidec import BUCKETS
        required = (max(BUCKETS) if adaptive else
                    min(512, max(128, args.max_users * 2)) if args.spec_k else 128)
        args.max_rows = next(b for b in BUCKETS if b >= required)
    if args.max_decode_rows is None:
        args.max_decode_rows = args.max_rows if args.spec_k else 64
    if args.capture_workload_ceiling is not None and not args.capture_workload_only:
        raise ValueError("--capture-workload-ceiling requires --capture-workload-only")
    if args.spec_calibration_repeats is not None:
        if not args.spec_k or not 3 <= args.spec_calibration_repeats <= 20:
            raise ValueError("--spec-calibration-repeats requires speculation and a value in 3..20")
    if args.capture_workload_only:
        required_capture_rows = int(args.adaptive_min_users if adaptive else args.max_users) * (int(args.spec_k) + 1)
        default_ceiling = args.max_rows if adaptive else required_capture_rows
        capture_ceiling = (default_ceiling if args.capture_workload_ceiling is None
                           else int(args.capture_workload_ceiling))
        if adaptive:
            from .bidec import BUCKETS
            if capture_ceiling not in BUCKETS:
                raise ValueError('adaptive capture ceiling must be a supported native row bucket')
        if capture_ceiling < required_capture_rows:
            raise ValueError(
                "--capture-workload-ceiling must cover max-users * (spec-k + 1) "
                f"({required_capture_rows})")
        if capture_ceiling > args.max_rows:
            raise ValueError("--capture-workload-ceiling cannot exceed --max-rows")
    else:
        capture_ceiling = None
    if not 0 <= args.port <= 65535:
        raise ValueError("--port must be between 0 and 65535")
    if not args.model_id:
        raise ValueError("--model-id must be non-empty")
    if args.host not in LOOPBACK and not args.api_key_file:
        raise ValueError(f"--host {args.host} is not loopback; pass --api-key-file to expose it")
    if not 1 <= args.loader_workers <= 16:
        raise ValueError("--loader-workers must be in [1,16]")
    for name in ("max_prompt", "max_new", "default_max_new", "max_queue", "max_body_bytes"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.default_max_new > args.max_new:
        raise ValueError("--default-max-new cannot exceed --max-new")
    if args.pages_total < 0:
        raise ValueError("--pages-total must be >= 0 (0 = automatic)")
    if not math.isfinite(args.vram_headroom_gib) or args.vram_headroom_gib < 0:
        raise ValueError("--vram-headroom-gib must be a finite non-negative number")
    if args.startup_failure_grace_s < 0:
        raise ValueError("--startup-failure-grace-s must be >= 0")
    if not math.isfinite(args.memory_interval_s) or args.memory_interval_s <= 0:
        raise ValueError("--memory-interval-s must be finite and positive")
    if getattr(args, "memory_allocator_attribution", False):
        if not args.memory_metrics or _memory_backend(args.device) != "cuda":
            raise ValueError("--memory-allocator-attribution requires --memory-metrics and CUDA")
    if args.dense and transcode_cache.explicit_cache_dir(args):
        raise ValueError("--transcode-cache applies to --package only")
    for name in ("mm_max_images", "mm_max_image_bytes", "mm_max_image_tokens"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
    _stream_options(args)
    transcode_cache.kwargs_from_args(args)
    settings = _validate_batch_settings(
        max_slots=args.max_users, max_rows=args.max_rows, pages_total=args.pages_total,
        max_ctx=args.max_context, max_rows_step=args.max_rows_step,
        prefill_chunk=args.prefill_chunk, prefill_priority=args.prefill_priority,
        loader_workers=args.loader_workers, allow_auto_pages=True,
        spec_k=args.spec_k, spec_ngram=args.spec_ngram, max_decode_rows=args.max_decode_rows,
        mtp_confirm_kv_only=getattr(args, 'mtp_confirm_kv_only', False))
    if settings["pages_total"]:
        need = settings["capacity"]["reserved_pages_per_full_context_slot"]
        if settings["pages_total"] < need:
            raise ValueError(f"--pages-total {settings['pages_total']} cannot hold one "
                             f"--max-context {settings['max_ctx']} request ({need} pages)")
    if adaptive:
        settings['adaptive_serving'] = adaptive
        settings['adaptive_min_users'] = args.adaptive_min_users
        settings['adaptive_capture_ceiling_explicit'] = args.capture_workload_ceiling is not None
    if args.max_prompt + 1 > settings["max_ctx"]:
        sys.stderr.write(f"[context-batch] note: --max-prompt {args.max_prompt} >= --max-context "
                         f"{settings['max_ctx']}; longer requests are refused as context overflow\n")
    if args.capture_workload_only:
        settings["capture_workload_only"] = True
        settings["capture_workload_ceiling"] = capture_ceiling
    if args.spec_calibration_repeats is not None:
        settings["spec_calibration_repeats"] = args.spec_calibration_repeats
    if args.optimise and args.spec_k and not args.spec_ngram:
        settings["spec_row_aware"] = True
    return settings


def _failed_phase(exc: BaseException) -> str:
    """The startup phase a failure belongs to.

    A ``PredictiveTBEStartupError`` names its phase and has already reported ``failed``, so
    the latest progress event is ``failed`` (with ``failed_phase``), not the phase itself.
    """
    named = getattr(exc, "phase", None)
    if isinstance(named, str) and named:
        return named
    latest = startup_progress.snapshot()["latest"]
    if latest.get("phase") == "failed" and latest.get("failed_phase"):
        return str(latest["failed_phase"])
    return str(latest.get("phase") or "init")


def main(argv: Optional[Sequence[str]] = None, *,
         runtime_factory: Optional[Callable[..., Any]] = None) -> int:
    """Start the listener, load the runtime, serve. Exit 2 on any startup failure."""
    parser = build_parser()
    args = parser.parse_args(argv)

    def failed(phase: str, error: str, service: Optional[BatchService] = None,
               httpd: Optional[Server] = None) -> int:
        _emit_line({"status": "startup_failed", "phase": phase, "error": error})
        if service is not None:
            service.fail_startup(phase, error)
            service.stop_memory_monitor()
        if httpd is not None:
            grace = service.startup_failure_grace_s if service is not None else 0.0
            if grace > 0:
                time.sleep(grace)
            httpd.shutdown()
            httpd.server_close()
        return 2

    try:
        settings = validate_args(args)
        api_keys = _read_api_keys(args.api_key_file)
        if bool(args.qualification_protocol) != bool(args.qualification_role):
            raise ValueError("--qualification-protocol and --qualification-role must be given together")
        if args.qualification_wheel and not args.qualification_protocol:
            raise ValueError("--qualification-wheel requires --qualification-protocol")
        if args.qualification_protocol and not args.memory_metrics:
            raise ValueError('qualification requires --memory-metrics')
        if args.qualification_protocol and args.spec_ngram:
            raise ValueError('MTP qualification cannot substitute n-gram speculation')
        _reject_unbound_runtime_compact_qualification(args)
        qualification_identity = _bind_qualification(args, max_requests=args.max_users)
        if qualification_identity is not None and 'adaptive_serving' in qualification_identity:
            # Only the verified identity is handed to resource admission; never reopen
            # the protocol after weights load to choose a convenient post-hoc profile.
            _freeze_adaptive_qualification_settings(settings, qualification_identity)
    except (ValueError, OSError) as exc:
        return failed("arguments", str(exc))

    settings_view = dict(settings, device=args.device, vram_headroom_gib=args.vram_headroom_gib,
                         pages_total_requested=args.pages_total)
    if args.optimise:
        settings_view["optimisation_profile"] = "native-tbe21" if args.package else "native-dense"
        if args.optimisation_kernel == "quad":
            settings_view["optimisation_profile"] = "native-tbe21-quad-experimental"
        elif args.optimisation_kernel == "pair-vote":
            settings_view["optimisation_profile"] = "native-tbe21-pair-vote-experimental"
        elif args.optimisation_kernel in CLI_PROFILES:
            settings_view["optimisation_profile"] = "native-tbe21-" + args.optimisation_kernel + "-experimental"
    service = BatchService(
        model_id=args.model_id,
        source_kind=(_tbe21_source_kind(args) if args.package and args.runtime_format == "tbe21" else
                     "predictive_tbe" if args.package else "dense_bf16"),
        source=str(args.package or args.dense),
        limits=Limits(max_prompt=args.max_prompt, max_new=args.max_new,
                      default_max_new=args.default_max_new, max_queue=args.max_queue,
                      exact_digest=bool(args.exact_digest),
                      enable_images=bool(args.enable_images)),
        settings=settings_view, api_keys=api_keys,
        startup_failure_grace_s=args.startup_failure_grace_s,
        memory_metrics=args.memory_metrics, memory_interval_s=args.memory_interval_s,
        **({"memory_allocator_attribution": True} if args.memory_allocator_attribution else {}))
    try:
        httpd = make_server(service, args.host, args.port, max_body_bytes=args.max_body_bytes)
    except OSError as exc:
        return failed("listen", f"{type(exc).__name__}: {exc}")
    port = int(httpd.server_address[1])
    listener = threading.Thread(target=httpd.serve_forever, name="context-batch-http", daemon=True)
    listener.start()
    startup_progress.set_sink(service.on_progress)
    _emit_line({"status": "loading", "phase": "init", "host": args.host, "port": port,
                "model": service.model_id, "source_kind": service.source_kind})

    def log(message: str) -> None:
        sys.stderr.write(str(message) + "\n")
        sys.stderr.flush()

    t0 = time.perf_counter()
    factory = runtime_factory or load_runtime
    try:
        service.start_memory_monitor()
        runtime = factory(args, settings, service=service, log=log)
        if qualification_identity is not None:
            _verify_qualification_runtime(qualification_identity, runtime, args)
            service.qualification_identity = qualification_identity
            _enable_bound_resident_startup(service, qualification_identity)
        startup_progress.set_sink(None)
        service.startup_timings["startup_total_s"] = round(time.perf_counter() - t0, 3)
        service.attach(runtime)
    except BaseException as exc:  # noqa: BLE001 - any startup failure is reported, then exit 2
        startup_progress.set_sink(None)
        if isinstance(exc, KeyboardInterrupt):
            service.stop_memory_monitor()
            httpd.shutdown()
            httpd.server_close()
            return 130
        phase = _failed_phase(exc)
        error = f"{type(exc).__name__}: {exc}"
        startup_progress.report("failed", failed_phase=phase, error=error)
        return failed(phase, error, service, httpd)
    _emit_line({"status": "ready", "host": args.host, "port": port, "model": service.model_id,
                "source_kind": service.source_kind, "settings": service.public_settings(),
                "startup_timings_s": service.startup_timings})
    try:
        listener.join()  # returns when the listener stops; Ctrl-C interrupts it
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()
        httpd.shutdown()
        httpd.server_close()
    return 0


def optimised_main(argv: Optional[Sequence[str]] = None, **kwargs) -> int:
    """Release entry point with native engine optimisation enabled explicitly."""
    values = list(sys.argv[1:] if argv is None else argv)
    dense = any(value == "--dense" or value.startswith("--dense=") for value in values)
    defaults = ["--optimise", "--spec-k", "1"]
    if not dense:
        defaults += ["--runtime-format", "tbe21"]
    return main(defaults + values, **kwargs)


__all__ = ["BatchService", "ChatRequest", "Limits", "RequestRejected", "Server", "Ticket",
           "build_parser", "load_runtime", "main", "make_handler", "make_server",
           "parse_chat_request", "token_callback", "transcode_cache_loader_kwargs",
           "validate_args", "optimised_main"]


if __name__ == "__main__":
    raise SystemExit(main())
