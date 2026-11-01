"""B1 serialized HTTP server for the opt-in Context FastSession runtime.

This server uses ``FastSession.generate`` for every token. The predictive
package path is loaded only through ``load_predictive_fast_session``; it has
no Transformers ``model.generate`` fallback. Greedy decoding is the only
sampling mode in this initial server.

The listener starts BEFORE the model loads: ``GET /health`` answers 503
``{"status": "loading", "phase": ..., "done": ..., "total": ...}`` during startup,
200 ``{"status": "ok", ...}`` once ready, and 503 ``{"status": "startup_failed"}``
for ``--startup-failure-grace-s`` seconds after a failed start (then exit 2).
The verified transcode cache is opt-in (``--transcode-cache auto`` selects its default directory).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional

from . import startup_progress
from . import transcode_cache
from .engine import Request, RequestError, SamplingParams, TokenSink, normalize_messages
from ._startup_limits import MAX_CPU_ENCODE_WORKERS


RECEIPT_SCHEMA = "georefine.context_fast_server_receipt.v1"
DEFAULT_MAX_TOKENS = 512
MAX_BODY_BYTES = 8 << 20
_ALLOWED_FIELDS = {
    "model", "messages", "stream", "stream_options", "max_tokens",
    "max_completion_tokens", "temperature", "top_p", "stop", "n",
    "top_k", "logprobs", "top_logprobs", "tools", "tool_choice",
    "presence_penalty", "frequency_penalty", "seed",
}


class FastServerError(ValueError):
    def __init__(self, message: str, status: int = 400, error_type: str = "invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.error_type = error_type


def _now() -> int:
    return int(time.time())


def _numeric(body: dict, name: str, default: float) -> float:
    value = body.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FastServerError(f"{name} must be numeric")
    return float(value)


def _request_options(body: Dict[str, Any], model_id: str, default_max_tokens: int, *, allow_ignore_eos=False):
    allowed=_ALLOWED_FIELDS | ({'ignore_eos'} if allow_ignore_eos else set())
    unknown = sorted(set(body) - allowed)
    if allow_ignore_eos and 'ignore_eos' in body and type(body['ignore_eos']) is not bool:
        raise FastServerError('ignore_eos must be boolean')
    if unknown:
        raise FastServerError(f"unsupported request field(s): {', '.join(unknown)}")
    model = body.get("model")
    if model is not None:
        if not isinstance(model, str):
            raise FastServerError("model must be a string")
        if model != model_id:
            raise FastServerError(f"model {model!r} is not served here", 404, "model_not_found")
    if not isinstance(body.get("messages"), list) or not body["messages"]:
        raise FastServerError("messages must be a non-empty array")

    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise FastServerError("stream must be boolean")
    stream_options = body.get("stream_options", {})
    if stream_options is None:
        stream_options = {}
    if not isinstance(stream_options, dict) or set(stream_options) - {"include_usage"}:
        raise FastServerError("only stream_options.include_usage is supported")
    include_usage = stream_options.get("include_usage", False)
    if not isinstance(include_usage, bool):
        raise FastServerError("stream_options.include_usage must be boolean")
    if include_usage and not stream:
        raise FastServerError("stream_options.include_usage requires stream=true")

    temperature = _numeric(body, "temperature", 0.0)
    if temperature != 0.0:
        raise FastServerError("only greedy temperature=0 decoding is supported")
    if _numeric(body, "top_p", 1.0) != 1.0:
        raise FastServerError("top_p must be 1 for greedy decoding")
    if body.get("top_k") is not None:
        raise FastServerError("top_k is unsupported")
    if body.get("seed") is not None:
        raise FastServerError("seed is unsupported for greedy decoding")
    for name in ("presence_penalty", "frequency_penalty"):
        if _numeric(body, name, 0.0) != 0.0:
            raise FastServerError(f"{name} is unsupported")
    n = body.get("n", 1)
    if isinstance(n, bool) or not isinstance(n, int) or n != 1:
        raise FastServerError("only n=1 is supported")
    if body.get("logprobs") not in (None, False, 0):
        raise FastServerError("logprobs are unsupported")
    if body.get("top_logprobs", 0) not in (None, 0):
        raise FastServerError("top_logprobs are unsupported")
    if body.get("tools") not in (None, []):
        raise FastServerError("tools are unsupported")
    if body.get("tool_choice") not in (None, "none"):
        raise FastServerError("tool_choice is unsupported")

    raw_max = body.get("max_completion_tokens", body.get("max_tokens", default_max_tokens))
    if isinstance(raw_max, bool) or not isinstance(raw_max, int) or raw_max <= 0:
        raise FastServerError("max_tokens must be a positive integer")
    stop = body.get("stop", [])
    if stop is None:
        stop = []
    if isinstance(stop, str):
        stop = [stop]
    if not isinstance(stop, list) or any(not isinstance(item, str) for item in stop):
        raise FastServerError("stop must be a string or an array of strings")
    return SamplingParams(max_tokens=raw_max, temperature=0.0, top_p=1.0,
                          stop=list(stop)), stream, include_usage


class FastServerState:
    """Server state. ``session=None`` starts in ``loading``; :meth:`attach` makes it ready.

    The listener runs during startup so ``/health`` can report the current startup phase
    and its progress (503 while loading or after a failed start, 200 once ready).
    """

    def __init__(self, session=None, *, model_id: str, max_context: int,
                 source_kind: str, source: str, default_max_tokens: int = DEFAULT_MAX_TOKENS):
        self.session = session
        self.model_id = model_id
        self.max_context = int(max_context)
        self.source_kind = source_kind
        self.source = source
        self.default_max_tokens = int(default_max_tokens)
        self.request_lock = threading.Lock()
        self.status = "ok" if session is not None else "loading"
        self.failed_phase: Optional[str] = None
        self.error: Optional[str] = None
        self.progress: Dict[str, Any] = {"phase": "init"}
        self.t_start = time.time()
        self._last_print = ("", 0.0)

    @property
    def ready(self) -> bool:
        return self.status == "ok" and self.session is not None

    def attach(self, session) -> None:
        self.session = session
        self.status = "ok"

    def fail_startup(self, phase: str, error: str) -> None:
        self.failed_phase = phase
        self.error = error
        self.status = "startup_failed"

    def on_progress(self, event: Dict[str, Any]) -> None:
        """``startup_progress`` sink: keep the latest event for /health; print phase changes
        (and at most one progress line per 10 s within a phase) as JSON."""
        self.progress = dict(event)
        phase = str(event.get("phase"))
        now = time.time()
        last_phase, last_t = self._last_print
        if phase != last_phase or now - last_t >= 10.0:
            self._last_print = (phase, now)
            _print_progress(event)

    def progress_phase(self) -> Optional[str]:
        """The startup phase this server last saw reported (``failed`` resolves to the phase
        that failed); None before any report."""
        phase = self.progress.get("phase")
        if phase == "failed":
            return self.progress.get("failed_phase") or None
        return None if phase in (None, "init", "ready") else str(phase)

    def health(self):
        out: Dict[str, Any] = {"status": self.status, "model": self.model_id,
                               "source_kind": self.source_kind, "dynamic_batching": False,
                               "uptime_s": round(time.time() - self.t_start, 3)}
        if self.status == "loading":
            out.update({k: v for k, v in self.progress.items() if k != "t"})
            out["status"] = "loading"
            out["phase"] = self.progress.get("phase", "init")
            out["phases"] = startup_progress.snapshot()["phases"]
            return 503, out
        if self.status == "startup_failed":
            out.update({"phase": self.failed_phase, "error": self.error,
                        "phases": startup_progress.snapshot()["phases"]})
            return 503, out
        meta = getattr(self.session, "meta", {}) or {}
        out.update({k: meta.get(k) for k in ("load_s", "ready_s") if k in meta})
        cache = meta.get("transcode_cache")
        if isinstance(cache, dict):
            out["transcode_cache"] = cache.get("status")
        return 200, out

    def receipt(self) -> Dict[str, Any]:
        return {
            "schema": RECEIPT_SCHEMA,
            "model": self.model_id,
            "source_kind": self.source_kind,
            "source": self.source,
            "device": str(getattr(self.session.engine, "device", "unknown")),
            "max_context": self.max_context,
            "batch_size": 1,
            "dynamic_batching": False,
            "sampling": "greedy_only",
            "session": getattr(self.session, "meta", {}),
        }

    def complete(self, body: Dict[str, Any], *, request_id: str,
                 on_begin: Optional[Callable[[], None]] = None,
                 on_delta: Optional[Callable[[str], None]] = None,
                 on_done: Optional[Callable[[str, dict], None]] = None,
                 on_exact: Optional[Callable[[dict], None]] = None) -> Dict[str, Any]:
        params, _stream, _include_usage = _request_options(
            body, self.model_id, self.default_max_tokens,
            allow_ignore_eos=bool(getattr(self.session,"supports_ignore_eos",False)))
        try:
            messages, images = normalize_messages(body["messages"], allow_http_images=False)
        except RequestError as exc:
            raise FastServerError(str(exc)) from exc

        if not self.ready:
            raise FastServerError(f"model is not ready (status {self.status})", 503,
                                  "service_unavailable")
        self.request_lock.acquire()
        iterator = None
        req = Request(kind="chat", params=params, messages=messages,
                      chat_template_kwargs={"enable_thinking": False})
        req.images = images
        try:
            inputs = self.session.engine.prepare([req])
            prompt_tokens = int(req.prompt_tokens)
            if prompt_tokens + params.max_tokens + 16 > self.max_context:
                raise FastServerError(
                    f"prompt ({prompt_tokens}) plus max_tokens ({params.max_tokens}) "
                    f"exceeds max_context ({self.max_context})")

            # Reuse the prepared token sequence for text. Images need the processor
            # tensors too, so FastSession prepares that multimodal prefill itself.
            if getattr(self.session, 'supports_prepared_inputs', False):
                st = self.session.prefill(prepared_inputs=inputs)
            elif images:
                st = self.session.prefill(
                    messages=messages, images=images,
                    chat_template_kwargs={"enable_thinking": False})
            else:
                token_ids = inputs["input_ids"][0].tolist()
                st = self.session.prefill(token_ids=token_ids)
            req.prompt_tokens = prompt_tokens
            stop_ids=[] if body.get("ignore_eos",False) else self.session.eos_ids
            sink = TokenSink(self.session.tokenizer, stop_ids)
            if on_begin is not None:
                on_begin()
            exact=bool(getattr(self.session,'exact_digest',False))
            diagnostics=[]
            generate_options={'with_digests':True} if exact else {}
            iterator = self.session.generate(
                st, max_tokens=params.max_tokens, stop_ids=stop_ids,
                temperature=0.0, top_p=1.0, spec_k=None, **generate_options)
            while True:
                try:
                    token = next(iterator)
                except StopIteration:
                    break
                if exact:
                    token,digest=token
                    if not isinstance(digest,dict) or not {'d','dtype','vocab'}<=set(digest):
                        raise RuntimeError('incomplete target-logit diagnostic record')
                    record=dict(digest,i=len(diagnostics),token=int(token),eos=int(token) in self.session.eos_ids)
                    diagnostics.append(record)
                    if on_exact is not None:on_exact(record)
                keep_going = sink.push(req, int(token))
                while not req.out.empty():
                    event = req.out.get_nowait()
                    if event[0] == "delta" and event[1] and on_delta is not None:
                        on_delta(event[1])
                    elif event[0] == "done":
                        reason, usage = event[1], event[2]
                        if on_done is not None:
                            on_done(reason, usage)
                if not keep_going:
                    break
            if not req.finished:
                sink.finish(req, "length" if len(req.gen_ids) >= params.max_tokens else "stop")
            while not req.out.empty():
                event = req.out.get_nowait()
                if event[0] == "delta" and event[1] and on_delta is not None:
                    on_delta(event[1])
                elif event[0] == "done":
                    reason, usage = event[1], event[2]
                    if on_done is not None:
                        on_done(reason, usage)
            return {"text": req.text, "finish_reason": req.finish_reason,
                    "usage": sink.usage(req), "prompt_tokens": prompt_tokens,
                    "completion_tokens": len(req.gen_ids),
                    **({'glc_exact':diagnostics} if exact else {})}
        finally:
            if iterator is not None:
                try:
                    iterator.close()
                except Exception:  # noqa: BLE001 - log it; the lock below must still be released
                    sys.stderr.write(f"[context-fast] request {request_id}: generator close failed:\n"
                                     f"{traceback.format_exc()}")
                    sys.stderr.flush()
            self.request_lock.release()


def make_handler(state: FastServerState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "context-fast/0.1"

        def log_message(self, fmt, *args):
            if os.environ.get("CONTEXT_FAST_ACCESS_LOG"):
                sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _json(self, status: int, value: Any) -> None:
            data = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: int, message: str, error_type: str = "invalid_request_error"):
            self._json(status, {"error": {"message": message, "type": error_type,
                                          "code": status}})

        def _body(self) -> Dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError as exc:
                raise FastServerError("invalid Content-Length") from exc
            if length < 0 or length > MAX_BODY_BYTES:
                raise FastServerError(f"request body exceeds {MAX_BODY_BYTES} bytes", 413)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise FastServerError(f"body is not valid UTF-8 JSON: {exc}") from exc
            if not isinstance(body, dict):
                raise FastServerError("body must be a JSON object")
            return body

        def _sse_start(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

        def _sse(self, value: Any) -> None:
            payload = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
            self.wfile.flush()

        def do_GET(self):
            path = self.path.rstrip("/") or "/"
            if path == "/health":
                code, body = state.health()
                return self._json(code, body)
            if path == "/v1/models":
                return self._json(200, {"object": "list", "data": [{
                    "id": state.model_id, "object": "model", "created": _now(),
                    "owned_by": "georefine", "capabilities": ["chat", "stream"],
                    "dynamic_batching": bool((getattr(state.session,'meta',{}) or {}).get('dynamic_batching',False)),
                }]})
            if path == "/v1/memory":
                if not state.ready:
                    code,body=state.health();return self._json(code,body)
                probe=getattr(state.session,'memory_receipt',None)
                if probe is None:return self._error(404,'memory instrumentation unavailable','not_found')
                try:return self._json(200,probe())
                except FastServerError as exc:return self._error(exc.status,str(exc),exc.error_type)
            if path == "/v1/receipt":
                if not state.ready:
                    code, body = state.health()
                    return self._json(code, body)
                return self._json(200, state.receipt())
            return self._error(404, f"no route {self.path}", "not_found")

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._error(404, f"no route {self.path}", "not_found")
            try:
                body = self._body()
                params, stream, include_usage = _request_options(
                    body, state.model_id, state.default_max_tokens,
                    allow_ignore_eos=bool(getattr(state.session,"supports_ignore_eos",False)))
                del params
                if not state.ready:
                    raise FastServerError(f"model is not ready (status {state.status})", 503,
                                          "service_unavailable")
            except FastServerError as exc:
                return self._error(exc.status, str(exc), exc.error_type)

            call_id = "chatcmpl-" + uuid.uuid4().hex[:24]
            created = _now()
            started = False

            def begin():
                nonlocal started
                self._sse_start()
                started = True
                base = {"id": call_id, "object": "chat.completion.chunk",
                        "created": created, "model": state.model_id}
                self._sse({**base, "choices": [{"index": 0,
                                                "delta": {"role": "assistant"},
                                                "finish_reason": None}]})

            def delta(text: str):
                base = {"id": call_id, "object": "chat.completion.chunk",
                        "created": created, "model": state.model_id}
                self._sse({**base, "choices": [{"index": 0,
                                                "delta": {"content": text},
                                                "finish_reason": None}]})

            def exact_record(record):
                self._sse({'id':call_id,'object':'chat.completion.chunk','created':created,
                           'model':state.model_id,'choices':[],'glc_exact':record})

            def done(reason: str, usage: dict):
                base = {"id": call_id, "object": "chat.completion.chunk",
                        "created": created, "model": state.model_id}
                self._sse({**base, "choices": [{"index": 0, "delta": {},
                                                "finish_reason": reason}]})
                if include_usage:
                    self._sse({**base, "choices": [], "usage": usage})
                self._sse("[DONE]")

            try:
                diagnostic_options={'on_exact':exact_record if stream else None} if getattr(state.session,'exact_digest',False) else {}
                result = state.complete(body, request_id=call_id,
                                        on_begin=begin if stream else None,
                                        on_delta=delta if stream else None,
                                        on_done=done if stream else None, **diagnostic_options)
                if stream:
                    return None
                return self._json(200, {
                    "id": call_id, "object": "chat.completion", "created": created,
                    "model": state.model_id,
                    "choices": [{"index": 0,
                                 "message": {"role": "assistant", "content": result["text"]},
                                 "finish_reason": result["finish_reason"]}],
                    "usage": result["usage"],
                    **({'glc_exact':result['glc_exact']} if 'glc_exact' in result else {}),
                })
            except FastServerError as exc:
                if started:
                    try:
                        self._sse({"error": {"message": str(exc), "code": exc.status}})
                        self._sse("[DONE]")
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return None
                return self._error(exc.status, str(exc), exc.error_type)
            except RequestError as exc:
                if started:
                    try:
                        self._sse({"error": {"message": str(exc), "code": 400}})
                        self._sse("[DONE]")
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return None
                return self._error(400, str(exc))
            except (BrokenPipeError, ConnectionResetError):
                return None
            except Exception as exc:  # noqa: BLE001 - surface model/runtime failure
                if started:
                    try:
                        self._sse({"error": {"message": f"{type(exc).__name__}: {exc}",
                                              "code": 500}})
                        self._sse("[DONE]")
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return None
                return self._error(500, f"{type(exc).__name__}: {exc}", "server_error")

    return Handler


def serve(state: FastServerState, host: str, port: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, int(port)), make_handler(state))
    httpd.daemon_threads = True
    return httpd


def _parse(argv=None):
    parser = argparse.ArgumentParser(prog="glc_serve.context_fast_server",
                                     description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--package", help="BCTX+PPCX predictive package")
    source.add_argument("--dense", help="BF16 parent checkpoint for same-engine baseline")
    parser.add_argument("--tune", required=True, help="FastDecoder tune JSON")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--gdn-hv16", action="store_true",
                        help="opt in to the Qwen0.8B GDN geometry (16 value heads, conv4)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id", default="context-fast")
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--verified-cpu-stream", action="store_true")
    parser.add_argument("--cpu-encode-workers", type=int, choices=range(1, MAX_CPU_ENCODE_WORKERS + 1), default=1)
    parser.add_argument("--cpu-encode-pipeline", action="store_true",
                        help="use the ordered bounded CPU encoder pipeline (requires verified stream and >1 worker)")
    parser.add_argument("--source-inflight-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--cpu-vision-dense", action="store_true")
    parser.add_argument("--reuse-package-validation", action="store_true",
                        help="reuse the live full-integrity resolver for middle package validation")
    transcode_cache.add_cli_args(parser)
    parser.add_argument("--startup-failure-grace-s", type=float, default=10.0,
                        help="seconds /health keeps reporting startup_failed before exit 2")
    args = parser.parse_args(argv)
    try:
        _stream_options(args)
        if args.dense and transcode_cache.explicit_cache_dir(args):
            raise ValueError("--transcode-cache applies to --package only")
        transcode_cache.kwargs_from_args(args)
        if not args.startup_failure_grace_s >= 0:
            raise ValueError("--startup-failure-grace-s must be >= 0")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _stream_options(args):
    stream = getattr(args, "verified_cpu_stream", False)
    vision = getattr(args, "cpu_vision_dense", False)
    workers = getattr(args, "cpu_encode_workers", 1)
    pipeline = getattr(args, "cpu_encode_pipeline", False)
    reuse = getattr(args, 'reuse_bhe_validation', False)
    reuse_package = getattr(args, 'reuse_package_validation', False)
    ready_install = getattr(args, 'cpu_ready_install', False)
    prefetch_mtp = getattr(args, 'prefetch_mtp', False)
    skip_single_capture = getattr(args, 'skip_single_session_capture', False)
    budget = getattr(args, "source_inflight_bytes", 8 * 1024**3)
    if type(stream) is not bool or type(vision) is not bool or type(pipeline) is not bool:
        raise ValueError("CPU stream flags must be bools")
    if type(reuse) is not bool:
        raise ValueError('reuse_bhe_validation must be a bool')
    if type(reuse_package) is not bool:
        raise ValueError('reuse_package_validation must be a bool')
    if reuse_package and (not stream or getattr(args, 'verify_package', 'auto') != 'full'):
        raise ValueError('--reuse-package-validation requires --verified-cpu-stream and --verify-package full')
    if reuse and (not stream or getattr(args, 'runtime_format', 'tbe') != 'bhe1'):
        raise ValueError('--reuse-bhe-validation requires BHE1 and --verified-cpu-stream')
    if type(workers) is not int or not 1 <= workers <= MAX_CPU_ENCODE_WORKERS:
        raise ValueError(f"--cpu-encode-workers must be in [1,{MAX_CPU_ENCODE_WORKERS}]")
    if type(ready_install) is not bool:
        raise ValueError('cpu_ready_install must be a bool')
    if ready_install and (not stream or not pipeline or workers <= 1
                          or getattr(args, 'runtime_format', 'tbe') != 'bhe1'):
        raise ValueError('--cpu-ready-install requires verified BHE1 CPU pipeline and multiple workers')
    if type(prefetch_mtp) is not bool or type(skip_single_capture) is not bool:
        raise ValueError('MTP prefetch and single-session capture flags must be bools')
    if (prefetch_mtp or skip_single_capture) and (
            not stream or not pipeline or workers <= 1
            or getattr(args, 'runtime_format', 'tbe') != 'bhe1'):
        raise ValueError('startup overlap flags require the verified BHE1 CPU pipeline')
    if prefetch_mtp and (not getattr(args, 'spec_k', 0) or getattr(args, 'spec_ngram', False)):
        raise ValueError('--prefetch-mtp requires MTP speculation')
    if not stream and (vision or pipeline or workers != 1 or budget != 8 * 1024**3):
        raise ValueError("CPU startup options require --verified-cpu-stream")
    if not stream:
        return {}
    if not args.package:
        raise ValueError("--verified-cpu-stream requires --package")
    if type(budget) is not int or not 0 < budget <= 8 * 1024**3:
        raise ValueError("--source-inflight-bytes must be in (0,8 GiB]")
    if pipeline and workers <= 1:
        raise ValueError("--cpu-encode-pipeline requires --cpu-encode-workers > 1")
    options = dict(verified_cpu_stream=True, cpu_vision_dense=vision,
                   cpu_encode_workers=workers, cpu_encode_pipeline=pipeline,
                   source_inflight_bytes=budget)
    if reuse_package:
        options['reuse_package_validation'] = True
    return options


def load_session(args, *, log=print):
    stream_options = _stream_options(args)
    _configure_exact_runtime()
    from .kernel_cache import ensure_serving_kernels

    ensure_serving_kernels("fast", device=args.device, log=log)
    if args.package:
        from .predictive_tbe import load_predictive_fast_session

        log(transcode_cache.describe(args))
        return load_predictive_fast_session(
            package=args.package, tune=args.tune, device=args.device,
            max_len=args.max_context, attention_implementation="sdpa",
            loader_workers=args.loader_workers, log=log, **stream_options,
            gdn_hv16=bool(getattr(args, "gdn_hv16", False)),
            **transcode_cache.kwargs_from_args(args),
            **transcode_cache.package_verify_kwargs(args))
    from .fastserve import FastSession

    return FastSession.load(
        dense=args.dense, tune=args.tune, device=args.device,
        max_len=args.max_context, server_flags=("--no-mtp", "--exec-mode", "exact"),
        log=log, gdn_hv16=bool(getattr(args, "gdn_hv16", False)))


def _configure_exact_runtime():
    import torch

    workspace = ":4096:8"
    if torch.cuda.is_initialized() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != workspace:
        raise RuntimeError("configure exact runtime before CUDA initialization")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False


def _print_progress(event: Dict[str, Any]) -> None:
    """Startup progress sink: one JSON line per phase boundary or progress report."""
    try:
        print(json.dumps({"status": "loading", **event}, default=str), flush=True)
    except Exception:  # noqa: BLE001 - a progress sink must not raise
        pass


def _log_stderr(message: str) -> None:
    sys.stderr.write(str(message) + "\n")
    sys.stderr.flush()


def main(argv=None) -> int:
    """Start the listener, load the session (``/health`` reports progress), then serve.

    Exit 2 on any startup failure, after ``/health`` has reported ``startup_failed`` for
    ``--startup-failure-grace-s`` seconds.
    """
    args = _parse(argv)
    if args.max_context <= 0:
        raise SystemExit("--max-context must be positive")
    if args.port < 0 or args.port > 65535:
        raise SystemExit("--port must be between 0 and 65535")
    if not args.model_id:
        raise SystemExit("--model-id must be a non-empty string")
    if not 1 <= args.loader_workers <= 16:
        raise SystemExit("--loader-workers must be in [1,16]")
    state = FastServerState(
        None, model_id=args.model_id, max_context=args.max_context,
        source_kind="predictive_package" if args.package else "dense_parent",
        source=args.package or args.dense)
    try:
        httpd = serve(state, args.host, args.port)
    except OSError as exc:
        print(json.dumps({"status": "startup_failed", "phase": "listen",
                          "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        return 2
    port = int(httpd.server_address[1])
    listener = threading.Thread(target=httpd.serve_forever, name="context-fast-http", daemon=True)
    listener.start()
    print(json.dumps({"status": "loading", "phase": "init", "host": args.host, "port": port,
                      "model": state.model_id, "source_kind": state.source_kind}), flush=True)
    startup_progress.set_sink(state.on_progress)
    try:
        session = load_session(args, log=_log_stderr)
        state.attach(session)
    except BaseException as exc:  # noqa: BLE001 - one machine-readable failure line, then exit 2
        startup_progress.set_sink(None)
        if isinstance(exc, KeyboardInterrupt):
            httpd.shutdown()
            httpd.server_close()
            return 130
        failed = getattr(exc, "phase", None) or state.progress_phase() or "startup"
        error = f"{type(exc).__name__}: {exc}"
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()
        print(json.dumps({"status": "startup_failed", "phase": failed, "error": error}), flush=True)
        state.fail_startup(failed, error)
        if args.startup_failure_grace_s > 0:
            time.sleep(args.startup_failure_grace_s)
        httpd.shutdown()
        httpd.server_close()
        return 2
    finally:
        startup_progress.set_sink(None)
    print(json.dumps({"status": "ready", "model": state.model_id,
                      "host": args.host, "port": port,
                      "dynamic_batching": False}), flush=True)
    try:
        listener.join()  # returns when the listener stops; Ctrl-C interrupts it
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        httpd.server_close()
    return 0


__all__ = ["FastServerState", "make_handler", "serve", "load_session", "_parse", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
