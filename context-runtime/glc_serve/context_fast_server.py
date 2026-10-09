"""B1 serialized HTTP server for the opt-in Context FastSession runtime.

This server uses ``FastSession.generate`` for every token. The predictive
package path is loaded only through ``load_predictive_fast_session``; it has
no Transformers ``model.generate`` fallback. Greedy decoding is the only
sampling mode in this initial server.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional

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


def _request_options(body: Dict[str, Any], model_id: str, default_max_tokens: int):
    unknown = sorted(set(body) - _ALLOWED_FIELDS)
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
    def __init__(self, session, *, model_id: str, max_context: int,
                 source_kind: str, source: str, default_max_tokens: int = DEFAULT_MAX_TOKENS):
        self.session = session
        self.model_id = model_id
        self.max_context = int(max_context)
        self.source_kind = source_kind
        self.source = source
        self.default_max_tokens = int(default_max_tokens)
        self.request_lock = threading.Lock()
        self.ready = True

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
                 on_done: Optional[Callable[[str, dict], None]] = None) -> Dict[str, Any]:
        params, _stream, _include_usage = _request_options(
            body, self.model_id, self.default_max_tokens)
        try:
            messages, images = normalize_messages(body["messages"], allow_http_images=False)
        except RequestError as exc:
            raise FastServerError(str(exc)) from exc

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
            if images:
                st = self.session.prefill(
                    messages=messages, images=images,
                    chat_template_kwargs={"enable_thinking": False})
            else:
                token_ids = inputs["input_ids"][0].tolist()
                st = self.session.prefill(token_ids=token_ids)
            req.prompt_tokens = prompt_tokens
            sink = TokenSink(self.session.tokenizer, self.session.eos_ids)
            if on_begin is not None:
                on_begin()
            iterator = self.session.generate(
                st, max_tokens=params.max_tokens, stop_ids=self.session.eos_ids,
                temperature=0.0, top_p=1.0, spec_k=None)
            while True:
                try:
                    token = next(iterator)
                except StopIteration:
                    break
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
                    "completion_tokens": len(req.gen_ids)}
        finally:
            if iterator is not None:
                try:
                    iterator.close()
                except Exception:  # cleanup must still release the serialized session
                    pass
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
                return self._json(200, {"status": "ok" if state.ready else "loading",
                                        "model": state.model_id,
                                        "dynamic_batching": False})
            if path == "/v1/models":
                return self._json(200, {"object": "list", "data": [{
                    "id": state.model_id, "object": "model", "created": _now(),
                    "owned_by": "georefine", "capabilities": ["chat", "stream"],
                    "dynamic_batching": False,
                }]})
            if path == "/v1/receipt":
                return self._json(200, state.receipt())
            return self._error(404, f"no route {self.path}", "not_found")

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._error(404, f"no route {self.path}", "not_found")
            try:
                body = self._body()
                params, stream, include_usage = _request_options(
                    body, state.model_id, state.default_max_tokens)
                del params
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

            def done(reason: str, usage: dict):
                base = {"id": call_id, "object": "chat.completion.chunk",
                        "created": created, "model": state.model_id}
                self._sse({**base, "choices": [{"index": 0, "delta": {},
                                                "finish_reason": reason}]})
                if include_usage:
                    self._sse({**base, "choices": [], "usage": usage})
                self._sse("[DONE]")

            try:
                result = state.complete(body, request_id=call_id,
                                        on_begin=begin if stream else None,
                                        on_delta=delta if stream else None,
                                        on_done=done if stream else None)
                if stream:
                    return None
                return self._json(200, {
                    "id": call_id, "object": "chat.completion", "created": created,
                    "model": state.model_id,
                    "choices": [{"index": 0,
                                 "message": {"role": "assistant", "content": result["text"]},
                                 "finish_reason": result["finish_reason"]}],
                    "usage": result["usage"],
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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id", default="context-fast")
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--verified-cpu-stream", action="store_true")
    parser.add_argument("--cpu-encode-workers", type=int, choices=range(1, MAX_CPU_ENCODE_WORKERS + 1), default=1)
    parser.add_argument("--source-inflight-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--cpu-vision-dense", action="store_true")
    args = parser.parse_args(argv)
    try:
        _stream_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _stream_options(args):
    stream = getattr(args, "verified_cpu_stream", False)
    vision = getattr(args, "cpu_vision_dense", False)
    workers = getattr(args, "cpu_encode_workers", 1)
    budget = getattr(args, "source_inflight_bytes", 8 * 1024**3)
    if type(stream) is not bool or type(vision) is not bool:
        raise ValueError("CPU stream flags must be bools")
    if type(workers) is not int or not 1 <= workers <= MAX_CPU_ENCODE_WORKERS:
        raise ValueError(f"--cpu-encode-workers must be in [1,{MAX_CPU_ENCODE_WORKERS}]")
    if not stream and (vision or workers != 1 or budget != 8 * 1024**3):
        raise ValueError("CPU startup options require --verified-cpu-stream")
    if not stream:
        return {}
    if not args.package:
        raise ValueError("--verified-cpu-stream requires --package")
    if type(budget) is not int or not 0 < budget <= 8 * 1024**3:
        raise ValueError("--source-inflight-bytes must be in (0,8 GiB]")
    return dict(verified_cpu_stream=True, cpu_vision_dense=vision,
                cpu_encode_workers=workers, source_inflight_bytes=budget)


def load_session(args, *, log=print):
    stream_options = _stream_options(args)
    _configure_exact_runtime()
    if args.package:
        from .predictive_tbe_private import load_predictive_fast_session

        return load_predictive_fast_session(
            package=args.package, tune=args.tune, device=args.device,
            max_len=args.max_context, attention_implementation="sdpa",
            loader_workers=args.loader_workers, log=log, **stream_options)
    from .fastserve import FastSession

    return FastSession.load(
        dense=args.dense, tune=args.tune, device=args.device,
        max_len=args.max_context, server_flags=("--no-mtp", "--exec-mode", "exact"),
        log=log)


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


def main(argv=None) -> int:
    args = _parse(argv)
    if args.max_context <= 0:
        raise SystemExit("--max-context must be positive")
    if args.port < 0 or args.port > 65535:
        raise SystemExit("--port must be between 0 and 65535")
    if not args.model_id:
        raise SystemExit("--model-id must be a non-empty string")
    if not 1 <= args.loader_workers <= 16:
        raise SystemExit("--loader-workers must be in [1,16]")
    session = load_session(args)
    state = FastServerState(
        session, model_id=args.model_id, max_context=args.max_context,
        source_kind="predictive_package" if args.package else "dense_parent",
        source=args.package or args.dense)
    httpd = serve(state, args.host, args.port)
    print(json.dumps({"status": "ready", "model": state.model_id,
                      "host": args.host, "port": httpd.server_address[1],
                      "dynamic_batching": False}))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


__all__ = ["FastServerState", "make_handler", "serve", "load_session", "_parse", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
