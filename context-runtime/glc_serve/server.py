"""OpenAI-compatible HTTP server over the glc_serve engine (stdlib only).

Endpoints::

    GET  /health                 {"status": "ok", ...} once the gate passed
    GET  /v1/models              one model
    GET  /v1/receipt             load + gate receipts (what was verified)
    POST /v1/chat/completions    messages (text + image_url parts), tools,
                                 stream (SSE), logprobs/top_logprobs, seed
    POST /v1/completions         prompt, stream (SSE), logprobs; echo +
                                 max_tokens=0 returns prompt log-probabilities

Launch (TBE container, the lossless tier)::

    python -m glc_serve.server --bundle gs://bucket/prefix --cache-dir ~/bundle-cache \\
        --backend tbe --embed-on-host --port 8000

Launch (the uncompressed parent, through identical engine code)::

    python -m glc_serve.server --dense ~/models/Qwen3.8-27B --port 8001

``ThreadingHTTPServer`` gives one thread per connection; every model call is
made by the single engine thread, so HTTP threads only enqueue and stream.
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
from pathlib import Path
from typing import Any, Dict, List, Optional

from .engine import Engine, Request, RequestError, SamplingParams, parse_tool_calls

SERVER_RECEIPT_SCHEMA = "georefine.tbe_serve_server_receipt.v1"


def _now() -> int:
    return int(time.time())


def sampling_from_body(body: Dict[str, Any], *, default_max_tokens: int) -> SamplingParams:
    stop = body.get("stop") or []
    if isinstance(stop, str):
        stop = [stop]
    max_tokens = body.get("max_completion_tokens", body.get("max_tokens"))
    lp = body.get("logprobs")
    top_lp = body.get("top_logprobs") or 0
    if isinstance(lp, int) and not isinstance(lp, bool):   # legacy completions: logprobs=N
        top_lp, lp = int(lp), True
    temperature = body.get("temperature")
    return SamplingParams(
        max_tokens=int(default_max_tokens if max_tokens is None else max_tokens),
        temperature=1.0 if temperature is None else float(temperature),
        top_p=float(body.get("top_p", 1.0) if body.get("top_p") is not None else 1.0),
        top_k=body.get("top_k"),
        seed=body.get("seed"),
        stop=[str(s) for s in stop],
        logprobs=bool(lp),
        top_logprobs=int(top_lp),
    )


class ServerState:
    def __init__(self, engine: Engine, *, model_name: str, receipts: Dict[str, Any],
                 default_max_tokens: int = 512):
        self.engine = engine
        self.model_name = model_name
        self.receipts = receipts
        self.default_max_tokens = int(default_max_tokens)
        self.ready = True


def _lp_items(tok, entries: List[Optional[Dict[str, Any]]], token_ids: List[int]):
    """OpenAI chat ``logprobs.content`` from engine logprob records."""
    out = []
    for tid, e in zip(token_ids, entries):
        if e is None:
            continue
        s = tok.decode([tid], skip_special_tokens=False)
        out.append({
            "token": s, "logprob": e["logprob"], "bytes": list(s.encode("utf-8")),
            "top_logprobs": [
                {"token": tok.decode([t], skip_special_tokens=False), "logprob": v,
                 "bytes": list(tok.decode([t], skip_special_tokens=False).encode("utf-8"))}
                for t, v in e.get("top", [])
            ],
        })
    return out


def make_handler(state: ServerState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "glc-serve/0.1"

        def log_message(self, fmt, *args):  # quiet by default
            if os.environ.get("GLC_SERVE_ACCESS_LOG"):
                sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        # -- helpers ----------------------------------------------------------
        def _json(self, code: int, obj: Any) -> None:
            data = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code: int, msg: str, etype: str = "invalid_request_error") -> None:
            self._json(code, {"error": {"message": msg, "type": etype, "code": code}})

        def _body(self) -> Dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except ValueError as exc:
                raise RequestError(f"body is not JSON: {exc}")
            if not isinstance(body, dict):
                raise RequestError("body must be a JSON object")
            return body

        def _sse_start(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

        def _sse(self, obj: Any) -> None:
            payload = obj if isinstance(obj, str) else json.dumps(obj)
            self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
            self.wfile.flush()

        # -- routes -----------------------------------------------------------
        def do_GET(self):
            if self.path.rstrip("/") in ("/health", "/v1/health"):
                return self._json(200, {"status": "ok" if state.ready else "loading",
                                        "model": state.model_name,
                                        "engine": state.engine.stats})
            if self.path.rstrip("/") == "/v1/models":
                return self._json(200, {"object": "list", "data": [{
                    "id": state.model_name, "object": "model", "created": _now(),
                    "owned_by": "georefine",
                    "capabilities": ["completion", "chat", "multimodal"]
                    if getattr(state.engine.processor, "image_processor", None) is not None
                    else ["completion", "chat"],
                }]})
            if self.path.rstrip("/") == "/v1/receipt":
                return self._json(200, {"schema": SERVER_RECEIPT_SCHEMA, **state.receipts,
                                        "engine": state.engine.stats})
            return self._error(404, f"no route {self.path}", "not_found")

        def do_POST(self):
            try:
                body = self._body()
                if self.path.rstrip("/") == "/v1/chat/completions":
                    return self._chat(body)
                if self.path.rstrip("/") == "/v1/completions":
                    return self._completion(body)
                return self._error(404, f"no route {self.path}", "not_found")
            except RequestError as exc:
                return self._error(400, str(exc))
            except (BrokenPipeError, ConnectionResetError):
                return None

        def _submit(self, req: Request) -> Request:
            return state.engine.submit(req)

        def _chat(self, body: Dict[str, Any]) -> None:
            params = sampling_from_body(body, default_max_tokens=state.default_max_tokens)
            tools = body.get("tools")
            if body.get("tool_choice") == "none":
                tools = None
            req = Request(kind="chat", params=params, messages=body.get("messages"),
                          tools=tools,
                          chat_template_kwargs=dict(body.get("chat_template_kwargs") or {}))
            self._submit(req)
            cid = "chatcmpl-" + req.rid
            created = _now()
            if body.get("stream"):
                include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
                return self._chat_stream(req, cid, created, include_usage, bool(tools))
            text_parts: List[str] = []
            lp_entries, lp_ids = [], []
            while True:
                ev = req.out.get()
                if ev[0] == "delta":
                    text_parts.append(ev[1])
                    if ev[2] is not None:
                        lp_ids.append(ev[2])
                        lp_entries.append(ev[3])
                elif ev[0] == "error":
                    return self._error(ev[1], ev[2], "server_error" if ev[1] >= 500
                                       else "invalid_request_error")
                elif ev[0] == "done":
                    reason, usage, spec = ev[1], ev[2], ev[3]
                    break
            text = "".join(text_parts)
            message: Dict[str, Any] = {"role": "assistant", "content": text}
            if tools:
                content, calls = parse_tool_calls(text)
                if calls:
                    message = {"role": "assistant", "content": content or None,
                               "tool_calls": calls}
                    reason = "tool_calls"
            choice: Dict[str, Any] = {"index": 0, "message": message, "finish_reason": reason}
            if params.logprobs:
                choice["logprobs"] = {"content": _lp_items(state.engine.tok, lp_entries, lp_ids)}
            resp = {"id": cid, "object": "chat.completion", "created": created,
                    "model": state.model_name, "choices": [choice], "usage": usage}
            if spec:
                resp["x_speculative"] = spec
            return self._json(200, resp)

        def _chat_stream(self, req: Request, cid: str, created: int,
                         include_usage: bool, tools: bool) -> None:
            self._sse_start()
            base = {"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": state.model_name}
            self._sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"},
                                            "finish_reason": None}]})
            full: List[str] = []
            while True:
                ev = req.out.get()
                if ev[0] == "delta":
                    if not ev[1] and ev[3] is None:
                        continue
                    full.append(ev[1])
                    chunk = {"index": 0, "delta": {"content": ev[1]}, "finish_reason": None}
                    if req.params.logprobs and ev[2] is not None:
                        chunk["logprobs"] = {"content": _lp_items(state.engine.tok, [ev[3]],
                                                                  [ev[2]])}
                    self._sse({**base, "choices": [chunk]})
                elif ev[0] == "error":
                    self._sse({"error": {"message": ev[2], "code": ev[1]}})
                    break
                elif ev[0] == "done":
                    reason, usage = ev[1], ev[2]
                    delta: Dict[str, Any] = {}
                    if tools:
                        _content, calls = parse_tool_calls("".join(full))
                        if calls:
                            delta = {"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]}
                            reason = "tool_calls"
                    self._sse({**base, "choices": [{"index": 0, "delta": delta,
                                                    "finish_reason": reason}]})
                    if include_usage:
                        self._sse({**base, "choices": [], "usage": usage})
                    break
            self._sse("[DONE]")

        def _completion(self, body: Dict[str, Any]) -> None:
            params = sampling_from_body(body, default_max_tokens=state.default_max_tokens)
            prompt = body.get("prompt", "")
            if isinstance(prompt, list):
                if len(prompt) != 1:
                    raise RequestError("batched prompts: send one request per prompt")
                prompt = prompt[0]
            if not isinstance(prompt, str):
                raise RequestError("prompt must be a string")
            echo = bool(body.get("echo"))
            cid = "cmpl-" + uuid.uuid4().hex[:24]
            created = _now()
            if echo and params.max_tokens == 0:
                req = Request(kind="score", params=params, prompt=prompt)
                self._submit(req)
                ev = req.out.get()
                if ev[0] == "error":
                    return self._error(ev[1], ev[2])
                items, usage = ev[1], ev[2]
                tok = state.engine.tok
                toks = [tok.decode([it["token_id"]], skip_special_tokens=False) for it in items]
                lp = {
                    "tokens": toks,
                    "token_logprobs": [it["logprob"] for it in items],
                    "top_logprobs": [
                        None if it["logprob"] is None else
                        {tok.decode([t], skip_special_tokens=False): v for t, v in it["top"]}
                        for it in items],
                    "text_offset": [sum(len(t) for t in toks[:i]) for i in range(len(toks))],
                }
                return self._json(200, {"id": cid, "object": "text_completion",
                                        "created": created, "model": state.model_name,
                                        "choices": [{"index": 0, "text": prompt,
                                                     "logprobs": lp, "finish_reason": "length"}],
                                        "usage": usage})
            req = Request(kind="completion", params=params, prompt=prompt)
            self._submit(req)
            if body.get("stream"):
                self._sse_start()
                base = {"id": cid, "object": "text_completion", "created": created,
                        "model": state.model_name}
                while True:
                    ev = req.out.get()
                    if ev[0] == "delta":
                        if ev[1]:
                            self._sse({**base, "choices": [{"index": 0, "text": ev[1],
                                                            "logprobs": None,
                                                            "finish_reason": None}]})
                    elif ev[0] == "error":
                        self._sse({"error": {"message": ev[2], "code": ev[1]}})
                        break
                    elif ev[0] == "done":
                        self._sse({**base, "choices": [{"index": 0, "text": "",
                                                        "logprobs": None,
                                                        "finish_reason": ev[1]}],
                                   "usage": ev[2]})
                        break
                self._sse("[DONE]")
                return None
            parts, ids, lps = [], [], []
            while True:
                ev = req.out.get()
                if ev[0] == "delta":
                    parts.append(ev[1])
                    if ev[2] is not None:
                        ids.append(ev[2])
                        lps.append(ev[3])
                elif ev[0] == "error":
                    return self._error(ev[1], ev[2])
                elif ev[0] == "done":
                    reason, usage = ev[1], ev[2]
                    break
            text = "".join(parts)
            lp_obj = None
            if params.logprobs:
                tok = state.engine.tok
                toks = [tok.decode([t], skip_special_tokens=False) for t in ids]
                lp_obj = {"tokens": toks,
                          "token_logprobs": [e["logprob"] if e else None for e in lps],
                          "top_logprobs": [
                              {tok.decode([t], skip_special_tokens=False): v
                               for t, v in (e or {}).get("top", [])} for e in lps]}
            return self._json(200, {"id": cid, "object": "text_completion", "created": created,
                                    "model": state.model_name,
                                    "choices": [{"index": 0, "text": (prompt + text) if echo
                                                 else text, "logprobs": lp_obj,
                                                 "finish_reason": reason}],
                                    "usage": usage})

    return Handler


def serve(state: ServerState, host: str, port: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    httpd.daemon_threads = True
    return httpd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse(argv=None):
    ap = argparse.ArgumentParser(prog="glc_serve.server", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--bundle", help="serving bundle: local dir, gs://..., or http(s)://...")
    src.add_argument("--dense", help="the uncompressed parent checkpoint directory")
    ap.add_argument("--cache-dir", help="local cache for a remote bundle (never /tmp)")
    ap.add_argument("--manifest-sha256", help="pin the bundle manifest digest")
    ap.add_argument("--backend", default="tbe", choices=("tbe", "fwp1", "materialize"))
    ap.add_argument("--exec-mode", default="fused", choices=("fused", "exact"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--embed-on-host", action="store_true")
    ap.add_argument("--text-only", action="store_true",
                    help="DEBUG ONLY: skip the vision tower and MTP head")
    ap.add_argument("--no-mtp", action="store_true", help="do not load the MTP head")
    ap.add_argument("--mtp-speculative", action="store_true",
                    help="MTP self-speculative greedy decoding for single requests")
    ap.add_argument("--gpu-weight-budget-gb", type=float, default=None,
                    help="stream text layers beyond this resident-weight budget")
    ap.add_argument("--memory-cap-gib", type=float, default=None,
                    help="EMULATION: cap this process's CUDA allocator (and refuse "
                         "to start if the load exceeds it)")
    ap.add_argument("--prefetch-decode", action="store_true")
    ap.add_argument("--fused-max-m", type=int, default=48)
    ap.add_argument("--fetch-depth", type=int, default=2)
    ap.add_argument("--evict-shards", action="store_true")
    ap.add_argument("--gate", default="sample", choices=("sample", "full", "off"))
    ap.add_argument("--reference-logits", help="dense reference logits (glc_serve.gate make-reference)")
    ap.add_argument("--prefill-chunk", type=int, default=4096,
                    help="chunk a long single text prompt's prefill (0 = off)")
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--batch-window-ms", type=float, default=10.0)
    ap.add_argument("--max-input-tokens", type=int, default=None)
    ap.add_argument("--default-max-tokens", type=int, default=512)
    ap.add_argument("--no-http-images", action="store_true")
    ap.add_argument("--chat-template-kwargs", default=None,
                    help='JSON merged under every request\'s own, e.g. \'{"enable_thinking": false}\' '
                         "(clients such as lm_eval cannot send it per request)")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--receipt-out", help="write the startup receipt JSON here")
    return ap.parse_args(argv)


def build_state(args, *, log=print) -> ServerState:
    import torch

    from .loader import ServeOptions, load_bundle_model, load_dense_parent

    t0 = time.perf_counter()
    cap_receipt = None
    if args.memory_cap_gib and torch.cuda.is_available():
        from .memcap import apply_memory_cap

        cap_receipt = apply_memory_cap(args.device, args.memory_cap_gib)
    opts = ServeOptions(
        backend=args.backend, exec_mode=args.exec_mode, device=args.device,
        embed_on_host=args.embed_on_host, text_only=args.text_only,
        load_mtp=not args.no_mtp,
        gpu_weight_budget_bytes=(int(args.gpu_weight_budget_gb * (1 << 30))
                                 if args.gpu_weight_budget_gb else None),
        prefetch_decode=args.prefetch_decode, fused_max_m=args.fused_max_m,
        fetch_depth=args.fetch_depth, evict_shards=args.evict_shards,
    )
    if args.dense:
        loaded = load_dense_parent(args.dense, opts, log=log)
        name = args.model_name or (Path(args.dense).name + "-bf16")
    else:
        from .bundle import open_bundle

        bundle = open_bundle(args.bundle, cache_dir=args.cache_dir,
                             expected_manifest_sha256=args.manifest_sha256)
        loaded = load_bundle_model(bundle, opts, log=log)
        name = args.model_name or (Path(str(args.bundle).rstrip("/")).name + f"-{args.backend}")
    from transformers import AutoProcessor, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(loaded.local_dir))
    processor = None
    if not args.text_only:
        try:
            processor = AutoProcessor.from_pretrained(str(loaded.local_dir))
        except Exception as exc:  # a text-only checkpoint has no processor
            log(f"[server] no multimodal processor: {type(exc).__name__}: {exc}")
    receipts: Dict[str, Any] = {"load": loaded.receipt, "memory_cap": cap_receipt}
    if args.gate != "off" and loaded.bundle is not None:
        from .gate import run_weight_gate

        g = run_weight_gate(loaded, scope=args.gate)
        receipts["weight_gate"] = g
        if g["status"] != "ok":
            raise SystemExit(f"bit-exact weight gate FAILED: {json.dumps(g)[:2000]}")
    engine = Engine(loaded, processor, tok, max_batch=args.max_batch,
                    batch_window_s=args.batch_window_ms / 1000.0,
                    enable_mtp=args.mtp_speculative,
                    max_input_tokens=args.max_input_tokens,
                    allow_http_images=not args.no_http_images,
                    prefill_chunk_tokens=args.prefill_chunk,
                    default_chat_template_kwargs=(json.loads(args.chat_template_kwargs)
                                                  if args.chat_template_kwargs else None),
                    log=log)
    if args.reference_logits:
        from .gate import check_against_reference

        lg = check_against_reference(engine, args.reference_logits)
        receipts["logits_gate"] = lg
        if lg["status"] != "ok":
            raise SystemExit(f"logits-vs-dense gate FAILED: {json.dumps(lg)[:2000]}")
    if cap_receipt is not None:
        from .memcap import assert_within_cap

        receipts["memory_cap_after_load"] = assert_within_cap(args.device, cap_receipt)
    receipts["startup_s"] = round(time.perf_counter() - t0, 3)
    return ServerState(engine.start(), model_name=name, receipts=receipts,
                       default_max_tokens=args.default_max_tokens)


def main(argv=None) -> int:
    args = _parse(argv)
    state = build_state(args)
    if args.receipt_out:
        p = Path(args.receipt_out)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(state.receipts, indent=1, default=str))
        os.replace(tmp, p)
    httpd = serve(state, args.host, args.port)
    print(json.dumps({"status": "serving", "host": args.host, "port": args.port,
                      "model": state.model_name, "startup_s": state.receipts.get("startup_s")}),
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.engine.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
