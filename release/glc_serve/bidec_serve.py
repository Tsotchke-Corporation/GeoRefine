#!/usr/bin/env python3
"""OpenAI-compatible HTTP server over the batch-invariant continuous-batching engine (bidec).

    python -m glc_serve.bidec_serve --parent DIR --gguf F.gguf --tune T.json --port 8290 --api-key-file K
    python -m glc_serve.bidec_serve --bundle DIR --tune T.json ...          # TBE codec

Endpoints: GET /health (open), GET /v1/models, POST /v1/chat/completions (text; stream or not).
Contract: every request's output is bitwise what it would be alone (G3a-BI; bi_gate_engine.py).
Loopback only.  Admission: --max-queue waiting requests, then 429.  Per-request caps:
--max-prompt, --max-new.  One engine thread; HTTP handlers only enqueue and stream.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import queue
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


class Server(ThreadingHTTPServer):
    """ThreadingHTTPServer with a backlog a multi-user benchmark cannot overflow.

    The stdlib default is a listen backlog of 5.  A closed-loop bench at 16 or 32 users opens
    that many connections at once and the kernel resets the overflow, which shows up as a client
    ConnectionResetError that looks like an engine failure and is not one.  Observed on the CPU
    HTTP tests at 12 concurrent streams before this was raised.
    """

    request_queue_size = 512
    daemon_threads = True
    allow_reuse_address = True

# The modules below import `glc_serve.*` absolutely, which is how the installed wheel is laid
# out.  Running this file as `release.glc_serve.bidec_serve` from a repository checkout (no
# install) puts `release/` on the path but not inside it, so make the same spelling work there.
if __package__ and __package__ != "glc_serve":
    _p = str(Path(__file__).resolve().parent.parent)
    if _p not in sys.path:
        sys.path.insert(0, _p)


class Service:
    def __init__(self, a):
        from glc_serve import bidec

        self.a = a
        self.keys = [l.strip() for l in Path(a.api_key_file).read_text().splitlines() if l.strip()]
        self.L = bidec.load_model(bundle=a.bundle, parent=a.parent, gguf=a.gguf, tune=a.tune, spec=a.spec_k > 0)
        self.tok = self.L["tok"]
        self.bd = bidec.BatchDecoder(self.L["fd"], max_slots=a.slots, max_rows=a.max_rows, pages_total=a.pages,
                                     max_ctx=a.max_ctx, R=a.spec_k + 1,
                                     gdn_ring_dtype=a.gdn_ring_dtype)
        self.capture = self.bd.capture()
        # chunk_align = PAGE: a prompt chunk that ends mid-block is exact but cannot publish a
        # cacheable prefix, because the GDN recurrent state is only snapshottable on a block
        # boundary.  Aligning here costs nothing and keeps the prefix cache usable.
        self.B = bidec.Batcher(self.bd, max_rows_step=a.max_rows, prefill_chunk=a.prefill_chunk,
                               max_active=a.max_users or a.slots, spec_k=a.spec_k,
                               chunk_align=bidec.PAGE)
        if a.prefill_chunk % bidec.PAGE:
            sys.stderr.write(f"[bidec_serve] note: --prefill-chunk {a.prefill_chunk} is not a "
                             f"multiple of the {bidec.PAGE}-token block; chunks are aligned down "
                             f"to the block boundary\n")
        self.fatal: str = ""
        self.stops = bidec.stop_ids_for(self.tok, a.bundle or a.parent)
        self.bidec = bidec
        self.wake = threading.Event()
        self.served = 0
        self.cancelled = 0
        self.meta = {"engine": "glc_serve.bidec (BI-GEMM continuous batching)", "load": self.L["rec"],
                     "capture_s": self.capture, "slots": a.slots, "pages": a.pages, "max_ctx": a.max_ctx,
                     "state_bytes": self.bd.state_bytes, "streamed_bytes_per_step": self.bd.streamed_bytes(),
                     "max_users": self.B.max_active, "max_decode_rows": self.B.max_decode_rows,
                     "gdn_ring_dtype": self.bd.gdn_ring_dtype}
        # Per-user cost and the context at which KV overtakes the fixed GDN state, from the
        # engine's own shapes: what --pages / --max-users / --max-ctx should be set from.
        try:
            from glc_serve import bidec_capacity as _cap
            g = _cap.geometry_of(self.bd)
            self.meta["geometry"] = _cap.asdict(g)
            self.meta["capacity_plan"] = g.plan(ctxs=(2048, a.max_ctx), state_ring=self.bd.R)
        except Exception as e:  # noqa: BLE001
            # CLASSIFICATION: production_fallback.  The capacity block is advisory sizing
            # information; if the shapes cannot be read the server still serves and the receipt
            # still grades, with the reason recorded rather than the key quietly absent.
            self.meta["capacity_plan_error"] = f"{type(e).__name__}: {e}"
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while True:
            if self.B.idle():
                self.wake.wait(0.05)
                self.wake.clear()
                continue
            try:
                self.B.step()
            except Exception as e:  # noqa: BLE001
                # A step failure is not recoverable (device buffers are in an unknown state), but
                # the engine thread must not die silently: record it, end every stream, and let
                # the HTTP layer answer 503 instead of hanging forever on an empty queue.
                self.fatal = f"{type(e).__name__}: {e}"
                sys.stderr.write(f"[bidec_serve] step failed, engine stopped: {self.fatal}\n")
                for s in list(self.B.waiting) + list(self.B.active):
                    s.done, s.finish = True, "error"
                    if s.on_token:
                        s.on_token(s, None)
                self.B.active, self.B.waiting = [], []
                return

    def submit(self, body):
        if self.fatal:
            raise RuntimeError(f"engine stopped: {self.fatal}")
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise ValueError("messages required")
        for m in msgs:
            if not isinstance(m.get("content"), str):
                raise ValueError("text-only server: content must be a string")
        if body.get("tools"):
            raise ValueError("tool calling is not supported")
        ctk = {"enable_thinking": False}
        ctk.update(body.get("chat_template_kwargs") or {})
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False, **ctk)
        if len(ids) > self.a.max_prompt:
            raise ValueError(f"prompt has {len(ids)} tokens > cap {self.a.max_prompt}")
        mx = body.get("max_completion_tokens", body.get("max_tokens")) or self.a.default_max_new
        mx = max(1, min(int(mx), self.a.max_new))
        temp = body.get("temperature")
        temp = 0.0 if temp is None else float(temp)
        if self.a.spec_k and temp > 0:
            raise ValueError("this server runs exact greedy speculation; temperature must be 0")
        q: "queue.Queue" = queue.Queue()
        s = self.bidec.Seq(rid=uuid.uuid4().hex, prompt=list(ids), max_new=mx, stop_ids=self.stops,
                           temperature=temp, top_p=float(body.get("top_p") or 1.0), seed=body.get("seed"))
        s.on_token = lambda seq, t: q.put(t)
        with self.B.lock:
            if len(self.B.waiting) >= self.a.max_queue:
                raise OverflowError("queue full")
        self.B.add(s)
        self.wake.set()
        return s, q


def make_handler(svc: Service):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code, obj, extra=None):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def _auth(self):
            h = self.headers.get("Authorization", "")
            tok = h[7:].strip() if h[:7].lower() == "bearer " else ""
            return bool(tok) and any(hmac.compare_digest(tok.encode(), k.encode()) for k in svc.keys)

        def do_GET(self):
            p = self.path.split("?", 1)[0]
            if p == "/health":
                st = svc.B.stats()
                st.update({"status": "error" if svc.fatal else "ok", "served": svc.served,
                           "cancelled": svc.cancelled, "error": svc.fatal or None,
                           "max_queue": svc.a.max_queue})
                return self._json(503 if svc.fatal else 200, st)
            if not self._auth():
                return self._json(401, {"error": {"message": "invalid or missing API key"}})
            if p == "/v1/models":
                return self._json(200, {"object": "list", "data": [{"id": svc.a.model_name, "object": "model"}]})
            if p == "/v1/receipt":
                return self._json(200, svc.meta)
            return self._json(404, {"error": {"message": "no route"}})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            if not self._auth():
                return self._json(401, {"error": {"message": "invalid or missing API key"}})
            if self.path.split("?", 1)[0] != "/v1/chat/completions":
                return self._json(404, {"error": {"message": "no route"}})
            try:
                body = json.loads(raw or b"{}")
                s, q = svc.submit(body)
            except OverflowError:
                return self._json(429, {"error": {"message": "server busy: queue full", "type": "rate_limit"}},
                                  {"Retry-After": "2"})
            except RuntimeError as e:
                return self._json(503, {"error": {"message": str(e), "type": "server_error"}})
            except (ValueError, TypeError) as e:
                return self._json(400, {"error": {"message": str(e)}})
            cid = "chatcmpl-" + s.rid[:24]
            stream = bool(body.get("stream"))
            toks, sent = [], ""
            dec = svc.tok

            def text_upto():
                t = dec.decode([x for x in toks if x not in svc.stops], skip_special_tokens=False)
                return t

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True

                def sse(o):
                    self.wfile.write(f"data: {json.dumps(o)}\n\n".encode())
                    self.wfile.flush()
                sse({"id": cid, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant"},
                                                                                  "finish_reason": None}]})
            try:
                while True:
                    t = q.get()
                    if t is None:
                        break
                    toks.append(t)
                    if stream:
                        txt = text_upto()
                        if not txt.endswith("�") and len(txt) > len(sent) and txt.startswith(sent):
                            sse({"id": cid, "object": "chat.completion.chunk",
                                 "choices": [{"index": 0, "delta": {"content": txt[len(sent):]},
                                              "finish_reason": None}]})
                            sent = txt
            except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                # The client hung up mid-stream.  Abort the request so its slot and KV pages go
                # back to the pool instead of generating max_new tokens nobody is reading.
                svc.B.cancel(s.rid)
                svc.cancelled += 1
                self.close_connection = True
                while q.get() is not None:                 # drain to the end-of-stream sentinel
                    pass
                return
            svc.served += 1
            txt = text_upto()
            ntok = len(toks)
            usage = {"prompt_tokens": len(s.prompt), "completion_tokens": ntok, "total_tokens": len(s.prompt) + ntok}
            timings = {"draft_n": s.prop, "draft_n_accepted": s.acc}
            if stream:
                if len(txt) > len(sent) and txt.startswith(sent):
                    sse({"id": cid, "object": "chat.completion.chunk",
                         "choices": [{"index": 0, "delta": {"content": txt[len(sent):]}, "finish_reason": None}]})
                sse({"id": cid, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {},
                                                                                  "finish_reason": s.finish}],
                     "usage": usage, "timings": timings})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            return self._json(200, {"id": cid, "object": "chat.completion", "model": svc.a.model_name,
                                    "choices": [{"index": 0, "message": {"role": "assistant", "content": txt},
                                                 "finish_reason": s.finish}], "usage": usage})
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle")
    ap.add_argument("--parent")
    ap.add_argument("--gguf")
    ap.add_argument("--tune", required=True)
    ap.add_argument("--slots", type=int, default=64, help="KV/state slots allocated on the device")
    ap.add_argument("--max-users", type=int, default=0,
                    help="concurrent requests admitted to the batch (0 = --slots); extra requests wait "
                         "in the queue up to --max-queue, then 429")
    ap.add_argument("--pages", type=int, default=800)
    ap.add_argument("--max-ctx", type=int, default=16384)
    ap.add_argument("--max-rows", type=int, default=256)
    ap.add_argument("--prefill-chunk", type=int, default=256)
    ap.add_argument("--gdn-ring-dtype", choices=("fp32", "fp16"), default="fp32",
                    help="storage dtype of the per-slot GDN recurrent ring. fp32 is the default "
                         "and the only one the kernel supports today; fp16 halves the ring "
                         "traffic and is measured behaviour-lossless on the 2B "
                         "(docs/serving/KV_STATE_DESCENT_20261003.md) but is refused until "
                         "b_gdn_recur carries a half-storage path, and until the SAE-survival "
                         "gate has run on the served lane")
    ap.add_argument("--spec-k", type=int, default=0, help="batched exact MTP speculation depth (0 = off)")
    ap.add_argument("--max-prompt", type=int, default=12288)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--default-max-new", type=int, default=1024)
    ap.add_argument("--max-queue", type=int, default=256)
    ap.add_argument("--model-name", default="qwen3.8-27b-georefine-bi")
    ap.add_argument("--api-key-file", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8290)
    a = ap.parse_args()
    if a.host not in LOOPBACK:
        raise SystemExit("loopback only")
    svc = Service(a)
    print(json.dumps({"status": "serving", "port": a.port, "meta": svc.meta}, default=str), flush=True)
    httpd = Server((a.host, a.port), make_handler(svc))
    httpd.serve_forever()


if __name__ == "__main__":
    main()
