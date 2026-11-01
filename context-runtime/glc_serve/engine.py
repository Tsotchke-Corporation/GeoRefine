"""Wave-batched generation engine with per-token streaming.

WHY WAVES AND NOT CONTINUOUS BATCHING.  Qwen3.5/3.8 is a hybrid: 48 of 64
layers are Gated-DeltaNet (a recurrent state per sequence), 16 are full
attention.  transformers' cache for it supports index-selecting rows
(``reorder_cache``) but not inserting a new sequence mid-flight with its own
prefill, and continuous batching would need exactly that.  So requests that
arrive within ``batch_window_s`` and share sampling settings form a WAVE: one
left-padded prefill (images included), one ``generate`` call, tokens streamed
per row as they are produced; a request arriving mid-wave waits for the next
wave.  That is simple request batching, stated as such.

Everything the model sees goes through transformers' own ``generate`` (M-RoPE
positions, rope deltas, left padding, the hybrid cache), so this engine adds
scheduling and streaming, not model semantics.  The one exception is the MTP
speculative path (``glc_serve.mtp``), used only for a single greedy request.
"""
from __future__ import annotations

import base64
import io
import queue
import threading
import time
import urllib.request
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

MAX_IMAGE_BYTES = 20 << 20


class RequestError(ValueError):
    """A client error (HTTP 400)."""


@dataclass
class SamplingParams:
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    seed: Optional[int] = None
    stop: List[str] = field(default_factory=list)
    logprobs: bool = False
    top_logprobs: int = 0

    def wave_key(self) -> Tuple:
        greedy = self.temperature <= 0
        return (greedy, None if greedy else round(self.temperature, 6),
                None if greedy else round(self.top_p, 6),
                None if greedy else self.top_k)


@dataclass
class Request:
    kind: str                                   # "chat" | "completion" | "score"
    params: SamplingParams
    messages: Optional[List[Dict[str, Any]]] = None
    prompt: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)
    rid: str = field(default_factory=lambda: uuid.uuid4().hex[:24])
    out: "queue.Queue" = field(default_factory=queue.Queue)
    created: float = field(default_factory=time.time)
    t_submit: float = field(default_factory=time.perf_counter)
    # filled by the engine
    images: List[Any] = field(default_factory=list)
    text: str = ""
    prompt_tokens: int = 0
    gen_ids: List[int] = field(default_factory=list)
    sent_text: str = ""
    finished: bool = False
    finish_reason: Optional[str] = None
    t_first: Optional[float] = None
    t_done: Optional[float] = None
    spec_stats: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# multimodal inputs
# ---------------------------------------------------------------------------
def load_image(url: str, *, allow_http: bool = True):
    from PIL import Image

    if url.startswith("data:"):
        head, _, data = url.partition(",")
        if ";base64" not in head:
            raise RequestError("image data URL must be base64")
        raw = base64.b64decode(data)
    elif url.startswith(("http://", "https://")):
        if not allow_http:
            raise RequestError("http(s) image URLs are disabled on this server")
        with urllib.request.urlopen(url, timeout=30) as r:
            raw = r.read(MAX_IMAGE_BYTES + 1)
    else:
        raise RequestError("image_url must be a data: URL or http(s) URL")
    if len(raw) > MAX_IMAGE_BYTES:
        raise RequestError(f"image larger than {MAX_IMAGE_BYTES} bytes")
    img = Image.open(io.BytesIO(raw))
    return img.convert("RGB")


def normalize_messages(messages: List[Dict[str, Any]], *, allow_http_images: bool = True
                       ) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """OpenAI chat messages -> (HF chat-template messages, ordered PIL images)."""
    out: List[Dict[str, Any]] = []
    images: List[Any] = []
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty list")
    for m in messages:
        if not isinstance(m, dict) or "role" not in m:
            raise RequestError("each message needs a role")
        msg = {k: v for k, v in m.items() if k not in ("content",)}
        content = m.get("content")
        if content is None or isinstance(content, str):
            msg["content"] = content if content is not None else ""
        elif isinstance(content, list):
            parts = []
            for p in content:
                ptype = p.get("type") if isinstance(p, dict) else None
                if ptype == "text":
                    parts.append({"type": "text", "text": str(p.get("text", ""))})
                elif ptype == "image_url":
                    iu = p.get("image_url")
                    url = iu.get("url") if isinstance(iu, dict) else iu
                    if not isinstance(url, str):
                        raise RequestError("image_url.url must be a string")
                    images.append(load_image(url, allow_http=allow_http_images))
                    parts.append({"type": "image"})
                elif ptype in ("video_url", "video", "input_audio"):
                    raise RequestError(
                        f"content part {ptype!r} is not supported by this server build "
                        "(images are; video input is not wired)")
                else:
                    raise RequestError(f"unsupported content part {ptype!r}")
            msg["content"] = parts
        else:
            raise RequestError("content must be a string or a list of parts")
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = dict(tc.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    import json

                    try:
                        fn["arguments"] = json.loads(args) if args else {}
                    except ValueError:
                        fn["arguments"] = {"raw": args}
                calls.append({"type": "function", "function": fn})
            msg["tool_calls"] = calls
        out.append(msg)
    return out, images


def parse_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Split Qwen3.5 XML tool calls out of generated text.

    Format (from the checkpoint's chat_template.jinja)::

        <tool_call>
        <function=NAME>
        <parameter=P>
        VALUE
        </parameter>
        </function>
        </tool_call>
    """
    import json
    import re

    calls = []
    start = text.find("<tool_call>")
    if start < 0:
        return text, []
    content = text[:start].rstrip()
    for block in re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", text[start:], flags=re.S):
        fm = re.search(r"<function=([^>\n]+)>", block)
        if not fm:
            continue
        args: Dict[str, Any] = {}
        for pm in re.finditer(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", block, flags=re.S):
            raw = pm.group(2)
            try:
                args[pm.group(1).strip()] = json.loads(raw)
            except ValueError:
                args[pm.group(1).strip()] = raw
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:24], "type": "function",
            "function": {"name": fm.group(1).strip(),
                         "arguments": json.dumps(args, ensure_ascii=False)},
        })
    return content, calls


# ---------------------------------------------------------------------------
# streaming plumbing
# ---------------------------------------------------------------------------
class _Observer:
    """LogitsProcessor that keeps the step's log-probabilities (read-only)."""

    def __init__(self):
        self.last: Optional[torch.Tensor] = None

    def __call__(self, input_ids, scores):
        self.last = torch.log_softmax(scores.float(), dim=-1)
        return scores


class _RowStop:
    def __init__(self, finished: List[bool]):
        self.finished = finished

    def __call__(self, input_ids, scores, **kwargs):
        return torch.tensor(self.finished, dtype=torch.bool, device=input_ids.device)


class TokenSink:
    """Per-request incremental detokenization, stop strings, events."""

    def __init__(self, tokenizer, eos_ids: Sequence[int]):
        self.tok = tokenizer
        self.eos = set(int(e) for e in eos_ids)

    def push(self, req: Request, token: int, lp: Optional[Dict[str, Any]] = None) -> bool:
        """Consume one token; returns True while the request wants more."""
        if req.finished:
            return False
        now = time.perf_counter()
        if req.t_first is None:
            req.t_first = now
        if token in self.eos:
            self.finish(req, "stop")
            return False
        req.gen_ids.append(int(token))
        text = self.tok.decode(req.gen_ids, skip_special_tokens=False)
        stop_hit = None
        for s in req.params.stop:
            if s and s in text[max(0, len(req.sent_text) - len(s)):]:
                idx = text.find(s, max(0, len(req.sent_text) - len(s)))
                if stop_hit is None or idx < stop_hit:
                    stop_hit = idx
        if stop_hit is not None:
            text = text[:stop_hit]
        if text.endswith("�") and stop_hit is None:
            delta = ""   # hold a partial UTF-8 sequence until it completes
        else:
            delta = text[len(req.sent_text):] if text.startswith(req.sent_text) else ""
            req.sent_text = text if delta or text == req.sent_text else req.sent_text
        req.out.put(("delta", delta, int(token), lp))
        if stop_hit is not None:
            self.finish(req, "stop")
            return False
        if len(req.gen_ids) >= int(req.params.max_tokens):
            self.finish(req, "length")
            return False
        return True

    def finish(self, req: Request, reason: str) -> None:
        if req.finished:
            return
        full = self.tok.decode(req.gen_ids, skip_special_tokens=False)
        for s in req.params.stop:
            if s and s in full:
                full = full[: full.find(s)]
        if len(full) > len(req.sent_text) and full.startswith(req.sent_text) and \
                not full.endswith("�"):
            req.out.put(("delta", full[len(req.sent_text):], None, None))
            req.sent_text = full
        req.text = req.sent_text
        req.finished = True
        req.finish_reason = reason
        req.t_done = time.perf_counter()
        req.out.put(("done", reason, self.usage(req), req.spec_stats))

    @staticmethod
    def usage(req: Request) -> Dict[str, Any]:
        t_first = req.t_first or req.t_done or time.perf_counter()
        t_done = req.t_done or time.perf_counter()
        n = len(req.gen_ids)
        return {
            "prompt_tokens": int(req.prompt_tokens),
            "completion_tokens": n,
            "total_tokens": int(req.prompt_tokens) + n,
            "x_timing": {
                "queue_to_first_token_s": round(t_first - req.t_submit, 6),
                "decode_s": round(t_done - t_first, 6),
                "decode_tok_s": round((n - 1) / (t_done - t_first), 3)
                if n > 1 and t_done > t_first else None,
            },
        }


class _WaveStreamer:
    """transformers BaseStreamer adapter: row i of each step -> request i."""

    def __init__(self, reqs: List[Request], sink: TokenSink, finished: List[bool],
                 observer: Optional[_Observer], top_k: int):
        self.reqs = reqs
        self.sink = sink
        self.finished = finished
        self.observer = observer
        self.top_k = top_k
        self._prompt_seen = False

    def put(self, value):
        if not self._prompt_seen:
            self._prompt_seen = True
            return
        toks = value.reshape(-1).tolist()
        for i, (req, tok) in enumerate(zip(self.reqs, toks)):
            if self.finished[i]:
                continue
            lp = None
            if self.observer is not None and self.observer.last is not None and \
                    req.params.logprobs:
                row = self.observer.last[i]
                k = max(0, int(req.params.top_logprobs))
                top = torch.topk(row, k) if k else None
                lp = {"logprob": float(row[tok].item()),
                      "top": [] if top is None else list(zip(top.indices.tolist(),
                                                             top.values.tolist()))}
            if not self.sink.push(req, int(tok), lp):
                self.finished[i] = True

    def end(self):
        for i, req in enumerate(self.reqs):
            if not req.finished:
                self.sink.finish(req, "length" if len(req.gen_ids) >= req.params.max_tokens
                                 else "stop")
            self.finished[i] = True


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------
class Engine:
    def __init__(self, loaded, processor, tokenizer, *, max_batch: int = 16,
                 batch_window_s: float = 0.01, enable_mtp: bool = False,
                 max_input_tokens: Optional[int] = None, allow_http_images: bool = True,
                 default_chat_template_kwargs: Optional[Dict[str, Any]] = None,
                 prefill_chunk_tokens: int = 4096,
                 log: Callable[[str], None] = print):
        self.loaded = loaded
        self.model = loaded.model
        self.processor = processor
        self.tok = tokenizer
        self.max_batch = int(max_batch)
        self.window = float(batch_window_s)
        self.enable_mtp = bool(enable_mtp) and loaded.mtp is not None
        self.max_input_tokens = max_input_tokens
        self.allow_http_images = bool(allow_http_images)
        self.default_chat_kwargs = dict(default_chat_template_kwargs or {})
        self.prefill_chunk = max(0, int(prefill_chunk_tokens))
        self.log = log
        gc = getattr(self.model, "generation_config", None)
        eos = getattr(gc, "eos_token_id", None) if gc is not None else None
        if eos is None:
            eos = tokenizer.eos_token_id
        self.eos_ids = [int(e) for e in (eos if isinstance(eos, (list, tuple)) else [eos])
                        if e is not None]
        pad = getattr(gc, "pad_token_id", None) if gc is not None else None
        self.pad_id = int(pad if pad is not None else (tokenizer.pad_token_id
                                                       if tokenizer.pad_token_id is not None
                                                       else self.eos_ids[0]))
        self.sink = TokenSink(tokenizer, self.eos_ids)
        self._q: "queue.Queue[Request]" = queue.Queue()
        self._held: deque = deque()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.stats = {"waves": 0, "requests": 0, "spec_requests": 0, "max_wave": 0}
        self.device = self._model_device()

    def _model_device(self) -> torch.device:
        for p in self.model.parameters():
            if not p.is_meta:
                return p.device
        return torch.device("cpu")

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> "Engine":
        self._thread = threading.Thread(target=self._loop, name="glc-serve-engine", daemon=True)
        self._thread.start()
        return self

    def shutdown(self) -> None:
        self._stop.set()
        self._q.put(None)  # type: ignore[arg-type]
        if self._thread is not None:
            self._thread.join(timeout=10)

    def submit(self, req: Request) -> Request:
        if req.kind == "chat":
            req.messages, req.images = normalize_messages(
                req.messages or [], allow_http_images=self.allow_http_images)
        self._q.put(req)
        return req

    # -- input preparation -----------------------------------------------------
    def render(self, req: Request) -> str:
        if req.kind == "chat":
            kw = {**self.default_chat_kwargs, **(req.chat_template_kwargs or {})}
            templ = self.processor if hasattr(self.processor, "apply_chat_template") else self.tok
            return templ.apply_chat_template(
                req.messages, tools=req.tools, tokenize=False, add_generation_prompt=True, **kw)
        return str(req.prompt or "")

    def prepare(self, reqs: List[Request]) -> Dict[str, torch.Tensor]:
        texts = [self.render(r) for r in reqs]
        images = [im for r in reqs for im in r.images]
        tok = self.tok
        old_side = getattr(tok, "padding_side", "right")
        tok.padding_side = "left"
        try:
            if images:
                if self.processor is None or not hasattr(self.processor, "image_processor"):
                    raise RequestError("this model/server has no image processor")
                self.processor.tokenizer.padding_side = "left"
                enc = self.processor(text=texts, images=images, padding=True,
                                     return_tensors="pt")
            elif self.processor is not None and hasattr(self.processor, "image_processor"):
                self.processor.tokenizer.padding_side = "left"
                enc = self.processor(text=texts, padding=True, return_tensors="pt")
            else:
                enc = tok(texts, padding=True, return_tensors="pt", add_special_tokens=False)
        finally:
            tok.padding_side = old_side
        enc = {k: v for k, v in dict(enc).items() if isinstance(v, torch.Tensor)}
        mask = enc.get("attention_mask")
        for i, r in enumerate(reqs):
            r.text = texts[i]
            r.prompt_tokens = int(mask[i].sum().item()) if mask is not None else int(
                enc["input_ids"].shape[1])
            if self.max_input_tokens and r.prompt_tokens > self.max_input_tokens:
                raise RequestError(
                    f"prompt has {r.prompt_tokens} tokens > max_input_tokens "
                    f"{self.max_input_tokens}")
        return {k: v.to(self.device) for k, v in enc.items()}

    # -- the loop ------------------------------------------------------------
    def _next_wave(self) -> List[Request]:
        first = self._held.popleft() if self._held else self._q.get()
        if first is None:
            return []
        wave = [first]
        if first.kind == "score":
            return wave
        key = first.params.wave_key()
        deadline = time.perf_counter() + self.window
        while len(wave) < self.max_batch:
            timeout = deadline - time.perf_counter()
            try:
                nxt = self._q.get(timeout=max(0.0, timeout)) if timeout > 0 \
                    else self._q.get_nowait()
            except queue.Empty:
                break
            if nxt is None:
                self._stop.set()
                break
            if nxt.kind == "score" or nxt.params.wave_key() != key:
                self._held.append(nxt)
                continue
            wave.append(nxt)
        return wave

    def _loop(self) -> None:
        while not self._stop.is_set():
            wave = self._next_wave()
            if not wave:
                continue
            try:
                self.run_wave(wave)
            except RequestError as exc:
                for r in wave:
                    if not r.finished:
                        r.finished = True
                        r.out.put(("error", 400, str(exc)))
            except Exception as exc:  # engine errors surface to every waiter
                import traceback

                self.log(f"[engine] wave failed: {type(exc).__name__}: {exc}\n"
                         f"{traceback.format_exc()}")
                for r in wave:
                    if not r.finished:
                        r.finished = True
                        r.out.put(("error", 500, f"{type(exc).__name__}: {exc}"))

    @torch.no_grad()
    def run_wave(self, wave: List[Request]) -> None:
        self.stats["waves"] += 1
        self.stats["requests"] += len(wave)
        self.stats["max_wave"] = max(self.stats["max_wave"], len(wave))
        if wave[0].kind == "score":
            self._score(wave[0])
            return
        inputs = self.prepare(wave)
        p = wave[0].params
        seq = int(inputs["input_ids"].shape[1])
        long_text = (len(wave) == 1 and not wave[0].images and self.prefill_chunk > 0
                     and seq > self.prefill_chunk)
        if (self.enable_mtp and len(wave) == 1 and p.temperature <= 0
                and not p.logprobs and not long_text):
            self._run_spec(wave[0], inputs)
            return
        cache = self._chunked_prefill(inputs) if long_text else None
        self._run_generate(wave, inputs, cache=cache)

    def _chunked_prefill(self, inputs: Dict[str, torch.Tensor]):
        """Prefill all but the last prompt token in fixed chunks.

        Bounds the activation peak of a long prompt to one chunk (the KV and
        recurrent state still grow with the prompt).  ``generate`` then sees a
        cache that already holds ``seq - 1`` tokens and processes only the
        tail -- verified on CPU to give token-identical output to an
        unchunked prefill (tests/test_glc_serve_mtp.py).
        """
        from .mtp import new_cache

        ids = inputs["input_ids"]
        cfg = self.model.config
        cache = new_cache(cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg)
        end = int(ids.shape[1]) - 1
        for s in range(0, end, self.prefill_chunk):
            e = min(end, s + self.prefill_chunk)
            self.model(input_ids=ids[:, s:e], past_key_values=cache, use_cache=True,
                       logits_to_keep=1)
        self.stats["chunked_prefills"] = self.stats.get("chunked_prefills", 0) + 1
        return cache

    def _generation_config(self, reqs: List[Request]):
        from transformers import GenerationConfig

        base = getattr(self.model, "generation_config", None)
        cfg = GenerationConfig.from_dict(base.to_dict()) if base is not None else GenerationConfig()
        p = reqs[0].params
        cfg.max_new_tokens = max(int(r.params.max_tokens) for r in reqs) + 1
        cfg.eos_token_id = self.eos_ids
        cfg.pad_token_id = self.pad_id
        if p.temperature <= 0:
            cfg.do_sample = False
            cfg.temperature = 1.0
            cfg.top_p = 1.0
            cfg.top_k = None
        else:
            cfg.do_sample = True
            cfg.temperature = float(p.temperature)
            cfg.top_p = float(p.top_p)
            if p.top_k is not None:
                cfg.top_k = int(p.top_k) if p.top_k > 0 else None
        cfg.use_cache = True
        return cfg

    def _run_generate(self, reqs: List[Request], inputs: Dict[str, torch.Tensor],
                      *, cache=None) -> None:
        from transformers import LogitsProcessorList, StoppingCriteriaList

        finished = [False] * len(reqs)
        want_lp = any(r.params.logprobs for r in reqs)
        observer = _Observer() if want_lp else None
        top_k = max((r.params.top_logprobs for r in reqs), default=0)
        streamer = _WaveStreamer(reqs, self.sink, finished, observer, top_k)
        seed = reqs[0].params.seed
        if seed is not None and len(reqs) == 1:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        kwargs = dict(inputs)
        kwargs.update(
            generation_config=self._generation_config(reqs),
            streamer=streamer,
            stopping_criteria=StoppingCriteriaList([_RowStop(finished)]),
        )
        if observer is not None:
            kwargs["logits_processor"] = LogitsProcessorList([observer])
        if cache is not None:
            kwargs["past_key_values"] = cache
        self.model.generate(**kwargs)
        streamer.end()

    def _run_spec(self, req: Request, inputs: Dict[str, torch.Tensor]) -> None:
        from .mtp import MTPSpeculator

        self.stats["spec_requests"] += 1
        spec = MTPSpeculator(self.model, self.loaded.mtp)
        stats = spec.generate(
            {k: v for k, v in inputs.items() if k != "attention_mask"}
            if int(inputs["input_ids"].shape[0]) == 1 else inputs,
            max_new_tokens=req.params.max_tokens, eos_ids=self.eos_ids,
            on_token=lambda t: self.sink.push(req, int(t)),
        )
        req.spec_stats = stats
        if not req.finished:
            self.sink.finish(req, "length" if len(req.gen_ids) >= req.params.max_tokens
                             else "stop")

    def _score(self, req: Request) -> None:
        """Prompt log-probabilities (``/v1/completions`` with ``echo``)."""
        enc = self.tok([req.prompt or ""], return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(self.device)
        req.prompt_tokens = int(ids.shape[1])
        out = self.model(input_ids=ids, use_cache=False, logits_to_keep=0)
        logits = out.logits[0].float()
        lps = torch.log_softmax(logits, dim=-1)
        k = max(0, int(req.params.top_logprobs))
        tokens = ids[0].tolist()
        items = [{"token_id": tokens[0], "logprob": None, "top": []}]
        for i in range(1, len(tokens)):
            row = lps[i - 1]
            top = torch.topk(row, k) if k else None
            items.append({"token_id": tokens[i], "logprob": float(row[tokens[i]].item()),
                          "top": [] if top is None else list(zip(top.indices.tolist(),
                                                                 top.values.tolist()))})
        req.finished = True
        req.finish_reason = "length"
        req.out.put(("score", items, {"prompt_tokens": len(tokens), "completion_tokens": 0,
                                       "total_tokens": len(tokens)}))


__all__ = [
    "Engine",
    "Request",
    "RequestError",
    "SamplingParams",
    "TokenSink",
    "load_image",
    "normalize_messages",
    "parse_tool_calls",
]
