"""Local-first OpenAI-compatible serving with serialized generation and a bounded queue.

This engine serializes requests against one model; it does not dynamically batch them.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import http.client
import ipaddress
import json
import math
import queue
import socket
import ssl
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence
from urllib.parse import urlsplit
import uuid


class RequestError(ValueError):
    """Invalid request, media, or generation parameters."""


@dataclass(frozen=True)
class GenerationOptions:
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int | None
    repetition_penalty: float
    seed: int | None
    stop: tuple[str, ...]
    do_sample: bool


def validate_generation_options(body: dict[str, Any], *, max_tokens_limit: int = 4096) -> GenerationOptions:
    def number(key, default, lo, hi, *, inclusive_lo=True, inclusive_hi=True):
        value = body.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RequestError(f"{key} must be numeric")
        value = float(value)
        if (not math.isfinite(value) or (value < lo if inclusive_lo else value <= lo)
                or (value > hi if inclusive_hi else value >= hi)):
            raise RequestError(f"{key} must be in {'[' if inclusive_lo else '('}{lo}, {hi}{']' if inclusive_hi else ')'}")
        return value

    max_tokens = body.get("max_tokens", body.get("max_completion_tokens", 256))
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= max_tokens_limit:
        raise RequestError(f"max_tokens must be an integer from 1 to {max_tokens_limit}")
    n = body.get("n", 1)
    if type(n) is not int or n != 1:
        raise RequestError("n must equal 1 for this serialized engine")
    temperature = number("temperature", 1.0, 0.0, 2.0)
    top_p = number("top_p", 1.0, 0.0, 1.0, inclusive_lo=False)
    repetition_penalty = number("repetition_penalty", 1.0, 0.0, 2.0, inclusive_lo=False)
    top_k = body.get("top_k")
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1):
        raise RequestError("top_k must be a positive integer")
    seed = body.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1):
        raise RequestError("seed must be an integer from 0 to 4294967295")
    stop = body.get("stop", ())
    if isinstance(stop, str):
        stop = (stop,)
    if not isinstance(stop, (list, tuple)) or any(not isinstance(x, str) or not x for x in stop):
        raise RequestError("stop must be a string or a list of nonempty strings")
    if len(stop) > 8 or any(len(x) > 256 for x in stop):
        raise RequestError("stop allows at most 8 strings of at most 256 characters")
    do_sample = temperature > 0
    return GenerationOptions(max_tokens, temperature, top_p, top_k,
                             repetition_penalty, seed, tuple(stop), do_sample)


def _host_allowed(host: str, allowlist: set[str]) -> bool:
    return host.casefold().rstrip(".") in allowlist


def _resolve_public_addresses(host: str, port: int) -> list[str]:
    try:
        literal = ipaddress.ip_address(host)
        addresses = [literal]
    except ValueError:
        try:
            addresses = [ipaddress.ip_address(item[4][0])
                         for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]
        except OSError as exc:
            raise RequestError("remote media host could not be resolved") from exc
    if not addresses or any(not addr.is_global for addr in addresses):
        raise RequestError("remote media host must resolve only to globally routable addresses")
    return [str(addr) for addr in addresses]


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection pinned to a previously checked public IP, with hostname TLS."""
    def __init__(self, hostname, address, port, timeout):
        super().__init__(hostname, port=port, timeout=timeout, context=ssl.create_default_context())
        self._pinned_address = address

    def connect(self):
        sock = socket.create_connection((self._pinned_address, self.port), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def _remote_media(url: str, allowlist: set[str], max_bytes: int) -> tuple[str, bytes]:
    parts = urlsplit(url)
    host = (parts.hostname or "").casefold().rstrip(".")
    if (parts.scheme != "https" or not host or parts.username or parts.password
            or parts.port not in (None, 443) or not _host_allowed(host, allowlist)
            or host.endswith((".localhost", ".local", ".internal"))
            or host in {"localhost", "metadata", "metadata.google.internal"}):
        raise RequestError("remote media URL is not allowed")
    addresses = _resolve_public_addresses(host, 443)
    conn = _PinnedHTTPSConnection(host, addresses[0], 443, timeout=5)
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    try:
        conn.request("GET", path, headers={"Host": host, "Accept": "image/*,video/*", "User-Agent": "BCTX-local-serving/1"})
        response = conn.getresponse()
        if 300 <= response.status < 400:
            raise RequestError("remote media redirects are disabled")
        if response.status != 200:
            raise RequestError("remote media fetch failed")
        length = response.getheader("Content-Length")
        if length is not None and (not length.isdigit() or int(length) > max_bytes):
            raise RequestError("remote media exceeds the size limit")
        mime = response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
        data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise RequestError("remote media exceeds the size limit")
        return mime, data
    except OSError as exc:
        raise RequestError("remote media fetch failed") from exc
    finally:
        conn.close()


def _decode_data_url(url: Any, *, media_kind: str, allowlist: set[str], max_bytes: int) -> bytes:
    if not isinstance(url, str):
        raise RequestError(f"{media_kind} URL must be a string")
    if url.startswith("data:"):
        try:
            head, encoded = url.split(",", 1)
            mime, *flags = head[5:].split(";")
            if flags != ["base64"]:
                raise RequestError("media data URLs must use base64 encoding")
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            if isinstance(exc, RequestError):
                raise
            raise RequestError(f"invalid base64 {media_kind} data URL") from exc
    elif url.startswith(("https://", "http://", "file://")):
        if not allowlist:
            raise RequestError("remote media URLs are disabled")
        mime, data = _remote_media(url, allowlist, max_bytes)
    else:
        raise RequestError(f"{media_kind} must use an inline base64 data URL")
    mime = mime.lower().strip()
    allowed = ({"image/png", "image/jpeg", "image/webp", "image/gif"}
               if media_kind == "image" else {"video/mp4", "video/webm", "video/quicktime"})
    if mime not in allowed:
        raise RequestError(f"unsupported {media_kind} media type")
    if not data or len(data) > max_bytes:
        raise RequestError(f"{media_kind} exceeds the size limit")
    return data


def _decode_image(data: bytes, max_pixels: int):
    try:
        from PIL import Image
        from io import BytesIO
        image = Image.open(BytesIO(data))
        if image.width <= 0 or image.height <= 0 or image.width * image.height > max_pixels:
            raise RequestError("image dimensions exceed the limit")
        image.load()
        return image.convert("RGB")
    except RequestError:
        raise
    except Exception as exc:
        raise RequestError("invalid image payload") from exc


class _VideoFrames(list):
    """Decoded frames plus presentation timestamps when PyAV exposes them."""
    def __init__(self, frames, timestamps):
        super().__init__(frames)
        self.timestamps = timestamps


def _validate_video_timestamps(timestamps, frame_count):
    if not isinstance(timestamps, (list, tuple)) or len(timestamps) != frame_count or frame_count == 0:
        raise RequestError("video presentation timestamps must cover every frame")
    result = []
    previous = None
    for value in timestamps:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RequestError("video presentation timestamps must be finite numbers")
        value = float(value)
        if value < 0 or previous is not None and value <= previous:
            raise RequestError("video presentation timestamps must be nonnegative and strictly ordered")
        result.append(value)
        previous = value
    microseconds = [int(round(value * 1_000_000)) for value in result]
    if any(b <= a for a, b in zip(microseconds, microseconds[1:])):
        raise RequestError("video timestamps are too close to preserve at microsecond precision")
    return result


def _video_template_kwargs(processor, messages):
    videos = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "video":
                frames = part.get("video")
                timestamps = _validate_video_timestamps(part.get("timestamps"), len(frames or ()))
                size = getattr(frames[0], "size", None) if frames else None
                width, height = size if isinstance(size, tuple) and len(size) == 2 else (None, None)
                indices = [int(round(ts * 1_000_000)) for ts in timestamps]
                videos.append((len(frames), width, height, indices))
    if not videos:
        return {}
    try:
        from transformers.video_utils import VideoMetadata
    except (ImportError, AttributeError) as exc:
        raise RequestError("processor runtime lacks explicit video metadata support") from exc
    metadata = [VideoMetadata(total_num_frames=count, fps=1_000_000,
                              width=width, height=height, frames_indices=indices)
                for count, width, height, indices in videos]
    return {"video_metadata": metadata, "do_sample_frames": False}


def _decode_video(data: bytes, *, max_frames: int, max_pixels: int, max_total_pixels: int):
    try:
        import av
        from io import BytesIO
        frames = []
        timestamps = []
        total_pixels = 0
        with av.open(BytesIO(data), mode="r") as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is None:
                raise RequestError("video contains no video stream")
            for frame in container.decode(stream):
                if len(frames) >= max_frames:
                    raise RequestError("video contains more frames than the configured limit")
                frame_pixels = frame.width * frame.height
                total_pixels += frame_pixels
                if frame_pixels > max_pixels or total_pixels > max_total_pixels:
                    raise RequestError("video frame dimensions exceed the limit")
                frames.append(frame.to_image().convert("RGB"))
                timestamp = (float(frame.pts * stream.time_base) if frame.pts is not None
                             else getattr(frame, "time", None))
                timestamps.append(float(timestamp) if timestamp is not None else None)
        if not frames:
            raise RequestError("video contains no decodable frames")
        return _VideoFrames(frames, timestamps)
    except RequestError:
        raise
    except ImportError as exc:
        raise RequestError("PyAV is required for inline video input") from exc
    except Exception as exc:
        raise RequestError("invalid video payload") from exc


def normalize_messages(messages: Any, *, remote_url_allowlist: Sequence[str] = (),
                       max_media_bytes: int = 20 << 20, max_image_pixels: int = 20_000_000,
                       max_video_frames: int = 16, max_video_pixels: int = 40_000_000,
                       image_decoder=None, video_decoder=None) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a nonempty array")
    allowlist = {str(host).casefold().rstrip(".") for host in remote_url_allowlist}
    result = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "developer", "user", "assistant", "tool"}:
            raise RequestError("each message needs a supported role")
        content = message.get("content")
        if content is None and message["role"] == "assistant" and isinstance(message.get("tool_calls"), list):
            normalized_content = ""
        elif isinstance(content, str):
            normalized_content: Any = content
        elif isinstance(content, list):
            normalized_content = []
            for part in content:
                if not isinstance(part, dict):
                    raise RequestError("message content parts must be objects")
                kind = part.get("type")
                if kind == "text" and isinstance(part.get("text"), str):
                    normalized_content.append({"type": "text", "text": part["text"]})
                elif kind in {"image_url", "image"}:
                    source = part.get("image_url", part.get("image"))
                    url = source.get("url") if isinstance(source, dict) else source
                    data = _decode_data_url(url, media_kind="image", allowlist=allowlist, max_bytes=max_media_bytes)
                    image = (image_decoder or (lambda d: _decode_image(d, max_image_pixels)))(data)
                    normalized_content.append({"type": "image", "image": image})
                elif kind in {"video_url", "video"}:
                    source = part.get("video_url", part.get("video"))
                    url = source.get("url") if isinstance(source, dict) else source
                    data = _decode_data_url(url, media_kind="video", allowlist=allowlist, max_bytes=max_media_bytes)
                    frames = (video_decoder or (lambda d: _decode_video(
                        d, max_frames=max_video_frames, max_pixels=max_image_pixels,
                        max_total_pixels=max_video_pixels)))(data)
                    timestamps = getattr(frames, "timestamps", None)
                    try:
                        frames = list(frames)
                    except TypeError as exc:
                        raise RequestError("video decoder must return an iterable of frames") from exc
                    if len(frames) > max_video_frames:
                        raise RequestError("video contains more frames than the configured limit")
                    video_part = {"type": "video", "video": frames}
                    if timestamps is None:
                        raise RequestError("video decoder must provide presentation timestamps for every frame")
                    video_part["timestamps"] = _validate_video_timestamps(timestamps, len(frames))
                    normalized_content.append(video_part)
                else:
                    raise RequestError("unsupported message content part")
        else:
            raise RequestError("message content must be text or an array of parts")
        item = {"role": message["role"], "content": normalized_content}
        for key in ("name", "tool_call_id", "tool_calls"):
            if key in message:
                item[key] = message[key]
        result.append(item)
    return result


def validate_chat_template_args(body: dict[str, Any]) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    tools = body.get("tools")
    if tools is not None and (not isinstance(tools, list) or any(not isinstance(t, dict) for t in tools)):
        raise RequestError("tools must be an array of objects")
    template_kwargs = body.get("chat_template_kwargs", {})
    if not isinstance(template_kwargs, dict):
        raise RequestError("chat_template_kwargs must be an object")
    reserved = {"tokenize", "add_generation_prompt", "return_dict", "return_tensors",
                "video_metadata", "do_sample_frames"}
    collisions = reserved.intersection(template_kwargs)
    if collisions:
        names = ", ".join(sorted(collisions))
        raise RequestError(f"chat_template_kwargs cannot override reserved argument(s): {names}")
    return tools, template_kwargs


def _next_or_end(iterator):
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


class BctxModelEngine:
    """HF generation wrapper whose weights are installed by the BCTX-only loader."""
    def __init__(self, model, processor, tokenizer, model_id: str, device: str):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.device = device
        self.model.eval()

    @classmethod
    def from_package(cls, package, *, device="cuda:0", attention_implementation="eager",
                     stride=1024, loader_workers=1, metadata_cache_dir=None, log=print):
        from scripts.bitexact_context_serving import load_bctx_model
        model, receipt = load_bctx_model(
            package, device=device, attention_implementation=attention_implementation,
            stride=stride, loader_workers=loader_workers,
            metadata_cache_dir=metadata_cache_dir, log=log,
        )
        try:
            from transformers import AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for chat serving") from exc
        metadata_dir = Path(receipt["metadata_dir"])
        processor = AutoProcessor.from_pretrained(str(metadata_dir), local_files_only=True)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(str(metadata_dir), local_files_only=True)
        return cls(model, processor, tokenizer, Path(package).name, device)

    def stream_chat(self, messages, options: GenerationOptions, *, cancel_event: threading.Event,
                    tools=None, chat_template_kwargs=None) -> Iterator[str]:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        if any(isinstance(part, dict) and part.get("type") in {"image", "video"}
               for msg in messages if isinstance(msg.get("content"), list) for part in msg["content"]):
            if self.processor is None:
                raise RequestError("model processor does not support image/video input")
        processor = self.processor or self.tokenizer
        template_kwargs = dict(chat_template_kwargs or {})
        if tools is not None:
            template_kwargs["tools"] = tools
        template_kwargs.update(_video_template_kwargs(processor, messages))
        template_messages = messages
        if self.processor is not None:
            # Qwen3VLProcessor's visual path expects structured content blocks
            # even for text-only turns. Keep the API-normalized messages intact
            # for tokenizer-only engines and copy before adapting processor input.
            template_messages = []
            for message in messages:
                item = dict(message)
                if isinstance(item.get("content"), str):
                    item["content"] = [{"type": "text", "text": item["content"]}]
                template_messages.append(item)
        encoded = processor.apply_chat_template(
            template_messages, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", **template_kwargs,
        )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in dict(encoded).items()}
        config = getattr(self.model, "config", None)
        text_config = getattr(config, "text_config", None)
        context_limit = getattr(text_config, "max_position_embeddings", None)
        if context_limit is None:
            context_limit = getattr(config, "max_position_embeddings", None)
        input_ids = inputs.get("input_ids")
        if context_limit is not None and input_ids is not None and hasattr(input_ids, "shape"):
            prompt_tokens = int(input_ids.shape[-1])
            if prompt_tokens + options.max_new_tokens > int(context_limit):
                raise RequestError(
                    f"prompt ({prompt_tokens} tokens) plus max_tokens ({options.max_new_tokens}) "
                    f"exceeds model context limit ({int(context_limit)})"
                )
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=0.25)
        generation_error: list[BaseException] = []
        finish_reason = ["stop"]

        class CancelOnDisconnect(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return cancel_event.is_set()

        kwargs = {
            **inputs,
            "max_new_tokens": options.max_new_tokens,
            "do_sample": options.do_sample,
            "streamer": streamer,
            "stopping_criteria": StoppingCriteriaList([CancelOnDisconnect()]),
        }
        if options.do_sample:
            kwargs.update(temperature=options.temperature, top_p=options.top_p)
            if options.top_k is not None:
                kwargs["top_k"] = options.top_k
        kwargs["repetition_penalty"] = options.repetition_penalty
        if options.stop:
            kwargs["stop_strings"] = list(options.stop)
            kwargs["tokenizer"] = self.tokenizer

        def generate():
            try:
                with torch.inference_mode():
                    if options.seed is None:
                        generated = self.model.generate(**kwargs)
                    else:
                        devices = [torch.device(self.device).index or 0] if str(self.device).startswith("cuda") else []
                        with torch.random.fork_rng(devices=devices):
                            torch.manual_seed(options.seed)
                            generated = self.model.generate(**kwargs)
                    sequences = getattr(generated, "sequences", generated)
                    if sequences is not None and hasattr(sequences, "shape"):
                        prompt_length = inputs.get("input_ids")
                        prompt_length = int(prompt_length.shape[-1]) if prompt_length is not None else 0
                        new_tokens = int(sequences.shape[-1]) - prompt_length
                        eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id",
                                      self.tokenizer.eos_token_id)
                        if isinstance(eos, int):
                            eos = {eos}
                        elif eos is None:
                            eos = set()
                        else:
                            eos = set(eos)
                        last_token = int(sequences[0, -1]) if sequences.shape[-1] else None
                        if new_tokens >= options.max_new_tokens and last_token not in eos:
                            finish_reason[0] = "length"
            except BaseException as exc:
                generation_error.append(exc)
            finally:
                # Ensure a generation exception cannot strand the HTTP stream.
                streamer.text_queue.put(streamer.stop_signal)

        worker = threading.Thread(target=generate, name="bctx-generate", daemon=True)
        worker.start()
        self.last_finish_reason = "stop"
        try:
            while True:
                if cancel_event.is_set():
                    break
                try:
                    text = streamer.text_queue.get(timeout=0.25)
                except queue.Empty:
                    yield None
                    continue
                if text == streamer.stop_signal:
                    break
                if text:
                    yield text
        finally:
            if cancel_event.is_set():
                pass
            # The caller holds the model lock until this iterator closes.
            worker.join()
        if generation_error:
            raise RuntimeError("model generation failed") from generation_error[0]
        self.last_finish_reason = finish_reason[0]


def create_app(*, engine=None, package: str | Path | None = None, model_id: str | None = None,
               device="cuda:0", attention_implementation="eager", stride=1024, loader_workers=1,
               metadata_cache_dir=None, remote_url_allowlist: Sequence[str] = (),
               max_request_bytes: int = 32 << 20, max_media_bytes: int = 20 << 20,
               max_image_pixels: int = 20_000_000, max_video_frames: int = 16,
               max_video_pixels: int = 40_000_000,
               max_pending_requests: int = 4, queue_timeout_seconds: float = 30.0,
               image_decoder=None, video_decoder=None):
    """Create the HTTP app; FastAPI is imported only when this function is called."""
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import StreamingResponse
    except ImportError as exc:
        raise RuntimeError("Install fastapi and uvicorn to serve a BCTX package") from exc
    if engine is None:
        if package is None:
            raise ValueError("provide a BCTX package or an injectable engine")
        engine = BctxModelEngine.from_package(
            package, device=device, attention_implementation=attention_implementation,
            stride=stride, loader_workers=loader_workers,
            metadata_cache_dir=metadata_cache_dir,
        )
    served_model = model_id or getattr(engine, "model_id", None) or (Path(package).name if package else "bctx-model")
    if (isinstance(max_pending_requests, bool) or max_pending_requests < 0 or max_pending_requests > 128
            or not math.isfinite(queue_timeout_seconds) or queue_timeout_seconds <= 0
            or max_request_bytes <= 0 or max_media_bytes <= 0 or max_image_pixels <= 0
            or max_video_frames < 1 or max_video_pixels <= 0):
        raise ValueError("request limits must be positive")
    app = FastAPI(title="BCTX local serving", docs_url=None, redoc_url=None)
    model_lock = threading.Lock()
    admission = threading.BoundedSemaphore(max_pending_requests + 1)
    remote_allowlist = tuple(remote_url_allowlist)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": served_model}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": served_model, "object": "model"}]}

    async def read_payload(request: Request):
        length = request.headers.get("content-length")
        if length is not None and (not length.isdigit() or int(length) > max_request_bytes):
            raise HTTPException(status_code=413, detail="request body too large")
        chunks = []
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > max_request_bytes:
                raise HTTPException(status_code=413, detail="request body too large")
            chunks.append(chunk)
        raw = b"".join(chunks)
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail="request body must be JSON") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="request body must be an object")
        if body.get("model") not in (None, served_model):
            raise HTTPException(status_code=400, detail="requested model does not match the loaded model")
        try:
            options = validate_generation_options(body)
            tools, template_kwargs = validate_chat_template_args(body)
            messages = normalize_messages(
                body.get("messages"), remote_url_allowlist=remote_allowlist,
                max_media_bytes=max_media_bytes, max_image_pixels=max_image_pixels,
                max_video_frames=max_video_frames, max_video_pixels=max_video_pixels,
                image_decoder=image_decoder,
                video_decoder=video_decoder,
            )
        except RequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return body, options, tools, template_kwargs, messages

    async def acquire_model_slot(request: Request):
        if not admission.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="generation queue is full")
        deadline = time.monotonic() + queue_timeout_seconds
        try:
            while not model_lock.acquire(blocking=False):
                if await request.is_disconnected():
                    raise HTTPException(status_code=499, detail="client disconnected")
                if time.monotonic() >= deadline:
                    raise HTTPException(status_code=503, detail="generation queue timed out")
                await asyncio.sleep(0.025)
        except BaseException:
            admission.release()
            raise
        return True

    def release_model_slot():
        model_lock.release()
        admission.release()

    def stream_iterator(messages, options, tools, template_kwargs, cancel_event):
        return iter(engine.stream_chat(messages, options, cancel_event=cancel_event,
                                       tools=tools, chat_template_kwargs=template_kwargs))

    async def chat_completions(request):
        body, options, tools, template_kwargs, messages = await read_payload(request)
        if "stream" in body and not isinstance(body["stream"], bool):
            raise HTTPException(status_code=400, detail="stream must be a boolean")
        stream = bool(body.get("stream", False))
        await acquire_model_slot(request)
        cancel_event = threading.Event()
        iterator = None

        if not stream:
            next_task = None
            try:
                iterator = stream_iterator(messages, options, tools, template_kwargs, cancel_event)
                chunks = []
                while True:
                    if await request.is_disconnected():
                        cancel_event.set()
                        break
                    next_task = asyncio.create_task(asyncio.to_thread(_next_or_end, iterator))
                    while not next_task.done():
                        if await request.is_disconnected():
                            cancel_event.set()
                            break
                        await asyncio.sleep(0.025)
                    has_item, item = await asyncio.shield(next_task)
                    next_task = None
                    if cancel_event.is_set():
                        break
                    if not has_item:
                        break
                    if item is not None:
                        chunks.append(item)
                text = "".join(chunks)
                if cancel_event.is_set():
                    raise HTTPException(status_code=499, detail="client disconnected")
                return {"id": "chatcmpl-" + uuid.uuid4().hex[:24], "object": "chat.completion",
                        "created": int(datetime.now(timezone.utc).timestamp()), "model": served_model,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                     "finish_reason": getattr(engine, "last_finish_reason", "stop")}]}
            except HTTPException:
                raise
            except RequestError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail="generation failed") from exc
            finally:
                import anyio
                with anyio.CancelScope(shield=True):
                    try:
                        if next_task is not None:
                            cancel_event.set()
                            try:
                                await next_task
                            except Exception:
                                # The result exception is handled by the request path above.
                                pass
                        if iterator is not None:
                            close = getattr(iterator, "close", None)
                            if callable(close):
                                await asyncio.to_thread(close)
                    finally:
                        release_model_slot()

        request_id = "chatcmpl-" + uuid.uuid4().hex[:24]

        async def events():
            nonlocal iterator
            complete = False
            next_task = None
            try:
                iterator = stream_iterator(messages, options, tools, template_kwargs, cancel_event)
                created = int(datetime.now(timezone.utc).timestamp())
                while True:
                    next_task = asyncio.create_task(asyncio.to_thread(_next_or_end, iterator))
                    has_item, item = await asyncio.shield(next_task)
                    next_task = None
                    if not has_item:
                        complete = True
                        break
                    if item is None:
                        yield ": keepalive\n\n"
                        continue
                    chunk = {"id": request_id, "object": "chat.completion.chunk", "created": created,
                             "model": served_model,
                             "choices": [{"index": 0, "delta": {"content": item}, "finish_reason": None}]}
                    yield "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
                if not cancel_event.is_set():
                    final = {"id": request_id, "object": "chat.completion.chunk", "created": created,
                             "model": served_model,
                             "choices": [{"index": 0, "delta": {},
                                          "finish_reason": getattr(engine, "last_finish_reason", "stop")}]}
                    yield "data: " + json.dumps(final) + "\n\n"
                    yield "data: [DONE]\n\n"
            except RequestError as exc:
                if not cancel_event.is_set():
                    error = {"error": {"message": str(exc), "type": "invalid_request_error"}}
                    yield "data: " + json.dumps(error) + "\n\n"
                    yield "data: [DONE]\n\n"
            except Exception:
                if not cancel_event.is_set():
                    error = {"error": {"message": "generation failed", "type": "server_error"}}
                    yield "data: " + json.dumps(error) + "\n\n"
                    yield "data: [DONE]\n\n"
            finally:
                if not complete:
                    cancel_event.set()
                import anyio
                with anyio.CancelScope(shield=True):
                    try:
                        if next_task is not None:
                            try:
                                await next_task
                            except Exception:
                                # The iterator error is handled by the SSE error path above.
                                pass
                        if iterator is not None:
                            close = getattr(iterator, "close", None)
                            if callable(close):
                                await asyncio.to_thread(close)
                    finally:
                        release_model_slot()

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    chat_completions.__annotations__["request"] = Request
    app.post("/v1/chat/completions")(chat_completions)

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id")
    parser.add_argument("--loader-workers", type=int, default=1)
    parser.add_argument("--metadata-cache-dir", type=Path)
    parser.add_argument("--max-pending-requests", type=int, default=4)
    parser.add_argument("--allow-remote-host", action="append", default=[],
                        help="exact HTTPS media host allowlist (remote media is off by default)")
    args = parser.parse_args(argv)
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install uvicorn to run the BCTX HTTP server") from exc
    app = create_app(package=args.package, device=args.device, model_id=args.model_id,
                     loader_workers=args.loader_workers,
                     metadata_cache_dir=args.metadata_cache_dir,
                     max_pending_requests=args.max_pending_requests,
                     remote_url_allowlist=args.allow_remote_host)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


__all__ = ["BctxModelEngine", "GenerationOptions", "RequestError", "create_app",
           "main", "normalize_messages", "validate_generation_options"]


if __name__ == "__main__":
    main()
