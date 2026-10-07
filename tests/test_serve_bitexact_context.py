import asyncio
import base64
import concurrent.futures
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from scripts.serve_bitexact_context import BctxModelEngine, GenerationOptions, RequestError, create_app


class FakeEngine:
    model_id = "bctx-test"

    def __init__(self, *, delay=0.0, long_stream=False, fail_once=False):
        self.delay = delay
        self.long_stream = long_stream
        self.fail_once = fail_once
        self.calls = []
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.cancelled = threading.Event()
        self.started = threading.Event()
        self._long_stream_started = False

    def stream_chat(self, messages, options, *, cancel_event, tools=None, chat_template_kwargs=None):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append({"messages": messages, "options": options, "tools": tools,
                               "chat_template_kwargs": chat_template_kwargs})
            long_this_call = self.long_stream and not self._long_stream_started
            if long_this_call:
                self._long_stream_started = True
            self.started.set()
        try:
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("injected generation failure")
            count = 1000 if long_this_call else 3
            for i in range(count):
                if cancel_event.wait(self.delay):
                    self.cancelled.set()
                    return
                yield f"chunk{i}"
        finally:
            with self._lock:
                self.active -= 1


def _app(engine, **kwargs):
    return create_app(engine=engine, **kwargs)


def _request(messages=None, **kwargs):
    return {"model": "bctx-test", "messages": messages or [{"role": "user", "content": "hello"}], **kwargs}


def test_health_and_models_disclose_only_ready_and_model():
    client = TestClient(_app(FakeEngine()))
    health = client.get("/health").json()
    assert health == {"status": "ok", "model": "bctx-test"}
    assert client.get("/v1/models").json()["data"][0]["id"] == "bctx-test"


def test_text_completion_validates_and_passes_tools_and_template_kwargs():
    engine = FakeEngine()
    client = TestClient(_app(engine))
    response = client.post("/v1/chat/completions", json=_request(
        max_tokens=7, temperature=0, top_p=0.8, tools=[{"type": "function", "function": {"name": "lookup"}}],
        chat_template_kwargs={"enable_thinking": False},
    ))
    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "chunk0chunk1chunk2"
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert engine.calls[0]["options"].do_sample is False
    assert engine.calls[0]["options"].max_new_tokens == 7
    assert engine.calls[0]["tools"][0]["function"]["name"] == "lookup"
    assert engine.calls[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_sse_stream_assembles_openai_chunks():
    client = TestClient(_app(FakeEngine()))
    with client.stream("POST", "/v1/chat/completions", json=_request(stream=True)) as response:
        assert response.status_code == 200
        lines = [line for line in response.iter_lines() if line.startswith("data: ")]
    objects = [json.loads(line[6:]) for line in lines if line != "data: [DONE]"]
    assert "".join(item["choices"][0].get("delta", {}).get("content", "") for item in objects) == "chunk0chunk1chunk2"
    assert any(item["choices"][0].get("finish_reason") == "stop" for item in objects)
    assert lines[-1] == "data: [DONE]"


def test_sse_stream_sends_heartbeat_while_engine_is_waiting():
    class HeartbeatEngine(FakeEngine):
        def stream_chat(self, messages, options, *, cancel_event, tools=None, chat_template_kwargs=None):
            yield None
            yield "after-heartbeat"

    client = TestClient(_app(HeartbeatEngine()))
    with client.stream("POST", "/v1/chat/completions", json=_request(stream=True)) as response:
        body = "".join(response.iter_text())
    assert ": keepalive\n\n" in body
    assert "after-heartbeat" in body


def test_completion_uses_engine_finish_reason():
    class LengthEngine(FakeEngine):
        def stream_chat(self, messages, options, *, cancel_event, tools=None, chat_template_kwargs=None):
            self.last_finish_reason = "length"
            yield "truncated"

    client = TestClient(_app(LengthEngine()))
    response = client.post("/v1/chat/completions", json=_request(max_tokens=1))
    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "length"


def test_request_body_limit_is_enforced_while_streaming():
    engine = FakeEngine()
    client = TestClient(_app(engine, max_request_bytes=64))
    response = client.post("/v1/chat/completions", content=b"{" + b" " * 80)
    assert response.status_code == 413
    assert engine.calls == []


@pytest.mark.parametrize("key,value", [
    ("tokenize", False), ("add_generation_prompt", False),
    ("return_dict", False), ("return_tensors", None),
    ("video_metadata", []), ("do_sample_frames", True),
])
def test_chat_template_kwargs_cannot_override_reserved_arguments(key, value):
    client = TestClient(_app(FakeEngine()))
    response = client.post("/v1/chat/completions", json=_request(chat_template_kwargs={key: value}))
    assert response.status_code == 400, response.text
    assert "cannot override reserved" in response.json()["detail"]


def test_tokenized_prompt_plus_generation_limit_fails_before_model_generate():
    import torch
    from types import SimpleNamespace

    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": torch.zeros((1, 6), dtype=torch.long)}

    class Model:
        config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=8))

        def eval(self):
            return self

        def generate(self, **kwargs):
            pytest.fail("generation must not start when the request exceeds context")

    engine = BctxModelEngine(Model(), Processor(), object(), "bctx-test", "cpu")
    options = GenerationOptions(3, 0, 1, None, 1, None, (), False)
    iterator = engine.stream_chat([], options, cancel_event=threading.Event())
    with pytest.raises(RequestError, match=r"prompt \(6 tokens\).*context limit \(8\)"):
        next(iterator)


def test_context_limit_request_error_is_clear_for_json_and_sse():
    class ContextErrorEngine(FakeEngine):
        def stream_chat(self, messages, options, *, cancel_event, tools=None, chat_template_kwargs=None):
            raise RequestError("prompt plus max_tokens exceeds model context limit")
            yield

    client = TestClient(_app(ContextErrorEngine()), raise_server_exceptions=False)
    response = client.post("/v1/chat/completions", json=_request())
    assert response.status_code == 400
    assert "exceeds model context limit" in response.json()["detail"]

    with client.stream("POST", "/v1/chat/completions", json=_request(stream=True)) as stream:
        body = "".join(stream.iter_text())
    assert '"type": "invalid_request_error"' in body
    assert "exceeds model context limit" in body


def test_inline_image_and_video_are_decoded_and_remote_file_urls_rejected():
    image_seen = object()
    video_seen = object()
    engine = FakeEngine()
    class TimedFrames(list):
        timestamps = [0.0]

    client = TestClient(_app(engine, image_decoder=lambda raw: image_seen,
                             video_decoder=lambda raw: TimedFrames([video_seen])))
    img = "data:image/png;base64," + base64.b64encode(b"image-bytes").decode()
    vid = "data:video/mp4;base64," + base64.b64encode(b"video-bytes").decode()
    response = client.post("/v1/chat/completions", json=_request(messages=[{"role": "user", "content": [
        {"type": "text", "text": "describe these"},
        {"type": "image_url", "image_url": {"url": img}},
        {"type": "video_url", "video_url": {"url": vid}},
    ]}]))
    assert response.status_code == 200
    parts = engine.calls[0]["messages"][0]["content"]
    assert parts[1] == {"type": "image", "image": image_seen}
    assert parts[2] == {"type": "video", "video": [video_seen], "timestamps": [0.0]}
    bad = client.post("/v1/chat/completions", json=_request(messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "file:///etc/passwd"}}
    ]}]))
    assert bad.status_code == 400
    assert "remote media URLs are disabled" in bad.json()["detail"]
    remote = client.post("/v1/chat/completions", json=_request(messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.invalid/image.png"}}
    ]}]))
    assert remote.status_code == 400
    assert "remote media URLs are disabled" in remote.json()["detail"]


def test_video_presentation_timestamps_reach_the_processor():
    class TimedFrames(list):
        timestamps = [0.0, 0.5]

    engine = FakeEngine()
    client = _app(engine, video_decoder=lambda raw: TimedFrames(["frame0", "frame1"]))
    payload = "data:video/mp4;base64," + base64.b64encode(b"video-bytes").decode()
    response = TestClient(client).post("/v1/chat/completions", json=_request(messages=[{
        "role": "user", "content": [{"type": "video", "video": payload}]
    }]))
    assert response.status_code == 200
    assert engine.calls[0]["messages"][0]["content"][0] == {
        "type": "video", "video": ["frame0", "frame1"], "timestamps": [0.0, 0.5]
    }


def test_video_metadata_and_sampling_flags_reach_actual_processor_kwargs():
    import torch
    from types import SimpleNamespace

    class Frame:
        size = (32, 24)

    class TimedFrames(list):
        timestamps = [0.0, 0.125, 0.6]

    class Processor:
        def __init__(self):
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = kwargs
            return {"input_ids": torch.zeros((1, 1), dtype=torch.long)}

    class Tokenizer:
        eos_token_id = None

        def decode(self, ids, **kwargs):
            return "ok"

    class Model:
        config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=16))
        generation_config = SimpleNamespace(eos_token_id=None)

        def eval(self):
            return self

        def generate(self, **kwargs):
            kwargs["streamer"].put(torch.tensor([[1]], dtype=torch.long))
            kwargs["streamer"].end()
            return SimpleNamespace(sequences=torch.tensor([[0, 1]]))

    processor = Processor()
    engine = BctxModelEngine(Model(), processor, Tokenizer(), "bctx-test", "cpu")
    messages = [{"role": "user", "content": [{"type": "video", "video": list(TimedFrames(Frame() for _ in range(3))),
                                                      "timestamps": [0.0, 0.125, 0.6]}]}]
    list(engine.stream_chat(messages, GenerationOptions(1, 0, 1, None, 1, None, (), False),
                            cancel_event=threading.Event(),
                            chat_template_kwargs={"enable_thinking": False}))
    assert processor.kwargs["video_metadata"][0].total_num_frames == 3
    assert processor.kwargs["video_metadata"][0].fps == 1_000_000
    assert processor.kwargs["video_metadata"][0].frames_indices == [0, 125_000, 600_000]
    assert processor.kwargs["video_metadata"][0].width == 32
    assert processor.kwargs["video_metadata"][0].height == 24
    assert processor.kwargs["do_sample_frames"] is False
    assert processor.kwargs["enable_thinking"] is False


def test_processor_receives_structured_text_content_and_message_metadata_is_preserved():
    import torch
    from types import SimpleNamespace

    class Processor:
        def __init__(self):
            self.messages = None

        def apply_chat_template(self, messages, **kwargs):
            assert all(not isinstance(m.get("content"), str) for m in messages)
            self.messages = messages
            return {"input_ids": torch.zeros((1, 1), dtype=torch.long)}

    class Tokenizer:
        eos_token_id = None

        def decode(self, ids, **kwargs):
            return "ok"

    class Model:
        config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=16))
        generation_config = SimpleNamespace(eos_token_id=None)

        def eval(self):
            return self

        def generate(self, **kwargs):
            kwargs["streamer"].put(torch.tensor([[1]], dtype=torch.long))
            kwargs["streamer"].end()
            return SimpleNamespace(sequences=torch.tensor([[0, 1]]))

    processor = Processor()
    engine = BctxModelEngine(Model(), processor, Tokenizer(), "bctx-test", "cpu")
    original = [{"role": "assistant", "content": "hello", "tool_calls": [{"id": "call-1"}]},
                {"role": "tool", "content": "world", "tool_call_id": "call-1", "name": "lookup"}]
    list(engine.stream_chat(original, GenerationOptions(1, 0, 1, None, 1, None, (), False),
                            cancel_event=threading.Event()))
    assert processor.messages == [
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}],
         "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "content": [{"type": "text", "text": "world"}],
         "tool_call_id": "call-1", "name": "lookup"},
    ]
    assert original[0]["content"] == "hello"
    assert original[1]["content"] == "world"


@pytest.mark.parametrize("timestamps", [None, [0.0, None], [0.5, 0.4], [0.0, float("nan")]])
def test_video_timestamp_missing_invalid_or_unordered_rejected(timestamps):
    class TimedFrames(list):
        pass

    frames = TimedFrames(["a", "b"])
    frames.timestamps = timestamps
    payload = "data:video/mp4;base64," + base64.b64encode(b"video-bytes").decode()
    response = TestClient(_app(FakeEngine(), video_decoder=lambda raw: frames)).post(
        "/v1/chat/completions", json=_request(messages=[{
            "role": "user", "content": [{"type": "video", "video": payload}]
        }]))
    assert response.status_code == 400
    assert "timestamp" in response.json()["detail"]


def test_video_frame_limit_rejects_extra_frames_instead_of_truncating():
    class TimedFrames(list):
        timestamps = list(range(17))

    payload = "data:video/mp4;base64," + base64.b64encode(b"video-bytes").decode()
    response = TestClient(_app(FakeEngine(), max_video_frames=16,
                               video_decoder=lambda raw: TimedFrames(range(17)))).post(
        "/v1/chat/completions", json=_request(messages=[{
            "role": "user", "content": [{"type": "video", "video": payload}]
        }]))
    assert response.status_code == 400
    assert "more frames" in response.json()["detail"]


def test_decode_video_frame_limit_rejects_the_first_excess_frame(monkeypatch):
    import sys
    from types import SimpleNamespace
    from scripts.serve_bitexact_context import _decode_video

    class Image:
        def convert(self, mode):
            return self

    class Frame:
        width = height = 1

        def __init__(self, pts):
            self.pts = pts

        def to_image(self):
            return Image()

    class Container:
        streams = [SimpleNamespace(type="video", time_base=1)]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def decode(self, stream):
            yield Frame(0)
            yield Frame(1)

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda *args, **kwargs: Container()))
    with pytest.raises(RequestError, match="more frames"):
        _decode_video(b"video", max_frames=1, max_pixels=10, max_total_pixels=10)


@pytest.mark.parametrize("overrides", [
    {"messages": []}, {"max_tokens": True}, {"max_tokens": 0},
    {"temperature": -0.1}, {"top_p": 0}, {"top_k": 0}, {"model": "other"},
    {"n": 2}, {"stream": "false"},
    {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,%%%"}}]}]},
])
def test_invalid_request_model_params_and_media_are_rejected(overrides):
    client = TestClient(_app(FakeEngine()))
    payload = _request()
    payload.update(overrides)
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 400


def test_concurrent_requests_are_isolated_and_serialized():
    engine = FakeEngine(delay=0.025)
    app = _app(engine, max_pending_requests=2)
    clients = [TestClient(app), TestClient(app)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(c.post, "/v1/chat/completions", json=_request(messages=[
            {"role": "user", "content": value}
        ])) for c, value in zip(clients, ("first", "second"))]
        outputs = [f.result(timeout=5).json()["choices"][0]["message"]["content"] for f in futures]
    assert outputs == ["chunk0chunk1chunk2", "chunk0chunk1chunk2"]
    assert engine.max_active == 1
    assert {call["messages"][0]["content"] for call in engine.calls} == {"first", "second"}


def test_pending_request_limit_is_bounded():
    engine = FakeEngine(delay=0.05)
    app = _app(engine, max_pending_requests=0)
    first_client, second_client = TestClient(app), TestClient(app)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(first_client.post, "/v1/chat/completions", json=_request())
        assert engine.started.wait(2)
        rejected = second_client.post("/v1/chat/completions", json=_request())
        assert rejected.status_code == 503
        assert "queue is full" in rejected.json()["detail"]
        assert first.result(timeout=3).status_code == 200


def test_failed_generation_releases_lock_for_next_request():
    engine = FakeEngine(fail_once=True)
    client = TestClient(_app(engine), raise_server_exceptions=False)
    first = client.post("/v1/chat/completions", json=_request())
    second = client.post("/v1/chat/completions", json=_request())
    assert first.status_code == 500
    assert second.status_code == 200
    assert engine.active == 0


def test_disconnect_cancels_generation_and_server_recovers():
    engine = FakeEngine(delay=0.005, long_stream=True)
    app = _app(engine)
    payload = json.dumps(_request(stream=True)).encode()

    async def exercise_disconnect():
        first_receive = True
        first_body = asyncio.Event()
        async def receive():
            nonlocal first_receive
            if first_receive:
                first_receive = False
                return {"type": "http.request", "body": payload, "more_body": False}
            await first_body.wait()
            return {"type": "http.disconnect"}

        sent = []
        async def send(message):
            sent.append(message)
            if message["type"] == "http.response.body" and message.get("body"):
                first_body.set()

        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
                 "http_version": "1.1", "method": "POST", "scheme": "http",
                 "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions",
                 "query_string": b"", "headers": [(b"content-type", b"application/json"),
                 (b"content-length", str(len(payload)).encode())],
                 "client": ("testclient", 123), "server": ("testserver", 80)}
        await app(scope, receive, send)
        return sent

    sent = asyncio.run(exercise_disconnect())
    assert any(message["type"] == "http.response.start" for message in sent)
    assert engine.cancelled.wait(2)
    response = TestClient(app).post("/v1/chat/completions", json=_request())
    assert response.status_code == 200
    assert engine.active == 0


def test_sse_disconnect_waits_for_slow_next_before_releasing_model_slot():
    class SlowNextEngine:
        model_id = "bctx-test"

        def __init__(self):
            self.lock = threading.Lock()
            self.active = 0
            self.max_active = 0
            self.calls = 0
            self.slow_next_started = threading.Event()
            self.slow_next_finished = threading.Event()

        def stream_chat(self, messages, options, *, cancel_event, tools=None, chat_template_kwargs=None):
            with self.lock:
                self.calls += 1
                call = self.calls
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                yield "first"
                if call == 1:
                    self.slow_next_started.set()
                    time.sleep(0.35)  # simulate a decoder step that cannot be interrupted
                    self.slow_next_finished.set()
                yield "second"
            finally:
                with self.lock:
                    self.active -= 1

    engine = SlowNextEngine()
    app = _app(engine)
    payload = json.dumps(_request(stream=True)).encode()

    async def disconnect_first_and_issue_second():
        request_seen = False

        async def receive():
            nonlocal request_seen
            if not request_seen:
                request_seen = True
                return {"type": "http.request", "body": payload, "more_body": False}
            await asyncio.to_thread(engine.slow_next_started.wait, 2)
            return {"type": "http.disconnect"}

        async def send(message):
            return None

        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
                 "http_version": "1.1", "method": "POST", "scheme": "http",
                 "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions",
                 "query_string": b"", "headers": [(b"content-type", b"application/json"),
                 (b"content-length", str(len(payload)).encode())],
                 "client": ("testclient", 123), "server": ("testserver", 80)}
        first_task = asyncio.create_task(app(scope, receive, send))
        assert await asyncio.to_thread(engine.slow_next_started.wait, 2)
        second_task = asyncio.create_task(asyncio.to_thread(
            TestClient(app).post, "/v1/chat/completions", json=_request()))
        response = await asyncio.wait_for(second_task, timeout=3)
        await asyncio.wait_for(first_task, timeout=3)
        return response

    response = asyncio.run(disconnect_first_and_issue_second())
    assert response.status_code == 200
    assert engine.slow_next_finished.is_set()
    assert engine.max_active == 1
    assert engine.active == 0
