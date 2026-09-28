"""CPU tests: the OpenAI-compatible server's protocol, on a tiny real model.

The server is started exactly as in production (``build_state`` from CLI
args, then ``ThreadingHTTPServer``) with the ``materialize`` backend on the
CPU, so the startup weight gate, the multimodal processor, SSE streaming,
logprobs, prompt scoring and tool-call parsing are all exercised end to end.
"""
from __future__ import annotations

import base64
import io
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "release", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from glc_serve.engine import parse_tool_calls  # noqa: E402
from glc_serve.server import _parse, build_state, serve  # noqa: E402


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    from glc_serve.pack import pack_from_dense
    from glc_serve_tiny import build_checkpoint

    root = tmp_path_factory.mktemp("glc_serve_srv")
    build_checkpoint(root / "ckpt")
    pack_from_dense(root / "ckpt", root / "bundle", min_numel=0, code_lm_head=True,
                    log=lambda *a: None)
    args = _parse(["--bundle", str(root / "bundle"), "--backend", "materialize",
                   "--device", "cpu", "--embed-on-host", "--gate", "full",
                   "--max-batch", "4", "--batch-window-ms", "30", "--port", "0",
                   "--model-name", "tiny"])
    state = build_state(args, log=lambda *a: None)
    httpd = serve(state, "127.0.0.1", 0)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    yield {"url": f"http://127.0.0.1:{httpd.server_address[1]}", "state": state}
    httpd.shutdown()
    state.engine.shutdown()


def _post(url, body, *, raw=False):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    return data if raw else json.loads(data)


def _sse(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    events = []
    with urllib.request.urlopen(req, timeout=120) as r:
        assert r.headers["Content-Type"].startswith("text/event-stream")
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: "):
                payload = line[len("data: "):]
                events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


def _png_data_url():
    from glc_serve.gate import synthetic_images

    buf = io.BytesIO()
    synthetic_images()[1].save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_health_models_receipt(server):
    with urllib.request.urlopen(server["url"] + "/health") as r:
        assert json.loads(r.read())["status"] == "ok"
    with urllib.request.urlopen(server["url"] + "/v1/models") as r:
        models = json.loads(r.read())
    assert models["data"][0]["id"] == "tiny"
    assert "multimodal" in models["data"][0]["capabilities"]
    with urllib.request.urlopen(server["url"] + "/v1/receipt") as r:
        rec = json.loads(r.read())
    assert rec["weight_gate"]["status"] == "ok"
    assert rec["weight_gate"]["scope"] == "full"
    assert rec["load"]["host_embed_bytes"] > 0


def test_chat_stream_matches_non_stream(server):
    body = {"model": "tiny", "messages": [{"role": "user", "content": "The capital of France is"}],
            "max_tokens": 10, "temperature": 0}
    full = _post(server["url"] + "/v1/chat/completions", body)
    text = full["choices"][0]["message"]["content"]
    assert full["usage"]["completion_tokens"] > 0
    events = _sse(server["url"] + "/v1/chat/completions",
                  dict(body, stream=True, stream_options={"include_usage": True}))
    assert events[-1] == "[DONE]"
    assert events[0]["choices"][0]["delta"] == {"role": "assistant"}
    streamed = "".join(e["choices"][0]["delta"].get("content", "")
                       for e in events[1:-1] if e.get("choices"))
    assert streamed == text
    finals = [e for e in events[:-1] if e.get("choices") and e["choices"][0]["finish_reason"]]
    assert finals and finals[-1]["choices"][0]["finish_reason"] in ("stop", "length")
    usage = [e for e in events[:-1] if e.get("usage")]
    assert usage and usage[-1]["usage"]["completion_tokens"] == full["usage"]["completion_tokens"]


def test_concurrent_requests_batch_into_one_wave(server):
    before = dict(server["state"].engine.stats)
    bodies = [{"messages": [{"role": "user", "content": f"Question {i}"}], "max_tokens": 6,
               "temperature": 0} for i in range(3)]
    results = [None] * 3

    def go(i):
        results[i] = _post(server["url"] + "/v1/chat/completions", bodies[i])

    ts = [threading.Thread(target=go, args=(i,)) for i in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert all(r["choices"][0]["message"]["content"] is not None for r in results)
    stats = server["state"].engine.stats
    assert stats["requests"] - before["requests"] == 3
    assert stats["max_wave"] >= 2
    # greedy results do not depend on batch composition here (left padding)
    solo = [_post(server["url"] + "/v1/chat/completions", b) for b in bodies]
    assert [s["choices"][0]["message"]["content"] for s in solo] == \
           [r["choices"][0]["message"]["content"] for r in results]


def test_image_chat(server):
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _png_data_url()}},
        {"type": "text", "text": "Which bar is the tallest?"}]}],
        "max_tokens": 6, "temperature": 0}
    out = _post(server["url"] + "/v1/chat/completions", body)
    assert out["usage"]["prompt_tokens"] > 20, "image placeholder tokens expanded"
    events = _sse(server["url"] + "/v1/chat/completions", dict(body, stream=True))
    streamed = "".join(e["choices"][0]["delta"].get("content", "")
                       for e in events[1:-1] if e.get("choices"))
    assert streamed == out["choices"][0]["message"]["content"]


def test_video_part_is_a_clean_400(server):
    body = {"messages": [{"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}]}]}
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(server["url"] + "/v1/chat/completions", body)
    assert exc.value.code == 400


def test_chat_logprobs(server):
    body = {"messages": [{"role": "user", "content": "Water boils at"}], "max_tokens": 5,
            "temperature": 0, "logprobs": True, "top_logprobs": 3}
    out = _post(server["url"] + "/v1/chat/completions", body)
    content = out["choices"][0]["logprobs"]["content"]
    assert len(content) == out["usage"]["completion_tokens"]
    for item in content:
        assert len(item["top_logprobs"]) == 3
        # greedy: the chosen token is the argmax
        assert abs(item["logprob"] - item["top_logprobs"][0]["logprob"]) < 1e-6


def test_completions_stream_and_echo_scoring(server):
    body = {"prompt": "In 1969, humans first walked on the", "max_tokens": 5, "temperature": 0}
    out = _post(server["url"] + "/v1/completions", body)
    events = _sse(server["url"] + "/v1/completions", dict(body, stream=True))
    streamed = "".join(e["choices"][0]["text"] for e in events[:-1] if e.get("choices"))
    assert streamed == out["choices"][0]["text"]
    score = _post(server["url"] + "/v1/completions",
                  {"prompt": "The chemical symbol for gold is", "max_tokens": 0, "echo": True,
                   "logprobs": 2})
    lp = score["choices"][0]["logprobs"]
    assert lp["token_logprobs"][0] is None
    assert all(isinstance(v, float) for v in lp["token_logprobs"][1:])
    assert len(lp["tokens"]) == score["usage"]["prompt_tokens"]


def test_tool_call_parsing():
    text = ("I will check.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n"
            "</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>")
    content, calls = parse_tool_calls(text)
    assert content == "I will check."
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris", "days": 3}
    assert parse_tool_calls("no tools here") == ("no tools here", [])
