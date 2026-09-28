"""CPU tests: MTP self-speculative decoding never changes the greedy output.

fp32 on the CPU so kernel rounding cannot create near-ties; the property
under test is the cache rollback (Gated-DeltaNet state restore + attention
crop) and the accept/reject bookkeeping, on text AND image prompts.  The
tiny model's MTP head is random, so an ORACLE draft (the true next greedy
token with probability p, a wrong token otherwise) drives the accept path.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "release", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from glc_serve.engine import Engine, Request, SamplingParams  # noqa: E402
from glc_serve.gate import synthetic_images  # noqa: E402
from glc_serve.mtp import MTPSpeculator  # noqa: E402


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    from transformers import AutoProcessor, AutoTokenizer

    from glc_serve.bundle import open_bundle
    from glc_serve.loader import ServeOptions, load_bundle_model
    from glc_serve.pack import pack_from_dense
    from glc_serve_tiny import build_checkpoint

    root = tmp_path_factory.mktemp("glc_serve_mtp")
    build_checkpoint(root / "ckpt")
    pack_from_dense(root / "ckpt", root / "bundle", min_numel=0, log=lambda *a: None)
    loaded = load_bundle_model(open_bundle(str(root / "bundle")),
                               ServeOptions(backend="materialize", device="cpu"))
    loaded.model.float()
    loaded.mtp.float()
    tok = AutoTokenizer.from_pretrained(str(loaded.local_dir))
    proc = AutoProcessor.from_pretrained(str(loaded.local_dir))
    return Engine(loaded, proc, tok)


def _greedy_reference(eng, inputs, n):
    with torch.no_grad():
        g = eng.model.generate(**inputs, max_new_tokens=n, do_sample=False,
                               eos_token_id=eng.eos_ids, pad_token_id=eng.pad_id)
    ref = []
    for t in g[0, inputs["input_ids"].shape[1]:].tolist():
        if t in eng.eos_ids:
            break
        ref.append(t)
    return ref


@pytest.mark.parametrize("with_image", [False, True])
@pytest.mark.parametrize("p_accept", [0.0, 0.5, 1.0])
def test_speculative_equals_greedy(setup, with_image, p_accept):
    eng = setup
    if with_image:
        content = [{"type": "image"}, {"type": "text", "text": "Describe"}]
    else:
        content = "The capital of France is"
    req = Request(kind="chat", params=SamplingParams(max_tokens=32, temperature=0),
                  messages=[{"role": "user", "content": content}])
    if with_image:
        req.images = [synthetic_images()[0]]
    inputs = eng.prepare([req])
    ref = _greedy_reference(eng, inputs, 32)
    got = []
    spec = MTPSpeculator(eng.model, eng.loaded.mtp)
    real_draft = spec._draft
    rng = random.Random(1234)

    def draft(hidden, tokens, pos, cache):
        guess = real_draft(hidden, tokens, pos, cache)   # keeps the MTP cache advancing
        k = len(got)
        if p_accept > 0 and k < len(ref) and rng.random() < p_accept:
            return ref[k]
        return guess if p_accept == 0 else (ref[k] + 1) % 500 if k < len(ref) else guess

    spec._draft = draft
    stats = spec.generate({k: v for k, v in inputs.items() if k != "attention_mask"},
                          max_new_tokens=32, eos_ids=eng.eos_ids,
                          on_token=lambda t: (got.append(int(t)) or True))
    assert got == ref
    if p_accept == 1.0:
        assert stats["accepted"] >= 3
    if p_accept == 0.5:
        assert stats["accepted"] >= 1 and stats["rejected"] >= 1
    assert stats.get("rerun_disagreed_with_verify", 0) == 0


def test_engine_routes_single_greedy_request_to_mtp(setup):
    eng = setup
    eng.enable_mtp = True
    req = Request(kind="chat", params=SamplingParams(max_tokens=8, temperature=0),
                  messages=[{"role": "user", "content": "Water boils at"}])
    eng.submit(req)
    eng.run_wave([eng._q.get()])
    events = []
    while not req.out.empty():
        events.append(req.out.get())
    assert events[-1][0] == "done"
    assert req.spec_stats is not None and "acceptance_rate" in req.spec_stats
    eng.enable_mtp = False


def test_chunked_prefill_matches_unchunked(setup):
    eng = setup
    prompt = " ".join(["capital water gold moon"] * 25)

    def run(chunk):
        eng.prefill_chunk = chunk
        req = Request(kind="completion", params=SamplingParams(max_tokens=10, temperature=0),
                      prompt=prompt)
        eng.submit(req)
        eng.run_wave([eng._q.get()])
        return list(req.gen_ids)

    before = eng.stats.get("chunked_prefills", 0)
    plain = run(0)
    chunked = run(16)
    eng.prefill_chunk = 4096
    assert plain == chunked and len(plain) > 0
    assert eng.stats.get("chunked_prefills", 0) == before + 1
