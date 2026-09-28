"""Fail-closed CPU tests for serving receipt logits gates."""
from __future__ import annotations

import sys
import weakref
from types import SimpleNamespace
from pathlib import Path

import torch
import pytest
REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "release"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from glc_serve.gate import compare  # noqa: E402
from glc_serve.bench import Arm  # noqa: E402
import glc_serve.bench as bench_module  # noqa: E402


def pair(*, ref_id="text-0", got_id=None, ref_image=None, got_image=None,
         ref_input="a" * 64, got_input="a" * 64, ref_logits=None, got_logits=None):
    ref_logits = ref_logits if ref_logits is not None else torch.tensor([1., 0., -1.])
    got_logits = got_logits if got_logits is not None else ref_logits.clone()
    return ([{"id": ref_id, "logits": ref_logits, "input_sha256": ref_input,
              "image_sha256": ref_image}],
            [{"id": got_id or ref_id, "logits": got_logits, "input_sha256": got_input,
              "image_sha256": got_image}])


def test_exact_mode_requires_bitwise_logits():
    ref_logits = torch.tensor([1., 0., -1.], dtype=torch.bfloat16)
    got_logits = torch.tensor([1.0078125, 0., -1.], dtype=torch.bfloat16)
    ref, got = pair(ref_logits=ref_logits, got_logits=got_logits)
    assert compare(ref, got)["status"] == "ok"
    exact = compare(ref, got, exact=True)
    assert exact["status"] == "failed"
    assert exact["exact_required"] is True
    assert exact["all_bitwise_equal"] is False


@pytest.mark.parametrize("ref_logits,got_logits", [
    (torch.tensor([0.0]), torch.tensor([-0.0])),
    (torch.tensor([1.0], dtype=torch.float32), torch.tensor([1.0], dtype=torch.float64)),
])
def test_exact_mode_checks_dtype_and_raw_bits(ref_logits, got_logits):
    ref, got = pair(ref_logits=ref_logits, got_logits=got_logits)
    assert compare(ref, got)["status"] == "ok"
    assert compare(ref, got, exact=True)["status"] == "failed"


def test_matching_image_probe_digests_pass():
    digest = "b" * 64
    ref, got = pair(ref_id="img-shapes", ref_image=digest, got_image=digest)
    assert compare(ref, got)["status"] == "ok"


def test_non_vector_logits_fail_cleanly():
    ref, got = pair(ref_logits=torch.ones((1, 3)), got_logits=torch.ones((1, 3)))
    result = compare(ref, got)
    assert result["status"] == "failed"
    assert result["per_probe"][0]["finite_logits"] is False


def test_missing_and_duplicate_probe_ids_fail_closed():
    ref, got = pair()
    ref.append({**ref[0], "id": "text-1"})
    assert compare(ref, got)["status"] == "failed"
    duplicate = compare(ref, [got[0], got[0]])
    assert duplicate["status"] == "failed"
    assert duplicate["duplicate_or_invalid_probe_ids"] is True


@pytest.mark.parametrize("kwargs", [
    {"ref_image": "a", "got_image": "b"},
    {"ref_input": "a", "got_input": "b"},
])
def test_digest_mismatch_fails(kwargs):
    ref, got = pair(**kwargs)
    assert compare(ref, got)["status"] == "failed"


def test_nonfinite_logits_fail_closed():
    ref, got = pair(got_logits=torch.tensor([float("nan"), 0., -1.]))
    result = compare(ref, got)
    assert result["status"] == "failed"
    assert result["per_probe"][0]["finite_logits"] is False


def test_partial_checkpoint_has_distinct_nonfinal_name(tmp_path):
    arm = object.__new__(Arm)
    arm.name = "cpu"
    arm.result = {"schema": "arm", "context": {"rows": []}}
    arm.checkpoint_partial(tmp_path)
    partial = tmp_path / "arm_cpu.partial.json"
    assert partial.exists()
    assert not (tmp_path / "arm_cpu.json").exists()
    assert '"receipt_status": "partial"' in partial.read_text()


def test_context_probe_records_separate_prefill_and_greedy_decode(monkeypatch):
    import glc_serve.mtp as mtp

    class Cache:
        def __init__(self):
            self.layers = [SimpleNamespace(keys=None, values=None)]
            self.length = 0

        def get_seq_length(self):
            return self.length

    class Model:
        config = SimpleNamespace()

        def __call__(self, *, input_ids, past_key_values, **kwargs):
            n = int(input_ids.shape[1])
            old = past_key_values.layers[0].keys
            add = torch.ones((1, 1, n, 2), dtype=torch.float32)
            past_key_values.layers[0].keys = add if old is None else torch.cat((old, add), 2)
            past_key_values.layers[0].values = past_key_values.layers[0].keys
            past_key_values.length += n
            logits = torch.tensor([[[0., 2., 1.]]]).expand(1, 1, 3)
            return SimpleNamespace(logits=logits)

    monkeypatch.setattr(mtp, "new_cache", lambda config: Cache())
    arm = object.__new__(Arm)
    arm.model = Model()
    arm.dev = torch.device("cpu")
    arm._vocab_ids = lambda shape, seed: torch.ones(shape, dtype=torch.long)
    arm.state = SimpleNamespace(receipts={})
    arm._steady_alloc = 0
    arm.name = "mock"
    arm.log = lambda *args: None
    arm.result = {}
    arm.context_probe([32, 64], 16)
    rows = arm.result["context"]["rows"]
    assert [row["status"] for row in rows] == ["ok", "ok"], rows
    assert all(row["decode_tokens"] == 16 for row in rows)
    assert all(row["prefill_tok_s"] > 0 and row["decode_tok_s"] > 0 for row in rows)


def test_prefill_releases_previous_output_before_next_forward():
    class Output:
        def __init__(self, cache):
            self.cache = cache

    class Model:
        def __init__(self):
            self.refs = None
            self.released_before_next_forward = False
            self.calls = 0

        def __call__(self, **kwargs):
            self.calls += 1
            if self.refs is not None:
                self.released_before_next_forward = all(ref() is None for ref in self.refs)
                assert self.released_before_next_forward
            cache = torch.ones((1, 1024))
            output = Output(cache)
            self.refs = (weakref.ref(output), weakref.ref(cache))
            return output

    arm = object.__new__(Arm)
    arm.name = "weakref"
    arm.dev = torch.device("cpu")
    arm.model = Model()
    arm.log = lambda *args: None
    arm.result = {}
    arm._vocab_ids = lambda shape, seed: torch.ones(shape, dtype=torch.long)
    arm.prefill_throughput([16])
    assert arm.model.calls == 2, (arm.model.calls, arm.result)
    assert arm.model.released_before_next_forward
    assert arm.result["prefill"][0]["status"] == "ok"


def test_decode_interval_rejects_nonpositive_duration():
    with pytest.raises(ValueError, match="decode interval must be positive"):
        bench_module._decode_interval(8.20337, 7.30369)


def test_decode_warms_shapes_before_timing_and_rejects_bad_interval(monkeypatch):
    class Model:
        def __init__(self):
            self.calls = []

        def generate(self, *, max_new_tokens, **kwargs):
            self.calls.append(max_new_tokens)
            return object()

    arm = object.__new__(Arm)
    arm.name = "timing"
    arm.dev = torch.device("cpu")
    arm.engine = SimpleNamespace(pad_id=0, eos_ids=[])
    arm.model = Model()
    arm.log = lambda *args: None
    arm.result = {}
    arm._vocab_ids = lambda shape, seed: torch.ones(shape, dtype=torch.long)
    timed_call_counts = []
    durations = iter((8.20337, 7.30369))

    def timed(fn):
        timed_call_counts.append(len(arm.model.calls))
        output = fn()
        return next(durations), output

    arm._timed = timed
    monkeypatch.setattr(bench_module, "_reset_peak", lambda dev: None)
    monkeypatch.setattr(bench_module, "_peak", lambda dev: {})
    arm.decode_throughput([16], 128, 32)
    row = arm.result["decode"][0]
    assert arm.model.calls == [2, 1, 32]
    assert timed_call_counts == [1, 2]
    assert row["status"] == "error"
    assert "decode interval must be positive" in row["error"]
    assert "decode_tok_s" not in row
