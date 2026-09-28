"""Fail-closed physical-card accounting for capped benchmark arms."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "release"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from glc_serve import memcap
from glc_serve import bench


def _final_measure(nvml_mib):
    return {
        "allocated": 70,
        "reserved": 80,
        "max_allocated": 90,
        "max_reserved": 90,
        "nvml_process_mib": nvml_mib,
        "device": "cuda:0",
    }


def test_phase_highwater_fails_even_when_final_nvml_is_below_card(monkeypatch):
    monkeypatch.setattr(memcap, "measure", lambda device: _final_measure(39226))

    with pytest.raises(RuntimeError, match="observed process peak 41094 MiB"):
        memcap.assert_within_cap(
            "cuda:0", {"cap_bytes": 100}, card_mib=40960,
            observed_nvml_mib=[41094, 39226],
        )


def test_missing_phase_nvml_cannot_certify_physical_card_fit(monkeypatch):
    monkeypatch.setattr(memcap, "measure", lambda device: _final_measure(39226))

    with pytest.raises(RuntimeError, match="phase NVML sample"):
        memcap.assert_within_cap(
            "cuda:0", {"cap_bytes": 100}, card_mib=40960,
            observed_nvml_mib=[None],
        )


def test_context_over_card_row_keeps_timings_but_is_excluded_from_fit(monkeypatch):
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

    monkeypatch.setattr("glc_serve.mtp.new_cache", lambda config: Cache())
    monkeypatch.setattr(bench, "_peak", lambda dev: {
        "max_allocated": 90, "max_reserved": 90, "nvml_process_mib": 41094,
    })
    arm = object.__new__(bench.Arm)
    arm.model = Model()
    arm.dev = torch.device("cpu")
    arm._vocab_ids = lambda shape, seed: torch.ones(shape, dtype=torch.long)
    arm.state = SimpleNamespace(receipts={})
    arm._steady_alloc = 0
    arm.card_mib = 40960
    arm.name = "cap40"
    arm.log = lambda *args: None
    arm.result = {}

    arm.context_probe([32, 64], 16)

    rows = arm.result["context"]["rows"]
    assert rows[0]["status"] == "over_card"
    assert rows[0]["physical_card_failure"].startswith("NVML process memory")
    assert rows[0]["prefill_tok_s"] > 0 and rows[0]["decode_tok_s"] > 0
    assert arm.result["context"]["max_context_measured_tokens"] == 0
    assert len(rows) == 1
