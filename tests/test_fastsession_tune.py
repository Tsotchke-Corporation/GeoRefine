"""The measured FastDecoder tune must ship with the installed codec package."""

import hashlib
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "release"))

from glc_serve.fastserve import _resolve_tune  # noqa: E402


def test_bundled_sm120_tune_and_explicit_override(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _dev: (12, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name",
                        lambda _dev: "NVIDIA RTX PRO 6000 Blackwell Server Edition")
    path = Path(_resolve_tune(None, "cuda:0"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == \
        "1c5375c91f6fc084a5263f7da9177faafbb17fd4bf2281fc520bca1f74ef8024"
    assert _resolve_tune("custom-tune.json", "cuda:0") == "custom-tune.json"


def test_other_cuda_device_requires_its_own_measured_tune(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _dev: (8, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _dev: "NVIDIA A100")
    with pytest.raises(RuntimeError, match="pass tune=PATH"):
        _resolve_tune(None, "cuda:0")
