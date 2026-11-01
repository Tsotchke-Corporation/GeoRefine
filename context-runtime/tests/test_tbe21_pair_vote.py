from types import SimpleNamespace

import pytest

from glc_serve import tbe21_v2_backend as backend


FLAGS = {"BI_TBE21_V2": "1", "BI_TBE21_KERNEL_VARIANT": "1",
         "BI_T21_PACKSCAN2": "1", "BI_T21_TILE128": "1", "BI_T21_TABREG": "1",
         "BI_T21_PAIR_VOTE": "1", "BI_T21_SCAN": "0", "BI_T21_SCAN4": "0",
         "BI_T21_PACKSCAN4": "0", "BI_T21_ASYNC_X": "0", "BI_T21_ROW_GEOMETRY": "0",
         "BI_T21_WIDE128": "0", "BI_T21_PIPE128": "0"}


def test_pair_vote_requires_exact_native_proof(monkeypatch, tmp_path):
    for k, v in FLAGS.items():
        monkeypatch.setenv(k, v)
    binary = tmp_path / "native.so"
    binary.write_bytes(b"fixture")
    module = SimpleNamespace(__file__=str(binary), kernel_variant=lambda: 1,
                             packed_scan2=lambda: True, tile128=lambda: True,
                             tabreg=lambda: True, pair_vote=lambda: True,
                             tile_for=lambda rows: 128 if rows > 64 else 64)
    adapter = backend.adapt(module)
    assert adapter._pair_vote is True
    module.pair_vote = lambda: 1
    with pytest.raises(RuntimeError, match="pair_vote"):
        backend.adapt(module)
    del module.pair_vote
    with pytest.raises(RuntimeError, match="compiled-flag proof"):
        backend.adapt(module)


def test_pair_vote_requires_sm120(monkeypatch):
    for k, v in FLAGS.items():
        monkeypatch.setenv(k, v)
    for arch in ("8.0", "12.0+PTX"):
        with pytest.raises(ValueError, match="exact SM120"):
            backend._ext_args(arch)
