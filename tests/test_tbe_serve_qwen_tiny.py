"""Whole Qwen3.8 graph: compressed safetensors must match dense logits."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safe_torch = pytest.importorskip("safetensors.torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5Config") or not hasattr(transformers, "AutoModelForMultimodalLM"):
    pytest.skip("Qwen3.8 Transformers support is unavailable", allow_module_level=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "release"))

from glc_loader.tbe_container import encode_tbe  # noqa: E402
from glc_loader.tbe_serve_portable import load_compressed_transformers  # noqa: E402


def test_tiny_qwen_whole_model_logits_match_with_resident_codec(tmp_path):
    # Derived from Qwen/Qwen3.8-27B's Apache-2.0 config, with tiny dimensions.
    config = json.loads((REPO / "tests/fixtures/tiny_qwen38_config.json").read_text())
    config.update(image_token_id=250, vision_start_token_id=251,
                  vision_end_token_id=252, video_token_id=253)
    torch.manual_seed(2701)
    parent = transformers.AutoModelForMultimodalLM.from_config(
        transformers.Qwen3_5Config(**config)
    ).eval()
    # Pretrained BF16 weights do not make config-derived rotary frequencies BF16.
    for parameter in parent.parameters():
        parameter.data = parameter.data.to(torch.bfloat16)
    tokens = torch.tensor([[1, 3, 4, 5]])
    with torch.no_grad():
        reference = parent(input_ids=tokens, use_cache=False).logits
        vision_inputs = {
            "input_ids": torch.tensor([[250, 3]]),
            "pixel_values": torch.zeros((4, 1536), dtype=torch.bfloat16),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "mm_token_type_ids": torch.tensor([[1, 0]]),
            "use_cache": False,
        }
        reference_vision = parent(**vision_inputs).logits

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "config.json").write_text(json.dumps(config))
    tensors, entries = {}, []
    for name, value in parent.named_parameters():
        weight = value.detach().cpu().contiguous()
        raw = weight.view(torch.int16).numpy().tobytes()
        entry = {"name": name, "shape": list(weight.shape), "dtype": str(weight.dtype),
                 "original_bytes": len(raw), "shard": 0,
                 "blake2b_source": hashlib.blake2b(raw).hexdigest()}
        owner, _, leaf = name.rpartition(".")
        if leaf == "weight" and isinstance(parent.get_submodule(owner), torch.nn.Linear):
            coded = encode_tbe(weight, layout="mma16")
            entry.update(kind="tbe", coded_shape=list(weight.shape), layout=coded.layout,
                         mode=coded.mode, base=coded.base, tiles=coded.tiles,
                         escapes=coded.escapes, superblock=coded.superblock)
            for part in ("planes", "smb", "esc", "sbbase"):
                tensor = getattr(coded, part)
                tensors[f"{name}.{part}"] = (
                    tensor.to(torch.int32).contiguous()
                    if part in ("planes", "sbbase") else tensor.contiguous()
                )
        else:
            entry["kind"] = "raw"
            tensors[name] = weight
        entries.append(entry)
    shard = bundle / "shards/shard-00000.safetensors"
    shard.parent.mkdir()
    safe_torch.save_file(tensors, shard)
    shard_bytes = shard.read_bytes()
    sidecar = (bundle / "config.json").read_bytes()
    (bundle / "serve_manifest.json").write_text(json.dumps({
        "format": "georefine.tbe.serve.v1", "tensors": entries,
        "shards": [{"file": "shards/shard-00000.safetensors", "bytes": len(shard_bytes),
                    "sha256": hashlib.sha256(shard_bytes).hexdigest()}],
        "sidecars": [{"file": "config.json", "bytes": len(sidecar),
                      "sha256": hashlib.sha256(sidecar).hexdigest()}],
    }))
    del tensors, shard_bytes
    compressed = load_compressed_transformers(bundle)
    with torch.no_grad():
        actual = compressed(input_ids=tokens, use_cache=False).logits
        actual_vision = compressed(**vision_inputs).logits
    assert torch.equal(reference.view(torch.int16), actual.view(torch.int16))
    assert torch.equal(reference_vision.view(torch.int16), actual_vision.view(torch.int16))
    assert compressed.georefine_tbe_receipt["coded_linears"] == 14
    assert compressed.georefine_tbe_receipt["base_tensors"] == len(entries) == 35
    assert all(not p.is_meta for _, p in compressed.named_parameters())
    assert all(not b.is_meta for _, b in compressed.named_buffers())
