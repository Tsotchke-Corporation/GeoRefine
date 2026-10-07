"""Synthetic unit coverage for genuine MTP wiring and cache rollback helpers.

These tests do not claim real-checkpoint, BCTX-GPU, or serving qualification.
"""
import hashlib
import json
from pathlib import Path

import pytest
torch = pytest.importorskip("torch")
from torch import nn
import torch.nn.functional as F

from scripts.bitexact_context_mtp import (
    BCTXLinear,
    MTP_LINEAR_PATHS,
    MTP_SHAPES,
    _causal_mask,
    _causal_attention_bias,
    _mrope_positions,
    _safe_frame_path,
    bctx_mtp_receipt,
    crop_attention,
    load_bctx_mtp,
    restore_recurrent,
    snapshot_recurrent,
    sequential_qwen35_gdn_cache,
)


class _DenseTensor:
    def __init__(self, value):
        self.value = value
        self.shape = tuple(value.shape)
        self.calls = 0

    def decode(self):
        self.calls += 1
        return self.value


def test_bctx_linear_decodes_per_call_and_matches_dense_linear():
    torch.manual_seed(7)
    w = torch.randn(5, 3, dtype=torch.bfloat16)
    x = torch.randn(2, 3, dtype=torch.bfloat16)
    descriptor = _DenseTensor(w)
    layer = BCTXLinear(descriptor, "mtp.test.weight")
    got = layer(x)
    expected = torch.nn.functional.linear(x, w)
    assert torch.equal(got, expected)
    assert descriptor.calls == layer.calls == 1


def test_manifest_inventory_is_15_weights_eight_matrices_seven_scales():
    assert len(MTP_SHAPES) == 15
    assert len(MTP_LINEAR_PATHS) == 8
    assert set(MTP_LINEAR_PATHS) <= set(MTP_SHAPES)
    assert len(set(MTP_SHAPES) - set(MTP_LINEAR_PATHS)) == 7
    assert MTP_SHAPES["mtp.fc.weight"] == (5120, 10240)
    assert MTP_SHAPES["mtp.layers.0.self_attn.q_proj.weight"] == (12288, 5120)


def test_strict_receipt_requires_each_real_matrix_module_to_run():
    class TinyMTP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = BCTXLinear(_DenseTensor(torch.ones(2, 2)), "fc")
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].self_attn = nn.Module()
            self.layers[0].mlp = nn.Module()
            for path in MTP_LINEAR_PATHS.values():
                if path == "fc":
                    continue
                parent_path, _, attr = path.rpartition(".")
                parent = self.get_submodule(parent_path)
                setattr(parent, attr, BCTXLinear(_DenseTensor(torch.ones(2, 2)), path))
            self._bctx_receipt = {"tensor_count": 15}

    mtp = TinyMTP()
    with pytest.raises(RuntimeError, match="not consumed"):
        bctx_mtp_receipt(mtp, require_consumed=True)
    for path in MTP_LINEAR_PATHS.values():
        mtp.get_submodule(path)(torch.ones(1, 2))
    receipt = bctx_mtp_receipt(mtp, require_consumed=True)
    assert receipt["consumed_linear_count"] == 8
    assert all(receipt["linear_calls"].values())


def test_recurrent_rollback_and_attention_crop_restore_preverify_cache():
    class Layer:
        pass

    class Cache:
        def __init__(self):
            self.layers = [Layer()]
            self.layers[0].conv_states = torch.tensor([1.0])
            self.layers[0].recurrent_states = torch.tensor([2.0])
            self.layers[0].has_previous_state = True
            self.key_cache = [torch.arange(5).view(1, 1, 5, 1)]
            self.value_cache = [torch.arange(5).view(1, 1, 5, 1)]

        def crop(self, length):
            self.key_cache[0] = self.key_cache[0][..., :length, :]
            self.value_cache[0] = self.value_cache[0][..., :length, :]

    cache = Cache()
    before = snapshot_recurrent(cache)
    cache.layers[0].conv_states.fill_(9)
    cache.layers[0].recurrent_states.fill_(8)
    cache.layers[0].has_previous_state = False
    crop_attention(cache, 3)
    restore_recurrent(cache, before)
    assert cache.layers[0].conv_states.item() == 1
    assert cache.layers[0].recurrent_states.item() == 2
    assert cache.layers[0].has_previous_state is True
    assert cache.key_cache[0].shape[-2] == cache.value_cache[0].shape[-2] == 3


def test_mrope_offset_and_causal_positions_match_qwen_mtp_contract():
    pos = _mrope_positions(torch.tensor([3, 4]), torch.tensor([2]))
    assert tuple(pos.shape) == (3, 1, 2)
    assert torch.equal(pos[:, 0, :], torch.tensor([[5, 6], [5, 6], [5, 6]]))
    mask = _causal_mask(t=2, past=3, device="cpu")
    assert torch.equal(mask[0, 0], torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.bool))


def test_eager_mtp_attention_mask_is_additive_and_blocks_future_tokens():
    bias = _causal_attention_bias(2, 1, "cpu", torch.float32)
    assert torch.equal(bias[0, 0], torch.tensor([[0.0, 0.0, float("-inf")], [0.0, 0.0, 0.0]]))


def test_cached_gdn_chunk_mode_runs_projections_once_and_threads_cached_state():
    class CacheLayer:
        has_previous_state = True
        conv_states = torch.zeros((1, 3, 2))
        recurrent_states = torch.tensor([[[5.0]]])

    class Cache:
        layers = [CacheLayer()]

        def has_previous_state(self, layer_idx):
            return self.layers[layer_idx].has_previous_state

        def update_conv_state(self, state, layer_idx):
            self.layers[layer_idx].conv_states = state

        def update_recurrent_state(self, state, layer_idx):
            self.layers[layer_idx].recurrent_states = state

    class Projection(nn.Module):
        def __init__(self, out_width):
            super().__init__()
            self.calls = 0
            self.out_width = out_width

        def forward(self, hidden):
            self.calls += 1
            return hidden.expand(*hidden.shape[:-1], self.out_width)

    class GDN(nn.Module):
        layer_idx = 0
        conv_kernel_size = 2
        head_v_dim = head_k_dim = key_dim = value_dim = num_v_heads = num_k_heads = 1
        activation = "silu"

        def __init__(self):
            super().__init__()
            self.in_proj_qkv = Projection(3)
            self.in_proj_z = Projection(1)
            self.in_proj_b = Projection(1)
            self.in_proj_a = Projection(1)
            self.conv1d = nn.Conv1d(3, 3, kernel_size=2, groups=3, bias=False)
            nn.init.ones_(self.conv1d.weight)
            self.causal_conv1d_fn = None
            self.A_log = nn.Parameter(torch.zeros(1))
            self.dt_bias = nn.Parameter(torch.zeros(1))
            self.norm = lambda core, z: core + z
            self.out_proj = nn.Identity()
            self.initial_state_seen = None

        def chunk_gated_delta_rule(self, query, key, value, *, initial_state, **kwargs):
            self.initial_state_seen = initial_state.clone()
            return value, initial_state + value.sum(dim=1, keepdim=True)

        def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
            raise AssertionError("chunk mode must use the state-preserving chunk path")

    gdn = type("Qwen3_5GatedDeltaNet", (GDN,), {})()
    model = nn.Sequential(gdn)
    cache = Cache()
    hidden = torch.tensor([[[1.0], [2.0]]])
    with sequential_qwen35_gdn_cache(model, mode="chunk"):
        output = gdn(hidden, cache, torch.ones((1, 2), dtype=torch.long))
    assert output.shape == (1, 2, 1)
    assert gdn.initial_state_seen.item() == 5.0
    assert cache.layers[0].recurrent_states.item() > 5.0
    assert [gdn.in_proj_qkv.calls, gdn.in_proj_z.calls, gdn.in_proj_b.calls, gdn.in_proj_a.calls] == [1, 1, 1, 1]
    assert cache.layers[0].conv_states.shape == (1, 3, 2)
    assert torch.equal(cache.layers[0].conv_states, torch.tensor([[[1.0, 2.0]] * 3]))


def test_chunk_convolution_matches_four_cached_single_token_steps_with_padded_conv_module():
    class CacheLayer:
        has_previous_state = True
        conv_states = torch.tensor([[[2.0, -1.0, 3.0, 0.5], [-2.0, 1.5, 0.0, 4.0],
                                     [1.0, -3.0, 2.0, -0.5]]])
        recurrent_states = torch.tensor([[[1.0]]])

    class Cache:
        layers = [CacheLayer()]

        def has_previous_state(self, layer_idx):
            return self.layers[layer_idx].has_previous_state

        def update_conv_state(self, state, layer_idx):
            self.layers[layer_idx].conv_states.copy_(state)

        def update_recurrent_state(self, state, layer_idx):
            self.layers[layer_idx].recurrent_states = state

    class Projection(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.width = width

        def forward(self, hidden):
            return hidden.expand(*hidden.shape[:-1], self.width)

    class GDN(nn.Module):
        layer_idx = 0
        conv_kernel_size = 4
        head_v_dim = head_k_dim = key_dim = value_dim = num_v_heads = num_k_heads = 1
        activation = "silu"

        def __init__(self):
            super().__init__()
            self.in_proj_qkv = Projection(3)
            self.in_proj_z = Projection(1)
            self.in_proj_b = Projection(1)
            self.in_proj_a = Projection(1)
            self.conv1d = nn.Conv1d(3, 3, kernel_size=4, padding=3, groups=3, bias=True)
            with torch.no_grad():
                self.conv1d.weight.copy_(torch.tensor([[[0.1, 0.2, 0.3, 0.4]],
                                                        [[-0.4, 0.3, -0.2, 0.1]],
                                                        [[0.5, -0.1, 0.2, -0.3]]]))
                self.conv1d.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
            self.causal_conv1d_fn = None
            self.A_log = nn.Parameter(torch.zeros(1))
            self.dt_bias = nn.Parameter(torch.zeros(1))
            self.norm = lambda core, z: core
            self.out_proj = nn.Identity()
            self.mixed_seen = None

        def chunk_gated_delta_rule(self, query, key, value, *, initial_state, **kwargs):
            self.mixed_seen = torch.cat((query, key, value), dim=-1)
            return value, initial_state + value.sum(dim=1, keepdim=True)

        def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
            raise AssertionError("chunk mode should bypass padded Conv1d.forward")

    gdn = type("Qwen3_5GatedDeltaNet", (GDN,), {})()
    model = nn.Sequential(gdn)
    cache = Cache()
    old_conv = cache.layers[0].conv_states.clone()
    hidden = torch.tensor([[[1.0], [-2.0], [3.0], [-4.0]]])
    token_qkv = hidden.expand(1, 4, 3).transpose(1, 2)
    state = old_conv.clone()
    expected = []
    for i in range(hidden.shape[1]):
        context = state[:, :, -3:]
        one_input = torch.cat((context, token_qkv[:, :, i:i + 1]), dim=-1)
        step = F.conv1d(one_input, gdn.conv1d.weight, gdn.conv1d.bias,
                        stride=1, padding=0, groups=gdn.conv1d.groups)
        expected.append(F.silu(step).transpose(1, 2))
        state = torch.cat((state[:, :, 1:], token_qkv[:, :, i:i + 1]), dim=-1)
    with sequential_qwen35_gdn_cache(model, mode="chunk"):
        output = gdn(hidden, cache, torch.ones((1, 4), dtype=torch.long))
    assert output.shape == (1, 4, 1)
    assert torch.allclose(gdn.mixed_seen.squeeze(2), torch.cat(expected, dim=1), atol=1e-6, rtol=1e-6)
    assert torch.equal(cache.layers[0].conv_states, state)


def test_loader_accepts_matrix_frame_geometry_for_one_dimensional_scales(monkeypatch):
    import scripts.bitexact_context_mtp as mtp_module

    class LinearStub(nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(shape, dtype=torch.bfloat16, device="meta"))
            self.bias = None

    class NormStub(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(width, dtype=torch.bfloat16, device="meta"))

    class TinyMTP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = LinearStub(MTP_SHAPES["mtp.fc.weight"])
            self.pre_fc_norm_embedding = NormStub(5120)
            self.pre_fc_norm_hidden = NormStub(5120)
            self.norm = NormStub(5120)
            layer = nn.Module()
            layer.input_layernorm = NormStub(5120)
            layer.post_attention_layernorm = NormStub(5120)
            layer.self_attn = nn.Module()
            layer.self_attn.q_norm = NormStub(256)
            layer.self_attn.k_norm = NormStub(256)
            for suffix in ("q_proj", "k_proj", "v_proj", "o_proj"):
                name = f"mtp.layers.0.self_attn.{suffix}.weight"
                setattr(layer.self_attn, suffix, LinearStub(MTP_SHAPES[name]))
            layer.mlp = nn.Module()
            for suffix in ("gate_proj", "up_proj", "down_proj"):
                name = f"mtp.layers.0.mlp.{suffix}.weight"
                setattr(layer.mlp, suffix, LinearStub(MTP_SHAPES[name]))
            self.layers = nn.ModuleList([layer])

    raw_frames = {}
    manifest = {"tensors": []}
    for i, (name, shape) in enumerate(MTP_SHAPES.items()):
        frame_name = f"frames/f{i:02d}.bctx"
        data = f"synthetic:{name}".encode()
        raw_frames[frame_name] = data
        manifest["tensors"].append({
            "name": name,
            "shape": list(shape),
            "dtype": "BF16",
            "frame": frame_name,
            "frame_bytes": len(data),
            "frame_sha256": hashlib.sha256(data).hexdigest(),
            "source_sha256": hashlib.sha256(name.encode()).hexdigest(),
        })

    class FakeTensor:
        def __init__(self, frame, *, stride, device):
            del stride, device
            entry = next(e for e in manifest["tensors"] if raw_frames[e["frame"]] == frame)
            shape = tuple(entry["shape"])
            self.shape = (1, shape[0]) if len(shape) == 1 else shape
            self.source_sha256 = entry["source_sha256"]
            self.frame_sha256 = entry["frame_sha256"]
            self.resident_bytes = 17

        def decode(self):
            return torch.ones(self.shape, dtype=torch.bfloat16)

    original_read_text = Path.read_text
    original_read_bytes = Path.read_bytes

    def read_text(path, *args, **kwargs):
        if path.name == "manifest.json":
            return json.dumps(manifest)
        return original_read_text(path, *args, **kwargs)

    def read_bytes(path):
        rel = path.as_posix().split("/virtual/frame-root/", 1)[-1]
        if rel in raw_frames:
            return raw_frames[rel]
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(mtp_module, "QwenMTP", lambda _config: TinyMTP())
    mtp = load_bctx_mtp(
        "/virtual/manifest.json", "/virtual/frame-root", object(),
        device="cpu", tensor_cls=FakeTensor,
    )
    receipt = bctx_mtp_receipt(mtp)
    assert receipt["tensor_count"] == 15
    assert len(receipt["scale_names"]) == 7
    assert receipt["scale_resident_bytes"] == sum(
        shape[0] * 2 for name, shape in MTP_SHAPES.items() if name not in MTP_LINEAR_PATHS
    )
    assert mtp.layers[0].input_layernorm.weight.shape == (5120,)
    assert mtp.layers[0].self_attn.q_norm.weight.shape == (256,)


def test_frame_path_cannot_escape_frame_root():
    with pytest.raises(ValueError, match="escapes frame_root"):
        _safe_frame_path(Path("/virtual/frame-root"), "../outside.bctx")


def test_cached_gdn_multitoken_forward_is_sequential_and_scope_restores():
    class CacheLayer:
        has_previous_state = True
        recurrent_states = torch.tensor(10.0)

    class Cache:
        layers = [CacheLayer()]

        def has_previous_state(self, layer_idx):
            return self.layers[layer_idx].has_previous_state

    class GDN(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = 0
            self.calls = []

        def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
            self.calls.append(tuple(hidden_states.shape))
            # Emulate the bad cached chunk path: multi-token forwards start at 0.
            state = cache_params.layers[0].recurrent_states if hidden_states.shape[1] == 1 else torch.tensor(0.0)
            values = []
            for token in hidden_states[0, :, 0]:
                state = state + token
                values.append(state)
            cache_params.layers[0].recurrent_states = state
            return torch.stack(values).view(1, -1, 1)

    gdn = type("Qwen3_5GatedDeltaNet", (GDN,), {} )()
    model = nn.Sequential(gdn)
    original = gdn.forward
    cache = Cache()
    with sequential_qwen35_gdn_cache(model):
        out = gdn(torch.tensor([[[1.0], [2.0]]]), cache, torch.ones((1, 2), dtype=torch.long))
        assert torch.equal(out, torch.tensor([[[11.0], [13.0]]]))
        assert gdn.calls == [(1, 1, 1), (1, 1, 1)]
    assert gdn.forward == original
    assert cache.layers[0].recurrent_states.item() == 13.0


def test_cached_gdn_scope_restores_forward_after_fallback_error():
    class Layer:
        has_previous_state = True

    class Cache:
        layers = [Layer()]

        def has_previous_state(self, layer_idx):
            return self.layers[layer_idx].has_previous_state

    class GDN(nn.Module):
        layer_idx = 0

        def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
            return hidden_states

    gdn = type("Qwen3_5GatedDeltaNet", (GDN,), {})()
    model = nn.Sequential(gdn)
    original = gdn.forward
    with pytest.raises(RuntimeError, match="cannot skip masked/padded"):
        with sequential_qwen35_gdn_cache(model):
            gdn(torch.ones((1, 2, 1)), Cache(), torch.tensor([[1, 0]]))
    assert gdn.forward == original


def test_new_mtp_request_drops_stale_multimodal_rope_batch_state():
    from types import SimpleNamespace
    from scripts.bitexact_context_mtp import _reset_request_rope
    model = SimpleNamespace(model=SimpleNamespace(rope_deltas=torch.ones(8, 1)))
    _reset_request_rope(model)
    assert model.model.rope_deltas is None
    _reset_request_rope(SimpleNamespace())
