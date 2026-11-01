"""The checkpoint's own MTP head, and self-speculative greedy decoding with it.

Qwen3.5/3.8 checkpoints ship one multi-token-prediction block (``mtp.*``, 15
tensors, 0.849 GB bf16 on the 27B) that transformers does not build
(``_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]``).  Its wiring, verified
against llama.cpp ``src/models/qwen35.cpp`` ``graph_mtp`` at f46bc30 and
vLLM's Qwen3-Next MTP:

    e = pre_fc_norm_embedding(embed(x_{p+1}))
    h = pre_fc_norm_hidden(h_p)            # h_p = trunk output AFTER final norm
    y = fc(concat(e, h))                   # [2H] -> [H], embeddings first
    y = full-attention decoder layer(y)    # its own KV cache
    logits = lm_head(norm(y))              # shared head

and the pair ``(h_p, x_{p+1})`` sits at position ``p + 1`` (llama.cpp
``common/speculative.cpp``: "pair (h_p, x_{p+1}) at MTP pos p+1").

SPECULATION IS LOSSLESS BY CONSTRUCTION.  The head only PROPOSES the token
after next; the full model verifies it.  A draft is kept only when it equals
the full model's own argmax; on a rejection the hybrid cache is rolled back
(Gated-DeltaNet conv/recurrent states restored from a snapshot, attention KV
cropped) and the rejected position is recomputed with a single-token step, so
every emitted token is the full model's greedy choice.  What speculation can
change is only numerics at bf16 near-ties: an accepted token was argmaxed out
of a 2-token verify step rather than a 1-token step, and those kernels round
differently.  The engine reports the measured token agreement against plain
greedy; it never assumes it.

Greedy only (temperature 0), batch 1.  Sampling needs rejection sampling
against the draft distribution, which this module does not implement.
"""
from __future__ import annotations

import copy
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn


class QwenMTP(nn.Module):
    def __init__(self, text_config):
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer,
            Qwen3_5RMSNorm,
        )

        cfg = copy.deepcopy(text_config)
        cfg.layer_types = ["full_attention"]
        cfg.num_hidden_layers = 1
        h = int(cfg.hidden_size)
        eps = float(cfg.rms_norm_eps)
        self.config = cfg
        self.fc = nn.Linear(2 * h, h, bias=False)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(h, eps=eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(h, eps=eps)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(cfg, 0)])
        self.norm = Qwen3_5RMSNorm(h, eps=eps)

    def forward(self, hidden: torch.Tensor, embeds: torch.Tensor,
                position_embeddings, attention_mask, cache) -> torch.Tensor:
        e = self.pre_fc_norm_embedding(embeds)
        h = self.pre_fc_norm_hidden(hidden.to(e.dtype))
        y = self.fc(torch.cat([e, h], dim=-1))
        y = self.layers[0](y, position_embeddings=position_embeddings,
                           attention_mask=attention_mask, past_key_values=cache,
                           use_cache=True)
        if isinstance(y, tuple):
            y = y[0]
        return self.norm(y)


# ---------------------------------------------------------------------------
# model plumbing
# ---------------------------------------------------------------------------
def _first_submodule(model: nn.Module, paths: Sequence[str]) -> nn.Module:
    for p in paths:
        try:
            return model.get_submodule(p)
        except AttributeError:
            continue
    raise AttributeError(f"none of {paths} in {type(model).__name__}")


def final_norm(model):
    return _first_submodule(model, ("model.language_model.norm", "model.norm"))


def rotary(model):
    return _first_submodule(model, ("model.language_model.rotary_emb", "model.rotary_emb"))


def text_model(model):
    return _first_submodule(model, ("model.language_model", "model"))


def new_cache(config):
    from transformers import DynamicCache

    try:
        return DynamicCache(config=config)
    except TypeError:
        return DynamicCache()


def cache_len(cache) -> int:
    return int(cache.get_seq_length())


def snapshot_recurrent(cache) -> List[Any]:
    """Clone every linear-attention layer's conv/recurrent state (small)."""
    snap = []
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for i, layer in enumerate(layers):
            if hasattr(layer, "recurrent_states") or hasattr(layer, "conv_states"):
                conv = getattr(layer, "conv_states", None)
                rec = getattr(layer, "recurrent_states", None)
                snap.append((i,
                             None if conv is None else conv.clone(),
                             None if rec is None else rec.clone(),
                             getattr(layer, "has_previous_state", None)))
        return snap
    for attr in ("conv_states", "recurrent_states", "ssm_states"):
        lst = getattr(cache, attr, None)
        if isinstance(lst, list):
            snap.append((attr, [None if t is None else t.clone() for t in lst]))
    return snap


def restore_recurrent(cache, snap) -> None:
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for i, conv, rec, has_prev in snap:
            layer = layers[i]
            if conv is not None:
                layer.conv_states.copy_(conv)
            if rec is not None:
                layer.recurrent_states.copy_(rec)
            if has_prev is not None:
                layer.has_previous_state = has_prev
        return
    for attr, lst in snap:
        cur = getattr(cache, attr)
        for j, t in enumerate(lst):
            if t is not None:
                cur[j].copy_(t)


def crop_attention(cache, length: int) -> None:
    if hasattr(cache, "crop"):
        cache.crop(int(length))
        return
    for attr in ("key_cache", "value_cache"):
        lst = getattr(cache, attr, None)
        if isinstance(lst, list):
            for j, t in enumerate(lst):
                if t is not None and t.dim() >= 3 and t.shape[-2] > length:
                    lst[j] = t[..., :length, :]


class _Capture:
    """Forward hooks: final-norm output (h_p) and the trunk's position ids."""

    def __init__(self, model):
        self.hidden: Optional[torch.Tensor] = None
        self.position_ids: Optional[torch.Tensor] = None
        self._h1 = final_norm(model).register_forward_hook(self._on_norm)
        self._h2 = text_model(model).register_forward_pre_hook(self._on_text, with_kwargs=True)

    def _on_norm(self, _m, _a, out):
        self.hidden = out

    def _on_text(self, _m, _args, kwargs):
        p = kwargs.get("position_ids")
        if isinstance(p, torch.Tensor):
            if p.dim() == 3 and p.shape[0] == 4:
                p = p[1:]
            self.position_ids = p
        else:
            self.position_ids = None

    def remove(self):
        self._h1.remove()
        self._h2.remove()


def _mrope_positions(pos_1d: torch.Tensor, delta: torch.Tensor | int) -> torch.Tensor:
    """[T] text positions -> [3, 1, T] M-RoPE positions (+ rope delta)."""
    p = pos_1d.view(1, 1, -1) + (delta if isinstance(delta, int) else delta.view(1, -1, 1))
    return p.expand(3, 1, -1)


def _rope_delta(model) -> Any:
    base = getattr(model, "model", None)
    d = getattr(base, "rope_deltas", None)
    return 0 if d is None else d.to(torch.long)


def _causal_mask(t: int, past: int, device) -> torch.Tensor:
    q = torch.arange(t, device=device).view(t, 1) + past
    k = torch.arange(past + t, device=device).view(1, past + t)
    return (k <= q).view(1, 1, t, past + t)


# ---------------------------------------------------------------------------
# the speculative loop
# ---------------------------------------------------------------------------
class MTPSpeculator:
    def __init__(self, model: nn.Module, mtp: QwenMTP, *, position_offset: int = 1):
        self.model = model
        self.mtp = mtp
        self.offset = int(position_offset)
        self.embed = model.get_input_embeddings()
        self.lm_head = model.get_output_embeddings()
        self.rot = rotary(model)

    def _draft(self, hidden, tokens, positions3, mtp_cache) -> int:
        emb = self.embed(tokens)
        dev = emb.device
        past = cache_len(mtp_cache) if len(getattr(mtp_cache, "layers", [])) else 0
        cos_sin = self.rot(emb, positions3.to(dev))
        mask = _causal_mask(tokens.shape[1], past, dev)
        y = self.mtp(hidden, emb, cos_sin, mask, mtp_cache)
        logits = self.lm_head(y[:, -1:, :])
        return int(logits[0, -1].argmax().item())

    @torch.no_grad()
    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        *,
        max_new_tokens: int,
        eos_ids: Sequence[int],
        on_token: Callable[[int], bool],
    ) -> Dict[str, Any]:
        """Greedy decode with MTP drafts.  ``on_token`` returns False to stop."""
        model = self.model
        eos = set(int(e) for e in eos_ids)
        cap = _Capture(model)
        stats = {"verify_steps": 0, "accepted": 0, "rejected": 0, "rerun_steps": 0,
                 "emitted": 0, "mtp_position_offset": self.offset}
        t0 = time.perf_counter()
        try:
            out = model(**inputs, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            ids = inputs["input_ids"]
            dev = ids.device
            plen = ids.shape[1]
            nxt = int(out.logits[0, -1].argmax().item())
            hid = cap.hidden                                 # [1, P, H]
            pos = cap.position_ids
            delta = _rope_delta(model)
            if pos is None:
                pos3 = _mrope_positions(torch.arange(plen, device=dev), 0)
            else:
                pos3 = pos.to(dev)
            if self.offset == 1:
                last = pos3[:, :, -1:] + 1
                mpos = torch.cat([pos3[:, :, 1:], last], dim=-1)
            else:
                mpos = pos3
            mtp_cache = new_cache(self.mtp.config)
            toks = torch.cat([ids[:, 1:], torch.tensor([[nxt]], device=dev)], dim=1)
            draft = self._draft(hid, toks, mpos, mtp_cache)
            done = False

            def emit(tok: int) -> bool:
                stats["emitted"] += 1
                if tok in eos:
                    return False
                keep = on_token(tok)
                return bool(keep) and stats["emitted"] < int(max_new_tokens)

            if not emit(nxt):
                done = True
            x_last = nxt
            while not done:
                L = cache_len(cache)
                snap = snapshot_recurrent(cache)
                step = torch.tensor([[x_last, draft]], device=dev)
                out = model(input_ids=step, past_key_values=cache, use_cache=True,
                            logits_to_keep=2)
                stats["verify_steps"] += 1
                h2 = cap.hidden
                y1 = int(out.logits[0, 0].argmax().item())
                if y1 == draft:
                    stats["accepted"] += 1
                    y2 = int(out.logits[0, 1].argmax().item())
                    if not emit(draft):
                        break
                    if not emit(y2):
                        break
                    p = torch.tensor([L + 1, L + 2], device=dev) if self.offset == 1 \
                        else torch.tensor([L, L + 1], device=dev)
                    draft = self._draft(h2, torch.tensor([[draft, y2]], device=dev),
                                        _mrope_positions(p, delta), mtp_cache)
                    x_last = y2
                else:
                    stats["rejected"] += 1
                    restore_recurrent(cache, snap)
                    crop_attention(cache, L)
                    out1 = model(input_ids=torch.tensor([[x_last]], device=dev),
                                 past_key_values=cache, use_cache=True, logits_to_keep=1)
                    stats["rerun_steps"] += 1
                    y = int(out1.logits[0, -1].argmax().item())
                    stats["rerun_disagreed_with_verify"] = stats.get(
                        "rerun_disagreed_with_verify", 0) + int(y != y1)
                    if not emit(y):
                        break
                    p = torch.tensor([L + 1], device=dev) if self.offset == 1 \
                        else torch.tensor([L], device=dev)
                    draft = self._draft(cap.hidden, torch.tensor([[y]], device=dev),
                                        _mrope_positions(p, delta), mtp_cache)
                    x_last = y
        finally:
            cap.remove()
        n = stats["accepted"] + stats["rejected"]
        stats["acceptance_rate"] = (stats["accepted"] / n) if n else None
        stats["tokens_per_target_forward"] = (
            stats["emitted"] / max(1, 1 + stats["verify_steps"] + stats["rerun_steps"]))
        stats["seconds"] = round(time.perf_counter() - t0, 4)
        return stats


__all__ = [
    "MTPSpeculator",
    "QwenMTP",
    "cache_len",
    "crop_attention",
    "final_norm",
    "new_cache",
    "restore_recurrent",
    "snapshot_recurrent",
]
