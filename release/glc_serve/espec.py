"""E-SPEC: MTP speculative greedy that is BITWISE the same engine's plain greedy.

Why the F4 MTP path (``glc_serve.mtp.MTPSpeculator``) matched plain greedy on
only 3/4 probes: its verify step feeds ``[x_last, draft]`` as ONE M=2 forward.
M=2 takes different code than an M=1 decode step -- the Gated-DeltaNet layers
run the CHUNKED kernel (``chunk_gated_delta_rule`` + ``causal_conv1d_fn``)
instead of the per-step recurrence, every projection is an M=2 GEMM, and
attention sees two queries -- and on acceptance that M=2 state is KEPT.  From
then on the cache holds bits plain greedy never computes, and any bf16
near-tie flips a token.

This module verifies ``k`` MTP drafts as ``k + 1`` SEQUENTIAL M=1 calls, so
every target forward is exactly a plain-greedy decode step (same kernels, same
shapes, same state), and on a rejection it rolls the Gated-DeltaNet state back
LAZILY (design E4, an internal kernel design note, section 3):

* before the verify cycle it snapshots the per-layer conv/recurrent state S_0;
* during the k+1 M=1 calls it records, per GDN layer and position, the exact
  arguments of the two state-updating calls (``causal_conv1d_update`` and the
  recurrent gated-delta rule) -- the compact factors (qkv column, q, k, v, g,
  beta), not the 151 MB state;
* when only ``a < k`` drafts are accepted it restores S_0 and RE-APPLIES the
  recorded updates of the ``a + 1`` correct positions with the same
  functions, in the same order -> the same bits as ``a + 1`` plain steps;
  full-attention KV is cropped to the accepted length (KV rows of accepted
  positions were computed from a correct prefix, so they are already exact).

``check_state=True`` additionally snapshots the state after every M=1 call
(eager) and compares the lazily rebuilt state against it bit for bit.

Greedy, batch 1.  Default-off: nothing uses this module unless a caller does.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .mtp import (
    _Capture,
    _causal_mask,
    _mrope_positions,
    _rope_delta,
    cache_len,
    new_cache,
    rotary,
)


def _gdn_modules(model: nn.Module) -> List[nn.Module]:
    return [m for m in model.modules()
            if hasattr(m, "recurrent_gated_delta_rule") and hasattr(m, "causal_conv1d_update")
            and hasattr(m, "layer_idx")]


def _clone_arg(a):
    return a.clone() if isinstance(a, torch.Tensor) else a


class GDNTape:
    """Records the state-updating calls of every GDN layer during M=1 steps.

    Installs instance-level wrappers (shadowing the class attributes set in
    ``__init__``) around ``causal_conv1d_update`` and
    ``recurrent_gated_delta_rule``.  Wrappers are pass-through unless
    ``recording`` is True; ``remove()`` restores the originals.
    """

    def __init__(self, model: nn.Module):
        self.mods = _gdn_modules(model)
        if not self.mods:
            raise RuntimeError("no Gated-DeltaNet layers found")
        self.recording = False
        self.steps: List[Dict[int, Dict[str, Any]]] = []
        self._orig: Dict[int, Tuple[Callable, Callable]] = {}
        for m in self.mods:
            oc, orc = m.causal_conv1d_update, m.recurrent_gated_delta_rule
            self._orig[id(m)] = (oc, orc)
            m.causal_conv1d_update = self._conv_wrapper(m, oc)
            m.recurrent_gated_delta_rule = self._rec_wrapper(m, orc)

    def _conv_wrapper(self, m, orig):
        def conv(*args, **kwargs):
            if self.recording:
                # args[1] is the live conv state: replaced at replay time
                self.steps[-1].setdefault(m.layer_idx, {})["conv"] = (
                    [_clone_arg(a) if i != 1 else None for i, a in enumerate(args)],
                    {k: _clone_arg(v) for k, v in kwargs.items()})
            return orig(*args, **kwargs)
        return conv

    def _rec_wrapper(self, m, orig):
        def rec(*args, **kwargs):
            if self.recording:
                self.steps[-1].setdefault(m.layer_idx, {})["rec"] = (
                    [_clone_arg(a) for a in args],
                    {k: _clone_arg(v) for k, v in kwargs.items() if k != "initial_state"})
            return orig(*args, **kwargs)
        return rec

    def new_step(self) -> None:
        self.steps.append({})

    def clear(self) -> None:
        self.steps = []

    def replay(self, cache, n_steps: int) -> None:
        """Re-apply the first ``n_steps`` recorded M=1 updates onto ``cache``."""
        for j in range(n_steps):
            rec = self.steps[j]
            for m in self.mods:
                e = rec.get(m.layer_idx)
                if e is None or "conv" not in e or "rec" not in e:
                    raise RuntimeError(f"step {j}: layer {m.layer_idx} was not recorded "
                                       "(not an M=1 cached decode step?)")
                oc, orc = self._orig[id(m)]
                layer = cache.layers[m.layer_idx]
                cargs, ckw = e["conv"]
                cargs = list(cargs)
                cargs[1] = layer.conv_states
                oc(*cargs, **ckw)                              # in-place, as in forward
                rargs, rkw = e["rec"]
                _, last = orc(*rargs, initial_state=layer.recurrent_states, **rkw)
                cache.update_recurrent_state(last, m.layer_idx)

    def remove(self) -> None:
        for m in self.mods:
            oc, orc = self._orig[id(m)]
            m.causal_conv1d_update = oc
            m.recurrent_gated_delta_rule = orc


def snapshot_gdn(cache, mods) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    return {m.layer_idx: (cache.layers[m.layer_idx].conv_states.clone(),
                          cache.layers[m.layer_idx].recurrent_states.clone()) for m in mods}


def restore_gdn(cache, snap) -> None:
    for i, (conv, rec) in snap.items():
        cache.layers[i].conv_states.copy_(conv)
        cache.layers[i].recurrent_states.copy_(rec)


def _bits(t: torch.Tensor) -> torch.Tensor:
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.view(torch.int16)
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    return t


def gdn_equal(cache, snap) -> Tuple[int, int]:
    """(#layers whose conv state differs, #layers whose recurrent state differs), bitwise."""
    dc = dr = 0
    for i, (conv, rec) in snap.items():
        lay = cache.layers[i]
        dc += int(not torch.equal(_bits(lay.conv_states), _bits(conv)))
        dr += int(not torch.equal(_bits(lay.recurrent_states), _bits(rec)))
    return dc, dr


def logits_digest(row: torch.Tensor) -> str:
    return hashlib.sha256(row.detach().contiguous().view(torch.int16).cpu().numpy()
                          .tobytes()).hexdigest()[:32]


class ExactSpeculator:
    """Plain greedy and MTP-k speculative greedy through the SAME M=1 calls."""

    def __init__(self, model: nn.Module, mtp: Optional[nn.Module], *, device,
                 check_state: bool = False, digest_logits: bool = True):
        self.model = model
        self.mtp = mtp
        self.check_state = bool(check_state)
        self.digest = bool(digest_logits)
        self.embed = model.get_input_embeddings()
        self.lm_head = model.get_output_embeddings()
        self.rot = rotary(model)
        self.dev = torch.device(device)

    # -- shared engine calls ------------------------------------------------
    def _reset_positions(self) -> None:
        base = getattr(self.model, "model", None)
        if base is not None and hasattr(base, "rope_deltas"):
            base.rope_deltas = None      # never inherit a previous prompt's M-RoPE delta

    def _prefill(self, inputs):
        self._reset_positions()
        out = self.model(**inputs, use_cache=True, logits_to_keep=1)
        return out.past_key_values, out.logits[0, -1]

    def _step(self, tok: int, cache):
        out = self.model(input_ids=torch.tensor([[int(tok)]], device=self.dev),
                         past_key_values=cache, use_cache=True, logits_to_keep=1)
        return out.logits[0, -1]

    # -- plain greedy ---------------------------------------------------------
    @torch.no_grad()
    def plain(self, inputs: Dict[str, torch.Tensor], *, max_new_tokens: int,
              eos_ids: Sequence[int]) -> Dict[str, Any]:
        eos = set(int(e) for e in eos_ids)
        t0 = time.perf_counter()
        cache, logits = self._prefill(inputs)
        toks, dig = [], []
        calls = 0
        while True:
            t = int(logits.argmax().item())
            toks.append(t)
            if self.digest:
                dig.append(logits_digest(logits))
            if t in eos or len(toks) >= max_new_tokens:
                break
            logits = self._step(t, cache)
            calls += 1
        return {"tokens": toks, "digests": dig, "target_m1_calls": calls,
                "seconds": round(time.perf_counter() - t0, 4)}

    # -- MTP drafting -----------------------------------------------------------
    def _mtp_call(self, hidden, tokens, positions3, mtp_cache, past: int):
        emb = self.embed(tokens)
        dev = emb.device
        cos_sin = self.rot(emb, positions3.to(dev))
        mask = _causal_mask(tokens.shape[1], past, dev)
        y = self.mtp(hidden.to(dev), emb, cos_sin, mask, mtp_cache)
        last = y[:, -1:, :]
        logits = self.lm_head(last)
        return last, int(logits[0, -1].argmax().item())

    # -- speculative greedy ----------------------------------------------------
    @torch.no_grad()
    def speculative(self, inputs: Dict[str, torch.Tensor], *, k: int, max_new_tokens: int,
                    eos_ids: Sequence[int]) -> Dict[str, Any]:
        if self.mtp is None:
            raise RuntimeError("no MTP head loaded")
        if k < 1:
            raise ValueError("k >= 1")
        eos = set(int(e) for e in eos_ids)
        model, dev = self.model, self.dev
        cap = _Capture(model)
        tape = GDNTape(model)
        st = {"k": k, "cycles": 0, "drafts_proposed": 0, "drafts_accepted": 0,
              "accept_len_hist": [0] * (k + 1), "target_m1_calls": 0, "rollbacks": 0,
              "replayed_steps": 0, "state_check_cycles": 0, "state_mismatch_cycles": 0,
              "state_mismatch_layers": 0}
        toks: List[int] = []
        dig: List[str] = []
        t0 = time.perf_counter()
        try:
            cache, logits = self._prefill(inputs)
            ids = inputs["input_ids"]
            plen = int(ids.shape[1])
            hid = cap.hidden                                   # [1, P, H]
            pos = cap.position_ids
            delta = _rope_delta(model)
            pos3 = (_mrope_positions(torch.arange(plen, device=dev), 0) if pos is None
                    else pos.to(dev))
            mpos = torch.cat([pos3[:, :, 1:], pos3[:, :, -1:] + 1], dim=-1)

            def emit(t: int, row) -> bool:
                toks.append(int(t))
                if self.digest:
                    dig.append(logits_digest(row))
                return not (int(t) in eos or len(toks) >= max_new_tokens)

            x_last = int(logits.argmax().item())
            if emit(x_last, logits):
                mtp_cache = new_cache(self.mtp.config)
                first = torch.cat([ids[:, 1:], torch.tensor([[x_last]], device=dev)], dim=1)
                g, d = self._mtp_call(hid, first, mpos, mtp_cache, past=0)
                L = plen                                       # target tokens consumed
                done = False
                while not done:
                    st["cycles"] += 1
                    # ---- draft k tokens (chain the MTP head on its own output) ----
                    drafts = [d]
                    gi = g
                    for i in range(1, k):
                        p = _mrope_positions(torch.tensor([L + i], device=dev), delta)
                        gi, di = self._mtp_call(gi, torch.tensor([[drafts[-1]]], device=dev), p,
                                                mtp_cache, past=L + i - 1)
                        drafts.append(di)
                    # ---- verify: k+1 sequential M=1 calls, recorded ----
                    s0 = snapshot_gdn(cache, tape.mods)
                    tape.clear()
                    eager = []
                    rows, hiddens = [], []
                    feed = [x_last] + drafts
                    tape.recording = True
                    try:
                        for i, t_in in enumerate(feed):
                            tape.new_step()
                            rows.append(self._step(t_in, cache))
                            hiddens.append(cap.hidden)
                            st["target_m1_calls"] += 1
                            if self.check_state:
                                eager.append(snapshot_gdn(cache, tape.mods))
                    finally:
                        tape.recording = False
                    outs = [int(r.argmax().item()) for r in rows]   # t_1 .. t_{k+1}
                    a = 0
                    while a < k and drafts[a] == outs[a]:
                        a += 1
                    st["drafts_proposed"] += k
                    st["drafts_accepted"] += a
                    st["accept_len_hist"][a] += 1
                    # ---- emit t_1..t_{a+1} ----
                    for i in range(a + 1):
                        if not emit(outs[i], rows[i]):
                            done = True
                            break
                    if done:
                        break
                    # ---- make the target state equal a+1 plain steps ----
                    if a < k:
                        st["rollbacks"] += 1
                        restore_gdn(cache, s0)
                        cache.crop(L + a + 1)
                        tape.replay(cache, a + 1)
                        st["replayed_steps"] += a + 1
                        if self.check_state:
                            dc, dr = gdn_equal(cache, eager[a])
                            st["state_check_cycles"] += 1
                            if dc or dr:
                                st["state_mismatch_cycles"] += 1
                                st["state_mismatch_layers"] += dc + dr
                    if cache_len(cache) != L + a + 1:
                        raise RuntimeError(f"attention KV length {cache_len(cache)} != {L + a + 1}")
                    # ---- MTP: drop speculative entries, add the confirmed pairs ----
                    mtp_cache.crop(L)
                    conf_h = torch.cat(hiddens[:a + 1], dim=1)     # h_L .. h_{L+a}
                    conf_t = torch.tensor([outs[:a + 1]], device=dev)
                    p = _mrope_positions(torch.arange(L + 1, L + a + 2, device=dev), delta)
                    g, d = self._mtp_call(conf_h, conf_t, p, mtp_cache, past=L)
                    x_last = outs[a]
                    L = L + a + 1
                    del s0, eager, rows, hiddens
        finally:
            tape.remove()
            cap.remove()
        n = st["drafts_proposed"]
        st["draft_acceptance"] = (st["drafts_accepted"] / n) if n else None
        st["mean_tokens_per_cycle"] = ((st["drafts_accepted"] + st["cycles"]) / st["cycles"]
                                       if st["cycles"] else None)
        st["seconds"] = round(time.perf_counter() - t0, 4)
        return {"tokens": toks, "digests": dig, "stats": st}


__all__ = ["ExactSpeculator", "GDNTape", "gdn_equal", "logits_digest", "restore_gdn",
           "snapshot_gdn"]
