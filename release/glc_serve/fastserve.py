"""In-process serving API over FastDecoder (batch 1, streaming).

    from glc_serve.fastserve import FastSession
    s = FastSession.load(bundle="/opt/kernel/bundle", tune="fastdec_tune.json")   # or dense=/q8_gguf=
    st = s.prefill(messages=[{"role": "user", "content": "Hi"}])                  # text or image chat
    for tok in s.generate(st, max_tokens=256, temperature=0.0, spec_k=5):
        print(s.tokenizer.decode([tok]), end="")
    s.extend(st, token_ids)          # append tokens (next chat turn) through the fast engine
    s.rollback(st, n)                # drop the last n tokens (n <= 15)

Contracts
* Greedy (temperature 0): speculative output is BITWISE the non-speculative output of the
  same engine (E-SPEC; gated on 6 probes + 200 gsm8k).  For the TBE codec, logits are
  bitwise the bf16 parent's through the same engine (G3a).
* Sampling (temperature > 0, top_p): speculative sampling with the MTP head's greedy draft
  (a point-mass proposal).  A draft d is accepted with probability p(d); on rejection the
  token is drawn from p with d removed and renormalised; after k acceptances the bonus
  token is drawn from the last row.  The output distribution is exactly the target
  distribution p (Leviathan et al. 2023 with q = one-hot).  Without spec_k it is plain
  ancestral sampling.  Seeded by ``seed``.
* Images: prefill runs through the HF model (vision tower, M-RoPE positions), whose
  caches are imported into the fast engine; decode is text.  ``prefill`` accepts the same
  ``messages`` (with ``{"type": "image"}`` parts + ``images=[PIL...]``) as glc_serve.
* One live sequence per session (the engine's buffers).  ``prefill``/``extend`` of another
  state re-imports; a State keeps the HF prefill cache so it can be regenerated.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

import torch

from .fastdec import MAX_M, FastDecoder, load_tune


@dataclass
class State:
    pf: Dict[str, Any]
    tokens: List[int] = field(default_factory=list)   # generated tokens (incl. the prefill's)
    plen: int = 0
    active: bool = False


class FastSession:
    def __init__(self, engine, fd: FastDecoder, meta: Dict[str, Any]):
        self.engine, self.fd, self.meta = engine, fd, meta
        self.model = engine.model
        self.mtp = engine.loaded.mtp
        self.tokenizer = engine.tokenizer if hasattr(engine, "tokenizer") else engine.processor
        self.eos_ids = self._stop_ids(engine, meta.get("source_dir"))
        self._live: Optional[State] = None
        from .espec import ExactSpeculator

        self._spec = ExactSpeculator(self.model, self.mtp, device=engine.device, digest_logits=False)

    @staticmethod
    def _stop_ids(engine, source_dir) -> List[int]:
        """Engine eos + the checkpoint's generation_config.json eos + <|im_end|>.

        A model built from a serving bundle's config.json skeleton does not read the
        bundle's generation_config.json, so glc_serve's own eos list is [248044] only and
        chat turns would run past <|im_end|> (248046); measured 2026-09-25."""
        import json
        from pathlib import Path

        ids = set(int(e) for e in engine.eos_ids)
        if source_dir:
            gp = Path(source_dir) / "generation_config.json"
            if gp.is_file():
                e = json.loads(gp.read_text()).get("eos_token_id")
                ids |= set(int(v) for v in (e if isinstance(e, list) else [e]) if v is not None)
        tok = getattr(engine, "tok", None)
        if tok is not None:
            try:
                im = tok.convert_tokens_to_ids("<|im_end|>")
                if isinstance(im, int) and im >= 0:
                    ids.add(int(im))
            except Exception:  # noqa: BLE001
                pass
        return sorted(ids)

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, *, bundle: Optional[str] = None, dense: Optional[str] = None,
             q8_gguf: Optional[str] = None, tune: str, device: str = "cuda:0",
             max_len: int = 32768, spec_ks: Sequence[int] = range(1, MAX_M),
             server_flags: Sequence[str] = (), log=print) -> "FastSession":
        from .server import _parse, build_state

        argv = list(server_flags) + ["--device", device, "--port", "0"]
        if bundle:
            argv += ["--bundle", bundle, "--backend", "tbe", "--exec-mode", "exact"]
        elif dense or q8_gguf:
            argv += ["--dense", dense]
        else:
            raise ValueError("bundle= or dense= required")
        t0 = time.time()
        state = build_state(_parse(argv))
        engine = state.engine
        tn = load_tune(tune)
        meta: Dict[str, Any] = {"server_argv": argv, "load_s": None,
                                "source_dir": bundle if bundle and "://" not in bundle else dense}
        if q8_gguf:                     # any GGUF (Q8_0 fused; registered types; else dense)
            from .q8serve import load_gguf

            meta["q8"] = load_gguf(engine.model, q8_gguf, tn, device=engine.device, log=log)
        fd = FastDecoder(engine.model, engine.loaded.mtp, tune=tn, max_len=max_len,
                         device=engine.device)
        if fd.descriptor_kinds().get("tbe"):
            from .tbe_desc import tune_tbe_per_m as tune_tbe

            meta["tbe_tune"] = tune_tbe(fd._descs(), log=log)
        kinds = [("ar", 1)] + [("verify", m) for m in range(1, MAX_M + 1)]
        if fd.mtp_ready:
            kinds += [("mtp", m) for m in range(1, MAX_M + 1)]
        meta["capture_s"] = fd.capture(kinds)
        meta["load_s"] = round(time.time() - t0, 1)
        meta["descriptors"] = fd.descriptor_kinds()
        meta["streamed_bytes_per_step"] = fd.streamed_bytes()
        log(f"[fastserve] ready in {meta['load_s']} s; {meta['descriptors']}")
        return cls(engine, fd, meta)

    # ------------------------------------------------------------------ prefill
    @torch.no_grad()
    def prefill(self, *, messages=None, prompt: Optional[str] = None, images=None,
                token_ids: Optional[Sequence[int]] = None, chat_template_kwargs=None) -> State:
        from .engine import Request, SamplingParams
        from .mtp import _Capture, _mrope_positions, _rope_delta, new_cache
        from .espec import logits_digest

        dev = self.engine.device
        if token_ids is not None:
            ids = torch.tensor([list(token_ids)], device=dev)
            inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        else:
            kw = chat_template_kwargs if chat_template_kwargs is not None else {"enable_thinking": False}
            if messages is not None:
                r = Request(kind="chat", messages=messages, chat_template_kwargs=kw,
                            params=SamplingParams(max_tokens=1, temperature=0.0))
            else:
                r = Request(kind="completion", prompt=prompt, chat_template_kwargs=kw,
                            params=SamplingParams(max_tokens=1, temperature=0.0))
            if images:
                r.images = list(images)
            inputs = self.engine.prepare([r])
        spec = self._spec
        cap = _Capture(self.model)
        try:
            cache, logits = spec._prefill(inputs)
            hid, pos = cap.hidden, cap.position_ids
        finally:
            cap.remove()
        ids = inputs["input_ids"]
        plen = int(ids.shape[1])
        if plen + 16 > self.fd.max_len:
            raise ValueError(f"prompt of {plen} tokens exceeds max_len {self.fd.max_len}")
        delta = _rope_delta(self.model)
        delta = int(delta.reshape(-1)[0].item()) if isinstance(delta, torch.Tensor) else int(delta)
        pf = {"cache": cache, "plen": plen, "delta": delta, "logits": logits.detach().clone(),
              "d0": logits_digest(logits), "mc": None, "g": None, "d1": None, "hid": hid,
              "pos3": pos, "input_ids": ids}
        st = State(pf=pf, plen=plen)
        self._import(st, first=None)
        return st

    def _import(self, st: State, first: Optional[int]) -> None:
        pf = st.pf
        x_last = int(pf["logits"].argmax().item()) if first is None else int(first)
        if self.fd.mtp_ready:
            from .mtp import _mrope_positions, new_cache

            dev = self.engine.device
            plen = pf["plen"]
            pos3 = (_mrope_positions(torch.arange(plen, device=dev), 0) if pf["pos3"] is None
                    else pf["pos3"].to(dev))
            mpos = torch.cat([pos3[:, :, 1:], pos3[:, :, -1:] + 1], dim=-1)
            mc = new_cache(self.mtp.config)
            first_ids = self._first_ids(st, x_last)
            g, d1 = self._spec._mtp_call(pf["hid"], first_ids, mpos, mc, past=0)
            pf["mc"], pf["g"], pf["d1"] = mc, g, d1
        self.fd.import_prefill(pf["cache"], pf["plen"], pf["delta"], x_last, pf["mc"], pf["g"],
                               pf["d1"])
        st.tokens = [x_last]
        st.active = True
        self._live = st

    def _first_ids(self, st: State, x_last: int) -> torch.Tensor:
        ids = st.pf.get("input_ids")
        if ids is None:
            raise RuntimeError("internal: prefill ids missing")
        return torch.cat([ids[:, 1:], torch.tensor([[x_last]], device=ids.device)], dim=1)

    # ------------------------------------------------------------------ generate
    @torch.no_grad()
    def generate(self, st: State, *, max_tokens: int, stop_ids: Optional[Sequence[int]] = None,
                 temperature: float = 0.0, top_p: float = 1.0, seed: Optional[int] = None,
                 spec_k: Optional[int] = None) -> Iterator[int]:
        """Stream up to ``max_tokens`` NEW tokens (the prefill's first token counts as the
        first).  Stops after a stop/eos token (which is yielded)."""
        if self._live is not st or not st.active:
            raise RuntimeError("state is not the live sequence; call prefill/resume first")
        stops = set(int(t) for t in (stop_ids if stop_ids is not None else self.eos_ids))
        fd = self.fd
        gen = torch.Generator(device=fd.dev)
        if seed is not None:
            gen.manual_seed(int(seed))
        n = 0
        # the first token comes from the prefill logits
        if len(st.tokens) == 1 and not st.pf.get("first_emitted"):
            t0 = st.tokens[0]
            if temperature > 0:
                t0 = self._sample(st.pf["logits"].reshape(1, -1), temperature, top_p, gen)[0]
                if t0 != st.tokens[0]:
                    self._import(st, first=t0)
            st.pf["first_emitted"] = True
            n += 1
            yield t0
            if t0 in stops or n >= max_tokens:
                return
        k = int(spec_k or 0)
        if k and not fd.mtp_ready:
            k = 0
        while n < max_tokens:
            L = st.plen + len(st.tokens) - 1          # position of the last token (engine input)
            if k == 0:
                fd.pos.fill_(L)
                fd.replay("verify", 1)
                if temperature > 0:
                    t = self._sample(fd.logits[:1], temperature, top_p, gen)[0]
                else:
                    t = int(fd.am[0].item())
                fd.vtok[0] = t
                st.tokens.append(t)
                n += 1
                yield t
                if t in stops:
                    return
                continue
            outs = self._spec_cycle(st, k, L, temperature, top_p, gen)
            for t in outs:
                n += 1
                yield t
                if t in stops or n >= max_tokens:
                    # drop over-generated positions from the engine's view
                    extra = len(outs) - outs.index(t) - 1
                    if extra:
                        st.tokens = st.tokens[:len(st.tokens) - extra]
                        self._resync(st)
                    return

    def _spec_cycle(self, st: State, k: int, L: int, temperature: float, top_p: float, gen) -> List[int]:
        fd = self.fd
        fd.pos.fill_(L)
        for _ in range(k - 1):
            fd.replay("mtp", 1)
        fd.replay("verify", k + 1)
        drafts = fd.vtok[1:k + 1].tolist()
        if temperature <= 0:
            outs = fd.am[:k + 1].tolist()
            a = 0
            while a < k and drafts[a] == outs[a]:
                a += 1
            emitted = outs[:a + 1]
        else:
            p = self._probs(fd.logits[:k + 1], temperature, top_p)
            u = torch.rand(k + 1, generator=gen, device=fd.dev)
            emitted = []
            a = 0
            while a < k:
                d = drafts[a]
                if float(u[a]) < float(p[a, d]):
                    emitted.append(d)
                    a += 1
                    continue
                q = p[a].clone()
                q[d] = 0
                s = q.sum()
                t = int(torch.multinomial(q / s, 1, generator=gen).item()) if float(s) > 0 else d
                emitted.append(t)
                break
            if a == k:
                emitted.append(int(torch.multinomial(p[k], 1, generator=gen).item()))
            a = len(emitted) - 1
        # confirm the accepted prefix into the MTP and set the next input
        am_rows = torch.tensor(emitted, dtype=torch.int32, device=fd.dev)
        fd.mhid[:a + 1].copy_(fd.xf[:a + 1])
        fd.mtok[:a + 1].copy_(am_rows[:a + 1])
        fd.vtok[0] = emitted[-1]
        fd.mkv.fill_(L)
        fd.slot.fill_(1)
        fd.replay("mtp", a + 1)
        st.tokens.extend(emitted)
        return emitted

    def _resync(self, st: State) -> None:
        """After dropping trailing tokens: the engine input is the (new) last token."""
        fd = self.fd
        fd.vtok[0] = st.tokens[-1]
        fd.pos.fill_(st.plen + len(st.tokens) - 1)
        if fd.mtp_ready:            # MTP needs one confirm step at the new position
            self._mtp_confirm_last(st)

    def _mtp_confirm_last(self, st: State) -> None:
        # re-derive the MTP draft for the last token: run a verify M=1 at the previous
        # position to get its trunk hidden, then a 1-row MTP confirm.  (Rare path.)
        fd = self.fd
        L = st.plen + len(st.tokens) - 2
        if L < st.plen:
            return
        fd.vtok[0] = st.tokens[-2]
        fd.pos.fill_(L)
        fd.replay("verify", 1)
        fd.mhid[:1].copy_(fd.xf[:1])
        fd.mtok[0] = st.tokens[-1]
        fd.mkv.fill_(L)
        fd.slot.fill_(1)
        fd.replay("mtp", 1)
        fd.vtok[0] = st.tokens[-1]
        fd.pos.fill_(L + 1)

    # ------------------------------------------------------------------ sampling helpers
    @staticmethod
    def _probs(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        p = torch.softmax(logits.float() / float(temperature), dim=-1)
        if top_p < 1.0:
            sp, si = torch.sort(p, dim=-1, descending=True)
            cum = torch.cumsum(sp, dim=-1)
            drop = (cum - sp) > top_p
            sp = sp.masked_fill(drop, 0.0)
            p = torch.zeros_like(p).scatter_(-1, si, sp)
            p = p / p.sum(dim=-1, keepdim=True)
        return p

    def _sample(self, logits, temperature, top_p, gen) -> List[int]:
        p = self._probs(logits, temperature, top_p)
        return torch.multinomial(p, 1, generator=gen).reshape(-1).tolist()

    # ------------------------------------------------------------------ extend / rollback
    @torch.no_grad()
    def extend(self, st: State, token_ids: Sequence[int]) -> None:
        """Append tokens (e.g. the next user turn) through the fast engine, M <= 8 per pass.
        The last generated token must already be in ``st.tokens``; afterwards the engine's
        next input is the last appended token."""
        if self._live is not st:
            raise RuntimeError("state is not live")
        fd = self.fd
        ids = list(int(t) for t in token_ids)
        seq = [st.tokens[-1]] + ids
        L = st.plen + len(st.tokens) - 1
        # feed all but the final token; the final one becomes the next decode input.  Each
        # chunk also runs the MTP head on its (hidden, next token) pairs so drafting resumes
        # with a complete MTP cache.
        feed = seq[:-1]
        i = 0
        while i < len(feed):
            m = min(MAX_M, len(feed) - i)
            fd.vtok[:m].copy_(torch.tensor(feed[i:i + m], dtype=torch.int32, device=fd.dev))
            fd.pos.fill_(L + i)
            fd.replay("verify", m)
            if fd.mtp_ready:
                fd.mhid[:m].copy_(fd.xf[:m])
                fd.mtok[:m].copy_(torch.tensor(seq[i + 1:i + m + 1], dtype=torch.int32,
                                               device=fd.dev))
                fd.mkv.fill_(L + i)
                fd.slot.fill_(1)
                fd.replay("mtp", m)
            i += m
        st.tokens.extend(ids)
        fd.vtok[0] = seq[-1]
        fd.pos.fill_(L + len(feed))
        st.pf["first_emitted"] = True

    def rollback(self, st: State, n: int) -> None:
        """Drop the last ``n`` tokens (n <= 15: the GDN state/conv rings are 16 deep)."""
        if n <= 0:
            return
        if n >= min(self.fd.R, 16):
            raise ValueError("rollback deeper than the 15-position state ring; re-prefill")
        if n >= len(st.tokens):
            raise ValueError("cannot roll back past the prompt")
        st.tokens = st.tokens[:-n]
        self._resync(st)


__all__ = ["FastSession", "State"]
