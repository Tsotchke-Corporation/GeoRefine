"""In-process benchmark of one serving configuration, and the receipt joiner.

One ARM = one process = one configuration (so every arm starts from an empty
allocator).  An arm is described with exactly the server's own flags::

    python -m glc_serve.bench arm --name tbe-96 --out ~/bench -- \\
        --bundle ~/bundles/qwen38-27b --backend tbe
    python -m glc_serve.bench arm --name dense-96 --out ~/bench -- \\
        --dense ~/models/Qwen3.8-27B
    python -m glc_serve.bench arm --name tbe-cap40 --card-mib 40960 --out ~/bench -- \\
        --bundle ~/bundles/qwen38-27b --backend tbe --embed-on-host --memory-cap-gib 39.5

and ``python -m glc_serve.bench aggregate --dir ~/bench --reference dense-96``
joins them into one receipt with every ratio against the parent and a plain
MEETS / FAILS per row (raw values, no rounding in anyone's favour).

Measured per arm (all on the card the arm ran on; ``--memory-cap-gib`` arms
are EMULATIONS of a smaller card -- they prove the FIT, their tok/s is the big
card's):

  * cold start: process start -> bundle open (+ shard hash) -> load -> gates
    -> first generated token;
  * resident VRAM after load (allocator + NVML whole-process), by component
    from the manifest, plus host-resident bytes;
  * decode tok/s at batch 1/8/16/32 (fixed-length greedy, prefill excluded by
    differencing a 1-token and an N-token run), TTFT per batch;
  * prefill tok/s at 512/2048/8192 tokens;
  * image+text TTFT (vision encode + prefill) and decode tok/s;
  * MTP self-speculative decode: acceptance, tok/s, and token identity against
    plain greedy on the same prompts;
  * cache bytes per token (attention KV) and per sequence (Gated-DeltaNet
    state), MEASURED from a live cache; and a long-context probe (chunked
    prefill of N tokens + one decode step, doubling N until it fails or hits
    262,144) -- the largest N that fit is the measured max context.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

T_PROCESS = time.time()

BENCH_ARM_SCHEMA = "georefine.tbe_serve_bench_arm.v1"
BENCH_RECEIPT_SCHEMA = "georefine.tbe_serve_bench_receipt.v1"

NATURAL_PROMPTS = [
    "Explain, step by step, why the sky appears blue during the day.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Summarise the main causes of the First World War in five sentences.",
    "A train travels 180 km in 2.5 hours. What is its average speed? Show your work.",
]


def _sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _empty_cache():
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _reset_peak(dev):
    import torch

    if torch.cuda.is_available() and torch.device(dev).type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)


def _peak(dev) -> Dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.device(dev).type != "cuda":
        return {}
    from .memcap import process_device_mib

    return {"max_allocated": int(torch.cuda.max_memory_allocated(dev)),
            "max_reserved": int(torch.cuda.max_memory_reserved(dev)),
            "nvml_process_mib": process_device_mib()}


def _physical_card_fit_status(peak: Dict[str, Any], card_mib: Optional[int]):
    if card_mib is None:
        return "ok", None
    observed = peak.get("nvml_process_mib")
    if observed is None:
        return "uncertified", "NVML process memory missing; emulated-card fit is uncertified"
    if int(observed) > int(card_mib):
        return "over_card", f"NVML process memory {observed} MiB exceeds emulated card {card_mib} MiB"
    return "ok", None


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"


def _positive_seconds(seconds: float, label: str) -> float:
    seconds = float(seconds)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{label} duration must be finite and positive; got {seconds!r}")
    return seconds


def _decode_interval(first_seconds: float, total_seconds: float) -> float:
    first = _positive_seconds(first_seconds, "first-token")
    total = _positive_seconds(total_seconds, "total")
    interval = total - first
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError(f"decode interval must be positive; first={first}, total={total}")
    return interval


def cache_bytes(cache) -> Dict[str, int]:
    """Bytes held by a live transformers cache, split attention / recurrent."""
    import torch

    att = rec = 0
    for layer in getattr(cache, "layers", []) or []:
        for name in ("keys", "values"):
            t = getattr(layer, name, None)
            if isinstance(t, torch.Tensor):
                att += t.numel() * t.element_size()
        for name in ("conv_states", "recurrent_states"):
            t = getattr(layer, name, None)
            if isinstance(t, torch.Tensor):
                rec += t.numel() * t.element_size()
    return {"attention_bytes": int(att), "recurrent_bytes": int(rec)}


class Arm:
    def __init__(self, name: str, server_argv: List[str], *, card_mib: Optional[int],
                 log=print):
        from .server import _parse, build_state

        self.name = name
        self.log = log
        self.args = _parse(server_argv + ["--port", "0"])
        t0 = time.time()
        self.state = build_state(self.args, log=log)
        self.t_ready = time.time()
        self.engine = self.state.engine
        self.model = self.engine.model
        self.dev = self.engine.device
        self.card_mib = card_mib
        self._nvml_observations: List[Dict[str, Any]] = []
        self.result: Dict[str, Any] = {
            "schema": BENCH_ARM_SCHEMA, "name": name, "server_argv": server_argv,
            "startup_s": round(self.t_ready - t0, 3),
            "process_start_to_ready_s": round(self.t_ready - T_PROCESS, 3),
            "receipts": self.state.receipts,
            "memory_observations": self._nvml_observations,
        }

    def _record_peak(self, phase: str, measurement: Optional[Dict[str, Any]] = None
                     ) -> Dict[str, Any]:
        peak = dict(_peak(self.dev) if measurement is None else measurement)
        if not hasattr(self, "_nvml_observations"):
            self._nvml_observations = []
        self.result.setdefault("memory_observations", self._nvml_observations)
        nvml = peak.get("nvml_process_mib")
        self._nvml_observations.append({"phase": phase, "nvml_process_mib": nvml})
        seen = [int(item["nvml_process_mib"]) for item in self._nvml_observations
                if item["nvml_process_mib"] is not None]
        peak["nvml_process_mib_highwater"] = max(seen) if seen else None
        peak["nvml_phase_sample_count"] = len(self._nvml_observations)
        return peak

    # -- helpers -------------------------------------------------------------
    def _vocab_ids(self, shape, seed):
        import torch

        g = torch.Generator().manual_seed(seed)
        hi = min(int(self.engine.tok.vocab_size), 150_000)
        lo = min(1000, hi // 4)
        return torch.randint(lo, hi, shape, generator=g).to(self.dev)

    def _gen(self, ids, n, **extra):
        import torch

        with torch.no_grad():
            return self.model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=n, min_new_tokens=n, do_sample=False,
                pad_token_id=self.engine.pad_id, eos_token_id=self.engine.eos_ids, **extra)

    def _timed(self, fn):
        _sync()
        t = time.perf_counter()
        out = fn()
        _sync()
        return time.perf_counter() - t, out

    def _warm(self, fn):
        """Run one untimed shape warmup and release its output before measurement."""
        import torch

        with torch.no_grad():
            output = fn()
        del output
        _sync()

    # -- phases ------------------------------------------------------------------
    def cold_first_token(self):
        from .engine import Request, SamplingParams

        req = Request(kind="chat", params=SamplingParams(max_tokens=1, temperature=0),
                      messages=[{"role": "user", "content": "Hello"}],
                      chat_template_kwargs={"enable_thinking": False})
        self.engine.submit(req)
        while True:
            ev = req.out.get()
            if ev[0] in ("done", "error"):
                break
        now = time.time()
        peak = self._record_peak("cold_start")
        self.result["cold_start"] = {
            "process_start_to_first_token_s": round(now - T_PROCESS, 3),
            "ready_to_first_token_s": round(now - self.t_ready, 3),
            "load_timing_s": (self.state.receipts.get("load") or {}).get("timing_s"),
            "first_request_status": ev[0],
            "peak": peak,
        }

    def memory_after_load(self):
        import torch

        from .memcap import measure

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        m = self._record_peak("memory_after_load", measure(self.dev))
        load = self.state.receipts.get("load") or {}
        self.result["memory_after_load"] = {
            **m,
            "host_embed_bytes": load.get("host_embed_bytes"),
            "streamer": load.get("streamer"),
            "pool_capacity_bytes": load.get("pool_capacity_bytes"),
            "manifest_accounting": load.get("manifest_accounting"),
            "offload_plan": load.get("offload_plan"),
        }
        self._steady_alloc = m["allocated"]

    def decode_throughput(self, batches, prompt_tokens, n_tokens):
        rows = []
        for b in batches:
            ids = self._vocab_ids((b, prompt_tokens), seed=100 + b)
            try:
                if n_tokens < 2:
                    raise ValueError("decode throughput requires at least two generated tokens")
                _reset_peak(self.dev)
                # Two tokens compile the prompt prefill and first cached decode
                # shapes without adding a second full-length run to every arm.
                self._warm(lambda: self._gen(ids, 2))
                t1, first_output = self._timed(lambda: self._gen(ids, 1))
                del first_output
                tn, total_output = self._timed(lambda: self._gen(ids, n_tokens))
                del total_output
                interval = _decode_interval(t1, tn)
                rows.append({
                    "batch": b, "prompt_tokens": prompt_tokens, "new_tokens": n_tokens,
                    "ttft_s": round(t1, 5), "total_s": round(tn, 5),
                    "decode_tok_s": round(b * (n_tokens - 1) / interval, 3),
                    "per_stream_tok_s": round((n_tokens - 1) / interval, 3),
                    "peak": self._record_peak(f"decode_batch_{b}"), "status": "ok",
                })
            except Exception as exc:
                rows.append({"batch": b, "status": "oom" if _is_oom(exc) else "error",
                             "peak": self._record_peak(f"decode_batch_{b}_error"),
                             "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
                gc.collect()
                _empty_cache()
            self.log(f"[bench:{self.name}] decode {rows[-1]}")
        self.result["decode"] = rows

    def prefill_throughput(self, lengths):
        import torch

        rows = []
        for n in lengths:
            ids = self._vocab_ids((1, n), seed=7 + n)
            try:
                _reset_peak(self.dev)
                times = []
                for _ in range(2):
                    output = None
                    try:
                        with torch.no_grad():
                            t, output = self._timed(lambda: self.model(
                                input_ids=ids, use_cache=True, logits_to_keep=1))
                        times.append(t)
                    finally:
                        del output
                rows.append({"tokens": n, "seconds": round(min(times), 5),
                             "prefill_tok_s": round(n / min(times), 2),
                             "peak": self._record_peak(f"prefill_{n}"), "status": "ok"})
            except Exception as exc:
                rows.append({"tokens": n, "status": "oom" if _is_oom(exc) else "error",
                             "peak": self._record_peak(f"prefill_{n}_error"),
                             "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
            gc.collect()
            _empty_cache()
            self.log(f"[bench:{self.name}] prefill {rows[-1]}")
        self.result["prefill"] = rows

    def image_request(self, side: int, n_tokens: int):
        import torch
        from PIL import Image, ImageDraw

        from .engine import Request, SamplingParams

        if getattr(self.engine.processor, "image_processor", None) is None:
            self.result["image"] = {"status": "skipped", "reason": "no image processor"}
            return
        img = Image.new("RGB", (side, side), (245, 245, 245))
        d = ImageDraw.Draw(img)
        for i in range(8):
            d.rectangle((40 + i * 120, side - 80 - 90 * i, 120 + i * 120, side - 40),
                        fill=(30 * i, 90, 200 - 20 * i))
        d.text((60, 60), "Quarterly revenue by region, 2026", fill=(0, 0, 0))
        req = Request(kind="chat", params=SamplingParams(max_tokens=n_tokens, temperature=0),
                      messages=[{"role": "user", "content": [
                          {"type": "image"}, {"type": "text", "text": "Describe this chart."}]}],
                      chat_template_kwargs={"enable_thinking": False})
        req.images = [img]
        try:
            if n_tokens < 2:
                raise ValueError("image decode throughput requires at least two generated tokens")
            t_prep, inputs = self._timed(lambda: self.engine.prepare([req]))
            def generate(count):
                return self.model.generate(
                    **inputs, max_new_tokens=count, min_new_tokens=count, do_sample=False,
                    pad_token_id=self.engine.pad_id, eos_token_id=self.engine.eos_ids)

            _reset_peak(self.dev)
            self._warm(lambda: generate(2))
            with torch.no_grad():
                t1, first_output = self._timed(lambda: generate(1))
                del first_output
                tn, total_output = self._timed(lambda: generate(n_tokens))
                del total_output
            interval = _decode_interval(t1, tn)
            self.result["image"] = {
                "status": "ok", "image_side_px": side,
                "prompt_tokens": int(inputs["input_ids"].shape[1]),
                "preprocess_s": round(t_prep, 5),
                "ttft_s_including_vision_encode": round(t1, 5),
                "ttft_s_including_preprocess": round(t1 + t_prep, 5),
                "decode_tok_s": round((n_tokens - 1) / interval, 3),
                "peak": self._record_peak("image"),
            }
        except Exception as exc:
            self.result["image"] = {"status": "oom" if _is_oom(exc) else "error",
                                    "peak": self._record_peak("image_error"),
                                    "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        self.log(f"[bench:{self.name}] image {self.result['image']}")

    def mtp_speculative(self, n_tokens: int):
        import torch

        from .engine import Request, SamplingParams
        from .mtp import MTPSpeculator

        loaded = self.engine.loaded
        if loaded.mtp is None:
            self.result["mtp"] = {"status": "skipped", "reason": "no MTP head loaded"}
            return
        _reset_peak(self.dev)
        rows = []
        for i, prompt in enumerate(NATURAL_PROMPTS):
            req = Request(kind="chat", params=SamplingParams(max_tokens=n_tokens, temperature=0),
                          messages=[{"role": "user", "content": prompt}],
                          chat_template_kwargs={"enable_thinking": False})
            inputs = self.engine.prepare([req])
            spec = MTPSpeculator(self.model, loaded.mtp)
            spec_inputs = {k: v for k, v in inputs.items() if k != "attention_mask"}
            self._warm(lambda: self.model.generate(
                **inputs, max_new_tokens=2, min_new_tokens=2, do_sample=False,
                pad_token_id=self.engine.pad_id, eos_token_id=self.engine.eos_ids))
            self._warm(lambda: spec.generate(
                spec_inputs, max_new_tokens=2, eos_ids=self.engine.eos_ids,
                on_token=lambda t: True))
            with torch.no_grad():
                tp, g = self._timed(lambda: self.model.generate(
                    **inputs, max_new_tokens=n_tokens, do_sample=False,
                    pad_token_id=self.engine.pad_id, eos_token_id=self.engine.eos_ids))
            plain = []
            for t in g[0, inputs["input_ids"].shape[1]:].tolist():
                if t in self.engine.eos_ids:
                    break
                plain.append(t)
            del g
            got: List[int] = []
            ts, st = self._timed(lambda: spec.generate(
                spec_inputs,
                max_new_tokens=n_tokens, eos_ids=self.engine.eos_ids,
                on_token=lambda t: (got.append(int(t)) or True)))
            tp = _positive_seconds(tp, "plain decode")
            ts = _positive_seconds(ts, "speculative decode")
            first_div = next((j for j, (a, b) in enumerate(zip(plain, got)) if a != b), None)
            rows.append({
                "prompt": i, "plain_tokens": len(plain), "spec_tokens": len(got),
                "identical": plain == got, "first_divergence": first_div,
                "plain_tok_s": round(len(plain) / tp, 3),
                "spec_tok_s": round(len(got) / ts, 3),
                "acceptance_rate": st.get("acceptance_rate"),
                "tokens_per_target_forward": st.get("tokens_per_target_forward"),
            })
            self.log(f"[bench:{self.name}] mtp {rows[-1]}")
        ok = [r for r in rows if r["plain_tokens"]]
        self.result["mtp"] = {
            "status": "ok", "rows": rows,
            "mean_acceptance": (sum(r["acceptance_rate"] or 0 for r in ok) / len(ok)) if ok else None,
            "speedup_vs_plain": (sum(r["spec_tok_s"] for r in ok) / max(1e-9, sum(
                r["plain_tok_s"] for r in ok))) if ok else None,
            "n_identical": sum(1 for r in rows if r["identical"]),
            "peak": self._record_peak("mtp"),
            "note": ("any token divergence from plain greedy decoding requires investigation; "
                     "this receipt does not attribute its cause"),
        }

    def context_probe(self, lengths, chunk: int):
        import torch

        from .mtp import new_cache

        rows = []
        cfg = self.model.config
        tcfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
        per_token = per_seq = None
        for n in lengths:
            ids = self._vocab_ids((1, n), seed=11)
            cache = None
            out = token = None
            try:
                _reset_peak(self.dev)
                cache = new_cache(tcfg)
                _sync()
                prefill_t0 = time.perf_counter()
                with torch.no_grad():
                    for s in range(0, n, chunk):
                        out = self.model(input_ids=ids[:, s:s + chunk], past_key_values=cache,
                                         use_cache=True, logits_to_keep=1)
                _sync()
                prefill_s = time.perf_counter() - prefill_t0
                prefill_s = _positive_seconds(prefill_s, "context prefill")
                decode_t0 = time.perf_counter()
                with torch.no_grad():
                    token = out.logits[:, -1:].argmax(dim=-1)
                    decode_steps = 16
                    for _ in range(decode_steps):
                        out = self.model(input_ids=token, past_key_values=cache, use_cache=True,
                                         logits_to_keep=1)
                        token = out.logits[:, -1:].argmax(dim=-1)
                _sync()
                decode_s = time.perf_counter() - decode_t0
                decode_s = _positive_seconds(decode_s, "context decode")
                cb = cache_bytes(cache)
                seq = int(cache.get_seq_length())
                per_token = cb["attention_bytes"] / max(1, seq)
                per_seq = cb["recurrent_bytes"]
                peak = self._record_peak(f"context_{n}")
                status, reason = _physical_card_fit_status(peak, getattr(self, "card_mib", None))
                row = {"tokens": n, "status": status,
                             "prefill_seconds": round(prefill_s, 5),
                             "prefill_tok_s": round(n / prefill_s, 3),
                             "decode_tokens": decode_steps,
                             "decode_seconds": round(decode_s, 5),
                             "decode_tok_s": round(decode_steps / decode_s, 3),
                             "seconds": round(prefill_s + decode_s, 3),
                             "cache": cb, "peak": peak}
                if reason:
                    row["physical_card_failure"] = reason
                rows.append(row)
            except Exception as exc:
                rows.append({"tokens": n, "status": "oom" if _is_oom(exc) else "error",
                             "peak": self._record_peak(f"context_{n}_error"),
                             "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
            del out, token, cache, ids
            gc.collect()
            _empty_cache()
            self.log(f"[bench:{self.name}] ctx {rows[-1]}")
            if rows[-1]["status"] != "ok":
                break
        fit = [r["tokens"] for r in rows if r["status"] == "ok"]
        cap = (self.state.receipts.get("memory_cap") or {}).get("cap_bytes")
        total = cap or (int(torch.cuda.get_device_properties(self.dev).total_memory)
                        if self.dev.type == "cuda" else None)
        headroom = (total - int(self._steady_alloc)) if total else None
        self.result["context"] = {
            "rows": rows, "chunk": chunk,
            "max_context_measured_tokens": max(fit) if fit else 0,
            "probe_stopped_at": rows[-1]["tokens"] if rows else None,
            "kv_bytes_per_token_measured": per_token,
            "recurrent_bytes_per_sequence_measured": per_seq,
            "memory_limit_bytes": total, "memory_limit_is_emulated_cap": bool(cap),
            "steady_allocated_after_load": int(self._steady_alloc),
            "headroom_bytes": headroom,
            "max_context_projected_tokens": (
                int((headroom - (per_seq or 0)) // per_token)
                if (per_token and headroom is not None) else None),
            "note": "KV is bf16 (the parent's own dtype); a quantized KV cache is lossy "
                    "and is never part of this measurement. max_context_measured_tokens is "
                    "the largest tested context that fit, a lower bound on capacity",
        }

    def checkpoint_partial(self, out_dir: Path):
        """Atomically preserve completed phase measurements without marking the arm complete."""
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(self.result)
        payload["receipt_status"] = "partial"
        p = out_dir / f"arm_{self.name}.partial.json"
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=1, default=str))
        os.replace(tmp, p)

    def finish(self, out_dir: Path):
        from .memcap import assert_within_cap

        cap = self.state.receipts.get("memory_cap")
        if cap:
            try:
                self.result["memory_cap_final"] = assert_within_cap(
                    self.dev, cap, card_mib=self.card_mib,
                    observed_nvml_mib=[item["nvml_process_mib"]
                                       for item in self._nvml_observations])
                self.result["memory_cap_final"]["status"] = "within_cap"
            except RuntimeError as exc:
                detail = str(exc)
                status = "UNCERTIFIED" if "uncertified" in detail or "missing" in detail else "EXCEEDED"
                self.result["memory_cap_final"] = {"status": status, "detail": detail}
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"arm_{self.name}.json"
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(self.result, indent=1, default=str))
        os.replace(tmp, p)
        return p


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------
def _decode_by_batch(arm):
    return {r["batch"]: r for r in arm.get("decode", []) if r.get("status") == "ok"}


def aggregate(arms: Dict[str, Dict[str, Any]], reference: str,
              rates: Dict[str, Any]) -> Dict[str, Any]:
    ref = arms.get(reference)
    rows = []

    def row(metric, arm_name, value, ref_value, better):
        if value is None or ref_value is None:
            verdict = "UNMEASURED"
            ratio = None
        else:
            ratio = value / ref_value if ref_value else None
            if better == "higher":
                verdict = "EXCEEDS" if value > ref_value else "MEETS" if value == ref_value else "FAILS"
            else:
                verdict = "EXCEEDS" if value < ref_value else "MEETS" if value == ref_value else "FAILS"
        rows.append({"metric": metric, "arm": arm_name, "value": value,
                     "parent_value": ref_value, "ratio_vs_parent": ratio,
                     "better": better, "verdict": verdict})

    for name, arm in arms.items():
        if name == reference:
            continue
        d, rd = _decode_by_batch(arm), _decode_by_batch(ref or {})
        for b in sorted(set(d) | set(rd)):
            row(f"decode_tok_s@B{b}", name, d.get(b, {}).get("decode_tok_s"),
                rd.get(b, {}).get("decode_tok_s"), "higher")
            row(f"ttft_s@B{b}", name, d.get(b, {}).get("ttft_s"),
                rd.get(b, {}).get("ttft_s"), "lower")
        pf = {r["tokens"]: r for r in arm.get("prefill", []) if r.get("status") == "ok"}
        rpf = {r["tokens"]: r for r in (ref or {}).get("prefill", []) if r.get("status") == "ok"}
        for n in sorted(set(pf) | set(rpf)):
            row(f"prefill_tok_s@{n}", name, pf.get(n, {}).get("prefill_tok_s"),
                rpf.get(n, {}).get("prefill_tok_s"), "higher")
        im, rim = arm.get("image") or {}, (ref or {}).get("image") or {}
        row("image_ttft_s", name, im.get("ttft_s_including_vision_encode"),
            rim.get("ttft_s_including_vision_encode"), "lower")
        m, rm = arm.get("memory_after_load") or {}, (ref or {}).get("memory_after_load") or {}
        row("resident_vram_allocated_bytes", name, m.get("allocated"), rm.get("allocated"),
            "lower")
        mtp, rmtp = arm.get("mtp") or {}, (ref or {}).get("mtp") or {}
        spec = [r["spec_tok_s"] for r in mtp.get("rows", [])] or None
        rspec = [r["spec_tok_s"] for r in rmtp.get("rows", [])] or None
        rplain = [r["plain_tok_s"] for r in rmtp.get("rows", [])] or None
        if spec and rspec:
            row("mtp_spec_tok_s@B1_vs_parent_mtp", name, sum(spec) / len(spec),
                sum(rspec) / len(rspec), "higher")
        if spec and rplain:
            row("mtp_spec_tok_s@B1_vs_parent_plain", name, sum(spec) / len(spec),
                sum(rplain) / len(rplain), "higher")
        c = arm.get("context") or {}
        row("max_context_measured_tokens", name, c.get("max_context_measured_tokens"),
            ((ref or {}).get("context") or {}).get("max_context_measured_tokens"), "higher")
    cost = {}
    for name, arm in arms.items():
        hw = arm.get("hardware_rate_usd_per_h")
        d = _decode_by_batch(arm)
        if hw:
            cost[name] = {
                f"B{b}": round(hw / (r["decode_tok_s"] * 3600) * 1e6, 4)
                for b, r in d.items() if r.get("decode_tok_s")
            }
    return {
        "schema": BENCH_RECEIPT_SCHEMA,
        "reference_arm": reference,
        "rows": rows,
        "usd_per_mtok_decode": cost,
        "rates": rates,
        "n_exceeds": sum(1 for r in rows if r["verdict"] == "EXCEEDS"),
        "n_meets": sum(1 for r in rows if r["verdict"] == "MEETS"),
        "n_fails": sum(1 for r in rows if r["verdict"] == "FAILS"),
        "arms": {k: {kk: v.get(kk) for kk in ("server_argv", "cold_start", "memory_after_load",
                                               "context", "mtp", "memory_cap_final",
                                               "receipts")} for k, v in arms.items()},
        "note": ("Capped arms are EMULATIONS; only a within_cap physical-card check certifies the fit. Their "
                 "tok/s is the big card's. Verdicts compare raw values; ratios are unrounded."),
    }


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "arm":
        if "--" not in argv:
            raise SystemExit("usage: bench arm --name N --out DIR [...] -- <server flags>")
        cut = argv.index("--")
        ap = argparse.ArgumentParser(prog="glc_serve.bench arm")
        ap.add_argument("--name", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--card-mib", type=int, default=None,
                        help="total MiB of the card a capped arm emulates (NVML check)")
        ap.add_argument("--rate-usd-per-h", type=float, default=None)
        ap.add_argument("--batches", default="1,8,16,32")
        ap.add_argument("--prompt-tokens", type=int, default=128)
        ap.add_argument("--decode-tokens", type=int, default=128)
        ap.add_argument("--prefill", default="512,2048,8192")
        ap.add_argument("--context", default="8192,32768,65536,131072,262144")
        ap.add_argument("--context-chunk", type=int, default=4096)
        ap.add_argument("--image-side", type=int, default=1024)
        ap.add_argument("--mtp-tokens", type=int, default=128)
        ap.add_argument("--skip", default="", help="comma list of phases to skip")
        a = ap.parse_args(argv[1:cut])
        arm = Arm(a.name, argv[cut + 1:], card_mib=a.card_mib)
        arm.result["hardware_rate_usd_per_h"] = a.rate_usd_per_h
        skip = set(filter(None, a.skip.split(",")))
        out_dir = Path(a.out)
        arm.cold_first_token()
        arm.checkpoint_partial(out_dir)
        arm.memory_after_load()
        arm.checkpoint_partial(out_dir)
        if "decode" not in skip:
            arm.decode_throughput([int(x) for x in a.batches.split(",")],
                                  a.prompt_tokens, a.decode_tokens)
            arm.checkpoint_partial(out_dir)
        if "prefill" not in skip:
            arm.prefill_throughput([int(x) for x in a.prefill.split(",")])
            arm.checkpoint_partial(out_dir)
        if "image" not in skip:
            arm.image_request(a.image_side, 64)
            arm.checkpoint_partial(out_dir)
        if "mtp" not in skip:
            try:
                arm.mtp_speculative(a.mtp_tokens)
            finally:
                arm.checkpoint_partial(out_dir)
        if "context" not in skip:
            try:
                arm.context_probe([int(x) for x in a.context.split(",")], a.context_chunk)
            finally:
                arm.checkpoint_partial(out_dir)
        p = arm.finish(out_dir)
        print(json.dumps({"status": "ok", "arm": a.name, "out": str(p)}))
        os._exit(0)   # the engine thread is a daemon; do not wait on CUDA teardown
    ap = argparse.ArgumentParser(prog="glc_serve.bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ag = sub.add_parser("aggregate")
    ag.add_argument("--dir", required=True)
    ag.add_argument("--reference", default="dense-96")
    ag.add_argument("--rates", default=None, help="JSON file with the cited $/h rates")
    ag.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    d = Path(a.dir)
    arms = {p.stem[len("arm_"):]: json.loads(p.read_text())
            for p in sorted(d.glob("arm_*.json")) if not p.name.endswith(".partial.json")}
    rates = json.loads(Path(a.rates).read_text()) if a.rates else {}
    rec = aggregate(arms, a.reference, rates)
    out = Path(a.out) if a.out else d / "bench_receipt.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=str))
    os.replace(tmp, out)
    print(json.dumps({"status": "ok", "out": str(out), "n_exceeds": rec["n_exceeds"],
                      "n_meets": rec["n_meets"], "n_fails": rec["n_fails"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
