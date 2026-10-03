#!/usr/bin/env python3
"""Closed-loop concurrency benchmark for any OpenAI-compatible chat server (stdlib only).

N workers each issue streaming /v1/chat/completions requests back to back for ``--duration``
seconds (no new request starts after that; in-flight requests finish).  Every streamed chunk
is timestamped.  Reported per run:

  aggregate_tok_s        completion tokens (server usage) / (last finish - first send)
  steady_tok_s           tokens whose chunk arrived inside [t0 + warm, t_stop] / window,
                         where tokens per chunk are apportioned from usage (a chunk may carry
                         several tokens: speculative bursts or merged SSE deltas)
  ttft p50/p95           send -> first content chunk (includes queueing and prefill)
  per-stream decode      (completion_tokens - 1) / (t_last_chunk - t_first_chunk) per request
  NVML                   device memory.used / utilization sampled every 0.5 s during the run

Prompts come from ``--prompts`` (JSONL: {"messages": [...], "max_tokens": n, "class": str}).
Nothing is written outside ``--out``.
"""
from __future__ import annotations

import argparse
import http.client
import json
import random
import statistics
import subprocess
import threading
import time
from pathlib import Path


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


class NvmlSampler:
    def __init__(self, path: Path, period_ms: int = 500):
        self.path = path
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu", "--format=csv,noheader,nounits",
             f"-lms", str(period_ms)], stdout=open(path, "w"), stderr=subprocess.DEVNULL)

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        mem, util = [], []
        for line in self.path.read_text().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 3:
                try:
                    mem.append(int(p[1]))
                    util.append(int(p[2]))
                except ValueError:
                    pass
        # The memory protocol compares a STEADY median against llama.cpp, not a peak: a peak is
        # one allocator spike and a median is what the card is actually holding.  Both are
        # reported, plus the last-half median as the steady-state figure.
        half = mem[len(mem) // 2:] or mem
        return {"samples": len(mem), "mem_used_mib_max": max(mem) if mem else None,
                "mem_used_mib_min": min(mem) if mem else None,
                "mem_used_mib_p50": round(statistics.median(mem)) if mem else None,
                "mem_used_mib_p50_steady": round(statistics.median(half)) if half else None,
                "util_mean": round(statistics.mean(util), 1) if util else None}


def one_request(host, port, key, model, item, extra, timeout):
    body = {"model": model, "messages": item["messages"], "max_tokens": int(item["max_tokens"]),
            "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}}
    body.update(extra)
    data = json.dumps(body).encode()
    rec = {"class": item.get("class"), "max_tokens": body["max_tokens"], "chunks": []}
    t_send = time.time()
    rec["t_send"] = t_send
    try:
        c = http.client.HTTPConnection(host, port, timeout=timeout)
        c.request("POST", "/v1/chat/completions", body=data,
                  headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        r = c.getresponse()
        if r.status != 200:
            rec["error"] = f"HTTP {r.status}: {r.read()[:300]!r}"
            return rec
        usage = None
        timings = None
        buf = b""
        while True:
            line = r.fp.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                o = json.loads(payload)
            except ValueError:
                continue
            now = time.time()
            if o.get("usage"):
                usage = o["usage"]
            if o.get("timings"):
                timings = o["timings"]
            for ch in o.get("choices") or []:
                d = ch.get("delta") or {}
                txt = d.get("content") or d.get("reasoning_content") or ch.get("text") or ""
                if txt:
                    rec["chunks"].append(now)
            if o.get("error"):
                rec["error"] = str(o["error"])[:300]
        c.close()
        rec["usage"] = usage
        if timings:
            rec["server_timings"] = timings
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["t_end"] = time.time()
    return rec


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    key = Path(args.api_key_file).read_text().split()[0].strip() if args.api_key_file else "none"
    items = [json.loads(l) for l in Path(args.prompts).read_text().splitlines() if l.strip()]
    if args.classes:
        keep = set(args.classes.split(","))
        items = [x for x in items if x.get("class") in keep]
    if args.max_tokens:
        for x in items:
            x["max_tokens"] = args.max_tokens
    extra = json.loads(args.extra) if args.extra else {}
    rng = random.Random(args.seed)
    order = list(range(len(items)))
    rng.shuffle(order)
    ctr = [0]
    lock = threading.Lock()
    recs = []
    t0 = time.time()
    t_stop = t0 + args.duration

    def worker(wi):
        n_done = 0
        while time.time() < t_stop and (args.max_requests_per_worker <= 0 or n_done < args.max_requests_per_worker):
            with lock:
                it = items[order[ctr[0] % len(order)]]
                ctr[0] += 1
            r = one_request(args.host, args.port, key, args.model, it, extra, args.timeout)
            r["worker"] = wi
            with lock:
                recs.append(r)
            n_done += 1

    nv = NvmlSampler(out / f"nvml_c{args.concurrency}.csv")
    ths = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(args.concurrency)]
    for i, t in enumerate(ths):
        t.start()
        if args.stagger > 0:
            time.sleep(args.stagger)
    for t in ths:
        t.join()
    nvs = nv.stop()
    ok = [r for r in recs if not r.get("error") and r.get("usage") and r["chunks"]]
    errs = [r for r in recs if r.get("error")]
    t_first = min(r["t_send"] for r in recs) if recs else t0
    t_last = max(r["t_end"] for r in recs) if recs else t0
    tot = sum(int(r["usage"]["completion_tokens"]) for r in ok)
    # steady window: tokens apportioned evenly over each request's chunks
    w0, w1 = t0 + args.warm, t_stop
    in_win = 0.0
    for r in ok:
        n = int(r["usage"]["completion_tokens"])
        per = n / len(r["chunks"])
        in_win += per * sum(1 for c in r["chunks"] if w0 <= c <= w1)
    ttft = [r["chunks"][0] - r["t_send"] for r in ok]
    dec = []
    for r in ok:
        n = int(r["usage"]["completion_tokens"])
        span = r["chunks"][-1] - r["chunks"][0]
        if n > 8 and span > 0:
            dec.append((n - 1) / span)
    ptoks = [int(r["usage"]["prompt_tokens"]) for r in ok]
    ctoks = [int(r["usage"]["completion_tokens"]) for r in ok]
    summ = {
        "label": args.label, "concurrency": args.concurrency, "duration_s": args.duration,
        "requests_ok": len(ok), "requests_err": len(errs), "errors_sample": [e["error"] for e in errs[:3]],
        "completion_tokens": tot, "wall_s": round(t_last - t_first, 2),
        "aggregate_tok_s": round(tot / max(t_last - t_first, 1e-9), 2),
        "steady_window_s": round(w1 - w0, 1),
        "steady_tok_s": round(in_win / max(w1 - w0, 1e-9), 2) if w1 > w0 else None,
        "ttft_s": {"p50": pct(ttft, 0.5), "p95": pct(ttft, 0.95), "p99": pct(ttft, 0.99),
                   "max": max(ttft) if ttft else None},
        "per_stream_decode_tok_s": {"p5": pct(dec, 0.05), "p50": pct(dec, 0.5), "p95": pct(dec, 0.95),
                                    "mean": round(statistics.mean(dec), 2) if dec else None},
        "prompt_tokens": {"mean": round(statistics.mean(ptoks), 1) if ptoks else None,
                          "min": min(ptoks) if ptoks else None, "max": max(ptoks) if ptoks else None},
        "completion_tokens_per_req": {"mean": round(statistics.mean(ctoks), 1) if ctoks else None,
                                      "min": min(ctoks) if ctoks else None, "max": max(ctoks) if ctoks else None},
        # Method parity with the arm being compared against: cost per million output tokens is
        # ($/h) / (steady tok/s) * 1e6 / 3600, and only comparable when both arms ran the same
        # prompt set, the same max_new and greedy sampling on the same card.  --card-usd-h and
        # --method-note record that, so a table cannot be assembled from mismatched arms.
        "cost_usd_per_m_output": (round(args.card_usd_h / (in_win / max(w1 - w0, 1e-9))
                                        * 1e6 / 3600, 2)
                                  if args.card_usd_h and w1 > w0 and in_win > 0 else None),
        "card_usd_h": args.card_usd_h or None, "method_note": args.method_note or None,
        "nvml": nvs, "extra": extra, "classes": args.classes, "t_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
    }
    st = [r.get("server_timings") for r in ok if r.get("server_timings")]
    if st:
        dn = sum(int(x.get("draft_n", 0) or 0) for x in st)
        da = sum(int(x.get("draft_n_accepted", 0) or 0) for x in st)
        if dn:
            summ["draft"] = {"proposed": dn, "accepted": da, "acceptance": round(da / dn, 4)}
    (out / f"summary_c{args.concurrency}.json").write_text(json.dumps(summ, indent=1))
    with open(out / f"requests_c{args.concurrency}.jsonl", "w") as f:
        for r in recs:
            rr = dict(r)
            ch = rr.pop("chunks", [])
            rr["n_chunks"] = len(ch)
            rr["t_first_chunk"] = ch[0] if ch else None
            rr["t_last_chunk"] = ch[-1] if ch else None
            f.write(json.dumps(rr) + "\n")
    print(json.dumps({k: summ[k] for k in ("label", "concurrency", "requests_ok", "requests_err", "aggregate_tok_s",
                                           "steady_tok_s", "ttft_s", "per_stream_decode_tok_s",
                                           "cost_usd_per_m_output", "nvml")}), flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="m")
    ap.add_argument("--api-key-file")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--classes", default="")
    ap.add_argument("--max-tokens", type=int, default=0, help="override every item's max_tokens")
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--warm", type=float, default=15.0)
    ap.add_argument("--max-requests-per-worker", type=int, default=0)
    ap.add_argument("--stagger", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--extra", default="", help="JSON merged into every request body")
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--label", default="")
    ap.add_argument("--card-usd-h", type=float, default=0.0,
                    help="card price in USD/hour; records cost_usd_per_m_output in the summary")
    ap.add_argument("--method-note", default="",
                    help="free text recording the compared arm's exact configuration, e.g. "
                         "'llama.cpp b4xxx -np 16 -cb -c 8192 -ub 512, greedy, same prompts'")
    ap.add_argument("--out", required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
