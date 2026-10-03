#!/usr/bin/env python3
"""G3a-HTTP: the gate that runs against a live multi-user server.

Claim under test: a request's output does not depend on who else is being served.  For each
probe prompt the gate records the reference stream with the server to itself (concurrency 1),
then replays the same probe inside concurrent batches of N = 1, 2, 4, 8, 16 alongside filler
traffic that joins and finishes at different times, and requires the streamed text and the
token count to be identical to the reference every time.  It also checks stream == non-stream
for the same probe, and that a cancelled stream (client hangs up) frees capacity.

This is an end-to-end gate over HTTP: it needs only a running server, no access to the engine.
The arithmetic-level receipts are separate and stronger:
  bi_gate_kernel.py      bitwise batch invariance of BI-GEMM            (GPU)
  bi_gate_engine.py      bitwise equality of every emitted logits row   (GPU)
  bi_gate_batchexact.py  scheduler exactness, with leak controls        (CPU)

    python scripts/batchserve/bi_gate_stream.py --port 8290 --api-key-file KEY \
        --prompts prompts.jsonl --concurrency 1,2,4,8,16 --out receipt_gate_http.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def _post(host, port, key, body, stream, timeout=1800.0):
    req = urllib.request.Request(f"http://{host}:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    txt, n, finish = "", 0, ""
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if not stream:
            o = json.loads(r.read())
            c = o["choices"][0]
            return c["message"]["content"], o.get("usage", {}).get("completion_tokens", 0), c.get("finish_reason", "")
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            o = json.loads(payload)
            d = o["choices"][0]
            txt += d.get("delta", {}).get("content") or ""
            if d.get("finish_reason"):
                finish = d["finish_reason"]
            n = o.get("usage", {}).get("completion_tokens", n)
    return txt, n, finish


def _body(model, prompt, new, extra):
    b = {"model": model, "messages": [{"role": "user", "content": prompt}],
         "max_tokens": new, "temperature": 0, "stream": True}
    b.update(extra)
    return b


def _concurrent(host, port, key, model, items, new, extra, stagger):
    """Fire every item at once (optionally staggered) and collect results by index."""
    out = {}
    errs = {}

    def work(i, prompt, mx):
        try:
            time.sleep(stagger * i)
            out[i] = _post(host, port, key, _body(model, prompt, mx, extra), True)
        except Exception as e:  # noqa: BLE001
            errs[i] = f"{type(e).__name__}: {e}"

    ts = [threading.Thread(target=work, args=(i, p, m)) for i, (p, m) in enumerate(items)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out, errs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--api-key-file", required=True)
    ap.add_argument("--model", default="m")
    ap.add_argument("--prompts", required=True, help="JSONL with a 'prompt' field per line")
    ap.add_argument("--n-probes", type=int, default=4)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--concurrency", default="1,2,4,8,16")
    ap.add_argument("--stagger", type=float, default=0.25, help="seconds between arrivals in a batch")
    ap.add_argument("--extra", default="", help="JSON merged into every request body")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    key = Path(a.api_key_file).read_text().strip().splitlines()[0]
    extra = json.loads(a.extra) if a.extra else {}
    rows = [json.loads(l) for l in Path(a.prompts).read_text().splitlines() if l.strip()]
    probes = [r["prompt"] for r in rows[: a.n_probes]]
    fillers = [r["prompt"] for r in rows[a.n_probes:]] or probes
    if not probes:
        raise SystemExit("no prompts")

    health = json.loads(urllib.request.urlopen(f"http://{a.host}:{a.port}/health", timeout=30).read())
    t0 = time.time()

    # reference: the server talking only to this probe
    ref = {}
    for i, p in enumerate(probes):
        ref[i] = _post(a.host, a.port, key, _body(a.model, p, a.new, extra), True)

    checks, fails = [], 0
    for N in [int(x) for x in a.concurrency.split(",") if x.strip()]:
        items = [(p, a.new) for p in probes]
        k = 0
        while len(items) < N:                                 # fillers of mixed length
            items.append((fillers[k % len(fillers)], max(4, a.new // 2 + (k % 7) * 3)))
            k += 1
        got, errs = _concurrent(a.host, a.port, key, a.model, items[:max(N, len(probes))],
                                a.new, extra, a.stagger)
        for i in range(len(probes)):
            rtxt, rn, rf = ref[i]
            if i in errs:
                ok, detail = False, errs[i]
                gt, gn, gf = "", 0, ""
            else:
                gt, gn, gf = got[i]
                ok = gt == rtxt and gn == rn and gf == rf
                detail = "" if ok else ("text differs" if gt != rtxt else f"tokens {gn} vs {rn} / finish {gf} vs {rf}")
            fails += 0 if ok else 1
            checks.append({"concurrency": N, "probe": i, "ok": ok, "detail": detail,
                           "ref_tokens": rn, "got_tokens": gn,
                           "ref_sha256": hashlib.sha256(rtxt.encode()).hexdigest()[:16],
                           "got_sha256": hashlib.sha256(gt.encode()).hexdigest()[:16]})

    # stream == non-stream
    for i, p in enumerate(probes):
        b = _body(a.model, p, a.new, extra)
        b["stream"] = False
        gt, gn, gf = _post(a.host, a.port, key, b, False)
        ok = gt == ref[i][0]
        fails += 0 if ok else 1
        checks.append({"concurrency": "nonstream", "probe": i, "ok": ok,
                       "detail": "" if ok else "non-stream text differs from stream",
                       "ref_tokens": ref[i][1], "got_tokens": gn,
                       "ref_sha256": hashlib.sha256(ref[i][0].encode()).hexdigest()[:16],
                       "got_sha256": hashlib.sha256(gt.encode()).hexdigest()[:16]})

    # cancellation: abort a long stream, then prove the server still answers exactly
    cancel_ok = None
    try:
        b = _body(a.model, probes[0], max(a.new * 8, 256), extra)
        req = urllib.request.Request(f"http://{a.host}:{a.port}/v1/chat/completions",
                                     data=json.dumps(b).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {key}"})
        r = urllib.request.urlopen(req, timeout=300)
        r.readline()
        r.close()                                             # hang up mid-stream
        time.sleep(3.0)
        h2 = json.loads(urllib.request.urlopen(f"http://{a.host}:{a.port}/health", timeout=30).read())
        gt, gn, gf = _post(a.host, a.port, key, _body(a.model, probes[0], a.new, extra), True)
        cancel_ok = bool(gt == ref[0][0] and h2.get("status") == "ok")
        checks.append({"concurrency": "after_cancel", "probe": 0, "ok": cancel_ok,
                       "detail": "" if cancel_ok else f"health={h2.get('status')} / output changed",
                       "ref_tokens": ref[0][1], "got_tokens": gn,
                       "ref_sha256": hashlib.sha256(ref[0][0].encode()).hexdigest()[:16],
                       "got_sha256": hashlib.sha256(gt.encode()).hexdigest()[:16],
                       "health_after_cancel": h2})
        fails += 0 if cancel_ok else 1
    except Exception as e:  # noqa: BLE001
        cancel_ok = False
        fails += 1
        checks.append({"concurrency": "after_cancel", "ok": False, "detail": f"{type(e).__name__}: {e}"})

    toks = sum(c.get("ref_tokens") or 0 for c in checks)
    rec = {"gate": "G3a-HTTP concurrent-stream exactness vs the single-user stream",
           "sample_size": {
               "comparisons": len(checks),
               "tokens_compared": toks,
               "logits_rows_compared": 0,
               "output_elements_compared": 0,
               "level": "token stream and decoded text, NOT logits elements",
               "sufficient_for_a_bitwise_claim": False,
               "why": "this gate observes what a client observes, so it compares tokens, not "
                      "output elements. On a bf16-output criterion an exact and a non-exact "
                      "transform are indistinguishable below ~1e5 compared elements, so a "
                      "bitwise claim must come from bi_gate_kernel.py / bi_gate_engine.py (GPU, "
                      "per-logits-row digests) or bi_gate_batchexact.py (CPU); this gate shows "
                      "that whatever the engine computes does not change with batch composition"},
           "verdict": "PASS" if fails == 0 else "FAIL", "failures": fails,
           "checks": len(checks), "concurrency": a.concurrency, "new": a.new,
           "n_probes": len(probes), "wall_s": round(time.time() - t0, 1),
           "server_health_before": health, "detail": checks,
           "note": "text-level identity of the streamed completion; the bitwise logits-row "
                   "receipts are bi_gate_kernel.py (GPU) and bi_gate_engine.py (GPU)"}
    rec["record_digest"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: rec[k] for k in ("gate", "verdict", "failures", "checks", "wall_s")}
                     | {"out": str(out)}, indent=2))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
