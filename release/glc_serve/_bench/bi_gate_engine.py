#!/usr/bin/env python3
"""G3a-BI end-to-end gate: a request's tokens AND the sha256 of every emitted logits row
are bitwise identical however it is batched.

Probe requests P (chat + ~1k-token code-context prompts) are run under several schedules:
  solo     each probe alone, prompt fed in chunks of 512
  batchA   probes + fillers, all admitted at once, prefill chunk 128, row budget 256
  batchB   probes join at staggered steps among fillers that join and LEAVE (short
           max_new), prefill chunk 97, row budget 160
  batchC   full slot count (--slots) of concurrent sequences, chunk 256
Verdict PASS iff for every probe and schedule the token list and the digest list equal solo's.
Also reported: generated text of each probe (sanity), step timings (not gated).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle")
    ap.add_argument("--parent")
    ap.add_argument("--gguf")
    ap.add_argument("--tune", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n-probes", type=int, default=6)
    ap.add_argument("--new", type=int, default=48)
    ap.add_argument("--slots", type=int, default=48)
    ap.add_argument("--pages", type=int, default=800)
    ap.add_argument("--max-ctx", type=int, default=16384)
    ap.add_argument("--max-rows", type=int, default=512)
    ap.add_argument("--spec-k", type=int, default=0, help="also gate batched E-SPEC at this k (AR solo = reference)")
    ap.add_argument("--kq-table")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-capture", action="store_true")
    a = ap.parse_args()
    from glc_serve import bidec

    t0 = time.time()
    if a.kq_table:
        from glc_serve import miv_kq
        miv_kq.load_table(a.kq_table)
    L = bidec.load_model(bundle=a.bundle, parent=a.parent, gguf=a.gguf, tune=a.tune, spec=a.spec_k > 0)
    tok = L["tok"]
    bd = bidec.BatchDecoder(L["fd"], max_slots=a.slots, max_rows=a.max_rows, pages_total=a.pages, max_ctx=a.max_ctx,
                            R=a.spec_k + 1)
    cap = {} if a.no_capture else bd.capture()
    stops = bidec.stop_ids_for(tok, a.bundle or a.parent)
    items = [json.loads(l) for l in Path(a.prompts).read_text().splitlines() if l.strip()]
    chat = [x for x in items if x["class"] == "chat"]
    ctx = [x for x in items if x["class"] == "ctx1k"]

    def ids(x):
        return tok.apply_chat_template(x["messages"], add_generation_prompt=True, tokenize=True,
                                       enable_thinking=False, return_dict=False)

    probes = [ids(x) for x in (chat[:a.n_probes // 2] + ctx[:a.n_probes - a.n_probes // 2])]
    fillers = [ids(x) for x in chat[a.n_probes // 2:] + ctx[a.n_probes - a.n_probes // 2:]]
    rng = random.Random(7)

    def run(schedule, seqs_spec, chunk, rows, join_at=None, spec_k=0):
        B = bidec.Batcher(bd, max_rows_step=rows, prefill_chunk=chunk, spec_k=spec_k)
        seqs = []
        for i, (p, n, is_probe) in enumerate(seqs_spec):
            seqs.append(bidec.Seq(rid=i, prompt=list(p), max_new=n, stop_ids=(), digest=is_probe))
        pending = list(range(len(seqs)))
        step = 0
        t = time.time()
        while pending or not B.idle():
            for i in list(pending):
                if join_at is None or join_at[i] <= step:
                    B.add(seqs[i])
                    pending.remove(i)
            B.step()
            step += 1
        torch.cuda.synchronize()
        return seqs, {"steps": step, "seconds": round(time.time() - t, 2), "rows": B.rows_total}

    rep = {"load": L["rec"], "capture_s": cap, "schedules": {}, "probe_text": []}
    solo = []
    tsolo = time.time()
    for p in probes:
        s, _ = run("solo", [(p, a.new, True)], a.max_rows, a.max_rows)
        solo.append(s[0])
    rep["schedules"]["solo"] = {"seconds": round(time.time() - tsolo, 2)}
    for s in solo:
        rep["probe_text"].append(tok.decode(s.out))
    results = {}
    # batchA: all at once
    specA = [(p, a.new, True) for p in probes] + [(f, a.new, False) for f in fillers[:max(0, a.slots - len(probes))]]
    rng.shuffle(specA)
    seqsA, stA = run("batchA", specA, 128, min(256, a.max_rows))
    results["batchA"] = (seqsA, stA)
    # batchB: staggered joins, fillers leave early
    specB, join = [], []
    for i, p in enumerate(probes):
        specB.append((p, a.new, True)); join.append(3 * i + 1)
    for j, f in enumerate(fillers[:24]):
        specB.append((f, 5 + (j % 7) * 6, False)); join.append(j % 11)
    seqsB, stB = run("batchB", specB, 97, 160, join_at=join)
    results["batchB"] = (seqsB, stB)
    # batchC: the full slot count, fillers cycled
    specC = [(p, a.new, True) for p in probes]
    k = 0
    while len(specC) < a.slots:
        specC.append((fillers[k % len(fillers)], a.new, False)); k += 1
    rng.shuffle(specC)
    seqsC, stC = run("batchC", specC, 256, a.max_rows)
    results["batchC"] = (seqsC, stC)
    if a.spec_k:
        sp_solo = []
        for p in probes:
            sq, _ = run("spec_solo", [(p, a.new, True)], a.max_rows, a.max_rows, spec_k=a.spec_k)
            sp_solo.append(sq[0])
        results["spec_solo"] = (sp_solo, {"steps": None})
        specS = [(p, a.new, True) for p in probes] + [(f, a.new, False) for f in fillers[:max(0, min(a.slots, 16) - len(probes))]]
        rng.shuffle(specS)
        seqsS, stS = run("spec_batch", specS, 128, a.max_rows, spec_k=a.spec_k)
        results["spec_batch"] = (seqsS, stS)
        allp = [s for s in sp_solo + seqsS if s.prop]
        rep["spec_acceptance"] = {"k": a.spec_k, "proposed": sum(s.prop for s in allp), "accepted": sum(s.acc for s in allp)}
    fails = 0
    for name, (seqs, st) in results.items():
        probe_seqs = [s for s in seqs if s.digest]
        # map back to probe index by prompt identity
        by_prompt = {tuple(s.prompt): s for s in probe_seqs}
        tok_mis = dig_mis = 0
        for ps in solo:
            b = by_prompt[tuple(ps.prompt)]
            tok_mis += int(b.out != ps.out)
            dig_mis += sum(1 for x, y in zip(b.digests, ps.digests) if x != y) + abs(len(b.digests) - len(ps.digests))
        fails += tok_mis + dig_mis
        rep["schedules"][name] = {**{k: v for k, v in st.items() if v is not None}, "sequences": len(seqs), "probe_token_mismatch": tok_mis,
                                  "probe_digest_rows_mismatch": dig_mis,
                                  "rows_compared": sum(len(s.digests) for s in solo)}
        print(name, json.dumps(rep["schedules"][name]), flush=True)
    rep["verdict"] = "PASS" if fails == 0 else "FAIL"
    rep["fails"] = fails
    rep["wall_s"] = round(time.time() - t0, 1)
    rep["nvml_peak_note"] = "see nvml csv from the driver script"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    print("VERDICT", rep["verdict"], "fails", fails)
    for t in rep["probe_text"][:3]:
        print("---", t[:300].replace("\n", " "))
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
