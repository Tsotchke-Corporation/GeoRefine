#!/usr/bin/env python3
"""G3a-SCHED: the batch-composition exactness gate for the bidec scheduler, on the CPU.

The arithmetic's batch invariance is gated on the GPU by ``bi_gate_kernel.py``.  This gate
covers the layer above it: that for ANY batch composition and ANY arrival order, each request
is fed exactly its own history and nothing else.  It runs the real ``glc_serve.bidec.Batcher``
over ``glc_serve.bidec_ref.RefDecoder`` -- a CPU decoder whose row value is defined to depend
on the slot's tokens read back through the KV page table plus a per-slot recurrent state, and
on nothing else.  No CUDA, no model weights, seconds to run.

Verdict PASS iff for every probe request and every schedule, the emitted token list and the
per-row logits digest list are identical to the probe's solo run.

Schedules:
  solo        one request at a time, prompt chunk 512
  all_at_once every request admitted together, chunk 128, rows 256
  staggered   probes join at different steps among fillers that join AND leave early
  odd_chunk   prompt chunk 97 and row budget 160 (chunk boundaries off every power of two)
  slot_churn  max_users 2 so probes are forced to reuse slots freed by finished fillers
  cancelled   fillers are cancelled mid-flight while probes keep decoding
  tiny_rows   row budget 8: probes are split across many steps

Negative controls (``--controls``): the same battery against RefDecoder(leak=...) for each of
batch / neighbour / noreset / nopages.  Every one MUST fail, or the gate is not measuring
anything.  The receipt records both halves.

    python scripts/batchserve/bi_gate_batchexact.py --out .scratch/gate_batchexact.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "release"))

from glc_serve import bidec                                     # noqa: E402
from glc_serve.bidec_ref import RefDecoder                      # noqa: E402

# MEASUREMENT TRAP (Lane I, docs/research/GROUP_THEORY_I_EXACT_GROUP_20261003.md): on a
# bf16-output criterion a bitwise-exact transform and a non-exact one are INDISTINGUISHABLE below
# ~1e5 compared output elements -- the divergence rate is 1.2e-4..4.9e-4 per element and grows
# with sqrt(k).  A gate that compares 73,728 elements cannot support the word "bitwise", however
# many schedules it runs.  So this gate counts the elements it actually compared, and its verdict
# requires that count to clear BITWISE_CLAIM_FLOOR.
V = 2048                      # logits width of the reference decoder (rows * V = elements)
BITWISE_CLAIM_FLOOR = 1_000_000
STOP = ()


def _mk(rng, n_probe, n_fill, plen, new):
    probes = [bidec.Seq(rid=f"p{i}", prompt=[rng.randrange(1, V) for _ in range(plen[i])],
                        max_new=new, stop_ids=STOP, digest=True) for i in range(n_probe)]
    fills = [bidec.Seq(rid=f"f{i}", prompt=[rng.randrange(1, V) for _ in range(rng.randrange(5, 400))],
                       max_new=rng.randrange(2, new), stop_ids=STOP) for i in range(n_fill)]
    return probes, fills


def _clone(s):
    return bidec.Seq(rid=s.rid, prompt=list(s.prompt), max_new=s.max_new, stop_ids=s.stop_ids,
                     digest=s.digest)


def _dec(leak, max_slots, pages):
    # V is passed explicitly: the element count in the receipt is rows * V, so the decoder's
    # logits width and the reported width must be the same number or the receipt lies.
    return RefDecoder(max_slots=max_slots, max_rows=256, pages_total=pages, max_ctx=4096,
                      V=V, leak=leak)


def _drain(B, limit=200000):
    for _ in range(limit):
        if B.idle():
            return
        if B.step() == 0 and B.idle():
            return
    raise RuntimeError("scheduler did not drain")


def _solo(probes, leak, pages):
    out = {}
    for p in probes:
        bd = _dec(leak, 1, pages)
        B = bidec.Batcher(bd, max_rows_step=256, prefill_chunk=512, max_active=1)
        c = _clone(p)
        B.add(c)
        _drain(B)
        bd.check_accounting()
        out[p.rid] = (list(c.out), list(c.digests), c.finish)
    return out


def _schedule(name, probes, fills, leak, pages, rng):
    """Run one schedule; return {rid: (tokens, digests, finish)} for the probes."""
    cfg = {"all_at_once": (256, 128, 16), "staggered": (160, 97, 16), "odd_chunk": (160, 97, 16),
           "slot_churn": (256, 64, 2), "cancelled": (256, 128, 16), "tiny_rows": (8, 8, 16)}
    rows, chunk, users = cfg[name]
    bd = _dec(leak, min(users, 16), pages)
    B = bidec.Batcher(bd, max_rows_step=rows, prefill_chunk=chunk, max_active=users)
    pc = [_clone(p) for p in probes]
    fc = [_clone(f) for f in fills]
    if name == "staggered":
        pending = [(rng.randrange(0, 12), x) for x in pc] + [(rng.randrange(0, 12), x) for x in fc]
        pending.sort(key=lambda t: t[0])
        step = 0
        while pending or not B.idle():
            while pending and pending[0][0] <= step:
                B.add(pending.pop(0)[1])
            B.step()
            step += 1
            if step > 200000:
                raise RuntimeError("no drain")
    else:
        for x in (fc[: len(fc) // 2] + pc + fc[len(fc) // 2:]):
            B.add(x)
        if name == "cancelled":
            for _ in range(6):
                B.step()
            for f in fc[::2]:
                B.cancel(f.rid)
        _drain(B)
    bd.check_accounting()
    for f in fc:
        if not f.done:
            raise AssertionError(f"filler {f.rid} never finished")
    return {p.rid: (list(p.out), list(p.digests), p.finish) for p in pc}


def battery(leak, seed, n_probe, n_fill, new, pages):
    rng = random.Random(seed)
    plen = [rng.randrange(1, 600) for _ in range(n_probe)]
    probes, fills = _mk(rng, n_probe, n_fill, plen, new)
    ref = _solo(probes, leak, pages)
    rows, bad = [], 0
    for name in ("all_at_once", "staggered", "odd_chunk", "slot_churn", "cancelled", "tiny_rows"):
        got = _schedule(name, probes, fills, leak, pages, random.Random(seed ^ hash(name) & 0xFFFF))
        for rid, (tk, dg, fin) in got.items():
            rtk, rdg, rfin = ref[rid]
            tok_mis = sum(1 for a, b in zip(tk, rtk) if a != b) + abs(len(tk) - len(rtk))
            dig_mis = sum(1 for a, b in zip(dg, rdg) if a != b) + abs(len(dg) - len(rdg))
            ok = tok_mis == 0 and dig_mis == 0 and fin == rfin
            bad += 0 if ok else 1
            rows.append({"schedule": name, "rid": rid, "tokens": len(rtk), "token_mismatch": tok_mis,
                         "digest_rows": len(rdg), "digest_mismatch": dig_mis,
                         "finish": fin, "finish_solo": rfin, "ok": ok})
    rows_cmp = sum(r["digest_rows"] for r in rows)
    probe_width = int(_dec(leak, 1, 8).logits.shape[1])
    if probe_width != V:
        raise AssertionError(f"reference decoder logits width {probe_width} != reported V {V}; "
                             f"the element count would be wrong")
    return {"leak": leak, "mismatched_probe_schedules": bad, "comparisons": len(rows),
            "logits_rows_compared": rows_cmp, "rows_compared": rows_cmp,
            "logits_width": V, "output_elements_compared": rows_cmp * V,
            "tokens_compared": sum(r["tokens"] for r in rows), "detail": rows}


def ring_control(seed: int = 7):
    """The state-ring half of the gate: re-feeding a position must be idempotent.

    A rejected speculative draft and a re-run prompt chunk both re-feed a position.  No batch
    composition does, so the schedule battery above cannot see a recurrent state that is folded
    monotonically instead of indexed by position -- and that bug is silent until speculation is
    switched on, then corrupts every row after a rejection.  This runs the re-feed directly and
    requires the `noring` control to diverge.
    """
    import random as _r
    rng = _r.Random(seed)
    toks = [rng.randrange(1, V) for _ in range(24)]

    def final(leak, refeed):
        bd = RefDecoder(max_slots=1, max_rows=64, pages_total=8, max_ctx=4096, R=4, V=V, leak=leak)
        slot = bd.alloc_slot()
        bd.reset_slot_state(slot)
        bd.run([(slot, 0, toks[:10])])
        if refeed:
            bd.run([(slot, 10, [toks[11]])])              # drafted, then rejected
        bd.run([(slot, 10, [toks[10]])])
        return bd.logits[0].clone()

    import torch
    clean = torch.equal(final(None, False), final(None, True))
    caught = not torch.equal(final("noring", False), final("noring", True))
    return {"refeed_is_idempotent": bool(clean), "noring_control_detected": bool(caught),
            "verdict": "PASS" if (clean and caught) else "FAIL",
            "what": "a rejected draft must leave the sequence exactly where it was"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--n-probes", type=int, default=4)
    ap.add_argument("--n-fillers", type=int, default=10)
    ap.add_argument("--new", type=int, default=24)
    ap.add_argument("--pages", type=int, default=96)
    ap.add_argument("--no-controls", action="store_true", help="skip the leak positive controls")
    ap.add_argument("--element-floor", type=int, default=BITWISE_CLAIM_FLOOR,
                    help="compared output elements required before the verdict may support a "
                         "bitwise claim")
    ap.add_argument("--allow-small-sample", action="store_true",
                    help="let the gate PASS below the element floor; the receipt still records "
                         "sufficient_for_a_bitwise_claim: false")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.time()
    main_run = battery(None, a.seed, a.n_probes, a.n_fillers, a.new, a.pages)
    controls = []
    if not a.no_controls:
        from glc_serve.bidec_ref import LEAK_MODES
        for mode in (m for m in LEAK_MODES if m != "noring"):   # noring: see ring_control()
            try:
                controls.append(battery(mode, a.seed, a.n_probes, a.n_fillers, a.new, a.pages))
            except Exception as e:  # a leak that crashes the scheduler is also a detection
                controls.append({"leak": mode, "mismatched_probe_schedules": -1,
                                 "raised": f"{type(e).__name__}: {e}", "detail": []})
    controls_ok = all(c["mismatched_probe_schedules"] != 0 for c in controls)
    ring = ring_control(a.seed)
    elements = main_run["output_elements_compared"]
    enough = elements >= a.element_floor
    verdict = "PASS" if (main_run["mismatched_probe_schedules"] == 0 and controls_ok
                         and ring["verdict"] == "PASS"
                         and (enough or a.allow_small_sample)) else "FAIL"
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        head = "unknown"
    rec = {"gate": "G3a-SCHED batch-composition exactness (CPU reference decoder)",
           "verdict": verdict, "commit": head, "seed": a.seed, "wall_s": round(time.time() - t0, 2),
           "args": vars(a), "main": main_run,
           "controls": [{k: v for k, v in c.items() if k != "detail"} for c in controls],
           "controls_all_detected": controls_ok, "state_ring": ring,
           "sample_size": {
               "logits_rows_compared": main_run["logits_rows_compared"],
               "logits_width": V,
               "output_elements_compared": elements,
               "tokens_compared": main_run["tokens_compared"],
               "bitwise_claim_floor": a.element_floor,
               "sufficient_for_a_bitwise_claim": bool(enough),
               "why": "on a bf16-output criterion an exact and a non-exact transform are "
                      "indistinguishable below ~1e5 compared elements (divergence rate "
                      "1.2e-4..4.9e-4 per element, growing with sqrt(k)); a verdict over fewer "
                      "elements is not evidence of bitwise equality, only of agreement",
               "allow_small_sample": bool(a.allow_small_sample)},
           "what_this_does_NOT_gate": [
               "device arithmetic (see bi_gate_kernel.py receipt on the GPU)",
               "the real model's logits (see bi_gate_engine.py on the GPU)"]}
    rec["record_digest"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: rec[k] for k in ("gate", "verdict", "controls_all_detected", "wall_s")}
                     | {"state_ring": ring["verdict"],
                        "comparisons": main_run["comparisons"],
                        "logits_rows": main_run["logits_rows_compared"],
                        "output_elements": elements,
                        "sufficient_for_a_bitwise_claim": bool(enough),
                        "out": str(out)}, indent=2))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
