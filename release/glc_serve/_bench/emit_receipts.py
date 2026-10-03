#!/usr/bin/env python3
"""Assemble the production-serving receipts (ps1..ps4) from what the gates and the bench wrote.

The ICC completion oracle ``georefine-production-serving`` fails on "receipt missing", not on a
measurement that went the wrong way.  This script turns the artefacts a benchmark run already
produces into ICC-shaped receipts under ``receipts/``, so one ``tar`` of that directory grades
the oracle.

It measures nothing itself and invents nothing.  Every item is emitted with
``status: "measured"`` and the numbers, or ``status: "not_run"`` and the exact command that
would produce it -- never a blank that reads as a pass.

    ps1  export bit-exactness        artifact digest + round-trip, from the codec verifier
    ps2  engine load                 weight bytes, resident VRAM, load time, from /v1/receipt + NVML
    ps3  smaller and faster than bf16 THROUGH the served lane, per N, codec arm vs dense arm
    ps4  behavioural identity THROUGH the served lane: served output == the offline single-user
         reference, per N, token-for-token

Inputs (all optional; what is absent becomes a not_run item):
    --verify-json      output of the codec verifier (glc_loader verify / georefine-verify)
    --server-receipt   GET /v1/receipt of the running server, saved to a file
    --nvml-load        two-line CSV of device memory.used before and after the load
    --load-seconds     wall seconds of the load
    --bench-codec      bench_serve.py output dirs for the codec arm (repeatable)
    --bench-dense      bench_serve.py output dirs for the dense bf16 arm (repeatable)
    --gate-http        bi_gate_stream.py receipt (served, per concurrency)
    --gate-engine      bi_gate_engine.py receipt (the offline single-user reference)
    --gate-sched       bi_gate_batchexact.py receipt (CPU scheduler gate)
    --gate-kernel      bi_gate_kernel.py receipt (BI-GEMM batch invariance)

    python scripts/batchserve/emit_receipts.py --out receipts/ \\
        --server-receipt receipt.json --gate-http receipt_http.json \\
        --gate-engine receipt_engine.json --gate-sched receipt_sched.json \\
        --bench-codec results/bidec_chat_c1 --bench-dense results/dense_chat_c1 ...

    python scripts/batchserve/emit_receipts.py --self-test     # CPU check of the assembler
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = "georefine.production_serving_receipt.v1"

# The ICC oracle `production-serving-compressed` grades by FILE NAME
# (.icc/completion-oracles.yaml:5838 onward), so each item is also written under the name its
# criterion requires.  Verified against that file, not guessed.
ORACLE_TARGET = "production-serving-compressed"
# Compared output elements below which an exact and a non-exact transform are not
# distinguishable on a bf16-output criterion (Lane I, 2026-10-03).
BITWISE_CLAIM_FLOOR = 1_000_000
ORACLE_FILES = {
    "ps3": ["serving_perf_receipt.json"],                      # PS0/PS3 smaller and faster
    "ps4": ["served_lane_behaviour_identity.json"],            # PS4 behaviour through the lane
    "ps1": ["tbe_serve_time_parity_receipt.json"],             # PS6 lossless container at parity
    "ps5": ["kv_cache_gate_receipt.json"],                     # PS5 KV gate
    "ps5v": ["kv_cache_vram_receipt.json"],                    # PS5 KV VRAM saving
    "ps0": ["serving_vram_speed_thresholds.json"],             # PS0 verdict + resource thresholds
}
NOT_RUN = "not_run"
MEASURED = "measured"


def _sh(args, cwd=None) -> str:
    try:
        return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def provenance(repo: Path, argv: List[str]) -> Dict[str, Any]:
    """The provenance axes `icc receipt-validate --compare` refuses to pair across."""
    return {"repo_commit": _sh(["git", "rev-parse", "HEAD"], cwd=str(repo)),
            "dirty_files": _sh(["git", "status", "--porcelain"], cwd=str(repo)).count("\n") + 1
            if _sh(["git", "status", "--porcelain"], cwd=str(repo)) else 0,
            "node": socket.gethostname(), "platform": platform.platform(),
            "python": sys.version.split()[0], "argv": argv,
            "produced_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "producer": "scripts/batchserve/emit_receipts.py"}


_LOAD_ERRORS: Dict[str, str] = {}


def _load(p: Optional[str]) -> Optional[Dict[str, Any]]:
    """None means "not supplied".  A supplied-but-unreadable file is recorded in _LOAD_ERRORS
    and surfaced in the receipt: silently turning it into not_run would hide a broken input."""
    if not p:
        return None
    f = Path(p)
    if not f.exists():
        _LOAD_ERRORS[str(p)] = "file does not exist"
        return None
    try:
        return json.loads(f.read_text())
    except Exception as e:  # noqa: BLE001
        _LOAD_ERRORS[str(p)] = f"{type(e).__name__}: {e}"
        return None


def _bench(d: str) -> List[Dict[str, Any]]:
    """Every summary_c*.json under a bench output dir (or the file itself)."""
    p = Path(d)
    files = sorted(p.glob("summary_c*.json")) if p.is_dir() else ([p] if p.exists() else [])
    out = []
    for f in files:
        try:
            out.append(json.loads(f.read_text()))
        except Exception:
            continue
    return out


def _nv(csv_path: Optional[str]) -> Optional[List[int]]:
    if not csv_path or not Path(csv_path).exists():
        return None
    vals = []
    for line in Path(csv_path).read_text().splitlines():
        for tok in line.replace(",", " ").split():
            if tok.isdigit():
                vals.append(int(tok))
    return vals or None


# --------------------------------------------------------------------------- items
def ps1(verify: Optional[Dict[str, Any]], server: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Export bit-exactness: the artifact's own digest and a round-trip check."""
    if not verify:
        return {"id": "ps1", "claim": "exported artifact is bit-exact and self-verifying",
                "status": NOT_RUN,
                "how": "run the codec verifier on the delivered bundle and pass its JSON as "
                       "--verify-json: `python -m glc_loader verify BUNDLE --json v.json`"}
    bitexact = verify.get("bitexact")
    if bitexact is None:
        bitexact = (verify.get("mismatched_elements") == 0) if "mismatched_elements" in verify else None
    return {"id": "ps1", "claim": "exported artifact is bit-exact and self-verifying",
            "status": MEASURED if bitexact is not None else NOT_RUN,
            "bitexact": bitexact,
            "mismatched_elements": verify.get("mismatched_elements"),
            "manifest_sha256": verify.get("manifest_sha256") or (server or {}).get("manifest_sha256"),
            "artifact_dir": verify.get("artifact_dir") or verify.get("bundle"),
            "verifier": verify.get("producer") or verify.get("schema"),
            "source": "verify-json"}


def ps2(server: Optional[Dict[str, Any]], nvml: Optional[List[int]],
        load_s: Optional[float]) -> Dict[str, Any]:
    """Engine load: weight bytes held, resident device memory, load time."""
    if not server:
        return {"id": "ps2", "claim": "the served engine loads with a recorded memory and time cost",
                "status": NOT_RUN,
                "how": "curl -s -H 'Authorization: Bearer KEY' localhost:PORT/v1/receipt > receipt.json "
                       "and pass it as --server-receipt; capture nvidia-smi memory.used before and "
                       "after the load into a two-line CSV for --nvml-load"}
    sb = server.get("state_bytes") or {}
    item = {"id": "ps2", "claim": "the served engine loads with a recorded memory and time cost",
            "status": MEASURED, "engine": server.get("engine"),
            # the GDN ring dtype changes the fixed per-slot term, so no memory number from this
            # lane is interpretable without it
            "gdn_ring_dtype": server.get("gdn_ring_dtype") or (sb or {}).get("gdn_ring_dtype"),
            "coded_weight_bytes_streamed_per_step": server.get("streamed_bytes_per_step"),
            "state_bytes": sb, "state_bytes_total": sum(v for v in sb.values() if isinstance(v, int)),
            "slots": server.get("slots"), "pages": server.get("pages"),
            "max_ctx": server.get("max_ctx"), "capture_s": server.get("capture_s"),
            "load_receipt": server.get("load"), "load_seconds": load_s}
    if nvml and len(nvml) >= 2:
        item["device_mib_before_load"] = nvml[0]
        item["device_mib_after_load"] = nvml[-1]
        item["resident_mib"] = nvml[-1] - nvml[0]
    else:
        item["resident_mib"] = None
        item["resident_mib_note"] = "no --nvml-load supplied; device residency NOT measured"
    return item


def _by_n(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out = {}
    for r in rows:
        try:
            out[int(r["concurrency"])] = r
        except Exception:
            continue
    return out


def ps3(codec: List[Dict[str, Any]], dense: List[Dict[str, Any]],
        server: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Smaller AND faster than bf16, measured through the served lane, per concurrency."""
    c, d = _by_n(codec), _by_n(dense)
    shared = sorted(set(c) & set(d))
    if not shared:
        return {"id": "ps3", "claim": "the codec is smaller and no slower than dense bf16 through "
                                      "the served lane, at every concurrency",
                "status": NOT_RUN,
                "n_codec_arms": len(c), "n_dense_arms": len(d),
                "how": "run bench_serve.py at the same concurrencies against (a) the codec server "
                       "and (b) a dense bf16 server of the same engine on the same card in the "
                       "same session, then pass both sets of output dirs as --bench-codec / "
                       "--bench-dense. Without the dense arm this is not a comparison."}
    table, regress = [], []
    for n in shared:
        a, b = c[n], d[n]
        sa, sb = a.get("steady_tok_s"), b.get("steady_tok_s")
        ma = (a.get("nvml") or {}).get("mem_used_mib_p50") or (a.get("nvml") or {}).get("mem_used_mib_max")
        mb = (b.get("nvml") or {}).get("mem_used_mib_p50") or (b.get("nvml") or {}).get("mem_used_mib_max")
        row = {"concurrency": n, "codec_steady_tok_s": sa, "dense_steady_tok_s": sb,
               "speed_ratio": round(sa / sb, 4) if sa and sb else None,
               "codec_mem_mib": ma, "dense_mem_mib": mb,
               "mem_ratio": round(ma / mb, 4) if ma and mb else None,
               "codec_ttft_p50": (a.get("ttft_s") or {}).get("p50"),
               "dense_ttft_p50": (b.get("ttft_s") or {}).get("p50"),
               "codec_usd_per_m": a.get("cost_usd_per_m_output"),
               "dense_usd_per_m": b.get("cost_usd_per_m_output"),
               "codec_method_note": a.get("method_note"), "dense_method_note": b.get("method_note")}
        if row["speed_ratio"] is not None and row["speed_ratio"] < 1.0:
            regress.append(n)
        if row["mem_ratio"] is not None and row["mem_ratio"] > 1.0:
            regress.append(n)
        table.append(row)
    return {"id": "ps3", "claim": "the codec is smaller and no slower than dense bf16 through the "
                                  "served lane, at every concurrency",
            "status": MEASURED, "per_concurrency": table,
            "concurrencies_measured": shared,
            "concurrencies_with_a_regression": sorted(set(regress)),
            "verdict": "PASS" if not regress else "FAIL",
            "coded_weight_bytes_streamed_per_step": (server or {}).get("streamed_bytes_per_step"),
            "gdn_ring_dtype": (server or {}).get("gdn_ring_dtype"),
            "note": "a missing nvml p50 makes mem_ratio null, which is not a pass; and two arms "
                    "with different gdn_ring_dtype are not comparable on memory"}


def ps4(http: Optional[Dict[str, Any]], engine: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Behavioural identity through the served lane: served output == the offline reference."""
    if not http:
        return {"id": "ps4", "claim": "served greedy output is identical to the offline "
                                      "single-user reference, at every concurrency",
                "status": NOT_RUN,
                "how": "run bi_gate_stream.py against the server and pass its receipt as "
                       "--gate-http; add bi_gate_engine.py's receipt as --gate-engine for the "
                       "offline engine reference and the per-logits-row bitwise comparison"}
    det = http.get("detail") or []
    per: Dict[Any, Dict[str, int]] = {}
    for ch in det:
        k = ch.get("concurrency")
        s = per.setdefault(k, {"checks": 0, "identical": 0})
        s["checks"] += 1
        s["identical"] += 1 if ch.get("ok") else 0
    sample = _sample_size(http, engine)
    item = {"id": "ps4", "claim": "served greedy output is identical to the offline single-user "
                                  "reference, at every concurrency",
            "status": MEASURED,
            "served_vs_single_user_served": {"verdict": http.get("verdict"),
                                             "failures": http.get("failures"),
                                             "checks": http.get("checks"),
                                             "per_concurrency": per,
                                             "level": "streamed text, token count, finish reason"},
            "verdict": http.get("verdict"), "sample_size": sample,
            "evidence_level": ("bitwise" if sample["sufficient_for_a_bitwise_claim"]
                               else "agreement, not demonstrated bitwise equality: "
                                    f"{sample['output_elements_compared']} output elements "
                                    f"compared against a floor of {BITWISE_CLAIM_FLOOR}")}
    if engine:
        tok = engine.get("probe_token_mismatch")
        dig = engine.get("probe_digest_rows_mismatch")
        item["offline_engine_bitwise"] = {
            "verdict": engine.get("verdict"), "probe_token_mismatch": tok,
            "probe_digest_rows_mismatch": dig, "rows_compared": engine.get("rows_compared"),
            "level": "sha256 of every emitted logits row, solo vs batch schedules"}
        if engine.get("verdict") not in (None, "PASS"):
            item["verdict"] = "FAIL"
    else:
        item["offline_engine_bitwise"] = {"status": NOT_RUN,
                                          "how": "bi_gate_engine.py (GPU) -> --gate-engine"}
    return item


def ps5(kv_gate: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """PS5a: KV quantisation on the served lane passes the KV behavioural gate.

    int8 per-token KV is the certified ceiling in this repo: it passed 6/6 on the 2B lane
    (SAE 0.997845, mass-cosine 0.996146), while int4 FAILED the chat gate at both schemes and
    int2 failed as the designed control.  So the gate to re-run on the served lane is int8, and
    int4 is the negative control that must fail.
    """
    if not kv_gate:
        return {"id": "ps5", "claim": "KV quantisation on the served lane passes the KV gate",
                "status": NOT_RUN,
                "certified_scheme": "int8 per-token (int4 fails the chat gate; int2 is the control)",
                "how": "python -m experiments.georefine.m2_recertify --kv-quant int8 "
                       "--kv-scheme per-token --kv-group 64 ... then pass its receipt as "
                       "--kv-gate; re-run with --kv-quant int4 as the negative control, which "
                       "must FAIL"}
    return {"id": "ps5", "claim": "KV quantisation on the served lane passes the KV gate",
            "status": MEASURED, "verdict": kv_gate.get("verdict") or kv_gate.get("status"),
            "scheme": kv_gate.get("kv_scheme") or kv_gate.get("scheme"),
            "mode": kv_gate.get("kv_quant") or kv_gate.get("mode"),
            "survival": kv_gate.get("survival") or kv_gate.get("rare_feature_survival"),
            "mass_cosine": kv_gate.get("mass_cosine"),
            "negative_control": kv_gate.get("negative_control"),
            "source": "kv-gate"}


def ps5v(kv_vram: Optional[Dict[str, Any]], capacity: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """PS5b: the KV saving is MEASURED on the lane, not projected.

    The repo's existing KV-quant path is fake-quant -- storage stays bf16 and its own receipts
    carry `memory_saving_is_projection: true`.  A projection is not a measurement, so this item
    stays not_run until a packed paged KV buffer exists and device memory moves.
    """
    item = {"id": "ps5v", "claim": "the KV-cache saving is measured on the served lane, "
                                   "not projected from bytes-per-token",
            "status": NOT_RUN,
            "how": "run bench_serve.py at a fixed N and context with KV quantisation off and "
                   "on, and report NVML steady device memory for both; a bytes-per-token "
                   "projection does not satisfy this item"}
    if capacity:
        item["capacity_plan"] = {"kv_bytes_per_token": capacity.get("kv_bytes_per_token"),
                                 "gdn_state_bytes_per_slot": capacity.get("gdn_state_bytes_per_slot"),
                                 "crossover_tokens": capacity.get("crossover_tokens"),
                                 "note": capacity.get("crossover_note")}
    if not kv_vram:
        return item
    item.update({"status": MEASURED,
                 "device_mib_kv_quant_off": kv_vram.get("off_mib"),
                 "device_mib_kv_quant_on": kv_vram.get("on_mib"),
                 "saving_mib": (kv_vram.get("off_mib") - kv_vram.get("on_mib"))
                 if kv_vram.get("off_mib") is not None and kv_vram.get("on_mib") is not None
                 else None,
                 "concurrency": kv_vram.get("concurrency"), "ctx": kv_vram.get("ctx"),
                 "measurement": "NVML steady device memory, same N and context both arms",
                 "projection": bool(kv_vram.get("memory_saving_is_projection"))})
    if item.get("projection"):
        item["status"] = NOT_RUN
        item["why_not_measured"] = ("the supplied receipt is flagged "
                                    "memory_saving_is_projection: a projection is not a "
                                    "measurement of device memory")
    return item


def _sample_size(*receipts) -> Dict[str, Any]:
    """Pool the compared-element counts of the receipts that carry one.

    A bit-exactness verdict is only evidence in proportion to how much output it compared: on a
    bf16-output criterion an exact and a non-exact transform are indistinguishable below ~1e5
    elements (divergence 1.2e-4..4.9e-4 per element, growing with sqrt(k)).  So every receipt
    that claims bitwise equality must state its element count, and this assembler refuses the
    word "bitwise" when the pooled count is short.
    """
    rows = elems = toks = 0
    sources, missing = [], []
    for r in receipts:
        if not r:
            continue
        ss = r.get("sample_size") or {}
        if not ss:
            missing.append(r.get("gate") or "unnamed receipt")
            continue
        rows += int(ss.get("logits_rows_compared") or 0)
        elems += int(ss.get("output_elements_compared") or 0)
        toks += int(ss.get("tokens_compared") or 0)
        sources.append({"gate": r.get("gate"), **ss})
    return {"logits_rows_compared": rows, "output_elements_compared": elems,
            "tokens_compared": toks, "bitwise_claim_floor": BITWISE_CLAIM_FLOOR,
            "sufficient_for_a_bitwise_claim": elems >= BITWISE_CLAIM_FLOOR,
            "receipts_without_a_sample_size": missing or None, "per_receipt": sources or None}


def ps0(http: Optional[Dict[str, Any]], engine: Optional[Dict[str, Any]],
        codec: List[Dict[str, Any]], ps2_item: Dict[str, Any],
        thresholds: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """PS0: the behavioural verdict and the resource numbers in ONE record, graded against
    thresholds that were declared before the run.

    Separating "it is exact" from "it is small and fast" lets a release quote whichever half
    looks better.  This item refuses that: the exactness verdict, resident bytes and steady
    tok/s per concurrency live in one receipt with one provenance block, and each is compared to
    a declared threshold.  No thresholds supplied -> the item is not_run, never a pass.
    """
    rows = []
    for r in sorted(codec, key=lambda x: int(x.get("concurrency") or 0)):
        nv = r.get("nvml") or {}
        rows.append({"concurrency": r.get("concurrency"),
                     "steady_tok_s": r.get("steady_tok_s"),
                     "ttft_p50_s": (r.get("ttft_s") or {}).get("p50"),
                     "ttft_p99_s": (r.get("ttft_s") or {}).get("p99"),
                     "resident_mib_steady": nv.get("mem_used_mib_p50_steady") or nv.get("mem_used_mib_p50"),
                     "resident_mib_peak": nv.get("mem_used_mib_max"),
                     "cost_usd_per_m_output": r.get("cost_usd_per_m_output")})
    identity = (http or {}).get("verdict")
    bitwise = (engine or {}).get("verdict")
    sample = _sample_size(http, engine)
    item = {"id": "ps0",
            "claim": "on the serving card: served greedy output equals the reference, resident "
                     "VRAM is under the declared target, and steady tok/s is at or above the "
                     "declared floor -- one record, one provenance block",
            "exactness": {"served_vs_single_user": identity, "offline_bitwise": bitwise,
                          "sample_size": sample},
            "per_concurrency": rows,
            "resident_mib_at_load": ps2_item.get("resident_mib"),
            "gdn_ring_dtype": ps2_item.get("gdn_ring_dtype")}
    if not thresholds:
        item["status"] = NOT_RUN
        item["how"] = ("declare the thresholds BEFORE the run in a JSON "
                       "{\"resident_mib_max\": N, \"steady_tok_s_min\": {\"1\": x, \"4\": y}, "
                       "\"bf16_reference_tok_s\": {...}} and pass it as --thresholds; a receipt "
                       "with no declared threshold cannot pass or fail")
        return item
    item["thresholds"] = thresholds
    checks, fails = [], []
    rmax = thresholds.get("resident_mib_max")
    floors = {str(k): v for k, v in (thresholds.get("steady_tok_s_min") or {}).items()}
    ref = {str(k): v for k, v in (thresholds.get("bf16_reference_tok_s") or {}).items()}
    for r in rows:
        n = str(r["concurrency"])
        res, tok = r["resident_mib_steady"], r["steady_tok_s"]
        c = {"concurrency": r["concurrency"]}
        if rmax is not None:
            c["resident_ok"] = (res is not None and res <= rmax)
        if n in floors:
            c["tok_s_floor_ok"] = (tok is not None and tok >= floors[n])
        if n in ref:
            c["at_or_above_bf16_ok"] = (tok is not None and tok >= ref[n])
        checks.append(c)
        for k, v in c.items():
            if k != "concurrency" and v is False:
                fails.append(f"N={n}:{k}")
    item["checks"] = checks
    item["failures"] = fails
    if not rows or not checks:
        item["status"] = NOT_RUN
        item["how"] = "no bench summaries supplied for the codec arm (--bench-codec)"
        return item
    exact_ok = identity == "PASS" and (bitwise in (None, "PASS"))
    item["status"] = MEASURED
    item["verdict"] = "PASS" if (exact_ok and not fails) else "FAIL"
    if bitwise is None:
        item["caveat"] = ("the offline bitwise gate was not supplied, so exactness rests on the "
                          "HTTP text-level gate alone")
    return item


def supporting(sched: Optional[Dict[str, Any]], kernel: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["scheduler_exactness"] = ({"verdict": sched.get("verdict"),
                                   "controls_all_detected": sched.get("controls_all_detected"),
                                   "comparisons": (sched.get("main") or {}).get("comparisons"),
                                   "rows_compared": (sched.get("main") or {}).get("rows_compared")}
                                  if sched else {"status": NOT_RUN,
                                                 "how": "bi_gate_batchexact.py (CPU) -> --gate-sched"})
    out["kernel_batch_invariance"] = ({"verdict": kernel.get("verdict"),
                                       "cases": kernel.get("cases") or kernel.get("n_cases"),
                                       "rows_differing": kernel.get("rows_differing")}
                                      if kernel else {"status": NOT_RUN,
                                                      "how": "bi_gate_kernel.py (GPU) -> --gate-kernel"})
    return out


def build(a, repo: Path, argv: List[str]) -> Dict[str, Any]:
    verify, server = _load(a.verify_json), _load(a.server_receipt)
    http, engine = _load(a.gate_http), _load(a.gate_engine)
    sched, kernel = _load(a.gate_sched), _load(a.gate_kernel)
    codec = [r for d in (a.bench_codec or []) for r in _bench(d)]
    dense = [r for d in (a.bench_dense or []) for r in _bench(d)]
    capacity = _load(getattr(a, "capacity", None))
    ps2_item = ps2(server, _nv(a.nvml_load), a.load_seconds)
    items = [ps0(http, engine, codec, ps2_item, _load(getattr(a, "thresholds", None))),
             ps1(verify, server), ps2_item,
             ps3(codec, dense, server), ps4(http, engine),
             ps5(_load(getattr(a, "kv_gate", None))),
             ps5v(_load(getattr(a, "kv_vram", None)), capacity)]
    rec = {"schema": SCHEMA, "oracle": ORACLE_TARGET,
           "oracle_invocation": "icc --allow-stale-store completion-oracle "
                                f"--repo <repo> --target {ORACLE_TARGET}",
           "serving_lane": "glc_serve.bidec (BI-GEMM continuous batching, multi-user)",
           "provenance": provenance(repo, argv), "items": items,
           "supporting": supporting(sched, kernel), "capacity_plan": capacity,
           "oracle_file_map": ORACLE_FILES,
           "input_errors": dict(_LOAD_ERRORS) or None,
           "sample_size": _sample_size(http, engine, sched, kernel),
           "summary": {"measured": [i["id"] for i in items if i["status"] == MEASURED],
                       "not_run": [i["id"] for i in items if i["status"] == NOT_RUN],
                       "failing": [i["id"] for i in items if i.get("verdict") == "FAIL"]}}
    rec["record_digest"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
    return rec


# --------------------------------------------------------------------------- self-test
def self_test() -> int:
    """Assemble from synthetic fixtures and check the assembler's logic, with no GPU."""
    import tempfile
    repo = Path(__file__).resolve().parents[2]
    fails = []
    with tempfile.TemporaryDirectory(dir=str(repo / ".scratch") if (repo / ".scratch").exists()
                                     else None) as td:
        t = Path(td)
        (t / "verify.json").write_text(json.dumps({"bitexact": True, "mismatched_elements": 0,
                                                   "manifest_sha256": "a" * 64, "bundle": "/b"}))
        (t / "srv.json").write_text(json.dumps({"engine": "bidec", "slots": 8, "pages": 200,
                                                "max_ctx": 4096, "streamed_bytes_per_step": 123,
                                                "state_bytes": {"kv": 10, "gdn_state": 5}}))
        (t / "nv.csv").write_text("memory.used [MiB]\n512\n40960\n")
        for arm, tok, mem in (("codec", 120.0, 30000), ("dense", 100.0, 40000)):
            d = t / arm
            d.mkdir()
            (d / "summary_c4.json").write_text(json.dumps(
                {"concurrency": 4, "steady_tok_s": tok, "nvml": {"mem_used_mib_p50": mem},
                 "ttft_s": {"p50": 1.0}, "cost_usd_per_m_output": 1.0, "method_note": arm}))
        (t / "http.json").write_text(json.dumps(
            {"verdict": "PASS", "failures": 0, "checks": 2,
             "detail": [{"concurrency": 4, "ok": True}, {"concurrency": 8, "ok": True}]}))
        (t / "eng.json").write_text(json.dumps({"verdict": "PASS", "probe_token_mismatch": 0,
                                                "probe_digest_rows_mismatch": 0,
                                                "rows_compared": 288}))
        (t / "sched.json").write_text(json.dumps(
            {"verdict": "PASS", "controls_all_detected": True,
             "main": {"comparisons": 24, "rows_compared": 576},
             "sample_size": {"logits_rows_compared": 576, "output_elements_compared": 1179648,
                             "sufficient_for_a_bitwise_claim": True}}))
        (t / "thr.json").write_text(json.dumps({"resident_mib_max": 60000,
                                                "steady_tok_s_min": {"4": 50},
                                                "bf16_reference_tok_s": {"4": 100}}))

        class A:
            verify_json = str(t / "verify.json")
            server_receipt = str(t / "srv.json")
            nvml_load = str(t / "nv.csv")
            load_seconds = 42.0
            bench_codec = [str(t / "codec")]
            bench_dense = [str(t / "dense")]
            gate_http = str(t / "http.json")
            gate_engine = str(t / "eng.json")
            gate_sched = str(t / "sched.json")
            gate_kernel = None
            thresholds = str(t / "thr.json")
            kv_gate = None
            kv_vram = None
            capacity = None

        r = build(A, repo, ["self-test"])
        byid = {i["id"]: i for i in r["items"]}
        def chk(c, why):
            if not c:
                fails.append(why)
        chk(byid["ps1"]["bitexact"] is True, "ps1 bitexact")
        chk(byid["ps2"]["resident_mib"] == 40448, "ps2 resident_mib from NVML")
        chk(byid["ps2"]["state_bytes_total"] == 15, "ps2 state bytes summed")
        chk(byid["ps2"]["load_seconds"] == 42.0, "ps2 load seconds")
        chk(byid["ps3"]["verdict"] == "PASS", "ps3 faster+smaller -> PASS")
        chk(byid["ps3"]["per_concurrency"][0]["speed_ratio"] == 1.2, "ps3 speed ratio")
        chk(byid["ps3"]["per_concurrency"][0]["mem_ratio"] == 0.75, "ps3 mem ratio")
        chk(byid["ps4"]["verdict"] == "PASS", "ps4 PASS")
        chk(byid["ps4"]["served_vs_single_user_served"]["per_concurrency"][4]["identical"] == 1,
            "ps4 per-concurrency rollup")
        chk(r["supporting"]["kernel_batch_invariance"]["status"] == NOT_RUN, "missing kernel = not_run")
        chk(sorted(r["summary"]["not_run"]) == ["ps5", "ps5v"],
            "ps0-ps4 measured, the two KV items not_run without their receipts")
        chk(byid["ps0"]["verdict"] == "PASS", "ps0 passes when exact and within thresholds")
        chk(byid["ps0"]["per_concurrency"][0]["steady_tok_s"] == 120.0, "ps0 carries tok/s")
        chk(ps0({"verdict": "PASS"}, {"verdict": "PASS"},
                _bench(str(t / "codec")), {}, None)["status"] == NOT_RUN,
            "ps0 without declared thresholds must be not_run, never a pass")
        chk(ps0({"verdict": "PASS"}, {"verdict": "PASS"}, _bench(str(t / "codec")), {},
                {"steady_tok_s_min": {"4": 10000}})["verdict"] == "FAIL",
            "ps0 must FAIL when the tok/s floor is missed")
        chk(ps0({"verdict": "FAIL"}, {"verdict": "PASS"}, _bench(str(t / "codec")), {},
                {"resident_mib_max": 10 ** 9})["verdict"] == "FAIL",
            "ps0 must FAIL when the served output is not identical, however fast it is")
        chk("kv_cache_gate_receipt.json" in ORACLE_FILES["ps5"], "oracle name for ps5")
        chk(ps5v({"off_mib": 40000, "on_mib": 32000, "memory_saving_is_projection": True},
                 None)["status"] == NOT_RUN,
            "a projected KV saving must not count as measured")
        chk(ps5v({"off_mib": 40000, "on_mib": 32000}, None)["saving_mib"] == 8000,
            "measured KV saving")

        # a slower codec arm must FAIL ps3, and a failing engine gate must FAIL ps4
        (t / "codec" / "summary_c4.json").write_text(json.dumps(
            {"concurrency": 4, "steady_tok_s": 90.0, "nvml": {"mem_used_mib_p50": 30000},
             "ttft_s": {"p50": 1.0}}))
        chk(ps3(_bench(str(t / "codec")), _bench(str(t / "dense")), None)["verdict"] == "FAIL",
            "ps3 must FAIL when the codec arm is slower")
        chk(ps4(json.loads((t / "http.json").read_text()),
                {"verdict": "FAIL", "probe_token_mismatch": 3})["verdict"] == "FAIL",
            "ps4 must FAIL when the offline bitwise gate fails")
        # no inputs at all: every item not_run, nothing silently passing
        class E:
            verify_json = server_receipt = nvml_load = gate_http = gate_engine = None
            gate_sched = gate_kernel = kv_gate = kv_vram = capacity = thresholds = None
            load_seconds = None
            bench_codec = bench_dense = []
        e = build(E, repo, ["self-test-empty"])
        chk(sorted(e["summary"]["not_run"]) == ["ps0", "ps1", "ps2", "ps3", "ps4", "ps5", "ps5v"],
            "empty = all not_run")
        _LOAD_ERRORS.clear()
        chk(_load(str(t / "nope.json")) is None and _LOAD_ERRORS, "a missing input is recorded")
        # the element floor must actually bind, in both directions
        big = {"sample_size": {"output_elements_compared": 2_000_000}}
        small = {"sample_size": {"output_elements_compared": 50_000}}
        chk(_sample_size(big)["sufficient_for_a_bitwise_claim"] is True, "floor cleared")
        chk(_sample_size(small)["sufficient_for_a_bitwise_claim"] is False, "floor binds")
        chk(_sample_size({"gate": "no counts"})["receipts_without_a_sample_size"] is not None,
            "a receipt with no element count is named, not ignored")
        chk(ps4({"verdict": "PASS", "detail": [], "sample_size": small["sample_size"]},
                None)["evidence_level"].startswith("agreement"),
            "a short sample is reported as agreement, not as bitwise")
        chk(ps4({"verdict": "PASS", "detail": [], "sample_size": big["sample_size"]},
                None)["evidence_level"] == "bitwise", "a sufficient sample is reported as bitwise")
        chk(all("how" in i for i in e["items"]), "every not_run item carries its command")
    for f in fails:
        print(f"FAIL: {f}")
    print(f"self-test: {'PASS' if not fails else 'FAIL'} ({29 - len(fails)}/29 checks)")
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="directory for the receipts (written as production_serving.json)")
    ap.add_argument("--verify-json")
    ap.add_argument("--server-receipt")
    ap.add_argument("--nvml-load")
    ap.add_argument("--load-seconds", type=float)
    ap.add_argument("--bench-codec", action="append")
    ap.add_argument("--bench-dense", action="append")
    ap.add_argument("--gate-http")
    ap.add_argument("--gate-engine")
    ap.add_argument("--gate-sched")
    ap.add_argument("--gate-kernel")
    ap.add_argument("--kv-gate", help="receipt of the KV behavioural gate (int8 per-token)")
    ap.add_argument("--kv-vram", help="measured device memory with KV quantisation off and on")
    ap.add_argument("--thresholds",
                    help="PS0 thresholds DECLARED BEFORE the run: resident_mib_max, "
                         "steady_tok_s_min per N, bf16_reference_tok_s per N")
    ap.add_argument("--capacity", help="output of python -m release.glc_serve.bidec_capacity")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.out:
        ap.error("--out is required (or use --self-test)")
    repo = Path(__file__).resolve().parents[2]
    rec = build(a, repo, sys.argv[1:])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "production_serving.json").write_text(json.dumps(rec, indent=2))
    for i in rec["items"]:
        body = json.dumps({"schema": SCHEMA, "oracle": ORACLE_TARGET,
                           "provenance": rec["provenance"], **i}, indent=2)
        (out / f"{i['id']}.json").write_text(body)
        for name in ORACLE_FILES.get(i["id"], []):       # the name the oracle grades by
            (out / name).write_text(body)
    print(json.dumps(rec["summary"], indent=2))
    print(f"receipts in {out}  ->  tar -czf serving_receipts.tgz {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
