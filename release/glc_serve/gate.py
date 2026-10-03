"""Startup gates: bit-exact weights, and logits against the dense parent.

WEIGHT GATE (``run_weight_gate``).  For a sample of tensors (the manifest's
deterministic ``gate.sample``: every coded head/MTP tensor, vision tensors,
the largest tensor, evenly spaced text tensors, raw embedding/norm/vision
tensors) -- or every tensor with ``scope="full"`` -- the weight is read back
FROM WHERE IT IS SERVED: a TBE linear is decoded on the GPU by the serving
kernel (``tbe_mma_decode``), an FWP1 linear by its own decoder, a
host-resident embedding from host RAM, a raw tensor from its device.  The
bytes are hashed (blake2b over the raw bf16 bit patterns) and compared to the
``blake2b_source`` the transcoder recorded when it read the bf16 checkpoint.
Equality is bit-identity with the source checkpoint; one mismatch fails the
gate.  Shard sha256s were already verified before any byte was used.

LOGITS GATE (``check_against_reference``).  Runs fixed probes -- four text
prompts (the certified receipt's) and two image+text prompts over
deterministic synthetic images -- and compares last-position logits with a
reference produced by the dense parent (``make-reference``), bf16 on both
sides.  Reported per probe: max |delta|, top-5 overlap, top-1 match, and
whether the logits are BITWISE equal.  Pass = max |delta| <= the identity
noise floor measured for this stack (0.1875, ``certify_receipt_v2``) and
top-1 match on every probe; ``exec_mode=exact`` additionally expects bitwise
equality and reports it (never assumed).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .bundle import blake2b_tensor_hex, component_of

GATE_SCHEMA = "georefine.tbe_serve_weight_gate.v1"
LOGITS_GATE_SCHEMA = "georefine.tbe_serve_logits_gate.v1"
REFERENCE_SCHEMA = "georefine.tbe_serve_logits_reference.v1"
IDENTITY_NOISE_FLOOR = 0.1875
IDENTITY_NOISE_SOURCE = ".icc/evidence/glc-tbe-serving-20260902/qwen38-27b/certify_receipt_v2.json"

TEXT_PROBES = [
    "The capital of France is Paris, and the capital of Japan is",
    "Water boils at a temperature of",
    "In 1969, humans first walked on the",
    "The chemical symbol for gold is",
]


def _weight_of(module: nn.Module, name: str) -> torch.Tensor:
    """The served bytes of checkpoint tensor ``name`` held by ``module``."""
    from .modules import HostEmbedding, TBEServeLinear

    if isinstance(module, TBEServeLinear):
        from glc_loader.tbe_mma import tbe_mma_decode

        return tbe_mma_decode(module.temporary_device_container())
    cls = type(module).__name__
    if cls == "GLCLinear":
        from glc_loader.container import decode_fwp1

        return decode_fwp1(module.container)
    if isinstance(module, HostEmbedding):
        return module.weight
    attr = name.rsplit(".", 1)[-1]
    t = getattr(module, attr, None)
    if not isinstance(t, torch.Tensor):
        raise KeyError(f"{name}: {cls} has no tensor {attr!r}")
    return t


def run_weight_gate(loaded, *, scope: str = "sample") -> Dict[str, Any]:
    bundle = loaded.bundle
    t0 = time.perf_counter()
    if bundle is None:
        return {"schema": GATE_SCHEMA, "status": "skipped", "reason": "no_bundle"}
    if scope == "full":
        names = [t["name"] for t in bundle.tensors]
    else:
        names = list((bundle.manifest.get("gate") or {}).get("sample") or [])
    by_name = {t["name"]: t for t in bundle.tensors}
    checked, mismatched, missing = [], [], []
    per_comp: Dict[str, int] = {}
    for name in names:
        module = loaded.name_to_module.get(name)
        if module is None:
            missing.append(name)   # e.g. vision/MTP not loaded in a debug config
            continue
        entry = by_name[name]
        w = _weight_of(module, name)
        shape = [int(d) for d in entry["shape"]]
        if list(w.shape) != shape:
            w = w.reshape(shape)
        got = blake2b_tensor_hex(w)
        ok = got == entry["blake2b_source"]
        checked.append(name)
        comp = component_of(name)
        per_comp[comp] = per_comp.get(comp, 0) + 1
        if not ok:
            mismatched.append({"name": name, "got": got[:16], "want": entry["blake2b_source"][:16]})
        del w
    shards_ok = all(r.get("sha256_ok") for r in loaded.receipt.get("shard_fetch", []))
    status = "ok" if (checked and not mismatched and shards_ok) else "failed"
    return {
        "schema": GATE_SCHEMA,
        "status": status,
        "scope": scope,
        "reference": "blake2b of the bf16 source bytes recorded at transcode (manifest)",
        "decoded_by": "the serving path (tbe_mma_decode on the GPU for TBE linears)",
        "n_checked": len(checked),
        "n_mismatched": len(mismatched),
        "mismatched": mismatched[:16],
        "not_loaded_in_this_config": missing[:32],
        "per_component": per_comp,
        "all_shards_sha256_verified": shards_ok,
        "n_shards_verified": len(loaded.receipt.get("shard_fetch", [])),
        "seconds": round(time.perf_counter() - t0, 3),
    }


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------
def synthetic_images() -> List[Any]:
    """Two deterministic images (no files, no network)."""
    from PIL import Image, ImageDraw

    a = Image.new("RGB", (448, 448), (255, 255, 255))
    d = ImageDraw.Draw(a)
    d.ellipse((40, 40, 200, 200), fill=(220, 30, 30))
    d.rectangle((240, 240, 400, 400), fill=(30, 60, 220))
    d.text((60, 300), "GEO 42", fill=(0, 0, 0))
    b = Image.new("RGB", (448, 448), (250, 250, 250))
    d = ImageDraw.Draw(b)
    for i, h in enumerate((120, 300, 200, 80)):
        x0 = 40 + i * 100
        d.rectangle((x0, 420 - h, x0 + 60, 420), fill=(40 + 50 * i, 120, 200 - 40 * i))
    return [a, b]


def image_probes() -> List[Dict[str, Any]]:
    imgs = synthetic_images()
    return [
        {"id": "img-shapes", "image": imgs[0],
         "text": "Describe the shapes and colours in this image, and read any text."},
        {"id": "img-bars", "image": imgs[1],
         "text": "Which bar in this chart is the tallest? Answer with its position."},
    ]


def _image_digest(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


@torch.no_grad()
def probe_logits(engine) -> List[Dict[str, Any]]:
    """Last-position logits for every probe, via the engine's own input path."""
    from .engine import Request, SamplingParams

    out = []
    probes = [{"id": f"text-{i}", "kind": "completion", "prompt": p}
              for i, p in enumerate(TEXT_PROBES)]
    if getattr(engine.processor, "image_processor", None) is not None:
        for ip in image_probes():
            probes.append({"id": ip["id"], "kind": "chat", "image": ip["image"],
                           "messages": [{"role": "user", "content": [
                               {"type": "image"}, {"type": "text", "text": ip["text"]}]}],
                           "image_sha256": _image_digest(ip["image"])})
    for p in probes:
        req = Request(kind=p["kind"], params=SamplingParams(max_tokens=1, temperature=0.0),
                      prompt=p.get("prompt"), messages=p.get("messages"),
                      chat_template_kwargs={"enable_thinking": False})
        if p.get("image") is not None:
            req.images = [p["image"]]
        inputs = engine.prepare([req])
        logits = engine.model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1]
        out.append({"id": p["id"], "logits": logits.detach().to("cpu"),
                    "image_sha256": p.get("image_sha256"),
                    "input_sha256": hashlib.sha256(
                        inputs["input_ids"].cpu().numpy().tobytes()).hexdigest()})
    return out


def compare(ref: List[Dict[str, Any]], got: List[Dict[str, Any]], *,
            gate: float = IDENTITY_NOISE_FLOOR, exact: bool = False) -> Dict[str, Any]:
    def ids(rows):
        vals = [r.get("id") for r in rows]
        return vals, {x for x in vals if x is not None}, len(vals) == len(set(vals)) and None not in vals

    ref_ids, ref_set, ref_unique = ids(ref)
    got_ids, got_set, got_unique = ids(got)
    duplicates_or_invalid = not ref_unique or not got_unique
    id_sets_match = ref_set == got_set and not duplicates_or_invalid
    by_id = {r["id"]: r for r in ref if r.get("id") is not None}
    rows = []
    for g in got:
        r = by_id.get(g.get("id"))
        if r is None:
            continue
        a_raw, b_raw = r.get("logits"), g.get("logits")
        if not isinstance(a_raw, torch.Tensor) or not isinstance(b_raw, torch.Tensor):
            finite = False
            delta = float("inf")
            a = b = torch.empty(0)
        else:
            compatible = a_raw.ndim == b_raw.ndim == 1 and a_raw.shape == b_raw.shape
            if compatible:
                a, b = a_raw.float(), b_raw.float()
            else:
                a = b = torch.empty(0)
            finite = (a_raw.ndim == b_raw.ndim == 1 and a_raw.shape == b_raw.shape and
                      a_raw.numel() > 0 and
                      bool(torch.isfinite(a_raw).all() and torch.isfinite(b_raw).all()))
            delta = float((a - b).abs().max().item()) if finite else float("inf")
        ta = set(torch.topk(a, min(5, a.numel())).indices.tolist()) if a.numel() else set()
        tb = set(torch.topk(b, min(5, b.numel())).indices.tolist()) if b.numel() else set()
        image_match = r.get("image_sha256") == g.get("image_sha256")
        if str(g.get("id", "")).startswith("img-"):
            image_match = (image_match and isinstance(g.get("image_sha256"), str) and
                           len(g["image_sha256"]) == 64)
        inputs_match = r.get("input_sha256") == g.get("input_sha256")
        inputs_match = (inputs_match and isinstance(g.get("input_sha256"), str) and
                        len(g["input_sha256"]) == 64)
        bitwise = (a_raw.dtype == b_raw.dtype and a_raw.shape == b_raw.shape and
                   bool(torch.equal(a_raw.contiguous().view(torch.uint8),
                                    b_raw.contiguous().view(torch.uint8)))
                   if isinstance(a_raw, torch.Tensor) and isinstance(b_raw, torch.Tensor)
                   else False)
        rows.append({
            "id": g["id"],
            "same_input": inputs_match,
            "same_image": image_match,
            "finite_logits": finite,
            "max_abs_delta": delta,
            "bitwise_equal": bitwise,
            "top1_match": bool(a.numel()) and int(a.argmax()) == int(b.argmax()),
            "top5_overlap": len(ta & tb),
        })
    ok = (id_sets_match and len(rows) == len(ref) == len(got) and bool(rows) and
          all(x["same_input"] and x["same_image"] and x["finite_logits"] and
              x["max_abs_delta"] <= gate and x["top1_match"] and
              (not exact or x["bitwise_equal"]) for x in rows))
    return {
        "schema": LOGITS_GATE_SCHEMA,
        "status": "ok" if ok else "failed",
        "gate_max_abs_delta": gate,
        "gate_source": IDENTITY_NOISE_SOURCE,
        "exact_required": exact,
        "probe_ids_match": id_sets_match,
        "duplicate_or_invalid_probe_ids": duplicates_or_invalid,
        "missing_probe_ids": sorted(ref_set - got_set),
        "unexpected_probe_ids": sorted(got_set - ref_set),
        "n_probes": len(rows),
        "n_image_probes": sum(1 for x in rows if x["id"].startswith("img")),
        "all_bitwise_equal": bool(rows) and all(x["bitwise_equal"] for x in rows),
        "max_abs_delta": max((x["max_abs_delta"] for x in rows), default=None),
        "per_probe": rows,
    }


def save_reference(path: os.PathLike | str, rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    torch.save({"schema": REFERENCE_SCHEMA, "meta": meta, "probes": rows}, str(tmp))
    os.replace(tmp, p)


def load_reference(path) -> Dict[str, Any]:
    ref = torch.load(str(path), map_location="cpu", weights_only=False)
    if ref.get("schema") != REFERENCE_SCHEMA:
        raise ValueError(f"{path}: not a {REFERENCE_SCHEMA} file")
    return ref


def check_against_reference(engine, path) -> Dict[str, Any]:
    ref = load_reference(path)
    got = probe_logits(engine)
    exact = getattr(engine.loaded.options, "exec_mode", None) == "exact"
    res = compare(ref["probes"], got, exact=exact)
    res["reference_meta"] = ref.get("meta")
    res["exec_mode"] = getattr(engine.loaded.options, "exec_mode", None)
    res["backend"] = engine.loaded.backend_label
    return res


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="glc_serve.gate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make-reference", help="dense parent logits for the probe set")
    mk.add_argument("--dense", required=True)
    mk.add_argument("--out", required=True)
    mk.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    from transformers import AutoProcessor, AutoTokenizer

    from .engine import Engine
    from .loader import ServeOptions, load_dense_parent

    loaded = load_dense_parent(args.dense, ServeOptions(device=args.device, load_mtp=False))
    tok = AutoTokenizer.from_pretrained(args.dense)
    try:
        proc = AutoProcessor.from_pretrained(args.dense)
    except Exception as exc:
        proc = None
        processor_error = exc
    else:
        processor_error = None
    config = getattr(loaded.model, "config", None)
    text_config = (config.get_text_config() if config is not None and
                   hasattr(config, "get_text_config") else config)
    multimodal = (getattr(config, "vision_config", None) is not None or
                  getattr(config, "vision_tower", None) is not None or
                  getattr(config, "image_token_index", None) is not None or
                  getattr(text_config, "image_token_index", None) is not None)
    if multimodal and getattr(proc, "image_processor", None) is None:
        message = "dense parent is multimodal but has no usable image processor"
        if processor_error is not None:
            raise RuntimeError(message) from processor_error
        raise RuntimeError(message)
    eng = Engine(loaded, proc, tok)
    rows = probe_logits(eng)
    if getattr(proc, "image_processor", None) is not None and not all(
            any(r.get("id") == image_id for r in rows) for image_id in ("img-shapes", "img-bars")):
        raise RuntimeError("multimodal reference is missing required image probes")
    save_reference(args.out, rows, {"dense": str(args.dense), "device": args.device,
                                     "torch": torch.__version__,
                                     "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                  time.gmtime())})
    print(json.dumps({"status": "ok", "out": args.out, "n_probes": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
