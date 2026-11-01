"""Build a ``georefine.tbe.serve.v1`` bundle once; serve it many times.

Two inputs, one output format:

``--from-container DIR``  re-pack an EXISTING TBE container (``georefine.tbe``
    v1 or v2: ``manifest.json`` + ``tbe_tensors.safetensors`` +
    ``raw_tensors.safetensors``) into layer-grouped shards.  Coded arrays are
    copied as stored -- nothing is decoded or re-encoded unless asked
    (``--code-lm-head`` / ``--code-embed`` encode those raw tensors on the CPU
    and certify the round trip bit-exact before writing).  Every raw tensor is
    re-hashed against the container manifest's ``blake2b_source`` on the way
    through (chain of custody); ``--verify-decode`` additionally CPU-decodes
    every coded tensor against it.  A v1 container carries no config or
    tokenizer, so ``--config-dir`` (the source checkpoint directory, or any
    directory holding its config/tokenizer/processor files) is required.

``--from-dense DIR``  CPU transcode of a bf16 safetensors checkpoint.  Peak
    host memory is one layer group plus its containers, never the checkpoint.
    Eligibility is the served-path rule (2-D bf16, ``numel >= min_numel``,
    ``in % 64 == 0``, ``out % 8 == 0``, not an embedding/head by name unless
    ``--code-lm-head`` / ``--code-embed``); a tensor whose escape rate exceeds
    ``--escape-band-pct`` is kept raw and NAMED in the manifest.

Both write per-shard sha256, per-sidecar sha256, per-tensor ``blake2b`` of the
source bytes, and a deterministic gate sample.  Usage::

    python -m glc_serve.pack --from-container ~/tbe-qwen38-27b \\
        --config-dir ~/models/Qwen3.8-27B --out ~/bundles/qwen38-27b-tbe \\
        --code-lm-head
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .bundle import (
    DEFAULT_TARGET_SHARD_BYTES,
    TBE_FIELDS,
    BundleError,
    BundleWriter,
    blake2b_tensor_hex,
    component_of,
    group_of,
    group_sort_key,
    layer_of,
)

DEFAULT_MIN_NUMEL = 1_048_576
DEFAULT_ESCAPE_BAND_PCT = 9.0
TILE = 64
MMA_N = 8
_SKIP_NAME_SUBSTRINGS = ("embed", "lm_head", "wte", "wpe")


def _int32_words(t):
    """Store 32-bit words as int32 exactly (wrap values >= 2**31)."""
    import torch

    v = t.reshape(t.shape).to(torch.int64)
    v = torch.where(v >= 0x80000000, v - 0x100000000, v)
    return v.to(torch.int32).contiguous()


def container_arrays(c) -> Dict[str, Any]:
    return {
        "planes": _int32_words(c.planes),
        "smb": c.smb.reshape(-1).contiguous(),
        "esc": c.esc.reshape(-1).contiguous(),
        "sbbase": _int32_words(c.sbbase),
    }


def container_entry_fields(c) -> Dict[str, Any]:
    bs = c.byte_size()
    return {
        "coded_shape": [int(c.shape[0]), int(c.shape[1])],
        "layout": str(c.layout),
        "mode": int(c.mode),
        "base": int(c.base),
        "tiles": int(c.tiles),
        "escapes": int(c.escapes),
        "superblock": int(c.superblock),
        "byte_size": bs,
        "resident_bytes": int(bs["total"]),
        "escape_rate_pct": 100.0 * float(c.escapes) / max(1, c.numel),
        "bits_per_element": float(bs["total"]) * 8.0 / max(1, c.numel),
    }


def encode_certified(weight, *, layout: str = "mma16"):
    """Encode on the CPU and prove the round trip bit-exact, or raise."""
    import torch
    from glc_loader.tbe_container import (
        choose_window, encode_tbe, exponent_histogram, tbe_certify,
    )

    with torch.device("cpu"):
        mode, base, _ = choose_window(exponent_histogram(weight))
        c = encode_tbe(weight, layout=layout, mode=mode, base=base)
        check = tbe_certify(c, weight)
    if not check["bitwise_ok"]:
        raise BundleError(
            "not_bit_exact",
            f"{check['mismatched_elements']} element(s) differ after round trip",
        )
    return c


def servable_linear(name: str, shape, dtype_str: str, *, min_numel: int,
                    code_lm_head: bool, code_embed: bool) -> Tuple[bool, str]:
    """``(codable, reason)`` under the SERVED rule (fragment kernel shapes)."""
    lname = name.lower()
    comp = component_of(name)
    if any(s in lname for s in _SKIP_NAME_SUBSTRINGS):
        if not ((comp == "lm_head" and code_lm_head) or (comp == "embed" and code_embed)):
            return False, "non_linear_name"
    if "bfloat16" not in dtype_str and dtype_str not in ("BF16",):
        return False, "not_bf16"
    if len(shape) != 2:
        return False, "non_2d"
    if not name.endswith(".weight"):
        return False, "not_a_weight_tensor"
    out_f, in_f = int(shape[0]), int(shape[1])
    if out_f * in_f < int(min_numel):
        return False, "below_min_numel"
    if in_f % TILE or out_f % MMA_N:
        return False, "shape_ineligible_for_mma"
    return True, "eligible"


def _base_entry(name: str, t, blake: str, reason: str) -> Dict[str, Any]:
    return {
        "name": name,
        "shape": [int(d) for d in t.shape],
        "dtype": str(t.dtype),
        "original_bytes": int(t.numel()) * int(t.element_size()),
        "blake2b_source": blake,
        "reason": reason,
        "component": component_of(name),
        "layer": layer_of(name),
    }


# ---------------------------------------------------------------------------
# from an existing container
# ---------------------------------------------------------------------------
def pack_from_container(
    container_dir: os.PathLike | str,
    out_dir: os.PathLike | str,
    *,
    config_dir: Optional[os.PathLike | str] = None,
    code_lm_head: bool = False,
    code_embed: bool = False,
    verify_raw: bool = True,
    verify_decode: bool = False,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    log=print,
) -> Dict[str, Any]:
    import torch
    from safetensors import safe_open

    from glc_loader.tbe_container import decode_tbe

    from .bundle import read_entry

    src = Path(container_dir)
    cmanifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    if cmanifest.get("status") != "ok":
        raise BundleError("container_not_ok", str(cmanifest.get("status")))
    entries = list(cmanifest["tensors"])
    cfg_dir = Path(config_dir) if config_dir else src
    if not (cfg_dir / "config.json").is_file():
        raise BundleError(
            "missing_config",
            f"{cfg_dir}/config.json absent; a v1 container has no config -- pass "
            "--config-dir <source checkpoint dir>",
        )
    writer = BundleWriter(out_dir, target_shard_bytes=target_shard_bytes)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        groups.setdefault(group_of(e["name"]), []).append(e)
    t0 = time.perf_counter()
    stats = {"n_copied_tbe": 0, "n_raw": 0, "n_newly_coded": 0,
             "n_raw_verified": 0, "n_decode_verified": 0}
    tbe_blob = src / "tbe_tensors.safetensors"
    raw_blob = src / "raw_tensors.safetensors"
    with safe_open(str(tbe_blob), framework="pt", device="cpu") as ht, \
            safe_open(str(raw_blob), framework="pt", device="cpu") as hr:
        for group in sorted(groups, key=group_sort_key):
            items = []
            for e in groups[group]:
                name = e["name"]
                comp = component_of(name)
                if e["kind"] == "tbe":
                    c = read_entry(ht, {**e, "coded_shape": e.get("coded_shape") or e["shape"]})
                    if verify_decode:
                        got = blake2b_tensor_hex(decode_tbe(c).reshape(e["shape"]))
                        if got != e["blake2b_source"]:
                            raise BundleError("stored_container_corrupt", name)
                        stats["n_decode_verified"] += 1
                    meta = {
                        "name": name, "kind": "tbe",
                        "shape": [int(d) for d in e["shape"]],
                        "dtype": e.get("dtype", "torch.bfloat16"),
                        "original_bytes": int(e["original_bytes"]),
                        "blake2b_source": e["blake2b_source"],
                        "reason": e.get("reason", "eligible"),
                        "component": comp, "layer": layer_of(name),
                        **container_entry_fields(c),
                    }
                    arrays = {f"{name}.{k}": v for k, v in container_arrays(c).items()}
                    items.append((meta, arrays))
                    stats["n_copied_tbe"] += 1
                    continue
                t = hr.get_tensor(name)
                if verify_raw:
                    got = blake2b_tensor_hex(t)
                    if got != e["blake2b_source"]:
                        raise BundleError(
                            "raw_tensor_hash_mismatch",
                            f"{name}: stored raw bytes do not match the container "
                            "manifest's blake2b_source",
                        )
                    stats["n_raw_verified"] += 1
                want_code = (
                    (comp == "lm_head" and code_lm_head) or (comp == "embed" and code_embed)
                ) and t.dtype == torch.bfloat16 and t.dim() == 2
                if want_code:
                    ok, _ = servable_linear(name, t.shape, str(t.dtype), min_numel=0,
                                            code_lm_head=code_lm_head, code_embed=code_embed)
                    if ok:
                        c = encode_certified(t)
                        meta = {**_base_entry(name, t, e["blake2b_source"], "eligible_head"),
                                "kind": "tbe", **container_entry_fields(c)}
                        arrays = {f"{name}.{k}": v for k, v in container_arrays(c).items()}
                        items.append((meta, arrays))
                        stats["n_newly_coded"] += 1
                        log(f"[pack] coded {name} {tuple(t.shape)} "
                            f"{meta['bits_per_element']:.3f} bpw, round trip bit-exact")
                        del t, c
                        continue
                meta = {**_base_entry(name, t, e["blake2b_source"], e.get("reason", "raw")),
                        "kind": "raw"}
                items.append((meta, {name: t}))
                stats["n_raw"] += 1
            writer.add_group(group, items)
            log(f"[pack] group {group}: {len(items)} tensors "
                f"({time.perf_counter() - t0:.1f}s)")
    sidecars = writer.copy_sidecars(cfg_dir)
    manifest = writer.finalize(sidecars=sidecars, meta={
        "source": {
            "kind": "tbe_container",
            "container_dir": str(src),
            "container_manifest_schema": cmanifest.get("schema"),
            "container_source": cmanifest.get("source"),
            "config_dir": str(cfg_dir),
        },
        "pack": {
            "tool": "glc_serve.pack", "mode": "from_container",
            "code_lm_head": bool(code_lm_head), "code_embed": bool(code_embed),
            "verify_raw": bool(verify_raw), "verify_decode": bool(verify_decode),
            "seconds": round(time.perf_counter() - t0, 2), **stats,
        },
        "container": {"layout": "mma16", "superblock": 32},
    })
    return manifest


# ---------------------------------------------------------------------------
# from a dense bf16 checkpoint
# ---------------------------------------------------------------------------
def _dense_index(model_dir: Path) -> List[Tuple[str, Path, List[int], str]]:
    """``(name, shard_path, shape, dtype)`` for every tensor, from headers only."""
    import struct

    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise BundleError("no_shards", str(model_dir))
    out = []
    seen = set()
    for p in shards:
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n).decode("utf-8"))
        for name, info in header.items():
            if name == "__metadata__" or name in seen:
                continue
            seen.add(name)
            out.append((name, p, list(info["shape"]), str(info["dtype"])))
    return out


def pack_from_dense(
    model_dir: os.PathLike | str,
    out_dir: os.PathLike | str,
    *,
    min_numel: int = DEFAULT_MIN_NUMEL,
    escape_band_pct: float = DEFAULT_ESCAPE_BAND_PCT,
    code_lm_head: bool = False,
    code_embed: bool = False,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    log=print,
) -> Dict[str, Any]:
    from safetensors import safe_open

    src = Path(model_dir)
    index = _dense_index(src)
    groups: Dict[str, List[Tuple[str, Path, List[int], str]]] = {}
    for item in index:
        groups.setdefault(group_of(item[0]), []).append(item)
    writer = BundleWriter(out_dir, target_shard_bytes=target_shard_bytes)
    handles: Dict[Path, Any] = {}
    over_band: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    n_coded = n_raw = 0
    try:
        for group in sorted(groups, key=group_sort_key):
            items = []
            for name, path, shape, dt in groups[group]:
                h = handles.get(path)
                if h is None:
                    h = safe_open(str(path), framework="pt", device="cpu")
                    h.__enter__()
                    handles[path] = h
                t = h.get_tensor(name)
                blake = blake2b_tensor_hex(t)
                ok, reason = servable_linear(name, shape, str(t.dtype), min_numel=min_numel,
                                             code_lm_head=code_lm_head, code_embed=code_embed)
                if ok:
                    c = encode_certified(t)
                    rate = 100.0 * float(c.escapes) / max(1, c.numel)
                    if rate > float(escape_band_pct):
                        over_band.append({"name": name, "escape_rate_pct": rate})
                        ok, reason = False, "escape_over_band"
                    else:
                        meta = {**_base_entry(name, t, blake, reason), "kind": "tbe",
                                **container_entry_fields(c)}
                        items.append((meta, {f"{name}.{k}": v
                                             for k, v in container_arrays(c).items()}))
                        n_coded += 1
                        continue
                meta = {**_base_entry(name, t, blake, reason), "kind": "raw"}
                items.append((meta, {name: t}))
                n_raw += 1
            writer.add_group(group, items)
            log(f"[pack] group {group}: {len(items)} tensors "
                f"({time.perf_counter() - t0:.1f}s)")
    finally:
        for h in handles.values():
            try:
                h.__exit__(None, None, None)
            except Exception:
                pass
    sidecars = writer.copy_sidecars(src)
    return writer.finalize(sidecars=sidecars, meta={
        "source": {"kind": "dense_checkpoint", "model_dir": str(src),
                   "n_tensors": len(index)},
        "pack": {
            "tool": "glc_serve.pack", "mode": "from_dense",
            "min_numel": int(min_numel), "escape_band_pct": float(escape_band_pct),
            "escape_over_band": over_band,
            "code_lm_head": bool(code_lm_head), "code_embed": bool(code_embed),
            "n_coded": n_coded, "n_raw": n_raw,
            "seconds": round(time.perf_counter() - t0, 2),
        },
        "container": {"layout": "mma16", "superblock": 32},
    })


def _parse(argv=None):
    ap = argparse.ArgumentParser(prog="glc_serve.pack", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-container", help="existing TBE container dir (v1 or v2)")
    src.add_argument("--from-dense", help="bf16 safetensors checkpoint dir (CPU transcode)")
    ap.add_argument("--config-dir", help="dir with config/tokenizer/processor files "
                    "(required for a v1 container)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--code-lm-head", action="store_true",
                    help="TBE-code lm_head (served through the fragment kernel)")
    ap.add_argument("--code-embed", action="store_true",
                    help="TBE-code embed_tokens (storage/transfer win only; decoded at load)")
    ap.add_argument("--target-shard-gib", type=float, default=2.0)
    ap.add_argument("--no-verify-raw", action="store_true")
    ap.add_argument("--verify-decode", action="store_true",
                    help="CPU-decode every stored container against blake2b_source (slow)")
    ap.add_argument("--min-numel", type=int, default=DEFAULT_MIN_NUMEL)
    ap.add_argument("--escape-band-pct", type=float, default=DEFAULT_ESCAPE_BAND_PCT)
    ap.add_argument("--threads", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse(argv)
    if args.threads:
        import torch
        torch.set_num_threads(int(args.threads))
    target = int(args.target_shard_gib * (1 << 30))
    try:
        if args.from_container:
            m = pack_from_container(
                args.from_container, args.out, config_dir=args.config_dir,
                code_lm_head=args.code_lm_head, code_embed=args.code_embed,
                verify_raw=not args.no_verify_raw, verify_decode=args.verify_decode,
                target_shard_bytes=target,
            )
        else:
            m = pack_from_dense(
                args.from_dense, args.out, min_numel=args.min_numel,
                escape_band_pct=args.escape_band_pct, code_lm_head=args.code_lm_head,
                code_embed=args.code_embed, target_shard_bytes=target,
            )
    except BundleError as exc:
        print(json.dumps({"status": "refused", "reason": exc.reason, "detail": exc.detail}))
        return 2
    print(json.dumps({
        "status": "ok", "out": str(args.out), "manifest_sha256": m.get("_sha256"),
        "n_shards": len(m["shards"]), "n_tensors": len(m["tensors"]),
        "accounting": m["accounting"]["total"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
