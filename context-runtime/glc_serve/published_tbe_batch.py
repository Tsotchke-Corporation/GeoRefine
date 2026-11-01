"""Public published-TBE ``georefine.tbe.serve.v1`` adapter for native B16 batching.

This loader consumes the published standalone TBE bundle through FastSession's
public bundle path. It does not read or transcode the Context BCTX/PPCX package.
"""
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional


def validate_published_tbe_bundle(bundle: str | Path, expected_manifest_sha256: str) -> dict[str, Any]:
    """CPU-preflight a local published TBE bundle before importing FastSession/Torch."""
    location = str(bundle)
    if location.startswith(("gs://", "http://", "https://")):
        raise ValueError("published TBE batch input must be a pre-downloaded local bundle directory")
    if not isinstance(expected_manifest_sha256, str) or re.fullmatch(
            r"[0-9a-fA-F]{64}", expected_manifest_sha256) is None:
        raise ValueError("expected_manifest_sha256 must be a 64-character SHA256")
    from .bundle import open_bundle

    opened = open_bundle(location, expected_manifest_sha256=expected_manifest_sha256)
    if opened.manifest.get("format") != "georefine.tbe.serve.v1":
        raise ValueError(f"published TBE manifest has unsupported format: {opened.manifest.get('format')!r}")
    if opened.manifest_sha256 != expected_manifest_sha256.lower():
        raise ValueError("published TBE manifest SHA256 mismatch")
    return {"bundle": Path(location).resolve(),
            "manifest_sha256": opened.manifest_sha256,
            "format": opened.manifest["format"],
            "schema": opened.manifest.get("schema"),
            "tensor_count": len(opened.manifest["tensors"]),
            "shard_count": len(opened.manifest["shards"])}


def load_published_tbe_batch_runtime(
    *, bundle: str | Path, expected_manifest_sha256: str, tune: str | Path,
    device: str = "cuda:0", max_slots: int = 16, max_rows: int = 128,
    pages_total: int = 512, max_ctx: int = 8192,
    max_rows_step: Optional[int] = None, prefill_chunk: int = 256,
    require_full_slot_capacity: bool = False, prefill_priority: float = 0.0,
    attention_implementation: str = "sdpa", enable_images: bool = False,
    log: Callable[[str], None] = print,
    vram_headroom_bytes: int = 6 * 1024**3,
    mm_options: Optional[Dict[str, int]] = None,
) -> Any:
    """Load published TBE weights into FastSession, then the same native B16 engine.

    The manifest and sidecars are checked before FastSession imports/initializes CUDA.
    Runtime settings intentionally mirror :func:`predictive_batch.load_dense_batch_runtime`.
    """
    pinned = validate_published_tbe_bundle(bundle, expected_manifest_sha256)
    if attention_implementation != "sdpa":
        raise ValueError("the native TBE comparison path requires SDPA attention")
    from . import predictive_batch
    from .fastserve import FastSession

    session = FastSession.load(
        bundle=str(pinned["bundle"]), tune=str(tune), device=device,
        max_len=int(max_ctx), history_slots=16,
        server_flags=("--no-mtp", "--exec-mode", "exact"), log=log)
    if getattr(session, "mtp", None) is not None or bool(getattr(session.fd, "mtp_ready", False)):
        raise RuntimeError("published TBE FastSession unexpectedly loaded MTP")
    loaded = getattr(getattr(session, "engine", None), "loaded", None)
    actual_bundle = getattr(loaded, "bundle", None)
    if (actual_bundle is None or
            actual_bundle.manifest_sha256 != pinned["manifest_sha256"] or
            actual_bundle.manifest.get("format") != "georefine.tbe.serve.v1"):
        raise RuntimeError("loaded FastSession did not retain the preflighted published TBE bundle")
    options = getattr(loaded, "options", None)
    if (getattr(options, "backend", None) != "tbe" or
            getattr(options, "exec_mode", None) != "exact" or
            getattr(options, "load_mtp", None) is not False):
        raise RuntimeError("loaded FastSession options differ from published TBE MTP-off exact path")

    settings = predictive_batch._validate_batch_settings(
        max_slots=max_slots, max_rows=max_rows, pages_total=pages_total,
        max_ctx=max_ctx, max_rows_step=max_rows_step, prefill_chunk=prefill_chunk,
        prefill_priority=prefill_priority, require_full_slot_capacity=require_full_slot_capacity,
        allow_auto_pages=True)
    settings, page_plan = predictive_batch.resolve_pages_and_settings(
        session, settings, device=device, enable_images=enable_images,
        headroom_bytes=vram_headroom_bytes,
        require_full_slot_capacity=require_full_slot_capacity, log=log)
    runtime = predictive_batch.build_native_batch_from_fast_session(
        session, source_kind="published_tbe", attention_implementation=attention_implementation,
        settings=settings, device=device, enable_images=enable_images, log=log,
        mm_options=mm_options, page_plan=page_plan)
    if int(runtime.metadata.get("descriptors", {}).get("tbe", 0)) <= 0:
        raise RuntimeError("published TBE runtime has no native TBE descriptors")
    if enable_images:
        # The published server's image path deliberately decodes its bundle vision tensors into
        # exact BF16 HF VisionTower weights. TBEServeLinear's fused vision arithmetic is a
        # different operation schedule. The text trunk stays TBE; images use the established
        # bit-exact native frontend with on-demand residency, matching bidec_load's default.
        from . import bidec_mm, fastdec
        frontend = bidec_mm.build_frontend(
            config=loaded.config, local_dir=loaded.local_dir, placement="ondemand",
            device=device, bundle=actual_bundle, parent=None,
            rotary=fastdec._text(session.model).rotary_emb, tokenizer=session.tokenizer,
            attn_implementation=attention_implementation, log=log, **(mm_options or {}))
        if frontend is None or not frontend.bitexact:
            raise RuntimeError("published TBE image frontend is absent or not bit-exact")
        runtime = replace(runtime, mm_frontend=frontend, processor=frontend.processor)
        runtime.batcher.mm_encoder = frontend.encode
        runtime.metadata["image_vision_bitexact"] = True
        runtime.metadata["image_visual_module_reused"] = False
        runtime.metadata["image_vision_memory"] = dict(frontend.tower.stats())
        runtime.metadata["image_vision_memory_components"] = (
            "BF16 published-bundle vision tensors decoded and source-verified by bidec_mm; "
            "on-demand upload/release residency; image vision arithmetic follows the exact "
            "HF BF16 module path.")
    runtime.metadata["source_kind"] = "published_tbe"
    runtime.metadata["weight_format"] = "published-tbe"
    runtime.metadata["published_tbe_manifest_sha256"] = pinned["manifest_sha256"]
    runtime.metadata["published_tbe_manifest_format"] = pinned["format"]
    return runtime


__all__ = ["load_published_tbe_batch_runtime", "validate_published_tbe_bundle"]
