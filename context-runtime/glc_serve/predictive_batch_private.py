"""Opt-in B16 adapter: current predictive TBE-v1 startup weights into native Batcher.

The predictive loader remains the authority for model construction and conversion. This
module only transfers its FastDecoder descriptors into the batch-invariant decoder and
allocates independent per-slot KV/GDN state. It does not create or load a dense checkpoint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ._startup_limits import MAX_CPU_ENCODE_WORKERS


@dataclass(frozen=True)
class PredictiveBatchRuntime:
    """Predictive model/session references plus the native continuous batch engine."""

    session: Any
    model: Any
    tokenizer: Any
    processor: Any
    engine: Any
    fd: Any
    bd: Any
    batcher: Any
    metadata: Dict[str, Any]
    mm_frontend: Any = None


class _LoadedVisionTower:
    """Thin adapter over the already-loaded model vision module; never owns weights."""

    def __init__(self, visual: Any, *, source_kind: str):
        self.visual = visual
        self._bitexact = source_kind == "dense_bf16"
        self.source = f"{source_kind}:loaded-model"
        self.placement = "device"
        self.device_bytes = 0
        self.dense_parameter_bytes = sum(
            int(p.numel() * p.element_size()) for p in visual.parameters())
        contexts = [getattr(module, "context_weight") for module in visual.modules()
                    if getattr(module, "context_weight", None) is not None]
        self.context_wrapper_count = len(contexts)
        if contexts:
            from .predictive_tbe_private import _context_resident_bytes

            self.context_resident_bytes = int(_context_resident_bytes(contexts))
        else:
            self.context_resident_bytes = 0
        self.host_bytes = 0
        self.encodes = 0
        self.last_encode: Dict[str, Any] = {}

    @property
    def bitexact(self) -> bool:
        return self._bitexact

    def encode(self, pixel_values: Any, image_grid_thw: Any, out_device: Any) -> Any:
        import time
        import torch
        from .bidec_mm import image_embeddings

        t0 = time.perf_counter()
        with torch.no_grad():
            embeddings = image_embeddings(
                self.visual, pixel_values.to(out_device), image_grid_thw.to(out_device))
        self.encodes += 1
        self.last_encode = {
            "placement": "loaded-model", "source": self.source,
            "seconds": round(time.perf_counter() - t0, 4),
            "n_image_tokens": int(embeddings.shape[0]), "uploaded_bytes": 0,
        }
        return embeddings

    def stats(self) -> Dict[str, Any]:
        return {
            "vision_source": self.source,
            "dense_parameter_bytes": self.dense_parameter_bytes,
            "context_wrapper_count": self.context_wrapper_count,
            "context_resident_bytes_unique": self.context_resident_bytes,
            "resident_weight_bytes_accounted": (
                self.dense_parameter_bytes + self.context_resident_bytes),
            "new_weight_bytes": 0,
            "encodes": self.encodes,
        }


def _loaded_mm_frontend(session: Any, *, source_kind: str, device: str,
                        enable_images: bool) -> Any:
    """Build admission frontend from this session's processor and visual module only."""
    if not enable_images:
        return None
    from . import bidec_mm, fastdec

    engine = session.engine
    processor = getattr(engine, "processor", None)
    if processor is None or getattr(processor, "image_processor", None) is None:
        raise RuntimeError("image admission enabled but loaded session has no image processor")
    loaded = getattr(engine, "loaded", None)
    config = getattr(loaded, "config", None) or getattr(session.model, "config", None)
    if config is None or getattr(config, "vision_config", None) is None:
        raise RuntimeError("image admission enabled but loaded model has no vision config")
    visual = None
    model = session.model
    get_submodule = getattr(model, "get_submodule", None)
    if callable(get_submodule):
        for path in ("model.visual", "visual"):
            try:
                visual = get_submodule(path)
            except (AttributeError, KeyError):
                continue
            if visual is not None:
                break
    if visual is None:
        raise RuntimeError("image admission enabled but loaded model has no visual module")
    text = fastdec._text(model)
    rotary = getattr(text, "rotary_emb", None)
    if rotary is None:
        raise RuntimeError("image admission enabled but loaded text model has no rotary module")
    tower = _LoadedVisionTower(visual, source_kind=source_kind)
    frontend = bidec_mm.MMFrontend(
        processor=processor, tokenizer=session.tokenizer, config=config, tower=tower,
        rotary=rotary, device=device)
    return frontend


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _full_context_capacity(max_slots: int, pages_total: int, max_ctx: int) -> Dict[str, Any]:
    """Conservative full-slot reservation bound from Batcher.add's page formula.

    Batcher.add reserves ``(L + PAGE) // PAGE + 1`` pages for L prompt-plus-generation
    tokens. Its context check permits L <= ceil(max_ctx/PAGE)*PAGE - 1, so the exact worst
    reservation per slot is ``max_pages + 1``. For 16 slots at 8192 context this is 528 pages.
    An 8192-token prompt plus 512 generated tokens needs max_ctx >= 8705 (rounded to 8960),
    where 16 full slots reserve 576 pages. These bounds follow the actual admission formula.
    """
    from . import bidec

    max_pages = (int(max_ctx) + bidec.PAGE - 1) // bidec.PAGE
    per_slot = max_pages + 1
    required = int(max_slots) * per_slot
    return {
        "page_size": int(bidec.PAGE),
        "max_pages_per_slot": max_pages,
        "reserved_pages_per_full_context_slot": per_slot,
        "required_pages_for_all_slots": required,
        "configured_pages": int(pages_total),
        "full_slot_context_capacity": int(pages_total) >= required,
        "max_full_context_slots": int(pages_total) // per_slot,
    }


# bidec_kernels.cu dispatch constants. The current public FastDecoder's kernels pass
# unit_offset=1 for hidden norms and q/k norms; its attention prep always carries q|gate.
_NORM_UNIT_OFFSET = 1


def _install_fastdec_batch_helpers() -> None:
    """Add the exact descriptor-only helpers consumed by bidec if absent in fastdec."""
    from . import fastdec as fdm

    if not hasattr(fdm, "split_attn_gdn"):
        def split_attn_gdn(layers):
            rows = list(layers)
            attn = next((row for row in rows if row.kind == "attn"), None)
            if attn is None:
                raise ValueError("native B16 requires at least one full-attention layer")
            gdn = next((row for row in rows if row.kind == "gdn"), None)
            return attn, gdn
        fdm.split_attn_gdn = split_attn_gdn
    if not hasattr(fdm, "gdn_template_dims"):
        def gdn_template_dims(gdn):
            if gdn is None:
                return 0, 0, 0
            return int(gdn.C), int(gdn.qkvzba.N), int(gdn.vdim)
        fdm.gdn_template_dims = gdn_template_dims


def _install_bidec_fastdec_metadata(fd: Any) -> None:
    """Expose only layout metadata absent from the qualified single-session FastDecoder.

    No tensor arithmetic is changed. Every assumption below is checked against the source
    modules and the already-built FastDecoder descriptors; an unsupported layout fails closed.
    """
    from . import fastdec as fdm

    _install_fastdec_batch_helpers()
    text_model = fdm._text(fd.model)
    layers = list(text_model.layers)
    if len(layers) != len(fd.layers):
        raise ValueError("FastDecoder layer descriptors do not match source model layers")
    layouts = set()
    for source, desc in zip(layers, fd.layers):
        if desc.kind != "attn":
            continue
        sa = source.self_attn
        cfg = text_model.config
        nq = int(cfg.num_attention_heads)
        nkv = int(cfg.num_key_value_heads)
        dim = int(getattr(sa, "head_dim", 0) or int(cfg.hidden_size) // nq)
        if desc.D != dim or desc.NQ != nq or desc.NKV != nkv:
            raise ValueError("FastDecoder attention descriptor shape disagrees with source modules")
        if int(sa.q_proj.out_features) != 2 * nq * dim:
            raise ValueError("native B16 expects q_proj output ordered as q|gate")
        if int(sa.k_proj.out_features) != nkv * dim or int(sa.v_proj.out_features) != nkv * dim:
            raise ValueError("native B16 attention expects k/v projections of NKV*head_dim")
        if int(sa.o_proj.in_features) != nq * dim:
            raise ValueError("native B16 expects o_proj input width NQ*head_dim")
        sliding = getattr(sa, "sliding_window", None) or getattr(cfg, "sliding_window", None)
        if sliding and getattr(cfg, "use_sliding_window", True) is not False:
            raise ValueError("native B16 does not support sliding-window attention")
        for name, module in (("q_proj", sa.q_proj), ("k_proj", sa.k_proj),
                             ("v_proj", sa.v_proj), ("o_proj", sa.o_proj)):
            if getattr(module, "bias", None) is not None:
                raise ValueError(f"native B16 does not represent attention {name} bias")
        qnorm, knorm = getattr(sa, "q_norm", None), getattr(sa, "k_norm", None)
        if qnorm is None or knorm is None or int(qnorm.weight.numel()) != dim or int(knorm.weight.numel()) != dim:
            raise ValueError("current FastDecoder requires per-head q/k norm weights for native B16")
        if desc.qkv.N != 2 * nq * dim + 2 * nkv * dim:
            raise ValueError("FastDecoder fused q|gate|k|v descriptor has an unsupported width")
        if desc.qnw is None or desc.knw is None:
            raise ValueError("FastDecoder q/k norm descriptors are missing")
        if hasattr(desc, "gated") and int(desc.gated) != 1:
            raise ValueError("FastDecoder attention gate metadata disagrees with q|gate layout")
        if hasattr(desc, "qk_norm") and int(desc.qk_norm) != _NORM_UNIT_OFFSET:
            raise ValueError("FastDecoder q/k norm mode is not the unit-offset mode")
        desc.gated = 1
        desc.qk_norm = _NORM_UNIT_OFFSET
        layouts.add((dim, nq, nkv, 1, _NORM_UNIT_OFFSET))
    if not layouts:
        raise ValueError("native B16 requires at least one full-attention layer")
    if len(layouts) != 1:
        raise ValueError("native B16 requires one uniform full-attention layout")
    if not hasattr(fd, "norm_mode"):
        # Public fastdec_kernels.cu passes literal unit_offset=1 to every RMSNorm call.
        fd.norm_mode = _NORM_UNIT_OFFSET
    elif int(fd.norm_mode) != _NORM_UNIT_OFFSET:
        raise ValueError("FastDecoder hidden RMSNorm mode is not unit-offset")
    dim, nq, nkv, gated, qk_norm = next(iter(layouts))
    fd.D, fd.NQ, fd.NKV = dim, nq, nkv
    fd.attn_gated, fd.qk_norm = gated, qk_norm


def _validate_batch_settings(*, max_slots: int, max_rows: int, pages_total: int,
                            max_ctx: int, max_rows_step: Optional[int], prefill_chunk: int,
                            stride: Optional[int] = None,
                            loader_workers: Optional[int] = None,
                            prefill_priority: float = 0.0,
                            require_full_slot_capacity: bool = False) -> Dict[str, Any]:
    from . import bidec

    slots = _positive_int("max_slots", max_slots)
    rows = _positive_int("max_rows", max_rows)
    pages = _positive_int("pages_total", pages_total)
    ctx = _positive_int("max_ctx", max_ctx)
    chunk = _positive_int("prefill_chunk", prefill_chunk)
    row_step = rows if max_rows_step is None else _positive_int("max_rows_step", max_rows_step)
    if rows not in bidec.BUCKETS:
        raise ValueError("max_rows must be a supported native batch bucket")
    if row_step > rows:
        raise ValueError("max_rows_step cannot exceed max_rows")
    if stride is not None:
        _positive_int("stride", stride)
    if loader_workers is not None:
        _positive_int("loader_workers", loader_workers)
    priority = float(prefill_priority)
    if not math.isfinite(priority) or not 0.0 <= priority <= 1.0:
        raise ValueError("prefill_priority must be finite and in [0,1]")
    capacity = _full_context_capacity(slots, pages, ctx)
    if require_full_slot_capacity and not capacity["full_slot_context_capacity"]:
        raise ValueError(
            f"pages_total={pages} cannot reserve all {slots} slots at max_ctx={ctx}; "
            f"needs {capacity['required_pages_for_all_slots']} pages")
    return dict(max_slots=slots, max_rows=rows, pages_total=pages, max_ctx=ctx,
                max_rows_step=row_step, prefill_chunk=chunk, prefill_priority=priority,
                capacity=capacity)


def build_native_batch_from_fast_session(
    session: Any,
    *,
    source_kind: str,
    attention_implementation: str,
    settings: Dict[str, Any],
    device: str = "cuda:0",
    enable_images: bool = False,
    log: Callable[[str], None] = print,
) -> PredictiveBatchRuntime:
    """Shared session-to-B16 constructor used by predictive and dense twin factories."""
    from . import bidec

    fd = session.fd
    if bool(getattr(fd, "mtp_ready", False)):
        raise RuntimeError("private B16 runtime requires an MTP-off FastSession")
    _install_bidec_fastdec_metadata(fd)
    mm_frontend = _loaded_mm_frontend(
        session, source_kind=source_kind, device=device, enable_images=enable_images)
    mm_rope_rows = 0
    if mm_frontend is not None:
        from .bidec_mm import DEFAULT_MM_ROPE_ROWS

        mm_rope_rows = DEFAULT_MM_ROPE_ROWS
    # The source FastDecoder graphs are single-session captures. The B16 decoder captures its
    # own row shapes; retaining both graph pools wastes capacity and has no generation use here.
    single_graphs = getattr(fd, "graphs", None)
    if isinstance(single_graphs, dict):
        single_graphs.clear()

    bd = bidec.BatchDecoder(
        fd, max_slots=settings["max_slots"], max_rows=settings["max_rows"],
        pages_total=settings["pages_total"], max_ctx=settings["max_ctx"], R=1,
        gdn_ring_dtype="fp32", conv_ring=16, log=log,
        mm_rope_rows=mm_rope_rows,
    )
    capture_s = bd.capture()
    batcher = bidec.Batcher(
        bd, max_rows_step=settings["max_rows_step"],
        prefill_chunk=settings["prefill_chunk"], max_active=settings["max_slots"], spec_k=0,
        prefill_priority=settings["prefill_priority"],
        mm_encoder=mm_frontend.encode if mm_frontend is not None else None,
    )
    engine = session.engine
    tokenizer = getattr(session, "tokenizer", None) or getattr(engine, "tok", None)
    # FastSession exposes the multimodal processor here for a VLM. Native text
    # fixtures need its actual tokenizer; the image frontend retains the processor.
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if tokenizer is None:
        raise RuntimeError("FastSession did not expose a tokenizer")
    source_meta = dict(getattr(session, "meta", {}))
    metadata: Dict[str, Any] = {
        "engine": "glc_serve.bidec BatchDecoder/Batcher",
        "source_kind": str(source_kind),
        "weight_format": "tbe-v1-predictive-startup" if source_kind == "predictive_tbe" else "bf16-dense",
        "comparison_backend": "native-bi-gemm",
        "attention_implementation": str(attention_implementation),
        "execution_mode": "exact",
        "mtp_enabled": False,
        "enable_images": bool(enable_images),
        "image_frontend_source_kind": str(source_kind) if mm_frontend is not None else None,
        "image_visual_module_reused": mm_frontend is not None,
        "image_vision_bitexact": bool(mm_frontend.bitexact) if mm_frontend is not None else None,
        "image_vision_memory": (dict(mm_frontend.tower.stats())
                                if mm_frontend is not None else None),
        "image_vision_memory_components": (
            "dense_parameter_bytes sums the reused visual module's Parameters; "
            "context_resident_bytes_unique sums unique resident_bytes over Context weights "
            "and references; accounted resident total is their sum; new_weight_bytes is 0"
            if mm_frontend is not None else None),
        "tune_sha256": source_meta.get("tune_sha256"),
        "capture_s": capture_s,
        **{k: settings[k] for k in ("max_slots", "max_rows", "max_rows_step",
                                    "pages_total", "max_ctx", "prefill_chunk", "prefill_priority")},
        "capacity": dict(settings["capacity"]),
        "gdn_ring_depth": 1,
        "gdn_ring_dtype": "fp32",
        "gemm_tile": int(bd.gemm_tile),
        "row_tile": int(batcher.row_tile),
        "prefill_priority": float(batcher.prefill_priority),
        "spec_k": int(batcher.spec_k),
        "single_session_capture_released": isinstance(single_graphs, dict),
        "state_bytes": dict(bd.state_bytes),
        "streamed_bytes_per_step": int(bd.streamed_bytes()),
        "descriptors": fd.descriptor_kinds(),
        "gemm_backend_census": bd.gemm_backend_census(),
        "source_session": source_meta,
    }
    return PredictiveBatchRuntime(
        session=session,
        model=session.model,
        tokenizer=tokenizer,
        processor=getattr(engine, "processor", None),
        engine=engine,
        fd=fd,
        bd=bd,
        batcher=batcher,
        metadata=metadata,
        mm_frontend=mm_frontend,
    )


def load_predictive_batch_runtime(
    *,
    package: str | Path,
    tune: str | Path,
    device: str = "cuda:0",
    max_slots: int = 16,
    max_rows: int = 128,
    pages_total: int = 512,
    max_ctx: int = 8192,
    max_rows_step: Optional[int] = None,
    prefill_chunk: int = 256,
    require_full_slot_capacity: bool = False,
    prefill_priority: float = 0.0,
    stride: int = 1024,
    loader_workers: int = 1,
    metadata_cache_dir: str | Path | None = None,
    attention_implementation: str = "eager",
    enable_images: bool = False,
    log: Callable[[str], None] = print,
    verified_cpu_stream: bool = False,
    cpu_encode_workers: int = 1,
    source_inflight_bytes: int = 8 * 1024**3,
    cpu_vision_dense: bool = False,
) -> PredictiveBatchRuntime:
    """Current predictive TBE-v1 startup loader, then native B16 batch runtime.

    Existing predictive defaults are retained. For a matched dense comparison, call both
    factories with ``attention_implementation="sdpa"`` and identical batch settings.
    """
    if type(verified_cpu_stream) is not bool:
        raise ValueError("verified_cpu_stream must be a bool")
    if type(cpu_vision_dense) is not bool:
        raise ValueError("cpu_vision_dense must be a bool")
    if type(cpu_encode_workers) is not int or not 1 <= cpu_encode_workers <= MAX_CPU_ENCODE_WORKERS:
        raise ValueError(f"cpu_encode_workers must be an integer between 1 and {MAX_CPU_ENCODE_WORKERS}")
    if cpu_encode_workers > 1 and not verified_cpu_stream:
        raise ValueError("cpu_encode_workers > 1 requires verified_cpu_stream")
    if cpu_vision_dense and not verified_cpu_stream:
        raise ValueError("cpu_vision_dense requires verified_cpu_stream")
    if verified_cpu_stream and (
        type(source_inflight_bytes) is not int
        or not 0 < source_inflight_bytes <= 8 * 1024**3
    ):
        raise ValueError("source_inflight_bytes must be between 1 byte and 8 GiB")
    settings = _validate_batch_settings(
        max_slots=max_slots, max_rows=max_rows, pages_total=pages_total, max_ctx=max_ctx,
        max_rows_step=max_rows_step, prefill_chunk=prefill_chunk, stride=stride,
        loader_workers=loader_workers, prefill_priority=prefill_priority,
        require_full_slot_capacity=require_full_slot_capacity,
    )
    from .predictive_tbe_private import load_predictive_fast_session

    session_kwargs = dict(
        package=package, tune=tune, device=device, max_len=settings["max_ctx"], stride=stride,
        loader_workers=loader_workers, metadata_cache_dir=metadata_cache_dir,
        attention_implementation=attention_implementation, log=log,
    )
    if verified_cpu_stream or cpu_vision_dense:
        session_kwargs.update(
            verified_cpu_stream=verified_cpu_stream,
            cpu_encode_workers=cpu_encode_workers,
            source_inflight_bytes=source_inflight_bytes,
            cpu_vision_dense=cpu_vision_dense,
        )
    session = load_predictive_fast_session(**session_kwargs)
    return build_native_batch_from_fast_session(
        session, source_kind="predictive_tbe", attention_implementation=attention_implementation,
        settings=settings, device=device, enable_images=enable_images, log=log,
    )


def assert_native_batch_pair_compatible(predictive: Any, dense: Any) -> Dict[str, Any]:
    """Check saved runtime metadata before a same-engine comparison; never grades speed."""
    left = getattr(predictive, "metadata", predictive)
    right = getattr(dense, "metadata", dense)
    if left.get("source_kind") != "predictive_tbe" or right.get("source_kind") != "dense_bf16":
        raise ValueError("expected predictive_tbe then dense_bf16 runtime metadata")
    fields = (
        "comparison_backend", "attention_implementation", "execution_mode", "mtp_enabled",
        "enable_images", "image_visual_module_reused",
        "tune_sha256", "max_slots", "max_rows", "max_rows_step", "pages_total", "max_ctx",
        "prefill_chunk", "prefill_priority", "gdn_ring_depth", "gdn_ring_dtype",
        "gemm_tile", "row_tile", "spec_k",
    )
    mismatches = {key: (left.get(key), right.get(key)) for key in fields
                  if left.get(key) != right.get(key)}
    if mismatches:
        raise ValueError(f"native B16 twin settings differ: {mismatches}")
    if not left.get("tune_sha256"):
        raise ValueError("native B16 twin metadata lacks the shared tune SHA256")
    return {key: left.get(key) for key in fields}


def load_dense_batch_runtime(
    *,
    dense: str | Path,
    tune: str | Path,
    device: str = "cuda:0",
    max_slots: int = 16,
    max_rows: int = 128,
    pages_total: int = 512,
    max_ctx: int = 8192,
    max_rows_step: Optional[int] = None,
    prefill_chunk: int = 256,
    require_full_slot_capacity: bool = False,
    prefill_priority: float = 0.0,
    attention_implementation: str = "sdpa",
    enable_images: bool = False,
    log: Callable[[str], None] = print,
) -> PredictiveBatchRuntime:
    """Optional dense BF16 twin through ``FastSession.load(dense=...)`` and native B16.

    The current dense loader's established server path uses SDPA. Other attention modes are
    refused so this factory cannot label a mismatched prefill implementation as a twin.
    """
    if attention_implementation != "sdpa":
        raise ValueError("the current FastSession dense path is fixed to sdpa attention")
    settings = _validate_batch_settings(
        max_slots=max_slots, max_rows=max_rows, pages_total=pages_total, max_ctx=max_ctx,
        max_rows_step=max_rows_step, prefill_chunk=prefill_chunk,
        prefill_priority=prefill_priority,
        require_full_slot_capacity=require_full_slot_capacity,
    )
    from .fastserve import FastSession

    session = FastSession.load(
        dense=str(dense), tune=str(tune), device=device, max_len=settings["max_ctx"],
        server_flags=("--no-mtp", "--exec-mode", "exact"), log=log,
    )
    loaded = getattr(session.engine, "loaded", None)
    actual_attention = getattr(getattr(loaded, "options", None), "attn_implementation", None)
    if actual_attention != attention_implementation:
        raise RuntimeError(
            f"dense FastSession attention is {actual_attention!r}; expected {attention_implementation!r}")
    return build_native_batch_from_fast_session(
        session, source_kind="dense_bf16", attention_implementation=attention_implementation,
        settings=settings, device=device, enable_images=enable_images, log=log,
    )



__all__ = ["PredictiveBatchRuntime", "assert_native_batch_pair_compatible",
           "build_native_batch_from_fast_session", "load_dense_batch_runtime",
           "load_predictive_batch_runtime"]
