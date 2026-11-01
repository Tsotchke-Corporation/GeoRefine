"""Opt-in B16 adapter: current predictive TBE-v1 startup weights into native Batcher.

The predictive loader remains the authority for model construction and conversion. This
module only transfers its FastDecoder descriptors into the batch-invariant decoder and
allocates independent per-slot KV/GDN state. It does not create or load a dense checkpoint.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

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

    def __init__(self, visual: Any, *, source_kind: str, vision_source: Optional[str] = None):
        self.visual = visual
        self._bitexact = source_kind == "dense_bf16" or vision_source == "tbe21_package"
        self.source = (f"{source_kind}:{vision_source}" if vision_source else
                       f"{source_kind}:loaded-model")
        self.placement = "device"
        self.device_bytes = 0
        self.dense_parameter_bytes = sum(
            int(p.numel() * p.element_size()) for p in visual.parameters())
        contexts = [getattr(module, "context_weight") for module in visual.modules()
                    if getattr(module, "context_weight", None) is not None]
        self.context_wrapper_count = len(contexts)
        if contexts:
            from .predictive_tbe import _context_resident_bytes

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


#: ``bidec_mm.MMFrontend`` request limits a server may set.
MM_OPTION_KEYS = ("max_images", "max_image_bytes", "max_image_tokens")
TBE21_SOURCE_KINDS = frozenset(("context_v21_transcode", "typed_context_tbe21", "tbe21_fullpack"))


def _loaded_mm_frontend(session: Any, *, source_kind: str, device: str,
                        enable_images: bool, mm_options: Optional[Dict[str, int]] = None,
                        vision_source: Optional[str] = None) -> Any:
    """Build admission frontend from this session's processor and visual module only.

    ``mm_options`` may set ``max_images``, ``max_image_bytes`` and ``max_image_tokens``
    (``bidec_mm.MMFrontend`` keywords); http(s) image fetching is never enabled here.
    """
    if not enable_images:
        return None
    if source_kind in TBE21_SOURCE_KINDS and vision_source != "tbe21_package":
        raise RuntimeError("TBE21 image admission requires the verified package vision tower")
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
    tower = (_LoadedVisionTower(visual, source_kind=source_kind) if vision_source is None else
             _LoadedVisionTower(visual, source_kind=source_kind, vision_source=vision_source))
    options = dict(mm_options or {})
    unknown = sorted(set(options) - set(MM_OPTION_KEYS))
    if unknown:
        raise ValueError(f"unsupported image frontend option(s): {', '.join(unknown)}")
    for key, value in options.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"image frontend option {key} must be a non-negative integer")
    frontend = bidec_mm.MMFrontend(
        processor=processor, tokenizer=session.tokenizer, config=config, tower=tower,
        rotary=rotary, device=device, fetch_urls=False, **options)
    return frontend


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _full_context_capacity(max_slots: int, pages_total: int, max_ctx: int) -> Dict[str, Any]:
    """Conservative full-slot reservation bound from Batcher.add's page formula.

    Batcher.add reserves ``ceil(L / PAGE) + 1`` pages for L prompt-plus-generation
    tokens and permits L <= max_ctx, including the declared context boundary.
    Thus the worst reservation per slot is ``max_pages + 1``. For 16 slots at
    8192 context this is 528 pages. An 8192-token prompt plus 512 generated tokens
    needs max_ctx >= 8704, where 16 full slots reserve 560 pages.
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


def adapt_fastdecoder_for_batch(fd: Any) -> Dict[str, int]:
    """Explicit adapter step: validate one FastDecoder's layout and record it for bidec.

    ``bidec.BatchDecoder`` reads attention-layout attributes (``D``, ``NQ``, ``NKV``,
    ``attn_gated``, ``qk_norm``, ``norm_mode`` and per-descriptor ``gated``/``qk_norm``)
    that the single-session FastDecoder does not carry. This step checks every assumption
    against the source modules and the already-built descriptors, fails closed on an
    unsupported layout, and only then sets those attributes on this FastDecoder instance and
    its attention descriptors. No module is modified and no tensor arithmetic changes.
    Returns the validated layout.
    """
    from . import fastdec as fdm

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
    return {"head_dim": dim, "num_q_heads": nq, "num_kv_heads": nkv, "attn_gated": gated,
            "qk_norm": qk_norm, "norm_mode": int(fd.norm_mode)}


def _validate_batch_settings(*, max_slots: int, max_rows: int, pages_total: int,
                            max_ctx: int, max_rows_step: Optional[int], prefill_chunk: int,
                            stride: Optional[int] = None,
                            loader_workers: Optional[int] = None,
                            prefill_priority: float = 0.0,
                            require_full_slot_capacity: bool = False,
                            allow_auto_pages: bool = False, spec_k: int = 0,
                            spec_ngram: bool = False, max_decode_rows: int = 64,
                            mtp_confirm_kv_only: bool = False) -> Dict[str, Any]:
    """Validate batch knobs. ``allow_auto_pages`` accepts ``pages_total=0`` (size it later
    with :func:`plan_pages_total`); the returned ``capacity`` is then ``None``."""
    from . import bidec

    slots = _positive_int("max_slots", max_slots)
    rows = _positive_int("max_rows", max_rows)
    auto_pages = bool(allow_auto_pages) and not isinstance(pages_total, bool) and pages_total == 0
    pages = 0 if auto_pages else _positive_int("pages_total", pages_total)
    ctx = _positive_int("max_ctx", max_ctx)
    chunk = _positive_int("prefill_chunk", prefill_chunk)
    row_step = rows if max_rows_step is None else _positive_int("max_rows_step", max_rows_step)
    if rows not in bidec.BUCKETS:
        raise ValueError(f"max_rows must be a supported native batch bucket {bidec.BUCKETS}")
    if row_step > rows:
        raise ValueError("max_rows_step cannot exceed max_rows")
    if type(spec_k) is not int or not 0 <= spec_k <= 7:
        raise ValueError("spec_k must be an integer in [0,7]")
    if type(spec_ngram) is not bool or (spec_ngram and not spec_k):
        raise ValueError("spec_ngram requires a positive spec_k")
    if type(mtp_confirm_kv_only) is not bool or (mtp_confirm_kv_only and (not spec_k or spec_ngram)):
        raise ValueError("mtp_confirm_kv_only requires a positive MTP spec_k")
    decode_rows = _positive_int("max_decode_rows", max_decode_rows)
    if spec_k and (decode_rows > row_step or spec_k + 1 > row_step):
        raise ValueError("speculation and decode-row budget must fit max_rows_step")
    if stride is not None:
        _positive_int("stride", stride)
    if loader_workers is not None:
        _positive_int("loader_workers", loader_workers)
    priority = float(prefill_priority)
    if not math.isfinite(priority) or not 0.0 <= priority <= 1.0:
        raise ValueError("prefill_priority must be finite and in [0,1]")
    capacity = None if auto_pages else _full_context_capacity(slots, pages, ctx)
    if (capacity is not None and require_full_slot_capacity
            and not capacity["full_slot_context_capacity"]):
        raise ValueError(
            f"pages_total={pages} cannot reserve all {slots} slots at max_ctx={ctx}; "
            f"needs {capacity['required_pages_for_all_slots']} pages")
    result = dict(max_slots=slots, max_rows=rows, pages_total=pages, max_ctx=ctx,
                max_rows_step=row_step, prefill_chunk=chunk, prefill_priority=priority,
                capacity=capacity)
    if spec_k or decode_rows != 64:
        result.update(spec_k=spec_k, spec_ngram=spec_ngram, max_decode_rows=decode_rows)
    if mtp_confirm_kv_only:
        result['mtp_confirm_kv_only'] = True
    return result


# ----------------------------------------------------------------------- KV page auto-sizing
#: FastDecoder single-stream state that ``bidec.BatchDecoder.__init__`` releases before it
#: allocates (``fd.kc = fd.vc = fd.hist = fd.state = None`` plus ``FD_DEAD_BUFFERS``).
_FD_SINGLE_STREAM_STATE = ("kc", "vc", "hist", "state")


def release_single_stream_buffers(fd: Any) -> int:
    """Release the FastDecoder single-session buffers and graphs the batch engine never uses.

    ``BatchDecoder.__init__`` releases exactly these attributes itself; doing it before the
    page pool is sized lets ``torch.cuda.mem_get_info`` see the memory they held. Returns
    the bytes of the released tensors (counted once per tensor object).
    """
    from . import bidec

    seen = set()
    released = 0
    for name in _FD_SINGLE_STREAM_STATE + tuple(bidec.BatchDecoder.FD_DEAD_BUFFERS):
        tensor = getattr(fd, name, None)
        if tensor is None:
            continue
        if hasattr(tensor, "numel") and id(tensor) not in seen:
            seen.add(id(tensor))
            released += int(tensor.numel()) * int(tensor.element_size())
        setattr(fd, name, None)
    graphs = getattr(fd, "graphs", None)
    if isinstance(graphs, dict):
        graphs.clear()
    return released


def batch_engine_shapes(fd: Any):
    """``state_budget.Shapes`` of the batch engine built over ``fd`` (after the adapter).

    Read off the FastDecoder descriptors exactly where ``BatchDecoder.__init__`` reads them:
    ``fd.H``/``fd.V``, ``fd.layers[0].F``, ``fd.n_att``/``fd.n_gdn``, the adapted
    ``fd.NQ``/``fd.NKV``/``fd.D``, ``fastdec.gdn_template_dims`` and ``fd.rope_dim``.
    """
    from . import fastdec as fdm
    from . import state_budget as sb

    _att0, gd0 = fdm.split_attn_gdn(fd.layers)
    conv_c, gdn_n, gdn_v = fdm.gdn_template_dims(gd0)
    _hk, gdn_hv, gdn_dk, gdn_dv, _conv_kernel, _ = fdm.gdn_template_geometry(gd0)
    rope_dim = getattr(fd, "rope_dim", None)
    if rope_dim is None:
        rope_dim = int(fd.cosT.shape[1])
    return sb.Shapes(
        name="loaded", H=int(fd.H), V=int(fd.V), F=int(fd.layers[0].F),
        n_att=int(fd.n_att), n_gdn=int(fd.n_gdn), NQ=int(fd.NQ), NKV=int(fd.NKV),
        D=int(fd.D), gdn_hv=int(gdn_hv or 48), gdn_dk=int(gdn_dk or 128),
        gdn_dv=int(gdn_dv or 128), conv_C=int(conv_c), gdn_N=int(gdn_n), gdn_vdim=int(gdn_v),
        rope_dim=int(rope_dim), gated=bool(getattr(fd, "attn_gated", 1)),
        qk_norm=int(getattr(fd, "qk_norm", 1)), norm_mode=int(getattr(fd, "norm_mode", 1)))


def _ledger_bytes(shapes: Any, *, slots: int, pages: int, max_ctx: int, max_rows: int,
                  conv_ring: int, sms: int, spec_k: int = 0,
                  mtp_enabled: Optional[bool] = None,
                  item_cap: Optional[int] = None) -> int:
    from . import state_budget as sb

    cfg = sb.EngineConfig(slots=slots, pages=pages, max_ctx=max_ctx, spec_k=spec_k,
                          max_rows=max_rows, conv_ring=conv_ring, dummy_state=False,
                          fd_buffers_live=False, sms=sms, mtp_enabled=mtp_enabled,
                          item_cap=item_cap)
    return int(sb.summarize(sb.ledger(shapes, cfg))["total_bytes"])


def _prefill_item_cap(pages_total: int, max_rows: int, max_ctx: int,
                      prefill_priority: float) -> Optional[int]:
    if prefill_priority <= 0:
        return None
    from .bidec_policy import plan_item_cap

    return plan_item_cap(pages_total, max_rows, max_ctx,
                         prefill_priority=prefill_priority)["item_cap"]


def _device_memory_state(device, mem_get_info=None, torch_cached_bytes=None, sms=None, *, strict=False):
    if mem_get_info is None or torch_cached_bytes is None or sms is None:
        import gc
        import torch

        dev = torch.device(device)
        if dev.type != "cuda":
            raise ValueError("automatic page sizing needs a CUDA device; pass pages_total")
        if mem_get_info is None:
            gc.collect()
            torch.cuda.synchronize(dev)
            torch.cuda.empty_cache()
            mem_get_info = lambda: torch.cuda.mem_get_info(dev)  # noqa: E731
        if torch_cached_bytes is None:
            reserved = torch.cuda.memory_reserved(dev)
            allocated = torch.cuda.memory_allocated(dev)
            if strict:
                if (type(reserved) is not int or reserved < 0 or type(allocated) is not int
                        or allocated < 0 or reserved < allocated):
                    raise ValueError("invalid CUDA allocator reserved/allocated byte counters")
                torch_cached_bytes = reserved - allocated
            else:
                torch_cached_bytes = max(0, int(reserved) - int(allocated))
        if sms is None:
            sms = int(torch.cuda.get_device_properties(dev).multi_processor_count)
    free_bytes, total_bytes = mem_get_info()
    if strict:
        values = (("free_bytes", free_bytes, 0), ("total_bytes", total_bytes, 1),
                  ("torch_cached_bytes", torch_cached_bytes, 0), ("sms", sms, 1))
        for name, value, minimum in values:
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be a {'positive' if minimum else 'nonnegative'} integer")
        if free_bytes > total_bytes:
            raise ValueError("free_bytes exceeds total_bytes")
        return free_bytes, total_bytes, torch_cached_bytes, sms
    return int(free_bytes), int(total_bytes), int(torch_cached_bytes), int(sms)


def plan_pages_total(fd: Any, *, max_slots: int, max_rows: int, max_ctx: int,
                     headroom_bytes: int, device: str = "cuda:0",
                     mm_rope_rows: int = 0, conv_ring: int = 16,
                     mem_get_info: Optional[Callable[[], Any]] = None,
                     torch_cached_bytes: Optional[int] = None,
                     sms: Optional[int] = None, spec_k: int = 0,
                     mtp_enabled: Optional[bool] = None,
                     prefill_priority: float = 0.0) -> Dict[str, Any]:
    """Size the KV page pool from the memory left after the weights are resident.

    ``available = free (cudaMemGetInfo) + PyTorch cached-but-unallocated bytes``. The engine's
    bytes are ``state_budget.ledger`` (``BatchDecoder.__init__`` allocations at this
    ``max_slots``/``max_rows``/``max_ctx``, R=1, ``conv_ring`` 16, no dummy-slot state), which
    is linear in ``pages``: ``engine(P) = fixed + P * per_page`` where ``per_page`` is one bf16
    K+V page over all attention layers plus one attention work-list entry (``g_cap`` grows by
    one per page). Image admission adds ``mm_rope_rows`` rotary scratch rows and the
    ``max_rows x H`` staging rows. Then

        pages_total = min(floor((available - headroom - fixed - mm_extra) / per_page),
                          max_slots * (ceil(max_ctx / 256) + 1))

    The upper bound is what every slot needs at full context (``Batcher.add`` reserves
    ``(L + 256) // 256 + 1`` pages, at most ``max_pages + 1``). Fewer pages than one
    ``max_ctx`` request needs is refused with a ValueError. ``headroom_bytes`` covers what the
    ledger does not model: CUDA-graph pools, allocator fragmentation, vision activations.
    Call after :func:`release_single_stream_buffers` so freed single-session state is counted.
    Positive prefill priority reserves its work list at the all-slots page
    ceiling before page admission. Its buffers are charged to fixed state,
    rather than borrowing the KV pool's headroom.
    """
    from . import bidec

    slots = _positive_int("max_slots", max_slots)
    ctx = _positive_int("max_ctx", max_ctx)
    if isinstance(headroom_bytes, bool) or int(headroom_bytes) < 0:
        raise ValueError("headroom_bytes must be a non-negative integer")
    headroom = int(headroom_bytes)
    rows = max(b for b in bidec.BUCKETS if b <= int(max_rows))
    free_bytes, total_bytes, torch_cached_bytes, sms = _device_memory_state(
        device, mem_get_info, torch_cached_bytes, sms)
    available = free_bytes + int(torch_cached_bytes)
    shapes = batch_engine_shapes(fd)
    max_pages = (ctx + bidec.PAGE - 1) // bidec.PAGE
    one_request = max_pages + 1
    all_slots = slots * one_request
    kw = dict(slots=slots, max_ctx=ctx, max_rows=rows, conv_ring=int(conv_ring), sms=int(sms),
              spec_k=spec_k, mtp_enabled=mtp_enabled)
    # Reserve the largest selected work list before sizing pages. Its cap is
    # constant over the admitted page range, keeping the page formula linear
    # and conservative even for fractional prefill priorities.
    item_cap = _prefill_item_cap(all_slots, rows, ctx, prefill_priority)
    if item_cap is not None:
        kw["item_cap"] = item_cap
    fixed = _ledger_bytes(shapes, pages=0, **kw)
    per_page = _ledger_bytes(shapes, pages=1, **kw) - fixed
    mm_extra = 0
    if int(mm_rope_rows) > 0:
        mm_extra = int(mm_rope_rows) * shapes.rope_dim * 2 * 2 + rows * shapes.H * 2
    bhe_extra = 0
    if callable(getattr(fd, '_descs', None)):
        from .bhe_desc import extra_workspace_bytes
        bhe_extra = extra_workspace_bytes(fd, rows)
    budget = available - headroom - fixed - mm_extra - bhe_extra
    fit = max(0, budget // per_page) if per_page > 0 else 0
    plan: Dict[str, Any] = {
        "mode": "auto",
        "formula": ("min(floor((free + torch_cached - headroom - fixed - mm_extra) / per_page), "
                    "max_slots * (ceil(max_ctx/256) + 1))"),
        "free_bytes": free_bytes, "device_total_bytes": total_bytes,
        "torch_cached_bytes": int(torch_cached_bytes), "available_bytes": available,
        "headroom_bytes": headroom, "fixed_engine_bytes": fixed,
        "per_slot_state_bytes": (_ledger_bytes(shapes, **{**kw, "slots": slots + 1}, pages=0)
                                 - fixed),
        "per_page_bytes": per_page, "mm_extra_bytes": mm_extra,
        "pages_fit": int(fit), "pages_one_max_context_request": one_request,
        "pages_all_slots_full_context": all_slots,
        "max_slots": slots, "max_rows": rows, "max_ctx": ctx, "sms": int(sms),
        "shapes": {k: getattr(shapes, k) for k in ("H", "V", "F", "n_att", "n_gdn", "NQ",
                                                   "NKV", "D", "conv_C", "gdn_N", "gdn_vdim",
                                                   "rope_dim")},
        "unmodelled": "CUDA-graph pools, allocator fragmentation, vision activations "
                      "(covered by headroom_bytes)",
    }
    if bhe_extra:
        plan['bhe_workspace_bytes'] = bhe_extra
        plan['formula'] = plan['formula'].replace('fixed - mm_extra', 'fixed - mm_extra - bhe_workspace')
    if item_cap is not None:
        plan['prefill_item_cap_reserved'] = item_cap
    if fit < one_request:
        per_slot = plan["per_slot_state_bytes"]
        gib = float(1 << 30)
        raise ValueError(
            f"not enough device memory for the KV page pool: {available / gib:.2f} GiB available "
            f"after weights, {headroom / gib:.2f} GiB headroom, {fixed / gib:.2f} GiB fixed "
            f"engine state for {slots} users ({per_slot / gib:.3f} GiB per user), "
            f"{per_page / (1 << 20):.2f} MiB per 256-token page -> {fit} pages fit, but one "
            f"max_context={ctx} request needs {one_request}. Lower --max-users, --max-context "
            f"or --vram-headroom-gib, or pass --pages-total explicitly.")
    plan["pages_total"] = int(min(fit, all_slots))
    plan["full_slot_context_capacity"] = plan["pages_total"] >= all_slots
    return plan


# ----------------------------------------------------------------------- session -> engine
def _progress(phase: str, **detail: Any) -> None:
    from . import startup_progress

    startup_progress.report(phase, **detail)


def _calibrate_spec_cost(decoder, settings):
    from .spec_calibration import calibrate_engine
    repeats = settings.get('spec_calibration_repeats')
    if repeats is not None and (type(repeats) is not int or not 3 <= repeats <= 20):
        raise ValueError('spec_calibration_repeats must be in 3..20')
    include_uncaptured = settings.get(
        'include_uncaptured', settings.get('capture_workload_only', False))
    if type(include_uncaptured) is not bool:
        raise ValueError('include_uncaptured must be a bool')
    kwargs = {} if repeats is None else {'repeats': repeats}
    if include_uncaptured:
        kwargs['include_uncaptured'] = True
    return calibrate_engine(decoder, **kwargs)


def build_native_batch_from_fast_session(
    session: Any,
    *,
    source_kind: str,
    attention_implementation: str,
    settings: Dict[str, Any],
    device: str = "cuda:0",
    enable_images: bool = False,
    log: Callable[[str], None] = print,
    mm_options: Optional[Dict[str, int]] = None,
    page_plan: Optional[Dict[str, Any]] = None,
    gdn_hv16: Optional[bool] = None,
    vision_source: Optional[str] = None,
) -> PredictiveBatchRuntime:
    """Shared session-to-B16 constructor used by predictive and dense twin factories."""
    import torch
    from . import bidec

    if gdn_hv16 is not None and type(gdn_hv16) is not bool:
        raise ValueError("gdn_hv16 must be a bool")
    fd = session.fd
    spec_k = int(settings.get("spec_k", 0))
    spec_ngram = bool(settings.get("spec_ngram", False))
    has_mtp = bool(getattr(fd, "mtp_ready", False))
    if has_mtp and not spec_k:
        raise RuntimeError("the multi-user batch runtime requires an MTP-off FastSession")
    if spec_k and not spec_ngram and not has_mtp:
        raise RuntimeError("MTP speculation requested but the loaded session has no MTP head")
    if settings.get("capacity") is None or int(settings.get("pages_total") or 0) <= 0:
        raise ValueError("settings need a resolved pages_total (see plan_pages_total)")
    layout = adapt_fastdecoder_for_batch(fd)
    mm_kwargs = {"mm_options": mm_options}
    if vision_source is not None:
        mm_kwargs["vision_source"] = vision_source
    mm_frontend = _loaded_mm_frontend(
        session, source_kind=source_kind, device=device, enable_images=enable_images, **mm_kwargs)
    mm_rope_rows = 0
    if mm_frontend is not None:
        from .bidec_mm import DEFAULT_MM_ROPE_ROWS

        mm_rope_rows = DEFAULT_MM_ROPE_ROWS
    # The source FastDecoder graphs are single-session captures. The B16 decoder captures its
    # own row shapes; retaining both graph pools wastes capacity and has no generation use here.
    single_graphs = getattr(fd, "graphs", None)
    if isinstance(single_graphs, dict):
        single_graphs.clear()

    _progress("batch_build", max_slots=settings["max_slots"],
              pages_total=settings["pages_total"], max_ctx=settings["max_ctx"],
              max_rows=settings["max_rows"])
    t_build = time.perf_counter()
    bd_kwargs = dict(
        max_slots=settings["max_slots"], max_rows=settings["max_rows"],
        pages_total=settings["pages_total"], max_ctx=settings["max_ctx"], R=spec_k + 1,
        gdn_ring_dtype="fp32", conv_ring=16, log=log, mm_rope_rows=mm_rope_rows,
    )
    item_cap = _prefill_item_cap(settings["pages_total"], settings["max_rows"],
                                settings["max_ctx"], settings["prefill_priority"])
    if settings.get("adaptive_serving"):
        from .serving_resources import native_worklist_item_cap
        expected_cap = native_worklist_item_cap(
            pages_total=settings["pages_total"], max_rows=settings["max_rows"],
            max_ctx=settings["max_ctx"], prefill_priority=settings["prefill_priority"])
        selected = (page_plan or {}).get("selected", {})
        controls = (page_plan or {}).get("effective_controls", {})
        if (type(settings.get("item_cap")) is not int
                or settings["item_cap"] != expected_cap
                or selected.get("item_cap") != expected_cap
                or controls.get("item_cap") != expected_cap
                or any(selected.get(k) != settings[k] for k in
                       ("max_slots", "max_rows", "pages_total", "max_ctx", "spec_k"))):
            raise ValueError("adaptive work-list capacity differs from the admitted plan")
        if settings.get("adaptive_expectation") is not None:
            from .qualification_identity import validate_adaptive_worklist_plan
            validate_adaptive_worklist_plan(page_plan, settings,
                settings["adaptive_expectation"], settings["adaptive_trusted_sources"])
        item_cap = expected_cap
    if item_cap is not None:
        bd_kwargs["item_cap"] = item_cap
    if gdn_hv16:
        bd_kwargs["gdn_hv16"] = True
    bd = bidec.BatchDecoder(fd, **bd_kwargs)
    if settings.get("adaptive_serving") and bd.g_cap != item_cap:
        raise ValueError("native work-list allocation differs from the admitted plan")
    build_s = time.perf_counter() - t_build
    capture_workload_only = bool(settings.get("capture_workload_only", False))
    capture_ceiling = settings.get("capture_workload_ceiling")
    if capture_workload_only:
        if capture_ceiling is None:
            raise ValueError("capture_workload_only requires capture_workload_ceiling")
        capture_buckets = bidec.capture_buckets_for_workload(
            max_rows=int(bd.max_rows), max_users=int(settings["max_slots"]),
            spec_k=spec_k, ceiling=int(capture_ceiling))
        _progress("batch_capture", buckets=len(capture_buckets),
                  workload_ceiling=int(capture_ceiling))
        capture_s = bd.capture(capture_buckets, telemetry=True)
    else:
        capture_buckets = tuple(getattr(bd, "buckets", ()) or ())
        _progress("batch_capture", buckets=len(capture_buckets))
        capture_s = bd.capture()
    batcher_kwargs = dict(
        max_rows_step=settings["max_rows_step"],
        prefill_chunk=settings["prefill_chunk"], max_active=settings["max_slots"], spec_k=spec_k,
        spec_ngram=spec_ngram, max_decode_rows=settings.get("max_decode_rows", 64),
        prefill_priority=settings["prefill_priority"],
        mm_encoder=mm_frontend.encode if mm_frontend is not None else None,
    )
    if "spec_row_aware" in settings:
        batcher_kwargs["spec_row_aware"] = settings["spec_row_aware"]
    if "mtp_confirm_kv_only" in settings:
        batcher_kwargs["mtp_confirm_kv_only"] = settings["mtp_confirm_kv_only"]
    batcher = bidec.Batcher(
        bd, **batcher_kwargs)
    if spec_k and torch.device(device).type == 'cuda':
        bd.spec_cost_calibration = _calibrate_spec_cost(bd, settings)
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
        "weight_format": ("bhe1-predictive-startup" if source_meta.get("runtime_format") == "bhe1" else
                          "tbe21" if source_meta.get("runtime_format") == "tbe21" else
                          "tbe-v1-predictive-startup" if source_kind == "predictive_tbe" else "bf16-dense"),
        "comparison_backend": "native-bi-gemm",
        "attention_implementation": str(attention_implementation),
        "execution_mode": "exact",
        "mtp_enabled": has_mtp,
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
        "batch_build_s": round(build_s, 3),
        "capture_s": capture_s,
        **{k: settings[k] for k in ("max_slots", "max_rows", "max_rows_step",
                                    "pages_total", "max_ctx", "prefill_chunk", "prefill_priority")},
        "capacity": dict(settings["capacity"]),
        "page_plan": dict(page_plan) if page_plan is not None else {"mode": "explicit"},
        "attention_layout": layout,
        "gdn_ring_depth": spec_k + 1,
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
    if settings.get("adaptive_serving"):
        from .qualification_identity import adaptive_worklist_allocation
        metadata["item_cap"] = int(bd.g_cap)
        metadata["worklist_allocation"] = adaptive_worklist_allocation(bd)
    from .bigemm import tbe21_tile_policy_from_census
    if getattr(batcher, "mtp_confirm_kv_only", False):
        metadata["mtp_confirm_kv_only"] = True
        metadata["capture_mtp_kv_buckets"] = sorted(getattr(bd, "graphs_mkv", {}))
    effective_tbe21_policy = tbe21_tile_policy_from_census(metadata["gemm_backend_census"])
    census = metadata["gemm_backend_census"]
    descriptor_rows = census.get("descriptors", []) if isinstance(census, dict) else []
    tbe21_descriptors = [row for row in descriptor_rows
                         if isinstance(row, dict) and row.get("format") == "tbe21"]
    if tbe21_descriptors and all(row.get("row_geometry") is True for row in tbe21_descriptors):
        bd.tbe21_row_geometry = True
        metadata["spec_cost_model"] = "verify_bucket_mtp_envelope"
    if effective_tbe21_policy is not None:
        metadata["tbe21_tile_policy"] = dict(effective_tbe21_policy)
        metadata["gemm_tile_scope"] = "legacy formats; TBE21 uses tbe21_tile_policy"
        bd.tbe21_tile_policy = dict(effective_tbe21_policy)
    if capture_workload_only:
        metadata.update({
            "capture_workload_only": True,
            "capture_workload_ceiling": int(capture_ceiling),
            "capture_captured_buckets": [int(bucket) for bucket in capture_buckets],
            "capture_bucket_timings_s": {
                str(bucket): float(seconds) for bucket, seconds in capture_s.items()
            },
        })
    if gdn_hv16:
        metadata["gdn_hv16"] = True
        metadata["gdn_geometry"] = {
            "key_heads": int(getattr(fd, "gdn_hk", 16)),
            "value_heads": int(getattr(fd, "gdn_hv", 48)),
            "key_dim": int(getattr(fd, "gdn_dk", 128)),
            "value_dim": int(getattr(fd, "gdn_dv", 128)),
            "conv_dim": int(getattr(fd, "gdn_conv_dim", 10240)),
            "conv_kernel": int(getattr(fd, "gdn_conv_kernel", 4)),
        }
    if mm_frontend is not None and vision_source is not None:
        metadata["image_vision_source"] = vision_source
    if hasattr(bd, "spec_cost_calibration"):
        metadata["spec_cost_calibration"] = bd.spec_cost_calibration
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


def resolve_pages_and_settings(session: Any, settings: Dict[str, Any], *,
                               device: str, enable_images: bool,
                               headroom_bytes: int = 6 * 1024**3,
                               require_full_slot_capacity: bool = False,
                               log: Callable[[str], None] = print,
                               mem_get_info: Optional[Callable[[], Any]] = None,
                               torch_cached_bytes: Optional[int] = None,
                               sms: Optional[int] = None,
                               ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Between session load and engine build: size ``pages_total`` when it is 0 (auto).

    Returns ``(settings with pages_total and capacity resolved, page plan or None)``.
    """
    spec_options = {k: settings[k] for k in ("spec_k", "spec_ngram", "max_decode_rows", "mtp_confirm_kv_only")
                    if k in settings}
    if settings.get('adaptive_serving'):
        return _resolve_adaptive_resources(session, settings, device=device,
            enable_images=enable_images, headroom_bytes=headroom_bytes, log=log,
            mem_get_info=mem_get_info, torch_cached_bytes=torch_cached_bytes, sms=sms)
    if int(settings.get("pages_total") or 0) > 0:
        if settings.get("capacity") is None:
            settings = _validate_batch_settings(
                **{k: settings[k] for k in ("max_slots", "max_rows", "pages_total", "max_ctx",
                                            "max_rows_step", "prefill_chunk",
                                            "prefill_priority")},
                require_full_slot_capacity=require_full_slot_capacity, **spec_options)
        return settings, None
    fd = session.fd
    adapt_fastdecoder_for_batch(fd)
    released = release_single_stream_buffers(fd)
    mm_rope_rows = 0
    if enable_images:
        from .bidec_mm import DEFAULT_MM_ROPE_ROWS

        mm_rope_rows = DEFAULT_MM_ROPE_ROWS
    requested_k = int(settings.get('spec_k', 0))
    effective_k = requested_k
    while True:
        try:
            plan = plan_pages_total(
                fd, max_slots=settings["max_slots"], max_rows=settings["max_rows"],
                max_ctx=settings["max_ctx"], headroom_bytes=headroom_bytes, device=device,
                mm_rope_rows=mm_rope_rows, mem_get_info=mem_get_info,
                torch_cached_bytes=torch_cached_bytes, sms=sms,
                spec_k=effective_k, mtp_enabled=bool(getattr(fd, "mtp_ready", False)),
                prefill_priority=settings["prefill_priority"])
            if require_full_slot_capacity and not plan['full_slot_context_capacity'] and effective_k > 1:
                effective_k -= 1
                continue
            break
        except ValueError as exc:
            if effective_k <= 1 or not str(exc).startswith('not enough device memory for the KV page pool:'):
                raise
            effective_k -= 1
    if requested_k:
        spec_options['spec_k'] = effective_k
        plan['requested_spec_k'] = requested_k
        plan['effective_spec_k'] = effective_k
        if effective_k != requested_k:
            log(f'[context-batch] memory admission capped speculative depth {requested_k} -> {effective_k}')
    plan["released_single_session_bytes"] = released
    log(f"[context-batch] page pool: {plan['pages_total']} pages "
        f"({plan['per_page_bytes'] / (1 << 20):.2f} MiB each) from "
        f"{plan['available_bytes'] / (1 << 30):.2f} GiB available, "
        f"{plan['headroom_bytes'] / (1 << 30):.2f} GiB headroom")
    resolved = _validate_batch_settings(
        **{**{k: settings[k] for k in ("max_slots", "max_rows", "max_ctx", "max_rows_step",
                                       "prefill_chunk", "prefill_priority")},
           "pages_total": plan["pages_total"]},
        require_full_slot_capacity=require_full_slot_capacity, **spec_options)
    # Preserve explicit capture/calibration/serving controls through auto sizing.
    return {**settings, **resolved}, plan


def _resolve_adaptive_resources(session, settings, *, device, enable_images,
                                headroom_bytes, log, mem_get_info, torch_cached_bytes, sms):
    from . import bidec
    from .serving_resources import plan_native_resources, native_worklist_item_cap

    if settings.get('pages_total'):
        raise ValueError('adaptive serving requires automatic pages_total=0')
    fd = session.fd
    import torch
    target = torch.device(device)
    if target.type == 'cuda' and target.index is None:
        target = torch.device('cuda', torch.cuda.current_device())
    resident = getattr(fd, 'dev', None)
    if resident is None:
        raise ValueError('adaptive serving needs an explicit resident model device')
    resident = torch.device(resident)
    if resident.type == 'cuda' and resident.index is None:
        resident = getattr(getattr(fd, 'embed_w', None), 'device', None)
        if resident is None or torch.device(resident).index is None:
            raise ValueError('adaptive serving cannot resolve the resident model device')
        resident = torch.device(resident)
    if resident != target:
        raise ValueError('resident model and resource budget must refer to the same device')
    # Freeze ambiguous CUDA aliases before any GC callback, query or allocation.
    fd.dev = target
    device = str(target)
    adapt_fastdecoder_for_batch(fd)
    released = release_single_stream_buffers(fd)
    # Adaptive admission only credits driver-reported free memory. Allocator
    # reserved slack remains visible telemetry but is not proven fungible space.
    free, total, cached, sms = _device_memory_state(
        device, mem_get_info, torch_cached_bytes, sms, strict=True)
    if sms <= 0:
        raise ValueError('actual device SM count must be positive')
    shapes = batch_engine_shapes(fd)
    spec_k = int(settings.get('spec_k', 0))
    mtp_enabled = bool(getattr(fd, 'mtp_ready', False))
    mm_rope_rows = 0
    if enable_images:
        from .bidec_mm import DEFAULT_MM_ROPE_ROWS
        mm_rope_rows = DEFAULT_MM_ROPE_ROWS

    def item_cap_for(pages, rows):
        return native_worklist_item_cap(pages_total=pages, max_rows=rows,
            max_ctx=settings['max_ctx'], prefill_priority=settings['prefill_priority'])

    def estimate(users, pages, rows):
        cap = item_cap_for(pages, rows)
        size = _ledger_bytes(shapes, slots=users, pages=pages, max_ctx=settings['max_ctx'],
            max_rows=rows, conv_ring=16, sms=sms, spec_k=spec_k,
            mtp_enabled=mtp_enabled, item_cap=cap)
        size += mm_rope_rows * shapes.rope_dim * 4 + (rows * shapes.H * 2 if mm_rope_rows else 0)
        if callable(getattr(fd, '_descs', None)):
            from .bhe_desc import extra_workspace_bytes
            size += extra_workspace_bytes(fd, rows)
        return size

    users_cap = settings['max_slots']
    if settings.get('adaptive_capture_ceiling_explicit'):
        users_cap = min(users_cap, settings['capture_workload_ceiling'] // (spec_k + 1))
    from .serving_resources import ResourceAdmissionError
    try:
        # Keep the declared physical reserve separate from noncredited allocator slack.
        plan = plan_native_resources(free_bytes=free, total_bytes=total, reclaimable_bytes=0,
            reserve_bytes=headroom_bytes, max_users=users_cap,
            min_users=settings.get('adaptive_min_users', 1), max_rows=settings['max_rows'],
            max_ctx=settings['max_ctx'], spec_k=spec_k, estimate_bytes=estimate,
            objective=settings['adaptive_serving'])
    except ResourceAdmissionError as exc:
        exc.detail.update(observed_allocator_unused_reserved_bytes=cached,
                          admission_cache_credit_bytes=0,
                          credited_available_bytes=free,
                          conservative_remaining_after_engine_and_reserve_bytes=(
                              free - headroom_bytes - exc.detail['minimum_engine_bytes']))
        raise
    plan.update(observed_allocator_unused_reserved_bytes=cached,
                admission_cache_credit_bytes=0,
                credited_available_bytes=free,
                conservative_remaining_after_engine_and_reserve_bytes=(
                    plan['selected']['remaining_after_engine_and_reserve_bytes']))
    for profile in plan['profiles']:
        profile['item_cap'] = item_cap_for(profile['pages_total'], profile['max_rows'])
        profile['prefill_priority'] = settings['prefill_priority']
    selected = plan['selected']
    from dataclasses import asdict
    from .state_budget import EngineConfig
    cfg = EngineConfig(slots=selected['max_slots'], pages=selected['pages_total'],
        max_ctx=settings['max_ctx'], max_rows=selected['max_rows'], spec_k=spec_k,
        item_cap=selected['item_cap'], conv_ring=16, sms=sms,
        dummy_state=False, fd_buffers_live=False, mtp_enabled=mtp_enabled)
    extra = 0
    if callable(getattr(fd, '_descs', None)):
        from .bhe_desc import extra_workspace_bytes
        extra = extra_workspace_bytes(fd, selected['max_rows'])
    selected['ledger'] = dict(shapes=dict(vars(shapes)), config=asdict(cfg),
        extra_workspace_bytes=extra, mm_rope_rows=mm_rope_rows,
        mm_device_bytes=mm_rope_rows * shapes.rope_dim * 4 +
            (selected['max_rows'] * shapes.H * 2 if mm_rope_rows else 0),
        ledger_device_bytes=_ledger_bytes(shapes, slots=cfg.slots, pages=cfg.pages,
            max_ctx=cfg.max_ctx, max_rows=cfg.max_rows, conv_ring=cfg.conv_ring,
            sms=cfg.sms, spec_k=cfg.spec_k, mtp_enabled=cfg.mtp_enabled, item_cap=cfg.item_cap))
    import hashlib
    from . import serving_resources, state_budget, bidec_policy
    plan['source_bindings'] = {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name, module in [('state_budget.py', state_budget), ('serving_resources.py', serving_resources),
                             ('bidec_policy.py', bidec_policy), ('bidec.py', bidec)]}
    plan['source_bindings']['predictive_batch.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plan['budget_basis'] = 'memory remaining after resident weights and release of owned single-session buffers/graphs'
    plan['execution_contract'] = {'engine_streams': 'serialized', 'contexts_per_engine': 1,
                                  'full_context_reservation': True,
                                  'peak_native_validation': 'UNMEASURED'}
    selected_spec = {k: settings[k] for k in ('spec_ngram', 'mtp_confirm_kv_only') if k in settings}
    selected_spec.update(spec_k=spec_k,
        max_decode_rows=min(settings.get('max_decode_rows', 64), selected['max_rows']))
    resolved = _validate_batch_settings(max_slots=selected['max_slots'],
        max_rows=selected['max_rows'], pages_total=selected['pages_total'], max_ctx=settings['max_ctx'],
        max_rows_step=min(settings['max_rows_step'], selected['max_rows']),
        prefill_chunk=settings['prefill_chunk'], prefill_priority=settings['prefill_priority'],
        require_full_slot_capacity=True, **selected_spec)
    resolved = {**settings, **resolved, 'item_cap': selected['item_cap']}
    if settings.get('capture_workload_only'):
        required = selected['max_slots'] * (spec_k + 1)
        ceiling = next(b for b in bidec.BUCKETS if b >= required)
        if settings.get('adaptive_capture_ceiling_explicit'):
            ceiling = min(settings['capture_workload_ceiling'], selected['max_rows'])
        resolved['capture_workload_ceiling'] = ceiling
    plan.update(mode='adaptive', pages_total=selected['pages_total'],
        pages_fit=selected['pages_total'], per_page_bytes=_ledger_bytes(shapes,
            slots=selected['max_slots'], pages=1, max_ctx=settings['max_ctx'],
            max_rows=selected['max_rows'], conv_ring=16, sms=sms, spec_k=spec_k,
            mtp_enabled=mtp_enabled, item_cap=selected['item_cap']) - _ledger_bytes(shapes,
            slots=selected['max_slots'], pages=0, max_ctx=settings['max_ctx'],
            max_rows=selected['max_rows'], conv_ring=16, sms=sms, spec_k=spec_k,
            mtp_enabled=mtp_enabled, item_cap=selected['item_cap']), headroom_bytes=headroom_bytes,
        sms=sms, device=str(device), released_single_session_bytes=released)
    plan['effective_controls'] = {key: resolved.get(key) for key in (
        'max_slots', 'max_rows', 'max_rows_step', 'pages_total', 'max_ctx',
        'spec_k', 'max_decode_rows', 'prefill_priority', 'capture_workload_only',
        'capture_workload_ceiling', 'item_cap')}
    if settings.get('adaptive_expectation') is not None:
        from .qualification_identity import validate_adaptive_worklist_plan
        validate_adaptive_worklist_plan(plan, resolved, settings['adaptive_expectation'],
                                       settings['adaptive_trusted_sources'])
    log(f"[context-batch] adaptive {settings['adaptive_serving']}: "
        f"{selected['max_slots']} users, {selected['max_rows']} rows, "
        f"{selected['pages_total']} pages, {selected['max_ctx']} context; performance unmeasured")
    return resolved, plan


def _validate_stream_options(*, verified_cpu_stream: bool, cpu_vision_dense: bool,
                             cpu_encode_workers: int, source_inflight_bytes: int,
                             cpu_encode_pipeline: bool = False) -> None:
    if type(verified_cpu_stream) is not bool:
        raise ValueError("verified_cpu_stream must be a bool")
    if type(cpu_vision_dense) is not bool:
        raise ValueError("cpu_vision_dense must be a bool")
    if type(cpu_encode_workers) is not int or not 1 <= cpu_encode_workers <= MAX_CPU_ENCODE_WORKERS:
        raise ValueError(f"cpu_encode_workers must be an integer between 1 and {MAX_CPU_ENCODE_WORKERS}")
    if cpu_encode_workers > 1 and not verified_cpu_stream:
        raise ValueError("cpu_encode_workers > 1 requires verified_cpu_stream")
    if type(cpu_encode_pipeline) is not bool:
        raise ValueError("cpu_encode_pipeline must be a bool")
    if cpu_encode_pipeline and (not verified_cpu_stream or cpu_encode_workers <= 1):
        raise ValueError("cpu_encode_pipeline requires verified_cpu_stream and cpu_encode_workers > 1")
    if cpu_vision_dense and not verified_cpu_stream:
        raise ValueError("cpu_vision_dense requires verified_cpu_stream")
    if verified_cpu_stream and (
        type(source_inflight_bytes) is not int
        or not 0 < source_inflight_bytes <= 8 * 1024**3
    ):
        raise ValueError("source_inflight_bytes must be between 1 byte and 8 GiB")


def load_predictive_session(
    *,
    package: str | Path,
    tune: str | Path | None,
    device: str = "cuda:0",
    max_ctx: int = 8192,
    stride: int = 1024,
    loader_workers: int = 1,
    metadata_cache_dir: str | Path | None = None,
    attention_implementation: str = "eager",
    log: Callable[[str], None] = print,
    verified_cpu_stream: bool = False,
    cpu_encode_workers: int = 1,
    cpu_encode_pipeline: bool = False,
    source_inflight_bytes: int = 8 * 1024**3,
    cpu_vision_dense: bool = False,
    loader_kwargs: Optional[Dict[str, Any]] = None, spec: bool = False,
    gdn_hv16: bool = False,
) -> Any:
    """Load the Context package into an MTP-off FastSession (weights resident, no batch engine).

    ``loader_kwargs`` are forwarded to ``load_predictive_fast_session`` unchanged (for
    example the transcode-cache keywords from ``transcode_cache.kwargs_from_args``).
    """
    if type(gdn_hv16) is not bool:
        raise ValueError("gdn_hv16 must be a bool")
    _validate_stream_options(verified_cpu_stream=verified_cpu_stream,
                             cpu_vision_dense=cpu_vision_dense,
                             cpu_encode_workers=cpu_encode_workers,
                             source_inflight_bytes=source_inflight_bytes,
                             cpu_encode_pipeline=cpu_encode_pipeline)
    from .predictive_tbe import load_predictive_fast_session

    session_kwargs = dict(
        package=package, tune=tune, device=device, max_len=_positive_int("max_ctx", max_ctx),
        stride=stride, loader_workers=loader_workers, metadata_cache_dir=metadata_cache_dir,
        attention_implementation=attention_implementation, log=log,
    )
    if gdn_hv16:
        session_kwargs["gdn_hv16"] = True
    if verified_cpu_stream or cpu_vision_dense:
        session_kwargs.update(
            verified_cpu_stream=verified_cpu_stream,
            cpu_encode_workers=cpu_encode_workers,
            cpu_encode_pipeline=cpu_encode_pipeline,
            source_inflight_bytes=source_inflight_bytes,
            cpu_vision_dense=cpu_vision_dense,
        )
    overlap = sorted(set(loader_kwargs or {}) & set(session_kwargs))
    if overlap:
        raise ValueError(f"loader_kwargs may not override {', '.join(overlap)}")
    session_kwargs.update(loader_kwargs or {})
    if spec:
        session_kwargs["enable_mtp"] = True
    return load_predictive_fast_session(**session_kwargs)


def load_dense_session(
    *,
    dense: str | Path,
    tune: str | Path | None,
    device: str = "cuda:0",
    max_ctx: int = 8192,
    attention_implementation: str = "sdpa",
    log: Callable[[str], None] = print, spec: bool = False,
    gdn_hv16: bool = False,
) -> Any:
    """Load the BF16 parent into an MTP-off exact FastSession (the same-engine comparator)."""
    if type(gdn_hv16) is not bool:
        raise ValueError("gdn_hv16 must be a bool")
    if attention_implementation != "sdpa":
        raise ValueError("the current FastSession dense path is fixed to sdpa attention")
    from .fastserve import FastSession

    session_kwargs = dict(
        dense=str(dense), tune=None if tune is None else str(tune), device=device,
        max_len=_positive_int("max_ctx", max_ctx),
        server_flags=(("--exec-mode", "exact") if spec else
                      ("--no-mtp", "--exec-mode", "exact")), log=log)
    if gdn_hv16:
        session_kwargs["gdn_hv16"] = True
    session = FastSession.load(**session_kwargs)
    loaded = getattr(session.engine, "loaded", None)
    actual_attention = getattr(getattr(loaded, "options", None), "attn_implementation", None)
    if actual_attention != attention_implementation:
        raise RuntimeError(
            f"dense FastSession attention is {actual_attention!r}; expected {attention_implementation!r}")
    return session


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
    cpu_encode_pipeline: bool = False,
    source_inflight_bytes: int = 8 * 1024**3,
    cpu_vision_dense: bool = False,
    vram_headroom_bytes: int = 6 * 1024**3,
    loader_kwargs: Optional[Dict[str, Any]] = None,
    mm_options: Optional[Dict[str, int]] = None,
    gdn_hv16: bool = False,
) -> PredictiveBatchRuntime:
    """Current predictive TBE-v1 startup loader, then native B16 batch runtime.

    Existing predictive defaults are retained. ``pages_total=0`` sizes the KV page pool from
    free device memory after the weights are resident (:func:`plan_pages_total`). For a
    matched dense comparison, call both factories with ``attention_implementation="sdpa"``
    and identical batch settings.
    """
    if type(gdn_hv16) is not bool:
        raise ValueError("gdn_hv16 must be a bool")
    _validate_stream_options(verified_cpu_stream=verified_cpu_stream,
                             cpu_vision_dense=cpu_vision_dense,
                             cpu_encode_workers=cpu_encode_workers,
                             source_inflight_bytes=source_inflight_bytes,
                             cpu_encode_pipeline=cpu_encode_pipeline)
    settings = _validate_batch_settings(
        max_slots=max_slots, max_rows=max_rows, pages_total=pages_total, max_ctx=max_ctx,
        max_rows_step=max_rows_step, prefill_chunk=prefill_chunk, stride=stride,
        loader_workers=loader_workers, prefill_priority=prefill_priority,
        require_full_slot_capacity=require_full_slot_capacity, allow_auto_pages=True,
    )
    session = load_predictive_session(
        package=package, tune=tune, device=device, max_ctx=settings["max_ctx"], stride=stride,
        loader_workers=loader_workers, metadata_cache_dir=metadata_cache_dir,
        attention_implementation=attention_implementation, log=log,
        verified_cpu_stream=verified_cpu_stream, cpu_encode_workers=cpu_encode_workers,
        cpu_encode_pipeline=cpu_encode_pipeline,
        source_inflight_bytes=source_inflight_bytes, cpu_vision_dense=cpu_vision_dense,
        loader_kwargs=loader_kwargs, **({"gdn_hv16": True} if gdn_hv16 else {}),
    )
    settings, plan = resolve_pages_and_settings(
        session, settings, device=device, enable_images=enable_images,
        headroom_bytes=vram_headroom_bytes,
        require_full_slot_capacity=require_full_slot_capacity, log=log)
    return build_native_batch_from_fast_session(
        session, source_kind="predictive_tbe", attention_implementation=attention_implementation,
        settings=settings, device=device, enable_images=enable_images, log=log,
        mm_options=mm_options, page_plan=plan, **({"gdn_hv16": True} if gdn_hv16 else {}),
    )


def assert_native_batch_pair_compatible(predictive: Any, dense: Any) -> Dict[str, Any]:
    """Check saved runtime metadata before a same-engine comparison; never grades speed."""
    left = getattr(predictive, "metadata", predictive)
    right = getattr(dense, "metadata", dense)
    if left.get("source_kind") != "predictive_tbe" or right.get("source_kind") != "dense_bf16":
        raise ValueError("expected predictive_tbe then dense_bf16 runtime metadata")
    fields = [
        "comparison_backend", "attention_implementation", "execution_mode", "mtp_enabled",
        "enable_images", "image_visual_module_reused",
        "tune_sha256", "max_slots", "max_rows", "max_rows_step", "pages_total", "max_ctx",
        "prefill_chunk", "prefill_priority", "gdn_ring_depth", "gdn_ring_dtype",
        "gemm_tile", "row_tile", "spec_k",
    ]
    if "gdn_hv16" in left or "gdn_hv16" in right:
        fields.append("gdn_hv16")
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
    vram_headroom_bytes: int = 6 * 1024**3,
    mm_options: Optional[Dict[str, int]] = None,
    gdn_hv16: bool = False,
) -> PredictiveBatchRuntime:
    """Optional dense BF16 twin through ``FastSession.load(dense=...)`` and native B16.

    The current dense loader's established server path uses SDPA. Other attention modes are
    refused so this factory cannot label a mismatched prefill implementation as a twin.
    ``pages_total=0`` sizes the page pool as in :func:`load_predictive_batch_runtime`.
    """
    if type(gdn_hv16) is not bool:
        raise ValueError("gdn_hv16 must be a bool")
    if attention_implementation != "sdpa":
        raise ValueError("the current FastSession dense path is fixed to sdpa attention")
    settings = _validate_batch_settings(
        max_slots=max_slots, max_rows=max_rows, pages_total=pages_total, max_ctx=max_ctx,
        max_rows_step=max_rows_step, prefill_chunk=prefill_chunk,
        prefill_priority=prefill_priority,
        require_full_slot_capacity=require_full_slot_capacity, allow_auto_pages=True,
    )
    dense_kwargs = dict(dense=dense, tune=tune, device=device,
                        max_ctx=settings["max_ctx"],
                        attention_implementation=attention_implementation, log=log)
    if gdn_hv16:
        dense_kwargs["gdn_hv16"] = True
    session = load_dense_session(**dense_kwargs)
    settings, plan = resolve_pages_and_settings(
        session, settings, device=device, enable_images=enable_images,
        headroom_bytes=vram_headroom_bytes,
        require_full_slot_capacity=require_full_slot_capacity, log=log)
    return build_native_batch_from_fast_session(
        session, source_kind="dense_bf16", attention_implementation=attention_implementation,
        settings=settings, device=device, enable_images=enable_images, log=log,
        mm_options=mm_options, page_plan=plan, **({"gdn_hv16": True} if gdn_hv16 else {}),
    )


__all__ = ["PredictiveBatchRuntime", "adapt_fastdecoder_for_batch",
           "assert_native_batch_pair_compatible", "batch_engine_shapes",
           "build_native_batch_from_fast_session", "load_dense_batch_runtime",
           "load_dense_session", "load_predictive_batch_runtime", "load_predictive_session",
           "plan_pages_total", "release_single_stream_buffers", "resolve_pages_and_settings"]
