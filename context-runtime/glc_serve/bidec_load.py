"""Model loading for the batch engine (``glc_serve.bidec``): ``load_model``, ``stop_ids_for``
and the ``--vision-placement`` mapping.

Split out of ``bidec.py`` without change; ``glc_serve.bidec`` re-exports every name here, so
``bidec.load_model`` / ``bidec.stop_ids_for`` / ``bidec.VISION_PLACEMENTS`` keep working.  This
module imports only ``bidec_mm`` and ``fastdec`` -- never ``bidec`` -- so there is no cycle.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from . import bidec_mm as bm
from . import fastdec as fdm

#: ``--vision-placement`` values accepted by ``load_model`` (docs/serving/BIDEC_MULTIMODAL_20261004.md;
#: the 2026-10-04 ENGINE_PLACEMENT values, now backed by a real image path):
#:   device    the native vision tower (``bidec_mm.VisionTower``: HF's vision module class on
#:             exact bf16 weights) resident on the accelerator.  Bit-exact by construction.
#:   ondemand  (default) its bf16 weights in pinned host RAM, uploaded for each image encode and
#:             released after.  Bit-exact by construction (the same module, weights and kernels
#:             as ``device``; only residency differs).  0 bytes of VRAM between image requests.
#:   host      weights in host RAM and the tower EXECUTED ON THE CPU.  NOT bit-exact against a
#:             GPU-run parent (CPU kernels are not GPU kernels); reported as such everywhere.
#:   off       no vision tower at all; image requests are refused (400).
#: In EVERY placement the engine's own skeleton is text-only: the loader's vision modules would
#: be TBE-serving linears (their own fused kernel, not the parent's F.linear arithmetic), so the
#: tower the engine runs is never the loader's.  (Before 2026-10-04 this flag only chose whether
#: the loader's tower was materialised -- and the bundle argv was built with the boolean
#: inverted, so ``host`` loaded it and ``device`` did not; see ``_bundle_argv``.)
VISION_PLACEMENTS = bm.VISION_PLACEMENTS


def _vision_text_only(vision_placement: str) -> bool:
    """Whether the bundle-path ENGINE skeleton is built text-only, given ``vision_placement``.

    Always True for a valid placement: the native image path runs its own
    ``bidec_mm.VisionTower`` (exact bf16 weights, HF's module class), never the loader's
    tower, so materialising the loader's tower would only be a second copy of ~0.92 GB that no
    code path reads.  Kept as a function so the mapping stays directly testable.
    """
    if vision_placement not in VISION_PLACEMENTS:
        raise ValueError(f"vision_placement must be one of {VISION_PLACEMENTS}, got {vision_placement!r}")
    return True


def _bundle_argv(bundle: str, device: str, *, text_only: bool, load_mtp: bool) -> List[str]:
    """``server._parse`` argv for the engine's bundle load.

    2026-10-04 regression: this was built inline as ``if not text_only: argv.append("--text-only")``
    -- the boolean inverted, so ``--vision-placement host`` (text_only=True) loaded the loader's
    vision tower and ``device`` skipped it.  ``--text-only`` is a store_true flag meaning
    "skip the vision tower": it is passed exactly when ``text_only`` is True.
    """
    argv = ["--bundle", bundle, "--backend", "tbe", "--exec-mode", "exact", "--device", device,
            "--gate", os.environ.get("BIDEC_GATE", "sample"), "--port", "0"]
    if text_only:
        argv.append("--text-only")
    if not load_mtp:
        argv.append("--no-mtp")
    return argv


# ---------------------------------------------------------------------------- loading
def load_model(*, bundle: Optional[str] = None, parent: Optional[str] = None, gguf: Optional[str] = None,
               tune: str, device: str = "cuda:0", spec: bool = False, weight_format: str = "tbe",
               log=print, embed_placement: str = "device", vision_placement: str = "ondemand",
               mtp_placement: str = "device", mm_options: Optional[Dict[str, Any]] = None,
               context_build_root: Optional[str] = None, context_typed: bool = False,
               context_manifest_sha256: Optional[str] = None,
               context_cache_root: Optional[str] = None, vision_from_package: bool = False,
               context_startup_workers: int = 0, context_startup_ram_bytes: Optional[int] = None,
               gdn_hv16: bool = False,
               before_tbe21_device_use: Optional[Any] = None) -> Dict[str, Any]:
    """The model for the batch engine, plus (unless ``vision_placement == "off"``) the native
    image frontend.
    bundle=<TBE bundle dir>                   -> exact TBE codec (G1-gated at load)
    bundle=<codec v2 artifact>, weight_format="tbe2"
                                              -> exact codec v2 streams through BI-GEMM Ld<F_TBE2>
                                                 (glc_serve.tbe2_load; config + tokenizer from the
                                                 artifact, else from parent=)
    parent=<HF dir>, gguf=<file>              -> the GGUF's linears (Q8_0 / K-quants fused) on the
                                                 parent's non-linear parameters; the parent is
                                                 read on the HOST, only the GGUF bytes go to the GPU
    parent=<HF dir>                           -> dense bf16 parent

    embed_placement / mtp_placement: docs/serving/ENGINE_PLACEMENT_20261004.md.
    vision_placement / mm_options: docs/serving/BIDEC_MULTIMODAL_20261004.md (``mm_options`` is
    passed to ``bidec_mm.MMFrontend``: max_images, max_image_bytes, fetch_urls, check_hf_rope).
    ``mtp_placement == "off"`` means the MTP head is not loaded at all (speculation with an
    MTP drafter is then unavailable -- ``bidec_serve`` refuses ``--spec-k`` > 0 without
    ``--spec-ngram`` before this function is ever called, so that refusal is loud and early).
    """
    from .context_source import (
        select_context_artifact_factory,
        tbe2_loader_kwargs,
        validate_context_selection,
        validate_typed_context_selection,
        validate_context_startup_selection,
    )

    validate_context_selection(build_root=context_build_root, bundle=bundle, parent=parent,
                               gguf=gguf, weight_format=weight_format)
    validate_context_startup_selection(workers=context_startup_workers,
        ram_bytes=context_startup_ram_bytes, build_root=context_build_root,
        context_typed=context_typed, weight_format=weight_format)
    validate_typed_context_selection(
        context_typed=context_typed, context_manifest_sha256=context_manifest_sha256,
        build_root=context_build_root, cache_root=context_cache_root, bundle=bundle,
        parent=parent, gguf=gguf, weight_format=weight_format)
    if type(vision_from_package) is not bool:
        raise ValueError("vision_from_package must be a bool")
    if vision_from_package and weight_format != "tbe21":
        raise ValueError("vision_from_package requires explicit tbe21")
    if vision_from_package and vision_placement != "device":
        raise ValueError("package TBE21 vision must use device placement")
    if weight_format == "tbe21":
        if embed_placement != "device":
            raise ValueError("TBE21 FastDecoder supports embed_placement='device' only")
        if mtp_placement not in ("device", "off"):
            raise ValueError("TBE21 FastDecoder supports mtp_placement='device' or 'off' only")
    from transformers import AutoConfig, AutoTokenizer

    load_mtp = bool(spec) and mtp_placement != "off"
    t0 = time.time()
    rec: Dict[str, Any] = {}
    package_processor = None
    generation_config = None
    # Every path-like argument is made a str HERE, at the one place they are spliced into an
    # argv.  rc7 run 6a: the tile gate's T3 passed `bundle` as a pathlib.Path, argparse's
    # _parse_optional indexes `arg_string[0]`, and the gate died with "TypeError: 'PosixPath'
    # object is not subscriptable" before it compared a single digest.  Callers may hand in a
    # Path (glc-bench's fetch_bundle returns one); the argv must not care.
    bundle = os.fspath(bundle) if bundle else bundle
    parent = os.fspath(parent) if parent else parent
    gguf = os.fspath(gguf) if gguf else gguf
    tune = os.fspath(tune) if tune else tune
    device = str(device)
    tn = fdm.load_tune(tune)
    if weight_format not in ("tbe", "tbe2", "tbe21"):
        raise ValueError(f"weight_format must be tbe, tbe2 or tbe21, got {weight_format!r}")
    if before_tbe21_device_use is not None and not callable(before_tbe21_device_use):
        raise ValueError("before_tbe21_device_use must be callable")
    text_only = _vision_text_only(vision_placement)
    mm_bundle = None
    if weight_format in ("tbe2", "tbe21"):
        if not bundle:
            raise SystemExit("--weight-format tbe2 needs --bundle <codec v2 artifact dir>")
        from .tbe2_load import load_v2_model

        # TBE2 retains its text-only loader. TBE21 can explicitly install the package's
        # vision tensors as dense HF parameters; that path never sources vision weights
        # from --parent.
        source_options = dict(context_typed=context_typed,
            context_manifest_sha256=context_manifest_sha256, build_root=context_build_root)
        if context_startup_workers:
            source_options.update(startup_workers=context_startup_workers,
                                  startup_ram_bytes=context_startup_ram_bytes)
        artifact_factory = select_context_artifact_factory(**source_options)
        v2_kwargs = tbe2_loader_kwargs(weight_format, embed_placement,
                                      artifact_factory=artifact_factory)
        if before_tbe21_device_use is not None:
            v2_kwargs = dict(v2_kwargs)
            v2_kwargs["before_tbe21_device_use"] = before_tbe21_device_use
        L2 = load_v2_model(bundle, config_dir=parent, device=device,
                           text_only=not vision_from_package, spec=load_mtp,
                           log=log, **v2_kwargs)
        model, mtp, tok = L2["model"], L2["mtp"], L2["tok"]
        rec[weight_format] = L2["rec"]
        if weight_format == "tbe21":
            rec["manifest_sha256"] = L2["rec"]["manifest_sha256"]
            rec["config"] = L2["rec"]["config"]
            rec["metadata_source"] = L2["rec"].get("metadata_source")
            rec["metadata_asset_sha256"] = L2["rec"].get("metadata_asset_sha256")
            if context_startup_workers:
                rec["context_startup"] = {"requested_workers": context_startup_workers,
                                          "ram_budget_bytes": context_startup_ram_bytes}
            rec["vision_source"] = L2["rec"].get("vision_source")
            rec["vision_tensor_count"] = L2["rec"].get("vision_tensor_count", 0)
            config = L2["config"]
            package_processor = L2.get("processor")
            generation_config = L2.get("generation_config")
        else:
            config = AutoConfig.from_pretrained(L2["rec"]["config"])
        local_dir = L2["rec"]["config"]
        if (vision_placement != "off" and getattr(config, "vision_config", None) is not None
                and not parent and not vision_from_package):
            raise SystemExit("--weight-format tbe2 with a multimodal checkpoint needs --parent "
                             "<HF snapshot> for the vision tower's weights, or --vision-placement off")
        src = bundle
    elif bundle:
        from .server import _parse, build_state

        # text_only selects the SKELETON (whether the loader's vision tower exists to receive
        # weights at all); load_mtp is independent of it (loader.py 2026-10-04 fix).
        st = build_state(_parse(_bundle_argv(bundle, device, text_only=text_only, load_mtp=load_mtp)),
                         log=log)
        model = st.engine.model
        mtp = st.engine.loaded.mtp if load_mtp else None
        tok = st.engine.tok
        rec["weight_gate"] = (getattr(st, "receipts", None) or {}).get("weight_gate")
        src = bundle
        config = st.engine.loaded.config
        local_dir = st.engine.loaded.local_dir
        mm_bundle = st.engine.loaded.bundle
    else:
        from .loader import _disable_cuda_only_conv_kernels, model_class_for

        config = AutoConfig.from_pretrained(parent)
        local_dir = parent
        cls = model_class_for(config, text_only=True)
        dev_map = {"": "cpu"} if gguf else {"": device}
        model = cls.from_pretrained(parent, dtype=torch.bfloat16, device_map=dev_map)
        model.eval()
        if gguf:
            from . import miv_kq
            from .q8serve import load_gguf

            miv_kq.register_with_engine()
            rec["gguf"] = load_gguf(model, gguf, tn, device=device, check_every=int(os.environ.get("BIDEC_Q8_CHECK", "16")),
                                    log=log)
            if rec["gguf"].get("check_fail"):
                raise SystemExit(f"GGUF Q8_0 blocks differ from gguf-py: {rec['gguf']['check_fail']}")
            model.to(device)
        _disable_cuda_only_conv_kernels(model, torch.device(device))
        tok = AutoTokenizer.from_pretrained(parent)
        src = gguf or parent
        mtp = None
        if load_mtp:
            from .loader import load_dense_mtp

            mtp = load_dense_mtp(parent, config, torch.device(device))
    if weight_format == "tbe21":
        # The public FastDecoder carries TBE21 streams directly but has no separate
        # host-placement accounting API. Keep this path device-resident and report
        # no synthetic FastDecoder byte totals below.
        fd = fdm.FastDecoder(model, mtp, tune=tn, max_len=512, R=1, device=device,
                             gdn_hv16=bool(gdn_hv16))
    else:
        fd_mtp_placement = mtp_placement if mtp_placement in fdm.MTP_PLACEMENTS else "device"
        fd = fdm.FastDecoder(model, mtp, tune=tn, max_len=512, R=1, device=device,
                             embed_placement=embed_placement, mtp_placement=fd_mtp_placement,
                             gdn_hv16=bool(gdn_hv16))
    # The native image frontend.  The rotary module is the engine's OWN text model's (the one
    # BatchDecoder builds its text table from), so an image prompt's scratch rows and the text
    # table come from one module.
    mm = None
    package_vision = None
    if vision_from_package:
        visual = None
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
            raise RuntimeError("TBE21 package vision tensors did not install a visual module")
        visual_parameters = list(visual.parameters())
        device_obj = torch.device(device)
        if any(p.is_meta or p.device != device_obj for p in visual_parameters):
            raise RuntimeError("TBE21 package vision parameters are not resident on the requested device")
        vision_bytes = sum(int(p.numel()) * int(p.element_size()) for p in visual_parameters)
        package_vision = {"vision_source": "tbe21_package", "vision_placement": "device",
                          "vision_resident_bytes": vision_bytes, "vision_bitexact": True}
    else:
        mm = bm.build_frontend(config=config, local_dir=local_dir, placement=vision_placement,
                               device=device, bundle=mm_bundle,
                               parent=parent if mm_bundle is None else None,
                               rotary=fdm._text(model).rotary_emb, tokenizer=tok, log=log,
                               **(mm_options or {}))
    rec["placement"] = {
        "embed_placement": embed_placement, "vision_placement": vision_placement,
        "mtp_placement": mtp_placement,
        "vision_resident_bytes": (mm.tower.device_bytes if mm is not None else
                                  package_vision["vision_resident_bytes"] if package_vision else 0),
        "vision": mm.stats() if mm is not None else package_vision,
    }
    if weight_format == "tbe21":
        rec["placement"]["fastdecoder_byte_accounting"] = "not reported by public FastDecoder"
    else:
        rec["placement"].update(fd.placement_bytes())
    torch.cuda.synchronize()
    rec["load_s"] = round(time.time() - t0, 1)
    rec["descriptors"] = fd.descriptor_kinds()
    rec["source"] = src
    rec["weight_format"] = weight_format
    log(f"[bidec] model loaded in {rec['load_s']} s; {rec['descriptors']}")
    result = {"model": model, "tok": tok, "fd": fd, "rec": rec, "mm": mm}
    if weight_format == "tbe21":
        result.update(config=config, processor=package_processor,
                      generation_config=generation_config)
    return result


def stop_ids_for(tok, source_dir: Optional[str], generation_config=None) -> List[int]:
    import json

    ids = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    if isinstance(generation_config, dict):
        e = generation_config.get("eos_token_id")
        ids |= set(int(v) for v in (e if isinstance(e, list) else [e]) if v is not None)
    for p in (Path(source_dir) / "generation_config.json",) if source_dir and Path(source_dir).is_dir() else ():
        if p.is_file():
            e = json.loads(p.read_text()).get("eos_token_id")
            ids |= set(int(v) for v in (e if isinstance(e, list) else [e]) if v is not None)
    im = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im, int) and im >= 0:
        ids.add(im)
    return sorted(ids)


__all__ = ["VISION_PLACEMENTS", "load_model", "stop_ids_for"]
