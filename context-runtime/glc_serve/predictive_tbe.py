"""Startup bridge from predictive Context weights to the FastSession/TBE v1 engine.

This is the Context serving startup adapter. It resolves the
current BCTX+PPCX package, converts exact BF16 weights one tensor at a time into
TBE v1 containers, and runs the existing FastDecoder path. It does not write a
dense checkpoint or publish an artifact. Runtime weights use TBE after startup;
this is not direct PPCX serving.

With ``transcode_cache`` set, every verified TBE container (and the dense input
embedding / dense vision tensors) is also written to a persistent cache keyed by
the package manifest; a later start with the same package, runtime version, GPU
arch and flags loads the containers from that cache instead of decoding and
re-encoding every tensor (see :mod:`glc_serve.transcode_cache`).
"""
from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib
import math
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import torch

from . import startup_progress
from . import transcode_cache as _tcache
from ._startup_limits import MAX_CPU_ENCODE_WORKERS, cpu_resolver_cache_limit

PROGRESS_INTERVAL_S = 10.0


class PredictiveTBEError(RuntimeError):
    """A verified predictive tensor could not be prepared for the TBE engine."""


class PredictiveTBEStartupError(PredictiveTBEError):
    """A named startup phase failed; ``phase`` names it, ``__cause__`` is the original error."""

    def __init__(self, phase: str, message: str):
        super().__init__(message)
        self.phase = phase


def _fail(phase: str, message: str) -> PredictiveTBEStartupError:
    startup_progress.report("failed", failed_phase=phase, error=message)
    return PredictiveTBEStartupError(phase, f"Context startup failed in phase '{phase}': {message}")


@contextlib.contextmanager
def _phase(name: str, timings: Optional[Dict[str, float]] = None):
    """Report a phase boundary; re-raise any failure as PredictiveTBEStartupError(name)."""
    startup_progress.report(name)
    started = time.perf_counter()
    try:
        yield
    except PredictiveTBEStartupError:
        raise
    except Exception as exc:
        raise _fail(name, f"{type(exc).__name__}: {exc}") from exc
    finally:
        if timings is not None:
            timings[name] = round(timings.get(name, 0.0) + time.perf_counter() - started, 3)


class _Progress:
    """Throttled progress: one log line at most every PROGRESS_INTERVAL_S plus a final line.

    Thread-safe; ``report_phase`` also forwards each emitted line to startup_progress.
    """

    def __init__(self, label: str, phase: str, *, log: Callable[[str], None], total_items: int = 0,
                 total_bytes: int = 0, unit: str = "tensors", report_phase: bool = True):
        self.label, self.phase, self.log, self.unit = label, phase, log, unit
        self.total_items, self.total_bytes = int(total_items), int(total_bytes)
        self.items = 0
        self.bytes = 0
        self.report_phase = report_phase
        self.started = time.monotonic()
        self._last = self.started
        self._lock = threading.Lock()
        self.lines = 0

    def add(self, items: int = 0, nbytes: int = 0) -> None:
        with self._lock:
            self.items += int(items)
            self.bytes += int(nbytes)
            line = self._due()
        if line is not None:
            self._emit(*line)

    def update(self, done_bytes: int, total_bytes: int, done_items: int, total_items: int) -> None:
        """Absolute update (the package verifier's progress callback signature)."""
        with self._lock:
            if int(done_bytes) < self.bytes or int(done_items) < self.items:
                # Resolver close starts a new verification pass. Loading time
                # between passes must not inflate this pass's elapsed time/ETA.
                self.started = self._last = time.monotonic()
            self.bytes, self.total_bytes = int(done_bytes), int(total_bytes)
            self.items, self.total_items = int(done_items), int(total_items)
            line = self._due()
        if line is not None:
            self._emit(*line)

    def finish(self) -> None:
        with self._lock:
            line = self._snapshot(time.monotonic())
        self._emit(*line)

    def _due(self):
        now = time.monotonic()
        if now - self._last < PROGRESS_INTERVAL_S:
            return None
        self._last = now
        return self._snapshot(now)

    def _snapshot(self, now: float):
        elapsed = now - self.started
        if self.total_bytes:
            frac = self.bytes / self.total_bytes
        elif self.total_items:
            frac = self.items / self.total_items
        else:
            frac = 1.0
        frac = min(1.0, max(0.0, frac))
        eta = elapsed * (1.0 - frac) / frac if frac > 0 else None
        return (self.items, self.bytes, frac, elapsed, eta)

    def _emit(self, items, nbytes, frac, elapsed, eta) -> None:
        eta_text = f"{eta:.0f} s" if eta is not None else "unknown"
        self.lines += 1
        self.log(f"[context-tbe] {self.label} {items}/{self.total_items} {self.unit} "
                 f"{nbytes / 1e9:.1f}/{self.total_bytes / 1e9:.1f} GB ({frac * 100:.0f}%) "
                 f"elapsed {elapsed:.0f} s ETA {eta_text}")
        if self.report_phase:
            startup_progress.report(self.phase, done=nbytes, total=self.total_bytes, unit="bytes",
                                    items_done=items, items_total=self.total_items,
                                    elapsed_s=round(elapsed, 1),
                                    eta_s=None if eta is None else round(eta, 1))


def _predictive_runtime():
    """Use the installed codec namespace, or the explicit source-checkout runtime."""
    try:
        return importlib.import_module("georefine_context_runtime.bitexact_predictive_serving")
    except ModuleNotFoundError as exc:
        if exc.name != "georefine_context_runtime":
            # A present but incomplete installation must not borrow modules from
            # an unrelated checkout and appear to be a qualified wheel.
            raise
    return importlib.import_module("scripts.bitexact_predictive_serving")


def _package_api():
    return importlib.import_module("georefine_context_runtime.bitexact_predictive_package")


def _session_deps():
    """Serving-engine pieces used after conversion (a seam for CPU tests)."""
    from transformers import AutoProcessor, AutoTokenizer

    from glc_loader.tbe_mma import resolve_tbe_mma_arch

    from .engine import Engine
    from .fastdec import MAX_M, FastDecoder, load_tune
    from .fastserve import FastSession, _resolve_tune
    from .tbe_desc import tune_tbe_per_m
    return SimpleNamespace(
        AutoProcessor=AutoProcessor, AutoTokenizer=AutoTokenizer,
        resolve_tbe_mma_arch=resolve_tbe_mma_arch, Engine=Engine, MAX_M=MAX_M,
        FastDecoder=FastDecoder, load_tune=load_tune, FastSession=FastSession,
        resolve_tune=_resolve_tune, tune_tbe_per_m=tune_tbe_per_m,
    )


def _make_tbe_linear(container, bias, device, name: str):
    """Upload one verified TBE v1 container as the FastDecoder's exact linear."""
    from glc_serve.modules import TBEServeLinear
    return TBEServeLinear(container, bias, torch.device(device), pool=None, name=name, exec_mode="exact")


def _text_model(model):
    from .fastdec import _text
    return _text(model)


def _device_identity(device: str, arch: Optional[str] = None) -> Dict[str, Any]:
    """Cache-key device identity: the TBE kernel arch and CUDA capability of ``device``."""
    dev = torch.device(device)
    ident: Dict[str, Any] = {"type": dev.type}
    if dev.type == "cuda":
        if arch is None:
            from glc_loader.tbe_mma import resolve_tbe_mma_arch
            arch = resolve_tbe_mma_arch(device)
        ident["tbe_mma_arch"] = str(arch)
        if torch.cuda.is_available():
            ident["capability"] = list(torch.cuda.get_device_capability(dev))
    elif arch is not None:
        ident["tbe_mma_arch"] = str(arch)
    return ident


def _compile_bhe_kernel_for_device(device: str) -> Dict[str, Any]:
    """Prepare required batch kernels and BHE GEMM on the selected CUDA device."""
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError("BHE kernel prebuild requires a CUDA device")
    started = time.perf_counter()
    with torch.cuda.device(dev):
        # Build FastDecoder's batch dependencies in this same startup task so
        # they overlap the verified CPU load. The server defers its generic
        # BHE prewarm only for this opted-in path.
        from .kernel_cache import ensure_serving_kernels
        ensure_serving_kernels(
            "batch", only=("miv_gemv", "fastdec", "bidec", "bi_gemm"), device=str(dev))
        # Resolve capability under the explicit device context before extension()
        # performs its architecture-sensitive build planning.
        capability = tuple(int(x) for x in torch.cuda.get_device_capability(dev))
        from georefine_context_runtime import context_exponent_gemm
        context_exponent_gemm.extension()
    return {"compile_seconds": time.perf_counter() - started,
            "compute_capability": list(capability)}


def _cache_flags(verified_cpu_stream: bool, cpu_vision_dense: bool) -> Dict[str, Any]:
    return {"layout": "mma16", "encoder": "cpu_numpy" if verified_cpu_stream else "torch",
            "cpu_vision_dense": bool(cpu_vision_dense)}


def _open_transcode_cache(transcode_cache, *, verify: str, package, device: str,
                          device_identity: Optional[Dict[str, Any]], verified_cpu_stream: bool,
                          cpu_vision_dense: bool, log) -> Optional[_tcache.TranscodeCache]:
    """Resolve the ``transcode_cache`` argument to a handle; never raises for cache problems."""
    if transcode_cache is None:
        return None
    if isinstance(transcode_cache, _tcache.TranscodeCache):
        return transcode_cache
    if verify not in _tcache.VERIFY_MODES:
        raise PredictiveTBEError(f"transcode_cache_verify must be one of {_tcache.VERIFY_MODES}")
    if package is None:
        raise PredictiveTBEError("transcode_cache requires the package path")
    from . import __version__
    try:
        ident = device_identity if device_identity is not None else _device_identity(device)
        return _tcache.TranscodeCache.for_package(
            transcode_cache, package=Path(package), runtime_version=__version__,
            device_identity=ident, flags=_cache_flags(verified_cpu_stream, cpu_vision_dense),
            verify=verify, log=log)
    except (OSError, _tcache.TranscodeCacheError) as exc:
        log(f"[transcode-cache] WARNING cache disabled: {type(exc).__name__}: {exc}")
        return None


def _disabled_cache_meta(reason: str) -> Dict[str, Any]:
    return {"enabled": False, "status": "disabled", "reason": reason, "bytes_read": 0,
            "bytes_written": 0, "seconds": {}}


def _same_device(actual: torch.device, target: torch.device) -> bool:
    if actual.type != target.type:
        return False
    if target.type != "cuda":
        return True
    current = torch.cuda.current_device() if torch.cuda.is_available() else 0
    want = target.index if target.index is not None else current
    have = actual.index if actual.index is not None else current
    return want == have


def _vision_module(model):
    """The loaded model's vision tower (``model.visual`` or ``visual``), or None."""
    get = getattr(model, "get_submodule", None)
    if not callable(get):
        return None
    for path in ("model.visual", "visual"):
        try:
            return get(path)
        except AttributeError:
            continue
    return None


def _context_devices(context) -> List[torch.device]:
    """Devices a Context weight decodes from: its own ``device`` (or its native tensor's),
    and those of every predictive reference. Unknown devices are not reported."""
    out: List[torch.device] = []
    stack, seen = [context], set()
    while stack:
        item = stack.pop()
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        dev = getattr(item, "device", None)
        if dev is None:
            dev = getattr(getattr(item, "native", None), "device", None)
        if dev is not None:
            try:
                out.append(torch.device(dev))
            except (TypeError, RuntimeError):
                pass
        stack.extend(getattr(item, "references", ()) or ())
    return out


def _module_misplacements(label: str, module, target: torch.device):
    """``(tensors checked, offending descriptions)`` for one module tree: parameters,
    buffers, uploaded TBE containers and remaining Context weights."""
    bad: List[str] = []
    checked = 0
    for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
        checked += 1
        if not _same_device(tensor.device, target):
            bad.append(f"{label}.{name}: {tensor.device}")
    for name, sub in module.named_modules():
        dev_container = getattr(sub, "_dev", None)
        if dev_container is not None and hasattr(sub, "target_device"):
            for field in ("planes", "smb", "esc", "sbbase"):
                value = getattr(dev_container, field, None)
                if isinstance(value, torch.Tensor):
                    checked += 1
                    if not _same_device(value.device, target):
                        bad.append(f"{label}.{name}.{field}: {value.device}")
        context = getattr(sub, "context_weight", None)
        if context is not None:
            for dev in _context_devices(context):
                checked += 1
                if not _same_device(dev, target):
                    bad.append(f"{label}.{name}.context_weight: decodes on {dev}")
                    break
    return checked, bad


def check_runtime_placement(model, device: str, *, log: Optional[Callable[[str], None]] = None) -> int:
    """Every serving tensor must live on ``device``.

    Checked and enforced: the text model, input embedding and output head (parameters,
    buffers, TBE containers), and the vision tower that serves image prompts (parameters,
    buffers and the decode device of any Context weight it still holds, so a vision weight
    left on the CPU resolver is caught here rather than by the first image request). Raises
    PredictiveTBEStartupError('placement') listing the offending tensors. Any other part of
    the model that is off ``device`` is named in one warning through ``log``. Returns the
    number of tensors checked.
    """
    target = torch.device(device)
    groups = [("text_model", _text_model(model)),
              ("input_embedding", model.get_input_embeddings()),
              ("output_head", model.get_output_embeddings())]
    vision = _vision_module(model)
    if vision is not None:
        groups.append(("vision_tower", vision))
    bad: List[str] = []
    checked = 0
    covered = set()
    for label, module in groups:
        if module is None:
            bad.append(f"{label}: missing")
            continue
        covered.update(id(m) for m in module.modules())
        n, problems = _module_misplacements(label, module, target)
        checked += n
        bad.extend(problems)
    if bad:
        shown = "; ".join(bad[:20])
        more = f" (+{len(bad) - 20} more)" if len(bad) > 20 else ""
        raise _fail("placement", f"{len(bad)} runtime tensor(s) are not on {target}: {shown}{more}")
    if log is not None:
        others: List[str] = []
        for name, sub in model.named_modules():
            if id(sub) in covered or not name:
                continue
            direct = [(f"{name}.{n}", t.device) for n, t in
                      list(sub.named_parameters(recurse=False)) + list(sub.named_buffers(recurse=False))
                      if not _same_device(t.device, target)]
            others.extend(f"{n}: {d}" for n, d in direct)
            context = getattr(sub, "context_weight", None)
            if context is not None:
                off = [d for d in _context_devices(context) if not _same_device(d, target)]
                if off:
                    others.append(f"{name}.context_weight: decodes on {off[0]}")
        if others:
            log(f"[context-tbe] WARNING {len(others)} tensor(s) outside the text model and vision "
                f"tower are not on {target} (not used by serving): " + "; ".join(others[:8])
                + (f" (+{len(others) - 8} more)" if len(others) > 8 else ""))
    return checked


def _align_engine_device(engine, model, device: str, log) -> None:
    """Make ``engine.device`` the serving device.

    ``Engine`` takes its device from the model's FIRST parameter, which in a vision-language
    model is a vision-tower parameter; prompts are moved to ``engine.device`` before the text
    prefill. The placement check has already proved the text model is on ``device``, so the
    engine follows it. Parameters elsewhere in the model are named in one warning.
    """
    target = torch.device(device)
    current = getattr(engine, "device", None)
    if current is None:
        engine.device = target
        return
    if _same_device(torch.device(current), target):
        return
    stray = [f"{name}: {p.device}" for name, p in model.named_parameters()
             if not p.is_meta and not _same_device(p.device, target)]
    shown = "; ".join(stray[:8]) + (f" (+{len(stray) - 8} more)" if len(stray) > 8 else "")
    log(f"[context-tbe] WARNING engine device was {current}, serving device is {target}; "
        f"using {target}. Parameters not on {target}: {shown or 'none'}")
    engine.device = target


def _text_manifest_names(tensor_names):
    """Select manifest entries for text blocks, token embeddings, and output head."""
    markers = ("language_model", "embed_tokens", "lm_head", "decoder.layers", "model.layers")
    return tuple(name for name in tensor_names if any(part in name.lower() for part in markers))



def _load_and_convert_predictive(
    load_model, *, package, device, attention_implementation, stride, loader_workers,
    metadata_cache_dir, log, verified_cpu_stream, cpu_encode_workers, cpu_encode_pipeline=False,
    source_inflight_bytes, cpu_vision_dense=False, transcode_cache=None,
    transcode_cache_verify="hash", verify_package="auto", device_identity=None, timings=None,
    runtime_format="tbe",
    reuse_bhe_validation=False, reuse_package_validation=False,
    cpu_ready_install=False,
    mtp_prefetch=None,
):
    """Load and convert with resolver lifetime spanning model construction and replacement.

    Returns ``(model, load_receipt, conversion, cpu_names)``; ``conversion`` carries
    the ``transcode_cache`` and ``package_integrity`` blocks for the session meta.
    """
    compact_choice = os.environ.get('GEOREFINE_CONTEXT_COMPACT_RANS')
    if compact_choice not in (None, '0', '1'):
        raise PredictiveTBEError('GEOREFINE_CONTEXT_COMPACT_RANS must be 0 or 1')
    if compact_choice == '1' and not verified_cpu_stream:
        raise PredictiveTBEError('compact CPU decoder requires verified_cpu_stream')
    if type(cpu_encode_workers) is not int or not 1 <= cpu_encode_workers <= MAX_CPU_ENCODE_WORKERS:
        raise PredictiveTBEError(f"cpu_encode_workers must be an integer between 1 and {MAX_CPU_ENCODE_WORKERS}")
    if type(reuse_bhe_validation) is not bool:
        raise PredictiveTBEError('reuse_bhe_validation must be a bool')
    if type(reuse_package_validation) is not bool:
        raise PredictiveTBEError('reuse_package_validation must be a bool')
    if reuse_package_validation and (not verified_cpu_stream or verify_package != 'full'):
        raise PredictiveTBEError('reuse_package_validation requires verified CPU stream and --verify-package full')
    if type(cpu_ready_install) is not bool:
        raise PredictiveTBEError('cpu_ready_install must be a bool')
    if cpu_ready_install and (runtime_format != 'bhe1' or not verified_cpu_stream
                              or not cpu_encode_pipeline or cpu_encode_workers <= 1):
        raise PredictiveTBEError('cpu_ready_install requires verified BHE1 CPU pipeline and multiple workers')
    if reuse_bhe_validation and (runtime_format != 'bhe1' or not verified_cpu_stream):
        raise PredictiveTBEError('reuse_bhe_validation requires BHE1 and verified_cpu_stream')
    if not verified_cpu_stream and cpu_encode_workers != 1:
        raise PredictiveTBEError("cpu_encode_workers > 1 requires verified_cpu_stream")
    if type(cpu_encode_pipeline) is not bool:
        raise PredictiveTBEError("cpu_encode_pipeline must be a bool")
    if cpu_encode_pipeline and (not verified_cpu_stream or type(cpu_encode_workers) is not int
                                or cpu_encode_workers <= 1):
        raise PredictiveTBEError("cpu_encode_pipeline requires verified_cpu_stream and cpu_encode_workers > 1")
    if type(cpu_vision_dense) is not bool:
        raise PredictiveTBEError("cpu_vision_dense must be a bool")
    if cpu_vision_dense and not verified_cpu_stream:
        raise PredictiveTBEError("cpu_vision_dense requires verified_cpu_stream")
    if verify_package not in _tcache.PACKAGE_VERIFY_MODES:
        raise PredictiveTBEError(f"verify_package must be one of {_tcache.PACKAGE_VERIFY_MODES}")
    timings = timings if timings is not None else {}
    effective_source_inflight_bytes = source_inflight_bytes
    mtp_source_reserve = 0
    cache = _open_transcode_cache(
        transcode_cache, verify=transcode_cache_verify, package=package, device=device,
        device_identity=device_identity, verified_cpu_stream=verified_cpu_stream,
        cpu_vision_dense=cpu_vision_dense, log=log)
    integrity: Dict[str, Any] = {"package_integrity": "full", "reason": "default"}
    resolver = None
    cpu_names = None
    if verified_cpu_stream:
        if type(source_inflight_bytes) is not int or not 0 < source_inflight_bytes <= 8 * 1024**3:
            raise PredictiveTBEError("source_inflight_bytes must be between 1 byte and 8 GiB")
        with _phase("resolver", timings):
            try:
                package_api = _package_api()
                resolver_type = package_api.VerifiedTensorResolver
            except (ImportError, AttributeError) as exc:
                raise PredictiveTBEError("installed predictive runtime lacks VerifiedTensorResolver") from exc
            resolver_kwargs: Dict[str, Any] = {"cache_bytes": cpu_resolver_cache_limit(cpu_encode_workers)}
            skip_ok = "nonframe" in getattr(package_api, "RESOLVER_INTEGRITY_MODES", ())
            if verify_package == "full":
                integrity["reason"] = "--verify-package full"
            elif cache is None:
                integrity["reason"] = "transcode cache disabled"
            elif not skip_ok:
                integrity["reason"] = "installed resolver has no non-frame integrity mode"
            elif cache.probe() is None:
                integrity["reason"] = f"no usable transcode cache ({cache.meta.get('reason')})"
            else:
                integrity = {
                    "package_integrity": "cache_bound_skip",
                    "reason": ("complete transcode cache is bound to this package manifest SHA-256 "
                               f"({cache.material['package_manifest_sha256']}) and per-tensor source SHA-256"),
                }
                resolver_kwargs["integrity"] = "nonframe"
            verify_progress = _Progress("verify", "verify", log=log, unit="files")
            if skip_ok:
                resolver_kwargs["progress"] = verify_progress.update
            resolver_started = time.perf_counter()
            log("[verified-cpu-stream] resolver start")
            if reuse_package_validation:
                resolver_kwargs["reuse_validation"] = True
            resolver = resolver_type(package, **resolver_kwargs)
            if os.environ.get('GEOREFINE_CONTEXT_COMPACT_RANS') == '1':
                select_decoder = getattr(resolver, 'enable_installed_compact_decoder', None)
                if select_decoder is None:
                    raise PredictiveTBEError('installed resolver lacks compact CPU decoder selection')
                decoder_identity = select_decoder()
                log('[verified-cpu-stream] installed compact rANS decoder ' +
                    str(decoder_identity['cpp_sha256']))
            if verify_progress.lines or verify_progress.total_items:
                verify_progress.finish()
            log(f"[verified-cpu-stream] resolver end seconds={time.perf_counter() - resolver_started:.3f}")
            integrity["resolver_open_seconds"] = round(time.perf_counter() - resolver_started, 3)
            if integrity["package_integrity"] == "full":
                integrity["checks"] = ("every package file SHA-256 at resolver open and close, plus the "
                                       "model loader's own package verification")
            else:
                integrity["checks"] = (
                    "structural checks plus every non-frame package file SHA-256 at resolver open, model "
                    "construction and close; each tensor frame SHA-256 is checked when it is read; text "
                    "tensors come from the transcode cache, so their frames are neither read nor trusted")
                integrity["nonframe_hashed_bytes"] = getattr(resolver, "nonframe_hashed_bytes", None)
            cpu_names = (tuple(resolver.tensor_names) if cpu_vision_dense
                         else _text_manifest_names(resolver.tensor_names))
            if not cpu_names:
                resolver.close()
                raise PredictiveTBEError("verified manifest contains no recognized text/embedding/head tensors")
    else:
        integrity["reason"] = "native (non verified-cpu-stream) path always runs the loader's full verification"

    try:
        model_started = time.perf_counter()
        if verified_cpu_stream:
            log("[verified-cpu-stream] model_construct start")
        extra: Dict[str, Any] = {}
        if verified_cpu_stream:
            extra = {"verified_cpu_source": resolver, "cpu_source_names": cpu_names}
            if integrity["package_integrity"] == "cache_bound_skip":
                live_resolver = resolver

                def verify_nonframe(_package, **_kwargs):
                    # Manifest unchanged since open + every non-frame file re-hashed.
                    return live_resolver.verify_integrity()

                extra["package_verifier"] = verify_nonframe
            elif reuse_package_validation:
                extra["package_verifier"] = resolver.reuse_full_validation
                integrity["middle_validation_reused"] = True
                integrity["verification_scope"] = "full_entry_reused_middle_close"
                integrity["checks"] = ("full resolver verification at entry and close; middle loader "
                                        "structural verification plus non-frame hashes via the live resolver")
        with _phase("model_construct", timings):
            try:
                model, load_receipt = load_model(
                    package, device=device, attention_implementation=attention_implementation,
                    stride=stride, loader_workers=loader_workers,
                    metadata_cache_dir=metadata_cache_dir, log=log, **extra,
                )
            finally:
                if verified_cpu_stream:
                    log(f"[verified-cpu-stream] model_construct end seconds={time.perf_counter() - model_started:.3f}")
            gc.collect()
            model.eval()
        if mtp_prefetch is not None:
            if resolver is None or runtime_format != 'bhe1' or not cpu_encode_pipeline:
                raise PredictiveTBEError('MTP prefetch requires the verified BHE CPU pipeline')
            from .predictive_mtp import _mtp_source_copy_budget
            mtp_source_reserve = _mtp_source_copy_budget(resolver)
            largest_source = max((resolver.entry(name)['source_bytes'] for name in cpu_names), default=0)
            effective_source_inflight_bytes = source_inflight_bytes - mtp_source_reserve
            if mtp_source_reserve <= 0 or effective_source_inflight_bytes < largest_source:
                raise PredictiveTBEError('MTP source-copy reservation exceeds the declared source budget')
            mtp_prefetch.start(package, model.config, device, resolver)
        conversion_started = time.perf_counter()
        if verified_cpu_stream:
            log("[verified-cpu-stream] conversion start")
        try:
            converter = convert_predictive_modules_to_tbe
            if runtime_format == 'bhe1':
                from .predictive_bhe import convert_predictive_modules_to_bhe
                converter = convert_predictive_modules_to_bhe
            conversion = converter(
                model, device=device, log=log,
                **({'validation_witnesses': True} if reuse_bhe_validation else {}),
                **({'ready_install': True} if cpu_ready_install else {}),
                **({"cpu_source_resolver": resolver} if runtime_format == 'bhe1' and cpu_encode_pipeline else {}),
                transcode_cache=cache, transcode_cache_verify=transcode_cache_verify,
                timings=timings,
                **({"verified_cpu_stream": True, "cpu_encode_workers": cpu_encode_workers,
                    "cpu_encode_pipeline": cpu_encode_pipeline,
                    "source_inflight_bytes": effective_source_inflight_bytes,
                    **({"cpu_vision_dense": True} if cpu_vision_dense else {})}
                   if verified_cpu_stream else {}),
            )
        finally:
            if verified_cpu_stream:
                log(f"[verified-cpu-stream] conversion end seconds={time.perf_counter() - conversion_started:.3f}")
        if resolver is not None and cpu_vision_dense:
            conversion["cpu_resolver_cache_bytes"] = int(resolver.cache_bytes)
        if mtp_prefetch is not None:
            try:
                mtp_prefetch.close()
            except BaseException as exc:
                raise _fail('mtp_load', str(exc)) from exc
            conversion['mtp_prefetch'] = dict(
                source_copy_reserve_bytes=mtp_source_reserve,
                bhe_source_inflight_bytes=effective_source_inflight_bytes,
                load_seconds=mtp_prefetch.elapsed_seconds)
        if resolver is not None:
            if hasattr(resolver, 'decoder_identity'):
                conversion['cpu_decoder_identity'] = dict(resolver.decoder_identity)
            with _phase("verify", timings):
                close_started = time.perf_counter()
                resolver.close()
                integrity["resolver_close_seconds"] = round(time.perf_counter() - close_started, 3)
                resolver = None
    except BaseException:
        if mtp_prefetch is not None:
            mtp_prefetch.abort()
        if resolver is not None:
            with contextlib.suppress(Exception):
                resolver.close()
        raise
    conversion["package_integrity"] = integrity
    if cache is None and "transcode_cache" not in conversion:
        conversion["transcode_cache"] = _disabled_cache_meta(
            "transcode_cache=None" if transcode_cache is None else "cache could not be opened")
    return model, load_receipt, conversion, cpu_names


def _sha256_bf16(tensor: torch.Tensor) -> str:
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.bfloat16:
        raise PredictiveTBEError("predictive linear decode must return a BF16 tensor")
    raw = tensor.detach().contiguous().view(torch.int16).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def encode_predictive_weight(context_weight: Any, expected_shape, *, name: str,
                             require_cpu: bool = False, cpu_numpy_tbe: bool = False):
    """Resolve and exact-encode one Context weight; CPU-only and independently testable."""
    from glc_loader.tbe_container import decode_tbe, encode_tbe
    if type(cpu_numpy_tbe) is not bool:
        raise PredictiveTBEError("cpu_numpy_tbe must be a bool")
    if cpu_numpy_tbe:
        from glc_loader.tbe_cpu_numpy import decode_tbe_numpy, encode_tbe_numpy
        encode_tbe, decode_tbe = encode_tbe_numpy, decode_tbe_numpy

    expected_shape = tuple(int(x) for x in expected_shape)
    shape = tuple(int(x) for x in getattr(context_weight, "shape", ()))
    if shape != expected_shape or len(shape) != 2:
        raise PredictiveTBEError(f"{name}: Context shape {shape} != expected linear shape {expected_shape}")
    if shape[0] % 8 or shape[1] % 64:
        raise PredictiveTBEError(
            f"{name}: FastDecoder TBE v1 needs N divisible by 8 and K divisible by 64; got {shape}"
        )
    source_sha = getattr(context_weight, "source_sha256", None)
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise PredictiveTBEError(f"{name}: missing Context source SHA-256")

    decoded = context_weight.decode(check=True)
    if not isinstance(decoded, torch.Tensor) or decoded.dtype != torch.bfloat16:
        raise PredictiveTBEError(f"{name}: Context decode did not return a BF16 tensor")
    if (require_cpu or cpu_numpy_tbe) and decoded.device.type != "cpu":
        raise PredictiveTBEError(f"{name}: streamed predictive decode must remain on CPU")
    if tuple(decoded.shape) != expected_shape:
        raise PredictiveTBEError(f"{name}: decoded shape {tuple(decoded.shape)} != {expected_shape}")
    decoded = decoded.detach().contiguous()
    actual_sha = _sha256_bf16(decoded)
    if actual_sha != source_sha:
        raise PredictiveTBEError(f"{name}: decoded source SHA mismatch")

    # Encode on CPU so temporary TBE arrays do not compete with source Context
    # buffers on the accelerator while the serving representation is uploaded.
    cpu_weight = decoded.to(device="cpu").contiguous()
    container = encode_tbe(cpu_weight, layout="mma16")
    if tuple(container.shape) != expected_shape:
        raise PredictiveTBEError(f"{name}: TBE encoder changed the source shape")
    roundtrip = decode_tbe(container)
    if not torch.equal(roundtrip.view(torch.int16), cpu_weight.view(torch.int16)):
        raise PredictiveTBEError(f"{name}: TBE v1 round-trip differs from predictive BF16 source")
    stored_bytes = int(container.byte_size()["total"])
    source_bytes = int(cpu_weight.numel() * cpu_weight.element_size())
    del roundtrip, cpu_weight, decoded
    return container, {
        "name": name,
        "shape": list(expected_shape),
        "source_sha256": actual_sha,
        "source_bf16_bytes": source_bytes,
        "tbe_v1_stored_bytes": stored_bytes,
    }


def _predictive_module_names(model, linear_type, embedding_type):
    rows = []
    context_name_by_id = {}
    for module_name, module in model.named_modules():
        if isinstance(module, (linear_type, embedding_type)):
            weight_name = f"{module_name}.weight" if module_name else "weight"
            rows.append((module_name, weight_name, id(module), id(module.context_weight)))
            context_name_by_id[id(module.context_weight)] = module_name
    return rows, context_name_by_id


def _dependency_order(model, rows, context_name_by_id):
    """Return dependent modules before their Context references to release buffers early."""
    module_names = {name for name, _weight, _module_id, _context_id in rows}
    visited, visiting, postorder = set(), set(), []

    def visit(name):
        if name in visited or name not in module_names:
            return
        if name in visiting:
            raise PredictiveTBEError("predictive Context reference cycle in loaded module graph")
        visiting.add(name)
        context = model.get_submodule(name).context_weight if name else model.context_weight
        for reference in getattr(context, "references", ()):
            dep_name = context_name_by_id.get(id(reference))
            if dep_name is not None:
                visit(dep_name)
        visiting.remove(name)
        visited.add(name)
        postorder.append(name)

    for name, _weight, _module_id, _context_id in rows:
        visit(name)
    # postorder puts references first; reverse it so dependent Context weights
    # are converted and released before the tensors they reference.
    return list(reversed(postorder))


def _parent_and_attr(model, dotted):
    parent_name, _, attr = dotted.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    return parent, attr


def _context_resident_bytes(contexts):
    """Count the unique Context/PPCX device allocations reachable from retained wrappers."""
    seen = set()
    total = 0

    def visit(context):
        nonlocal total
        key = id(context)
        if key in seen:
            return
        seen.add(key)
        resident = getattr(context, "resident_bytes", 0)
        if type(resident) is int and resident >= 0:
            total += resident
        for reference in getattr(context, "references", ()):
            visit(reference)

    for context in contexts:
        visit(context)
    return total



_VISION_PARTS = {"vision", "visual", "vision_tower", "vision_model", "vision_encoder"}


def _is_vision_module(module_name: str) -> bool:
    return any(part.lower() in _VISION_PARTS for part in module_name.split("."))


def _dense_embedding(weight: torch.Tensor, original, device) -> torch.nn.Embedding:
    replacement = torch.nn.Embedding.from_pretrained(
        weight.to(device=device).contiguous(), freeze=True, padding_idx=original.padding_idx,
    )
    replacement.train(original.training)
    return replacement


def _dense_vision_replacement(module, decoded, *, weight_name, device, BctxLinear, BctxEmbedding, log):
    """Dense runtime module for one verified vision weight (bias copied from the live wrapper)."""
    expected = tuple(int(x) for x in decoded.shape)
    if isinstance(module, BctxLinear):
        out_features, in_features = expected
        replacement = torch.nn.Linear(in_features, out_features, bias=False, device="meta", dtype=decoded.dtype)
        replacement.weight = torch.nn.Parameter(decoded.to(device=device).contiguous(), requires_grad=False)
        runtime_kind = _tcache.KIND_VISION_LINEAR
    elif isinstance(module, BctxEmbedding):
        replacement = torch.nn.Embedding.from_pretrained(
            decoded.to(device=device).contiguous(), freeze=True,
            padding_idx=module.padding_idx,
        )
        runtime_kind = _tcache.KIND_VISION_EMBEDDING
    else:
        raise PredictiveTBEError(f"{weight_name}: unsupported vision Context module")
    bias_receipt = None
    bias = getattr(module, "bias", None)
    if bias is not None:
        if not isinstance(module, BctxLinear) or tuple(bias.shape) != (expected[0],):
            raise PredictiveTBEError(f"{weight_name}: vision linear bias shape mismatch")
        copied_bias = bias.detach().to(device=device).contiguous()
        replacement.bias = torch.nn.Parameter(copied_bias, requires_grad=False)
        bias_bytes = int(bias.numel() * bias.element_size())
        bias_receipt = {
            "name": weight_name.rsplit(".", 1)[0] + ".bias", "component": "vision",
            "source_kind": "verified_cpu_resolver_dense_bias",
            "source_bytes": bias_bytes, "decoded_bytes": bias_bytes,
            "runtime_bytes": int(copied_bias.numel() * copied_bias.element_size()),
            "runtime_kind": "dense_bias", "shape": list(bias.shape),
            "runtime_dtype": str(copied_bias.dtype),
        }
        log(f"[cpu-vision-dense] tensor={bias_receipt['name']} component=vision decoded_bytes={bias_bytes} source_bytes={bias_bytes} runtime_bytes={bias_receipt['runtime_bytes']} source_kind=verified_cpu_resolver_dense_bias runtime_kind=dense_bias")
    replacement.train(module.training)
    return replacement, runtime_kind, bias_receipt


def _expected_cache_entries(model, text_rows, preserved_rows, BctxLinear, BctxEmbedding, cpu_vision_dense):
    entries = []
    for module_name, weight_name, _module_id, _context_id in text_rows:
        parent, attr = _parent_and_attr(model, module_name)
        module = getattr(parent, attr)
        context = module.context_weight
        if isinstance(module, BctxLinear):
            kind, shape = _tcache.KIND_TBE, (int(module.out_features), int(module.in_features))
        else:
            kind, shape = _tcache.KIND_EMBEDDING, (int(module.num_embeddings), int(module.embedding_dim))
        entries.append(_tcache.expected_entry(module_name, weight_name, kind, shape,
                                              getattr(context, "source_sha256", None)))
    if cpu_vision_dense:
        for module_name, weight_name, _module_id, _context_id in preserved_rows:
            if not _is_vision_module(module_name):
                continue
            parent, attr = _parent_and_attr(model, module_name)
            module = getattr(parent, attr)
            if isinstance(module, BctxLinear):
                kind = _tcache.KIND_VISION_LINEAR
            elif isinstance(module, BctxEmbedding):
                kind = _tcache.KIND_VISION_EMBEDDING
            else:
                continue
            context = module.context_weight
            entries.append(_tcache.expected_entry(module_name, weight_name, kind, context.shape,
                                                  getattr(context, "source_sha256", None)))
    return entries


def _apply_transcode_cache(model, cache, index, expected, *, device, log, BctxLinear, BctxEmbedding,
                           cpu_vision_dense):
    """Replace wrappers from a verified cache entry; restores every original on any failure."""
    cache.check_expected(index, expected)
    total_bytes = sum(int(s.get("bytes") or 0) for s in index["shards"])
    progress = _Progress("cache-load", "cache_load", log=log, total_items=len(index["tensors"]),
                         total_bytes=total_bytes)
    originals: Dict[str, Any] = {}
    # Rows were written in cold-conversion order, so receipts come back in that order.
    receipts: List[Dict[str, Any]] = []
    vision_dense: List[Dict[str, Any]] = []
    vision_source = vision_runtime = 0
    on_cpu = torch.device(device).type == "cpu"
    try:
        for row, payload in cache.iter_load(index, on_bytes=lambda n: progress.add(nbytes=n)):
            module_name, weight_name, kind = row["module_name"], row["weight_name"], row["runtime_kind"]
            parent, attr = _parent_and_attr(model, module_name)
            original = getattr(parent, attr)
            if on_cpu and isinstance(payload, torch.Tensor):
                payload = payload.clone()  # own storage instead of a view of the shard buffer
            if kind == _tcache.KIND_TBE:
                if not isinstance(original, BctxLinear):
                    raise _tcache.CacheVerifyError("live module is not a Context linear", weight_name)
                if original.bias is not None:
                    raise PredictiveTBEError(
                        f"{weight_name}: FastDecoder's projection descriptor path does not apply linear bias")
                replacement = _make_tbe_linear(payload, original.bias, device, weight_name)
                record = {
                    "name": weight_name, "shape": list(row["shape"]),
                    "source_sha256": row["source_sha256"],
                    "source_bf16_bytes": int(row["source_bf16_bytes"]),
                    "tbe_v1_stored_bytes": int(payload.byte_size()["total"]),
                    "runtime_kind": "tbe_v1_linear",
                    "tbe_v1_resident_bytes": int(replacement.resident_bytes),
                }
            elif kind == _tcache.KIND_EMBEDDING:
                if not isinstance(original, BctxEmbedding):
                    raise _tcache.CacheVerifyError("live module is not a Context embedding", weight_name)
                replacement = _dense_embedding(payload, original, device)
                nbytes = int(payload.numel() * payload.element_size())
                record = {
                    "name": weight_name, "shape": list(row["shape"]),
                    "source_sha256": row["source_sha256"], "source_bf16_bytes": nbytes,
                    "tbe_v1_stored_bytes": 0, "runtime_kind": "bf16_input_embedding",
                    "runtime_dense_bytes": nbytes,
                }
            else:
                replacement, runtime_kind, bias_receipt = _dense_vision_replacement(
                    original, payload, weight_name=weight_name, device=device,
                    BctxLinear=BctxLinear, BctxEmbedding=BctxEmbedding, log=log)
                source_bytes = int(payload.numel() * payload.element_size())
                runtime_bytes = int(replacement.weight.numel() * replacement.weight.element_size())
                vision_dense.append({
                    "name": weight_name, "component": "vision", "source_kind": "transcode_cache",
                    "shape": list(row["shape"]), "source_sha256": row["source_sha256"],
                    "source_bytes": source_bytes, "decoded_bytes": source_bytes,
                    "runtime_bytes": runtime_bytes, "runtime_kind": runtime_kind,
                })
                vision_source += source_bytes
                vision_runtime += runtime_bytes
                if bias_receipt is not None:
                    vision_dense.append(bias_receipt)
                    vision_source += bias_receipt["source_bytes"]
                    vision_runtime += bias_receipt["runtime_bytes"]
                record = None
            if record is not None and cpu_vision_dense:
                record.update({
                    "component": "text", "source_kind": "transcode_cache",
                    "decoded_bytes": record["source_bf16_bytes"],
                    "source_bytes": record["source_bf16_bytes"],
                    "runtime_bytes": record.get("tbe_v1_resident_bytes", record.get("runtime_dense_bytes", 0)),
                })
            originals.setdefault(module_name, original)
            setattr(parent, attr, replacement)
            if record is not None:
                receipts.append(record)
            del payload, replacement, original
            progress.add(items=1)
    except BaseException:
        for module_name, original in originals.items():
            parent, attr = _parent_and_attr(model, module_name)
            setattr(parent, attr, original)
        raise
    progress.finish()
    return receipts, vision_dense, vision_source, vision_runtime


def convert_predictive_modules_to_tbe(
    model, *, device: str, log: Callable[[str], None] = print,
    verified_cpu_stream: bool = False, cpu_encode_workers: int = 1,
    cpu_encode_pipeline: bool = False,
    source_inflight_bytes: int = 8 * 1024**3,
    cpu_vision_dense: bool = False,
    transcode_cache=None, transcode_cache_verify: str = "hash", package=None,
    timings: Optional[Dict[str, float]] = None,
):
    """Replace predictive Linear/Embedding wrappers with TBE linears/dense embedding.

    Embeddings remain BF16 because FastDecoder's current input-embedding path
    expects a dense table. Linear weights become uploaded TBE v1 device containers.

    ``transcode_cache`` is None (disabled), a cache root (``package`` required) or a
    :class:`glc_serve.transcode_cache.TranscodeCache`. On a verified hit no Context
    tensor is decoded or encoded; on a miss (or a failed verification, which is
    logged and quarantined) the normal conversion runs and streams every verified
    container into a new cache entry.
    """
    if type(cpu_encode_workers) is not int or not 1 <= cpu_encode_workers <= MAX_CPU_ENCODE_WORKERS:
        raise PredictiveTBEError(f"cpu_encode_workers must be between 1 and {MAX_CPU_ENCODE_WORKERS}")
    if not verified_cpu_stream and cpu_encode_workers != 1:
        raise PredictiveTBEError("cpu_encode_workers > 1 requires verified_cpu_stream")
    if type(cpu_encode_pipeline) is not bool:
        raise PredictiveTBEError("cpu_encode_pipeline must be a bool")
    if cpu_encode_pipeline and (not verified_cpu_stream or type(cpu_encode_workers) is not int
                                or cpu_encode_workers <= 1):
        raise PredictiveTBEError("cpu_encode_pipeline requires verified_cpu_stream and cpu_encode_workers > 1")
    if type(cpu_vision_dense) is not bool:
        raise PredictiveTBEError("cpu_vision_dense must be a bool")
    if cpu_vision_dense and not verified_cpu_stream:
        raise PredictiveTBEError("cpu_vision_dense requires verified_cpu_stream")
    if verified_cpu_stream and (
        type(source_inflight_bytes) is not int or not 0 < source_inflight_bytes <= 8 * 1024**3
    ):
        raise PredictiveTBEError("source_inflight_bytes must be between 1 byte and 8 GiB")
    timings = timings if timings is not None else {}
    runtime = _predictive_runtime()
    BctxEmbedding, BctxLinear = runtime.BctxEmbedding, runtime.BctxLinear
    cache = _open_transcode_cache(
        transcode_cache, verify=transcode_cache_verify, package=package, device=device,
        device_identity=None, verified_cpu_stream=verified_cpu_stream,
        cpu_vision_dense=cpu_vision_dense, log=log)

    # FastDecoder consumes the text model's projections/input embedding plus the
    # model output head. Vision-tower modules are used by prefill and stay on
    # their existing Context wrappers, including any real linear biases.
    text_model = _text_model(model)
    all_rows, _ = _predictive_module_names(model, BctxLinear, BctxEmbedding)
    selected_module_ids = {
        id(module) for module in text_model.modules()
        if isinstance(module, (BctxLinear, BctxEmbedding))
    }
    output_head = model.get_output_embeddings()
    if isinstance(output_head, (BctxLinear, BctxEmbedding)):
        selected_module_ids.add(id(output_head))
    text_rows = [
        row for row in all_rows if row[2] in selected_module_ids
    ]
    context_name_by_id = {
        context_id: name for name, _weight, _module_id, context_id in text_rows
    }
    preserved_rows = [
        row for row in all_rows if row[2] not in selected_module_ids
    ]
    order = _dependency_order(model, text_rows, context_name_by_id)
    if verified_cpu_stream:
        # CPU contexts carry no native device allocations to release. Resolve
        # references first so the bounded CPU cache can reuse them, rather than
        # starting with the deepest prediction chain as the native path does.
        order.reverse()
    weight_name_by_module = {
        name: weight_name for name, weight_name, _module_id, _ctx_id in text_rows
    }
    receipts = []
    vision_dense = []
    vision_dense_source_bytes = 0
    vision_dense_runtime_bytes = 0
    t0 = time.perf_counter()
    dev = torch.device(device)
    cuda_memory = None
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(dev)
        allocated_before = int(torch.cuda.memory_allocated(dev))
        reserved_before = int(torch.cuda.memory_reserved(dev))
        torch.cuda.reset_peak_memory_stats(dev)
        cuda_memory = {
            "allocated_before_bytes": allocated_before,
            "reserved_before_bytes": reserved_before,
        }

    expected = _expected_cache_entries(model, text_rows, preserved_rows, BctxLinear, BctxEmbedding,
                                       cpu_vision_dense)
    cache_meta = cache.meta if cache is not None else _disabled_cache_meta(
        "transcode_cache=None" if transcode_cache is None else "cache could not be opened")
    cache_hit = False
    if cache is not None:
        index = cache.probe()
        if index is not None:
            load_started = time.perf_counter()
            try:
                with _phase("cache_load", timings):
                    try:
                        receipts, vision_dense, vision_dense_source_bytes, vision_dense_runtime_bytes = (
                            _apply_transcode_cache(
                                model, cache, index, expected, device=device, log=log,
                                BctxLinear=BctxLinear, BctxEmbedding=BctxEmbedding,
                                cpu_vision_dense=cpu_vision_dense))
                    except (_tcache.CacheVerifyError, OSError) as exc:
                        where = getattr(exc, "where", "") or "cache"
                        log(f"[transcode-cache] WARNING verification failed ({where}): {exc}; "
                            "falling back to a full transcode")
                        cache.quarantine(f"{type(exc).__name__}: {exc}")
                        cache_meta.update({"status": "fallback",
                                           "reason": f"verification failed: {type(exc).__name__}: {exc}",
                                           "failed": where})
                        receipts, vision_dense = [], []
                        vision_dense_source_bytes = vision_dense_runtime_bytes = 0
                    else:
                        cache_hit = True
                        cache_meta.update({"status": "hit", "reason": None})
            finally:
                cache_meta["seconds"]["load"] = round(time.perf_counter() - load_started, 3)
            if cache_hit:
                log(f"[transcode-cache] hit {cache.dir} verify={cache.verify} "
                    f"tensors={len(index['tensors'])} bytes_read={cache_meta['bytes_read']} "
                    f"seconds={cache_meta['seconds']['load']}")

    writer = None
    if cache is not None and not cache_hit:
        estimate = 0
        for entry in expected:
            nbytes = math.prod(entry["shape"]) * 2
            estimate += int(nbytes * _tcache.TBE_STORED_FRACTION_ESTIMATE) if entry["runtime_kind"] == _tcache.KIND_TBE else nbytes
        log(f"[transcode-cache] no usable entry under {cache.root} ({cache.meta.get('reason')}); "
            f"this start transcodes and needs ~{estimate / 1e9:.1f} GB there "
            f"(+{_tcache.FREE_SPACE_RESERVE_BYTES / 1e9:.1f} GB free-space reserve)")
        cache_meta["estimate_bytes"] = int(estimate)
        writer = cache.writer(estimate)
        if writer is not None:
            write_progress = _Progress("cache-write", "cache_write", log=log, total_bytes=estimate,
                                       total_items=len(expected), report_phase=False)
            writer.on_bytes = lambda n: write_progress.add(nbytes=n)
            log(f"[transcode-cache] miss; writing {writer.partial} (estimate {estimate / 1e9:.1f} GB)")

    try:
        if not cache_hit:
            with _phase("transcode", timings):
                receipts = _transcode_text(
                    model, order, weight_name_by_module, BctxLinear, BctxEmbedding, device=device, log=log,
                    verified_cpu_stream=verified_cpu_stream, cpu_encode_workers=cpu_encode_workers,
                    cpu_encode_pipeline=cpu_encode_pipeline,
                    source_inflight_bytes=source_inflight_bytes, cpu_vision_dense=cpu_vision_dense,
                    writer=writer, expected=expected)
    except BaseException:
        if writer is not None:
            writer.abort("conversion failed")
        raise
    preserved = []
    preserved_contexts = []
    try:
        for module_name, weight_name, _module_id, _context_id in preserved_rows:
            parent, attr = _parent_and_attr(model, module_name)
            module = getattr(parent, attr)
            if cache_hit and not isinstance(module, (BctxLinear, BctxEmbedding)):
                continue  # already replaced from the cache (dense vision)
            context = module.context_weight
            is_vision = _is_vision_module(module_name)
            if cpu_vision_dense and is_vision:
                with _phase("transcode", timings):
                    expected_shape = tuple(int(x) for x in context.shape)
                    source_bytes = math.prod(expected_shape) * 2
                    if source_bytes > source_inflight_bytes:
                        raise PredictiveTBEError(f"{weight_name}: tensor exceeds CPU source in-flight budget")
                    decoded = context.decode(check=True)
                    if not isinstance(decoded, torch.Tensor) or decoded.dtype != torch.bfloat16:
                        raise PredictiveTBEError(f"{weight_name}: verified vision decode did not return BF16")
                    if tuple(decoded.shape) != expected_shape:
                        raise PredictiveTBEError(f"{weight_name}: decoded vision shape mismatch")
                    decoded = decoded.detach().contiguous()
                    actual_sha = _sha256_bf16(decoded)
                    if actual_sha != getattr(context, "source_sha256", None):
                        raise PredictiveTBEError(f"{weight_name}: vision source SHA mismatch")
                    if decoded.device.type != "cpu":
                        raise PredictiveTBEError(f"{weight_name}: verified vision decode must remain on CPU")
                    if decoded.numel() * decoded.element_size() != source_bytes:
                        raise PredictiveTBEError(f"{weight_name}: decoded vision byte count mismatch")
                    replacement, runtime_kind, bias_receipt = _dense_vision_replacement(
                        module, decoded, weight_name=weight_name, device=device,
                        BctxLinear=BctxLinear, BctxEmbedding=BctxEmbedding, log=log)
                    runtime_bytes = int(replacement.weight.numel() * replacement.weight.element_size())
                    row = {
                        "name": weight_name, "component": "vision",
                        "source_kind": "verified_cpu_resolver",
                        "shape": list(expected_shape), "source_sha256": actual_sha,
                        "source_bytes": source_bytes, "decoded_bytes": source_bytes,
                        "runtime_bytes": runtime_bytes,
                        "runtime_kind": runtime_kind,
                    }
                    if writer is not None:
                        writer.add_dense(module_name, weight_name, runtime_kind, decoded, source_sha256=actual_sha)
                    vision_dense.append(row)
                    vision_dense_source_bytes += source_bytes
                    vision_dense_runtime_bytes += runtime_bytes
                    if bias_receipt is not None:
                        vision_dense.append(bias_receipt)
                        vision_dense_source_bytes += bias_receipt["source_bytes"]
                        vision_dense_runtime_bytes += bias_receipt["runtime_bytes"]
                    log(f"[cpu-vision-dense] tensor={weight_name} component=vision decoded_bytes={source_bytes} source_bytes={source_bytes} runtime_bytes={row['runtime_bytes']} source_kind=verified_cpu_resolver")
                    setattr(parent, attr, replacement)
                    del decoded, replacement, module
                continue
            preserved_contexts.append(context)
            visual = "visual" in module_name.lower()
            preserved.append({
                "name": weight_name,
                "kind": "predictive_reference_vision" if visual else "predictive_reference_nontext",
                "shape": [int(x) for x in context.shape],
                "source_sha256": getattr(context, "source_sha256", None),
                "own_resident_context_bytes": int(getattr(context, "resident_bytes", 0)),
                "bias_present": bool(getattr(module, "bias", None) is not None),
            })
    except BaseException:
        if writer is not None:
            writer.abort("conversion failed")
        raise
    if writer is not None:
        write_started = time.perf_counter()
        with _phase("cache_write", timings):
            published = writer.commit()
        cache_meta["seconds"]["write_commit"] = round(time.perf_counter() - write_started, 3)
        if published:
            cache_meta["written"] = True
            log(f"[transcode-cache] wrote {cache.dir} ({cache_meta['bytes_written'] / 1e9:.1f} GB in "
                f"{cache_meta['write']['shards']} shards)")
    if cuda_memory is not None:
        torch.cuda.synchronize(dev)
        peak = int(torch.cuda.max_memory_allocated(dev))
        cuda_memory.update({
            "allocated_after_bytes": int(torch.cuda.memory_allocated(dev)),
            "reserved_after_bytes": int(torch.cuda.memory_reserved(dev)),
            "peak_allocated_bytes": peak,
            "peak_incremental_bytes": max(0, peak - cuda_memory["allocated_before_bytes"]),
        })
    receipt = {
        "tensor_count": len(receipts),
        "linear_count": sum(r["runtime_kind"] == "tbe_v1_linear" for r in receipts),
        "embedding_count": sum(r["runtime_kind"] == "bf16_input_embedding" for r in receipts),
        "source_bf16_bytes": sum(r["source_bf16_bytes"] for r in receipts),
        "tbe_v1_linear_source_bytes": sum(
            r["source_bf16_bytes"] for r in receipts if r["runtime_kind"] == "tbe_v1_linear"
        ),
        "tbe_v1_stored_bytes": sum(r["tbe_v1_stored_bytes"] for r in receipts),
        "tbe_v1_linear_storage_ratio": (
            sum(r["source_bf16_bytes"] for r in receipts if r["runtime_kind"] == "tbe_v1_linear")
            / max(1, sum(r["tbe_v1_stored_bytes"] for r in receipts))
        ),
        "tbe_v1_resident_bytes": sum(r.get("tbe_v1_resident_bytes", 0) for r in receipts),
        "preserved_context_module_count": len(preserved),
        "preserved_context_resident_bytes": _context_resident_bytes(preserved_contexts),
        "preserved_contexts": preserved,
        "cuda_memory": cuda_memory,
        "conversion_seconds": time.perf_counter() - t0,
        "source": "transcode_cache" if cache_hit else "context_transcode",
        "tensors": receipts,
        "transcode_cache": cache_meta,
    }
    if cpu_vision_dense:
        receipt.update({
            "cpu_vision_dense": True,
            "vision_dense_tensor_count": len(vision_dense),
            "vision_dense_source_bytes": vision_dense_source_bytes,
            "vision_dense_runtime_bytes": vision_dense_runtime_bytes,
            "vision_dense_tensors": vision_dense,
        })
    return receipt


def _ordered_bounded_encode(items, encode, *, workers: int, byte_budget: int):
    """Yield ordered encoder results under worker-count and source-byte bounds.

    Each item is ``(name, source_bytes, run_in_worker)``. An item with
    ``run_in_worker=False`` is converted by the main-thread consumer (the dense
    embedding); its byte reservation remains held until the consumer resumes
    this generator after conversion.
    """
    executor = ThreadPoolExecutor(max_workers=workers)
    pending = deque()
    iterator = iter(items)
    deferred = None
    exhausted = False
    reserved = 0
    failed = threading.Event()
    failure_lock = threading.Lock()

    def invoke(name):
        try:
            return encode(name)
        except BaseException:
            with failure_lock:
                failed.set()
            raise

    try:
        while pending or not exhausted or deferred is not None:
            while len(pending) < workers and not exhausted and not failed.is_set():
                if deferred is None:
                    try:
                        item = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                else:
                    item, deferred = deferred, None
                name, size, run_in_worker = item
                if type(size) is not int or size < 0:
                    raise PredictiveTBEError(f"{name}: invalid source byte count {size!r}")
                if size > byte_budget:
                    raise PredictiveTBEError(
                        f"{name}: tensor exceeds CPU source in-flight budget ({size} > {byte_budget})")
                if reserved + size > byte_budget:
                    deferred = item
                    break
                reserved += size
                if run_in_worker:
                    # If a worker has failed, do not enqueue work behind it.
                    with failure_lock:
                        if failed.is_set():
                            reserved -= size
                            deferred = item
                            break
                        future = executor.submit(invoke, name)
                    pending.append((name, size, future))
                else:
                    pending.append((name, size, None))

            if not pending:
                if failed.is_set():
                    raise PredictiveTBEError("CPU encode worker failed before producing an ordered result")
                if exhausted:
                    break
                continue

            name, size, future = pending.popleft()
            try:
                value = None if future is None else future.result()
                yield name, value, size
            except BaseException:
                with failure_lock:
                    failed.set()
                raise
            finally:
                reserved -= size
                # The consumer has finished uploading this tensor. Release both
                # references before refilling the freed slot.
                value = None
                future = None
    finally:
        for _name, _size, future in pending:
            if future is not None:
                future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def _transcode_text(model, order, weight_name_by_module, BctxLinear, BctxEmbedding, *, device, log,
                    verified_cpu_stream, cpu_encode_workers, cpu_encode_pipeline,
                    source_inflight_bytes,
                    cpu_vision_dense, writer, expected):
    """Decode, verify, TBE-encode and upload every selected text tensor (the cold path)."""
    first_tensor_trace = {"started": False}
    first_tensor_trace_lock = threading.Lock()
    text_bytes = {e["module_name"]: math.prod(e["shape"]) * 2 for e in expected
                  if e["runtime_kind"] in (_tcache.KIND_TBE, _tcache.KIND_EMBEDDING)}
    progress = _Progress("transcode", "transcode", log=log, total_items=len(order),
                         total_bytes=sum(text_bytes.values()))
    receipts = []

    def encode_one(module_name):
        parent, attr = _parent_and_attr(model, module_name)
        original = getattr(parent, attr)
        if not isinstance(original, BctxLinear):
            return module_name, None, None
        if original.bias is not None:
            raise PredictiveTBEError(
                f"{weight_name_by_module[module_name]}: FastDecoder's projection descriptor path does not apply linear bias"
            )
        expected_shape = (int(original.out_features), int(original.in_features))
        nbytes = math.prod(expected_shape) * 2
        if verified_cpu_stream and nbytes > source_inflight_bytes:
            raise PredictiveTBEError(f"{weight_name_by_module[module_name]}: tensor exceeds CPU source in-flight budget")
        first_trace = False
        if verified_cpu_stream:
            with first_tensor_trace_lock:
                if not first_tensor_trace["started"]:
                    first_tensor_trace["started"] = True
                    first_trace = True
            if first_trace:
                log(f"[verified-cpu-stream] first_tensor start name={weight_name_by_module[module_name]}")
        tensor_started = time.perf_counter()
        try:
            if verified_cpu_stream and cpu_encode_workers > 1:
                result = encode_predictive_weight(
                    original.context_weight, expected_shape, name=weight_name_by_module[module_name],
                    require_cpu=True,
                    cpu_numpy_tbe=True,
                )
            else:
                result = encode_predictive_weight(
                    original.context_weight, expected_shape, name=weight_name_by_module[module_name],
                    require_cpu=cpu_vision_dense or verified_cpu_stream,
                    **({"cpu_numpy_tbe": True} if verified_cpu_stream else {}),
                )
        finally:
            if first_trace:
                log(f"[verified-cpu-stream] first_tensor end seconds={time.perf_counter() - tensor_started:.3f}")
        return module_name, result, nbytes

    def batches():
        current, used = [], 0
        for module_name in order:
            parent, attr = _parent_and_attr(model, module_name)
            module = getattr(parent, attr)
            if not isinstance(module, BctxLinear):
                yield [module_name]
                continue
            size = math.prod((int(module.out_features), int(module.in_features))) * 2
            if current and (
                (verified_cpu_stream and used + size > source_inflight_bytes)
                or len(current) >= cpu_encode_workers
            ):
                yield current
                current, used = [], 0
            if verified_cpu_stream and size > source_inflight_bytes:
                raise PredictiveTBEError(f"{weight_name_by_module[module_name]}: tensor exceeds CPU source in-flight budget")
            current.append(module_name)
            used += size
        if current:
            yield current

    def encoded_in_order():
        if cpu_encode_pipeline:
            def pipeline_items():
                for module_name in order:
                    parent, attr = _parent_and_attr(model, module_name)
                    module = getattr(parent, attr)
                    if isinstance(module, BctxLinear):
                        size = math.prod((int(module.out_features), int(module.in_features))) * 2
                        yield module_name, size, True
                    elif isinstance(module, BctxEmbedding):
                        size = math.prod((int(module.num_embeddings), int(module.embedding_dim))) * 2
                        yield module_name, size, False
                    else:
                        yield module_name, 0, False

            yield from _ordered_bounded_encode(
                pipeline_items(), lambda name: encode_one(name)[1],
                workers=cpu_encode_workers, byte_budget=source_inflight_bytes,
            )
            return
        for batch in batches():
            if cpu_encode_workers == 1 or len(batch) == 1:
                for name in batch:
                    yield encode_one(name)
            else:
                # Future count and source bytes are both bounded by the batch builder.
                with ThreadPoolExecutor(max_workers=cpu_encode_workers) as pool:
                    futures = [pool.submit(encode_one, name) for name in batch]
                    for future in futures:
                        yield future.result()

    encoded_rows = encoded_in_order()
    try:
        for module_name, encoded, _source_bytes in encoded_rows:
            weight_name = weight_name_by_module[module_name]
            # Fetch the live module instead of retaining all original wrappers in a list.
            parent, attr = _parent_and_attr(model, module_name)
            original = getattr(parent, attr)
            if isinstance(original, BctxLinear):
                container, record = encoded
                replacement = _make_tbe_linear(container, original.bias, device, weight_name)
                record["runtime_kind"] = "tbe_v1_linear"
                record["tbe_v1_resident_bytes"] = int(replacement.resident_bytes)
                if writer is not None:
                    writer.add_tbe(module_name, weight_name, container, source_sha256=record["source_sha256"],
                                   source_bf16_bytes=record["source_bf16_bytes"])
            else:
                # FastDecoder reads the input embedding table directly. Keep this one
                # table dense and drop its original Context/PPCX references promptly.
                decoded = original.context_weight.decode(check=True).detach().contiguous()
                expected_shape = (int(original.num_embeddings), int(original.embedding_dim))
                source_bytes = math.prod(expected_shape) * 2
                if verified_cpu_stream and source_bytes > source_inflight_bytes:
                    raise PredictiveTBEError(f"{weight_name}: tensor exceeds CPU source in-flight budget")
                if verified_cpu_stream and cpu_encode_workers > 1 and decoded.device.type != "cpu":
                    raise PredictiveTBEError(f"{weight_name}: streamed predictive decode must remain on CPU")
                if tuple(decoded.shape) != expected_shape:
                    raise PredictiveTBEError(f"{weight_name}: decoded embedding shape mismatch")
                source_sha = getattr(original.context_weight, "source_sha256", None)
                actual_sha = _sha256_bf16(decoded)
                if actual_sha != source_sha:
                    raise PredictiveTBEError(f"{weight_name}: embedding source SHA mismatch")
                replacement = _dense_embedding(decoded, original, device)
                record = {
                    "name": weight_name, "shape": list(expected_shape),
                    "source_sha256": actual_sha,
                    "source_bf16_bytes": int(decoded.numel() * decoded.element_size()),
                    "tbe_v1_stored_bytes": 0,
                }
                record["runtime_kind"] = "bf16_input_embedding"
                record["runtime_dense_bytes"] = int(decoded.numel() * decoded.element_size())
                if writer is not None:
                    writer.add_dense(module_name, weight_name, _tcache.KIND_EMBEDDING, decoded,
                                     source_sha256=actual_sha)
                del decoded
            if cpu_vision_dense:
                record.update({
                    "component": "text", "source_kind": "verified_cpu_resolver",
                    "decoded_bytes": record["source_bf16_bytes"],
                    "source_bytes": record["source_bf16_bytes"],
                    "runtime_bytes": record.get("tbe_v1_resident_bytes", record.get("runtime_dense_bytes", 0)),
                })
                log(f"[cpu-vision-dense] tensor={weight_name} component=text decoded_bytes={record['decoded_bytes']} source_bytes={record['source_bf16_bytes']} runtime_bytes={record['runtime_bytes']} source_kind=verified_cpu_resolver runtime_kind={record['runtime_kind']}")
            setattr(parent, attr, replacement)
            receipts.append(record)
            if isinstance(original, BctxLinear):
                del container
            del original, replacement
            progress.add(items=1, nbytes=text_bytes.get(module_name, 0))
            if cpu_encode_pipeline:
                del encoded
    finally:
        # Upload/cache-write failures must also cancel and join our encoder jobs.
        encoded_rows.close()
    progress.finish()
    return receipts


def load_predictive_fast_session(
    *, package: str | Path, tune: str | Path, device: str = "cuda:0",
    max_len: int = 32768, stride: int = 1024, loader_workers: int = 1,
    history_slots: int = 16,
    metadata_cache_dir: str | Path | None = None,
    attention_implementation: str = "eager", log: Callable[[str], None] = print,
    verified_cpu_stream: bool = False, cpu_encode_workers: int = 1,
    cpu_encode_pipeline: bool = False,
    source_inflight_bytes: int = 8 * 1024**3,
    cpu_vision_dense: bool = False,
    transcode_cache: str | Path | None = None,
    transcode_cache_verify: str = "hash",
    verify_package: str = "auto",
    enable_mtp: bool = False,
    gdn_hv16: bool = False,
    runtime_format: str = "tbe",
    reuse_bhe_validation: bool = False, reuse_package_validation: bool = False,
    cpu_ready_install: bool = False,
    prefetch_mtp: bool = False,
    capture_single_session: bool = True,
):
    """Load the current predictive package into the TBE-v1 FastSession path.

    MTP is disabled by default; ``enable_mtp`` loads its original verified package tensors.
    No dense checkpoint is created; each
    selected predictive linear is validated, encoded, uploaded, then its original
    Context wrapper is unlinked from the model. The opt-in ``cpu_vision_dense``
    experiment requires ``verified_cpu_stream`` and materializes verified vision
    Context weights as dense runtime modules after text TBE conversion.

    ``transcode_cache`` (a directory, or None to disable) enables the persistent
    verified transcode cache; ``transcode_cache_verify`` is ``"hash"`` or ``"full"``.
    ``verify_package="auto"`` lets a complete cache bound to this package manifest
    replace the whole-package re-hash by non-frame hashing plus per-read frame
    checks (verified-CPU-stream path only); ``"full"`` always hashes every file.
    Every failure is raised as :class:`PredictiveTBEStartupError` naming its phase.
    """
    if runtime_format not in ('tbe', 'bhe1'):
        raise _fail('config', 'runtime_format must be tbe or bhe1')
    if type(prefetch_mtp) is not bool or type(capture_single_session) is not bool:
        raise _fail('config', 'MTP prefetch and single-session capture flags must be bools')
    if prefetch_mtp and (not enable_mtp or runtime_format != 'bhe1'
                         or not verified_cpu_stream or not cpu_encode_pipeline
                         or type(cpu_encode_workers) is not int
                         or not 1 < cpu_encode_workers <= MAX_CPU_ENCODE_WORKERS):
        raise _fail('config', 'prefetch_mtp requires MTP and the verified BHE CPU pipeline')
    if type(reuse_bhe_validation) is not bool:
        raise _fail('config', 'reuse_bhe_validation must be a bool')
    if type(reuse_package_validation) is not bool:
        raise _fail('config', 'reuse_package_validation must be a bool')
    if reuse_package_validation and (not verified_cpu_stream or verify_package != 'full'):
        raise _fail('config', 'reuse_package_validation requires verified CPU stream and --verify-package full')
    if type(cpu_ready_install) is not bool:
        raise _fail('config', 'cpu_ready_install must be a bool')
    if cpu_ready_install and (runtime_format != 'bhe1' or not verified_cpu_stream
                              or not cpu_encode_pipeline or type(cpu_encode_workers) is not int
                              or not 1 < cpu_encode_workers <= MAX_CPU_ENCODE_WORKERS):
        raise _fail('config', 'cpu_ready_install requires verified BHE1 CPU pipeline and multiple workers')
    if reuse_bhe_validation and (runtime_format != 'bhe1' or not verified_cpu_stream):
        raise _fail('config', 'reuse_bhe_validation requires BHE1 and verified_cpu_stream')
    if runtime_format == 'bhe1' and (transcode_cache is not None or cpu_vision_dense):
        raise _fail('config', 'BHE1 requires no TBE transcode cache and no dense vision conversion')
    start = time.perf_counter()
    timings: Dict[str, float] = {}
    if type(enable_mtp) is not bool:
        raise _fail("config", "enable_mtp must be a bool")
    if type(gdn_hv16) is not bool:
        raise _fail("config", "gdn_hv16 must be a bool")
    if type(cpu_vision_dense) is not bool:
        raise _fail("config", "cpu_vision_dense must be a bool")
    if cpu_vision_dense and not verified_cpu_stream:
        raise _fail("config", "cpu_vision_dense requires verified_cpu_stream")
    if type(cpu_encode_pipeline) is not bool:
        raise _fail("config", "cpu_encode_pipeline must be a bool")
    if cpu_encode_pipeline and (not verified_cpu_stream or type(cpu_encode_workers) is not int
                                or cpu_encode_workers <= 1):
        raise _fail("config", "cpu_encode_pipeline requires verified_cpu_stream and cpu_encode_workers > 1")
    if transcode_cache_verify not in _tcache.VERIFY_MODES:
        raise _fail("config", f"transcode_cache_verify must be one of {_tcache.VERIFY_MODES}")
    if verify_package not in _tcache.PACKAGE_VERIFY_MODES:
        raise _fail("config", f"verify_package must be one of {_tcache.PACKAGE_VERIFY_MODES}")
    with _phase("config", timings):
        deps = _session_deps()
        history_slots = deps.FastSession._validate_history_slots(history_slots, mtp_disabled=not enable_mtp)
        runtime = _predictive_runtime()
        load_predictive_model = runtime.load_predictive_model

        package = Path(package).expanduser().resolve()
        device = str(device)
        # Reject unsupported devices before opening the CPU resolver or model package.
        tbe_arch = deps.resolve_tbe_mma_arch(device)
        # The normal bundle loader initializes this before any TBE prefill. This
        # path bypasses that loader, so derive the arch here as well.
        tune_path = deps.resolve_tune(str(tune) if tune is not None else None, device)
        tune_file = Path(tune_path).expanduser().resolve()
        try:
            tune_sha256 = hashlib.sha256()
            with tune_file.open("rb") as tune_stream:
                for chunk in iter(lambda: tune_stream.read(1024 * 1024), b""):
                    tune_sha256.update(chunk)
        except OSError as exc:
            raise PredictiveTBEError(f"cannot read FastDecoder tune file {tune_file}: {exc}") from exc
        tune_sha256 = tune_sha256.hexdigest()
        device_identity = _device_identity(device, tbe_arch)

    bhe_kernel_executor = None
    bhe_kernel_future = None
    bhe_kernel_started = None
    bhe_kernel_build_record = None

    def abort_bhe_kernel_build():
        nonlocal bhe_kernel_executor, bhe_kernel_future
        if bhe_kernel_executor is not None:
            if bhe_kernel_future is not None:
                bhe_kernel_future.cancel()
            bhe_kernel_executor.shutdown(wait=True, cancel_futures=True)
            bhe_kernel_executor = None
            bhe_kernel_future = None

    def finish_bhe_kernel_build():
        nonlocal bhe_kernel_executor, bhe_kernel_future, bhe_kernel_build_record
        if bhe_kernel_executor is None:
            return
        try:
            with _phase("bhe_kernel_build", timings):
                build_result = bhe_kernel_future.result()
                if not isinstance(build_result, dict):
                    raise ValueError("BHE kernel compiler returned malformed build receipt")
                compile_seconds = float(build_result.get("compile_seconds"))
                if not math.isfinite(compile_seconds) or compile_seconds < 0:
                    raise ValueError("BHE kernel compiler returned invalid duration")
                capability = build_result.get("compute_capability")
                if (not isinstance(capability, list) or len(capability) != 2
                        or any(type(x) is not int or x < 0 for x in capability)):
                    raise ValueError("BHE kernel compiler returned invalid device capability")
                bhe_kernel_build_record = {
                    "status": "ready", "device": device,
                    "compute_capability": capability,
                    "compile_seconds": round(compile_seconds, 3),
                }
        finally:
            timings["bhe_kernel_build_elapsed_s"] = round(time.perf_counter() - bhe_kernel_started, 3)
            bhe_kernel_executor.shutdown(wait=True, cancel_futures=True)
            bhe_kernel_executor = None
            bhe_kernel_future = None

    if (runtime_format == "bhe1" and verified_cpu_stream and cpu_encode_pipeline
            and torch.device(device).type == "cuda"):
        try:
            capability = device_identity.get("capability")
            if capability and not os.environ.get("TORCH_CUDA_ARCH_LIST"):
                # Pin the selected device before any loading/build threads run.
                os.environ["TORCH_CUDA_ARCH_LIST"] = ".".join(map(str, capability))
            bhe_kernel_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bhe-kernel-build")
            bhe_kernel_started = time.perf_counter()
            log(f"[bhe-kernel-build] preparing BHE serving kernels for {device} during CPU load")
            bhe_kernel_future = bhe_kernel_executor.submit(_compile_bhe_kernel_for_device, device)
        except Exception as exc:
            abort_bhe_kernel_build()
            raise _fail("bhe_kernel_build", f"{type(exc).__name__}: {exc}") from exc

    if bhe_kernel_future is not None:
        original_log = log

        def log(message):
            # A failed compiler must not leave an expensive CPU conversion
            # running until the final join. Loader progress checks it early.
            future = bhe_kernel_future
            if future is not None and future.done() and not future.cancelled():
                failure = future.exception()
                if failure is not None:
                    raise _fail("bhe_kernel_build", f"{type(failure).__name__}: {failure}") from failure
            original_log(message)

    mtp_prefetch_controller = None
    if prefetch_mtp:
        from .startup_mtp_prefetch import _MTPPrefetchController
        mtp_prefetch_controller = _MTPPrefetchController()
    # Collect the loader's recursive tensor table before replacing wrappers.
    try:
        try:
            model, load_receipt, conversion, cpu_names = _load_and_convert_predictive(
                load_predictive_model, package=package, device=device,
                attention_implementation=attention_implementation, stride=stride,
                loader_workers=loader_workers, metadata_cache_dir=metadata_cache_dir,
                log=log, verified_cpu_stream=verified_cpu_stream,
                cpu_encode_workers=cpu_encode_workers,
                cpu_encode_pipeline=cpu_encode_pipeline,
                source_inflight_bytes=source_inflight_bytes,
                cpu_vision_dense=cpu_vision_dense,
                transcode_cache=transcode_cache, transcode_cache_verify=transcode_cache_verify,
                verify_package=verify_package, device_identity=device_identity, timings=timings,
                **({'reuse_package_validation': True} if reuse_package_validation else {}),
                **({'reuse_bhe_validation': True} if reuse_bhe_validation else {}),
                **({'cpu_ready_install': True} if cpu_ready_install else {}),
                **({'mtp_prefetch': mtp_prefetch_controller} if prefetch_mtp else {}),
                **({"runtime_format": runtime_format} if runtime_format != "tbe" else {}),
            )
            cache_meta = conversion.pop("transcode_cache")
            integrity = conversion.pop("package_integrity")
            gc.collect()
        except BaseException:
            abort_bhe_kernel_build()
            raise
        try:
            with _phase("placement", timings):
                placement_checked = check_runtime_placement(model, device, log=log)
        except BaseException:
            abort_bhe_kernel_build()
            raise
        try:
            with _phase("processor", timings):
                metadata_dir = Path(load_receipt["metadata_dir"])
                processor = deps.AutoProcessor.from_pretrained(str(metadata_dir), local_files_only=True)
                tokenizer = getattr(processor, "tokenizer", None)
                if tokenizer is None:
                    tokenizer = deps.AutoTokenizer.from_pretrained(str(metadata_dir), local_files_only=True)
        except BaseException:
            abort_bhe_kernel_build()
            raise

        mtp, mtp_receipt = None, None
        if enable_mtp:
            try:
                with _phase("mtp_load", timings):
                    from .predictive_mtp import load_package_mtp

                    if prefetch_mtp:
                        mtp, mtp_receipt = mtp_prefetch_controller.result()
                    else:
                        mtp, mtp_receipt = load_package_mtp(package, model.config, device=device)
            except BaseException:
                abort_bhe_kernel_build()
                raise

        # The BHE exponent/GEMM module must exist before FastDecoder builds BHE
        # descriptors. Waiting here overlaps its compile with CPU model work.
        finish_bhe_kernel_build()
    except BaseException:
        abort_bhe_kernel_build()
        raise
    with _phase("fastdecoder", timings):
        # Engine retains FastSession's request preparation and tokenizer behavior.
        loaded = SimpleNamespace(
            model=model, mtp=mtp, config=getattr(model, "config", None),
            local_dir=metadata_dir, receipt=load_receipt, options=None,
            streamer=None, pools=None, name_to_module={}, bundle=None, backend_label=runtime_format,
        )
        engine = deps.Engine(
            loaded, processor, tokenizer, max_batch=1, enable_mtp=enable_mtp,
            default_chat_template_kwargs={"enable_thinking": False}, log=log,
        )
        engine.tokenizer = tokenizer
        _align_engine_device(engine, model, device, log)
        tune_config = deps.load_tune(tune_path)
        fd_kwargs = dict(max_len=max_len, R=history_slots, device=device)
        if gdn_hv16:
            fd_kwargs["gdn_hv16"] = True
        fd = deps.FastDecoder(model, mtp, tune=tune_config, **fd_kwargs)
    meta: Dict[str, Any] = {
        "server_argv": ["--predictive-package", str(package), "--device", device,
                        "--exec-mode", "exact", "--mtp", "on" if enable_mtp else "off"],
        "load_s": round(time.perf_counter() - start, 1),
        "source_dir": str(metadata_dir),
        "tune_path": str(tune_file),
        "tune_sha256": tune_sha256,
        "predictive_runtime_module": runtime.__name__,
        "tbe_mma_arch": tbe_arch,
        "predictive_load_receipt": load_receipt,
        "predictive_to_tbe_conversion": conversion,
        "transcode_cache": cache_meta,
        "package_integrity": integrity,
        "placement_check": {"device": device, "tensors_checked": placement_checked},
        "untuned_shapes": dict(getattr(fd, "untuned_shapes", {}) or {}),
    }
    if gdn_hv16:
        meta["gdn_hv16"] = True
    if bhe_kernel_build_record is not None:
        meta["bhe_kernel_build"] = dict(bhe_kernel_build_record)
    if runtime_format != 'tbe':
        meta['runtime_format'] = runtime_format
        meta['predictive_to_bhe_conversion'] = meta.pop('predictive_to_tbe_conversion')
    if verified_cpu_stream:
        meta["verified_cpu_stream"] = {
            "enabled": True, "selected_tensor_count": len(cpu_names),
            "cpu_encode_workers": cpu_encode_workers,
            "cpu_encode_pipeline": cpu_encode_pipeline,
            "source_inflight_bytes": source_inflight_bytes,
            "resolver_cache_bytes": cpu_resolver_cache_limit(cpu_encode_workers),
            "phases": {"load_and_convert_seconds": round(time.perf_counter() - start, 3)},
        }
        if cpu_vision_dense:
            meta["verified_cpu_stream"].update({
                "cpu_vision_dense": True,
                "resolver_cache_bytes_before_drain": conversion.get("cpu_resolver_cache_bytes"),
            })
    with _phase("tune", timings):
        if fd.descriptor_kinds().get("tbe"):
            meta["tbe_tune"] = deps.tune_tbe_per_m(fd._descs(), log=log)
    if mtp_receipt is not None:
        meta["mtp_load_receipt"] = mtp_receipt
    dense_param_bytes = sum(int(p.numel() * p.element_size()) for p in model.parameters())
    if mtp is not None:
        dense_param_bytes += sum(int(p.numel() * p.element_size()) for p in mtp.parameters())
    dense_buffer_bytes = sum(int(b.numel() * b.element_size()) for b in model.buffers())
    descs = fd._descs()
    descriptor_bytes = sum(int(d.bytes) for d in descs)
    dense_descriptor_bytes = sum(int(d.bytes) for d in descs if getattr(d, "kind", None) == "bf16")
    dense_parameter_remainder = max(0, dense_param_bytes - dense_descriptor_bytes)
    preserved_context_bytes = int(conversion["preserved_context_resident_bytes"])
    source_model_bytes = sum(
        math.prod(int(d) for d in row["shape"]) * 2
        for row in load_receipt.get("consumed_tensors", {}).values()
        if isinstance(row, dict) and isinstance(row.get("shape"), list)
    )
    if mtp_receipt is not None:
        source_model_bytes += int(mtp_receipt["resident_dense_bytes"])
    total_runtime_weight_bytes = descriptor_bytes + dense_parameter_remainder + preserved_context_bytes
    meta["runtime_representation"] = {
        "tbe_fastdec_descriptor_bytes": descriptor_bytes,
        "dense_parameter_bytes": dense_param_bytes,
        "dense_descriptor_backing_bytes": dense_descriptor_bytes,
        "unrepresented_dense_parameter_bytes": dense_parameter_remainder,
        "preserved_predictive_context_bytes": preserved_context_bytes,
        "dense_buffer_bytes": dense_buffer_bytes,
        "total_weight_bytes": total_runtime_weight_bytes,
        "total_weight_and_buffer_bytes": total_runtime_weight_bytes + dense_buffer_bytes,
        "source_model_bf16_bytes": source_model_bytes,
        "gpu_representation_ratio": (
            source_model_bytes / max(1, total_runtime_weight_bytes)
        ),
        "gpu_representation_ratio_definition": (
            "BF16 source parameter bytes / (FastDecoder descriptor bytes + remaining dense "
            "parameter bytes + preserved predictive Context bytes); excludes workspace and allocator reserve"
        ),
    }
    if runtime_format == 'bhe1':
        representation = meta['runtime_representation']
        representation['bhe1_fastdec_descriptor_bytes'] = representation.pop('tbe_fastdec_descriptor_bytes')
    if cpu_vision_dense:
        meta["runtime_representation"].pop("gpu_representation_ratio", None)
        meta["runtime_representation"].pop("gpu_representation_ratio_definition", None)
        dense_consumed = load_receipt.get("consumed_tensors", {})
        runtime_parameters = dict(model.named_parameters())
        dense_norm_rows = []
        dense_vision_rows = []
        dense_other_rows = []
        converted_vision_names = {
            row["name"] for row in conversion.get("vision_dense_tensors", [])
            if isinstance(row, dict) and isinstance(row.get("name"), str)
        }
        for name, item in dense_consumed.items():
            if not isinstance(item, dict) or item.get("kind") not in ("dense_parameter", "dense_bias"):
                continue
            if name in converted_vision_names:
                continue
            shape = item.get("shape")
            if not isinstance(shape, list):
                continue
            source_bytes = math.prod(int(dim) for dim in shape) * 2
            param = runtime_parameters.get(name)
            runtime_bytes = int(param.numel() * param.element_size()) if param is not None else 0
            row = {"name": name, "source_kind": "verified_cpu_resolver",
                   "source_bytes": source_bytes, "decoded_bytes": source_bytes,
                   "runtime_bytes": runtime_bytes, "shape": shape,
                   "runtime_dtype": item.get("runtime_dtype")}
            lower = name.lower()
            if any(token in lower for token in ("norm", "layer_norm", "rmsnorm", "ln_")):
                row["component"] = "dense_norm"
                dense_norm_rows.append(row)
            elif any(part in {"visual", "vision", "vision_tower", "vision_model", "vision_encoder"}
                     for part in name.split(".")[:-1]):
                row["component"] = "dense_vision_parameter"
                dense_vision_rows.append(row)
            else:
                row["component"] = "dense_other_parameter"
                dense_other_rows.append(row)
            log(f"[cpu-vision-dense] tensor={name} component={row['component']} decoded_bytes={source_bytes} source_bytes={source_bytes} runtime_bytes={runtime_bytes} source_kind=verified_cpu_resolver runtime_kind=dense_parameter")
        meta["cpu_source_weight_ledger"] = {
            "cpu_vision_dense": True,
            "source_kind": "verified_cpu_resolver",
            "cpu_resolver_cache_bytes": conversion.get("cpu_resolver_cache_bytes"),
            "text_runtime": conversion.get("tensors", []),
            "vision_dense": conversion.get("vision_dense_tensors", []),
            "dense_norm": dense_norm_rows,
            "dense_vision_parameters": dense_vision_rows,
            "dense_other_parameters": dense_other_rows,
        }
    if history_slots < deps.MAX_M + 1:
        # The B1 greedy session replays verify1; capture it for reduced history.
        kinds = [("ar", 1), ("verify", 1)]
    else:
        kinds = [("ar", 1)] + [("verify", m) for m in range(1, deps.MAX_M + 1)]
    if enable_mtp:
        kinds += [("mtp", m) for m in range(1, deps.MAX_M + 1)]
    meta["history_slots"] = int(fd.R)
    meta["mtp_enabled"] = bool(fd.mtp_ready)
    meta["captured_kinds"] = [list(kind) for kind in kinds] if capture_single_session else []
    if not capture_single_session:
        meta['single_session_capture_skipped'] = True
    with _phase("capture", timings):
        meta["capture_s"] = fd.capture(kinds) if capture_single_session else {}
        if torch.device(device).type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
            meta["cuda_after_capture"] = {
                "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "startup_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
        meta["descriptors"] = fd.descriptor_kinds()
        meta["streamed_bytes_per_step"] = fd.streamed_bytes()
    meta["ready_s"] = round(time.perf_counter() - start, 1)
    meta["startup_phase_seconds"] = dict(timings)
    meta["startup_progress"] = startup_progress.snapshot()
    startup_progress.report("ready", ready_s=meta["ready_s"], load_s=meta["load_s"],
                            transcode_cache=cache_meta.get("status"),
                            package_integrity=integrity.get("package_integrity"))
    log(f"[context-tbe] ready in {meta['ready_s']} s (load {meta['load_s']} s, transcode cache "
        f"{cache_meta.get('status')}, package integrity {integrity.get('package_integrity')}); "
        f"{meta['descriptors']}")
    return deps.FastSession(engine, fd, meta)


__all__ = [
    "PredictiveTBEError", "PredictiveTBEStartupError", "check_runtime_placement",
    "convert_predictive_modules_to_tbe", "encode_predictive_weight", "load_predictive_fast_session",
]
