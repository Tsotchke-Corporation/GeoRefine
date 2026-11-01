"""``enable_hf_tbe_serving`` -- the TBE opt-in, mirroring ``enable_hf_fwp1_serving``.

``experiments.georefine._glc_serving_engine.enable_hf_fwp1_serving`` (added
2026-08-22) is an explicit opt-in boundary: refuse CPU/mixed-device models,
non-BF16 candidates, empty conversions, malformed converter receipts, and any
round-trip that was not measured bitwise; conversion is transactional, so a
later failure restores the original modules rather than leaving the model
half-converted.  This module is the same boundary for the lossless TBE
serving line, self-contained in ``release/glc_loader`` because nothing here
may import ``experiments.georefine`` (``tests/test_glc_release.py`` asserts
that for every ``*.py`` in this package).

DEFAULT OFF.  Nothing calls this function automatically -- a caller opts in
by importing and invoking it explicitly, exactly as
``georefine_fwp1_serving`` / ``enable_hf_fwp1_serving`` is default-off today.
The paired name for this surface is ``georefine_tbe_serving`` /
``enable_hf_tbe_serving``; see ``.icc/default_off_ledger.yaml``.

The conversion loop lives in :func:`_run_tbe_swap_linears`, a single
monkeypatchable unit, the same shape ``experiments.georefine.
_glc_serving_engine.enable_hf_fwp1_serving`` gets from ``_run_fwp1_swap_
linears`` -- so a CPU test can substitute a fake converter and exercise every
refusal, rollback, and receipt-shape path without a real CUDA device, exactly
as ``tests/test_glc_serving_engine.py`` already does for FWP1.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from .modules import GLCEmbedding, GLCLinear, GLCTiedLMHead
from .tbe_container import TBETensor, choose_window, encode_tbe, exponent_histogram, tbe_certify
from .tbe_device_map import (
    TBEDeviceMapError,
    devices_in_map,
    resolve_device,
    single_device_map,
)
from .tbe_mma import MMA_N, TILE, TBEMMAError, resolve_tbe_mma_arch
from .tbe_modules import (
    GLCTBEExpertBank,
    GLCTBELinear,
    TBEBackendError,
    TBEPoolRegistry,
    TBETransientPool,
)

TBE_SERVING_RECEIPT_SCHEMA = "georefine.tbe_serving_receipt.v1"

#: Names this codec refuses to convert -- the tied embedding/lm_head needs a
#: shared container with two consumers, which this pass does not build.
#: Verbatim policy match with ``experiments.georefine._glc_serving``.
_SKIP_NAME_SUBSTRINGS = ("embed", "lm_head", "wte", "wpe")

DEFAULT_MIN_NUMEL = 1_048_576


class TBEServingError(RuntimeError):
    """Raised when an explicitly requested TBE serving path cannot be proven."""

    def __init__(self, receipt: Dict[str, Any]):
        self.receipt = dict(receipt)
        failure = self.receipt.get("failure") or {}
        reason = str(failure.get("reason") or "unknown")
        detail = str(failure.get("detail") or "")
        message = f"TBE serving enablement failed: {reason}"
        if detail:
            message += f": {detail}"
        super().__init__(message)


def _raise_tbe_serving_error(
    receipt: Dict[str, Any], *, reason: str, detail: str,
    cause: Optional[Exception] = None,
) -> None:
    failed = {
        **receipt,
        "status": "failed",
        "failure": {
            "reason": str(reason),
            "detail": str(detail),
            "error_type": type(cause).__name__ if cause is not None else None,
        },
    }
    error = TBEServingError(failed)
    if cause is not None:
        raise error from cause
    raise error


def _eligible_tbe_linears(
    model: nn.Module, min_numel: int,
) -> Tuple[List[Tuple[nn.Module, str, str, nn.Linear]], int, int, int]:
    """``(candidates, n_tiny, n_name_skipped, n_shape_skipped)``.

    A candidate is a plain ``nn.Linear`` (not already a coded module of any
    kind) with a 2-D weight, ``numel >= min_numel``, a name that does not
    match ``_SKIP_NAME_SUBSTRINGS``, and a shape the fragment kernel's tiling
    can serve: ``in_features % TILE == 0`` and ``out_features % MMA_N == 0``.
    """
    candidates: List[Tuple[nn.Module, str, str, nn.Linear]] = []
    n_tiny = 0
    n_name = 0
    n_shape = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if isinstance(module, (GLCLinear, GLCTBELinear)):
            continue
        w = module.weight
        if w is None or w.dim() != 2:
            continue
        lname = name.lower()
        if any(s in lname for s in _SKIP_NAME_SUBSTRINGS):
            n_name += 1
            continue
        if int(w.numel()) < int(min_numel):
            n_tiny += 1
            continue
        out_f, in_f = int(w.shape[0]), int(w.shape[1])
        if in_f % TILE or out_f % MMA_N:
            n_shape += 1
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        candidates.append((parent, attr, name, module))
    return candidates, n_tiny, n_name, n_shape


def _eligible_tbe_expert_banks(
    model: nn.Module, min_numel: int,
) -> Tuple[List[Tuple[nn.Module, str, str, torch.Tensor]], int, int]:
    """``(banks, n_tiny_skipped, n_shape_skipped)`` -- fused 3-D MoE weights.

    A mixture-of-experts FFN in transformers is NOT a set of ``nn.Linear``
    modules: ``Qwen3NextExperts`` / ``Qwen3_5MoeExperts`` (and the wider
    family the Flash-Next generation follows) store ``gate_up_proj``
    ``[E, 2 * intermediate, hidden]`` and ``down_proj`` ``[E, hidden,
    intermediate]`` as plain 3-D ``nn.Parameter`` tensors and index them per
    routed expert.  :func:`_eligible_tbe_linears` cannot see a single byte of
    that, so on a 512-expert model it would code the attention projections
    and leave the bulk of the weight dense.

    Eligibility is per EXPERT SLICE and uses the identical rule the linear
    scan uses -- 2-D, bf16, ``in % TILE == 0``, ``out % MMA_N == 0``,
    ``numel >= min_numel`` -- because each slice is exactly a ``[out, in]``
    weight matrix, the same thing ``F.linear`` is handed.  A parameter whose
    slices do not qualify is left alone (kept bf16), never partially coded.
    """
    banks: List[Tuple[nn.Module, str, str, torch.Tensor]] = []
    n_tiny = 0
    n_shape = 0
    for module_name, module in model.named_modules():
        for attr, param in list(module.named_parameters(recurse=False)):
            if param is None or param.dim() != 3:
                continue
            name = f"{module_name}.{attr}" if module_name else attr
            if any(s in name.lower() for s in _SKIP_NAME_SUBSTRINGS):
                continue
            e, out_f, in_f = (int(v) for v in param.shape)
            if e < 1:
                continue
            if int(out_f) * int(in_f) < int(min_numel):
                n_tiny += 1
                continue
            if in_f % TILE or out_f % MMA_N:
                n_shape += 1
                continue
            banks.append((module, attr, name, param))
    return banks, n_tiny, n_shape


def _tbe_candidate_device(
    candidates: List[Tuple[nn.Module, str, str, nn.Linear]],
) -> torch.device:
    devices = {linear.weight.device for _, _, _, linear in candidates}
    if len(devices) != 1:
        rendered = sorted(str(device) for device in devices)
        raise ValueError(
            "TBE serving requires one device for every convertible linear; "
            f"found {rendered}"
        )
    return next(iter(devices))


def _standalone_composition_markers(model: nn.Module) -> List[str]:
    """Names of pre-existing coded surfaces TBE must not be layered over.

    Mirrors ``hf_runtime_export.py``'s refusal of FWP1-over-standalone-GLC:
    a model that already carries GLC-FWP1 modules, or an FWP1 serving
    receipt, has already committed its linears to a different compressed
    representation, and swapping them again would either double-encode or
    silently discard the first conversion.
    """
    markers: List[str] = []
    if getattr(model, "georefine_fwp1_serving_receipt", None) is not None:
        markers.append("georefine_fwp1_serving_receipt")
    if getattr(model, "georefine_tbe_serving_receipt", None) is not None:
        markers.append("georefine_tbe_serving_receipt")
    for module in model.modules():
        if isinstance(module, (GLCLinear, GLCTiedLMHead, GLCEmbedding)):
            markers.append(f"{type(module).__name__} module present")
            break
    return markers


def _encode_on_device(
    w: torch.Tensor, *, layout: str, device: torch.device,
) -> TBETensor:
    """Encode ``w`` into a container whose EVERY array lives on ``device``.

    The codec (``tbe_container``) is allocator-agnostic by construction: its
    accumulators (``planes_out``/``smb_out``/``per_tile``/``sbbase``) come
    from bare ``torch.empty``/``torch.zeros``, i.e. the AMBIENT default
    device, while ``esc`` is sliced out of the weight and therefore follows
    ``w.device``.  Handed a CUDA-resident weight with a CPU default device --
    which is exactly what serving a live model does -- it returns a MIXED
    container: ``esc`` on cuda, everything else on cpu.  The first thing that
    trips over such a container is ``upload_tbe``'s
    ``torch.cat([esc, torch.zeros(ESC_PAD)])``, which raises "Expected all
    tensors to be on the same device, but found at least two devices, cuda:0
    and cpu!" -- the failure that refused the 27B ``resident_vram`` stage on
    2026-09-02.

    The fix belongs here, in the serving conversion, because this is the
    layer that OWNS ``device``; the codec is vendored byte-for-byte from the
    research module and its CPU-input behaviour must not move.  Running the
    encode under ``torch.device(device)`` puts the ambient default device
    where the weight already is, so accumulator and escape arrays agree and
    the container is uniformly device-resident.

    ``mode``/``base`` are chosen BEFORE entering the context and passed in
    explicitly: ``choose_window(exponent_histogram(w))`` is the identical
    selection ``encode_tbe`` performs internally when they are ``None``
    (same default ``target_elems``), but ``exponent_histogram`` accumulates
    into a bare ``torch.zeros(256)`` and adds an explicitly ``.cpu()``
    histogram to it, so it -- and only it -- requires a CPU default device.
    Hoisting it also saves the second full pass over the weight.
    """
    with torch.device("cpu"):
        mode, base, _escape_rate = choose_window(exponent_histogram(w))
    with torch.device(device):
        return encode_tbe(w, layout=layout, mode=mode, base=base)


def _same_device(a: Any, b: Any) -> bool:
    """Device equality that tolerates an unindexed spelling of the same device.

    ``torch.device("mps") != torch.device("mps:0")`` and
    ``torch.device("cuda") != torch.device("cuda:0")`` compare unequal even
    though a tensor placed by either lands in the same memory; an index of
    ``None`` means "the current device of this type", so it matches.
    """
    da, db = torch.device(a), torch.device(b)
    if da.type != db.type:
        return False
    if da.index is None or db.index is None:
        return True
    return da.index == db.index


def _assert_container_on_device(
    c: TBETensor, device: torch.device, name: str,
) -> None:
    """Fail closed on any container array that is not on ``device``.

    :func:`_encode_on_device` gets its placement from an ambient default
    device, which is an implicit mechanism; this turns it into a checked
    invariant, so a placement regression surfaces here -- named, per tensor --
    instead of downstream as an opaque cross-device error out of ``cat``.
    """
    want = torch.device(device)
    off = {
        field: str(getattr(c, field).device)
        for field in ("planes", "smb", "esc", "sbbase")
        if not _same_device(getattr(c, field).device, want)
    }
    if off:
        raise TBEBackendError(
            f"{name}: encoded container is not uniformly resident on {want}; "
            f"off-device arrays: {off}. A mixed container cannot be uploaded."
        )


def _run_tbe_swap_linears(
    candidates: List[Tuple[nn.Module, str, str, nn.Linear]],
    device: torch.device,
    *,
    layout: str,
    pool: TBETransientPool,
) -> Dict[str, Any]:
    """Encode, certify, and install one :class:`GLCTBELinear` per candidate.

    Mutates ``candidates`` in place (``setattr(parent, attr, tbe_linear)``)
    exactly as far as it gets, then returns a receipt fragment.  It never
    rolls back partial work itself -- that is ``enable_hf_tbe_serving``'s job,
    which holds the pre-conversion snapshots this function does not see. Any
    exception it raises is caught by the caller and treated as a full
    ``conversion_error``.
    """
    n_swapped = 0
    n_bitwise_verified = 0
    n_bitwise_failed = 0
    resident_bytes = 0
    dense_bytes = 0
    per_linear: List[Dict[str, Any]] = []
    for parent, attr, name, linear in candidates:
        w = linear.weight.data
        container = _encode_on_device(w, layout=layout, device=device)
        _assert_container_on_device(container, device, name)
        with torch.device(device):
            check = tbe_certify(container, w)
        if not check["bitwise_ok"]:
            n_bitwise_failed += 1
            per_linear.append({
                "name": name, "bitwise_ok": False,
                "mismatched_elements": check["mismatched_elements"],
            })
            continue
        n_bitwise_verified += 1
        bias = None if linear.bias is None else linear.bias.data
        with torch.device(device):
            tbe_linear = GLCTBELinear(container, bias, device, pool)
        uploaded_device = getattr(
            getattr(tbe_linear, "container", None), "device", None,
        )
        if uploaded_device is not None and not _same_device(uploaded_device, device):
            raise TBEBackendError(
                f"{name}: uploaded container landed on "
                f"{uploaded_device}, not {device}"
            )
        setattr(parent, attr, tbe_linear)
        n_swapped += 1
        resident_bytes += tbe_linear.resident_bytes
        dense_bytes += tbe_linear.dense_bytes
        per_linear.append({
            "name": name, "bitwise_ok": True,
            "resident_bytes": tbe_linear.resident_bytes,
            "dense_bytes": tbe_linear.dense_bytes,
            "escape_rate": container.escape_rate(),
        })
    return {
        "n_swapped": n_swapped,
        "n_bitwise_verified": n_bitwise_verified,
        "n_bitwise_failed": n_bitwise_failed,
        "all_bitwise_ok": n_bitwise_failed == 0,
        "resident_bytes": resident_bytes,
        "dense_bytes": dense_bytes,
        "ratio": (dense_bytes / resident_bytes) if resident_bytes else 1.0,
        "per_linear": per_linear,
    }


def _run_tbe_swap_expert_banks(
    banks: List[Tuple[nn.Module, str, str, torch.Tensor]],
    device: torch.device,
    *,
    layout: str,
    pool: TBETransientPool,
) -> Dict[str, Any]:
    """Encode, certify, and install one :class:`GLCTBEExpertBank` per parameter.

    Same contract as :func:`_run_tbe_swap_linears`: mutates as far as it
    gets, never rolls back itself, returns a receipt fragment.  Every expert
    slice is certified bit-exact individually before ANY of them is
    installed, so a bank is either fully certified or not installed at all --
    a partially-coded expert set would serve some tokens from a container
    that was never checked.
    """
    n_swapped = 0
    n_experts_coded = 0
    n_bitwise_failed = 0
    resident_bytes = 0
    dense_bytes = 0
    per_bank: List[Dict[str, Any]] = []
    for module, attr, name, param in banks:
        containers: List[TBETensor] = []
        failed_expert = None
        n_experts = int(param.shape[0])
        for e in range(n_experts):
            w = param.data[e].contiguous()
            container = _encode_on_device(w, layout=layout, device=device)
            _assert_container_on_device(container, device, f"{name}[{e}]")
            with torch.device(device):
                check = tbe_certify(container, w)
            if not check["bitwise_ok"]:
                failed_expert = e
                break
            containers.append(container)
        if failed_expert is not None:
            n_bitwise_failed += 1
            per_bank.append({
                "name": name, "bitwise_ok": False,
                "failed_expert": failed_expert, "num_experts": n_experts,
            })
            continue
        with torch.device(device):
            bank = GLCTBEExpertBank(containers, device, pool)
        # Replacing a Parameter with a Module needs the parameter slot
        # cleared first: nn.Module.__setattr__ refuses to shadow a registered
        # parameter with a submodule, and leaving both would keep the dense
        # tensor resident forever -- the exact opposite of the point.
        del module._parameters[attr]
        setattr(module, attr, bank)
        n_swapped += 1
        n_experts_coded += n_experts
        resident_bytes += bank.resident_bytes
        dense_bytes += bank.dense_bytes
        per_bank.append({
            "name": name, "bitwise_ok": True, "num_experts": n_experts,
            "resident_bytes": bank.resident_bytes,
            "dense_bytes": bank.dense_bytes,
        })
    return {
        "n_expert_banks_swapped": n_swapped,
        "n_experts_coded": n_experts_coded,
        "n_expert_banks_bitwise_failed": n_bitwise_failed,
        "expert_all_bitwise_ok": n_bitwise_failed == 0,
        "expert_resident_bytes": resident_bytes,
        "expert_dense_bytes": dense_bytes,
        "per_expert_bank": per_bank,
    }


_MERGE_SUM_KEYS = (
    "n_swapped", "n_bitwise_verified", "n_bitwise_failed",
    "resident_bytes", "dense_bytes",
    "n_expert_banks_swapped", "n_experts_coded",
    "n_expert_banks_bitwise_failed",
    "expert_resident_bytes", "expert_dense_bytes",
)


def _merge_converter_receipts(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sum per-device converter fragments into ONE model-level receipt.

    A multi-device conversion runs :func:`_run_tbe_swap_linears` once per
    device (each with that device's own pool), so the model-level numbers
    have to be summed rather than taken from whichever fragment came last.
    The single-device shape is preserved exactly -- merging one fragment
    returns the same keys with the same values, plus zeroed expert fields --
    so the receipt schema does not fork between the one-card and eight-card
    paths.  ``ratio`` stays the LINEAR ratio it has always been; the coded
    total across linears and expert banks is reported separately as
    ``total_ratio``, because silently redefining a field that receipts
    already quote would make two runs incomparable.
    """
    total: Dict[str, Any] = {k: 0 for k in _MERGE_SUM_KEYS}
    total["per_linear"] = []
    total["per_expert_bank"] = []
    for part in parts:
        for key in _MERGE_SUM_KEYS:
            total[key] += int(part.get(key) or 0)
        total["per_linear"].extend(part.get("per_linear") or [])
        total["per_expert_bank"].extend(part.get("per_expert_bank") or [])
    total["all_bitwise_ok"] = total["n_bitwise_failed"] == 0
    total["expert_all_bitwise_ok"] = total["n_expert_banks_bitwise_failed"] == 0
    total["ratio"] = (
        (total["dense_bytes"] / total["resident_bytes"])
        if total["resident_bytes"] else 1.0
    )
    resident = total["resident_bytes"] + total["expert_resident_bytes"]
    dense = total["dense_bytes"] + total["expert_dense_bytes"]
    total["total_resident_bytes"] = resident
    total["total_dense_bytes"] = dense
    total["total_ratio"] = (dense / resident) if resident else 1.0
    return total


def _normalize_device_map(
    device_map: Any, model: nn.Module,
) -> Dict[str, str]:
    """Accept a dict, a single device, or an ``hf_device_map`` and canonicalise.

    ``"auto"`` is deliberately NOT accepted here.  Planning "auto" needs the
    coded byte sizes, which live in the transcode manifest, not in the live
    model; ``tbe_device_map.plan_device_map_from_manifest`` is where that
    decision is made, and it is made from measured bytes before any weight is
    touched.  Accepting the string here would mean guessing.
    """
    if isinstance(device_map, str) and device_map.strip().lower() == "auto":
        raise TBEDeviceMapError(
            "device_map='auto' is not resolvable at conversion time: plan it "
            "with tbe_device_map.plan_device_map_from_manifest() (which reads "
            "measured coded bytes) and pass the resulting dict"
        )
    if isinstance(device_map, Mapping):
        if not device_map:
            raise TBEDeviceMapError("device_map is empty")
        return {str(k): str(torch.device(v)) for k, v in device_map.items()}
    return single_device_map(device_map)


def _group_by_device(
    items: List[Tuple[Any, str, str, Any]], device_map: Dict[str, str],
) -> "Dict[str, List[Tuple[Any, str, str, Any]]]":
    """Bucket ``(parent, attr, name, obj)`` tuples by their mapped device."""
    groups: Dict[str, List[Tuple[Any, str, str, Any]]] = {}
    for item in items:
        name = item[2]
        dev = resolve_device(name, device_map)
        groups.setdefault(str(dev), []).append(item)
    return groups


def _coded_parameter_names(model: nn.Module) -> set:
    """Names whose storage is already owned by a coded module."""
    owned = set()
    for name, module in model.named_modules():
        if isinstance(module, (GLCTBELinear, GLCTBEExpertBank)):
            owned.add(name)
    return owned


def place_uncoded_tensors(
    model: nn.Module, device_map: Any,
) -> Dict[str, Any]:
    """Move every tensor the codec did NOT take onto its mapped device.

    Embeddings, norms, router/gate weights and biases are never coded -- the
    router in particular decides which expert a token reaches, and a router
    is a tiny bf16 matrix whose exactness the whole MoE routing depends on --
    but they still have to land on the device that owns their layer, or the
    first forward pays a host round-trip per token.  Coded modules are
    skipped: their storage is the container, which the conversion already
    placed.
    """
    dmap = _normalize_device_map(device_map, model)
    owned = _coded_parameter_names(model)
    moved = 0
    per_device: Dict[str, int] = {}
    for module_name, module in model.named_modules():
        if any(
            module_name == o or module_name.startswith(o + ".") for o in owned
        ):
            continue
        for attr, param in list(module.named_parameters(recurse=False)):
            if param is None:
                continue
            name = f"{module_name}.{attr}" if module_name else attr
            dev = resolve_device(name, dmap)
            if param.device != dev:
                module._parameters[attr] = nn.Parameter(
                    param.data.to(dev), requires_grad=param.requires_grad,
                )
                moved += 1
            per_device[str(dev)] = per_device.get(str(dev), 0) + 1
        for attr, buf in list(module.named_buffers(recurse=False)):
            if buf is None:
                continue
            name = f"{module_name}.{attr}" if module_name else attr
            dev = resolve_device(name, dmap)
            if buf.device != dev:
                module._buffers[attr] = buf.to(dev)
                moved += 1
            per_device[str(dev)] = per_device.get(str(dev), 0) + 1
    return {"n_moved": moved, "per_device_tensor_counts": per_device}


def enable_hf_tbe_serving(
    model: nn.Module,
    *,
    min_numel: int = DEFAULT_MIN_NUMEL,
    layout: str = "mma16",
    device_map: Any = None,
    experts: bool = False,
) -> nn.Module:
    """Install certified TBE linears behind the fragment-kernel serving path.

    This is an explicit opt-in boundary, mirroring ``enable_hf_fwp1_serving``:
    it refuses CPU/mixed-device models, non-BF16 candidates, empty
    conversions, an unsupported compute capability, and any round-trip that
    was not measured bitwise.  Conversion is transactional: original modules
    are restored if a later encode, upload, certification, or arch-resolution
    step fails.  There is no dense fallback anywhere in this function --
    every failure raises :class:`TBEServingError` with a typed reason rather
    than silently degrading to a slower or larger serving path.

    MULTI-DEVICE.  ``device_map`` is an accelerate-shaped ``{module_name:
    device}`` mapping (see :mod:`.tbe_device_map`), or a single device, or
    ``None`` -- which keeps the historical behaviour exactly: every
    convertible linear must already share ONE device, and mixed placement is
    refused.  With a map, candidates are grouped by their mapped device and
    each group is converted on its own device with its OWN transient pool
    (:class:`TBEPoolRegistry`); nothing is ever allocated on one card for a
    linear that lives on another.  ``"auto"`` is not accepted here -- see
    :func:`_normalize_device_map`.

    ``experts=True`` additionally codes fused 3-D MoE expert parameters
    (:func:`_eligible_tbe_expert_banks`), which no ``nn.Linear`` scan can
    see and which are the majority of a 512-expert model's weight.  It is
    opt-in because installing a :class:`GLCTBEExpertBank` changes what
    ``self.gate_up_proj[e]`` returns from a Parameter view to a decoded pool
    view: correct for the per-routed-expert loop transformers ships, and a
    loud failure -- never a silent mis-decode -- for any implementation that
    consumes the whole 3-D tensor at once.
    """
    if isinstance(min_numel, bool) or not isinstance(min_numel, int):
        raise TypeError("min_numel must be an integer")
    if min_numel < 0:
        raise ValueError("min_numel must be >= 0")
    if layout != "mma16":
        raise ValueError(
            f"the TBE fragment kernel requires layout='mma16', got {layout!r}"
        )

    candidates, n_tiny, n_name, n_shape = _eligible_tbe_linears(model, min_numel)
    banks: List[Tuple[nn.Module, str, str, torch.Tensor]] = []
    n_bank_tiny = n_bank_shape = 0
    if experts:
        banks, n_bank_tiny, n_bank_shape = _eligible_tbe_expert_banks(
            model, min_numel,
        )
    base_receipt: Dict[str, Any] = {
        "schema": TBE_SERVING_RECEIPT_SCHEMA,
        "status": "preflight",
        "requested": True,
        "min_numel": int(min_numel),
        "layout": layout,
        "certify": True,
        "n_tiny_skipped": int(n_tiny),
        "n_name_skipped": int(n_name),
        "n_shape_skipped": int(n_shape),
        "experts_requested": bool(experts),
        "n_expert_banks_found": len(banks),
        "n_expert_banks_tiny_skipped": int(n_bank_tiny),
        "n_expert_banks_shape_skipped": int(n_bank_shape),
        "failure": None,
    }

    markers = _standalone_composition_markers(model)
    if markers:
        _raise_tbe_serving_error(
            base_receipt,
            reason="standalone_composition_forbidden",
            detail=(
                "TBE serving cannot be layered over an already-coded model: "
                + ", ".join(markers)
            ),
        )

    if not candidates and not banks:
        _raise_tbe_serving_error(
            base_receipt,
            reason="no_eligible_linears",
            detail="no BF16 TBE shape candidate is available at this threshold",
        )

    if device_map is None:
        try:
            device = _tbe_candidate_device(candidates)
        except Exception as exc:
            _raise_tbe_serving_error(
                base_receipt,
                reason="mixed_or_unknown_device",
                detail=str(exc),
                cause=exc,
            )
        dmap = single_device_map(device)
        base_receipt["device"] = str(device)
    else:
        try:
            dmap = _normalize_device_map(device_map, model)
        except TBEDeviceMapError as exc:
            _raise_tbe_serving_error(
                base_receipt,
                reason="invalid_device_map",
                detail=str(exc),
                cause=exc,
            )
        mapped = devices_in_map(dmap)
        base_receipt["device"] = (
            str(mapped[0]) if len(mapped) == 1 else "multi"
        )
    base_receipt["device_map"] = dict(dmap)

    try:
        linear_groups = _group_by_device(candidates, dmap)
        bank_groups = _group_by_device(banks, dmap)
    except TBEDeviceMapError as exc:
        _raise_tbe_serving_error(
            base_receipt,
            reason="unmapped_module",
            detail=str(exc),
            cause=exc,
        )
    device_names = list(linear_groups) + [
        d for d in bank_groups if d not in linear_groups
    ]
    devices = [torch.device(d) for d in device_names]
    base_receipt["devices"] = [str(d) for d in devices]

    non_cuda = [str(d) for d in devices if d.type != "cuda"]
    if non_cuda:
        _raise_tbe_serving_error(
            base_receipt,
            reason="non_cuda_device",
            detail=f"TBE serving requires CUDA; candidates are on {non_cuda[0]}",
        )

    non_bf16 = [
        name for _, _, name, linear in candidates
        if linear.weight.dtype != torch.bfloat16
    ] + [
        name for _, _, name, param in banks
        if param.dtype != torch.bfloat16
    ]
    if non_bf16:
        _raise_tbe_serving_error(
            base_receipt,
            reason="non_bfloat16_weights",
            detail="non-BF16 candidate linears: " + ", ".join(non_bf16[:8]),
        )

    arch_by_device: Dict[str, str] = {}
    for dev in devices:
        try:
            arch_by_device[str(dev)] = resolve_tbe_mma_arch(dev)
        except TBEMMAError as exc:
            _raise_tbe_serving_error(
                base_receipt,
                reason="unsupported_device_capability",
                detail=str(exc),
                cause=exc,
            )
    archs = sorted(set(arch_by_device.values()))
    base_receipt["arch"] = archs[0] if len(archs) == 1 else archs
    base_receipt["arch_by_device"] = dict(arch_by_device)

    linear_snapshots = [
        (parent, attr, linear) for parent, attr, _, linear in candidates
    ]
    bank_snapshots = [
        (module, attr, param) for module, attr, _, param in banks
    ]

    def rollback() -> None:
        for parent, attr, linear in linear_snapshots:
            setattr(parent, attr, linear)
        for module, attr, param in bank_snapshots:
            # Whatever the conversion left in the slot -- a coded bank, or a
            # half-installed module from a converter that failed after
            # mutating -- has to be cleared from BOTH registries before the
            # original Parameter can go back: register_parameter refuses to
            # shadow an existing attribute.
            module._modules.pop(attr, None)
            module._buffers.pop(attr, None)
            module._parameters.pop(attr, None)
            module.__dict__.pop(attr, None)
            module.register_parameter(attr, param)

    registry = TBEPoolRegistry()
    per_device_receipts: Dict[str, Any] = {}
    try:
        converter_receipt = _merge_converter_receipts([])
        for dev_name, group in linear_groups.items():
            dev = torch.device(dev_name)
            part = _run_tbe_swap_linears(
                group, dev, layout=layout,
                pool=registry.pool_for(dev, "linear"),
            )
            per_device_receipts.setdefault(dev_name, {}).update(part)
            converter_receipt = _merge_converter_receipts(
                [converter_receipt, part],
            )
        for dev_name, group in bank_groups.items():
            dev = torch.device(dev_name)
            part = _run_tbe_swap_expert_banks(
                group, dev, layout=layout,
                pool=registry.pool_for(dev, "experts"),
            )
            per_device_receipts.setdefault(dev_name, {}).update(part)
            converter_receipt = _merge_converter_receipts(
                [converter_receipt, part],
            )
    except Exception as exc:
        rollback()
        registry.free_all()
        _raise_tbe_serving_error(
            base_receipt,
            reason="conversion_error",
            detail=f"{type(exc).__name__}: {exc}",
            cause=exc,
        )

    receipt = {
        **base_receipt,
        **converter_receipt,
        "schema": TBE_SERVING_RECEIPT_SCHEMA,
        "status": "converted",
        "transient_pool_capacity_bytes": registry.total_capacity_bytes,
        "transient_pool_capacity_bytes_by_device": (
            registry.capacity_bytes_by_device()
        ),
        "per_device": per_device_receipts,
        "failure": None,
    }
    n_swapped = receipt.get("n_swapped")
    n_banks = receipt.get("n_expert_banks_swapped") or 0
    certification_ok = (
        isinstance(n_swapped, int)
        and not isinstance(n_swapped, bool)
        and (n_swapped + int(n_banks)) > 0
        and receipt.get("all_bitwise_ok") is True
        and receipt.get("n_bitwise_failed") == 0
        and receipt.get("expert_all_bitwise_ok", True) is True
        and (receipt.get("n_expert_banks_bitwise_failed") or 0) == 0
    )
    if not certification_ok:
        rollback()
        registry.free_all()
        reason = (
            "no_eligible_linears" if (n_swapped == 0 and not n_banks)
            else "certification_error"
        )
        _raise_tbe_serving_error(
            receipt, reason=reason,
            detail="converter did not return a positive fully certified swap",
        )

    receipt["status"] = "enabled"
    model.georefine_tbe_serving_receipt = receipt
    model.georefine_tbe_pool_registry = registry
    return model


__all__ = [
    "DEFAULT_MIN_NUMEL",
    "TBE_SERVING_RECEIPT_SCHEMA",
    "TBEServingError",
    "enable_hf_tbe_serving",
    "place_uncoded_tensors",
]
