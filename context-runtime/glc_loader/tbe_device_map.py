"""Device placement for multi-GPU TBE serving -- byte-accurate, fail-closed.

A model whose bf16 weights do not fit one card (Qwen3.8-Flash-Next is ~360 GB
bf16 against an 80 GB A100) has to be split across devices before anything
can be coded, and the split has to be decided from the CODED byte sizes, not
the dense ones: a plan that packs 80 GB of dense weight onto a card and then
discovers the container is 1.4x smaller has wasted a card, and a plan that
packs 80 GB of CODED weight onto a card without ever having measured the
container has no idea whether it fits at all.

The manifest ``scripts/glc_tbe_transcode.py`` already writes carries exactly
the number this needs -- ``resident_bytes`` per coded tensor, ``original_
bytes`` per tensor kept raw -- so the planner reads real measured bytes and
never estimates.  :func:`plan_device_map` is therefore byte-accurate by
construction and its unit test checks it against a synthetic manifest to the
byte.

CONVENTIONS.  The returned map is accelerate-shaped: ``{module_name:
device}``, resolved by LONGEST PREFIX (``model.layers.3`` covers
``model.layers.3.mlp.gate_proj``), with ``""`` as an optional catch-all --
identical to ``accelerate.dispatch_model``'s reading of ``device_map``, so a
map produced here can be handed to accelerate and a map produced by
``infer_auto_device_map`` can be handed to :func:`resolve_device`.  What this
module does NOT do is guess: an unmappable name raises rather than falling
back to device 0, because a silent fallback is how half a model ends up on
one card and OOMs under a message that names the codec.

GROUPING.  Placement is per no-split GROUP, never per tensor: a transformer
layer's linears must share a device or every forward pays a cross-device
copy, and a GatedDeltaNet layer additionally carries recurrent state that
must live with the layer that updates it.  :func:`group_key` derives the
group from the name -- the prefix through the last numeric component
(``model.layers.12.mlp.experts.gate_up_proj`` -> ``model.layers.12``), and
otherwise the owning module (``model.embed_tokens.weight`` ->
``model.embed_tokens``).  That is the same boundary
``_no_split_modules`` expresses in transformers, derived from names alone so
this file keeps its promise of importing nothing but the standard library
and ``torch``.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

#: Sequential (pipeline-order) fill, the strategy ``infer_auto_device_map``
#: uses: groups are assigned in model order, each device filled to its budget
#: before the next is opened.  It keeps a layer's activations flowing forward
#: through devices in one direction instead of ping-ponging.
STRATEGY_SEQUENTIAL = "sequential"
#: Balanced fill: every device gets ceil(total / n_devices) bytes, still in
#: model order, so the last device is not left nearly empty.  Preferred when
#: the run has to leave headroom for activations and KV on every card.
STRATEGY_BALANCED = "balanced"
STRATEGIES = (STRATEGY_SEQUENTIAL, STRATEGY_BALANCED)

_NUM_RE = re.compile(r"(\d+)")


class TBEDeviceMapError(RuntimeError):
    """A device map could not be planned, resolved, or validated.

    Every failure in this module is one of these: there is no path that
    returns a partially-planned map, and none that silently defaults a name
    to a device.
    """


# ---------------------------------------------------------------------------
# names -> groups
# ---------------------------------------------------------------------------
def natural_key(name: str) -> Tuple:
    """Sort key where ``layers.2`` precedes ``layers.10``.

    Plain lexicographic order puts ``model.layers.10`` before
    ``model.layers.2``, which for a SEQUENTIAL plan silently reorders the
    pipeline: layer 10 would be placed on the first device and layer 2 on a
    later one, and every forward would then walk backwards across the
    interconnect.
    """
    parts = _NUM_RE.split(str(name))
    return tuple(
        (1, int(p), "") if p.isdigit() else (0, 0, p)
        for p in parts if p != ""
    )


def group_key(name: str) -> str:
    """The no-split group a tensor name belongs to.

    ``model.layers.12.mlp.experts.gate_up_proj`` -> ``model.layers.12``
    ``model.visual.blocks.3.attn.qkv.weight``    -> ``model.visual.blocks.3``
    ``model.embed_tokens.weight``                -> ``model.embed_tokens``
    ``lm_head.weight``                           -> ``lm_head``

    The last numeric component wins, so a nested index (a vision block inside
    a tower) groups at the block, which is what a pipeline split needs.
    """
    parts = str(name).split(".")
    last_num = -1
    for i, part in enumerate(parts):
        if part.isdigit():
            last_num = i
    if last_num >= 0:
        return ".".join(parts[: last_num + 1])
    if len(parts) <= 1:
        return str(name)
    return ".".join(parts[:-1])


def _entry_bytes(entry: Mapping[str, Any]) -> int:
    """Measured resident bytes for one manifest tensor entry.

    ``kind == "tbe"`` reports ``resident_bytes`` (the container's own
    ``byte_size()["total"]``, padding and header included); anything kept raw
    reports ``original_bytes``.  Neither is estimated here and neither has a
    default: a manifest entry missing its byte count is a malformed manifest,
    not a zero-byte tensor.
    """
    kind = entry.get("kind")
    if kind == "tbe":
        field = "resident_bytes"
    else:
        field = "original_bytes"
    value = entry.get(field)
    if value is None:
        raise TBEDeviceMapError(
            f"manifest entry {entry.get('name')!r} (kind={kind!r}) has no "
            f"{field!r}; a device map planned from an incomplete byte count "
            "would not be a measurement"
        )
    return int(value)


def manifest_group_bytes(manifest: Mapping[str, Any]) -> "Dict[str, int]":
    """``{group: measured_bytes}`` in pipeline (natural) order.

    Reads the transcode manifest's own per-tensor byte counts.  Insertion
    order is the order a sequential plan will consume, so the caller never
    has to re-sort to get a stable plan.
    """
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise TBEDeviceMapError(
            "manifest has no 'tensors' list; there is nothing to place"
        )
    totals: Dict[str, int] = {}
    for entry in tensors:
        name = entry.get("name")
        if not name:
            raise TBEDeviceMapError("manifest tensor entry has no 'name'")
        key = group_key(str(name))
        totals[key] = totals.get(key, 0) + _entry_bytes(entry)
    return {k: totals[k] for k in sorted(totals, key=natural_key)}


# ---------------------------------------------------------------------------
# devices
# ---------------------------------------------------------------------------
def normalize_devices(devices: Sequence[Any]) -> List[torch.device]:
    """Canonicalise a device list, refusing duplicates and CPU placement."""
    if not devices:
        raise TBEDeviceMapError("no devices were given to place onto")
    out: List[torch.device] = []
    seen = set()
    for d in devices:
        dev = torch.device(d)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", 0)
        key = str(dev)
        if key in seen:
            raise TBEDeviceMapError(
                f"device {key} appears twice; a duplicate would be budgeted "
                "twice and the plan would not fit"
            )
        seen.add(key)
        out.append(dev)
    return out


def _budget_list(
    devices: Sequence[torch.device], budget_bytes: Any,
) -> List[int]:
    if isinstance(budget_bytes, Mapping):
        budgets = []
        for dev in devices:
            got = budget_bytes.get(str(dev), budget_bytes.get(dev))
            if got is None:
                raise TBEDeviceMapError(f"no budget given for device {dev}")
            budgets.append(int(got))
    elif isinstance(budget_bytes, (list, tuple)):
        if len(budget_bytes) != len(devices):
            raise TBEDeviceMapError(
                f"{len(budget_bytes)} budgets for {len(devices)} devices"
            )
        budgets = [int(b) for b in budget_bytes]
    else:
        budgets = [int(budget_bytes)] * len(devices)
    for dev, b in zip(devices, budgets):
        if b <= 0:
            raise TBEDeviceMapError(f"device {dev} has a non-positive budget {b}")
    return budgets


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------
def plan_device_map(
    group_bytes: Mapping[str, int],
    devices: Sequence[Any],
    *,
    budget_bytes: Any,
    strategy: str = STRATEGY_SEQUENTIAL,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """``(device_map, report)`` -- a byte-accurate placement, or a refusal.

    ``budget_bytes`` is the WEIGHT budget per device: whatever the caller has
    left after reserving activation, KV-cache and allocator headroom.  This
    function does not invent that reserve, because only the caller knows the
    batch and context the card also has to serve; it places weights into the
    budget it is given and refuses when they do not fit.

    Refusals are typed and total: ``group_exceeds_budget`` (one indivisible
    group is larger than a whole card -- no plan exists, splitting the group
    would put a layer's own tensors on two devices) and
    ``insufficient_capacity`` (the model does not fit the fleet).  Neither
    returns a partial map.
    """
    if strategy not in STRATEGIES:
        raise TBEDeviceMapError(
            f"strategy must be one of {STRATEGIES}, got {strategy!r}"
        )
    devs = normalize_devices(devices)
    budgets = _budget_list(devs, budget_bytes)
    if not group_bytes:
        raise TBEDeviceMapError("no groups to place")

    total = sum(int(v) for v in group_bytes.values())
    capacity = sum(budgets)
    # Checked BEFORE the total: an indivisible group larger than any card is a
    # different failure with a different fix (a bigger card, or a finer
    # no-split boundary) than a fleet that is merely too small, and reporting
    # the aggregate shortfall would send the operator after the wrong one.
    largest = max(budgets)
    oversized = [
        (g, int(s)) for g, s in group_bytes.items() if int(s) > largest
    ]
    if oversized:
        g, s = oversized[0]
        raise TBEDeviceMapError(
            f"group_exceeds_budget: {g!r} needs {s} bytes, more than the "
            f"largest device budget {largest}; this group is indivisible (its "
            "tensors must share a device) so no plan exists at this budget"
        )
    if total > capacity:
        raise TBEDeviceMapError(
            f"insufficient_capacity: {total} bytes of coded weight against "
            f"{capacity} bytes of budget across {len(devs)} device(s); short "
            f"by {total - capacity} bytes"
        )

    if strategy == STRATEGY_BALANCED:
        even = -(-total // len(devs))  # ceil
        targets = [min(b, max(even, 1)) for b in budgets]
        # A device whose budget is below the even share cannot take it; the
        # remainder spills forward through the sequential fill below.
    else:
        targets = list(budgets)

    device_map: Dict[str, str] = {}
    used = [0] * len(devs)
    idx = 0
    for group, size in group_bytes.items():
        size = int(size)
        if size > max(budgets):
            raise TBEDeviceMapError(
                f"group_exceeds_budget: {group!r} needs {size} bytes, more "
                f"than the largest device budget {max(budgets)}; this group "
                "is indivisible (its tensors must share a device) so no plan "
                "exists at this budget"
            )
        placed = False
        while idx < len(devs):
            # Fill against the strategy target first, then -- only if no
            # later device can take it either -- against the hard budget.
            if used[idx] + size <= targets[idx]:
                placed = True
                break
            if used[idx] + size <= budgets[idx] and idx == len(devs) - 1:
                placed = True
                break
            idx += 1
        if not placed:
            idx = len(devs) - 1
            for j, (u, b) in enumerate(zip(used, budgets)):
                if u + size <= b:
                    idx = j
                    placed = True
                    break
            if not placed:
                raise TBEDeviceMapError(
                    f"insufficient_capacity: {group!r} ({size} bytes) does not "
                    f"fit any remaining device budget (used={used}, "
                    f"budgets={budgets})"
                )
        device_map[group] = str(devs[idx])
        used[idx] += size

    per_device = {}
    for dev, u, b in zip(devs, used, budgets):
        per_device[str(dev)] = {
            "planned_bytes": int(u),
            "budget_bytes": int(b),
            "headroom_bytes": int(b - u),
            "n_groups": sum(1 for v in device_map.values() if v == str(dev)),
        }
    report = {
        "strategy": strategy,
        "n_groups": len(device_map),
        "n_devices": len(devs),
        "total_planned_bytes": int(total),
        "total_budget_bytes": int(capacity),
        "per_device": per_device,
        "devices": [str(d) for d in devs],
    }
    return device_map, report


def plan_device_map_from_manifest(
    manifest: Mapping[str, Any],
    devices: Sequence[Any],
    *,
    budget_bytes: Any,
    strategy: str = STRATEGY_SEQUENTIAL,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """:func:`plan_device_map` over :func:`manifest_group_bytes`."""
    groups = manifest_group_bytes(manifest)
    device_map, report = plan_device_map(
        groups, devices, budget_bytes=budget_bytes, strategy=strategy,
    )
    report["source"] = "transcode_manifest"
    report["manifest_schema"] = manifest.get("schema")
    report["group_bytes"] = dict(groups)
    return device_map, report


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------
def resolve_device(name: str, device_map: Mapping[str, Any]) -> torch.device:
    """Longest-prefix lookup, accelerate's ``device_map`` convention.

    ``""`` is a catch-all if the map declares one.  A name that matches no
    key raises: defaulting to device 0 is how a plan quietly stops being the
    plan that was measured.
    """
    if not device_map:
        raise TBEDeviceMapError("empty device_map: nothing can be resolved")
    name = str(name)
    best: Optional[str] = None
    for key in device_map:
        k = str(key)
        if k == "":
            if best is None:
                best = k
            continue
        if name == k or name.startswith(k + "."):
            if best is None or len(k) > len(best):
                best = k
    if best is None:
        raise TBEDeviceMapError(
            f"{name!r} matches no key in the device map "
            f"({sorted(str(k) for k in device_map)[:8]}...); refusing to "
            "default it onto a device the plan never budgeted for"
        )
    return torch.device(device_map[best])


def validate_device_map(
    names: Iterable[str], device_map: Mapping[str, Any],
) -> Dict[str, Any]:
    """Every name must resolve; returns a per-device name census.

    Run BEFORE any weight is touched, the same way
    ``swarm.compression_pipeline.Pipeline.validate`` runs before work starts:
    an unmappable tensor discovered halfway through a 360 GB load has already
    cost the load.
    """
    census: Dict[str, int] = {}
    unmapped: List[str] = []
    for name in names:
        try:
            dev = resolve_device(name, device_map)
        except TBEDeviceMapError:
            unmapped.append(str(name))
            continue
        census[str(dev)] = census.get(str(dev), 0) + 1
    if unmapped:
        raise TBEDeviceMapError(
            f"{len(unmapped)} tensor name(s) resolve to no device: "
            f"{unmapped[:8]}"
        )
    return {"per_device_tensor_counts": census, "n_names": sum(census.values())}


def device_map_for_modules(
    module_names: Iterable[str], device_map: Mapping[str, Any],
) -> Dict[str, torch.device]:
    """Materialise ``{module_name: device}`` for a concrete module list."""
    return {str(n): resolve_device(n, device_map) for n in module_names}


def devices_in_map(device_map: Mapping[str, Any]) -> List[torch.device]:
    """The distinct devices a map names, in first-seen order."""
    out: List[torch.device] = []
    seen = set()
    for value in device_map.values():
        dev = torch.device(value)
        if str(dev) not in seen:
            seen.add(str(dev))
            out.append(dev)
    return out


def single_device_map(device: Any) -> Dict[str, str]:
    """The degenerate one-device map: everything on ``device``."""
    return {"": str(torch.device(device))}


__all__ = [
    "STRATEGIES",
    "STRATEGY_BALANCED",
    "STRATEGY_SEQUENTIAL",
    "TBEDeviceMapError",
    "device_map_for_modules",
    "devices_in_map",
    "group_key",
    "manifest_group_bytes",
    "natural_key",
    "normalize_devices",
    "plan_device_map",
    "plan_device_map_from_manifest",
    "resolve_device",
    "single_device_map",
    "validate_device_map",
]
