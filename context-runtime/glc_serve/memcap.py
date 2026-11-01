"""Emulate a smaller card on a bigger one -- stated as an emulation, always.

``torch.cuda.set_per_process_memory_fraction`` makes the PyTorch caching
allocator refuse to grow past the cap (an over-cap allocation raises OOM, it
is not silently served).  What it does NOT cap is memory outside that
allocator: the CUDA context itself and any library that allocates with the
driver API directly.  So every capped run also records the process's total
device memory from NVML (``nvidia-smi --query-compute-apps``), and
``assert_within_cap`` fails when EITHER the allocator's peak reservation or the
whole-process figure exceeds the card being emulated.
"""
from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, Optional

import torch

GIB = 1 << 30

#: Real cards the caps stand for (total device memory, MiB, from the vendor).
KNOWN_CARDS_MIB = {
    "a100-40": 40960,
    "l40s-48": 46068,
    "rtx-5090-32": 32607,
    "l4-24": 23034,
}


def process_device_mib(pid: Optional[int] = None) -> Optional[int]:
    """This process's device memory per NVML, in MiB (None if unavailable)."""
    pid = int(pid or os.getpid())
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
    except Exception:
        return None
    total = 0
    found = False
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == pid:
            total += int(parts[1])
            found = True
    return total if found else None


def apply_memory_cap(device, cap_gib: float) -> Dict[str, Any]:
    dev = torch.device(device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    total = int(torch.cuda.get_device_properties(idx).total_memory)
    cap = int(float(cap_gib) * GIB)
    if cap > total:
        raise ValueError(f"cap {cap} B exceeds the physical card ({total} B)")
    frac = cap / total
    torch.cuda.set_per_process_memory_fraction(frac, idx)
    torch.cuda.reset_peak_memory_stats(idx)
    return {
        "emulated": True,
        "cap_bytes": cap,
        "cap_gib": float(cap_gib),
        "device_total_bytes": total,
        "device_name": torch.cuda.get_device_name(idx),
        "fraction": frac,
        "note": ("EMULATION: the PyTorch caching allocator is capped on a larger card. "
                 "The CUDA context and driver-level allocations are outside the cap and "
                 "are reported separately from NVML as whole-process memory."),
    }


def measure(device) -> Dict[str, Any]:
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "max_allocated": 0, "max_reserved": 0,
                "nvml_process_mib": None, "device": str(dev)}
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    return {
        "allocated": int(torch.cuda.memory_allocated(idx)),
        "reserved": int(torch.cuda.memory_reserved(idx)),
        "max_allocated": int(torch.cuda.max_memory_allocated(idx)),
        "max_reserved": int(torch.cuda.max_memory_reserved(idx)),
        "nvml_process_mib": process_device_mib(),
    }


def assert_within_cap(device, cap: Dict[str, Any], *, card_mib: Optional[int] = None,
                      observed_nvml_mib: Optional[list[Optional[int]]] = None
                      ) -> Dict[str, Any]:
    m = measure(device)
    ok_alloc = m["max_reserved"] <= cap["cap_bytes"]
    nvml = m["nvml_process_mib"]
    card = card_mib
    samples = list(observed_nvml_mib or [])
    numeric_samples = [int(v) for v in samples if v is not None]
    if nvml is not None:
        numeric_samples.append(int(nvml))
    peak_nvml = max(numeric_samples) if numeric_samples else None
    missing_samples = sum(v is None for v in samples)
    if card is None:
        ok_nvml = True
        reason = None
    elif not samples:
        ok_nvml = False
        reason = "no phase NVML samples; physical-card fit is uncertified"
    elif missing_samples:
        ok_nvml = False
        reason = f"{missing_samples} phase NVML sample(s) missing; physical-card fit is uncertified"
    elif nvml is None:
        ok_nvml = False
        reason = "final NVML sample missing; physical-card fit is uncertified"
    elif peak_nvml is None:
        ok_nvml = False
        reason = "NVML process memory unavailable; physical-card fit is uncertified"
    elif peak_nvml > card:
        ok_nvml = False
        reason = f"observed process peak {peak_nvml} MiB exceeds emulated card {card} MiB"
    else:
        ok_nvml = True
        reason = None
    out = {**m, "cap_bytes": cap["cap_bytes"], "allocator_within_cap": ok_alloc,
           "emulated_card_mib": card,
           "whole_process_within_card": ok_nvml if card is not None else None,
           "nvml_process_peak_mib": peak_nvml,
           "nvml_phase_sample_count": len(samples),
           "nvml_missing_phase_sample_count": missing_samples,
           "physical_card_failure": reason}
    if not ok_alloc or not ok_nvml:
        raise RuntimeError(f"memory cap exceeded: {out}")
    return out


__all__ = ["KNOWN_CARDS_MIB", "apply_memory_cap", "assert_within_cap", "measure",
           "process_device_mib"]
