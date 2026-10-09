"""Experimental default-off M=1 paired-warp MIV-TBE kernel.

This module is a standalone opt-in path. It does not alter ``miv_tbe``
dispatch. Four physical warps (or eight by explicit build control) execute the
existing eight logical warp/slice schedule; arithmetic and reduction order
match the v2 kernel. Build controls are read only when the extension is first
requested: ``MIV_TBE_PAIR_PHYSICAL_WARPS=4|8`` (default 4) and
``MIV_TBE_PAIR_MAX_REGISTERS=0|32|40|48|64`` (default 0).
"""
from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import NamedTuple, Optional

import torch


_CSRC = Path(__file__).resolve().parent / "miv_tbe_csrc"
_EXT = None
_EXT_CONFIG = None
_EXT_LOCK = threading.Lock()
_WPR = (1, 2, 4, 8)
_UNROLL = (1, 2, 4)
_PHYSICAL_WARPS = (4, 8)
_MAX_REGISTERS = (0, 32, 40, 48, 64)


class BuildConfig(NamedTuple):
    physical_warps: int
    max_registers: int

    @property
    def tag(self) -> str:
        return f"pw{self.physical_warps}_r{self.max_registers}"


def _build_config(environ=None) -> BuildConfig:
    env = os.environ if environ is None else environ
    values = (
        ("MIV_TBE_PAIR_PHYSICAL_WARPS", 4, _PHYSICAL_WARPS),
        ("MIV_TBE_PAIR_MAX_REGISTERS", 0, _MAX_REGISTERS),
    )
    parsed = []
    for name, default, allowed in values:
        raw = env.get(name)
        try:
            value = default if raw is None else int(raw, 10)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be one of {allowed}") from exc
        if value not in allowed:
            raise ValueError(f"{name} must be one of {allowed}")
        parsed.append(value)
    return BuildConfig(*parsed)


def _compile_cuda_flags(config: BuildConfig) -> list[str]:
    flags = [
        "-O3", "--fmad=true",
        f"-DMIV_TBE_PAIR_PHYSICAL_WARPS={config.physical_warps}",
        f"-DMIV_TBE_PAIR_MAX_REGISTERS={config.max_registers}",
    ]
    if config.max_registers:
        flags.append(f"--maxrregcount={config.max_registers}")
    return flags


def _validate(planes, smb, esc, rowparam, escbase, x, y, N, K, wpr, unroll,
              bias=None) -> None:
    N, K, wpr, unroll = int(N), int(K), int(wpr), int(unroll)
    if N <= 0 or K <= 0:
        raise ValueError("N and K must be positive")
    if wpr not in _WPR:
        raise ValueError(f"wpr must be one of {_WPR}")
    if unroll not in _UNROLL:
        raise ValueError(f"unroll must be one of {_UNROLL}")
    if K % (64 * wpr):
        raise ValueError("K must be divisible by 64*wpr")
    if x.dim() != 2 or tuple(x.shape) != (1, K):
        raise ValueError("paired MIV-TBE supports only x with shape [1, K]")
    if y.dim() != 2 or tuple(y.shape) != (1, N):
        raise ValueError("y must have shape [1, N]")
    expected = (
        (planes, torch.int32, N * (K // 64) * 6, "planes"),
        (smb, torch.uint8, N * K, "smb"),
        (rowparam, torch.int32, N, "rowparam"),
        (escbase, torch.int32, N * wpr, "escbase"),
    )
    for tensor, dtype, size, name in expected:
        if tensor.dtype != dtype or tensor.dim() != 1 or tensor.numel() != size:
            raise ValueError(f"{name} must be contiguous {dtype} with {size} values")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if esc.dtype != torch.uint8 or esc.dim() != 1 or esc.numel() < 16:
        raise ValueError("esc must be a uint8 vector with at least 16 pad bytes")
    if not esc.is_contiguous():
        raise ValueError("esc must be contiguous")
    if x.dtype != torch.bfloat16 or y.dtype != torch.bfloat16:
        raise ValueError("x and y must be bfloat16")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("x and y must be contiguous")
    if bias is not None and (bias.dtype != torch.bfloat16 or tuple(bias.shape) != (N,)
                             or not bias.is_contiguous()):
        raise ValueError("bias must be contiguous bfloat16 [N]")
    tensors = [planes, smb, esc, rowparam, escbase, x, y]
    if bias is not None:
        tensors.append(bias)
    if any(t.device != x.device for t in tensors):
        raise ValueError("all tensors must be on the same device")
    if not x.is_cuda:
        raise ValueError("paired MIV-TBE requires CUDA tensors")


def _build_dir(config: Optional[BuildConfig] = None) -> Path:
    config = config or _build_config()
    configured = os.environ.get("MIV_TBE_PAIR_BUILD_DIR")
    root = Path(configured).expanduser() if configured else (
        Path(__file__).resolve().parents[2] / ".scratch" / "miv-tbe-pair-build")
    resolved = (root / config.tag).resolve()
    if ".scratch" not in resolved.parts:
        raise RuntimeError("MIV_TBE_PAIR_BUILD_DIR must be under a durable .scratch directory")
    if resolved.is_relative_to(Path("/tmp")) or resolved.is_relative_to(Path("/private/tmp")):
        raise RuntimeError("refusing a volatile MIV-TBE pair build directory")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def extension():
    """Build only the experimental M1 translation units when explicitly used."""
    global _EXT, _EXT_CONFIG
    config = _build_config()
    if _EXT is not None:
        if _EXT_CONFIG != config:
            raise RuntimeError(
                f"paired MIV-TBE extension cached for {_EXT_CONFIG}; requested {config}; "
                "restart the process to change build controls")
        return _EXT
    with _EXT_LOCK:
        if _EXT is not None:
            if _EXT_CONFIG != config:
                raise RuntimeError(
                    f"paired MIV-TBE extension cached for {_EXT_CONFIG}; requested {config}; "
                    "restart the process to change build controls")
            return _EXT
        if _EXT is None:
            if not torch.cuda.is_available():
                raise RuntimeError("paired MIV-TBE requires CUDA")
            from torch.utils.cpp_extension import load

            major, minor = torch.cuda.get_device_capability()
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
            build_name = f"miv_tbe_pair_sm{major}{minor}_{config.tag}"
            _EXT = load(
                name=build_name,
                sources=[str(_CSRC / "miv_tbe_pair_torch.cpp"),
                         str(_CSRC / "miv_tbe_pair_m1.cu")],
                extra_include_paths=[str(_CSRC)],
                extra_cflags=["-O3"],
                extra_cuda_cflags=_compile_cuda_flags(config),
                build_directory=str(_build_dir(config)),
                verbose=bool(int(os.environ.get("MIV_TBE_PAIR_VERBOSE", "0"))),
            )
            _EXT_CONFIG = config
    return _EXT


def out(planes, smb, esc, rowparam, escbase, x, y, N, K, wpr, unroll,
        bias: Optional[torch.Tensor] = None) -> None:
    """Write one M=1 MIV-TBE output row into ``y``; no allocation or dispatch."""
    _validate(planes, smb, esc, rowparam, escbase, x, y, N, K, wpr, unroll, bias)
    extension().miv_tbe_pair_out(planes, smb, esc, rowparam, escbase, x, y,
                                 int(N), int(K), int(wpr), int(unroll), bias)


__all__ = ["out"]
