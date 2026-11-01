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


_SOURCES = ("miv_tbe_pair_torch.cpp", "miv_tbe_pair_m1.cu")
_CFLAGS = ["-O3"]


def kernel_build_plan(config: Optional[BuildConfig] = None):
    """(extension name, build directory, kernel_cache.ExtensionSpec) for the current device."""
    from glc_serve import kernel_cache

    config = config or _build_config()
    major, minor = torch.cuda.get_device_capability()
    arch = os.environ.get("TORCH_CUDA_ARCH_LIST") or f"{major}.{minor}"
    name = f"miv_tbe_pair_sm{major}{minor}_{config.tag}"
    spec = kernel_cache.ExtensionSpec(
        name=name, sources=tuple(str(_CSRC / s) for s in _SOURCES), include_dirs=(str(_CSRC),),
        flags=tuple(_CFLAGS) + ("--cuda--",) + tuple(_compile_cuda_flags(config)), arch=arch)
    return name, _build_dir(config, spec=spec), spec


def _build_dir(config: Optional[BuildConfig] = None, *, spec=None) -> Path:
    """``MIV_TBE_PAIR_BUILD_DIR/<config tag>`` when set -- it must lie under a durable
    ``.scratch`` directory or under the kernel cache root, never /tmp -- otherwise the
    versioned kernel-cache directory (whose key carries the config through the flags)."""
    from glc_serve import kernel_cache

    config = config or _build_config()
    configured = os.environ.get("MIV_TBE_PAIR_BUILD_DIR")
    if not configured:
        if spec is None:
            spec = kernel_build_plan(config)[2]
        return kernel_cache.extension_dir(spec)
    resolved = (Path(configured).expanduser() / config.tag).resolve()
    try:
        under_cache = resolved.is_relative_to(kernel_cache.cache_root().resolve())
    except kernel_cache.KernelCacheError:
        under_cache = False
    if ".scratch" not in resolved.parts and not under_cache:
        raise RuntimeError("MIV_TBE_PAIR_BUILD_DIR must be under a durable .scratch directory "
                           "or the kernel cache root (GEOREFINE_CONTEXT_KERNEL_CACHE)")
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
            from glc_serve import kernel_cache

            major, minor = torch.cuda.get_device_capability()
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
            build_name = f"miv_tbe_pair_sm{major}{minor}_{config.tag}"
            name, build_dir, spec = kernel_build_plan(config)
            _EXT = kernel_cache.build(
                name, build_dir,
                lambda: load(
                    name=build_name,
                    sources=[str(_CSRC / s) for s in _SOURCES],
                    extra_include_paths=[str(_CSRC)],
                    extra_cflags=list(_CFLAGS),
                    extra_cuda_cflags=_compile_cuda_flags(config),
                    build_directory=str(build_dir),
                    verbose=bool(int(os.environ.get("MIV_TBE_PAIR_VERBOSE", "0"))),
                ),
                spec=spec)
            _EXT_CONFIG = config
    return _EXT


def out(planes, smb, esc, rowparam, escbase, x, y, N, K, wpr, unroll,
        bias: Optional[torch.Tensor] = None) -> None:
    """Write one M=1 MIV-TBE output row into ``y``; no allocation or dispatch."""
    _validate(planes, smb, esc, rowparam, escbase, x, y, N, K, wpr, unroll, bias)
    extension().miv_tbe_pair_out(planes, smb, esc, rowparam, escbase, x, y,
                                 int(N), int(K), int(wpr), int(unroll), bias)


__all__ = ["out"]
