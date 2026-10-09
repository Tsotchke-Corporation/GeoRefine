"""Opt-in MIV-TBE prototype with lane-major flat SMB bytes.

The canonical container remains mma16. ``flatten_smb`` / ``restore_smb``
perform only the byte permutation between that public representation and this
prototype's kernel layout.
"""
from __future__ import annotations
import os
import threading
from pathlib import Path
from typing import Optional
import torch

_CSRC = Path(__file__).resolve().parent / "miv_tbe_csrc"
_EXT = None
_LOCK = threading.Lock()


def flatten_smb(smb: torch.Tensor) -> torch.Tensor:
    """Convert [T,4,8,2] stored-order SMB bytes to [T,8,4,2] lane-major bytes."""
    if smb.dtype != torch.uint8 or smb.numel() % 64:
        raise ValueError("smb must contain uint8 [T,64] tiles")
    return smb.reshape(-1, 4, 8, 2).permute(0, 2, 1, 3).contiguous().reshape(-1)


def restore_smb(flat: torch.Tensor) -> torch.Tensor:
    """Restore lane-major flat SMB bytes to canonical [T,64] order."""
    if flat.dtype != torch.uint8 or flat.numel() % 64:
        raise ValueError("flat SMB must contain uint8 [T,64] tiles")
    return flat.reshape(-1, 8, 4, 2).permute(0, 2, 1, 3).contiguous().reshape(-1)


def extension():
    global _EXT
    if _EXT is not None:
        return _EXT
    with _LOCK:
        if _EXT is None:
            from torch.utils.cpp_extension import load
            major, minor = torch.cuda.get_device_capability()
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
            os.environ.setdefault("MAX_JOBS", "4")
            root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
            build = Path(os.environ.get("MIV_TBE_FLAT_SMB_BUILD_DIR", root / "georefine" / "miv_tbe_flat_smb"))
            build.mkdir(parents=True, exist_ok=True)
            _EXT = load(name=f"miv_tbe_flat_smb_sm{major}{minor}",
                        sources=[str(_CSRC / "miv_tbe_flat_smb.cu"), str(_CSRC / "miv_tbe_flat_smb_torch.cpp")],
                        extra_include_paths=[str(_CSRC)], extra_cflags=["-O3"],
                        extra_cuda_cflags=["-O3", "--fmad=true", "-lineinfo"],
                        build_directory=str(build), verbose=bool(int(os.environ.get("MIV_TBE_FLAT_SMB_VERBOSE", "0"))))
    return _EXT


def _inputs(desc, x, out):
    from .miv_tbe import TBEDesc
    if not isinstance(desc, TBEDesc):
        raise TypeError("flat SMB requires a TBEDesc")
    if desc.wpr not in (1, 2, 4, 8) or desc.unroll not in (1, 2, 4):
        raise ValueError("WPR must be 1/2/4/8 and U must be 1/2/4")
    if (x.device.type != "cuda" or x.dtype != torch.bfloat16 or x.dim() != 2 or
            x.shape != (1, desc.k) or x.stride(1) != 1 or x.stride(0) % 8 or x.data_ptr() % 16):
        raise ValueError("flat SMB out requires aligned CUDA bf16 x with shape [1,K]")
    if desc.k % (64 * desc.wpr):
        raise ValueError("K must be divisible by 64*WPR")
    if out is None:
        out = torch.empty((1, desc.n), dtype=torch.bfloat16, device=x.device)
    if out.device != x.device or out.dtype != torch.bfloat16 or out.shape != (1, desc.n):
        raise ValueError("out must be CUDA bf16 [1,N] on x device")
    return out


def _storage(desc, prepared):
    if prepared is None:
        return flatten_smb(desc.smb)
    if (prepared.dtype != torch.uint8 or prepared.device != desc.smb.device
            or prepared.dim() != 1 or prepared.numel() != desc.n * desc.k
            or not prepared.is_contiguous()):
        raise ValueError("prepared flat SMB must be a contiguous uint8 [N*K] buffer on descriptor device")
    return prepared


def out(desc, x: torch.Tensor, out: Optional[torch.Tensor] = None, *,
        flat_smb: Optional[torch.Tensor] = None) -> torch.Tensor:
    """M1 fused output using flat SMB storage; default dispatch is untouched."""
    y = _inputs(desc, x, out)
    extension().out(x, desc.planes, _storage(desc, flat_smb), desc.esc, desc.rowparam,
                    desc.escbase, desc.bias, y, desc.n, desc.k, desc.wpr, desc.unroll)
    return y


def decode_out(desc, out: Optional[torch.Tensor] = None, *,
               flat_smb: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Decode full [N,K] bf16 matrix through the flat-SMB kernel."""
    from .miv_tbe import TBEDesc
    if not isinstance(desc, TBEDesc):
        raise TypeError("flat SMB requires a TBEDesc")
    if out is None:
        out = torch.empty((desc.n, desc.k), dtype=torch.bfloat16, device=desc.planes.device)
    if out.device != desc.planes.device or out.dtype != torch.bfloat16 or out.shape != (desc.n, desc.k):
        raise ValueError("out must be CUDA bf16 [N,K] on descriptor device")
    extension().decode_out(desc.planes, _storage(desc, flat_smb), desc.esc, desc.rowparam,
                           desc.escbase, out, desc.n, desc.k, desc.wpr, desc.unroll)
    return out


__all__ = ["decode_out", "extension", "flatten_smb", "out", "restore_smb"]
