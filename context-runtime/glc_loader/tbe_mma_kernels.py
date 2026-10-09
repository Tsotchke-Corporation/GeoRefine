"""Host wrapper that JIT-builds the bundled TBE fragment-kernel extension.

``tbe_mma.py`` calls this wrapper with the CUDA extension's positional
signature. The sibling ``tbe_mma_kernel.cu`` ships in the codec source archive
and wheel, then compiles on first CUDA use. Importing this module alone never
builds an extension. Build products go to a writable user cache, separate from
the installed package. Neither this wrapper nor the kernel imports research
modules from the GeoRefine repository.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import torch

from .tbe_mma import TBEMMAError, resolve_tbe_mma_arch, tbe_mma_capability

_HERE = os.path.dirname(os.path.abspath(__file__))

#: Bundled in the codec wheel and source archive.
_KERNEL_SOURCE = os.path.join(_HERE, "tbe_mma_kernel.cu")

#: Writable per-user JIT cache; GLC_TBE_MMA_BUILD_DIR can override it.
_DEFAULT_BUILD_DIR = os.path.join(
    os.path.expanduser(os.environ.get("XDG_CACHE_HOME", "~/.cache")),
    "georefine", "tbe_mma_build",
)

_EXT: Optional[Any] = None
_EXT_ERROR: Optional[BaseException] = None


def _build_extension(device: Optional["torch.device"] = None) -> Any:
    """Build (once) and return the native extension module, or raise.

    Every refusal here is a :class:`TBEMMAError` with a message naming
    exactly what is missing -- no silent fallback, no generic traceback.
    The kernel source is a sibling of this wrapper in the installed package.
    """
    global _EXT, _EXT_ERROR
    if _EXT is not None:
        return _EXT
    if _EXT_ERROR is not None:
        raise TBEMMAError(
            f"TBE MMA extension previously failed to build: {_EXT_ERROR}"
        )
    if not torch.cuda.is_available():
        raise TBEMMAError(
            "TBE fragment kernels are CUDA-only; no CUDA device is visible "
            "on this machine, so the vendored extension cannot be built"
        )
    if not os.path.isfile(_KERNEL_SOURCE):
        raise TBEMMAError(
            f"bundled kernel source not found at {_KERNEL_SOURCE!r}; "
            "reinstall a complete glc-loader wheel or source archive"
        )
    target = (torch.device(device) if device is not None
              else torch.device("cuda", torch.cuda.current_device()))
    arch = resolve_tbe_mma_arch(target)
    capability = torch.cuda.get_device_capability(target)
    expected = tbe_mma_capability()
    if tuple(capability) != expected:
        raise TBEMMAError(
            f"the vendored TBE fragment kernel targets capability "
            f"{expected[0]}.{expected[1]} (GLC_TBE_MMA_ARCH={arch!r}), the "
            f"active device is {tuple(capability)}"
        )

    from torch.utils.cpp_extension import load

    build_dir = os.environ.get("GLC_TBE_MMA_BUILD_DIR", _DEFAULT_BUILD_DIR)
    os.makedirs(build_dir, exist_ok=True)
    previous_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        with torch.cuda.device(target):
            _EXT = load(
                name="glc_tbe_mma_ext",
                sources=[_KERNEL_SOURCE],
                build_directory=build_dir,
                verbose=bool(int(os.environ.get("GLC_TBE_MMA_VERBOSE", "0"))),
                extra_cuda_cflags=["-O3", "-lineinfo"],
            )
    except BaseException as exc:
        _EXT_ERROR = exc
        raise TBEMMAError(
            f"failed to build/load the vendored TBE MMA extension: {exc}"
        ) from exc
    finally:
        if previous_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch_list
    return _EXT


def tbe_mma_decode(planes, smb, esc, sbbase, out, n, k,
                    superblock, e01, e23, stream):
    """Delegate to the built extension's ``tbe_mma_decode``.

    Argument order is exactly what ``tbe_mma.py::tbe_mma_decode`` passes and
    what the pybind binding takes -- this wrapper does no translation, only
    build-and-dispatch, so a shape or dtype bug surfaces at the real call
    site rather than being masked here.
    """
    device = out.device if isinstance(out, torch.Tensor) else None
    ext = _build_extension(device)
    return ext.tbe_mma_decode(
        planes, smb, esc, sbbase, out, n, k, superblock, e01, e23, stream,
    )


def tbe_mma_gemm(xb, planes, smb, esc, sbbase, bias, ws, out, m, n, k,
                  superblock, e01, e23, split_k, stream):
    """Delegate to the built extension's ``tbe_mma_gemm``. See ``tbe_mma_decode``."""
    device = out.device if isinstance(out, torch.Tensor) else None
    ext = _build_extension(device)
    return ext.tbe_mma_gemm(
        xb, planes, smb, esc, sbbase, bias, ws, out, m, n, k,
        superblock, e01, e23, split_k, stream,
    )


__all__ = ["tbe_mma_decode", "tbe_mma_gemm"]
