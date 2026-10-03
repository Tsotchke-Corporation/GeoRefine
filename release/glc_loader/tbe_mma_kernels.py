"""Host wrapper that JIT-builds the vendored TBE fragment-kernel extension.

``tbe_mma.py``'s ``load_tbe_kernels()`` does ``from . import tbe_mma_kernels``
and expects this module to expose ``tbe_mma_decode`` / ``tbe_mma_gemm`` with
the exact positional signature the CUDA extension's pybind bindings take (see
``experiments/georefine/_glc_tbe_mma_kernel.cu``'s ``PYBIND11_MODULE`` block).

This file is checked into source control -- unlike ``fwp1_kernels.py``, which
is nothing but a copied Triton source file, the TBE fragment kernel needs a
JIT *build* step (``torch.utils.cpp_extension.load``), so the piece that
belongs in the loader package is this host wrapper, not the CUDA source
itself.  The CUDA source is the part that mirrors ``fwp1_kernels.py``
exactly: ``experiments/georefine/_glc_tbe_mma_kernel.cu`` is copied to
``tbe_mma_kernel.cu`` beside this file at GLC-RELEASE build time by
``_copy_loader`` in ``experiments/georefine/_glc_release.py`` -- the same
function, same ``shutil.copy2`` + digest mechanism, that vendors
``fwp1_kernels.py``.  An editable checkout of this package alone (this
repository's own ``release/glc_loader/``, with no release build run) has no
``tbe_mma_kernel.cu`` next to this file, and correctly refuses rather than
guessing at a source location -- that refusal is what makes this file's CPU
tests possible without ever touching CUDA.

Nothing in this file reaches back into ``experiments.georefine``;
``tests/test_glc_release.py::test_vendored_loader_imports_nothing_from_this_
repository`` asserts that for every ``*.py`` in this package, this file
included.  The one repository import here is the sibling ``.tbe_mma`` module,
which ships in the same artifact -- not a reach outside it.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import torch

from .tbe_mma import TBEMMAError, resolve_tbe_mma_arch, tbe_mma_capability

_HERE = os.path.dirname(os.path.abspath(__file__))

#: Vendored at GLC-RELEASE build time (see module docstring).  Not present in
#: this repository's own ``release/glc_loader/`` -- only in a built artifact.
_KERNEL_SOURCE = os.path.join(_HERE, "tbe_mma_kernel.cu")

#: Where the JIT build's object files and ``.so`` land.  Release-local (a
#: sibling of the vendored source, not a shared torch extensions cache) so
#: two artifacts on the same machine, or two arches of the same artifact,
#: never collide.  ``GLC_TBE_MMA_BUILD_DIR`` overrides it, mirroring the
#: research module's env var of the same name.
_DEFAULT_BUILD_DIR = os.path.join(_HERE, "_tbe_mma_build")

_EXT: Optional[Any] = None
_EXT_ERROR: Optional[BaseException] = None


def _build_extension(device: Optional["torch.device"] = None) -> Any:
    """Build (once) and return the native extension module, or raise.

    Every refusal here is a :class:`TBEMMAError` with a message naming
    exactly what is missing -- no silent fallback, no generic traceback.
    Mirrors ``experiments.georefine._glc_tbe_mma._load_ext``, adapted to
    build from a vendored sibling file rather than a path inside this
    repository.
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
            f"vendored kernel source not found at {_KERNEL_SOURCE!r}; this "
            "checkout of glc_loader was not produced by a GLC-RELEASE build "
            "(see release/glc_loader/tbe_mma_kernels.py's module docstring "
            "-- the .cu source is copied in at build time, not shipped in "
            "the repository)"
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
