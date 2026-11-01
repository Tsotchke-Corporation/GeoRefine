"""Metal (MLX) decode path for the GLC-TBE container.

Companion to the CUDA fragment kernel (``release/glc_loader/tbe_mma.py``):
same container bytes, same bit-exact contract, a different device.  See
``tbe_decode_mlx.py`` for the kernel and ``tbe_linear_mlx.py`` for the
``nn.Linear``-shaped module that decodes into a scratch buffer and calls
``mx.matmul``.
"""
from __future__ import annotations

__all__ = [
    "TBEDeviceMLX",
    "TBELinearMLX",
    "decode_tbe_metal",
    "mlx_metal_available",
    "upload_tbe_mlx",
]


def __getattr__(name):
    """Keep status-guard imports CPU-safe; load MLX only on API use."""
    if name in {"TBEDeviceMLX", "decode_tbe_metal", "mlx_metal_available", "upload_tbe_mlx"}:
        from . import tbe_decode_mlx
        return getattr(tbe_decode_mlx, name)
    if name == "TBELinearMLX":
        from .tbe_linear_mlx import TBELinearMLX
        return TBELinearMLX
    raise AttributeError(name)
