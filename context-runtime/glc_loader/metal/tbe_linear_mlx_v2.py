"""``TBELinearMLXV2`` -- decode-then-matmul over the phase-2 tile-per-thread decoder.

Same shape and contract as ``tbe_linear_mlx.TBELinearMLX``; the only
difference is which decode kernel it calls (``decode_tbe_metal_v2`` instead
of phase 1's ``decode_tbe_metal``).  Kept as a separate class in a separate
file rather than adding a ``variant`` flag to the phase-1 class because
``tbe_linear_mlx.py`` was mid-commit when this was written -- see
``tbe_decode_mlx_v2.py``'s module docstring.
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx

from .tbe_decode_mlx_v2 import TBEDeviceMLX, decode_tbe_metal_v2


class TBELinearMLXV2Error(RuntimeError):
    """A phase-2 TBE-served MLX module cannot be built or run as requested."""


class TBELinearMLXV2:
    """Decode-then-matmul linear layer over the phase-2 tile-per-thread decoder."""

    def __init__(
        self,
        container: TBEDeviceMLX,
        bias: Optional[mx.array] = None,
        variant: str = "tile1",
    ) -> None:
        self.container = container
        self.bias = bias
        self.variant = variant
        self.out_features, self.in_features = container.shape

    @property
    def resident_bytes(self) -> int:
        return self.container.resident_bytes

    @property
    def dense_bytes(self) -> int:
        return self.container.dense_bytes

    def decode(self) -> mx.array:
        return decode_tbe_metal_v2(self.container, variant=self.variant)

    def __call__(self, x: mx.array) -> mx.array:
        flat = x.reshape(-1, x.shape[-1])
        if int(flat.shape[-1]) != self.in_features:
            raise TBELinearMLXV2Error(
                f"input has {int(flat.shape[-1])} features, weight expects "
                f"{self.in_features}"
            )
        w = self.decode()
        y = mx.matmul(flat.astype(mx.bfloat16), w.T)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*x.shape[:-1], self.out_features)


__all__ = ["TBELinearMLXV2", "TBELinearMLXV2Error"]
