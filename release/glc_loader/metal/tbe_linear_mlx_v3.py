"""``TBELinearMLXV3`` -- decode-then-matmul over the phase-3 SIMD-group decoder.

Same shape and contract as ``tbe_linear_mlx.TBELinearMLX`` /
``tbe_linear_mlx_v2.TBELinearMLXV2``; the only difference is which decode
kernel it calls (``decode_tbe_metal_v3``).  Kept as a separate file, same
precedent as v2 (see ``tbe_decode_mlx_v3.py``'s module docstring).
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx

from .tbe_decode_mlx_v3 import TBEDeviceMLX, decode_tbe_metal_v3


class TBELinearMLXV3Error(RuntimeError):
    """A phase-3 TBE-served MLX module cannot be built or run as requested."""


class TBELinearMLXV3:
    """Decode-then-matmul linear layer over the phase-3 SIMD-group decoder."""

    def __init__(
        self,
        container: TBEDeviceMLX,
        bias: Optional[mx.array] = None,
        variant: str = "t1_g1",
        broadcast_load: bool = True,
    ) -> None:
        self.container = container
        self.bias = bias
        self.variant = variant
        self.broadcast_load = broadcast_load
        self.out_features, self.in_features = container.shape

    @property
    def resident_bytes(self) -> int:
        return self.container.resident_bytes

    @property
    def dense_bytes(self) -> int:
        return self.container.dense_bytes

    def decode(self) -> mx.array:
        return decode_tbe_metal_v3(
            self.container, variant=self.variant, broadcast_load=self.broadcast_load
        )

    def __call__(self, x: mx.array) -> mx.array:
        flat = x.reshape(-1, x.shape[-1])
        if int(flat.shape[-1]) != self.in_features:
            raise TBELinearMLXV3Error(
                f"input has {int(flat.shape[-1])} features, weight expects "
                f"{self.in_features}"
            )
        w = self.decode()
        y = mx.matmul(flat.astype(mx.bfloat16), w.T)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*x.shape[:-1], self.out_features)


__all__ = ["TBELinearMLXV3", "TBELinearMLXV3Error"]
