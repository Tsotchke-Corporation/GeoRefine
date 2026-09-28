"""``TBELinearMLXV4`` -- decode-then-matmul with NO per-layer synchronisation.

Same shape and contract as ``TBELinearMLX`` / ``V2`` / ``V3``.  Two
differences, both measured (``docs/research/TBE_METAL_ACCESS_PATTERN_20260903.md``):

* it calls :func:`decode_tbe_metal_v4`, which leaves the decode in the MLX
  graph instead of forcing an ``mx.eval`` per layer (~167 us round trip on
  M2 Ultra, paid 252 times per token by the certified Qwen3-4B artifact);
* it defaults to the ``t4_g4`` geometry (279 GB/s at 256 MB) rather than v1's
  element-per-thread kernel (107 GB/s) or ``TBELinearMLXV3``'s ``t1_g1``
  default (92 GB/s).

Nothing about the decoded bytes changes: same kernel, same container, same
bit-exact output.
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx

from .tbe_decode_mlx_v4 import DEFAULT_VARIANT, TBEDeviceMLX, decode_tbe_metal_v4


class TBELinearMLXV4Error(RuntimeError):
    """A phase-4 TBE-served MLX module cannot be built or run as requested."""


class TBELinearMLXV4:
    """Decode-then-matmul linear layer over the batchable phase-4 dispatch."""

    def __init__(
        self,
        container: TBEDeviceMLX,
        bias: Optional[mx.array] = None,
        variant: str = DEFAULT_VARIANT,
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

    def decode(self, eval_now: bool = False) -> mx.array:
        return decode_tbe_metal_v4(
            self.container, variant=self.variant,
            broadcast_load=self.broadcast_load, eval_now=eval_now,
        )

    def __call__(self, x: mx.array) -> mx.array:
        flat = x.reshape(-1, x.shape[-1])
        if int(flat.shape[-1]) != self.in_features:
            raise TBELinearMLXV4Error(
                f"input has {int(flat.shape[-1])} features, weight expects "
                f"{self.in_features}"
            )
        w = self.decode()
        y = mx.matmul(flat.astype(mx.bfloat16), w.T)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*x.shape[:-1], self.out_features)


__all__ = ["TBELinearMLXV4", "TBELinearMLXV4Error"]
