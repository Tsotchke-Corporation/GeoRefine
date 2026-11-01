"""``TBELinearMLX`` -- the Metal twin of ``tbe_modules.GLCTBELinear``'s M > 16 arm.

``GLCTBELinear`` on CUDA dispatches on ``M``: fused fragments for ``M <= 16``,
decode-into-a-shared-scratch-buffer-then-``torch.matmul`` for ``M > 16``.  The
fused arm is CUDA tensor-core-specific (``mma16`` layout, PTX ``mma.sync``)
and is explicitly out of scope for this port.  This module ports the OTHER
arm -- decode-then-matmul -- which has no CUDA-specific instruction in it at
all: it is "read the compressed bytes, produce a dense bf16 tile, call the
platform's own matmul", and that is exactly what :func:`decode_tbe_metal` and
``mx.matmul`` give on Apple Silicon.

WHERE THE "TRANSIENT POOL" DOES AND DOES NOT PORT
--------------------------------------------------
``TBETransientPool`` (``tbe_modules.py``) exists because CUDA/torch tensors
are mutable, addressable buffers: the pool allocates ONE bf16 buffer sized to
the model's largest coded linear and every layer's forward decodes INTO that
same memory, so at most one dense materialisation is resident at a time and
it is a single allocation for the model's life.

MLX arrays are not that -- they are immutable graph values under lazy
evaluation, and ``mx.fast.metal_kernel`` always allocates a fresh output
array; there is no public API to decode into a caller-supplied buffer.  What
plays the pool's role here is MLX's own caching allocator: freed array
buffers are held in a reuse pool (``mx.get_cache_memory()``) rather than
returned to the OS, so a sequence of same-sized decode calls does not grow
resident memory call over call -- it recycles the previous call's freed
block.  ``mx.clear_cache()`` (called between benchmark rotations, never
inside a forward) is the explicit "give it back" this module relies on to
keep total Metal allocation bounded, which is the property the CUDA pool was
built to guarantee.  This is documented rather than silently assumed: a
caller that needs a hard guarantee should call ``mx.clear_cache()`` after
this module's ``__call__`` returns, exactly as the benchmark script does.
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx

from .tbe_decode_mlx import TBEDeviceMLX, decode_tbe_metal


class TBELinearMLXError(RuntimeError):
    """A TBE-served MLX module cannot be built or run as requested."""


class TBELinearMLX:
    """Decode-then-matmul linear layer over a Metal-resident TBE container.

    Not an ``mlx.nn.Module`` subclass on purpose: this module owns no
    trainable parameters (the container is a fixed, already-compressed
    weight) and mixing it into the ``nn.Module`` parameter tree would make it
    a candidate for ``mx.nn.Module.parameters()``/optimizer traversal, which
    is never correct for compressed weight bytes.  It is a plain callable,
    the same shape ``GLCTBELinear.forward`` is (a method on a state holder),
    without inheriting machinery this state does not need.
    """

    def __init__(self, container: TBEDeviceMLX, bias: Optional[mx.array] = None) -> None:
        self.container = container
        self.bias = bias
        self.out_features, self.in_features = container.shape

    @property
    def resident_bytes(self) -> int:
        return self.container.resident_bytes

    @property
    def dense_bytes(self) -> int:
        return self.container.dense_bytes

    def decode(self) -> mx.array:
        """Decode the weight to a bf16 ``[out_features, in_features]`` tile.

        See the module docstring for why this is not a pool-owned mutable
        buffer the way the CUDA transient pool's ``get()`` is.
        """
        return decode_tbe_metal(self.container)

    def __call__(self, x: mx.array) -> mx.array:
        flat = x.reshape(-1, x.shape[-1])
        if int(flat.shape[-1]) != self.in_features:
            raise TBELinearMLXError(
                f"input has {int(flat.shape[-1])} features, weight expects "
                f"{self.in_features}"
            )
        w = self.decode()
        y = mx.matmul(flat.astype(mx.bfloat16), w.T)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*x.shape[:-1], self.out_features)


__all__ = ["TBELinearMLX", "TBELinearMLXError"]
