"""``GLCTBELinear`` -- an ``nn.Linear`` whose weight lives in a GLC-TBE container.

Per-M dispatch, following ``docs/research/GLC_KERNEL_DECISION_20260901.md``
Addendum C's decision:

  M <= 16   fused fragments (:func:`tbe_mma_gemm`) -- the container is never
            materialised dense; the MMA consumes decoded fragments straight
            from registers.
  M > 16    decode-only (:func:`tbe_mma_decode`) into a per-layer TRANSIENT
            bf16 buffer, then ``torch.matmul``.  The buffer is owned by a
            :class:`TBETransientPool` shared across every ``GLCTBELinear`` in
            a model and sized to the largest coded linear, so at most ONE
            dense materialisation is ever resident at a time and it is freed
            with the pool -- never a persistent dense copy of any weight.

No dense fallback exists anywhere in this module: a device this kernel was
not built for, or a shape the kernel does not cover, raises a typed error
rather than silently routing through ``torch.matmul`` on a decoded copy that
was never asked for.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .tbe_container import TBETensor
from .tbe_mma import (
    MAX_M,
    TBEDevice,
    TBEMMAError,
    resolve_tbe_mma_arch,
    tbe_mma_decode,
    tbe_mma_gemm,
    upload_tbe,
)


class TBEBackendError(RuntimeError):
    """A TBE-served module cannot be built or run on this machine."""


class TBETransientPool:
    """One reused bf16 scratch buffer, shared across every M > 16 linear.

    ``forward()`` for a linear with M > 16 needs a dense ``[out, in]`` bf16
    tile to hand ``torch.matmul``.  Allocating one per call per layer is a
    transient the size of the largest weight in the model, repeated at every
    such call; sharing ONE buffer sized to the largest coded linear bounds
    that transient to a single allocation for the life of the model, exactly
    the reasoning ``release/glc_loader/loader.py``'s ``plan_dense_carry``
    already applies to the ``resident`` FWP1 backend.  ``reserve`` grows the
    buffer (never shrinks it); ``get`` returns a ``[n, k]`` view into it;
    ``free`` drops the reference so the allocator can reclaim it.
    """

    def __init__(self, device: torch.device) -> None:
        if torch.device(device).type != "cuda":
            raise TBEBackendError(
                f"the transient pool is CUDA-only, got device={device}"
            )
        self.device = torch.device(device)
        self._buf: Optional[torch.Tensor] = None
        self._capacity_elems = 0
        self.n_grows = 0
        self.n_gets = 0

    @property
    def capacity_bytes(self) -> int:
        return int(self._capacity_elems) * 2  # bf16

    def reserve(self, numel: int) -> None:
        """Ensure the pool can hand out ``numel`` contiguous bf16 elements."""
        numel = int(numel)
        if numel <= self._capacity_elems:
            return
        self._buf = torch.empty(numel, dtype=torch.bfloat16, device=self.device)
        self._capacity_elems = numel
        self.n_grows += 1

    def get(self, n: int, k: int) -> torch.Tensor:
        """A ``[n, k]`` bf16 view into the shared buffer, growing it if short.

        The view is only valid until the NEXT call to :meth:`get` on this
        pool -- callers must consume it (matmul, then discard) before asking
        for another layer's tile, which is exactly the per-layer decode order
        of a forward pass.
        """
        numel = int(n) * int(k)
        self.reserve(numel)
        self.n_gets += 1
        assert self._buf is not None
        return self._buf[:numel].view(int(n), int(k))

    def free(self) -> None:
        """Release the buffer.  Called on unload; leaves nothing resident."""
        self._buf = None
        self._capacity_elems = 0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"TBETransientPool(capacity_bytes={self.capacity_bytes}, "
            f"n_grows={self.n_grows}, n_gets={self.n_gets}, device={self.device})"
        )


class TBEPoolRegistry:
    """One :class:`TBETransientPool` per (device, tag) -- never one global pool.

    A single-GPU model can share one scratch buffer across every coded
    linear.  A model split across eight cards cannot: a buffer allocated on
    ``cuda:0`` is not addressable from ``cuda:3``'s kernels, and handing it to
    them either raises a cross-device error or -- worse, through a
    peer-accessible allocation -- silently serialises every big-batch decode
    through one card's memory.  The registry keeps placement an invariant:
    a linear only ever gets the pool bound to its OWN device.

    ``tag`` separates consumers whose decodes could otherwise interleave.
    :class:`TBETransientPool` hands out a view that is valid only until the
    next ``get`` on that pool, which is safe for a chain of decode-then-
    consume calls but not for two consumers holding views at once; the
    linear path and the expert-bank path therefore take different tags.
    """

    def __init__(self) -> None:
        self._pools: dict = {}

    def pool_for(self, device: torch.device, tag: str = "linear") -> TBETransientPool:
        dev = torch.device(device)
        key = (str(dev), str(tag))
        pool = self._pools.get(key)
        if pool is None:
            pool = TBETransientPool(dev)
            self._pools[key] = pool
        return pool

    @property
    def pools(self) -> dict:
        return dict(self._pools)

    def capacity_bytes_by_device(self) -> dict:
        out: dict = {}
        for (dev, _tag), pool in self._pools.items():
            out[dev] = out.get(dev, 0) + int(pool.capacity_bytes)
        return out

    @property
    def total_capacity_bytes(self) -> int:
        return sum(int(p.capacity_bytes) for p in self._pools.values())

    def free_all(self) -> None:
        for pool in self._pools.values():
            pool.free()

    def __len__(self) -> int:  # pragma: no cover - cosmetic
        return len(self._pools)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"TBEPoolRegistry(n_pools={len(self._pools)}, "
            f"total_capacity_bytes={self.total_capacity_bytes})"
        )


class GLCTBELinear(nn.Module):
    """``nn.Linear`` whose weight lives in a GLC-TBE container.

    Construction uploads the container to ``device`` (:func:`upload_tbe`) and
    resolves ``GLC_TBE_MMA_ARCH`` from the device's compute capability unless
    it is already set (:func:`resolve_tbe_mma_arch`) -- both raise a typed
    :class:`TBEBackendError`-compatible error on CPU or an unsupported
    device, before any weight is ever touched.  There is no path in this
    class that decodes to a persistent dense parameter.
    """

    def __init__(
        self,
        t: TBETensor,
        bias: Optional[torch.Tensor],
        device: torch.device,
        transient_pool: Optional[TBETransientPool] = None,
    ) -> None:
        super().__init__()
        dev = torch.device(device)
        if dev.type != "cuda":
            raise TBEBackendError(
                f"GLCTBELinear requires a CUDA device, got {dev}. TBE fragment "
                "kernels have no CPU path and this module never falls back to "
                "one silently."
            )
        try:
            self.arch = resolve_tbe_mma_arch(dev)
        except TBEMMAError as exc:
            raise TBEBackendError(str(exc)) from exc
        try:
            self._container = upload_tbe(t, dev)
        except TBEMMAError as exc:
            raise TBEBackendError(f"failed to upload TBE container: {exc}") from exc
        self.out_features, self.in_features = self._container.shape
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.to(dev), requires_grad=False)
        self.transient_pool = transient_pool or TBETransientPool(dev)
        if self.transient_pool.device != dev:
            raise TBEBackendError(
                f"transient_pool is bound to {self.transient_pool.device}, "
                f"this linear is on {dev}"
            )
        # Reserve now so the pool's peak size is known at conversion time
        # rather than discovered lazily on the first big-batch forward.
        self.transient_pool.reserve(self.out_features * self.in_features)

    @property
    def container(self) -> TBEDevice:
        return self._container

    @property
    def resident_bytes(self) -> int:
        return self._container.resident_bytes

    @property
    def dense_bytes(self) -> int:
        return self.out_features * self.in_features * 2

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, backend=tbe_mma, arch={self.arch}, "
            f"ratio={self._container.ratio():.4f}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        m = int(flat.shape[0])
        if m <= MAX_M:
            y = tbe_mma_gemm(self._container, flat, self.bias)
        else:
            w = tbe_mma_decode(
                self._container,
                out=self.transient_pool.get(self.out_features, self.in_features),
            )
            y = torch.matmul(flat.to(torch.bfloat16), w.t())
            if self.bias is not None:
                y = y + self.bias
        return y.reshape(*x.shape[:-1], self.out_features).to(x.dtype)


class GLCTBEExpertBank(nn.Module):
    """A fused 3-D MoE expert weight, coded one expert at a time.

    WHY THIS CLASS EXISTS.  transformers does NOT store a mixture-of-experts
    FFN as ``num_experts`` ``nn.Linear`` modules.  ``Qwen3NextExperts`` (and
    ``Qwen3_5MoeExperts``, and every recent MoE port that follows the same
    shape) stores two 3-D ``nn.Parameter`` tensors::

        gate_up_proj  [E, 2 * intermediate, hidden]
        down_proj     [E, hidden, intermediate]

    and its forward indexes them per routed expert::

        gate, up = F.linear(state, self.gate_up_proj[expert_idx]).chunk(2, -1)
        out      = F.linear(h,     self.down_proj[expert_idx])

    ``GLCTBELinear`` cannot see any of that: ``enable_hf_tbe_serving``'s
    eligibility scan walks ``nn.Linear`` modules, and there are none here.
    On a 512-expert model that is the overwhelming majority of the weight,
    which would be served bf16 while the receipt quoted a ratio measured over
    the handful of attention projections -- a ratio the artifact could not
    deliver.

    WHAT IT DOES.  It holds ONE container per expert slice and returns a
    decoded ``[out, in]`` bf16 tile from ``__getitem__``, so the module it
    replaces keeps working verbatim: ``self.gate_up_proj[expert_idx]`` still
    yields the same matrix, byte for byte (every container is certified
    bit-exact at install time), and only the experts the router actually hit
    are ever decoded -- one tile at a time, into the shared transient pool,
    never a dense copy of all E experts.

    WHAT IT REFUSES.  The returned tile is a POOL VIEW, valid until the next
    ``__getitem__`` on the same pool.  That is exactly the lifetime the
    per-expert loop above needs (each tile is consumed by the ``F.linear``
    on the very next expression) and it is NOT enough for an implementation
    that batches all experts at once -- a ``torch.bmm(x, self.gate_up_proj)``
    style forward.  Such a forward does not index this object at all; it
    passes the module itself to an op that requires a Tensor and raises
    immediately.  There is no path here that returns something a batched op
    would silently mis-consume.
    """

    def __init__(
        self,
        containers,
        device: torch.device,
        transient_pool: Optional[TBETransientPool] = None,
    ) -> None:
        super().__init__()
        dev = torch.device(device)
        if dev.type != "cuda":
            raise TBEBackendError(
                f"GLCTBEExpertBank requires a CUDA device, got {dev}. TBE "
                "fragment kernels have no CPU path."
            )
        containers = list(containers)
        if not containers:
            raise TBEBackendError("an expert bank needs at least one expert")
        try:
            self.arch = resolve_tbe_mma_arch(dev)
        except TBEMMAError as exc:
            raise TBEBackendError(str(exc)) from exc
        uploaded = []
        for i, c in enumerate(containers):
            try:
                uploaded.append(upload_tbe(c, dev))
            except TBEMMAError as exc:
                raise TBEBackendError(
                    f"expert {i}: failed to upload TBE container: {exc}"
                ) from exc
        shapes = {tuple(u.shape) for u in uploaded}
        if len(shapes) != 1:
            raise TBEBackendError(
                f"every expert in a bank must share one shape; got {sorted(shapes)}"
            )
        self._experts = uploaded
        self.num_experts = len(uploaded)
        self.out_features, self.in_features = uploaded[0].shape
        self.device_ = dev
        self.transient_pool = transient_pool or TBETransientPool(dev)
        if self.transient_pool.device != dev:
            raise TBEBackendError(
                f"transient_pool is bound to {self.transient_pool.device}, "
                f"this expert bank is on {dev}"
            )
        self.transient_pool.reserve(self.out_features * self.in_features)

    @property
    def shape(self):
        return (self.num_experts, self.out_features, self.in_features)

    @property
    def resident_bytes(self) -> int:
        return sum(int(u.resident_bytes) for u in self._experts)

    @property
    def dense_bytes(self) -> int:
        return self.num_experts * self.out_features * self.in_features * 2

    def container(self, index: int):
        return self._experts[int(index)]

    def __len__(self) -> int:
        return self.num_experts

    def __getitem__(self, index) -> torch.Tensor:
        """Decode expert ``index`` into the shared pool and return the view."""
        if isinstance(index, torch.Tensor):
            if index.numel() != 1:
                raise TBEBackendError(
                    "an expert bank is indexed by ONE expert at a time; a "
                    f"{index.numel()}-element index would need a batched "
                    "decode this container does not provide"
                )
            index = int(index.reshape(()).item())
        if not isinstance(index, int) or isinstance(index, bool):
            raise TBEBackendError(
                f"expert index must be an int, got {type(index).__name__}. "
                "Slicing or advanced indexing an expert bank is refused "
                "rather than silently materialising every expert."
            )
        if index < 0:
            index += self.num_experts
        if not 0 <= index < self.num_experts:
            raise IndexError(
                f"expert {index} out of range for {self.num_experts} experts"
            )
        return tbe_mma_decode(
            self._experts[index],
            out=self.transient_pool.get(self.out_features, self.in_features),
        )

    def forward(self, *args, **kwargs):  # pragma: no cover - never callable
        raise TBEBackendError(
            "GLCTBEExpertBank is a weight store, not a layer; it is indexed "
            "per routed expert by the MoE block that owns it"
        )

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, out_features={self.out_features}, "
            f"in_features={self.in_features}, backend=tbe_mma, arch={self.arch}"
        )


__all__ = [
    "GLCTBEExpertBank",
    "GLCTBELinear",
    "TBEBackendError",
    "TBEPoolRegistry",
    "TBETransientPool",
]
