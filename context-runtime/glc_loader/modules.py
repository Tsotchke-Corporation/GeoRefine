"""Serving modules that hold a GLC-FWP1 container instead of a dense weight.

Three backends, all producing identical bits:

``materialize``  decode once at load, keep a dense bf16 weight.  No VRAM saving.
                 Maximum compatibility -- this is a plain ``nn.Linear`` after
                 load and every downstream tool behaves as if nothing happened,
                 and its logits are bit-identical to the dense model's.
``resident``     keep the planes resident, decode inside ``forward``.  Also
                 bit-identical to dense, because it hands cuBLAS the same
                 weight in the same shape.  Pure torch, so it runs on CPU, MPS
                 and CUDA.  Slow: it decodes on every call.
``triton``       compute-in-domain: the GEMV reads the compressed bytes and
                 reconstructs weights in registers, so no dense weight is ever
                 written to HBM.  Requires CUDA + Triton; falls back loudly,
                 never silently.

"All producing identical bits" was the old summary of this file and it was
wrong in a way worth stating precisely, because it is the difference between
two claims a customer cares about separately.  ALL THREE decode the container
bit-exactly -- that is the lossless claim, and it is verified element by
element.  All three are deterministic run to run.  What only ``materialize``
and ``resident`` have is bitwise equality with what cuBLAS would have computed
from a dense weight, and that is equality with cuBLAS's rounding rather than
with the value: measured against the same checkpoint in float64, ``triton`` is
CLOSER to the exact logits than cuBLAS is.  See :func:`resolve_backend`.

The default is ``triton`` wherever it runs and ``materialize`` otherwise.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .container import FWP1Tensor, decode_fwp1, decode_fwp1_rows, load_kernels

BACKENDS = ("materialize", "resident", "triton")


class GLCBackendError(RuntimeError):
    """A requested backend is not available on this machine."""


def resolve_backend(requested: Optional[str], device: torch.device) -> str:
    if requested is not None and requested not in BACKENDS:
        raise GLCBackendError(
            f"unknown backend {requested!r}; choose one of {BACKENDS}"
        )
    if requested == "triton":
        if load_kernels() is None:
            raise GLCBackendError(
                "backend='triton' requested but Triton kernels are unavailable "
                "(needs a CUDA device and the `triton` package). Refusing to "
                "silently substitute a different backend."
            )
        return "triton"
    if requested is not None:
        return requested
    # THE DEFAULT IS `triton` WHERE IT RUNS, AND THAT IS A CORRECTED DEFAULT.
    #
    # It used to be `materialize` everywhere, which handed a client following
    # the README exactly ZERO of the memory the container exists to save.  The
    # reason given was that `triton` is not bit-identical to the dense model.
    # That reason confuses three different properties, and separating them
    # reverses the conclusion:
    #
    #   1. CONTAINER BIT-EXACTNESS -- decode(encode(W)) == W.  Every backend
    #      has it, including this one; it is a property of the container and
    #      it is verified element by element at build time.
    #   2. DETERMINISM -- the same input gives the same output twice.  Every
    #      backend has it; the split-K reduce is a fixed `tl.static_range` and
    #      `tl.atomic_add` was refused precisely because arrival-order
    #      accumulation is not.
    #   3. BITWISE EQUALITY WITH cuBLAS.  Only `materialize` and `resident`
    #      have it -- and it is equality with cuBLAS's ROUNDING, not with the
    #      value.
    #
    # Measured on an RTX PRO 6000 Blackwell, Llama-3.2-1B-Instruct, eager
    # attention, batch 1, one arm per process, against the same checkpoint run
    # in float64 -- which holds these bf16 weights exactly and is therefore the
    # arithmetically exact answer for them, over 9,490,944 logits:
    #
    #                     worst |err| vs exact   mean |err| vs exact   top-1
    #   dense / materialize          0.527292             0.023044     74/74
    #   triton                       0.314947             0.021833     74/74
    #
    # `triton` is CLOSER TO THE TRUE VALUE than cuBLAS is, on the worst case by
    # 1.67x and on the mean by 5.3%, and picks the same top-1 token at every one
    # of the 74 positions.  It differs from dense on 7,929,487 of those logits,
    # and every one of those differences is dense being further from exact than
    # we are.  Defaulting away from it was protecting a baseline, not accuracy.
    #
    # And what that default costs, same run:
    #
    #   arm            resident   peak     tok/s
    #   dense           2357.1   2397.7   150.19
    #   materialize     2357.1   2397.7   152.19    <- the old default
    #   triton          1842.5   1912.1   108.35
    #
    # Off the Triton path there is no configuration that is both fast and
    # smaller, so the fallback is `materialize` and the client is told plainly
    # that `resident` is the lever if VRAM is what they need.  A default should
    # be the best available answer, not the one that changes nothing.
    if device.type == "cuda" and load_kernels() is not None:
        return "triton"
    return "materialize"


class _ContainerHolder(nn.Module):
    """Registers the three planes as buffers so ``.to()`` and state_dict work."""

    def __init__(self, t: FWP1Tensor, backend: str) -> None:
        super().__init__()
        self.register_buffer("smb", t.smb, persistent=True)
        self.register_buffer("eidx", t.eidx, persistent=True)
        self.register_buffer("pal", t.pal, persistent=True)
        self.glc_shape = (int(t.shape[0]), int(t.shape[1]))
        self.glc_group = int(t.group)
        self.backend = backend

    @property
    def container(self) -> FWP1Tensor:
        return FWP1Tensor(
            smb=self.smb, eidx=self.eidx, pal=self.pal,
            shape=self.glc_shape, group=self.glc_group,
        )

    @property
    def resident_bytes(self) -> int:
        return (
            self.smb.numel() + self.eidx.numel() + self.pal.numel()
        )

    @property
    def dense_bytes(self) -> int:
        return self.glc_shape[0] * self.glc_shape[1] * 2


class GLCLinear(_ContainerHolder):
    """``nn.Linear`` whose weight lives in a GLC-FWP1 container."""

    def __init__(
        self,
        t: FWP1Tensor,
        bias: Optional[torch.Tensor],
        backend: str,
    ) -> None:
        super().__init__(t, backend)
        self.out_features, self.in_features = self.glc_shape
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias, requires_grad=False)
        self._kernels = load_kernels() if backend == "triton" else None
        if backend == "triton" and self._kernels is None:
            raise GLCBackendError("triton backend selected without kernels")

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, backend={self.backend}, "
            f"group={self.glc_group}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "triton":
            return self._forward_triton(x)
        w = decode_fwp1(self.container)
        return F.linear(x, w, self.bias)

    def _forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        k = self._kernels
        t = self.container
        flat = x.reshape(-1, x.shape[-1])
        if flat.shape[0] == 1:
            y = k.fwp1_gemv(t, flat[0], self.bias)
            y = y.reshape(1, -1)
        else:
            y = k.fwp1_gemm(t, flat, self.bias)
        return y.reshape(*x.shape[:-1], self.out_features).to(x.dtype)


class GLCEmbedding(_ContainerHolder):
    """``nn.Embedding`` whose table lives in a GLC-FWP1 container.

    A lookup decodes only the rows requested.  On a 128k-row table that is the
    difference between touching 525 MB and touching a few kilobytes.
    """

    def __init__(
        self,
        t: FWP1Tensor,
        padding_idx: Optional[int],
        backend: str,
    ) -> None:
        super().__init__(t, backend)
        self.num_embeddings, self.embedding_dim = self.glc_shape
        self.padding_idx = padding_idx

    def extra_repr(self) -> str:
        return (
            f"{self.num_embeddings}, {self.embedding_dim}, "
            f"backend={self.backend}, group={self.glc_group}"
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1)
        rows = decode_fwp1_rows(self.container, flat)
        out = rows.reshape(*ids.shape, self.embedding_dim)
        if self.padding_idx is not None:
            out = out.masked_fill(
                (ids == self.padding_idx).unsqueeze(-1), 0.0
            )
        return out


class GLCTiedLMHead(nn.Module):
    """The output head sharing ONE container with the input embedding.

    Tied weights are the whole reason the embedding is worth coding: a single
    ``[vocab, hidden]`` matrix that both the token lookup and the logit
    projection read.  Storing it once, coded, is what turns 27% of a small
    model's resident weights from raw into compressed.

    The projection is blocked over vocabulary rows so peak transient memory is
    ``block_rows * hidden * 2`` bytes rather than the whole table.
    """

    def __init__(
        self,
        source: GLCEmbedding,
        bias: Optional[torch.Tensor],
        backend: str,
        block_rows: int = 0,
    ) -> None:
        super().__init__()
        if not block_rows:
            # Default to one block, i.e. no blocking.  Splitting the projection
            # over vocabulary rows changes which GEMM cuBLAS picks and
            # therefore its accumulation order over K, so the logits are NOT
            # bit-identical to the dense model.
            #
            # THERE IS NO SAFE BLOCK SIZE.  Re-measured 2026-08-08 on an RTX PRO
            # 6000 Blackwell (cuBLAS 13.0) by blocking a plain bf16 `F.linear`
            # -- no container involved, so this is a property of the BLAS and
            # not of us -- over three head geometries and five block sizes at
            # M in {1, 2, 8, 17, 64}.  Worst cell: 128256x2048 at M=17, block
            # 4096, 910226 of 2180352 words differing, max 0.03125.  The
            # pattern is not monotone in the block size and not monotone in M:
            # 16384 is exact at M=17 and wrong on 24% of words at M=8, while
            # 8192 is exact at M=8 and wrong at M=17.  A block size chosen
            # because it was exact on one shape at one batch is a coincidence
            # a customer would ship.
            #
            # So blocking is a peak-memory control the caller may ask for, it
            # is never a default, and the loader's carry planner does not use
            # it: to lower peak while staying bit-identical it carries the
            # whole table dense instead and keeps the saving elsewhere.  See
            # `.icc/evidence/serving-codec-complete-20260808/rowblock_bits.json`.
            block_rows = int(source.glc_shape[0])
        # Deliberately NOT a submodule: the container is registered once, on the
        # embedding, so parameter counts and .to() moves happen exactly once.
        object.__setattr__(self, "_source", source)
        self.backend = backend
        self.block_rows = int(block_rows)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias, requires_grad=False)
        self._kernels = load_kernels() if backend == "triton" else None
        self.out_features, self.in_features = source.glc_shape

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"tied=True, backend={self.backend}, block_rows={self.block_rows}"
        )

    @property
    def weight(self) -> torch.Tensor:  # pragma: no cover - compatibility shim
        return decode_fwp1(self._source.container)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self._source.container
        flat = x.reshape(-1, x.shape[-1])
        if self.backend == "triton" and self._kernels is not None:
            if flat.shape[0] == 1:
                y = self._kernels.fwp1_gemv(t, flat[0], self.bias).reshape(1, -1)
            else:
                y = self._kernels.fwp1_gemm(t, flat, self.bias)
            return y.reshape(*x.shape[:-1], self.out_features).to(x.dtype)
        n = self.out_features
        if self.block_rows >= n:
            w = decode_fwp1(t)
            y = F.linear(flat, w.to(flat.dtype), self.bias)
            return y.reshape(*x.shape[:-1], n)
        chunks = []
        for r0 in range(0, n, self.block_rows):
            r1 = min(n, r0 + self.block_rows)
            w = decode_fwp1(t.row_slice(r0, r1))
            b = None if self.bias is None else self.bias[r0:r1]
            chunks.append(F.linear(flat, w.to(flat.dtype), b))
        y = torch.cat(chunks, dim=-1)
        return y.reshape(*x.shape[:-1], n)


__all__ = [
    "BACKENDS",
    "GLCBackendError",
    "GLCEmbedding",
    "GLCLinear",
    "GLCTiedLMHead",
    "resolve_backend",
]
