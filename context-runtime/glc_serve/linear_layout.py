"""Prepare arbitrary leading dimensions for a row-wise linear kernel."""

from __future__ import annotations

from typing import Callable

import torch


def prepare_linear_rows(
    x: torch.Tensor, y: torch.Tensor, K: int, N: int
) -> tuple[torch.Tensor, torch.Tensor, Callable[[], None]]:
    """Return matrix views/copies and a callback that commits output to ``y``.

    The native kernel consumes contiguous (or otherwise supported) row matrices.
    ``xrows`` is normalized to contiguous row-major storage when needed. Output
    uses a direct view only when flattening preserves the requested shape and
    has no internal overlap; otherwise the callback copies the temporary back.
    """
    if isinstance(K, bool) or not isinstance(K, int) or K <= 0:
        raise ValueError("K must be a positive integer")
    if isinstance(N, bool) or not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer")
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("x and y must be tensors")
    if x.ndim < 1 or x.shape[-1] != K:
        raise ValueError("x must have rank >= 1 and trailing dimension K")
    if tuple(y.shape) != (*x.shape[:-1], N):
        raise ValueError("y shape must equal (*x.shape[:-1], N)")
    if x.dtype != torch.bfloat16 or y.dtype != torch.bfloat16:
        raise TypeError("x and y must have dtype bfloat16")
    if x.device != y.device:
        raise ValueError("x and y must be on the same device")

    # A sufficient injectivity proof for standard transposes/slices/permutations.
    # Reject exotic as_strided layouts whose non-overlap this proof cannot show.
    dims = sorted(
        (stride, size) for size, stride in zip(y.shape, y.stride()) if size > 1
    )
    covered = 1
    nonoverlap = True
    for stride, size in dims:
        if stride < covered:
            nonoverlap = False
            break
        covered += (size - 1) * stride
    if not nonoverlap:
        raise ValueError("y has internal output overlap")
    try:
        aliases = torch._C._overlaps(x, y)
    except (AttributeError, RuntimeError):
        aliases = True  # Fail closed where the overlap query is unavailable.
    if aliases:
        raise ValueError("x and y may not alias")

    rows = 1
    for size in x.shape[:-1]:
        rows *= size
    if x.ndim == 2:
        xrows = x
    else:
        xrows = x.reshape(rows, K)
    native_rows = (xrows.stride(1) == 1 and xrows.stride(0) >= K
                   and xrows.stride(0) % 8 == 0 and xrows.data_ptr() % 16 == 0)
    if not native_rows and not xrows.is_contiguous():
        xrows = xrows.contiguous()
    if rows and xrows.data_ptr() % 16:
        # contiguous() may return the same unaligned storage-offset view.
        xrows = xrows.clone(memory_format=torch.contiguous_format)

    direct = False
    try:
        yrows = y if y.ndim == 2 else y.view(rows, N)
        direct = yrows.stride(1) == 1 and yrows.stride(0) >= N
    except RuntimeError:
        pass
    if not direct:
        yrows = torch.empty((rows, N), dtype=y.dtype, device=y.device)

    def commit() -> None:
        if not direct:
            y.copy_(yrows.reshape(y.shape))

    return xrows, yrows, commit
