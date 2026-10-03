"""Opt-in sm_80/86/89/120 host module for the TBE fragment kernel.

Static, source-controlled twin of ``experiments.georefine._glc_tbe_mma``'s
host-side (Python) logic -- the CUDA source and its ``@torch.utils.cpp_extension
.load`` build step are a separately vendored artifact (mirroring how
``container.py``'s Triton kernels ship as ``fwp1_kernels.py`` beside it,
copied in at release-build time rather than imported from this repository).
Nothing in this file reaches back into ``experiments.georefine``;
``tests/test_glc_release.py`` asserts that for every ``*.py`` in this package.

``docs/research/GLC_KERNEL_DECISION_20260901.md`` Addendum C is the decision
record: rung-1d is the kernel that ships, per-M dispatch (fused fragments for
M <= 16, decode-then-matmul for M > 16), one arch build per device class
(8.0 / 8.6 / 8.9 / 12.0), and no dense fallback -- a device this kernel was
not built for is refused, not silently downgraded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch

from .tbe_container import HEADER_BYTES, MODE_W6Z, MODE_W7, TILE, TBETensor

WARPS_PER_CTA = 4
NSLAB = 2
MMA_M = 16
MMA_N = 8
MMA_K = 16
N_PER_CTA = WARPS_PER_CTA * NSLAB * MMA_N   # 64
MAX_M = MMA_M

#: The one layout whose element order matches the fragment kernel's lane map.
LAYOUT = "mma16"

#: Zero bytes appended to esc[] on upload.  The kernel issues four
#: unconditional byte loads at ``esc + base``; padding keeps that read valid
#: for every base <= E.  It is accounted, not hidden: ``TBEDevice.
#: allocated_bytes`` separates it from ``resident_bytes``.
ESC_PAD = 16

#: (major, minor) capability -> the arch string this module hands to
#: ``GLC_TBE_MMA_ARCH`` / ``TORCH_CUDA_ARCH_LIST``.  The decision doc's four
#: supported device classes: A100 (8.0), RTX 3090/A6000 (8.6), L40S/RTX 4090
#: (8.9), Blackwell RTX PRO 6000 (12.0).  A capability outside this table has
#: no built kernel and is refused rather than guessed at.
SUPPORTED_CAPABILITIES = {
    (8, 0): "8.0",
    (8, 6): "8.6",
    (8, 9): "8.9",
    (12, 0): "12.0",
}


class TBEMMAError(RuntimeError):
    """Raised when the TBE fragment-kernel contract is not satisfied."""


def exponent_table(mode: int, base: int) -> Tuple[int, int]:
    """The eight exponent bytes the kernel reads with one PRMT."""
    if mode not in (MODE_W7, MODE_W6Z):
        raise TBEMMAError(f"unknown TBE mode {mode!r}")
    if not 0 <= int(base) <= 255:
        raise TBEMMAError(f"base must be a byte, got {base}")
    table = [0] * 8
    span = 7 if mode == MODE_W7 else 6
    for code in range(1, span + 1):
        value = int(base) + code - 1
        if not 0 <= value <= 255:
            raise TBEMMAError(
                f"window [{base}, {base + span - 1}] leaves the exponent byte"
            )
        table[code] = value
    if mode == MODE_W6Z:
        table[7] = 0
    e01 = table[0] | (table[1] << 8) | (table[2] << 16) | (table[3] << 24)
    e23 = table[4] | (table[5] << 8) | (table[6] << 16) | (table[7] << 24)
    return e01, e23


def _as_int32_words(values: torch.Tensor) -> torch.Tensor:
    v = values.reshape(-1).to(torch.int64)
    # Container words are stored in int32 tensors, so values with the high bit
    # set arrive as negative signed integers.  Accept those bit patterns as
    # well as callers that provide the equivalent unsigned values.
    if int(v.numel()) and (
        int(v.min().item()) < -0x80000000
        or int(v.max().item()) > 0xFFFFFFFF
    ):
        raise TBEMMAError("container word does not fit 32 bits")
    v = torch.where((v >= 0x80000000) & (v <= 0xFFFFFFFF), v - 0x100000000, v)
    return v.to(torch.int32).contiguous()


@dataclass
class TBEDevice:
    """One TBE tensor resident on one CUDA device, byte for byte."""

    planes: torch.Tensor
    smb: torch.Tensor
    esc: torch.Tensor
    sbbase: torch.Tensor
    shape: Tuple[int, int]
    mode: int
    base: int
    superblock: int
    tiles: int
    escapes: int
    e01: int
    e23: int
    esc_pad: int = 0

    @property
    def device(self) -> torch.device:
        return self.planes.device

    @property
    def numel(self) -> int:
        return int(self.shape[0]) * int(self.shape[1])

    @property
    def padding_bytes(self) -> int:
        return int(self.esc_pad)

    @property
    def array_bytes(self) -> int:
        return (int(self.planes.numel()) * 4 + int(self.smb.numel()) +
                int(self.esc.numel()) - int(self.esc_pad) +
                int(self.sbbase.numel()) * 4)

    @property
    def allocated_bytes(self) -> int:
        return self.resident_bytes + int(self.esc_pad)

    @property
    def resident_bytes(self) -> int:
        return self.array_bytes + HEADER_BYTES

    def bits_per_element(self) -> float:
        return self.resident_bytes * 8.0 / self.numel

    def ratio(self) -> float:
        return 16.0 / self.bits_per_element()


def upload_tbe(c: TBETensor, device: Any) -> TBEDevice:
    """Move an encoded tensor to one CUDA device, adding only esc padding."""
    if c.layout != LAYOUT:
        raise TBEMMAError(
            f"the fragment kernel is built on layout {LAYOUT!r}, got "
            f"{c.layout!r}; {LAYOUT!r} is the layout that gives a lane a "
            "contiguous 16-element run"
        )
    n, k = int(c.shape[0]), int(c.shape[1])
    if k % TILE:
        raise TBEMMAError(f"K must be a multiple of {TILE}, got {k}")
    if n % MMA_N:
        raise TBEMMAError(f"N must be a multiple of {MMA_N}, got {n}")
    dev = torch.device(device)
    if dev.type != "cuda":
        raise TBEMMAError(f"TBE fragment kernels are CUDA-only, got {dev}")
    e01, e23 = exponent_table(int(c.mode), int(c.base))
    esc = torch.cat([
        c.esc.reshape(-1).contiguous().to(torch.uint8),
        torch.zeros(ESC_PAD, dtype=torch.uint8),
    ])
    if int(esc.numel()) != int(c.esc.numel()) + ESC_PAD:
        raise TBEMMAError("esc padding did not land")
    if int(esc[int(c.esc.numel()):].max().item() if ESC_PAD else 0) != 0:
        raise TBEMMAError("esc padding must be zero")
    out = TBEDevice(
        planes=_as_int32_words(c.planes).to(dev),
        smb=c.smb.reshape(-1).contiguous().to(dev),
        esc=esc.to(dev),
        sbbase=_as_int32_words(c.sbbase).to(dev),
        shape=(n, k), mode=int(c.mode), base=int(c.base),
        superblock=int(c.superblock), tiles=int(c.tiles), escapes=int(c.escapes),
        e01=e01, e23=e23, esc_pad=ESC_PAD,
    )
    want = c.byte_size()["total"]
    if out.resident_bytes != want:
        raise TBEMMAError(
            f"upload changed the container size: {out.resident_bytes} resident "
            f"bytes against {want} encoded bytes"
        )
    if int(out.esc.numel()) < int(c.escapes) + ESC_PAD:
        raise TBEMMAError(
            f"esc[] holds {int(out.esc.numel())} bytes for {int(c.escapes)} "
            f"escapes; the kernel reads four bytes unconditionally at base <= "
            f"E and needs {ESC_PAD} bytes of padding past the last entry"
        )
    if out.allocated_bytes != want + ESC_PAD:
        raise TBEMMAError("esc padding is not accounted separately")
    return out


# ---------------------------------------------------------------------------
# arch selection -- device-derived, never guessed silently
# ---------------------------------------------------------------------------
def capability_to_arch(capability: Tuple[int, int]) -> str:
    """Map a CUDA compute capability to a supported ``GLC_TBE_MMA_ARCH`` string.

    Raises :class:`TBEMMAError` for any capability with no built kernel --
    the refusal surface this module exists to keep loud.  CPU has no
    capability at all and is refused earlier, by :func:`resolve_tbe_mma_arch`.
    """
    got = tuple(int(x) for x in capability)
    arch = SUPPORTED_CAPABILITIES.get(got)
    if arch is None:
        supported = ", ".join(
            f"{maj}.{minv} ({name})" for (maj, minv), name in
            sorted(SUPPORTED_CAPABILITIES.items())
        )
        raise TBEMMAError(
            f"CUDA capability {got[0]}.{got[1]} has no built TBE fragment "
            f"kernel; supported device classes are {supported}. Refusing "
            "rather than silently falling back to a different backend."
        )
    return arch


def resolve_tbe_mma_arch(device: Optional[Any] = None) -> str:
    """Choose ``GLC_TBE_MMA_ARCH`` for ``device``, honoring an existing setting.

    If ``GLC_TBE_MMA_ARCH`` is already set in the environment it is left
    alone -- an operator who pinned an arch explicitly is not overridden.
    Otherwise the arch is derived from ``torch.cuda.get_device_capability``
    and written into the environment so :mod:`torch.utils.cpp_extension`
    picks it up.  CPU and any unsupported capability raise
    :class:`TBEMMAError`; there is no silent fallback.
    """
    existing = os.environ.get("GLC_TBE_MMA_ARCH")
    if existing:
        return existing
    if not torch.cuda.is_available():
        raise TBEMMAError(
            "TBE fragment kernels are CUDA-only; no CUDA device is available "
            "to derive GLC_TBE_MMA_ARCH from, and CPU has no compute "
            "capability. Refusing rather than defaulting to a build."
        )
    target = (torch.device(device) if device is not None
              else torch.device("cuda", torch.cuda.current_device()))
    capability = torch.cuda.get_device_capability(target)
    arch = capability_to_arch(capability)
    os.environ["GLC_TBE_MMA_ARCH"] = arch
    return arch


def tbe_mma_arch() -> str:
    """The CUDA arch this extension is built for (env-var, not device-derived).

    Mirrors ``experiments.georefine._glc_tbe_mma.tbe_mma_arch``: reads
    ``GLC_TBE_MMA_ARCH`` verbatim.  Callers that want the arch chosen FROM the
    current device use :func:`resolve_tbe_mma_arch`, which sets this variable
    if it is not already set.
    """
    return os.environ.get("GLC_TBE_MMA_ARCH") or ""


def tbe_mma_capability() -> Tuple[int, int]:
    """The ``(major, minor)`` device capability :func:`tbe_mma_arch` names."""
    arch = tbe_mma_arch()
    if not arch:
        raise TBEMMAError(
            "GLC_TBE_MMA_ARCH is unset; call resolve_tbe_mma_arch(device) "
            "first so the arch is derived rather than assumed"
        )
    head = arch.split(";")[0].split("+")[0].strip()
    major, _, minor = head.partition(".")
    major_digits = "".join(ch for ch in major if ch.isdigit())
    minor_digits = "".join(ch for ch in minor if ch.isdigit())
    if not major_digits:
        raise TBEMMAError(f"GLC_TBE_MMA_ARCH={arch!r} is not an arch")
    return int(major_digits), int(minor_digits or "0")


def default_split_k(n: int, k: int, target_warps: int = 2048) -> int:
    """Deterministic k split, chosen only to fill the machine.

    Verbatim port of ``experiments.georefine._glc_tbe_mma.default_split_k``.
    One warp owns 16 output columns and walks its whole k chunk, so a warp
    count of N/16 is all the parallelism the shape offers; the split raises
    that without changing which thread owns an output -- partials are reduced
    in index order by a separate pass.
    """
    warps = max(1, int(n) // (NSLAB * MMA_N))
    tiles_per_row = max(1, int(k) // TILE)
    split = max(1, min(8, (int(target_warps) + warps - 1) // warps))
    return max(1, min(split, tiles_per_row))


def _validate_device_container(dev: TBEDevice) -> Tuple[int, int]:
    if not isinstance(dev, TBEDevice):
        raise TBEMMAError(f"expected a TBEDevice, got {type(dev).__name__}")
    n, k = int(dev.shape[0]), int(dev.shape[1])
    if k % TILE:
        raise TBEMMAError(f"K must be a multiple of {TILE}, got {k}")
    if n % MMA_N:
        raise TBEMMAError(f"N must be a multiple of {MMA_N}, got {n}")
    if dev.device.type != "cuda":
        raise TBEMMAError("the container must be CUDA-resident")
    if int(dev.esc.numel()) < int(dev.escapes) + ESC_PAD:
        raise TBEMMAError(
            "esc[] must carry at least "
            f"{ESC_PAD} zero bytes past its last entry; the escape path loads "
            "four bytes unconditionally"
        )
    capability = torch.cuda.get_device_capability(dev.device)
    expected = tbe_mma_capability()
    if tuple(capability) != expected:
        raise TBEMMAError(
            f"the TBE fragment kernel requires capability "
            f"{expected[0]}.{expected[1]}, got {tuple(capability)}"
        )
    return n, k


# ---------------------------------------------------------------------------
# kernel extension -- vendored the same way container.py vendors fwp1_kernels
# ---------------------------------------------------------------------------
def load_tbe_kernels() -> Optional[Any]:
    """Return the vendored native TBE-MMA extension module, or ``None``.

    Mirrors ``container.load_kernels()``: a speed/VRAM optimisation, never a
    correctness dependency, and the reason ``forward()`` refuses instead of
    substituting a different backend when this returns ``None``.  The
    extension is not built from this file -- ``_glc_tbe_mma_kernel.cu`` and
    its Python host wrapper are copied into a GLC-RELEASE artifact at build
    time (analogous to ``fwp1_kernels.py``), so an editable checkout of this
    package alone has nothing to import here and correctly returns ``None``.
    """
    if not torch.cuda.is_available():
        return None
    try:
        from . import tbe_mma_kernels  # type: ignore
    except Exception:
        return None
    return tbe_mma_kernels


def tbe_mma_available() -> bool:
    """Whether the native extension can be found and the arch matches."""
    if not torch.cuda.is_available():
        return False
    try:
        arch = resolve_tbe_mma_arch()
    except TBEMMAError:
        return False
    if load_tbe_kernels() is None:
        return False
    try:
        capability = torch.cuda.get_device_capability(
            torch.device("cuda", torch.cuda.current_device())
        )
    except Exception:
        return False
    return capability_to_arch(capability) == arch


def tbe_mma_decode(dev: TBEDevice, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Decode the container to a dense bf16 ``[N, K]`` with the fragment kernel.

    This is the M > 16 serving path's first half.  ``out``, when given, is
    decoded INTO rather than allocated fresh -- the extension call already
    takes the destination tensor as an argument, so accepting one here (a
    superset of ``experiments.georefine._glc_tbe_mma.tbe_mma_decode``, which
    always allocates) is what lets a caller reuse one per-layer TRANSIENT
    buffer across every M > 16 linear instead of growing a fresh dense copy
    on every call.
    """
    kernels = load_tbe_kernels()
    if kernels is None:
        raise TBEMMAError(
            "TBE MMA kernels are unavailable on this machine (no CUDA, no "
            "vendored extension, or an arch mismatch); refusing rather than "
            "decoding through a slower substitute silently"
        )
    n, k = _validate_device_container(dev)
    if out is None:
        out = torch.empty((n, k), dtype=torch.bfloat16, device=dev.device)
    elif tuple(out.shape) != (n, k) or out.dtype != torch.bfloat16 or out.device != dev.device:
        raise TBEMMAError(
            f"out has shape {tuple(out.shape)} dtype {out.dtype} device "
            f"{out.device}; the container decodes to {(n, k)} bfloat16 on "
            f"{dev.device}"
        )
    stream = int(torch.cuda.current_stream(dev.device).cuda_stream)
    kernels.tbe_mma_decode(
        dev.planes, dev.smb, dev.esc, dev.sbbase, out, n, k,
        int(dev.superblock), int(dev.e01), int(dev.e23), stream,
    )
    return out


def tbe_mma_gemm(
    dev: TBEDevice, x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    split_k: Optional[int] = None,
) -> torch.Tensor:
    """``y[M, N] = x[M, K] @ W[N, K]^T + bias``, ``W`` read from the container.

    M is padded to the 16-row MMA tile; rows past M are masked to zero rather
    than read.  Above 16 this refuses instead of falling back: the M > 16
    path is a different mainloop (decode-then-matmul), and a silent reroute
    would make the receipt describe a kernel that did not run.
    """
    kernels = load_tbe_kernels()
    if kernels is None:
        raise TBEMMAError(
            "TBE MMA kernels are unavailable on this machine (no CUDA, no "
            "vendored extension, or an arch mismatch); refusing rather than "
            "computing through a slower substitute silently"
        )
    flat = x.reshape(-1, x.shape[-1]) if x.dim() > 1 else x.reshape(1, -1)
    m = int(flat.shape[0])
    if m < 1 or m > MAX_M:
        raise TBEMMAError(
            f"this rung serves M = 1..{MAX_M} by one m16n8k16 tile, got M={m}; "
            "M > 16 needs the decode-then-matmul path (see GLCTBELinear)"
        )
    n, k = _validate_device_container(dev)
    if int(flat.shape[1]) != k:
        raise TBEMMAError(f"x has K={int(flat.shape[1])}, container has K={k}")
    if flat.device != dev.device:
        raise TBEMMAError(f"x is on {flat.device}, expected {dev.device}")
    xb = flat.contiguous().to(torch.bfloat16)
    if bias is None:
        b = torch.empty(0, dtype=torch.bfloat16, device=dev.device)
    else:
        if bias.device != dev.device:
            raise TBEMMAError(f"bias is on {bias.device}, expected {dev.device}")
        if int(bias.numel()) != n:
            raise TBEMMAError(f"bias has {int(bias.numel())} entries, expected {n}")
        b = bias.reshape(-1).contiguous().to(torch.bfloat16)
    sk = default_split_k(n, k) if split_k is None else int(split_k)
    tiles_per_row = k // TILE
    if sk < 1 or sk > tiles_per_row:
        raise TBEMMAError(f"split_k must be in [1, {tiles_per_row}], got {sk}")
    ws = torch.zeros((sk, MMA_M, n), dtype=torch.float32, device=dev.device)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=dev.device)
    stream = int(torch.cuda.current_stream(dev.device).cuda_stream)
    kernels.tbe_mma_gemm(
        xb, dev.planes, dev.smb, dev.esc, dev.sbbase, b, ws, out, m, n, k,
        int(dev.superblock), int(dev.e01), int(dev.e23), sk, stream,
    )
    return out


__all__ = [
    "ESC_PAD",
    "LAYOUT",
    "MAX_M",
    "MMA_K",
    "MMA_M",
    "MMA_N",
    "NSLAB",
    "N_PER_CTA",
    "SUPPORTED_CAPABILITIES",
    "WARPS_PER_CTA",
    "TBEDevice",
    "TBEMMAError",
    "capability_to_arch",
    "default_split_k",
    "exponent_table",
    "load_tbe_kernels",
    "resolve_tbe_mma_arch",
    "tbe_mma_arch",
    "tbe_mma_available",
    "tbe_mma_capability",
    "tbe_mma_decode",
    "tbe_mma_gemm",
    "upload_tbe",
]
