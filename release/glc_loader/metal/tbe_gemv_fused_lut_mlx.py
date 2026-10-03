"""GLC-TBE flat64 fused decode-in-GEMV with **table-lookup unpack** -- phase 6.

WHY THIS FILE EXISTS
--------------------
Phase 5 measured the fused decode-in-GEMV at 0.57-0.72x of MLX dense bf16 GEMV
at M=1 and located the deficit precisely: not bytes, but **decode instructions
on the critical path** (``.icc/evidence/tbe-metal-decode-20260903/phase5/
summary.json``, ``ablation_where_the_time_goes``).

``docs/research/LITERATURE_BOUNDED_PROBLEM_STATEMENTS_20260904.md`` sec.1.3 names
the one lever the published art says exists for exactly this deficit, and names
it on exactly this chip:

    T-MAC (arXiv:2407.00088, EuroSys 2025), **measured on an M2 Ultra**: in
    dequantization-based kernels "scaling down bits from 4 bit to 1 bit even
    increases latency cost for most of the cases."  Deleting the arithmetic
    unpack in favour of a table lookup inverts it.

    LUT-GEMM (arXiv:2206.09557, ICLR 2024): OPTQ INT3 reaches 38% of ideal and
    LUT-GEMM INT3 60%; "the 38->60 gap *is* dequant cost, measured."

This module is that arm, and **only** that arm.  It is a strict ablation of
:mod:`release.glc_loader.metal.tbe_gemv_fused_mlx`:

* the GEMV body -- accumulator declarations, the TPG unroll, the split-k stride,
  the ``simd_sum`` epilogue, the threadgroup reduction, the ``x`` MAC -- is not
  re-typed here.  It is **imported** from that module's ``_build_gemv_source``
  and the ONLY edit applied is inserting one extra pointer argument into the
  ``tbe_decode_pair`` call.  The insertion is asserted to hit exactly the
  expected number of call sites, so a future refactor of the base file breaks
  loudly rather than silently measuring a different loop.
* the tile-header load block (``half`` / ``redundant`` / ``broadcast``) is
  imported verbatim from the same module, so the arms differ in the unpack and
  in nothing else.

WHAT IS ACTUALLY REPLACED
-------------------------
Arithmetic unpack, per element, in the phase-5 kernel::

    inwin = b + TBE_DELTA[code]            // constant-array read + int add
    inwin = (is_w6z && code == 7) ? 0 : inwin   // W6Z select
    mexp  = esc ? esc[rank] : inwin             // escape (DATA -- not removable)
    word  = (sign << 8) | ((mexp & 0xFF) << 7) | mantissa   // mask/shift/or

``base`` and ``mode`` are **container-uniform** -- one scalar each for the whole
tensor -- so the map ``code -> exponent field`` is a fixed 8-entry function known
on the host before the kernel launches.  Two tables follow, both built on the
host and passed as an ordinary kernel input (MLX promotes a small input to the
``constant`` address space, which is the broadcast-cached path):

``exp8``   8 x uint16.  ``lut[code]`` is the pre-shifted exponent field
           ``((base + delta(code)) & 0xFF) << 7``, with the W6Z zero already
           folded in.  Deletes the constant-array read, the add, the select and
           the shift; the sign/mantissa or-tree stays.
``word2k`` 2048 x uint16, indexed ``(code << 8) | smb``.  One load returns the
           **entire** non-escape bf16 word.  Deletes the whole reconstruction.
``word2k_tg`` the same 2048-entry table, staged into threadgroup memory once per
           threadgroup and read from there.  Included because the literature's
           phrasing is "register-resident / threadgroup-resident"; on this
           kernel's split-k geometry (one threadgroup per output row) the
           staging cost is charged against very little work, and the
           measurement is reported either way rather than dropped.
``exp8_const`` / ``word2k_const``
           the same two tables **baked into the Metal source as a compile-time
           ``constant`` array**, specialised on ``(base, mode)`` and cached per
           specialisation.  This is the strongest available form of the lever and
           the one the phrase "register-resident" actually describes on this
           compiler: no buffer binding, no runtime load, the table is immediate
           data the compiler may fold into registers or the constant bank.
           It exists because the runtime-buffer forms measured NEGATIVE, and a
           negative result on a weak implementation is not a result.

           It also exposes the uncomfortable fact the phase-5 kernel already had:
           ``constant int TBE_DELTA[8]`` is itself a compile-time lookup table.
           The arithmetic arm is not table-free; it is table-plus-one-add.

The escape byte read is **data**, not arithmetic: the escaped exponents are the
container's payload and no table can supply them.  It stays predicated, exactly
as in phase 5.  At a ~2.5-3.1% escape rate it is untouched by every arm here.

BIT-EXACTNESS
-------------
Unchanged in definition from phase 5 and re-checked per arm:

(a) :func:`fused_decode_probe_lut` runs THIS module's ``tbe_decode_pair`` over
    every tile and writes the result, for element-for-element comparison with
    ``tbe_container.decode_tbe``;
(b) GEMV output is compared to ``mx.matmul`` with
    :func:`~release.glc_loader.metal.tbe_gemv_fused_mlx.ulp_report`.

A LUT arm that fails either is refused a number, not reported with a caveat.

NEW FILE.  ``tbe_gemv_fused_mlx.py`` is not edited; nor is any bench, container,
decoder or receipt.
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx

from .tbe_decode_mlx import (  # noqa: F401  (re-exported for callers)
    METAL_MEMORY_LIMIT_BYTES,
    TBEDeviceMLX,
    mlx_metal_available,
    upload_tbe_mlx,
)
from .tbe_gemv_fused_mlx import (
    DEFAULT_LOAD,
    DEFAULT_VARIANT,
    LOAD_MODES,
    MAX_FUSED_BATCH,
    VARIANTS,
    TBEFusedGemvError,
    _LOAD_BLOCKS,
    _PROBE_SOURCE,
    _VARIANT_GRID,
    _build_gemv_source,
    fused_tileable,
    ulp_report,
)
from ..tbe_container import MODE_W6Z, TILE

#: ``TBE_DELTA`` from the phase-5 kernel, on the host.  code -> exponent offset.
TBE_DELTA = (0, 0, 1, 2, 3, 4, 5, 6)

#: LUT constructions.  ``none`` is not offered: this module is the LUT arm and
#: the arithmetic arm is the other module, unmodified, so the control cannot
#: drift.
LUT_MODES = ("exp8", "word2k", "word2k_tg", "exp8_const", "word2k_const")

#: Modes whose table is baked into the Metal source; the kernel cache key must
#: then carry ``(base, mode)`` because two containers with different exponent
#: bases are two different kernels.
CONST_LUT_MODES = ("exp8_const", "word2k_const")

#: The call site the base module emits, once per unrolled slot plus once in the
#: loop tail.  Asserted, never assumed.
_CALL_NEEDLE = "tbe_decode_pair(planes, smb, esc, tile_base,"
_CALL_PATCH = "tbe_decode_pair(planes, smb, esc, tile_base, lut,"


# ---------------------------------------------------------------------------
# host-side table construction -- the arithmetic, done once, outside the kernel
# ---------------------------------------------------------------------------
def exponent_field_table(base: int, mode: int) -> "list[int]":
    """``((exponent) & 0xFF) << 7`` for each 3-bit code, non-escape semantics.

    Mirrors the phase-5 kernel line for line::

        inwin = b + TBE_DELTA[code]
        inwin = (is_w6z && code == 7) ? 0 : inwin
        field = (inwin & 0xFF) << 7

    Code 0 is the escape code: its entry is never consumed (the kernel
    overwrites it from ``esc[]``) but is filled with the same expression rather
    than a sentinel, so a table dump is readable and a mis-predicated kernel
    would produce the *phase-5* answer, not garbage.
    """
    is_w6z = int(mode) == MODE_W6Z
    out = []
    for code in range(8):
        exp = int(base) + TBE_DELTA[code]
        if is_w6z and code == 7:
            exp = 0
        out.append(((exp & 0xFF) << 7) & 0xFFFF)
    return out


def word_table(base: int, mode: int) -> "list[int]":
    """2048 x uint16: the complete non-escape bf16 word for ``(code, smb)``.

    ``word = (sign << 15) | (exponent << 7) | mantissa`` with ``sign`` and
    ``mantissa`` read out of the container's ``smb`` byte, exactly as the
    phase-5 kernel assembles them.
    """
    fields = exponent_field_table(base, mode)
    out = [0] * (8 * 256)
    for code in range(8):
        f = fields[code]
        for smb in range(256):
            out[(code << 8) | smb] = (((smb & 0x80) << 8) | f | (smb & 0x7F)) & 0xFFFF
    return out


def build_lut(c: TBEDeviceMLX, lut_mode: str) -> mx.array:
    """The uint16 table this container needs for ``lut_mode``."""
    if lut_mode not in LUT_MODES:
        raise TBEFusedGemvError(
            f"unknown lut mode {lut_mode!r}; expected one of {LUT_MODES}"
        )
    if lut_mode in CONST_LUT_MODES:
        # The table is compile-time data inside the kernel; this 8-entry array
        # is bound only so the kernel signature and the GEMV call sites stay
        # byte-identical to the runtime-table arms.  It is never read
        # (`(void)lut;`), which the unpack block states explicitly.
        vals = exponent_field_table(int(c.base), int(c.mode))
    elif lut_mode == "exp8":
        vals = exponent_field_table(int(c.base), int(c.mode))
    else:
        vals = word_table(int(c.base), int(c.mode))
    return mx.array(vals, dtype=mx.uint16)


# ---------------------------------------------------------------------------
# the decode routine -- the ONLY thing that differs from phase 5
# ---------------------------------------------------------------------------
_DECODE_PROLOGUE = r"""
// LUT unpack.  `b` and `is_w6z` are accepted so the signature and the call
// sites stay identical to the phase-5 arithmetic kernel; they are unused here
// BY CONSTRUCTION -- that is the whole point of the arm, the exponent map they
// parameterise having been evaluated on the host into `lut`.
template <typename PLANES_T, typename SMB_T, typename ESC_T, typename TB_T,
          typename LUT_T>
inline void tbe_decode_pair(
        PLANES_T planes, SMB_T smb, ESC_T esc, TB_T tile_base, LUT_T lut,
        uint tile, uint lane, int b, uint is_w6z, uint esc_len,
        thread ushort& word_a, thread ushort& word_b) {
    (void)b; (void)is_w6z;
    uint off = tile * 6u;
    bool hi = lane >= 16u;
%(load_block)s
    uint bit_a = 2u * lane;
    uint bit_b = bit_a + 1u;
    uint shA = hi ? (bit_a - 32u) : bit_a;
    uint shB = hi ? (bit_b - 32u) : bit_b;

    uint code_a = ((w0 >> shA) & 1u) | (((w1 >> shA) & 1u) << 1) | (((w2 >> shA) & 1u) << 2);
    uint code_b = ((w0 >> shB) & 1u) | (((w1 >> shB) & 1u) << 1) | (((w2 >> shB) & 1u) << 2);

    bool esc_a = (code_a == 0u);
    bool esc_b = (code_b == 0u);
    uint lane_esc = (esc_a ? 1u : 0u) + (esc_b ? 1u : 0u);
    uint prefix = simd_prefix_exclusive_sum(lane_esc);
    uint rank_before = tile_base[tile] + prefix;
    uint rank_a = rank_before;
    uint rank_b = rank_before + (esc_a ? 1u : 0u);

    uint elem_a = tile * 64u + bit_a;
    uint smb_a = uint(smb[elem_a]);
    uint smb_b = uint(smb[elem_a + 1u]);
%(unpack_block)s
    // ESCAPE: data, not arithmetic.  No table can supply an escaped exponent,
    // so this stays exactly as phase 5 left it -- predicated, clamped on the
    // taken branch only.
    if (esc_a) {
        uint e_ = uint(esc[rank_a < esc_len ? rank_a : (esc_len - 1u)]);
        word_a = (ushort(smb_a & 0x80u) << 8) | (ushort(e_ & 0xFFu) << 7) | ushort(smb_a & 0x7Fu);
    }
    if (esc_b) {
        uint e_ = uint(esc[rank_b < esc_len ? rank_b : (esc_len - 1u)]);
        word_b = (ushort(smb_b & 0x80u) << 8) | (ushort(e_ & 0xFFu) << 7) | ushort(smb_b & 0x7Fu);
    }
}
"""

#: exp8: one table read replaces the delta read, the add, the W6Z select and
#: the exponent shift.  The sign/mantissa or-tree survives.
_UNPACK_EXP8 = r"""
    ushort fa_ = lut[code_a];
    ushort fb_ = lut[code_b];
    word_a = (ushort(smb_a & 0x80u) << 8) | fa_ | ushort(smb_a & 0x7Fu);
    word_b = (ushort(smb_b & 0x80u) << 8) | fb_ | ushort(smb_b & 0x7Fu);
"""

#: word2k: one table read replaces the ENTIRE non-escape reconstruction.
_UNPACK_WORD2K = r"""
    word_a = lut[(code_a << 8) | smb_a];
    word_b = lut[(code_b << 8) | smb_b];
"""

#: exp8_const / word2k_const: identical arithmetic to the two above, but the
#: table is ``TBE_LUT``, a compile-time ``constant`` array emitted into the
#: header, not a bound buffer.  ``lut`` is still accepted and still bound so the
#: kernel signature and the GEMV call sites are byte-identical across every arm.
_UNPACK_EXP8_CONST = r"""
    (void)lut;
    ushort fa_ = TBE_LUT[code_a];
    ushort fb_ = TBE_LUT[code_b];
    word_a = (ushort(smb_a & 0x80u) << 8) | fa_ | ushort(smb_a & 0x7Fu);
    word_b = (ushort(smb_b & 0x80u) << 8) | fb_ | ushort(smb_b & 0x7Fu);
"""

_UNPACK_WORD2K_CONST = r"""
    (void)lut;
    word_a = TBE_LUT[(code_a << 8) | smb_a];
    word_b = TBE_LUT[(code_b << 8) | smb_b];
"""

_UNPACK_BLOCKS = {
    "exp8": _UNPACK_EXP8,
    "word2k": _UNPACK_WORD2K,
    "word2k_tg": _UNPACK_WORD2K,
    "exp8_const": _UNPACK_EXP8_CONST,
    "word2k_const": _UNPACK_WORD2K_CONST,
}

#: Threadgroup staging prologue for ``word2k_tg``.  Prepended to the imported
#: GEMV body; the call-site patch then points ``tbe_decode_pair`` at ``tglut``
#: instead of the device/constant ``lut``.
_TG_STAGE = r"""
    threadgroup ushort tglut[2048];
    {
        uint tid_ = thread_position_in_threadgroup.x;
        for (uint q_ = tid_; q_ < 2048u; q_ += %(tgw)du) { tglut[q_] = lut[q_]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""


def _const_table_source(lut_mode: str, base: int, mode: int) -> str:
    """``constant ushort TBE_LUT[N] = {...};`` for a baked-in table."""
    vals = (exponent_field_table(base, mode) if lut_mode == "exp8_const"
            else word_table(base, mode))
    body = ",".join(str(int(v)) for v in vals)
    return f"constant ushort TBE_LUT[{len(vals)}] = {{{body}}};\n"


def _decode_header(load: str, lut_mode: str, base: int = 0, mode: int = 0) -> str:
    try:
        block = _LOAD_BLOCKS[load]
    except KeyError:
        raise TBEFusedGemvError(
            f"unknown tile-header load mode {load!r}; expected one of {LOAD_MODES}"
        ) from None
    try:
        unpack = _UNPACK_BLOCKS[lut_mode]
    except KeyError:
        raise TBEFusedGemvError(
            f"unknown lut mode {lut_mode!r}; expected one of {LUT_MODES}"
        ) from None
    prefix = (_const_table_source(lut_mode, base, mode)
              if lut_mode in CONST_LUT_MODES else "")
    return prefix + _DECODE_PROLOGUE % {"load_block": block, "unpack_block": unpack}


def _patch_calls(src: str, expected: int, target: str = "lut") -> str:
    """Insert the LUT pointer into every ``tbe_decode_pair`` call in ``src``.

    Asserted, not assumed: if the base module's emitter ever changes the call
    text or the number of call sites, this raises instead of quietly compiling
    a kernel whose GEMV body is no longer the phase-5 body.
    """
    got = src.count(_CALL_NEEDLE)
    if got != expected:
        raise TBEFusedGemvError(
            "the imported phase-5 GEMV body no longer has the expected "
            f"tbe_decode_pair call shape: found {got} occurrences of "
            f"{_CALL_NEEDLE!r}, expected {expected}. This module is a strict "
            "ablation of that body and refuses to compile a drifted copy."
        )
    patch = _CALL_PATCH if target == "lut" else _CALL_PATCH.replace(" lut,", f" {target},")
    return src.replace(_CALL_NEEDLE, patch)


def _build_lut_gemv_source(mode: str, tpg: int, spt: int, batch: int,
                           lut_mode: str) -> str:
    """The phase-5 GEMV body, verbatim, with the LUT pointer threaded in."""
    src = _build_gemv_source(mode, tpg, spt, batch)
    # one call per unrolled slot + one in the scalar tail
    if lut_mode == "word2k_tg":
        src = _TG_STAGE % {"tgw": int(spt) * 32} + src
        return _patch_calls(src, tpg + 1, target="tglut")
    return _patch_calls(src, tpg + 1)


_GEMV_CACHE: dict = {}
_PROBE_CACHE: dict = {}


def _gemv_kernel(mode: str, tpg: int, spt: int, batch: int, load: str,
                 lut_mode: str, base: int = 0, cmode: int = 0):
    # A baked-in table is part of the program text, so two containers with
    # different exponent bases are two different kernels and must not share a
    # cache slot. Runtime-table arms are base-independent and do not.
    spec = (int(base), int(cmode)) if lut_mode in CONST_LUT_MODES else None
    key = (mode, tpg, spt, batch, load, lut_mode, spec)
    if key not in _GEMV_CACHE:
        tag = f"{mode}_t{tpg}_g{spt}_m{batch}_{load}_{lut_mode}"
        if spec is not None:
            tag += f"_b{int(base)}_w{int(cmode)}"
        _GEMV_CACHE[key] = mx.fast.metal_kernel(
            name=f"tbe_gemv_fused_lut_{tag}",
            input_names=[
                "planes", "smb", "esc", "tile_base", "lut", "x",
                "base", "w6z", "n_rows_a", "tiles_per_row", "esc_len_arr",
            ],
            output_names=["out"],
            source=_build_lut_gemv_source(mode, tpg, spt, batch, lut_mode),
            header=_decode_header(load, lut_mode, base, cmode),
        )
    return _GEMV_CACHE[key]


def _probe_kernel(spt: int, load: str, lut_mode: str, base: int = 0,
                  cmode: int = 0):
    spec = (int(base), int(cmode)) if lut_mode in CONST_LUT_MODES else None
    key = (spt, load, lut_mode, spec)
    if key not in _PROBE_CACHE:
        tag = f"g{spt}_{load}_{lut_mode}"
        if spec is not None:
            tag += f"_b{int(base)}_w{int(cmode)}"
        src = _PROBE_SOURCE % {"spt": int(spt)}
        if lut_mode == "word2k_tg":
            src = _TG_STAGE % {"tgw": int(spt) * 32} + src
            src = _patch_calls(src, 1, target="tglut")
        else:
            src = _patch_calls(src, 1)
        _PROBE_CACHE[key] = mx.fast.metal_kernel(
            name=f"tbe_fused_lut_decode_probe_{tag}",
            input_names=[
                "planes", "smb", "esc", "tile_base", "lut",
                "base", "w6z", "n_elem", "n_tiles", "esc_len_arr",
            ],
            output_names=["out"],
            source=src,
            header=_decode_header(load, lut_mode, base, cmode),
        )
    return _PROBE_CACHE[key]


_UNIFORMS_ATTR = "_glc_tbe_fused_lut_uniforms"


def _uniforms(c: TBEDeviceMLX):
    got = getattr(c, _UNIFORMS_ATTR, None)
    if got is None:
        rows, cols = int(c.shape[0]), int(c.shape[1])
        got = (
            mx.array([int(c.base)], dtype=mx.int32),
            mx.array([1 if int(c.mode) == MODE_W6Z else 0], dtype=mx.uint32),
            mx.array([rows], dtype=mx.uint32),
            mx.array([cols // TILE], dtype=mx.uint32),
            mx.array([int(c.esc.size)], dtype=mx.uint32),
        )
        try:
            setattr(c, _UNIFORMS_ATTR, got)
        except Exception:
            pass
    return got


_LUT_ATTR = "_glc_tbe_fused_lut_table"


def cached_lut(c: TBEDeviceMLX, lut_mode: str) -> mx.array:
    """The container's table, built once and kept on the container.

    Built at upload time in serving; here it is memoised so the timed loop
    measures the GEMV, not a table rebuild.  The table is 16 B (``exp8``) or
    4 KiB (``word2k``) -- it is not a residency term.
    """
    store = getattr(c, _LUT_ATTR, None)
    if store is None:
        store = {}
        try:
            setattr(c, _LUT_ATTR, store)
        except Exception:
            return build_lut(c, lut_mode)
    if lut_mode not in store:
        t = build_lut(c, lut_mode)
        mx.eval(t)
        store[lut_mode] = t
    return store[lut_mode]


def lut_bytes(lut_mode: str) -> int:
    """Resident data cost of the table, per container.

    Zero for the ``*_const`` arms: the table is immediate data in the compiled
    kernel, not a resident buffer.  (An 8-entry dummy is still bound so the
    kernel signature matches the runtime-table arms; it is never read.)
    """
    if lut_mode in CONST_LUT_MODES:
        return 0
    return 16 if lut_mode == "exp8" else 4096


# ---------------------------------------------------------------------------
# entry points -- signature-compatible with the phase-5 module
# ---------------------------------------------------------------------------
def tbe_gemv_fused_lut(
    c: TBEDeviceMLX,
    x: mx.array,
    variant: str = DEFAULT_VARIANT,
    load: str = DEFAULT_LOAD,
    lut_mode: str = "word2k",
    eval_now: bool = False,
) -> mx.array:
    """``y = x @ W.T`` from the TBE container, unpacking through a table.

    Same contract as
    :func:`~release.glc_loader.metal.tbe_gemv_fused_mlx.tbe_gemv_fused`,
    including the refusal to fall back on an untileable container: a receipt
    must never say "fused-LUT" and measure something else.
    """
    if variant not in _VARIANT_GRID:
        raise TBEFusedGemvError(
            f"unknown variant {variant!r}; expected one of {VARIANTS}"
        )
    if lut_mode not in LUT_MODES:
        raise TBEFusedGemvError(
            f"unknown lut mode {lut_mode!r}; expected one of {LUT_MODES}"
        )
    mode, tpg, spt = _VARIANT_GRID[variant]

    rows, cols = int(c.shape[0]), int(c.shape[1])
    if cols % TILE != 0:
        raise TBEFusedGemvError(
            f"fused decode-in-GEMV needs in_features divisible by {TILE}; got "
            f"shape {c.shape}. This path refuses rather than falling back."
        )
    if int(c.tiles) * TILE != rows * cols:
        raise TBEFusedGemvError(
            f"container tile count {c.tiles} does not cover {rows}x{cols} exactly"
        )

    if x.ndim == 1:
        x2 = x.reshape(1, -1)
        squeeze = True
    else:
        x2 = x.reshape(-1, x.shape[-1])
        squeeze = False
    batch = int(x2.shape[0])
    if int(x2.shape[-1]) != cols:
        raise TBEFusedGemvError(
            f"activation has {int(x2.shape[-1])} features, weight expects {cols}"
        )
    if batch < 1 or batch > MAX_FUSED_BATCH:
        raise TBEFusedGemvError(
            f"fused GEMV is built for M in [1, {MAX_FUSED_BATCH}]; got M={batch}"
        )
    xb = x2.astype(mx.bfloat16)

    if mode == "row":
        n_tg = max(1, -(-rows // spt))
    else:
        n_tg = rows
    tg_width = spt * 32

    kernel = _gemv_kernel(mode, tpg, spt, batch, load, lut_mode,
                          int(c.base), int(c.mode))
    out = kernel(
        inputs=[c.planes, c.smb, c.esc, c.tile_base, cached_lut(c, lut_mode),
                xb, *_uniforms(c)],
        template=[],
        grid=(n_tg * tg_width, 1, 1),
        threadgroup=(tg_width, 1, 1),
        output_shapes=[(batch * rows,)],
        output_dtypes=[mx.bfloat16],
    )[0].reshape(batch, rows)
    if squeeze:
        out = out.reshape(rows)
    if eval_now:
        mx.eval(out)
    return out


def fused_decode_probe_lut(
    c: TBEDeviceMLX,
    simdgroups_per_tg: int = 4,
    load: str = DEFAULT_LOAD,
    lut_mode: str = "word2k",
    eval_now: bool = True,
) -> mx.array:
    """Decode ``c`` through **this module's** ``tbe_decode_pair``.

    Same operational definition of "bit-exact weights" as phase 5: the GEMV and
    this probe are compiled from the same decode text, so an element-for-element
    match against ``tbe_container.decode_tbe`` pins the LUT kernel's view of
    every weight without the GEMV ever writing one.
    """
    n = c.numel
    if n == 0:
        return mx.zeros(c.shape, dtype=mx.bfloat16)
    n_tiles = max(1, int(c.tiles))
    spt = int(simdgroups_per_tg)
    n_tg = max(1, -(-n_tiles // spt))
    tg_width = spt * 32

    kernel = _probe_kernel(spt, load, lut_mode, int(c.base), int(c.mode))
    out = kernel(
        inputs=[
            c.planes, c.smb, c.esc, c.tile_base, cached_lut(c, lut_mode),
            mx.array([int(c.base)], dtype=mx.int32),
            mx.array([1 if int(c.mode) == MODE_W6Z else 0], dtype=mx.uint32),
            mx.array([n], dtype=mx.uint32),
            mx.array([int(c.tiles)], dtype=mx.uint32),
            mx.array([int(c.esc.size)], dtype=mx.uint32),
        ],
        template=[],
        grid=(n_tg * tg_width, 1, 1),
        threadgroup=(tg_width, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.bfloat16],
    )[0].reshape(c.shape)
    if eval_now:
        mx.eval(out)
    return out


class TBELinearFusedLUTMLX:
    """Linear served by the LUT-unpack fused GEMV at small M.

    Deliberately NOT registered in ``tbe_mlx_model.DECODERS``: the phase-5 gate
    (>= 0.9x of same-session dense at M=1) governs both arms, and this one is
    wired in only if it meets it.
    """

    def __init__(
        self,
        container: TBEDeviceMLX,
        bias: Optional[mx.array] = None,
        variant: str = DEFAULT_VARIANT,
        load: str = DEFAULT_LOAD,
        lut_mode: str = "word2k",
    ) -> None:
        if not fused_tileable(container):
            raise TBEFusedGemvError(
                f"container {container.shape} has in_features not divisible by "
                f"{TILE}; the fused LUT GEMV refuses it"
            )
        self.container = container
        self.bias = bias
        self.variant = variant
        self.load = load
        self.lut_mode = lut_mode
        self.out_features, self.in_features = container.shape
        self.n_fused_calls = 0
        self.n_fallback_calls = 0

    @property
    def resident_bytes(self) -> int:
        return self.container.resident_bytes + lut_bytes(self.lut_mode)

    @property
    def dense_bytes(self) -> int:
        return self.container.dense_bytes

    def decode(self, eval_now: bool = False) -> mx.array:
        from .tbe_decode_mlx_v4 import decode_tbe_metal_v4

        return decode_tbe_metal_v4(self.container, eval_now=eval_now)

    def __call__(self, x: mx.array) -> mx.array:
        flat = x.reshape(-1, x.shape[-1])
        if int(flat.shape[-1]) != self.in_features:
            raise TBEFusedGemvError(
                f"input has {int(flat.shape[-1])} features, weight expects "
                f"{self.in_features}"
            )
        m = int(flat.shape[0])
        if m <= MAX_FUSED_BATCH:
            self.n_fused_calls += 1
            y = tbe_gemv_fused_lut(self.container, flat, variant=self.variant,
                                   load=self.load, lut_mode=self.lut_mode)
        else:
            self.n_fallback_calls += 1
            y = mx.matmul(flat.astype(mx.bfloat16), self.decode().T)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*x.shape[:-1], self.out_features)


__all__ = [
    "CONST_LUT_MODES",
    "LUT_MODES",
    "TBE_DELTA",
    "TBEFusedGemvError",
    "TBELinearFusedLUTMLX",
    "build_lut",
    "cached_lut",
    "exponent_field_table",
    "fused_decode_probe_lut",
    "fused_tileable",
    "lut_bytes",
    "mlx_metal_available",
    "tbe_gemv_fused_lut",
    "ulp_report",
    "upload_tbe_mlx",
    "word_table",
]
