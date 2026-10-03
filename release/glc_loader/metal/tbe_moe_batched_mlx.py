"""Phase 7 -- ONE Metal launch that decodes and multiplies MANY TBE containers.

WHY THIS EXISTS
---------------
``docs/research/TDEC_SWEEP_20260904.md`` sec.8 named the term that dominates what is
left of the flagship's decode budget, and it is not the unpack:

    "The launch tax on the grouped expert block: 413-452 us/layer, ~19-22 ms/token,
     larger than the entire per-shape decode cost of all 1,440 expert GEMVs.  A
     fused kernel that decodes and multiplies SEVERAL containers in one launch
     attacks a bigger term than any unpack change."

The phase-5/6 fused GEMV is bound to ONE container, because base / mode / esc /
tile_base are per-tensor uniforms.  A routed MoE layer therefore issues 30 launches
(10 experts x {gate, up, down}) where MLX's dense ``gather_mm`` issues 3 over a
stacked ``[K, ...]`` tensor -- and gather_mm is 1.605x the 30-launch dense loop on
the identical draw.  This module removes exactly that asymmetry: it concatenates E
same-shape containers into one bundle and runs them under one grid.

WHAT IS AND IS NOT NEW HERE
---------------------------
The decode routine is **imported verbatim** from ``tbe_gemv_fused_mlx`` --
``_DECODE_FN`` via :func:`_decode_header`, unedited, byte for byte.  This module
does not re-type flat64 bit semantics anywhere; the only Metal text it authors is
the grid mapping, the per-expert uniform fetch and the epilogue.  Two mechanical
substitutions are applied to the phase-5 MAC templates (``esc`` -> ``esc_e`` for the
expert's escape slice, ``x`` -> ``xp`` for the expert's activation row), each behind
an assertion on the number of call sites, so a refactor of the base file raises
instead of silently measuring a different loop -- the same discipline
``tbe_gemv_fused_lut_mlx`` uses.

BIT-IDENTITY (the reason the arms are comparable at all)
--------------------------------------------------------
For a given (expert e, row r) the batched kernel walks the SAME tiles in the SAME
order with the SAME per-lane decode and the SAME ``simd_sum``, differing only in the
tile INDEX (``e * tiles_per_expert`` is added), which addresses the same bytes.  The
float accumulation order is therefore identical to running
``tbe_gemv_fused(container_e, x, variant=V)`` E times, and the output is
**bit-identical**, not merely within 1 ulp.  :func:`tbe_moe_batched_gemv` and the
per-container loop are two spellings of one arithmetic.

``combine=True`` is the ONE place that is not bit-identical to the loop, and it says
so: the weighted expert sum is accumulated in fp32 across experts in FIXED ASCENDING
EXPERT ORDER, whereas the loop rounds each expert's sum to bf16 before any host-side
combine.  It is exact against the fp32 definition (:func:`combine_reference_fp32`)
and its distance from the bf16-rounded loop is measured, never assumed.

BUNDLE LAYOUT (all E containers must share one shape)
-----------------------------------------------------
  planes     uint32 [E * T * 6]   expert e's planes at [e*T*6, (e+1)*T*6)
  smb        uint8  [E * T * 64]
  esc        uint8  [sum_e E_e]   ragged: expert e's slice at [esc_off[e], +esc_len[e])
  tile_base  uint32 [E * T]       per-expert escape ranks, unchanged
  meta       int32  [E * 4]       (esc_off, esc_len, base, is_w6z) per expert
  dims       uint32 [8]           (n_rows, tiles_per_row, n_experts,
                                   x_expert_stride, tiles_per_expert, n_out, 0, 0)

T = n_rows * tiles_per_row is uniform across experts, which is why planes / smb /
tile_base need no offset table.  A MoE layer's experts are same-shape by
construction; a mixed-shape bundle is REFUSED rather than padded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

try:
    import mlx.core as mx
except Exception:  # pragma: no cover - exercised only on a non-MLX host
    mx = None  # type: ignore

from .tbe_decode_mlx import TBEDeviceMLX
from .tbe_gemv_fused_mlx import (
    MAX_FUSED_BATCH,
    TILE,
    _decode_header,
    _TILE_MAC,
    _VARIANT_GRID,
    _X_MAC,
    DEFAULT_LOAD,
    VARIANTS,
)


class TBEMoEBatchedError(RuntimeError):
    """The batched multi-expert path refuses this bundle or this call.

    Every refusal in this module is one of these.  There is no fallback to the
    per-container loop: a receipt that says "batched" must have measured a batched
    launch, and silently degrading to 30 launches is precisely how a dispatch
    measurement reports the number it was built to remove.
    """


# ---------------------------------------------------------------------------
# the two mechanical substitutions, asserted
# ---------------------------------------------------------------------------
_ESC_CALL = "tbe_decode_pair(planes, smb, esc, tile_base,"
_X_LOAD = "(x + %(M)du * n_cols + col_)"

if _TILE_MAC.count(_ESC_CALL) != 1:
    raise TBEMoEBatchedError(
        "tbe_gemv_fused_mlx._TILE_MAC no longer calls tbe_decode_pair with the "
        f"expected argument list ({_ESC_CALL!r} appears "
        f"{_TILE_MAC.count(_ESC_CALL)} times). The batched kernel rewrites exactly "
        "that one call site to pass the expert's escape slice; refusing rather "
        "than emitting a kernel built against a stale template."
    )
if _X_MAC.count(_X_LOAD) != 1:
    raise TBEMoEBatchedError(
        "tbe_gemv_fused_mlx._X_MAC no longer loads the activation with the "
        f"expected expression ({_X_LOAD!r} appears {_X_MAC.count(_X_LOAD)} times). "
        "The batched kernel rewrites exactly that one site to offset x by the "
        "expert; refusing rather than emitting a kernel built against a stale "
        "template."
    )

#: ``_TILE_MAC`` with the escape pointer swapped for the expert's slice.
_TILE_MAC_E = _TILE_MAC.replace(
    _ESC_CALL, "tbe_decode_pair(planes, smb, esc_e, tile_base,")
#: ``_X_MAC`` with the activation pointer swapped for the expert's row.
_X_MAC_E = _X_MAC.replace(_X_LOAD, "(xp + %(M)du * n_cols + col_)")


def _emit_tile_mac(jj: str, batch: int) -> str:
    xmac = "".join(_X_MAC_E % {"M": m} for m in range(batch))
    return _TILE_MAC_E % {"JJ": jj, "xmac": xmac}


# ---------------------------------------------------------------------------
# the batched kernel source
# ---------------------------------------------------------------------------
#: Read once per SIMD group.  ``e`` is a runtime divide, not a template
#: constant: the flagship's top-k is a router output, so a kernel specialised on
#: E would recompile on every distinct draw.
_PER_EXPERT_PROLOGUE = r"""
    uint n_rows  = dims[0];
    uint tpr     = dims[1];
    uint n_exp   = dims[2];
    uint xstr    = dims[3];
    uint tpe     = dims[4];
    uint n_out   = dims[5];
    uint n_cols  = tpr * 64u;
    uint e   = gr / n_rows;
    uint row = gr - e * n_rows;
    uint esc_off = uint(meta[4u * e + 0u]);
    uint esc_len = uint(meta[4u * e + 1u]);
    int  b       = meta[4u * e + 2u];
    uint is_w6z  = uint(meta[4u * e + 3u]);
    // Address-space-preserving pointer shift: `esc` may be `device` or
    // `constant` depending on how MLX promoted the buffer, and `auto` keeps
    // whichever it is.  tile_base[] already holds the expert's OWN escape
    // ranks, so shifting the base pointer is the whole of the per-expert
    // escape indirection -- the imported decode routine is untouched.
    auto esc_e = esc + esc_off;
    auto xp = x + e * xstr;
    uint tile0 = e * tpe + row * tpr;
"""


def _build_batched_source(mode: str, tiles_per_group: int,
                          simdgroups_per_tg: int, batch: int,
                          combine: bool) -> str:
    tpg = int(tiles_per_group)
    spt = int(simdgroups_per_tg)
    decls = "".join(f"    float acc{m} = 0.0f;\n" for m in range(batch))

    if combine and mode != "row":
        raise TBEMoEBatchedError(
            "combine=True is implemented only for mode 'row': the weighted "
            "expert sum must run in one SIMD group in fixed ascending expert "
            "order, and splitk already owns the threadgroup for its tile split."
        )

    if combine:
        # One SIMD group owns one OUTPUT ROW and loops every expert in fixed
        # ascending order, accumulating w[e] * <fp32 row dot> .  The expert
        # loop is the OUTER loop, so the combine order is a property of the
        # source text, not of the launch.
        prologue = r"""
    uint lane = thread_index_in_simdgroup;
    uint sgid = simdgroup_index_in_threadgroup;
    uint out_row = threadgroup_position_in_grid.x * %(spt)du + sgid;
    uint n_rows  = dims[0];
    if (out_row >= n_rows) { return; }
""" % {"spt": spt}
        body = r"""
    uint tpr    = dims[1];
    uint n_exp  = dims[2];
    uint xstr   = dims[3];
    uint tpe    = dims[4];
    uint n_cols = tpr * 64u;
%(cdecls)s
    for (uint e = 0u; e < n_exp; e++) {
        uint esc_off = uint(meta[4u * e + 0u]);
        uint esc_len = uint(meta[4u * e + 1u]);
        int  b       = meta[4u * e + 2u];
        uint is_w6z  = uint(meta[4u * e + 3u]);
        auto esc_e = esc + esc_off;
        auto xp = x + e * xstr;
        uint tile0 = e * tpe + out_row * tpr;
%(decls)s
        uint j = 0u;
        uint jend = tpr;
        for (; j + %(tpg)du <= jend; j += %(tpg)du) {
%(unrolled)s
        }
        for (; j < jend; j += 1u) {
%(tail)s
        }
        float we = float(w[e]);
%(fold)s
    }
%(store)s
""" % {
            "cdecls": "".join(f"    float tot{m} = 0.0f;\n" for m in range(batch)),
            "decls": "".join(f"        float acc{m} = 0.0f;\n" for m in range(batch)),
            "tpg": tpg,
            "unrolled": "".join(
                _emit_tile_mac(f"j + {u}u", batch) for u in range(tpg)),
            "tail": _emit_tile_mac("j", batch),
            "fold": "".join(
                "        { float s%d = simd_sum(acc%d); tot%d += we * s%d; }\n"
                % (m, m, m, m) for m in range(batch)),
            "store": "".join(
                "    if (lane == 0u) { out[%du * n_rows + out_row] = "
                "bfloat16_t(tot%d); }\n" % (m, m) for m in range(batch)),
        }
        return prologue + body

    if mode == "row":
        prologue = r"""
    uint lane = thread_index_in_simdgroup;
    uint sgid = simdgroup_index_in_threadgroup;
    uint gr = threadgroup_position_in_grid.x * %(spt)du + sgid;
    if (gr >= dims[5]) { return; }
""" % {"spt": spt} + _PER_EXPERT_PROLOGUE
        loop = r"""
    uint j = 0u;
    uint jend = tpr;
    for (; j + %(tpg)du <= jend; j += %(tpg)du) {
%(unrolled)s
    }
    for (; j < jend; j += 1u) {
%(tail)s
    }
""" % {"tpg": tpg,
       "unrolled": "".join(_emit_tile_mac(f"j + {u}u", batch) for u in range(tpg)),
       "tail": _emit_tile_mac("j", batch)}
        epilogue = "".join(
            r"""
    {
        float s%(M)d = simd_sum(acc%(M)d);
        if (lane == 0u) { out[%(M)du * n_out + gr] = bfloat16_t(s%(M)d); }
    }
""" % {"M": m} for m in range(batch)
        )
        return prologue + decls + loop + epilogue

    if mode == "splitk":
        prologue = r"""
    uint lane = thread_index_in_simdgroup;
    uint sgid = simdgroup_index_in_threadgroup;
    uint gr = threadgroup_position_in_grid.x;
    threadgroup float part[%(spt)du * %(batch)du];
""" % {"spt": spt, "batch": batch} + _PER_EXPERT_PROLOGUE
        loop = r"""
    uint j = sgid;
    uint jend = tpr;
    for (; j + %(stride)du <= jend; j += %(stride)du) {
%(unrolled)s
    }
    for (; j < jend; j += %(spt)du) {
%(tail)s
    }
""" % {"stride": tpg * spt, "spt": spt,
       "unrolled": "".join(
           _emit_tile_mac(f"j + {u * spt}u", batch) for u in range(tpg)),
       "tail": _emit_tile_mac("j", batch)}
        store = "".join(
            "    { float p%d = simd_sum(acc%d);"
            " if (lane == 0u) { part[sgid * %du + %du] = p%d; } }\n"
            % (m, m, batch, m, m) for m in range(batch))
        reduce = r"""
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u && lane == 0u) {
""" + "".join(
            r"""
        {
            float s%(M)d = 0.0f;
            for (uint g = 0u; g < %(spt)du; g++) { s%(M)d += part[g * %(batch)du + %(M)du]; }
            out[%(M)du * n_out + gr] = bfloat16_t(s%(M)d);
        }
""" % {"M": m, "spt": spt, "batch": batch} for m in range(batch)
        ) + "    }\n"
        return prologue + decls + loop + store + reduce

    raise TBEMoEBatchedError(f"unknown batched mode {mode!r}")


_KERNEL_CACHE: dict = {}


def _batched_kernel(mode: str, tpg: int, spt: int, batch: int, load: str,
                    combine: bool):
    key = (mode, tpg, spt, batch, load, combine)
    if key not in _KERNEL_CACHE:
        tag = f"{mode}_t{tpg}_g{spt}_m{batch}_{load}" + ("_wsum" if combine else "")
        names = ["planes", "smb", "esc", "tile_base", "x", "meta", "dims"]
        if combine:
            names.append("w")
        _KERNEL_CACHE[key] = mx.fast.metal_kernel(
            name=f"tbe_moe_batched_{tag}",
            input_names=names,
            output_names=["out"],
            source=_build_batched_source(mode, tpg, spt, batch, combine),
            header=_decode_header(load),
        )
    return _KERNEL_CACHE[key]


# ---------------------------------------------------------------------------
# the bundle
# ---------------------------------------------------------------------------
@dataclass
class TBEExpertBundle:
    """E same-shape TBE containers concatenated for one launch.

    Built ONCE at load time, exactly as a dense MoE keeps its experts in one
    ``[E, O, I]`` tensor for ``gather_mm``.  It is not built per token and must
    never be timed inside a decode measurement -- the dense arm's stack is not
    timed either, and timing one side's preparation and not the other's is how a
    dispatch comparison stops being one.
    """

    planes: "mx.array"      # uint32 [E * T * 6]
    smb: "mx.array"         # uint8  [E * T * 64]
    esc: "mx.array"         # uint8  [sum_e max(E_e, 1)]
    tile_base: "mx.array"   # uint32 [E * T]
    meta: "mx.array"        # int32  [E * 4]
    n_experts: int
    n_rows: int
    n_cols: int
    tiles_per_expert: int

    @property
    def numel(self) -> int:
        return self.n_experts * self.n_rows * self.n_cols

    @property
    def resident_bytes(self) -> int:
        def _b(a) -> int:
            return int(a.size) * int(a.itemsize)
        return (_b(self.planes) + _b(self.smb) + _b(self.esc)
                + _b(self.tile_base) + _b(self.meta))

    @property
    def dense_bytes(self) -> int:
        return self.numel * 2

    def ratio(self) -> float:
        return self.dense_bytes / max(1, self.resident_bytes)


def build_expert_bundle(containers: Sequence[TBEDeviceMLX]) -> TBEExpertBundle:
    """Concatenate E same-shape flat64 containers into one launchable bundle.

    Refuses a mixed-shape or non-tileable set rather than padding: a padded
    expert would change the tile count and therefore the tile INDEX arithmetic
    the whole kernel rests on.
    """
    if mx is None:  # pragma: no cover
        raise TBEMoEBatchedError("MLX is not importable on this host")
    cs = list(containers)
    if not cs:
        raise TBEMoEBatchedError("an expert bundle needs at least one container")
    rows, cols = int(cs[0].shape[0]), int(cs[0].shape[1])
    if cols % TILE != 0:
        raise TBEMoEBatchedError(
            f"batched fused decode needs in_features divisible by {TILE} so a "
            f"flat64 tile never straddles a row of W; got {(rows, cols)}. This "
            "path refuses rather than falling back."
        )
    tpe = rows * (cols // TILE)
    for i, c in enumerate(cs):
        if (int(c.shape[0]), int(c.shape[1])) != (rows, cols):
            raise TBEMoEBatchedError(
                f"expert {i} has shape {tuple(c.shape)}, expert 0 has "
                f"{(rows, cols)}; a bundle is same-shape by construction "
                "(planes/smb/tile_base carry no per-expert offset table)"
            )
        if int(c.tiles) != tpe:
            raise TBEMoEBatchedError(
                f"expert {i} has {c.tiles} tiles, {rows}x{cols} needs exactly "
                f"{tpe}; the batched kernel indexes tiles by (expert, row) and "
                "cannot use a padded final tile"
            )

    planes = mx.concatenate([c.planes for c in cs], axis=0)
    smb = mx.concatenate([c.smb for c in cs], axis=0)
    tile_base = mx.concatenate([c.tile_base for c in cs], axis=0)
    esc = mx.concatenate([c.esc for c in cs], axis=0)

    meta = np.zeros((len(cs), 4), dtype=np.int32)
    off = 0
    for i, c in enumerate(cs):
        n = int(c.esc.size)
        meta[i, 0] = off
        meta[i, 1] = n
        meta[i, 2] = int(c.base)
        meta[i, 3] = 1 if int(c.mode) != 0 else 0
        off += n
    return TBEExpertBundle(
        planes=planes, smb=smb, esc=esc, tile_base=tile_base,
        meta=mx.array(meta.reshape(-1)),
        n_experts=len(cs), n_rows=rows, n_cols=cols, tiles_per_expert=tpe,
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def tbe_moe_batched_gemv(
    bundle: TBEExpertBundle,
    x: "mx.array",
    *,
    variant: str = "row_t1_g4",
    load: str = DEFAULT_LOAD,
    per_expert_x: bool = False,
    weights: "mx.array | None" = None,
    eval_now: bool = False,
) -> "mx.array":
    """One launch: ``y[e] = x_e @ W_e.T`` for every expert in ``bundle``.

    ``x`` is ``[M, I]`` when ``per_expert_x`` is False (gate / up: every expert
    sees the same token) and ``[E, M, I]`` when True (down: each expert consumes
    its own gated hidden).  Returns ``[M, E * O]`` -- expert-major, so
    ``out[:, e*O:(e+1)*O]`` is exactly what ``tbe_gemv_fused(containers[e], ...)``
    returns, bit for bit.

    With ``weights`` given (``[E]``), the kernel instead returns ``[M, O]``: the
    router-weighted expert sum, accumulated in fp32 in ascending expert order.
    That is NOT bit-identical to the per-container loop -- see the module
    docstring and :func:`combine_reference_fp32`.
    """
    if variant not in _VARIANT_GRID:
        raise TBEMoEBatchedError(
            f"unknown variant {variant!r}; expected one of {VARIANTS}")
    mode, tpg, spt = _VARIANT_GRID[variant]
    combine = weights is not None
    E, O, I = bundle.n_experts, bundle.n_rows, bundle.n_cols

    if x.ndim == 1:
        x2 = x.reshape(1, -1)
        squeeze = True
    else:
        x2 = x.reshape(-1, x.shape[-1])
        squeeze = False
    if int(x2.shape[-1]) != I:
        raise TBEMoEBatchedError(
            f"activation has {int(x2.shape[-1])} features, the bundle's experts "
            f"expect {I}")
    n_x = int(x2.shape[0])
    if per_expert_x:
        if n_x % E != 0:
            raise TBEMoEBatchedError(
                f"per_expert_x needs the activation to carry E={E} blocks; got "
                f"{n_x} rows, which is not a multiple of E")
        batch = n_x // E
        x_expert_stride = batch * I
    else:
        batch = n_x
        x_expert_stride = 0
    if batch < 1 or batch > MAX_FUSED_BATCH:
        raise TBEMoEBatchedError(
            f"the batched fused GEMV is built for M in [1, {MAX_FUSED_BATCH}]; "
            f"got M={batch}. Larger M is a tiled GEMM problem, not this kernel.")

    if combine:
        w = weights.reshape(-1)
        if int(w.size) != E:
            raise TBEMoEBatchedError(
                f"router weights have {int(w.size)} entries, the bundle holds "
                f"{E} experts")
        if mode != "row":
            raise TBEMoEBatchedError(
                "the weighted-combine epilogue needs a 'row' variant so one "
                f"SIMD group owns an output row; got {variant!r}")
        w = w.astype(mx.float32)
        n_out = O
    else:
        w = None
        n_out = E * O

    xb = x2.astype(mx.bfloat16)
    dims = mx.array(
        np.array([O, I // TILE, E, x_expert_stride, bundle.tiles_per_expert,
                  E * O, 0, 0], dtype=np.uint32))

    if combine or mode == "row":
        n_grid_rows = O if combine else E * O
        n_tg = max(1, -(-n_grid_rows // spt))
    else:
        n_tg = E * O
    tg_width = spt * 32

    inputs = [bundle.planes, bundle.smb, bundle.esc, bundle.tile_base,
              xb, bundle.meta, dims]
    if combine:
        inputs.append(w)

    kernel = _batched_kernel(mode, tpg, spt, batch, load, combine)
    out = kernel(
        inputs=inputs,
        template=[],
        grid=(n_tg * tg_width, 1, 1),
        threadgroup=(tg_width, 1, 1),
        output_shapes=[(batch * n_out,)],
        output_dtypes=[mx.bfloat16],
    )[0].reshape(batch, n_out)
    if squeeze and batch == 1:
        out = out.reshape(n_out)
    if eval_now:
        mx.eval(out)
    return out


def combine_reference_fp32(per_expert: "mx.array", weights: "mx.array",
                           n_experts: int) -> "mx.array":
    """The definition ``combine=True`` is exact against.

    ``per_expert`` is ``[M, E * O]`` in fp32 (the UNROUNDED per-expert dots);
    the reference folds them in ascending expert order in fp32 and rounds once.
    A bf16-rounded per-expert loop is a DIFFERENT definition and the distance
    between them is measured, not assumed.
    """
    m = int(per_expert.shape[0])
    o = int(per_expert.shape[-1]) // n_experts
    acc = mx.zeros((m, o), dtype=mx.float32)
    w = weights.reshape(-1).astype(mx.float32)
    for e in range(n_experts):
        acc = acc + w[e] * per_expert[:, e * o:(e + 1) * o].astype(mx.float32)
    return acc.astype(mx.bfloat16)


def batched_launches_per_moe_layer(combine: bool = False) -> int:
    """Launches this path issues for one routed MoE layer.

    gate+up in ONE launch over 2E containers, one elementwise gate*up, one
    down launch over E containers.  The dependency chain gate -> multiply ->
    down is why this is 3 and not 1, and why the 3-launch tax is irreducible.
    """
    return 3


__all__ = [
    "TBEMoEBatchedError",
    "TBEExpertBundle",
    "build_expert_bundle",
    "tbe_moe_batched_gemv",
    "combine_reference_fp32",
    "batched_launches_per_moe_layer",
]
