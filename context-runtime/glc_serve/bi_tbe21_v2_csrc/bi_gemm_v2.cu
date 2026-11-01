// BI-GEMM v2 instantiations (docs/serving/KERNEL_V2_SPEC_20261017.md): phase-split decode +
// double-buffered pipeline.  F_BF16, F_TBE, F_TBE1T.  v1 (bi_gemm_dense.cu) is a separate TU and
// is untouched; this file adds the v2 entry points bi_torch.cpp calls when variant == 2.
#define bi bi_tbe21_v2_private
#include "bi_gemm.cuh"
#include "bi_tbe21_v2_api.h"

// The isolated backend keeps its historic ABI name. The selected kernel schedule is
// separately bound by the loader, compiler key, runtime identity and qualification receipt.
#ifndef GEOR_TBE21_KERNEL_VARIANT
#define GEOR_TBE21_KERNEL_VARIANT 2
#endif
static_assert(GEOR_TBE21_KERNEL_VARIANT == 1 || GEOR_TBE21_KERNEL_VARIANT == 2,
              "GEOR_TBE21_KERNEL_VARIANT must be 1 or 2");

static_assert(BI_T21_PACKSCAN2 == 0 || BI_T21_PACKSCAN2 == 1, "BI_T21_PACKSCAN2 must be 0 or 1");
static_assert(!BI_T21_PACKSCAN2 || GEOR_TBE21_KERNEL_VARIANT == 1, "paired scan requires schedule 1");
int bi_tbe21_kernel_variant() { return GEOR_TBE21_KERNEL_VARIANT; }
bool bi_tbe21_packed_scan2() { return BI_T21_PACKSCAN2 == 1; }
#ifndef BI_T21_SCAN4
#define BI_T21_SCAN4 0
#endif
static_assert(BI_T21_SCAN4 == 0 || BI_T21_SCAN4 == 1, "BI_T21_SCAN4 must be 0 or 1");
static_assert(!BI_T21_SCAN4 || (GEOR_TBE21_KERNEL_VARIANT == 1 && BI_T21_PACKSCAN2 == 1),
              "BI_T21_SCAN4 requires schedule 1 and paired scan");
bool bi_tbe21_scan4() { return BI_T21_SCAN4 == 1; }
#ifndef BI_T21_TILE128
#define BI_T21_TILE128 0
#endif
static_assert(BI_T21_TILE128 == 0 || BI_T21_TILE128 == 1, "BI_T21_TILE128 must be 0 or 1");
static_assert(!BI_T21_TILE128 || (GEOR_TBE21_KERNEL_VARIANT == 1 && BI_T21_PACKSCAN2 == 1),
              "BI_T21_TILE128 requires schedule 1 and paired scan");
bool bi_tbe21_tile128() { return BI_T21_TILE128 == 1; }
int bi_tbe21_tile_for(int64_t rows) { return BI_T21_TILE128 && rows > 64 ? 128 : 64; }
#ifndef BI_T21_ROW_GEOMETRY
#define BI_T21_ROW_GEOMETRY 0
#endif
static_assert(BI_T21_ROW_GEOMETRY == 0 || BI_T21_ROW_GEOMETRY == 1,
              "BI_T21_ROW_GEOMETRY must be 0 or 1");
static_assert(!BI_T21_ROW_GEOMETRY ||
                  (GEOR_TBE21_KERNEL_VARIANT == 1 && BI_T21_PACKSCAN2 == 1 && BI_T21_TILE128 == 1),
              "BI_T21_ROW_GEOMETRY requires schedule 1, PACKSCAN2 and TILE128");
bool bi_tbe21_row_geometry() { return BI_T21_ROW_GEOMETRY == 1; }
#ifndef BI_T21_EXEC_LAYOUT
#define BI_T21_EXEC_LAYOUT 0
#endif
static_assert(BI_T21_EXEC_LAYOUT == 0 || BI_T21_EXEC_LAYOUT == 1,
              "BI_T21_EXEC_LAYOUT must be 0 or 1");
static_assert(!BI_T21_EXEC_LAYOUT ||
                  (GEOR_TBE21_KERNEL_VARIANT == 1 && BI_T21_PACKSCAN2 && BI_T21_TILE128),
              "BI_T21_EXEC_LAYOUT requires schedule 1, PACKSCAN2 and TILE128");
bool bi_tbe21_exec_layout() { return BI_T21_EXEC_LAYOUT == 1; }
#if BI_T21_EXEC_LAYOUT
cudaError_t bi_tbe21_v2_decode_exec(const BiDesc& d, int S, __nv_bfloat16* out,
                                  cudaStream_t st, int rows) {
  if (rows == 64) return bi::launch_exec_decode<2, 2>(d, S, out, st);
  if (rows == 128) return bi::launch_exec_decode<8, 1>(d, S, out, st);
  return cudaErrorInvalidValue;
}
#endif
#ifndef BI_T21_WIDE128
#define BI_T21_WIDE128 0
#endif
// Keep this selector block valid in standalone host compile checks.
#ifndef BI_T21_SCAN4
#define BI_T21_SCAN4 0
#endif
static_assert(BI_T21_WIDE128 == 0 || BI_T21_WIDE128 == 1, "BI_T21_WIDE128 must be 0 or 1");
static_assert(!BI_T21_WIDE128 ||
                  (GEOR_TBE21_KERNEL_VARIANT == 1 && BI_T21_PACKSCAN2 == 1 && BI_T21_TILE128 == 1 && BI_T21_SCAN4 == 0),
              "BI_T21_WIDE128 requires schedule 1, PACKSCAN2, TILE128 and SCAN4 off");
bool bi_tbe21_wide128() { return BI_T21_WIDE128 == 1; }


#ifndef BI_T21_ASYNC_X
#define BI_T21_ASYNC_X 0
#endif
static_assert(BI_T21_ASYNC_X == 0 || BI_T21_ASYNC_X == 1, "BI_T21_ASYNC_X must be 0 or 1");
static_assert(!BI_T21_ASYNC_X || GEOR_TBE21_KERNEL_VARIANT == 1, "async-X requires schedule 1");
bool bi_tbe21_async_x() { return BI_T21_ASYNC_X == 1; }

#ifndef BI_T21_PIPE128
#define BI_T21_PIPE128 0
#endif
static_assert(BI_T21_PIPE128 == 0 || BI_T21_PIPE128 == 1, "BI_T21_PIPE128 must be 0 or 1");
#if BI_T21_PIPE128
static_assert(!BI_T21_PIPE128 || (GEOR_TBE21_KERNEL_VARIANT == 1 && BI_T21_WIDE128 &&
              BI_T21_PACKSCAN2 && BI_T21_TILE128 && !BI_T21_SCAN4 && !BI_T21_ASYNC_X),
              "BI_T21_PIPE128 requires compact M128, paired scans and synchronous producer X staging");
#endif
bool bi_tbe21_pipe128() { return BI_T21_PIPE128 == 1; }

#ifndef BI_T21_PACKSCAN4
#define BI_T21_PACKSCAN4 0
#endif
static_assert(BI_T21_PACKSCAN4 == 0 || BI_T21_PACKSCAN4 == 1, "BI_T21_PACKSCAN4 must be 0 or 1");
static_assert(!BI_T21_PACKSCAN4 || (GEOR_TBE21_KERNEL_VARIANT == 1 && BI_T21_PACKSCAN2),
              "BI_T21_PACKSCAN4 requires schedule 1 and paired scan");
bool bi_tbe21_packed_scan4() { return BI_T21_PACKSCAN4 == 1; }

#ifndef BI_T21_PAIR_VOTE
#define BI_T21_PAIR_VOTE 0
#endif
#if BI_T21_PAIR_VOTE
static_assert(GEOR_TBE21_KERNEL_VARIANT == 1 && BI_T21_TILE128,
              "PAIR_VOTE requires schedule 1 and selective tile128");
#endif
bool bi_tbe21_pair_vote() { return BI_T21_PAIR_VOTE == 1; }

#if BI_T21_EXEC_LAYOUT
static_assert(BI_T21_TABREG == 1 && BI_T21_V1_MINBLOCKS == 2,
              "BI_T21_EXEC_LAYOUT requires TABREG=1 and V1_MINBLOCKS=2");
static_assert(BI_T21_SCAN == 0 && BI_T21_SCAN4 == 0 && BI_T21_ASYNC_X == 0,
              "BI_T21_EXEC_LAYOUT requires scan, scan4 and async-X off");
static_assert(BI_T21_ROW_GEOMETRY == 0 && BI_T21_WIDE128 == 0 && BI_T21_PIPE128 == 0,
              "BI_T21_EXEC_LAYOUT requires row geometry, wide128 and pipe128 off");
static_assert(BI_T21_PACKSCAN4 == 0 && BI_T21_PAIR_VOTE == 0,
              "BI_T21_EXEC_LAYOUT requires packed_scan4 and pair_vote off");
static_assert(BI_T21_TIER_PREFIX == 0 && BI_T21_TPF_PREFETCH == 0
                  && BI_T21_TPF_WG1 == 0 && BI_T21_TPF_PREFETCH_WG1 == 0,
              "BI_T21_EXEC_LAYOUT requires tier-prefix controls off");
#endif

cudaError_t bi_tbe21_v2_gemm(int fmt, const __nv_bfloat16* x, int64_t ldx, int M, const BiDesc& d,
                             int S, __nv_bfloat16* y, int64_t ldy, float* ws, cudaStream_t st,
                             int tile, int spec) {
  switch (fmt) {
    case bi::F_BF16:
      return spec ? bi::launch_fmt_v2<bi::F_BF16, 0, 1>(x, ldx, M, d, S, y, ldy, ws, st, tile)
                  : bi::launch_fmt_v2<bi::F_BF16, 0, 0>(x, ldx, M, d, S, y, ldy, ws, st, tile);
    case bi::F_TBE:
      return spec ? bi::launch_fmt_v2<bi::F_TBE, 0, 1>(x, ldx, M, d, S, y, ldy, ws, st, tile)
                  : bi::launch_fmt_v2<bi::F_TBE, 0, 0>(x, ldx, M, d, S, y, ldy, ws, st, tile);
    case bi::F_TBE1T:
      return spec ? bi::launch_fmt_v2<bi::F_TBE1T, 0, 1>(x, ldx, M, d, S, y, ldy, ws, st, tile)
                  : bi::launch_fmt_v2<bi::F_TBE1T, 0, 0>(x, ldx, M, d, S, y, ldy, ws, st, tile);
    case bi::F_TBE21:  // SPEC has no speculative path in the TBE21 adapter; one instantiation
      if (BI_T21_TILE128 && tile != bi_tbe21_tile_for(M)) return cudaErrorInvalidValue;
#if GEOR_TBE21_KERNEL_VARIANT == 1
#if BI_T21_WIDE128
      if (M == 128) return bi::launch_tbe21_wide128(x, ldx, M, d, S, y, ldy, ws, st);
#endif
#if BI_T21_ROW_GEOMETRY
      if (M == 64) return bi::launch_tbe21_row_geometry<2>(x, ldx, M, d, S, y, ldy, ws, st);
      if (M >= 129 && M <= 512)
        return bi::launch_tbe21_row_geometry<8>(x, ldx, M, d, S, y, ldy, ws, st);
#endif
      return bi::launch_fmt<bi::F_TBE21, 0>(x, ldx, M, d, S, y, ldy, ws, st, tile);
#else
      return bi::launch_fmt_v2<bi::F_TBE21, 0, 0>(x, ldx, M, d, S, y, ldy, ws, st, tile);
#endif
    default: return cudaErrorInvalidValue;
  }
}

cudaError_t bi_tbe21_v2_decode(int fmt, const BiDesc& d, int S, __nv_bfloat16* out,
                               cudaStream_t st, int spec) {
  switch (fmt) {
    case bi::F_BF16:
      return spec ? bi::launch_decode_v2<bi::F_BF16, 0, 1>(d, S, out, st)
                  : bi::launch_decode_v2<bi::F_BF16, 0, 0>(d, S, out, st);
    case bi::F_TBE:
      return spec ? bi::launch_decode_v2<bi::F_TBE, 0, 1>(d, S, out, st)
                  : bi::launch_decode_v2<bi::F_TBE, 0, 0>(d, S, out, st);
    case bi::F_TBE1T:
      return spec ? bi::launch_decode_v2<bi::F_TBE1T, 0, 1>(d, S, out, st)
                  : bi::launch_decode_v2<bi::F_TBE1T, 0, 0>(d, S, out, st);
    case bi::F_TBE21:
#if GEOR_TBE21_KERNEL_VARIANT == 1
#if BI_T21_TIER_PREFIX
      return bi_tbe21_v2_decode_prefix(d, S, out, st, 2);
#elif BI_T21_WIDE128
      return bi::launch_decode<bi::F_TBE21, 0>(d, S, out, st, 4);
#else
      return bi::launch_decode<bi::F_TBE21, 0>(d, S, out, st, 1);
#endif
#else
      return bi::launch_decode_v2<bi::F_TBE21, 0, 0>(d, S, out, st);
#endif
    default: return cudaErrorInvalidValue;
  }
}

static_assert(!BI_T21_TIER_PREFIX || GEOR_TBE21_KERNEL_VARIANT == 1,
              "tier prefix requires schedule 1");
bool bi_tbe21_tier_prefix() { return BI_T21_TIER_PREFIX == 1; }
bool bi_tbe21_tier_prefetch() { return BI_T21_TPF_PREFETCH == 1; }
bool bi_tbe21_tier_prefix_wg1() { return BI_T21_TPF_WG1 == 1; }
bool bi_tbe21_tier_prefetch_wg1() { return BI_T21_TPF_PREFETCH_WG1 == 1; }

cudaError_t bi_tbe21_make_tier_headers(const BiDesc& d, uint32_t* out, cudaStream_t st) {
#if BI_T21_TIER_PREFIX
  return bi::launch_tpf_headers(d, out, st);
#else
  return cudaErrorNotSupported;
#endif
}

cudaError_t bi_tbe21_v2_decode_prefix(const BiDesc& d, int S, __nv_bfloat16* out,
                                    cudaStream_t st, int wg) {
#if BI_T21_TIER_PREFIX
  if (!d.tpf_counts || (wg != 2 && !(wg == 1 && BI_T21_TPF_WG1)))
    return cudaErrorInvalidValue;
  return bi::launch_decode<bi::F_TBE21, 0>(d, S, out, st, wg);
#else
  return cudaErrorNotSupported;
#endif
}

cudaError_t bi_tbe21_v2_attrs(int fmt, int mt, int spec, BiKernelAttrs* a) {
  switch (fmt) {
    case bi::F_BF16:
      return spec ? bi::attrs_fmt_v2<bi::F_BF16, 0, 1>(mt, a)
                  : bi::attrs_fmt_v2<bi::F_BF16, 0, 0>(mt, a);
    case bi::F_TBE:
      return spec ? bi::attrs_fmt_v2<bi::F_TBE, 0, 1>(mt, a)
                  : bi::attrs_fmt_v2<bi::F_TBE, 0, 0>(mt, a);
    case bi::F_TBE1T:
      return spec ? bi::attrs_fmt_v2<bi::F_TBE1T, 0, 1>(mt, a)
                  : bi::attrs_fmt_v2<bi::F_TBE1T, 0, 0>(mt, a);
    case bi::F_TBE21:
#if GEOR_TBE21_KERNEL_VARIANT == 1
      return bi::attrs_fmt<bi::F_TBE21, 0>(mt, 1, a);
#else
      return bi::attrs_fmt_v2<bi::F_TBE21, 0, 0>(mt, a);
#endif
    default: return cudaErrorInvalidValue;
  }
}

// Query attributes for the exact TBE21 GEMM specialization selected by M and this
// translation unit's compiled schedule/tile policy.
cudaError_t bi_tbe21_v2_runtime_attrs(int M, BiKernelAttrs* a) {
  if (M <= 0 || a == nullptr) return cudaErrorInvalidValue;
#if BI_T21_WIDE128
  if (M == 128) return bi::attrs_of<bi::F_TBE21, 0, 2, 4>(a);
#endif
  const int tile = bi_tbe21_tile_for(M);
  if (tile != 64 && tile != 128) return cudaErrorInvalidValue;
  int wg = tile / 64;
  int mt = wg == 2 ? (M <= 96 ? 3 : 4) : (M >= 64 ? 4 : (M + 15) / 16);
#if BI_T21_ROW_GEOMETRY
  if (M == 64) { wg = 2; mt = 2; }
  else if (M >= 129 && M <= 512) { wg = 2; mt = 8; }
#endif
#if BI_T21_EXEC_LAYOUT
  // The C64 execution contract uses the compact two-warp-group producer
  // (MT2/WG2, still a tile64/512-thread launch).  The C128 contract remains
  // the existing MT8/WG1 specialization.  Keep all other row ranges on the
  // ordinary schedule.
  if (M >= 49 && M <= 64) { wg = 2; mt = 2; }
  else if (M >= 97 && M <= 128) { wg = 1; mt = 8; }
#endif
#if GEOR_TBE21_KERNEL_VARIANT == 1
  return bi::attrs_fmt<bi::F_TBE21, 0>(mt, wg, a);
#else
  return bi::attrs_fmt_v2<bi::F_TBE21, 0, 0>(mt, a);
#endif
}
