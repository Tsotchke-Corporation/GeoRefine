#pragma once
#include "bi_api.h"

// Private v2 wrappers. Keep these names unique across loaded CUDA extensions.
cudaError_t bi_tbe21_v2_gemm(int fmt, const __nv_bfloat16* x, int64_t ldx, int M,
                              const BiDesc& d, int S, __nv_bfloat16* y, int64_t ldy,
                              float* ws, cudaStream_t st, int tile, int spec);
cudaError_t bi_tbe21_v2_decode(int fmt, const BiDesc& d, int S, __nv_bfloat16* out,
                                cudaStream_t st, int spec);
cudaError_t bi_tbe21_v2_attrs(int fmt, int mt, int spec, BiKernelAttrs* a);
cudaError_t bi_tbe21_v2_runtime_attrs(int M, BiKernelAttrs* a);

// Compiled translation-unit schedule, used to bind the runtime receipt.
int bi_tbe21_kernel_variant();
bool bi_tbe21_packed_scan2();
bool bi_tbe21_scan4();
bool bi_tbe21_wide128();
bool bi_tbe21_pipe128();
bool bi_tbe21_packed_scan4();
bool bi_tbe21_pair_vote();
bool bi_tbe21_tier_prefix();
bool bi_tbe21_tier_prefetch();
bool bi_tbe21_tier_prefix_wg1();
bool bi_tbe21_tier_prefetch_wg1();
cudaError_t bi_tbe21_make_tier_headers(const BiDesc& d, uint32_t* out, cudaStream_t st);
cudaError_t bi_tbe21_v2_decode_prefix(const BiDesc& d, int S, __nv_bfloat16* out,
                                    cudaStream_t st, int wg);

// Default-off selective row-tile policy, implemented in the compiled CUDA TU.
bool bi_tbe21_tile128();
int bi_tbe21_tile_for(int64_t rows);
bool bi_tbe21_row_geometry();
bool bi_tbe21_exec_layout();
bool bi_tbe21_async_x();

#if BI_T21_EXEC_LAYOUT
cudaError_t bi_tbe21_v2_decode_exec(const BiDesc& d, int S, __nv_bfloat16* out,
                                  cudaStream_t st, int rows);
#endif
