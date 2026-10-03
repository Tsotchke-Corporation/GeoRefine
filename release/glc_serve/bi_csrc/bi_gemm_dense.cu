// BI-GEMM instantiations: BF16, Q8_0, TBE.  See bi_gemm.cuh.
#include "bi_gemm.cuh"

cudaError_t bi_gemm_dense(int fmt, const __nv_bfloat16* x, int64_t ldx, int M, const BiDesc& d,
                          int S, __nv_bfloat16* y, int64_t ldy, float* ws, cudaStream_t st) {
  switch (fmt) {
    case bi::F_BF16: return bi::launch_fmt<bi::F_BF16, 0>(x, ldx, M, d, S, y, ldy, ws, st);
    case bi::F_Q8: return bi::launch_fmt<bi::F_Q8, 0>(x, ldx, M, d, S, y, ldy, ws, st);
    case bi::F_TBE: return bi::launch_fmt<bi::F_TBE, 0>(x, ldx, M, d, S, y, ldy, ws, st);
    default: return cudaErrorInvalidValue;
  }
}
