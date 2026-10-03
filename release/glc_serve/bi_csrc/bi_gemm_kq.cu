// BI-GEMM instantiations: llama.cpp K-quant / i-quant types (miv_kq decode).  See bi_gemm.cuh.
#include "bi_gemm.cuh"

cudaError_t bi_gemm_kq(int kqt, const __nv_bfloat16* x, int64_t ldx, int M, const BiDesc& d,
                       int S, __nv_bfloat16* y, int64_t ldy, float* ws, cudaStream_t st) {
  using namespace bi;
  switch (kqt) {
    case KQ_Q2_K: return launch_fmt<F_KQ, KQ_Q2_K>(x, ldx, M, d, S, y, ldy, ws, st);
    case KQ_Q3_K: return launch_fmt<F_KQ, KQ_Q3_K>(x, ldx, M, d, S, y, ldy, ws, st);
    case KQ_Q4_K: return launch_fmt<F_KQ, KQ_Q4_K>(x, ldx, M, d, S, y, ldy, ws, st);
    case KQ_Q5_K: return launch_fmt<F_KQ, KQ_Q5_K>(x, ldx, M, d, S, y, ldy, ws, st);
    case KQ_Q6_K: return launch_fmt<F_KQ, KQ_Q6_K>(x, ldx, M, d, S, y, ldy, ws, st);
    case KQ_IQ2_XS: return launch_fmt<F_KQ, KQ_IQ2_XS>(x, ldx, M, d, S, y, ldy, ws, st);
    case KQ_IQ2_S: return launch_fmt<F_KQ, KQ_IQ2_S>(x, ldx, M, d, S, y, ldy, ws, st);
    case KQ_IQ3_XXS: return launch_fmt<F_KQ, KQ_IQ3_XXS>(x, ldx, M, d, S, y, ldy, ws, st);
    case KQ_IQ3_S: return launch_fmt<F_KQ, KQ_IQ3_S>(x, ldx, M, d, S, y, ldy, ws, st);
    case KQ_IQ4_XS: return launch_fmt<F_KQ, KQ_IQ4_XS>(x, ldx, M, d, S, y, ldy, ws, st);
    default: return cudaErrorInvalidValue;
  }
}
