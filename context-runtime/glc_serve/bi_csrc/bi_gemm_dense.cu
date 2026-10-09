// BI-GEMM instantiations: BF16, Q8_0, TBE, TBE2 (codec v2), TBE21 (codec v2.1).  See bi_gemm.cuh.
#include "bi_gemm.cuh"

cudaError_t bi_gemm_dense(int fmt, const __nv_bfloat16* x, int64_t ldx, int M, const BiDesc& d,
                          int S, __nv_bfloat16* y, int64_t ldy, float* ws, cudaStream_t st,
                          int tile) {
  switch (fmt) {
    case bi::F_BF16: return bi::launch_fmt<bi::F_BF16, 0>(x, ldx, M, d, S, y, ldy, ws, st, tile);
    case bi::F_Q8: return bi::launch_fmt<bi::F_Q8, 0>(x, ldx, M, d, S, y, ldy, ws, st, tile);
    case bi::F_TBE: return bi::launch_fmt<bi::F_TBE, 0>(x, ldx, M, d, S, y, ldy, ws, st, tile);
    case bi::F_TBE2: return bi::launch_fmt<bi::F_TBE2, 0>(x, ldx, M, d, S, y, ldy, ws, st, tile);
    case bi::F_TBE21:
      if (S > 1 && d.ckS != S) return cudaErrorInvalidValue;   // checkpoints built for another S
      return bi::launch_fmt<bi::F_TBE21, 0>(x, ldx, M, d, S, y, ldy, ws, st, tile);
    default: return cudaErrorInvalidValue;
  }
}

cudaError_t bi_decode_dense(int fmt, const BiDesc& d, int S, __nv_bfloat16* out, cudaStream_t st,
                            int wg) {
  switch (fmt) {
    case bi::F_BF16: return bi::launch_decode<bi::F_BF16, 0>(d, S, out, st, wg);
    case bi::F_TBE: return bi::launch_decode<bi::F_TBE, 0>(d, S, out, st, wg);
    case bi::F_TBE2: return bi::launch_decode<bi::F_TBE2, 0>(d, S, out, st, wg);
    case bi::F_TBE21:
      if (S > 1 && d.ckS != S) return cudaErrorInvalidValue;
      return bi::launch_decode<bi::F_TBE21, 0>(d, S, out, st, wg);
    default: return cudaErrorInvalidValue;
  }
}

cudaError_t bi_attrs_dense(int fmt, int mt, int wg, BiKernelAttrs* a) {
  switch (fmt) {
    case bi::F_BF16: return bi::attrs_fmt<bi::F_BF16, 0>(mt, wg, a);
    case bi::F_TBE: return bi::attrs_fmt<bi::F_TBE, 0>(mt, wg, a);
    case bi::F_TBE2: return bi::attrs_fmt<bi::F_TBE2, 0>(mt, wg, a);
    case bi::F_TBE21: return bi::attrs_fmt<bi::F_TBE21, 0>(mt, wg, a);
    default: return cudaErrorInvalidValue;
  }
}

cudaError_t bi_t21_checkpoints(const BiDesc& d, int S, uint32_t* out, cudaStream_t st) {
  return bi::launch_t21_ckpt<0>(d, S, out, st);
}
