// Separate experimental extension. Original BI source/default library is untouched.
// Function-local GNU-unique statics must not alias the legacy DSO's
// optin_smem::done. Keep literal arithmetic/header bytes unchanged.
#define bi bi_tbe2_opt_private
#include "bi_tbe2_opt.cuh"
#undef bi
cudaError_t bi_gemm_dense(int fmt, const __nv_bfloat16* x, int64_t ldx, int M, const BiDesc& d,
                          int S, __nv_bfloat16* y, int64_t ldy, float* ws, cudaStream_t st, int tile) {
  switch (fmt) {
    case BI_BF16: return bi_tbe2_opt_private::launch_fmt<BI_BF16,0>(x,ldx,M,d,S,y,ldy,ws,st,tile);
    case BI_TBE2: return bi_tbe2_opt_private::launch_fmt<BI_TBE2,0>(x,ldx,M,d,S,y,ldy,ws,st,tile);
    default: return cudaErrorInvalidValue;
  }
}
cudaError_t bi_decode_dense(int fmt,const BiDesc& d,int S,__nv_bfloat16* out,cudaStream_t st,int wg) {
  switch (fmt) {
    case BI_BF16: return bi_tbe2_opt_private::launch_decode<BI_BF16,0>(d,S,out,st,wg);
    case BI_TBE2: return bi_tbe2_opt_private::launch_decode<BI_TBE2,0>(d,S,out,st,wg);
    default: return cudaErrorInvalidValue;
  }
}
cudaError_t bi_attrs_dense(int fmt,int mt,int wg,BiKernelAttrs* a) {
  switch (fmt) {
    case BI_BF16: return bi_tbe2_opt_private::attrs_fmt<BI_BF16,0>(mt,wg,a);
    case BI_TBE2: return bi_tbe2_opt_private::attrs_fmt<BI_TBE2,0>(mt,wg,a);
    default: return cudaErrorInvalidValue;
  }
}
#ifndef GEOR_REFINED_CONTEXT_ONLY
cudaError_t bi_gemm_kq(int,const __nv_bfloat16*,int64_t,int,const BiDesc&,int,__nv_bfloat16*,int64_t,
                      float*,cudaStream_t,int) { return cudaErrorInvalidValue; }
#endif

// Shared Context bindings expose this symbol; this recovered backend is TBE2-only.
cudaError_t bi_t21_checkpoints(const BiDesc&, int, uint32_t*, cudaStream_t) {
  return cudaErrorInvalidValue;
}
