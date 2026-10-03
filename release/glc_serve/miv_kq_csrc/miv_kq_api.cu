#include "miv_kq_kernels.cuh"

cudaError_t miv_kq_gemv_m1(const MivKqDesc&, const __nv_bfloat16*, int64_t, __nv_bfloat16*, int64_t, cudaStream_t);
cudaError_t miv_kq_gemv_m2(const MivKqDesc&, const __nv_bfloat16*, int64_t, __nv_bfloat16*, int64_t, cudaStream_t);
cudaError_t miv_kq_gemv_m3(const MivKqDesc&, const __nv_bfloat16*, int64_t, __nv_bfloat16*, int64_t, cudaStream_t);
cudaError_t miv_kq_gemv_m4(const MivKqDesc&, const __nv_bfloat16*, int64_t, __nv_bfloat16*, int64_t, cudaStream_t);
cudaError_t miv_kq_gemv_m5(const MivKqDesc&, const __nv_bfloat16*, int64_t, __nv_bfloat16*, int64_t, cudaStream_t);
cudaError_t miv_kq_gemv_m6(const MivKqDesc&, const __nv_bfloat16*, int64_t, __nv_bfloat16*, int64_t, cudaStream_t);
cudaError_t miv_kq_gemv_m7(const MivKqDesc&, const __nv_bfloat16*, int64_t, __nv_bfloat16*, int64_t, cudaStream_t);
cudaError_t miv_kq_gemv_m8(const MivKqDesc&, const __nv_bfloat16*, int64_t, __nv_bfloat16*, int64_t, cudaStream_t);

static bool ok(const MivKqDesc& d) {
  if (d.n <= 0 || d.k <= 0 || d.k % 256) return false;
  if (!(d.wpr == 1 || d.wpr == 2 || d.wpr == 4 || d.wpr == 8)) return false;
  if (!(d.unroll == 1 || d.unroll == 2 || d.unroll == 4)) return false;
  if (!d.a0 || !d.a1) return false;
  const bool grid = d.type == KQ_IQ3_XXS || d.type == KQ_IQ3_S || d.type == KQ_IQ2_XS || d.type == KQ_IQ2_S;
  if (grid && !d.grid) return false;
  if ((d.type == KQ_IQ3_XXS || d.type == KQ_IQ2_XS) && !d.ksigns) return false;
  return true;
}

cudaError_t miv_kq_gemv(const MivKqDesc& d, const __nv_bfloat16* x, int64_t ldx, int M,
                        __nv_bfloat16* y, int64_t ldy, cudaStream_t st) {
  if (!ok(d) || (ldx & 7) || (reinterpret_cast<uintptr_t>(x) & 15)) return cudaErrorInvalidValue;
  switch (M) {
    case 1: return miv_kq_gemv_m1(d, x, ldx, y, ldy, st);
    case 2: return miv_kq_gemv_m2(d, x, ldx, y, ldy, st);
    case 3: return miv_kq_gemv_m3(d, x, ldx, y, ldy, st);
    case 4: return miv_kq_gemv_m4(d, x, ldx, y, ldy, st);
    case 5: return miv_kq_gemv_m5(d, x, ldx, y, ldy, st);
    case 6: return miv_kq_gemv_m6(d, x, ldx, y, ldy, st);
    case 7: return miv_kq_gemv_m7(d, x, ldx, y, ldy, st);
    case 8: return miv_kq_gemv_m8(d, x, ldx, y, ldy, st);
    default: return cudaErrorInvalidValue;
  }
}

template <int T>
static cudaError_t deq_t(const MivKqDesc& d, float* w, cudaStream_t st) {
  const int64_t nvec = (int64_t)d.n * d.k / 8;
  miv_kq::kq_dequant_kernel<T><<<(unsigned)((nvec + 255) / 256), 256, 0, st>>>(
      miv_kq::make_args(d), w, nvec);
  return cudaGetLastError();
}

cudaError_t miv_kq_dequant_f32(const MivKqDesc& d, float* w, cudaStream_t st) {
  if (!ok(d) || (reinterpret_cast<uintptr_t>(w) & 15)) return cudaErrorInvalidValue;
  switch (d.type) {
    case KQ_Q4_K: return deq_t<KQ_Q4_K>(d, w, st);
    case KQ_Q5_K: return deq_t<KQ_Q5_K>(d, w, st);
    case KQ_Q6_K: return deq_t<KQ_Q6_K>(d, w, st);
    case KQ_IQ4_XS: return deq_t<KQ_IQ4_XS>(d, w, st);
    case KQ_IQ3_XXS: return deq_t<KQ_IQ3_XXS>(d, w, st);
    case KQ_IQ3_S: return deq_t<KQ_IQ3_S>(d, w, st);
    case KQ_Q2_K: return deq_t<KQ_Q2_K>(d, w, st);
    case KQ_Q3_K: return deq_t<KQ_Q3_K>(d, w, st);
    case KQ_IQ2_XS: return deq_t<KQ_IQ2_XS>(d, w, st);
    case KQ_IQ2_S: return deq_t<KQ_IQ2_S>(d, w, st);
    default: return cudaErrorInvalidValue;
  }
}
