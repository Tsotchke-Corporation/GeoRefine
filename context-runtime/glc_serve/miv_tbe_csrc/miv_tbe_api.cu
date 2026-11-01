// MIV-TBE C API: M dispatch, decode-only twin, load-time index helper.
#include "miv_tbe_kernels.cuh"

namespace miv_tbe {

template <int WPR, int U>
inline cudaError_t decode_wu(const MivTbeDesc& d, __nv_bfloat16* w, cudaStream_t st) {
  constexpr int RPC = 8 / WPR;
  const int grid = (d.n + RPC - 1) / RPC;
  miv_tbe_decode_kernel_v2<WPR, U><<<grid, 256, 0, st>>>(make_args(d), w);
  return cudaGetLastError();
}

template <int WPR>
inline cudaError_t decode_w(const MivTbeDesc& d, __nv_bfloat16* w, cudaStream_t st) {
  switch (d.unroll) {
    case 1: return decode_wu<WPR, 1>(d, w, st);
    case 2: return decode_wu<WPR, 2>(d, w, st);
    case 4: return decode_wu<WPR, 4>(d, w, st);
    default: return cudaErrorInvalidValue;
  }
}

__global__ void tile_escapes_kernel(const uint32_t* __restrict__ planes, int64_t T,
                                    int32_t* __restrict__ out) {
  const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  const uint32_t* p = planes + t * 6;
  out[t] = __popc(~(p[0] | p[2] | p[4])) + __popc(~(p[1] | p[3] | p[5]));
}

}  // namespace miv_tbe

static bool desc_ok(const MivTbeDesc& d) {
  if (d.n <= 0 || d.k <= 0) return false;
  if (!(d.wpr == 1 || d.wpr == 2 || d.wpr == 4 || d.wpr == 8)) return false;
  if (!(d.unroll == 1 || d.unroll == 2 || d.unroll == 4)) return false;
  if (d.k % (64 * d.wpr)) return false;
  if (!d.planes || !d.smb || !d.esc || !d.rowparam || !d.escbase) return false;
  if (reinterpret_cast<uintptr_t>(d.esc) & 3) return false;
  return true;
}

cudaError_t miv_tbe_gemv(const MivTbeDesc& d, const __nv_bfloat16* x, int64_t ldx, int M,
                         __nv_bfloat16* y, int64_t ldy, cudaStream_t st) {
  if (!desc_ok(d) || (ldx & 7) || (reinterpret_cast<uintptr_t>(x) & 15)) return cudaErrorInvalidValue;
  switch (M) {
    case 1: return miv_tbe_gemv_m1(d, x, ldx, y, ldy, st);
    case 2: return miv_tbe_gemv_m2(d, x, ldx, y, ldy, st);
    case 3: return miv_tbe_gemv_m3(d, x, ldx, y, ldy, st);
    case 4: return miv_tbe_gemv_m4(d, x, ldx, y, ldy, st);
    case 5: return miv_tbe_gemv_m5(d, x, ldx, y, ldy, st);
    case 6: return miv_tbe_gemv_m6(d, x, ldx, y, ldy, st);
    case 7: return miv_tbe_gemv_m7(d, x, ldx, y, ldy, st);
    case 8: return miv_tbe_gemv_m8(d, x, ldx, y, ldy, st);
    default: return cudaErrorInvalidValue;
  }
}

cudaError_t miv_tbe_decode(const MivTbeDesc& d, __nv_bfloat16* w, cudaStream_t st) {
  if (!desc_ok(d) || (reinterpret_cast<uintptr_t>(w) & 15)) return cudaErrorInvalidValue;
  switch (d.wpr) {
    case 1: return miv_tbe::decode_w<1>(d, w, st);
    case 2: return miv_tbe::decode_w<2>(d, w, st);
    case 4: return miv_tbe::decode_w<4>(d, w, st);
    case 8: return miv_tbe::decode_w<8>(d, w, st);
    default: return cudaErrorInvalidValue;
  }
}

cudaError_t miv_tbe_tile_escapes(const uint32_t* planes, int64_t T, int32_t* out,
                                 cudaStream_t st) {
  if (T <= 0) return cudaSuccess;
  miv_tbe::tile_escapes_kernel<<<(unsigned)((T + 255) / 256), 256, 0, st>>>(planes, T, out);
  return cudaGetLastError();
}
