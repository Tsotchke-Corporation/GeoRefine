// Experimental configurable-physical-warp M1 kernel. No default dispatch uses this TU.
#include "miv_tbe_kernels.cuh"

#ifndef MIV_TBE_PAIR_PHYSICAL_WARPS
#define MIV_TBE_PAIR_PHYSICAL_WARPS 4
#endif
#if MIV_TBE_PAIR_PHYSICAL_WARPS != 4 && MIV_TBE_PAIR_PHYSICAL_WARPS != 8
#error "MIV_TBE_PAIR_PHYSICAL_WARPS must be 4 or 8"
#endif

#ifndef MIV_TBE_PAIR_MAX_REGISTERS
#define MIV_TBE_PAIR_MAX_REGISTERS 0
#endif
#if MIV_TBE_PAIR_MAX_REGISTERS != 0 && MIV_TBE_PAIR_MAX_REGISTERS != 32 && \
    MIV_TBE_PAIR_MAX_REGISTERS != 40 && MIV_TBE_PAIR_MAX_REGISTERS != 48 && \
    MIV_TBE_PAIR_MAX_REGISTERS != 64
#error "MIV_TBE_PAIR_MAX_REGISTERS must be 0, 32, 40, 48, or 64"
#endif

namespace {

template <int WPR, int U>
__global__ void __launch_bounds__(MIV_TBE_PAIR_PHYSICAL_WARPS * 32)
miv_tbe_pair_kernel_v2(const __nv_bfloat16* __restrict__ x,
                            int64_t ldx, miv_tbe::Args a,
                            __nv_bfloat16* __restrict__ y, int64_t ldy) {
  constexpr int RPC = 8 / WPR;
  const int physical_warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  __shared__ float red[8];

  // Each physical warp serially executes logical warp ids separated by the
  // physical warp count. Four warps therefore process two phases; eight process
  // one. Keeping walk_v2's lane id and traversal unchanged preserves escape
  // prefix gathers and per-lane vector order.
#pragma unroll
  for (int phase = 0; phase < 8 / MIV_TBE_PAIR_PHYSICAL_WARPS; ++phase) {
    const int warp = physical_warp + phase * MIV_TBE_PAIR_PHYSICAL_WARPS;
    const int rloc = warp / WPR;
    const int slice = warp % WPR;
    const int row = blockIdx.x * RPC + rloc;
    float acc[1] = {0.0f};
    if (row < a.N) {
      miv_tbe::FmaSink<1> sink{acc, x, ldx};
      miv_tbe::walk_v2<WPR, U>(a, row, slice, lane, sink);
    }
    for (int off = 16; off > 0; off >>= 1)
      acc[0] = __fadd_rn(acc[0], __shfl_xor_sync(0xffffffffu, acc[0], off));
    if (lane == 0) red[warp] = acc[0];
    __syncthreads();
  }

  // Match epilogue<M=1,WPR>: fixed slice-order additions, then bias, then one
  // bf16 round. Logical row indexing keeps the old 8-warp schedule's outputs.
  if (threadIdx.x < RPC) {
    const int r = threadIdx.x;
    const int row = blockIdx.x * RPC + r;
    if (row < a.N) {
      float value = red[r * WPR];
#pragma unroll
      for (int j = 1; j < WPR; ++j) value = __fadd_rn(value, red[r * WPR + j]);
      if (a.bias) value = __fadd_rn(value, __bfloat162float(a.bias[row]));
      y[row] = __float2bfloat16_rn(value);
    }
  }
}

template <int WPR, int U>
cudaError_t launch(const MivTbeDesc& d, const __nv_bfloat16* x, int64_t ldx,
                   __nv_bfloat16* y, int64_t ldy, cudaStream_t stream) {
  constexpr int RPC = 8 / WPR;
  const int grid = (d.n + RPC - 1) / RPC;
  miv_tbe_pair_kernel_v2<WPR, U><<<grid, MIV_TBE_PAIR_PHYSICAL_WARPS * 32, 0, stream>>>(
      x, ldx, miv_tbe::make_args(d), y, ldy);
  return cudaGetLastError();
}

template <int WPR>
cudaError_t launch_unroll(const MivTbeDesc& d, const __nv_bfloat16* x, int64_t ldx,
                          __nv_bfloat16* y, int64_t ldy, cudaStream_t stream) {
  switch (d.unroll) {
    case 1: return launch<WPR, 1>(d, x, ldx, y, ldy, stream);
    case 2: return launch<WPR, 2>(d, x, ldx, y, ldy, stream);
    case 4: return launch<WPR, 4>(d, x, ldx, y, ldy, stream);
    default: return cudaErrorInvalidValue;
  }
}

}  // namespace

cudaError_t miv_tbe_pair_m1(const MivTbeDesc& d, const __nv_bfloat16* x, int64_t ldx,
                            __nv_bfloat16* y, int64_t ldy, cudaStream_t stream) {
  switch (d.wpr) {
    case 1: return launch_unroll<1>(d, x, ldx, y, ldy, stream);
    case 2: return launch_unroll<2>(d, x, ldx, y, ldy, stream);
    case 4: return launch_unroll<4>(d, x, ldx, y, ldy, stream);
    case 8: return launch_unroll<8>(d, x, ldx, y, ldy, stream);
    default: return cudaErrorInvalidValue;
  }
}
