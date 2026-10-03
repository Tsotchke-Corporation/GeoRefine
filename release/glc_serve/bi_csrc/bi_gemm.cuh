// BI-GEMM: batch-invariant tensor-core GEMM for continuous-batching decode.
//
//   y[m, n] = round_bf16( sum_k x[m, k] * W[n, k]  (+ bias[n]) ),   1 <= M <= 4096
//
// Batch-invariance contract (G3a-BI): row m of the output is a function of row m of x
// and of W ONLY.  It does not depend on M, on the position of the row inside the batch,
// or on the other rows' values:
//   * the K reduction of every output element is a FIXED sequence: K is cut into
//     128-wide chunks; the chunks are split into S contiguous ranges where S depends on
//     (N, K) only (never on M); inside a range, chunks run in increasing order and each
//     chunk runs 8 mma.m16n8k16 (bf16 in, fp32 accumulate) in increasing k;
//   * the S fp32 partials are added in split order 0..S-1 by one thread per element
//     (no atomics), then + bias in fp32, then one bf16 rounding;
//   * a row sits in some m16 tile of some CTA; the MMA for an output element reads only
//     its own A row and B column, so neither the tile, nor the slot inside it, nor the
//     number of tiles changes its bits.  (Gated by test, not assumed: bi_gate.py.)
//
// Weights are decoded IN THE KERNEL into shared memory, 128 columns x 64 rows at a time,
// by the SAME device decode functions the M-invariant MIV kernels use:
//   BF16  -- the bf16 bytes,
//   Q8_0  -- bf16_rn(d * q)                       (miv_gemv.py dq8, copied verbatim),
//   TBE   -- miv_tbe::decode_vec_v2               (the exact TBE codec: bits == parent),
//   KQ    -- miv_kq::load<T> + miv_kq::deq_bf16<T> (bf16_rn of gguf-py's dequantize).
// So for G1-identical weights, the TBE-coded GEMM is bitwise the bf16 GEMM on the parent,
// and the K-quant GEMM is bitwise the bf16 GEMM on gguf-py's dequantized weights.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "bi_api.h"
#include "miv_kq_kernels.cuh"
#include "miv_tbe_kernels.cuh"

namespace bi {

constexpr int BN = 64;          // weight rows per CTA
constexpr int CH = 128;         // K columns per chunk
constexpr int LDS = CH + 8;     // smem row stride in bf16 (272 B: ldmatrix conflict-free)
constexpr int MTILE = 64;       // activation rows per CTA (4 m16 tiles)

enum Fmt { F_BF16 = BI_BF16, F_Q8 = BI_Q8, F_TBE = BI_TBE, F_KQ = BI_KQ };

struct WDesc {
  int N, K;
  // BF16
  const __nv_bfloat16* __restrict__ w;
  // Q8_0 (SoA): qs int8 [N][K], sc fp16 [N][K/32]
  const int8_t* __restrict__ qs;
  const unsigned short* __restrict__ sc;
  // TBE (mma16 layout, glc_serve.tbe_desc.TBELin arrays)
  const uint32_t* __restrict__ planes;
  const uint8_t* __restrict__ smb;
  const uint8_t* __restrict__ esc;
  const int32_t* __restrict__ rowparam;
  const int32_t* __restrict__ escidx;   // [N][K/128]: index in esc of the chunk's first escape
  // KQ
  miv_kq::Args kq;
  const __nv_bfloat16* __restrict__ bias;
};

__device__ __forceinline__ uint4 ld_nc_u4(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint2 ld_nc_u2(const void* p) {
  uint2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t ld_nc_u16(const void* p) {
  unsigned short r;
  asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(r) : "l"(p));
  return (uint32_t)r;
}
__device__ __forceinline__ uint32_t ld_nc_u32(const void* p) {
  uint32_t r;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}

// ---------------------------------------------------------------- Q8_0 (verbatim miv_gemv.py dq8)
__device__ __forceinline__ uint32_t bfbits(float f) {
  return (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(f));
}
__device__ __forceinline__ uint4 dq8(const uint2 q, const uint32_t s16) {
  const float d = __half2float(__ushort_as_half((unsigned short)s16));
  uint32_t h[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const uint32_t word = e < 4 ? q.x : q.y;
    const int8_t qi = (int8_t)((word >> (8 * (e & 3))) & 0xffu);
    h[e] = bfbits(__fmul_rn(d, (float)qi));
  }
  uint4 w;
  w.x = h[0] | (h[1] << 16); w.y = h[2] | (h[3] << 16);
  w.z = h[4] | (h[5] << 16); w.w = h[6] | (h[7] << 16);
  return w;
}

// ---------------------------------------------------------------- per-format loaders
// Each warp owns 8 weight rows of the CTA's 64; a half-warp (16 lanes) covers one row's
// 128-column chunk, lane j of the half producing columns 8j .. 8j+7.  load() issues the
// global loads (raw registers); dec() turns them into the 8 bf16 weights.  dec() is called
// by ALL 32 lanes (the TBE decode shuffles).
template <int FMT, int KQT>
struct Ld;

template <int KQT>
struct Ld<F_BF16, KQT> {
  struct R { uint4 v; };
  static __device__ __forceinline__ R load(const WDesc& d, int row, int kk, int) {
    R r; r.v = ld_nc_u4(d.w + (int64_t)row * d.K + kk); return r;
  }
  static __device__ __forceinline__ uint4 dec(const WDesc&, const R& r, int, int, int, int) { return r.v; }
};

template <int KQT>
struct Ld<F_Q8, KQT> {
  struct R { uint2 q; uint32_t s; };
  static __device__ __forceinline__ R load(const WDesc& d, int row, int kk, int) {
    R r;
    r.q = ld_nc_u2(d.qs + (int64_t)row * d.K + kk);
    r.s = ld_nc_u16(d.sc + (int64_t)row * (d.K >> 5) + (kk >> 5));
    return r;
  }
  static __device__ __forceinline__ uint4 dec(const WDesc&, const R& r, int, int, int, int) {
    return dq8(r.q, r.s);
  }
};

template <int KQT>
struct Ld<F_KQ, KQT> {
  struct R { miv_kq::Raw raw; };
  static __device__ __forceinline__ R load(const WDesc& d, int row, int kk, int) {
    R r; r.raw = miv_kq::load<KQT>(d.kq, (int64_t)row * (d.K >> 8) + (kk >> 8), kk & 255); return r;
  }
  static __device__ __forceinline__ uint4 dec(const WDesc& d, const R& r, int, int kk, int, int) {
    return miv_kq::deq_bf16<KQT>(d.kq, r.raw, kk & 255);
  }
};

template <int KQT>
struct Ld<F_TBE, KQT> {
  struct R { uint2 p0, p1, p2; uint32_t s[4]; uint32_t eb; uint32_t rp; };
  // kk = chunk*128 + 8j; the chunk holds tiles 2c, 2c+1; half-lane j<8 -> tile 2c, j>=8 -> 2c+1
  static __device__ __forceinline__ R load(const WDesc& d, int row, int kk, int lane) {
    R r;
    const int j = lane & 15;
    const int64_t t = (int64_t)row * (d.K >> 6) + (kk >> 6);
    const uint32_t* pp = d.planes + t * 6;
    r.p0 = ld_nc_u2(pp); r.p1 = ld_nc_u2(pp + 2); r.p2 = ld_nc_u2(pp + 4);
    const uint8_t* sp = d.smb + t * 64 + 2 * (j & 7);
#pragma unroll
    for (int q = 0; q < 4; ++q) r.s[q] = ld_nc_u16(sp + 16 * q);
    r.eb = (uint32_t)__ldg(d.escidx + (int64_t)row * (d.K / CH) + (kk / CH));
    r.rp = (uint32_t)__ldg(d.rowparam + row);
    return r;
  }
  static __device__ __forceinline__ uint4 dec(const WDesc& d, const R& r, int, int, int lane, int) {
    const int j = lane & 15;
    const uint32_t cnt = __popc(~(r.p0.x | r.p1.x | r.p2.x)) + __popc(~(r.p0.y | r.p1.y | r.p2.y));
    const uint32_t c0 = __shfl_sync(0xffffffffu, cnt, lane & 16);      // tile 2c of this half
    const uint32_t tb = r.eb + (j >= 8 ? c0 : 0u);
    const uint32_t* e4 = reinterpret_cast<const uint32_t*>(d.esc) + (tb >> 2);
    const uint32_t d0 = ld_nc_u32(e4), d1 = ld_nc_u32(e4 + 1), d2 = ld_nc_u32(e4 + 2);
    const uint32_t sh = (tb & 3u) * 8u;
    const uint32_t wl = __funnelshift_r(d0, d1, sh), wh = __funnelshift_r(d1, d2, sh);
    uint32_t e01, e23;
    miv_tbe::exp_table(r.rp, e01, e23);
    const miv_tbe::LaneC c = miv_tbe::lane_const(j);
    return miv_tbe::decode_vec_v2(r.p0, r.p1, r.p2, r.s, e01, e23, wl, wh, tb, cnt, d.esc, c);
  }
};

// ---------------------------------------------------------------- MMA helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x2(uint32_t (&r)[2], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(r[0]), "=r"(r[1]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---------------------------------------------------------------- the kernel
// grid (ceil(N/64), S, ceil(M/64)); 256 threads; dynamic smem (64 + MT*16) * LDS * 2 bytes.
// MT = m16 tiles per CTA (1..4).  S == 1: y written directly.  S > 1: fp32 partials to
// ws[s][m][n] (row stride N), reduced by bi_reduce_kernel in split order.
template <int FMT, int KQT, int MT>
__global__ void __launch_bounds__(256, 2)
bi_gemm_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx, int M, WDesc d, int S,
               __nv_bfloat16* __restrict__ y, int64_t ldy, float* __restrict__ ws) {
  extern __shared__ __align__(16) unsigned char smraw[];
  __nv_bfloat16* Ws = reinterpret_cast<__nv_bfloat16*>(smraw);
  __nv_bfloat16* Xs = Ws + BN * LDS;
  using L = Ld<FMT, KQT>;
  const int n0 = blockIdx.x * BN, s = blockIdx.y, m0 = blockIdx.z * MTILE;
  const int nch = d.K / CH;
  const int c0 = (int)(((int64_t)s * nch) / S), c1 = (int)(((int64_t)(s + 1) * nch) / S);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int half = lane >> 4, j = lane & 15;
  int rowl[4], rowg[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    rowl[i] = warp * 8 + 2 * i + half;
    rowg[i] = min(n0 + rowl[i], d.N - 1);          // tail rows: duplicate loads, never stored
  }
  float acc[MT][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[mt][e] = 0.0f;

  typename L::R raw[4];
  if (c0 < c1) {
#pragma unroll
    for (int i = 0; i < 4; ++i) raw[i] = L::load(d, rowg[i], c0 * CH + 8 * j, lane);
  }
  for (int c = c0; c < c1; ++c) {
    // activations: rows m0 .. m0 + MT*16, columns c*CH .. c*CH + 127 (rows >= M -> zero)
#pragma unroll
    for (int it = 0; it < MT; ++it) {
      const int i = threadIdx.x + 256 * it;           // MT*16 rows * 16 vectors = MT*256
      const int r = i >> 4, cv = i & 15;
      const int m = m0 + r;
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (m < M) v = __ldg(reinterpret_cast<const uint4*>(x + (int64_t)m * ldx + c * CH + cv * 8));
      *reinterpret_cast<uint4*>(Xs + r * LDS + cv * 8) = v;
    }
    // weights: decode this chunk into smem, then issue the next chunk's loads
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint4 w = L::dec(d, raw[i], rowg[i], c * CH + 8 * j, lane, j);
      *reinterpret_cast<uint4*>(Ws + rowl[i] * LDS + 8 * j) = w;
    }
    if (c + 1 < c1) {
#pragma unroll
      for (int i = 0; i < 4; ++i) raw[i] = L::load(d, rowg[i], (c + 1) * CH + 8 * j, lane);
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < CH; kk += 16) {
      uint32_t b[2];
      ldsm_x2(b, Ws + (warp * 8 + (lane & 7)) * LDS + kk + ((lane >> 3) & 1) * 8);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        uint32_t a[4];
        ldsm_x4(a, Xs + (mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LDS + kk + (lane >> 4) * 8);
        mma16816(acc[mt], a, b);
      }
    }
    __syncthreads();
  }
  // epilogue: c0,c1 -> (row g, cols 2t, 2t+1); c2,c3 -> (row g+8, same cols)
  const int g = lane >> 2, t4 = lane & 3;
  const int n = n0 + warp * 8 + 2 * t4;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int m = m0 + mt * 16 + g + 8 * h;
      if (m >= M) continue;
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int nn = n + e;
        if (nn >= d.N) continue;
        const float v = acc[mt][2 * h + e];
        if (S == 1) {
          const float o = d.bias ? __fadd_rn(v, __bfloat162float(d.bias[nn])) : v;
          y[(int64_t)m * ldy + nn] = __float2bfloat16_rn(o);
        } else {
          ws[((int64_t)s * M + m) * d.N + nn] = v;
        }
      }
    }
  }
}

// S partials added in split order, + bias, one rounding.  grid (ceil(N/256), M), 256 threads.
static __global__ void bi_reduce_kernel(const float* __restrict__ ws, int S, int M, int N,
                                 const __nv_bfloat16* __restrict__ bias,
                                 __nv_bfloat16* __restrict__ y, int64_t ldy) {
  const int n = blockIdx.x * 256 + threadIdx.x, m = blockIdx.y;
  if (n >= N) return;
  float t = ws[(int64_t)m * N + n];
  for (int s = 1; s < S; ++s) t = __fadd_rn(t, ws[((int64_t)s * M + m) * N + n]);
  if (bias) t = __fadd_rn(t, __bfloat162float(bias[n]));
  y[(int64_t)m * ldy + n] = __float2bfloat16_rn(t);
}

inline WDesc to_wdesc(const BiDesc& b) {
  WDesc d{};
  d.N = b.N; d.K = b.K; d.w = b.w; d.qs = b.qs; d.sc = b.sc;
  d.planes = b.planes; d.smb = b.smb; d.esc = b.esc; d.rowparam = b.rowparam; d.escidx = b.escidx;
  d.kq.a0 = b.a0; d.kq.a1 = b.a1; d.kq.a2 = b.a2; d.kq.a3 = b.a3; d.kq.a4 = b.a4;
  d.kq.grid = b.grid; d.kq.ksigns = b.ksigns; d.kq.bias = nullptr; d.kq.N = b.N; d.kq.K = b.K;
  d.bias = b.bias;
  return d;
}

template <int FMT, int KQT>
inline cudaError_t launch_fmt(const __nv_bfloat16* x, int64_t ldx, int M, const BiDesc& bd, int S,
                              __nv_bfloat16* y, int64_t ldy, float* ws, cudaStream_t st) {
  const WDesc d = to_wdesc(bd);
  const int mt = M >= MTILE ? 4 : (M + 15) / 16;
  dim3 grid((d.N + BN - 1) / BN, S, (M + MTILE - 1) / MTILE);
  const size_t smem = (size_t)(BN + mt * 16) * LDS * 2;
  switch (mt) {
    case 1: bi_gemm_kernel<FMT, KQT, 1><<<grid, 256, smem, st>>>(x, ldx, M, d, S, y, ldy, ws); break;
    case 2: bi_gemm_kernel<FMT, KQT, 2><<<grid, 256, smem, st>>>(x, ldx, M, d, S, y, ldy, ws); break;
    case 3: bi_gemm_kernel<FMT, KQT, 3><<<grid, 256, smem, st>>>(x, ldx, M, d, S, y, ldy, ws); break;
    default: bi_gemm_kernel<FMT, KQT, 4><<<grid, 256, smem, st>>>(x, ldx, M, d, S, y, ldy, ws); break;
  }
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess || S == 1) return e;
  bi_reduce_kernel<<<dim3((d.N + 255) / 256, M), 256, 0, st>>>(ws, S, M, d.N, d.bias, y, ldy);
  return cudaGetLastError();
}

}  // namespace bi

