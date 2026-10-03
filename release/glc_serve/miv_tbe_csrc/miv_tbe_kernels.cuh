// MIV-TBE device code.  See miv_tbe.h for the contract.
//
// fma_vec / bf_lo / bf_hi / ld_stream and the epilogue are copied VERBATIM
// from release/glc_serve/miv_gemv.py @ c155643be; they are the whole of the
// arithmetic.  Everything else here only produces the uint4 weight vector
// fma_vec consumes, bit-identical to the parent's bf16 bytes.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "miv_tbe.h"

namespace miv_tbe {

// ------------------------- VERBATIM from miv_gemv.py -------------------------
__device__ __forceinline__ float bf_lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }

// Fold one 16-byte weight vector (8 bf16, columns col..col+7) into acc[0..M).
// Element order 0..7, one __fmaf_rn each: identical for every M.
template <int M>
__device__ __forceinline__ void fma_vec(float (&acc)[M], const uint4 w,
                                        const __nv_bfloat16* __restrict__ x,
                                        int64_t ldx, int64_t col) {
  const float w0 = bf_lo(w.x), w1 = bf_hi(w.x), w2 = bf_lo(w.y), w3 = bf_hi(w.y);
  const float w4 = bf_lo(w.z), w5 = bf_hi(w.z), w6 = bf_lo(w.w), w7 = bf_hi(w.w);
#pragma unroll
  for (int m = 0; m < M; ++m) {
    const uint4 xv = __ldg(reinterpret_cast<const uint4*>(x + m * ldx + col));
    float a = acc[m];
    a = __fmaf_rn(bf_lo(xv.x), w0, a);
    a = __fmaf_rn(bf_hi(xv.x), w1, a);
    a = __fmaf_rn(bf_lo(xv.y), w2, a);
    a = __fmaf_rn(bf_hi(xv.y), w3, a);
    a = __fmaf_rn(bf_lo(xv.z), w4, a);
    a = __fmaf_rn(bf_hi(xv.z), w5, a);
    a = __fmaf_rn(bf_lo(xv.w), w6, a);
    a = __fmaf_rn(bf_hi(xv.w), w7, a);
    acc[m] = a;
  }
}
// -----------------------------------------------------------------------------

struct Args {
  const uint32_t* __restrict__ planes;
  const uint8_t* __restrict__ smb;
  const uint8_t* __restrict__ esc;
  const int32_t* __restrict__ rowparam;
  const int32_t* __restrict__ escbase;
  const __nv_bfloat16* __restrict__ bias;
  int N, K;
};

__device__ __forceinline__ uint2 ld_nc_u2(const void* p) {
  uint2 r;
  asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
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

// Epilogue: fixed xor butterfly, slice partials in slice order, optional bias
// in FP32, one bf16 rounding.  Verbatim from miv_gemv_kernel.
template <int M, int WPR>
__device__ __forceinline__ void epilogue(float (&acc)[M], int warp, int lane, int row,
                                         const Args& a, __nv_bfloat16* __restrict__ y,
                                         int64_t ldy) {
  constexpr int RPC = 8 / WPR;
  const int N = a.N;
  const __nv_bfloat16* bias = a.bias;
#pragma unroll
  for (int m = 0; m < M; ++m) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc[m] = __fadd_rn(acc[m], __shfl_xor_sync(0xffffffffu, acc[m], off));
  }
  if (WPR == 1) {
    if (lane == 0 && row < N) {
      const float b = bias ? __bfloat162float(bias[row]) : 0.0f;
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const float r = bias ? __fadd_rn(acc[m], b) : acc[m];
        y[m * ldy + row] = __float2bfloat16_rn(r);
      }
    }
  } else {
    __shared__ float red[8][M];
    if (lane == 0) {
#pragma unroll
      for (int m = 0; m < M; ++m) red[warp][m] = acc[m];
    }
    __syncthreads();
    if (threadIdx.x < RPC * M) {
      const int r = threadIdx.x / M, m = threadIdx.x % M;
      const int orow = blockIdx.x * RPC + r;
      if (orow < N) {
        float t = red[r * WPR][m];
#pragma unroll
        for (int j = 1; j < WPR; ++j) t = __fadd_rn(t, red[r * WPR + j][m]);
        if (bias) t = __fadd_rn(t, __bfloat162float(bias[orow]));
        y[m * ldy + orow] = __float2bfloat16_rn(t);
      }
    }
  }
}

// =============================================================================
// VERSION 2: PRMT decode.  Per lane per 16-byte vector (8 weights), ~80 integer
// ops, no per-element branch.  Lane l of a warp handles vector v = vg + 32u + l;
// i = l & 7 is the vector's index inside its 64-element tile (v0 is a multiple
// of 8 because K % (64*WPR) == 0), so element r (0..7) of the vector is
// original in-tile element 8i + r, stored (mma16) at position
//     p(r) = 16*(r >> 1) + 2i + (r & 1)
// i.e. bits {2i, 2i+1, 16+2i, 17+2i} of plane word x (r = 0..3) and the same
// bits of plane word y (r = 4..7).
// =============================================================================
struct LaneC {
  uint32_t psel;   // PRMT selector gathering the lane's plane bytes from (x, y)
  uint32_t zs;     // bit offset of the lane's 2-bit pair inside those bytes
  uint32_t lm_a;   // (1 << 2i) - 1
  uint32_t lm_b;   // (1 << (16 + 2i)) - 1
  uint32_t smb_off;
};

__device__ __forceinline__ LaneC lane_const(int lane) {
  const uint32_t i = (uint32_t)lane & 7u, q = i >> 2;
  LaneC c;
  c.psel = q | ((2u + q) << 4) | ((4u + q) << 8) | ((6u + q) << 12);
  c.zs = 2u * (i & 3u);
  c.lm_a = (1u << (2u * i)) - 1u;
  c.lm_b = (1u << (16u + 2u * i)) - 1u;
  c.smb_off = 2u * i;
  return c;
}

// Exponent table E[0..7] as two PRMT operands: E[0] = 0 (escape placeholder),
// E[c] = base + c - 1, W6Z: E[7] = 0 (exact zero).  The encoder guarantees the
// window stays inside a byte, so the byte-wise adds below never carry.
__device__ __forceinline__ void exp_table(uint32_t rp, uint32_t& e01, uint32_t& e23) {
  const uint32_t base = rp & 0xffu;
  const bool w6z = ((rp >> 8) & 0xffu) == 1u;
  e01 = (base * 0x01010101u + 0x03020100u) << 8;          // 0, b, b+1, b+2
  e23 = base * 0x01010101u + 0x06050403u;                 // b+3, b+4, b+5, b+6
  if (w6z) e23 &= 0x00ffffffu;                            // E[7] = 0
}

// The lane's 8 three-bit codes, one per nibble (nibble r = element r).
__device__ __forceinline__ uint32_t codes8(const uint2 p0, const uint2 p1, const uint2 p2,
                                           const LaneC& c) {
  const uint32_t z0 = __byte_perm(p0.x, p0.y, c.psel) >> c.zs;
  const uint32_t z1 = __byte_perm(p1.x, p1.y, c.psel) >> c.zs;
  const uint32_t z2 = __byte_perm(p2.x, p2.y, c.psel) >> c.zs;
  // byte j of z holds elements (2j, 2j+1) at bits 0, 1 -> nibbles 2j, 2j+1
  const uint32_t n0 = (z0 & 0x01010101u) | ((z0 << 3) & 0x10101010u);
  const uint32_t n1 = (z1 & 0x01010101u) | ((z1 << 3) & 0x10101010u);
  const uint32_t n2 = (z2 & 0x01010101u) | ((z2 << 3) & 0x10101010u);
  return n0 | (n1 << 1) | (n2 << 2);
}

// Decode one vector.  tb = index in esc of the first escape of this tile,
// tile_cnt = escapes in this tile, (w_lo, w_hi) = esc[tb .. tb+8).
__device__ __forceinline__ uint4 decode_vec_v2(const uint2 p0, const uint2 p1, const uint2 p2,
                                               const uint32_t (&s)[4], uint32_t e01,
                                               uint32_t e23, uint32_t w_lo, uint32_t w_hi,
                                               uint32_t tb, uint32_t tile_cnt,
                                               const uint8_t* __restrict__ esc,
                                               const LaneC& c) {
  const uint32_t code = codes8(p0, p1, p2, c);
  const uint32_t e_lo = __byte_perm(e01, e23, code);
  const uint32_t e_hi = __byte_perm(e01, e23, code >> 16);
  // escaped elements: code == 0 -> bit 4r of escn
  const uint32_t escn = ~(code | (code >> 1) | (code >> 2)) & 0x11111111u;
  // ranks of the lane's elements among the tile's escapes (stored order)
  const uint32_t m0 = ~(p0.x | p1.x | p2.x), m1 = ~(p0.y | p1.y | p2.y);
  const uint32_t pc0 = __popc(m0);
  const uint32_t r0 = __popc(m0 & c.lm_a), r1 = __popc(m0 & c.lm_b);
  const uint32_t r2 = pc0 + __popc(m1 & c.lm_a), r3 = pc0 + __popc(m1 & c.lm_b);
  uint32_t eb_lo, eb_hi;
  if (tile_cnt <= 8u) {
    // nibble r = rank of element r (garbage >= 8 only for non-escaped ones)
    const uint32_t rs_lo = r0 * 0x0011u + r1 * 0x1100u + ((escn << 4) & 0x1010u);
    const uint32_t rs_hi = r2 * 0x0011u + r3 * 0x1100u + ((escn >> 12) & 0x1010u);
    eb_lo = __byte_perm(w_lo, w_hi, rs_lo);
    eb_hi = __byte_perm(w_lo, w_hi, rs_hi);
  } else {
    // rare (P ~ 1e-4 per tile): a rank can pass the 8-byte window
    const uint32_t rr[4] = {r0, r1, r2, r3};
    eb_lo = 0u;
    eb_hi = 0u;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      if ((escn >> (4 * r)) & 1u) {
        const uint32_t rank = rr[r >> 1] + ((r & 1) ? ((escn >> (4 * (r - 1))) & 1u) : 0u);
        const uint32_t e = (uint32_t)__ldg(esc + tb + rank);
        if (r < 4) eb_lo |= e << (8 * r);
        else eb_hi |= e << (8 * (r - 4));
      }
    }
  }
  // pick table byte (non-escaped) or escape byte (escaped) per element
  const uint32_t x_lo = __byte_perm(e_lo, eb_lo, 0x3210u | ((escn & 0x1111u) << 2));
  const uint32_t x_hi = __byte_perm(e_hi, eb_hi, 0x3210u | ((escn >> 14) & 0x4444u));
  // [S, S, S', S'] & 0x807F807F: both mantissas and both signs in place;
  // {0, E, 0, E'} >> 1: both exponents on bits 7..14 and 23..30.
  uint4 w;
  w.x = (__byte_perm(s[0], 0u, 0x1100u) & 0x807F807Fu) | (__byte_perm(x_lo, 0u, 0x1404u) >> 1);
  w.y = (__byte_perm(s[1], 0u, 0x1100u) & 0x807F807Fu) | (__byte_perm(x_lo, 0u, 0x3424u) >> 1);
  w.z = (__byte_perm(s[2], 0u, 0x1100u) & 0x807F807Fu) | (__byte_perm(x_hi, 0u, 0x1404u) >> 1);
  w.w = (__byte_perm(s[3], 0u, 0x1100u) & 0x807F807Fu) | (__byte_perm(x_hi, 0u, 0x3424u) >> 1);
  return w;
}

// The traversal shared by the fused GEMV and the decode-only kernel: calls
// sink(v, w) for every vector the (row, slice s) warp owns, in increasing v
// per lane -- the MIV lane order.
template <int WPR, int U, class Sink>
__device__ __forceinline__ void walk_v2(const Args& a, int row, int s, int lane, Sink& sink) {
  const int KV = a.K >> 3;
  const int v0 = (int)(((int64_t)s * KV) / WPR);
  const int v1 = (int)(((int64_t)(s + 1) * KV) / WPR);
  const int64_t tile0 = (int64_t)row * (a.K >> 6);
  uint32_t e01, e23;
  exp_table((uint32_t)a.rowparam[row], e01, e23);
  const LaneC c = lane_const(lane);
  const int g = lane >> 3;
  uint32_t ebase = (uint32_t)a.escbase[(int64_t)row * WPR + s];
  for (int vg = v0; vg < v1; vg += 32 * U) {
    uint2 P0[U], P1[U], P2[U];
    uint32_t S[U][4];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int v = vg + 32 * u + lane;
      if (v < v1) {
        const int64_t t = tile0 + (v >> 3);
        const uint32_t* pp = a.planes + t * 6;
        P0[u] = ld_nc_u2(pp);
        P1[u] = ld_nc_u2(pp + 2);
        P2[u] = ld_nc_u2(pp + 4);
        const uint8_t* sp = a.smb + t * 64 + c.smb_off;
#pragma unroll
        for (int q = 0; q < 4; ++q) S[u][q] = ld_nc_u16(sp + 16 * q);
      } else {
        P0[u] = P1[u] = P2[u] = make_uint2(0xffffffffu, 0xffffffffu);   // no escapes
#pragma unroll
        for (int q = 0; q < 4; ++q) S[u][q] = 0u;
      }
    }
    uint32_t TB[U], TC[U], WL[U], WH[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const uint32_t cnt = __popc(~(P0[u].x | P1[u].x | P2[u].x)) +
                           __popc(~(P0[u].y | P1[u].y | P2[u].y));
      const uint32_t c0 = __shfl_sync(0xffffffffu, cnt, 0);
      const uint32_t c1 = __shfl_sync(0xffffffffu, cnt, 8);
      const uint32_t c2 = __shfl_sync(0xffffffffu, cnt, 16);
      const uint32_t c3 = __shfl_sync(0xffffffffu, cnt, 24);
      const uint32_t pre = (g > 0 ? c0 : 0u) + (g > 1 ? c1 : 0u) + (g > 2 ? c2 : 0u);
      TB[u] = ebase + pre;
      TC[u] = cnt;
      ebase += c0 + c1 + c2 + c3;
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int v = vg + 32 * u + lane;
      if (v < v1) {
        // esc[TB .. TB+8): three aligned dwords (esc carries >= 16 pad bytes)
        const uint32_t* e4 = reinterpret_cast<const uint32_t*>(a.esc) + (TB[u] >> 2);
        const uint32_t d0 = ld_nc_u32(e4), d1 = ld_nc_u32(e4 + 1), d2 = ld_nc_u32(e4 + 2);
        const uint32_t sh = (TB[u] & 3u) * 8u;
        WL[u] = __funnelshift_r(d0, d1, sh);
        WH[u] = __funnelshift_r(d1, d2, sh);
      } else {
        WL[u] = WH[u] = 0u;
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int v = vg + 32 * u + lane;
      if (v < v1) {
        const uint4 w = decode_vec_v2(P0[u], P1[u], P2[u], S[u], e01, e23, WL[u], WH[u], TB[u],
                                      TC[u], a.esc, c);
        sink(v, w);
      }
    }
  }
}

// VERSION 3: v2's decode with the coded stream software-pipelined one group
// ahead: the planes/smb loads of group g+1 are in flight while group g's
// escape window loads and decode run.  Same lane -> vector order, same
// decode function, so the same bits as v2 (and as bf16 MIV).
template <int U>
__device__ __forceinline__ void load_group(const Args& a, int64_t tile0, int vg, int v1, int lane,
                                           const LaneC& c, uint2 (&P0)[U], uint2 (&P1)[U],
                                           uint2 (&P2)[U], uint32_t (&S)[U][4]) {
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const int v = vg + 32 * u + lane;
    if (v < v1) {
      const int64_t t = tile0 + (v >> 3);
      const uint32_t* pp = a.planes + t * 6;
      P0[u] = ld_nc_u2(pp);
      P1[u] = ld_nc_u2(pp + 2);
      P2[u] = ld_nc_u2(pp + 4);
      const uint8_t* sp = a.smb + t * 64 + c.smb_off;
#pragma unroll
      for (int q = 0; q < 4; ++q) S[u][q] = ld_nc_u16(sp + 16 * q);
    } else {
      P0[u] = P1[u] = P2[u] = make_uint2(0xffffffffu, 0xffffffffu);
#pragma unroll
      for (int q = 0; q < 4; ++q) S[u][q] = 0u;
    }
  }
}

template <int WPR, int U, class Sink>
__device__ __forceinline__ void walk_v3(const Args& a, int row, int s, int lane, Sink& sink) {
  const int KV = a.K >> 3;
  const int v0 = (int)(((int64_t)s * KV) / WPR);
  const int v1 = (int)(((int64_t)(s + 1) * KV) / WPR);
  const int64_t tile0 = (int64_t)row * (a.K >> 6);
  uint32_t e01, e23;
  exp_table((uint32_t)a.rowparam[row], e01, e23);
  const LaneC c = lane_const(lane);
  const int g = lane >> 3;
  uint32_t ebase = (uint32_t)a.escbase[(int64_t)row * WPR + s];
  uint2 P0[U], P1[U], P2[U];
  uint32_t S[U][4];
  load_group<U>(a, tile0, v0, v1, lane, c, P0, P1, P2, S);
  for (int vg = v0; vg < v1; vg += 32 * U) {
    uint32_t TB[U], TC[U], WL[U], WH[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const uint32_t cnt = __popc(~(P0[u].x | P1[u].x | P2[u].x)) +
                           __popc(~(P0[u].y | P1[u].y | P2[u].y));
      const uint32_t c0 = __shfl_sync(0xffffffffu, cnt, 0);
      const uint32_t c1 = __shfl_sync(0xffffffffu, cnt, 8);
      const uint32_t c2 = __shfl_sync(0xffffffffu, cnt, 16);
      const uint32_t c3 = __shfl_sync(0xffffffffu, cnt, 24);
      const uint32_t pre = (g > 0 ? c0 : 0u) + (g > 1 ? c1 : 0u) + (g > 2 ? c2 : 0u);
      TB[u] = ebase + pre;
      TC[u] = cnt;
      ebase += c0 + c1 + c2 + c3;
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int v = vg + 32 * u + lane;
      if (v < v1) {
        const uint32_t* e4 = reinterpret_cast<const uint32_t*>(a.esc) + (TB[u] >> 2);
        const uint32_t d0 = ld_nc_u32(e4), d1 = ld_nc_u32(e4 + 1), d2 = ld_nc_u32(e4 + 2);
        const uint32_t sh = (TB[u] & 3u) * 8u;
        WL[u] = __funnelshift_r(d0, d1, sh);
        WH[u] = __funnelshift_r(d1, d2, sh);
      } else {
        WL[u] = WH[u] = 0u;
      }
    }
    uint2 Q0[U], Q1[U], Q2[U];
    uint32_t T[U][4];
    load_group<U>(a, tile0, vg + 32 * U, v1, lane, c, Q0, Q1, Q2, T);
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int v = vg + 32 * u + lane;
      if (v < v1) {
        const uint4 w = decode_vec_v2(P0[u], P1[u], P2[u], S[u], e01, e23, WL[u], WH[u], TB[u],
                                      TC[u], a.esc, c);
        sink(v, w);
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      P0[u] = Q0[u]; P1[u] = Q1[u]; P2[u] = Q2[u];
#pragma unroll
      for (int q = 0; q < 4; ++q) S[u][q] = T[u][q];
    }
  }
}

template <int M>
struct FmaSink {
  float (&acc)[M];
  const __nv_bfloat16* __restrict__ x;
  int64_t ldx;
  __device__ __forceinline__ void operator()(int v, const uint4 w) {
    fma_vec<M>(acc, w, x, ldx, (int64_t)v * 8);
  }
};

struct StoreSink {
  uint4* __restrict__ out;   // row base, as uint4 (8 bf16) vectors
  __device__ __forceinline__ void operator()(int v, const uint4 w) { out[v] = w; }
};

template <int M, int WPR, int U>
__global__ void __launch_bounds__(256)
miv_tbe_kernel_v2(const __nv_bfloat16* __restrict__ x, int64_t ldx, Args a,
                  __nv_bfloat16* __restrict__ y, int64_t ldy) {
  constexpr int RPC = 8 / WPR;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int rloc = warp / WPR, s = warp % WPR;
  const int row = blockIdx.x * RPC + rloc;
  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.0f;
  if (row < a.N) {
    FmaSink<M> sink{acc, x, ldx};
    walk_v2<WPR, U>(a, row, s, lane, sink);
  }
  epilogue<M, WPR>(acc, warp, lane, row, a, y, ldy);
}

template <int M, int WPR, int U>
__global__ void __launch_bounds__(256)
miv_tbe_kernel_v3(const __nv_bfloat16* __restrict__ x, int64_t ldx, Args a,
                  __nv_bfloat16* __restrict__ y, int64_t ldy) {
  constexpr int RPC = 8 / WPR;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int rloc = warp / WPR, s = warp % WPR;
  const int row = blockIdx.x * RPC + rloc;
  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.0f;
  if (row < a.N) {
    FmaSink<M> sink{acc, x, ldx};
    walk_v3<WPR, U>(a, row, s, lane, sink);
  }
  epilogue<M, WPR>(acc, warp, lane, row, a, y, ldy);
}

template <int WPR, int U>
__global__ void __launch_bounds__(256)
miv_tbe_decode_kernel_v2(Args a, __nv_bfloat16* __restrict__ w_out) {
  constexpr int RPC = 8 / WPR;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int rloc = warp / WPR, s = warp % WPR;
  const int row = blockIdx.x * RPC + rloc;
  if (row < a.N) {
    StoreSink sink{reinterpret_cast<uint4*>(w_out + (int64_t)row * a.K)};
    walk_v2<WPR, U>(a, row, s, lane, sink);
  }
}

// =============================================================================
// VERSION 1: the engine agent's untested draft (miv_gemv.py working tree on
// feat/miv-gemv-20260924, 2026-09-24), kept as the scalar reference decode.
// Per element: code from three plane bits, branch on code == 0 with a
// dependent esc load.  Bias added to its epilogue (shared epilogue above).
// =============================================================================
template <int M, int WPR, int U>
__global__ void __launch_bounds__(256)
miv_tbe_kernel_v1(const __nv_bfloat16* __restrict__ x, int64_t ldx, Args A,
                  __nv_bfloat16* __restrict__ y, int64_t ldy) {
  constexpr int RPC = 8 / WPR;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int rloc = warp / WPR, s = warp % WPR;
  const int row = blockIdx.x * RPC + rloc;
  const int N = A.N, K = A.K;
  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.0f;
  if (row < N) {
    const int KV = K >> 3;
    const int v0 = (int)(((int64_t)s * KV) / WPR);
    const int v1 = (int)(((int64_t)(s + 1) * KV) / WPR);
    const int64_t tile0 = (int64_t)row * (K >> 6);
    const int rp = A.rowparam[row];
    const uint32_t ebase_win = (uint32_t)(rp & 0xff);
    const bool w6z = (rp >> 8) == 1;
    int ebase = A.escbase[(int64_t)row * WPR + s];
    const int i = lane & 7, g = lane >> 3;
    for (int vg = v0; vg < v1; vg += 32 * U) {
      uint2 P0[U], P1[U], P2[U];
      uint32_t S[U][4];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int v = vg + 32 * u + lane;
        if (v < v1) {
          const int64_t t = tile0 + (v >> 3);
          const uint32_t* pp = A.planes + t * 6;
          P0[u] = ld_nc_u2(pp); P1[u] = ld_nc_u2(pp + 2); P2[u] = ld_nc_u2(pp + 4);
          const uint8_t* sp = A.smb + t * 64 + 2 * i;
#pragma unroll
          for (int q = 0; q < 4; ++q) S[u][q] = ld_nc_u16(sp + 16 * q);
        } else {
          P0[u] = P1[u] = P2[u] = make_uint2(0xffffffffu, 0xffffffffu);
#pragma unroll
          for (int q = 0; q < 4; ++q) S[u][q] = 0;
        }
      }
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int v = vg + 32 * u + lane;
        const uint32_t m0 = ~(P0[u].x | P1[u].x | P2[u].x);
        const uint32_t m1 = ~(P0[u].y | P1[u].y | P2[u].y);
        const int cnt = __popc(m0) + __popc(m1);
        const int c0 = __shfl_sync(0xffffffffu, cnt, 0);
        const int c1 = __shfl_sync(0xffffffffu, cnt, 8);
        const int c2 = __shfl_sync(0xffffffffu, cnt, 16);
        const int c3 = __shfl_sync(0xffffffffu, cnt, 24);
        const int pre = (g > 0 ? c0 : 0) + (g > 1 ? c1 : 0) + (g > 2 ? c2 : 0);
        if (v < v1) {
          uint32_t hw[8];
#pragma unroll
          for (int r = 0; r < 8; ++r) {
            const int wsel = r >> 2;
            const int bit = 16 * ((r >> 1) & 1) + 2 * i + (r & 1);
            const uint32_t a0 = wsel ? P0[u].y : P0[u].x;
            const uint32_t a1 = wsel ? P1[u].y : P1[u].x;
            const uint32_t a2 = wsel ? P2[u].y : P2[u].x;
            const uint32_t code = ((a0 >> bit) & 1u) | (((a1 >> bit) & 1u) << 1) |
                                  (((a2 >> bit) & 1u) << 2);
            uint32_t e;
            if (code == 0u) {
              const int rank = wsel ? (__popc(m0) + __popc(m1 & ((1u << bit) - 1u)))
                                    : __popc(m0 & ((1u << bit) - 1u));
              e = (uint32_t)__ldg(A.esc + ebase + pre + rank);
            } else if (w6z && code == 7u) {
              e = 0u;
            } else {
              e = code - 1u + ebase_win;
            }
            const uint32_t sm = (S[u][r >> 1] >> (8 * (r & 1))) & 0xffu;
            hw[r] = ((sm & 0x80u) << 8) | ((e & 0xffu) << 7) | (sm & 0x7fu);
          }
          uint4 w;
          w.x = hw[0] | (hw[1] << 16); w.y = hw[2] | (hw[3] << 16);
          w.z = hw[4] | (hw[5] << 16); w.w = hw[6] | (hw[7] << 16);
          fma_vec<M>(acc, w, x, ldx, (int64_t)v * 8);
        }
        ebase += c0 + c1 + c2 + c3;
      }
    }
  }
  epilogue<M, WPR>(acc, warp, lane, row, A, y, ldy);
}

// ------------------------------- host dispatch -------------------------------
inline Args make_args(const MivTbeDesc& d) {
  Args a;
  a.planes = d.planes; a.smb = d.smb; a.esc = d.esc; a.rowparam = d.rowparam;
  a.escbase = d.escbase; a.bias = d.bias; a.N = d.n; a.K = d.k;
  return a;
}

template <int M, int WPR, int U>
inline cudaError_t launch_mwu(const MivTbeDesc& d, const __nv_bfloat16* x, int64_t ldx,
                              __nv_bfloat16* y, int64_t ldy, cudaStream_t st) {
  constexpr int RPC = 8 / WPR;
  const int grid = (d.n + RPC - 1) / RPC;
  const Args a = make_args(d);
  if (d.version == 1)
    miv_tbe_kernel_v1<M, WPR, U><<<grid, 256, 0, st>>>(x, ldx, a, y, ldy);
  else if (d.version == 3)
    miv_tbe_kernel_v3<M, WPR, U><<<grid, 256, 0, st>>>(x, ldx, a, y, ldy);
  else
    miv_tbe_kernel_v2<M, WPR, U><<<grid, 256, 0, st>>>(x, ldx, a, y, ldy);
  return cudaGetLastError();
}

template <int M, int WPR>
inline cudaError_t launch_mw(const MivTbeDesc& d, const __nv_bfloat16* x, int64_t ldx,
                             __nv_bfloat16* y, int64_t ldy, cudaStream_t st) {
  switch (d.unroll) {
    case 1: return launch_mwu<M, WPR, 1>(d, x, ldx, y, ldy, st);
    case 2: return launch_mwu<M, WPR, 2>(d, x, ldx, y, ldy, st);
    case 4: return launch_mwu<M, WPR, 4>(d, x, ldx, y, ldy, st);
    default: return cudaErrorInvalidValue;
  }
}

template <int M>
inline cudaError_t launch_m(const MivTbeDesc& d, const __nv_bfloat16* x, int64_t ldx,
                            __nv_bfloat16* y, int64_t ldy, cudaStream_t st) {
  switch (d.wpr) {
    case 1: return launch_mw<M, 1>(d, x, ldx, y, ldy, st);
    case 2: return launch_mw<M, 2>(d, x, ldx, y, ldy, st);
    case 4: return launch_mw<M, 4>(d, x, ldx, y, ldy, st);
    case 8: return launch_mw<M, 8>(d, x, ldx, y, ldy, st);
    default: return cudaErrorInvalidValue;
  }
}

}  // namespace miv_tbe

// Per-M entry points, one translation unit each (parallel build).
cudaError_t miv_tbe_gemv_m1(const MivTbeDesc&, const __nv_bfloat16*, int64_t, __nv_bfloat16*, int64_t, cudaStream_t);
cudaError_t miv_tbe_gemv_m2(const MivTbeDesc&, const __nv_bfloat16*, int64_t, __nv_bfloat16*, int64_t, cudaStream_t);
cudaError_t miv_tbe_gemv_m3(const MivTbeDesc&, const __nv_bfloat16*, int64_t, __nv_bfloat16*, int64_t, cudaStream_t);
cudaError_t miv_tbe_gemv_m4(const MivTbeDesc&, const __nv_bfloat16*, int64_t, __nv_bfloat16*, int64_t, cudaStream_t);
cudaError_t miv_tbe_gemv_m5(const MivTbeDesc&, const __nv_bfloat16*, int64_t, __nv_bfloat16*, int64_t, cudaStream_t);
cudaError_t miv_tbe_gemv_m6(const MivTbeDesc&, const __nv_bfloat16*, int64_t, __nv_bfloat16*, int64_t, cudaStream_t);
cudaError_t miv_tbe_gemv_m7(const MivTbeDesc&, const __nv_bfloat16*, int64_t, __nv_bfloat16*, int64_t, cudaStream_t);
cudaError_t miv_tbe_gemv_m8(const MivTbeDesc&, const __nv_bfloat16*, int64_t, __nv_bfloat16*, int64_t, cudaStream_t);

#define MIV_TBE_DEFINE_M(MM)                                                              \
  cudaError_t miv_tbe_gemv_m##MM(const MivTbeDesc& d, const __nv_bfloat16* x, int64_t ldx, \
                                 __nv_bfloat16* y, int64_t ldy, cudaStream_t st) {         \
    return miv_tbe::launch_m<MM>(d, x, ldx, y, ldy, st);                                   \
  }
