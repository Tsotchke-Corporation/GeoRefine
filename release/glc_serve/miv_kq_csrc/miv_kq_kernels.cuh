// MIV-KQ: M-invariant skinny GEMM (M = 1..8) over llama.cpp K-quant / i-quant
// weights, dequantized in registers.  Contract: miv_kq.h.
//
// The arithmetic is the verbatim bf16 MIV chain (fma_vec, butterfly, slice sum,
// one rounding; miv_gemv.py @ c155643be).  The weight vector fed to fma_vec is
// bf16_rn(v) where v is the FP32 value gguf-py's reference ``dequantize``
// computes, reproduced operation by operation (explicit __fmul_rn/__fsub_rn;
// no contraction).  So for the same WPR the output is bitwise bf16 MIV run on
// torch.from_numpy(gguf.quants.dequantize(...)).to(torch.bfloat16).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "miv_kq.h"

namespace miv_kq {

// ------------------------- VERBATIM from miv_gemv.py -------------------------
__device__ __forceinline__ float bf_lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }

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

template <int M, int WPR>
__device__ __forceinline__ void epilogue(float (&acc)[M], int warp, int lane, int row, int N,
                                         const __nv_bfloat16* __restrict__ bias,
                                         __nv_bfloat16* __restrict__ y, int64_t ldy) {
  constexpr int RPC = 8 / WPR;
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
// -----------------------------------------------------------------------------

__device__ __forceinline__ uint2 ld_u2(const void* p) {
  uint2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t ld_u32(const void* p) {
  uint32_t r;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t ld_u16(const void* p) {
  unsigned short r;
  asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(r) : "l"(p));
  return (uint32_t)r;
}
__device__ __forceinline__ uint32_t ld_u8(const void* p) {
  unsigned short r;
  asm volatile("ld.global.nc.u8 %0, [%1];" : "=h"(r) : "l"(p));
  return (uint32_t)r & 0xffu;
}
__device__ __forceinline__ float h2f(uint32_t h) {
  return __half2float(__ushort_as_half((unsigned short)(h & 0xffffu)));
}
__device__ __forceinline__ uint32_t byte_of(uint2 v, int r) {
  return ((r < 4 ? v.x : v.y) >> (8 * (r & 3))) & 0xffu;
}
__device__ __forceinline__ uint32_t bfbits(float f) {
  return (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(f));
}

// Small non-negative integers to FP32 without I2F: bits 0x4B4000XX are exactly
// 12582912 + XX, so PRMT byte k of w into byte 0 of the magic constant and
// subtract (exact for every value < 2^22).  Same value as (float)q, bit for bit.
__device__ __forceinline__ float mbyte(uint32_t w, int k, float bias) {
  return __fadd_rn(__int_as_float(__byte_perm(w, 0x4B400000u, 0x7650u + (uint32_t)k)), bias);
}
constexpr float kMagic = 12582912.0f;

struct Args {
  const uint8_t* __restrict__ a0;
  const uint8_t* __restrict__ a1;
  const uint8_t* __restrict__ a2;
  const uint8_t* __restrict__ a3;
  const uint8_t* __restrict__ a4;
  const uint32_t* __restrict__ grid;    // IQ3 grids (4 values per entry, one byte each)
  const uint8_t* __restrict__ ksigns;   // IQ3_XXS sign table (128 bytes)
  const __nv_bfloat16* __restrict__ bias;
  int N, K;
};

// Raw bytes one lane needs for one 8-element vector; filled by load<T>, used by deq<T>.
struct Raw {
  uint2 q;
  uint2 h;
  uint32_t s0, s1, s2, d;
};

// ============================ loads (per type) ================================
// b = super-block index (row * K/256 + col/256), o = element offset in the
// 256-element super-block (a multiple of 8).
template <int T>
__device__ __forceinline__ Raw load(const Args& a, int64_t b, int o);

// Q4_K  a0 = {fp16 d, fp16 dmin} u32 [nb]   a1 = scales u8 [nb*12]   a2 = qs u8 [nb*128]
template <>
__device__ __forceinline__ Raw load<KQ_Q4_K>(const Args& a, int64_t b, int o) {
  Raw r;
  r.q = ld_u2(a.a2 + b * 128 + 32 * (o >> 6) + (o & 31));
  const uint8_t* s = a.a1 + b * 12;
  r.s0 = ld_u32(s); r.s1 = ld_u32(s + 4); r.s2 = ld_u32(s + 8);
  r.d = ld_u32(a.a0 + b * 4);
  r.h = make_uint2(0, 0);
  return r;
}
// Q5_K  a0 = dm u32 [nb]   a1 = scales [nb*12]   a2 = qh [nb*32]   a3 = qs [nb*128]
template <>
__device__ __forceinline__ Raw load<KQ_Q5_K>(const Args& a, int64_t b, int o) {
  Raw r;
  r.q = ld_u2(a.a3 + b * 128 + 32 * (o >> 6) + (o & 31));
  r.h = ld_u2(a.a2 + b * 32 + (o & 31));
  const uint8_t* s = a.a1 + b * 12;
  r.s0 = ld_u32(s); r.s1 = ld_u32(s + 4); r.s2 = ld_u32(s + 8);
  r.d = ld_u32(a.a0 + b * 4);
  return r;
}
// Q6_K  a0 = ql [nb*128]   a1 = qh [nb*64]   a2 = scales i8 [nb*16]   a3 = d fp16 [nb]
template <>
__device__ __forceinline__ Raw load<KQ_Q6_K>(const Args& a, int64_t b, int o) {
  Raw r;
  r.q = ld_u2(a.a0 + b * 128 + 64 * (o >> 7) + (o & 63));
  r.h = ld_u2(a.a1 + b * 64 + 32 * (o >> 7) + (o & 31));
  r.s0 = ld_u8(a.a2 + b * 16 + (o >> 4));
  r.d = ld_u16(a.a3 + b * 2);
  r.s1 = r.s2 = 0;
  return r;
}
// IQ4_XS  a0 = d fp16 [nb]   a1 = scales_h u16 [nb]   a2 = scales_l [nb*4]   a3 = qs [nb*128]
template <>
__device__ __forceinline__ Raw load<KQ_IQ4_XS>(const Args& a, int64_t b, int o) {
  Raw r;
  r.q = ld_u2(a.a3 + b * 128 + 16 * (o >> 5) + (o & 15));
  r.s0 = ld_u8(a.a2 + b * 4 + (o >> 6));
  r.s1 = ld_u16(a.a1 + b * 2);
  r.d = ld_u16(a.a0 + b * 2);
  r.h = make_uint2(0, 0);
  r.s2 = 0;
  return r;
}
// IQ3_XXS  a0 = d [nb]   a1 = qs (grid idx) [nb*64]   a2 = scales/signs u32 [nb*8]
template <>
__device__ __forceinline__ Raw load<KQ_IQ3_XXS>(const Args& a, int64_t b, int o) {
  Raw r;
  r.q = make_uint2(ld_u16(a.a1 + b * 64 + (o >> 2)), 0);
  r.s0 = ld_u32(a.a2 + (b * 8 + (o >> 5)) * 4);
  r.d = ld_u16(a.a0 + b * 2);
  r.h = make_uint2(0, 0);
  r.s1 = r.s2 = 0;
  return r;
}
// IQ3_S  a0 = d [nb]   a1 = qs [nb*64]   a2 = qh [nb*8]   a3 = signs [nb*32]   a4 = scales [nb*4]
template <>
__device__ __forceinline__ Raw load<KQ_IQ3_S>(const Args& a, int64_t b, int o) {
  Raw r;
  r.q = make_uint2(ld_u16(a.a1 + b * 64 + (o >> 2)), 0);
  r.s0 = ld_u8(a.a2 + b * 8 + (o >> 5));
  r.s1 = ld_u8(a.a3 + b * 32 + (o >> 3));
  r.s2 = ld_u8(a.a4 + b * 4 + (o >> 6));
  r.d = ld_u16(a.a0 + b * 2);
  r.h = make_uint2(0, 0);
  return r;
}

// Q2_K  a0 = scales [nb*16]   a1 = qs [nb*64]   a2 = {d, dmin} u32 [nb]
template <>
__device__ __forceinline__ Raw load<KQ_Q2_K>(const Args& a, int64_t b, int o) {
  Raw r;
  r.q = ld_u2(a.a1 + b * 64 + 32 * (o >> 7) + (o & 31));
  r.s0 = ld_u8(a.a0 + b * 16 + (o >> 4));
  r.d = ld_u32(a.a2 + b * 4);
  r.h = make_uint2(0, 0);
  r.s1 = r.s2 = 0;
  return r;
}
// Q3_K  a0 = hmask [nb*32]   a1 = qs [nb*64]   a2 = scales [nb*12]   a3 = d fp16 [nb]
template <>
__device__ __forceinline__ Raw load<KQ_Q3_K>(const Args& a, int64_t b, int o) {
  Raw r;
  r.q = ld_u2(a.a1 + b * 64 + 32 * (o >> 7) + (o & 31));
  r.h = ld_u2(a.a0 + b * 32 + (o & 31));
  const uint8_t* s = a.a2 + b * 12;
  r.s0 = ld_u32(s); r.s1 = ld_u32(s + 4); r.s2 = ld_u32(s + 8);
  r.d = ld_u16(a.a3 + b * 2);
  return r;
}
// IQ2_XS  a0 = d [nb]   a1 = qs u16 [nb*32]   a2 = scales [nb*8]
template <>
__device__ __forceinline__ Raw load<KQ_IQ2_XS>(const Args& a, int64_t b, int o) {
  Raw r;
  r.q = make_uint2(ld_u16(a.a1 + b * 64 + 2 * (o >> 3)), 0);
  r.s0 = ld_u8(a.a2 + b * 8 + (o >> 5));
  r.d = ld_u16(a.a0 + b * 2);
  r.h = make_uint2(0, 0);
  r.s1 = r.s2 = 0;
  return r;
}
// IQ2_S  a0 = d [nb]   a1 = qs [nb*32]   a2 = signs [nb*32]   a3 = qh [nb*8]   a4 = scales [nb*8]
template <>
__device__ __forceinline__ Raw load<KQ_IQ2_S>(const Args& a, int64_t b, int o) {
  Raw r;
  const int g = o >> 3;
  r.q = make_uint2(ld_u8(a.a1 + b * 32 + g), 0);
  r.s1 = ld_u8(a.a2 + b * 32 + g);
  r.s0 = ld_u8(a.a3 + b * 8 + (g >> 2));
  r.s2 = ld_u8(a.a4 + b * 8 + (o >> 5));
  r.d = ld_u16(a.a0 + b * 2);
  r.h = make_uint2(0, 0);
  return r;
}

// ======================= dequant (per type, gguf-py order) ====================
template <int T>
__device__ __forceinline__ void deq(const Args& a, const Raw& r, int o, float (&v)[8]);

__device__ __forceinline__ uint32_t sbyte(const Raw& r, int w, int i) {
  const uint32_t word = w == 0 ? r.s0 : (w == 1 ? r.s1 : r.s2);
  return (word >> (8 * i)) & 0xffu;
}
// Q4_K.get_scale_min for sub-block j
__device__ __forceinline__ void scale_min(const Raw& r, int j, uint32_t& sc, uint32_t& m) {
  if (j < 4) {
    sc = sbyte(r, 0, j) & 63u;
    m = sbyte(r, 1, j) & 63u;
  } else {
    const int i = j - 4;
    sc = (sbyte(r, 2, i) & 15u) | ((sbyte(r, 0, i) >> 6) << 4);
    m = (sbyte(r, 2, i) >> 4) | ((sbyte(r, 1, i) >> 6) << 4);
  }
}

template <>
__device__ __forceinline__ void deq<KQ_Q4_K>(const Args&, const Raw& r, int o, float (&v)[8]) {
  uint32_t sc, m;
  scale_min(r, o >> 5, sc, m);
  const float d = __fmul_rn(h2f(r.d), (float)sc);           // d * sc
  const float dm = __fmul_rn(h2f(r.d >> 16), (float)m);     // dmin * m
  const int sh = (o & 32) ? 4 : 0;
  const uint32_t w[2] = {(r.q.x >> sh) & 0x0F0F0F0Fu, (r.q.y >> sh) & 0x0F0F0F0Fu};
#pragma unroll
  for (int e = 0; e < 8; ++e)
    v[e] = __fsub_rn(__fmul_rn(d, mbyte(w[e >> 2], e & 3, -kMagic)), dm);   // d * q - dm
}
template <>
__device__ __forceinline__ void deq<KQ_Q5_K>(const Args&, const Raw& r, int o, float (&v)[8]) {
  uint32_t sc, m;
  const int j = o >> 5;
  scale_min(r, j, sc, m);
  const float d = __fmul_rn(h2f(r.d), (float)sc);
  const float dm = __fmul_rn(h2f(r.d >> 16), (float)m);
  const int sh = (o & 32) ? 4 : 0;
  const uint32_t w[2] = {((r.q.x >> sh) & 0x0F0F0F0Fu) | (((r.h.x >> j) & 0x01010101u) << 4),
                         ((r.q.y >> sh) & 0x0F0F0F0Fu) | (((r.h.y >> j) & 0x01010101u) << 4)};
#pragma unroll
  for (int e = 0; e < 8; ++e)
    v[e] = __fsub_rn(__fmul_rn(d, mbyte(w[e >> 2], e & 3, -kMagic)), dm);
}
template <>
__device__ __forceinline__ void deq<KQ_Q6_K>(const Args&, const Raw& r, int o, float (&v)[8]) {
  const float d = __fmul_rn(h2f(r.d), (float)(int8_t)(r.s0 & 0xffu));   // d * scales
  const int qs = ((o >> 6) & 1) * 4, hs = ((o >> 5) & 3) * 2;
  const uint32_t w[2] = {((r.q.x >> qs) & 0x0F0F0F0Fu) | (((r.h.x >> hs) & 0x03030303u) << 4),
                         ((r.q.y >> qs) & 0x0F0F0F0Fu) | (((r.h.y >> hs) & 0x03030303u) << 4)};
#pragma unroll
  for (int e = 0; e < 8; ++e)
    v[e] = __fmul_rn(d, mbyte(w[e >> 2], e & 3, -(kMagic + 32.0f)));    // q - 32, exact
}
template <>
__device__ __forceinline__ void deq<KQ_IQ4_XS>(const Args&, const Raw& r, int o, float (&v)[8]) {
  const int ib = o >> 5;
  const uint32_t lo = (r.s0 >> (4 * (ib & 1))) & 15u;
  const uint32_t hi = ((r.s1 >> (2 * ib)) & 0xffu) & 3u;
  const int sc = (int)(int8_t)(uint8_t)(lo | (hi << 4)) - 32;
  const float dl = __fmul_rn(h2f(r.d), (float)sc);
  const int sh = (o & 16) ? 4 : 0;
  // IQ4_NL.kvalues + 128 as unsigned bytes: entries 0..7 in (K0, K1), 8..15 in (K2, K3)
  const uint32_t K0 = 0x3F2D1801u, K1 = 0x766A5D4Fu, K2 = 0xA6998D81u, K3 = 0xF1D9C5B5u;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint32_t nib = ((h ? r.q.y : r.q.x) >> sh) & 0x0F0F0F0Fu;         // 4 codes, one per byte
    const uint32_t t = nib | (nib >> 4);
    const uint32_t sel = __byte_perm(t, 0u, 0x4420u) & 0x7777u;             // 4 selector nibbles
    const uint32_t m = ((nib >> 3) & 0x01010101u) * 0xFFu;                   // code >= 8
    const uint32_t kv = (__byte_perm(K0, K1, sel) & ~m) | (__byte_perm(K2, K3, sel) & m);
#pragma unroll
    for (int e = 0; e < 4; ++e)
      v[4 * h + e] = __fmul_rn(dl, mbyte(kv, e, -(kMagic + 128.0f)));       // kvalue, exact
  }
}
template <>
__device__ __forceinline__ void deq<KQ_IQ3_XXS>(const Args& a, const Raw& r, int o,
                                                 float (&v)[8]) {
  const float db = __fmul_rn(__fmul_rn(h2f(r.d), __fadd_rn(0.5f, (float)(r.s0 >> 28))), 0.5f);
  const uint32_t sg = __ldg(a.ksigns + ((r.s0 >> (7 * ((o >> 3) & 3))) & 127u));
  const uint32_t g0 = __ldg(a.grid + (r.q.x & 0xffu));
  const uint32_t g1 = __ldg(a.grid + ((r.q.x >> 8) & 0xffu));
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const float s = ((sg >> e) & 1u) ? -1.0f : 1.0f;
    v[e] = __fmul_rn(__fmul_rn(db, mbyte(e < 4 ? g0 : g1, e & 3, -kMagic)), s);
  }
}
template <>
__device__ __forceinline__ void deq<KQ_IQ3_S>(const Args& a, const Raw& r, int o,
                                               float (&v)[8]) {
  const int ib = o >> 5;
  const uint32_t sc = (r.s2 >> (4 * (ib & 1))) & 15u;
  const float db = __fmul_rn(h2f(r.d), (float)(1u + 2u * sc));
  const int gi = (o >> 2) & 7;                                // group index within the qh byte
  const uint32_t i0 = (r.q.x & 0xffu) | (((r.s0 >> gi) & 1u) << 8);
  const uint32_t i1 = ((r.q.x >> 8) & 0xffu) | (((r.s0 >> (gi + 1)) & 1u) << 8);
  const uint32_t g0 = __ldg(a.grid + i0), g1 = __ldg(a.grid + i1);
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const float s = ((r.s1 >> e) & 1u) ? -1.0f : 1.0f;
    v[e] = __fmul_rn(__fmul_rn(db, mbyte(e < 4 ? g0 : g1, e & 3, -kMagic)), s);
  }
}

template <>
__device__ __forceinline__ void deq<KQ_Q2_K>(const Args&, const Raw& r, int o, float (&v)[8]) {
  const float dl = __fmul_rn(h2f(r.d), (float)(r.s0 & 15u));          // d * (sc & 0xF)
  const float ml = __fmul_rn(h2f(r.d >> 16), (float)(r.s0 >> 4));     // dmin * (sc >> 4)
  const int sh = 2 * ((o >> 5) & 3);
  const uint32_t w[2] = {(r.q.x >> sh) & 0x03030303u, (r.q.y >> sh) & 0x03030303u};
#pragma unroll
  for (int e = 0; e < 8; ++e)
    v[e] = __fsub_rn(__fmul_rn(dl, mbyte(w[e >> 2], e & 3, -kMagic)), ml);   // dl * q - ml
}
template <>
__device__ __forceinline__ void deq<KQ_Q3_K>(const Args&, const Raw& r, int o, float (&v)[8]) {
  const int j = o >> 4;                                               // 16 sub-blocks of 16
  const uint32_t l = j < 8 ? (sbyte(r, j >> 2, j & 3) & 15u) : (sbyte(r, (j - 8) >> 2, (j - 8) & 3) >> 4);
  const uint32_t hs = (sbyte(r, 2, j & 3) >> (2 * (j >> 2))) & 3u;
  const int sc = (int)(int8_t)(uint8_t)(l | (hs << 4)) - 32;
  const float dl = __fmul_rn(h2f(r.d), (float)sc);
  const int sh = 2 * ((o >> 5) & 3), hb = o >> 5;
  // ql - 4 * (1 - h)  ==  (ql + 4h) - 4, with ql + 4h in 0..7
  const uint32_t w[2] = {((r.q.x >> sh) & 0x03030303u) | (((r.h.x >> hb) & 0x01010101u) << 2),
                         ((r.q.y >> sh) & 0x03030303u) | (((r.h.y >> hb) & 0x01010101u) << 2)};
#pragma unroll
  for (int e = 0; e < 8; ++e)
    v[e] = __fmul_rn(dl, mbyte(w[e >> 2], e & 3, -(kMagic + 4.0f)));
}
template <>
__device__ __forceinline__ void deq<KQ_IQ2_XS>(const Args& a, const Raw& r, int o, float (&v)[8]) {
  const uint32_t s = (r.s0 >> (4 * ((o >> 4) & 1))) & 15u;
  const float db = __fmul_rn(__fmul_rn(h2f(r.d), __fadd_rn(0.5f, (float)s)), 0.25f);
  const uint32_t qs = r.q.x & 0xffffu;
  const uint32_t sg = __ldg(a.ksigns + (qs >> 9));
  const uint32_t g0 = __ldg(a.grid + 2 * (qs & 511u)), g1 = __ldg(a.grid + 2 * (qs & 511u) + 1);
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const float sgn = ((sg >> e) & 1u) ? -1.0f : 1.0f;
    v[e] = __fmul_rn(__fmul_rn(db, mbyte(e < 4 ? g0 : g1, e & 3, -kMagic)), sgn);
  }
}
template <>
__device__ __forceinline__ void deq<KQ_IQ2_S>(const Args& a, const Raw& r, int o, float (&v)[8]) {
  const uint32_t s = (r.s2 >> (4 * ((o >> 4) & 1))) & 15u;
  const float db = __fmul_rn(__fmul_rn(h2f(r.d), __fadd_rn(0.5f, (float)s)), 0.25f);
  const int g = o >> 3;
  const uint32_t idx = (r.q.x & 0xffu) | (((r.s0 >> (2 * (g & 3))) & 3u) << 8);
  const uint32_t g0 = __ldg(a.grid + 2 * idx), g1 = __ldg(a.grid + 2 * idx + 1);
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const float sgn = ((r.s1 >> e) & 1u) ? -1.0f : 1.0f;
    v[e] = __fmul_rn(__fmul_rn(db, mbyte(e < 4 ? g0 : g1, e & 3, -kMagic)), sgn);
  }
}

template <int T>
__device__ __forceinline__ uint4 deq_bf16(const Args& a, const Raw& r, int o) {
  float v[8];
  deq<T>(a, r, o, v);
  uint4 w;
  w.x = bfbits(v[0]) | (bfbits(v[1]) << 16);
  w.y = bfbits(v[2]) | (bfbits(v[3]) << 16);
  w.z = bfbits(v[4]) | (bfbits(v[5]) << 16);
  w.w = bfbits(v[6]) | (bfbits(v[7]) << 16);
  return w;
}

// ================================ kernels =====================================
template <int T, int M, int WPR, int U>
__global__ void __launch_bounds__(256)
miv_kq_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx, Args a,
              __nv_bfloat16* __restrict__ y, int64_t ldy) {
  constexpr int RPC = 8 / WPR;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int rloc = warp / WPR, s = warp % WPR;
  const int row = blockIdx.x * RPC + rloc;
  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.0f;
  if (row < a.N) {
    const int KV = a.K >> 3;
    const int v0 = (int)(((int64_t)s * KV) / WPR);
    const int v1 = (int)(((int64_t)(s + 1) * KV) / WPR);
    const int64_t b0 = (int64_t)row * (a.K >> 8);
    int v = v0 + lane;
    for (; v + 32 * (U - 1) < v1; v += 32 * U) {
      Raw rw[U];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int vv = v + 32 * u;
        rw[u] = load<T>(a, b0 + (vv >> 5), (vv & 31) * 8);
      }
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int vv = v + 32 * u;
        fma_vec<M>(acc, deq_bf16<T>(a, rw[u], (vv & 31) * 8), x, ldx, (int64_t)vv * 8);
      }
    }
    for (; v < v1; v += 32) {
      const Raw r1 = load<T>(a, b0 + (v >> 5), (v & 31) * 8);
      fma_vec<M>(acc, deq_bf16<T>(a, r1, (v & 31) * 8), x, ldx, (int64_t)v * 8);
    }
  }
  epilogue<M, WPR>(acc, warp, lane, row, a.N, a.bias, y, ldy);
}

// R rows per warp: lane l still walks vectors v0 + l, v0 + l + 32, ... of EVERY
// row it owns, in the same order, with one accumulator per (row, m); only the
// x loads/unpacks are shared between the R rows.  Per-row arithmetic is the
// identical __fmaf_rn chain, butterfly and slice sum -> the same bits as R = 1
// (and as bf16 MIV).
template <int M, int R>
__device__ __forceinline__ void fma_vec_r(float (&acc)[R][M], const uint4 (&w)[R],
                                          const __nv_bfloat16* __restrict__ x, int64_t ldx,
                                          int64_t col) {
  float wf[R][8];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    wf[r][0] = bf_lo(w[r].x); wf[r][1] = bf_hi(w[r].x); wf[r][2] = bf_lo(w[r].y);
    wf[r][3] = bf_hi(w[r].y); wf[r][4] = bf_lo(w[r].z); wf[r][5] = bf_hi(w[r].z);
    wf[r][6] = bf_lo(w[r].w); wf[r][7] = bf_hi(w[r].w);
  }
#pragma unroll
  for (int m = 0; m < M; ++m) {
    const uint4 xv = __ldg(reinterpret_cast<const uint4*>(x + m * ldx + col));
    const float x0 = bf_lo(xv.x), x1 = bf_hi(xv.x), x2 = bf_lo(xv.y), x3 = bf_hi(xv.y);
    const float x4 = bf_lo(xv.z), x5 = bf_hi(xv.z), x6 = bf_lo(xv.w), x7 = bf_hi(xv.w);
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float a = acc[r][m];
      a = __fmaf_rn(x0, wf[r][0], a);
      a = __fmaf_rn(x1, wf[r][1], a);
      a = __fmaf_rn(x2, wf[r][2], a);
      a = __fmaf_rn(x3, wf[r][3], a);
      a = __fmaf_rn(x4, wf[r][4], a);
      a = __fmaf_rn(x5, wf[r][5], a);
      a = __fmaf_rn(x6, wf[r][6], a);
      a = __fmaf_rn(x7, wf[r][7], a);
      acc[r][m] = a;
    }
  }
}

template <int T, int M, int WPR, int U, int R>
__global__ void __launch_bounds__(256)
miv_kq_kernel_r(const __nv_bfloat16* __restrict__ x, int64_t ldx, Args a,
                __nv_bfloat16* __restrict__ y, int64_t ldy) {
  constexpr int WROWS = 8 / WPR;              // warp-rows per CTA
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int rloc = warp / WPR, s = warp % WPR;
  const int row0 = (blockIdx.x * WROWS + rloc) * R;
  float acc[R][M];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;
  if (row0 < a.N) {
    const int KV = a.K >> 3;
    const int v0 = (int)(((int64_t)s * KV) / WPR);
    const int v1 = (int)(((int64_t)(s + 1) * KV) / WPR);
    const int nb = a.K >> 8;
    int64_t brow[R];
#pragma unroll
    for (int r = 0; r < R; ++r) brow[r] = (int64_t)min(row0 + r, a.N - 1) * nb;   // tail rows: dup, discarded
    int v = v0 + lane;
    for (; v + 32 * (U - 1) < v1; v += 32 * U) {
      Raw rw[U][R];
#pragma unroll
      for (int u = 0; u < U; ++u)
#pragma unroll
        for (int r = 0; r < R; ++r) {
          const int vv = v + 32 * u;
          rw[u][r] = load<T>(a, brow[r] + (vv >> 5), (vv & 31) * 8);
        }
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int vv = v + 32 * u;
        uint4 w[R];
#pragma unroll
        for (int r = 0; r < R; ++r) w[r] = deq_bf16<T>(a, rw[u][r], (vv & 31) * 8);
        fma_vec_r<M, R>(acc, w, x, ldx, (int64_t)vv * 8);
      }
    }
    for (; v < v1; v += 32) {
      uint4 w[R];
#pragma unroll
      for (int r = 0; r < R; ++r) w[r] = deq_bf16<T>(a, load<T>(a, brow[r] + (v >> 5), (v & 31) * 8), (v & 31) * 8);
      fma_vec_r<M, R>(acc, w, x, ldx, (int64_t)v * 8);
    }
  }
  // epilogue: identical per-row reduction
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        acc[r][m] = __fadd_rn(acc[r][m], __shfl_xor_sync(0xffffffffu, acc[r][m], off));
  if (WPR == 1) {
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
        if (row < a.N) {
          const float b = a.bias ? __bfloat162float(a.bias[row]) : 0.0f;
#pragma unroll
          for (int m = 0; m < M; ++m) {
            const float o = a.bias ? __fadd_rn(acc[r][m], b) : acc[r][m];
            y[m * ldy + row] = __float2bfloat16_rn(o);
          }
        }
      }
    }
  } else {
    __shared__ float red[8][R][M];
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) red[warp][r][m] = acc[r][m];
    }
    __syncthreads();
    if (threadIdx.x < WROWS * R * M) {
      const int wr = threadIdx.x / (R * M), r = (threadIdx.x / M) % R, m = threadIdx.x % M;
      const int orow = (blockIdx.x * WROWS + wr) * R + r;
      if (orow < a.N) {
        float t = red[wr * WPR][r][m];
#pragma unroll
        for (int j = 1; j < WPR; ++j) t = __fadd_rn(t, red[wr * WPR + j][r][m]);
        if (a.bias) t = __fadd_rn(t, __bfloat162float(a.bias[orow]));
        y[m * ldy + orow] = __float2bfloat16_rn(t);
      }
    }
  }
}

// decode-only twin: same load/deq; writes the FP32 values (pre-bf16) for G1-q
template <int T>
__global__ void kq_dequant_kernel(Args a, float* __restrict__ out, int64_t nvec) {
  const int64_t v = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= nvec) return;
  const int64_t b = v >> 5;
  const int o = (int)(v & 31) * 8;
  const Raw r = load<T>(a, b, o);
  float w[8];
  deq<T>(a, r, o, w);
  float4* p = reinterpret_cast<float4*>(out + v * 8);
  p[0] = make_float4(w[0], w[1], w[2], w[3]);
  p[1] = make_float4(w[4], w[5], w[6], w[7]);
}

inline Args make_args(const MivKqDesc& d) {
  Args a;
  a.a0 = d.a0; a.a1 = d.a1; a.a2 = d.a2; a.a3 = d.a3; a.a4 = d.a4;
  a.grid = d.grid; a.ksigns = d.ksigns; a.bias = d.bias; a.N = d.n; a.K = d.k;
  return a;
}

template <int T, int M, int WPR, int U>
inline cudaError_t launch_twmu(const MivKqDesc& d, const __nv_bfloat16* x, int64_t ldx,
                               __nv_bfloat16* y, int64_t ldy, cudaStream_t st) {
  constexpr int RPC = 8 / WPR;
  if (d.rpw == 2) {          // rows per warp = 2 (same bits; shares x unpack)
    constexpr int RR = 2 * RPC;
    miv_kq_kernel_r<T, M, WPR, U, 2><<<(d.n + RR - 1) / RR, 256, 0, st>>>(x, ldx, make_args(d), y, ldy);
  } else {
    miv_kq_kernel<T, M, WPR, U><<<(d.n + RPC - 1) / RPC, 256, 0, st>>>(x, ldx, make_args(d), y, ldy);
  }
  return cudaGetLastError();
}
template <int T, int M, int WPR>
inline cudaError_t launch_twm(const MivKqDesc& d, const __nv_bfloat16* x, int64_t ldx,
                              __nv_bfloat16* y, int64_t ldy, cudaStream_t st) {
  switch (d.unroll) {
    case 1: return launch_twmu<T, M, WPR, 1>(d, x, ldx, y, ldy, st);
    case 2: return launch_twmu<T, M, WPR, 2>(d, x, ldx, y, ldy, st);
    case 4: return launch_twmu<T, M, WPR, 4>(d, x, ldx, y, ldy, st);
    default: return cudaErrorInvalidValue;
  }
}
template <int T, int M>
inline cudaError_t launch_tm(const MivKqDesc& d, const __nv_bfloat16* x, int64_t ldx,
                             __nv_bfloat16* y, int64_t ldy, cudaStream_t st) {
  switch (d.wpr) {
    case 1: return launch_twm<T, M, 1>(d, x, ldx, y, ldy, st);
    case 2: return launch_twm<T, M, 2>(d, x, ldx, y, ldy, st);
    case 4: return launch_twm<T, M, 4>(d, x, ldx, y, ldy, st);
    case 8: return launch_twm<T, M, 8>(d, x, ldx, y, ldy, st);
    default: return cudaErrorInvalidValue;
  }
}
template <int M>
inline cudaError_t launch_m(const MivKqDesc& d, const __nv_bfloat16* x, int64_t ldx,
                            __nv_bfloat16* y, int64_t ldy, cudaStream_t st) {
  switch (d.type) {
    case KQ_Q4_K: return launch_tm<KQ_Q4_K, M>(d, x, ldx, y, ldy, st);
    case KQ_Q5_K: return launch_tm<KQ_Q5_K, M>(d, x, ldx, y, ldy, st);
    case KQ_Q6_K: return launch_tm<KQ_Q6_K, M>(d, x, ldx, y, ldy, st);
    case KQ_IQ4_XS: return launch_tm<KQ_IQ4_XS, M>(d, x, ldx, y, ldy, st);
    case KQ_IQ3_XXS: return launch_tm<KQ_IQ3_XXS, M>(d, x, ldx, y, ldy, st);
    case KQ_IQ3_S: return launch_tm<KQ_IQ3_S, M>(d, x, ldx, y, ldy, st);
    case KQ_Q2_K: return launch_tm<KQ_Q2_K, M>(d, x, ldx, y, ldy, st);
    case KQ_Q3_K: return launch_tm<KQ_Q3_K, M>(d, x, ldx, y, ldy, st);
    case KQ_IQ2_XS: return launch_tm<KQ_IQ2_XS, M>(d, x, ldx, y, ldy, st);
    case KQ_IQ2_S: return launch_tm<KQ_IQ2_S, M>(d, x, ldx, y, ldy, st);
    default: return cudaErrorInvalidValue;
  }
}

}  // namespace miv_kq

#define MIV_KQ_DEFINE_M(MM)                                                               \
  cudaError_t miv_kq_gemv_m##MM(const MivKqDesc& d, const __nv_bfloat16* x, int64_t ldx,  \
                                __nv_bfloat16* y, int64_t ldy, cudaStream_t st) {          \
    return miv_kq::launch_m<MM>(d, x, ldx, y, ldy, st);                                    \
  }
