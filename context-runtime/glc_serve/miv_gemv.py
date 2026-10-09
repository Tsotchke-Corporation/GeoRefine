"""MIV-GEMV: an M-invariant bf16 weight-streaming skinny GEMM (M = 1..8).

``y[m, n] = round_bf16( sum_k x[m, k] * W[n, k]  (+ bias[n]) )`` for the
``nn.Linear`` layout ``W = [N, K]``, with a reduction order that does NOT
depend on M.  Row ``m`` of an M=8 call is therefore bitwise identical to the
M=1 call on that row -- the property a speculative verify step (M = k+1)
needs to reproduce the plain M=1 decode bit for bit (design E1,
``~/Desktop/Selene/KERNEL_DESIGN_ICC_20260924.md`` section 3).

The contract, instruction by instruction:

* one warp owns one (output row, K-slice) pair; the slice split ``WPR``
  (warps per row, 1/2/4/8) is a per-SHAPE constant from ``CONFIG`` and never
  a function of M;
* each lane walks its 16-byte weight vectors in increasing K order
  (``v = v0 + lane, v0 + lane + 32, ...``) and folds the 8 products of a
  vector into an FP32 accumulator with explicit ``__fmaf_rn`` in element
  order; the unroll factor only batches LOADS, it never reorders the FMA
  chain;
* the 32 lane partials are combined by a fixed xor-butterfly
  (``__shfl_xor_sync`` 16, 8, 4, 2, 1); the WPR warp partials are then summed
  in slice order 0..WPR-1 by one thread;
* the bias is added in FP32 (``__fadd_rn``) and the result is rounded to bf16
  exactly once.

M only changes how many independent accumulators each lane carries, so the
arithmetic of every output element is the same sequence of IEEE operations
for every M.  This is tested, not assumed (``scripts/miv_gemv_bench.py`` and
``tests/test_miv_gemv.py``).

Weights are streamed with ``ld.global.nc.L1::no_allocate`` (they are read once
per call); activations go through the read-only cache.

Default-off: nothing imports this module unless a caller does, and
``install_*`` must be called explicitly.  ``modules.set_exact_linear`` is the
single hook into the TBE exact path; unset, that path is byte-identical.
``MIV_GEMV_TBE_MIN_BLOCKS=4|5|6`` is a separate default-off experiment that
sets launch bounds only on this module's coded-weight TBE kernel; unset or zero
preserves the original extension name, cache directory, and compiler flags.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_M = 8
_THREADS = 256
_WARPS = _THREADS // 32
WPR_CHOICES = (1, 2, 4, 8)
UNROLL_CHOICES = (2, 4, 8)
TBE_UNROLL_CHOICES = (1, 2, 4)

# Per-shape (N, K) -> (WPR, UNROLL).  A shape constant: it may differ between
# shapes but never with M.  Tuned on RTX PRO 6000 Blackwell Server (sm_120) at
# M=1 (scripts/miv_gemv_bench.py --tune); receipts in the F-MIV memo.
CONFIG: Dict[Tuple[int, int], Tuple[int, int]] = {}


def default_config(n: int, k: int) -> Tuple[int, int]:
    """Shape-only fallback: enough CTAs to fill ~188 SMs several times."""
    wpr = 1
    while wpr < 8 and (n * wpr) // _WARPS < 1024 and (k // 8) // (wpr * 2) >= 64:
        wpr *= 2
    return wpr, 4


def config_for(n: int, k: int) -> Tuple[int, int]:
    return CONFIG.get((int(n), int(k))) or default_config(n, k)


_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef MIV_GEMV_TBE_MIN_BLOCKS
#define MIV_GEMV_TBE_MIN_BLOCKS 0
#endif

namespace {

__device__ __forceinline__ uint4 ld_stream(const void* p) {
  uint4 r;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
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

template <int M, int WPR, int U>
__global__ void __launch_bounds__(256)
miv_gemv_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx,
                const __nv_bfloat16* __restrict__ W,
                const __nv_bfloat16* __restrict__ bias,
                __nv_bfloat16* __restrict__ y, int64_t ldy, int N, int K) {
  constexpr int RPC = 8 / WPR;               // output rows per CTA
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int rloc = warp / WPR, s = warp % WPR;
  const int row = blockIdx.x * RPC + rloc;
  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.0f;
  if (row < N) {
    const int KV = K >> 3;                   // 16-byte vectors per row
    const int v0 = (int)(((int64_t)s * KV) / WPR);
    const int v1 = (int)(((int64_t)(s + 1) * KV) / WPR);
    const __nv_bfloat16* wrow = W + (int64_t)row * K;
    int v = v0 + lane;
    for (; v + 32 * (U - 1) < v1; v += 32 * U) {
      uint4 wv[U];
#pragma unroll
      for (int u = 0; u < U; ++u) wv[u] = ld_stream(wrow + (int64_t)(v + 32 * u) * 8);
#pragma unroll
      for (int u = 0; u < U; ++u) fma_vec<M>(acc, wv[u], x, ldx, (int64_t)(v + 32 * u) * 8);
    }
    for (; v < v1; v += 32) {
      const uint4 w1 = ld_stream(wrow + (int64_t)v * 8);
      fma_vec<M>(acc, w1, x, ldx, (int64_t)v * 8);
    }
  }
  // fixed xor butterfly: every lane ends with the same bits (a+b == b+a)
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

template <int M, int WPR, int U>
void launch(const at::Tensor& x, const at::Tensor& W, const __nv_bfloat16* b, at::Tensor& y,
            cudaStream_t st) {
  const int N = (int)W.size(0), K = (int)W.size(1);
  constexpr int RPC = 8 / WPR;
  const int grid = (N + RPC - 1) / RPC;
  miv_gemv_kernel<M, WPR, U><<<grid, 256, 0, st>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()), x.stride(0),
      reinterpret_cast<const __nv_bfloat16*>(W.data_ptr<at::BFloat16>()), b,
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()), y.stride(0), N, K);
}

template <int M, int WPR>
void by_u(int u, const at::Tensor& x, const at::Tensor& W, const __nv_bfloat16* b, at::Tensor& y,
          cudaStream_t st) {
  switch (u) {
    case 2: launch<M, WPR, 2>(x, W, b, y, st); break;
    case 4: launch<M, WPR, 4>(x, W, b, y, st); break;
    case 8: launch<M, WPR, 8>(x, W, b, y, st); break;
    default: TORCH_CHECK(false, "miv_gemv: unroll must be 2/4/8, got ", u);
  }
}

template <int M>
void by_wpr(int wpr, int u, const at::Tensor& x, const at::Tensor& W, const __nv_bfloat16* b,
            at::Tensor& y, cudaStream_t st) {
  switch (wpr) {
    case 1: by_u<M, 1>(u, x, W, b, y, st); break;
    case 2: by_u<M, 2>(u, x, W, b, y, st); break;
    case 4: by_u<M, 4>(u, x, W, b, y, st); break;
    case 8: by_u<M, 8>(u, x, W, b, y, st); break;
    default: TORCH_CHECK(false, "miv_gemv: wpr must be 1/2/4/8, got ", wpr);
  }
}


// ---------------------------------------------------------------------------
// MIV-TBE: the SAME per-lane FMA sequence as miv_gemv_kernel, with the weight
// vector decoded in registers from the GLC-TBE mma16 container (E2).  Lane l
// of the warp owning (row, slice s) visits vectors v = v0 + l, v0 + l + 32, ...
// exactly as above and feeds each decoded uint4 to the identical fma_vec<M>;
// the butterfly and the slice sum are the identical code.  So for bit-identical
// weights (G1) every output is bitwise the bf16 MIV output for the same
// (N, K, WPR): G3a by construction, and tested.
//
// Container (glc_loader/tbe_container.py): tile t = 64 consecutive row-major
// elements; planes int32[T][3][2] (plane b, word j: bit p&31 of stored
// position p = 32j + bit), smb uint8[T*64] ((sign<<7)|mantissa, stored order),
// esc uint8 (raw exponents, tile-major then position-ascending), mma16 stored
// position of original in-tile element q = 8*i + r (vector i, element r):
//   p = 16*(r>>1) + 2*i + (r&1).
// rowparam[row] = base | mode << 8 (per row, so concatenated tensors keep
// their own windows); escbase[row*WPR + s] = index in esc of the first escape
// of slice s of that row (a load-time prefix over the stored planes).
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint2 ld_nc_u2(const void* p) {
  uint2 r;
  asm("ld.global.nc.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t ld_nc_u16(const void* p) {
  unsigned short r;
  asm("ld.global.nc.u16 %0, [%1];" : "=h"(r) : "l"(p));
  return (uint32_t)r;
}

template <int M, int WPR, int U>
#if MIV_GEMV_TBE_MIN_BLOCKS == 0
__global__ void __launch_bounds__(256)
#else
__global__ void __launch_bounds__(256, MIV_GEMV_TBE_MIN_BLOCKS)
#endif
miv_tbe_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx,
               const uint32_t* __restrict__ planes, const uint8_t* __restrict__ smb,
               const uint8_t* __restrict__ esc, const int32_t* __restrict__ rowparam,
               const int32_t* __restrict__ escbase,
               __nv_bfloat16* __restrict__ y, int64_t ldy, int N, int K) {
  constexpr int RPC = 8 / WPR;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int rloc = warp / WPR, s = warp % WPR;
  const int row = blockIdx.x * RPC + rloc;
  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.0f;
  if (row < N) {                              // warp-uniform
    const int KV = K >> 3;
    const int v0 = (int)(((int64_t)s * KV) / WPR);
    const int v1 = (int)(((int64_t)(s + 1) * KV) / WPR);
    const int64_t tile0 = (int64_t)row * (K >> 6);
    const int rp = rowparam[row];
    const uint32_t ebase_win = (uint32_t)(rp & 0xff);
    const bool w6z = (rp >> 8) == 1;
    int ebase = escbase[(int64_t)row * WPR + s];
    const int i = lane & 7, g = lane >> 3;
    for (int vg = v0; vg < v1; vg += 32 * U) {
      uint2 P0[U], P1[U], P2[U];
      uint32_t S[U][4];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int v = vg + 32 * u + lane;
        if (v < v1) {
          const int64_t t = tile0 + (v >> 3);
          const uint32_t* pp = planes + t * 6;
          P0[u] = ld_nc_u2(pp); P1[u] = ld_nc_u2(pp + 2); P2[u] = ld_nc_u2(pp + 4);
          const uint8_t* sp = smb + t * 64 + 2 * i;
#pragma unroll
          for (int a = 0; a < 4; ++a) S[u][a] = ld_nc_u16(sp + 16 * a);
        } else {
          P0[u] = P1[u] = P2[u] = make_uint2(0xffffffffu, 0xffffffffu);
#pragma unroll
          for (int a = 0; a < 4; ++a) S[u][a] = 0;
        }
      }
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int v = vg + 32 * u + lane;
        const uint32_t m0 = ~(P0[u].x | P1[u].x | P2[u].x);
        const uint32_t m1 = ~(P0[u].y | P1[u].y | P2[u].y);
        const int cnt = __popc(m0) + __popc(m1);          // 0 for inactive tiles
        const int c0 = __shfl_sync(0xffffffffu, cnt, 0);
        const int c1 = __shfl_sync(0xffffffffu, cnt, 8);
        const int c2 = __shfl_sync(0xffffffffu, cnt, 16);
        const int c3 = __shfl_sync(0xffffffffu, cnt, 24);
        const int pre = (g > 0 ? c0 : 0) + (g > 1 ? c1 : 0) + (g > 2 ? c2 : 0);
        if (v < v1) {
          uint32_t hw[8];
#pragma unroll
          for (int r = 0; r < 8; ++r) {
            const int wsel = r >> 2;                                  // p < 32 iff r < 4
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
              e = (uint32_t)__ldg(esc + ebase + pre + rank);
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
#pragma unroll
  for (int m = 0; m < M; ++m) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc[m] = __fadd_rn(acc[m], __shfl_xor_sync(0xffffffffu, acc[m], off));
  }
  if (WPR == 1) {
    if (lane == 0 && row < N) {
#pragma unroll
      for (int m = 0; m < M; ++m) y[m * ldy + row] = __float2bfloat16_rn(acc[m]);
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
        y[m * ldy + orow] = __float2bfloat16_rn(t);
      }
    }
  }
}

// per-tile escape counts (load-time index build)
__global__ void tbe_tile_escapes_kernel(const uint32_t* __restrict__ planes, int64_t T,
                                        int32_t* __restrict__ out) {
  const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  const uint32_t* p = planes + t * 6;
  out[t] = __popc(~(p[0] | p[2] | p[4])) + __popc(~(p[1] | p[3] | p[5]));
}

template <int M, int WPR, int U>
void launch_tbe(const at::Tensor& x, const at::Tensor& planes, const at::Tensor& smb,
                const at::Tensor& esc, const at::Tensor& rowparam, const at::Tensor& escbase,
                at::Tensor& y, int N, int K, cudaStream_t st) {
  constexpr int RPC = 8 / WPR;
  const int grid = (N + RPC - 1) / RPC;
  miv_tbe_kernel<M, WPR, U><<<grid, 256, 0, st>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()), x.stride(0),
      reinterpret_cast<const uint32_t*>(planes.data_ptr<int32_t>()),
      smb.data_ptr<uint8_t>(), esc.data_ptr<uint8_t>(), rowparam.data_ptr<int32_t>(),
      escbase.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()), y.stride(0), N, K);
}

template <int M, int WPR>
void tbe_by_u(int u, const at::Tensor& x, const at::Tensor& pl, const at::Tensor& sm,
              const at::Tensor& es, const at::Tensor& rp, const at::Tensor& eb, at::Tensor& y,
              int N, int K, cudaStream_t st) {
  switch (u) {
    case 1: launch_tbe<M, WPR, 1>(x, pl, sm, es, rp, eb, y, N, K, st); break;
    case 2: launch_tbe<M, WPR, 2>(x, pl, sm, es, rp, eb, y, N, K, st); break;
    case 4: launch_tbe<M, WPR, 4>(x, pl, sm, es, rp, eb, y, N, K, st); break;
    default: TORCH_CHECK(false, "miv_tbe: unroll must be 1/2/4, got ", u);
  }
}

template <int M>
void tbe_by_wpr(int wpr, int u, const at::Tensor& x, const at::Tensor& pl, const at::Tensor& sm,
                const at::Tensor& es, const at::Tensor& rp, const at::Tensor& eb, at::Tensor& y,
                int N, int K, cudaStream_t st) {
  switch (wpr) {
    case 1: tbe_by_u<M, 1>(u, x, pl, sm, es, rp, eb, y, N, K, st); break;
    case 2: tbe_by_u<M, 2>(u, x, pl, sm, es, rp, eb, y, N, K, st); break;
    case 4: tbe_by_u<M, 4>(u, x, pl, sm, es, rp, eb, y, N, K, st); break;
    case 8: tbe_by_u<M, 8>(u, x, pl, sm, es, rp, eb, y, N, K, st); break;
    default: TORCH_CHECK(false, "miv_tbe: wpr must be 1/2/4/8, got ", wpr);
  }
}


// ---------------------------------------------------------------------------
// MIV-Q8: the SAME per-lane FMA sequence as miv_gemv_kernel over Q8_0 weights
// (SoA: qs int8 [N][K], sc fp16 [N][K/32]).  Each 8-element vector is
// dequantized in registers to bf16_rn(d * q) -- the exact bits of gguf-py's
// reference dequantize cast to bf16 -- and fed to the identical fma_vec<M>.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint2 ld_stream_u2(const void* p) {
  uint2 r;
  asm("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}
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

template <int M, int WPR>
__device__ __forceinline__ void miv_epilogue(float (&acc)[M], int warp, int lane, int row, int N,
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
#pragma unroll
      for (int m = 0; m < M; ++m) y[m * ldy + row] = __float2bfloat16_rn(acc[m]);
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
        y[m * ldy + orow] = __float2bfloat16_rn(t);
      }
    }
  }
}

template <int M, int WPR, int U>
__global__ void __launch_bounds__(256)
miv_q8_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx, const int8_t* __restrict__ qs,
              const unsigned short* __restrict__ sc, __nv_bfloat16* __restrict__ y, int64_t ldy,
              int N, int K) {
  constexpr int RPC = 8 / WPR;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int rloc = warp / WPR, s = warp % WPR;
  const int row = blockIdx.x * RPC + rloc;
  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.0f;
  if (row < N) {
    const int KV = K >> 3;
    const int v0 = (int)(((int64_t)s * KV) / WPR);
    const int v1 = (int)(((int64_t)(s + 1) * KV) / WPR);
    const int8_t* qrow = qs + (int64_t)row * K;
    const unsigned short* srow = sc + (int64_t)row * (K >> 5);
    int v = v0 + lane;
    for (; v + 32 * (U - 1) < v1; v += 32 * U) {
      uint2 qv[U];
      uint32_t sv[U];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        qv[u] = ld_stream_u2(qrow + (int64_t)(v + 32 * u) * 8);
        sv[u] = (uint32_t)__ldg(srow + ((v + 32 * u) >> 2));
      }
#pragma unroll
      for (int u = 0; u < U; ++u) fma_vec<M>(acc, dq8(qv[u], sv[u]), x, ldx, (int64_t)(v + 32 * u) * 8);
    }
    for (; v < v1; v += 32) {
      const uint2 q1 = ld_stream_u2(qrow + (int64_t)v * 8);
      const uint32_t s1 = (uint32_t)__ldg(srow + (v >> 2));
      fma_vec<M>(acc, dq8(q1, s1), x, ldx, (int64_t)v * 8);
    }
  }
  miv_epilogue<M, WPR>(acc, warp, lane, row, N, y, ldy);
}

__global__ void q8_dequant_kernel(const int8_t* __restrict__ qs, const unsigned short* __restrict__ sc,
                                  __nv_bfloat16* __restrict__ out, int64_t nvec) {
  const int64_t v = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= nvec) return;
  const uint2 q = *reinterpret_cast<const uint2*>(qs + v * 8);
  *reinterpret_cast<uint4*>(out + v * 8) = dq8(q, (uint32_t)sc[v >> 2]);
}

template <int M, int WPR, int U>
void launch_q8(const at::Tensor& x, const at::Tensor& qs, const at::Tensor& sc, at::Tensor& y,
               int N, int K, cudaStream_t st) {
  constexpr int RPC = 8 / WPR;
  miv_q8_kernel<M, WPR, U><<<(N + RPC - 1) / RPC, 256, 0, st>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()), x.stride(0),
      qs.data_ptr<int8_t>(), reinterpret_cast<const unsigned short*>(sc.data_ptr<at::Half>()),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()), y.stride(0), N, K);
}
template <int M, int WPR>
void q8_by_u(int u, const at::Tensor& x, const at::Tensor& qs, const at::Tensor& sc, at::Tensor& y,
             int N, int K, cudaStream_t st) {
  switch (u) {
    case 2: launch_q8<M, WPR, 2>(x, qs, sc, y, N, K, st); break;
    case 4: launch_q8<M, WPR, 4>(x, qs, sc, y, N, K, st); break;
    case 8: launch_q8<M, WPR, 8>(x, qs, sc, y, N, K, st); break;
    default: TORCH_CHECK(false, "miv_q8: unroll 2/4/8");
  }
}
template <int M>
void q8_by_wpr(int wpr, int u, const at::Tensor& x, const at::Tensor& qs, const at::Tensor& sc,
               at::Tensor& y, int N, int K, cudaStream_t st) {
  switch (wpr) {
    case 1: q8_by_u<M, 1>(u, x, qs, sc, y, N, K, st); break;
    case 2: q8_by_u<M, 2>(u, x, qs, sc, y, N, K, st); break;
    case 4: q8_by_u<M, 4>(u, x, qs, sc, y, N, K, st); break;
    case 8: q8_by_u<M, 8>(u, x, qs, sc, y, N, K, st); break;
    default: TORCH_CHECK(false, "miv_q8: wpr 1/2/4/8");
  }
}

}  // namespace

at::Tensor miv_gemv(at::Tensor x, at::Tensor W, c10::optional<at::Tensor> bias,
                    int64_t wpr, int64_t unroll) {
  TORCH_CHECK(x.is_cuda() && W.is_cuda(), "miv_gemv: CUDA tensors required");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && W.scalar_type() == at::kBFloat16,
              "miv_gemv: bf16 x and W required");
  TORCH_CHECK(x.dim() == 2 && W.dim() == 2, "miv_gemv: x [M,K], W [N,K]");
  TORCH_CHECK(W.is_contiguous(), "miv_gemv: W must be contiguous");
  TORCH_CHECK(x.stride(1) == 1, "miv_gemv: x rows must be contiguous");
  const int64_t M = x.size(0), K = x.size(1), N = W.size(0);
  TORCH_CHECK(W.size(1) == K, "miv_gemv: K mismatch");
  TORCH_CHECK(M >= 1 && M <= 8, "miv_gemv: 1 <= M <= 8, got ", M);
  TORCH_CHECK(K % 8 == 0 && x.stride(0) % 8 == 0, "miv_gemv: K and ldx must be multiples of 8");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0 &&
              (reinterpret_cast<uintptr_t>(W.data_ptr()) & 15) == 0,
              "miv_gemv: 16-byte aligned x and W required");
  TORCH_CHECK(N < (1LL << 31) && K < (1LL << 31), "miv_gemv: shape too large");
  const __nv_bfloat16* b = nullptr;
  if (bias.has_value() && bias->defined()) {
    TORCH_CHECK(bias->is_cuda() && bias->scalar_type() == at::kBFloat16 && bias->numel() == N &&
                bias->is_contiguous(), "miv_gemv: bias must be contiguous bf16 [N] on CUDA");
    b = reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr<at::BFloat16>());
  }
  const c10::cuda::CUDAGuard guard(x.device());
  auto y = at::empty({M, N}, x.options());
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  switch (M) {
    case 1: by_wpr<1>((int)wpr, (int)unroll, x, W, b, y, st); break;
    case 2: by_wpr<2>((int)wpr, (int)unroll, x, W, b, y, st); break;
    case 3: by_wpr<3>((int)wpr, (int)unroll, x, W, b, y, st); break;
    case 4: by_wpr<4>((int)wpr, (int)unroll, x, W, b, y, st); break;
    case 5: by_wpr<5>((int)wpr, (int)unroll, x, W, b, y, st); break;
    case 6: by_wpr<6>((int)wpr, (int)unroll, x, W, b, y, st); break;
    case 7: by_wpr<7>((int)wpr, (int)unroll, x, W, b, y, st); break;
    case 8: by_wpr<8>((int)wpr, (int)unroll, x, W, b, y, st); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

void miv_gemv_out(at::Tensor x, at::Tensor W, c10::optional<at::Tensor> bias, at::Tensor y,
                  int64_t wpr, int64_t unroll) {
  const int64_t M = x.size(0), K = x.size(1), N = W.size(0);
  TORCH_CHECK(x.is_cuda() && W.is_cuda() && y.is_cuda(), "miv_gemv_out: CUDA tensors");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && W.scalar_type() == at::kBFloat16 &&
              y.scalar_type() == at::kBFloat16, "miv_gemv_out: bf16");
  TORCH_CHECK(W.is_contiguous() && x.stride(1) == 1 && y.stride(1) == 1, "miv_gemv_out: layout");
  TORCH_CHECK(W.size(1) == K && y.size(0) == M && y.size(1) == N, "miv_gemv_out: shapes");
  TORCH_CHECK(M >= 1 && M <= 8 && K % 8 == 0 && x.stride(0) % 8 == 0, "miv_gemv_out: M/K");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0 &&
              (reinterpret_cast<uintptr_t>(W.data_ptr()) & 15) == 0, "miv_gemv_out: align");
  const __nv_bfloat16* b = nullptr;
  if (bias.has_value() && bias->defined())
    b = reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr<at::BFloat16>());
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  switch (M) {
    case 1: by_wpr<1>((int)wpr, (int)unroll, x, W, b, y, st); break;
    case 2: by_wpr<2>((int)wpr, (int)unroll, x, W, b, y, st); break;
    case 3: by_wpr<3>((int)wpr, (int)unroll, x, W, b, y, st); break;
    case 4: by_wpr<4>((int)wpr, (int)unroll, x, W, b, y, st); break;
    case 5: by_wpr<5>((int)wpr, (int)unroll, x, W, b, y, st); break;
    case 6: by_wpr<6>((int)wpr, (int)unroll, x, W, b, y, st); break;
    case 7: by_wpr<7>((int)wpr, (int)unroll, x, W, b, y, st); break;
    case 8: by_wpr<8>((int)wpr, (int)unroll, x, W, b, y, st); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void miv_tbe_out(at::Tensor x, at::Tensor planes, at::Tensor smb, at::Tensor esc,
                 at::Tensor rowparam, at::Tensor escbase, at::Tensor y, int64_t N, int64_t K,
                 int64_t wpr, int64_t unroll) {
  const int64_t M = x.size(0);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 &&
              x.size(1) == K && x.stride(0) % 8 == 0, "miv_tbe_out: x");
  TORCH_CHECK(y.scalar_type() == at::kBFloat16 && y.size(0) == M && y.size(1) == N &&
              y.stride(1) == 1, "miv_tbe_out: y");
  TORCH_CHECK(planes.scalar_type() == at::kInt && smb.scalar_type() == at::kByte &&
              esc.scalar_type() == at::kByte && rowparam.scalar_type() == at::kInt &&
              escbase.scalar_type() == at::kInt, "miv_tbe_out: dtypes");
  TORCH_CHECK(K % (64 * wpr) == 0, "miv_tbe_out: K must be a multiple of 64*WPR");
  TORCH_CHECK(planes.numel() == N * (K / 64) * 6 && smb.numel() == N * K &&
              rowparam.numel() == N && escbase.numel() == N * wpr, "miv_tbe_out: sizes");
  TORCH_CHECK(M >= 1 && M <= 8, "miv_tbe_out: 1 <= M <= 8");
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const int n = (int)N, k = (int)K, w = (int)wpr, u = (int)unroll;
  switch (M) {
    case 1: tbe_by_wpr<1>(w, u, x, planes, smb, esc, rowparam, escbase, y, n, k, st); break;
    case 2: tbe_by_wpr<2>(w, u, x, planes, smb, esc, rowparam, escbase, y, n, k, st); break;
    case 3: tbe_by_wpr<3>(w, u, x, planes, smb, esc, rowparam, escbase, y, n, k, st); break;
    case 4: tbe_by_wpr<4>(w, u, x, planes, smb, esc, rowparam, escbase, y, n, k, st); break;
    case 5: tbe_by_wpr<5>(w, u, x, planes, smb, esc, rowparam, escbase, y, n, k, st); break;
    case 6: tbe_by_wpr<6>(w, u, x, planes, smb, esc, rowparam, escbase, y, n, k, st); break;
    case 7: tbe_by_wpr<7>(w, u, x, planes, smb, esc, rowparam, escbase, y, n, k, st); break;
    case 8: tbe_by_wpr<8>(w, u, x, planes, smb, esc, rowparam, escbase, y, n, k, st); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor tbe_tile_escapes(at::Tensor planes) {
  TORCH_CHECK(planes.is_cuda() && planes.scalar_type() == at::kInt && planes.numel() % 6 == 0,
              "tbe_tile_escapes: int32 [T*6] on CUDA");
  const int64_t T = planes.numel() / 6;
  auto out = at::empty({T}, planes.options());
  if (T) {
    tbe_tile_escapes_kernel<<<(unsigned)((T + 255) / 256), 256, 0,
                              at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint32_t*>(planes.data_ptr<int32_t>()), T, out.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

void miv_q8_out(at::Tensor x, at::Tensor qs, at::Tensor sc, at::Tensor y, int64_t wpr, int64_t unroll) {
  const int64_t M = x.size(0), K = x.size(1), N = qs.size(0);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 &&
              x.stride(0) % 8 == 0 && (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0, "miv_q8: x");
  TORCH_CHECK(qs.scalar_type() == at::kChar && qs.is_contiguous() && qs.size(1) == K, "miv_q8: qs");
  TORCH_CHECK(sc.scalar_type() == at::kHalf && sc.is_contiguous() && sc.size(0) == N &&
              sc.size(1) * 32 == K, "miv_q8: sc");
  TORCH_CHECK(y.scalar_type() == at::kBFloat16 && y.size(0) == M && y.size(1) == N && y.stride(1) == 1,
              "miv_q8: y");
  TORCH_CHECK(M >= 1 && M <= 8 && K % 32 == 0, "miv_q8: M/K");
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const int n = (int)N, k = (int)K, w = (int)wpr, u = (int)unroll;
  switch (M) {
    case 1: q8_by_wpr<1>(w, u, x, qs, sc, y, n, k, st); break;
    case 2: q8_by_wpr<2>(w, u, x, qs, sc, y, n, k, st); break;
    case 3: q8_by_wpr<3>(w, u, x, qs, sc, y, n, k, st); break;
    case 4: q8_by_wpr<4>(w, u, x, qs, sc, y, n, k, st); break;
    case 5: q8_by_wpr<5>(w, u, x, qs, sc, y, n, k, st); break;
    case 6: q8_by_wpr<6>(w, u, x, qs, sc, y, n, k, st); break;
    case 7: q8_by_wpr<7>(w, u, x, qs, sc, y, n, k, st); break;
    case 8: q8_by_wpr<8>(w, u, x, qs, sc, y, n, k, st); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void q8_dequant(at::Tensor qs, at::Tensor sc, at::Tensor out) {
  TORCH_CHECK(qs.is_contiguous() && sc.is_contiguous() && out.is_contiguous() &&
              out.numel() == qs.numel() && qs.numel() % 32 == 0, "q8_dequant");
  const int64_t nvec = qs.numel() / 8;
  q8_dequant_kernel<<<(unsigned)((nvec + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      qs.data_ptr<int8_t>(), reinterpret_cast<const unsigned short*>(sc.data_ptr<at::Half>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), nvec);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
at::Tensor miv_gemv(at::Tensor x, at::Tensor W, c10::optional<at::Tensor> bias,
                    int64_t wpr, int64_t unroll);
void miv_gemv_out(at::Tensor x, at::Tensor W, c10::optional<at::Tensor> bias, at::Tensor y,
                  int64_t wpr, int64_t unroll);
void miv_tbe_out(at::Tensor x, at::Tensor planes, at::Tensor smb, at::Tensor esc,
                 at::Tensor rowparam, at::Tensor escbase, at::Tensor y, int64_t N, int64_t K,
                 int64_t wpr, int64_t unroll);
at::Tensor tbe_tile_escapes(at::Tensor planes);
void miv_q8_out(at::Tensor x, at::Tensor qs, at::Tensor sc, at::Tensor y, int64_t wpr, int64_t unroll);
void q8_dequant(at::Tensor qs, at::Tensor sc, at::Tensor out);
"""

_EXT = None
_EXT_TBE_MIN_BLOCKS = None
_EXT_LOCK = threading.Lock()


def _tbe_min_blocks(environ=None) -> int:
    """Opt-in launch-bounds target for the coded MIV-TBE engine kernel."""
    env = os.environ if environ is None else environ
    raw = env.get("MIV_GEMV_TBE_MIN_BLOCKS")
    try:
        value = 0 if raw is None else int(raw, 10)
    except (TypeError, ValueError) as exc:
        raise ValueError("MIV_GEMV_TBE_MIN_BLOCKS must be one of (0, 4, 5, 6)") from exc
    if value not in (0, 4, 5, 6):
        raise ValueError("MIV_GEMV_TBE_MIN_BLOCKS must be one of (0, 4, 5, 6)")
    return value


def _extension_name(major: int, minor: int, min_blocks: int) -> str:
    name = f"miv_gemv_sm{int(major)}{int(minor)}"
    return name if min_blocks == 0 else f"{name}_tbe_mb{min_blocks}"


def _compile_cuda_flags(min_blocks: int) -> list[str]:
    flags = ["-O3", "--fmad=true", "-lineinfo"]
    if min_blocks:
        flags.append(f"-DMIV_GEMV_TBE_MIN_BLOCKS={min_blocks}")
    return flags


def _build_dir(min_blocks: Optional[int] = None) -> Path:
    min_blocks = _tbe_min_blocks() if min_blocks is None else int(min_blocks)
    if min_blocks not in (0, 4, 5, 6):
        raise ValueError("min_blocks must be one of (0, 4, 5, 6)")
    d = os.environ.get("MIV_GEMV_BUILD_DIR")
    if d:
        p = Path(d)
    else:
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
        p = cache_root / "georefine" / "miv_gemv_build"
    if min_blocks:
        p = p / f"tbe_mb{min_blocks}"
    if str(p).startswith(("/tmp", "/private/tmp")):
        raise RuntimeError(f"refusing a volatile build dir {p}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def extension():
    """JIT-build (once) and return the CUDA extension."""
    global _EXT, _EXT_TBE_MIN_BLOCKS
    min_blocks = _tbe_min_blocks()
    if _EXT is not None and _EXT_TBE_MIN_BLOCKS != min_blocks:
        raise RuntimeError(
            "MIV_GEMV_TBE_MIN_BLOCKS changed after extension load; restart the process"
        )
    if _EXT is not None:
        return _EXT
    with _EXT_LOCK:
        min_blocks = _tbe_min_blocks()
        if _EXT is not None and _EXT_TBE_MIN_BLOCKS != min_blocks:
            raise RuntimeError(
                "MIV_GEMV_TBE_MIN_BLOCKS changed after extension load; restart the process"
            )
        if _EXT is None:
            from torch.utils.cpp_extension import load_inline

            major, minor = torch.cuda.get_device_capability()
            arch = f"{major}.{minor}"
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch)
            _EXT = load_inline(
                name=_extension_name(major, minor, min_blocks),
                cpp_sources=[_CPP_SRC],
                cuda_sources=[_CUDA_SRC],
                functions=["miv_gemv", "miv_gemv_out", "miv_tbe_out", "tbe_tile_escapes", "miv_q8_out",
                           "q8_dequant"],
                extra_cuda_cflags=_compile_cuda_flags(min_blocks),
                build_directory=str(_build_dir(min_blocks)),
                verbose=False,
            )
            _EXT_TBE_MIN_BLOCKS = min_blocks
    return _EXT


def load_config(path) -> int:
    """Load a ``scripts/miv_gemv_bench.py`` tune receipt's ``config`` table."""
    import json

    cfg = json.loads(Path(path).read_text())["config"]
    for key, (wpr, u) in cfg.items():
        n, k = (int(v) for v in key.split("x"))
        CONFIG[(n, k)] = (int(wpr), int(u))
    return len(cfg)


def miv_gemv(x2: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
             *, config: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """``x2 [M, K]`` bf16 (M <= 8) times ``weight [N, K]`` -> ``[M, N]`` bf16."""
    n, k = int(weight.shape[0]), int(weight.shape[1])
    wpr, unroll = config or config_for(n, k)
    if x2.stride(-1) != 1 or x2.stride(0) % 8 or x2.data_ptr() % 16:
        x2 = x2.contiguous()
        if x2.data_ptr() % 16:
            x2 = x2.clone()
    return extension().miv_gemv(x2, weight, bias, int(wpr), int(unroll))


# ---------------------------------------------------------------------------
# model integration (explicit, default-off)
# ---------------------------------------------------------------------------
class Stats:
    """Counts which path every linear call took: proves MIV calls were real."""

    def __init__(self):
        self.miv_by_m: Dict[int, int] = {}
        self.fallback_by_reason: Dict[str, int] = {}

    def as_dict(self) -> Dict[str, Any]:
        return {"miv_calls_by_m": dict(sorted(self.miv_by_m.items())),
                "fallback_calls_by_reason": dict(sorted(self.fallback_by_reason.items()))}


STATS = Stats()
_ENABLED = True


def set_enabled(flag: bool) -> None:
    """Toggle installed wrappers between MIV-GEMV and ``F.linear`` (A/B in one process)."""
    global _ENABLED
    _ENABLED = bool(flag)


def _eligible(x: torch.Tensor, weight: torch.Tensor) -> Optional[str]:
    if not (x.is_cuda and weight.is_cuda):
        return "not_cuda"
    if weight.dtype != torch.bfloat16 or weight.dim() != 2:
        return "weight_not_bf16_2d"
    k = int(weight.shape[1])
    if k % 8:
        return "k_not_multiple_of_8"
    m = x.numel() // max(1, x.shape[-1])
    if m < 1 or m > MAX_M:
        return f"m_{'gt8' if m > MAX_M else 'lt1'}"
    if not weight.is_contiguous():
        return "weight_not_contiguous"
    return None


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
           ) -> torch.Tensor:
    """``F.linear`` semantics; MIV-GEMV when 1 <= M <= 8, else ``F.linear``."""
    why = "disabled" if not _ENABLED else _eligible(x, weight)
    if why is not None:
        STATS.fallback_by_reason[why] = STATS.fallback_by_reason.get(why, 0) + 1
        return F.linear(x, weight, bias)
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if x2.dtype != torch.bfloat16:
        x2 = x2.to(torch.bfloat16)
    b = None
    if bias is not None:
        b = bias if (bias.dtype == torch.bfloat16 and bias.is_contiguous()) else \
            bias.to(torch.bfloat16).contiguous()
    m = int(x2.shape[0])
    STATS.miv_by_m[m] = STATS.miv_by_m.get(m, 0) + 1
    y = miv_gemv(x2, weight, b)
    return y.reshape(*lead, weight.shape[0]).to(x.dtype)


def install_dense(root: nn.Module, *, prefix_filter: Optional[Iterable[str]] = None) -> int:
    """Route every bf16 ``nn.Linear`` under ``root`` through :func:`linear`.

    ``prefix_filter`` keeps only modules whose qualified name starts with one
    of the prefixes (e.g. skip the vision tower).  Returns the count wrapped.
    """
    prefixes = tuple(prefix_filter) if prefix_filter else None
    n = 0
    for name, mod in root.named_modules():
        if type(mod) is not nn.Linear:
            continue
        if prefixes and not name.startswith(prefixes):
            continue
        if getattr(mod, "_miv_installed", False):
            continue

        def fwd(x, _m=mod):
            return linear(x, _m.weight, _m.bias)

        mod.forward = fwd
        mod._miv_installed = True
        n += 1
    return n


def install_tbe_exact() -> None:
    """Make the TBE ``exact`` path's GEMM this kernel (decode-then-MIV)."""
    from . import modules

    modules.set_exact_linear(linear)


def uninstall_tbe_exact() -> None:
    from . import modules

    modules.set_exact_linear(None)


__all__ = ["CONFIG", "MAX_M", "STATS", "config_for", "extension", "install_dense",
           "install_tbe_exact", "linear", "load_config", "miv_gemv", "set_enabled",
           "uninstall_tbe_exact"]
