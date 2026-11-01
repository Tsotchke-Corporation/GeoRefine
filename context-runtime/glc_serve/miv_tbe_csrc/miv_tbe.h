// MIV-TBE: M-invariant skinny GEMM (M = 1..8) that reads GLC-TBE *coded*
// weights and reconstructs every bf16 weight in registers.  Public C++ API.
//
//   y[m, n] = round_bf16( sum_k x[m, k] * W[n, k]  (+ bias[n]) ),  1 <= M <= 8
//
// The arithmetic of every output element is, instruction for instruction, the
// arithmetic of bf16 MIV-GEMV (release/glc_serve/miv_gemv.py @ c155643be) run
// with the SAME (N, K, WPR): lane l of the warp owning (row, K-slice s) folds
// the 16-byte weight vectors v = v0 + l, v0 + l + 32, ... in increasing order,
// eight __fmaf_rn per vector in element order, then the fixed xor butterfly,
// then the WPR slice partials in slice order, then (optional) bias in FP32 and
// one bf16 rounding.  The decoded weight vector is bit-identical to the parent
// (the container is exact), so the output is bitwise equal to bf16 MIV on the
// parent weights -- G3a by construction, and tested per shape.
//
// WPR MUST equal the WPR the bf16 MIV reference uses for the same (N, K); the
// unroll U is free (it only batches loads).  M never changes the arithmetic.
//
// Weight descriptor (all device pointers; one descriptor may be the row-wise
// concatenation of several TBE tensors with different exponent windows):
//
//   planes    uint32 [N*K/64][3][2]  plane b of tile t, word j: bit (p & 31)
//                                    is bit b of the 3-bit code of stored
//                                    position p = 32*j + bit  (layout mma16)
//   smb       uint8  [N*K]           (sign << 7) | mantissa, stored order
//   esc       uint8  [E + 16]        raw exponents of code-0 elements, tile-
//                                    major then stored-position ascending;
//                                    4-byte aligned base, >= 16 zero pad bytes
//   rowparam  int32  [N]             base | mode << 8   (mode 0 = W7, 1 = W6Z)
//   escbase   int32  [N*WPR]         index in esc of the first escape of
//                                    K-slice s of row n (load-time prefix)
//   bias      bf16   [N] or nullptr
//
// Preconditions: K % (64*WPR) == 0; x rows 16-byte aligned, ldx % 8 == 0.
// No allocation, no host sync: safe inside CUDA-graph capture.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

struct MivTbeDesc {
  int32_t n;
  int32_t k;
  int32_t wpr;       // 1, 2, 4, 8 -- per-shape constant shared with bf16 MIV
  int32_t unroll;    // 1, 2, 4    -- load batching only
  int32_t version;   // decode variant: 2 = PRMT decode (default), 1 = scalar (reference draft)
  int32_t reserved;
  const uint32_t* planes;
  const uint8_t* smb;
  const uint8_t* esc;
  const int32_t* rowparam;
  const int32_t* escbase;
  const __nv_bfloat16* bias;
};

// y = x W^T (+ bias).  x: [M, k] (row stride ldx), y: [M, n] (row stride ldy).
cudaError_t miv_tbe_gemv(const MivTbeDesc& d, const __nv_bfloat16* x, int64_t ldx, int M,
                         __nv_bfloat16* y, int64_t ldy, cudaStream_t stream);

// Decode-only twin (same traversal, same decode function): writes W [n, k] bf16.
cudaError_t miv_tbe_decode(const MivTbeDesc& d, __nv_bfloat16* w_out, cudaStream_t stream);

// Load-time helper: escapes per 64-element tile, out[t] for t < T.
cudaError_t miv_tbe_tile_escapes(const uint32_t* planes, int64_t T, int32_t* out,
                                 cudaStream_t stream);
