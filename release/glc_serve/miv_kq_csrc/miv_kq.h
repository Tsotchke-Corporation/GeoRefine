// MIV-KQ: M-invariant skinny GEMM (M = 1..8) over llama.cpp K-quant / i-quant
// weights (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS), dequantized in registers.
//
//   y[m, n] = round_bf16( sum_k x[m, k] * Wq[n, k]  (+ bias[n]) ),   1 <= M <= 8
//   Wq[n, k] = bf16_rn( gguf-py dequantize(block)[k] )   (FP32 reference, then RNE)
//
// Arithmetic = bf16 MIV-GEMV verbatim (miv_gemv.py @ c155643be) with the SAME WPR
// per (N, K): output is bitwise bf16 MIV on the dequantized weights (G3a-q).
//
// (grids: one byte per value, 4 values per u32; 8-value grids take two u32 per entry)
// Weights stay in GGUF row order (row n = n-th GGUF row; K % 256 == 0) and are
// split ONCE at load into structure-of-arrays planes -- byte-for-byte the block
// payload, no re-quantization, no padding:
//   Q4_K    a0 = {fp16 d, fp16 dmin} u32[nb]  a1 = scales u8[nb*12]  a2 = qs u8[nb*128]
//   Q5_K    a0 = dm u32[nb]  a1 = scales[nb*12]  a2 = qh[nb*32]  a3 = qs[nb*128]
//   Q6_K    a0 = ql[nb*128]  a1 = qh[nb*64]  a2 = scales i8[nb*16]  a3 = d fp16[nb]
//   IQ4_XS  a0 = d fp16[nb]  a1 = scales_h u16[nb]  a2 = scales_l[nb*4]  a3 = qs[nb*128]
//   IQ3_XXS a0 = d[nb]  a1 = qs[nb*64]  a2 = scales+signs u32[nb*8];  grid u32[256], ksigns u8[128]
//   IQ3_S   a0 = d[nb]  a1 = qs[nb*64]  a2 = qh[nb*8]  a3 = signs[nb*32]  a4 = scales[nb*4]; grid u32[512]
//   Q2_K    a0 = scales[nb*16]  a1 = qs[nb*64]  a2 = {fp16 d, fp16 dmin} u32[nb]
//   Q3_K    a0 = hmask[nb*32]  a1 = qs[nb*64]  a2 = scales[nb*12]  a3 = d fp16[nb]
//   IQ2_XS  a0 = d[nb]  a1 = qs u16[nb*32]  a2 = scales[nb*8];  grid u32[512*2], ksigns u8[128]
//   IQ2_S   a0 = d[nb]  a1 = qs[nb*32]  a2 = signs[nb*32]  a3 = qh[nb*8]  a4 = scales[nb*8]; grid u32[1024*2]
// nb = N*K/256, super-block b = row*(K/256) + col/256.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ggml type ids
enum MivKqType { KQ_Q2_K = 10, KQ_Q3_K = 11, KQ_Q4_K = 12, KQ_Q5_K = 13, KQ_Q6_K = 14,
                 KQ_IQ2_XS = 17, KQ_IQ3_XXS = 18, KQ_IQ3_S = 21, KQ_IQ2_S = 22, KQ_IQ4_XS = 23 };

struct MivKqDesc {
  int32_t type;      // MivKqType
  int32_t n, k;
  int32_t wpr;       // 1, 2, 4, 8 -- SAME as bf16 MIV for (n, k)
  int32_t unroll;    // 1, 2, 4    -- load batching only
  int32_t rpw;       // rows per warp: 0/1 = 1, 2 = 2 (x unpack shared; identical bits)
  const uint8_t* a0;
  const uint8_t* a1;
  const uint8_t* a2;
  const uint8_t* a3;
  const uint8_t* a4;
  const uint32_t* grid;     // IQ3_* only
  const uint8_t* ksigns;    // IQ3_XXS only
  const __nv_bfloat16* bias;
};

cudaError_t miv_kq_gemv(const MivKqDesc& d, const __nv_bfloat16* x, int64_t ldx, int M,
                        __nv_bfloat16* y, int64_t ldy, cudaStream_t stream);
// decode-only twin: the FP32 values (pre-bf16) into w_out [n, k]
cudaError_t miv_kq_dequant_f32(const MivKqDesc& d, float* w_out, cudaStream_t stream);
