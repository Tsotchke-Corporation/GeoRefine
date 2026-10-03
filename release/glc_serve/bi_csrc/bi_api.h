// BI-GEMM host API (plain C++; no device code).  Contract: bi_gemm.cuh.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

enum BiFmt { BI_BF16 = 0, BI_Q8 = 1, BI_TBE = 2, BI_KQ = 3 };

struct BiDesc {
  int N, K;
  const __nv_bfloat16* w;                                   // BF16 [N][K]
  const int8_t* qs; const unsigned short* sc;               // Q8_0 SoA
  const uint32_t* planes; const uint8_t* smb; const uint8_t* esc;
  const int32_t* rowparam; const int32_t* escidx;           // TBE (escidx [N][K/128])
  const uint8_t* a0; const uint8_t* a1; const uint8_t* a2; const uint8_t* a3; const uint8_t* a4;
  const uint32_t* grid; const uint8_t* ksigns;              // KQ (miv_kq.h planes)
  const __nv_bfloat16* bias;
};

cudaError_t bi_gemm_dense(int fmt, const __nv_bfloat16* x, int64_t ldx, int M, const BiDesc& d,
                          int S, __nv_bfloat16* y, int64_t ldy, float* ws, cudaStream_t st);
cudaError_t bi_gemm_kq(int kqt, const __nv_bfloat16* x, int64_t ldx, int M, const BiDesc& d,
                       int S, __nv_bfloat16* y, int64_t ldy, float* ws, cudaStream_t st);
