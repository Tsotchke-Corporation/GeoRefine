// BI-GEMM host API (plain C++; no device code).  Contract: bi_gemm.cuh.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef GEOR_REFINED_CONTEXT_ONLY
enum BiFmt { BI_BF16 = 0, BI_Q8 = 1, BI_TBE = 2, BI_TBE2 = 4, BI_TBE21 = 5 };
#else
enum BiFmt { BI_BF16 = 0, BI_Q8 = 1, BI_TBE = 2, BI_KQ = 3, BI_TBE2 = 4, BI_TBE21 = 5 };
#endif

struct BiDesc {
  int N, K;
  const __nv_bfloat16* w;                                   // BF16 [N][K]
  const int8_t* qs; const unsigned short* sc;               // Q8_0 SoA
  const uint32_t* planes; const uint8_t* smb; const uint8_t* esc;
  const int32_t* rowparam; const int32_t* escidx;           // TBE (escidx [N][K/128])
#ifndef GEOR_REFINED_CONTEXT_ONLY
  const uint8_t* a0; const uint8_t* a1; const uint8_t* a2; const uint8_t* a3; const uint8_t* a4;
  const uint32_t* grid; const uint8_t* ksigns;              // KQ (miv_kq.h planes)
#endif
  const __nv_bfloat16* bias;
  // TBE2 = GLC codec v2 (docs/research/CODEC_V2_FORMAT_20261004.md).  `planes` (u32
  // [N][nch][8]) and `smb` (u8 [N][K]) above carry the v2 planes / sign-mantissa streams; the
  // rest is here.  Appended, so every field above keeps its offset.
  const uint32_t* ovf;      // overflow bit stream (+ >= 16 zero tail words)
  const uint8_t* len;       // u8 [N][nchp] chunk region lengths (+ 16 zero tail bytes)
  const uint32_t* grp;      // u32 [N][ngrp] group word bases
  const uint8_t* cb;        // u8 [ncb][16] codebooks (cb[0..12], 3 zero bytes)
  const int32_t* rowp2;     // [N]: codebook index | log2(len_unit) << 8
  int nchp, ngrp, ncb;
  // TBE21 = GLC codec v2.1 (docs/research/CODEC_V21_FORMAT_20261004.md).  Reuses `planes` (the
  // lane-interleaved level-1 codes, u16 [N][nch][16]), `smb`, `ovf`, `cb` (u8 [ncb][16], 16
  // symbols), `rowp2` (codebook index per row) and `ncb` above; appended, so offsets above hold.
  const uint32_t* rowoff;   // u32 [N]: word index of each row's first chunk region
  const uint32_t* ckpt;     // u32 [N][ckS-1]: RESIDENT split-start bit offsets (row-relative)
  int ckS;                  // the split count the checkpoints were built for (must == launch S)
};

// Compiler/occupancy facts of one bi_gemm_kernel instantiation, read off the card
// (cudaFuncGetAttributes + cudaOccupancyMaxActiveBlocksPerMultiprocessor).
struct BiKernelAttrs {
  int regs, local_bytes, max_blocks_per_sm, smem_bytes, threads;
};

// `tile` is the activation-row tile height: 64 (the shipped default) or 128.  It changes how
// many rows share one in-kernel weight decode and NOTHING about the K reduction -- see the WG
// note in bi_gemm.cuh and the CPU proof in glc_serve/bigemm_tile_ref128.py.
cudaError_t bi_gemm_dense(int fmt, const __nv_bfloat16* x, int64_t ldx, int M, const BiDesc& d,
                          int S, __nv_bfloat16* y, int64_t ldy, float* ws, cudaStream_t st,
                          int tile);
// Decode-only twin of bi_gemm_kernel's weight path (same Drv<Ld<FMT>> start/load/dec, same
// CTA row mapping, same split traversal), writing the decoded bf16 rows to out[N][K] instead of
// feeding the MMA.  The G1 gate (bi_gate_tbe2.py) hashes its output against the census.
cudaError_t bi_decode_dense(int fmt, const BiDesc& d, int S, __nv_bfloat16* out, cudaStream_t st,
                            int wg);
cudaError_t bi_attrs_dense(int fmt, int mt, int wg, BiKernelAttrs* a);
// TBE21 (codec v2.1) load-time split checkpoints for launch split count S: out = u32 [N][S-1]
// (the decode walk at S = 1, recording the cursor at every split start).  No-op for S == 1.
cudaError_t bi_t21_checkpoints(const BiDesc& d, int S, uint32_t* out, cudaStream_t st);
#ifndef GEOR_REFINED_CONTEXT_ONLY
cudaError_t bi_gemm_kq(int kqt, const __nv_bfloat16* x, int64_t ldx, int M, const BiDesc& d,
                       int S, __nv_bfloat16* y, int64_t ldy, float* ws, cudaStream_t st,
                       int tile);
#endif
