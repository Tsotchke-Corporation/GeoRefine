#ifndef BI_T21_EXEC_LAYOUT
#define BI_T21_EXEC_LAYOUT 0
#endif
#ifndef BI_T21_TIER_PREFIX
#define BI_T21_TIER_PREFIX 0
#endif
#ifndef BI_T21_TPF_PREFETCH
#define BI_T21_TPF_PREFETCH 0
#endif
#ifndef BI_T21_TPF_WG1
#define BI_T21_TPF_WG1 0
#endif
#ifndef BI_T21_TPF_PREFETCH_WG1
#define BI_T21_TPF_PREFETCH_WG1 0
#endif
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
//   KQ    -- miv_kq::load<T> + miv_kq::deq_bf16<T> (bf16_rn of gguf-py's dequantize),
//   TBE2  -- Ld<F_TBE2> below: GLC codec v2 (docs/research/CODEC_V2_FORMAT_20261004.md),
//            the per-lane decode of codec_v2.lane_model_decode_chunk with a carried cursor
//            (CPU twin, bitwise: glc_serve/bigemm_tbe2_ref.py).
// So for G1-identical weights, the TBE-coded GEMM is bitwise the bf16 GEMM on the parent,
// and the K-quant GEMM is bitwise the bf16 GEMM on gguf-py's dequantized weights.  A decoder
// only ever writes the 128 x 64 bf16 tile Ws; the MMA loop never knows which one ran.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "bi_api.h"
#ifndef GEOR_REFINED_CONTEXT_ONLY
#include "miv_kq_kernels.cuh"
#endif
#include "miv_tbe_kernels.cuh"

namespace bi {

constexpr int BN = 64;          // weight rows per CTA
constexpr int CH = 128;         // K columns per chunk
constexpr int LDS = CH + 8;     // smem row stride in bf16 (272 B: ldmatrix conflict-free)
constexpr int MTILE = 64;       // activation rows per WARP GROUP (4 m16 tiles)
constexpr int MAXWG = 2;        // warp groups per CTA: 1 -> 64-row tile, 2 -> 128-row tile

#if BI_T21_EXEC_LAYOUT
// Experimental single-group instances keep each eight-bf16 vector contiguous
// while distributing its eight row providers across all32 shared-memory banks.
// Other instances retain the padded layout. No persistent weight bytes change.
template <bool XOR>
__device__ __forceinline__ __nv_bfloat16* exec_smem_ptr(__nv_bfloat16* base, int row, int col) {
  if constexpr (XOR) return base + row * CH + (col ^ ((row & 7) << 3));
  else return base + row * LDS + col;
}
#endif
#ifndef BI_T21_ASYNC_X
#define BI_T21_ASYNC_X 0
#endif
#ifndef BI_T21_ROW_GEOMETRY
#define BI_T21_ROW_GEOMETRY 0
#endif
#ifndef BI_T21_PIPE128
#define BI_T21_PIPE128 0
#endif
#ifndef BI_T21_WIDE128
#define BI_T21_WIDE128 0
#endif
static_assert(BI_T21_ASYNC_X == 0 || BI_T21_ASYNC_X == 1, "BI_T21_ASYNC_X must be 0 or 1");

// ---------------------------------------------------------------- the row tile (WG)
// WG is the number of 8-warp groups in one CTA.  Each group owns its own MT m16 tiles of
// activations and reads the SAME decoded weight tile out of shared memory, so one in-kernel
// weight decode serves 64 * WG activation rows instead of 64.  That is the ONLY thing WG
// changes.  Why it cannot change a single output bit -- the loop structure, read off this
// file:
//
//   * the K reduction lives in the `for (c = c0; c < c1; ++c)` loop and the
//     `for (kk = 0; kk < CH; kk += 16)` loop inside it.  Neither bound mentions M, MT, WG,
//     blockIdx.z or threadIdx: c0/c1 come from (s, S, K/CH) and S = S(N, K) alone
//     (glc_serve.bigemm.split_for takes no M), and CH/16 is a compile-time 8.
//   * the M direction lives ONLY in (a) the `for (mt = 0; mt < MT; ++mt)` loop, (b) the
//     warp-group index wg = warp >> 3, and (c) m0 = blockIdx.z * MTILE * WG.  Each of those
//     selects WHICH acc[mt][*] register file and WHICH Xs row an mma reads.  A given output
//     element (m, n) is accumulated into exactly one acc[mt][e] of exactly one thread, and
//     that accumulator is touched by exactly one mma per (c, kk) -- in increasing (c, kk).
//     So the fp32 partial-sum tree for (m, n) is: the hardware's fixed 16-wide mma tree,
//     chained over k-steps in increasing k, over chunks in increasing c, within one split;
//     then the S splits added in split order by bi_reduce_kernel; then bias; then one
//     __float2bfloat16_rn.  Raising WG renames the thread that holds the chain.  It does
//     not reorder, regroup, re-split or re-round it.
//   * mma.m16n8k16 reads only its own A row (one activation row) and B column (one weight
//     row).  Rows m' != m of the batch never enter m's accumulator, so a row riding in a
//     128-row tile gets the bits it would get alone.
//
// What WG DOES change, and the arithmetic for it:
//   * dynamic smem = (BN + MT*16*WG) * LDS * 2 bytes = 34,816 B at WG=1,MT=4;
//     52,224 B at WG=2,MT=4.  52,224 > the 48 KiB default cap, so the WG=2 kernel needs
//     cudaFuncAttributeMaxDynamicSharedMemorySize (set in launch_fmt).
//   * threads = 256 * WG, so the __launch_bounds__ minimum-CTAs drops 2 -> 1.  Warps per SM
//     is 2*8 = 16 at WG=1 and 1*16 = 16 at WG=2 -- UNCHANGED -- and the register budget per
//     thread is 65,536/(256*2) = 128 at WG=1 and 65,536/(512*1) = 128 at WG=2, also
//     unchanged.  acc[MT][4] stays 16 fp32 at MT=4 because MT is per warp group, not per
//     CTA; that is why the row tile is widened with warps and not with MT=8.
//   * smem per SM FALLS: 2 * 34,816 = 69,632 B at WG=1 vs 1 * 52,224 B at WG=2.
//   * weight bytes streamed per GEMM = (whole weight set) * ceil(M / (64*WG)).  That is the
//     entire point: see the dispatch rule in glc_serve.bigemm.tile_for.

#ifdef GEOR_REFINED_CONTEXT_ONLY
enum Fmt { F_BF16 = BI_BF16, F_Q8 = BI_Q8, F_TBE = BI_TBE, F_TBE2 = BI_TBE2, F_TBE21 = BI_TBE21, F_TBE1T = BI_TBE1T };
#else
enum Fmt { F_BF16 = BI_BF16, F_Q8 = BI_Q8, F_TBE = BI_TBE, F_KQ = BI_KQ, F_TBE2 = BI_TBE2, F_TBE21 = BI_TBE21, F_TBE1T = BI_TBE1T };
#endif

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
  #ifndef GEOR_REFINED_CONTEXT_ONLY
  // KQ
  miv_kq::Args kq;
  #endif
  const __nv_bfloat16* __restrict__ bias;
  // TBE2 (codec v2): planes -> `planes` (u32 [N][nch][8]), smb -> `smb` (u8 [N][K]), plus
  const uint32_t* __restrict__ t2_ovf;     // overflow words, >= 16 zero tail words
  const uint8_t* __restrict__ t2_len;      // u8 [N][nchp], 16 zero tail bytes
  const uint32_t* __restrict__ t2_grp;     // u32 [N][ngrp]
  const uint8_t* __restrict__ t2_cb;       // u8 [ncb][16]
  const int32_t* __restrict__ t2_rowp;     // [N] codebook index | log2(len_unit) << 8
  int t2_nchp, t2_ngrp, t2_ncb;
  // TBE21 (codec v2.1): l1 codes -> `planes` (u16 [N][nch][16]), smb, t2_ovf, t2_cb, t2_rowp, t2_ncb, plus
  const uint32_t* __restrict__ t21_rowoff;  // u32 [N] row word offsets
  const uint32_t* __restrict__ t21_ckpt;    // u32 [N][t21_ckS-1] resident split-start bit offsets
  int t21_ckS;
  // TBE1T (docs/serving/KERNEL_V2_SPEC_20261017.md sections 5-6): escidx16/rowsb REPLACE the
  // load-time-derived `escidx` above for tbe1t tensors; esc4 (W4E16, nullable) is a secondary
  // escape stream read only when rowparam's bit 24 is set for that row.
  const uint16_t* __restrict__ escidx16;  // u16 [N][K/128]
  const uint32_t* __restrict__ rowsb;     // u32 [N][nsb]
  const uint8_t* __restrict__ esc4;       // u8 [ceil(E/2)], nullable
  int nsb;
#if BI_T21_TIER_PREFIX
  const uint32_t* __restrict__ tpf_counts; // u32 [N][K/128], packed tier counts
#endif
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
// opledger addendum C2/F4(a): a signed-32 twin of ld_nc_u32, same inline-asm shape as every
// other ld_nc_* here -- unlike __ldg (a real compiler intrinsic, which needs the compiler to
// PROVE its operand is a global-space address, inserting a cvta.to.global at the use site
// whenever it cannot statically track that across a pointer that changes every call, e.g. an
// induction pointer), inline asm takes whatever address value it is given with no
// address-space proof obligation.  Replacing the one `__ldg` LdTbeV2 had (on the escidx
// induction pointer) with this removes that per-call cvta.
__device__ __forceinline__ int32_t ld_nc_i32(const void* p) {
  int32_t r;
  asm volatile("ld.global.nc.s32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
// KERNEL V2 opledger T1 (docs/serving/KERNEL_V2_OPLEDGER_20261018.md W3): explicit lop3.b32
// with immLut for the 2/3-input logic chains the v2 decode path LOP3-folds by hand instead of
// hoping ptxas fuses the OR/AND/NOT sequence -- it makes the CPU opcount gate honest about the
// real instruction now, on the card later.  immLut convention (PTX ISA, lop3.b32): bit i of
// lut is F(a_bit, b_bit, c_bit) for index i = (a_bit<<2)|(b_bit<<1)|c_bit -- i.e. the canonical
// a_lut=0xF0, b_lut=0xCC, c_lut=0xAA decomposition.  Placed here (inside the ld_nc_* span) so
// scripts/batchserve/bigemm_opcount.py's extraction compiles it to the real `lop3.b32`
// instruction, and scripts/batchserve/bigemm_v2_simt.py's extraction (which never cuts this
// span, same as ld_nc_u32 et al.) resolves the unqualified call to the host model
// bigemm_v2_simt/simt.h provides, by ordinary enclosing-namespace lookup.
template<uint32_t LUT>
__device__ __forceinline__ uint32_t lop3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;" : "=r"(r) : "r"(a), "r"(b), "r"(c), "n"(LUT));
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

#ifndef GEOR_REFINED_CONTEXT_ONLY
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
#endif

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

// ---------------------------------------------------------------- TBE1T (docs/serving/
// KERNEL_V2_SPEC_20261017.md sections 5-6; release/glc_loader/tbe1t_container.py).  A tbe1t
// tensor's planes/smb/esc are byte-identical to TBE v1 for the same input (encode_tbe1t calls
// encode_tbe verbatim); only the escape INDEX (escidx16+rowsb replace escidx) and, per tensor,
// the escape BYTES (esc4 nibbles + base2 replace esc, opt-in W4E16) differ.  This is a
// SEPARATE, NEW format -- F_TBE/Ld<F_TBE,...>/miv_tbe::decode_vec_v2 above are not edited by
// anything below.
//
// decode_vec_v2_tbe1t is a copy of miv_tbe::decode_vec_v2 (not decode_vec_v2_lop3: kept
// independent on purpose so the already-gated F_TBE opcount numbers above can never regress
// from a tbe1t-only change) with two differences: (1) m0/m1/pc0 are parameters, shared with the
// cnt/c0 computation the caller already did (same CSE argument as decode_vec_v2_lop3);
// (2) the tile_cnt > 8 rare path's escape-byte fetch branches on `w4e`: plain esc (byte) or
// esc4 (nibble, `tb` indexes NIBBLES: byte = esc4[idx>>1], nibble = byte >> (4*(idx&1))) plus
// base2 (spec 6.3's rare-path line, transcribed verbatim).  The fast-path window bytes (wl, wh)
// are ALREADY correct bf16-escape-byte-pairs by the time they reach this function regardless of
// esc vs esc4 -- that branch lives in the loader (Ld<F_TBE1T,...>/LdTbe1tV2 below), per spec
// 6.3 ("these (wl, wh) feed decode_vec_v2's fast path unchanged").
__device__ __forceinline__ uint4 decode_vec_v2_tbe1t(const uint2 p0, const uint2 p1,
                                                     const uint2 p2, const uint32_t (&s)[4],
                                                     uint32_t e01, uint32_t e23, uint32_t w_lo,
                                                     uint32_t w_hi, uint32_t tb, uint32_t tile_cnt,
                                                     const uint8_t* __restrict__ esc,
                                                     const miv_tbe::LaneC& c, uint32_t m0,
                                                     uint32_t m1, uint32_t pc0, bool w4e,
                                                     uint32_t base2,
                                                     const uint8_t* __restrict__ esc4) {
  const uint32_t code = miv_tbe::codes8(p0, p1, p2, c);
  const uint32_t e_lo = __byte_perm(e01, e23, code);
  const uint32_t e_hi = __byte_perm(e01, e23, code >> 16);
  const uint32_t escn = ~(code | (code >> 1) | (code >> 2)) & 0x11111111u;
  const uint32_t r0 = __popc(m0 & c.lm_a), r1 = __popc(m0 & c.lm_b);
  const uint32_t r2 = pc0 + __popc(m1 & c.lm_a), r3 = pc0 + __popc(m1 & c.lm_b);
  uint32_t eb_lo, eb_hi;
  if (tile_cnt <= 8u) {
    const uint32_t rs_lo = r0 * 0x0011u + r1 * 0x1100u + ((escn << 4) & 0x1010u);
    const uint32_t rs_hi = r2 * 0x0011u + r3 * 0x1100u + ((escn >> 12) & 0x1010u);
    eb_lo = __byte_perm(w_lo, w_hi, rs_lo);
    eb_hi = __byte_perm(w_lo, w_hi, rs_hi);
  } else {
    const uint32_t rr[4] = {r0, r1, r2, r3};
    eb_lo = 0u;
    eb_hi = 0u;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      if ((escn >> (4 * r)) & 1u) {
        const uint32_t rank = rr[r >> 1] + ((r & 1) ? ((escn >> (4 * (r - 1))) & 1u) : 0u);
        const uint32_t idx = tb + rank;
        // spec 6.3 rare path, verbatim: w4e rows read a NIBBLE (idx indexes nibbles into esc4);
        // plain rows read the BYTE (idx indexes bytes into esc), exactly as v1/F_TBE.
        const uint32_t e = w4e
            ? base2 + (((uint32_t)__ldg(esc4 + (idx >> 1)) >> (4u * (idx & 1u))) & 0xFu)
            : (uint32_t)__ldg(esc + idx);
        if (r < 4) eb_lo |= e << (8 * r);
        else eb_hi |= e << (8 * (r - 4));
      }
    }
  }
  const uint32_t x_lo = __byte_perm(e_lo, eb_lo, 0x3210u | ((escn & 0x1111u) << 2));
  const uint32_t x_hi = __byte_perm(e_hi, eb_hi, 0x3210u | ((escn >> 14) & 0x4444u));
  uint4 w;
  w.x = (__byte_perm(s[0], 0u, 0x1100u) & 0x807F807Fu) | (__byte_perm(x_lo, 0u, 0x1404u) >> 1);
  w.y = (__byte_perm(s[1], 0u, 0x1100u) & 0x807F807Fu) | (__byte_perm(x_lo, 0u, 0x3424u) >> 1);
  w.z = (__byte_perm(s[2], 0u, 0x1100u) & 0x807F807Fu) | (__byte_perm(x_hi, 0u, 0x1404u) >> 1);
  w.w = (__byte_perm(s[3], 0u, 0x1100u) & 0x807F807Fu) | (__byte_perm(x_hi, 0u, 0x3424u) >> 1);
  return w;
}

// v1-scheduled Ld<F_TBE1T,...>: same Drv<L> generic adapter as F_TBE (a pure function of
// row/kk), so tbe1t serves under variant 1 and G1 gates it without v2 in the loop (spec 5.2).
template <int KQT>
struct Ld<F_TBE1T, KQT> {
  struct R { uint2 p0, p1, p2; uint32_t s[4]; uint32_t eb; uint32_t rp; };
  static __device__ __forceinline__ R load(const WDesc& d, int row, int kk, int lane) {
    R r;
    const int j = lane & 15;
    const int64_t t = (int64_t)row * (d.K >> 6) + (kk >> 6);
    const uint32_t* pp = d.planes + t * 6;
    r.p0 = ld_nc_u2(pp); r.p1 = ld_nc_u2(pp + 2); r.p2 = ld_nc_u2(pp + 4);
    const uint8_t* sp = d.smb + t * 64 + 2 * (j & 7);
#pragma unroll
    for (int q = 0; q < 4; ++q) r.s[q] = ld_nc_u16(sp + 16 * q);
    // D1 (spec 5.2): tb = rowsb[row][c>>5] + escidx16[row][c] -- two INDEPENDENT __ldg, no
    // plane dependency (replaces F_TBE's single escidx lookup).
    const int c = kk >> 7, sb = c >> 5;
    const uint32_t e16 = (uint32_t)__ldg(d.escidx16 + (int64_t)row * (d.K >> 7) + c);
    const uint32_t rb = (uint32_t)__ldg(d.rowsb + (int64_t)row * d.nsb + sb);
    r.eb = rb + e16;
    r.rp = (uint32_t)__ldg(d.rowparam + row);
    return r;
  }
  static __device__ __forceinline__ uint4 dec(const WDesc& d, const R& r, int, int, int lane, int) {
    const int j = lane & 15;
    const uint32_t m0 = ~(r.p0.x | r.p1.x | r.p2.x), m1 = ~(r.p0.y | r.p1.y | r.p2.y);
    const uint32_t pc0 = __popc(m0), cnt = pc0 + __popc(m1);
    const uint32_t c0 = __shfl_sync(0xffffffffu, cnt, lane & 16);
    const uint32_t tb = r.eb + (j >= 8 ? c0 : 0u);
    const bool w4e = ((r.rp >> 24) & 1u) != 0u;
    const uint32_t base2 = (r.rp >> 16) & 0xffu;
    uint32_t wl, wh;
    if (w4e) {
      // spec 6.2/6.3: esc4 nibble window, `tb` indexes NIBBLES.
      const uint32_t on = tb & 7u;
      const uint32_t* e4 = reinterpret_cast<const uint32_t*>(d.esc4) + (tb >> 3);
      const uint32_t a0 = ld_nc_u32(e4), a1 = ld_nc_u32(e4 + 1);
      const uint32_t v = __funnelshift_r(a0, a1, on * 4u);
      const uint32_t ev = v & 0x0F0F0F0Fu, od = (v >> 4) & 0x0F0F0F0Fu;
      const uint32_t bb = base2 * 0x01010101u;
      wl = __byte_perm(ev, od, 0x5140u) + bb;
      wh = __byte_perm(ev, od, 0x7362u) + bb;
    } else {
      const uint32_t* e4 = reinterpret_cast<const uint32_t*>(d.esc) + (tb >> 2);
      const uint32_t d0 = ld_nc_u32(e4), d1 = ld_nc_u32(e4 + 1), d2 = ld_nc_u32(e4 + 2);
      const uint32_t sh = (tb & 3u) * 8u;
      wl = __funnelshift_r(d0, d1, sh); wh = __funnelshift_r(d1, d2, sh);
    }
    uint32_t e01, e23;
    miv_tbe::exp_table(r.rp, e01, e23);
    const miv_tbe::LaneC c = miv_tbe::lane_const(j);
    return decode_vec_v2_tbe1t(r.p0, r.p1, r.p2, r.s, e01, e23, wl, wh, tb, cnt, d.esc, c, m0, m1,
                               pc0, w4e, base2, d.esc4);
  }
};

// ---------------------------------------------------------------- TBE2 (GLC codec v2)
// Frozen format: docs/research/CODEC_V2_FORMAT_20261004.md (section 2-4).  Executable spec:
// glc_loader/codec_v2.py lane_model_decode_chunk (decode) and kernel_chunk_offset /
// kernel_next_offset (cursor).  CPU twin of THIS code, transcribed line for line and checked
// bitwise against the source words of every tensor it is run on: glc_serve/bigemm_tbe2_ref.py.
//
// Per row, per 128-column chunk, one half-warp (16 lanes, lane j = columns 8j .. 8j+7):
//   load(c)  2 x u8 plane bytes (j, 16+j), 8 sign|mantissa bytes, ovf window word ow+j
//   dec(c)   level 1 (escape byte = p0 & p1) -> scan -> level-2 digits (one window get) ->
//            scan -> level-3 digits (one get) -> [warp-uniform: scan -> raw bytes (two gets)]
//            -> symbol nibbles -> exponent by PRMT from the row's 16-byte codebook in SMEM ->
//            splice with sign|mantissa -> uint4 of 8 bf16; cursor += region (rounded to the
//            length unit), word-aligned when chunk c+1 opens a 16-chunk group.
//   start(c0) the ONLY index read: 4 x u32 of `len` (byte-masked, __dp4a) + one `grp` word, at
//            the split start.  In steady state the window address is the carried cursor.
//
// TABLE-DRIVEN on purpose: the exponent alphabet is data (`cb`, per tensor, in shared memory),
// the code structure (2|2|3 + raw) is the only thing compiled in.  A code-family change that
// keeps "symbol nibble -> exponent byte" (e.g. a different codebook policy, a larger table)
// changes the table, not this loop.
constexpr int T2_MAXCB = 8;                    // codebooks per launch (fused q|k|v, gate|up, ...)
constexpr int T2_TAB_BYTES = 16 * T2_MAXCB;    // 128 B of dynamic smem, TBE2 launches only

// Instruction-count builds only (scripts/batchserve/bigemm_opcount.py): -DBI_OPCOUNT_COMMON_PATH
// compiles the rare branches (raw escapes, slow-path window reads) out, so the static SASS of a
// loop body is the per-chunk common path.  Never defined in a real build.
#ifdef BI_OPCOUNT_COMMON_PATH
#define BI_RARE(x) false
#else
#define BI_RARE(x) (x)
#endif

// =================================================================================================
// KERNEL V2 (docs/serving/KERNEL_V2_SPEC_20261017.md): phase-split TBE decode (dec_a/dec_b),
// an optional speculative escape-window fast path (SPEC=1), and a double-buffered chunk pipeline
// (bi_gemm_kernel_v2 below).  F_BF16 and F_TBE only -- variant_for (bigemm_tune.py) never routes
// anything else here.  v1 (Ld<F_TBE, KQT>, bi_gemm_kernel, bi_decode_kernel above) is UNTOUCHED:
// not one of its lines is edited by this block, so its PTX cannot change (G-D).
//
// v2 changes WHERE bytes come from and WHEN loads issue -- never the decoded bytes, the
// per-element mma chain, the split traversal, the kk order, or the epilogue rounding (spec
// contract 0.1).  Bitwise proof sketch (spec 2.3): for every (row, chunk, lane) the bytes that
// reach decode_vec_v2 are (wl, wh) = esc[tb .. tb+8) under BOTH SPEC paths -- fast path by the
// coverage condition below, fallback by being v1's three-dword reload verbatim -- and
// (tb, cnt, esc) are path-independent, so decode_vec_v2 (a pure function of its arguments)
// returns the identical uint4 either way.

// cp.async helpers (sm_80+; LDGSTS).  The fallback body keeps the TU compilable on older
// arches but is never dispatched there (variant_for gates v2 to tile==64, bf16/tbe only, and the
// extension only ever builds for the active device's own arch).  Zero-fill proof: v1 stores
// make_uint4(0,0,0,0) for rows >= M (bi_gemm_kernel, "rows >= M -> zero"); cp.async with a
// src-size of 0 writes 16 zero bytes to the destination -- identical Xs contents either way.
__device__ __forceinline__ void cp_async_cg_16(void* dst, const void* src, bool valid) {
#if __CUDA_ARCH__ >= 800
  const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  const unsigned n = valid ? 16u : 0u;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(d), "l"(src), "r"(n));
#else
  uint4 v = make_uint4(0u, 0u, 0u, 0u);
  if (valid) v = __ldg(reinterpret_cast<const uint4*>(src));
  *reinterpret_cast<uint4*>(dst) = v;
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n");
#endif
}
__device__ __forceinline__ void cp_async_wait_all() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n");
#endif
}

#if BI_T21_ASYNC_X
// Schedule1 only: stage into the existing Xs while the unchanged weight decode
// runs. The source pointer remains in-bounds even for a zero-filled tail row.
__device__ __forceinline__ void cp_async_t21_x16(void* dst, const void* src, bool valid) {
#if __CUDA_ARCH__ >= 800
  const unsigned shared = (unsigned)__cvta_generic_to_shared(dst);
  if (valid) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src) : "memory");
  } else {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, 0;" :: "r"(shared), "l"(src) : "memory");
  }
#else
  uint4 v = make_uint4(0u, 0u, 0u, 0u);
  if (valid) v = __ldg(reinterpret_cast<const uint4*>(src));
  *reinterpret_cast<uint4*>(dst) = v;
#endif
}
__device__ __forceinline__ void cp_async_t21_x_commit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}
__device__ __forceinline__ void cp_async_t21_x_wait() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}
#endif

// opledger T1/T5 (W3/W4): a v2-OWNED copy of miv_tbe::decode_vec_v2 (miv_tbe_kernels.cuh is
// untouched -- v1 and the MIV family keep calling the original, byte for byte).  Two trims vs
// the original: (1) m0/m1 (the "escaped?" plane masks) and the escn merge are explicit
// lop3.b32 3-input NOR/OR instead of OR+OR+NOT chains; (2) m0/m1/pc0 are taken as PARAMETERS
// (computed once in dec_a, shared with the cnt/c0 computation already needed there) instead of
// being recomputed a second time from p0/p1/p2.  Every other line is copied verbatim; the
// bitwise argument is CSE + a boolean-identity substitution, not a new algorithm.
__device__ __forceinline__ uint32_t codes8_lop3(const uint2 p0, const uint2 p1, const uint2 p2,
                                                 const miv_tbe::LaneC& c) {
  const uint32_t z0 = __byte_perm(p0.x, p0.y, c.psel) >> c.zs;
  const uint32_t z1 = __byte_perm(p1.x, p1.y, c.psel) >> c.zs;
  const uint32_t z2 = __byte_perm(p2.x, p2.y, c.psel) >> c.zs;
  // opledger addendum C2/F3 (INT -3, -1/plane): pre-masking the first term turns the
  // AND+AND+OR merge into one lop3, `a | (b & c)` (LUT 0xF8, same identity as F1). Verified
  // exhaustively (per-byte x 4 lanes + 1e7 random u32; .scratch/kernel-v2-20261017/
  // verify_folds.py) before this edit.
  const uint32_t n0 = lop3<0xF8u>(z0 & 0x01010101u, z0 << 3, 0x10101010u);
  const uint32_t n1 = lop3<0xF8u>(z1 & 0x01010101u, z1 << 3, 0x10101010u);
  const uint32_t n2 = lop3<0xF8u>(z2 & 0x01010101u, z2 << 3, 0x10101010u);
  return lop3<0xFEu>(n0, n1 << 1, n2 << 2);   // n0 | (n1<<1) | (n2<<2): pure 3-input OR
}
// opledger addendum C3/C5-5 (T6, OPTIONAL, card-decided): BI_V2_SPLICE_INT in {0,1,2,3,4} --
// how many of the splice's 4 output words take the all-INT arm (w.x, then w.y, then w.z, then
// w.w, in that order) instead of F1's 2-PRMT/word lop3 arm.  Per converted word: PRMT -2,
// INT +7..8 (the "t"/"e2" sub-expressions are NOT shared across words in this build -- each
// word recomputes its own, so the measured INT delta is the full per-word cost, not the
// partially-shared estimate in the addendum's C3 note).  Both arms are always compiled;
// default 0 (PRMT arm only, matching today's shipped behavior) -- the card's µs/call sweep
// decides whether any n > 0 is worth shipping.  Every arm value is SIMT-gated (G-A re-run at
// every knob value) and was verified against the F1 reference form by an exhaustive/random
// host sweep (.scratch/kernel-v2-20261017/verify_t6.py, 2e6 random (s,x) pairs per word
// shape) before this code was written.
#ifndef BI_V2_SPLICE_INT
#define BI_V2_SPLICE_INT 0
#endif
static_assert(BI_V2_SPLICE_INT >= 0 && BI_V2_SPLICE_INT <= 4, "BI_V2_SPLICE_INT must be 0..4");

__device__ __forceinline__ uint4 decode_vec_v2_lop3(const uint2 p0, const uint2 p1, const uint2 p2,
                                                     const uint32_t (&s)[4], uint32_t e01,
                                                     uint32_t e23, uint32_t w_lo, uint32_t w_hi,
                                                     uint32_t tb, uint32_t tile_cnt,
                                                     const uint8_t* __restrict__ esc,
                                                     const miv_tbe::LaneC& c, uint32_t m0,
                                                     uint32_t m1, uint32_t pc0) {
  const uint32_t code = codes8_lop3(p0, p1, p2, c);
  const uint32_t e_lo = __byte_perm(e01, e23, code);
  const uint32_t e_hi = __byte_perm(e01, e23, code >> 16);
  // opledger addendum C2/F5 (INT -1): the &0x11111111 mask is dead -- every consumer below
  // (rs_lo/rs_hi's <<4/>>12 &0x1010 terms, x_lo/x_hi's F2-folded selectors, and the rare
  // path's (escn>>4r)&1) reads only bits == 0 (mod 4), so garbage in the other bits of the
  // raw NOR output never reaches a consumer regardless.  Verified exhaustively (256 flag
  // combos x 10,000 garbage fills per consumer, incl. the rare-path arithmetic;
  // .scratch/kernel-v2-20261017/verify_folds.py) before this edit, plus SIMT G-A (which
  // exercises the tile_cnt>8 rare path via the all_escape_tile/w6z_specials fixtures).
  const uint32_t escn = lop3<0x01u>(code, code >> 1, code >> 2);   // ~(a|b|c), unmasked
  const uint32_t r0 = __popc(m0 & c.lm_a), r1 = __popc(m0 & c.lm_b);
  const uint32_t r2 = pc0 + __popc(m1 & c.lm_a), r3 = pc0 + __popc(m1 & c.lm_b);
  uint32_t eb_lo, eb_hi;
  if (tile_cnt <= 8u) {
    const uint32_t rs_lo = r0 * 0x0011u + r1 * 0x1100u + ((escn << 4) & 0x1010u);
    const uint32_t rs_hi = r2 * 0x0011u + r3 * 0x1100u + ((escn >> 12) & 0x1010u);
    eb_lo = __byte_perm(w_lo, w_hi, rs_lo);
    eb_hi = __byte_perm(w_lo, w_hi, rs_hi);
  } else {
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
  // opledger addendum C2/F2 (INT -2): `(a & K1) | K2` with two IMMEDIATE operands folds into
  // one lop3 (LUT 0xEA = (A&B)|C). `(escn & 0x1111) << 2 == (escn << 2) & 0x4444` is a
  // shift-mask commute (0x1111 << 2 == 0x4444, no nibble carries) -- the hi selector already
  // had the shift done first, so it folds directly with no commute needed.  Verified
  // exhaustively (2e6 random u32 + 256 flag combos x 10,000 garbage fills;
  // .scratch/kernel-v2-20261017/verify_folds.py) before this edit.
  const uint32_t x_lo = __byte_perm(e_lo, eb_lo, lop3<0xEAu>(escn << 2, 0x4444u, 0x3210u));
  const uint32_t x_hi = __byte_perm(e_hi, eb_hi, lop3<0xEAu>(escn >> 14, 0x4444u, 0x3210u));
  uint4 w;
  // opledger addendum C2/F1 (INT -4, -1/word): the exponent term (bits 14..7/30..23) and the
  // sign|mantissa keep-mask (bits 15,6..0/31,22..16) are disjoint, so `(b & c) | a` folds into
  // one lop3 instead of AND then OR.  LUT 0xF8 = a | (b & c) (A=0xF0,B=0xCC,C=0xAA: 0xF0 |
  // (0xCC & 0xAA) = 0xF0 | 0x88 = 0xF8) -- verified exhaustively against the old expression
  // (2e6 random u32 pairs x 2 selectors, .scratch/kernel-v2-20261017/verify_folds.py) before
  // this edit, in addition to the SIMT G-A bitwise proof.
#if BI_V2_SPLICE_INT >= 1
  {
    const uint32_t t = lop3<0xF8u>(s[0] & 0xFFu, s[0] << 8, 0x00FF0000u);
    const uint32_t e2 = lop3<0xF8u>(x_lo & 0xFFu, x_lo << 8, 0x00FF0000u);
    w.x = lop3<0xFCu>(lop3<0xEAu>(t << 8, 0x80008000u, t & 0x007F007Fu), e2 << 7, 0u);
  }
#else
  w.x = lop3<0xF8u>(__byte_perm(x_lo, 0u, 0x1404u) >> 1, __byte_perm(s[0], 0u, 0x1100u), 0x807F807Fu);
#endif
#if BI_V2_SPLICE_INT >= 2
  {
    const uint32_t t = lop3<0xF8u>(s[1] & 0xFFu, s[1] << 8, 0x00FF0000u);
    const uint32_t xh = x_lo >> 16;
    const uint32_t e2 = lop3<0xF8u>(xh & 0xFFu, xh << 8, 0x00FF0000u);
    w.y = lop3<0xFCu>(lop3<0xEAu>(t << 8, 0x80008000u, t & 0x007F007Fu), e2 << 7, 0u);
  }
#else
  w.y = lop3<0xF8u>(__byte_perm(x_lo, 0u, 0x3424u) >> 1, __byte_perm(s[1], 0u, 0x1100u), 0x807F807Fu);
#endif
#if BI_V2_SPLICE_INT >= 3
  {
    const uint32_t t = lop3<0xF8u>(s[2] & 0xFFu, s[2] << 8, 0x00FF0000u);
    const uint32_t e2 = lop3<0xF8u>(x_hi & 0xFFu, x_hi << 8, 0x00FF0000u);
    w.z = lop3<0xFCu>(lop3<0xEAu>(t << 8, 0x80008000u, t & 0x007F007Fu), e2 << 7, 0u);
  }
#else
  w.z = lop3<0xF8u>(__byte_perm(x_hi, 0u, 0x1404u) >> 1, __byte_perm(s[2], 0u, 0x1100u), 0x807F807Fu);
#endif
#if BI_V2_SPLICE_INT >= 4
  {
    const uint32_t t = lop3<0xF8u>(s[3] & 0xFFu, s[3] << 8, 0x00FF0000u);
    const uint32_t xh = x_hi >> 16;
    const uint32_t e2 = lop3<0xF8u>(xh & 0xFFu, xh << 8, 0x00FF0000u);
    w.w = lop3<0xFCu>(lop3<0xEAu>(t << 8, 0x80008000u, t & 0x007F007Fu), e2 << 7, 0u);
  }
#else
  w.w = lop3<0xF8u>(__byte_perm(x_hi, 0u, 0x3424u) >> 1, __byte_perm(s[3], 0u, 0x1100u), 0x807F807Fu);
#endif
  return w;
}

// v2 TBE loader: same bytes, split phases, optional speculative escape window.  SPEC == 0:
// dec_a issues the three dependent escape dwords (batched across NR rows by the kernel, one
// exposed latency per chunk instead of NR); SPEC == 1: load() prefetches a 16-byte escape
// window one chunk ahead and dec_a issues dependent loads only when the window cannot cover the
// upper tile (spec 2).  Decoded bytes identical in all cases (spec 2.3).
//
// opledger T2 (W2): per-row induction pointers (St), advanced by a FIXED per-chunk stride
// instead of re-deriving `row*(K>>6)+(kk>>6)` (a 64-bit multiply + two widenings) from scratch
// on every load() call.  start() computes the one-time initial address from the exact
// expression load() used to recompute every time; by induction the pointer value at chunk c+1
// is identical to that expression evaluated at kk = (c+1)*CH+8j, since every stride below is
// exactly that expression's delta for +1 chunk (planes: 2 tiles * 6 words = 12; smb: 2 tiles *
// 64 B = 128; escidx: +1 chunk).  Addresses only -- no arithmetic that reaches a decoded byte
// changes, so this is bit-safe by construction (SIMT-gated anyway, never assumed).
template <int KQT, int SPEC>
struct LdTbeV2 {
  struct St {
    const uint32_t* pp;   // d.planes + t*6, t = row*(K>>6) + (kk>>6)
    const uint8_t* sp;    // d.smb + t*64 + 2*(j&7)  (j baked in: fixed per lane for the kernel)
    const int32_t* ei;    // d.escidx + row*(K/CH) + (kk/CH)
    const uint8_t* esc;   // d.esc -- loop-invariant, carried so load() never re-reads WDesc
  };
  struct R {
    uint2 p0, p1, p2;        // planes, both words of the half's tile
    uint32_t s[4];           // sign|mantissa u16 pairs
    uint32_t eb;             // escidx: escapes before this chunk (absolute)
    uint32_t q0, q1, q2, q3; // SPEC==1: esc[eb & ~3 .. +16)  (unused when SPEC==0)
  };
  struct E {                 // dec_a -> dec_b carried state
    uint32_t tb, cnt;        // escape base of this lane's tile, tile escape count
    uint32_t m0, m1, pc0;    // T5: shared with decode_vec_v2_lop3 (no second recompute)
    uint32_t d0, d1, d2;     // escape window dwords (fallback or copied from q*)
    uint32_t sh;             // C4 (spec 3.3): shift dec_b needs, computed once in dec_a instead
                             // of recomputed there (+1 reg, -1 ternary/compare in dec_b).
  };
  static __device__ __forceinline__ St start(const WDesc& d, int row, int kk0, int lane) {
    St st;
    const int j = lane & 15;
    const int64_t t0 = (int64_t)row * (d.K >> 6) + (kk0 >> 6);
    st.pp = d.planes + t0 * 6;
    st.sp = d.smb + t0 * 64 + 2 * (j & 7);
    st.ei = d.escidx + (int64_t)row * (d.K / CH) + (kk0 / CH);
    st.esc = d.esc;
    return st;
  }
  static __device__ __forceinline__ R load(St& st, int lane) {
    R r;
    (void)lane;
    r.p0 = ld_nc_u2(st.pp); r.p1 = ld_nc_u2(st.pp + 2); r.p2 = ld_nc_u2(st.pp + 4);
#pragma unroll
    for (int q = 0; q < 4; ++q) r.s[q] = ld_nc_u16(st.sp + 16 * q);
    // opledger addendum C2/F4(a): ld_nc_i32 instead of __ldg -- st.ei is an INDUCTION pointer
    // (a different address every call), and __ldg's address-space proof obligation re-fires a
    // cvta.to.global at every such call; the inline-asm ld_nc_* family (already used for every
    // other load in this function) takes the address value as given, no proof needed.
    r.eb = (uint32_t)ld_nc_i32(st.ei);
    if constexpr (SPEC) {
      // opledger addendum C2/F4(b): byte-offset addressing (`eb & ~3u`) instead of word-offset
      // pointer arithmetic (`(uint32_t*)esc + (eb>>2)`) -- the same final address
      // ((eb>>2)*4 == eb&~3u for unsigned eb), avoiding the shr-then-implicit-scale-by-4 PTX
      // the old form emits.
      const uint32_t* e4 = reinterpret_cast<const uint32_t*>(st.esc + (r.eb & ~3u));
      r.q0 = ld_nc_u32(e4);     r.q1 = ld_nc_u32(e4 + 1);
      r.q2 = ld_nc_u32(e4 + 2); r.q3 = ld_nc_u32(e4 + 3);
    }
    st.pp += 12; st.sp += 128; st.ei += 1;   // T2: fixed per-chunk stride, induction only
    return r;
  }
  static __device__ __forceinline__ E dec_a(const WDesc& d, const R& r, int lane) {
    E e;
    const int j = lane & 15;
    // T1 (LOP3) + T5 (share with decode_vec_v2_lop3): m0/m1 computed ONCE via lop3(a,b,c,NOR)
    // instead of 2 OR + 1 NOT each, and carried into dec_b/decode_vec_v2_lop3 so that function
    // never recomputes them (nor the pc0 popcount) a second time.
    e.m0 = lop3<0x01u>(r.p0.x, r.p1.x, r.p2.x);   // ~(a|b|c)
    e.m1 = lop3<0x01u>(r.p0.y, r.p1.y, r.p2.y);
    e.pc0 = __popc(e.m0);
    e.cnt = e.pc0 + __popc(e.m1);
    const uint32_t c0 = __shfl_sync(0xffffffffu, e.cnt, lane & 16);
    e.tb = r.eb + (j >= 8 ? c0 : 0u);
    if constexpr (SPEC) {
      // o = byte offset of the lane's window inside the prefetched quad.  Lower half:
      // o = eb & 3 <= 3 (always covered).  Upper half: o = (eb & 3) + c0; covered iff
      // o <= 7 (wl: q[o>>2], q[o>>2+1]; wh: q[o>>2+1], q[o>>2+2]; o <= 7 -> o>>2 <= 1 ->
      // highest word index 3) (spec 2.1b).
      const uint32_t o = (r.eb & 3u) + (j >= 8 ? c0 : 0u);
      const bool fallback = BI_RARE(o > 7u && e.cnt <= 8u);
      if (fallback) {
        // F4(b): byte-offset addressing, same identity as load()'s SPEC quad above.
        const uint32_t* e4 = reinterpret_cast<const uint32_t*>(d.esc + (e.tb & ~3u));
        e.d0 = ld_nc_u32(e4); e.d1 = ld_nc_u32(e4 + 1); e.d2 = ld_nc_u32(e4 + 2);
      } else {
        const uint32_t s0 = (o >> 2) & 3u;            // 0 or 1 on the fast path
        e.d0 = s0 ? r.q1 : r.q0;
        e.d1 = s0 ? r.q2 : r.q1;
        e.d2 = s0 ? r.q3 : r.q2;
      }
      // C4: fold here instead of recomputing `o`/`fallback` in dec_b -- fast path shifts by
      // (o & 3), fallback by (tb & 3) (identical value there: fallback reloaded at tb >> 2).
      e.sh = (fallback ? (e.tb & 3u) : (o & 3u)) * 8u;
    } else {
      // F4(b): byte-offset addressing, same identity as above (SPEC==0 path).
      const uint32_t* e4 = reinterpret_cast<const uint32_t*>(d.esc + (e.tb & ~3u));
      e.d0 = ld_nc_u32(e4); e.d1 = ld_nc_u32(e4 + 1); e.d2 = ld_nc_u32(e4 + 2);
      e.sh = (e.tb & 3u) * 8u;
    }
    return e;
  }
  static __device__ __forceinline__ uint4 dec_b(const WDesc& d, const R& r, const E& e,
                                                int lane, uint32_t e01, uint32_t e23,
                                                const miv_tbe::LaneC& c) {
    const uint32_t wl = __funnelshift_r(e.d0, e.d1, e.sh), wh = __funnelshift_r(e.d1, e.d2, e.sh);
    return decode_vec_v2_lop3(r.p0, r.p1, r.p2, r.s, e01, e23, wl, wh, e.tb, e.cnt, d.esc, c,
                              e.m0, e.m1, e.pc0);
  }
};

// Uniform v2 decode-stage adapter the kernel drives: load() / dec_a() / dec_b(), the same three
// calls for every format.  F_BF16 degenerates (dec_a is a no-op; dec_b is the one-shot v1 dec());
// F_TBE is LdTbeV2.  No format besides these two is ever instantiated (variant_for gates it).
template <int FMT, int KQT, int SPEC>
struct D2Sel;

template <int KQT, int SPEC>
struct D2Sel<F_BF16, KQT, SPEC> {
  using L = Ld<F_BF16, KQT>;
  using R = typename L::R;
  struct E {};
  // T2 applies trivially to BF16 too: a single induction pointer, no multiply ever existed
  // here to strength-reduce, but the uniform St/start/load interface stays one shape for both
  // formats the kernel drives.
  struct St { const __nv_bfloat16* p; };
  static __device__ __forceinline__ St start(const WDesc& d, int row, int kk0, int) {
    St st; st.p = d.w + (int64_t)row * d.K + kk0; return st;
  }
  static __device__ __forceinline__ R load(St& st, int) {
    R r; r.v = ld_nc_u4(st.p); st.p += CH; return r;
  }
  static __device__ __forceinline__ E dec_a(const WDesc&, const R&, int) { return E{}; }
  static __device__ __forceinline__ uint4 dec_b(const WDesc& d, const R& r, const E&, int lane,
                                                uint32_t, uint32_t, const miv_tbe::LaneC&) {
    return L::dec(d, r, 0, 0, lane, 0);
  }
};
template <int KQT, int SPEC>
struct D2Sel<F_TBE, KQT, SPEC> {
  using L2 = LdTbeV2<KQT, SPEC>;
  using R = typename L2::R;
  using E = typename L2::E;
  using St = typename L2::St;
  static __device__ __forceinline__ St start(const WDesc& d, int row, int kk0, int lane) {
    return L2::start(d, row, kk0, lane);
  }
  static __device__ __forceinline__ R load(St& st, int lane) {
    return L2::load(st, lane);
  }
  static __device__ __forceinline__ E dec_a(const WDesc& d, const R& r, int lane) {
    return L2::dec_a(d, r, lane);
  }
  static __device__ __forceinline__ uint4 dec_b(const WDesc& d, const R& r, const E& e, int lane,
                                                uint32_t e01, uint32_t e23,
                                                const miv_tbe::LaneC& c) {
    return L2::dec_b(d, r, e, lane, e01, e23, c);
  }
};

// v2 TBE1T loader (spec 5.2/6.3): same phase split as LdTbeV2, but eb = rowsb + escidx16 (two
// independent __ldg, D1) instead of one escidx lookup, and a per-row w4e branch (spec 6.2's
// rowparam bit 24) between the plain-esc SPEC/fallback path (byte for byte LdTbeV2's) and a
// direct esc4 nibble-window fetch.  Scope note (handoff doc): unlike LdTbeV2, this loader does
// NOT carry T2's induction-pointer optimization or SPEC's speculative quad for w4e rows (spec
// 6.3 explicitly permits "SPEC=0 for esc4 rows... in round 1") -- it recomputes addresses from
// (row, kk) every call, like TBE's pre-T2 loader.  Correctness, not the opcount gate (which the
// opledger scopes to F_TBE only), is this format's bar.
template <int KQT, int SPEC>
struct LdTbe1tV2 {
  struct R {
    uint2 p0, p1, p2;
    uint32_t s[4];
    uint32_t eb, rp;
    uint32_t q0, q1, q2, q3;   // SPEC==1, plain-esc (non-w4e) rows only
  };
  struct E {
    uint32_t tb, cnt, m0, m1, pc0;
    uint32_t wl, wh;    // resolved fully in dec_a (esc fast/fallback AND esc4 nibble window)
  };
  static __device__ __forceinline__ R load(const WDesc& d, int row, int kk, int lane) {
    R r;
    const int j = lane & 15;
    const int64_t t = (int64_t)row * (d.K >> 6) + (kk >> 6);
    const uint32_t* pp = d.planes + t * 6;
    r.p0 = ld_nc_u2(pp); r.p1 = ld_nc_u2(pp + 2); r.p2 = ld_nc_u2(pp + 4);
    const uint8_t* sp = d.smb + t * 64 + 2 * (j & 7);
#pragma unroll
    for (int q = 0; q < 4; ++q) r.s[q] = ld_nc_u16(sp + 16 * q);
    const int c = kk >> 7, sb = c >> 5;
    const uint32_t e16 = (uint32_t)__ldg(d.escidx16 + (int64_t)row * (d.K >> 7) + c);
    const uint32_t rb = (uint32_t)__ldg(d.rowsb + (int64_t)row * d.nsb + sb);
    r.eb = rb + e16;
    r.rp = (uint32_t)__ldg(d.rowparam + row);
    if constexpr (SPEC) {
      const bool w4e = ((r.rp >> 24) & 1u) != 0u;
      if (!w4e) {
        const uint32_t* e4 = reinterpret_cast<const uint32_t*>(d.esc) + (r.eb >> 2);
        r.q0 = ld_nc_u32(e4);     r.q1 = ld_nc_u32(e4 + 1);
        r.q2 = ld_nc_u32(e4 + 2); r.q3 = ld_nc_u32(e4 + 3);
      }
    }
    return r;
  }
  static __device__ __forceinline__ E dec_a(const WDesc& d, const R& r, int lane) {
    E e;
    const int j = lane & 15;
    e.m0 = lop3<0x01u>(r.p0.x, r.p1.x, r.p2.x);
    e.m1 = lop3<0x01u>(r.p0.y, r.p1.y, r.p2.y);
    e.pc0 = __popc(e.m0);
    e.cnt = e.pc0 + __popc(e.m1);
    const uint32_t c0 = __shfl_sync(0xffffffffu, e.cnt, lane & 16);
    e.tb = r.eb + (j >= 8 ? c0 : 0u);
    const bool w4e = ((r.rp >> 24) & 1u) != 0u;
    const uint32_t base2 = (r.rp >> 16) & 0xffu;
    if (w4e) {
      const uint32_t on = e.tb & 7u;
      const uint32_t* e4 = reinterpret_cast<const uint32_t*>(d.esc4) + (e.tb >> 3);
      const uint32_t a0 = ld_nc_u32(e4), a1 = ld_nc_u32(e4 + 1);
      const uint32_t v = __funnelshift_r(a0, a1, on * 4u);
      const uint32_t ev = v & 0x0F0F0F0Fu, od = (v >> 4) & 0x0F0F0F0Fu;
      const uint32_t bb = base2 * 0x01010101u;
      e.wl = __byte_perm(ev, od, 0x5140u) + bb;
      e.wh = __byte_perm(ev, od, 0x7362u) + bb;
    } else if constexpr (SPEC) {
      const uint32_t o = (r.eb & 3u) + (j >= 8 ? c0 : 0u);
      const bool fallback = BI_RARE(o > 7u && e.cnt <= 8u);
      uint32_t d0, d1, d2;
      if (fallback) {
        const uint32_t* e4 = reinterpret_cast<const uint32_t*>(d.esc) + (e.tb >> 2);
        d0 = ld_nc_u32(e4); d1 = ld_nc_u32(e4 + 1); d2 = ld_nc_u32(e4 + 2);
      } else {
        const uint32_t s0 = (o >> 2) & 3u;
        d0 = s0 ? r.q1 : r.q0;
        d1 = s0 ? r.q2 : r.q1;
        d2 = s0 ? r.q3 : r.q2;
      }
      const uint32_t sh = (fallback ? (e.tb & 3u) : (o & 3u)) * 8u;
      e.wl = __funnelshift_r(d0, d1, sh); e.wh = __funnelshift_r(d1, d2, sh);
    } else {
      const uint32_t* e4 = reinterpret_cast<const uint32_t*>(d.esc) + (e.tb >> 2);
      const uint32_t d0 = ld_nc_u32(e4), d1 = ld_nc_u32(e4 + 1), d2 = ld_nc_u32(e4 + 2);
      const uint32_t sh = (e.tb & 3u) * 8u;
      e.wl = __funnelshift_r(d0, d1, sh); e.wh = __funnelshift_r(d1, d2, sh);
    }
    return e;
  }
  static __device__ __forceinline__ uint4 dec_b(const WDesc& d, const R& r, const E& e,
                                                int lane, uint32_t e01, uint32_t e23,
                                                const miv_tbe::LaneC& c) {
    const bool w4e = ((r.rp >> 24) & 1u) != 0u;
    const uint32_t base2 = (r.rp >> 16) & 0xffu;
    return decode_vec_v2_tbe1t(r.p0, r.p1, r.p2, r.s, e01, e23, e.wl, e.wh, e.tb, e.cnt, d.esc, c,
                               e.m0, e.m1, e.pc0, w4e, base2, d.esc4);
  }
};
// D2Sel<F_TBE1T,...>: unlike F_TBE's D2Sel, St carries plain (row, kk) -- LdTbe1tV2 is not
// pointer-strength-reduced (see its own comment) -- advanced by CH per load(), matching the
// uniform St/start/load(St&, lane) shape bi_gemm_kernel_v2/bi_decode_kernel_v2 drive generically.
template <int KQT, int SPEC>
struct D2Sel<F_TBE1T, KQT, SPEC> {
  using L3 = LdTbe1tV2<KQT, SPEC>;
  using R = typename L3::R;
  using E = typename L3::E;
  struct St { const WDesc* d; int row, kk; };
  static __device__ __forceinline__ St start(const WDesc& d, int row, int kk0, int) {
    St st; st.d = &d; st.row = row; st.kk = kk0; return st;
  }
  static __device__ __forceinline__ R load(St& st, int lane) {
    R r = L3::load(*st.d, st.row, st.kk, lane);
    st.kk += CH;
    return r;
  }
  static __device__ __forceinline__ E dec_a(const WDesc& d, const R& r, int lane) {
    return L3::dec_a(d, r, lane);
  }
  static __device__ __forceinline__ uint4 dec_b(const WDesc& d, const R& r, const E& e, int lane,
                                                uint32_t e01, uint32_t e23,
                                                const miv_tbe::LaneC& c) {
    return L3::dec_b(d, r, e, lane, e01, e23, c);
  }
};
// =================================================================================================


// Exclusive prefix sum over the 16 lanes of this half-warp (shfl_up ladder) + the half's total.
// Called by all 32 lanes (both halves scan independently: width 16).
__device__ __forceinline__ uint32_t t2_hscan16(uint32_t v, int j, uint32_t& tot) {
  uint32_t inc = v;
#pragma unroll
  for (int off = 1; off < 16; off <<= 1) {
    const uint32_t t = __shfl_up_sync(0xffffffffu, inc, off, 16);
    if (j >= off) inc += t;
  }
  tot = __shfl_sync(0xffffffffu, inc, 15, 16);
  return inc - v;
}

// Bits [q, q+w) of the overflow stream counted from word `ow` (w <= 32; w == 0 -> 0).  Window
// words 0..15 are the 16 lanes' `win` registers.  Both shuffles run on every lane (they are
// collective); a lane whose bit range ends past window word 15 -- a chunk region > 480 bits, the
// format's slow path -- replaces the two words with global loads.  A fast-path read whose high
// word index is 16 wraps to lane 0: its bits are above the mask, so the value is unaffected.
__device__ __forceinline__ uint32_t t2_get(const uint32_t* __restrict__ ovf, uint32_t ow, uint32_t win,
                                           uint32_t q, uint32_t w) {
  const uint32_t lo = q >> 5;
  uint32_t a = __shfl_sync(0xffffffffu, win, (int)(lo & 15u), 16);
  uint32_t b = __shfl_sync(0xffffffffu, win, (int)((lo + 1u) & 15u), 16);
  if (w != 0u && ((q + w - 1u) >> 5) > 15u) {
    a = __ldg(ovf + ow + lo);
    b = __ldg(ovf + ow + lo + 1u);
  }
  const uint32_t v = __funnelshift_r(a, b, q & 31u);
  return w >= 32u ? v : (v & ((1u << w) - 1u));
}

// ---- SWAR helpers (all lanes; no data-dependent control flow) ----
// nibble i (bit 4i) <- bit i of x, x < 256
__device__ __forceinline__ uint32_t t2_spread8(uint32_t x) {
  x = (x | (x << 12)) & 0x000F000Fu;
  x = (x | (x << 6)) & 0x03030303u;
  return (x | (x << 3)) & 0x11111111u;
}
// nibble i <- 2-bit field i of x (fields 0..7 in bits 0..15)
__device__ __forceinline__ uint32_t t2_spread2(uint32_t x) {
  x = __byte_perm(x, 0u, 0x4140u);                     // [x.b0, 0, x.b1, 0]
  x = (x | (x << 4)) & 0x0F0F0F0Fu;
  return (x | (x << 2)) & 0x33333333u;
}
// nibble i <- 3-bit field i of x (fields 0..7 in bits 0..23)
__device__ __forceinline__ uint32_t t2_spread3(uint32_t x) {
  x = (x & 0x00000FFFu) | ((x << 4) & 0x0FFF0000u);
  x = (x & 0x003F003Fu) | ((x << 2) & 0x3F003F00u);
  return (x & 0x07070707u) | ((x << 1) & 0x70707070u);
}
// every byte rotated right by one bit: e -> (e >> 1) | (e & 1) << 7
__device__ __forceinline__ uint32_t t2_rotr8x4(uint32_t x) {
  return ((x >> 1) & 0x7F7F7F7Fu) | ((x << 7) & 0x80808080u);
}
// four bf16 halves of two elements from (sign|mantissa bytes, rotated exponent bytes):
// X = [s_a, E_a, s_b, E_b], Y = [E_a, s_a, E_b, s_b]; word = (X & 0x7F7F7F7F) | (Y & 0x80808080)
__device__ __forceinline__ uint32_t t2_pair(uint32_t s, uint32_t e, uint32_t selx, uint32_t sely) {
  const uint32_t x = __byte_perm(s, e, selx), y = __byte_perm(s, e, sely);
  return (x & 0x7F7F7F7Fu) | (y & 0x80808080u);
}

// Merge selector for "replace the flagged bytes": flags F at bit 4i (i = 0..7).  PRMT reads only
// the selector's low 16 bits, so F * 4 + 0x3210 is, in its low half, 0x3210 + 4 * flag for
// elements 0..3 (no nibble carries: 4 + 3 < 8), and (F >> 14) + 0x3210 the same for elements 4..7.
__device__ __forceinline__ uint32_t t2_msel_lo(uint32_t f) { return f * 4u + 0x3210u; }
__device__ __forceinline__ uint32_t t2_msel_hi(uint32_t f) { return (f >> 14) + 0x3210u; }
// t2_get for a read of w < 32 bits (every common-path read): no w >= 32 select.
__device__ __forceinline__ uint32_t t2_get31(const uint32_t* __restrict__ ovf, uint32_t ow, uint32_t win,
                                             uint32_t q, uint32_t w) {
  const uint32_t lo = q >> 5;
  uint32_t a = __shfl_sync(0xffffffffu, win, (int)(lo & 15u), 16);
  uint32_t b = __shfl_sync(0xffffffffu, win, (int)((lo + 1u) & 15u), 16);
  if (BI_RARE(w != 0u && ((q + w - 1u) >> 5) > 15u)) {
    a = __ldg(ovf + ow + lo);
    b = __ldg(ovf + ow + lo + 1u);
  }
  return __funnelshift_r(a, b, q & 31u) & ((1u << w) - 1u);
}

// t2_get for a read that provably ends inside the 16-word window (no slow path): the level-2
// digits end at bit ob + 2 * n1 <= 31 + 2 * 128 = 287, so both words are window words (<= 9).
// 0 < w < 32 is not required: w == 0 yields 0.
__device__ __forceinline__ uint32_t t2_get_win(uint32_t win, uint32_t q, uint32_t w) {
  const uint32_t lo = q >> 5;
  const uint32_t a = __shfl_sync(0xffffffffu, win, (int)(lo & 15u), 16);
  const uint32_t b = __shfl_sync(0xffffffffu, win, (int)((lo + 1u) & 15u), 16);
  return __funnelshift_r(a, b, q & 31u) & ((1u << w) - 1u);
}

template <int KQT>
struct Ld<F_TBE2, KQT> {
  struct R { uint32_t pb; uint2 s8; uint32_t win; };   // plane bytes (p0 | p1 << 8), smb, window
  struct C { uint32_t ow, ob, rp; };                    // cursor word, bit in word, row param
  // codebook table: ncb x 16 bytes -> smem, once per CTA (a __syncthreads follows in the kernel)
  static __device__ __forceinline__ void prologue(const WDesc& d, uint32_t* tab) {
    if ((int)threadIdx.x < 4 * d.t2_ncb)
      tab[threadIdx.x] = __ldg(reinterpret_cast<const uint32_t*>(d.t2_cb) + threadIdx.x);
  }
  // split start: O(row, c) = 32 * grp[row, c >> 4] + unit * sum(len[row, 16g .. c-1])
  static __device__ __forceinline__ void start(const WDesc& d, int row, int c, int, C& k) {
    const int g = c >> 4, r = c & 15;
    const uint32_t* lw = reinterpret_cast<const uint32_t*>(d.t2_len + (int64_t)row * d.t2_nchp + 16 * g);
    uint32_t sum = 0u;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int keep = min(max(r - 4 * q, 0), 4);                   // len bytes of chunks < c
      const uint32_t m = keep >= 4 ? 0xffffffffu : ((1u << (8 * keep)) - 1u);
      sum = __dp4a(__ldg(lw + q) & m, 0x01010101u, sum);
    }
    k.rp = (uint32_t)__ldg(d.t2_rowp + row);
    const uint32_t bits = sum << ((k.rp >> 8) & 0xffu);
    k.ow = __ldg(d.t2_grp + (int64_t)row * d.t2_ngrp + g) + (bits >> 5);
    k.ob = bits & 31u;
  }
  static __device__ __forceinline__ R load(const WDesc& d, int row, int kk, int lane, const C& k) {
    R r;
    const int j = lane & 15, c = kk >> 7;
    const uint8_t* pl = reinterpret_cast<const uint8_t*>(d.planes) + ((int64_t)row * (d.K >> 7) + c) * 32;
    r.pb = (uint32_t)__ldg(pl + j) | ((uint32_t)__ldg(pl + 16 + j) << 8);
    r.s8 = __ldg(reinterpret_cast<const uint2*>(d.smb + (int64_t)row * d.K + kk));
    r.win = __ldg(d.t2_ovf + k.ow + (uint32_t)j);
    return r;
  }
  static __device__ __forceinline__ uint4 dec(const WDesc& d, const R& r, int, int kk, int, int j,
                                              C& k, const uint32_t* tab) {
    const uint32_t b0 = r.pb & 0xffu, b1 = (r.pb >> 8) & 0xffu, ob = k.ob;
    // level 1: code 3 is the escape
    const uint32_t e1 = b0 & b1;
    const uint32_t k1 = __popc(e1);
    uint32_t n1;
    const uint32_t rho1 = t2_hscan16(k1, j, n1);
    // level 2: this lane's escaped elements own digits rho1 .. rho1+k1-1 (2 bits each)
    const uint32_t f2 = t2_get(d.t2_ovf, k.ow, r.win, ob + 2u * rho1, 2u * k1);
    uint32_t sym = 0u, e2 = 0u, cnt = 0u;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t code = ((b0 >> i) & 1u) | (((b1 >> i) & 1u) << 1);
      const uint32_t d2 = (f2 >> (2u * cnt)) & 3u;
      const uint32_t esc = (e1 >> i) & 1u;
      const uint32_t s = esc ? 3u + d2 : code;
      e2 |= (esc & (uint32_t)(d2 == 3u)) << i;
      cnt += esc;
      sym |= s << (4 * i);
    }
    // level 3: after the half's n1 level-2 digits, 3 bits each
    const uint32_t k2 = __popc(e2);
    uint32_t n2;
    const uint32_t rho2 = t2_hscan16(k2, j, n2);
    const uint32_t f3 = t2_get(d.t2_ovf, k.ow, r.win, ob + 2u * n1 + 3u * rho2, 3u * k2);
    uint32_t e3 = 0u;
    cnt = 0u;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t d3 = (f3 >> (3u * cnt)) & 7u;
      const uint32_t esc = (e2 >> i) & 1u;
      if (esc) sym = (sym & ~(0xfu << (4 * i))) | ((6u + d3) << (4 * i));   // 6 + 7 = 13 = RAW
      e3 |= (esc & (uint32_t)(d3 == 7u)) << i;
      cnt += esc;
    }
    // raw exponent bytes: warp-uniform branch (19.2 % of warp-iterations on the 2B), 8 bits each
    // after the level-3 digits; a lane's <= 8 bytes are contiguous -> two 32-bit gets
    uint32_t n3 = 0u, g0 = 0u, g1 = 0u;
    if (__any_sync(0xffffffffu, e3)) {
      const uint32_t k3 = __popc(e3);
      const uint32_t rho3 = t2_hscan16(k3, j, n3);
      const uint32_t base = ob + 2u * n1 + 3u * n2 + 8u * rho3;
      g0 = t2_get(d.t2_ovf, k.ow, r.win, base, min(8u * k3, 32u));
      g1 = t2_get(d.t2_ovf, k.ow, r.win, base + 32u, 8u * k3 > 32u ? 8u * k3 - 32u : 0u);
    }
    // symbol -> exponent (PRMT on the row's codebook, from smem), splice with sign|mantissa
    const uint4 cb = *reinterpret_cast<const uint4*>(tab + 4u * (k.rp & 0xffu));
    uint32_t h[8];
    cnt = 0u;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t s = (sym >> (4 * i)) & 0xfu;
      uint32_t ex = (s < 8u ? __byte_perm(cb.x, cb.y, s & 7u) : __byte_perm(cb.z, cb.w, s & 7u)) & 0xffu;
      const uint32_t esc = (e3 >> i) & 1u;
      const uint32_t sh = 8u * cnt;
      const uint32_t rb = (sh < 32u ? (g0 >> (sh & 31u)) : (g1 >> ((sh - 32u) & 31u))) & 0xffu;
      ex = esc ? rb : ex;
      cnt += esc;
      const uint32_t sm = ((i < 4 ? r.s8.x : r.s8.y) >> (8 * (i & 3))) & 0xffu;
      h[i] = ((sm & 0x80u) << 8) | (ex << 7) | (sm & 0x7fu);
    }
    // cursor: o += region(c); when chunk c+1 opens a group, round up to the next word
    const uint32_t L = 2u * n1 + 3u * n2 + 8u * n3;
    const uint32_t ush = (k.rp >> 8) & 0xffu;
    const uint32_t t = ob + (((L + (1u << ush) - 1u) >> ush) << ush);
    k.ow += t >> 5;
    k.ob = t & 31u;
    if ((((kk >> 7) + 1) & 15) == 0 && k.ob) { k.ow += 1u; k.ob = 0u; }
    uint4 w;
    w.x = h[0] | (h[1] << 16); w.y = h[2] | (h[3] << 16);
    w.z = h[4] | (h[5] << 16); w.w = h[6] | (h[7] << 16);
    return w;
  }
};

// ---------------------------------------------------------------- decode driver
// The kernel calls weight decoders through Drv<L>.  Formats whose chunk address is a pure
// function of (row, chunk) -- BF16, Q8_0, TBE, KQ -- get the identity adapter: no cursor, no
// table, and the calls forward to the unchanged Ld<> (so their code is what rc7 compiled).
// TBE2 carries a per-row cursor across the chunks of a split and reads its codebooks from smem.
struct NoCur {};
template <class L>
struct Drv {
  using R = typename L::R;
  using C = NoCur;
  static constexpr int kTab = 0;
  static __device__ __forceinline__ void prologue(const WDesc&, uint32_t*) {}
  static __device__ __forceinline__ void start(const WDesc&, int, int, int, int, int, C&) {}
  static __device__ __forceinline__ R load(const WDesc& d, int row, int kk, int lane, const C&) {
    return L::load(d, row, kk, lane);
  }
  static __device__ __forceinline__ uint4 dec(const WDesc& d, const R& r, int row, int kk, int lane,
                                              int j, C&, const uint32_t*) {
    return L::dec(d, r, row, kk, lane, j);
  }
};
template <int KQT>
struct Drv<Ld<F_TBE2, KQT>> {
  using L = Ld<F_TBE2, KQT>;
  using R = typename L::R;
  using C = typename L::C;
  static constexpr int kTab = T2_TAB_BYTES;
  static __device__ __forceinline__ void prologue(const WDesc& d, uint32_t* tab) { L::prologue(d, tab); }
  static __device__ __forceinline__ void start(const WDesc& d, int row, int c, int, int, int lane, C& k) {
    L::start(d, row, c, lane, k);
  }
  static __device__ __forceinline__ R load(const WDesc& d, int row, int kk, int lane, const C& k) {
    return L::load(d, row, kk, lane, k);
  }
  static __device__ __forceinline__ uint4 dec(const WDesc& d, const R& r, int row, int kk, int lane,
                                              int j, C& k, const uint32_t* tab) {
    return L::dec(d, r, row, kk, lane, j, k, tab);
  }
};

// ---------------------------------------------------------------- TBE21 (GLC codec v2.1)
// Format: docs/research/CODEC_V21_FORMAT_20261004.md; executable spec glc_loader/codec_v21.py
// (lane_model_decode_chunk, split_checkpoints); CPU twin of THIS code: glc_serve/bigemm_tbe21_ref.py.
// Differences from TBE2 that the decode sees:
//   * code (2,2,2,3) + raw: one more rank-addressed level (L3: 2-bit digits, symbols 6..8);
//   * level-1 codes are one u16 per lane, lane-interleaved -> nibbles in one shift + one LOP3;
//   * no chunk index: the cursor advances by the exact region bits (no length unit, no group
//     word alignment); a split start reads rowoff[row] + the RESIDENT checkpoint of split s;
//   * the codebook table is 32 B per codebook (5 rotated words, see prologue), up to 8 per launch.
constexpr int T21_TAB_BYTES = 32 * T2_MAXCB;   // 256 B of dynamic smem, TBE21 launches only

// TBE21-only folds; shared TBE2 helpers remain unchanged in default builds.
__device__ __forceinline__ uint32_t t21_spread2(uint32_t x) {
  x = __byte_perm(x, 0u, 0x4140u);
  x = lop3<0xA8u>(x, x << 4, 0x0F0F0F0Fu);
  return lop3<0xA8u>(x, x << 2, 0x33333333u);
}
__device__ __forceinline__ uint32_t t21_spread3(uint32_t x) {
  x = lop3<0xF8u>(x & 0x00000FFFu, x << 4, 0x0FFF0000u);
  x = lop3<0xF8u>(x & 0x003F003Fu, x << 2, 0x3F003F00u);
  return lop3<0xF8u>(x & 0x07070707u, x << 1, 0x70707070u);
}
__device__ __forceinline__ uint32_t t21_rotr8x4(uint32_t x) {
  return lop3<0xD8u>(x >> 1, x << 7, 0x80808080u);
}
__device__ __forceinline__ uint32_t t21_pair(uint32_t s, uint32_t e, uint32_t selx, uint32_t sely) {
  const uint32_t x = __byte_perm(s, e, selx), y = __byte_perm(s, e, sely);
  return lop3<0xD8u>(x, y, 0x80808080u);
}
#ifndef BI_T21_SCAN
#define BI_T21_SCAN 0
#endif
#ifndef BI_T21_TABREG
#define BI_T21_TABREG 0
#endif
#ifndef BI_T21_V1_MINBLOCKS
#define BI_T21_V1_MINBLOCKS 2
#endif
__device__ __forceinline__ uint32_t t21_hscan16(uint32_t v, int lane, int j, uint32_t& tot) {
#if BI_T21_SCAN
  const uint32_t hb = (uint32_t)(lane & 16), hm = 0xFFFFu << hb;
  const uint32_t lt = ((1u << j) - 1u) << hb;
  uint32_t rho = 0u, t = 0u;
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    const uint32_t mb = (uint32_t)__ballot_sync(0xffffffffu, (v >> b) & 1u);
    rho += (uint32_t)__popc(mb & lt) << b;
    t += (uint32_t)__popc(mb & hm) << b;
  }
  tot = t;
  return rho;
#else
  (void)lane;
  return t2_hscan16(v, j, tot);
#endif
}

#ifndef BI_T21_PACKSCAN2
#define BI_T21_PACKSCAN2 0
#endif
#ifndef BI_T21_SCAN4
#define BI_T21_SCAN4 0
#endif
#ifndef BI_T21_PACKSCAN4
#define BI_T21_PACKSCAN4 0
#endif
static_assert(BI_T21_PACKSCAN4 == 0 || BI_T21_PACKSCAN4 == 1, "BI_T21_PACKSCAN4 must be 0 or 1");
static_assert(!BI_T21_PACKSCAN4 || BI_T21_PACKSCAN2, "PACKSCAN4 requires PACKSCAN2");
static_assert(!BI_T21_PACKSCAN4 || (!BI_T21_SCAN && !BI_T21_SCAN4 && !BI_T21_TABREG &&
    !BI_T21_ROW_GEOMETRY && !BI_T21_WIDE128 && !BI_T21_PIPE128 && !BI_T21_ASYNC_X),
    "PACKSCAN4 requires the synchronous common GEMM path and compact shared-table cursors");
#if BI_T21_PACKSCAN4
// Four rows, four byte fields; existing five-shuffle scan, not SCAN4.
__device__ __forceinline__ void t21_hscan16_quad(const uint32_t (&v)[4], int j,
    uint32_t (&rank)[4], uint32_t (&total)[4]) {
  uint32_t packed_total;
  const uint32_t packed = v[0] | (v[1] << 8) | (v[2] << 16) | (v[3] << 24);
  const uint32_t packed_rank = t2_hscan16(packed, j, packed_total);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    rank[i] = (packed_rank >> (8 * i)) & 0xffu;
    total[i] = (packed_total >> (8 * i)) & 0xffu;
  }
}

#endif

static_assert(BI_T21_SCAN4 == 0 || BI_T21_SCAN4 == 1, "BI_T21_SCAN4 must be 0 or 1");
static_assert(!BI_T21_SCAN4 || BI_T21_PACKSCAN2, "BI_T21_SCAN4 requires paired scan");
// Two independent rows in 16-bit fields.  Each lane contributes <= 8;
// every half-warp partial/total <= 128 < 65536, so additions never carry
// between fields. inc >= v fieldwise, so exclusive subtraction never borrows.
__device__ __forceinline__ void t21_hscan16_pair(uint32_t a, uint32_t b, int j,
    uint32_t& ra, uint32_t& rb, uint32_t& na, uint32_t& nb) {
  uint32_t total;
  const uint32_t rank = t2_hscan16(a | (b << 16), j, total);
  ra = rank & 0xffffu; rb = rank >> 16;
  na = total & 0xffffu; nb = total >> 16;
}

#if BI_T21_SCAN4
// Byte fields: [A group sum, A exclusive prefix, B group sum, B exclusive prefix].
// At stage off, the XOR peer owns the adjacent off-lane subtree. Its sum always
// extends our group sum and extends our prefix only when our lane is in the upper
// subtree. PRMT discards the peer's prefix fields. Each sum <= 128 and each
// prefix <= 120, so the single packed add cannot carry between bytes. All lanes
// reach four width16 collectives; no final broadcast or exclusive subtraction.
__device__ __forceinline__ void t21_hscan16_pair4(uint32_t a, uint32_t b, int j,
    uint32_t& ra, uint32_t& rb, uint32_t& na, uint32_t& nb) {
  uint32_t packed = a | (b << 16);
#pragma unroll
  for (int off = 1; off < 16; off <<= 1) {
    const uint32_t peer = __shfl_xor_sync(0xffffffffu, packed, off, 16);
    const uint32_t selector = (j & off) ? 0x2200u : 0x4240u;
    packed += __byte_perm(peer, 0u, selector);
  }
  ra = (packed >> 8) & 0xffu; rb = packed >> 24;
  na = packed & 0xffu; nb = (packed >> 16) & 0xffu;
}
#endif

#ifndef BI_T21_PAIR_VOTE
#define BI_T21_PAIR_VOTE 0
#endif
#if BI_T21_PAIR_VOTE
static_assert(BI_T21_PAIR_VOTE == 1, "BI_T21_PAIR_VOTE must be 0 or 1");
static_assert(BI_T21_PACKSCAN2 && BI_T21_TABREG && !BI_T21_SCAN4 && !BI_T21_SCAN &&
              !BI_T21_PACKSCAN4 && !BI_T21_ASYNC_X && !BI_T21_ROW_GEOMETRY &&
              !BI_T21_WIDE128 && !BI_T21_PIPE128,
              "PAIR_VOTE requires the paired register-table common GEMM path");
#if defined(__CUDA_ARCH__)
static_assert(__CUDA_ARCH__ == 1200, "PAIR_VOTE is initially qualified for SM120 only");
#endif
#include "pair_vote.cuh"
#endif
__device__ __forceinline__ void t21_scan_pair(uint32_t a, uint32_t b, int j,
    uint32_t& ra, uint32_t& rb, uint32_t& na, uint32_t& nb) {
#if BI_T21_SCAN4
  t21_hscan16_pair4(a, b, j, ra, rb, na, nb);
#else
  t21_hscan16_pair(a, b, j, ra, rb, na, nb);
#endif
}

#if BI_T21_TIER_PREFIX || BI_T21_TPF_PREFETCH || BI_T21_TPF_WG1 || BI_T21_TPF_PREFETCH_WG1
static_assert(BI_T21_TIER_PREFIX==0 || BI_T21_TIER_PREFIX==1, "tier prefix must be 0 or 1");
static_assert(BI_T21_TPF_PREFETCH==0 || BI_T21_TPF_PREFETCH==1, "tier prefetch must be 0 or 1");
static_assert(BI_T21_TPF_WG1==0 || BI_T21_TPF_WG1==1, "tier WG1 must be 0 or 1");
static_assert(BI_T21_TPF_PREFETCH_WG1==0 || BI_T21_TPF_PREFETCH_WG1==1, "tier WG1 prefetch must be 0 or 1");
static_assert(BI_T21_TIER_PREFIX || !(BI_T21_TPF_PREFETCH || BI_T21_TPF_WG1 || BI_T21_TPF_PREFETCH_WG1), "tier submodes require tier prefix");
static_assert(!BI_T21_TPF_PREFETCH_WG1 || (BI_T21_TPF_PREFETCH && BI_T21_TPF_WG1), "WG1 prefetch requires prefix WG1 and prefetch");
#if BI_T21_TIER_PREFIX
static_assert(BI_T21_PACKSCAN2 && BI_T21_TILE128 && BI_T21_TABREG && BI_T21_V1_MINBLOCKS==2 &&
  !BI_T21_SCAN && !BI_T21_SCAN4 && !BI_T21_PACKSCAN4 && !BI_T21_ASYNC_X &&
  !BI_T21_WIDE128 && !BI_T21_PIPE128 && !BI_T21_PAIR_VOTE,
  "tier prefix requires paired register-table schedule with incompatible experiments off");
#if defined(__CUDA_ARCH__)
static_assert(__CUDA_ARCH__==1200, "tier prefix is restricted to exact SM120 experiments");
#endif
#endif
#endif

template <int KQT>
struct Ld<F_TBE21, KQT> {
  struct R { uint32_t c16; uint2 s8; uint32_t win; };  // level-1 u16, smb, window word
#if BI_T21_SCAN4
  // Fixed-address streams can be prefetched before the current chunk has advanced
  // its overflow cursor. Keep the cursor-dependent window load at its original point.
  struct Fixed { uint32_t c16, s0, s1; };
  static __device__ __forceinline__ Fixed load_fixed(const WDesc& d, int row, int kk, int lane) {
    Fixed r;
    const int j = lane & 15, c = kk >> 7;
    const uint16_t* l1 = reinterpret_cast<const uint16_t*>(d.planes) + ((int64_t)row * (d.K >> 7) + c) * 16;
    r.c16 = (uint32_t)__ldg(l1 + j);
    const uint2 s8 = __ldg(reinterpret_cast<const uint2*>(d.smb + (int64_t)row * d.K + kk));
    r.s0 = s8.x; r.s1 = s8.y;
    return r;
  }
#endif
#if BI_T21_TABREG
  struct C { uint32_t ow, ob, rp; uint4 ta; uint32_t t4; };
#else
  struct C { uint32_t ow, ob, rp; };                     // cursor word, bit, codebook index
#endif
  struct CompactC { uint32_t ow, ob, rp; };
  // resident table per codebook (8 words, rotr8 bytes): w0 = cb0..cb3, w1 = cb4 cb5 0 0,
  // w2 = cb6 cb7 cb8 0, w3 = cb9..cb12, w4 = cb13 cb14 cb15 0, w5..w7 = 0.  Placeholder bytes
  // (escapes) are 0 and always overwritten.
  static __device__ __forceinline__ void prologue(const WDesc& d, uint32_t* tab) {
    if ((int)threadIdx.x < 8 * d.t2_ncb) {
      const uint32_t* src = reinterpret_cast<const uint32_t*>(d.t2_cb) + 4 * (threadIdx.x >> 3);
      const uint32_t q = threadIdx.x & 7u;
      const uint32_t c0 = __ldg(src), c1 = __ldg(src + 1), c2 = __ldg(src + 2), c3 = __ldg(src + 3);
      uint32_t v = 0u;
      if (q == 0u) v = c0;
      else if (q == 1u) v = c1 & 0x0000FFFFu;
      else if (q == 2u) v = __byte_perm(c1, c2, 0x0432u) & 0x00FFFFFFu;
      else if (q == 3u) v = __byte_perm(c2, c3, 0x4321u);
      else if (q == 4u) v = c3 >> 8;
      tab[threadIdx.x] = t21_rotr8x4(v);
    }
  }
  // split start: row word offset + (s == 0 ? 0 : resident checkpoint of split s)
  static __device__ __forceinline__ void start(const WDesc& d, int row, int s, int S, C& k) {
    const uint32_t bits = s == 0 ? 0u : __ldg(d.t21_ckpt + (int64_t)row * (S - 1) + (s - 1));
    k.rp = (uint32_t)__ldg(d.t2_rowp + row);
    k.ow = __ldg(d.t21_rowoff + row) + (bits >> 5);
    k.ob = bits & 31u;
#if BI_T21_TABREG
    const uint32_t* src = reinterpret_cast<const uint32_t*>(d.t2_cb) + 4 * k.rp;
    const uint32_t c0 = __ldg(src), c1 = __ldg(src + 1), c2 = __ldg(src + 2), c3 = __ldg(src + 3);
    k.ta.x = t21_rotr8x4(c0);
    k.ta.y = t21_rotr8x4(c1 & 0x0000FFFFu);
    k.ta.z = t21_rotr8x4(__byte_perm(c1, c2, 0x0432u) & 0x00FFFFFFu);
    k.ta.w = t21_rotr8x4(__byte_perm(c2, c3, 0x4321u));
    k.t4 = t21_rotr8x4(c3 >> 8);
#endif
  }
  static __device__ __forceinline__ void start_compact(const WDesc& d, int row, int s, int S, CompactC& k) {
    const uint32_t bits = s == 0 ? 0u : __ldg(d.t21_ckpt + (int64_t)row * (S - 1) + (s - 1));
    k.rp = (uint32_t)__ldg(d.t2_rowp + row);
    k.ow = __ldg(d.t21_rowoff + row) + (bits >> 5);
    k.ob = bits & 31u;

  }
  static __device__ __forceinline__ R load(const WDesc& d, int row, int kk, int lane, const C& k) {
    R r;
    const int j = lane & 15, c = kk >> 7;
    const uint16_t* l1 = reinterpret_cast<const uint16_t*>(d.planes) + ((int64_t)row * (d.K >> 7) + c) * 16;
    r.c16 = (uint32_t)__ldg(l1 + j);
    r.s8 = __ldg(reinterpret_cast<const uint2*>(d.smb + (int64_t)row * d.K + kk));
    r.win = __ldg(d.t2_ovf + k.ow + (uint32_t)j);
    return r;
  }
  static __device__ __forceinline__ R load(const WDesc& d, int row, int kk, int lane, const CompactC& k) {
    R r;
    const int j = lane & 15, c = kk >> 7;
    const uint16_t* l1 = reinterpret_cast<const uint16_t*>(d.planes) + ((int64_t)row * (d.K >> 7) + c) * 16;
    r.c16 = (uint32_t)__ldg(l1 + j);
    r.s8 = __ldg(reinterpret_cast<const uint2*>(d.smb + (int64_t)row * d.K + kk));
    r.win = __ldg(d.t2_ovf + k.ow + (uint32_t)j);
    return r;
  }
  static __device__ __forceinline__ uint4 dec(const WDesc& d, const R& r, int, int, int lane, int j,
                                              C& k, const uint32_t* tab) {
    const uint32_t ob = k.ob;
    const uint32_t C1 = lop3<0xA8u>(r.c16, r.c16 << 14, 0x33333333u);
    const uint32_t T1 = lop3<0x80u>(C1, C1 >> 1, 0x11111111u);
    const uint32_t P1 = T1 * 0x11111111u;
    const uint32_t k1 = P1 >> 28, R1 = P1 - T1;
    uint32_t n1;
    const uint32_t rho1 = t21_hscan16(k1, lane, j, n1);
#if BI_T21_TABREG
    const uint4 ta = k.ta;
    const uint32_t t4 = k.t4;
    (void)tab;
#else
    const uint4 ta = *reinterpret_cast<const uint4*>(tab + 8u * k.rp);
    const uint32_t t4 = tab[8u * k.rp + 4u];
#endif
    const uint32_t E1lo = __byte_perm(ta.x, ta.y, C1), E1hi = __byte_perm(ta.x, ta.y, C1 >> 16);
    // L2 (rank-1 order): digit d -> symbol 3 + d (bytes 3..5; 3 -> byte 6 placeholder)
    const uint32_t f2 = t2_get_win(r.win, ob + 2u * rho1, 2u * k1);
    const uint32_t N2 = t21_spread2(f2);
    const uint32_t S2 = N2 + 0x33333333u;
    uint32_t X2lo = __byte_perm(ta.x, ta.y, S2), X2hi = __byte_perm(ta.x, ta.y, S2 >> 16);
    const uint32_t Z2 = lop3<0x80u>(N2, N2 >> 1, 0x11111111u);
    const uint32_t P2 = Z2 * 0x11111111u;
    const uint32_t k2 = P2 >> 28, R2 = P2 - Z2;
    uint32_t n2;
    const uint32_t rho2 = t21_hscan16(k2, lane, j, n2);
    // L3 (rank-2 order): digit d -> symbol 6 + d (w2 bytes 0..2; 3 -> byte 3 placeholder)
    const uint32_t f3 = t2_get31(d.t2_ovf, k.ow, r.win, ob + 2u * n1 + 2u * rho2, 2u * k2);
    const uint32_t N3 = t21_spread2(f3);
    uint32_t X3lo = __byte_perm(ta.z, ta.z, N3), X3hi = __byte_perm(ta.z, ta.z, N3 >> 16);
    const uint32_t Z3 = lop3<0x80u>(N3, N3 >> 1, 0x11111111u);
    const uint32_t P3 = Z3 * 0x11111111u;
    const uint32_t k3 = P3 >> 28, R3 = P3 - Z3;
    uint32_t n3;
    const uint32_t rho3 = t21_hscan16(k3, lane, j, n3);
    // L4 (rank-3 order): digit d -> symbol 9 + d (w3:w4 bytes 0..6; 7 -> raw, placeholder)
    const uint32_t b4 = ob + 2u * n1 + 2u * n2;
    const uint32_t f4 = t2_get31(d.t2_ovf, k.ow, r.win, b4 + 3u * rho3, 3u * k3);
    const uint32_t N4 = t21_spread3(f4);
    uint32_t X4lo = __byte_perm(ta.w, t4, N4), X4hi = __byte_perm(ta.w, t4, N4 >> 16);
    const uint32_t Z4 = lop3<0x80u>(N4, N4 >> 1, N4 >> 2) & 0x11111111u;
    uint32_t n4 = 0u;
    if (BI_RARE(__any_sync(0xffffffffu, Z4))) {
      const uint32_t P4 = Z4 * 0x11111111u;
      const uint32_t k4 = P4 >> 28, R4 = P4 - Z4;
      const uint32_t rho4 = t21_hscan16(k4, lane, j, n4);
      const uint32_t base = b4 + 3u * n3 + 8u * rho4;
      const uint32_t g0 = t21_rotr8x4(t2_get(d.t2_ovf, k.ow, r.win, base, min(8u * k4, 32u)));
      const uint32_t g1 = t21_rotr8x4(t2_get(d.t2_ovf, k.ow, r.win, base + 32u, 8u * k4 > 32u ? 8u * k4 - 32u : 0u));
      X4lo = __byte_perm(X4lo, __byte_perm(g0, g1, R4), t2_msel_lo(Z4));
      X4hi = __byte_perm(X4hi, __byte_perm(g0, g1, R4 >> 16), t2_msel_hi(Z4));
    }
    X3lo = __byte_perm(X3lo, __byte_perm(X4lo, X4hi, R3), t2_msel_lo(Z3));
    X3hi = __byte_perm(X3hi, __byte_perm(X4lo, X4hi, R3 >> 16), t2_msel_hi(Z3));
    X2lo = __byte_perm(X2lo, __byte_perm(X3lo, X3hi, R2), t2_msel_lo(Z2));
    X2hi = __byte_perm(X2hi, __byte_perm(X3lo, X3hi, R2 >> 16), t2_msel_hi(Z2));
    const uint32_t Elo = __byte_perm(E1lo, __byte_perm(X2lo, X2hi, R1), t2_msel_lo(T1));
    const uint32_t Ehi = __byte_perm(E1hi, __byte_perm(X2lo, X2hi, R1 >> 16), t2_msel_hi(T1));
    // cursor: exact region bits; rows are the only padded unit, and a split never crosses a row
    const uint32_t t = b4 + 3u * n3 + 8u * n4;
    k.ow += t >> 5;
    k.ob = t & 31u;
    uint4 w;
    w.x = t21_pair(r.s8.x, Elo, 0x5140u, 0x1504u);
    w.y = t21_pair(r.s8.x, Elo, 0x7362u, 0x3726u);
    w.z = t21_pair(r.s8.y, Ehi, 0x5140u, 0x1504u);
    w.w = t21_pair(r.s8.y, Ehi, 0x7362u, 0x3726u);
    return w;
  }
#if BI_T21_TIER_PREFIX
  static __device__ __forceinline__ uint4 dec_header(const WDesc& d, const R& r, int, int, int lane, int j,
                                              C& k, const uint32_t* tab, uint32_t& header) {
    const uint32_t ob = k.ob;
    const uint32_t C1 = lop3<0xA8u>(r.c16, r.c16 << 14, 0x33333333u);
    const uint32_t T1 = lop3<0x80u>(C1, C1 >> 1, 0x11111111u);
    const uint32_t P1 = T1 * 0x11111111u;
    const uint32_t k1 = P1 >> 28, R1 = P1 - T1;
    uint32_t n1;
    const uint32_t rho1 = t21_hscan16(k1, lane, j, n1);
#if BI_T21_TABREG
    const uint4 ta = k.ta;
    const uint32_t t4 = k.t4;
    (void)tab;
#else
    const uint4 ta = *reinterpret_cast<const uint4*>(tab + 8u * k.rp);
    const uint32_t t4 = tab[8u * k.rp + 4u];
#endif
    const uint32_t E1lo = __byte_perm(ta.x, ta.y, C1), E1hi = __byte_perm(ta.x, ta.y, C1 >> 16);
    // L2 (rank-1 order): digit d -> symbol 3 + d (bytes 3..5; 3 -> byte 6 placeholder)
    const uint32_t f2 = t2_get_win(r.win, ob + 2u * rho1, 2u * k1);
    const uint32_t N2 = t21_spread2(f2);
    const uint32_t S2 = N2 + 0x33333333u;
    uint32_t X2lo = __byte_perm(ta.x, ta.y, S2), X2hi = __byte_perm(ta.x, ta.y, S2 >> 16);
    const uint32_t Z2 = lop3<0x80u>(N2, N2 >> 1, 0x11111111u);
    const uint32_t P2 = Z2 * 0x11111111u;
    const uint32_t k2 = P2 >> 28, R2 = P2 - Z2;
    uint32_t n2;
    const uint32_t rho2 = t21_hscan16(k2, lane, j, n2);
    // L3 (rank-2 order): digit d -> symbol 6 + d (w2 bytes 0..2; 3 -> byte 3 placeholder)
    const uint32_t f3 = t2_get31(d.t2_ovf, k.ow, r.win, ob + 2u * n1 + 2u * rho2, 2u * k2);
    const uint32_t N3 = t21_spread2(f3);
    uint32_t X3lo = __byte_perm(ta.z, ta.z, N3), X3hi = __byte_perm(ta.z, ta.z, N3 >> 16);
    const uint32_t Z3 = lop3<0x80u>(N3, N3 >> 1, 0x11111111u);
    const uint32_t P3 = Z3 * 0x11111111u;
    const uint32_t k3 = P3 >> 28, R3 = P3 - Z3;
    uint32_t n3;
    const uint32_t rho3 = t21_hscan16(k3, lane, j, n3);
    // L4 (rank-3 order): digit d -> symbol 9 + d (w3:w4 bytes 0..6; 7 -> raw, placeholder)
    const uint32_t b4 = ob + 2u * n1 + 2u * n2;
    const uint32_t f4 = t2_get31(d.t2_ovf, k.ow, r.win, b4 + 3u * rho3, 3u * k3);
    const uint32_t N4 = t21_spread3(f4);
    uint32_t X4lo = __byte_perm(ta.w, t4, N4), X4hi = __byte_perm(ta.w, t4, N4 >> 16);
    const uint32_t Z4 = lop3<0x80u>(N4, N4 >> 1, N4 >> 2) & 0x11111111u;
    uint32_t n4 = 0u;
    if (BI_RARE(__any_sync(0xffffffffu, Z4))) {
      const uint32_t P4 = Z4 * 0x11111111u;
      const uint32_t k4 = P4 >> 28, R4 = P4 - Z4;
      const uint32_t rho4 = t21_hscan16(k4, lane, j, n4);
      const uint32_t base = b4 + 3u * n3 + 8u * rho4;
      const uint32_t g0 = t21_rotr8x4(t2_get(d.t2_ovf, k.ow, r.win, base, min(8u * k4, 32u)));
      const uint32_t g1 = t21_rotr8x4(t2_get(d.t2_ovf, k.ow, r.win, base + 32u, 8u * k4 > 32u ? 8u * k4 - 32u : 0u));
      X4lo = __byte_perm(X4lo, __byte_perm(g0, g1, R4), t2_msel_lo(Z4));
      X4hi = __byte_perm(X4hi, __byte_perm(g0, g1, R4 >> 16), t2_msel_hi(Z4));
    }
    X3lo = __byte_perm(X3lo, __byte_perm(X4lo, X4hi, R3), t2_msel_lo(Z3));
    X3hi = __byte_perm(X3hi, __byte_perm(X4lo, X4hi, R3 >> 16), t2_msel_hi(Z3));
    X2lo = __byte_perm(X2lo, __byte_perm(X3lo, X3hi, R2), t2_msel_lo(Z2));
    X2hi = __byte_perm(X2hi, __byte_perm(X3lo, X3hi, R2 >> 16), t2_msel_hi(Z2));
    const uint32_t Elo = __byte_perm(E1lo, __byte_perm(X2lo, X2hi, R1), t2_msel_lo(T1));
    const uint32_t Ehi = __byte_perm(E1hi, __byte_perm(X2lo, X2hi, R1 >> 16), t2_msel_hi(T1));
    header=n1 | (n2<<8) | (n3<<16) | (n4<<24);
    // cursor: exact region bits; rows are the only padded unit, and a split never crosses a row
    const uint32_t t = b4 + 3u * n3 + 8u * n4;
    k.ow += t >> 5;
    k.ob = t & 31u;
    uint4 w;
    w.x = t21_pair(r.s8.x, Elo, 0x5140u, 0x1504u);
    w.y = t21_pair(r.s8.x, Elo, 0x7362u, 0x3726u);
    w.z = t21_pair(r.s8.y, Ehi, 0x5140u, 0x1504u);
    w.w = t21_pair(r.s8.y, Ehi, 0x7362u, 0x3726u);
    return w;
  }
#endif
#if BI_T21_PACKSCAN2
  // Phase two row-pair decodes together; only the three mandatory rank scans
  // are packed. Raw-byte scans retain the scalar path and its warp-uniform guard.
  template <typename Q, typename Cursor>
  static __device__ __forceinline__ uint4 finish_compact(const WDesc& d, const R& r,
      int lane, int j, Cursor& k, const uint32_t* tab, const Q& q) {
    const uint32_t b4 = k.ob + 2u * q.n1 + 2u * q.n2;
    const uint32_t f4 = t2_get31(d.t2_ovf, k.ow, r.win, b4 + 3u * q.rho3, 3u * q.k3);
    const uint32_t N4 = t21_spread3(f4);
    const uint32_t Z4 = lop3<0x80u>(N4, N4 >> 1, N4 >> 2) & 0x11111111u;
    uint32_t n4 = 0u, rawlo = 0u, rawhi = 0u;
    const bool raw = __any_sync(0xffffffffu, Z4);
    if (BI_RARE(raw)) {
      const uint32_t P4 = Z4 * 0x11111111u;
      const uint32_t k4 = P4 >> 28, R4 = P4 - Z4;
      const uint32_t rho4 = t21_hscan16(k4, lane, j, n4);
      const uint32_t base = b4 + 3u * q.n3 + 8u * rho4;
      const uint32_t g0 = t21_rotr8x4(t2_get(d.t2_ovf, k.ow, r.win, base, min(8u * k4, 32u)));
      const uint32_t g1 = t21_rotr8x4(t2_get(d.t2_ovf, k.ow, r.win, base + 32u, 8u * k4 > 32u ? 8u * k4 - 32u : 0u));
      rawlo = __byte_perm(g0, g1, R4);
      rawhi = __byte_perm(g0, g1, R4 >> 16);
    }
    const uint4 ta = *reinterpret_cast<const uint4*>(tab + 8u * k.rp);
    const uint32_t t4 = tab[8u * k.rp + 4u];
    uint32_t X4lo = __byte_perm(ta.w, t4, N4), X4hi = __byte_perm(ta.w, t4, N4 >> 16);
    if (BI_RARE(raw)) {
      X4lo = __byte_perm(X4lo, rawlo, t2_msel_lo(Z4));
      X4hi = __byte_perm(X4hi, rawhi, t2_msel_hi(Z4));
    }
    uint32_t X3lo = __byte_perm(ta.z, ta.z, q.N3), X3hi = __byte_perm(ta.z, ta.z, q.N3 >> 16);
    X3lo = __byte_perm(X3lo, __byte_perm(X4lo, X4hi, q.R3), t2_msel_lo(q.Z3));
    X3hi = __byte_perm(X3hi, __byte_perm(X4lo, X4hi, q.R3 >> 16), t2_msel_hi(q.Z3));
    const uint32_t S2 = q.N2 + 0x33333333u;
    uint32_t X2lo = __byte_perm(ta.x, ta.y, S2), X2hi = __byte_perm(ta.x, ta.y, S2 >> 16);
    X2lo = __byte_perm(X2lo, __byte_perm(X3lo, X3hi, q.R2), t2_msel_lo(q.Z2));
    X2hi = __byte_perm(X2hi, __byte_perm(X3lo, X3hi, q.R2 >> 16), t2_msel_hi(q.Z2));
    const uint32_t E1lo = __byte_perm(ta.x, ta.y, q.C1), E1hi = __byte_perm(ta.x, ta.y, q.C1 >> 16);
    const uint32_t T1 = lop3<0x80u>(q.C1, q.C1 >> 1, 0x11111111u);
    const uint32_t Elo = __byte_perm(E1lo, __byte_perm(X2lo, X2hi, q.R1), t2_msel_lo(T1));
    const uint32_t Ehi = __byte_perm(E1hi, __byte_perm(X2lo, X2hi, q.R1 >> 16), t2_msel_hi(T1));
    const uint32_t t = b4 + 3u * q.n3 + 8u * n4;
    k.ow += t >> 5; k.ob = t & 31u;
    uint4 w;
    w.x = t21_pair(r.s8.x, Elo, 0x5140u, 0x1504u);
    w.y = t21_pair(r.s8.x, Elo, 0x7362u, 0x3726u);
    w.z = t21_pair(r.s8.y, Ehi, 0x5140u, 0x1504u);
    w.w = t21_pair(r.s8.y, Ehi, 0x7362u, 0x3726u);
    return w;
  }
#if BI_T21_EXEC_LAYOUT
  // EXEC-only late materialization: preserve the packed 16-byte codebook in global memory
  // and recreate the same five rotated words that Ld<F_TBE21>::start() would have made.
  // The decode expressions below intentionally match start() byte-for-byte.
  template <typename Q, typename Cursor>
  static __device__ __forceinline__ uint4 finish_compact_global(const WDesc& d, const R& r,
      int lane, int j, Cursor& k, const Q& q) {
    const uint32_t b4 = k.ob + 2u * q.n1 + 2u * q.n2;
    const uint32_t f4 = t2_get31(d.t2_ovf, k.ow, r.win, b4 + 3u * q.rho3, 3u * q.k3);
    const uint32_t N4 = t21_spread3(f4);
    const uint32_t Z4 = lop3<0x80u>(N4, N4 >> 1, N4 >> 2) & 0x11111111u;
    uint32_t n4 = 0u, rawlo = 0u, rawhi = 0u;
    const bool raw = __any_sync(0xffffffffu, Z4);
    if (BI_RARE(raw)) {
      const uint32_t P4 = Z4 * 0x11111111u;
      const uint32_t k4 = P4 >> 28, R4 = P4 - Z4;
      const uint32_t rho4 = t21_hscan16(k4, lane, j, n4);
      const uint32_t base = b4 + 3u * q.n3 + 8u * rho4;
      const uint32_t g0 = t21_rotr8x4(t2_get(d.t2_ovf, k.ow, r.win, base, min(8u * k4, 32u)));
      const uint32_t g1 = t21_rotr8x4(t2_get(d.t2_ovf, k.ow, r.win, base + 32u, 8u * k4 > 32u ? 8u * k4 - 32u : 0u));
      rawlo = __byte_perm(g0, g1, R4);
      rawhi = __byte_perm(g0, g1, R4 >> 16);
    }
    const uint32_t* src = reinterpret_cast<const uint32_t*>(d.t2_cb) + 4 * k.rp;
    const uint32_t c0 = __ldg(src), c1 = __ldg(src + 1), c2 = __ldg(src + 2), c3 = __ldg(src + 3);
    const uint4 ta = make_uint4(t21_rotr8x4(c0), t21_rotr8x4(c1 & 0x0000FFFFu),
                                t21_rotr8x4(__byte_perm(c1, c2, 0x0432u) & 0x00FFFFFFu),
                                t21_rotr8x4(__byte_perm(c2, c3, 0x4321u)));
    const uint32_t t4 = t21_rotr8x4(c3 >> 8);
    uint32_t X4lo = __byte_perm(ta.w, t4, N4), X4hi = __byte_perm(ta.w, t4, N4 >> 16);
    if (BI_RARE(raw)) {
      X4lo = __byte_perm(X4lo, rawlo, t2_msel_lo(Z4));
      X4hi = __byte_perm(X4hi, rawhi, t2_msel_hi(Z4));
    }
    uint32_t X3lo = __byte_perm(ta.z, ta.z, q.N3), X3hi = __byte_perm(ta.z, ta.z, q.N3 >> 16);
    X3lo = __byte_perm(X3lo, __byte_perm(X4lo, X4hi, q.R3), t2_msel_lo(q.Z3));
    X3hi = __byte_perm(X3hi, __byte_perm(X4lo, X4hi, q.R3 >> 16), t2_msel_hi(q.Z3));
    const uint32_t S2 = q.N2 + 0x33333333u;
    uint32_t X2lo = __byte_perm(ta.x, ta.y, S2), X2hi = __byte_perm(ta.x, ta.y, S2 >> 16);
    X2lo = __byte_perm(X2lo, __byte_perm(X3lo, X3hi, q.R2), t2_msel_lo(q.Z2));
    X2hi = __byte_perm(X2hi, __byte_perm(X3lo, X3hi, q.R2 >> 16), t2_msel_hi(q.Z2));
    const uint32_t E1lo = __byte_perm(ta.x, ta.y, q.C1), E1hi = __byte_perm(ta.x, ta.y, q.C1 >> 16);
    const uint32_t T1 = lop3<0x80u>(q.C1, q.C1 >> 1, 0x11111111u);
    const uint32_t Elo = __byte_perm(E1lo, __byte_perm(X2lo, X2hi, q.R1), t2_msel_lo(T1));
    const uint32_t Ehi = __byte_perm(E1hi, __byte_perm(X2lo, X2hi, q.R1 >> 16), t2_msel_hi(T1));
    const uint32_t t = b4 + 3u * q.n3 + 8u * n4;
    k.ow += t >> 5; k.ob = t & 31u;
    uint4 w;
    w.x = t21_pair(r.s8.x, Elo, 0x5140u, 0x1504u);
    w.y = t21_pair(r.s8.x, Elo, 0x7362u, 0x3726u);
    w.z = t21_pair(r.s8.y, Ehi, 0x5140u, 0x1504u);
    w.w = t21_pair(r.s8.y, Ehi, 0x7362u, 0x3726u);
    return w;
  }
#endif
#if BI_T21_EXEC_LAYOUT
  template <bool REG, bool DEFER, bool REMAT, typename Cursor, bool DIRECT = false, bool SWIZZLE = false, bool GLOBAL = false>
#else
  template <bool REG, bool DEFER, bool REMAT, typename Cursor, bool DIRECT = false, bool SWIZZLE = false>
#endif
  static __device__ __forceinline__ void dec_pair_impl(const WDesc& d, const R* r,
      int lane, int j, Cursor* k, const uint32_t* tab, uint4* out, uint16_t* dst = nullptr, int row0 = 0, int pitch = 0) {
    struct Q { uint32_t C1, T1, R1, k1, n1, rho1, X2lo, X2hi, Z2, R2, k2, n2, rho2, X3lo, X3hi, Z3, R3, k3, n3, rho3;
      uint32_t N2, N3;
    } q[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      q[i].C1 = lop3<0xA8u>(r[i].c16, r[i].c16 << 14, 0x33333333u);
      const uint32_t T1 = lop3<0x80u>(q[i].C1, q[i].C1 >> 1, 0x11111111u);
      if constexpr (!REMAT) q[i].T1 = T1;
      const uint32_t P1 = T1 * 0x11111111u;
      q[i].k1 = P1 >> 28; q[i].R1 = P1 - T1;
    }
    t21_scan_pair(q[0].k1, q[1].k1, j, q[0].rho1, q[1].rho1, q[0].n1, q[1].n1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      uint4 ta; uint32_t t4;
      if constexpr (!DEFER) {
      if constexpr (REG) { ta = k[i].ta; t4 = k[i].t4; }
      else { ta = *reinterpret_cast<const uint4*>(tab + 8u * k[i].rp); t4 = tab[8u * k[i].rp + 4u]; }
      }

      const uint32_t f2 = t2_get_win(r[i].win, k[i].ob + 2u * q[i].rho1, 2u * q[i].k1);
      const uint32_t N2 = t21_spread2(f2), S2 = N2 + 0x33333333u;
      if constexpr (DEFER) q[i].N2 = N2;
      else { q[i].X2lo = __byte_perm(ta.x, ta.y, S2); q[i].X2hi = __byte_perm(ta.x, ta.y, S2 >> 16); }
      q[i].Z2 = lop3<0x80u>(N2, N2 >> 1, 0x11111111u);
      const uint32_t P2 = q[i].Z2 * 0x11111111u;
      q[i].k2 = P2 >> 28; q[i].R2 = P2 - q[i].Z2;
    }
    t21_scan_pair(q[0].k2, q[1].k2, j, q[0].rho2, q[1].rho2, q[0].n2, q[1].n2);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      uint4 ta; uint32_t t4;
      if constexpr (!DEFER) {
      if constexpr (REG) { ta = k[i].ta; t4 = k[i].t4; }
      else { ta = *reinterpret_cast<const uint4*>(tab + 8u * k[i].rp); t4 = tab[8u * k[i].rp + 4u]; }
      }

      const uint32_t f3 = t2_get31(d.t2_ovf, k[i].ow, r[i].win,
          k[i].ob + 2u * q[i].n1 + 2u * q[i].rho2, 2u * q[i].k2);
      const uint32_t N3 = t21_spread2(f3);
      if constexpr (DEFER) q[i].N3 = N3;
      else { q[i].X3lo = __byte_perm(ta.z, ta.z, N3); q[i].X3hi = __byte_perm(ta.z, ta.z, N3 >> 16); }
      q[i].Z3 = lop3<0x80u>(N3, N3 >> 1, 0x11111111u);
      const uint32_t P3 = q[i].Z3 * 0x11111111u;
      q[i].k3 = P3 >> 28; q[i].R3 = P3 - q[i].Z3;
    }
    t21_scan_pair(q[0].k3, q[1].k3, j, q[0].rho3, q[1].rho3, q[0].n3, q[1].n3);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      if constexpr (!REG && DEFER && REMAT) {
#if BI_T21_EXEC_LAYOUT
        uint4 w;
        if constexpr (GLOBAL) w = finish_compact_global(d, r[i], lane, j, k[i], q[i]);
        else w = finish_compact(d, r[i], lane, j, k[i], tab, q[i]);
#else
        const uint4 w = finish_compact(d, r[i], lane, j, k[i], tab, q[i]);
#endif
        if constexpr (DIRECT) {
          if constexpr (SWIZZLE) {
            const int row = row0 + 2 * i;
            *reinterpret_cast<uint4*>(dst + row * pitch + ((8 * j) ^ ((row & 7) * 8))) = w;
          } else *reinterpret_cast<uint4*>(dst + (row0 + 2 * i) * pitch + 8 * j) = w;
        }
        else out[i] = w;
      }
      else {
      const uint32_t ob = k[i].ob;
      uint4 ta;
      uint32_t t4;
      if constexpr (REG) { ta = k[i].ta; t4 = k[i].t4; }
      else { ta = *reinterpret_cast<const uint4*>(tab + 8u * k[i].rp); t4 = tab[8u * k[i].rp + 4u]; }

      if constexpr (REMAT) q[i].T1 = lop3<0x80u>(q[i].C1, q[i].C1 >> 1, 0x11111111u);
      const uint32_t E1lo = __byte_perm(ta.x, ta.y, q[i].C1);
      const uint32_t E1hi = __byte_perm(ta.x, ta.y, q[i].C1 >> 16);
      // L4 (rank-3 order): digit d -> symbol 9 + d (w3:w4 bytes 0..6; 7 -> raw, placeholder)
      const uint32_t b4 = ob + 2u * q[i].n1 + 2u * q[i].n2;
      const uint32_t f4 = t2_get31(d.t2_ovf, k[i].ow, r[i].win, b4 + 3u * q[i].rho3, 3u * q[i].k3);
      const uint32_t N4 = t21_spread3(f4);
      uint32_t X4lo = __byte_perm(ta.w, t4, N4), X4hi = __byte_perm(ta.w, t4, N4 >> 16);
      const uint32_t Z4 = lop3<0x80u>(N4, N4 >> 1, N4 >> 2) & 0x11111111u;
      uint32_t n4 = 0u;
      if (BI_RARE(__any_sync(0xffffffffu, Z4))) {
        const uint32_t P4 = Z4 * 0x11111111u;
        const uint32_t k4 = P4 >> 28, R4 = P4 - Z4;
        const uint32_t rho4 = t21_hscan16(k4, lane, j, n4);
        const uint32_t base = b4 + 3u * q[i].n3 + 8u * rho4;
        const uint32_t g0 = t21_rotr8x4(t2_get(d.t2_ovf, k[i].ow, r[i].win, base, min(8u * k4, 32u)));
        const uint32_t g1 = t21_rotr8x4(t2_get(d.t2_ovf, k[i].ow, r[i].win, base + 32u, 8u * k4 > 32u ? 8u * k4 - 32u : 0u));
        X4lo = __byte_perm(X4lo, __byte_perm(g0, g1, R4), t2_msel_lo(Z4));
        X4hi = __byte_perm(X4hi, __byte_perm(g0, g1, R4 >> 16), t2_msel_hi(Z4));
      }
      if constexpr (DEFER) {

      // Retain one packed selector per level instead of two exponent vectors
      // across the rank-scan ladder; these lookups are independent of the scans.
      q[i].X3lo = __byte_perm(ta.z, ta.z, q[i].N3);
      q[i].X3hi = __byte_perm(ta.z, ta.z, q[i].N3 >> 16);
      const uint32_t S2 = q[i].N2 + 0x33333333u;
      q[i].X2lo = __byte_perm(ta.x, ta.y, S2);
      q[i].X2hi = __byte_perm(ta.x, ta.y, S2 >> 16);
      }
      q[i].X3lo = __byte_perm(q[i].X3lo, __byte_perm(X4lo, X4hi, q[i].R3), t2_msel_lo(q[i].Z3));
      q[i].X3hi = __byte_perm(q[i].X3hi, __byte_perm(X4lo, X4hi, q[i].R3 >> 16), t2_msel_hi(q[i].Z3));
      q[i].X2lo = __byte_perm(q[i].X2lo, __byte_perm(q[i].X3lo, q[i].X3hi, q[i].R2), t2_msel_lo(q[i].Z2));
      q[i].X2hi = __byte_perm(q[i].X2hi, __byte_perm(q[i].X3lo, q[i].X3hi, q[i].R2 >> 16), t2_msel_hi(q[i].Z2));
      const uint32_t Elo = __byte_perm(E1lo, __byte_perm(q[i].X2lo, q[i].X2hi, q[i].R1), t2_msel_lo(q[i].T1));
      const uint32_t Ehi = __byte_perm(E1hi, __byte_perm(q[i].X2lo, q[i].X2hi, q[i].R1 >> 16), t2_msel_hi(q[i].T1));
      // cursor: exact region bits; rows are the only padded unit, and a split never crosses a row
      const uint32_t t = b4 + 3u * q[i].n3 + 8u * n4;
      k[i].ow += t >> 5;
      k[i].ob = t & 31u;
      uint4 w;
      w.x = t21_pair(r[i].s8.x, Elo, 0x5140u, 0x1504u);
      w.y = t21_pair(r[i].s8.x, Elo, 0x7362u, 0x3726u);
      w.z = t21_pair(r[i].s8.y, Ehi, 0x5140u, 0x1504u);
      w.w = t21_pair(r[i].s8.y, Ehi, 0x7362u, 0x3726u);
      out[i] = w;
      }
    }
  }
#if BI_T21_PAIR_VOTE
  template <bool REG, bool DEFER, bool REMAT, typename Cursor, bool DIRECT = false, bool SWIZZLE = false>
  static __device__ __forceinline__ void dec_pair_ballot_impl(const WDesc& d, const R* r,
      int lane, int j, Cursor* k, const uint32_t* tab, uint4* out, uint16_t* dst = nullptr, int row0 = 0, int pitch = 0) {
    struct Q { uint32_t C1, T1, R1, k1, n1, rho1, X2lo, X2hi, Z2, R2, k2, n2, rho2, X3lo, X3hi, Z3, R3, k3, n3, rho3;
      uint32_t N2, N3;
    } q[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      q[i].C1 = lop3<0xA8u>(r[i].c16, r[i].c16 << 14, 0x33333333u);
      const uint32_t T1 = lop3<0x80u>(q[i].C1, q[i].C1 >> 1, 0x11111111u);
      if constexpr (!REMAT) q[i].T1 = T1;
      const uint32_t P1 = T1 * 0x11111111u;
      q[i].k1 = P1 >> 28; q[i].R1 = P1 - T1;
    }
    t21_hscan16_pair_ballot(q[0].k1, q[1].k1, lane, j, q[0].rho1, q[1].rho1, q[0].n1, q[1].n1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      uint4 ta; uint32_t t4;
      if constexpr (!DEFER) {
      if constexpr (REG) { ta = k[i].ta; t4 = k[i].t4; }
      else { ta = *reinterpret_cast<const uint4*>(tab + 8u * k[i].rp); t4 = tab[8u * k[i].rp + 4u]; }
      }

      const uint32_t f2 = t2_get_win(r[i].win, k[i].ob + 2u * q[i].rho1, 2u * q[i].k1);
      const uint32_t N2 = t21_spread2(f2), S2 = N2 + 0x33333333u;
      if constexpr (DEFER) q[i].N2 = N2;
      else { q[i].X2lo = __byte_perm(ta.x, ta.y, S2); q[i].X2hi = __byte_perm(ta.x, ta.y, S2 >> 16); }
      q[i].Z2 = lop3<0x80u>(N2, N2 >> 1, 0x11111111u);
      const uint32_t P2 = q[i].Z2 * 0x11111111u;
      q[i].k2 = P2 >> 28; q[i].R2 = P2 - q[i].Z2;
    }
    t21_hscan16_pair_ballot(q[0].k2, q[1].k2, lane, j, q[0].rho2, q[1].rho2, q[0].n2, q[1].n2);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      uint4 ta; uint32_t t4;
      if constexpr (!DEFER) {
      if constexpr (REG) { ta = k[i].ta; t4 = k[i].t4; }
      else { ta = *reinterpret_cast<const uint4*>(tab + 8u * k[i].rp); t4 = tab[8u * k[i].rp + 4u]; }
      }

      const uint32_t f3 = t2_get31(d.t2_ovf, k[i].ow, r[i].win,
          k[i].ob + 2u * q[i].n1 + 2u * q[i].rho2, 2u * q[i].k2);
      const uint32_t N3 = t21_spread2(f3);
      if constexpr (DEFER) q[i].N3 = N3;
      else { q[i].X3lo = __byte_perm(ta.z, ta.z, N3); q[i].X3hi = __byte_perm(ta.z, ta.z, N3 >> 16); }
      q[i].Z3 = lop3<0x80u>(N3, N3 >> 1, 0x11111111u);
      const uint32_t P3 = q[i].Z3 * 0x11111111u;
      q[i].k3 = P3 >> 28; q[i].R3 = P3 - q[i].Z3;
    }
    t21_hscan16_pair_ballot(q[0].k3, q[1].k3, lane, j, q[0].rho3, q[1].rho3, q[0].n3, q[1].n3);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      if constexpr (!REG && DEFER && REMAT) {
        const uint4 w = finish_compact(d, r[i], lane, j, k[i], tab, q[i]);
        if constexpr (DIRECT) {
          if constexpr (SWIZZLE) {
            const int row = row0 + 2 * i;
            *reinterpret_cast<uint4*>(dst + row * pitch + ((8 * j) ^ ((row & 7) * 8))) = w;
          } else *reinterpret_cast<uint4*>(dst + (row0 + 2 * i) * pitch + 8 * j) = w;
        }
        else out[i] = w;
      }
      else {
      const uint32_t ob = k[i].ob;
      uint4 ta;
      uint32_t t4;
      if constexpr (REG) { ta = k[i].ta; t4 = k[i].t4; }
      else { ta = *reinterpret_cast<const uint4*>(tab + 8u * k[i].rp); t4 = tab[8u * k[i].rp + 4u]; }

      if constexpr (REMAT) q[i].T1 = lop3<0x80u>(q[i].C1, q[i].C1 >> 1, 0x11111111u);
      const uint32_t E1lo = __byte_perm(ta.x, ta.y, q[i].C1);
      const uint32_t E1hi = __byte_perm(ta.x, ta.y, q[i].C1 >> 16);
      // L4 (rank-3 order): digit d -> symbol 9 + d (w3:w4 bytes 0..6; 7 -> raw, placeholder)
      const uint32_t b4 = ob + 2u * q[i].n1 + 2u * q[i].n2;
      const uint32_t f4 = t2_get31(d.t2_ovf, k[i].ow, r[i].win, b4 + 3u * q[i].rho3, 3u * q[i].k3);
      const uint32_t N4 = t21_spread3(f4);
      uint32_t X4lo = __byte_perm(ta.w, t4, N4), X4hi = __byte_perm(ta.w, t4, N4 >> 16);
      const uint32_t Z4 = lop3<0x80u>(N4, N4 >> 1, N4 >> 2) & 0x11111111u;
      uint32_t n4 = 0u;
      if (BI_RARE(__any_sync(0xffffffffu, Z4))) {
        const uint32_t P4 = Z4 * 0x11111111u;
        const uint32_t k4 = P4 >> 28, R4 = P4 - Z4;
        const uint32_t rho4 = t21_hscan16(k4, lane, j, n4);
        const uint32_t base = b4 + 3u * q[i].n3 + 8u * rho4;
        const uint32_t g0 = t21_rotr8x4(t2_get(d.t2_ovf, k[i].ow, r[i].win, base, min(8u * k4, 32u)));
        const uint32_t g1 = t21_rotr8x4(t2_get(d.t2_ovf, k[i].ow, r[i].win, base + 32u, 8u * k4 > 32u ? 8u * k4 - 32u : 0u));
        X4lo = __byte_perm(X4lo, __byte_perm(g0, g1, R4), t2_msel_lo(Z4));
        X4hi = __byte_perm(X4hi, __byte_perm(g0, g1, R4 >> 16), t2_msel_hi(Z4));
      }
      if constexpr (DEFER) {

      // Retain one packed selector per level instead of two exponent vectors
      // across the rank-scan ladder; these lookups are independent of the scans.
      q[i].X3lo = __byte_perm(ta.z, ta.z, q[i].N3);
      q[i].X3hi = __byte_perm(ta.z, ta.z, q[i].N3 >> 16);
      const uint32_t S2 = q[i].N2 + 0x33333333u;
      q[i].X2lo = __byte_perm(ta.x, ta.y, S2);
      q[i].X2hi = __byte_perm(ta.x, ta.y, S2 >> 16);
      }
      q[i].X3lo = __byte_perm(q[i].X3lo, __byte_perm(X4lo, X4hi, q[i].R3), t2_msel_lo(q[i].Z3));
      q[i].X3hi = __byte_perm(q[i].X3hi, __byte_perm(X4lo, X4hi, q[i].R3 >> 16), t2_msel_hi(q[i].Z3));
      q[i].X2lo = __byte_perm(q[i].X2lo, __byte_perm(q[i].X3lo, q[i].X3hi, q[i].R2), t2_msel_lo(q[i].Z2));
      q[i].X2hi = __byte_perm(q[i].X2hi, __byte_perm(q[i].X3lo, q[i].X3hi, q[i].R2 >> 16), t2_msel_hi(q[i].Z2));
      const uint32_t Elo = __byte_perm(E1lo, __byte_perm(q[i].X2lo, q[i].X2hi, q[i].R1), t2_msel_lo(q[i].T1));
      const uint32_t Ehi = __byte_perm(E1hi, __byte_perm(q[i].X2lo, q[i].X2hi, q[i].R1 >> 16), t2_msel_hi(q[i].T1));
      // cursor: exact region bits; rows are the only padded unit, and a split never crosses a row
      const uint32_t t = b4 + 3u * q[i].n3 + 8u * n4;
      k[i].ow += t >> 5;
      k[i].ob = t & 31u;
      uint4 w;
      w.x = t21_pair(r[i].s8.x, Elo, 0x5140u, 0x1504u);
      w.y = t21_pair(r[i].s8.x, Elo, 0x7362u, 0x3726u);
      w.z = t21_pair(r[i].s8.y, Ehi, 0x5140u, 0x1504u);
      w.w = t21_pair(r[i].s8.y, Ehi, 0x7362u, 0x3726u);
      out[i] = w;
      }
    }
  }
  static __device__ __forceinline__ void dec_pair_ballot(const WDesc& d, const R* r,
      int lane,int j,C* k,const uint32_t* tab,uint4* out) {
    dec_pair_ballot_impl<(BI_T21_TABREG != 0),false,false>(d,r,lane,j,k,tab,out);
  }
#endif


#if BI_T21_PACKSCAN4
  template <bool REG, bool DEFER, bool REMAT, typename Cursor, bool DIRECT = false, bool SWIZZLE = false>
  static __device__ __forceinline__ void dec_quad_impl(const WDesc& d, const R* r,
      int lane, int j, Cursor* k, const uint32_t* tab, uint4* out, uint16_t* dst = nullptr, int row0 = 0, int pitch = 0) {
    struct Q { uint32_t C1, T1, R1, k1, n1, rho1, X2lo, X2hi, Z2, R2, k2, n2, rho2, X3lo, X3hi, Z3, R3, k3, n3, rho3;
      uint32_t N2, N3;
    } q[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      q[i].C1 = lop3<0xA8u>(r[i].c16, r[i].c16 << 14, 0x33333333u);
      const uint32_t T1 = lop3<0x80u>(q[i].C1, q[i].C1 >> 1, 0x11111111u);
      if constexpr (!REMAT) q[i].T1 = T1;
      const uint32_t P1 = T1 * 0x11111111u;
      q[i].k1 = P1 >> 28; q[i].R1 = P1 - T1;
    }
    {
      uint32_t counts[4], ranks[4], totals[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) counts[i] = q[i].k1;
      t21_hscan16_quad(counts, j, ranks, totals);
#pragma unroll
      for (int i = 0; i < 4; ++i) { q[i].rho1 = ranks[i]; q[i].n1 = totals[i]; }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint4 ta; uint32_t t4;
      if constexpr (!DEFER) {
      if constexpr (REG) { ta = k[i].ta; t4 = k[i].t4; }
      else { ta = *reinterpret_cast<const uint4*>(tab + 8u * k[i].rp); t4 = tab[8u * k[i].rp + 4u]; }
      }

      const uint32_t f2 = t2_get_win(r[i].win, k[i].ob + 2u * q[i].rho1, 2u * q[i].k1);
      const uint32_t N2 = t21_spread2(f2), S2 = N2 + 0x33333333u;
      if constexpr (DEFER) q[i].N2 = N2;
      else { q[i].X2lo = __byte_perm(ta.x, ta.y, S2); q[i].X2hi = __byte_perm(ta.x, ta.y, S2 >> 16); }
      q[i].Z2 = lop3<0x80u>(N2, N2 >> 1, 0x11111111u);
      const uint32_t P2 = q[i].Z2 * 0x11111111u;
      q[i].k2 = P2 >> 28; q[i].R2 = P2 - q[i].Z2;
    }
    {
      uint32_t counts[4], ranks[4], totals[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) counts[i] = q[i].k2;
      t21_hscan16_quad(counts, j, ranks, totals);
#pragma unroll
      for (int i = 0; i < 4; ++i) { q[i].rho2 = ranks[i]; q[i].n2 = totals[i]; }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint4 ta; uint32_t t4;
      if constexpr (!DEFER) {
      if constexpr (REG) { ta = k[i].ta; t4 = k[i].t4; }
      else { ta = *reinterpret_cast<const uint4*>(tab + 8u * k[i].rp); t4 = tab[8u * k[i].rp + 4u]; }
      }

      const uint32_t f3 = t2_get31(d.t2_ovf, k[i].ow, r[i].win,
          k[i].ob + 2u * q[i].n1 + 2u * q[i].rho2, 2u * q[i].k2);
      const uint32_t N3 = t21_spread2(f3);
      if constexpr (DEFER) q[i].N3 = N3;
      else { q[i].X3lo = __byte_perm(ta.z, ta.z, N3); q[i].X3hi = __byte_perm(ta.z, ta.z, N3 >> 16); }
      q[i].Z3 = lop3<0x80u>(N3, N3 >> 1, 0x11111111u);
      const uint32_t P3 = q[i].Z3 * 0x11111111u;
      q[i].k3 = P3 >> 28; q[i].R3 = P3 - q[i].Z3;
    }
    {
      uint32_t counts[4], ranks[4], totals[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) counts[i] = q[i].k3;
      t21_hscan16_quad(counts, j, ranks, totals);
#pragma unroll
      for (int i = 0; i < 4; ++i) { q[i].rho3 = ranks[i]; q[i].n3 = totals[i]; }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if constexpr (!REG && DEFER && REMAT) {
        const uint4 w = finish_compact(d, r[i], lane, j, k[i], tab, q[i]);
        if constexpr (DIRECT) {
          if constexpr (SWIZZLE) {
            const int row = row0 + 2 * i;
            *reinterpret_cast<uint4*>(dst + row * pitch + ((8 * j) ^ ((row & 7) * 8))) = w;
          } else *reinterpret_cast<uint4*>(dst + (row0 + 2 * i) * pitch + 8 * j) = w;
        }
        else out[i] = w;
      }
      else {
      const uint32_t ob = k[i].ob;
      uint4 ta;
      uint32_t t4;
      if constexpr (REG) { ta = k[i].ta; t4 = k[i].t4; }
      else { ta = *reinterpret_cast<const uint4*>(tab + 8u * k[i].rp); t4 = tab[8u * k[i].rp + 4u]; }

      if constexpr (REMAT) q[i].T1 = lop3<0x80u>(q[i].C1, q[i].C1 >> 1, 0x11111111u);
      const uint32_t E1lo = __byte_perm(ta.x, ta.y, q[i].C1);
      const uint32_t E1hi = __byte_perm(ta.x, ta.y, q[i].C1 >> 16);
      // L4 (rank-3 order): digit d -> symbol 9 + d (w3:w4 bytes 0..6; 7 -> raw, placeholder)
      const uint32_t b4 = ob + 2u * q[i].n1 + 2u * q[i].n2;
      const uint32_t f4 = t2_get31(d.t2_ovf, k[i].ow, r[i].win, b4 + 3u * q[i].rho3, 3u * q[i].k3);
      const uint32_t N4 = t21_spread3(f4);
      uint32_t X4lo = __byte_perm(ta.w, t4, N4), X4hi = __byte_perm(ta.w, t4, N4 >> 16);
      const uint32_t Z4 = lop3<0x80u>(N4, N4 >> 1, N4 >> 2) & 0x11111111u;
      uint32_t n4 = 0u;
      if (BI_RARE(__any_sync(0xffffffffu, Z4))) {
        const uint32_t P4 = Z4 * 0x11111111u;
        const uint32_t k4 = P4 >> 28, R4 = P4 - Z4;
        const uint32_t rho4 = t21_hscan16(k4, lane, j, n4);
        const uint32_t base = b4 + 3u * q[i].n3 + 8u * rho4;
        const uint32_t g0 = t21_rotr8x4(t2_get(d.t2_ovf, k[i].ow, r[i].win, base, min(8u * k4, 32u)));
        const uint32_t g1 = t21_rotr8x4(t2_get(d.t2_ovf, k[i].ow, r[i].win, base + 32u, 8u * k4 > 32u ? 8u * k4 - 32u : 0u));
        X4lo = __byte_perm(X4lo, __byte_perm(g0, g1, R4), t2_msel_lo(Z4));
        X4hi = __byte_perm(X4hi, __byte_perm(g0, g1, R4 >> 16), t2_msel_hi(Z4));
      }
      if constexpr (DEFER) {

      // Retain one packed selector per level instead of two exponent vectors
      // across the rank-scan ladder; these lookups are independent of the scans.
      q[i].X3lo = __byte_perm(ta.z, ta.z, q[i].N3);
      q[i].X3hi = __byte_perm(ta.z, ta.z, q[i].N3 >> 16);
      const uint32_t S2 = q[i].N2 + 0x33333333u;
      q[i].X2lo = __byte_perm(ta.x, ta.y, S2);
      q[i].X2hi = __byte_perm(ta.x, ta.y, S2 >> 16);
      }
      q[i].X3lo = __byte_perm(q[i].X3lo, __byte_perm(X4lo, X4hi, q[i].R3), t2_msel_lo(q[i].Z3));
      q[i].X3hi = __byte_perm(q[i].X3hi, __byte_perm(X4lo, X4hi, q[i].R3 >> 16), t2_msel_hi(q[i].Z3));
      q[i].X2lo = __byte_perm(q[i].X2lo, __byte_perm(q[i].X3lo, q[i].X3hi, q[i].R2), t2_msel_lo(q[i].Z2));
      q[i].X2hi = __byte_perm(q[i].X2hi, __byte_perm(q[i].X3lo, q[i].X3hi, q[i].R2 >> 16), t2_msel_hi(q[i].Z2));
      const uint32_t Elo = __byte_perm(E1lo, __byte_perm(q[i].X2lo, q[i].X2hi, q[i].R1), t2_msel_lo(q[i].T1));
      const uint32_t Ehi = __byte_perm(E1hi, __byte_perm(q[i].X2lo, q[i].X2hi, q[i].R1 >> 16), t2_msel_hi(q[i].T1));
      // cursor: exact region bits; rows are the only padded unit, and a split never crosses a row
      const uint32_t t = b4 + 3u * q[i].n3 + 8u * n4;
      k[i].ow += t >> 5;
      k[i].ob = t & 31u;
      uint4 w;
      w.x = t21_pair(r[i].s8.x, Elo, 0x5140u, 0x1504u);
      w.y = t21_pair(r[i].s8.x, Elo, 0x7362u, 0x3726u);
      w.z = t21_pair(r[i].s8.y, Ehi, 0x5140u, 0x1504u);
      w.w = t21_pair(r[i].s8.y, Ehi, 0x7362u, 0x3726u);
      out[i] = w;
      }
    }
  }

  // Insert inside Ld<F_TBE21>; caller uses CompactC[4], R[4], uint4 out[4].
  static __device__ __forceinline__ void dec_quad(const WDesc& d, const R* r,
      int lane, int j, CompactC* k, const uint32_t* tab, uint4* out) {
    dec_quad_impl<false, true, true>(d, r, lane, j, k, tab, out);
  }
#endif

  static __device__ __forceinline__ void dec_pair(const WDesc& d, const R* r,
      int lane, int j, C* k, const uint32_t* tab, uint4* out) {
    dec_pair_impl<(BI_T21_TABREG != 0), (BI_T21_SCAN4 != 0), false>(d, r, lane, j, k, tab, out);
  }
  // Compact M128 producers retain indices/selectors, not table/exponent registers.
  static __device__ __forceinline__ void dec_pair_compact(const WDesc& d, const R* r,
      int lane, int j, CompactC* k, const uint32_t* tab, uint4* out) {
    dec_pair_impl<false, true, true>(d, r, lane, j, k, tab, out);
  }
  static __device__ __forceinline__ void dec_pair_compact_to_shared(const WDesc& d, const R* r,
      int lane, int j, CompactC* k, const uint32_t* tab, uint16_t* dst, int row0, int pitch) {
    dec_pair_impl<false, true, true, CompactC, true>(d, r, lane, j, k, tab, nullptr, dst, row0, pitch);
  }
  static __device__ __forceinline__ void dec_pair_compact_to_swizzled(const WDesc& d, const R* r,
      int lane, int j, CompactC* k, const uint32_t* tab, uint16_t* dst, int row0) {
    dec_pair_impl<false, true, true, CompactC, true, true>(d, r, lane, j, k, tab, nullptr, dst, row0, 128);
  }
#if BI_T21_EXEC_LAYOUT
  static __device__ __forceinline__ void dec_pair_compact_global_to_swizzled(const WDesc& d, const R* r,
      int lane, int j, CompactC* k, uint16_t* dst, int row0) {
    dec_pair_impl<false, true, true, CompactC, true, true, true>(d, r, lane, j, k, nullptr, nullptr, dst, row0, 128);
  }
#endif

#endif

};
#if BI_T21_TIER_PREFIX
#include "tier_prefix_literal.cuh"
#define TPF_FAIL() asm volatile("trap;")
// Experimental optional descriptor adapter, bound by four compiled flag getters.
// The full production arithmetic remains downstream of the decoded uint4 values.
__device__ __forceinline__ uint32_t tpf_fixed_get(const uint32_t* ovf,
    uint32_t ow, uint32_t base, uint32_t length, uint32_t width, int j) {
  const uint32_t first = 8u * (uint32_t)j;
  // Every lane loads a valid word pair, including empty tails. Saturating the
  // index at the stream end keeps both reads within the existing 16-word pad.
  // This avoids divergent load branches before full-warp collectives.
  const uint32_t index = min(first, length);
  const uint32_t count = min(8u, length - index);
  const uint32_t bit = base + width * index;
  const uint32_t a = __ldg(ovf + ow + (bit >> 5));
  const uint32_t b = __ldg(ovf + ow + (bit >> 5) + 1u);
  return __funnelshift_r(a, b, bit & 31u) & ((1u << (count * width)) - 1u);
}

template <int KQT, typename Cursor>
__device__ __forceinline__ void tpf_dec_pair(const WDesc& d,
    const typename Ld<F_TBE21, KQT>::R* raw, int lane, int j,
    Cursor* cur, uint4* out, const int* rows, int chunk) {
  if (!d.tpf_counts) TPF_FAIL();
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const uint32_t h = __ldg(d.tpf_counts + (int64_t)rows[i] * (d.K >> 7) + chunk);
    const uint32_t n1 = h & 255u, n2 = (h >> 8) & 255u;
    const uint32_t n3 = (h >> 16) & 255u, n4 = h >> 24;
    if (__any_sync(0xffffffffu, !(n4 <= n3 && n3 <= n2 && n2 <= n1 && n1 <= 128u))) TPF_FAIL();
    TpfInput in;
    in.l1c16 = raw[i].c16;
    in.l2f16 = tpf_fixed_get(d.t2_ovf, cur[i].ow, cur[i].ob, n1, 2u, j);
    in.l3f16 = tpf_fixed_get(d.t2_ovf, cur[i].ow, cur[i].ob + 2u*n1, n2, 2u, j);
    // Keep every full-mask collective warp-uniform, including mixed halfwarps.
    const bool active4 = __any_sync(0xffffffffu, n3 != 0u);
    in.l4f24 = active4 ? tpf_fixed_get(d.t2_ovf, cur[i].ow,
        cur[i].ob + 2u*n1 + 2u*n2, n3, 3u, j) : 0u;
    in.header_n1 = n1; in.header_n2 = n2; in.header_n3 = n3; in.header_n4 = n4;
    TpfOutput q;
    tpf_literal<true>(in, lane, q, active4);
    if (__any_sync(0xffffffffu, !q.header_match)) TPF_FAIL();

    // Same dictionary lookups, nested replacement, splice and cursor as Ld::dec.
    const uint4 ta = cur[i].ta;
    const uint32_t t4 = cur[i].t4;
    const uint32_t C1 = lop3<0xA8u>(raw[i].c16, raw[i].c16 << 14, 0x33333333u);
    const uint32_t T1 = lop3<0x80u>(C1, C1 >> 1, 0x11111111u);
    const uint32_t R1 = T1 * 0x11111111u - T1;
    const uint32_t N2 = t21_spread2(q.f2), S2 = N2 + 0x33333333u;
    const uint32_t Z2 = lop3<0x80u>(N2, N2 >> 1, 0x11111111u);
    const uint32_t R2 = Z2 * 0x11111111u - Z2;
    const uint32_t N3 = t21_spread2(q.f3);
    uint32_t X3lo = __byte_perm(ta.z, ta.z, N3), X3hi = __byte_perm(ta.z, ta.z, N3 >> 16);
    const uint32_t b4 = cur[i].ob + 2u*n1 + 2u*n2;
    if (active4) {
      const uint32_t Z3 = lop3<0x80u>(N3, N3 >> 1, 0x11111111u);
      const uint32_t R3 = Z3 * 0x11111111u - Z3;
      const uint32_t N4 = t21_spread3(q.f4);
      const uint32_t Z4 = lop3<0x80u>(N4, N4 >> 1, N4 >> 2) & 0x11111111u;
      uint32_t X4lo = __byte_perm(ta.w, t4, N4), X4hi = __byte_perm(ta.w, t4, N4 >> 16);
      if (__any_sync(0xffffffffu, Z4)) {
        const uint32_t R4 = Z4 * 0x11111111u - Z4;
        const uint32_t base = b4 + 3u*n3 + 8u*q.rho4;
        const uint32_t g0 = t21_rotr8x4(t2_get(d.t2_ovf, cur[i].ow, raw[i].win,
            base, min(8u*q.k4, 32u)));
        const uint32_t g1 = t21_rotr8x4(t2_get(d.t2_ovf, cur[i].ow, raw[i].win,
            base + 32u, 8u*q.k4 > 32u ? 8u*q.k4 - 32u : 0u));
        X4lo = __byte_perm(X4lo, __byte_perm(g0, g1, R4), t2_msel_lo(Z4));
        X4hi = __byte_perm(X4hi, __byte_perm(g0, g1, R4 >> 16), t2_msel_hi(Z4));
      }
      X3lo = __byte_perm(X3lo, __byte_perm(X4lo, X4hi, R3), t2_msel_lo(Z3));
      X3hi = __byte_perm(X3hi, __byte_perm(X4lo, X4hi, R3 >> 16), t2_msel_hi(Z3));
    }
    uint32_t X2lo = __byte_perm(ta.x, ta.y, S2), X2hi = __byte_perm(ta.x, ta.y, S2 >> 16);
    X2lo = __byte_perm(X2lo, __byte_perm(X3lo, X3hi, R2), t2_msel_lo(Z2));
    X2hi = __byte_perm(X2hi, __byte_perm(X3lo, X3hi, R2 >> 16), t2_msel_hi(Z2));
    const uint32_t E1lo = __byte_perm(ta.x, ta.y, C1), E1hi = __byte_perm(ta.x, ta.y, C1 >> 16);
    const uint32_t Elo = __byte_perm(E1lo, __byte_perm(X2lo, X2hi, R1), t2_msel_lo(T1));
    const uint32_t Ehi = __byte_perm(E1hi, __byte_perm(X2lo, X2hi, R1 >> 16), t2_msel_hi(T1));
    const uint32_t bits = b4 + 3u*n3 + 8u*n4;
    cur[i].ow += bits >> 5; cur[i].ob = bits & 31u;
    out[i].x = t21_pair(raw[i].s8.x, Elo, 0x5140u, 0x1504u);
    out[i].y = t21_pair(raw[i].s8.x, Elo, 0x7362u, 0x3726u);
    out[i].z = t21_pair(raw[i].s8.y, Ehi, 0x5140u, 0x1504u);
    out[i].w = t21_pair(raw[i].s8.y, Ehi, 0x7362u, 0x3726u);
  }
}

#include "tier_prefix_cursor.cuh"
template <int KQT>
struct TpfPrefetched {
  typename Ld<F_TBE21,KQT>::R raw;
  uint32_t ow, ob;
};

// Call only when chunk+1 belongs to the current split. These reads are immutable
// and independent of the current decode's nested ranks. No MMA or cursor mutates.
template <int KQT, typename Cursor>
__device__ __forceinline__ TpfPrefetched<KQT> tpf_prefetch_next(
    const WDesc& d, int row, int chunk, int lane, int j, const Cursor& cur) {
  if (!d.tpf_counts) TPF_FAIL();
  const uint32_t h=__ldg(d.tpf_counts+(int64_t)row*(d.K>>7)+chunk);
  const uint32_t n1=h&255u, n2=(h>>8)&255u, n3=(h>>16)&255u, n4=h>>24;
  if (__any_sync(0xffffffffu, !(n4<=n3 && n3<=n2 && n2<=n1 && n1<=128u))) TPF_FAIL();
  const TpfPredictedCursor next=tpf_predict_cursor(cur.ow,cur.ob,h);
  const typename Ld<F_TBE21,KQT>::CompactC temp{next.ow,next.ob,cur.rp};
  return {Ld<F_TBE21,KQT>::load(d,row,(chunk+1)*128+8*j,lane,temp),next.ow,next.ob};
}

#undef TPF_FAIL
#endif
template <int KQT>
struct Drv<Ld<F_TBE21, KQT>> {
  using L = Ld<F_TBE21, KQT>;
  using R = typename L::R;
  using C = typename L::C;
  static constexpr int kTab = T21_TAB_BYTES;
  static __device__ __forceinline__ void prologue(const WDesc& d, uint32_t* tab) { L::prologue(d, tab); }
  static __device__ __forceinline__ void start(const WDesc& d, int row, int, int s, int S, int, C& k) {
    L::start(d, row, s, S, k);
  }
  static __device__ __forceinline__ R load(const WDesc& d, int row, int kk, int lane, const C& k) {
    return L::load(d, row, kk, lane, k);
  }
  static __device__ __forceinline__ uint4 dec(const WDesc& d, const R& r, int row, int kk, int lane,
                                              int j, C& k, const uint32_t* tab) {
    return L::dec(d, r, row, kk, lane, j, k, tab);
  }
};

// v2 decode-stage adapter for TBE21 (codec v2.1) under bi_{gemm,decode}_kernel_v2's
// phase-split schedule.  Round 1 is a SCHEDULING port only: load()/dec() are the v1
// Ld<F_TBE21> calls verbatim (bitwise identity is by construction), with the per-row ovf
// cursor carried in St across the chunk loop exactly as v1's kernel threads its C& cur.
// dec_a has nothing format-independent to hoist yet (the window word is already loaded in
// load(); the cursor advance is decode-dependent); the SWAR/T-pass trims land in dec_b in a
// later round.  TBE21 needs the smem codebook table, which the F_BF16/F_TBE v2 kernels do
// not allocate -- the kernels below allocate + prologue it under `if constexpr`, and the
// TBE21 calls take start21/dec_b21 (split-aware start, tab-taking dec) so the uniform
// F_BF16/F_TBE/F_TBE1T surface and codegen stay byte-for-byte what the opcount gate froze.
template <int KQT, int SPEC>
struct D2Sel<F_TBE21, KQT, SPEC> {
  using L = Ld<F_TBE21, KQT>;
  using R = typename L::R;
  using C = typename L::C;
  struct E {};
  // T2-style induction pointers (opledger T2, applied to TBE21's load): l1 advances 16 u16
  // per chunk, smb advances CH bytes, so load() issues no 64-bit address multiply.  The ovf
  // base pointer replaces the WDesc indirection -- its chunk offset is the carried cursor.
  struct St {
    const uint16_t* l1p;      // lane-shared chunk base; load() adds (lane & 15)
    const uint8_t* smbp;      // per-lane (kk0 includes 8*j); advances CH
    const uint32_t* ovf;      // d.t2_ovf
    C cur;
  };
  static constexpr int kTab = T21_TAB_BYTES;
  static __device__ __forceinline__ void prologue(const WDesc& d, uint32_t* tab) {
    L::prologue(d, tab);
  }
  static __device__ __forceinline__ St start21(const WDesc& d, int row, int kk0, int s, int S,
                                               int) {
    St st;
    st.l1p = reinterpret_cast<const uint16_t*>(d.planes) +
             ((int64_t)row * (d.K >> 7) + (kk0 >> 7)) * 16;
    st.smbp = d.smb + (int64_t)row * d.K + kk0;
    st.ovf = d.t2_ovf;
    L::start(d, row, s, S, st.cur);
    return st;
  }
  // Same bytes as L::load (bi_gemm.cuh Ld<F_TBE21>::load), addressed by induction.
  static __device__ __forceinline__ R load(St& st, int lane) {
    R r;
    const int j = lane & 15;
    r.c16 = (uint32_t)__ldg(st.l1p + j);
    r.s8 = __ldg(reinterpret_cast<const uint2*>(st.smbp));
    r.win = __ldg(st.ovf + st.cur.ow + (uint32_t)j);
    st.l1p += 16;
    st.smbp += CH;
    return r;
  }
  static __device__ __forceinline__ E dec_a(const WDesc&, const R&, int) { return E{}; }
  static __device__ __forceinline__ uint4 dec_b21(const WDesc& d, const R& r, const E&, int lane,
                                                  int j, St& st, const uint32_t* tab) {
    return L::dec(d, r, 0, 0, lane, j, st.cur, tab);
  }
};

template <int KQT> struct T21CompactDrv {
  using L = Ld<F_TBE21, KQT>;
  using R = typename L::R;
  using C = typename L::CompactC;
  static constexpr int kTab = T21_TAB_BYTES;
  static __device__ __forceinline__ void prologue(const WDesc& d, uint32_t* tab) { L::prologue(d, tab); }
  static __device__ __forceinline__ void start(const WDesc& d, int row, int, int s, int S, int, C& k) { L::start_compact(d, row, s, S, k); }
  static __device__ __forceinline__ R load(const WDesc& d, int row, int kk, int lane, const C& k) { return L::load(d, row, kk, lane, k); }
};
#if BI_T21_EXEC_LAYOUT
template <int KQT>
struct T21GlobalCompactDrv : T21CompactDrv<KQT> {
  static constexpr int kTab = 0;
};
template <bool B, typename Regular, int KQT, bool GLOBAL = false> struct T21DrvSelection { using type = Regular; };
template <typename Regular, int KQT> struct T21DrvSelection<true, Regular, KQT, false> { using type = T21CompactDrv<KQT>; };
template <typename Regular, int KQT> struct T21DrvSelection<true, Regular, KQT, true> { using type = T21GlobalCompactDrv<KQT>; };

// The selected GEMMs and raw-word oracle use one literal producer.
template <bool GLOBAL_CB, int NR, typename D>
__device__ __forceinline__ void t21_exec_decode_chunk(const WDesc& d, int c,
    int lane, int j, const int (&rowg)[NR], const int (&rowl)[NR],
    typename D::C* cur, __nv_bfloat16* Ws, const uint32_t* tab) {
#pragma unroll
  for (int i = 0; i < NR; i += 2) {
    typename D::R pair[2];
#pragma unroll
    for (int h = 0; h < 2; ++h)
      pair[h] = D::load(d, rowg[i + h], c * CH + 8 * j, lane, cur[i + h]);
    if constexpr (GLOBAL_CB)
      Ld<F_TBE21, 0>::dec_pair_compact_global_to_swizzled(d, pair, lane, j, cur + i,
          reinterpret_cast<uint16_t*>(Ws), rowl[i]);
    else
      Ld<F_TBE21, 0>::dec_pair_compact_to_swizzled(d, pair, lane, j, cur + i, tab,
          reinterpret_cast<uint16_t*>(Ws), rowl[i]);
    if (i + 2 < NR) __syncwarp();
  }
}
#else
template <bool B, typename Regular, int KQT> struct T21DrvSelection { using type = Regular; };
template <typename Regular, int KQT> struct T21DrvSelection<true, Regular, KQT> { using type = T21CompactDrv<KQT>; };
#endif

constexpr int T21_PIPE_SMEM = 2 * (BN + 128) * CH * 2 + T21_TAB_BYTES;
constexpr int T21_PIPE_G1_SMEM = BN * CH * 2 + T21_TAB_BYTES;

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


#if BI_T21_PIPE128
// Matched full-CTA phases may occur at different PCs in the two complete-warp roles.
__device__ __forceinline__ void t21_pipe_phase() {
  asm volatile("barrier.cta.sync 0, 1024;" ::: "memory");
}
__device__ __forceinline__ int t21_pipe_address(int row, int col) {
  return row * 128 + (col ^ ((row & 7) * 8));
}
template <int KQT>
__device__ __forceinline__ void t21_pipe128_body(const __nv_bfloat16* x, int64_t ldx, int M,
    WDesc d, int S, __nv_bfloat16* y, int64_t ldy, float* ws, unsigned char* smraw) {
  using L = Ld<F_TBE21, KQT>;
  constexpr int WWORDS = 64 * 128, STRIDE = (64 + 128) * 128;
  __nv_bfloat16* buffers = reinterpret_cast<__nv_bfloat16*>(smraw);
  uint32_t* tab = reinterpret_cast<uint32_t*>(buffers + 2 * STRIDE);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n0 = blockIdx.x * BN, split = blockIdx.y, m0 = blockIdx.z * 256;
  const int nch = d.K / CH;
  const int c0 = (int)(((int64_t)split * nch) / S), c1 = (int)(((int64_t)(split + 1) * nch) / S);
  L::prologue(d, tab);
  t21_pipe_phase();
  if (warp < 16) {
    const int half = lane >> 4, j = lane & 15, row0 = warp * 4 + half;
    int rowg[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) rowg[i] = min(n0 + row0 + 2 * i, d.N - 1);
    typename L::CompactC cur[2]; typename L::R raw[2];
    if (c0 < c1) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        L::start_compact(d, rowg[i], split, S, cur[i]);
        raw[i] = L::load(d, rowg[i], c0 * CH + 8 * j, lane, cur[i]);
      }
    }
    // One prime, n-1 overlapped fills, and a final no-fill phase match consumers.
    for (int c = c0; c <= c1; ++c) {
      if (c < c1) {
        __nv_bfloat16* W = buffers + ((c - c0) & 1) * STRIDE;
        __nv_bfloat16* X = W + WWORDS;
#pragma unroll
        for (int it = 0; it < 4; ++it) {
          const int index = threadIdx.x + 512 * it, row = index >> 4, cv = index & 15;
          const int m = m0 + row;
          uint4 v = make_uint4(0u, 0u, 0u, 0u);
          if (m < M) v = __ldg(reinterpret_cast<const uint4*>(x + (int64_t)m * ldx + c * CH + cv * 8));
          *reinterpret_cast<uint4*>(X + t21_pipe_address(row, cv * 8)) = v;
        }
        L::dec_pair_compact_to_swizzled(d, raw, lane, j, cur, tab,
                                        reinterpret_cast<uint16_t*>(W), row0);
        if (c + 1 < c1) {
#pragma unroll
          for (int i = 0; i < 2; ++i) raw[i] = L::load(d, rowg[i], (c + 1) * CH + 8 * j, lane, cur[i]);
        }
      }
      t21_pipe_phase();
    }
  } else {
    const int worker = warp - 16, wn = worker & 7, wg = worker >> 3;
    float acc[4][4];
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[mt][e] = 0.0f;
    t21_pipe_phase();
    for (int c = c0; c < c1; ++c) {
      const __nv_bfloat16* W = buffers + ((c - c0) & 1) * STRIDE;
      const __nv_bfloat16* X = W + WWORDS;
#pragma unroll
      for (int kk = 0; kk < CH; kk += 16) {
        uint32_t b[2];
        ldsm_x2(b, W + t21_pipe_address(wn * 8 + (lane & 7), kk + ((lane >> 3) & 1) * 8));
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
          uint32_t a[4];
          ldsm_x4(a, X + t21_pipe_address(wg * 64 + mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8,
                                         kk + (lane >> 4) * 8));
          mma16816(acc[mt], a, b);
        }
      }
      t21_pipe_phase();
    }
    const int g = lane >> 2, t4 = lane & 3;
    const int n = n0 + wn * 8 + 2 * t4;
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int m = m0 + wg * 64 + mt * 16 + g + 8 * h;
        if (m >= M) continue;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int nn = n + e;
          if (nn >= d.N) continue;
          const float v = acc[mt][2 * h + e];
          if (S == 1) {
            const float o = d.bias ? __fadd_rn(v, __bfloat162float(d.bias[nn])) : v;
            y[(int64_t)m * ldy + nn] = __float2bfloat16_rn(o);
          } else ws[((int64_t)split * M + m) * d.N + nn] = v;
        }
      }
  }
}

template <int KQT>
__device__ __forceinline__ void t21_pipe_decode_body(WDesc d, int S,
    __nv_bfloat16* out, unsigned char* smraw) {
  using L = Ld<F_TBE21, KQT>;
  __nv_bfloat16* W = reinterpret_cast<__nv_bfloat16*>(smraw);
  uint32_t* tab = reinterpret_cast<uint32_t*>(W + BN * CH);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n0 = blockIdx.x * BN, split = blockIdx.y, nch = d.K / CH;
  const int c0 = (int)(((int64_t)split * nch) / S), c1 = (int)(((int64_t)(split + 1) * nch) / S);
  L::prologue(d, tab);
  t21_pipe_phase();
  if (warp < 16) {
    const int j = lane & 15, row0 = warp * 4 + (lane >> 4);
    typename L::CompactC cur[2]; typename L::R raw[2];
    int rows[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) rows[i] = min(n0 + row0 + 2 * i, d.N - 1);
    if (c0 < c1) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        L::start_compact(d, rows[i], split, S, cur[i]);
        raw[i] = L::load(d, rows[i], c0 * CH + 8 * j, lane, cur[i]);
      }
    }
    for (int c = c0; c < c1; ++c) {
      L::dec_pair_compact_to_swizzled(d, raw, lane, j, cur, tab,
                                      reinterpret_cast<uint16_t*>(W), row0);
      t21_pipe_phase();
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int row = row0 + 2 * i;
        if (n0 + row < d.N)
          *reinterpret_cast<uint4*>(out + (int64_t)(n0 + row) * d.K + c * CH + 8 * j) =
              *reinterpret_cast<const uint4*>(W + t21_pipe_address(row, 8 * j));
      }
      t21_pipe_phase();
      if (c + 1 < c1) {
#pragma unroll
        for (int i = 0; i < 2; ++i) raw[i] = L::load(d, rows[i], (c + 1) * CH + 8 * j, lane, cur[i]);
      }
    }
  } else {
    // Consumers participate in every generation: no early exit before a CTA phase.
    for (int c = c0; c < c1; ++c) {
      t21_pipe_phase();
      t21_pipe_phase();
    }
  }
}
#endif

// ---------------------------------------------------------------- the kernel
// Shared by the opt-in row-geometry launch and its row indexing. Multiple
// 256-row CTAs must advance by256, rather than the legacy128-row tile stride.
#if BI_T21_ROW_GEOMETRY
template <int MT, int WG>
__host__ __device__ constexpr int tbe21_row_span() { return MT * 16 * WG; }
#endif

// grid (ceil(N/64), S, ceil(M/(64*WG))); 256*WG threads; dynamic smem
// (64 + MT*16*WG) * LDS * 2 bytes.  MT = m16 tiles per WARP GROUP (1..4), WG = warp groups
// per CTA (1 or 2 -> a 64- or 128-row tile; see the WG note above for why the bits do not
// move).  S == 1: y written directly.  S > 1: fp32 partials to ws[s][m][n] (row stride N),
// reduced by bi_reduce_kernel in split order.
template <int FMT, int KQT, int MT, int WG>
#if BI_T21_EXEC_LAYOUT
// The compact MT2/WG2 instance was screened at64 registers without spills.
// Two512-thread CTAs fit the register budget; runtime residency stays separate.
__global__ void __launch_bounds__(256 * WG, FMT == F_TBE21 && KQT == 0 && MT == 2 && WG == 2 ? 2 :
    (WG == 1 ? (FMT == F_TBE21 ? BI_T21_V1_MINBLOCKS : 2) : 1))
#else
__global__ void __launch_bounds__(256 * WG, WG == 1 ? (FMT == F_TBE21 ? BI_T21_V1_MINBLOCKS : 2) : 1)
#endif
bi_gemm_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx, int M, WDesc d, int S,
               __nv_bfloat16* __restrict__ y, int64_t ldy, float* __restrict__ ws) {
  extern __shared__ __align__(16) unsigned char smraw[];
#if BI_T21_PIPE128
  if constexpr (BI_T21_WIDE128 && FMT == F_TBE21 && MT == 2 && WG == 4) {
    static_assert(BI_T21_PACKSCAN2 && !BI_T21_SCAN4 && !BI_T21_ASYNC_X, "pipe128 requires PACKSCAN2 and synchronous producer X staging");
    t21_pipe128_body<KQT>(x, ldx, M, d, S, y, ldy, ws, smraw);
    return;
  }
#endif

  __nv_bfloat16* Ws = reinterpret_cast<__nv_bfloat16*>(smraw);
#if BI_T21_EXEC_LAYOUT
  constexpr bool EXEC_LAYOUT = FMT == F_TBE21 && KQT == 0 &&
      ((MT == 2 && WG == 2) || (MT == 8 && WG == 1));
  constexpr bool GLOBAL_CB = EXEC_LAYOUT && MT == 8 && WG == 1;
  constexpr int EXEC_LDS = EXEC_LAYOUT ? CH : LDS;
  __nv_bfloat16* Xs = Ws + BN * EXEC_LDS;
#else
  __nv_bfloat16* Xs = Ws + BN * LDS;
#endif
  constexpr bool COMPACT128 = BI_T21_WIDE128 && FMT == F_TBE21 && MT == 2 && WG == 4;
#if BI_T21_PACKSCAN4
  constexpr bool QUAD = FMT == F_TBE21;
  static_assert(!QUAD || WG == 1 || WG == 2, "PACKSCAN4 requires WG=1 or WG=2");
#if BI_T21_EXEC_LAYOUT
  using D = typename T21DrvSelection<COMPACT128 || QUAD || EXEC_LAYOUT, Drv<Ld<FMT, KQT>>, KQT>::type;
#else
  using D = typename T21DrvSelection<COMPACT128 || QUAD, Drv<Ld<FMT, KQT>>, KQT>::type;
#endif
#else
#if BI_T21_EXEC_LAYOUT
  using D = typename T21DrvSelection<COMPACT128 || EXEC_LAYOUT, Drv<Ld<FMT, KQT>>, KQT, GLOBAL_CB>::type;
#else
  using D = typename T21DrvSelection<COMPACT128, Drv<Ld<FMT, KQT>>, KQT>::type;
#endif
#endif
  // TBE2 codebook table: after Xs, only allocated (launch_fmt) when D::kTab > 0.
#if BI_T21_EXEC_LAYOUT
  uint32_t* tab = reinterpret_cast<uint32_t*>(Xs + MT * 16 * WG * EXEC_LDS);
#else
  uint32_t* tab = reinterpret_cast<uint32_t*>(Xs + MT * 16 * WG * LDS);
#endif
  constexpr int THR = 256 * WG;          // threads per CTA
  static_assert(!COMPACT128 || (BI_T21_PACKSCAN2 && !BI_T21_SCAN4), "compact M128 requires paired scans and SCAN4 off");
#if BI_T21_PACKSCAN4
  constexpr int NR = QUAD ? 4 : (COMPACT128 ? 2 : 4 / WG);
#else
  constexpr int NR = COMPACT128 ? 2 : 4 / WG;
#endif             // weight-row PAIRS each warp decodes (WG=1: 4, WG=2: 2)
#if BI_T21_ROW_GEOMETRY
  constexpr int ROW_STRIDE = FMT == F_TBE21 && KQT == 0 && MT == 8 && WG == 2
      ? tbe21_row_span<MT, WG>() : MTILE * WG;
  const int n0 = blockIdx.x * BN, s = blockIdx.y, m0 = blockIdx.z * ROW_STRIDE;
#else
  const int n0 = blockIdx.x * BN, s = blockIdx.y, m0 = blockIdx.z * (MTILE * WG);
#endif
  const int nch = d.K / CH;
  const int c0 = (int)(((int64_t)s * nch) / S), c1 = (int)(((int64_t)(s + 1) * nch) / S);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#if BI_T21_PACKSCAN4
  // Complete warps 0..7 own weights. Other warps still stage X and join every
  // MMA/barrier; no cursor or raw value is read by a non-owner.
  const bool weight_owner = QUAD ? warp < 8 : (!COMPACT128 || warp < 16);
#else
  const bool weight_owner = !COMPACT128 || warp < 16;
#endif
  const int wn = warp & 7;               // which 8 of the CTA's 64 weight rows this warp MMAs
  const int wg = warp >> 3;              // warp group -> which MT*16 activation rows (0 at WG=1)
  const int half = lane >> 4, j = lane & 15;
  // The CTA's 64 weight rows are spread over its 8*WG warps: 2*NR rows per warp, a half-warp
  // per row, exactly as at WG=1 (where 2*NR == 8 and this is the old expression verbatim).
  // Which lane decodes a weight does not change the decoded value -- Ld<>::dec is a pure
  // function of the loaded bytes -- and Ws[] ends up byte-identical either way.
#if BI_T21_PACKSCAN4
  const int decode_warp = QUAD ? min(warp, 7) : (COMPACT128 ? min(warp, 15) : warp);
#else
  const int decode_warp = COMPACT128 ? min(warp, 15) : warp;
#endif
  int rowl[NR], rowg[NR];
#pragma unroll
  for (int i = 0; i < NR; ++i) {
    rowl[i] = decode_warp * (2 * NR) + 2 * i + half;
    rowg[i] = min(n0 + rowl[i], d.N - 1);          // tail rows: duplicate loads, never stored
  }
  float acc[MT][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[mt][e] = 0.0f;

  typename D::R raw[NR];
  typename D::C cur[NR];                 // per-row decode cursor (TBE2); empty for the others
  if constexpr (D::kTab > 0) {
    D::prologue(d, tab);
    __syncthreads();
  }
#if BI_T21_ASYNC_X
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr bool ASYNC_T21_X = FMT == F_TBE21;
#else
  constexpr bool ASYNC_T21_X = false;
#endif
#endif
  if (weight_owner && c0 < c1) {
#pragma unroll
    for (int i = 0; i < NR; ++i) {
      if constexpr (COMPACT128) Ld<F_TBE21, KQT>::start_compact(d, rowg[i], s, S, cur[i]);
      else D::start(d, rowg[i], c0, s, S, lane, cur[i]);
#if BI_T21_EXEC_LAYOUT
      if constexpr (!EXEC_LAYOUT)
#endif
      raw[i] = D::load(d, rowg[i], c0 * CH + 8 * j, lane, cur[i]);
    }
  }
  for (int c = c0; c < c1; ++c) {
#if BI_T21_TPF_PREFETCH
    typename D::R tpf_next_raw[NR];
    if constexpr (FMT==F_TBE21 && (WG==2 || BI_T21_TPF_PREFETCH_WG1)) {
      if (c+1<c1) {
#pragma unroll
        for (int i=0;i<NR;++i)
          tpf_next_raw[i]=tpf_prefetch_next<KQT>(d,rowg[i],c,lane,j,cur[i]).raw;
      }
    }
#endif

#if BI_T21_SCAN4
    typename Ld<F_TBE21, KQT>::Fixed next_fixed[NR];
    if constexpr (FMT == F_TBE21) {
      if (c + 1 < c1) {
#pragma unroll
        for (int i = 0; i < NR; ++i)
          next_fixed[i] = Ld<F_TBE21, KQT>::load_fixed(d, rowg[i], (c + 1) * CH + 8 * j, lane);
      }
    }
#endif
    // activations: rows m0 .. m0 + MT*16*WG, columns c*CH .. c*CH + 127 (rows >= M -> zero)
#pragma unroll
    for (int it = 0; it < MT; ++it) {
      const int i = threadIdx.x + THR * it;           // MT*16*WG rows * 16 vectors = MT*THR
      const int r = i >> 4, cv = i & 15;
      const int m = m0 + r;
#if BI_T21_ASYNC_X
      if constexpr (ASYNC_T21_X) {
        const int source_row = min(m, M - 1);
        cp_async_t21_x16(Xs + r * LDS + cv * 8,
                        x + (int64_t)source_row * ldx + c * CH + cv * 8, m < M);
      } else {
#endif
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (m < M) v = __ldg(reinterpret_cast<const uint4*>(x + (int64_t)m * ldx + c * CH + cv * 8));
#if BI_T21_EXEC_LAYOUT
      *reinterpret_cast<uint4*>(exec_smem_ptr<EXEC_LAYOUT>(Xs, r, cv * 8)) = v;
#else
      *reinterpret_cast<uint4*>(Xs + r * LDS + cv * 8) = v;
#endif
#if BI_T21_ASYNC_X
      }
#endif
    }
#if BI_T21_ASYNC_X
    if constexpr (ASYNC_T21_X) cp_async_t21_x_commit();
#endif
    if (weight_owner) {
#if BI_T21_EXEC_LAYOUT
    if constexpr (EXEC_LAYOUT) {
      // Load a pair only at its decode phase, instead of carrying next-chunk
      // raw vectors through the tensor-core chunk. A warp phase boundary
      // prevents the second pair's decode temporaries overlapping the first.
      t21_exec_decode_chunk<GLOBAL_CB, NR, D>(d, c, lane, j, rowg, rowl, cur, Ws, tab);
    } else {
#endif
    // weights: decode this chunk into smem, then issue the next chunk's loads
#if BI_T21_PACKSCAN4
    if constexpr (QUAD) {
      uint4 w[4];
      Ld<F_TBE21, KQT>::dec_quad(d, raw, lane, j, cur, tab, w);
#pragma unroll
      for (int i = 0; i < 4; ++i)
#if BI_T21_EXEC_LAYOUT
        *reinterpret_cast<uint4*>(exec_smem_ptr<EXEC_LAYOUT>(Ws, rowl[i], 8 * j)) = w[i];
#else
        *reinterpret_cast<uint4*>(Ws + rowl[i] * LDS + 8 * j) = w[i];
#endif
    } else
#endif
#if BI_T21_PACKSCAN2
    if constexpr (FMT == F_TBE21 && NR >= 2) {
#pragma unroll
      for (int i = 0; i < NR; i += 2) {
        if constexpr (COMPACT128) {
          Ld<F_TBE21, KQT>::dec_pair_compact_to_shared(d, raw + i, lane, j, cur + i, tab,
              reinterpret_cast<uint16_t*>(Ws), warp * 4 + half + 2 * i, LDS);
        } else {
          uint4 w[2];
          #if BI_T21_PAIR_VOTE
          if constexpr (WG == 2) Ld<F_TBE21, KQT>::dec_pair_ballot(d,raw+i,lane,j,cur+i,tab,w);
          else
#endif
#if BI_T21_TIER_PREFIX
          if constexpr (WG == 2 || BI_T21_TPF_WG1) tpf_dec_pair<KQT>(d, raw+i, lane, j, cur+i, w, rowg+i, c);
          else
#endif
          Ld<F_TBE21, KQT>::dec_pair(d, raw + i, lane, j, cur + i, tab, w);
#pragma unroll
          for (int h = 0; h < 2; ++h)
#if BI_T21_EXEC_LAYOUT
            *reinterpret_cast<uint4*>(exec_smem_ptr<EXEC_LAYOUT>(Ws, rowl[i + h], 8 * j)) = w[h];
#else
            *reinterpret_cast<uint4*>(Ws + rowl[i + h] * LDS + 8 * j) = w[h];
#endif
        }
      }
    } else
#endif
    {
#pragma unroll
    for (int i = 0; i < NR; ++i) {
      const uint4 w = D::dec(d, raw[i], rowg[i], c * CH + 8 * j, lane, j, cur[i], tab);
#if BI_T21_EXEC_LAYOUT
      *reinterpret_cast<uint4*>(exec_smem_ptr<EXEC_LAYOUT>(Ws, rowl[i], 8 * j)) = w;
#else
      *reinterpret_cast<uint4*>(Ws + rowl[i] * LDS + 8 * j) = w;
#endif
    }
    }
    if (c + 1 < c1) {
#if BI_T21_SCAN4
      if constexpr (FMT == F_TBE21) {
#pragma unroll
        for (int i = 0; i < NR; ++i) {
          raw[i].c16 = next_fixed[i].c16;
          raw[i].s8.x = next_fixed[i].s0;
          raw[i].s8.y = next_fixed[i].s1;
          raw[i].win = __ldg(d.t2_ovf + cur[i].ow + (uint32_t)j);
        }
      } else
#endif
      {
#if BI_T21_TPF_PREFETCH
      if constexpr (FMT==F_TBE21 && (WG==2 || BI_T21_TPF_PREFETCH_WG1)) {
#pragma unroll
        for (int i=0;i<NR;++i) raw[i]=tpf_next_raw[i];
      } else
#endif
      {
#pragma unroll
      for (int i = 0; i < NR; ++i) raw[i] = D::load(d, rowg[i], (c + 1) * CH + 8 * j, lane, cur[i]);
      }
      }
    }
#if BI_T21_EXEC_LAYOUT
    }
#endif
    }
#if BI_T21_ASYNC_X
    if constexpr (ASYNC_T21_X) cp_async_t21_x_wait();
#endif
    __syncthreads();
#if BI_T21_EXEC_LAYOUT
// Keep one K16 operand set live for the selected instances. K order is unchanged;
// compiler resource receipts, rather than loop shape, decide residency.
#pragma unroll (EXEC_LAYOUT ? 1 : 8)
#else
#pragma unroll
#endif
    for (int kk = 0; kk < CH; kk += 16) {
      uint32_t b[2];
#if BI_T21_EXEC_LAYOUT
      ldsm_x2(b, exec_smem_ptr<EXEC_LAYOUT>(Ws, wn * 8 + (lane & 7), kk + ((lane >> 3) & 1) * 8));
#else
      ldsm_x2(b, Ws + (wn * 8 + (lane & 7)) * LDS + kk + ((lane >> 3) & 1) * 8);
#endif
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        uint32_t a[4];
#if BI_T21_EXEC_LAYOUT
        ldsm_x4(a, exec_smem_ptr<EXEC_LAYOUT>(Xs, wg * MT * 16 + mt * 16 + (lane & 7) +
                       ((lane >> 3) & 1) * 8, kk + (lane >> 4) * 8));
#else
        ldsm_x4(a, Xs + (wg * MT * 16 + mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LDS + kk +
                       (lane >> 4) * 8);
#endif
        mma16816(acc[mt], a, b);
      }
    }
    __syncthreads();
  }
  // epilogue: c0,c1 -> (row g, cols 2t, 2t+1); c2,c3 -> (row g+8, same cols)
  const int g = lane >> 2, t4 = lane & 3;
  const int n = n0 + wn * 8 + 2 * t4;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int m = m0 + wg * MT * 16 + mt * 16 + g + 8 * h;
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

// Decode-only twin of the weight path above: the same Drv<Ld<FMT>> calls in the same order
// (prologue, start at c0, load one chunk ahead, dec), the same CTA row mapping (rowl/rowg with
// the tail clamp) and the same split traversal c0..c1 -- only the destination differs: the
// uint4 that bi_gemm_kernel writes into Ws is written to out[row][c*CH + 8j] instead.  So the
// bytes this kernel emits are the bytes the MMA consumes.  grid (ceil(N/64), S), 256*WG threads.
#if BI_T21_EXEC_LAYOUT
// Dump the exact selected producer's XOR tile, with no MMA/conversion that
// could canonicalize a NaN payload or signed zero.
template <int MT, int WG>
__global__ void __launch_bounds__(256 * WG)
t21_exec_decode_kernel(WDesc d, int S, __nv_bfloat16* __restrict__ out) {
  static_assert((MT == 2 && WG == 2) || (MT == 8 && WG == 1), "unsupported EXEC oracle geometry");
  constexpr bool GLOBAL_CB = MT == 8 && WG == 1;
  using D = typename T21DrvSelection<true, Drv<Ld<F_TBE21, 0>>, 0, GLOBAL_CB>::type;
  constexpr int NR = 4 / WG;
  extern __shared__ __align__(16) unsigned char smraw[];
  __nv_bfloat16* Ws = reinterpret_cast<__nv_bfloat16*>(smraw);
  uint32_t* tab = reinterpret_cast<uint32_t*>(Ws + (BN + MT * 16 * WG) * CH);
  const int n0 = blockIdx.x * BN, s = blockIdx.y, nch = d.K / CH;
  const int c0 = (int)(((int64_t)s * nch) / S), c1 = (int)(((int64_t)(s + 1) * nch) / S);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int half = lane >> 4, j = lane & 15;
  int rowl[NR], rowg[NR];
#pragma unroll
  for (int i = 0; i < NR; ++i) {
    rowl[i] = warp * (2 * NR) + 2 * i + half;
    rowg[i] = min(n0 + rowl[i], d.N - 1);
  }
  typename D::C cur[NR];
  if constexpr (D::kTab > 0) {
    D::prologue(d, tab);
    __syncthreads();
  }
  if (c0 < c1) {
#pragma unroll
    for (int i = 0; i < NR; ++i) D::start(d, rowg[i], c0, s, S, lane, cur[i]);
  }
  for (int c = c0; c < c1; ++c) {
    t21_exec_decode_chunk<GLOBAL_CB, NR, D>(d, c, lane, j, rowg, rowl, cur, Ws, tab);
    __syncthreads();
#pragma unroll
    for (int i = 0; i < NR; ++i) {
      const int row = n0 + rowl[i];
      if (row < d.N) {
        const uint4 words = *reinterpret_cast<const uint4*>(exec_smem_ptr<true>(Ws, rowl[i], 8 * j));
        *reinterpret_cast<uint4*>(out + (int64_t)row * d.K + c * CH + 8 * j) = words;
      }
    }
    __syncthreads();
  }
}
#endif

template <int FMT, int KQT, int WG>
__global__ void __launch_bounds__(256 * WG, WG == 1 ? 2 : 1)
bi_decode_kernel(WDesc d, int S, __nv_bfloat16* __restrict__ out) {
  extern __shared__ __align__(16) unsigned char smraw[];
#if BI_T21_PIPE128
  if constexpr (BI_T21_WIDE128 && FMT == F_TBE21 && WG == 4) {
    t21_pipe_decode_body<KQT>(d, S, out, smraw);
    return;
  }
#endif

  constexpr bool COMPACT128 = BI_T21_WIDE128 && FMT == F_TBE21 && WG == 4;
#if BI_T21_PACKSCAN4
  constexpr bool QUAD = FMT == F_TBE21;
  static_assert(!QUAD || WG == 1 || WG == 2, "PACKSCAN4 decode requires WG=1 or WG=2");
  using D = typename T21DrvSelection<COMPACT128 || QUAD, Drv<Ld<FMT, KQT>>, KQT>::type;
#else
  using D = typename T21DrvSelection<COMPACT128, Drv<Ld<FMT, KQT>>, KQT>::type;
#endif
  uint32_t* tab = reinterpret_cast<uint32_t*>(smraw);
  static_assert(!COMPACT128 || (BI_T21_PACKSCAN2 && !BI_T21_SCAN4), "compact decode requires paired scans and SCAN4 off");
#if BI_T21_PACKSCAN4
  constexpr int NR = QUAD ? 4 : (COMPACT128 ? 2 : 4 / WG);
#else
  constexpr int NR = COMPACT128 ? 2 : 4 / WG;
#endif
  const int n0 = blockIdx.x * BN, s = blockIdx.y;
  const int nch = d.K / CH;
  const int c0 = (int)(((int64_t)s * nch) / S), c1 = (int)(((int64_t)(s + 1) * nch) / S);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int half = lane >> 4, j = lane & 15;
  int rowl[NR], rowg[NR];
#pragma unroll
  for (int i = 0; i < NR; ++i) {
    rowl[i] = warp * (2 * NR) + 2 * i + half;
    rowg[i] = min(n0 + rowl[i], d.N - 1);
  }
  typename D::R raw[NR];
  typename D::C cur[NR];
  if constexpr (D::kTab > 0) {
    D::prologue(d, tab);
    __syncthreads();
  }
  // No CTA barriers follow; all non-producer warps exit together after the prologue.
  if constexpr (COMPACT128) { if (warp >= 16) return; }
#if BI_T21_PACKSCAN4
  if constexpr (QUAD) { if (warp >= 8) return; }
#endif
  if (c0 < c1) {
#pragma unroll
    for (int i = 0; i < NR; ++i) {
      if constexpr (COMPACT128) Ld<F_TBE21, KQT>::start_compact(d, rowg[i], s, S, cur[i]);
      else D::start(d, rowg[i], c0, s, S, lane, cur[i]);
      raw[i] = D::load(d, rowg[i], c0 * CH + 8 * j, lane, cur[i]);
    }
  }
  for (int c = c0; c < c1; ++c) {
#if BI_T21_TIER_PREFIX && BI_T21_TPF_PREFETCH
    typename D::R tpf_next_raw[NR];
    if constexpr (FMT==F_TBE21 && (WG==2 || BI_T21_TPF_PREFETCH_WG1)) {
      if(c+1<c1) {
#pragma unroll
        for(int i=0;i<NR;++i) tpf_next_raw[i]=tpf_prefetch_next<KQT>(d,rowg[i],c,lane,j,cur[i]).raw;
      }
    }
#endif

#if BI_T21_PACKSCAN4
    if constexpr (QUAD) {
      uint4 w[4];
      Ld<F_TBE21, KQT>::dec_quad(d, raw, lane, j, cur, tab, w);
#pragma unroll
      for (int i = 0; i < 4; ++i)
        if (n0 + rowl[i] < d.N)
          *reinterpret_cast<uint4*>(out + (int64_t)(n0 + rowl[i]) * d.K + c * CH + 8 * j) = w[i];
    } else
#endif
#if BI_T21_PACKSCAN2
    if constexpr (FMT == F_TBE21 && NR >= 2) {
#pragma unroll
      for (int i = 0; i < NR; i += 2) {
        uint4 w[2];
        if constexpr (COMPACT128) Ld<F_TBE21, KQT>::dec_pair_compact(d, raw + i, lane, j, cur + i, tab, w);
        else {
#if BI_T21_TIER_PREFIX
          if constexpr(WG==2 || BI_T21_TPF_WG1) tpf_dec_pair<KQT>(d,raw+i,lane,j,cur+i,w,rowg+i,c);
          else
#endif
          Ld<F_TBE21,KQT>::dec_pair(d,raw+i,lane,j,cur+i,tab,w);
        }
#pragma unroll
        for (int h = 0; h < 2; ++h)
          if (n0 + rowl[i + h] < d.N)
            *reinterpret_cast<uint4*>(out + (int64_t)(n0 + rowl[i + h]) * d.K + c * CH + 8 * j) = w[h];
      }
    } else
#endif
    {
#pragma unroll
    for (int i = 0; i < NR; ++i) {
      const uint4 w = D::dec(d, raw[i], rowg[i], c * CH + 8 * j, lane, j, cur[i], tab);
      if (n0 + rowl[i] < d.N)
        *reinterpret_cast<uint4*>(out + (int64_t)(n0 + rowl[i]) * d.K + c * CH + 8 * j) = w;
    }
    }
    if (c + 1 < c1) {
#if BI_T21_TIER_PREFIX && BI_T21_TPF_PREFETCH
      if constexpr(FMT==F_TBE21 && (WG==2 || BI_T21_TPF_PREFETCH_WG1)) {
#pragma unroll
        for(int i=0;i<NR;++i) raw[i]=tpf_next_raw[i];
      } else
#endif
      {
#pragma unroll
      for (int i = 0; i < NR; ++i) raw[i] = D::load(d, rowg[i], (c + 1) * CH + 8 * j, lane, cur[i]);
      }
    }
  }
}

// Load-time RESIDENT split checkpoints for TBE21 (codec v2.1 stores no chunk index): the decode
// walk of bi_decode_kernel at S = 1 -- same Drv<Ld<F_TBE21>> prologue/start/load/dec, same CTA
// row mapping -- recording, for the launch split count S, the cursor at the first chunk of every
// split s = 1..S-1 as a bit offset relative to the row start: ckpt[row*(S-1) + s-1].  Lane 0 of
// each half-warp writes its row's entries.  grid (ceil(N/64)), 256*WG threads, smem kTab.
template <int KQT, int WG>
__global__ void __launch_bounds__(256 * WG, WG == 1 ? 2 : 1)
bi_t21_ckpt_kernel(WDesc d, int S, uint32_t* __restrict__ ckpt) {
  extern __shared__ __align__(16) unsigned char smraw[];
  using D = Drv<Ld<F_TBE21, KQT>>;
  uint32_t* tab = reinterpret_cast<uint32_t*>(smraw);
  constexpr int NR = 4 / WG;
  const int n0 = blockIdx.x * BN;
  const int nch = d.K / CH;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int half = lane >> 4, j = lane & 15;
  int rowl[NR], rowg[NR];
#pragma unroll
  for (int i = 0; i < NR; ++i) {
    rowl[i] = warp * (2 * NR) + 2 * i + half;
    rowg[i] = min(n0 + rowl[i], d.N - 1);
  }
  typename D::R raw[NR];
  typename D::C cur[NR];
  D::prologue(d, tab);
  __syncthreads();
#pragma unroll
  for (int i = 0; i < NR; ++i) {
    D::start(d, rowg[i], 0, 0, 1, lane, cur[i]);
    raw[i] = D::load(d, rowg[i], 8 * j, lane, cur[i]);
  }
  int s_next = 1;
  int c_next = (int)(((int64_t)s_next * nch) / S);       // first chunk of split s_next
  for (int c = 0; c < nch; ++c) {
    if (s_next < S && c == c_next) {                     // split starts strictly increase (S <= nch)
#pragma unroll
      for (int i = 0; i < NR; ++i)
        if (j == 0 && n0 + rowl[i] < d.N)
          ckpt[(int64_t)(n0 + rowl[i]) * (S - 1) + (s_next - 1)] =
              32u * (cur[i].ow - __ldg(d.t21_rowoff + rowg[i])) + cur[i].ob;
      ++s_next;
      c_next = (int)(((int64_t)s_next * nch) / S);
    }
#pragma unroll
    for (int i = 0; i < NR; ++i) (void)D::dec(d, raw[i], rowg[i], c * CH + 8 * j, lane, j, cur[i], tab);
    if (c + 1 < nch) {
#pragma unroll
      for (int i = 0; i < NR; ++i) raw[i] = D::load(d, rowg[i], (c + 1) * CH + 8 * j, lane, cur[i]);
    }
  }
}

#if BI_T21_TIER_PREFIX
// Redundant load-time metadata, derived from the original scalar stream walk.
// No split checkpoints or tier-prefix metadata are used to produce these counts.
// All warp lanes participate in scalar collectives, including clamped tail rows.
template <int KQT>
__global__ void __launch_bounds__(256, 2)
bi_tpf_headers_kernel(WDesc d, uint32_t* __restrict__ headers) {
  extern __shared__ __align__(16) unsigned char smraw[];
  using L = Ld<F_TBE21, KQT>;
  uint32_t* tab = reinterpret_cast<uint32_t*>(smraw);
  constexpr int NR = 4;
  const int n0 = blockIdx.x * BN, nch = d.K / CH;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int half = lane >> 4, j = lane & 15;
  int rowl[NR], rowg[NR];
  typename L::R raw[NR];
  typename L::C cur[NR];
  L::prologue(d, tab);
  __syncthreads();
#pragma unroll
  for (int i = 0; i < NR; ++i) {
    rowl[i] = warp * (2 * NR) + 2 * i + half;
    rowg[i] = min(n0 + rowl[i], d.N - 1);
    L::start(d, rowg[i], 0, 1, cur[i]);
    raw[i] = L::load(d, rowg[i], 8 * j, lane, cur[i]);
  }
  for (int c = 0; c < nch; ++c) {
#pragma unroll
    for (int i = 0; i < NR; ++i) {
      uint32_t header;
      (void)L::dec_header(d, raw[i], rowg[i], c * CH + 8 * j, lane, j,
                          cur[i], tab, header);
      if (j == 0 && n0 + rowl[i] < d.N)
        headers[(int64_t)(n0 + rowl[i]) * nch + c] = header;
    }
    if (c + 1 < nch) {
#pragma unroll
      for (int i = 0; i < NR; ++i)
        raw[i] = L::load(d, rowg[i], (c + 1) * CH + 8 * j, lane, cur[i]);
    }
  }
}
#endif

inline WDesc to_wdesc(const BiDesc& b) {
  WDesc d{};
  d.N = b.N; d.K = b.K; d.w = b.w; d.qs = b.qs; d.sc = b.sc;
  d.planes = b.planes; d.smb = b.smb; d.esc = b.esc; d.rowparam = b.rowparam; d.escidx = b.escidx;
#ifndef GEOR_REFINED_CONTEXT_ONLY
  d.kq.a0 = b.a0; d.kq.a1 = b.a1; d.kq.a2 = b.a2; d.kq.a3 = b.a3; d.kq.a4 = b.a4;
  d.kq.grid = b.grid; d.kq.ksigns = b.ksigns; d.kq.bias = nullptr; d.kq.N = b.N; d.kq.K = b.K;
#endif
  d.bias = b.bias;
  d.t2_ovf = b.ovf; d.t2_len = b.len; d.t2_grp = b.grp; d.t2_cb = b.cb; d.t2_rowp = b.rowp2;
  d.t2_nchp = b.nchp; d.t2_ngrp = b.ngrp; d.t2_ncb = b.ncb;
  d.t21_rowoff = b.rowoff; d.t21_ckpt = b.ckpt; d.t21_ckS = b.ckS;
  d.escidx16 = b.escidx16; d.rowsb = b.rowsb; d.esc4 = b.esc4; d.nsb = b.nsb;
#if BI_T21_TIER_PREFIX
  d.tpf_counts=b.tpf_counts;
#endif
  return d;
}

// The WG=2 kernel needs 52,224 B of dynamic smem, above the 48 KiB default cap, so it must
// opt in once per (kernel instantiation, device).  Idempotent, so the unsynchronised static
// is safe: a race costs a second identical call, never a wrong launch.
template <int FMT, int KQT, int MT, int WG>
inline cudaError_t optin_smem(size_t smem) {
  static bool done = false;
#if BI_T21_EXEC_LAYOUT
  constexpr bool FORCE = FMT == F_TBE21 && KQT == 0 && MT == 8 && WG == 1;
  if (done || (!FORCE && smem <= 48 * 1024)) return cudaSuccess;
#else
  if (done || smem <= 48 * 1024) return cudaSuccess;
#endif
  cudaError_t e = cudaFuncSetAttribute(reinterpret_cast<const void*>(&bi_gemm_kernel<FMT, KQT, MT, WG>),
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  if (e == cudaSuccess) done = true;
  return e;
}

// TBE21 row geometry uses the same kernel body and split reduction as launch_fmt, with
// two source-validated row-count specializations selected by the isolated adapter.
template <int MT>
inline cudaError_t launch_tbe21_row_geometry(const __nv_bfloat16* x, int64_t ldx, int M,
                                              const BiDesc& bd, int S, __nv_bfloat16* y,
                                              int64_t ldy, float* ws, cudaStream_t st) {
  static_assert(MT == 2 || MT == 8, "unsupported TBE21 row geometry");
  constexpr int WG = 2;
  constexpr size_t TAB = (size_t)Drv<Ld<F_TBE21, 0>>::kTab;
  const WDesc d = to_wdesc(bd);
  const size_t smem = (size_t)(BN + MT * 16 * WG) * LDS * 2 + TAB;
  cudaError_t e = optin_smem<F_TBE21, 0, MT, WG>(smem);
  if (e != cudaSuccess) return e;
#if BI_T21_ROW_GEOMETRY
  constexpr int ROW_SPAN = tbe21_row_span<MT, WG>();
  bi_gemm_kernel<F_TBE21, 0, MT, WG><<<dim3((d.N + BN - 1) / BN, S, (M + ROW_SPAN - 1) / ROW_SPAN), 512, smem, st>>>(
#else
  bi_gemm_kernel<F_TBE21, 0, MT, WG><<<dim3((d.N + BN - 1) / BN, S, 1), 512, smem, st>>>(
#endif
      x, ldx, M, d, S, y, ldy, ws);
  e = cudaGetLastError();
  if (e != cudaSuccess || S == 1) return e;
  bi_reduce_kernel<<<dim3((d.N + 255) / 256, M), 256, 0, st>>>(ws, S, M, d.N, d.bias, y, ldy);
  return cudaGetLastError();
}

#if BI_T21_WIDE128
// M128 only: four groups each own two m16 tiles. All 32 warps decode one row
// pair through the original scalar decoder (NR=1). m0's general WG stride is
// 256, while this specialization covers128 rows, so grid.z MUST remain one.
inline cudaError_t launch_tbe21_wide128(const __nv_bfloat16* x, int64_t ldx, int M,
                                     const BiDesc& bd, int S, __nv_bfloat16* y,
                                     int64_t ldy, float* ws, cudaStream_t st) {
  if (M != 128) return cudaErrorInvalidValue;
  constexpr int MT = 2, WG = 4;
  constexpr size_t TAB = (size_t)Drv<Ld<F_TBE21, 0>>::kTab;
  const WDesc d = to_wdesc(bd);
  const size_t smem = BI_T21_PIPE128 ? (size_t)T21_PIPE_SMEM : (size_t)(BN + MT * 16 * WG) * LDS * 2 + TAB;
  cudaError_t e = optin_smem<F_TBE21, 0, MT, WG>(smem);
  if (e != cudaSuccess) return e;
  bi_gemm_kernel<F_TBE21, 0, MT, WG><<<dim3((d.N + BN - 1) / BN, S, 1), 1024, smem, st>>>(
      x, ldx, M, d, S, y, ldy, ws);
  e = cudaGetLastError();
  if (e != cudaSuccess || S == 1) return e;
  bi_reduce_kernel<<<dim3((d.N + 255) / 256, M), 256, 0, st>>>(ws, S, M, d.N, d.bias, y, ldy);
  return cudaGetLastError();
}
#endif

#if BI_T21_EXEC_LAYOUT
// M49..64 uses MT2/WG2; M97..128 uses MT8/WG1. Both have one row CTA.
// In particular MT8 MUST NOT use generic ceil(M/64), which duplicates outputs.
template <int MT, int WG>
inline cudaError_t launch_tbe21_exec_layout(const __nv_bfloat16* x, int64_t ldx, int M,
                                          const BiDesc& bd, int S, __nv_bfloat16* y,
                                          int64_t ldy, float* ws, cudaStream_t st) {
  static_assert((MT == 2 && WG == 2) || (MT == 8 && WG == 1), "unsupported TBE21 execution layout");
  if constexpr (MT == 2) {
    if (M < 49 || M > 64) return cudaErrorInvalidValue;
  } else {
    if (M < 97 || M > 128) return cudaErrorInvalidValue;
  }
  constexpr size_t TAB = MT == 8 && WG == 1 ? 0u : (size_t)Drv<Ld<F_TBE21, 0>>::kTab;
  const WDesc d = to_wdesc(bd);
  const size_t smem = (size_t)(BN + MT * 16 * WG) * CH * 2 + TAB;
  cudaError_t e = optin_smem<F_TBE21, 0, MT, WG>(smem);
  if (e != cudaSuccess) return e;
  bi_gemm_kernel<F_TBE21, 0, MT, WG><<<dim3((d.N + BN - 1) / BN, S, 1), 256 * WG, smem, st>>>(
      x, ldx, M, d, S, y, ldy, ws);
  e = cudaGetLastError();
  if (e != cudaSuccess || S == 1) return e;
  bi_reduce_kernel<<<dim3((d.N + 255) / 256, M), 256, 0, st>>>(ws, S, M, d.N, d.bias, y, ldy);
  return cudaGetLastError();
}
#endif

#if BI_T21_EXEC_LAYOUT
template <int MT, int WG>
inline cudaError_t launch_exec_decode(const BiDesc& bd, int S, __nv_bfloat16* out, cudaStream_t st) {
  static_assert((MT == 2 && WG == 2) || (MT == 8 && WG == 1), "unsupported EXEC oracle geometry");
  if (bd.N <= 0 || bd.K <= 0 || bd.K % 256 || S < 1 || S > bd.K / CH || out == nullptr)
    return cudaErrorInvalidValue;
  constexpr bool GLOBAL_CB = MT == 8 && WG == 1;
  using D = typename T21DrvSelection<true, Drv<Ld<F_TBE21, 0>>, 0, GLOBAL_CB>::type;
  const size_t smem = (size_t)(BN + MT * 16 * WG) * CH * 2 + D::kTab;
  if (smem >= 48 * 1024) {
    cudaError_t e = cudaFuncSetAttribute(t21_exec_decode_kernel<MT, WG>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) return e;
  }
  const WDesc d = to_wdesc(bd);
  t21_exec_decode_kernel<MT, WG><<<dim3((d.N + BN - 1) / BN, S, 1), 256 * WG, smem, st>>>(d, S, out);
  return cudaGetLastError();
}
#endif

template <int FMT, int KQT>
inline cudaError_t launch_fmt(const __nv_bfloat16* x, int64_t ldx, int M, const BiDesc& bd, int S,
                              __nv_bfloat16* y, int64_t ldy, float* ws, cudaStream_t st, int tile) {
  const WDesc d = to_wdesc(bd);
  if (tile != MTILE && tile != MTILE * MAXWG) return cudaErrorInvalidValue;
#if BI_T21_EXEC_LAYOUT
  if constexpr (FMT == F_TBE21 && KQT == 0) {
    if (tile == 64 && M >= 49 && M <= 64)
      return launch_tbe21_exec_layout<2, 2>(x, ldx, M, bd, S, y, ldy, ws, st);
    if (tile == 128 && M >= 97 && M <= 128)
      return launch_tbe21_exec_layout<8, 1>(x, ldx, M, bd, S, y, ldy, ws, st);
  }
#endif
  const int wg = tile / MTILE;
  // The 128-row tile is only ever asked for above 64 rows (glc_serve.bigemm.tile_for), where
  // it strictly halves weight traffic; below that a 64-row tile reads the same weights and
  // does less MMA, so it is strictly cheaper. The TBE21 TILE128 opt-in uses MT=3 for
  // 65..96 rows: both groups cover48 rows, eliminating32 padded rows of MMA/shared X.
  // This range has a single row CTA, so the existing128-row grid stride remains valid.
  // + the TBE2 codebook table (128 B); 0 for every other format, so their launches are rc7's
  constexpr size_t TAB = (size_t)Drv<Ld<FMT, KQT>>::kTab;
  if (wg == 2) {
#if BI_T21_TILE128
    if constexpr (FMT == F_TBE21) {
      if (M > 64 && M <= 96) {
        const size_t smem = (size_t)(BN + 3 * 16 * 2) * LDS * 2 + TAB;
        cudaError_t oe = optin_smem<FMT, KQT, 3, 2>(smem);
        if (oe != cudaSuccess) return oe;
        bi_gemm_kernel<FMT, KQT, 3, 2><<<dim3((d.N + BN - 1) / BN, S, 1), 512, smem, st>>>(
            x, ldx, M, d, S, y, ldy, ws);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess || S == 1) return e;
        bi_reduce_kernel<<<dim3((d.N + 255) / 256, M), 256, 0, st>>>(ws, S, M, d.N, d.bias, y, ldy);
        return cudaGetLastError();
      }
    }
#endif
    const int mt = 4;
    dim3 grid((d.N + BN - 1) / BN, S, (M + 2 * MTILE - 1) / (2 * MTILE));
    const size_t smem = (size_t)(BN + mt * 16 * 2) * LDS * 2 + TAB;
    cudaError_t oe = optin_smem<FMT, KQT, 4, 2>(smem);
    if (oe != cudaSuccess) return oe;
    bi_gemm_kernel<FMT, KQT, 4, 2><<<grid, 512, smem, st>>>(x, ldx, M, d, S, y, ldy, ws);
  } else {
    const int mt = M >= MTILE ? 4 : (M + 15) / 16;
    dim3 grid((d.N + BN - 1) / BN, S, (M + MTILE - 1) / MTILE);
    const size_t smem = (size_t)(BN + mt * 16) * LDS * 2 + TAB;
    switch (mt) {
      case 1: bi_gemm_kernel<FMT, KQT, 1, 1><<<grid, 256, smem, st>>>(x, ldx, M, d, S, y, ldy, ws); break;
      case 2: bi_gemm_kernel<FMT, KQT, 2, 1><<<grid, 256, smem, st>>>(x, ldx, M, d, S, y, ldy, ws); break;
      case 3: bi_gemm_kernel<FMT, KQT, 3, 1><<<grid, 256, smem, st>>>(x, ldx, M, d, S, y, ldy, ws); break;
      default: bi_gemm_kernel<FMT, KQT, 4, 1><<<grid, 256, smem, st>>>(x, ldx, M, d, S, y, ldy, ws); break;
    }
  }
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess || S == 1) return e;
  bi_reduce_kernel<<<dim3((d.N + 255) / 256, M), 256, 0, st>>>(ws, S, M, d.N, d.bias, y, ldy);
  return cudaGetLastError();
}

template <int FMT, int KQT>
inline cudaError_t launch_decode(const BiDesc& bd, int S, __nv_bfloat16* out, cudaStream_t st, int wg) {
  const WDesc d = to_wdesc(bd);
  const size_t smem = (size_t)Drv<Ld<FMT, KQT>>::kTab;
  dim3 grid((d.N + BN - 1) / BN, S);
#if BI_T21_WIDE128
  if constexpr (FMT == F_TBE21) {
    if (wg == 4) {
      bi_decode_kernel<FMT, KQT, 4><<<grid, 1024, BI_T21_PIPE128 ? (size_t)T21_PIPE_G1_SMEM : smem, st>>>(d, S, out);
      return cudaGetLastError();
    }
  }
#endif
  if (wg == 2) bi_decode_kernel<FMT, KQT, 2><<<grid, 512, smem, st>>>(d, S, out);
  else if (wg == 1) bi_decode_kernel<FMT, KQT, 1><<<grid, 256, smem, st>>>(d, S, out);
  else return cudaErrorInvalidValue;
  return cudaGetLastError();
}

// TBE21 resident checkpoints for launch split count S (no-op when S == 1): out = u32 [N][S-1].
#if BI_T21_TIER_PREFIX
inline cudaError_t launch_tpf_headers(const BiDesc& bd, uint32_t* out, cudaStream_t st) {
  if (bd.N <= 0 || bd.K <= 0 || bd.K % CH != 0 || !out) return cudaErrorInvalidValue;
  const WDesc d = to_wdesc(bd);
  bi_tpf_headers_kernel<0><<<dim3((d.N + BN - 1) / BN), 256, (size_t)T21_TAB_BYTES, st>>>(d, out);
  return cudaGetLastError();
}
#endif

template <int KQT>
inline cudaError_t launch_t21_ckpt(const BiDesc& bd, int S, uint32_t* out, cudaStream_t st) {
  if (S <= 1) return cudaSuccess;
  const WDesc d = to_wdesc(bd);
  bi_t21_ckpt_kernel<KQT, 1><<<dim3((d.N + BN - 1) / BN), 256, (size_t)T21_TAB_BYTES, st>>>(d, S, out);
  return cudaGetLastError();
}

// Registers, spill, smem and resident CTAs/SM of one GEMM instantiation, as the card reports
// them.  The smem/occupancy arithmetic in docs/serving/BIGEMM_V2_20261004.md is a prediction;
// this is the measurement that replaces it.
template <int FMT, int KQT, int MT, int WG>
inline cudaError_t attrs_of(BiKernelAttrs* a) {
  const void* f = reinterpret_cast<const void*>(&bi_gemm_kernel<FMT, KQT, MT, WG>);
  cudaFuncAttributes fa;
  cudaError_t e = cudaFuncGetAttributes(&fa, f);
  if (e != cudaSuccess) return e;
#if BI_T21_EXEC_LAYOUT
  constexpr int STRIDE = FMT == F_TBE21 && KQT == 0 &&
      ((MT == 2 && WG == 2) || (MT == 8 && WG == 1)) ? CH : LDS;
  constexpr size_t TAB = FMT == F_TBE21 && KQT == 0 && MT == 8 && WG == 1 ?
      0u : (size_t)Drv<Ld<FMT, KQT>>::kTab;
  const size_t smem = BI_T21_PIPE128 && FMT == F_TBE21 && MT == 2 && WG == 4 ?
      (size_t)T21_PIPE_SMEM : (size_t)(BN + MT * 16 * WG) * STRIDE * 2 + TAB;
#else
  const size_t smem = BI_T21_PIPE128 && FMT == F_TBE21 && MT == 2 && WG == 4 ?
      (size_t)T21_PIPE_SMEM : (size_t)(BN + MT * 16 * WG) * LDS * 2 + (size_t)Drv<Ld<FMT, KQT>>::kTab;
#endif
  e = optin_smem<FMT, KQT, MT, WG>(smem);
  if (e != cudaSuccess) return e;
  int nb = 0;
  e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, f, 256 * WG, smem);
  a->regs = fa.numRegs; a->local_bytes = (int)fa.localSizeBytes; a->max_blocks_per_sm = nb;
  a->smem_bytes = (int)smem; a->threads = 256 * WG;
  return e;
}

template <int FMT, int KQT>
inline cudaError_t attrs_fmt(int mt, int wg, BiKernelAttrs* a) {
#if BI_T21_EXEC_LAYOUT
  if constexpr (FMT == F_TBE21 && KQT == 0) {
    if (wg == 2 && mt == 2) return attrs_of<FMT, KQT, 2, 2>(a);
    if (wg == 1 && mt == 8) return attrs_of<FMT, KQT, 8, 1>(a);
  }
#endif
  if (wg == 2) {
#if BI_T21_ROW_GEOMETRY
    if constexpr (FMT == F_TBE21) {
      if (mt == 2) return attrs_of<FMT, KQT, 2, 2>(a);
      if (mt == 8) return attrs_of<FMT, KQT, 8, 2>(a);
    }
#endif
#if BI_T21_TILE128
    if constexpr (FMT == F_TBE21)
      if (mt == 3) return attrs_of<FMT, KQT, 3, 2>(a);
#endif
    return mt == 4 ? attrs_of<FMT, KQT, 4, 2>(a) : cudaErrorInvalidValue;
  }
  switch (mt) {
    case 1: return attrs_of<FMT, KQT, 1, 1>(a);
    case 2: return attrs_of<FMT, KQT, 2, 1>(a);
    case 3: return attrs_of<FMT, KQT, 3, 1>(a);
    case 4: return attrs_of<FMT, KQT, 4, 1>(a);
    default: return cudaErrorInvalidValue;
  }
}

// =================================================================================================
// KERNEL V2: bi_gemm_kernel_v2 / bi_decode_kernel_v2 (docs/serving/KERNEL_V2_SPEC_20261017.md §4).
// F_BF16 / F_TBE, WG == 1 only.  No new WDesc field, no edit to any line above this banner's
// opening twin (the LdTbeV2/D2Sel block right after BI_RARE) -- v1 is physically untouched.
//
// MT <= 2: __launch_bounds__(256, 3) (85-reg budget); the BI_V2_MINBLOCKS compile knob flips
// that to (256, 2) if ptxas reports a spill on the card (spec 4.4).  MT in {3,4}: (256, 2), as v1.
#ifndef BI_V2_MINBLOCKS
#define BI_V2_MINBLOCKS 3
#endif
#define BI_V2_BOUNDS(mt) __launch_bounds__(256, (mt) <= 2 ? BI_V2_MINBLOCKS : 2)

// grid (ceil(N/64), S, ceil(M/64)); 256 threads; dynamic smem 2*(BN + MT*16)*LDS*2 bytes
// (double-buffered Ws/Xs, no codebook table -- F_BF16/F_TBE carry none).  S == 1: y written
// directly.  S > 1: fp32 partials to ws[s][m][n], reduced by bi_reduce_kernel (shared with v1)
// in split order.
template <int FMT, int KQT, int MT, int WG, int SPEC>
__global__ void BI_V2_BOUNDS(MT)
bi_gemm_kernel_v2(const __nv_bfloat16* __restrict__ x, int64_t ldx, int M, WDesc d, int S,
                  __nv_bfloat16* __restrict__ y, int64_t ldy, float* __restrict__ ws) {
  extern __shared__ __align__(16) unsigned char smraw[];
  using D2 = D2Sel<FMT, KQT, SPEC>;
  __nv_bfloat16* Ws_[2] = {reinterpret_cast<__nv_bfloat16*>(smraw),
                          reinterpret_cast<__nv_bfloat16*>(smraw) + BN * LDS};
  __nv_bfloat16* Xs_[2] = {Ws_[1] + BN * LDS, Ws_[1] + BN * LDS + MT * 16 * WG * LDS};
  // TBE21 only: codebook table after the double-buffered Xs (T21_TAB_BYTES extra dynamic
  // smem, sized by launch_fmt_v2); dead pointer for the other formats.
  uint32_t* tab = reinterpret_cast<uint32_t*>(Xs_[1] + MT * 16 * WG * LDS);
  constexpr int THR = 256 * WG;
  constexpr int NR = 4 / WG;
  const int n0 = blockIdx.x * BN, s = blockIdx.y, m0 = blockIdx.z * (MTILE * WG);
  const int nch = d.K / CH;
  const int c0 = (int)(((int64_t)s * nch) / S), c1 = (int)(((int64_t)(s + 1) * nch) / S);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int wn = warp & 7, wg = warp >> 3, half = lane >> 4, j = lane & 15;
  int rowl[NR], rowg[NR];
#pragma unroll
  for (int i = 0; i < NR; ++i) {
    rowl[i] = warp * (2 * NR) + 2 * i + half;
    rowg[i] = min(n0 + rowl[i], d.N - 1);
  }
  float acc[MT][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[mt][e] = 0.0f;

  // hoists (spec 3.2): per-row exp_table (TBE only) + one LaneC, instead of recomputed per dec().
  uint32_t e01[NR] = {}, e23[NR] = {};
  if constexpr (FMT == F_TBE || FMT == F_TBE1T) {
#pragma unroll
    for (int i = 0; i < NR; ++i) {
      const uint32_t rp = (uint32_t)__ldg(d.rowparam + rowg[i]);
      miv_tbe::exp_table(rp, e01[i], e23[i]);
    }
  }
  const miv_tbe::LaneC lc = miv_tbe::lane_const(j);

  typename D2::St st[NR];
  typename D2::R raw[NR];
  if constexpr (FMT == F_TBE21) {
    D2Sel<F_TBE21, KQT, SPEC>::prologue(d, tab);
    __syncthreads();
  }
  if (c0 < c1) {
#pragma unroll
    for (int i = 0; i < NR; ++i) {
      if constexpr (FMT == F_TBE21)
        st[i] = D2Sel<F_TBE21, KQT, SPEC>::start21(d, rowg[i], c0 * CH + 8 * j, s, S, lane);
      else
        st[i] = D2::start(d, rowg[i], c0 * CH + 8 * j, lane);
      raw[i] = D2::load(st[i], lane);
    }
    // prologue: issue cp.async X(c0) into Xs(c0&1), commit (spec 4.2).
#pragma unroll
    for (int it = 0; it < MT; ++it) {
      const int i = threadIdx.x + THR * it, r = i >> 4, cv = i & 15, m = m0 + r;
      cp_async_cg_16(Xs_[c0 & 1] + r * LDS + cv * 8, x + (int64_t)m * ldx + c0 * CH + cv * 8,
                     m < M);
    }
    cp_async_commit();
  }
  for (int c = c0; c < c1; ++c) {
    const int b = c & 1;
    __nv_bfloat16* Ws = Ws_[b];
    __nv_bfloat16* Xs = Xs_[b];
    // weights: dec_a for all NR rows, then dec_b + st.shared (phase split, spec 1).
    typename D2::E ea[NR];
#pragma unroll
    for (int i = 0; i < NR; ++i) ea[i] = D2::dec_a(d, raw[i], lane);
#pragma unroll
    for (int i = 0; i < NR; ++i) {
      uint4 w;
      if constexpr (FMT == F_TBE21)
        w = D2Sel<F_TBE21, KQT, SPEC>::dec_b21(d, raw[i], ea[i], lane, j, st[i], tab);
      else
        w = D2::dec_b(d, raw[i], ea[i], lane, e01[i], e23[i], lc);
      *reinterpret_cast<uint4*>(Ws + rowl[i] * LDS + 8 * j) = w;
    }
    if (c + 1 < c1) {                       // W-prefetch chunk c+1 (regs, as v1)
#pragma unroll
      for (int i = 0; i < NR; ++i) raw[i] = D2::load(st[i], lane);
    }
    cp_async_wait_all();                    // X(c) landed (this thread's group)
    __syncthreads();                        // ONE sync: Ws(b) visible, X(c) visible to all
    if (c + 1 < c1) {                       // X-prefetch chunk c+1 into Xs(b^1)
#pragma unroll
      for (int it = 0; it < MT; ++it) {
        const int i = threadIdx.x + THR * it, r = i >> 4, cv = i & 15, m = m0 + r;
        cp_async_cg_16(Xs_[b ^ 1] + r * LDS + cv * 8,
                       x + (int64_t)m * ldx + (c + 1) * CH + cv * 8, m < M);
      }
      cp_async_commit();
    }
    // MMA over Ws(b), Xs(b): v1 verbatim with buffered pointers.
#pragma unroll
    for (int kk = 0; kk < CH; kk += 16) {
      uint32_t bmat[2];
      ldsm_x2(bmat, Ws + (wn * 8 + (lane & 7)) * LDS + kk + ((lane >> 3) & 1) * 8);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        uint32_t a[4];
        ldsm_x4(a, Xs + (wg * MT * 16 + mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LDS + kk +
                       (lane >> 4) * 8);
        mma16816(acc[mt], a, bmat);
      }
    }
    // NO second __syncthreads: the hazard argument (spec 4.2) -- cp.async(c+1) writes
    // Xs(b^1) = Xs((c-1)&1), which every warp's mma(c-1) has finished reading (mma(c-1) sits
    // between sync(c-1) and sync(c) in program order for every warp); Ws(b) is next written at
    // iteration c+2 (same parity), reached only after sync(c+1), itself after every warp's
    // mma(c) reads; X(c) is waited on by its own issuing thread before sync(c), which then
    // publishes it to every warp before mma(c) reads it.
  }
  // epilogue: copy v1 (bi_gemm_kernel) verbatim -- acc indices, bias add, rounding, S>1 spill.
  const int g = lane >> 2, t4 = lane & 3;
  const int n = n0 + wn * 8 + 2 * t4;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int m = m0 + wg * MT * 16 + mt * 16 + g + 8 * h;
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

// Decode-only twin of bi_gemm_kernel_v2's weight path (spec 4.3): same D2Sel load/dec_a/dec_b
// order, same CTA row mapping, same split traversal c0..c1 as bi_decode_kernel -- only the
// destination differs: out[row][c*CH + 8j] instead of Ws.  No Xs, no cp.async, no sync.  This is
// the SIMT-provable object (spec 8): its output must equal v1's bi_decode_kernel bitwise.
template <int FMT, int KQT, int WG, int SPEC>
__global__ void __launch_bounds__(256 * WG, WG == 1 ? 2 : 1)
bi_decode_kernel_v2(WDesc d, int S, __nv_bfloat16* __restrict__ out) {
  using D2 = D2Sel<FMT, KQT, SPEC>;
  // TBE21 only: smem codebook table (T21_TAB_BYTES dynamic smem; the other formats launch
  // with 0 dynamic smem and this pointer is dead -- their codegen is untouched).
  extern __shared__ __align__(16) unsigned char smraw[];
  uint32_t* tab = reinterpret_cast<uint32_t*>(smraw);
  constexpr int NR = 4 / WG;
  const int n0 = blockIdx.x * BN, s = blockIdx.y;
  const int nch = d.K / CH;
  const int c0 = (int)(((int64_t)s * nch) / S), c1 = (int)(((int64_t)(s + 1) * nch) / S);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int half = lane >> 4, j = lane & 15;
  if constexpr (FMT == F_TBE21) {
    D2Sel<F_TBE21, KQT, SPEC>::prologue(d, tab);
    __syncthreads();
  }
  int rowl[NR], rowg[NR];
#pragma unroll
  for (int i = 0; i < NR; ++i) {
    rowl[i] = warp * (2 * NR) + 2 * i + half;
    rowg[i] = min(n0 + rowl[i], d.N - 1);
  }
  uint32_t e01[NR] = {}, e23[NR] = {};
  if constexpr (FMT == F_TBE || FMT == F_TBE1T) {
#pragma unroll
    for (int i = 0; i < NR; ++i) {
      const uint32_t rp = (uint32_t)__ldg(d.rowparam + rowg[i]);
      miv_tbe::exp_table(rp, e01[i], e23[i]);
    }
  }
  const miv_tbe::LaneC lc = miv_tbe::lane_const(j);
  typename D2::St st[NR];
  typename D2::R raw[NR];
  // opledger addendum C2/F4(c): induction output pointer per row (the twin's own artifact --
  // the real GEMM kernel stores to smem, unaffected).  `n0 + rowl[i] < d.N` and the row base
  // address are BOTH loop-invariant (rowl[i]/n0/d.N never change across chunks), hoisted once
  // instead of recomputing `(n0+rowl[i])*d.K` (a 64-bit multiply) every iteration.
  __nv_bfloat16* outp[NR];
  bool valid[NR];
#pragma unroll
  for (int i = 0; i < NR; ++i) {
    valid[i] = n0 + rowl[i] < d.N;
    outp[i] = out + (int64_t)(n0 + rowl[i]) * d.K + c0 * CH + 8 * j;
  }
  if (c0 < c1) {
#pragma unroll
    for (int i = 0; i < NR; ++i) {
      if constexpr (FMT == F_TBE21)
        st[i] = D2Sel<F_TBE21, KQT, SPEC>::start21(d, rowg[i], c0 * CH + 8 * j, s, S, lane);
      else
        st[i] = D2::start(d, rowg[i], c0 * CH + 8 * j, lane);
      raw[i] = D2::load(st[i], lane);
    }
  }
  for (int c = c0; c < c1; ++c) {
    typename D2::E ea[NR];
#pragma unroll
    for (int i = 0; i < NR; ++i) ea[i] = D2::dec_a(d, raw[i], lane);
#pragma unroll
    for (int i = 0; i < NR; ++i) {
      uint4 w;
      if constexpr (FMT == F_TBE21)
        w = D2Sel<F_TBE21, KQT, SPEC>::dec_b21(d, raw[i], ea[i], lane, j, st[i], tab);
      else
        w = D2::dec_b(d, raw[i], ea[i], lane, e01[i], e23[i], lc);
      if (valid[i]) *reinterpret_cast<uint4*>(outp[i]) = w;
      outp[i] += CH;
    }
    if (c + 1 < c1) {
#pragma unroll
      for (int i = 0; i < NR; ++i) raw[i] = D2::load(st[i], lane);
    }
  }
}

template <int FMT, int KQT, int MT, int SPEC>
inline cudaError_t optin_smem_v2(size_t smem) {
  static bool done = false;
  if (done || smem <= 48 * 1024) return cudaSuccess;
  cudaError_t e = cudaFuncSetAttribute(
      reinterpret_cast<const void*>(&bi_gemm_kernel_v2<FMT, KQT, MT, 1, SPEC>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  if (e == cudaSuccess) done = true;
  return e;
}

// v2 is a 64-row-tile kernel only (WG == 1); variant_for (bigemm_tune.py) never asks for tile
// 128 under variant 2, and launch_fmt_v2 refuses it explicitly as a second line of defense.
template <int FMT, int KQT, int SPEC>
inline cudaError_t launch_fmt_v2(const __nv_bfloat16* x, int64_t ldx, int M, const BiDesc& bd,
                                 int S, __nv_bfloat16* y, int64_t ldy, float* ws,
                                 cudaStream_t st, int tile) {
  if (tile != MTILE) return cudaErrorInvalidValue;
  const WDesc d = to_wdesc(bd);
  const int mt = M >= MTILE ? 4 : (M + 15) / 16;
  dim3 grid((d.N + BN - 1) / BN, S, (M + MTILE - 1) / MTILE);
  const size_t smem = 2 * (size_t)(BN + mt * 16) * LDS * 2 +
                      (FMT == F_TBE21 ? (size_t)T21_TAB_BYTES : 0);
  cudaError_t oe;
  switch (mt) {
    case 1:
      oe = optin_smem_v2<FMT, KQT, 1, SPEC>(smem);
      if (oe != cudaSuccess) return oe;
      bi_gemm_kernel_v2<FMT, KQT, 1, 1, SPEC><<<grid, 256, smem, st>>>(x, ldx, M, d, S, y, ldy, ws);
      break;
    case 2:
      oe = optin_smem_v2<FMT, KQT, 2, SPEC>(smem);
      if (oe != cudaSuccess) return oe;
      bi_gemm_kernel_v2<FMT, KQT, 2, 1, SPEC><<<grid, 256, smem, st>>>(x, ldx, M, d, S, y, ldy, ws);
      break;
    case 3:
      oe = optin_smem_v2<FMT, KQT, 3, SPEC>(smem);
      if (oe != cudaSuccess) return oe;
      bi_gemm_kernel_v2<FMT, KQT, 3, 1, SPEC><<<grid, 256, smem, st>>>(x, ldx, M, d, S, y, ldy, ws);
      break;
    default:
      oe = optin_smem_v2<FMT, KQT, 4, SPEC>(smem);
      if (oe != cudaSuccess) return oe;
      bi_gemm_kernel_v2<FMT, KQT, 4, 1, SPEC><<<grid, 256, smem, st>>>(x, ldx, M, d, S, y, ldy, ws);
      break;
  }
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess || S == 1) return e;
  bi_reduce_kernel<<<dim3((d.N + 255) / 256, M), 256, 0, st>>>(ws, S, M, d.N, d.bias, y, ldy);
  return cudaGetLastError();
}

template <int FMT, int KQT, int SPEC>
inline cudaError_t launch_decode_v2(const BiDesc& bd, int S, __nv_bfloat16* out, cudaStream_t st) {
  const WDesc d = to_wdesc(bd);
  dim3 grid((d.N + BN - 1) / BN, S);
  const size_t smem = FMT == F_TBE21 ? (size_t)T21_TAB_BYTES : 0;
  bi_decode_kernel_v2<FMT, KQT, 1, SPEC><<<grid, 256, smem, st>>>(d, S, out);
  return cudaGetLastError();
}

template <int FMT, int KQT, int MT, int SPEC>
inline cudaError_t attrs_of_v2(BiKernelAttrs* a) {
  const void* f = reinterpret_cast<const void*>(&bi_gemm_kernel_v2<FMT, KQT, MT, 1, SPEC>);
  cudaFuncAttributes fa;
  cudaError_t e = cudaFuncGetAttributes(&fa, f);
  if (e != cudaSuccess) return e;
  const size_t smem = 2 * (size_t)(BN + MT * 16) * LDS * 2 +
                      (FMT == F_TBE21 ? (size_t)T21_TAB_BYTES : 0);
  e = optin_smem_v2<FMT, KQT, MT, SPEC>(smem);
  if (e != cudaSuccess) return e;
  int nb = 0;
  e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, f, 256, smem);
  a->regs = fa.numRegs; a->local_bytes = (int)fa.localSizeBytes; a->max_blocks_per_sm = nb;
  a->smem_bytes = (int)smem; a->threads = 256;
  return e;
}

template <int FMT, int KQT, int SPEC>
inline cudaError_t attrs_fmt_v2(int mt, BiKernelAttrs* a) {
  switch (mt) {
    case 1: return attrs_of_v2<FMT, KQT, 1, SPEC>(a);
    case 2: return attrs_of_v2<FMT, KQT, 2, SPEC>(a);
    case 3: return attrs_of_v2<FMT, KQT, 3, SPEC>(a);
    case 4: return attrs_of_v2<FMT, KQT, 4, SPEC>(a);
    default: return cudaErrorInvalidValue;
  }
}
// =================================================================================================

}  // namespace bi
