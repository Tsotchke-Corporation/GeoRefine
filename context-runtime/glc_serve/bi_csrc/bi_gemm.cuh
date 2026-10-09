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
enum Fmt { F_BF16 = BI_BF16, F_Q8 = BI_Q8, F_TBE = BI_TBE, F_TBE2 = BI_TBE2, F_TBE21 = BI_TBE21 };
#else
enum Fmt { F_BF16 = BI_BF16, F_Q8 = BI_Q8, F_TBE = BI_TBE, F_KQ = BI_KQ, F_TBE2 = BI_TBE2, F_TBE21 = BI_TBE21 };
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

template <int KQT>
struct Ld<F_TBE21, KQT> {
  struct R { uint32_t c16; uint2 s8; uint32_t win; };  // level-1 u16, smb, window word
  struct C { uint32_t ow, ob, rp; };                     // cursor word, bit, codebook index
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
      tab[threadIdx.x] = t2_rotr8x4(v);
    }
  }
  // split start: row word offset + (s == 0 ? 0 : resident checkpoint of split s)
  static __device__ __forceinline__ void start(const WDesc& d, int row, int s, int S, C& k) {
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
  static __device__ __forceinline__ uint4 dec(const WDesc& d, const R& r, int, int, int, int j,
                                              C& k, const uint32_t* tab) {
    const uint32_t ob = k.ob;
    const uint32_t C1 = (r.c16 | (r.c16 << 14)) & 0x33333333u;   // nibble i = level-1 code
    const uint32_t T1 = C1 & (C1 >> 1) & 0x11111111u;             // code 3 = escape
    const uint32_t P1 = T1 * 0x11111111u;
    const uint32_t k1 = P1 >> 28, R1 = P1 - T1;
    uint32_t n1;
    const uint32_t rho1 = t2_hscan16(k1, j, n1);
    const uint4 ta = *reinterpret_cast<const uint4*>(tab + 8u * k.rp);
    const uint32_t t4 = tab[8u * k.rp + 4u];
    const uint32_t E1lo = __byte_perm(ta.x, ta.y, C1), E1hi = __byte_perm(ta.x, ta.y, C1 >> 16);
    // L2 (rank-1 order): digit d -> symbol 3 + d (bytes 3..5; 3 -> byte 6 placeholder)
    const uint32_t f2 = t2_get_win(r.win, ob + 2u * rho1, 2u * k1);
    const uint32_t N2 = t2_spread2(f2);
    const uint32_t S2 = N2 + 0x33333333u;
    uint32_t X2lo = __byte_perm(ta.x, ta.y, S2), X2hi = __byte_perm(ta.x, ta.y, S2 >> 16);
    const uint32_t Z2 = N2 & (N2 >> 1) & 0x11111111u;
    const uint32_t P2 = Z2 * 0x11111111u;
    const uint32_t k2 = P2 >> 28, R2 = P2 - Z2;
    uint32_t n2;
    const uint32_t rho2 = t2_hscan16(k2, j, n2);
    // L3 (rank-2 order): digit d -> symbol 6 + d (w2 bytes 0..2; 3 -> byte 3 placeholder)
    const uint32_t f3 = t2_get31(d.t2_ovf, k.ow, r.win, ob + 2u * n1 + 2u * rho2, 2u * k2);
    const uint32_t N3 = t2_spread2(f3);
    uint32_t X3lo = __byte_perm(ta.z, ta.z, N3), X3hi = __byte_perm(ta.z, ta.z, N3 >> 16);
    const uint32_t Z3 = N3 & (N3 >> 1) & 0x11111111u;
    const uint32_t P3 = Z3 * 0x11111111u;
    const uint32_t k3 = P3 >> 28, R3 = P3 - Z3;
    uint32_t n3;
    const uint32_t rho3 = t2_hscan16(k3, j, n3);
    // L4 (rank-3 order): digit d -> symbol 9 + d (w3:w4 bytes 0..6; 7 -> raw, placeholder)
    const uint32_t b4 = ob + 2u * n1 + 2u * n2;
    const uint32_t f4 = t2_get31(d.t2_ovf, k.ow, r.win, b4 + 3u * rho3, 3u * k3);
    const uint32_t N4 = t2_spread3(f4);
    uint32_t X4lo = __byte_perm(ta.w, t4, N4), X4hi = __byte_perm(ta.w, t4, N4 >> 16);
    const uint32_t Z4 = N4 & (N4 >> 1) & (N4 >> 2) & 0x11111111u;
    uint32_t n4 = 0u;
    if (BI_RARE(__any_sync(0xffffffffu, Z4))) {
      const uint32_t P4 = Z4 * 0x11111111u;
      const uint32_t k4 = P4 >> 28, R4 = P4 - Z4;
      const uint32_t rho4 = t2_hscan16(k4, j, n4);
      const uint32_t base = b4 + 3u * n3 + 8u * rho4;
      const uint32_t g0 = t2_rotr8x4(t2_get(d.t2_ovf, k.ow, r.win, base, min(8u * k4, 32u)));
      const uint32_t g1 = t2_rotr8x4(t2_get(d.t2_ovf, k.ow, r.win, base + 32u, 8u * k4 > 32u ? 8u * k4 - 32u : 0u));
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
    w.x = t2_pair(r.s8.x, Elo, 0x5140u, 0x1504u);
    w.y = t2_pair(r.s8.x, Elo, 0x7362u, 0x3726u);
    w.z = t2_pair(r.s8.y, Ehi, 0x5140u, 0x1504u);
    w.w = t2_pair(r.s8.y, Ehi, 0x7362u, 0x3726u);
    return w;
  }
};
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
// grid (ceil(N/64), S, ceil(M/(64*WG))); 256*WG threads; dynamic smem
// (64 + MT*16*WG) * LDS * 2 bytes.  MT = m16 tiles per WARP GROUP (1..4), WG = warp groups
// per CTA (1 or 2 -> a 64- or 128-row tile; see the WG note above for why the bits do not
// move).  S == 1: y written directly.  S > 1: fp32 partials to ws[s][m][n] (row stride N),
// reduced by bi_reduce_kernel in split order.
template <int FMT, int KQT, int MT, int WG>
__global__ void __launch_bounds__(256 * WG, WG == 1 ? 2 : 1)
bi_gemm_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx, int M, WDesc d, int S,
               __nv_bfloat16* __restrict__ y, int64_t ldy, float* __restrict__ ws) {
  extern __shared__ __align__(16) unsigned char smraw[];
  __nv_bfloat16* Ws = reinterpret_cast<__nv_bfloat16*>(smraw);
  __nv_bfloat16* Xs = Ws + BN * LDS;
  using D = Drv<Ld<FMT, KQT>>;
  // TBE2 codebook table: after Xs, only allocated (launch_fmt) when D::kTab > 0.
  uint32_t* tab = reinterpret_cast<uint32_t*>(Xs + MT * 16 * WG * LDS);
  constexpr int THR = 256 * WG;          // threads per CTA
  constexpr int NR = 4 / WG;             // weight-row PAIRS each warp decodes (WG=1: 4, WG=2: 2)
  const int n0 = blockIdx.x * BN, s = blockIdx.y, m0 = blockIdx.z * (MTILE * WG);
  const int nch = d.K / CH;
  const int c0 = (int)(((int64_t)s * nch) / S), c1 = (int)(((int64_t)(s + 1) * nch) / S);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int wn = warp & 7;               // which 8 of the CTA's 64 weight rows this warp MMAs
  const int wg = warp >> 3;              // warp group -> which MT*16 activation rows (0 at WG=1)
  const int half = lane >> 4, j = lane & 15;
  // The CTA's 64 weight rows are spread over its 8*WG warps: 2*NR rows per warp, a half-warp
  // per row, exactly as at WG=1 (where 2*NR == 8 and this is the old expression verbatim).
  // Which lane decodes a weight does not change the decoded value -- Ld<>::dec is a pure
  // function of the loaded bytes -- and Ws[] ends up byte-identical either way.
  int rowl[NR], rowg[NR];
#pragma unroll
  for (int i = 0; i < NR; ++i) {
    rowl[i] = warp * (2 * NR) + 2 * i + half;
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
  if (c0 < c1) {
#pragma unroll
    for (int i = 0; i < NR; ++i) {
      D::start(d, rowg[i], c0, s, S, lane, cur[i]);
      raw[i] = D::load(d, rowg[i], c0 * CH + 8 * j, lane, cur[i]);
    }
  }
  for (int c = c0; c < c1; ++c) {
    // activations: rows m0 .. m0 + MT*16*WG, columns c*CH .. c*CH + 127 (rows >= M -> zero)
#pragma unroll
    for (int it = 0; it < MT; ++it) {
      const int i = threadIdx.x + THR * it;           // MT*16*WG rows * 16 vectors = MT*THR
      const int r = i >> 4, cv = i & 15;
      const int m = m0 + r;
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (m < M) v = __ldg(reinterpret_cast<const uint4*>(x + (int64_t)m * ldx + c * CH + cv * 8));
      *reinterpret_cast<uint4*>(Xs + r * LDS + cv * 8) = v;
    }
    // weights: decode this chunk into smem, then issue the next chunk's loads
#pragma unroll
    for (int i = 0; i < NR; ++i) {
      const uint4 w = D::dec(d, raw[i], rowg[i], c * CH + 8 * j, lane, j, cur[i], tab);
      *reinterpret_cast<uint4*>(Ws + rowl[i] * LDS + 8 * j) = w;
    }
    if (c + 1 < c1) {
#pragma unroll
      for (int i = 0; i < NR; ++i) raw[i] = D::load(d, rowg[i], (c + 1) * CH + 8 * j, lane, cur[i]);
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < CH; kk += 16) {
      uint32_t b[2];
      ldsm_x2(b, Ws + (wn * 8 + (lane & 7)) * LDS + kk + ((lane >> 3) & 1) * 8);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        uint32_t a[4];
        ldsm_x4(a, Xs + (wg * MT * 16 + mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LDS + kk +
                       (lane >> 4) * 8);
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
template <int FMT, int KQT, int WG>
__global__ void __launch_bounds__(256 * WG, WG == 1 ? 2 : 1)
bi_decode_kernel(WDesc d, int S, __nv_bfloat16* __restrict__ out) {
  extern __shared__ __align__(16) unsigned char smraw[];
  using D = Drv<Ld<FMT, KQT>>;
  uint32_t* tab = reinterpret_cast<uint32_t*>(smraw);
  constexpr int NR = 4 / WG;
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
  if (c0 < c1) {
#pragma unroll
    for (int i = 0; i < NR; ++i) {
      D::start(d, rowg[i], c0, s, S, lane, cur[i]);
      raw[i] = D::load(d, rowg[i], c0 * CH + 8 * j, lane, cur[i]);
    }
  }
  for (int c = c0; c < c1; ++c) {
#pragma unroll
    for (int i = 0; i < NR; ++i) {
      const uint4 w = D::dec(d, raw[i], rowg[i], c * CH + 8 * j, lane, j, cur[i], tab);
      if (n0 + rowl[i] < d.N)
        *reinterpret_cast<uint4*>(out + (int64_t)(n0 + rowl[i]) * d.K + c * CH + 8 * j) = w;
    }
    if (c + 1 < c1) {
#pragma unroll
      for (int i = 0; i < NR; ++i) raw[i] = D::load(d, rowg[i], (c + 1) * CH + 8 * j, lane, cur[i]);
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
  return d;
}

// The WG=2 kernel needs 52,224 B of dynamic smem, above the 48 KiB default cap, so it must
// opt in once per (kernel instantiation, device).  Idempotent, so the unsynchronised static
// is safe: a race costs a second identical call, never a wrong launch.
template <int FMT, int KQT, int MT, int WG>
inline cudaError_t optin_smem(size_t smem) {
  static bool done = false;
  if (done || smem <= 48 * 1024) return cudaSuccess;
  cudaError_t e = cudaFuncSetAttribute(reinterpret_cast<const void*>(&bi_gemm_kernel<FMT, KQT, MT, WG>),
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  if (e == cudaSuccess) done = true;
  return e;
}

template <int FMT, int KQT>
inline cudaError_t launch_fmt(const __nv_bfloat16* x, int64_t ldx, int M, const BiDesc& bd, int S,
                              __nv_bfloat16* y, int64_t ldy, float* ws, cudaStream_t st, int tile) {
  const WDesc d = to_wdesc(bd);
  if (tile != MTILE && tile != MTILE * MAXWG) return cudaErrorInvalidValue;
  const int wg = tile / MTILE;
  // The 128-row tile is only ever asked for above 64 rows (glc_serve.bigemm.tile_for), where
  // it strictly halves weight traffic; below that a 64-row tile reads the same weights and
  // does less MMA, so it is strictly cheaper.  A WG=2 CTA always runs MT=4 per group: rows
  // past M load as zero and are never stored, which costs MMA on zeros and no bits.
  // + the TBE2 codebook table (128 B); 0 for every other format, so their launches are rc7's
  constexpr size_t TAB = (size_t)Drv<Ld<FMT, KQT>>::kTab;
  if (wg == 2) {
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
  if (wg == 2) bi_decode_kernel<FMT, KQT, 2><<<grid, 512, smem, st>>>(d, S, out);
  else if (wg == 1) bi_decode_kernel<FMT, KQT, 1><<<grid, 256, smem, st>>>(d, S, out);
  else return cudaErrorInvalidValue;
  return cudaGetLastError();
}

// TBE21 resident checkpoints for launch split count S (no-op when S == 1): out = u32 [N][S-1].
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
  const size_t smem = (size_t)(BN + MT * 16 * WG) * LDS * 2 + (size_t)Drv<Ld<FMT, KQT>>::kTab;
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
  if (wg == 2) return mt == 4 ? attrs_of<FMT, KQT, 4, 2>(a) : cudaErrorInvalidValue;
  switch (mt) {
    case 1: return attrs_of<FMT, KQT, 1, 1>(a);
    case 2: return attrs_of<FMT, KQT, 2, 1>(a);
    case 3: return attrs_of<FMT, KQT, 3, 1>(a);
    case 4: return attrs_of<FMT, KQT, 4, 1>(a);
    default: return cudaErrorInvalidValue;
  }
}

}  // namespace bi
