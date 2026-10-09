// GLC-TBE register-level decode straight into Tensor Core B fragments (sm_80).
//
// Rung 1 of docs/research/GLC_KERNEL_DECISION_20260901.md section 5: y[M,N] =
// x[M,K] @ W[N,K]^T with W held in the tile-bitmap container of
// docs/research/TBE_KERNEL_DESIGN_20260808.md.  M is padded to 16 and every
// product is one mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 chain.
//
// WHAT IS DELIBERATELY ABSENT, AND WHY
//   * no shared-memory staging of weights   -- CUTLASS ex.55 / FireQ / Marlin
//                                              all upcast in the register file
//   * no __syncthreads in the k loop        -- nothing is shared between warps
//   * no per-element FMA                    -- the MMA does the arithmetic
//   * no bf16<->fp32 conversion in decode   -- cvt is quarter rate on sm80
//   * no warp shuffle in decode             -- shfl is half rate on sm80
//   * no atomics, no dense weight buffer    -- one deterministic owner/output
//
// LAYOUT CHOICE: mma16 (rung 1c).
//   Tile t is still elements [64t, 64t+64) of row-major W[N,K] -- flat64's
//   tiles -- but PERMUTED inside the tile so a lane's work is contiguous:
//
//       tile position p = 16*j + 4*s + e   holds k-offset  16*s + 2*j + O[e]
//       with O = {0, 1, 8, 9},  j = p >> 4,  s = (p >> 2) & 3,  e = p & 3.
//
//   Three consequences, and they are the whole reason rung 1b lost:
//     * lane j's 16 smb bytes are tile bytes [16j, 16j+16)  -> ONE uint4 load
//       per lane per tile, where flat64 needed a broadcast LDG.128 and two SEL
//       per k16 step;
//     * lane j's plane bits are bits [16j, 16j+16) of each 64-bit plane word
//       -> ONE 16-bit field per plane per TILE.  The 3-bit codes are spread
//       into PRMT selector nibbles once per tile instead of once per step, and
//       the four steps then only shift;
//     * lane j's escape bytes are CONTIGUOUS in esc[], starting at
//       tile_base + popc(esc_mask & ((1 << 16j) - 1)) -- so the escape path is
//       four unconditional byte loads and one PRMT, with no branch, no
//       per-element dependent address and no cross-lane operation.
//
// WHY THAT LAST POINT DECIDED THE REWRITE
//   The 20260901 SASS histogram
//   (.icc/evidence/glc-tbe-fragment-kernel-20260901/remote/sass_gemm_loop.txt)
//   measured 22.16 thread-instructions per element: 12.09 always-executed and
//   10.06 behind `if (ebits & 0x303u)`.  That guard was WARP-WIDE over the 128
//   elements a quad covers, so at the measured 2.124% per-element escape rate
//   it fired on 93.4% of steps.  The escape block was never rare.  Here it is
//   unconditional and costs about 2 instructions per element.
//
// ELEMENT -> LANE MAP (the contract the CPU test re-derives independently)
//   warp handles rows n_base + s'*8 + gid for s' in [0, NSLAB), 8 rows each.
//   lane L:  gid = L >> 2   (0..7)  -> which of the slab's 8 rows / N columns
//            j   = L & 3    (0..3)  -> which k pair inside the 16-wide block
//   for k16 step s (0..3) of tile kt of row r, with k = 64*kt + 16*s + 2*j:
//       b0 = {W[r][k], W[r][k+1]}, b1 = {W[r][k+8], W[r][k+9]}  -- exactly the
//       m16n8k16 .col B fragment for column gid.  Unchanged from rung 1b: the
//       fragment layout is fixed by the instruction; only WHERE those four
//       bytes and bits live inside the tile changed.
//
// DECODE COST -- hand count from this file; the same table, with the same
// totals, is in the module docstring of _glc_tbe_mma.py.
//
// AMORTISED, per lane per TILE (16 elements), per n8 slab:
//   3 LDG.64 + 1 LDG.128 + 4 LDG.U8      8  planes, the lane's smb uint4, esc
//   3 SEL + 3 SHF + 3 LOP3               9  the lane's three 16-bit fields
//   3 x 13 SHF/LOP3                     39  bit q -> nibble q, both halves
//   2 x (2 SHF + 1 LOP3)                 6  planes or'd into one PRMT selector
//   2 x 2 LOP3                           4  escape nibbles = NOR3 of the spreads
//   2 LOP3 + 3 POPC                      5  escape bitmask and its two counts
//   2 LOP3 + 2 POPC + 2 SEL + 4 addr    13  the lane's base and the tile total
//   3 SHF + 2 LOP3                       5  four escape bytes into one register
//   3 IMAD + 2 SHF + 2 LOP3              7  ranks by multiply-prefix-sum
//   2 SHF + 4 PRMT                       6  escape bytes -> one word per step
//   1 ISETP + branch                     4  the n > 4 guard, not taken
//   1 IADD3 + 2 x 2                      5  tile_base, plane and smb pointers
//                                      ---
//                                      111  = 6.94 per element
//
// ALWAYS-ON, per k16 step (4 elements), four steps per tile:
//   0.5 SHF                            0.5  pick the step's four nibbles
//   1 PRMT + 1 LOP3                      2  exponent table, escapes or'd in
//   2 x (PRMT + SHF)                     4  bytes -> exponent at bits 7 and 23
//   2 x (PRMT + LOP3)                    4  mantissas and both signs, one mask
//   1 HMMA                               1
//   1 LDG.32 (2 at M > 8)                1  A fragments, shared by the 2 slabs
//   loop compare/branch + A pointer    0.75
//                                    -----
//                                    13.25  = 3.31 per element
//
//   TOTAL  ~10.25 thread-instructions per element at M <= 8
//          ~10.50                            at M = 9..16
// against 22.16 measured for rung 1b, the note's 17.9 for FWP1, its <= 10.2
// parity threshold and the 6.4 that would make the kernel bandwidth-bound.
// The escape mechanism is 40 of the 111 amortised instructions -- 2.5 per
// element, always on -- where rung 1b paid 10.06 on 93.4% of steps.
//
// The exponent table E[0..7] lives in two registers and is read with PRMT, so
// base + code - 1 (TBE section 1), the W6Z code-7-is-exact-zero amendment
// (section 6) and the code-0 escape all fall out of the SAME instruction: the
// table holds E[0] = 0 (escape placeholder AND the PRMT zero byte), E[c] =
// base + c - 1, and in W6Z E[7] = 0.  Escapes then only OR their raw exponent
// into an already-zero field.

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kTile = 64;            // elements per TBE tile
constexpr int kWarpSize = 32;
constexpr int kWarpsPerCta = 4;
constexpr int kThreads = kWarpSize * kWarpsPerCta;
constexpr int kMmaM = 16;            // M is padded to the MMA tile
constexpr int kMmaN = 8;
constexpr int kMmaK = 16;
constexpr int kNSlab = 2;            // n8 slabs per warp: amortises the A loads

// ===================== BEGIN DECODE PATH =====================
// Everything between the DECODE PATH markers is the register-level decode the
// static tests police: no shuffle, no conversion, no shared memory, no atomic,
// and -- new in rung 1c -- no branch on the escape bits.

// Bytes of zero padding esc[] must carry past its last real entry.  The four
// escape-byte loads below are UNCONDITIONAL, so the address esc + base + 3
// must be readable for every base <= E.  Four would do; sixteen keeps the tail
// on one sector and leaves room for a wider load later.
constexpr int kEscPad = 16;

// Escapes in one 64-element tile, recovered from bits already in registers.
// esc_mask = ~(p0 | p1 | p2) -- one LOP3 (NOR3) per 32-bit word.
__device__ __forceinline__ uint32_t tbe_esc_word(uint32_t a, uint32_t b,
                                                 uint32_t c) {
    return ~(a | b | c);
}

// Everything a decode needs that depends only on the lane's index in its quad.
// Hoisted out of the k loop by construction, not by hope.
struct TbeLaneConst {
    uint32_t sh;         // 16 * (j & 1): where the lane's 16-bit field starts
    uint32_t half_mask;  // (j & 1) ? 0xFFFF : 0: the escape bits below it
    uint32_t smb_off;    // 16 * j: the lane's own 16 smb bytes
    uint32_t hi;         // j >> 1: the lane's field lives in plane word 1
};

__device__ __forceinline__ TbeLaneConst tbe_lane_const(uint32_t j) {
    TbeLaneConst c;
    c.sh = (j & 1u) << 4;
    c.half_mask = (j & 1u) ? 0x0000FFFFu : 0u;
    c.smb_off = j << 4;
    c.hi = j >> 1;
    return c;
}

// Bit q of a lane's 16-bit plane field -> nibble q of a 32-bit word: eight bits
// per register, three stages, no table and no cross-lane op.  This is the only
// place the 3-bit code is unpacked, and it runs ONCE PER TILE -- the four k16
// steps are then a shift of the result.  `_lo` takes bits 0..7 (steps 0 and 1),
// `_hi` bits 8..15 (steps 2 and 3); `_lo` needs f masked to 16 bits, `_hi` does
// not (its first mask already discards everything above bit 15).
__device__ __forceinline__ uint32_t tbe_spread_lo(uint32_t f) {
    const uint32_t y = (f | (f << 12)) & 0x000F000Fu;
    const uint32_t z = (y | (y << 6)) & 0x03030303u;
    return (z | (z << 3)) & 0x11111111u;
}

__device__ __forceinline__ uint32_t tbe_spread_hi(uint32_t f) {
    const uint32_t y = ((f >> 8) | (f << 4)) & 0x000F000Fu;
    const uint32_t z = (y | (y << 6)) & 0x03030303u;
    return (z | (z << 3)) & 0x11111111u;
}

// One lane's whole tile, decoded down to the four things a k16 step consumes.
struct TbeLaneTile {
    uint32_t sel[2];    // code nibbles: sel[0] steps 0/1, sel[1] steps 2/3
    uint32_t escw[4];   // per step: raw escape exponent bytes, zero elsewhere
    uint32_t sm[4];     // the lane's 16 smb bytes; word s serves step s
    uint32_t tile_esc;  // escapes in the WHOLE tile: advances the running base
};

// The per-tile half of the decode.  Everything here is amortised over the 16
// elements the lane owns in this tile.
__device__ __forceinline__ void tbe_lane_tile(
    const uint2* __restrict__ planes_t, const uint8_t* __restrict__ smb_tile,
    const uint8_t* __restrict__ esc, uint32_t tile_base, TbeLaneConst lc,
    TbeLaneTile& o) {
    const uint2 q0 = planes_t[0];
    const uint2 q1 = planes_t[1];
    const uint2 q2 = planes_t[2];

    // -- the lane's own 16-bit field of each plane: SEL, SHF, AND.
    const uint32_t f0 = ((lc.hi ? q0.y : q0.x) >> lc.sh) & 0x0000FFFFu;
    const uint32_t f1 = ((lc.hi ? q1.y : q1.x) >> lc.sh) & 0x0000FFFFu;
    const uint32_t f2 = ((lc.hi ? q2.y : q2.x) >> lc.sh) & 0x0000FFFFu;

    // -- bit q -> nibble q, once per tile; the three planes then or together
    //    into ONE selector whose nibble q is element q's 3-bit code.
    const uint32_t a0 = tbe_spread_lo(f0);
    const uint32_t a1 = tbe_spread_lo(f1);
    const uint32_t a2 = tbe_spread_lo(f2);
    const uint32_t c0 = tbe_spread_hi(f0);
    const uint32_t c1 = tbe_spread_hi(f1);
    const uint32_t c2 = tbe_spread_hi(f2);
    o.sel[0] = a0 | (a1 << 1) | (a2 << 2);
    o.sel[1] = c0 | (c1 << 1) | (c2 << 2);

    // -- code == 0 iff all three plane bits are 0 (TBE section 3).  Taken in
    //    the nibble domain for the ranks and in the bit domain for the counts.
    const uint32_t ea = (~(a0 | a1 | a2)) & 0x11111111u;
    const uint32_t eb = (~(c0 | c1 | c2)) & 0x11111111u;
    const uint32_t emask = (~(f0 | f1 | f2)) & 0x0000FFFFu;
    const uint32_t n_esc = __popc(emask);
    const uint32_t cnt_lo = __popc(emask & 0xFFu);

    // -- the whole tile's escape words: the lane's base is the escapes of the
    //    lanes BELOW it, and the running base advances by the tile total.
    const uint32_t ew0 = tbe_esc_word(q0.x, q1.x, q2.x);
    const uint32_t ew1 = tbe_esc_word(q0.y, q1.y, q2.y);
    const uint32_t pc0 = __popc(ew0);
    o.tile_esc = pc0 + __popc(ew1);
    const uint32_t below = (lc.hi ? pc0 : 0u) +
                           __popc((lc.hi ? ew1 : ew0) & lc.half_mask);
    const uint32_t at = tile_base + below;

    // -- TWO UNCONDITIONAL DWORD LOADS (rung 1d).  esc[] is 4-byte aligned at
    //    its base and carries kEscPad = 16 zero bytes past its last entry, so
    //    the two aligned dwords that straddle the lane's four bytes are always
    //    readable (highest byte touched is at + 7 < E + kEscPad) and the funnel
    //    shift extracts exactly the same four bytes in the same order as the
    //    four byte loads did.  Still unconditional, still no branch and no
    //    per-element dependent address -- but two scattered loads in flight
    //    instead of four, which is what the long-scoreboard stall is made of.
    const uint32_t* e4 = reinterpret_cast<const uint32_t*>(esc) + (at >> 2);
    const uint32_t e_bytes =
        __funnelshift_r(__ldg(e4), __ldg(e4 + 1), (at & 3u) * 8u);

    // -- rank of every position, ONE multiply per half: with nibbles that are
    //    0 or 1, x * 0x11111110 leaves the EXCLUSIVE prefix sum in nibble i
    //    (each partial sum is <= 7 in the low half and <= 15 in the high half,
    //    so nothing carries).  Non-escaped nibbles are or'd with 4, which
    //    points PRMT at the zero operand -- the placeholder byte, not a branch.
    const uint32_t ra = ea * 0x11111110u;
    const uint32_t rb = eb * 0x11111110u + cnt_lo * 0x11111111u;
    const uint32_t sa = ra | ((ea << 2) ^ 0x44444444u);
    const uint32_t sb = rb | ((eb << 2) ^ 0x44444444u);
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const uint32_t sc = (s < 2) ? sa : sb;
        o.escw[s] = __byte_perm(e_bytes, 0u, (s & 1) ? (sc >> 16) : sc);
    }

    // -- more than four escapes among one lane's 16 elements.  At the measured
    //    2.124% rate that is P = 1.4e-5 per lane per tile; it is EXACT, it is
    //    tested adversarially (all-64-escape tiles included), and it is the
    //    only branch left in the decode.
    if (n_esc > 4u) {
        uint32_t r = at;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            uint32_t w = 0u;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const uint32_t bit = (emask >> (4 * s + e)) & 1u;
                const uint32_t byte =
                    bit ? static_cast<uint32_t>(__ldg(esc + r)) : 0u;
                w |= byte << (8 * e);
                r += bit;
            }
            o.escw[s] = w;
        }
    }

    // -- the lane's 16 smb bytes: ONE uint4, 16-byte aligned by construction
    //    (a tile is 64 bytes and the offset is 16j).  Word s is step s's four
    //    bytes, in element order, which is why the two PRMT selectors below are
    //    lane-invariant compile-time constants.
    const uint4 v = *reinterpret_cast<const uint4*>(smb_tile + lc.smb_off);
    o.sm[0] = v.x;
    o.sm[1] = v.y;
    o.sm[2] = v.z;
    o.sm[3] = v.w;
}

// One k16 step -> the two 32-bit m16n8k16 B fragment registers.  Element e of
// step s is tile position 16j + 4s + e and carries the row's k-offset
// 16s + 2j + (0,1,8,9)[e], so b0 = {k, k+1} and b1 = {k+8, k+9}.
__device__ __forceinline__ void tbe_step_fragment(const TbeLaneTile& t, int s,
                                                  uint32_t e01, uint32_t e23,
                                                  uint32_t& b0, uint32_t& b1) {
    const uint32_t sc = t.sel[s >> 1];
    const uint32_t sel = (s & 1) ? (sc >> 16) : sc;
    // E[0] = 0 is the escape placeholder, so an escaped element's raw exponent
    // is or'd into a field the table already left empty: one LOP3, no select.
    const uint32_t tbl = __byte_perm(e01, e23, sel) | t.escw[s];
    // Re-lay the bytes as {0, E0, 0, E1} and shift right once: the exponent
    // fields land exactly on bits 7..14 and 23..30.  Byte index 4 reads the
    // zero operand, so no mask is needed.
    const uint32_t ef0 = __byte_perm(tbl, 0u, 0x1404u) >> 1;
    const uint32_t ef1 = __byte_perm(tbl, 0u, 0x3424u) >> 1;
    const uint32_t sw = t.sm[s];
    // [S, S, S', S'] & 0x807F807F puts both mantissas and both signs in place
    // with ONE mask: byte 0 supplies bits 0..6, byte 1 supplies bit 15, byte 2
    // bits 16..22 and byte 3 bit 31.  No shift, and no conversion instruction
    // touches the value.
    b0 = (__byte_perm(sw, 0u, 0x1100u) & 0x807F807Fu) | ef0;
    b1 = (__byte_perm(sw, 0u, 0x3322u) & 0x807F807Fu) | ef1;
}

// Escapes consumed before tile `t`: the superblock directory plus a bounded
// walk of at most (superblock - 1) tiles of plane words.  This is the
// "random access" arm of TBE section 3 and is what lets a k-chunk start
// anywhere without a per-row derived index -- zero extra resident bytes.
__device__ __forceinline__ uint32_t tbe_escape_base(
    const uint2* __restrict__ planes, const uint32_t* __restrict__ sbbase,
    uint32_t superblock, uint32_t t) {
    const uint32_t sb_idx = t / superblock;
    uint32_t base = sbbase[sb_idx];
    for (uint32_t u = sb_idx * superblock; u < t; ++u) {
        const uint2 q0 = planes[u * 3 + 0];
        const uint2 q1 = planes[u * 3 + 1];
        const uint2 q2 = planes[u * 3 + 2];
        base += __popc(tbe_esc_word(q0.x, q1.x, q2.x));
        base += __popc(tbe_esc_word(q0.y, q1.y, q2.y));
    }
    return base;
}

// ====================== END DECODE PATH ======================

struct TbeView {
    const uint2* planes;       // [T, 3] uint2  == [T, 3, 2] uint32
    const uint8_t* smb;        // [T * 64]
    const uint8_t* esc;        // [E]
    const uint32_t* sbbase;    // [ceil(T / superblock)]
    uint32_t superblock;
    uint32_t e01;              // exponent table bytes 0..3
    uint32_t e23;              // exponent table bytes 4..7
    int tiles_per_row;         // K / 64
};

// Decode-only proof kernel.  It runs the SAME tbe_lane_tile /
// tbe_step_fragment helpers, with the SAME element->lane map, as the GEMM
// below, so bit-exactness measured here is bit-exactness of the fragments the
// MMA consumes.  Each warp owns 8 rows and a contiguous run of tiles.
__global__ void _tbe_mma_decode_kernel(TbeView v, uint16_t* __restrict__ out,
                                       int n, int k, int tiles_per_chunk) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t gid = static_cast<uint32_t>(lane >> 2);
    const uint32_t j = static_cast<uint32_t>(lane & 3);
    const TbeLaneConst lc = tbe_lane_const(j);
    const int row = (blockIdx.x * kWarpsPerCta + warp) * 8 + static_cast<int>(gid);
    if (row >= n) {
        return;
    }
    const int kt0 = blockIdx.y * tiles_per_chunk;
    if (kt0 >= v.tiles_per_row) {
        return;
    }
    int kt1 = kt0 + tiles_per_chunk;
    if (kt1 > v.tiles_per_row) {
        kt1 = v.tiles_per_row;
    }

    const uint32_t t0 = static_cast<uint32_t>(row) *
                            static_cast<uint32_t>(v.tiles_per_row) +
                        static_cast<uint32_t>(kt0);
    uint32_t tile_base = tbe_escape_base(v.planes, v.sbbase, v.superblock, t0);
    const uint2* planes_t = v.planes + static_cast<size_t>(t0) * 3;
    const uint8_t* smb_t = v.smb + static_cast<size_t>(t0) * kTile;
    uint16_t* row_out = out + static_cast<size_t>(row) * k;

    for (int kt = kt0; kt < kt1; ++kt) {
        TbeLaneTile lt;
        tbe_lane_tile(planes_t, smb_t, v.esc, tile_base, lc, lt);
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            uint32_t b0, b1;
            tbe_step_fragment(lt, s, v.e01, v.e23, b0, b1);
            const int kbase = kt * kTile + s * 16 + static_cast<int>(2u * j);
            *reinterpret_cast<uint32_t*>(row_out + kbase) = b0;
            *reinterpret_cast<uint32_t*>(row_out + kbase + 8) = b1;
        }
        tile_base += lt.tile_esc;
        planes_t += 3;
        smb_t += kTile;
    }
}


// ====================== BEGIN MMA PATH =======================
// One mma.sync per (warp, n8 slab, k16 step).  fp32 accumulate, bf16 operands,
// no conversion instruction anywhere in the chain: the B fragment is assembled
// as raw bf16 bit patterns and the A fragment is loaded as raw bf16 pairs.
__device__ __forceinline__ void mma_m16n8k16_bf16(float (&d)[4],
                                                  const uint32_t (&a)[4],
                                                  uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Rows >= M are masked to zero rather than loaded: the MMA tile is 16 rows and
// this rung serves M = 1..16, so most of it is padding at M = 1 -- only the
// lanes with gid == 0 touch x at all.  The row pointer is hoisted to the top of
// the kernel and advanced by one tile per iteration, so a k16 step's load
// carries a compile-time immediate offset and no address arithmetic.
__device__ __forceinline__ uint32_t load_a_pair(const uint16_t* __restrict__ p,
                                                bool live, int offset) {
    if (!live) {
        return 0u;
    }
    return *reinterpret_cast<const uint32_t*>(p + offset);
}

// Partial GEMM.  grid.x indexes CTAs of kWarpsPerCta warps x kNSlab n8 slabs;
// grid.y indexes the deterministic k split.  Each (split, m, n) has exactly one
// owner and the reduction is a separate fixed-order pass -- no atomics.
//
// LOOP SHAPE.  The A fragments of all four k16 steps are loaded once and shared
// by the n8 slabs; each slab is then decoded and consumed in full before the
// next one starts, so only ONE slab's per-tile state is ever live.  That is
// what keeps the register file inside __launch_bounds__(kThreads, 6).
__global__ __launch_bounds__(kThreads, 6) void _tbe_mma_gemm_kernel(
    TbeView v, const uint16_t* __restrict__ x, float* __restrict__ ws, int m,
    int n, int k, int tiles_per_chunk) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t gid = static_cast<uint32_t>(lane >> 2);
    const uint32_t j = static_cast<uint32_t>(lane & 3);
    const TbeLaneConst lc = tbe_lane_const(j);
    const int n_base =
        (blockIdx.x * kWarpsPerCta + warp) * (kNSlab * kMmaN);
    if (n_base >= n) {
        return;
    }
    const int kt0 = blockIdx.y * tiles_per_chunk;
    if (kt0 >= v.tiles_per_row) {
        return;
    }
    int kt1 = kt0 + tiles_per_chunk;
    if (kt1 > v.tiles_per_row) {
        kt1 = v.tiles_per_row;
    }

    float acc[kNSlab][4];
    uint32_t tile_base[kNSlab];
    const uint2* planes_t[kNSlab];
    const uint8_t* smb_t[kNSlab];
#pragma unroll
    for (int sl = 0; sl < kNSlab; ++sl) {
        acc[sl][0] = 0.f;
        acc[sl][1] = 0.f;
        acc[sl][2] = 0.f;
        acc[sl][3] = 0.f;
        const int r = n_base + sl * kMmaN + static_cast<int>(gid);
        const int row = (r < n) ? r : (n - 1);   // clamped read, masked store
        const uint32_t t = static_cast<uint32_t>(row) *
                               static_cast<uint32_t>(v.tiles_per_row) +
                           static_cast<uint32_t>(kt0);
        tile_base[sl] = tbe_escape_base(v.planes, v.sbbase, v.superblock, t);
        planes_t[sl] = v.planes + static_cast<size_t>(t) * 3;
        smb_t[sl] = v.smb + static_cast<size_t>(t) * kTile;
    }

    // A-fragment row pointers, hoisted once and clamped so a masked-off lane
    // never even forms an out-of-range address.
    const bool live0 = static_cast<int>(gid) < m;
    const bool live8 = static_cast<int>(gid) + 8 < m;
    const uint32_t r0 = live0 ? gid : 0u;
    const uint32_t r8 = live8 ? (gid + 8u) : 0u;
    const int a_off = kt0 * kTile + static_cast<int>(2u * j);
    const uint16_t* xa0 = x + static_cast<size_t>(r0) * k + a_off;
    const uint16_t* xa8 = x + static_cast<size_t>(r8) * k + a_off;

    for (int kt = kt0; kt < kt1; ++kt) {
        // A fragments: 32-bit bf16 pairs, shared by both n8 slabs.  Rows >= M
        // are constant zero registers, never a load.
        uint32_t a[4][4];
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            a[s][0] = load_a_pair(xa0, live0, s * 16);
            a[s][1] = load_a_pair(xa8, live8, s * 16);
            a[s][2] = load_a_pair(xa0, live0, s * 16 + 8);
            a[s][3] = load_a_pair(xa8, live8, s * 16 + 8);
        }
#pragma unroll
        for (int sl = 0; sl < kNSlab; ++sl) {
            TbeLaneTile lt;
            tbe_lane_tile(planes_t[sl], smb_t[sl], v.esc, tile_base[sl], lc, lt);
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                uint32_t b0, b1;
                tbe_step_fragment(lt, s, v.e01, v.e23, b0, b1);
                mma_m16n8k16_bf16(acc[sl], a[s], b0, b1);
            }
            tile_base[sl] += lt.tile_esc;
            planes_t[sl] += 3;
            smb_t[sl] += kTile;
        }
        xa0 += kTile;
        xa8 += kTile;
    }

    // Epilogue: lane L owns D[gid][2j], D[gid][2j+1], D[gid+8][2j],
    // D[gid+8][2j+1] of each n8 slab -- one owner, no reduction inside the warp.
    float* base = ws + static_cast<size_t>(blockIdx.y) * kMmaM * n;
#pragma unroll
    for (int sl = 0; sl < kNSlab; ++sl) {
        if (n_base + sl * kMmaN >= n) {
            continue;   // N % 8 == 0, so a slab is wholly in or wholly out
        }
        const int col = n_base + sl * kMmaN + static_cast<int>(2u * j);
        float2 lo, hi;
        lo.x = acc[sl][0];
        lo.y = acc[sl][1];
        hi.x = acc[sl][2];
        hi.y = acc[sl][3];
        *reinterpret_cast<float2*>(base + static_cast<size_t>(gid) * n + col) =
            lo;
        *reinterpret_cast<float2*>(base +
                                   static_cast<size_t>(gid + 8) * n + col) = hi;
    }
}
// ======================= END MMA PATH ========================

// Fixed-order split-K reduction and epilogue.  This is the ONLY place a
// float -> bf16 conversion happens, and it is round-to-nearest-even, which is
// what torch uses.  Summing the splits in index order makes the result
// deterministic run to run.
__global__ void _tbe_mma_reduce_kernel(const float* __restrict__ ws,
                                       const uint16_t* __restrict__ bias,
                                       uint16_t* __restrict__ out, int m, int n,
                                       int split_k) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) {
        return;
    }
    const int row = idx / n;
    const int col = idx - row * n;
    float total = 0.f;
    for (int s = 0; s < split_k; ++s) {
        total += ws[(static_cast<size_t>(s) * kMmaM + row) * n + col];
    }
    if (bias != nullptr) {
        total += __bfloat162float(__ushort_as_bfloat16(bias[col]));
    }
    out[idx] = __bfloat16_as_ushort(__float2bfloat16_rn(total));
}

TbeView make_view(const torch::Tensor& planes, const torch::Tensor& smb,
                  const torch::Tensor& esc, const torch::Tensor& sbbase,
                  int64_t superblock, int64_t e01, int64_t e23,
                  int64_t tiles_per_row) {
    TbeView v;
    v.planes = reinterpret_cast<const uint2*>(planes.data_ptr());
    v.smb = smb.data_ptr<uint8_t>();
    v.esc = esc.data_ptr<uint8_t>();
    v.sbbase = reinterpret_cast<const uint32_t*>(sbbase.data_ptr());
    v.superblock = static_cast<uint32_t>(superblock);
    v.e01 = static_cast<uint32_t>(e01);
    v.e23 = static_cast<uint32_t>(e23);
    v.tiles_per_row = static_cast<int>(tiles_per_row);
    return v;
}

void check_container(const torch::Tensor& planes, const torch::Tensor& smb,
                     const torch::Tensor& esc, const torch::Tensor& sbbase,
                     int64_t n, int64_t k, int64_t superblock) {
    TORCH_CHECK(k > 0 && k % kTile == 0,
                "TBE MMA requires K a positive multiple of 64, got ", k);
    TORCH_CHECK(n > 0 && n % kMmaN == 0,
                "TBE MMA requires N a positive multiple of 8, got ", n);
    TORCH_CHECK(planes.is_cuda() && smb.is_cuda() && sbbase.is_cuda(),
                "TBE MMA container fields must be CUDA-resident");
    TORCH_CHECK(planes.device() == smb.device() &&
                    planes.device() == sbbase.device() &&
                    (esc.numel() == 0 || esc.device() == planes.device()),
                "TBE MMA container fields must share one device");
    TORCH_CHECK(planes.is_contiguous() && smb.is_contiguous() &&
                    esc.is_contiguous() && sbbase.is_contiguous(),
                "TBE MMA container fields must be contiguous");
    TORCH_CHECK(planes.scalar_type() == at::kInt,
                "planes must be int32 (uint32 words)");
    TORCH_CHECK(smb.scalar_type() == at::kByte, "smb must be uint8");
    TORCH_CHECK(esc.scalar_type() == at::kByte, "esc must be uint8");
    TORCH_CHECK(sbbase.scalar_type() == at::kInt, "sbbase must be int32");
    const int64_t tiles = n * k / kTile;
    TORCH_CHECK(planes.numel() == tiles * 6,
                "planes must hold 6 words per tile, expected ", tiles * 6,
                " got ", planes.numel());
    TORCH_CHECK(smb.numel() == tiles * kTile,
                "smb must hold 64 bytes per tile, expected ", tiles * kTile,
                " got ", smb.numel());
    // The escape path issues four UNCONDITIONAL byte loads at esc + base, so
    // esc[] must be readable kEscPad bytes past its last real entry.  The host
    // uploader pads it and charges the padding separately; refusing here is
    // what stops an unpadded buffer from reading whatever follows it.
    TORCH_CHECK(esc.numel() >= kEscPad,
                "esc must carry at least ", kEscPad,
                " bytes of zero padding past its last entry, got ",
                esc.numel());
    TORCH_CHECK(superblock > 0, "superblock must be positive");
    TORCH_CHECK(sbbase.numel() == (tiles + superblock - 1) / superblock,
                "sbbase must hold one base per superblock");
}

void tbe_mma_decode(torch::Tensor planes, torch::Tensor smb, torch::Tensor esc,
                    torch::Tensor sbbase, torch::Tensor out, int64_t n,
                    int64_t k, int64_t superblock, int64_t e01, int64_t e23,
                    int64_t stream_ptr) {
    c10::cuda::CUDAGuard device_guard(planes.device());
    check_container(planes, smb, esc, sbbase, n, k, superblock);
    TORCH_CHECK(out.is_cuda() && out.device() == planes.device(),
                "decode output must share the container device");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16, "decode output must be bf16");
    TORCH_CHECK(out.is_contiguous() && out.numel() == n * k,
                "decode output must be a contiguous [N, K] tensor");
    const int tiles_per_row = static_cast<int>(k / kTile);
    TbeView v = make_view(planes, smb, esc, sbbase, superblock, e01, e23,
                          tiles_per_row);
    const int grid_x = static_cast<int>((n + kWarpsPerCta * 8 - 1) /
                                        (kWarpsPerCta * 8));
    int chunks = (2048 + grid_x - 1) / (grid_x > 0 ? grid_x : 1);
    if (chunks < 1) {
        chunks = 1;
    }
    if (chunks > tiles_per_row) {
        chunks = tiles_per_row;
    }
    const int tiles_per_chunk = (tiles_per_row + chunks - 1) / chunks;
    const int grid_y = (tiles_per_row + tiles_per_chunk - 1) / tiles_per_chunk;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    _tbe_mma_decode_kernel<<<dim3(grid_x, grid_y), kThreads, 0, stream>>>(
        v, reinterpret_cast<uint16_t*>(out.data_ptr()), static_cast<int>(n),
        static_cast<int>(k), tiles_per_chunk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tbe_mma_gemm(torch::Tensor x, torch::Tensor planes, torch::Tensor smb,
                  torch::Tensor esc, torch::Tensor sbbase, torch::Tensor bias,
                  torch::Tensor ws, torch::Tensor out, int64_t m, int64_t n,
                  int64_t k, int64_t superblock, int64_t e01, int64_t e23,
                  int64_t split_k, int64_t stream_ptr) {
    c10::cuda::CUDAGuard device_guard(planes.device());
    check_container(planes, smb, esc, sbbase, n, k, superblock);
    TORCH_CHECK(m >= 1 && m <= kMmaM,
                "this rung serves M = 1..16 by one m16n8k16 tile, got ", m);
    TORCH_CHECK(x.is_cuda() && out.is_cuda() && ws.is_cuda() && bias.is_cuda(),
                "TBE MMA GEMM operands must be CUDA-resident");
    TORCH_CHECK(x.device() == planes.device() && out.device() == planes.device() &&
                    ws.device() == planes.device() &&
                    bias.device() == planes.device(),
                "TBE MMA GEMM operands must share the container device");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 &&
                    out.scalar_type() == at::kBFloat16 &&
                    bias.scalar_type() == at::kBFloat16,
                "x, bias and out must be bf16");
    TORCH_CHECK(ws.scalar_type() == at::kFloat, "workspace must be fp32");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous() && ws.is_contiguous() &&
                    bias.is_contiguous(),
                "TBE MMA GEMM operands must be contiguous");
    TORCH_CHECK(x.numel() == m * k, "x must be [M, K]");
    TORCH_CHECK(out.numel() == m * n, "out must be [M, N]");
    TORCH_CHECK(bias.numel() == 0 || bias.numel() == n, "bias must be [N]");
    const int tiles_per_row = static_cast<int>(k / kTile);
    TORCH_CHECK(split_k >= 1 && split_k <= tiles_per_row,
                "split_k must be in [1, K/64], got ", split_k);
    TORCH_CHECK(ws.numel() == split_k * kMmaM * n,
                "workspace must be [split_k, 16, N] fp32");
    TbeView v = make_view(planes, smb, esc, sbbase, superblock, e01, e23,
                          tiles_per_row);
    const int tiles_per_chunk =
        (tiles_per_row + static_cast<int>(split_k) - 1) / static_cast<int>(split_k);
    const int grid_x = static_cast<int>(
        (n + kWarpsPerCta * kNSlab * kMmaN - 1) / (kWarpsPerCta * kNSlab * kMmaN));
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    _tbe_mma_gemm_kernel<<<dim3(grid_x, static_cast<int>(split_k)), kThreads, 0,
                           stream>>>(
        v, reinterpret_cast<const uint16_t*>(x.data_ptr()),
        ws.data_ptr<float>(), static_cast<int>(m), static_cast<int>(n),
        static_cast<int>(k), tiles_per_chunk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int total = static_cast<int>(m * n);
    const uint16_t* bias_ptr =
        bias.numel() ? reinterpret_cast<const uint16_t*>(bias.data_ptr())
                     : nullptr;
    _tbe_mma_reduce_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        ws.data_ptr<float>(), bias_ptr,
        reinterpret_cast<uint16_t*>(out.data_ptr()), static_cast<int>(m),
        static_cast<int>(n), static_cast<int>(split_k));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("tbe_mma_decode", &tbe_mma_decode,
               "TBE fragment decode proof (dense bf16 out)");
    module.def("tbe_mma_gemm", &tbe_mma_gemm,
               "TBE decode straight into m16n8k16 B fragments, M = 1..16");
}
