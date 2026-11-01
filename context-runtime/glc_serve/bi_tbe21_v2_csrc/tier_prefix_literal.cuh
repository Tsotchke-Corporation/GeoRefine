#pragma once
#include <stdint.h>

// Standalone literal CUDA-style core. Included after packedscan_simt/simt.h by
// the host harness; the same function body uses documented CUDA shuffle forms.
struct TpfInput {
  uint32_t l1c16;  // eight 2-bit L1 symbols, one original source group
  uint32_t l2f16;  // eight fixed-group 2-bit L2 digits
  uint32_t l3f16;  // eight fixed-group 2-bit L3 digits
  uint32_t l4f24;  // eight fixed-group 3-bit L4 digits
  uint32_t header_n1, header_n2, header_n3, header_n4;
};

struct TpfOutput {
  uint32_t f2, f3, f4;
  uint32_t q1, q2, q3, rho4;
  uint32_t k1, k2, k3, k4, n4;
  uint32_t n1, n2, n3, header_match;
};

__device__ __forceinline__ uint32_t tpf_mask2(uint32_t payload, uint32_t escape) {
  uint32_t x = payload ^ (escape * 0x5555u);
  x = ~(x | (x >> 1)) & 0x5555u;
  x = (x | (x >> 1)) & 0x3333u;
  x = (x | (x >> 2)) & 0x0f0fu;
  return (x | (x >> 4)) & 0xffu;
}

__device__ __forceinline__ uint32_t tpf_mask3(uint32_t payload, uint32_t escape) {
  uint32_t x = payload ^ (escape * 0x249249u);
  x = ~(x | (x >> 1) | (x >> 2)) & 0x249249u;
  x = (x | (x >> 2)) & 0x030c30c3u;
  x = (x | (x >> 4)) & 0x0300f00fu;
  x = (x | (x >> 8)) & 0xff0000ffu;
  return (x | (x >> 16)) & 0xffu;
}

__device__ __forceinline__ uint32_t tpf_pop2(uint32_t payload, uint32_t count, uint32_t escape) {
  return __popc(tpf_mask2(payload, escape) & ((1u << count) - 1u));
}

__device__ __forceinline__ uint32_t tpf_pop3(uint32_t payload, uint32_t count, uint32_t escape) {
  return __popc(tpf_mask3(payload, escape) & ((1u << count) - 1u));
}

__device__ __forceinline__ uint32_t tpf_span(uint64_t pair, uint32_t offset,
                                              uint32_t count, uint32_t width) {
  const uint32_t bits = count * width;
  const uint64_t mask = bits == 0 ? 0ull : ((1ull << bits) - 1ull);
  return (uint32_t)((pair >> (offset * width)) & mask);
}

// Inputs are fixed-eight groups: lane j owns one L1 source group and the jth
// padded payload group for each compact tier stream. Every lane in both
// halfwarps executes the same four scan and six gather collectives.
template <bool SkipEmpty4 = false>
__device__ __forceinline__ void tpf_literal(const TpfInput& in, int lane, TpfOutput& out, bool active4 = true) {
  const uint32_t j = (uint32_t)(lane & 15);
  const uint32_t c1 = __popc(tpf_mask2(in.l1c16, 3u));
  const uint32_t c2 = __popc(tpf_mask2(in.l2f16, 3u));
  const uint32_t c3 = __popc(tpf_mask2(in.l3f16, 3u));
  const uint32_t c4 = __popc(tpf_mask3(in.l4f24, 7u));
  const uint32_t counts = c1 | (c2 << 8) | (c3 << 16) | (c4 << 24);

  uint32_t inclusive = counts;
#pragma unroll
  for (int distance = 1; distance < 16; distance <<= 1) {
    const uint32_t peer = __shfl_up_sync(0xffffffffu, inclusive, distance, 16);
    if (j >= (uint32_t)distance) inclusive += peer;
  }
  const uint32_t exclusive = inclusive - counts;
  // Header counts are supplied for the explicit endpoint (128) and checked
  // against the row total using one uniform broadcast from lane 15.
  const uint32_t packed_total = __shfl_sync(0xffffffffu, inclusive, 15, 16);
  const uint32_t total1 = packed_total & 255u;
  const uint32_t total2 = (packed_total >> 8) & 255u;
  const uint32_t total3 = (packed_total >> 16) & 255u;
  const uint32_t total4 = (packed_total >> 24) & 255u;
  const uint32_t e1 = (exclusive >> 0) & 255u;
  const uint32_t e2 = (exclusive >> 8) & 255u;
  const uint32_t e3 = (exclusive >> 16) & 255u;
  const uint32_t e4 = (exclusive >> 24) & 255u;

  out.k1 = c1; out.k2 = c2; out.k3 = c3; out.k4 = c4;
  out.q1 = e1;
  out.n1 = total1; out.n2 = total2; out.n3 = total3; out.n4 = total4;
  out.header_match = (total1 == in.header_n1 && total2 == in.header_n2 &&
                      total3 == in.header_n3 && total4 == in.header_n4 &&
                      (!SkipEmpty4 || active4 || in.header_n3 == 0u));

  // One lane-local package carries payload + escape mask + exclusive prefix.
  const uint32_t l2mask = tpf_mask2(in.l2f16, 3u);
  const uint32_t l3mask = tpf_mask2(in.l3f16, 3u);
  const uint32_t l2pkg = (in.l2f16 & 0xffffu) | (l2mask << 16) | (e2 << 24);
  const uint32_t l3pkg = (in.l3f16 & 0xffffu) | (l3mask << 16) | (e3 << 24);
  const uint32_t l4pkg = (in.l4f24 & 0x00ffffffu) | (e4 << 24);

  const uint32_t g2 = e1 >> 3, o2 = e1 & 7u;
  const uint32_t l2a = __shfl_sync(0xffffffffu, l2pkg, (int)g2, 16);
  const uint32_t l2b = __shfl_sync(0xffffffffu, l2pkg, (int)((g2 + 1u) & 15u), 16);
  const uint32_t l2pair = (l2a & 0xffffu) | ((uint64_t)(l2b & 0xffffu) << 16);
  out.f2 = tpf_span(l2pair, o2, c1, 2u);
  const uint32_t l2mask0 = (l2a >> 16) & 255u;
  const uint32_t before2 = o2 ? ((1u << o2) - 1u) : 0u;
  out.q2 = ((l2a >> 24) & 255u) + __popc(l2mask0 & before2);
  out.k2 = tpf_pop2(out.f2, c1, 3u);

  const uint32_t g3 = out.q2 >> 3, o3 = out.q2 & 7u;
  const uint32_t l3a = __shfl_sync(0xffffffffu, l3pkg, (int)g3, 16);
  const uint32_t l3b = __shfl_sync(0xffffffffu, l3pkg, (int)((g3 + 1u) & 15u), 16);
  const uint32_t l3pair = (l3a & 0xffffu) | ((uint64_t)(l3b & 0xffffu) << 16);
  out.f3 = tpf_span(l3pair, o3, out.k2, 2u);
  const uint32_t l3mask0 = (l3a >> 16) & 255u;
  const uint32_t before3 = o3 ? ((1u << o3) - 1u) : 0u;
  out.q3 = ((l3a >> 24) & 255u) + __popc(l3mask0 & before3);
  out.k3 = tpf_pop2(out.f3, out.k2, 3u);

  // Caller derives active4 by a full-warp predicate. The actual tier counts
  // above are still checked, including malformed caller/header combinations.
  if constexpr (SkipEmpty4) {
    if (!active4) {
      out.f4 = 0u; out.rho4 = 0u; out.k4 = 0u;
      return;
    }
  }
  const uint32_t g4 = out.q3 >> 3, o4 = out.q3 & 7u;
  const uint32_t l4a = __shfl_sync(0xffffffffu, l4pkg, (int)g4, 16);
  const uint32_t l4b = __shfl_sync(0xffffffffu, l4pkg, (int)((g4 + 1u) & 15u), 16);
  const uint64_t l4pair = (l4a & 0x00ffffffu) | ((uint64_t)(l4b & 0x00ffffffu) << 24);
  out.f4 = tpf_span(l4pair, o4, out.k3, 3u);
  const uint32_t l4mask0 = tpf_mask3(l4a & 0x00ffffffu, 7u);
  const uint32_t before4 = o4 ? ((1u << o4) - 1u) : 0u;
  out.rho4 = ((l4a >> 24) & 255u) + __popc(l4mask0 & before4);
  out.k4 = tpf_pop3(out.f4, out.k3, 7u);
}

// Host/device endpoint contract used by scalar tests; t==128 must not read mask[16].
__host__ __device__ __forceinline__ uint32_t tpf_rank_endpoint(
    uint32_t prefix_before_group, uint32_t mask8, uint32_t t, uint32_t total) {
  if (t == 128u) return total;
  const uint32_t within = t & 7u;
  const uint32_t before = within ? ((1u << within) - 1u) : 0u;
#if defined(__CUDA_ARCH__)
  return prefix_before_group + __popc(mask8 & before);
#else
  return prefix_before_group + (uint32_t)__builtin_popcount(mask8 & before);
#endif
}
