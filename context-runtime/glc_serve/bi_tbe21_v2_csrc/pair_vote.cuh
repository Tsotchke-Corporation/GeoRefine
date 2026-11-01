#pragma once
#include <stdint.h>

// Paired prefix helper. Caller supplies hardware lane 0..31 and j == (lane & 15).
// All 32 lanes must execute every ballot; counts must be in [0,8].
template <int B>
__device__ __forceinline__ void t21_pair_plane(uint32_t a, uint32_t b,
    uint32_t lt, uint32_t hm, uint32_t& rank, uint32_t& total) {
  const uint32_t ma = (uint32_t)__ballot_sync(0xffffffffu, (a >> B) & 1u);
  const uint32_t mb = (uint32_t)__ballot_sync(0xffffffffu, (b >> B) & 1u);
  rank = (uint32_t)__popc(ma & lt) | ((uint32_t)__popc(mb & lt) << 16);
  total = (uint32_t)__popc(ma & hm) | ((uint32_t)__popc(mb & hm) << 16);
}

__device__ __forceinline__ void t21_hscan16_pair_ballot(uint32_t a, uint32_t b,
    int lane, int j, uint32_t& ra, uint32_t& rb, uint32_t& na, uint32_t& nb) {
  const uint32_t hb = (uint32_t)(lane & 16), hm = 0xFFFFu << hb;
  const uint32_t lt = ((1u << j) - 1u) << hb;
  uint32_t r0, r1, r2, r3, n0, n1, n2, n3;
  t21_pair_plane<0>(a, b, lt, hm, r0, n0);
  t21_pair_plane<1>(a, b, lt, hm, r1, n1);
  const uint32_t r01 = r0 + (r1 << 1), n01 = n0 + (n1 << 1);
  t21_pair_plane<2>(a, b, lt, hm, r2, n2);
  t21_pair_plane<3>(a, b, lt, hm, r3, n3);
  const uint32_t r23 = (r2 << 2) + (r3 << 3);
  const uint32_t n23 = (n2 << 2) + (n3 << 3);
  const uint32_t rank = r01 + r23, total = n01 + n23;
  ra = rank & 0xffffu; rb = rank >> 16;
  na = total & 0xffffu; nb = total >> 16;
}
