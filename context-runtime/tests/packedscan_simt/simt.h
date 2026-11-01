// SIMT on a CPU: run the LITERAL device source of bi_gemm.cuh's TBE2 decode, one OS thread per
// CUDA lane, with the warp collectives it uses implemented by their documented semantics.
//
//   __shfl_sync(m, v, src, w)     lane reads v of lane (seg_base + src % w); full mask enforced
//   __shfl_up_sync(m, v, d, w)    lane reads lane - d, or its own v when (lane % w) < d
//   __any_sync(m, p)              OR of p over the 32 lanes
//   __syncthreads()               barrier of the threads that run together (see harness.cpp)
//   __popc, __funnelshift_r, __dp4a (unsigned), __byte_perm, min/max: their PTX definitions
//   __ldg(p)                      *p, after checking p lies inside a registered buffer
//                                 (an out-of-bounds read aborts: the format's tail-padding
//                                 contract is checked, not assumed)
//
// Every collective is a two-phase std::barrier exchange among the 32 lanes, and each lane
// carries a collective sequence number: a lane reading a value posted at a DIFFERENT collective
// (divergent shuffles -- undefined on the GPU) aborts the run.
#pragma once
#include <algorithm>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define __device__
#define __host__
#define __global__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __shared__
#define __align__(n)                 // the harness defines smraw itself, 16-byte aligned

struct uint2 { uint32_t x, y; };
struct uint4 { uint32_t x, y, z, w; };
struct dim3s { unsigned x = 0, y = 0, z = 0; };

namespace simt {
struct Warp {
  std::barrier<>* bar = nullptr;
  uint64_t val[32];
  uint64_t seq[32];
};
inline thread_local Warp* g_warp = nullptr;
inline thread_local int g_lane = 0;
inline thread_local uint64_t g_seq = 0;
inline std::barrier<>* g_cta_bar = nullptr;
struct Range { const char* b; const char* e; const char* what; };
inline std::vector<Range> g_ranges;
inline std::atomic<uint64_t> g_collectives{0};

[[noreturn]] inline void die(const char* m) {
  std::fprintf(stderr, "SIMT ABORT: %s\n", m);
  std::fflush(stderr);
  std::_Exit(3);
}
inline void reg(const void* p, size_t n, const char* what) {
  g_ranges.push_back({(const char*)p, (const char*)p + n, what});
}
inline void check(const void* p, size_t n) {
  const char* c = (const char*)p;
  for (auto& r : g_ranges)
    if (c >= r.b && c + n <= r.e) return;
  char buf[160];
  std::snprintf(buf, sizeof buf, "out-of-bounds __ldg of %zu bytes at %p", n, p);
  die(buf);
}
inline uint64_t xchg(uint64_t v, int src, unsigned mask) {
  if (mask != 0xffffffffu) die("partial warp mask");
  Warp& w = *g_warp;
  w.val[g_lane] = v;
  w.seq[g_lane] = g_seq;
  w.bar->arrive_and_wait();
  if (w.seq[src] != g_seq) die("divergent collective: source lane is at a different shuffle");
  const uint64_t r = w.val[src];
  w.bar->arrive_and_wait();
  ++g_seq;
  if (g_lane == 0) g_collectives.fetch_add(1, std::memory_order_relaxed);
  return r;
}
}  // namespace simt

inline thread_local dim3s threadIdx, blockIdx;

template <class T> inline T __ldg(const T* p) { simt::check(p, sizeof(T)); return *p; }
// bi_gemm.cuh's ld_nc_u4/u2/u32/u16 are `asm volatile("ld.global.nc...")` -- a non-coherent-cache
// HINT, semantic-free on a CPU.  Host versions: bounds-checked plain reads (KERNEL_V2_SPEC
// §ground-truth: "scripts/batchserve/bigemm_v2_simt.py ... does NOT provide ld_nc_u4/u2/u32/u16
// ... add host versions").
// Global scope (not namespace bi): the extraction never cuts bi_gemm.cuh's own asm-based
// ld_nc_* (lines 117-137), so the extracted Ld<F_TBE>/LdTbeV2 text's unqualified calls resolve
// to these by ordinary enclosing-namespace lookup, exactly as __ldg/__popc/__byte_perm do today.
inline uint4 ld_nc_u4(const void* p) { simt::check(p, 16); return *(const uint4*)p; }
inline uint2 ld_nc_u2(const void* p) { simt::check(p, 8); return *(const uint2*)p; }
inline uint32_t ld_nc_u32(const void* p) { simt::check(p, 4); return *(const uint32_t*)p; }
inline int32_t ld_nc_i32(const void* p) { simt::check(p, 4); return *(const int32_t*)p; }
inline uint32_t ld_nc_u16(const void* p) { simt::check(p, 2); return (uint32_t) * (const unsigned short*)p; }
inline uint4 make_uint4(uint32_t x, uint32_t y, uint32_t z, uint32_t w) { return uint4{x, y, z, w}; }
// bi_gemm.cuh's `lop3` (opledger T1) is `asm volatile("lop3.b32 ...")`, placed beside ld_nc_u32
// so the SIMT extraction never cuts it (same reasoning as ld_nc_*): a host model by the exact
// PTX immLut semantics -- bit i of lut is F(a_bit,b_bit,c_bit) for index i=(a<<2)|(b<<1)|c (the
// canonical a_lut=0xF0/b_lut=0xCC/c_lut=0xAA decomposition) -- resolves the extracted,
// unqualified `lop3(...)` call by ordinary enclosing-namespace lookup.
inline uint32_t lop3(uint32_t a, uint32_t b, uint32_t c, uint32_t lut) {
  uint32_t r = 0;
  for (int bit = 0; bit < 32; ++bit) {
    const unsigned idx = (((a >> bit) & 1u) << 2) | (((b >> bit) & 1u) << 1) | ((c >> bit) & 1u);
    r |= ((lut >> idx) & 1u) << bit;
  }
  return r;
}
inline uint32_t __shfl_sync(unsigned m, uint32_t v, int src, int w = 32) {
  const int base = simt::g_lane & ~(w - 1);
  return (uint32_t)simt::xchg(v, base + (src & (w - 1)), m);
}
inline uint32_t __shfl_up_sync(unsigned m, uint32_t v, unsigned d, int w = 32) {
  const int l = simt::g_lane;
  const int src = ((l & (w - 1)) < (int)d) ? l : l - (int)d;
  return (uint32_t)simt::xchg(v, src, m);
}
inline uint32_t __shfl_xor_sync(unsigned m, uint32_t v, int delta, int w = 32) {
  const int l = simt::g_lane;
  const int peer = l ^ delta;
  const int src = peer < (l & ~(w - 1)) + w ? peer : l;
  return (uint32_t)simt::xchg(v, src, m);
}
inline int __any_sync(unsigned m, unsigned p) {
  if (m != 0xffffffffu) simt::die("partial warp mask");
  simt::Warp& w = *simt::g_warp;
  w.val[simt::g_lane] = p != 0;
  w.seq[simt::g_lane] = simt::g_seq;
  w.bar->arrive_and_wait();
  int any = 0;
  for (int i = 0; i < 32; ++i) {
    if (w.seq[i] != simt::g_seq) simt::die("divergent __any_sync");
    any |= (int)w.val[i];
  }
  w.bar->arrive_and_wait();
  ++simt::g_seq;
  return any;
}
// __ballot_sync by its documented semantics (bit i of the result = lane i's predicate), the
// same two-phase barrier exchange as __any_sync.  Used by t21_hscan16's BI_T21_SCAN=1 arm.
inline unsigned __ballot_sync(unsigned m, unsigned p) {
  if (m != 0xffffffffu) simt::die("partial warp mask");
  simt::Warp& w = *simt::g_warp;
  w.val[simt::g_lane] = p != 0;
  w.seq[simt::g_lane] = simt::g_seq;
  w.bar->arrive_and_wait();
  unsigned mask = 0;
  for (int i = 0; i < 32; ++i) {
    if (w.seq[i] != simt::g_seq) simt::die("divergent __ballot_sync");
    mask |= (w.val[i] ? 1u : 0u) << i;
  }
  w.bar->arrive_and_wait();
  ++simt::g_seq;
  return mask;
}
inline void __syncthreads() { simt::g_cta_bar->arrive_and_wait(); }
inline int __popc(uint32_t x) { return __builtin_popcount(x); }
inline uint32_t __funnelshift_r(uint32_t lo, uint32_t hi, uint32_t sh) {
  return (uint32_t)((((uint64_t)hi << 32) | lo) >> (sh & 31u));
}
inline unsigned __dp4a(unsigned a, unsigned b, unsigned c) {
  for (int i = 0; i < 4; ++i) c += ((a >> (8 * i)) & 0xffu) * ((b >> (8 * i)) & 0xffu);
  return c;
}
// PTX prmt.b32 (default mode) as exposed by __byte_perm: result byte n = byte (s >> 4n) & 7 of
// the 8-byte value {y, x}.  (__byte_perm ignores the selector's sign-replicate bit.)
inline uint32_t __byte_perm(uint32_t x, uint32_t y, uint32_t s) {
  const uint64_t v = ((uint64_t)y << 32) | x;
  uint32_t r = 0;
  for (int n = 0; n < 4; ++n) r |= (uint32_t)((v >> (8 * ((s >> (4 * n)) & 7u))) & 0xffu) << (8 * n);
  return r;
}
inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }
inline unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }
inline unsigned max(unsigned a, unsigned b) { return a > b ? a : b; }
