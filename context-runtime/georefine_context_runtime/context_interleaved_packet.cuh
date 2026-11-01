#pragma once
#include <cstdint>

// Shared bounded integer decoder: compiled by nvcc for CUDA and by the focused
// CPU harness for arithmetic/bounds verification. No host-only decoder fallback.
#ifdef __CUDACC__
#define BCI_HD __host__ __device__
#else
#define BCI_HD
#endif
namespace bci {
constexpr uint32_t L = 1u << 23;
constexpr uint64_t MAX_WORDS = 65536;
BCI_HD inline uint32_t u32(const uint8_t* p) {
  return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
BCI_HD inline uint32_t u16(const uint8_t* p) { return uint32_t(p[0]) | uint32_t(p[1]) << 8; }
BCI_HD inline bool span(uint64_t offset, uint64_t length, uint64_t size) {
  return offset <= size && length <= size-offset;
}
// Status bits: 1 geometry/bounds; 2 sparse CDF; 4 stream/state;
// 8 residual; 16 noncanonical termination. One status per packet/lane.
BCI_HD inline uint32_t decode_lane(const uint8_t* data, uint64_t size,
    const int64_t* d, uint32_t rows, uint32_t cols, uint32_t nr, uint32_t nc,
    uint64_t labels, uint32_t lane, uint16_t* out, uint64_t out_size) {
  if (!rows || !cols || uint64_t(rows)*cols > MAX_WORDS ||
      !nr || !nc || uint64_t(nr)*nc > 1024 || lane >= 32 ||
      out_size != uint64_t(rows)*cols || !span(labels,uint64_t(rows)+cols,size)) return 1;
  for (int k=0;k<9;++k) if (d[k]<0) return 1;
  uint64_t mode=d[0], row0=d[1], pr=d[2], ts=d[3], nt=d[4], ms=d[5], nm=d[6], rs=d[7], nd=d[8];
  if (mode>1 || !pr || row0>=rows || pr>rows-row0 ||
      !span(ts,nt,size) || !span(ms,nm,size) || !span(rs,nd,size) ||
      ts+nt!=ms || ms+nm!=rs) return 1;
  const uint64_t n=pr*cols, output_start=row0*cols;
  if (mode) {
    if (nt || nd || nm != 2*n) return 1;
    for (uint64_t i=lane;i<n;i+=32) out[output_start+i]=uint16_t(u16(data+ms+2*i));
    return 0;
  }
  const uint32_t nctx=nr*nc;
  const uint64_t dir_bytes=4ull*(nctx+1), es=ts+dir_bytes;
  if (nt<dir_bytes || nm<260 || nd!=(6*n+7)/8) return 1;
  if (u32(data+ts)!=0 || uint64_t(u32(data+ts+4ull*nctx))*6!=nt-dir_bytes) return 2;
  if (u32(data+ms)!=0 || u32(data+ms+128)!=nm-132) return 4;
  uint64_t begin=u32(data+ms+4*lane), end=u32(data+ms+4*(lane+1));
  if (begin>end || end>nm-132 || end-begin<4) return 4;
  uint64_t pos=ms+132+begin, stop=ms+132+end;
  uint32_t state=u32(data+pos); pos+=4;
  if (state<L || state>=uint64_t(L)*256) return 4;
  if ((6*n)%8 && data[rs+nd-1] >> ((6*n)%8)) return 8;
  for (uint64_t i=lane;i<n;i+=32) {
    uint32_t r=data[labels+row0+i/cols], c=data[labels+rows+i%cols];
    if (r>=nr || c>=nc) return 1;
    uint32_t cx=r*nc+c, a=u32(data+ts+4*cx), b=u32(data+ts+4*(cx+1));
    uint32_t total_entries=(nt-dir_bytes)/6;
    if (a>=b || b>total_entries || b-a>1024) return 2;
    if (u32(data+es+6ull*(b-1)+2)!=65536) return 2;
    uint32_t slot=state&65535, low=a, high=b;
    while (low<high) {
      uint32_t mid=low+(high-low)/2;
      uint32_t upper=u32(data+es+6ull*mid+2);
      if (upper<=slot) low=mid+1; else high=mid;
    }
    if (low>=b) return 2;
    uint32_t sym=u16(data+es+6ull*low), upper=u32(data+es+6ull*low+2);
    uint32_t lower=low==a ? 0 : u32(data+es+6ull*(low-1)+2);
    if (sym>=1024 || upper>65536 || lower>slot || slot>=upper) return 2;
    uint64_t next=uint64_t(upper-lower)*(state>>16)+slot-lower;
    if (next>=uint64_t(L)*256) return 4;
    state=uint32_t(next);
    while (state<L) {
      if (pos>=stop) return 4;
      state=(state<<8)|data[pos++];
    }
    uint64_t bit=6*i, byte=bit/8;
    if (byte>=nd) return 8;
    uint32_t residual=data[rs+byte]>>(bit%8);
    if (bit%8>2) {
      if (byte+1>=nd) return 8;
      residual|=uint32_t(data[rs+byte+1])<<(8-bit%8);
    }
    residual&=63;
    out[output_start+i]=uint16_t((sym<<5)|(residual&31)|((residual>>5)<<15));
  }
  return (state==L && pos==stop) ? 0 : 16;
}
} // namespace bci
#undef BCI_HD
