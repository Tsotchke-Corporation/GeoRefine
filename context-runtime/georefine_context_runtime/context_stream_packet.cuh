#pragma once
#include "context_tiled_packet.cuh"
#ifdef __CUDACC__
#define BCS_HD __host__ __device__
#else
#define BCS_HD
#endif

namespace bcs {
constexpr uint32_t N=64, CH=128, LDS=136;
// Immutable validated metadata view; keep it and its source bytes alive and
// unchanged for the cursor lifetime. No full decoded weight allocation.
struct PacketView {
  const uint8_t* data=nullptr;
  uint32_t size=0,K=0,nr=0,nc=0,labels=0,ts=0,nt=0;
  uint32_t row0=0,mode=0,ms=0,nm=0,rs=0,nd=0,es=0,entries=0,error=1;
};
struct LaneCursor {
  uint32_t state=0,pos=0,stop=0,next_chunk=0,error=0,lane=0;
  const PacketView* view=nullptr;
};
// Status:1 geometry/bounds/order;2 CDF;4 stream/state;8 residual;16 termination.
BCS_HD inline uint32_t prepare_packet(const uint8_t* data,uint64_t size,const int64_t* d,
    uint32_t K,uint32_t nr,uint32_t nc,uint64_t labels,uint64_t ts,uint64_t nt,
    uint32_t split_k,PacketView& v) {
  v=PacketView{};
  if (!data || !d || size>1048576 || (K!=512 && K!=1024) || split_k!=1 ||
      (nr!=1 && nr!=4 && nr!=16 && nr!=64) || (nc!=1 && nc!=4 && nc!=16) || !bct::span(labels,N+K,size) || !bct::span(ts,nt,size)) return 1;
  for (int i=0;i<9;++i) if (d[i]<0) return 1;
  if (d[0]>1 || d[1]>=64 || d[1]%16 || d[2]!=0 || d[3]!=16 || d[4]!=K ||
      !bct::span(d[5],d[6],size) || !bct::span(d[7],d[8],size) || uint64_t(d[5])+d[6]!=uint64_t(d[7])) return 1;
  v.data=data;v.size=size;v.K=K;v.nr=nr;v.nc=nc;v.labels=labels;v.ts=ts;v.nt=nt;
  v.mode=d[0];v.row0=d[1];v.ms=d[5];v.nm=d[6];v.rs=d[7];v.nd=d[8];
  if (v.mode) {
    if (v.nm!=32*K || v.nd) return v.error=1;
  } else {
    const uint32_t dir=4*(nr*nc+1);
    if (nt<dir || v.nm<260 || v.nd!=12*K) return v.error=1;
    v.es=v.ts+dir;v.entries=(nt-dir)/6;
    if (bct::u32(data+ts)!=0 || uint64_t(bct::u32(data+ts+4*nr*nc))*6!=nt-dir) return v.error=2;
    // Validate the complete sparse directory/CDF once, not a new probability model.
    for (uint32_t cx=0;cx<nr*nc;++cx) {
      const uint32_t a=bct::u32(data+ts+4*cx),b=bct::u32(data+ts+4*(cx+1));
      if (a>b || b>v.entries || b-a>1024) return v.error=2;
      uint32_t last=0,symbol=0;
      for (uint32_t j=a;j<b;++j) {
        const uint32_t sym=bct::u16(data+v.es+6*j),upper=bct::u32(data+v.es+6*j+2);
        if (sym>=1024 || (j>a && sym<=symbol) || upper<=last || upper>65536) return v.error=2;
        symbol=sym;last=upper;
      }
      if (a<b && last!=65536) return v.error=2;
    }
    if (bct::u32(data+v.ms)!=0 || bct::u32(data+v.ms+128)!=v.nm-132) return v.error=4;
    for (uint32_t lane=0;lane<32;++lane) {
      const uint32_t begin=bct::u32(data+v.ms+4*lane),end=bct::u32(data+v.ms+4*(lane+1));
      if (begin>end || end>v.nm-132 || end-begin<4) return v.error=4;
    }
  }
  for (uint32_t i=0;i<N+K;++i)
    if (data[labels+i]>=(i<N ? nr : nc)) return v.error=1;
  return v.error=0;
}
BCS_HD inline uint32_t init_lane(const PacketView& v,uint32_t lane,LaneCursor& c) {
  c=LaneCursor{};c.view=&v;c.lane=lane;
  if (v.error || lane>=32) return c.error=v.error ? v.error : 1;
  if (v.mode) return 0;
  c.pos=v.ms+132+bct::u32(v.data+v.ms+4*lane);
  c.stop=v.ms+132+bct::u32(v.data+v.ms+4*(lane+1));
  c.state=bct::u32(v.data+c.pos);c.pos+=4;
  if (c.state<bct::L || c.state>=uint64_t(bct::L)*256) return c.error=4;
  return 0;
}
BCS_HD inline uint32_t decode_next_slab(const PacketView& v,uint32_t lane,uint32_t chunk,
    LaneCursor& c,uint16_t* slab,uint64_t slab_words) {
  if (c.error) return c.error;
  if (v.error || c.view!=&v || lane>=32 || lane!=c.lane || chunk!=c.next_chunk || chunk>=v.K/CH ||
      !slab || slab_words!=uint64_t(N)*LDS) return c.error=1;
  if (v.mode) {
    for (uint32_t i=lane;i<2048;i+=32)
      slab[(v.row0+(i>>7))*LDS+(i&127)]=uint16_t(bct::u16(v.data+v.ms+2*(2048*chunk+i)));
    ++c.next_chunk;return 0;
  }
  if (c.pos>c.stop || c.stop>v.ms+v.nm || c.state<bct::L || c.state>=uint64_t(bct::L)*256) return c.error=4;
  for (uint32_t i=lane;i<2048;i+=32) {
    const uint32_t row=v.row0+(i>>7),col=chunk*CH+(i&127);
    const uint32_t r=v.data[v.labels+row],cc=v.data[v.labels+N+col];
    if (r>=v.nr || cc>=v.nc) return c.error=1;
    const uint32_t cx=r*v.nc+cc,a=bct::u32(v.data+v.ts+4*cx),b=bct::u32(v.data+v.ts+4*(cx+1));
    if (a>=b || b>v.entries || b-a>1024) return c.error=2;
    if (bct::u32(v.data+v.es+6*(b-1)+2)!=65536) return c.error=2;
    const uint32_t slot=c.state&65535;uint32_t lo=a,hi=b;
    while (lo<hi) {
      const uint32_t mid=lo+(hi-lo)/2;
      if (bct::u32(v.data+v.es+6*mid+2)<=slot) lo=mid+1; else hi=mid;
    }
    if (lo>=b) return c.error=2;
    const uint32_t sym=bct::u16(v.data+v.es+6*lo),upper=bct::u32(v.data+v.es+6*lo+2);
    const uint32_t lower=lo==a ? 0 : bct::u32(v.data+v.es+6*(lo-1)+2);
    if (sym>=1024 || upper>65536 || lower>slot || slot>=upper) return c.error=2;
    const uint64_t next=uint64_t(upper-lower)*(c.state>>16)+slot-lower;
    if (next>=uint64_t(bct::L)*256) return c.error=4;
    c.state=next;
    while (c.state<bct::L) {
      if (c.pos>=c.stop) return c.error=4;
      c.state=(c.state<<8)|v.data[c.pos++];
    }
    const uint32_t bit=6*(2048*chunk+i),byte=bit>>3,shift=bit&7;
    if (byte>=v.nd) return c.error=8;
    uint32_t residual=v.data[v.rs+byte]>>shift;
    if (shift>2) {
      if (byte+1>=v.nd) return c.error=8;
      residual|=uint32_t(v.data[v.rs+byte+1])<<(8-shift);
    }
    residual&=63;
    slab[row*LDS+(i&127)]=uint16_t((sym<<5)|(residual&31)|((residual>>5)<<15));
  }
  ++c.next_chunk;
  // There is intentionally no terminal-state check at intermediate CH boundaries.
  if (c.next_chunk==v.K/CH && (c.state!=bct::L || c.pos!=c.stop)) return c.error=16;
  return 0;
}
} // namespace bcs
#undef BCS_HD
