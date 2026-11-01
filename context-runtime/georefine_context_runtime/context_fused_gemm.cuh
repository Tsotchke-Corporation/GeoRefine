#pragma once
#include "context_tiled_packet.cuh"

// Bounded experimental geometry. Reuse the qualified integer decoder verbatim;
// one packet is exactly the next 16-row by 128-K slab consumed by MMA.
#ifdef __CUDACC__
#define BCF_HD __host__ __device__
#define BCF_INLINE __forceinline__
#else
#define BCF_HD
#define BCF_INLINE inline
#endif
namespace bcf {
constexpr uint32_t N = 64, CH = 128, LDS = CH+8, PR = 16;
BCF_HD BCF_INLINE uint32_t cdf_lut_lookup(const uint32_t* coarse, uint64_t coarse_count,
    const uint64_t* records, uint64_t record_count, const uint32_t* offsets,
    uint64_t offset_count, uint32_t nctx, uint32_t context, uint32_t slot,
    uint32_t* symbol, uint32_t* lower_out, uint32_t* frequency_out) {
  if (!coarse || !records || !offsets || !symbol || !lower_out || !frequency_out ||
      !nctx || nctx>1024 || context>=nctx || slot>=65536 ||
      coarse_count!=uint64_t(nctx)*256 || offset_count!=uint64_t(nctx)+1 ||
      record_count>uint64_t(nctx)*1024) return 2;
  const uint32_t a=offsets[context], b=offsets[context+1];
  if (a>=b || b>record_count || b-a>1024) return 2;
  const uint32_t packed=coarse[uint64_t(context)*256+(slot>>8)];
  uint32_t low=packed&65535, high=packed>>16;
  if (low>=high || high>b-a) return 2;
  while (low<high) {
    const uint32_t mid=low+((high-low)>>1);
    const uint64_t rec=records[a+mid];
    if ((rec>>43)!=0) return 2;
    const uint32_t lower=uint32_t(rec&65535), freq=uint32_t((rec>>16)&0x1ffff), sym=uint32_t((rec>>33)&0x3ff);
    if (!freq || lower+freq>65536 || sym>=1024) return 2;
    if (lower+freq<=slot) low=mid+1; else high=mid;
  }
  if (low>=b-a) return 2;
  const uint64_t rec=records[a+low];
  if ((rec>>43)!=0) return 2;
  const uint32_t lower=uint32_t(rec&65535), freq=uint32_t((rec>>16)&0x1ffff), sym=uint32_t((rec>>33)&0x3ff);
  if (!freq || lower+freq>65536 || sym>=1024 || lower>slot || slot>=lower+freq) return 2;
  *symbol=sym; *lower_out=lower; *frequency_out=freq;
  return 0;
}
BCF_HD inline uint32_t decode_slab_lane(const uint8_t* data, uint64_t size,
    const int64_t* descriptor, uint32_t K, uint32_t nr, uint32_t nc,
    uint64_t labels, uint64_t ts, uint64_t nt, uint32_t chunk,
    uint32_t packet_row, uint32_t lane, uint16_t* slab) {
  if ((K != 512 && K != 1024) || chunk >= K/CH || packet_row >= N/PR ||
      descriptor[1] != packet_row*PR || descriptor[2] != chunk*CH ||
      descriptor[3] != PR || descriptor[4] != CH) return 1;
  bct::OutputView output{slab,0,chunk*CH,N,CH,N*LDS,LDS};
  return bct::decode_lane(data,size,descriptor,N,K,nr,nc,labels,ts,nt,lane,output);
}

// Opt-in fixed geometry decoder. Packet framing, states and CDF probabilities
// are identical to decode_lane. Bounded offsets narrow only AFTER span checks;
// all per-word coordinates are uint32 shifts/masks, never runtime division.
BCF_HD BCF_INLINE uint32_t decode_slab_lane_opt(const uint8_t* data, uint64_t size,
    const int64_t* d, uint32_t K, uint32_t nr, uint32_t nc,
    uint64_t labels, uint64_t ts, uint64_t nt, uint32_t chunk,
    uint32_t packet_row, uint32_t lane, uint16_t* slab) {
  if (!data || !d || !slab || size>1048576 || (K!=512 && K!=1024) ||
      chunk>=K/CH || packet_row>=4 || lane>=32 || !nr || !nc ||
      uint64_t(nr)*nc>1024 || !bct::span(labels,N+K,size) ||
      !bct::span(ts,nt,size)) return 1;
  for (int k=0;k<9;++k) if (d[k]<0) return 1;
  const uint32_t row0=packet_row*PR, col0=chunk*CH;
  if (d[0]>1 || d[1]!=row0 || d[2]!=col0 || d[3]!=PR || d[4]!=CH ||
      !bct::span(d[5],d[6],size) || !bct::span(d[7],d[8],size) ||
      uint64_t(d[5])+d[6]!=uint64_t(d[7])) return 1;
  const uint32_t ms=d[5], nm=d[6], rs=d[7], nd=d[8];
  if (d[0]) {
    if (nd || nm!=2*PR*CH) return 1;
    // Every 4-word group and every padded row is 8-byte aligned when these
    // bases are aligned. BCT2 framing does not guarantee this: retain fallback.
    if (((reinterpret_cast<uintptr_t>(data+ms)|reinterpret_cast<uintptr_t>(slab))&7)==0) {
      for (uint32_t i=lane*4;i<PR*CH;i+=128)
        *reinterpret_cast<uint64_t*>(slab+(row0+(i>>7))*LDS+(i&127))=
            *reinterpret_cast<const uint64_t*>(data+ms+2*i);
    } else if (((reinterpret_cast<uintptr_t>(data+ms)|reinterpret_cast<uintptr_t>(slab))&3)==0) {
      // Common BCT2 raw payloads are 4-byte aligned but not 8-byte aligned.
      // Two-word groups never cross a row; LDS preserves their alignment.
      for (uint32_t i=lane*2;i<PR*CH;i+=64)
        *reinterpret_cast<uint32_t*>(slab+(row0+(i>>7))*LDS+(i&127))=
            *reinterpret_cast<const uint32_t*>(data+ms+2*i);
    } else {
      for (uint32_t i=lane;i<PR*CH;i+=32)
        slab[(row0+(i>>7))*LDS+(i&127)]=uint16_t(bct::u16(data+ms+2*i));
    }
    return 0;
  }
  const uint32_t nctx=nr*nc, dir_bytes=4*(nctx+1), es=uint32_t(ts)+dir_bytes;
  if (nt<dir_bytes || nm<260 || nd!=1536) return 1;
  if (bct::u32(data+ts)!=0 || uint64_t(bct::u32(data+ts+4*nctx))*6!=nt-dir_bytes) return 2;
  if (bct::u32(data+ms)!=0 || bct::u32(data+ms+128)!=nm-132) return 4;
  const uint32_t begin=bct::u32(data+ms+4*lane), end=bct::u32(data+ms+4*(lane+1));
  if (begin>end || end>nm-132 || end-begin<4) return 4;
  uint32_t pos=ms+132+begin, stop=ms+132+end;
  uint32_t state=bct::u32(data+pos); pos+=4;
  if (state<bct::L || state>=uint64_t(bct::L)*256) return 4;
  const uint32_t total_entries=(nt-dir_bytes)/6;
  for (uint32_t i=lane;i<PR*CH;i+=32) {
    const uint32_t row=row0+(i>>7), col=i&127;
    uint32_t r=data[labels+row], c=data[labels+N+col0+col];
    if (r>=nr || c>=nc) return 1;
    const uint32_t cx=r*nc+c, a=bct::u32(data+ts+4*cx), b=bct::u32(data+ts+4*(cx+1));
    if (a>=b || b>total_entries || b-a>1024) return 2;
    if (bct::u32(data+es+6*(b-1)+2)!=65536) return 2;
    const uint32_t slot=state&65535;
    uint32_t low=a, high=b;
    while (low<high) {
      const uint32_t mid=low+((high-low)>>1), upper=bct::u32(data+es+6*mid+2);
      if (upper<=slot) low=mid+1; else high=mid;
    }
    if (low>=b) return 2;
    const uint32_t sym=bct::u16(data+es+6*low), upper=bct::u32(data+es+6*low+2);
    const uint32_t lower=low==a ? 0 : bct::u32(data+es+6*(low-1)+2);
    if (sym>=1024 || upper>65536 || lower>slot || slot>=upper) return 2;
    const uint64_t next=uint64_t(upper-lower)*(state>>16)+slot-lower;
    if (next>=uint64_t(bct::L)*256) return 4;
    state=uint32_t(next);
    while (state<bct::L) {
      if (pos>=stop) return 4;
      state=(state<<8)|data[pos++];
    }
    const uint32_t bit=6*i, byte=bit>>3, shift=bit&7;
    if (byte>=nd) return 8;
    uint32_t residual=data[rs+byte]>>shift;
    if (shift>2) {
      if (byte+1>=nd) return 8;
      residual|=uint32_t(data[rs+byte+1])<<(8-shift);
    }
    residual&=63;
    slab[row*LDS+col]=uint16_t((sym<<5)|(residual&31)|((residual>>5)<<15));
  }
  return (state==bct::L && pos==stop) ? 0 : 16;
}

// Separate opt-in path using coarse bounds and packed CDF records derived from
// the exact packet table bytes. Every selected record is still checked before
// it participates in integer rANS decoding.
BCF_HD BCF_INLINE uint32_t decode_slab_lane_lut(const uint8_t* data, uint64_t size,
    const int64_t* d, uint32_t K, uint32_t nr, uint32_t nc,
    uint64_t labels, uint64_t ts, uint64_t nt, uint32_t chunk,
    uint32_t packet_row, uint32_t lane, uint16_t* slab,
    const uint32_t* coarse, uint64_t coarse_count,
    const uint64_t* records, uint64_t record_count,
    const uint32_t* offsets, uint64_t offset_count) {
  if (!data || !d || !slab || size>1048576 || (K!=512 && K!=1024) ||
      chunk>=K/CH || packet_row>=4 || lane>=32 || !nr || !nc ||
      uint64_t(nr)*nc>1024 || !bct::span(labels,N+K,size) || !bct::span(ts,nt,size)) return 1;
  for (int k=0;k<9;++k) if (d[k]<0) return 1;
  const uint32_t row0=packet_row*PR, col0=chunk*CH;
  if (d[0]>1 || d[1]!=row0 || d[2]!=col0 || d[3]!=PR || d[4]!=CH ||
      !bct::span(d[5],d[6],size) || !bct::span(d[7],d[8],size) || uint64_t(d[5])+d[6]!=uint64_t(d[7])) return 1;
  const uint32_t ms=d[5], nm=d[6], rs=d[7], nd=d[8];
  if (d[0]) {
    if (nd || nm!=2*PR*CH) return 1;
    if (((reinterpret_cast<uintptr_t>(data+ms)|reinterpret_cast<uintptr_t>(slab))&7)==0) {
      for (uint32_t i=lane*4;i<PR*CH;i+=128)
        *reinterpret_cast<uint64_t*>(slab+(row0+(i>>7))*LDS+(i&127))=
            *reinterpret_cast<const uint64_t*>(data+ms+2*i);
    } else if (((reinterpret_cast<uintptr_t>(data+ms)|reinterpret_cast<uintptr_t>(slab))&3)==0) {
      for (uint32_t i=lane*2;i<PR*CH;i+=64)
        *reinterpret_cast<uint32_t*>(slab+(row0+(i>>7))*LDS+(i&127))=
            *reinterpret_cast<const uint32_t*>(data+ms+2*i);
    } else {
      for (uint32_t i=lane;i<PR*CH;i+=32)
        slab[(row0+(i>>7))*LDS+(i&127)]=uint16_t(bct::u16(data+ms+2*i));
    }
    return 0;
  }
  const uint32_t nctx=nr*nc, dir_bytes=4*(nctx+1), es=uint32_t(ts)+dir_bytes;
  if (nt<dir_bytes || nm<260 || nd!=1536 || !coarse || !offsets ||
      coarse_count!=uint64_t(nctx)*256 || offset_count!=uint64_t(nctx)+1 ||
      record_count>uint64_t(nctx)*1024) return 1;
  const uint32_t packet_entries=bct::u32(data+ts+4*nctx);
  if (bct::u32(data+ts)!=0 || uint64_t(packet_entries)*6!=nt-dir_bytes ||
      packet_entries!=record_count || offsets[0]!=0 || offsets[nctx]!=record_count) return 2;
  if (bct::u32(data+ms)!=0 || bct::u32(data+ms+128)!=nm-132) return 4;
  const uint32_t begin=bct::u32(data+ms+4*lane), end=bct::u32(data+ms+4*(lane+1));
  if (begin>end || end>nm-132 || end-begin<4) return 4;
  uint32_t pos=ms+132+begin, stop=ms+132+end;
  uint32_t state=bct::u32(data+pos); pos+=4;
  if (state<bct::L || state>=uint64_t(bct::L)*256) return 4;
  for (uint32_t i=lane;i<PR*CH;i+=32) {
    const uint32_t row=row0+(i>>7), col=i&127;
    const uint32_t r=data[labels+row], c=data[labels+N+col0+col];
    if (r>=nr || c>=nc) return 1;
    const uint32_t cx=r*nc+c, slot=state&65535;
    uint32_t sym, lower, freq;
    if (cdf_lut_lookup(coarse,coarse_count,records,record_count,offsets,offset_count,
                       nctx,cx,slot,&sym,&lower,&freq)) return 2;
    const uint64_t next=uint64_t(freq)*(state>>16)+slot-lower;
    if (next>=uint64_t(bct::L)*256) return 4;
    state=uint32_t(next);
    while (state<bct::L) {
      if (pos>=stop) return 4;
      state=(state<<8)|data[pos++];
    }
    const uint32_t bit=6*i, byte=bit>>3, shift=bit&7;
    if (byte>=nd) return 8;
    uint32_t residual=data[rs+byte]>>shift;
    if (shift>2) {
      if (byte+1>=nd) return 8;
      residual|=uint32_t(data[rs+byte+1])<<(8-shift);
    }
    residual&=63;
    slab[row*LDS+col]=uint16_t((sym<<5)|(residual&31)|((residual>>5)<<15));
  }
  return (state==bct::L && pos==stop) ? 0 : 16;
}

// Independent resident panels share geometry, not stream state. Descriptor
// offsets remain panel-local; a malformed packet cannot read a neighbour blob.
BCF_HD BCF_INLINE uint32_t decode_panel_slab_lane(const uint8_t* data, uint64_t size,
    const int64_t* descriptors, uint64_t descriptor_count, uint32_t panels,
    uint32_t panel, uint64_t panel_bytes, uint32_t K, uint32_t nr, uint32_t nc,
    uint64_t labels, uint64_t ts, uint64_t nt, uint32_t chunk,
    uint32_t packet_row, uint32_t lane, uint16_t* slab) {
  if (!data || !descriptors || panels<1 || panels>1024 || panel>=panels ||
      !panel_bytes || panel_bytes>1048576 || size!=uint64_t(panels)*panel_bytes ||
      (K!=512 && K!=1024) || chunk>=K/CH || packet_row>=4 ||
      descriptor_count!=uint64_t(panels)*4*(K/CH)*9) return 1;
  const uint64_t offset=(uint64_t(panel)*4*(K/CH)+4*chunk+packet_row)*9;
  return decode_slab_lane_opt(data+uint64_t(panel)*panel_bytes,panel_bytes,
      descriptors+offset,K,nr,nc,labels,ts,nt,chunk,packet_row,lane,slab);
}

BCF_HD BCF_INLINE uint32_t decode_panel_slab_lane_lut(const uint8_t* data, uint64_t size,
    const int64_t* descriptors, uint64_t descriptor_count, uint32_t panels,
    uint32_t panel, uint64_t panel_bytes, uint32_t K, uint32_t nr, uint32_t nc,
    uint64_t labels, uint64_t ts, uint64_t nt, uint32_t chunk,
    uint32_t packet_row, uint32_t lane, uint16_t* slab,
    const uint32_t* coarse, uint64_t coarse_count, const uint64_t* records,
    uint64_t record_count, const uint32_t* offsets, uint64_t offset_count) {
  if (!data || !descriptors || panels<1 || panels>1024 || panel>=panels ||
      !panel_bytes || panel_bytes>1048576 || size!=uint64_t(panels)*panel_bytes ||
      (K!=512 && K!=1024) || chunk>=K/CH || packet_row>=4 ||
      descriptor_count!=uint64_t(panels)*4*(K/CH)*9) return 1;
  const uint64_t offset=(uint64_t(panel)*4*(K/CH)+4*chunk+packet_row)*9;
  return decode_slab_lane_lut(data+uint64_t(panel)*panel_bytes,panel_bytes,
      descriptors+offset,K,nr,nc,labels,ts,nt,chunk,packet_row,lane,slab,
      coarse,coarse_count,records,record_count,offsets,offset_count);
}
} // namespace bcf
#undef BCF_HD
#undef BCF_INLINE
