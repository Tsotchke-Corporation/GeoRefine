#pragma once
#include <stdint.h>
#include <cstring>
#ifdef __CUDACC__
#define BCA_HD __host__ __device__
#else
#define BCA_HD
#endif
namespace bca {
constexpr uint32_t N=64,CH=128,LDS=136,L=65536;
BCA_HD inline uint32_t u16(const uint8_t* p){return uint32_t(p[0])|(uint32_t(p[1])<<8);}
BCA_HD inline uint32_t u32(const uint8_t* p){return u16(p)|(u16(p+2)<<16);}
// Only the derived metadata directory uses this aligned load. Packet initial
// states may start at a two-byte offset and retain the bytewise u32 helper.
BCA_HD inline uint32_t context_u32_aligned(const uint8_t* p){
#ifdef __CUDA_ARCH__
  return *reinterpret_cast<const uint32_t*>(p); // ordinary load; no cache-policy change
#else
  uint32_t value;std::memcpy(&value,p,sizeof(value));return value;
#endif
}
BCA_HD inline bool span(uint64_t p,uint64_t n,uint64_t size){return p<=size && n<=size-p;}
struct PacketView {
  const uint8_t *data=nullptr,*meta=nullptr;
  uint64_t size=0,ms=0,nm=0,rs=0,nd=0;
  uint32_t K=0,nr=0,nc=0,buckets=0,offsets=0,shifts=0,labels=0;
  uint32_t row0=0,mode=0,error=1,label_rows=64,row_base=0;
};
struct LaneCursor {
  uint64_t pos=0,stop=0;
  uint32_t state=0,next_chunk=0,error=0,lane=0;
  const PacketView* view=nullptr;
};
// Metadata was exhaustively validated before private plan sealing. This bounded
// per-packet setup never scans labels, contexts, or all records.
template<bool AlignedOffsets=false>
BCA_HD inline uint32_t prepare_packet(const uint8_t* data,uint64_t size,const int64_t* d,
    const uint8_t* meta,uint64_t meta_size,uint32_t buckets,uint32_t K,
    uint32_t nr,uint32_t nc,PacketView& v,uint32_t label_rows=64,uint32_t row_base=0) {
  v=PacketView{};
  if constexpr(AlignedOffsets){if(reinterpret_cast<uintptr_t>(meta)%alignof(uint32_t))return 1;}
  if (!data || !d || !meta || size>uint64_t(4096)*32*1048576 || label_rows<64 || label_rows>4096*64 || label_rows%64 || row_base%64 || !span(row_base,64,label_rows) || K<128 || K>32768 || K%128 ||
      (nr!=1 && nr!=4 && nr!=16 && nr!=64) || (nc!=1 && nc!=4 && nc!=16) || buckets>nr*nc*1024 ||
      meta_size!=uint64_t(buckets)*16+4*(nr*nc+1)+nr*nc+label_rows+K+96) return 1;
  for(int j=0;j<9;++j)if(d[j]<0)return 1;
  if(d[0]>1 || d[1]>=N || d[1]%16 || d[2] || d[3]!=16 || d[4]!=K ||
      !span(d[5],d[6],size) || !span(d[7],d[8],size) || uint64_t(d[5])+d[6]!=uint64_t(d[7]))return 1;
  v.data=data;v.meta=meta;v.size=size;v.label_rows=label_rows;v.row_base=row_base;v.K=K;v.nr=nr;v.nc=nc;v.buckets=buckets;
  v.offsets=16*buckets;v.shifts=v.offsets+4*(nr*nc+1);v.labels=v.shifts+nr*nc;
  v.mode=d[0];v.row0=d[1];v.ms=d[5];v.nm=d[6];v.rs=d[7];v.nd=d[8];
  if(v.mode ? (v.nm!=32*K || v.nd) : (v.nm<128 || v.nm%2 || v.nm>128+32*K || v.nd!=12*K))return v.error=1;
  return v.error=0;
}
BCA_HD inline uint32_t init_lane(const PacketView& v,uint32_t lane,LaneCursor& c){
  c=LaneCursor{};c.lane=lane;c.view=&v;
  if(v.error || lane>=32)return c.error=v.error?v.error:1;
  c.pos=v.ms+128;c.stop=v.ms+v.nm;
  if(!v.mode){c.state=u32(v.data+v.ms+4*lane);if(c.state<L)return c.error=4;}
  return 0;
}
BCA_HD inline uint32_t begin_slab(const PacketView& v,uint32_t lane,uint32_t chunk,LaneCursor& c,uint16_t* slab,uint64_t words){
  if(c.error)return c.error;
  if(v.error || c.view!=&v || lane>=32 || c.lane!=lane || chunk!=c.next_chunk || chunk>=v.K/CH || !slab || words!=N*LDS)return c.error=1;
  if(!v.mode && (c.pos>c.stop || c.stop>v.ms+v.nm || c.state<L))return c.error=4;
  return 0;
}
template<bool ScalarRecords=false,bool AlignedOffsets=false>
BCA_HD inline uint32_t next_symbol(const PacketView& v,uint32_t row,uint32_t col,LaneCursor& c,uint32_t& symbol){
  const uint32_t r=v.meta[v.labels+v.row_base+row],cc=v.meta[v.labels+v.label_rows+col];
  if(r>=v.nr || cc>=v.nc)return c.error=1;
  const uint32_t cx=r*v.nc+cc;uint32_t first,end;
  if constexpr(AlignedOffsets){
    // prepare_packet/private plan checked base alignment. offsets=16*buckets
    // and each directory stride is4, so both addresses remain four-byte aligned.
    first=context_u32_aligned(v.meta+v.offsets+4*cx);
    end=context_u32_aligned(v.meta+v.offsets+4*(cx+1));
  } else {
    first=u32(v.meta+v.offsets+4*cx);end=u32(v.meta+v.offsets+4*(cx+1));
  }
  const uint32_t h=v.meta[v.shifts+cx];
  if(h>10 || first>=end || end>v.buckets || end-first!=(1u<<h))return c.error=2;
  const uint32_t slot=c.state&65535,width=65536>>h,within=slot&(width-1),index=first+(slot>>(16-h));
  if(index>=end)return c.error=2;
  if constexpr(ScalarRecords){
    // Keep all four packed words in named registers. In the baseline, the
    // dynamic record[primary?0:1] subscript materializes the symbol pair on
    // the local stack (STL + LDL.U16 once per decoded symbol on sm_120).
#ifdef __CUDA_ARCH__
    const uint4 packed=__ldg(reinterpret_cast<const uint4*>(v.meta+16*index));
    const uint32_t a=packed.x,b=packed.y,d=packed.z,e=packed.w;
#else
    const uint32_t a=u32(v.meta+16*index),b=u32(v.meta+16*index+4),
                   d=u32(v.meta+16*index+8),e=u32(v.meta+16*index+12);
#endif
    const uint32_t primary_symbol=a&65535,alias_symbol=a>>16;
    const uint32_t cut16=b&65535,primary_base=b>>16,alias_base=d&65535;
    const uint32_t primary16=d>>16,alias16=e&65535,flags=e>>16;
    const uint32_t cut=flags&1?65536:cut16;
    if(flags&~7u || cut>width || ((flags&1) && cut16) || ((flags&2) && primary16) || ((flags&4) && alias16))return c.error=2;
    const bool primary=within<cut;
    const uint32_t freq=primary?(flags&2?65536:primary16):(flags&4?65536:alias16);
    const uint32_t rank=primary?primary_base+within:alias_base+within-cut;
    symbol=primary?primary_symbol:alias_symbol;
    if(symbol>=1024 || !freq || rank>=freq)return c.error=2;
    const uint64_t next=uint64_t(freq)*(c.state>>16)+rank;
    if(next>0xffffffffull)return c.error=4;
    c.state=uint32_t(next);return 0;
  } else {
  alignas(16) uint16_t record[8];
#ifdef __CUDA_ARCH__
  *reinterpret_cast<uint4*>(record)=__ldg(reinterpret_cast<const uint4*>(v.meta+16*index));
#else
  for(int j=0;j<8;++j)record[j]=u16(v.meta+16*index+2*j);
#endif
  const uint32_t flags=record[7],cut=flags&1?65536:record[2];
  if(flags&~7u || cut>width || ((flags&1) && record[2]) || ((flags&2) && record[5]) || ((flags&4) && record[6]))return c.error=2;
  const bool primary=within<cut;
  const uint32_t freq=primary?(flags&2?65536:record[5]):(flags&4?65536:record[6]);
  const uint32_t rank=primary?record[3]+within:record[4]+within-cut;
  symbol=record[primary?0:1];
  if(symbol>=1024 || !freq || rank>=freq)return c.error=2;
  const uint64_t next=uint64_t(freq)*(c.state>>16)+rank;
  if(next>0xffffffffull)return c.error=4;
  c.state=uint32_t(next);return 0;
  }
}
BCA_HD inline uint32_t write_symbol(const PacketView& v,uint32_t chunk,uint32_t i,uint32_t symbol,LaneCursor& c,uint16_t* slab){
  const uint32_t bit=6*(2048*chunk+i),byte=bit>>3,shift=bit&7;
  if(byte>=v.nd || (shift>2 && byte+1>=v.nd))return c.error=8;
  uint32_t residual=v.data[v.rs+byte]>>shift;
  if(shift>2)residual|=uint32_t(v.data[v.rs+byte+1])<<(8-shift);
  residual&=63;
  slab[(v.row0+(i>>7))*LDS+(i&127)]=uint16_t((symbol<<5)|(residual&31)|((residual>>5)<<15));return 0;
}
BCA_HD inline uint32_t end_slab(const PacketView& v,LaneCursor& c){
  ++c.next_chunk;
  if(!v.mode && c.next_chunk==v.K/CH && (c.state!=L || c.pos!=c.stop))return c.error=16;
  return c.error;
}
#ifndef __CUDACC__
// CPU harness uses the same per-access decoder, exercising the common cursor
// and lane ordering as a 32-lane group, never independent lane streams.
template<bool ScalarRecords=false,bool AlignedOffsets=false>
inline uint32_t decode_group(const PacketView& v,uint32_t chunk,LaneCursor* c,uint16_t* slab,uint64_t words){
  uint32_t error=0;
  for(uint32_t l=0;l<32;++l)error|=begin_slab(v,l,chunk,c[l],slab,words);
  if(error)return error;
  for(uint32_t t=0;t<64;++t){
    uint32_t symbols[32],mask=0;
    for(uint32_t l=0;l<32;++l){
      const uint32_t i=32*t+l;
      if(v.mode)slab[(v.row0+(i>>7))*LDS+(i&127)]=u16(v.data+v.ms+2*(2048*chunk+i));
      else{error|=next_symbol<ScalarRecords,AlignedOffsets>(v,v.row0+(i>>7),chunk*CH+(i&127),c[l],symbols[l]);if(c[l].state<L)mask|=1u<<l;}
    }
    if(error)return error;
    if(!v.mode){
      uint64_t pos=c[0].pos;
      for(uint32_t l=0;l<32;++l){
        if(c[l].pos!=c[0].pos)return c[l].error=1;
        if(mask&(1u<<l)){
          if(!span(pos,2,c[l].stop))return c[l].error=4;
          c[l].state=(c[l].state<<16)|u16(v.data+pos);pos+=2;
          if(c[l].state<L)return c[l].error=4;
        }
        error|=write_symbol(v,chunk,32*t+l,symbols[l],c[l],slab);
      }
      for(uint32_t l=0;l<32;++l)c[l].pos=pos;
    }
  }
  for(uint32_t l=0;l<32;++l)error|=end_slab(v,c[l]);return error;
}
#else
template<bool ScalarRecords=false,bool AlignedOffsets=false>
__device__ inline uint32_t decode_next_slab(const PacketView& v,uint32_t lane,uint32_t chunk,LaneCursor& c,uint16_t* slab,uint64_t words){
  begin_slab(v,lane,chunk,c,slab,words);
  if(__any_sync(0xffffffffu,c.error!=0))return c.error?c.error:32;
  for(uint32_t t=0;t<64;++t){
    const uint32_t i=32*t+lane;uint32_t symbol=0;
    if(v.mode)slab[(v.row0+(i>>7))*LDS+(i&127)]=u16(v.data+v.ms+2*(2048*chunk+i));
    else{
      next_symbol<ScalarRecords,AlignedOffsets>(v,v.row0+(i>>7),chunk*CH+(i&127),c,symbol);
      if(__any_sync(0xffffffffu,c.error!=0))return c.error?c.error:32;
      const uint32_t mask=__ballot_sync(0xffffffffu,c.state<L);
      const uint64_t base=__shfl_sync(0xffffffffu,static_cast<unsigned long long>(c.pos),0);
      const uint32_t count=__popc(mask);
      if(c.pos!=base || !span(base,2*count,c.stop))c.error=4;
      if(__any_sync(0xffffffffu,c.error!=0))return c.error?c.error:32;
      if(mask&(1u<<lane)){
        const uint32_t rank=__popc(mask&((1u<<lane)-1u));
        c.state=(c.state<<16)|__ldg(reinterpret_cast<const uint16_t*>(v.data+base+2*rank));
        if(c.state<L)c.error=4;
      }
      c.pos=base+2*count;
      if(!c.error)write_symbol(v,chunk,i,symbol,c,slab);
      if(__any_sync(0xffffffffu,c.error!=0))return c.error?c.error:32;
    }
  }
  end_slab(v,c);
  return c.error;
}
#endif
// Explicit trusted-plan consumer. Only factory-owned immutable metadata is in
// this domain; arbitrary post-seal metadata/register corruption is not covered.
// Generic decoders above remain the checked raw-harness/default paths.
#ifdef __CUDACC__
#define BCA_FORCE __host__ __device__ __forceinline__
#else
#define BCA_FORCE inline
#endif
struct TrustedPacket {
  const uint8_t *main=nullptr,*residual=nullptr,*rows=nullptr,*cols=nullptr;
  const uint8_t *records=nullptr,*shifts=nullptr;
  const uint32_t* offsets=nullptr;
  uint32_t main_bytes=0,row0=0,nc=0,mode=0;
};
struct TrustedCursor {uint32_t state=0,pos=0,error=0;};
constexpr uint32_t MAIN_FIFO_WINDOW=512,MAIN_FIFO_BYTES=576,RAW6_STAGE_BYTES=1536;
struct TrustedFifo {uint16_t* words=nullptr;uint32_t base=128;bool valid=false;};
struct TrustedContext {uint32_t first,shift,mask;};
BCA_FORCE uint32_t init_trusted(const PacketView& v,uint32_t lane,TrustedPacket& t,TrustedCursor& c){
  t=TrustedPacket{};c=TrustedCursor{};
  if(v.error || lane>=32)return c.error=v.error?v.error:1;
  // Initial geometry/spans were checked by prepare_packet. Main lengths are
  // <=128+32*K<=1048704, so the relative cursor fits u32 even for >4GB bases.
  t.main=v.data+v.ms;t.residual=v.data+v.rs;t.records=v.meta;
  t.rows=v.meta+v.labels+v.row_base+v.row0;t.cols=v.meta+v.labels+v.label_rows;
  t.offsets=reinterpret_cast<const uint32_t*>(v.meta+v.offsets);t.shifts=v.meta+v.shifts;
  t.main_bytes=uint32_t(v.nm);t.row0=v.row0;t.nc=v.nc;t.mode=v.mode;
  if(reinterpret_cast<uintptr_t>(t.records)%16 || reinterpret_cast<uintptr_t>(t.main)%2)return c.error=1;
  if(!t.mode){c.pos=128;c.state=u32(t.main+4*lane);if(c.state<L)return c.error=4;}
  return 0;
}
BCA_FORCE TrustedContext trusted_context(const TrustedPacket& t,uint32_t cx){
  const uint32_t first=context_u32_aligned(reinterpret_cast<const uint8_t*>(t.offsets+cx));
  const uint32_t shift=16-t.shifts[cx];
  return TrustedContext{first,shift,(1u<<shift)-1u};
}
BCA_FORCE uint32_t trusted_symbol(const TrustedPacket& t,const TrustedContext& cx,TrustedCursor& c){
  const uint32_t slot=c.state&65535,within=slot&cx.mask,index=cx.first+(slot>>cx.shift);
#ifdef __CUDA_ARCH__
  const uint4 packed=__ldg(reinterpret_cast<const uint4*>(t.records+16*index));
  const uint32_t a=packed.x,b=packed.y,d=packed.z,e=packed.w;
#else
  const uint32_t a=u32(t.records+16*index),b=u32(t.records+16*index+4),
                 d=u32(t.records+16*index+8),e=u32(t.records+16*index+12);
#endif
  const uint32_t flags=e>>16,cut=flags&1?65536:b&65535;
  const bool primary=within<cut;
  const uint32_t freq=primary?(flags&2?65536:d>>16):(flags&4?65536:e&65535);
  const uint32_t rank=primary?(b>>16)+within:(d&65535)+within-cut;
  // Immutable complete alias-map proof gives1<=freq<=65536 andrank<freq.
  // With normalized u32 state:1<=freq*(state>>16)+rank<=UINT32_MAX.
  c.state=freq*(c.state>>16)+rank;
  return primary?a&65535:a>>16;
}
// Cooperative loaders never widen past the two-byte source alignment. FIFO
// lookahead outside the validated main span is explicitly initialized to zero;
// the real dynamic refill-span check still runs before any buffered consumption.
BCA_FORCE uint16_t fifo_prefetch_word(const TrustedPacket& t,uint32_t base,uint32_t index){
  const uint32_t at=base+2*index;
  if(at>t.main_bytes || t.main_bytes-at<2)return 0;
#ifdef __CUDA_ARCH__
  return __ldg(reinterpret_cast<const uint16_t*>(t.main+at));
#else
  return uint16_t(u16(t.main+at));
#endif
}
BCA_FORCE uint16_t raw6_prefetch_word(const TrustedPacket& t,uint32_t chunk,uint32_t index){
  const uint8_t* p=t.residual+RAW6_STAGE_BYTES*chunk+2*index;
#ifdef __CUDA_ARCH__
  return *reinterpret_cast<const uint16_t*>(p); // ordinary cache policy, like raw6 byte reads
#else
  return uint16_t(u16(p));
#endif
}
template<bool StageRaw6=false>
BCA_FORCE void trusted_store(const TrustedPacket& t,uint32_t chunk,uint32_t i,uint32_t symbol,uint16_t* slab,const uint8_t* staged=nullptr){
  const uint32_t bit=6*(StageRaw6?i:2048*chunk+i),byte=bit>>3,shift=bit&7;
  const uint8_t* data=StageRaw6?staged:t.residual;
  uint32_t residual=data[byte]>>shift;
  if(shift>2)residual|=uint32_t(data[byte+1])<<(8-shift);
  residual&=63;
  slab[(t.row0+(i>>7))*LDS+(i&127)]=uint16_t((symbol<<5)|(residual&31)|((residual>>5)<<15));
}
BCA_FORCE uint32_t trusted_finish(const TrustedPacket& t,bool last,TrustedCursor& c){
  if(last && !t.mode && (c.state!=L || c.pos!=t.main_bytes))c.error=16;
  return c.error;
}
#ifndef __CUDACC__
// CPU collective simulator shares actual symbol/context/store helpers. Its
// inputs must be admitted by prepare/init and the canonical private factory.
template<bool MainFifo=false,bool StageRaw6=false>
inline uint32_t decode_trusted_group(const TrustedPacket& t,uint32_t chunk,bool last,
    TrustedCursor* cursors,uint16_t* slab,TrustedFifo* fifo=nullptr,uint8_t* staged=nullptr){
  static_assert(!StageRaw6 || MainFifo,"raw6 staging is a separate FIFO extension");
  if(t.mode){
    for(uint32_t i=0;i<2048;++i)slab[(t.row0+(i>>7))*LDS+(i&127)]=u16(t.main+2*(2048*chunk+i));
    return 0;
  }
  if constexpr(StageRaw6){
    for(uint32_t index=0;index<RAW6_STAGE_BYTES/2;++index)
      reinterpret_cast<uint16_t*>(staged)[index]=raw6_prefetch_word(t,chunk,index);
  }
  uint32_t columns[32][4];
  for(uint32_t lane=0;lane<32;++lane)for(uint32_t g=0;g<4;++g)columns[lane][g]=t.cols[chunk*CH+32*g+lane];
  for(uint32_t row=0;row<16;++row){
    const uint32_t base=uint32_t(t.rows[row])*t.nc;
    TrustedContext contexts[32][4];
    for(uint32_t lane=0;lane<32;++lane)for(uint32_t g=0;g<4;++g)contexts[lane][g]=trusted_context(t,base+columns[lane][g]);
    for(uint32_t g=0;g<4;++g){
      uint32_t symbols[32],mask=0;
      for(uint32_t lane=0;lane<32;++lane){
        symbols[lane]=trusted_symbol(t,contexts[lane][g],cursors[lane]);
        if(cursors[lane].state<L)mask|=1u<<lane;
      }
      const uint32_t count=__builtin_popcount(mask),pos=cursors[0].pos,bytes=2*count;
      if(pos>t.main_bytes || bytes>t.main_bytes-pos){
        for(uint32_t lane=0;lane<32;++lane)cursors[lane].error=4;
        return 4;
      }
      if constexpr(MainFifo){
        if(bytes && (!fifo->valid || pos-fifo->base>=MAIN_FIFO_WINDOW)){
          fifo->base=128+((pos-128)&~(MAIN_FIFO_WINDOW-1));
          for(uint32_t index=0;index<MAIN_FIFO_BYTES/2;++index)fifo->words[index]=fifo_prefetch_word(t,fifo->base,index);
          fifo->valid=true;
        }
      }
      uint32_t next=pos;
      for(uint32_t lane=0;lane<32;++lane){
        if(mask&(1u<<lane)){
          uint16_t word;
          if constexpr(MainFifo)word=fifo->words[(next-fifo->base)/2];else word=uint16_t(u16(t.main+next));
          cursors[lane].state=(cursors[lane].state<<16)|word;next+=2;
        }
        cursors[lane].pos=pos+bytes;
        trusted_store<StageRaw6>(t,chunk,128*row+32*g+lane,symbols[lane],slab,staged);
      }
    }
  }
  uint32_t error=0;for(uint32_t lane=0;lane<32;++lane)error|=trusted_finish(t,last,cursors[lane]);return error;
}
#else
template<bool MainFifo=false,bool StageRaw6=false>
__device__ __forceinline__ uint32_t trusted_step(const TrustedPacket& t,const TrustedContext& cx,
    uint32_t lane,uint32_t chunk,uint32_t i,TrustedCursor& c,uint16_t* slab,TrustedFifo* fifo,const uint8_t* staged){
  const uint32_t symbol=trusted_symbol(t,cx,c);
  const uint32_t mask=__ballot_sync(0xffffffffu,c.state<L),bytes=2*__popc(mask);
  // Every lane starts at128 and adds the same ballot count. This branch is
  // uniform; it precedes every refill load. No per-lane early exit/vote/shuffle.
  if(c.pos>t.main_bytes || bytes>t.main_bytes-c.pos)return c.error=4;
  if constexpr(MainFifo){
    if(bytes && (!fifo->valid || c.pos-fifo->base>=MAIN_FIFO_WINDOW)){
      const uint32_t base=128+((c.pos-128)&~(MAIN_FIFO_WINDOW-1));
      __syncwarp(0xffffffffu); // finish all prior reads before reusing this warp's buffer
#pragma unroll
      for(uint32_t index=lane;index<MAIN_FIFO_BYTES/2;index+=32)
        fifo->words[index]=fifo_prefetch_word(t,base,index);
      __syncwarp(0xffffffffu); // every real/lookahead/tail word is initialized
      fifo->base=base;fifo->valid=true;
    }
  }
  if(mask&(1u<<lane)){
    const uint32_t rank=__popc(mask&((1u<<lane)-1u));
    uint16_t word;
    if constexpr(MainFifo)word=fifo->words[(c.pos-fifo->base)/2+rank];
    else word=__ldg(reinterpret_cast<const uint16_t*>(t.main+c.pos+2*rank));
    c.state=(c.state<<16)|word;
  }
  c.pos+=bytes;
  trusted_store<StageRaw6>(t,chunk,i,symbol,slab,staged);return 0;
}
template<bool MainFifo=false,bool StageRaw6=false>
__device__ __forceinline__ uint32_t decode_trusted_slab(const TrustedPacket& t,uint32_t lane,
    uint32_t chunk,bool last,TrustedCursor& c,uint16_t* slab,TrustedFifo* fifo=nullptr,uint8_t* staged=nullptr){
  static_assert(!StageRaw6 || MainFifo,"raw6 staging is a separate FIFO extension");
  if(t.mode){
    for(uint32_t i=lane;i<2048;i+=32)slab[(t.row0+(i>>7))*LDS+(i&127)]=u16(t.main+2*(2048*chunk+i));
    return 0;
  }
  if constexpr(StageRaw6){
    __syncwarp(0xffffffffu);
#pragma unroll
    for(uint32_t index=lane;index<RAW6_STAGE_BYTES/2;index+=32)
      reinterpret_cast<uint16_t*>(staged)[index]=raw6_prefetch_word(t,chunk,index);
    __syncwarp(0xffffffffu);
  }
  const uint32_t col=chunk*CH+lane;
  const uint32_t cc0=t.cols[col],cc1=t.cols[col+32],cc2=t.cols[col+64],cc3=t.cols[col+96];
#pragma unroll 1
  for(uint32_t row=0;row<16;++row){
    const uint32_t base=uint32_t(t.rows[row])*t.nc;
    // Named snapshots avoid dynamic local-array indexing. Context loads are
    // state-independent; each symbol still uses the one original lane state.
    const TrustedContext x0=trusted_context(t,base+cc0),x1=trusted_context(t,base+cc1),
                         x2=trusted_context(t,base+cc2),x3=trusted_context(t,base+cc3);
    const uint32_t i=128*row+lane;
    if(trusted_step<MainFifo,StageRaw6>(t,x0,lane,chunk,i,c,slab,fifo,staged))return c.error;
    if(trusted_step<MainFifo,StageRaw6>(t,x1,lane,chunk,i+32,c,slab,fifo,staged))return c.error;
    if(trusted_step<MainFifo,StageRaw6>(t,x2,lane,chunk,i+64,c,slab,fifo,staged))return c.error;
    if(trusted_step<MainFifo,StageRaw6>(t,x3,lane,chunk,i+96,c,slab,fifo,staged))return c.error;
  }
  return trusted_finish(t,last,c);
}
#endif
#undef BCA_FORCE

} // namespace bca
#undef BCA_HD
