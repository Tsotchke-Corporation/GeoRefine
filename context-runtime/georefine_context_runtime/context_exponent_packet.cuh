#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define BHE_HD __host__ __device__ __forceinline__
#define BHE_GROUP __host__ __device__ __noinline__
#else
#define BHE_HD inline
#define BHE_GROUP inline
#endif

namespace bhe {
constexpr uint32_t CH=128,LDS=136,BN=64,BRANCH=1u<<30;
struct Context {const uint32_t* table;uint32_t words;int32_t singleton;uint32_t maximum;};
struct Group {uint32_t symbols=0,count=0;};
struct Cursor {
  const uint8_t* data=nullptr;uint64_t bytes=0,start=0,end=0,next_word=0,buffer=0;
  uint32_t valid=0,consumed=0,error=0;
};
BHE_HD uint32_t load_be32(const uint8_t* p){
#ifdef __CUDA_ARCH__
  return __byte_perm(*reinterpret_cast<const uint32_t*>(p),0,0x0123);
#else
  return (uint32_t(p[0])<<24)|(uint32_t(p[1])<<16)|(uint32_t(p[2])<<8)|p[3];
#endif
}
BHE_HD uint32_t init(Cursor& c,const uint8_t* data,uint64_t bytes,uint64_t start,uint64_t end){
  c=Cursor{};c.data=data;c.bytes=bytes;c.start=start;c.end=end;
  if(bytes>UINT64_MAX/8 || start>end || end>bytes*8 || end-start>32768u*31u)return c.error=1;
  if(start==end)return 0;
  const uint64_t offset=(start>>5)*4;
  if(offset>bytes || bytes-offset<4)return c.error=4;
  c.valid=32-uint32_t(start&31);c.buffer=load_be32(data+offset)&((uint64_t(1)<<c.valid)-1);
  c.next_word=(start>>5)+1;return 0;
}
BHE_HD uint32_t peek(Cursor& c,uint32_t bits,uint32_t& value){
  if(c.error)return c.error;
  if(!bits || bits>31)return c.error=2;
  if(c.valid<bits){
    const uint64_t offset=c.next_word*4;
    if(offset>c.bytes || c.bytes-offset<4)return c.error=4;
    c.buffer=(c.buffer<<32)|load_be32(c.data+offset);c.valid+=32;++c.next_word;
  }
  value=uint32_t(c.buffer>>(c.valid-bits))&((uint32_t(1)<<bits)-1);return 0;
}
BHE_HD uint32_t consume(Cursor& c,uint32_t bits){
  if(c.error)return c.error;
  if(!bits || bits>c.valid || c.consumed>c.end-c.start || bits>c.end-c.start-c.consumed)return c.error=4;
  c.valid-=bits;c.consumed+=bits;return 0;
}
// CUDA12.9/sm120 fused kernels produced invalid cursor statuses with this
// helper forced inline; the identical out-of-line decode passed exact output
// and all terminal checks. Keep the production finish/status guards unchanged.
BHE_HD uint32_t group_impl(const Context& t,Cursor& c,uint32_t remaining,Group& out){
  out=Group{};
  if(c.error)return c.error;
  if(!remaining)return c.error=2;
  if(t.singleton>=0){
    if(t.singleton>255)return c.error=2;
    out.count=remaining>=3?3:remaining;
    out.symbols=uint32_t(t.singleton)*0x010101u;return 0;
  }
  if(t.singleton!=-1 || !t.table || t.words<320 || !t.maximum || t.maximum>31)return c.error=2;
  const uint32_t lookup_words=t.words-64;
  uint32_t prefix=0;if(peek(c,8,prefix))return c.error;
  uint32_t entry=t.table[prefix];
  if((entry>>30)==0){
    const uint32_t count=(entry>>28)&3,used=(entry>>24)&15;
    if(!count || !used || used>8 || ((entry&0xffffffu)>>(count*8)))return c.error=2;
    if(count<=remaining){
      if(consume(c,used))return c.error;
      out.symbols=entry&0xffffffu;out.count=count;return 0;
    }
    // CH/stream tail: do not consume a future chunk's symbol.
    const uint32_t symbol=entry&255;
    const uint32_t length=(t.table[lookup_words+symbol/4]>>(8*(symbol&3)))&255;
    if(!length || length>t.maximum || length>used)return c.error=2;
    if(consume(c,length))return c.error;
    out.symbols=symbol;out.count=1;return 0;
  }
  uint32_t depth=8;
  for(uint32_t step=0;step<8;++step){
    if((entry>>30)!=1)return c.error=2;
    const uint32_t width=(entry>>26)&15,offset=entry&0x03ffffffu;
    if(!width || width>4 || depth+width>31 || offset<256 || offset>lookup_words || (1u<<width)>lookup_words-offset)return c.error=2;
    if(peek(c,depth+width,prefix))return c.error;
    entry=t.table[offset+(prefix&((1u<<width)-1))];depth+=width;
    if((entry>>30)==0){
      const uint32_t symbol=entry&255,length=(entry>>8)&31;
      if(entry>>13 || !length || length>t.maximum || length>depth)return c.error=2;
      if(consume(c,length))return c.error;
      out.symbols=symbol;out.count=1;return 0;
    }
  }
  return c.error=2;
}
// Experimental value snapshots keep decoder state separate from the caller's
// tensor-core accumulators. The default retains the qualified call boundary.
#if defined(GEOREFINE_BHE_INLINE_VALUE) && GEOREFINE_BHE_INLINE_VALUE == 1
BHE_HD uint32_t group(const Context& t,Cursor& c,uint32_t remaining,Group& out){
  const Context context=t;
  Cursor cursor=c;
  Group result{};
  const uint32_t error=group_impl(context,cursor,remaining,result);
  c=cursor;
  out=result;
  return error;
}
#else
BHE_GROUP uint32_t group(const Context& t,Cursor& c,uint32_t remaining,Group& out){
  return group_impl(t,c,remaining,out);
}
#endif
BHE_HD uint32_t finish(Cursor& c){
  if(c.error)return c.error;
  if(c.consumed!=c.end-c.start)return c.error=5;
  return 0;
}
BHE_HD uint16_t combine(uint32_t exponent,uint8_t sm){
  return uint16_t((exponent<<7)|(sm&127u)|((uint32_t(sm)&128u)<<8));
}
} // namespace bhe
#undef BHE_HD
#undef BHE_GROUP
