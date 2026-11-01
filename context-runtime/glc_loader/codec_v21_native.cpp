// Exact scalar, single-thread TBE21 packing. Codebook selection stays in Python.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <cstring>
#ifndef T21_SOURCE_SHA256
#error "T21_SOURCE_SHA256 must bind the compiled source"
#endif

namespace {
constexpr uint64_t WORD_LIMIT = uint64_t(1) << 32;
enum Status { PACK_OK=0, PACK_SHAPE=1, PACK_CAPACITY=2, PACK_SYMBOL=3, PACK_OVERFLOW=4, PACK_OFFSETS=5 };
bool product(uint64_t a, uint64_t b, uint64_t& out) {
  if (b && a > std::numeric_limits<uint64_t>::max()/b) return false;
  out=a*b; return out <= std::numeric_limits<size_t>::max()/sizeof(uint16_t);
}
int validate(const uint16_t* words,uint64_t words_n,uint64_t n,uint64_t k,
             const uint8_t* lut,uint64_t lut_n,uint32_t ncb,const uint8_t* sel,uint64_t sel_n) {
  uint64_t count;
  if (!words || !lut || !sel || !n || !k || k%128 || !product(n,k,count) || count!=words_n ||
      ncb<1 || ncb>2 || lut_n!=uint64_t(ncb)*256 || sel_n!=n) return PACK_SHAPE;
  for(uint64_t i=0;i<lut_n;++i) if(lut[i]>16) return PACK_SYMBOL;
  for(uint64_t r=0;r<n;++r) if(sel[r]>=ncb) return PACK_SYMBOL;
  return PACK_OK;
}
uint32_t length(const uint8_t* symbols,uint32_t* counts) {
  for(int q=0;q<4;++q) counts[q]=0;
  for(int i=0;i<128;++i) {
    const uint8_t s=symbols[i];
    counts[0]+=(s>=3);counts[1]+=(s>=6);counts[2]+=(s>=9);counts[3]+=(s==16);
  }
  return 2*counts[0]+2*counts[1]+3*counts[2]+8*counts[3];
}
struct Bits {
  uint32_t* out;uint64_t cursor,limit;
  bool put(uint32_t value,unsigned width) {
    if(!width || width>8 || cursor>limit || width>limit-cursor || value>=(1u<<width)) return false;
    const uint64_t word=cursor>>5;const unsigned shift=unsigned(cursor&31);
    out[word]|=value<<shift;
    if(shift+width>32) out[word+1]|=value>>(32-shift);
    cursor+=width;return true;
  }
};
}

// Used by the real planning pass; also exposes overflow boundaries without huge allocations.
extern "C" int t21_advance_row(uint64_t base,uint64_t bits,uint32_t* offset,uint64_t* next) {
  if(!offset || !next) return PACK_SHAPE;
  if(bits>std::numeric_limits<uint64_t>::max()-31) return PACK_OVERFLOW;
  const uint64_t count=(bits+31)/32;
  if(base>=WORD_LIMIT || count>=WORD_LIMIT-base) return PACK_OVERFLOW; // existing strict <2^32 bound
  *offset=uint32_t(base);*next=base+count;return PACK_OK;
}

// stats: n1,n2,n3,n4,pad_bits,max_chunk_payload_bits,total_overflow_words.
extern "C" int t21_plan(const uint16_t* words,uint64_t words_n,uint64_t n,uint64_t k,
    const uint8_t* lut,uint64_t lut_n,uint32_t ncb,const uint8_t* sel,uint64_t sel_n,
    uint32_t* rowoff,uint64_t rowoff_n,uint64_t* stats,uint64_t stats_n) {
  int result=validate(words,words_n,n,k,lut,lut_n,ncb,sel,sel_n);if(result) return result;
  if(!rowoff || rowoff_n!=n || !stats || stats_n!=7) return PACK_CAPACITY;
  std::memset(stats,0,7*sizeof(uint64_t));uint64_t base=0;
  for(uint64_t r=0;r<n;++r) {
    const uint8_t* table=lut+uint64_t(sel[r])*256;uint64_t bits=0;
    for(uint64_t c=0;c<k;c+=128) {
      uint8_t symbols[128];uint32_t counts[4];
      for(int i=0;i<128;++i) symbols[i]=table[(words[r*k+c+i]>>7)&255];
      const uint32_t chunk=length(symbols,counts);
      if(bits>std::numeric_limits<uint64_t>::max()-chunk) return PACK_OVERFLOW;
      bits+=chunk;stats[5]=std::max(stats[5],uint64_t(chunk));
      for(int q=0;q<4;++q) {
        if(stats[q]>std::numeric_limits<uint64_t>::max()-counts[q]) return PACK_OVERFLOW;
        stats[q]+=counts[q];
      }
    }
    uint64_t next;result=t21_advance_row(base,bits,rowoff+r,&next);if(result) return result;
    const uint64_t padding=(next-base)*32-bits;
    if(stats[4]>std::numeric_limits<uint64_t>::max()-padding) return PACK_OVERFLOW;
    stats[4]+=padding;base=next;
  }
  stats[6]=base;return PACK_OK;
}

extern "C" int t21_fill(const uint16_t* words,uint64_t words_n,uint64_t n,uint64_t k,
    const uint8_t* lut,uint64_t lut_n,uint32_t ncb,const uint8_t* sel,uint64_t sel_n,
    const uint32_t* rowoff,uint64_t rowoff_n,uint8_t* smb,uint64_t smb_n,
    uint16_t* l1,uint64_t l1_n,uint32_t* ovf,uint64_t ovf_n) {
  int result=validate(words,words_n,n,k,lut,lut_n,ncb,sel,sel_n);if(result) return result;
  if(!rowoff || rowoff_n!=n || !smb || smb_n!=words_n || !l1 || l1_n!=words_n/8 ||
      (!ovf && ovf_n) || ovf_n>=WORD_LIMIT || ovf_n>std::numeric_limits<size_t>::max()/4) return PACK_CAPACITY;
  if(rowoff[0]!=0) return PACK_OFFSETS;
  for(uint64_t r=0;r<n;++r) if(rowoff[r]>ovf_n || (r && rowoff[r]<rowoff[r-1])) return PACK_OFFSETS;
  if(ovf_n) std::memset(ovf,0,size_t(ovf_n)*4);
  for(uint64_t r=0;r<n;++r) {
    const uint64_t end=r+1<n?rowoff[r+1]:ovf_n;
    Bits bits{ovf,uint64_t(rowoff[r])*32,end*32};const uint8_t* table=lut+uint64_t(sel[r])*256;
    for(uint64_t c=0;c<k;c+=128) {
      uint8_t symbols[128],exps[128];
      for(int i=0;i<128;++i) {
        const uint16_t w=words[r*k+c+i];exps[i]=uint8_t((w>>7)&255);symbols[i]=table[exps[i]];
        smb[r*k+c+i]=uint8_t(((w>>8)&128)|(w&127));
      }
      for(int lane=0;lane<16;++lane) {
        uint32_t packed=0;
        for(int i=0;i<8;++i) packed|=uint32_t(std::min(unsigned(symbols[8*lane+i]),3u))<<((i%4)*4+(i>=4?2:0));
        l1[(r*(k/128)+c/128)*16+lane]=uint16_t(packed);
      }
      for(int i=0;i<128;++i) if(symbols[i]>=3 && !bits.put(std::min(unsigned(symbols[i])-3,3u),2)) return PACK_CAPACITY;
      for(int i=0;i<128;++i) if(symbols[i]>=6 && !bits.put(std::min(unsigned(symbols[i])-6,3u),2)) return PACK_CAPACITY;
      for(int i=0;i<128;++i) if(symbols[i]>=9 && !bits.put(std::min(unsigned(symbols[i])-9,7u),3)) return PACK_CAPACITY;
      for(int i=0;i<128;++i) if(symbols[i]==16 && !bits.put(exps[i],8)) return PACK_CAPACITY;
    }
    if(bits.cursor>bits.limit || bits.limit-bits.cursor>=32) return PACK_OFFSETS;
  }
  return PACK_OK;
}

namespace {
bool read_bits(const uint32_t* stream, uint64_t begin, uint64_t end,
               uint64_t offset, unsigned width, uint32_t& value) {
  if (!width || width > 8 || end < begin || offset > end-begin || width > end-begin-offset) return false;
  const uint64_t bit = begin + offset;
  const uint64_t word = bit >> 5;
  const unsigned shift = unsigned(bit & 31);
  uint64_t pair = stream[word];
  if (shift + width > 32) pair |= uint64_t(stream[word + 1]) << 32;
  value = uint32_t((pair >> shift) & ((uint32_t(1) << width) - 1));
  return true;
}

bool decode_chunk(const uint16_t* packed, const uint8_t* smb, const uint8_t* cb,
                  const uint32_t* stream, uint64_t row_begin, uint64_t row_end,
                  uint64_t& cursor, uint16_t* out) {
  uint8_t symbols[128], tier2[128]={}, tier3[128]={}, tier4[128]={}, raw_values[128]={};
  uint32_t n1=0, n2=0, n3=0, n4=0;
  for (unsigned i=0;i<128;++i) {
    const unsigned pos=i%8;
    const unsigned shift = pos < 4 ? 4*pos : 4*(pos-4)+2;
    symbols[i]=uint8_t((packed[i/8] >> shift) & 3);
  }
  const uint64_t tier1_start=cursor-row_begin;
  for(unsigned i=0;i<128;++i) if(symbols[i]==3) {
    uint32_t digit;
    if(!read_bits(stream,row_begin,row_end,tier1_start+uint64_t(n1)*2,2,digit)) return false;
    ++n1; symbols[i]=uint8_t(3+digit); tier2[i]=uint8_t(digit==3);
  }
  cursor += uint64_t(n1)*2;
  const uint64_t tier2_start=cursor-row_begin;
  for(unsigned i=0;i<128;++i) if(tier2[i]) {
    uint32_t digit;
    if(!read_bits(stream,row_begin,row_end,tier2_start+uint64_t(n2)*2,2,digit)) return false;
    ++n2; symbols[i]=uint8_t(6+digit); tier3[i]=uint8_t(digit==3);
  }
  cursor += uint64_t(n2)*2;
  const uint64_t tier3_start=cursor-row_begin;
  for(unsigned i=0;i<128;++i) if(tier3[i]) {
    uint32_t digit;
    if(!read_bits(stream,row_begin,row_end,tier3_start+uint64_t(n3)*3,3,digit)) return false;
    ++n3; symbols[i]=uint8_t(9+digit); tier4[i]=uint8_t(digit==7);
  }
  cursor += uint64_t(n3)*3;
  const uint64_t raw_start=cursor-row_begin;
  for(unsigned i=0;i<128;++i) if(tier4[i]) {
    uint32_t exponent;
    if(!read_bits(stream,row_begin,row_end,raw_start+uint64_t(n4)*8,8,exponent)) return false;
    raw_values[i]=uint8_t(exponent);
    ++n4;
  }
  cursor += uint64_t(n4)*8;
  for(unsigned i=0;i<128;++i) {
    uint8_t exponent;
    if(tier4[i]) {
      exponent=raw_values[i];
    } else {
      if(symbols[i]>=16) return false;
      exponent=cb[symbols[i]];
    }
    const uint8_t mantissa=smb[i];
    out[i]=uint16_t((uint16_t(mantissa&0x80)<<8) | (uint16_t(exponent)<<7) | (mantissa&0x7f));
  }
  return cursor <= row_end;
}
}

// Decode the serialized TBE21 streams directly to padded BF16 words.
// Statuses: 0 OK, 1 shape, 2 capacity, 3 selector/codebook, 4 overflow,
// 5 offsets/padding, 6 malformed or truncated tier payload.
extern "C" int t21_decode(const uint8_t* smb,uint64_t smb_n,const uint16_t* l1,uint64_t l1_n,
    const uint32_t* ovf,uint64_t ovf_n,const uint32_t* rowoff,uint64_t rowoff_n,
    const uint8_t* cb,uint64_t cb_n,uint32_t ncb,const uint8_t* cbsel,uint64_t cbsel_n,
    uint64_t n,uint64_t k,uint16_t* out,uint64_t out_n) {
  uint64_t words, chunks, packed_words;
  if(!smb || !l1 || !rowoff || !cb || !cbsel || !out || !n || !k || k%128 || ncb<1 || ncb>2 ||
     !product(n,k,words) || smb_n!=words || !product(n,k/8,packed_words) || l1_n!=packed_words ||
     cb_n!=uint64_t(ncb)*16 || cbsel_n!=(n+7)/8 || out_n!=words ||
     (ovf_n && !ovf) || ovf_n>=WORD_LIMIT || ovf_n>std::numeric_limits<size_t>::max()/4)
    return PACK_SHAPE;
  chunks=k/128;
  if(rowoff_n!=n) return PACK_CAPACITY;
  for(uint64_t r=0;r<n;++r) {
    const uint8_t selector=uint8_t((cbsel[r>>3]>>(r&7))&1);
    if(selector>=ncb) return PACK_SYMBOL;
  }
  if(rowoff[0]!=0) return PACK_OFFSETS;
  for(uint64_t r=0;r<n;++r) {
    if(rowoff[r]>ovf_n || (r && rowoff[r]<rowoff[r-1])) return PACK_OFFSETS;
  }
  for(uint64_t r=0;r<n;++r) {
    const uint64_t start=rowoff[r], finish=(r+1<n?rowoff[r+1]:ovf_n);
    if(finish<start || finish>ovf_n) return PACK_OFFSETS;
    const uint64_t begin=start*32, end=finish*32;
    uint64_t cursor=begin;
    const uint8_t selector=uint8_t((cbsel[r>>3]>>(r&7))&1);
    const uint8_t* table=cb+uint64_t(selector)*16;
    for(uint64_t c=0;c<chunks;++c) {
      const uint64_t index=r*chunks+c;
      uint16_t decoded[128];
      const uint8_t* smb_chunk=smb+r*k+c*128;
      if(!decode_chunk(l1+index*16,smb_chunk,table,ovf,begin,end,cursor,decoded)) return 6;
      std::memcpy(out+r*k+c*128,decoded,sizeof(decoded));
    }
    if(cursor>end || end-cursor>=32) return PACK_OFFSETS;
  }
  return PACK_OK;
}
extern "C" uint32_t t21_pack_abi(){return 1;}
extern "C" const char* t21_pack_source_sha256(){return T21_SOURCE_SHA256;}
