#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <vector>

extern "C" {
static constexpr uint32_t TOT = 1u << 16, LOWER = 1u << 23;
int ppcx_checkpoint(const uint8_t *stream, size_t bytes, uint64_t n,
                    const uint16_t *ctx, uint32_t nctx, uint32_t vocab,
                    const uint32_t *freq, uint32_t stride,
                    uint32_t *states, uint32_t *offsets, uint16_t *symbols) {
  try {
    if (!stream || bytes < 4 || bytes > UINT32_MAX || !n || !ctx || !nctx ||
        !vocab || vocab > TOT || !freq || !stride || !states || !offsets ||
        !symbols || n > 100000000) return 1;
    for (uint32_t c=0;c<nctx;c++) {
      uint64_t sum=0; for(uint32_t s=0;s<vocab;s++) { auto f=freq[size_t(c)*vocab+s]; if(f>TOT || sum+f>TOT)return 2; sum+=f; }
      if(sum && sum!=TOT)return 2;
    }
    uint32_t state=uint32_t(stream[0])|(uint32_t(stream[1])<<8)|(uint32_t(stream[2])<<16)|(uint32_t(stream[3])<<24);
    if(state<LOWER)return 3; size_t ip=4; uint64_t cp=0; uint32_t row=0;
    std::vector<uint32_t> st((n+stride-1)/stride), off(st.size());
    for(uint64_t i=0;i<n;i++) {
      if(ctx[i]>=nctx)return 4;
      if(i==cp){st[row]=state;off[row]=uint32_t(ip);row++;cp+=stride;}
      uint32_t c=ctx[i],slot=state&65535, cum=0,sym=vocab;
      for(uint32_t s=0;s<vocab;s++){uint32_t f=freq[size_t(c)*vocab+s]; if(slot<cum+f){sym=s;break;}cum+=f;}
      if(sym==vocab)return 5; uint32_t f=freq[size_t(c)*vocab+sym]; symbols[i]=uint16_t(sym);
      uint64_t next=uint64_t(f)*(state>>16)+(slot-cum); if(next>UINT32_MAX)return 3;state=uint32_t(next);
      while(state<LOWER){if(ip>=bytes)return 3;state=(state<<8)|stream[ip++];}
    }
    if(ip!=bytes||state!=LOWER)return 3;std::copy(st.begin(),st.end(),states);std::copy(off.begin(),off.end(),offsets);return 0;
  } catch (...) { return 6; }
}
}
