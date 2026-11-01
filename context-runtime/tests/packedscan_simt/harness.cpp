#include "simt.h"
#include <fstream>
#include <thread>
#include <string>
template <unsigned LUT> inline uint32_t lop3(uint32_t a, uint32_t b, uint32_t c) { return lop3(a,b,c,LUT); }
#define BI_RARE(x) (x)
constexpr int T2_MAXCB = 8;
struct WDesc {
  int K, t2_ncb;
  const uint32_t *planes, *t2_ovf, *t21_rowoff, *t21_ckpt;
  const uint8_t *smb, *t2_cb;
  const int32_t *t2_rowp;
};
constexpr int F_TBE21=5;
template <int FMT, int KQT> struct Ld;
#include "decode.inc"
template <class T> std::vector<T> read(const std::string& path) {
  std::ifstream f(path,std::ios::binary|std::ios::ate);
  if (!f) simt::die("missing fixture");
  size_t bytes=f.tellg(); f.seekg(0);
  std::vector<T> v(bytes/sizeof(T)); f.read((char*)v.data(),bytes);
  simt::reg(v.data(),bytes,path.c_str()); return v;
}
int main(int argc,char**argv) {
  if(argc!=5) return 2;
  std::string dir=argv[1]; int N=atoi(argv[2]), K=atoi(argv[3]), S=atoi(argv[4]);
  auto l1=read<uint32_t>(dir+"/l1.bin"), ovf=read<uint32_t>(dir+"/ovf.bin");
  auto rowoff=read<uint32_t>(dir+"/rowoff.bin"), ckpt=read<uint32_t>(dir+"/ckpt.bin");
  auto rowp=read<int32_t>(dir+"/rowp.bin");
  auto smb=read<uint8_t>(dir+"/smb.bin"), cb=read<uint8_t>(dir+"/cb.bin");
  auto expected=read<uint16_t>(dir+"/expected.bin");
  WDesc d{K,int(cb.size()/16),l1.data(),ovf.data(),rowoff.data(),ckpt.data(),smb.data(),cb.data(),rowp.data()};
  using L=Ld<F_TBE21,0>; uint64_t words=0;
  for(int base=0;base<N;base+=4) {
    std::barrier bar(32); simt::Warp warp; warp.bar=&bar;
    std::vector<std::thread> threads;
    for(int lane=0;lane<32;++lane) threads.emplace_back([&,lane] {
      simt::g_lane=lane; simt::g_warp=&warp; simt::g_seq=0; threadIdx.x=lane;
      const int j=lane&15; uint32_t table[64]{};
      for (int pattern=0;pattern<3;++pattern) {
        uint32_t a=pattern==0?8u:((lane>>4)?0u:8u), b=pattern==2?0u:8u;
        uint32_t ra,rb,na,nb;
        t21_scan_pair(a,b,j,ra,rb,na,nb);
        if(ra!=a*j || rb!=b*j || na!=a*16 || nb!=b*16) simt::die("packed bound/halfwarp mismatch");
      }
      // TABREG=1 is selected; scalar table fallback is not exercised here.
      for(int s=0;s<S;++s) {
        L::C scalar[2], paired[2];
        for(int i=0;i<2;++i) {
          int row=std::min(base+2*i+(lane>>4),N-1);
          L::start(d,row,s,S,scalar[i]); paired[i]=scalar[i];
        }
        int c0=s*(K/128)/S,c1=(s+1)*(K/128)/S;
#if BI_T21_SCAN4
        L::R staged[2];
        if(c0<c1) for(int i=0;i<2;++i) {
          int row=std::min(base+2*i+(lane>>4),N-1);
          staged[i]=L::load(d,row,c0*128+8*j,lane,paired[i]);
        }
#endif
        for(int c=c0;c<c1;++c) {
#if BI_T21_SCAN4
          L::Fixed next_fixed[2];
          if(c+1<c1) for(int i=0;i<2;++i) {
            int row=std::min(base+2*i+(lane>>4),N-1);
            next_fixed[i]=L::load_fixed(d,row,(c+1)*128+8*j,lane);
          }
#endif
          L::R raw[2]; uint4 ref[2],got[2];
          for(int i=0;i<2;++i) {
            int row=std::min(base+2*i+(lane>>4),N-1);
            raw[i]=L::load(d,row,c*128+8*j,lane,scalar[i]);
            ref[i]=L::dec(d,raw[i],row,c*128+8*j,lane,j,scalar[i],table);
          }
#if BI_T21_SCAN4
          for(int i=0;i<2;++i)
            if(raw[i].c16!=staged[i].c16 || raw[i].s8.x!=staged[i].s8.x ||
               raw[i].s8.y!=staged[i].s8.y || raw[i].win!=staged[i].win)
              simt::die("prefetched load differs from scalar load");
          L::dec_pair(d,staged,lane,j,paired,table,got);
#else
          L::dec_pair(d,raw,lane,j,paired,table,got);
#endif
          for(int i=0;i<2;++i) {
            int row=std::min(base+2*i+(lane>>4),N-1);
            if(memcmp(&ref[i],&got[i],16)) simt::die("paired differs from scalar");
            if(memcmp(&got[i],expected.data()+row*K+c*128+8*j,16)) simt::die("uint16 decode mismatch");
            if(scalar[i].ow!=paired[i].ow || scalar[i].ob!=paired[i].ob) simt::die("cursor mismatch");
          }
#if BI_T21_SCAN4
          if(c+1<c1) for(int i=0;i<2;++i) {
            staged[i].c16=next_fixed[i].c16;
            staged[i].s8.x=next_fixed[i].s0;
            staged[i].s8.y=next_fixed[i].s1;
            staged[i].win=__ldg(d.t2_ovf+paired[i].ow+(uint32_t)j);
          }
#endif
        }
      }
    });
    for(auto& t:threads) t.join(); words+=std::min(4,N-base)*K;
  }
  printf("PASS words=%llu scalar=paired=source, cursors exact; collectives=%llu\n",(unsigned long long)words,(unsigned long long)simt::g_collectives.load());
}
