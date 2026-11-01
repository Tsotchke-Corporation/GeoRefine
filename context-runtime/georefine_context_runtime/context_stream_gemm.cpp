#include "context_stream_packet.cuh"
#include <new>
// Mirror the CUDA shared storage contract, including reserved padding.
static_assert(sizeof(bcs::PacketView)<=128 && alignof(bcs::PacketView)<=16,
              "prepared view must fit its aligned 128-byte shared slot");
#ifdef BCS_GEMM_CPU_TEST
// Native decoder harness only: no CPU GEMM twin or runtime fallback.
extern "C" void bcsg_test_decode(const uint8_t* data,uint64_t panel_bytes,const int64_t* desc,
    uint32_t panels,uint32_t K,uint32_t nr,uint32_t nc,uint64_t labels,uint64_t ts,uint64_t nt,
    uint16_t* slabs,uint32_t* status,uint32_t* preparations) {
  if (!panels || panels>2 || (K!=512 && K!=1024)) return;
  for (uint32_t p=0;p<panels;++p) {
    alignas(16) unsigned char storage[4*128];
    bcs::PacketView* views[4];bcs::LaneCursor cursors[4][32];
    uint32_t failed=0;
    for (uint32_t w=0;w<4;++w) {
      views[w]=new(storage+w*128) bcs::PacketView{};
      const auto* d=desc+(p*4+w)*9;
      bcs::prepare_packet(data+uint64_t(p)*panel_bytes,panel_bytes,d,K,nr,nc,labels,ts,nt,1,*views[w]);
      if (d[1]!=16*w) views[w]->error=1;
      ++preparations[p];
      for (uint32_t lane=0;lane<32;++lane)
        failed|=bcs::init_lane(*views[w],lane,cursors[w][lane]);
    }
    for (uint32_t c=0;c<K/128 && !failed;++c) {
      for (uint32_t w=0;w<4;++w)
        for (uint32_t lane=0;lane<32;++lane)
          failed|=bcs::decode_next_slab(*views[w],lane,c,cursors[w][lane],
                    slabs+(uint64_t(p)*(K/128)+c)*64*136,64*136);
    }
    for (uint32_t w=0;w<4;++w)
      for (uint32_t lane=0;lane<32;++lane)
        status[p*128+w*32+lane]=cursors[w][lane].error ? cursors[w][lane].error : failed ? 32 : 0;
  }
}
#else
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> bcsg_cuda(torch::Tensor data,torch::Tensor desc,torch::Tensor x,
    int64_t nr,int64_t nc,int64_t labels,int64_t ts,int64_t nt);
torch::Tensor bcsg_dense_cuda(torch::Tensor w,torch::Tensor x);

static void check_x(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kBFloat16 && x.dim()==2 && x.is_contiguous(),
              "contiguous CUDA bf16 x required");
  TORCH_CHECK((x.size(0)==1 || x.size(0)==16 || x.size(0)==64 || x.size(0)==128) &&
              (x.size(1)==512 || x.size(1)==1024), "bounded M/K required");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr())%16==0,"16-byte aligned x required");
}
std::vector<torch::Tensor> gemm(torch::Tensor data,torch::Tensor desc,torch::Tensor x,
    int64_t nr,int64_t nc,int64_t labels,int64_t ts,int64_t nt) {
  check_x(x);
  TORCH_CHECK(data.is_cuda() && desc.is_cuda() && data.device()==x.device() && desc.device()==x.device(),
              "same CUDA device required");
  TORCH_CHECK(data.scalar_type()==torch::kUInt8 && data.dim()==2 && data.is_contiguous() &&
              data.size(0)>=1 && data.size(0)<=1024 && data.size(1)>0 && data.size(1)<=1048576,
              "bounded byte panels [P,B], 1<=P<=1024 required");
  TORCH_CHECK(desc.scalar_type()==torch::kInt64 && desc.dim()==3 && desc.size(0)==data.size(0) &&
              desc.size(1)==4 && desc.size(2)==9 && desc.is_contiguous(), "wide descriptors [P,4,9] required");
  TORCH_CHECK((nr==1 || nr==4 || nr==16 || nr==64) && (nc==1 || nc==4 || nc==16),
              "supported context classes required");
  const auto size=data.size(1);
  TORCH_CHECK(labels>=0 && labels<=size && 64+x.size(1)<=size-labels &&
              ts>=0 && ts<=size && nt>=0 && nt<=size-ts,"bounded panel-local metadata required");
  return bcsg_cuda(data,desc,x,nr,nc,labels,ts,nt);
}
torch::Tensor dense(torch::Tensor w,torch::Tensor x) {
  check_x(x);
  TORCH_CHECK(w.is_cuda() && w.device()==x.device() && w.scalar_type()==torch::kBFloat16 &&
              w.dim()==2 && w.size(0)>=64 && w.size(0)<=65536 && w.size(0)%64==0 &&
              w.size(1)==x.size(1) && w.is_contiguous(), "contiguous CUDA bf16 w[P*64,K] required");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(w.data_ptr())%16==0,"16-byte aligned weights required");
  return bcsg_dense_cuda(w,x);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("gemm",&gemm,"Experimental BCS3 persistent stream consumer (y, final status)");
  m.def("dense",&dense,"Matched BI BF16 S1 tile64 panel control");
}
#endif
