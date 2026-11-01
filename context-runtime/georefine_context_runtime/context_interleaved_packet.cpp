#include "context_interleaved_packet.cuh"
#ifdef BCI_CPU_TEST
// Test-only ABI exercises the exact host/device integer function without torch.
extern "C" void bci_test_decode(const uint8_t* data, uint64_t size, const int64_t* desc,
    uint32_t packets, uint32_t rows, uint32_t cols, uint32_t nr, uint32_t nc,
    uint64_t labels, uint16_t* out, uint32_t* status) {
  for (uint32_t p=0;p<packets;++p)
    for (uint32_t lane=0;lane<32;++lane)
      status[p*32+lane]=bci::decode_lane(data,size,desc+9*p,rows,cols,nr,nc,labels,lane,out,uint64_t(rows)*cols);
}
#else
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> bci_decode_cuda(torch::Tensor data, torch::Tensor desc,
    int64_t rows, int64_t cols, int64_t nr, int64_t nc, int64_t labels);
std::vector<torch::Tensor> decode(torch::Tensor data, torch::Tensor desc,
    int64_t rows, int64_t cols, int64_t nr, int64_t nc, int64_t labels) {
  TORCH_CHECK(data.is_cuda() && desc.is_cuda() && data.device()==desc.device(), "same CUDA device required");
  TORCH_CHECK(data.scalar_type()==torch::kUInt8 && data.dim()==1 && data.is_contiguous(), "contiguous byte bundle required");
  TORCH_CHECK(desc.scalar_type()==torch::kInt64 && desc.dim()==2 && desc.size(1)==9 && desc.is_contiguous(), "contiguous int64 packet descriptors required");
  TORCH_CHECK(rows>0 && rows<=65536 && cols>0 && cols<=65536 && rows*cols<=65536, "bounded geometry required");
  TORCH_CHECK(nr>0 && nr<=64 && nc>0 && nc<=16 && nr*nc<=1024, "bounded context classes required");
  TORCH_CHECK(desc.size(0)>0 && desc.size(0)<=4096 && desc.size(0)<=rows, "bounded packet count required");
  TORCH_CHECK(data.numel()<=1048576 && labels>=0 && labels<=data.numel() && rows+cols<=data.numel()-labels, "bounded bundle/labels required");
  return bci_decode_cuda(data,desc,rows,cols,nr,nc,labels);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode", &decode, "Bounded BCI1 CUDA decode (words, per-packet/lane status)");
}
#endif
