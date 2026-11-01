#include "context_tiled_packet.cuh"
#ifdef BCT_CPU_TEST
// Test-only ABI executes the exact host/device helper without torch or CUDA.
extern "C" void bct_test_decode(const uint8_t* data, uint64_t size, const int64_t* desc,
    uint32_t packets, uint32_t rows, uint32_t cols, uint32_t nr, uint32_t nc,
    uint64_t labels, uint64_t table_start, uint64_t table_bytes,
    uint32_t output_row0, uint32_t output_col0, uint32_t output_rows, uint32_t output_cols,
    uint16_t* out, uint32_t* status) {
  bct::OutputView output{out,output_row0,output_col0,output_rows,output_cols,uint64_t(output_rows)*output_cols};
  for (uint32_t p=0;p<packets;++p)
    for (uint32_t lane=0;lane<32;++lane)
      status[p*32+lane]=bct::decode_lane(data,size,desc+9*p,rows,cols,nr,nc,labels,
                                         table_start,table_bytes,lane,output);
}
#else
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> bct_decode_cuda(torch::Tensor data, torch::Tensor desc,
    int64_t rows, int64_t cols, int64_t nr, int64_t nc, int64_t labels,
    int64_t table_start, int64_t table_bytes,
    int64_t output_row0, int64_t output_col0, int64_t output_rows, int64_t output_cols);
std::vector<torch::Tensor> decode(torch::Tensor data, torch::Tensor desc,
    int64_t rows, int64_t cols, int64_t nr, int64_t nc, int64_t labels,
    int64_t table_start, int64_t table_bytes,
    int64_t output_row0, int64_t output_col0, int64_t output_rows, int64_t output_cols) {
  TORCH_CHECK(data.is_cuda() && desc.is_cuda() && data.device()==desc.device(), "same CUDA device required");
  TORCH_CHECK(data.scalar_type()==torch::kUInt8 && data.dim()==1 && data.is_contiguous(), "contiguous byte bundle required");
  TORCH_CHECK(desc.scalar_type()==torch::kInt64 && desc.dim()==2 && desc.size(1)==9 && desc.is_contiguous(), "contiguous int64 tile descriptors required");
  TORCH_CHECK(rows>0 && rows<=65536 && cols>0 && cols<=65536 && rows*cols<=65536, "bounded geometry required");
  TORCH_CHECK(nr>0 && nr<=64 && nc>0 && nc<=16 && nr*nc<=1024, "bounded context classes required");
  TORCH_CHECK(desc.size(0)>0 && desc.size(0)<=4096 && desc.size(0)<=rows*cols, "bounded tile count required");
  TORCH_CHECK(data.numel()<=1048576 && labels>=0 && labels<=data.numel() && rows+cols<=data.numel()-labels, "bounded bundle/labels required");
  TORCH_CHECK(table_start>=0 && table_start<=data.numel() && table_bytes>=0 && table_bytes<=data.numel()-table_start, "bounded shared CDF required");
  TORCH_CHECK(output_row0>=0 && output_row0<rows && output_col0>=0 && output_col0<cols &&
              output_rows>0 && output_cols>0 && output_rows<=rows-output_row0 && output_cols<=cols-output_col0,
              "bounded output view required");
  return bct_decode_cuda(data,desc,rows,cols,nr,nc,labels,table_start,table_bytes,
                         output_row0,output_col0,output_rows,output_cols);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode", &decode, "Bounded BCT2 CUDA tile decode (words, per-tile/lane status)");
}
#endif
