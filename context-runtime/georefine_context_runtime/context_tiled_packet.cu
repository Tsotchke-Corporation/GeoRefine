#include <torch/extension.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "context_tiled_packet.cuh"

__global__ void bct_kernel(const uint8_t* data, uint64_t size, const int64_t* desc,
    uint32_t rows, uint32_t cols, uint32_t nr, uint32_t nc, uint64_t labels,
    uint64_t table_start, uint64_t table_bytes, bct::OutputView output, int32_t* status) {
  const uint32_t packet=blockIdx.x, lane=threadIdx.x;
  status[packet*32+lane]=bct::decode_lane(data,size,desc+9*packet,rows,cols,nr,nc,
                                        labels,table_start,table_bytes,lane,output);
}
std::vector<torch::Tensor> bct_decode_cuda(torch::Tensor data, torch::Tensor desc,
    int64_t rows, int64_t cols, int64_t nr, int64_t nc, int64_t labels,
    int64_t table_start, int64_t table_bytes,
    int64_t output_row0, int64_t output_col0, int64_t output_rows, int64_t output_cols) {
  const c10::cuda::CUDAGuard guard(data.device());
  auto out=torch::zeros({output_rows*output_cols},data.options().dtype(torch::kUInt16));
  auto status=torch::zeros({desc.size(0),32},data.options().dtype(torch::kInt32));
  bct::OutputView output{out.data_ptr<uint16_t>(),static_cast<uint32_t>(output_row0),
      static_cast<uint32_t>(output_col0),static_cast<uint32_t>(output_rows),static_cast<uint32_t>(output_cols),
      static_cast<uint64_t>(out.numel())};
  bct_kernel<<<static_cast<unsigned>(desc.size(0)),32,0,at::cuda::getCurrentCUDAStream()>>>(
      data.data_ptr<uint8_t>(),data.numel(),desc.data_ptr<int64_t>(),rows,cols,nr,nc,
      labels,table_start,table_bytes,output,status.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out,status};
}
