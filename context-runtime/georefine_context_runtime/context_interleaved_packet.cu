#include <torch/extension.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "context_interleaved_packet.cuh"

__global__ void bci_kernel(const uint8_t* data, uint64_t size, const int64_t* desc,
    uint32_t rows, uint32_t cols, uint32_t nr, uint32_t nc, uint64_t labels,
    uint16_t* out, int32_t* status) {
  const uint32_t packet=blockIdx.x, lane=threadIdx.x;
  status[packet*32+lane]=bci::decode_lane(data,size,desc+9*packet,rows,cols,nr,nc,
                                        labels,lane,out,uint64_t(rows)*cols);
}
std::vector<torch::Tensor> bci_decode_cuda(torch::Tensor data, torch::Tensor desc,
    int64_t rows, int64_t cols, int64_t nr, int64_t nc, int64_t labels) {
  const c10::cuda::CUDAGuard guard(data.device());
  auto out=torch::zeros({rows*cols},data.options().dtype(torch::kUInt16));
  auto status=torch::zeros({desc.size(0),32},data.options().dtype(torch::kInt32));
  bci_kernel<<<static_cast<unsigned>(desc.size(0)),32,0,at::cuda::getCurrentCUDAStream()>>>(
      data.data_ptr<uint8_t>(),data.numel(),desc.data_ptr<int64_t>(),rows,cols,nr,nc,
      labels,out.data_ptr<uint16_t>(),status.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out,status};
}
