// Torch binding for the experimental paired-warp M=1 path.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "miv_tbe.h"

cudaError_t miv_tbe_pair_m1(const MivTbeDesc&, const __nv_bfloat16*, int64_t,
                            __nv_bfloat16*, int64_t, cudaStream_t);

void miv_tbe_pair_out(at::Tensor planes, at::Tensor smb, at::Tensor esc,
                      at::Tensor rowparam, at::Tensor escbase, at::Tensor x,
                      at::Tensor y, int64_t N, int64_t K, int64_t wpr,
                      int64_t unroll, c10::optional<at::Tensor> bias) {
  TORCH_CHECK(wpr == 1 || wpr == 2 || wpr == 4 || wpr == 8,
              "miv_tbe_pair: wpr must be 1, 2, 4, or 8");
  TORCH_CHECK(unroll == 1 || unroll == 2 || unroll == 4,
              "miv_tbe_pair: unroll must be 1, 2, or 4");
  TORCH_CHECK(N > 0 && K > 0 && N < (1LL << 31) && K < (1LL << 31),
              "miv_tbe_pair: invalid shape");
  TORCH_CHECK(K % (64 * wpr) == 0, "miv_tbe_pair: K must be divisible by 64*wpr");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() &&
              x.dim() == 2 && x.size(0) == 1 && x.size(1) == K,
              "miv_tbe_pair: M=1 contiguous bf16 x [1,K] required");
  TORCH_CHECK(y.is_cuda() && y.device() == x.device() &&
              y.scalar_type() == at::kBFloat16 && y.is_contiguous() &&
              y.dim() == 2 && y.size(0) == 1 && y.size(1) == N,
              "miv_tbe_pair: contiguous bf16 y [1,N] on x device required");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0,
              "miv_tbe_pair: x must be 16-byte aligned");
  TORCH_CHECK(planes.is_cuda() && smb.is_cuda() && esc.is_cuda() && rowparam.is_cuda() &&
              escbase.is_cuda(), "miv_tbe_pair: CUDA descriptor tensors required");
  TORCH_CHECK(planes.device() == x.device() && smb.device() == x.device() &&
              esc.device() == x.device() && rowparam.device() == x.device() &&
              escbase.device() == x.device(), "miv_tbe_pair: descriptor device mismatch");
  TORCH_CHECK(planes.scalar_type() == at::kInt && smb.scalar_type() == at::kByte &&
              esc.scalar_type() == at::kByte && rowparam.scalar_type() == at::kInt &&
              escbase.scalar_type() == at::kInt, "miv_tbe_pair: descriptor dtypes");
  TORCH_CHECK(planes.is_contiguous() && smb.is_contiguous() && esc.is_contiguous() &&
              rowparam.is_contiguous() && escbase.is_contiguous(),
              "miv_tbe_pair: contiguous descriptors required");
  TORCH_CHECK(planes.numel() == N * (K / 64) * 6 && smb.numel() == N * K &&
              esc.numel() >= 16 && rowparam.numel() == N && escbase.numel() == N * wpr,
              "miv_tbe_pair: descriptor sizes do not match N,K,wpr");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(esc.data_ptr()) & 3) == 0,
              "miv_tbe_pair: esc must be four-byte aligned");
  MivTbeDesc d{};
  d.n = static_cast<int32_t>(N); d.k = static_cast<int32_t>(K);
  d.wpr = static_cast<int32_t>(wpr); d.unroll = static_cast<int32_t>(unroll);
  d.version = 2;
  d.planes = reinterpret_cast<const uint32_t*>(planes.data_ptr<int32_t>());
  d.smb = smb.data_ptr<uint8_t>(); d.esc = esc.data_ptr<uint8_t>();
  d.rowparam = rowparam.data_ptr<int32_t>(); d.escbase = escbase.data_ptr<int32_t>();
  d.bias = nullptr;
  if (bias.has_value() && bias->defined()) {
    TORCH_CHECK(bias->is_cuda() && bias->device() == x.device() &&
                bias->scalar_type() == at::kBFloat16 && bias->is_contiguous() &&
                bias->dim() == 1 && bias->numel() == N,
                "miv_tbe_pair: bias must be contiguous bf16 [N] on x device");
    d.bias = reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr<at::BFloat16>());
  }
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaError_t result = miv_tbe_pair_m1(
      d, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
      x.stride(0), reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
      y.stride(0), at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(result == cudaSuccess, "miv_tbe_pair_m1: ", cudaGetErrorString(result));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("miv_tbe_pair_out", &miv_tbe_pair_out,
             "experimental paired-warp M=1 TBE MIV output");
}
