#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "miv_kq.h"

static const uint8_t* p8(const c10::optional<at::Tensor>& t) {
  return (t.has_value() && t->defined()) ? reinterpret_cast<const uint8_t*>(t->data_ptr()) : nullptr;
}

static MivKqDesc mk(int64_t type, int64_t n, int64_t k, int64_t wpr, int64_t unroll, int64_t rpw,
                    const std::vector<at::Tensor>& arrs, const c10::optional<at::Tensor>& grid,
                    const c10::optional<at::Tensor>& ksigns, const c10::optional<at::Tensor>& bias) {
  TORCH_CHECK(arrs.size() >= 2 && arrs.size() <= 5, "miv_kq: 2..5 arrays");
  for (const auto& a : arrs)
    TORCH_CHECK(a.is_cuda() && a.is_contiguous(), "miv_kq: arrays must be contiguous CUDA");
  MivKqDesc d{};
  d.type = (int32_t)type; d.n = (int32_t)n; d.k = (int32_t)k; d.wpr = (int32_t)wpr;
  d.unroll = (int32_t)unroll;
  d.rpw = (int32_t)rpw;
  const uint8_t* ps[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
  for (size_t i = 0; i < arrs.size(); ++i) ps[i] = reinterpret_cast<const uint8_t*>(arrs[i].data_ptr());
  d.a0 = ps[0]; d.a1 = ps[1]; d.a2 = ps[2]; d.a3 = ps[3]; d.a4 = ps[4];
  d.grid = reinterpret_cast<const uint32_t*>(p8(grid));
  d.ksigns = p8(ksigns);
  d.bias = nullptr;
  if (bias.has_value() && bias->defined()) {
    TORCH_CHECK(bias->scalar_type() == at::kBFloat16 && bias->numel() == n, "miv_kq: bias");
    d.bias = reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr<at::BFloat16>());
  }
  return d;
}

void miv_kq_out(at::Tensor x, int64_t type, int64_t n, int64_t k, int64_t wpr, int64_t unroll,
                int64_t rpw, std::vector<at::Tensor> arrs, c10::optional<at::Tensor> grid,
                c10::optional<at::Tensor> ksigns, c10::optional<at::Tensor> bias, at::Tensor y) {
  const int64_t M = x.size(0);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.stride(1) == 1 &&
              x.size(1) == k && x.stride(0) % 8 == 0 &&
              (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0, "miv_kq: x");
  TORCH_CHECK(y.is_cuda() && y.scalar_type() == at::kBFloat16 && y.dim() == 2 && y.size(0) == M &&
              y.size(1) == n && y.stride(1) == 1, "miv_kq: y");
  TORCH_CHECK(M >= 1 && M <= 8, "miv_kq: 1 <= M <= 8");
  const MivKqDesc d = mk(type, n, k, wpr, unroll, rpw, arrs, grid, ksigns, bias);
  const c10::cuda::CUDAGuard g(x.device());
  const cudaError_t e = miv_kq_gemv(d, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                                    x.stride(0), (int)M,
                                    reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
                                    y.stride(0), at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == cudaSuccess, "miv_kq_gemv: ", cudaGetErrorString(e));
}

void miv_kq_dequant(int64_t type, int64_t n, int64_t k, std::vector<at::Tensor> arrs,
                    c10::optional<at::Tensor> grid, c10::optional<at::Tensor> ksigns, at::Tensor w) {
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kFloat && w.is_contiguous() && w.numel() == n * k,
              "miv_kq_dequant: w fp32 [n,k]");
  const MivKqDesc d = mk(type, n, k, 1, 1, 1, arrs, grid, ksigns, c10::nullopt);
  const c10::cuda::CUDAGuard g(w.device());
  const cudaError_t e = miv_kq_dequant_f32(d, w.data_ptr<float>(), at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == cudaSuccess, "miv_kq_dequant: ", cudaGetErrorString(e));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("miv_kq_out", &miv_kq_out, "fused K/I-quant dequant MIV GEMV into y");
  m.def("miv_kq_dequant", &miv_kq_dequant, "decode-only twin (fp32)");
}
