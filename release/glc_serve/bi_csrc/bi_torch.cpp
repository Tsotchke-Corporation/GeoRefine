// Torch binding for BI-GEMM (glc_serve.bigemm).  Checks, then the C entry points.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include "bi_api.h"

template <typename T>
static const T* ptr_or_null(const std::vector<at::Tensor>& a, size_t i) {
  if (i >= a.size() || !a[i].defined() || a[i].numel() == 0) return nullptr;
  TORCH_CHECK(a[i].is_cuda() && a[i].is_contiguous(), "bi_gemm: array ", i, " must be contiguous CUDA");
  return reinterpret_cast<const T*>(a[i].data_ptr());
}

// fmt: 0 bf16 [w]; 1 q8 [qs, sc]; 2 tbe [planes, smb, esc, rowparam, escidx]; 3 kq [a0..a4, grid, ksigns]
void bi_gemm_out(at::Tensor x, int64_t M, int64_t fmt, int64_t kqt, int64_t N, int64_t K,
                 std::vector<at::Tensor> arrs, c10::optional<at::Tensor> bias, at::Tensor y,
                 int64_t S, at::Tensor ws) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 &&
              x.stride(0) % 8 == 0 && (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0 &&
              x.size(0) >= M && x.size(1) >= K, "bi_gemm: x");
  TORCH_CHECK(y.is_cuda() && y.scalar_type() == at::kBFloat16 && y.stride(1) == 1 && y.size(0) >= M &&
              y.size(1) >= N, "bi_gemm: y");
  TORCH_CHECK(K % 256 == 0 && M >= 1 && S >= 1 && S <= K / 128, "bi_gemm: K % 256, M >= 1, 1 <= S <= K/128");
  TORCH_CHECK(S == 1 || (ws.is_cuda() && ws.scalar_type() == at::kFloat && ws.numel() >= S * M * N),
              "bi_gemm: workspace too small");
  const at::cuda::CUDAGuard guard(x.device());
  BiDesc d{};
  d.N = (int)N; d.K = (int)K;
  switch (fmt) {
    case BI_BF16: d.w = ptr_or_null<__nv_bfloat16>(arrs, 0); TORCH_CHECK(d.w, "bf16 w"); break;
    case BI_Q8:
      d.qs = ptr_or_null<int8_t>(arrs, 0); d.sc = ptr_or_null<unsigned short>(arrs, 1);
      TORCH_CHECK(d.qs && d.sc, "q8 arrays"); break;
    case BI_TBE:
      d.planes = ptr_or_null<uint32_t>(arrs, 0); d.smb = ptr_or_null<uint8_t>(arrs, 1);
      d.esc = ptr_or_null<uint8_t>(arrs, 2); d.rowparam = ptr_or_null<int32_t>(arrs, 3);
      d.escidx = ptr_or_null<int32_t>(arrs, 4);
      TORCH_CHECK(d.planes && d.smb && d.esc && d.rowparam && d.escidx, "tbe arrays");
      TORCH_CHECK(arrs[4].numel() == N * (K / 128), "tbe escidx size");
      break;
    case BI_KQ:
      d.a0 = ptr_or_null<uint8_t>(arrs, 0); d.a1 = ptr_or_null<uint8_t>(arrs, 1);
      d.a2 = ptr_or_null<uint8_t>(arrs, 2); d.a3 = ptr_or_null<uint8_t>(arrs, 3);
      d.a4 = ptr_or_null<uint8_t>(arrs, 4); d.grid = ptr_or_null<uint32_t>(arrs, 5);
      d.ksigns = ptr_or_null<uint8_t>(arrs, 6); 
      break;
    default: TORCH_CHECK(false, "bi_gemm: fmt");
  }
  d.bias = nullptr;
  if (bias.has_value() && bias->defined()) {
    TORCH_CHECK(bias->scalar_type() == at::kBFloat16 && bias->numel() == N, "bi_gemm: bias");
    d.bias = reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr());
  }
  auto st = at::cuda::getCurrentCUDAStream();
  auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  float* wsp = S > 1 ? ws.data_ptr<float>() : nullptr;
  cudaError_t e = fmt == BI_KQ
      ? bi_gemm_kq((int)kqt, xp, x.stride(0), (int)M, d, (int)S, yp, y.stride(0), wsp, st)
      : bi_gemm_dense((int)fmt, xp, x.stride(0), (int)M, d, (int)S, yp, y.stride(0), wsp, st);
  TORCH_CHECK(e == cudaSuccess, "bi_gemm launch: ", cudaGetErrorString(e));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bi_gemm_out", &bi_gemm_out, "batch-invariant tensor-core GEMM (in-kernel weight decode)");
}
