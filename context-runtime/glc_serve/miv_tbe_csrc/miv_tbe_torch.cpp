// Torch bindings for MIV-TBE (glc_serve.miv_tbe).  Thin: checks, then the C API.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "miv_tbe.h"

static MivTbeDesc make_desc(const at::Tensor& planes, const at::Tensor& smb, const at::Tensor& esc,
                            const at::Tensor& rowparam, const at::Tensor& escbase,
                            const c10::optional<at::Tensor>& bias, int64_t N, int64_t K,
                            int64_t wpr, int64_t unroll, int64_t version) {
  TORCH_CHECK(planes.is_cuda() && smb.is_cuda() && esc.is_cuda() && rowparam.is_cuda() &&
              escbase.is_cuda(), "miv_tbe: CUDA tensors required");
  TORCH_CHECK(planes.scalar_type() == at::kInt && smb.scalar_type() == at::kByte &&
              esc.scalar_type() == at::kByte && rowparam.scalar_type() == at::kInt &&
              escbase.scalar_type() == at::kInt, "miv_tbe: dtypes");
  TORCH_CHECK(planes.is_contiguous() && smb.is_contiguous() && esc.is_contiguous() &&
              rowparam.is_contiguous() && escbase.is_contiguous(), "miv_tbe: contiguous");
  TORCH_CHECK(K % (64 * wpr) == 0, "miv_tbe: K must be a multiple of 64*WPR");
  TORCH_CHECK(planes.numel() == N * (K / 64) * 6 && smb.numel() == N * K &&
              rowparam.numel() == N && escbase.numel() == N * wpr, "miv_tbe: sizes");
  TORCH_CHECK(N < (1LL << 31) && K < (1LL << 31), "miv_tbe: shape too large");
  MivTbeDesc d{};
  d.n = (int32_t)N; d.k = (int32_t)K; d.wpr = (int32_t)wpr; d.unroll = (int32_t)unroll;
  d.version = (int32_t)version;
  d.planes = reinterpret_cast<const uint32_t*>(planes.data_ptr<int32_t>());
  d.smb = smb.data_ptr<uint8_t>();
  d.esc = esc.data_ptr<uint8_t>();
  d.rowparam = rowparam.data_ptr<int32_t>();
  d.escbase = escbase.data_ptr<int32_t>();
  d.bias = nullptr;
  if (bias.has_value() && bias->defined()) {
    TORCH_CHECK(bias->is_cuda() && bias->scalar_type() == at::kBFloat16 && bias->numel() == N &&
                bias->is_contiguous(), "miv_tbe: bias must be contiguous bf16 [N]");
    d.bias = reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr<at::BFloat16>());
  }
  return d;
}

void miv_tbe_out(at::Tensor x, at::Tensor planes, at::Tensor smb, at::Tensor esc,
                 at::Tensor rowparam, at::Tensor escbase, c10::optional<at::Tensor> bias,
                 at::Tensor y, int64_t N, int64_t K, int64_t wpr, int64_t unroll,
                 int64_t version) {
  const int64_t M = x.size(0);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 &&
              x.stride(1) == 1 && x.size(1) == K && x.stride(0) % 8 == 0 &&
              (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0, "miv_tbe: x");
  TORCH_CHECK(y.is_cuda() && y.scalar_type() == at::kBFloat16 && y.dim() == 2 &&
              y.size(0) == M && y.size(1) == N && y.stride(1) == 1, "miv_tbe: y");
  TORCH_CHECK(M >= 1 && M <= 8, "miv_tbe: 1 <= M <= 8, got ", M);
  const MivTbeDesc d = make_desc(planes, smb, esc, rowparam, escbase, bias, N, K, wpr, unroll,
                                 version);
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaError_t e = miv_tbe_gemv(d, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                                     x.stride(0), (int)M,
                                     reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
                                     y.stride(0), at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == cudaSuccess, "miv_tbe_gemv: ", cudaGetErrorString(e));
}

void miv_tbe_decode_out(at::Tensor planes, at::Tensor smb, at::Tensor esc, at::Tensor rowparam,
                        at::Tensor escbase, at::Tensor w, int64_t N, int64_t K, int64_t wpr,
                        int64_t unroll) {
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kBFloat16 && w.is_contiguous() &&
              w.numel() == N * K, "miv_tbe_decode: w");
  const MivTbeDesc d = make_desc(planes, smb, esc, rowparam, escbase, c10::nullopt, N, K, wpr,
                                 unroll, 2);
  const c10::cuda::CUDAGuard guard(w.device());
  const cudaError_t e = miv_tbe_decode(d, reinterpret_cast<__nv_bfloat16*>(w.data_ptr<at::BFloat16>()),
                                       at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == cudaSuccess, "miv_tbe_decode: ", cudaGetErrorString(e));
}

at::Tensor miv_tbe_tile_escapes_t(at::Tensor planes) {
  TORCH_CHECK(planes.is_cuda() && planes.scalar_type() == at::kInt && planes.is_contiguous() &&
              planes.numel() % 6 == 0, "tile_escapes: int32 [T*6] CUDA");
  const int64_t T = planes.numel() / 6;
  auto out = at::empty({T}, planes.options());
  const c10::cuda::CUDAGuard guard(planes.device());
  const cudaError_t e = miv_tbe_tile_escapes(reinterpret_cast<const uint32_t*>(planes.data_ptr<int32_t>()),
                                             T, out.data_ptr<int32_t>(),
                                             at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == cudaSuccess, "tile_escapes: ", cudaGetErrorString(e));
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("miv_tbe_out", &miv_tbe_out, "fused TBE-decode MIV GEMV into y");
  m.def("miv_tbe_decode_out", &miv_tbe_decode_out, "decode-only twin into w");
  m.def("tile_escapes", &miv_tbe_tile_escapes_t, "escapes per 64-element tile");
}
