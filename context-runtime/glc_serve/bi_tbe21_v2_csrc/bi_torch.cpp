// Dedicated, TBE21-only binding for the isolated kernel-v2 extension.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "bi_tbe21_v2_api.h"

namespace {
static std::vector<int64_t> tbe21_v2_bi_kernel_attrs(int64_t rows, int64_t device_index) {
  TORCH_CHECK(rows > 0 && rows <= std::numeric_limits<int>::max(),
              "bi_tbe21_v2 attrs: rows must be a positive int32 value");
  TORCH_CHECK(device_index >= 0 &&
              device_index <= std::numeric_limits<c10::DeviceIndex>::max(),
              "bi_tbe21_v2 attrs: device must be a valid CUDA DeviceIndex");
  const auto index = static_cast<c10::DeviceIndex>(device_index);
  const c10::Device target(c10::DeviceType::CUDA, index);
  const c10::cuda::CUDAGuard guard(target);
  BiKernelAttrs attrs{};
  const cudaError_t e = bi_tbe21_v2_runtime_attrs(static_cast<int>(rows), &attrs);
  TORCH_CHECK(e == cudaSuccess, "bi_tbe21_v2 attrs: ", cudaGetErrorString(e));
  return {attrs.regs, attrs.local_bytes, attrs.max_blocks_per_sm,
          attrs.smem_bytes, attrs.threads};
}

template <typename T>
static const T* ptr_or_null(const std::vector<at::Tensor>& a, size_t i,
                            const c10::Device& device) {
  if (i >= a.size() || !a[i].defined() || a[i].numel() == 0) return nullptr;
  TORCH_CHECK(a[i].is_cuda() && a[i].device() == device && a[i].is_contiguous(),
              "bi_tbe21_v2: array ", i, " must be contiguous CUDA on the input device");
  return reinterpret_cast<const T*>(a[i].data_ptr());
}

// Exact TBE21 array ABI and checks from pinned 90e83dd8: [planes, smb, ovf, rowoff,
// cb, rowp2, ckpt]. Do not reinterpret or normalize codec payload bytes here.
static void fill_tbe21(BiDesc& d, int64_t N, int64_t K, int64_t S,
                       const std::vector<at::Tensor>& arrs, const c10::Device& device,
                       bool require_tier = true) {
  TORCH_CHECK(K % 128 == 0, "tbe21: K % 128");
  const int64_t nch = K / 128;
  d.planes = ptr_or_null<uint32_t>(arrs, 0, device); d.smb = ptr_or_null<uint8_t>(arrs, 1, device);
  d.ovf = ptr_or_null<uint32_t>(arrs, 2, device); d.rowoff = ptr_or_null<uint32_t>(arrs, 3, device);
  d.cb = ptr_or_null<uint8_t>(arrs, 4, device); d.rowp2 = ptr_or_null<int32_t>(arrs, 5, device);
  d.ckpt = ptr_or_null<uint32_t>(arrs, 6, device);
  TORCH_CHECK(d.planes && d.smb && d.ovf && d.rowoff && d.cb && d.rowp2,
              "tbe21 arrays");
  TORCH_CHECK(arrs[0].numel() * arrs[0].element_size() == N * nch * 32,
              "tbe21 l1 size");
  TORCH_CHECK(arrs[1].numel() * arrs[1].element_size() == N * K,
              "tbe21 smb size");
  TORCH_CHECK(arrs[2].numel() * arrs[2].element_size() >= 16 * 4,
              "tbe21 ovf lacks its 16-word tail");
  TORCH_CHECK(arrs[3].numel() * arrs[3].element_size() == N * 4,
              "tbe21 rowoff size");
  const int64_t cbb = arrs[4].numel() * arrs[4].element_size();
  TORCH_CHECK(cbb % 16 == 0 && cbb >= 16 && cbb <= 16 * 8,
              "tbe21 codebooks: 1..8 x 16 bytes");
  TORCH_CHECK(arrs[5].numel() == N && arrs[5].scalar_type() == at::kInt,
              "tbe21 rowp size");
  const int64_t ckb = arrs.size() > 6 && arrs[6].defined()
      ? arrs[6].numel() * arrs[6].element_size() : 0;
  TORCH_CHECK(ckb == N * (S - 1) * 4 && (S == 1 || d.ckpt),
              "tbe21 checkpoints must be built for S=", S);
  TORCH_CHECK((reinterpret_cast<uintptr_t>(d.smb) & 7) == 0 &&
              (reinterpret_cast<uintptr_t>(d.cb) & 15) == 0,
              "tbe21 stream alignment");
  d.ncb = (int)(cbb / 16); d.ckS = (int)S;
#if BI_T21_TIER_PREFIX
  if (require_tier) {
    TORCH_CHECK(arrs.size() == 8 && arrs[7].defined() && arrs[7].is_cuda() &&
                arrs[7].device() == device && arrs[7].is_contiguous() &&
                arrs[7].scalar_type() == at::kInt && arrs[7].dim() == 2 &&
                arrs[7].size(0) == N && arrs[7].size(1) == nch,
                "tbe21 tier-prefix requires prepared contiguous CUDA int32 headers [N,K/128]");
    d.tpf_counts = reinterpret_cast<const uint32_t*>(arrs[7].data_ptr<int32_t>());
  }
#else
  (void)require_tier;
#endif
}
}  // namespace

at::Tensor tbe21_make_tier_headers(int64_t N, int64_t K,
                                  std::vector<at::Tensor> arrs, int64_t S) {
  TORCH_CHECK(bi_tbe21_tier_prefix(), "TBE21 tier-prefix metadata is not compiled in");
  TORCH_CHECK(N > 0 && K > 0 && N <= std::numeric_limits<int>::max() &&
              K <= std::numeric_limits<int>::max() && K % 256 == 0 &&
              S >= 1 && S <= K / 128,
              "TBE21 metadata: positive int32 N/K, K % 256, 1 <= S <= K/128");
  TORCH_CHECK(arrs.size() == 7 && arrs[0].defined() && arrs[0].is_cuda(),
              "TBE21 metadata requires seven base CUDA arrays");
  const at::cuda::CUDAGuard guard(arrs[0].device());
  auto st = at::cuda::getCurrentCUDAStream();
  cudaStreamCaptureStatus capture;
  const cudaError_t capture_error = cudaStreamIsCapturing(st, &capture);
  TORCH_CHECK(capture_error == cudaSuccess, "TBE21 metadata capture query: ",
              cudaGetErrorString(capture_error));
  TORCH_CHECK(capture == cudaStreamCaptureStatusNone,
              "TBE21 tier metadata must be prepared before CUDA graph capture");
  BiDesc d{};
  d.N = static_cast<int>(N); d.K = static_cast<int>(K);
  fill_tbe21(d, N, K, S, arrs, arrs[0].device(), false);
  auto out = at::empty({N, K / 128}, arrs[0].options().dtype(at::kInt));
  const cudaError_t e = bi_tbe21_make_tier_headers(
      d, reinterpret_cast<uint32_t*>(out.data_ptr<int32_t>()), st);
  TORCH_CHECK(e == cudaSuccess, "TBE21 metadata launch: ", cudaGetErrorString(e));
  return out;
}

// Match public BILin's existing call ABI; reject all other formats/variants locally.
void tbe21_v2_bi_gemm_out(at::Tensor x, int64_t M, int64_t fmt, int64_t kqt,
                 int64_t N, int64_t K, std::vector<at::Tensor> arrs,
                 c10::optional<at::Tensor> bias, at::Tensor y, int64_t S,
                 at::Tensor ws, int64_t tile) {
  TORCH_CHECK(fmt == BI_TBE21 && kqt == 0,
              "bi_tbe21_v2 supports only TBE21 format code 5");
  if (bi_tbe21_tile128()) {
    TORCH_CHECK(tile == bi_tbe21_tile_for(M), "bi_tbe21_v2 tile differs from compiled row policy");
  } else {
    TORCH_CHECK(tile == 64, "bi_tbe21_v2 requires tile == 64");
  }
  TORCH_CHECK(N > 0 && K > 0 && N <= std::numeric_limits<int>::max() &&
              K <= std::numeric_limits<int>::max() && M >= 1 &&
              M <= std::numeric_limits<int>::max() && S >= 1 &&
              S <= std::numeric_limits<int>::max(),
              "bi_tbe21_v2: N, K, M, S must be positive int32 values");
  TORCH_CHECK(x.dim() == 2, "bi_gemm: x must be rank 2");
  TORCH_CHECK(y.dim() == 2, "bi_gemm: y must be rank 2");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 &&
              x.stride(0) >= K && x.stride(0) % 8 == 0 &&
              (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0 &&
              x.size(0) >= M && x.size(1) >= K, "bi_gemm: x");
  TORCH_CHECK(y.is_cuda() && y.device() == x.device() &&
              y.scalar_type() == at::kBFloat16 && y.stride(1) == 1 && y.stride(0) >= N &&
              y.size(0) >= M && y.size(1) >= N, "bi_gemm: y");
  TORCH_CHECK(K % 256 == 0 && M >= 1 && S >= 1 && S <= K / 128,
              "bi_gemm: K % 256, M >= 1, 1 <= S <= K/128");
  TORCH_CHECK(S == 1 || (S <= std::numeric_limits<int64_t>::max() / (M * N) && ws.is_cuda() &&
                         ws.device() == x.device() && ws.is_contiguous() &&
                         ws.scalar_type() == at::kFloat && ws.numel() >= S * M * N),
              "bi_gemm: workspace too small or invalid");
  const at::cuda::CUDAGuard guard(x.device());
  BiDesc d{};
  d.N = (int)N; d.K = (int)K;
  fill_tbe21(d, N, K, S, arrs, x.device());
  d.bias = nullptr;
  if (bias.has_value() && bias->defined()) {
    TORCH_CHECK(bias->is_cuda() && bias->device() == x.device() &&
                bias->is_contiguous() && bias->scalar_type() == at::kBFloat16 &&
                bias->numel() == N, "bi_gemm: bias");
    d.bias = reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr());
  }
  auto st = at::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto* yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  float* wsp = S > 1 ? ws.data_ptr<float>() : nullptr;
  const cudaError_t e = bi_tbe21_v2_gemm((int)fmt, xp, x.stride(0), (int)M, d,
                                          (int)S, yp, y.stride(0), wsp, st,
                                          (int)tile, 0);
  TORCH_CHECK(e == cudaSuccess, "bi_gemm v2 launch: ", cudaGetErrorString(e));
}

// G1 must invoke this DSO's selected decode body, including PIPE128's swizzle.
// Calling the legacy extension here would test a different implementation.
void tbe21_v2_bi_decode_out(int64_t fmt, int64_t N, int64_t K,
                          std::vector<at::Tensor> arrs, at::Tensor out, int64_t S) {
  TORCH_CHECK(fmt == BI_TBE21, "bi_tbe21_v2 decode: only TBE21 format code 5");
  TORCH_CHECK(N > 0 && K > 0 && N <= std::numeric_limits<int>::max() &&
              K <= std::numeric_limits<int>::max() && K % 256 == 0 &&
              S >= 1 && S <= K / 128,
              "bi_tbe21_v2 decode: positive int32 N/K, K % 256, 1 <= S <= K/128");
  TORCH_CHECK(out.defined() && out.is_cuda() && out.scalar_type() == at::kBFloat16 &&
              out.dim() == 2 && out.size(0) == N && out.size(1) == K && out.is_contiguous() &&
              (reinterpret_cast<uintptr_t>(out.data_ptr()) & 15) == 0,
              "bi_tbe21_v2 decode: aligned contiguous CUDA BF16 output [N,K] required");
  const at::cuda::CUDAGuard guard(out.device());
  BiDesc d{};
  d.N = static_cast<int>(N); d.K = static_cast<int>(K);
  fill_tbe21(d, N, K, S, arrs, out.device());
  auto st = at::cuda::getCurrentCUDAStream();
  auto* output = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  const cudaError_t e = bi_tbe21_v2_decode(BI_TBE21, d, static_cast<int>(S), output, st, 0);
  TORCH_CHECK(e == cudaSuccess, "bi_tbe21_v2 decode launch: ", cudaGetErrorString(e));
}

#if BI_T21_EXEC_LAYOUT
void tbe21_v2_bi_decode_exec_out(int64_t fmt, int64_t N, int64_t K,
                               std::vector<at::Tensor> arrs, at::Tensor out,
                               int64_t S, int64_t rows) {
  TORCH_CHECK(bi_tbe21_exec_layout() && fmt == BI_TBE21 && (rows == 64 || rows == 128),
              "EXEC raw decode requires compiled EXEC layout, TBE21 and rows64/128");
  TORCH_CHECK(N > 0 && K > 0 && N <= std::numeric_limits<int>::max() &&
              K <= std::numeric_limits<int>::max() && K % 256 == 0 && S >= 1 && S <= K / 128,
              "EXEC raw decode dimensions/splits");
  TORCH_CHECK(arrs.size() == 7, "EXEC raw decode requires unchanged seven-array ABI");
  TORCH_CHECK(out.defined() && out.is_cuda() && out.scalar_type() == at::kBFloat16 &&
              out.dim() == 2 && out.size(0) == N && out.size(1) == K && out.is_contiguous() &&
              (reinterpret_cast<uintptr_t>(out.data_ptr()) & 15) == 0,
              "EXEC raw decode requires aligned contiguous CUDA BF16 output [N,K]");
  const at::cuda::CUDAGuard guard(out.device());
  BiDesc d{}; d.N = static_cast<int>(N); d.K = static_cast<int>(K);
  fill_tbe21(d, N, K, S, arrs, out.device(), false);
  const cudaError_t e = bi_tbe21_v2_decode_exec(d, static_cast<int>(S),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), at::cuda::getCurrentCUDAStream(), static_cast<int>(rows));
  TORCH_CHECK(e == cudaSuccess, "EXEC raw decode launch: ", cudaGetErrorString(e));
}
#endif

void tbe21_bi_decode_prefix_out(int64_t fmt, int64_t N, int64_t K,
                               std::vector<at::Tensor> arrs, at::Tensor out,
                               int64_t S, int64_t wg) {
  TORCH_CHECK(bi_tbe21_tier_prefix() && fmt == BI_TBE21,
              "TBE21 prefix decode requires the compiled TBE21 tier-prefix path");
  TORCH_CHECK(wg == 2 || (wg == 1 && bi_tbe21_tier_prefix_wg1()),
              "TBE21 tier-prefix decode geometry is not enabled");
  TORCH_CHECK(N > 0 && K > 0 && N <= std::numeric_limits<int>::max() &&
              K <= std::numeric_limits<int>::max() && K % 256 == 0 &&
              S >= 1 && S <= K / 128, "TBE21 prefix decode dimensions/splits");
  TORCH_CHECK(out.defined() && out.is_cuda() && out.scalar_type() == at::kBFloat16 &&
              out.dim() == 2 && out.size(0) == N && out.size(1) == K &&
              out.is_contiguous() && (reinterpret_cast<uintptr_t>(out.data_ptr()) & 15) == 0,
              "TBE21 prefix decode requires aligned contiguous CUDA BF16 output [N,K]");
  const at::cuda::CUDAGuard guard(out.device());
  BiDesc d{};
  d.N = static_cast<int>(N); d.K = static_cast<int>(K);
  fill_tbe21(d, N, K, S, arrs, out.device());
  const cudaError_t e = bi_tbe21_v2_decode_prefix(d, static_cast<int>(S),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      at::cuda::getCurrentCUDAStream(), static_cast<int>(wg));
  TORCH_CHECK(e == cudaSuccess, "TBE21 prefix decode launch: ", cudaGetErrorString(e));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tier_prefix", &bi_tbe21_tier_prefix);
  m.def("tier_prefetch", &bi_tbe21_tier_prefetch);
  m.def("tier_prefix_wg1", &bi_tbe21_tier_prefix_wg1);
  m.def("tier_prefetch_wg1", &bi_tbe21_tier_prefetch_wg1);
  m.def("tier_prefix_host_abi", []() { return BI_T21_TIER_PREFIX == 1; });
  m.def("make_tier_headers", &tbe21_make_tier_headers,
        pybind11::arg("N"), pybind11::arg("K"), pybind11::arg("arrs"), pybind11::arg("S"));
  m.def("bi_decode_prefix_out", &tbe21_bi_decode_prefix_out,
        pybind11::arg("fmt"), pybind11::arg("N"), pybind11::arg("K"),
        pybind11::arg("arrs"), pybind11::arg("out"), pybind11::arg("S"), pybind11::arg("wg"));
  m.def("bi_decode_out", &tbe21_v2_bi_decode_out, "Selected TBE21 decode body for G1",
        pybind11::arg("fmt"), pybind11::arg("N"), pybind11::arg("K"),
        pybind11::arg("arrs"), pybind11::arg("out"), pybind11::arg("S"));
  m.def("async_x", &bi_tbe21_async_x, "Compiled TBE21 async-X activation staging");
#if BI_T21_EXEC_LAYOUT
  m.def("exec_decode_oracle", &bi_tbe21_exec_layout, "Compiled selected EXEC raw-word oracle");
  m.def("bi_decode_exec_out", &tbe21_v2_bi_decode_exec_out,
        "Raw words from the selected EXEC GEMM producer and XOR tile",
        pybind11::arg("fmt"), pybind11::arg("N"), pybind11::arg("K"),
        pybind11::arg("arrs"), pybind11::arg("out"), pybind11::arg("S"), pybind11::arg("rows"));
#endif
  m.def("tile128", &bi_tbe21_tile128, "Compiled selective TBE21 tile128 policy");
  m.def("tile_for", &bi_tbe21_tile_for, "Compiled TBE21 tile selected for row count");
  m.def("row_geometry", &bi_tbe21_row_geometry, "Compiled TBE21 row geometry dispatch");
  m.def("exec_layout", &bi_tbe21_exec_layout, "Compiled TBE21 exact-M97..128 execution layout");
  m.def("bi_kernel_attrs", &tbe21_v2_bi_kernel_attrs,
        "Runtime attributes for the compiled TBE21 GEMM row specialization",
        pybind11::arg("rows"), pybind11::arg("device"));
  m.def("packed_scan2", &bi_tbe21_packed_scan2, "Compiled TBE21 paired scan");
  m.def("scan4", &bi_tbe21_scan4, "Compiled TBE21 four-round paired scan");
  m.def("wide128", &bi_tbe21_wide128, "Compiled TBE21 M128 four-group geometry");
  m.def("pipe128", &bi_tbe21_pipe128, "Compiled TBE21 M128 producer-consumer pipeline");
  m.def("packed_scan4", &bi_tbe21_packed_scan4, "Compiled TBE21 four-row byte-packed scan");
  m.def("pair_vote", &bi_tbe21_pair_vote, "Compiled TBE21 large-batch paired ballot prefix");
  m.def("kernel_variant", &bi_tbe21_kernel_variant, "Compiled TBE21 kernel schedule");
  m.def("bi_gemm_out", &tbe21_v2_bi_gemm_out, "TBE21 phase-split BI-GEMM",
        pybind11::arg("x"), pybind11::arg("M"), pybind11::arg("fmt"),
        pybind11::arg("kqt"), pybind11::arg("N"), pybind11::arg("K"),
        pybind11::arg("arrs"), pybind11::arg("bias"), pybind11::arg("y"),
        pybind11::arg("S"), pybind11::arg("ws"), pybind11::arg("tile") = 64);
}
