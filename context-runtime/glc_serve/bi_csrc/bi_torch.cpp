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

// TBE2 (codec v2) arrays: [planes i32 N*nch*8, smb u8 N*K, ovf i32 (+>=16 tail words),
// len u8 N*nchp (+16 tail bytes), grp i32 N*ngrp, cb u8 ncb*16, rowp i32 N].  nchp / ngrp are
// functions of K (nch = K/128, nchp = ceil4(nch), ngrp = ceil(nch/16)); ncb = cb.numel() / 16.
static void fill_tbe2(BiDesc& d, int64_t N, int64_t K, const std::vector<at::Tensor>& arrs) {
  TORCH_CHECK(K % 128 == 0, "tbe2: K % 128");
  const int64_t nch = K / 128, nchp = (nch + 3) / 4 * 4, ngrp = (nch + 15) / 16;
  d.planes = ptr_or_null<uint32_t>(arrs, 0); d.smb = ptr_or_null<uint8_t>(arrs, 1);
  d.ovf = ptr_or_null<uint32_t>(arrs, 2); d.len = ptr_or_null<uint8_t>(arrs, 3);
  d.grp = ptr_or_null<uint32_t>(arrs, 4); d.cb = ptr_or_null<uint8_t>(arrs, 5);
  d.rowp2 = ptr_or_null<int32_t>(arrs, 6);
  TORCH_CHECK(d.planes && d.smb && d.ovf && d.len && d.grp && d.cb && d.rowp2, "tbe2 arrays");
  TORCH_CHECK(arrs[0].numel() * arrs[0].element_size() == N * nch * 32, "tbe2 planes size");
  TORCH_CHECK(arrs[1].numel() * arrs[1].element_size() == N * K, "tbe2 smb size");
  TORCH_CHECK(arrs[2].numel() * arrs[2].element_size() >= 16 * 4, "tbe2 ovf lacks its 16-word tail");
  TORCH_CHECK(arrs[3].numel() * arrs[3].element_size() >= N * nchp + 16, "tbe2 len lacks its 16-byte tail");
  TORCH_CHECK(arrs[4].numel() * arrs[4].element_size() == N * ngrp * 4, "tbe2 grp size");
  const int64_t cbb = arrs[5].numel() * arrs[5].element_size();
  TORCH_CHECK(cbb % 16 == 0 && cbb >= 16 && cbb <= 16 * 8, "tbe2 codebooks: 1..8 x 16 bytes");
  TORCH_CHECK(arrs[6].numel() == N && arrs[6].scalar_type() == at::kInt, "tbe2 rowp size");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(d.smb) & 7) == 0 && (reinterpret_cast<uintptr_t>(d.len) & 3) == 0 &&
              (reinterpret_cast<uintptr_t>(d.cb) & 15) == 0, "tbe2 stream alignment");
  d.nchp = (int)nchp; d.ngrp = (int)ngrp; d.ncb = (int)(cbb / 16);
}

// TBE21 (codec v2.1) arrays: [l1 i16 N*nch*16, smb u8 N*K, ovf i32 (+>=16 tail words), rowoff i32 N,
// cb u8 ncb*16, rowp i32 N, ckpt i32 N*(S-1) (empty when S == 1)].  The checkpoints are RESIDENT,
// built at load for the launch split count S (glc_loader.codec_v21.split_checkpoints).
static void fill_tbe21(BiDesc& d, int64_t N, int64_t K, int64_t S, const std::vector<at::Tensor>& arrs) {
  TORCH_CHECK(K % 128 == 0, "tbe21: K % 128");
  const int64_t nch = K / 128;
  d.planes = ptr_or_null<uint32_t>(arrs, 0); d.smb = ptr_or_null<uint8_t>(arrs, 1);
  d.ovf = ptr_or_null<uint32_t>(arrs, 2); d.rowoff = ptr_or_null<uint32_t>(arrs, 3);
  d.cb = ptr_or_null<uint8_t>(arrs, 4); d.rowp2 = ptr_or_null<int32_t>(arrs, 5);
  d.ckpt = ptr_or_null<uint32_t>(arrs, 6);
  TORCH_CHECK(d.planes && d.smb && d.ovf && d.rowoff && d.cb && d.rowp2, "tbe21 arrays");
  TORCH_CHECK(arrs[0].numel() * arrs[0].element_size() == N * nch * 32, "tbe21 l1 size");
  TORCH_CHECK(arrs[1].numel() * arrs[1].element_size() == N * K, "tbe21 smb size");
  TORCH_CHECK(arrs[2].numel() * arrs[2].element_size() >= 16 * 4, "tbe21 ovf lacks its 16-word tail");
  TORCH_CHECK(arrs[3].numel() * arrs[3].element_size() == N * 4, "tbe21 rowoff size");
  const int64_t cbb = arrs[4].numel() * arrs[4].element_size();
  TORCH_CHECK(cbb % 16 == 0 && cbb >= 16 && cbb <= 16 * 8, "tbe21 codebooks: 1..8 x 16 bytes");
  TORCH_CHECK(arrs[5].numel() == N && arrs[5].scalar_type() == at::kInt, "tbe21 rowp size");
  const int64_t ckb = arrs.size() > 6 && arrs[6].defined() ? arrs[6].numel() * arrs[6].element_size() : 0;
  TORCH_CHECK(ckb == N * (S - 1) * 4 && (S == 1 || d.ckpt), "tbe21 checkpoints must be built for S=", S);
  TORCH_CHECK((reinterpret_cast<uintptr_t>(d.smb) & 7) == 0 && (reinterpret_cast<uintptr_t>(d.cb) & 15) == 0,
              "tbe21 stream alignment");
  d.ncb = (int)(cbb / 16); d.ckS = (int)S;
}

// fmt: 0 bf16 [w]; 1 q8 [qs, sc]; 2 tbe [planes, smb, esc, rowparam, escidx]; 3 kq [a0..a4, grid, ksigns];
// 4 tbe2 (see fill_tbe2); 5 tbe21 (see fill_tbe21)
void bi_gemm_out(at::Tensor x, int64_t M, int64_t fmt, int64_t kqt, int64_t N, int64_t K,
                 std::vector<at::Tensor> arrs, c10::optional<at::Tensor> bias, at::Tensor y,
                 int64_t S, at::Tensor ws, int64_t tile) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 &&
              x.stride(0) % 8 == 0 && (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0 &&
              x.size(0) >= M && x.size(1) >= K, "bi_gemm: x");
  TORCH_CHECK(y.is_cuda() && y.scalar_type() == at::kBFloat16 && y.stride(1) == 1 && y.size(0) >= M &&
              y.size(1) >= N, "bi_gemm: y");
  TORCH_CHECK(tile == 64 || tile == 128, "bi_gemm: tile must be 64 or 128, got ", tile);
  TORCH_CHECK(K % 256 == 0 && M >= 1 && S >= 1 && S <= K / 128, "bi_gemm: K % 256, M >= 1, 1 <= S <= K/128");
  TORCH_CHECK(S == 1 || (ws.is_cuda() && ws.scalar_type() == at::kFloat && ws.numel() >= S * M * N),
              "bi_gemm: workspace too small");
  const at::cuda::CUDAGuard guard(x.device());
  BiDesc d{};
  d.N = (int)N; d.K = (int)K;
  switch (fmt) {
    case BI_BF16: d.w = ptr_or_null<__nv_bfloat16>(arrs, 0); TORCH_CHECK(d.w, "bf16 w"); break;
    case BI_Q8:
#ifdef GEOR_REFINED_CONTEXT_ONLY
      TORCH_CHECK(false, "bi_gemm: Q8/GGUF is unavailable in the BF16/TBE-only Context build");
#else
      d.qs = ptr_or_null<int8_t>(arrs, 0); d.sc = ptr_or_null<unsigned short>(arrs, 1);
      TORCH_CHECK(d.qs && d.sc, "q8 arrays"); break;
#endif
    case BI_TBE:
      d.planes = ptr_or_null<uint32_t>(arrs, 0); d.smb = ptr_or_null<uint8_t>(arrs, 1);
      d.esc = ptr_or_null<uint8_t>(arrs, 2); d.rowparam = ptr_or_null<int32_t>(arrs, 3);
      d.escidx = ptr_or_null<int32_t>(arrs, 4);
      TORCH_CHECK(d.planes && d.smb && d.esc && d.rowparam && d.escidx, "tbe arrays");
      TORCH_CHECK(arrs[4].numel() == N * (K / 128), "tbe escidx size");
      break;
#ifndef GEOR_REFINED_CONTEXT_ONLY
    case BI_KQ:
      d.a0 = ptr_or_null<uint8_t>(arrs, 0); d.a1 = ptr_or_null<uint8_t>(arrs, 1);
      d.a2 = ptr_or_null<uint8_t>(arrs, 2); d.a3 = ptr_or_null<uint8_t>(arrs, 3);
      d.a4 = ptr_or_null<uint8_t>(arrs, 4); d.grid = ptr_or_null<uint32_t>(arrs, 5);
      d.ksigns = ptr_or_null<uint8_t>(arrs, 6);
      break;
#else
    case 3: TORCH_CHECK(false, "bi_gemm: KQ is unavailable in the BF16/TBE-only Context build");
#endif
    case BI_TBE2: fill_tbe2(d, N, K, arrs); break;
    case BI_TBE21: fill_tbe21(d, N, K, S, arrs); break;
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
#ifndef GEOR_REFINED_CONTEXT_ONLY
  cudaError_t e = fmt == BI_KQ
      ? bi_gemm_kq((int)kqt, xp, x.stride(0), (int)M, d, (int)S, yp, y.stride(0), wsp, st, (int)tile)
      : bi_gemm_dense((int)fmt, xp, x.stride(0), (int)M, d, (int)S, yp, y.stride(0), wsp, st,
                      (int)tile);
#else
  TORCH_CHECK(fmt != 3, "bi_gemm: KQ is unavailable in the BF16/TBE-only Context build");
  cudaError_t e = bi_gemm_dense((int)fmt, xp, x.stride(0), (int)M, d, (int)S, yp, y.stride(0), wsp,
                                st, (int)tile);
#endif
  TORCH_CHECK(e == cudaSuccess, "bi_gemm launch: ", cudaGetErrorString(e));
}

// Decode-only: the weight as bi_gemm_kernel's decode produces it, out[N][K] bf16, through the
// same CTA mapping and split traversal (S) -- the G1 gate's device side.  fmt 0 / 2 / 4 / 5.
void bi_decode_out(int64_t fmt, int64_t N, int64_t K, std::vector<at::Tensor> arrs, at::Tensor out,
                   int64_t S, int64_t wg) {
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.is_contiguous() &&
              out.numel() == N * K, "bi_decode: out");
  TORCH_CHECK(K % 128 == 0 && S >= 1 && S <= K / 128 && (wg == 1 || wg == 2), "bi_decode: K, S, wg");
  const at::cuda::CUDAGuard guard(out.device());
  BiDesc d{};
  d.N = (int)N; d.K = (int)K;
  switch (fmt) {
    case BI_BF16: d.w = ptr_or_null<__nv_bfloat16>(arrs, 0); TORCH_CHECK(d.w, "bf16 w"); break;
    case BI_TBE:
      d.planes = ptr_or_null<uint32_t>(arrs, 0); d.smb = ptr_or_null<uint8_t>(arrs, 1);
      d.esc = ptr_or_null<uint8_t>(arrs, 2); d.rowparam = ptr_or_null<int32_t>(arrs, 3);
      d.escidx = ptr_or_null<int32_t>(arrs, 4);
      TORCH_CHECK(d.planes && d.smb && d.esc && d.rowparam && d.escidx, "tbe arrays");
      break;
    case BI_TBE2: fill_tbe2(d, N, K, arrs); break;
    case BI_TBE21: fill_tbe21(d, N, K, S, arrs); break;
    default: TORCH_CHECK(false, "bi_decode: fmt");
  }
  cudaError_t e = bi_decode_dense((int)fmt, d, (int)S,
                                  reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                                  at::cuda::getCurrentCUDAStream(), (int)wg);
  TORCH_CHECK(e == cudaSuccess, "bi_decode launch: ", cudaGetErrorString(e));
}

// TBE21 load-time checkpoints: arrs as for fmt 5 with an EMPTY ckpt (index 6 absent or empty);
// out = int32 [N*(S-1)] receives the row-relative bit offsets of every split start.
void bi_t21_ckpt_out(int64_t N, int64_t K, std::vector<at::Tensor> arrs, int64_t S, at::Tensor out) {
  TORCH_CHECK(K % 128 == 0 && S >= 1 && S <= K / 128, "bi_t21_ckpt: K, S");
  TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == at::kInt &&
              out.numel() == N * (S - 1), "bi_t21_ckpt: out");
  const at::cuda::CUDAGuard guard(out.device());
  BiDesc d{};
  d.N = (int)N; d.K = (int)K;
  std::vector<at::Tensor> a(arrs.begin(), arrs.begin() + std::min<size_t>(arrs.size(), 6));
  fill_tbe21(d, N, K, 1, a);                       // S = 1: the walk needs no checkpoints
  cudaError_t e = bi_t21_checkpoints(d, (int)S, reinterpret_cast<uint32_t*>(out.data_ptr()),
                                     at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == cudaSuccess, "bi_t21_ckpt launch: ", cudaGetErrorString(e));
}

std::vector<int64_t> bi_kernel_attrs(int64_t fmt, int64_t mt, int64_t wg) {
  BiKernelAttrs a{};
  cudaError_t e = bi_attrs_dense((int)fmt, (int)mt, (int)wg, &a);
  TORCH_CHECK(e == cudaSuccess, "bi_kernel_attrs: ", cudaGetErrorString(e));
  return {a.regs, a.local_bytes, a.max_blocks_per_sm, a.smem_bytes, a.threads};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bi_t21_ckpt_out", &bi_t21_ckpt_out, "TBE21 load-time split checkpoints",
        pybind11::arg("N"), pybind11::arg("K"), pybind11::arg("arrs"), pybind11::arg("S"), pybind11::arg("out"));
  m.def("bi_decode_out", &bi_decode_out, "BI-GEMM decode-only twin (G1)",
        pybind11::arg("fmt"), pybind11::arg("N"), pybind11::arg("K"), pybind11::arg("arrs"),
        pybind11::arg("out"), pybind11::arg("S"), pybind11::arg("wg") = 1);
  m.def("bi_kernel_attrs", &bi_kernel_attrs, "regs, local bytes, CTAs/SM, smem, threads",
        pybind11::arg("fmt"), pybind11::arg("mt"), pybind11::arg("wg"));
  m.def("bi_gemm_out", &bi_gemm_out, "batch-invariant tensor-core GEMM (in-kernel weight decode)",
        pybind11::arg("x"), pybind11::arg("M"), pybind11::arg("fmt"), pybind11::arg("kqt"),
        pybind11::arg("N"), pybind11::arg("K"), pybind11::arg("arrs"), pybind11::arg("bias"),
        pybind11::arg("y"), pybind11::arg("S"), pybind11::arg("ws"), pybind11::arg("tile") = 64);
}
