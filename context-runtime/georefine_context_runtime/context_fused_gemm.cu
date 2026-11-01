#include <torch/extension.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "context_fused_gemm.cuh"
#include "bi_gemm.cuh"

// Same per-output accumulator chain and epilogue as bi_gemm_kernel, S=1.
// Only the producer changes. Ws uses the baseline BI stride 136; BCT2's explicit
// physical output stride writes it directly. No decoded weight HBM exists.
// This first correctness kernel has no producer/consumer overlap or speed claim.
template<int MT, bool Optimized=false, bool Panels=false, bool LUT=false>
__global__ void bcf_kernel(const uint8_t* data, uint64_t size, const int64_t* desc,
    const __nv_bfloat16* x, uint32_t K, uint32_t M, uint32_t nr, uint32_t nc,
    uint64_t labels, uint64_t ts, uint64_t nt,
    const uint32_t* coarse, uint64_t coarse_count, const uint64_t* records,
    uint64_t record_count, const uint32_t* offsets, uint64_t offset_count,
    __nv_bfloat16* y, int32_t* status) {
  extern __shared__ __align__(16) unsigned char storage[];
  auto* Ws=reinterpret_cast<__nv_bfloat16*>(storage);
  static_assert(bcf::LDS==bi::LDS, "decoder and BI shared strides must match");
  auto* Xs=Ws+bcf::N*bi::LDS;
  __shared__ uint32_t failed;
  const int warp=threadIdx.x>>5, lane=threadIdx.x&31;
  const int m0=blockIdx.x*64;
  const uint32_t panel=Panels ? blockIdx.y : 0;
  const uint32_t output_stride=Panels ? gridDim.y*bcf::N : bcf::N;
  if constexpr (Panels)
    status+=uint64_t(panel)*gridDim.x*(K/bcf::CH)*128;
  float acc[MT][4];
#pragma unroll
  for (int mt=0;mt<MT;++mt)
#pragma unroll
    for (int e=0;e<4;++e) acc[mt][e]=0.0f;
  for (uint32_t c=0;c<K/bcf::CH;++c) {
    if (threadIdx.x==0) failed=0;
    __syncthreads();
    // Four independent tile streams, each with 32 entropy states. Other warps
    // load activations concurrently; all threads join before the MMA consumer.
    if (warp<4) {
      uint32_t code;
      if constexpr (LUT && Panels)
        code=bcf::decode_panel_slab_lane_lut(data,size*gridDim.y,desc,
            uint64_t(gridDim.y)*4*(K/bcf::CH)*9,gridDim.y,panel,size,K,nr,nc,
            labels,ts,nt,c,warp,lane,reinterpret_cast<uint16_t*>(Ws),coarse,
            coarse_count,records,record_count,offsets,offset_count);
      else if constexpr (LUT)
        code=bcf::decode_slab_lane_lut(data,size,desc+9*(4*c+warp),K,nr,nc,
            labels,ts,nt,c,warp,lane,reinterpret_cast<uint16_t*>(Ws),coarse,
            coarse_count,records,record_count,offsets,offset_count);
      else if constexpr (Panels)
        code=bcf::decode_panel_slab_lane(data,size*gridDim.y,desc,
            uint64_t(gridDim.y)*4*(K/bcf::CH)*9,gridDim.y,panel,size,K,nr,nc,
            labels,ts,nt,c,warp,lane,reinterpret_cast<uint16_t*>(Ws));
      else if constexpr (Optimized)
        code=bcf::decode_slab_lane_opt(data,size,desc+9*(4*c+warp),K,nr,nc,
            labels,ts,nt,c,warp,lane,reinterpret_cast<uint16_t*>(Ws));
      else
        code=bcf::decode_slab_lane(data,size,desc+9*(4*c+warp),K,nr,nc,
            labels,ts,nt,c,warp,lane,reinterpret_cast<uint16_t*>(Ws));
      status[(blockIdx.x*(K/bcf::CH)+c)*128+threadIdx.x]=code;
      if (code) atomicOr(&failed,code);
    }
#pragma unroll
    for (int it=0;it<MT;++it) {
      const int i=threadIdx.x+256*it, r=i>>4, cv=i&15, m=m0+r;
      uint4 v=make_uint4(0u,0u,0u,0u);
      if (m<M) v=__ldg(reinterpret_cast<const uint4*>(x+int64_t(m)*K+c*bcf::CH+cv*8));
      *reinterpret_cast<uint4*>(Xs+r*bi::LDS+cv*8)=v;
    }
    __syncthreads();
    if (failed) {
      // Every status slot remains defined even on an early CTA exit. Code 32
      // means skipped after an earlier failing slab. Native y is unspecified
      // on any failure; the Python API checks status before exposing it.
      if (threadIdx.x<128)
        for (uint32_t rest=c+1;rest<K/bcf::CH;++rest)
          status[(blockIdx.x*(K/bcf::CH)+rest)*128+threadIdx.x]=32;
      return;
    }
#pragma unroll
    for (int kk=0;kk<bi::CH;kk+=16) {
      uint32_t b[2];
      bi::ldsm_x2(b,Ws+(warp*8+(lane&7))*bi::LDS+kk+((lane>>3)&1)*8);
#pragma unroll
      for (int mt=0;mt<MT;++mt) {
        uint32_t a[4];
        bi::ldsm_x4(a,Xs+(mt*16+(lane&7)+((lane>>3)&1)*8)*bi::LDS+kk+(lane>>4)*8);
        bi::mma16816(acc[mt],a,b);
      }
    }
    __syncthreads();
  }
  const int g=lane>>2, t4=lane&3, n=warp*8+2*t4;
#pragma unroll
  for (int mt=0;mt<MT;++mt)
#pragma unroll
    for (int h=0;h<2;++h) {
      const int m=m0+mt*16+g+8*h;
      if (m>=M) continue;
#pragma unroll
      for (int e=0;e<2;++e)
        y[int64_t(m)*output_stride+panel*bcf::N+n+e]=__float2bfloat16_rn(acc[mt][2*h+e]);
    }
}

std::vector<torch::Tensor> bcf_gemm_cuda(torch::Tensor data, torch::Tensor desc,
    torch::Tensor x, int64_t nr, int64_t nc, int64_t labels, int64_t ts, int64_t nt, bool optimized) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int M=x.size(0),K=x.size(1),blocks=(M+63)/64,mt=M>=64?4:1;
  // A valid launch writes every y and status element. Avoid separate memset
  // kernels; failure still writes every status, while y must then be discarded.
  auto y=torch::empty({M,64},x.options());
  auto status=torch::empty({blocks,K/128,4,32},x.options().dtype(torch::kInt32));
  const size_t smem=(bcf::N+mt*16)*bi::LDS*2;
  auto stream=at::cuda::getCurrentCUDAStream();
  const auto* xp=reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  auto* yp=reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>());
  if (optimized && mt==1)
    bcf_kernel<1,true><<<blocks,256,smem,stream>>>(data.data_ptr<uint8_t>(),data.numel(),desc.data_ptr<int64_t>(),
        xp,K,M,nr,nc,labels,ts,nt,nullptr,0,nullptr,0,nullptr,0,yp,status.data_ptr<int32_t>());
  else if (optimized)
    bcf_kernel<4,true><<<blocks,256,smem,stream>>>(data.data_ptr<uint8_t>(),data.numel(),desc.data_ptr<int64_t>(),
        xp,K,M,nr,nc,labels,ts,nt,nullptr,0,nullptr,0,nullptr,0,yp,status.data_ptr<int32_t>());
  else if (mt==1)
    bcf_kernel<1><<<blocks,256,smem,stream>>>(data.data_ptr<uint8_t>(),data.numel(),desc.data_ptr<int64_t>(),
        xp,K,M,nr,nc,labels,ts,nt,nullptr,0,nullptr,0,nullptr,0,yp,status.data_ptr<int32_t>());
  else
    bcf_kernel<4><<<blocks,256,smem,stream>>>(data.data_ptr<uint8_t>(),data.numel(),desc.data_ptr<int64_t>(),
        xp,K,M,nr,nc,labels,ts,nt,nullptr,0,nullptr,0,nullptr,0,yp,status.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y,status};
}

std::vector<torch::Tensor> bcf_gemm_cuda_lut(torch::Tensor data, torch::Tensor desc,
    torch::Tensor x, int64_t nr, int64_t nc, int64_t labels, int64_t ts, int64_t nt,
    torch::Tensor coarse, torch::Tensor records, torch::Tensor offsets) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int M=x.size(0),K=x.size(1),blocks=(M+63)/64,mt=M>=64?4:1;
  auto y=torch::empty({M,64},x.options());
  auto status=torch::empty({blocks,K/128,4,32},x.options().dtype(torch::kInt32));
  const size_t smem=(bcf::N+mt*16)*bi::LDS*2;
  auto stream=at::cuda::getCurrentCUDAStream();
  const auto* xp=reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  auto* yp=reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>());
  const auto* cp=coarse.numel()?coarse.data_ptr<uint32_t>():nullptr;
  const auto* rp=records.numel()?records.data_ptr<uint64_t>():nullptr;
  const auto* op=offsets.numel()?offsets.data_ptr<uint32_t>():nullptr;
  if (mt==1)
    bcf_kernel<1,true,false,true><<<blocks,256,smem,stream>>>(data.data_ptr<uint8_t>(),data.numel(),desc.data_ptr<int64_t>(),
        xp,K,M,nr,nc,labels,ts,nt,cp,coarse.numel(),rp,records.numel(),op,offsets.numel(),yp,status.data_ptr<int32_t>());
  else
    bcf_kernel<4,true,false,true><<<blocks,256,smem,stream>>>(data.data_ptr<uint8_t>(),data.numel(),desc.data_ptr<int64_t>(),
        xp,K,M,nr,nc,labels,ts,nt,cp,coarse.numel(),rp,records.numel(),op,offsets.numel(),yp,status.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y,status};
}

torch::Tensor bcf_dense_cuda(torch::Tensor w, torch::Tensor x) {
  const c10::cuda::CUDAGuard guard(x.device());
  auto y=torch::empty({x.size(0),64},x.options());
  BiDesc d{};
  d.N=64;d.K=x.size(1);d.w=reinterpret_cast<const __nv_bfloat16*>(w.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK((bi::launch_fmt<bi::F_BF16,0>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),x.size(1),x.size(0),d,1,
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),64,nullptr,at::cuda::getCurrentCUDAStream(),64)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// Separate entry: panel replication increases independent N tiles, not K splits
// or activation rows. Each CTA keeps the existing ordered accumulator chain.
std::vector<torch::Tensor> bcf_gemm_panels_cuda(torch::Tensor data, torch::Tensor desc,
    torch::Tensor x, int64_t nr, int64_t nc, int64_t labels, int64_t ts, int64_t nt) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int P=data.size(0),M=x.size(0),K=x.size(1),blocks=(M+63)/64,mt=M>=64?4:1;
  auto y=torch::empty({M,P*64},x.options());
  auto status=torch::empty({P,blocks,K/128,4,32},x.options().dtype(torch::kInt32));
  const size_t smem=(bcf::N+mt*16)*bi::LDS*2;
  auto stream=at::cuda::getCurrentCUDAStream();
  const auto* xp=reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  auto* yp=reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>());
  const dim3 grid(blocks,P);
  if (mt==1)
    bcf_kernel<1,true,true><<<grid,256,smem,stream>>>(data.data_ptr<uint8_t>(),data.size(1),
        desc.data_ptr<int64_t>(),xp,K,M,nr,nc,labels,ts,nt,nullptr,0,nullptr,0,nullptr,0,yp,status.data_ptr<int32_t>());
  else
    bcf_kernel<4,true,true><<<grid,256,smem,stream>>>(data.data_ptr<uint8_t>(),data.size(1),
        desc.data_ptr<int64_t>(),xp,K,M,nr,nc,labels,ts,nt,nullptr,0,nullptr,0,nullptr,0,yp,status.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y,status};
}

std::vector<torch::Tensor> bcf_gemm_panels_cuda_lut(torch::Tensor data, torch::Tensor desc,
    torch::Tensor x, int64_t nr, int64_t nc, int64_t labels, int64_t ts, int64_t nt,
    torch::Tensor coarse, torch::Tensor records, torch::Tensor offsets) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int P=data.size(0),M=x.size(0),K=x.size(1),blocks=(M+63)/64,mt=M>=64?4:1;
  auto y=torch::empty({M,P*64},x.options());
  auto status=torch::empty({P,blocks,K/128,4,32},x.options().dtype(torch::kInt32));
  const size_t smem=(bcf::N+mt*16)*bi::LDS*2;
  auto stream=at::cuda::getCurrentCUDAStream();
  const auto* xp=reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  auto* yp=reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>());
  const auto* cp=coarse.numel()?coarse.data_ptr<uint32_t>():nullptr;
  const auto* rp=records.numel()?records.data_ptr<uint64_t>():nullptr;
  const auto* op=offsets.numel()?offsets.data_ptr<uint32_t>():nullptr;
  const dim3 grid(blocks,P);
  if (mt==1)
    bcf_kernel<1,true,true,true><<<grid,256,smem,stream>>>(data.data_ptr<uint8_t>(),data.size(1),
        desc.data_ptr<int64_t>(),xp,K,M,nr,nc,labels,ts,nt,cp,coarse.numel(),rp,records.numel(),
        op,offsets.numel(),yp,status.data_ptr<int32_t>());
  else
    bcf_kernel<4,true,true,true><<<grid,256,smem,stream>>>(data.data_ptr<uint8_t>(),data.size(1),
        desc.data_ptr<int64_t>(),xp,K,M,nr,nc,labels,ts,nt,cp,coarse.numel(),rp,records.numel(),
        op,offsets.numel(),yp,status.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y,status};
}

torch::Tensor bcf_dense_panels_cuda(torch::Tensor w, torch::Tensor x) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int N=w.size(0);
  auto y=torch::empty({x.size(0),N},x.options());
  BiDesc d{};
  d.N=N;d.K=x.size(1);d.w=reinterpret_cast<const __nv_bfloat16*>(w.data_ptr<at::BFloat16>());
  C10_CUDA_CHECK((bi::launch_fmt<bi::F_BF16,0>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),x.size(1),x.size(0),d,1,
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),N,nullptr,at::cuda::getCurrentCUDAStream(),64)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
