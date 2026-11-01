#include <torch/extension.h>
#include <vector>
#include <new>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "context_stream_packet.cuh"
#include "bi_gemm.cuh"

constexpr int BCSG_VIEW_SLOT=128, BCSG_VIEW_BYTES=4*BCSG_VIEW_SLOT;
static_assert(sizeof(bcs::PacketView)<=BCSG_VIEW_SLOT && alignof(bcs::PacketView)<=16,
              "prepared view must fit its aligned shared slot");
static_assert(bcs::LDS==bi::LDS,"decoder and BI shared strides must match");

// Identical per-output BI accumulator chain; only the weight producer differs.
// Prepare each wide packet once, keep its view immutable, and carry lane cursors
// across CH128. No decoded HBM, no LUT, no producer/consumer overlap claim.
template<int MT>
__global__ void bcsg_kernel(const uint8_t* data,uint64_t bytes,const int64_t* desc,
    const __nv_bfloat16* x,uint32_t K,uint32_t M,uint32_t nr,uint32_t nc,
    uint64_t labels,uint64_t ts,uint64_t nt,__nv_bfloat16* y,int32_t* status) {
  extern __shared__ __align__(16) unsigned char storage[];
  auto* Ws=reinterpret_cast<__nv_bfloat16*>(storage);
  auto* Xs=Ws+bcs::N*bi::LDS;
  auto* view_storage=storage+(bcs::N+MT*16)*bi::LDS*2;
  __shared__ uint32_t failed;
  const int warp=threadIdx.x>>5,lane=threadIdx.x&31,m0=blockIdx.x*64;
  const uint32_t panel=blockIdx.y,N=gridDim.y*64;
  const uint8_t* panel_data=data+uint64_t(panel)*bytes;
  const int64_t* panel_desc=desc+uint64_t(panel)*4*9;
  int32_t* final_status=status+(uint64_t(panel)*gridDim.x+blockIdx.x)*128;
  if (threadIdx.x==0) failed=0;
  if (warp<4 && lane==0) {
    auto* view=new(view_storage+warp*BCSG_VIEW_SLOT) bcs::PacketView{};
    bcs::prepare_packet(panel_data,bytes,panel_desc+9*warp,K,nr,nc,labels,ts,nt,1,*view);
    if (panel_desc[9*warp+1]!=16*warp) view->error=1;
  }
  __syncthreads();
  bcs::LaneCursor cursor{};
  if (warp<4) {
    const auto* view=reinterpret_cast<const bcs::PacketView*>(view_storage+warp*BCSG_VIEW_SLOT);
    const uint32_t code=bcs::init_lane(*view,lane,cursor);
    if (code) atomicOr(&failed,code);
  }
  __syncthreads();
  if (failed) {
    if (warp<4) final_status[threadIdx.x]=cursor.error ? cursor.error : 32;
    return;
  }
  float acc[MT][4];
#pragma unroll
  for (int mt=0;mt<MT;++mt)
#pragma unroll
    for (int e=0;e<4;++e) acc[mt][e]=0.0f;
  for (uint32_t c=0;c<K/bcs::CH;++c) {
    if (warp<4) {
      const auto* view=reinterpret_cast<const bcs::PacketView*>(view_storage+warp*BCSG_VIEW_SLOT);
      const uint32_t code=bcs::decode_next_slab(*view,lane,c,cursor,
          reinterpret_cast<uint16_t*>(Ws),bcs::N*bcs::LDS);
      if (code) atomicOr(&failed,code);
    }
#pragma unroll
    for (int it=0;it<MT;++it) {
      const int i=threadIdx.x+256*it,r=i>>4,cv=i&15,m=m0+r;
      uint4 v=make_uint4(0u,0u,0u,0u);
      if (m<M) v=__ldg(reinterpret_cast<const uint4*>(x+int64_t(m)*K+c*bcs::CH+cv*8));
      *reinterpret_cast<uint4*>(Xs+r*bi::LDS+cv*8)=v;
    }
    __syncthreads();
    if (failed) {
      // Every final status is defined on every CTA exit. Code32 means aborted
      // because another lane failed; y must be discarded on any nonzero status.
      if (warp<4) final_status[threadIdx.x]=cursor.error ? cursor.error : 32;
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
  if (warp<4) final_status[threadIdx.x]=0;
  const int g=lane>>2,t4=lane&3,n=panel*64+warp*8+2*t4;
#pragma unroll
  for (int mt=0;mt<MT;++mt)
#pragma unroll
    for (int h=0;h<2;++h) {
      const int m=m0+mt*16+g+8*h;
      if (m>=M) continue;
#pragma unroll
      for (int e=0;e<2;++e)
        y[int64_t(m)*N+n+e]=__float2bfloat16_rn(acc[mt][2*h+e]);
    }
}

std::vector<torch::Tensor> bcsg_cuda(torch::Tensor data,torch::Tensor desc,torch::Tensor x,
    int64_t nr,int64_t nc,int64_t labels,int64_t ts,int64_t nt) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int P=data.size(0),M=x.size(0),K=x.size(1),blocks=(M+63)/64,mt=M>=64?4:1;
  auto y=torch::empty({M,P*64},x.options());
  auto status=torch::empty({P,blocks,4,32},x.options().dtype(torch::kInt32));
  const size_t smem=(bcs::N+mt*16)*bi::LDS*2+BCSG_VIEW_BYTES;
  const dim3 grid(blocks,P);
  auto stream=at::cuda::getCurrentCUDAStream();
  const auto* xp=reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  auto* yp=reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>());
  if (mt==1)
    bcsg_kernel<1><<<grid,256,smem,stream>>>(data.data_ptr<uint8_t>(),data.size(1),desc.data_ptr<int64_t>(),
        xp,K,M,nr,nc,labels,ts,nt,yp,status.data_ptr<int32_t>());
  else
    bcsg_kernel<4><<<grid,256,smem,stream>>>(data.data_ptr<uint8_t>(),data.size(1),desc.data_ptr<int64_t>(),
        xp,K,M,nr,nc,labels,ts,nt,yp,status.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y,status};
}

torch::Tensor bcsg_dense_cuda(torch::Tensor w,torch::Tensor x) {
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
