#include <torch/extension.h>
#include <vector>
#include <new>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "context_alias_packet.cuh"
#include "bi_gemm.cuh"

constexpr int BCAG_VIEW_SLOT=128, BCAG_VIEW_BYTES=4*BCAG_VIEW_SLOT;
static_assert(sizeof(bca::PacketView)<=BCAG_VIEW_SLOT && alignof(bca::PacketView)<=16,
              "prepared view must fit its aligned shared slot");
static_assert(bca::LDS==bi::LDS,"decoder and BI shared strides must match");

// Identical per-output BI accumulator chain; only the weight producer differs.
// Prepare each wide packet once, keep its view immutable, and carry lane cursors
// across CH128. No decoded HBM, no LUT, no producer/consumer overlap claim.
template<int MT,bool ScalarRecords=false,bool AlignedOffsets=false,bool Trusted=false,bool MainFifo=false,bool StageRaw6=false>
__global__ void bcag_kernel(const uint8_t* data,uint64_t bytes,const int64_t* desc,
    const __nv_bfloat16* x,uint32_t K,uint32_t M,uint32_t nr,uint32_t nc,
    const uint8_t* metadata,uint64_t metadata_bytes,uint32_t buckets,uint32_t label_rows,bool matrix,__nv_bfloat16* y,int32_t* status) {
  extern __shared__ __align__(16) unsigned char storage[];
  auto* Ws=reinterpret_cast<__nv_bfloat16*>(storage);
  auto* Xs=Ws+bca::N*bi::LDS;
  auto* view_storage=storage+(bca::N+MT*16)*bi::LDS*2;
  __shared__ uint32_t failed;
  const int warp=threadIdx.x>>5,lane=threadIdx.x&31,m0=blockIdx.x*64;
  const uint32_t panel=blockIdx.y,N=gridDim.y*64;
  const uint8_t* panel_data=data+(matrix?0:uint64_t(panel)*bytes);
  const int64_t* panel_desc=desc+uint64_t(panel)*4*9;
  int32_t* final_status=status+(uint64_t(panel)*gridDim.x+blockIdx.x)*128;
  if (threadIdx.x==0) failed=0;
  if (warp<4 && lane==0) {
    auto* view=new(view_storage+warp*BCAG_VIEW_SLOT) bca::PacketView{};
    bca::prepare_packet<AlignedOffsets>(panel_data,bytes,panel_desc+9*warp,metadata,metadata_bytes,buckets,K,nr,nc,*view,label_rows,matrix?panel*64:0);
    if (panel_desc[9*warp+1]!=16*warp) view->error=1;
  }
  __syncthreads();
  bca::LaneCursor cursor{};bca::TrustedPacket trusted_view{};bca::TrustedCursor trusted_cursor{};
  bca::TrustedFifo fifo{};uint8_t* staged_raw6=nullptr;
  if (warp<4) {
    if constexpr(MainFifo)fifo.words=reinterpret_cast<uint16_t*>(view_storage+BCAG_VIEW_BYTES+warp*bca::MAIN_FIFO_BYTES);
    if constexpr(StageRaw6)staged_raw6=view_storage+BCAG_VIEW_BYTES+4*bca::MAIN_FIFO_BYTES+warp*bca::RAW6_STAGE_BYTES;
    const auto* view=reinterpret_cast<const bca::PacketView*>(view_storage+warp*BCAG_VIEW_SLOT);
    uint32_t code;
    if constexpr(Trusted)code=bca::init_trusted(*view,lane,trusted_view,trusted_cursor);
    else code=bca::init_lane(*view,lane,cursor);
    if (code) atomicOr(&failed,code);
  }
  __syncthreads();
  if (failed) {
    if (warp<4) final_status[threadIdx.x]=(Trusted?trusted_cursor.error:cursor.error) ? (Trusted?trusted_cursor.error:cursor.error) : 32;
    return;
  }
  float acc[MT][4];
#pragma unroll
  for (int mt=0;mt<MT;++mt)
#pragma unroll
    for (int e=0;e<4;++e) acc[mt][e]=0.0f;
  for (uint32_t c=0;c<K/bca::CH;++c) {
    if (warp<4) {
      const auto* view=reinterpret_cast<const bca::PacketView*>(view_storage+warp*BCAG_VIEW_SLOT);
      uint32_t code;
      if constexpr(Trusted)code=bca::decode_trusted_slab<MainFifo,StageRaw6>(trusted_view,lane,c,c+1==K/bca::CH,trusted_cursor,reinterpret_cast<uint16_t*>(Ws),&fifo,staged_raw6);
      else code=bca::decode_next_slab<ScalarRecords,AlignedOffsets>(*view,lane,c,cursor,
          reinterpret_cast<uint16_t*>(Ws),bca::N*bca::LDS);
      if (code) atomicOr(&failed,code);
    }
#pragma unroll
    for (int it=0;it<MT;++it) {
      const int i=threadIdx.x+256*it,r=i>>4,cv=i&15,m=m0+r;
      uint4 v=make_uint4(0u,0u,0u,0u);
      if (m<M) v=__ldg(reinterpret_cast<const uint4*>(x+int64_t(m)*K+c*bca::CH+cv*8));
      *reinterpret_cast<uint4*>(Xs+r*bi::LDS+cv*8)=v;
    }
    __syncthreads();
    if (failed) {
      // Every final status is defined on every CTA exit. Code32 means aborted
      // because another lane failed; y must be discarded on any nonzero status.
      if (warp<4) final_status[threadIdx.x]=(Trusted?trusted_cursor.error:cursor.error) ? (Trusted?trusted_cursor.error:cursor.error) : 32;
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

std::vector<torch::Tensor> bcag_cuda(torch::Tensor data,torch::Tensor desc,torch::Tensor x,
    torch::Tensor metadata,int64_t nr,int64_t nc,int64_t buckets,int64_t label_rows,bool matrix,bool optimized,bool aligned_offsets,bool trusted,bool main_fifo,bool stage_raw6) {
  TORCH_CHECK((!stage_raw6 || main_fifo) && (!main_fifo || trusted),"invalid stream staging variant");
  const c10::cuda::CUDAGuard guard(x.device());
  const int P=desc.size(0),M=x.size(0),K=x.size(1),blocks=(M+63)/64,mt=M>=64?4:1;
  auto y=torch::empty({M,P*64},x.options());
  auto status=torch::empty({P,blocks,4,32},x.options().dtype(torch::kInt32));
  const size_t smem=(bca::N+mt*16)*bi::LDS*2+BCAG_VIEW_BYTES+
      (main_fifo?4*bca::MAIN_FIFO_BYTES:0)+(stage_raw6?4*bca::RAW6_STAGE_BYTES:0);
  const dim3 grid(blocks,P);
  auto stream=at::cuda::getCurrentCUDAStream();
  c10::cuda::CUDACachingAllocator::recordStream(data.storage().data_ptr(),stream);
  c10::cuda::CUDACachingAllocator::recordStream(desc.storage().data_ptr(),stream);
  c10::cuda::CUDACachingAllocator::recordStream(metadata.storage().data_ptr(),stream);
  const auto* xp=reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  auto* yp=reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>());
  if(stage_raw6){
  if (mt==1)
    bcag_kernel<1,true,true,true,true,true><<<grid,256,smem,stream>>>(data.data_ptr<uint8_t>(),matrix?data.numel():data.size(1),desc.data_ptr<int64_t>(),
        xp,K,M,nr,nc,metadata.data_ptr<uint8_t>(),metadata.numel(),buckets,label_rows,matrix,yp,status.data_ptr<int32_t>());
  else
    bcag_kernel<4,true,true,true,true,true><<<grid,256,smem,stream>>>(data.data_ptr<uint8_t>(),matrix?data.numel():data.size(1),desc.data_ptr<int64_t>(),
        xp,K,M,nr,nc,metadata.data_ptr<uint8_t>(),metadata.numel(),buckets,label_rows,matrix,yp,status.data_ptr<int32_t>());
  } else if(main_fifo){
  if (mt==1)
    bcag_kernel<1,true,true,true,true,false><<<grid,256,smem,stream>>>(data.data_ptr<uint8_t>(),matrix?data.numel():data.size(1),desc.data_ptr<int64_t>(),
        xp,K,M,nr,nc,metadata.data_ptr<uint8_t>(),metadata.numel(),buckets,label_rows,matrix,yp,status.data_ptr<int32_t>());
  else
    bcag_kernel<4,true,true,true,true,false><<<grid,256,smem,stream>>>(data.data_ptr<uint8_t>(),matrix?data.numel():data.size(1),desc.data_ptr<int64_t>(),
        xp,K,M,nr,nc,metadata.data_ptr<uint8_t>(),metadata.numel(),buckets,label_rows,matrix,yp,status.data_ptr<int32_t>());
  } else if(trusted){
  if (mt==1)
    bcag_kernel<1,true,true,true><<<grid,256,smem,stream>>>(data.data_ptr<uint8_t>(),matrix?data.numel():data.size(1),desc.data_ptr<int64_t>(),
        xp,K,M,nr,nc,metadata.data_ptr<uint8_t>(),metadata.numel(),buckets,label_rows,matrix,yp,status.data_ptr<int32_t>());
  else
    bcag_kernel<4,true,true,true><<<grid,256,smem,stream>>>(data.data_ptr<uint8_t>(),matrix?data.numel():data.size(1),desc.data_ptr<int64_t>(),
        xp,K,M,nr,nc,metadata.data_ptr<uint8_t>(),metadata.numel(),buckets,label_rows,matrix,yp,status.data_ptr<int32_t>());
  } else if(aligned_offsets){
  if (mt==1)
    bcag_kernel<1,true,true><<<grid,256,smem,stream>>>(data.data_ptr<uint8_t>(),matrix?data.numel():data.size(1),desc.data_ptr<int64_t>(),
        xp,K,M,nr,nc,metadata.data_ptr<uint8_t>(),metadata.numel(),buckets,label_rows,matrix,yp,status.data_ptr<int32_t>());
  else
    bcag_kernel<4,true,true><<<grid,256,smem,stream>>>(data.data_ptr<uint8_t>(),matrix?data.numel():data.size(1),desc.data_ptr<int64_t>(),
        xp,K,M,nr,nc,metadata.data_ptr<uint8_t>(),metadata.numel(),buckets,label_rows,matrix,yp,status.data_ptr<int32_t>());
  } else if(optimized){
  if (mt==1)
    bcag_kernel<1,true><<<grid,256,smem,stream>>>(data.data_ptr<uint8_t>(),matrix?data.numel():data.size(1),desc.data_ptr<int64_t>(),
        xp,K,M,nr,nc,metadata.data_ptr<uint8_t>(),metadata.numel(),buckets,label_rows,matrix,yp,status.data_ptr<int32_t>());
  else
    bcag_kernel<4,true><<<grid,256,smem,stream>>>(data.data_ptr<uint8_t>(),matrix?data.numel():data.size(1),desc.data_ptr<int64_t>(),
        xp,K,M,nr,nc,metadata.data_ptr<uint8_t>(),metadata.numel(),buckets,label_rows,matrix,yp,status.data_ptr<int32_t>());
  } else {
  if (mt==1)
    bcag_kernel<1><<<grid,256,smem,stream>>>(data.data_ptr<uint8_t>(),matrix?data.numel():data.size(1),desc.data_ptr<int64_t>(),
        xp,K,M,nr,nc,metadata.data_ptr<uint8_t>(),metadata.numel(),buckets,label_rows,matrix,yp,status.data_ptr<int32_t>());
  else
    bcag_kernel<4><<<grid,256,smem,stream>>>(data.data_ptr<uint8_t>(),matrix?data.numel():data.size(1),desc.data_ptr<int64_t>(),
        xp,K,M,nr,nc,metadata.data_ptr<uint8_t>(),metadata.numel(),buckets,label_rows,matrix,yp,status.data_ptr<int32_t>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y,status};
}

torch::Tensor bcag_dense_cuda(torch::Tensor w,torch::Tensor x) {
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

// Diagnostic only. Reuses the production warp decoder to expose every word
// including NaN payloads. No GEMM entry allocates this full decoded HBM tensor.
template<bool Produce,bool ScalarRecords=false,bool AlignedOffsets=false,bool Trusted=false,bool MainFifo=false,bool StageRaw6=false>
__global__ void bcag_decode_kernel(const uint8_t* data,uint64_t bytes,const int64_t* desc,
    const uint8_t* meta,uint64_t meta_bytes,uint32_t buckets,uint32_t K,uint32_t nr,uint32_t nc,uint32_t label_rows,bool matrix,
    uint16_t* output,uint32_t* checksum,int32_t* status){
  extern __shared__ __align__(16) unsigned char storage[];
  auto* slab=reinterpret_cast<uint16_t*>(storage);
  auto* views=storage+bca::N*bca::LDS*2;
  __shared__ uint32_t failed;
  const uint32_t panel=blockIdx.x,warp=threadIdx.x>>5,lane=threadIdx.x&31;
  if(threadIdx.x==0)failed=0;
  if(lane==0){
    auto* view=new(views+warp*BCAG_VIEW_SLOT)bca::PacketView{};
    const int64_t* d=desc+(uint64_t(panel)*4+warp)*9;
    bca::prepare_packet<AlignedOffsets>(data+(matrix?0:uint64_t(panel)*bytes),bytes,d,meta,meta_bytes,buckets,K,nr,nc,*view,label_rows,matrix?panel*64:0);
    if(d[1]!=16*warp)view->error=1;
  }
  __syncthreads();
  const auto* view=reinterpret_cast<const bca::PacketView*>(views+warp*BCAG_VIEW_SLOT);
  bca::LaneCursor cursor{};bca::TrustedPacket trusted_view{};bca::TrustedCursor trusted_cursor{};
  bca::TrustedFifo fifo{};uint8_t* staged_raw6=nullptr;
  if constexpr(MainFifo)fifo.words=reinterpret_cast<uint16_t*>(views+BCAG_VIEW_BYTES+warp*bca::MAIN_FIFO_BYTES);
  if constexpr(StageRaw6)staged_raw6=views+BCAG_VIEW_BYTES+4*bca::MAIN_FIFO_BYTES+warp*bca::RAW6_STAGE_BYTES;
  uint32_t code;
  if constexpr(Trusted)code=bca::init_trusted(*view,lane,trusted_view,trusted_cursor);
  else code=bca::init_lane(*view,lane,cursor);
  if(code)atomicOr(&failed,code);
  __syncthreads();
  uint32_t hash=2166136261u;
  for(uint32_t c=0;c<K/bca::CH && !failed;++c){
    if constexpr(Trusted)code=bca::decode_trusted_slab<MainFifo,StageRaw6>(trusted_view,lane,c,c+1==K/bca::CH,trusted_cursor,slab,&fifo,staged_raw6);
    else code=bca::decode_next_slab<ScalarRecords,AlignedOffsets>(*view,lane,c,cursor,slab,bca::N*bca::LDS);
    if(code)atomicOr(&failed,code);
    __syncthreads();
    if(!failed)for(uint32_t i=threadIdx.x;i<bca::N*bca::CH;i+=128){
      const uint32_t row=i/bca::CH,col=i%bca::CH;
      const uint16_t word=slab[row*bca::LDS+col];
      if constexpr(Produce)hash=(hash^uint32_t(word))*16777619u;
      else output[(uint64_t(panel)*bca::N+row)*K+c*bca::CH+col]=word;
    }
    __syncthreads();
  }
  if constexpr(Produce)checksum[panel*128+threadIdx.x]=hash;
  const uint32_t lane_error=Trusted?trusted_cursor.error:cursor.error;
  status[panel*128+threadIdx.x]=lane_error?lane_error:failed?32:0;
}
std::vector<torch::Tensor> bcag_decode_cuda(torch::Tensor data,torch::Tensor desc,torch::Tensor metadata,
    int64_t K,int64_t nr,int64_t nc,int64_t buckets,int64_t label_rows,bool matrix,bool produce,bool optimized,bool aligned_offsets,bool trusted,bool main_fifo,bool stage_raw6){
  TORCH_CHECK((!stage_raw6 || main_fifo) && (!main_fifo || trusted),"invalid stream staging variant");
  const c10::cuda::CUDAGuard guard(data.device());
  const int64_t panels=desc.size(0);
  auto y=produce ? torch::empty({panels,4,32},data.options().dtype(torch::kInt32)) :
      torch::empty({panels*64,K},data.options().dtype(torch::kBFloat16));
  auto status=torch::empty({panels,4,32},data.options().dtype(torch::kInt32));
  auto stream=at::cuda::getCurrentCUDAStream();
  c10::cuda::CUDACachingAllocator::recordStream(data.storage().data_ptr(),stream);
  c10::cuda::CUDACachingAllocator::recordStream(desc.storage().data_ptr(),stream);
  c10::cuda::CUDACachingAllocator::recordStream(metadata.storage().data_ptr(),stream);
  const size_t smem=bca::N*bca::LDS*2+BCAG_VIEW_BYTES+
      (main_fifo?4*bca::MAIN_FIFO_BYTES:0)+(stage_raw6?4*bca::RAW6_STAGE_BYTES:0);
  if(stage_raw6){
  if(produce)
    bcag_decode_kernel<true,true,true,true,true,true><<<panels,128,smem,stream>>>(
        data.data_ptr<uint8_t>(),matrix?data.numel():data.size(1),desc.data_ptr<int64_t>(),metadata.data_ptr<uint8_t>(),
        metadata.numel(),buckets,K,nr,nc,label_rows,matrix,nullptr,reinterpret_cast<uint32_t*>(y.data_ptr<int32_t>()),status.data_ptr<int32_t>());
  else
    bcag_decode_kernel<false,true,true,true,true,true><<<panels,128,smem,stream>>>(
        data.data_ptr<uint8_t>(),matrix?data.numel():data.size(1),desc.data_ptr<int64_t>(),metadata.data_ptr<uint8_t>(),
        metadata.numel(),buckets,K,nr,nc,label_rows,matrix,reinterpret_cast<uint16_t*>(y.data_ptr<at::BFloat16>()),nullptr,status.data_ptr<int32_t>());
  } else if(main_fifo){
  if(produce)
    bcag_decode_kernel<true,true,true,true,true,false><<<panels,128,smem,stream>>>(
        data.data_ptr<uint8_t>(),matrix?data.numel():data.size(1),desc.data_ptr<int64_t>(),metadata.data_ptr<uint8_t>(),
        metadata.numel(),buckets,K,nr,nc,label_rows,matrix,nullptr,reinterpret_cast<uint32_t*>(y.data_ptr<int32_t>()),status.data_ptr<int32_t>());
  else
    bcag_decode_kernel<false,true,true,true,true,false><<<panels,128,smem,stream>>>(
        data.data_ptr<uint8_t>(),matrix?data.numel():data.size(1),desc.data_ptr<int64_t>(),metadata.data_ptr<uint8_t>(),
        metadata.numel(),buckets,K,nr,nc,label_rows,matrix,reinterpret_cast<uint16_t*>(y.data_ptr<at::BFloat16>()),nullptr,status.data_ptr<int32_t>());
  } else if(trusted){
  if(produce)
    bcag_decode_kernel<true,true,true,true><<<panels,128,smem,stream>>>(
        data.data_ptr<uint8_t>(),matrix?data.numel():data.size(1),desc.data_ptr<int64_t>(),metadata.data_ptr<uint8_t>(),
        metadata.numel(),buckets,K,nr,nc,label_rows,matrix,nullptr,reinterpret_cast<uint32_t*>(y.data_ptr<int32_t>()),status.data_ptr<int32_t>());
  else
    bcag_decode_kernel<false,true,true,true><<<panels,128,smem,stream>>>(
        data.data_ptr<uint8_t>(),matrix?data.numel():data.size(1),desc.data_ptr<int64_t>(),metadata.data_ptr<uint8_t>(),
        metadata.numel(),buckets,K,nr,nc,label_rows,matrix,reinterpret_cast<uint16_t*>(y.data_ptr<at::BFloat16>()),nullptr,status.data_ptr<int32_t>());
  } else if(aligned_offsets){
  if(produce)
    bcag_decode_kernel<true,true,true><<<panels,128,smem,stream>>>(
        data.data_ptr<uint8_t>(),matrix?data.numel():data.size(1),desc.data_ptr<int64_t>(),metadata.data_ptr<uint8_t>(),
        metadata.numel(),buckets,K,nr,nc,label_rows,matrix,nullptr,reinterpret_cast<uint32_t*>(y.data_ptr<int32_t>()),status.data_ptr<int32_t>());
  else
    bcag_decode_kernel<false,true,true><<<panels,128,smem,stream>>>(
        data.data_ptr<uint8_t>(),matrix?data.numel():data.size(1),desc.data_ptr<int64_t>(),metadata.data_ptr<uint8_t>(),
        metadata.numel(),buckets,K,nr,nc,label_rows,matrix,reinterpret_cast<uint16_t*>(y.data_ptr<at::BFloat16>()),nullptr,status.data_ptr<int32_t>());
  } else if(optimized){
  if(produce)
    bcag_decode_kernel<true,true><<<panels,128,smem,stream>>>(
        data.data_ptr<uint8_t>(),matrix?data.numel():data.size(1),desc.data_ptr<int64_t>(),metadata.data_ptr<uint8_t>(),
        metadata.numel(),buckets,K,nr,nc,label_rows,matrix,nullptr,reinterpret_cast<uint32_t*>(y.data_ptr<int32_t>()),status.data_ptr<int32_t>());
  else
    bcag_decode_kernel<false,true><<<panels,128,smem,stream>>>(
        data.data_ptr<uint8_t>(),matrix?data.numel():data.size(1),desc.data_ptr<int64_t>(),metadata.data_ptr<uint8_t>(),
        metadata.numel(),buckets,K,nr,nc,label_rows,matrix,reinterpret_cast<uint16_t*>(y.data_ptr<at::BFloat16>()),nullptr,status.data_ptr<int32_t>());
  } else {
  if(produce)
    bcag_decode_kernel<true><<<panels,128,smem,stream>>>(
        data.data_ptr<uint8_t>(),matrix?data.numel():data.size(1),desc.data_ptr<int64_t>(),metadata.data_ptr<uint8_t>(),
        metadata.numel(),buckets,K,nr,nc,label_rows,matrix,nullptr,reinterpret_cast<uint32_t*>(y.data_ptr<int32_t>()),status.data_ptr<int32_t>());
  else
    bcag_decode_kernel<false><<<panels,128,smem,stream>>>(
        data.data_ptr<uint8_t>(),matrix?data.numel():data.size(1),desc.data_ptr<int64_t>(),metadata.data_ptr<uint8_t>(),
        metadata.numel(),buckets,K,nr,nc,label_rows,matrix,reinterpret_cast<uint16_t*>(y.data_ptr<at::BFloat16>()),nullptr,status.data_ptr<int32_t>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {y,status};
}
