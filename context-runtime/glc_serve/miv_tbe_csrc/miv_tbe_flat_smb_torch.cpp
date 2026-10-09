#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "miv_tbe.h"

namespace miv_tbe {
cudaError_t flat_smb_gemv(const MivTbeDesc&,const __nv_bfloat16*,int64_t,__nv_bfloat16*,int64_t,cudaStream_t);
cudaError_t flat_smb_decode(const MivTbeDesc&,__nv_bfloat16*,cudaStream_t);
}
static MivTbeDesc desc(at::Tensor p,at::Tensor s,at::Tensor e,at::Tensor rp,at::Tensor eb,int64_t n,int64_t k,int64_t wpr,int64_t u,const c10::optional<at::Tensor>& bias) {
 TORCH_CHECK(p.is_cuda()&&s.is_cuda()&&e.is_cuda()&&rp.is_cuda()&&eb.is_cuda(),"flat smb: CUDA tensors required");
 TORCH_CHECK(p.scalar_type()==at::kInt&&s.scalar_type()==at::kByte&&e.scalar_type()==at::kByte&&rp.scalar_type()==at::kInt&&eb.scalar_type()==at::kInt,"flat smb: dtype");
 TORCH_CHECK(p.is_contiguous()&&s.is_contiguous()&&e.is_contiguous()&&rp.is_contiguous()&&eb.is_contiguous(),"flat smb: contiguous");
 TORCH_CHECK((wpr==1||wpr==2||wpr==4||wpr==8)&&(u==1||u==2||u==4)&&k%(64*wpr)==0&&p.numel()==n*(k/64)*6&&s.numel()==n*k&&rp.numel()==n&&eb.numel()==n*wpr,"flat smb: shape");
 MivTbeDesc d{};d.n=n;d.k=k;d.wpr=wpr;d.unroll=u;d.planes=(const uint32_t*)p.data_ptr<int32_t>();d.smb=s.data_ptr<uint8_t>();d.esc=e.data_ptr<uint8_t>();d.rowparam=rp.data_ptr<int32_t>();d.escbase=eb.data_ptr<int32_t>();
 if(bias.has_value()&&bias->defined()){TORCH_CHECK(bias->is_cuda()&&bias->scalar_type()==at::kBFloat16&&bias->numel()==n&&bias->is_contiguous(),"flat smb: bias");d.bias=(const __nv_bfloat16*)bias->data_ptr<at::BFloat16>();}
 return d;
}
void out(at::Tensor x,at::Tensor p,at::Tensor s,at::Tensor e,at::Tensor rp,at::Tensor eb,c10::optional<at::Tensor> bias,at::Tensor y,int64_t n,int64_t k,int64_t wpr,int64_t u){
 TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kBFloat16&&x.dim()==2&&x.size(0)==1&&x.size(1)==k&&x.stride(1)==1&&x.stride(0)%8==0,"flat smb: x must be [1,K] bf16");
 TORCH_CHECK(y.is_cuda()&&y.scalar_type()==at::kBFloat16&&y.dim()==2&&y.size(0)==1&&y.size(1)==n,"flat smb: y must be [1,N] bf16");
 auto d=desc(p,s,e,rp,eb,n,k,wpr,u,bias);c10::cuda::CUDAGuard guard(x.device());auto er=miv_tbe::flat_smb_gemv(d,(const __nv_bfloat16*)x.data_ptr<at::BFloat16>(),x.stride(0),( __nv_bfloat16*)y.data_ptr<at::BFloat16>(),y.stride(0),at::cuda::getCurrentCUDAStream());TORCH_CHECK(er==cudaSuccess,"flat smb gemv: ",cudaGetErrorString(er));
}
void decode_out(at::Tensor p,at::Tensor s,at::Tensor e,at::Tensor rp,at::Tensor eb,at::Tensor w,int64_t n,int64_t k,int64_t wpr,int64_t u){
 TORCH_CHECK(w.is_cuda()&&w.scalar_type()==at::kBFloat16&&w.is_contiguous()&&w.numel()==n*k,"flat smb: decode output");auto d=desc(p,s,e,rp,eb,n,k,wpr,u,c10::nullopt);c10::cuda::CUDAGuard guard(w.device());auto er=miv_tbe::flat_smb_decode(d,(__nv_bfloat16*)w.data_ptr<at::BFloat16>(),at::cuda::getCurrentCUDAStream());TORCH_CHECK(er==cudaSuccess,"flat smb decode: ",cudaGetErrorString(er));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("out",&out);m.def("decode_out",&decode_out);}
