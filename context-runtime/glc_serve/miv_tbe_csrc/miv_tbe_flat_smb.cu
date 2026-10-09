#include "miv_tbe_flat_smb.cuh"

namespace miv_tbe {
template<int WPR, int U>
static cudaError_t launch_g(const MivTbeDesc& d, const __nv_bfloat16* x, int64_t ldx,
                            __nv_bfloat16* y, int64_t ldy, cudaStream_t st) {
  constexpr int RPC=8/WPR;
  flat_smb_kernel<1,WPR,U><<<(d.n+RPC-1)/RPC,256,0,st>>>(x,ldx,make_args(d),y,ldy);
  return cudaGetLastError();
}
template<int WPR>
static cudaError_t launch_gu(const MivTbeDesc& d,const __nv_bfloat16* x,int64_t ldx,
                             __nv_bfloat16* y,int64_t ldy,cudaStream_t st) {
  switch(d.unroll) { case 1:return launch_g<WPR,1>(d,x,ldx,y,ldy,st); case 2:return launch_g<WPR,2>(d,x,ldx,y,ldy,st); case 4:return launch_g<WPR,4>(d,x,ldx,y,ldy,st); default:return cudaErrorInvalidValue; }
}
cudaError_t flat_smb_gemv(const MivTbeDesc& d,const __nv_bfloat16* x,int64_t ldx,__nv_bfloat16* y,int64_t ldy,cudaStream_t st) {
  if(d.n<=0||d.k<=0||d.k%(64*d.wpr)||!(d.wpr==1||d.wpr==2||d.wpr==4||d.wpr==8)||!(d.unroll==1||d.unroll==2||d.unroll==4)) return cudaErrorInvalidValue;
  switch(d.wpr) {case 1:return launch_gu<1>(d,x,ldx,y,ldy,st);case 2:return launch_gu<2>(d,x,ldx,y,ldy,st);case 4:return launch_gu<4>(d,x,ldx,y,ldy,st);case 8:return launch_gu<8>(d,x,ldx,y,ldy,st);default:return cudaErrorInvalidValue;}
}
template<int WPR,int U>
static cudaError_t launch_d(const MivTbeDesc& d,__nv_bfloat16* out,cudaStream_t st) {
 constexpr int RPC=8/WPR; flat_smb_decode_kernel<WPR,U><<<(d.n+RPC-1)/RPC,256,0,st>>>(make_args(d),out); return cudaGetLastError();
}
template<int WPR>
static cudaError_t launch_du(const MivTbeDesc& d,__nv_bfloat16* out,cudaStream_t st) {
 switch(d.unroll){case 1:return launch_d<WPR,1>(d,out,st);case 2:return launch_d<WPR,2>(d,out,st);case 4:return launch_d<WPR,4>(d,out,st);default:return cudaErrorInvalidValue;}
}
cudaError_t flat_smb_decode(const MivTbeDesc& d,__nv_bfloat16* out,cudaStream_t st) {
 if(d.n<=0||d.k<=0||d.k%(64*d.wpr)) return cudaErrorInvalidValue;
 switch(d.wpr){case 1:return launch_du<1>(d,out,st);case 2:return launch_du<2>(d,out,st);case 4:return launch_du<4>(d,out,st);case 8:return launch_du<8>(d,out,st);default:return cudaErrorInvalidValue;}
}
}
