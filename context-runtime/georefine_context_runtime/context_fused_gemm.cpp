#include "context_fused_gemm.cuh"
#ifdef BCF_CPU_TEST
// Test-only ABI for the actual on-chip decoder's slab mapping, not a GEMM twin.
extern "C" uint32_t bcf_test_lut_lookup(const uint32_t* coarse, uint64_t coarse_count,
    const uint64_t* records, uint64_t record_count, const uint32_t* offsets,
    uint64_t offset_count, uint32_t nctx, uint32_t context, uint32_t slot,
    uint32_t* result) {
  return bcf::cdf_lut_lookup(coarse,coarse_count,records,record_count,offsets,
      offset_count,nctx,context,slot,result,result+1,result+2);
}
extern "C" void bcf_test_slab(const uint8_t* data, uint64_t size, const int64_t* desc,
    uint32_t K, uint32_t nr, uint32_t nc, uint64_t labels, uint64_t ts, uint64_t nt,
    uint32_t chunk, uint16_t* slab, uint32_t* status) {
  for (uint32_t p=0;p<4;++p)
    for (uint32_t lane=0;lane<32;++lane)
      status[p*32+lane]=bcf::decode_slab_lane(data,size,desc+9*p,K,nr,nc,
                                             labels,ts,nt,chunk,p,lane,slab);
}
extern "C" void bcf_test_slab_opt(const uint8_t* data, uint64_t size, const int64_t* desc,
    uint32_t K, uint32_t nr, uint32_t nc, uint64_t labels, uint64_t ts, uint64_t nt,
    uint32_t chunk, uint16_t* slab, uint32_t* status) {
  for (uint32_t p=0;p<4;++p)
    for (uint32_t lane=0;lane<32;++lane)
      status[p*32+lane]=bcf::decode_slab_lane_opt(data,size,desc+9*p,K,nr,nc,
                                                labels,ts,nt,chunk,p,lane,slab);
}
extern "C" void bcf_test_slab_lut(const uint8_t* data, uint64_t size, const int64_t* desc,
    uint32_t K, uint32_t nr, uint32_t nc, uint64_t labels, uint64_t ts, uint64_t nt,
    uint32_t chunk, const uint32_t* coarse, uint64_t coarse_count,
    const uint64_t* records, uint64_t record_count, const uint32_t* offsets,
    uint64_t offset_count, uint16_t* slab, uint32_t* status) {
  for (uint32_t p=0;p<4;++p)
    for (uint32_t lane=0;lane<32;++lane)
      status[p*32+lane]=bcf::decode_slab_lane_lut(data,size,desc+9*p,K,nr,nc,
          labels,ts,nt,chunk,p,lane,slab,coarse,coarse_count,records,record_count,
          offsets,offset_count);
}
extern "C" void bcf_test_panel_slab(const uint8_t* data, uint64_t size,
    const int64_t* desc, uint64_t descriptor_count, uint32_t panels, uint32_t panel,
    uint64_t panel_bytes, uint32_t K, uint32_t nr, uint32_t nc,
    uint64_t labels, uint64_t ts, uint64_t nt, uint32_t chunk,
    uint16_t* slab, uint32_t* status) {
  for (uint32_t p=0;p<4;++p)
    for (uint32_t lane=0;lane<32;++lane)
      status[p*32+lane]=bcf::decode_panel_slab_lane(data,size,desc,descriptor_count,
          panels,panel,panel_bytes,K,nr,nc,labels,ts,nt,chunk,p,lane,slab);
}
extern "C" void bcf_test_panel_slab_lut(const uint8_t* data, uint64_t size,
    const int64_t* desc, uint64_t descriptor_count, uint32_t panels, uint32_t panel,
    uint64_t panel_bytes, uint32_t K, uint32_t nr, uint32_t nc,
    uint64_t labels, uint64_t ts, uint64_t nt, uint32_t chunk,
    const uint32_t* coarse, uint64_t coarse_count, const uint64_t* records,
    uint64_t record_count, const uint32_t* offsets, uint64_t offset_count,
    uint16_t* slab, uint32_t* status) {
  for (uint32_t p=0;p<4;++p)
    for (uint32_t lane=0;lane<32;++lane)
      status[p*32+lane]=bcf::decode_panel_slab_lane_lut(data,size,desc,descriptor_count,
          panels,panel,panel_bytes,K,nr,nc,labels,ts,nt,chunk,p,lane,slab,
          coarse,coarse_count,records,record_count,offsets,offset_count);
}
extern "C" void bcf_test_output_view(const uint8_t* data, uint64_t size, const int64_t* desc,
    uint32_t K, uint32_t nr, uint32_t nc, uint64_t labels, uint64_t ts, uint64_t nt,
    uint32_t chunk, uint32_t stride, uint64_t output_size, uint16_t* slab, uint32_t* status) {
  bct::OutputView output{slab,0,chunk*bcf::CH,bcf::N,bcf::CH,output_size,stride};
  for (uint32_t p=0;p<4;++p)
    for (uint32_t lane=0;lane<32;++lane)
      status[p*32+lane]=bct::decode_lane(data,size,desc+9*p,bcf::N,K,nr,nc,
                                         labels,ts,nt,lane,output);
}
#else
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> bcf_gemm_cuda(torch::Tensor data, torch::Tensor desc,
    torch::Tensor x, int64_t nr, int64_t nc, int64_t labels, int64_t ts, int64_t nt, bool optimized=false);
std::vector<torch::Tensor> bcf_gemm_cuda_lut(torch::Tensor data, torch::Tensor desc,
    torch::Tensor x, int64_t nr, int64_t nc, int64_t labels, int64_t ts, int64_t nt,
    torch::Tensor coarse, torch::Tensor records, torch::Tensor offsets);
torch::Tensor bcf_dense_cuda(torch::Tensor w, torch::Tensor x);
std::vector<torch::Tensor> bcf_gemm_panels_cuda(torch::Tensor data, torch::Tensor desc,
    torch::Tensor x, int64_t nr, int64_t nc, int64_t labels, int64_t ts, int64_t nt);
std::vector<torch::Tensor> bcf_gemm_panels_cuda_lut(torch::Tensor data, torch::Tensor desc,
    torch::Tensor x, int64_t nr, int64_t nc, int64_t labels, int64_t ts, int64_t nt,
    torch::Tensor coarse, torch::Tensor records, torch::Tensor offsets);
torch::Tensor bcf_dense_panels_cuda(torch::Tensor w, torch::Tensor x);

void check_x(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kBFloat16 && x.dim()==2 && x.is_contiguous(),
              "contiguous CUDA bf16 x required");
  TORCH_CHECK((x.size(0)==1 || x.size(0)==16 || x.size(0)==64 || x.size(0)==128) &&
              (x.size(1)==512 || x.size(1)==1024), "bounded M/K required");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr())%16==0, "16-byte aligned x required");
}
std::vector<torch::Tensor> gemm_impl(torch::Tensor data, torch::Tensor desc,
    torch::Tensor x, int64_t nr, int64_t nc, int64_t labels, int64_t ts, int64_t nt, bool optimized) {
  check_x(x);
  TORCH_CHECK(data.is_cuda() && desc.is_cuda() && data.device()==x.device() && desc.device()==x.device(),
              "same CUDA device required");
  TORCH_CHECK(data.scalar_type()==torch::kUInt8 && data.dim()==1 && data.is_contiguous() &&
              data.numel()<=1048576, "bounded contiguous byte bundle required");
  TORCH_CHECK(desc.scalar_type()==torch::kInt64 && desc.dim()==2 && desc.size(1)==9 &&
              desc.size(0)==4*(x.size(1)/128) && desc.is_contiguous(), "chunk-major tile descriptors required");
  TORCH_CHECK(nr>0 && nr<=64 && nc>0 && nc<=16 && nr*nc<=1024, "bounded contexts required");
  TORCH_CHECK(labels>=0 && labels<=data.numel() && 64+x.size(1)<=data.numel()-labels &&
              ts>=0 && ts<=data.numel() && nt>=0 && nt<=data.numel()-ts, "bounded metadata required");
  return bcf_gemm_cuda(data,desc,x,nr,nc,labels,ts,nt,optimized);
}
std::vector<torch::Tensor> gemm(torch::Tensor data, torch::Tensor desc,
    torch::Tensor x, int64_t nr, int64_t nc, int64_t labels, int64_t ts, int64_t nt) {
  return gemm_impl(data,desc,x,nr,nc,labels,ts,nt,false);
}
std::vector<torch::Tensor> gemm_opt(torch::Tensor data, torch::Tensor desc,
    torch::Tensor x, int64_t nr, int64_t nc, int64_t labels, int64_t ts, int64_t nt) {
  return gemm_impl(data,desc,x,nr,nc,labels,ts,nt,true);
}
void check_lut(torch::Tensor data, torch::Tensor desc, torch::Tensor x,
    int64_t nr, int64_t nc, int64_t labels, int64_t ts, int64_t nt,
    torch::Tensor coarse, torch::Tensor records, torch::Tensor offsets, bool panels) {
  check_x(x);
  TORCH_CHECK(data.is_cuda() && desc.is_cuda() && coarse.is_cuda() && records.is_cuda() && offsets.is_cuda() &&
              data.device()==x.device() && desc.device()==x.device() && coarse.device()==x.device() &&
              records.device()==x.device() && offsets.device()==x.device(), "same CUDA device required");
  TORCH_CHECK(data.scalar_type()==torch::kUInt8 && data.is_contiguous() &&
              (panels ? data.dim()==2 && data.size(0)>=1 && data.size(0)<=1024 && data.size(1)>0 && data.size(1)<=1048576
                      : data.dim()==1 && data.numel()<=1048576), "bounded contiguous byte bundle required");
  TORCH_CHECK(desc.scalar_type()==torch::kInt64 && desc.is_contiguous() &&
              (panels ? desc.dim()==3 && desc.size(0)==data.size(0) && desc.size(1)==4*(x.size(1)/128) && desc.size(2)==9
                      : desc.dim()==2 && desc.size(0)==4*(x.size(1)/128) && desc.size(1)==9),
              "chunk-major tile descriptors required");
  TORCH_CHECK(nr>0 && nr<=64 && nc>0 && nc<=16 && nr*nc<=1024, "bounded contexts required");
  const int64_t nctx=nr*nc, size=panels?data.size(1):data.numel();
  TORCH_CHECK(labels>=0 && labels<=size && 64+x.size(1)<=size-labels &&
              ts>=0 && ts<=size && nt>=0 && nt<=size-ts, "bounded metadata required");
  TORCH_CHECK(coarse.scalar_type()==torch::kUInt32 && records.scalar_type()==torch::kUInt64 &&
              offsets.scalar_type()==torch::kUInt32 && coarse.is_contiguous() && records.is_contiguous() && offsets.is_contiguous(),
              "contiguous uint32/uint64 CDF lookup arrays required");
  TORCH_CHECK(coarse.device()==x.device() && records.device()==x.device() && offsets.device()==x.device(),
              "CDF lookup arrays must share the activation device");
  if (nt==0) {
    TORCH_CHECK(coarse.dim()==2 && coarse.size(0)==0 && coarse.size(1)==256 &&
                records.dim()==1 && records.numel()==0 && offsets.dim()==1 && offsets.numel()==0,
                "raw-only lookup tensors must be empty");
  } else {
    TORCH_CHECK(coarse.dim()==2 && coarse.size(0)==nctx && coarse.size(1)==256 &&
                records.dim()==1 && records.numel()<=nctx*1024 &&
                offsets.dim()==1 && offsets.numel()==nctx+1,
                "CDF lookup tensor geometry mismatch");
  }
}
std::vector<torch::Tensor> gemm_opt_lut(torch::Tensor data, torch::Tensor desc,
    torch::Tensor x, int64_t nr, int64_t nc, int64_t labels, int64_t ts, int64_t nt,
    torch::Tensor coarse, torch::Tensor records, torch::Tensor offsets) {
  check_lut(data,desc,x,nr,nc,labels,ts,nt,coarse,records,offsets,false);
  return bcf_gemm_cuda_lut(data,desc,x,nr,nc,labels,ts,nt,coarse,records,offsets);
}
torch::Tensor dense(torch::Tensor w, torch::Tensor x) {
  check_x(x);
  TORCH_CHECK(w.is_cuda() && w.device()==x.device() && w.scalar_type()==torch::kBFloat16 &&
              w.dim()==2 && w.size(0)==64 && w.size(1)==x.size(1) && w.is_contiguous(),
              "contiguous CUDA bf16 weight [64,K] required");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(w.data_ptr())%16==0, "16-byte aligned weights required");
  return bcf_dense_cuda(w,x);
}
std::vector<torch::Tensor> gemm_opt_panels(torch::Tensor data, torch::Tensor desc,
    torch::Tensor x, int64_t nr, int64_t nc, int64_t labels, int64_t ts, int64_t nt) {
  check_x(x);
  TORCH_CHECK(data.is_cuda() && desc.is_cuda() && data.device()==x.device() && desc.device()==x.device(),
              "same CUDA device required");
  TORCH_CHECK(data.scalar_type()==torch::kUInt8 && data.dim()==2 && data.is_contiguous() &&
              data.size(0)>=1 && data.size(0)<=1024 && data.size(1)>0 && data.size(1)<=1048576,
              "bounded contiguous byte panels [P,B], 1<=P<=1024 required");
  TORCH_CHECK(desc.scalar_type()==torch::kInt64 && desc.dim()==3 && desc.size(0)==data.size(0) &&
              desc.size(1)==4*(x.size(1)/128) && desc.size(2)==9 && desc.is_contiguous(),
              "panel/chunk-major descriptors [P,packets,9] required");
  TORCH_CHECK(nr>0 && nr<=64 && nc>0 && nc<=16 && nr*nc<=1024, "bounded contexts required");
  const auto size=data.size(1);
  TORCH_CHECK(labels>=0 && labels<=size && 64+x.size(1)<=size-labels &&
              ts>=0 && ts<=size && nt>=0 && nt<=size-ts, "bounded panel-local metadata required");
  return bcf_gemm_panels_cuda(data,desc,x,nr,nc,labels,ts,nt);
}
std::vector<torch::Tensor> gemm_opt_panels_lut(torch::Tensor data, torch::Tensor desc,
    torch::Tensor x, int64_t nr, int64_t nc, int64_t labels, int64_t ts, int64_t nt,
    torch::Tensor coarse, torch::Tensor records, torch::Tensor offsets) {
  check_lut(data,desc,x,nr,nc,labels,ts,nt,coarse,records,offsets,true);
  return bcf_gemm_panels_cuda_lut(data,desc,x,nr,nc,labels,ts,nt,coarse,records,offsets);
}
torch::Tensor dense_panels(torch::Tensor w, torch::Tensor x) {
  check_x(x);
  TORCH_CHECK(w.is_cuda() && w.device()==x.device() && w.scalar_type()==torch::kBFloat16 &&
              w.dim()==2 && w.size(0)>=64 && w.size(0)<=65536 && w.size(0)%64==0 &&
              w.size(1)==x.size(1) && w.is_contiguous(),
              "contiguous CUDA bf16 weight [P*64,K], 1<=P<=1024 required");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(w.data_ptr())%16==0, "16-byte aligned weights required");
  return bcf_dense_panels_cuda(w,x);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm", &gemm, "BCT2 fused slab decode / fixed-order BI-GEMM (y, status)");
  m.def("gemm_opt", &gemm_opt, "Opt-in fixed16x128 BCT2 fused decoder (y, status)");
  m.def("gemm_opt_lut", &gemm_opt_lut, "Opt-in BCT2 fused decoder with validated sparse CDF lookup (y, status)");
  m.def("gemm_opt_panels", &gemm_opt_panels, "Opt-in independent resident BCT2 panels (y, status)");
  m.def("gemm_opt_panels_lut", &gemm_opt_panels_lut, "Opt-in BCT2 panels with shared sparse CDF lookup (y, status)");
  m.def("dense_panels", &dense_panels, "BI dense panel control, S=1, tile=64, no bias");
  m.def("dense", &dense, "Existing BI-GEMM BF16 engine, S=1, tile=64, no bias");
}
#endif
