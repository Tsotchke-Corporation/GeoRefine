#include "context_exponent_packet.cuh"
#ifdef BHE_CPU_TEST
#include <algorithm>
#include <vector>
extern "C" int bhe_cpu_decode(const uint8_t* main,uint64_t main_bytes,const uint8_t* raw,uint64_t raw_bytes,
    const uint32_t* tables,uint64_t table_words,const int64_t* contexts,const uint32_t* offsets,
    const int32_t* rows,const int64_t* classes,const uint8_t* positions,const int64_t* counts,const int64_t* panels,
    uint32_t N,uint32_t K,uint32_t nc,uint32_t tiles,uint16_t* output,int32_t* status,uint32_t* sink){
  if(!N || N>262144 || N%64 || K<128 || K>32768 || K%128 || (nc!=1&&nc!=2&&nc!=4))return 1;
  std::vector<uint16_t> slab(64*bhe::LDS);
  for(uint32_t tile=0;tile<tiles;++tile){
    bhe::Cursor cursors[256]{};bhe::Context ctx[256]{};uint32_t hashes[256];
    std::fill(hashes,hashes+256,2166136261u);uint32_t failed=0;
    const int32_t* rowmap=rows+uint64_t(tile)*64;
    for(uint32_t tid=0;tid<256;++tid){
      const uint32_t r=tid/nc,cc=tid%nc;
      if(r>=64 || rowmap[r]<0)continue;
      const uint32_t original=rowmap[r];const int64_t* pd=panels+uint64_t(original/64)*3;
      if(pd[2])continue;
      const int64_t* d=contexts+(classes[tile]*nc+cc)*4;
      if(d[0]<0 || d[1]<0 || uint64_t(d[0])>table_words || uint64_t(d[1])>table_words-uint64_t(d[0]))return 2;
      ctx[tid]={tables+d[0],uint32_t(d[1]),int32_t(d[2]),uint32_t(d[3])};
      const uint32_t* span=offsets+uint64_t(original/64)*(64*nc+1)+(original%64)*nc+cc;
      failed|=bhe::init(cursors[tid],main,main_bytes,uint64_t(pd[0])*8+span[0],uint64_t(pd[0])*8+span[1]);
    }
    for(uint32_t chunk=0;chunk<K/128 && !failed;++chunk){
      std::fill(slab.begin(),slab.end(),0);
      for(uint32_t tid=0;tid<256;++tid){
        const uint32_t r=tid/nc,cc=tid%nc;if(r>=64 || rowmap[r]<0)continue;
        const int64_t original=rowmap[r];if(original>=N)return 2;
        const int64_t* pd=panels+uint64_t(original/64)*3;
        const uint64_t rawbase=uint64_t(pd[1])+uint64_t(original%64)*K*(pd[2]?2:1);
        if(pd[2]){
          if(cc)continue;
          const uint64_t pos=rawbase+uint64_t(chunk)*256;
          if(pos>raw_bytes || raw_bytes-pos<256){failed|=4;continue;}
          for(uint32_t col=0;col<128;++col)slab[r*bhe::LDS+col]=uint16_t(raw[pos+col*2])|uint16_t(raw[pos+col*2+1])<<8;
          continue;
        }
        const int64_t* bounds=counts+uint64_t(chunk)*(nc+1);
        if(bounds[cc]<0 || bounds[cc+1]<bounds[cc] || bounds[cc+1]>128)return 2;
        const uint32_t first=uint32_t(bounds[cc]),count=uint32_t(bounds[cc+1]-bounds[cc]);
        for(uint32_t produced=0;produced<count;){
          bhe::Group group;if(bhe::group(ctx[tid],cursors[tid],count-produced,group)){failed|=cursors[tid].error;break;}
          for(uint32_t j=0;j<group.count;++j){
            const uint32_t col=positions[uint64_t(chunk)*128+first+produced+j];
            const uint64_t pos=rawbase+uint64_t(chunk)*128+col;
            if(col>=128 || pos>=raw_bytes){failed|=4;break;}
            slab[r*bhe::LDS+col]=bhe::combine((group.symbols>>(8*j))&255,raw[pos]);
          }
          produced+=group.count;
        }
      }
      if(!failed)for(uint32_t tid=0;tid<256;++tid)for(uint32_t i=tid;i<8192;i+=256){
        const uint32_t r=i>>7,col=i&127;const int64_t original=rowmap[r];
        if(original<0)continue;
        const uint16_t word=slab[r*bhe::LDS+col];output[uint64_t(original)*K+chunk*128+col]=word;
        hashes[tid]=(hashes[tid]^uint32_t(word))*16777619u;
      }
    }
    for(uint32_t tid=0;tid<256;++tid){
      const uint32_t r=tid/nc;
      if(r<64 && rowmap[r]>=0 && panels[uint64_t(rowmap[r]/64)*3+2]==0)failed|=bhe::finish(cursors[tid]);
    }
    for(uint32_t tid=0;tid<256;++tid){
      status[tile*256+tid]=cursors[tid].error?cursors[tid].error:failed?32:0;
      sink[tile*256+tid]=hashes[tid];
    }
  }
  return 0;
}
#else
#include <torch/extension.h>
#include <atomic>
#include <memory>
#include <string>
#include <vector>
namespace py=pybind11;
std::vector<torch::Tensor> bhe_gemm_cuda(const std::vector<torch::Tensor>&,torch::Tensor,int64_t,int64_t,int64_t,int64_t);
std::vector<torch::Tensor> bhe_gemm_out_cuda(const std::vector<torch::Tensor>&,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);
std::vector<torch::Tensor> bhe_gemm_trace_cuda(const std::vector<torch::Tensor>&,torch::Tensor,int64_t,int64_t,int64_t,int64_t);
std::vector<torch::Tensor> bhe_decode_cuda(const std::vector<torch::Tensor>&,int64_t,int64_t,int64_t,int64_t,bool);
torch::Tensor bhe_dense_cuda(torch::Tensor,torch::Tensor,int64_t,int64_t);
int bhe_prepare_cuda(torch::Tensor,int64_t);
class ResidentExponentPlan {
  std::vector<torch::Tensor> tensors_;
  int64_t N_=0,K_=0,nc_=0,stage_words_=0;
  int64_t splits_=1;
  std::string provenance_;
  friend std::shared_ptr<ResidentExponentPlan> seal(py::dict,const std::string&,int64_t);
  friend std::vector<torch::Tensor> gemm(std::shared_ptr<ResidentExponentPlan>,torch::Tensor);
  friend std::vector<torch::Tensor> gemm_out(std::shared_ptr<ResidentExponentPlan>,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);
  friend std::vector<torch::Tensor> gemm_trace(std::shared_ptr<ResidentExponentPlan>,torch::Tensor);
  friend std::vector<torch::Tensor> decode(std::shared_ptr<ResidentExponentPlan>);
  friend std::vector<torch::Tensor> produce(std::shared_ptr<ResidentExponentPlan>);
  friend py::dict plan_provenance(std::shared_ptr<ResidentExponentPlan>);
};
static std::atomic<uint64_t> generation{0};
std::shared_ptr<ResidentExponentPlan> seal(py::dict arrays,const std::string& device,int64_t splits=1){
  auto torchpy=py::module_::import("torch"),np=py::module_::import("numpy");auto dev=torchpy.attr("device")(device);
  TORCH_CHECK(py::str(dev.attr("type")).cast<std::string>()=="cuda" && !dev.attr("index").is_none(),"explicit CUDA device required");
  auto p=std::make_shared<ResidentExponentPlan>();
  p->N_=arrays["N"].cast<int64_t>();p->K_=arrays["K"].cast<int64_t>();p->nc_=arrays["nc"].cast<int64_t>();
  TORCH_CHECK(splits>=1 && splits<=8 && splits<=p->K_/128,"splits must be 1..8 and no greater than K/128");
  p->splits_=splits;
  const int64_t nr=arrays["nr"].cast<int64_t>();
  TORCH_CHECK(p->N_>=64 && p->N_<=262144 && p->N_%64==0 && p->K_>=128 && p->K_<=32768 && p->K_%128==0 &&
      (p->nc_==1||p->nc_==2||p->nc_==4) && nr>=1 && nr<=256,"unsupported exponent plan geometry");
  py::object dir=arrays["context_dir"];
  const auto counts=dir.attr("__getitem__")(py::make_tuple(py::slice(py::none(),py::none(),py::none()),1));
  p->stage_words_=counts.attr("reshape")(nr,p->nc_).attr("sum")(1).attr("max")().cast<int64_t>();
  TORCH_CHECK(p->stage_words_>=0 && p->stage_words_<=16384,"context staging exceeds bounded64KiB");
  const char* keys[]={"main","raw","tables","context_dir","stream_offsets","row_map","tile_rc","positions","counts","panel_desc"};
  int64_t resident=0,largest=0;
  for(const char* key:keys){
    py::object array=arrays[key];
    if(std::string(key)=="tables"||std::string(key)=="stream_offsets")array=array.attr("view")(np.attr("int32"));
    auto t=torchpy.attr("from_numpy")(array.attr("copy")()).attr("to")(dev).cast<torch::Tensor>();
    TORCH_CHECK(t.is_contiguous(),"private plan tensors must be contiguous");
    const int64_t bytes=t.numel()*t.element_size();resident+=bytes;largest=std::max(largest,bytes);
    p->tensors_.push_back(std::move(t));
  }
  if(splits>1){
    TORCH_CHECK(arrays.contains("split_checkpoints"),
        "split_checkpoints required when splits>1");
    auto ck=torchpy.attr("from_numpy")(arrays["split_checkpoints"].attr("copy")()).attr("to")(dev).cast<torch::Tensor>();
    TORCH_CHECK(ck.is_contiguous() && ck.scalar_type()==torch::kInt64 && ck.numel()==p->N_*p->nc_*(splits-1),
        "invalid split_checkpoints shape/type");
    const int64_t checkpoint_bytes=ck.numel()*ck.element_size();
    resident+=checkpoint_bytes;largest=std::max(largest,checkpoint_bytes);
    p->tensors_.push_back(std::move(ck));
  }
  TORCH_CHECK(reinterpret_cast<uintptr_t>(p->tensors_[0].data_ptr())%4==0 &&
      reinterpret_cast<uintptr_t>(p->tensors_[1].data_ptr())%16==0,"private payload alignment");
  const int max_tile=bhe_prepare_cuda(p->tensors_[0],p->stage_words_);
  torchpy.attr("cuda").attr("synchronize")(dev);
  py::dict provenance;
  for(const char* key:{"N","K","nr","nc","tiles","panels","matrix_mode","host_wire_bytes","source_digests",
      "matrix_source_digest","row_permutation_digest","source_cdf_digest","map_digest","host_peak_decoded_row_bytes"})
    provenance[key]=arrays[key];
  provenance["generation"]=++generation;provenance["device"]=device;provenance["sealed"]=true;
  provenance["gpu_plan_bytes"]=resident;provenance["gpu_upload_tensor_peak_bytes"]=resident;
  const int64_t host_arrays=resident+arrays["expanded_diagnostic_bytes"].cast<int64_t>();
  provenance["host_staging_bytes"]=resident;provenance["host_array_bytes"]=host_arrays;
  provenance["host_staging_peak_bytes"]=host_arrays+largest;
  provenance["host_peak_is_complete"]=false;
  provenance["host_peak_exclusions"]="wire/table construction temporaries, Python objects, allocator/driver reservations";
  provenance["context_stage_words"]=p->stage_words_;provenance["S"]=p->splits_;provenance["splits"]=p->splits_;provenance["tiles"]=arrays["tiles"];provenance["activation_tile"]=64;
  provenance["max_activation_tile"]=max_tile;
  provenance["split_status_layout"]="[S,tiles,blocks,256]";
  provenance["split_checkpoint_words"]=p->splits_>1?p->N_*p->nc_*(p->splits_-1):0;
  provenance["shared_configured_at_prepare"]=true;provenance["shared_configuration_monotonic_per_device"]=true;
  provenance["allocator_driver_peak_measured"]=false;
  if(arrays.contains("validation_witnesses_reused"))
    provenance["validation_witnesses_reused"]=arrays["validation_witnesses_reused"];
  if(arrays.contains("validation_full_decodes"))
    provenance["validation_full_decodes"]=arrays["validation_full_decodes"];
  provenance["source_binding_kind"]="ordered original-row panel source SHA256 digests plus row permutation digest";
  provenance["producer_checksum_cost"]="per-thread word FNV sink over shared CH slabs, skipped padded rows; latency includes checksum";
  p->provenance_=py::module_::import("json").attr("dumps")(provenance).cast<std::string>();return p;
}
std::shared_ptr<ResidentExponentPlan> prepare(py::bytes blob,const std::string& device,int64_t panels,int64_t splits=1){
  TORCH_CHECK(panels>=1&&panels<=4096,"panels must be1..4096");
  auto packet=py::module_::import("georefine_context_runtime.context_exponent_packet");
  auto arrays=packet.attr("_resident_arrays")(blob,panels).cast<py::dict>();
  if(splits>1)arrays=py::module_::import("georefine_context_runtime.context_exponent_splits").attr("add_split_checkpoints")(arrays,splits).cast<py::dict>();
  return seal(arrays,device,splits);
}
std::shared_ptr<ResidentExponentPlan> prepare_matrix(py::list blobs,const std::string& device,
    int64_t splits=1,py::object validation_witnesses=py::none()){
  TORCH_CHECK(py::len(blobs)>=1&&py::len(blobs)<=4096,"matrix panels must be1..4096");
  auto packet=py::module_::import("georefine_context_runtime.context_exponent_packet");
  py::object arrays;
  if(validation_witnesses.is_none()){
    arrays=packet.attr("_resident_matrix_arrays")(blobs);
  }else{
    arrays=packet.attr("_resident_matrix_arrays")(blobs,py::none(),validation_witnesses);
  }
  auto typed_arrays=arrays.cast<py::dict>();
  if(splits>1)typed_arrays=py::module_::import("georefine_context_runtime.context_exponent_splits").attr("add_split_checkpoints")(typed_arrays,splits).cast<py::dict>();
  return seal(typed_arrays,device,splits);
}
static void check_x(torch::Tensor x){
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kBFloat16 && x.dim()==2 && x.is_contiguous(),"contiguous CUDA bf16 x required");
  TORCH_CHECK((x.size(0)>=1&&x.size(0)<=512) && x.size(1)>=128 && x.size(1)<=32768 && x.size(1)%128==0,"bounded M/K required");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr())%16==0,"aligned x required");
}
std::vector<torch::Tensor> gemm(std::shared_ptr<ResidentExponentPlan> p,torch::Tensor x){
  TORCH_CHECK(p && !p->tensors_.empty(),"sealed exponent plan required");check_x(x);
  TORCH_CHECK(p->tensors_[0].device()==x.device() && p->K_==x.size(1),"wrong device or shape for exponent plan");
  if(p->splits_==1)return bhe_gemm_cuda(p->tensors_,x,p->N_,p->K_,p->nc_,p->stage_words_);
  auto out=torch::empty({x.size(0),p->N_},x.options());
  auto ws=torch::empty({p->splits_,x.size(0),p->N_},x.options().dtype(torch::kFloat));
  auto status=torch::empty({p->splits_,p->tensors_[6].numel(),(x.size(0)+63)/64,256},x.options().dtype(torch::kInt32));
  auto r=bhe_gemm_out_cuda(p->tensors_,x,out,ws,status,p->N_,p->K_,p->nc_,p->stage_words_,p->splits_,64);
  return {r[0],r[2]};
}
std::vector<torch::Tensor> gemm_out(std::shared_ptr<ResidentExponentPlan> p,torch::Tensor x,torch::Tensor out,torch::Tensor ws,torch::Tensor status,int64_t tile=64){
  TORCH_CHECK(tile==64||tile==128,"BHE1 tile must be64 or128");
  TORCH_CHECK(p && !p->tensors_.empty(),"sealed exponent plan required");check_x(x);
  TORCH_CHECK(out.is_cuda() && out.device()==x.device() && out.scalar_type()==torch::kBFloat16 && out.dim()==2 && out.size(0)==x.size(0) && out.size(1)==p->N_ && out.stride(1)==1 && out.stride(0)>=p->N_,"CUDA bf16 out [M,N] with unit columns required");
  TORCH_CHECK(ws.is_cuda() && ws.device()==x.device() && ws.scalar_type()==torch::kFloat && ws.is_contiguous() && ws.numel()>=p->splits_*x.size(0)*p->N_,"contiguous CUDA fp32 workspace capacity S*M*N required");
  const bool status_ok=p->splits_==1 ? (status.dim()==3 && status.size(0)==p->tensors_[6].numel() && status.size(1)==(x.size(0)+tile-1)/tile && status.size(2)==256) :
      (status.dim()==4 && status.size(0)==p->splits_ && status.size(1)==p->tensors_[6].numel() && status.size(2)==(x.size(0)+tile-1)/tile && status.size(3)==256);
  TORCH_CHECK(status.is_cuda() && status.device()==x.device() && status.scalar_type()==torch::kInt32 && status_ok && status.is_contiguous(),"contiguous CUDA status [S,tiles,blocks,256] required");
  TORCH_CHECK(p->tensors_[0].device()==x.device() && p->K_==x.size(1),"wrong device or shape for exponent plan");
  return bhe_gemm_out_cuda(p->tensors_,x,out,ws,status,p->N_,p->K_,p->nc_,p->stage_words_,p->splits_,tile);
}
std::vector<torch::Tensor> decode(std::shared_ptr<ResidentExponentPlan> p){
  TORCH_CHECK(p && !p->tensors_.empty(),"sealed exponent plan required");
  return bhe_decode_cuda(p->tensors_,p->N_,p->K_,p->nc_,p->stage_words_,false);
}
std::vector<torch::Tensor> gemm_trace(std::shared_ptr<ResidentExponentPlan> p,torch::Tensor x){
  TORCH_CHECK(p && !p->tensors_.empty(),"sealed exponent plan required");check_x(x);
  TORCH_CHECK(p->tensors_[0].device()==x.device() && p->K_==x.size(1),"wrong device or shape for exponent plan");
  return bhe_gemm_trace_cuda(p->tensors_,x,p->N_,p->K_,p->nc_,p->stage_words_);
}
std::vector<torch::Tensor> produce(std::shared_ptr<ResidentExponentPlan> p){
  TORCH_CHECK(p && !p->tensors_.empty(),"sealed exponent plan required");
  return bhe_decode_cuda(p->tensors_,p->N_,p->K_,p->nc_,p->stage_words_,true);
}
py::dict plan_provenance(std::shared_ptr<ResidentExponentPlan> p){
  TORCH_CHECK(p,"sealed exponent plan required");return py::module_::import("json").attr("loads")(p->provenance_).cast<py::dict>();
}
torch::Tensor dense(torch::Tensor w,torch::Tensor x,int64_t splits=1,int64_t tile=64){
  TORCH_CHECK(tile==64||tile==128,"dense tile must be64 or128");
  check_x(x);
  TORCH_CHECK(w.is_cuda() && w.device()==x.device() && w.scalar_type()==torch::kBFloat16 && w.dim()==2 && w.size(0)>=64 &&
      w.size(0)<=262144 && w.size(0)%64==0 && w.size(1)==x.size(1) && w.is_contiguous(),"contiguous CUDA bf16 weights[N,K] required");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(w.data_ptr())%16==0,"aligned weights required");return bhe_dense_cuda(w,x,splits,tile);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
  py::class_<ResidentExponentPlan,std::shared_ptr<ResidentExponentPlan>>(m,"ResidentExponentPlan");
  m.def("prepare",&prepare,py::arg("blob"),py::arg("device"),py::arg("panels"),py::arg("splits")=1);m.def("prepare_matrix",&prepare_matrix,py::arg("blobs"),py::arg("device"),py::arg("splits")=1,py::arg("validation_witnesses")=py::none());m.def("plan_provenance",&plan_provenance);
  m.def("gemm",&gemm);m.def("gemm_out",&gemm_out,py::arg("plan"),py::arg("x"),py::arg("out"),py::arg("workspace"),py::arg("status"),py::arg("tile")=64);m.def("decode",&decode);m.def("produce",&produce);m.def("dense",&dense,py::arg("w"),py::arg("x"),py::arg("splits")=1,py::arg("tile")=64);
  m.def("gemm_trace",&gemm_trace,"Diagnostic fused cursor snapshots before terminal checks; not for timing");
}
#endif
