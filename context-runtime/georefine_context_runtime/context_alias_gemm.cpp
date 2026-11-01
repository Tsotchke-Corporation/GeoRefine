#include "context_alias_packet.cuh"
#include <new>
static_assert(sizeof(bca::PacketView)<=128 && alignof(bca::PacketView)<=16,"prepared view must fit aligned shared slot");
#ifdef BCA_GEMM_CPU_TEST
#ifndef BCA_TEST_SCALAR_RECORDS
#define BCA_TEST_SCALAR_RECORDS false
#endif
#ifndef BCA_TEST_ALIGNED_OFFSETS
#define BCA_TEST_ALIGNED_OFFSETS false
#endif
extern "C" void bcag_test_decode(const uint8_t* data,uint64_t bytes,const int64_t* desc,
    const uint8_t* meta,uint64_t meta_bytes,uint32_t buckets,uint32_t panels,uint32_t K,
    uint32_t nr,uint32_t nc,uint16_t* slabs,uint32_t* status,uint32_t* preparations) {
  if(!panels || panels>2 || (K!=512 && K!=1024))return;
  for(uint32_t p=0;p<panels;++p){
    bca::PacketView views[4];bca::LaneCursor cursors[4][32];uint32_t failed=0;
    for(uint32_t w=0;w<4;++w){
      const auto* d=desc+(p*4+w)*9;
      bca::prepare_packet<BCA_TEST_ALIGNED_OFFSETS>(data+uint64_t(p)*bytes,bytes,d,meta,meta_bytes,buckets,K,nr,nc,views[w]);
      if(d[1]!=16*w)views[w].error=1;
      ++preparations[p];
      for(uint32_t lane=0;lane<32;++lane)failed|=bca::init_lane(views[w],lane,cursors[w][lane]);
    }
    for(uint32_t c=0;c<K/128 && !failed;++c)
      for(uint32_t w=0;w<4;++w)
        failed|=bca::decode_group<BCA_TEST_SCALAR_RECORDS,BCA_TEST_ALIGNED_OFFSETS>(views[w],c,cursors[w],slabs+(uint64_t(p)*(K/128)+c)*64*136,64*136);
    for(uint32_t w=0;w<4;++w)for(uint32_t lane=0;lane<32;++lane)
      status[p*128+w*32+lane]=cursors[w][lane].error?cursors[w][lane].error:failed?32:0;
  }
}
extern "C" uint32_t bcag_test_order(const uint8_t* data,uint64_t bytes,const int64_t* desc,
    const uint8_t* meta,uint64_t meta_bytes,uint32_t buckets,uint32_t K,uint32_t nr,uint32_t nc,
    uint16_t* slab,uint32_t chunk,uint64_t words) {
  bca::PacketView v;bca::LaneCursor c[32];
  bca::prepare_packet<BCA_TEST_ALIGNED_OFFSETS>(data,bytes,desc,meta,meta_bytes,buckets,K,nr,nc,v);
  for(uint32_t l=0;l<32;++l)bca::init_lane(v,l,c[l]);
  return bca::decode_group<BCA_TEST_SCALAR_RECORDS,BCA_TEST_ALIGNED_OFFSETS>(v,chunk,c,slab,words);
}
extern "C" void bcag_test_all_slots(const uint8_t* meta,uint64_t meta_bytes,uint32_t buckets,
    uint16_t* symbols,uint32_t* next_states,uint32_t* status){
  uint8_t data=0;const int64_t d[9]={1,0,0,16,512,0,32*512,32*512,0};
  bca::PacketView v;
  const uint32_t prep=bca::prepare_packet<BCA_TEST_ALIGNED_OFFSETS>(&data,32*512,d,meta,meta_bytes,buckets,512,1,1,v);
  for(uint32_t slot=0;slot<65536;++slot){
    bca::LaneCursor c{};c.state=65536|slot;uint32_t symbol=0;
    status[slot]=prep?prep:bca::next_symbol<BCA_TEST_SCALAR_RECORDS,BCA_TEST_ALIGNED_OFFSETS>(v,0,0,c,symbol);
    symbols[slot]=uint16_t(symbol);next_states[slot]=c.state;
  }
}
extern "C" uint32_t bcag_test_large_offset(){
  alignas(16) uint8_t sentinel=0;bca::PacketView v;
  const int64_t offset=int64_t(1)<<32;
  const int64_t d[9]={1,0,0,16,512,offset,32*512,offset+32*512,0};
  const uint32_t code=bca::prepare_packet<BCA_TEST_ALIGNED_OFFSETS>(&sentinel,uint64_t(offset)+32*512,d,
      &sentinel,4*2+1+128+512+96,0,512,1,1,v,128,64);
  // Preparation never dereferences payload/metadata: this tests >u32 absolute
  // offsets without allocating or claiming a decoded multi-gigabyte fixture.
  return code || v.ms!=uint64_t(offset) || v.rs!=uint64_t(offset)+32*512 || sizeof(bca::LaneCursor{}.pos)!=8;
}
extern "C" void bcag_test_matrix(const uint8_t* data,uint64_t bytes,const int64_t* desc,
    const uint8_t* meta,uint64_t meta_bytes,uint32_t buckets,uint32_t panels,uint32_t K,
    uint32_t nr,uint32_t nc,uint16_t* slabs,uint32_t* status,uint32_t* checksum){
  if(!panels || panels>2 || K!=512)return;
  for(uint32_t p=0;p<panels;++p){
    bca::PacketView views[4];bca::LaneCursor cursors[4][32];uint32_t failed=0;
    for(uint32_t l=0;l<128;++l)checksum[p*128+l]=2166136261u;
    for(uint32_t w=0;w<4;++w){
      const auto* d=desc+(p*4+w)*9;
      bca::prepare_packet<BCA_TEST_ALIGNED_OFFSETS>(data,bytes,d,meta,meta_bytes,buckets,K,nr,nc,views[w],panels*64,p*64);
      if(d[1]!=16*w)views[w].error=1;
      for(uint32_t lane=0;lane<32;++lane)failed|=bca::init_lane(views[w],lane,cursors[w][lane]);
    }
    for(uint32_t c=0;c<K/128 && !failed;++c){
      auto* slab=slabs+(uint64_t(p)*(K/128)+c)*64*136;
      for(uint32_t w=0;w<4;++w)failed|=bca::decode_group<BCA_TEST_SCALAR_RECORDS,BCA_TEST_ALIGNED_OFFSETS>(views[w],c,cursors[w],slab,64*136);
      if(!failed)for(uint32_t l=0;l<128;++l)for(uint32_t row=0;row<64;++row)
        checksum[p*128+l]=(checksum[p*128+l]^uint32_t(slab[row*136+l]))*16777619u;
    }
    for(uint32_t w=0;w<4;++w)for(uint32_t lane=0;lane<32;++lane)
      status[p*128+w*32+lane]=cursors[w][lane].error?cursors[w][lane].error:failed?32:0;
  }
}
template<bool MainFifo=false,bool StageRaw6=false>
static void bcag_test_trusted_decode_impl(const uint8_t* data,uint64_t bytes,const int64_t* desc,
    const uint8_t* meta,uint64_t meta_bytes,uint32_t buckets,uint32_t panels,uint32_t K,
    uint32_t nr,uint32_t nc,uint32_t matrix,uint16_t* slabs,uint32_t* status,uint32_t* checksum){
  if(!panels || panels>2 || uint64_t(panels)*64*K>65536 || (K!=512 && K!=1024) || matrix>1)return;
  for(uint32_t p=0;p<panels;++p){
    bca::PacketView views[4];bca::TrustedPacket trusted[4];bca::TrustedCursor cursors[4][32];uint32_t failed=0;
    bca::TrustedFifo fifo[4];
    alignas(16) uint16_t fifo_storage[4][bca::MAIN_FIFO_BYTES/2+32];
    alignas(16) uint16_t raw6_storage[4][bca::RAW6_STAGE_BYTES/2+32];
    for(uint32_t w=0;w<4;++w){
      for(uint32_t i=0;i<bca::MAIN_FIFO_BYTES/2+32;++i)fifo_storage[w][i]=0xcafe;
      for(uint32_t i=0;i<bca::RAW6_STAGE_BYTES/2+32;++i)raw6_storage[w][i]=0xcafe;
      fifo[w].words=fifo_storage[w]+16;
    }
    for(uint32_t lane=0;lane<128;++lane)checksum[p*128+lane]=2166136261u;
    for(uint32_t w=0;w<4;++w){
      const auto* d=desc+(p*4+w)*9;
      bca::prepare_packet<true>(data+(matrix?0:uint64_t(p)*bytes),bytes,d,meta,meta_bytes,buckets,K,nr,nc,views[w],matrix?panels*64:64,matrix?p*64:0);
      if(d[1]!=16*w)views[w].error=1;
      for(uint32_t lane=0;lane<32;++lane)failed|=bca::init_trusted(views[w],lane,trusted[w],cursors[w][lane]);
    }
    for(uint32_t chunk=0;chunk<K/128 && !failed;++chunk){
      auto* slab=slabs+(uint64_t(p)*(K/128)+chunk)*64*136;
      for(uint32_t w=0;w<4;++w)failed|=bca::decode_trusted_group<MainFifo,StageRaw6>(trusted[w],chunk,chunk+1==K/128,cursors[w],slab,&fifo[w],reinterpret_cast<uint8_t*>(raw6_storage[w]+16));
      if(!failed)for(uint32_t lane=0;lane<128;++lane)for(uint32_t row=0;row<64;++row)
        checksum[p*128+lane]=(checksum[p*128+lane]^uint32_t(slab[row*136+lane]))*16777619u;
    }
    for(uint32_t w=0;w<4;++w)for(uint32_t lane=0;lane<32;++lane)
      status[p*128+w*32+lane]=cursors[w][lane].error?cursors[w][lane].error:failed?32:0;
    for(uint32_t w=0;w<4;++w)for(uint32_t i=0;i<16;++i){
      if constexpr(MainFifo)if(fifo_storage[w][i]!=0xcafe || fifo_storage[w][bca::MAIN_FIFO_BYTES/2+16+i]!=0xcafe)status[p*128]=64;
      if constexpr(StageRaw6)if(raw6_storage[w][i]!=0xcafe || raw6_storage[w][bca::RAW6_STAGE_BYTES/2+16+i]!=0xcafe)status[p*128]=64;
    }
  }
}
extern "C" void bcag_test_trusted_decode(const uint8_t* data,uint64_t bytes,const int64_t* desc,
    const uint8_t* meta,uint64_t meta_bytes,uint32_t buckets,uint32_t panels,uint32_t K,
    uint32_t nr,uint32_t nc,uint32_t matrix,uint16_t* slabs,uint32_t* status,uint32_t* checksum){
  bcag_test_trusted_decode_impl<>(data,bytes,desc,meta,meta_bytes,buckets,panels,K,nr,nc,matrix,slabs,status,checksum);
}
extern "C" void bcag_test_fifo_decode(const uint8_t* data,uint64_t bytes,const int64_t* desc,
    const uint8_t* meta,uint64_t meta_bytes,uint32_t buckets,uint32_t panels,uint32_t K,
    uint32_t nr,uint32_t nc,uint32_t matrix,uint16_t* slabs,uint32_t* status,uint32_t* checksum,uint32_t raw6){
  if(raw6)bcag_test_trusted_decode_impl<true,true>(data,bytes,desc,meta,meta_bytes,buckets,panels,K,nr,nc,matrix,slabs,status,checksum);
  else bcag_test_trusted_decode_impl<true,false>(data,bytes,desc,meta,meta_bytes,buckets,panels,K,nr,nc,matrix,slabs,status,checksum);
}
extern "C" void bcag_test_fifo_fill(const uint8_t* main,uint32_t bytes,uint32_t base,uint16_t* output){
  bca::TrustedPacket t;t.main=main;t.main_bytes=bytes;
  for(uint32_t index=0;index<bca::MAIN_FIFO_BYTES/2;++index)output[index]=bca::fifo_prefetch_word(t,base,index);
}
extern "C" void bcag_test_trusted_slots(const uint8_t* meta,uint64_t meta_bytes,uint32_t buckets,
    uint32_t high,uint16_t* symbols,uint32_t* next_states,uint32_t* status){
  alignas(16) uint8_t data=0;const int64_t d[9]={1,0,0,16,512,0,32*512,32*512,0};
  bca::PacketView v;bca::TrustedPacket t;bca::TrustedCursor c;
  bca::prepare_packet<true>(&data,32*512,d,meta,meta_bytes,buckets,512,1,1,v);
  uint32_t error=bca::init_trusted(v,0,t,c);if(!high || high>65535)error=4;
  bca::TrustedContext context{};if(!error)context=bca::trusted_context(t,0);
  for(uint32_t slot=0;slot<65536;++slot){
    c.state=(high<<16)|slot;status[slot]=error;
    symbols[slot]=error?0:uint16_t(bca::trusted_symbol(t,context,c));next_states[slot]=c.state;
  }
}
extern "C" uint32_t bcag_test_trusted_init(const uint8_t* data,uint64_t bytes,const int64_t* desc,
    const uint8_t* meta,uint64_t meta_bytes,uint32_t buckets,uint32_t K,uint32_t nr,uint32_t nc,uint32_t lane){
  bca::PacketView v;bca::TrustedPacket t;bca::TrustedCursor c;
  bca::prepare_packet<true>(data,bytes,desc,meta,meta_bytes,buckets,K,nr,nc,v);
  return bca::init_trusted(v,lane,t,c);
}
#else
#include <torch/extension.h>
#include <vector>
#include <memory>
#include <atomic>
#include <algorithm>
std::vector<torch::Tensor> bcag_cuda(torch::Tensor data,torch::Tensor desc,torch::Tensor x,
    torch::Tensor metadata,int64_t nr,int64_t nc,int64_t buckets,int64_t label_rows,bool matrix,bool optimized=false,bool aligned_offsets=false,bool trusted=false,bool main_fifo=false,bool stage_raw6=false);
std::vector<torch::Tensor> bcag_decode_cuda(torch::Tensor data,torch::Tensor desc,torch::Tensor metadata,int64_t K,int64_t nr,int64_t nc,int64_t buckets,int64_t label_rows,bool matrix,bool produce=false,bool optimized=false,bool aligned_offsets=false,bool trusted=false,bool main_fifo=false,bool stage_raw6=false);
torch::Tensor bcag_dense_cuda(torch::Tensor w,torch::Tensor x);
namespace py=pybind11;
// No public constructor or tensor getters. Host bytes are fully parsed and
// decoded inside this factory before GPU allocation; metadata and buffers are
// copied directly CPU->CUDA once and are owned privately for plan lifetime.
class ResidentAliasPlan {
  torch::Tensor data_,desc_,metadata_;
  int64_t K_,nr_,nc_,buckets_,label_rows_;
  bool matrix_,aligned_offsets_;
  py::dict provenance_;
  friend std::shared_ptr<ResidentAliasPlan> seal(py::dict,const std::string&);
  friend std::vector<torch::Tensor> gemm(std::shared_ptr<ResidentAliasPlan>,torch::Tensor);
  friend std::vector<torch::Tensor> gemm_opt(std::shared_ptr<ResidentAliasPlan>,torch::Tensor);
  friend std::vector<torch::Tensor> decode(std::shared_ptr<ResidentAliasPlan>);
  friend std::vector<torch::Tensor> produce(std::shared_ptr<ResidentAliasPlan>);
  friend std::vector<torch::Tensor> decode_opt(std::shared_ptr<ResidentAliasPlan>);
  friend std::vector<torch::Tensor> produce_opt(std::shared_ptr<ResidentAliasPlan>);
  friend std::vector<torch::Tensor> gemm_aligned(std::shared_ptr<ResidentAliasPlan>,torch::Tensor);
  friend std::vector<torch::Tensor> decode_aligned(std::shared_ptr<ResidentAliasPlan>);
  friend std::vector<torch::Tensor> produce_aligned(std::shared_ptr<ResidentAliasPlan>);
  friend std::vector<torch::Tensor> gemm_trusted(std::shared_ptr<ResidentAliasPlan>,torch::Tensor);
  friend std::vector<torch::Tensor> decode_trusted(std::shared_ptr<ResidentAliasPlan>);
  friend std::vector<torch::Tensor> produce_trusted(std::shared_ptr<ResidentAliasPlan>);
  friend std::vector<torch::Tensor> gemm_fifo(std::shared_ptr<ResidentAliasPlan>,torch::Tensor);
  friend std::vector<torch::Tensor> decode_fifo(std::shared_ptr<ResidentAliasPlan>);
  friend std::vector<torch::Tensor> produce_fifo(std::shared_ptr<ResidentAliasPlan>);
  friend std::vector<torch::Tensor> gemm_fifo_raw6(std::shared_ptr<ResidentAliasPlan>,torch::Tensor);
  friend std::vector<torch::Tensor> decode_fifo_raw6(std::shared_ptr<ResidentAliasPlan>);
  friend std::vector<torch::Tensor> produce_fifo_raw6(std::shared_ptr<ResidentAliasPlan>);
  friend py::dict plan_provenance(std::shared_ptr<ResidentAliasPlan>);
};
static std::atomic<uint64_t> generation{0};
std::shared_ptr<ResidentAliasPlan> seal(py::dict arrays,const std::string& device){
  auto torchpy=py::module_::import("torch");
  auto dev=torchpy.attr("device")(device);
  TORCH_CHECK(py::str(dev.attr("type")).cast<std::string>()=="cuda" && !dev.attr("index").is_none(),"explicit CUDA device index required");
  auto p=std::make_shared<ResidentAliasPlan>();
  p->K_=arrays["K"].cast<int64_t>();p->nr_=arrays["nr"].cast<int64_t>();
  p->nc_=arrays["nc"].cast<int64_t>();p->buckets_=arrays["buckets"].cast<int64_t>();
  p->label_rows_=arrays["label_rows"].cast<int64_t>();p->matrix_=arrays["matrix_mode"].cast<bool>();
  auto transfer=[&](const char* key){
    // copy() creates writable host storage, never a second GPU allocation.
    return torchpy.attr("from_numpy")(arrays[key].attr("copy")()).attr("to")(dev).cast<torch::Tensor>();
  };
  p->data_=transfer("data");p->desc_=transfer("desc");p->metadata_=transfer("metadata");
  // Private immutable metadata allocation: directory begins at16*buckets and
  // entries advance by4. This cached proof is inaccessible to mutable tensors.
  p->aligned_offsets_=reinterpret_cast<uintptr_t>(p->metadata_.data_ptr<uint8_t>())%alignof(uint32_t)==0;
  torchpy.attr("cuda").attr("synchronize")(dev);
  p->provenance_=py::dict();
  for(const char* key:{"K","nr","nc","buckets","host_wire_bytes","host_inverse_bytes","source_cdf_digest","alias_map_digest","label_rows","matrix_mode","host_validation_decoded_weight_transient_bytes"})p->provenance_[key]=arrays[key];
  p->provenance_["generation"]=++generation;p->provenance_["panels"]=arrays["panels"];
  if(arrays.contains("matrix_source_digest")){
    p->provenance_["matrix_source_digest"]=arrays["matrix_source_digest"];
    p->provenance_["matrix_source_digest_kind"]=py::str("sha256 of ordered panel source SHA256 digests");
  }
  p->provenance_["context_offsets_aligned"]=p->aligned_offsets_;
  p->provenance_["device"]=device;p->provenance_["sealed"]=true;
  p->provenance_["payload_bytes"]=p->data_.numel();
  p->provenance_["descriptor_bytes"]=p->desc_.numel()*8;
  p->provenance_["metadata_bytes"]=p->metadata_.numel();
  p->provenance_["host_staging_bytes"]=p->data_.numel()+p->desc_.numel()*8+p->metadata_.numel();
  p->provenance_["host_staging_peak_bytes"]=p->data_.numel()+p->desc_.numel()*8+p->metadata_.numel()+
      std::max(p->data_.numel(),std::max(p->desc_.numel()*8,p->metadata_.numel()));
  p->provenance_["gpu_upload_tensor_peak_bytes"]=p->data_.numel()+p->desc_.numel()*8+p->metadata_.numel();
  p->provenance_["allocator_driver_peak_measured"]=false;
  p->provenance_["producer_checksum_cost"]=py::str("per-lane FNV1a-like 32-bit sink over shared decoded words; included in producer latency");
  return p;
}
std::shared_ptr<ResidentAliasPlan> prepare(py::bytes blob,const std::string& device,int64_t panels){
  TORCH_CHECK(panels>=1 && panels<=1024,"panels outside [1,1024]");
  auto packet=py::module_::import("georefine_context_runtime.context_alias_packet");
  return seal(packet.attr("_resident_arrays")(blob,panels).cast<py::dict>(),device);
}
std::shared_ptr<ResidentAliasPlan> prepare_matrix(py::list blobs,const std::string& device){
  TORCH_CHECK(py::len(blobs)>=1 && py::len(blobs)<=4096,"matrix panels outside [1,4096]");
  auto packet=py::module_::import("georefine_context_runtime.context_alias_packet");
  return seal(packet.attr("_resident_matrix_arrays")(blobs).cast<py::dict>(),device);
}
static void check_x(torch::Tensor x){
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kBFloat16 && x.dim()==2 && x.is_contiguous(),"contiguous CUDA bf16 x required");
  TORCH_CHECK((x.size(0)==1 || x.size(0)==16 || x.size(0)==64 || x.size(0)==128) && x.size(1)>=128 && x.size(1)<=32768 && x.size(1)%128==0,"bounded M/K required");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr())%16==0,"aligned x required");
}
std::vector<torch::Tensor> gemm(std::shared_ptr<ResidentAliasPlan> p,torch::Tensor x){
  TORCH_CHECK(p,"sealed alias plan required");check_x(x);
  TORCH_CHECK(p->data_.defined() && p->data_.device()==x.device() && p->K_==x.size(1),"wrong, stale or cross-device plan");
  return bcag_cuda(p->data_,p->desc_,x,p->metadata_,p->nr_,p->nc_,p->buckets_,p->label_rows_,p->matrix_);
}
std::vector<torch::Tensor> decode(std::shared_ptr<ResidentAliasPlan> p){
  TORCH_CHECK(p && p->data_.defined(),"sealed alias plan required");
  return bcag_decode_cuda(p->data_,p->desc_,p->metadata_,p->K_,p->nr_,p->nc_,p->buckets_,p->label_rows_,p->matrix_);
}
std::vector<torch::Tensor> produce(std::shared_ptr<ResidentAliasPlan> p){
  TORCH_CHECK(p && p->data_.defined(),"sealed alias plan required");
  return bcag_decode_cuda(p->data_,p->desc_,p->metadata_,p->K_,p->nr_,p->nc_,p->buckets_,p->label_rows_,p->matrix_,true);
}
std::vector<torch::Tensor> gemm_opt(std::shared_ptr<ResidentAliasPlan> p,torch::Tensor x){
  TORCH_CHECK(p,"sealed alias plan required");check_x(x);
  TORCH_CHECK(p->data_.defined() && p->data_.device()==x.device() && p->K_==x.size(1),"wrong, stale or cross-device plan");
  return bcag_cuda(p->data_,p->desc_,x,p->metadata_,p->nr_,p->nc_,p->buckets_,p->label_rows_,p->matrix_,true);
}
std::vector<torch::Tensor> decode_opt(std::shared_ptr<ResidentAliasPlan> p){
  TORCH_CHECK(p && p->data_.defined(),"sealed alias plan required");
  return bcag_decode_cuda(p->data_,p->desc_,p->metadata_,p->K_,p->nr_,p->nc_,p->buckets_,p->label_rows_,p->matrix_,false,true);
}
std::vector<torch::Tensor> produce_opt(std::shared_ptr<ResidentAliasPlan> p){
  TORCH_CHECK(p && p->data_.defined(),"sealed alias plan required");
  return bcag_decode_cuda(p->data_,p->desc_,p->metadata_,p->K_,p->nr_,p->nc_,p->buckets_,p->label_rows_,p->matrix_,true,true);
}
std::vector<torch::Tensor> gemm_aligned(std::shared_ptr<ResidentAliasPlan> p,torch::Tensor x){
  TORCH_CHECK(p && p->aligned_offsets_,"sealed aligned alias plan required");check_x(x);
  TORCH_CHECK(p->data_.defined() && p->data_.device()==x.device() && p->K_==x.size(1),"wrong, stale or cross-device plan");
  return bcag_cuda(p->data_,p->desc_,x,p->metadata_,p->nr_,p->nc_,p->buckets_,p->label_rows_,p->matrix_,true,true);
}
std::vector<torch::Tensor> decode_aligned(std::shared_ptr<ResidentAliasPlan> p){
  TORCH_CHECK(p && p->data_.defined() && p->aligned_offsets_,"sealed aligned alias plan required");
  return bcag_decode_cuda(p->data_,p->desc_,p->metadata_,p->K_,p->nr_,p->nc_,p->buckets_,p->label_rows_,p->matrix_,false,true,true);
}
std::vector<torch::Tensor> produce_aligned(std::shared_ptr<ResidentAliasPlan> p){
  TORCH_CHECK(p && p->data_.defined() && p->aligned_offsets_,"sealed aligned alias plan required");
  return bcag_decode_cuda(p->data_,p->desc_,p->metadata_,p->K_,p->nr_,p->nc_,p->buckets_,p->label_rows_,p->matrix_,true,true,true);
}
// The opaque factory is the sole constructor; it validates the entire wire,
// canonical map and decoded payload before private immutable upload. This API
// does not authenticate a caller's intended tensor identity or post-seal writes.
std::vector<torch::Tensor> gemm_trusted(std::shared_ptr<ResidentAliasPlan> p,torch::Tensor x){
  TORCH_CHECK(p && p->aligned_offsets_,"sealed private alias plan required");check_x(x);
  TORCH_CHECK(p->data_.defined() && p->data_.device()==x.device() && p->K_==x.size(1),"wrong device or shape for alias plan");
  return bcag_cuda(p->data_,p->desc_,x,p->metadata_,p->nr_,p->nc_,p->buckets_,p->label_rows_,p->matrix_,true,true,true);
}
std::vector<torch::Tensor> decode_trusted(std::shared_ptr<ResidentAliasPlan> p){
  TORCH_CHECK(p && p->data_.defined() && p->aligned_offsets_,"sealed private alias plan required");
  return bcag_decode_cuda(p->data_,p->desc_,p->metadata_,p->K_,p->nr_,p->nc_,p->buckets_,p->label_rows_,p->matrix_,false,true,true,true);
}
std::vector<torch::Tensor> produce_trusted(std::shared_ptr<ResidentAliasPlan> p){
  TORCH_CHECK(p && p->data_.defined() && p->aligned_offsets_,"sealed private alias plan required");
  return bcag_decode_cuda(p->data_,p->desc_,p->metadata_,p->K_,p->nr_,p->nc_,p->buckets_,p->label_rows_,p->matrix_,true,true,true,true);
}
std::vector<torch::Tensor> gemm_fifo(std::shared_ptr<ResidentAliasPlan> p,torch::Tensor x){
  TORCH_CHECK(p && p->aligned_offsets_,"sealed private alias plan required");check_x(x);
  TORCH_CHECK(p->data_.defined() && p->data_.device()==x.device() && p->K_==x.size(1),"wrong device or shape for alias plan");
  return bcag_cuda(p->data_,p->desc_,x,p->metadata_,p->nr_,p->nc_,p->buckets_,p->label_rows_,p->matrix_,true,true,true,true,false);
}
std::vector<torch::Tensor> decode_fifo(std::shared_ptr<ResidentAliasPlan> p){
  TORCH_CHECK(p && p->data_.defined() && p->aligned_offsets_,"sealed private alias plan required");
  return bcag_decode_cuda(p->data_,p->desc_,p->metadata_,p->K_,p->nr_,p->nc_,p->buckets_,p->label_rows_,p->matrix_,false,true,true,true,true,false);
}
std::vector<torch::Tensor> produce_fifo(std::shared_ptr<ResidentAliasPlan> p){
  TORCH_CHECK(p && p->data_.defined() && p->aligned_offsets_,"sealed private alias plan required");
  return bcag_decode_cuda(p->data_,p->desc_,p->metadata_,p->K_,p->nr_,p->nc_,p->buckets_,p->label_rows_,p->matrix_,true,true,true,true,true,false);
}
std::vector<torch::Tensor> gemm_fifo_raw6(std::shared_ptr<ResidentAliasPlan> p,torch::Tensor x){
  TORCH_CHECK(p && p->aligned_offsets_,"sealed private alias plan required");check_x(x);
  TORCH_CHECK(p->data_.defined() && p->data_.device()==x.device() && p->K_==x.size(1),"wrong device or shape for alias plan");
  return bcag_cuda(p->data_,p->desc_,x,p->metadata_,p->nr_,p->nc_,p->buckets_,p->label_rows_,p->matrix_,true,true,true,true,true);
}
std::vector<torch::Tensor> decode_fifo_raw6(std::shared_ptr<ResidentAliasPlan> p){
  TORCH_CHECK(p && p->data_.defined() && p->aligned_offsets_,"sealed private alias plan required");
  return bcag_decode_cuda(p->data_,p->desc_,p->metadata_,p->K_,p->nr_,p->nc_,p->buckets_,p->label_rows_,p->matrix_,false,true,true,true,true,true);
}
std::vector<torch::Tensor> produce_fifo_raw6(std::shared_ptr<ResidentAliasPlan> p){
  TORCH_CHECK(p && p->data_.defined() && p->aligned_offsets_,"sealed private alias plan required");
  return bcag_decode_cuda(p->data_,p->desc_,p->metadata_,p->K_,p->nr_,p->nc_,p->buckets_,p->label_rows_,p->matrix_,true,true,true,true,true,true);
}
py::dict plan_provenance(std::shared_ptr<ResidentAliasPlan> p){
  TORCH_CHECK(p,"sealed alias plan required");return py::dict(p->provenance_.attr("copy")());
}
torch::Tensor dense(torch::Tensor w,torch::Tensor x){
  check_x(x);
  TORCH_CHECK(w.is_cuda() && w.device()==x.device() && w.scalar_type()==torch::kBFloat16 && w.dim()==2 && w.size(0)>=64 && w.size(0)<=262144 && w.size(0)%64==0 && w.size(1)==x.size(1) && w.is_contiguous(),"contiguous CUDA bf16 w[P*64,K] required");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(w.data_ptr())%16==0,"aligned weights required");
  return bcag_dense_cuda(w,x);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
  py::class_<ResidentAliasPlan,std::shared_ptr<ResidentAliasPlan>>(m,"ResidentAliasPlan");
  m.def("prepare",&prepare,"Validate and seal private BCA4 resident plan");
  m.def("prepare_matrix",&prepare_matrix,"Validate distinct row panels and seal one matrix plan");
  m.def("produce",&produce,"Diagnostic actual producer plus checksum sink (checksum,status); no decoded HBM");
  m.def("plan_provenance",&plan_provenance);
  m.def("gemm",&gemm,"Experimental BCA4 resident consumer (y,status)");
  m.def("decode",&decode,"Diagnostic full word reconstruction; decoded HBM is separately billed");
  m.def("gemm_opt",&gemm_opt,"Explicit alias register-record ablation (y,status)");
  m.def("decode_opt",&decode_opt,"Explicit alias register-record diagnostic decode");
  m.def("produce_opt",&produce_opt,"Explicit alias register-record producer diagnostic");
  m.def("gemm_aligned",&gemm_aligned,"Explicit aligned context-directory load ablation (y,status)");
  m.def("decode_aligned",&decode_aligned,"Explicit aligned context-directory diagnostic decode");
  m.def("produce_aligned",&produce_aligned,"Explicit aligned context-directory producer diagnostic");
  m.def("gemm_trusted",&gemm_trusted,"Explicit immutable-plan trusted consumer (y,status)");
  m.def("decode_trusted",&decode_trusted,"Explicit immutable-plan trusted diagnostic decode");
  m.def("produce_trusted",&produce_trusted,"Explicit immutable-plan trusted producer diagnostic");
  m.def("gemm_fifo",&gemm_fifo,"Explicit fifo stream staging ablation");
  m.def("decode_fifo",&decode_fifo,"Explicit fifo stream staging ablation");
  m.def("produce_fifo",&produce_fifo,"Explicit fifo stream staging ablation");
  m.def("gemm_fifo_raw6",&gemm_fifo_raw6,"Explicit fifo_raw6 stream staging ablation");
  m.def("decode_fifo_raw6",&decode_fifo_raw6,"Explicit fifo_raw6 stream staging ablation");
  m.def("produce_fifo_raw6",&produce_fifo_raw6,"Explicit fifo_raw6 stream staging ablation");
  m.def("dense",&dense,"Matched BI bf16 S1 tile64 panel control");
}
#endif
