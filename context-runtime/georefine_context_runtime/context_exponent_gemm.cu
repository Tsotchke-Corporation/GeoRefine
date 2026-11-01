#include <torch/extension.h>
#include <vector>
#include <array>
#include <mutex>
#include <unordered_map>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "context_exponent_packet.cuh"
#include "bi_gemm.cuh"

static_assert(bhe::LDS==bi::LDS,"shared weight strides must match");
struct ExponentView {
  const uint8_t *main,*raw,*positions;
  const uint32_t *tables,*offsets;
  const int32_t* rows;
  const int64_t *contexts,*panels,*classes,*counts;
  const uint64_t *checkpoints;
  uint64_t main_bytes,raw_bytes,table_words;
  uint32_t N,K,nc,stage_words;
  uint32_t splits;
};

// Every class-contained CTA has one persistent (row,column-class) cursor per
// active thread. Entropy traversal changes; original Ws coordinates and each
// CH128/kk16 MMA accumulator chain do not.
template<int MT,int MODE>
__global__ void bhe_kernel(ExponentView v,const __nv_bfloat16* x,uint32_t M,
    __nv_bfloat16* y,uint32_t* sink,int32_t* status){
  extern __shared__ __align__(16) unsigned char storage[];
  auto* Ws=reinterpret_cast<__nv_bfloat16*>(storage);
  auto* Xs=Ws+64*bhe::LDS;
  auto* raw_stage=reinterpret_cast<uint8_t*>(Xs)+(MODE==0?MT*16*bhe::LDS*2:0);
  auto* tables=reinterpret_cast<uint32_t*>(raw_stage+64*128);
  __shared__ uint32_t failed;
  const uint32_t tid=threadIdx.x,warp=tid>>5,lane=tid&31;
  const uint32_t tile=MODE==0?blockIdx.y:blockIdx.x,m0=MODE==0?blockIdx.x*64:0;
  const uint32_t blocks=MODE==0?gridDim.x:1;
  const int32_t* rows=v.rows+uint64_t(tile)*64;
  int32_t* final_status=status+(uint64_t(tile)*blocks+(MODE==0?blockIdx.x:0))*256;
  if(tid==0)failed=0;
  __syncthreads();
  const uint32_t cc=tid%v.nc,row=tid/v.nc;
  bhe::Context context{};
  uint32_t table_cursor=0;
  const int64_t rc=v.classes[tile];
  for(uint32_t j=0;j<v.nc;++j){
    const int64_t* d=v.contexts+(uint64_t(rc)*v.nc+j)*4;
    const uint64_t begin=d[0],words=d[1];
    const bool valid=begin<=v.table_words && words<=v.table_words-begin &&
        table_cursor<=v.stage_words && words<=v.stage_words-table_cursor;
    if(!valid){if(tid==0)atomicOr(&failed,2u);}
    else {
      for(uint32_t i=tid;i<words;i+=256)tables[table_cursor+i]=v.tables[begin+i];
      if(cc==j)context={tables+table_cursor,uint32_t(words),int32_t(d[2]),uint32_t(d[3])};
      table_cursor+=uint32_t(words);
    }
  }
  bhe::Cursor cursor{};
  const bool active=row<64 && rows[row]>=0;
  const uint32_t original=active?uint32_t(rows[row]):0;
  const int64_t* panel=v.panels+uint64_t(original/64)*3;
  const bool coded=active && panel[2]==0;
  if(coded){
    const uint32_t* d=v.offsets+uint64_t(original/64)*(64*v.nc+1)+(original%64)*v.nc+cc;
    const uint64_t base=uint64_t(panel[0])*8;
    if(bhe::init(cursor,v.main,v.main_bytes,base+d[0],base+d[1]))atomicOr(&failed,cursor.error);
  }
  __syncthreads();
  float acc[MT][4];
#pragma unroll
  for(int mt=0;mt<MT;++mt)
#pragma unroll
    for(int e=0;e<4;++e)acc[mt][e]=0.0f;
  uint32_t hash=2166136261u;
  for(uint32_t chunk=0;chunk<v.K/128 && !failed;++chunk){
    // Raw8 is staged by coalesced 16-byte row loads. Raw-fallback rows directly
    // fill Ws; zero-padding class tails initializes every subsequently read word.
#pragma unroll
    for(uint32_t it=0;it<4;++it){
      const uint32_t i=tid+256*it,r=i>>4,cv=i&15;
      const int32_t output_row=rows[r];
      const int64_t* pd=v.panels+uint64_t(output_row>=0?output_row/64:0)*3;
      const int64_t mode=pd[2];
      const uint64_t base=uint64_t(pd[1])+uint64_t(output_row>=0?output_row%64:0)*v.K*(mode?2:1);
      if(output_row<0){
        *reinterpret_cast<uint4*>(Ws+r*bhe::LDS+cv*8)=make_uint4(0,0,0,0);
      } else if(mode==1){
        const uint64_t pos=base+uint64_t(chunk)*256+cv*16;
        if(pos>v.raw_bytes || v.raw_bytes-pos<16)atomicOr(&failed,4u);
        else *reinterpret_cast<uint4*>(Ws+r*bhe::LDS+cv*8)=
            *reinterpret_cast<const uint4*>(v.raw+pos);
      } else if(cv<8){
        const uint64_t pos=base+uint64_t(chunk)*128+cv*16;
        if(pos>v.raw_bytes || v.raw_bytes-pos<16)atomicOr(&failed,4u);
        else *reinterpret_cast<uint4*>(raw_stage+r*128+cv*16)=
            *reinterpret_cast<const uint4*>(v.raw+pos);
      }
    }
    __syncthreads();
    if(coded && !failed){
      const int64_t* bounds=v.counts+uint64_t(chunk)*(v.nc+1);
      const uint32_t first=uint32_t(bounds[cc]),count=uint32_t(bounds[cc+1]-bounds[cc]);
      uint32_t output=0;
      while(output<count && !cursor.error){
        bhe::Group group;
        if(bhe::group(context,cursor,count-output,group))break;
#pragma unroll
        for(uint32_t j=0;j<3;++j)if(j<group.count){
          const uint32_t col=v.positions[uint64_t(chunk)*128+first+output+j];
          const uint16_t word=bhe::combine((group.symbols>>(8*j))&255,raw_stage[row*128+col]);
          reinterpret_cast<uint16_t*>(Ws)[row*bhe::LDS+col]=word;
        }
        output+=group.count;
      }
      if(cursor.error)atomicOr(&failed,cursor.error);
    }
    if constexpr(MODE==0){
#pragma unroll
      for(int it=0;it<MT;++it){
        const uint32_t i=tid+256*it,r=i>>4,cv=i&15,m=m0+r;
        uint4 value=make_uint4(0,0,0,0);
        if(m<M)value=__ldg(reinterpret_cast<const uint4*>(x+uint64_t(m)*v.K+chunk*128+cv*8));
        *reinterpret_cast<uint4*>(Xs+r*bhe::LDS+cv*8)=value;
      }
    }
    __syncthreads();
    if(!failed){
      if constexpr(MODE==0){
#pragma unroll
        for(int kk=0;kk<128;kk+=16){
          uint32_t b[2];
          bi::ldsm_x2(b,Ws+(warp*8+(lane&7))*bhe::LDS+kk+((lane>>3)&1)*8);
#pragma unroll
          for(int mt=0;mt<MT;++mt){
            uint32_t a[4];
            bi::ldsm_x4(a,Xs+(mt*16+(lane&7)+((lane>>3)&1)*8)*bhe::LDS+kk+(lane>>4)*8);
            bi::mma16816(acc[mt],a,b);
          }
        }
      } else {
        for(uint32_t i=tid;i<64*128;i+=256){
          const uint32_t r=i>>7,col=i&127;
          const int64_t original=rows[r];
          if(original>=0){
            const uint16_t word=reinterpret_cast<uint16_t*>(Ws)[r*bhe::LDS+col];
            if constexpr(MODE==2)hash=(hash^uint32_t(word))*16777619u;
            else reinterpret_cast<uint16_t*>(y)[uint64_t(original)*v.K+chunk*128+col]=word;
          }
        }
      }
    }
    __syncthreads();
  }
  if(coded && !cursor.error && bhe::finish(cursor))atomicOr(&failed,cursor.error);
  __syncthreads();
  final_status[tid]=cursor.error?cursor.error:failed?32:0;
  if constexpr(MODE==2)sink[uint64_t(tile)*256+tid]=hash;
  if constexpr(MODE==0){
    if(!failed){
      const int g=lane>>2,t4=lane&3;
#pragma unroll
      for(int mt=0;mt<MT;++mt)
#pragma unroll
        for(int h=0;h<2;++h){
          const uint32_t m=m0+mt*16+g+8*h;
          if(m<M){
#pragma unroll
            for(int e=0;e<2;++e){
              const int64_t original=rows[warp*8+2*t4+e];
              if(original>=0)y[uint64_t(m)*v.N+original]=__float2bfloat16_rn(acc[mt][2*h+e]);
            }
          }
        }
    }
  }
}

// Split-K variant. It is separate from bhe_kernel so the default S=1 code path
// retains its launch and arithmetic. Each CTA owns one contiguous chunk range and
// writes FP32 partials; a later reduction performs the prescribed split-order sum.
template<int MT>
__global__ void bhe_split_kernel(ExponentView v,const __nv_bfloat16* x,uint32_t M,
    float* ws,int32_t* status){
  extern __shared__ __align__(16) unsigned char storage[];
  auto* Ws=reinterpret_cast<__nv_bfloat16*>(storage);
  auto* Xs=Ws+64*bhe::LDS;
  auto* raw_stage=reinterpret_cast<uint8_t*>(Xs)+MT*16*bhe::LDS*2;
  auto* tables=reinterpret_cast<uint32_t*>(raw_stage+64*128);
  __shared__ uint32_t failed;
  const uint32_t tid=threadIdx.x,warp=tid>>5,lane=tid&31;
  const uint32_t tile=blockIdx.y,split=blockIdx.z,m0=blockIdx.x*(MT==8?128:64),blocks=gridDim.x;
  const int32_t* rows=v.rows+uint64_t(tile)*64;
  int32_t* final_status=status+((uint64_t(split)*gridDim.y*blocks+uint64_t(tile)*blocks+blockIdx.x)*256);
  if(tid==0)failed=0;
  __syncthreads();
  const uint32_t cc=tid%v.nc,row=tid/v.nc;
  bhe::Context context{};uint32_t table_cursor=0;
  const int64_t rc=v.classes[tile];
  for(uint32_t j=0;j<v.nc;++j){
    const int64_t* d=v.contexts+(uint64_t(rc)*v.nc+j)*4;
    const uint64_t begin=d[0],words=d[1];
    const bool valid=begin<=v.table_words && words<=v.table_words-begin &&
        table_cursor<=v.stage_words && words<=v.stage_words-table_cursor;
    if(!valid){if(tid==0)atomicOr(&failed,2u);}
    else {for(uint32_t i=tid;i<words;i+=256)tables[table_cursor+i]=v.tables[begin+i];
      if(cc==j)context={tables+table_cursor,uint32_t(words),int32_t(d[2]),uint32_t(d[3])};
      table_cursor+=uint32_t(words);}
  }
  bhe::Cursor cursor{};const bool active=row<64&&rows[row]>=0;
  const uint32_t original=active?uint32_t(rows[row]):0;
  const int64_t* panel=v.panels+uint64_t(original/64)*3;const bool coded=active&&panel[2]==0;
  const uint32_t nch=v.K/128,c0=(uint32_t)((uint64_t(split)*nch)/v.splits),c1=(uint32_t)(((uint64_t(split)+1)*nch)/v.splits);
  if(coded){
    const uint32_t* d=v.offsets+uint64_t(original/64)*(64*v.nc+1)+(original%64)*v.nc+cc;
    const uint64_t base=uint64_t(panel[0])*8;uint64_t start=base+d[0],end=base+d[1];
    if(split){start=v.checkpoints[(uint64_t(original)*v.nc+cc)*(v.splits-1)+(split-1)];}
    if(split+1<v.splits)end=v.checkpoints[(uint64_t(original)*v.nc+cc)*(v.splits-1)+split];
    if(bhe::init(cursor,v.main,v.main_bytes,start,end))atomicOr(&failed,cursor.error);
  }
  __syncthreads();
  float acc[MT][4];
#pragma unroll
  for(int mt=0;mt<MT;++mt)for(int e=0;e<4;++e)acc[mt][e]=0.0f;
  for(uint32_t chunk=c0;chunk<c1&&!failed;++chunk){
#pragma unroll
    for(uint32_t it=0;it<4;++it){const uint32_t i=tid+256*it,r=i>>4,cv=i&15;const int32_t output_row=rows[r];
      const int64_t* pd=v.panels+uint64_t(output_row>=0?output_row/64:0)*3;const int64_t mode=pd[2];
      const uint64_t base=uint64_t(pd[1])+uint64_t(output_row>=0?output_row%64:0)*v.K*(mode?2:1);
      if(output_row<0)*reinterpret_cast<uint4*>(Ws+r*bhe::LDS+cv*8)=make_uint4(0,0,0,0);
      else if(mode==1){const uint64_t pos=base+uint64_t(chunk)*256+cv*16;if(pos>v.raw_bytes||v.raw_bytes-pos<16)atomicOr(&failed,4u);else *reinterpret_cast<uint4*>(Ws+r*bhe::LDS+cv*8)=*reinterpret_cast<const uint4*>(v.raw+pos);}
      else if(cv<8){const uint64_t pos=base+uint64_t(chunk)*128+cv*16;if(pos>v.raw_bytes||v.raw_bytes-pos<16)atomicOr(&failed,4u);else *reinterpret_cast<uint4*>(raw_stage+r*128+cv*16)=*reinterpret_cast<const uint4*>(v.raw+pos);}
    }
    __syncthreads();
    if(coded&&!failed){const int64_t* bounds=v.counts+uint64_t(chunk)*(v.nc+1);const uint32_t first=uint32_t(bounds[cc]),count=uint32_t(bounds[cc+1]-bounds[cc]);uint32_t output=0;
      while(output<count&&!cursor.error){bhe::Group group;if(bhe::group(context,cursor,count-output,group))break;for(uint32_t j=0;j<3;++j)if(j<group.count){const uint32_t col=v.positions[uint64_t(chunk)*128+first+output+j];reinterpret_cast<uint16_t*>(Ws)[row*bhe::LDS+col]=bhe::combine((group.symbols>>(8*j))&255,raw_stage[row*128+col]);}output+=group.count;}
      if(cursor.error)atomicOr(&failed,cursor.error);
    }
#pragma unroll
    for(int it=0;it<MT;++it){const uint32_t i=tid+256*it,r=i>>4,cv=i&15,m=m0+r;uint4 value=make_uint4(0,0,0,0);if(m<M)value=__ldg(reinterpret_cast<const uint4*>(x+uint64_t(m)*v.K+uint64_t(chunk)*128+cv*8));*reinterpret_cast<uint4*>(Xs+r*bhe::LDS+cv*8)=value;}
    __syncthreads();
#pragma unroll
    for(int kk=0;kk<128;kk+=16){uint32_t b[2];bi::ldsm_x2(b,Ws+(warp*8+(lane&7))*bhe::LDS+kk+((lane>>3)&1)*8);for(int mt=0;mt<MT;++mt){uint32_t a[4];bi::ldsm_x4(a,Xs+(mt*16+(lane&7)+((lane>>3)&1)*8)*bhe::LDS+kk+(lane>>4)*8);bi::mma16816(acc[mt],a,b);}}
    __syncthreads();
  }
  if(coded&&!cursor.error&&bhe::finish(cursor))atomicOr(&failed,cursor.error);
  __syncthreads();
  final_status[tid]=cursor.error?cursor.error:failed?32:0;
  if(!failed){const int g=lane>>2,t4=lane&3;for(int mt=0;mt<MT;++mt)for(int h=0;h<2;++h){const uint32_t m=m0+mt*16+g+8*h;if(m<M)for(int e=0;e<2;++e){const int64_t original_row=rows[warp*8+2*t4+e];if(original_row>=0)ws[((uint64_t(split)*M+m)*v.N)+uint64_t(original_row)]=acc[mt][2*h+e];}}}
}

__global__ void bhe_split_reduce(const float* ws,uint32_t S,uint32_t M,uint32_t N,__nv_bfloat16* y,int64_t ldy){
  const uint32_t n=blockIdx.x*256+threadIdx.x,m=blockIdx.y;if(n>=N||m>=M)return;float v=ws[(uint64_t(m)*N)+n];for(uint32_t s=1;s<S;++s)v=__fadd_rn(v,ws[(uint64_t(s)*M+m)*N+n]);y[uint64_t(m)*ldy+n]=__float2bfloat16_rn(v);
}
__global__ void bhe_status_guard(const int32_t* status,uint64_t count){
  for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=uint64_t(gridDim.x)*blockDim.x)
    if(status[i])asm volatile("trap;");
}

// Diagnostic copy: keep baseline bhe_kernel unchanged while tracing fused failures.
template<int MT,int MODE>
__global__ void bhe_trace_kernel(ExponentView v,const __nv_bfloat16* x,uint32_t M,
    __nv_bfloat16* y,uint32_t* sink,int32_t* status,uint64_t* trace){
  static_assert(MODE==0,"trace is a diagnostic GEMM specialization");
  extern __shared__ __align__(16) unsigned char storage[];
  auto* Ws=reinterpret_cast<__nv_bfloat16*>(storage);
  auto* Xs=Ws+64*bhe::LDS;
  auto* raw_stage=reinterpret_cast<uint8_t*>(Xs)+(MODE==0?MT*16*bhe::LDS*2:0);
  auto* tables=reinterpret_cast<uint32_t*>(raw_stage+64*128);
  __shared__ uint32_t failed;
  const uint32_t tid=threadIdx.x,warp=tid>>5,lane=tid&31;
  const uint32_t tile=MODE==0?blockIdx.y:blockIdx.x,m0=MODE==0?blockIdx.x*64:0;
  const uint32_t blocks=MODE==0?gridDim.x:1;
  const int32_t* rows=v.rows+uint64_t(tile)*64;
  int32_t* final_status=status+(uint64_t(tile)*blocks+(MODE==0?blockIdx.x:0))*256;
  if(tid==0)failed=0;
  __syncthreads();
  const uint32_t cc=tid%v.nc,row=tid/v.nc;
  bhe::Context context{};
  uint32_t table_cursor=0;
  const int64_t rc=v.classes[tile];
  for(uint32_t j=0;j<v.nc;++j){
    const int64_t* d=v.contexts+(uint64_t(rc)*v.nc+j)*4;
    const uint64_t begin=d[0],words=d[1];
    const bool valid=begin<=v.table_words && words<=v.table_words-begin &&
        table_cursor<=v.stage_words && words<=v.stage_words-table_cursor;
    if(!valid){if(tid==0)atomicOr(&failed,2u);}
    else {
      for(uint32_t i=tid;i<words;i+=256)tables[table_cursor+i]=v.tables[begin+i];
      if(cc==j)context={tables+table_cursor,uint32_t(words),int32_t(d[2]),uint32_t(d[3])};
      table_cursor+=uint32_t(words);
    }
  }
  bhe::Cursor cursor{};
  const bool active=row<64 && rows[row]>=0;
  const uint32_t original=active?uint32_t(rows[row]):0;
  const int64_t* panel=v.panels+uint64_t(original/64)*3;
  const bool coded=active && panel[2]==0;
  if(coded){
    const uint32_t* d=v.offsets+uint64_t(original/64)*(64*v.nc+1)+(original%64)*v.nc+cc;
    const uint64_t base=uint64_t(panel[0])*8;
    if(bhe::init(cursor,v.main,v.main_bytes,base+d[0],base+d[1]))atomicOr(&failed,cursor.error);
  }
  __syncthreads();
  float acc[MT][4];
#pragma unroll
  for(int mt=0;mt<MT;++mt)
#pragma unroll
    for(int e=0;e<4;++e)acc[mt][e]=0.0f;
  // The first 15 words are the stable final snapshot.  Each following
  // TRACE_CHUNK_WORDS block is one per-chunk cursor diagnostic:
  // pre(consumed,valid,next_word), post(consumed,valid,next_word),
  // last/failed group table entry, requested remaining, group count/error,
  // pre/post buffer, last-group consumed/valid/next_word/buffer and tail length.
  constexpr uint32_t TRACE_CHUNK_WORDS=17;
  const uint32_t trace_chunks=v.K/128;
  uint64_t* lane_trace=trace+((uint64_t(tile)*blocks+blockIdx.x)*256+tid)*
      (15+uint64_t(trace_chunks)*TRACE_CHUNK_WORDS);
  uint32_t hash=2166136261u,completed_chunks=0;
  for(uint32_t chunk=0;chunk<v.K/128 && !failed;++chunk){
    uint64_t* chunk_trace=lane_trace+15+uint64_t(chunk)*TRACE_CHUNK_WORDS;
    chunk_trace[0]=cursor.consumed;chunk_trace[1]=cursor.valid;chunk_trace[2]=cursor.next_word;
    chunk_trace[3]=cursor.consumed;chunk_trace[4]=cursor.valid;chunk_trace[5]=cursor.next_word;
    chunk_trace[6]=UINT64_MAX;chunk_trace[7]=UINT64_MAX;chunk_trace[8]=UINT64_MAX;chunk_trace[9]=UINT64_MAX;
    chunk_trace[10]=cursor.buffer;chunk_trace[11]=cursor.buffer;
    for(uint32_t f=12;f<17;++f)chunk_trace[f]=UINT64_MAX;
    // Raw8 is staged by coalesced 16-byte row loads. Raw-fallback rows directly
    // fill Ws; zero-padding class tails initializes every subsequently read word.
#pragma unroll
    for(uint32_t it=0;it<4;++it){
      const uint32_t i=tid+256*it,r=i>>4,cv=i&15;
      const int32_t output_row=rows[r];
      const int64_t* pd=v.panels+uint64_t(output_row>=0?output_row/64:0)*3;
      const int64_t mode=pd[2];
      const uint64_t base=uint64_t(pd[1])+uint64_t(output_row>=0?output_row%64:0)*v.K*(mode?2:1);
      if(output_row<0){
        *reinterpret_cast<uint4*>(Ws+r*bhe::LDS+cv*8)=make_uint4(0,0,0,0);
      } else if(mode==1){
        const uint64_t pos=base+uint64_t(chunk)*256+cv*16;
        if(pos>v.raw_bytes || v.raw_bytes-pos<16)atomicOr(&failed,4u);
        else *reinterpret_cast<uint4*>(Ws+r*bhe::LDS+cv*8)=
            *reinterpret_cast<const uint4*>(v.raw+pos);
      } else if(cv<8){
        const uint64_t pos=base+uint64_t(chunk)*128+cv*16;
        if(pos>v.raw_bytes || v.raw_bytes-pos<16)atomicOr(&failed,4u);
        else *reinterpret_cast<uint4*>(raw_stage+r*128+cv*16)=
            *reinterpret_cast<const uint4*>(v.raw+pos);
      }
    }
    __syncthreads();
    if(coded && !failed){
      const int64_t* bounds=v.counts+uint64_t(chunk)*(v.nc+1);
      const uint32_t first=uint32_t(bounds[cc]),count=uint32_t(bounds[cc+1]-bounds[cc]);
      uint32_t output=0;
      while(output<count && !cursor.error){
        const uint32_t requested=count-output;
        chunk_trace[12]=cursor.consumed;chunk_trace[13]=cursor.valid;
        chunk_trace[14]=cursor.next_word;chunk_trace[15]=cursor.buffer;
        bhe::Cursor probe=cursor;uint32_t prefix=0;
        if(context.singleton<0 && context.table && context.words>=256 &&
            !bhe::peek(probe,8,prefix)){
          const uint32_t selected=context.table[prefix];chunk_trace[6]=selected;
          if((selected>>30)==0 && context.words>=320){
            const uint32_t symbol=selected&255u;
            chunk_trace[16]=(context.table[context.words-64+symbol/4]>>(8*(symbol&3)))&255u;
          }
        }
        chunk_trace[7]=requested;
        bhe::Group group;
        if(bhe::group(context,cursor,count-output,group))break;
        chunk_trace[8]=group.count;chunk_trace[9]=0;
#pragma unroll
        for(uint32_t j=0;j<3;++j)if(j<group.count){
          const uint32_t col=v.positions[uint64_t(chunk)*128+first+output+j];
          const uint16_t word=bhe::combine((group.symbols>>(8*j))&255,raw_stage[row*128+col]);
          reinterpret_cast<uint16_t*>(Ws)[row*bhe::LDS+col]=word;
        }
        output+=group.count;
      }
      if(cursor.error){chunk_trace[8]=UINT64_MAX;chunk_trace[9]=cursor.error;atomicOr(&failed,cursor.error);}
    }
    if constexpr(MODE==0){
#pragma unroll
      for(int it=0;it<MT;++it){
        const uint32_t i=tid+256*it,r=i>>4,cv=i&15,m=m0+r;
        uint4 value=make_uint4(0,0,0,0);
        if(m<M)value=__ldg(reinterpret_cast<const uint4*>(x+uint64_t(m)*v.K+chunk*128+cv*8));
        *reinterpret_cast<uint4*>(Xs+r*bhe::LDS+cv*8)=value;
      }
    }
    __syncthreads();
    if(!failed){
      if constexpr(MODE==0){
#pragma unroll
        for(int kk=0;kk<128;kk+=16){
          uint32_t b[2];
          bi::ldsm_x2(b,Ws+(warp*8+(lane&7))*bhe::LDS+kk+((lane>>3)&1)*8);
#pragma unroll
          for(int mt=0;mt<MT;++mt){
            uint32_t a[4];
            bi::ldsm_x4(a,Xs+(mt*16+(lane&7)+((lane>>3)&1)*8)*bhe::LDS+kk+(lane>>4)*8);
            bi::mma16816(acc[mt],a,b);
          }
        }
      } else {
        for(uint32_t i=tid;i<64*128;i+=256){
          const uint32_t r=i>>7,col=i&127;
          const int64_t original=rows[r];
          if(original>=0){
            const uint16_t word=reinterpret_cast<uint16_t*>(Ws)[r*bhe::LDS+col];
            if constexpr(MODE==2)hash=(hash^uint32_t(word))*16777619u;
            else reinterpret_cast<uint16_t*>(y)[uint64_t(original)*v.K+chunk*128+col]=word;
          }
        }
      }
    }
    __syncthreads();
    chunk_trace[3]=cursor.consumed;chunk_trace[4]=cursor.valid;chunk_trace[5]=cursor.next_word;
    chunk_trace[11]=cursor.buffer;
    ++completed_chunks;
  }
  // Diagnostic-only snapshot. All lanes publish before any finish() call can
  // modify shared failure state, distinguishing primary errors from aborts.
  uint64_t* entry=lane_trace;
  entry[0]=cursor.error;entry[1]=failed;entry[2]=cursor.start;entry[3]=cursor.end;
  entry[4]=cursor.consumed;entry[5]=cursor.valid;
  entry[6]=active?uint64_t(original):UINT64_MAX;entry[7]=uint64_t(rc)*v.nc+cc;
  entry[8]=completed_chunks;entry[9]=coded?1:0;
  entry[10]=v.N;entry[11]=v.K;entry[12]=M;entry[13]=v.nc;entry[14]=v.stage_words;
  __syncthreads();
  if(coded && !cursor.error && bhe::finish(cursor))atomicOr(&failed,cursor.error);
  __syncthreads();
  final_status[tid]=cursor.error?cursor.error:failed?32:0;
  if constexpr(MODE==2)sink[uint64_t(tile)*256+tid]=hash;
  if constexpr(MODE==0){
    if(!failed){
      const int g=lane>>2,t4=lane&3;
#pragma unroll
      for(int mt=0;mt<MT;++mt)
#pragma unroll
        for(int h=0;h<2;++h){
          const uint32_t m=m0+mt*16+g+8*h;
          if(m<M){
#pragma unroll
            for(int e=0;e<2;++e){
              const int64_t original=rows[warp*8+2*t4+e];
              if(original>=0)y[uint64_t(m)*v.N+original]=__float2bfloat16_rn(acc[mt][2*h+e]);
            }
          }
        }
    }
  }
}

static ExponentView view(const std::vector<torch::Tensor>& a,int64_t N,int64_t K,int64_t nc,int64_t stage,int64_t splits=1){
  return {a[0].data_ptr<uint8_t>(),a[1].data_ptr<uint8_t>(),a[7].data_ptr<uint8_t>(),
      reinterpret_cast<const uint32_t*>(a[2].data_ptr<int32_t>()),reinterpret_cast<const uint32_t*>(a[4].data_ptr<int32_t>()),
      a[5].data_ptr<int32_t>(),a[3].data_ptr<int64_t>(),a[9].data_ptr<int64_t>(),a[6].data_ptr<int64_t>(),a[8].data_ptr<int64_t>(),
      splits>1?reinterpret_cast<const uint64_t*>(a[10].data_ptr<int64_t>()):nullptr,
      uint64_t(a[0].numel()),uint64_t(a[1].numel()),uint64_t(a[2].numel()),uint32_t(N),uint32_t(K),uint32_t(nc),uint32_t(stage),uint32_t(splits)};
}
static void record(const std::vector<torch::Tensor>& a,c10::cuda::CUDAStream stream){
  for(const auto& t:a)c10::cuda::CUDACachingAllocator::recordStream(t.storage().data_ptr(),stream);
}
namespace {
// CUDA function attributes are device-global, rather than properties of a plan
// or a graph node. Preparation may increase them, but a later small plan must
// never lower the limit required by an already captured larger plan.
std::mutex shared_config_mutex;
std::unordered_map<int,std::array<size_t,4>> shared_config;
std::unordered_map<int,std::array<size_t,2>> trace_shared_config;
std::unordered_map<int,std::array<size_t,3>> split_shared_config;
template<int MT,int MODE> constexpr int config_slot(){
  static_assert((MODE==0 && (MT==1 || MT==4)) || (MT==1 && (MODE==1 || MODE==2)),
                "unsupported exponent kernel specialization");
  return MODE==0?(MT==1?0:1):(MODE==1?2:3);
}
template<int MT,int MODE> size_t required_shared(int64_t stage){
  return size_t(64+(MODE==0?MT*16:0))*bhe::LDS*2+8192+size_t(stage)*4;
}
template<int MT,int MODE>
void configure_shared(std::array<size_t,4>& configured,size_t requested,int optin_limit){
  const auto kernel=reinterpret_cast<const void*>(&bhe_kernel<MT,MODE>);
  cudaFuncAttributes attributes{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes,kernel));
  TORCH_CHECK(attributes.sharedSizeBytes<=size_t(optin_limit) &&
      requested<=size_t(optin_limit)-attributes.sharedSizeBytes,
      "exponent shared-memory request exceeds device opt-in limit: dynamic=",requested,
      ", static=",attributes.sharedSizeBytes,", device limit=",optin_limit);
  auto& prior=configured[config_slot<MT,MODE>()];
  if(requested>prior){
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(requested)));
    prior=requested; // only publish successful device configuration
  }
}
template<int MT,int MODE> void require_shared(int device,size_t requested){
  std::lock_guard<std::mutex> lock(shared_config_mutex);
  const auto found=shared_config.find(device);
  const auto slot=config_slot<MT,MODE>();
  TORCH_CHECK(found!=shared_config.end() && found->second[slot]>=requested,
      "exponent kernel shared-memory configuration is missing; prepare a sealed plan first");
}
template<int MT> void configure_trace_shared(std::array<size_t,2>& configured,size_t requested,int optin_limit){
  const auto kernel=reinterpret_cast<const void*>(&bhe_trace_kernel<MT,0>);
  cudaFuncAttributes attributes{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes,kernel));
  TORCH_CHECK(attributes.sharedSizeBytes<=size_t(optin_limit) && requested<=size_t(optin_limit)-attributes.sharedSizeBytes,
      "diagnostic exponent shared-memory request exceeds device opt-in limit");
  auto& prior=configured[MT==1?0:1];
  if(requested>prior){
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(requested)));
    prior=requested;
  }
}
template<int MT> void require_trace_shared(int device,size_t requested){
  std::lock_guard<std::mutex> lock(shared_config_mutex);
  const auto found=trace_shared_config.find(device);
  TORCH_CHECK(found!=trace_shared_config.end() && found->second[MT==1?0:1]>=requested,
      "diagnostic exponent kernel requires a prepared plan");
}
} // namespace

int bhe_prepare_cuda(torch::Tensor reference,int64_t stage){
  TORCH_CHECK(reference.is_cuda() && stage>=0 && stage<=16384,"invalid exponent preparation geometry");
  const c10::cuda::CUDAGuard guard(reference.device());
  const int device=reference.device().index();
  const auto stream=at::cuda::getCurrentCUDAStream();
  cudaStreamCaptureStatus capture{};
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream,&capture));
  TORCH_CHECK(capture==cudaStreamCaptureStatusNone,"exponent plans must be prepared outside CUDA graph capture");
  int optin_limit=0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin_limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));
  TORCH_CHECK(optin_limit>0,"device does not report an opt-in shared-memory limit");
  std::lock_guard<std::mutex> lock(shared_config_mutex);
  auto& configured=shared_config[device];
  configure_shared<1,0>(configured,required_shared<1,0>(stage),optin_limit);
  configure_shared<4,0>(configured,required_shared<4,0>(stage),optin_limit);
  configure_shared<1,1>(configured,required_shared<1,1>(stage),optin_limit);
  configure_shared<1,2>(configured,required_shared<1,2>(stage),optin_limit);
  const size_t split1=required_shared<1,0>(stage),split4=required_shared<4,0>(stage);
  TORCH_CHECK(split1<=size_t(optin_limit) && split4<=size_t(optin_limit),"split exponent shared-memory request exceeds device limit");
  cudaFuncAttributes sa1{},sa4{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&sa1,reinterpret_cast<const void*>(&bhe_split_kernel<1>)));
  C10_CUDA_CHECK(cudaFuncGetAttributes(&sa4,reinterpret_cast<const void*>(&bhe_split_kernel<4>)));
  TORCH_CHECK(sa1.sharedSizeBytes<=size_t(optin_limit)-split1 && sa4.sharedSizeBytes<=size_t(optin_limit)-split4,"split exponent static+dynamic shared memory exceeds device limit");
  auto& split_configured=split_shared_config[device];
  if(split1>split_configured[0]){
    C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(&bhe_split_kernel<1>),cudaFuncAttributeMaxDynamicSharedMemorySize,int(split1)));
    split_configured[0]=split1;
  }
  if(split4>split_configured[1]){
    C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(&bhe_split_kernel<4>),cudaFuncAttributeMaxDynamicSharedMemorySize,int(split4)));
    split_configured[1]=split4;
  }
  const size_t split8=required_shared<8,0>(stage);
  cudaFuncAttributes sa8{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&sa8,reinterpret_cast<const void*>(&bhe_split_kernel<8>)));
  if(split8+sa8.sharedSizeBytes<=size_t(optin_limit) && split8>split_configured[2]){
    C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(&bhe_split_kernel<8>),cudaFuncAttributeMaxDynamicSharedMemorySize,int(split8)));
    split_configured[2]=split8;
  }
  auto& trace_configured=trace_shared_config[device];
  configure_trace_shared<1>(trace_configured,required_shared<1,0>(stage),optin_limit);
  configure_trace_shared<4>(trace_configured,required_shared<4,0>(stage),optin_limit);
  return split8+sa8.sharedSizeBytes<=size_t(optin_limit)?128:64;
}
template<int MT,int MODE>
static void launch(ExponentView v,int device,dim3 grid,size_t shared,cudaStream_t stream,const __nv_bfloat16* x,
    uint32_t M,__nv_bfloat16* y,uint32_t* sink,int32_t* status){
  require_shared<MT,MODE>(device,shared); // host-only check; no CUDA setters during capture/dispatch
  bhe_kernel<MT,MODE><<<grid,256,shared,stream>>>(v,x,M,y,sink,status);
}
std::vector<torch::Tensor> bhe_gemm_cuda(const std::vector<torch::Tensor>& a,torch::Tensor x,
    int64_t N,int64_t K,int64_t nc,int64_t stage){
  const c10::cuda::CUDAGuard guard(a[0].device());const int64_t tiles=a[6].numel(),M=x.size(0),blocks=(M+63)/64;
  auto y=torch::empty({M,N},x.options());auto status=torch::empty({tiles,blocks,256},x.options().dtype(torch::kInt32));
  const auto stream=at::cuda::getCurrentCUDAStream();record(a,stream);
  c10::cuda::CUDACachingAllocator::recordStream(x.storage().data_ptr(),stream);
  const auto v=view(a,N,K,nc,stage);const dim3 grid(blocks,tiles,1);
  const size_t shared=(64+(M>16?64:16))*bhe::LDS*2+8192+stage*4;
  if(M>16)launch<4,0>(v,a[0].device().index(),grid,shared,stream,reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),M,
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),nullptr,status.data_ptr<int32_t>());
  else launch<1,0>(v,a[0].device().index(),grid,shared,stream,reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),M,
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),nullptr,status.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {y,status};
}
std::vector<torch::Tensor> bhe_gemm_out_cuda(const std::vector<torch::Tensor>& a,torch::Tensor x,torch::Tensor y,
    torch::Tensor ws,torch::Tensor status,int64_t N,int64_t K,int64_t nc,int64_t stage,int64_t splits,int64_t tile){
  const c10::cuda::CUDAGuard guard(a[0].device());const int64_t tiles=a[6].numel(),M=x.size(0),blocks=(M+tile-1)/tile;
  const auto stream=at::cuda::getCurrentCUDAStream();record(a,stream);
  c10::cuda::CUDACachingAllocator::recordStream(x.storage().data_ptr(),stream);
  c10::cuda::CUDACachingAllocator::recordStream(y.storage().data_ptr(),stream);
  c10::cuda::CUDACachingAllocator::recordStream(ws.storage().data_ptr(),stream);
  c10::cuda::CUDACachingAllocator::recordStream(status.storage().data_ptr(),stream);
  const auto v=view(a,N,K,nc,stage,splits);const dim3 grid(blocks,tiles,splits);
  const size_t shared=(64+(tile==128?128:(M>16?64:16)))*bhe::LDS*2+8192+stage*4;
  auto* xp=reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  auto* wp=ws.data_ptr<float>();auto* sp=status.data_ptr<int32_t>();
  {
    std::lock_guard<std::mutex> lock(shared_config_mutex);
    const auto found=split_shared_config.find(a[0].device().index());
    const int slot=tile==128?2:(M>16?1:0);
    TORCH_CHECK(found!=split_shared_config.end() && found->second[slot]>=shared,
        "requested BHE1 tile exceeds prepared shared-memory capacity");
  }
  if(tile==128)bhe_split_kernel<8><<<grid,256,shared,stream>>>(v,xp,M,wp,sp);
  else if(M>16)bhe_split_kernel<4><<<grid,256,shared,stream>>>(v,xp,M,wp,sp);
  else bhe_split_kernel<1><<<grid,256,shared,stream>>>(v,xp,M,wp,sp);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  bhe_status_guard<<<dim3(128),256,0,stream>>>(sp,uint64_t(splits)*uint64_t(tiles)*uint64_t(blocks)*256u);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  bhe_split_reduce<<<dim3((N+255)/256,M),256,0,stream>>>(wp,uint32_t(splits),uint32_t(M),uint32_t(N),reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),y.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y,ws,status};
}
std::vector<torch::Tensor> bhe_gemm_trace_cuda(const std::vector<torch::Tensor>& a,torch::Tensor x,
    int64_t N,int64_t K,int64_t nc,int64_t stage){
  const c10::cuda::CUDAGuard guard(a[0].device());const int64_t tiles=a[6].numel(),M=x.size(0),blocks=(M+63)/64;
  auto y=torch::empty({M,N},x.options());auto status=torch::empty({tiles,blocks,256},x.options().dtype(torch::kInt32));
  constexpr int64_t trace_final_words=15,trace_chunk_words=17;
  const int64_t chunks=K/128;
  const int64_t trace_words=trace_final_words+chunks*trace_chunk_words;
  const uint64_t trace_bytes=uint64_t(tiles)*uint64_t(blocks)*256u*uint64_t(trace_words)*sizeof(uint64_t);
  TORCH_CHECK(chunks>=1 && trace_bytes<=512ull*1024ull*1024ull,
      "exponent diagnostic trace exceeds bounded allocation (bytes=",trace_bytes,
      ", limit=536870912)");
  auto trace=torch::zeros({tiles,blocks,256,trace_words},x.options().dtype(torch::kInt64));
  const auto stream=at::cuda::getCurrentCUDAStream();record(a,stream);
  c10::cuda::CUDACachingAllocator::recordStream(x.storage().data_ptr(),stream);
  const auto v=view(a,N,K,nc,stage);const dim3 grid(blocks,tiles,1);
  const size_t shared=(64+(M>16?64:16))*bhe::LDS*2+8192+stage*4;
  auto* xp=reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  auto* yp=reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>());
  auto* tp=reinterpret_cast<uint64_t*>(trace.data_ptr<int64_t>());
  if(M>16){
    require_trace_shared<4>(a[0].device().index(),shared);
    bhe_trace_kernel<4,0><<<grid,256,shared,stream>>>(v,xp,M,yp,nullptr,status.data_ptr<int32_t>(),tp);
  } else {
    require_trace_shared<1>(a[0].device().index(),shared);
    bhe_trace_kernel<1,0><<<grid,256,shared,stream>>>(v,xp,M,yp,nullptr,status.data_ptr<int32_t>(),tp);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {y,status,trace};
}
std::vector<torch::Tensor> bhe_decode_cuda(const std::vector<torch::Tensor>& a,int64_t N,int64_t K,
    int64_t nc,int64_t stage,bool produce){
  const c10::cuda::CUDAGuard guard(a[0].device());const int64_t tiles=a[6].numel();
  auto y=produce?torch::empty({tiles,256},a[0].options().dtype(torch::kInt32)):
      torch::empty({N,K},a[0].options().dtype(torch::kBFloat16));
  auto status=torch::empty({tiles,256},a[0].options().dtype(torch::kInt32));
  const auto stream=at::cuda::getCurrentCUDAStream();record(a,stream);
  const auto v=view(a,N,K,nc,stage);const size_t shared=64*bhe::LDS*2+8192+stage*4;
  if(produce)launch<1,2>(v,a[0].device().index(),dim3(tiles),shared,stream,nullptr,0,nullptr,
      reinterpret_cast<uint32_t*>(y.data_ptr<int32_t>()),status.data_ptr<int32_t>());
  else launch<1,1>(v,a[0].device().index(),dim3(tiles),shared,stream,nullptr,0,
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),nullptr,status.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {y,status};
}
torch::Tensor bhe_dense_cuda(torch::Tensor w,torch::Tensor x,int64_t splits,int64_t tile){
  const c10::cuda::CUDAGuard guard(x.device());const int M=x.size(0),N=w.size(0),K=w.size(1);
  TORCH_CHECK(splits>=1&&splits<=8&&splits<=K/128,"dense splits must be 1..8 and no greater than K/128");
  auto y=torch::empty({M,N},x.options());bi::WDesc d{};d.N=N;d.K=K;
  d.w=reinterpret_cast<const __nv_bfloat16*>(w.data_ptr<at::BFloat16>());
  const auto stream=at::cuda::getCurrentCUDAStream();
  c10::cuda::CUDACachingAllocator::recordStream(x.storage().data_ptr(),stream);
  c10::cuda::CUDACachingAllocator::recordStream(w.storage().data_ptr(),stream);
  const size_t shared=(64+(tile==128?128:(M>16?64:16)))*bi::LDS*2;
  const dim3 grid((N+63)/64,splits,(M+tile-1)/tile);auto* xp=reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  if(tile==128){
    cudaStreamCaptureStatus capture{};C10_CUDA_CHECK(cudaStreamIsCapturing(stream,&capture));
    TORCH_CHECK(capture==cudaStreamCaptureStatusNone, "dense tile128 control is outside capture only");
    C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(&bi::bi_gemm_kernel<bi::F_BF16,0,4,2>),cudaFuncAttributeMaxDynamicSharedMemorySize,int(shared)));
  }
  if(splits==1){
    if(tile==128)bi::bi_gemm_kernel<bi::F_BF16,0,4,2><<<grid,512,shared,stream>>>(xp,K,M,d,1,reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),N,nullptr);
    else if(M>16)bi::bi_gemm_kernel<bi::F_BF16,0,4,1><<<grid,256,shared,stream>>>(xp,K,M,d,1,reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),N,nullptr);
    else bi::bi_gemm_kernel<bi::F_BF16,0,1,1><<<grid,256,shared,stream>>>(xp,K,M,d,1,reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),N,nullptr);
  } else {
    auto ws=torch::empty({splits,M,N},x.options().dtype(torch::kFloat));
    float* wp=ws.data_ptr<float>();
    if(tile==128)bi::bi_gemm_kernel<bi::F_BF16,0,4,2><<<grid,512,shared,stream>>>(xp,K,M,d,int(splits),nullptr,N,wp);
    else if(M>16)bi::bi_gemm_kernel<bi::F_BF16,0,4,1><<<grid,256,shared,stream>>>(xp,K,M,d,int(splits),nullptr,N,wp);
    else bi::bi_gemm_kernel<bi::F_BF16,0,1,1><<<grid,256,shared,stream>>>(xp,K,M,d,int(splits),nullptr,N,wp);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    bi::bi_reduce_kernel<<<dim3((N+255)/256,M),256,0,stream>>>(wp,int(splits),M,N,nullptr,reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),N);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}
