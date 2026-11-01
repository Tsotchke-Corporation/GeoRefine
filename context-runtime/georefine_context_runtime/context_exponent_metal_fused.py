"""BHE1 decode directly into bounded threadgroup tiles, then SIMD-group GEMM.

No N*K dense device weight is produced. The FP32 SIMD accumulation order is an
explicit new backend: full-model parity and speed need hardware qualification.
The CPU control executes the same cursor/tile decoder; it does not emulate GPU
SIMD rounding. Importing this module never imports MLX or initializes a GPU.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import numpy as np
from . import context_exponent_metal as metal

TILE_M=16
THREADGROUP_BYTES=64*128*2 + 32*64*4 + 16*32*4 + 4

# Shared by the actual Metal kernel and the CPU corruption/exactness control.
TILE_DECODER=r'''
template <typename TablePtr,typename DataPtr>
struct FusedLane {
  bhe::Cursor<DataPtr> cursor; bhe::Context<TablePtr> coding;
  uint64_t rawbase=0; uint32_t original=0,cc=0,error=0;
  bool active=false,raw_mode=false;
};
template <typename TablePtr,typename DataPtr,typename MainPtr,typename TablesPtr,typename ContextPtr,
          typename StreamPtr,typename RowPtr,typename TilePtr,
          typename PanelPtr,typename DimsPtr>
inline void init_lane(thread FusedLane<TablePtr,DataPtr>& l,uint32_t tile,uint32_t tid,
  MainPtr main_data,TablesPtr tables,ContextPtr context_dir,
  StreamPtr stream_offsets,RowPtr row_map,TilePtr tile_rc,
  PanelPtr panel_desc,DimsPtr dims){
  const uint32_t nc=uint32_t(dims[2]);
  if(nc!=1&&nc!=2&&nc!=4){l.error=1;return;}
  const uint32_t row=tid/nc;l.cc=tid%nc;
  if(row>=64)return;
  const int32_t mapped=row_map[uint64_t(tile)*64+row];
  if(mapped<-1){l.error=1;return;}
  if(mapped<0)return;
  if(uint32_t(mapped)>=dims[0]){l.error=1;return;}
  l.active=true;l.original=uint32_t(mapped);
  const uint32_t panel=l.original/64;
  auto pd=panel_desc+uint64_t(panel)*3;
  if(pd[0]<0||pd[1]<0||(pd[2]!=0&&pd[2]!=1)){l.error=1;return;}
  l.raw_mode=pd[2]==1;
  l.rawbase=uint64_t(pd[1])+uint64_t(l.original%64)*dims[1]*(l.raw_mode?2:1);
  if(l.raw_mode)return;
  if(tile_rc[l.original]<0){l.error=1;return;}
  const uint64_t context=uint64_t(tile_rc[l.original])*nc+l.cc;
  if(context>=dims[7]){l.error=2;return;}
  auto d=context_dir+context*4;
  if(d[0]<0||d[1]<0||uint64_t(d[0])>dims[6]||uint64_t(d[1])>dims[6]-uint64_t(d[0])){l.error=2;return;}
  l.coding=bhe::Context<decltype(tables)>{tables+d[0],uint32_t(d[1]),int32_t(d[2]),uint32_t(d[3])};
  const uint64_t at=uint64_t(panel)*(64*nc+1)+(l.original%64)*nc+l.cc;
  const uint64_t base=uint64_t(pd[0])*8;
  l.error=bhe::init(l.cursor,main_data,dims[4],base+stream_offsets[at],base+stream_offsets[at+1]);
}
template <typename TablePtr,typename DataPtr,typename RawPtr,typename PositionPtr,typename CountsPtr,typename DimsPtr>
inline void decode_tile_chunk(thread FusedLane<TablePtr,DataPtr>& l,uint32_t chunk,uint32_t tid,
  RawPtr raw,PositionPtr positions,CountsPtr counts,DimsPtr dims,
  threadgroup uint16_t* words){
  if(!l.active||l.error)return;
  const uint32_t nc=uint32_t(dims[2]),row=tid/nc;
  if(l.raw_mode){
    for(uint32_t col=l.cc;col<128;col+=nc){
      const uint64_t at=l.rawbase+(uint64_t(chunk)*128+col)*2;
      if(at>=dims[5]||dims[5]-at<2){l.error=4;return;}
      words[row*128+col]=uint16_t(raw[at])|(uint16_t(raw[at+1])<<8);
    }
    return;
  }
  auto bounds=counts+uint64_t(chunk)*(nc+1);
  if(bounds[l.cc]<0||bounds[l.cc+1]<bounds[l.cc]||bounds[l.cc+1]>128){l.error=2;return;}
  const uint32_t first=uint32_t(bounds[l.cc]),count=uint32_t(bounds[l.cc+1]-bounds[l.cc]);
  uint32_t produced=0;
  while(produced<count&&!l.cursor.error){
    bhe::Group group;
    if(bhe::group(l.coding,l.cursor,count-produced,group))break;
    for(uint32_t j=0;j<group.count;++j){
      const uint32_t col=positions[uint64_t(chunk)*128+first+produced+j];
      const uint64_t at=l.rawbase+uint64_t(chunk)*128+col;
      if(col>=128||at>=dims[5]){l.cursor.error=4;break;}
      words[row*128+col]=bhe::combine((group.symbols>>(8*j))&255,raw[at]);
    }
    produced+=group.count;
  }
  l.error=l.cursor.error;
}
template <typename TablePtr,typename DataPtr>
inline uint32_t finish_lane(thread FusedLane<TablePtr,DataPtr>& l){
  if(l.active&&!l.raw_mode&&!l.error)l.error=bhe::finish(l.cursor);
  return l.error;
}
inline float from_bf16(uint16_t word){return as_type<float>(uint32_t(word)<<16);}
inline uint16_t to_bf16(float value){
  uint32_t bits=as_type<uint32_t>(value);
  if((bits&0x7f800000u)==0x7f800000u&&(bits&0x007fffffu))return uint16_t((bits>>16)|0x40u);
  return uint16_t((bits+0x7fffu+((bits>>16)&1u))>>16);
}
'''

FUSED_SOURCE=r'''
const uint32_t tid=thread_index_in_threadgroup,sg=simdgroup_index_in_threadgroup;
const uint32_t tile=threadgroup_position_in_grid.x,mtile=threadgroup_position_in_grid.y;
const uint32_t M=uint32_t(call_dims[0]),N=uint32_t(call_dims[1]),K=uint32_t(dims[1]);
const uint64_t status_at=(uint64_t(mtile)*dims[3]+tile)*256+tid;
threadgroup uint16_t words[64*128];
threadgroup float weight32[32*64];
threadgroup float input32[16*32];
threadgroup atomic_uint failure;
if(tid==0)atomic_store_explicit(&failure,0u,memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);
FusedLane<decltype(tables),decltype(main_data)> lane;
#ifndef BHE_DENSE_REFERENCE
init_lane(lane,tile,tid,main_data,tables,context_dir,stream_offsets,row_map,tile_rc,panel_desc,dims);
if(lane.error)atomic_fetch_or_explicit(&failure,lane.error,memory_order_relaxed);
#endif
simdgroup_float8x8 acc0=make_filled_simdgroup_matrix<float,8,8>(0.0f);
simdgroup_float8x8 acc1=make_filled_simdgroup_matrix<float,8,8>(0.0f);
for(uint32_t chunk=0;chunk<K/128;++chunk){
  for(uint32_t i=tid;i<64*128;i+=256)words[i]=0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
#ifdef BHE_DENSE_REFERENCE
  for(uint32_t i=tid;i<64*128;i+=256){
    const int32_t mapped=row_map[uint64_t(tile)*64+i/128];
    if(mapped>=0&&uint32_t(mapped)<dims[0])words[i]=dense_words[uint64_t(mapped)*K+chunk*128+i%128];
  }
#else
  decode_tile_chunk(lane,chunk,tid,raw,positions,counts,dims,words);
  if(lane.error)atomic_fetch_or_explicit(&failure,lane.error,memory_order_relaxed);
#endif
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(atomic_load_explicit(&failure,memory_order_relaxed))break;
  for(uint32_t slice=0;slice<4;++slice){
    for(uint32_t i=tid;i<32*64;i+=256)weight32[i]=from_bf16(words[(i%64)*128+slice*32+i/64]);
    for(uint32_t i=tid;i<16*32;i+=256){
      const uint32_t m=mtile*16+i/32,k=chunk*128+slice*32+i%32;
      input32[i]=m<M?from_bf16(activation_words[uint64_t(m)*K+k]):0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint32_t k=0;k<32;k+=8){
      simdgroup_float8x8 a0,a1,b;
      simdgroup_load(a0,input32+k,32);
      simdgroup_load(a1,input32+8*32+k,32);
      simdgroup_load(b,weight32+k*64+sg*8,64);
      simdgroup_multiply_accumulate(acc0,a0,b,acc0);
      simdgroup_multiply_accumulate(acc1,a1,b,acc1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
#ifndef BHE_DENSE_REFERENCE
if(finish_lane(lane))atomic_fetch_or_explicit(&failure,lane.error,memory_order_relaxed);
#endif
status[status_at]=int32_t(lane.error);
threadgroup_barrier(mem_flags::mem_threadgroup);
if(atomic_load_explicit(&failure,memory_order_relaxed))return;
// Reuse weight32 after the final multiply; there is no global dense-weight buffer.
simdgroup_store(acc0,weight32+sg*8,64);
simdgroup_store(acc1,weight32+8*64+sg*8,64);
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint32_t i=tid;i<16*64;i+=256){
  const uint32_t m=mtile*16+i/64;
  const int32_t original=row_map[uint64_t(tile)*64+i%64];
  if(m<M&&original>=0&&uint32_t(original)<N)output[uint64_t(m)*N+uint32_t(original)]=to_bf16(weight32[i]);
}
'''


def header(*,cpu=False,dense=False):
    pre=metal.cursor_header(cpu=cpu)
    if cpu:
        pre+='\n#define threadgroup\n#include <cstring>\n#include <cmath>\n#include <algorithm>\ntemplate<class O,class I> O as_type(I x){static_assert(sizeof(O)==sizeof(I));O y;std::memcpy(&y,&x,sizeof(y));return y;}\n'
    else:pre+='\n#include <metal_simdgroup_matrix>\n'
    if dense:pre+='\n#define BHE_DENSE_REFERENCE 1\n'
    return pre+TILE_DECODER


def standalone_source(*,dense=False):
    params=[f'device const {t}* {n} [[buffer({i})]]' for i,(n,t,_) in enumerate(metal._INPUTS)]
    names=[('activation_words','ushort'),('call_dims','ulong')]
    if dense:names.append(('dense_words','ushort'))
    for name,kind in names:params.append(f'device const {kind}* {name} [[buffer({len(params)})]]')
    params.append(f'device ushort* output [[buffer({len(params)})]]')
    params.append(f'device int* status [[buffer({len(params)})]]')
    params+=['uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]',
             'uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]',
             'uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]']
    return header(dense=dense)+'\nkernel void bhe_fused('+','.join(params)+'){\n'+FUSED_SOURCE+'\n}\n'


def host_inputs(blobs, *, _table_cache=None):
    """Pack full output tiles while retaining each row's own coding context.

    Class-grouped decoder tiles can contain mostly padding. A fused GEMM must
    not spend matrix work on those holes, so invert their verified row mapping.
    """
    arrays,_=metal.host_inputs(blobs,_table_cache=_table_cache)
    n=int(arrays['N']);contexts=np.full(n,-1,dtype=np.int64);seen=np.zeros(n,dtype=np.bool_)
    for tile,rows in enumerate(arrays['row_map'].reshape(-1,64)):
        for mapped in rows:
            mapped=int(mapped)
            if mapped<0:continue
            if mapped>=n or seen[mapped]:raise metal.MetalDecodeError('invalid or duplicate fused row mapping')
            seen[mapped]=True;contexts[mapped]=int(arrays['tile_rc'].reshape(-1)[tile])
    if not seen.all():raise metal.MetalDecodeError('fused row mapping omits source rows')
    arrays['row_map']=np.arange(n,dtype=np.int32).reshape(-1,64)
    arrays['tile_rc']=contexts
    arrays['tiles']=n//64
    types={'uchar':np.uint8,'uint':np.uint32,'long':np.int64,'int':np.int32}
    inputs=[np.ascontiguousarray(arrays[key],dtype=types[kind]).reshape(-1) for _,kind,key in metal._INPUTS[:-1]]
    dims=np.array([n,arrays['K'],arrays['nc'],arrays['tiles'],inputs[0].size,inputs[1].size,
                   inputs[2].size,arrays['context_dir'].shape[0]],dtype=np.uint64)
    return arrays,(*inputs,dims)


def allocation_ledger(arrays,*,max_rows):
    if type(max_rows) is not int or not 1<=max_rows<=512:raise ValueError('max_rows must be 1..512')
    resident=sum(arrays[k].nbytes for _,_,k in metal._INPUTS[:-1])+8*8
    status=((max_rows+15)//16)*int(arrays['tiles'])*256*4
    transient=2*max_rows*(int(arrays['N'])+int(arrays['K']))+status+16
    return dict(resident_bytes=resident,known_transient_bytes=transient,known_total_bytes=resident+transient,
                dense_weight_device_bytes=0,threadgroup_bytes=THREADGROUP_BYTES,
                exclusions=['allocator reserve','other model tensors','driver/compiler overhead'])


@dataclass(frozen=True)
class FusedPlan:
    _inputs:tuple
    N:int
    K:int
    tiles:int
    max_rows:int
    ledger:dict
    _header:str

    @property
    def source_header_sha256(self):return hashlib.sha256((self._header+FUSED_SOURCE).encode()).hexdigest()


def prepare(blobs,*,max_rows=128,allocation_budget_bytes=512*1024**2):
    arrays,inputs=host_inputs(blobs)
    ledger=allocation_ledger(arrays,max_rows=max_rows)
    if type(allocation_budget_bytes) is not int or allocation_budget_bytes<=0:raise ValueError('positive allocation budget required')
    if ledger['known_total_bytes']>allocation_budget_bytes:raise metal.MetalDecodeError('fused Metal plan exceeds allocation budget')
    if arrays['nc'] not in (1,2,4) or arrays['K']%128:raise metal.MetalDecodeError('unsupported fused packet geometry')
    import mlx.core as mx
    if not mx.metal.is_available():raise metal.MetalDecodeError('Metal GPU is unavailable')
    resident=tuple(mx.array(a) for a in inputs);mx.eval(*resident)
    return FusedPlan(resident,int(arrays['N']),int(arrays['K']),int(arrays['tiles']),max_rows,ledger,header())


@lru_cache(maxsize=8)
def _kernel(source_header,dense=False):
    import mlx.core as mx
    names=[n for n,_,_ in metal._INPUTS]+['activation_words','call_dims']
    if dense:names.append('dense_words')
    return mx.fast.metal_kernel(name='bhe1_fused_dense' if dense else 'bhe1_fused',input_names=names,
        output_names=['output','status'],header=source_header,source=FUSED_SOURCE)


def matmul(plan,x,*,logical_rows=None,dense_reference=None):
    import mlx.core as mx
    if not isinstance(plan,FusedPlan):raise TypeError('prepare a FusedPlan first')
    n=plan.N if logical_rows is None else logical_rows
    if type(n) is not int or not 1<=n<=plan.N:raise ValueError('invalid logical rows')
    if x.ndim<2 or x.dtype!=mx.bfloat16 or x.shape[-1]!=plan.K:raise ValueError('matching BF16 activations required')
    flat=x.reshape(-1,plan.K);m=flat.shape[0]
    if not 1<=m<=plan.max_rows:raise metal.MetalDecodeError('fused Metal call exceeds admitted row capacity')
    inputs=list(plan._inputs)+[mx.view(flat,mx.uint16),mx.array([m,n],dtype=mx.uint64)]
    dense=dense_reference is not None
    if dense:
        if dense_reference.shape!=(plan.N,plan.K) or dense_reference.dtype!=mx.bfloat16:raise ValueError('dense reference shape/dtype mismatch')
        inputs.append(mx.view(dense_reference,mx.uint16))
    source_header=plan._header+('\n#define BHE_DENSE_REFERENCE 1\n' if dense else '')
    words,status=_kernel(source_header,dense)(inputs=inputs,grid=(plan.tiles*256,(m+15)//16,1),threadgroup=(256,1,1),
        output_shapes=[(m,n),((m+15)//16,plan.tiles,256)],output_dtypes=[mx.uint16,mx.int32])
    from glc_loader.metal.status_guard import register
    if register(status, words):
        return mx.view(words,mx.bfloat16).reshape(*x.shape[:-1],n)
    mx.eval(status,words)
    if np.any(np.asarray(status)):raise metal.MetalDecodeError('fused Metal decoder returned a nonzero device status')
    return mx.view(words,mx.bfloat16).reshape(*x.shape[:-1],n)


class FusedBHELinearMLX:
    def __init__(self,plan,*,logical_rows):
        self.plan=plan;self.in_features=plan.K;self.out_features=logical_rows
    @property
    def resident_bytes(self):return self.plan.ledger['resident_bytes']
    def __call__(self,x):return matmul(self.plan,x,logical_rows=self.out_features)


def prepare_verified_linear(blobs,*,shape,source_sha256,max_rows=128,allocation_budget_bytes=512*1024**2):
    n,k=metal.verify_source(blobs,shape=shape,source_sha256=source_sha256)
    plan=prepare(blobs,max_rows=max_rows,allocation_budget_bytes=allocation_budget_bytes)
    return FusedBHELinearMLX(plan,logical_rows=n)


def cpu_source():
    types={'uchar':'uint8_t','uint':'uint32_t','long':'int64_t','ulong':'uint64_t','int':'int32_t'}
    params=[f'const {types[t]}* {n}' for n,t,_ in metal._INPUTS]
    params+=['const uint16_t* activation_words','const uint64_t* call_dims','uint16_t* output','int32_t* status']
    driver=r'''
const uint32_t M=uint32_t(call_dims[0]),N=uint32_t(call_dims[1]),K=uint32_t(dims[1]);
for(uint32_t mtile=0;mtile<(M+15)/16;++mtile)for(uint32_t tile=0;tile<dims[3];++tile){
  FusedLane<const uint32_t*,const uint8_t*> lanes[256];uint16_t words[64*128];float accum[16*64]={};bool bad=false;
  for(uint32_t tid=0;tid<256;++tid)init_lane(lanes[tid],tile,tid,main_data,tables,context_dir,stream_offsets,row_map,tile_rc,panel_desc,dims);
  for(uint32_t chunk=0;chunk<K/128;++chunk){
    std::fill(words,words+64*128,uint16_t(0));
    for(uint32_t tid=0;tid<256;++tid){decode_tile_chunk(lanes[tid],chunk,tid,raw,positions,counts,dims,words);bad=bad||lanes[tid].error;}
    if(bad)break;
    for(uint32_t m=0;m<16&&mtile*16+m<M;++m)for(uint32_t n=0;n<64;++n)
      for(uint32_t k=0;k<128;++k)accum[m*64+n]=std::fma(from_bf16(activation_words[uint64_t(mtile*16+m)*K+chunk*128+k]),from_bf16(words[n*128+k]),accum[m*64+n]);
  }
  for(uint32_t tid=0;tid<256;++tid){const uint32_t error=finish_lane(lanes[tid]);status[(uint64_t(mtile)*dims[3]+tile)*256+tid]=error;bad=bad||error;}
  if(!bad)for(uint32_t m=0;m<16&&mtile*16+m<M;++m)for(uint32_t n=0;n<64;++n){
    const int32_t original=row_map[uint64_t(tile)*64+n];
    if(original>=0&&uint32_t(original)<N)output[uint64_t(mtile*16+m)*N+original]=to_bf16(accum[m*64+n]);
  }
}
'''
    dense=r'''
extern "C" void bhe_dense_cpu(const uint16_t* x,const uint16_t* w,uint32_t M,uint32_t N,uint32_t K,uint16_t* out){
  for(uint32_t m=0;m<M;++m)for(uint32_t n=0;n<N;++n){float value=0;
    for(uint32_t k=0;k<K;++k)value=std::fma(from_bf16(x[uint64_t(m)*K+k]),from_bf16(w[uint64_t(n)*K+k]),value);
    out[uint64_t(m)*N+n]=to_bf16(value);}
}
'''
    return header(cpu=True)+'\nextern "C" void bhe_fused_cpu('+','.join(params)+'){'+driver+'}\n'+dense
