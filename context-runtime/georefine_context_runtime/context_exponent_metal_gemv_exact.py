"""Experimental BHE GEMV following MLX 0.30.4's selected BF16 GEMV schedules.

This backend is deliberately separate from ``context_exponent_metal_fused``.
It decodes BHE rows into a bounded threadgroup tile and accumulates scalar
float results with the same lane assignment and shuffle order as MLX's
``gemv_bfloat16_bm1_bn8_sm1_sn32_tm4_tn4`` for small-N split-K and
``gemv_bfloat16_bm4_bn1_sm1_sn32_tm4_tn4`` for larger-N regular GEMV. M=2
still routes to MLX Steel GEMM and remains separately qualified. It is a parity
prototype only: callers must compare every output word before using it.

Reference: MLX v0.30.4, ``mlx/backend/metal/kernels/gemv.metal``, released
at commit 2f324cc.  The relevant source is retained in the qualification
receipt; this module does not change the existing fused operator.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import numpy as np
from pathlib import Path

from . import context_exponent_metal as metal
from . import context_exponent_metal_fused as fused


def _steel_closure():
    """Inline the pinned Steel template closure for the experimental kernel."""
    root=Path(__file__).with_name('metal_steel_0304')
    pins=json.loads((root/'SOURCE.json').read_text())['files']
    files=('params.h','transforms.h','mma.h','loader.h','gemm.h')
    shim=r'''
#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define METAL_FUNC inline
namespace mlx { namespace steel {
template <typename T> using pointer_element_t=typename metal::remove_pointer<T>::type;
template <int V> struct Int { static constant constexpr int value=V; constexpr operator int() const { return V; } };
struct complex64_t { float real; float imag; complex64_t(float r,float i):real(r),imag(i){} };
}}
'''
    chunks=[shim]
    for name in files:
        raw=(root/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=pins[name]:
            raise RuntimeError('pinned Steel header integrity mismatch: '+name)
        text=raw.decode()
        text='\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('#include'))
        chunks.append(text)
    return '\n'.join(chunks)


EXACT_GEMV_HEADER = fused.header()+_steel_closure()+r'''
#define BHE_EXACT_GEMV_EXPLICIT_ACC 1
#ifndef BHE_EXACT_SPLIT
#define BHE_EXACT_SPLIT 0
#define BHE_EXACT_SMALL_ROWS 64
#endif
inline float bhe_mlx_reduce(threadgroup const float* p,uint base){
  float a0=p[base]+p[base+64],a8=p[base+32]+p[base+96];
  float a4=p[base+16]+p[base+80],a12=p[base+48]+p[base+112];
  float r=a0+a8;
  r=r+(a4+a12);
  float b2=p[base+8]+p[base+72],b10=p[base+40]+p[base+104];
  float b6=p[base+24]+p[base+88],b14=p[base+56]+p[base+120];
  float s=b2+b10;
  s=s+(b6+b14);
  r=r+s;
  float c1=p[base+4]+p[base+68],c9=p[base+36]+p[base+100];
  float c5=p[base+20]+p[base+84],c13=p[base+52]+p[base+116];
  float t=c1+c9;
  t=t+(c5+c13);
  float d3=p[base+12]+p[base+76],d11=p[base+44]+p[base+108];
  float d7=p[base+28]+p[base+92],d15=p[base+60]+p[base+124];
  float u=d3+d11;
  u=u+(d7+d15);
  return r+(t+u);
}
'''

EXACT_GEMV_SOURCE = r'''
const uint32_t tid=thread_position_in_threadgroup.x;
const uint32_t sg=tid/32;
const uint32_t tile=threadgroup_position_in_grid.x;
const uint32_t out_block=threadgroup_position_in_grid.y;
const uint32_t m=threadgroup_position_in_grid.z;
const uint32_t M=uint32_t(call_dims[0]),N=uint32_t(call_dims[1]),K=uint32_t(dims[1]);
const bool m2=(M==2);
const bool split_k=BHE_EXACT_SPLIT;
const uint32_t out_blocks=1;
const uint64_t status_at=(uint64_t(m)*out_blocks+out_block)*dims[3]*256+uint64_t(tile)*256+tid;
threadgroup uint16_t words[64*128];
threadgroup atomic_uint failure;
threadgroup float partial[8*32*4];
if(tid==0) atomic_store_explicit(&failure,0u,memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);
FusedLane<decltype(tables),decltype(main_data)> lane;
init_lane(lane,tile,tid,main_data,tables,context_dir,stream_offsets,row_map,
          tile_rc,panel_desc,dims);
if(lane.error) atomic_fetch_or_explicit(&failure,lane.error,memory_order_relaxed);
if(m2){
  using steel_kernel=mlx::steel::GEMMKernel<bfloat16_t,float,16,64,16,2,4,false,true,true,true>;
  threadgroup bfloat16_t steel_a[steel_kernel::tgp_mem_size_a];
  threadgroup bfloat16_t steel_b[steel_kernel::tgp_mem_size_b];
  threadgroup float steel_partial[16*2*64];
  const uint32_t sgid=tid/32,slane=tid%32;
  thread steel_kernel::mma_t mma_op(sgid,slane);
  const uint32_t k_tiles=K/16;
  const uint32_t partitions=k_tiles<16?2:(k_tiles<32?4:(k_tiles<64?8:16));
  const uint32_t partition_size=(k_tiles/partitions)*16;
  for(uint32_t chunk=0;chunk<K/128;++chunk){
    for(uint32_t i=tid;i<64*128;i+=256) words[i]=0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    decode_tile_chunk(lane,chunk,tid,raw,positions,counts,dims,words);
    if(lane.error) atomic_fetch_or_explicit(&failure,lane.error,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(!atomic_load_explicit(&failure,memory_order_relaxed)){
      for(uint32_t block=0;block<8;++block){
        const uint32_t g=chunk*128+block*16;
        const uint32_t p=min(g/partition_size,partitions-1);
        if(g==p*partition_size){
          for(short i=0;i<decltype(mma_op.Ctile)::kElemsPerTile;++i) mma_op.Ctile.elems()[i]=0.0f;
        }
        if(tid<256){
          for(uint32_t i=tid;i<16*16;i+=256){
            const uint32_t r=i/16,c=i%16;
            steel_a[r*24+c]=(r<2)?bfloat16_t(from_bf16(activation_words[uint64_t(r)*K+g+c])):bfloat16_t(0);
          }
          for(uint32_t i=tid;i<64*16;i+=256){
            const uint32_t c=i/16,r=i%16;
            steel_b[c*24+r]=bfloat16_t(from_bf16(words[c*128+(g%128)+r]));
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(steel_a,steel_b);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint32_t partition_end=(p==partitions-1)?K:(p+1)*partition_size;
        if(g+16==partition_end && mma_op.sm<2)
          mma_op.Ctile.template store<float,2,4,64,1>(steel_partial+p*128+mma_op.sm*64+mma_op.sn);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if(finish_lane(lane)) atomic_fetch_or_explicit(&failure,lane.error,memory_order_relaxed);
  status[status_at]=int32_t(lane.error);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(!atomic_load_explicit(&failure,memory_order_relaxed) && tid<128){
    const uint32_t local_row=tid%64, output_m=tid/64;
    float outv=0.0f;
    for(uint32_t p=0;p<partitions;++p) outv=outv+steel_partial[p*128+tid];
    const int32_t original=row_map[uint64_t(tile)*64+local_row];
    if(original>=0 && uint32_t(original)<N) output[uint64_t(output_m)*N+uint32_t(original)]=to_bf16(outv);
  }
  return;
}
constexpr uint32_t result_count=BHE_EXACT_SPLIT?BHE_EXACT_SMALL_ROWS:8;
thread float results[result_count];
for(uint32_t r=0;r<result_count;++r) results[r]=0.0f;
for(uint32_t chunk=0;chunk<K/128;++chunk){
  for(uint32_t i=tid;i<64*128;i+=256) words[i]=0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  decode_tile_chunk(lane,chunk,tid,raw,positions,counts,dims,words);
  if(lane.error) atomic_fetch_or_explicit(&failure,lane.error,memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(atomic_load_explicit(&failure,memory_order_relaxed)) break;
  if((!split_k || sg==(chunk%8)) && m<M){
  const uint32_t simd_lid=tid%32;
      const uint32_t kbase=chunk*128+simd_lid*4;
      const uint32_t row=split_k?0:sg*8;
      for(uint32_t tn=0;tn<4;++tn){
        const float xv=from_bf16(activation_words[uint64_t(m)*K+kbase+tn]);
        for(uint32_t r=0;r<result_count;++r)
          results[r]=fma(xv,from_bf16(words[(row+r)*128+(kbase%128)+tn]),results[r]);
      }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
if(finish_lane(lane)) atomic_fetch_or_explicit(&failure,lane.error,memory_order_relaxed);
status[status_at]=int32_t(lane.error);
threadgroup_barrier(mem_flags::mem_threadgroup);
if(atomic_load_explicit(&failure,memory_order_relaxed)) return;
for(uint32_t base=0;base<result_count;base+=4){
  for(uint32_t r=0;r<4;++r)
    partial[(sg*32+(tid%32))*4+r]=results[base+r];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if((tid%32)==0){
    for(uint32_t r=0;r<4;++r)
      partial[sg*128+r]=bhe_mlx_reduce(partial,sg*128+r);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if((!split_k || sg==0) && (tid%32)==0){
    for(uint32_t r=0;r<4;++r){
      float value=partial[sg*128+r];
      if(split_k) for(uint32_t g=1;g<8;++g) value=value+partial[g*128+r];
      const uint32_t row=(split_k?0:sg*8)+base+r;
      const int32_t original=row_map[uint64_t(tile)*64+row];
      if(original>=0 && uint32_t(original)<N)
        output[uint64_t(m)*N+uint32_t(original)]=to_bf16(value);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
'''


def standalone_source() -> str:
    params=[f'device const {t}* {n} [[buffer({i})]]' for i,(n,t,_) in enumerate(metal._INPUTS)]
    params += [f'device const ushort* activation_words [[buffer({len(params)})]]',
               f'device const ulong* call_dims [[buffer({len(params)+1})]]',
               f'device ushort* output [[buffer({len(params)+2})]]',
               f'device int* status [[buffer({len(params)+3})]]',
               'uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]',
               'uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]',
               'uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]']
    return EXACT_GEMV_HEADER+'\n[[kernel, max_total_threads_per_threadgroup(256)]] void bhe_exact_gemv('+','.join(params)+'){\n'+EXACT_GEMV_SOURCE+'\n}\n'


@dataclass(frozen=True)
class ExactGEMVPlan:
    _inputs: tuple
    N: int
    K: int
    tiles: int
    source_header_sha256: str
    ledger: dict
    max_rows: int


host_inputs = fused.host_inputs


def allocation_ledger(arrays, *, max_rows):
    if type(max_rows) is not int or not 1<=max_rows<=2:
        raise ValueError('exact Metal operator supports max_rows=1 or 2')
    ledger=fused.allocation_ledger(arrays,max_rows=max_rows)
    # M1 split-K has up to sixteen output groups; M2 uses four. Admit
    # the larger status allocation even when the logical N is not known yet.
    status=16*max_rows*int(arrays['tiles'])*256*4
    transient=2*max_rows*(int(arrays['N'])+int(arrays['K']))+status+16
    return dict(ledger,known_transient_bytes=transient,
                known_total_bytes=ledger['resident_bytes']+transient,
                threadgroup_bytes=32768)


def prepare(blobs, *, max_rows=2, allocation_budget_bytes=512*1024**2):
    if type(allocation_budget_bytes) is not int or not 0<allocation_budget_bytes<=900*1024**2:
        raise ValueError('Metal allocation budget must be positive and at most 900 MiB')
    arrays, inputs=fused.host_inputs(blobs)
    ledger=allocation_ledger(arrays,max_rows=max_rows)
    if ledger['known_total_bytes']>allocation_budget_bytes:
        raise metal.MetalDecodeError('exact GEMV plan exceeds allocation budget')
    if arrays['nc'] not in (1,2,4) or arrays['K']%128:
        raise metal.MetalDecodeError('unsupported exact GEMV packet geometry')
    import mlx.core as mx
    if not mx.metal.is_available():
        raise metal.MetalDecodeError('Metal GPU is unavailable')
    previous=mx.set_memory_limit(allocation_budget_bytes)
    mx.set_memory_limit(min(previous,allocation_budget_bytes))
    resident=tuple(mx.array(a) for a in inputs); mx.eval(*resident)
    return ExactGEMVPlan(resident,int(arrays['N']),int(arrays['K']),int(arrays['tiles']),
                         hashlib.sha256((EXACT_GEMV_HEADER+EXACT_GEMV_SOURCE).encode()).hexdigest(),
                         ledger,max_rows)


def _kernel_name(source_header,split_k,small_rows):
    # Give every complete source/specialization a distinct exported name.
    # Deferred graphs can retain several specialized pipelines at once.
    header=f'#define BHE_EXACT_SPLIT {int(split_k)}\n#define BHE_EXACT_SMALL_ROWS {small_rows}\n'+source_header
    suffix=hashlib.sha256((header+'\n'+EXACT_GEMV_SOURCE).encode()).hexdigest()[:16]
    return f'bhe_exact_gemv_v2_{suffix}_s{int(split_k)}_r{int(small_rows)}'


@lru_cache(maxsize=8)
def _kernel(source_header,split_k,small_rows):
    import mlx.core as mx
    names=[n for n,_,_ in metal._INPUTS]+['activation_words','call_dims']
    name=_kernel_name(source_header,split_k,small_rows)
    return mx.fast.metal_kernel(name=name,input_names=names,
        output_names=['output','status'],
        header=f'#define BHE_EXACT_SPLIT {int(split_k)}\n#define BHE_EXACT_SMALL_ROWS {small_rows}\n'+source_header,
        source=EXACT_GEMV_SOURCE)


def matmul(plan, x, *, logical_rows=None):
    import mlx.core as mx
    if not isinstance(plan,ExactGEMVPlan): raise TypeError('prepare an ExactGEMVPlan first')
    n=plan.N if logical_rows is None else logical_rows
    if type(n) is not int or not 1<=n<=plan.N: raise ValueError('invalid logical rows')
    if x.ndim<2 or x.dtype!=mx.bfloat16 or x.shape[-1]!=plan.K:
        raise ValueError('matching BF16 activations required')
    flat=x.reshape(-1,plan.K); m=flat.shape[0]
    if m not in (1,2) or m>plan.max_rows:
        raise metal.MetalDecodeError('exact GEMV call exceeds admitted row capacity')
    inputs=list(plan._inputs)+[mx.view(flat,mx.uint16),mx.array([m,n],dtype=mx.uint64)]
    split_k=plan.K>=16*n
    out_blocks=1
    grid_m=1 if m==2 else m
    words,status=_kernel(EXACT_GEMV_HEADER,split_k,16 if n<=16 else 64)(inputs=inputs,grid=(plan.tiles*256,out_blocks,grid_m),
        threadgroup=(256,1,1),output_shapes=[(m,n),(grid_m*out_blocks,plan.tiles,256)],
        output_dtypes=[mx.uint16,mx.int32])
    from glc_loader.metal.status_guard import register
    if register(status,words):
        return mx.view(words,mx.bfloat16).reshape(*x.shape[:-1],n)
    mx.eval(status,words)
    if np.any(np.asarray(status)): raise metal.MetalDecodeError('exact GEMV returned nonzero device status')
    return mx.view(words,mx.bfloat16).reshape(*x.shape[:-1],n)


class ExactBHELinearMLX:
    def __init__(self,plan,*,logical_rows):
        self.plan=plan;self.in_features=plan.K;self.out_features=logical_rows
    @property
    def resident_bytes(self):
        return self.plan.ledger['resident_bytes']
    def __call__(self,x):
        return matmul(self.plan,x,logical_rows=self.out_features)


def prepare_verified_linear(blobs,*,shape,source_sha256,max_rows=2,
                            allocation_budget_bytes=512*1024**2):
    if type(max_rows) is not int or not 1<=max_rows<=2:
        raise ValueError('exact Metal operator supports max_rows=1 or 2')
    n,k=metal.verify_source(blobs,shape=shape,source_sha256=source_sha256)
    plan=prepare(blobs,max_rows=max_rows,allocation_budget_bytes=allocation_budget_bytes)
    return ExactBHELinearMLX(plan,logical_rows=n)
