"""Explicit BHE1 Metal decode and MLX linear adapter.

The cursor is generated from the same header used by CUDA and the native CPU
codec. This decode-then-matmul path is experimental; it is not a qualified fused
serving kernel, and it does not claim CUDA's accumulation order.
Importing this module neither imports MLX nor initializes a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np

from . import context_exponent_packet as packet


class MetalDecodeError(RuntimeError):
    """Metal execution or verified packet preparation failed closed."""


_INPUTS = (
    ('main_data', 'uchar', 'main'), ('raw', 'uchar', 'raw'),
    ('tables', 'uint', 'tables'), ('context_dir', 'long', 'context_dir'),
    ('stream_offsets', 'uint', 'stream_offsets'), ('row_map', 'int', 'row_map'),
    ('tile_rc', 'long', 'tile_rc'), ('positions', 'uchar', 'positions'),
    ('counts', 'long', 'counts'), ('panel_desc', 'long', 'panel_desc'),
    ('dims', 'ulong', None),
)


def cursor_header(*, cpu=False):
    """Address-space adaptation only; fail if the shared header's layout drifts."""
    source = Path(__file__).with_name('context_exponent_packet.cuh').read_text()
    # CUDA's opt-in value snapshots leave Metal's shared decoder unchanged.
    # Recognize the exact reviewed wrapper before adapting address spaces.
    marker = '#if defined(GEOREFINE_BHE_INLINE_VALUE)'
    if marker in source:
        import hashlib
        if source.count(marker) != 1:
            raise MetalDecodeError('CUDA cursor wrapper declaration drift')
        start = source.index(marker)
        terminal = '#endif\nBHE_HD uint32_t finish'
        if source.count(terminal) != 1:
            raise MetalDecodeError('CUDA cursor wrapper terminal drift')
        end = source.index(terminal, start) + len('#endif\n')
        wrapper = source[start:end]
        expected = '96e0ae50c74c09e9181537b94c21594de135d7ec002f145899623a0e1330bf54'
        if hashlib.sha256(wrapper.encode()).hexdigest() != expected:
            raise MetalDecodeError('CUDA cursor wrapper body drift')
        source = source[:start] + source[end:]
        declaration = 'BHE_HD uint32_t group_impl('
        if source.count(declaration) != 1:
            raise MetalDecodeError('shared cursor group implementation drift')
        source = source.replace(declaration, 'BHE_GROUP uint32_t group(')
    substitutions = (
        ('constexpr uint32_t CH=', 'constant constexpr uint32_t CH=', 1),
        ('const uint8_t*', 'device const uint8_t*', 3),
        ('const uint32_t*', 'device const uint32_t*', 2),
        ('Cursor&', 'thread Cursor&', 5), ('Group&', 'thread Group&', 1),
        ('const Context&', 'thread const Context&', 1),
        ('uint32_t&', 'thread uint32_t&', 1),
    )
    for old, new, expected in substitutions:
        if source.count(old) != expected:
            raise MetalDecodeError(f'shared cursor declaration drift: {old}')
        source = source.replace(old, new)
    # MLX may bind small inputs as constant buffers.  Keep their address space
    # in inferred pointer types and in the Huffman table context.
    context_decl = 'struct Context {device const uint32_t* table;'
    if source.count(context_decl) != 1:
        context_decl = 'struct Context {const uint32_t* table;'
    if source.count(context_decl) != 1:
        raise MetalDecodeError('shared cursor Context declaration drift')
    source = source.replace(
        context_decl,
        'template <typename TablePtr>\nstruct Context {TablePtr table;',
    )
    group_decl = 'BHE_GROUP uint32_t group(thread const Context& t,thread Cursor& c,'
    if source.count(group_decl) != 1:
        raise MetalDecodeError('shared cursor group declaration drift')
    source = source.replace(
        group_decl,
        'template <typename DataPtr, typename TablePtr>\n'
        'BHE_GROUP uint32_t group(thread const Context<TablePtr>& t,'
        'thread Cursor<DataPtr>& c,',
    )
    cursor_decl = 'struct Cursor {\n  device const uint8_t* data=nullptr;'
    if source.count(cursor_decl) != 1:
        raise MetalDecodeError('shared cursor Cursor declaration drift')
    source = source.replace(
        cursor_decl,
        'template <typename DataPtr>\nstruct Cursor {DataPtr data=nullptr;',
    )
    cursor_replacements = (
        ('BHE_HD uint32_t load_be32(device const uint8_t* p){',
         'template <typename DataPtr>\n'
         'BHE_HD uint32_t load_be32(DataPtr p){'),
        ('BHE_HD uint32_t init(thread Cursor& c,device const uint8_t* data,',
         'template <typename DataPtr>\n'
         'BHE_HD uint32_t init(thread Cursor<DataPtr>& c,DataPtr data,'),
        ('BHE_HD uint32_t peek(thread Cursor& c,',
         'template <typename DataPtr>\n'
         'BHE_HD uint32_t peek(thread Cursor<DataPtr>& c,'),
        ('BHE_HD uint32_t consume(thread Cursor& c,',
         'template <typename DataPtr>\n'
         'BHE_HD uint32_t consume(thread Cursor<DataPtr>& c,'),
        ('BHE_HD uint32_t finish(thread Cursor& c){',
         'template <typename DataPtr>\n'
         'BHE_HD uint32_t finish(thread Cursor<DataPtr>& c){'),
    )
    for old, new in cursor_replacements:
        if source.count(old) != 1:
            raise MetalDecodeError(f'shared cursor function declaration drift: {old}')
        source = source.replace(old, new)
    if source.count('c=Cursor{};') != 1:
        raise MetalDecodeError('shared cursor initialization drift')
    source = source.replace('c=Cursor{};', 'c=Cursor<DataPtr>{};')
    if cpu:
        return '#define device\n#define thread\n#define constant\n' + source
    aliases = ('#include <metal_stdlib>\nusing namespace metal;\n'
               'using uint8_t=uchar; using uint16_t=ushort; using uint32_t=uint;\n'
               'using uint64_t=ulong; using int32_t=int; using int64_t=long;\n'
               '#ifndef UINT64_MAX\n#define UINT64_MAX 0xffffffffffffffffUL\n#endif\n')
    return source.replace('#include <cstdint>', aliases)


DECODE_SOURCE = r'''
const uint32_t index=thread_position_in_grid.x;
const uint32_t tile=index/256, tid=index%256;
const uint32_t N=uint32_t(dims[0]), K=uint32_t(dims[1]), nc=uint32_t(dims[2]);
if(tile>=dims[3])return;
status[index]=0;
const uint32_t row=tid/nc, cc=tid%nc;
if(row>=64)return;
const int32_t mapped=row_map[uint64_t(tile)*64+row];
if(mapped<0)return;
if(uint32_t(mapped)>=N){status[index]=1;return;}
const uint32_t original=uint32_t(mapped), panel=original/64;
auto pd=panel_desc+uint64_t(panel)*3;
const uint64_t rawbase=uint64_t(pd[1])+uint64_t(original%64)*K*(pd[2]?2:1);
if(pd[2]==1){
  for(uint32_t col=cc;col<K;col+=nc){
    const uint64_t at=rawbase+uint64_t(col)*2;
    if(at>=dims[5]||dims[5]-at<2){status[index]=4;return;}
    output[uint64_t(original)*K+col]=uint16_t(raw[at])|(uint16_t(raw[at+1])<<8);
  }
  return;
}
if(pd[2]!=0||tile_rc[tile]<0){status[index]=1;return;}
const uint64_t context=uint64_t(tile_rc[tile])*nc+cc;
if(context>=dims[7]){status[index]=2;return;}
auto d=context_dir+context*4;
if(d[0]<0||d[1]<0||uint64_t(d[0])>dims[6]||uint64_t(d[1])>dims[6]-uint64_t(d[0])){
  status[index]=2;return;
}
bhe::Context<decltype(tables)> coding{tables+d[0],uint32_t(d[1]),int32_t(d[2]),uint32_t(d[3])};
const uint64_t offset=uint64_t(panel)*(64*nc+1)+(original%64)*nc+cc;
const uint64_t bitbase=uint64_t(pd[0])*8;
bhe::Cursor<decltype(main_data)> cursor;
if(bhe::init(cursor,main_data,dims[4],bitbase+stream_offsets[offset],bitbase+stream_offsets[offset+1])){
  status[index]=int32_t(cursor.error);return;
}
for(uint32_t chunk=0;chunk<K/128&&!cursor.error;++chunk){
  auto bounds=counts+uint64_t(chunk)*(nc+1);
  if(bounds[cc]<0||bounds[cc+1]<bounds[cc]||bounds[cc+1]>128){cursor.error=2;break;}
  const uint32_t first=uint32_t(bounds[cc]), count=uint32_t(bounds[cc+1]-bounds[cc]);
  uint32_t produced=0;
  while(produced<count&&!cursor.error){
    bhe::Group group;
    if(bhe::group(coding,cursor,count-produced,group))break;
    for(uint32_t j=0;j<group.count;++j){
      const uint32_t col=positions[uint64_t(chunk)*128+first+produced+j];
      const uint64_t at=rawbase+uint64_t(chunk)*128+col;
      if(col>=128||at>=dims[5]){cursor.error=4;break;}
      output[uint64_t(original)*K+chunk*128+col]=bhe::combine((group.symbols>>(8*j))&255,raw[at]);
    }
    produced+=group.count;
  }
}
status[index]=int32_t(bhe::finish(cursor));
'''


def standalone_source():
    """Actual Metal source for offline compiler checks without GPU execution."""
    parameters = [f'device const {dtype}* {name} [[buffer({i})]]'
                  for i,(name,dtype,_) in enumerate(_INPUTS)]
    parameters += ['device ushort* output [[buffer(11)]]',
                   'device int* status [[buffer(12)]]',
                   'uint3 thread_position_in_grid [[thread_position_in_grid]]']
    return cursor_header() + '\nkernel void bhe_decode(' + ',\n'.join(parameters) + '){\n' + DECODE_SOURCE + '\n}\n'


def cpu_source():
    """Compile the identical kernel body serially for independent CPU validation."""
    types = {'uchar':'uint8_t','uint':'uint32_t','long':'int64_t','int':'int32_t','ulong':'uint64_t'}
    params = [f'const {types[t]}* {n}' for n,t,_ in _INPUTS]
    params += ['uint16_t* output','int32_t* status']
    names = [n for n,_,_ in _INPUTS] + ['output','status']
    return (cursor_header(cpu=True) + '\nstruct Position {uint32_t x;};\n'
            + 'static void lane(' + ','.join(params) + ',Position thread_position_in_grid){\n'
            + DECODE_SOURCE + '\n}\nextern "C" void bhe_metal_cpu(' + ','.join(params) + '){\n'
            + 'for(uint32_t i=0;i<dims[3]*256;++i)lane(' + ','.join(names) + ',Position{i});\n}\n')


def host_inputs(blobs, *, _table_cache=None):
    arrays = packet._resident_matrix_arrays(blobs,_table_cache=_table_cache)
    inputs = []
    for _, dtype, key in _INPUTS[:-1]:
        types = {'uchar':np.uint8,'uint':np.uint32,'long':np.int64,'int':np.int32}
        inputs.append(np.ascontiguousarray(arrays[key],dtype=types[dtype]).reshape(-1))
    dims = np.array([arrays['N'],arrays['K'],arrays['nc'],arrays['tiles'],
                     inputs[0].size,inputs[1].size,inputs[2].size,arrays['context_dir'].shape[0]],dtype=np.uint64)
    return arrays, (*inputs,dims)


def allocation_ledger(arrays, *, max_rows):
    if type(max_rows) is not int or not 1<=max_rows<=512:
        raise ValueError('max_rows must be 1..512')
    resident = sum(arrays[k].nbytes for _,_,k in _INPUTS[:-1]) + 8*8
    transient = 2*int(arrays['N'])*int(arrays['K']) + 4*int(arrays['tiles'])*256
    transient += 2*max_rows*(int(arrays['N'])+int(arrays['K']))
    return dict(resident_bytes=resident, known_transient_bytes=transient,
                known_total_bytes=resident+transient,
                exclusions=['MLX matmul workspace','allocator reserve','other model tensors'])


@dataclass(frozen=True)
class MetalPlan:
    _inputs: tuple
    N: int
    K: int
    tiles: int
    max_rows: int
    ledger: dict
    source_digests: tuple
    _header: str

    @property
    def source_header_sha256(self):
        return hashlib.sha256(self._header.encode()).hexdigest()


def prepare(blobs, *, max_rows=128, allocation_budget_bytes=512*1024**2):
    arrays, inputs = host_inputs(blobs)
    ledger = allocation_ledger(arrays,max_rows=max_rows)
    if type(allocation_budget_bytes) is not int or not 0<allocation_budget_bytes<=900*1024**2:
        raise ValueError('Metal allocation budget must be positive and at most 900 MiB')
    if ledger['known_total_bytes']>allocation_budget_bytes:
        raise MetalDecodeError('BHE Metal plan exceeds the allocation budget')
    import mlx.core as mx
    if not mx.metal.is_available():
        raise MetalDecodeError('Metal GPU is unavailable')
    # Preserve an existing stricter cap rather than enlarging it.
    previous = mx.set_memory_limit(allocation_budget_bytes)
    mx.set_memory_limit(min(previous,allocation_budget_bytes))
    resident = tuple(mx.array(a) for a in inputs)
    mx.eval(*resident)
    return MetalPlan(resident,int(arrays['N']),int(arrays['K']),int(arrays['tiles']),max_rows,
                     ledger,tuple(arrays['source_digests']),cursor_header())


@lru_cache(maxsize=8)
def _kernel(header):
    import mlx.core as mx
    return mx.fast.metal_kernel(name='bhe1_decode',input_names=[n for n,_,_ in _INPUTS],
                                output_names=['output','status'],header=header,source=DECODE_SOURCE)


def decode(plan):
    if not isinstance(plan,MetalPlan):
        raise TypeError("prepare a MetalPlan before decoding")
    import mlx.core as mx
    words,status = _kernel(plan._header)(inputs=list(plan._inputs),grid=(plan.tiles*256,1,1),threadgroup=(256,1,1),
                             output_shapes=[(plan.N,plan.K),(plan.tiles,256)],
                             output_dtypes=[mx.uint16,mx.int32])
    from glc_loader.metal.status_guard import register
    if register(status, words):
        return mx.view(words,mx.bfloat16)
    mx.eval(status)
    if np.any(np.asarray(status)):
        raise MetalDecodeError('BHE Metal decoder returned a nonzero device status')
    return mx.view(words,mx.bfloat16)


class BHELinearMLX:
    """Bounded decode-then-matmul; full-model speed and parity remain unqualified."""
    def __init__(self,plan,*,logical_rows=None):
        self.plan=plan
        self.in_features=plan.K
        self.out_features=plan.N if logical_rows is None else logical_rows
        if type(self.out_features) is not int or not 1<=self.out_features<=plan.N:
            raise ValueError('invalid logical output rows')

    @property
    def resident_bytes(self):
        return self.plan.ledger['resident_bytes']

    def __call__(self,x):
        import mlx.core as mx
        if x.ndim<2 or x.shape[-1]!=self.in_features or x.dtype!=mx.bfloat16:
            raise ValueError('BHE Metal linear requires matching BF16 activations')
        flat=x.reshape(-1,self.in_features)
        if not 1<=flat.shape[0]<=self.plan.max_rows:
            raise MetalDecodeError('BHE Metal call exceeds admitted row capacity')
        weight=decode(self.plan)
        result=mx.matmul(flat,weight[:self.out_features].T)
        # Finish this temporary weight before the next layer can allocate another.
        mx.eval(result)
        return result.reshape(*x.shape[:-1],self.out_features)


def verify_source(blobs, *, shape, source_sha256):
    """Bind logical source bytes before any MLX import or device upload.

    ``source_sha256`` must come from the model/package manifest, not from the
    packet being checked. Partial final panels contain canonical zero padding.
    """
    if (not isinstance(shape,(tuple,list)) or len(shape)!=2
            or any(type(n) is not int or n<=0 for n in shape)):
        raise ValueError('expected positive integer linear shape')
    if not isinstance(source_sha256,str) or len(source_sha256)!=64:
        raise ValueError('expected source SHA256')
    try:
        if len(bytes.fromhex(source_sha256))!=32:
            raise ValueError('expected source SHA256')
    except ValueError as exc:
        raise ValueError('expected source SHA256') from exc
    n,k=shape
    if not isinstance(blobs,(tuple,list)) or len(blobs)!=(n+63)//64:
        raise MetalDecodeError('packet count differs from the expected source shape')
    digest=hashlib.sha256()
    table_cache={}
    for index,blob in enumerate(blobs):
        words=packet.decode_cpu(blob,_table_cache=table_cache)
        if words.shape!=(64,k):
            raise MetalDecodeError('packet columns differ from the expected source shape')
        take=min(64,n-index*64)
        if np.any(words[take:]):
            raise MetalDecodeError('nonzero logical row padding')
        digest.update(np.ascontiguousarray(words[:take],dtype='<u2').tobytes())
    if digest.hexdigest()!=source_sha256.lower():
        raise MetalDecodeError('BHE Metal source SHA256 mismatch')
    return n,k


def prepare_verified_linear(blobs, *, shape, source_sha256, max_rows=128,
                            allocation_budget_bytes=512*1024**2):
    n,k=verify_source(blobs,shape=shape,source_sha256=source_sha256)
    plan=prepare(blobs,max_rows=max_rows,allocation_budget_bytes=allocation_budget_bytes)
    return BHELinearMLX(plan,logical_rows=n)
