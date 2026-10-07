"""Indexed CUDA decoding of original BCTX frames; no alternate-codec transcode.

The checkpoint index is derived from the exact rANS streams on the CPU once.
All index, table and label allocations are included in resident_bytes.
"""
from __future__ import annotations
import ctypes
import hashlib
import math
from pathlib import Path
import struct
import subprocess
import threading
import numpy as np
try:
    from scripts import bitexact_context_codec as base
except ImportError:
    import bitexact_context_codec as base

_LOCK = threading.Lock()
_LIB = None


def _index_lib():
    global _LIB
    with _LOCK:
        if _LIB is not None:
            return _LIB
        source = Path(__file__).with_name('bitexact_context_gpu_index.cpp')
        dependency = source.with_name('bitexact_context_ans.cpp')
        identity = hashlib.sha256(source.read_bytes() + dependency.read_bytes()).hexdigest()[:20]
        cache = source.parent.parent / '.scratch/context-gpu-index'
        cache.mkdir(parents=True, exist_ok=True)
        binary = cache / ('index-' + identity + '.so')
        if not binary.exists():
            subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(source),str(dependency),'-o',str(binary)],check=True,capture_output=True)
        lib = ctypes.CDLL(str(binary))
        U8 = ctypes.POINTER(ctypes.c_uint8)
        U16 = ctypes.POINTER(ctypes.c_uint16)
        U32 = ctypes.POINTER(ctypes.c_uint32)
        lib.bcx_checkpoint.argtypes = [U8,ctypes.c_size_t,ctypes.c_uint64,ctypes.c_uint32,ctypes.c_uint32,U8,U8,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_uint32,U16,U32,ctypes.c_uint32,U32,U32,U16]
        lib.bcx_checkpoint.restype = ctypes.c_int
        _LIB = lib
        return lib


def _ptr(a, ctype):
    return a.ctypes.data_as(ctypes.POINTER(ctype))


def checkpoints(stream, freq, rows, columns, shape, nr, nc, stride, main=None):
    n = math.prod(shape)
    data = np.frombuffer(stream, dtype=np.uint8)
    states = np.empty((n + stride - 1) // stride, dtype=np.uint32)
    offsets = np.empty_like(states)
    out = np.empty(n, dtype=np.uint16)
    context = ctypes.POINTER(ctypes.c_uint16)() if main is None else _ptr(main,ctypes.c_uint16)
    status = _index_lib().bcx_checkpoint(_ptr(data,ctypes.c_uint8),len(data),n,*shape,
        _ptr(rows,ctypes.c_uint8),_ptr(columns,ctypes.c_uint8),nr,nc,int(main is not None),
        context,_ptr(freq,ctypes.c_uint32),stride,_ptr(states,ctypes.c_uint32),_ptr(offsets,ctypes.c_uint32),_ptr(out,ctypes.c_uint16))
    if status:
        raise base.CodecError(f'checkpoint validation failed ({status})')
    return states, offsets, out


def index_frame(frame: bytes, stride: int = 1024):
    if not isinstance(stride,int) or not 1 <= stride <= 65536:
        raise base.CodecError('invalid checkpoint stride')
    if len(frame) < base.HEADER.size + 32 or hashlib.sha256(frame[:-32]).digest() != frame[-32:]:
        raise base.CodecError('invalid frame checksum/length')
    magic,version,mode,nr,nc,r,c,ml,tl,sl,dl,digest = base.HEADER.unpack_from(frame)
    if magic != base.MAGIC or version != base.VERSION or nr not in base.ROWS or nc not in base.COLS:
        raise base.CodecError('unsupported BCTX frame')
    n = r*c
    if not r or not c or n > base.MAX_WORDS:
        raise base.CodecError('invalid frame geometry')
    out = {'shape':(r,c),'mode':mode,'nr':nr,'nc':nc,'stride':stride,'source_sha256':digest.hex(),'frame_sha256':hashlib.sha256(frame).hexdigest(),'frame_bytes':len(frame)}
    if mode == 1:
        if any((ml,tl,sl,dl)) or len(frame) != base.HEADER.size + 2*n + 32:
            raise base.CodecError('invalid raw frame')
        raw = frame[base.HEADER.size:-32]
        if hashlib.sha256(raw).digest() != digest:raise base.CodecError('raw source mismatch')
        out['raw'] = np.frombuffer(raw,dtype=np.uint16).copy()
        return out
    if mode not in (0,128) or base.HEADER.size+ml+tl+sl+dl+32 != len(frame):
        raise base.CodecError('invalid coded frame')
    p=base.HEADER.size
    labels=base._inflate_exact(frame[p:p+ml],r+c);p+=ml
    rows=np.frombuffer(labels[:r],dtype=np.uint8).copy();columns=np.frombuffer(labels[r:],dtype=np.uint8).copy()
    if rows.max(initial=0)>=nr or columns.max(initial=0)>=nc:raise base.CodecError('invalid context labels')
    main_bytes=nr*nc*1024*4
    tables=base._inflate_exact(frame[p:p+tl],main_bytes+(1024*64*4 if mode==128 else 0));p+=tl
    sf=np.frombuffer(tables[:main_bytes],dtype='<u4').copy().reshape(nr*nc,1024)
    sy=frame[p:p+sl];p+=sl;rs=frame[p:p+dl]
    states,offsets,main=checkpoints(sy,sf,rows,columns,(r,c),nr,nc,stride)
    out.update(rows=rows,columns=columns,main_freq=sf,main_stream=np.frombuffer(sy,dtype=np.uint8).copy(),main_states=states,main_offsets=offsets,residual_stream=np.frombuffer(rs,dtype=np.uint8).copy())
    if mode==128:
        rf=np.frombuffer(tables[main_bytes:],dtype='<u4').copy().reshape(1024,64)
        states,offsets,residual=checkpoints(rs,rf,rows,columns,(r,c),nr,nc,stride,main)
        out.update(residual_freq=rf,residual_states=states,residual_offsets=offsets)
    else:
        residual=base._unpack6(base._lib(),rs,n)
    # Verify the new CPU index walk against the original raw-byte digest.
    words=(((main.astype(np.uint32)>>2)<<7)|((main.astype(np.uint32)&3)<<5)|(residual&31)|((residual.astype(np.uint32)>>5)<<15)).astype('<u2')
    if hashlib.sha256(words.tobytes()).digest()!=digest:raise base.CodecError('indexed source checksum mismatch')
    return out


class CudaContextTensor:
    def __init__(self, frame: bytes, *, stride: int = 1024, device='cuda:0'):
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
        indexed=index_frame(frame,stride)
        self.shape=indexed['shape'];self.n=math.prod(self.shape);self.stride=stride
        self.mode=indexed['mode'];self.nr=indexed['nr'];self.nc=indexed['nc']
        self.source_sha256=indexed['source_sha256'];self.frame_sha256=indexed['frame_sha256'];self.frame_bytes=indexed['frame_bytes']
        self.device=torch.device(device);self.buffers={}
        for name,array in indexed.items():
            if isinstance(array,np.ndarray):self.buffers[name]=torch.from_numpy(array).to(self.device)
        for name in ('main','residual'):
            key=name+'_freq'
            if key in indexed:
                f=indexed[key];cum=np.zeros((len(f),f.shape[1]+1),dtype=np.uint32)
                cum[:,1:]=np.cumsum(f,axis=1,dtype=np.uint64).astype(np.uint32)
                self.buffers[name+'_cum']=torch.from_numpy(cum).to(self.device)
        self.error=torch.zeros((),device=self.device,dtype=torch.int32)
        self.all_blocks=torch.zeros(1,device=self.device,dtype=torch.int64)
        self.resident_bytes=sum(t.numel()*t.element_size() for t in self.buffers.values())+self.error.numel()*self.error.element_size()+self.all_blocks.numel()*self.all_blocks.element_size()

    def _decode_blocks(self, blocks=None, *, check=True):
        import torch
        import triton
        try:from scripts import bitexact_context_gpu_kernels as kernels
        except ImportError:import bitexact_context_gpu_kernels as kernels
        direct=blocks is None
        if direct:
            count=(self.n+self.stride-1)//self.stride
            blocks=self.all_blocks
        else:
            blocks=blocks.to(device=self.device,dtype=torch.int64).contiguous().reshape(-1)
            count=blocks.numel()
        result=torch.empty(count*self.stride,device=self.device,dtype=torch.uint16)
        self.error.zero_();b=self.buffers
        rname='residual' if self.mode==128 else 'main'
        args=(b['main_stream'],b['main_states'],b['main_offsets'],b['main_freq'],b['main_cum'],
            b['residual_stream'],b[rname+'_states'],b[rname+'_offsets'],b[rname+'_freq'],b[rname+'_cum'],
            b['rows'],b['columns'],blocks,result,self.error,self.n,self.shape[1],self.nc,
            b['main_stream'].numel(),b['residual_stream'].numel(),count)
        kernels.decode_words[(triton.cdiv(count,64),)](*args,RAW6=self.mode==0,DIRECT=direct,
            STRIDE=self.stride,BLOCK=64,num_warps=2)
        if check and self.error.item()!=0:raise base.CodecError('CUDA decoder checkpoint mismatch')
        return result.view(torch.bfloat16)

    def decode(self, *, check=True):
        import torch
        if self.mode==1:return self.buffers['raw'].view(torch.bfloat16).reshape(self.shape)
        return self._decode_blocks(check=check)[:self.n].reshape(self.shape)

    def gather_rows(self, ids, *, check=True):
        import torch
        ids=ids.to(device=self.device,dtype=torch.int64)
        if ids.numel()==0:return torch.empty((*ids.shape,self.shape[1]),device=self.device,dtype=torch.bfloat16)
        if torch.any((ids<0)|(ids>=self.shape[0])).item():raise IndexError('embedding row outside tensor')
        if self.mode==1 or self.shape[1]%self.stride:
            return self.decode(check=check)[ids]
        per_row=self.shape[1]//self.stride
        blocks=(ids.reshape(-1,1)*per_row+torch.arange(per_row,device=self.device)).reshape(-1)
        return self._decode_blocks(blocks,check=check).reshape(*ids.shape,self.shape[1])
