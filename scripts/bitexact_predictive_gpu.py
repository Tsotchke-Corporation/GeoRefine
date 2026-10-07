"""CUDA decoder for PPCX frames."""
from __future__ import annotations
import ctypes, fcntl, hashlib, math, os, platform, struct, subprocess, sys, tempfile, threading
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
CACHE_ROOT=Path(os.environ.get('BITEXACT_PREDICTIVE_CACHE_DIR',HERE.parent/'.scratch'/'predictive-codec'))
import bitexact_predictive_codec as codec

_compile_thread_lock=threading.Lock()

def _compile_shared(src,stem):
    source_hash=hashlib.sha256(src.read_bytes()).hexdigest()
    suffix='.dylib' if sys.platform=='darwin' else '.so'
    arch=platform.machine().lower().replace('/','_')
    out=CACHE_ROOT/'build'/f'{stem}_{source_hash}_{sys.platform}_{arch}{suffix}'
    out.parent.mkdir(parents=True,exist_ok=True)
    with _compile_thread_lock:
        with out.with_suffix(out.suffix+'.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not out.exists():
                fd,temp_name=tempfile.mkstemp(prefix=f'.{out.name}.',suffix='.tmp',dir=out.parent)
                os.close(fd)
                try:
                    p=subprocess.run([os.environ.get('CXX','c++'),'-O3','-std=c++17','-shared','-fPIC',str(src),'-o',temp_name],capture_output=True)
                    if p.returncode: raise RuntimeError(p.stderr.decode(errors='replace'))
                    if not Path(temp_name).is_file() or Path(temp_name).stat().st_size==0: raise RuntimeError('native compiler produced an empty library')
                    os.replace(temp_name,out)
                finally:
                    try: os.unlink(temp_name)
                    except FileNotFoundError: pass
    return out


def _index_lib():
    src=HERE/'bitexact_predictive_gpu_index.cpp'; out=_compile_shared(src,'libppcx_index')
    lib=ctypes.CDLL(str(out)); U8=ctypes.POINTER(ctypes.c_uint8); U16=ctypes.POINTER(ctypes.c_uint16); U32=ctypes.POINTER(ctypes.c_uint32)
    lib.ppcx_checkpoint.argtypes=[U8,ctypes.c_size_t,ctypes.c_uint64,U16,ctypes.c_uint32,ctypes.c_uint32,U32,ctypes.c_uint32,U32,U32,U16];lib.ppcx_checkpoint.restype=ctypes.c_int
    return lib

def _ptr(a,t): return a.ctypes.data_as(ctypes.POINTER(t))
def _checkpoints(stream,ctx,freq,vocab,stride):
    ctx=np.ascontiguousarray(ctx,dtype='<u2'); freq=np.ascontiguousarray(freq,dtype='<u4'); n=len(ctx)
    states=np.empty((n+stride-1)//stride,dtype='<u4'); offsets=np.empty_like(states); symbols=np.empty(n,dtype='<u2'); data=np.frombuffer(stream,dtype=np.uint8)
    rc=_index_lib().ppcx_checkpoint(_ptr(data,ctypes.c_uint8),len(data),n,_ptr(ctx,ctypes.c_uint16),len(freq),vocab,_ptr(freq,ctypes.c_uint32),stride,_ptr(states,ctypes.c_uint32),_ptr(offsets,ctypes.c_uint32),_ptr(symbols,ctypes.c_uint16))
    if rc: raise codec.CodecError(f'PPCX checkpoint validation failed ({rc})')
    return states,offsets,symbols

def _inflate(data,expected):
    return codec._inflate(data,expected)

def _read_frame(frame,reference,stride):
    if not isinstance(frame,(bytes,bytearray,memoryview)): raise codec.CodecError('frame must be bytes')
    b=bytes(frame); H=codec.HEADER
    if len(b)<H.size+32 or hashlib.sha256(b[:-32]).digest()!=b[-32:]: raise codec.CodecError('frame checksum mismatch')
    magic,ver,mode,flags,r,c,ml,tl,sl,rh,th=H.unpack_from(b)
    if magic!=codec.MAGIC or ver!=codec.VERSION or mode not in (0,1,2) or flags or not r or not c or r*c>codec.MAX_WORDS: raise codec.CodecError('invalid header')
    ref=codec._arr(reference)
    if ref.shape!=(r,c) or hashlib.sha256(ref.tobytes()).digest()!=rh: raise codec.CodecError('reference mismatch')
    end=H.size+ml+tl+sl
    if end>len(b)-32: raise codec.CodecError('section lengths exceed frame')
    p=H.size; meta=b[p:p+ml];p+=ml; table=b[p:p+tl];p+=tl; stream=b[p:p+sl];p+=sl; data=b[p:-32]
    result=dict(shape=(r,c),mode=mode,frame_bytes=len(b),frame_sha256=hashlib.sha256(b).hexdigest(),source_sha256=th.hex(),target_sha256=th.hex(),reference_sha256=rh.hex())
    if mode==2:
        if ml or tl or sl or len(data)!=2*r*c: raise codec.CodecError('raw-word length mismatch')
        target=np.frombuffer(data,dtype='<u2').copy().reshape(r,c)
        if hashlib.sha256(target.tobytes()).digest()!=th: raise codec.CodecError('target checksum mismatch')
        result['raw']=target
        return result
    m=_inflate(meta,r*2+30); fmain=_inflate(table,256*2048*4)
    if len(m)!=r*2+30 or len(fmain)!=256*2048*4: raise codec.CodecError('table length mismatch')
    labels=np.frombuffer(m[:r],dtype=np.uint8).copy(); flips=np.frombuffer(m[r:2*r],dtype=np.uint8).copy(); thresholds=np.frombuffer(m[2*r:],dtype='<u2').copy()
    if np.any(labels>=16) or np.any(flips>1): raise codec.CodecError('invalid metadata values')
    if np.any(thresholds[1:]<thresholds[:-1]): raise codec.CodecError('thresholds are not monotone')
    freq=np.frombuffer(fmain,dtype='<u4').copy().reshape(256,2048)
    # Exact predictor alignment: flip sign bit in encoded uint16 before ordinal mapping.
    aligned=ref.copy(); aligned[flips.astype(bool)]^=np.uint16(0x8000); keys=codec._ord(aligned)
    bins=np.searchsorted(thresholds,keys,side='right').astype(np.uint16)
    mainctx=(labels[:,None].astype(np.uint16)*16+bins).ravel()
    ms,mo,main=_checkpoints(stream,mainctx,freq,2048,stride)
    result.update(labels=labels,flips=flips,thresholds=thresholds,main_freq=freq,main_stream=np.frombuffer(stream,dtype=np.uint8).copy(),main_states=ms,main_offsets=mo)
    if mode==0:
        if len(data)!=(r*c*5+7)//8: raise codec.CodecError('raw5 length mismatch')
        bits=np.unpackbits(np.frombuffer(data,dtype=np.uint8),bitorder='big')
        if r*c*5%8 and np.any(bits[r*c*5:]): raise codec.CodecError('raw5 padding invalid')
        resid=np.zeros(r*c,dtype=np.uint8)
        for j in range(5): resid=(resid<<1)|bits[j::5][:r*c]
        result.update(raw5=np.frombuffer(data,dtype=np.uint8).copy(),residual_symbols=resid)
    else:
        if len(data)<8: raise codec.CodecError('conditional residual missing')
        fl=struct.unpack_from('<I',data)[0]
        if fl>len(data)-4: raise codec.CodecError('conditional table length invalid')
        fb=_inflate(data[4:4+fl],2048*32*4)
        if len(fb)!=2048*32*4: raise codec.CodecError('residual table length mismatch')
        rf=np.frombuffer(fb,dtype='<u4').copy().reshape(2048,32); rs=data[4+fl:]
        rc,ro,resid=_checkpoints(rs,main,rf,32,stride)
        result.update(residual_freq=rf,residual_stream=np.frombuffer(rs,dtype=np.uint8).copy(),residual_states=rc,residual_offsets=ro)
    target=((main.astype(np.uint32)&1024)<<5)|(((main.astype(np.uint32)>>2)&255)<<7)|((main.astype(np.uint32)&3)<<5)|resid.astype(np.uint32)
    target=target.astype('<u2').reshape(r,c)
    if hashlib.sha256(target.tobytes()).digest()!=th: raise codec.CodecError('target checksum mismatch')
    return result

class CudaPredictiveTensor:
    def __init__(self,frame,reference_uint16_numpy,stride=1024,device='cuda:0'):
        if not isinstance(stride,int) or not 1<=stride<=65536: raise codec.CodecError('invalid checkpoint stride')
        ix=_read_frame(frame,reference_uint16_numpy,stride)
        # CPU independently validates source/target against the frozen reference decoder.
        target=codec.decode(frame,reference_uint16_numpy)
        if hashlib.sha256(target.tobytes()).hexdigest()!=ix['target_sha256']: raise codec.CodecError('target validation failed')
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
        self.shape=ix['shape']; self.n=math.prod(self.shape); self.stride=stride; self.mode=ix['mode']; self.device=torch.device(device)
        self.source_sha256=ix['source_sha256']; self.target_sha256=ix['target_sha256']; self.reference_sha256=ix['reference_sha256']; self.frame_sha256=ix['frame_sha256']; self.frame_bytes=ix['frame_bytes']; self.buffers={}
        for k,v in ix.items():
            if isinstance(v,np.ndarray) and k not in ('residual_symbols',):
                self.buffers[k]=torch.from_numpy(np.ascontiguousarray(v)).to(self.device)
        for name in ('main','residual'):
            key=name+'_freq'
            if key in ix:
                f=ix[key]; cum=np.zeros((len(f),f.shape[1]+1),dtype=np.uint32); cum[:,1:]=np.cumsum(f,axis=1,dtype=np.uint64).astype(np.uint32)
                self.buffers[name+'_cum']=torch.from_numpy(cum).to(self.device)
        self.error=torch.zeros((),device=self.device,dtype=torch.int32)
        self.resident_bytes=sum(t.numel()*t.element_size() for t in self.buffers.values())+4

    def decode(self,reference_bf16_cuda,check=True):
        import torch, triton
        if self.mode==2: return self.buffers['raw'].view(torch.bfloat16).reshape(self.shape)
        try: from bitexact_predictive_gpu_kernels import decode_words
        except ImportError: from .predictive_gpu_kernels import decode_words
        ref=reference_bf16_cuda.to(device=self.device).contiguous()
        if tuple(ref.shape)!=self.shape or ref.dtype!=torch.bfloat16: raise ValueError('reference CUDA tensor shape/dtype mismatch')
        words=ref.view(torch.uint16).reshape(-1); n=self.n; stride=self.stride; count=(n+stride-1)//stride
        out=torch.empty(count*stride,device=self.device,dtype=torch.uint16); self.error.zero_(); b=self.buffers
        if self.mode==0:
            rstream=b['raw5']; rstates=roffsets=rfreq=rcum=b['main_states']
        else:
            rstream=b['residual_stream']; rstates=b['residual_states']; roffsets=b['residual_offsets']; rfreq=b['residual_freq']; rcum=b['residual_cum']
        decode_words[(triton.cdiv(count,64),)](b['main_stream'],b['main_states'],b['main_offsets'],b['main_freq'],b['main_cum'],rstream,rstates,roffsets,rfreq,rcum,b['labels'],b['flips'],b['thresholds'],words,out,self.error,n,self.shape[1],b['main_stream'].numel(),rstream.numel(),self.mode,stride,BLOCK=64,num_warps=2)
        if check and self.error.item(): raise codec.CodecError('CUDA PPCX checkpoint/context mismatch')
        return out[:n].view(torch.bfloat16).reshape(self.shape)
