"""Lossless PPCX encoding for BF16 word matrices."""
from __future__ import annotations
import ctypes, fcntl, hashlib, os, platform, struct, subprocess, sys, tempfile, threading, zlib
from pathlib import Path
import numpy as np
import shutil

MAGIC=b"PPCX"; VERSION=1
HEADER=struct.Struct("<4sBBHIIQQQ32s32s")
MAX_WORDS=100_000_000
# Canonical standard-normal Lloyd boundaries after 100 updates from exact
# centers -3 + 2*i/5. Rounded Q64 CDF values are generated at 80/120 decimal
# digits by qualification/predictive/generate_lloyd_constants.py. Integer ranks
# avoid libm drift across platforms, especially around the exact median 1/2.
LLOYD_CDF_Q64 = (
    145284247103925396,
    582212398777477136,
    1352191675752570914,
    2453172269933728313,
    3853228202891598545,
    5497963060014953111,
    7315835767424166417,
    9223372036854775808,
    11130908306285385199,
    12948781013694598505,
    14593515870817953071,
    15993571803775823303,
    17094552397956980702,
    17864531674932074480,
    18301459826605626220,
)
ROOT=Path(__file__).resolve().parent
CACHE_ROOT=Path(os.environ.get('BITEXACT_PREDICTIVE_CACHE_DIR',ROOT.parent/'.scratch'/'predictive-codec'))
BUILD=CACHE_ROOT/'build'
class CodecError(ValueError): pass

_compile_thread_lock=threading.Lock()

def _compile_shared(src,stem):
    source_hash=hashlib.sha256(src.read_bytes()).hexdigest()
    suffix='.dylib' if sys.platform=='darwin' else '.so'
    arch=platform.machine().lower().replace('/','_')
    out=BUILD/f'{stem}_{source_hash}_{sys.platform}_{arch}{suffix}'
    out.parent.mkdir(parents=True,exist_ok=True)
    with _compile_thread_lock:
        with out.with_suffix(out.suffix+'.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not out.exists():
                fd,temp_name=tempfile.mkstemp(prefix=f'.{out.name}.',suffix='.tmp',dir=out.parent)
                os.close(fd)
                try:
                    cxx=os.environ.get('CXX','c++')
                    p=subprocess.run([shutil.which(cxx) or cxx,'-O3','-std=c++17','-shared','-fPIC',str(src),'-o',temp_name],capture_output=True,close_fds=sys.platform!='darwin')
                    if p.returncode: raise RuntimeError(p.stderr.decode(errors='replace'))
                    if not Path(temp_name).is_file() or Path(temp_name).stat().st_size==0: raise RuntimeError('native compiler produced an empty library')
                    os.replace(temp_name,out)
                finally:
                    try: os.unlink(temp_name)
                    except FileNotFoundError: pass
    return out

def _lib():
    src=ROOT/'bitexact_predictive_ans.cpp'; out=_compile_shared(src,'libpredictive_ans')
    lib=ctypes.CDLL(str(out)); U16=ctypes.POINTER(ctypes.c_uint16); U32=ctypes.POINTER(ctypes.c_uint32); U8=ctypes.POINTER(ctypes.c_uint8)
    lib.p_encode.argtypes=[U16,U16,ctypes.c_uint64,ctypes.c_uint32,ctypes.c_uint32,ctypes.POINTER(U32),ctypes.POINTER(U8),ctypes.POINTER(ctypes.c_size_t)];lib.p_encode.restype=ctypes.c_int
    lib.p_decode.argtypes=[U8,ctypes.c_size_t,ctypes.c_uint64,U16,ctypes.c_uint32,ctypes.c_uint32,U32,U16];lib.p_decode.restype=ctypes.c_int
    lib.p_free.argtypes=[ctypes.c_void_p];return lib

def _arr(x):
    a=np.asarray(x)
    if a.ndim!=2 or a.dtype.kind not in 'ui' or a.dtype.itemsize!=2 or not all(a.shape) or a.size>MAX_WORDS: raise CodecError('expected nonempty equal-shaped uint16 matrices within limit')
    if a.dtype.kind=='i' and (np.any(a<0) or np.any(a>65535)): raise CodecError('word outside uint16')
    return np.ascontiguousarray(a,dtype='<u2')
def _float(a):
    w=a.astype(np.uint32); sign=(w>>15)!=0; mag=w&0x7fff
    # Decode BF16 finite words only; exponent 255 is excluded from fitting.
    e=(mag>>7)&255; m=mag&127
    z=np.ldexp(1.0+(m.astype(np.float64)/128.0),e.astype(np.int32)-127)
    z=np.where(e==0,np.ldexp(m.astype(np.float64)/128.0,-126),z)
    z=np.where(sign,-z,z); return np.where(e==255,np.nan,z)
def _ord(w):
    w=w.astype(np.uint16); return np.where((w&0x8000)!=0,np.bitwise_not(w),w^np.uint16(0x8000)).astype(np.uint16)
def _aligned_keys(x,flips):
    aligned=x.copy(); aligned[np.asarray(flips,dtype=bool)]^=np.uint16(0x8000)
    return _ord(aligned)
def _fit(y,x,row_profile='quartile',bin_profile='quantile'):
    if row_profile not in ('quartile','residual','tail'): raise CodecError('unknown row profile')
    if bin_profile not in ('quantile','lloyd'): raise CodecError('unknown bin profile')
    r,c=y.shape; yf=_float(y); xf=_float(x); flips=np.zeros(r,dtype=np.uint8); labels=np.zeros(r,dtype=np.uint8)
    rowmag=np.zeros(r); corr=np.zeros(r)
    for i in range(r):
        ok=np.isfinite(yf[i])&np.isfinite(xf[i]); a=yf[i,ok]; b=xf[i,ok]
        if len(a): rowmag[i]=np.mean(np.abs(a))
        if len(a)>1:
            aa=a-a.mean(); bb=b-b.mean(); den=np.sqrt(np.dot(aa,aa)*np.dot(bb,bb))
            co=float(np.dot(aa,bb)/den) if den else 0.0
            if np.isfinite(co): flips[i]=co<0; corr[i]=abs(co)
    def ranks(v):
        # Deterministic quartile labels; ties break by original row index.
        order=np.argsort(v,kind='stable'); q=np.empty(r,dtype=np.uint8); q[order]=np.minimum(3,np.arange(r)*4//r); return q
    if row_profile=='quartile':
        labels=(ranks(rowmag)*4+ranks(corr)).astype(np.uint8)
    elif row_profile=='residual':
        residual_scale=rowmag*np.sqrt(np.maximum(1-corr*corr,1e-8)); order=np.argsort(residual_scale,kind='stable'); labels=np.empty(r,dtype=np.uint8); labels[order]=np.minimum(15,np.arange(r)*16//r)
    else:
        labels=(ranks(rowmag)*4+np.searchsorted(np.array([0.5,0.75,0.9]),corr,side='right')).astype(np.uint8)
    keys=_aligned_keys(x,flips)
    vals=keys.ravel()[np.isfinite(xf).ravel()]
    thresholds=np.zeros(15,dtype='<u2')
    if vals.size:
        s=np.sort(vals,kind='stable')
        if bin_profile=='quantile':
            for j in range(1,16): thresholds[j-1]=s[min(len(s)-1,(j*len(s))//16)]
        else:
            for j,numerator in enumerate(LLOYD_CDF_Q64):
                rank=(numerator*len(s))>>64
                thresholds[j]=s[min(len(s)-1,rank)]
    return labels,flips,thresholds

def _packbits(v,bits):
    vals=np.asarray(v,dtype=np.uint8).ravel()
    if bits<1 or bits>8 or (vals.size and np.any(vals >= (1<<bits))):
        raise CodecError('value exceeds packed bit width')
    # Keep chunk boundaries byte-aligned by processing whole groups of eight
    # values. Each value remains MSB-first, matching the original wire format.
    chunk_values=(1<<20)//8*8
    chunks=[]
    shifts=np.arange(bits-1,-1,-1,dtype=np.uint8)
    for start in range(0,vals.size,chunk_values):
        part=vals[start:start+chunk_values]
        planes=((part[:,None]>>shifts)&1).astype(np.uint8,copy=False)
        chunks.append(np.packbits(planes.reshape(-1),bitorder='big'))
    return b''.join(c.tobytes() for c in chunks)
def _inflate(data,expected):
    try:
        d=zlib.decompressobj(); raw=d.decompress(data,expected+1)
        if len(raw)>expected or d.unconsumed_tail: raise CodecError('compressed section exceeds declared length')
        raw+=d.flush()
    except zlib.error as e: raise CodecError('invalid compressed section') from e
    if len(raw)!=expected or not d.eof or d.unused_data: raise CodecError('compressed section length mismatch')
    return raw
def _rans_encode(vals,ctx,nctx,vocab):
    lib=_lib(); vals=np.ascontiguousarray(vals,dtype='<u2');ctx=np.ascontiguousarray(ctx,dtype='<u2'); fp=ctypes.POINTER(ctypes.c_uint32)();sp=ctypes.POINTER(ctypes.c_uint8)();sz=ctypes.c_size_t()
    rc=lib.p_encode(vals.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),ctx.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),vals.size,nctx,vocab,ctypes.byref(fp),ctypes.byref(sp),ctypes.byref(sz))
    if rc: raise CodecError(f'rANS encode failed ({rc})')
    try: f=ctypes.string_at(fp,nctx*vocab*4); s=ctypes.string_at(sp,sz.value)
    finally: lib.p_free(fp);lib.p_free(sp)
    return zlib.compress(f,6),s

def encode(y,ref,row_profile='quartile',bin_profile='quantile'):
    a=_arr(y); x=_arr(ref)
    if a.shape!=x.shape: raise CodecError('shape mismatch')
    r,c=a.shape; labels,flips,th=_fit(a,x,row_profile,bin_profile); keys=_aligned_keys(x,flips)
    bins=np.searchsorted(th,keys,side='right').astype(np.uint16); main=((a>>15)&1)*1024+((a>>7)&255)*4+((a&127)>>5); residual=a&31
    mc=(labels[:,None].astype(np.uint16)*16+bins).ravel(); mf,ms=_rans_encode(main.ravel(),mc,256,2048)
    # meta stores labels, row flips, and decoder thresholds. Tables use exact integer coder normalization.
    meta=zlib.compress(labels.tobytes()+flips.tobytes()+th.tobytes(),6)
    base=ms
    # Compare complete residual representations including table and section overhead.
    raw5=_packbits(residual.ravel(),5)
    rf,rs=_rans_encode(residual.ravel(),main.ravel(),2048,32)
    cond=struct.pack('<I',len(rf))+rf+rs
    forms=[(0,b'',raw5),(1,b'',cond),(2,b'',a.tobytes())]
    target=hashlib.sha256(a.tobytes()).digest(); refhash=hashlib.sha256(x.tobytes()).digest()
    frames=[]
    for mode,rt,rd in forms:
        if mode==2: ml=tl=sl=0; payload=b''
        else: ml,tl,sl=len(meta),len(mf),len(base); payload=meta+mf+base
        body=HEADER.pack(MAGIC,VERSION,mode,0,r,c,ml,tl,sl,refhash,target)+payload+rd
        frames.append(body+hashlib.sha256(body).digest())
    return min(frames,key=len)

def decode(frame,ref):
    if not isinstance(frame,(bytes,bytearray,memoryview)) or len(frame)<HEADER.size+32: raise CodecError('short frame')
    b=bytes(frame)
    if hashlib.sha256(b[:-32]).digest()!=b[-32:]: raise CodecError('frame checksum mismatch')
    magic,ver,mode,flags,r,c,ml,tl,sl,rh,th=HEADER.unpack_from(b)
    if magic!=MAGIC or ver!=VERSION or mode not in (0,1,2) or flags or not r or not c or r*c>MAX_WORDS: raise CodecError('invalid header')
    x=_arr(ref)
    if x.shape!=(r,c) or hashlib.sha256(x.tobytes()).digest()!=rh: raise CodecError('reference mismatch')
    end=HEADER.size+ml+tl+sl
    if end>len(b)-32: raise CodecError('section lengths exceed frame')
    p=HEADER.size; meta=b[p:p+ml];p+=ml; mtab=b[p:p+tl];p+=tl; stream=b[p:p+sl];p+=sl; data=b[p:-32]
    if p>len(b)-32: raise CodecError('truncated sections')
    if mode==2:
        m=b''; fmain=b''
    else:
        m=_inflate(meta,r*2+30)
        fmain=_inflate(mtab,256*2048*4)
    if mode!=2:
        if len(m)!=r*2+30: raise CodecError('metadata length mismatch')
        labels=np.frombuffer(m[:r],dtype=np.uint8).copy(); flips=np.frombuffer(m[r:2*r],dtype=np.uint8).copy(); thresholds=np.frombuffer(m[2*r:],dtype='<u2').copy()
        if np.any(labels>=16) or np.any(flips>1): raise CodecError('metadata values invalid')
        if np.any(thresholds[1:]<thresholds[:-1]): raise CodecError('metadata thresholds are not monotone')
        keys=_aligned_keys(x,flips); bins=np.searchsorted(thresholds,keys,side='right').astype(np.uint16);ctx=(labels[:,None].astype(np.uint16)*16+bins).ravel()
    n=r*c
    if mode==2:
        if ml or tl or sl or len(meta) or len(mtab) or len(stream) or len(data)!=2*n: raise CodecError('raw-word length mismatch')
        out=np.frombuffer(data,dtype='<u2').copy().reshape(r,c)
    else:
        if len(fmain)!=256*2048*4 or len(stream)<4: raise CodecError('main table/stream length mismatch')
        lib=_lib(); mf=np.frombuffer(fmain,dtype='<u4').copy(); mv=np.empty(n,dtype='<u2'); sb=ctypes.create_string_buffer(stream)
        rc=lib.p_decode(ctypes.cast(sb,ctypes.POINTER(ctypes.c_uint8)),len(stream),n,ctx.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),256,2048,mf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),mv.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)))
        if rc: raise CodecError(f'main rANS decode failed ({rc})')
        if mode==0:
            if len(data)!=(n*5+7)//8: raise CodecError('raw5 length mismatch')
            bits=np.unpackbits(np.frombuffer(data,dtype=np.uint8),bitorder='big')
            if n*5%8 and np.any(bits[n*5:]): raise CodecError('raw5 padding invalid')
            residual=np.zeros(n,dtype=np.uint8)
            for j in range(5): residual=(residual<<1)|bits[j::5][:n]
        else:
            if len(data)<8: raise CodecError('conditional residual missing')
            fl=struct.unpack_from('<I',data)[0]
            if fl>len(data)-4: raise CodecError('conditional table length invalid')
            fb=_inflate(data[4:4+fl],2048*32*4)
            if len(fb)!=2048*32*4: raise CodecError('residual table length mismatch')
            rf=np.frombuffer(fb,dtype='<u4').copy(); rs=data[4+fl:]
            if len(rs)<4: raise CodecError('residual stream truncated')
            rr=np.empty(n,dtype='<u2'); rsb=ctypes.create_string_buffer(rs)
            status=lib.p_decode(ctypes.cast(rsb,ctypes.POINTER(ctypes.c_uint8)),len(rs),n,mv.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),2048,32,rf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),rr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)))
            if status: raise CodecError(f'residual rANS decode failed ({status})')
            residual=rr.astype(np.uint8)
        words=((mv.astype(np.uint32)&1024)<<5)|(((mv.astype(np.uint32)>>2)&255)<<7)|((mv.astype(np.uint32)&3)<<5)|residual.astype(np.uint32)
        out=words.astype('<u2').reshape(r,c)
    if hashlib.sha256(out.tobytes()).digest()!=th: raise CodecError('target checksum mismatch')
    return out
