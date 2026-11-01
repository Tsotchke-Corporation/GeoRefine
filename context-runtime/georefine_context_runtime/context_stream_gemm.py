"""Explicit BCS3 experimental raw6 consumer; no loader selects this module.

Bounded N64,K512/1024,S1 fixtures, optionally independently replicated resident
panels. No BCTX mode128/archive-ratio bridge, fullmodel, or user-count claim.
CUDA is loaded only on explicit extension()/gemm_cuda() calls, never import.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import numpy as np
from . import context_stream_packet as packet

PacketError=packet.PacketError
FORMAT='BCS3-experimental-persistent-BIGEMM-S1-tile64-v1'
VIEW_SLOT_BYTES=128
encode=packet.encode
decode_cpu=packet.decode_cpu
_parse=packet._parse


def _plan(blob,panels,m):
    if type(panels) is not int or not 1<=panels<=1024:
        raise PacketError('panels must be an integer in [1,1024]')
    if type(m) is not int or m not in (1,16,64,128):
        raise PacketError('M must be 1/16/64/128')
    blob,info=packet._parse(blob)
    k=info['cols'];blocks=(m+63)//64
    gpu=dict(serialized_blob=panels*len(blob),descriptors=panels*4*9*8,
             output=m*panels*64*2,lane_status=panels*blocks*4*32*4,
             decoded_weights=0,derived_tables=0,cursor_snapshots=0)
    wx=(64+(64 if m>=64 else 16))*136*2
    ledger=dict(format=FORMAT,wire_format=packet.FORMAT,experimental_raw6=True,
        scope='bounded replicated fixture; no fullmodel/archive-ratio/user qualification',
        panels=panels,replicated_fixture=True,N=64*panels,K=k,M=m,S=1,bias=None,activation_tile=64,
        data_shape=[panels,len(blob)],descriptor_shape=[panels,4,9],
        output_shape=[m,64*panels],status_shape=[panels,blocks,4,32],
        grid=[blocks,panels,1],grid_ctas=blocks*panels,threads_per_cta=256,
        gpu_allocation_bytes=gpu,gpu_owned_bytes=sum(gpu.values()),activation_bytes=m*k*2,
        gpu_resident_bytes_including_x=sum(gpu.values())+m*k*2,
        gpu_weight_and_status_bytes=gpu['serialized_blob']+gpu['descriptors']+gpu['lane_status'],
        dense_control_weight_bytes=panels*64*k*2,
        dense_control_total_bytes=panels*64*k*2+m*k*2+gpu['output'],
        shared_weight_slab_bytes=64*136*2,shared_weight_row_stride_words=136,
        shared_weight_activation_bytes_per_cta=wx,shared_packet_view_bytes_per_cta=4*VIEW_SLOT_BYTES,
        shared_dynamic_bytes_per_cta=wx+4*VIEW_SLOT_BYTES,shared_static_bytes_per_cta=4,
        source_digest=info['source_digest'],context_identity=info['context_identity'])
    return blob,info,ledger


def inspect(blob):
    """BCS3 wire accounting plus the explicit consumer contract."""
    return dict(packet.inspect(blob),consumer=FORMAT,activation_tile=64,allowed_m=[1,16,64,128],
                shared_weight_slab_bytes=64*136*2,shared_weight_row_stride_words=136)


def inspect_panels(blob,*,panels,m):
    """Bill explicit resident tensors and reserved shared bytes; no GPU work.

    Input x is shared across panels. Driver/allocator peaks and compiler local
    memory require separate measurement; final status has no K-chunk dimension.
    """
    return _plan(blob,panels,m)[2]


def _arrays(blob,info,panels):
    return (np.tile(np.frombuffer(blob,dtype=np.uint8),(panels,1)),
            np.tile(np.array([[q[k] for k in packet.DESCRIPTOR_KEYS] for q in info['packets']],
                             dtype=np.int64),(panels,1,1)))


def source_paths():
    base=Path(__file__).resolve().parent;bi=base.parent/'glc_serve'/'bi_csrc'
    tbe=base.parent/'glc_serve'/'miv_tbe_csrc'
    return tuple([base/('context_stream_gemm'+s) for s in ('.py','.cpp','.cu')]+
        [base/'context_stream_packet.py',base/'context_stream_packet.cuh',
         base/'context_tiled_packet.py',base/'context_tiled_packet.cuh',base/'bitexact_context_codec.py',
         bi/'bi_gemm.cuh',bi/'bi_api.h',tbe/'miv_tbe_kernels.cuh',tbe/'miv_tbe.h'])


def kernel_build_plan():
    from glc_serve import kernel_cache
    arch=os.environ.get('TORCH_CUDA_ARCH_LIST')
    if not arch:
        import torch
        major,minor=torch.cuda.get_device_capability();arch=f'{major}.{minor}'
        os.environ['TORCH_CUDA_ARCH_LIST']=arch
    base=Path(__file__).resolve().parent
    name='context_stream_gemm_v1'
    includes=tuple(str(base.parent/'glc_serve'/s) for s in ('bi_csrc','miv_tbe_csrc'))
    sources=tuple(str(base/('context_stream_gemm'+s)) for s in ('.cpp','.cu'))
    flags=('-O3','-DGEOR_REFINED_CONTEXT_ONLY=1','--cuda--','-O3','--fmad=true',
           '-lineinfo','-std=c++17','-DGEOR_REFINED_CONTEXT_ONLY=1')
    spec=kernel_cache.ExtensionSpec(name=name,sources=sources,include_dirs=includes,
        inline_sources=tuple(p.read_text() for p in source_paths() if p.suffix=='.py'),flags=flags,arch=arch,
        extra=(('consumer',FORMAT),))
    return name,kernel_cache.extension_dir(spec),spec


def extension():
    from glc_serve import kernel_cache
    from torch.utils.cpp_extension import load
    name,directory,spec=kernel_build_plan()
    return kernel_cache.build(name,directory,lambda:load(name=name,sources=list(spec.sources),
        extra_include_paths=list(spec.include_dirs),build_directory=str(directory),
        extra_cflags=['-O3','-DGEOR_REFINED_CONTEXT_ONLY=1'],
        extra_cuda_cflags=['-O3','--fmad=true','-lineinfo','-std=c++17','-DGEOR_REFINED_CONTEXT_ONLY=1'],
        verbose=False),spec=spec)


def gemm_cuda(blob,x,*,device,panels=1):
    """Explicit CUDA experiment returning y[M,P*64] and final per-lane status.

    Host checks BCS3 framing/hashes and transfers independent resident replicas;
    CUDA performs all decoding/GEMM. Status is synchronized before exposing y.
    Native y is unspecified on failure; code32 marks a lane aborted because a
    peer failed. Repeated calls include parsing, transfers and allocation;
    resident native latency must be reported separately. No CPU decode fallback.
    """
    blob,info,_=_plan(blob,panels,1)
    import torch
    dev=torch.device(device)
    if dev.type!='cuda' or dev.index is None:
        raise PacketError('explicit CUDA device index required, e.g. cuda:0')
    if (not isinstance(x,torch.Tensor) or x.device!=dev or x.dtype!=torch.bfloat16 or x.ndim!=2 or
            x.shape[0] not in (1,16,64,128) or x.shape[1]!=info['cols'] or not x.is_contiguous() or x.data_ptr()%16):
        raise PacketError('aligned contiguous CUDA bf16 x[M,K] with M=1/16/64/128 required')
    _,_,ledger=_plan(blob,panels,int(x.shape[0]))
    with torch.cuda.device(dev):
        ext=extension()
        data_host,desc_host=_arrays(blob,info,panels)
        data=torch.from_numpy(data_host).to(dev);desc=torch.from_numpy(desc_host).to(dev)
        y,status=ext.gemm(data,desc,x,info['row_classes'],info['col_classes'],
                          packet.HEADER.size,info['table_start'],info['table_bytes'])
        codes=status.cpu().numpy().reshape(-1);bad=np.flatnonzero(codes)
        if bad.size:
            sample=[(int(i),int(codes[i])) for i in bad[:8]]
            raise PacketError(f'CUDA stream status failure: count={bad.size}, first={sample}')
        from glc_serve import kernel_cache
        name,directory,spec=kernel_build_plan();binary=Path(ext.__file__).resolve()
        root=Path(__file__).resolve().parents[1]
        provenance=dict(ledger,decoded_on='cuda-shared-memory',output_digest_verified=False,
            source_bindings={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths()},
            extension=name,extension_path=str(binary),extension_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            build_key=kernel_cache.spec_digest(spec),build_directory=str(directory),device=str(dev),
            torch_version=torch.__version__,cuda_version=torch.version.cuda,device_name=torch.cuda.get_device_name(dev),
            reduction='existing bi::mma16816; increasing CH128/kk16; S1; bf16_rn epilogue',
            host_staging_bytes=data_host.nbytes+desc_host.nbytes)
        return dict(y=y,status=status,provenance=provenance)
