"""Experimental BCT2 decode fused into fixed-order BI-GEMM, explicit opt-in only.

N=64, K=512/1024, M=1/16/64/128, S=1, no bias. Independent 16x128 BCT2
packets reconstruct one 64x128 shared-memory slab, then immediately feed the
existing BI MMA helpers. No full decoded weight allocation or CPU fallback.
This correctness milestone makes no throughput/compression advantage claim.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from . import context_tiled_packet as packet

FORMAT = 'BCT2-fused-BIGEMM-N64-K512or1024-S1-v1'
PacketError = packet.PacketError


def encode(words, *, row_classes=16, col_classes=4):
    """Encode a synthetic bounded [64,K] uint16 fixture as ordinary BCT2."""
    a = np.asarray(words)
    if a.ndim != 2 or a.shape not in ((64,512),(64,1024)):
        raise PacketError('fused prototype requires N=64 and K=512 or 1024')
    return packet.encode(a,rows_per_packet=16,cols_per_packet=128,
                         row_classes=row_classes,col_classes=col_classes)


def _parse(blob):
    blob, info = packet._parse(blob)
    if (info['rows'],info['cols']) not in ((64,512),(64,1024)):
        raise PacketError('fused prototype requires N=64 and K=512 or 1024')
    packets = sorted(info['packets'],key=lambda q:(q['col0'],q['row0']))
    expected = [(r,c,16,128) for c in range(0,info['cols'],128) for r in range(0,64,16)]
    actual = [(q['row0'],q['col0'],q['packet_rows'],q['packet_cols']) for q in packets]
    if actual != expected:
        raise PacketError('fused prototype requires independent 16x128 BCT2 packets')
    info['packets'] = packets
    return blob, info


def inspect(blob):
    """Return complete BCT2 byte accounting plus the explicit fused contract."""
    blob, _ = _parse(blob)
    return dict(**packet.inspect(blob),consumer=FORMAT,split_k=1,bias=None,
                decoded_weight_hbm_bytes=0,shared_weight_slab_bytes=64*136*2,
                shared_weight_row_stride_words=136,
                activation_tile=64,allowed_m=[1,16,64,128])


def decode_cpu(blob):
    """Fixture validation only; gemm_cuda never calls this CPU decoder."""
    blob, _ = _parse(blob)
    return packet.decode_cpu(blob)


def source_paths():
    """Exact local source closure consumed by this extension and packet parser."""
    base = Path(__file__).resolve().parent
    bi = base.parent/'glc_serve'/'bi_csrc'
    tbe = base.parent/'glc_serve'/'miv_tbe_csrc'
    return tuple([base/('context_fused_gemm'+s) for s in ('.py','.cpp','.cu','.cuh')]+
                 [base/'context_cdf_lookup.py',base/'context_tiled_packet.py',base/'context_tiled_packet.cuh',
                  base/'bitexact_context_codec.py',bi/'bi_gemm.cuh',bi/'bi_api.h',
                  tbe/'miv_tbe_kernels.cuh',tbe/'miv_tbe.h'])


def kernel_build_plan():
    from glc_serve import kernel_cache
    arch = os.environ.get('TORCH_CUDA_ARCH_LIST')
    if not arch:
        import torch
        major, minor = torch.cuda.get_device_capability()
        arch = f'{major}.{minor}'
        os.environ['TORCH_CUDA_ARCH_LIST'] = arch
    base = Path(__file__).resolve().parent
    includes = tuple(str(base.parent/'glc_serve'/s) for s in ('bi_csrc','miv_tbe_csrc'))
    name = 'context_fused_gemm_v1'
    sources = tuple(str(base/('context_fused_gemm'+s)) for s in ('.cpp','.cu'))
    flags = ('-O3','-DGEOR_REFINED_CONTEXT_ONLY=1','--cuda--','-O3','--fmad=true',
             '-lineinfo','-std=c++17','-DGEOR_REFINED_CONTEXT_ONLY=1')
    spec = kernel_cache.ExtensionSpec(name=name,sources=sources,include_dirs=includes,
        inline_sources=tuple(p.read_text() for p in source_paths() if p.suffix=='.py'),
        flags=flags,arch=arch,extra=(('consumer',FORMAT),))
    return name,kernel_cache.extension_dir(spec),spec


def extension():
    from glc_serve import kernel_cache
    from torch.utils.cpp_extension import load
    name,directory,spec = kernel_build_plan()
    return kernel_cache.build(name,directory,lambda:load(name=name,sources=list(spec.sources),
        extra_include_paths=list(spec.include_dirs),build_directory=str(directory),
        extra_cflags=['-O3','-DGEOR_REFINED_CONTEXT_ONLY=1'],
        extra_cuda_cflags=['-O3','--fmad=true','-lineinfo','-std=c++17','-DGEOR_REFINED_CONTEXT_ONLY=1'],
        verbose=False),spec=spec)


def _build_cdf_lookup(blob, info):
    """Derive immutable search arrays from the exact validated packet CDF."""
    nctx = info['row_classes'] * info['col_classes']
    if info['table_bytes'] == 0:
        return None
    from .context_cdf_lookup import build_cdf_lookup
    start, size = info['table_start'], info['table_bytes']
    raw = blob[start:start+size]
    exact_digest = hashlib.sha256(raw).digest()
    lookup = build_cdf_lookup(raw, nctx, expected_sha256=exact_digest)
    if lookup.source_sha256 != exact_digest.hex():
        raise PacketError('derived CDF lookup digest does not match exact packet CDF')
    return lookup


def _lookup_arrays(lookup, nctx):
    if lookup is None:
        return (np.empty((0,256),dtype=np.uint32),np.empty((0,),dtype=np.uint64),
                np.empty((0,),dtype=np.uint32))
    if len(lookup.offsets) != nctx + 1:
        raise PacketError('derived CDF lookup context geometry mismatch')
    return lookup.coarse.copy(),lookup.records.copy(),lookup.offsets.copy()


def _lookup_bytes(lookup):
    return 0 if lookup is None else int(lookup.resident_bytes)


def gemm_cuda(blob, x, *, device, optimized=False, cdf_lookup=False):
    """Return CUDA bf16 y[M,64], per-CTA/tile/lane status, and source/DSO binding.

    optimized=True selects the fixed16x128 native decoder specialization.
    It adds no tables/allocations and does not modify the BCT2 wire bytes.
    x must already be contiguous bf16 on the explicit device. Framing/hashes are
    checked on the host; integer entropy decoding and GEMM execute only in CUDA.
    Status is synchronized and checked before exposing y. Repeated calls include
    host validation, blob transfer, and output allocation; report that separately
    from resident native kernel latency. This does not certify output digests.
    Native status code 32 means a slab was skipped after an earlier failure;
    native y is unspecified on any failure and must be discarded.
    """
    if type(optimized) is not bool:
        raise PacketError("optimized must be a bool")
    if type(cdf_lookup) is not bool:
        raise PacketError("cdf_lookup must be a bool")
    if cdf_lookup and not optimized:
        raise PacketError("cdf_lookup requires optimized=True")
    blob,info = _parse(blob)
    import torch
    dev = torch.device(device)
    if dev.type!='cuda' or dev.index is None:
        raise PacketError('explicit CUDA device index required, e.g. cuda:0')
    if (not isinstance(x,torch.Tensor) or x.device!=dev or x.dtype!=torch.bfloat16 or
            x.ndim!=2 or x.shape[0] not in (1,16,64,128) or x.shape[1]!=info['cols'] or
            not x.is_contiguous() or x.data_ptr()%16):
        raise PacketError('aligned contiguous CUDA bf16 x[M,K] with M=1/16/64/128 required')
    lookup = _build_cdf_lookup(blob,info) if cdf_lookup else None
    with torch.cuda.device(dev):
        ext = extension()
        data = torch.from_numpy(np.frombuffer(blob,dtype=np.uint8).copy()).to(dev)
        desc = torch.tensor([[q[k] for k in packet.DESCRIPTOR_KEYS] for q in info['packets']],
                            dtype=torch.int64,device=dev)
        host_lookup_bytes = 0
        if cdf_lookup:
            coarse_host,records_host,offsets_host=_lookup_arrays(lookup,info['row_classes']*info['col_classes'])
            host_lookup_bytes=coarse_host.nbytes+records_host.nbytes+offsets_host.nbytes
            coarse=torch.from_numpy(coarse_host).to(dev)
            records=torch.from_numpy(records_host).to(dev)
            offsets=torch.from_numpy(offsets_host).to(dev)
            y,status=ext.gemm_opt_lut(data,desc,x,info['row_classes'],info['col_classes'],
                packet.HEADER.size,info['table_start'],info['table_bytes'],coarse,records,offsets)
        else:
            native_gemm = ext.gemm_opt if optimized else ext.gemm
            y,status = native_gemm(data,desc,x,info['row_classes'],info['col_classes'],
                               packet.HEADER.size,info['table_start'],info['table_bytes'])
        codes = status.cpu().reshape(-1).tolist()
        if any(codes):
            raise PacketError(f'CUDA fused packet status failure: {codes}')
        from glc_serve import kernel_cache
        name,directory,spec = kernel_build_plan()
        root = Path(__file__).resolve().parents[1]
        bindings = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths()}
        binary = Path(ext.__file__).resolve()
        m = int(x.shape[0])
        provenance = dict(format=FORMAT,wire_format=packet.FORMAT,source_bindings=bindings,
            decoder="fixed16x128-v1" if optimized else "generic-v1",optimized=optimized,
            extension=name,extension_path=str(binary),extension_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            build_key=kernel_cache.spec_digest(spec),build_directory=str(directory),device=str(dev),
            torch_version=torch.__version__,cuda_version=torch.version.cuda,device_name=torch.cuda.get_device_name(dev),
            source_digest=info['source_digest'],context_identity=info['context_identity'],
            reduction='existing bi::mma16816; increasing CH128/kk16; S1; bf16_rn epilogue',
            N=64,K=info['cols'],M=m,S=1,bias=None,activation_tile=64,
            decoded_on='cuda-shared-memory',shared_weight_slab_bytes=64*136*2,
            shared_weight_row_stride_words=136,
            shared_total_bytes=(64+(64 if m>=64 else 16))*136*2,
            output_digest_verified=False,
            gpu_allocation_bytes=dict(serialized_blob=data.numel(),descriptors=desc.numel()*8,
                output=y.numel()*2,lane_status=status.numel()*4,decoded_weights=0,
                derived_tables=_lookup_bytes(lookup)),
            cdf_lookup=cdf_lookup,cdf_lookup_sha256=None if lookup is None else lookup.source_sha256,
            cdf_lookup_resident_bytes=_lookup_bytes(lookup),host_lookup_staging_bytes=host_lookup_bytes)
        return dict(y=y,status=status,provenance=provenance)


PANELS_FORMAT = 'BCT2-fused-independent-panels-N64-K512or1024-S1-v1'


def _panel_spec(blob, panels, m, *, cdf_lookup=False, lookup=None):
    if type(panels) is not int or not 1<=panels<=1024:
        raise PacketError('panels must be an integer in [1,1024]')
    if type(m) is not int or m not in (1,16,64,128):
        raise PacketError('panel M must be 1/16/64/128')
    if type(cdf_lookup) is not bool:
        raise PacketError('cdf_lookup must be a bool')
    blob,info = _parse(blob)
    if cdf_lookup and lookup is None:
        lookup = _build_cdf_lookup(blob,info)
    if not 0<len(blob)<=1048576:
        raise PacketError('bounded panel bytes required')
    k,npackets,blocks = info['cols'],len(info['packets']),(m+63)//64
    table_bytes = _lookup_bytes(lookup) if cdf_lookup else 0
    owned = dict(serialized_blob=panels*len(blob),descriptors=panels*npackets*9*8,
                 output=m*panels*64*2,lane_status=panels*blocks*(k//128)*4*32*4,
                 decoded_weights=0,derived_tables=table_bytes)
    accounting = dict(format=PANELS_FORMAT,wire_format=packet.FORMAT,panels=panels,
        replicated_fixture=True,N=panels*64,K=k,M=m,S=1,bias=None,activation_tile=64,
        data_shape=[panels,len(blob)],descriptor_shape=[panels,npackets,9],
        output_shape=[m,panels*64],status_shape=[panels,blocks,k//128,4,32],
        grid=[blocks,panels,1],grid_ctas=blocks*panels,threads_per_cta=256,
        gpu_allocation_bytes=owned,gpu_owned_bytes=sum(owned.values()),
        activation_bytes=m*k*2,gpu_resident_bytes_including_x=sum(owned.values())+m*k*2,
        dense_control_weight_bytes=panels*64*k*2,
        shared_weight_slab_bytes=64*136*2,shared_weight_row_stride_words=136,
        shared_dynamic_bytes_per_cta=(64+(64 if m>=64 else 16))*136*2,
        shared_static_bytes_per_cta=4,
        source_digest=info['source_digest'],context_identity=info['context_identity'],
        cdf_lookup=cdf_lookup,cdf_lookup_sha256=None if lookup is None else lookup.source_sha256,
        cdf_lookup_resident_bytes=table_bytes,
        scope='replicated bounded fixture grid experiment; not fullmodel/serving/users')
    return blob,info,accounting


def inspect_panels(blob, *, panels, m, cdf_lookup=False):
    """Validate a replicated-panel experiment and bill all explicit GPU tensors.

    x is shared by all panels. Shared memory is per resident CTA, not an HBM
    allocation. Driver/allocator overhead and measured peak memory are separate.
    """
    return _panel_spec(blob,panels,m,cdf_lookup=cdf_lookup)[2]


def _panel_arrays(blob, info, panels):
    # Materialize independent CPU staging arrays; no expanded zero-stride CUDA
    # views and no temporary single-panel CUDA copies are used.
    data = np.tile(np.frombuffer(blob,dtype=np.uint8),(panels,1))
    desc = np.tile(np.array([[q[k] for k in packet.DESCRIPTOR_KEYS] for q in info['packets']],
                            dtype=np.int64),(panels,1,1))
    return data,desc


def gemm_panels_cuda(blob, x, *, panels, device, cdf_lookup=False):
    """Explicit opt-in resident replication experiment with shared x[M,K].

    Every panel stores its own identical validated blob and descriptors; native
    gemm_opt_panels also accepts independent equal-size panels with shared scalar
    metadata geometry. Returns y[M,P*64] only after checking every panel status.
    No production loader selects this entry; M and P are not user counts.
    """
    # Validate panel bounds before importing torch or allocating staging arrays.
    if type(cdf_lookup) is not bool:
        raise PacketError('cdf_lookup must be a bool')
    blob,info,_ = _panel_spec(blob,panels,1)
    import torch
    dev = torch.device(device)
    if dev.type!='cuda' or dev.index is None:
        raise PacketError('explicit CUDA device index required, e.g. cuda:0')
    if (not isinstance(x,torch.Tensor) or x.device!=dev or x.dtype!=torch.bfloat16 or
            x.ndim!=2 or x.shape[0] not in (1,16,64,128) or x.shape[1]!=info['cols'] or
            not x.is_contiguous() or x.data_ptr()%16):
        raise PacketError('aligned contiguous CUDA bf16 x[M,K] with M=1/16/64/128 required')
    lookup = _build_cdf_lookup(blob,info) if cdf_lookup else None
    _,_,accounting = _panel_spec(blob,panels,int(x.shape[0]),cdf_lookup=cdf_lookup,lookup=lookup)
    with torch.cuda.device(dev):
        ext = extension()
        data_host,desc_host = _panel_arrays(blob,info,panels)
        data = torch.from_numpy(data_host).to(dev)
        desc = torch.from_numpy(desc_host).to(dev)
        if cdf_lookup:
            coarse_host,records_host,offsets_host=_lookup_arrays(lookup,info['row_classes']*info['col_classes'])
            host_lookup_bytes=coarse_host.nbytes+records_host.nbytes+offsets_host.nbytes
            coarse=torch.from_numpy(coarse_host).to(dev)
            records=torch.from_numpy(records_host).to(dev)
            offsets=torch.from_numpy(offsets_host).to(dev)
            y,status=ext.gemm_opt_panels_lut(data,desc,x,info['row_classes'],info['col_classes'],
                packet.HEADER.size,info['table_start'],info['table_bytes'],coarse,records,offsets)
        else:
            host_lookup_bytes=0
            y,status = ext.gemm_opt_panels(data,desc,x,info['row_classes'],info['col_classes'],
                                          packet.HEADER.size,info['table_start'],info['table_bytes'])
        codes = status.cpu().numpy().reshape(-1)
        bad = np.flatnonzero(codes)
        if bad.size:
            sample = [(int(i),int(codes[i])) for i in bad[:8]]
            raise PacketError(f'CUDA fused panel status failure: count={bad.size}, first={sample}')
        from glc_serve import kernel_cache
        name,directory,spec = kernel_build_plan()
        root = Path(__file__).resolve().parents[1]
        binary = Path(ext.__file__).resolve()
        provenance = dict(accounting,
            decoder='fixed16x128-v1',optimized=True,decoded_on='cuda-shared-memory',
            source_bindings={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in source_paths()},
            extension=name,extension_path=str(binary),extension_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            build_key=kernel_cache.spec_digest(spec),build_directory=str(directory),device=str(dev),
            torch_version=torch.__version__,cuda_version=torch.version.cuda,device_name=torch.cuda.get_device_name(dev),
            reduction='existing bi::mma16816; increasing CH128/kk16; S1; bf16_rn epilogue',
            output_digest_verified=False,
            host_staging_bytes=data_host.nbytes+desc_host.nbytes+host_lookup_bytes,
            host_lookup_staging_bytes=host_lookup_bytes)
        return dict(y=y,status=status,provenance=provenance)
