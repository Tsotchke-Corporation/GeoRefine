"""Explicit BCA4 experimental raw6 consumer; no loader selects this module.

N64,S1 fixtures (K512/1024 default; opt-in multiples128 through32768), optionally replicated resident
panels. No BCTX mode128/archive-ratio bridge, fullmodel, or user-count claim.
CUDA is loaded only on explicit extension()/gemm_cuda() calls, never import.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import numpy as np
from . import context_alias_packet as packet

PacketError=packet.PacketError
FORMAT='BCA4-experimental-persistent-BIGEMM-S1-tile64-v1'
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
    c=info['row_classes']*info['col_classes'];t=info['tables']
    bucket_bytes=t.bucket_records.nbytes if t else 0
    payload_bytes=sum(p['main_bytes']+p['residual_bytes'] for p in info['packets'])
    gpu=dict(compact_payloads=panels*payload_bytes,descriptors=panels*4*9*8,
             alias_records=bucket_bytes,alias_offsets=4*(c+1),alias_shifts=c,
             labels=64+k,digest_binding=64,plan_control=32,
             output=m*panels*64*2,lane_status=panels*blocks*4*32*4,
             decoded_weights=0,serialized_blob=0,serialized_cdf=0,cursor_snapshots=0)
    wx=(64+(64 if m>=64 else 16))*136*2
    ledger=dict(format=FORMAT,wire_format=packet.FORMAT,experimental_raw6=True,
        scope='bounded replicated fixture; no fullmodel/archive-ratio/user qualification',
        panels=panels,replicated_fixture=True,N=64*panels,K=k,M=m,S=1,bias=None,activation_tile=64,
        data_shape=[panels,payload_bytes],descriptor_shape=[panels,4,9],
        output_shape=[m,64*panels],status_shape=[panels,blocks,4,32],
        grid=[blocks,panels,1],grid_ctas=blocks*panels,threads_per_cta=256,
        gpu_allocation_bytes=gpu,gpu_owned_bytes=sum(gpu.values()),activation_bytes=m*k*2,
        gpu_resident_bytes_including_x=sum(gpu.values())+m*k*2,
        gpu_weight_and_status_bytes=sum(gpu.values())-gpu['output'],
        host_wire_bytes=len(blob),host_inverse_bytes=t.inverse_bytes if t else 0,
        host_validation_decoded_weight_transient_bytes=128*k,
        diagnostic_decode_output_bytes=panels*128*k,diagnostic_decode_status_bytes=panels*128*4,
        metadata_replication=1,preparation='host full wire/map/payload validation, private GPU upload once',
        preparation_peak='separate host staging/inverse and driver/allocator measurements required',
        dense_control_weight_bytes=panels*64*k*2,
        dense_control_total_bytes=panels*64*k*2+m*k*2+gpu['output'],
        shared_weight_slab_bytes=64*136*2,shared_weight_row_stride_words=136,
        shared_weight_activation_bytes_per_cta=wx,shared_packet_view_bytes_per_cta=4*VIEW_SLOT_BYTES,
        shared_dynamic_bytes_per_cta=wx+4*VIEW_SLOT_BYTES,shared_static_bytes_per_cta=4,
        source_digest=info['source_digest'],context_identity=info['context_identity'],
        source_cdf_digest=info['source_cdf_digest'],alias_map_digest=info['alias_map_digest'])
    return blob,info,ledger


def inspect(blob):
    """BCA4 wire accounting plus the explicit consumer contract."""
    return dict(packet.inspect(blob),consumer=FORMAT,activation_tile=64,allowed_m=[1,16,64,128],
                shared_weight_slab_bytes=64*136*2,shared_weight_row_stride_words=136)


def inspect_panels(blob,*,panels,m,variant="baseline"):
    """Bill explicit resident tensors and reserved shared bytes; no GPU work.

    Input x is shared across panels. Driver/allocator peaks and compiler local
    memory require separate measurement; final status has no K-chunk dimension.
    """
    return _variant_ledger(_plan(blob,panels,m)[2],variant)


def source_paths():
    base=Path(__file__).resolve().parent;bi=base.parent/'glc_serve'/'bi_csrc'
    tbe=base.parent/'glc_serve'/'miv_tbe_csrc'
    return tuple([base/('context_alias_gemm'+s) for s in ('.py','.cpp','.cu')]+
        [base/'context_alias_packet.py',base/'context_alias_packet.cuh',base/'context_alias_ans.py',
         base/'context_cdf_lookup.py',
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
    name='context_alias_gemm_v1'
    includes=tuple(str(base.parent/'glc_serve'/s) for s in ('bi_csrc','miv_tbe_csrc'))
    sources=tuple(str(base/('context_alias_gemm'+s)) for s in ('.cpp','.cu'))
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


def prepare_cuda(blob,*,device,panels=1):
    """Validate host wire and full payload once; return an opaque resident plan.

    Plan owns private allocations; no tensor aliases or mutable raw-entry bypass.
    Creation includes CPU decode verification and upload synchronization. Native
    resident gemm does not repeat host/global validation or retain decoded HBM.
    """
    if type(panels) is not int or not 1<=panels<=1024:raise PacketError('invalid panels')
    if not isinstance(blob,(bytes,bytearray,memoryview)):raise PacketError('bundle must be bytes')
    return extension().prepare(bytes(blob),str(device),panels)


def plan_provenance(plan):
    return dict(extension().plan_provenance(plan))


def gemm_cuda(blob,x,*,device,panels=1):
    """Explicit experiment; y is exposed only after final status succeeds."""
    import torch
    dev=torch.device(device)
    if dev.type!='cuda' or dev.index is None:raise PacketError('explicit CUDA device required')
    if not isinstance(x,torch.Tensor) or x.device!=dev:raise PacketError('x must match plan device')
    _,_,ledger=_plan(blob,panels,int(x.shape[0]))
    with torch.cuda.device(dev):
        plan=prepare_cuda(blob,device=dev,panels=panels)
        ext=extension();y,status=ext.gemm(plan,x)
        codes=status.cpu().numpy().reshape(-1);bad=np.flatnonzero(codes)
        if bad.size:raise PacketError(f'CUDA alias status failure: {[(int(i),int(codes[i])) for i in bad[:8]]}')
        from glc_serve import kernel_cache
        name,directory,spec=kernel_build_plan();binary=Path(ext.__file__).resolve()
        root=Path(__file__).resolve().parents[1]
        provenance=dict(ledger,decoded_on='cuda-shared-memory',output_digest_verified=False,
            source_bindings={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths()},
            extension=name,extension_path=str(binary),extension_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            build_key=kernel_cache.spec_digest(spec),build_directory=str(directory),device=str(dev),
            torch_version=torch.__version__,cuda_version=torch.version.cuda,device_name=torch.cuda.get_device_name(dev),
            reduction='existing bi::mma16816; increasing CH128/kk16; S1; bf16_rn epilogue',
            resident_plan=plan_provenance(plan))
        return dict(y=y,status=status,provenance=provenance)


def prepare_matrix_cuda(blobs,*,device):
    """Seal distinct N64 row panels as one matrix, N<=262144 and S1.

    Every panel must carry identical original CDF/map and column labels. Encode
    with shared_metadata=True to retain exact CDF even for raw-only panels.
    One global native grid covers all rows; no replication or per-panel launches.
    N16/N32 tails remain explicitly unsupported.
    """
    if not isinstance(blobs,(list,tuple)) or not 1<=len(blobs)<=4096:
        raise PacketError('matrix requires 1..4096 row panels, N multiple64')
    if any(not isinstance(blob,(bytes,bytearray,memoryview)) for blob in blobs):
        raise PacketError('matrix row panels must be bytes')
    return extension().prepare_matrix([bytes(blob) for blob in blobs],str(device))


def inspect_matrix(blobs,*,m,variant="baseline"):
    """Complete tensor ledger for distinct row panels, with explicit peak limits.

    Host wires repeat source CDFs; GPU metadata is stored once. Tensor byte peaks
    are exact, host validation bound is conservative; Python object overhead,
    compiler spills and allocator/driver reservations need separate measurement.
    """
    if type(m) is not int or m not in (1,16,64,128):raise PacketError('M must be 1/16/64/128')
    panels=packet._matrix_panels(blobs);p=len(panels);first=panels[0][1];n=64*p;k=first['cols']
    tables=first['tables'];c=first['row_classes']*first['col_classes'];blocks=(m+63)//64
    payload=sum(q['main_bytes']+q['residual_bytes'] for _,info in panels for q in info['packets'])
    records=tables.bucket_records.nbytes if tables else 0
    gpu=dict(compact_payloads=payload,descriptors=p*4*9*8,alias_records=records,
             alias_offsets=4*(c+1),alias_shifts=c,labels=n+k,digest_binding=64,plan_control=32,
             output=m*n*2,lane_status=p*blocks*4*32*4,decoded_weights=0,
             serialized_blob=0,serialized_cdf=0,cursor_snapshots=0)
    metadata=records+4*(c+1)+c+n+k+96
    resident=payload+p*4*9*8+metadata
    staging_peak=resident+max(payload,p*4*9*8,metadata)
    wire=sum(len(blob) for blob,_ in panels)
    inverse=tables.inverse_bytes if tables else 0
    wx=(64+(64 if m>=64 else 16))*136*2
    ledger=dict(format=FORMAT,wire_format=packet.FORMAT,experimental_raw6=True,
        scope='distinct full matrix row panels; serving and full-model qualification pending',
        panels=p,replicated_fixture=False,matrix_mode=True,N=n,K=k,M=m,S=1,bias=None,activation_tile=64,
        data_shape=[payload],descriptor_shape=[p,4,9],output_shape=[m,n],status_shape=[p,blocks,4,32],
        grid=[blocks,p,1],grid_ctas=blocks*p,threads_per_cta=256,native_gemm_launches=1,
        gpu_allocation_bytes=gpu,gpu_owned_bytes=sum(gpu.values()),activation_bytes=m*k*2,
        gpu_resident_bytes_including_x=sum(gpu.values())+m*k*2,
        gpu_weight_and_status_bytes=resident+gpu['lane_status'],gpu_plan_bytes=resident,
        gpu_upload_tensor_peak_bytes=resident,host_staging_bytes=resident,
        host_staging_peak_bytes=staging_peak,host_wire_bytes=wire,
        host_wire_repeated_cdf_bytes=sum(info['table_bytes'] for _,info in panels),
        host_inverse_bytes=inverse,host_validation_decoded_weight_transient_bytes=128*k,
        host_decode_numpy_frame_peak_bound_bytes=512*k+4096,
        host_wire_validation_copy_bound_bytes=2*max(len(blob) for blob,_ in panels),
        host_known_buffer_peak_upper_bound_bytes=wire+staging_peak+inverse+records+4*(c+1)+c+512*k+4096+2*first['table_bytes']+2*max(len(blob) for blob,_ in panels),
        peak_exclusions=['Python object/container overhead','compiler local/spill storage','allocator/driver reserved memory'],
        allocator_driver_peak_measured=False,metadata_replication=1,
        diagnostic_decode_output_bytes=2*n*k,diagnostic_decode_status_bytes=p*128*4,
        producer_decoded_weight_hbm_bytes=0,producer_checksum_bytes=p*128*4,
        producer_status_bytes=p*128*4,producer_grid=[p,1,1],producer_threads_per_cta=128,
        producer_checksum_cost='per-lane word FNV1a-like uint32 sink; included in producer latency; not a correctness digest',
        producer_shared_dynamic_bytes_per_cta=64*136*2+4*VIEW_SLOT_BYTES,
        producer_shared_static_bytes_per_cta=4,
        dense_control_weight_bytes=2*n*k,dense_control_total_bytes=2*n*k+2*m*k+2*m*n,
        shared_weight_slab_bytes=64*136*2,shared_weight_row_stride_words=136,
        shared_weight_activation_bytes_per_cta=wx,shared_packet_view_bytes_per_cta=4*VIEW_SLOT_BYTES,
        shared_dynamic_bytes_per_cta=wx+4*VIEW_SLOT_BYTES,shared_static_bytes_per_cta=4,
        source_cdf_digest=first['source_cdf_digest'],alias_map_digest=first['alias_map_digest'],
        matrix_source_digest_kind='sha256 of ordered panel source SHA256 digests',
        matrix_source_digest=packet._hash(b''.join(bytes.fromhex(info['source_digest']) for _,info in panels)).hex())

    return _variant_ledger(ledger,variant)



def _variant_ledger(ledger,variant):
    """Only shared stream staging varies; every GPU-owned tensor is unchanged."""
    if type(variant) is not str or variant not in ('baseline','register','aligned','trusted','fifo','fifo_raw6'):
        raise PacketError('unsupported alias consumer variant')
    if variant not in ('fifo','fifo_raw6'):return ledger
    fifo=4*576 if variant in ('fifo','fifo_raw6') else 0
    raw6=4*1536 if variant=='fifo_raw6' else 0
    extra=fifo+raw6
    return dict(ledger,consumer_variant=variant,
        shared_main_fifo_bytes_per_cta=fifo,shared_raw6_stage_bytes_per_cta=raw6,
        shared_stream_staging_bytes_per_cta=extra,
        shared_dynamic_bytes_per_cta=ledger['shared_dynamic_bytes_per_cta']+extra,
        producer_shared_dynamic_bytes_per_cta=64*136*2+4*VIEW_SLOT_BYTES+extra,
        diagnostic_decode_shared_dynamic_bytes_per_cta=64*136*2+4*VIEW_SLOT_BYTES+extra,
        main_fifo_bytes_per_warp=576 if fifo else 0,
        main_fifo_window_bytes_per_warp=512 if fifo else 0,
        raw6_stage_bytes_per_warp=1536 if raw6 else 0,
        shared_storage_scope='dynamic bytes exact; compiler/driver static allocation and occupancy require measurement')
