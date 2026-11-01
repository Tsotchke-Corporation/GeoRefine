"""Explicit BHE1 exponent-prefix experiment; never selected by default loaders.

Class-contained N64 CTAs, eight producer warps, S1 and tile64. Full-model resident
ratio, parent S4/tile128 integration, and serving performance remain unqualified.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import numpy as np
from . import context_exponent_packet as packet

FORMAT='BHE1-experimental-vector-prefix-BIGEMM-S1-tile64-v1'
PacketError=packet.PacketError
encode=packet.encode
decode_cpu=packet.decode_cpu
_parse=packet._parse
ARRAY_KEYS=('main','raw','tables','context_dir','stream_offsets','row_map','tile_rc','positions','counts','panel_desc')


def source_paths():
    base=Path(__file__).resolve().parent;bi=base.parent/'glc_serve'/'bi_csrc';tbe=base.parent/'glc_serve'/'miv_tbe_csrc'
    return tuple([base/('context_exponent_gemm'+suffix) for suffix in ('.py','.cpp','.cu')]+
        [base/'context_exponent_packet.py',base/'context_exponent_packet.cuh',base/'context_exponent_tables.py',
         base/'context_exponent_splits.py',
         base/'context_tiled_packet.py',base/'bitexact_context_codec.py',bi/'bi_gemm.cuh',bi/'bi_api.h',
         tbe/'miv_tbe_kernels.cuh',tbe/'miv_tbe.h'])


def kernel_build_plan():
    from glc_serve import kernel_cache, _winenv
    arch=os.environ.get('TORCH_CUDA_ARCH_LIST')
    if not arch:
        import torch
        major,minor=torch.cuda.get_device_capability();arch=f'{major}.{minor}';os.environ['TORCH_CUDA_ARCH_LIST']=arch
    base=Path(__file__).resolve().parent;name='context_exponent_gemm_v1'
    includes=tuple(str(base.parent/'glc_serve'/s) for s in ('bi_csrc','miv_tbe_csrc'))
    sources=tuple(str(base/('context_exponent_gemm'+s)) for s in ('.cpp','.cu'))
    host_flags=tuple(_winenv.host_cflags()) + (('/D' if _winenv.is_windows() else '-D')+'GEOR_REFINED_CONTEXT_ONLY=1',)
    flags=host_flags+('--cuda--','-O3','--fmad=true','-lineinfo','-std=c++17','-DGEOR_REFINED_CONTEXT_ONLY=1')
    spec=kernel_cache.ExtensionSpec(name=name,sources=sources,include_dirs=includes,
        inline_sources=tuple(p.read_text() for p in source_paths() if p.suffix=='.py'),flags=flags,arch=arch,extra=(('consumer',FORMAT),))
    return name,kernel_cache.extension_dir(spec),spec


def extension():
    from glc_serve import kernel_cache
    from torch.utils.cpp_extension import load
    name,directory,spec=kernel_build_plan()
    return kernel_cache.build(name,directory,lambda:load(name=name,sources=list(spec.sources),
        extra_include_paths=list(spec.include_dirs),build_directory=str(directory),
        extra_cflags=list(spec.flags[:spec.flags.index('--cuda--')]),
        extra_cuda_cflags=['-O3','--fmad=true','-lineinfo','-std=c++17','-DGEOR_REFINED_CONTEXT_ONLY=1'],verbose=False),spec=spec)


def prepare_cuda(blob,*,device,panels=1,splits=1):
    if type(splits) is not int or not 1<=splits<=8:raise PacketError("splits must be1..8")
    if type(panels) is not int or not 1<=panels<=4096:raise PacketError('panels must be1..4096')
    if not isinstance(blob,(bytes,bytearray,memoryview)):raise PacketError('BHE1 bytes required')
    return extension().prepare(bytes(blob),str(device),panels,splits)


def _prepare_matrix_cuda_bound(blobs,*,device,splits=1,validation_witnesses=None):
    """Validate inputs and bind the plan to its loaded native module once."""
    if type(splits) is not int or not 1<=splits<=8:raise PacketError("splits must be1..8")
    if not isinstance(blobs,(list,tuple)) or not 1<=len(blobs)<=4096 or any(not isinstance(b,(bytes,bytearray,memoryview)) for b in blobs):
        raise PacketError('matrix requires1..4096 BHE1 panel blobs')
    if validation_witnesses is not None:
        if (not isinstance(validation_witnesses,(list,tuple)) or
                len(validation_witnesses) != len(blobs)):
            raise PacketError('BHE1 validation witness count must match panel count')
    backend=extension()
    # Preserve exact bytes identities only on the opt-in path.  The default
    # continues to normalize all bytes-like inputs before entering native code.
    native_blobs = list(blobs) if validation_witnesses is not None else [bytes(b) for b in blobs]
    if validation_witnesses is None:
        return backend.prepare_matrix(native_blobs,str(device),splits),backend
    return backend.prepare_matrix(native_blobs,str(device),splits,
                                  validation_witnesses=validation_witnesses),backend


def prepare_matrix_cuda(blobs,*,device,splits=1,validation_witnesses=None):
    return _prepare_matrix_cuda_bound(blobs,device=device,splits=splits,
                                      validation_witnesses=validation_witnesses)[0]


def plan_provenance(plan):
    return dict(extension().plan_provenance(plan))


def _ledger(arrays,m):
    if type(m) is not int or not 1<=m<=512:raise PacketError('M must be1..512')
    n,k,nc,nr,tiles=(int(arrays[key]) for key in ('N','K','nc','nr','tiles'))
    stage=int(arrays['context_dir'][:,1].reshape(nr,nc).sum(axis=1).max())
    sizes={key:int(arrays[key].nbytes) for key in ARRAY_KEYS};resident=sum(sizes.values());blocks=(m+63)//64
    status=tiles*blocks*256*4;output=m*n*2
    allocations=dict(sizes,output=output,lane_status=status,decoded_weights=0,serialized_blob=0,serialized_cdf=0)
    known_host_arrays=sum(v.nbytes for v in arrays.values() if isinstance(v,np.ndarray))
    shared=(64+(64 if m>16 else 16))*136*2+8192+4*stage
    producer_shared=64*136*2+8192+4*stage
    return dict(format=FORMAT,wire_format=packet.FORMAT,scope='experimental exact source panels; no fullmodel or serving qualification',
        N=n,K=k,M=m,S=1,bias=None,activation_tile=64,panels=arrays['panels'],matrix_mode=arrays['matrix_mode'],
        replicated_fixture=not arrays['matrix_mode'],tiles=tiles,class_tail_padding_rows=tiles*64-n,
        grid=[blocks,tiles,1],grid_ctas=blocks*tiles,threads_per_cta=256,native_gemm_launches=1,
        output_shape=[m,n],status_shape=[tiles,blocks,256],producer_status_shape=[tiles,256],
        producer_checksum_shape=[tiles,256],diagnostic_decode_shape=[n,k],
        gpu_allocation_bytes=allocations,gpu_plan_bytes=resident,gpu_owned_bytes=resident+status+output,
        gpu_weight_and_status_bytes=resident+status,gpu_resident_bytes_including_x=resident+status+output+2*m*k,
        activation_bytes=2*m*k,gpu_upload_tensor_peak_bytes=resident,metadata_replication=1,
        host_wire_bytes=arrays['host_wire_bytes'],host_staging_bytes=resident,
        host_staging_peak_bytes=known_host_arrays+max(sizes.values()),host_validation_decoded_weight_transient_bytes=128*k,
        host_known_buffer_peak_lower_bound_bytes=arrays['host_wire_bytes']+known_host_arrays+max(sizes.values())+128*k,
        peak_exclusions=['Python table/tree/container objects','wire concatenation temporaries','allocator/driver reservations','compiler local/spills'],
        allocator_driver_peak_measured=False,host_peak_is_complete=False,
        shared_weight_slab_bytes=64*136*2,shared_weight_row_stride_words=136,
        shared_raw8_bytes_per_cta=8192,shared_context_table_bytes_per_cta=4*stage,
        shared_dynamic_bytes_per_cta=shared,shared_static_bytes_per_cta=4,
        producer_shared_dynamic_bytes_per_cta=producer_shared,producer_shared_static_bytes_per_cta=4,
        producer_threads_per_cta=256,producer_grid=[tiles,1,1],producer_decoded_weight_hbm_bytes=0,
        producer_checksum_bytes=tiles*256*4,producer_status_bytes=tiles*256*4,
        producer_checksum_cost='per-thread word FNV sink over CH slabs, padded rows skipped; checksum time included',
        diagnostic_decode_output_bytes=2*n*k,diagnostic_decode_status_bytes=tiles*256*4,
        dense_control_weight_bytes=2*n*k,dense_control_total_bytes=2*n*k+2*m*k+output,
        context_stage_words=stage,source_digests=arrays['source_digests'],source_digest=arrays['matrix_source_digest'],
        matrix_source_digest=arrays['matrix_source_digest'],row_permutation_digest=arrays['row_permutation_digest'],
        source_cdf_digest=arrays['source_cdf_digest'],map_digest=arrays['map_digest'],
        row_order='stable global class grouping, original output-row scatter',
        planned_S4_stream_start_checkpoint_bytes=4*n*nc*4,
        planned_S4_FP32_partial_bytes=4*m*n*4,
        integration_holds=['parent S4 contiguous split checkpoints/reduction','parent M128 tile128','fullmodel registry/source binding','full resident byte ledger','serving speed gates'])


def inspect(blob):
    return dict(packet.inspect(blob),consumer=FORMAT,allowed_m=list(range(1,513)),S=1,activation_tile=64)


def inspect_panels(blob,*,panels=1,m):
    return _ledger(packet._resident_arrays(blob,panels),m)


def inspect_matrix(blobs,*,m):
    return _ledger(packet._resident_matrix_arrays(blobs),m)


def gemm_cuda(blob,x,*,device,panels=1):
    import torch
    dev=torch.device(device)
    if dev.type!='cuda' or dev.index is None or not isinstance(x,torch.Tensor) or x.device!=dev:
        raise PacketError('explicit CUDA device and matching x required')
    with torch.cuda.device(dev):
        plan=prepare_cuda(blob,device=dev,panels=panels);ext=extension();y,status=ext.gemm(plan,x)
        values=status.cpu().numpy().reshape(-1);bad=np.flatnonzero(values)
        if bad.size:raise PacketError(f'CUDA exponent status failure: {[(int(i),int(values[i])) for i in bad[:8]]}')
        binary=Path(ext.__file__).resolve();root=Path(__file__).resolve().parents[1]
        provenance=dict(format=FORMAT,resident_plan=plan_provenance(plan),decoded_on='cuda-shared-memory',
            extension_path=str(binary),extension_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            source_bindings={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths()},
            reduction='existing BI mma16816 CH128/kk16 order, S1, bf16_rn epilogue',device=str(dev))
        return dict(y=y,status=status,provenance=provenance)
