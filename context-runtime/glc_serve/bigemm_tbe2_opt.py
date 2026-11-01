"""Recovered TBE2 reference backend; ordinary Context TBE-v1 remains the default.

Origin: attention 932e6d91a031db0a5191e2ecfba99109647e0e0a.
Context adaptation uses the Context-only host ABI and versioned kernel cache.
This does not implement a predictive-package-to-native-TBE2 bridge. CUDA validation
of this recovered Context build is pending.
"""
import hashlib
import os
from pathlib import Path
import threading
from . import bigemm as bi

_EXT = None
_LOCK = threading.Lock()


def check_real_build():
    for name in ('BI_OPCOUNT_COMMON_PATH','CFLAGS','CXXFLAGS','CUDAFLAGS','NVCC_FLAGS','TORCH_NVCC_FLAGS'):
        value = os.environ.get(name)
        if value is not None and (name == 'BI_OPCOUNT_COMMON_PATH' or 'BI_OPCOUNT_COMMON_PATH' in value):
            raise ValueError('BI_OPCOUNT_COMMON_PATH is forbidden: raw and slow-window branches are mandatory')


def source_bindings():
    here = Path(__file__).resolve().parent
    names = ['bigemm_tbe2_opt.py', 'bi_csrc/bi_tbe2_opt.cuh',
             'bi_csrc/bi_tbe2_opt_dense.cu', 'bi_csrc/bi_torch.cpp',
             'bi_csrc/bi_api.h', 'miv_tbe_csrc/miv_tbe_kernels.cuh',
             'miv_tbe_csrc/miv_tbe.h']
    return {name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in names}


def _ext_args(arch):
    """A separate Context-only DSO; no legacy extension or flags are mutated."""
    here = Path(__file__).resolve().parent
    major, minor = arch.split(';')[0].replace('+PTX', '').split('.')
    return dict(
        name=f'glc_bi_tbe2_opt_sm{major}{minor}_context',
        sources=[str(here/'bi_csrc'/name) for name in
                 ('bi_torch.cpp', 'bi_tbe2_opt_dense.cu')],
        extra_include_paths=[str(here/'bi_csrc'), str(here/'miv_tbe_csrc')],
        extra_cflags=bi._winenv.host_cflags() +
                     (['/DGEOR_REFINED_CONTEXT_ONLY=1'] if bi._winenv.is_windows()
                      else ['-DGEOR_REFINED_CONTEXT_ONLY=1']),
        extra_cuda_cflags=['-O3', '--fmad=true', '-lineinfo', '-std=c++17',
                          '-DGEOR_REFINED_CONTEXT_ONLY=1'])


def kernel_build_plan():
    from . import kernel_cache

    check_real_build()
    arch = bi._arch_for_build()
    args = _ext_args(arch)
    spec = kernel_cache.ExtensionSpec(
        name=args['name'], sources=tuple(args['sources']),
        include_dirs=tuple(args['extra_include_paths']),
        flags=tuple(args['extra_cflags']) + ('--cuda--',) + tuple(args['extra_cuda_cflags']),
        arch=arch)
    return args['name'], kernel_cache.resolve_build_dir(spec, 'BI_TBE2_OPT_BUILD_DIR'), spec


def extension():
    global _EXT
    check_real_build()
    if _EXT is not None: return _EXT
    with _LOCK:
        if _EXT is None:
            from glc_loader._jitenv import ensure_build_tools_on_path
            ensure_build_tools_on_path()
            from torch.utils.cpp_extension import load
            from . import kernel_cache

            arch = bi._arch_for_build()
            os.environ.setdefault('TORCH_CUDA_ARCH_LIST', arch)
            name, out, spec = kernel_build_plan()
            args = _ext_args(arch)
            _EXT = kernel_cache.build(
                name, out, lambda: load(**args, build_directory=str(out), verbose=False),
                spec=spec)
    return _EXT


class OptimizedLin:
    def __init__(self, legacy):
        if getattr(legacy,'fmt',None) not in ('tbe2','bf16'): raise ValueError('experimental extension supports only unchanged TBE2 and BF16')
        self.legacy=legacy
        self.N,self.K,self.S=legacy.N,legacy.K,legacy.S
        self.fmt=legacy.fmt

    def __call__(self,x,y,*,tile=64):
        d=self.legacy;m=int(x.shape[0])
        if tile not in (64,128): raise ValueError('tile must be64 or128')
        ws=d.ws.get(d.S*m*d.N) if d.S>1 else d.ws.buf
        extension().bi_gemm_out(x,m,bi.FMT[d.fmt],0,d.N,d.K,d.arrs,d.bias,y,d.S,ws,tile)
        return y

    def decode(self,*,wg=1):
        import torch
        d=self.legacy
        if wg not in (1,2): raise ValueError('wg must be1 or2')
        out=torch.empty((d.N,d.K),dtype=torch.bfloat16,device=d.arrs[0].device)
        extension().bi_decode_out(bi.FMT[d.fmt],d.N,d.K,d.arrs,out,d.S,wg)
        return out
