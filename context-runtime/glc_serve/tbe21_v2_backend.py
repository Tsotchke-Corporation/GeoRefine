"""Isolated, opt-in TBE21 kernel-v2 backend; legacy BI-GEMM stays the default."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import threading

_EXT = None
_OVERLAP = None
_LOCK = threading.Lock()
_FLAG = "BI_TBE21_V2"
_FORMAT_TBE21 = 5
_TIER_FLAGS = ("BI_T21_TIER_PREFIX", "BI_T21_TPF_PREFETCH", "BI_T21_TPF_WG1",
               "BI_T21_TPF_PREFETCH_WG1")
_TIER_METADATA_ABI = 1

# SHA256 of the exact pinned 90e83dd8 source files before Context-only guard and
# private-symbol normalization. Kept beside loaded source hashes for provenance.
_PINNED_SOURCE_SHA256 = {
    "bi_api.h": "734b899c9f334dd063ac52d9e0571921f3fb942c976fbeea0f5ca10394bf473a",
    "bi_gemm.cuh": "2a417671cbf4e8d820258da1f2254e769b3cfde506c56d343570ebc9ecad5ee3",
    "bi_gemm_v2.cu": "aa8f87f24447334f4b600e182460f5c713fd3fd048f7c64d8a9ebcd1d72e32a6",
    "miv_tbe_kernels.cuh": "eaba6073abde4adcd549df0f095bf11a6bddfaf3901ea40a842a0ffbe2844fb6",
    "miv_tbe.h": "0aa9cde196788768920b5b941799e210ef29a40e4e797597f0599e8d62ffd486",
}


def enabled() -> bool:
    """Strict default-off selector. Malformed values fail closed."""
    value = os.environ.get(_FLAG, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{_FLAG} must be exactly 0 or 1")
    return value == "1"


def source_bindings() -> dict[str, str]:
    """Hashes of this loader, private binding, and complete kernel/header closure."""
    here = Path(__file__).resolve().parent
    names = [
        "tbe21_v2_backend.py",
        "bi_tbe21_v2_csrc/bi_torch.cpp",
        "bi_tbe21_v2_csrc/bi_tbe21_v2_api.h",
        "bi_tbe21_v2_csrc/bi_gemm_v2.cu",
        "bi_tbe21_v2_csrc/bi_gemm.cuh",
        "bi_tbe21_v2_csrc/pair_vote.cuh",
        "bi_tbe21_v2_csrc/bi_api.h",
        "bi_tbe21_v2_csrc/miv_tbe_kernels.cuh",
        "bi_tbe21_v2_csrc/miv_tbe.h",
        "bi_tbe21_v2_csrc/tier_prefix_literal.cuh",
        "bi_tbe21_v2_csrc/tier_prefix_cursor.cuh",
    ]
    return {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in names}


def check_real_build() -> None:
    """Reject local diagnostic/opcount overrides that could change compiled code."""
    for name in ("BI_OPCOUNT_COMMON_PATH", "CFLAGS", "CXXFLAGS", "CUDAFLAGS",
                 "NVCC_FLAGS", "TORCH_NVCC_FLAGS"):
        value = os.environ.get(name)
        if value is not None and (name == "BI_OPCOUNT_COMMON_PATH" or
                                  "BI_OPCOUNT_COMMON_PATH" in value):
            raise ValueError(f"{name} diagnostic override is forbidden for TBE21 v2 builds")


def selected_kernel_variant(environ=None) -> int:
    """Explicit schedule for the same TBE21 codec; omitted selection retains schedule 2."""
    env = os.environ if environ is None else environ
    value = env.get("BI_TBE21_KERNEL_VARIANT", "2")
    if value not in ("1", "2"):
        raise ValueError("BI_TBE21_KERNEL_VARIANT must be exactly 1 or 2")
    return int(value)


def selected_packed_scan2(environ=None) -> bool:
    env = os.environ if environ is None else environ
    raw = env.get("BI_T21_PACKSCAN2", "0")
    if raw not in ("0", "1"):
        raise ValueError("BI_T21_PACKSCAN2 must be exactly 0 or 1")
    if raw == "1" and selected_kernel_variant(env) != 1:
        raise ValueError("BI_T21_PACKSCAN2 requires TBE21 kernel variant 1")
    return raw == "1"


def selected_tile128(environ=None) -> bool:
    env = os.environ if environ is None else environ
    raw = env.get("BI_T21_TILE128", "0")
    if raw not in ("0", "1"):
        raise ValueError("BI_T21_TILE128 must be exactly 0 or 1")
    if raw == "1" and (selected_kernel_variant(env) != 1 or not selected_packed_scan2(env)):
        raise ValueError("BI_T21_TILE128 requires schedule 1 and BI_T21_PACKSCAN2=1")
    return raw == "1"

def selected_async_x(environ=None) -> bool:
    env = os.environ if environ is None else environ
    raw = env.get("BI_T21_ASYNC_X", "0")
    if raw not in ("0", "1"):
        raise ValueError("BI_T21_ASYNC_X must be exactly 0 or 1")
    if raw == "1" and selected_kernel_variant(env) != 1:
        raise ValueError("BI_T21_ASYNC_X requires TBE21 kernel variant 1")
    return raw == "1"


def selected_row_geometry(environ=None) -> bool:
    env = os.environ if environ is None else environ
    raw = env.get("BI_T21_ROW_GEOMETRY", "0")
    if raw not in ("0", "1"):
        raise ValueError("BI_T21_ROW_GEOMETRY must be exactly 0 or 1")
    if raw == "1" and (env.get(_FLAG, "0") != "1" or not selected_tile128(env)):
        raise ValueError("BI_T21_ROW_GEOMETRY requires the native backend, schedule 1, paired scan and tile128")
    return raw == "1"


def selected_exec_layout(environ=None) -> bool:
    """Select the experimental exact-M64 and M97..128 TBE21 execution layouts."""
    env = os.environ if environ is None else environ
    raw = env.get("BI_T21_EXEC_LAYOUT", "0")
    if raw not in ("0", "1"):
        raise ValueError("BI_T21_EXEC_LAYOUT must be exactly 0 or 1")
    if raw == "1" and (
            env.get(_FLAG, "0") != "1" or selected_kernel_variant(env) != 1
            or not selected_packed_scan2(env) or not selected_tile128(env)
            or env.get("BI_T21_TABREG", "0") != "1"
            or env.get("BI_T21_V1_MINBLOCKS", "2") != "2"
            or any(env.get(name, "0") != "0" for name in (
                "BI_T21_SCAN", "BI_T21_SCAN4", "BI_T21_PACKSCAN4", "BI_T21_ASYNC_X",
                "BI_T21_ROW_GEOMETRY", "BI_T21_WIDE128", "BI_T21_PIPE128",
                "BI_T21_PAIR_VOTE", "BI_T21_TIER_PREFIX", "BI_T21_TPF_PREFETCH",
                "BI_T21_TPF_WG1", "BI_T21_TPF_PREFETCH_WG1"))):
        raise ValueError("exec layout requires schedule 1, PACKSCAN2, TILE128, TABREG, MINBLOCKS=2 and incompatible paths off")
    return raw == "1"


def selected_scan4(environ=None) -> bool:
    env = os.environ if environ is None else environ
    raw = env.get("BI_T21_SCAN4", "0")
    if raw not in ("0", "1"):
        raise ValueError("BI_T21_SCAN4 must be exactly 0 or 1")
    if raw == "1" and (env.get(_FLAG, "0") != "1" or not selected_packed_scan2(env)):
        raise ValueError("BI_T21_SCAN4 requires the native backend, schedule 1 and paired scan")
    return raw == "1"


def selected_wide128(environ=None) -> bool:
    env = os.environ if environ is None else environ
    raw = env.get("BI_T21_WIDE128", "0")
    if raw not in ("0", "1"):
        raise ValueError("BI_T21_WIDE128 must be exactly 0 or 1")
    if raw == "1" and (env.get(_FLAG, "0") != "1" or not selected_tile128(env) or selected_scan4(env)):
        raise ValueError("BI_T21_WIDE128 requires the native backend, schedule 1, paired scan, tile128 and SCAN4 off")
    return raw == "1"


def selected_pipe128(environ=None) -> bool:
    env = os.environ if environ is None else environ
    raw = env.get("BI_T21_PIPE128", "0")
    if raw not in ("0", "1"):
        raise ValueError("BI_T21_PIPE128 must be exactly 0 or 1")
    if raw == "1" and (not selected_wide128(env) or selected_async_x(env)):
        raise ValueError("BI_T21_PIPE128 requires compact M128 and BI_T21_ASYNC_X off")
    return raw == "1"


def selected_packed_scan4(environ=None) -> bool:
    env = os.environ if environ is None else environ
    raw = env.get("BI_T21_PACKSCAN4", "0")
    if raw not in ("0", "1"):
        raise ValueError("BI_T21_PACKSCAN4 must be exactly 0 or 1")
    if raw == "1":
        if env.get(_FLAG, "0") != "1" or not selected_packed_scan2(env):
            raise ValueError("BI_T21_PACKSCAN4 requires the native backend, schedule 1 and paired scan")
        incompatible = ("BI_T21_SCAN", "BI_T21_SCAN4", "BI_T21_TABREG", "BI_T21_ASYNC_X",
                        "BI_T21_ROW_GEOMETRY", "BI_T21_WIDE128", "BI_T21_PIPE128")
        if any(env.get(name, "0") != "0" for name in incompatible):
            raise ValueError("BI_T21_PACKSCAN4 requires scalar compact tables and incompatible experiments off")
    return raw == "1"


def selected_pair_vote(environ=None) -> bool:
    env = os.environ if environ is None else environ
    raw = env.get("BI_T21_PAIR_VOTE", "0")
    if raw not in ("0", "1"):
        raise ValueError("BI_T21_PAIR_VOTE must be exactly 0 or 1")
    if raw == "1":
        if (env.get(_FLAG, "0") != "1" or selected_kernel_variant(env) != 1
                or not selected_packed_scan2(env) or not selected_tile128(env)
                or env.get("BI_T21_TABREG", "0") != "1"
                or any(env.get(name, "0") != "0" for name in
                       ("BI_T21_SCAN", "BI_T21_SCAN4", "BI_T21_PACKSCAN4", "BI_T21_ASYNC_X",
                        "BI_T21_ROW_GEOMETRY", "BI_T21_WIDE128", "BI_T21_PIPE128"))):
            raise ValueError("BI_T21_PAIR_VOTE requires schedule 1, paired scan, tile128, TABREG and incompatible paths off")
    return raw == "1"


def selected_tier_prefix(environ=None) -> bool:
    env = os.environ if environ is None else environ
    values = {}
    for name in _TIER_FLAGS:
        raw = env.get(name, "0")
        if raw not in ("0", "1"):
            raise ValueError(f"{name} must be exactly 0 or 1")
        values[name] = raw == "1"
    prefix = values[_TIER_FLAGS[0]]
    if any(values[name] for name in _TIER_FLAGS[1:]) and not prefix:
        raise ValueError("TBE21 tier-prefix variants require BI_T21_TIER_PREFIX=1")
    if prefix:
        selected_row_geometry(env)
        if (env.get(_FLAG, "0") != "1" or selected_kernel_variant(env) != 1
                or not selected_packed_scan2(env) or not selected_tile128(env)
                or env.get("BI_T21_TABREG", "0") != "1"
                or env.get("BI_T21_V1_MINBLOCKS", "2") != "2"
                or any(env.get(name, "0") != "0" for name in
                       ("BI_T21_SCAN", "BI_T21_SCAN4", "BI_T21_PACKSCAN4", "BI_T21_ASYNC_X",
                        "BI_T21_WIDE128", "BI_T21_PIPE128", "BI_T21_PAIR_VOTE"))):
            raise ValueError("TBE21 tier-prefix requires schedule 1, PACKSCAN2, TILE128, TABREG, MINBLOCKS=2 and incompatible paths off")
    if values["BI_T21_TPF_PREFETCH_WG1"]:
        if not values["BI_T21_TPF_WG1"] or not values["BI_T21_TPF_PREFETCH"]:
            raise ValueError("BI_T21_TPF_PREFETCH_WG1 requires BI_T21_TPF_WG1=1 and BI_T21_TPF_PREFETCH=1")
    return prefix


def _tier_prefix_flags():
    prefix = selected_tier_prefix()
    return [f"-D{name}=1" for name in _TIER_FLAGS if os.environ.get(name, "0") == "1"] if prefix else []


def tile_policy() -> dict:
    return {"rows_le_64": 64, "rows_gt_64": 128}


def _t21_kernel_flags() -> list[str]:
    """Strict, default-off TBE21 compiler controls included in the private build identity."""
    values = (("BI_T21_TILE128", {"0", "1"}), ("BI_T21_PACKSCAN2", {"0", "1"}), ("BI_T21_SCAN", {"0", "1"}), ("BI_T21_TABREG", {"0", "1"}),
              ("BI_T21_V1_MINBLOCKS", {"2", "3"}))
    selected_packed_scan2()
    selected_tile128()
    row_geometry = selected_row_geometry()
    scan4 = selected_scan4()
    wide128 = selected_wide128()
    pipe128 = selected_pipe128()
    packed_scan4 = selected_packed_scan4()
    pair_vote = selected_pair_vote()
    exec_layout = selected_exec_layout()
    tier_prefix = selected_tier_prefix()
    flags = []
    for name, allowed in values:
        value = os.environ.get(name)
        if value is not None:
            if value not in allowed:
                choices = " or ".join(sorted(allowed))
                raise ValueError(f"{name} must be exactly {choices}")
            flags.append(f"-D{name}={value}")
    if row_geometry:
        flags.append("-DBI_T21_ROW_GEOMETRY=1")
    if scan4:
        flags.append("-DBI_T21_SCAN4=1")
    if wide128:
        flags.append("-DBI_T21_WIDE128=1")
    if pipe128:
        flags.append("-DBI_T21_PIPE128=1")
    if packed_scan4:
        flags.append("-DBI_T21_PACKSCAN4=1")
    if pair_vote:
        flags.append("-DBI_T21_PAIR_VOTE=1")
    if exec_layout:
        flags.append("-DBI_T21_EXEC_LAYOUT=1")
    flags.extend(_tier_prefix_flags())
    return flags


def _ext_args(arch: str) -> dict:
    here = Path(__file__).resolve().parent
    major, minor = arch.split(";")[0].replace("+PTX", "").split(".")
    src = here / "bi_tbe21_v2_csrc"
    async_x = selected_async_x()
    if selected_pipe128() and int(major) < 8:
        raise ValueError("BI_T21_PIPE128 requires SM80 or newer")
    if selected_packed_scan4() and int(major) < 8:
        raise ValueError("BI_T21_PACKSCAN4 requires SM80 or newer")
    if selected_pair_vote() and arch.split(";")[0] != "12.0":
        raise ValueError("BI_T21_PAIR_VOTE requires exact SM120")
    if selected_tier_prefix() and arch != "12.0":
        raise ValueError("BI_T21_TIER_PREFIX requires exact SM120")
    if async_x and int(major) < 8:
        raise ValueError("BI_T21_ASYNC_X requires SM80 or newer")
    from glc_serve import _winenv

    return dict(
        name=f"glc_bi_tbe21_v2_sm{major}{minor}_context" + ("_asyncx" if async_x else "")
             + ("_rowgeom" if selected_row_geometry() else "")
             + ("_scan4" if selected_scan4() else "")
             + ("_wide128" if selected_wide128() else "")
             + ("_pipe128" if selected_pipe128() else "")
             + ("_packscan4" if selected_packed_scan4() else "")
             + ("_pairvote" if selected_pair_vote() else "")
             + ("_execlayout" if selected_exec_layout() else "")
             + ("_tpf" + "".join(("p" if os.environ.get("BI_T21_TPF_PREFETCH", "0") == "1" else "0",
                                  "w" if os.environ.get("BI_T21_TPF_WG1", "0") == "1" else "0",
                                  "q" if os.environ.get("BI_T21_TPF_PREFETCH_WG1", "0") == "1" else "0"))
                if selected_tier_prefix() else ""),
        sources=[str(src / "bi_torch.cpp"), str(src / "bi_gemm_v2.cu")],
        extra_include_paths=[str(src)],
        extra_cflags=_winenv.host_cflags() + (["/DGEOR_REFINED_CONTEXT_ONLY=1", "/DBI_T21_EXEC_LAYOUT=1"] if selected_exec_layout() else ["/DGEOR_REFINED_CONTEXT_ONLY=1"])
        + (["/DBI_T21_TIER_PREFIX=1"] if selected_tier_prefix() else [])
        if _winenv.is_windows()
        else _winenv.host_cflags() + ["-DGEOR_REFINED_CONTEXT_ONLY=1"]
        + (["-DBI_T21_EXEC_LAYOUT=1"] if selected_exec_layout() else [])
        + (["-DBI_T21_TIER_PREFIX=1"] if selected_tier_prefix() else []),
        extra_cuda_cflags=["-O3", "--fmad=true", "-lineinfo", "-std=c++17",
                           "-DGEOR_REFINED_CONTEXT_ONLY=1"] + _t21_kernel_flags()
                          + (["-DGEOR_TBE21_KERNEL_VARIANT=1"] if selected_kernel_variant() == 1 else [])
                          + (["-DBI_T21_ASYNC_X=1"] if async_x else []),
    )


def kernel_build_plan():
    check_real_build()
    from glc_serve import bigemm as bi
    from glc_serve import kernel_cache

    arch = bi._arch_for_build()
    args = _ext_args(arch)
    spec = kernel_cache.ExtensionSpec(
        name=args["name"], sources=tuple(args["sources"]),
        include_dirs=tuple(args["extra_include_paths"]),
        flags=tuple(args["extra_cflags"]) + ("--cuda--",) + tuple(args["extra_cuda_cflags"]),
        arch=arch,
    )
    return args["name"], kernel_cache.resolve_build_dir(spec, "BI_TBE21_V2_BUILD_DIR"), spec


class BackendAdapter:
    """Narrow adapter with the legacy BILin ``bi_gemm_out`` call signature."""

    def __init__(self, module):
        self._module = module
        self._kernel_variant = selected_kernel_variant()
        self._packed_scan2 = selected_packed_scan2()
        self._tile128 = selected_tile128()
        self._async_x = selected_async_x()
        self._row_geometry = selected_row_geometry()
        self._exec_layout = selected_exec_layout()
        self._scan4 = selected_scan4()
        self._wide128 = selected_wide128()
        self._pipe128 = selected_pipe128()
        self._packed_scan4 = selected_packed_scan4()
        self._pair_vote = selected_pair_vote()
        self._tier_prefix = selected_tier_prefix()
        self._tier_prefetch = os.environ.get("BI_T21_TPF_PREFETCH", "0") == "1"
        self._tier_wg1 = os.environ.get("BI_T21_TPF_WG1", "0") == "1"
        self._tier_prefetch_wg1 = os.environ.get("BI_T21_TPF_PREFETCH_WG1", "0") == "1"
        self._check_tier_native(module)
        quad_probe = getattr(module, "packed_scan4", None)
        if quad_probe is None:
            if self._packed_scan4:
                raise RuntimeError("selected TBE21 packed_scan4 lacks native compiled-flag proof")
        else:
            actual_quad = quad_probe()
            if type(actual_quad) is not bool or actual_quad != self._packed_scan4:
                raise RuntimeError("native TBE21 packed_scan4 differs from selected flag")
        pipe_probe = getattr(module, "pipe128", None)
        if pipe_probe is None:
            if self._pipe128:
                raise RuntimeError("selected TBE21 pipe128 lacks native compiled-flag proof")
        else:
            actual_pipe = pipe_probe()
            if type(actual_pipe) is not bool or actual_pipe != self._pipe128:
                raise RuntimeError("native TBE21 pipe128 differs from selected flag")
        wide_probe = getattr(module, "wide128", None)
        if wide_probe is None:
            if self._wide128:
                raise RuntimeError("selected TBE21 wide128 lacks native compiled-flag proof")
        else:
            actual_wide = wide_probe()
            if type(actual_wide) is not bool or actual_wide != self._wide128:
                raise RuntimeError("native TBE21 wide128 differs from selected flag")
        scan4_probe = getattr(module, "scan4", None)
        if scan4_probe is None:
            if self._scan4:
                raise RuntimeError("selected TBE21 scan4 lacks native compiled-flag proof")
        else:
            actual_scan4 = scan4_probe()
            if type(actual_scan4) is not bool or actual_scan4 != self._scan4:
                raise RuntimeError("native TBE21 scan4 differs from selected flag")
        row_probe = getattr(module, "row_geometry", None)
        if row_probe is None:
            if self._row_geometry:
                raise RuntimeError("selected TBE21 row geometry lacks native compiled-flag proof")
        else:
            actual_rows = row_probe()
            if type(actual_rows) is not bool or actual_rows != self._row_geometry:
                raise RuntimeError("native TBE21 row geometry differs from selected flag")
        exec_probe = getattr(module, "exec_layout", None)
        if exec_probe is None:
            if self._exec_layout:
                raise RuntimeError("selected TBE21 exec layout lacks native compiled-flag proof")
        else:
            actual_exec = exec_probe()
            if type(actual_exec) is not bool or actual_exec != self._exec_layout:
                raise RuntimeError("native TBE21 exec layout differs from selected flag")
        async_probe = getattr(module, "async_x", None)
        if async_probe is None:
            if self._async_x:
                raise RuntimeError("selected TBE21 async-X lacks native compiled-flag proof")
        else:
            actual_async = async_probe()
            if type(actual_async) is not bool or actual_async != self._async_x:
                raise RuntimeError("native TBE21 async-X differs from selected flag")
        tile_probe = getattr(module, "tile128", None)
        selector = getattr(module, "tile_for", None)
        if tile_probe is None:
            if self._tile128:
                raise RuntimeError("selected TBE21 tile128 lacks native compiled-flag proof")
        else:
            actual_tile128 = tile_probe()
            if type(actual_tile128) is not bool or actual_tile128 != self._tile128:
                raise RuntimeError("native TBE21 tile128 differs from selected flag")
        if self._tile128 and selector is None:
            raise RuntimeError("selected TBE21 tile128 lacks native compiled-selector proof")
        if selector is not None:
            for rows in ((64, 65, 128, 256, 512) if self._row_geometry else (64, 65, 128, 256)):
                expected = 128 if self._tile128 and rows > 64 else 64
                actual_tile = selector(rows)
                if type(actual_tile) is not int or actual_tile != expected:
                    raise RuntimeError("native TBE21 tile selector differs from selected policy")
        packed_probe = getattr(module, "packed_scan2", None)
        if packed_probe is None:
            if self._packed_scan2:
                raise RuntimeError("selected TBE21 paired scan lacks native compiled-flag proof")
        else:
            actual_packed = packed_probe()
            if type(actual_packed) is not bool or actual_packed != self._packed_scan2:
                raise RuntimeError("native TBE21 packed_scan2 differs from selected flag")
        pair_probe = getattr(module, "pair_vote", None)
        if pair_probe is None:
            if self._pair_vote:
                raise RuntimeError("selected TBE21 pair_vote lacks native compiled-flag proof")
        else:
            actual_pair = pair_probe()
            if type(actual_pair) is not bool or actual_pair != self._pair_vote:
                raise RuntimeError("native TBE21 pair_vote differs from selected flag")
        probe = getattr(module, "kernel_variant", None)
        if probe is None:
            if self._kernel_variant != 2:
                raise RuntimeError("selected TBE21 schedule lacks native compiled-variant proof")
        else:
            actual = probe()
            if type(actual) is not int or actual != self._kernel_variant:
                raise RuntimeError("native TBE21 kernel schedule differs from selected schedule")

    def _check_tier_native(self, module):
        names = ("tier_prefix", "tier_prefetch", "tier_prefix_wg1", "tier_prefetch_wg1")
        selected = (self._tier_prefix, self._tier_prefetch, self._tier_wg1, self._tier_prefetch_wg1)
        for name, expected in zip(names, selected):
            getter = getattr(module, name, None)
            if not callable(getter):
                if self._tier_prefix:
                    raise RuntimeError("selected TBE21 tier-prefix lacks native compiled-flag proof: " + name)
                continue
            actual = getter()
            if type(actual) is not bool or actual != expected:
                raise RuntimeError("native TBE21 " + name + " differs from selected flag")
        abi = getattr(module, "tier_prefix_host_abi", None)
        if not callable(abi):
            if self._tier_prefix:
                raise RuntimeError("TBE21 tier-prefix lacks native optional descriptor ABI proof")
            actual_abi = False
        else:
            actual_abi = abi()
        if type(actual_abi) is not bool or actual_abi != self._tier_prefix:
            raise RuntimeError("native TBE21 optional descriptor ABI differs from selected flag")
        if self._tier_prefix:
            if not callable(getattr(module, "make_tier_headers", None)):
                raise RuntimeError("selected TBE21 tier-prefix lacks native metadata builder")
            if not callable(getattr(module, "bi_decode_prefix_out", None)):
                raise RuntimeError("selected TBE21 tier-prefix lacks native prefix decoder")

    def _assert_tier_selection(self):
        if selected_exec_layout() != self._exec_layout:
            raise RuntimeError("TBE21 exec-layout selection changed after backend load; restart required")
        selected = selected_tier_prefix()
        current = (selected, os.environ.get("BI_T21_TPF_PREFETCH", "0") == "1",
                   os.environ.get("BI_T21_TPF_WG1", "0") == "1",
                   os.environ.get("BI_T21_TPF_PREFETCH_WG1", "0") == "1")
        frozen = (self._tier_prefix, self._tier_prefetch, self._tier_wg1, self._tier_prefetch_wg1)
        if current != frozen or (self._tier_prefix and selected_row_geometry() != self._row_geometry):
            raise RuntimeError("TBE21 tier-prefix selection changed after backend load; restart required")

    def prepare_arrays(self, N, K, arrs, S):
        self._assert_tier_selection()
        if not self._tier_prefix:
            return arrs
        if len(arrs) == 8:
            return arrs
        if len(arrs) != 7:
            raise ValueError("TBE21 tier-prefix requires seven base arrays or eight prepared arrays")
        import torch
        if torch.cuda.is_available():
            # A failed capture query on an available CUDA runtime is ambiguous: fail closed
            # rather than risk constructing metadata inside an active graph capture.
            capturing = bool(torch.cuda.is_current_stream_capturing())
        else:
            capturing = False
        if capturing:
            raise RuntimeError("TBE21 tier metadata must be prepared before CUDA graph capture")
        headers = self._module.make_tier_headers(int(N), int(K), arrs, int(S))
        if (not isinstance(headers, torch.Tensor) or headers.dtype != torch.int32
                or tuple(headers.shape) != (int(N), int(K) // 128)
                or headers.device != arrs[0].device or not headers.is_contiguous()):
            raise RuntimeError("native TBE21 tier metadata builder returned malformed tensor")
        return list(arrs) + [headers]

    def bi_decode_prefix_out(self, fmt, N, K, arrs, out, S, wg=2):
        self._assert_tier_selection()
        if not self._tier_prefix:
            raise RuntimeError("TBE21 prefix decode requires BI_T21_TIER_PREFIX=1")
        if type(wg) is not int or wg not in ((1, 2) if self._tier_wg1 else (2,)):
            raise ValueError("TBE21 tier-prefix decode geometry is not enabled")
        prepared = self.prepare_arrays(N, K, arrs, S)
        entry = self._module.bi_decode_prefix_out
        return entry(fmt, N, K, prepared, out, S, wg)

    @property
    def __file__(self):
        return getattr(self._module, "__file__", None)

    def bi_gemm_out(self, x, M, fmt, kqt, N, K, arrs, bias, y, S, ws, tile):
        self._assert_tier_selection()
        if self._tier_prefix and len(arrs) != 8:
            raise RuntimeError("TBE21 tier-prefix GEMM requires metadata prepared before the hot call")
        if int(fmt) != _FORMAT_TBE21:
            raise ValueError("TBE21 v2 backend accepts only format code 5")
        if int(kqt) != 0:
            raise ValueError("TBE21 v2 backend requires kqt=0")
        if int(tile) != self.tile_for(M):
            raise ValueError("TBE21 tile differs from compiled row policy")
        return self._module.bi_gemm_out(x, M, fmt, kqt, N, K, arrs, bias, y, S, ws, tile)

    def tile_for(self, rows) -> int:
        expected = 128 if self._tile128 and int(rows) > 64 else 64
        selector = getattr(self._module, "tile_for", None)
        if selector is not None:
            actual = selector(int(rows))
            if type(actual) is not int or actual != expected:
                raise RuntimeError("native TBE21 tile selector differs from selected policy")
        return expected

    def bi_decode_out(self, fmt, N, K, arrs, out, S, wg=1):
        """Decode through the selected private DSO, never the legacy oracle.

        Geometry is compiled into this backend; the public default wg=1 means
        use that selected geometry, not override its warp count.
        """
        if int(fmt) != _FORMAT_TBE21:
            raise ValueError("TBE21 v2 decode accepts only format code 5")
        self._assert_tier_selection()
        if type(wg) is not int or wg != 1:
            raise ValueError("TBE21 v2 decode uses compiled geometry; wg override unsupported")
        if self._tier_prefix:
            return self.bi_decode_prefix_out(fmt, N, K, arrs, out, S, wg=2)
        entry = getattr(self._module, "bi_decode_out", None)
        if entry is None:
            raise RuntimeError("selected TBE21 backend lacks its native decode oracle")
        return entry(fmt, N, K, arrs, out, S)

    def bi_decode_exec_out(self, fmt, N, K, arrs, out, S, rows):
        """Decode through the native execution-layout oracle for exact row tiles.

        This is deliberately a separate, capability-gated entry point.  It never
        falls back to the ordinary decoder: an absent or malformed native proof
        is an error.
        """
        if not self._exec_layout:
            raise RuntimeError("TBE21 exec decode requires BI_T21_EXEC_LAYOUT=1")
        self._assert_tier_selection()
        if not selected_exec_layout():
            raise RuntimeError("TBE21 exec-layout selection is not enabled")
        oracle_probe = getattr(self._module, "exec_decode_oracle", None)
        if not callable(oracle_probe):
            raise RuntimeError("selected TBE21 backend lacks exec decode oracle capability proof")
        oracle_enabled = oracle_probe()
        if type(oracle_enabled) is not bool or oracle_enabled is not True:
            raise RuntimeError("native TBE21 exec decode oracle capability is not exact")
        entry = getattr(self._module, "bi_decode_exec_out", None)
        if not callable(entry):
            raise RuntimeError("selected TBE21 backend lacks native exec decoder")
        if type(fmt) is not int or fmt != _FORMAT_TBE21:
            raise ValueError("TBE21 exec decode accepts only format code 5")
        int32_max = 2**31 - 1
        for name, value in (("N", N), ("K", K)):
            if type(value) is not int or not 1 <= value <= int32_max:
                raise ValueError(f"TBE21 exec decode {name} must be a positive int32")
        if K % 256 != 0:
            raise ValueError("TBE21 exec decode K must be divisible by 256")
        if type(S) is not int or not 1 <= S <= K // 128:
            raise ValueError("TBE21 exec decode S must be in 1..K/128")
        if type(rows) is not int or rows not in (64, 128):
            raise ValueError("TBE21 exec decode rows must be exactly 64 or 128")
        if type(arrs) is not list or len(arrs) != 7:
            raise ValueError("TBE21 exec decode requires exactly seven arrays")
        before = self.backend_identity()
        result = entry(fmt, N, K, arrs, out, S, rows)
        if self.backend_identity() != before:
            raise RuntimeError("TBE21 exec decode changed backend identity during native delegation")
        return result

    def kernel_attrs(self, rows: int, *, device: int = 0) -> dict:
        """Query this module's actual selected GEMM, without a launch or legacy fallback.

        local_bytes is CUDA's per-thread local memory size, not a spill-only count.
        Device identity belongs to the caller's surrounding device-qualified receipt.
        """
        if type(rows) is not int or not 1 <= rows <= 2**31 - 1:
            raise ValueError("resource query rows must be a positive int32")
        if type(device) is not int or not 0 <= device <= 127:
            raise ValueError("resource query device must be a nonnegative DeviceIndex")
        selected = (selected_kernel_variant(), selected_packed_scan2(),
                    selected_tile128(), selected_async_x(), selected_row_geometry(), selected_exec_layout(), selected_scan4(), selected_wide128(), selected_pipe128(), selected_packed_scan4(), selected_pair_vote(), selected_tier_prefix(), os.environ.get("BI_T21_TPF_PREFETCH", "0") == "1", os.environ.get("BI_T21_TPF_WG1", "0") == "1", os.environ.get("BI_T21_TPF_PREFETCH_WG1", "0") == "1")
        frozen = (self._kernel_variant, self._packed_scan2, self._tile128, self._async_x, self._row_geometry, self._exec_layout, self._scan4, self._wide128, self._pipe128, self._packed_scan4, self._pair_vote, self._tier_prefix, self._tier_prefetch, self._tier_wg1, self._tier_prefetch_wg1)
        if selected != frozen:
            raise RuntimeError("resource query selection changed after backend load")
        for name, value in zip(("kernel_variant", "packed_scan2", "tile128", "async_x", "row_geometry", "exec_layout", "scan4", "wide128", "pipe128", "packed_scan4", "pair_vote", "tier_prefix", "tier_prefetch", "tier_prefix_wg1", "tier_prefetch_wg1"), frozen):
            getter = getattr(self._module, name, None)
            optional_off_proofs = ("row_geometry", "exec_layout", "scan4", "wide128", "pipe128", "packed_scan4", "pair_vote")
            tier_names = ("tier_prefix", "tier_prefetch", "tier_prefix_wg1", "tier_prefetch_wg1")
            tier_is_off = not any(frozen[-4:])
            if (name in optional_off_proofs or name in tier_names and tier_is_off) and getter is None and value is False:
                continue
            actual = getter() if callable(getter) else None
            if type(actual) is not type(value) or actual != value:
                raise RuntimeError("resource query lacks matching compiled " + name + " proof")
        query = getattr(self._module, "bi_kernel_attrs", None)
        if not callable(query):
            raise RuntimeError("isolated TBE21 resource query unavailable; no legacy fallback")
        tile = self.tile_for(rows)
        wg = tile // 64
        mt = (3 if rows <= 96 else 4) if wg == 2 else (4 if rows >= 64 else (rows + 15) // 16)
        if self._exec_layout and 49 <= rows <= 64:
            mt, wg = 2, 2
        elif self._exec_layout and 97 <= rows <= 128:
            mt, wg = 8, 1
        elif self._row_geometry and rows == 64:
            mt, wg = 2, 2
        elif self._row_geometry and 129 <= rows <= 512:
            mt, wg = 8, 2
        if self._wide128 and rows == 128:
            mt, wg = 2, 4
        before = self.backend_identity()
        if not before.get("binary_sha256"):
            raise RuntimeError("resource query requires loaded native binary identity")
        raw = query(rows, device)
        if (not isinstance(raw, (list, tuple)) or len(raw) != 5
                or any(type(x) is not int or x < 0 for x in raw)
                or raw[0] == 0 or raw[3] == 0 or raw[4] != 256 * wg):
            raise RuntimeError("isolated TBE21 resource query returned malformed attributes")
        if wg == 2 and mt == 3 and raw[3] != 43776:
            raise RuntimeError("isolated TBE21 resource query differs from selected96-row specialization")
        if self._pipe128 and rows == 128 and (raw[0] > 64 or raw[1] != 0 or raw[2] < 1):
            raise RuntimeError("TBE21 pipe128 resource gate requires <=64 registers, zero local bytes and a resident CTA")
        if self._wide128 and rows == 128 and raw[3] != (98560 if self._pipe128 else 52480):
            raise RuntimeError("isolated TBE21 resource query differs from selected wide128 geometry")
        if self._exec_layout and 49 <= rows <= 64 and raw[3] != 33024:
            raise RuntimeError("isolated TBE21 resource query differs from selected C64 specialization")
        if self._exec_layout and 49 <= rows <= 64 and (raw[0] > 64 or raw[1] != 0 or raw[2] < 2):
            raise RuntimeError("TBE21 C64 resource gate requires <=64 registers, zero local bytes and at least two CTAs")
        if self._exec_layout and 97 <= rows <= 128 and raw[3] != 49152:
            raise RuntimeError("isolated TBE21 resource query differs from selected M128 global-codebook specialization")
        if self._exec_layout and 97 <= rows <= 128 and (raw[0] > 128 or raw[1] != 0 or raw[2] < 2):
            raise RuntimeError("TBE21 M128 resource gate requires <=128 registers, zero local bytes and at least two CTAs")
        if self._row_geometry and (rows == 64 or 129 <= rows <= 512):
            expected_smem = 35072 if rows == 64 else 87296
            if raw[3] != expected_smem:
                raise RuntimeError("isolated TBE21 resource query differs from selected row geometry")
        if self.backend_identity() != before:
            raise RuntimeError("resource query source or native binary changed during measurement")
        return {"schema": "georefine.tbe21.kernel_resources.v1", "rows": rows,
                "device_index": device, "tile": tile, "mt": mt, "wg": wg,
                "backend_identity": before,
                "attributes": dict(zip(("regs", "local_bytes", "max_blocks_per_sm",
                                        "smem_bytes", "threads"), raw))}

    def backend_identity(self) -> dict:
        path = self.__file__
        identity = {
            "backend": "tbe21-v2",
            "module_path": path,
            "binary_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None,
            "source_bindings": source_bindings(),
            "pinned_source_sha256": dict(_PINNED_SOURCE_SHA256),
            "format_code": _FORMAT_TBE21,
            "kqt": 0,
            "spec": 0,
            "tile": 64,
        }

        if self._kernel_variant != 2:
            identity["kernel_variant"] = self._kernel_variant
        if self._packed_scan2:
            identity["packed_scan2"] = True
        if self._tile128:
            identity.pop("tile")
            identity.update(tile128=True, tile_policy=tile_policy())
        if self._async_x:
            identity["async_x"] = True
        if self._row_geometry:
            identity["row_geometry"] = True
        if self._exec_layout:
            identity["exec_layout"] = True
        if self._scan4:
            identity["scan4"] = True
        if self._wide128:
            identity["wide128"] = True
        if self._pipe128:
            identity["pipe128"] = True
        if self._packed_scan4:
            identity["packed_scan4"] = True
        if self._pair_vote:
            identity["pair_vote"] = True
        if self._tier_prefix:
            identity.update(tier_prefix=True, tier_prefetch=self._tier_prefetch,
                            tier_prefix_wg1=self._tier_wg1,
                            tier_prefetch_wg1=self._tier_prefetch_wg1,
                            tier_metadata_abi=_TIER_METADATA_ABI,
                            tier_decode_default_wg=2,
                            tier_decode_geometries=[1, 2] if self._tier_wg1 else [2])
        return identity


def adapt(module) -> BackendAdapter:
    """Wrap a native or mocked extension while enforcing the TBE21-only ABI."""
    return BackendAdapter(module)


def extension() -> BackendAdapter:
    """Load the private extension only after explicit opt-in; never fall back."""
    global _EXT
    if not enabled():
        raise RuntimeError(f"set {_FLAG}=1 before loading the TBE21 v2 backend")
    check_real_build()
    if _OVERLAP is not None:
        adapter = _OVERLAP.result()
        if (adapter._kernel_variant != selected_kernel_variant()
                or adapter._packed_scan2 != selected_packed_scan2()
                or adapter._tile128 != selected_tile128()
                or adapter._async_x != selected_async_x()
                or adapter._row_geometry != selected_row_geometry()
                or getattr(adapter, "_exec_layout", False) != selected_exec_layout()
                or adapter._scan4 != selected_scan4()
                or adapter._wide128 != selected_wide128()
                or getattr(adapter, "_pipe128", False) != selected_pipe128()
                or getattr(adapter, "_packed_scan4", False) != selected_packed_scan4()
                or getattr(adapter, "_pair_vote", False) != selected_pair_vote()
                or getattr(adapter, "_tier_prefix", False) != selected_tier_prefix()
                or getattr(adapter, "_tier_prefetch", False) != (os.environ.get("BI_T21_TPF_PREFETCH", "0") == "1")
                or getattr(adapter, "_tier_wg1", False) != (os.environ.get("BI_T21_TPF_WG1", "0") == "1")
                or getattr(adapter, "_tier_prefetch_wg1", False) != (os.environ.get("BI_T21_TPF_PREFETCH_WG1", "0") == "1")):
            raise RuntimeError("TBE21 schedule changed after startup build")
        return adapter
    if _EXT is not None:
        if (_EXT._kernel_variant != selected_kernel_variant()
                or _EXT._packed_scan2 != selected_packed_scan2()
                or _EXT._tile128 != selected_tile128()
                or _EXT._async_x != selected_async_x()
                or _EXT._row_geometry != selected_row_geometry()
                or getattr(_EXT, "_exec_layout", False) != selected_exec_layout()
                or _EXT._scan4 != selected_scan4()
                or _EXT._wide128 != selected_wide128()
                or getattr(_EXT, "_pipe128", False) != selected_pipe128()
                or getattr(_EXT, "_packed_scan4", False) != selected_packed_scan4()
                or getattr(_EXT, "_pair_vote", False) != selected_pair_vote()
                or getattr(_EXT, "_tier_prefix", False) != selected_tier_prefix()
                or getattr(_EXT, "_tier_prefetch", False) != (os.environ.get("BI_T21_TPF_PREFETCH", "0") == "1")
                or getattr(_EXT, "_tier_wg1", False) != (os.environ.get("BI_T21_TPF_WG1", "0") == "1")
                or getattr(_EXT, "_tier_prefetch_wg1", False) != (os.environ.get("BI_T21_TPF_PREFETCH_WG1", "0") == "1")):
            raise RuntimeError("TBE21 schedule changed after backend load; restart required")
        return _EXT
    with _LOCK:
        if _EXT is None:
            from glc_loader._jitenv import ensure_build_tools_on_path
            from glc_serve import bigemm as bi
            from glc_serve import kernel_cache
            from torch.utils.cpp_extension import load

            ensure_build_tools_on_path()
            arch = bi._arch_for_build()
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch)
            name, out, spec = kernel_build_plan()
            args = _ext_args(arch)
            module = kernel_cache.build(
                name, out, lambda: load(**args, build_directory=str(out), verbose=False),
                spec=spec,
            )
            _EXT = BackendAdapter(module)
    return _EXT


# This optional startup path uses a process group because Python threads cannot
# cancel torch/ninja/nvcc safely. The existing builder above is unchanged.
_COMPILER_RESERVE_BYTES = 16 << 30
_RUNTIME_RESERVE_BYTES = 8 << 30
_SERIAL_CONTEXT_RESERVE_BYTES = 32 << 30
_BUILD_TIMEOUT_S = 600.0
_OVERLAP_SERVING_TARGETS = ("miv_gemv", "fastdec", "bidec", "bi_gemm")


def _serving_dependency_plans(device):
    """Exact current plans for the generic batch extensions shared with TBE21 startup."""
    from . import kernel_cache
    kernel_cache._set_device(device)
    rows = []
    for target in _OVERLAP_SERVING_TARGETS:
        name, build_dir, spec, _entry = kernel_cache._plan(target, device)
        rows.append({
            "target": target,
            "extension": name,
            "build_dir": str(build_dir),
            "cache_key": kernel_cache.spec_digest(spec),
            "source_spec": {
                "name": spec.name,
                "sources": [{"path": str(Path(path).resolve()),
                             "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                            for path in spec.sources],
                "headers": [{"path": str(path.resolve()),
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                            for path in kernel_cache._header_files(
                                [Path(path) for path in spec.include_dirs]
                                + [Path(path).parent for path in spec.sources])],
                "inline_sources_sha256": [hashlib.sha256(text.encode()).hexdigest()
                                           for text in spec.inline_sources],
                "include_dirs": [str(Path(path).resolve()) for path in spec.include_dirs],
                "flags": list(spec.flags),
                "arch": spec.arch,
                "extra": [list(item) for item in spec.extra],
            },
        })
    return rows


def _dependency_receipts(plans, records):
    """Bind successful generic prewarm records to current plan and on-disk binaries."""
    from . import kernel_cache
    if (not isinstance(records, list) or len(records) != len(plans)
            or any(not isinstance(record, dict) for record in records)):
        raise RuntimeError("overlap serving dependency receipts are missing or duplicated")
    by_target = {record.get("target"): record for record in records}
    if len(by_target) != len(plans) or set(by_target) != {plan["target"] for plan in plans}:
        raise RuntimeError("overlap serving dependency receipts are missing or duplicated")
    receipts = []
    for plan in plans:
        record = by_target[plan["target"]]
        binary = kernel_cache.library_path(plan["build_dir"], plan["extension"])
        if (record.get("ok") is not True or record.get("extension") != plan["extension"]
                or record.get("dir") != plan["build_dir"]
                or not kernel_cache.is_cached(plan["build_dir"], plan["extension"], plan["cache_key"])
                or not binary.is_file()):
            raise RuntimeError("overlap serving dependency build failed or is unstamped: " + plan["target"])
        receipts.append({**plan, "binary_path": str(binary),
                         "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                         "ok": True})
    return receipts


def validate_overlap_selection(args):
    selected = getattr(args, "context_overlap_tbe21_build", False)
    if type(selected) is not bool:
        raise ValueError("context_overlap_tbe21_build must be bool")
    if not selected:
        return False
    if (getattr(args, "runtime_format", None) != "tbe21"
            or not getattr(args, "package", None)
            or not getattr(args, "context_build_root", None)
            or getattr(args, "context_typed", False) or not enabled()
            or str(getattr(args, "device", "")).split(":", 1)[0] != "cuda"):
        raise ValueError("--context-overlap-tbe21-build requires Context v21 transcode, CUDA and BI_TBE21_V2=1")
    import sys
    if not sys.platform.startswith("linux"):
        raise ValueError("--context-overlap-tbe21-build requires Linux process-group cleanup")
    return True


def _available_memory_bytes():
    # MemAvailable includes reclaimable pages. Cgroup v2 headroom may be smaller.
    rows = Path("/proc/meminfo").read_text().splitlines()
    available = next(int(row.split()[1]) * 1024 for row in rows if row.startswith("MemAvailable:"))
    groups = Path("/proc/self/cgroup").read_text().splitlines()
    if any(row.split(":", 2)[1] and "memory" in row.split(":", 2)[1].split(",") for row in groups):
        raise RuntimeError("overlap memory estimate requires host or cgroup v2 accounting")
    group = next((row.split(":", 2)[2] for row in groups if row.startswith("0::")), "/")
    base = Path("/sys/fs/cgroup")
    candidate = base / group.lstrip("/")
    if not candidate.is_dir():
        candidate = base  # namespaced cgroup mount
    for directory in (candidate, *candidate.parents):
        if not directory.is_relative_to(base):
            break
        maximum, current = directory / "memory.max", directory / "memory.current"
        if maximum.is_file():
            limit = maximum.read_text().strip()
            if limit != "max":
                available = min(available, max(0, int(limit) - int(current.read_text())))
    return available


def overlap_memory_estimate(args):
    budget = getattr(args, "context_startup_ram_bytes", None)
    context = budget if getattr(args, "context_startup_workers", 0) else _SERIAL_CONTEXT_RESERVE_BYTES
    if type(context) is not int or context <= 0:
        raise ValueError("overlap requires a positive Context memory estimate")
    available = _available_memory_bytes()
    required = context + _COMPILER_RESERVE_BYTES + _RUNTIME_RESERVE_BYTES
    if available < required:
        raise RuntimeError(f"insufficient estimated overlap memory headroom: {available} < {required}")
    return {"context_budget_bytes_estimate": context,
            "compiler_reserve_bytes_estimate": _COMPILER_RESERVE_BYTES,
            "runtime_reserve_bytes_estimate": _RUNTIME_RESERVE_BYTES,
            "required_available_bytes_estimate": required,
            "available_bytes_at_start": available,
            "accounting": "estimate_not_rss"}




def record_cleanup_failure(error, cleanup):
    """Preserve the original failure and cleanup evidence on supported Python 3.10+."""
    note = getattr(error, "add_note", None)
    if callable(note):
        note(f"TBE21 compiler cleanup failed: {cleanup}")
    else:
        error.__cause__ = cleanup
        error.__suppress_context__ = True

def _live_group_members(pgid):
    """Non-zombie members of our Linux process group; inability to inspect fails closed."""
    members = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            # comm may contain spaces or parentheses; fields after its final ')' are stable.
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == pgid and fields[0] not in ("Z", "X"):
            members.add(int(entry.name))
    return members

class StartupBuild:
    """Owned compile process; success and failure are sticky across first uses."""

    def __init__(self, args):
        import json
        import subprocess
        import sys
        import time
        import uuid
        from . import kernel_cache
        self.lock = threading.Lock()
        self.value = None
        self.error = None
        self.process = None
        self.parent_fd = None
        self.memory = overlap_memory_estimate(args)
        kernel_cache._set_device(args.device)
        self.name, out, spec = kernel_build_plan()
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", spec.arch)
        self.device = str(args.device)
        self.serving_plans = _serving_dependency_plans(self.device)
        self.plan = {"name": self.name, "build_dir": str(out),
                     "key": kernel_cache.spec_digest(spec), "arch": spec.arch,
                     "source_bindings": source_bindings(), "device": self.device,
                     "serving_dependencies": self.serving_plans}
        self.path = Path(args.context_build_root).resolve() / ("tbe21-build-" + uuid.uuid4().hex)
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "request.json").write_text(json.dumps(self.plan, sort_keys=True))
        self.started = time.monotonic()
        read_fd, self.parent_fd = os.pipe()
        env = dict(os.environ, TORCH_CUDA_ARCH_LIST=spec.arch)
        env["MAX_JOBS"] = "1"
        # Preserve the exact imported package in source checkouts as well as wheels.
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            with (self.path / "compiler.log").open("wb") as log:
                self.process = subprocess.Popen([
                    sys.executable, "-m", "glc_serve.tbe21_v2_backend",
                    str(self.path), str(read_fd)], env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(read_fd,))
        except BaseException:
            os.close(self.parent_fd)
            self.parent_fd = None
            raise
        finally:
            os.close(read_fd)

    def _close_pipe(self):
        if self.parent_fd is not None:
            os.close(self.parent_fd)
            self.parent_fd = None

    def _stop(self):
        import signal
        import subprocess
        if self.process is None:
            self._close_pipe()
            return
        # Signal the owned group even if its leader already exited.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(self.process.pid, sig)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                continue
            if sig == signal.SIGKILL:
                break
        self._close_pipe()
        import time
        deadline = time.monotonic() + 5
        while True:
            members = _live_group_members(self.process.pid)
            if self.process.poll() is not None and not members:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f"owned TBE21 compiler group still has live members after SIGKILL: {sorted(members)}")
            time.sleep(0.05)

    def cancel(self):
        with self.lock:
            if self.value is not None:
                return
            if self.error is None:
                self.error = RuntimeError("TBE21 startup compile cancelled after loader failure")
            self._stop()

    def result(self):
        import importlib.util
        import json
        import time
        with self.lock:
            if self.error is not None:
                raise self.error
            if self.value is not None:
                return self.value
            try:
                remaining = max(0.001, _BUILD_TIMEOUT_S - (time.monotonic() - self.started))
                if self.process.wait(timeout=remaining) != 0:
                    raise RuntimeError(f"TBE21 startup compiler failed; see {self.path / 'compiler.log'}")
                if _live_group_members(self.process.pid):
                    raise RuntimeError("TBE21 compiler leader exited with live descendants")
                self._close_pipe()
                report = json.loads((self.path / "result.json").read_text())
                if report.get("plan") != self.plan or source_bindings() != self.plan["source_bindings"]:
                    raise RuntimeError("TBE21 startup compiler source or target drift")
                current_serving_plans = _serving_dependency_plans(self.device)
                if current_serving_plans != self.serving_plans:
                    raise RuntimeError("overlap serving dependency source or target drift")
                name, out, current_spec = kernel_build_plan()
                from . import kernel_cache
                if {"name": name, "build_dir": str(out), "key": kernel_cache.spec_digest(current_spec),
                    "arch": current_spec.arch, "source_bindings": source_bindings(),
                    "device": self.device, "serving_dependencies": current_serving_plans} != self.plan:
                    raise RuntimeError("TBE21 startup current source or target drift")
                serving_receipts = _dependency_receipts(
                    self.serving_plans, report.get("serving_dependency_records", []))
                if report.get("serving_dependency_receipts") != serving_receipts:
                    raise RuntimeError("TBE21 startup serving dependency receipt drift")
                build = report.get("kernel_build")
                if (not isinstance(build, dict) or build.get("ok") is not True
                        or build.get("extension") != self.name or build.get("dir") != self.plan["build_dir"]):
                    raise RuntimeError("TBE21 startup compiler successful build receipt missing")
                identity = report["backend_identity"]
                binary = kernel_cache.library_path(Path(self.plan["build_dir"]), self.name)
                if (identity.get("module_path") != str(binary)
                        or identity.get("binary_sha256") != hashlib.sha256(binary.read_bytes()).hexdigest()):
                    raise RuntimeError("TBE21 startup compiler binary drift")
                spec = importlib.util.spec_from_file_location(self.name, binary)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                adapter = BackendAdapter(module)
                if adapter.backend_identity() != identity:
                    raise RuntimeError("TBE21 startup loaded backend differs from compiler receipt")
                self.receipt = {"enabled": True, "joined_before_ready": True,
                    "plan": {key: self.plan[key] for key in
                             ("name", "build_dir", "key", "arch", "source_bindings")},
                    "backend_identity": identity,
                    "kernel_build": report["kernel_build"],
                    "serving_dependencies": serving_receipts,
                    "elapsed_s_measured": time.monotonic() - self.started,
                    "memory": self.memory}
                self.value = adapter
                return self.value
            except BaseException as exc:
                self.error = exc
                try:
                    self._stop()
                except BaseException as cleanup:
                    record_cleanup_failure(exc, cleanup)
                raise


def start_overlap(args):
    global _OVERLAP
    if not validate_overlap_selection(args):
        raise ValueError("TBE21 overlap was not explicitly selected")
    with _LOCK:
        if _OVERLAP is not None or _EXT is not None:
            raise RuntimeError("TBE21 overlap must start once before first backend use")
        _OVERLAP = StartupBuild(args)
        return _OVERLAP


def _compile_child(path, parent_fd):
    import json
    import signal
    from . import kernel_cache
    def watch_parent():
        # EOF also covers abrupt parent death; kill our whole compiler group.
        try:
            os.read(parent_fd, 1)
        finally:
            os.killpg(os.getpgrp(), signal.SIGKILL)
    threading.Thread(target=watch_parent, daemon=True).start()
    path = Path(path)
    request = json.loads((path / "request.json").read_text())
    kernel_cache._set_device(request.get("device"))
    name, out, spec = kernel_build_plan()
    actual = {"name": name, "build_dir": str(out), "key": kernel_cache.spec_digest(spec),
              "arch": spec.arch, "source_bindings": source_bindings(),
              "device": request.get("device"),
              "serving_dependencies": _serving_dependency_plans(request.get("device"))}
    if actual != request:
        raise RuntimeError("TBE21 child build plan differs from startup request")
    serving_records = kernel_cache.ensure_serving_kernels(
        "batch", device=actual["device"], only=_OVERLAP_SERVING_TARGETS)
    serving_receipts = _dependency_receipts(actual["serving_dependencies"], serving_records)
    adapter = extension()
    record = kernel_cache.results()[-1]
    if record.get("extension") != name or record.get("ok") is not True:
        raise RuntimeError("TBE21 compiler lacks successful kernel cache receipt")
    (path / "result.json").write_text(json.dumps({"plan": actual,
        "serving_dependency_records": serving_records,
        "serving_dependency_receipts": serving_receipts,
        "backend_identity": adapter.backend_identity(), "kernel_build": record}, sort_keys=True))


if __name__ == "__main__":
    import sys
    _compile_child(sys.argv[1], int(sys.argv[2]))
