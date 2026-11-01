"""BI-GEMM descriptors: batch-invariant tensor-core GEMM over every engine weight format.

``bi_csrc/bi_gemm.cuh`` holds the contract.  In one sentence: row m of ``y = x W^T`` is a
function of row m of x and of W only -- never of M, of the row's slot in the batch, or of
the other rows -- because the K reduction is one fixed sequence (128-wide chunks, a split
count S that depends on (N, K) alone, mma.m16n8k16 in increasing k, split partials added
in split order, one bf16 rounding).  Weights are decoded in the kernel by the MIV decode
functions, so TBE-coded weights give the bits of the bf16 parent and K-quant weights the
bits of gguf-py's dequantize, through the same GEMM.

``wrap(desc)`` turns a FastDecoder descriptor (``fastdec.Lin``, ``tbe_desc.TBELin``,
``q8serve.Q8Lin``, ``miv_kq.KQDesc``, ``q8serve.GroupLin``, ``q8serve.PermIn``,
``tbe2_desc.TBE2Lin``) into a ``BILin`` with ``__call__(x, y)`` for ANY number of rows.
Descriptors also accept ``x[..., K]`` and ``y[..., N]`` with matching leading axes.
The N-D boundary preserves logical row order, normalizes unsupported input layouts,
and commits temporary outputs back into strided destinations. Matrix calls retain
their original views and launch policy; no rows are padded to select a kernel.

Weight formats: ``bf16``, ``q8``, ``tbe`` (codec v1, rc7's served format) and ``tbe2`` (GLC codec
v2, docs/research/CODEC_V2_FORMAT_20261004.md; kernel design docs/serving/BIGEMM_V2_20261004.md).
``decode_weight`` runs the kernel's decode path alone (``bi_decode_kernel``: same CTA mapping and
split traversal as the GEMM) -- the device side of the G1 gate.
"""
from __future__ import annotations

import os
import hashlib
import json
import threading
from functools import lru_cache

from glc_serve import _winenv
from glc_serve import bigemm_tune as _tune
from pathlib import Path
from typing import List, Optional

import torch

_CSRC = Path(__file__).resolve().parent / "bi_csrc"
_EXT = None
_LOCK = threading.Lock()
BN, CH = 64, 128
FMT = {"bf16": 0, "q8": 1, "tbe": 2, "kq": 3, "tbe2": 4, "tbe21": 5, "tbe_cursor": 6}
#: The served weight formats `--weight-format` may select (bidec_serve / glc-bench).  "tbe" is
#: the default and is rc7 byte for byte: no "tbe2" code runs unless it is asked for.
WEIGHT_FORMATS = ("tbe", "tbe2", "tbe21")
WEIGHT_FORMAT_DEFAULT = "tbe"
#: Activation-row tile heights the kernel can be built for (``bi::MTILE * {1, MAXWG}``).
TILES = (64, 128)
#: Shipped default.  64 reproduces rc6 byte for byte: it is the same template instantiation
#: (WG=1) with the same launch bounds, grid and smem as before this knob existed.
TILE_DEFAULT = 64


@lru_cache(maxsize=1)
def context_build_enabled() -> bool:
    """Whether the generated Context distribution opts out of KQ/GGUF support."""
    marker = Path(__file__).with_name("CONTEXT_BUILD.json")
    if not marker.exists():
        return False
    try:
        spec = json.loads(marker.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"invalid Context build marker: {marker}") from exc
    if (not isinstance(spec, dict) or spec.get("schema") != "georefine-context-build-v1" or
            spec.get("context_only") is not True or
            spec.get("formats") != ["bf16", "tbe"]):
        raise RuntimeError(f"Context marker does not authorize BF16/TBE-only scope: {marker}")
    return True


def cursor_build_enabled() -> bool:
    """Experimental kernel builds only; the serving format default is unchanged."""
    raw = os.environ.get("BI_GEMM_TBE_CURSOR", "0")
    if raw not in ("0", "1"):
        raise ValueError("BI_GEMM_TBE_CURSOR must be 0 or 1")
    return raw == "1"


def _format_code(fmt: str) -> int:
    if fmt == "tbe_cursor":
        if not cursor_build_enabled():
            raise ValueError("tbe_cursor requires BI_GEMM_TBE_CURSOR=1 before extension load")
        if _EXT is not None and not _EXT.__name__.endswith("_cursor"):
            raise RuntimeError("restart the process to enable the TBE cursor kernel")
    if context_build_enabled() and fmt in ("q8", "kq"):
        raise ValueError(f"{fmt} is unavailable in the BF16/TBE-only Context build")
    return FMT[fmt]


@lru_cache(maxsize=None)
def _device_arch_for(device: int):
    """Return one CUDA device as ``(major, minor, SM count)``, cached by device index."""
    try:
        major, minor = torch.cuda.get_device_capability(device)
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        return int(major), int(minor), int(sms)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _device_arch():
    """Return the actual current CUDA device architecture, or ``None`` without CUDA."""
    try:
        if not torch.cuda.is_available():
            return None
        return _device_arch_for(torch.cuda.current_device())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _descriptor_arch(device):
    device = torch.device(device)
    if device.type != "cuda":
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _device_arch_for(index)


def _extension_on_device(loader, device):
    if torch.device(device).type == "cuda":
        with torch.cuda.device(device):
            return loader()
    return loader()


def tile_cap(default: int = TILE_DEFAULT) -> int:
    """The largest tile this process may use: ``BI_GEMM_TILE`` if set, else ``default``."""
    v = os.environ.get("BI_GEMM_TILE")
    t = int(v) if v else int(default)
    if t not in TILES:
        raise ValueError(f"BI_GEMM_TILE must be one of {TILES}, got {t}")
    return t


def tile_for(m: int, cap: int = TILE_DEFAULT, N: Optional[int] = None,
             K: Optional[int] = None, arch: Optional[tuple] = None) -> int:
    """The tile height to launch for an ``m``-row GEMM, given the per-engine cap.

    rc8: when the weight's shape ``(N, K)`` is given -- and every serving call gives it, see
    ``BILin.__call__`` -- the 128-row tile is launched only where this exact GPU's MEASURED tune
    table (``glc_serve.bigemm_tune``, run 6a T2) says it is faster for that shape at that row
    bucket. ``arch`` is an explicit override for tests; production reads the actual CUDA device.
    Run 6a measured the GDN input projection (16480x5120) at **0.931x** under the 128 tile at
    M=128: the shape-blind rule below made that GEMM slower.  A shape or bucket the table does
    not carry keeps the 64 tile.  The bits are the same either way (T1: 332,689,280 elements,
    0 differing), so this is a speed decision only.

    Without a shape the structural rule below applies unchanged (it is what the arithmetic
    model and the tile gate's forced comparisons reason about).

    THE DISPATCH RULE.  Weight bytes streamed per GEMM are ``W * ceil(m / tile)``: every CTA
    row of ``blockIdx.z`` decodes the whole weight set once.  MMA work is ``ceil(m/tile)``
    *full tiles* (the WG=2 kernel always runs MT=4 per group; rows past m are zeros).  So

        T(m, tile)  ~=  ceil(m/tile) * (W + mma(tile))

    and with the measured gate|up numbers (N=34816, K=5120, RTX PRO 6000 Blackwell SE,
    docs/serving/PREFILL_BUDGET_20261004.md): W ~= 238 us is the weight stream and
    mma(64) - mma(<=32) = 261.2 - 237.9 = 23.3 us is one 64-row tile's MMA.  Then

        m <= 64 :  ceil(m/64) = ceil(m/128) = 1, so the two tiles stream the SAME weights and
                   the 128 tile does mma(128) ~= 2*23.3 where the 64 tile does mma(m) <=
                   23.3.  The 64 tile is STRICTLY cheaper -- never widen below the knee.
        m >  64 :  ceil(m/64) >= ceil(m/128) + 1 for every m > 64 (m = 128q + r: 2q +
                   ceil(r/64) vs q + [r>0], and the gap is >= 1 once q >= 1 or r > 64).  Each
                   saved CTA row saves W ~= 238 us and costs at most mma(128) ~= 47 us of
                   zero-row MMA, i.e. ~5x more is saved than is spent.  Widen.

    Hence: 128 iff m > 64.  Expressed as the inequality rather than the constant so a future
    tile (256) needs no new branch.
    """
    m = int(m)
    if N is not None and K is not None:
        device_arch = arch if arch is not None else _device_arch()
        table = _tune.table_for(device_arch)
        if int(cap) >= 128 and _tune.wins_128(int(N), int(K), m, table):
            return 128
        return TILE_DEFAULT
    best = TILE_DEFAULT
    for t in TILES:
        if t > int(cap):
            continue
        if -(-m // t) < -(-m // best):
            best = t
    return best


def _arch_for_build() -> str:
    """The TORCH_CUDA_ARCH_LIST ``extension()`` builds with (env value, else the device's)."""
    arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if not arch:
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}.{minor}"
    return arch


def _t21_kernel_flags() -> list[str]:
    """Strict, default-off TBE21 compiler controls; flags also bind the cache identity."""
    values = (("BI_T21_SCAN", {"0", "1"}), ("BI_T21_TABREG", {"0", "1"}),
              ("BI_T21_V1_MINBLOCKS", {"2", "3"}))
    flags = []
    for name, allowed in values:
        value = os.environ.get(name)
        if value is not None:
            if value not in allowed:
                choices = " or ".join(sorted(allowed))
                raise ValueError(f"{name} must be exactly {choices}")
            flags.append(f"-D{name}={value}")
    return flags


def _ext_args(arch: str) -> dict:
    """Name, sources and flags of the BI-GEMM extension for ``arch`` -- exactly the arguments
    ``extension()`` passes to ``cpp_extension.load`` (minus the build directory)."""
    major, minor = arch.split(";")[0].replace("+PTX", "").split(".")
    here = Path(__file__).resolve().parent
    context_only = context_build_enabled()
    source_names = ("bi_gemm_dense.cu", "bi_torch.cpp")
    include_paths = [str(_CSRC), str(here / "miv_tbe_csrc")]
    if not context_only:
        source_names = ("bi_gemm_dense.cu", "bi_gemm_kq.cu", "bi_torch.cpp")
        include_paths.insert(1, str(here / "miv_kq_csrc"))
    scope_suffix = "_context" if context_only else ""
    scope_flag = ["-DGEOR_REFINED_CONTEXT_ONLY=1"] if context_only else []
    host_scope_flag = (["/DGEOR_REFINED_CONTEXT_ONLY=1"] if _winenv.is_windows() else scope_flag)
    if cursor_build_enabled():
        scope_suffix += "_cursor"
        scope_flag = scope_flag + ["-DGEOR_TBE_CURSOR=1"]
        host_scope_flag = host_scope_flag + (["/DGEOR_TBE_CURSOR=1"] if _winenv.is_windows()
                                             else ["-DGEOR_TBE_CURSOR=1"])
    return dict(name=f"bi_gemm_sm{major}{minor}{scope_suffix}",
                sources=[str(_CSRC / s) for s in source_names],
                extra_include_paths=include_paths,
                extra_cflags=_winenv.host_cflags() + host_scope_flag,
                # -Xptxas=-v under BI_GEMM_VERBOSE: the 128-row tile runs 512 threads
                # at one CTA/SM, so the per-thread register count (budget 128) and any
                # spill is the one occupancy fact only the compiler can state.  See
                # docs/serving/BIGEMM_TILE128_20261004.md section 4.
                extra_cuda_cflags=(["-O3", "--fmad=true", "-lineinfo", "-std=c++17"] + scope_flag
                                   + _t21_kernel_flags()
                                   + (["-Xptxas=-v"]
                                      if int(os.environ.get("BI_GEMM_VERBOSE", "0")) else [])))


def kernel_build_plan():
    """(extension name, build directory, kernel_cache.ExtensionSpec) for the current arch."""
    from glc_serve import kernel_cache

    arch = _arch_for_build()
    a = _ext_args(arch)
    spec = kernel_cache.ExtensionSpec(
        name=a["name"], sources=tuple(a["sources"]), include_dirs=tuple(a["extra_include_paths"]),
        flags=tuple(a["extra_cflags"]) + ("--cuda--",) + tuple(a["extra_cuda_cflags"]), arch=arch)
    # BI_GEMM_BUILD_DIR (operator override) is used as given; otherwise the versioned
    # <cache root>/<torch>-cu<cuda>-<arch>-py<ver>/<name>/<key>/ directory.  bidec.ext() builds
    # into this directory's parent / "bidec_build".
    return a["name"], kernel_cache.resolve_build_dir(spec, "BI_GEMM_BUILD_DIR"), spec


def _build_dir() -> Path:
    return kernel_build_plan()[1]


def extension():
    global _EXT
    if _EXT is not None:
        return _EXT
    with _LOCK:
        if _EXT is None:
            # G1: a console script invoked by absolute path does not put the venv's bin/ on
            # PATH, and cpp_extension resolves ninja with shutil.which.  Put the declared,
            # already-installed tool where the build will actually look for it.
            from glc_loader._jitenv import ensure_build_tools_on_path

            ensure_build_tools_on_path()
            from torch.utils.cpp_extension import load
            from glc_serve import kernel_cache

            arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
            if not arch:
                major, minor = torch.cuda.get_device_capability()
                arch = f"{major}.{minor}"
                os.environ["TORCH_CUDA_ARCH_LIST"] = arch
            os.environ.setdefault("MAX_JOBS", "8")
            name, build_dir, spec = kernel_build_plan()
            a = _ext_args(arch)
            _EXT = kernel_cache.build(
                name, build_dir,
                lambda: load(name=a["name"], sources=a["sources"],
                             extra_include_paths=a["extra_include_paths"],
                             extra_cflags=a["extra_cflags"],
                             extra_cuda_cflags=a["extra_cuda_cflags"],
                             build_directory=str(build_dir),
                             verbose=bool(int(os.environ.get("BI_GEMM_VERBOSE", "0")))),
                spec=spec)
    return _EXT


def split_for(N: int, K: int, sms: Optional[int] = None, *, device=None) -> int:
    """Split-K count: a function of (N, K) and the card ONLY -- never of M (batch invariance)."""
    if sms is None:
        sms = torch.cuda.get_device_properties(0 if device is None else device).multi_processor_count
    ctas = (N + BN - 1) // BN
    s = max(1, min(8, (2 * sms) // max(ctas, 1)))
    nch = K // CH
    while s > 1 and nch // s < 4:
        s -= 1
    return s


class Workspace:
    """One fp32 split-K workspace shared by every BILin of an engine (calls are serialised on
    one stream)."""

    def __init__(self, device):
        self.buf = torch.empty(0, dtype=torch.float32, device=device)
        # An unindexed "cuda" argument must not migrate if current_device changes.
        self.device = self.buf.device
        self._frozen = False

    def get(self, n: int) -> torch.Tensor:
        if self.buf.numel() < n:
            if self._frozen:
                raise RuntimeError('captured workspace cannot grow; reserve the full row ceiling before capture')
            self.buf = torch.empty(n, dtype=torch.float32, device=self.device)
        return self.buf

    def reserve(self, n: int) -> None:
        self.get(n)

    def freeze(self) -> None:
        """Keep the pre-reserved address valid for the lifetime of captured graphs."""
        self._frozen = True


_EMPTY = {}


def _empty(dev):
    if dev not in _EMPTY:
        _EMPTY[dev] = torch.empty(0, dtype=torch.uint8, device=dev)
    return _EMPTY[dev]


def tbe2_opt_enabled() -> bool:
    """Explicit experimental selection; malformed configuration never falls back."""
    value = os.environ.get("BI_TBE2_OPT", "0")
    if value not in ("0", "1"):
        raise ValueError("BI_TBE2_OPT must be exactly 0 or 1")
    return value == "1"


_BACKEND_IDENTITY_LOCK = threading.Lock()
_BACKEND_IDENTITIES = {}


def _optimized_backend_identity(module) -> dict:
    """Hash each immutable loaded module/source set once, including concurrent builds."""
    with _BACKEND_IDENTITY_LOCK:
        if module not in _BACKEND_IDENTITIES:
            from . import bigemm_tbe2_opt as opt

            path = getattr(module, "__file__", None)
            _BACKEND_IDENTITIES[module] = dict(module_path=path,
                binary_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None,
                source_bindings=opt.source_bindings())
        details = _BACKEND_IDENTITIES[module]
        return dict(details, source_bindings=dict(details["source_bindings"]))


def _call_nd_linear(descriptor, x, y, tile):
    """Flatten leading axes once; retain the compressed matrix path and row order."""
    from .linear_layout import prepare_linear_rows

    rows, output, commit = prepare_linear_rows(x, y, descriptor.K, descriptor.N)
    if rows.shape[0]:
        if isinstance(descriptor, BIPerm):
            descriptor.reserve(int(rows.shape[0]))
        descriptor(rows, output, tile)
    commit()


class BILin:
    """One weight of one format through BI-GEMM."""

    def __init__(self, fmt: str, N: int, K: int, arrs: List[torch.Tensor], ws: Workspace, *,
                 kqt: int = 0, bias: Optional[torch.Tensor] = None, src=None, S: Optional[int] = None,
                 tile: Optional[int] = None):
        self.fmt, self.N, self.K, self.arrs, self.ws, self.kqt, self.bias = fmt, int(N), int(K), arrs, ws, kqt, bias
        self.device = arrs[0].device
        if any(a.device != self.device for a in arrs) or ws.buf.device != self.device:
            raise ValueError("descriptor streams and workspace must share one device")
        if bias is not None and bias.device != self.device:
            raise ValueError("descriptor bias must share its weight device")
        self.S = int(S) if S else split_for(self.N, self.K, device=self.device)
        self.src = src
        self.kind = f"bi-{fmt}"
        # The per-weight tile CAP, not the tile: the launch height is chosen per call from the
        # row count by `tile_for`, because the right tile depends on M and nothing else.
        if tile is not None and int(tile) not in TILES:
            raise ValueError(f"tile must be one of {TILES}, got {tile}")
        self.tile_cap = int(tile) if tile is not None else tile_cap()
        enabled = tbe2_opt_enabled()
        self._backend = None
        self._backend_kind = "legacy"
        self._backend_calls = 0
        self._backend_tile_calls = {64: 0, 128: 0}
        self._backend_details = {}
        # A dedicated Context-only DSO; never replace the legacy global extension.
        if fmt == "tbe21":
            from . import tbe21_v2_backend as v2

            v2.selected_tier_prefix()
            if v2.enabled():
                if enabled:
                    raise ValueError("BI_TBE21_V2 conflicts with BI_TBE2_OPT")
                if self.kqt != 0:
                    raise ValueError("BI_TBE21_V2 requires kqt=0")
                if v2.selected_tile128():
                    # Format-specific opt-in policy supersedes the legacy descriptor cap;
                    # it does not change the batcher's legacy row-budget quantum.
                    self.tile_cap = 128
                elif self.tile_cap != 64:
                    raise ValueError("BI_TBE21_V2 requires tile cap 64 and kqt=0")
                self._backend = _extension_on_device(v2.extension, self.device)
                if v2.selected_tier_prefix():
                    self.arrs = self._backend.prepare_arrays(self.N, self.K, self.arrs, self.S)
                self._backend_kind = "tbe21-v2"
                self._backend_details = _extension_on_device(self._backend.backend_identity, self.device)
                self._backend_details.pop("backend", None)
        if enabled and fmt == "tbe21":
            raise ValueError("BI_TBE2_OPT supports unchanged TBE2 only, not TBE21")
        if enabled and fmt == "tbe2":
            from . import bigemm_tbe2_opt as opt

            # Constructor runs before BatchDecoder.capture(). Never compile inside
            # graph capture, replace the global extension, or silently fall back.
            self._backend = _extension_on_device(opt.extension, self.device)
            self._backend_kind = "tbe2-opt"
            self._backend_details = _extension_on_device(
                lambda: _optimized_backend_identity(self._backend), self.device)

    def backend_identity(self) -> dict:
        """Selected implementation plus successful Python dispatches (not graph replays)."""
        details = dict(self._backend_details)
        if "source_bindings" in details:
            details["source_bindings"] = dict(details["source_bindings"])
        if self.tile_cap >= 128 and self.fmt in ("bf16", "tbe"):
            from .bigemm_measured_tiles import policy_applies, policy_identity
            if policy_applies(self.fmt, self.tile_cap, _descriptor_arch(self.device)):
                details["tile_dispatch_policy"] = policy_identity()
        if details.get("tile128") is True:
            details["tile_policy"] = dict(details["tile_policy"])
            details["tile_dispatch_counts"] = {str(k): v for k, v in self._backend_tile_calls.items()}
        return dict(backend=self._backend_kind, format=self.fmt, N=self.N, K=self.K,
                    S=self.S, tile_cap=self.tile_cap, dispatch_calls=self._backend_calls,
                    graph_replays_counted=False, **details)

    @property
    def bytes(self) -> int:
        return int(getattr(self.src, "bytes", 0))

    @property
    def coded_bytes(self) -> int:
        return int(getattr(self.src, "coded_bytes", self.bytes))

    def reserve(self, max_m: int) -> None:
        if self.S > 1:
            self.ws.reserve(self.S * max_m * self.N)

    def __call__(self, x: torch.Tensor, y: torch.Tensor, tile: Optional[int] = None) -> None:
        """``tile`` FORCES a launch height, bypassing the dispatch rule.

        Only the tile gate uses it: the dispatch rule picks 64 at or below the knee, so a
        bitwise 64-vs-128 comparison at M <= 64 would otherwise compare a tiling with itself.
        Serving never passes it.
        """
        if x.device != self.device or y.device != self.device:
            raise ValueError("activations and outputs must share the descriptor device")
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] == 0:
            return _call_nd_linear(self, x, y, tile)
        m = int(x.shape[0])
        ws = self.ws.get(self.S * m * self.N) if self.S > 1 else self.ws.buf
        if self._backend_details.get("tile128") is True:
            expected = self._backend.tile_for(m)
            if tile is not None and int(tile) != expected:
                raise ValueError("forced tile differs from selected TBE21 row policy")
            t = expected
        elif tile is None:
            measured_tile = None
            if self.tile_cap >= 128 and self.fmt in ("bf16", "tbe"):
                from .bigemm_measured_tiles import select_measured_tile
                measured_tile = select_measured_tile(
                    self.fmt, self.N, self.K, m, self.tile_cap, _descriptor_arch(self.device))
            arch = _descriptor_arch(self.device)
            t = (measured_tile if measured_tile is not None else
                 tile_for(m, self.tile_cap, self.N, self.K, arch=arch or (0, 0, 0)))
        else:
            t = int(tile)
        backend = (self._backend if self._backend is not None else
                   _extension_on_device(extension, self.device))
        backend.bi_gemm_out(x, m, _format_code(self.fmt), self.kqt, self.N, self.K, self.arrs, self.bias, y,
                            self.S, ws, t)
        self._backend_calls += 1
        if self._backend_details.get("tile128") is True:
            self._backend_tile_calls[t] += 1


def backend_census(descriptors) -> dict:
    """Inspect nested descriptors without loading or changing a backend."""
    rows, unknown = [], []
    def visit(d):
        if isinstance(d, BILin) or getattr(d, "kind", None) == "bhe1":
            rows.append(d.backend_identity())
        elif isinstance(d, BIGroup):
            for member, _n in d.members:
                visit(member)
        elif isinstance(d, BIPerm):
            visit(d.d)
        else:
            unknown.append(dict(type=type(d).__module__ + "." + type(d).__qualname__,
                                kind=getattr(d, "kind", None)))
    for descriptor in descriptors:
        visit(descriptor)
    return dict(descriptor_count=len(rows), unknown_descriptor_count=len(unknown),
                coverage_complete=not unknown, unknown_descriptors=unknown,
                optimized_tbe2_descriptors=sum(r['backend']=='tbe2-opt' for r in rows),
                optimized_tbe21_descriptors=sum(r['backend']=='tbe21-v2' and r['format']=='tbe21' for r in rows),
                optimized_python_dispatches=sum(r['dispatch_calls'] for r in rows if r['backend'] in ('tbe2-opt', 'tbe21-v2')),
                graph_replays_counted=False, descriptors=rows)


def tbe21_tile_policy_from_census(census) -> Optional[dict]:
    rows = [row for row in census.get("descriptors", []) if row.get("format") == "tbe21"]
    selected = [row for row in rows if row.get("tile128") is True]
    if not selected:
        return None
    from . import tbe21_v2_backend as v2
    policy = v2.tile_policy()
    if len(selected) != len(rows) or any(row.get("tile_policy") != policy for row in selected):
        raise RuntimeError("mixed TBE21 tile policy in actual descriptor census")
    return policy


class BIGroup:
    kind = "bi-group"

    def __init__(self, members):
        self.members = list(members)            # [(BILin|BIPerm, n)]
        self.N = sum(n for _, n in self.members)
        self.K = self.members[0][0].K

    @property
    def bytes(self):
        return sum(d.bytes for d, _ in self.members)

    @property
    def coded_bytes(self):
        return sum(d.coded_bytes for d, _ in self.members)

    def reserve(self, max_m):
        for d, _ in self.members:
            d.reserve(max_m)

    def __call__(self, x, y, tile=None):
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] == 0:
            return _call_nd_linear(self, x, y, tile)
        r = 0
        for d, n in self.members:
            d(x, y[:, r:r + n], tile)
            r += n


class BIPerm:
    kind = "bi-perm"

    def __init__(self, d, perm: torch.Tensor):
        self.d, self.perm = d, perm
        self.N, self.K = d.N, d.K
        self.buf = torch.empty(0, self.K, dtype=torch.bfloat16, device=perm.device)

    @property
    def bytes(self):
        return self.d.bytes

    @property
    def coded_bytes(self):
        return self.d.coded_bytes

    def reserve(self, max_m):
        if self.buf.shape[0] < max_m:
            self.buf = torch.empty(max_m, self.K, dtype=torch.bfloat16, device=self.perm.device)
        self.d.reserve(max_m)

    def __call__(self, x, y, tile=None):
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] == 0:
            return _call_nd_linear(self, x, y, tile)
        m = int(x.shape[0])
        torch.index_select(x, 1, self.perm, out=self.buf[:m])     # per-row gather: M-invariant
        self.d(self.buf[:m], y, tile)


def tbe_escidx(t, *, max_chunk_tiles: int = 262144) -> torch.Tensor:
    """[N][K/128] index in ``esc`` of each 128-column chunk's first escape, from the per-tile
    escape counts, anchored at TBELin's per-row slice-0 base; cross-checked against every
    other slice base TBELin computed (a load-time consistency gate)."""
    from . import miv_gemv as mg

    N, K, wpr = t.N, t.K, t.wpr
    tpr = K // 64
    if N <= 0 or K <= 0 or K % 128 or wpr <= 0 or tpr % wpr:
        raise ValueError("invalid TBE escape-index geometry")
    if (isinstance(max_chunk_tiles, bool) or not isinstance(max_chunk_tiles, int)
            or not tpr <= max_chunk_tiles <= 262144):
        raise ValueError("escape-index budget must hold one row and be <= 262144 tiles")
    if t.planes.numel() != N * tpr * 6 or t.escbase.numel() != N * wpr:
        raise ValueError("TBE escape-index source geometry mismatch")
    rows_per_chunk = max_chunk_tiles // tpr
    planes = t.planes.view(-1)
    bases = t.escbase.view(N, wpr)
    out = torch.empty((N, tpr // 2), dtype=torch.int32, device=planes.device)
    counter = mg.extension().tbe_tile_escapes
    for row0 in range(0, N, rows_per_chunk):
        row1 = min(N, row0 + rows_per_chunk)
        cnt = counter(planes[row0*tpr*6:row1*tpr*6]).to(torch.int64).view(row1-row0, tpr)
        if bool(((cnt < 0) | (cnt > 64)).any().item()):
            raise ValueError("invalid TBE per-tile escape count")
        excl = torch.cumsum(cnt, 1) - cnt
        base = bases[row0:row1].to(torch.int64)
        for s in range(1, wpr):
            if not torch.equal(base[:, 0] + excl[:, s * (tpr // wpr)], base[:, s]):
                raise ValueError("TBE escape prefix disagrees with TBELin.escbase")
        idx = base[:, :1] + excl[:, ::2]
        if int(idx.min()) < 0 or int(idx.max()) >= 2 ** 31:
            raise ValueError("escape index exceeds int32")
        out[row0:row1].copy_(idx)
        del cnt, excl, base, idx
    return out


def tbe_split_checkpoints(t, S: int, *, max_chunk_tiles: int = 262144) -> torch.Tensor:
    """Verified [N,S] escape starts for an experimental sequential TBE-v1 decoder.

    Split s starts at floor(s*(K/128)/S), exactly as bi_gemm_kernel. Prefix work
    is bounded by max_chunk_tiles; the full [N,K/128] index is never allocated.
    This does not change a stored TBE payload or select a serving implementation.
    """
    from . import miv_gemv as mg

    N, K, wpr = t.N, t.K, t.wpr
    tpr = K // 64
    if N <= 0 or K <= 0 or K % 128 or wpr <= 0 or tpr % wpr:
        raise ValueError("invalid TBE escape-checkpoint geometry")
    if type(S) is not int or not 1 <= S <= K // 128:
        raise ValueError("invalid TBE checkpoint split count")
    if (type(max_chunk_tiles) is not int or not tpr <= max_chunk_tiles <= 262144):
        raise ValueError("TBE checkpoint budget must hold one row and be <=262144 tiles")
    if t.planes.numel() != N * tpr * 6 or t.escbase.numel() != N * wpr:
        raise ValueError("TBE escape-checkpoint source geometry mismatch")
    rows_per_chunk = max_chunk_tiles // tpr
    planes = t.planes.view(-1)
    bases = t.escbase.view(N, wpr)
    out = torch.empty((N, S), dtype=torch.int32, device=planes.device)
    starts = torch.tensor([2 * (s * (K // 128) // S) for s in range(S)],
                          dtype=torch.int64, device=planes.device)
    counter = mg.extension().tbe_tile_escapes
    for row0 in range(0, N, rows_per_chunk):
        row1 = min(N, row0 + rows_per_chunk)
        cnt = counter(planes[row0*tpr*6:row1*tpr*6]).to(torch.int64).view(row1-row0, tpr)
        if bool(((cnt < 0) | (cnt > 64)).any().item()):
            raise ValueError("invalid TBE checkpoint per-tile escape count")
        excl = torch.cumsum(cnt, 1) - cnt
        base = bases[row0:row1].to(torch.int64)
        for s in range(1, wpr):
            if not torch.equal(base[:, 0] + excl[:, s * (tpr // wpr)], base[:, s]):
                raise ValueError("TBE checkpoint prefix disagrees with TBELin.escbase")
        idx = base[:, :1] + excl.index_select(1, starts)
        if int(idx.min()) < 0 or int(idx.max()) >= 2 ** 31:
            raise ValueError("TBE checkpoint exceeds int32")
        out[row0:row1].copy_(idx)
    return out


def wrap_tbe_cursor(desc, ws: Workspace, *, S: Optional[int] = None,
                    tile: Optional[int] = None):
    """Explicit experimental wrapper; the ordinary wrap() and server stay unchanged."""
    _format_code("tbe_cursor")
    if getattr(desc, "kind", None) != "tbe":
        raise TypeError("cursor prototype requires a TBE-v1 descriptor")
    split = split_for(desc.N, desc.K, device=desc.planes.device) if S is None else S
    checkpoints = tbe_split_checkpoints(desc, split)
    return BILin("tbe_cursor", desc.N, desc.K,
                 [desc.planes, desc.smb, desc.esc, desc.rowparam, checkpoints],
                 ws, src=desc, S=split, tile=tile)


def wrap(desc, ws: Workspace, tile: Optional[int] = None):
    """FastDecoder descriptor -> BI descriptor (same weight bytes; views, no copies)."""
    kind = getattr(desc, "kind", None)
    if kind == "bhe1":
        from .bhe_desc import BHELin
        if not isinstance(desc, BHELin):
            raise TypeError('BHE1 dispatch requires a verified resident descriptor')
        desc.share_workspace(ws, tile)
        return desc
    if kind == "group":
        return BIGroup([(wrap(d, ws, tile), n) for d, n in desc.members])
    if kind == "perm":
        return BIPerm(wrap(desc.d, ws, tile), desc.perm)
    if kind == "bf16":
        return BILin("bf16", desc.N, desc.K, [desc.w], ws, src=desc, tile=tile)
    if kind == "q8_0":
        pk = desc.p
        return BILin("q8", pk.N, pk.K, [pk.qs, pk.sc], ws, src=desc, tile=tile)
    if kind == "tbe":
        return BILin("tbe", desc.N, desc.K, [desc.planes, desc.smb, desc.esc, desc.rowparam, tbe_escidx(desc)],
                     ws, src=desc, tile=tile)
    if kind == "tbe2":
        return BILin("tbe2", desc.N, desc.K, list(desc.arrays), ws, src=desc, tile=tile)
    if kind == "tbe21":     # the descriptor's resident checkpoints were built for desc.S
        return BILin("tbe21", desc.N, desc.K, list(desc.arrays), ws, S=desc.S, src=desc, tile=tile)
    if kind == "kq":
        dev = desc.arrays[0].device
        arrs = list(desc.arrays) + [_empty(dev)] * (5 - len(desc.arrays))
        arrs += [desc.grid if desc.grid is not None else _empty(dev),
                 desc.ksigns if desc.ksigns is not None else _empty(dev)]
        return BILin("kq", desc.N, desc.K, arrs, ws, kqt=desc.type_id, bias=desc.bias, src=desc,
                     tile=tile)
    raise TypeError(f"no BI-GEMM path for descriptor {type(desc).__name__} kind={kind}")


def decode_weight(fmt: str, N: int, K: int, arrs: List[torch.Tensor], *, S: Optional[int] = None,
                  wg: int = 1, out: Optional[torch.Tensor] = None,
                  prefix_wg: Optional[int] = None, exec_rows: Optional[int] = None) -> torch.Tensor:
    """The weight exactly as BI-GEMM's in-kernel decode produces it: bf16 [N, K] from
    ``bi_decode_kernel`` (the GEMM's own Drv<Ld<FMT>> start/load/dec sequence, CTA row mapping
    and split traversal; S defaults to the GEMM's ``split_for(N, K)``).  fmt bf16 / tbe / tbe2 /
    tbe21 (whose arrays must carry checkpoints built for that S)."""
    if exec_rows is not None:
        from . import tbe21_v2_backend as v2
        if (fmt != "tbe21" or type(exec_rows) is not int or exec_rows not in (64, 128)
                or prefix_wg is not None or type(wg) is not int or wg != 1
                or not v2.enabled() or not v2.selected_exec_layout()):
            raise ValueError("exec_rows requires selected TBE21 EXEC, rows64/128 and no geometry override")
    if not arrs:
        raise ValueError('decode requires device-owned stream tensors')
    dev = arrs[0].device
    if any(a.device != dev for a in arrs) or (out is not None and out.device != dev):
        raise ValueError('decode streams and output must use the same device')
    S = int(S) if S else split_for(int(N), int(K), device=dev)
    if out is None:
        out = torch.empty(int(N), int(K), dtype=torch.bfloat16, device=dev)
    backend = None
    if fmt == "tbe21":
        from . import tbe21_v2_backend as v2
        v2.selected_tier_prefix()
        if v2.enabled():
            backend = _extension_on_device(v2.extension, dev)
    if backend is not None:
        if exec_rows is not None:
            backend.bi_decode_exec_out(_format_code(fmt), int(N), int(K), list(arrs), out, S, exec_rows)
        elif v2.selected_tier_prefix():
            # Preserve the legacy wg default as selected G1; prefix_wg is the
            # explicit geometry used by full tier-profile startup checks.
            if type(wg) is not int or wg != 1:
                raise ValueError("TBE21 selected prefix oracle uses its default geometry; use bi_decode_prefix_out for explicit WG1/WG2")
            selected_wg = 2 if prefix_wg is None else prefix_wg
            backend.bi_decode_prefix_out(_format_code(fmt), int(N), int(K), list(arrs), out, S, wg=selected_wg)
        else:
            if prefix_wg is not None:
                raise ValueError("prefix_wg requires the selected tier-prefix backend")
            backend.bi_decode_out(_format_code(fmt), int(N), int(K), list(arrs), out, S, wg)
    else:
        if prefix_wg is not None:
            raise ValueError("prefix_wg requires the selected tier-prefix backend")
        _extension_on_device(extension, dev).bi_decode_out(
            _format_code(fmt), int(N), int(K), list(arrs), out, S, int(wg))
    return out


def kernel_attrs(fmt: str, mt: int, wg: int) -> dict:
    """Registers / spill bytes / resident CTAs per SM / smem / threads of one GEMM instantiation,
    as the card's runtime reports them (cudaFuncGetAttributes + occupancy API)."""
    v = extension().bi_kernel_attrs(_format_code(fmt), int(mt), int(wg))
    return dict(zip(("regs", "local_bytes", "max_blocks_per_sm", "smem_bytes", "threads"),
                    (int(x) for x in v)))


__all__ = ["BILin", "BIGroup", "BIPerm", "TILES", "TILE_DEFAULT", "WEIGHT_FORMATS",
           "WEIGHT_FORMAT_DEFAULT", "Workspace", "decode_weight", "extension", "kernel_attrs",
           "split_for", "tbe_escidx", "tile_cap", "tile_for", "wrap", "wrap_tbe_cursor",
           "tbe_split_checkpoints", "cursor_build_enabled"]
