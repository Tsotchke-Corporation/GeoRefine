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
FMT = {"bf16": 0, "q8": 1, "tbe": 2, "kq": 3, "tbe2": 4, "tbe21": 5}
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


def _format_code(fmt: str) -> int:
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


def _build_dir() -> Path:
    d = os.environ.get("BI_GEMM_BUILD_DIR")
    p = Path(d) if d else Path(__file__).resolve().parents[2] / ".scratch" / "bi_gemm_build"
    if str(p.resolve()).startswith(("/tmp", "/private/tmp")):
        raise RuntimeError(f"refusing a volatile build dir {p}")
    p.mkdir(parents=True, exist_ok=True)
    return p


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

            arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
            if not arch:
                major, minor = torch.cuda.get_device_capability()
                arch = f"{major}.{minor}"
                os.environ["TORCH_CUDA_ARCH_LIST"] = arch
            major, minor = arch.split(";")[0].replace("+PTX", "").split(".")
            os.environ.setdefault("MAX_JOBS", "8")
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
            _EXT = load(name=f"bi_gemm_sm{major}{minor}{scope_suffix}",
                        sources=[str(_CSRC / s) for s in source_names],
                        extra_include_paths=include_paths,
                        extra_cflags=_winenv.host_cflags() + host_scope_flag,
                        # -Xptxas=-v under BI_GEMM_VERBOSE: the 128-row tile runs 512 threads
                        # at one CTA/SM, so the per-thread register count (budget 128) and any
                        # spill is the one occupancy fact only the compiler can state.  See
                        # docs/serving/BIGEMM_TILE128_20261004.md section 4.
                        extra_cuda_cflags=(["-O3", "--fmad=true", "-lineinfo", "-std=c++17"] + scope_flag
                                           + (["-Xptxas=-v"]
                                              if int(os.environ.get("BI_GEMM_VERBOSE", "0")) else [])),
                        build_directory=str(_build_dir()),
                        verbose=bool(int(os.environ.get("BI_GEMM_VERBOSE", "0"))))
    return _EXT


def split_for(N: int, K: int, sms: Optional[int] = None) -> int:
    """Split-K count: a function of (N, K) and the card ONLY -- never of M (batch invariance)."""
    if sms is None:
        sms = torch.cuda.get_device_properties(0).multi_processor_count
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
        self.device = device
        self.buf = torch.empty(0, dtype=torch.float32, device=device)

    def get(self, n: int) -> torch.Tensor:
        if self.buf.numel() < n:
            self.buf = torch.empty(n, dtype=torch.float32, device=self.device)
        return self.buf

    def reserve(self, n: int) -> None:
        self.get(n)


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


class BILin:
    """One weight of one format through BI-GEMM."""

    def __init__(self, fmt: str, N: int, K: int, arrs: List[torch.Tensor], ws: Workspace, *,
                 kqt: int = 0, bias: Optional[torch.Tensor] = None, src=None, S: Optional[int] = None,
                 tile: Optional[int] = None):
        self.fmt, self.N, self.K, self.arrs, self.ws, self.kqt, self.bias = fmt, int(N), int(K), arrs, ws, kqt, bias
        self.S = int(S) if S else split_for(self.N, self.K)
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
        self._backend_details = {}
        if enabled and fmt in ("tbe2", "tbe21"):
            from . import bigemm_tbe2_opt as opt

            # Constructor runs before BatchDecoder.capture(). Never compile inside
            # graph capture, replace the global extension, or silently fall back.
            self._backend = opt.extension()
            self._backend_kind = "tbe2-opt"
            self._backend_details = _optimized_backend_identity(self._backend)

    def backend_identity(self) -> dict:
        """Selected implementation plus successful Python dispatches (not graph replays)."""
        details = dict(self._backend_details)
        if "source_bindings" in details:
            details["source_bindings"] = dict(details["source_bindings"])
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
        m = int(x.shape[0])
        ws = self.ws.get(self.S * m * self.N) if self.S > 1 else self.ws.buf
        t = tile_for(m, self.tile_cap, self.N, self.K) if tile is None else int(tile)
        backend = self._backend if self._backend is not None else extension()
        backend.bi_gemm_out(x, m, _format_code(self.fmt), self.kqt, self.N, self.K, self.arrs, self.bias, y,
                            self.S, ws, t)
        self._backend_calls += 1


def backend_census(descriptors) -> dict:
    """Inspect nested descriptors without loading or changing a backend."""
    rows, unknown = [], []
    def visit(d):
        if isinstance(d, BILin):
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
                optimized_tbe21_descriptors=sum(r['backend']=='tbe2-opt' and r['format']=='tbe21' for r in rows),
                optimized_python_dispatches=sum(r['dispatch_calls'] for r in rows if r['backend']=='tbe2-opt'),
                graph_replays_counted=False, descriptors=rows)


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


def wrap(desc, ws: Workspace, tile: Optional[int] = None):
    """FastDecoder descriptor -> BI descriptor (same weight bytes; views, no copies)."""
    kind = getattr(desc, "kind", None)
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
                  wg: int = 1, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The weight exactly as BI-GEMM's in-kernel decode produces it: bf16 [N, K] from
    ``bi_decode_kernel`` (the GEMM's own Drv<Ld<FMT>> start/load/dec sequence, CTA row mapping
    and split traversal; S defaults to the GEMM's ``split_for(N, K)``).  fmt bf16 / tbe / tbe2 /
    tbe21 (whose arrays must carry checkpoints built for that S)."""
    S = int(S) if S else split_for(int(N), int(K))
    dev = arrs[0].device
    if out is None:
        out = torch.empty(int(N), int(K), dtype=torch.bfloat16, device=dev)
    extension().bi_decode_out(_format_code(fmt), int(N), int(K), list(arrs), out, S, int(wg))
    return out


def kernel_attrs(fmt: str, mt: int, wg: int) -> dict:
    """Registers / spill bytes / resident CTAs per SM / smem / threads of one GEMM instantiation,
    as the card's runtime reports them (cudaFuncGetAttributes + occupancy API)."""
    v = extension().bi_kernel_attrs(_format_code(fmt), int(mt), int(wg))
    return dict(zip(("regs", "local_bytes", "max_blocks_per_sm", "smem_bytes", "threads"),
                    (int(x) for x in v)))


__all__ = ["BILin", "BIGroup", "BIPerm", "TILES", "TILE_DEFAULT", "WEIGHT_FORMATS",
           "WEIGHT_FORMAT_DEFAULT", "Workspace", "decode_weight", "extension", "kernel_attrs",
           "split_for", "tbe_escidx", "tile_cap", "tile_for", "wrap"]
