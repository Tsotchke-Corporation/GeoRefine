"""One persistent, versioned build root for every JIT-built CUDA extension of the Context runtime.

Why this module exists
----------------------
Every CUDA kernel in this package is compiled on first use by ``torch.utils.cpp_extension``.
Before 0.1.1 each extension picked its own build directory -- one of them inside the
installed package tree (``Path(__file__).parents[2] / ".scratch"``, which for a wheel is an
arbitrary spot inside the venv), others under ``~/.cache/georefine/...`` with no record of the
torch / CUDA / GPU they were built for.  On an A100-40 a server's readiness probe timed out at
900 s while kernel compilation was still running after a 487 s weight verification and an 814 s
load.  Compilation is a one-time cost per (torch, CUDA, GPU arch, source) -- it must be paid
once, visibly, ideally ahead of time, and never again.

Layout
------
::

    <root>/<torch.__version__>-cu<torch.version.cuda>-<arch tag>-py<maj><min>/<extension>/<key>/
        <extension>.so                     the built module (.pyd on Windows)
        <extension>.georefine-kernel.json  stamp: written after a successful build/load
        .georefine-build.lock              fcntl lock: one builder per directory, across processes
        .georefine-build.owner             pid/host of the builder while a build runs
        lock                               torch's own FileBaton (exists only during a build)

``<root>`` is ``$GEOREFINE_CONTEXT_KERNEL_CACHE``, else ``$XDG_CACHE_HOME/georefine-context/kernels``,
else ``~/.cache/georefine-context/kernels``.  A root under ``/tmp`` or ``/private/tmp`` is refused:
those are wiped on reboot, so every restart would silently pay the full compile again.

``<arch tag>`` is the effective ``TORCH_CUDA_ARCH_LIST`` (``8.0`` -> ``sm80``, ``9.0+PTX`` ->
``sm90ptx``, several entries joined with ``-``).  ``<key>`` is the first 16 hex digits of a SHA-256
over: the extension name; every source file (absolute path and bytes); every header in the
include directories and in the sources' own directories; inline sources (``load_inline``); every
compile flag in order; the build-relevant environment (``TORCH_CUDA_ARCH_LIST``, ``CUDA_HOME``,
``CUDA_PATH``, ``CC``, ``CXX``, ``CUDAHOSTCXX``, ``NVCC_PREPEND_FLAGS``, ``NVCC_APPEND_FLAGS``,
``NVCC_CCBIN``, ``PYTORCH_NVCC``, ``GLC_TBE_MMA_ARCH``); and the toolchain identity (torch version
and install directory, torch's CUDA version, the CUDA toolkit torch resolved and its real path, the
Python version and include directory, OS and machine).  Changing any of these lands in a NEW
directory, so a stale ``.so`` can never be loaded for a different torch / CUDA / GPU / source.
Absolute paths are part of the key on purpose: torch's generated ``build.ninja`` embeds them, so a
moved venv rebuilds anyway and the ``cached`` flag must say so.

Per-module overrides keep working and win over the cache root: ``BI_GEMM_BUILD_DIR``,
``MIV_GEMV_BUILD_DIR``, ``MIV_TBE_BUILD_DIR``, ``MIV_TBE_PAIR_BUILD_DIR``,
``MIV_TBE_FLAT_SMB_BUILD_DIR``, ``GLC_TBE_MMA_BUILD_DIR``.  An override is used as given (it is
the operator's directory; versioning it is the operator's choice); it is still refused under
``/tmp``.

Two extensions derive their directory from another module's (their files are owned elsewhere):
``fastdec`` builds into ``miv_gemv._build_dir() / "fastdec"`` and ``bidec`` into
``bigemm._build_dir().parent / "bidec_build"``.  Both therefore inherit the platform tag (torch /
CUDA / arch / Python); their own source changes are caught by ``load_inline`` (it rewrites the
generated sources only when their text changes) and ninja's timestamp check, and the stamp makes
the ``cached`` flag honest about it.

Locking
-------
``torch.utils.cpp_extension`` serialises builders with a ``FileBaton``: ``os.open(<dir>/lock,
O_CREAT | O_EXCL)``, and a process that finds the file waits -- polling, with no timeout -- until
it disappears.  A builder that is killed mid-compile (exactly what a readiness timeout does) leaves
the file behind, and every later start then waits forever.  :func:`build_lock` takes an ``fcntl``
lock on ``<dir>/.georefine-build.lock`` first; the kernel releases it when the holder dies, so it
can never go stale. A torch baton is removed only when its owner names a process on this
host and that process no longer exists. Age alone cannot establish that a build ended.
Unowned, foreign-host and live-process batons are preserved; startup refuses rather than
waiting indefinitely or overwriting their ownership. The lock is re-entrant
within a thread.  On Windows ``msvcrt.locking`` stands in for ``fcntl``.

Progress
--------
Around every build :func:`build` logs ``[kernels] building <name> (first start only; cached in
<dir>)`` or ``[kernels] <name> cached`` and reports ``startup_progress.report("kernel_build",
done=i, total=n, extension=name, cached=bool, seconds=...)``.  A hit is decided BEFORE torch is
called: the expected ``.so`` exists and its stamp carries the expected key.

Ahead-of-time warm-up
---------------------
``georefine-context-build-kernels`` (this module's :func:`main`) builds every extension the Context
serving paths load, for the GPU it runs on, and prints a JSON summary::

    georefine-context-build-kernels                 # batch path (a superset of the fast path)
    georefine-context-build-kernels --path fast     # single-session FastDecoder path only
    georefine-context-build-kernels --parallel 3    # one subprocess per extension
    georefine-context-build-kernels --check         # report only; exit 3 if anything would build

Extensions (see :data:`SERVING_PATHS`):

* fast path (``context_fast_server.load_session`` -> ``predictive_tbe.load_predictive_fast_session``
  -> ``fastdec.FastDecoder``): ``fastdec_sm<XY>`` (``fastdec.ext``), ``miv_gemv_sm<XY>``
  (``miv_gemv.extension``), ``miv_tbe_sm<XY>`` (``tbe_desc.tune_tbe_per_m`` tries the ``v2``
  implementation at startup) and ``glc_tbe_mma_ext`` (``glc_loader.tbe_mma_kernels``: the HF
  prefill through ``modules.TBEServeLinear`` -- built on the FIRST REQUEST if not warmed).
* batch path (``predictive_batch.load_predictive_batch_runtime``): everything above (it loads the
  fast session first) plus ``bidec_sm<XY>`` (``bidec.ext``) and ``bi_gemm_sm<XY>_context``
  (``bigemm.extension``), both forced by ``bidec.BatchDecoder.__init__``.

Exit status: 0 success; 1 failure (missing toolchain, no CUDA device, build error -- the message
names which); 2 bad arguments; 3 ``--check`` found extensions that would build.

Shipping prebuilt binaries (considered; NOT implemented)
--------------------------------------------------------
Prebuilt sm_80 / sm_90 / sm_120 binaries inside the wheel would remove the first-start compile and
the runtime need for nvcc.  Assessment:

* ABI.  These are pybind11 Python modules linked against ``libtorch`` / ``libc10`` /
  ``libtorch_python``; they load only into the exact torch build they were compiled against (C++
  ABI, pybind11 internals, c10 symbols).  ``pyproject.toml`` pins ``torch==2.8.0`` exactly, which
  makes it possible, but the binaries would be valid for one torch CUDA variant (the cu129 build
  the release targets) and must refuse to load -- falling back to JIT -- on any other
  ``torch.__version__``.  They link torch's CUDA runtime dynamically, so nothing from the CUDA
  toolkit is bundled.
* Platform wheels.  The package is pure-Python today.  Binaries make it a platform wheel per
  CPython version (no abi3: these are not limited-API modules) and per platform: cp310..cp313 x
  manylinux_2_28 x86_64 (+ aarch64 for GH200; + win_amd64, where the CUDA path has never been
  built), i.e. eight or more wheels, built by a CI image with the matching nvcc, with
  ``auditwheel repair --exclude libtorch*.so --exclude libc10*.so --exclude libcudart*``.
* Arches.  One fatbin per extension with SASS for the supported classes (8.0, 8.6, 8.9, 12.0 for
  ``tbe_mma``, which refuses anything else; 9.0 for the BI-GEMM/MIV kernels) plus PTX for forward
  compatibility.  sm_120 needs CUDA 12.8 or newer.
* Size.  Not measured.  ``miv_tbe`` alone is ten translation units; measure one fatbin per arch
  before deciding.
* Exactness.  The kernels compile with ``--fmad=true``, so FMA contraction is the compiler's
  choice: a different nvcc can produce different bits.  A prebuilt binary is a single, known
  compiler -- arguably MORE reproducible than per-user JIT -- but each shipped binary must pass the
  served-path digest gates on real hardware for every arch it claims.
* Packaging.  ``fastdec``, ``bidec`` and ``miv_gemv`` are ``load_inline`` extensions; a
  setuptools ``CUDAExtension`` needs them as files with byte-identical generated bindings.  The
  loaders would need a "prebuilt first" import keyed by a manifest (torch version, arch list,
  source digest -- the key computed here) with JIT fallback.
* Licensing.  nvcc output from our own sources is ours to distribute and no NVIDIA component would
  be bundled; whether the native codec licence (``licenses/NATIVE_CODEC_LICENSE.txt``) covers
  binary distribution of the codec kernels needs the owner's ruling, not an engineering guess.

Intermediate option needing no wheel changes: run this CLI in the deployment image build (same
install path, same torch) and ship the cache directory with the image; the key is path-sensitive,
so it is portable exactly when the install path is identical.

Triton kernels (``georefine_context_runtime/*_gpu_kernels.py``) keep triton's own persistent
cache (``~/.triton/cache`` or ``TRITON_CACHE_DIR``); they are not built here.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import sysconfig
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from . import startup_progress

__all__ = [
    "ENV_ROOT", "KernelCacheError", "ExtensionSpec", "SERVING_PATHS",
    "is_volatile", "refuse_volatile", "cache_root", "arch_list", "arch_tag", "platform_tag",
    "build_env", "spec_digest", "extension_dir", "resolve_build_dir", "library_path",
    "stamp_path", "is_cached", "build_lock", "build", "planned", "results",
    "toolchain_problems", "ensure_serving_kernels", "check_serving_kernels", "set_logger", "main",
]

ENV_ROOT = "GEOREFINE_CONTEXT_KERNEL_CACHE"
LOCK_NAME = ".georefine-build.lock"
OWNER_NAME = ".georefine-build.owner"
STAMP_SUFFIX = ".georefine-kernel.json"
TORCH_BATON = "lock"
VOLATILE_ROOTS = ("/tmp", "/private/tmp")
KEY_SCHEMA = "georefine-context-kernel-key-v1"
KEY_HEX = 16
BUILD_ENV_KEYS = (
    "TORCH_CUDA_ARCH_LIST", "CUDA_HOME", "CUDA_PATH", "CC", "CXX", "CUDAHOSTCXX",
    "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "NVCC_CCBIN", "PYTORCH_NVCC", "GLC_TBE_MMA_ARCH",
)
HEADER_SUFFIXES = (".h", ".hh", ".hpp", ".cuh", ".inl", ".inc")
_PROCESS_START = time.time()
_HOST = socket.gethostname()


class KernelCacheError(RuntimeError):
    """A kernel cache refusal or build failure, with a message naming the remedy."""


def _default_log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


_LOG: Callable[[str], None] = _default_log


def set_logger(fn: Optional[Callable[[str], None]]) -> None:
    """Route ``[kernels]`` lines (default: stderr).  ``None`` restores the default."""
    global _LOG
    _LOG = fn or _default_log


# ---------------------------------------------------------------------------
# root and refusal
# ---------------------------------------------------------------------------
def is_volatile(path: Any) -> bool:
    """True for a path at or under ``/tmp`` or ``/private/tmp`` (raw or symlink-resolved)."""
    p = Path(path).expanduser()
    candidates = {str(p.absolute())}
    try:
        candidates.add(str(p.resolve()))
    except OSError:
        pass
    for c in candidates:
        for root in VOLATILE_ROOTS:
            if c == root or c.startswith(root + "/"):
                return True
    return False


def refuse_volatile(path: Any, what: str = "kernel build directory") -> Path:
    p = Path(path).expanduser()
    if is_volatile(p):
        raise KernelCacheError(
            f"refusing a volatile {what} {p}: /tmp and /private/tmp are wiped on reboot, so every "
            f"restart would recompile every CUDA kernel (minutes each). Set {ENV_ROOT} (or the "
            "module's *_BUILD_DIR override) to a persistent directory; the default is "
            "~/.cache/georefine-context/kernels.")
    return p


def cache_root(environ: Optional[Mapping[str, str]] = None) -> Path:
    """``$GEOREFINE_CONTEXT_KERNEL_CACHE`` > ``$XDG_CACHE_HOME/georefine-context/kernels`` >
    ``~/.cache/georefine-context/kernels``.  Never under /tmp."""
    env = os.environ if environ is None else environ
    explicit = env.get(ENV_ROOT)
    if explicit:
        root, source = Path(explicit).expanduser(), ENV_ROOT
    else:
        xdg = env.get("XDG_CACHE_HOME")
        base, source = ((Path(xdg).expanduser(), "XDG_CACHE_HOME") if xdg
                        else (Path(os.path.expanduser("~")) / ".cache", "the home directory"))
        root = base / "georefine-context" / "kernels"
    if is_volatile(root):
        raise KernelCacheError(
            f"refusing the volatile kernel cache root {root} (from {source}): /tmp and "
            f"/private/tmp are wiped on reboot, so every restart would recompile every CUDA "
            f"kernel. Set {ENV_ROOT} to a persistent directory.")
    return root


# ---------------------------------------------------------------------------
# keys
# ---------------------------------------------------------------------------
def arch_list(environ: Optional[Mapping[str, str]] = None, device: Any = None) -> str:
    """The effective ``TORCH_CUDA_ARCH_LIST``: the env value, else the device capability --
    the same fallback every builder in this package applies before calling torch."""
    env = os.environ if environ is None else environ
    value = (env.get("TORCH_CUDA_ARCH_LIST") or "").strip()
    if value:
        return value
    import torch

    if not torch.cuda.is_available():
        raise KernelCacheError("TORCH_CUDA_ARCH_LIST is unset and no CUDA device is visible to "
                               "derive the GPU architecture from")
    major, minor = torch.cuda.get_device_capability(device)
    return f"{major}.{minor}"


def arch_tag(arch: str) -> str:
    """``"8.0"`` -> ``"sm80"``; ``"9.0+PTX"`` -> ``"sm90ptx"``; ``"8.0;12.0"`` -> ``"sm80-sm120"``."""
    tags = []
    for entry in re.split(r"[;,\s]+", str(arch).strip()):
        if not entry:
            continue
        ptx = entry.upper().endswith("+PTX")
        core = entry[:-4] if ptx else entry
        core = re.sub(r"[^0-9A-Za-z]+", "", core).lower()
        tags.append(f"sm{core}" + ("ptx" if ptx else ""))
    if not tags:
        raise KernelCacheError(f"empty CUDA arch list {arch!r}")
    return "-".join(tags)


_UNSET = object()


def platform_tag(arch: str, *, torch_version: Any = _UNSET, cuda_version: Any = _UNSET,
                 python: Optional[Tuple[int, int]] = None) -> str:
    """``<torch.__version__>-cu<torch.version.cuda>-<arch tag>-py<maj><min>``."""
    if torch_version is _UNSET or cuda_version is _UNSET:
        import torch

        if torch_version is _UNSET:
            torch_version = torch.__version__
        if cuda_version is _UNSET:
            cuda_version = torch.version.cuda
    py = tuple(sys.version_info[:2]) if python is None else tuple(python)
    tag = f"{torch_version}-cu{cuda_version or 'none'}-{arch_tag(arch)}-py{py[0]}{py[1]}"
    return re.sub(r"[^0-9A-Za-z.+_-]+", "_", tag)


@dataclasses.dataclass(frozen=True)
class ExtensionSpec:
    """What determines an extension's binary.  ``flags`` = every cflag / cuda cflag / ldflag
    passed to torch, verbatim and in order; ``arch`` = the effective TORCH_CUDA_ARCH_LIST
    (None: derive with :func:`arch_list`); ``extra`` = other build-defining strings (e.g. the
    ``functions`` list of a ``load_inline`` call)."""

    name: str
    sources: Tuple[str, ...] = ()
    inline_sources: Tuple[str, ...] = ()
    include_dirs: Tuple[str, ...] = ()
    flags: Tuple[str, ...] = ()
    arch: Optional[str] = None
    extra: Tuple[Tuple[str, str], ...] = ()


def build_env(arch: Optional[str] = None,
              environ: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The build-relevant environment; ``arch`` (when given) is the TORCH_CUDA_ARCH_LIST used."""
    env = os.environ if environ is None else environ
    out = {k: str(env[k]) for k in BUILD_ENV_KEYS if env.get(k)}
    if arch is not None:
        out["TORCH_CUDA_ARCH_LIST"] = str(arch)
    return out


@lru_cache(maxsize=1)
def toolchain_identity() -> Tuple[Tuple[str, str], ...]:
    """Process-constant facts that end up in torch's generated build: torch, its CUDA, the CUDA
    toolkit torch resolves, Python and the platform."""
    import torch

    try:
        from torch.utils import cpp_extension as ce

        cuda_home = ce.CUDA_HOME
    except Exception:  # noqa: BLE001 -- a broken toolchain must not break key derivation
        cuda_home = None
    ident = {
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "torch_dir": os.path.dirname(os.path.abspath(torch.__file__)),
        "cuda_home": str(cuda_home),
        "cuda_home_real": os.path.realpath(cuda_home) if cuda_home else "",
        "python": sys.version.split()[0],
        "python_include": str(sysconfig.get_paths().get("include", "")),
        "system": platform.system(),
        "machine": platform.machine(),
    }
    return tuple(sorted(ident.items()))


def _header_files(dirs: Sequence[Path]) -> List[Path]:
    seen, out = set(), []
    for d in dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix in HEADER_SUFFIXES and f.resolve() not in seen:
                seen.add(f.resolve())
                out.append(f)
    return out


def spec_digest(spec: ExtensionSpec, *, environ: Optional[Mapping[str, str]] = None,
                identity: Optional[Mapping[str, str]] = None) -> str:
    """Full SHA-256 hex digest of everything that determines the extension binary."""
    arch = spec.arch if spec.arch is not None else arch_list(environ)
    h = hashlib.sha256()

    def feed(tag: str, data: bytes) -> None:
        t = tag.encode()
        h.update(len(t).to_bytes(4, "little"))
        h.update(t)
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)

    feed("schema", KEY_SCHEMA.encode())
    feed("name", spec.name.encode())
    for s in spec.sources:
        p = Path(s)
        feed("source-path", str(p.resolve()).encode())
        feed("source", p.read_bytes())
    for text in spec.inline_sources:
        feed("inline", text.encode())
    include_dirs = [Path(d) for d in spec.include_dirs]
    for d in include_dirs:
        feed("include-dir", str(d.resolve()).encode())
    for f in _header_files(include_dirs + [Path(s).parent for s in spec.sources]):
        feed("header:" + str(f.resolve()), f.read_bytes())
    for flag in spec.flags:
        feed("flag", str(flag).encode())
    for k, v in spec.extra:
        feed("extra:" + str(k), str(v).encode())
    for k, v in sorted(build_env(arch, environ).items()):
        feed("env:" + k, v.encode())
    ident = dict(toolchain_identity()) if identity is None else dict(identity)
    for k, v in sorted(ident.items()):
        feed("toolchain:" + str(k), str(v).encode())
    return h.hexdigest()


def extension_dir(spec: ExtensionSpec, *, environ: Optional[Mapping[str, str]] = None,
                  root: Any = None, identity: Optional[Mapping[str, str]] = None,
                  create: bool = True) -> Path:
    """``<root>/<platform tag>/<name>/<key16>`` for ``spec``."""
    arch = spec.arch if spec.arch is not None else arch_list(environ)
    spec = dataclasses.replace(spec, arch=arch)
    base = cache_root(environ) if root is None else refuse_volatile(root, "kernel cache root")
    if identity is not None:
        ident = dict(identity)
        tag = platform_tag(arch, torch_version=ident.get("torch"),
                           cuda_version=ident.get("torch_cuda"))
    else:
        tag = platform_tag(arch)
    name = re.sub(r"[^0-9A-Za-z._+-]+", "_", spec.name)
    d = base / tag / name / spec_digest(spec, environ=environ, identity=identity)[:KEY_HEX]
    refuse_volatile(d)
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_build_dir(spec: ExtensionSpec, override_var: Optional[str] = None, *,
                      environ: Optional[Mapping[str, str]] = None, create: bool = True) -> Path:
    """The module's ``*_BUILD_DIR`` override (used as given, refused under /tmp) or the
    versioned cache directory."""
    env = os.environ if environ is None else environ
    override = env.get(override_var) if override_var else None
    if override:
        d = refuse_volatile(Path(override).expanduser(), f"{override_var} build directory")
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d
    return extension_dir(spec, environ=environ, create=create)


def library_path(build_dir: Any, name: str) -> Path:
    """Where torch puts the built module: ``<dir>/<name>.so`` (``.pyd`` on Windows)."""
    return Path(build_dir) / f"{name}{'.pyd' if os.name == 'nt' else '.so'}"


def stamp_path(build_dir: Any, name: str) -> Path:
    return Path(build_dir) / f"{name}{STAMP_SUFFIX}"


def is_cached(build_dir: Any, name: str, key: Optional[str]) -> bool:
    """The library exists and (when a key is known) its stamp carries that key."""
    if not library_path(build_dir, name).is_file():
        return False
    if key is None:
        return True
    try:
        stamp = json.loads(stamp_path(build_dir, name).read_text())
    except (OSError, ValueError):
        return False
    return isinstance(stamp, dict) and stamp.get("key") == key


def _write_json_atomic(path: Path, value: Dict[str, Any]) -> None:
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, sort_keys=True) + "\n")
    os.replace(part, path)


def _write_stamp(build_dir: Path, name: str, key: str, seconds: float) -> None:
    try:
        _write_json_atomic(stamp_path(build_dir, name), {
            "schema": KEY_SCHEMA, "name": name, "key": key, "build_seconds": round(seconds, 3),
            "written_at": time.time(), "host": _HOST,
        })
    except OSError as exc:  # the stamp only labels; never fail a good build over it
        _LOG(f"[kernels] could not write stamp for {name} in {build_dir}: {exc}")


# ---------------------------------------------------------------------------
# locking
# ---------------------------------------------------------------------------
_TLS = threading.local()


def _try_lock(fh) -> bool:
    if os.name == "nt":  # pragma: no cover - exercised on Windows only
        import msvcrt

        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    import fcntl

    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, PermissionError):
        return False
    except OSError as exc:
        import errno

        if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
            return False
        raise


def _unlock(fh) -> None:
    if os.name == "nt":  # pragma: no cover
        import msvcrt

        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        return
    import fcntl

    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _read_owner(build_dir: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads((build_dir / OWNER_NAME).read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _write_owner(build_dir: Path, name: str) -> None:
    try:
        _write_json_atomic(build_dir / OWNER_NAME,
                           {"pid": os.getpid(), "host": _HOST, "extension": name, "t": time.time()})
    except OSError:
        pass


def _remove_owner(build_dir: Path) -> None:
    try:
        (build_dir / OWNER_NAME).unlink()
    except OSError:
        pass


def clear_stale_baton(build_dir: Any, *, log: Optional[Callable[[str], None]] = None) -> bool:
    """Remove torch's ``<dir>/lock`` when it is provably stale.  Call ONLY while holding
    :func:`build_lock` for ``build_dir``.  Returns whether a file was removed."""
    log = log or _LOG
    d = Path(build_dir)
    baton = d / TORCH_BATON
    try:
        mtime = baton.stat().st_mtime
    except OSError:
        return False
    owner = _read_owner(d)
    if (owner is None or owner.get("host") != _HOST
            or type(owner.get("pid")) is not int or owner["pid"] <= 0):
        log(f"[kernels] torch build lock {baton} is outside this cache's lock ownership; "
            "preserving it because its builder cannot be verified")
        return False
    try:
        os.kill(owner["pid"], 0)
    except ProcessLookupError:
        pass  # The recorded local process is gone; this owned baton is stale.
    except OSError:
        return False  # Permission errors do not establish that a process is gone.
    else:
        return False
    try:
        baton.unlink()
    except FileNotFoundError:
        return False
    who = f" by pid {owner.get('pid')} on {owner.get('host')}" if owner else ""
    log(f"[kernels] removed a stale torch build lock {baton} (left{who} by an interrupted build, "
        f"{max(0.0, time.time() - mtime):.0f} s old)")
    return True


@contextmanager
def build_lock(build_dir: Any, *, name: str = "", poll: float = 0.25, notice_every: float = 30.0,
               timeout: Optional[float] = None,
               log: Optional[Callable[[str], None]] = None) -> Iterator[Path]:
    """Exclusive, cross-process, crash-safe lock for one build directory (re-entrant per thread).

    Waiting is logged (another process is compiling the same extension -- its result will be
    reused).  ``timeout`` (seconds, default none) raises :class:`KernelCacheError`."""
    log = log or _LOG
    d = Path(build_dir)
    d.mkdir(parents=True, exist_ok=True)
    key = str(d.resolve())
    held = getattr(_TLS, "held", None)
    if held is None:
        held = _TLS.held = set()
    if key in held:
        yield d
        return
    fh = open(d / LOCK_NAME, "a+b")
    try:
        t0 = time.monotonic()
        next_notice = 0.0
        while not _try_lock(fh):
            waited = time.monotonic() - t0
            if waited >= next_notice:
                owner = _read_owner(d) or {}
                log(f"[kernels] waiting for another process building {name or d.name} "
                    f"(pid {owner.get('pid', '?')}) in {d}; {waited:.0f} s so far")
                next_notice = waited + notice_every
            if timeout is not None and waited > timeout:
                raise KernelCacheError(f"timed out after {waited:.0f} s waiting for the build "
                                       f"lock {d / LOCK_NAME}")
            time.sleep(poll)
        held.add(key)
        try:
            clear_stale_baton(d, log=log)
            yield d
        finally:
            held.discard(key)
            _unlock(fh)
    finally:
        fh.close()


# ---------------------------------------------------------------------------
# build + progress
# ---------------------------------------------------------------------------
_STATE_LOCK = threading.Lock()
_STATE: Dict[str, Any] = {"total": None, "done": 0}
_RESULTS: List[Dict[str, Any]] = []


@contextmanager
def planned(total: int) -> Iterator[None]:
    """Declare that ``total`` builds follow, so progress events carry ``done/total``."""
    with _STATE_LOCK:
        prev = dict(_STATE)
        _STATE.update(total=int(total), done=0)
    try:
        yield
    finally:
        with _STATE_LOCK:
            _STATE.update(prev)


def results() -> List[Dict[str, Any]]:
    """Every build/load this process performed through :func:`build`, in order."""
    with _STATE_LOCK:
        return [dict(r) for r in _RESULTS]


def build(name: str, build_dir: Any, loader: Callable[[], Any], *,
          spec: Optional[ExtensionSpec] = None, key: Optional[str] = None,
          log: Optional[Callable[[str], None]] = None) -> Any:
    """Run ``loader`` (the module's unchanged ``cpp_extension.load``/``load_inline`` call into
    ``build_dir``) under :func:`build_lock`, with cache-hit detection, log lines, progress events,
    a stamp, and a result record.  Exceptions propagate unchanged after a ``failed`` event."""
    log = log or _LOG
    d = Path(build_dir)
    if key is None and spec is not None:
        key = spec_digest(spec)
    # A console script run by absolute path does not have the venv's bin/ (ninja) on PATH;
    # glc_loader._jitenv explains.  Idempotent; never raises.
    try:
        from glc_loader._jitenv import ensure_build_tools_on_path

        ensure_build_tools_on_path()
    except ImportError:
        pass
    with build_lock(d, name=name, log=log):
        if (d / TORCH_BATON).exists():
            raise KernelCacheError(f"build lock {d / TORCH_BATON} has a live or unverified owner; "
                                   "wait for that builder or use a separate kernel cache")
        cached = is_cached(d, name, key)
        with _STATE_LOCK:
            done, total = _STATE["done"], _STATE["total"]
        if cached:
            log(f"[kernels] {name} cached")
        else:
            log(f"[kernels] building {name} (first start only; cached in {d})")
        startup_progress.report("kernel_build", done=done, total=total, extension=name,
                                cached=cached, dir=str(d),
                                status="loading" if cached else "building")
        _write_owner(d, name)
        t0 = time.perf_counter()
        try:
            ext = loader()
        except BaseException as exc:
            seconds = time.perf_counter() - t0
            startup_progress.report("kernel_build", done=done, total=total, extension=name,
                                    cached=cached, dir=str(d), seconds=round(seconds, 3),
                                    status="failed", error=f"{type(exc).__name__}: {exc}"[:500])
            with _STATE_LOCK:
                _RESULTS.append({"extension": name, "dir": str(d), "cached": cached,
                                 "seconds": round(seconds, 3), "ok": False,
                                 "error": f"{type(exc).__name__}: {exc}"[:2000]})
            log(f"[kernels] {name} FAILED after {seconds:.1f} s: {type(exc).__name__}: {exc}")
            raise
        finally:
            _remove_owner(d)
        seconds = time.perf_counter() - t0
        if key is not None and library_path(d, name).is_file() and not cached:
            _write_stamp(d, name, key, seconds)
    with _STATE_LOCK:
        _STATE["done"] += 1
        done, total = _STATE["done"], _STATE["total"]
        _RESULTS.append({"extension": name, "dir": str(d), "cached": cached,
                         "seconds": round(seconds, 3), "ok": True})
    startup_progress.report("kernel_build", done=done, total=total, extension=name,
                            cached=cached, dir=str(d), seconds=round(seconds, 3), status="ready")
    if not cached:
        log(f"[kernels] built {name} in {seconds:.1f} s")
    return ext


# ---------------------------------------------------------------------------
# the serving extensions
# ---------------------------------------------------------------------------
def _capability() -> Tuple[int, int]:
    import torch

    major, minor = torch.cuda.get_device_capability()
    return int(major), int(minor)


def _plan_fastdec() -> Tuple[str, Path, ExtensionSpec, Callable[[], Any]]:
    # Mirrors fastdec.ext() (glc_serve/fastdec.py): arch default, name, inline sources, flags,
    # build directory.  fastdec.py is not edited here; its own call does the build.
    from . import fastdec

    major, minor = _capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    name = f"fastdec_sm{major}{minor}"
    src = Path(fastdec.__file__).with_name("fastdec_kernels.cu").read_text()
    spec = ExtensionSpec(name=name, inline_sources=(fastdec._CPP, src), flags=("-O3", "-lineinfo"),
                         arch=os.environ["TORCH_CUDA_ARCH_LIST"],
                         extra=(("functions", ",".join(fastdec._FUNCS)),))
    return name, fastdec._own_build_dir(), spec, fastdec.ext


def _plan_bidec() -> Tuple[str, Path, ExtensionSpec, Callable[[], Any]]:
    # Mirrors bidec.ext() (glc_serve/bidec.py); bidec.py is not edited here.
    from . import bidec
    from . import bigemm as bg

    arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if not arch:
        major, minor = _capability()
        arch = f"{major}.{minor}"
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    tag = arch.split(";")[0].replace("+PTX", "").replace(".", "")
    name = f"bidec_sm{tag}"
    d = bg._build_dir().parent / "bidec_build"
    d.mkdir(parents=True, exist_ok=True)
    src = Path(bidec.__file__).with_name("bidec_kernels.cu").read_text()
    spec = ExtensionSpec(name=name, inline_sources=(bidec._CPP, src), flags=("-O3", "-lineinfo"),
                         arch=arch, extra=(("functions", ",".join(bidec._FUNCS)),))
    return name, d, spec, bidec.ext


def _plan_module(modname: str) -> Callable[[], Tuple[str, Path, ExtensionSpec, Callable[[], Any]]]:
    def plan():
        import importlib

        mod = importlib.import_module(modname)
        name, d, spec = mod.kernel_build_plan()
        return name, d, spec, mod.extension
    return plan


def _plan_tbe_mma(device: Any = None):
    from glc_loader import tbe_mma_kernels
    from glc_loader.tbe_mma import resolve_tbe_mma_arch

    resolve_tbe_mma_arch(device)
    name, d, spec = tbe_mma_kernels.kernel_build_plan()
    return name, d, spec, lambda: tbe_mma_kernels._build_extension(device)


#: target -> (one-line role, planner).  A planner returns (extension name, build dir, spec,
#: the module's own build entry point).  Modules owned by this package call :func:`build`
#: from inside their entry point; fastdec/bidec are wrapped here.
TARGETS: Dict[str, Tuple[str, Callable[..., Tuple[str, Path, ExtensionSpec, Callable[[], Any]]], bool]] = {
    "tbe_mma": ("TBE v1 fragment kernels: HF prefill through modules.TBEServeLinear",
                _plan_tbe_mma, True),
    "miv_gemv": ("MIV-GEMV + engine MIV-TBE decode: every FastDecoder projection",
                 _plan_module("glc_serve.miv_gemv"), False),
    "fastdec": ("FastDecoder small ops (embed, rmsnorm, GDN, attention, argmax)", _plan_fastdec,
                False),
    "miv_tbe": ("MIV-TBE v2 decode, tried by tbe_desc.tune_tbe_per_m at startup",
                _plan_module("glc_serve.miv_tbe"), False),
    "bidec": ("BatchDecoder per-row/per-sequence kernels", _plan_bidec, False),
    "bi_gemm": ("BI-GEMM batch-invariant projections (Context BF16/TBE scope)",
                _plan_module("glc_serve.bigemm"), False),
}

#: The extensions each Context serving path loads at startup or first request.
SERVING_PATHS: Dict[str, Tuple[str, ...]] = {
    "fast": ("tbe_mma", "miv_gemv", "fastdec", "miv_tbe"),
    "batch": ("tbe_mma", "miv_gemv", "fastdec", "miv_tbe", "bidec", "bi_gemm"),
}

# Builds whose entry point already calls build() internally (no outer wrapper).
_SELF_INSTRUMENTED = ("tbe_mma", "miv_gemv", "miv_tbe", "bi_gemm")


def _plan(target: str, device: Any = None):
    role, planner, takes_device = TARGETS[target]
    return planner(device) if takes_device else planner()


def _build_target(target: str, device: Any, log: Callable[[str], None]) -> Dict[str, Any]:
    name, d, spec, entry = _plan(target, device)
    before = len(results())
    t0 = time.perf_counter()
    if target in _SELF_INSTRUMENTED:
        entry()
    else:
        build(name, d, entry, spec=spec, log=log)
    mine = [r for r in results()[before:] if r["extension"] == name]
    rec = dict(mine[-1]) if mine else {
        # already loaded in this process: nothing was built or loaded now
        "extension": name, "dir": str(d), "cached": True,
        "seconds": round(time.perf_counter() - t0, 3), "ok": True}
    rec["target"] = target
    return rec


def _targets_for(path: str, only: Optional[Sequence[str]]) -> List[str]:
    if path not in SERVING_PATHS:
        raise KernelCacheError(f"unknown serving path {path!r}; expected one of "
                               f"{sorted(SERVING_PATHS)}")
    names = list(SERVING_PATHS[path])
    if only:
        bad = [t for t in only if t not in TARGETS]
        if bad:
            raise KernelCacheError(f"unknown extension target(s) {bad}; expected {sorted(TARGETS)}")
        names = [t for t in names if t in only] + [t for t in only if t not in names]
    return names


def _set_device(device: Any) -> None:
    import torch

    if device is not None and str(device) != "cuda" and torch.device(device).type == "cuda":
        torch.cuda.set_device(torch.device(device))


def ensure_serving_kernels(path: str = "batch", *, device: Any = None,
                           only: Optional[Sequence[str]] = None,
                           log: Optional[Callable[[str], None]] = None) -> List[Dict[str, Any]]:
    """Build (or load from cache) every extension of a serving path, in order, with progress.
    Call it at server start BEFORE the weight load: a missing toolchain then fails in seconds,
    and every build is visible.  Returns one record per extension."""
    log = log or _LOG
    targets = _targets_for(path, only)
    _set_device(device)
    out = []
    with planned(len(targets)):
        for target in targets:
            out.append(_build_target(target, device, log))
    return out


def check_serving_kernels(path: str = "batch", *, device: Any = None,
                          only: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Report, without building, whether each extension of ``path`` would be a cache hit."""
    _set_device(device)
    out = []
    for target in _targets_for(path, only):
        name, d, spec, _entry = _plan(target, device)
        out.append({"target": target, "extension": name, "dir": str(d),
                    "cached": is_cached(d, name, spec_digest(spec))})
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cuda_home_nvcc() -> Optional[str]:
    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:  # noqa: BLE001
        return None
    if not CUDA_HOME:
        return None
    exe = Path(CUDA_HOME) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
    return str(exe) if exe.is_file() else None


def toolchain_problems() -> List[str]:
    """What is missing for a JIT build, resolved the way the build resolves it (empty = ok)."""
    from glc_loader import _jitenv

    rep = _jitenv.build_tool_report()
    problems = []
    if not rep.get("ninja"):
        problems.append("ninja not found on PATH or in this interpreter's script directory "
                        "(it is a declared dependency: pip install ninja)")
    if not (rep.get("nvcc") or _cuda_home_nvcc()):
        problems.append("nvcc not found: install the CUDA toolkit matching torch.version.cuda and "
                        "set CUDA_HOME (or put its bin/ on PATH)")
    if not rep.get("host_compiler"):
        problems.append(f"host C++ compiler not found ({rep.get('host_compiler_name')})")
    return problems


def _summary(ok: bool, path: str, records: List[Dict[str, Any]], seconds: float,
             error: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": ok, "path": path, "extensions": records,
                           "seconds": round(seconds, 3)}
    try:
        out["root"] = str(cache_root())
    except KernelCacheError as exc:
        out["root_error"] = str(exc)
    if error:
        out["error"] = error
    return out


def _run_parallel(targets: List[str], args, log: Callable[[str], None]) -> List[Dict[str, Any]]:
    pkg_parent = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = pkg_parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    records: Dict[str, Dict[str, Any]] = {}
    lock = threading.Lock()

    def one(target: str) -> None:
        cmd = [sys.executable, "-m", "glc_serve.kernel_cache", "--path", args.path,
               "--only", target, "--parallel", "1"]
        if args.device:
            cmd += ["--device", args.device]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, env=env, text=True)
        try:
            body = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            body = {"ok": False, "error": f"no JSON from the {target} builder "
                                          f"(exit {proc.returncode})", "extensions": []}
        recs = body.get("extensions") or [{"target": target, "ok": False}]
        if proc.returncode != 0 or not body.get("ok"):
            for r in recs:
                r.setdefault("ok", False)
                r.setdefault("error", body.get("error"))
        with lock:
            records[target] = recs[0]
            done = len(records)
        startup_progress.report("kernel_build", done=done, total=len(targets),
                                extension=recs[0].get("extension", target),
                                cached=recs[0].get("cached"), seconds=recs[0].get("seconds"),
                                status="ready" if recs[0].get("ok", True) else "failed")

    with ThreadPoolExecutor(max_workers=max(1, int(args.parallel))) as pool:
        list(pool.map(one, targets))
    return [records[t] for t in targets]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="georefine-context-build-kernels",
        description="Build (once) every CUDA extension the Context serving paths load, into the "
                    "persistent kernel cache, so server start-up never compiles.")
    ap.add_argument("--path", choices=sorted(SERVING_PATHS), default="batch",
                    help="serving path to warm: 'fast' (single-session FastDecoder) or 'batch' "
                         "(multi-user bidec; a superset of 'fast'). Default: batch")
    ap.add_argument("--only", action="append", choices=sorted(TARGETS), default=None,
                    help="build only this extension target (repeatable)")
    ap.add_argument("--device", default=None, help="CUDA device to build for (default: current)")
    ap.add_argument("--parallel", type=int, default=1,
                    help="build up to N extensions concurrently, one subprocess each (default 1)")
    ap.add_argument("--check", action="store_true",
                    help="build nothing; report cache status, exit 3 if anything would build")
    args = ap.parse_args(argv)
    if args.parallel < 1:
        ap.error("--parallel must be >= 1")
    log = _LOG
    t0 = time.perf_counter()

    def fail(msg: str, records: Optional[List[Dict[str, Any]]] = None) -> int:
        log(f"[kernels] ERROR: {msg}")
        print(json.dumps(_summary(False, args.path, records or [], time.perf_counter() - t0, msg)))
        return 1

    try:
        targets = _targets_for(args.path, args.only)
        cache_root()
    except KernelCacheError as exc:
        return fail(str(exc))
    if not args.check:
        problems = toolchain_problems()
        if problems:
            return fail("cannot build CUDA kernels: " + "; ".join(problems))
    import torch

    if not torch.cuda.is_available():
        return fail("no CUDA device is visible; kernels are built for the GPU they will run on")
    try:
        if args.check:
            records = check_serving_kernels(args.path, device=args.device, only=args.only)
            print(json.dumps(_summary(True, args.path, records, time.perf_counter() - t0)))
            return 0 if all(r["cached"] for r in records) else 3
        if args.parallel > 1 and len(targets) > 1:
            records = _run_parallel(targets, args, log)
            bad = [r for r in records if not r.get("ok", True)]
            if bad:
                return fail("build failed for " + ", ".join(
                    f"{r.get('target')}: {r.get('error')}" for r in bad), records)
        else:
            records = ensure_serving_kernels(args.path, device=args.device, only=args.only,
                                             log=log)
    except Exception as exc:  # noqa: BLE001 -- the CLI's job is a clear message + exit 1
        done = [dict(r) for r in results()]
        return fail(f"{type(exc).__name__}: {exc}", done)
    print(json.dumps(_summary(True, args.path, records, time.perf_counter() - t0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
