"""Explicit opt-in portable C++ CPU helpers for BHE1 Huffman streams.

The packet module remains responsible for framing, metadata, hashes and raw
fallback policy. This module owns the bit-tight stream traversal and decoding.
The library is compiled into a persistent cache; failures are explicit and do
not fall back to Python.
"""
from __future__ import annotations

import ctypes
from .bitexact_context_codec import _native_file_lock
from contextlib import contextmanager
from collections import OrderedDict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import threading

import numpy as np

_SOURCE = Path(__file__).with_name("context_exponent_native.cpp")
_CACHE_ENV = "GEOREFINE_BHE1_NATIVE_CACHE"
_LOADED: dict[Path, ctypes.CDLL] = {}
_LOCK = threading.RLock()
_SESSION_LOCAL = threading.local()
_SESSION_TABLE_LIMIT = 2


class _NativeCpuSession:
    """Short-lived per-thread cache for repeated packet-panel preparation."""
    def __init__(self):
        self.libraries: dict[Path, ctypes.CDLL] = {}
        self.cache_directories: dict[tuple[bool, str | bytes | None], tuple[Path, tuple]] = {}
        self.tables: OrderedDict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()


@contextmanager
def native_cpu_session():
    """Reuse selected native state only within this thread and lexical scope.

    Nested calls get an isolated cache and restore the outer scope afterward;
    no session state is retained globally or inherited by another thread.
    """
    stack = getattr(_SESSION_LOCAL, "stack", None)
    if stack is None:
        stack = []
        _SESSION_LOCAL.stack = stack
    session = _NativeCpuSession()
    stack.append(session)
    try:
        yield
    finally:
        if not stack or stack[-1] is not session:
            raise RuntimeError("BHE1 native CPU sessions exited out of order")
        stack.pop()
        if not stack:
            del _SESSION_LOCAL.stack


def _active_session():
    stack = getattr(_SESSION_LOCAL, "stack", None)
    return stack[-1] if stack else None


def _is_temporary_cache_path(path, *, system=None, environ=None) -> bool:
    system = os.name if system is None else system
    environ = os.environ if environ is None else environ
    roots = ["/tmp", "/private/tmp", "/var/folders", "/private/var/folders"]
    if system == "nt":
        for key in ("LOCALAPPDATA", "SYSTEMROOT"):
            base = environ.get(key)
            if base:
                roots.append(str(Path(base) / "Temp"))
    path = Path(path).resolve()
    forbidden = (Path(root).resolve() for root in roots)
    return any(path == root or root in path.parents for root in forbidden)


def _cache_request(cache_dir=None) -> tuple[Path, tuple[bool, str | bytes | None]]:
    requested = cache_dir if cache_dir is not None else os.environ.get(_CACHE_ENV)
    if requested is not None:
        requested_path = os.fspath(requested)
        if not requested_path.strip():
            raise RuntimeError("BHE1 native cache directory must be a non-empty persistent path")
        path = Path(requested_path).expanduser()
    else:
        requested_path = None
        path = Path.home() / ".cache" / "georefine" / "bhe1-native"
    return path.resolve(), (cache_dir is None, requested_path)


def _cache_environment_signature() -> tuple:
    return (os.name, os.environ.get("LOCALAPPDATA"), os.environ.get("SYSTEMROOT"))


def _cache_directory(cache_dir=None) -> Path:
    path, _ = _cache_request(cache_dir)
    if _is_temporary_cache_path(path):
        raise RuntimeError("BHE1 native cache may not use temporary storage")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _platform_name(system=None) -> str:
    system = platform.system() if system is None else system
    if system not in {"Darwin", "Linux", "Windows"}:
        raise RuntimeError(f"BHE1 native codec does not support platform {system!r}")
    return system


def _compiler_path(system: str) -> str:
    configured = os.environ.get("GEOREFINE_NATIVE_CXX") or os.environ.get("CXX")
    candidates = [configured] if configured else (["g++"] if system == "Windows" else ["clang++"] if system == "Darwin" else ["c++", "g++", "clang++"])
    for candidate in candidates:
        if candidate:
            found = shutil.which(candidate)
            if found:
                # Keep the selected driver spelling in argv[0]. On macOS clang++
                # is commonly a symlink to clang-XX; resolving it changes the invoked
                # basename and clang selects C mode, which breaks std::exception links.
                # os.stat below still follows the link for executable fingerprinting.
                return os.path.abspath(found)
    raise RuntimeError(f"BHE1 native codec requires a C++ compiler ({configured or ', '.join(candidates)})")


@lru_cache(maxsize=16)
def _compiler_identity(compiler: str, fingerprint: tuple[int, int, int, int, int]) -> str:
    try:
        result = subprocess.run([compiler, "--version"], check=False, capture_output=True, text=True)
    except OSError as exc:
        raise RuntimeError(f"BHE1 compiler identity query failed: {exc}") from exc
    if result.returncode:
        raise RuntimeError(f"BHE1 compiler identity query failed ({result.returncode}): {result.stderr.strip()}")
    version = (result.stdout or result.stderr).strip()
    if not version:
        raise RuntimeError("BHE1 compiler returned an empty version identity")
    return f"{compiler}\n{version}"


def _build_command(compiler: str, source: Path, output: Path, *, system=None) -> list[str]:
    system = _platform_name(system)
    if system == "Windows":
        return [compiler, "-std=c++17", "-O3", "-shared", "-static", "-static-libgcc",
                "-static-libstdc++", "-Wl,--export-all-symbols", str(source), "-o", str(output)]
    return [compiler, "-std=c++17", "-O3", "-fPIC",
            "-dynamiclib" if system == "Darwin" else "-shared",
            str(source), "-o", str(output)]


def _load_library_uncached(cache_dir=None) -> ctypes.CDLL:
    if not _SOURCE.is_file():
        raise RuntimeError(f"BHE1 native source is missing: {_SOURCE}")
    system = _platform_name()
    compiler = _compiler_path(system)
    try:
        stat = os.stat(compiler)
    except OSError as exc:
        raise RuntimeError(f"BHE1 compiler stat failed: {exc}") from exc
    source_hash = hashlib.sha256(_SOURCE.read_bytes() + _SOURCE.with_name("context_exponent_packet.cuh").read_bytes()).hexdigest()
    cache_root = _cache_directory(cache_dir)
    identity = json.dumps({
        "source_sha256": source_hash,
        "compiler": _compiler_identity(compiler, (stat.st_dev, stat.st_ino, stat.st_size,
                                                    stat.st_mtime_ns, stat.st_ctime_ns)),
        "compiler_path": compiler, "compiler_device": stat.st_dev,
        "compiler_inode": stat.st_ino, "compiler_size": stat.st_size,
        "compiler_mtime_ns": stat.st_mtime_ns, "compiler_ctime_ns": stat.st_ctime_ns,
        "platform": system, "machine": platform.machine(), "cache_path": str(cache_root),
        "flags": _build_command(compiler, Path("source.cpp"), Path("output"), system=system)[1:],
    }, sort_keys=True, separators=(",", ":")).encode()
    key = hashlib.sha256(identity).hexdigest()[:24]
    build_dir = cache_root / key
    build_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".dll" if system == "Windows" else ".dylib" if system == "Darwin" else ".so"
    library_path = build_dir / f"libbhe1_{key}{suffix}"
    with _LOCK:
        loaded = _LOADED.get(library_path)
        if loaded is not None:
            return loaded
        if not library_path.is_file():
            with (build_dir / "build.lock").open("a+b") as lock_file, _native_file_lock(lock_file):
                if not library_path.is_file():
                    fd, name = tempfile.mkstemp(prefix=f"libbhe1_{key}.", suffix=f".building{suffix}", dir=build_dir)
                    os.close(fd)
                    temporary = Path(name)
                    env = os.environ.copy()
                    if system == "Darwin":
                        env["DYLD_LIBRARY_PATH"] = os.pathsep.join(filter(
                            None, ["/opt/homebrew/opt/readline/lib", env.get("DYLD_LIBRARY_PATH", "")]))
                    try:
                        completed = subprocess.run(_build_command(compiler, _SOURCE, temporary, system=system),
                                                   check=False, capture_output=True, text=True, env=env)
                        if completed.returncode:
                            raise RuntimeError(f"BHE1 native compilation failed ({completed.returncode}): {completed.stderr.strip()}")
                        if not temporary.is_file() or temporary.stat().st_size == 0:
                            raise RuntimeError("BHE1 native compiler did not produce a library")
                        os.replace(temporary, library_path)
                    except OSError as exc:
                        raise RuntimeError(f"BHE1 native compilation could not start: {exc}") from exc
                    finally:
                        if temporary.exists():
                            temporary.unlink()
        library = ctypes.CDLL(str(library_path))
        voidp, u32 = ctypes.c_void_p, ctypes.c_uint32
        library.bhe1_encode_main.argtypes = [voidp, u32, u32, voidp, voidp, voidp, voidp,
                                             voidp, voidp, u32, voidp, voidp, voidp]
        library.bhe1_encode_main.restype = ctypes.c_int
        library.bhe1_decode_main.argtypes = [voidp, u32, u32, voidp, voidp, u32, u32,
                                             voidp, voidp, voidp, voidp, voidp, voidp]
        library.bhe1_decode_main.restype = ctypes.c_int
        library.bhe1_split_checkpoints.argtypes = [voidp, ctypes.c_uint64, voidp, ctypes.c_uint64,
            voidp, voidp, voidp, voidp, voidp, u32, u32, u32, u32, u32, voidp]
        library.bhe1_split_checkpoints.restype = ctypes.c_int
        _LOADED[library_path] = library
        return library


def _load_library(cache_dir=None) -> ctypes.CDLL:
    session = _active_session()
    if session is None:
        return _load_library_uncached(cache_dir)
    cache_root, request_key = _cache_request(cache_dir)
    environment_signature = _cache_environment_signature()
    cached_directory = session.cache_directories.get(request_key)
    if cached_directory is None or cached_directory[0] != cache_root:
        cache_root = _cache_directory(cache_dir)
        session.cache_directories[request_key] = (cache_root, environment_signature)
    elif cached_directory[1] != environment_signature:
        cache_root = _cache_directory(cache_dir)
        session.cache_directories[request_key] = (cache_root, environment_signature)
    else:
        cache_root.mkdir(parents=True, exist_ok=True)
    library = session.libraries.get(cache_root)
    if library is None:
        library = _load_library_uncached(cache_root)
        session.libraries[cache_root] = library
    return library


def _ptr(array: np.ndarray):
    return ctypes.c_void_p(int(array.ctypes.data))


def _labels(values, count: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (count,) or array.dtype.kind not in "iu" or np.any(array < 0) or np.any(array > 255):
        raise ValueError(f"invalid BHE1 {name}")
    return np.ascontiguousarray(array, dtype=np.uint8)


def _table_arrays_uncached(tables, nc: int):
    if len(tables.singletons) != len(tables.codes) or len(tables.codes) % nc:
        raise ValueError("BHE1 exponent table geometry is invalid")
    contexts = len(tables.codes)
    if contexts > 256 * nc:
        raise ValueError("BHE1 exponent table geometry is invalid")
    code_values = np.zeros(256 * nc * 256, dtype="<u4")
    code_lengths = np.zeros(256 * nc * 256, dtype=np.uint8)
    singletons = np.full(256 * nc, -2, dtype="<i2")
    for cx, context_codes in enumerate(tables.codes):
        singletons[cx] = int(tables.singletons[cx])
        for code, length, symbol in context_codes:
            if not 0 <= int(symbol) < 256 or not 1 <= int(length) <= 31:
                raise ValueError("invalid BHE1 exponent code table")
            index = cx * 256 + int(symbol)
            code_values[index] = int(code)
            code_lengths[index] = int(length)
    return (np.ascontiguousarray(code_values), np.ascontiguousarray(code_lengths),
            np.ascontiguousarray(singletons))


def _table_arrays(tables, nc: int):
    session = _active_session()
    if session is None:
        return _table_arrays_uncached(tables, nc)
    # Built Tables are immutable tuples. The full exact contents are the key,
    # so changed or otherwise mutable caller-supplied tables cannot reuse stale
    # arrays; unhashable forms simply take the uncached validated path.
    codes, singletons = tables.codes, tables.singletons
    immutable = type(codes) is tuple and type(singletons) is tuple
    if immutable:
        for context in codes:
            if type(context) is not tuple or any(
                    type(entry) is not tuple or len(entry) != 3 or
                    any(type(value) is not int for value in entry) for entry in context):
                immutable = False
                break
        if any(type(value) is not int for value in singletons):
            immutable = False
    if not immutable:
        return _table_arrays_uncached(tables, nc)
    key = (nc, codes, singletons)
    try:
        hash(key)
    except TypeError:
        return _table_arrays_uncached(tables, nc)
    cached = session.tables.get(key)
    if cached is not None:
        session.tables.move_to_end(key)
        return cached
    arrays = _table_arrays_uncached(tables, nc)
    for array in arrays:
        array.flags.writeable = False
    session.tables[key] = arrays
    while len(session.tables) > _SESSION_TABLE_LIMIT:
        session.tables.popitem(last=False)
    return arrays


def _native_error(code: int, operation: str) -> None:
    if code:
        reasons = {
            "encode": {1: "invalid input geometry", 3: "invalid Huffman code table", 4: "output span overflow"},
            "decode": {1: "invalid input geometry", 2: "invalid bit directory or padding",
                       3: "invalid Huffman code table", 5: "empty exponent context carries columns",
                       6: "singleton stream carries bits", 7: "nonterminal exponent stream",
                       8: "truncated exponent stream", 9: "invalid exponent codeword"},
        }[operation]
        raise ValueError(f"native BHE1 {operation}: {reasons.get(code, f'codec error {code}')}")


def encode_main_native(exponents, row_labels, col_labels, tables, *, col_classes: int,
                       cache_dir=None) -> tuple[bytes, np.ndarray, bool]:
    """Encode all row/class streams with one global, bit-tight BHE1 buffer.

    Missing symbols produce ``coding_possible=False`` so the packet caller can
    select its established raw fallback. The returned offsets are absolute bit
    offsets and match the Python codec's stream directory.
    """
    source = np.asarray(exponents)
    if source.ndim != 2 or source.shape[0] != 64 or source.dtype.kind != "u" or source.dtype.itemsize != 1:
        raise ValueError("BHE1 exponents must be a uint8 N64,K matrix")
    if type(col_classes) is not int or col_classes not in (1, 2, 4):
        raise ValueError("invalid BHE1 column class count")
    source = np.ascontiguousarray(source, dtype=np.uint8)
    k = source.shape[1]
    rows = _labels(row_labels, 64, "row labels")
    cols = _labels(col_labels, k, "column labels")
    if np.any(cols >= col_classes):
        raise ValueError("invalid BHE1 column labels")
    if rows.size and int(rows.max()) * col_classes >= len(tables.codes):
        raise ValueError("BHE1 row labels exceed exponent table contexts")
    codes, lengths, singletons = _table_arrays(tables, col_classes)
    capacity = (64 * k * 31 + 7) // 8
    output = ctypes.create_string_buffer(max(1, capacity))
    offsets = np.empty(64 * col_classes + 1, dtype="<u4")
    byte_count = ctypes.c_uint32()
    coding_possible = ctypes.c_uint32()
    code = _load_library(cache_dir).bhe1_encode_main(
        _ptr(source), k, col_classes, _ptr(rows), _ptr(cols), _ptr(codes), _ptr(lengths),
        _ptr(singletons), ctypes.cast(output, ctypes.c_void_p), capacity, _ptr(offsets),
        ctypes.byref(byte_count), ctypes.byref(coding_possible))
    _native_error(code, "encode")
    return output.raw[:byte_count.value], offsets, bool(coding_possible.value)


def decode_main_native(main, offsets, raw8, row_labels, col_labels, tables, *,
                       col_classes: int, cache_dir=None) -> np.ndarray:
    """Decode and validate BHE1 Huffman streams into their exact uint16 words."""
    if not isinstance(main, (bytes, bytearray, memoryview)):
        raise TypeError("BHE1 main stream must be bytes-like")
    main = bytes(main)
    raw = np.asarray(raw8)
    if raw.ndim != 2 or raw.shape[0] != 64 or raw.dtype.kind != "u" or raw.dtype.itemsize != 1:
        raise ValueError("BHE1 compact source must be a uint8 N64,K matrix")
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    k = raw.shape[1]
    if type(col_classes) is not int or col_classes not in (1, 2, 4):
        raise ValueError("invalid BHE1 column class count")
    rows = _labels(row_labels, 64, "row labels")
    cols = _labels(col_labels, k, "column labels")
    if np.any(cols >= col_classes):
        raise ValueError("invalid BHE1 column labels")
    if rows.size and int(rows.max()) * col_classes >= len(tables.codes):
        raise ValueError("BHE1 row labels exceed exponent table contexts")
    bit_offsets = np.asarray(offsets)
    if (bit_offsets.shape != (64 * col_classes + 1,) or bit_offsets.dtype.kind not in "iu" or
            np.any(bit_offsets < 0) or np.any(bit_offsets > 0xffffffff)):
        raise ValueError("invalid BHE1 bit directory")
    bit_offsets = np.ascontiguousarray(bit_offsets, dtype="<u4")
    bit_count = int(bit_offsets[-1])
    if int(bit_offsets[0]) != 0 or np.any(bit_offsets[:-1] > bit_offsets[1:]):
        raise ValueError("invalid BHE1 bit directory")
    if len(main) != (bit_count + 7) // 8:
        raise ValueError("noncanonical BHE1 main byte length")
    if bit_count & 7 and main and main[-1] & ((1 << (8 - (bit_count & 7))) - 1):
        raise ValueError("nonzero unused BHE1 terminal bits")
    codes, lengths, singletons = _table_arrays(tables, col_classes)
    output = np.empty((64, k), dtype="<u2")
    main_buffer = ctypes.create_string_buffer(main, max(1, len(main)))
    code = _load_library(cache_dir).bhe1_decode_main(
        ctypes.cast(main_buffer, ctypes.c_void_p), len(main), bit_count, _ptr(bit_offsets), _ptr(raw),
        k, col_classes, _ptr(rows), _ptr(cols), _ptr(codes), _ptr(lengths), _ptr(singletons),
        ctypes.c_void_p(int(output.ctypes.data)))
    _native_error(code, "decode")
    return output


def split_checkpoints_native(arrays, splits: int, *, cache_dir=None) -> np.ndarray:
    """Build absolute resident stream checkpoints with the native Cursor/group walk."""
    if type(splits) is not int or not 1 <= splits <= 8:
        raise ValueError("splits must be 1..8")
    main = np.ascontiguousarray(arrays["main"], dtype=np.uint8)
    tables = np.ascontiguousarray(arrays["tables"], dtype=np.uint32)
    context = np.ascontiguousarray(arrays["context_dir"], dtype=np.int64)
    streams = np.ascontiguousarray(arrays["streams"], dtype=np.int64)
    rows = np.ascontiguousarray(arrays["row_map"], dtype=np.int32)
    classes = np.ascontiguousarray(arrays["tile_rc"], dtype=np.int64)
    counts = np.ascontiguousarray(arrays["counts"], dtype=np.int64)
    n = int(arrays["N"]); k = int(arrays["K"]); nc = int(arrays["nc"])
    tiles = int(arrays.get("tiles", rows.shape[0]))
    if splits > k // 128: raise ValueError("splits exceeds chunk count")
    if not (1 <= n <= 262144 and 128 <= k <= 32768 and k % 128 == 0 and nc in (1,2,4)
            and 1 <= tiles <= n and rows.shape == (tiles,64)
            and streams.shape == (tiles,64,nc,2) and classes.shape == (tiles,)
            and counts.shape == (k//128,nc+1) and context.ndim == 2 and context.shape[1] == 4):
        raise ValueError('invalid resident checkpoint geometry')
    active = rows[rows >= 0]
    if (active.size != n or not np.array_equal(np.sort(active), np.arange(n))
            or np.any(classes < 0) or np.any((classes+1)*nc > len(context))
            or np.any(streams < 0) or np.any(streams[...,0] > streams[...,1])
            or np.any(streams[...,1] > main.size*8)
            or np.any(counts[:,0] != 0) or np.any(counts[:,-1] != 128)
            or np.any(np.diff(counts,axis=1) < 0)):
        raise ValueError('invalid resident checkpoint indices')
    out = np.zeros((n, nc, splits - 1), dtype=np.int64)
    lib = _load_library(cache_dir)
    code = lib.bhe1_split_checkpoints(_ptr(main), main.size, _ptr(tables), tables.size, _ptr(context),
        _ptr(streams), _ptr(rows), _ptr(classes), _ptr(counts), n, k, nc, tiles, splits, _ptr(out))
    _native_error(code, "split checkpoints")
    return out
