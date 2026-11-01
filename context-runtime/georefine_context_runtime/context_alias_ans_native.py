"""Explicit opt-in portable C++ CPU implementation of the AAN2 word codec.

The library is compiled on demand into a persistent user cache by default.
Callers can select a persistent cache directory directly or with
``GEOREFINE_AAN2_NATIVE_CACHE``. Compilation errors are surfaced; this module
never falls back to the Python codec.
"""
from __future__ import annotations

import ctypes
from .bitexact_context_codec import _native_file_lock
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

from .context_alias_ans import (
    AliasTables, LANES, MAX_CONTEXTS, MAX_SYMBOL_ID, MAX_SYMBOLS,
    WORD_HEADER, WORD_MAGIC, WORD_RADIX, WORD_RANS_L, WORD_STATE_BYTES,
    WORD_VERSION, _sequence_count, _validate_max_symbols, _vector,
)

_HEADER_SIZE = WORD_HEADER.size
_SOURCE = Path(__file__).with_name("context_alias_ans_native.cpp")
_CACHE_ENV = "GEOREFINE_AAN2_NATIVE_CACHE"
_LOADED_LIBRARIES: dict[Path, ctypes.CDLL] = {}
_LOADED_LIBRARIES_LOCK = threading.RLock()


def _cache_directory(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    requested = cache_dir if cache_dir is not None else os.environ.get(_CACHE_ENV)
    if requested is not None and not os.fspath(requested).strip():
        raise RuntimeError("AAN2 native cache directory must be a non-empty persistent path")
    path = Path(requested).expanduser() if requested is not None else Path.home() / ".cache" / "georefine" / "aan2-native"
    path = path.resolve()
    forbidden = {
        Path(root).resolve()
        for root in ("/tmp", "/private/tmp", "/var/folders", "/private/var/folders")
    }
    if any(path == root or root in path.parents for root in forbidden):
        raise RuntimeError("AAN2 native cache may not use temporary storage")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _platform_name(system: str | None = None) -> str:
    system = platform.system() if system is None else system
    if system not in {"Darwin", "Linux", "Windows"}:
        raise RuntimeError(f"AAN2 native codec does not support platform {system!r}")
    return system


def _compiler_path(system: str) -> str:
    configured = os.environ.get("GEOREFINE_NATIVE_CXX") or os.environ.get("CXX")
    candidates = [configured] if configured else (["g++"] if system == "Windows" else ["clang++"] if system == "Darwin" else ["c++", "g++", "clang++"])
    for candidate in candidates:
        if candidate:
            compiler = shutil.which(candidate)
            if compiler:
                return str(Path(compiler).resolve())
    requested = configured or ", ".join(candidates)
    raise RuntimeError(f"AAN2 native codec requires a C++ compiler ({requested})")


@lru_cache(maxsize=16)
def _compiler_identity(compiler: str) -> str:
    try:
        result = subprocess.run([compiler, "--version"], check=False, capture_output=True, text=True)
    except OSError as exc:
        raise RuntimeError(f"AAN2 compiler identity query failed: {exc}") from exc
    if result.returncode:
        raise RuntimeError(f"AAN2 compiler identity query failed ({result.returncode}): {result.stderr.strip()}")
    version = (result.stdout or result.stderr).strip()
    if not version:
        raise RuntimeError("AAN2 compiler returned an empty version identity")
    return f"{compiler}\n{version}"


def _library_suffix(system: str) -> str:
    return ".dll" if system == "Windows" else ".dylib" if system == "Darwin" else ".so"


def _build_command(compiler: str, source: Path, output: Path, *, system: str | None = None) -> list[str]:
    system = _platform_name(system)
    if system == "Windows":
        return [compiler, "-std=c++17", "-O3", "-shared", "-static", "-static-libgcc",
                "-static-libstdc++", "-Wl,--export-all-symbols", str(source), "-o", str(output)]
    linker_mode = "-dynamiclib" if system == "Darwin" else "-shared"
    return [compiler, "-std=c++17", "-O3", "-fPIC", linker_mode, str(source), "-o", str(output)]


def _load_library(cache_dir: str | os.PathLike[str] | None = None) -> ctypes.CDLL:
    if not _SOURCE.is_file():
        raise RuntimeError(f"AAN2 native source is missing: {_SOURCE}")
    system = _platform_name()
    machine = platform.machine()
    compiler = _compiler_path(system)
    try:
        compiler_stat = os.stat(compiler)
    except OSError as exc:
        raise RuntimeError(f"AAN2 compiler stat failed: {exc}") from exc
    with _LOADED_LIBRARIES_LOCK:
        compiler_id = _compiler_identity(compiler)
    source_hash = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()
    cache_root = _cache_directory(cache_dir)
    build_identity = json.dumps({
        "source_sha256": source_hash,
        "compiler": compiler_id,
        "compiler_path": compiler,
        "compiler_device": compiler_stat.st_dev,
        "compiler_inode": compiler_stat.st_ino,
        "compiler_size": compiler_stat.st_size,
        "compiler_mtime_ns": compiler_stat.st_mtime_ns,
        "compiler_ctime_ns": compiler_stat.st_ctime_ns,
        "platform": system,
        "machine": machine,
        "cache_path": str(cache_root),
        "flags": (_build_command(compiler, Path("source.cpp"), Path("output"), system=system)[1:]
                  if system == "Windows" else
                  ["-std=c++17", "-O3", "-fPIC", "-dynamiclib" if system == "Darwin" else "-shared"]),
    }, sort_keys=True, separators=(",", ":")).encode()
    build_key = hashlib.sha256(build_identity).hexdigest()[:24]
    build_dir = cache_root / build_key
    build_dir.mkdir(parents=True, exist_ok=True)
    library_path = build_dir / f"libaan2_{build_key}{_library_suffix(system)}"
    with _LOADED_LIBRARIES_LOCK:
        loaded = _LOADED_LIBRARIES.get(library_path)
        if loaded is not None:
            return loaded
        if not library_path.is_file():
            lock_path = build_dir / "build.lock"
            with lock_path.open("a+b") as lock_file, _native_file_lock(lock_file):
                if not library_path.is_file():
                    fd, temporary_name = tempfile.mkstemp(prefix=f"libaan2_{build_key}.", suffix=f".building{_library_suffix(system)}", dir=build_dir)
                    os.close(fd)
                    temporary = Path(temporary_name)
                    command = _build_command(compiler, _SOURCE, temporary, system=system)
                    environment = os.environ.copy()
                    if system == "Darwin":
                        readline = "/opt/homebrew/opt/readline/lib"
                        environment["DYLD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [readline, environment.get("DYLD_LIBRARY_PATH", "")]))
                    try:
                        completed = subprocess.run(command, check=False, capture_output=True, text=True, env=environment)
                        if completed.returncode:
                            raise RuntimeError(f"AAN2 native compilation failed ({completed.returncode}): {completed.stderr.strip()}")
                        if not temporary.is_file() or temporary.stat().st_size == 0:
                            raise RuntimeError("AAN2 native compiler did not produce a library")
                        os.replace(temporary, library_path)
                    except OSError as exc:
                        raise RuntimeError(f"AAN2 native compilation could not start: {exc}") from exc
                    finally:
                        if temporary.exists():
                            temporary.unlink()
        library = ctypes.CDLL(str(library_path))
        voidp = ctypes.c_void_p
        uint32 = ctypes.c_uint32
        library.aan2_encode.argtypes = [voidp, voidp, uint32, voidp, uint32, voidp, voidp, voidp,
                                      uint32, voidp, voidp, voidp, voidp, uint32, uint32,
                                      voidp, voidp]
        library.aan2_encode.restype = ctypes.c_int
        library.aan2_decode.argtypes = [voidp, uint32, uint32, voidp, voidp, uint32, voidp, voidp,
                                      voidp, uint32, voidp, voidp, voidp, voidp, uint32, uint32,
                                      voidp]
        library.aan2_decode.restype = ctypes.c_int
        _LOADED_LIBRARIES[library_path] = library
        return library


def _table_arrays(tables: AliasTables):
    if not isinstance(tables, AliasTables):
        raise TypeError("tables must be AliasTables")
    if type(tables.source_cdf_digest) is not bytes or len(tables.source_cdf_digest) != 32:
        raise ValueError("alias table CDF digest is invalid")
    arrays = (
        np.ascontiguousarray(tables.bucket_records, dtype="<u2"),
        np.ascontiguousarray(tables.bucket_offsets, dtype="<u4"),
        np.ascontiguousarray(tables.context_shifts, dtype="u1"),
        np.ascontiguousarray(tables.active_symbols, dtype="<u2"),
        np.ascontiguousarray(tables.active_frequencies, dtype="<u4"),
        np.ascontiguousarray(tables.active_offsets, dtype="<u4"),
        np.ascontiguousarray(tables.inverse_symbol_offsets, dtype="<u4"),
        np.ascontiguousarray(tables.inverse_segments, dtype="<u4"),
    )
    records, bucket_offsets, shifts, active_symbols, frequencies, active_offsets, segment_offsets, segments = arrays
    contexts = tables.context_count
    if not 1 <= contexts <= MAX_CONTEXTS or records.ndim != 2 or records.shape[1:] != (8,):
        raise ValueError("alias table decoder geometry is invalid")
    if bucket_offsets.shape != (contexts + 1,) or shifts.shape != (contexts,):
        raise ValueError("alias table context arrays are invalid")
    if active_symbols.ndim != 1 or frequencies.shape != active_symbols.shape or active_offsets.shape != (contexts + 1,):
        raise ValueError("alias table active-symbol arrays are invalid")
    if segment_offsets.shape != (len(active_symbols) + 1,) or segments.ndim != 2 or segments.shape[1:] != (3,):
        raise ValueError("alias table inverse arrays are invalid")
    return arrays, contexts


def _ptr(array: np.ndarray):
    return ctypes.c_void_p(int(array.ctypes.data))


def _native_error(code: int, operation: str) -> None:
    if code:
        encode_reasons = {
            1: "alias table geometry is corrupt", 2: "symbol is absent from alias context",
            3: "word rANS requires multiple refill words", 4: "inverse alias map is corrupt",
            5: "encoded state is outside uint32 range", 6: "inactive tail lane state changed",
            7: "word payload exceeded its bound",
        }
        decode_reasons = {
            1: "alias table geometry is corrupt", 2: "invalid AAN2 initial state",
            3: "inactive AAN2 tail lane state is noncanonical", 4: "AAN2 lane state left uint32 range",
            5: "AAN2 alias table lookup is invalid", 6: "AAN2 state transition overflow",
            7: "AAN2 renormalization state is zero", 8: "truncated AAN2 common word stream",
            9: "AAN2 refill state is outside uint32 range", 10: "AAN2 terminal cursor mismatch",
            11: "AAN2 terminal state mismatch",
        }
        reasons = encode_reasons if operation == "encode" else decode_reasons
        raise ValueError(f"native AAN2 {operation}: {reasons.get(code, f'codec error {code}')}")


def encode32_word_native(symbols, contexts, tables: AliasTables, *,
                         max_symbols: int = MAX_SYMBOLS,
                         cache_dir: str | os.PathLike[str] | None = None) -> bytes:
    """Encode canonical AAN2 bytes with the explicit C++ CPU implementation."""
    cap = _validate_max_symbols(max_symbols)
    symbol_count = _sequence_count(symbols, "symbols")
    context_count = _sequence_count(contexts, "contexts")
    if symbol_count > cap or context_count > cap:
        raise ValueError("symbol/context count exceeds max_symbols")
    if symbol_count != context_count:
        raise ValueError("symbols and contexts have different lengths")
    values = _vector(symbols, "symbols", maximum=cap, max_value=MAX_SYMBOL_ID)
    context_ids = _vector(contexts, "contexts", maximum=cap, max_value=MAX_CONTEXTS - 1)
    arrays, table_contexts = _table_arrays(tables)
    records, bucket_offsets, shifts, active_symbols, frequencies, active_offsets, segment_offsets, segments = arrays
    body = ctypes.create_string_buffer(WORD_STATE_BYTES + 2 * symbol_count)
    payload_bytes = ctypes.c_uint32()
    code = _load_library(cache_dir).aan2_encode(
        _ptr(values), _ptr(context_ids), symbol_count, _ptr(records), len(records),
        _ptr(bucket_offsets), _ptr(shifts), _ptr(active_symbols), len(active_symbols),
        _ptr(frequencies), _ptr(active_offsets), _ptr(segment_offsets), _ptr(segments),
        len(segments), table_contexts, ctypes.cast(body, ctypes.c_void_p), ctypes.byref(payload_bytes),
    )
    _native_error(code, "encode")
    body_bytes = body.raw[:WORD_STATE_BYTES + payload_bytes.value]
    header = WORD_HEADER.pack(WORD_MAGIC, WORD_VERSION, LANES, 0, symbol_count, payload_bytes.value,
                              tables.source_cdf_digest, hashlib.sha256(body_bytes).digest())
    return header + body_bytes


def decode32_word_native(frame: bytes | bytearray | memoryview, contexts,
                         tables: AliasTables, *, max_symbols: int = MAX_SYMBOLS,
                         cache_dir: str | os.PathLike[str] | None = None) -> np.ndarray:
    """Decode and fully validate an AAN2 frame with the C++ CPU implementation."""
    cap = _validate_max_symbols(max_symbols)
    if not isinstance(frame, (bytes, bytearray, memoryview)):
        raise TypeError("frame must be bytes-like")
    view = memoryview(frame)
    if view.ndim != 1 or view.itemsize != 1 or not view.c_contiguous:
        raise ValueError("frame must be a contiguous byte buffer")
    context_count = _sequence_count(contexts, "contexts")
    if context_count > cap:
        raise ValueError("context count exceeds max_symbols")
    if view.nbytes < _HEADER_SIZE + WORD_STATE_BYTES:
        raise ValueError("short AAN2 frame")
    magic, version, lanes, flags, count, payload_bytes, digest, body_digest = WORD_HEADER.unpack_from(view)
    if magic != WORD_MAGIC or version != WORD_VERSION or lanes != LANES or flags:
        raise ValueError("unsupported AAN2 header")
    if count != context_count or count > cap:
        raise ValueError("AAN2 symbol count mismatch")
    if payload_bytes > 2 * count or payload_bytes & 1:
        raise ValueError("AAN2 word payload length is noncanonical")
    arrays, table_contexts = _table_arrays(tables)
    if digest != tables.source_cdf_digest:
        raise ValueError("AAN2 CDF digest mismatch")
    if view.nbytes != _HEADER_SIZE + WORD_STATE_BYTES + payload_bytes:
        raise ValueError("AAN2 frame length mismatch")
    body_view = view[_HEADER_SIZE:]
    body_bytes = bytes(body_view)
    if hashlib.sha256(body_bytes).digest() != body_digest:
        raise ValueError("AAN2 body checksum mismatch")
    context_ids = _vector(contexts, "contexts", maximum=cap, max_value=MAX_CONTEXTS - 1)
    records, bucket_offsets, shifts, active_symbols, frequencies, active_offsets, segment_offsets, segments = arrays
    output = np.empty(count, dtype="<u2")
    code = _load_library(cache_dir).aan2_decode(
        ctypes.c_char_p(body_bytes), count, payload_bytes, _ptr(context_ids), _ptr(records), len(records),
        _ptr(bucket_offsets), _ptr(shifts), _ptr(active_symbols), len(active_symbols), _ptr(frequencies),
        _ptr(active_offsets), _ptr(segment_offsets), _ptr(segments), len(segments), table_contexts,
        ctypes.c_void_p(int(output.ctypes.data)),
    )
    _native_error(code, "decode")
    return output
