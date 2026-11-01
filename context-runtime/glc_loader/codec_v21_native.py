"""Default-off exact C++ TBE21 packing and bounded CPU inverse verification.

Only compiled code is cached. No source words, codebooks or model payloads are retained.
V21Tensor serialization and the independent NumPy reference decoder are unchanged.
"""
from __future__ import annotations
import ctypes
import hashlib
from functools import lru_cache
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading

import numpy as np

_FLAG = "GEOREFINE_TBE21_NATIVE_PACK"
_CACHE_ENV = "GEOREFINE_TBE21_NATIVE_CACHE"
_SOURCE = Path(__file__).with_suffix('.cpp')
_LOCK = threading.RLock()
_LOADED = {}
_CONFIG_CACHE = None
_COMPILER_ID_TIMEOUT_S = 10
_COMPILE_TIMEOUT_S = 60


class NativePackError(RuntimeError):
    """A selected native packer fails closed; there is no Python fallback."""


def enabled(environ=None):
    env = os.environ if environ is None else environ
    value = env.get(_FLAG, '0')
    if value not in ('0', '1'):
        raise NativePackError(f'{_FLAG} must be exactly 0 or 1')
    return value == '1'


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    return {name: _digest(Path(__file__).with_name(name)) for name in
            ('codec_v21.py', 'codec_v21_native.py', 'codec_v21_native.cpp')}


def _cache_root(cache_dir=None):
    requested = cache_dir if cache_dir is not None else (_CONFIG_CACHE or os.environ.get(_CACHE_ENV))
    if requested is None:
        requested = Path.home() / '.cache' / 'georefine' / 'tbe21-native'
    if not os.fspath(requested).strip():
        raise NativePackError('native pack cache must be a non-empty durable path')
    root = Path(requested).expanduser().resolve()
    forbidden = ('/tmp', '/private/tmp', '/var/folders', '/private/var/folders')
    if any(root.is_relative_to(Path(p)) for p in forbidden) or root.is_relative_to(_SOURCE.parent.resolve()):
        raise NativePackError('native pack cache must be durable and outside the installed loader')
    root.mkdir(parents=True, exist_ok=True)
    return root


def configure_build_root(build_root):
    """Point spawned/serial startup at the explicit source-only compiler directory."""
    global _CONFIG_CACHE
    if not enabled():
        return
    root = Path(build_root).expanduser().resolve()
    if '.scratch' not in root.parts:
        raise NativePackError('native pack startup build root must be durable .scratch')
    _CONFIG_CACHE = _cache_root(root / 'tbe21-native')


@lru_cache(maxsize=8)
def _compiler_version(compiler, fingerprint):
    try:
        result = subprocess.run([compiler, '--version'], capture_output=True, text=True,
                                timeout=_COMPILER_ID_TIMEOUT_S, close_fds=sys.platform != 'darwin')
    except subprocess.TimeoutExpired as exc:
        raise NativePackError(f'native TBE21 compiler identity exceeded {_COMPILER_ID_TIMEOUT_S}s: {compiler}') from exc
    except OSError as exc:
        raise NativePackError(f'native TBE21 compiler identity could not start: {compiler}: {exc}') from exc
    if result.returncode or not (result.stdout or result.stderr).strip():
        raise NativePackError('native TBE21 compiler identity query failed')
    return (result.stdout or result.stderr).strip()


def _recipe():
    system = platform.system()
    if system not in ('Darwin', 'Linux') or sys.byteorder != 'little':
        raise NativePackError('selected native packer currently requires little-endian Linux or macOS')
    requested = os.environ.get('GEOREFINE_NATIVE_CXX') or os.environ.get('CXX') or 'c++'
    compiler = shutil.which(requested)
    if compiler is None:
        raise NativePackError(f'native TBE21 packer requires a C++ compiler: {requested}')
    compiler = os.path.abspath(compiler)  # retain clang++ driver spelling, including symlink name
    stat = Path(compiler).stat()
    fingerprint = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    compiler_version = _compiler_version(compiler, fingerprint)
    sources = source_identity()
    flags = ['-O3', '-std=c++17', '-fPIC', '-dynamiclib' if system == 'Darwin' else '-shared',
             f'-DT21_SOURCE_SHA256="{sources["codec_v21_native.cpp"]}"']
    return {'abi': 1, 'source_sha256': sources, 'compiler': compiler,
            'compiler_version': compiler_version,
            'compiler_stat': [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns],
            'flags': flags, 'system': system, 'machine': platform.machine(),
            'pointer_bytes': ctypes.sizeof(ctypes.c_void_p), 'threads': 1}


def _bind(lib, expected_cpp_sha):
    U64 = ctypes.c_uint64; U32 = ctypes.c_uint32
    P8 = ctypes.POINTER(ctypes.c_uint8); P16 = ctypes.POINTER(ctypes.c_uint16)
    P32 = ctypes.POINTER(U32); P64 = ctypes.POINTER(U64)
    common = [P16, U64, U64, U64, P8, U64, U32, P8, U64, P32, U64]
    lib.t21_plan.argtypes = common + [P64, U64]; lib.t21_plan.restype = ctypes.c_int
    lib.t21_fill.argtypes = common + [P8, U64, P16, U64, P32, U64]; lib.t21_fill.restype = ctypes.c_int
    lib.t21_advance_row.argtypes = [U64, U64, P32, P64]; lib.t21_advance_row.restype = ctypes.c_int
    lib.t21_pack_abi.argtypes = []; lib.t21_pack_abi.restype = U32
    if lib.t21_pack_abi() != 1:
        raise NativePackError('native TBE21 ABI identity mismatch')
    lib.t21_pack_source_sha256.argtypes = []
    lib.t21_pack_source_sha256.restype = ctypes.c_char_p
    if lib.t21_pack_source_sha256() != expected_cpp_sha.encode('ascii'):
        raise NativePackError('native TBE21 compiled CPP source identity mismatch')
    lib.t21_decode.argtypes = [P8, U64, P16, U64, P32, U64, P32, U64,
                               P8, U64, U32, P8, U64, U64, U64, P16, U64]
    lib.t21_decode.restype = ctypes.c_int
    return lib


def library(cache_dir=None):
    from georefine_context_runtime.bitexact_context_codec import _native_file_lock
    recipe = _recipe()
    key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    directory = _cache_root(cache_dir) / key
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / ('tbe21_pack.dylib' if recipe['system'] == 'Darwin' else 'tbe21_pack.so')
    record = directory / 'build.json'
    with _LOCK, (directory / 'build.lock').open('a+b') as lock, _native_file_lock(lock):
        if binary.exists() or record.exists():
            if (binary.is_symlink() or record.is_symlink() or not binary.is_file() or not record.is_file()):
                raise NativePackError('native TBE21 build cache is incomplete or unsafe')
            try:
                receipt = json.loads(record.read_text())
            except (OSError, ValueError) as exc:
                raise NativePackError(f'native TBE21 build receipt is unreadable: {record}') from exc
            if (not isinstance(receipt, dict) or set(receipt) != {'recipe', 'key', 'binary_path', 'binary_sha256'}
                    or receipt.get('recipe') != recipe or receipt.get('key') != key
                    or receipt.get('binary_path') != str(binary) or receipt.get('binary_sha256') != _digest(binary)):
                raise NativePackError('native TBE21 build cache identity mismatch')
        else:
            fd, temporary = tempfile.mkstemp(prefix='.packer-', suffix='.partial', dir=directory)
            os.close(fd)
            source_fd, source_temporary = tempfile.mkstemp(prefix='.packer-source-', suffix='.cpp', dir=directory)
            os.close(source_fd)
            try:
                snapshot = _SOURCE.read_bytes()
                expected_cpp_sha = recipe['source_sha256']['codec_v21_native.cpp']
                if hashlib.sha256(snapshot).hexdigest() != expected_cpp_sha:
                    raise NativePackError('native TBE21 CPP source identity changed before build')
                Path(source_temporary).write_bytes(snapshot)
                command = [recipe['compiler'], *recipe['flags'], source_temporary, '-o', temporary]
                try:
                    run = subprocess.run(command, capture_output=True, text=True,
                                         timeout=_COMPILE_TIMEOUT_S, close_fds=sys.platform != 'darwin')
                except subprocess.TimeoutExpired as exc:
                    raise NativePackError(f'native TBE21 compile exceeded {_COMPILE_TIMEOUT_S}s; compiler={recipe["compiler"]}, cache={directory}') from exc
                except OSError as exc:
                    raise NativePackError(f'native TBE21 compiler could not start: {recipe["compiler"]}: {exc}') from exc
                (directory / 'compiler.log').write_text(run.stdout + run.stderr)
                if run.returncode:
                    raise NativePackError('native TBE21 build failed: ' + run.stderr[-2000:])
                if _recipe() != recipe or _digest(source_temporary) != expected_cpp_sha:
                    raise NativePackError('native TBE21 source/compiler identity changed during build; retry from stable sources')
                os.replace(temporary, binary)
            finally:
                if Path(temporary).exists():
                    Path(temporary).unlink()
                if Path(source_temporary).exists():
                    Path(source_temporary).unlink()
            receipt = {'recipe': recipe, 'key': key, 'binary_path': str(binary),
                       'binary_sha256': _digest(binary)}
            partial = directory / 'build.json.partial'
            partial.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
            os.replace(partial, record)
        binary_sha = receipt['binary_sha256']
        loaded = _LOADED.get(binary)
        if loaded is not None and loaded[0] != binary_sha:
            raise NativePackError('native TBE21 loaded binary identity changed; restart the process')
        if loaded is None:
            try:
                loaded = (binary_sha, _bind(ctypes.CDLL(str(binary)), recipe['source_sha256']['codec_v21_native.cpp']))
            except (OSError, AttributeError) as exc:
                raise NativePackError(f'native TBE21 ABI could not load: {binary}: {exc}') from exc
            _LOADED[binary] = loaded
        emitted = dict(receipt, loaded_binary_sha256=loaded[0],
                       compiled_cpp_sha256=loaded[1].t21_pack_source_sha256().decode('ascii'),
                       build_record_sha256=_digest(record))
        if emitted['compiled_cpp_sha256'] != recipe['source_sha256']['codec_v21_native.cpp']:
            raise NativePackError('native TBE21 compiled CPP source identity changed')
        return loaded[1], emitted


def prewarm(build_root=None):
    if not enabled():
        return None
    if build_root is not None:
        configure_build_root(build_root)
    return library()[1]


def _pointer(array, scalar):
    return array.ctypes.data_as(ctypes.POINTER(scalar))


def _check(status, phase):
    if status:
        raise NativePackError(f'native TBE21 {phase} failed with status {status}')


def pack(rows, luts, selectors):
    """Pack only prepared integer inputs; no codebook policy or floating arithmetic here."""
    if (not isinstance(rows, np.ndarray) or rows.dtype != np.uint16 or rows.ndim != 2
            or not rows.flags.c_contiguous or not all(rows.shape) or rows.shape[1] % 128):
        raise NativePackError('native pack rows must be non-empty contiguous uint16 with K%128=0')
    if (not isinstance(luts, np.ndarray) or luts.dtype != np.uint8 or luts.ndim != 2
            or luts.shape[1] != 256 or luts.shape[0] not in (1, 2) or not luts.flags.c_contiguous):
        raise NativePackError('native pack LUT must be contiguous uint8 [1|2,256]')
    if (not isinstance(selectors, np.ndarray) or selectors.dtype != np.uint8
            or selectors.shape != (rows.shape[0],) or not selectors.flags.c_contiguous):
        raise NativePackError('native pack selectors must be contiguous uint8 [N]')
    lib, receipt = library()
    n, k = rows.shape; rowoff = np.empty(n, np.uint32); stats = np.zeros(7, np.uint64)
    common = (_pointer(rows, ctypes.c_uint16), rows.size, n, k,
              _pointer(luts, ctypes.c_uint8), luts.size, luts.shape[0],
              _pointer(selectors, ctypes.c_uint8), selectors.size,
              _pointer(rowoff, ctypes.c_uint32), rowoff.size)
    _check(lib.t21_plan(*common, _pointer(stats, ctypes.c_uint64), stats.size), 'plan')
    smb = np.empty(rows.shape, np.uint8); l1 = np.empty((n, k // 128, 16), np.uint16)
    ovf = np.empty(int(stats[6]), np.uint32)
    _check(lib.t21_fill(*common, _pointer(smb, ctypes.c_uint8), smb.size,
                      _pointer(l1, ctypes.c_uint16), l1.size,
                      _pointer(ovf, ctypes.c_uint32), ovf.size), 'fill')
    counts = {name: int(stats[i]) for i, name in enumerate(('n1', 'n2', 'n3', 'n4', 'pad_bits'))}
    counts['max_chunk_payload_bits'] = int(stats[5])
    counts['slow_path_chunks_upper'] = None
    counts['ncb'] = int(luts.shape[0])
    return smb, l1, ovf, rowoff, counts, receipt


def decode_tensor(t, check_crc=True, check_sha=True):
    """Decode a TBE21 tensor with the compiled CPU decoder and Python integrity checks."""
    from . import codec_v21 as cv
    if t.geom.mode == cv.MODE_RAW:
        return cv.decode_tensor(t, check_crc=check_crc, check_sha=check_sha)
    g = t.geom
    if g.mode not in (cv.MODE_ROWS, cv.MODE_FLAT) or t.dtype != 'BF16':
        raise NativePackError('native TBE21 decoder supports coded BF16 rows and flat streams only')
    try:
        if cv.geometry_for(g.shape, t.dtype) != g:
            raise NativePackError('native decode geometry disagrees with the tensor shape')
    except (TypeError, ValueError, OverflowError) as exc:
        raise NativePackError('native decode geometry is invalid') from exc
    n, k = int(g.N), int(g.Kp)
    if (n <= 0 or k <= 0 or k % 128 or g.K <= 0 or g.K > k or
            g.nch != k // 128 or g.Kp != g.nch * 128):
        raise NativePackError('native decode geometry is invalid')
    expected = {
        'smb': (t.smb, np.uint8, (n, k)),
        'l1': (t.l1, np.uint16, (n, k // 128, 16)),
        'ovf': (t.ovf, np.uint32, None),
        'rowoff': (t.rowoff, np.uint32, (n,)),
        'cb': (t.cb, np.uint8, None),
        'cbsel': (t.cbsel, np.uint8, ((n + 7) // 8,)),
    }
    for name, (array, dtype, shape) in expected.items():
        ndim = 1 if name in ('ovf', 'cbsel', 'rowoff') else (3 if name == 'l1' else 2)
        if not isinstance(array, np.ndarray) or array.dtype != dtype or array.ndim != ndim:
            raise NativePackError(f'native decode {name} has invalid dtype or dimensions')
        if shape is not None and array.shape != shape:
            raise NativePackError(f'native decode {name} has invalid shape')
        if not array.flags.c_contiguous or not array.flags.aligned:
            raise NativePackError(f'native decode {name} must be aligned and C-contiguous')
    ncb = int(t.cb.shape[0])
    if ncb not in (1, 2) or t.cb.shape != (ncb, 16):
        raise NativePackError('native decode codebook must have shape [1|2,16]')
    if t.cbsel.shape != ((n + 7) // 8,) or t.ovf.size >= 1 << 32:
        raise NativePackError('native decode stream capacity is invalid')
    if check_crc:
        cv.verify_crc(t)
    out = np.empty((n, k), dtype=np.uint16)
    lib, receipt = library()
    status = lib.t21_decode(
        _pointer(t.smb, ctypes.c_uint8), t.smb.size,
        _pointer(t.l1, ctypes.c_uint16), t.l1.size,
        _pointer(t.ovf, ctypes.c_uint32) if t.ovf.size else ctypes.POINTER(ctypes.c_uint32)(), t.ovf.size,
        _pointer(t.rowoff, ctypes.c_uint32), t.rowoff.size,
        _pointer(t.cb, ctypes.c_uint8), t.cb.size, ncb,
        _pointer(t.cbsel, ctypes.c_uint8), t.cbsel.size,
        n, k, _pointer(out, ctypes.c_uint16), out.size)
    _check(status, 'decode')
    flat = out[:, :g.K].reshape(-1) if g.mode == cv.MODE_ROWS else out.reshape(-1)[:g.numel]
    result = np.ascontiguousarray(flat).reshape(g.shape)
    if check_sha and hashlib.sha256(result.view(np.uint8).reshape(-1).data).hexdigest() != t.sha256:
        raise cv.CodecV21Error(f'{t.name}: sha256 mismatch after native decode')
    t._native_decoder_receipt = dict(receipt, mode='native_tbe21_roundtrip',
                                     check_crc=bool(check_crc), check_sha=bool(check_sha))
    return result


def encode_prepared(name, dtype, geom, sha256, rows, luts, codebooks, selectors):
    from . import codec_v21 as cv
    if (cv.CH, cv.PROFILE, cv.RAW, cv.FIRST) != (128, (2, 2, 2, 3), 16, (0, 3, 6, 9)):
        raise NativePackError('native TBE21 packer does not implement this codec profile')
    smb, l1, ovf, rowoff, counts, receipt = pack(rows, luts, selectors)
    tensor = cv.V21Tensor(name, dtype, geom, sha256, cb=codebooks.astype(np.uint8),
                         cbsel=np.packbits(selectors.astype(np.uint8), bitorder='little'),
                         smb=smb, l1=l1, ovf=ovf, rowoff=rowoff)
    tensor.counts = counts
    tensor.crc = {key: cv._crc(value) for key, value in tensor.streams().items()}
    # Provenance is deliberately outside the frozen serialized codec/header bytes.
    tensor._native_packer_receipt = receipt
    return tensor
