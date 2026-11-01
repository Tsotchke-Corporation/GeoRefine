"""On-disk cache of the verified Context-to-TBE startup transcode.

Interface module shared by every Context server entry point. The cache itself
(key derivation, atomic write, verified load) is implemented below the CLI
helpers.

Layout (one directory per cache key under the cache root)::

    <root>/<key>/                     complete entry, renamed into place last
        shard-00000.safetensors ...   ~2 GiB safetensors shards
        index.json                    per-tensor rows + per-shard SHA-256
        COMPLETE                      SHA-256 of index.json, written last
    <root>/<key>.partial-<host>-<pid>-<nonce>/   in-progress write (ignored by readers)

The key is the SHA-256 of: the package ``manifest.json`` SHA-256, the runtime
version, :data:`CACHE_FORMAT`, the TBE kernel arch / CUDA capability of the target
device, and every conversion flag that changes cached content. Each tensor row
also records the package ``source_sha256`` it was transcoded from, and a reader
refuses any row whose source SHA-256 differs from the live package wrapper.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import socket
import struct
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

VERIFY_MODES = ("hash", "full")
PACKAGE_VERIFY_MODES = ("auto", "full")
ENV_CACHE_DIR = "GEOREFINE_CONTEXT_CACHE"

CACHE_FORMAT = 1
INDEX_SCHEMA = "georefine.context_transcode_cache.v1"
INDEX_NAME = "index.json"
COMPLETE_NAME = "COMPLETE"
SHARD_TARGET_BYTES = 2 * 1024**3
# Free space that must remain after the cache is written.
FREE_SPACE_RESERVE_BYTES = 2 * 1024**3
# TBE v1 stores 0.375 B/elem planes + 1 B/elem sign/mantissa + escapes; 0.75 of
# the BF16 bytes is a deliberate over-estimate for the up-front free-space check.
TBE_STORED_FRACTION_ESTIMATE = 0.75
READ_CHUNK_BYTES = 8 * 1024**2
MAX_HEADER_BYTES = 256 * 1024**2

KIND_TBE = "tbe_v1_linear"
KIND_EMBEDDING = "bf16_input_embedding"
KIND_VISION_LINEAR = "dense_bf16_linear"
KIND_VISION_EMBEDDING = "dense_bf16_embedding"
KINDS = (KIND_TBE, KIND_EMBEDDING, KIND_VISION_LINEAR, KIND_VISION_EMBEDDING)


CACHE_AUTO = "auto"
CACHE_OFF = "off"


def default_cache_root() -> Path:
    """``$GEOREFINE_CONTEXT_CACHE``, else ``$XDG_CACHE_HOME/georefine-context/transcode``,
    else ``~/.cache/georefine-context/transcode``."""
    env = os.environ.get(ENV_CACHE_DIR)
    if env and env.strip().lower() not in (CACHE_AUTO, CACHE_OFF):
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "georefine-context" / "transcode"


def add_cli_args(parser) -> None:
    """Add the transcode-cache flags to an argparse parser."""
    parser.add_argument(
        "--transcode-cache", default=None, metavar="auto|off|DIR",
        help=("verified Context-to-TBE transcode cache (disabled unless requested). 'auto' uses "
              f"${ENV_CACHE_DIR}, else $XDG_CACHE_HOME/georefine-context/transcode, else "
              "~/.cache/georefine-context/transcode; a DIR uses that directory; 'off' (or "
              f"{ENV_CACHE_DIR}=off with 'auto') disables it. The first start writes about "
              "0.75x the BF16 text-weight bytes there; if the directory is unusable or lacks "
              "the space, the server logs one warning and serves without the cache"))
    parser.add_argument(
        "--transcode-cache-verify", choices=VERIFY_MODES, default="hash",
        help=("warm-start verification: 'hash' checks every cache file's SHA-256 against the "
              "cache index and every tensor's source SHA-256 against the package manifest; "
              "'full' additionally decodes every cached tensor and re-checks its BF16 SHA-256"))
    parser.add_argument(
        "--verify-package", choices=PACKAGE_VERIFY_MODES, default="auto",
        help=("package integrity at startup: 'full' hashes every package file at resolver "
              "open and close; 'auto' does that unless a complete transcode cache bound to "
              "this exact package manifest exists, in which case it hashes every non-frame "
              "file and each tensor frame when it is read (unread frames are not used)"))


def resolve_setting(raw: Any = None) -> Dict[str, Any]:
    """Resolve a ``--transcode-cache`` value to ``{"mode", "root", "source", "reason"}``.

    ``None`` uses an explicit environment setting, otherwise disables the cache. ``mode`` is
    ``"dir"`` with ``root`` set, or ``"off"`` with ``root`` None and ``reason`` saying why.
    Raises ValueError for an empty value.
    """
    implicit_env = os.environ.get(ENV_CACHE_DIR) if raw is None else None
    value = (implicit_env or CACHE_OFF) if raw is None else str(raw)
    if not value.strip():
        raise ValueError("--transcode-cache must be 'auto', 'off' or a directory")
    lowered = value.strip().lower()
    if lowered == CACHE_OFF:
        return {"mode": "off", "root": None, "source": "--transcode-cache off",
                "reason": "disabled by --transcode-cache off"}
    if lowered != CACHE_AUTO:
        return {"mode": "dir", "root": str(Path(value).expanduser()),
                "source": f"${ENV_CACHE_DIR}" if implicit_env else "--transcode-cache",
                "reason": None}
    env = os.environ.get(ENV_CACHE_DIR)
    if env is not None and env.strip().lower() == CACHE_OFF:
        return {"mode": "off", "root": None, "source": f"${ENV_CACHE_DIR}=off",
                "reason": f"disabled by ${ENV_CACHE_DIR}=off"}
    if env and env.strip().lower() != CACHE_AUTO:
        source = f"${ENV_CACHE_DIR}"
    elif os.environ.get("XDG_CACHE_HOME"):
        source = "$XDG_CACHE_HOME default"
    else:
        source = "~/.cache default"
    try:
        root = default_cache_root()
    except (RuntimeError, KeyError, OSError) as exc:  # no resolvable home directory
        return {"mode": "off", "root": None, "source": source,
                "reason": f"no default cache directory ({type(exc).__name__}: {exc})"}
    return {"mode": "dir", "root": str(root), "source": source, "reason": None}


def kwargs_from_args(args) -> Dict[str, Any]:
    """Loader keyword arguments for ``load_predictive_fast_session`` from parsed flags.

    The cache is opt-in by flag or environment. Explicit ``auto`` selects the default
    directory; ``off`` always disables it.
    """
    verify = getattr(args, "transcode_cache_verify", "hash")
    if verify not in VERIFY_MODES:
        raise ValueError(f"--transcode-cache-verify must be one of {VERIFY_MODES}")
    setting = resolve_setting(getattr(args, "transcode_cache", None))
    return {"transcode_cache": setting["root"], "transcode_cache_verify": verify}


def explicit_cache_dir(args) -> bool:
    """True when ``--transcode-cache`` names a directory (not ``auto``/``off``/unset)."""
    raw = getattr(args, "transcode_cache", None)
    return raw is not None and str(raw).strip().lower() not in (CACHE_AUTO, CACHE_OFF)


def describe(args) -> str:
    """One log line naming the resolved cache directory and where it came from."""
    setting = resolve_setting(getattr(args, "transcode_cache", None))
    if setting["mode"] == "off":
        return f"[transcode-cache] disabled ({setting['reason']})"
    root = setting["root"]
    line = (f"[transcode-cache] directory {root} ({setting['source']}; "
            "disable with --transcode-cache off)")
    if _is_volatile(Path(root)):
        line += "; WARNING this directory is under /tmp and may be cleared on reboot"
    return line


def _is_volatile(path: Path) -> bool:
    try:
        resolved = path.expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        resolved = path
    parts = resolved.parts
    return len(parts) >= 2 and (parts[1] == "tmp" or parts[1:3] == ("private", "tmp"))


def package_verify_kwargs(args) -> Dict[str, Any]:
    """``{"verify_package": ...}`` for ``load_predictive_fast_session`` from parsed flags."""
    mode = getattr(args, "verify_package", "auto")
    if mode not in PACKAGE_VERIFY_MODES:
        raise ValueError(f"--verify-package must be one of {PACKAGE_VERIFY_MODES}")
    return {"verify_package": mode}


# ---------------------------------------------------------------------------
# errors and small helpers
# ---------------------------------------------------------------------------
class TranscodeCacheError(RuntimeError):
    """The cache could not be used; startup falls back to a full transcode."""


class CacheVerifyError(TranscodeCacheError):
    """A cache entry failed verification. ``where`` names the file or tensor."""

    def __init__(self, message: str, where: str = ""):
        super().__init__(f"{where}: {message}" if where else message)
        self.where = where


def file_sha256(path: Path, chunk: int = READ_CHUNK_BYTES) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def cache_key(material: Dict[str, Any]) -> str:
    """SHA-256 over the canonical JSON key material."""
    return hashlib.sha256(_canonical(material)).hexdigest()


def key_material(*, manifest_sha256: str, runtime_version: str, device_identity: Dict[str, Any],
                 flags: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64:
        raise TranscodeCacheError("package manifest SHA-256 is required for the cache key")
    return {
        "cache_format": CACHE_FORMAT,
        "schema": INDEX_SCHEMA,
        "package_manifest_sha256": manifest_sha256,
        "runtime_version": str(runtime_version),
        "device": dict(device_identity),
        "flags": dict(flags),
    }


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_file_durable(path: Path, data: bytes) -> None:
    with path.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


# ---------------------------------------------------------------------------
# tensor (de)serialization: a minimal safetensors writer/reader
# ---------------------------------------------------------------------------
_ST_DTYPES = {
    "U8": ("uint8", 1), "I32": ("int32", 4), "I64": ("int64", 8),
    "BF16": ("bfloat16", 2), "F16": ("float16", 2), "F32": ("float32", 4),
}


def _torch_dtype(code: str):
    import torch
    return getattr(torch, _ST_DTYPES[code][0])


def _st_code(dtype) -> str:
    name = str(dtype).replace("torch.", "")
    for code, (tname, _size) in _ST_DTYPES.items():
        if tname == name:
            return code
    raise TranscodeCacheError(f"unsupported cache tensor dtype {dtype}")


def _raw_bytes(tensor) -> memoryview:
    import torch
    t = tensor.detach().contiguous().cpu().reshape(-1)
    if t.numel() == 0:
        return memoryview(b"")
    return memoryview(t.view(torch.uint8).numpy()).cast("B")


def _u32_words_int32(values):
    """int32 tensor holding the same 32-bit words as an int32/int64 TBE word array."""
    import numpy as np
    import torch
    if values.dtype == torch.int32:
        return values.detach().contiguous().cpu()
    arr = values.detach().contiguous().cpu().numpy()
    if arr.size and (int(arr.min()) < -0x80000000 or int(arr.max()) > 0xFFFFFFFF):
        raise TranscodeCacheError("TBE word array does not fit 32 bits")
    return torch.from_numpy(np.ascontiguousarray(arr.astype("<i8") & 0xFFFFFFFF).astype("<u4").view("<i4"))


def _restore_words(stored, dtype_name: str):
    """Inverse of :func:`_u32_words_int32` for the recorded original dtype."""
    import numpy as np
    import torch
    if dtype_name == "int32":
        return stored
    if dtype_name != "int64":
        raise CacheVerifyError(f"unsupported recorded word dtype {dtype_name!r}")
    arr = stored.contiguous().numpy().view("<u4").astype("<i8")
    return torch.from_numpy(np.ascontiguousarray(arr))


def write_safetensors(path: Path, tensors: Sequence[Tuple[str, Any]],
                      metadata: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    """Write a safetensors file, fsync it and return ``(bytes, sha256)``.

    Data is streamed tensor by tensor (no full-file buffer) and hashed as written.
    """
    header: Dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    offset = 0
    for name, tensor in tensors:
        nbytes = int(tensor.numel() * tensor.element_size())
        header[name] = {"dtype": _st_code(tensor.dtype), "shape": [int(x) for x in tensor.shape],
                        "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw_header += b" " * ((-len(raw_header)) % 8)
    h = hashlib.sha256()
    total = 0
    with Path(path).open("wb") as f:
        for block in (struct.pack("<Q", len(raw_header)), raw_header):
            f.write(block)
            h.update(block)
            total += len(block)
        for _name, tensor in tensors:
            block = _raw_bytes(tensor)
            f.write(block)
            h.update(block)
            total += len(block)
        f.flush()
        os.fsync(f.fileno())
    return total, h.hexdigest()


def read_verified_shard(path: Path, expected_sha256: str, expected_bytes: int,
                        on_bytes: Optional[Callable[[int], None]] = None) -> Dict[str, Any]:
    """Read one shard in chunks, check size and SHA-256, then parse its tensors.

    Tensors are zero-copy views of the verified buffer when aligned, else copies.
    """
    import torch
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise CacheVerifyError(f"cannot stat shard: {exc}", str(path)) from exc
    if size != expected_bytes:
        raise CacheVerifyError(f"shard size {size} != indexed {expected_bytes}", str(path))
    buf = bytearray(size)
    view = memoryview(buf)
    h = hashlib.sha256()
    off = 0
    with path.open("rb", buffering=0) as f:
        while off < size:
            n = f.readinto(view[off:off + READ_CHUNK_BYTES])
            if not n:
                raise CacheVerifyError("shard truncated while reading", str(path))
            h.update(view[off:off + n])
            off += n
            if on_bytes is not None:
                on_bytes(n)
    if h.hexdigest() != expected_sha256:
        raise CacheVerifyError("shard SHA-256 differs from cache index", str(path))
    if size < 8:
        raise CacheVerifyError("shard too small for a safetensors header", str(path))
    (hlen,) = struct.unpack("<Q", bytes(view[:8]))
    if hlen > MAX_HEADER_BYTES or 8 + hlen > size:
        raise CacheVerifyError("invalid safetensors header length", str(path))
    try:
        header = json.loads(bytes(view[8:8 + hlen]).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise CacheVerifyError(f"invalid safetensors header: {exc}", str(path)) from exc
    data0 = 8 + hlen
    out: Dict[str, Any] = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        try:
            code = spec["dtype"]
            shape = [int(x) for x in spec["shape"]]
            begin, end = (int(x) for x in spec["data_offsets"])
            _tname, itemsize = _ST_DTYPES[code]
        except (KeyError, TypeError, ValueError) as exc:
            raise CacheVerifyError(f"malformed tensor header for {name}", str(path)) from exc
        count = math.prod(shape)
        if begin < 0 or end < begin or data0 + end > size or end - begin != count * itemsize:
            raise CacheVerifyError(f"tensor {name} offsets out of range", str(path))
        if count == 0:
            out[name] = torch.empty(shape, dtype=_torch_dtype(code))
            continue
        start = data0 + begin
        raw = torch.frombuffer(buf, dtype=torch.uint8, count=end - begin, offset=start)
        if start % itemsize:
            raw = raw.clone()
        out[name] = raw.view(_torch_dtype(code)).reshape(shape)
    return out


# ---------------------------------------------------------------------------
# entries
# ---------------------------------------------------------------------------
def expected_entry(module_name: str, weight_name: str, runtime_kind: str, shape, source_sha256) -> Dict[str, Any]:
    if runtime_kind not in KINDS:
        raise TranscodeCacheError(f"unknown cache kind {runtime_kind!r}")
    return {"module_name": module_name, "weight_name": weight_name, "runtime_kind": runtime_kind,
            "shape": [int(x) for x in shape], "source_sha256": source_sha256}


def _container_fields(container) -> Dict[str, Any]:
    return {
        "shape": [int(x) for x in container.shape], "layout": str(container.layout),
        "mode": int(container.mode), "base": int(container.base), "tiles": int(container.tiles),
        "escapes": int(container.escapes), "superblock": int(container.superblock),
        "planes_dtype": str(container.planes.dtype).replace("torch.", ""),
        "planes_shape": [int(x) for x in container.planes.shape],
        "sbbase_dtype": str(container.sbbase.dtype).replace("torch.", ""),
        "sbbase_shape": [int(x) for x in container.sbbase.shape],
        "smb_shape": [int(x) for x in container.smb.shape],
        "esc_shape": [int(x) for x in container.esc.shape],
    }


def _rebuild_container(row: Dict[str, Any], tensors: Dict[str, Any]):
    from glc_loader.tbe_container import TBETensor
    meta = row["container"]
    key = row["weight_name"]
    try:
        planes = _restore_words(tensors[f"{key}::planes"], meta["planes_dtype"]).reshape(meta["planes_shape"])
        sbbase = _restore_words(tensors[f"{key}::sbbase"], meta["sbbase_dtype"]).reshape(meta["sbbase_shape"])
        smb = tensors[f"{key}::smb"].reshape(meta["smb_shape"])
        esc = tensors[f"{key}::esc"].reshape(meta["esc_shape"])
    except KeyError as exc:
        raise CacheVerifyError(f"shard lacks tensor plane {exc}", key) from exc
    except RuntimeError as exc:
        raise CacheVerifyError(f"tensor plane shape mismatch: {exc}", key) from exc
    container = TBETensor(
        shape=tuple(int(x) for x in meta["shape"]), layout=meta["layout"], mode=int(meta["mode"]),
        base=int(meta["base"]), tiles=int(meta["tiles"]), escapes=int(meta["escapes"]),
        planes=planes, smb=smb, esc=esc, sbbase=sbbase, superblock=int(meta["superblock"]),
    )
    if [int(x) for x in container.shape] != [int(x) for x in row["shape"]]:
        raise CacheVerifyError("container shape differs from indexed tensor shape", key)
    return container


def _bf16_sha256(tensor) -> str:
    import torch
    raw = tensor.detach().contiguous().view(torch.int16).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _full_check(row: Dict[str, Any], payload) -> None:
    """Decode a cached TBE container on CPU and compare its BF16 SHA-256 with the source."""
    if row["runtime_kind"] != KIND_TBE:
        return
    from glc_loader.tbe_cpu_numpy import decode_tbe_numpy
    try:
        decoded = decode_tbe_numpy(payload)
    except Exception as exc:  # noqa: BLE001 - any decode failure is a corrupt entry
        raise CacheVerifyError(f"cached TBE container does not decode: {type(exc).__name__}: {exc}",
                               row["weight_name"]) from exc
    if _bf16_sha256(decoded) != row["source_sha256"]:
        raise CacheVerifyError("decoded cached tensor SHA-256 differs from package source", row["weight_name"])


# ---------------------------------------------------------------------------
# the cache handle
# ---------------------------------------------------------------------------
class TranscodeCache:
    """One cache entry (``<root>/<key>``) for one package/device/flag combination."""

    def __init__(self, root: str | Path, material: Dict[str, Any], *, verify: str = "hash",
                 log: Callable[[str], None] = print, load_workers: Optional[int] = None):
        if verify not in VERIFY_MODES:
            raise TranscodeCacheError(f"transcode_cache_verify must be one of {VERIFY_MODES}")
        self.root = Path(root).expanduser()
        self.material = dict(material)
        self.key = cache_key(self.material)
        self.dir = self.root / self.key
        self.verify = verify
        self.log = log
        self.load_workers = int(load_workers or max(1, min(4, os.cpu_count() or 1)))
        self._probe: Optional[Dict[str, Any]] = None
        self._probed = False
        self.meta: Dict[str, Any] = {
            "enabled": True, "root": str(self.root), "key": self.key, "verify": verify,
            "status": "miss", "reason": None, "bytes_read": 0, "bytes_written": 0,
            "seconds": {},
        }

    @classmethod
    def for_package(cls, root, *, package: Path, runtime_version: str,
                    device_identity: Dict[str, Any], flags: Dict[str, Any], **kwargs) -> "TranscodeCache":
        manifest = Path(package) / "manifest.json"
        material = key_material(manifest_sha256=file_sha256(manifest), runtime_version=runtime_version,
                                device_identity=device_identity, flags=flags)
        return cls(root, material, **kwargs)

    # -- lookup -------------------------------------------------------------
    def probe(self) -> Optional[Dict[str, Any]]:
        """Cheap presence check: complete entry with a self-consistent index (no shard reads)."""
        if self._probed:
            return self._probe
        self._probed = True
        started = time.perf_counter()
        try:
            self._probe = self._read_index()
        except CacheVerifyError as exc:
            self._probe = None
            self.meta["reason"] = f"index invalid: {exc}"
            self.log(f"[transcode-cache] WARNING index for {self.dir} failed verification ({exc}); "
                     "moving it aside and transcoding")
            self.quarantine(str(exc))
        except OSError as exc:
            # e.g. a default ~/.cache owned by another user: the lookup itself is not
            # permitted. Serve uncached (the writer reports its own warning) instead of
            # failing startup.
            self._probe = None
            self.meta["reason"] = f"cache directory not readable: {type(exc).__name__}: {exc}"
            self.log(f"[transcode-cache] WARNING cannot read {self.dir} ({type(exc).__name__}: {exc}); "
                     "transcoding without a cache hit")
        finally:
            self.meta["seconds"]["probe"] = round(time.perf_counter() - started, 3)
        if self._probe is None and self.meta["reason"] is None:
            self.meta["reason"] = "no complete cache entry"
        return self._probe

    def _read_index(self) -> Optional[Dict[str, Any]]:
        if not self.dir.is_dir():
            return None
        complete = self.dir / COMPLETE_NAME
        index_path = self.dir / INDEX_NAME
        if not complete.is_file():
            raise CacheVerifyError("COMPLETE marker missing", str(complete))
        try:
            marker = json.loads(complete.read_text(encoding="utf-8"))
            raw = index_path.read_bytes()
        except (OSError, ValueError) as exc:
            raise CacheVerifyError(f"cannot read cache index: {exc}", str(index_path)) from exc
        if not isinstance(marker, dict) or marker.get("index_sha256") != hashlib.sha256(raw).hexdigest():
            raise CacheVerifyError("index.json SHA-256 differs from COMPLETE marker", str(index_path))
        try:
            index = json.loads(raw)
        except ValueError as exc:
            raise CacheVerifyError(f"index.json is not JSON: {exc}", str(index_path)) from exc
        if (index.get("schema") != INDEX_SCHEMA or index.get("cache_format") != CACHE_FORMAT
                or index.get("key") != self.key or index.get("key_material") != self.material):
            raise CacheVerifyError("index schema/key does not match this package and device", str(index_path))
        if not isinstance(index.get("tensors"), list) or not isinstance(index.get("shards"), list):
            raise CacheVerifyError("index lacks tensors/shards", str(index_path))
        return index

    def quarantine(self, reason: str) -> None:
        """Move a bad entry aside (``<key>.corrupt-<ts>``), log it, then delete it to free space."""
        if not self.dir.exists():
            return
        aside = self.root / f"{self.key}.corrupt-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        try:
            os.rename(self.dir, aside)
        except OSError as exc:
            self.log(f"[transcode-cache] WARNING could not move bad cache entry {self.dir} aside: {exc}")
            return
        self.log(f"[transcode-cache] moved bad cache entry aside: {aside} (reason: {reason})")
        shutil.rmtree(aside, ignore_errors=True)
        if not aside.exists():
            self.log(f"[transcode-cache] deleted bad cache entry {aside}; it will be rewritten")
        self._probe = None

    # -- read path ----------------------------------------------------------
    def check_expected(self, index: Dict[str, Any], expected: Sequence[Dict[str, Any]]) -> None:
        """Every live tensor must have exactly one row with matching kind, shape and source SHA."""
        rows = {row.get("weight_name"): row for row in index["tensors"]}
        if len(rows) != len(index["tensors"]):
            raise CacheVerifyError("duplicate tensor rows in cache index", INDEX_NAME)
        want = {e["weight_name"]: e for e in expected}
        missing = sorted(set(want) - set(rows))
        extra = sorted(set(rows) - set(want))
        if missing or extra:
            raise CacheVerifyError(
                f"cache index does not match the live model (missing {missing[:4]}, extra {extra[:4]})",
                INDEX_NAME)
        for name, exp in want.items():
            row = rows[name]
            if row.get("source_sha256") != exp["source_sha256"]:
                raise CacheVerifyError("source SHA-256 differs from the package manifest", name)
            for field in ("module_name", "runtime_kind", "shape"):
                if row.get(field) != exp[field]:
                    raise CacheVerifyError(f"{field} differs from the live model", name)

    def iter_load(self, index: Dict[str, Any], *, on_bytes: Optional[Callable[[int], None]] = None,
                  ) -> Iterator[Tuple[Dict[str, Any], Any]]:
        """Yield ``(row, payload)`` per tensor, shard by shard, after verifying each shard.

        Shards are read and hashed on a bounded thread pool (at most ``load_workers``
        shards in flight). ``payload`` is a ``TBETensor`` or a dense BF16 tensor.
        """
        shards = index["shards"]
        by_shard: Dict[int, List[Dict[str, Any]]] = {}
        for row in index["tensors"]:
            shard = row.get("shard")
            if type(shard) is not int or not 0 <= shard < len(shards):
                raise CacheVerifyError("tensor row names an unknown shard", str(row.get("weight_name")))
            by_shard.setdefault(shard, []).append(row)
        counter_lock = threading.Lock()

        def count(n: int) -> None:
            with counter_lock:
                self.meta["bytes_read"] += n
            if on_bytes is not None:
                on_bytes(n)

        def load_one(i: int):
            spec = shards[i]
            name = spec.get("file")
            if not isinstance(name, str) or "/" in name or name.startswith("."):
                raise CacheVerifyError("invalid shard file name", str(name))
            tensors = read_verified_shard(self.dir / name, spec.get("sha256"), int(spec.get("bytes", -1)),
                                          on_bytes=count)
            out = []
            for row in by_shard.get(i, []):
                if row["runtime_kind"] == KIND_TBE:
                    payload = _rebuild_container(row, tensors)
                    if self.verify == "full":
                        _full_check(row, payload)
                else:
                    payload = tensors.get(f"{row['weight_name']}::weight")
                    if payload is None:
                        raise CacheVerifyError("shard lacks dense tensor", row["weight_name"])
                    if list(payload.shape) != row["shape"]:
                        raise CacheVerifyError("dense tensor shape differs from index", row["weight_name"])
                    # Dense rows are BF16 source words: re-hash them in both modes.
                    if _bf16_sha256(payload) != row["source_sha256"]:
                        raise CacheVerifyError("dense tensor SHA-256 differs from package source",
                                               row["weight_name"])
                out.append((row, payload))
            return out

        order = list(range(len(shards)))
        with ThreadPoolExecutor(max_workers=self.load_workers, thread_name_prefix="tcache-load") as pool:
            pending: List[Any] = []
            nxt = 0
            try:
                while nxt < len(order) and len(pending) < self.load_workers:
                    pending.append(pool.submit(load_one, order[nxt]))
                    nxt += 1
                while pending:
                    fut = pending.pop(0)
                    result = fut.result()
                    if nxt < len(order):
                        pending.append(pool.submit(load_one, order[nxt]))
                        nxt += 1
                    for item in result:
                        yield item
                    del result
            finally:
                for fut in pending:
                    fut.cancel()

    # -- write path ---------------------------------------------------------
    def writer(self, estimate_bytes: int) -> Optional["CacheWriter"]:
        """A writer for a miss, or None (with a one-line warning) if it cannot be written."""
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self._clean_stale_partials()
            free = shutil.disk_usage(self.root).free
        except OSError as exc:
            self._skip_write(f"cache root {self.root} is not usable ({exc})")
            return None
        need = int(estimate_bytes) + FREE_SPACE_RESERVE_BYTES
        if free < need:
            self._skip_write(f"insufficient free space in {self.root}: {free / 1e9:.1f} GB free, "
                             f"~{need / 1e9:.1f} GB needed")
            return None
        if not os.access(self.root, os.W_OK):
            self._skip_write(f"cache root {self.root} is not writable")
            return None
        partial = self.root / f"{self.key}.partial-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        try:
            partial.mkdir()
        except OSError as exc:
            self._skip_write(f"cannot create {partial} ({exc})")
            return None
        return CacheWriter(self, partial)

    def _skip_write(self, why: str) -> None:
        self.meta["write"] = {"status": "skipped", "reason": why}
        self.log(f"[transcode-cache] WARNING {why}; serving without writing a transcode cache")

    def _clean_stale_partials(self) -> None:
        host = socket.gethostname()
        prefix = f"{self.key}.partial-{host}-"
        for path in self.root.glob(f"{self.key}.partial-*"):
            rest = path.name[len(prefix):] if path.name.startswith(prefix) else None
            if rest is None:
                continue
            try:
                pid = int(rest.split("-", 1)[0])
            except ValueError:
                continue
            if pid != os.getpid() and not _pid_alive(pid):
                shutil.rmtree(path, ignore_errors=True)
                self.log(f"[transcode-cache] removed stale partial write {path} (pid {pid} is gone)")


class CacheWriter:
    """Streams verified tensors into ``<key>.partial-*`` shards; ``commit`` renames into place."""

    def __init__(self, cache: TranscodeCache, partial: Path):
        self.cache = cache
        self.partial = partial
        self.rows: List[Dict[str, Any]] = []
        self.shards: List[Dict[str, Any]] = []
        self._current: List[Tuple[str, Any]] = []
        self._current_rows: List[Dict[str, Any]] = []
        self._current_bytes = 0
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tcache-write")
        self._pending = None
        self.active = True
        self.error: Optional[str] = None
        self.started = time.perf_counter()
        self.on_bytes: Optional[Callable[[int], None]] = None

    def add_tbe(self, module_name: str, weight_name: str, container, *, source_sha256: str,
                source_bf16_bytes: int) -> None:
        if not self.active:
            return
        try:
            row = {
                "module_name": module_name, "weight_name": weight_name, "runtime_kind": KIND_TBE,
                "shape": [int(x) for x in container.shape], "source_sha256": source_sha256,
                "source_bf16_bytes": int(source_bf16_bytes),
                "tbe_v1_stored_bytes": int(container.byte_size()["total"]),
                "container": _container_fields(container),
            }
            planes = _u32_words_int32(container.planes)
            sbbase = _u32_words_int32(container.sbbase)
            self._add(row, [(f"{weight_name}::planes", planes), (f"{weight_name}::sbbase", sbbase),
                            (f"{weight_name}::smb", container.smb.detach().contiguous().cpu()),
                            (f"{weight_name}::esc", container.esc.detach().contiguous().cpu())])
        except Exception as exc:  # noqa: BLE001 - the cache must never break startup
            self.abort(f"cannot stage {weight_name}: {type(exc).__name__}: {exc}")

    def add_dense(self, module_name: str, weight_name: str, runtime_kind: str, tensor, *,
                  source_sha256: str) -> None:
        if not self.active:
            return
        try:
            cpu = tensor.detach().contiguous().cpu()
            row = {"module_name": module_name, "weight_name": weight_name, "runtime_kind": runtime_kind,
                   "shape": [int(x) for x in cpu.shape], "source_sha256": source_sha256,
                   "source_bf16_bytes": int(cpu.numel() * cpu.element_size())}
            self._add(row, [(f"{weight_name}::weight", cpu)])
        except Exception as exc:  # noqa: BLE001
            self.abort(f"cannot stage {weight_name}: {type(exc).__name__}: {exc}")

    def _add(self, row: Dict[str, Any], tensors: List[Tuple[str, Any]]) -> None:
        nbytes = sum(int(t.numel() * t.element_size()) for _n, t in tensors)
        if self._current and self._current_bytes + nbytes > SHARD_TARGET_BYTES:
            self._flush()
        if not self.active:
            return
        self._current.extend(tensors)
        self._current_rows.append(row)
        self._current_bytes += nbytes

    def _flush(self) -> None:
        if not self._current:
            return
        self._wait()
        if not self.active:
            return
        index = len(self.shards)
        name = f"shard-{index:05d}.safetensors"
        try:
            free = shutil.disk_usage(self.partial).free
        except OSError as exc:
            self.abort(f"cannot stat cache disk: {exc}")
            return
        if free < self._current_bytes + FREE_SPACE_RESERVE_BYTES:
            self.abort(f"free space fell to {free / 1e9:.1f} GB while writing")
            return
        tensors, rows = self._current, self._current_rows
        for row in rows:
            row["shard"] = index
        self.rows.extend(rows)
        self.shards.append({"file": name, "bytes": None, "sha256": None})
        self._current, self._current_rows, self._current_bytes = [], [], 0
        self._pending = self._pool.submit(self._write_shard, index, name, tensors)

    def _write_shard(self, index: int, name: str, tensors) -> None:
        nbytes, digest = write_safetensors(self.partial / name, tensors,
                                           metadata={"format": INDEX_SCHEMA, "shard": str(index)})
        self.shards[index].update({"bytes": nbytes, "sha256": digest})
        self.cache.meta["bytes_written"] += nbytes
        if self.on_bytes is not None:
            self.on_bytes(nbytes)

    def _wait(self) -> None:
        fut, self._pending = self._pending, None
        if fut is None:
            return
        try:
            fut.result()
        except Exception as exc:  # noqa: BLE001
            self.abort(f"shard write failed: {type(exc).__name__}: {exc}")

    def abort(self, why: str) -> None:
        if not self.active:
            return
        self.active = False
        self.error = why
        self._current, self._current_rows = [], []
        fut, self._pending = self._pending, None
        if fut is not None:
            try:
                fut.result()
            except Exception:  # noqa: BLE001 - already aborting; the partial dir is removed below
                pass
        self._pool.shutdown(wait=True)
        shutil.rmtree(self.partial, ignore_errors=True)
        self.cache.meta["write"] = {"status": "aborted", "reason": why}
        self.cache.log(f"[transcode-cache] WARNING cache write aborted ({why}); serving continues without it")

    def commit(self) -> bool:
        """Flush, fsync and atomically publish the entry. Returns True when published."""
        if not self.active:
            return False
        try:
            self._flush()
            self._wait()
            if not self.active:
                return False
            self._pool.shutdown(wait=True)
            index = {
                "schema": INDEX_SCHEMA, "cache_format": CACHE_FORMAT, "key": self.cache.key,
                "key_material": self.cache.material,
                "created_unix": int(time.time()),
                "shards": self.shards, "tensors": self.rows,
                "totals": {
                    "tensor_count": len(self.rows),
                    "shard_bytes": sum(int(s["bytes"]) for s in self.shards),
                    "source_bf16_bytes": sum(int(r.get("source_bf16_bytes", 0)) for r in self.rows),
                },
            }
            raw = json.dumps(index, sort_keys=True, indent=1).encode("utf-8")
            _write_file_durable(self.partial / INDEX_NAME, raw)
            _write_file_durable(self.partial / COMPLETE_NAME,
                                _canonical({"index_sha256": hashlib.sha256(raw).hexdigest()}))
            _fsync_dir(self.partial)
            try:
                os.rename(self.partial, self.cache.dir)
            except OSError as exc:
                if self.cache.dir.exists():
                    shutil.rmtree(self.partial, ignore_errors=True)
                    self.active = False
                    self.cache.meta["write"] = {"status": "lost_race",
                                                "reason": "another process published this entry first"}
                    self.cache.log(f"[transcode-cache] another process published {self.cache.dir} first; "
                                   "keeping theirs")
                    return False
                raise exc
            _fsync_dir(self.cache.root)
            self.active = False
            self.cache.meta["write"] = {"status": "written", "shards": len(self.shards),
                                        "seconds_total": round(time.perf_counter() - self.started, 3)}
            return True
        except Exception as exc:  # noqa: BLE001
            self.abort(f"commit failed: {type(exc).__name__}: {exc}")
            return False


__all__ = ["VERIFY_MODES", "PACKAGE_VERIFY_MODES", "ENV_CACHE_DIR", "CACHE_FORMAT",
           "CACHE_AUTO", "CACHE_OFF", "default_cache_root", "add_cli_args", "resolve_setting",
           "kwargs_from_args", "explicit_cache_dir", "describe", "package_verify_kwargs",
           "TranscodeCache", "CacheWriter", "TranscodeCacheError", "CacheVerifyError",
           "cache_key", "key_material", "expected_entry", "read_verified_shard",
           "write_safetensors", "file_sha256",
           "KIND_TBE", "KIND_EMBEDDING", "KIND_VISION_LINEAR", "KIND_VISION_EMBEDDING"]
