"""Parallel, disk-cached Context -> TBE21 serving bridge (release task 1).

``bctx_v21_transcode.ContextV21TranscodeArtifact`` is a correct but synchronous,
single-thread, in-memory-only bridge: every tensor is decoded from its Context frame and
re-encoded to v2.1 the first time the model loader asks for it, nothing is written to disk,
and nothing is cached across a server restart.  Measured single-thread rate (native-entropy
census, docs/handoff/native-entropy_HANDOFF_20261018.md): ~16 MB/s frame decode plus
~5.3 s/178 MB v2.1 encode -- a ~36 GB TBE21 resident set at that rate is tens of minutes,
which is fine for a one-off research run but fails the release's startup bar (<= ~4-6 min
cold, <= 60 s warm).

This module adds exactly two things on top of the existing bridge, without touching it or
its default-off CLI surface (``--context-build-root`` keeps working byte for byte):

1. **Parallel transcode.**  Every tensor's (decode, encode, round-trip-verify) is independent
   of every other tensor's, so a bounded pool of worker PROCESSES (not threads: the decode
   step calls into a JIT-built C++ extension and the encode step is numpy/C -- neither
   reliably releases the GIL across its whole body, so processes are the only parallelism
   unit this module can trust) does the whole package's worth of work in parallel, bounded by
   CPU count and a conservative estimate of available RAM.

2. **A verified, versioned disk cache.**  The parallel build's output (the exact same
   ``glc_loader.codec_v21.V21Tensor`` objects the synchronous bridge produces, byte for byte,
   since both paths call the same ``encode_tensor`` / ``decode_tensor`` functions and the same
   source-identity checks) is persisted under a cache directory keyed on the Context
   package's manifest sha256, the runtime format, the GPU architecture, and any extra
   serving flags that could change what gets cached.  A warm restart with a matching, complete
   cache skips the pool entirely; each cached tensor is still decoded and checked against
   the source hash and geometry pinned by the Context probe before it is trusted.

Fails closed on every identity mismatch (wrong cache key, truncated/tampered cache file,
unknown runtime format) exactly the way the synchronous bridge does: raise
``ContextTranscodeError``, never silently fall back to re-deriving or skipping a check.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from glc_loader import codec_v21 as cv21

from .bctx_v21_transcode import (
    ContextTranscodeError,
    ContextV21TranscodeArtifact,
)

#: The only runtime formats this cache knows how to build/serve.  A caller asking for
#: anything else is refused here, before any device or model state exists -- "fail closed on
#: unknown formats" from the task spec, not merely from whatever error the loader eventually
#: raises three layers down.
SUPPORTED_RUNTIME_FORMATS = ("tbe21",)

#: Schema tag for the cache key / cache_manifest.json -- bumped if the on-disk layout or the
#: set of identity inputs to the key ever changes, so an old cache is never matched by name.
CACHE_SCHEMA = "context-v21-parallel-cache-v2"

CACHE_MANIFEST_NAME = "cache_manifest.json"
PROGRESS_NAME = ".progress.json"

DEFAULT_MIN_WORKER_MEMORY_BYTES = 64 << 20  # a stated, not-tuned, floor per worker


def _recipe_sources():
    """Files whose bytes can affect cached TBE21 streams or their interpretation.

    Keep this explicit and fail closed: a renamed/removed helper must not silently
    leave an old cache reusable.
    """
    here = Path(__file__).resolve().parent
    return {
        "cache": Path(__file__),
        "transcoder": here / "bctx_v21_transcode.py",
        "codec": Path(cv21.__file__),
        "descriptor_helper": here / "tbe2_desc.py",
        "reference_helper": here / "bigemm_tbe21_ref.py",
    }


def _source_recipe():
    recipe = {}
    for role, source in sorted(_recipe_sources().items()):
        try:
            raw = source.read_bytes()
        except OSError as exc:
            raise ContextTranscodeError(
                f"cannot identify cache recipe source {role} at {source}") from exc
        recipe[role] = {"path": source.name, "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest()}
    return recipe


def _canonical_sha256(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _normalized_flags(extra_flags):
    return {str(k): extra_flags[k] for k in sorted((extra_flags or {}).keys(), key=str)}


def _durable_dir(path, *, label: str) -> Path:
    p = Path(path).expanduser().resolve()
    if ".scratch" not in p.parts or p.is_relative_to(Path("/tmp")) or p.is_relative_to(Path("/private/tmp")):
        raise ContextTranscodeError(f"{label} must be a durable .scratch directory, not {p}")
    return p


def cache_key(manifest_sha256: str, runtime_format: str, gpu_arch: str,
             extra_flags: Optional[Dict[str, Any]] = None) -> str:
    """The cache directory name: a pure function of everything that could change the bytes
    a build produces.  Any one of these changing must produce a DIFFERENT key, never a
    collision silently served from a stale build."""
    recipe = _source_recipe()
    return _cache_key(manifest_sha256, runtime_format, gpu_arch, extra_flags, recipe)


def _cache_key(manifest_sha256, runtime_format, gpu_arch, extra_flags, recipe):
    payload = json.dumps({
        "schema": CACHE_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "runtime_format": runtime_format,
        "codec_format": cv21.FORMAT,
        "codec_format_version": cv21.FORMAT_VERSION,
        "gpu_arch": str(gpu_arch),
        "extra_flags": _normalized_flags(extra_flags),
        "source_recipe": recipe,
        "source_recipe_sha256": _canonical_sha256(recipe),
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def detect_gpu_arch() -> str:
    """``"cpu"`` off a CUDA-less box, else ``"sm_<major><minor>"`` -- never guessed, never
    defaulted to something that could silently collide with a real arch's cache entries."""
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return f"sm_{major}{minor}"
    except Exception:  # noqa: BLE001 -- any import/driver problem just means "no GPU to ask"
        pass
    return "cpu"


def _available_memory_bytes() -> int:
    """Best-effort free RAM, in bytes.  This bounds parallelism, not correctness: an
    under-estimate only means fewer workers (slower, still correct); it never produces a
    wrong transcode.  Falls back to a stated conservative constant rather than raising, since
    a worker-count knob is not worth failing startup over."""
    try:
        import psutil  # optional dependency; used if present

        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001
        pass
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        avail = os.sysconf("SC_AVPHYS_PAGES")
        return int(page) * int(avail)
    except (ValueError, OSError, AttributeError):
        pass
    try:
        import subprocess

        total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=2))
        return total // 2  # unknown availability on this path; assume half is free
    except Exception:  # noqa: BLE001
        pass
    return 8 << 30  # stated fallback: 8 GiB, not tuned


def default_max_workers(total_source_bytes: int, n_tensors: int, *,
                        requested: int = 0) -> int:
    """``min(CPU count, a RAM-derived bound, tensor count)``, all stated and all overridable
    by an explicit ``requested`` count (0 = auto)."""
    cpu = os.cpu_count() or 1
    if requested and requested > 0:
        cpu = min(cpu, int(requested))
    if n_tensors <= 0:
        return 1
    avg_bytes = max(1, total_source_bytes // max(1, n_tensors))
    # Each worker holds the source words array, the v2.1 encode intermediate arrays, and the
    # decode round-trip array at once; 4x the average tensor size is a stated safety margin,
    # not a measurement, biased toward under-parallelizing rather than OOM-ing a worker.
    per_worker = max(avg_bytes * 4, DEFAULT_MIN_WORKER_MEMORY_BYTES)
    mem_bound = max(1, _available_memory_bytes() // per_worker)
    return max(1, min(cpu, mem_bound, n_tensors))


def _write_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True))
    os.replace(tmp, path)


def read_progress(cache_dir) -> Optional[dict]:
    """The ``/health`` phase-progress block for a build in flight (or just finished).  Never
    raises: a missing or unreadable progress file just means "nothing to report"."""
    p = Path(cache_dir) / PROGRESS_NAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _safe_tensor_filename(name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
    return f"{digest}.v21st"


def _write_cache_tensor(cache_dir: Path, name: str, encoded) -> dict:
    header, blob = cv21.serialize(encoded)
    fname = _safe_tensor_filename(name)
    (cache_dir / fname).write_bytes(blob)
    return {
        "name": name, "blob_file": fname, "blob_bytes": len(blob),
        "blob_sha256": hashlib.sha256(blob).hexdigest(),
        "header": header, "source_sha256": encoded.sha256,
    }


def _read_cache_tensor(cache_dir: Path, entry: dict, *, expected_record: dict):
    if not isinstance(entry, dict) or not isinstance(expected_record, dict):
        raise ContextTranscodeError("invalid cache entry or trusted Context probe record")
    expected_name = expected_record.get("name")
    shape = expected_record.get("shape")
    dtype = expected_record.get("dtype")
    source_bytes = expected_record.get("source_bytes")
    expected_source_sha256 = expected_record.get("source_sha256")
    if (not isinstance(expected_name, str) or not expected_name
            or entry.get("name") != expected_name
            or dtype != "BF16" or not isinstance(shape, list) or not shape
            or any(type(dim) is not int or dim <= 0 for dim in shape)
            or type(source_bytes) is not int or source_bytes != 2 * math.prod(shape)
            or not isinstance(expected_source_sha256, str)):
        raise ContextTranscodeError("invalid trusted Context probe record or cache tensor name")

    expected_blob_file = _safe_tensor_filename(expected_name)
    if entry.get("blob_file") != expected_blob_file:
        raise ContextTranscodeError(f"{expected_name}: unsafe or mismatched cache blob path")
    root = Path(cache_dir)
    try:
        root_mode = root.lstat().st_mode
        path = root / expected_blob_file
        path_mode = path.lstat().st_mode
    except OSError as exc:
        raise ContextTranscodeError(f"{expected_name}: cache blob missing or unreadable") from exc
    if (stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode)
            or stat.S_ISLNK(path_mode) or not stat.S_ISREG(path_mode)
            or path.resolve().parent != root.resolve()):
        raise ContextTranscodeError(f"{expected_name}: cache blob is not a regular file in cache directory")
    try:
        blob = path.read_bytes()
    except OSError as exc:
        raise ContextTranscodeError(f"{expected_name}: cache blob missing or unreadable") from exc
    if len(blob) != entry["blob_bytes"] or hashlib.sha256(blob).hexdigest() != entry["blob_sha256"]:
        raise ContextTranscodeError(f"{expected_name}: cache blob identity mismatch "
                                    "(tampered, truncated, or a stale write) -- refusing to serve it")
    tensor = cv21.deserialize(entry["header"], blob)
    if (tensor.name != expected_name or tensor.dtype != dtype
            or tuple(tensor.geom.shape) != tuple(shape)):
        raise ContextTranscodeError(f"{expected_name}: cached tensor name, dtype, or shape mismatch")
    # Cache metadata and blob hashes are stored beside the blob and can be changed
    # together. Decode the actual uint16 source words and compare them to the hash
    # in the already-verified Context probe record before trusting this cache entry.
    if tensor.sha256 != entry["source_sha256"] or tensor.sha256 != expected_source_sha256:
        raise ContextTranscodeError(f"{expected_name}: cache source sha256 disagrees with Context probe")
    import numpy as np

    decoded = cv21.decode_tensor(tensor, check_crc=True, check_sha=True)
    if (decoded.dtype != np.uint16 or tuple(decoded.shape) != tuple(shape)
            or decoded.nbytes != source_bytes):
        raise ContextTranscodeError(f"{expected_name}: cached uint16 shape or byte count mismatch")
    raw_sha256 = hashlib.sha256(memoryview(np.ascontiguousarray(decoded)).cast("B")).hexdigest()
    if raw_sha256 != expected_source_sha256:
        raise ContextTranscodeError(f"{expected_name}: cached decoded uint16 source sha256 mismatch")
    return tensor


def _transcode_worker(release_dir: str, package_path: str, build_root: str,
                      manifest_sha256: str, cache_dir: str, name: str) -> dict:
    """Runs in a WORKER PROCESS: decode this one Context frame, encode it to v2.1, verify
    the round trip (identical checks the synchronous bridge performs), and write the cached
    blob to disk.  Explicit sys.path setup rather than relying on fork-inherited state, so
    this is correct under either the ``fork`` or ``spawn`` multiprocessing start method."""
    import sys

    if release_dir not in sys.path:
        sys.path.insert(0, release_dir)
    from glc_serve.bctx_v21_transcode import ContextV21TranscodeArtifact as _Art

    art = _Art(package_path, build_root=build_root, expected_manifest_sha256=manifest_sha256)
    encoded = art.read(name)
    return _write_cache_tensor(Path(cache_dir), name, encoded)


def _load_cache_manifest(cache_dir: Path, *, key: str, manifest_sha256: str,
                         runtime_format: str, gpu_arch: str, extra_flags,
                         source_recipe: dict, source_recipe_sha256: str,
                         names) -> Optional[dict]:
    p = cache_dir / CACHE_MANIFEST_NAME
    if not p.exists():
        return None
    try:
        cm = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if (cm.get("schema") != CACHE_SCHEMA or cm.get("cache_key") != key
            or cm.get("manifest_sha256") != manifest_sha256
            or cm.get("runtime_format") != runtime_format
            or cm.get("gpu_arch") != gpu_arch
            or cm.get("codec_format") != cv21.FORMAT
            or cm.get("codec_format_version") != cv21.FORMAT_VERSION
            or cm.get("extra_flags", {}) != _normalized_flags(extra_flags)
            or cm.get("source_recipe") != source_recipe
            or cm.get("source_recipe_sha256") != source_recipe_sha256
            or cm.get("complete") is not True
            or set(cm.get("tensors", {})) != set(names)):
        return None
    return cm


def build_cache(package_path, *, build_root, cache_root, gpu_arch: Optional[str] = None,
                runtime_format: str = "tbe21", extra_flags: Optional[Dict[str, Any]] = None,
                max_workers: int = 0,
                expected_manifest_sha256: Optional[str] = None,
                progress_cb: Optional[Callable[[dict], None]] = None) -> Dict[str, Any]:
    """Resolve (reuse or build) the parallel transcode cache for ``package_path``.

    ``expected_manifest_sha256=None`` (the default) accepts either of the two FROZEN,
    published manifests ``ContextV21TranscodeArtifact`` itself pins (the baseline schema and
    the predictive schema, ``bctx_v21_transcode.FROZEN_CONTEXT_MANIFEST_SHA256`` /
    ``FROZEN_PREDICTIVE_MANIFEST_SHA256``) -- anything else needs an explicit pin and
    otherwise fails closed, exactly as the underlying bridge does.

    Returns ``{"cache_dir": Path, "cache_manifest": dict, "warm": bool, "manifest_sha256":
    str, "cache_key": str}``.  Fails closed (``ContextTranscodeError``) on an unsupported
    ``runtime_format``, a Context manifest sha256 mismatch, or any tensor whose cached (or
    freshly transcoded) bytes do not verify -- it never returns a partially-valid cache.
    """
    if runtime_format not in SUPPORTED_RUNTIME_FORMATS:
        raise ContextTranscodeError(
            f"unknown runtime_format {runtime_format!r}; supported: {SUPPORTED_RUNTIME_FORMATS}")
    release_dir = str(Path(__file__).resolve().parent.parent)
    build_root = _durable_dir(build_root, label="--context-build-root")
    cache_root = _durable_dir(cache_root, label="--context-cache-root")
    package_path = str(Path(package_path).expanduser().resolve())
    gpu_arch = gpu_arch or detect_gpu_arch()

    # One probe construction in the parent: validates the manifest (sha256-pinned, schema,
    # completeness -- ContextV21TranscodeArtifact's own constructor checks) and gets the
    # decoder JIT-built exactly once before any worker races to do the same thing. The
    # constructor's own write-or-compare logic is already safe under a race, but doing it
    # here means a build failure is attributable to "cache build", not to which of N workers
    # happened to lose the JIT race.
    probe = ContextV21TranscodeArtifact(package_path, build_root=str(build_root),
                                        expected_manifest_sha256=expected_manifest_sha256)
    manifest_sha256 = probe.manifest_sha256
    names = probe.names()
    total_source_bytes = sum(int(probe.records[n]["source_bytes"]) for n in names)

    recipe = _source_recipe()
    recipe_sha256 = _canonical_sha256(recipe)
    key = _cache_key(manifest_sha256, runtime_format, gpu_arch, extra_flags, recipe)
    cache_dir = cache_root / key
    cache_dir.mkdir(parents=True, exist_ok=True)

    existing = _load_cache_manifest(
        cache_dir, key=key, manifest_sha256=manifest_sha256,
        runtime_format=runtime_format, gpu_arch=gpu_arch, extra_flags=extra_flags,
        source_recipe=recipe, source_recipe_sha256=recipe_sha256, names=names)
    if existing is not None:
        progress = {"phase": "done", "source": "warm", "tensors_done": len(names),
                   "tensors_total": len(names), "bytes_done": total_source_bytes,
                   "bytes_total": total_source_bytes, "workers": 0,
                   "updated_at": time.time()}
        _write_json_atomic(cache_dir / PROGRESS_NAME, progress)
        if progress_cb:
            progress_cb(dict(progress))
        return {"cache_dir": cache_dir, "cache_manifest": existing, "warm": True,
                "manifest_sha256": manifest_sha256, "cache_key": key}

    # Going cold: the decoder module the probe construction above loaded is part of the
    # frozen, sha256-pinned Context package (this repo cannot change it), and its own
    # native build step -- invoked lazily, the first time anything actually decodes a
    # frame -- is guarded ONLY by an in-process threading.Lock(). With N worker PROCESSES
    # about to race to decode their first tensor at roughly the same moment, that guard
    # gives no cross-process protection at all: two processes can race to compile the SAME
    # shared .dylib path and one dlopen()s a file the other has not finished writing
    # (observed directly: a CPU-only parallel-pool run hit exactly this as an intermittent
    # "no such file" dlopen error). Decoding one tensor HERE, in the parent, before any
    # worker starts, forces that build to happen exactly once, synchronously, before the
    # race window opens; every worker then finds the shared library already on disk with a
    # fresh mtime and never attempts to build it itself. Skipped entirely on the warm path
    # above, which needs no decode at all.
    if names:
        probe.read_source_words(names[0])

    workers = default_max_workers(total_source_bytes, len(names), requested=max_workers)
    progress = {"phase": "transcoding", "source": "cold", "tensors_done": 0,
               "tensors_total": len(names), "bytes_done": 0, "bytes_total": total_source_bytes,
               "workers": workers, "started_at": time.time(), "updated_at": time.time()}
    _write_json_atomic(cache_dir / PROGRESS_NAME, progress)
    if progress_cb:
        progress_cb(dict(progress))

    entries: Dict[str, dict] = {}
    errors: Dict[str, str] = {}
    byte_by_name = {n: int(probe.records[n]["source_bytes"]) for n in names}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_transcode_worker, release_dir, package_path, str(build_root),
                       manifest_sha256, str(cache_dir), name): name
            for name in names
        }
        pending = set(futures)
        while pending:
            done, pending = wait(pending, return_when=FIRST_EXCEPTION)
            progress_changed = False
            for fut in done:
                name = futures[fut]
                try:
                    entries[name] = fut.result()
                except Exception as exc:  # noqa: BLE001 -- recorded, then fails closed below
                    errors[name] = f"{type(exc).__name__}: {exc}"
                progress["tensors_done"] = len(entries) + len(errors)
                progress["bytes_done"] += byte_by_name[name]
                progress_changed = True
            if progress_changed:
                progress["updated_at"] = time.time()
                _write_json_atomic(cache_dir / PROGRESS_NAME, progress)
                if progress_cb:
                    progress_cb(dict(progress))
            if errors:
                for other in pending:
                    other.cancel()
                break

    if errors:
        progress.update(phase="failed", updated_at=time.time(), errors=errors)
        _write_json_atomic(cache_dir / PROGRESS_NAME, progress)
        raise ContextTranscodeError(
            f"parallel Context->{runtime_format} transcode failed for "
            f"{len(errors)}/{len(names)} tensor(s): {errors}")

    cache_manifest = {
        "schema": CACHE_SCHEMA, "cache_key": key, "manifest_sha256": manifest_sha256,
        "runtime_format": runtime_format, "gpu_arch": gpu_arch,
        "extra_flags": _normalized_flags(extra_flags),
        "codec_format": cv21.FORMAT, "codec_format_version": cv21.FORMAT_VERSION,
        "source_recipe": recipe, "source_recipe_sha256": recipe_sha256,
        "tensors": entries, "complete": True, "built_at": time.time(),
    }
    _write_json_atomic(cache_dir / CACHE_MANIFEST_NAME, cache_manifest)
    progress.update(phase="done", updated_at=time.time())
    _write_json_atomic(cache_dir / PROGRESS_NAME, progress)
    if progress_cb:
        progress_cb(dict(progress))
    return {"cache_dir": cache_dir, "cache_manifest": cache_manifest, "warm": False,
            "manifest_sha256": manifest_sha256, "cache_key": key}


class CachedContextV21Artifact:
    """The V21 artifact-reader interface (``names``/``stored_record``/``is_alias``/
    ``has_config``/``release``/``read``/``decode_host``, the exact surface
    ``glc_serve.tbe2_load.load_v2_model`` calls), backed by a parallel, verified, disk-cached
    Context -> v2.1 transcode instead of the synchronous per-tensor bridge.

    Construction resolves (builds if cold, reuses if warm) the cache before returning, so by
    the time the model loader calls ``read()`` for the first tensor every tensor's bytes are
    already decoded, encoded, verified and sitting on disk -- the sequential model-load loop
    that follows validates each cached tensor's decoded uint16 payload and geometry against
    the source hash and record pinned by the Context probe.
    """

    def __init__(self, package_path, *, build_root, cache_root, gpu_arch: Optional[str] = None,
                runtime_format: str = "tbe21", extra_flags: Optional[Dict[str, Any]] = None,
                max_workers: int = 0,
                expected_manifest_sha256: Optional[str] = None,
                progress_cb: Optional[Callable[[dict], None]] = None):
        if runtime_format not in SUPPORTED_RUNTIME_FORMATS:
            raise ContextTranscodeError(
                f"unknown runtime_format {runtime_format!r}; supported: {SUPPORTED_RUNTIME_FORMATS}")
        self.runtime_format = runtime_format
        self.format = cv21.FORMAT
        # The probe build does a second, cheap manifest-only construction to recover
        # `.records`/`.path`/`.manifest` without re-reading any tensor data -- build_cache's
        # own probe already paid that cost; this one is a second, independent read-and-verify
        # of the manifest bytes (cheap: KB, not GB) rather than threading the first probe
        # across a public function boundary, which keeps build_cache usable standalone.
        self._source = ContextV21TranscodeArtifact(package_path, build_root=str(build_root),
                                                    expected_manifest_sha256=expected_manifest_sha256)
        self.path = self._source.path
        self.manifest = self._source.manifest
        self.manifest_sha256 = self._source.manifest_sha256
        self.records = self._source.records
        self._names = self._source.names()
        self.config_path = None

        result = build_cache(package_path, build_root=build_root, cache_root=cache_root,
                             gpu_arch=gpu_arch, runtime_format=runtime_format,
                             extra_flags=extra_flags, max_workers=max_workers,
                             expected_manifest_sha256=expected_manifest_sha256,
                             progress_cb=progress_cb)
        self.cache_dir = result["cache_dir"]
        self.cache_manifest = result["cache_manifest"]
        self.warm = result["warm"]
        self.cache_key = result["cache_key"]
        self._entries = result["cache_manifest"]["tensors"]
        self._cached_name = None
        self._cached_tensor = None
        self.transcode_receipts = []

    def names(self):
        return list(self._names)

    def stored_record(self, name):
        return self.records[name]

    def is_alias(self, name):
        self.stored_record(name)
        return False

    def has_config(self):
        return False

    def release(self, name=None):
        if name is None or self._cached_name == name:
            self._cached_name = None
            self._cached_tensor = None

    def read(self, name, *, check_crc=True):
        if self._cached_name == name and self._cached_tensor is not None:
            return self._cached_tensor
        entry = self._entries.get(name)
        if entry is None:
            raise ContextTranscodeError(f"{name}: not present in the transcode cache manifest")
        started = time.monotonic()
        row = self.stored_record(name)
        tensor = _read_cache_tensor(self.cache_dir, entry, expected_record=row)
        self.transcode_receipts.append({
            "name": name, "source_sha256": row["source_sha256"],
            "v21_stream_bytes": tensor.total_bytes(), "cache_read_seconds": time.monotonic() - started,
            "warm": self.warm,
        })
        self._cached_name, self._cached_tensor = name, tensor
        return tensor

    def decode_host(self, name):
        row = self.stored_record(name)
        tensor = self.read(name)
        decoded = cv21.decode_tensor(tensor, check_crc=True, check_sha=True)
        import numpy as np

        if hashlib.sha256(memoryview(np.ascontiguousarray(decoded)).cast("B")).hexdigest() != row["source_sha256"]:
            raise ContextTranscodeError(f"{name}: cached decode source sha256 mismatch")
        return decoded


def cached_context_artifact_factory(*, build_root: str, cache_root: str,
                                    gpu_arch: Optional[str] = None,
                                    runtime_format: str = "tbe21",
                                    extra_flags: Optional[Dict[str, Any]] = None,
                                    max_workers: int = 0,
                                    progress_cb: Optional[Callable[[dict], None]] = None
                                    ) -> Callable[[str], CachedContextV21Artifact]:
    """Factory matching ``glc_serve.context_source.context_artifact_factory``'s calling
    convention (``factory(package_path) -> artifact reader``), for the parallel+cached path."""

    def build(package_path: str) -> CachedContextV21Artifact:
        return CachedContextV21Artifact(
            package_path, build_root=build_root, cache_root=cache_root, gpu_arch=gpu_arch,
            runtime_format=runtime_format, extra_flags=extra_flags, max_workers=max_workers,
            progress_cb=progress_cb)

    return build


__all__ = [
    "SUPPORTED_RUNTIME_FORMATS", "CachedContextV21Artifact", "build_cache", "cache_key",
    "cached_context_artifact_factory", "default_max_workers", "detect_gpu_arch",
    "read_progress",
]
