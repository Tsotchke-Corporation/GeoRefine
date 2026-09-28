"""``georefine.tbe.serve.v1`` -- a TBE container laid out for streaming.

WHY A SECOND LAYOUT.  The certified 27B container (``georefine.tbe.v1``/``v2``)
is two blobs: a 35.2 GB ``tbe_tensors.safetensors`` and a 5.4 GB
``raw_tensors.safetensors``.  One 35 GB object cannot be hash-verified before
use without reading all of it first, cannot be fetched in parallel with
loading, and a truncated transfer poisons the whole artifact.  A serving
bundle holds the SAME bytes (every coded tensor keeps its stored
``planes``/``smb``/``esc``/``sbbase`` arrays byte for byte) re-packed into
size-bounded shards grouped by layer, with a manifest that records:

  * ``sha256`` and byte length of every shard and every sidecar file;
  * per tensor: shard, kind (``tbe``/``raw``), shape, container fields, the
    ``blake2b`` of the SOURCE bf16 bytes (``blake2b_source``, carried from
    the transcode manifest) -- the reference the bit-exact gate checks the
    GPU-decoded weights against;
  * per component (text / vision / mtp / embed / lm_head) resident and dense
    byte accounting, so VRAM projections are read off measured sizes.

A shard is NEVER used before its sha256 has been recomputed and matched.
Remote shards are hashed while they stream to the local cache, so the check
costs no second read; a mismatching download is deleted, never renamed into
place.  Host RAM is bounded: fetch uses fixed-size chunks and tensor reads
use ``safe_open`` (mmap), so the high-water mark is one tensor, not a shard.

Nothing here imports torch at module scope except where tensors are read.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

BUNDLE_FORMAT = "georefine.tbe.serve.v1"
MANIFEST_SCHEMA = "georefine.tbe_serve_manifest.v1"
MANIFEST_FILENAME = "serve_manifest.json"
SHARD_DIR = "shards"
DEFAULT_TARGET_SHARD_BYTES = 2 << 30  # 2 GiB
FETCH_CHUNK_BYTES = 8 << 20
TBE_FIELDS = ("planes", "smb", "esc", "sbbase")

#: Sidecars a bundle copies from the source checkpoint so it is self-contained:
#: the architecture, the tokenizer and chat template, and the multimodal
#: processor configs (the owner's product serves images; a bundle without its
#: processor configs cannot).
SIDECAR_CANDIDATES: Tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
    "LICENSE",
    "LICENSE.txt",
    "NOTICE",
)

COMPONENTS = ("embed", "vision", "text", "lm_head", "mtp")

_TEXT_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


class BundleError(RuntimeError):
    """A typed refusal: ``reason`` is stable, ``detail`` says what to do."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = str(reason)
        self.detail = str(detail)
        super().__init__(f"{reason}: {detail}" if detail else str(reason))


# ---------------------------------------------------------------------------
# naming
# ---------------------------------------------------------------------------
def component_of(name: str) -> str:
    """Which part of the model a checkpoint tensor belongs to."""
    n = str(name)
    if n.startswith("mtp."):
        return "mtp"
    if ".visual." in f".{n}" or n.startswith("visual."):
        return "vision"
    if "embed_tokens" in n or n.endswith("wte.weight"):
        return "embed"
    if n.startswith("lm_head") or ".lm_head." in n:
        return "lm_head"
    return "text"


def layer_of(name: str) -> Optional[int]:
    """Decoder-layer index of a text tensor, else ``None``."""
    if component_of(name) != "text":
        return None
    m = _TEXT_LAYER_RE.search(str(name))
    return int(m.group(1)) if m else None


def group_of(name: str) -> str:
    """The streaming group a tensor travels in (one group never splits)."""
    comp = component_of(name)
    if comp == "text":
        layer = layer_of(name)
        return f"layer:{layer:05d}" if layer is not None else "text:other"
    if comp == "vision":
        return "vision"  # the whole tower is ~0.73 GB resident: one group
    return comp


#: Streaming order: vision first (small, needed by any image request), then
#: the text trunk in layer order, then head, MTP and finally the embedding
#: (which may go to host RAM and so never competes for the GPU budget).
def group_sort_key(group: str) -> Tuple[int, str]:
    order = {"vision": 0, "text:other": 1, "lm_head": 3, "mtp": 4, "embed": 5}
    if group.startswith("layer:"):
        return (2, group)
    return (order.get(group, 6), group)


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------
def sha256_file(path: os.PathLike | str, chunk: int = FETCH_CHUNK_BYTES) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def blake2b_tensor_hex(t) -> str:
    """Byte-identical to ``glc_loader.tbe_artifact._blake2b_tensor``.

    bf16 is viewed as int16 so the digest is over the raw 16-bit patterns;
    every other dtype is hashed as its own bytes.
    """
    import torch

    x = t.detach().contiguous().cpu()
    if x.dtype == torch.bfloat16:
        x = x.view(torch.int16)
    return hashlib.blake2b(x.numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# manifest helpers
# ---------------------------------------------------------------------------
def tbe_resident_bytes(entry: Dict[str, Any]) -> int:
    """Resident bytes of a coded entry, from its recorded array sizes."""
    if "resident_bytes" in entry and entry["resident_bytes"] is not None:
        return int(entry["resident_bytes"])
    bs = entry.get("byte_size") or {}
    return int(bs.get("total", 0))


def entry_dense_bytes(entry: Dict[str, Any]) -> int:
    if entry.get("original_bytes") is not None:
        return int(entry["original_bytes"])
    n = 1
    for d in entry.get("shape") or []:
        n *= int(d)
    return n * 2


def accounting(tensors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-component byte census, from the manifest alone."""
    out: Dict[str, Dict[str, int]] = {}
    for t in tensors:
        comp = t.get("component") or component_of(t["name"])
        slot = out.setdefault(comp, {
            "n_tbe": 0, "n_raw": 0, "dense_bytes": 0, "resident_bytes": 0,
        })
        dense = entry_dense_bytes(t)
        slot["dense_bytes"] += dense
        if t["kind"] == "tbe":
            slot["n_tbe"] += 1
            slot["resident_bytes"] += tbe_resident_bytes(t)
        else:
            slot["n_raw"] += 1
            slot["resident_bytes"] += dense
    total = {
        k: sum(v[k] for v in out.values())
        for k in ("n_tbe", "n_raw", "dense_bytes", "resident_bytes")
    }
    return {"by_component": out, "total": total}


def validate_manifest(manifest: Dict[str, Any]) -> None:
    """Fail closed on anything that is not a well-formed serving manifest."""
    if not isinstance(manifest, dict):
        raise BundleError("malformed_manifest", "manifest is not an object")
    if manifest.get("format") != BUNDLE_FORMAT:
        raise BundleError(
            "unsupported_format",
            f"format {manifest.get('format')!r}; this loader reads {BUNDLE_FORMAT!r}",
        )
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise BundleError("unsupported_schema", str(manifest.get("schema")))
    if manifest.get("status") != "ok":
        raise BundleError("manifest_not_ok", str(manifest.get("status")))
    shards = manifest.get("shards")
    tensors = manifest.get("tensors")
    if not isinstance(shards, list) or not shards:
        raise BundleError("malformed_manifest", "no shards")
    if not isinstance(tensors, list) or not tensors:
        raise BundleError("malformed_manifest", "no tensors")
    for i, s in enumerate(shards):
        for key in ("file", "bytes", "sha256"):
            if key not in s:
                raise BundleError("malformed_manifest", f"shard {i} lacks {key!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(s["sha256"])):
            raise BundleError("malformed_manifest", f"shard {i} sha256 is not hex64")
        rel = str(s["file"])
        if rel.startswith("/") or ".." in Path(rel).parts:
            raise BundleError("unsafe_path", rel)
    names = set()
    for t in tensors:
        name = t.get("name")
        if not name or name in names:
            raise BundleError("malformed_manifest", f"missing or duplicate tensor {name!r}")
        names.add(name)
        if t.get("kind") not in ("tbe", "raw"):
            raise BundleError("malformed_manifest", f"{name}: kind {t.get('kind')!r}")
        shard = t.get("shard")
        if not isinstance(shard, int) or not 0 <= shard < len(shards):
            raise BundleError("malformed_manifest", f"{name}: shard {shard!r} out of range")
        if not t.get("blake2b_source"):
            raise BundleError(
                "missing_source_hash",
                f"{name}: no blake2b_source; the bit-exact gate would have "
                "nothing to check this tensor against",
            )
    for s in manifest.get("sidecars") or []:
        rel = str(s.get("file", ""))
        if not rel or rel.startswith("/") or ".." in Path(rel).parts:
            raise BundleError("unsafe_path", rel)


# ---------------------------------------------------------------------------
# sources: where shard bytes come from
# ---------------------------------------------------------------------------
def _gcs_token() -> Optional[str]:
    """An OAuth token from the GCE metadata server, else gcloud, else None."""
    try:
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/"
            "service-accounts/default/token",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=2) as r:
            return json.loads(r.read().decode())["access_token"]
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True, text=True, timeout=30, check=True,
        )
        tok = out.stdout.strip()
        return tok or None
    except Exception:
        return None


def resolve_url(base: str, rel: str) -> Tuple[str, Dict[str, str]]:
    """``(https_url, headers)`` for ``rel`` under an http(s):// or gs:// base."""
    base = str(base).rstrip("/")
    if base.startswith("gs://"):
        bucket, _, prefix = base[len("gs://"):].partition("/")
        obj = f"{prefix}/{rel}" if prefix else rel
        url = (
            "https://storage.googleapis.com/download/storage/v1/b/"
            f"{urllib.parse.quote(bucket, safe='')}/o/"
            f"{urllib.parse.quote(obj, safe='')}?alt=media"
        )
        tok = _gcs_token()
        return url, ({"Authorization": f"Bearer {tok}"} if tok else {})
    if base.startswith(("http://", "https://")):
        return f"{base}/{urllib.parse.quote(rel)}", {}
    raise BundleError("unsupported_url", base)


def is_remote(location: str) -> bool:
    return str(location).startswith(("gs://", "http://", "https://"))


@dataclass
class FetchResult:
    rel: str
    path: Path
    bytes: int
    sha256: str
    source: str            # "local" | "cache" | "remote"
    fetch_s: float
    verify_s: float


class BundleSource:
    """Local directory or remote prefix; hands out VERIFIED local paths."""

    def __init__(
        self,
        location: str,
        *,
        cache_dir: Optional[os.PathLike | str] = None,
        opener: Optional[Callable[[str, Dict[str, str]], Any]] = None,
    ):
        self.location = str(location)
        self.remote = is_remote(self.location)
        if self.remote:
            if cache_dir is None:
                raise BundleError(
                    "no_cache_dir",
                    "a remote bundle is fetched into a local cache; pass "
                    "cache_dir (never /tmp on this fleet)",
                )
            self.root = Path(cache_dir)
            self.root.mkdir(parents=True, exist_ok=True)
        else:
            self.root = Path(self.location)
            if not self.root.is_dir():
                raise BundleError("not_a_directory", str(self.root))
        self._opener = opener or (
            lambda url, headers: urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=120,
            )
        )

    # -- raw read of the manifest (tiny; hashed by the caller if pinned) ----
    def read_small(self, rel: str) -> bytes:
        local = self.root / rel
        if not self.remote:
            return local.read_bytes()
        url, headers = resolve_url(self.location, rel)
        with self._opener(url, headers) as r:
            data = r.read()
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_name(local.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, local)
        return data

    def fetch(self, rel: str, *, sha256: str, nbytes: int) -> FetchResult:
        """A local path whose sha256 has JUST been checked against ``sha256``."""
        local = self.root / rel
        t0 = time.perf_counter()
        if local.is_file():
            got = sha256_file(local)
            dt = time.perf_counter() - t0
            if got == sha256 and local.stat().st_size == int(nbytes):
                return FetchResult(rel, local, int(nbytes), got,
                                   "cache" if self.remote else "local", 0.0, dt)
            if not self.remote:
                raise BundleError(
                    "shard_hash_mismatch",
                    f"{local}: sha256 {got} != manifest {sha256} "
                    f"(size {local.stat().st_size} vs {nbytes}); refusing to use it",
                )
            local.unlink()  # stale/corrupt cache entry: re-download
        if not self.remote:
            raise BundleError("missing_shard", str(local))
        url, headers = resolve_url(self.location, rel)
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_name(local.name + ".tmp")
        h = hashlib.sha256()
        n = 0
        t0 = time.perf_counter()
        with self._opener(url, headers) as r, open(tmp, "wb") as f:
            while True:
                b = r.read(FETCH_CHUNK_BYTES)
                if not b:
                    break
                h.update(b)
                f.write(b)
                n += len(b)
        dt = time.perf_counter() - t0
        got = h.hexdigest()
        if got != sha256 or n != int(nbytes):
            tmp.unlink(missing_ok=True)
            raise BundleError(
                "shard_hash_mismatch",
                f"{url}: downloaded {n} bytes sha256 {got}; manifest says "
                f"{nbytes} bytes sha256 {sha256}. Deleted, not used.",
            )
        os.replace(tmp, local)
        return FetchResult(rel, local, n, got, "remote", dt, 0.0)


# ---------------------------------------------------------------------------
# the opened bundle
# ---------------------------------------------------------------------------
@dataclass
class Bundle:
    source: BundleSource
    manifest: Dict[str, Any]
    manifest_sha256: str
    local_dir: Path                       # sidecars live here after open
    sidecar_receipts: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def tensors(self) -> List[Dict[str, Any]]:
        return self.manifest["tensors"]

    @property
    def shards(self) -> List[Dict[str, Any]]:
        return self.manifest["shards"]

    def entries_for_shard(self, index: int) -> List[Dict[str, Any]]:
        return [t for t in self.tensors if t["shard"] == index]

    def entry(self, name: str) -> Dict[str, Any]:
        for t in self.tensors:
            if t["name"] == name:
                return t
        raise KeyError(name)

    def total_bytes(self) -> int:
        return sum(int(s["bytes"]) for s in self.shards)


def open_bundle(
    location: str,
    *,
    cache_dir: Optional[os.PathLike | str] = None,
    expected_manifest_sha256: Optional[str] = None,
    opener=None,
) -> Bundle:
    """Read and validate the manifest, fetch + verify every sidecar."""
    src = BundleSource(location, cache_dir=cache_dir, opener=opener)
    raw = src.read_small(MANIFEST_FILENAME)
    msha = hashlib.sha256(raw).hexdigest()
    if expected_manifest_sha256 and msha != expected_manifest_sha256.lower():
        raise BundleError(
            "manifest_hash_mismatch",
            f"{MANIFEST_FILENAME} sha256 {msha} != pinned {expected_manifest_sha256}",
        )
    manifest = json.loads(raw.decode("utf-8"))
    validate_manifest(manifest)
    receipts = []
    for sc in manifest.get("sidecars") or []:
        res = src.fetch(sc["file"], sha256=sc["sha256"], nbytes=int(sc["bytes"]))
        receipts.append({"file": sc["file"], "sha256": res.sha256, "source": res.source})
    return Bundle(
        source=src, manifest=manifest, manifest_sha256=msha,
        local_dir=src.root, sidecar_receipts=receipts,
    )


# ---------------------------------------------------------------------------
# prefetching shard stream
# ---------------------------------------------------------------------------
class ShardStream:
    """Yield ``(shard_index, FetchResult)`` in manifest order, prefetching.

    Up to ``depth`` shards are fetched and hash-verified CONCURRENTLY ahead of
    the consumer (one connection each), so network, hashing and GPU upload
    overlap and a remote cold start is not limited to one TCP stream.  Shards
    are still handed out strictly in order.  With ``evict=True`` a remote
    shard's cached file is deleted once the consumer moves past it, bounding
    local disk to ``depth + 1`` shards.
    """

    def __init__(self, bundle: Bundle, *, depth: int = 2, evict: bool = False,
                 order: Optional[List[int]] = None):
        self.bundle = bundle
        self.depth = max(1, int(depth))
        self.evict = bool(evict) and bundle.source.remote
        self.order = list(order) if order is not None else list(range(len(bundle.shards)))
        self.fetch_log: List[Dict[str, Any]] = []

    def _fetch(self, idx: int) -> "FetchResult":
        s = self.bundle.shards[idx]
        return self.bundle.source.fetch(s["file"], sha256=s["sha256"], nbytes=int(s["bytes"]))

    def __iter__(self) -> Iterator[Tuple[int, FetchResult]]:
        from concurrent.futures import ThreadPoolExecutor

        pending = list(self.order)
        with ThreadPoolExecutor(max_workers=self.depth,
                                thread_name_prefix="glc-serve-shard-fetch") as pool:
            inflight = []
            while pending and len(inflight) < self.depth:
                idx = pending.pop(0)
                inflight.append((idx, pool.submit(self._fetch, idx)))
            try:
                while inflight:
                    idx, fut = inflight.pop(0)
                    t_wait = time.perf_counter()
                    res = fut.result()          # re-raises a hash mismatch here
                    waited = time.perf_counter() - t_wait
                    if pending:
                        nxt = pending.pop(0)
                        inflight.append((nxt, pool.submit(self._fetch, nxt)))
                    self.fetch_log.append({
                        "shard": idx, "file": res.rel, "bytes": res.bytes,
                        "source": res.source, "fetch_s": round(res.fetch_s, 4),
                        "verify_s": round(res.verify_s, 4),
                        "consumer_wait_s": round(waited, 4), "sha256_ok": True,
                    })
                    yield idx, res
                    if self.evict:
                        try:
                            res.path.unlink()
                        except OSError:
                            pass
            finally:
                for _idx, fut in inflight:
                    fut.cancel()


def read_entry(handle, entry: Dict[str, Any]):
    """The stored payload of one manifest entry from an open safetensors handle.

    ``tbe`` entries come back as a ``glc_loader.tbe_container.TBETensor``
    with the arrays EXACTLY as stored (no decode); ``raw`` entries as the
    stored tensor.
    """
    if entry["kind"] == "raw":
        return handle.get_tensor(entry["name"])
    from glc_loader.tbe_container import TBETensor

    coded = entry.get("coded_shape") or entry["shape"]
    if len(coded) != 2:
        raise BundleError("uncodable_entry_shape", f"{entry['name']}: {coded}")
    name = entry["name"]
    return TBETensor(
        shape=(int(coded[0]), int(coded[1])),
        layout=str(entry["layout"]),
        mode=int(entry["mode"]),
        base=int(entry["base"]),
        tiles=int(entry["tiles"]),
        escapes=int(entry["escapes"]),
        planes=handle.get_tensor(f"{name}.planes"),
        smb=handle.get_tensor(f"{name}.smb"),
        esc=handle.get_tensor(f"{name}.esc"),
        sbbase=handle.get_tensor(f"{name}.sbbase"),
        superblock=int(entry["superblock"]),
    )


def iter_bundle_entries(
    bundle: Bundle, *, depth: int = 2, evict: bool = False,
    stream: Optional[ShardStream] = None,
) -> Iterator[Tuple[Dict[str, Any], Any]]:
    """``(entry, payload)`` for every tensor, one at a time, shards verified."""
    from safetensors import safe_open

    stream = stream or ShardStream(bundle, depth=depth, evict=evict)
    for idx, res in stream:
        with safe_open(str(res.path), framework="pt", device="cpu") as h:
            for entry in bundle.entries_for_shard(idx):
                yield entry, read_entry(h, entry)


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------
class BundleWriter:
    """Write groups of tensors into size-bounded shards, then the manifest.

    A group (one decoder layer, the vision tower, ...) is never split across
    shards.  Shards are written ``.tmp`` then renamed, hashed from disk after
    the write, and the manifest is written last -- a bundle with a manifest
    is a complete bundle.
    """

    def __init__(self, out_dir: os.PathLike | str, *,
                 target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES):
        self.out = Path(out_dir)
        if self.out.exists() and any(self.out.iterdir()):
            raise BundleError("out_not_empty", f"{self.out} exists and is not empty")
        (self.out / SHARD_DIR).mkdir(parents=True, exist_ok=True)
        self.target = int(target_shard_bytes)
        self._pending: Dict[str, Any] = {}
        self._pending_entries: List[Dict[str, Any]] = []
        self._pending_bytes = 0
        self._pending_groups: List[str] = []
        self.shards: List[Dict[str, Any]] = []
        self.tensors: List[Dict[str, Any]] = []

    @staticmethod
    def _nbytes(t) -> int:
        return int(t.numel()) * int(t.element_size())

    def add_group(self, group: str, items: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> None:
        """``items`` = ``[(entry_meta, {key: tensor})]`` for one group."""
        gbytes = sum(self._nbytes(t) for _, arrays in items for t in arrays.values())
        if self._pending and self._pending_bytes + gbytes > self.target:
            self.flush()
        for entry, arrays in items:
            for key, t in arrays.items():
                if key in self._pending:
                    raise BundleError("duplicate_key", key)
                self._pending[key] = t.contiguous()
            e = dict(entry)
            e["shard"] = len(self.shards)
            e["group"] = group
            self._pending_entries.append(e)
        self._pending_bytes += gbytes
        self._pending_groups.append(group)
        if self._pending_bytes >= self.target:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        from safetensors.torch import save_file

        idx = len(self.shards)
        rel = f"{SHARD_DIR}/shard-{idx:05d}.safetensors"
        path = self.out / rel
        tmp = path.with_name(path.name + ".tmp")
        save_file(self._pending, str(tmp), metadata={
            "format": BUNDLE_FORMAT, "shard": str(idx),
        })
        os.replace(tmp, path)
        self.shards.append({
            "file": rel,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "n_tensors": len(self._pending_entries),
            "groups": list(self._pending_groups),
            "payload_bytes": int(self._pending_bytes),
        })
        self.tensors.extend(self._pending_entries)
        self._pending = {}
        self._pending_entries = []
        self._pending_bytes = 0
        self._pending_groups = []

    def copy_sidecars(self, source_dir: os.PathLike | str) -> List[Dict[str, Any]]:
        src = Path(source_dir)
        out = []
        for name in SIDECAR_CANDIDATES:
            p = src / name
            if p.is_file():
                shutil.copy2(p, self.out / name)
                out.append({"file": name, "bytes": p.stat().st_size,
                            "sha256": sha256_file(self.out / name)})
        return out

    def finalize(self, *, sidecars: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
        self.flush()
        names = {s["file"] for s in sidecars}
        for required in ("config.json",):
            if required not in names:
                raise BundleError(
                    "missing_config",
                    "a serving bundle builds its skeleton from its own "
                    "config.json; pass --config-dir with the source checkpoint's files",
                )
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "format": BUNDLE_FORMAT,
            "status": "ok",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **meta,
            "sidecars": sidecars,
            "shards": self.shards,
            "tensors": self.tensors,
            "accounting": accounting(self.tensors),
        }
        manifest.setdefault("gate", {"sample": default_gate_sample(self.tensors)})
        body = json.dumps(manifest, indent=1, sort_keys=True).encode("utf-8")
        tmp = self.out / (MANIFEST_FILENAME + ".tmp")
        tmp.write_bytes(body)
        os.replace(tmp, self.out / MANIFEST_FILENAME)
        digest = hashlib.sha256(body).hexdigest()
        (self.out / (MANIFEST_FILENAME + ".sha256")).write_text(
            f"{digest}  {MANIFEST_FILENAME}\n", encoding="utf-8",
        )
        manifest["_sha256"] = digest
        return manifest


def default_gate_sample(tensors: List[Dict[str, Any]], *, n_text: int = 8) -> List[str]:
    """A deterministic sample the startup gate always decodes and hashes.

    Every coded head/MTP tensor, two coded vision tensors, the largest coded
    tensor, ``n_text`` evenly spaced coded text tensors, and three raw tensors
    (the embedding when present, one norm, one vision tensor).
    """
    coded = [t for t in tensors if t["kind"] == "tbe"]
    raw = [t for t in tensors if t["kind"] == "raw"]
    pick: List[str] = []

    def add(name):
        if name not in pick:
            pick.append(name)

    for t in coded:
        if component_of(t["name"]) in ("lm_head", "mtp", "embed"):
            add(t["name"])
    vision = [t for t in coded if component_of(t["name"]) == "vision"]
    for t in vision[:1] + vision[-1:]:
        add(t["name"])
    if coded:
        add(max(coded, key=entry_dense_bytes)["name"])
    text = [t for t in coded if component_of(t["name"]) == "text"]
    if text:
        step = max(1, len(text) // max(1, n_text))
        for t in text[::step][:n_text]:
            add(t["name"])
    for comp in ("embed", "text", "vision"):
        for t in raw:
            if component_of(t["name"]) == comp:
                add(t["name"])
                break
    return pick


__all__ = [
    "BUNDLE_FORMAT",
    "Bundle",
    "BundleError",
    "BundleSource",
    "BundleWriter",
    "COMPONENTS",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA",
    "SIDECAR_CANDIDATES",
    "ShardStream",
    "accounting",
    "blake2b_tensor_hex",
    "component_of",
    "default_gate_sample",
    "group_of",
    "group_sort_key",
    "is_remote",
    "iter_bundle_entries",
    "layer_of",
    "open_bundle",
    "read_entry",
    "resolve_url",
    "sha256_file",
    "tbe_resident_bytes",
    "validate_manifest",
]
