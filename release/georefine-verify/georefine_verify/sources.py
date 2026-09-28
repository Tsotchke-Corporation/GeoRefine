"""Byte sources (local file, HTTP range), safetensors headers, Hugging Face lookups.

Standard library only.  No safetensors or huggingface_hub dependency, so the
parsing the verifier relies on is all visible here.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
USER_AGENT = "georefine-verify/1.0"
_RETRIES = 6


class SourceError(RuntimeError):
    pass


# ------------------------------------------------------------------ byte sources
class ByteSource:
    label: str

    def size(self) -> int:
        raise NotImplementedError

    def read(self, offset: int, n: int) -> bytes:
        raise NotImplementedError


class LocalFile(ByteSource):
    def __init__(self, path: os.PathLike | str):
        self.path = Path(path)
        self.label = str(self.path)
        self._fd = os.open(self.path, os.O_RDONLY)
        self._size = os.fstat(self._fd).st_size
        self._read_lock = threading.Lock()

    def size(self) -> int:
        return self._size

    def read(self, offset: int, n: int) -> bytes:
        if offset < 0 or offset + n > self._size:
            raise SourceError(f"{self.label}: read [{offset}, {offset + n}) past end {self._size}")
        out = bytearray()
        while len(out) < n:
            amount = min(n - len(out), 1 << 26)
            at = offset + len(out)
            if hasattr(os, "pread"):
                b = os.pread(self._fd, amount, at)
            else:  # Windows does not expose os.pread.
                with self._read_lock:
                    os.lseek(self._fd, at, os.SEEK_SET)
                    b = os.read(self._fd, amount)
            if not b:
                raise SourceError(f"{self.label}: short read at {offset + len(out)}")
            out += b
        return bytes(out)

    def close(self):
        try:
            os.close(self._fd)
        except OSError:
            pass


def _hf_token() -> Optional[str]:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    p = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
    try:
        return p.read_text().strip() or None
    except OSError:
        return None


_GCS_TOKEN: Dict[str, object] = {}


def _gcs_token() -> str:
    tok = os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN")
    if tok:
        return tok
    now = time.time()
    if _GCS_TOKEN.get("t") and now - float(_GCS_TOKEN["at"]) < 1800:
        return str(_GCS_TOKEN["t"])
    try:
        tok = subprocess.run(["gcloud", "auth", "print-access-token"], check=True,
                             capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SourceError(f"gs:// needs `gcloud auth print-access-token` or "
                          f"GOOGLE_OAUTH_ACCESS_TOKEN: {exc}") from exc
    _GCS_TOKEN.update(t=tok, at=now)
    return tok


def http_headers(url: str) -> Dict[str, str]:
    h = {"User-Agent": USER_AGENT}
    if url.startswith(HF_ENDPOINT):
        tok = _hf_token()
        if tok:
            h["Authorization"] = f"Bearer {tok}"
    elif url.startswith("https://storage.googleapis.com/"):
        h["Authorization"] = f"Bearer {_gcs_token()}"
    return h


def http_get(url: str, byte_range: Optional[Tuple[int, int]] = None, timeout: float = 120) -> bytes:
    """GET (optionally bytes [a, b) ) with retries.  Returns the body."""
    last = None
    for attempt in range(_RETRIES):
        h = http_headers(url)
        if byte_range is not None:
            h["Range"] = f"bytes={byte_range[0]}-{byte_range[1] - 1}"
        req = urllib.request.Request(url, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
            if byte_range is not None:
                want = byte_range[1] - byte_range[0]
                if len(body) != want:
                    raise SourceError(f"range read returned {len(body)} bytes, wanted {want}")
            return body
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403, 404):
                raise SourceError(f"HTTP {exc.code} for {url}") from exc
            last = exc
        except (urllib.error.URLError, OSError, SourceError) as exc:
            last = exc
        time.sleep(min(60, 2 ** attempt))
    raise SourceError(f"GET {url} failed after {_RETRIES} attempts: {last}")


class HttpFile(ByteSource):
    def __init__(self, url: str, size: Optional[int] = None):
        self.url = url
        self.label = url
        self._size = size

    def size(self) -> int:
        if self._size is None:
            req = urllib.request.Request(self.url, headers={**http_headers(self.url),
                                                            "Range": "bytes=0-0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                cr = r.headers.get("Content-Range", "")
                m = re.search(r"/(\d+)$", cr)
                if not m:
                    raise SourceError(f"{self.url}: no Content-Range size")
                self._size = int(m.group(1))
        return self._size

    def read(self, offset: int, n: int) -> bytes:
        if n == 0:
            return b""
        out = bytearray()
        step = 1 << 27
        while len(out) < n:
            a = offset + len(out)
            b = min(offset + n, a + step)
            out += http_get(self.url, (a, b))
        return bytes(out)

    def close(self):
        pass


# ------------------------------------------------------------------ locations
@dataclass
class Location:
    """A directory of files: local, Hugging Face repo@rev, https base, or gs://."""
    kind: str          # local | hf | https | gs
    base: str          # dir path, repo id, base url, or gs://bucket/prefix
    revision: Optional[str] = None

    def url(self, rel: str) -> str:
        if self.kind == "hf":
            return f"{HF_ENDPOINT}/{self.base}/resolve/{self.revision or 'main'}/{urllib.parse.quote(rel)}"
        if self.kind == "https":
            return self.base.rstrip("/") + "/" + urllib.parse.quote(rel)
        if self.kind == "gs":
            bucket, _, prefix = self.base[len("gs://"):].partition("/")
            obj = (prefix.rstrip("/") + "/" + rel) if prefix else rel
            return f"https://storage.googleapis.com/{bucket}/{urllib.parse.quote(obj)}"
        raise SourceError("local location has no URL")

    def open(self, rel: str, size: Optional[int] = None) -> ByteSource:
        if self.kind == "local":
            return LocalFile(Path(self.base) / rel)
        return HttpFile(self.url(rel), size)

    def read_small(self, rel: str) -> bytes:
        if self.kind == "local":
            return (Path(self.base) / rel).read_bytes()
        return http_get(self.url(rel))

    def describe(self) -> str:
        if self.kind == "hf":
            return f"{self.base}@{self.revision}"
        return self.base


_HF_ID = re.compile(r"^(hf://)?([A-Za-z0-9][\w.\-]*/[\w.\-]+)(@([\w.\-/]+))?$")


def parse_location(spec: str) -> Location:
    if spec.startswith("gs://"):
        return Location("gs", spec)
    if spec.startswith(("http://", "https://")):
        return Location("https", spec)
    if spec.startswith("hf://") or (not os.path.exists(spec) and _HF_ID.match(spec)):
        m = _HF_ID.match(spec)
        if not m:
            raise SourceError(f"cannot parse Hugging Face id {spec!r}")
        return Location("hf", m.group(2), m.group(4) or "main")
    if os.path.isdir(spec):
        return Location("local", os.path.abspath(spec))
    raise SourceError(f"{spec!r} is neither a directory, a URL, gs://, nor an HF repo id")


# ------------------------------------------------------------------ Hugging Face API
def hf_revision_info(repo: str, revision: str) -> dict:
    """Full commit sha + per-file size and LFS sha256 at a revision."""
    url = f"{HF_ENDPOINT}/api/models/{repo}/revision/{urllib.parse.quote(revision, safe='')}?blobs=true"
    d = json.loads(http_get(url))
    files = {}
    for s in d.get("siblings", []):
        lfs = s.get("lfs") or {}
        files[s["rfilename"]] = {"size": s.get("size", lfs.get("size")), "sha256": lfs.get("sha256")}
    return {"sha": d["sha"], "files": files, "license": (d.get("cardData") or {}).get("license")}


# ------------------------------------------------------------------ safetensors
DTYPE_SIZE = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "I16": 2, "U16": 2,
              "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}


@dataclass
class STEntry:
    dtype: str
    shape: Tuple[int, ...]
    offset: int     # absolute byte offset in the file
    nbytes: int


def read_safetensors_header(src: ByteSource) -> Tuple[Dict[str, STEntry], dict, int]:
    """-> ({name: entry}, __metadata__, data_start)."""
    (hlen,) = struct.unpack("<Q", src.read(0, 8))
    if hlen > 512 << 20:
        raise SourceError(f"{src.label}: implausible safetensors header length {hlen}")
    hdr = json.loads(src.read(8, hlen))
    start = 8 + hlen
    meta = hdr.pop("__metadata__", {}) or {}
    out = {}
    for name, e in hdr.items():
        a, b = e["data_offsets"]
        shape = tuple(int(x) for x in e["shape"])
        numel = 1
        for d in shape:
            numel *= d
        if e["dtype"] in DTYPE_SIZE and numel * DTYPE_SIZE[e["dtype"]] != b - a:
            raise SourceError(f"{src.label}: {name} has {b - a} bytes for {e['dtype']}{list(shape)}")
        out[name] = STEntry(e["dtype"], shape, start + int(a), int(b) - int(a))
    return out, meta, start


# ------------------------------------------------------------------ hashing / download
def sha256_source(src: ByteSource, chunk: int = 1 << 26, progress=None) -> str:
    h = hashlib.sha256()
    n = src.size()
    for off in range(0, n, chunk):
        h.update(src.read(off, min(chunk, n - off)))
        if progress:
            progress(min(n, off + chunk), n)
    return h.hexdigest()


def download(url: str, dest: Path, *, size: Optional[int], sha256: Optional[str],
             max_rate: Optional[float] = None, log=print) -> str:
    """Stream ``url`` to ``dest`` (resuming a ``.part``), hashing as it goes.
    Returns the sha256.  Raises if it disagrees with ``sha256`` or ``size``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    have = 0
    if part.exists():
        with open(part, "rb") as f:
            while True:
                b = f.read(1 << 24)
                if not b:
                    break
                h.update(b)
                have += len(b)
        if size is not None and have > size:
            part.unlink()
            h, have = hashlib.sha256(), 0
    t0, last = time.time(), 0.0
    start_have = have          # bytes already on disk do not count toward the rate
    attempt = 0
    while size is None or have < size:
        hdr = http_headers(url)
        if have:
            hdr["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=120) as r, \
                    open(part, "ab") as f:
                if have and r.status != 206:
                    raise SourceError("server ignored the resume Range header")
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    f.write(b)
                    h.update(b)
                    have += len(b)
                    attempt = 0
                    el = time.time() - t0
                    if max_rate:
                        ahead = (have - start_have) / (max_rate * 1e6) - el
                        if ahead > 0:
                            time.sleep(ahead)
                    if time.time() - last > 15:
                        last = time.time()
                        tot = f"/{size / 1e9:.2f}" if size else ""
                        log(f"    download {dest.name}: {have / 1e9:.2f}{tot} GB "
                            f"({(have - start_have) / max(1e-9, el) / 1e6:.0f} MB/s)")
            if size is None:
                break
        except (urllib.error.URLError, OSError, SourceError) as exc:
            attempt += 1
            if attempt > _RETRIES:
                raise SourceError(f"download of {url} failed: {exc}") from exc
            time.sleep(min(60, 2 ** attempt))
            # the hash state is only valid for bytes actually written; re-derive it
            h, have = hashlib.sha256(), 0
            with open(part, "rb") as f:
                while True:
                    b = f.read(1 << 24)
                    if not b:
                        break
                    h.update(b)
                    have += len(b)
    got = h.hexdigest()
    if size is not None and have != size:
        raise SourceError(f"{dest.name}: {have} bytes, want {size}")
    if sha256 and got != sha256:
        part.rename(dest.with_name(dest.name + ".BAD"))
        raise SourceError(f"{dest.name}: sha256 {got} != published {sha256}")
    os.replace(part, dest)
    return got


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)
