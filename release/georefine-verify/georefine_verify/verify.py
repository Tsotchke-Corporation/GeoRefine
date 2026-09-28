"""Verify that decoding a GeoRefine TBE bundle reproduces the parent's published tensors.

Chain of evidence (nothing of ours is trusted):

1. The parent's files are fetched at a pinned commit of the Hugging Face repo and
   each shard's sha256 is checked against the LFS sha256 Hugging Face publishes
   for that commit (``download`` / local modes).  ``range`` mode skips this step
   and says so in the receipt.
2. The tensor list comes from the parent's ``model.safetensors.index.json``;
   every parent tensor must have exactly one bundle entry and vice versa.
3. Each bundle tensor is rebuilt on the CPU: ``raw`` entries are read as stored,
   ``tbe`` entries are decoded by :mod:`georefine_verify.tbe`.  The manifest
   supplies only decode parameters (shape, mode, base, layout); a wrong
   parameter can only produce a mismatch, never a false pass.
4. The rebuilt bytes are compared with the parent's bytes, element for element,
   and both sides are sha256-hashed.  A tensor passes only if the dtype, shape
   and every byte agree.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import __version__
from .sources import (
    ByteSource, Location, SourceError, STEntry, download, eprint, hf_revision_info,
    read_safetensors_header, sha256_source,
)
from .tbe import TBEFormatError, TBEParams, iter_decode

MANIFEST_NAMES = ("serve_manifest.json",)
RAW_CHUNK = 1 << 26


def _dtype_tag(manifest_dtype: str) -> str:
    d = manifest_dtype.lower()
    for tag, keys in (("BF16", ("bfloat16", "bf16")), ("F16", ("float16", "f16", "half")),
                      ("F32", ("float32", "f32")), ("F64", ("float64",)), ("I64", ("int64",)),
                      ("I32", ("int32",)), ("U8", ("uint8",)), ("BOOL", ("bool",))):
        if any(d.endswith(k) or d == k for k in keys):
            return tag
    return manifest_dtype


def sha256_of_sources(files) -> str:
    h = hashlib.sha256()
    for p in sorted(files):
        h.update(Path(p).name.encode() + b"\0" + Path(p).read_bytes())
    return h.hexdigest()


@dataclass
class Options:
    bundle: str
    reference: str
    reference_dir: Optional[str] = None
    reference_mode: str = "download"         # download | local | range
    bundle_mode: str = "download"            # download | range (remote bundles only)
    work_dir: str = "georefine-verify-work"
    tensors: Optional[str] = None            # regex filter
    limit: Optional[int] = None
    evict_reference: bool = False
    max_rate_mbps: Optional[float] = None
    check_bundle_shards: bool = True
    expect_manifest_sha256: Optional[str] = None
    chunk_tiles: int = 1 << 16
    receipt: Optional[str] = None
    fresh: bool = False
    log_every_s: float = 10.0
    extra: dict = field(default_factory=dict)


class Verifier:
    def __init__(self, opt: Options, log=eprint):
        self.opt = opt
        self.log = log
        self.work = Path(opt.work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        self._open: Dict[str, ByteSource] = {}
        self._headers: Dict[str, Dict[str, STEntry]] = {}

    # ---------------------------------------------------------------- setup
    def _src(self, loc: Location, rel: str, size=None) -> ByteSource:
        key = f"{loc.kind}:{loc.base}:{loc.revision}:{rel}"
        if key not in self._open:
            self._open[key] = loc.open(rel, size)
        return self._open[key]

    def _header(self, key: str, src: ByteSource) -> Dict[str, STEntry]:
        if key not in self._headers:
            self._headers[key] = read_safetensors_header(src)[0]
        return self._headers[key]

    def setup(self) -> dict:
        from .sources import parse_location
        opt = self.opt
        # -- bundle
        self.bloc = parse_location(opt.bundle)
        raw = None
        for name in MANIFEST_NAMES:
            try:
                raw = self.bloc.read_small(name)
                self.manifest_name = name
                break
            except (OSError, SourceError):
                continue
        if raw is None:
            raise SourceError(f"no serve_manifest.json in {opt.bundle}")
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        if opt.expect_manifest_sha256 and not self.manifest_sha256.startswith(
                opt.expect_manifest_sha256.lower()):
            raise SourceError(f"bundle manifest sha256 {self.manifest_sha256} != expected "
                              f"{opt.expect_manifest_sha256}")
        self.manifest = json.loads(raw)
        if not str(self.manifest.get("format", "")).startswith("georefine.tbe.serve"):
            raise SourceError(f"unknown bundle format {self.manifest.get('format')!r}")
        self.bentries = {t["name"]: t for t in self.manifest["tensors"]}
        self.bshards = self.manifest["shards"]
        # -- reference
        rspec = opt.reference
        if "@" not in rspec and not os.path.isdir(rspec):
            raise SourceError("--reference must pin a revision: REPO@COMMIT")
        if os.path.isdir(rspec):
            self.ref_repo, self.ref_rev, self.hf = None, None, None
            opt.reference_dir = rspec
            opt.reference_mode = "local"
        else:
            self.ref_repo, _, rev = rspec.partition("@")
            self.hf = hf_revision_info(self.ref_repo, rev)
            if not self.hf["sha"].startswith(rev):
                raise SourceError(f"revision {rev} resolved to {self.hf['sha']}")
            self.ref_rev = self.hf["sha"]
            if opt.reference_dir:
                opt.reference_mode = "local"
        if opt.reference_mode == "local":
            self.rloc = Location("local", os.path.abspath(opt.reference_dir))
        elif opt.reference_mode == "range":
            self.rloc = Location("hf", self.ref_repo, self.ref_rev)
        else:
            self.rloc = Location("local", str((self.work / "reference").resolve()))
        index_loc = Location("hf", self.ref_repo, self.ref_rev) if self.hf else self.rloc
        self.index_raw = index_loc.read_small("model.safetensors.index.json")
        self.weight_map: Dict[str, str] = json.loads(self.index_raw)["weight_map"]
        # -- coverage (the population is the PARENT's, not ours)
        rnames, bnames = set(self.weight_map), set(self.bentries)
        self.coverage = {
            "parent_tensors": len(rnames), "bundle_tensors": len(bnames),
            "missing_from_bundle": sorted(rnames - bnames),
            "extra_in_bundle": sorted(bnames - rnames),
        }
        self.coverage["ok"] = not self.coverage["missing_from_bundle"] and \
            not self.coverage["extra_in_bundle"]
        # -- selection
        names = sorted(rnames & bnames)
        if opt.tensors:
            import re
            rx = re.compile(opt.tensors)
            names = [n for n in names if rx.search(n)]
        if opt.limit:
            names = names[: opt.limit]
        self.selected = names
        self.full = (not opt.tensors and not opt.limit)
        return {
            "bundle": self.bloc.describe(), "manifest_sha256": self.manifest_sha256,
            "reference": f"{self.ref_repo}@{self.ref_rev}" if self.hf else opt.reference_dir,
            "reference_mode": opt.reference_mode, "selected": len(names), **self.coverage,
        }

    # ---------------------------------------------------------------- resume
    def _run_key(self) -> dict:
        return {"manifest_sha256": self.manifest_sha256,
                "reference": f"{self.ref_repo}@{self.ref_rev}",
                "reference_index_sha256": hashlib.sha256(self.index_raw).hexdigest(),
                "verifier_version": __version__}

    def _load_progress(self) -> Dict[str, dict]:
        p = self.work / "progress.jsonl"
        done: Dict[str, dict] = {}
        if self.opt.fresh and p.exists():
            p.rename(self.work / f"progress.{int(time.time())}.jsonl")
        if p.exists():
            with open(p) as f:
                first = f.readline()
                if first and json.loads(first).get("run_key") == self._run_key():
                    for line in f:
                        try:
                            r = json.loads(line)
                        except json.JSONDecodeError:
                            continue          # a torn final line from a kill
                        if "name" in r:
                            done[r["name"]] = r
                else:
                    p.rename(self.work / f"progress.stale.{int(time.time())}.jsonl")
        if not p.exists():
            with open(p, "w") as f:
                f.write(json.dumps({"run_key": self._run_key()}) + "\n")
        self._progress = open(p, "a")
        return done

    def _record(self, r: dict):
        self._progress.write(json.dumps(r, sort_keys=True) + "\n")
        self._progress.flush()
        os.fsync(self._progress.fileno())

    # ---------------------------------------------------------------- shards
    def _reference_shard(self, shard: str) -> dict:
        """Make the parent shard available; returns the shard digest record."""
        pub = (self.hf or {}).get("files", {}).get(shard, {}) if self.hf else {}
        want_sha, want_size = pub.get("sha256"), pub.get("size")
        rec = {"file": shard, "published_sha256": want_sha, "published_size": want_size}
        marker = self.work / "reference-verified" / (shard + ".json")
        if self.opt.reference_mode == "range":
            rec.update(sha256=None, checked=False, note="range mode: shard not hashed")
            return rec
        if marker.exists():
            return json.loads(marker.read_text())
        path = Path(self.rloc.base) / shard
        if self.opt.reference_mode == "download":
            if not path.exists():
                self.log(f"[ref] downloading {shard} ({(want_size or 0) / 1e9:.2f} GB) "
                         f"at {self.ref_rev[:12]}")
                got = download(Location("hf", self.ref_repo, self.ref_rev).url(shard), path,
                               size=want_size, sha256=want_sha,
                               max_rate=self.opt.max_rate_mbps, log=self.log)
            else:
                got = sha256_source(self._src(self.rloc, shard))
        else:
            self.log(f"[ref] hashing {path}")
            got = sha256_source(self._src(self.rloc, shard))
        rec["sha256"] = got
        rec["size"] = path.stat().st_size
        rec["checked"] = want_sha is not None
        rec["ok"] = (want_sha is None or got == want_sha) and \
            (want_size is None or rec["size"] == want_size)
        if not rec["ok"]:
            raise SourceError(f"parent shard {shard}: sha256 {got} != published {want_sha}")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(rec, sort_keys=True))
        return rec

    def _bundle_shard_check(self, idx: int) -> dict:
        s = self.bshards[idx]
        rec = {"file": s["file"], "manifest_sha256": s["sha256"], "bytes": s["bytes"]}
        marker = self.work / "bundle-verified" / (Path(s["file"]).name + ".json")
        if self.bloc.kind != "local" or not self.opt.check_bundle_shards:
            rec.update(checked=False)
            return rec
        if marker.exists():
            return json.loads(marker.read_text())
        self.log(f"[bundle] hashing {s['file']}")
        got = sha256_source(self._src(self.bloc, s["file"], s["bytes"]))
        rec.update(sha256=got, checked=True, ok=got == s["sha256"])
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(rec, sort_keys=True))
        return rec

    def _check_sidecars(self) -> List[dict]:
        """config / tokenizer / template files: the bundle's copy must be byte-identical
        to the parent's file of the same name at the pinned revision."""
        out = []
        hf_files = (self.hf or {}).get("files", {})
        ploc = Location("hf", self.ref_repo, self.ref_rev) if self.hf else self.rloc
        for sc in self.manifest.get("sidecars", []):
            f = sc["file"]
            r = {"file": f}
            try:
                b = self.bloc.read_small(f)
                r["bundle_sha256"] = hashlib.sha256(b).hexdigest()
                r["bundle_matches_manifest"] = r["bundle_sha256"] == sc.get("sha256")
            except (OSError, SourceError) as exc:
                r.update(bundle_sha256=None, bundle_matches_manifest=False, error=str(exc))
            in_parent = f in hf_files if self.hf else (Path(self.rloc.base) / f).is_file()
            if in_parent:
                r["parent_sha256"] = hashlib.sha256(ploc.read_small(f)).hexdigest()
                r["identical"] = r["parent_sha256"] == r["bundle_sha256"]
            else:
                r.update(parent_sha256=None, identical=None, note="not a file of the parent")
            out.append(r)
            if r.get("identical") is False:
                self.log(f"[FAIL] sidecar {f} differs from the parent's")
        n_same = sum(1 for r in out if r.get("identical"))
        self.log(f"[sidecars] {n_same}/{len(out)} bundle sidecar files byte-identical to the parent's")
        return out

    def _download_bundle(self):
        """Fetch the bundle shards the selection needs into WORK_DIR/bundle, each
        sha256-checked against the bundle manifest, then read them locally."""
        dest = (self.work / "bundle").resolve()
        dest.mkdir(parents=True, exist_ok=True)
        (dest / self.manifest_name).write_bytes(self.bloc.read_small(self.manifest_name))
        need = sorted({int(self.bentries[x]["shard"]) for x in self.selected})
        for i in need:
            s = self.bshards[i]
            path = dest / s["file"]
            marker = self.work / "bundle-verified" / (Path(s["file"]).name + ".json")
            if path.exists() and marker.exists():
                continue
            self.log(f"[bundle] downloading {s['file']} ({s['bytes'] / 1e9:.2f} GB)")
            got = download(self.bloc.url(s["file"]), path, size=int(s["bytes"]),
                           sha256=s["sha256"], max_rate=self.opt.max_rate_mbps, log=self.log)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"file": s["file"], "manifest_sha256": s["sha256"],
                                          "bytes": s["bytes"], "sha256": got, "checked": True,
                                          "ok": got == s["sha256"]}, sort_keys=True))
        self.bundle_origin = self.bloc.describe()
        self.bloc = Location("local", str(dest))

    # ---------------------------------------------------------------- one tensor
    def verify_tensor(self, name: str) -> dict:
        e = self.bentries[name]
        rshard = self.weight_map[name]
        rsrc = self._src(self.rloc, rshard, (self.hf or {}).get("files", {}).get(rshard, {}).get("size")
                         if self.opt.reference_mode == "range" else None)
        rh = self._header("ref:" + rshard, rsrc)[name]
        bshard = self.bshards[int(e["shard"])]
        bsrc = self._src(self.bloc, bshard["file"], bshard["bytes"])
        bh = self._header("bundle:" + bshard["file"], bsrc)
        t0 = time.time()
        r = {"name": name, "kind": e["kind"], "shape": list(rh.shape), "dtype": rh.dtype,
             "bytes": rh.nbytes, "parent_shard": rshard, "bundle_shard": bshard["file"]}
        problems: List[str] = []
        if list(rh.shape) != [int(x) for x in e["shape"]]:
            problems.append(f"shape: parent {list(rh.shape)} bundle {e['shape']}")
        if _dtype_tag(str(e.get("dtype", ""))) != rh.dtype:
            problems.append(f"dtype: parent {rh.dtype} bundle {e.get('dtype')}")
        href, hdec = hashlib.sha256(), hashlib.sha256()
        mismatched, first_bad = 0, None
        if problems:
            pass
        elif e["kind"] == "raw":
            be = bh.get(name)
            if be is None:
                problems.append("raw tensor absent from its bundle shard")
            elif be.dtype != rh.dtype or list(be.shape) != list(rh.shape):
                problems.append(f"bundle stores {be.dtype}{list(be.shape)}")
            else:
                for off in range(0, rh.nbytes, RAW_CHUNK):
                    n = min(RAW_CHUNK, rh.nbytes - off)
                    a = rsrc.read(rh.offset + off, n)
                    b = bsrc.read(be.offset + off, n)
                    href.update(a)
                    hdec.update(b)
                    if a != b:
                        av = np.frombuffer(a, np.uint8)
                        bv = np.frombuffer(b, np.uint8)
                        d = np.flatnonzero(av != bv)
                        mismatched += int(d.size)
                        if first_bad is None:
                            first_bad = off + int(d[0])
                r["mismatch_unit"] = "byte"
        elif e["kind"] == "tbe":
            r.update(mode=int(e["mode"]), base=int(e["base"]), escapes=int(e["escapes"]),
                     layout=e["layout"])
            try:
                mismatched, first_bad = self._verify_tbe(e, rh, rsrc, bh, bsrc, href, hdec)
            except TBEFormatError as exc:
                problems.append(f"malformed TBE payload: {exc}")
            r["mismatch_unit"] = "bf16 element"
        else:
            problems.append(f"unknown bundle entry kind {e['kind']!r}")
        r["parent_sha256"] = href.hexdigest() if not problems else None
        r["decoded_sha256"] = hdec.hexdigest() if not problems else None
        r["mismatched"] = mismatched
        r["first_mismatch"] = first_bad
        r["problems"] = problems
        r["match"] = (not problems and mismatched == 0
                      and r["parent_sha256"] == r["decoded_sha256"])
        r["seconds"] = round(time.time() - t0, 3)
        return r

    def _verify_tbe(self, e, rh, rsrc, bh, bsrc, href, hdec):
        name = e["name"]
        n, k = (int(x) for x in (e.get("coded_shape") or e["shape"]))
        if rh.dtype != "BF16":
            raise TBEFormatError(f"parent dtype {rh.dtype}; TBE codes bf16 only")
        if n * k * 2 != rh.nbytes:
            raise TBEFormatError(f"coded shape {[n, k]} does not cover the parent's {rh.shape}")
        p = TBEParams(n=n, k=k, layout=str(e["layout"]), mode=int(e["mode"]),
                      base=int(e["base"]), superblock=int(e.get("superblock", 32)))
        arr = {}
        for part, dt in (("planes", "I32"), ("smb", "U8"), ("esc", "U8"), ("sbbase", "I32")):
            a = bh.get(f"{name}.{part}")
            if a is None:
                raise TBEFormatError(f"{name}.{part} absent from {e['shard']}")
            if a.dtype != dt:
                raise TBEFormatError(f"{name}.{part} is {a.dtype}, want {dt}")
            arr[part] = a
        if arr["planes"].nbytes != p.tiles * 24:
            raise TBEFormatError(f"planes {arr['planes'].nbytes} B for {p.tiles} tiles")
        if arr["smb"].nbytes != p.tiles * 64:
            raise TBEFormatError(f"smb {arr['smb'].nbytes} B for {p.tiles} tiles")
        sbbase = np.frombuffer(bsrc.read(arr["sbbase"].offset, arr["sbbase"].nbytes), "<i4")

        def reader(a):
            return lambda off, nb: bsrc.read(a.offset + off, nb)

        mismatched, first_bad = 0, None
        for e0, words in iter_decode(p, reader(arr["planes"]), reader(arr["smb"]),
                                     reader(arr["esc"]), sbbase, arr["esc"].nbytes,
                                     chunk_tiles=self.opt.chunk_tiles):
            ref_b = rsrc.read(rh.offset + 2 * e0, 2 * words.size)
            dec_b = words.astype("<u2").tobytes()
            href.update(ref_b)
            hdec.update(dec_b)
            if ref_b != dec_b:
                d = np.flatnonzero(np.frombuffer(ref_b, "<u2") != words)
                mismatched += int(d.size)
                if first_bad is None:
                    first_bad = e0 + int(d[0])
        return mismatched, first_bad

    # ---------------------------------------------------------------- run
    def run(self) -> dict:
        t_start = time.time()
        info = self.setup()
        self.log(f"[setup] bundle {info['bundle']}  manifest sha256 {self.manifest_sha256}")
        self.log(f"[setup] parent {info['reference']}  mode={info['reference_mode']}  "
                 f"parent tensors {info['parent_tensors']}  bundle tensors {info['bundle_tensors']}  "
                 f"coverage {'OK' if self.coverage['ok'] else 'FAIL'}  selected {info['selected']}")
        done = self._load_progress()
        self.sidecars = self._check_sidecars()
        if self.bloc.kind != "local" and self.opt.bundle_mode == "download":
            self._download_bundle()
        by_shard: Dict[str, List[str]] = {}
        for nme in self.selected:
            by_shard.setdefault(self.weight_map[nme], []).append(nme)
        results: Dict[str, dict] = {}
        ref_shards: List[dict] = []
        bundle_shards: Dict[int, dict] = {}
        total_bytes = sum(self._bytes_hint(nme) for nme in self.selected)
        seen_bytes, n_done = 0, 0
        last_log = 0.0
        for shard in sorted(by_shard):
            names = by_shard[shard]
            todo = [x for x in names if not done.get(x, {}).get("match")]
            if not todo:
                for x in names:
                    results[x] = done[x]
                    seen_bytes += done[x]["bytes"]
                    n_done += 1
                marker = self.work / "reference-verified" / (shard + ".json")
                if marker.exists():
                    ref_shards.append(json.loads(marker.read_text()))
                continue
            ref_shards.append(self._reference_shard(shard))
            if self.opt.reference_mode != "range":
                hdr = self._header("ref:" + shard, self._src(self.rloc, shard))
                names = sorted(names, key=lambda x: hdr[x].offset)
            for nme in names:
                if done.get(nme, {}).get("match"):
                    results[nme] = done[nme]
                else:
                    bi = int(self.bentries[nme]["shard"])
                    if bi not in bundle_shards:
                        bundle_shards[bi] = self._bundle_shard_check(bi)
                    try:
                        r = self.verify_tensor(nme)
                    except (SourceError, OSError) as exc:
                        r = {"name": nme, "match": False, "bytes": self._bytes_hint(nme),
                             "problems": [f"io: {exc}"], "kind": self.bentries[nme]["kind"]}
                    self._record(r)
                    results[nme] = r
                    if not r["match"]:
                        self.log(f"[FAIL] {nme}: {r.get('problems') or ''} "
                                 f"mismatched={r.get('mismatched')}")
                n_done += 1
                seen_bytes += results[nme]["bytes"]
                now = time.time()
                if now - last_log > self.opt.log_every_s or n_done == len(self.selected):
                    last_log = now
                    el = now - t_start
                    rate = seen_bytes / max(el, 1e-9)
                    eta = (total_bytes - seen_bytes) / max(rate, 1)
                    npass = sum(1 for v in results.values() if v.get("match"))
                    self.log(f"[{n_done}/{len(self.selected)}] {npass} bit-identical  "
                             f"{seen_bytes / 1e9:.1f}/{total_bytes / 1e9:.1f} GB  "
                             f"{rate / 1e6:.0f} MB/s  eta {eta / 60:.1f} min  last {nme}")
            self._close_shard(shard)
        # bundle shards never touched in this invocation (resume) are still reported
        for bi in sorted({int(self.bentries[x]["shard"]) for x in self.selected}):
            if bi not in bundle_shards:
                bundle_shards[bi] = self._bundle_shard_check(bi)
        return self._receipt(results, ref_shards, bundle_shards, time.time() - t_start)

    def _bytes_hint(self, name: str) -> int:
        e = self.bentries[name]
        return int(e.get("original_bytes") or 0)

    def _close_shard(self, shard: str):
        key = f"{self.rloc.kind}:{self.rloc.base}:{self.rloc.revision}:{shard}"
        src = self._open.pop(key, None)
        if src is not None:
            src.close()
        if self.opt.evict_reference and self.opt.reference_mode == "download":
            p = Path(self.rloc.base) / shard
            if p.exists():
                p.unlink()
                self.log(f"[ref] evicted {shard} after verification")

    def _receipt(self, results, ref_shards, bundle_shards, seconds) -> dict:
        recs = [results[n] for n in self.selected if n in results]
        n_match = sum(1 for r in recs if r.get("match"))
        by_kind: Dict[str, Dict[str, int]] = {}
        for r in recs:
            k = by_kind.setdefault(r.get("kind", "?"), {"n": 0, "match": 0, "bytes": 0})
            k["n"] += 1
            k["match"] += int(bool(r.get("match")))
            k["bytes"] += int(r.get("bytes") or 0)
        ref_ok = all(s.get("checked") and s.get("ok") for s in ref_shards)
        bundle_ok = all(s.get("ok", True) for s in bundle_shards.values())
        complete = self.full and len(recs) == self.coverage["parent_tensors"]
        caveats = []
        if not complete:
            caveats.append(f"subset: {len(recs)} of {self.coverage['parent_tensors']} parent tensors")
        if not ref_ok:
            caveats.append("parent shards not checked against Hugging Face's published LFS sha256")
        bundle_checked = all(s.get("checked") for s in bundle_shards.values())
        if not bundle_checked:
            caveats.append("bundle shard sha256 not checked against the bundle manifest")
        sidecars = getattr(self, "sidecars", [])
        sidecars_ok = all(r.get("identical") is not False and r.get("bundle_matches_manifest")
                          for r in sidecars)
        if any(r.get("identical") is None for r in sidecars):
            caveats.append("bundle carries files the parent does not have: "
                           + ", ".join(r["file"] for r in sidecars if r.get("identical") is None))
        if n_match != len(recs) or not self.coverage["ok"] or not bundle_ok or not sidecars_ok:
            verdict = "FAIL"
        elif not complete or not ref_ok or not bundle_checked:
            verdict = "PASS_PARTIAL"
        else:
            verdict = "PASS"
        src_files = sorted(Path(__file__).parent.glob("*.py"))
        receipt = {
            "schema": "georefine.public_verify.v1",
            "verdict": verdict,
            "caveats": caveats,
            "summary": {
                "bit_identical": n_match, "verified": len(recs),
                "parent_tensors": self.coverage["parent_tensors"],
                "by_kind": by_kind,
                "parent_shards_checked_against_hf_lfs": ref_ok,
                "sidecars_identical": sum(1 for r in sidecars if r.get("identical")),
                "sidecars": len(sidecars),
                "bundle_shards_checked_against_manifest": all(
                    s.get("checked") for s in bundle_shards.values()) and bundle_ok,
                "complete": complete,
            },
            "coverage": self.coverage,
            "sidecars": sidecars,
            "parent": {
                "repo": self.ref_repo, "revision": self.ref_rev, "mode": self.opt.reference_mode,
                "index_sha256": hashlib.sha256(self.index_raw).hexdigest(),
                "license": (self.hf or {}).get("license"), "shards": ref_shards,
            },
            "bundle": {
                "location": getattr(self, "bundle_origin", None) or self.bloc.describe(),
                "manifest_file": self.manifest_name,
                "manifest_sha256": self.manifest_sha256, "format": self.manifest.get("format"),
                "shards": [bundle_shards[i] for i in sorted(bundle_shards)],
            },
            "verifier": {
                "version": __version__, "source_sha256": sha256_of_sources(src_files),
                "python": sys.version.split()[0], "numpy": np.__version__,
                "platform": platform.platform(), "host": socket.gethostname(),
                "argv": sys.argv, "seconds": round(seconds, 1),
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            "tensors": recs,
        }
        out = Path(self.opt.receipt) if self.opt.receipt else self.work / "receipt.json"
        out.write_text(json.dumps(receipt, indent=1, sort_keys=False))
        receipt["receipt_path"] = str(out)
        return receipt
