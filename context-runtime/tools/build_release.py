"""Build the georefine-context-fast-runtime wheel and sdist from a clean export of a commit.

    python context-runtime/tools/build_release.py [--out DIR] [--rev REV] [--check-twice]

The build never reads the working tree: ``git archive <rev>`` exports ``context-runtime/``
into a fresh directory under ``DIR`` and the PEP 517 hooks of setuptools build there. A
stale ``build/`` or ``*.egg-info`` directory, an untracked module, or an uncommitted edit
in the checkout therefore cannot reach the artifacts (uncommitted changes are reported).

The artifacts are reproducible for a given commit: ``SOURCE_DATE_EPOCH`` is the commit
time, the wheel's zip entries carry that timestamp, and the sdist is re-packed with sorted
members, that mtime, uid/gid 0, normalised modes and a fixed gzip header.
``--check-twice`` builds a second time in a separate directory and fails unless both
SHA-256 digests match.

After the build the wheel's RECORD is checked against the packaged
``georefine_context_runtime/SOURCE_MANIFEST.json``: every packaged file must be listed with
the same SHA-256, and every listed file must be in the wheel. On a mismatch, run
``tools/refresh_source_manifest.py``, commit, and build again.

``DIR`` defaults to ``<repo>/.scratch/dist``. Paths under /tmp or /private/tmp are refused
(they are wiped on reboot). TMPDIR is never set; the build backend's temporary
directories are created inside ``DIR``.
"""
from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
RUNTIME_DIR_NAME = "context-runtime"
PACKAGES = ("georefine_context_runtime", "glc_loader", "glc_serve")
MANIFEST_REL = "georefine_context_runtime/SOURCE_MANIFEST.json"
VOLATILE = (Path("/tmp"), Path("/private/tmp"))


class BuildError(RuntimeError):
    """The release build or one of its checks failed; the message says which."""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                          text=True).stdout.strip()


def _refuse_volatile(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    for root in VOLATILE:
        if resolved == root or root in resolved.parents:
            raise BuildError(f"refusing output directory {resolved}: /tmp and /private/tmp are "
                             "wiped on reboot")
    return resolved


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def export_tree(repo: Path, rev: str, dest: Path) -> Path:
    """``git archive <rev> context-runtime`` extracted into ``dest``; returns the project dir."""
    dest.mkdir(parents=True)
    archive = subprocess.run(["git", "archive", "--format=tar", rev, RUNTIME_DIR_NAME],
                             cwd=repo, check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        for member in tar.getmembers():
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir()):
                raise BuildError(f"unexpected archive member {member.name!r}")
        tar.extractall(dest)
    project = dest / RUNTIME_DIR_NAME
    if not (project / "pyproject.toml").is_file():
        raise BuildError(f"{rev} has no {RUNTIME_DIR_NAME}/pyproject.toml")
    return project


def apply_profile(project: Path, profile: str) -> None:
    """Change only exported distribution metadata; the default export is untouched."""
    if profile == "cuda":
        return
    if profile not in ("mlx", "cuda129"):
        raise BuildError(f"unknown dependency profile: {profile}")
    path = project / "pyproject.toml"
    text = path.read_text()
    if profile == "cuda129" and 'version = "0.1.1+cu129"' in text:
        required = ('  "torch==2.9.1+cu129",', '  "torchvision==0.24.1+cu129",',
                    '  "transformers==5.5.0",', '  "jinja2>=3.1.0",')
        if any(line not in text for line in required):
            raise BuildError("incomplete pre-profiled CUDA12.9 source metadata")
        return
    if profile == "mlx":
        label = "MLX"
        replacements = {
            'version = "0.1.1"': 'version = "0.1.1+mlx"',
            '  "transformers==5.5.0",': '  "transformers==5.0.0rc3",\n  "mlx==0.30.4",\n  "mlx-lm==0.30.5",',
        }
    elif profile == "cuda129":
        label = "CUDA 12.9"
        replacements = {
            'version = "0.1.1"': 'version = "0.1.1+cu129"',
            '  "torch==2.8.0",': '  "torch==2.9.1+cu129",',
            '  "torchvision==0.23.0",': '  "torchvision==0.24.1+cu129",',
            '  "transformers==5.5.0",': '  "transformers==5.5.0",\n  "jinja2>=3.1.0",',
        }
    else:
        raise BuildError(f"unknown dependency profile: {profile}")
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise BuildError(f"{label} dependency profile no longer matches the pinned base metadata")
        text = text.replace(old, new)
    path.write_text(text)


def run_backend(project: Path, out: Path, epoch: int) -> Tuple[Path, Path]:
    """Build wheel and sdist through setuptools' PEP 517 hooks in a child interpreter."""
    out.mkdir(parents=True)
    env = dict(os.environ)
    env["SOURCE_DATE_EPOCH"] = str(epoch)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)
    code = ("import json, sys, setuptools.build_meta as b; out = sys.argv[1]; "
            "print(json.dumps({'wheel': b.build_wheel(out), 'sdist': b.build_sdist(out)}))")
    proc = subprocess.run([sys.executable, "-c", code, str(out)], cwd=project, env=env,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise BuildError(f"build backend failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    names = json.loads(proc.stdout.strip().splitlines()[-1])
    return out / names["wheel"], out / names["sdist"]


def normalize_wheel(path: Path, epoch: int) -> None:
    """Rewrite the wheel with sorted entries, the epoch timestamp and fixed modes.

    Contents and RECORD are unchanged (RECORD hashes file contents, not zip metadata)."""
    stamp = _zip_time(epoch)
    with zipfile.ZipFile(path) as src:
        items = [(info, src.read(info.filename)) for info in src.infolist()]
    items.sort(key=lambda item: (item[0].filename.endswith("RECORD"), item[0].filename))
    tmp = path.with_name(path.name + ".partial")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for info, data in items:
            new = zipfile.ZipInfo(info.filename, date_time=stamp)
            new.compress_type = zipfile.ZIP_DEFLATED
            new.external_attr = (0o644 << 16)
            new.create_system = 3
            dst.writestr(new, data)
    os.replace(tmp, path)


def _zip_time(epoch: int):
    import time
    t = time.gmtime(max(epoch, 315532800))  # zip cannot store dates before 1980
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)


def normalize_sdist(path: Path, epoch: int) -> None:
    """Re-pack the sdist: sorted members, mtime=epoch, uid/gid 0, no names, fixed gzip header."""
    with tarfile.open(path, "r:gz") as src:
        members = []
        for member in src.getmembers():
            data = src.extractfile(member).read() if member.isfile() else None
            members.append((member, data))
    members.sort(key=lambda item: item[0].name)
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as dst:
        for member, data in members:
            info = tarfile.TarInfo(member.name)
            info.type = member.type
            info.mtime = epoch
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o755 if member.isdir() or member.mode & 0o111 else 0o644
            if data is not None:
                info.size = len(data)
                dst.addfile(info, io.BytesIO(data))
            else:
                dst.addfile(info)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=epoch, compresslevel=9) as gz:
            gz.write(raw.getvalue())
    os.replace(tmp, path)


def _record_hash_hex(value: str) -> str:
    algo, _, b64 = value.partition("=")
    if algo != "sha256":
        raise BuildError(f"unexpected RECORD hash algorithm {algo!r}")
    return base64.urlsafe_b64decode(b64 + "=" * (-len(b64) % 4)).hex()


def check_record_against_manifest(wheel: Path) -> Dict[str, int]:
    """Every packaged file is in SOURCE_MANIFEST.json with its SHA-256, and vice versa."""
    with zipfile.ZipFile(wheel) as zf:
        record_name = next((n for n in zf.namelist() if n.endswith(".dist-info/RECORD")), None)
        if record_name is None:
            raise BuildError("wheel has no RECORD")
        rows = list(csv.reader(io.StringIO(zf.read(record_name).decode("utf-8"))))
        manifest = json.loads(zf.read(MANIFEST_REL))
        for name in zf.namelist():                     # RECORD itself must be truthful
            if name == record_name or name.endswith("/"):
                continue
            want = {r[0]: r[1] for r in rows if len(r) >= 2}.get(name)
            if want is None:
                raise BuildError(f"{name} is in the wheel but not in RECORD")
            if _record_hash_hex(want) != hashlib.sha256(zf.read(name)).hexdigest():
                raise BuildError(f"RECORD hash mismatch for {name}")
    packaged: Dict[str, str] = {}
    for row in rows:
        if len(row) < 2 or not row[1]:
            continue
        top = row[0].split("/", 1)[0]
        if top in PACKAGES and row[0] != MANIFEST_REL:
            packaged[row[0]] = _record_hash_hex(row[1])
    listed = {r["installed_path"]: r["installed_sha256"] for r in manifest["sources"]}
    problems: List[str] = []
    for name, digest in sorted(packaged.items()):
        if name not in listed:
            problems.append(f"in wheel, not in manifest: {name}")
        elif listed[name] != digest:
            problems.append(f"sha256 differs from manifest: {name}")
    for name in sorted(set(listed) - set(packaged)):
        problems.append(f"in manifest, not in wheel: {name}")
    tree = hashlib.sha256(json.dumps(listed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if tree != manifest.get("candidate_tree_sha256"):
        problems.append("candidate_tree_sha256 does not match the manifest's sources")
    if problems:
        shown = "\n  ".join(problems[:40])
        raise BuildError(f"wheel RECORD does not match {MANIFEST_REL} ({len(problems)} problem(s)):\n"
                         f"  {shown}\nrun context-runtime/tools/refresh_source_manifest.py, commit, "
                         "and build again")
    return {"packaged_files": len(packaged), "manifest_files": len(listed)}


def build_once(repo: Path, rev: str, commit: str, epoch: int, workdir: Path, profile: str = "cuda") -> Dict[str, str]:
    if workdir.exists() or workdir.is_symlink():
        raise BuildError(f"refusing existing build staging directory {workdir}; retained files require a new output namespace")
    project = export_tree(repo, commit, workdir / "src")
    apply_profile(project, profile)
    wheel, sdist = run_backend(project, workdir / "dist", epoch)
    normalize_wheel(wheel, epoch)
    normalize_sdist(sdist, epoch)
    return {"wheel": str(wheel), "sdist": str(sdist),
            "wheel_sha256": sha256_file(wheel), "sdist_sha256": sha256_file(sdist)}


def main(argv=None) -> int:
    repo_default = Path(_git(HERE, "rev-parse", "--show-toplevel"))
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(repo_default / ".scratch" / "dist"),
                        help="output directory (default <repo>/.scratch/dist; never /tmp)")
    parser.add_argument("--rev", default="HEAD", help="commit to build (default HEAD)")
    parser.add_argument("--profile", choices=("cuda", "mlx", "cuda129"), default="cuda",
                        help="explicit distribution dependency profile; default unchanged")
    parser.add_argument("--source-date-epoch", type=int, default=None,
                        help="explicit original artifact epoch for reproducing the frozen release; Git commit time remains unchanged")
    parser.add_argument("--check-twice", action="store_true",
                        help="build twice and require identical SHA-256 digests")
    args = parser.parse_args(argv)
    try:
        out = _refuse_volatile(Path(args.out))
        repo = repo_default
        commit = _git(repo, "rev-parse", "--verify", f"{args.rev}^{{commit}}")
        epoch = int(_git(repo, "log", "-1", "--format=%ct", commit))
        if args.source_date_epoch is not None:
            if not 315532800 <= args.source_date_epoch <= epoch:
                raise BuildError("explicit source epoch must be a valid ZIP epoch no later than the source commit")
            epoch = args.source_date_epoch
        if args.rev == "HEAD":
            dirty = _git(repo, "status", "--porcelain", "--untracked-files=no", "--", RUNTIME_DIR_NAME)
            if dirty:
                print(f"[build-release] note: uncommitted changes under {RUNTIME_DIR_NAME}/ are NOT "
                      "in this build:\n" + dirty, file=sys.stderr)
        out.mkdir(parents=True, exist_ok=True)
        work = out / (f".build-{commit[:12]}" + ("-" + args.profile if args.profile != "cuda" else ""))
        first = build_once(repo, args.rev, commit, epoch, work / "a", profile=args.profile)
        result = {"commit": commit, "source_date_epoch": epoch, **first}
        if args.profile != "cuda":
            result["dependency_profile"] = args.profile
        result["record_check"] = check_record_against_manifest(Path(first["wheel"]))
        if args.check_twice:
            second = build_once(repo, args.rev, commit, epoch, work / "b", profile=args.profile)
            same = (first["wheel_sha256"], first["sdist_sha256"]) == (
                second["wheel_sha256"], second["sdist_sha256"])
            result["reproducible"] = same
            if not same:
                raise BuildError(f"two builds of {commit} differ: {first} vs {second}")
        final = {}
        for key in ("wheel", "sdist"):
            src = Path(first[key])
            dst = out / src.name
            if dst.exists() or dst.is_symlink():
                if dst.is_symlink() or not dst.is_file() or sha256_file(dst) != first[f"{key}_sha256"]:
                    raise BuildError(f"refusing to overwrite existing artifact {dst}")
            else:
                with src.open("rb") as source, dst.open("xb") as target:
                    shutil.copyfileobj(source, target)
            if sha256_file(dst) != first[f"{key}_sha256"]:
                raise BuildError(f"copy of {src.name} changed its digest")
            final[key] = str(dst)
        result["retained_build_root"] = str(work)
        result.update(final)
    except (BuildError, subprocess.CalledProcessError, OSError, ValueError) as exc:
        detail = getattr(exc, "stderr", "") or ""
        print(f"[build-release] FAILED: {exc} {detail}".strip(), file=sys.stderr)
        return 1
    print(f"{result['wheel_sha256']}  {Path(result['wheel']).name}")
    print(f"{result['sdist_sha256']}  {Path(result['sdist']).name}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
