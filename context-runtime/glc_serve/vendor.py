"""Vendor the GPU kernel sources next to ``glc_loader`` (a release-build step).

``glc_loader.tbe_mma_kernels`` JIT-builds ``tbe_mma_kernel.cu`` from its own
directory and ``glc_loader.container.load_kernels`` imports
``fwp1_kernels.py`` from its own directory; neither is tracked in this
repository -- ``experiments/georefine/_glc_release.py::_copy_loader`` copies
them in when it builds a release.  A serving box running from a source
checkout needs the same two copies, which is all this does (a file copy by
path plus a sha256 receipt; nothing is imported from ``experiments``).

    python -m glc_serve.vendor --repo /path/to/attention
    python -m glc_serve.vendor --clean     # remove the copies again

Run it on a serving box, not in a development checkout you then test:
``tests/test_glc_release.py`` scans ``release/glc_loader/*.py`` for imports of
``experiments`` and the vendored ``fwp1_kernels.py`` is a verbatim copy of a
research module (it lazily imports ``experiments.georefine._glc_fwp1_sm80``
behind a default-off flag), so that check flags it -- exactly as it would a
release staging directory.  ``--clean`` removes both copies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

PAIRS = (
    ("experiments/georefine/_glc_tbe_mma_kernel.cu", "release/glc_loader/tbe_mma_kernel.cu"),
    ("experiments/georefine/_glc_lut_gemv.py", "release/glc_loader/fwp1_kernels.py"),
)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def vendor(repo: os.PathLike | str) -> dict:
    root = Path(repo)
    out = {}
    for src_rel, dst_rel in PAIRS:
        src, dst = root / src_rel, root / dst_rel
        if not src.is_file():
            raise FileNotFoundError(src)
        if not dst.is_file() or _sha(dst) != _sha(src):
            tmp = dst.with_name(dst.name + ".tmp")
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
        out[dst_rel] = {"from": src_rel, "sha256": _sha(dst)}
    return out


def clean(repo: os.PathLike | str) -> list:
    removed = []
    for _src, dst_rel in PAIRS:
        p = Path(repo) / dst_rel
        if p.is_file():
            p.unlink()
            removed.append(dst_rel)
    return removed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="glc_serve.vendor")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args(argv)
    out = {"removed": clean(args.repo)} if args.clean else vendor(args.repo)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
