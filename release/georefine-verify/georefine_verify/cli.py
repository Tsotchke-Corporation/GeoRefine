"""georefine-verify command line."""
from __future__ import annotations

import argparse
import sys

from . import __version__
from .sources import SourceError, eprint
from .verify import Options, Verifier

EPILOG = """\
examples:
  # full check: download the parent at the pinned commit (sha256-checked against
  # Hugging Face's LFS digests), decode every bundle tensor on the CPU, compare bytes
  georefine-verify --bundle ./Qwen3.8-27B-GeoRefine-TBE --reference Qwen/Qwen3.8-27B@1d4bf0f2

  # use a parent copy you already have (its shards are still hashed and checked)
  georefine-verify --bundle ./bundle --reference Qwen/Qwen3.8-27B@1d4bf0f2 --reference-dir ~/qwen

  # quick spot check: layer 0 only, parent bytes fetched by HTTP range request
  georefine-verify --bundle ./bundle --reference Qwen/Qwen3.8-27B@1d4bf0f2 \\
      --reference-mode range --tensors 'layers\\.0\\.'

exit status: 0 = PASS (every parent tensor bit-identical, all digests checked),
             3 = PASS_PARTIAL (everything checked matched, but the check was not complete),
             1 = FAIL, 2 = usage or I/O error.  Re-running with the same --work-dir resumes.
"""


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="georefine-verify", epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Check that a GeoRefine TBE bundle decodes to the parent's published tensors "
                    "bit for bit. CPU only.")
    ap.add_argument("--bundle", required=True,
                    help="bundle: local dir, HF repo id[@rev], https:// base URL, or gs://bucket/prefix")
    ap.add_argument("--reference", required=True,
                    help="parent as HF_REPO@COMMIT (e.g. Qwen/Qwen3.8-27B@1d4bf0f2), or a local dir")
    ap.add_argument("--reference-dir", help="local copy of the parent's shards (still hashed and "
                                            "checked against the published LFS sha256)")
    ap.add_argument("--reference-mode", choices=("download", "range"), default="download",
                    help="download: fetch whole shards into WORK_DIR/reference and check their "
                         "sha256 (default); range: fetch only tensor bytes by HTTP range request "
                         "(fast spot checks; shard digests are then NOT checked)")
    ap.add_argument("--bundle-mode", choices=("download", "range"), default="download",
                    help="for a remote bundle: download shards into WORK_DIR/bundle (sha256-checked "
                         "against the bundle manifest; default) or read tensors by range request")
    ap.add_argument("--work-dir", default="georefine-verify-work",
                    help="progress, downloads and receipt (default ./georefine-verify-work)")
    ap.add_argument("--receipt", help="receipt path (default WORK_DIR/receipt.json)")
    ap.add_argument("--tensors", help="only tensors whose name matches this regex")
    ap.add_argument("--limit", type=int, help="only the first N tensors (sorted by name)")
    ap.add_argument("--evict-reference", action="store_true",
                    help="delete each downloaded parent shard once its tensors are verified "
                         "(peak disk ~4 GB instead of ~56 GB)")
    ap.add_argument("--max-rate", type=float, metavar="MB_PER_S",
                    help="cap each download's rate (parent and bundle shards)")
    ap.add_argument("--skip-bundle-shard-hash", action="store_true",
                    help="do not sha256 local bundle shards against the bundle manifest")
    ap.add_argument("--expect-manifest-sha256", help="refuse a bundle whose manifest hash differs "
                                                     "(prefix allowed)")
    ap.add_argument("--chunk-tiles", type=int, default=1 << 16,
                    help="decode granularity in 64-element tiles (memory ~ 60 B/tile)")
    ap.add_argument("--fresh", action="store_true", help="ignore previous progress in WORK_DIR")
    ap.add_argument("--version", action="version", version=f"georefine-verify {__version__}")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    opt = Options(bundle=a.bundle, reference=a.reference, reference_dir=a.reference_dir,
                  reference_mode=a.reference_mode, bundle_mode=a.bundle_mode,
                  work_dir=a.work_dir, tensors=a.tensors,
                  limit=a.limit, evict_reference=a.evict_reference, max_rate_mbps=a.max_rate,
                  check_bundle_shards=not a.skip_bundle_shard_hash,
                  expect_manifest_sha256=a.expect_manifest_sha256, chunk_tiles=a.chunk_tiles,
                  receipt=a.receipt, fresh=a.fresh)
    try:
        rec = Verifier(opt).run()
    except SourceError as exc:
        eprint(f"error: {exc}")
        return 2
    s = rec["summary"]
    print(f"{rec['verdict']}: {s['bit_identical']}/{s['verified']} tensors bit-identical "
          f"(parent has {s['parent_tensors']})")
    for k, v in sorted(s["by_kind"].items()):
        print(f"  {k:4s} {v['match']}/{v['n']}  {v['bytes'] / 1e9:.2f} GB")
    print(f"  parent  {rec['parent']['repo']}@{rec['parent']['revision']}  "
          f"shards checked vs HF LFS sha256: {s['parent_shards_checked_against_hf_lfs']}")
    print(f"  sidecars (config, tokenizer, templates) identical to the parent's: "
          f"{s['sidecars_identical']}/{s['sidecars']}")
    print(f"  bundle  manifest sha256 {rec['bundle']['manifest_sha256']}  "
          f"shards checked: {s['bundle_shards_checked_against_manifest']}")
    for c in rec.get("caveats", []):
        print(f"  caveat: {c}")
    bad = [t for t in rec["tensors"] if not t.get("match")]
    for t in bad[:20]:
        print(f"  MISMATCH {t['name']}: {t.get('problems')} mismatched={t.get('mismatched')} "
              f"first={t.get('first_mismatch')}")
    print(f"  receipt {rec['receipt_path']}")
    return {"PASS": 0, "PASS_PARTIAL": 3}.get(rec["verdict"], 1)


if __name__ == "__main__":
    sys.exit(main())
