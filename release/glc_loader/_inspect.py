"""Format detection, three-scope accounting and typed verification.

This module is the part of the client CLI that has to be HONEST about which
kind of directory it was handed, because four different things in this
project's history have all been called "the artifact":

  ``GLC-RELEASE/1``   the self-contained FWP1 release (``MANIFEST.json``,
                      ``CERTIFICATE.json``, vendored loader, ``model/``).
                      Reads with :mod:`.artifact`; loads with :mod:`.loader`.
  ``georefine.tbe.v2`` the self-contained TBE container (``manifest.json``,
                      ``compression_info.json``, ``SHA256SUMS``, config +
                      tokenizer, two safetensors blobs).  Reads with
                      :mod:`.tbe_artifact`.
  TBE v1 transcode    ``manifest.json`` + two blobs and NOTHING else -- no
                      config, no tokenizer, no format tag.  It cannot be
                      loaded without the dense checkpoint it was made from,
                      on either the CUDA or the Metal path.  This is a
                      refusal, never a best-effort load.
  M2 artifact         ``manifest.json`` + ``blobs/``.  A different codec
                      entirely, audited by the build repository's
                      ``experiments.georefine.lossless_audit``.  Named here
                      only so it gets a typed refusal instead of being
                      mistaken for a broken container.

Nothing here imports the repository that built any of them: the whole point
of the release format is that a stranger can run this with ``pip install
glc-loader`` and no checkout.

EXIT CODES.  ``verify`` extends the 0/1/2/3 scheme
``experiments.georefine.lossless_audit`` established, with one addition:

  0  verified
  1  an integrity check FAILED (a digest or a decoded tensor did not match)
  2  malformed, unreadable, or not a GeoRefine container at all
  3  verified bit-exact, but the artifact EXPANDS (stores more bytes than
     the dense weights it replaces)
  4  the format is recognised but this directory CANNOT be verified from
     what it ships -- a v1 TBE transcode output is the case that exists

4 is the code that matters for honesty.  ``lossless_audit`` on a TBE
directory exits 2 ("no blobs/"), which reads as "malformed" when the truth is
"correct artifact, wrong auditor".  A typed reason beats a misleading verdict.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- kinds -----------------------------------------------------------------
KIND_GLC_RELEASE_V1 = "GLC-RELEASE/1"
KIND_TBE_V2 = "georefine.tbe.v2"
KIND_TBE_V1_TRANSCODE = "georefine.tbe_transcode/v1"
KIND_M2 = "georefine.m2"
KIND_UNKNOWN = "unknown"

# --- exit codes ------------------------------------------------------------
EXIT_OK = 0
EXIT_INTEGRITY_FAILED = 1
EXIT_MALFORMED = 2
EXIT_EXPANSION = 3
EXIT_UNVERIFIABLE = 4

TBE_MANIFEST_SCHEMA = "georefine.tbe_transcode_manifest.v1"

#: Filenames a TBE certify receipt is conventionally written under when it is
#: kept beside the artifact.  ``scripts/glc_tbe_certify.py`` takes ``--out``
#: and imposes no name, so absence is normal and is reported as such -- it is
#: never read as a pass.
_CERTIFICATE_CANDIDATES: Tuple[str, ...] = (
    "CERTIFICATE.json",
    "certificate.json",
    "CERTIFY.json",
    "certify.json",
    "tbe_certify.json",
    "certify_receipt.json",
)


class InspectError(RuntimeError):
    """A typed refusal.  ``reason`` is stable; ``detail`` says what to do."""

    def __init__(self, reason: str, detail: str = "", exit_code: int = EXIT_MALFORMED):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail
        self.exit_code = int(exit_code)


@dataclass
class Detected:
    """What a directory turned out to be, plus everything already parsed."""

    kind: str
    root: Path
    manifest: Dict[str, Any] = field(default_factory=dict)
    compression_info: Dict[str, Any] = field(default_factory=dict)
    certificate: Dict[str, Any] = field(default_factory=dict)
    certificate_path: Optional[str] = None
    reason: str = ""
    detail: str = ""

    @property
    def understood(self) -> bool:
        return self.kind in (KIND_GLC_RELEASE_V1, KIND_TBE_V2)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
    return obj if isinstance(obj, dict) else None


def find_certificate(root: Path) -> Tuple[Optional[str], Dict[str, Any]]:
    """The first recognisable certificate/receipt in the directory, or none.

    A file that parses but carries neither a ``verdict`` nor a ``gates``
    section is not a certificate and is ignored rather than half-read.
    """
    for name in _CERTIFICATE_CANDIDATES:
        obj = _read_json(root / name)
        if obj is None:
            continue
        if "verdict" in obj or "gates" in obj or "stages" in obj:
            return name, obj
    return None, {}


def detect(path: os.PathLike | str) -> Detected:
    """Classify a directory by its declared format tag, never by file names.

    A plain Hugging Face checkpoint ships ``config.json`` and
    ``model.safetensors``; a detector that inferred "container" from files on
    disk would claim every model on the hub.  Every branch below reads a
    declared string.
    """
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise InspectError(
            "not_a_directory",
            f"{root} is not a directory. Pass the artifact DIRECTORY, not a "
            "file inside it.",
        )

    # 1. GLC-RELEASE/1 declares itself in MANIFEST.json's "format".
    upper = _read_json(root / "MANIFEST.json")
    if upper is not None and upper.get("format") == KIND_GLC_RELEASE_V1:
        cert_name, cert = find_certificate(root)
        return Detected(
            kind=KIND_GLC_RELEASE_V1, root=root, manifest=upper,
            certificate=cert, certificate_path=cert_name,
        )

    # 2. georefine.tbe.v2 declares itself in compression_info.json.
    info = _read_json(root / "compression_info.json")
    tag = info.get("artifact_format") if isinstance(info, dict) else None
    lower = _read_json(root / "manifest.json")
    if isinstance(tag, str) and tag:
        if tag != KIND_TBE_V2:
            return Detected(
                kind=KIND_UNKNOWN, root=root, compression_info=info or {},
                reason="unsupported_artifact_format",
                detail=(
                    f"{root} declares artifact_format {tag!r}; this loader "
                    f"reads {KIND_TBE_V2!r} and {KIND_GLC_RELEASE_V1!r}."
                ),
            )
        cert_name, cert = find_certificate(root)
        return Detected(
            kind=KIND_TBE_V2, root=root, manifest=lower or {},
            compression_info=info or {}, certificate=cert,
            certificate_path=cert_name,
        )

    # 3. A TBE transcode manifest with no tag is the legacy v1 output.
    if lower is not None and lower.get("schema") == TBE_MANIFEST_SCHEMA:
        return Detected(
            kind=KIND_TBE_V1_TRANSCODE, root=root, manifest=lower,
            reason="legacy_v1_container",
            detail=(
                f"{root} is a v1 TBE transcode directory: a manifest and two "
                "blobs, with no compression_info.json, no config.json and no "
                "tokenizer. It CANNOT be loaded without the dense checkpoint "
                "it was made from -- the CUDA path "
                "(tbe_stream_loader.stream_load_tbe_model) re-encodes the "
                "dense shards and never opens the stored containers at all, "
                "and the Metal path (metal.tbe_mlx_model.load_tbe_model) "
                "calls mlx_lm.load() on the dense checkpoint first. Upgrade "
                "it to georefine.tbe.v2 in the build repository, or verify "
                "it there against its source."
            ),
        )

    # 4. An M2 artifact: right project, wrong auditor.
    if lower is not None and (root / "blobs").is_dir():
        return Detected(
            kind=KIND_M2, root=root, manifest=lower,
            reason="m2_artifact_not_a_glc_container",
            detail=(
                f"{root} looks like an M2 artifact (manifest.json + blobs/). "
                "That is a different codec with its own auditor: run "
                "`python -m experiments.georefine.lossless_audit <dir>` from "
                "the build repository. Note the converse also holds and is a "
                "filed defect: lossless_audit exits 2 on a TBE directory "
                "because there is no blobs/ -- which reads as 'malformed' "
                "when the truth is 'wrong auditor'."
            ),
        )

    return Detected(
        kind=KIND_UNKNOWN, root=root,
        reason="not_a_georefine_artifact",
        detail=(
            f"{root} declares no GeoRefine format tag. A GLC-RELEASE artifact "
            "has MANIFEST.json with \"format\": \"GLC-RELEASE/1\"; a TBE v2 "
            "artifact has compression_info.json with \"artifact_format\": "
            f"\"{KIND_TBE_V2}\". Neither is present, and file names are never "
            "treated as a claim."
        ),
    )


# ---------------------------------------------------------------------------
# the three scopes -- never conflated, never invented
# ---------------------------------------------------------------------------
def _scope(
    *, ratio: Optional[float], measured_bytes: Optional[int],
    dense_bytes: Optional[int], status: str, basis: str,
) -> Dict[str, Any]:
    return {
        "ratio": ratio,
        "bytes": measured_bytes,
        "dense_equivalent_bytes": dense_bytes,
        "status": status,
        "basis": basis,
    }


def _unmeasured(reason: str) -> Dict[str, Any]:
    return _scope(
        ratio=None, measured_bytes=None, dense_bytes=None,
        status="unmeasured", basis=reason,
    )


def scope_ratios(det: Detected) -> Dict[str, Dict[str, Any]]:
    """``{stored, served_resident, whole_process}``, each with its own basis.

    A scope whose number this artifact does not carry comes back
    ``status="unmeasured"`` with ``ratio=None``.  It is NEVER filled in from
    another scope: "stored" and "served/resident" differ by the safetensors
    header and the file padding, "whole-process" differs from both by the
    activations, the KV cache and the allocator's slack, and this project has
    a standing rule that the three are never conflated.
    """
    if det.kind == KIND_GLC_RELEASE_V1:
        return _glc_v1_scopes(det)
    if det.kind == KIND_TBE_V2:
        return _tbe_scopes(det)
    if det.kind == KIND_TBE_V1_TRANSCODE:
        # The accounting block is the same shape; report it, and let the
        # caller's refusal carry the "not loadable" verdict.
        return _tbe_scopes(det)
    return {
        "stored": _unmeasured("format not recognised"),
        "served_resident": _unmeasured("format not recognised"),
        "whole_process": _unmeasured("format not recognised"),
    }


def _glc_v1_scopes(det: Detected) -> Dict[str, Dict[str, Any]]:
    acc = det.manifest.get("accounting", {}) or {}
    inventory = det.manifest.get("inventory", {}) or {}

    # STORED: measured from the inventory's own file sizes, over the weight
    # shards only.  The manifest's scope_checkpoint counts tensor bytes; the
    # bytes on disk additionally carry the safetensors header, and a "stored"
    # ratio that ignores its own container overhead is not a stored ratio.
    stored_bytes = sum(
        int(v.get("bytes", 0))
        for k, v in inventory.items()
        if k.startswith("model/") and k.endswith(".safetensors")
    )
    dense = acc.get("weight_bytes_dense")
    if stored_bytes and isinstance(dense, (int, float)) and dense:
        stored = _scope(
            ratio=float(dense) / float(stored_bytes),
            measured_bytes=int(stored_bytes),
            dense_bytes=int(dense),
            status="measured",
            basis=(
                "MANIFEST.inventory file sizes of model/*.safetensors vs "
                "accounting.weight_bytes_dense (includes container overhead)"
            ),
        )
    else:
        stored = _unmeasured(
            "the manifest inventory lists no model/*.safetensors shard"
        )

    served = acc.get("scope_served_graph", {}) or {}
    if served.get("resident_bytes") and served.get("dense_bytes"):
        served_scope = _scope(
            ratio=float(served["dense_bytes"]) / float(served["resident_bytes"]),
            measured_bytes=int(served["resident_bytes"]),
            dense_bytes=int(served["dense_bytes"]),
            status="measured",
            basis="MANIFEST.accounting.scope_served_graph (weights only)",
        )
    else:
        served_scope = _unmeasured("accounting.scope_served_graph is absent")

    whole = _whole_process_from_certificate(det)
    if whole is None:
        whole = _unmeasured(
            "GLC-RELEASE/1 records no process-level VRAM measurement; "
            "MANIFEST.accounting.whole_model_ratio is a WEIGHT ratio over "
            "the whole checkpoint, not a whole-PROCESS ratio, and is not "
            "reported here as one"
        )
    return {"stored": stored, "served_resident": served_scope,
            "whole_process": whole}


def _tbe_scopes(det: Detected) -> Dict[str, Dict[str, Any]]:
    acc = det.manifest.get("accounting", {}) or {}
    stored_blk = acc.get("stored", {}) or {}
    served_blk = acc.get("served_resident", {}) or {}
    whole_blk = acc.get("whole_process", {}) or {}

    if stored_blk.get("stored_bytes"):
        stored = _scope(
            ratio=stored_blk.get("ratio"),
            measured_bytes=int(stored_blk["stored_bytes"]),
            dense_bytes=stored_blk.get("dense_equivalent_bytes"),
            status="measured",
            basis="manifest.accounting.stored (bytes on disk)",
        )
    else:
        stored = _unmeasured("manifest.accounting.stored is absent")

    if served_blk.get("resident_bytes"):
        served = _scope(
            ratio=served_blk.get("ratio"),
            measured_bytes=int(served_blk["resident_bytes"]),
            dense_bytes=served_blk.get("dense_equivalent_bytes"),
            status="measured",
            basis="manifest.accounting.served_resident (what the device holds)",
        )
    else:
        served = _unmeasured("manifest.accounting.served_resident is absent")

    whole = _whole_process_from_certificate(det)
    if whole is None:
        if whole_blk.get("ratio") is not None:
            whole = _scope(
                ratio=whole_blk.get("ratio"),
                measured_bytes=whole_blk.get("measured_vram_bytes"),
                dense_bytes=whole_blk.get("dense_equivalent_bytes"),
                status="measured",
                basis="manifest.accounting.whole_process",
            )
        else:
            whole = _unmeasured(
                "the transcode step leaves whole_process blank by design "
                f"(status={whole_blk.get('status', 'absent')!r}); it is "
                "filled only by a certify run on the target card, and no "
                "certify receipt was found in this directory"
            )
    return {"stored": stored, "served_resident": served, "whole_process": whole}


def _whole_process_from_certificate(det: Detected) -> Optional[Dict[str, Any]]:
    """Whole-process VRAM, if a certify receipt in this directory measured it.

    ``scripts/glc_tbe_certify.py``'s ``resident_vram`` stage is the only
    producer of this number in the project.  Nothing else is accepted.
    """
    cert = det.certificate
    if not cert:
        return None
    stage = (cert.get("stages", {}) or {}).get("resident_vram", {}) or {}
    ratios = stage.get("ratios", {}) or {}
    ratio = ratios.get("whole_process")
    if ratio is None:
        return None
    return _scope(
        ratio=ratio,
        measured_bytes=stage.get("whole_process_measured_bytes"),
        dense_bytes=None,
        status="measured",
        basis=(
            f"{det.certificate_path}: stages.resident_vram.ratios."
            "whole_process (measured peak dense vs measured peak coded on "
            "one card; note the certify harness also reports "
            "whole_process_steady and whole_process_including_conversion, "
            "which are different numbers)"
        ),
    )


# ---------------------------------------------------------------------------
# what is NOT certified
# ---------------------------------------------------------------------------
def not_certified(det: Detected) -> List[str]:
    """Everything a reader would otherwise assume and should not.

    Written as a list of plain sentences because the alternative -- a silent
    absence -- is how a compression claim gets read as a quality claim.
    """
    out: List[str] = []
    scopes = scope_ratios(det)
    for name, blk in scopes.items():
        if blk["status"] != "measured":
            out.append(f"{name} ratio: NOT measured -- {blk['basis']}")

    cert = det.certificate
    if not cert:
        out.append(
            "no certificate or certify receipt is present in this directory, "
            "so no gate verdict of any kind is carried with the weights"
        )
    else:
        cov = cert.get("does_not_cover")
        if isinstance(cov, (list, tuple)):
            out.extend(str(x) for x in cov)
        elif isinstance(cov, str):
            out.append(cov)
        refused = cert.get("refused_stages")
        if refused:
            out.append(f"certify stages REFUSED (not measured): {list(refused)}")

    if det.kind == KIND_TBE_V2:
        out.append(
            "bit-exactness here is checked against the manifest's recorded "
            "blake2b_source digests, not against the dense checkpoint: the "
            "chain is dense bytes -> blake2b at transcode time -> manifest "
            "-> sha256 in SHA256SUMS -> this check. It detects every "
            "post-transcode corruption; it cannot detect a transcoder that "
            "recorded a digest of something other than what it read"
        )
    out.append(
        "decode SPEED is not certified by any artifact. Measured decode "
        "throughput is 0.94x dense on a Blackwell RTX PRO 6000 and 0.57-0.62x "
        "on an A100 (single runs, +/-0.05 spread): this format is a VRAM "
        "lever, not a speed lever"
    )
    out.append(
        "downstream task quality (MMLU/GSM8K/HumanEval and the like) is not "
        "measured by any check in this package"
    )
    return out


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def _expansion_verdict(det: Detected) -> Optional[str]:
    """A message iff the artifact stores MORE bytes than the dense weights."""
    stored = scope_ratios(det)["stored"]
    if stored["status"] != "measured" or not stored["ratio"]:
        return None
    if float(stored["ratio"]) < 1.0:
        return (
            f"the artifact EXPANDS: stored ratio {stored['ratio']:.4f}x means "
            f"{stored['bytes']} bytes on disk for "
            f"{stored['dense_equivalent_bytes']} dense bytes. A lossless tier "
            "that enlarges what it stores is an honesty defect even when "
            "every byte round-trips."
        )
    return None


def verify_artifact(
    det: Detected, *, deep: bool = False, progress: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the strongest integrity check this format supports, and say so.

    Returns a dict that always carries ``status``, ``exit_code``,
    ``reason`` and ``checked`` (a plain-English statement of what was proved).
    Never raises for an artifact-shaped problem -- the exit code carries it.
    """
    base: Dict[str, Any] = {
        "artifact": str(det.root),
        "kind": det.kind,
        "deep": bool(deep),
    }

    if det.kind == KIND_GLC_RELEASE_V1:
        return {**base, **_verify_glc_v1(det, deep=deep, progress=progress)}
    if det.kind == KIND_TBE_V2:
        return {**base, **_verify_tbe_v2(det, deep=deep, limit=limit)}

    if det.kind == KIND_TBE_V1_TRANSCODE:
        return {
            **base,
            "status": "UNVERIFIABLE",
            "exit_code": EXIT_UNVERIFIABLE,
            "reason": det.reason,
            "detail": det.detail,
            "checked": (
                "nothing. The manifest records a blake2b_source per tensor, "
                "but a v1 directory carries no SHA256SUMS binding that "
                "manifest to the blobs, so a check against it would be a "
                "check of the manifest against itself."
            ),
        }
    return {
        **base,
        "status": "FAIL",
        "exit_code": EXIT_MALFORMED,
        "reason": det.reason or "not_a_georefine_artifact",
        "detail": det.detail,
        "checked": "nothing",
    }


def _verify_glc_v1(det: Detected, *, deep: bool, progress: bool) -> Dict[str, Any]:
    from .artifact import GLCArtifactError, verify as _verify

    try:
        res = _verify(str(det.root), deep=deep, progress=progress)
    except GLCArtifactError as exc:
        return {
            "status": "FAIL",
            "exit_code": EXIT_INTEGRITY_FAILED,
            "reason": "glc_release_verify_failed",
            "detail": f"{type(exc).__name__}: {exc}",
            "checked": "sha256 of every inventory file, then the tensor index",
        }
    expansion = _expansion_verdict(det)
    checked = (
        "every file in the manifest inventory hashes to its recorded sha256, "
        "and every indexed tensor is present with its declared shape"
    )
    if deep:
        checked += (
            "; additionally every coded tensor was decoded and matched to the "
            "source digest the builder recorded BEFORE encoding, and "
            "re-encoded to byte-identical planes"
        )
    if expansion:
        return {
            "status": "EXPANSION",
            "exit_code": EXIT_EXPANSION,
            "reason": "artifact_expands",
            "detail": expansion,
            "checked": checked,
            "result": res,
        }
    return {
        "status": "PASS",
        "exit_code": EXIT_OK,
        "reason": "verified",
        "checked": checked,
        "result": res,
    }


def _verify_tbe_v2(
    det: Detected, *, deep: bool, limit: Optional[int],
) -> Dict[str, Any]:
    try:
        from .tbe_artifact import (
            TBEArtifactError, open_tbe_artifact, verify_sha256sums,
            verify_standalone_bit_exact,
        )
    except ImportError as exc:  # pragma: no cover - packaging guard
        return {
            "status": "FAIL",
            "exit_code": EXIT_MALFORMED,
            "reason": "tbe_artifact_module_unavailable",
            "detail": (
                f"this build of glc_loader has no tbe_artifact module ({exc}); "
                f"{KIND_TBE_V2} artifacts need one that does"
            ),
            "checked": "nothing",
        }

    try:
        art = open_tbe_artifact(str(det.root), verify_digests=True)
    except TBEArtifactError as exc:
        return {
            "status": "FAIL",
            "exit_code": EXIT_MALFORMED,
            "reason": getattr(exc, "reason", "open_failed"),
            "detail": getattr(exc, "detail", str(exc)),
            "checked": "nothing",
        }

    checked = [
        "every metadata file (config, tokenizer, manifest, compression_info) "
        "hashes to its recorded sha256 in SHA256SUMS"
    ]
    try:
        blobs = verify_sha256sums(str(det.root))
    except TBEArtifactError as exc:
        return {
            "status": "FAIL",
            "exit_code": EXIT_INTEGRITY_FAILED,
            "reason": getattr(exc, "reason", "digest_mismatch"),
            "detail": getattr(exc, "detail", str(exc)),
            "checked": "; ".join(checked),
        }
    checked.append(
        f"every one of the {blobs['n_verified']} files in SHA256SUMS -- the "
        "tensor blobs included -- hashes to its recorded sha256"
    )

    bitexact: Dict[str, Any] = {"status": "skipped"}
    if deep:
        bitexact = verify_standalone_bit_exact(art, limit=limit)
        if bitexact.get("status") == "failed":
            return {
                "status": "FAIL",
                "exit_code": EXIT_INTEGRITY_FAILED,
                "reason": "bit_exactness_failed",
                "detail": (
                    f"{bitexact.get('n_mismatched')} coded and "
                    f"{bitexact.get('n_raw_mismatched')} raw tensor(s) did "
                    "not reproduce their recorded source digest"
                ),
                "checked": "; ".join(checked),
                "bit_exact": bitexact,
            }
        if bitexact.get("status") == "refused":
            return {
                "status": "UNVERIFIABLE",
                "exit_code": EXIT_UNVERIFIABLE,
                "reason": bitexact.get("reason", "bit_exact_refused"),
                "detail": bitexact.get("detail", ""),
                "checked": "; ".join(checked),
                "bit_exact": bitexact,
            }
        checked.append(
            f"{bitexact.get('n_checked')} coded and "
            f"{bitexact.get('n_raw_checked')} raw tensors decoded and "
            "reproduced the blake2b_source digest recorded at transcode time"
        )

    expansion = _expansion_verdict(det)
    if expansion:
        return {
            "status": "EXPANSION",
            "exit_code": EXIT_EXPANSION,
            "reason": "artifact_expands",
            "detail": expansion,
            "checked": "; ".join(checked),
            "bit_exact": bitexact,
        }
    return {
        "status": "PASS",
        "exit_code": EXIT_OK,
        "reason": "verified" if deep else "verified_digests_only",
        "checked": "; ".join(checked),
        "bit_exact": bitexact,
        "note": None if deep else (
            "digests only. Pass --deep to decode every container and check "
            "it against the source digest recorded at transcode time."
        ),
    }


# ---------------------------------------------------------------------------
# certificate summary, shared by verify and info
# ---------------------------------------------------------------------------
def certificate_summary(det: Detected) -> Dict[str, Any]:
    """A verdict-first view of whatever certificate the directory carries.

    ``present: False`` is a first-class answer.  A missing verdict is never
    read as a pass -- that rule is enforced in
    ``loader.assert_certificate_usable`` for GLC-RELEASE and repeated here so
    the CLI cannot report something softer than the loader would accept.
    """
    cert = det.certificate
    if not cert:
        return {
            "present": False,
            "path": None,
            "verdict": None,
            "detail": (
                "no certificate in the artifact directory. "
                "scripts/glc_tbe_certify.py writes its receipt to whatever "
                "--out names and imposes no convention, so a certified "
                "artifact can legitimately have its receipt stored elsewhere "
                "-- but nothing in THIS directory attests to anything."
            ),
        }
    out: Dict[str, Any] = {
        "present": True,
        "path": det.certificate_path,
        "verdict": cert.get("verdict"),
    }
    gates = cert.get("gates")
    if isinstance(gates, dict) and gates:
        out["gates"] = {
            k: {
                "passed": v.get("passed") if isinstance(v, dict) else None,
                "measured": v.get("measured") if isinstance(v, dict) else None,
            }
            for k, v in sorted(gates.items())
        }
        out["gates_without_a_verdict"] = sorted(
            k for k, v in gates.items()
            if not isinstance(v, dict) or "passed" not in v
        )
    stages = cert.get("stages")
    if isinstance(stages, dict) and stages:
        out["stages"] = {
            k: (v.get("status") if isinstance(v, dict) else None)
            for k, v in sorted(stages.items())
        }
        out["refused_stages"] = list(cert.get("refused_stages") or [])
    if "frame" in cert:
        out["frame"] = cert["frame"]
    if "measured_at_utc" in cert:
        out["measured_at_utc"] = cert["measured_at_utc"]
    return out


def source_identity(det: Detected) -> Dict[str, Any]:
    """Source model, licence and build provenance, or an honest ``None``."""
    if det.kind == KIND_GLC_RELEASE_V1:
        prov = det.manifest.get("provenance", {}) or {}
        return {
            "source_model": prov.get("source_model"),
            "built_at_utc": prov.get("built_at_utc"),
            "builder_git_sha": prov.get("builder_git_sha"),
            "torch_version_at_build": prov.get("torch_version"),
            "transformers_version_at_build": prov.get("transformers_version"),
            "license": _license_of(det),
        }
    info = det.compression_info or {}
    src = det.manifest.get("source", {}) or {}
    model = info.get("source_model")
    if not model and src.get("model_dir"):
        model = f"(not declared; transcoded from {src['model_dir']})"
    return {
        "source_model": model,
        "built_at_utc": info.get("built_at_utc"),
        "builder_git_sha": info.get("builder_git_sha"),
        "license": _license_of(det),
    }


def _license_of(det: Detected) -> Any:
    """The licence, from a declared field or a shipped file -- never guessed.

    Neither MANIFEST.json nor manifest.json has a licence field today, so the
    common answer is the honest one: a filename if the artifact ships the
    text, otherwise ``None`` and a note.
    """
    declared = (det.compression_info or {}).get("license") or \
        (det.manifest.get("provenance", {}) or {}).get("license")
    if declared:
        return {"declared": declared, "file": None}
    for name in ("LICENSE", "LICENSE.txt", "LICENCE", "LICENSE.md", "NOTICE"):
        p = det.root / name
        if p.is_file():
            first = ""
            try:
                first = p.read_text(encoding="utf-8", errors="replace").strip()
                first = first.splitlines()[0][:120] if first else ""
            except OSError:
                first = ""
            return {"declared": None, "file": name, "first_line": first}
    return {
        "declared": None,
        "file": None,
        "note": (
            "this artifact declares no licence and ships no LICENSE file. "
            "The compressed weights inherit the source model's licence; "
            "check the source model before redistributing."
        ),
    }
