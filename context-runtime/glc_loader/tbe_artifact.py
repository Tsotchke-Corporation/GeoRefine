"""``georefine.tbe.v2`` -- a TBE artifact that can be loaded with no source.

THE DEFECT THIS FIXES.  A ``glc_tbe_transcode`` v1 output directory holds
``manifest.json``, ``tbe_tensors.safetensors`` and ``raw_tensors.safetensors``
and nothing else: no ``config.json``, no tokenizer, no architecture.  Both
load paths therefore reach back to the ORIGINAL DENSE CHECKPOINT --
``tbe_stream_loader.stream_load_tbe_model`` builds its skeleton from
``build_meta_skeleton(reference_dir)`` and then re-encodes the dense shards it
walks (the stored containers are never opened at all), and
``metal.tbe_mlx_model.load_tbe_model`` calls ``mlx_lm.load(dense_path)`` and
only then swaps linears.  An artifact that needs the 53.79 GB checkpoint it
was made from is not a releasable artifact.

WHAT v2 ADDS.  The same two tensor blobs plus:

  ``config.json``            the architecture, so a skeleton can be built
  tokenizer files           so text can go in and come out
  ``compression_info.json`` the FORMAT TAG and the loader entry point
  ``SHA256SUMS``            a digest over every other file in the directory

``manifest.json`` is EXTENDED, never rewritten: its ``schema`` field keeps
the value ``scripts/glc_tbe_certify.load_manifest`` asserts on
(``georefine.tbe_transcode_manifest.v1``), and v2 adds ``artifact_format``,
``compression_schema_version`` and ``sidecars`` beside it.  Every existing
consumer of a v1 manifest reads a v2 manifest unchanged.

DISPATCH IS BY FORMAT TAG, NEVER BY FILE PRESENCE.  A plain Hugging Face
checkpoint ships ``config.json``, a tokenizer and ``model.safetensors``; a
loader that decided "this looks like a container" from the files on disk
would claim it.  :func:`detect_artifact_format` reads
``compression_info.json`` and returns the ``artifact_format`` string it
declares, or ``None``.  Nothing else is consulted, in either direction.

v1 STILL WORKS, AND FAILS CLOSED HERE.  A v1 directory has no format tag, so
:func:`detect_artifact_format` returns ``None`` and every v1 consumer keeps
using the dense-reference path exactly as before.  Asking THIS module to load
one raises :class:`TBEArtifactError` with reason ``legacy_v1_container`` and
the command that upgrades it -- never a silent fallback to reading a dense
checkpoint the caller believed was not involved.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .tbe_container import TBETensor, decode_tbe

#: The format tag.  Present in ``compression_info.json`` and mirrored into
#: ``manifest.json``; its presence and value are the ONLY dispatch signal.
ARTIFACT_FORMAT = "georefine.tbe.v2"

#: ``compression_info.json``'s own schema, versioned independently of the
#: artifact format so a field can be added without reissuing artifacts.
COMPRESSION_INFO_SCHEMA = "georefine.tbe_compression_info.v1"

#: The integer the manifest and compression_info both carry.  v1 containers
#: carry no such field at all -- absence, not 1, is how they are recognised.
COMPRESSION_SCHEMA_VERSION = 2

#: The importable entry point recorded in ``compression_info.json``'s
#: ``loader`` field.  ``module:function``; the module is importable from an
#: artifact directory with ``release/`` on ``sys.path`` and imports nothing
#: from the repository that built the artifact.
STANDALONE_LOADER = "glc_loader.tbe_artifact:load_standalone"

COMPRESSION_INFO_FILENAME = "compression_info.json"
MANIFEST_FILENAME = "manifest.json"
SHA256SUMS_FILENAME = "SHA256SUMS"
TBE_BLOB_FILENAME = "tbe_tensors.safetensors"
RAW_BLOB_FILENAME = "raw_tensors.safetensors"
CONFIG_FILENAME = "config.json"

TBE_STANDALONE_RECEIPT_SCHEMA = "georefine.tbe_standalone_load_receipt.v1"
TBE_STANDALONE_BITEXACT_SCHEMA = "georefine.tbe_standalone_bit_exact.v1"

#: Every tokenizer/config sidecar a v2 artifact copies out of the source
#: checkpoint when it is present.  Absence of an OPTIONAL file is not an
#: error; absence of the REQUIRED set below is.
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
    "processor_config.json",
    "LICENSE",
    "LICENSE.txt",
    "NOTICE",
)

#: A tokenizer is present iff ``tokenizer_config.json`` is there AND at least
#: one file that actually carries a vocabulary.  ``tokenizer_config.json``
#: alone names a tokenizer class and no tokens; a vocabulary alone has no
#: class to instantiate.  Both halves, or the artifact refuses.
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
TOKENIZER_VOCAB_FILES: Tuple[str, ...] = (
    "tokenizer.json", "tokenizer.model", "vocab.json",
)

#: Files that are never covered by ``SHA256SUMS`` -- the digest file itself
#: (it cannot contain its own digest) and OS detritus.
_DIGEST_EXCLUDE = (SHA256SUMS_FILENAME, ".DS_Store")

_ST_TO_TORCH = {
    "BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64",
    "U8": "uint8", "I8": "int8", "I16": "int16", "I32": "int32",
    "I64": "int64", "BOOL": "bool",
}


class TBEArtifactError(RuntimeError):
    """A typed refusal from the self-contained TBE artifact path.

    ``reason`` is a stable machine-readable token; ``detail`` says what to do
    about it.  Every refusal in this module raises this and nothing falls
    back to reading a dense checkpoint.
    """

    def __init__(self, reason: str, detail: str = ""):
        self.reason = str(reason)
        self.detail = str(detail)
        super().__init__(f"{reason}: {detail}" if detail else str(reason))


# ---------------------------------------------------------------------------
# digests
# ---------------------------------------------------------------------------
def _sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def iter_digestable_files(directory: os.PathLike | str) -> List[str]:
    """Every regular file under ``directory``, POSIX-relative and sorted.

    Recursive, so a tokenizer shipped in a subdirectory is covered too.
    ``SHA256SUMS`` itself is excluded -- a file cannot contain its own digest.
    """
    root = Path(directory)
    out: List[str] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel in _DIGEST_EXCLUDE or rel.endswith("/.DS_Store"):
            continue
        out.append(rel)
    return sorted(out)


def write_sha256sums(directory: os.PathLike | str) -> Dict[str, str]:
    """Write ``SHA256SUMS`` over every other file.  Returns the mapping.

    Format is ``sha256sum(1)``'s: ``<64 hex>  <relative path>``, two spaces,
    sorted by path, so ``sha256sum -c SHA256SUMS`` verifies it with no code
    of ours involved.
    """
    root = Path(directory)
    digests = {rel: _sha256_file(root / rel) for rel in iter_digestable_files(root)}
    body = "".join(f"{digests[rel]}  {rel}\n" for rel in sorted(digests))
    (root / SHA256SUMS_FILENAME).write_text(body, encoding="utf-8")
    return digests


def read_sha256sums(directory: os.PathLike | str) -> Dict[str, str]:
    """Parse ``SHA256SUMS``.  Raises if absent or malformed."""
    path = Path(directory) / SHA256SUMS_FILENAME
    if not path.is_file():
        raise TBEArtifactError(
            "missing_sha256sums",
            f"{path} is absent; a v2 artifact carries a digest over every "
            "other file in the directory",
        )
    out: Dict[str, str] = {}
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1,
    ):
        if not line.strip():
            continue
        digest, sep, rel = line.partition("  ")
        if not sep or len(digest) != 64:
            raise TBEArtifactError(
                "malformed_sha256sums",
                f"{path}:{lineno}: expected '<64 hex>  <path>', got {line!r}",
            )
        out[rel.strip()] = digest.lower()
    return out


def verify_sha256sums(
    directory: os.PathLike | str, *, files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Recompute and compare every digest.  Raises on ANY discrepancy.

    ``files`` restricts the check to a subset (by relative path) -- used to
    verify the metadata of a multi-gigabyte artifact without re-hashing the
    blobs.  A file listed in ``SHA256SUMS`` and missing on disk, a file on
    disk and absent from ``SHA256SUMS``, and a digest that does not match are
    all refusals; none of them is a warning.
    """
    root = Path(directory)
    recorded = read_sha256sums(root)
    on_disk = set(iter_digestable_files(root))

    if files is None:
        missing = sorted(set(recorded) - on_disk)
        extra = sorted(on_disk - set(recorded))
        if missing:
            raise TBEArtifactError(
                "digest_file_missing",
                f"SHA256SUMS lists {len(missing)} file(s) that are not on "
                f"disk: {missing[:8]}",
            )
        if extra:
            raise TBEArtifactError(
                "digest_file_uncovered",
                f"{len(extra)} file(s) in the artifact are not covered by "
                f"SHA256SUMS: {extra[:8]}",
            )
        check = sorted(recorded)
    else:
        check = []
        for rel in files:
            if rel not in recorded:
                raise TBEArtifactError(
                    "digest_file_uncovered",
                    f"{rel!r} is not covered by SHA256SUMS",
                )
            if rel not in on_disk:
                raise TBEArtifactError(
                    "digest_file_missing", f"{rel!r} is not on disk",
                )
            check.append(rel)

    mismatched: List[str] = []
    for rel in check:
        if _sha256_file(root / rel) != recorded[rel]:
            mismatched.append(rel)
    if mismatched:
        raise TBEArtifactError(
            "digest_mismatch",
            f"{len(mismatched)} file(s) do not match their recorded SHA256: "
            f"{mismatched[:8]}",
        )
    return {
        "status": "ok",
        "n_verified": len(check),
        "n_recorded": len(recorded),
        "scope": "all" if files is None else "subset",
    }


# ---------------------------------------------------------------------------
# format detection -- by TAG, never by file presence
# ---------------------------------------------------------------------------
def detect_artifact_format(path: os.PathLike | str) -> Optional[str]:
    """The declared ``artifact_format`` of ``path``, or ``None``.

    ``None`` means "this directory does not claim to be a GeoRefine container"
    -- a plain HF checkpoint, a v1 transcode output, an empty directory and a
    file all return ``None``.  This function NEVER infers a format from which
    files exist: ``model.safetensors`` is shipped by every HF checkpoint on
    the hub, and inferring from it is how a loader claims a model it cannot
    read.  A ``compression_info.json`` that exists but is unparseable, or has
    no ``artifact_format`` string, also returns ``None``: a broken tag is not
    a claim.
    """
    p = Path(path)
    info_path = p / COMPRESSION_INFO_FILENAME
    if not info_path.is_file():
        return None
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
    if not isinstance(info, dict):
        return None
    tag = info.get("artifact_format")
    return tag if isinstance(tag, str) and tag else None


def is_legacy_v1_container(path: os.PathLike | str) -> bool:
    """True for a v1 transcode output: a TBE manifest and no format tag.

    Used to turn "cannot load this" into "this is a v1 container, here is the
    command that upgrades it" rather than a generic refusal.
    """
    p = Path(path)
    if detect_artifact_format(p) is not None:
        return False
    manifest = p / MANIFEST_FILENAME
    if not manifest.is_file():
        return False
    try:
        m = json.loads(manifest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return False
    schema = m.get("schema") if isinstance(m, dict) else None
    return isinstance(schema, str) and schema.startswith(
        "georefine.tbe_transcode_manifest."
    )


# ---------------------------------------------------------------------------
# the artifact handle
# ---------------------------------------------------------------------------
@dataclass
class TBEArtifact:
    """An opened, verified ``georefine.tbe.v2`` directory.

    Holds paths and parsed JSON only -- no tensor is read until
    :func:`iter_artifact_tensors` is called, and then one at a time.
    """

    path: Path
    compression_info: Dict[str, Any]
    manifest: Dict[str, Any]
    digest_receipt: Dict[str, Any] = field(default_factory=dict)

    @property
    def artifact_format(self) -> str:
        return str(self.compression_info["artifact_format"])

    @property
    def layout(self) -> str:
        return str(self.compression_info["layout"])

    @property
    def tbe_blob(self) -> Path:
        return self.path / TBE_BLOB_FILENAME

    @property
    def raw_blob(self) -> Path:
        return self.path / RAW_BLOB_FILENAME

    @property
    def coded_entries(self) -> List[Dict[str, Any]]:
        return [t for t in self.manifest["tensors"] if t["kind"] == "tbe"]

    @property
    def raw_entries(self) -> List[Dict[str, Any]]:
        return [t for t in self.manifest["tensors"] if t["kind"] == "raw"]


def is_tbe_artifact(obj: object) -> bool:
    """Whether *obj* is an opened artifact handle, across duplicate imports.

    ``glc_loader`` is deliberately importable under more than one module name
    in this repository -- as ``glc_loader`` (``release/`` on ``sys.path``), as
    ``release.glc_loader``, and, in
    ``experiments.georefine._glc_release``, as a copy vendored into a staging
    directory, which that module purges from ``sys.modules`` before and after
    every gate run so a later ``import glc_loader`` cannot resolve into a
    directory that is about to be deleted.  Each of those imports builds its
    OWN :class:`TBEArtifact` class object, so a bare ``isinstance`` says False
    for a perfectly good artifact opened either side of such a purge, and the
    caller then hands the handle to ``Path()`` and gets ``TypeError: expected
    str, bytes or os.PathLike``.  Identity of the class is not the contract;
    being this class, from this file, is.
    """
    if isinstance(obj, TBEArtifact):
        return True
    cls = type(obj)
    return (
        cls.__name__ == "TBEArtifact"
        and cls.__module__.rsplit(".", 1)[-1] == "tbe_artifact"
        and all(hasattr(obj, f) for f in TBEArtifact.__dataclass_fields__)
    )


def open_tbe_artifact(
    path: os.PathLike | str | TBEArtifact, *, verify_digests: bool = True,
) -> TBEArtifact:
    """Open a v2 artifact, fail closed on anything that is not one.

    ``verify_digests`` re-hashes the METADATA files (config, tokenizer,
    manifest, compression_info) against ``SHA256SUMS`` on every open; the
    multi-gigabyte tensor blobs are covered by ``SHA256SUMS`` but are not
    re-hashed here, because the per-tensor ``blake2b_source`` check in
    :func:`verify_standalone_bit_exact` is a strictly stronger statement
    about them and a whole-file hash of 50 GB on every load is a tax nobody
    would pay (and would therefore be turned off).  Pass
    ``verify_digests=False`` only in a test.
    """
    if is_tbe_artifact(path):
        # Idempotent: an already-opened artifact is returned unchanged, so a
        # handle that outlived a ``sys.modules`` purge is still loadable.
        return path  # type: ignore[return-value]

    p = Path(path)
    if not p.is_dir():
        raise TBEArtifactError(
            "not_a_directory", f"{p} is not a directory",
        )

    tag = detect_artifact_format(p)
    if tag is None:
        if is_legacy_v1_container(p):
            raise TBEArtifactError(
                "legacy_v1_container",
                f"{p} is a v1 TBE container: it has a transcode manifest but "
                f"no {COMPRESSION_INFO_FILENAME}, so it carries no config, no "
                "tokenizer and no format tag, and it CANNOT be loaded without "
                "the dense checkpoint it was made from. Load it the v1 way "
                "(stream_load_tbe_model / load_tbe_model, both of which take "
                "the dense reference directory), or upgrade it in place-free "
                "form with: python -m scripts.glc_tbe_transcode --upgrade "
                f"{p} --source <dense checkpoint> --out <new v2 dir>",
            )
        raise TBEArtifactError(
            "not_a_tbe_artifact",
            f"{p} does not declare an artifact_format in "
            f"{COMPRESSION_INFO_FILENAME}; it is not a GeoRefine TBE "
            "container. (Format is decided by that tag alone -- the presence "
            "of config.json/model.safetensors is what every Hugging Face "
            "checkpoint looks like and is never treated as a claim.)",
        )
    if tag != ARTIFACT_FORMAT:
        raise TBEArtifactError(
            "unsupported_artifact_format",
            f"{p} declares artifact_format {tag!r}; this loader reads "
            f"{ARTIFACT_FORMAT!r}",
        )

    info = json.loads((p / COMPRESSION_INFO_FILENAME).read_text(encoding="utf-8"))
    version = info.get("compression_schema_version")
    if version != COMPRESSION_SCHEMA_VERSION:
        raise TBEArtifactError(
            "unsupported_compression_schema_version",
            f"{p} declares compression_schema_version {version!r}; this "
            f"loader reads {COMPRESSION_SCHEMA_VERSION}",
        )

    manifest_path = p / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise TBEArtifactError(
            "missing_manifest",
            f"{manifest_path} is absent; the format tag claims a container "
            "but the tensor index is not there",
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ok":
        raise TBEArtifactError(
            "manifest_not_ok",
            f"manifest status is {manifest.get('status')!r}, not 'ok'; a "
            "refused or partial transcode is not loadable",
        )

    if not (p / CONFIG_FILENAME).is_file():
        raise TBEArtifactError(
            "missing_config",
            f"{p / CONFIG_FILENAME} is absent; a standalone load builds the "
            "model skeleton from the container's own config and cannot "
            "proceed without it",
        )
    _assert_tokenizer_present(p)

    receipt: Dict[str, Any] = {"status": "skipped"}
    if verify_digests:
        meta_files = [
            rel for rel in iter_digestable_files(p)
            if rel not in (TBE_BLOB_FILENAME, RAW_BLOB_FILENAME)
        ]
        receipt = verify_sha256sums(p, files=meta_files)

    return TBEArtifact(
        path=p, compression_info=info, manifest=manifest,
        digest_receipt=receipt,
    )


def _assert_tokenizer_present(p: Path) -> None:
    have_config = (p / TOKENIZER_CONFIG_FILE).is_file()
    have_vocab = [f for f in TOKENIZER_VOCAB_FILES if (p / f).is_file()]
    if have_config and have_vocab:
        return
    missing = []
    if not have_config:
        missing.append(TOKENIZER_CONFIG_FILE)
    if not have_vocab:
        missing.append(f"one of {list(TOKENIZER_VOCAB_FILES)}")
    raise TBEArtifactError(
        "missing_tokenizer",
        f"{p} has no usable tokenizer (missing: {missing}). A v2 artifact is "
        "self-contained by definition; an artifact that cannot turn text into "
        "tokens is not releasable, and silently loading without one would "
        "push the failure to first inference instead of to load.",
    )


# ---------------------------------------------------------------------------
# reading tensors OUT OF THE CONTAINER (never out of a dense reference)
# ---------------------------------------------------------------------------
def _container_for_entry(entry: Dict[str, Any], handle) -> TBETensor:
    """Rebuild the stored :class:`TBETensor` for one manifest entry.

    Uses ``coded_shape`` (the 2-D shape the codec actually saw), falling back
    to ``shape`` only for entries written before ``coded_shape`` existed.
    ``decode_tbe`` unpacks ``n, k = c.shape``, so handing it a rank-3
    ``shape`` -- which is what an ``nd_policy=code`` MoE expert stack records
    -- raises rather than decodes.  ``scripts/glc_tbe_certify.py`` and
    ``scripts/glc_tbe_certify_mlx.py`` both build the container from
    ``entry["shape"]``; on a rank-3 coded entry that is a live defect (see
    this module's tests).
    """
    name = entry["name"]
    coded = entry.get("coded_shape") or entry["shape"]
    if len(coded) != 2:
        raise TBEArtifactError(
            "uncodable_entry_shape",
            f"{name}: coded_shape {coded} is not 2-D; the codec only ever "
            "encodes a 2-D view and the manifest entry is inconsistent",
        )
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


def iter_artifact_tensors(
    artifact: TBEArtifact, *, device: str = "cpu",
) -> Iterator[Tuple[str, "Any"]]:
    """Yield ``(name, dense_tensor)`` for EVERY tensor, from the container.

    This is the drop-in replacement for
    ``tbe_stream_loader.iter_safetensors_shards(dense_reference_dir)``: same
    contract, same per-tensor high-water mark, and the source is the artifact
    itself.  Coded tensors are decoded from their stored containers and
    ``view``-ed back to the original (possibly rank-3) shape; raw tensors are
    read from ``raw_tensors.safetensors`` unchanged.

    One tensor is materialised at a time.  A dense checkpoint is never
    opened, and this function has no parameter through which one could be.
    """
    from safetensors import safe_open

    coded = artifact.coded_entries
    raw = artifact.raw_entries

    if coded:
        if not artifact.tbe_blob.is_file():
            raise TBEArtifactError(
                "missing_blob",
                f"the manifest declares {len(coded)} coded tensor(s) but "
                f"{artifact.tbe_blob} is absent",
            )
        with safe_open(str(artifact.tbe_blob), framework="pt", device=device) as h:
            for entry in coded:
                container = _container_for_entry(entry, h)
                dense = decode_tbe(container)
                shape = [int(d) for d in entry["shape"]]
                if list(dense.shape) != shape:
                    dense = dense.reshape(shape)
                yield entry["name"], dense
                del dense, container

    if raw:
        if not artifact.raw_blob.is_file():
            raise TBEArtifactError(
                "missing_blob",
                f"the manifest declares {len(raw)} raw tensor(s) but "
                f"{artifact.raw_blob} is absent",
            )
        with safe_open(str(artifact.raw_blob), framework="pt", device=device) as h:
            stored = set(h.keys())
            for entry in raw:
                name = entry["name"]
                if name not in stored:
                    raise TBEArtifactError(
                        "raw_tensor_missing",
                        f"{name!r} is a raw entry in the manifest but is not "
                        f"in {artifact.raw_blob}",
                    )
                yield name, h.get_tensor(name)


# ---------------------------------------------------------------------------
# the standalone reproduction of the certify harness's bit_exact_weights
# ---------------------------------------------------------------------------
def verify_standalone_bit_exact(
    artifact: TBEArtifact, *, limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Decode every stored container and check it against the manifest hash.

    WHAT THIS IS AND IS NOT.  ``scripts/glc_tbe_certify.certify_bit_exact_weights``
    decodes each stored container and compares the bytes to the DENSE
    CHECKPOINT, re-read from disk.  That comparison is unavailable to a
    released artifact by construction -- the dense checkpoint is exactly what
    a release does not ship.  This function makes the same statement against
    the ``blake2b_source`` digest the transcoder recorded for that tensor at
    the moment it read it from the dense checkpoint, which is bound to the
    artifact by ``SHA256SUMS`` (over ``manifest.json``) and verified on
    ``open_tbe_artifact``.

    So the chain is: dense bytes -> blake2b at transcode time -> manifest ->
    sha256 in SHA256SUMS -> this check.  It reproduces the certify stage's
    verdict for a consumer who never had the dense checkpoint, and it is
    NOT the same measurement as the certify stage: it is one digest link
    longer, and it cannot detect a transcoder that recorded a hash of
    something other than what it read.  It CAN detect every post-transcode
    corruption of the stored blobs, which is what a released artifact's
    consumer actually needs to detect.

    Raw tensors are checked too, against their own ``blake2b_source``: a v1
    certify never looked at them at all.
    """
    import torch
    from safetensors import safe_open

    from .tbe_container import decode_tbe as _decode

    coded = artifact.coded_entries
    if not coded:
        return {
            "schema": TBE_STANDALONE_BITEXACT_SCHEMA,
            "status": "refused",
            "reason": "no_coded_tensors",
            "detail": "the manifest coded zero tensors; nothing to certify",
        }
    if not artifact.tbe_blob.is_file():
        return {
            "schema": TBE_STANDALONE_BITEXACT_SCHEMA,
            "status": "refused",
            "reason": "missing_blob",
            "detail": f"{artifact.tbe_blob} is absent",
        }

    entries = coded[:limit] if limit else coded
    n_checked = 0
    n_mismatched = 0
    n_no_recorded_hash = 0
    mismatched: List[str] = []
    no_hash: List[str] = []

    with safe_open(str(artifact.tbe_blob), framework="pt", device="cpu") as h:
        for entry in entries:
            name = entry["name"]
            recorded = entry.get("blake2b_source")
            if not recorded:
                n_no_recorded_hash += 1
                no_hash.append(name)
                continue
            container = _container_for_entry(entry, h)
            decoded = _decode(container)
            got = _blake2b_tensor(decoded)
            n_checked += 1
            if got != recorded:
                n_mismatched += 1
                mismatched.append(name)
            del decoded, container

    n_raw_checked = 0
    n_raw_mismatched = 0
    raw_mismatched: List[str] = []
    raw = artifact.raw_entries
    if raw and artifact.raw_blob.is_file() and limit is None:
        with safe_open(str(artifact.raw_blob), framework="pt", device="cpu") as h:
            keys = set(h.keys())
            for entry in raw:
                name = entry["name"]
                recorded = entry.get("blake2b_source")
                if not recorded or name not in keys:
                    continue
                got = _blake2b_tensor(h.get_tensor(name))
                n_raw_checked += 1
                if got != recorded:
                    n_raw_mismatched += 1
                    raw_mismatched.append(name)

    if n_no_recorded_hash:
        return {
            "schema": TBE_STANDALONE_BITEXACT_SCHEMA,
            "status": "refused",
            "reason": "manifest_entry_without_source_hash",
            "detail": (
                f"{n_no_recorded_hash} coded entry(ies) carry no "
                f"blake2b_source, so their bytes cannot be checked without "
                f"the dense checkpoint: {no_hash[:8]}"
            ),
        }

    ok = n_mismatched == 0 and n_raw_mismatched == 0
    return {
        "schema": TBE_STANDALONE_BITEXACT_SCHEMA,
        "status": "ok" if ok else "failed",
        "n_checked": n_checked,
        "n_mismatched": n_mismatched,
        "mismatched_tensors": mismatched[:32],
        "all_bit_exact": ok,
        "n_raw_checked": n_raw_checked,
        "n_raw_mismatched": n_raw_mismatched,
        "raw_mismatched_tensors": raw_mismatched[:32],
        "reference": "manifest blake2b_source (no dense checkpoint read)",
    }


def _blake2b_tensor(t) -> str:
    """Byte-identical to ``scripts/glc_tbe_transcode._blake2b_hex``.

    Reimplemented here rather than imported: this module must work from an
    artifact directory with no build repository present, which is the whole
    point of the format, and ``scripts/`` is part of the build repository.
    The two are pinned together by a test.
    """
    import torch

    x = t.detach().contiguous().cpu()
    if x.dtype == torch.bfloat16:
        x = x.view(torch.int16)
    return hashlib.blake2b(x.numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# skeleton + tokenizer, from the CONTAINER's own files
# ---------------------------------------------------------------------------
def build_meta_skeleton_from_artifact(artifact: TBEArtifact, *, dtype=None):
    """A parameter-free skeleton on ``meta``, built from the artifact.

    Identical in shape to ``tbe_stream_loader.build_meta_skeleton`` except
    that the config comes from the CONTAINER, not from a dense reference
    directory.  Allocates no storage, so a 360 GB architecture costs nothing.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    if dtype is None:
        dtype = torch.bfloat16
    config = AutoConfig.from_pretrained(str(artifact.path))
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    model.eval()
    return model


def load_tokenizer_from_artifact(artifact: TBEArtifact):
    """The tokenizer shipped inside the artifact."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(artifact.path))


# ---------------------------------------------------------------------------
# the entry point named in compression_info.json
# ---------------------------------------------------------------------------
def load_standalone(
    path: os.PathLike | str,
    *,
    device_map: Any = None,
    dtype=None,
    verify_digests: bool = True,
    with_tokenizer: bool = True,
    **kwargs,
):
    """Load a v2 artifact with NO dense checkpoint anywhere.

    Returns ``(model, tokenizer_or_None, receipt)``.  Delegates the actual
    placement to ``tbe_stream_loader.stream_load_tbe_model_standalone`` --
    imported lazily so that opening, verifying and reading an artifact does
    not drag in the serving stack.
    """
    from .tbe_stream_loader import stream_load_tbe_model_standalone

    artifact = open_tbe_artifact(path, verify_digests=verify_digests)
    model, receipt = stream_load_tbe_model_standalone(
        artifact, device_map, dtype=dtype, **kwargs
    )
    tokenizer = load_tokenizer_from_artifact(artifact) if with_tokenizer else None
    return model, tokenizer, receipt


__all__ = [
    "ARTIFACT_FORMAT",
    "COMPRESSION_INFO_FILENAME",
    "COMPRESSION_INFO_SCHEMA",
    "COMPRESSION_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "RAW_BLOB_FILENAME",
    "SHA256SUMS_FILENAME",
    "SIDECAR_CANDIDATES",
    "STANDALONE_LOADER",
    "TBE_BLOB_FILENAME",
    "TBE_STANDALONE_BITEXACT_SCHEMA",
    "TBE_STANDALONE_RECEIPT_SCHEMA",
    "TBEArtifact",
    "TBEArtifactError",
    "build_meta_skeleton_from_artifact",
    "detect_artifact_format",
    "is_legacy_v1_container",
    "iter_artifact_tensors",
    "iter_digestable_files",
    "load_standalone",
    "load_tokenizer_from_artifact",
    "open_tbe_artifact",
    "read_sha256sums",
    "verify_sha256sums",
    "verify_standalone_bit_exact",
    "write_sha256sums",
]
