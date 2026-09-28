"""Reading, verifying and expanding a GLC-RELEASE artifact directory.

Everything here runs with no build repository and no source checkpoint on the
machine.  The only third-party imports are ``torch``, ``safetensors`` and --
for :func:`load_model` alone -- ``transformers``.

LAYOUT
------
    <artifact>/
        MANIFEST.json        provenance, effective config, file inventory
        CERTIFICATE.json     gate verdicts, each explicit; see the README
        README.md
        glc_loader/          this package, vendored
        model/
            config.json, tokenizer*, generation_config.json, ...
            glc_index.json   per-tensor container parameters and digests
            glc-*.safetensors
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from .container import (
    FWP1Tensor,
    decode_fwp1,
    encode_fwp1,
    fwp1_certify,
)

FORMAT = "GLC-RELEASE/1"
MANIFEST_NAME = "MANIFEST.json"
CERTIFICATE_NAME = "CERTIFICATE.json"
INDEX_NAME = "glc_index.json"
MODEL_DIR = "model"
PLANE_SEP = "::"
PLANES = ("smb", "eidx", "pal")


class GLCArtifactError(RuntimeError):
    """The artifact is malformed, incomplete, or fails verification."""


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise GLCArtifactError(f"missing required file: {path}")
    with path.open("rb") as fh:
        raw = fh.read()
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise GLCArtifactError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise GLCArtifactError(f"{path} must contain a JSON object")
    return obj


def _sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sha256_tensor(t: torch.Tensor) -> str:
    a = t.detach().to("cpu").contiguous()
    return hashlib.sha256(
        a.view(torch.uint8).numpy().tobytes() if a.dtype != torch.uint8
        else a.numpy().tobytes()
    ).hexdigest()


@dataclass
class Artifact:
    root: Path
    manifest: Dict[str, Any]
    certificate: Dict[str, Any]
    index: Dict[str, Any]

    @property
    def model_dir(self) -> Path:
        return self.root / MODEL_DIR

    @property
    def tensors(self) -> Dict[str, Any]:
        return self.index["tensors"]

    def entry(self, name: str) -> Dict[str, Any]:
        try:
            return self.tensors[name]
        except KeyError as exc:
            raise GLCArtifactError(f"tensor not in index: {name}") from exc

    def summary(self) -> Dict[str, Any]:
        return {
            "format": self.manifest.get("format"),
            "container": self.manifest.get("container", {}).get("name"),
            "source_model": self.manifest.get("provenance", {}).get("source_model"),
            "built_at": self.manifest.get("provenance", {}).get("built_at_utc"),
            "builder_git_sha": self.manifest.get("provenance", {}).get(
                "builder_git_sha"
            ),
            "tensors_total": len(self.tensors),
            "tensors_coded": sum(
                1 for v in self.tensors.values() if v["kind"] == "fwp1"
            ),
            "weight_bytes_dense": self.manifest["accounting"]["weight_bytes_dense"],
            "weight_bytes_resident": self.manifest["accounting"][
                "weight_bytes_resident"
            ],
            "weight_ratio": self.manifest["accounting"]["weight_ratio"],
            "whole_model_ratio": self.manifest["accounting"]["whole_model_ratio"],
        }


def open_artifact(
    path: os.PathLike | str, *, require_certificate: bool = True,
) -> Artifact:
    """Open an artifact directory.

    ``require_certificate=False`` exists for exactly one caller: the builder,
    which must load the staged artifact to *measure* the gates that produce the
    certificate.  It is not a client-facing escape hatch -- the CLI exposes it
    only behind ``--no-certificate``, and the default everywhere is to require
    a complete, passing certificate.
    """
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise GLCArtifactError(f"not a directory: {root}")
    manifest = _read_json(root / MANIFEST_NAME)
    if manifest.get("format") != FORMAT:
        raise GLCArtifactError(
            f"unsupported artifact format {manifest.get('format')!r}; "
            f"this loader reads {FORMAT}"
        )
    cert_path = root / CERTIFICATE_NAME
    if require_certificate or cert_path.is_file():
        certificate = _read_json(cert_path)
    else:
        certificate = {}
    index = _read_json(root / MODEL_DIR / INDEX_NAME)
    return Artifact(root=root, manifest=manifest, certificate=certificate,
                    index=index)


# ---------------------------------------------------------------------------
# tensor access
# ---------------------------------------------------------------------------
class _ShardReader:
    """Lazy safetensors readers, one per shard, opened on first use."""

    def __init__(self, model_dir: Path, framework: str = "pt") -> None:
        from safetensors import safe_open

        self._safe_open = safe_open
        self._model_dir = model_dir
        self._open: Dict[str, Any] = {}
        self._framework = framework

    def get(self, shard: str, key: str, device: str = "cpu") -> torch.Tensor:
        fh = self._open.get(shard)
        if fh is None:
            p = self._model_dir / shard
            if not p.is_file():
                raise GLCArtifactError(f"missing shard: {p}")
            fh = self._safe_open(str(p), framework=self._framework, device="cpu")
            fh.__enter__()
            self._open[shard] = fh
        try:
            t = fh.get_tensor(key)
        except Exception as exc:
            raise GLCArtifactError(
                f"shard {shard} does not carry tensor {key!r}: {exc}"
            ) from exc
        return t if device == "cpu" else t.to(device)

    def header_shape(self, shard: str, key: str) -> Tuple[int, ...]:
        fh = self._open.get(shard)
        if fh is None:
            self.get(shard, key)
            fh = self._open[shard]
        return tuple(fh.get_slice(key).get_shape())

    def close(self) -> None:
        for fh in self._open.values():
            try:
                fh.__exit__(None, None, None)
            except Exception:
                pass
        self._open.clear()


def _container_from_entry(
    reader: _ShardReader, name: str, entry: Dict[str, Any], device: str,
) -> FWP1Tensor:
    planes = entry["planes"]
    shard = entry["shard"]
    tensors = {
        p: reader.get(shard, f"{name}{PLANE_SEP}{p}", device=device)
        for p in PLANES
    }
    shape = tuple(int(x) for x in entry["shape"])
    group = int(entry["group"])
    n, k = shape
    expect = {
        "smb": (n, k),
        "eidx": (n, k // 2),
        "pal": (n * (k // group), 16),
    }
    for p in PLANES:
        got = tuple(int(x) for x in tensors[p].shape)
        if got != expect[p]:
            raise GLCArtifactError(
                f"{name}: plane {p} has shape {got}, the header of a "
                f"{shape} group={group} container requires {expect[p]}. "
                "Refusing to build a model from planes that do not describe "
                "the declared tensor."
            )
        recorded = [int(x) for x in planes.get(p, [])]
        if recorded != [int(x) for x in got]:
            # A disagreement between the index and the safetensors header is a
            # corrupted artifact, not a recoverable condition.
            raise GLCArtifactError(
                f"{name}: index records plane {p} as {recorded} but the "
                f"safetensors header says {list(got)}"
            )
    return FWP1Tensor(
        smb=tensors["smb"], eidx=tensors["eidx"], pal=tensors["pal"],
        shape=(n, k), group=group,
    )


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def verify(
    path: os.PathLike | str,
    *,
    deep: bool = False,
    device: str = "cpu",
    progress: bool = False,
) -> Dict[str, Any]:
    """Verify the artifact against its own recorded digests.

    Shallow (default): every shard file's sha256 matches the manifest inventory,
    and every indexed tensor is present with the declared shape.

    ``deep=True``: additionally decodes every coded tensor and checks the
    sha256 of the decoded bf16 bytes against the digest the builder recorded
    for the *source* tensor.  Because that digest was taken from the original
    checkpoint before encoding, a deep pass is an end-to-end proof that the
    bytes this artifact produces are the bytes the source model had --
    reproducible on the client's machine, with no source checkpoint present.
    It also re-encodes each decoded tensor and requires byte-identical planes,
    so a client never has to take the encoder on trust either.
    """
    art = open_artifact(path)
    inventory = art.manifest["inventory"]
    # An empty or partial inventory must not read as a clean verification.
    # This is the single highest-consequence shape in this codebase: a
    # provenance check elsewhere in the tree iterates two candidate fingerprint
    # filenames, ``continue``s when neither exists, falls through to
    # ``return True``, and drives an exit code of 0 -- a release that passed
    # provenance validation having hashed nothing. The loop below would have
    # done exactly the same thing on an inventory of ``{}``. So the covering
    # relation is asserted before the loop, not inferred from it completing.
    if not isinstance(inventory, dict) or not inventory:
        raise GLCArtifactError(
            "the manifest carries no file inventory; there is nothing to "
            "verify against and an unverifiable artifact is not a verified one"
        )
    present = {
        p.relative_to(art.root).as_posix()
        for p in art.root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    present.discard(MANIFEST_NAME)
    unlisted = sorted(present - set(inventory))
    if unlisted:
        raise GLCArtifactError(
            "these files are in the artifact but not in the manifest "
            f"inventory, so nothing checks them: {unlisted[:12]}"
        )
    for shard in art.index.get("shards") or []:
        if f"{MODEL_DIR}/{shard}" not in inventory:
            raise GLCArtifactError(
                f"the index names shard {shard!r} but the inventory does not "
                "cover it; a shard nothing hashes can be swapped undetected"
            )
    files_checked: List[str] = []
    for rel, expected in sorted(inventory.items()):
        p = art.root / rel
        if not p.is_file():
            raise GLCArtifactError(f"inventory lists a missing file: {rel}")
        got = _sha256_file(p)
        if got != expected["sha256"]:
            raise GLCArtifactError(
                f"{rel}: sha256 {got} != recorded {expected['sha256']}"
            )
        if int(p.stat().st_size) != int(expected["bytes"]):
            raise GLCArtifactError(f"{rel}: size differs from the inventory")
        files_checked.append(rel)

    reader = _ShardReader(art.model_dir)
    coded = 0
    raw = 0
    elements_checked = 0
    mismatched = 0
    reencode_checked = 0
    try:
        for name, entry in sorted(art.tensors.items()):
            if entry["kind"] == "raw":
                t = reader.get(entry["shard"], name, device="cpu")
                if tuple(int(x) for x in t.shape) != tuple(
                    int(x) for x in entry["shape"]
                ):
                    raise GLCArtifactError(
                        f"{name}: raw tensor shape {tuple(t.shape)} != index "
                        f"{tuple(entry['shape'])}"
                    )
                if deep:
                    got = _sha256_tensor(t)
                    if got != entry["sha256_source"]:
                        raise GLCArtifactError(
                            f"{name}: raw bytes sha256 {got} != source digest "
                            f"{entry['sha256_source']}"
                        )
                raw += 1
                continue
            c = _container_from_entry(reader, name, entry, device)
            coded += 1
            if not deep:
                continue
            if not entry.get("sha256_source"):
                raise GLCArtifactError(
                    f"{name}: the index carries no source digest, so a deep "
                    "verify of this tensor would compare against nothing. "
                    "Refusing to report a pass for a check that cannot run."
                )
            dec = decode_fwp1(c)
            got = _sha256_tensor(dec)
            if got != entry["sha256_source"]:
                mismatched += 1
                raise GLCArtifactError(
                    f"{name}: decoded bytes sha256 {got} != the digest taken "
                    f"from the source checkpoint {entry['sha256_source']}. "
                    "This artifact does not reproduce the source model."
                )
            elements_checked += int(dec.numel())
            re = encode_fwp1(dec, group=int(entry["group"]))
            for p, a, b in (
                ("smb", re.smb, c.smb),
                ("eidx", re.eidx, c.eidx),
                ("pal", re.pal, c.pal),
            ):
                if not torch.equal(a.cpu(), b.cpu()):
                    raise GLCArtifactError(
                        f"{name}: re-encoding the decoded tensor does not "
                        f"reproduce the shipped {p} plane"
                    )
            reencode_checked += 1
            if progress:
                print(f"  verified {name}", flush=True)
    finally:
        reader.close()

    if coded == 0:
        raise GLCArtifactError(
            "no coded tensor was verified. Either the index has none -- in "
            "which case this is not a compressed artifact -- or the loop "
            "skipped them all. Both are failures; neither is a pass."
        )
    if len(files_checked) != len(inventory):
        raise GLCArtifactError(
            f"checked {len(files_checked)} files but the inventory lists "
            f"{len(inventory)}; a verification that did not cover its own "
            "inventory has not verified the artifact"
        )
    if deep and elements_checked == 0:
        raise GLCArtifactError(
            "a deep verify decoded zero elements; the depth is a claim about "
            "work done and no work was done"
        )
    return {
        "status": "OK",
        "artifact": str(art.root),
        "depth": "deep" if deep else "shallow",
        "files_checked": len(files_checked),
        "inventory_entries": len(inventory),
        "tensors_coded": coded,
        "tensors_raw": raw,
        "elements_decoded_and_matched": int(elements_checked),
        "tensors_reencoded_byte_identical": int(reencode_checked),
        "mismatched_tensors": int(mismatched),
        "source_digest_source": (
            "sha256 of the source checkpoint tensor bytes, recorded at build"
        ),
    }


# ---------------------------------------------------------------------------
# expansion -- the escape hatch to any other engine
# ---------------------------------------------------------------------------
def expand(
    path: os.PathLike | str,
    out_dir: os.PathLike | str,
    *,
    max_shard_bytes: int = 4 * (1 << 30),
    progress: bool = False,
) -> Dict[str, Any]:
    """Write a plain, uncompressed HuggingFace checkpoint.

    This is the deliberate exit from our runtime.  A client who wants
    llama.cpp, LM Studio, vLLM, TensorRT-LLM or anything else runs this once
    and gets an ordinary ``model.safetensors`` directory that no tool in the
    world needs to know about GLC to read.  Compression is then a property of
    how they received the model, not a dependency of how they run it.
    """
    from safetensors.torch import save_file

    art = open_artifact(path)
    out = Path(out_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise GLCArtifactError(f"refusing to write into a non-empty directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    reader = _ShardReader(art.model_dir)
    shards: List[Dict[str, torch.Tensor]] = [{}]
    sizes = [0]
    order: List[Tuple[str, int]] = []
    tie = bool(art.manifest["structure"].get("tie_word_embeddings", False))
    tied_names = set(art.manifest["structure"].get("tied_tensor_names", []))
    try:
        for name, entry in art.tensors.items():
            if name in tied_names:
                continue
            if entry["kind"] == "raw":
                t = reader.get(entry["shard"], name, device="cpu")
            else:
                t = decode_fwp1(_container_from_entry(reader, name, entry, "cpu"))
            nbytes = t.numel() * t.element_size()
            if sizes[-1] and sizes[-1] + nbytes > max_shard_bytes:
                shards.append({})
                sizes.append(0)
            shards[-1][name] = t
            sizes[-1] += nbytes
            order.append((name, len(shards) - 1))
            if progress:
                print(f"  expanded {name}", flush=True)
    finally:
        reader.close()

    n = len(shards)
    names = [
        f"model-{i + 1:05d}-of-{n:05d}.safetensors" if n > 1
        else "model.safetensors"
        for i in range(n)
    ]
    for i, sh in enumerate(shards):
        save_file(sh, str(out / names[i]), metadata={"format": "pt"})
    if n > 1:
        weight_map = {nm: names[i] for nm, i in order}
        (out / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": int(sum(sizes))},
                    "weight_map": weight_map,
                },
                indent=2, sort_keys=True,
            ),
            encoding="utf-8",
        )
    for rel in art.manifest["structure"]["runtime_files"]:
        src = art.model_dir / rel
        if src.is_file():
            (out / rel).write_bytes(src.read_bytes())
    return {
        "status": "OK",
        "out_dir": str(out),
        "shards": names,
        "tensors_written": len(order),
        "tied_tensors_omitted": sorted(tied_names) if tie else [],
        "total_bytes": int(sum(sizes)),
        "note": (
            "an ordinary HuggingFace checkpoint; no GLC code is needed to read it"
        ),
    }


__all__ = [
    "Artifact",
    "CERTIFICATE_NAME",
    "FORMAT",
    "GLCArtifactError",
    "INDEX_NAME",
    "MANIFEST_NAME",
    "MODEL_DIR",
    "PLANES",
    "PLANE_SEP",
    "_ShardReader",
    "_container_from_entry",
    "_sha256_file",
    "_sha256_tensor",
    "expand",
    "open_artifact",
    "verify",
]
