"""Swap ``mlx-lm`` ``nn.Linear`` modules for ``TBELinearMLX`` per a GLC-TBE manifest.

This is the model-level glue the certify script and the tests need: given a
dense ``mlx-lm`` model object (as returned by ``mlx_lm.load``) and a GLC-TBE
transcode manifest (``scripts/glc_tbe_transcode.py``'s output, produced with
``layout='flat64'`` -- see the module docstring in ``tbe_decode_mlx.py`` for
why flat64 rather than the CUDA-default ``mma16``), replace every ``nn.Linear``
the manifest coded with a ``TBELinearMLX`` holding the container's bytes as
resident MLX arrays.  Every ``nn.Linear`` the manifest left raw is untouched
-- it stays exactly the bf16 (or other) array ``mlx_lm.load`` already put
there.  Nothing here re-derives the container format; it is read verbatim
via ``release.glc_loader.tbe_container.TBETensor`` and handed to
``upload_tbe_mlx``, the same two calls ``scripts/glc_tbe_certify.py``'s
``certify_bit_exact_weights`` stage makes for its own CPU-side decode.

FAIL CLOSED ON NAME MISMATCH, BOTH DIRECTIONS.  A coded tensor named in the
manifest with no matching ``nn.Linear`` in the live model (manifest built
from a different checkpoint, or this model's parameter names drifted) is a
``TBESwapError``, not a silently-skipped tensor.  Symmetrically, an
``nn.Linear`` in the live model with NO entry at all in the manifest --
neither coded nor raw -- is also a ``TBESwapError``: a raw tensor's OWN
entry is expected and fine (embeddings, small layers, escape-band raw), but
a linear the manifest never saw at all means the manifest was not built
from this model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..tbe_container import TBETensor
from .tbe_decode_mlx import TBEDeviceMLX, upload_tbe_mlx
from .tbe_linear_mlx import TBELinearMLX
from .tbe_linear_mlx_v4 import TBELinearMLXV4

#: Decoder implementations a swap can be built on.  ``v1`` is the shipped
#: default and the one every existing receipt was measured through; ``v4`` is
#: the batchable phase-4 dispatch (same Metal kernel as v3 ``t4_g4``, no
#: per-layer ``mx.eval``) -- see
#: ``docs/research/TBE_METAL_ACCESS_PATTERN_20260903.md``.  Opt-in, so no
#: certified receipt changes meaning without a flag being passed.
DECODERS = {"v1": TBELinearMLX, "v4": TBELinearMLXV4}


class TBESwapError(RuntimeError):
    """The manifest and the live model disagree about what tensors exist.

    Never raised for an ordinary raw tensor -- that is the expected outcome
    for embeddings, small layers and anything the escape band pushed over.
    Only raised when reconciliation itself fails: a name present on one side
    and absent on the other, a layout the MLX kernel cannot decode, or a
    manifest that does not look like a GLC-TBE transcode at all.
    """


@dataclass
class TBESwapReceipt:
    """What ``swap_model_to_tbe`` actually did, in bytes and counts."""

    n_swapped: int
    n_raw_linear: int
    coded_resident_bytes: int
    raw_bf16_bytes: int
    dense_equivalent_bytes: int
    swapped_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        served_bytes = self.coded_resident_bytes + self.raw_bf16_bytes
        return {
            "n_swapped": self.n_swapped,
            "n_raw_linear": self.n_raw_linear,
            "coded_resident_bytes": self.coded_resident_bytes,
            "raw_bf16_bytes": self.raw_bf16_bytes,
            "dense_equivalent_bytes": self.dense_equivalent_bytes,
            "served_resident_bytes": served_bytes,
            "ratio_resident_vs_dense": (
                self.dense_equivalent_bytes / served_bytes if served_bytes else 1.0
            ),
            "swapped_names": self.swapped_names,
        }


# ---------------------------------------------------------------------------
# tree walk -- mlx.nn.Module children can be a Module, a list of Modules, or
# a dict of Modules (mlx-lm stores transformer blocks as a plain python list
# under `.layers`); this walker covers all three and returns the exact
# (parent, key) pair needed to mutate the tree in place.
# ---------------------------------------------------------------------------
def _collect_linears(node, prefix, parent, key, out) -> None:
    if isinstance(node, nn.Linear):
        out.append((prefix, parent, key, node))
        return
    if isinstance(node, nn.Module):
        for name, child in node.children().items():
            _collect_linears(
                child, f"{prefix}.{name}" if prefix else name, node, name, out,
            )
    elif isinstance(node, (list, tuple)):
        for i, child in enumerate(node):
            _collect_linears(
                child, f"{prefix}.{i}" if prefix else str(i), node, i, out,
            )
    elif isinstance(node, dict):
        for k, child in node.items():
            _collect_linears(
                child, f"{prefix}.{k}" if prefix else str(k), node, k, out,
            )


def find_linears(model: nn.Module) -> List[Tuple[str, Any, Any, nn.Linear]]:
    """Every ``(dotted_path, parent, key, nn.Linear)`` in ``model``.

    ``dotted_path`` is built to match the checkpoint's own tensor names
    (list indices rendered as plain integers, e.g.
    ``model.layers.0.self_attn.q_proj``) -- the same convention
    ``scripts/glc_tbe_transcode.py`` reads tensor names in, verified against
    a live ``mlx_lm`` Qwen3 model before this module was written.
    """
    out: List[Tuple[str, Any, Any, nn.Linear]] = []
    for name, child in model.children().items():
        _collect_linears(child, name, model, name, out)
    return out


def _set_child(parent, key, value) -> None:
    if isinstance(key, int):
        parent[key] = value
    else:
        setattr(parent, key, value)


# ---------------------------------------------------------------------------
# manifest plumbing
# ---------------------------------------------------------------------------
def load_manifest(transcode_dir: "os.PathLike | str") -> Dict[str, Any]:
    """Load a ``glc_tbe_transcode`` manifest.  Fails closed on a bad status."""
    path = Path(transcode_dir) / "manifest.json"
    if not path.is_file():
        raise TBESwapError(f"no manifest.json under {transcode_dir}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ok":
        raise TBESwapError(
            f"manifest status is {manifest.get('status')!r}, not 'ok'; a "
            "refused or partial transcode cannot be swapped in"
        )
    return manifest


# ---------------------------------------------------------------------------
# the swap
# ---------------------------------------------------------------------------
def swap_model_to_tbe(
    model: nn.Module,
    manifest: Dict[str, Any],
    blob_path: "os.PathLike | str",
    *,
    model_name: Optional[str] = None,
    decoder: str = "v1",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Replace every coded ``nn.Linear`` in ``model`` with a ``TBELinearMLX``.

    Mutates ``model`` in place (the tree is walked by object reference, not
    copied) and also returns it, so callers can use either
    ``model, receipt = swap_model_to_tbe(model, ...)`` or discard the return
    and keep using their own ``model`` handle.
    """
    try:
        linear_cls = DECODERS[decoder]
    except KeyError:
        raise TBESwapError(
            f"unknown decoder {decoder!r}; expected one of {sorted(DECODERS)}"
        ) from None

    if "schema" not in manifest:
        raise TBESwapError(
            "manifest has no 'schema' field; not a GLC-TBE transcode manifest"
        )
    if model_name is not None:
        recorded = str(manifest.get("source", {}).get("model_dir", ""))
        if recorded and model_name not in recorded and recorded not in model_name:
            raise TBESwapError(
                f"manifest/model name mismatch: manifest was built from "
                f"{recorded!r}, swap requested for {model_name!r}"
            )

    tensors = manifest.get("tensors", [])
    by_name: Dict[str, Dict[str, Any]] = {t["name"]: t for t in tensors}
    tbe_names = {n for n, t in by_name.items() if t["kind"] == "tbe"}

    linears = find_linears(model)
    linear_weight_names = {f"{path}.weight" for path, *_ in linears}

    # direction 1: every coded tensor must land on an actual nn.Linear here
    missing_in_model = sorted(tbe_names - linear_weight_names)
    if missing_in_model:
        raise TBESwapError(
            f"{len(missing_in_model)} coded tensor(s) in the manifest have no "
            f"matching nn.Linear in this model (first 8): {missing_in_model[:8]}"
        )

    # direction 2: every nn.Linear here must be accounted for, coded or raw
    # -- an unlisted linear means the manifest was built from another model
    unaccounted = sorted(n for n in linear_weight_names if n not in by_name)
    if unaccounted:
        raise TBESwapError(
            f"{len(unaccounted)} nn.Linear weight(s) in this model have no "
            f"entry at all in the manifest (first 8): {unaccounted[:8]}"
        )

    from safetensors import safe_open

    n_swapped = 0
    n_raw = 0
    coded_resident_bytes = 0
    raw_bf16_bytes = 0
    dense_equivalent_bytes = 0
    swapped_names: List[str] = []

    with safe_open(str(blob_path), framework="pt", device="cpu") as handle:
        for path, parent, key, lin in linears:
            wname = f"{path}.weight"
            entry = by_name[wname]
            out_f, in_f = int(lin.weight.shape[0]), int(lin.weight.shape[1])
            dense_equivalent_bytes += out_f * in_f * 2

            if entry["kind"] != "tbe":
                n_raw += 1
                raw_bf16_bytes += out_f * in_f * 2
                continue

            if entry["layout"] != "flat64":
                raise TBESwapError(
                    f"{wname}: container layout is {entry['layout']!r}; the "
                    "MLX decode kernel (tbe_decode_mlx.py) requires 'flat64'"
                )

            container = TBETensor(
                shape=tuple(entry["shape"]), layout=entry["layout"],
                mode=int(entry["mode"]), base=int(entry["base"]),
                tiles=int(entry["tiles"]), escapes=int(entry["escapes"]),
                planes=handle.get_tensor(f"{wname}.planes"),
                smb=handle.get_tensor(f"{wname}.smb"),
                esc=handle.get_tensor(f"{wname}.esc"),
                sbbase=handle.get_tensor(f"{wname}.sbbase"),
                superblock=int(entry["superblock"]),
            )
            device_container: TBEDeviceMLX = upload_tbe_mlx(container)
            bias = getattr(lin, "bias", None)
            replacement = linear_cls(device_container, bias=bias)
            _set_child(parent, key, replacement)

            n_swapped += 1
            coded_resident_bytes += device_container.resident_bytes
            swapped_names.append(wname)

    receipt = TBESwapReceipt(
        n_swapped=n_swapped, n_raw_linear=n_raw,
        coded_resident_bytes=coded_resident_bytes,
        raw_bf16_bytes=raw_bf16_bytes,
        dense_equivalent_bytes=dense_equivalent_bytes,
        swapped_names=swapped_names,
    )
    return model, receipt.to_dict()


def load_tbe_model(
    model_path: "os.PathLike | str",
    transcode_dir: "os.PathLike | str",
    *,
    model_name: Optional[str] = None,
    decoder: str = "v1",
):
    """``mlx_lm.load`` a dense checkpoint, then swap it to TBE in place.

    Returns ``(model, tokenizer, swap_receipt)``.  Two full weight sets are
    never resident at once beyond the ordinary lifetime of ``mlx_lm.load``'s
    own dense arrays plus this call's per-tensor decode scratch -- the swap
    replaces ``nn.Linear`` objects one at a time and the freed dense weight
    arrays are reclaimed by MLX's caching allocator the same way any other
    freed array is.
    """
    import mlx_lm

    manifest = load_manifest(transcode_dir)
    model, tokenizer = mlx_lm.load(str(model_path))
    blob_path = Path(transcode_dir) / "tbe_tensors.safetensors"
    model, receipt = swap_model_to_tbe(
        model, manifest, blob_path, model_name=model_name or str(model_path),
        decoder=decoder,
    )
    return model, tokenizer, receipt


def _artifact_module():
    """``glc_loader.tbe_artifact``, imported without a package-name clash.

    Relative import: this module already lives inside the package, and the
    artifact module deliberately imports nothing from ``scripts/`` so it
    works from an artifact directory with no build repo present.
    """
    from .. import tbe_artifact

    return tbe_artifact


def swap_model_to_tbe_from_artifact(
    model: nn.Module,
    artifact,
    *,
    decoder: str = "v1",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """:func:`swap_model_to_tbe` against an opened v2 artifact.

    Thin adapter: the manifest and the blob path both come from the artifact,
    and ``model_name`` checking is skipped because the artifact IS the source
    of the manifest -- there is no second directory whose name could disagree.
    """
    return swap_model_to_tbe(
        model, artifact.manifest, artifact.tbe_blob, decoder=decoder,
    )


def load_tbe_model_standalone(
    artifact_path: "os.PathLike | str",
    *,
    decoder: str = "v1",
    verify_digests: bool = True,
    strict: bool = True,
):
    """Load a ``georefine.tbe.v2`` artifact under MLX with NO dense checkpoint.

    :func:`load_tbe_model` -- the v1 path above -- calls
    ``mlx_lm.load(dense_path)`` and only THEN swaps linears, so it cannot run
    without the checkpoint the container was made from, and the artifact
    directory alone is unusable.  This function never touches a dense
    checkpoint: the model class and its args come from the ARTIFACT's own
    ``config.json``, the raw (uncoded) tensors come from the artifact's
    ``raw_tensors.safetensors``, the coded ones from its
    ``tbe_tensors.safetensors``, and the tokenizer from the artifact
    directory.  It has no parameter through which a dense path could be
    supplied.

    Returns ``(model, tokenizer, receipt)``, the same triple
    :func:`load_tbe_model` returns.

    TWO FULL WEIGHT SETS ARE NEVER RESIDENT.  ``model_class(model_args)``
    constructs MLX arrays for every ``nn.Linear``, but MLX arrays are lazy:
    nothing is materialised until an ``mx.eval``, and every coded linear's
    freshly-constructed array is REPLACED by its ``TBELinearMLX`` (and every
    raw one overwritten by ``load_weights``) before this function evaluates
    anything.  The receipt records ``mx.get_peak_memory()`` so the claim is
    checkable rather than asserted.

    ``strict`` (default True) refuses a load that left any parameter holding
    its constructor's initialisation instead of a stored weight -- a silent
    partial load is how a model that produces fluent nonsense gets shipped.
    """
    import json as _json

    from mlx_lm.utils import _get_classes, load_tokenizer as _load_tokenizer

    ta = _artifact_module()
    artifact = ta.open_tbe_artifact(artifact_path, verify_digests=verify_digests)

    mx.reset_peak_memory()

    config = _json.loads(
        (artifact.path / ta.CONFIG_FILENAME).read_text(encoding="utf-8")
    )
    model_class, model_args_class = _get_classes(config=config)
    model = model_class(model_args_class.from_dict(config))

    # -- raw tensors, straight out of the artifact -------------------------
    raw_entries = artifact.raw_entries
    raw_names: List[str] = []
    if raw_entries:
        if not artifact.raw_blob.is_file():
            raise ta.TBEArtifactError(
                "missing_blob",
                f"the manifest declares {len(raw_entries)} raw tensor(s) but "
                f"{artifact.raw_blob} is absent",
            )
        raw_weights = mx.load(str(artifact.raw_blob))
        if hasattr(model, "sanitize"):
            raw_weights = model.sanitize(raw_weights)
        raw_names = sorted(raw_weights)
        model.load_weights(list(raw_weights.items()), strict=False)
        del raw_weights

    # -- coded tensors, straight out of the artifact -----------------------
    model, receipt = swap_model_to_tbe_from_artifact(
        model, artifact, decoder=decoder,
    )

    # -- nothing may still be holding its constructor's initialisation -----
    coded_owners = {n.rpartition(".weight")[0] for n in receipt["swapped_names"]}
    supplied = set(raw_names)
    unsupplied = [
        path for path, _parent, _key, _lin in find_linears(model)
        if path not in coded_owners and f"{path}.weight" not in supplied
    ]
    if unsupplied and strict:
        raise ta.TBEArtifactError(
            "unfilled_parameters",
            f"{len(unsupplied)} nn.Linear(s) were neither coded nor supplied "
            f"by the artifact's raw tensors and still hold their "
            f"constructor's initialisation: {sorted(unsupplied)[:8]}",
        )

    if hasattr(model, "eval"):
        model.eval()
    mx.eval(model.parameters())

    tokenizer = _load_tokenizer(artifact.path)

    receipt = {
        **receipt,
        "schema": ta.TBE_STANDALONE_RECEIPT_SCHEMA,
        "loader": "mlx_standalone",
        "source": "container",
        "artifact_dir": str(artifact.path),
        "artifact_format": artifact.artifact_format,
        "dense_reference_read": False,
        "decoder": decoder,
        "n_raw_tensors_loaded": len(raw_names),
        "n_unsupplied_linears": len(unsupplied),
        "digest_verification": artifact.digest_receipt,
        "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
    }
    return model, tokenizer, receipt


__all__ = [
    "DECODERS",
    "TBESwapError",
    "TBESwapReceipt",
    "find_linears",
    "load_manifest",
    "load_tbe_model",
    "load_tbe_model_standalone",
    "swap_model_to_tbe",
    "swap_model_to_tbe_from_artifact",
]
