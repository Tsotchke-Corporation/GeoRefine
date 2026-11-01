"""Load the original MTP head through the package's verified CPU tensor resolver."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch


def _manifest_sha256(package: Path) -> str:
    digest = hashlib.sha256()
    with (package / "manifest.json").open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_borrowed_resolver(package, resolver) -> Path:
    """Check the identity required before sharing a live resolver with a worker."""
    root = Path(package).resolve()
    resolver_root = getattr(resolver, "package", None)
    if resolver_root is None or Path(resolver_root).resolve() != root:
        raise ValueError("borrowed resolver package mismatch")
    expected_manifest = _manifest_sha256(root)
    if getattr(resolver, "manifest_sha256", None) != expected_manifest:
        raise ValueError("borrowed resolver manifest identity mismatch")
    return root


def _mtp_source_copy_budget(resolver) -> int:
    """Bound temporary NumPy copies from the verified MTP source entries."""
    largest = 0
    for name in resolver.tensor_names:
        if not name.startswith("mtp."):
            continue
        try:
            row = resolver.entry(name)
            shape = tuple(row["shape"])
            source_bytes = int(row["source_bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid verified MTP source entry: {name}") from exc
        if not shape or any(type(dim) is not int or dim <= 0 for dim in shape):
            raise ValueError(f"invalid verified MTP source shape: {name}")
        expected = math.prod(shape) * 2
        if source_bytes != expected:
            raise ValueError(f"invalid verified MTP source byte count: {name}")
        largest = max(largest, expected)
    return largest


def _load_package_mtp_from_resolver(package, config, *, device, resolver,
                                    skeleton_factory, include_copy_budget=False):
    from .loader import split_parent, _place_param

    mtp = skeleton_factory(config)
    expected = dict(mtp.named_parameters())
    if not expected:
        raise ValueError("MTP skeleton has no parameters")
    hashes, total, max_source_copy = {}, 0, 0
    required = {"mtp." + name for name in expected}
    present = {name for name in resolver.tensor_names if name.startswith("mtp.")}
    if present != required:
        raise ValueError("MTP tensor set does not match the model skeleton")
    selected = resolver.manifest.get("selected_candidates", {})
    if required.intersection(selected):
        raise ValueError("MTP tensors must be unchanged baseline frames")
    for local_name, parameter in expected.items():
        name = "mtp." + local_name
        row = resolver.entry(name)
        words = resolver.resolve(name)
        if (row.get("dtype") != "BF16" or words.dtype != np.dtype("<u2")
                or tuple(row.get("shape", ())) != tuple(parameter.shape)
                or tuple(words.shape) != tuple(parameter.shape)):
            raise ValueError(f"MTP tensor shape/dtype mismatch: {name}")
        source_bytes = row.get("source_bytes")
        actual_sha = hashlib.sha256(words.tobytes()).hexdigest()
        if source_bytes != words.nbytes or row.get("source_sha256") != actual_sha:
            raise ValueError(f"MTP tensor source SHA mismatch: {name}")
        value = torch.from_numpy(words.copy()).view(torch.bfloat16)
        parent, attr = split_parent(mtp, local_name)
        _place_param(parent, attr, value, device)
        hashes[name] = actual_sha
        total += words.nbytes
        max_source_copy = max(max_source_copy, words.nbytes)
        del value, words
    if any(p.is_meta for p in mtp.parameters()):
        raise ValueError("MTP parameters remain on meta")
    if any(b.is_meta for b in mtp.buffers()):
        raise ValueError("MTP buffers remain on meta")
    mtp = mtp.to(device).eval()
    receipt = dict(source="verified_package_baseline", tensor_count=len(hashes),
                   source_sha256=hashes, resident_dense_bytes=total,
                   materialized_checkpoint=False)
    if include_copy_budget:
        receipt['max_mtp_source_copy_bytes'] = max_source_copy
    return mtp, receipt


def load_package_mtp(package, config, *, device, resolver_factory=None,
                     skeleton_factory=None, verified_resolver=None, prepared_skeleton=None):
    from georefine_context_runtime.bitexact_predictive_package import VerifiedTensorResolver

    resolver_factory = resolver_factory or VerifiedTensorResolver
    from .loader import build_mtp_skeleton

    if prepared_skeleton is not None:
        if skeleton_factory is not None:
            raise ValueError('provide either an MTP skeleton or a skeleton factory')
        skeleton_factory = lambda _config: prepared_skeleton
    else:
        skeleton_factory = skeleton_factory or build_mtp_skeleton
    if verified_resolver is not None:
        root = _validate_borrowed_resolver(package, verified_resolver)
        return _load_package_mtp_from_resolver(
            root, config, device=device, resolver=verified_resolver,
            skeleton_factory=skeleton_factory, include_copy_budget=True)
    with resolver_factory(Path(package), cache_bytes=0) as resolver:
        return _load_package_mtp_from_resolver(
            package, config, device=device, resolver=resolver,
            skeleton_factory=skeleton_factory)


__all__ = ["load_package_mtp", "_mtp_source_copy_budget",
           "_validate_borrowed_resolver"]
