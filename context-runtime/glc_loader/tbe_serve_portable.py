"""Portable compressed-weight loader for ``georefine.tbe.serve.v1``.

The weight tensors stay TBE-encoded in resident storage. The CPU reference
backend decodes one linear weight transiently for each call and discards it
after the matmul. Faster device kernels can replace that forward path without
changing the bundle or the Transformers model graph.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from .tbe_container import TBETensor, decode_tbe


class PortableTBEError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 26), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hydrate_derived_buffers(model: nn.Module) -> None:
    """Rebuild Qwen rotary buffers that a meta skeleton does not allocate."""
    for module in model.modules():
        missing = [name for name, value in module.named_buffers(recurse=False) if value.is_meta]
        if not missing:
            continue
        kind = type(module).__name__
        with torch.device("cpu"):
            if kind == "Qwen3_5VisionRotaryEmbedding":
                fresh = type(module)(module.dim, module.theta)
            elif kind == "Qwen3_5TextRotaryEmbedding":
                fresh = type(module)(module.config, device="cpu")
            else:
                raise PortableTBEError(f"unhandled derived meta buffers in {kind}: {missing}")
        for name in missing:
            replacement = fresh._buffers.get(name)
            if replacement is None or replacement.is_meta:
                raise PortableTBEError(f"failed to initialize {kind}.{name}")
            module._buffers[name] = replacement


class PortableTBELinear(nn.Module):
    """Linear layer backed by encoded buffers, with a transient BF16 weight."""

    def __init__(self, entry: dict, arrays: dict[str, torch.Tensor], bias: bool):
        super().__init__()
        self.entry = {k: entry[k] for k in ("shape", "coded_shape", "layout", "mode", "base",
                                           "tiles", "escapes", "superblock") if k in entry}
        shape = [int(x) for x in entry["shape"]]
        if len(shape) != 2:
            raise PortableTBEError(f"linear weight is not rank 2: {entry['name']}")
        self.out_features, self.in_features = shape
        for part in ("planes", "smb", "esc", "sbbase"):
            self.register_buffer(part, arrays[part].cpu(), persistent=False)
        self.bias = nn.Parameter(torch.empty(self.out_features, device="meta"),
                                 requires_grad=False) if bias else None

    @property
    def resident_weight_bytes(self) -> int:
        return sum(getattr(self, p).numel() * getattr(self, p).element_size()
                   for p in ("planes", "smb", "esc", "sbbase"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.entry
        n, k = (int(v) for v in (e.get("coded_shape") or e["shape"]))
        encoded = TBETensor(shape=(n, k), layout=e["layout"], mode=int(e["mode"]),
                            base=int(e["base"]), tiles=int(e["tiles"]),
                            escapes=int(e["escapes"]), planes=self.planes,
                            smb=self.smb, esc=self.esc, sbbase=self.sbbase,
                            superblock=int(e.get("superblock", 32)))
        weight = decode_tbe(encoded).reshape(self.out_features, self.in_features)
        if x.device.type != "cpu":
            weight = weight.to(x.device)
        if weight.dtype != x.dtype:
            weight = weight.to(x.dtype)
        if self.bias is not None and self.bias.is_meta:
            raise PortableTBEError("linear bias was not loaded")
        return F.linear(x, weight, self.bias)


def load_compressed_transformers(bundle: str | Path, *, device: str = "cpu",
                                 expect_manifest_sha256: str | None = None):
    """Return a Qwen Transformers module with resident TBE weights.

    The 15 MTP-only checkpoint tensors are retained in the bundle for native
    speculation; standard ``AutoModelForMultimodalLM`` has no MTP module.
    Other unexpected or missing tensors are fatal.
    """
    try:
        from transformers import AutoConfig, AutoModelForMultimodalLM
    except ImportError as exc:
        raise PortableTBEError("this Qwen3.8 path requires Transformers with "
                               "AutoModelForMultimodalLM support") from exc

    root = Path(bundle)
    if not root.is_dir():
        from huggingface_hub import snapshot_download

        repo, marker, revision = str(bundle).partition("@")
        root = Path(snapshot_download(repo_id=repo, revision=revision if marker else None))
    raw = (root / "serve_manifest.json").read_bytes()
    manifest_sha = hashlib.sha256(raw).hexdigest()
    if expect_manifest_sha256 and manifest_sha != expect_manifest_sha256:
        raise PortableTBEError("manifest SHA-256 differs")
    manifest = json.loads(raw)
    if manifest.get("format") != "georefine.tbe.serve.v1":
        raise PortableTBEError("unsupported bundle format")
    config = AutoConfig.from_pretrained(str(root), local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForMultimodalLM.from_config(config)
    _hydrate_derived_buffers(model)
    expected_names = {name for name, _ in model.named_parameters()}
    entries = manifest["tensors"]
    names = {e["name"] for e in entries if not e["name"].startswith("mtp.")}
    if len(names) != len(entries) - sum(e["name"].startswith("mtp.") for e in entries):
        raise PortableTBEError("duplicate tensor name")
    if names != expected_names:
        raise PortableTBEError(f"model tensor coverage differs: missing={len(expected_names - names)} "
                               f"extra={len(names - expected_names)}")
    by_shard: dict[int, list[dict]] = {i: [] for i in range(len(manifest["shards"]))}
    for e in entries:
        by_shard[int(e["shard"])].append(e)
    seen = set()
    for i, shard in enumerate(manifest["shards"]):
        path = root / shard["file"]
        if path.stat().st_size != shard["bytes"] or _sha256(path) != shard["sha256"]:
            raise PortableTBEError(f"codec shard hash differs: {shard['file']}")
        with safe_open(str(path), framework="pt", device="cpu") as source:
            for e in by_shard[i]:
                name = e["name"]
                if name.startswith("mtp."):
                    continue
                parent_name, _, leaf = name.rpartition(".")
                parent = model.get_submodule(parent_name)
                if e["kind"] == "raw":
                    tensor = source.get_tensor(name)
                    if list(tensor.shape) != e["shape"]:
                        raise PortableTBEError(f"raw tensor shape differs: {name}")
                    parent._parameters[leaf] = nn.Parameter(tensor.to(device), requires_grad=False)
                elif e["kind"] == "tbe":
                    if leaf != "weight" or not isinstance(parent, nn.Linear):
                        raise PortableTBEError(f"coded tensor is not a Linear weight: {name}")
                    arrays = {part: source.get_tensor(f"{name}.{part}")
                              for part in ("planes", "smb", "esc", "sbbase")}
                    replacement = PortableTBELinear(e, arrays, parent.bias is not None)
                    for parameter_name, parameter in parent._parameters.items():
                        if parameter_name != "weight" and parameter is not None and not parameter.is_meta:
                            replacement.register_parameter(parameter_name, parameter)
                    owner, _, module_name = parent_name.rpartition(".")
                    model.get_submodule(owner)._modules[module_name] = replacement
                else:
                    raise PortableTBEError(f"unknown tensor kind: {name}")
                seen.add(name)
    residual = [name for name, p in model.named_parameters() if p.is_meta]
    residual_buffers = [name for name, b in model.named_buffers() if b.is_meta]
    if seen != expected_names or residual or residual_buffers:
        raise PortableTBEError(f"incomplete model: unseen={len(expected_names - seen)} "
                               f"meta={residual[:8]} buffers={residual_buffers[:8]}")
    model.eval()
    model.georefine_tbe_receipt = {
        "format": manifest["format"], "manifest_sha256": manifest_sha,
        "base_tensors": len(seen), "mtp_tensors_retained": len(entries) - len(seen),
        "coded_linears": sum(isinstance(m, PortableTBELinear) for m in model.modules()),
        "resident_coded_bytes": sum(m.resident_weight_bytes for m in model.modules()
                                    if isinstance(m, PortableTBELinear)),
        "backend": "torch-cpu-reference", "device": str(device),
    }
    return model
