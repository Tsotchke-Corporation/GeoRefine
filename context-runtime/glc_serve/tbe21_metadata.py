"""Construct pinned Transformers metadata from verified TBE21 package bytes."""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from types import SimpleNamespace


class TBE21MetadataError(ValueError):
    """The selected TBE21 package does not seal usable HF metadata assets."""


_BASE_ASSETS = ("config.json", "tokenizer.json", "tokenizer_config.json",
                "generation_config.json")
_VISION_ASSETS = ("preprocessor_config.json", "video_preprocessor_config.json")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _pin(row):
    if (not isinstance(row, Mapping) or type(row.get("bytes")) is not int or row["bytes"] < 0
            or not isinstance(row.get("sha256"), str) or len(row["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in row["sha256"])):
        raise TBE21MetadataError("invalid metadata asset integrity pin")


def _verify(raw, row, label):
    _pin(row)
    if len(raw) != row["bytes"] or _sha(raw) != row["sha256"]:
        raise TBE21MetadataError(f"metadata asset integrity mismatch: {label}")
    return raw


def _typed_assets(artifact):
    manifest = artifact.manifest
    rows = manifest.get("assets") if isinstance(manifest, Mapping) else None
    if not isinstance(rows, (list, tuple)):
        raise TBE21MetadataError("typed package has no asset inventory")
    by_source = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("source_path"), str):
            raise TBE21MetadataError("typed package contains an invalid asset receipt")
        path = row["source_path"]
        if path in by_source:
            raise TBE21MetadataError(f"duplicate typed metadata asset receipt: {path}")
        by_source[path] = row

    def read(name):
        candidates = (name, f"sidecars/{name}")
        present = [path for path in candidates if path in by_source]
        if len(present) != 1:
            raise TBE21MetadataError(f"typed package must seal exactly one {name!r} asset")
        path = present[0]
        row = by_source[path]
        raw = artifact.read_asset(path)
        return _verify(raw, row, path)

    read.asset_names = set(by_source)
    return read


def _direct_sidecars(artifact, names):
    manifest = artifact.manifest
    rows = manifest.get("files") if isinstance(manifest, Mapping) else None
    if not isinstance(rows, (list, tuple)):
        return None
    inventory = {}
    for row in rows:
        if isinstance(row, Mapping) and isinstance(row.get("path"), str):
            if row["path"] in inventory:
                raise TBE21MetadataError(f"duplicate file inventory row: {row['path']}")
            inventory[row["path"]] = row
    deployment = getattr(artifact, "deployment", None)
    deployment_assets = deployment.get("metadata_assets") if isinstance(deployment, dict) else None
    root = Path(getattr(artifact, "root", getattr(artifact, "path", ""))).resolve()
    if not root.is_dir():
        return None
    pin_source = deployment_assets if isinstance(deployment_assets, dict) else inventory
    selected = [f"sidecars/{name}" for name in names]
    present = [name for name in selected if name in pin_source]
    if not present:
        return None
    if len(present) != len(selected):
        raise TBE21MetadataError("package has an incomplete sealed sidecar inventory")

    def read(name):
        rel = f"sidecars/{name}"
        pin = pin_source.get(rel)
        if pin is None:
            raise TBE21MetadataError(f"package sidecar is not sealed by its inventory: {rel}")
        _pin(pin)
        path = root.joinpath(*PurePosixPath(rel).parts)
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as exc:
            raise TBE21MetadataError(f"package sidecar is missing or escapes package: {rel}") from exc
        parts = PurePosixPath(rel).parts
        if (any(root.joinpath(*parts[:index]).is_symlink() for index in range(1, len(parts) + 1))
                or not path.is_file()):
            raise TBE21MetadataError(f"package sidecar is not a regular file: {rel}")
        raw = path.read_bytes()
        _verify(raw, pin, rel)
        if isinstance(deployment_assets, dict):
            inventory_pin = inventory.get(rel)
            if inventory_pin is not None and (inventory_pin.get("bytes"), inventory_pin.get("sha256")) != (
                    pin["bytes"], pin["sha256"]):
                raise TBE21MetadataError(f"deployment and file inventory disagree for {rel}")
        return raw

    for name in names:
        read(name)
    read.asset_names = set(deployment_assets if isinstance(deployment_assets, dict) else inventory)
    return read


def _archive_sidecars(artifact, names):
    manifest = artifact.manifest
    rows = manifest.get("files") if isinstance(manifest, Mapping) else None
    if not isinstance(rows, (list, tuple)):
        return None
    pins = [row for row in rows if isinstance(row, Mapping) and row.get("path") == "model-assets.tar.xz"]
    if len(pins) != 1:
        return None
    root = Path(getattr(artifact, "root", getattr(artifact, "path", ""))).resolve()
    if not root.is_dir():
        return None
    path = root / "model-assets.tar.xz"
    if path.is_symlink() or not path.is_file():
        raise TBE21MetadataError("model-assets.tar.xz is not a regular package file")
    archive_raw = _verify(path.read_bytes(), pins[0], "model-assets.tar.xz")
    try:
        archive = tarfile.open(fileobj=io.BytesIO(archive_raw), mode="r:xz")
    except (tarfile.TarError, OSError) as exc:
        raise TBE21MetadataError("pinned model-assets.tar.xz is invalid") from exc
    with archive:
        members = {}
        asset_data = {}
        for member in archive.getmembers():
            rel = PurePosixPath(member.name)
            if (rel.is_absolute() or rel.as_posix() != member.name or ".." in rel.parts
                    or not member.isfile()):
                raise TBE21MetadataError(f"unsafe model metadata archive member: {member.name!r}")
            if member.name in members:
                raise TBE21MetadataError(f"duplicate model metadata archive member: {member.name!r}")
            members[member.name] = member
            stream = archive.extractfile(member)
            if stream is None:
                raise TBE21MetadataError(f"cannot decode metadata archive member: {member.name!r}")
            asset_data[member.name] = stream.read()
        if len(members) != 13:
            raise TBE21MetadataError("pinned Context metadata archive must contain exactly 13 assets")

        def read(name):
            rel = f"sidecars/{name}"
            if rel not in members:
                raise TBE21MetadataError(f"pinned metadata archive lacks sidecars/{name}")
            return asset_data[rel]

        for name in names:
            read(name)
        read.asset_names = {PurePosixPath(member).name for member in members
                            if member.startswith("sidecars/")}
        return read


def _sealed_reader(artifact, *, include_vision):
    source = artifact
    if not callable(getattr(source, "read_asset", None)) and getattr(source, "_source", None) is not None:
        source = source._source
    names = list(_BASE_ASSETS + _VISION_ASSETS if include_vision else _BASE_ASSETS)
    manifest = getattr(source, "manifest", None)

    schema = manifest.get("schema") if isinstance(manifest, Mapping) else None
    if schema == "georefine-typed-context-package-v1" and callable(getattr(source, "read_asset", None)):
        read = _typed_assets(source)
        for name in names:
            read(name)
    else:
        read = _direct_sidecars(source, names)
        if read is None:
            read = _archive_sidecars(source, names)
    if read is None:
        raise TBE21MetadataError("TBE21 package format has no sealed metadata asset route")
    available = getattr(read, "asset_names", set())
    template_present = ("chat_template.jinja" in available
                        or "sidecars/chat_template.jinja" in available)
    return read, tuple(names + (["chat_template.jinja"] if template_present else []))


def _json_object(read, name):
    try:
        value = json.loads(read(name).decode("utf-8"), object_pairs_hook=_unique_pairs,
                           parse_constant=_reject_constant)
    except TBE21MetadataError:
        raise
    except (UnicodeError, json.JSONDecodeError, KeyError) as exc:
        raise TBE21MetadataError(f"invalid sealed metadata JSON {name!r}: {exc}") from exc
    if not isinstance(value, dict):
        raise TBE21MetadataError(f"sealed metadata JSON {name!r} must be an object")
    return value


def _unique_pairs(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise TBE21MetadataError(f"duplicate metadata JSON key {key!r}")
        output[key] = value
    return output


def _reject_constant(token):
    raise TBE21MetadataError(f"non-standard JSON constant {token}")


def _construct_tokenizer(read, *, chat_template_present):
    from tokenizers import Tokenizer
    from transformers.tokenization_utils_base import AddedToken
    from transformers.tokenization_utils_tokenizers import TokenizersBackend

    config = _json_object(read, "tokenizer_config.json")
    raw = read("tokenizer.json")
    try:
        backend = Tokenizer.from_str(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise TBE21MetadataError(f"invalid tokenizer.json: {exc}") from exc
    options = dict(config)
    for key in ("tokenizer_class", "auto_map", "name_or_path"):
        options.pop(key, None)
    decoder = options.pop("added_tokens_decoder", {})
    if not isinstance(decoder, dict):
        raise TBE21MetadataError("tokenizer added_tokens_decoder must be an object")
    try:
        options["added_tokens_decoder"] = {
            int(index): (AddedToken(**token) if isinstance(token, dict) else token)
            for index, token in decoder.items()
        }
    except (TypeError, ValueError) as exc:
        raise TBE21MetadataError(f"invalid tokenizer added_tokens_decoder: {exc}") from exc
    extra = options.get("extra_special_tokens")
    if isinstance(extra, dict):
        named = options.setdefault("model_specific_special_tokens", {})
        if not isinstance(named, dict):
            raise TBE21MetadataError("model_specific_special_tokens must be an object")
        named.update(extra)
        options.pop("extra_special_tokens")
    if chat_template_present:
        try:
            options["chat_template"] = read("chat_template.jinja").decode("utf-8")
        except UnicodeError as exc:
            raise TBE21MetadataError("invalid chat_template.jinja UTF-8") from exc
    return TokenizersBackend(tokenizer_object=backend, **options), config


def _construct_processor(read, tokenizer):
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    preprocessor = _json_object(read, "preprocessor_config.json")
    video_config = _json_object(read, "video_preprocessor_config.json")
    if (preprocessor.get("processor_class") != "Qwen3VLProcessor"
            or preprocessor.get("image_processor_type") != "Qwen2VLImageProcessorFast"
            or video_config.get("video_processor_type") != "Qwen3VLVideoProcessor"):
        raise TBE21MetadataError("unsupported sealed TBE21 vision metadata class")
    image = Qwen2VLImageProcessor.from_dict(preprocessor)
    video = Qwen3VLVideoProcessor.from_dict(video_config)
    components = {"tokenizer": tokenizer, "image_processor": image,
                  "video_processor": video}
    attributes = list(Qwen3VLProcessor.get_attributes())
    unsupported = [name for name in attributes if name not in components]
    missing = [name for name in attributes if name in components and components[name] is None]
    if unsupported or missing:
        raise TBE21MetadataError(f"unsupported/missing processor attributes: {unsupported + missing}")
    args = [components[name] for name in attributes]
    processor_config = _json_object(read, "preprocessor_config.json")
    processor = Qwen3VLProcessor.from_args_and_dict(
        args, processor_config, **({"chat_template": tokenizer.chat_template}
                                  if getattr(tokenizer, "chat_template", None) is not None else {}))
    return processor


def load_tbe21_metadata(artifact, *, include_vision: bool = False):
    """Construct outer config, tokenizer and optional processor from package bytes.

    Only artifacts with typed asset receipts, inventory-sealed fullpack sidecars,
    or an inventory-pinned Context ``model-assets.tar.xz`` are accepted. No files
    are materialized. For multimodal startup this supports the frozen Qwen3.5 /
    Qwen3VL sidecar class set and fails closed for other processor families.
    """
    read, asset_names = _sealed_reader(artifact, include_vision=include_vision)
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise TBE21MetadataError("Transformers is required for TBE21 metadata") from exc
    config_values = _json_object(read, "config.json")
    model_type = config_values.pop("model_type", None)
    if not isinstance(model_type, str) or not model_type:
        raise TBE21MetadataError("config.json has no model_type")
    config = AutoConfig.for_model(model_type, **config_values)
    tokenizer, _tokenizer_config = _construct_tokenizer(
        read, chat_template_present="chat_template.jinja" in asset_names)
    processor = None
    if include_vision:
        if getattr(config, "model_type", None) not in ("qwen3_5", "qwen3_5_moe"):
            raise TBE21MetadataError("package vision metadata supports only Qwen3.5 config classes")
        processor = _construct_processor(read, tokenizer)
    generation_config = None
    if "generation_config.json" in asset_names:
        generation_config = _json_object(read, "generation_config.json")
    hashes = {name: _sha(read(name)) for name in asset_names}
    return SimpleNamespace(config=config, tokenizer=tokenizer, processor=processor,
                           generation_config=generation_config, asset_sha256=hashes)


__all__ = ["TBE21MetadataError", "load_tbe21_metadata"]
