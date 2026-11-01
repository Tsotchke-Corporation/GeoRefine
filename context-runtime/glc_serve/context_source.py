"""Explicit selection helpers for the frozen Context -> v2.1 serving bridge."""
from __future__ import annotations

from pathlib import Path
from typing import Callable


def validate_context_selection(*, build_root, bundle, parent, gguf, weight_format,
                               reference_engine=False) -> None:
    """Fail before model/device startup unless Context is explicitly paired with tbe21."""
    if build_root is None:
        return
    if not str(build_root).strip():
        raise ValueError("--context-build-root must name a durable build/cache directory")
    if not bundle:
        raise ValueError("--context-build-root requires --bundle (Context package)")
    if weight_format != "tbe21":
        raise ValueError("--context-build-root requires --weight-format tbe21")
    if gguf:
        raise ValueError("--context-build-root cannot be combined with --gguf")
    if reference_engine:
        raise ValueError("--context-build-root cannot be used with --reference-engine")
    root = Path(build_root).expanduser().resolve()
    if (".scratch" not in root.parts or root.is_relative_to(Path("/tmp"))
            or root.is_relative_to(Path("/private/tmp"))):
        raise ValueError("--context-build-root must be under a durable .scratch directory")
    package = Path(bundle).expanduser().resolve()
    if root.is_relative_to(package):
        raise ValueError("--context-build-root must be outside the Context package")


def context_artifact_factory(build_root: str) -> Callable[[str], object]:
    """Create the pinned Context reader factory; the manifest hash is never user-overridden."""
    from .bctx_v21_transcode import ContextV21TranscodeArtifact

    def build(package_path: str):
        return ContextV21TranscodeArtifact(package_path, build_root=build_root)

    return build


def validate_context_startup_selection(*, workers=0, ram_bytes=None, build_root=None,
                                       context_typed=False, weight_format="tbe21") -> None:
    """Parallel startup is an explicit frozen-Context path with a declared RAM budget."""
    if type(workers) is not int or not 0 <= workers <= 12:
        raise ValueError("context startup workers must be an integer in 0..12")
    if workers == 0:
        if ram_bytes is not None:
            raise ValueError("context startup RAM budget requires nonzero workers")
        return
    if type(ram_bytes) is not int or ram_bytes <= 0:
        raise ValueError("parallel Context startup requires a positive RAM budget in bytes")
    if not build_root or context_typed or weight_format != "tbe21":
        raise ValueError("parallel startup requires explicit frozen Context with tbe21 and a build root")


def parallel_context_artifact_factory(*, build_root, workers, ram_bytes):
    from .context_parallel_startup import ParallelContextV21Artifact
    from glc_loader import codec_v21_native

    def build(package_path):
        return ParallelContextV21Artifact(package_path, build_root=build_root,
            max_workers=workers, max_inflight_bytes=ram_bytes,
            streaming=codec_v21_native.enabled())

    return build


def validate_package_selection(*, package, runtime_format, build_root, cache_root,
                               bundle, parent, gguf, reference_engine=False) -> None:
    """Fail before model/device startup unless ``--package`` (the parallel, disk-cached
    Context -> TBE21 bridge, ``glc_serve.bctx_v21_cache``) is explicitly paired with
    ``--runtime-format tbe21``, a durable JIT build root, and a durable cache root.

    Independent of :func:`validate_context_selection` (the legacy, synchronous
    ``--context-build-root`` + ``--bundle`` path): this never touches that path or its
    error messages, and the two are mutually exclusive with ``--bundle``.
    """
    if package is None:
        return
    if not str(package).strip():
        raise ValueError("--package must name a Context package directory")
    if bundle:
        raise ValueError("--package cannot be combined with --bundle")
    if not parent:
        raise ValueError("--package requires --parent")
    if gguf:
        raise ValueError("--package cannot be combined with --gguf")
    if reference_engine:
        raise ValueError("--package cannot be used with --reference-engine")
    if runtime_format != "tbe21":
        raise ValueError("--package requires --runtime-format tbe21")
    if not build_root or not str(build_root).strip():
        raise ValueError("--package requires --context-build-root <durable .scratch dir>")
    if not cache_root or not str(cache_root).strip():
        raise ValueError("--package requires --context-cache-root <durable .scratch dir>")
    package_resolved = Path(package).expanduser().resolve()
    for label, root in (("--context-build-root", build_root), ("--context-cache-root", cache_root)):
        resolved = Path(root).expanduser().resolve()
        if (".scratch" not in resolved.parts or resolved.is_relative_to(Path("/tmp"))
                or resolved.is_relative_to(Path("/private/tmp"))):
            raise ValueError(f"{label} must be under a durable .scratch directory")
        if resolved.is_relative_to(package_resolved):
            raise ValueError(f"{label} must be outside the Context package")


def cached_context_artifact_factory(*, build_root: str, cache_root: str,
                                    runtime_format: str = "tbe21",
                                    gpu_arch=None, max_workers: int = 0,
                                    progress_cb=None) -> Callable[[str], object]:
    """Create the parallel, disk-cached Context reader factory (``--package`` path)."""
    from .bctx_v21_cache import cached_context_artifact_factory as _factory

    return _factory(build_root=build_root, cache_root=cache_root, runtime_format=runtime_format,
                    gpu_arch=gpu_arch, max_workers=max_workers, progress_cb=progress_cb)


def validate_typed_context_selection(*, context_typed: bool, context_manifest_sha256,
                                     build_root, cache_root, bundle, parent, gguf,
                                     weight_format, reference_engine=False) -> None:
    """Require an explicit, manifest-pinned typed package selection for TBE21.

    The typed adapter consumes the immutable package directly and currently has
    no transcode-cache representation. Config, tokenizer and processor metadata
    are decoded from sealed package assets; no parent metadata tree is required.
    """
    if type(context_typed) is not bool:
        raise ValueError("context_typed must be a bool")
    if not context_typed:
        if context_manifest_sha256 is not None:
            raise ValueError("context_manifest_sha256 requires context_typed=True")
        if cache_root:
            raise ValueError("context_cache_root has no public bidec loader route")
        return
    if (not isinstance(context_manifest_sha256, str) or len(context_manifest_sha256) != 64
            or any(c not in "0123456789abcdef" for c in context_manifest_sha256)):
        raise ValueError("context_typed requires an exact lowercase context_manifest_sha256 pin")
    if not bundle:
        raise ValueError("context_typed requires --bundle <typed Context package>")
    if weight_format != "tbe21":
        raise ValueError("context_typed requires --weight-format tbe21")
    if gguf:
        raise ValueError("context_typed cannot be combined with --gguf")
    if reference_engine:
        raise ValueError("context_typed cannot be used with --reference-engine")
    if build_root:
        raise ValueError("context_typed cannot be combined with --context-build-root")
    if cache_root:
        raise ValueError("typed Context cache is unsupported")


def typed_context_artifact_factory(*, expected_manifest_sha256: str) -> Callable[[str], object]:
    """Create the explicitly selected typed-package reader with its required manifest pin."""
    from .bctx_v21_typed import typed_context_artifact_factory as _factory

    return _factory(expected_manifest_sha256=expected_manifest_sha256)


def select_context_artifact_factory(*, context_typed: bool = False,
                                    context_manifest_sha256=None, build_root=None,
                                    startup_workers=0, startup_ram_bytes=None):
    """Select a source adapter only when its corresponding source option is explicit."""
    validate_context_startup_selection(workers=startup_workers, ram_bytes=startup_ram_bytes,
        build_root=build_root, context_typed=context_typed)
    if startup_workers:
        return parallel_context_artifact_factory(build_root=build_root,
            workers=startup_workers, ram_bytes=startup_ram_bytes)
    if context_typed:
        return typed_context_artifact_factory(
            expected_manifest_sha256=context_manifest_sha256)
    if context_manifest_sha256 is not None:
        raise ValueError("context_manifest_sha256 requires context_typed=True")
    if build_root is not None:
        return context_artifact_factory(build_root)
    return None


def tbe2_loader_kwargs(weight_format: str, embed_placement: str, *, artifact_factory=None) -> dict:
    """Build codec-loader kwargs while keeping the legacy TBE2 call empty."""
    if weight_format == "tbe21":
        result = {"weight_format": "tbe21", "embed_placement": embed_placement}
        if artifact_factory is not None:
            result["artifact_factory"] = artifact_factory
        return result
    return {}
