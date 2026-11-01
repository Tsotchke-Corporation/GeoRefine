"""``glc_loader`` -- read a GLC-RELEASE artifact with no build repo present.

    import sys; sys.path.insert(0, "/path/to/artifact")
    from glc_loader import load_model
    model, tokenizer, receipt = load_model("/path/to/artifact", device="cuda")

or, without writing any code at all:

    python -m glc_loader generate --artifact /path/to/artifact --prompt "hello"
    python -m glc_loader expand   --artifact /path/to/artifact --out ./dense

Third-party requirements: ``torch`` and ``safetensors`` for the container;
``transformers`` additionally for :func:`load_model` (it supplies the model
class and the tokenizer, not the weights).  ``triton`` is optional and only
selects a faster backend.  Nothing here imports the repository that built the
artifact -- that is the point of the format, and it is enforced by a test.
"""
from .artifact import (
    Artifact,
    FORMAT,
    GLCArtifactError,
    expand,
    open_artifact,
    verify,
)
from .container import (
    CONTAINER_MAGIC,
    FWP1Error,
    FWP1Tensor,
    decode_fwp1,
    decode_fwp1_rows,
    encode_fwp1,
    fwp1_certify,
)
from .loader import (
    CertificateError,
    assert_certificate_usable,
    load_model,
    load_tokenizer,
)
from .modules import (
    BACKENDS,
    GLCBackendError,
    GLCEmbedding,
    GLCLinear,
    GLCTiedLMHead,
)
from .tbe_artifact import (
    ARTIFACT_FORMAT,
    COMPRESSION_INFO_FILENAME,
    COMPRESSION_SCHEMA_VERSION,
    STANDALONE_LOADER,
    TBEArtifact,
    TBEArtifactError,
    build_meta_skeleton_from_artifact,
    detect_artifact_format,
    is_legacy_v1_container,
    iter_artifact_tensors,
    load_standalone,
    open_tbe_artifact,
    verify_sha256sums,
    verify_standalone_bit_exact,
    write_sha256sums,
)
from .tbe_container import (
    TBEError,
    TBETensor,
    decode_tbe,
    encode_tbe,
    tbe_certify,
)
from .tbe_serve_portable import PortableTBELinear, PortableTBEError, load_compressed_transformers
from .tbe_device_map import (
    STRATEGY_BALANCED,
    STRATEGY_SEQUENTIAL,
    TBEDeviceMapError,
    devices_in_map,
    group_key,
    manifest_group_bytes,
    plan_device_map,
    plan_device_map_from_manifest,
    resolve_device,
    single_device_map,
    validate_device_map,
)
from .tbe_mma import (
    TBEDevice,
    TBEMMAError,
    resolve_tbe_mma_arch,
    tbe_mma_available,
    upload_tbe,
)
from .tbe_modules import (
    GLCTBEExpertBank,
    GLCTBELinear,
    TBEBackendError,
    TBEPoolRegistry,
    TBETransientPool,
)
from .tbe_serving import (
    TBEServingError,
    enable_hf_tbe_serving,
    place_uncoded_tensors,
)
from .tbe_stream_loader import (
    TBEStreamLoadError,
    build_meta_skeleton,
    iter_safetensors_shards,
    stream_load_tbe_model,
    stream_load_tbe_model_standalone,
    stream_place_tensors,
)

__version__ = "1.1.1"

__all__ = [
    "ARTIFACT_FORMAT",
    "Artifact",
    "BACKENDS",
    "COMPRESSION_INFO_FILENAME",
    "COMPRESSION_SCHEMA_VERSION",
    "CONTAINER_MAGIC",
    "CertificateError",
    "FORMAT",
    "FWP1Error",
    "FWP1Tensor",
    "GLCArtifactError",
    "GLCBackendError",
    "GLCEmbedding",
    "GLCLinear",
    "GLCTBEExpertBank",
    "GLCTBELinear",
    "GLCTiedLMHead",
    "STANDALONE_LOADER",
    "STRATEGY_BALANCED",
    "STRATEGY_SEQUENTIAL",
    "TBEArtifact",
    "TBEArtifactError",
    "TBEBackendError",
    "TBEDevice",
    "TBEDeviceMapError",
    "TBEError",
    "TBEMMAError",
    "TBEPoolRegistry",
    "TBEServingError",
    "TBEStreamLoadError",
    "TBETensor",
    "TBETransientPool",
    "__version__",
    "assert_certificate_usable",
    "build_meta_skeleton",
    "build_meta_skeleton_from_artifact",
    "decode_fwp1",
    "decode_fwp1_rows",
    "decode_tbe",
    "detect_artifact_format",
    "devices_in_map",
    "enable_hf_tbe_serving",
    "encode_fwp1",
    "encode_tbe",
    "expand",
    "fwp1_certify",
    "group_key",
    "is_legacy_v1_container",
    "iter_artifact_tensors",
    "iter_safetensors_shards",
    "load_model",
    "load_standalone",
    "load_tokenizer",
    "manifest_group_bytes",
    "open_artifact",
    "open_tbe_artifact",
    "place_uncoded_tensors",
    "plan_device_map",
    "plan_device_map_from_manifest",
    "resolve_device",
    "resolve_tbe_mma_arch",
    "single_device_map",
    "stream_load_tbe_model",
    "stream_load_tbe_model_standalone",
    "stream_place_tensors",
    "tbe_certify",
    "tbe_mma_available",
    "upload_tbe",
    "validate_device_map",
    "verify",
    "verify_sha256sums",
    "verify_standalone_bit_exact",
    "write_sha256sums",
]
