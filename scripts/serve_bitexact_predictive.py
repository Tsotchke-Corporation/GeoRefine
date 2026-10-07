#!/usr/bin/env python3
"""Serve an immutable bit-exact predictive context package over the BCTX HTTP API."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def create_predictive_app(*, package, device="cuda:0", attention_implementation="eager",
                          stride=1024, loader_workers=1, metadata_cache_dir=None,
                          model_id=None, **http_options):
    """Load predictive weights, then reuse the established BCTX HTTP backend."""
    from scripts.bitexact_predictive_serving import load_predictive_model
    from scripts.serve_bitexact_context import BctxModelEngine, create_app
    from transformers import AutoProcessor, AutoTokenizer

    model, receipt = load_predictive_model(
        package, device=device, attention_implementation=attention_implementation,
        stride=stride, loader_workers=loader_workers,
        metadata_cache_dir=metadata_cache_dir,
    )
    metadata_dir = Path(receipt["metadata_dir"])
    processor = AutoProcessor.from_pretrained(str(metadata_dir), local_files_only=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(str(metadata_dir), local_files_only=True)
    engine = BctxModelEngine(model, processor, tokenizer, model_id or Path(package).name, device)
    return create_app(engine=engine, model_id=model_id, **http_options)


def _configure_exact_runtime():
    """Apply the deterministic CUDA settings used by the predictive qualification path."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention-implementation", default="eager")
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id")
    parser.add_argument("--loader-workers", type=int, default=1)
    parser.add_argument("--metadata-cache-dir", type=Path)
    parser.add_argument("--max-pending-requests", type=int, default=4)
    parser.add_argument("--allow-remote-host", action="append", default=[],
                        help="exact HTTPS media host allowlist (remote media is off by default)")
    args = parser.parse_args(argv)
    _configure_exact_runtime()
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install uvicorn to serve a predictive package") from exc
    app = create_predictive_app(
        package=args.package, device=args.device,
        attention_implementation=args.attention_implementation, stride=args.stride,
        loader_workers=args.loader_workers, metadata_cache_dir=args.metadata_cache_dir,
        model_id=args.model_id, max_pending_requests=args.max_pending_requests,
        remote_url_allowlist=args.allow_remote_host,
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
