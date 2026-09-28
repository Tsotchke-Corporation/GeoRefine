"""glc_serve -- serve a GLC-TBE container (or its bf16 parent) behind an OpenAI API.

Product layer over ``glc_loader``.  Four pieces, each usable alone:

``bundle``   the SERVING BUNDLE format (``georefine.tbe.serve.v1``): the TBE
             container re-packed into size-bounded safetensors shards with a
             manifest carrying the sha256 of every shard and sidecar and the
             ``blake2b`` of every source tensor.  Shards are fetched from local
             disk, HTTP(S) or ``gs://`` and hash-verified before a single byte
             of them is used.
``pack``     build a bundle once (from an existing TBE container, or by CPU
             transcode of a bf16 checkpoint); serve it many times.
``loader``   stream a bundle straight onto the GPU: coded tensors are uploaded
             AS STORED (no decode, no re-encode, no dense-resident build peak),
             with opt-in host-resident embeddings, a coded ``lm_head`` and
             budgeted layer streaming over PCIe.
``server``   an OpenAI-compatible HTTP server (``/v1/chat/completions``,
             ``/v1/completions``, SSE streaming, images, tools, logprobs) over a
             wave-batched generation engine; the same server also serves the
             uncompressed parent (``--backend dense``) so a head-to-head runs
             both arms through identical code.

Nothing in this package imports ``experiments.georefine``.
"""
from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
