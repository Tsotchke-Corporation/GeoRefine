# GeoRefine codec

GeoRefine is developed by **Tsotchke Corporation**.
Copyright 2026 Tsotchke Corporation. The codec, runtime, and verifier source
are licensed under Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
The Qwen model weights retain their upstream license and attribution.

GeoRefine TBE stores pretrained BF16 weights in a reversible encoded format
and serves them without keeping a dense copy of every coded weight resident.
This repository contains the Apache-2.0 codec, runtime, verifier, and tests.
Model weights live in the [Tsotchke Corporation Qwen3.8-27B TBE model repository](https://huggingface.co/Tsotchke-Corporation/Qwen3.8-27B-GeoRefine-TBE).

## First release scope

For Qwen3.8-27B at upstream revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, the complete bundle has
39,832,462,895 encoded tensor bytes versus 55,562,855,904 BF16 tensor bytes
(1.395× smaller). The independent verifier matched **1,199/1,199** tensors
and **10/10** sidecars, checking all parent and bundle shard hashes. The
[complete verifier receipt](proofs/qwen38-27b/bit_exact_receipt.json) is in
this repository.

An installed `glc-loader` wheel with its bundled SM120 tune ran the model on
one NVIDIA RTX PRO 6000 Blackwell Server Edition card. Its `FastSession`
path matched the BF16 parent's token IDs on eight fixed text prompts and one
synthetic image. Batch-1 decode measured **37.880 versus 28.288 tok/s**
(1.339×); end-to-end throughput measured **32.314 versus 25.762 tok/s**
(1.254×). One 128-token text probe measured **68.820 tok/s** with MTP k=5
versus **37.743 tok/s** plain, with identical text and image token sequences
in those probes. See the [full runtime result](proofs/qwen38-27b/FASTSESSION_RESULT.json)
and linked receipts for the exact conditions and limitations.

The compressed PyTorch reference loader also ran the full model's text and
minimal image forwards on Linux CPU. It is a compatibility path and is too
slow for practical 27B serving. Fast decode has been measured in this package
only on the RTX PRO 6000 SM120. Other NVIDIA GPUs require an explicit tune
and their own correctness and speed tests; a fast Metal path for this model
has not been verified. The standard `glc-serve` HTTP generation route has a
different performance contract from `FastSession`.

## Packages

| Path | Distribution | Purpose |
|---|---|---|
| [`release/`](release/) | `glc-loader` 1.1.1 | Portable TBE loader, CUDA FastSession, CPU reference, model tooling |
| [`release/georefine-verify/`](release/georefine-verify/) | `georefine-verify` 1.1.2 | Independent CPU bit-exact verifier and BF16 exporter |

Install in an isolated environment with the hardware and Python dependencies
listed in each package's README. From this source checkout:

```sh
python -m pip install ./release
python -m pip install ./release/georefine-verify
```

For the measured RTX path, use `FastSession.load` with the model bundle's
manifest pinned. The full [SM120 example](release/README.md) shows the call;
the model repository provides version-pinned wheels, source archives, and
checksums. A JIT build needs a CUDA toolkit, C++ compiler, Python headers,
and Ninja. The bundled tune is selected only for the measured RTX PRO 6000;
other GPU names fail closed until an explicit tune is supplied.

## Verification and boundaries

The `georefine-verify` package checks the bundle against Qwen's published
revision without relying on GeoRefine's serving engine. Runtime receipts in
[`proofs/qwen38-27b/`](proofs/qwen38-27b/) retain original receipt digests;
only machine-local paths were removed for publication. The eight prompts and
one synthetic image establish a narrow runtime result, not a general
capability certificate or a claim of equal speed on every device. The model
card identifies additional unverified contexts and platforms.

The parent [Qwen3.8-27B model](https://huggingface.co/Qwen/Qwen3.8-27B) is
Apache-2.0. Its LICENSE and notice are retained with the encoded model.
GeoRefine's source packages are Apache-2.0 as well.

## Predictive BF16 packages

Predictive packages combine BCTX frames with reference-conditioned PPCX frames.
Readers verify the package inventory, reference dependencies, and tensor hashes.
The CUDA loader keeps weight frames compressed and decodes matrices for model
operations. Decoder caches live outside the model package.

Verify a package and restore its original checkpoint files:

```bash
python -m scripts.bitexact_predictive_package verify /path/to/package
python -m scripts.bitexact_predictive_package restore /path/to/package /path/to/restored
```

Serve the package through the chat-completions API:

```bash
python -m scripts.serve_bitexact_predictive \
  --package /path/to/package --loader-workers 4 \
  --host 127.0.0.1 --port 8000
```

GPU serving uses CUDA-enabled PyTorch, Triton, Transformers, and Accelerate.
The HTTP endpoint also uses FastAPI, Uvicorn, Pillow, and PyAV. Use a separate
Python environment for these dependencies. Set `BITEXACT_PREDICTIVE_CACHE_DIR`
to choose the external native-build cache location.

The Python interface exposes `load_predictive_model` and `load_predictive_mtp`
in `scripts.bitexact_predictive_serving`. The MTP head can be used with
`MTPSpeculator` from `scripts.bitexact_context_mtp`.

To generate candidate frames and build a package from a BCTX package, see
`python -m scripts.bitexact_predictive_package --help`. Candidate generation
checks exact reconstruction and resumes from receipts whose hashes match.
