# Context codec v2 — Linux CUDA release

The codec source belongs on GitHub; the large model weights belong on Hugging Face. This release links the two with the same immutable artifact checksums.

- Model and published runtime: [v2-linux-20261101](https://huggingface.co/Tsotchke-Corporation/Qwen3.8-27B-GeoRefine-Context/tree/v2-linux-20261101)
- Python distribution: `georefine-context-fast-runtime` `0.1.1+cu129`
- Codec source: [`context-runtime/`](../context-runtime/)
- Measured scope: [`summary.json`](../proofs/context-linux-v2/summary.json)
- Frozen benchmark configuration: [`benchmark-config.json`](../proofs/context-linux-v2/benchmark-config.json)
- Source export provenance: [`source-snapshot.json`](../proofs/context-linux-v2/source-snapshot.json)

The complete package/reference storage ratio is 1.530696×. Blackwell96 measurements show per-request token-generation decode ratios 1.2106×/1.4197× at64/128users and complete-window aggregate ratios 1.2333×/1.3412× versus the same-engine BF16 reference. Candidate natural MTP is enabled (`spec_k=1`); reference MTP is disabled. Each cell measures three sequential requests per worker following excluded warmup. These are combined codec/runtime/batching/MTP results; no isolated codec-only speedup is asserted.

A100 resident-worker functional/capacity measurements are documented separately, including its ~907s cold startup and the failed compiler-observer classification. Independent HF-parent runtime parity, universal hardware certification and Metal/Windows qualification are not claimed. See the public model card for the full conditions and limits.

## Download and install

Use the exact wheel from either the GitHub release assets or the Hugging Face tag. For a fresh Python3.10 environment with a CUDA12.9 toolkit, C++17 compiler and development headers:

```sh
python3.10 -m venv .venv-context
. .venv-context/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu129 \
  ./georefine_context_fast_runtime-0.1.1+cu129-py3-none-any.whl
```

The wheel pins Torch2.9.1+cu129, TorchVision0.24.1+cu129 and Transformers5.5.0. Model download, weight verification/restoration and the guarded Blackwell launcher are documented on the linked model card. V1 `glc-loader` and the Context runtime have overlapping import namespaces; install them in separate environments.

## Artifact identities

| Artifact | SHA-256 |
|---|---|
| Matched CUDA12.9 wheel | `e64de51241c387efbdbc63f7185985b6106e5310350eb9540415dc3a6339e752` |
| Frozen original source distribution | `d45ef9f97ce7366b7c9586021f44422be8441ad6640169ded5d21f95d8e9dce4` |
| Runtime source manifest | `49c8e64604bb6209fde937914943b95e87f2eeed1e6c76b9d5da21d7009d50ff` |
| Model package manifest | `bbac23fdeba3020f8951d7a0a005016985c40add08979f37a85df2f22ccbacee` |

The GitHub source snapshot introduces no runtime arithmetic, serialization or CUDA changes. All234 installed runtime entries are checked against the frozen manifest. Public source comes from the already-published source distribution, without private development history, credentials or weights. Generated packaging metadata and unrelated test harnesses are not part of this curated tree; the source-export manifest records the selection. Six CPU codec test modules are included.

## Reproduce the wheel

```sh
python context-runtime/tools/build_release.py --profile cuda129 --check-twice \
  --source-date-epoch 1793253060 --out .scratch/context-v2-build
```

The explicit epoch preserves the original artifact timestamp while new Git commits retain their actual dates. The build checks every wheel RECORD entry, all234 runtime manifest rows and byte equality between two builds. Build directories are retained; an existing staging directory is refused rather than removed. Use a fresh output namespace for another build. Documentation-only edits may change a newly generated source archive; the original source archive above remains separately pinned.

## CPU source checks

With the test dependencies installed in the development environment:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/context-runtime" \
  TMPDIR="$PWD/.scratch/tests" python -m pytest -q context-runtime/tests
```

Create the persistent `.scratch/tests` directory first. This command covers the selected CPU codec modules; it does not claim a fresh GPU run or the larger historical qualification suite. The measured GPU receipts remain preserved under their original scope and status.
