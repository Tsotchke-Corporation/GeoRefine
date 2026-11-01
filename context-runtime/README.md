# GeoRefine Context codec v2 — Linux CUDA

This source matches the [public Linux release](https://huggingface.co/Tsotchke-Corporation/Qwen3.8-27B-GeoRefine-Context/tree/v2-linux-20261101). Runtime package version: `0.1.1+cu129`. Codec generation and Python distribution versions are distinct.

All 234 runtime source entries retain the frozen `49c8e64604bb6209fde937914943b95e87f2eeed1e6c76b9d5da21d7009d50ff` manifest. No runtime arithmetic, serialization or CUDA source changed for this GitHub snapshot. Private development history and model weights are not part of the export.

Use a fresh Python3.10 environment and CUDA12.9/C++17 toolchain. The source pins Torch2.9.1+cu129, TorchVision0.24.1+cu129 and Transformers5.5.0.

From the repository root, reproduce the exact released wheel:

```sh
python context-runtime/tools/build_release.py --profile cuda129 --check-twice \
  --source-date-epoch 1793253060 --out .scratch/context-v2-build
```

The explicit epoch is the original artifact build epoch, not a changed Git commit date. Expected wheelSHA256: `e64de51241c387efbdbc63f7185985b6106e5310350eb9540415dc3a6339e752`. Staging is retained; use a fresh output namespace for another build.

See [measurements](../proofs/context-linux-v2/summary.json) and [release guide](../docs/context-v2-linux-release.md). Metal/Windows remain unqualified. The original source-distribution README is retained in `docs/context-0.1.1-original-source-readme.md` as historical documentation; old candidate/gating language there is not the current Linux release status.
