# GeoRefine Context customer quickstart

This is a private candidate canary. It is unqualified for production and makes no performance promise.

The qualification target is Linux x86_64, one NVIDIA RTX PRO 6000 Blackwell 96 GB GPU, 48 CPU cores and 180 GB host RAM. Use Python 3.10, its development headers, a C++ compiler, and the CUDA 12.9 toolkit (including `nvcc`) in an isolated virtual environment. GPU validation of this candidate is pending. From the GitHub checkout, install the complete Context runtime in this directory:

```bash
python3.10 -m venv .venv-context
source .venv-context/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install './context-runtime[cuda]'
python -m pip install 'huggingface_hub>=0.30'
hf --help >/dev/null
```

The separate release candidate artifacts include a source archive and wheel. This directory contains the complete source needed to build the same runtime. Preserve its Apache-2.0 license and bundled third-party notices when redistributing.

Checksums: source archive `5e049643b2d0f42741cba0790247cecbd3dc88e94ee8edc3bf30a0c21c1d2ace`; wheel `fd4b3efd142c4edba03c2ce6e8e7a021a5b147150eca32b5225d887ea4d88ebf`.

Download the pinned private candidate model:

```bash
hf download Tsotchke-Corporation/Qwen3.8-27B-GeoRefine-Context \
  --revision f8c6b089ae0a68de9596bc9086befb6f771cd877 \
  --local-dir ./context-package
sha256sum context-package/manifest.json  # expected bbac23fdeba3020f8951d7a0a005016985c40add08979f37a85df2f22ccbacee
```

Verify and restore the mixed BCTX/PPCX package when needed. Restore requires roughly 56 GB of additional disk space. The Context download is approximately 36.31 GB versus 55.56 GB of original BF16 tensor data (1.53× smaller on disk). Serving transcodes Context frames into TBE weights during startup. The storage ratio is not a GPU-memory ratio.

```bash
georefine-predictive-package verify ./context-package
georefine-predictive-package restore ./context-package ./restored-parent
```

Use the installed tune (`glc_serve/tunes/fastdec_sm120.json`) and start the exact Context server:

```bash
TUNE=$(python -c 'from importlib.resources import files; print(files("glc_serve").joinpath("tunes/fastdec_sm120.json"))')
georefine-context-fast-serve --package ./context-package --tune "$TUNE" \
  --device cuda:0 --loader-workers 4 --verified-cpu-stream \
  --cpu-vision-dense --cpu-encode-workers 16 --source-inflight-bytes 8589934592 \
  --host 127.0.0.1 --port 8000 --max-context 8960
```

Use the conservative candidate context until GPU qualification completes. A local smoke request is:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"context-fast","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":16,"temperature":0}'
```

The canary and its GPU qualification remain separate; do not infer readiness from startup or this quickstart.
