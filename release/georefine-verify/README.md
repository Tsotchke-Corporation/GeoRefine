# GeoRefine TBE tools

Copyright 2026 Tsotchke Corporation. This verifier and exporter are licensed
under Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).

`georefine-verify` independently compares a GeoRefine TBE bundle with the
published parent model, byte for byte. `georefine-export` restores a TBE bundle
to ordinary Hugging Face safetensors without downloading or reading the parent.
Both tools use NumPy and the Python standard library, and run on CPU.

```sh
pip install .
georefine-export --bundle Tsotchke-Corporation/Qwen3.8-27B-GeoRefine-TBE \
  --output ./Qwen3.8-27B-GeoRefine-BF16 \
  --expect-manifest-sha256 10028416e07c802b47e4dc93a86e9f0dd35ce6021f0e07b96c70bd9cb7a72bc3
```

The output has `model.safetensors.index.json`, standard safetensors shards,
and the original config, tokenizer and vision files. Load it with a current
Transformers release compatible with the parent model:

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor
model = AutoModelForMultimodalLM.from_pretrained("./Qwen3.8-27B-GeoRefine-BF16")
processor = AutoProcessor.from_pretrained("./Qwen3.8-27B-GeoRefine-BF16")
```

The export is BF16: plan for about 56 GB of output storage, plus one codec shard
of scratch space when downloading from the Hub, and enough memory or GPU
capacity to run Qwen3.8-27B. This path establishes portable model compatibility;
it does not retain the compressed model's runtime memory or speed advantages.
The native TBE engine is a separate optimized backend. A full 27B export and
Transformers load remain release gates, beyond the existing small-fixture tests.
