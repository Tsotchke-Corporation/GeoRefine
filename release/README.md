# glc-loader

Read, verify and serve GLC compressed model artifacts — with no checkout of
the repository that built them.

The compression is **bit-exact**: the weights this package hands back are the
same bytes the source checkpoint had, and `glc-loader verify --deep` proves
that on your machine, from the artifact alone. Runtime speed depends on the
backend and hardware; see [Hardware reality](#hardware-reality).

---

## Install

```bash
pip install glc-loader
```

That is enough to inspect, verify, expand and CPU-load any artifact. Two
optional extras add accelerated serving; **neither is required to import the
package**, and neither is a dependency of the other:

```bash
pip install "glc-loader[cuda]"    # + triton and Ninja for CUDA JIT backends
pip install "glc-loader[metal]"   # + mlx, mlx-lm, for Apple Silicon
pip install "glc-loader[qwen38]"  # Qwen3.8 multimodal runtime and processor
```

Requires Python ≥ 3.10, `torch` ≥ 2.1, `safetensors` ≥ 0.4, `transformers`
≥ 4.56.

*Tested against* torch 2.6.0, safetensors 0.8.0, transformers 5.13.0 on
macOS/arm64, Python 3.11. The version floors above are **declared**, not
measured — they are the lowest releases whose APIs this code uses, not the
lowest that have been run.

---

## The three commands

### `glc-loader info <artifact-dir>`

What the artifact is, where it came from, its ratios **at all three scopes**,
and — the part that usually goes missing — what it does **not** certify.

```bash
glc-loader info ./my-artifact
```

Ratios come back as three separate blocks, each with its own `basis` string
and a `status` of `measured` or `unmeasured`. A scope this artifact does not
carry comes back `null`. It is never filled in from another scope:

| scope | what it counts |
|---|---|
| `stored` | bytes on disk |
| `served_resident` | bytes the device holds for weights |
| `whole_process` | measured peak process VRAM, dense vs coded — the only one that includes activations, the KV cache and allocator slack |

`whole_process` is normally `unmeasured`. The transcode step leaves it blank
by construction; it is filled only by a certify run on the target card. If
`info` tells you it is unmeasured, that is the truth, not a gap in the tool.

### `glc-loader verify <artifact-dir>`

Integrity, with exit codes you can put in CI:

| exit | meaning |
|---|---|
| 0 | verified |
| 1 | an integrity check **failed** — a digest or a decoded tensor did not match |
| 2 | malformed, unreadable, or not a GeoRefine container at all |
| 3 | verified bit-exact, but the artifact **expands** (stores more bytes than the dense weights it replaces) |
| 4 | the format is recognised but this directory **cannot be verified from what it ships** |

```bash
glc-loader verify ./my-artifact          # digests
glc-loader verify ./my-artifact --deep   # + decode every container
```

Without `--deep` this checks digests only, and says so in the output. With
`--deep` it decodes every stored container and checks the result against the
source digest recorded before encoding — an end-to-end proof, reproducible on
your machine, that the bytes this artifact produces are the bytes the source
model had.

**Exit 4 is deliberate.** A `glc_tbe_transcode` v1 directory is a correct
artifact with no self-contained verifier: it ships a manifest and two tensor
blobs, and nothing binds the manifest to the blobs, so checking one against
the other would be checking the manifest against itself. Reporting that as
exit 2 ("malformed") would be a lie about a well-formed artifact, and
reporting it as 0 would be a lie about what was checked.

### `glc-loader load <artifact-dir>`

Smoke-load the artifact, generate a few tokens, print the serving receipt.

```bash
glc-loader load ./my-artifact --device cuda --prompt "The capital of France is"
```

This is a smoke test, not a benchmark. The `tokens_per_second` it prints is
one greedy run on whatever device you gave it.

Two more commands exist for `GLC-RELEASE/1` artifacts specifically:
`glc-loader expand --out ./dense` writes a plain Hugging Face checkpoint for
any other engine, and `glc-loader generate --prompt ...` is the older,
more configurable generation path.

---

## From Python

```python
from glc_loader import load_model

model, tokenizer, receipt = load_model("./my-artifact", device="cuda")
print(receipt["backend"])
```

For a `georefine.tbe.v2` container:

```python
from glc_loader import load_standalone, single_device_map

model, tokenizer, receipt = load_standalone(
    "./my-artifact", device_map=single_device_map("cuda:0"),
)
```

---

## Hardware reality

### Qwen3.8-27B serve-v1 compatibility prototype

The `georefine.tbe.serve.v1` bundle has a source-free compressed PyTorch path:

```python
from glc_loader import load_compressed_transformers

model = load_compressed_transformers(
    "Tsotchke-Corporation/Qwen3.8-27B-GeoRefine-TBE",
    expect_manifest_sha256="10028416e07c802b47e4dc93a86e9f0dd35ce6021f0e07b96c70bd9cb7a72bc3",
)
```

Install `glc-loader[qwen38]` for the required Qwen Transformers class and Hub
download support. The bundle remains encoded in safetensors arrays (`planes`,
`smb`, `esc`, `sbbase`). `PortableTBELinear` decodes one weight transiently per
call, performs a PyTorch linear operation, and discards the dense weight. This
is a correctness and portability fallback, **not** the measured fused-kernel
speed path. The base Qwen model's 1,184 tensors map exactly to the manifest;
the remaining 15 MTP weights are retained for native speculative inference.
The full 27B graph loaded on Linux CPU and produced finite text and minimal
image logits while retaining encoded weights. Full generation on this CPU
reference path is unverified, and its forward speed is not a serving claim.

**Decode speed is a property of the engine, not of the format.** The same
GLC-TBE weights have been measured on two decode paths:

| decode path | card | measured scope |
|---|---|---|
| portable PyTorch reference | Linux CPU | full 27B text/image forwards passed; too slow for serving |
| standard `glc-serve` HTTP, exact | Blackwell RTX PRO 6000 | 7.99 vs dense 17.50 tok/s end to end; 8/8 text + image parity |
| standard `glc-serve` HTTP, fused | Blackwell RTX PRO 6000 | 14.95 vs dense 17.50 tok/s; 5/8 text parity |
| `FastSession` whole-step CUDA graph, **bundled SM120 tune** | Blackwell RTX PRO 6000 | **37.880 vs dense 28.288 decode tok/s (1.339×)**; 8/8 text token-ID + image parity |

The standard HTTP and `FastSession` rows use eight fixed greedy prompts and
one synthetic image on one card. The `FastSession` result was reproduced from
an installed wheel and its packaged tune; it is not a cross-platform speed
guarantee or a general capability certificate.

Earlier standalone-loader measurements were **0.94×** dense on RTX PRO and
**0.57–0.62×** on A100. Those were single runs with roughly ±0.05 spread on a
different decode path; they do not measure the packaged `FastSession` engine.

The engine row is Qwen3.8-27B, TBE bundle, batch 1, greedy AR on one card
(receipts in `KERNEL_SPEED_20260925.md`): **37.84 tok/s**, against 28.29 tok/s
for the bf16 parent in the same engine and 27.4–27.6 tok/s for llama.cpp bf16
(tg128) on the same card. Peak VRAM (NVML) was 49.9–50.0 GB, against
65.4–72.8 GB for the dense engine. The engine's output is bitwise equal to the
bf16 parent on its gate probes (G3a 711/711 rows).

Memory is the saving that holds on every path. For orientation, one real
artifact (`unsloth/Llama-3.2-1B-Instruct`, GLC-FWP1) reports a weight ratio of
**1.279×** at the served-graph scope — 1.93 GB resident where the dense
checkpoint needs 2.47 GB. Your artifact's own numbers are in
`glc-loader info`; do not carry that one across.

On the tested RTX PRO 6000, choose `FastSession` for the measured fast path.
The standard HTTP route and portable CPU path have different speed contracts.

```python
from huggingface_hub import snapshot_download
from glc_serve.fastserve import FastSession

bundle = snapshot_download("Tsotchke-Corporation/Qwen3.8-27B-GeoRefine-TBE")
session = FastSession.load(
    bundle=bundle, tune=None,  # bundled tune: RTX PRO 6000 Blackwell only
    server_flags=["--gate", "full", "--manifest-sha256",
                  "10028416e07c802b47e4dc93a86e9f0dd35ce6021f0e07b96c70bd9cb7a72bc3"],
)
state = session.prefill(messages=[{"role": "user", "content": "Hello"}])
ids = list(session.generate(state, max_tokens=48, temperature=0.0))
print(session.tokenizer.decode(ids, skip_special_tokens=True))
```

The public Hugging Face manifest hash above differs from the internal GCS
source manifest because private build paths were removed. Allow room for the
40 GB bundle and about 50 GB peak GPU memory on the measured SM120 path. An
RTX PRO 6000 is the only card whose tune and full-model speed have been
verified in this wheel; other devices need an explicit tune and separate
correctness and performance measurements.

---

## CUDA package scope

This wheel includes the TBE MMA source (`tbe_mma_kernel.cu`), `FastDecoder` and MIV-TBE CUDA
sources, plus the measured RTX PRO 6000 SM120 tune. They JIT-build in a user
cache; a CUDA toolkit, C++ compiler, Python headers and Ninja are required.
`FastSession.load` selects the bundled tune only on the measured SM120 card.
Other GPUs require an explicit tune and their own exactness and speed tests.
The experimental FWP1 Triton source is still artifact-local (`fwp1_kernels.py`) and is not in
this wheel; that optional backend refuses or uses its documented fallback.

---

## Which formats this reads

| format | self-contained? | `verify` | `load` |
|---|---|---|---|
| `GLC-RELEASE/1` (FWP1) | yes | digests, and `--deep` bit-exactness | yes |
| `georefine.tbe.v2` | yes | `SHA256SUMS`, and `--deep` bit-exactness | yes, on a supported device |
| `glc_tbe_transcode` v1 | **no** — no config, no tokenizer, no format tag | exit 4, with the reason | refused |
| anything else | — | exit 2 | refused |

Format is decided by a **declared tag** — `MANIFEST.json`'s `"format"`, or
`compression_info.json`'s `"artifact_format"` — never by which files happen to
be on disk. A plain Hugging Face checkpoint ships `config.json` and
`model.safetensors`, and a loader that inferred "container" from those would
claim every model on the hub. Point this at an ordinary checkpoint and it
declines, by design.

---

## Licence

The compressed weights inherit the **source model's** licence. `glc-loader
info` reports whatever licence the artifact declares or ships, and reports
`null` with a note when it declares none — which is the common case today.
Check the source model before redistributing.

The `glc-loader` source is Copyright 2026 Tsotchke Corporation and licensed
under Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). The compressed
weights retain the source model's license and attribution.
