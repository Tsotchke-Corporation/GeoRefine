# glc-loader

Read, verify and serve GLC compressed model artifacts — with no checkout of
the repository that built them.

The compression is **bit-exact**: the weights this package hands back are the
same bytes the source checkpoint had, and `glc-loader verify --deep` proves
that on your machine, from the artifact alone. What it buys you is memory. It
does not buy you speed — see [Hardware reality](#hardware-reality), which is
the first thing you should read if you are deciding whether to use this.

---

## New in 1.2.0rc1: multi-user serving, and one command that measures it

```bash
pip install "glc-loader[cuda] @ git+https://github.com/Tsotchke-Corporation/GeoRefine.git@v1.2.0-rc1#subdirectory=release"
glc-bench
```

`glc-bench` has **no required flags**. It checks the driver and torch, downloads the
pinned GLC-TBE artifact, runs the four correctness gates, benchmarks the batched
server against `llama.cpp` with matched continuous batching at 1/4/8/16 users on short
prompts and 16/32 users on 8k-token prompts, prints one table, and writes
`glc-bench-results-<date>.tar.gz`.

Every printed figure is labelled `MEASURED-ON-THIS-CARD`; gates print PASS/FAIL with
the number of output elements compared; anything that could not run prints `NOT RUN`
with the reason. `glc-bench --cpu-smoke` checks the install on a machine with no GPU
and measures nothing else. `glc-bench --help` documents the rest.

The multi-user path is a release candidate and its GPU gates have not run on any card
yet. The single-user `FastSession` path is unchanged from 1.1.1.

---

## Install

```bash
pip install glc-loader
```

That is enough to inspect, verify, expand and CPU-load any artifact. Two
optional extras add accelerated serving; **neither is required to import the
package**, and neither is a dependency of the other:

```bash
pip install "glc-loader[cuda]"    # + triton, for the FWP1 GEMV backend
pip install "glc-loader[metal]"   # + mlx, mlx-lm, for Apple Silicon
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

**This format is a VRAM lever, not a speed lever.** Decode throughput has been
measured at:

| card | decode, relative to dense |
|---|---|
| Blackwell RTX PRO 6000 | **0.94×** |
| A100 | **0.57–0.62×** |

Those are single runs with roughly ±0.05 spread. They are not averages over
repeated trials and should not be quoted as though they were. Slower than
dense on both cards; on the A100, substantially so.

What you get in exchange is memory. For orientation, one real artifact
(`unsloth/Llama-3.2-1B-Instruct`, GLC-FWP1) reports a weight ratio of
**1.279×** at the served-graph scope — 1.93 GB resident where the dense
checkpoint needs 2.47 GB. Your artifact's own numbers are in
`glc-loader info`; do not carry that one across.

Run this only if holding a larger model on a smaller card is worth spending
decode time on. If throughput is your constraint, it is not.

---

## What is *not* in the wheel

The accelerated CUDA kernels are **not** shipped by `pip install glc-loader`.
`fwp1_kernels.py` (the Triton FWP1 GEMV) and `tbe_mma_kernel.cu` (the TBE
fragment kernel, JIT-built by `torch.utils.cpp_extension`) are vendored beside
the loader at artifact-build time and live only inside a built
`GLC-RELEASE/1` artifact. Installing `[cuda]` gets you `triton`, but the
kernel source still has to come from an artifact.

Both consumers refuse by name rather than guessing:
`container.load_kernels()` returns `None` and the loader falls back to a
pure-torch backend that produces the same bits, and
`tbe_mma_kernels` raises a typed error naming the missing file. Nothing
silently degrades.

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

This package is proprietary to Tsotchke Corporation.
