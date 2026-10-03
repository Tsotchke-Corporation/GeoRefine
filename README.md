# GeoRefine codec — v1.2.0-rc2 (multi-user serving preview)

GeoRefine is developed by **Tsotchke Corporation**.
Copyright 2026 Tsotchke Corporation. The codec, runtime, and verifier source are
licensed under Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). The Qwen
model weights retain their upstream license and attribution.

---

## On Windows, start here

**[README-WINDOWS.md](README-WINDOWS.md)** — the whole procedure for an NVIDIA card
in a Windows machine, for someone who does not want to debug a toolchain. One
PowerShell line, then three lines in Ubuntu. WSL2 is the supported path and the page
says plainly what is and is not verified on native Windows.

## Two commands

```bash
pip install "glc-loader[cuda] @ https://github.com/Tsotchke-Corporation/GeoRefine/releases/download/v1.2.0-rc2/glc_loader-1.2.0rc2-py3-none-any.whl"
```

```bash
glc-bench
```

Keep the double quotes: the `[cuda]` is a shell glob in `zsh` and a wildcard pattern
in PowerShell.

That is the whole thing. `glc-bench` takes **no required flags**. It checks your
driver and torch, downloads the model, runs the correctness gates, benchmarks our
server against `llama.cpp`, prints one table, and writes
`glc-bench-results-<date>.tar.gz`.

**Send that tarball back.**

---

## What you'll see

Roughly, in order:

1. An environment block (card, driver, torch, free disk). If something is missing it
   stops right there and prints the one command that fixes it.
2. A ~40 GB model download. This is the slow part. It resumes if interrupted.
3. Four correctness gates, each printing `PASS` or `FAIL` **and how many output
   elements it compared** — a pass on a small sample is not a pass on a large one, so
   the count is always shown.
4. The benchmark: 1, 4, 8 and 16 concurrent users on short prompts, then 16 and 32
   users on 8k-token prompts, against our server and then against `llama.cpp` with
   matched continuous batching.
5. One table, `GLC vs llama.cpp`: aggregate tokens/sec, per-user tokens/sec, TTFT p50
   and p99, whether the median stream met the latency budget, $/M output tokens, and
   steady-state VRAM — for both engines.

Every number is labelled `MEASURED-ON-THIS-CARD`. Anything that could not run prints
`NOT RUN` with the reason; nothing is filled in from elsewhere.

Expect roughly 1–3 hours end to end, most of it the download and the gates.

---

## Troubleshooting

The three things that actually go wrong:

| Symptom | Fix |
|---|---|
| `torch is installed but reports no CUDA device` | You have a CPU-only torch wheel. `pip uninstall -y torch && pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| `llama-server is not on PATH` | The run continues with our side only. For the comparison, build llama.cpp (`glc-bench` prints the three commands), then re-run with `--llama-server <path> --llama-gguf <model.gguf>` |
| Out of memory, or the server dies during load | `glc-bench --slots 16` (and `--slots 8` if that still OOMs) |
| Not sure the machine is set up at all | `glc-bench --windows-check` — every check except executing a CUDA kernel: platform, WSL detection, torch, driver, `nvcc`, host compiler, free disk, install, plumbing. Downloads nothing. |

`glc-bench --help` lists everything else. Two flags worth knowing:
`--dollars-per-hour 1.23` fills in the $/M output column at the rate you are actually
paying, and `--llama-gguf <file>` supplies the llama.cpp baseline.

---

## What this is

GeoRefine TBE stores pretrained BF16 weights in a reversible encoded format and
serves them without keeping a dense copy of every coded weight resident. The
compression is **bit-exact**: the weights handed to the model are the same bytes the
source checkpoint had, and the verifier proves that on your machine from the artifact
alone. For Qwen3.8-27B the complete bundle is 39,832,462,895 encoded tensor bytes
against 55,562,855,904 BF16 tensor bytes (1.395× smaller), with 1,199/1,199 tensors
and 10/10 sidecars matched by the independent verifier.

**v1.2.0-rc2 adds the multi-user path**: a batched decoder with continuous batching,
paged KV, and a batch-invariance guarantee — a stream's output must not depend on who
else happened to be in the batch with it. The gates in `glc-bench` are what check
that claim.

**What changed in rc2.** rc1 was verified on macOS CPU only and could not be
installed at all on a Windows machine: the `[cuda]` extra required `triton`, which
publishes no Win32 wheel, so `pip install` failed before anything else could be
tried. rc2 fixes that (the Triton requirement is now Linux-only, and nothing in
`glc-bench` imports it), passes MSVC's `/O2` rather than GCC's `-O3` to the host
compiler in the JIT kernel builds, sets `TMP`/`TEMP` as well as `TMPDIR` so Windows
temporaries do not spill onto the system drive, stops child servers with
`terminate()` instead of a POSIX signal, detects WSL2 and native Windows in
preflight, prints the 40 GB artifact size and the free space **before** downloading,
and adds `glc-bench --windows-check`: every check up to but not including CUDA
execution. Native-Windows wheel install, `--cpu-smoke` and `--windows-check` are
checked in CI on `windows-latest` (no GPU there, so no kernel and no gate).
Serving and codec behaviour are unchanged from rc1.

**Be clear about the status.** This is a release candidate. The single-user
`FastSession` path from v1.1.1 is unchanged and has been measured on an RTX PRO 6000
Blackwell. The multi-user path has passed its CPU reference gates, and the four GPU
gates `glc-bench` runs **have never been run on a GPU before**. Your card is where
they first execute. A `FAIL` is a real result and we want the tarball either way. We
have not measured our multi-user throughput against `llama.cpp` on any card yet, so
this release ships no such claim — the table `glc-bench` prints is the first one.

The bundled kernel autotune table is measured on SM120 (RTX PRO 6000). On a different
architecture it is a starting point, not a tuned configuration, and the throughput
numbers should be read with that in mind.

---

## Packages

| Path | Distribution | Purpose |
|---|---|---|
| [`release/`](release/) | `glc-loader` 1.2.0rc2 | TBE loader, CUDA FastSession, batched multi-user server, CPU reference, benchmark |
| [`docs/serving/MULTIUSER_SERVING.md`](docs/serving/MULTIUSER_SERVING.md) | — | The engine audit, the exactness-gate design, and the measurement protocol |
| [`README-WINDOWS.md`](README-WINDOWS.md) | — | WSL2-first Windows procedure, with a verified/unverified table for native Windows |

Console scripts installed by the wheel:

| Command | What it does |
|---|---|
| `glc-bench` | the one-command benchmark described above |
| `glc-loader info \| verify \| load` | inspect and bit-exactness-verify an artifact |
| `glc-serve` | single-user HTTP server (`FastSession`, as in v1.1.1) |
| `glc-multiuser` | the batched multi-user server, run directly |

Model weights live in the
[Qwen3.8-27B GeoRefine TBE model repository](https://huggingface.co/Tsotchke-Corporation/Qwen3.8-27B-GeoRefine-TBE).
`glc-bench` pulls revision `2725c98a0b152b3092a9f5c692e5b28729d6c052` by default, so
two runs on two machines compare the same bytes.
