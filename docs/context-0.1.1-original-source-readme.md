# GeoRefine Context customer quickstart

Context 0.1.1 is an unqualified release candidate. GPU startup, HTTP concurrency and parent-parity qualification remain pending; it makes no performance promise.

`GEOREFINE_TBE21_NATIVE_PACK=1` selects an experimental, single-thread CPU packer
after the existing Python codebook and selector decisions. Unset or `0` retains
the original encoder and launches no packer compiler. The native path preserves
the serialized codec bytes and the independent NumPy CRC/SHA round-trip check.
It records source, compiler, flags and binary hashes outside the codec header;
only compiled code is cached, under the explicit startup `.scratch` build root
or `GEOREFINE_TBE21_NATIVE_CACHE`. Selected build/packing errors fail closed.
Qualification requires an explicit `context_v21_transcode` source `cpu_encoder`
binding: mode `native_tbe21`, all three encoder source hashes, ABI1, one thread
and the fixed Linux/macOS compiler flags. The server command must explicitly
select the matching environment flag. Unbound or mismatched selection fails
before model restoration; native provenance is also rejected for an unbound
source after the flag is cleared. Every coded tensor must carry a matching
producer receipt, including its loaded binary hash and compiled C++ source-hash
getter. Portable validators check the persisted, hash-bound producer evidence;
they do not access remote binary paths. This has synthetic CPU proof only; no
model startup or accelerator performance claim applies.

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

The separate release candidate artifacts include a source archive and wheel. This directory contains the complete source needed to build the same runtime. `python context-runtime/tools/build_release.py --check-twice` builds the wheel and sdist from a clean `git archive` export of HEAD into `.scratch/dist`, checks the wheel RECORD against `georefine_context_runtime/SOURCE_MANIFEST.json`, confirms two builds are byte-identical and prints both SHA-256 digests. Do not build from a working tree: stale `build/` or `*.egg-info` directories there can add modules that are not in the source. Preserve its Apache-2.0 license and bundled third-party notices when redistributing.

The explicit `--profile cuda129` distribution uses version `0.1.1+cu129` and
pins Torch `2.9.1+cu129` with Torchvision `0.24.1+cu129`, matching the Linux
CUDA 12.9 operator qualification family. These local-version dependencies
come from the [official PyTorch CUDA 12.9 index](https://download.pytorch.org/whl/cu129/).
Use a new isolated environment and that index for the pinned framework wheels.
The default `cuda` and explicit `mlx` distributions retain their existing pins.
This build profile is distribution metadata only; it grants no GPU execution,
platform certification or release qualification. The frozen V4 operator wheel
remains a distinct artifact and its measurements must retain its original hash.

Build-specific checksums belong to the qualification receipt for the wheel being tested. Earlier 0.1.0 wheel checksums do not identify this 0.1.1 source.

The opt-in `generic_tensor_codec` stores arbitrary tensor payload bytes with an opaque dtype descriptor, shape and optional packed element width. The separate `typed_context_package` family preserves original safetensors headers, assets and tensor bits, including mixed BF16/FP32 weights, integers, buffers and future storage dtype tokens. Its restore command reconstructs the original files exactly. Existing predictive packages and BF16 kernel formats retain their wire behavior.

Native materialization requires the destination framework to support the source dtype; unsupported types remain losslessly stored and fail explicitly when requested for execution. Typed Torch loading is available through `typed_context_serving`; verified MLX loading and `mlx_engine.load_engine` accept `package_format="typed"`. BF16 compressed Metal linears require original BF16 weights. Other supported source tensors retain their dtype without numerical conversion. These paths remain unqualified until their source-bound platform receipts pass.

Batch startup can opt into `--capture-workload-only`, which captures graph buckets through `max_users * (spec_k + 1)` while retaining the configured row buffers. Larger uncaptured buckets execute the same eager body once. The default captures the existing full bucket set. This startup option does not reduce native kernel qualification coverage.

`--reuse-package-validation` requires `--verified-cpu-stream --verify-package full`. It reuses a live resolver only within the same process and unchanged package inventory. Full package hashes still run at resolver entry and close; the middle loader check retains structural validation and non-frame hashes, and every tensor read retains frame and source SHA checks. The default performs all three full verification passes.

Speculative startup cost calibration can explicitly select `--spec-calibration-repeats` in the supported range 3 through 20. The default remains five CUDA-event samples per captured padding graph. The calibration receipt records the actual sample count and medians; release startup and performance thresholds remain separate checks.

With workload-limited capture, cost calibration still covers every planner bucket through the configured row capacity. It measures uncaptured buckets with the same eager body on padding rows before accepting requests, and records graph/eager execution in the calibration receipt.

With the native CPU packer and parallel Context startup explicitly selected,
preparation streams bounded same-layer batches to the loader. It retains the
predictive-reference memo, reserves parent materialization and worker workspace,
and validates every transferred tensor before upload. Native round-trip decoding
retains CRC and reconstructed-source SHA checks. The default unselected path,
encoded format, and GPU arithmetic remain unchanged. This startup path requires
its own source-bound qualification; single-weight timings do not establish the
complete-model startup deadline.

Download the pinned private candidate model:

```bash
hf download Tsotchke-Corporation/Qwen3.8-27B-GeoRefine-Context \
  --revision 4c906d290aa5f6a3a06045710a6111d483768467 \
  --local-dir ./context-package
sha256sum context-package/package/manifest.json  # expected bbac23fdeba3020f8951d7a0a005016985c40add08979f37a85df2f22ccbacee
```

Verify and restore the mixed BCTX/PPCX package when needed. Restore requires roughly 56 GB of additional disk space. The Context download is approximately 36.31 GB versus 55.56 GB of original BF16 tensor data (1.53× smaller on disk). Serving transcodes Context frames into TBE weights during startup. The storage ratio is not a GPU-memory ratio.

```bash
georefine-predictive-package verify ./context-package/package
georefine-predictive-package restore ./context-package/package ./restored-parent
```

Use the installed tune (`glc_serve/tunes/fastdec_sm120.json`) and start the exact Context server:

```bash
TUNE=$(python -c 'from importlib.resources import files; print(files("glc_serve").joinpath("tunes/fastdec_sm120.json"))')
georefine-context-fast-serve --package ./context-package/package --tune "$TUNE" \
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

For continuous batching, use `georefine-context-batch-serve` with the same package, tune and CPU-stream flags. Its `/health` endpoint reports loading phases before requests are accepted. `--max-users` sets concurrent slots; GPU memory and batch invariance still require measurement on the target card.

The startup transcode cache is opt-in; `--transcode-cache auto` enables the standard cache directory. The first start transcodes the Context frames and writes the verified TBE containers, the dense input embedding and any dense vision tensors to `$GEOREFINE_CONTEXT_CACHE`, else `$XDG_CACHE_HOME/georefine-context/transcode`, else `~/.cache/georefine-context/transcode`; later starts with the same package, runtime version, GPU architecture and flags load them from there. The server logs the resolved directory at startup and, on a miss, the space the entry needs (about 0.75 of the BF16 text-weight bytes plus a 2 GB free-space reserve). If that directory cannot be read or written or lacks the space, the server logs one warning and serves without the cache. Pass `--transcode-cache DIR` to choose the directory, or `--transcode-cache off` (or `GEOREFINE_CONTEXT_CACHE=off`) to disable it. A directory under `/tmp` is accepted but logged as volatile. `--transcode-cache-verify full` additionally decodes cached containers against the original BF16 hashes. With `--verify-package full`, every package file is hashed at open and close; `auto` reports any cache-bound reduction in package reads explicitly. No warm-start timing is claimed until measured.

Both servers listen before the weights load. While loading, `/health` answers 503 with the current startup `phase`, its progress (`done`, `total`, `fraction`, `eta_s` where known) and the phases entered so far; chat requests get 503 until the server is ready. A failed start reports `startup_failed` with the failed phase on stdout and on `/health` for `--startup-failure-grace-s` seconds (default 10), then exits with status 2. Loader logs go to stderr; stdout carries one JSON status line per phase change.

Both server entry points build their required serving kernels before loading weights. `georefine-context-build-kernels --help` describes the separate warm-up command. Kernel build locks with live or unknown owners are preserved; use a separate cache or wait for the builder.

GDN recurrent state remains fp32. fp16 is rejected with `ValueError` because the CUDA recurrence kernel consumes `float*`; passing half storage would corrupt state. This validation preserves the original safety refusal and does not enable fp16.

## Experimental MLX serving

`georefine-context-mlx-serve` exposes the existing `/health` and
`/v1/chat/completions` interface (including SSE) through the verified MLX loader.
The dense Qwen3.5/3.8 text factory preserves source norm weights, transposes only
convolution layout, and loads the original MTP head when `--spec-k` is nonzero.
The default remains serialized text serving. `--multimodal` loads the complete
vision tower and enables image chat with verified processor metadata and
multimodal rotary positions. `--fused-batching --max-requests N` enables concurrent
HTTP requests through one model thread, merging independent KV/recurrent caches
for batched target and MTP operations. Rejected drafts replay only their accepted
prefix; cancellation, queue overflow and per-request failures release that request.
MoE, scaled RoPE, video HTTP input and non-greedy sampling remain explicitly
unsupported by this endpoint.

Pass `--package`, `--model-budget-bytes`, `--cache-budget-bytes`, and optionally
`--spec-k 1..7`. `--exclusions` names a JSON array of every source tensor omitted
from the text trunk (including MTP tensors loaded separately). Unknown source
tensors cannot disappear silently. `--panels` supplies a JSON object mapping
module paths to ordered `{ "path": "relative-file", "sha256": "..." }` BHE
panels; their decoded source hashes are checked against the verified model.
Without panels, the loaded model is dense BF16 and no resident compression is
claimed. `--device cpu` is for controls; BHE panels require `--device metal`.

The default loader refuses model budgets over 900 MiB. Larger explicitly
budgeted runs require `--allow-large-model`; source working memory is separately
bounded with `--source-working-budget-bytes`. Cache admission reserves KV,
recurrent state and rollback copies before prefill. These are array-payload
budgets, not guarantees about total process memory or allocator overhead.
`--max-batch-rows` (default 128, maximum 512) bounds fused verification rows.
Fused serving and BHE panels also bound prefill to that capacity. For otherwise
unchanged serial dense serving, opt into chunking with `--prefill-chunk-rows`;
MTP warmup and image positions continue across chunks.
`--vision-working-budget-bytes` bounds estimated image working arrays before vision
execution, separately from model residency and request caches. The HTTP queue
provides per-request delivery limits so a slow client cannot stall every request.

The integration tests use MLX 0.30.4 / MLX-LM 0.30.5 with Transformers 5.0.0rc3
in a separate environment. The standard CUDA package's Transformers 5.5.0 pin
has not been qualified with this MLX dependency profile. Tests use tiny random
CPU models and an independent Transformers Qwen3.5 reference; full checkpoint,
Metal execution, real-model image/state parity and throughput qualification remain pending.

Build an explicit MLX distribution with
`python context-runtime/tools/build_release.py --profile mlx --check-twice`.
This produces version `0.1.1+mlx` with MLX 0.30.4, MLX-LM 0.30.5 and
Transformers 5.0.0rc3 declared together. Install that wheel in a separate macOS
Apple Silicon environment; it is an alternative dependency profile of the same
runtime, not a second package to co-install with the CUDA distribution. Omitting
`--profile` preserves the existing CUDA package metadata. Reproducible packaging
and CPU tests do not establish fresh-environment installation or Metal performance.

For exactness qualification, add `--exact-digest` to the MLX server. Its streaming
and non-streaming responses then include `glc_exact` records with the token index,
token ID, dtype, vocabulary size, and the same 16-hex raw-logits SHA256 prefix used
by the CUDA qualifier. Hashes describe the target rows that chose emitted tokens;
a rejection replay cannot replace them with a more favorable hash. Nonfinite
logits fail the diagnostic run. Without the flag, logits are not copied or hashed.

MLX requests also accept the explicit boolean `ignore_eos` for fixed-token-count
qualification. Omitting it preserves ordinary EOS stopping. Terminal EOS tokens
still have diagnostic records even when the text completion usage excludes EOS.
The existing `qualification/context-0.1.1/batch_invariance.py` client can consume
this stream. Digest mode synchronizes and copies logits, so use a separate run
with it disabled for throughput and latency qualification.

`--panels <inventory.json> --metal-operator fused` explicitly selects the
experimental compressed-weight Metal operator. It decodes BHE1 packets into
bounded threadgroup tiles and multiplies there, without allocating a complete
dense device weight. The default remains `decode`. The fused operator uses a
new SIMD accumulation order: offline compilation and CPU decoder checks do not
establish GPU numerical parity or speed. Run the bounded
`qualification/context-0.1.1/metal_fused_probe.py` on an authorized Metal host
before attempting full-model qualification. Its default invocation only writes
a `NOT_RUN` preflight; `--execute-metal` is required for device work.

`--memory-metrics` enables `/v1/memory`, including sampled process RSS and the
selected backend's allocator/device counters, with units, measurement sources,
sample maxima, and errors. `--memory-interval-s` controls the positive sampling
interval. Sampling can miss short peaks; OS/allocator peak counters are reported
separately. Missing measurements are errors, not zero usage. These measurements
are separate from the operator's admission allocation estimate.

The CUDA batch server accepts the same memory flags. Its ready-window CUDA
receipt retains startup readings under `startup_memory_receipt`: before a CUDA
context exists, process attribution may be unavailable. Those startup errors
remain visible; they are not substituted with zeros or included in the new
ready-window failure counter. Check both scopes when qualifying startup memory.

For artifact-bound runs, use `--qualification-protocol <protocol.json>` and
`--qualification-role candidate` (or `reference`) together with memory metrics.
Startup verifies the installed source closure against the selected wheel,
package manifest and prompt pool, then publishes their hashes and the actual
process/host identity in `/v1/receipt`. `--qualification-wheel` selects a
different reference wheel. Populate the explicit paths, identities and
thresholds in `qualification/context-0.1.1/platform_protocol.example.json`;
the template is not a passing result. The platform runner owns only the server
processes it starts and keeps reference and candidate arms sequential.

The platform command now defaults to **CPU preflight only**. First capture the
required CPU controls from the qualification checkout:

```sh
python qualification/context-0.1.1/codec_preflight.py --check-code .scratch/CPU_READINESS.json
python qualification/context-0.1.1/platform_qualify.py /path/to/protocol.json
```

Pin that readiness receipt's path and SHA256 in `cpu_readiness`. The receipt
binds the tested source tree and raw pytest evidence; source or evidence changes
invalidate it. Preflight verifies wheel contents, launch arguments, inventory
hashes, complete package storage and declared memory ceilings without loading a
model. The storage inventories live outside the inventoried package directories
and enumerate every file's relative `path`, `bytes`, and `sha256`. Include the
compressed panels and their inventory in the candidate package's storage total.

For MLX, also run the readiness command with `--profile mlx` in the compatible
MLX environment and pin its result as `mlx_cpu_readiness`. This runs the actual
CPU model, MTP factory, vision, batching and HTTP suites. Set
`GEOREFINE_TORCH_REFERENCE_PYTHON` when the independent Torch reference uses a
separate interpreter. Both receipts must match the current source tree. Enabled
MTP sources are partitioned automatically into the separately verified head;
other intentionally unused sources still require an explicit exclusions list.

Only a separate, explicit `--execute` starts servers after preflight passes.
The candidate must declare and actually exercise nonzero `spec_k`; runtime
identity pins depth, batching capacity, backend and panel-inventory hash. Memory
limits are checked against measured maxima for both arms, and storage must meet
at least 1.53x. A passing protocol frame remains distinct from full cross-platform
release qualification. CPU readiness never asserts GPU correctness or speed.

The existing CUDA `run_all.sh` and `run_arm.sh` entry points also refuse to
launch by default. They require `GEOREFINE_EXECUTE_QUALIFICATION=1`,
`GEOREFINE_CPU_READINESS`, and `GEOREFINE_CPU_READINESS_SHA256`; the receipt is
revalidated before setup or an arm starts. These guards do not replace the
standing A100-40/27B and Blackwell acceptance requirements.
For a direct `run_arm.sh` invocation, also supply
`GEOREFINE_QUALIFICATION_WHEEL` and `GEOREFINE_QUALIFICATION_WHEEL_SHA256`;
`run_all.sh` exports these from its pinned wheel arguments. The wheel's source
manifest must match the CPU-tested source manifest.

Use `--metal-operator fused_auto` to derive BHE panels directly from verified
source tensors at startup. The first CPU pass verifies every binding, computes
bounded native-codec packets and admits the complete device-memory plan. Only
packet hashes and size records survive that pass. The second pass re-verifies
and converts one linear at a time. Canonical zero padding and source hashes are
checked before upload; embeddings and biased or unsupported linears remain
dense. This mode needs no separately shipped panel inventory. Its source working
budget includes the current tensor, encoded packets, census and assembly buffers.

CUDA qualification uses the same protocol runner through
`glc_serve.context_batch_server`. It pins the actual runtime format, MTP depth,
row capacity and measured tune, and records the device name, capacity and compute
capability. The native BHE status guard traps decoder errors, including during graph replay;
its existing checks remain active.

The target-specific protocol examples cover A100-40, Blackwell and Windows CUDA.
The A100 example runs the compressed candidate alone and checks capacity and
absolute performance, avoiding a dense reference allocation on that card. Paired
parent comparisons and 64/128-user throughput remain separate Blackwell gates.
Every example contains deliberately invalid placeholders requiring the frozen
artifact hashes, parameter counts, budgets and acceptance thresholds.

`qualification/context-0.1.1/release_matrix.py` combines pinned target results
offline. It revalidates raw evidence and current code, requires the same complete
27B artifact on A100/Blackwell, native platform proofs, acceptance/rejection
coverage, and source-bound CUDA kernel proofs on both acceptance cards. It emits
the six ICC serving-completion events only as PASS when the entire matrix passes.
Missing targets, synthetic component reports and stale hashes block release.
Use `release_matrix.example.json` as its input schema. This command launches no
servers or accelerators.

The offline `qualification/context-0.1.1/release_matrix.py` verdict requires
all four target receipts, both acceptance-card kernel proofs, and the
`hf_parent_proof` entries shown in `release_matrix.example.json`. HF proof pins
the reference inputs, raw logits, candidate and strict-forward reports, and
runtime metadata. The verifier recomputes all seven probes against the four HF
schedules, checks the registered `config.env` thresholds and strict bitwise
floor, and binds the parent, compressed wheel/package and Blackwell card.
Missing or stale proof blocks the release verdict. The standing 27B A100 40GB
capacity requirement is separate from the Blackwell 64/128-user paired run;
a candidate-only capacity pass makes no dense-parent speedup claim.

Performance protocols distinguish inclusive `min_speedup_vs_reference` (`>=`)
from strict `speedup_gt_reference` (`>`). Use the latter for the recorded
owner requirement `>1.3x`; equality must fail. The historical MTP-off codec
workload matrix and its paired trial requirements remain separate from the
64/128-user speculative-serving matrix; passing one does not establish the other.

The primary MTP-off matrix is also mandatory in `release_matrix.py`. Its
`primary_codec_proof` pins fifteen fresh-process raw reports: five each for Context, dense BF16, and published TBE.
`primary_codec_run.py --work-root <persistent-scratch> prepare ...` prepares
inputs from the recovered canonical audit and explicit image packet. Its
`paired-run` subcommand requires `--runtime`, `--fixture`, `--package`, `--dense`, `--published-tbe`,
`--output-dir`, `--output`, and `--binding-json`; see
`primary_codec_binding.example.json` for wheel, CPU readiness, model, fixture
and tune pins. It refuses execution without `--execute`. Every arm verifies
those pins before loading the runtime and records them with the raw evidence.
The offline verifier recomputes complete text and image coverage, warmup output
and logit equality, alternating nonoverlapping process order, and the original
five-pair bootstrap lower-bound rule (>1.3). Historical unbound reports cannot
qualify the final artifact. This is still distinct from public release approval.

Each platform arm records monotonic process-start and health-ready timestamps
and their elapsed duration. The startup deadline includes process creation and
a health response that arrives after the deadline is rejected. Resume validation
checks that saved timing remains finite, ordered and internally consistent.
Full-model A100/Blackwell release acceptance also requires candidate startup
within 600 seconds, independently of a more generous operational timeout.

The primary runner alternates `(Context, dense, published TBE)` and the reverse
order across five trial groups. Each group shares one dense reference; all
fifteen processes must be distinct and nonoverlapping. The published TBE input
is a separately pinned local `georefine.tbe.serve.v1` bundle with
`serve_manifest.json`, not a Context startup transcode. Its text path uses
native TBE descriptors and its image path uses the published bundle's verified
BF16-decoded vision frontend. Both comparisons retain complete raw text/image
and warmup evidence. Only the Context comparison supplies the strict >1.3x
release gate; TBE measurements remain explicitly labeled comparison results.
The former two-arm comparator remains diagnostic and cannot pass the complete
three-arm release gate.

### Optimised batch engine with speculative decoding

`georefine-context-optimised-serve` selects the native TBE21 batch engine and
MTP speculative depth 1 by default. It enables schedule 1, paired scans,
register-resident decode tables, and the existing tile64/tile128 policy. It
captures the graph buckets required by `max-users * (spec-k + 1)` and admits
the requested decode rows rather than retaining a 64-row cap at 128 users.
The speculative policy retains its full verification row budget.
For a real MTP head the profile selects calibrated row-aware head costs without
enabling experimental kernel row geometry. Drafting remains a cost-based choice;
tokens still pass the same acceptance/rejection and rewind rules. N-gram runs
do not activate an MTP cost model.

```bash
georefine-context-optimised-serve --package /path/to/context-package \
  --parent /path/to/parent-config --context-build-root /repo/.scratch/context-build \
  --max-users 64 --max-context 4096
```

Pass `--spec-k 0` for a non-speculative comparison or an explicit depth to
select another supported MTP workload. Missing MTP tensors, unsupported graph
capacity and conflicting kernel environment controls are errors. The same
entry with `--dense /path/to/parent` uses the same batching, graph-capture and
speculative settings without installing compressed-kernel controls.

The existing batch entry point also accepts `--optimise`; omitting it preserves
its prior selection. `--no-optimise` overrides the optimized entry point.
Health/settings report the selected `optimisation_profile`. PIPE128, WIDE128,
SCAN4 and alternate row geometry are excluded from this profile. The native
baseline kernels passed the operator/N-D GPU checks; the combined full engine
and speculative profile still require their own hardware qualification.

Earlier full-engine wins over the HF dense wrapper and isolated BI-GEMM dense
operator timings are separate comparisons. Neither substitutes for matched
optimized-engine comparisons, with and without speculation, across throughput,
prefill, first-token/decode latency, startup, memory and cost. Exactness and all
existing acceptance thresholds remain required.

The isolated TBE21 backend retains its default kernel schedule 2. For qualification,
`BI_TBE21_KERNEL_VARIANT=1` explicitly selects schedule 1 for the same TBE21 codec.
`BI_T21_SCAN=0 BI_T21_TABREG=1 BI_T21_V1_MINBLOCKS=2` selects the measured decoder
controls; every value is validated and enters the native compiler cache identity.
The backend's historical identifier remains `tbe21-v2`; a schedule-1 receipt also
carries `kernel_variant: 1`, checked against the compiled native translation unit.
The qualification source binding must declare the same variant. Changing selection
after loading requires a fresh process. None of these selectors changes the model
package, numerical reference, storage threshold, startup deadline or speed gate.
These integrated paths require fresh native qualification; prior operator timings
are not performance claims for this runtime.

`BI_T21_PACKSCAN2=1` opts into the paired prefix scan only with TBE21 kernel
schedule 1. Qualification declares `packed_scan2: true` in `tbe21_source` and
checks the actual CUDA translation unit flag in backend, dispatch and overlap
receipts. The default is off; changing the flag requires a fresh process.

The optional `BI_T21_TILE128=1` policy requires `BI_TBE21_KERNEL_VARIANT=1` and
`BI_T21_PACKSCAN2=1`. It keeps format-5 launches at tile64 for at most64 rows and
uses tile128 above64 rows. Both Python and the compiled native selector enforce
this rule. Other formats, the split/reduction order, row-budget settings and MTP
selection are unchanged. The default remains off. Runtime identity reports the
policy and successful Python/native submission counts by actual tile; CUDA graph
replays are explicitly uncounted. This path still requires separate serving and
release qualification.

Qualification protocols may opt into `kernel_cache_role_isolation` with an
absolute `root` inside their declared `work_roots`. The qualifier gives candidate
and reference servers separate cache directories and records those paths in each
stage. HF follow-up must reuse `platform_qualify.server_kernel_cache_environment`
for the candidate role, preserving its exact compiled binaries. Omitted selection
keeps the inherited build environment. Cache isolation changes no codec controls,
model settings, acceptance thresholds or binary/source verification.
