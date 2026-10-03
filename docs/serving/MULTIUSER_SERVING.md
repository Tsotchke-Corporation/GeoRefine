# GLC multi-user serving: audit, gates, and the measurement protocol

2026-10-02, release `v1.2.0-rc1`. Written for whoever runs the
GPU validation window. Every number in this document that is not labelled MEASURED is a
protocol, not a result: the multi-user path has not yet run on a GPU.

## 1. Why this exists

An independent replication measured FastSession against llama.cpp on the same card:

| users | FastSession | llama.cpp | verdict |
|---|---|---|---|
| 1 | 37.3 tok/s, 29% cheaper | 26.5 tok/s | we win (41% faster) |
| 4 | $15.55/M | $7.12/M | llama.cpp 2.2x cheaper |
| 8 | — | $4.00/M | llama.cpp 3.9x cheaper |

The cause is not the codec. `FastSession` serves **one conversation at a time**: at N users its
throughput is one user's throughput, while llama.cpp batches. The tester also correctly flagged
that our "15–23 GB memory saving" was measured against **our own dense engine**, not against
llama.cpp — so it is not a comparison anyone outside should accept. Both gaps are addressed
here: a batched multi-user engine, and a memory protocol whose baseline is llama.cpp on the
same card.

The batched path is **not** a new codec. It is the same bit-exact TBE / Q8_0 / K-quant decode,
run through `BI-GEMM` — one batch-invariant tensor-core GEMM used for every batch size, weights
decoded inside the kernel. That is what makes "batched" and "bitwise identical to single-user"
compatible claims rather than a trade-off.

## 2. Audit of the batched engine (bidec)

Files are under `release/glc_serve/` and `scripts/batchserve/` unless stated.

| Capability | State | Where |
|---|---|---|
| Batch-invariant GEMM, weights decoded in-kernel (TBE / Q8_0 / K+I-quants) | **DONE, GPU-gated** | `bi_csrc/bi_gemm.cuh`, `bigemm.py`; receipt `scripts/batchserve/receipts_kernel_gate_v1.json` = PASS, 0 rows differ over prefixes M=1..200, permutation, composition, repeat |
| BI-GEMM timing M=1..128 | **DONE** | `receipts_kernel_timing_v1.json` |
| Batched engine: paged KV (page = attention split), per-slot GDN conv ring + fp32 state, prefill through the same step | **DONE, code** | `bidec.py` `BatchDecoder`, `bidec_kernels.cu` |
| Continuous batching, admission, page reservation at admission | **DONE, code** | `bidec.py` `Batcher._admit` / `step` |
| **Request cancellation / client disconnect** | **DONE (new here)** | `Batcher.cancel`, `_reap_cancelled`; `bidec_serve.py` catches a broken SSE pipe |
| **`--max-users` (batch admission cap, distinct from device `--slots`)** | **DONE (new here)** | `bidec_serve.py` |
| **Capacity-reporting `/health`, 503 when the engine thread stops** | **DONE (new here)** | `Batcher.stats`, `bidec_serve.py`; previously a failed step killed the engine thread and the server accepted requests that then hung forever |
| OpenAI `/v1/chat/completions`, SSE streaming, bearer auth, 429 on a full queue | **DONE, code** | `bidec_serve.py` |
| Prefill while decoding (chunked prompt rows share a step with decode rows) | **DONE, code** | `Batcher.step` row budget |
| Exact MTP speculation inside the batch (E-SPEC) | **DONE, code** | `Batcher._step_spec`, `BatchDecoder.run_mtp` |
| **Scheduler exactness gate (slot reuse, page recycling, state reset, chunking, composition, cancellation) with leak positive controls** | **DONE (new here), CPU, PASS** | `bi_gate_batchexact.py`, `bidec_ref.py` |
| **Concurrent-stream exactness gate over HTTP (what the tester runs)** | **DONE (new here), needs a running server** | `bi_gate_stream.py` |
| **CPU test suite for the scheduler and the HTTP layer** | **DONE (new here), 34 passing** | `tests/test_bidec_scheduler.py` |
| Bench harness: closed-loop concurrency, TTFT p50/p95, steady tok/s, NVML sampling | **DONE, code** | `bench_serve.py`, `build_prompts.py`, `arm_engine.sh`, `arm_llamacpp.sh` |
| Engine-level bitwise gate on the real model (every emitted logits row) | **PARTIAL — written, never run** | `bi_gate_engine.py`; **UNVERIFIED-ON-GPU** |
| Batched E-SPEC gated against AR solo | **PARTIAL — written, never run** | `bi_gate_engine.py --spec-k`; **UNVERIFIED-ON-GPU** |
| Multi-user throughput / $-per-M numbers | **MISSING (no run)** | protocol in §5; **UNVERIFIED-ON-GPU** |
| Memory vs llama.cpp on the same card | **MISSING (no run)** | protocol in §6; **UNVERIFIED-ON-GPU** |
| Tool calling, multimodal input on the batched server | **MISSING by design** | `bidec_serve.py` rejects `tools` and non-string content; the single-user `fastserve` HTTP path has tools |
| Shared prefix cache, SLO-aware admission policy, CUDA-graph persistent decode | **MISSING (separate workstreams)** | interfaces reserved in `bidec_iface.py` |

The honest summary: **everything needed to serve many users concurrently is written, and every
layer that can be gated without a GPU now is gated. Nothing in the batched path has been run on
a GPU at all** — not one token. That is the whole of the remaining risk, and §7 is the plan to
retire it.

## 2a. Solving the concurrency ceiling (the architecture, not a caveat)

The question "how many users fit" has a closed form for this model, and `bidec_capacity.py`
computes it from the engine's own tensor shapes — no GPU, no forward pass
(`python -m release.glc_serve.bidec_capacity --vram-gib 96 --weight-gib 20`, and the server
publishes the same block in `GET /v1/receipt`). Two terms compete:

* **full-attention KV grows with context**: `16 layers x 2 x n_kv x head_dim x 2 B` per token.
  For a 27B-class GQA shape (4 KV heads, 128 dims) that is **32,768 B/token** — 264 MiB per user
  at 8k, 520 MiB at 16k.
* **the Gated-DeltaNet state does not grow with context**: a fixed `n_heads x d_k x d_v` fp32
  matrix plus the conv ring, over 48 layers — **~102 MiB per slot, constant**.

So there is a crossover, and for that shape it is at **~3,328 tokens** of context:

| context | KV / user | GDN state / user | binding term | levers that help |
|---|---|---|---|---|
| 2,048 | 72 MiB | 102 MiB | **GDN state** | state precision, state snapshot reuse |
| 8,192 | 264 MiB | 102 MiB | **KV** | paging, prefix sharing, KV quantisation |
| 16,384 | 520 MiB | 102 MiB | **KV** | the same, with more headroom to win |

This is the part that is usually got wrong, including by us: **below the crossover, no KV
technique raises the user count at all** — the binding term is a fixed per-slot tax — and
**compressing weights never changes the per-user cost; it changes the budget** the users are
drawn from. `test_bidec_capacity.py` asserts exactly that distinction, so it cannot be restated
loosely later.

**The plan follows from the arithmetic, and every lever below already has a receipt in this
repository.** Nothing here is a hope.

**Above the crossover (long context, the 8k/16-user case):**

1. **int8 per-token KV quantisation on the full-attention layers — ~2x KV, already certified.**
   `experiments/georefine/_kv_quant.py` has the layer plan (it reads the 48/16 split out of the
   config, no hardcoding), the schemes, and a tensor-level `quantize_kv_pair(key, value, cfg)`
   that takes raw tensors and so can point at a paged buffer. On the 2B lane int8 per-token
   **passed 6/6** (SAE 0.997845, mass-cosine 0.996146); **int4 FAILED the chat gate** at both
   schemes and int2 failed as the designed control. So int8 is the ceiling on this lever and int4
   is its negative control — stated that way round because the measurement says so. The work
   remaining is the **packed paged buffer**: today's path is fake-quant, storage stays bf16, and
   its own receipts carry `memory_saving_is_projection: true`. A projection is not a saving, and
   `emit_receipts.py` refuses to mark such a receipt as measured.
2. **Shared-prefix KV reuse.** `bidec_prefix.py` (sibling branch) with BLAKE2b block-chained
   hashing at the 256-token block, token-by-token verification so a hash collision cannot
   decide anything, and an `adopt` that installs the **GDN recurrent snapshot** as well as the
   page table — a page table alone is not enough, which is why its chain tip refuses a partial
   restore. It owns private page copies and never aliases the shared pool, so it does not
   perturb the page accounting this engine gates.
3. **GQA KV-head reduction** (`geometric_pruning_scalable.py::surgical_prune_heads`) physically
   rebuilds `k_proj`/`v_proj` at a smaller width and updates `num_key_value_groups`. It is a
   production path — and it has **never been run on this hybrid's 16 full-attention layers**,
   which are precisely the layers that carry the KV. That is the cheapest untried experiment on
   the list.


**The GDN ring dtype is a measured lever, held behind the kernel.** Storing the per-slot
recurrent ring in fp16 is behaviour-lossless on Qwen3.5-2B (greedy 1.000000 at 8k and over 64
round-trips, top-5 0.995, rare-tail gap <= 0.17 pt, error growing as sqrt(round-trips) with no
amplification) and halves ring traffic: 36 -> 18 MiB per user per step, 2.25 -> 1.125 GiB per
step at 64 users. On the 27B shape it saves 48 MiB per slot and moves the KV/state crossover
from ~3,328 to ~1,792 tokens, which `bidec_capacity.py` reports as `gdn_ring_alternative` and
labels PROJECTED. `--gdn-ring-dtype {fp32,fp16}` is plumbed through the engine and the server and
recorded in every receipt, and fp16 **refuses** rather than allocating: `b_gdn_recur` types the
ring as `float* state_all` (`bidec_kernels.cu:84`, `:99`, called with `P<float>(state)` at
`:371`), so a half buffer would be reinterpreted as fp32 and corrupt every sequence. The kernel
needs a half-storage path (fp32 math in registers, fp16 in memory); until then the saving is
projected, and when it lands the arm is labelled BEHAVIOUR-LOSSLESS pending the SAE
rare-feature survival gate on the served lane. Measured alongside it: int8 KV *pages* are
greedy-lossless but 0.95 on top-5, which does not clear the behaviour-lossless bar, and int4
pages were rejected. Source: `docs/serving/KV_STATE_DESCENT_20261003.md`.

**Below the crossover (chat-length, where the fixed state tax binds):**

4. **Temporal-delta coding of the GDN recurrent state — 1.885x, bit-exact, receipted.**
   `experiments/georefine/_state_codec.py` extends the lossless weight codec to the recurrent
   and conv state with a delta/XOR prefilter across decode steps; the receipt
   (`.icc/evidence/gdn-state-codec-20260717/`) records `all_roundtrip_bitexact: true` with mean
   1.7279x on the recurrent stream (best per-layer 1.885x) against 1.5475x without the temporal
   prefilter. It needs a device-side decode kernel to become a serving win, and it is bit-exact,
   so it costs nothing in behaviour.
5. **Per-layer state precision, where it is bit-identical.** fp32 is there for recurrence
   stability; whether all 48 layers need it is a measurement, not an architecture question. Any
   layer whose state trajectory is **bit-identical** at lower precision over 8k steps is free
   memory under both invariants; any layer where it is merely close is inadmissible.

**What we are explicitly not doing, and why — each a measured negative, not a hunch:**

* **A lossless KV codec as a ratio play.** Three independent efforts (July, an August GPU
  census, and an October CPU census) all land at **~1.47–1.53x ideal** on K and V, the shuffled
  control sits within 0.02% of the in-order result (so there is **no structural component** to
  exploit), and our own ledger has the bf16 mantissa measured memoryless. The ceiling is the
  exponent plane, and it is already known. It is exact, but it is not the lever.
* **Lossy KV below int8** (KIVI/KVQuant/QServe-class 2–4 bit). Our own int4 result is a chat-gate
  failure, and the published "comparable quality" claims are aggregate benchmarks that have been
  blind to exactly our failure mode before (the gdn-safe 3.99% rare-feature sign flip at L10 was
  caught only by the attribution gate).
* **KV-token-dropping offload** (InfiniGen-class). It drops entries chosen by a rehearsal, so it
  is neither bitwise per-request nor parent-equivalent. Lossless paging of **cold** KV pages to
  host memory stays admissible — nothing is dropped and a page's bytes are identical whichever
  side of PCIe they came from — and it is a latency trade, not a free win.
* **`_mla_pruning.py` as "owned MLA machinery".** It prunes an already-MLA model and cannot
  convert GQA to MLA; `apply_mla_plan_to_layer` defaults to attribute names no `transformers`
  MLA module has, every lookup is None-guarded, and `MLAPrunePhase` produces a plan that nothing
  consumes. Several docs in this repo advertise it as working; they are wrong and should be
  corrected. Post-hoc MHA/GQA-to-MLA conversion (MHA2MLA, arXiv:2502.14837, reports 92.19% KV
  reduction for 0.5% LongBench on Llama2-7B using 0.3–0.6% of corpus) is a real and attractive
  lane for exactly our 16 layers, but it is a conversion-plus-heal programme that must clear the
  behavioural cert against the parent, not a flag.

**Roadmap beyond the levers above**, with the number each is worth and the gate it needs:

* **Shared-prefix cascade attention.** FlashInfer's cascade inference reports up to 31x on
  shared-prefix attention *compute*, but only at a 32,768-token shared prefix, batch >= 128 and
  unique suffixes <= 256 tokens; at N <= 64 with ordinary chat prefixes it is worth single
  digits, so build it for latency, not for the ceiling. The gate is the hard part and it is
  precise: the attention merge is commutative in algebra and **not** in floating point, so the
  merge tree must be a pure function of the row's own length and adopted-prefix length, never of
  how many requests happen to share the prefix. In this engine that means `b_attn_split` needs a
  second item class keyed by `(shared_page, head)` and `b_attn_combine`'s fixed ascending
  `s = 0..ns-1` walk must be extended without becoming batch-dependent — and collapsing page
  partials into one cascade partial is only bitwise equal if the cascade partial is produced by
  the same sequential page-order log-sum-exp accumulation that `combine` does today.
* **Order-independent chunk-partial summation in BI-GEMM** (a kernel design parameter, for
  whoever owns the kernel next). Today batch invariance is bought by fixing the reduction order:
  a split count fixed by `(N, K)`, partials summed in split order. Making the chunk-partial sum
  *order-independent* instead — a sorted or exactly-rounded accumulation — would make
  head-granular and expert-granular **permutations** bit-exact as well, not just batch
  compositions. It is worth noting that this is the same root cause as batch invariance seen from
  the other side: both are "the answer must not depend on the order the partials arrive in". The
  cost is the accumulation itself; the gate is a permutation sweep over head and expert order
  with the element floor above.
* **Overlap / async scheduling** (SGLang v0.4-class, 1.1–1.3x): build step t+1's admission, page
  allocation and metadata on the CPU while step t's graph replays. The thing exposing the host
  today is the device-to-host `.tolist()` per step; the fix is a pinned async copy plus an event
  and consuming the previous step's tokens one step late. Pipeline depth changes, the token
  stream does not — which is exactly what G3a-SCHED must confirm before it ships unflagged.
* **Nano-batching** (NanoFlow, arXiv:2408.12757, 1.91x over vLLM/TensorRT-LLM at 50–72% of
  theoretical optimal) for intra-step overlap of GEMM and attention. Only meaningful after the
  overlap scheduler, and its premise ("serving is compute-bound") holds at throughput-maximising
  batch on dense 70B-class models, not at N <= 64 on a hybrid with a fixed per-slot state tax.
* **Chunked prefill is already banked** (Sarathi-Serve reports 2.6x–5.6x serving capacity and we
  have it), and **prefill/decode disaggregation** (DistServe 7.4x, Mooncake +75% on a real
  workload) is a goodput-under-SLO move across a cluster. Neither raises per-box concurrency when
  the limit is memory capacity: scheduling factors and memory factors are different currencies
  and must never be multiplied together.

One correctness note we checked rather than assumed, because the scheduling literature makes it
a hazard: Sarathi-style chunked prefill sizes a chunk to fill a token budget **alongside**
concurrent decodes, so the chunk boundary is batch-dependent. Ours is too — and it does not
matter here, because exactness in this engine does not come from request-determined chunking. It
comes from every row being computed from its own sequence only, with a split count that is a
function of the row's own position. G3a-SCHED varies the chunk size and the row budget on purpose
(chunk 97, row budget 8, 160, 256) and the probes' tokens and per-row logits digests do not move.
`chunk_align` exists for the prefix cache's snapshot boundary, not for exactness.


## 3. The exactness gate design

The claim a multi-user server has to earn: *a request's output does not depend on who else is
being served.* It is gated in three independent layers, deliberately — each catches what the
others cannot.

**Layer 1 — arithmetic (GPU, PASS).** `bi_gate_kernel.py`. BI-GEMM uses one `mma.m16n8k16`
schedule for every batch size, a split-K count fixed by `(N, K)` alone, no atomics, partials
summed in split order, one bf16 rounding. Gated bitwise: 0 rows differ across row-count
prefixes, permutations of the batch, probe rows placed at arbitrary slots among random rows, and
repeats — for bf16, Q8_0 and TBE on three shapes plus Q5_K / Q4_K / IQ4_XS / IQ3_S tensors of a
real GGUF. TBE-coded equals the bf16 parent, Q8_0 equals `bf16_rn(d*q)`, K-quant equals the
`gguf-py` dequant twin.

**Layer 2 — scheduler (CPU, PASS, new).** The layer where a multi-user server actually leaks:
a slot reused without resetting the recurrent state, a KV page recycled while still live, a
page-table entry never reserved, a prompt chunked across a page boundary, a request admitted in
a different order. None of that needs a tensor core. `bidec_ref.RefDecoder` is a CPU decoder
whose row value is *defined* to depend on exactly two things:

1. the slot's tokens at positions `0..p`, **read back through the KV page table** (not from a
   per-slot Python list) — so a stale, unreserved or double-mapped page changes the answer;
2. a per-slot recurrent state advanced one row at a time in position order — so a slot handed
   to a second user without `reset_slot_state` changes the answer.

Freed pages are deliberately **not** cleared, so a wrong read finds the previous user's token.
`bi_gate_batchexact.py` then runs the real `Batcher` over it and requires each probe's token
list *and* per-row logits digest list to equal its solo run under six schedules:
`all_at_once`, `staggered` (probes join among fillers that join and leave), `odd_chunk`
(chunk 97, row budget 160 — off every power of two), `slot_churn` (`max_users` 2, so slots are
forcibly reused), `cancelled` (fillers aborted mid-flight), `tiny_rows` (row budget 8).

Crucially the gate **fails on purpose on demand**: `RefDecoder(leak=...)` makes the row depend
on the padded batch width, on the neighbouring row, or disables state reset / page reservation.
The verdict is PASS only if the clean run has 0 mismatches **and all four leak controls are
detected**. First run: PASS, 24 comparisons / 576 rows, controls caught 24 / 21 / 4 / 24
mismatched probe-schedules, 14.9 s. (The `noreset` control is caught only by the slot-reuse
schedules — which is exactly why `slot_churn` is in the battery.)

**Layer 3 — end to end over HTTP (needs a card, new).** `bi_gate_stream.py`: each probe's
stream with the server to itself, then the same probe inside concurrent batches of
N = 1, 2, 4, 8, 16 with staggered filler traffic, requiring identical streamed text, token count
and finish reason every time; plus stream == non-stream, plus exactness after a client hangs up
mid-stream. This is the one the tester runs; it is text-level identity, and it names the
bitwise logits receipts it is *not* a substitute for.

`bi_gate_engine.py` (GPU, written, never run) closes the loop: the sha256 of **every emitted
logits row** of the real model, solo versus four batch schedules, plus batched E-SPEC against
autoregressive solo.

### How much output a "bitwise" claim has to compare

A bit-exactness verdict is only evidence in proportion to how much output it compared. On a
bf16-output criterion, an exact transform and a non-exact one are **indistinguishable below
~1e5 compared output elements**: the divergence rate is 1.2e-4 to 4.9e-4 per element and grows
with the square root of the perturbation size, so a gate that compares a few tens of thousands
of elements will report agreement either way. A verdict over too little output is not wrong, it
is uninformative — and it is an easy way to ship a false exactness claim in good faith.

So every gate here states its sample size, and the word "bitwise" is reserved for a pooled count
of **at least 1,000,000 output elements**:

| gate | what it compares | sample size |
|---|---|---|
| `bi_gate_kernel.py` | GEMM output rows, per M and per format | stated in the receipt |
| `bi_gate_batchexact.py` | per-row logits digests, 6 schedules + a state-ring re-feed | **576 logits rows x 2048 = 1,179,648 output elements** — above the floor, asserted at run time against the decoder's real logits width |
| `bi_gate_engine.py` | per-row logits digests of the real model | stated in the receipt; **UNVERIFIED-ON-GPU** |
| `bi_gate_stream.py` | token stream and decoded text | **0 output elements by construction** — it observes what a client observes, so it shows that the engine's output does not change with batch composition, and it is explicitly *not* the source of a bitwise claim |

`bi_gate_batchexact.py` FAILS below `--element-floor` (1e6 by default) rather than passing
quietly; `--allow-small-sample` lets it pass for a quick smoke, and the receipt still records
`sufficient_for_a_bitwise_claim: false`. `emit_receipts.py` pools the counts across receipts,
names any receipt that carries no count rather than ignoring it, and labels PS4's evidence level
"agreement, not demonstrated bitwise equality" with the numbers whenever the pool is short. This
is also why the earlier version of our own scheduler gate was not good enough: it compared
73,728 elements, which is inside the indistinguishable regime.

## 4. Running the multi-user server

One command. `--slots` is device capacity (KV + GDN state buffers); `--max-users` is how many
requests are admitted into the batch; further requests queue up to `--max-queue`, then 429.

From a repository checkout, at the repo root (`release/` is a PEP 420 namespace package, and
`bidec_serve` puts it on the path so the absolute `glc_serve.*` imports resolve either way):

```bash
python -m release.glc_serve.bidec_serve \
  --bundle /path/to/Qwen3.8-27B-GeoRefine-TBE \
  --tune /path/to/fastdec_tune.json \
  --slots 32 --max-users 16 --pages 1600 --max-ctx 16384 \
  --max-rows 256 --prefill-chunk 256 \
  --api-key-file /path/to/key.txt --host 127.0.0.1 --port 8290
```

From the installed wheel (`pip install ./release`) the same flags are taken by the
`glc-multiuser` console script, which is the form in the tester's README.

For a GGUF (Q8_0 / K-quant) instead of the TBE bundle, replace `--bundle` with
`--parent DIR --gguf FILE.gguf`. Add `--spec-k 5` for exact MTP speculation (greedy only;
the server rejects `temperature > 0` in that mode). Loopback only, by design.

```bash
curl -s localhost:8290/health | jq      # active, waiting, max_users, slots_free, pages_reserved
curl -s localhost:8290/v1/chat/completions -H 'Authorization: Bearer KEY' \
  -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":64,"stream":true}'
```

`--pages` sizing: a page is 256 positions; the pool must hold
`sum over concurrent users of ceil((prompt + max_new)/256) + 1`. Pages are reserved at
admission, so an admitted request can never run out of KV mid-generation — a request that does
not fit waits instead of being killed.

## 5. Throughput and cost protocol

Report `$/M output tokens = ($/h of the card) / (steady output tok/s) * 1e6 / 3600`, the same
arithmetic as the tester's table, so the numbers are directly comparable.

```bash
# prompts: a fixed pool, identical for every arm (classes chat / ctx1k / ctx8k)
python scripts/batchserve/build_prompts.py --src . --out prompts.jsonl

# table 1: short prompts, N = 1/4/8/16
for N in 1 4 8 16; do
  python scripts/batchserve/bench_serve.py --port 8290 --api-key-file key.txt \
    --prompts prompts.jsonl --classes chat --concurrency $N --duration 120 --warm 20 \
    --max-tokens 256 --card-usd-h "$USD_H" --label "bidec chat c$N" \
    --method-note "$METHOD" --out results/bidec_chat_c$N
done

# table 2: THE SCENARIO THAT DECIDES -- 8k-token prompts at 16 and 32 users
for N in 16 32; do
  python scripts/batchserve/bench_serve.py --port 8290 --api-key-file key.txt \
    --prompts prompts.jsonl --classes ctx8k --concurrency $N --duration 180 --warm 30 \
    --max-tokens 256 --card-usd-h "$USD_H" --label "bidec ctx8k c$N" \
    --method-note "$METHOD" --out results/bidec_ctx8k_c$N
done
python scripts/batchserve/make_tables.py results/*/summary_c*.json
```

### The ctx8k case is the one that matters

An internal cost study records llama.cpp at 8k context collapsing to **3–4 tok/s per
user with TTFT 45–139 s at 16–32 users**: a long prompt's prefill monopolises the device and
starves every decoding request. That is a *scheduling* outcome, not a weights outcome, and it is
exactly what `bidec`'s chunked prefill is built to avoid — prompt chunks share a step with
decode rows under one row budget (`--prefill-chunk`, `--max-rows`), so prefill cannot run to
completion ahead of everyone else.

So this is the case where our architecture, rather than our codec, is on trial. Report for it:
TTFT **p50 and p99**, per-user decode tok/s (p5/p50/p95 — the p5 is the starved user),
aggregate tok/s, and `$/M`. Run the identical scenario on the llama.cpp arm with its own best
settings (`-np N -cb -c 8192 -ub <its best>`). `bench_serve.py` records `cost_usd_per_m_output`
from `--card-usd-h` and the compared arm's configuration from `--method-note`, so a mismatched
pair of arms cannot be assembled into a table by accident.

**Method parity, recorded in every summary:** same prompt pool and class, same `--max-tokens`,
greedy sampling in both arms (`temperature 0`), same card and session, closed loop, and the
llama.cpp build commit plus its exact flags (`-np`, `-cb`, `-c`, `-ub`) in `--method-note`.

Rules for a number to count:
- same card, same driver, same context length, same `max_tokens`, same prompt file for both
  engines, in the same session;
- closed-loop concurrency (each worker sends the next request only when its previous one
  finished) — an open-loop sender measures the queue, not the engine;
- drop the warm-up window; report steady-state output tok/s, and TTFT p50/p95 separately,
  because batching buys throughput by spending latency and both belong in the table;
- llama.cpp arm (`arm_llamacpp.sh`) run with its own best settings for the same card —
  continuous batching on, `-np N`, a comparable context. We do not get to hobble the baseline.

### Method rules that make a comparison legitimate

1. **One client drives both engines.** `bench_serve.py` is the only sender for both arms. Two
   different harnesses measure two different things, and the difference will be attributed to
   the engine.
2. **Never compare a pooled-throughput number with a decode-only number.** llama.cpp's
   aggregate tok/s includes prompt tokens; our steady figure is output tokens. Compare output
   tok/s to output tok/s, and state which is which in the table.
3. **Report p99 TTFT and goodput under an SLO beside aggregate tok/s.** A 64-user "win" that is
   9.3 tok/s per user at 17.9 s TTFT is not a win for anyone using it. Pick an SLO (e.g. TTFT
   <= 2 s and >= 15 tok/s per user), and report the share of requests meeting it — goodput — at
   every N.
4. **`$/M` is a curve over N, not a point.** The crossover is the finding; a single N is a
   selected number.
5. **Every throughput number is co-submitted with its quality gate.** A `$/M` without the
   matching greedy-bitwise receipt from the same session is not reportable. `emit_receipts.py`
   binds them: ps3 (speed and memory) and ps4 (identity) are items of the same receipt with the
   same provenance block, and `icc receipt-validate --compare` refuses to pair receipts that
   differ on an undeclared provenance axis.

## 6. Memory protocol: vs llama.cpp, same card

The tester is right that a saving measured against our own dense engine is not a saving. The
baseline is llama.cpp on the same card at the same context and concurrency.

1. Idle: `nvidia-smi --query-gpu=memory.total,memory.used --format=csv` with nothing loaded —
   record the floor (driver + display).
2. Start arm A (llama.cpp, `-np N`, context C). Reach steady state: run the bench at
   concurrency N for 60 s. Sample `memory.used` every 0.5 s for the last 30 s
   (`bench_serve.py` does this via NVML and writes `nvml_c<N>.csv`). Record **peak** and
   **steady median**, minus the idle floor.
3. Stop arm A. Confirm `memory.used` returns to the floor.
4. Start arm B (`bidec_serve` with `--max-users N`, `--max-ctx C`, `--pages` sized for N*C).
   Repeat step 2 identically.
5. Report, for each N: peak and steady device memory for both arms, the difference, and the
   weight bytes each arm holds (`/v1/receipt` reports `state_bytes` and
   `streamed_bytes_per_step` for ours; llama.cpp's is the file size of its quantised GGUF).

Report model bytes and KV bytes **separately**. They move for different reasons: the codec
changes the weight bytes; `--pages` and `--max-users` change the KV bytes, and a KV pool sized
for 16 users is not a regression against a baseline serving one. Any single-number "we save X
GB" without N, the context, and both arms' configuration should not be published.

## 6a. Receipts: one tarball that grades the oracle

The ICC completion oracle `georefine-production-serving` is blocked on **missing receipts**, not
on bad measurements. `scripts/batchserve/emit_receipts.py` turns the artefacts above into
ICC-shaped receipts so one `tar` grades it. It measures nothing itself: each of the four items
is emitted either with `status: "measured"` and the numbers, or with `status: "not_run"` **and
the exact command that would produce it** — never a blank that reads as a pass.

| item | claim | fed by |
|---|---|---|
| ps1 | the exported artifact is bit-exact and self-verifying | `--verify-json` (the codec verifier's output) |
| ps2 | the served engine's weight bytes, resident device memory and load time | `--server-receipt` (`GET /v1/receipt`), `--nvml-load`, `--load-seconds` |
| ps3 | smaller **and** no slower than dense bf16 **through the served lane**, per N | `--bench-codec` and `--bench-dense` (the dense arm is required; without it there is no comparison) |
| ps4 | served greedy output is identical to the offline single-user reference, per N | `--gate-http`, plus `--gate-engine` for the per-logits-row bitwise level |

```bash
python scripts/batchserve/emit_receipts.py --out receipts/ \
  --verify-json verify.json --server-receipt receipt.json \
  --nvml-load nvml_load.csv --load-seconds "$LOAD_S" \
  --bench-codec results/bidec_chat_c1 --bench-codec results/bidec_chat_c4 \
  --bench-dense results/dense_chat_c1 --bench-dense results/dense_chat_c4 \
  --gate-http receipt_http.json --gate-engine receipt_engine.json \
  --gate-sched receipt_sched.json --gate-kernel receipt_kernel.json
tar -czf serving_receipts.tgz receipts/
```

ps3 reports `FAIL` if the codec arm is slower or larger than the dense arm at any measured
concurrency, and ps4 reports `FAIL` if the offline bitwise gate fails — the assembler is not a
rubber stamp, and its own logic is checked on the CPU by
`python scripts/batchserve/emit_receipts.py --self-test` (14 checks, including that an empty
input set yields four `not_run` items and that a slower codec arm fails ps3).

## 8. What must not be claimed yet

The three constraints that bind in the multi-user regime, and the engineering answer to each
(stated the same way in the tester's README, so the two cannot drift):

1. **KV binds before weight bytes do at N <= 64.** A codec saving is a fixed byte count; what
   fills the card with several users resident is KV. Answer, and the reason the batched engine
   exists: paged KV (page = the attention split, from a shared pool), pages reserved at
   admission so the pool can run near full, and shared-prefix KV reuse for the users that share
   a prefix (`bidec_prefix.py`, CPU-gated on a sibling branch; interface reserved in
   `bidec_iface.py`).
2. **The hybrid's 16 full-attention layers are the whole concurrency budget.** The per-slot GDN
   recurrent state is tiny beside a full-attention layer's KV, but that per-layer ratio is not a
   concurrency multiplier — a 3:1 mix asymptotes to roughly 4x whole-model. The useful
   consequence is that every KV lever should aim only at those 16 layers: paging, prefix
   sharing, and KV quantisation on the full-attention layers (the `--kv-quant` path exists in
   the recertification tooling and is the next thing to wire in). `/v1/receipt` reports
   `state_bytes` split `kv` / `gdn_state` so the split is measured.
3. **The batched path needs its first GPU tokens.** Everything gateable without a GPU is gated
   (BI-GEMM receipted on GPU, scheduler receipted on CPU with leak controls, 37 CPU tests), the
   four remaining receipts are each one documented command, and §7's 5060 smoke is the cheapest
   way to get the first tokens through it.

**Determinism, priced correctly.** The only published cost of batch-invariant inference we could
verify from a primary source is Thinking Machines' *Defeating Nondeterminism in LLM Inference*
(Sept 2025): on Qwen-3-8B, 1000 sequences, vLLM's default path takes 26 s, the deterministic
path 55 s (**2.1x**), and 42 s (**1.6x**) with an improved attention kernel — and they attribute
much of the remainder to an unoptimised FlexAttention integration. Their diagnosis is also ours:
the cause of nondeterminism is batch non-invariance (kernels are run-to-run deterministic but
numerically batch-size-dependent, and server load varies the batch size), and the fix needs a
consistent RMSNorm reduction, fixed matmul tiles with no adaptive split-K, and a **fixed
attention split size**. Our page = 256 = the attention split is exactly that prerequisite,
arrived at independently.

BI-GEMM is flat M=1..64 at a measured **+9.8%**, and it is bitwise equal to the **bf16 parent**,
not merely self-consistent — a strictly stronger property than the published work claims. So the
comparison to make is +9.8% against 60–110%, and the parent-equality on top. (An earlier
internal note quoted a "34–63%" figure for the cost of determinism; we could not trace it to a
primary source, so it is not used here. The 1.6–2.1x above is sourced.)

Still unmeasured, and labelled so everywhere:

- Any multi-user throughput, latency or `$/M` figure for the batched engine: **none measured.**
- That the batched engine is bitwise equal to the single-user path *on the real model*: the
  arithmetic is gated, the scheduler is gated on CPU, the end-to-end gate is written and
  **never run**.
- Any memory saving against llama.cpp: **not measured against llama.cpp.**
- Batched E-SPEC speedup: written, never run.

What *can* be said today: BI-GEMM is bitwise batch-invariant and bitwise equal to the bf16
parent (receipted on GPU); the scheduler is exact under every batch composition, arrival order
and slot-reuse pattern tested, with leak controls that prove the test can fail (receipted on
CPU); and the server now has the admission, cancellation and failure behaviour a multi-user
deployment needs.
