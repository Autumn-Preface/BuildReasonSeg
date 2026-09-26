# TO_DSH — Task 6C.5: Training Pipeline Throughput Audit & Value-Preserving Optimization

> Status: **ACTIVE**
>
> Repository: `BuildReasonSeg`
>
> Purpose: diagnose the observed **CPU≈100% / GPU≈40%** training imbalance and implement **value-preserving training-pipeline optimizations** before continuing algorithm work.
>
> This is a performance-engineering task, not a model-quality task.
>
> **Do not start Task 6D. Do not change the model architecture, loss, dataset semantics, training subset, optimizer semantics, or effective batch size.**

## 0. User-facing language

All narrative text visible to the user in DSH web/chat must be **Chinese**.

Commands, code, paths, metric names, profiler labels and raw logs may remain English.

`handoff/FROM_DSH.md` / `handoff/PROJECT_STATE.md` may remain English.

## 1. Current hardware / motivation

Target machine:

- Windows 11
- Intel Core Ultra 9 275HX
- 32 GB RAM
- RTX 5080 Laptop GPU, ~15.9 GiB VRAM
- NVMe SSD

Observed by the user during current training:

- GPU utilization often around **40%**
- CPU utilization often near **100%**

Current Task 6C training loop is visibly serial:

```text
Sample
→ PIL image read
→ Qwen processor / chat construction / tokenizer
→ CPU tensors
→ H2D
→ SAM feature lookup / encode
→ GPU forward/backward
→ optimizer
→ next sample
```

Task 6C also trains with **real micro-batch = 1** and strict deterministic mode.

Task 6C already improved the SAM2 CPU feature cache to 480 images, with a measured footprint of ~7.5 GiB.

The goal now is to determine exactly where wall time is spent and make the GPU wait less.

## 2. Scope and hard boundaries

### Allowed

You may:

- add profiling/timing utilities;
- add CPU input caches for deterministic, model-independent data;
- cache/precompute Qwen preprocessing outputs that are independent of trainable weights;
- cache source images / GT masks if useful;
- add pinned-memory support;
- add non-blocking CPU→GPU copies;
- add a bounded prefetch/DataLoader implementation if measured useful;
- benchmark DataLoader worker counts;
- fix avoidable repeated PIL / tokenizer / processor work;
- benchmark gradient checkpointing ON vs OFF;
- benchmark strict deterministic ON vs OFF;
- improve feature-cache transfer mechanics;
- implement a faster default pipeline **only if it is output-equivalent under the acceptance tests below**.

### Forbidden

Do not:

- change Qwen3-VL-2B to another model;
- change SAM2 model;
- use 4B;
- add `[REF]`;
- change the SAM bridge;
- change LoRA rank/targets;
- change loss weights;
- change learning rates;
- change optimizer;
- change scheduler;
- change number/order of optimizer steps;
- change effective batch size;
- adopt batch size 2+ in this task;
- use gradient accumulation to pretend GPU occupancy improved;
- change BuildSpatialReason data;
- use test split;
- modify GT/inference semantics;
- add Spatial Relation Encoder;
- add Spatial Consistency Loss;
- resume model-quality experiments.

**Task 6C.5 must preserve the current batch-1 optimization semantics.**

True batching can be considered later, after this audit.

## 3. UU Accelerator policy — ignore completely

The user explicitly states that **UU is their own currently used accelerator and does not need to be inspected or managed**.

Therefore:

- do not search for UU processes;
- do not inspect UU state;
- do not mention UU in routine task output;
- do not close, stop, modify or diagnose UU;
- do not treat unrelated hosts entries as a project problem.

Only Watt Toolkit lifecycle matters for DSH-managed networking.

## 4. Watt Toolkit ownership policy

At task start inspect **only Watt Toolkit / Steam++ state**.

### If Watt is already running before DSH starts this task

Record:

`watt_preexisting = true`

Then:

- leave it alone;
- do not stop it at task end;
- training still runs offline from local cache;
- if Git push works, use it without claiming DSH started Watt.

### If Watt is not running

Record:

`watt_preexisting = false`

Keep it off during profiling/training.

If final `git push` needs networking, DSH may use the already verified `FULL_AUTO_OK` lifecycle:

1. launch Watt;
2. verify its accelerator;
3. push;
4. close with `WM_SYSCOMMAND / SC_CLOSE`;
5. verify Watt processes/listeners/Steam++ hosts block are gone.

Only close Watt if **DSH itself started it during this task**.

Never:
- force-kill;
- edit hosts;
- change certificates;
- use `verify=False`.

## 5. Reuse current environment / current codebase

Use exactly:

`C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-mvp`

Use current main after Task 6C.

Current formal training defaults remain:

- Qwen3-VL-2B
- SAM2.1 Base+
- BF16
- batch-1 semantics
- strict deterministic mode
- Phase-B joint path

The representative performance workload should use the **Task 6C P_C path** unless profiling shows bridge choice has measurable runtime impact. The bridge difference is not the subject of this task.

# PART A — Baseline profiling

## 6. Create a fixed benchmark workload

Create a deterministic benchmark set from the existing **Task 6C P training subset**:

- 64 records;
- preferably 32 paired images × 2 instructions;
- fixed IDs before benchmarking;
- representative mix of L1/L2/L3;
- no validation/test data.

Create:

`evaluation/task6c5_benchmark_ids.json`

Use the same record order for every benchmark.

## 7. Separate cold-start and steady-state performance

Do not mix cache construction with steady-state training.

Measure separately:

### Cold-start

Includes:

- model/runtime load;
- SAM feature cache population if needed;
- Qwen preprocessing-cache population if added;
- input cache memory growth.

### Warm steady-state

After all intended caches are warm:

- run warmup steps first;
- then measure at least **64 optimization steps**;
- use the same samples in the same order.

Primary performance decisions must use **warm steady-state throughput**.

Also report total first-epoch cost separately because it matters in real use.

## 8. Instrument the existing pipeline before optimizing

Create a profiling path that measures at minimum:

```text
image_io
target_mask_io
qwen_prepare_total
  ├─ image preprocessing
  ├─ chat/template + tokenizer where separable
cpu_to_gpu
sam_feature_lookup
sam_feature_encode_on_miss
qwen_forward
sam_projection_decode_forward
loss
backward
optimizer_step
whole_step_wall
```

If exact Qwen forward vs SAM forward separation is awkward in backward, use:

1. a small **synchronized stage-profile** run with CUDA events;
2. a separate **unsynchronized end-to-end throughput** benchmark.

Do not put `torch.cuda.synchronize()` inside the normal optimized training loop merely for measurement.

For GPU-side timings use CUDA events where possible.

For CPU-side timings use a high-resolution monotonic timer.

## 9. System-utilization sampling

During each steady-state benchmark record:

- GPU utilization from `nvidia-smi` or an already-installed NVML binding;
- GPU VRAM used;
- GPU power draw if available;
- CPU total utilization;
- process CPU utilization;
- process RSS;
- system RAM usage;
- samples/sec;
- ms/sample.

Sample utilization periodically (e.g. ~0.2–1.0 s) without installing new monitoring software.

Do not trust Windows Task Manager alone.

Report:
- mean;
- median;
- p90;
- min/max where useful.

If `nvidia-smi` sampling itself measurably perturbs the run, quantify it and use a lower sample frequency.

# PART B — Optimization candidates

## 10. Baseline B0

Benchmark the current Task 6C path unchanged:

`B0_current`

Conditions:

- batch semantics = 1 sample / optimizer step;
- strict deterministic ON;
- gradient checkpointing current value;
- current 480-image SAM CPU cache;
- no new Qwen/input cache;
- normal synchronous `.to(cuda)`.

Record full stage breakdown.

This is the reference.

## 11. Candidate B1 — cache immutable source data

Implement only if measured useful:

- source RGB image caching;
- target-mask caching.

Rules:

- CPU-only;
- keyed by immutable sample/image identifiers;
- no model outputs;
- no hidden states;
- no gradients;
- no inference-time GT leakage — this cache is only training supervision/input I/O acceleration.

Verify cached image/mask arrays are byte-identical to disk-loaded versions on multiple records.

Benchmark:

`B1_source_cache`

## 12. Candidate B2 — cache deterministic Qwen preprocessing

This is likely the highest-value CPU optimization.

The following are fixed for a given training sample and do **not** depend on trainable weights:

- processor-generated image tensors;
- `input_ids`;
- `attention_mask`;
- `labels`;
- `image_grid_thw`;
- other deterministic processor tensors;
- prompt/sequence metadata.

Implement a CPU-side preprocessing cache for the fixed training sample.

Preferred design:

- do **not** cache Qwen hidden states;
- do **not** cache logits;
- do **not** cache any trainable-model output;
- preserve exact `TeacherForcedBatch` semantics;
- avoid duplicating large image tensors for same-image paired samples if practical;
- cache may live in RAM, local ignored files, or a hybrid, selected by measured cost.

Before adoption, compare uncached vs cached prepared tensors across at least 16 samples:

- shape identical;
- dtype identical;
- exact tensor equality where expected;
- same `[SEG]` position;
- same labels;
- same visual-token count.

Benchmark:

`B2_preprocessed_cache`

Record:
- cache-build seconds;
- RAM/disk bytes;
- cache hit time;
- first-epoch break-even estimate.

## 13. Candidate B3 — pinned memory + non-blocking H2D

If CPU tensors are cached/prepared, add a safe pinned-memory path.

Potentially add:

```python
TeacherForcedBatch.pin_memory()
TeacherForcedBatch.to(device, non_blocking=True)
```

including tensor-valued `extra_inputs`.

Also support pinned cached target masks / SAM CPU features only if practical.

Rules:

- no semantic changes;
- no unsafe lifetime/reuse bugs;
- no pinning the entire 7.5 GiB SAM cache blindly if Windows/page-lock pressure becomes excessive.

Benchmark variants:

- ordinary H2D;
- pinned + non_blocking.

Reject if slower or destabilizes RAM.

Benchmark:

`B3_pinned_nonblocking`

## 14. Candidate B4 — bounded CPU prefetch / DataLoader

Only after B2/B3 are implemented/measured.

The current training loop consumes one sample synchronously. Test whether a small background input pipeline helps.

Because this is Windows and process spawning/pickling large tensors can be expensive, benchmark conservatively:

- worker/prefetch baseline: 0;
- 2 workers;
- 4 workers;
- optionally 8 workers **only if 4 workers is still beneficial and RAM is safe**.

Use:
- `persistent_workers=True` where applicable;
- bounded `prefetch_factor`;
- pinned memory only if B3 proved useful.

Do not duplicate the full multi-GiB cache into every worker.

A thread-based prefetch queue is allowed instead of multiprocessing if it performs better on this Windows workload.

Choose by measurement, not convention.

Benchmark winner as:

`B4_prefetch_best`

Record why workers or threads were chosen.

## 15. Candidate B5 — gradient checkpointing cost measurement

Current config uses:

`gradient_checkpointing: true`

Measure the same steady-state workload with:

- checkpointing ON;
- checkpointing OFF.

Do **not** automatically adopt OFF.

For OFF record:
- throughput;
- GPU utilization;
- peak allocated/reserved VRAM;
- output/loss/gradient equivalence;
- deterministic status.

Adoption rule:

Checkpointing OFF may become the optimized default **only if all hold**:

1. peak reserved VRAM < **14.0 GiB**;
2. no OOM across the full benchmark;
3. fixed-sample loss/gradient comparison is bit-identical or meets a separately documented strict numerical-equivalence tolerance;
4. steady-state throughput improves by >= **10%**;
5. strict deterministic mode still works.

If not, keep checkpointing ON.

## 16. Determinism cost measurement — measure, do not weaken formal default

Benchmark:

- strict deterministic ON;
- deterministic algorithms OFF while retaining the same seed setup.

Purpose: quantify the speed tax.

Do **not** change formal experiment default from strict deterministic ON in this task.

If OFF is materially faster, document a possible future **development-only fast mode**, but do not silently use it for paper/ablation results.

No Task 6C.5 headline speed claim may mix deterministic modes.

## 17. Batch-size policy

Do **not** implement or adopt true batch size > 1 in Task 6C.5.

Reason:

- current `TeacherForcedBatch` / `[SEG]` extraction / SAM bridge are written around single-sample semantics;
- changing optimizer batch semantics would confound performance engineering with algorithmic changes.

At the end, estimate whether remaining GPU headroom justifies a later dedicated batching task.

Gradient accumulation is not a substitute and must not be presented as a GPU-utilization optimization.

# PART C — Equivalence gates

## 18. Value-preserving optimization gate

Before an optimization becomes the new default pipeline, compare baseline vs optimized on a fixed mini-run.

Use:

- same clean initialization;
- same 8–16 samples;
- same sample order;
- same seed;
- strict deterministic mode;
- same optimizer states;
- same training recipe.

For input/pipeline-only changes, require:

- prepared tensors identical;
- per-step losses bit-identical;
- gradient fingerprint identical;
- post-step trainable-parameter fingerprint identical.

If an optimization cannot meet bit identity because of an inherently different valid execution path (e.g. checkpointing OFF), require a **separate explicit numerical-equivalence report** and do not merge it with the bit-equivalent pipeline group.

The safest bit-equivalent winner should become the default.

## 19. Resource safety

System RAM is 32 GB.

During optimization:

- avoid paging/swap;
- prefer process RSS + cache footprint comfortably below ~24 GiB;
- stop expanding RAM cache if system commit pressure becomes unsafe.

GPU:

- total ~15.9 GiB;
- keep normal optimized strict pipeline with adequate safety margin;
- do not chase 99% GPU utilization at the cost of OOM instability.

No optimization is accepted solely because GPU utilization percentage is higher.

Primary metric is:

> **end-to-end samples/sec at identical training semantics**

Secondary:
- GPU utilization;
- CPU utilization;
- VRAM/RAM;
- first-epoch cost.

# PART D — Decision logic

## 20. Optimization decision

Select one final optimized batch-1 strict-deterministic pipeline.

Prefer the smallest set of changes that delivers most of the gain.

Example structure:

```text
B0 current
→ B2 preprocessed input cache
→ B3 pinned/nonblocking
→ B4 bounded prefetch (only if useful)
→ B5 checkpointing decision
```

Do not keep complexity that adds <~5% benefit unless it solves a clear bottleneck.

## 21. Success categories

Use exactly one:

- `OPTIMIZATION_SUCCESS`
- `OPTIMIZATION_PARTIAL`
- `NO_MEANINGFUL_BOTTLENECK_FIX`
- `INVALID_BENCHMARK`

### `OPTIMIZATION_SUCCESS`

Require:

- accepted optimized path is value-preserving;
- no OOM/paging;
- warm steady-state throughput improves by >= **20%** over B0;
- or GPU utilization rises >= **15 percentage points** with >=10% throughput gain.

### `OPTIMIZATION_PARTIAL`

Correct/value-preserving improvements exist but gains are below the full gate.

### `NO_MEANINGFUL_BOTTLENECK_FIX`

Measured candidates do not improve throughput materially.

### `INVALID_BENCHMARK`

Use only for invalid comparisons, thermal/power instability so severe results are unusable, mismatched model states, or benchmark instrumentation dominating runtime.

# PART E — Thermal / laptop controls

## 22. Avoid thermal benchmarking mistakes

Because this is a laptop GPU/CPU:

- record GPU temperature if available;
- record GPU power;
- keep AC power connected assumption only if observed, otherwise report unknown;
- allow warmup before measurement;
- avoid comparing one cold run against one thermally saturated run.

Interleave/repeat baseline and final winner once if practical:

```text
B0
optimized
B0 repeat
optimized repeat
```

Use repeat variation to judge benchmark stability.

Do not modify BIOS, fan curves, Windows power plan, undervolt/overclock settings, or OEM utilities.

# PART F — Implementation requirements

## 23. Keep optimization modular

Suggested modules/scripts:

```text
buildreasonseg_mvp/input_cache.py
buildreasonseg_mvp/perf.py
scripts/task6c5_profile.py
scripts/task6c5_benchmark.py
configs/mvp/task6c5_throughput.yaml
```

Names may vary, but avoid embedding profiling hacks permanently inside core model logic.

Profiling must be switchable/off by default.

## 24. Tests

Add tests for at least:

1. source-image cache equality;
2. target-mask cache equality;
3. Qwen prepared-batch cache exact equality;
4. cache does not contain hidden states/logits;
5. pinned batch preserves tensors;
6. non-blocking transfer preserves values;
7. optimized and baseline 8-step loss sequence identical;
8. optimized and baseline gradient fingerprint identical;
9. optimized and baseline post-step parameter fingerprint identical;
10. strict deterministic mode remains enabled for the accepted pipeline;
11. feature cache remains CPU-resident between accesses;
12. no cache mutation / VRAM leak regression;
13. no test split used;
14. batch semantics remain one sample per optimizer step;
15. no architecture/loss/optimizer changes;
16. profiler disabled path has negligible/no semantic impact.

Run:

`python -m pytest tests/ -q`

Do not make ordinary pytest load/download full weights unless the existing test policy already isolates such tests.

# PART G — Required artifacts

## 25. Machine-readable outputs

Create:

```text
evaluation/task6c5_benchmark_ids.json
evaluation/task6c5_profile_baseline.json
evaluation/task6c5_variants.json
evaluation/task6c5_equivalence.json
evaluation/task6c5_final_benchmark.json
evaluation/task6c5_resource_usage.json
```

`task6c5_variants.json` must include every tried variant, including slower/rejected ones.

For each variant include:

- exact code/config flags;
- steps measured;
- warmup steps;
- seconds;
- ms/sample;
- samples/sec;
- GPU util mean/median/p90;
- CPU util;
- process RSS;
- system RAM;
- VRAM allocated/reserved peak;
- GPU temperature/power if available;
- cache bytes;
- cold-build time;
- equivalence status;
- adopted/rejected;
- rejection reason.

## 26. Human-readable report

Create:

`docs/task6c5_training_optimization.md`

It must answer:

1. What actually caused CPU≈100% / GPU≈40%?
2. What percentage of step wall time was CPU preparation / H2D / GPU compute?
3. Which cache/preprocessing changes helped?
4. Did pinned memory help?
5. Did workers/prefetch help on Windows?
6. What did gradient checkpointing cost?
7. What did strict determinism cost?
8. What is the final recommended pipeline?
9. What is the achieved speedup?
10. What bottleneck remains?
11. Is true batching likely worth a later task?

# PART H — Do not touch model-quality conclusions

## 27. No Task 6C result reinterpretation

Task 6C model findings remain unchanged:

- no arm solved instruction-conditioned segmentation;
- paired probe remains 0/20;
- deeper conditioning-interface problem remains unresolved.

Task 6C.5 may not claim model-quality improvement because it is not a model-quality experiment.

Do not rerun Task 6C four-arm results unless needed for a tiny equivalence check.

# PART I — Git and Watt

## 28. Git hygiene

Before commit:

- no model weights;
- no checkpoints;
- no large profiler trace files;
- no local cache tensors;
- no `.conda`;
- no dataset JSONL edits;
- no test outputs unrelated to this task.

Large profiler traces remain gitignored.

Recommended commit:

`perf: optimize batch-1 training pipeline`

## 29. Final push

Attempt normal push.

Apply Watt ownership rule from §4:

- pre-existing Watt: use but never close;
- DSH-started Watt: push then close and verify cleanup;
- no Watt needed: do nothing.

Ignore UU completely.

# 30. Handoff

Update `handoff/FROM_DSH.md` with:

1. Verdict
2. Baseline Bottleneck
3. Benchmark Method
4. B0 Baseline
5. Source Cache
6. Qwen Preprocessing Cache
7. Pinned / Nonblocking Transfer
8. Prefetch / Worker Benchmark
9. Gradient Checkpointing Benchmark
10. Determinism Cost
11. Equivalence Proof
12. Final Optimized Pipeline
13. Throughput / GPU / CPU Improvement
14. RAM / VRAM / Thermals
15. Remaining Bottleneck
16. Tests
17. Git / Watt State
18. Recommendation for next project task

# 31. Final DSH web response — Chinese only

Report concisely:

- Task 6C.5 verdict;
- baseline samples/sec;
- optimized samples/sec;
- speedup percentage;
- baseline vs optimized GPU utilization;
- baseline vs optimized CPU utilization;
- main bottleneck found;
- optimizations adopted;
- optimizations rejected;
- strict determinism cost;
- checkpointing ON/OFF result;
- final RAM/VRAM peak;
- whether semantics are bit-equivalent;
- tests;
- commit hash;
- push result;
- Watt handling result.

# 32. STOP

After Task 6C.5:

**STOP.**

Do not start Task 6D.
Do not resume model architecture work.
Do not run 4B.
Do not add `[REF]`.
Do not start full training.

Wait for ChatGPT review.
