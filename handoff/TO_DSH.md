# TO_DSH — Task 6C.7: Frozen Qwen Visual-Feature Cache + Sync-Hotspot Cleanup

> Status: **ACTIVE**
>
> Repository: `BuildReasonSeg`
>
> Purpose: continue performance optimization after Task 6C.6.
>
> Task 6C.6 correctly established that:
> - host data preparation is not the bottleneck;
> - batch-1 execution launches ~57k CUDA kernels/step;
> - ~2,545 host↔device synchronization events remain per step;
> - `torch.compile` is not usable/adoptable in the current environment;
> - the Task 6C.5 gradient-norm optimization is now integrated.
>
> However, **do not declare batch-1 optimization exhausted yet**.
>
> Two issues remain:
> 1. the formal `task6c_train.py` Phase-B path still performs an unnecessary device copy (`moved = batch.to(...)`) and then passes the original CPU `batch` to `runtime.train_step`, which performs its own `.to(device)`;
> 2. the Qwen visual tower is frozen / has no visual LoRA, but its deterministic image features are recomputed every optimization step. Those features may be cacheable without caching any trainable language-model hidden state.
>
> This task investigates and, if safe, adopts those optimizations.
>
> **Do not start batch>1. Do not start Task 6D model-quality work.**

## 0. User-facing language

All narrative DSH UI output must be **Chinese**.

Commands, paths, profiler operator names and metric identifiers may remain English.

## 1. Hard boundaries

Do not change:
- Qwen3-VL-2B base model;
- SAM2.1 Base+;
- LoRA rank/targets;
- `[SEG]`;
- SAM bridge;
- loss weights;
- optimizer/scheduler;
- sample order;
- number of optimizer steps;
- effective batch size;
- dataset;
- inference/GT semantics;
- strict-determinism policy.

Do not use:
- 4B;
- `[REF]`;
- Spatial Relation Encoder;
- Spatial Consistency Loss;
- true batching;
- gradient accumulation as a throughput claim;
- new external datasets.

This is performance engineering only.

## 2. UU / Watt

Ignore UU completely.

For Watt Toolkit:
- inspect only Watt/Steam++ state at task start;
- if pre-existing, leave it alone;
- if DSH starts it for final push, close it afterward through the verified `WM_SYSCOMMAND / SC_CLOSE` lifecycle and verify cleanup;
- training and profiling remain offline.

# PART A — Fix the formal-loop redundant device transfer

## 3. Verify and remove the redundant transfer

Current Phase-B structure in `scripts/task6c_train.py` is effectively:

```python
batch, image = runtime.prepare(sample)
moved = batch.to(runtime.device)

# Phase B:
result = runtime.train_step(batch, ...)
```

and `runtime.train_step()` itself performs:

```python
batch = batch.to(self.device)
```

Therefore the first `moved` allocation is unused in Phase B.

Refactor the formal loop so:
- Phase A moves the batch exactly once because it directly calls Qwen;
- Phase B either passes already-moved `moved` to `runtime.train_step`, or does not create `moved` at all and lets `runtime.train_step` move once.

Prefer the simpler path.

Add a regression test proving Phase B performs one logical device transfer, not two.

Measure the isolated effect using the same Task 6C.6 benchmark protocol.

This is expected to be small; do not overclaim it.

# PART B — Identify the remaining synchronization hotspots

## 4. Resolve the unexplained `aten::item` count

Task 6C.6 profiler reports approximately:
- `aten::item`: 1,051 per step;
- `cudaStreamSynchronize`: ~130 per step;
- total sync events: ~2,545 per step.

Project-owned explicit scalar conversions explain only a few events per step, so the remaining large count is not yet localized.

Run a short profiler with stack/module attribution sufficient to answer:
- which call sites produce the majority of `aten::item`;
- which call sites produce `cudaStreamSynchronize`;
- whether those calls originate from Qwen visual tower, Qwen language model, PEFT/trainable-token wrappers, gradient checkpointing, SAM decoder, project logging/loss code, or other Transformers internals.

Do not commit large traces.

Create a compact ranked table:

```text
callsite/module
item/sync events per step
share
avoidable? yes/no/unknown
```

If stack attribution is unavailable, use scoped profiling by temporarily profiling:
- Qwen vision only;
- Qwen LM only;
- SAM only;
- optimizer only;
without changing training semantics.

Do not patch third-party Transformers internals blindly.

# PART C — Frozen Qwen visual-feature cache

## 5. Establish cache eligibility first

The visual tower must be proven cache-safe before implementation.

Verify on the current runtime:
1. all Qwen visual-tower parameters have `requires_grad=False`;
2. zero LoRA modules exist in the visual tower;
3. visual output for the same image is deterministic under actual training-mode/runtime settings;
4. no training-time state in the visual tower changes between steps;
5. the visual representation injected into the language model is independent of instruction text, LoRA parameters, `[SEG]` adapter and optimizer state.

If any condition fails, do **not** cache visual outputs.

Record:
`evaluation/task6c7_visual_cache_eligibility.json`

## 6. Find the narrowest safe cache boundary

Inspect the installed `transformers` Qwen3-VL implementation and current model forward path.

Find the exact boundary:

```text
pixel_values / image_grid_thw
        ↓
frozen Qwen visual encoder
        ↓
image token embeddings / visual features
        ↓
trainable language-model path with LoRA
```

Prefer caching the **last output produced exclusively by the frozen visual tower**, before it is mixed with trainable text hidden states.

Potential interfaces may include:
- an official `get_image_features`/visual method;
- a model submodule call;
- a carefully defined helper mirroring the official forward.

Do not cache:
- language-model hidden states;
- `[SEG]` hidden states;
- logits;
- anything downstream of trainable LoRA.

Do not monkey-patch opaque internals unless there is no cleaner interface.

## 7. Exact-equivalence proof for visual features

For at least 16 representative samples compare normal forward visual features vs cached/reused features.

Require:
- same shape;
- same dtype;
- exact equality if possible;
- otherwise document max abs/relative difference before proceeding.

Then run an end-to-end 12-step training equivalence gate:
- clean identical initialization;
- same samples/order;
- same per-run seed;
- strict deterministic ON;
- one run recomputes Qwen visual features;
- one run uses cached Qwen visual features.

Preferred adoption category:
`BIT_EQUIVALENT`

Require:
- per-step losses identical;
- gradient fingerprints identical;
- post-step parameter fingerprints identical.

A `NUMERICALLY_EQUIVALENT` path is allowed only with explicit tight tolerances and caution.

Do not adopt if equivalence fails.

## 8. Cache design

If eligible, implement a bounded CPU cache.

Key by immutable **source image identity**, not sample id, because multiple instructions can share one image.

Record:
- feature tensor shape;
- dtype;
- bytes/image;
- 480-image Task 6C estimate;
- full 2,508-WHU-train-image estimate;
- actual cache-build time;
- hit latency.

Stay within 32 GB system RAM.

Do not exceed approximately 8 GB for this new cache without explicit justification.

If the native exact visual feature is BF16, keep it BF16.

Do not pin the whole cache unless separately measured and justified.

# PART D — Benchmark

## 9. Benchmark variants

Use the same fixed Task 6C.5/6C.6 64-record benchmark IDs and protocol:
- 8 warmup;
- 64 measured;
- batch 1;
- strict deterministic ON;
- SAM cache warm;
- `collect_grad_norms=false`;
- same initial state;
- bracket/interleave references due laptop drift.

Benchmark:

### V0
Current integrated Task 6C.6 runtime.

### V1
Formal-loop duplicate-H2D fix only.

### V2
V1 + frozen Qwen visual-feature cache.

### V3
Only if useful: V2 + removal/deferment of project-owned scalar logging syncs.

Do not combine unproven variants before isolated measurements exist.

Report:
- samples/sec;
- speedup vs interpolated/reference baseline;
- GPU utilization;
- CPU utilization;
- kernel count/step;
- `aten::item` count/step;
- sync count/step;
- VRAM;
- RSS;
- cache bytes;
- first-epoch/cold-cache cost.

# PART E — Project-owned scalar synchronization

## 10. Optimize only if it matters

The project currently materializes the four loss tensors and clip norm as Python floats each step.

If profiling proves these are a meaningful fraction of remaining synchronization cost, test deferring scalar materialization to logging intervals.

However:
- do not weaken NaN/Inf safety silently;
- scheduler/optimizer semantics must not change;
- equivalence is mandatory.

If the project-owned scalar reads are negligible, leave them alone.

# PART F — Decision rules

## 11. Adoption rules

Adopt the visual-feature cache only if:
1. eligibility passes;
2. end-to-end equivalence is acceptable;
3. no trainable signal is bypassed;
4. no OOM/paging;
5. warm throughput improves by >= **8%** over V1, or >=5% with a material reduction in sync/kernel count;
6. cold-cache build has a reasonable break-even.

Adopt the duplicate-H2D fix if:
- equivalence is exact;
- code is simpler/correcter even if speed gain is small.

Prefer the simplest valid winner.

## 12. Do not declare batch-1 exhausted until this closes

Only after:
- redundant Phase-B H2D is removed;
- remaining `aten::item`/sync hotspots are localized;
- frozen Qwen visual-cache feasibility is measured;

may the report conclude that true batch>1 is the next performance lever.

# PART G — Artifacts

## 13. Required outputs

Create:

```text
evaluation/task6c7_visual_cache_eligibility.json
evaluation/task6c7_sync_hotspots.json
evaluation/task6c7_equivalence.json
evaluation/task6c7_variants.json
evaluation/task6c7_final_benchmark.json
evaluation/task6c7_resource_usage.json
docs/task6c7_visual_cache_optimization.md
```

If visual caching is impossible, eligibility must explain exactly why.

# PART H — Tests

## 14. Required tests

Add/regress tests for:
1. Phase-B formal loop no longer performs redundant batch transfer;
2. Phase-A behavior remains correct;
3. visual-tower params all frozen under current formal config;
4. zero visual LoRA;
5. visual-cache key is image-based;
6. cache contains no trainable-language hidden state;
7. repeated same-image visual feature is deterministic;
8. cache hit returns correct shape/dtype;
9. uncached/cached visual features equal or within declared tolerance;
10. 12-step end-to-end equivalence;
11. no test split;
12. batch remains 1;
13. no architecture/loss/optimizer/data changes;
14. strict determinism preserved;
15. no `[REF]`/4B/SRE/SCL.

Run:
`python -m pytest tests/ -q`

# PART I — Git / Watt

## 15. Git hygiene

Do not stage:
- visual cache tensors;
- weights;
- checkpoints;
- raw profiler traces;
- `.conda`;
- `local_cache`;
- dataset JSONL changes.

Recommended commit:
`perf: cache frozen Qwen visual features`

If visual caching is rejected and only cleanup remains:
`perf: close remaining batch-1 sync overhead`

## 16. Final push

Apply Watt ownership rule.

Ignore UU entirely.

# PART J — Handoff / final UI

## 17. `handoff/FROM_DSH.md`

Include:
1. Verdict
2. Formal H2D Fix
3. Remaining Sync Hotspots
4. Visual-Tower Cache Eligibility
5. Cache Boundary
6. Equivalence
7. V0/V1/V2/V3 Results
8. Kernel / Sync Changes
9. Throughput / GPU / CPU
10. RAM / VRAM / Cold Cost
11. Tests
12. Git / Watt
13. Whether true batching is now justified

## 18. Final DSH UI — Chinese only

Report:
- verdict;
- duplicate-H2D finding/fix;
- top source of remaining ~1,051 `aten::item`;
- visual cache eligible or not;
- cached tensor shape / bytes per image;
- baseline and final samples/sec;
- speedup;
- GPU utilization;
- sync/kernel-count reduction;
- equivalence category;
- VRAM/RAM;
- tests;
- commit/push;
- Watt handling;
- whether Task 6C.8 true batching should be next.

# 19. STOP

After Task 6C.7:

**STOP.**

Do not start batch>1, Task 6D, 4B, `[REF]`, Spatial Relation Encoder, Spatial Consistency Loss or full training.

Wait for ChatGPT review.
