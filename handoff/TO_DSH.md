# TO_DSH — Task 6C.6: Batch-1 GPU Launch-Overhead Optimization & Formal-Path Integration

> Status: **ACTIVE**
>
> Repository: `BuildReasonSeg`
>
> Purpose: continue training-performance work after Task 6C.5 showed that data loading/preprocessing is not the bottleneck.
>
> Task 6C.5 found that host-side data work is only ≈2.4% of step time, while Qwen forward/backward dominate. GPU utilization remains ~37–41% at batch 1. Skipping unused gradient-norm instrumentation improves throughput by ~10.24%, while caches/pinning/prefetch do not help.
>
> This task targets **model-call / kernel-launch overhead while preserving batch-1 training semantics**.
>
> **Do not begin Task 6D model-quality work. Do not use true batch size > 1 yet.**

## 0. User-facing language

All narrative text shown to the user in DSH web/chat must be **Chinese**.

Commands, paths, code identifiers, raw profiler names and metric names may remain English.

# PART A — First fix: integrate the already-adopted Task 6C.5 optimization

## 1. Task 6C.5 adoption is not yet wired into the formal training loop

Current `scripts/task6c_train.py` still calls:

```python
runtime.train_step(batch, gt_mask, features, optimizer=optimizer)
```

which defaults to:

```python
collect_grad_norms=True
```

Therefore the measured Task 6C.5 winner is not yet actually active in the main Task 6C-style training path.

Fix this first.

### Required behavior

Add a config-controlled setting, preferably under:

```yaml
training:
  collect_grad_norms: false
```

or an equally explicit name.

The formal joint-training loop must pass:

```python
collect_grad_norms=cfg_value
```

to `runtime.train_step`.

Rules:

- Stage-1 / smoke diagnostics that explicitly need per-group gradient norms may keep it `true`.
- Formal multi-step training should default to `false` unless the caller genuinely consumes the detailed gradient-norm dictionary.
- `clip_grad_norm_` is **not** removed. Only the unused diagnostic sweep is removed.
- Existing behavior remains available through config.

Add a regression test proving the formal training path actually consumes the setting.

# PART B — Watt policy

## 2. Ignore UU completely

Do not inspect, search, stop, diagnose or mention UU.

Only Watt Toolkit matters.

## 3. Watt ownership rule

At task start inspect Watt/Steam++ only.

- If Watt is already running: record `watt_preexisting=true`, use it if necessary, and **do not close it**.
- If Watt is not running: keep it off during benchmarks. If push needs it, DSH may start it using the already validated lifecycle, then close it with the validated `WM_SYSCOMMAND / SC_CLOSE` path and verify cleanup.

Never force-kill or edit hosts/certificates/TLS verification.

All model benchmarks remain offline from local cache.

# PART C — Benchmark baseline

## 4. New baseline B0.6

The baseline for this task is **Task 6C.5's adopted batch-1 path**, after §1 is integrated:

- Qwen3-VL-2B
- SAM2.1 Base+
- BF16
- batch 1
- strict determinism ON
- gradient checkpointing ON
- no source/preprocess cache
- no pinning/prefetch
- `collect_grad_norms=false`
- same P_C representative path
- same 64-record Task 6C.5 benchmark IDs
- 8 warmup + 64 measured optimization steps
- SAM feature cache warm

Run an interleaved confirmation of the integrated baseline.

Target reference is approximately 2.37 samples/s, but do not force agreement; report current measured value.

# PART D — Diagnose the launch-bound model execution more directly

## 5. Short PyTorch profiler diagnosis

Task 6C.5 established that forward/backward dominate, but "kernel-launch latency" was inferred rather than fully isolated.

Run a short profiling window (e.g. 6–12 representative steps) using `torch.profiler` or an equivalent already-installed PyTorch profiler.

Do not commit huge raw traces.

Report:

- CPU self time by top operators;
- CUDA time by top operators;
- number of CUDA kernel launches if obtainable;
- average CUDA-kernel duration;
- number of CPU↔CUDA synchronization events;
- major graph breaks / Python hotspots if visible;
- SDPA/attention kernels actually selected;
- whether the workload is dominated by many small kernels, synchronization, memory-bound ops, or another mechanism.

Create a compact machine-readable summary, not a multi-hundred-MB trace.

The report must distinguish:

> "data preparation is not the bottleneck"

from the stronger claim:

> "kernel launch overhead is definitely the bottleneck"

Only make the stronger claim if this profiler supports it.

# PART E — Candidate C1: `torch.compile`

## 6. Test `torch.compile` without changing training semantics

This is the main candidate because Task 6C.5 indicates model-call granularity / dispatch overhead.

Test carefully, in increasing scope.

### C1a — compile Qwen language model only

Test supported modes such as:

- default
- `mode="reduce-overhead"`

Do not use `max-autotune` unless compile time is reasonable and it does not introduce unsafe/unstable behavior.

### C1b — compile only Projection + SAM mask-decoder path

This path is smaller and may or may not matter.

### C1c — compile the combined trainable model path

Only if C1a/b show the stack is compatible.

### Requirements

For each compile variant record:

- compile success/failure;
- compile time;
- first-step latency;
- recompilation count;
- graph-break count/reasons where observable;
- warm steady-state samples/s;
- GPU utilization;
- CPU utilization;
- VRAM;
- numerical-equivalence status;
- whether strict deterministic mode remains effective.

Do not hide compile failures.

### Dynamic-shape caution

Sequence lengths vary across samples.

If this causes recompilation:

- report it;
- try dynamic-shape support if available;
- do not introduce fixed-length padding in this task unless a separate exact-equivalence gate proves it safe.

No architecture modification.

# PART F — Candidate C2: optimizer / gradient clipping implementation

## 7. Audit optimizer and clipping launch overhead

Current optimizer/clipping is a smaller share than Qwen forward/backward, but it is measurable.

Test without changing optimizer mathematics:

### AdamW implementation

Inspect whether current PyTorch automatically uses foreach, fused, or single-tensor path.

Benchmark explicit:

- current/default;
- `foreach=True` if supported;
- `fused=True` if supported on current parameter groups/dtypes.

### Gradient clipping

Inspect whether `torch.nn.utils.clip_grad_norm_` is using foreach.

Benchmark explicit `foreach=True` where supported.

### Equivalence

Because fused/foreach paths can change floating-point reduction order:

- run strict numerical-equivalence checks;
- do not call them bit-equivalent unless they truly are;
- preserve strict determinism;
- if optimizer-state/parameter differences exceed a tight predeclared tolerance, reject.

Do not change LR, beta values, weight decay, clipping threshold or parameter groups.

# PART G — Candidate C3: CUDA graph / reduce-overhead execution

## 8. CUDA graph feasibility

Only test if installed PyTorch / `torch.compile(mode="reduce-overhead")` exposes a safe path.

Do not build custom brittle CUDA-graph infrastructure unless clearly justified.

Record blockers such as:

- dynamic sequence shape;
- dropout RNG behavior;
- PEFT wrappers;
- gradient checkpointing;
- allocation pattern;
- dynamic SAM tensors.

If unsupported, document and reject cleanly.

# PART H — Candidate C4: attention backend audit

## 9. Verify actual SDPA backend

Do not install FlashAttention or new CUDA extensions in this task.

Inspect which PyTorch SDPA backend is actually used for the Qwen workload on this RTX 5080.

If PyTorch offers multiple already-built backends, benchmark only already-available safe choices.

Examples may include flash SDPA, memory-efficient SDPA, or math fallback.

Do not force unsupported kernels.

Report actual backend selection and whether any accidental fallback is occurring.

# PART I — Candidate C5: checkpointing interaction

## 10. Gradient checkpointing interaction

Task 6C.5 found checkpointing OFF slower in one clean sweep.

Because that result was not interleaved and is counterintuitive, do not use it as a universal claim.

For the **best compile candidate only**, compare checkpointing ON vs OFF using an interleaved repeat if VRAM is safe.

Do not adopt OFF unless:

- throughput improves reproducibly;
- peak reserved VRAM < 14.0 GiB;
- strict determinism/equivalence remains acceptable.

Otherwise retain ON.

# PART J — No true batch > 1 yet

## 11. Batch policy

This task deliberately remains true batch 1.

Reason: batch > 1 changes optimization semantics unless the training loop is redesigned carefully; the current goal is to exhaust launch-overhead reductions that can preserve current semantics.

At the end, if none of C1–C5 yields a material gain, formally recommend a separate **Task 6C.7 true-batching feasibility experiment**.

Do not implement it here.

# PART K — Equivalence gates

## 12. Baseline equivalence

After integrating `collect_grad_norms=false`, prove again:

- identical prepared tensors;
- same 12-step losses;
- same gradient fingerprints;
- same post-step parameter fingerprints

against the pre-integration `collect_grad_norms=true` path under per-run reseeding.

This should be bit-identical.

## 13. Candidate equivalence categories

Classify candidates as:

### `BIT_EQUIVALENT`
Exact loss/gradient/post-step fingerprints.

### `NUMERICALLY_EQUIVALENT`
Not bit-identical, but within predeclared tight tolerances and no semantic difference.

Record maximum loss absolute/relative difference, gradient difference, and parameter max absolute/relative difference after fixed steps.

### `NOT_EQUIVALENT`
Differences exceed tolerance or formal determinism breaks.

Do not adopt `NOT_EQUIVALENT`.

For formal paper/ablation runs, prefer `BIT_EQUIVALENT`.

# PART L — Performance decision rules

## 14. Primary metric

Primary:

> warm steady-state end-to-end samples/sec at identical batch-1 training semantics

Secondary:

- GPU utilization;
- CPU utilization;
- kernel-launch/graph-break metrics;
- compile overhead;
- VRAM/RAM;
- first-epoch wall time.

## 15. Adoption threshold

Adopt a new runtime optimization only if:

- equivalence is acceptable;
- strict deterministic mode remains available;
- no OOM/paging;
- warm throughput improves >= 8% over integrated B0.6 **or**
- it improves >=5% while materially reducing run-to-run variance / CPU overhead / compile graph breaks.

Prefer the simplest winner.

Do not stack optimizations that each add complexity for <3% additional gain unless they address a demonstrated blocker.

# PART M — Outputs

## 16. Required artifacts

Create:

```text
evaluation/task6c6_integrated_baseline.json
evaluation/task6c6_profiler_summary.json
evaluation/task6c6_compile_variants.json
evaluation/task6c6_optimizer_variants.json
evaluation/task6c6_equivalence.json
evaluation/task6c6_final_benchmark.json
evaluation/task6c6_resource_usage.json
docs/task6c6_launch_optimization.md
```

If a candidate cannot run, keep its failure record.

Do not commit huge profiler trace files.

# PART N — Tests

## 17. Required tests

Add/regress tests for at least:

1. formal training config consumes `collect_grad_norms`;
2. setting false removes diagnostic sweep but leaves `clip_grad_norm_`;
3. true path remains available for Stage-1 diagnostics;
4. integration is bit-equivalent over fixed mini-run;
5. compile wrapper can be disabled cleanly;
6. compile failure falls back only when explicitly configured — no silent fallback in formal benchmark;
7. no architecture/loss/data/optimizer semantics changed;
8. strict deterministic mode preserved for accepted candidate;
9. no test split;
10. no batch > 1;
11. no `[REF]`, 4B, SRE, SCL;
12. profiler is disabled in normal training.

Run full:

`python -m pytest tests/ -q`

# PART O — Watt / Git

## 18. Final push

Training/profiling is offline.

At final push:

- if Watt was pre-existing, use it if necessary and leave it running;
- if DSH started Watt, close it after push using validated `WM_SYSCOMMAND / SC_CLOSE` and verify cleanup;
- ignore UU entirely.

Recommended commit:

`perf: reduce batch-1 launch overhead`

# PART P — Handoff

## 19. `handoff/FROM_DSH.md`

Include:

1. Verdict
2. Task 6C.5 Integration Fix
3. Integrated B0.6
4. Profiler Evidence
5. `torch.compile` Results
6. Optimizer / Clipping Results
7. CUDA-Graph / Reduce-Overhead Results
8. SDPA Backend Audit
9. Checkpointing Interaction
10. Equivalence
11. Final Adopted Runtime
12. Throughput / GPU / CPU / VRAM
13. Compile Cold Cost / Break-even
14. Remaining Bottleneck
15. Tests
16. Git / Watt
17. Recommendation: Task 6D vs Task 6C.7 batching

## 20. Final DSH UI response — Chinese only

Report:

- verdict;
- integrated baseline samples/s;
- best accepted samples/s and speedup;
- GPU utilization before/after;
- profiler-supported bottleneck;
- compile success/failure and graph breaks;
- optimizer/clipping result;
- SDPA backend;
- checkpointing result;
- equivalence category;
- VRAM;
- tests;
- commit/push;
- Watt handling;
- whether batch>1 is now the next performance lever.

# 21. STOP

After Task 6C.6:

**STOP.**

Do not start true batch > 1, Task 6D, 4B, `[REF]`, Spatial Relation Encoder, Spatial Consistency Loss, or full training.

Wait for ChatGPT review.
