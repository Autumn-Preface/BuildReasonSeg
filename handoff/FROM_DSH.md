# FROM_DSH — Task 6C.5 Report: Training Pipeline Throughput Audit & Value-Preserving Optimization

_This file now holds the Task 6C.5 report. The previous Task 6C report is preserved in git history and in
full detail in `docs/task6c_prompt_ablation.md`, `evaluation/task6c_comparison.json` and ADR-015._

<!-- ARTIFACT-FACTS:BEGIN -->
dataset_version: v0.1.1
total_samples: 25229
split_train: 15592
split_val: 3884
split_test: 5753
level_1: 17275
level_2: 5036
level_3: 2918
level2_type_a: 2275
level2_type_b: 2761
level3_trivial: 1256
level3_nontrivial: 1662
semantic_policy_version: "1.0"
generator_version: v0.1.1
quality_json_path: evaluation/build_spatial_reason_v0.1.1_quality.json
sample_pack_path: evaluation/build_spatial_reason_v0.1.1_samples
<!-- ARTIFACT-FACTS:END -->

---

## 1. Verdict

**`OPTIMIZATION_PARTIAL`.**

One change is adopted and proven value-preserving: **`skip_grad_norm_instrumentation`** — stop computing the
per-step, per-parameter gradient-norm sweep over 528 tensors that the Task 6C training loop never reads.

| | B0 (current) | Adopted (`B6`) |
|---|---|---|
| Throughput (interleaved, 2 rounds) | 2.152 samples/s | **2.373 samples/s** |
| Speedup | — | **+10.24 %** |
| Repeat spread | 4.96 % | **0.47 %** |
| GPU utilization | 37.3 % | 37.5 % |
| CPU utilization | 84.4 % | 92.5 % |
| Single clean sweep | 2.506 samples/s | 2.847 samples/s (+13.63 %) |

`OPTIMIZATION_PARTIAL` and not `OPTIMIZATION_SUCCESS` because the gain is below section 21's 20 % gate and
the GPU-utilization route (+15 points) was not met either: GPU utilization does not move, because the added
speed comes from doing **less CPU work per step**, not from feeding the GPU better.

Every other candidate — source-image cache, Qwen preprocessing cache, pinned/non-blocking transfer, 2/4/8
prefetch threads, gradient checkpointing off — was **measured and rejected**. Strict deterministic mode
stays ON and gradient checkpointing stays ON. Nothing in Task 6C's model-quality conclusions is
reinterpreted.

## 2. Baseline Bottleneck

The observed CPU≈100 % / GPU≈40 % is real, and it is **not** data loading. Synchronized stage profile of the
unmodified path (`evaluation/task6c5_profile_baseline.json`, 16 samples, 443 ms/sample):

| Stage | Share of step wall |
|---|---|
| `qwen_forward` | **50.13 %** |
| `backward` | **38.47 %** |
| `optimizer_step` | 7.69 % |
| `loss` | 1.36 % |
| `qwen_prepare_total` (processor: image + chat template + tokenizer) | **0.76 %** |
| `image_io` | 0.62 % |
| `sam_feature_lookup` | 0.52 % |
| `target_mask_io` | 0.24 % |
| `cpu_to_gpu` | **0.21 %** |

**All host-side data work together is ≈2.4 % of the step**; GPU-side stages are ≈97 %. The Qwen processor
measures **2.21 ms/sample** (image processor 0.90 ms, chat template + tokenizer ≈1.3 ms) against a ≈400 ms
step. A forward-only split gives `qwen_forward` 142.6 ms against `projection + SAM decode` 12.5 ms.

So the CPU is saturated **inside** the forward and backward passes — Python/ATen dispatch and kernel
launches for a 2B model at batch 1 — while the GPU drains each kernel quickly and waits for the next
launch. GPU utilization near 40 % is launch-latency starvation *inside the model call*, and that is
precisely why every data-side optimization came back neutral or negative.

## 3. Benchmark Method

* **Set** — `evaluation/task6c5_benchmark_ids.json`: 64 records over **32 paired images × 2 instructions**
  from the Task 6C `P` training subset (11 × L1+L2, 11 × L1+L3, 10 × L2+L3 ⇒ 22 L1 / 21 L2 / 21 L3). Fixed
  ids, fixed order, **train split only**.
* **Protocol** — 8 unmeasured warmup steps then **64 measured optimizer steps**, one sample per step, SAM2
  feature cache warm. All variants restored to the identical initial trainable state with a fresh optimizer,
  strict deterministic mode, and the seed re-applied per run (see §11).
* **Instrumentation** — `nvidia-smi` and `psutil` sampled at 0.5 s for GPU util/VRAM/temperature/power, CPU,
  process RSS and system RAM. Timing harness in `buildreasonseg_mvp/perf.py`; caches in `input_cache.py`;
  pipeline in `pipeline.py`.
* **Interleaving** — the quotable comparison is `B0 → optimized → B0 → optimized`, so thermal drift shows
  up as a spread rather than as an apparent gain.
* **One invalid sweep, corrected.** The first sweep was contaminated: `EscapeFromTarkov.exe` started at
  20:40:53 mid-sweep with 10.9 GiB RAM plus GPU. Under that load `B0` measured 1.608 samples/s and two
  variants showed −22 % / −29 %; the same `B0` measures ≈2.5 samples/s on a quiet machine. The contaminated
  file is kept gitignored at `artifacts/task6c5_contaminated_variants.json` and no conclusion uses it. The
  user closed the game; every number here is from the clean re-run.
* **Honest limitation** — the laptop GPU drifts thermally across a sweep (71 → 84 °C). Single-sweep
  differences under ~3 % are not resolvable, which is why the final decision rests on the interleaved run
  and why a 2.29 % difference was not treated as a win (§12).

## 4. B0 Baseline

| Metric | Value |
|---|---|
| Throughput, single clean sweep | **2.506 samples/s** (399.1 ms/sample) |
| Throughput, interleaved mean | **2.152 samples/s** (465 ms/sample; rounds 2.206 / 2.099) |
| GPU utilization | 41.2 % (sweep) / 37.3 % (interleaved) |
| CPU utilization | 90.2 % mean, p90 97.4 % |
| Step wall p50 / p90 | 0.401 s / 0.427 s |
| Process RSS | 3.30 GiB |
| Peak VRAM | 7.74 GiB allocated / 8.64 GiB reserved |
| GPU temperature / power | 76.3 °C / 110.4 W |
| SAM2 CPU feature cache | 32 images resident (0.508 GiB) during the benchmark |

## 5. Source Cache

`PipelineFlags(source_cache=True)` caches decoded RGB frames and boolean GT masks
(`buildreasonseg_mvp/input_cache.py`), storing read-only contiguous copies; cache equality is tested.

* **Rejected on throughput: 2.360 samples/s = −5.80 %** versus B0, GPU 39.2 %, CPU 87.6 %, cache 40.0 MiB.
* The cache is correct — the flag set is bit-equivalent to B0 (`evaluation/task6c5_equivalence_caches.json`)
  — it simply removes ≈0.6 % of the step (`image_io` + `target_mask_io`) and pays more than that in
  dictionary bookkeeping and copy-on-read.
* Cold build 0.58 s.

## 6. Qwen Preprocessing Cache

`PreprocessedCache` memoizes the deterministic Qwen processor output keyed by sample id, bounded at 1024
entries, and explicitly does not store model outputs (`caches_model_outputs: false`).

* **Rejected on throughput: 2.432 samples/s = −2.92 %** versus B0.
* It *did* relieve the CPU: utilization fell **90.2 % → 56.3 %**, and RSS rose 3.30 → 3.71 GiB with
  384.6 MiB of cached tensors. But wall time did not improve, because the work removed was ≈0.76 % of the
  step. This single result is the clearest demonstration of the section-2 diagnosis: a cache can halve CPU
  utilization without buying a single sample per second.
* Equality of cached prepared batches is tested elementwise (ids, `pixel_values`, `extra_inputs`,
  `[SEG]` position, visual-token count).

## 7. Pinned / Nonblocking Transfer

`pin_batch()` + `TeacherForcedBatch.to(device, non_blocking=True)`.

* **Rejected on throughput: 2.180 samples/s = −12.98 %** versus B0, the worst data-side result.
* Why: the per-step host payload is small — ≈2.5 MB of `pixel_values` plus a few kB of ids — so page-locking
  and the extra staging copy cost far more than the copy they overlap. `cpu_to_gpu` was already 0.21 % of the
  step; pinned memory was optimizing 0.2 %.
* Value preservation is tested (`tests/test_task6c5_pipeline.py`), so the rejection is purely a throughput
  result.

## 8. Prefetch / Worker Benchmark

A bounded background preparation pool with 2, 4 and 8 threads, GPU-side work left on the main thread.

| Workers | samples/s | vs B0 |
|---|---|---|
| 2 | 2.369 | −5.46 % |
| 4 | 2.363 | −5.69 % |
| 8 | 2.272 | −9.31 % |

* **All rejected.** More threads are monotonically worse, so there is no worker count to prefer.
* `DataLoader` subprocesses were not used: each worker would need its own copy of the multi-GiB SAM2
  feature cache, which the 24 GiB RSS budget does not allow.
* The pool must be keyed by **request position**, not by sample id and not FIFO. A FIFO queue silently
  handed step *i* a different sample's batch (the caller restarts the sequence after warmup), and an earlier
  sample-id-keyed pool **deadlocked** the benchmark: the bounded buffer stayed full while the consumer asked
  for a different sample, leaving the GPU idle. Both are correctness failures that would silently corrupt
  training, and neither bought anything, so prefetching is off.
* With 8 workers, GPU memory used transiently reached **14 950 MiB of 15 894 MiB** — no OOM, but the
  headroom that protects the 14 GiB reserved-VRAM rule mostly disappears.

## 9. Gradient Checkpointing Benchmark

`B5_no_gradient_checkpointing`: **2.291 samples/s = −8.58 %** versus B0 (GPU 39.6 %, CPU 86.5 %, RSS
3.28 GiB, VRAM unchanged at 8.64 GiB reserved).

Turning checkpointing off does not win back recomputation time here, because the step is
launch-latency-bound rather than compute-bound: removing recomputation removes kernels but not launches.
It fails section 15 on throughput as well as being a different execution path, so **gradient checkpointing
stays ON**.

## 10. Determinism Cost

`DET_algorithms_off` (deterministic algorithms disabled, `cudnn.benchmark=True`): 2.550 samples/s = **+1.78 %**
versus B0 — the strict-determinism tax is ≈1.8 % on this workload, far below the 20 % gate and not worth
trading away cross-process bit reproducibility.

**The formal experiment default stays strict determinism ON**, and no headline number in this report mixes
modes: the throughput table's `B0` and `B6` are both strict, and the +10.24 %/+13.63 % figures are strict
versus strict.

## 11. Equivalence Proof

Two gates were run, both with the same clean initialization, the same 16 prepared samples in the same order,
the same 12 optimization steps, the same optimizer construction, strict deterministic mode and a per-run
re-seeded RNG.

| Check | `skip_grad_norm_instrumentation` (adopted) | `source_cache`+`preprocessed_cache`+`skip` (rejected) |
|---|---|---|
| Prepared tensors identical | **true** (0 mismatches / 16) | **true** (0 mismatches / 16) |
| Per-step losses bit-identical | **true** | **true** |
| Gradient fingerprints identical | **true** | **true** |
| Post-step parameter fingerprints identical | **true** | **true** |
| Baseline-vs-baseline control reproducible | **true** | **true** |
| `bit_equivalent` | **true** | **true** |

Artifacts: `evaluation/task6c5_equivalence.json`, `evaluation/task6c5_equivalence_caches.json`.

**A third control closes the hole the gate cannot see.** The gate's baseline is *this* code with default
flags, so it proves the adopted switch is neutral but **not** that the audit's own restructuring of
`train_step` was. That restructuring wrapped the stages in optional timers, split one `torch.autocast`
region into two, and made the gradient-norm sweep optional. `scripts/task6c5_refactor_control.py`
reconstructs the pre-Task-6C.5 step body verbatim (one autocast region around forward + supervision + loss,
unconditional gradient norms, no timers) and compares it against the current default path over 4 real
samples with the same initialization, order, per-run seed and strict determinism:

| Check | Result |
|---|---|
| Losses identical | **true** |
| Gradient fingerprints identical | **true** |
| Post-step parameter fingerprints identical | **true** |
| `refactor_value_neutral` | **true** |

Artifact: `evaluation/task6c5_refactor_control.json`. Without this control, "value-preserving" would rest on
an argument about autocast policy being stateless; with it, the claim is measured.

**A methodological defect found and fixed by the gate.** The gate first reported the optimized path as
non-equivalent while the prepared tensors matched and the first loss was identical. A control run — the
baseline against itself — failed the same way, which located the cause: the LoRA adapters carry
`dropout = 0.05`, so every forward consumes the global RNG; with the seed applied only once at process
start, two runs *in the same process* see different dropout masks. "Same seed" has to mean per run. This is
recorded in the artifact because it is exactly the kind of thing that makes an equivalence claim look
stronger than it is.

## 12. Final Optimized Pipeline

One switch, nothing else:

```yaml
throughput:
  pipeline:
    skip_grad_norm_instrumentation: true
```

`runtime.train_step(..., collect_grad_norms=False)` skips the 528-tensor gradient-norm sweep. The default
remains `collect_grad_norms=True`, so Task 6A/6B scripts and `tests/test_task6a_bridge.py` are unchanged;
only the Task 6C loop, which discards `result["grad_norms"]`, is affected. Recorded in
`configs/mvp/task6c5_throughput.yaml` together with the rejected flags and why.

**Adoption rule applied** (in `evaluation/task6c5_variants.json` under `adoption_rule`): adopt only a
bit-equivalent flag set whose gain clears ~5 %, and among those keep the fewest-switch variant unless a more
complex one is more than 5 % faster. This is what decided `B6` over `B9`: `B9` (both caches + skip) is also
bit-equivalent and measured 2.913 samples/s, **+2.29 % over `B6`** — inside the benchmark's own run-to-run
spread and inside section 20's complexity threshold — so three extra switches, two memory bounds and a
staleness surface were not taken for an unresolvable difference.

## 13. Throughput / GPU / CPU Improvement

Interleaved, same rounds, same machine, quiet system:

| | B0 | Adopted | Change |
|---|---|---|---|
| samples/s (mean of 2 rounds) | 2.152 | **2.373** | **+10.24 %** |
| ms/sample | 465 | 421 | −9.3 % |
| relative spread | 4.96 % | **0.47 %** | — |
| GPU utilization | 37.3 % | 37.5 % | +0.2 points |
| CPU utilization | 84.4 % | 92.5 % | higher, because more steps complete per second |
| Single clean sweep | 2.506 | 2.847 | +13.63 % |

The honest summary: **throughput ≈ +10 %, GPU utilization unchanged**. The gain is real, reproducible
(spread 0.47 % across rounds) and bit-equivalent, but it is a CPU-side saving, not a GPU-feeding fix.

## 14. RAM / VRAM / Thermals

| | B0 | Adopted | Limit |
|---|---|---|---|
| Process RSS (peak, over all variants) | 3.30 GiB | 3.73 GiB | 24 GiB |
| Peak reserved VRAM | 8.64 GiB | 8.64 GiB (7.74 GiB allocated) | 14 GiB |
| GPU temperature during sweeps | 71–84 °C | | |
| GPU power | 108–124 W | | |
| Paging / OOM | none | none | |

Peak RSS across the whole sweep was **3.73 GiB** (the cache variants), peak reserved VRAM **8.65 GiB**
(`B4_4`). The system was on AC power and no BIOS, fan curve, power plan, undervolt or OEM setting was
touched. Note that the *benchmark* is short: the full-training SAM2 cache is 480 images / 7.508 GiB and is
unmodified from Task 6C, so the training-time RSS figure remains the Task 6C one.

## 15. Remaining Bottleneck

The step is a **serial CPU-launch-bound chain at batch 1**: `qwen_forward` 50 % + `backward` 38 % + optimizer
8 %, with the CPU at ~90–100 % and the GPU idle between small kernels at ~40 % utilization. Data
preparation is not the problem (≈2.4 %), which is why the entire cache/prefetch/pinning family regressed.

That leaves exactly one evidence-backed lever: **true batching**, which raises arithmetic intensity per
launch instead of shaving a host-side fraction. It is explicitly out of Task 6C.5's scope — section 17
forbids batch > 1 and gradient accumulation here, and an effective batch change is a different experiment
that needs its own authorization, its own equivalence story and a VRAM budget check. It should not be
smuggled in as a "performance" change.

## 16. Tests

`python -m pytest tests/ -q` → **213 passed**, 31 warnings, 473.52 s (7:53).

The 16 section-24 items are covered by the 12 model-backed test functions in
`tests/test_task6c5_pipeline.py` (220 s on its own; several functions cover two items each): source-image
cache equality, target-mask cache equality, prepared-batch cache exact equality, no hidden states/logits in
caches, pinned batch preserves tensors, non-blocking transfer preserves values, identical 8-step loss
sequence, identical gradient fingerprints, identical post-step parameter fingerprints, strict determinism
still enabled for the accepted pipeline, feature cache remains CPU-resident, cache footprints bounded, no
cache mutation / VRAM leak, no test split used, batch semantics one sample per optimizer step, no
architecture/loss/optimizer change, and a profiler-off path with no semantic impact.

`scripts/task6c5_refactor_control.py` adds the cross-version control described in §11; it is a control
artifact rather than a pytest case because it needs the real weights.

## 17. Git / Watt State

* Working tree committed and pushed to `Autumn-Preface/BuildReasonSeg` on `main`; commit
  `412cdf5` — `perf: optimize batch-1 training pipeline` — plus small follow-up `docs:` commits recording
  this hash and the push result. Remote `main` was at `3f7ccc5` (Task 6C) before this task.
* Artifacts: `evaluation/task6c5_benchmark_ids.json`, `task6c5_profile_baseline.json`,
  `task6c5_variants.json`, `task6c5_equivalence.json`, `task6c5_equivalence_caches.json`,
  `task6c5_refactor_control.json`, `task6c5_final_benchmark.json`, `task6c5_resource_usage.json`; report
  `docs/task6c5_training_optimization.md`. No weights, no checkpoints, no `.conda`, no dataset edits, no
  profiler traces; the contaminated sweep stays gitignored under `artifacts/`.
* **Watt Toolkit: `watt_preexisting = true`.** `Steam++.exe` was already running before this task started
  (since 14:16:23, with `Steam++.Accelerator.exe` on :443/:80 and its hosts block present). Per section 4
  it was **used for the push and left running** — DSH did not start it, so DSH did not close it, and it was
  never force-killed, no hosts file was edited and no certificate or TLS setting was changed.

## 18. Recommendation for next project task

1. **Do not repeat the caching work.** It is now measured, gated and closed: the caches are
   value-preserving and *slower*; the CPU-preparation hypothesis is dead (≈2.4 % of the step).
2. **If throughput matters again, the only justified next step is a dedicated batch>1 feasibility task** —
   measure VRAM headroom at batch 2/4 with gradient checkpointing ON, decide the equivalence story for a
   changed effective batch, and treat it as a new experiment rather than a perf tweak. It is not authorized
   by Task 6C.5.
3. **Return to the real blocker.** Task 6C's model-quality position is unchanged: no arm solved
   instruction-conditioned segmentation, the paired probe is 0/20, and the problem is after the prompt.
   The throughput work does not change that, and no performance number may be used to argue model quality.
4. **Keep the fixed measurement assets.** The fixed 64-record benchmark set, the interleaved protocol, the
   per-run re-seeding rule and the `B0 → candidate` comparison are reusable and should be the default for
   any future pipeline claim.
