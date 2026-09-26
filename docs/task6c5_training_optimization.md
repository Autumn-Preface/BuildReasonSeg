# Task 6C.5 — Training-pipeline throughput audit and value-preserving optimization

**Verdict: `OPTIMIZATION_PARTIAL`** — one value-preserving change is adopted, worth **+10.24 %**
interleaved (repeat spread 0.47 %), but that is below the 20 % success gate. The caching and prefetch
candidates that motivated the task were measured and **rejected**: they do not help, and the audit
explains why.

This is a performance-engineering report. Nothing here changes model quality, and the Task 6C model
results are untouched.

## 1. Method

* **Benchmark set** — `evaluation/task6c5_benchmark_ids.json`: 64 records over **32 paired images × 2
  instructions**, drawn deterministically from the Task 6C `P` training subset (11 × L1+L2, 11 × L1+L3,
  10 × L2+L3 ⇒ 22 L1 / 21 L2 / 21 L3). Fixed ids, fixed order, train split only.
* **Warm steady-state** — 8 unmeasured warmup steps then **64 measured optimizer steps**, one sample per
  step, after the SAM2 feature cache is warm. Cold costs are reported separately.
* **Same starting point** — every variant is restored to the identical initial trainable state and gets a
  fresh optimizer, and (see §7) the seed is re-applied per run.
* **Instrumentation** — `nvidia-smi` for GPU utilization/VRAM/temperature/power, `psutil` for CPU and RSS,
  sampled every 0.5 s. A separate **synchronized CUDA-event stage profile** answers where the time goes and
  is never used for throughput numbers.
* **Interleaved repeats** — the quotable comparison is `B0 → optimized → B0 → optimized`.

**An environment correction that mattered.** The first sweep was invalid: `EscapeFromTarkov.exe` started
at 20:40:53 in the middle of it, holding 10.9 GiB of RAM and the GPU. Under that load `B0` measured 1.608
samples/s and two variants measured −22 % / −29 %; the same `B0` measures ≈2.5 samples/s on a quiet
machine. The contaminated sweep is kept in `artifacts/task6c5_contaminated_variants.json` (gitignored) and
is not used for any conclusion. The user closed the game and every number below is from the quiet re-run.

## 2. What actually caused CPU≈100 % / GPU≈40 %?

Both halves of the observation are real, and they are **not** a "CPU data preparation starves the GPU"
story. The synchronized stage profile of the current path answers it (16 samples, 443 ms/sample):

| Stage | Seconds (16 samples) | Share of wall | CUDA-event seconds |
|---|---|---|---|
| `qwen_forward` | 3.554 | **50.1 %** | 3.553 |
| `backward` | 2.727 | **38.5 %** | 2.722 |
| `optimizer_step` | 0.545 | 7.7 % | 0.543 |
| `loss` | 0.097 | 1.4 % | 0.096 |
| `qwen_prepare_total` | 0.054 | **0.76 %** | 0.053 |
| `image_io` | 0.044 | 0.62 % | 0.036 |
| `sam_feature_lookup` | 0.037 | 0.52 % | 0.037 |
| `target_mask_io` | 0.017 | 0.24 % | 0.016 |
| `cpu_to_gpu` | 0.015 | **0.21 %** | 0.014 |

**All CPU data preparation together is ≈2.4 % of the step**; the GPU-side stages account for ≈97 %.
Independently measured: the Qwen processor costs **2.21 ms/sample** (image processor 0.90 ms, chat
template + tokenizer ≈1.3 ms), and a forward-only split gives `qwen_forward` 142.6 ms against
`projection + SAM decode` 12.5 ms.

So the CPU is at ~90 % because it is busy **inside** the forward and backward passes — Python/ATen
dispatch and kernel launches for a 2B model at batch 1 — while the GPU finishes each kernel quickly and
waits for the next launch. That is why GPU utilization sits near 40 %: the GPU is starved by *launch
latency inside the model call*, not by data loading, and it is why every data-side optimization measured
as neutral or negative.

## 3. B0 baseline and the adopted change

| | B0 (current) | Adopted (`B6`) |
|---|---|---|
| Throughput, interleaved mean of 2 rounds | **2.152 samples/s** (465 ms/sample) | **2.373 samples/s** (421 ms/sample) |
| Repeat spread | 4.96 % (2.206 / 2.099) | **0.47 %** (2.378 / 2.367) |
| Speedup | — | **+10.24 %** (per-round +7.8 %, +12.8 %) |
| GPU utilization | 37.3 % | 37.5 % |
| CPU utilization | 84.4 % | 92.5 % |
| Process RSS | 3.30 GiB | 3.73 GiB |
| Peak VRAM | 5.37 GiB allocated / 8.65 GiB reserved | same |
| Single clean sweep | 2.506 samples/s | 2.847 samples/s (+13.6 %) |

The adopted change is one switch:

```
pipeline.skip_grad_norm_instrumentation = true
```

`runtime.train_step` computed a per-parameter gradient-norm sweep over 528 tensors on **every** step and
returned it in `result["grad_norms"]` — and the Task 6C training loop never reads that field. The default
stays `collect_grad_norms=True`, so every existing consumer (Task 6A/6B scripts,
`tests/test_task6a_bridge.py`) is unchanged. Config:
`configs/mvp/task6c5_throughput.yaml`.

## 4. Candidates, including the rejected ones

Single clean sweep; `B0` = 2.506 samples/s. Every variant below is in
`evaluation/task6c5_variants.json` with its flags and its rejection reason.

| Variant | samples/s | vs B0 | GPU % | CPU % | Verdict |
|---|---|---|---|---|---|
| `B0_current` | 2.506 | — | 41.2 | 90.2 | reference |
| `B1_source_cache` | 2.360 | **−5.8 %** | 39.2 | 87.6 | rejected |
| `B2_preprocessed_cache` | 2.432 | **−2.9 %** | 40.0 | 56.3 | rejected |
| `B3_pinned_nonblocking` | 2.180 | **−13.0 %** | 37.6 | 82.7 | rejected |
| `B4_prefetch_threads_2` | 2.369 | **−5.5 %** | 38.5 | 55.4 | rejected |
| `B4_prefetch_threads_4` | 2.363 | **−5.7 %** | 39.0 | 55.7 | rejected |
| `B4_prefetch_threads_8` | 2.272 | **−9.3 %** | 39.5 | 59.0 | rejected |
| **`B6_skip_grad_norm_instrumentation`** | **2.847** | **+13.6 %** | 42.2 | 97.4 | **adopted** |
| `B9_caches_plus_skip_grad_norms` | 2.913 | +16.2 % | 43.7 | 64.4 | bit-equivalent but rejected — see below |
| `B7_caches_skip_prefetch2` | 2.665 | +6.4 % | 40.4 | 60.4 | rejected |
| `B5_no_gradient_checkpointing` | 2.291 | **−8.6 %** | 39.6 | 86.5 | rejected |
| `DET_algorithms_off` | 2.550 | +1.8 % | 38.4 | 90.2 | measured only |

**Why the caches were rejected.** They do reduce CPU work — `B2` drops CPU utilization from 90 % to 56 %
and adds only 0.41 GiB of RAM — but they do not reduce wall time, because the CPU work they remove is
≈2 % of the step (§2). Their measured effect is −2.9 % to −13.0 %, i.e. the caching bookkeeping costs
more than the work it saves at this scale. That is the central, counter-intuitive result of this audit,
and it is the reason the answer to "which cache helped?" is **none**.

This is a throughput rejection, not a correctness one. The cache flag set was gated separately
(`evaluation/task6c5_equivalence_caches.json`: prepared tensors identical over 16 samples, 12-step losses,
gradients and post-step parameters all identical, baseline control reproducible) — so
**`B9` = caches + skip is bit-equivalent to `B0` and measured 2.913 samples/s**. It was still not adopted,
for two reasons that the artifact records: the gain over `B6` is **+2.29 %**, which is *inside this
benchmark's own run-to-run spread* (B0 spread 4.96 %, thermal drift 71→84 °C across the sweep) and far
inside the ~5 % complexity threshold section 20 sets for keeping extra machinery; and every cache adds a
memory bound and a whole class of staleness bugs for that unresolvable difference. The rule the summarizer
applies — *keep the fewest-switch bit-equivalent variant unless a more complex one is more than 5 %
faster* — is recorded in `evaluation/task6c5_variants.json` under `adoption_rule`.

**Why pinned memory was rejected.** The per-step host payload is small (≈2.5 MB of `pixel_values` plus a
few kB of ids), so page-locking and the extra copy inside `pin_batch` cost more than the copy they
accelerate (−13.0 %).

**Why prefetch was rejected.** 2, 4 and 8 background threads were all slower (−5.5 %, −5.7 %, −9.3 %) and
more threads were monotonically worse, so there is no worker count to prefer. Threads were used instead
of `DataLoader` processes because every worker would otherwise need its own copy of the multi-GiB caches,
which section 14 forbids. There is also a correctness reason to keep it off: the pool must be keyed by
**request position** so that step *i* trains on sample *i*; a FIFO queue silently reorders, and a first
implementation keyed by sample id **deadlocked** the benchmark when the warmup pass restarted the
sequence. A pool that can reorder or stall a run is not worth ~1 %.

## 5. Gradient checkpointing and the determinism tax

* **Gradient checkpointing OFF (`B5`)**: 2.291 samples/s versus 2.506 ON → **8.6 % slower**, so it fails
  the section 15 adoption rule on throughput as well as on being a different execution path. It stays
  **ON**.
* **Deterministic algorithms OFF (`DET`)**: 2.550 versus 2.506 → **+1.8 %**, i.e. the strict
  deterministic-algorithms tax is ≈1.8 % on this workload. The formal experiment default stays **strict
  determinism ON**, and no headline number in this report mixes deterministic modes.

## 6. Equivalence gate

`evaluation/task6c5_equivalence.json`, from `scripts/task6c5_equivalence.py`. Same 16 prepared samples,
same 12 optimization steps, same initial trainable state, same optimizer construction, strict
deterministic mode, and the seed re-applied to every run. Result for the adopted flag:

| Check | Adopted (`skip_grad_norm_instrumentation`) | Rejected caches (`source_cache` + `preprocessed_cache` + `skip`) |
|---|---|---|
| prepared tensors identical (elementwise, incl. `extra_inputs`, `[SEG]` position, visual-token count) | **true**, 0 mismatches over 16 samples | **true**, 0 mismatches over 16 samples |
| per-step losses bit-identical (12 steps) | **true** | **true** |
| per-step gradient fingerprints identical | **true** | **true** |
| per-step post-step parameter fingerprints identical | **true** | **true** |
| baseline-vs-baseline control run reproducible | **true** | **true** |
| **bit_equivalent** | **true** | **true** |

The second column comes from `evaluation/task6c5_equivalence_caches.json`; it exists because a candidate
that is rejected should be rejected for a stated reason. It shows the caches are *value-preserving and
still slower*, which is a different and more useful statement than "rejected, unverified".

Both columns share a blind spot: their baseline is *this* code with default flags. The audit also
restructured `train_step` — optional stage timers, the single `torch.autocast` region split into two, an
optional gradient-norm sweep — so a separate control,
`evaluation/task6c5_refactor_control.json` from `scripts/task6c5_refactor_control.py`, reconstructs the
pre-Task-6C.5 step body verbatim (one autocast region, unconditional gradient norms, no timers) and compares
it against the current default path over 4 real samples with the same initialization, order, per-run seed
and strict determinism: losses, gradient fingerprints and post-step parameter fingerprints are all
**identical**, `refactor_value_neutral: true`. The restructuring did not change training semantics.

The control run is part of the artifact because the gate failed the first time for an instructive reason:
with the seed applied only once at start-up, **the baseline did not reproduce itself**. The LoRA adapters
carry `dropout = 0.05`, so every forward draws dropout masks from the global RNG; two runs in one process
therefore differ even when the pipeline is identical. Section 18's "same seed" has to mean *per run*, and
the artifact now records that explicitly.

## 7. Resource accounting

| | Value |
|---|---|
| Peak process RSS | 3.73 GiB (limit 24 GiB) |
| Peak reserved VRAM | 8.65 GiB (limit 14 GiB of 15.894 GiB) |
| SAM2 CPU feature cache | 32 images resident during the benchmark (0.508 GiB) of the 480-image / 7.508 GiB full-training cache (unchanged from Task 6C; shared constants) |
| Paging / OOM | none |
| GPU temperature during the sweeps | 62–84 °C |
| GPU power | 108–124 W |

## 8. What Task 6C.5 did **not** do

No change to the model, SAM bridge, LoRA rank/targets, loss weights, learning rates, optimizer,
scheduler, number or order of optimizer steps, effective batch size, training subset, sample order, data,
or inference/GT semantics. **No batch size > 1 and no gradient accumulation.** No Task 6C
reinterpretation: the four arms still show no arm solving instruction-conditioned segmentation and the
paired probe is still 0/20. No 4B, no `[REF]`, no relation encoder, no consistency loss.

## 9. Tests

`python -m pytest tests/ -q` → **213 passed** in 473 s (7:53). The 16 section-24 items are covered by the 12
model-backed test functions in `tests/test_task6c5_pipeline.py` (several cover two items each; the file runs
in 220 s on its own): cache equality, pinned/non-blocking value preservation, identical
loss/gradient/parameter sequences, strict determinism still on, CPU-resident feature cache, cache
footprints bounded, no cache mutation or VRAM leak, no test split, one sample per optimizer step, no
architecture/loss/optimizer change, and a profiler-off path with no semantic impact. The profiler is off by
default; instrumentation has to be requested explicitly.

## 10. Reproduce

```bash
python scripts/task6c5_benchmark.py --runtime-mode default      # variant sweep
python scripts/task6c5_benchmark.py --runtime-mode no_checkpointing
python scripts/task6c5_benchmark.py --runtime-mode det_off
python scripts/task6c5_profile.py                               # synchronized stage profile
python scripts/task6c5_equivalence.py --flags skip_grad_norm_instrumentation
python scripts/task6c5_equivalence.py \
    --flags source_cache,preprocessed_cache,skip_grad_norm_instrumentation \
    --output evaluation/task6c5_equivalence_caches.json   # gate for the rejected caches
python scripts/task6c5_refactor_control.py                      # pre-6C.5 step body vs current default
python scripts/task6c5_final_benchmark.py                       # interleaved B0 vs adopted
python scripts/task6c5_summarize.py                             # verdict + resource accounting
```
