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

# FROM_DSH — Task 6C.6 Report: Batch-1 Launch-Overhead Optimization & Formal-Path Integration

_This file holds the Task 6C.6 report. The Task 6C.5 report is preserved in git history and in
`docs/task6c5_training_optimization.md`; the Task 6C report is in `docs/task6c_prompt_ablation.md`._

## 1. Verdict

**`OPTIMIZATION_PARTIAL`.**

The Task 6C.5 winner is now genuinely active in the formal training loop, and that single change is
worth **+14.7 %** in an interleaved head-to-head with a `BIT_EQUIVALENT` gate. **No new candidate from
sections 6–10 is adopted**, and each rejection is measured rather than assumed:

| Candidate family | Outcome |
|---|---|
| section 1 integration (`collect_grad_norms=false`) | **adopted**, `BIT_EQUIVALENT`, +14.7 % |
| C1 `torch.compile` | **not adoptable**: inductor cannot run (no Triton); every backend that runs is 9–25 % slower, and all are numerically different |
| C2 optimizer / clipping implementations | **rejected**: the default is already the foreach fast path; `fused=True` is noise-level and not equivalent; `foreach=False` is slower |
| C3 CUDA graphs | **rejected**: `cudagraphs` is −24.7 % and breaks on the first sequence-length change |
| C4 SDPA backends | **no change**: torch already mixes efficient CUTLASS FMHA (248 of 472 calls/step) with math (224); flash is not compiled in; pinning a backend fails or is worse |
| C5 checkpointing OFF | **rejected**: −2.03 % with the two interleaved rounds disagreeing in sign → not reproducible |

Nothing in Task 6C's model-quality conclusions is reinterpreted, and no model, loss, data,
sample-order, optimizer, scheduler or batch-size semantics changed.

## 2. Task 6C.5 Integration Fix

`scripts/task6c_train.py` called

```python
result = runtime.train_step(batch, gt_mask, features, optimizer=optimizer)
```

which defaults to `collect_grad_norms=True`: the measured Task 6C.5 winner was never active in the
formal path. The loop now reads `training.collect_grad_norms` from the config and passes it explicitly,
and the config sets it to `false` for the formal experiment.

* The 528-tensor diagnostic sweep is skipped; **`clip_grad_norm_` still runs** with the same threshold
  and the same parameter groups. Measured: the clip total norm is identical (mean 8.444049) with the
  switch on and off.
* `MvpRuntime.train_step` still defaults to `True`, so Task 6A/6B scripts and
  `tests/test_task6a_bridge.py` are unaffected; a Stage-1 caller that wants the per-group dictionary
  either sets the config key back to `true` or omits the argument.
* The value is recorded in every phase result (`collect_grad_norms`), so a run's report states which
  diagnostic mode it used.
* Regression test: `tests/test_task6c6_launch.py::test_formal_training_path_consumes_collect_grad_norms`
  runs the real `train_phase` against a stub runtime and asserts the forwarded argument for `False`,
  `True` and an absent key.

## 3. Integrated B0.6

B0.6 is the Task 6C.5 adopted path with section 1 wired in: Qwen3-VL-2B + SAM2.1 Base+, BF16, batch 1,
strict determinism ON, gradient checkpointing ON, no source/preprocess cache, no pinning, no prefetch,
`collect_grad_norms=false`, the `P_C` representative path, the fixed 64-record Task 6C.5 benchmark ids,
8 warmup + 64 measured steps, SAM feature cache warm.

| | B0.6 | Pre-integration control |
|---|---|---|
| Final head-to-head, round 1 | 2.5453 samples/s | 2.1882 samples/s |
| Final head-to-head, round 2 | 2.4652 samples/s | 2.1792 samples/s |
| Mean | **2.5053 samples/s** | **2.1837 samples/s** |
| Round-to-round spread | 3.2 % | 0.41 % |
| GPU utilization | 40.3 / 39.6 % | 36.9 / 37.4 % |
| CPU utilization | ~92 % | ~86 % |
| Reserved VRAM | 8.58 GiB | 8.58 GiB |

Measured integration gain: **+14.7 %** head-to-head (the challenger runs at −12.84 % of B0.6's rate), and
+21.5 % in the earlier four-run baseline group whose reference drift was larger (19 %). The Task 6C.5
reference of ≈2.37 samples/s is in the right region but is not reproduced exactly — see §12.

## 4. Profiler Evidence

`evaluation/task6c6_profiler_summary.json`: 8 profiled steps on B0.6, plus a profiled
pre-integration control. Compact summary only; no raw trace is written or committed.

| Measurement | B0.6 | Control |
|---|---|---|
| CUDA kernels per step | **57,341** | 60,930 |
| Median / mean kernel duration | **2.11 µs** / 8.99 µs | — / 8.63 µs |
| Host↔device sync events per step | **2,545** | 4,593 |
| CPU self time per step (profiler-inflated) | 452 ms | 503 ms |
| CUDA launch path share of CPU self time | **31.9 %** | 35 % |
| Top CPU row | `cudaLaunchKernel`, 118,008 calls @ 8.26 µs | 126,200 calls |

Syncs for B0.6 per step: `aten::item` ×1,051, `cudaStreamSynchronize` ×130, `cudaMemcpyAsync` ×313.

**Gradient-norm sweep, quantified:** +3,589 kernels/step, **+2,048 syncs/step**, +35.7 ms CPU self time
and +10.4 ms GPU kernel time per step. That is why removing it was worth 10–21 %.

**Mechanism verdict.** The weak claim is confirmed (host preparation ≈2.4 % of the step, Task 6C.5) and
the **strong claim is now supported** on inflation-independent evidence: >10,000 kernels/step, median
kernel <5 µs, >100 syncs/step, launch path >20 % of CPU self time, and unprofiled GPU utilization
averaging 33 % (26 % / 39 % across the two B0.6 rounds).

**Stated caveat.** CUPTI inflates the profiled wall time by ~an order of magnitude (5.74 s/step vs
~0.4 s unprofiled), so all `*_fraction_of_wall` fields in the artifact are unusable for statements about
the real step; the verdict deliberately uses counts, kernel durations and CPU-self-time composition
only, with GPU utilization taken from the unprofiled baseline. The first version of the verdict said
"not established" because it read the inflated CPU fraction — the criterion was wrong, not the data, and
the corrected criteria are recorded in the artifact.

## 5. `torch.compile` Results

| Variant | Result |
|---|---|
| C1a Qwen, inductor default | **`runtime_failed`: `TritonMissing`** |
| C1a Qwen, `mode="reduce-overhead"` | **`runtime_failed`: `TritonMissing`** |
| C1c combined, inductor default | **`runtime_failed`: `TritonMissing`** |
| C3 combined, `backend="cudagraphs"` | ran, **−24.7 %**, reserved VRAM 12.63 GiB |
| C1a Qwen, `backend="eager"` | ran, −9.2 % |
| C1a Qwen, `backend="aot_eager"` | ran, −10.4 % |
| C1b decoder tail, `backend="aot_eager"` | ran, −11.3 % |
| C1c combined, `backend="aot_eager"` | ran, **−21.2 %**, reserved VRAM **14.41 GiB** (over budget) |

Inductor needs Triton; this environment has no `triton` package and no MSVC toolchain, so the wrapper
installs (`torch.compile()` only builds a wrapper — `compile_seconds` ≈ 1 ms) and then dies on the first
step with `Cannot find a working triton installation`. Failures are recorded, not hidden: a failed
candidate has `status: runtime_failed`, its exception text, a traceback tail, and **no**
`samples_per_sec`. `failure_summary.count = 3`, all three `TritonMissing`.

Recompilation and graph breaks: dynamo reported `aot_autograd.ok`, `frames.ok/total`, and graph breaks
whose cause is `unsupported Tensor.item() call with capture_scalar_outputs=False` /
`aten::_local_scalar_dense.default` — the per-step host read-back of loss scalars and the clip norm.
Variable sequence length (291→321→295→318 tokens) caused `backend="eager"` to fail on its first
post-shape-change step and `cudagraphs` on its third, both with
`size of tensor a (321) must match the size of tensor b (291)`. Fixed-length padding is the workaround
and section 6 forbids it without its own equivalence gate, so it was not attempted.

**Note on seeing the error at all:** without `PYTHONUTF8=1`, the Windows GBK locale turns the inductor
message into a `UnicodeDecodeError` while decoding it, hiding the real cause. The recorded artifact was
produced with UTF-8 mode enabled; the earlier GBK text is not a second failure mode, it is a locale
artefact.

## 6. Optimizer / Clipping Results

| Variant | vs interpolation | Uncertainty | Resolvable | Equivalence |
|---|---|---|---|---|
| `adamw_foreach` | −2.85 % | 2.68 % | yes (negative) | `BIT_EQUIVALENT` (code-identical control) |
| `adamw_fused` | −0.71 % | 2.68 % | no | `NOT_EQUIVALENT` (loss 0.175, params 0.0037) |
| `clip_foreach_true` | −3.37 % | 2.68 % | yes (negative) | `BIT_EQUIVALENT` (code-identical control) |
| `clip_foreach_false` | −2.24 % | 2.68 % | no | trajectory diverges (clip norm mean 8.44 → 11.07) |

* `_default_to_fused_or_foreach` on the actual 528 fp32 parameters resolves to
  `fused=False, foreach=True`, so **the default is already the multi-tensor path** and `foreach=True` is
  a no-op — which is exactly why it is a control: two code-identical variants measured −2.85 % and
  −3.37 %, giving a **noise floor of 3.37 %** for a sequential group on this machine.
* `fused=True` does not help and is not equivalent, so it is rejected on both grounds.
* Learning rates, betas, weight decay, clip threshold and parameter groups are unchanged: every variant
  comes from the same config and the same `trainable_parameter_groups` call.

## 7. CUDA-Graph / Reduce-Overhead Results

`mode="reduce-overhead"` is unreachable (inductor/Triton), and the underlying `cudagraphs` backend was
probed directly: it runs but is **−24.7 %**, and it is **unsafe on this workload** — a captured graph is
replayed for a different token count and raises a shape mismatch on the third step. Blockers recorded
for the record: dynamic sequence shape (fatal, measured), no Triton (fatal for the intended path),
PEFT wrappers and gradient checkpointing (graph breaks visible in the dynamo counters), and the
allocation pattern (reserved VRAM +4 GiB for every compile variant). No brittle custom CUDA-graph
infrastructure was built, per section 8.

## 8. SDPA Backend Audit

| Evidence | Result |
|---|---|
| Attention calls per step | 472 |
| … memory-efficient CUTLASS FMHA (`fmha_cutlassF_bf16_aligned_64x64_rf_sm80`) | **248** |
| … math (`aten::_scaled_dot_product_attention_math`, `softmax_warp_forward`) | **224** |
| … flash / cuDNN | 0 / 0 |
| Force `FLASH_ATTENTION` on the real step | fails: `No available kernel` (torch not compiled with flash attention) |
| Force `EFFICIENT_ATTENTION` on the real step | fails: `No available kernel` (224 sites cannot use it) |
| Force `MATH` on the real step | runs but **changes numerics** (first-step total 9.4481 vs 9.4255) and ~9.7 % slower (unbracketed) |
| One-step default-vs-math identity check | `default_is_math_backend: false` |

Conclusion: torch's automatic selection is already a **mixture** of the efficient kernel where shapes
allow and math elsewhere; there is **no accidental global fallback** to fix, flash attention is simply
absent from this PyTorch build, and installation of extensions is out of scope by section 9. Leaving
backend selection alone is the measured optimum.

## 9. Checkpointing Interaction

Section 10 asks for an interleaved ON/OFF repeat on the best compile candidate. **There is no compile
candidate**, so the comparison was run on the integrated baseline and is labelled as such in the
artifact.

| | Round rates | Mean | Spread |
|---|---|---|---|
| ON | 2.7376 / 2.5844 | **2.6610** | 5.76 % |
| OFF | 2.6241 / 2.5898 | **2.6070** | 1.32 % |

OFF is −2.03 % with the rounds disagreeing in sign (+4.3 %, −0.2 %) → **not reproducible**. Task 6C.5's
−8.58 % single-sweep figure does not survive interleaving, so it becomes "turning checkpointing off does
not help here". **Checkpointing stays ON**, as section 10 requires.

## 10. Equivalence

`evaluation/task6c6_equivalence.json`, 12 steps, same initial trainable state, same samples in the same
order, strict determinism, per-run re-seeding (the Task 6C.5 lesson: LoRA dropout consumes the global
RNG).

| Object | Category | Detail |
|---|---|---|
| **Integration gate** (`collect_grad_norms` false vs true) | **`BIT_EQUIVALENT`** | prepared tensors identical (0 mismatches / 12), losses, gradients and post-step parameters identical |
| Reference run-to-run control | reproducible | losses, gradients and parameters identical |
| `adamw_foreach` | `BIT_EQUIVALENT` | max loss difference 0.0 — code-identical to the default |
| `clip_foreach_true` | `BIT_EQUIVALENT` | max loss difference 0.0 — code-identical to the default |
| `adamw_fused` | `NOT_EQUIVALENT` | loss 0.175, gradients, parameters 0.0037 |
| `det_algorithms_off` | `NOT_EQUIVALENT` | loss 0.0386, parameters 0.0035 — disabling strict determinism changes results, not just speed |
| `compile_qwen_aot_eager` | `NOT_EQUIVALENT` | loss 0.234 |
| `compile_decoder_tail_aot_eager` | `NOT_EQUIVALENT` | loss 0.042 |
| `compile_combined_aot_eager` | `NOT_EQUIVALENT` | loss 0.227 |

Tolerances were predeclared before the runs (loss abs 1e-3, loss rel 1e-4, gradient abs 1e-5,
parameter abs/rel 1e-6) and are recorded in the artifact. The two code-identical controls passing while
every real change fails is evidence that the gate discriminates rather than rubber-stamping.

## 11. Final Adopted Runtime

```
model            Qwen3-VL-2B-Instruct (BF16) + SAM2.1 Hiera Base+ + projection MLP, unchanged
batch            1 sample per optimizer step, no accumulation
determinism      strict (use_deterministic_algorithms, CUBLAS_WORKSPACE_CONFIG=:4096:8)
checkpointing    gradient checkpointing ON
pipeline         no source cache, no preprocessing cache, no pinning, no prefetch
diagnostics      collect_grad_norms = false          <-- the only change adopted in 6C.6
compile          none
optimizer        torch.optim.AdamW, default resolution (foreach=True), config LRs/betas/wd unchanged
clipping         clip_grad_norm_ default (foreach), threshold 1.0
sam bridge       P_C representative path (bridge mode "centre")
```

Recorded in `configs/mvp/task6c_2b_ablation.yaml` (the switch) and
`evaluation/task6c6_final_benchmark.json` (`adopted_runtime`).

## 12. Throughput / GPU / CPU / VRAM

| | B0.6 | Pre-integration |
|---|---|---|
| samples/s (final head-to-head) | **2.5053** | 2.1837 |
| ms/sample | 399 | 458 |
| GPU utilization | 40.3 / 39.6 % | 36.9 / 37.4 % |
| CPU utilization | ~92 % | ~86 % |
| Process RSS | **3.295 GiB** (limit 24) | 3.29 GiB |
| Reserved VRAM | **8.582 GiB** (limit 14) | 8.58 GiB |
| Allocated VRAM | 7.658 GiB | 7.66 GiB |
| Paging / OOM | none | none |

Peaks across *all* experiments, including rejected candidates: reserved VRAM **14.406 GiB**
(`C1c_combined_aot_eager`) and RSS **8.807 GiB** (the profiler process holding Kineto events); system RAM
peak 17.664 GiB of 31.38 GiB; hottest GPU 82.1 °C; highest power 114.7 W.

**On the Task 6C.5 reference.** The spec's ≈2.37 samples/s is not reproduced exactly, and the honest
reading is that this laptop's absolute throughput is not a stable quantity: B0.6 measured 3.0162 and
2.5534 in the baseline group, 2.5453 and 2.4652 in the final group, while the pre-integration control
measured 2.2976 / 2.2865 / 2.1882 / 2.1792 (spread 0.5 %). The interleaved *ratio* is the reproducible
quantity: +14.7 % (final) and +21.5 % (baseline group), both far above the 8 % gate.

## 13. Compile Cold Cost / Break-even

Not applicable, and that is a finding rather than an omission. No inductor variant reaches codegen, and
the backends that do run are net-negative from the first step: `aot_eager` first-step latency was
0.27–15.9 ms against a ~400 ms step, while steady-state throughput was 9–25 % *below* eager. There is no
compile cost to amortise because there is no compile gain — with one caveat for the future: on an
install with Triton, `mode="reduce-overhead"` would still have to solve the variable-sequence-length
problem measured in §7 before it could be evaluated at all.

## 14. Remaining Bottleneck

The step is **dispatch/launch dominated**: 57,341 CUDA kernels per step at a 2.1 µs median, 2,545
host↔device synchronizations per step, the CUDA launch path consuming 31.9 % of CPU self time, and the
GPU only ~33 % utilized. Data preparation is not a factor (≈2.4 %). Every lever that reduces *work per
launch* without changing semantics has now been measured: caching, pinning, prefetch, gradient-norm
instrumentation (adopted), compile, optimizer/clipping implementations, SDPA backends and checkpointing.

What is left is **arithmetic intensity**, i.e. true batching, which is exactly what section 11 keeps out
of scope. The second-order observation worth recording: the 5 per-step host read-backs of loss scalars
and the clip norm (`aten::item`, 1,051 per step in the profiled window) are dynamo graph-break sources
and synchronization points, but at ~30 µs each they are ≈0.04 % of the step — a candidate for later
clean-up, not for this task.

## 15. Tests

`python -m pytest tests/ -q` → **225 passed** (Task 6C.5's 213 plus the 12 new ones), no failures.

`tests/test_task6c6_launch.py` covers all 12 section-17 items: the formal loop consumes
`collect_grad_norms` (functional, stub runtime), `False` removes the sweep while `clip_grad_norm_` still
runs, `True` remains available for Stage-1 diagnostics and as the library default, the integration is
bit-equivalent over a fixed mini-run, `torch.compile` has exactly one opt-in call site and the formal
path never compiles, compile failures carry an explicit status and no throughput number, no
architecture/loss/data/optimizer semantics changed, strict determinism is effective for every measured
variant, the benchmark set is train-split only, batch size is 1, no `[REF]`/4B/SRE/SCL appears, and the
profiler only runs from its own opt-in script.

## 16. Git / Watt

* Committed and pushed to `Autumn-Preface/BuildReasonSeg` on `main`: commit **`ab6a49a`**
  (`perf: reduce batch-1 launch overhead`) plus a follow-up `docs:` commit recording this hash and the
  push result. Remote `main` was at `1e2383c` (Task 6C.5) before this task; the push result was
  `1e2383c..ab6a49a  main -> main`.
* Artifacts: `evaluation/task6c6_integrated_baseline.json`, `task6c6_profiler_summary.json`,
  `task6c6_compile_variants.json`, `task6c6_optimizer_variants.json`, `task6c6_checkpointing.json`,
  `task6c6_equivalence.json`, `task6c6_final_benchmark.json`, `task6c6_resource_usage.json`; report
  `docs/task6c6_launch_optimization.md`. No weights, no checkpoints, no `.conda`, no dataset edits, no
  profiler traces.
* **Watt Toolkit: `watt_preexisting = true`** (`Steam++.exe` running since 14:16:23, accelerator on
  :443/:80). Used for the push if needed and **left running** — DSH did not start it, so DSH did not
  close it. Nothing was force-killed, no hosts file was edited, no certificate or TLS setting changed.
  Every model benchmark ran offline from local cache.

## 17. Recommendation: Task 6D vs Task 6C.7 batching

1. **Task 6C.6's performance work is complete and closed.** With `collect_grad_norms=false` now in the
   formal loop, the batch-1 launch-overhead budget is exhausted by measurement: the remaining cost is
   the shape of the execution (57k tiny kernels per step), not any removable overhead.
2. **If throughput becomes blocking, the next step is a dedicated Task 6C.7 true-batching feasibility
   experiment** — VRAM headroom at batch 2/4 with checkpointing ON, an explicit equivalence story for a
   changed effective batch, and its own adoption gate. Section 11 forbids doing it here, and nothing in
   this task's results authorises it as a "perf tweak".
3. **If throughput is not blocking, return to the model problem, i.e. Task 6D after the prompt.** Task
   6C.5's and 6C.6's results do not change the model position: no arm solved instruction-conditioned
   segmentation, the paired probe is 0/20, and the failure is after the prompt. No performance number
   may be used to argue model quality.
4. **Keep the measurement method.** The fixed 64-record benchmark set, the A/B/A/B interleave, the
   bracketed reference with an interpolated comparison, the `PYTHONUTF8=1` requirement for reading
   `torch.compile` errors, and the per-run re-seeding rule are all reusable and should be the default for
   any future pipeline claim on this machine.
