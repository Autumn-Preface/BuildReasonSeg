# Task 6C.6 — Batch-1 launch-overhead optimization and formal-path integration

**Verdict: `OPTIMIZATION_PARTIAL`.** The Task 6C.5 winner is now genuinely active in the formal
training loop, and that integration is worth **+14.7 %** head-to-head (interleaved) with a
bit-equivalent gate. **No new candidate is adopted**: `torch.compile` cannot run on this install at
all, and every backend that does run is slower or numerically different; the optimizer/clipping
implementations are already optimal or worse; SDPA is already using the best backend it can under
strict determinism; gradient checkpointing stays ON.

This is a performance-engineering report. Nothing here changes the model, the loss, the data, the
optimizer mathematics or the sample order, and it does not reinterpret any Task 6C result.

## 1. Verdict summary

| | Value |
|---|---|
| Integration wired into the formal loop | `training.collect_grad_norms: false`, passed explicitly by `scripts/task6c_train.py` |
| Integrated baseline B0.6 | **2.505 samples/s** (final head-to-head mean; 2 rounds: 2.545 / 2.465) |
| Pre-integration control | **2.184 samples/s** (2.188 / 2.179) |
| Integration gain, interleaved head-to-head | **+14.7 %** (B0.6 is 12.8 % faster than the challenger's rate) |
| Integration gain, 4-run baseline group | +21.5 % (higher reference drift) |
| Integration equivalence | **`BIT_EQUIVALENT`** (prepared tensors, 12-step losses, gradients, post-step parameters) |
| New candidates adopted | **none** |
| Adoption category | `OPTIMIZATION_PARTIAL` |

## 2. The integration fix (task section 1)

`scripts/task6c_train.py` called `runtime.train_step(batch, gt_mask, features, optimizer=optimizer)`,
which defaults to `collect_grad_norms=True`, so the Task 6C.5 winner was measured but never active.
It now reads the configured value and passes it explicitly:

```yaml
training:
  collect_grad_norms: false
```

* `clip_grad_norm_` is untouched: same threshold, same parameter groups, same position. Only the
  unused 528-tensor diagnostic sweep is gone. Measured: the clip total norm is **identical**
  (8.444049 mean, and bit-identical first step) with the sweep on and off.
* The library default is still `True`, so Task 6A/6B, `tests/test_task6a_bridge.py` and any Stage-1
  script that reads `result["grad_norms"]` keep working; `tests/test_task6c6_launch.py` runs the
  formal `train_phase` against a stub runtime and asserts the forwarded value for `False`, `True`
  and an absent key.
* Equivalence gate (`evaluation/task6c6_equivalence.json`): prepared tensors identical
  (0 mismatches / 12 samples), 12-step losses identical, gradient fingerprints identical, post-step
  parameter fingerprints identical, baseline control reproducible → **`BIT_EQUIVALENT`**.

## 3. Measurement method, and the drift problem

Hardware: RTX 5080 Laptop (15.894 GiB), 32 GiB RAM, Windows 11, torch 2.13.0+cu132.

Protocol: Task 6C.5's fixed benchmark set `evaluation/task6c5_benchmark_ids.json` (64 records over 32
paired images, **train split only**), 8 warmup + 64 measured optimizer steps, batch 1, warm SAM2
feature cache, `nvidia-smi` + `psutil` sampled at 0.5 s, strict determinism ON, gradient checkpointing
ON. Every variant gets a clean runtime; the measured step is the **formal** path (`runtime.prepare` →
`features_for` → `runtime.train_step`), not the Task 6C.5 pipeline wrapper.

**The machine drifts, and that is a first-class result of this task.** Identical configurations
measured minutes apart differ by up to **19 %** (`reference_drift_percent` in
`evaluation/task6c6_compile_variants.json`). The pattern is systematic: the first measured variant in a
process is the fastest, and the pre-integration control path is far more stable (spread 0.4–0.5 %) than
the integrated fast path (spread 3–17 %). The likely reason is that the control is dominated by
~2,048 extra host↔device synchronizations per step (measured below), which makes it latency-bound and
therefore insensitive to clock/thermal state, while B0.6 still has a compute-sensitive component.

Consequences, applied everywhere in this task:

* every group brackets its candidates with reference runs, and candidates are compared against the
  **linear interpolation** of the bracketing references (`speedup_percent_vs_interpolated_reference`);
* each candidate carries `local_reference_uncertainty_percent`, and a gain inside that band is marked
  `gain_resolvable: false` and is **not** adoptable on that number;
* the headline comparison (B0.6 vs pre-integration) is a strict **A/B/A/B** interleave.

Two measured controls bound what a sequential group can resolve: `adamw_foreach` and
`clip_foreach_true` are **code-identical to the reference** on this build (torch already resolves
`AdamW` to `foreach=True` for these 528 fp32 parameters, and `clip_grad_norm_` already defaults to
foreach), yet they measured **−2.85 %** and **−3.37 %**. That is the noise floor
(`no_op_control_noise_floor_percent: 3.369`), and it is why a 2–3 % "win" from any candidate here would
have been meaningless.

## 4. Profiler evidence (task section 5)

`evaluation/task6c6_profiler_summary.json`, 8 profiled steps on B0.6 plus a profiled pre-integration
control. No raw trace is written or committed.

| Measurement | B0.6 | Pre-integration control |
|---|---|---|
| CUDA kernels per step | **57,341** | 60,930 |
| Median kernel duration | **2.11 µs** | — |
| Mean kernel duration | 8.99 µs | 8.63 µs |
| Host↔device sync events per step | **2,545** | 4,593 |
| CPU self time per step (profiler-inflated) | 452 ms | 503 ms |
| CUDA launch path as share of CPU self time | **31.9 %** | 35 % |
| Largest CPU self-time row | `cudaLaunchKernel`, 118,008 calls at 8.26 µs | 126,200 calls |

Sync breakdown for B0.6 per step: `aten::item` ×1,051, `cudaStreamSynchronize` ×130,
`cudaMemcpyAsync` ×313, `cudaDeviceSynchronize` ×0.25.

**The gradient-norm sweep, quantified:** +3,589 kernels per step, **+2,048 synchronizations per step**,
+35.7 ms CPU self time and +10.4 ms GPU kernel time per step. That is the mechanism behind the
integration's value: the sweep is a host-side loop over 528 parameters, each `.norm()` forcing a
device read-back.

**Verdict on the mechanism** (`mechanism_verdict`): the *weak* claim is confirmed (host data
preparation is ≈2.4 % of the step, Task 6C.5), and the **strong claim is now supported** on
inflation-independent evidence — >10,000 kernels/step, median kernel <5 µs, >100 syncs/step, launch
path >20 % of CPU self time, and unprofiled GPU utilization averaging 33 % (26 % and 39 % across the two
B0.6 rounds).

**One caveat, stated rather than hidden:** CUPTI inflates the profiled wall time by roughly an order of
magnitude (5.74 s per step versus ~0.4 s unprofiled), so every `*_fraction_of_wall` field in the
artifact is unusable as a statement about the real step. The verdict deliberately uses only counts,
per-kernel durations and CPU-self-time composition; the unprofiled GPU utilization comes from the
baseline artifact. This is why the earlier version of the verdict said "not established" — that version
was reading the inflated fraction, and the criterion was wrong, not the measurement.

## 5. `torch.compile` (task sections 6 and 8)

**`torch.compile` cannot be used as intended on this machine.** The failures are structural, not
matters of tuning:

| Variant | Result |
|---|---|
| C1a Qwen, inductor default | **failed: `TritonMissing`** |
| C1a Qwen, `mode="reduce-overhead"` | **failed: `TritonMissing`** |
| C1c combined, inductor default | **failed: `TritonMissing`** |
| C3 combined, `backend="cudagraphs"` | ran, **−24.7 %** |
| C1a Qwen, `backend="aot_eager"` | ran, −10.4 % |
| C1b decoder tail, `backend="aot_eager"` | ran, −11.3 % |
| C1c combined, `backend="aot_eager"` | ran, **−21.2 %**, reserved VRAM 14.41 GiB (over the 14 GiB budget) |
| C1a Qwen, `backend="eager"` | ran, −9.2 % |

Inductor needs Triton to generate GPU kernels; this environment has no `triton` package and no MSVC
toolchain, so the wrapper installs and then dies on the first step. `reduce-overhead` inherits the same
failure, which also settles the CUDA-graph question in its intended form.

`apply_compile` reports `compiled: true` for an inductor variant because `torch.compile()` only builds
a wrapper; the real error surfaces on first call, so `run_variant` catches it and records
`status: runtime_failed` with the exception text and the traceback tail. Nothing falls back silently:
a failed candidate has no `samples_per_sec` at all.

Two further measured blockers for the graph-replay backends, both on **variable sequence length**
(token counts 291→321→295→318 across consecutive samples):

* `backend="eager"` raised `The size of tensor a (321) must match the size of tensor b (291)` on its
  first step after a shape change;
* `backend="cudagraphs"` raised the same class of error on its third step — a captured CUDA graph
  replayed for a different token count.

Fixed-length padding would be the workaround, and section 6 forbids it without its own exact-equivalence
gate. Graph breaks reported by dynamo point at one cause:
`unsupported Tensor.item() call with capture_scalar_outputs=False` and
`aten._local_scalar_dense.default`, i.e. the per-step host read-back of loss scalars and the clip norm.

Equivalence: every `aot_eager` candidate that ran is **`NOT_EQUIVALENT`** (maximum loss difference
0.042–0.234 against a predeclared 1e-3 tolerance), so none could be adopted even if it had been faster.

**Cold cost / break-even:** not applicable — no compile variant reaches a steady state worth
comparing. Compile time itself is unmeasurable for inductor (it fails before codegen) and ≈1 ms for the
wrapper-only backends; the `aot_eager` runs pay their cost on the first step (5.3 ms to 15.9 ms versus
0.27 ms for the first step of an already-warm compiled Qwen), and then run slower than eager anyway.

## 6. Optimizer and clipping (task section 7)

| Variant | vs interpolation | Uncertainty | Resolvable | Equivalence |
|---|---|---|---|---|
| `adamw_foreach` | −2.85 % | 2.68 % | yes (negative) | `BIT_EQUIVALENT` — code-identical control |
| `adamw_fused` | −0.71 % | 2.68 % | no | `NOT_EQUIVALENT` (loss 0.175, params 0.0037) |
| `clip_foreach_true` | −3.37 % | 2.68 % | yes (negative) | `BIT_EQUIVALENT` — code-identical control |
| `clip_foreach_false` | −2.24 % | 2.68 % | no | changes the trajectory (clip norm mean 8.44 → 11.07) |

What the audit established:

* **The default is already the fast path.** `_default_to_fused_or_foreach` on the actual 528 fp32
  parameter tensors resolves to `fused=False, foreach=True`, and `clip_grad_norm_` already uses
  foreach, so `foreach=True` is a no-op on both.
* **`fused=True` does not help.** It is inside the noise band, it changes the numerics out of
  tolerance, and it is therefore rejected on both counts.
* **Learning rates, betas, weight decay, clipping threshold and parameter groups were not changed** —
  all three optimizer variants come from the same config and the same
  `trainable_parameter_groups` call; only the reduction implementation differs.

Decision: **no optimizer or clipping change is adopted.**

## 7. SDPA backend audit (task section 9)

The workload is **not** stuck on a global math fallback, and it is not using flash or cuDNN:

| Evidence | Result |
|---|---|
| Attention calls per profiled step | 472 |
| … memory-efficient CUTLASS FMHA | **248** (`fmha_cutlassF_bf16_aligned_64x64_rf_sm80`, `aten::_efficient_attention_forward`) |
| … math | **224** (`aten::_scaled_dot_product_attention_math`, `softmax_warp_forward`) |
| … flash | 0 |
| … cuDNN | 0 |
| Forcing `FLASH_ATTENTION` on the real step | fails: `No available kernel` ("Torch was not compiled with flash attention") |
| Forcing `EFFICIENT_ATTENTION` on the real step | fails: `No available kernel` (224 call sites cannot use it, so a global force has no candidate) |
| Forcing `MATH` on the real step | runs, but **numerically different** (first-step total 9.4481 vs 9.4255) and ~9.7 % slower (unbracketed) |
| One-step identity check | `default_is_math_backend: false` |

So torch's own selection is already a **mixture**: the efficient CUTLASS kernel where the shapes allow
it, math elsewhere. FlashAttention is simply not compiled into this PyTorch build and no extension was
installed (section 9 forbids that). Pinning any single backend either fails or makes things worse, so
the audit's answer is: **leave backend selection to torch**; there is no accidental fallback to fix and
no free win.

## 8. Checkpointing interaction (task section 10)

Task 6C.5 measured checkpointing OFF as 8.58 % slower in a single non-interleaved sweep and the spec
asked for an interleaved repeat on the best compile candidate. **No compile candidate exists**, so the
comparison was run on the integrated baseline path instead and is recorded as such.

| | samples/s per round | mean | spread |
|---|---|---|---|
| Checkpointing ON | 2.7376 / 2.5844 | **2.6610** | 5.76 % |
| Checkpointing OFF | 2.6241 / 2.5898 | **2.6070** | 1.32 % |

OFF is **−2.03 %** with the two rounds disagreeing in sign (+4.3 %, −0.2 %), which is inside the
round-to-round spread → **not reproducible, so checkpointing stays ON.** Task 6C.5's conclusion
survives, but its magnitude does not: "-8.58 %" was a single-sweep artefact of this machine's drift,
and the honest statement is "turning checkpointing off does not help here".

## 9. Resource accounting (task sections 14 and 16)

| | Adopted runtime (B0.6) | Peak over all experiments |
|---|---|---|
| Reserved VRAM | **8.582 GiB** | 14.406 GiB (`C1c_combined_aot_eager`, rejected) |
| Allocated VRAM | 7.658 GiB | 9.995 GiB |
| Process RSS | **3.295 GiB** (limit 24 GiB) | 8.807 GiB (profiler process, holds Kineto events) |
| System RAM used | — | 17.664 GiB of 31.38 GiB |
| GPU temperature | up to 82.1 °C | |
| GPU power | up to 114.7 W | |
| Paging / OOM | none | none |

Two findings worth keeping: every `torch.compile` variant inflates reserved VRAM by ~4 GiB
(8.6 → 12.5–12.9 GiB), and the combined `aot_eager` candidate exceeds the 14 GiB budget outright —
another independent reason not to adopt it. The byte-level accounting distinguishes the adopted
runtime from rejected experiments, because conflating them would misstate the budget.

## 10. Tests

`python -m pytest tests/ -q` → see `handoff/FROM_DSH.md` §15 for the recorded result.

`tests/test_task6c6_launch.py` covers the 12 section-17 items: the formal loop consumes
`collect_grad_norms` (functional, against a stub runtime), `False` removes the sweep while
`clip_grad_norm_` still runs, `True` stays available for Stage-1 diagnostics, the integration is
bit-equivalent over a fixed mini-run, the compile wrapper is opt-in and can be absent, compile failures
are recorded with an explicit status and no throughput number, no architecture/loss/data/optimizer
semantics changed, strict determinism is effective for every measured variant, the benchmark set is
train-split only, batch size is 1, no `[REF]`/4B/SRE/SCL appears, and the profiler only runs from its own
opt-in script.

## 11. Reproduce

```bash
python scripts/task6c6_benchmark.py --group baseline
python scripts/task6c6_profile.py                       # profiler + SDPA audit (+ control window)
python scripts/task6c6_benchmark.py --group compile
python scripts/task6c6_benchmark.py --group optimizer
python scripts/task6c6_equivalence.py --include-compile
python scripts/task6c6_benchmark.py --group checkpointing
python scripts/task6c6_benchmark.py --group final --winner-collect-grad-norms true
python scripts/task6c6_summarize.py                     # decisions + resource accounting
```

`PYTHONUTF8=1` is required to *see* the underlying `torch.compile` error on this machine: without it
the Windows GBK locale turns the message into a `UnicodeDecodeError` while decoding it, which hides the
real cause (`TritonMissing`). That locale behaviour is itself worth knowing for any future Windows
profiling session.

## 12. What this task did not do

No batch size > 1 and no gradient accumulation: the step is still one sample per optimizer step, which
is exactly why the remaining lever is dispatch count rather than arithmetic intensity. No model, loss,
data, sample-order, scheduler or optimizer-mathematics change. No `[REF]`, no 4B, no Spatial Relation
Encoder, no Spatial Consistency Loss, no full training. No Task 6C reinterpretation: the four arms still
show no arm solving instruction-conditioned segmentation and the paired probe is still 0/20.
