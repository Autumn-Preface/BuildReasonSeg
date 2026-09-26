# PROJECT_STATE — BuildReasonSeg

_Last updated by DSH at the end of Task 6C.6._

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

The block above is machine-checked against
`evaluation/build_spatial_reason_artifact_index.json` by
`scripts/check_artifact_consistency.py`. Do not hand-edit numbers anywhere else.

## Identity

| Field | Value |
|---|---|
| Project codename | **BuildReasonSeg** |
| Repository | `Autumn-Preface/BuildReasonSeg` (branch `main`) |
| Legacy evidence (read-only) | `../WHU_Building_Segment/` |
| Reasoning dataset | **BuildSpatialReason v0.1.1** (frozen, audited, PASS) |
| Legacy dataset version | **v0.1** (frozen, superseded — never use for training) |
| MVP environment | **`.conda/buildreasonseg-mvp`** (conda `--prefix`, Python 3.11.16, PyTorch 2.13.0+cu132) |
| MVP stack (measured) | **Qwen3-VL-2B-Instruct + SAM2.1 Hiera Base+ + `[SEG]`** (ADR-013) |
| Design stack (unmeasured) | Qwen3-VL-4B-Instruct + SAM 2.1 hiera-large (ADR-012) |
| Network posture | offline (`HF_HUB_OFFLINE=1`); Watt Toolkit stopped during training; no hosts edit, no cert-store edit, no insecure TLS flag |
| Reproducibility | **strict deterministic mode, cross-process bit reproducible** (Task 6C) |

## Completed tasks

| Task | Scope | Result |
|---|---|---|
| 1 | Project foundation + baseline freeze | done |
| 2 | Polygon → building component representation | done (4,038 maps, 36,926 components) |
| 3A | Geometry statistics + relation engine + thresholds | done |
| 3B | Relation semantics correction + freeze | done |
| Naming | `SpatialReasoningSeg` → `BuildReasonSeg` | done |
| 4 | BuildSpatialReason-v0.1 dataset generator | done (32,284 records) |
| 5 | v0.1 validator + semantic quality audit | done → `FAIL_REQUIRES_REVISION` |
| 5B | v0.1.1 corrective regeneration + acceptance audit | done → `PASS` |
| 5C | Acceptance hardening + artifact consistency | done → `PASS` |
| 5.5 | External research, model-stack verification, MVP design freeze | done → ADR-012 |
| 6A | Native-Windows env bootstrap + 2B `[SEG]` MVP smoke/overfit | done → `PASS` |
| 6B | Network cleanup + 2B real mini-train + first generalization audit | done → `FAIL_REQUIRES_DEBUG` |
| Watt | Standalone Watt Toolkit lifecycle test (3 rounds) | done → `FULL_AUTO_OK` |
| 6C | **Paired counterfactual training × neutral SAM prompt 2×2 ablation** | **done → `EXPERIMENT_COMPLETE_PARTIAL_IMPROVEMENT`** |
| 6C.5 | **Batch-1 training-pipeline throughput audit + value-preserving optimization** | **done → `OPTIMIZATION_PARTIAL`** |
| 6C.6 | **Launch-overhead candidates + formal-path integration of the 6C.5 winner** | **done → `OPTIMIZATION_PARTIAL`** |

## Task 6C measured results

Valid, bit-reproducible 2×2 on the frozen ADR-013 stack. Same recipe, same seed, same validation material;
only the training subset (U/P) and the SAM bridge (C/L) differ. Full detail:
`docs/task6c_prompt_ablation.md`, ADR-015, `evaluation/task6c_comparison.json`.

| Arm | strict e2e mIoU | Dice | emission | paired | mean margin | projected effective rank | top-1 variance |
|---|---|---|---|---|---|---|---|
| `U_C` (Task 6B baseline) | 0.10516 | 0.17360 | 120/120 | **0/20** | −0.000001 | 1.516 | 0.9023 |
| `U_L` (language bridge) | 0.10872 | 0.17846 | 120/120 | **0/20** | −0.003530 | 1.491 | 0.9087 |
| `P_C` (paired data) | 0.10604 | 0.17851 | 120/120 | **0/20** | +0.000007 | **4.100** | **0.5892** |
| `P_L` (both) | 0.09284 | 0.15626 | 120/120 | **0/20** | −0.000003 | **3.371** | **0.6430** |

## Task 6C.5 measured results

Performance-only task; it changed no model, loss, optimizer, scheduler, data or sample order, and it does not
reinterpret Task 6C. Full detail: `docs/task6c5_training_optimization.md`,
`evaluation/task6c5_variants.json`, `evaluation/task6c5_final_benchmark.json`.

| | B0 (current) | Adopted (`skip_grad_norm_instrumentation`) |
|---|---|---|
| Throughput (interleaved, 2 rounds, 8 warmup + 64 steps, batch 1) | 2.152 samples/s | **2.373 samples/s → +10.24 %** |
| Repeat spread | 4.96 % | 0.47 % |
| GPU utilization | 37.3 % | 37.5 % (unchanged) |
| CPU utilization | 84.4 % | 92.5 % |
| Single clean sweep | 2.506 samples/s | 2.847 samples/s (+13.63 %) |
| Peak RSS / reserved VRAM | 3.30 GiB / 8.64 GiB | 3.29 GiB / 8.64 GiB |

1. **The CPU≈100 % / GPU≈40 % cause is per-operator launch overhead inside Qwen forward and backward, not
   data preparation.** Synchronized profiling puts `qwen_forward` 50.1 %, `backward` 38.5 %, `optimizer_step`
   7.7 % against **all host-side preparation ≈2.4 %** (processor 0.76 %, image I/O 0.62 %, SAM feature
   lookup 0.52 %, target-mask I/O 0.24 %, H2D 0.21 %). The processor costs 2.21 ms/sample against a ≈400 ms
   step.
2. **Every data-side candidate was measured and rejected**: source cache −5.80 %, Qwen preprocessing cache
   −2.92 % (it halves CPU utilization 90 → 56 % and still loses), pinned + non-blocking −12.98 %, prefetch
   with 2/4/8 threads −5.46/−5.69/−9.31 %. The preprocessing cache is the clean proof that the hypothesis was
   wrong: CPU utilization is not the binding constraint.
3. **The adopted change is one switch**: drop the per-step 528-tensor gradient-norm sweep that the Task 6C
   loop computes and discards. Default stays `collect_grad_norms=True`, so Task 6A/6B and their tests are
   unaffected. Bit-equivalent gate: prepared tensors (0 mismatches / 16 samples), 12-step losses, gradient
   fingerprints and post-step parameter fingerprints all identical, baseline control reproducible. A
   separate refactor control proves the audit's own restructuring of `train_step` (optional stage timers,
   split `torch.autocast` regions) is value-neutral against the pre-6C.5 step body.
4. **Rejected caches were still value-preserving.** `source_cache + preprocessed_cache + skip` is also
   bit-equivalent (second gate artifact) and 2.29 % faster than the adopted variant — inside the benchmark's
   own 4.96 % spread — so three extra switches were not taken for an unresolvable difference (section 20's
   ~5 % complexity rule, recorded as `adoption_rule`).
5. **Gradient checkpointing stays ON** (−8.58 % when off) and **strict determinism stays ON** (the
   determinism tax is only +1.78 %, well below the gate).
6. **Remaining bottleneck is batch-1 launch serialization**; data-side work is closed out. Raising arithmetic
   intensity per launch (batch > 1) is the only evidence-backed lever and is explicitly a different
   experiment, not a Task 6C.5 optimization.
7. **Measurement defect found and fixed**: the LoRA adapters' `dropout = 0.05` consumes the global RNG, so an
   equivalence gate must **re-seed per run** — with a single start-up seed the baseline did not reproduce
   itself. Any future equivalence claim needs the same per-run re-seeding rule.
8. **Network / Watt for 6C.5**: the accelerator was **pre-existing** (`watt_preexisting = true`) and was only
   used to push; nothing was closed, force-killed, or reconfigured, no hosts file was edited and no TLS
   verification was disabled. The MVP stack stayed offline (`HF_HUB_OFFLINE=1`) throughout.

## Task 6C.6 measured results

Performance-only task. The one runtime change it adopts is wiring Task 6C.5's winner into the formal
training loop; every other candidate was measured and rejected. Full detail:
`docs/task6c6_launch_optimization.md`, `evaluation/task6c6_*.json`.

| | Value |
|---|---|
| Formal-path integration | `training.collect_grad_norms: false` in `configs/mvp/task6c_2b_ablation.yaml`, passed explicitly by `scripts/task6c_train.py` |
| Integration equivalence | **`BIT_EQUIVALENT`** (prepared tensors, 12-step losses, gradients, post-step parameters) |
| Integrated baseline B0.6 | **2.505 samples/s** (final head-to-head mean; 2.545 / 2.465) |
| Pre-integration control | 2.184 samples/s (2.188 / 2.179; spread 0.41 %) |
| Integration gain (interleaved) | **+14.7 %** (+21.5 % in the 4-run baseline group) |
| New candidates adopted | **none** |
| Adopted runtime footprint | 8.582 GiB reserved VRAM, 3.295 GiB RSS, no OOM |

1. **The step is dispatch/launch dominated, now measured rather than inferred**: **57,341 CUDA kernels
   per step** at a **2.1 µs median**, **2,545 host↔device synchronizations per step**, the CUDA launch
   path consuming **31.9 % of CPU self time**, and the GPU only ~33 % utilized in unprofiled runs. Host
   data preparation is ≈2.4 % of the step (Task 6C.5) and is not a factor. The gradient-norm sweep that
   Task 6C.5 removed accounted for 3,589 kernels and 2,048 synchronizations per step.
2. **`torch.compile` cannot run on this install.** The inductor backend fails with `TritonMissing`
   (no Triton package, no MSVC). Everything that does run is slower: `cudagraphs` −24.7 %, combined
   `aot_eager` −21.2 %, decoder tail −11.3 %, Qwen `aot_eager` −10.4 %, Qwen `eager` −9.2 %. All compile
   candidates are also `NOT_EQUIVALENT` (max loss difference 0.042–0.234 against a 1e-3 tolerance), all
   inflate reserved VRAM by ~4 GiB, and `combined_aot_eager` exceeds the 14 GiB budget at 14.41 GiB.
   `backend="eager"` and `cudagraphs` additionally fail on a **sequence-length change** (291/321/295/318
   tokens) with a tensor-size mismatch — CUDA graphs are unsafe here without fixed-length padding, which
   is out of scope.
3. **Optimizer/clipping is already optimal.** `_default_to_fused_or_foreach` resolves to
   `foreach=True` for the 528 fp32 parameters, so `foreach=True` is a no-op; `fused=True` is inside the
   noise band and `NOT_EQUIVALENT`; `foreach=False` is slower. Two *code-identical* controls
   (`adamw_foreach`, `clip_foreach_true`) measured −2.85 % and −3.37 %, which fixes the **noise floor of
   a sequential group at 3.37 %** on this machine.
4. **SDPA is already mixed and correct.** Per step, 472 attention calls split into **248
   memory-efficient CUTLASS FMHA** and **224 math**, with 0 flash and 0 cuDNN. FlashAttention is not
   compiled into this PyTorch build; forcing flash or mem-efficient fails with `No available kernel`, and
   forcing math changes the numerics (9.4481 vs 9.4255) and is ~9.7 % slower. No accidental fallback to
   fix, no win available.
5. **Gradient checkpointing stays ON.** Interleaved ON/OFF gave OFF −2.03 % with the two rounds
   disagreeing in sign (spread ON 5.76 %, OFF 1.32 %) → Task 6C.5's −8.58 % single-sweep figure does not
   survive interleaving.
6. **This laptop's absolute throughput is not a stable quantity** (up to 19 % between identical runs;
   the first run in a process is systematically fastest). The reproducible quantity is the interleaved
   *ratio*, which is why every comparison brackets its candidates with reference runs and compares
   against the interpolated reference, and why a non-resolvable gain is never adopted.
7. **Methodology carried forward**: A/B/A/B interleaving for headline numbers, bracketed+interpolated
   references for screening, per-run re-seeding in every equivalence gate, and `PYTHONUTF8=1` when
   reading `torch.compile` errors on this Windows locale (otherwise the real `TritonMissing` cause is
   hidden behind a `UnicodeDecodeError`).

## What Task 6C changed in the project's understanding

1. **Neither of Task 6B's two confounds is the cause.** All four arms are 0/20 on the paired unseen probe
   with mean margins within ±0.0036 of zero. Paired counterfactual training, the removal of the fixed
   centre point, and their combination all fail to create instruction-conditional mask selection.
2. **But the factors are not inert.** Paired training materially changes the prompt representation
   (effective rank 1.50 → 3.74, top-1 variance 0.905 → 0.616) without changing the masks. The projection
   uses more dimensions; two instructions on one image still produce the same mask.
3. **The fixed positive centre point is not the culprit.** Its norm is 11.39 against a projected language
   norm of 266.8 — a **23.4×** ratio in the language vector's favour — and removing it (the `L` bridge)
   changes little. "The centre point overrides the language" is not supported by the measurement.
4. **The collapse is directional and low-rank, not constant.** Same-image projected cosine > 0.9999 with
   top-1 variance 0.59–0.91 and a non-zero effective rank. Task 6B's looser wording is corrected.
5. **The remaining problem is after the prompt**, i.e. in how the `[SEG]` hidden state is formed and how
   SAM's decoder turns a prompt direction into a region. Prompt normalisation, a multi-token prompt,
   auxiliary point supervision or `[REF]` are justified now, and were not before.
6. **Determinism is real now.** `training.deterministic` is consumed, strict deterministic algorithms are
   active with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and two independent processes produce identical losses,
   gradients and post-step parameters. The `U_C` arm was run before and after the cache fix and is
   bit-identical, which also proves the fix value-preserving.

## Correctness fixes carried forward (do not regress)

1. Determinism (above).
2. Per-arm checkpoint directories come from `cfg["paths"]["checkpoints"]` — no module-level constant.
3. The operation-chain lookup is built from the **train split only**; validation text never extends it.
4. The SAM2 CPU feature cache holds all 480 training images inside the budget by sharing the two
   image-independent tensors once (7.508 GiB instead of 11.25 GiB), and it never mutates stored entries.

## Measured limitations (carry into Task 6D)

1. **Instruction conditioning of the mask is still absent** (0/20 in all four arms). This remains the
   blocking defect for the MVP.
2. **Mask quality is not usable**: strict end-to-end mIoU 0.093–0.109, L3 nontrivial 0.076–0.088.
3. **Language metrics are template metrics**: `reasoning_zh` has 21 distinct values in the whole training
   mini-set, so exact match and operation-chain accuracy cannot support a reasoning claim.
4. **More prompt diversity did not become mask diversity**: effective rank 3.4–4.1 still yields
   IoU(pred_A, pred_B) 0.999.
5. **The best mIoU arm is not the most diverse arm** (`U_L` 0.1087 vs `P_L` 0.0928), so mIoU alone remains
   a misleading selection signal.
6. **Training throughput is launch-bound at batch 1** and the removable overhead is now exhausted:
   2.505 samples/s after Task 6C.6 (integrated B0.6), GPU utilization ~40 %, 57k tiny kernels per step.
   Caching, pinning, prefetch, `torch.compile`, optimizer/clipping implementations, SDPA backends and
   checkpointing-off have all been measured and rejected. The remaining lever is arithmetic intensity
   (true batching ≥2), which is a new experiment, not a perf tweak.

## Current blockers

**None for the next task to start.** The failure is characterised, the two candidate causes are excluded by
a valid controlled experiment, the throughput question is audited and closed at batch 1, and the remaining
search space is narrow and explicit.

## Recommended next task

**ChatGPT review of the pushed Task 6C.6 results**, then either

* **Task 6D aimed after the prompt** (if model quality is the priority):

  1. make the `[SEG]` hidden state instruction-conditional (auxiliary point-inside-the-mask supervision from
     ground-truth geometry used *as supervision only*, or a second explicitly negative prompt slot);
  2. only then test prompt normalisation / whitening and a multi-token prompt, which Task 6C has now earned
     the right to try;
  3. keep the paired instruction-dependence probe as the primary gate and keep the four correctness fixes;
  4. do not scale to 4B and do not add `[REF]` on the strength of anything measured so far;

  or

* **Task 6C.7 true-batching feasibility** (only if throughput is blocking) — VRAM headroom at batch 2/4 with
  checkpointing ON, an explicit equivalence story for a changed effective batch, and its own adoption gate.
  It is not authorised by Task 6C.6.

Do not re-open pipeline caching, `torch.compile`, the optimizer/clipping implementations, SDPA backend
pinning or checkpointing-off: all six are measured, gated and closed for this machine.

Full detail: `handoff/FROM_DSH.md`, `docs/task6c_prompt_ablation.md`,
`docs/task6c5_training_optimization.md`, `docs/task6c6_launch_optimization.md`,
`evaluation/task6c_comparison.json`, `evaluation/task6c6_final_benchmark.json`,
`docs/architecture_decisions.md` (ADR-014 amendment, ADR-015).
