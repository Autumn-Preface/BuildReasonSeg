# PROJECT_STATE — BuildReasonSeg

_Last updated by DSH at the end of Task 6D._

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
| 6C.7 | **Frozen Qwen visual-feature cache + remaining batch-1 sync audit** | **done → `OPTIMIZATION_PARTIAL`** |
| 6D | **Spatial Grounding Bridge v0.1 (oracle diagnostic + geometry head)** | **done → `GROUNDING_REPRESENTATION_FAILED`** |

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

## Task 6D measured results

Architecture task: the first explicit spatial-grounding mechanism, and the first measurement that
localizes the Task 6C failure to a specific component. Full detail:
`docs/task6d_spatial_grounding_bridge.md`, `evaluation/task6d_*.json`.

| | Value |
|---|---|
| Oracle point (GT point prompt → SAM2) | mIoU **0.4876**, Dice 0.6055, paired **18/20** |
| Oracle box (GT box prompt → SAM2) | mIoU **0.7506**, Dice 0.8507, paired **20/20**, IoU(pred_A, pred_B) **0.0000** |
| Section 5 geometry choice | **BOX** (point missed the 0.50 mIoU gate; box passed it and led by +0.2630) |
| G0 (2 epochs × 480 paired records) | emission **120/120** ✅, geometry paired **0/20** ❌, box IoU **0.0082** ❌ |
| G1 | **not run** (section 9: stop on G0 failure, do not compensate) |
| Free-generation strict e2e mIoU | **0.0066** (Task 6C `P_C` reference 0.10604) |
| Paired mask probe | **0/20**, own ≈ cross ≈ 4e-10, IoU(pred_A, pred_B) **0.919** |

1. **The segmenter was never the problem.** Given the correct geometry, SAM2 recovers the target at
   **0.7506 mIoU** (box) against Task 6C's **0.10604**, with a paired probe of 20/20 and *zero* overlap
   between the two targets' predicted masks. The Task 6C failure was entirely in prompt generation.
2. **The `[SEG]` hidden state does not carry the target's location.** A 1.05M-parameter head supervised on
   480 GT boxes did not fit even the training boxes; by epoch 2 it collapsed to the single constant box
   `(0.457, 0.455, 0.547, 0.539)` with per-coordinate spread `[0.000, 0.000, 0.001, 0.000]` — the optimal
   constant predictor under SmoothL1 for an uninformative read-out.
3. **The read-out is template-dominated, not merely instruction-independent.** Four-condition control on
   `[SEG]` hidden cosines: different image + **same template** 0.99988 (most similar) vs same image +
   different template 0.99898 vs different image + different template 0.99869 (least similar). Changing
   the whole image moves the hidden state *less* than changing the instruction wording.
4. **The language objective cannot fix it.** LM CE saturates at 0.0000 because the assistant target is one
   of only 21 distinct reasoning templates (Task 6C's finding), so it is memorized long before it could
   pressure the read-out to encode *which* building.
5. **This reproduces Task 6C with a different read-out.** Task 6C measured the projected prompt at
   effective rank 1.5 with same-image cosine > 0.9999; Task 6D measures a *supervised geometry head* on
   the raw `[SEG]` hidden and finds the same collapse. Two independent read-outs agreeing makes the
   diagnosis strong.
6. **Dataset is not the binding constraint (yet).** The 116/120 failures carry `tiny_target` (93 %) and
   `border_truncation` (30 %) flags, but the artifact's degeneracy guard shows all of them fail from one
   constant mask regardless of content, so a dataset-selection task is not indicated by this evidence.

## Task 6C.7 measured results

Performance-only task; two changes adopted, and the model/loss/data/optimizer semantics are untouched.
Full detail: `docs/task6c7_visual_cache_optimization.md`, `evaluation/task6c7_*.json`.

| | Value |
|---|---|
| Section 3 Phase-B duplicate H2D | **removed** (exact; also fixed a `del moved` `UnboundLocalError` the refactor introduced) |
| Frozen Qwen visual-feature cache | **adopted** — `BIT_EQUIVALENT`, **+9.94 %** warm paired throughput |
| Cache footprint | 4.0 MiB/image (`pooler_output` 1 MiB + 3 × `deepstack_features` 1 MiB, bf16), bound 512 images = **2.0 GiB** |
| Kernel / sync change | **−8.44 %** CUDA kernels, **−43.47 %** host↔device sync ops per step |
| Adopted runtime footprint | 8.582 GiB reserved VRAM, 4.751 GiB RSS + 128 MiB cache, no OOM |
| Config | `training.visual_feature_cache: true`, `visual_feature_cache_max_images: 512` |

1. **The visual tower is provably cacheable.** 315 visual parameters, all bf16 and all frozen; zero LoRA
   modules in the tower (`lora.text_only: true`); no dropout, so it consumes no RNG; recomputation is
   bit-identical (max abs difference 0.0); nothing in the tower moves during a real optimizer step
   (parameters *and* buffers checked); and 8 same-image instruction pairs produce identical
   `pixel_values` **and** identical visual features — the tower is instruction-independent.
2. **The cached boundary is the narrowest available**: `Qwen3VLModel.get_image_features(...)` output,
   i.e. `pooler_output` + `deepstack_features`, before they are mixed with trainable text hidden states.
   The wrapper is an instance-level replacement on the project's own Qwen module with the original bound
   method preserved, so the miss path is byte-identical. Nothing downstream of a trainable parameter is
   cached, and `last_hidden_state` is deliberately not cached (the language model never reads it).
3. **Equivalence is bit-exact** over 16 samples' features and a 12-step gate with per-run re-seeding, for
   both key strategies (source-image identity and pixel-content hash). First-step loss identical.
4. **The gain is real but only visible under a drift-robust design.** Whole-variant runs drift by −24.6 %
   and the four-run interleaved V1/V2 comparison produced rounds disagreeing in sign (+15.7 % V1, then
   +8.9 % V2). A paired ablation — alternating 8-step cache-off/cache-on blocks inside one runtime — gives
   **+10.41 % (all pairs) / +9.94 % (warm), 4/4 pairs in favour**. Warming the cache while it is disabled
   stores nothing, and that mistake alone had reported +2.31 % and would have rejected the change.
5. **The remaining ~1,051 scalar read-backs are PyTorch's, not ours.** Scoped windows plus a
   `sys.setprofile` `c_call` tracer attribute 99.5 % of them to `optimizer.step()` →
   `torch/optim/adam.py:770-776` converting each parameter's **CPU-hosted** `step` counter twice per step
   (`_get_value` → `.item()`); the upstream comment states the CPU hosting is deliberate. The optimizer
   window issues 1,024 reads with **one** sync op, so they are cheap; backward contributes 5 reads;
   disabling gradient checkpointing changes nothing. `fused=True` would remove them, but Task 6C.6
   measured it at −0.71 % and it is `NOT_EQUIVALENT`, and section 4 forbids patching third-party
   internals. Project-owned reads are ~7/step (<1 %), so V3 (deferred scalar logging) was not implemented.
6. **Batch-1 optimization is now exhausted on evidence**, which is what makes the batching recommendation
   earned: the redundant H2D is gone, the sync traffic is localized and dismissed, and the last large
   removable win is adopted. What remains is execution shape (~52,500 kernels/step, ~2 µs median, GPU
   ~40 %).

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

1. **Instruction conditioning of the mask is still absent**, and Task 6D localizes it: the `[SEG]` hidden
   state is template-dominated (different-image/same-template cosine 0.99988 > same-image/different-template
   0.99898) and carries no usable target location. This remains the blocking defect for the MVP.
2. **Mask quality is not usable**: strict end-to-end mIoU 0.093–0.109, L3 nontrivial 0.076–0.088.
3. **Language metrics are template metrics**: `reasoning_zh` has 21 distinct values in the whole training
   mini-set, so exact match and operation-chain accuracy cannot support a reasoning claim.
4. **More prompt diversity did not become mask diversity**: effective rank 3.4–4.1 still yields
   IoU(pred_A, pred_B) 0.999.
5. **The best mIoU arm is not the most diverse arm** (`U_L` 0.1087 vs `P_L` 0.0928), so mIoU alone remains
   a misleading selection signal.
6. **Training throughput is launch-bound at batch 1** and the removable overhead is now exhausted:
   the adopted runtime is Task 6C.6's integration plus Task 6C.7's frozen visual-feature cache, worth
   +9.94 % on top (paired ablation), with 8.4 % fewer kernels and 43 % fewer sync ops. Caching (SAM),
   pinning, prefetch, `torch.compile`, optimizer/clipping implementations, SDPA backends and
   checkpointing-off are all measured and closed; the remaining lever is arithmetic intensity (true
   batching ≥2), which is a new experiment, not a perf tweak.

## Current blockers

**None for the next task to start.** The failure is characterised, the two candidate causes are excluded by
a valid controlled experiment, the throughput question is audited and closed at batch 1, and the remaining
search space is narrow and explicit.

## Recommended next task

**ChatGPT review of the pushed Task 6D results**, then an architecture task aimed at the read-out, since
Task 6D localized the defect precisely:

1. **Make the target position a supervised *token*, not a free hidden vector.** The `[SEG]` hidden is a
   consequence of the reasoning template and the language loss saturates at 0.0000 before it can impose
   spatial content. Emitting geometry as text tokens (quantized coordinates / grid tokens appended to the
   reasoning) makes the geometry a first-class target of the existing LM head, so the loss differs *per
   sample* instead of per template. No new module, no 4B, no `[REF]`.
2. **Or supervise the read-out directly** (a learned spatial code on the `[SEG]` hidden trained with the
   same oracle geometry) so that a constant output cannot be optimal — Task 6D's constant-box optimum is
   the evidence that this is needed.
3. **Keep the oracle diagnostic as the standing ceiling check** (~9 min) for every future hypothesis: it
   separates "the segmenter cannot" from "the model does not say where".
4. **Do not re-run G1 and do not re-open the geometry choice** (box is chosen); do not add `[REF]`, a
   Spatial Relation Encoder, a Spatial Consistency Loss, 4B, a new dataset or full training.
5. **Dataset work is not indicated yet** — the failure population is model-caused.

Full detail: `handoff/FROM_DSH.md`, `docs/task6d_spatial_grounding_bridge.md`,
`evaluation/task6d_oracle_prompt_diagnostic.json`, `evaluation/task6d_representation.json`,
`evaluation/task6d_paired_probe.json`, `docs/task6c_prompt_ablation.md`,
`docs/task6c7_visual_cache_optimization.md`, `docs/architecture_decisions.md` (ADR-014 amendment, ADR-015).
