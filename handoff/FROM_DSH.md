# FROM_DSH — Task 6B Report: Network Cleanup + 2B `[SEG]` Real Mini-Train & First Generalization Audit

**Date:** 2026-09-26
**Actor:** DSH
**Task:** Task 6B (from `handoff/TO_DSH.md`)
**Verdict: `FAIL_REQUIRES_DEBUG`**

> Reported against the Task 6B section 17 minimums. Nine of ten are met; **paired validation
> instruction-dependence is 0/20 against a ≥14/20 bar**, and section 17 names "pair generalization fails
> badly" as a debug-required condition. Measured: 120/120 valid `[SEG]` emission, strict end-to-end val
> mIoU **0.1102**, operation-chain accuracy **1.000**, paired **0/20**.

> The Task 6A report is preserved verbatim in `handoff/ARCHIVE_task6a_report.md`. This file is the Task 6B
> report.

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

**`FAIL_REQUIRES_DEBUG`.**

The Task 6B mini-train solved the language half of the problem and failed the segmentation half. On 120
unseen validation records the model emits exactly one `[SEG]` **120/120** times, reproduces the expected
`reasoning_zh` exactly **120/120** times and gets the operation chain right **120/120** times — while
strict end-to-end mIoU is **0.1102** and the paired instruction-dependence probe scores **0/20**.

The failure is not vague. Two different instructions on the same image produce masks whose IoU is
**0.9999**, because the 256-d sparse prompt the projection emits is a near-constant vector (cosine
**0.999995** between two instructions on one image, **1.000000** between two different images). The mask
decoder is being asked to segment the same thing every time. The single permitted bounded recipe
adjustment was tried and made the result slightly worse (0.1102 → 0.0950), so the cause is not loss
weighting or the mask-pathway learning rate.

## 2. Network / Watt Cleanup

Full record: `evaluation/task6b_network_cleanup.json`.

* hosts entries for `github.com` / `raw.githubusercontent.com` / `huggingface.co` / `github.io`: **none**;
  `workaround_present: false`; 0 blackhole lines; 0 accelerator marker lines.
* listeners on 443 and on the accelerator's usual proxy ports: **none**.
* certifi restored to the standard 121-certificate bundle, **no accelerator root present**
  (sha256 `9cc2a774b5198dcff14d9be1e66091f538975d867ce029a96bce15a55dfd730f`).
* the automatic trust mutation is **removed** from `buildreasonseg_mvp/local_env.py`; the legacy merge
  helper is opt-in and a no-op without `confirm=True`. Importing the project no longer touches certifi.
* no insecure TLS flag is passed anywhere (`insecure_flags_used: false`).
* read-only with respect to system configuration: no hosts edit, no certificate-store edit, no
  registry/driver/`PATH`/proxy change.

**The Watt-specific part of the problem is gone. The underlying connectivity is not normal.** Details in
§4.

## 3. Conda Environment Integrity

`.conda/buildreasonseg-mvp`, created with `conda create --prefix` inside the project directory as
required; Python 3.11.16, torch 2.13.0+cu132, transformers 5.17.0, peft 0.21.0, scipy 1.17.1, SAM2 1.0 at
revision `2b90b9f5ceec907a1c18123530e92e794ad901a4`. `.conda/` is in `.gitignore`. No `venv` was created;
`base` and `yolo_sam_env` were not modified.

## 4. Direct Hugging Face / GitHub Validation

| Probe | Result |
|---|---|
| `https://api.github.com/...` (urllib + httpx) | **200**, standard TLS |
| `https://raw.githubusercontent.com/...` (urllib) | **200** |
| `https://pypi.org/simple/` (urllib + httpx) | **200** |
| `https://huggingface.co/api/models/Qwen/Qwen3-VL-2B-Instruct` | **DNS failure**, `[Errno 11001] getaddrinfo failed` |
| offline load of processor + SAM2 from `local_cache/` | **succeeds in ≈30 s** |

**`huggingface.co` does not resolve on this network at all** (system resolver `202.195.70.70`; public
resolvers return unrelated addresses). Task 6B therefore runs with `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` against the cached Qwen3-VL-2B snapshot
`89644892e4d85e24eaac8bacfd4f463576704203` and the cached SAM2.1 Base+ weights. Every Task 6B question can
be answered offline; downloading new weights cannot.

**git smart-HTTP to GitHub is intermittent.** `github.com` resolves to `20.205.243.166`, which black-holes
TCP 443 (~21 s timeout) while other GitHub edge addresses answer in under 300 ms
(`140.82.113.4`, `20.205.243.168`, `185.199.108.133`). Five consecutive `git ls-remote` attempts failed,
then the same command succeeded minutes later with no change to the machine. This is a property of the
current connection, not a leftover of the accelerator — the hosts file is clean, nothing is listening and
no certificate is being injected. The recorded, non-persistent fallback is

```
git -c http.curloptResolve=github.com:443:140.82.113.4 <command>
```

which only chooses the address the connection is opened to; TLS still verifies the real hostname against
the standard bundle. Nothing was written to the hosts file, the git config, the certificate store, the
registry, `PATH`, drivers or the system proxy.

The `X-Repo-Commit` / `LocalEntryNotFoundError` failure Task 6A saw from the accelerator's TLS proxy is
**not reproducible**: with `huggingface.co` unresolvable, no Hub metadata request is attempted at all.

## 5. Frozen Architecture

Unchanged from ADR-013 and verified per phase:

| Component | State |
|---|---|
| Qwen3-VL-2B-Instruct, BF16 | base LLM frozen; **vision tower frozen in both phases** (`qwen_visual_trainable: []`) |
| text LoRA r=16 α=32 | trainable, 196 modules, 17,432,576 parameters |
| `[SEG]` row (id 151669) | trainable, one tied row shared by input embedding and `lm_head`, 2,048 parameters |
| Projection MLP | frozen in Phase A (`projection_trainable: false`), trainable in Phase B |
| SAM2 image encoder / prompt encoder / memory | frozen |
| SAM2 mask decoder | frozen in Phase A (`sam_mask_decoder_trainable: false`), trainable in Phase B |
| Phase A trainable tensors | 393 of 1,636 |
| Phase B trainable tensors | 528 of 1,636 |
| total trainable | **24,010,309 / 2,227,632,642 = 1.078 %** |

No `[REF]`. No 4B. No Spatial Relation Encoder. No Spatial Consistency Loss. No external dataset.

## 6. Train / Validation Subsets

`evaluation/task6b_subset_ids.json`, decided before training and never revised using model results.
Round-robin over per-level buckets ordered by `sample_id`, so selection is independent of file order and
of model behaviour.

| Subset | Size | Composition | Unique images | Image reuse |
|---|---|---|---|---|
| train | 480 | 160 L1 + 160 L2 + 160 nontrivial L3 | 480 | 0 |
| val | 120 | 40 L1 + 40 L2 + 40 nontrivial L3 | 120 | 0 |
| paired | 40 records / 20 pairs | 20 val images × 2 instructions, different targets, 20/20 different query types | 20 | — |

No trivial L3. Train and val come from disjoint splits. Test split never read
(`test_split_used: false`).

## 7. Fresh Baseline

Same 120 val records, clean ADR-013 architecture, no Task 6A checkpoint used as an initialisation.

| Metric | Value |
|---|---|
| valid `[SEG]` emission | **0 / 120** |
| strict end-to-end mIoU / Dice | 0.0000 / 0.0000 |
| conditional mIoU | n/a |
| operation-chain accuracy | 0.000 |
| teacher-forced mIoU / Dice | 0.00626 / 0.01004 |
| teacher-forced LM CE / perplexity | 3.4576 / 31.74 |
| teacher-forced reasoning-token accuracy | 0.4522 |
| teacher-forced `[SEG]` token accuracy | **0.000** |

The untrained model never emits `[SEG]`, and even teacher-forced it never puts probability on `[SEG]`.

## 8. Phase A Language Warm-up

Text LoRA + `[SEG]` only; projection and mask decoder frozen; LM CE only; 1 epoch = 480 steps; no SAM2
computation at all.

| | Baseline | After Phase A |
|---|---|---|
| training LM CE | — | 0.0003 at step 40, ~0.0000 by step 120 |
| val LM CE | 3.4576 | **0.2299** |
| valid `[SEG]` emission | 0.000 | **1.000** |
| operation-chain accuracy | 0.000 | 0.667 |
| strict end-to-end mIoU | 0.000 | 0.00708 |

Phase A alone moves the model from never emitting `[SEG]` to emitting exactly one on every unseen record.
The mask stays at chance, as designed, because the projection and decoder are frozen.

## 9. Phase B Joint Mini-Train

Continue from Phase A; text LoRA + `[SEG]` + projection MLP + SAM2 mask decoder; base and vision frozen.
Section 12 initial objective `L = 2.0·L_lm_ce + 2.0·L_mask_bce + 1.0·L_mask_dice`; 5 epochs / 2400-step
cap; early-stop patience 2; supervision at the original 512×512 resolution. The `best_joint` rule
(emission → strict e2e mIoU → operation chain → LM CE) was declared before Phase B and never changed.

| Epoch | Phase-B steps | valid `[SEG]` | strict e2e mIoU | chain acc. | val LM CE |
|---|---|---|---|---|---|
| 1 | 480 | 1.000 | 0.0932 | 0.667 | 0.2101 |
| 2 | 960 | 1.000 | 0.1101 | 0.867 | 0.0148 |
| **3** | **1440** | **1.000** | **0.1102** | **1.000** | 0.0020 |
| 4 | 1920 | 1.000 | 0.0991 | 1.000 | 0.000003 |
| 5 | 2400 | 1.000 | 0.0992 | 1.000 | 0.000002 |

Early stopping fired after two epochs without `best_joint` improvement; the best checkpoint is at **step
1440 (epoch 3)**. No NaN, no Inf, no OOM. Training mask loss never descended: `mask_dice` stayed in
0.65–1.00 and `mask_bce` in 0.01–0.05 for all 2400 steps — the cost signature of a near-all-background
prediction on small targets.

**The one permitted bounded adjustment (section 12)** was made once, after the first complete documented
epoch, because the segmentation objective clearly failed while the language objective was solved:

| | original (headline) | adjusted |
|---|---|---|
| loss weights | `lm_ce 2.0 / mask_bce 2.0 / mask_dice 1.0` | `lm_ce 1.5 / mask_bce 4.0 / mask_dice 3.0` (Task 6A's mask-dominant weighting) |
| Phase-B `decoder_lr` | 3e-4 | 1e-3 |
| strict end-to-end val mIoU | **0.11018** | **0.09498** |
| paired probe | 0 / 20 | 0 / 20 |
| best epoch / step | 3 / 1440 | 3 / 1440 |
| Phase-B epochs run | 5 | 5 |
| per-epoch mIoU | 0.0932, 0.1101, 0.1102, 0.0991, 0.0992 | 0.0903, 0.0948, 0.0950, 0.0874, 0.0776 |

**The adjustment did not help.** It is recorded rather than reverted. The headline run's artifacts are the
canonical `evaluation/task6b_*.json`, the adjustment is kept in full as
`evaluation/task6b_*_adjusted_recipe.json`, and the comparison is embedded in
`task6b_training_report.json → recipe_adjustment`.

No other configuration was tried.

## 10. Teacher-Forced Validation

Expected `reasoning_zh + [SEG]` given, 120 unseen val records, headline checkpoint:

| Metric | Baseline | Headline |
|---|---|---|
| mIoU / Dice | 0.00626 / 0.01004 | **0.11018 / 0.17874** |
| LM CE / perplexity | 3.4576 / 31.74 | **0.00200 / 1.0020** |
| assistant-token accuracy | 0.4405 | **1.000** |
| reasoning-token accuracy | 0.4522 | **1.000** |
| `[SEG]` token accuracy | 0.000 | **1.000** |
| greedy-decoded reasoning exact match | 0.000 | **1.000** |
| greedy-decoded operation-chain accuracy | 0.000 | **1.000** |

Teacher-forced mIoU equals free-generation mIoU exactly (`0.11017863780842695`). With a perfect prefix the
mask is identical, so the mask error is not a generation problem.

## 11. Free-Generation End-to-End Validation

Input is only the image and the instruction; the generated sequence is re-forwarded, the generated `[SEG]`
located, and its hidden state drives the mask. Zero or multiple `[SEG]` score IoU = 0, Dice = 0.

| Metric | Baseline | Headline |
|---|---|---|
| valid `[SEG]` emission | 0 / 120 (0.000) | **120 / 120 (1.000)** |
| **strict end-to-end mIoU** | 0.0000 | **0.11018** |
| strict end-to-end Dice | 0.0000 | 0.17874 |
| conditional mIoU / Dice | n/a | 0.11018 / 0.17874 |
| collapsed samples | 0 | 0 |
| generated reasoning exact match | 0.000 | **1.000** |
| generated reasoning char similarity | 0.2464 | **1.000** |
| generated operation-chain accuracy | 0.000 | **1.000** |
| mean generated tokens | 53.3 | 35.5 |

Improvement over baseline: **+0.11018 absolute**, clearing section 17's +0.10 bar by 0.010 — but see §9 and
§17: the margin is smaller than the run-to-run spread of Phase B, so the criterion is met **marginally**.
Conditional and strict mIoU coincide because emission is 120/120 — there is no format failure to hide
behind, and none to blame either.

## 12. Language / Reasoning Metrics

Reported as first-class, with the caveat that matters more than the numbers: **`reasoning_zh` is
template-generated.**

| | records | distinct `reasoning_zh` | distinct `template_id` |
|---|---|---|---|
| train mini-set | 480 | **21** | 39 |
| val mini-set | 120 | **20** | 37 |

All 20 distinct val reasoning strings also occur in the training mini-set. L1 has 6 distinct strings for
160 records; L3 has 4 for 160. This is why training LM CE collapses to ~0 within 40 optimizer steps.

Consequences: reasoning-token accuracy and reasoning exact match are **not** evidence of reasoning;
`operation_chain_accuracy` is the only language metric here that carries information because it checks the
chain rather than the wording; the interesting comparison is the mask, not the text.

Headline language metrics (120 unseen records): generation exact match 1.000, generation operation-chain
accuracy 1.000, generation char similarity 1.000, teacher-forced reasoning-token accuracy 1.000, teacher-
forced LM CE 0.00041.

## 13. L1 / L2 / L3 Breakdown

Strict end-to-end mIoU, headline checkpoint; emission is 1.000 in every stratum.

| Stratum | strict e2e mIoU | conditional mIoU |
|---|---|---|
| L1 | **0.1446** | 0.1446 |
| L2 | 0.1082 | 0.1082 |
| L3 nontrivial | **0.0778** | 0.0778 |

| Query family | strict e2e mIoU |
|---|---|
| `extreme` | **0.1496** |
| `size` | 0.1341 |
| `direction` | 0.0878 |
| `nearest` | **0.1898** |
| `multi_hop_direction_to_nearest` | **0.0778** |

Difficulty grows with the number of hops, as expected, but every stratum sits between 0.08 and 0.19: the
model is nowhere near a usable mask.

## 14. Paired Instruction Probe

20 unseen val images, two instructions per image with **different targets** and different query types. A
pair passes only if each prediction overlaps its own ground truth more than the other target's.

| | Headline | Adjusted |
|---|---|---|
| pairs passed | **0 / 20** | **0 / 20** |
| mean own-target IoU | 0.13885 | 0.10489 |
| mean cross-target IoU | 0.13873 | 0.10490 |
| pairs with both emissions valid | 20 / 20 | 20 / 20 |

Own-target and cross-target IoU agree to four decimals: the model is not choosing the wrong object, it is
producing nearly the same mask for both instructions.

**Diagnosis.** `scripts/task6b_diagnose_prompt.py`, recorded in
`evaluation/task6b_prompt_diagnosis_original_recipe.json` and
`evaluation/task6b_prompt_diagnosis_adjusted_recipe.json`, measures each stage of
`[SEG] hidden → Projection MLP → 256-d prompt → SAM2 mask decoder`, teacher-forced, with the same
measurement repeated across different images for scale:

| Stage | headline: same image, different instruction | headline: different images | adjusted: same / different images |
|---|---|---|---|
| `[SEG]` hidden cosine (2,048-d) | 0.9656 | 0.9975 | 0.9952 / 0.9996 |
| projected prompt cosine (256-d) | **0.999952** | **0.999997** | **0.999995** / **1.000000** |
| decoded mask IoU (a vs b) | **0.9970** | — | **0.9999** |

1. The `[SEG]` hidden state carries little referent (or even image) information: two instructions on one
   image are almost as similar to each other as two different images are.
2. The projection emits a **near-constant** 256-d prompt.
3. The masks are therefore nearly identical, and the decoder — which has the most trainable capacity — is
   not the failing component.

Gradient evidence agrees: at the first Phase-B step the gradient norms were `sam_mask_decoder 114.3`,
`projection 13.2`, `seg_token 1.57`, `lora 1.04` against a clip norm of 1.0, so the decoder consumes ≈99 %
of the clipped budget while the projection and LoRA adapters (the only places a referent-specific prompt
can come from) receive ≈11 % and <1 %.

## 15. VRAM / Runtime

RTX 5080 Laptop, 15.894 GiB:

| | Value |
|---|---|
| Phase A | 480 steps, 218.9 s |
| Phase B | 5 epochs × 480 steps = 2400 steps; 423.6 s for epoch 1 (feature-cache warm-up), then 311.8–317.3 s per epoch |
| optimisation time (Phase A + Phase B) | 1898.5 s ≈ 31.6 min |
| teacher-forced validation (120 records) | 18.5 s per pass |
| free-generation validation (120 records) | 375.0 s per pass |
| peak VRAM Phase A | 6.06 GiB allocated / 7.10 GiB reserved |
| peak VRAM Phase B | 6.17 GiB allocated / 7.13 GiB reserved |
| overall peak allocation | 5.37 GiB allocated / 7.13 GiB reserved of 15.894 GiB |
| SAM2 feature cache | LRU on CPU, 160 images ≈ 1.7 GB (section 18 cap 8 GB) |

The 160-image cache is smaller than the 480-image training set, so the LRU thrashes and features are
recomputed most epochs. Wall-clock cost only, worth raising in Task 6C.

## 16. Checkpoints

`evaluation/task6b_checkpoint_manifest.json` (+ `..._original_recipe.json`,
`..._adjusted_recipe.json`), with sha256 per role: `phaseA_last`, `best_mask`, `best_language`,
`best_joint`, `last`. The canonical manifest describes the headline checkpoint
`artifacts/checkpoints/task6b/best_joint.pt` at step 960, and the exact hash is also recorded in
`evaluation/task6b_validation.json → checkpoint`.

Two recorded facts about the checkpoints:

* checkpoints contain **only trainable tensors** (626 frozen base keys are absent by design;
  `unexpected_keys: 0`), so they are only meaningful on top of the same base models;
* `scripts/task6b_train.py` uses a module-level `CHECKPOINT_DIR` rather than `paths.checkpoints` from the
  config, so the adjusted run overwrote the first execution's checkpoint bytes. The headline recipe was
  re-run to regenerate a checkpoint whose bytes match the recorded manifest; the reproducibility
  comparison — including the finding that Phase B is not bit-reproducible — is in
  `task6b_training_report.json → recipe_adjustment.reproducibility_of_the_headline_recipe`.

## 17. Tests

See `docs/task6b_minitrain.md` §13 for the five defects found and fixed during Task 6B. The notable ones:

* teacher-forced token accuracy indexed `lm_logits` one position early (labels are pre-shifted), which
  reported ≈3 % accuracy alongside LM CE 0.00041 — contradictory, and the contradiction exposed the bug.
  Fixed and re-measured with `scripts/task6b_revalidate.py`; the baseline reasoning-token accuracy is
  0.4522 (not the 0.0160 first reported) and the trained value is 1.000;
* the teacher-forced path hard-coded its text-quality fields to zero by construction, so the first report
  showed `reasoning_exact_match 0.0` next to `generation_reasoning_exact_match 1.0`;
* `optimizer.*.token_lr` was silently unused because `trainable_parameter_groups` gave the `[SEG]` row
  `decoder_lr`. The headline recipe is unaffected (both are 3e-4); the adjusted recipe's `[SEG]` row
  trained at 1e-3 rather than the configured 3e-4. Fixed by adding an explicit `token_lr` argument;
* a test asserted the wrong label convention for the assistant tail; corrected.

Suite result: `python -m pytest tests/ -q` → **177 passed, 30 warnings, exit 0** (472.4 s). Full list of
defects in `docs/task6b_minitrain.md` §13.

* **`training.deterministic: true` is a dead flag — recorded, deliberately not changed.**
  `buildreasonseg_mvp/runtime.py` calls `set_seed` and never `torch.use_deterministic_algorithms` /
  `torch.backends.cudnn.deterministic`, so sampled CUDA kernels in the SAM2 path are not reproducible
  step-for-step. Re-running the identical headline recipe reproduced the fresh baseline and Phase A
  **bit-identically**, but not Phase B: epoch-1 mIoU was 0.1008 in the first execution and 0.0932 in the
  second, and the best-epoch mIoU was 0.11018644 versus 0.11017864. Enforcing determinism would change the
  numerics of an already-recorded recipe, so it was left alone and is the first recommended change for
  Task 6C. Consequence: the section 17 +0.10 bar is met **marginally**, not comfortably.

## 18. Negative Results

Recorded because they are the useful part of Task 6B:

1. **Mask-dominant loss weighting plus a 3.3× mask learning rate did not help.** 0.11019 → 0.09498, paired
   probe unchanged at 0/20. The failure is not a weighting or LR artefact.
2. **A 20-sample overfit result does not transfer.** Task 6A's 0.9807 mIoU / 10-per-10 pairs became 0.1102
   mIoU / 0-of-20 pairs at 480 training records.
3. **Perfect `[SEG]` emission is not evidence of segmentation capability.** 120/120 emission coexists with
   near-constant masks.
4. **Teacher-forced and free-generation mIoU being identical is a warning sign, not a good sign.** It means
   the mask carries no dependence on the language prefix.
5. **The language metrics cannot support a reasoning claim** on this dataset: 21 distinct reasoning strings
   in the whole training mini-set.
6. **A 4B scale-up is not justified as the fix.** The collapse is at the `[SEG]`→prompt interface and would
   plausibly reproduce.

## 19. Git Commit / Push

See the final DSH response for the exact commit hash and push result. Intended commit message:

```
train: validate 2B SEG mini-generalization
```

Staged: the Task 6B code, tests, configs, docs, ADR-014 and the `evaluation/task6b_*.json` artifacts.
Not staged: `.conda/`, `local_cache/`, `artifacts/`, `__pycache__/`, model weights, checkpoint bytes and the
frozen BuildSpatialReason JSONL.

## 20. Recommendation for Task 6C

The measured failure is a prompt-interface failure, so Task 6C should target the interface, not model scale:

1. **Stop the projection collapsing.** Nothing in the objective rewards a prompt that varies with the
   instruction. Candidates: normalise or whiten the 256-d prompt; add a point-inside-the-mask auxiliary
   supervision built from ground-truth geometry (supervision only, never an input); or give SAM2 more than
   one prompt channel.
2. **Fix the gradient budget.** Per-group clipping, or a lower decoder LR with a higher LoRA LR. This is a
   *different* intervention from the section 12 adjustment already tried and failed.
3. **Raise `feature_cache_images` to 480.**
4. **Keep the paired instruction-dependence probe as the primary gate.** It is the only metric that
   detected the real failure; mIoU alone read as a modest success.
5. **Only then consider 4B, and only as a scale measurement.**
