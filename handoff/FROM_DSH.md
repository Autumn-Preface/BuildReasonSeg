# FROM_DSH — Task 6C Report: Paired Counterfactual Training × Neutral SAM Prompt Ablation

**Date:** 2026-09-26
**Actor:** DSH
**Task:** Task 6C (from `handoff/TO_DSH.md`)
**Verdict: `EXPERIMENT_COMPLETE_PARTIAL_IMPROVEMENT`**

> A valid, bit-reproducible 2×2 on the frozen ADR-013 stack. **No arm clears the fix gate**: all four
> score **0/20** on the paired unseen validation probe with mean own-minus-cross margins between −0.0035
> and +0.000007. But paired counterfactual training materially changes the prompt representation
> (projected effective rank 1.50 → 3.74, top-1 variance 0.905 → 0.616), so the verdict is a partial
> improvement rather than "no fix".

> The Task 6B report is preserved verbatim in `handoff/ARCHIVE_task6b_report.md`; the Task 6A report is in
> `handoff/ARCHIVE_task6a_report.md`. This file is the Task 6C report.

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

**`EXPERIMENT_COMPLETE_PARTIAL_IMPROVEMENT`.**

The experiment is valid (§6: identical initialization fingerprints, four distinct checkpoint directories,
valid U/P subsets, no test-split contact, cross-process bit reproducibility). No arm clears the fix gate:
all four are 0/20 on the paired probe with margins within ±0.0036, and strict end-to-end mIoU runs
0.0928–0.1087 against a 0.11 bar. Two arms improve materially over `U_C` on the prompt representation:
`P_C` (effective rank +2.58, top-1 variance −0.313) and `P_L` (+1.85, −0.259).

The causal reading is `deeper_representation_problem`: neither the sampling scheme nor the fixed centre
point is the cause, and both were excluded by measurement, not by argument.

## 2. Correctness Fixes

Four Task 6B defects were fixed before any arm ran; each has a regression test.

| Fix | Was | Is |
|---|---|---|
| Determinism | `training.deterministic: true` never consumed; Phase B not reproducible across processes | `build_runtime` → `enable_determinism`: seeds `random`/NumPy/torch/CUDA, `cudnn.benchmark=False`, `cudnn.deterministic=True`, `torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, each setting reported individually |
| Checkpoint directory | module-level constant `CHECKPOINT_DIR` in `scripts/task6b_train.py`; the adjusted run overwrote the headline run | `cfg["paths"]["checkpoints"] / <ARM>` → four distinct directories; the constant is gone |
| Evaluation lookup | operation-chain lookup built from `train + val` text | built from the **train split only**; the arm report carries `lookup_audit.val_text_used = false` |
| Feature cache | 160 images for 480 training images | 480 images, with `image_pe` and the no-mask dense embedding shared once instead of once per image |

A fifth defect was found while fixing the fourth: `Sam2FeatureCache` mutated the stored entry when moving
it to the device, which both leaked 8 MiB of VRAM per cached image and double-counted the constants. The
cache now builds copies (`dataclasses.replace`) and never mutates what it stores.

## 3. Determinism Status

**Genuinely active, in strict mode, with cross-process bit reproducibility.**

`evaluation/task6c_determinism.json`: two independent processes ran the same 2-sample forward/backward plus
one optimizer step with the same seed.

| Check | Result |
|---|---|
| `torch.use_deterministic_algorithms(True)` accepted | **yes** (no non-deterministic operation on the path) |
| `warn_only` fallback needed | **no** |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` |
| losses identical | **yes** |
| gradient digests identical | **yes** |
| post-step trainable-state digests identical | **yes** |
| `cross_process_bit_reproducible` | **true** |
| `bit_reproducibility_claimed` | **true** |

An independent confirmation: the `U_C` arm was run once before the cache change and once after, and both
executions produced **bit-identical** metrics (strict e2e mIoU 0.10515983039163602, Dice 0.1735980396815246,
LM CE 0.003345464280168168, paired margin −8.79e-07). The cache change is therefore provably
value-preserving, not merely assumed to be.

## 4. Subset Construction

`evaluation/task6c_subset_ids.json`. Both subsets come from the **train** split; `test_split_used: false`.

**U** — Task 6B's exact 480 sample ids, unchanged: 480 records over 480 unique images, 160/160/160.

**P** — 480 records over **240 unique images × 2 instructions**, 2 records per image, built deterministically:

* composition **80 × (L1+L2) + 80 × (L1+L3nt) + 80 × (L2+L3nt)** → exactly **160 L1 + 160 L2 + 160 L3**;
* every pair has **different target components** and **different query types** (240/240 both);
* **no trivial L3** (160 nontrivial);
* the preferred 80/80/80 split was **feasible exactly** — the train split offers 1454 eligible images for
  L1+L2, 754 for L1+L3 and 618 for L2+L3 — so no relaxation was needed and none was made;
* query types are balanced: L1-side 26–28 per type, L2-side 16 across the two L2 buckets, L3-side 20 per
  direction.

Selection orders candidates by query type, image id and sample id and, at each of the 240 slots, takes the
candidate whose query-type usage is smallest; no model output participates. Rebuilding twice gives
identical ids.

Validation material is Task 6B's verbatim: the same 120 records (40/40/40) and the same 20 paired images.

## 5. Four Experimental Arms

| Arm | Subset | Bridge | Definition |
|---|---|---|---|
| `U_C` | U | centre | Task 6B behaviour: fixed point (0.5, 0.5), positive label, projected vector added |
| `U_L` | U | language | `sparse_prompt_embeddings = projected[:, None, :]`; no point, box or mask prompt |
| `P_C` | P | centre | as `U_C` but paired counterfactual data |
| `P_L` | P | language | as `U_L` but paired counterfactual data |

Shared: Qwen3-VL-2B snapshot `89644892e4d85e24eaac8bacfd4f463576704203`, SAM2.1 Base+ checkpoint, tokenizer
and `[SEG]` id 151669, seed 20260926, Phase A 1 epoch (480 steps, LoRA + `[SEG]`, LM CE only), Phase B
3 epochs (1440 steps, LoRA + `[SEG]` + Projection MLP + SAM2 mask decoder), loss
`2.0·L_lm_ce + 2.0·L_mask_bce + 1.0·L_mask_dice`, LoRA 1e-4 / token 3e-4 / decoder 3e-4, warmup 20, cosine,
grad clip 1.0, supervision at 512×512, `multimask_output=False`. No per-epoch selection and no tuning.

The `L` bridge leaves the official prompt encoder in charge of the image positional encoding and the
no-mask dense embedding (both produced inside `Sam2Encoder.encode`); it simply creates no point
coordinates, no point labels, no boxes and no mask prompt. The measured sparse-prompt shape is `(1, 1, 256)`
for `L` and `(1, 2, 256)` for `C`, because SAM's prompt encoder appends its own padding slot in the `C`
path. That difference is recorded rather than hidden.

## 6. Initialization Equality

Every arm starts from a clean base (fresh LoRA, fresh `[SEG]` adapter, fresh Projection MLP, original SAM2
mask decoder). Before the first optimizer step each arm hashes every trainable tensor:

| Arm | initial trainable SHA-256 | tensors | parameters |
|---|---|---|---|
| `U_C` | `97daa58a4cff0ae904512d79a7ae50ec4b96abbd36e25c2b37f95d8956416acf` | 528 | 24,010,309 |
| `U_L` | identical | 528 | 24,010,309 |
| `P_C` | identical | 528 | 24,010,309 |
| `P_L` | identical | 528 | 24,010,309 |

Checkpoint directories are distinct: `artifacts/checkpoints/task6c/{U_C,U_L,P_C,P_L}/`. No arm can
overwrite another.

## 7. U_C Result — the Task 6B baseline arm

| Metric | Value |
|---|---|
| valid `[SEG]` emission | 1.000 (120/120) |
| strict e2e mIoU / Dice | 0.10516 / 0.17360 |
| conditional mIoU | 0.10516 |
| teacher-forced mIoU | 0.10516 |
| operation-chain accuracy | 0.983 |
| val LM CE | 0.00335 |
| L1 / L2 / nontrivial L3 | 0.1231 / 0.1117 / 0.0807 |
| **paired** | **0/20**, own 0.13149, cross 0.13149, margin −0.000001, IoU(pred_A,pred_B) 0.99971 |
| projected cosine (same image) / effective rank | 0.999997 / 1.516 |

This reproduces Task 6B's failure with three fixed epochs instead of Task 6B's best epoch, i.e. the collapse
is not an artefact of Task 6B's epoch selection or of its missing determinism.

## 8. U_L Result — language-only bridge on the same data

| Metric | Value |
|---|---|
| valid `[SEG]` emission | 1.000 |
| strict e2e mIoU / Dice | **0.10872** / 0.17846 |
| operation-chain accuracy | 0.808 |
| val LM CE | 0.04900 |
| L1 / L2 / nontrivial L3 | 0.1213 / 0.1171 / 0.0878 |
| **paired** | **0/20**, own 0.12833, cross 0.13186, margin **−0.003530**, IoU(pred_A,pred_B) 0.97903 |
| projected cosine / effective rank | 0.999909 / 1.491 |

Removing the fixed centre point slightly raises mIoU and makes the decoded masks a little more variable
(prediction-to-prediction IoU 0.9997 → 0.9790), but the paired probe stays at zero and the mean margin is
slightly *negative* — the prediction matches the other target marginally better than its own. Instruction
conditioning does not appear.

## 9. P_C Result — paired data, centre bridge

| Metric | Value |
|---|---|
| valid `[SEG]` emission | 1.000 |
| strict e2e mIoU / Dice | 0.10604 / 0.17851 |
| operation-chain accuracy | 1.000 |
| val LM CE | 0.00000 |
| L1 / L2 / nontrivial L3 | 0.1144 / 0.1227 / 0.0810 |
| **paired** | **0/20**, own 0.13475, cross 0.13474, margin +0.000007, IoU(pred_A,pred_B) 0.99945 |
| projected cosine / **effective rank** | 0.999996 / **4.100** (top-1 variance 0.589, was 0.902) |

The representation is materially more diverse than `U_C` while the mask-level behaviour is unchanged: the
projection spreads across more dimensions but two instructions on one image still produce the same mask.

## 10. P_L Result — paired data, language-only bridge

| Metric | Value |
|---|---|
| valid `[SEG]` emission | 1.000 |
| strict e2e mIoU / Dice | **0.09284** / 0.15626 |
| operation-chain accuracy | 1.000 |
| val LM CE | 0.00000 |
| L1 / L2 / nontrivial L3 | 0.1005 / 0.1018 / 0.0763 |
| **paired** | **0/20**, own 0.12374, cross 0.12375, margin −0.000003, IoU(pred_A,pred_B) 0.99972 |
| projected cosine / **effective rank** | 0.999999 / **3.371** (top-1 variance 0.643) |

The combination of both interventions is the worst arm on mIoU (0.0928) while also improving the
representation. More prompt diversity did not translate into mask quality or conditioning.

## 11. Paired Validation Comparison

| Arm | passed | own IoU | cross IoU | mean margin | median margin | mean IoU(pred_A, pred_B) |
|---|---|---|---|---|---|---|
| `U_C` | 0/20 | 0.13149 | 0.13149 | −0.000001 | 0.000000 | 0.99971 |
| `U_L` | 0/20 | 0.12833 | 0.13186 | −0.003530 | 0.000000 | 0.97903 |
| `P_C` | 0/20 | 0.13475 | 0.13474 | +0.000007 | 0.000000 | 0.99945 |
| `P_L` | 0/20 | 0.12374 | 0.12375 | −0.000003 | 0.000000 | 0.99972 |

The gate needs ≥ 14/20 and a mean margin > 0.05. Every arm is 0/20 and every margin is within ±0.0036 of
zero; the median margin is exactly 0.0 in all four arms. The best individual pair margin anywhere in the
480 pair-evaluations is +0.0002.

## 12. Prompt Representation Diagnostics

Ten same-image pairs and ten cross-image references per arm, teacher-forced, identical code path.

| Arm | `[SEG]` hidden cos (same img) | projected cos (same img) | sparse-prompt cos | prediction IoU (A vs B) | top-1 var. | top-5 var. | effective rank |
|---|---|---|---|---|---|---|---|
| `U_C` | 0.99238 | 0.999997 | 0.999997 | 0.9999 | 0.9023 | 0.9959 | 1.516 |
| `U_L` | 0.98976 | 0.999909 | 0.999909 | 0.9595 | 0.9087 | 0.9952 | 1.491 |
| `P_C` | 0.99719 | 0.999996 | 0.999996 | 0.9993 | **0.5892** | 0.9109 | **4.100** |
| `P_L` | 0.99898 | 0.999999 | 0.999999 | 0.9999 | **0.6430** | 0.9512 | **3.371** |

* Wording used: **directional collapse / low-rank collapse / near-constant direction** — not "constant".
  The directions are nearly identical while the norms and L2 distances still vary, and the effective rank
  is non-zero (1.5–4.1).
* Factor effects: sampling **material** (effective rank +2.23, top-1 variance −0.289, both beyond the
  pre-declared thresholds), bridge **not material** (rank −0.377, top-1 +0.030).
* Bridge-specific: for `C`, the point embedding norm is 11.39 and the projected language norm 266.8, a
  ratio of **23.4×** (`P_C` 18.3×) — the language vector dominates the sparse prompt by an order of
  magnitude. For `L`, the sparse prompt equals the projected token exactly
  (`sparse_prompt_equals_projected_all = true`).

## 13. 2×2 Causal Interpretation

Pre-declared rule: a factor is materially better when the mean paired passed count improves by ≥ 3 or the
mean margin by ≥ 0.05.

| Comparison | Paired passed | Mean margin | Representation diversity |
|---|---|---|---|
| **P vs U** | 0.0 vs 0.0 | +0.000002 vs −0.001765 | **material** (rank 3.74 vs 1.50, top-1 0.616 vs 0.905) |
| **L vs C** | 0.0 vs 0.0 | −0.001766 vs +0.000003 | not material (rank 2.43 vs 2.81, top-1 0.776 vs 0.746) |

**Code: `deeper_representation_problem`.** Because neither factor moves the paired probe while the factors
do move the representation, the failure is not fixed by paired counterfactual sampling nor by removing the
fixed positive centre point. The sampling scheme and the point prior are both excluded; what remains is how
the `[SEG]` hidden state is formed and how SAM's decoder turns a prompt direction into a region. Prompt
normalisation, a multi-token prompt, auxiliary point supervision or `[REF]` are justified now — and were
not before.

An important supporting measurement: the Task 6B bridge's fixed point is *not* drowning the language
signal (ratio 23.4× in language's favour), so "the centre point overrides the language vector" is not
supported. The mask stays put even when the language vector is essentially the whole prompt.

## 14. Language Metric Caveat

Unchanged and repeated deliberately: `reasoning_zh` is template-generated (21 distinct strings in the whole
training mini-set), so reasoning exact match and operation-chain accuracy measure **language-format /
template mapping**, not reasoning. All four arms emit exactly one `[SEG]` on 120/120 unseen records and
`P_C`/`P_L` reach 1.000 operation-chain accuracy while their masks remain instruction-independent — which is
precisely why the paired mask probe, not the text metrics, is the reasoning evidence here.

## 15. Runtime / VRAM

| | Value |
|---|---|
| Optimisation per arm | 1141–1256 s (Phase A ≈ 168 s; Phase B ≈ 1000 s over three epochs) |
| Wall clock per arm | ≈ 35–37 min including model load, three teacher-forced epochs, the final free-generation pass, the paired probe and the diagnosis |
| Peak VRAM | 5.37 GiB allocated / **8.42 GiB reserved** of 15.894 GiB — identical across all four arms |
| SAM2 CPU feature cache | U arms 480 images = **7.508 GiB** (16 MiB per image + 8 MiB shared constants); P arms 240 images = 5.805 GiB. Within the 8 GiB budget in GiB terms; 8.06 GB in decimal units, stated rather than rounded down |
| Cache sharing | enables the whole 480-image subset to fit; without it the same run measured 11.25 GiB |

## 16. Tests

Task 6C adds three test files covering all twenty items of the section 17 list:

| File | Items | Result |
|---|---|---|
| `tests/test_task6c_fixes.py` | 1, 2, 3, 4, 5 + cache budget and cache-sharing correctness | 8/8 |
| `tests/test_task6c_subsets.py` | 6, 7, 8, 9, 10, 11, 12, 13 | 8/8 |
| `tests/test_task6c_bridge.py` | 14, 15, 16, 17, 18, 19, 20 | 8/8 |

Full-suite collected / passed / failed / skipped / exit code are reported in the final DSH response.

## 17. Git / Watt Push Lifecycle

**What actually happened, including a process miss.** Task 6C intended to keep Watt Toolkit closed for the
whole training and evaluation phase. It was **not** closed: Watt Toolkit (`Steam++.exe`) was already running
when this task began, with its start time recorded as **14:16:23** — before the first command of the task —
and `Steam++.Accelerator.exe` held `:443`/`:80` throughout the 15:16–17:40 training window. DSH did not
verify the accelerator state at the start of Task 6C (it had been verified at the end of the preceding
lifecycle test) and did not notice until the final state check. The miss is recorded rather than papered
over.

**Why the measurements are still sound.** Every Task 6C entry point sets `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` and loads Qwen and SAM2 from `local_cache/`; the determinism, subset, training,
comparison and visualization scripts make **no network calls at all**. The hosts redirection therefore
changed DNS resolution for four host names and touched nothing the runs read. What cannot be claimed is a
pristine network environment during training.

**The push.** `git push origin main` succeeded on the **first attempt** (`3bd7341..c80169b`), with no need to
use the `FULL_AUTO_OK` start procedure. Because the hosts file redirects `github.com` to `127.0.0.1` while
Watt is active, that push was most likely carried by Watt's acceleration rather than by a direct
connection, so this round does **not** demonstrate that a push succeeds with the accelerator off. The
confirmed check afterwards: `github.com`, `api.github.com` and `huggingface.co` all resolve to `127.0.0.1`,
and `git ls-remote origin` returns the pushed commit.

**Nothing was stopped.** At the time of the final check `uu_launcher.exe` had just been started (18:03:00)
and had rewritten the hosts file (18:03:27), so the accelerators appeared to be in active use by the user.
DSH therefore did **not** close Watt Toolkit and did not touch the hosts file: the task's Watt authorization
covers starting and closing an accelerator that DSH itself started for the push, and that did not happen.
The machine is left with Watt Toolkit running and the hosts file carrying both the `#uu_acc` and
`# Steam++` blocks; the user has been told. `taskkill /F`, `Stop-Process -Force`, hosts edits,
certificate-store edits and `verify=False` were never used.

## 18. Negative Results

1. **Paired counterfactual training does not create instruction conditioning.** `P_C` 0/20, `P_L` 0/20,
   margins ≈ 0. The most natural explanation of Task 6B's failure is therefore not the sampling scheme.
2. **Removing the fixed positive centre point does not fix it.** `U_L` 0/20 with a slightly *negative*
   margin; the L-vs-C diversity difference is below the pre-declared threshold.
3. **Neither does the combination.** `P_L` is the worst arm on strict mIoU (0.0928).
4. **More prompt diversity did not become mask diversity.** `P_C`/`P_L` raise the projected effective rank
   to 3.4–4.1 while IoU(pred_A, pred_B) stays at 0.999.
5. **The centre point is not the culprit by magnitude.** Ratio 23.4× in favour of the language vector.
6. **Task 6B's defect list was longer than reported**: `training.deterministic` was dead, the checkpoint
   directory was hard-coded, the operation lookup saw val text, and the feature cache both thrashed and —
   after the first fix — leaked VRAM by mutating stored entries. All are fixed and tested.

## 19. Recommendation for Task 6D

The remaining problem is after the prompt, so Task 6D should target it rather than model scale:

1. **Make the `[SEG]` hidden state instruction-conditional before it reaches the projection.** Candidates:
   supervise the prompt against the target region (a point-inside-the-mask auxiliary loss built from
   ground-truth geometry as *supervision only*), or add a second, explicitly *negative* prompt slot so the
   decoder has something to contrast.
2. **Then test prompt conditioning directly.** Normalisation/whitening of the 256-d prompt and a
   multi-token prompt are now justified, because §13 excluded the two cheaper explanations.
3. **Keep the paired probe as the primary gate** and keep the four Task 6C correctness fixes; without
   determinism and per-arm checkpoints, a four-arm comparison is not interpretable.
4. **Do not scale to 4B yet.** Nothing measured here suggests more parameters would make an
   instruction-independent prompt instruction-conditional.
5. **`[REF]` remains future work** and is still prior art; the novelty claim stays with the relation-level
   encoder and loss, which Task 6D still does not touch.
