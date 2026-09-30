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

# FROM_DSH — Task 7D Report: Oracle-Reference L3 Relation-Guided Global Competition Decoder

_This file holds the Task 7D report. The Task 7C report is preserved in git history at commit `632c9c0`;
Task 7B at `b325585`; Task 7A at `6ee3d0d`._

**Note on the legacy `ARTIFACT-FACTS` block above:** those numbers describe the superseded
BuildSpatialReason **v0.1.1** dataset (read-only legacy evidence). The active canonical reasoning dataset
for all Task 6L+ work is **BuildSpatialReason v0.2 over WHU-EA-NativeVector v1.0**.

Full design notes: `docs/task7d_relation_guided_global_competition.md`.

## 1. Verdict

**`GLOBAL_COMPETITION_NO_MEANINGFUL_GAIN`** — section 30 priority order applied literally:

1. `INVALID_EXPERIMENT` — no: frozen paths and packs unchanged, no test use, no predicted reference, no
   competition supervision, no attention/Transformer/GNN, no GRCL.
2. `TASK6Z_PACK_MISMATCH` — no: all four Task 6Z pack SHA256 values match the frozen manifest.
3. `TASK6Z_BASELINE_REPRODUCTION_FAIL` — **no: D-B0 reproduced exactly** (mIoU Δ 0.0, Dice Δ 0.0, Paired
   15/20, margin Δ 0.0).
4. `GLOBAL_COMPETITION_NOT_LEARNABLE` — no: D-B2 Overfit20 mIoU **0.9678** / Dice **0.9835**.
5. `GLOBAL_COMPETITION_VISUAL_ONLY` — no: D-B2−D-B3 = **+0.1670** ≥ 0.08.
6. `GLOBAL_COMPETITION_PROTOTYPE_NOT_HELPFUL` — no: D-B2−D-B4 = −0.0026 < 0.03 but D-B4−D-B0 = +0.0049
   < 0.05.
7. **`GLOBAL_COMPETITION_NO_MEANINGFUL_GAIN`** — D-B2 mIoU **0.3265 < 0.38**, D-B2−D-B0 **+0.0023 < 0.05**
   and D-B2−D-B1 **−0.0714 < 0.03**. ← **verdict**
8. `GLOBAL_COMPETITION_COUNTERFACTUAL_WEAK` — not reached.
9. `RELATION_GUIDED_GLOBAL_COMPETITION_FEASIBLE` — no (4 of 11 section-28 criteria pass).

No threshold was changed after seeing results.

## 2. Recorded Task 7C outcome and frozen parser status

`L3_20CLASS_COMPOSITIONAL_ROBUSTNESS_FAIL`: full v0.2 val 18,222/18,222 = 1.0000, macro F1 1.0000, every
class recall 1.0000, Z-MiniVal240 240/240, Z-Paired members 40/40; fixed24 5/24, compact 0/8, minimal96
60/96, stress accuracy 0.6667 / macro F1 0.5924 / L3 macro recall 0.4479; checkpoint `c1505736…d58d9a`.
Parser status: **canonical/program-template parser frozen usable for development**, **free-form L3
paraphrase robustness a known unsolved limitation**, no further parser hardening authorized. Task 7D neither
uses nor retrains that parser.

## 3. Literature-position boundary (fixed note, no independent search performed)

Current 2026 RRSIS literature already covers semantic-role decomposition, relation-aware graph reasoning,
long-range/global dependency modelling and progressive mask refinement; SRGFormer (2026) uses
semantic-role-guided graph reasoning with a global sink node for same-class instance discrimination. Task 7D
therefore claims **no** novelty for global reasoning, relation-aware attention or long-range dependency
modelling; the narrower hypothesis under test is that explicit geometric relation fields can parameterize a
differentiable global competition distribution over dense frozen visual tokens whose winning distribution
synthesizes a target visual prototype for mask decoding, without target proposals or graph construction.

## 4. Frozen setting and D-B0 reproduction

Exactly the four L3 programs (canonical ids), oracle `oracle_native_gt` largest reference, byte-identical
Task 6Z packs (`z_overfit20`, `z_mini_train_1200`, `z_mini_val_240`, `z_paired_val20`; all SHA256 verified),
frozen fields (`alpha 1.2`, `tau 0.04`, `s_axis 0.02`, `s_margin 0.02`; `sigma_diag 0.05`) and the frozen
SAM2.1 Hiera Base+ `V ∈ R^(256×64×64)`.

D-B0: mIoU **0.3242128983** (Δ **0.0**), Dice **0.4389840056** (Δ **0.0**), Paired **15/20**, margin
**+0.3003540897** (Δ **0.0**) → `TASK6Z_BASELINE_REPRODUCTION_PASS`.

## 5. Architecture and training

Commons: `F = Conv1x1(256→128) + GroupNorm(8,128) + GELU`; direction embedding vocab 4 / dim 16 (no nearest
embedding); `temperature = 1.0`, `eps = 1e-6`, 4096 spatial tokens. Learned competition
`A = softmax(S.flatten(2)/1.0, dim=-1)` (never detached; measured spatial sum exactly 1.000000), field
competition (D-B1) `A_fixed = clamp(P_dir*P_near) / (sum + eps)` with `INVALID_FIELD_MASS` on zero mass,
prototype `q = Σ A_i F_i` (no stop-gradient), similarity `C_i = <F_norm_i, q_norm>` (plain cosine). Trunk
`Conv3x3(in→128)+GN+GELU → Conv3x3(128→64)+GN+GELU → Conv1x1(64→1)`, bilinear 64→512; loss exactly
`BCEWithLogitsLoss + DiceLoss`.

| Variant | Score head in | Decoder in | Params | Overfit20 | MiniVal mIoU | Dice | Pr@0.5 | Paired | margin |
|---|---|---|---|---|---|---|---|---|---|
| D-B0 | — (frozen Z-B3) | — | 275,777 | — | 0.3242 | 0.4390 | 0.5331 | 15/20 | +0.3004 |
| **D-B1** | none (deterministic) | 148 | 278,081 | 0.9623/0.9807 | **0.3979** | **0.5273** | **0.5620** | **19/20** | **+0.3888** |
| D-B2 (primary) | 146 | 148 | 362,434 | **0.9678/0.9835** | 0.3265 | 0.4373 | 0.5454 | 17/20 | +0.2956 |
| D-B3 | 144 | 146 | 358,978 | 0.9591/0.9790 | 0.1595 | 0.2311 | 0.2760 | 8/20 | +0.1629 |
| D-B4 | 146 | 147 | 361,282 | 0.9679/0.9836 | 0.3291 | 0.4488 | 0.4618 | 19/20 | +0.3056 |

D1: lr 1e-3 / wd 1e-4 / batch 4 / 1200 steps / eval every 100 → `GLOBAL_COMPETITION_OVERFIT_PASS`.
D2: lr 3e-4 / wd 1e-4 / batch 8 / ≤25 epochs / patience 5 / selection by MiniVal240 mIoU / seed 20261001;
wall 32.1-96.8 s and peak VRAM 0.667-0.698 GB per variant. Deltas: **D-B1−D-B0 +0.0737**,
D-B2−D-B0 **+0.0023**, D-B2−D-B1 **−0.0714**, D-B2−D-B3 **+0.1670**, D-B2−D-B4 **−0.0026**,
D-B3−D-B0 −0.1647, D-B4−D-B0 +0.0049.

## 6. Competition diagnostics

| Variant | target mass | argmax-in-target | reference mass | normalized entropy | top-1 | top-16 | top-64 |
|---|---|---|---|---|---|---|---|
| D-B1 | **0.0675** | 0.0083 | 0.0024 | **0.6933** | 0.0122 | **0.1670** | **0.4751** |
| D-B2 | 0.0067 | 0.0000 | 0.0862 | 0.8848 | 0.0105 | 0.0923 | 0.2202 |
| D-B3 | 0.0189 | 0.0208 | 0.0968 | 0.8844 | 0.0091 | 0.0876 | 0.2164 |
| D-B4 | 0.0051 | 0.0000 | 0.1040 | 0.8495 | 0.0146 | 0.1224 | 0.2800 |

Flags: `global_competition_mask_gain` **false**, `prototype_gain` **false**, `field_guidance_gain` **true**,
`competition_localizes_target` **false**.

## 7. Tests, storage, git

`python -m pytest tests/ -q` → **1243 passed, 1 skipped** (Task 7C ended at 1192 passed / 1 skipped; no prior
passing test was reduced). `tests/test_task7d_global_competition.py` adds the 51 section-R checks.

Not committed: the four Task 7D checkpoints, caches, model weights (YOLO/SAM2/Z-B3), source imagery/vectors,
`.conda`. Committed: decoder/data code, small JSON evaluations, scripts, tests, docs, handoff.

Task 7D downloaded nothing and installed nothing. Watt was **not needed** in Task 7D: the pre-existing Watt
instance is transport-only, is not owned by this project and was left running per the ownership rule; no
proxy, host, certificate or TLS setting was read or modified.

## 8. Interpretation boundary

DSH reports measurements only. Global competition is **not** claimed as a novelty; no final architecture is
claimed; no predicted reference was integrated; parser/reference were not changed; no graph/attention module
was added; no supervision was added to the competition map; no formal full-data training was started; the
test split was not accessed. Measured reading (no repair proposed): the deterministic field-weighted
prototype variant D-B1 is the only variant that clearly beats the frozen Z-B3 baseline
(+0.0737 mIoU, 19/20 paired, margin +0.3888), while the learned relation-guided competition (D-B2) is
statistically indistinguishable from the baseline (+0.0023) and its competition maps remain near-uniform
(entropy 0.8848 of the 4096-token maximum) and do not localize the target (mass 0.0067, argmax-in-target
0.0000).

## 9. Recommended next step (exact wording required by Part P)

等待 ChatGPT 根据 Task 7D 的 oracle-reference global competition 因果结果决定是否替换 Z-B3，不自行加入 attention/graph、predicted reference 或正式全量训练。

## 10. STOP

Task 7D stops here: no further parser experiment, no reference re-hardening, no attention/graph module, no
predicted-reference integration, no formal full training, no test access, no GUI. Waiting for the ChatGPT
audit.
