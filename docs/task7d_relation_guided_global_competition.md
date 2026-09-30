# Task 7D — Oracle-Reference L3 Relation-Guided Global Competition Decoder

> Task: `handoff/TO_DSH.md` (Task 7D) · Base commit: `632c9c0` · Predecessor: Task 7C →
> `L3_20CLASS_COMPOSITIONAL_ROBUSTNESS_FAIL`
> **Verdict: `GLOBAL_COMPETITION_NO_MEANINGFUL_GAIN`** · D-B0 reproduced **exactly** · D-B2 overfit **passed**
> Tests: `tests/test_task7d_global_competition.py` · Evidence: `evaluation/task7d_*.json`

Task 7D tests one architecture-class pivot: **relation-guided global competition over dense frozen SAM2
tokens, followed by a dynamically selected target visual prototype used for mask decoding**. The oracle
reference and canonical L3 program ids keep parser and reference errors out of the question. No parser,
reference, field or SAM2 change; no attention/Transformer/GNN module; no target proposals; no auxiliary
competition loss; no GRCL.

## 1. Recorded Task 7C outcome and frozen parser status

Verdict `L3_20CLASS_COMPOSITIONAL_ROBUSTNESS_FAIL`. Canonical behaviour: full v0.2 val **18,222/18,222 =
1.0000**, macro F1 1.0000, every class recall 1.0000, Z-MiniVal240 240/240, Z-Paired members 40/40.
Held-out natural-language robustness: Task 7A fixed24 **5/24**, compact **0/8**, minimal96 **60/96**, stress
accuracy 0.6667, stress macro F1 0.5924, stress L3 macro recall 0.4479. Task 7C checkpoint
`artifacts/checkpoints/task7c/program_parser_l3_rehearsal_v1.pt` (SHA256 `c1505736…d58d9a`).

Parser status after Task 7C: **canonical/program-template parser status frozen usable for development**;
**free-form L3 paraphrase robustness is a known, unsolved limitation**; no further parser hardening is
authorized here. Task 7D does not use or retrain that parser.

## 2. Literature-position boundary

Current 2026 RRSIS literature already covers semantic-role decomposition, relation-aware graph reasoning,
long-range/global dependency modelling and progressive mask refinement; SRGFormer (2026) uses
semantic-role-guided graph reasoning with a global sink node for same-class instance discrimination. Task 7D
therefore claims **no** novelty for global reasoning, relation-aware attention or long-range dependency
modelling. The narrower experimental hypothesis is:

> **Explicit geometric relation fields can parameterize a differentiable global competition distribution over
> dense frozen visual tokens; the winning distribution can synthesize a target visual prototype, which then
> guides pixel-level mask decoding without target proposals or graph construction.**

This is an architecture hypothesis, not a novelty claim.

## 3. Frozen setting

Exactly the four L3 programs (`largest_to_{left_of,right_of,above,below}_to_nearest`), canonical program ids
directly, oracle `oracle_native_gt` largest reference, and the byte-identical Task 6Z packs —
`z_overfit20`, `z_mini_train_1200`, `z_mini_val_240`, `z_paired_val20` — with all four SHA256 values verified
against `evaluation/task6z_pack_manifest.json`. Frozen fields (`alpha 1.2`, `tau 0.04`, `s_axis 0.02`,
`s_margin 0.02`; `sigma_diag 0.05`) generate `P_dir_64`/`P_near_64`, and the frozen SAM2.1 Hiera Base+
`V ∈ R^(256×64×64)` is reused with no backbone training.

## 4. Global competition primitives

`buildreasonseg_mvp/task7d_global_competition_decoder.py`. Commons: `F = Conv1x1(256→128) +
GroupNorm(8,128) + GELU`; trainable direction embedding vocab 4 / dim 16 (no nearest embedding);
`competition_temperature = 1.0`, `eps = 1e-6`, 4096 spatial tokens — all frozen.

```text
learned:  S      = Conv3x3(in→64) + GroupNorm(8,64) + GELU + Conv1x1(64→1)
          A_flat = softmax(S.flatten(2) / 1.0, dim=-1) ;  A = reshape(B,1,64,64) ;  A_vis = A * 4096
deterministic (D-B1): W = clamp(P_dir_64 * P_near_64, 0, 1) ;  A_fixed = W / (sum W + eps)
prototype: q = Σ_i A_i F_i (no stop-gradient) ;  C_i = <F_norm_i, q_norm> (plain cosine, no learned scale)
```

`A` is never detached; the measured spatial sum is exactly `1.000000` for every variant. `INVALID_FIELD_MASS`
is raised whenever the field mass is ≤ eps. Decoder trunk (all variants): `Conv3x3(in→128) + GN + GELU →
Conv3x3(128→64) + GN + GELU → Conv1x1(64→1)`, logits bilinearly upsampled 64→512 with
`align_corners=False`; loss exactly `BCEWithLogitsLoss + DiceLoss`.

| Variant | Score head | Decoder inputs | Channels | Params |
|---|---|---|---|---|
| D-B0 | frozen Task 6Z Z-B3 baseline (not retrained) | — | — | 275,777 |
| D-B1 | none (deterministic field competition) | F + P_dir + P_near + embed + A_fixed_vis + C_fixed | 148 | 278,081 |
| **D-B2** | 146 = F + P_dir + P_near + embed | + A_vis + C | 148 | **362,434** |
| D-B3 | 144 = F + embed | F + embed + A_vis + C | 146 | 358,978 |
| D-B4 | 146 = F + P_dir + P_near + embed | F + P_dir + P_near + embed + A_vis (no q, no C) | 147 | 361,282 |

## 5. D-B0 reproduction, stages D1 and D2

**D-B0 (section 16)** re-ran the frozen Z-B3 through the frozen Task 6Z evaluation path:
mIoU **0.3242128983** (Δ **0.0**), Dice **0.4389840056** (Δ **0.0**), Paired **15/20**, margin
**+0.3003540897** (Δ **0.0**) → `TASK6Z_BASELINE_REPRODUCTION_PASS`.

**D1 Overfit20** (AdamW lr 1e-3, wd 1e-4, batch 4, 1200 steps, eval every 100, seed 20261001, bf16):
D-B1 0.9623/0.9807, **D-B2 0.9678/0.9835 (gate ✓)**, D-B3 0.9591/0.9790, D-B4 0.9679/0.9836 →
`GLOBAL_COMPETITION_OVERFIT_PASS`.

**D2 MiniTrain1200 → MiniVal240** (AdamW lr 3e-4, wd 1e-4, batch 8, ≤25 epochs, patience 5, selection by
MiniVal240 mIoU, seed 20261001).

## 6. Evaluation

| Variant | MiniVal mIoU | Dice | Pr@0.5 | PairedVal | own-cross margin | best epoch | wall | peak VRAM |
|---|---|---|---|---|---|---|---|---|
| D-B0 (frozen Z-B3) | 0.3242 | 0.4390 | 0.5331 | 15/20 | +0.3004 | — | — | — |
| **D-B1** | **0.3979** | **0.5273** | **0.5620** | **19/20** | **+0.3888** | 7 | 96.8 s | 0.667 GB |
| D-B2 (primary) | 0.3265 | 0.4373 | 0.5454 | 17/20 | +0.2956 | 10 | 48.6 s | 0.698 GB |
| D-B3 | 0.1595 | 0.2311 | 0.2760 | 8/20 | +0.1629 | 7 | 36.7 s | 0.698 GB |
| D-B4 | 0.3291 | 0.4488 | 0.4618 | 19/20 | +0.3056 | 6 | 32.1 s | 0.681 GB |

Deltas: **D-B1−D-B0 +0.0737**, D-B2−D-B0 **+0.0023**, D-B2−D-B1 **−0.0714**, D-B2−D-B3 **+0.1670**,
D-B2−D-B4 **−0.0026**, D-B3−D-B0 −0.1647, D-B4−D-B0 +0.0049. D-B2 per direction: above 0.2714 /
below 0.3085 / left 0.3633 / right 0.3627 (area and boundary-distance quartiles are in the artifact).

Competition diagnostics (GT offline only, target area-downsampled to 64×64 and thresholded at >0.5):

| Variant | target mass | argmax-in-target | reference mass | normalized entropy | top-1 | top-16 | top-64 |
|---|---|---|---|---|---|---|---|
| D-B1 | **0.0675** | 0.0083 | 0.0024 | **0.6933** | 0.0122 | **0.1670** | **0.4751** |
| D-B2 | 0.0067 | 0.0000 | 0.0862 | 0.8848 | 0.0105 | 0.0923 | 0.2202 |
| D-B3 | 0.0189 | 0.0208 | 0.0968 | 0.8844 | 0.0091 | 0.0876 | 0.2164 |
| D-B4 | 0.0051 | 0.0000 | 0.1040 | 0.8495 | 0.0146 | 0.1224 | 0.2800 |

Every `A` sums to exactly 1.000000. The learned competition maps are nearly uniform (normalized entropy
0.85-0.88 of the 4096-token maximum) and place almost no mass on the target (0.005-0.019), while the
deterministic field competition (D-B1) is markedly more concentrated (entropy 0.6933, top-64 mass 0.4751)
and localizes the target best — but still far below the predeclared target-mass bar.

## 7. Criteria and verdict (Parts M-N)

Section 28 primary D-B2 criteria — **4 of 11 pass**: overfit ✓, D-B2−D-B3 +0.1670 ≥ 0.08 ✓, paired 17/20
≥ 16 ✓, no target GT input ✓; failing: D-B2 mIoU **0.3265 < 0.38**, D-B2−D-B0 **+0.0023 < +0.05**,
D-B2−D-B1 **−0.0714 < +0.03**, D-B2−D-B4 **−0.0026 < +0.03**, margin **+0.2956 < 0.30**, target mass
**0.0067 < 0.10**, argmax-in-target **0.0000 < 0.45**.

Section 29 flags: `global_competition_mask_gain` **false**, `prototype_gain` **false**,
`field_guidance_gain` **true** (D-B2 − D-B3 = +0.1670 ≥ +0.05), `competition_localizes_target` **false**.

Section 30 priority: protocol clean, packs exact, D-B0 reproduced, D-B2 learnable; the visual-only condition
does not apply (D-B2−D-B3 = +0.1670 ≥ 0.08) and the prototype condition does not apply either
(D-B4−D-B0 = +0.0049 < 0.05), so the no-meaningful-gain condition applies →
**`GLOBAL_COMPETITION_NO_MEANINGFUL_GAIN`**.

Measured reading (reported only, no repair proposed): the *deterministic* field competition with a
prototype (D-B1) is the only variant that clearly beats the frozen Z-B3 baseline (+0.0737 mIoU, 19/20 paired,
margin +0.3888), while the *learned* relation-guided competition (D-B2) is statistically indistinguishable
from the baseline (+0.0023) and worse than both D-B1 and D-B4; its competition maps stay near-uniform and do
not localize the target, so the hypothesized learned global competition → prototype → mask chain is not
supported by this oracle-reference experiment.

## 8. Interpretation boundary

DSH reports measurements only. Global competition is **not** claimed as a novelty; no final architecture is
claimed; no predicted reference was integrated; parser/reference were not changed; no graph/attention module
was added; no supervision was added to `A`; no formal full-data training was started; the test split was not
accessed. Final recommendation exactly:

`等待 ChatGPT 根据 Task 7D 的 oracle-reference global competition 因果结果决定是否替换 Z-B3，不自行加入 attention/graph、predicted reference 或正式全量训练。`

## 9. Reproduce

```text
python scripts/task7d_train.py --stage baseline       # D-B0 exact reproduction (STOP on failure)
python scripts/task7d_train.py --stage d1             # Overfit20 + D-B2 learnability gate
python scripts/task7d_train.py --stage d2             # MiniTrain1200 -> MiniVal240
python scripts/task7d_evaluate.py                     # MiniVal240 + PairedVal20
python scripts/task7d_competition_diagnostics.py      # competition-map diagnostics
python scripts/task7d_report.py                       # criteria, flags, verdict
```

Checkpoints and caches live in the gitignored `artifacts/checkpoints/task7d/` and `artifacts/task7d/`; the
frozen SAM2 feature cache is reused from `artifacts/task6n/features`. Run in
`.conda/buildreasonseg-proposal` with `HF_HUB_OFFLINE=1`; the report step also runs in
`.conda/buildreasonseg-mvp`.
