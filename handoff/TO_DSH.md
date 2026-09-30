# TO_DSH — Task 7D: Oracle-Reference L3 Relation-Guided Global Competition Decoder

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Base commit: `632c9c02a373ba8eea7153aafad69281946bae26`
>
> Predecessor: Task 7C → `L3_20CLASS_COMPOSITIONAL_ROBUSTNESS_FAIL`
>
> Research decision already made by ChatGPT:
>
> 1. **Parser hardening stops here.** Task 7C is the final parser-specific experiment.
> 2. Keep Task 7C as the current development parser for canonical 20-class operation because it restores full canonical behaviour.
> 3. Record free-form L3 paraphrase robustness as an unresolved limitation.
> 4. Do NOT run another parser repair in Task 7D.
> 5. Reference hardening remains frozen. Do NOT reopen Tasks 6P–6X.
> 6. Task 6Z/7A show that the current shallow Z-B3 decoder is now a meaningful core bottleneck.
> 7. Task 7D tests one architecture-class pivot:
>
>    **Relation-guided global competition over dense frozen SAM2 tokens, followed by a dynamically selected
>    target visual prototype used for mask decoding.**
>
> 8. This task uses oracle reference and canonical L3 program ids so parser/reference errors do not contaminate the decoder question.
> 9. DSH is an executor. Do not alter architecture, formulas, losses, packs, thresholds, or next task.

All user-facing DSH output must be Chinese.

---

# 0. DSH role

DSH MAY:
- reuse exact Task 6Z packs and frozen SAM2 features;
- implement the exact global-competition decoder below;
- train/evaluate only the fixed variants D-B0 ... D-B4;
- compute the specified competition-map diagnostics;
- solve ordinary code/runtime bugs without changing the experiment.

DSH MUST NOT:
- retrain or modify ProgramHead;
- use natural-language parsing in the main Task 7D experiment;
- use predicted/proposal reference;
- modify GeometricRelationField v0.2;
- modify NearestBoundaryField v0.1;
- modify SAM2;
- introduce graph reasoning;
- introduce Transformer/self-attention/cross-attention modules;
- introduce target proposals/candidate masks as model inputs;
- add auxiliary attention/center/ranking loss;
- add GRCL;
- change BCE+Dice;
- tune softmax temperature;
- access test split;
- choose Task 7E.

If a prohibited change is required, STOP and report.

---

# PART A — Record Task 7C outcome and freeze parser status

## 1. Task 7C result

Copy into `docs/task7d_relation_guided_global_competition.md`:

Task 7C verdict:
`L3_20CLASS_COMPOSITIONAL_ROBUSTNESS_FAIL`

Canonical behaviour:
- full v0.2 val: 18,222/18,222 = 1.0000
- macro F1 = 1.0000
- every class recall = 1.0000
- Z-MiniVal240 = 240/240
- Z-Paired members = 40/40

Held-out natural-language robustness:
- Task 7A fixed24 = 5/24
- compact = 0/8
- minimal96 = 60/96
- stress accuracy = 0.6667
- stress macro F1 = 0.5924
- stress L3 macro recall = 0.4479

Task 7C checkpoint:
`artifacts/checkpoints/task7c/program_parser_l3_rehearsal_v1.pt`

SHA256:
`c150573613c421098f55776a4b2a26e1536b806dd5c210f716715fd2ded58d9a`

Task 7D does not use this parser for the causal experiment and does not retrain it.

## 2. Parser status after Task 7C

Record:
- canonical/program-template parser status: **frozen usable for development**
- free-form L3 paraphrase robustness: **known limitation / not solved**
- no further parser hardening task is authorized here.

Do not modify Task 7B/7C frozen results.

---

# PART B — Literature-position boundary

## 3. Fixed note for documentation

Copy this note into the Task 7D design document. DSH must not perform an independent literature search.

Current 2026 RRSIS literature already includes semantic-role decomposition, relation-aware graph reasoning, long-range/global dependency modeling and progressive mask refinement. SRGFormer (2026) uses semantic-role-guided graph reasoning and a global sink node to improve same-class instance discrimination.

Therefore Task 7D MUST NOT claim:
- global reasoning itself is novel;
- relation-aware attention itself is novel;
- long-range dependency modeling itself is novel.

The narrower experimental hypothesis is:

> **Explicit geometric relation fields can parameterize a differentiable global competition distribution over dense frozen visual tokens; the winning distribution can synthesize a target visual prototype, which then guides pixel-level mask decoding without target proposals or graph construction.**

This is an architecture hypothesis, not a novelty claim.

---

# PART C — Frozen Task 6Z causal setting

## 4. Exact program scope

Use only:
- `largest_to_left_of_to_nearest`
- `largest_to_right_of_to_nearest`
- `largest_to_above_to_nearest`
- `largest_to_below_to_nearest`

Use canonical program ids directly.
No parser.

## 5. Exact reference source

Use:
`oracle_native_gt`

Reference:
- canonical largest reference from each frozen Task 6Z record.

No predicted reference.
No YOLO proposal input.

## 6. Reuse exact frozen packs

Byte-for-byte reuse:
- Task 6Z Z-Overfit20
- Task 6Z Z-MiniTrain1200
- Task 6Z Z-MiniVal240
- Task 6Z Z-PairedVal20

Verify hashes against:
`evaluation/task6z_pack_manifest.json`

If mismatch:
STOP `TASK6Z_PACK_MISMATCH`.

## 7. Frozen fields

Read-only:
- `buildreasonseg_mvp/geometric_relation_field_v02.py`
- `buildreasonseg_mvp/nearest_boundary_field.py`

Constants unchanged:
- direction alpha = 1.2
- direction tau = 0.04
- direction s_axis = 0.02
- direction s_margin = 0.02
- nearest sigma_diag = 0.05

Generate:
- `P_dir_64`
- `P_near_64`

## 8. Frozen visual representation

Use exactly:
- SAM2.1 Hiera Base+
- frozen `V ∈ R^(256×64×64)`
- same checkpoint/config/cache as Task 6Z
- no backbone training.

---

# PART D — Common feature definitions

## 9. Visual projection

For trainable D-B1 ... D-B4:

```text
F =
Conv1x1(256 → 128)
GroupNorm(8,128)
GELU
```

## 10. Direction embedding

Trainable:
- vocab size 4
- dim 16
- ids left_of/right_of/above/below

Broadcast over 64×64.

No nearest embedding.

## 11. Numerical constants

Freeze:
- `competition_temperature = 1.0`
- `eps = 1e-6`
- spatial tokens = 4096

No sweep.

---

# PART E — Global competition primitives

## 12. Learned score head

For learned-competition variants:

```text
Conv3x3(in_channels → 64, padding=1)
GroupNorm(8,64)
GELU
Conv1x1(64 → 1)
```

Output `S`.

Global competition:

```text
A_flat = softmax(S.flatten(2) / 1.0, dim=-1)
A = A_flat.reshape(B,1,64,64)
A_vis = A * 4096
```

Require finite values and spatial sum `1 ± 1e-6`.

Do not detach A.

## 13. Target visual prototype

Flatten `F → B×128×4096`.

```text
q = sum_i A_i * F_i
```

No stop-gradient.

## 14. Prototype similarity

```text
F_norm_i = F_i / (||F_i||_2 + eps)
q_norm   = q   / (||q||_2 + eps)
C_i = dot(F_norm_i, q_norm)
```

Reshape C to `B×1×64×64`.

No learned scale.
No sigmoid.
No threshold.

---

# PART F — Deterministic field-weighted prototype

## 15. D-B1 fixed competition

```text
W = clamp(P_dir_64 * P_near_64, 0, 1)
A_fixed = W / (sum_spatial(W) + eps)
A_fixed_vis = A_fixed * 4096
```

If field mass <= eps:
record `INVALID_FIELD_MASS` and STOP.

Use A_fixed to compute q and C exactly as sections 13–14.

No learned score head.

---

# PART G — Exact variants

## 16. D-B0 — frozen Z-B3 baseline

Do NOT retrain.

Reproduce frozen Task 6Z selected Z-B3.

Require:
- MiniVal mIoU delta <= 1e-6
- Dice delta <= 1e-6
- Paired exactly 15/20
- own-cross margin delta <= 1e-6.

Write:
`evaluation/task7d_baseline_reproduction.json`

If fail:
STOP `TASK6Z_BASELINE_REPRODUCTION_FAIL`.

## 17. D-B1 — deterministic field-weighted visual prototype

Decoder inputs:

```text
F_128
P_dir
P_near
direction_embed_16
A_fixed_vis
C_fixed
```

Total 148 channels.

## 18. D-B2 — learned relation-guided global competition + prototype

PRIMARY.

Score-head inputs:

```text
F_128
P_dir
P_near
direction_embed_16
```

146 channels.

Decoder inputs:

```text
F_128
P_dir
P_near
direction_embed_16
A_vis
C
```

148 channels.

## 19. D-B3 — learned visual-only global competition

Score-head inputs:
```text
F_128
direction_embed_16
```
144 channels.

Decoder inputs:
```text
F_128
direction_embed_16
A_vis
C
```
146 channels.

No relation fields anywhere.

## 20. D-B4 — relation-guided competition map, no prototype

Score-head inputs same as D-B2.

Compute A and A_vis.

Do NOT compute/use q or C.

Decoder inputs:
```text
F_128
P_dir
P_near
direction_embed_16
A_vis
```
147 channels.

No other trainable variant.

---

# PART H — Common mask decoder

## 21. Decoder trunk

D-B1/D-B2/D-B3/D-B4:

```text
Conv3x3(in_channels → 128, padding=1)
GroupNorm(8,128)
GELU
Conv3x3(128 → 64, padding=1)
GroupNorm(8,64)
GELU
Conv1x1(64 → 1)
```

Upsample logits:
- bilinear
- 64→512
- align_corners=False.

Loss exactly:

`BCEWithLogitsLoss + DiceLoss`

No competition supervision.
No center/attention/ranking loss.
No contrastive loss.
No GRCL.
No auxiliary decoder.

Report exact parameter counts.

---

# PART I — Competition diagnostics

## 22. Offline evaluation only

For D-B1/D-B2/D-B3/D-B4, GT target is evaluation-only.

At 64×64, downsample GT target with area interpolation, then threshold `>0.5`.

Report:

1. argmax(A) inside target rate
2. target mass = sum A over target cells
3. reference mass = sum A over reference cells
4. normalized entropy:
   `-sum(A log(A+eps))/log(4096)`
5. top-1, top-16, top-64 spatial mass

Write:
`evaluation/task7d_competition_diagnostics.json`

Do not tune from these metrics.

---

# PART J — Stage D1: Overfit20

## 23. Training

Train D-B1/D-B2/D-B3/D-B4 separately, fresh initialization, exact Z-Overfit20.

- AdamW
- lr 1e-3
- weight_decay 1e-4
- batch 4
- max steps 1200
- no scheduler
- no augmentation
- seed 20261001
- same bfloat16 AMP as Task 6Z
- eval every 100.

Record:
- mIoU
- Dice
- Pr@0.5
- competition diagnostics
- params
- wall time
- peak VRAM.

Write:
`evaluation/task7d_overfit20.json`

## 24. D-B2 learnability gate

Require:
- mIoU >= 0.85
- Dice >= 0.90

If fail:
STOP `GLOBAL_COMPETITION_NOT_LEARNABLE`.

---

# PART K — Stage D2: MiniTrain1200 → MiniVal240

## 25. Training

Only if D-B2 overfit passes.

Train D-B1/D-B2/D-B3/D-B4 from fresh initialization on exact Z-MiniTrain1200.

- AdamW
- lr 3e-4
- weight_decay 1e-4
- batch 8
- max epochs 25
- early stopping patience 5
- checkpoint selection = MiniVal240 mIoU
- seed 20261001
- no scheduler
- no augmentation
- same AMP.

No test.

Write:
`evaluation/task7d_training.json`

---

# PART L — Evaluation

## 26. MiniVal240

Report D-B0...D-B4:
- mIoU
- Dice
- Pr@0.5
- per direction mIoU
- target area quartiles
- target boundary-distance quartiles
- params
- best epoch
- wall time
- peak VRAM.

Also include competition diagnostics for D-B1...D-B4.

Write:
`evaluation/task7d_mini_val.json`

## 27. PairedVal20

Use exact Z-PairedVal20.

Report:
- pass /20
- mean own IoU
- mean cross IoU
- own-cross margin.

Write:
`evaluation/task7d_paired_val.json`

---

# PART M — Predeclared criteria

## 28. Primary D-B2 feasibility

Require ALL:

1. D-B2 Overfit passes
2. D-B2 MiniVal mIoU >= 0.38
3. D-B2 - D-B0 >= +0.05
4. D-B2 - D-B1 >= +0.03
5. D-B2 - D-B3 >= +0.08
6. D-B2 - D-B4 >= +0.03
7. D-B2 Paired >= 16/20
8. D-B2 own-cross margin >= 0.30
9. D-B2 mean target competition mass >= 0.10
10. D-B2 argmax-in-target rate >= 0.45
11. no target GT input.

## 29. Diagnostic flags

Always report:

`global_competition_mask_gain = true`
iff D-B2 >= D-B0 + 0.03

`prototype_gain = true`
iff D-B2 >= D-B4 + 0.02

`field_guidance_gain = true`
iff D-B2 >= D-B3 + 0.05

`competition_localizes_target = true`
iff target mass >=0.08 AND argmax-in-target >=0.35

---

# PART N — Verdict

## 30. Exactly one

Priority:

1. `INVALID_EXPERIMENT`
2. `TASK6Z_PACK_MISMATCH`
3. `TASK6Z_BASELINE_REPRODUCTION_FAIL`
4. `GLOBAL_COMPETITION_NOT_LEARNABLE`
5. `GLOBAL_COMPETITION_VISUAL_ONLY`
   - D-B2-D-B3 <0.08 and D-B3 >= D-B0+0.05
6. `GLOBAL_COMPETITION_PROTOTYPE_NOT_HELPFUL`
   - D-B2-D-B4 <0.03 and D-B4 >= D-B0+0.05
7. `GLOBAL_COMPETITION_NO_MEANINGFUL_GAIN`
   - D-B2 mIoU <0.38 OR D-B2-D-B0 <0.05 OR D-B2-D-B1 <0.03,
     and verdict 5/6 does not apply
8. `GLOBAL_COMPETITION_COUNTERFACTUAL_WEAK`
   - mask/ablation criteria pass but paired/margin fail
9. `RELATION_GUIDED_GLOBAL_COMPETITION_FEASIBLE`
   - all section-28 criteria pass

No other verdict.

---

# PART O — Interpretation boundary

DSH may report measurements only.

Do NOT:
- claim global competition as novelty;
- claim final architecture;
- integrate predicted reference;
- change parser/reference;
- add graph/attention;
- add supervision to A;
- start formal full-data training;
- access test.

Final recommendation exactly:

`等待 ChatGPT 根据 Task 7D 的 oracle-reference global competition 因果结果决定是否替换 Z-B3，不自行加入 attention/graph、predicted reference 或正式全量训练。`

---

# PART P — Required artifacts

Create:

```text
buildreasonseg_mvp/task7d_global_competition_decoder.py

evaluation/task7d_baseline_reproduction.json
evaluation/task7d_overfit20.json
evaluation/task7d_training.json
evaluation/task7d_mini_val.json
evaluation/task7d_paired_val.json
evaluation/task7d_competition_diagnostics.json
evaluation/task7d_verdict.json

docs/task7d_relation_guided_global_competition.md

scripts/task7d_train.py
scripts/task7d_evaluate.py
scripts/task7d_competition_diagnostics.py
scripts/task7d_report.py
```

Checkpoints/caches:
`artifacts/checkpoints/task7d/`
`artifacts/task7d/`
gitignored.

Update:
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

No parser/reference checkpoint change.

---

# PART Q — Tests

Task 7C ended at:
`1189 passed, 1 skipped`

Add tests for at least:

1. Task 7C artifacts unchanged
2. parser not trained in Task 7D
3. exactly four L3 programs
4. exact Task 6Z packs reused
5. pack hashes verified
6. no test split
7. oracle reference explicitly recorded
8. no predicted/proposal reference
9. directional field v0.2 unchanged
10. nearest field v0.1 unchanged
11. SAM2 feature path unchanged
12. visual projection exact 256→128
13. direction embedding vocab 4
14. embedding dim 16
15. competition temperature exactly 1.0
16. softmax global over 4096
17. competition sum = 1
18. A_vis = A*4096
19. prototype weighted sum exact
20. prototype not detached
21. cosine similarity exact
22. no learned similarity scale
23. D-B0 frozen reproduction
24. D-B1 product competition exact
25. D-B1 no learned score head
26. D-B2 exact score inputs
27. D-B2 exact decoder inputs
28. D-B3 has no relation fields
29. D-B4 has no prototype/similarity
30. exactly four trainable variants
31. no attention/Transformer/GNN
32. no target proposal/candidate-mask input
33. common decoder trunk exact
34. BCE+Dice only
35. no GRCL
36. no competition auxiliary loss
37. GT target only label/evaluation
38. competition diagnostics GT offline only
39. same schedule across trainable variants
40. Overfit gate exact
41. MiniVal gates exact
42. PairedVal exact reuse
43. no ProgramHead change
44. no reference resolver change
45. no new dataset/download/install/GUI
46. previous suite preserved.

Run:
`python -m pytest tests/ -q`

Do not reduce previous passing tests.

---

# PART R — Git/storage

Do not commit:
- Task 7D checkpoints;
- parser/YOLO/SAM2/Z-B3 weights;
- feature caches;
- source imagery/vectors;
- `.conda`.

Commit:
- decoder code;
- small JSON evaluations;
- scripts;
- tests;
- docs;
- handoff.

Suggested commits:
1. `feat: add relation-guided global competition decoder`
2. `eval: test global target competition causally`
3. optional docs/handoff commit

---

# PART S — DSH model policy

Default:
- **DeepSeek V4.1 Flash + High**

Use Max only for genuine runtime/cross-file/CUDA bugs.

No installs or downloads.

---

# PART T — STOP

After Task 7D:
- commit;
- push;
- update handoff;
- STOP.

Do NOT:
- run another parser experiment;
- reopen reference hardening;
- add attention/graph modules;
- integrate predicted reference;
- start formal full training;
- use test;
- build GUI.

Wait for ChatGPT audit.
