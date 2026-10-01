# Task 7H — Development Architecture Freeze: BuildReasonSeg-DevFreeze-2026-10

> Task: `handoff/TO_DSH.md` (Task 7H) · Base commit: `22401fe` · Predecessor: Task 7G →
> `LARGEST_SELECTOR_NOT_LEARNABLE`
> **Verdict: `DEVELOPMENT_ARCHITECTURE_FROZEN`** (10/10 section-22 conditions) · **no training, no test access**
> Registries: `evaluation/task7h_evidence_registry.json`, `..._limitation_registry.json`,
> `..._formal_experiment_protocol.json`, `..._test_lock.json`, `..._claim_registry.json`
> Formal protocol: `docs/task7h_formal_experiment_protocol.md`

`BuildReasonSeg-DevFreeze-2026-10` is the frozen **development** architecture of the current project version.
It is **not** a new checkpoint, **not** the final paper model, **not** an unrestricted natural-language system
and **not** an end-to-end solved system.

## 1. Recorded Task 7G result

Verdict `LARGEST_SELECTOR_NOT_LEARNABLE`.

```text
G-I0 deterministic:  mean selected reference IoU 0.5199913587
                     oracle-best exact top-1     0.4887892377
                     mean best-minus-selected    0.2789719922

G-I1 learned:        mean selected reference IoU 0.6198877726
                     median selected IoU         0.7895902547
                     Pr(selected IoU >= 0.50)    0.7309417040
                     oracle-best exact top-1     0.6591928251
                     mean best-minus-selected    0.1790755783
                     mean IoU gain over G-I0    +0.0998964138
```

Predeclared internal gate: gain ≥ +0.08 **PASS** · mean selected IoU ≥ 0.62 **FAIL** (0.6198878) · oracle
top-1 ≥ 0.55 **PASS** · mean gap ≤ 0.14 **FAIL** (0.1790756).

Consequences: the external E-HoldoutL3 stage **did not run**; the Task 7G selector is **not adopted**; **no
scene-disjoint result may be claimed** for it; reference intervention stops.

**Correct interpretation.** The Task 7G selector showed a meaningful **internal** improvement but did not clear
the predeclared internal learnability gate. Calling it a scene-disjoint failure would be scientifically
incorrect, because the external scene-disjoint stage was never executed. The development system therefore keeps
the deterministic selector — not because it is best in principle, but because **no learned replacement has
passed the frozen adoption protocol**.

## 2. Evidence registry (summary)

Full machine-readable registry: `evaluation/task7h_evidence_registry.json`.

| Module | Role | Origin | SHA256 | Frozen |
|---|---|---|---|---|
| WHU-EA-NativeVector v1.0 | source vector dataset | 6M-era | identity + split metadata | ✓ |
| BuildSpatialReason v0.2 (`scene_disjoint_v1`) | canonical reasoning dataset | 6L-era | train/val verified (test not read) | ✓ |
| YOLO26m-seg Task 6M.1 | proposal generator (U-C1) | 6M.1 | `ef852b58…61f474` ✓ | ✓ |
| Task 7C ProgramHead | controlled-language 20-class parser | 7C | `c1505736…d58d9a` ✓ | ✓ |
| GeometricRelationField v0.2 | directional field (P_dir) | 6P | module hash | ✓ |
| NearestBoundaryField v0.1 | nearest field (P_near) | 6Y | module hash | ✓ |
| Task 6O N-B3 | directional-field visual segmentation component | 6O | `7556e4a4…c7d6ab` ✓ | ✓ |
| Task 6Z Z-B3 | L3 baseline/ablation | 6Z | `74f308e1…fc0f0bc` ✓ | ✓ |
| Task 7D D-B1 | **preferred** L3 target decoder | 7D | `6df31909…21a89c0` ✓ | ✓ |

D-B1 evidence (Task 7E untouched oracle-reference holdout, 669 records): Z-B3 mIoU `0.3141113773`, D-B1
`0.3854957053`, delta `+0.0713843280`, bootstrap 95 % CI `[+0.0586233648, +0.0842626436]`, paired `18/20`,
margin `+0.3193402994`. Practical predicted-reference chain: D-B1 strict `0.2454050104`, limitation attributed
mainly to reference selection.

Frozen **negative** evidence (must never enter the development chain): Task 6P ReferenceMaskHead
(`REFERENCE_HEAD_INSUFFICIENT`), Task 6U ProposalSetRanker v0.1 (`REFERENCE_RANKER_NOT_HELPFUL`), Task 6W
ProposalQualityEstimator v0.1 (`QUALITY_FILTER_NOT_HELPFUL`), Task 6X SAM2 refinement
(`SAM2_REFINEMENT_NOT_HELPFUL`), Task 7G SetContextLargestSelector v1 (`LARGEST_SELECTOR_NOT_LEARNABLE`),
Task 7D D-B2 learned global competition (map did not localize the target: mass 0.0067, argmax-in-target 0.0000,
entropy 0.8848).

## 3. Frozen development architecture

```text
controlled instruction
→ Task 7C Qwen3-VL-2B text-only ProgramHead (20 canonical programs)
→ U-C1 YOLO26m-seg proposals (imgsz 640, conf 0.05, max_det 300, default NMS, no TTA, no tiling)
→ deterministic largest reference selector (max predicted mask area → higher confidence → lower index)
→ GeometricRelationField v0.2 (P_dir) + NearestBoundaryField v0.1 (P_near)
→ Task 7D D-B1 target decoder
→ target mask
```

**Front end.** `user instruction → Task 7C ProgramHead → one of 20 canonical spatial programs`. Verified:
canonical/template-like 20-class program classification. **Not** verified: robust unrestricted free-form
language. Documentation must therefore say **canonical-program / controlled-language development interface**,
never "arbitrary natural-language understanding", and must not hide the Task 7C fixed24 `5/24` / compact `0/8` /
stress `0.6667` weakness.

**Reference stage.** Frozen proposal generator as above; the deterministic `largest` rule uses the Task 6Q/6U
family eligibility (non-empty, not border-touching, bbox extent ratio ≤ 0.20) with maximum predicted mask area,
tie higher confidence, then lower original index; the `smallest` family keeps the exact Task 6Q eligibility and
ranking for the supported L2 path. The Task 7G selector is **not** part of the chain.

**Relation representation.** The reference mask feeds `GeometricRelationField v0.2` for directional relations
and `NearestBoundaryField v0.1` for the nearest relation. For L3 both fields are used **separately**; the
deterministic product `W = clamp(P_dir * P_near)` is used only for D-B1 prototype weighting as defined by Task
7D, and the two decoder-visible fields must never be replaced by the product alone.

**Directional L2 path (frozen).** `reference + directional field + frozen SAM2 dense visual feature + relation
embedding → Task 6O N-B3`. Evidence boundary: field-guided visual segmentation is supported; universal relation
reasoning is not claimed; N-B2/N-B4 must not be substituted for N-B3.

**Nearest-only L2 status:** `experimental/limited`. Task 6Y B2 verified the field semantics and measured a clear
gain over the visual baseline, but failed the absolute/generalization/counterfactual feasibility gate
(`NEAREST_FIELD_NO_MEANINGFUL_GAIN`, MiniVal 0.2864, paired 6/20). It is not a headline final capability and must
not be silently promoted.

**L3 target path (frozen D-B1).**

```text
predicted/oracle reference
→ P_dir, P_near
→ W = clamp(P_dir * P_near, 0, 1)
→ A_fixed = W / (sum(W) + eps) ; A_fixed_vis = A_fixed * 4096
→ frozen SAM2 dense feature F
→ q = Σ A_fixed_i F_i
→ C_i = cosine(F_i, q)
→ decoder input: F + P_dir + P_near + direction embedding + A_fixed_vis + C
→ target mask
```

No learned competition head. D-B1's role is **preferred L3 target-decoder architecture candidate**. This must
not be called the final paper model, an unrestricted natural-language model or an end-to-end solved system.

## 4. Frozen limitations

Full registry: `evaluation/task7h_limitation_registry.json`.

| ID | Limitation | Status |
|---|---|---|
| L-01 | Reference selection (F-R0 0.245405 → F-R1 0.364909, gain +0.119504 = 85.3 % of the gap; Task 7G not adopted) | **unresolved practical bottleneck** |
| L-02 | Proposal coverage (best eligible coverage@0.50 = 0.846039; coverage gain +0.054316; 103 uncovered records) | **secondary unresolved bottleneck** |
| L-03 | Proposal-mask geometry (covered-subset gain +0.004253) | **not a major current bottleneck** |
| L-04 | Free-form L3 language (canonical 1.0 vs fixed24 5/24, compact 0/8, stress 0.6667) | **controlled-language interface only** |
| L-05 | nearest-only (Task 6Y B2 MiniVal 0.2864, paired 6/20, `NEAREST_FIELD_NO_MEANINGFUL_GAIN`) | **not validated as standalone final capability** |
| L-06 | unseen-city/domain generalization | **not established** — the native split demonstrates raster/scene separation only |

## 5. Formal experiment protocol (frozen, not executed)

`evaluation/task7h_formal_experiment_protocol.json` + `docs/task7h_formal_experiment_protocol.md`. Data policy:
train for gradients, val for checkpoint selection/early stopping, test only after every choice is frozen; the
historical Task 6M test access for the J4-v2 proposal-baseline audit is disclosed, so future reporting may only
say **`final frozen-architecture test evaluation`**, never "untouched test". Formal L3 population: the four L3
programs (train 1344 = 323/347/338/336, val 936). D-B1 retraining from fresh weights with the schedule copied
from `evaluation/task7d_training.json` (AdamW, lr 3e-4, wd 1e-4, batch 8, ≤25 epochs, patience 5, bf16 AMP,
selection by val mIoU) and formal seeds `20261001/20261002/20261003`. Practical and oracle reference results are
reported separately; Z-B3 is baseline B-L3-0 and D-B1 is B-L3-1 with the historical Task 6Z/7D controls kept as
ablations.

## 6. Test lock

```text
status                    = LOCKED
architecture_head         = D-B1
reference_policy          = U-C1 deterministic largest
parser_role               = controlled-language/canonical interface
formal_seeds              = [20261001, 20261002, 20261003]
test_execution_authorized = false
unlock_condition          = ChatGPT audit after formal train/val completion
```

Any future DSH task must read `evaluation/task7h_test_lock.json` before a test evaluation. Task 7H did **not**
unlock it.

## 7. Claim registry

`evaluation/task7h_claim_registry.json` — no novelty or "first" claim anywhere.

| Claim | Status |
|---|---|
| C1 explicit reference-conditioned directional fields improve same-class target segmentation | `SUPPORTED` |
| C2 directional + nearest priors compose for L3 direction→nearest reasoning | `SUPPORTED_WITH_LIMITATION` |
| C3 relation-conditioned deterministic weighting extracts a useful target prototype | `SUPPORTED` |
| C4 learned global competition is superior | `NOT_SUPPORTED` |
| C5 the practical system solves unrestricted natural-language L3 segmentation | `NOT_SUPPORTED` |
| C6 reference selection is the dominant practical bottleneck | `SUPPORTED_WITH_LIMITATION` |
| C7 proposal-mask geometry is the main reference problem | `NOT_SUPPORTED` |
| C8 performance generalizes to unseen cities | `NOT_SUPPORTED` |
| C9 D-B1 is the final paper-ready architecture | `NOT_SUPPORTED` |

## 8. Interpretation boundary

DSH reports and freezes only. No model was trained (nothing was trained in Task 7H); no test record was read,
hashed or evaluated; no threshold, field, loss, model structure or data split was changed; nothing was
downloaded or installed; the Task 7G selector was not run on E-Holdout after its STOP; no Task 7I
implementation detail beyond the frozen protocol was chosen; D-B1 is not called the final paper model;
unrestricted natural language, end-to-end success, novelty and unseen-city generalization are not claimed, and
the test split is never called untouched. Final recommendation exactly:

`等待 ChatGPT 审核 Task 7H 的开发版架构冻结与正式实验协议；在审核通过前不启动正式三种子训练，不解锁 test。`
