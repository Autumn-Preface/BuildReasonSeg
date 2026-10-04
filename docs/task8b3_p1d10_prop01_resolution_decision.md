# Task 8B.3-P1D10 — PROP-01 Resolution Decision Audit (docs only)

## 1. Scope / starting HEAD

Docs-only resolution audit for `RC1-DEMO-PROP-01`; **no model or test execution and no functional modification**.

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD   = 70e2e2361e9e16cca891138de5d2712318282b2b
model runs / pytest / predict / training / inference = NONE
functional files modified = NO
```

## 2. Frozen P1D1–P1D9 evidence table

| task | probe | frozen result |
|---|---|---|
| P1D1 | historical A2 artifact forensics | A2 `raw=0`, `merged=0`, `E401`; one NMS time-limit warning; conclusion superseded to `PROP01_INSUFFICIENT_EVIDENCE` (masks-None is an independent empty path) |
| P1D2 | active YOLO26m-seg, 9 frozen tiles | 9/9 tiles `boxes_count = 0`, wrapper output 0; `PROP01_MODEL_ZERO_AT_FROZEN_CONF_CONFIRMED` |
| P1D3 | active, full-frame 1024×1024 | `boxes_count = 0`; `PROP01_GLOBAL_ZERO_AT_FROZEN_CONF_CONFIRMED` (tiling is not the cause) |
| P1D4 | active, 9 tiles, diagnostic `conf = 0.001` | all threshold counts 0, `global_max_confidence = None`; `PROP01_NO_MEANINGFUL_SUBTHRESHOLD_SIGNAL` |
| P1D5 / P1D5-R1 | detector provenance / A2 domain audit | active = YOLO26m-seg `ef852b58…` = documented Task 6M1 `best.pt`; 1 class `building`; A2 provenance NOT ESTABLISHED; alternates exist but none validated on A2/RC1; detector swap touches frozen claims; `PROP01_DOMAIN_GAP_INSUFFICIENT_EVIDENCE` |
| P1D6 | epoch-18 YOLO26m-seg, 9 tiles | 9/9 `boxes_count = 0`; `PROP01_EPOCH18_ALSO_ZERO` (not a continuation regression) |
| P1D7 | independent YOLOv8m-seg WHU baseline, pinned runtime, 9 tiles | 9/9 `boxes_count = 0`; `PROP01_YOLOV8M_ALSO_ZERO` (cross-lineage) |
| P1D8 / P1D8-R1 | non-model input-domain audit over all 17 388 training images | A2 not a strong photometric outlier (combined 4/4 core metrics ≥ p05); exact-hash match absent; no PNG encoding anomaly; split robustness `SENSITIVE_TO_SPLIT_COMPOSITION` |
| P1D9 | pre-declared validation-moment affine transform (`alpha = 4.259966488571338`, `beta = −188.2294346811672`), active detector, 9 tiles | transformed `Y_std` 26.8645, `Y_dynamic_98` 133.1520, clipping 0.2070 % → still 9/9 `boxes_count = 0`; `PROP01_VALIDATION_MOMENT_RESCUE_REMAINS_ZERO` |

Net picture: A2 yields zero post-NMS proposals across two independent detector lineages, at the product threshold and at
`conf = 0.001`, with tiling and full-frame context, and after a strong validation-moment photometric normalisation;
A2 is not a strong photometric outlier in the combined training distribution, and its own provenance is unknown.

## 3. Demo-suite provenance / policy

```text
Demo suite definition      : frozen six-image suite (A1–A4 PNG 1024×1024, B1/B2 TIFF 5000×5000), byte-frozen hashes
R4B recorded outcome       : language 6/6 · runtime 3/6 · manual end-to-end semantic 0/6 (qualitative six-sample audit)
A1–A4 input provenance     : NOT ESTABLISHED (only size/hash/format recorded)
B1/B2 input provenance     : NOT ESTABLISHED
policy documentation       : PARTIAL — suite mechanics and results are documented; the input domain is not
```

## 4. Scientific freeze vs RC1 engineering separation

```text
frozen research side : final architecture in model metadata (detector = "YOLO26m-seg (frozen U-C1 proposal model)",
                       decoder D-B1 seed 20261003, SAM2.1 Hiera Base+, Qwen3-VL-2B-Instruct + Task 7C ProgramHead);
                       research baseline 6c2b915dbc64acdeb099d194005d74c7180c95fa with task7j_test_status = FINAL_TEST_CONSUMED
RC1 delivery side    : runnable package, CLI, tiling orchestration, diagnostics, manifest 135/135
rule                 : policy/scope statements change no architecture; detector substitution would change the proposal
                       source that the frozen results were produced with and therefore touches frozen claims
```

## 5. Six intervention impact classifications (exact enums)

| # | intervention | impact enum |
|---|---|---|
| 1 | retrain / fine-tune current YOLO26 detector | `TOUCHES_FROZEN_RESEARCH_ARCHITECTURE` (the frozen checkpoint identity is part of the recorded architecture) |
| 2 | replace YOLO26 with another trained detector | `TOUCHES_FROZEN_RESEARCH_ARCHITECTURE` |
| 3 | add a second proposal-source fallback | `ENGINEERING_ONLY_IF_SEPARATELY_VERSIONED_AND_REVALIDATED` |
| 4 | add image preprocessing before the detector | `ENGINEERING_ONLY_IF_SEPARATELY_VERSIONED_AND_REVALIDATED` (condition: separately versioned; P1D9 shows it does not recover A2) |
| 5 | keep detector, define explicit supported-input/domain policy | `POLICY_ONLY_NO_ARCHITECTURE_CHANGE` |
| 6 | replace A2 in a future Demo suite, keeping it as documented stress case | `POLICY_ONLY_NO_ARCHITECTURE_CHANGE` (condition: selection rule independent of detector outcome) |

## 6. Adaptation feasibility audit

```text
A2 ground-truth building masks available?      NOT ESTABLISHED
A2-like labeled images available?              NOT ESTABLISHED (WHU data exists; "A2-like" domain unproven)
A2 source/domain provenance available?         NO
existing adaptation dataset specification?     NO
existing held-out adaptation validation set?   NO
existing acceptance metric/threshold?          NO
```

```text
DETECTOR_ADAPTATION_NOT_READY
```

## 7. Demo-policy feasibility audit

```text
existing scope already limits inputs to overhead/aerial building imagery?  YES (WHU-derived training lineage; RC1 docs)
WHU-like building instance domain?                                          YES (single class `building`)
image size/format expectations documented?                                  PARTIAL (PNG/TIFF accepted; 1024² and 5000² exercised)
A2 honestly retainable as documented unsupported/stress failure?             YES (it is already recorded as a failure,
                                                                             not counted as a success case)
future success case selectable from a predeclared pool without cherry-picking? YES (see §8)
```

```text
DEMO_POLICY_PATH_READY
```

## 8. Anti-cherry-picking replacement rule readiness

Declared **before** any replacement is selected:

```text
eligible source pool        : artifacts/task6m_yolo_native/images/val (frozen validation split, 3 618 images)
selection rule              : deterministic filename-ordered scan, take the first k images satisfying the declared
                              constraints; no iteration on success
required relation type      : one of the four frozen programs with a valid reference/target pair
required image constraints  : readable PNG/TIFF, ≥ 512×512, no padding-only content
required no-leakage condition: image must belong to the val split (never train), no A2/A1/A3/A4 reuse
required detector precheck  : declared in advance (format/size only); NOT used to accept or reject candidates
detector outcome consulted during selection : NO
```

```text
replacement selection policy = READY
```

Rule: the replacement case must not be selected by trying images until one succeeds; the rule above is
outcome-independent, so it satisfies that requirement.

## 9. Four-option comparison

| option | scientific honesty | engineering feasibility now | required new data/code | revalidation burden | overfitting/cherry-pick risk | Demo credibility |
|---|---|---|---|---|---|---|
| **A** immediate detector adaptation | honest only if the frozen claim basis is re-frozen alongside | NOT feasible now (no A2 GT, no adaptation spec, no held-out protocol) | labeled A2-like data + training + protocol | high (re-freeze + re-run frozen results) | high if tuned toward A2 | highest if it worked, but unverifiable now |
| **B** engineering fallback proposal source | honest only as a separately versioned fork | feasible in principle, but the fallback source itself must be validated and is unproven for A2 | fallback implementation + versioning + validation | medium (fork must not silently rewrite frozen results) | medium (could mask the defect) | medium |
| **C** supported-domain Demo policy | fully honest: documents the domain, keeps A2 as a stress case, changes no frozen result | feasible now — documentation/policy only | none beyond policy text + selection rule | none for frozen results | low if the selection rule is outcome-independent (§8) | credible if the limitation is stated plainly |
| **D** keep A2 as mandatory 6/6 and block release | maximally conservative | blocks release indefinitely; no path is currently available (P1D2–P1D9) | n/a | n/a | none | lowest (no Demo) |

## 10. Exact primary resolution

```text
PROP01_RESOLUTION_DEMO_POLICY
```

Criteria: detector adaptation is NOT READY (§6); the bounded diagnostics P1D2–P1D9 give strong evidence that A2 is a
persistent unsupported/blind-spot case rather than a tiling, threshold, lineage or photometric effect; a transparent
supported-domain/stress-case policy is feasible (§7); and no project requirement demands arbitrary-image robustness.

## 11. Exact PROP-01 defect status

```text
PROP01_RECLASSIFIED_SUPPORTED_DOMAIN_FAILURE
```

Reclassification is explicitly **not** closure: the defect record stays open, and implementing the policy still
requires a later task.

## 12. Rationale

The evidence chain shows that six independent attempts to explain A2's zero-proposal outcome through engineering,
model-lineage, geometric or photometric mechanisms all fail, while every other frozen sample (including the two
5000×5000 large images after the MEM-01 fix) produces proposals. The honest statement is therefore that the frozen
proposal detector's effective domain does not include A2, not that A2 reveals a bug with a known fix. Option C keeps
all frozen scientific results and the delivered detector untouched, documents the limitation, and reserves any future
success-case replacement to a pre-declared, outcome-independent rule.

## 13. Release-language draft (≤ 120 Chinese characters)

```text
RC1 的 proposal detector 在 WHU 类航空建筑影像上验证；A2 属其有效域外的压力/失败样例，已如实记录，不计入成功演示；本版本不承诺任意航空影像的鲁棒性。
```

The draft claims no arbitrary aerial-image robustness, does not claim A2 was fixed, and does not claim any frozen
result used a different detector.

## 14. Exact next gate (recommended — NOT executed)

```text
NEXT = PROP01_SUPPORTED_DOMAIN_POLICY_IMPLEMENTATION
```

## 15. No model execution

No detector, Qwen, SAM2, D-B1, `predict.py`, six-image suite, pytest or `check_setup.py` run occurred; no training,
fine-tuning, download or checkpoint access happened.

## 16. No functional modification

Only `docs/task8b3_p1d10_prop01_resolution_decision.md`, `handoff/FROM_DSH.md` and `handoff/TO_DSH.md` changed; no
runtime, test, manifest, configuration, checkpoint or external-delivery functional file was modified.

## 17. MEM-01 remains CLOSED

`RC1-DEMO-MEM-01 = CLOSED` (M1B.1–M1B.3-R1; real 5000×5000 B1/B2 runs succeeded with post-run integrity confirmed).

## 18. REF-01 / MASK-01 remain OPEN and untouched

`RC1-DEMO-REF-01 = OPEN` (user-facing "largest building" ≠ implementation "largest eligible detected proposal") and
`RC1-DEMO-MASK-01 = OPEN` (SUCCESS validity has no minimum-area/instance gate); neither was modified or repaired.


---

## 19. P1D10-R1 — corrected resolution logic and candidate-pool audit

Docs-only correction. No model, test, training or functional execution/modification occurred.

### 19.1 Corrected logic findings

Two errors in the original §8/§10/§11 were identified and are corrected here:

1. **Invalid pool choice.** The original replacement pool was `artifacts/task6m_yolo_native/images/val`, i.e. the
   active detector's own validation split. Per the audit rules a detector validation split must not be treated as
   clean merely because it is not training data, so that pool is a **model-selection leakage risk** and cannot ground
   the Demo selection contract.
2. **Unsupported reclassification.** The original §11 reclassified PROP-01 as a supported-domain failure. A2's
   provenance/domain is **NOT ESTABLISHED** (P1D5-R1, P1D8-R1), and unknown provenance is not proof of a domain
   violation; therefore the reclassification is withdrawn.

### 19.2 Candidate Demo-pool audit table (§5)

| pool | exists | frozen before P1D10 | image identity | Reference→Relation→Target metadata | program labels | reference/target instance identity | scene/split identity | detector training | detector validation / model selection | final research test | selectable without detector output | classification |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A `artifacts/task6m_yolo_native/images/val` (3 618 `.tif`) | YES | YES | YES | NO | NO | NO | YES (split dir) | NO | **YES** | NO | YES | **MODEL_SELECTION_LEAKAGE_RISK** |
| B `artifacts/task6m_yolo_native/images/test` (3 726 `.tif`) | YES | YES | YES | NO | NO | NO | YES (split dir) | NO | NO | NO | YES | **MISSING_RELATION_METADATA** |
| C `datasets/whu_native_vector/v1.0` (8 index/metadata files) | YES | YES | YES (`tile_id`, `source_image_ref`) | NO | NO | partial (`tile_instance_id`, `source_feature_id`) | YES (`scene_disjoint_split`, `legacy_compat_split`) | NO | NO | NO | YES | **MISSING_RELATION_METADATA** |
| D `artifacts/whu_native_vector/reasoning_view/scene_disjoint_v1` (17 392 files) | YES | YES | YES (component PNGs) | NOT ESTABLISHED in the audited sample | NOT ESTABLISHED | NOT ESTABLISHED | YES (scene-disjoint view) | NO | NOT ESTABLISHED | NOT ESTABLISHED | NOT ESTABLISHED | **POOL_STATUS_INCOMPLETE** |
| E1 `artifacts/task6m1_demo` (18 cases: `supported_2_*`, `ood_*`) | YES | YES | YES (`image`) | YES (`parsed_program`, `status`, `proposal_count`) | YES | supported cases record outputs/selection | YES (case ids; OOD cases separate) | NO | NO (post-selection showcase runs) | NO | YES | **USABLE_WITH_DISCLOSURE** |
| E2 `artifacts/task6m_demo` (13 cases) | YES | YES | YES (`image`) | YES (`parsed_program`, `selected`, `outputs`, `reasoning_zh`) | YES | YES | YES (case ids) | NO | NO | NO | YES | **USABLE_WITH_DISCLOSURE** |

Best Demo candidate pool: `artifacts/task6m1_demo` (18 cases; supported cases used, OOD cases retained as abstention
demonstrations), with `artifacts/task6m_demo` (13 cases) as secondary.

### 19.3 Replacement-selection contract (§7), declared before any selection

```text
exact pool                : artifacts/task6m1_demo (primary) — pre-existing, documented before PROP-01 diagnostics
exact split/version       : Task 6M1 demo bundle (paths recorded in each case result.json)
eligible program/relation set : the four frozen programs (largest_to_{left_of,right_of,above,below}_to_nearest)
required reference/target validity : case metadata must record parsed_program and a completed status
image-format/size rule    : the image recorded in the case metadata (readable PNG/TIFF)
scene-disjoint/no-train rule : demo cases are not detector training data and not the detector validation split
deterministic ordering    : lexical ordering of case directory names within the pool
number of cases k         : declared by the future task book (not selected here)
tie-breaking              : lexical order of the case id
whether model/detector outputs may be consulted = NO
whether manual visual quality may be consulted  = NO before selection
```

This contract is executable from metadata alone and is independent of detector outcome, so it satisfies the
anti-cherry-picking requirement. No image was selected in this task.

### 19.4 Re-evaluated Demo-policy feasibility (§8)

```text
DEMO_POLICY_PATH_READY
```

Reason: a positive support scope is already grounded in pre-existing provenance (the WHU-derived aerial building
instance domain the frozen detector was trained and validated on — 1 class `building`, `data.yaml` provenance), and a
replacement-selection contract exists that is executable without detector or manual outcome consultation (§19.3).

### 19.5 Re-evaluated primary resolution (§9) and PROP-01 status (§10)

```text
primary resolution = PROP01_RESOLUTION_DEMO_POLICY
PROP-01 status     = PROP01_OPEN_ENGINEERING_DEFECT
```

`DEMO_POLICY` is retained because the transparent support scope and the non-cherry-picked contract are both ready;
adaptation remains **DETECTOR_ADAPTATION_NOT_READY** (no A2 ground truth, no adaptation dataset specification, no held-out protocol, no
acceptance metric).

The defect status is corrected to **`PROP01_OPEN_ENGINEERING_DEFECT`**: A2's provenance is unknown, so no independently defined
supported-domain contract currently proves a domain violation, and unknown provenance is not proof. PROP-01 stays
open — reclassification would require such a contract, and in any case reclassification is not closure.

### 19.6 Corrected release wording (≤ 120 Chinese characters, §11)

```text
RC1 的 proposal detector 已在 WHU 类航空建筑实例域上完成验证；演示用例取自该域内既有资产，选择规则在检测结果之外预先确定；A2 为已如实记录的未检出样例，本版本不承诺任意航空影像的鲁棒性。
```

The wording states the verified support scope positively, calls A2 a documented non-detection case, avoids the
unsupported claim that A2 is outside the domain, and implies neither arbitrary-image robustness nor that A2 was fixed.

### 19.7 Scientific freeze (§12) and next gate (§13)

```text
scientific freeze preserved = YES
NEXT = PROP01_SUPPORTED_DOMAIN_POLICY_IMPLEMENTATION   (recommended, NOT executed)
```

Policy documentation and deterministic Demo-case selection remain separate from the frozen research metrics and
architecture; the frozen detector and all frozen results are untouched.
