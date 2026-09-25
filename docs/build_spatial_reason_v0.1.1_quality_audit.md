# BuildSpatialReason v0.1.1 — Acceptance Quality Audit

**Verdict: `PASS`**

Audited by `spatial_reasoning/dataset_validator.py` via
`scripts/validate_build_spatial_reason.py --version v0.1.1`, over all **25,229**
records, **plus** an independent semantic oracle
(`spatial_reasoning/semantic_oracle.py`) that never imports
`semantic_policy.py`.

Machine-readable results: `evaluation/build_spatial_reason_v0.1.1_quality.json`
Visualization pack: `evaluation/build_spatial_reason_v0.1.1_samples/`
Consistency gate: `evaluation/build_spatial_reason_v0.1.1_consistency.json`

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

> v0.1 and v0.1.1 JSONL records were **not** modified. Both versions still hash
> to their recorded values (`verify_v01_unchanged`, `verify_v011_unchanged`).

---

## 1. Independent oracle — the non-circular check

The generator, the target recomputation and the validator all call one shared
module, `spatial_reasoning/semantic_policy.py`. Consistency between them is not
evidence of correctness: a wrong shared definition is wrong three times.

`spatial_reasoning/semantic_oracle.py` is a **second, independent
implementation** restricted to auditing and testing. It never imports or calls
`semantic_policy` (enforced statically by an AST scan and at runtime by a
`sys.meta_path` poison) and re-derives every semantic quantity from raw component
geometry, the frozen relation config, the frozen direction predicate and its own
recomputation of the component quality flags.

| Oracle metric | Value |
|---|---|
| records checked | **25,229 / 25,229** |
| exact semantic target matches | **25,229** |
| target mismatches | **0** |
| candidate-set mismatches | **0** |
| reference mismatches | **0** |
| ambiguity / admissibility mismatches | **0** |
| Level-3 trivial-flag agreements | **2,918 / 2,918** |
| independent semantic violations | **0** |

Independent hidden-eligibility counts (oracle-derived, not policy-derived), all
required to be zero:

```
hidden_eligibility_largest          = 0
hidden_eligibility_smallest         = 0
hidden_eligibility_nearest          = 0
hidden_eligibility_level3_nearest   = 0
semantic_violation_flag_instances   = 0
semantic_violation_unique_records   = 0
```

---

## 2. Production-policy audit — all blocking counts zero

| Check | Count |
|---|---|
| target recomputation failures | **0** |
| `hidden_eligibility_largest` | **0** |
| `hidden_eligibility_smallest` | **0** |
| `hidden_eligibility_nearest` | **0** |
| `hidden_eligibility_level3_nearest` | **0** |
| instruction semantic mismatch | **0** |
| reasoning component-ID leakage | **0** |
| target in distractors | **0** |
| reference in distractors | **0** |
| missing / empty target mask | **0** |
| `template_id` mismatch | **0** |
| duplicate sample IDs | **0** |
| structured-program errors | **0** |
| artifact-consistency violations | **0** |

`issues` is an empty dict: the audit found **no** error and **no** warning.

```
semantic_violation_unique_records = 0
semantic_clean_records            = 25229
```

---

## 3. Required positive checks

| Check | Result |
|---|---|
| Target recomputation (production policy) | **25,229 / 25,229 = 100%** |
| Target recomputation (independent oracle) | **25,229 / 25,229 = 100%** |
| `template_id` verification | **25,229 / 25,229 = 100%** |
| Type A + Type B = Level 2 total | **2,275 + 2,761 = 5,036** ✓ |
| Level-3 trivial + nontrivial = Level 3 | **1,256 + 1,662 = 2,918** ✓ |
| Exact cross-split byte duplicates | **0** |
| zh/en semantic parity | **100%** (zero mismatches) |
| v0.1 unchanged | **True** |
| v0.1.1 unchanged | **True** |

---

## 4. Frozen data integrity

| File | SHA256 | Recorded | Match |
|---|---|---|---|
| `v0.1.1/train.jsonl` | `85e3ae16…bb594eee` | same | ✓ |
| `v0.1.1/val.jsonl` | `c561d747…cd246d55` | same | ✓ |
| `v0.1.1/test.jsonl` | `6f7e9525…16d1c5b29d` | same | ✓ |
| `v0.1/train.jsonl` | `5848157b…b2cc54975` | same | ✓ |
| `v0.1/val.jsonl` | `93aea13d…d3c3c163c6` | same | ✓ |
| `v0.1/test.jsonl` | `a0b6bfe4…5d218255cc` | same | ✓ |

A change to any v0.1.1 JSONL hash would fail this task outright
(`v011_modified` is blocking). None changed.

---

## 5. Artifact consistency gate (Task 5C §7)

`scripts/check_artifact_consistency.py` verifies two-way agreement between:

`evaluation/build_spatial_reason_artifact_index.json` ↔ `manifest.json` ↔
`statistics.json` ↔ the quality JSON ↔ every Markdown file carrying an
`ARTIFACT-FACTS` block.

Status: **`consistent`**, 0 violations.

What it caught during Task 5C: the Task 5B handoff documents published Level-2
Type A/B as **2,272 / 2,764** and Level-3 trivial/nontrivial as
**1,261 / 1,657**, while `statistics.json`, `manifest.json` and the quality JSON
all said **2,275 / 2,761** and **1,256 / 1,662**. A direct recount from the JSONL
confirmed the machine-readable values; the handoff text was wrong, the dataset was
not. Those stale numbers are corrected in `handoff/FROM_DSH.md` and
`handoff/PROJECT_STATE.md`, and the gate now prevents recurrence.

Canonical paths after Task 5C (the historical `v011` abbreviation is removed):

```
evaluation/build_spatial_reason_v0.1.1_quality.json
evaluation/build_spatial_reason_v0.1.1_samples/
evaluation/build_spatial_reason_v0.1.1_consistency.json
docs/build_spatial_reason_v0.1.1_quality_audit.md
```

---

## 6. Reasoning leakage

| Counter | Value |
|---|---|
| `leak_reasoning_records` | **0** |
| `leak_reasoning_fields` | **0** |
| `leak_reasoning_mentions` | **0** |
| `leak_instruction_records` | **0** |

In v0.1 this was 32,284 records / 159,014 mentions.

---

## 7. Distractor policy

| Counter | Value |
|---|---|
| target in distractors | 0 |
| **reference in distractors** | **0** (v0.1: 12,918) |
| duplicate distractors | 0 |
| missing distractor components | 0 |

---

## 8. Scene-level leakage

`scene_level_split_leakage = unverified`

The derived YOLO dataset carries no scene/geographic grouping metadata, and val
was a random 20% subset of the original WHU train pool. Scene-level disjointness
therefore **cannot** be established from available metadata. Exact image
duplication **is** ruled out: **0** byte-identical images across splits (3,920
images hashed). WHU was **not** resplit in this task.

---

## 9. Level-3 trivial / nontrivial

| Subset | Count | Meaning |
|---|---|---|
| nontrivial | **1,662** | more than one admissible candidate after filtering → a real distance comparison is required |
| trivial | **1,256** | exactly one admissible candidate → the filter alone determines the answer |
| total | 2,918 | |

The `trivial_selection` flag was independently re-verified twice: against
`semantic_policy.admissible_nearest_ids` (production, 0 mismatches) and against
`semantic_oracle.admissible_nearest_ids` (2,918/2,918 agreements).

Trivial samples are **kept** and reported separately. The primary multi-hop
reasoning metric should use the **nontrivial** subset.

---

## 10. Scope

Established by this audit:

- program-internal consistency — **sound**;
- geometry truth of targets and masks — **sound**;
- natural-language semantic truth — **sound**, confirmed by an independent
  implementation rather than by the shared production policy;
- split / duplication integrity — **sound**, scene level unverifiable;
- documentation ↔ machine-readable agreement — **enforced by a gate**;
- MLLM-supervision suitability — **suitable**.

Not established, and not claimed: whether the queries are pedagogically
sufficient for multi-hop spatial reasoning, and whether template-generated wording
is linguistically diverse enough for robust generalisation. `template_id` is
recorded so the latter can be measured.
