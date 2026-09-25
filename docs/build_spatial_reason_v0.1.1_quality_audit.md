# BuildSpatialReason v0.1.1 — Acceptance Quality Audit

**Verdict: `PASS`**

Audited by `spatial_reasoning/dataset_validator.py` via
`scripts/validate_build_spatial_reason.py --version v0.1.1`, over all **25,229**
records.

Machine-readable results: `evaluation/build_spatial_reason_v0.1.1_quality.json`
Visualization pack: `evaluation/build_spatial_reason_v0.1.1_samples/`

> v0.1 was **not** modified. Its five artefacts still hash to their recorded
> values (verified by `dataset_validator.verify_v01_unchanged`).

---

## 1. Blocking counts — all zero

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

`issues` is an empty dict: the audit found **no** error and **no** warning.

Semantic union counters (machine-readable, not prose-derived):

```
semantic_violation_unique_records = 0
semantic_clean_records            = 25229
```

---

## 2. Required positive checks

| Check | Result |
|---|---|
| Target recomputation | **25,229 / 25,229 = 100%** |
| `template_id` verification | **25,229 / 25,229 = 100%** |
| Type A + Type B = Level 2 total | **2,272 + 2,764 = 5,036** ✓ |
| Exact cross-split byte duplicates | **0** |
| zh/en semantic parity | **100%** (zero mismatches) |
| v0.1 unchanged | **True** |

---

## 3. Reasoning leakage

| Counter | Value |
|---|---|
| `leak_reasoning_records` | **0** |
| `leak_reasoning_fields` | **0** |
| `leak_reasoning_mentions` | **0** |
| `leak_instruction_records` | **0** |

In v0.1 this was 32,284 records / 159,014 mentions. Natural-language reasoning is
now fully role-based and ID-free, while `reasoning_steps` retains numeric ids for
machine verification.

---

## 4. Distractor policy

| Counter | Value |
|---|---|
| target in distractors | 0 |
| **reference in distractors** | **0** (v0.1: 12,918) |
| duplicate distractors | 0 |
| missing distractor components | 0 |

---

## 5. Scene-level leakage

`scene_level_split_leakage = unverified`

The derived YOLO dataset carries no scene/geographic grouping metadata, and val
was a random 20% subset of the original WHU train pool. Scene-level disjointness
therefore **cannot** be established from available metadata. Exact image
duplication **is** ruled out: **0** byte-identical images across splits (3,920
images hashed). WHU was **not** resplit in this task.

---

## 6. Level-3 trivial / nontrivial

| Subset | Count | Meaning |
|---|---|---|
| nontrivial | **1,657** | more than one admissible candidate after filtering → a real distance comparison is required |
| trivial | **1,261** | exactly one admissible candidate → the filter alone determines the answer |
| total | 2,918 | |

The `trivial_selection` flag was independently re-verified against
`semantic_policy.admissible_nearest_ids` for every record: **0** mismatches.

Trivial samples are **kept** and reported separately. The primary multi-hop
reasoning metric should use the **nontrivial** subset.

---

## 7. Scope

Established by this audit:

- program-internal consistency — **sound**;
- geometry truth of targets and masks — **sound**;
- natural-language semantic truth — **sound** (zero hidden-eligibility
  violations);
- split / duplication integrity — **sound**, scene level unverifiable;
- MLLM-supervision suitability — **suitable** (ID-free reasoning, explicit
  `template_id`, clean distractors).

Not established, and not claimed: whether the queries are pedagogically
sufficient for multi-hop spatial reasoning, and whether template-generated wording
is linguistically diverse enough for robust generalisation. `template_id` is now
recorded so the latter can be measured.
