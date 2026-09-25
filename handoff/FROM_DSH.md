# FROM_DSH — Task 5C Report: Acceptance Audit Hardening & Artifact Consistency

**Date:** 2026-08-04
**Actor:** DSH
**Task:** Task 5C (from `handoff/TO_DSH.md`)
**Verdict: `PASS`**

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

**`PASS`** — the v0.1.1 dataset is semantically correct under an **independent**
implementation, and every repository artifact now agrees with one declared truth.

| Gate | Result |
|---|---|
| Independent-oracle target match | **25,229 / 25,229** |
| Independent semantic violations | **0** |
| Production-policy hidden violations | **0** |
| Artifact consistency violations | **0** |
| v0.1 / v0.1.1 JSONL hashes | **unchanged** |
| Provenance digests vs on-disk files | **8 / 8 match** |
| Test suite | see §10 |

**No JSONL was regenerated or modified.** The only dataset-directory change is
`v0.1.1/manifest.json`, whose provenance list was extended as section 9 requires.

**Where the responsibility for the Task 5B count defect lies.** The Task 5B
handoff text published Level-2 Type A/B as 2,272/2,764 and Level-3
trivial/nontrivial as 1,261/1,657. `statistics.json`, `manifest.json` and the
quality JSON all said 2,275/2,761 and 1,256/1,662, and a direct recount from the
JSONL confirms those machine-readable values. **The dataset was always correct;
my Task 5B report was wrong.** It is corrected here, and the new consistency gate
makes the class of error detectable rather than silent.

---

## 2. Files Modified

**Created**

| File | Purpose |
|---|---|
| `spatial_reasoning/semantic_oracle.py` | Independent audit-only semantic implementation (never imports `semantic_policy`) |
| `scripts/check_artifact_consistency.py` | Repository-wide artifact consistency gate |
| `tests/test_semantic_oracle.py` | 8 oracle checks (independence, semantics, full-dataset, frozen hashes) |
| `tests/test_artifact_consistency.py` | 8 consistency-gate checks |
| `evaluation/build_spatial_reason_artifact_index.json` | Declared single source of truth for counts/versions/paths |
| `evaluation/build_spatial_reason_v0.1.1_consistency.json` | Consistency gate result |
| `evaluation/build_spatial_reason_v0.1.1_quality.json` | Renamed from `…_v011_quality.json` (canonical naming) |
| `evaluation/build_spatial_reason_v0.1.1_samples/` | Renamed from `…_v011_samples/` |

**Modified**

| File | Change |
|---|---|
| `spatial_reasoning/dataset_validator.py` | `V011_FROZEN_SHA256`, `verify_v011_unchanged()`, oracle/artifact blocking codes |
| `scripts/validate_build_spatial_reason.py` | Oracle audit per record, `independent_oracle` + `v011_frozen_integrity` fields, canonical naming, embedded consistency gate |
| `scripts/build_spatial_reason.py` | `semantic_policy.py` added to provenance list; duplicate dict key removed; stale KL-05/06/07 wording fixed |
| `datasets/build_spatial_reason/v0.1.1/manifest.json` | `generator_file_sha256` refreshed + `semantic_policy.py`; KL-06/KL-07 wording |
| `docs/architecture_decisions.md` | **ADR-011** added |
| `docs/build_spatial_reason_v0.1.1.md` | Artifact-facts block, oracle section, canonical paths |
| `docs/build_spatial_reason_v0.1.1_quality_audit.md` | Rewritten: correct counts, oracle section, consistency gate |
| `handoff/FROM_DSH.md`, `handoff/PROJECT_STATE.md` | This report; stale counts replaced by machine-checked facts |

**Deleted** — the superseded `evaluation/build_spatial_reason_v011_quality.json`
and `evaluation/build_spatial_reason_v011_samples/` (renamed, not duplicated).

**Not modified:** `datasets/build_spatial_reason/v0.1/**`,
`datasets/build_spatial_reason/v0.1.1/*.jsonl`, `../WHU_Building_Segment/**`,
`handoff/TO_DSH.md`.

---

## 3. Dataset Freeze Verification

Hashes recorded at the start of Task 5C and re-verified afterwards:

| File | SHA256 (before Task 5C = after Task 5C) |
|---|---|
| `v0.1.1/train.jsonl` | `85e3ae168e8d2f6a15bf675eb18fea986e1e0e5732cc93cbdd80e806bb594eee` |
| `v0.1.1/val.jsonl` | `c561d74778c962600bdb33473df2e93de498712a8755634099f9ba27cd246d55` |
| `v0.1.1/test.jsonl` | `6f7e9525f29044779f0ad8123870f61830449202c13207723b540216d1c5b29d` |
| `v0.1/train.jsonl` | `5848157b9748ea0a280e73b8ed14ab6be5f83a018b4ad2687bffb62b2cc54975` |
| `v0.1/val.jsonl` | `93aea13d63f94b25f3c90fe1c3f0c27f891f555a9d67c6c16050fbd3c3c163c6` |
| `v0.1/test.jsonl` | `a0b6bfe461fe3579356133d12b0c099d8a23af827f965ba1a8c3cb5d218255cc` |

`verify_v01_unchanged().unchanged = True`, `verify_v011_unchanged().unchanged =
True`. Both are enforced as blocking codes (`v01_modified`, `v011_modified`).

---

## 4. Canonical Artifact Paths

```
datasets/build_spatial_reason/v0.1.1/manifest.json
datasets/build_spatial_reason/v0.1.1/statistics.json
evaluation/build_spatial_reason_v0.1.1_quality.json
evaluation/build_spatial_reason_v0.1.1_samples/
evaluation/build_spatial_reason_v0.1.1_consistency.json
docs/build_spatial_reason_v0.1.1_quality_audit.md
```

The validator now derives the evaluation file names directly from the version
string (`build_spatial_reason_{version}_quality.json`), so `v0.1.1` produces
`…_v0.1.1_quality.json`. The old `version.replace(".", "")` rule produced
`…_v011_quality.json`, which no document referenced. Only one quality JSON per
version exists — no equally-authoritative duplicate is retained.

---

## 5. Independent Oracle Design

`spatial_reasoning/semantic_oracle.py` (oracle version `1.0`) re-implements the
five semantic questions **from scratch**:

| # | Question | Oracle computation |
|---|---|---|
| 1 | `largest` | global `argmax(area_px)` over all visible components; then frozen margin + eligibility |
| 2 | `smallest` | global `argmin(area_px)`; then frozen margin + eligibility |
| 3 | `<ref> -> nearest` | `argmin` boundary gap over all other visible; then anchor eligibility, target eligibility, frozen margin |
| 4 | `<ref> -> direction` | frozen predicate re-derived from centroids + `alpha`/`tau` |
| 5 | `<ref> -> dir -> nearest` | (4) then (3), with the full direction set as the candidate universe |

**Independence contract**

* Never imports or calls `semantic_policy`, `SP.resolve_size_extreme`,
  `SP.resolve_nearest`, `SP.direction_candidates_over_visible` or
  `SP.admissible_nearest_ids`. Verified by an AST scan of the oracle source and,
  at runtime, by exercising every entry point while a `sys.meta_path` hook makes
  importing `semantic_policy` raise.
* Quality flags are recomputed from `area_px`, the bounding box and the border
  flag rather than read from `component_quality`.
* The direction predicate is re-implemented from the frozen definition instead of
  calling `relations.evaluate_direction`.
* Permitted reuse (per section 4): the component metadata loader and the
  low-level geometric primitive `geometry.component_box_distance`, which is a
  pixel-distance measurement, not an acceptance policy.

**Failure semantics.** A non-admissible oracle resolution for a stored record is
`oracle_ambiguity_mismatch`; a different target is `oracle_target_mismatch`; a
different direction set is `oracle_candidate_set_mismatch`; a different reference
is `oracle_reference_mismatch`; a different trivial flag is
`oracle_trivial_flag_mismatch`. All five are blocking.

---

## 6. Full-Dataset Oracle Audit

| Metric | Value |
|---|---|
| records checked | **25,229** |
| exact semantic target matches | **25,229** |
| target mismatches | **0** |
| semantic-program mismatches | **0** |
| candidate-set mismatches | **0** |
| reference mismatches | **0** |
| ambiguity-policy mismatches | **0** |
| Level-3 trivial-flag agreements | **2,918 / 2,918** |

Acceptance target (§5) of 25,229/25,229 with zero semantic mismatches is met
exactly. Both the production recomputation and the oracle recomputation are 100%.

---

## 7. Hidden Semantic Audit

Independent, oracle-derived (never via the production policy):

```
hidden_eligibility_largest          = 0
hidden_eligibility_smallest         = 0
hidden_eligibility_nearest          = 0
hidden_eligibility_level3_nearest   = 0
semantic_violation_flag_instances   = 0
semantic_violation_unique_records   = 0
semantic_clean_records              = 25229
```

The production policy independently reports the same zeros, so the two
implementations agree on the negative as well as the positive claims.

---

## 8. Artifact Consistency Audit

`scripts/check_artifact_consistency.py` — status **`consistent`**, **0**
violations. It verifies:

1. dataset / semantic-policy / generator / relation-config versions agree across
   the index, `manifest.json` and the quality JSON;
2. total, split, level, Level-2 Type A/B and Level-3 trivial/nontrivial counts
   agree across the index, `manifest.json`, `statistics.json` and the quality
   JSON, plus the `A + B == Level 2` partition invariant recomputed on each;
3. canonical artifacts exist at their canonical names;
4. the deprecated `v011` artifacts do not exist;
5. `manifest.generator_file_sha256` covers every generation-critical file and
   every digest matches the file on disk;
6. v0.1 and v0.1.1 JSONL records still hash to their recorded values;
7. every required Markdown file carries a matching `ARTIFACT-FACTS` block.

Negative tests prove the gate is not vacuous: injecting a count drift, or
resurrecting the deprecated `v011` quality JSON, both make it fail.

The gate's result is embedded in the quality JSON as `artifact_consistency` and
also published as `evaluation/build_spatial_reason_v0.1.1_consistency.json`.

**Stale counts corrected** (section 3.1): Level-2 Type A/B 2,272/2,764 →
**2,275/2,761**; Level-3 trivial/nontrivial 1,261/1,657 → **1,256/1,662**.

---

## 9. Provenance Audit

`v0.1.1/manifest.json → generator_file_sha256` now covers **8** files:

```
spatial_reasoning/annotator.py              (unchanged digest)
spatial_reasoning/semantic_policy.py        (ADDED in Task 5C)
spatial_reasoning/templates.py              (unchanged digest)
spatial_reasoning/relations.py              (unchanged digest)
spatial_reasoning/geometry.py               (unchanged digest)
spatial_reasoning/thresholds.py             (unchanged digest)
spatial_reasoning/component_quality.py      (unchanged digest)
scripts/build_spatial_reason.py             (refreshed: edited by Task 5C)
```

**6 of the 7 previously recorded digests are byte-identical**, which independently
confirms that `annotator.py`, `templates.py`, `relations.py`, `geometry.py`,
`thresholds.py` and `component_quality.py` have not changed since v0.1.1 was
generated. Only `scripts/build_spatial_reason.py` changed, because Task 5C edited
it, and its digest was refreshed accordingly. Every digest now matches the file
on disk, and the consistency gate re-checks this on every run.

`scripts/build_spatial_reason_samples.py` was deliberately **not** added: it only
renders a visualization pack from an existing report and cannot affect the
dataset. `dataset_validator.py` and `semantic_oracle.py` are audit-only.

Also fixed while reviewing the generator: `build_manifest()` declared
`source_component_representation_version` twice (the second silently won), and
three known-limitation entries carried v0.1-era wording.

---

## 10. Test Summary

Exact commands (run from the repository root, conda env `yolo_sam_env`):

```
python tests/test_component_conversion.py     ->  11/11 checks passed   exit 0
python tests/test_geometry.py                 ->  17/17 checks passed   exit 0
python tests/test_relations.py                ->  35/35 checks passed   exit 0
python tests/test_annotator.py                ->  22/22 checks passed   exit 0
python tests/test_dataset_validator.py        ->  18/18 checks passed   exit 0
python tests/test_v011_acceptance.py          ->  19/19 checks passed   exit 0
python tests/test_semantic_oracle.py          ->   8/8  checks passed   exit 0
python tests/test_artifact_consistency.py     ->   8/8  checks passed   exit 0
```

| Metric | Value |
|---|---|
| suites | **8** |
| checks | **138** |
| passed | **138** |
| failed | **0** |
| skipped | **0** |
| exit codes | **0** |

Also run:

```
python scripts/validate_build_spatial_reason.py --version v0.1.1   -> exit 0, verdict PASS
python scripts/check_artifact_consistency.py                       -> exit 0, status consistent
```

Note on §10.6 ("full-dataset oracle target match"): the acceptance audit sweeps
all 25,229 records and its result is asserted by the test from the quality JSON;
the test additionally re-runs the oracle live over up to 4,000 records **per
split** (11,884 in practice) spanning every level and query type. Set
`SPATIAL_ORACLE_FULL=1` to make the test itself sweep all 25,229.

---

## 11. Remaining Limitations

1. **Border truncation reduces coverage** — unchanged from v0.1.1; the semantic
   answer must itself be eligible, so images whose global extreme touches the tile
   edge yield no query of that type. Correctness is traded for coverage
   deliberately (ADR-010).
2. **`scene_level_split_leakage = unverified`** — no scene/geographic grouping
   metadata; val was a random 20% of the original train pool. Exact image
   duplication is ruled out (0). WHU was not resplit.
3. **`peak_memory_mb` is unconfirmed** — the Windows ctypes fallback returns 0.
4. **The oracle shares the geometric distance primitive with production.** This is
   explicitly permitted, but it means a defect inside
   `geometry.component_box_distance` would be invisible to both. The distance
   function is itself covered by `tests/test_geometry.py`.
5. **Oracle independence is structural, not adversarial.** It is a second
   implementation written by the same agent from the same frozen specification. It
   protects against the v0.1 failure mode (policy drift between components) but
   cannot detect a misreading of the *specification* that both implementations
   share. An external reviewer reading the frozen config remains the final check.
6. **Template diversity is still unquantified** — `template_id` is now explicit
   and gate-checked, so this is now measurable, but it has not been measured.
7. **v0.1 remains superseded and non-regenerable** byte-identically, because the
   semantic policy changed after it was produced. It is preserved and verified
   unchanged; regeneration is intentionally not attempted.
8. **Scene-level leakage in v0.1** is unchanged and still unverifiable.

---

## 12. Git Commit / Push

Commit message: `audit: harden BuildSpatialReason v0.1.1 acceptance`

Pre-staging checks performed: `git status --short` inspected; no JSONL is staged;
no weights/checkpoints; no out-of-repo writes. The commit hash and push result are
recorded in the final DSH chat summary.

---

## 13. Ready for Task 5.5?

**Yes, for the v0.1.1 dataset line** — with two conditions attached:

* Task 5.5 should consume `datasets/build_spatial_reason/v0.1.1/` and cite
  `evaluation/build_spatial_reason_v0.1.1_quality.json` as the quality evidence.
  It must **not** use v0.1.
* Task 5.5 should not re-derive counts by hand. It should read
  `evaluation/build_spatial_reason_artifact_index.json` (or run the consistency
  gate), because hand-copied numbers are exactly what went wrong in Task 5B.

**Task 5C did not start Task 5.5, MLLM integration, training, or any later
stage.**
