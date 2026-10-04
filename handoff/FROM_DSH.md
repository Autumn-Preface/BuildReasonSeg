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

# FROM_DSH — Task 8B.3-P1D11B-R2 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-P1D11B-R2` |
| Status | **COMPLETE** (external Git-canonical migration + policy sync executed) |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| Starting HEAD | `1f5641a0c0a1f43956f55da8f6a28262d52ee574` |
| Canonical Git manifest | 135/135 PASS · Git SHA256 `c72c8ed88b9b62ec49ca7f29f44a2a2b68af820a27b29441a5e7876c9aab9604` |
| Pre-sync external check | {"checked": 135, "match": 97, "missing": 0, "mismatch": 38} — authorised 97/38 gate SATISFIED |
| Pre-sync external setup | READY |
| Manifest-listed files copied/verified | 135/135 · failures 0 |
| External source_manifest Git SHA match | YES |
| Post-sync external check | {"checked": 135, "match": 135, "missing": 0, "mismatch": 0} → **135/135 PASS** |
| Protected assets / Qwen / runtime dirs changed | NO / NO / NO |
| Post-sync external setup | READY |
| Extra CRLF-derived gates applied | NONE (authorised 97/38 gate only) |
| pytest / model inference / locked candidate runs | NONE / NONE / NONE |
| Canonical RC1 written | NO (read-only Git blob access) |
| Frozen metrics / architecture / locked candidates | UNCHANGED |
| PROP-01 status | OPEN (not closed) |
| Report | `docs/task8b3_p1d11b_git_canonical_external_sync.md` |
| Next action | Awaiting ChatGPT audit; locked-candidate evaluation needs its own task book |

Watt was not needed for Task 8B.3-P1D11B-R2 (no downloads, no transfers).

The external delivery was modified only through the controlled 135-file Git-canonical helper sync plus the separately
written `source_manifest.json`; no model binary, Qwen asset, input/output/log/run/dataset content or unlisted file was
touched, and no canonical RC1 file was modified.
