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

# FROM_DSH — Task 8B.3-P1D11M1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-P1D11M1` |
| Status | **COMPLETE** (canonical manifest identity normalized) |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| Starting HEAD | `ca0e9f4217f0abfb58bffe36ac297c46206fe016` |
| Pre-normalization classification | both 93 / Git-only 4 / disk-only 38 / neither 0 (reproduced exactly) |
| Consumer semantics | sync helper enumerates by `path` and compares actual content; manifest `bytes`/`sha256` are not used for validation; `source_manifest.json` is not self-listed → normalization safe |
| Identity convention | **GIT_CANONICAL_BLOB_BYTES** (`identity_basis` + `identity_basis_note` added; schema/task/source_delivery/canonical_root/copy_policy unchanged) |
| Entry identities changed / unchanged | **38 / 97** |
| Entry count / order / paths | 135 / preserved / 0 additions-removals |
| Post-normalization Git-canonical check | **135/135 PASS** |
| Runtime / tests / README / model card / `.gitattributes` | UNCHANGED |
| External write sync / model runs | NONE / NONE |
| PROP-01 status | OPEN (not closed) |
| Locked candidates | 4 immutable v0.2 TEST sample IDs (unchanged) |
| Report | `docs/task8b3_p1d11_manifest_normalization.md` |
| STOP reason | none |
| Next action | Awaiting ChatGPT audit; the P1D11A policy implementation retry and any canonical → external sync need their own task books |

Watt was not needed for Task 8B.3-P1D11M1 (no downloads, no transfers).

Only `delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json`, this report and the handoff changed.
