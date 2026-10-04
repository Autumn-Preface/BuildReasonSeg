# Task 8B.3-P1D11M1 — Canonical Source-Manifest Identity Normalization

## 1. Task and scope

Unify the canonical `source_manifest.json` identity convention to **Git canonical blob/index bytes** and freeze that
semantics in the manifest metadata. No model, test, runtime, README/model-card or external-delivery change.

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD   = ca0e9f4217f0abfb58bffe36ac297c46206fe016
```

## 2. Reproduced mixed-convention finding (§4)

```text
both = 93 · git-only = 4 · disk-only = 38 · neither = 0
```

This matches the P1D11A-R1 measurement exactly.

## 3. Consumer semantics verified before normalization (§5)

`scripts/sync_advisor_rc1_delivery.py`:

```text
path field used to enumerate sync files            = True
manifest bytes used to validate source/destination = False
manifest sha256 used to validate source/destination = False
actual source-vs-destination byte/hash comparison  = True
source_manifest.json self-listed among the 135 entries = False
```

The helper enumerates by `path` and compares actual source/destination content rather than trusting the manifest
`bytes`/`sha256`, and the manifest does not list itself, so normalization is semantically safe.

## 4. Frozen identity semantics (§6)

```json
"identity_basis": "GIT_CANONICAL_BLOB_BYTES",
"identity_basis_note": "bytes and sha256 are computed from Git canonical blob/index bytes; working-tree line-ending expansion is not part of source identity"
```

`schema` remains `BuildReasonSeg.AdvisorRC1.SourceManifest.v1`; `task`, `source_delivery`, `canonical_root` and
`copy_policy` are unchanged; no per-entry metadata field was added.

## 5. Normalized entry identities (§7)

Entry count 135, entry order preserved, 0 path additions/removals; **38 identities changed** (the working-tree-only
entries) and **97 unchanged**.

| entry | bytes before | bytes after | sha256 before | sha256 after |
|---|---:|---:|---|---|
| `README.md` | 9484 | 9335 | `ecbcac3f5cf9…` | `e5ddbd5b602d…` |
| `buildreasonseg/runtime/_frozen/PORT_PROVENANCE.md` | 717 | 704 | `4bcbc589eb43…` | `d5a6070821a4…` |
| `buildreasonseg/runtime/_frozen/__init__.py` | 86 | 85 | `b2c82567e038…` | `ffcfe449612e…` |
| `buildreasonseg/runtime/_frozen/mvp/geometric_relation_field_v02.py` | 8639 | 8427 | `9e990665d753…` | `269b2e2e120c…` |
| `buildreasonseg/runtime/_frozen/mvp/grcl_directional.py` | 11698 | 11401 | `1c3f32781376…` | `9e60d038a738…` |
| `buildreasonseg/runtime/_frozen/mvp/native_vector_adapter.py` | 16214 | 15826 | `e04b19946081…` | `236c40992704…` |
| `buildreasonseg/runtime/_frozen/mvp/task6m_eval.py` | 15469 | 15077 | `008fa9fbec28…` | `6f8fbcb92151…` |
| `buildreasonseg/runtime/_frozen/mvp/task6m_structured.py` | 17563 | 17163 | `a54e0a9bd5a0…` | `101d52ca607c…` |
| `buildreasonseg/runtime/_frozen/mvp/task6n_relation_decoder.py` | 18784 | 18306 | `826334853d62…` | `88efe10d0ecd…` |
| `buildreasonseg/runtime/_frozen/mvp/task6p_reference_head.py` | 5363 | 5234 | `174b2e58de27…` | `6c423e79e478…` |
| `buildreasonseg/runtime/_frozen/mvp/task6q_reference_resolver.py` | 10326 | 10038 | `d6d3f06fe39d…` | `6a356761f2e0…` |
| `buildreasonseg/runtime/_frozen/mvp/task6s_directional_pipeline.py` | 14371 | 14015 | `04977262e1b2…` | `d7b3be598a7c…` |
| `buildreasonseg/runtime/_frozen/mvp/task6u_common.py` | 12965 | 12660 | `8f5d9d195229…` | `6a4659aff014…` |
| `buildreasonseg/runtime/_frozen/mvp/task6v_family_reference_resolver.py` | 6854 | 6690 | `6486f155b0eb…` | `c6e449dd241a…` |
| `buildreasonseg/runtime/_frozen/mvp/task6w_quality_reference_resolver.py` | 5860 | 5733 | `75c507643267…` | `93e1053b17f2…` |
| `buildreasonseg/runtime/_frozen/mvp/task6x_sam2_reference_refiner.py` | 9922 | 9673 | `00062c2a2d66…` | `f497985bcd61…` |
| `buildreasonseg/runtime/_frozen/mvp/task6z_field_composition.py` | 6443 | 6291 | `a1ed3cef7795…` | `5626167a9f85…` |
| `buildreasonseg/runtime/_frozen/mvp/task6z_l3_decoder.py` | 10385 | 10153 | `be091de3b5ac…` | `ebcb33392c7a…` |
| `buildreasonseg/runtime/_frozen/mvp/task7a_l3_pipeline.py` | 15188 | 14857 | `e0834d3af19f…` | `6c1e88da47fa…` |
| `buildreasonseg/runtime/_frozen/mvp/task7d_data.py` | 4332 | 4230 | `015e6ead6c2f…` | `a2523d6bb14a…` |
| `buildreasonseg/runtime/_frozen/mvp/task7e_l3_decoder_adapter.py` | 5712 | 5595 | `42311fc5ba7e…` | `c8c4a14dfabd…` |
| `buildreasonseg/runtime/_frozen/mvp/whu_vector_audit.py` | 17457 | 17001 | `c87b5f029b06…` | `2fb131cf8367…` |
| `buildreasonseg/runtime/pipeline.py` | 19113 | 18736 | `0dabcdfe5b3a…` | `3a48fcf0f2c0…` |
| `check_setup.py` | 14823 | 14494 | `5b846390a4c9…` | `7a3db0aae132…` |
| `docs/command_grammar.md` | 5129 | 5033 | `221b6b778040…` | `98935e6f127e…` |
| `docs/model_card.md` | 5997 | 5900 | `ec81072cc682…` | `4d4949fe94b4…` |
| `docs/runtime_mapping.md` | 2311 | 2286 | `b2cde6eaffd3…` | `654f10a40951…` |
| `environment.yml` | 906 | 875 | `c03d734cb07e…` | `a02fbad6effc…` |
| `model/buildreasonseg_advisor/metadata.json` | 6973 | 6802 | `72b371d9deb5…` | `a77f5b76bd8e…` |
| `model/buildreasonseg_advisor/model.yaml` | 2337 | 2268 | `fb7e357c5baf…` | `83f0f04c62eb…` |
| `model/components/program_head/qwen_asset_manifest.json` | 2058 | 1995 | `4cf1efff8d55…` | `5c4cf25e31e1…` |
| `predict.py` | 18264 | 17911 | `9e5d64d22297…` | `96ae67697d41…` |
| `requirements.txt` | 454 | 435 | `56de38bd614e…` | `adfdd3b481a0…` |
| `tests/test_cli_contract.py` | 7049 | 6877 | `9b175af5ab32…` | `b3894aeb782b…` |
| `tests/test_language_contract.py` | 6270 | 6127 | `083671ad1dc2…` | `c73f0f38d27a…` |
| `tests/test_model_package.py` | 7971 | 7799 | `ff1fb7c4493e…` | `3431d274fbdf…` |
| `tests/test_setup_checker.py` | 10661 | 10431 | `f605f45e947f…` | `acc7455ac6f2…` |
| `tests/test_task8b1_fallback_ux.py` | 10862 | 10627 | `d6adf64addac…` | `ae8d28dd7438…` |

## 6. Authoritative post-normalization check (§8)

```text
Git-canonical normalized manifest = 135/135 PASS (0 mismatches against `git show HEAD:<path>`)
```

Normalization changes only the recorded identity values; the working tree still contains the CRLF-expanded copies that
the external delivery consumes, which is expected and explicitly **not** a claim that CRLF-expanded external files have
byte-identical content to every manifest entry.

## 7. Unchanged by this task

```text
runtime / tests / README / model card / .gitattributes = UNCHANGED
external delivery (no write sync)                     = UNCHANGED
model runs / inference                                = NONE
PROP-01 status                                        = OPEN (not closed)
```

## 8. Next step (not executed)

The supported-domain policy implementation (P1D11A retry) can now pass its pre-edit gate; a canonical → external
controlled sync remains a separate authorised task.
