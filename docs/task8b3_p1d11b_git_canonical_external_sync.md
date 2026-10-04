# Task 8B.3-P1D11B-R2 — External Git-Canonical Migration + Policy Sync (executed)

## 1. Task and scope

One controlled migration of all 135 manifest-listed external source/config files to Git canonical identity, plus the
separate Git-canonical `source_manifest.json` write, executed strictly against the authorised pre-sync gate
`checked 135 · match 97 · missing 0 · mismatch 38` with **no additional CRLF-derived gate**. No pytest, no model
inference, no locked-candidate run.

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD   = 1f5641a0c0a1f43956f55da8f6a28262d52ee574
helper invocations: 1 check (pre) · 1 sync · 1 check (post)
setup checker: 1 (pre) · 1 (post)
```

## 2. Canonical Git identity gate

```text
entries = 135 · identity mismatches = none
canonical source_manifest Git SHA256 = c72c8ed88b9b62ec49ca7f29f44a2a2b68af820a27b29441a5e7876c9aab9604
```

## 3. Pre-sync external check (authorised gate)

```text
exit = 1 · counts = {"checked": 135, "match": 97, "missing": 0, "mismatch": 38}
expected exactly: checked 135 · match 97 · missing 0 · mismatch 38  → SATISFIED
```

Measured pre-sync mismatch paths (recorded, not used as an extra gate):

| pre-sync mismatch path |
|---|
| `README.md` |
| `buildreasonseg/runtime/_frozen/PORT_PROVENANCE.md` |
| `buildreasonseg/runtime/_frozen/__init__.py` |
| `buildreasonseg/runtime/_frozen/mvp/geometric_relation_field_v02.py` |
| `buildreasonseg/runtime/_frozen/mvp/grcl_directional.py` |
| `buildreasonseg/runtime/_frozen/mvp/native_vector_adapter.py` |
| `buildreasonseg/runtime/_frozen/mvp/task6m_eval.py` |
| `buildreasonseg/runtime/_frozen/mvp/task6m_structured.py` |
| `buildreasonseg/runtime/_frozen/mvp/task6n_relation_decoder.py` |
| `buildreasonseg/runtime/_frozen/mvp/task6p_reference_head.py` |
| `buildreasonseg/runtime/_frozen/mvp/task6q_reference_resolver.py` |
| `buildreasonseg/runtime/_frozen/mvp/task6s_directional_pipeline.py` |
| `buildreasonseg/runtime/_frozen/mvp/task6u_common.py` |
| `buildreasonseg/runtime/_frozen/mvp/task6v_family_reference_resolver.py` |
| `buildreasonseg/runtime/_frozen/mvp/task6w_quality_reference_resolver.py` |
| `buildreasonseg/runtime/_frozen/mvp/task6x_sam2_reference_refiner.py` |
| `buildreasonseg/runtime/_frozen/mvp/task6z_field_composition.py` |
| `buildreasonseg/runtime/_frozen/mvp/task6z_l3_decoder.py` |
| `buildreasonseg/runtime/_frozen/mvp/task7a_l3_pipeline.py` |
| `buildreasonseg/runtime/_frozen/mvp/task7d_data.py` |
| `buildreasonseg/runtime/_frozen/mvp/task7e_l3_decoder_adapter.py` |
| `buildreasonseg/runtime/_frozen/mvp/whu_vector_audit.py` |
| `buildreasonseg/runtime/pipeline.py` |
| `check_setup.py` |
| `docs/command_grammar.md` |
| `docs/model_card.md` |
| `docs/runtime_mapping.md` |
| `environment.yml` |
| `model/buildreasonseg_advisor/metadata.json` |
| `model/buildreasonseg_advisor/model.yaml` |
| `model/components/program_head/qwen_asset_manifest.json` |
| `predict.py` |
| `requirements.txt` |
| `tests/test_cli_contract.py` |
| `tests/test_language_contract.py` |
| `tests/test_model_package.py` |
| `tests/test_setup_checker.py` |
| `tests/test_task8b1_fallback_ux.py` |

## 4. Pre-sync setup and protected asset snapshot

```text
pre-sync setup: exit 0 · READY True · python C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe
```

| protected asset | bytes |
|---|---:|
| `model/buildreasonseg_advisor/decoder.pt` | 1117495 |
| `model/buildreasonseg_advisor/detector.pt` | 54480241 |
| `model/components/sam2/sam2.1_hiera_base_plus.pt` | 323606802 |
| `model/components/sam2/sam2.1_hiera_b+.yaml` | 3766 |
| `model/components/program_head/program_parser_l3_rehearsal_v1.pt` | 70090713 |

Qwen directory: exists=True · regular files=10 ·
total bytes=4266640306

## 5. Controlled 135-file migration

```text
exit = 0 · counts = {"copied": 135, "verified": 135, "failures": 0}
expected: copied 135 · verified 135 · failures 0  → SATISFIED
```

## 6. Separate Git-canonical source_manifest write

```text
external SHA256 = c72c8ed88b9b62ec49ca7f29f44a2a2b68af820a27b29441a5e7876c9aab9604
canonical Git SHA256 = c72c8ed88b9b62ec49ca7f29f44a2a2b68af820a27b29441a5e7876c9aab9604
match = YES
```

## 7. Post-sync external check

```text
exit = 0 · counts = {"checked": 135, "match": 135, "missing": 0, "mismatch": 0} → 135/135 PASS
```

## 8. Protected assets and post-sync setup

```text
protected files changed = NONE
Qwen directory changed  = False
runtime/user dirs changed = NONE
post-sync setup: exit 0 · READY True
```

## 9. Unchanged by this task

```text
canonical RC1 source/config (read-only Git blob access) = UNCHANGED
pytest / model inference / locked-candidate runs        = NONE / NONE / NONE
frozen metrics / architecture / locked candidates       = UNCHANGED
PROP-01 status                                          = OPEN (not closed)
```

## 10. Next step (not executed)

The external delivery now carries Git canonical identity for all 135 manifest-listed files and its control manifest.
Evaluating the four locked v0.2 TEST qualitative candidates requires a separate authorised task.
