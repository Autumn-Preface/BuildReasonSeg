# Task MASK01_D1_R1D_SOURCE_MANIFEST_CANONICALIZATION

## 1. Task record

```text
task_id = MASK01_D1_R1D_SOURCE_MANIFEST_CANONICALIZATION
status  = COMPLETE
starting branch/head = fix/task8b3-mask01-success-semantics-r1c / 3bf9dafac69175d36a1abb6a83d019ef118ca72c
task branch = fix/task8b3-mask01-success-semantics-r1d
only file modified = delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
identity source = HEAD Git canonical blob bytes (working-tree bytes NOT used)
```

## 2. Entry count and path-order preservation

```text
entry_count_before = 135 · entry_count_after = 135 · preserved = True
path_order_preserved = True
missing at HEAD = NONE
```

## 3. Entries recomputed from Git canonical blobs

| path | old bytes | new bytes | old sha256 | new sha256 |
|---|---:|---:|---|---|
| `README.md` | 10499 | 11171 | `f709903048d4` | `3a17a3c20a7f` |
| `buildreasonseg/runtime/pipeline.py` | 18736 | 19237 | `3a48fcf0f2c0` | `f4c305a5d89f` |
| `docs/model_card.md` | 6954 | 7507 | `8cbb29b72e69` | `dd7a0c663c13` |
| `docs/runtime_mapping.md` | 2286 | 2781 | `654f10a40951` | `ec49552fea57` |
| `inference/README.md` | 1055 | 1545 | `50a24ce6859d` | `7a587312a53a` |
| `predict.py` | 17911 | 18681 | `96ae67697d41` | `a342e92b515c` |
| `tests/test_cli_contract.py` | 6877 | 10188 | `b3894aeb782b` | `f6cfe97b2b16` |
| `tests/test_task8b_runtime.py` | 28128 | 29711 | `71915e8abfbe` | `c25ef4569805` |

```text
updated = 8 · already canonical = 127
```

## 4. All-entry `entry_source_bytes()` Git canonical validation

```text
helper = scripts/sync_advisor_rc1_delivery.py::entry_source_bytes(entry: 'dict', basis: 'str | None', root: 'Path') -> 'bytes | None'
entries validated = 135 · failures = NONE
result = PASS
```

## 5. Manifest contract validation

```text
{
 "schema_present": true,
 "schema_value": "BuildReasonSeg.AdvisorRC1.SourceManifest.v1",
 "identity_basis": "GIT_CANONICAL_BLOB_BYTES",
 "identity_basis_is_git_canonical": true,
 "entry_count": 135,
 "unique_paths": true,
 "all_paths_exist_at_head": true,
 "no_control_file_entry": true
}
contract_pass = True
```

## 6. Untouched and prohibitions

```text
product source / tests / canonical docs / sync script = UNCHANGED (True)
pytest = not run · model inference = false · training = false · external sync = false
inspect-proposals = untouched
github_persistence_policy = ALL_TASK_OUTCOMES_PUSHED
next_gate = CHATGPT_R1D_REMOTE_AUDIT
```
