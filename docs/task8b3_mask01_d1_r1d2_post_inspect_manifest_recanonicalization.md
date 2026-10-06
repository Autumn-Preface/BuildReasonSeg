# Task MASK01_D1_R1D2_POST_INSPECT_MANIFEST_RECANONICALIZATION

## 1. Task record

```text
task_id = MASK01_D1_R1D2_POST_INSPECT_MANIFEST_RECANONICALIZATION
status  = COMPLETE
starting branch/head = fix/rc1-inspect-proposals-preflight-order / efc48eb5ee8b0851a84e8a3d95533f1acd23cacf
task branch = fix/task8b3-mask01-success-semantics-r1d2
only file modified = delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
identity source = HEAD Git canonical blob bytes (working-tree bytes NOT used)
```

## 2. Ordered path digests

```text
ordered_path_digest_before = 7967127fcfadfe4f7f19a3511163c9f77476673774dcac6e8f5a7e49ea20a7bd
ordered_path_digest_after  = 7967127fcfadfe4f7f19a3511163c9f77476673774dcac6e8f5a7e49ea20a7bd
digest preserved = True · count preserved = True (135 -> 135)
path_order_preserved = True
missing at HEAD = NONE
```

## 3. Entries recomputed from HEAD Git blobs

| path | old bytes | new bytes | old sha256 | new sha256 |
|---|---:|---:|---|---|
| `predict.py` | 18681 | 18887 | `a342e92b515c` | `16e18bd5af20` |
| `tests/test_cli_contract.py` | 10188 | 10877 | `f6cfe97b2b16` | `7ff3efdf3466` |

```text
updated = 2 · already canonical = 133
```

## 4. All-entry `entry_source_bytes()` Git canonical validation

```text
helper = scripts/sync_advisor_rc1_delivery.py::entry_source_bytes(entry: 'dict', basis: 'str | None', root: 'Path') -> 'bytes | None'
entries validated = 135 · failures = NONE · result = PASS
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
 "no_control_file_entry": true,
 "ordered_path_digest_before": "7967127fcfadfe4f7f19a3511163c9f77476673774dcac6e8f5a7e49ea20a7bd",
 "ordered_path_digest_after": "7967127fcfadfe4f7f19a3511163c9f77476673774dcac6e8f5a7e49ea20a7bd"
}
contract_pass = True
```

## 6. Untouched and prohibitions

```text
product source / tests / canonical docs / sync script = UNCHANGED (True)
pytest = not run · model inference = false · training = false · external sync = false
github_persistence_policy = ALL_TASK_OUTCOMES_PUSHED · next_gate = CHATGPT_R1D2_REMOTE_AUDIT
```
