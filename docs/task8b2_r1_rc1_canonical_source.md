# Task 8B.2-R1 — Register RC1 Canonical Delivery Source

## 1. Task

`Task 8B.2-R1 — Register RC1 Canonical Delivery Source`

## 2. ChatGPT Decision

ChatGPT accepted the prior Task 8B.2 Phase D1 STOP and froze the canonical source path to:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/
```

The canonical tree is a **source/config snapshot**, not a binary delivery; the external delivery remains the
runnable local package; the Ultralytics/readiness fix stays deferred to Task 8B.2-R2.

## 3. Starting State

| item | value |
|---|---|
| branch | `audit/task8b2-rc1-runtime-closure` |
| starting HEAD | `6c2b915dbc64acdeb099d194005d74c7180c95fa` |
| allowed initial working-tree state | `M handoff/TO_DSH.md` (the Task 8B.2-R1 task book supplied by ChatGPT); no other tracked modification or untracked file |
| external delivery path | `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1` |
| origin | `https://github.com/Autumn-Preface/BuildReasonSeg.git` |

## 4. Registration Policy

- **Included extensions** (case-insensitive): `.py .pyw .json .yaml .yml .toml .ini .cfg .conf .txt .md .csv`,
  plus the extensionless files `LICENSE` / `NOTICE` if present (neither exists in the delivery).
- **Excluded directories** (any path component): `.git .conda __pycache__ .pytest_cache logs log cache caches tmp
  temp runs outputs output predictions`, plus the exact subtrees `inference/input` and `inference/output`.
- **Excluded binary/model/data extensions**: `.pt .pth .ckpt .safetensors .onnx .engine .tflite .h5 .pb .bin .npy
  .npz .tif .tiff .jpg .jpeg .png .bmp .webp .gif .zip .7z .rar .tar .gz` (the delivery contains `.pt`,
  `.safetensors`, `.png`, `.tif` files, all excluded).
- **Symlink / junction policy**: symlinks, junctions and reparse-point targets are never followed or copied.
  The inventory walk found **no** symlink/junction candidate, so the Phase C4 STOP did not trigger.
- Delivery inventory (local-only, not committed): `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\logs\task8b2_r1_delivery_inventory.txt`
  (`relative_path<TAB>size_bytes`, lexicographic, 160 entries).

## 5. Snapshot Result

| item | value |
|---|---|
| copied file count (manifest entries) | **146** |
| total copied bytes | **12771766** |
| largest copied file | 7,032,403 bytes (`model/components/program_head/Qwen3-VL-2B-Instruct/tokenizer.json`) |
| manifest | `delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json` |
| canonical files on disk (incl. manifest + notice) | **148** |
| source↔canonical size + SHA256 verification | **PASS for all 146 files** (verified during copy; any mismatch would have STOPped) |

Determinism: manifest entries are one per file, sorted lexicographically by normalized forward-slash `path`, with
no duplicate path, no absolute path and no `..`; `schema` =
`BuildReasonSeg.AdvisorRC1.SourceManifest.v1`.

## 6. Required RC1 Files

| required path | canonical counterpart | note |
|---|---|---|
| `check_setup.py` | present | source only; not modified in this task |
| `environment.yml` | present | source only; Ultralytics pin deferred to Task 8B.2-R2 |
| `predict.py` | present | source only |
| `buildreasonseg/` | present | all `.py` sources |
| `model/buildreasonseg_advisor/` | present | only lightweight config/metadata: `model.yaml`, `metadata.json`, `metrics.json` |

**No model weights were copied**: `decoder.pt`, `detector.pt`,
`model/components/sam2/sam2.1_hiera_base_plus.pt` and the Qwen `.safetensors` weights are excluded by extension
policy. Note for audit transparency: Qwen **tokenizer/config** files matching the allowed extensions
(`.json`/`.txt`) *are* included by the mechanical extension policy, because the policy excludes only the listed
binary extensions:

```text
model/components/program_head/Qwen3-VL-2B-Instruct/chat_template.json
model/components/program_head/Qwen3-VL-2B-Instruct/config.json
model/components/program_head/Qwen3-VL-2B-Instruct/generation_config.json
model/components/program_head/Qwen3-VL-2B-Instruct/merges.txt
model/components/program_head/Qwen3-VL-2B-Instruct/preprocessor_config.json
model/components/program_head/Qwen3-VL-2B-Instruct/tokenizer.json
model/components/program_head/Qwen3-VL-2B-Instruct/tokenizer_config.json
model/components/program_head/Qwen3-VL-2B-Instruct/video_preprocessor_config.json
model/components/program_head/Qwen3-VL-2B-Instruct/vocab.json
```

## 7. Sync Helper

- script path: `scripts/sync_advisor_rc1_delivery.py`
- canonical source: fixed internally to `delivery_src/BuildReasonSeg_Advisor_RC1` (no CLI option can change it)
- CLI: `--destination <path>` and `--destination <path> --check`
- **normal mode**: read the manifest → for each manifest-listed file copy canonical → destination byte-for-byte
  (creating parent directories), overwrite only listed counterparts, never delete destination files, never touch
  weights/assets/images/logs/outputs, recompute destination SHA256 and require equality, print
  `copied / verified / failures` counts
- **check mode**: read-only; reports `MATCH` / `MISSING` / `MISMATCH` per manifest file and a
  `checked / match / missing / mismatch` summary; exit code `0` only when every manifest-listed file matches
- **safety**: refuses destination == canonical root, destination inside canonical root, non-existent destination,
  duplicate manifest paths, `..` paths and absolute manifest paths; never runs pip/conda, never imports model
  frameworks, never invokes predict, never deletes destination content, never alters model assets

## 8. Validation

| Check | Result |
|---|---|
| delivery inventory created locally | PASS (160 entries; local-only, not committed) |
| canonical snapshot created | PASS (146 manifest files at the frozen path) |
| byte identity | PASS (all 146 files) |
| SHA256 identity | PASS (all 146 files) |
| manifest deterministic | PASS (sorted, unique, relative, no `..`, no absolute path) |
| external delivery `--check` | PASS (`checked=146 match=146 missing=0 mismatch=0`, exit 0, delivery unmodified) |
| dedicated sync tests | PASS (`pytest tests/test_sync_advisor_rc1_delivery.py -q` → 14 passed) |
| repository suite | PASS (`pytest tests/ -q` → 1525 passed, 1 skipped; 1511 pre-existing + 14 new) |
| no binary/model assets staged | PASS (0 binary/model files in the canonical tree; largest file 7.03 MiB < 10 MiB) |
| no runtime behaviour changed | PASS (no RC1 runtime file edited; no package install; no inference run) |

Additional Phase J observation (factual, no action taken): `git diff --check` reports trailing whitespace inside
`handoff/TO_DSH.md`; that whitespace belongs to the task book text supplied by ChatGPT and was not introduced or
edited by this task.

## 9. Deferred Issue

> `RC1-ENV-01` is intentionally not fixed in Task 8B.2-R1. Ultralytics dependency installation/pinning, readiness validation, runtime/provenance display and A1 inference remain deferred to Task 8B.2-R2 after ChatGPT audit.

## 10. Scientific Scope

> No training, checkpoint modification, final-test access, architecture change, parser-semantic change, threshold tuning or research conclusion change occurred in Task 8B.2-R1.
