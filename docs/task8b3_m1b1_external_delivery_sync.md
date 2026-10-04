# Task 8B.3-M1B.1 — Controlled Canonical → External Delivery Sync

## 1. Task and scope

Controlled synchronization of the M1A compact-proposal canonical source/config set into the existing runnable
external RC1 delivery, followed by integrity and readiness verification. No pytest, no inference, no policy change.

## 2. Starting HEAD

```text
branch = fix/task8b3-mem01-compact-proposals
HEAD   = a215db142caec155e2f6787804378e442fbb55d5
tracked tree at start = only M handoff/TO_DSH.md
```

## 3. Canonical manifest gate result

```text
schema            = BuildReasonSeg.AdvisorRC1.SourceManifest.v1
entry count       = 135
path/size/sha256  = 135/135 PASS (no mismatch)
```

## 4. Canonical `source_manifest.json` SHA256

```text
1135d8b44d882e5462ed9cfb627322a5b2895c4874ec97fcdd61d6f2e6567497
```

## 5. Pre-sync external `check_setup.py` result

```text
exit code = 0
BuildReasonSeg environment: READY
READY = True
```

## 6. Protected asset pre-sync snapshot

| file | exists | bytes | sha256 |
|---|---|---|---|
| `model/buildreasonseg_advisor/decoder.pt` | True | 1117495 | 9187b133ee4c71ca… |
| `model/buildreasonseg_advisor/detector.pt` | True | 54480241 | ef852b5801e6bdf9… |
| `model/components/sam2/sam2.1_hiera_base_plus.pt` | True | 323606802 | a2345aede8715ab1… |
| `model/components/sam2/sam2.1_hiera_b+.yaml` | True | 3766 | ef47e14197a65c1f… |
| `model/components/program_head/program_parser_l3_rehearsal_v1.pt` | True | 70090713 | c150573613c42109… |
| `model/components/program_head/qwen_asset_manifest.json` | True | 2058 | 4cf1efff8d552532… |

Qwen base directory `model/components/program_head/Qwen3-VL-2B-Instruct`:

```text
exists            = True
regular files     = 10
total bytes       = 4266640306
first file names  = ['chat_template.json', 'config.json', 'generation_config.json']
```

Not sync targets (recorded only): `inference/input`, `inference/output`, `logs`, `runs`, `datasets`, plus every model
binary/component asset that is absent from the 135-entry source manifest.

## 7. Sync policy

* exactly the **135** manifest-listed relative paths were copied, canonical → external, one file at a time;
* the source bytes and SHA256 of each file were re-verified against the manifest entry **before** copying;
* only the exact manifest-listed destination path was overwritten; no directory mirroring, no recursive copy;
* **no external file was deleted**; no unlisted external file was touched;
* `source_manifest.json` was copied separately afterwards;
* parent directories were created only where a manifest-listed path required them
  (0 new parent dirs: none);
* the sync script lives outside the repository and is not committed.

## 8. Post-sync external 135/135 result

```text
external manifest-listed files = 135/135 PASS
mismatches = none
source_manifest byte-identical to canonical = True
```

## 9. Protected asset post-sync comparison

```text
protected files changed = NONE
Qwen directory count/names/bytes identical = True
```

## 10. Post-sync `check_setup.py` result

```text
exit code = 0
BuildReasonSeg environment: READY
READY = True
```

## 11. pytest

`NOT RUN` — explicitly excluded by this task book (external full regression is Task M1B.2 after audit).

## 12. Real inference

`NOT RUN` — no `predict.py`, no detector/Qwen/SAM2/D-B1 execution, no A1–B2 sample.

## 13. PROP-01 / REF-01 / MASK-01

`UNCHANGED / UNCHANGED / UNCHANGED` — no policy, threshold, tiling, merge, reference or validity code was touched
beyond the already-frozen M1A compact-proposal change now transferred by this sync.

## 14. Next action

Awaiting ChatGPT audit before the M1B.2 external full regression suite.
