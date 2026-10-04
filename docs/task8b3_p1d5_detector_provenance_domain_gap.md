# Task 8B.3-P1D5 — Detector Provenance and A2 Domain-Gap Audit (read-only)

## 1. Task and scope

Read-only audit of the active proposal-detector provenance, the A2 data/domain provenance, locally available
alternative detectors and the scientific-freeze impact of changing the detector. **No model inference, pytest,
`predict.py`, training, fine-tuning, download, checkpoint replacement or functional code change occurred**; the
checkpoint binary was not loaded (only hashes/metadata were read).

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD   = 15a90ec11ad66c8f365284bf53eb320661a1e73f
```

## 2. Active checkpoint identity

```text
external checkpoint : model/buildreasonseg_advisor/detector.pt
bytes               : 54 480 241
sha256              : ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474
canonical copy      : NOT PRESENT (canonical snapshot tracks source/config only)
```

Tool-verified provenance: this exact size+hash equals
`artifacts/checkpoints/task6m1/runs/m1_yolo26m_seg_continued/weights/best.pt` (54 480 241 bytes,
`ef852b5801e6bdf9…`), which `model.yaml` documents as the detector source with the same sha256
(`ef852b5801e6bdf9…`) — the delivery checkpoint is byte-consistent with the documented research artifact.

## 3. Model family / task / training provenance

```text
family / task        = YOLO26m-seg, instance segmentation (metadata.json: "detector": "YOLO26m-seg
                       (frozen U-C1 proposal model)")
framework            = ultralytics 8.4.164 · PyTorch 2.13.0+cu132 (metadata.json)
pretrained base      = artifacts/checkpoints/task6m/pretrained/yolo26m-seg.pt (54 750 385 B, 16b636f04e8fb6a3…)
stage-1 run          = artifacts/checkpoints/task6m/runs/m1_yolo26m_seg (args.yaml data = artifacts/task6m_yolo_native/data.yaml)
stage-2 (active) run = artifacts/checkpoints/task6m1/runs/m1_yolo26m_seg_continued
                       model = …/weights/last_resume.pt (resume), data = artifacts/task6m_yolo_native/data.yaml
continuation source  = artifacts/checkpoints/task6m1/source_epoch18_snapshot (sha fd407db634a8a7ef…,
                       identical to task6m/runs/m1_yolo26m_seg/weights/best.pt)
training dataset     = artifacts/task6m_yolo_native/data.yaml → WHU building dataset lineage
                       (datasets/whu/, README.md); legacy baseline record references whu_building_v1
epochs / hyperparameters of the active run : NOT ESTABLISHED (args.yaml hyperparameter block not parsed here)
```

## 4. U-C1 selection evidence

```text
status = ESTABLISHED
basis  : predict_buildreasonseg_directional.py and predict_buildreasonseg_l3.py default to
         --proposal-checkpoint artifacts\checkpoints\task6m1\runs\m1_yolo26m_seg_continued\weights\best.pt;
         every artifacts/task6m1_demo/*/result.json records that same proposal_checkpoint;
         model.yaml records family+sha256 for the delivered detector.
nuance : the legacy script predict_structured.py still points at the earlier
         artifacts\checkpoints\task6m\runs\m1_yolo26m_seg\weights\best.pt (pre-continuation epoch-18 model).
```

## 5. A2 data source / domain

```text
A2.png identity : 1 677 040 bytes, 1024x1024 RGB, sha256 10286b1e76db9e38c474635a465c9e677dbcf58375c1d39f7b742eeb991f434f
recorded origin : NOT ESTABLISHED — the frozen six-image suite documentation records size/hash/format only;
                  no dataset, site, sensor, season, GSD or geographic provenance for A2 exists in the repository
                  evidence examined.
```

The detector's training/validation domain is the WHU building dataset (§3); whether A2 belongs to that same domain
cannot be decided from the available evidence.

## 6. Locally available alternative detectors

```text
classification = VALIDATED_ALTERNATE_AVAILABLE (evidence normalized in §12.6)
```

| candidate | path | bytes | sha256 prefix | prior use evidence |
|---|---|---:|---|---|
| YOLO26m-seg epoch-18 (pre-continuation) | `artifacts/checkpoints/task6m/runs/m1_yolo26m_seg/weights/best.pt` (= `task6m1/source_epoch18_snapshot/weights/best.pt`) | 162 481 487 | `fd407db634a8a7ef…` | used by `predict_structured.py` default and by `artifacts/task6m_demo/*/result.json` |
| YOLOv8m-seg WHU baseline | `WHU_Building_Segment/runs/segment/logs/whu_building_v1/weights/best.pt` | 54 835 548 | `d9a6a65b7e0819ce…` | `artifacts/task6j_yolo_proposals/provenance.json` — "frozen YOLOv8m-seg-WHU baseline … trained 100 epochs on the legacy WHU YOLO dataset" |
| YOLO26m-seg pretrained (no WHU fine-tune) | `artifacts/checkpoints/task6m/pretrained/yolo26m-seg.pt` | 54 750 385 | `16b636f04e8fb6a3…` | base for stage-1 training |
| YOLO26s-seg pretrained (smaller variant) | `artifacts/checkpoints/task6m/pretrained/yolo26s-seg.pt` | 23 467 933 | `3da1d83e31caec96…` | smoke runs only |
| legacy YOLO baseline record | `baseline/yolo_whu/` (README) | — | — | documented as "frozen record only … reference only" |

So earlier validated/used detector artifacts do exist locally, but **none has been validated on A2 or against the RC1
product path**; `README.md` explicitly keeps the legacy baseline as a reference-only record.

## 7. Scientific-freeze compatibility

```text
classification = DETECTOR_CHANGE_TOUCHES_FROZEN_RESEARCH_CLAIMS
```

`model/buildreasonseg_advisor/metadata.json` lists the detector inside the frozen RC1 architecture
("detector": "YOLO26m-seg (frozen U-C1 proposal model)") and records
`research_baseline.task7j_test_status = FINAL_TEST_CONSUMED` for the frozen research state
(`6c2b915dbc64acdeb099d194005d74c7180c95fa`). The proposal detector feeds the frozen downstream chain
(SAM2.1 Hiera Base+, D-B1 decoder, Reference/relation semantics), so replacing it is not an engineering-only change:
it would alter the proposal source that the frozen task results and claims were produced with. Any detector change
therefore requires explicit re-validation (and, for research claims, a documented re-freeze), not a silent swap.

## 8. Primary diagnosis (exactly one)

```text
PROP01_DOMAIN_GAP_INSUFFICIENT_EVIDENCE
```

Because checkpoint provenance **is** established (§2–§4) but A2's provenance/domain is **NOT ESTABLISHED** (§5), the
evidence cannot distinguish a genuine detector domain mismatch from an in-domain single-image blind spot. The
stronger classifications are therefore not supportable, and `PROP01_CHECKPOINT_PROVENANCE_INCOMPLETE` does not apply
either.

## 9. Next gate (recommended only — NOT executed)

```text
NEXT = DOMAIN_EVIDENCE_RECOVERY
```

Derivation against the task book's rule order: a validated alternate does exist but is **not** scientifically
compatible without re-validation (§7), so `CONTROLLED_ALTERNATE_DETECTOR_A2_PROBE` is not selected; the domain-gap,
in-domain-blind-spot and provenance-incomplete branches do not apply (§8); the remaining branch is
`DOMAIN_EVIDENCE_RECOVERY`, i.e. recover A2's data provenance/domain so the mismatch question can be answered.

## 10. Missing information (explicitly NOT ESTABLISHED)

```text
A2 data source / domain / sensor / GSD / site                       : NOT ESTABLISHED
active checkpoint training epochs and hyperparameters               : NOT ESTABLISHED
whether any alternative checkpoint was validated on A2 or RC1 path  : NOT ESTABLISHED
whether A2 was ever part of the WHU training/validation splits       : NOT ESTABLISHED
```

## 11. Scope statement

No inference, pytest, `predict.py`, training, fine-tuning, download, checkpoint substitution or functional edit
occurred; the `.pt` files were hashed, not loaded; `RC1-DEMO-PROP-01` remains open; REF-01 / MASK-01 / Task 8B.4 /
Task 8C were not entered.


---

## 12. P1D5-R1 — evidence completion (docs only)

This section completes the evidence contract flagged by the P1D5 audit. No model inference, pytest, `predict.py`,
training, fine-tuning, alternate-detector run or checkpoint load occurred; only the report and handoff changed.

### 12.1 Active detector class names / count

```text
class_count = 1
class_names = {0: building}
source      = artifacts/task6m_yolo_native/data.yaml (name: task6m_yolo_native,
              path: artifacts/task6m_yolo_native, train: images/train, val: images/val, test: images/test,
              names: 0: building)
```

The run-local `data.yaml` copies are **NOT PRESENT** in the run directories, so the dataset YAML above is the exact
class definition source.

### 12.2 Stage-1 / stage-2 training settings (parsed)

| field | stage 1 (`task6m/runs/m1_yolo26m_seg`) | stage 2 (`task6m1/runs/m1_yolo26m_seg_continued`) |
|---|---|---|
| model / source checkpoint | `artifacts/checkpoints/task6m/pretrained/yolo26m-seg.pt` | `…/m1_yolo26m_seg_continued/weights/last_resume.pt` |
| data | `artifacts/task6m_yolo_native/data.yaml` | `artifacts/task6m_yolo_native/data.yaml` |
| epochs | 80 | 80 |
| imgsz | 640 | 640 |
| batch | 16 | 16 |
| device | `0` (CUDA) | `0` (CUDA) |
| seed | 20260812 | 20260812 |
| resume | false | `…/weights/last_resume.pt` |
| pretrained | true | true |
| optimizer | auto | auto |
| close_mosaic | 10 | 10 |
| patience | 15 | 15 |
| name / save_dir | `m1_yolo26m_seg` | `m1_yolo26m_seg_continued` |

Run metrics present in both runs (`results.csv`): stage 1 stopped after **18** logged epochs with final
`metrics/mAP50(B)=0.69842`, `mAP50-95(B)=0.39715`, `mAP50(M)=0.68266`, `mAP50-95(M)=0.35437`; stage 2 logged **55**
epochs with final `mAP50(B)=0.73888`, `mAP50-95(B)=0.44319`, `mAP50(M)=0.72415`, `mAP50-95(M)=0.39172`. These are
quantitative validation-split results of the **active** lineage (not of any alternate).

### 12.3 `baseline/yolo_whu` exact relation

```text
SEPARATE_HISTORICAL_BASELINE
```

Evidence: `baseline/yolo_whu/README.md` — "Baseline: YOLOv8m-seg on WHU Building Dataset", "**Status: FROZEN.
Read-only reference. Not part of the proposed method.**", "is NOT … an inference-time component of the proposed
pipeline", "The proposed method must not import any YOLO code"; `artifacts/task6j_yolo_proposals/provenance.json` —
`model_sha256 = d9a6a65b7e0819ce4ecbbd9d44a5c8f9dcd2e60ea78203ba8fdf90ba6aaa1f91`, trained **100 epochs** on the
legacy WHU YOLO dataset, invoked read-only with ultralytics 8.4.67 from a separate legacy project
(`C:\D\resources\project\WHU_Building_Segment`). The active detector is YOLO26m-seg
(`ef852b58…`) with its own lineage (pretrained `yolo26m-seg.pt` → `task6m` → `task6m1`). Different family, different
checkpoint hash and no lineage evidence ⇒ neither `SAME_CHECKPOINT` nor `DIRECT_ANCESTOR_OR_FINETUNE_SOURCE`.

### 12.4 A2 non-model pixel statistics

```text
file            = inference/input/A2.png · 1 677 040 bytes · PNG · RGB · 1024x1024
sha256          = 10286b1e76db9e38c474635a465c9e677dbcf58375c1d39f7b742eeb991f434f
array dtype     = uint8 (8-bit unsigned per channel, mode RGB)
R: min 23  max 148  mean 60.0654  std 6.3738  frac_eq_0 0.000000  frac_eq_255 0.000000
G: min 29  max 159  mean 68.4044  std 6.4936  frac_eq_0 0.000000  frac_eq_255 0.000000
B: min 30  max 151  mean 66.4532  std 6.9203  frac_eq_0 0.000000  frac_eq_255 0.000000
unique RGB colours = 15 211
luminance       = mean 64.9744  std 6.3520  frac<16 0.000000  frac>240 0.000000
```

(Statistics at 4-decimal precision, computed with numpy on the decoded uint8 array; no model call.)

### 12.5 Training domain vs A2 — comparison table

| dimension | training/validation domain (active detector) | A2 |
|---|---|---|
| detector family/task | YOLO26m-seg instance segmentation | same detector applied |
| dataset | `artifacts/task6m_yolo_native/data.yaml` (WHU building lineage, `datasets/whu`) | NOT ESTABLISHED |
| classes | 1 class `building` | unknown building content distribution |
| training tile geometry | WHU native tiles, training `imgsz=640`, batch 16 | 1024×1024 RGB PNG |
| radiometry | NOT ESTABLISHED (no dataset pixel statistics recorded) | mean RGB ≈ (60.1, 68.4, 66.5), per-channel std ≈ 6.4–6.9, no saturated pixels, 15 211 unique colours |
| geographic/sensor provenance | NOT ESTABLISHED | NOT ESTABLISHED |
| A2 membership in training/val/test splits | — | NOT ESTABLISHED |
| validation metrics | mAP50(B) 0.73888 / mAP50-95(B) 0.44319 / mAP50(M) 0.72415 / mAP50-95(M) 0.39172 (active lineage) | not measured (no A2/RC1 validation exists) |

The comparison cannot establish either a domain mismatch or in-domain membership, because the dataset's radiometric
and geographic descriptors are not recorded and A2's provenance is unknown.

### 12.6 Alternate detector validation status (normalized)

| candidate | family | bytes / sha256 prefix | normalized validation status |
|---|---|---:|---|
| epoch-18 YOLO26m-seg (`task6m/runs/m1_yolo26m_seg/weights/best.pt` = `task6m1/source_epoch18_snapshot/weights/best.pt`) | YOLO26m-seg | 162 481 487 / `fd407db634a8a7ef…` | **project validation evidence**: full `results.csv` metrics on the WHU validation split (final mAP50(M) 0.68266), and the run that the active continuation resumed from; **not** validated on A2/RC1 |
| YOLOv8m-seg WHU baseline (`WHU_Building_Segment/runs/segment/logs/whu_building_v1/weights/best.pt`, recorded as `baseline/yolo_whu/`) | YOLOv8m-seg | 54 835 548 / `d9a6a65b7e0819ce…` | **project validation evidence present in the frozen record** (`baseline/yolo_whu/run_record/results.csv`, 17 463 bytes of 100-epoch metrics) plus Task 6J read-only proposal-count runs; **not** validated on A2/RC1; README keeps it reference-only |
| pretrained base `task6m/pretrained/yolo26m-seg.pt` | YOLO26m-seg | 54 750 385 / `16b636f04e8fb6a3…` | **NOT project validation** — pretrained base only |
| pretrained base `task6m/pretrained/yolo26s-seg.pt` | YOLO26s-seg | 23 467 933 / `3da1d83e31caec96…` | **NOT project validation** — used only in smoke runs |

Overall enum (unchanged, now evidence-normalized):

```text
VALIDATED_ALTERNATE_AVAILABLE
```

Per the task book, this does **not** authorize adopting any alternate into RC1.

### 12.7 Scientific-freeze impact and permitted diagnostic use

```text
DETECTOR_CHANGE_TOUCHES_FROZEN_RESEARCH_CLAIMS
```

A. adopting/replacing the RC1 proposal detector would: contradict `model/buildreasonseg_advisor/metadata.json`, which
lists the detector as frozen RC1 architecture ("YOLO26m-seg (frozen U-C1 proposal model)") under a research baseline
whose `task7j_test_status = FINAL_TEST_CONSUMED`; it would also change the proposal source underlying the frozen
Task 6M/6M1/7 results, so the frozen research claims would need documented re-validation and a re-freeze before any
adoption.

```text
B. merely running a future isolated alternate-detector A2 diagnostic, without changing RC1 or research claims, would:
   ALLOWED_AS_SEPARATE_DIAGNOSTIC
```

B is permitted because such a one-off diagnostic leaves the delivered detector, the product configuration and the
frozen research artifacts untouched; it produces diagnostic evidence only and authorizes no adoption.

### 12.8 Re-evaluated primary diagnosis and next gate

```text
primary diagnosis = PROP01_DOMAIN_GAP_INSUFFICIENT_EVIDENCE
NEXT              = DOMAIN_EVIDENCE_RECOVERY
```

Unchanged: no new direct evidence establishes A2's domain (§12.4–§12.5), so the classification is not strengthened;
the next-gate logic then falls through the alternate branch (available but not scientifically compatible without
re-validation, §12.7A) and the domain-gap / in-domain-blind-spot / provenance-incomplete branches, leaving
`DOMAIN_EVIDENCE_RECOVERY`. Neither is executed here.
