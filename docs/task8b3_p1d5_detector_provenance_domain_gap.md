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
classification = VALIDATED_ALTERNATE_AVAILABLE
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
