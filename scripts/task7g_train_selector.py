"""Task 7G Parts E-G — train `SetContextLargestSelector v1` and run the internal learnability gate.

Internal split (section 20): tile-level 80/20 by SHA256 with seed 20261001, applied to
`G-RefTrainUniqueLargest` before any proposal-set row enters train/holdout, with zero tile overlap, zero
`(tile_id, reference_source_feature_id)` overlap and >= 50 holdout tiles. Uncovered / no-eligible records stay
counted in their partition but are excluded from the loss and the accuracy denominators.

Optimization (section 21): fresh initialization, AdamW lr 1e-3, weight decay 1e-4, batch 32 proposal sets,
max 40 epochs, early-stopping patience 6, seed 20261001, no scheduler, no augmentation, FP32, no sweep. The one
loss is the listwise `CrossEntropyLoss` on the oracle-best candidate index.

Checkpoint selection uses the train-internal holdout only (never E-HoldoutL3), ranked by
(1) highest mean selected-reference GT IoU, (2) higher oracle-best exact top-1 accuracy, (3) lower mean
`best_iou - selected_iou`, (4) earlier epoch. Checkpoint:
`artifacts/checkpoints/task7g/largest_set_context_selector_v1.pt` (gitignored).

Internal gate (section 23) — continue only if G-I1 mean selected IoU >= G-I0 + 0.08, >= 0.62, oracle-best exact
top-1 >= 0.55 and mean gap <= 0.14, otherwise STOP `LARGEST_SELECTOR_NOT_LEARNABLE`.

Writes `evaluation/task7g_training.json` and `evaluation/task7g_internal_holdout.json`.

    python scripts/task7g_train_selector.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.task7g_largest_reference_selector import (  # noqa: E402
    SetContextLargestSelector,
    select_with_scores,
    selector_report,
)

EVAL = REPO_ROOT / "evaluation"
ROWS_PATH = REPO_ROOT / "artifacts" / "task7g" / "selector_dataset" / "selector_rows.jsonl"
CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task7g" / "largest_set_context_selector_v1.pt"
MANIFEST = EVAL / "task7g_training_dataset_manifest.json"
OUT_TRAINING = EVAL / "task7g_training.json"
OUT_INTERNAL = EVAL / "task7g_internal_holdout.json"
SEED = 20261001
TRAIN_TILE_FRACTION = 0.80
BATCH = 32
MAX_EPOCHS = 40
PATIENCE = 6
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-4
INTERNAL_GATE = {"mean_gain": 0.08, "mean_selected_iou": 0.62, "oracle_top1": 0.55,
                 "mean_gap_max": 0.14}
MIN_HOLDOUT_TILES = 50


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows() -> list[dict]:
    rows = []
    with ROWS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def tile_assignment(tile_id: str) -> str:
    digest = hashlib.sha256(f"{SEED}:{tile_id}".encode("utf-8")).hexdigest()
    return "train" if int(digest[:8], 16) / 0xFFFFFFFF < TRAIN_TILE_FRACTION else "holdout"


def partition(rows: list[dict]) -> dict[str, list[dict]]:
    partitions = {"train": [], "holdout": []}
    for row in rows:
        partitions[tile_assignment(row["tile_id"])].append(row)
    return partitions


def deterministic_index(row: dict) -> int:
    """The frozen G-I0 / G-S0 rule recomputed from the stored candidate statistics."""

    statistics = row["candidate_statistics"]
    return max(range(len(statistics)),
               key=lambda position: (int(statistics[position]["area"]),
                                     float(statistics[position]["confidence"]),
                                     -int(statistics[position]["index"])))


@torch.no_grad()
def selector_index(model, row: dict) -> int:
    """G-I1 selection: highest score, exact-float tie -> higher confidence, then lower index."""

    model.eval()
    features = torch.as_tensor(np.asarray(row["features"], dtype=np.float32))
    scores = model(features).cpu().numpy()
    statistics = row["candidate_statistics"]
    order = sorted(range(len(scores)),
                   key=lambda position: (-float(scores[position]),
                                         -float(statistics[position]["confidence"]),
                                         int(statistics[position]["index"])))
    return order[0]


def summarise_rows(rows: list[dict], indices: list[int], *, label: str) -> dict:
    selected = np.asarray([float(row["candidate_ious"][index])
                           for row, index in zip(rows, indices)], dtype=np.float64)
    best = np.asarray([float(row["best_eligible_iou"]) for row in rows], dtype=np.float64)
    top1 = np.asarray([1.0 if index == int(row["target_index"]) else 0.0
                       for row, index in zip(rows, indices)], dtype=np.float64)
    if not rows:
        return {"records": 0, "label": label}
    return {
        "records": len(rows), "label": label,
        "mean_selected_iou": float(selected.mean()),
        "median_selected_iou": float(np.median(selected)),
        "selected_iou_at_0_5": float((selected >= 0.50).mean()),
        "oracle_best_top1": float(top1.mean()),
        "mean_best_minus_selected": float((best - selected).mean()),
        "mean_best_eligible_iou": float(best.mean()),
    }


def evaluate_partition(model, rows: list[dict]) -> dict:
    deterministic = summarise_rows(rows, [deterministic_index(row) for row in rows],
                                   label="G-I0 deterministic max-area")
    learned = summarise_rows(rows, [selector_index(model, row) for row in rows],
                             label="G-I1 learned selector")
    return {"g_i0_deterministic_max_area": deterministic, "g_i1_learned_selector": learned}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    started = time.time()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["verdict"] != "SELECTOR_TRAIN_DATA_READY":
        write_json(OUT_TRAINING, {"_doc": "Task 7G section 9.", "task": "7G",
                                  "verdict": "SELECTOR_TRAIN_DATA_INSUFFICIENT"})
        print("[7g.train] STOP SELECTOR_TRAIN_DATA_INSUFFICIENT", flush=True)
        return 3
    rows = read_rows()
    trainable = [row for row in rows if row["state"] == "trainable"]
    partitions = partition(trainable)
    train_rows, holdout_rows = partitions["train"], partitions["holdout"]
    train_tiles = {row["tile_id"] for row in train_rows}
    holdout_tiles = {row["tile_id"] for row in holdout_rows}
    overlap = train_tiles & holdout_tiles

    if not holdout_rows or len(holdout_tiles) < MIN_HOLDOUT_TILES or overlap:
        write_json(OUT_TRAINING, {"_doc": "Task 7G section 20.", "task": "7G",
                                  "verdict": "SELECTOR_TRAIN_DATA_INSUFFICIENT",
                                  "holdout_tiles": len(holdout_tiles), "tile_overlap": len(overlap)})
        print(f"[7g.train] STOP SELECTOR_TRAIN_DATA_INSUFFICIENT (holdout tiles {len(holdout_tiles)}, "
              f"overlap {len(overlap)})", flush=True)
        return 3

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = SetContextLargestSelector().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    generator = np.random.default_rng(SEED)

    history = []
    best = {"key": (-1.0, -1.0, -1.0, 0), "epoch": 0, "state": None, "metrics": None}
    stale = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = generator.permutation(len(train_rows))
        losses = []
        for start in range(0, len(order), BATCH):
            chunk = [train_rows[int(index)] for index in order[start: start + BATCH]]
            batch_loss = torch.zeros((), device=args.device)
            for row in chunk:
                features = torch.as_tensor(np.asarray(row["features"], dtype=np.float32),
                                           device=args.device)
                scores = model(features)
                target = torch.as_tensor([int(row["target_index"])], device=args.device)
                batch_loss = batch_loss + torch.nn.functional.cross_entropy(scores.unsqueeze(0),
                                                                            target)
            batch_loss = batch_loss / max(1, len(chunk))
            optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            optimizer.step()
            losses.append(float(batch_loss.detach()))
        if not losses:
            break
        metrics = evaluate_partition(model, holdout_rows)
        learned = metrics["g_i1_learned_selector"]
        oracle_top1 = learned["oracle_best_top1"]
        mean_iou = learned["mean_selected_iou"]
        gap = learned["mean_best_minus_selected"]
        key = (mean_iou, oracle_top1, -gap, -epoch)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "holdout_g_i1": learned,
                        "holdout_g_i0": metrics["g_i0_deterministic_max_area"]})
        print(f"[7g.train] epoch {epoch}: loss {np.mean(losses):.4f} | holdout mean selected IoU "
              f"{mean_iou:.4f} top1 {oracle_top1:.4f} gap {gap:.4f}", flush=True)
        if key > best["key"]:
            best = {"key": key, "epoch": epoch, "metrics": metrics,
                    "state": {name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"[7g.train] early stop at epoch {epoch}", flush=True)
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "seed": SEED, "epoch": best["epoch"],
                "architecture": "SetContextLargestSelector", "version": "v1",
                "metrics": best["metrics"]}, CHECKPOINT)

    holdout_metrics = evaluate_partition(model, holdout_rows)
    deterministic = holdout_metrics["g_i0_deterministic_max_area"]
    learned = holdout_metrics["g_i1_learned_selector"]

    gate = {
        "1_mean_gain": {"required": INTERNAL_GATE["mean_gain"],
                        "measured": learned["mean_selected_iou"] - deterministic["mean_selected_iou"],
                        "passed": (learned["mean_selected_iou"] - deterministic["mean_selected_iou"])
                        >= INTERNAL_GATE["mean_gain"]},
        "2_mean_selected_iou": {"required": INTERNAL_GATE["mean_selected_iou"],
                                "measured": learned["mean_selected_iou"],
                                "passed": learned["mean_selected_iou"]
                                >= INTERNAL_GATE["mean_selected_iou"]},
        "3_oracle_top1": {"required": INTERNAL_GATE["oracle_top1"],
                          "measured": learned["oracle_best_top1"],
                          "passed": learned["oracle_best_top1"] >= INTERNAL_GATE["oracle_top1"]},
        "4_mean_gap": {"required": f"<= {INTERNAL_GATE['mean_gap_max']}",
                       "measured": learned["mean_best_minus_selected"],
                       "passed": learned["mean_best_minus_selected"]
                       <= INTERNAL_GATE["mean_gap_max"]},
    }
    gate_passed = all(entry["passed"] for entry in gate.values())

    write_json(OUT_INTERNAL, {
        "_doc": ("Task 7G sections 22-23. Train-internal tile-disjoint holdout comparison of the deterministic "
                 "max-area selector (G-I0) and the frozen Task 7G selector (G-I1), with the section-23 "
                 "learnability gate. E-HoldoutL3 was not used for checkpoint selection or early stopping."),
        "task": "7G", "stage": "G-internal-holdout",
        "holdout": {"tiles": len(holdout_tiles), "trainable_sets": len(holdout_rows),
                    "train_tiles": len(train_tiles), "trainable_train_sets": len(train_rows),
                    "tile_overlap": len(overlap),
                    "reference_overlap": len({(row["tile_id"], row["reference_source_feature_id"])
                                              for row in train_rows}
                                             & {(row["tile_id"], row["reference_source_feature_id"])
                                                for row in holdout_rows}),
                    "excluded_from_loss": {"no_eligible": manifest["labels"]["counts"]["no_eligible"],
                                           "untrainable_not_covered":
                                               manifest["labels"]["counts"]["untrainable_not_covered"]}},
        "g_i0_deterministic_max_area": deterministic, "g_i1_learned_selector": learned,
        "gate": gate, "gate_constants": INTERNAL_GATE, "gate_passed": gate_passed,
        "selected_epoch": best["epoch"], "checkpoint_sha256": sha256_file(CHECKPOINT),
        "training_performed": True, "test_split_used": False,
        "external_holdout_used": False,
    })
    write_json(OUT_TRAINING, {
        "_doc": ("Task 7G section 21. Training of SetContextLargestSelector v1 on "
                 "G-RefTrainUniqueLargest with the exact frozen protocol and tile-level 80/20 internal split."),
        "task": "7G", "stage": "F-training", "seed": SEED,
        "dataset": {"rows": len(rows), "trainable": len(trainable),
                    "trainable_train": len(train_rows), "trainable_holdout": len(holdout_rows),
                    "train_tiles": len(train_tiles), "holdout_tiles": len(holdout_tiles),
                    "tile_assignment": "SHA256(seed:tile_id) < 0.80 -> train"},
        "protocol": {"optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY, "batch": BATCH,
                     "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "seed": SEED,
                     "scheduler": "none", "augmentation": "none", "precision": "fp32",
                     "sweep": False, "loss": "CrossEntropyLoss (listwise, single loss)"},
        "architecture": selector_report(), "history": history, "selected_epoch": best["epoch"],
        "selection_metrics": {"mean_selected_iou": best["key"][0], "oracle_best_top1": best["key"][1],
                              "mean_best_minus_selected": -best["key"][2]},
        "internal_holdout_metrics": holdout_metrics,
        "checkpoint": {"path": str(CHECKPOINT), "sha256": sha256_file(CHECKPOINT),
                       "bytes": CHECKPOINT.stat().st_size, "committed": False},
        "internal_gate_passed": gate_passed,
        "verdict": ("LARGEST_SELECTOR_TRAINED" if gate_passed
                    else "LARGEST_SELECTOR_NOT_LEARNABLE"),
        "training_performed": True, "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    })
    print(f"[7g.train] G-I0 mean selected IoU {deterministic['mean_selected_iou']:.4f} vs G-I1 "
          f"{learned['mean_selected_iou']:.4f} (gain "
          f"{learned['mean_selected_iou'] - deterministic['mean_selected_iou']:+.4f}) | top1 "
          f"{learned['oracle_best_top1']:.4f} | gap "
          f"{learned['mean_best_minus_selected']:.4f} -> "
          f"{'PASS' if gate_passed else 'FAIL'}", flush=True)
    return 0 if gate_passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
