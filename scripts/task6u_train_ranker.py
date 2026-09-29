"""Task 6U Part G — train ProposalSetRanker v0.1 on the train-only U-RankerTrain split.

Training examples come from the **selected frozen proposal configuration** (U-C1) over U-RankerTrain:

* keep only the eligible proposals of each record's own family;
* best eligible IoU vs the GT reference < 0.50 → the record is excluded from ranker-loss training and
  counted as `untrainable_not_covered` (the ranker cannot create missing proposals);
* otherwise the target class is the eligible proposal with the highest GT IoU (ties: higher YOLO
  confidence, then lower original proposal index).

GT is used only to build training labels; inference never sees it.

Internal 90/10 split of the ranker-trainable records by reference-key hash (seed 20260930, stratified by
family) is the **only** set used for checkpoint selection. AdamW, lr 1e-3, weight decay 1e-4, batch 64
reference sets, max 50 epochs, early-stopping patience 6, selection metric = internal-holdout top-1
selection accuracy, tie-break mean selected-reference IoU. No hyperparameter sweep.

Checkpoint: `artifacts/checkpoints/task6u/reference_ranker_v01.pt` (gitignored).
Writes `evaluation/task6u_ranker_training.json`.

    python scripts/task6u_train_ranker.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.task6q_reference_resolver import is_eligible  # noqa: E402
from buildreasonseg_mvp.task6u_reference_ranker import (  # noqa: E402
    FEATURE_DIM,
    ProposalSetRanker,
    proposal_features,
    score_sets,
    set_logits,
)
from scripts.task6u_common import (  # noqa: E402
    CONFIGS,
    EVAL,
    PROPOSAL_CHECKPOINT,
    PROPOSAL_CHECKPOINT_SHA256,
    gt_reference_mask,
    group_records_by_tile,
    iou,
    proposals_for_tile,
    record_image_path,
    sha256_file,
)

OUT = EVAL / "task6u_ranker_training.json"
OUT_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6u" / "reference_ranker_v01.pt"
SPLIT = EVAL / "task6u_reference_train_split.json"
SELECTED = EVAL / "task6u_selected_proposal_config.json"
SEED = 20260930
COVERED_IOU = 0.50
HOLDOUT_FRACTION = 0.10


def key_hash(record: dict) -> str:
    key = "|".join((str(record["split"]), str(record["tile_id"]),
                    str(record["reference_source_feature_id"]), str(record["reference_family"])))
    return hashlib.sha256(f"{SEED}:{key}".encode("utf-8")).hexdigest()


def build_examples(records: list[dict], model, config: dict, device: str) -> tuple[list[dict], dict]:
    """Eligible-proposal feature sets with GT-derived positive labels (train-side only)."""

    grouped = group_records_by_tile(records)
    examples: list[dict] = []
    stats = Counter()
    for tile_id, tile_records in grouped.items():
        image_path = record_image_path(tile_records[0])
        proposals, _run = proposals_for_tile(model, config, tile_id, image_path, device,
                                            use_cache=True)
        for record in tile_records:
            family = str(record["reference_family"])
            eligible = [proposal for proposal in proposals if is_eligible(proposal, family)]
            if not eligible:
                stats["no_eligible"] += 1
                examples.append({"record": record, "trainable": False, "reason": "no_eligible",
                                 "features": np.zeros((0, FEATURE_DIM), dtype=np.float32)})
                continue
            truth = gt_reference_mask(record)
            ious = [iou(proposal.mask, truth) for proposal in eligible]
            best = float(max(ious))
            if best < COVERED_IOU:
                stats["untrainable_not_covered"] += 1
                examples.append({"record": record, "trainable": False,
                                 "reason": "untrainable_not_covered", "best_eligible_iou": best,
                                 "features": proposal_features(eligible, family)})
                continue
            # target: highest GT IoU, ties -> higher confidence, then lower original index
            order = sorted(range(len(eligible)),
                           key=lambda index: (-ious[index], -eligible[index].confidence,
                                              eligible[index].index))
            stats["trainable"] += 1
            examples.append({
                "record": record, "trainable": True, "family": family,
                "features": proposal_features(eligible, family), "target": int(order[0]),
                "eligible_count": len(eligible), "best_eligible_iou": best,
                "gt_ious": ious,
                "proposal_indices": [int(proposal.index) for proposal in eligible],
            })
    return examples, dict(stats)


def split_examples(examples: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deterministic 90/10 by reference-key hash, stratified by family."""

    by_family: dict[str, list[dict]] = defaultdict(list)
    for example in examples:
        by_family[example["family"]].append(example)
    train, holdout = [], []
    for family in sorted(by_family):
        ordered = sorted(by_family[family], key=lambda example: key_hash(example["record"]))
        cut = max(1, int(round(len(ordered) * HOLDOUT_FRACTION))) if ordered else 0
        holdout.extend(ordered[:cut])
        train.extend(ordered[cut:])
    return train, holdout


def evaluate_examples(model: ProposalSetRanker, examples: list[dict], device: str) -> dict:
    """Top-1 selection accuracy and mean selected-reference IoU on a labelled example set."""

    if not examples:
        return {"records": 0, "top1_accuracy": None, "mean_selected_iou": None}
    logits, mask = set_logits(model, [example["features"] for example in examples], device=device)
    correct, selected_ious = 0, []
    for index, example in enumerate(examples):
        scores = logits[index][mask[index]]
        choice = int(np.argmax(scores))
        correct += int(choice == example["target"])
        selected_ious.append(float(example["gt_ious"][choice]))
    return {
        "records": len(examples),
        "top1_accuracy": correct / len(examples),
        "mean_selected_iou": float(np.mean(selected_ious)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--torch-device", default="cuda")
    args = parser.parse_args(argv)
    started = time.time()

    checkpoint_sha = sha256_file(PROPOSAL_CHECKPOINT) if PROPOSAL_CHECKPOINT.is_file() else None
    if checkpoint_sha != PROPOSAL_CHECKPOINT_SHA256:
        write_json(OUT, {"_doc": "Task 6U section 16.", "task": "6U",
                         "verdict": "PROPOSAL_CHECKPOINT_UNAVAILABLE"})
        return 2

    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    config = CONFIGS[selected["selected_config"]]
    ranker_records = split["u_rankertrain"].get("records")
    if ranker_records is None:
        # the split artifact stores counts and hashes for U-RankerTrain; rebuild the records from the
        # frozen Task 6P pack by excluding the frozen U-Calib200 keys
        from scripts.task6u_freeze_reference_split import unique_key

        packs = json.loads((REPO_ROOT / "artifacts" / "task6p" / "reference_packs"
                            / "ref_train_unique.json").read_text(encoding="utf-8"))["records"]
        calib_keys = {unique_key(record) for record in split["u_calib200"]["records"]}
        ranker_records = [record for record in packs if unique_key(record) not in calib_keys]
        assert len(ranker_records) == split["u_rankertrain"]["count"], len(ranker_records)
    print(f"[6u.ranker] config {config['id']} (imgsz {config['imgsz']}, conf {config['conf']}) over "
          f"{len(ranker_records)} U-RankerTrain references", flush=True)

    from ultralytics import YOLO

    yolo = YOLO(str(PROPOSAL_CHECKPOINT))
    examples, build_stats = build_examples(ranker_records, yolo, config, args.device)
    trainable = [example for example in examples if example["trainable"]]
    train, holdout = split_examples(trainable)
    print(f"[6u.ranker] examples {len(examples)}: trainable {len(trainable)} "
          f"(train {len(train)} / holdout {len(holdout)}), stats {build_stats}", flush=True)

    torch.manual_seed(SEED)
    model = ProposalSetRanker().to(args.torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    if args.torch_device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    best = {"key": (-1.0, -1.0), "epoch": 0, "holdout": None}
    history = []
    stale = 0
    for epoch in range(1, 51):
        model.train()
        generator = np.random.default_rng(SEED + epoch)
        order = generator.permutation(len(train))
        losses = []
        for start in range(0, len(order), 64):
            batch = [train[int(index)] for index in order[start: start + 64]]
            if not batch:
                continue
            logits_tensor, mask_tensor = score_sets(
                model, [example["features"] for example in batch],
                device=args.torch_device, requires_grad=True)
            logits_tensor = logits_tensor.masked_fill(~mask_tensor, float("-inf"))
            targets = torch.as_tensor([example["target"] for example in batch],
                                      device=args.torch_device)
            loss = torch.nn.functional.cross_entropy(logits_tensor, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        holdout_metrics = evaluate_examples(model, holdout, args.torch_device)
        key = (holdout_metrics["top1_accuracy"] or 0.0, holdout_metrics["mean_selected_iou"] or 0.0)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None,
                        "holdout_top1_accuracy": holdout_metrics["top1_accuracy"],
                        "holdout_mean_selected_iou": holdout_metrics["mean_selected_iou"]})
        print(f"[6u.ranker] epoch {epoch}: loss {np.mean(losses):.4f} holdout top1 "
              f"{holdout_metrics['top1_accuracy']:.4f} mean IoU "
              f"{holdout_metrics['mean_selected_iou']:.4f}", flush=True)
        if key > best["key"]:
            best = {"key": key, "epoch": epoch, "holdout": holdout_metrics}
            OUT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "feature_dim": FEATURE_DIM,
                        "architecture": model.architecture_report(), "epoch": epoch,
                        "config": config["id"], "seed": SEED}, OUT_CHECKPOINT)
            stale = 0
        else:
            stale += 1
            if stale >= 6:
                print(f"[6u.ranker] early stop at epoch {epoch}", flush=True)
                break

    if OUT_CHECKPOINT.is_file():
        model.load_state_dict(torch.load(OUT_CHECKPOINT, map_location=args.torch_device,
                                         weights_only=False)["state_dict"])
    final_holdout = evaluate_examples(model, holdout, args.torch_device)
    final_train = evaluate_examples(model, train, args.torch_device)

    payload = {
        "_doc": (
            "Task 6U sections 13-16. ProposalSetRanker v0.1 trained only on the train-side "
            "U-RankerTrain split using the frozen selected proposal configuration (U-C1). It addresses "
            "REFERENCE_SELECTION_WRONG only: records whose best eligible proposal is below IoU 0.50 are "
            "excluded from the loss and counted as untrainable_not_covered. GT builds training labels "
            "only; inference never uses GT. Support infrastructure, not a claimed novelty."
        ),
        "task": "6U", "stage": "G-ranker-training",
        "proposal_config": {"id": config["id"], "imgsz": config["imgsz"], "conf": config["conf"],
                            "max_det": config["max_det"],
                            "checkpoint_sha256": checkpoint_sha},
        "selection_scope": "REFERENCE_SELECTION_WRONG only; the ranker cannot create missing proposals",
        "data": {
            "split": str(SPLIT), "records": len(ranker_records),
            "by_family": dict(Counter(record["reference_family"] for record in ranker_records)),
            "examples": len(examples),
            "trainable": len(trainable),
            "untrainable_not_covered": build_stats.get("untrainable_not_covered", 0),
            "no_eligible": build_stats.get("no_eligible", 0),
            "internal_train": len(train), "internal_holdout": len(holdout),
            "coverage_threshold": COVERED_IOU,
            "labeling": "target = eligible proposal with highest GT IoU; ties -> higher confidence -> "
                        "lower original proposal index",
        },
        "feature_vector": {
            "dim": FEATURE_DIM, "names": model.architecture_report()["feature_names"],
            "family_one_hot": {"largest": [1, 0], "smallest": [0, 1]},
            "excluded": ["GT-derived features", "target relation", "centroid x/y", "image location",
                         "SAM2 features", "target mask", "source feature id"],
        },
        "architecture": model.architecture_report(),
        "optimizer": {"name": "AdamW", "lr": 1.0e-3, "weight_decay": 1.0e-4, "batch_sets": 64,
                      "max_epochs": 50, "early_stopping_patience": 6, "seed": SEED,
                      "sweep": False,
                      "selection_metric": ["internal holdout top-1 accuracy",
                                           "tie-break mean selected-reference IoU"]},
        "history": history,
        "selected_epoch": best["epoch"],
        "internal_holdout": final_holdout,
        "internal_train_final": final_train,
        "checkpoint": {
            "path": str(OUT_CHECKPOINT), "exists": OUT_CHECKPOINT.is_file(),
            "sha256": sha256_file(OUT_CHECKPOINT) if OUT_CHECKPOINT.is_file() else None,
            "committed": False,
            "wall_seconds": round(time.time() - started, 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
            if args.torch_device != "cpu" else None,
        },
        "test_split_used": False,
        "refval_used_for_training": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    payload["verdict"] = "RANKER_TRAINED" if payload["checkpoint"]["exists"] else "RANKER_TRAINING_FAILED"
    write_json(OUT, payload)
    print(f"[6u.ranker] selected epoch {best['epoch']} holdout top1 {final_holdout['top1_accuracy']:.4f}"
          f" mean IoU {final_holdout['mean_selected_iou']:.4f} -> {payload['verdict']}", flush=True)
    return 0 if payload["verdict"] == "RANKER_TRAINED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
