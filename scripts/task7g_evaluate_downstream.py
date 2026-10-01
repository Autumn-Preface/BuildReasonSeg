"""Task 7G Parts I-J — frozen D-B1 external downstream audit (guarded by the section-23/24 stages).

Section 27 chain per `E-HoldoutL3` record:

```text
canonical L3 program -> U-C1 proposals -> G-S0 or G-S1 largest reference selector
-> selected predicted reference mask -> P_dir v0.2 -> P_near v0.1 -> frozen SAM2 feature
-> frozen D-B1 -> target mask
```

No parser; no GT inference input. Section 28 requires G-S0 to reproduce Task 7F F-R0 (strict
`0.24540501038500215`, answered-only `0.24614085749260334`, abstentions `2`) and section 29 requires G-S0 to
reproduce the Task 7F F-R0 paired result (6/20, margin `0.1250628820`).

The script refuses to run when the section-23 internal gate did not pass, so no downstream artifact is
fabricated after a STOP.

    python scripts/task7g_evaluate_downstream.py
"""

from __future__ import annotations

import argparse
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
from buildreasonseg_mvp.task7e_l3_decoder_adapter import FrozenL3Decoder, frozen_metadata  # noqa: E402
from buildreasonseg_mvp.task7g_largest_reference_selector import (  # noqa: E402
    SetContextLargestSelector,
    deterministic_max_area,
    proposal_feature_matrix,
    select_with_scores,
)
from scripts.task6n_train import MaskStore  # noqa: E402
from scripts.task7e_evaluate_oracle import read_holdout_rows  # noqa: E402
from scripts.task7e_evaluate_predicted_reference import _ReferenceOverride  # noqa: E402
from buildreasonseg_mvp.task7f_reference_ceiling import (  # noqa: E402
    YOLO_SHA256,
    dice_of,
    iou_of,
    sha256_file,
)

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task7g_external_downstream.json"
INTERNAL = EVAL / "task7g_internal_holdout.json"
EXTERNAL_REFERENCE = EVAL / "task7g_external_reference.json"
TASK7F_DOWNSTREAM = EVAL / "task7f_downstream_reference_modes.json"
CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task7g" / "largest_set_context_selector_v1.pt"
TOLERANCE = 1.0e-6
DIRECTIONS = ("above", "below", "left", "right")
EXPECTED = {"strict_miou": 0.24540501038500215, "answered_only_miou": 0.24614085749260334,
            "abstentions": 2}


def summarise(rows: list[dict]) -> dict:
    answered = [row for row in rows if not row["abstained"]]
    reference_ok = [row for row in rows if not row["abstained"] and row["reference_iou"] >= 0.50]
    return {
        "records": len(rows),
        "miou": float(np.mean([row["miou"] for row in rows])),
        "dice": float(np.mean([row["dice"] for row in rows])),
        "precision_at_0_5": float(np.mean([row["precision_at_0_5"] for row in rows])),
        "answered": len(answered),
        "abstentions": len(rows) - len(answered),
        "answered_only_miou": float(np.mean([row["miou"] for row in answered]))
        if answered else None,
        "answered_only_dice": float(np.mean([row["dice"] for row in answered]))
        if answered else None,
        "reference_iou_at_least_0_5_subset_miou":
            float(np.mean([row["miou"] for row in reference_ok])) if reference_ok else None,
        "per_direction": {direction: {
            "records": sum(1 for row in rows if row["direction"] == direction),
            "miou": float(np.mean([row["miou"] for row in rows if row["direction"] == direction]))
            if any(row["direction"] == direction for row in rows) else None}
            for direction in DIRECTIONS},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--yolo-device", default="0")
    args = parser.parse_args(argv)
    started = time.time()

    internal = json.loads(INTERNAL.read_text(encoding="utf-8")) if INTERNAL.is_file() else None
    reference = (json.loads(EXTERNAL_REFERENCE.read_text(encoding="utf-8"))
                 if EXTERNAL_REFERENCE.is_file() else None)
    if not internal or not internal.get("gate_passed") or not reference \
            or not reference.get("executed"):
        write_json(OUT, {
            "_doc": ("Task 7G sections 27-28. Frozen D-B1 external downstream audit, conditional on the "
                     "section-23 internal gate and the section-24 external reference stage. Those did not "
                     "run, so the downstream stage did not run either."),
            "task": "7G", "stage": "I-external-downstream", "executed": False,
            "reason": "section 23 gate failed (LARGEST_SELECTOR_NOT_LEARNABLE); external stages not run",
            "internal_gate_passed": bool(internal and internal.get("gate_passed")),
            "training_performed": False, "test_split_used": False,
        })
        print("[7g.downstream] not executed: section 23 gate failed", flush=True)
        return 4

    from buildreasonseg_mvp.task6q_reference_resolver import is_eligible

    from buildreasonseg_mvp.task7a_l3_pipeline import L3Pipeline

    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = SetContextLargestSelector()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    records = read_holdout_rows()
    decoder = FrozenL3Decoder("D-B1", args.device)
    pipeline = L3Pipeline(device=args.device, yolo_device=args.yolo_device)
    store = pipeline.store
    masks = MaskStore()
    eligible_cache: dict[str, list] = {}

    def eligible_of(tile_id: str, image_path) -> list:
        if tile_id not in eligible_cache:
            proposals = pipeline.proposals(tile_id, Path(image_path))
            eligible_cache[tile_id] = [proposal for proposal in proposals
                                       if is_eligible(proposal, "largest")]
        return eligible_cache[tile_id]

    def predict_with(reference_mask: np.ndarray, record: dict) -> np.ndarray:
        synthetic_id = -5
        overridden = dict(record)
        overridden["reference_source_feature_id"] = synthetic_id
        overridden["sample_id"] = f"{record['sample_id']}:g"
        wrapper = _ReferenceOverride(masks, {(record["tile_id"], synthetic_id): reference_mask})
        return decoder.predict([overridden], store, wrapper, {}, batch_size=1)[0]

    rows = {"G-S0": [], "G-S1": []}
    for record in records:
        eligible = eligible_of(record["tile_id"], record["image_path"])
        gt_reference = np.asarray(masks.mask(record["tile_id"],
                                             record["reference_source_feature_id"]), dtype=bool)
        gt_target = np.asarray(masks.mask(record["tile_id"],
                                          record["target_source_feature_id"]), dtype=bool)
        if not eligible:
            for mode in ("G-S0", "G-S1"):
                rows[mode].append({"sample_id": record["sample_id"], "mode": mode,
                                   "direction": record["direction"], "miou": 0.0, "dice": 0.0,
                                   "precision_at_0_5": 0.0, "abstained": True, "reference_iou": 0.0})
            continue
        features = torch.as_tensor(proposal_feature_matrix(eligible))
        with torch.no_grad():
            scores = model(features).numpy()
        selected = {"G-S0": deterministic_max_area(eligible),
                    "G-S1": select_with_scores(scores, eligible)}
        for mode, index in selected.items():
            mask = np.asarray(eligible[index].mask, dtype=bool)
            prediction = predict_with(mask, record)
            intersection = float((prediction & gt_target).sum())
            rows[mode].append({
                "sample_id": record["sample_id"], "mode": mode,
                "direction": record["direction"],
                "miou": (intersection + TOLERANCE)
                / (float((prediction | gt_target).sum()) + TOLERANCE),
                "dice": dice_of(prediction, gt_target),
                "precision_at_0_5": (intersection + TOLERANCE)
                / (float(prediction.sum()) + TOLERANCE),
                "abstained": False, "reference_iou": iou_of(mask, gt_reference)})

    results = {mode: summarise(rows[mode]) for mode in ("G-S0", "G-S1")}
    frozen = json.loads(TASK7F_DOWNSTREAM.read_text(encoding="utf-8"))["results"]["F-R0"]["strict"]
    reproduction = {
        "strict_miou": {"measured": results["G-S0"]["miou"], "expected": EXPECTED["strict_miou"],
                        "delta": abs(results["G-S0"]["miou"] - EXPECTED["strict_miou"]),
                        "passed": abs(results["G-S0"]["miou"] - EXPECTED["strict_miou"]) <= TOLERANCE},
        "answered_only_miou": {"measured": results["G-S0"]["answered_only_miou"],
                               "expected": EXPECTED["answered_only_miou"],
                               "delta": abs(results["G-S0"]["answered_only_miou"]
                                            - EXPECTED["answered_only_miou"]),
                               "passed": abs(results["G-S0"]["answered_only_miou"]
                                             - EXPECTED["answered_only_miou"]) <= TOLERANCE},
        "abstentions": {"measured": results["G-S0"]["abstentions"],
                        "expected": EXPECTED["abstentions"],
                        "passed": results["G-S0"]["abstentions"] == EXPECTED["abstentions"]},
    }
    reproduction["passed"] = all(entry["passed"] for entry in reproduction.values()
                                 if isinstance(entry, dict))
    write_json(OUT, {
        "_doc": ("Task 7G section 28. Frozen D-B1 external downstream audit for the deterministic G-S0 and "
                 "learned G-S1 largest-reference selections over the frozen E-HoldoutL3 population. The only "
                 "difference between the two runs is the selected predicted reference mask."),
        "task": "7G", "stage": "I-external-downstream", "executed": True,
        "checkpoints": {**frozen_metadata(), "selector": {"path": str(CHECKPOINT),
                                                          "sha256": sha256_file(CHECKPOINT)}},
        "yolo_sha256": sha256_file(
            __import__("scripts.task6u_common", fromlist=["PROPOSAL_CHECKPOINT"]).PROPOSAL_CHECKPOINT),
        "yolo_sha256_expected": YOLO_SHA256,
        "records": len(records), "results": results, "task7f_f_r0_reference": frozen,
        "reproduction": reproduction, "reproduction_passed": reproduction["passed"],
        "selection_gain": results["G-S1"]["miou"] - results["G-S0"]["miou"],
        "training_performed": False, "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    })
    print(f"[7g.downstream] G-S0 strict {results['G-S0']['miou']:.6f} (Δ"
          f"{reproduction['strict_miou']['delta']:.2e}) | G-S1 strict {results['G-S1']['miou']:.6f} | "
          f"gain {results['G-S1']['miou'] - results['G-S0']['miou']:+.4f} | "
          f"reproduction {'PASS' if reproduction['passed'] else 'FAIL'}", flush=True)
    return 0 if reproduction["passed"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
