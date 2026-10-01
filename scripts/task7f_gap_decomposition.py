"""Task 7F Parts G-I — covered-subset geometry decomposition, gap decomposition and diagnostic labels.

Part G: the COVERED50 subset (at least one eligible U-C1 proposal and best eligible IoU >= 0.50) — exactly the
non-abstaining population of F-R2 — is scored twice with the frozen D-B1:

* `G-PRED` uses the F-R1 oracle-selected **predicted** proposal mask;
* `G-GT` uses the canonical GT reference mask.

`geometry_gain_covered = G-GT - G-PRED` is the primary proposal-mask geometry gap; paired diagnostics are
reported for pairs whose shared reference is COVERED50.

Part I: `selection_gain = M1 - M0`, `coverage_gain = M3 - M2`, `total_reference_gap = M3 - M0` (all strict
all-record D-B1 mIoU), plus the selection/coverage fractions. Components are deliberately not forced to sum to
100 %.

Part J: the four predeclared diagnostic labels with their exact thresholds.

Writes `evaluation/task7f_gap_decomposition.json`.

    python scripts/task7f_downstream_reference_modes.py   # (cache already built by task7f_reference_modes.py)
    python scripts/task7f_gap_decomposition.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.task7f_reference_ceiling import (  # noqa: E402
    DIRECTIONS,
    MODES,
    MODE_LABELS,
    TOLERANCE,
    read_cache,
)

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task7f_gap_decomposition.json"
DOWNSTREAM = EVAL / "task7f_downstream_reference_modes.json"
PAIRED = EVAL / "task7f_paired_reference_modes.json"
THRESHOLDS = {
    "selection_gain": 0.05, "f_r1_miou": 0.29, "f_r1_paired": 10, "f_r1_margin": 0.18,
    "geometry_gain_covered": 0.05, "coverage_gain": 0.04, "f_r2_coverage_rate": 0.85,
    "f_r2_miou": 0.30, "f_r2_paired": 12, "f_r2_margin": 0.22,
}


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {"records": 0, "miou": None, "dice": None, "precision_at_0_5": None}
    return {"records": len(rows),
            "miou": float(np.mean([row["miou"] for row in rows])),
            "dice": float(np.mean([row["dice"] for row in rows])),
            "precision_at_0_5": float(np.mean([row["precision_at_0_5"] for row in rows]))}


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()
    cache = read_cache()
    downstream = json.loads(DOWNSTREAM.read_text(encoding="utf-8"))
    paired = json.loads(PAIRED.read_text(encoding="utf-8"))
    per_record = {entry["sample_id"]: entry for entry in cache["per_record_modes"]}
    covered_ids = {sample_id for sample_id, entry in per_record.items() if entry["covered50"]}
    target_rows = cache["target_rows"]

    covered_pred = [row for row in target_rows["F-R1"]
                    if row["sample_id"] in covered_ids and not row["abstained"]]
    covered_gt = [row for row in target_rows["F-R3"] if row["sample_id"] in covered_ids]
    g_pred = summarise(covered_pred)
    g_gt = summarise(covered_gt)

    # paired diagnostics restricted to pairs whose shared reference is COVERED50
    pair_rows = [row for row in paired["rows"] if row["covered50"]]
    paired_covered = {"pairs": len(pair_rows)}
    for mode in ("F-R1", "F-R3"):
        answered = [row for row in pair_rows if not row["modes"][mode]["abstained"]]
        own = [value for row in answered for value in row["modes"][mode]["own_iou"]]
        cross = [value for row in answered for value in row["modes"][mode]["cross_iou"]]
        paired_covered[mode] = {
            "passed": sum(1 for row in answered if row["modes"][mode]["passed"]),
            "answered_pairs": len(answered),
            "mean_own_iou": float(np.mean(own)) if own else None,
            "mean_cross_iou": float(np.mean(cross)) if cross else None,
            "own_cross_margin": float(np.mean(own) - np.mean(cross)) if own else None,
        }

    miou = {mode: downstream["results"][mode]["strict"]["miou"] for mode in MODES}
    selection_gain = miou["F-R1"] - miou["F-R0"]
    coverage_gain = miou["F-R3"] - miou["F-R2"]
    total_reference_gap = miou["F-R3"] - miou["F-R0"]
    geometry_gain_covered = (g_gt["miou"] - g_pred["miou"]) if (g_pred["miou"] is not None
                                                                and g_gt["miou"] is not None) else None
    coverage_rate = downstream["results"]["F-R2"]["strict"]["records"] and (
        cache["coverage"]["covered50"] / cache["records"])
    f_r1_paired = paired["results"]["F-R1"]
    f_r2_paired = paired["results"]["F-R2"]

    labels = {
        "SELECTION_IS_ACTIONABLE": {
            "conditions": {
                "selection_gain": {"required": THRESHOLDS["selection_gain"],
                                   "measured": selection_gain,
                                   "passed": selection_gain >= THRESHOLDS["selection_gain"]},
                "f_r1_strict_miou": {"required": THRESHOLDS["f_r1_miou"], "measured": miou["F-R1"],
                                     "passed": miou["F-R1"] >= THRESHOLDS["f_r1_miou"]},
                "f_r1_paired": {"required": f">= {THRESHOLDS['f_r1_paired']}/20",
                                "measured": f_r1_paired["passed"],
                                "passed": f_r1_paired["passed"] >= THRESHOLDS["f_r1_paired"]},
                "f_r1_margin": {"required": THRESHOLDS["f_r1_margin"],
                                "measured": f_r1_paired["own_cross_margin"],
                                "passed": (f_r1_paired["own_cross_margin"] or 0.0)
                                >= THRESHOLDS["f_r1_margin"]},
            }},
        "PROPOSAL_GEOMETRY_IS_MAJOR": {
            "conditions": {
                "geometry_gain_covered": {"required": THRESHOLDS["geometry_gain_covered"],
                                          "measured": geometry_gain_covered,
                                          "passed": (geometry_gain_covered or 0.0)
                                          >= THRESHOLDS["geometry_gain_covered"]}}},
        "PROPOSAL_COVERAGE_IS_MAJOR": {
            "conditions": {
                "coverage_gain": {"required": THRESHOLDS["coverage_gain"],
                                  "measured": coverage_gain,
                                  "passed": coverage_gain >= THRESHOLDS["coverage_gain"]},
                "f_r2_coverage_rate_below": {"required": f"< {THRESHOLDS['f_r2_coverage_rate']}",
                                             "measured": coverage_rate,
                                             "passed": coverage_rate
                                             < THRESHOLDS["f_r2_coverage_rate"]}}},
        "CURRENT_PROPOSAL_SET_HAS_USABLE_CEILING": {
            "conditions": {
                "f_r2_strict_miou": {"required": THRESHOLDS["f_r2_miou"], "measured": miou["F-R2"],
                                     "passed": miou["F-R2"] >= THRESHOLDS["f_r2_miou"]},
                "f_r2_paired": {"required": f">= {THRESHOLDS['f_r2_paired']}/20",
                                "measured": f_r2_paired["passed"],
                                "passed": f_r2_paired["passed"] >= THRESHOLDS["f_r2_paired"]},
                "f_r2_margin": {"required": THRESHOLDS["f_r2_margin"],
                                "measured": f_r2_paired["own_cross_margin"],
                                "passed": (f_r2_paired["own_cross_margin"] or 0.0)
                                >= THRESHOLDS["f_r2_margin"]},
            }},
    }
    for name, entry in labels.items():
        if name == "PROPOSAL_COVERAGE_IS_MAJOR":
            entry["value"] = any(condition["passed"]
                                 for condition in entry["conditions"].values())
        else:
            entry["value"] = all(condition["passed"] for condition in entry["conditions"].values())

    payload = {
        "_doc": ("Task 7F sections 13-19. Covered-subset geometry decomposition, exact ceiling gaps and the "
                 "four predeclared diagnostic labels. All numbers are strict all-record frozen D-B1 mIoU "
                 "unless stated; GT appears only inside the declared F-R1/F-R2/F-R3 diagnostic modes."),
        "task": "7F", "stage": "G-I-gap-decomposition",
        "covered50": {
            "definition": "at least one eligible U-C1 proposal and best eligible IoU >= 0.50 (exactly the "
                          "non-abstaining population of F-R2)",
            "threshold_iou": 0.50, "records": len(covered_ids),
            "coverage_rate": coverage_rate,
            "g_pred": {"mode": "F-R1 oracle-selected predicted proposal mask", **g_pred},
            "g_gt": {"mode": "F-R3 canonical GT reference mask", **g_gt},
            "geometry_gain_covered": geometry_gain_covered,
            "paired_covered50": paired_covered,
            "per_direction": {
                direction: {
                    "g_pred": summarise([row for row in covered_pred
                                         if row["direction"] == direction]),
                    "g_gt": summarise([row for row in covered_gt if row["direction"] == direction])}
                for direction in DIRECTIONS},
        },
        "gaps": {
            "M0_f_r0": miou["F-R0"], "M1_f_r1": miou["F-R1"], "M2_f_r2": miou["F-R2"],
            "M3_f_r3": miou["F-R3"],
            "selection_gain": selection_gain,
            "geometry_gain_covered": geometry_gain_covered,
            "coverage_gain": coverage_gain,
            "total_reference_gap": total_reference_gap,
            "selection_fraction": (max(selection_gain, 0.0) / total_reference_gap
                                   if total_reference_gap else None),
            "coverage_fraction": (max(coverage_gain, 0.0) / total_reference_gap
                                  if total_reference_gap else None),
            "geometry_fraction_note": "geometry_gain_covered has a different subset/denominator and is "
                                      "reported separately; components are not forced to sum to 100%",
            "tolerance": TOLERANCE,
        },
        "labels": labels, "thresholds": THRESHOLDS,
        "reference_modes": {mode: MODE_LABELS[mode] for mode in MODES},
        "training_performed": False, "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT, payload)
    print(f"[7f.gaps] M0 {miou['F-R0']:.4f} M1 {miou['F-R1']:.4f} M2 {miou['F-R2']:.4f} M3 "
          f"{miou['F-R3']:.4f} | selection {selection_gain:+.4f} coverage {coverage_gain:+.4f} total "
          f"{total_reference_gap:+.4f} | geometry(covered50) "
          f"{(geometry_gain_covered or 0.0):+.4f}", flush=True)
    print("[7f.labels] " + json.dumps({name: entry["value"] for name, entry in labels.items()}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
