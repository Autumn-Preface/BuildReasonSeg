"""Task 6H: assemble `evaluation/task6h_paired_probe.json`.

The paired probe always exists: with H1 it is the fixed 20 paired validation images (pair ranking,
paired point selection, same-image point distance and heatmap A/B IoU) plus the mask side when H2
ran; with H0 (task stopped at the H0 gate) it is the 10-pair H0 probe. Nothing is recomputed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from task6h_common import EVAL, write_json  # noqa: E402

H0_JSON = EVAL / "task6h_h0_overfit.json"
H1_JSON = EVAL / "task6h_h1_training.json"
SEG_JSON = EVAL / "task6h_segmentation_eval.json"
OUT = EVAL / "task6h_paired_probe.json"


def main() -> int:
    if H1_JSON.exists():
        h1 = json.loads(H1_JSON.read_text(encoding="utf-8"))
        segmentation = json.loads(SEG_JSON.read_text(encoding="utf-8")) if SEG_JSON.exists() else None
        report = {
            "_doc": (
                "Task 6H paired probe on the fixed 20 paired validation images for the selected H1 "
                "model: own-vs-cross region ranking, paired point selection, same-image point "
                "distance and heatmap A/B IoU. The mask side is present only when H2 ran."
            ),
            "task": "6H",
            "stage": "H1",
            "selected_epoch": int(h1["selection"]["selected_epoch"]),
            "checkpoint": h1["selection"]["selected_checkpoint"],
            "pairs": h1["final"]["validation_pairs"],
            "pair_rows": h1["final"]["pair_rows"],
            "val": h1["final"]["val"],
            "mask": None if segmentation is None else segmentation["segmentation"]["paired"],
            "mask_ran": segmentation is not None,
            "mask_note": (
                "H2 did not run (section 18 gates it on the H1 gate)."
                if segmentation is None
                else "frozen SAM2 positive-point masks from the counterfactually trained heatmaps."
            ),
        }
    else:
        h0 = json.loads(H0_JSON.read_text(encoding="utf-8"))
        report = {
            "_doc": (
                "Task 6H paired probe. The task stopped at H0 (section 12: "
                "COUNTERFACTUAL_QUERY_SIGNAL_FAILED), so this is the 10-pair H0 probe; the fixed "
                "20-image validation probe belongs to H1, which was not run."
            ),
            "task": "6H",
            "stage": "H0",
            "selected_epoch": None,
            "checkpoint": h0["checkpoint"],
            "pairs": h0["final"]["pair"],
            "pair_rows": h0["final"]["pairs"],
            "val": None,
            "mask": None,
            "mask_ran": False,
            "mask_note": "H1/H2 not run: the task stopped at H0.",
        }
    write_json(OUT, report)
    print(
        f"[task6h:paired] stage {report['stage']} ranking "
        f"{report['pairs']['pair_ranking_pass']}/{report['pairs']['pair_count']} paired point "
        f"{report['pairs']['paired_point_selection_pass']}/{report['pairs']['pair_count']}",
        flush=True,
    )
    print(f"[task6h:paired] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
