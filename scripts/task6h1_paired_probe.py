"""Task 6H.1: assemble `evaluation/task6h1_paired_probe.json`.

With H1-R it is the fixed 20 paired validation images scored by the bounded objective (own/cross
probability mass, pair preference, paired point selection, same-image point distance and
probability-map overlap) plus the mask side when H2-R ran; with H0-R (task stopped at the gate) it is
the 10-pair H0-R probe. Nothing is recomputed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from task6h1_common import EVAL, write_json  # noqa: E402

H0R_JSON = EVAL / "task6h1_h0r_overfit.json"
H1R_JSON = EVAL / "task6h1_h1r_training.json"
SEG_JSON = EVAL / "task6h1_segmentation_eval.json"
OUT = EVAL / "task6h1_paired_probe.json"


def main() -> int:
    if H1R_JSON.exists():
        h1r = json.loads(H1R_JSON.read_text(encoding="utf-8"))
        segmentation = json.loads(SEG_JSON.read_text(encoding="utf-8")) if SEG_JSON.exists() else None
        report = {
            "_doc": (
                "Task 6H.1 paired probe on the fixed 20 paired validation images for the selected "
                "H1-R model: bounded own-vs-cross target probability mass, pair preference, paired "
                "point selection, same-image point distance and probability-map overlap. The mask "
                "side is present only when H2-R ran."
            ),
            "task": "6H.1",
            "stage": "H1-R",
            "selected_epoch": int(h1r["selection"]["selected_epoch"]),
            "checkpoint": h1r["selection"]["selected_checkpoint"],
            "pairs": h1r["final"]["validation_pairs"],
            "pair_rows": h1r["final"]["pair_rows"],
            "val": h1r["final"]["val"],
            "mask": None if segmentation is None else segmentation["segmentation"]["paired"],
            "mask_ran": segmentation is not None,
            "mask_note": (
                "H2-R did not run (section 20 gates it on the H1-R gate)."
                if segmentation is None
                else "frozen SAM2 positive-point masks from the point-supervised argmax points."
            ),
        }
    else:
        h0r = json.loads(H0R_JSON.read_text(encoding="utf-8"))
        report = {
            "_doc": (
                "Task 6H.1 paired probe. The task stopped at H0-R (section 13: "
                "BOUNDED_POINT_OBJECTIVE_FAILED), so this is the 10-pair H0-R probe; the fixed "
                "20-image validation probe belongs to H1-R, which was not run."
            ),
            "task": "6H.1",
            "stage": "H0-R",
            "selected_epoch": None,
            "checkpoint": h0r["checkpoint"],
            "pairs": h0r["final"]["pair"],
            "pair_rows": h0r["final"]["pair_rows"],
            "val": None,
            "mask": None,
            "mask_ran": False,
            "mask_note": "H1-R/H2-R not run: the task stopped at H0-R.",
        }
    write_json(OUT, report)
    print(
        f"[task6h1:paired] stage {report['stage']} preference "
        f"{report['pairs']['pair_ranking_pass']}/{report['pairs']['pair_count']} paired point "
        f"{report['pairs']['paired_point_selection_pass']}/{report['pairs']['pair_count']}",
        flush=True,
    )
    print(f"[task6h1:paired] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
