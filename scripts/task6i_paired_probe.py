"""Task 6I: assemble `evaluation/task6i_paired_probe.json`.

With I1 it is the fixed 20 paired validation images scored by the bounded objective (own/cross
probability mass, pair preference, paired point selection, same-image q0/q1 separation, attention
masses and probability-map overlap) plus the mask side when I2 ran; with I0 (task stopped at the
gate) it is the 10-pair I0 probe. Nothing is recomputed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from task6i_common import EVAL, write_json  # noqa: E402

I0_JSON = EVAL / "task6i_i0_overfit.json"
I1_JSON = EVAL / "task6i_i1_training.json"
SEG_JSON = EVAL / "task6i_segmentation_eval.json"
OUT = EVAL / "task6i_paired_probe.json"


def main() -> int:
    if I1_JSON.exists():
        i1 = json.loads(I1_JSON.read_text(encoding="utf-8"))
        segmentation = json.loads(SEG_JSON.read_text(encoding="utf-8")) if SEG_JSON.exists() else None
        report = {
            "_doc": (
                "Task 6I paired probe on the fixed 20 paired validation images for the selected "
                "I1 model: bounded own-vs-cross target probability mass, pair preference, paired "
                "point selection, same-image q0/q1 separation, attention masses and "
                "probability-map overlap. The mask side is present only when I2 ran."
            ),
            "task": "6I",
            "stage": "I1",
            "selected_epoch": int(i1["selection"]["selected_epoch"]),
            "checkpoint": i1["selection"]["selected_checkpoint"],
            "pairs": i1["final"]["validation_pairs"],
            "pair_rows": i1["final"]["pair_rows"],
            "val": i1["final"]["val"],
            "mask": None if segmentation is None else segmentation["segmentation"]["paired"],
            "mask_ran": segmentation is not None,
            "mask_note": (
                "I2 did not run (section 12 gates it on the I1 gate)."
                if segmentation is None
                else "frozen SAM2 positive-point masks from the refined argmax points."
            ),
        }
    else:
        i0 = json.loads(I0_JSON.read_text(encoding="utf-8"))
        report = {
            "_doc": (
                "Task 6I paired probe. The task stopped at I0 (section 8: "
                "VISUAL_QUERY_REFINEMENT_FAILED_AT_OVERFIT), so this is the 10-pair I0 probe; the "
                "fixed 20-image validation probe belongs to I1, which was not run."
            ),
            "task": "6I",
            "stage": "I0",
            "selected_epoch": None,
            "checkpoint": i0["checkpoint"],
            "pairs": i0["final"]["pair"],
            "pair_rows": i0["final"]["pair_rows"],
            "val": None,
            "mask": None,
            "mask_ran": False,
            "mask_note": "I1/I2 not run: the task stopped at I0.",
        }
    write_json(OUT, report)
    print(
        f"[task6i:paired] stage {report['stage']} preference "
        f"{report['pairs']['pair_ranking_pass']}/{report['pairs']['pair_count']} paired point "
        f"{report['pairs']['paired_point_selection_pass']}/{report['pairs']['pair_count']}",
        flush=True,
    )
    print(f"[task6i:paired] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
