"""Task 6G: assemble `evaluation/task6g_paired_probe.json`.

The paired probe always exists: with G0 it is the 10-pair G0 evaluation probe (the fixed 20-image
probe requires G1, which section 11 forbids running after a G0 failure); with G1 it is the
20-pair point probe plus (when G2 ran) the mask side. Nothing is recomputed here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from task6g_common import EVAL, write_json  # noqa: E402

G0_JSON = EVAL / "task6g_g0_overfit.json"
G1_JSON = EVAL / "task6g_g1_training.json"
OUT = EVAL / "task6g_paired_probe.json"


def main() -> int:
    if G1_JSON.exists():
        g1 = json.loads(G1_JSON.read_text(encoding="utf-8"))
        geometry = g1["final"]["paired"]
        report = {
            "_doc": (
                "Task 6G paired point probe on the fixed 20 paired validation images: the two "
                "instructions' argmax points, same-image point distance and heatmap A/B IoU at the "
                "selected G1 model. The mask side is added by G2 (only when G1 passes)."
            ),
            "task": "6G",
            "stage": "G1",
            "selected_epoch": int(g1["selection"]["selected_epoch"]),
            "checkpoint": g1["selection"]["selected_checkpoint"],
            "geometry": geometry,
            "mask": None,
            "mask_ran": False,
            "mask_note": "G2 did not run (section 16 gates it on the G1 gate).",
        }
    else:
        g0 = json.loads(G0_JSON.read_text(encoding="utf-8"))
        report = {
            "_doc": (
                "Task 6G paired point probe. The task stopped at G0 (section 11: "
                "DENSE_GROUNDING_IMPLEMENTATION_FAILED), so this is the G0 10-pair evaluation "
                "probe; the fixed 20-image validation probe belongs to G1, which was not run."
            ),
            "task": "6G",
            "stage": "G0",
            "selected_epoch": None,
            "checkpoint": g0["checkpoint"],
            "geometry": g0["final"]["paired"],
            "mask": None,
            "mask_ran": False,
            "mask_note": "G1/G2 not run: the task stopped at G0.",
        }
    write_json(OUT, report)
    print(
        f"[task6g:paired] stage {report['stage']} paired "
        f"{report['geometry']['paired_point_selection_pass']}/{report['geometry']['paired_total']} "
        f"distinct {report['geometry']['pairs_with_distinct_points']}/"
        f"{report['geometry']['paired_total']}",
        flush=True,
    )
    print(f"[task6g:paired] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
