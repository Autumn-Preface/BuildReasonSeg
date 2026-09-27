"""Task 6E: assemble `evaluation/task6e_paired_probe.json` from the recorded evaluations.

The paired probe always exists after E1: geometry (generated boxes for the two instructions of
each of the fixed 20 paired validation images) and, when E2 ran, the mask side (own vs cross
IoU, and the IoU between the two predicted masks). Nothing is recomputed here — the artifact is
an assembly of `task6e_geometry_eval.json` and `task6e_segmentation_eval.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from task6e_common import EVAL, write_json  # noqa: E402

GEOMETRY = EVAL / "task6e_geometry_eval.json"
SEGMENTATION = EVAL / "task6e_segmentation_eval.json"
OUT = EVAL / "task6e_paired_probe.json"


def main() -> int:
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    segmentation = (
        json.loads(SEGMENTATION.read_text(encoding="utf-8")) if SEGMENTATION.exists() else None
    )
    mask = None if segmentation is None else segmentation["segmentation"]["paired"]
    report = {
        "_doc": (
            "Task 6E paired probe on the fixed 20 paired validation images. Geometry: the two "
            "instructions' generated boxes, whether the token sequences differ, and the box L1. "
            "Mask: only present when E2 ran (E1 gate passed), and then it is the frozen SAM2 mask "
            "from each generated box."
        ),
        "task": "6E",
        "bins": geometry.get("bins"),
        "selected_epoch": geometry.get("selected_epoch"),
        "checkpoint": geometry.get("checkpoint"),
        "geometry": geometry.get("paired"),
        "same_image_instruction_divergence": geometry.get("same_image_instruction_divergence"),
        "mask": mask,
        "mask_ran": segmentation is not None,
        "mask_note": (
            "E2 did not run because the E1 geometry gate did not pass (section 14)."
            if segmentation is None
            else "frozen SAM2 box-prompt masks from the generated boxes only."
        ),
        "sources": [
            str(GEOMETRY.relative_to(REPO_ROOT)).replace("\\", "/"),
            *(
                []
                if segmentation is None
                else [str(SEGMENTATION.relative_to(REPO_ROOT)).replace("\\", "/")]
            ),
        ],
    }
    write_json(OUT, report)
    print(
        f"[task6e:paired] geometry paired "
        f"{report['geometry']['geometry_paired_pass']}/{report['geometry']['paired_total']} "
        f"different-token pairs "
        f"{report['same_image_instruction_divergence']['pairs_with_different_tokens']}/"
        f"{report['same_image_instruction_divergence']['compared_pairs']} mask_ran "
        f"{report['mask_ran']}",
        flush=True,
    )
    print(f"[task6e:paired] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
