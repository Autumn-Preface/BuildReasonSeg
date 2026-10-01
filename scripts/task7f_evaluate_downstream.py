"""Task 7F Part F — frozen D-B1 downstream evaluation for the four reference modes.

For every mode the chain is the frozen one:

```text
canonical L3 program -> mode-specific reference mask -> P_dir v0.2 -> P_near v0.1
-> frozen SAM2 feature -> frozen D-B1 -> target mask
```

Abstaining records count as strict IoU/Dice = 0. This script reads the shared Task 7F measurement cache
(created by `scripts/task7f_reference_modes.py`) and performs the two required reproductions:

* F-R0 must reproduce the Task 7E predicted-reference D-B1 result exactly (strict mIoU `0.24540501038500215`,
  answered-only `0.24614085749260334`, abstentions `2`);
* F-R3 must reproduce the Task 7E oracle-reference D-B1 result exactly (mIoU `0.38549570532647004`).

Either failure stops with `TASK7E_NUMERIC_REPRODUCTION_FAIL`.

Writes `evaluation/task7f_downstream_reference_modes.json`.

    python scripts/task7f_evaluate_downstream.py
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
OUT = EVAL / "task7f_downstream_reference_modes.json"
EXPECTED = {
    "F-R0": {"strict_miou": 0.24540501038500215, "answered_only_miou": 0.24614085749260334,
             "abstentions": 2},
    "F-R3": {"strict_miou": 0.38549570532647004, "answered_only_miou": None, "abstentions": 0},
}


def summarise(rows: list[dict]) -> dict:
    answered = [row for row in rows if not row["abstained"]]
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
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()
    cache = read_cache()
    target_rows = cache["target_rows"]
    per_record = cache["per_record_modes"]

    results = {}
    for mode in MODES:
        rows = target_rows[mode]
        reference_ok = [row for row in rows
                        if not row["abstained"]
                        and next(entry for entry in per_record
                                 if entry["sample_id"] == row["sample_id"])["covered50"]]
        results[mode] = {
            "label": MODE_LABELS[mode],
            "strict": summarise(rows),
            "reference_ok_style_subset": summarise(reference_ok),
            "per_direction": {direction: summarise([row for row in rows
                                                    if row["direction"] == direction])
                              for direction in DIRECTIONS},
        }

    reproduction = {}
    for mode, expected in EXPECTED.items():
        measured = results[mode]["strict"]
        entry = {"strict_miou": measured["miou"], "expected_strict_miou": expected["strict_miou"],
                 "strict_delta": abs(measured["miou"] - expected["strict_miou"]),
                 "strict_ok": abs(measured["miou"] - expected["strict_miou"]) <= TOLERANCE,
                 "abstentions": measured["abstentions"],
                 "expected_abstentions": expected["abstentions"],
                 "abstentions_ok": measured["abstentions"] == expected["abstentions"]}
        if expected["answered_only_miou"] is not None:
            entry.update({
                "answered_only_miou": measured["answered_only_miou"],
                "expected_answered_only_miou": expected["answered_only_miou"],
                "answered_only_delta": abs(measured["answered_only_miou"]
                                           - expected["answered_only_miou"]),
                "answered_only_ok": abs(measured["answered_only_miou"]
                                        - expected["answered_only_miou"]) <= TOLERANCE})
        reproduction[mode] = entry
    reproduction["passed"] = all(entry["strict_ok"] and entry["abstentions_ok"]
                                 and entry.get("answered_only_ok", True)
                                 for entry in reproduction.values()
                                 if isinstance(entry, dict))

    payload = {
        "_doc": ("Task 7F sections 11-12. Frozen D-B1 downstream evaluation for F-R0 ... F-R3 over the frozen "
                 "Task 7E E-HoldoutL3 population. The only difference between the four runs is the reference "
                 "mask; the program id, the fields, the SAM2 feature, the decoder weights and the loss are "
                 "untouched, and abstentions count as strict IoU 0."),
        "task": "7F", "stage": "F-downstream-reference-modes",
        "records": cache["records"], "checks": cache["checks"],
        "modes": {mode: MODE_LABELS[mode] for mode in MODES},
        "results": results, "reproduction": reproduction,
        "reproduction_passed": reproduction["passed"],
        "verdict": ("TASK7E_NUMERIC_REPRODUCTION_PASS" if reproduction["passed"]
                    else "TASK7E_NUMERIC_REPRODUCTION_FAIL"),
        "training_performed": False, "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT, payload)
    for mode in MODES:
        measured = results[mode]["strict"]
        print(f"[7f.downstream] {mode}: strict {measured['miou']:.4f} Dice {measured['dice']:.4f} "
              f"Pr@0.5 {measured['precision_at_0_5']:.4f} answered {measured['answered']}/"
              f"{measured['records']} answered-only "
              f"{(measured['answered_only_miou'] or 0.0):.4f}", flush=True)
    print(f"[7f.downstream] reproduction {reproduction['passed']} | F-R0 Δ"
          f"{reproduction['F-R0']['strict_delta']:.2e} | F-R3 Δ"
          f"{reproduction['F-R3']['strict_delta']:.2e}", flush=True)
    return 0 if reproduction["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
