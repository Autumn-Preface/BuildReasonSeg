"""Task 7E Part K — optional canonical parser interface regression (section 20).

This script only runs when the section-19 predicted-reference gate
(`DB1_PREDICTED_REFERENCE_USABLE`) passed. When it did not pass, the canonical parser integration is never
executed and this script exits with the recorded reason, because section 20 is explicitly conditional.

When it does run (only under a passing `DB1_PREDICTED_REFERENCE_USABLE`), it:

1. runs the frozen Task 7C canonical parser on the canonical query strings associated with `E-HoldoutL3`
   (no free-form paraphrases), requiring canonical accuracy >= 0.995, and records a lower value as an
   interface regression;
2. runs the chain Task 7C canonical parser -> U-C1 predicted reference -> frozen D-B1 and reports
   strict/answered target metrics.

No parser training happens here.

    python scripts/task7e_parser_integration.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.task7e_l3_decoder_adapter import frozen_metadata  # noqa: E402
from scripts.task7e_build_holdout import HOLDOUT_ROOT  # noqa: E402
from scripts.task7e_evaluate_predicted_reference import (  # noqa: E402
    OUT_HOLDOUT as PREDICTED_HOLDOUT,
)
from scripts.task7e_evaluate_oracle import read_holdout_rows  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task7e_canonical_parser_integration.json"
DATA = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2"
CANONICAL_ACCURACY_MIN = 0.995


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--yolo-device", default="0")
    args = parser.parse_args(argv)
    started = time.time()

    predicted = json.loads(PREDICTED_HOLDOUT.read_text(encoding="utf-8"))
    if not predicted.get("DB1_PREDICTED_REFERENCE_USABLE"):
        write_json(OUT, {
            "_doc": ("Task 7E section 20. Canonical parser interface regression, which is conditional on the "
                     "section-19 predicted-reference gate. That gate did not pass, so the parser integration "
                     "was not executed."),
            "task": "7E", "stage": "K-parser-integration", "executed": False,
            "reason": "DB1_PREDICTED_REFERENCE_USABLE is false; section 20 is conditional on section 19",
            "predicted_reference_gate": predicted.get("gate"),
            "canonical_parser_accuracy_min": CANONICAL_ACCURACY_MIN,
            "training_performed": False, "parser_trained": False, "free_form_paraphrase_used": False,
            "test_split_used": False, "runtime_seconds": round(time.time() - started, 1),
        })
        print("[7e.parser] not executed: DB1_PREDICTED_REFERENCE_USABLE is false", flush=True)
        return 3

    # ---- runs only under a passing section-19 gate
    import numpy as np

    from buildreasonseg_mvp.program_parser import build_program_parser, load_parser_checkpoint
    from buildreasonseg_mvp.runtime import load_config
    from buildreasonseg_mvp.structured_grounding import EXPECTED_QUERY_TYPES
    from buildreasonseg_mvp.task6s_directional_pipeline import parse_instruction
    from buildreasonseg_mvp.task7a_l3_pipeline import default_l3_parser_checkpoint

    records = read_holdout_rows()
    canonical = {}
    with (DATA / "val.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            canonical[str(record["sample_id"])] = str(record["instruction_en"])
    parser_checkpoint = default_l3_parser_checkpoint()
    cfg = load_config(REPO_ROOT / "configs" / "mvp" / "task6j_program_parser.yaml")
    runtime = build_program_parser(cfg, device=args.device, verbose=False)
    load_parser_checkpoint(parser_checkpoint, runtime)
    vocabulary = tuple(EXPECTED_QUERY_TYPES)
    correct = 0
    for record in records:
        prediction = parse_instruction(runtime, canonical[record["sample_id"]], vocabulary)
        correct += int(str(prediction["program"]) == str(record["program_id"]))
    accuracy = correct / max(1, len(records))
    payload = {
        "_doc": ("Task 7E section 20. Canonical parser interface regression on the canonical E-HoldoutL3 "
                 "query strings (no free-form paraphrases) feeding the frozen U-C1 reference and frozen "
                 "D-B1. The parser is loaded read-only and is not trained."),
        "task": "7E", "stage": "K-parser-integration", "executed": True,
        "checkpoints": frozen_metadata(),
        "parser": {"path": str(parser_checkpoint), "trained": False},
        "canonical_accuracy": accuracy, "records": len(records),
        "canonical_accuracy_min": CANONICAL_ACCURACY_MIN,
        "canonical_accuracy_passed": accuracy >= CANONICAL_ACCURACY_MIN,
        "interface_regression": accuracy < CANONICAL_ACCURACY_MIN,
        "free_form_paraphrase_used": False, "architecture_selected_by_parser": False,
        "training_performed": False, "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT, payload)
    print(f"[7e.parser] canonical accuracy {accuracy:.4f} -> "
          f"{'PASS' if payload['canonical_accuracy_passed'] else 'INTERFACE_REGRESSION'}", flush=True)
    return 0 if payload["canonical_accuracy_passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
