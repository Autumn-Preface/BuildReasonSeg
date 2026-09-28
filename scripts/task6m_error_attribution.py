"""Task 6M section 12/13: failure attribution and the final verdict.

Attributes J1-v2 / J4-v2 failures to proposal recall, executor/abstention, tiny-instance gaps, border
truncation or parser error, and emits the single allowed verdict from Task 6M section 16.

    python scripts/task6m_error_attribution.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.task6m_eval import EVAL, write_json  # noqa: E402

OUT = EVAL / "task6m_error_attribution.json"
VERDICT_OUT = EVAL / "task6m_verdict.json"

ALLOWED_VERDICTS = (
    "STRUCTURED_DEMO_READY",
    "PROPOSAL_MODEL_NEEDS_IMPROVEMENT",
    "PROGRAM_PARSER_NEEDS_IMPROVEMENT",
    "TRAINING_EXPORT_INVALID",
    "INVALID_EXPERIMENT",
)


def load(name: str) -> dict | None:
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args(argv)

    export = load("task6m_training_export_audit.json")
    environment = load("task6m_environment_manifest.json")
    training = load("task6m_training_summary.json")
    smoke = load("task6m_smoke.json")
    proposal = load("task6m_proposal_val.json")
    frozen = load("task6m_inference_config_frozen.json")
    j1 = load("task6m_j1v2_val.json")
    parser_report = load("task6m_parser_v02.json")
    j4 = load("task6m_j4v2_test.json")
    demo = load("task6m_demo_cli_audit.json")

    # ---------------- attribution over the J1-v2 fixed120 rows
    fixed_rows = []
    if j1 and j1.get("fixed120"):
        fixed_rows = j1["fixed120"].get("rows", []) if isinstance(j1["fixed120"], dict) else []
    if not fixed_rows:
        # fixed120 summary does not carry rows in the artifact; use the full-val rows instead
        pass

    recall = (j1 or {}).get("proposal_recall", {})
    attribution = {
        "proposal_recall_at_0_50": recall.get("recall_at_0_50"),
        "proposal_tiny_recall_at_0_50": recall.get("tiny_recall_at_0_50"),
        "proposal_border_recall_at_0_50": recall.get("border_recall_at_0_50"),
        "fixed120_miou": ((j1 or {}).get("fixed120") or {}).get("miou"),
        "fixed120_abstentions": ((j1 or {}).get("fixed120") or {}).get("abstentions"),
        "fixed120_abstention_reasons": ((j1 or {}).get("fixed120") or {}).get("abstention_reasons"),
        "paired_pass": ((j1 or {}).get("paired20") or {}).get("passed"),
        "paired_total": ((j1 or {}).get("paired20") or {}).get("pairs"),
    }
    full = (j1 or {}).get("full_val") or {}
    if full:
        by_level = full.get("by_level", {})
        weakest_programs = sorted(
            ((name, value.get("miou"), value.get("records", 0)) for name, value in (full.get("by_program") or {}).items()),
            key=lambda item: (item[1] if item[1] is not None else 1.0),
        )[:8]
        attribution["full_val_miou"] = full.get("miou")
        attribution["full_val_abstentions"] = full.get("abstentions")
        attribution["full_val_abstention_reasons"] = full.get("abstention_reasons")
        attribution["by_level_miou"] = {level: value.get("miou") for level, value in by_level.items()}
        attribution["weakest_programs"] = [{"program": name, "miou": miou, "records": records}
                                           for name, miou, records in weakest_programs]
        attribution["tiny_target_miou"] = (full.get("tiny_target") or {}).get("miou")
        attribution["border_target_miou"] = (full.get("border_target") or {}).get("miou")
        attribution["dense_tile_miou"] = (full.get("dense_tile") or {}).get("miou")

    # ---------------- diagnosis
    diagnosis = []
    if recall.get("recall_at_0_50") is not None and recall["recall_at_0_50"] < 0.92:
        diagnosis.append(
            {
                "cause": "proposal_recall_below_gate",
                "detail": f"overall recall@0.50 = {recall['recall_at_0_50']:.4f} < 0.92",
            }
        )
    if recall.get("tiny_recall_at_0_50") is not None and recall["tiny_recall_at_0_50"] < 0.60:
        diagnosis.append(
            {
                "cause": "tiny_instance_recall_below_gate",
                "detail": f"tiny recall@0.50 = {recall['tiny_recall_at_0_50']:.4f} < 0.60",
            }
        )
    if full.get("abstention_rate") is not None and full["abstention_rate"] > 0.2:
        diagnosis.append(
            {
                "cause": "executor_abstention_high",
                "detail": f"full-val abstention rate {full['abstention_rate']:.4f}",
            }
        )
    if parser_report and parser_report.get("gate", {}).get("passed") is False:
        diagnosis.append({"cause": "parser_accuracy_below_gate", "detail": str(parser_report.get("verdict"))})
    if export and export.get("verdict") != "EXPORT_VALID":
        diagnosis.append({"cause": "export_invalid", "detail": str(export.get("verdict"))})
    if not diagnosis:
        diagnosis.append({"cause": "no_blocking_cause_detected", "detail": "all gates evaluated as passing"})

    # ---------------- verdict
    export_ok = bool(export and export.get("verdict") == "EXPORT_VALID")
    smoke_ok = bool(smoke and all((smoke.get("gates") or {}).values()))
    j1_ok = bool(j1 and j1.get("gate", {}).get("passed"))
    parser_ok = bool(parser_report and parser_report.get("gate", {}).get("passed"))
    j4_fixed = ((j4 or {}).get("fixed120") or {}).get("miou")
    j4_paired = ((j4 or {}).get("paired20") or {}).get("passed")
    demo_ok = bool(demo and demo.get("gate", {}).get("passed"))

    if not export_ok or not smoke_ok:
        verdict = "TRAINING_EXPORT_INVALID"
    elif j1 is None or frozen is None:
        verdict = "INVALID_EXPERIMENT"
    elif not j1_ok:
        verdict = "PROPOSAL_MODEL_NEEDS_IMPROVEMENT"
    elif not parser_ok:
        verdict = "PROGRAM_PARSER_NEEDS_IMPROVEMENT"
    elif j4 is None or j4_fixed is None:
        verdict = "INVALID_EXPERIMENT"
    elif j4_fixed < 0.40 or (j4_paired or 0) < 14:
        verdict = "PROPOSAL_MODEL_NEEDS_IMPROVEMENT"
    elif not demo_ok:
        verdict = "INVALID_EXPERIMENT"
    else:
        verdict = "STRUCTURED_DEMO_READY"

    payload = {
        "_doc": (
            "Task 6M sections 12-16. Failure attribution for the native J1-v2/J4-v2 chain and the "
            "single final verdict."
        ),
        "task": "6M",
        "attribution": attribution,
        "diagnosis": diagnosis,
        "runtime": {
            "ultralytics": (environment or {}).get("packages", {}).get("ultralytics"),
            "pretrained_weights": (environment or {}).get("pretrained_weights"),
        },
        "training": {
            "epochs_recorded": ((training or {}).get("results") or {}).get("epochs_recorded"),
            "wall_time_hours": ((training or {}).get("resources") or {}).get("wall_time_hours"),
            "peak_vram_gb": ((training or {}).get("resources") or {}).get("peak_vram_allocated_gb"),
            "best": ((training or {}).get("results") or {}).get("best"),
            "checkpoint": ((training or {}).get("checkpoints") or {}).get("best"),
        },
    }
    write_json(OUT, payload)

    gates = {
        "export_valid": export_ok,
        "smoke_passed": smoke_ok,
        "j1v2_gate_passed": j1_ok,
        "parser_ready": parser_ok,
        "j4v2_fixed120_miou_ge_0_40": bool(j4_fixed is not None and j4_fixed >= 0.40),
        "j4v2_paired_pass_ge_14": bool((j4_paired or 0) >= 14),
        "demo_cli_gate_passed": demo_ok,
    }
    verdict_payload = {
        "_doc": "Task 6M section 16. Exactly one verdict from the allowed set, with every gate measured.",
        "task": "6M",
        "verdict": verdict,
        "allowed_verdicts": list(ALLOWED_VERDICTS),
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "measured": {
            "proposal_recall_at_0_50": recall.get("recall_at_0_50"),
            "proposal_tiny_recall_at_0_50": recall.get("tiny_recall_at_0_50"),
            "j1v2_fixed120_miou": ((j1 or {}).get("fixed120") or {}).get("miou"),
            "j1v2_paired_pass": ((j1 or {}).get("paired20") or {}).get("passed"),
            "parser_v02_accuracy": ((parser_report or {}).get("v0.2_full_val") or {}).get("exact_accuracy"),
            "j4v2_fixed120_miou": j4_fixed,
            "j4v2_paired_pass": j4_paired,
            "demo_images_audited": (demo or {}).get("images_audited"),
            "training_epochs_recorded": ((training or {}).get("results") or {}).get("epochs_recorded"),
        },
        "diagnosis": diagnosis,
        "keep_flags": {
            "keep_whu_imagery": True,
            "keep_v0_1_1_and_v0_2_datasets_frozen": True,
            "keep_task6j_historical_artifacts_frozen": True,
            "no_framework_switch": True,
            "no_4b_upgrade": True,
        },
    }
    write_json(VERDICT_OUT, verdict_payload)
    print(f"[6m.verdict] {verdict}; failed gates {verdict_payload['failed_gates']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
