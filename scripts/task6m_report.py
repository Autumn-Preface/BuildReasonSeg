"""Print a compact Task 6M summary from every artifact (for the handoff and the final report).

    python scripts/task6m_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL = REPO_ROOT / "evaluation"


def load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def show(title: str, value) -> None:
    if value is None:
        print(f"{title}: (pending)")
        return
    print(f"{title}: {value}")


def main() -> int:
    packs = load("task6m_eval_pack_manifest.json")
    export = load("task6m_training_export_audit.json")
    env = load("task6m_environment_manifest.json")
    smoke = load("task6m_smoke.json")
    training = load("task6m_training_summary.json")
    proposal = load("task6m_proposal_val.json")
    frozen = load("task6m_inference_config_frozen.json")
    j1 = load("task6m_j1v2_val.json")
    parser = load("task6m_parser_v02.json")
    j4 = load("task6m_j4v2_test.json")
    demo = load("task6m_demo_cli_audit.json")
    verdict = load("task6m_verdict.json")

    print("=== VERDICT ===")
    show("verdict", (verdict or {}).get("verdict"))
    show("failed gates", (verdict or {}).get("failed_gates"))

    print("\n=== EVAL PACKS ===")
    if packs:
        for name, entry in packs["artifacts"].items():
            print(f"  {name}: {entry['sha256'][:16]} {entry['bytes']}B")
        show("frozen before training", packs["construction_policy"]["frozen_before_training"])

    print("\n=== EXPORT ===")
    if export:
        show("verdict", export["verdict"])
        show("tiles / polygons / empty labels",
             (export["counts"]["tiles"], export["counts"]["instances_polygons"], export["counts"]["empty_label_files"]))
        show("image linking", export["image_linking"])
        show("mean tile union IoU", round(export["fidelity"]["mean_tile_union_iou"], 5))
        show("per-instance IoU (>=9px)", round(export["fidelity"]["mean_per_instance_iou_area_ge_9px"], 5))
        show("malformed / missing / holes / repaired",
             (export["fidelity"]["malformed_labels"], export["fidelity"]["missing_non_hole_instances"],
              export["fidelity"]["hole_instances_affected"], export["fidelity"]["sub_pixel_polygons_repaired"]))
        show("tiny recovered", f"{export['fidelity']['tiny_instances_recovered']}/{export['fidelity']['tiny_instances_total']}")

    print("\n=== ENVIRONMENT ===")
    if env:
        show("python / ultralytics", (env["environment"]["python"], env["environment"]["packages"].get("ultralytics")))
        show("torch / gpu", (env["environment"]["packages"].get("torch"), env["hardware"]["gpu"]))
        for name, entry in env["pretrained_weights"].items():
            print(f"  {name}: {entry['sha256'][:16]} {entry['bytes']}B")

    print("\n=== SMOKE (M0) ===")
    if smoke:
        show("gates", smoke["gates"])
        show("subset", smoke["subset"])

    print("\n=== TRAINING (M1) ===")
    if training:
        show("epochs recorded", training["results"]["epochs_recorded"])
        show("best", training["results"]["best"])
        show("mean epoch seconds", training["resources"]["mean_epoch_seconds"])
        show("stop", training["stop"])
        show("best checkpoint", (training["checkpoints"]["best"] or {}).get("sha256", "")[:16])

    print("\n=== PROPOSAL METRICS (val) ===")
    if proposal:
        metrics = proposal["metrics"]
        show("recall@0.25/0.50/0.75", metrics["recall_at"])
        show("mask precision/recall (union)", (metrics["mask_precision_union"], metrics["mask_recall_union"]))
        show("validator mask mAP50 / 50-95",
             (proposal["validator_metrics"].get("mask_map50"), proposal["validator_metrics"].get("mask_map50_95")))
        show("proposals/tile", round(metrics["proposals_per_tile"], 2))
        show("empty-tile false-proposal rate", metrics["empty_tile_false_proposal_rate"])
        show("tiny / border / dense recall@0.5",
             (metrics["tiny_recall_at_0_50"], metrics["border_recall_at_0_50"], metrics["dense_recall_at_0_50"]))
        show("size breakdown", metrics["size_breakdown"])
        show("sweep", proposal["sweep"])
    show("frozen config", None if not frozen else {"conf": frozen["conf"], "max_det": frozen["max_det"],
                                                   "frozen_before_test": frozen["frozen_before_test"]})

    print("\n=== J1-v2 (val, oracle program) ===")
    if j1:
        show("proposal recall", {k: j1["proposal_recall"][k] for k in ("recall_at_0_25", "recall_at_0_50", "recall_at_0_75")})
        show("tiny recall@0.5", j1["proposal_recall"]["tiny_recall_at_0_50"])
        show("fixed120 mIoU / Dice", (j1["fixed120"]["miou"], j1["fixed120"]["mdice"]))
        show("fixed120 abstentions", j1["fixed120"]["abstentions"])
        show("paired20", (j1["paired20"]["passed"], j1["paired20"]["pairs"],
                          j1["paired20"]["mean_own_iou"], j1["paired20"]["mean_cross_iou"]))
        if j1.get("full_val"):
            show("full val mIoU / Dice", (j1["full_val"]["miou"], j1["full_val"]["mdice"]))
            show("full val by level", j1["full_val"]["by_level"])
            show("full val abstention rate", j1["full_val"]["abstention_rate"])
        show("gate", j1["gate"]["checks"])

    print("\n=== PARSER (v0.2) ===")
    if parser:
        show("full val accuracy / macro F1",
             (parser["v0.2_full_val"]["exact_accuracy"], parser["v0.2_full_val"]["macro_f1"]))
        show("fixed120 accuracy", parser["v0.2_fixed120"]["exact_accuracy"])
        show("checkpoint", (parser.get("checkpoint") or {}).get("path"))
        show("verdict", parser["verdict"])

    print("\n=== J4-v2 (test) ===")
    if j4:
        show("parser accuracy", j4["parser_accuracy"])
        show("fixed120 mIoU / Dice", (j4["fixed120"]["miou"], j4["fixed120"]["mdice"]))
        show("paired20", (j4["paired20"]["passed"], j4["paired20"]["pairs"]))
        if j4.get("full_test"):
            show("full test mIoU", j4["full_test"]["miou"])
            show("full test by level", j4["full_test"]["by_level"])
        show("gate", j4["gate"]["checks"])

    print("\n=== DEMO CLI ===")
    if demo:
        show("images audited", demo["images_audited"])
        show("summary", demo["summary"])
        show("gate", demo["gate"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
