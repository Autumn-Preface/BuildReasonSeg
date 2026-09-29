"""Task 6M section 14: native J4-v2 — instruction -> Qwen2B parser -> YOLO26m-seg -> executor on TEST.

Runs ONCE, after the proposal inference config and the parser are frozen. No ground truth is used in
inference; GT only scores the result.

    python scripts/task6m_j4v2.py [--limit N] [--skip-full]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "spatial_reasoning"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.task6m_eval import EVAL, tile_ids, write_json  # noqa: E402
from buildreasonseg_mvp.task6m_structured import (  # noqa: E402
    evaluate_fixed120,
    evaluate_full_split,
    evaluate_paired20,
    load_frozen_config,
    load_pack,
    predict_tiles,
    relation_config,
    v02_records_by_sample,
)

OUT = EVAL / "task6m_j4v2_test.json"
GATES = {"fixed120_miou_min": 0.40, "paired_pass_min": 14}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--split", default="test", choices=("val", "test"),
                        help="the graded J4-v2 run uses test; val exists only for dry-runs")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--config", type=Path, default=None,
                        help="frozen inference config to use (Task 6M.1 passes its own freeze)")
    parser.add_argument("--parser-report", type=Path, default=EVAL / "task6m_parser_v02.json",
                        help="parser artifact whose checkpoint is used (Task 6M.1 reuses the 6M parser)")
    parser.add_argument("--device", default="0")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    pack_fixed = f"task6m_{args.split}_fixed120.json"
    pack_paired = f"task6m_{args.split}_paired20.json"
    run_kind = "graded_test_run" if args.split == "test" else "dry_run_on_val"

    started = time.time()
    import torch
    from ultralytics import YOLO

    frozen = load_frozen_config(args.config) if args.config else load_frozen_config()
    checkpoint = Path(frozen["checkpoint"]["path"])
    parser_report = json.loads(Path(args.parser_report).read_text(encoding="utf-8"))
    parser_checkpoint_path = Path(parser_report["checkpoint"]["path"])
    config = relation_config()

    # ---------------- PROPOSALS (no GT)
    ids = tile_ids(args.split)
    if args.limit:
        ids = ids[: args.limit]
    model = YOLO(str(checkpoint))
    predictions = predict_tiles(
        model, ids, conf=float(frozen["conf"]), max_det=int(frozen["max_det"]),
        imgsz=int(frozen.get("imgsz", 640)), device=args.device, verbose=not args.quiet,
    )

    # ---------------- PARSER (instruction text only, no GT)
    from buildreasonseg_mvp.program_parser import build_program_parser, load_parser_checkpoint
    from buildreasonseg_mvp.runtime import load_config

    cfg = load_config(REPO_ROOT / "configs" / "mvp" / "task6j_program_parser.yaml")
    runtime = build_program_parser(cfg, device="cuda" if args.device != "cpu" else "cpu", verbose=not args.quiet)
    load_parser_checkpoint(parser_checkpoint_path, runtime)

    records = v02_records_by_sample("test")
    sample_ids = [str(record["sample_id"]) for record in records.values()
                  if str(record["image_id"]) in set(ids)]
    instructions = [str(records[sample_id]["instruction_en"]) for sample_id in sample_ids]
    predicted_programs: dict[str, str] = {}
    batch_size = 32
    for start in range(0, len(sample_ids), batch_size):
        chunk_ids = sample_ids[start: start + batch_size]
        batch = runtime.build_batch(
            [records[sample_id]["instruction_en"] for sample_id in chunk_ids],
            [records[sample_id]["query_type"] for sample_id in chunk_ids],
        ).to(runtime.device)
        with torch.no_grad():
            logits, _hidden = runtime.forward(batch)
        for sample_id, index in zip(chunk_ids, logits.argmax(dim=1).cpu().numpy().tolist()):
            predicted_programs[sample_id] = runtime.reports["program_ids"][int(index)]
        if not args.quiet and (start // batch_size) % 50 == 0:
            print(f"[6m.j4v2] parsed {min(start + batch_size, len(sample_ids))}/{len(sample_ids)}",
                  flush=True)

    parser_accuracy = float(
        np.mean([predicted_programs[sample_id] == str(records[sample_id]["query_type"])
                 for sample_id in sample_ids])
    ) if sample_ids else None

    def program_of(record):
        sample_id = record.get("sample_id")
        if sample_id is None:
            return str(record["query_type"])
        return predicted_programs.get(str(sample_id), str(record["query_type"]))

    fixed = evaluate_fixed120(predictions, load_pack(pack_fixed), config, program_of=program_of)
    paired = evaluate_paired20(predictions, load_pack(pack_paired), config, program_of=program_of)
    full = None if args.skip_full else evaluate_full_split(
        predictions, ids, config, split=args.split, program_of=program_of
    )

    checks = {
        "fixed120_miou": bool(fixed["miou"] is not None and fixed["miou"] >= GATES["fixed120_miou_min"]),
        "paired_pass": bool(paired["passed"] >= GATES["paired_pass_min"]),
    }
    report = {
        "_doc": (
            "Task 6M section 14. Native J4-v2, run ONCE on the test split after the proposal "
            "inference config and the parser were frozen. The chain is instruction -> Qwen2B "
            "ProgramHead -> canonical program; image -> YOLO26m-seg proposals; program + predicted "
            "geometry -> deterministic executor. No ground truth participates in inference."
        ),
        "task": "6M",
        "split": args.split,
        "run_kind": run_kind,
        "single_run": args.split == "test",
        "proposal_checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "parser_checkpoint": {
            "path": str(parser_checkpoint_path),
            "sha256": sha256_file(parser_checkpoint_path) if parser_checkpoint_path.is_file() else None,
        },
        "inference_config": {
            "conf": frozen["conf"],
            "max_det": frozen["max_det"],
            "imgsz": frozen.get("imgsz", 640),
            "frozen_before_test": frozen.get("frozen_before_test"),
        },
        "parser_accuracy": parser_accuracy,
        "parser_samples": len(sample_ids),
        "fixed120": {key: value for key, value in fixed.items() if key != "rows"},
        "paired20": {key: value for key, value in paired.items() if key != "rows"},
        "full_test": None if full is None else {key: value for key, value in full.items() if key != "rows_sample"},
        "gate": {"thresholds": GATES, "checks": checks, "passed": all(checks.values())},
        "verdict": "J4V2_PASS" if all(checks.values()) else "J4V2_FAIL",
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(Path(args.out), report)
    print(
        f"[6m.j4v2] parser acc {parser_accuracy}; fixed120 mIoU {fixed['miou']:.4f}; "
        f"paired {paired['passed']}/{paired['pairs']}; gate {'PASS' if all(checks.values()) else 'FAIL'}",
        flush=True,
    )
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
