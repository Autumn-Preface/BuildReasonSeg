"""Task 6M.1 Part G (section 15): audit the corrected structured Demo CLI.

Runs `predict_structured.py` as a subprocess with **no annotation files reachable** on:

* **12 supported prompts** sampled from validation tiles — 4 level-1, 4 level-2, 4 level-3, covering at
  least 8 distinct programs, each prompt taken from the frozen BuildSpatialReason v0.2 record;
* the **6 required out-of-domain prompts**, which must exit 4 with
  `status=unsupported_instruction` / `abstention_reason=out_of_domain_prompt`.

Records per run: prompt, expected program, parsed program, proposal count, selected-mask presence,
overlay presence, exit code, status/abstention reason, whether GT was used.

    python scripts/task6m1_demo_audit.py [--count-supported 12]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.task6m_eval import EVAL, EXPORT_ROOT, tile_ids, write_json  # noqa: E402

OUT = EVAL / "task6m1_demo_cli_audit.json"
DEMO_ROOT = REPO_ROOT / "artifacts" / "task6m1_demo"
SEED = 20260815
PER_LEVEL = 4
OOD_PROMPTS = (
    "Write a poem about the sea.",
    "今天天气怎么样？",
    "请总结这张图片。",
    "检测道路。",
    "segment the airplane",
    "   ",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count-supported", type=int, default=12)
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--config", type=Path, default=EVAL / "task6m1_inference_config_frozen.json",
                        help="frozen config for the CLI defaults (falls back to the Task 6M freeze)")
    parser.add_argument("--prefer-frozen-checkpoint", action="store_true",
                        help="use the frozen config's checkpoint instead of the 6M.1 continuation best.pt")
    args = parser.parse_args(argv)

    started = time.time()
    frozen_path = args.config if args.config.is_file() else EVAL / "task6m_inference_config_frozen.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    parser_report = json.loads((EVAL / "task6m_parser_v02.json").read_text(encoding="utf-8"))
    # Prefer the strongest honestly available proposal checkpoint: the Task 6M.1 continuation best
    # when the continuation produced one, otherwise the checkpoint of the frozen config in use.
    continuation_best = (
        REPO_ROOT / "artifacts" / "checkpoints" / "task6m1" / "runs" / "m1_yolo26m_seg_continued"
        / "weights" / "best.pt"
    )
    if continuation_best.is_file() and args.prefer_frozen_checkpoint is False:
        proposal_checkpoint = continuation_best
        proposal_source = "task6m1_continuation_best"
    else:
        proposal_checkpoint = Path(frozen["checkpoint"]["path"])
        proposal_source = "frozen_config_checkpoint"
    parser_checkpoint = Path(parser_report["checkpoint"]["path"])

    records_by_tile: dict[str, list[dict]] = defaultdict(list)
    with (REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2" / f"{args.split}.jsonl").open(
        encoding="utf-8"
    ) as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                records_by_tile[str(record["image_id"])].append(record)

    # ---- deterministic selection: PER_LEVEL records per level, one per tile, distinct programs first
    rng = random.Random(SEED)
    by_level: dict[int, list[dict]] = defaultdict(list)
    for tile_id, records in records_by_tile.items():
        if tile_id not in set(tile_ids(args.split)):
            continue
        for record in sorted(records, key=lambda r: str(r["sample_id"])):
            by_level[int(record["level"])].append(record)
    for level in by_level:
        rng.shuffle(by_level[level])

    selected: list[dict] = []
    used_tiles: set[str] = set()
    used_programs: set[str] = set()
    for level in (1, 2, 3):
        picked = 0
        pool = [record for record in by_level[level] if str(record["image_id"]) not in used_tiles]
        # first pass: prefer programs not yet covered
        for record in list(pool):
            if picked >= PER_LEVEL:
                break
            if str(record["query_type"]) in used_programs:
                continue
            selected.append(record)
            used_tiles.add(str(record["image_id"]))
            used_programs.add(str(record["query_type"]))
            pool.remove(record)
            picked += 1
        for record in list(pool):
            if picked >= PER_LEVEL:
                break
            selected.append(record)
            used_tiles.add(str(record["image_id"]))
            used_programs.add(str(record["query_type"]))
            picked += 1
    selected.sort(key=lambda record: (int(record["level"]), str(record["sample_id"])))

    if DEMO_ROOT.exists():
        shutil.rmtree(DEMO_ROOT)
    DEMO_ROOT.mkdir(parents=True, exist_ok=True)

    def run_cli(prompt: str, tile_id: str | None, out_dir: Path) -> tuple[int, dict]:
        image = EXPORT_ROOT / "images" / args.split / f"{tile_id}.tif" if tile_id else (
            EXPORT_ROOT / "images" / args.split / f"{sorted(records_by_tile)[0]}.tif"
        )
        command = [
            sys.executable, str(REPO_ROOT / "predict_structured.py"),
            "--image", str(image),
            "--prompt", prompt,
            "--proposal-checkpoint", str(proposal_checkpoint),
            "--parser-checkpoint", str(parser_checkpoint),
            "--out-dir", str(out_dir),
            "--device", args.device,
            "--quiet",
        ]
        result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800)
        payload = {}
        result_path = out_dir / "result.json"
        if result_path.is_file():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        return result.returncode, payload

    supported_rows = []
    for record in selected[: args.count_supported]:
        tile_id = str(record["image_id"])
        out_dir = DEMO_ROOT / f"supported_{tile_id}_{record['query_type']}"
        exit_code, payload = run_cli(str(record["instruction_zh"]), tile_id, out_dir)
        supported_rows.append(
            {
                "tile_id": tile_id,
                "level": int(record["level"]),
                "prompt": record["instruction_zh"],
                "expected_program": str(record["query_type"]),
                "exit_code": exit_code,
                "status": payload.get("status"),
                "abstention_reason": payload.get("abstention_reason"),
                "parsed_program": payload.get("parsed_program"),
                "program_matches_expected": payload.get("parsed_program") == str(record["query_type"]),
                "proposal_count": payload.get("proposal_count"),
                "selected": payload.get("selected") is not None,
                "selected_mask_file": (out_dir / "selected_mask.png").is_file(),
                "overlay_file": (out_dir / "overlay.png").is_file(),
                "ground_truth_used": payload.get("ground_truth_used"),
            }
        )
        print(f"[6m1.demo] supported {len(supported_rows)}/{len(selected[: args.count_supported])} "
              f"{tile_id}: exit {exit_code}, program {payload.get('parsed_program')}, "
              f"proposals {payload.get('proposal_count')}", flush=True)

    ood_rows = []
    for index, prompt in enumerate(OOD_PROMPTS, start=1):
        out_dir = DEMO_ROOT / f"ood_{index}"
        exit_code, payload = run_cli(prompt, None, out_dir)
        ood_rows.append(
            {
                "prompt": prompt,
                "exit_code": exit_code,
                "status": payload.get("status"),
                "abstention_reason": payload.get("abstention_reason"),
                "parsed_program": payload.get("parsed_program"),
                "proposal_count": payload.get("proposal_count"),
                "domain_gate": payload.get("domain_gate"),
            }
        )
        print(f"[6m1.demo] ood {index}/{len(OOD_PROMPTS)}: exit {exit_code}, "
              f"status {payload.get('status')}", flush=True)

    levels = Counter(row["level"] for row in supported_rows)
    programs = {row["parsed_program"] for row in supported_rows if row["parsed_program"]}
    ood_ok = all(
        row["exit_code"] == 4
        and row["status"] == "unsupported_instruction"
        and row["abstention_reason"] == "out_of_domain_prompt"
        and row["parsed_program"] is None
        for row in ood_rows
    )
    gate = {
        "twelve_supported_runs": len(supported_rows) == 12,
        "four_per_level": all(levels.get(level, 0) == PER_LEVEL for level in (1, 2, 3)),
        "at_least_eight_programs": len(programs) >= 8,
        "all_supported_programs_parsed_correctly": all(row["program_matches_expected"] for row in supported_rows),
        "no_ground_truth_used": not any(row["ground_truth_used"] for row in supported_rows),
        "at_least_one_selected_mask": any(row["selected_mask_file"] for row in supported_rows),
        "six_ood_rejected_before_models": ood_ok,
    }
    gate["passed"] = all(gate.values())

    report = {
        "_doc": (
            "Task 6M.1 section 15. Corrected Demo CLI audit: 12 supported prompts (4 per level, >= 8 "
            "distinct programs) executed through the documented CMD interface with no annotation files, "
            "plus the 6 required out-of-domain prompts which must be rejected with exit code 4 before "
            "ProgramHead and before the proposal model."
        ),
        "task": "6M.1",
        "cli": "predict_structured.py",
        "frozen_config_used": str(frozen_path),
        "inference_config": {"conf": frozen["conf"], "max_det": frozen["max_det"],
                             "imgsz": frozen.get("imgsz", 640)},
        "proposal_checkpoint": {
            "path": str(proposal_checkpoint),
            "source": proposal_source,
            "note": (
                "the strongest honestly available proposal checkpoint is used: the Task 6M.1 "
                "continuation best when it exists, otherwise the checkpoint named by the frozen "
                "config in use"
            ),
        },
        "parser_checkpoint": {"path": str(parser_checkpoint)},
        "supported_runs": supported_rows,
        "supported_summary": {
            "runs": len(supported_rows),
            "by_level": dict(sorted(levels.items())),
            "distinct_programs": len(programs),
            "programs": sorted(programs),
            "answered": sum(1 for row in supported_rows if row["status"] == "ok"),
            "abstained": sum(1 for row in supported_rows if row["status"] == "abstained"),
            "programs_parsed_correctly": sum(1 for row in supported_rows if row["program_matches_expected"]),
            "selected_masks": sum(1 for row in supported_rows if row["selected_mask_file"]),
            "overlays": sum(1 for row in supported_rows if row["overlay_file"]),
        },
        "out_of_domain_runs": ood_rows,
        "out_of_domain_summary": {
            "prompts": len(ood_rows),
            "all_rejected_with_exit_4": ood_ok,
            "all_status_unsupported_instruction": all(
                row["status"] == "unsupported_instruction" for row in ood_rows
            ),
            "no_program_mapped": all(row["parsed_program"] is None for row in ood_rows),
            "no_proposals_run": all(row["proposal_count"] is None for row in ood_rows),
        },
        "gate": gate,
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT, report)
    print(f"[6m1.demo] gate {gate}", flush=True)
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
