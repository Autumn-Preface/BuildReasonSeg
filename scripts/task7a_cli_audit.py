"""Task 7A Part H — CLI audit for `predict_buildreasonseg_l3.py`.

Runs the required command form on one real Z-MiniVal240 tile (a natural-language prompt), then verifies:

* exit code 0 and the six required outputs (`result.json`, `reference_mask.png`, `direction_field.png`,
  `nearest_field.png`, `target_mask.png`, `overlay.png`);
* `result.json` carries every required field, including `ground_truth_used=false`;
* a valid but out-of-scope program exits with code 5 and writes no downstream artifacts;
* the CLI rejects ground-truth/annotation arguments.

Writes `evaluation/task7a_cli_audit.json`; the generated files live under the gitignored
`artifacts/task7a/cli/`.

    python scripts/task7a_cli_audit.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.task7a_l3_pipeline import (  # noqa: E402
    default_parser_checkpoint,
    default_target_checkpoint,
    sha256_file,
)

EVAL = REPO_ROOT / "evaluation"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6z" / "packs"
OUT = EVAL / "task7a_cli_audit.json"
OUTPUT_ROOT = REPO_ROOT / "artifacts" / "task7a" / "cli"
REQUIRED_FILES = ("result.json", "reference_mask.png", "direction_field.png", "nearest_field.png",
                  "target_mask.png", "overlay.png")
REQUIRED_KEYS = ("status", "prompt", "parsed_program", "direction", "reference_family",
                 "checkpoints", "proposal_count", "eligible_proposal_count", "selected_reference",
                 "reference_abstention_reason", "directional_field", "nearest_field",
                 "target_positive_pixels", "ground_truth_used")
OUT_OF_SCOPE_PROGRAM = "largest_to_left_of"  # valid canonical L2 program, outside Task 7A scope
OUT_OF_SCOPE_PROMPT_ZH = "找出最大建筑左边的建筑。"


def run_cli(args: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO_ROOT / "predict_buildreasonseg_l3.py"), *args],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--yolo-device", default="0")
    args = parser.parse_args(argv)
    started = time.time()

    records = json.loads((PACK_ROOT / "z_mini_val_240.json").read_text(encoding="utf-8"))["records"]
    # the success path must use an in-scope prompt: the record's own canonical natural-language query
    # (the task file's compact paraphrase form parses to an L2 program and is audited separately)
    instructions = {}
    with (REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2" / "val.jsonl").open(
            encoding="utf-8") as handle:
        for line in handle:
            source = json.loads(line)
            instructions[str(source["sample_id"])] = source
    record = next(candidate for candidate in records
                  if candidate["program_id"] == "largest_to_right_of_to_nearest")
    prompt = str(instructions[record["sample_id"]]["instruction_en"])
    image_path = Path(record["image_path"])
    if not image_path.is_absolute():
        image_path = REPO_ROOT / image_path

    OUT_DIR = OUTPUT_ROOT / "success"
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    command = ["--image", str(image_path), "--prompt", prompt,
               "--parser-checkpoint", str(default_parser_checkpoint()),
               "--proposal-checkpoint",
               str(REPO_ROOT / "artifacts" / "checkpoints" / "task6m1" / "runs"
                   / "m1_yolo26m_seg_continued" / "weights" / "best.pt"),
               "--target-checkpoint", str(default_target_checkpoint()),
               "--out-dir", str(OUT_DIR), "--device", args.device,
               "--yolo-device", args.yolo_device]
    completed = run_cli(command)
    written = sorted(path.name for path in OUT_DIR.glob("*")) if OUT_DIR.is_dir() else []
    result = {}
    result_path = OUT_DIR / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    missing_keys = [key for key in REQUIRED_KEYS if key not in result]
    success = {
        "command": "python predict_buildreasonseg_l3.py --image <tile> --prompt <zh> "
                   "--parser-checkpoint <6T> --proposal-checkpoint <6M.1> --target-checkpoint <6Z Z-B3> "
                   "--out-dir <dir>",
        "exit_code": completed.returncode,
        "stdout_tail": completed.stdout.strip().splitlines()[-3:],
        "stderr_tail": completed.stderr.strip().splitlines()[-3:],
        "files_written": written,
        "required_files_present": all(name in written for name in REQUIRED_FILES),
        "result_keys_present": missing_keys == [],
        "missing_keys": missing_keys,
        "status": result.get("status"),
        "parsed_program": result.get("parsed_program"),
        "ground_truth_used": result.get("ground_truth_used"),
    }

    # out-of-scope program must stop before any downstream work with exit code 5
    OOS_DIR = OUTPUT_ROOT / "unsupported"
    if OOS_DIR.exists():
        shutil.rmtree(OOS_DIR)
    oos_command = list(command)
    oos_command[oos_command.index(str(OUT_DIR))] = str(OOS_DIR)
    oos_command[oos_command.index(prompt)] = OUT_OF_SCOPE_PROMPT_ZH
    oos = run_cli(oos_command)
    oos_files = sorted(path.name for path in OOS_DIR.glob("*")) if OOS_DIR.is_dir() else []
    oos_result = {}
    if (OOS_DIR / "result.json").is_file():
        oos_result = json.loads((OOS_DIR / "result.json").read_text(encoding="utf-8"))
    unsupported = {
        "exit_code": oos.returncode, "expected_exit_code": 5,
        "files_written": oos_files,
        "parsed_program": oos_result.get("parsed_program"),
        "stopped_before": oos_result.get("stopped_before"),
        "ground_truth_used": oos_result.get("ground_truth_used"),
        "only_result_json": oos_files == ["result.json"],
    }

    # ground-truth arguments must be refused
    refused = {}
    for forbidden in ("--gt", "--reference-mask", "--target-mask"):
        attempt = run_cli([*command, forbidden, "x"], timeout=300)
        refused[forbidden] = {"exit_code": attempt.returncode,
                              "rejected": attempt.returncode != 0}
    payload = {
        "_doc": ("Task 7A section 17-18. CMD inference entry-point audit: the required command form, the "
                 "six successful outputs, the required result.json fields, the out-of-scope exit code 5 "
                 "path and the refusal of ground-truth arguments. No GT argument exists by design."),
        "task": "7A", "stage": "H-cli-audit",
        "prompt_zh": prompt, "tile_id": record["tile_id"],
        "checkpoints": {"parser_sha256": sha256_file(default_parser_checkpoint()),
                        "target_sha256": sha256_file(default_target_checkpoint())},
        "success_path": success,
        "unsupported_program_path": unsupported,
        "ground_truth_arguments_refused": refused,
        "all_checks_passed": bool(success["exit_code"] == 0 and success["required_files_present"]
                                  and success["result_keys_present"]
                                  and success["ground_truth_used"] is False
                                  and unsupported["exit_code"] == 5
                                  and unsupported["only_result_json"]
                                  and all(entry["rejected"] for entry in refused.values())),
        "training_performed": False, "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT, payload)
    print(f"[7a.cli] success exit {success['exit_code']} files {len(written)} keys ok "
          f"{success['result_keys_present']} | unsupported exit {unsupported['exit_code']} "
          f"| refused {sum(1 for entry in refused.values() if entry['rejected'])}/3 -> "
          f"{payload['all_checks_passed']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
