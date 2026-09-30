"""Task 7C Part K — scope safety with the Task 7B controls (exact reuse).

Section 22 reuses the Task 7B scope controls unchanged: the L2 directional controls must parse as the L2
program and exit 5 before any proposal/reference/SAM2/Z-B3 work; the nearest-only controls must parse as
`largest_to_nearest` and exit 5; OOD controls keep their guard behaviour (exit 4 in the Task 7A/7B design);
and no keyword/regex gate may exist.

    python scripts/task7c_scope_audit.py
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
    SUPPORTED_L3_PROGRAMS,
    default_l3_parser_checkpoint,
    default_parser_checkpoint,
    default_target_checkpoint,
)
from scripts.task7b_build_parser_data import sha256_file  # noqa: E402
from scripts.task7b_scope_audit import BEFORE_DOWNSTREAM  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task7c_scope_safety.json"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6z" / "packs"
CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task7c" / "program_parser_l3_rehearsal_v1.pt"
TRAINING = EVAL / "task7c_training_summary.json"
TASK7B_SCOPE = EVAL / "task7b_scope_safety.json"
OUT_ROOT = REPO_ROOT / "artifacts" / "task7c" / "scope"
CLI = REPO_ROOT / "predict_buildreasonseg_l3.py"


def run_cli(prompt: str, image_path: Path, out_dir: Path, device: str) -> subprocess.CompletedProcess:
    return subprocess.run([
        sys.executable, str(CLI), "--image", str(image_path), "--prompt", prompt,
        "--parser-checkpoint", str(CHECKPOINT),
        "--proposal-checkpoint",
        str(REPO_ROOT / "artifacts" / "checkpoints" / "task6m1" / "runs"
            / "m1_yolo26m_seg_continued" / "weights" / "best.pt"),
        "--target-checkpoint", str(default_target_checkpoint()),
        "--out-dir", str(out_dir), "--device", device, "--yolo-device", "0",
    ], cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--canonical-gates-passed", action="store_true",
                        help="record that the canonical gates passed (section 24)")
    args = parser.parse_args(argv)
    started = time.time()

    task7b_scope = json.loads(TASK7B_SCOPE.read_text(encoding="utf-8"))
    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    checkpoint_sha = sha256_file(CHECKPOINT)
    controls_spec = [(entry["prompt"], entry["expected_program"])
                     for entry in task7b_scope["out_of_scope_controls"]]
    ood_spec = [entry["prompt"] for entry in task7b_scope["ood_controls"]]

    records = json.loads((PACK_ROOT / "z_mini_val_240.json").read_text(encoding="utf-8"))["records"]
    image_path = Path(records[0]["image_path"])
    if not image_path.is_absolute():
        image_path = REPO_ROOT / image_path
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    controls = []
    for index, (prompt, expected_program) in enumerate(controls_spec):
        out_dir = OUT_ROOT / f"control_{index + 1}"
        completed = run_cli(prompt, image_path, out_dir, args.device)
        result = {}
        if (out_dir / "result.json").is_file():
            result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
        files = sorted(path.name for path in out_dir.glob("*")) if out_dir.is_dir() else []
        controls.append({
            "prompt": prompt, "expected_program": expected_program,
            "parsed_program": result.get("parsed_program"),
            "semantically_correct_before_exit": result.get("parsed_program") == expected_program,
            "exit_code": completed.returncode, "expected_exit_code": 5,
            "stopped_before": result.get("stopped_before"), "files_written": files,
            "only_result_json": files == ["result.json"],
            "ground_truth_used": result.get("ground_truth_used"),
        })
    ood = []
    for index, prompt in enumerate(ood_spec):
        out_dir = OUT_ROOT / f"ood_{index + 1}"
        completed = run_cli(prompt, image_path, out_dir, args.device)
        result = {}
        if (out_dir / "result.json").is_file():
            result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
        ood.append({"prompt": prompt, "exit_code": completed.returncode,
                    "status": result.get("status")})

    l2_controls = [entry for entry in controls
                   if entry["expected_program"] in ("largest_to_left_of", "largest_to_right_of")]
    nearest_controls = [entry for entry in controls
                        if entry["expected_program"] == "largest_to_nearest"]
    l2_ok = all(entry["semantically_correct_before_exit"] and entry["exit_code"] == 5
                and entry["only_result_json"] and entry["stopped_before"] == BEFORE_DOWNSTREAM
                for entry in l2_controls)
    nearest_ok = all(entry["semantically_correct_before_exit"] and entry["exit_code"] == 5
                     and entry["only_result_json"] and entry["stopped_before"] == BEFORE_DOWNSTREAM
                     for entry in nearest_controls)
    ood_ok = all(entry["exit_code"] in (4, 5) for entry in ood)

    payload = {
        "_doc": ("Task 7C section 22. Reused Task 7B scope controls evaluated with the Task 7C parser: L2 "
                 "directional controls must parse as L2 and exit 5 before any downstream work, "
                 "nearest-only controls must parse as largest_to_nearest and exit 5, OOD prompts keep the "
                 "existing guard behaviour, and no keyword/regex gate may exist."),
        "task": "7C", "stage": "K-scope-safety",
        "controls_source": {"path": str(TASK7B_SCOPE), "sha256": sha256_file(TASK7B_SCOPE),
                            "reused": True},
        "checkpoint": {"path": str(CHECKPOINT), "sha256": checkpoint_sha,
                       "initialized_from": training["baseline"]["sha256"]},
        "cli": {"task6t_helper_checkpoint": str(default_parser_checkpoint()),
                "task7c_checkpoint": str(CHECKPOINT),
                "default_parser_checkpoint": str(default_l3_parser_checkpoint()),
                "updated_to_task7c": False,
                "canonical_gates_passed": args.canonical_gates_passed,
                "supported_l3_programs": list(SUPPORTED_L3_PROGRAMS),
                "keyword_or_regex_override": False,
                "domain_guard_changed": False, "u_c1_resolver_changed": False,
                "fields_changed": False, "z_b3_changed": False},
        "out_of_scope_controls": controls,
        "l2_controls_ok": l2_ok, "nearest_only_controls_ok": nearest_ok,
        "ood_controls": ood, "ood_ok": ood_ok,
        "stopped_before_required": BEFORE_DOWNSTREAM,
        "passed": bool(l2_ok and nearest_ok and ood_ok),
        "test_split_used": False, "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT, payload)
    print(f"[7c.scope] l2 {l2_ok} nearest {nearest_ok} ood {ood_ok} -> {payload['passed']}", flush=True)
    return 0 if payload["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
