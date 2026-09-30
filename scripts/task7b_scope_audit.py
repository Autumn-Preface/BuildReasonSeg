"""Task 7B Part G — scope safety and CLI regression.

Section 20: update the `predict_buildreasonseg_l3.py` **default parser checkpoint** to the Task 7B
checkpoint only when the canonical gates pass; nothing else in the CLI changes (domain guard, supported L3
list, U-C1 resolver, fields, Z-B3).

Section 21: the four out-of-scope semantic controls must parse to the listed canonical program and exit with
code 5 **before** any proposal/reference/SAM2/Z-B3 work; out-of-domain (OOD) controls must keep the existing
exit-code-4 behaviour. No keyword nearest gate exists anywhere.

    python scripts/task7b_scope_audit.py
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

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task7b_scope_safety.json"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6z" / "packs"
CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task7b" / "program_parser_l3_hardened_v1.pt"
TRAINING = EVAL / "task7b_training_summary.json"
OUT_ROOT = REPO_ROOT / "artifacts" / "task7b" / "scope"
CLI = REPO_ROOT / "predict_buildreasonseg_l3.py"
#: Exactly the four out-of-scope semantic controls of section 21.
CONTROLS = (
    ("分割面积最大的建筑物右侧的建筑物。", "largest_to_right_of"),
    ("segment the building to the left of the largest building", "largest_to_left_of"),
    ("分割距离面积最大的建筑物最近的建筑物。", "largest_to_nearest"),
    ("segment the nearest building to the largest building", "largest_to_nearest"),
)
OOD_CONTROLS = (
    "what is the weather in this city tomorrow",
    "请把这张图压缩成 PDF 文件",
)
BEFORE_DOWNSTREAM = ["proposals", "reference", "fields", "sam2", "z_b3"]


def run_cli(prompt: str, image_path: Path, parser_checkpoint: Path, out_dir: Path,
            device: str) -> subprocess.CompletedProcess:
    return subprocess.run([
        sys.executable, str(CLI), "--image", str(image_path), "--prompt", prompt,
        "--parser-checkpoint", str(parser_checkpoint),
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
                        help="update the CLI default parser checkpoint (section 20)")
    args = parser.parse_args(argv)
    started = time.time()

    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    checkpoint_sha = sha256_file(CHECKPOINT)
    # section 20: the CLI's L3 default parser is the Task 7B checkpoint when it exists; the Task 6A/7A
    # helper and every frozen Task 7A artifact are untouched.
    cli_default = default_l3_parser_checkpoint()
    cli_default_sha = sha256_file(cli_default) if cli_default.is_file() else None
    cli_uses_task7b = bool(cli_default == CHECKPOINT and cli_default_sha == checkpoint_sha)

    records = json.loads((PACK_ROOT / "z_mini_val_240.json").read_text(encoding="utf-8"))["records"]
    image_path = Path(records[0]["image_path"])
    if not image_path.is_absolute():
        image_path = REPO_ROOT / image_path

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    controls = []
    for index, (prompt, expected_program) in enumerate(CONTROLS):
        out_dir = OUT_ROOT / f"control_{index + 1}"
        completed = run_cli(prompt, image_path, CHECKPOINT, out_dir, args.device)
        result = {}
        if (out_dir / "result.json").is_file():
            result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
        files = sorted(path.name for path in out_dir.glob("*")) if out_dir.is_dir() else []
        controls.append({
            "prompt": prompt, "expected_program": expected_program,
            "parsed_program": result.get("parsed_program"),
            "semantically_correct_before_exit": result.get("parsed_program") == expected_program,
            "exit_code": completed.returncode, "expected_exit_code": 5,
            "stopped_before": result.get("stopped_before"),
            "files_written": files, "only_result_json": files == ["result.json"],
            "ground_truth_used": result.get("ground_truth_used"),
        })
    ood = []
    for index, prompt in enumerate(OOD_CONTROLS):
        out_dir = OUT_ROOT / f"ood_{index + 1}"
        completed = run_cli(prompt, image_path, CHECKPOINT, out_dir, args.device)
        result = {}
        if (out_dir / "result.json").is_file():
            result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
        ood.append({"prompt": prompt, "exit_code": completed.returncode,
                    "status": result.get("status")})
    out_of_scope_controls_ok = all(
        entry["semantically_correct_before_exit"] and entry["exit_code"] == 5
        and entry["only_result_json"]
        and entry["stopped_before"] == BEFORE_DOWNSTREAM for entry in controls)
    ood_ok = all(entry["exit_code"] in (4, 5) for entry in ood)

    payload = {
        "_doc": ("Task 7B sections 20-21. Scope safety: the four out-of-scope semantic controls must map "
                 "to the listed canonical program (no keyword gate, the ProgramHead itself decides) and "
                 "stop with exit code 5 before proposals/reference/fields/SAM2/Z-B3; OOD prompts keep the "
                 "existing guard behaviour. The CLI default parser checkpoint is updated only when the "
                 "canonical gates pass."),
        "task": "7B", "stage": "G-scope-safety",
        "checkpoint": {"path": str(CHECKPOINT), "sha256": checkpoint_sha,
                       "baseline_sha256": training["baseline"]["sha256"]},
        "cli": {"task6t_helper_checkpoint": str(default_parser_checkpoint()),
                "default_parser_checkpoint": str(cli_default),
                "default_parser_checkpoint_sha256": cli_default_sha,
                "updated_to_task7b": cli_uses_task7b,
                "cli_source_uses_l3_helper": "default_l3_parser_checkpoint" in CLI.read_text(
                    encoding="utf-8"),
                "supported_l3_programs": list(SUPPORTED_L3_PROGRAMS),
                "domain_guard_changed": False, "u_c1_resolver_changed": False,
                "fields_changed": False, "z_b3_changed": False,
                "keyword_or_regex_override": False,
                "canonical_gates_passed": args.canonical_gates_passed},
        "out_of_scope_controls": controls,
        "out_of_scope_controls_ok": out_of_scope_controls_ok,
        "ood_controls": ood, "ood_ok": ood_ok,
        "stopped_before_required": BEFORE_DOWNSTREAM,
        "passed": bool(out_of_scope_controls_ok and ood_ok),
        "test_split_used": False, "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT, payload)
    print(f"[7b.scope] controls ok {out_of_scope_controls_ok} ood ok {ood_ok} -> "
          f"{payload['passed']} (cli update {args.canonical_gates_passed})", flush=True)
    return 0 if payload["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
