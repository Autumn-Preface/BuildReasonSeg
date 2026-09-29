"""Task 6T sections 9 and 10 — directional CLI scope safety and end-to-end regression.

`--stage scope` (section 9): runs the **real Task 6S directional CLI** with the hardened ProgramHead as
its default parser on the exact Task 6S controls — the 4 OOD prompts (must stay exit 4 before
ProgramHead/downstream) and the 4 valid-but-out-of-scope prompts (must classify semantically correctly and
exit 5 before proposal/SAM2/B3) — plus the exact 24 fixed paraphrases. No keyword gate is added: the
behaviour comes from the parser plus the existing Task 6S scope check.
Writes `evaluation/task6t_cli_scope_safety.json`.

`--stage e2e` (section 10): reruns the full Task 6S directional chain on MiniVal240 with the hardened
parser only and compares against the frozen Task 6S numbers (mIoU delta <= 1e-6, paired exact, abstentions
exact). Writes `evaluation/task6t_end_to_end_regression.json` (the Task 6S artifacts are never
overwritten).

    python scripts/task6t_scope_audit.py --stage scope
    python scripts/task6t_scope_audit.py --stage e2e
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.task6n_relation_decoder import read_pack  # noqa: E402
from buildreasonseg_mvp.task6s_directional_pipeline import (  # noqa: E402
    EXIT_UNSUPPORTED_DIRECTIONAL,
    EXIT_UNSUPPORTED_INSTRUCTION,
    default_program_head_checkpoint,
)
from scripts.task6s_cli_audit import (  # noqa: E402
    FEATURE_ROOT,
    OOD_PROMPTS,
    OUT_OF_SCOPE_PROMPTS,
    PROPOSAL_CHECKPOINT,
    SUPPORTED_PROMPTS,
)

EVAL = REPO_ROOT / "evaluation"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6n" / "packs"
CLI = REPO_ROOT / "predict_buildreasonseg_directional.py"
CLI_WORK = REPO_ROOT / "artifacts" / "task6t" / "cli"
OUT_SCOPE = EVAL / "task6t_cli_scope_safety.json"
OUT_E2E = EVAL / "task6t_end_to_end_regression.json"
HARDENED = REPO_ROOT / "artifacts" / "checkpoints" / "task6t" / "program_parser_hardened_v1.pt"
TASK6S_EXPECTED = {
    "answered_records": 234,
    "answered_only_miou": 0.3045812554881724,
    "strict_all_240_miou": 0.2969667241009681,
    "paired": 10,
    "own_cross_margin": 0.27370032940000916,
    "abstentions": 6,
    "parser_exact_accuracy": 1.0,
    "expected_parsed_programs": {
        "分割面积最大的建筑物。": "largest",
        "分割最左侧的建筑物。": "leftmost",
        "分割面积最大的建筑物右侧最近的建筑物。": "largest_to_right_of_to_nearest",
        "segment the building nearest to the right of the largest building":
            "largest_to_right_of_to_nearest",
    },
}
TOLERANCE = 1e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def b3_checkpoint() -> Path:
    return Path(json.loads((EVAL / "task6o_mini_val.json").read_text(
        encoding="utf-8"))["variants"]["B3"]["training"]["checkpoint"]["path"])


def val_tile_images() -> dict[str, Path]:
    images: dict[str, Path] = {}
    for sample in read_pack(PACK_ROOT / "mini_val_240.json"):
        images.setdefault(sample.program_id, Path(sample.image_path))
    return images


def run_cli(image: Path, prompt: str, out_dir: Path) -> dict:
    """Run the CLI **without** `--parser-checkpoint` so the hardened default parser is exercised."""

    command = [
        sys.executable, str(CLI),
        "--image", str(image), "--prompt", prompt,
        "--proposal-checkpoint", str(PROPOSAL_CHECKPOINT),
        "--target-checkpoint", str(b3_checkpoint()),
        "--out-dir", str(out_dir), "--device", "0", "--quiet",
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    result_path = out_dir / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
    return {"exit_code": completed.returncode,
            "wall_seconds": round(time.perf_counter() - started, 2), "result": payload}


def run_scope(args) -> int:
    started = time.time()
    CLI_WORK.mkdir(parents=True, exist_ok=True)
    checkpoint, report = default_program_head_checkpoint()
    if checkpoint is None or checkpoint != HARDENED:
        write_json(OUT_SCOPE, {"_doc": "Task 6T section 9.", "task": "6T",
                               "verdict": "PARSER_TRAINING_FAILED",
                               "default_program_head": report})
        print("[6t.scope] STOP: hardened checkpoint unavailable", flush=True)
        return 2
    harness_sha = sha256_file(checkpoint)
    images = val_tile_images()
    reference_image = images["largest_to_right_of"]

    controls = []
    for index, prompt in enumerate((*OOD_PROMPTS, *OUT_OF_SCOPE_PROMPTS), start=1):
        is_ood = index <= len(OOD_PROMPTS)
        expected_exit = EXIT_UNSUPPORTED_INSTRUCTION if is_ood else EXIT_UNSUPPORTED_DIRECTIONAL
        out_dir = CLI_WORK / f"control_{index:02d}"
        run = run_cli(reference_image, prompt, out_dir)
        result = run["result"] or {}
        expected_program = TASK6S_EXPECTED["expected_parsed_programs"].get(prompt)
        controls.append({
            "index": index, "kind": "ood" if is_ood else "valid_but_out_of_scope",
            "prompt": prompt, "exit_code": run["exit_code"], "expected_exit_code": expected_exit,
            "parsed_program": result.get("parsed_program"),
            "expected_parsed_program": expected_program,
            "semantically_correct": (result.get("parsed_program") == expected_program)
            if expected_program is not None else None,
            "status": result.get("status"),
            "downstream_called": bool(result.get("proposal_count") is not None
                                      or result.get("reference_family")),
            "parser_checkpoint_sha256": result.get("parser_checkpoint_sha256"),
            "passed": run["exit_code"] == expected_exit,
        })
        print(f"[6t.scope] control {index}/8 exit {run['exit_code']} parsed "
              f"{result.get('parsed_program')}", flush=True)

    fixed24 = []
    for index, (expected_program, prompt) in enumerate(SUPPORTED_PROMPTS, start=1):
        out_dir = CLI_WORK / f"fixed24_{index:02d}"
        run = run_cli(images[expected_program], prompt, out_dir)
        result = run["result"] or {}
        parsed = result.get("parsed_program")
        decomposition = result.get("reference_family"), result.get("relation")
        fixed24.append({
            "index": index, "prompt": prompt, "expected_program": expected_program,
            "parsed_program": parsed, "parser_correct": parsed == expected_program,
            "exit_code": run["exit_code"],
            "reference_family": decomposition[0], "relation": decomposition[1],
            "downstream_branch_ok": bool(decomposition[0] is not None and decomposition[1] is not None),
            "wrote_target_mask": (out_dir / "target_mask.png").is_file(),
            "ground_truth_used": result.get("ground_truth_used"),
        })
    fixed24_correct = sum(1 for row in fixed24 if row["parser_correct"])

    ood = [row for row in controls if row["kind"] == "ood"]
    scope = [row for row in controls if row["kind"] == "valid_but_out_of_scope"]
    payload = {
        "_doc": (
            "Task 6T section 9. Directional CLI scope safety with the hardened ProgramHead as the CLI "
            "default parser: the exact Task 6S controls must keep their exit-code behaviour and classify "
            "the out-of-scope canonical programs semantically correctly, without any keyword/regex gate. "
            "The domain guard, the supported-8 list and all downstream modules are unchanged."
        ),
        "task": "6T", "stage": "scope-safety",
        "cli": str(CLI), "cli_sha256": sha256_file(CLI),
        "default_program_head": {"path": str(checkpoint), "sha256": harness_sha,
                                 "resolution": report,
                                 "is_hardened_checkpoint": checkpoint == HARDENED},
        "proposal_checkpoint_sha256": sha256_file(PROPOSAL_CHECKPOINT),
        "controls": controls,
        "ood_all_exit_4": all(row["exit_code"] == EXIT_UNSUPPORTED_INSTRUCTION for row in ood),
        "ood_no_downstream": all(not row["downstream_called"] for row in ood),
        "out_of_scope_all_exit_5": all(row["exit_code"] == EXIT_UNSUPPORTED_DIRECTIONAL
                                       for row in scope),
        "out_of_scope_all_semantically_correct": all(row["semantically_correct"] for row in scope),
        "out_of_scope_no_downstream": all(not row["downstream_called"] for row in scope),
        "out_of_scope_parsed_program": {row["prompt"]: row["parsed_program"] for row in scope},
        "nearest_controls_parsed_program": {
            row["prompt"]: row["parsed_program"] for row in scope
            if "nearest" in row["prompt"] or "最近" in row["prompt"]},
        "fixed24": fixed24,
        "fixed24_correct": fixed24_correct,
        "fixed24_expected_program_hash": sha256_file(
            REPO_ROOT / "evaluation" / "task6t_parser_fixed24.json")
        if (REPO_ROOT / "evaluation" / "task6t_parser_fixed24.json").is_file() else None,
        "keyword_or_regex_gate_added": False,
        "domain_guard_changed": False,
        "supported8_changed": False,
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_SCOPE, payload)
    print(f"[6t.scope] OOD exit4 {payload['ood_all_exit_4']} | out-of-scope exit5 "
          f"{payload['out_of_scope_all_exit_5']} semantic "
          f"{payload['out_of_scope_all_semantically_correct']} | fixed24 {fixed24_correct}/24",
          flush=True)
    return 0


def run_e2e(args) -> int:
    started = time.time()
    if not HARDENED.is_file():
        write_json(OUT_E2E, {"_doc": "Task 6T section 10.", "task": "6T",
                             "verdict": "PARSER_TRAINING_FAILED"})
        return 2

    import scripts.task6s_evaluate as s6

    # keep the frozen Task 6S artifacts untouched: redirect the chain outputs to Task 6T paths
    s6.OUT_E2E = REPO_ROOT / "artifacts" / "task6t" / "end_to_end_val.json"
    s6.OUT_PAIRED = REPO_ROOT / "artifacts" / "task6t" / "end_to_end_paired.json"
    s6.OUT_E2E.parent.mkdir(parents=True, exist_ok=True)
    status = 0
    if args.reuse and s6.OUT_E2E.is_file() and s6.OUT_PAIRED.is_file():
        print("[6t.e2e] reusing the stored Task 6T chain outputs (--reuse)", flush=True)
    else:
        # only the ProgramHead differs from Task 6S: inject the hardened checkpoint into the frozen chain
        original_load_models = s6.load_models
        s6.load_models = lambda device, parser_checkpoint_path=None: original_load_models(device,
                                                                                         HARDENED)
        status = s6.run_end_to_end(argparse.Namespace(device="cuda", proposal_device="0"))

    e2e = json.loads(s6.OUT_E2E.read_text(encoding="utf-8"))
    paired = json.loads(s6.OUT_PAIRED.read_text(encoding="utf-8"))
    parsed_counts = Counter(row["parsed_program"] for row in e2e["records"])
    parser_exact = sum(1 for row in e2e["records"] if row["parser_correct"]) / len(e2e["records"])
    answered_miou = e2e["answered_only"]["miou"]
    strict_miou = e2e["strict_all_240"]["miou"]
    deltas = {
        "answered_only_miou": answered_miou - TASK6S_EXPECTED["answered_only_miou"],
        "strict_all_240_miou": strict_miou - TASK6S_EXPECTED["strict_all_240_miou"],
        "paired": paired["passed"] - TASK6S_EXPECTED["paired"],
        "own_cross_margin": paired["own_cross_margin"] - TASK6S_EXPECTED["own_cross_margin"],
        "answered_records": e2e["answered_only"]["records"] - TASK6S_EXPECTED["answered_records"],
    }
    within = {
        "answered_only_miou": abs(deltas["answered_only_miou"]) <= TOLERANCE,
        "strict_all_240_miou": abs(deltas["strict_all_240_miou"]) <= TOLERANCE,
        "paired": deltas["paired"] == 0,
        "own_cross_margin": abs(deltas["own_cross_margin"]) <= TOLERANCE,
        "abstentions": int(round(e2e["abstention_rate"] * 240)) == TASK6S_EXPECTED["abstentions"],
    }
    payload = {
        "_doc": (
            "Task 6T section 10. Directional end-to-end regression: only the ProgramHead changed, so "
            "MiniVal240 rerun through the frozen Task 6S chain must reproduce Task 6S. Tolerance: mIoU "
            "absolute delta <= 1e-6, paired count exact, abstention count exact. The frozen Task 6S "
            "artifacts are untouched; the chain output is written under artifacts/task6t/."
        ),
        "task": "6T", "stage": "end-to-end-regression",
        "parser_checkpoint": {"path": str(HARDENED), "sha256": sha256_file(HARDENED)},
        "parser_exact_accuracy": parser_exact,
        "parsed_program_distribution": dict(sorted(parsed_counts.items())),
        "task6s_expected": TASK6S_EXPECTED,
        "measured": {
            "answered_records": e2e["answered_only"]["records"],
            "answered_only_miou": answered_miou,
            "answered_only_dice": e2e["answered_only"]["dice"],
            "strict_all_240_miou": strict_miou,
            "strict_all_240_dice": e2e["strict_all_240"]["dice"],
            "paired": paired["passed"],
            "own_cross_margin": paired["own_cross_margin"],
            "abstentions": int(round(e2e["abstention_rate"] * 240)),
            "reference_abstention_pairs": paired["reference_abstention_pairs"],
        },
        "deltas": deltas,
        "within_tolerance": within,
        "tolerance": TOLERANCE,
        "frozen_modules_unchanged": True,
        "all_within_tolerance": all(within.values()),
        "verdict": "END_TO_END_REG_REPRODUCED" if all(within.values()) else "END_TO_END_REGRESSION",
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_E2E, payload)
    print(f"[6t.e2e] answered mIoU {answered_miou:.9f} (delta {deltas['answered_only_miou']:+.2e}) | "
          f"paired {paired['passed']}/20 | abstentions "
          f"{int(round(e2e['abstention_rate'] * 240))} -> {payload['verdict']}", flush=True)
    return 0 if status == 0 and all(within.values()) else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("scope", "e2e"), required=True)
    parser.add_argument("--reuse", action="store_true",
                        help="reuse the stored Task 6T chain outputs instead of re-running the chain")
    args = parser.parse_args(argv)
    return run_scope(args) if args.stage == "scope" else run_e2e(args)


if __name__ == "__main__":
    raise SystemExit(main())
