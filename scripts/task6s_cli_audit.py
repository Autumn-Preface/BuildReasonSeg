"""Task 6S Part F — fixed paraphrase/CLI prompt pack and the on-the-fly SAM2 proof.

`--stage cli` (sections 18-19, 13): runs the **real CLI** (`predict_buildreasonseg_directional.py`) as a
subprocess on the exact 24 fixed supported paraphrases and the 8 unsupported controls, plus an
on-the-fly SAM2 proof on an image that is not addressed through any precomputed Task 6N feature-cache
key. Writes `evaluation/task6s_cli_prompt_audit.json`.

Failure attribution lives in `scripts/task6s_failure_attribution.py` (Part G).

    python scripts/task6s_cli_audit.py --stage cli
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.task6m_eval import write_json  # noqa: E402
from buildreasonseg_mvp.task6n_relation_decoder import read_pack  # noqa: E402
from buildreasonseg_mvp.task6s_directional_pipeline import (  # noqa: E402
    EXIT_UNSUPPORTED_DIRECTIONAL,
    EXIT_UNSUPPORTED_INSTRUCTION,
    PROGRAM_DECOMPOSITION,
    SUPPORTED_PROGRAMS,
)

EVAL = REPO_ROOT / "evaluation"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6n" / "packs"
FEATURE_ROOT = REPO_ROOT / "artifacts" / "task6n" / "features"
CLI = REPO_ROOT / "predict_buildreasonseg_directional.py"
CLI_WORK = REPO_ROOT / "artifacts" / "task6s" / "cli"
OUT_CLI = EVAL / "task6s_cli_prompt_audit.json"
V02 = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2"

PROPOSAL_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6m1" / "runs" \
    / "m1_yolo26m_seg_continued" / "weights" / "best.pt"
PROGRAM_HEAD_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6m" \
    / "program_parser_v02_best.pt"
PROPOSAL_DEVICE = "0"

#: Task 6S section 18 — the exact 24 fixed supported paraphrases (16 Chinese + 8 English).
SUPPORTED_PROMPTS: tuple[tuple[str, str], ...] = (
    ("largest_to_left_of", "分割面积最大的建筑物左侧的建筑物。"),
    ("largest_to_left_of", "找出最大建筑左边的建筑物。"),
    ("largest_to_left_of", "segment the building to the left of the largest building"),
    ("largest_to_right_of", "分割面积最大的建筑物右侧的建筑物。"),
    ("largest_to_right_of", "找出最大建筑右边的建筑物。"),
    ("largest_to_right_of", "segment the building to the right of the largest building"),
    ("largest_to_above", "分割面积最大的建筑物上方的建筑物。"),
    ("largest_to_above", "找出最大建筑上面的建筑物。"),
    ("largest_to_above", "segment the building above the largest building"),
    ("largest_to_below", "分割面积最大的建筑物下方的建筑物。"),
    ("largest_to_below", "找出最大建筑下面的建筑物。"),
    ("largest_to_below", "segment the building below the largest building"),
    ("smallest_to_left_of", "分割面积最小的建筑物左侧的建筑物。"),
    ("smallest_to_left_of", "找出最小建筑左边的建筑物。"),
    ("smallest_to_left_of", "segment the building to the left of the smallest building"),
    ("smallest_to_right_of", "分割面积最小的建筑物右侧的建筑物。"),
    ("smallest_to_right_of", "找出最小建筑右边的建筑物。"),
    ("smallest_to_right_of", "segment the building to the right of the smallest building"),
    ("smallest_to_above", "分割面积最小的建筑物上方的建筑物。"),
    ("smallest_to_above", "找出最小建筑上面的建筑物。"),
    ("smallest_to_above", "segment the building above the smallest building"),
    ("smallest_to_below", "分割面积最小的建筑物下方的建筑物。"),
    ("smallest_to_below", "找出最小建筑下面的建筑物。"),
    ("smallest_to_below", "segment the building below the smallest building"),
)

#: Task 6S section 19 — the exact 8 unsupported controls (4 OOD, 4 valid-but-out-of-scope).
OOD_PROMPTS: tuple[str, ...] = (
    "Write a poem about the sea.",
    "今天天气怎么样？",
    "检测道路。",
    "",
)
OUT_OF_SCOPE_PROMPTS: tuple[str, ...] = (
    "分割面积最大的建筑物。",
    "分割最左侧的建筑物。",
    "分割面积最大的建筑物右侧最近的建筑物。",
    "segment the building nearest to the right of the largest building",
)


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
    command = [
        sys.executable, str(CLI),
        "--image", str(image), "--prompt", prompt,
        "--parser-checkpoint", str(PROGRAM_HEAD_CHECKPOINT),
        "--proposal-checkpoint", str(PROPOSAL_CHECKPOINT),
        "--target-checkpoint", str(b3_checkpoint()),
        "--out-dir", str(out_dir), "--device", PROPOSAL_DEVICE, "--quiet",
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    result_path = out_dir / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
    return {
        "command": [Path(item).name if Path(item).is_absolute() else item for item in command],
        "exit_code": completed.returncode,
        "wall_seconds": round(time.perf_counter() - started, 2),
        "result": payload,
        "stderr_tail": completed.stderr.strip().splitlines()[-3:] if completed.stderr else [],
    }


def run_cli_stage(args) -> int:
    started = time.time()
    CLI_WORK.mkdir(parents=True, exist_ok=True)
    images = val_tile_images()

    # ---------------- 24 fixed supported paraphrases
    prompt_rows = []
    for index, (expected_program, prompt) in enumerate(SUPPORTED_PROMPTS, start=1):
        image = images[expected_program]
        out_dir = CLI_WORK / f"supported_{index:02d}"
        run = run_cli(image, prompt, out_dir)
        result = run["result"] or {}
        parsed = result.get("parsed_program")
        expected_family, expected_relation = PROGRAM_DECOMPOSITION[expected_program]
        prompt_rows.append({
            "index": index, "expected_program": expected_program, "prompt": prompt,
            "image": str(image), "exit_code": run["exit_code"],
            "parsed_program": parsed,
            "parser_correct": parsed == expected_program,
            "supported": parsed in SUPPORTED_PROGRAMS if parsed else False,
            "reference_family": result.get("reference_family"),
            "relation": result.get("relation"),
            "status": result.get("status"),
            "downstream_branch_ok": bool(result.get("reference_family") == expected_family
                                         and result.get("relation") == expected_relation),
            "ground_truth_used": result.get("ground_truth_used"),
            "wrote": sorted(path.name for path in out_dir.glob("*")),
            "wall_seconds": run["wall_seconds"],
        })
        print(f"[6s.cli] supported {index}/24 exit {run['exit_code']} parsed {parsed} "
              f"(expected {expected_program})", flush=True)
    supported_correct = sum(1 for row in prompt_rows if row["parser_correct"])

    # ---------------- 8 unsupported controls
    control_rows = []
    for index, prompt in enumerate((*OOD_PROMPTS, *OUT_OF_SCOPE_PROMPTS), start=1):
        is_ood = index <= len(OOD_PROMPTS)
        expected_exit = EXIT_UNSUPPORTED_INSTRUCTION if is_ood else EXIT_UNSUPPORTED_DIRECTIONAL
        out_dir = CLI_WORK / f"control_{index:02d}"
        run = run_cli(images["largest_to_right_of"], prompt, out_dir)
        result = run["result"] or {}
        control_rows.append({
            "index": index, "kind": "ood" if is_ood else "valid_but_out_of_scope",
            "prompt": prompt, "exit_code": run["exit_code"], "expected_exit_code": expected_exit,
            "status": result.get("status"), "parsed_program": result.get("parsed_program"),
            "downstream_called": bool(result.get("proposal_count") is not None
                                      or result.get("reference_family")),
            "passed": run["exit_code"] == expected_exit,
            "wrote": sorted(path.name for path in out_dir.glob("*")),
        })
        print(f"[6s.cli] control {index}/8 exit {run['exit_code']} "
              f"({control_rows[-1]['kind']})", flush=True)

    # ---------------- on-the-fly SAM2 on a source image with no feature-cache key
    packed_tiles = {sample.tile_id for sample in read_pack(PACK_ROOT / "mini_val_240.json")}
    image_root = REPO_ROOT / "artifacts" / "task6m_yolo_native" / "images" / "val"
    on_the_fly = None
    with (V02 / "val.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            tile_id = record.get("image_id") or (record.get("native_vector") or {}).get("tile_id")
            if not tile_id or tile_id in packed_tiles \
                    or (FEATURE_ROOT / f"{tile_id}.npy").is_file():
                continue
            if record.get("query_type") not in SUPPORTED_PROGRAMS:
                continue  # the audit needs a record whose parsed program reaches SAM2/B3
            image = image_root / f"{tile_id}.tif"
            if not image.is_file():
                continue
            out_dir = CLI_WORK / "on_the_fly"
            run = run_cli(image, record["instruction_en"], out_dir)
            result = run["result"] or {}
            runtime = result.get("runtime") or {}
            on_the_fly = {
                "sample_id": record["sample_id"], "tile_id": tile_id, "image": str(image),
                "feature_cache_key_present": False,
                "in_mini_val_240": False,
                "exit_code": run["exit_code"], "status": result.get("status"),
                "parsed_program": result.get("parsed_program"),
                "sam2_feature_seconds": runtime.get("sam2_feature"),
                "target_positive_pixels": result.get("target_positive_pixels"),
                "sam2_encoder_path_exercised": bool(runtime.get("sam2_feature") is not None
                                                    and runtime["sam2_feature"] > 0.01),
                "wrote": sorted(path.name for path in out_dir.glob("*")),
            }
            print(f"[6s.cli] on-the-fly SAM2 {tile_id}: exit {run['exit_code']} "
                  f"feature {runtime.get('sam2_feature')}s", flush=True)
            break

    payload = {
        "_doc": (
            "Task 6S sections 18-19 and 13. The real CLI executed as a subprocess on the exact 24 fixed "
            "supported paraphrases and the 8 unsupported controls, plus an on-the-fly frozen SAM2 "
            "encoder proof on a source image with no precomputed feature-cache key. No retraining and "
            "no threshold tuning followed any paraphrase failure; GT is not available to the CLI."
        ),
        "task": "6S", "stage": "F-cli-prompt-audit",
        "cli": str(CLI), "cli_sha256": sha256_file(CLI),
        "checkpoints": {
            "program_head": {"path": str(PROGRAM_HEAD_CHECKPOINT),
                             "sha256": sha256_file(PROGRAM_HEAD_CHECKPOINT)},
            "proposal": {"path": str(PROPOSAL_CHECKPOINT),
                         "sha256": sha256_file(PROPOSAL_CHECKPOINT)},
            "target": {"path": str(b3_checkpoint()), "sha256": sha256_file(b3_checkpoint())},
        },
        "supported_prompts": prompt_rows,
        "supported_count": len(prompt_rows),
        "supported_parser_correct": supported_correct,
        "supported_exact_prompts_match_spec": len(prompt_rows) == 24,
        "supported_gate": {"required": 22, "measured": supported_correct,
                           "passed": supported_correct >= 22},
        "controls": control_rows,
        "control_count": len(control_rows),
        "controls_exactly_eight": len(control_rows) == 8,
        "ood_all_exit_4": all(row["exit_code"] == EXIT_UNSUPPORTED_INSTRUCTION
                              for row in control_rows if row["kind"] == "ood"),
        "out_of_scope_all_exit_5": all(row["exit_code"] == EXIT_UNSUPPORTED_DIRECTIONAL
                                       for row in control_rows
                                       if row["kind"] == "valid_but_out_of_scope"),
        "downstream_never_called_for_controls": not any(row["downstream_called"]
                                                        for row in control_rows),
        "on_the_fly_sam2": on_the_fly,
        "ground_truth_required_by_cli": False,
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_CLI, payload)
    print(f"[6s.cli] supported {supported_correct}/24 (gate >=22) | OOD exit4 "
          f"{payload['ood_all_exit_4']} | out-of-scope exit5 "
          f"{payload['out_of_scope_all_exit_5']} | on-the-fly SAM2 "
          f"{bool(on_the_fly and on_the_fly['sam2_encoder_path_exercised'])}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cli",), default="cli")
    args = parser.parse_args(argv)
    return run_cli_stage(args)


if __name__ == "__main__":
    raise SystemExit(main())
