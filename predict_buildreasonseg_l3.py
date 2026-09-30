"""Task 7A CMD inference entry point for the four supported L3 programs.

Example (section 17):

```cmd
python predict_buildreasonseg_l3.py ^
  --image path\to\image.tif ^
  --prompt "分割面积最大的建筑物右侧最近的建筑物。" ^
  --parser-checkpoint <Task6T hardened checkpoint> ^
  --proposal-checkpoint artifacts\checkpoints\task6m1\runs\m1_yolo26m_seg_continued\weights\best.pt ^
  --target-checkpoint <Task6Z Z-B3 checkpoint> ^
  --out-dir outputs\buildreasonseg_l3
```

There is **no** ground-truth/annotation argument. Successful inference writes `result.json`,
`reference_mask.png`, `direction_field.png`, `nearest_field.png`, `target_mask.png` and `overlay.png`.
A valid but non-Task-7A program exits with code 5 and stops before any proposal/SAM2/Z-B3 work.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.task7a_l3_pipeline import (  # noqa: E402
    SUPPORTED_L3_PROGRAMS,
    L3Pipeline,
    default_parser_checkpoint,
    default_target_checkpoint,
    sha256_file,
)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_RUNTIME = 4
EXIT_UNSUPPORTED_PROGRAM = 5
FORBIDDEN_ARGUMENTS = ("--reference-mask", "--gt", "--ground-truth", "--annotation",
                       "--target-mask", "--candidate-ids", "--source-feature-id")


def save_image(array: np.ndarray, path: Path, *, mode: str = "L") -> None:
    from PIL import Image

    if mode == "L":
        image = Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    else:
        image = Image.fromarray(array.astype(np.uint8), mode="RGB")
    image.save(path)


def overlay_image(rgb: np.ndarray, reference: np.ndarray | None, target: np.ndarray | None) -> np.ndarray:
    canvas = np.array(rgb, dtype=np.uint8).copy()
    if reference is not None and reference.any():
        canvas[reference] = (0.4 * canvas[reference] + 0.6 * np.array([255, 64, 64])).astype(np.uint8)
    if target is not None and target.any():
        canvas[target] = (0.4 * canvas[target] + 0.6 * np.array([64, 128, 255])).astype(np.uint8)
    return canvas


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--parser-checkpoint", type=Path, default=None)
    parser.add_argument("--proposal-checkpoint", required=True, type=Path)
    parser.add_argument("--target-checkpoint", type=Path, default=None)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--yolo-device", default="0")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    for forbidden in FORBIDDEN_ARGUMENTS:
        if any(argument == forbidden or argument.startswith(forbidden + "=") for argument in argv):
            print(f"[7a.cli] ground-truth argument {forbidden} is not accepted", file=sys.stderr)
            return EXIT_ERROR
    args = build_arg_parser().parse_args(argv)
    started = time.perf_counter()

    from buildreasonseg_mvp.program_parser import build_program_parser, load_parser_checkpoint
    from buildreasonseg_mvp.runtime import load_config
    from buildreasonseg_mvp.structured_grounding import EXPECTED_QUERY_TYPES
    from buildreasonseg_mvp.task6s_directional_pipeline import parse_instruction

    parser_checkpoint = Path(args.parser_checkpoint) if args.parser_checkpoint \
        else default_parser_checkpoint()
    target_checkpoint = Path(args.target_checkpoint) if args.target_checkpoint \
        else default_target_checkpoint()
    for path in (args.image, args.proposal_checkpoint, parser_checkpoint, target_checkpoint):
        if not Path(path).is_file():
            print(f"[7a.cli] missing file: {path}", file=sys.stderr)
            return EXIT_ERROR

    config = load_config(REPO_ROOT / "configs" / "mvp" / "task6j_program_parser.yaml")
    runtime = build_program_parser(config, device=args.device, verbose=False)
    load_parser_checkpoint(parser_checkpoint, runtime)
    parsed = parse_instruction(runtime, args.prompt, tuple(EXPECTED_QUERY_TYPES))
    program = str(parsed["program"])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if program not in SUPPORTED_L3_PROGRAMS:
        payload = {
            "status": "unsupported_l3_program", "prompt": args.prompt, "parsed_program": program,
            "supported_programs": list(SUPPORTED_L3_PROGRAMS),
            "parsed_program_is_canonical": True,
            "direction": None, "reference_family": "largest",
            "checkpoints": {"parser_sha256": sha256_file(parser_checkpoint),
                            "proposal_sha256": sha256_file(args.proposal_checkpoint),
                            "target_sha256": sha256_file(target_checkpoint)},
            "stopped_before": ["proposals", "reference", "fields", "sam2", "z_b3"],
            "ground_truth_used": False,
        }
        write_json(args.out_dir / "result.json", payload)
        print(f"[7a.cli] program {program} is outside Task 7A scope; exit code "
              f"{EXIT_UNSUPPORTED_PROGRAM}", flush=True)
        return EXIT_UNSUPPORTED_PROGRAM

    try:
        pipeline = L3Pipeline(device=args.device, yolo_device=args.yolo_device,
                              proposal_checkpoint=Path(args.proposal_checkpoint),
                              target_checkpoint=target_checkpoint)
        tile_id = Path(args.image).stem
        outcome = pipeline.infer(image_path=Path(args.image), tile_id=tile_id, program_id=program)
    except Exception as error:  # pragma: no cover - runtime failure path
        write_json(args.out_dir / "result.json",
                   {"status": "runtime_error", "prompt": args.prompt, "parsed_program": program,
                    "error": str(error), "ground_truth_used": False})
        print(f"[7a.cli] runtime error: {error}", file=sys.stderr)
        return EXIT_RUNTIME

    fields = outcome.get("fields") or {}
    selection = outcome.get("selection") or {}
    direction = program.split("_to_")[1].replace("_to_nearest", "")
    payload = {
        "status": outcome["status"], "prompt": args.prompt, "parsed_program": program,
        "direction": direction, "reference_family": "largest",
        "checkpoints": {"parser_sha256": sha256_file(parser_checkpoint),
                        "proposal_sha256": sha256_file(args.proposal_checkpoint),
                        "target_sha256": pipeline.target_sha256,
                        "target_variant": pipeline.target_variant},
        "proposal_count": outcome.get("proposal_count"),
        "eligible_proposal_count": outcome.get("eligible_count"),
        "selected_reference": {
            "index": selection.get("index"), "confidence": selection.get("confidence"),
            "area_px": selection.get("area_px"), "bbox_xyxy": selection.get("bbox_xyxy"),
            "reason": selection.get("reason"),
        } if selection else None,
        "reference_abstention_reason": outcome.get("abstention_reason"),
        "directional_field": None, "nearest_field": None,
        "target_positive_pixels": None,
        "runtime_seconds": outcome.get("timing", {}).get("total_seconds")
        if isinstance(outcome.get("timing"), dict) else None,
        "runtime_breakdown": outcome.get("timing") if isinstance(outcome.get("timing"), dict) else None,
        "ground_truth_used": False,
    }
    if fields:
        for name, key in (("directional_field", "P_dir_512"), ("nearest_field", "P_near_512")):
            tensor = fields[key].numpy()
            payload[name] = {"min": float(tensor.min()), "max": float(tensor.max()),
                             "mean": float(tensor.mean())}
    if outcome.get("mask") is not None:
        mask = np.asarray(outcome["mask"], dtype=bool)
        payload["target_positive_pixels"] = int(mask.sum())
        save_image(fields["P_dir_512"].numpy(), args.out_dir / "direction_field.png")
        save_image(fields["P_near_512"].numpy(), args.out_dir / "nearest_field.png")
        save_image(mask.astype(np.float32), args.out_dir / "target_mask.png")
        save_image(np.asarray(outcome["reference_mask"], dtype=np.float32),
                   args.out_dir / "reference_mask.png")
        from buildreasonseg_mvp.task6n_relation_decoder import read_rgb_tile

        overlay = overlay_image(read_rgb_tile(Path(args.image)),
                                np.asarray(outcome["reference_mask"], dtype=bool), mask)
        save_image(overlay, args.out_dir / "overlay.png", mode="RGB")
    write_json(args.out_dir / "result.json", payload)
    print(f"[7a.cli] {program}: status {payload['status']}, target pixels "
          f"{payload['target_positive_pixels']} -> {args.out_dir}", flush=True)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
