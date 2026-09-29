r"""Task 6S section 11 — clean directional end-to-end CMD entry point.

```cmd
python predict_buildreasonseg_directional.py ^
  --image path\to\image.tif ^
  --prompt "分割面积最大的建筑物右侧的建筑物" ^
  --parser-checkpoint artifacts\checkpoints\task6j\j2_best.pt ^
  --proposal-checkpoint artifacts\checkpoints\task6m1\runs\m1_yolo26m_seg_continued\weights\best.pt ^
  --target-checkpoint <Task6O_B3_checkpoint> ^
  --out-dir outputs\buildreasonseg_directional
```

The CLI needs only an image, a natural-language prompt and the frozen checkpoints — no GT masks, no
annotation JSON, no BuildSpatialReason record, no source feature ids, no oracle reference and no target
id. It writes `result.json`, `target_mask.png`, `overlay.png`, `reference_mask.png` and
`relation_field.png` for a successful supported prompt.

Exit codes: `0` answered, `3` reference abstention, `4` out-of-domain prompt, `5` valid-but-unsupported
canonical program.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from buildreasonseg_mvp.task6s_directional_pipeline import (  # noqa: E402
    EXIT_ANSWERED,
    EXIT_REFERENCE_ABSTENTION,
    EXIT_UNSUPPORTED_DIRECTIONAL,
    EXIT_UNSUPPORTED_INSTRUCTION,
    STATUS_OK,
    SUPPORTED_PROGRAMS,
    ChainModels,
    resolve_program_head_checkpoint,
    run_directional_chain,
    sha256_file,
)

import predict_structured  # noqa: E402  (frozen Task 6M.1 domain gate)

DEFAULT_B3 = (Path("artifacts") / "checkpoints" / "task6o" / "run" / "placeholder")
FEATURE_ROOT = Path("artifacts") / "task6n" / "features"


def load_models(args) -> tuple[ChainModels | None, dict]:
    """Load the frozen ProgramHead, proposal model, SAM2 encoder/store and B3 decoder."""

    provenance: dict = {}
    parser_checkpoint, parser_candidates = resolve_program_head_checkpoint(args.parser_checkpoint)
    provenance["program_head_candidates"] = parser_candidates
    if parser_checkpoint is None:
        return None, provenance
    provenance["program_head"] = {"path": str(parser_checkpoint),
                                  "sha256": sha256_file(parser_checkpoint),
                                  "text_only": True, "retrained": False}

    from buildreasonseg_mvp.program_parser import build_program_parser, load_parser_checkpoint
    from buildreasonseg_mvp.runtime import load_config
    from buildreasonseg_mvp.task6n_relation_decoder import FrozenFeatureStore, load_frozen_sam2_encoder
    from buildreasonseg_mvp.task6q_reference_resolver import PROPOSAL_CHECKPOINT_SHA256
    from ultralytics import YOLO

    config = load_config(REPO_ROOT / "configs" / "mvp" / "task6j_program_parser.yaml")
    parser_runtime = build_program_parser(config, device="cuda", verbose=False)
    load_parser_checkpoint(parser_checkpoint, parser_runtime)

    device = "cuda" if args.device != "cpu" else "cpu"
    proposal_model = YOLO(str(args.proposal_checkpoint))
    provenance["proposal"] = {"path": str(args.proposal_checkpoint),
                              "sha256": sha256_file(args.proposal_checkpoint),
                              "expected_sha256": PROPOSAL_CHECKPOINT_SHA256,
                              "matches_expected": sha256_file(args.proposal_checkpoint)
                              == PROPOSAL_CHECKPOINT_SHA256}
    provenance["proposal_device"] = args.device

    encoder, _ = load_frozen_sam2_encoder(device=device)
    feature_root = Path(args.feature_cache) if args.feature_cache else FEATURE_ROOT
    store = FrozenFeatureStore(feature_root, encoder=encoder, device=device)

    from buildreasonseg_mvp.task6n_relation_decoder import DecoderConfig, RelationMaskDecoder

    b3 = RelationMaskDecoder("B3", DecoderConfig()).to(device)
    import torch

    b3.load_state_dict(torch.load(args.target_checkpoint, map_location=device,
                                  weights_only=False)["state_dict"])
    b3.eval()
    provenance["target_decoder"] = {"path": str(args.target_checkpoint),
                                    "sha256": sha256_file(args.target_checkpoint),
                                    "architecture": "Task 6O B3 (visual + field + relation)",
                                    "retrained": False}
    models = ChainModels(
        parser_runtime=parser_runtime, proposal_model=proposal_model, target_model=b3,
        feature_store=store, device=device, parser_checkpoint=parser_checkpoint,
        proposal_checkpoint=Path(args.proposal_checkpoint),
        target_checkpoint=Path(args.target_checkpoint), provenance=provenance,
    )
    return models, provenance


def save_visuals(out_dir: Path, image_path: Path, result: dict) -> None:
    import cv2

    from buildreasonseg_mvp.task6n_relation_decoder import read_rgb_tile

    out_dir.mkdir(parents=True, exist_ok=True)
    if result["status"] != STATUS_OK:
        return
    target = result["target_mask"].astype(np.uint8) * 255
    reference = np.asarray(result["reference_mask"]).astype(np.uint8) * 255
    field = result["field_array"]
    cv2.imwrite(str(out_dir / "target_mask.png"), target)
    cv2.imwrite(str(out_dir / "reference_mask.png"), reference)
    field_vis = ((np.clip(field, 0.0, 1.0) ** 0.5) * 255).astype(np.uint8)
    field_big = cv2.resize(field_vis, (512, 512), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out_dir / "relation_field.png"), field_big)
    image = read_rgb_tile(image_path)
    overlay = image.copy()
    overlay[reference.astype(bool)] = (0.5 * overlay[reference.astype(bool)]
                                       + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    overlay[target.astype(bool)] = (0.4 * overlay[target.astype(bool)]
                                    + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
    cv2.imwrite(str(out_dir / "overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--parser-checkpoint", type=Path, default=None)
    parser.add_argument("--proposal-checkpoint", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, default=None,
                        help="optional frozen feature cache directory (on-the-fly SAM2 otherwise)")
    parser.add_argument("--device", default="0")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.image.is_file():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 2
    if not args.proposal_checkpoint.is_file():
        print(f"error: proposal checkpoint not found: {args.proposal_checkpoint}", file=sys.stderr)
        return 2
    if not args.target_checkpoint.is_file():
        print(f"error: target checkpoint not found: {args.target_checkpoint}", file=sys.stderr)
        return 2

    # ---------------- domain gate first: no parser, no proposal, no SAM2, no B3
    gate = predict_structured.check_domain(args.prompt)
    if not gate.get("supported", False):
        payload = {
            "status": predict_structured.UNSUPPORTED_STATUS,
            "exit_code": EXIT_UNSUPPORTED_INSTRUCTION,
            "prompt": args.prompt,
            "domain_gate": {**gate, "checked_before_parser": True,
                            "checked_before_proposal": True, "checked_before_sam2": True,
                            "checked_before_target_decoder": True},
            "parsed_program": None, "reference_family": None, "relation": None,
            "image": str(args.image), "ground_truth_used": False,
            "runtime_seconds": round(time.perf_counter() - started, 3),
        }
        (out_dir / "result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[6s] status: {payload['status']} ({gate.get('reason')}); no parser/proposal/"
              f"SAM2/B3 was called")
        return EXIT_UNSUPPORTED_INSTRUCTION

    models, provenance = load_models(args)
    if models is None:
        payload = {"status": "PARSER_CHECKPOINT_UNAVAILABLE", "exit_code": 2,
                   "prompt": args.prompt, "program_head_candidates": provenance,
                   "ground_truth_used": False}
        (out_dir / "result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("[6s] STOP PARSER_CHECKPOINT_UNAVAILABLE", file=sys.stderr)
        return 2

    from buildreasonseg_mvp.structured_grounding import EXPECTED_QUERY_TYPES

    result = run_directional_chain(
        models, args.image, args.prompt, program_ids=tuple(EXPECTED_QUERY_TYPES),
        domain_gate=predict_structured.check_domain, want_visuals=True,
    )
    save_visuals(out_dir, args.image, result)

    target_mask = result.pop("target_mask", None)
    reference_mask = result.pop("reference_mask", None)
    result.pop("field_array", None)
    payload = {
        **result,
        "image": str(args.image),
        "parser_checkpoint_sha256": provenance["program_head"]["sha256"],
        "proposal_checkpoint_sha256": provenance["proposal"]["sha256"],
        "target_checkpoint_sha256": provenance["target_decoder"]["sha256"],
        "parser_checkpoint_path": provenance["program_head"]["path"],
        "proposal_checkpoint_path": provenance["proposal"]["path"],
        "target_checkpoint_path": provenance["target_decoder"]["path"],
        "supported_programs": list(SUPPORTED_PROGRAMS),
        "chain": ["program_head", "decomposition", "task6q_proposal_reference_resolver",
                  "geometric_relation_field_v02", "frozen_sam2_feature", "frozen_task6o_b3"],
        "grcl_used": False,
        "oracle_reference_used": False,
        "ground_truth_used": False,
        "ground_truth_required": False,
        "target_positive_pixels": result.get("target_positive_pixels"),
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    (out_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not args.quiet:
        print(f"[6s] status: {payload['status']}")
        print(f"[6s] parsed program: {payload.get('parsed_program')} "
              f"({payload.get('reference_family')}, {payload.get('relation')})")
        if payload.get("proposal_count") is not None:
            print(f"[6s] proposals: {payload['proposal_count']} "
                  f"(eligible {payload.get('eligible_reference_proposals')})")
        if target_mask is not None:
            print(f"[6s] target positive pixels: {int(target_mask.sum())}")
            for name in ("target_mask.png", "overlay.png", "reference_mask.png",
                         "relation_field.png"):
                print(f"[6s] wrote {out_dir / name}")
        else:
            print(f"[6s] no target mask ({payload['status']}: "
                  f"{payload.get('reference_abstention_reason')})")
        print(f"[6s] result json: {out_dir / 'result.json'}")
        print(f"[6s] runtime: {payload.get('runtime')}")
    return int(result.get("exit_code", EXIT_ANSWERED))


if __name__ == "__main__":
    raise SystemExit(main())
