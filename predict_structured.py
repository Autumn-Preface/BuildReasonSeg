#!/usr/bin/env python
"""BuildReasonSeg structured Demo CLI (Task 6M section 15).

CMD example::

    python predict_structured.py ^
      --image path\\to\\image.tif ^
      --prompt "分割面积最大的建筑物右侧最近的建筑物" ^
      --proposal-checkpoint artifacts\\checkpoints\\task6m\\runs\\m1_yolo26m_seg\\weights\\best.pt ^
      --parser-checkpoint artifacts\\checkpoints\\task6j\\j2_best.pt ^
      --out-dir outputs\\demo

Chain: instruction -> Qwen2B ProgramHead -> canonical program
       image -> YOLO26m-seg proposals -> mask/bbox/centroid/area
       program + proposal geometry -> deterministic relation executor -> selected proposal mask

Requires NO ground truth or annotation files. An unsupported instruction/program fails explicitly
and is never mapped to an unrelated program.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
for extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "spatial_reasoning"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.structured_grounding import EXPECTED_QUERY_TYPES  # noqa: E402
from buildreasonseg_mvp.task6m_structured import (  # noqa: E402
    candidate_set_from_predictions,
    load_frozen_config,
    relation_config,
)

DEFAULT_CONF = 0.25
DEFAULT_MAX_DET = 300
DEFAULT_IMGSZ = 640

# ------------------------------------------------------------------ domain gate (Task 6M.1 section 13)
#
# A closed-Demo grammar/domain guard, NOT open-domain OOD detection: a prompt is accepted only when
# it names a building/object anchor AND a supported relation/selection anchor. The check runs before
# ProgramHead and before YOLO, so an unsupported prompt costs no inference at all.

DOMAIN_OBJECT_ANCHORS = (
    # Chinese
    "建筑", "建筑物", "建筑区域", "房屋", "楼",
    # English
    "building", "buildings", "structure",
)
DOMAIN_RELATION_ANCHORS = (
    # Chinese
    "最大", "最小", "最左", "最右", "最上", "最下",
    "最靠左", "最靠右", "最靠上", "最靠下",
    "最近", "左侧", "右侧", "上方", "下方",
    "左边", "右边", "上面", "下面",
    # Chinese: the frozen BuildSpatialReason v0.2 templates phrase topmost/bottommost as 位置最高 /
    # 位置最低, so the two equivalents of the documented 最上 / 最下 anchors are required; without
    # them every topmost/bottommost record (899 of 9,111 val records) is falsely rejected.
    "最高", "最低",
    # English
    "largest", "smallest", "leftmost", "rightmost", "topmost", "bottommost",
    "nearest", "closest", "left of", "right of", "above", "below",
    # English: the v0.2 templates also phrase these selections as highest / lowest / furthest
    # left|right / furthest to the left|right (the greatest area == largest).
    "highest", "lowest", "greatest",
    "furthest left", "furthest right", "furthest to the left", "furthest to the right",
)

UNSUPPORTED_STATUS = "unsupported_instruction"
UNSUPPORTED_REASON = "out_of_domain_prompt"
UNSUPPORTED_EXIT_CODE = 4


def check_domain(prompt: str) -> dict:
    """Deterministic domain gate: object anchor AND relation/selection anchor are both required."""

    text = (prompt or "").strip()
    if not text:
        return {
            "supported": False,
            "object_anchor": None,
            "relation_anchor": None,
            "reason": "empty_prompt",
        }
    lowered = text.lower()
    object_anchor = next(
        (anchor for anchor in DOMAIN_OBJECT_ANCHORS if anchor in text or anchor.lower() in lowered),
        None,
    )
    relation_anchor = next(
        (anchor for anchor in DOMAIN_RELATION_ANCHORS if anchor in text or anchor.lower() in lowered),
        None,
    )
    if object_anchor is None:
        reason = "missing_object_anchor"
    elif relation_anchor is None:
        reason = "missing_relation_anchor"
    else:
        reason = None
    return {
        "supported": object_anchor is not None and relation_anchor is not None,
        "object_anchor": object_anchor,
        "relation_anchor": relation_anchor,
        "reason": reason,
    }


def parse_instruction(runtime, prompt: str, verbose: bool) -> tuple[str, dict]:
    """Text-only program classification; the image never enters the parser."""

    program = runtime.predict(prompt)
    if program not in EXPECTED_QUERY_TYPES:
        raise SystemExit(f"error: parser returned an unknown program: {program}")
    return program, {"model_input": "instruction text only", "program": program}


def predict_proposals(model, image_path: Path, conf: float, max_det: int, imgsz: int, device: str) -> dict:
    result = model.predict(
        source=str(image_path), imgsz=imgsz, conf=conf, max_det=max_det, verbose=False,
        device=device, retina_masks=True,
    )[0]
    masks = []
    confidences = []
    if result.masks is not None and result.boxes is not None:
        from buildreasonseg_mvp.task6m_eval import normalize_mask

        array = result.masks.data.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        for index in range(array.shape[0]):
            masks.append(normalize_mask(array[index]))
            confidences.append(float(confs[index]))
    return {"masks": masks, "confidences": confidences}


def reason(steps: list[dict]) -> tuple[str, str]:
    """Compact, id-free reasoning trace in zh/en (same wording family as the dataset)."""

    zh, en = [], []
    for step in steps:
        operation = step.get("operation")
        if operation == "argmax_area":
            zh.append("在候选建筑中选择面积最大者")
            en.append("select the largest candidate")
        elif operation == "argmin_area":
            zh.append("在候选建筑中选择面积最小者")
            en.append("select the smallest candidate")
        elif operation == "argmin_boundary_distance":
            anchor = "参考建筑"
            zh.append(f"在与{anchor}边界距离最近的候选中选择")
            en.append("select the candidate with the smallest boundary distance to the reference")
        elif operation == "argmax_x":
            zh.append("选择最靠右的候选")
            en.append("select the rightmost candidate")
        elif operation == "argmin_x":
            zh.append("选择最靠左的候选")
            en.append("select the leftmost candidate")
        elif operation == "argmax_y":
            zh.append("选择最靠下的候选")
            en.append("select the bottommost candidate")
        elif operation == "argmin_y":
            zh.append("选择最靠上的候选")
            en.append("select the topmost candidate")
        elif operation == "filter_relation":
            relation = str(step.get("relation") or step.get("direction") or "direction")
            zh.append(f"按 {relation} 关系筛选候选")
            en.append(f"filter candidates by {relation}")
        else:
            zh.append(str(operation))
            en.append(str(operation))
    return " -> ".join(zh), " -> ".join(en)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True, help="instruction text (zh or en)")
    parser.add_argument("--proposal-checkpoint", type=Path, required=True)
    parser.add_argument("--parser-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--max-det", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--device", default="0")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    if not args.image.is_file():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 2
    if not args.proposal_checkpoint.is_file():
        print(f"error: proposal checkpoint not found: {args.proposal_checkpoint}", file=sys.stderr)
        return 2
    if not args.parser_checkpoint.is_file():
        print(f"error: parser checkpoint not found: {args.parser_checkpoint}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- domain gate FIRST: no parser, no proposal model, no inference
    gate = check_domain(args.prompt)
    if not gate["supported"]:
        payload = {
            "status": UNSUPPORTED_STATUS,
            "abstention_reason": UNSUPPORTED_REASON,
            "prompt": args.prompt,
            "domain_gate": {
                "supported": False,
                "reason": gate["reason"],
                "object_anchor": gate["object_anchor"],
                "relation_anchor": gate["relation_anchor"],
                "object_anchors": list(DOMAIN_OBJECT_ANCHORS),
                "relation_anchors": list(DOMAIN_RELATION_ANCHORS),
                "checked_before_parser": True,
                "checked_before_proposal_model": True,
            },
            "parsed_program": None,
            "proposal_count": None,
            "image": str(args.image),
            "ground_truth_used": False,
            "runtime_seconds": round(time.time() - started, 2),
        }
        (out_dir / "result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"[demo] status: {UNSUPPORTED_STATUS} ({UNSUPPORTED_REASON}: {gate['reason']})")
        print("[demo] the parser and the proposal model were NOT called")
        print(f"[demo] result json: {out_dir / 'result.json'}")
        return UNSUPPORTED_EXIT_CODE

    frozen_path = REPO_ROOT / "evaluation" / "task6m_inference_config_frozen.json"
    conf = args.conf
    max_det = args.max_det
    frozen_info = None
    if frozen_path.is_file():
        frozen = load_frozen_config(frozen_path)
        frozen_info = frozen
        conf = conf if conf is not None else float(frozen["conf"])
        max_det = max_det if max_det is not None else int(frozen["max_det"])
    conf = DEFAULT_CONF if conf is None else conf
    max_det = DEFAULT_MAX_DET if max_det is None else max_det

    # ---------------- parser
    from buildreasonseg_mvp.program_parser import build_program_parser, load_parser_checkpoint
    from buildreasonseg_mvp.runtime import load_config

    cfg = load_config(REPO_ROOT / "configs" / "mvp" / "task6j_program_parser.yaml")
    runtime = build_program_parser(cfg, device="cuda" if args.device != "cpu" else "cpu", verbose=not args.quiet)
    load_parser_checkpoint(args.parser_checkpoint, runtime)
    program, parser_trace = parse_instruction(runtime, args.prompt, verbose=not args.quiet)

    # ---------------- proposals
    from ultralytics import YOLO

    model = YOLO(str(args.proposal_checkpoint))
    prediction = predict_proposals(model, args.image, conf, max_det, args.imgsz, args.device)
    candidate_set = candidate_set_from_predictions("", prediction)
    if not candidate_set.candidates:
        payload = {
            "status": "abstained",
            "abstention_reason": "no_proposals",
            "prompt": args.prompt,
            "program": program,
            "proposals": 0,
        }
        (out_dir / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
        print("[demo] abstained: no proposals for this image")
        return 3

    # ---------------- deterministic execution (predicted geometry only)
    from buildreasonseg_mvp.structured_grounding import canonical_program_template, execute_program_by_id

    config = relation_config()
    template = canonical_program_template(program)
    result = execute_program_by_id(program, candidate_set, config)
    reasoning_zh, reasoning_en = reason(template)

    payload = {
        "status": "ok" if not result.abstained else "abstained",
        "abstention_reason": None if not result.abstained else str(result.reason or "executor_abstained"),
        "prompt": args.prompt,
        "parsed_program": program,
        "program_template": template,
        "parser_trace": parser_trace,
        "reasoning_zh": reasoning_zh,
        "reasoning_en": reasoning_en,
        "proposal_count": len(candidate_set.candidates),
        "inference_config": {
            "conf": conf,
            "max_det": max_det,
            "imgsz": args.imgsz,
            "source": "evaluation/task6m_inference_config_frozen.json" if frozen_info else "cli defaults",
        },
        "image": str(args.image),
        "proposal_checkpoint": str(args.proposal_checkpoint),
        "parser_checkpoint": str(args.parser_checkpoint),
        "ground_truth_used": False,
    }

    if not result.abstained and result.selected_id is not None:
        from PIL import Image

        selected = next(
            (c for c in candidate_set.candidates if c.candidate_id == int(result.selected_id)), None
        )
        if selected is not None:
            mask_path = out_dir / "selected_mask.png"
            Image.fromarray((selected.mask.astype(np.uint8) * 255), mode="L").save(mask_path)
            image = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
            if image.shape[0] != selected.mask.shape[0] or image.shape[1] != selected.mask.shape[1]:
                import cv2

                image = cv2.resize(image, (selected.mask.shape[1], selected.mask.shape[0]))
            overlay = image.copy()
            overlay[selected.mask] = (0.45 * overlay[selected.mask] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)
            overlay_path = out_dir / "overlay.png"
            Image.fromarray(overlay).save(overlay_path)
            payload["selected"] = {
                "proposal_index": int(selected.candidate_id),
                "area_px": int(selected.area_px),
                "bbox_xyxy_px": list(selected.bbox_xyxy_px),
                "centroid_px": [float(selected.centroid_px[0]), float(selected.centroid_px[1])],
                "confidence": float(selected.confidence) if selected.confidence is not None else None,
                "touches_image_border": bool(selected.touches_image_border),
            }
            payload["outputs"] = {"selected_mask": str(mask_path), "overlay": str(overlay_path)}

    payload["runtime_seconds"] = round(time.time() - started, 2)
    (out_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[demo] program: {program}")
    print(f"[demo] proposals: {len(candidate_set.candidates)}")
    print(f"[demo] status: {payload['status']}"
          + (f" ({payload['abstention_reason']})" if payload["abstention_reason"] else ""))
    if payload.get("outputs"):
        print(f"[demo] selected mask: {payload['outputs']['selected_mask']}")
        print(f"[demo] overlay: {payload['outputs']['overlay']}")
    print(f"[demo] result json: {out_dir / 'result.json'}")
    return 0 if payload["status"] == "ok" else 3


if __name__ == "__main__":
    raise SystemExit(main())
