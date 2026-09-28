"""Task 6M shared structured-chain evaluation (J1-v2 and J4-v2).

Chain: instruction -> canonical program (oracle for J1-v2, Qwen2B ProgramHead for J4-v2)
       image -> YOLO26m-seg proposals (predicted geometry only)
       program + predicted geometry -> frozen deterministic relation executor -> selected proposal

Ground truth is used ONLY for scoring. Nothing here repairs, filters or re-ranks a proposal with GT.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "spatial_reasoning"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.structured_grounding import (  # noqa: E402
    EXPECTED_QUERY_TYPES,
    Candidate,
    CandidateSet,
    canonical_program_template,
    execute_program_by_id,
)

from buildreasonseg_mvp.task6m_eval import (  # noqa: E402
    EVAL,
    EXPORT_ROOT,
    canonical_instances,
    dice,
    iou,
    normalize_mask,
    percentile_summary,
    tile_ids,
)

RELATION_CONFIG = REPO_ROOT / "configs" / "spatial_relations_v1.yaml"


def load_frozen_config(path: Path = EVAL / "task6m_inference_config_frozen.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relation_config():
    sys.path.insert(0, str(REPO_ROOT / "spatial_reasoning"))
    import thresholds as T

    return T.load_config(str(RELATION_CONFIG))


# ------------------------------------------------------------------ predictions


def predict_tiles(model, ids, conf: float, max_det: int, imgsz: int = 640, device: str = "0",
                  verbose: bool = True) -> dict:
    """Predicted proposals per tile (masks at the canonical resolution + geometry)."""

    predictions: dict[str, dict] = {}
    for position, tile_id in enumerate(ids, start=1):
        image_path = EXPORT_ROOT / "images" / "val" / f"{tile_id}.tif"
        if not image_path.is_file():
            image_path = EXPORT_ROOT / "images" / "test" / f"{tile_id}.tif"
        result = model.predict(
            source=str(image_path), imgsz=imgsz, conf=conf, max_det=max_det, verbose=False,
            device=device, retina_masks=True,
        )[0]
        masks: list[np.ndarray] = []
        confidences: list[float] = []
        if result.masks is not None and result.boxes is not None:
            array = result.masks.data.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for index in range(array.shape[0]):
                masks.append(normalize_mask(array[index]))
                confidences.append(float(confs[index]))
        predictions[tile_id] = {"masks": masks, "confidences": confidences}
        if verbose and position % 500 == 0:
            print(f"[6m.structured] predicted {position}/{len(ids)} tiles", flush=True)
    return predictions


def candidate_set_from_predictions(tile_id: str, prediction: dict, source: str = "yolo26m-seg"):
    """Build a Task 6J `CandidateSet` from PREDICTED proposals only."""

    import cv2

    label_map = np.zeros((512, 512), dtype=np.int32)
    candidates: list[Candidate] = []
    for index, mask in enumerate(prediction["masks"], start=1):
        if not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        area = int(mask.sum())
        label_map[mask] = index
        candidates.append(
            Candidate(
                candidate_id=index,
                mask=mask,
                bbox_xyxy_px=bbox,
                centroid_px=(float(xs.mean()), float(ys.mean())),
                area_px=area,
                touches_image_border=bool(
                    bbox[0] == 0 or bbox[1] == 0 or bbox[2] >= 512 or bbox[3] >= 512
                ),
                confidence=float(prediction["confidences"][index - 1]) if prediction["confidences"] else None,
                source=source,
            )
        )
    return CandidateSet(width=512, height=512, candidates=candidates, label_map=label_map, source=source)


# ------------------------------------------------------------------ packs / records


def load_pack(name: str) -> dict:
    return json.loads((EVAL / name).read_text(encoding="utf-8"))


def v02_records_by_sample(split: str) -> dict[str, dict]:
    path = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2" / f"{split}.jsonl"
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                records[str(record["sample_id"])] = record
    return records


# ------------------------------------------------------------------ structured scoring


def score_one(prediction: dict, program: str, target_instance: int, config) -> dict:
    """Execute one program on the predicted proposals and score the selected mask vs the GT target."""

    candidate_set = candidate_set_from_predictions("", prediction)
    if not candidate_set.candidates:
        return {
            "abstained": True,
            "reason": "no_proposals",
            "iou": 0.0,
            "dice": 0.0,
            "selected_id": None,
            "proposals": 0,
        }
    result = execute_program_by_id(program, candidate_set, config)
    if result.abstained or result.selected_id is None:
        return {
            "abstained": True,
            "reason": str(result.reason or "executor_abstained"),
            "iou": 0.0,
            "dice": 0.0,
            "selected_id": None,
            "proposals": len(candidate_set.candidates),
        }
    selected = candidate_set.by_id().get(int(result.selected_id))
    if selected is None:
        return {
            "abstained": True,
            "reason": "selected_id_not_found",
            "iou": 0.0,
            "dice": 0.0,
            "selected_id": None,
            "proposals": len(candidate_set.candidates),
        }
    return {
        "abstained": False,
        "reason": None,
        "iou": iou(selected.mask, target_instance),
        "dice": dice(selected.mask, target_instance),
        "selected_id": int(result.selected_id),
        "proposals": len(candidate_set.candidates),
        "selected_mask": selected.mask,
    }


def evaluate_fixed120(predictions: dict, pack: dict, config, program_of: callable) -> dict:
    rows = []
    for record in pack["records"]:
        tile_id = str(record["tile_id"])
        prediction = predictions.get(tile_id)
        if prediction is None:
            continue
        gt = {g.tile_instance_id: g for g in canonical_instances(tile_id)}
        target = gt.get(int(record["target_instance"]))
        if target is None:
            rows.append({"sample_id": record["sample_id"], "skipped": "target_not_in_gt"})
            continue
        program = program_of(record)
        outcome = score_one(prediction, program, target.mask, config)
        rows.append(
            {
                "sample_id": record["sample_id"],
                "tile_id": tile_id,
                "level": record["level"],
                "query_type": record["query_type"],
                "oracle_query_type": record["query_type"],
                "program_used": program,
                "program_correct": bool(program == record["query_type"]),
                "target_instance": int(record["target_instance"]),
                "abstained": outcome["abstained"],
                "reason": outcome["reason"],
                "iou": outcome["iou"],
                "dice": outcome["dice"],
                "proposals": outcome["proposals"],
                "target_tiny": target.tiny,
                "target_border": target.touches_border,
                "target_area_px": target.area_px,
            }
        )
    scored = [row for row in rows if "skipped" not in row]
    abstained = [row for row in scored if row["abstained"]]
    return {
        "records": len(scored),
        "skipped": len(rows) - len(scored),
        "miou": float(np.mean([row["iou"] for row in scored])) if scored else None,
        "mdice": float(np.mean([row["dice"] for row in scored])) if scored else None,
        "miou_among_answered": float(np.mean([row["iou"] for row in scored if not row["abstained"]]))
        if any(not row["abstained"] for row in scored) else None,
        "answered": len(scored) - len(abstained),
        "abstentions": len(abstained),
        "abstention_reasons": dict(Counter(row["reason"] for row in abstained)),
        "program_accuracy": float(np.mean([row["program_correct"] for row in scored])) if scored else None,
        "iou_summary": percentile_summary([row["iou"] for row in scored]),
        "by_level": {
            str(level): {
                "records": sum(1 for row in scored if row["level"] == level),
                "miou": float(np.mean([row["iou"] for row in scored if row["level"] == level]))
                if any(row["level"] == level for row in scored) else None,
            }
            for level in (1, 2, 3)
        },
        "by_program": {
            program: {
                "records": len([row for row in scored if row["program_used"] == program]),
                "miou": float(np.mean([row["iou"] for row in scored if row["program_used"] == program]))
                if any(row["program_used"] == program for row in scored) else None,
            }
            for program in sorted({row["program_used"] for row in scored})
        },
        "tiny_target_miou": float(np.mean([row["iou"] for row in scored if row["target_tiny"]]))
        if any(row["target_tiny"] for row in scored) else None,
        "border_target_miou": float(np.mean([row["iou"] for row in scored if row["target_border"]]))
        if any(row["target_border"] for row in scored) else None,
        "rows": rows,
    }


def evaluate_paired20(predictions: dict, pack: dict, config, program_of: callable) -> dict:
    """Own-vs-cross: the program that asks for target A must match A better than it matches B."""

    rows = []
    for pair in pack["pairs"]:
        tile_id = str(pair["tile_id"])
        prediction = predictions.get(tile_id)
        if prediction is None:
            continue
        gt = {g.tile_instance_id: g for g in canonical_instances(tile_id)}
        mask_a = gt.get(int(pair["target_instance_a"]))
        mask_b = gt.get(int(pair["target_instance_b"]))
        if mask_a is None or mask_b is None:
            continue
        program_a = program_of({"sample_id": pair.get("sample_id_a"), "query_type": pair["query_type_a"]})
        program_b = program_of({"sample_id": pair.get("sample_id_b"), "query_type": pair["query_type_b"]})
        outcome_a = score_one(prediction, program_a, mask_a.mask, config)
        outcome_b = score_one(prediction, program_b, mask_b.mask, config)
        cross_a = score_one(prediction, program_a, mask_b.mask, config)
        cross_b = score_one(prediction, program_b, mask_a.mask, config)
        own_a, own_b = outcome_a["iou"], outcome_b["iou"]
        cr_a, cr_b = cross_a["iou"], cross_b["iou"]
        passed = bool(own_a > cr_a and own_b > cr_b)
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "tile_id": tile_id,
                "query_type_a": pair["query_type_a"],
                "query_type_b": pair["query_type_b"],
                "program_a": program_a,
                "program_b": program_b,
                "own_iou_a": own_a,
                "cross_iou_a": cr_a,
                "own_iou_b": own_b,
                "cross_iou_b": cr_b,
                "abstained_a": outcome_a["abstained"],
                "abstained_b": outcome_b["abstained"],
                "passed": passed,
            }
        )
    return {
        "pairs": len(rows),
        "passed": sum(1 for row in rows if row["passed"]),
        "pass_rate": (sum(1 for row in rows if row["passed"]) / len(rows)) if rows else None,
        "mean_own_iou": float(np.mean([(row["own_iou_a"] + row["own_iou_b"]) / 2 for row in rows])) if rows else None,
        "mean_cross_iou": float(np.mean([(row["cross_iou_a"] + row["cross_iou_b"]) / 2 for row in rows]))
        if rows else None,
        "rows": rows,
    }


def evaluate_full_split(predictions: dict, ids: list[str], config, split: str = "val",
                        program_of: callable = None) -> dict:
    """Structured evaluation over every v0.2 record of a split.

    With `program_of=None` the oracle program (the record's own `query_type`) is executed — J1-v2.
    With a `program_of(record)` callback the program comes from the parser — J4-v2.
    """

    records = v02_records_by_sample(split)
    by_tile: dict[str, list[dict]] = defaultdict(list)
    for record in records.values():
        by_tile[str(record["image_id"])].append(record)

    rows = []
    for tile_id in ids:
        prediction = predictions.get(tile_id)
        if prediction is None:
            continue
        gt = {g.tile_instance_id: g for g in canonical_instances(tile_id)}
        for record in by_tile.get(tile_id, []):
            target = gt.get(int(record["target_component_id"]))
            if target is None:
                continue
            program = program_of(record) if program_of else str(record["query_type"])
            outcome = score_one(prediction, program, target.mask, config)
            rows.append(
                {
                    "sample_id": record["sample_id"],
                    "tile_id": tile_id,
                    "level": int(record["level"]),
                    "query_type": str(record["query_type"]),
                    "program_used": program,
                    "program_correct": bool(program == str(record["query_type"])),
                    "abstained": outcome["abstained"],
                    "reason": outcome["reason"],
                    "iou": outcome["iou"],
                    "dice": outcome["dice"],
                    "proposals": outcome["proposals"],
                    "target_tiny": target.tiny,
                    "target_border": target.touches_border,
                    "target_area_px": target.area_px,
                    "dense_tile": len(gt) >= 10,
                }
            )
    abstained = [row for row in rows if row["abstained"]]
    return {
        "split": split,
        "program_source": "parser" if program_of else "oracle_program",
        "program_accuracy": float(np.mean([row["program_correct"] for row in rows])) if rows else None,
        "records": len(rows),
        "miou": float(np.mean([row["iou"] for row in rows])) if rows else None,
        "mdice": float(np.mean([row["dice"] for row in rows])) if rows else None,
        "miou_among_answered": float(np.mean([row["iou"] for row in rows if not row["abstained"]]))
        if any(not row["abstained"] for row in rows) else None,
        "answered": len(rows) - len(abstained),
        "abstentions": len(abstained),
        "abstention_rate": len(abstained) / len(rows) if rows else None,
        "abstention_reasons": dict(Counter(row["reason"] for row in abstained)),
        "iou_summary": percentile_summary([row["iou"] for row in rows]),
        "by_level": {
            str(level): {
                "records": sum(1 for row in rows if row["level"] == level),
                "miou": float(np.mean([row["iou"] for row in rows if row["level"] == level]))
                if any(row["level"] == level for row in rows) else None,
                "abstentions": sum(1 for row in rows if row["level"] == level and row["abstained"]),
            }
            for level in (1, 2, 3)
        },
        "by_program": {
            program: {
                "records": len([row for row in rows if row["program_used"] == program]),
                "miou": float(np.mean([row["iou"] for row in rows if row["program_used"] == program]))
                if any(row["program_used"] == program for row in rows) else None,
                "abstentions": sum(1 for row in rows if row["program_used"] == program and row["abstained"]),
            }
            for program in sorted({row["program_used"] for row in rows})
        },
        "tiny_target": {
            "records": sum(1 for row in rows if row["target_tiny"]),
            "miou": float(np.mean([row["iou"] for row in rows if row["target_tiny"]]))
            if any(row["target_tiny"] for row in rows) else None,
        },
        "border_target": {
            "records": sum(1 for row in rows if row["target_border"]),
            "miou": float(np.mean([row["iou"] for row in rows if row["target_border"]]))
            if any(row["target_border"] for row in rows) else None,
        },
        "dense_tile": {
            "records": sum(1 for row in rows if row["dense_tile"]),
            "miou": float(np.mean([row["iou"] for row in rows if row["dense_tile"]]))
            if any(row["dense_tile"] for row in rows) else None,
        },
        "rows_sample": rows[:200],
    }
