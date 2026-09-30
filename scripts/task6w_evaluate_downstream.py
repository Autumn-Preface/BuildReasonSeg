"""Task 6W Parts I-J — downstream causal evaluation and the hardened-parser integration.

`--stage causal` (sections 18-19): MiniVal240 and PairedVal20 with **canonical program ids** through the
two reference systems and the frozen field v0.2 / SAM2 / B3 chain:

* **W-S0** = U-C1 proposals + deterministic Task 6Q selector;
* **W-SQ** = U-C1 proposals + learned quality filter (threshold 0.50) + deterministic selector.

`--stage parser` (section 20): Task 6T hardened ProgramHead → W-SQ → field v0.2 → SAM2 → B3 on the exact
MiniVal240 queries (must stay 240/240). No nearest/L3 execution.

Writes `task6w_downstream_minival240.json`, `task6w_downstream_pairedval20.json` and
`task6w_hardened_parser_integration.json`.

    python scripts/task6w_evaluate_downstream.py --stage causal
    python scripts/task6w_evaluate_downstream.py --stage parser
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.task6n_relation_decoder import (  # noqa: E402
    FrozenFeatureStore,
    Task6NSample,
    load_frozen_sam2_encoder,
    read_pack,
)
from buildreasonseg_mvp.task6q_reference_resolver import is_eligible, select_reference  # noqa: E402
from buildreasonseg_mvp.task6s_directional_pipeline import (  # noqa: E402
    PROGRAM_DECOMPOSITION,
    default_program_head_checkpoint,
    parse_instruction,
)
from buildreasonseg_mvp.task6w_proposal_quality import ProposalQualityEstimator  # noqa: E402
from buildreasonseg_mvp.task6w_quality_reference_resolver import QualityReferenceResolver  # noqa: E402
from scripts.task6u_common import (  # noqa: E402
    CONFIGS,
    EVAL,
    PROPOSAL_CHECKPOINT,
    PROPOSAL_CHECKPOINT_SHA256,
    centroid_error,
    gt_masks,
    iou,
    proposals_for_tile,
    record_image_path,
    sha256_file,
)
from scripts.task6u_evaluate_downstream import load_target_models, predict_target  # noqa: E402
from task6n_evaluate import target_flags  # noqa: E402
from task6n_train import FEATURE_ROOT, MaskStore  # noqa: E402

OUT_MINI = EVAL / "task6w_downstream_minival240.json"
OUT_PAIRED = EVAL / "task6w_downstream_pairedval20.json"
OUT_PARSER = EVAL / "task6w_hardened_parser_integration.json"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6n" / "packs"
QUALITY_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6w" / "proposal_quality_v01.pt"
TRAINING = EVAL / "task6w_quality_training.json"
THRESHOLD = 0.50
RELATION_ORDER = ("left_of", "right_of", "above", "below")
SYSTEMS = ("W-S0", "W-SQ")
REFERENCE_BUCKETS = ("NO_PROPOSALS", "NO_ELIGIBLE_PROPOSALS", "QUALITY_FILTER_ALL_REJECTED",
                     "REFERENCE_NOT_COVERED_IOU50", "REFERENCE_SELECTION_WRONG",
                     "SELECTED_MASK_GEOMETRY_POOR")


def build_resolver(torch_device: str):
    payload = torch.load(QUALITY_CHECKPOINT, map_location=torch_device, weights_only=False)
    model = ProposalQualityEstimator().to(torch_device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    encoder, _report = load_frozen_sam2_encoder(device=torch_device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=torch_device)
    return QualityReferenceResolver(model, store, device=torch_device, threshold=THRESHOLD), payload


def proposal_cache(yolo, records: list, device: str) -> dict[str, list]:
    cache: dict[str, list] = {}
    for record in records:
        tile_id = str(record["tile_id"] if isinstance(record, dict) else record.tile_id)
        if tile_id in cache:
            continue
        image_path = record_image_path(record) if isinstance(record, dict) else Path(record.image_path)
        proposals, _run = proposals_for_tile(yolo, CONFIGS["U-C1"], tile_id, image_path, device,
                                             use_cache=True)
        cache[tile_id] = proposals
    return cache


def select_reference_mask(mode: str, proposals: list, family: str, tile_id: str, image_path,
                          resolver) -> dict:
    if mode == "W-S0":
        selection = select_reference(proposals, family)
        if selection.abstained:
            return {"abstained": True, "reason": selection.reason, "mask": None,
                    "filtered": False}
        return {"abstained": False, "reason": None, "mask": selection.proposal.mask,
                "filtered": False}
    outcome = resolver.select(proposals, family, tile_id, image_path)
    return {"abstained": outcome["abstained"], "reason": outcome["reason"],
            "mask": None if outcome["abstained"] else outcome["mask"],
            "filtered": outcome["reason"] == "no_quality_eligible_proposals"}


def run_causal(args) -> int:
    started = time.time()
    checkpoint_sha = sha256_file(PROPOSAL_CHECKPOINT) if PROPOSAL_CHECKPOINT.is_file() else None
    if checkpoint_sha != PROPOSAL_CHECKPOINT_SHA256 or not QUALITY_CHECKPOINT.is_file():
        write_json(OUT_MINI, {"_doc": "Task 6W section 18.", "task": "6W",
                              "verdict": "FROZEN_ASSET_UNAVAILABLE"})
        return 2
    resolver, _payload = build_resolver(args.torch_device)
    target_model, store, target_info = load_target_models(args.device)
    masks = MaskStore()
    samples = read_pack(PACK_ROOT / "mini_val_240.json")
    pairs = json.loads((PACK_ROOT / "paired_val_20.json").read_text(encoding="utf-8"))["pairs"]
    from ultralytics import YOLO

    yolo = YOLO(str(PROPOSAL_CHECKPOINT))
    cache = proposal_cache(yolo, samples, args.device)

    def evaluate(mode: str) -> dict:
        rows = []
        for sample in samples:
            family, relation = PROGRAM_DECOMPOSITION[sample.program_id]
            proposals = cache[sample.tile_id]
            truth_reference = gt_masks(sample.tile_id)[int(sample.reference_source_feature_id)]
            eligible = [proposal for proposal in proposals if is_eligible(proposal, family)]
            best_eligible = max((iou(proposal.mask, truth_reference) for proposal in eligible),
                                default=0.0)
            selection = select_reference_mask(mode, proposals, family, sample.tile_id,
                                              Path(sample.image_path), resolver)
            truth_target = np.asarray(masks.mask(sample.tile_id, sample.target_source_feature_id),
                                      dtype=bool)
            flags = target_flags(masks, sample)
            row = {"sample_id": sample.sample_id, "reference_family": family, "relation": relation,
                   "proposal_count": len(proposals), "eligible_count": len(eligible),
                   "best_eligible_iou": best_eligible, "reference_abstained": selection["abstained"],
                   "reference_abstention_reason": selection["reason"],
                   "quality_filtered_all": selection["filtered"],
                   "target_touches_border": flags["target_touches_border"],
                   "target_tiny": flags["target_tiny"]}
            if selection["abstained"]:
                row.update({"target_iou": 0.0, "target_dice": 0.0, "strict_iou": 0.0,
                            "reference_iou": None,
                            "bucket": ("NO_PROPOSALS" if not proposals else
                                       "NO_ELIGIBLE_PROPOSALS" if not eligible else
                                       "QUALITY_FILTER_ALL_REJECTED" if selection["filtered"] else
                                       "REFERENCE_NOT_COVERED_IOU50")})
                rows.append(row)
                continue
            reference = selection["mask"]
            reference_iou = iou(reference, truth_reference)
            reference_centroid = centroid_error(reference, truth_reference)
            target = predict_target(target_model, store, sample, reference, relation, args.device)
            target_iou = iou(target, truth_target)
            intersection = float(np.logical_and(target, truth_target).sum())
            row.update({
                "reference_iou": reference_iou, "reference_centroid_error": reference_centroid,
                "target_iou": target_iou,
                "target_dice": (2.0 * intersection + 1e-6)
                / (float(target.sum()) + float(truth_target.sum()) + 1e-6),
                "strict_iou": target_iou,
                "bucket": ("NO_PROPOSALS" if not proposals else
                           "NO_ELIGIBLE_PROPOSALS" if not eligible else
                           "REFERENCE_NOT_COVERED_IOU50" if best_eligible < 0.50 else
                           "REFERENCE_SELECTION_WRONG" if reference_iou < 0.50 else
                           "SELECTED_MASK_GEOMETRY_POOR" if reference_centroid > 0.05 else
                           "TARGET_FAIL_WITH_REFERENCE_OK" if target_iou < 0.50 else "TARGET_OK"),
            })
            rows.append(row)

        answered = [row for row in rows if not row["reference_abstained"]]
        buckets = Counter(row["bucket"] for row in rows)
        return {
            "rows": rows,
            "summary": {
                "records": len(rows),
                "strict_all_miou": float(np.mean([row["target_iou"] for row in rows])),
                "strict_all_dice": float(np.mean([row["target_dice"] for row in rows])),
                "answered_records": len(answered),
                "answered_only_miou": float(np.mean([row["target_iou"] for row in answered]))
                if answered else None,
                "answered_only_dice": float(np.mean([row["target_dice"] for row in answered]))
                if answered else None,
                "precision_at_0_5": float(np.mean([row["target_iou"] >= 0.50 for row in rows])),
                "abstentions": len(rows) - len(answered),
                "reference_fail_count": sum(buckets.get(name, 0) for name in REFERENCE_BUCKETS),
                "target_fail_with_reference_ok_count": buckets.get("TARGET_FAIL_WITH_REFERENCE_OK", 0),
                "buckets": {name: buckets.get(name, 0) for name in
                            (*REFERENCE_BUCKETS, "TARGET_FAIL_WITH_REFERENCE_OK", "TARGET_OK")},
                "largest": {"records": sum(1 for row in rows if row["reference_family"] == "largest"),
                            "miou": float(np.mean([row["target_iou"] for row in rows
                                                   if row["reference_family"] == "largest"]))
                            if any(row["reference_family"] == "largest" for row in rows) else None},
                "smallest": {"records": sum(1 for row in rows
                                            if row["reference_family"] == "smallest"),
                             "miou": float(np.mean([row["target_iou"] for row in rows
                                                    if row["reference_family"] == "smallest"]))
                             if any(row["reference_family"] == "smallest" for row in rows) else None},
                "per_direction": {relation: float(np.mean([row["target_iou"] for row in rows
                                                           if row["relation"] == relation]))
                                  for relation in RELATION_ORDER
                                  if any(row["relation"] == relation for row in rows)},
                "border_target": {"records": sum(1 for row in rows
                                                 if row["target_touches_border"]),
                                  "miou": float(np.mean([row["target_iou"] for row in rows
                                                         if row["target_touches_border"]]))
                                  if any(row["target_touches_border"] for row in rows) else None},
                "tiny_target": {"records": sum(1 for row in rows if row["target_tiny"]),
                                "miou": float(np.mean([row["target_iou"] for row in rows
                                                       if row["target_tiny"]]))
                                if any(row["target_tiny"] for row in rows) else None},
            },
        }

    results = {mode: evaluate(mode) for mode in SYSTEMS}
    payload = {
        "_doc": (
            "Task 6W section 18. Downstream causal evaluation on MiniVal240 with canonical program ids "
            "(no parser) for W-S0 (U-C1 + deterministic) and W-SQ (U-C1 + learned quality filter "
            "quality>=0.50 + deterministic), through frozen field v0.2, SAM2 and B3."
        ),
        "task": "6W", "stage": "I-downstream-minival240",
        "pack": {"path": str(PACK_ROOT / "mini_val_240.json"), "records": len(samples),
                 "sha256": sha256_file(PACK_ROOT / "mini_val_240.json")},
        "config": CONFIGS["U-C1"]["id"],
        "quality_threshold": THRESHOLD,
        "quality_checkpoint": {"path": str(QUALITY_CHECKPOINT),
                               "sha256": sha256_file(QUALITY_CHECKPOINT)},
        "target_decoder": target_info,
        "systems": {mode: results[mode]["summary"] for mode in SYSTEMS},
        "parser_used": False,
        "parser_fail_count": 0,
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_MINI, payload)

    pair_rows = []
    for pair in pairs:
        members = [Task6NSample(**{key: value for key, value in pair[side].items()
                                   if key in Task6NSample.__dataclass_fields__})
                   for side in ("a", "b")]
        entry = {"tile_id": members[0].tile_id, "systems": {}}
        for mode in SYSTEMS:
            own, cross, abstained = [], [], 0
            for index, member in enumerate(members):
                family, relation = PROGRAM_DECOMPOSITION[member.program_id]
                proposals = cache.setdefault(member.tile_id, None)
                if proposals is None:
                    proposals, _run = proposals_for_tile(yolo, CONFIGS["U-C1"], member.tile_id,
                                                         Path(member.image_path), args.device,
                                                         use_cache=True)
                    cache[member.tile_id] = proposals
                selection = select_reference_mask(mode, proposals, family, member.tile_id,
                                                  Path(member.image_path), resolver)
                if selection["abstained"]:
                    abstained += 1
                    own.append(0.0)
                    cross.append(0.0)
                    continue
                target = predict_target(target_model, store, member, selection["mask"], relation,
                                        args.device)
                own_mask = np.asarray(masks.mask(member.tile_id, member.target_source_feature_id),
                                      dtype=bool)
                own.append(float(np.logical_and(target, own_mask).sum())
                           / float(np.logical_or(target, own_mask).sum() + 1e-6))
                other = members[1 - index]
                other_mask = np.asarray(masks.mask(other.tile_id, other.target_source_feature_id),
                                        dtype=bool)
                cross.append(float(np.logical_and(target, other_mask).sum())
                             / float(np.logical_or(target, other_mask).sum() + 1e-6))
            entry["systems"][mode] = {"own": own, "cross": cross, "abstained": abstained,
                                      "passes": bool(own[0] > cross[0] and own[1] > cross[1])}
        pair_rows.append(entry)

    paired_payload = {
        "_doc": ("Task 6W section 19. PairedVal20 for W-S0 and W-SQ with canonical program ids."),
        "task": "6W", "stage": "I-downstream-pairedval20",
        "pack": {"path": str(PACK_ROOT / "paired_val_20.json"), "pairs": len(pair_rows),
                 "sha256": sha256_file(PACK_ROOT / "paired_val_20.json")},
        "systems": {},
        "test_split_used": False,
    }
    for mode in SYSTEMS:
        rows = [entry["systems"][mode] for entry in pair_rows]
        own = [value for row in rows for value in row["own"]]
        cross = [value for row in rows for value in row["cross"]]
        paired_payload["systems"][mode] = {
            "passed": sum(1 for row in rows if row["passes"]), "pairs": len(rows),
            "mean_own_iou": float(np.mean(own)), "mean_cross_iou": float(np.mean(cross)),
            "own_cross_margin": float(np.mean(own) - np.mean(cross)),
            "reference_abstention_pairs": sum(1 for row in rows if row["abstained"]),
        }
    write_json(OUT_PAIRED, paired_payload)
    for mode in SYSTEMS:
        summary = payload["systems"][mode]
        print(f"[6w.down] {mode}: strict {summary['strict_all_miou']:.4f} answered "
              f"{summary['answered_only_miou']:.4f} abstain {summary['abstentions']} ref_fail "
              f"{summary['reference_fail_count']} target_fail "
              f"{summary['target_fail_with_reference_ok_count']} | paired "
              f"{paired_payload['systems'][mode]['passed']}/20 margin "
              f"{paired_payload['systems'][mode]['own_cross_margin']:+.4f}", flush=True)
    return 0


def run_parser(args) -> int:
    started = time.time()
    from buildreasonseg_mvp.program_parser import build_program_parser, load_parser_checkpoint
    from buildreasonseg_mvp.runtime import load_config
    from buildreasonseg_mvp.structured_grounding import EXPECTED_QUERY_TYPES

    resolver, _payload = build_resolver(args.torch_device)
    target_model, store, target_info = load_target_models(args.device)
    masks = MaskStore()
    samples = read_pack(PACK_ROOT / "mini_val_240.json")
    instructions = {}
    with (REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2" / "val.jsonl").open(
            encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            instructions[record["sample_id"]] = record
    checkpoint, _report = default_program_head_checkpoint()
    training = json.loads((EVAL / "task6t_training_summary.json").read_text(encoding="utf-8"))
    if checkpoint is None or sha256_file(checkpoint) != training["checkpoint"]["sha256"]:
        write_json(OUT_PARSER, {"_doc": "Task 6W section 20.", "task": "6W",
                                "verdict": "FROZEN_ASSET_UNAVAILABLE"})
        return 2
    cfg = load_config(REPO_ROOT / "configs" / "mvp" / "task6j_program_parser.yaml")
    runtime = build_program_parser(cfg, device=args.device, verbose=False)
    load_parser_checkpoint(checkpoint, runtime)
    from ultralytics import YOLO

    yolo = YOLO(str(PROPOSAL_CHECKPOINT))
    cache = proposal_cache(yolo, samples, args.device)
    rows = []
    for sample in samples:
        query = str(instructions[sample.sample_id]["instruction_en"])
        parsed = parse_instruction(runtime, query, tuple(EXPECTED_QUERY_TYPES))
        decomposition = PROGRAM_DECOMPOSITION.get(parsed["program"])
        row = {"sample_id": sample.sample_id, "expected_program": sample.program_id,
               "parsed_program": parsed["program"],
               "parser_correct": parsed["program"] == sample.program_id,
               "reference_abstained": None}
        if decomposition is None:
            row.update({"target_iou": 0.0, "strict_iou": 0.0, "bucket": "PARSER_WRONG",
                        "reference_iou": None, "reference_centroid_error": None})
            rows.append(row)
            continue
        family, relation = decomposition
        proposals = cache[sample.tile_id]
        truth_reference = gt_masks(sample.tile_id)[int(sample.reference_source_feature_id)]
        eligible = [proposal for proposal in proposals if is_eligible(proposal, family)]
        best_eligible = max((iou(proposal.mask, truth_reference) for proposal in eligible),
                            default=0.0)
        selection = select_reference_mask("W-SQ", proposals, family, sample.tile_id,
                                          Path(sample.image_path), resolver)
        row["reference_abstained"] = selection["abstained"]
        if selection["abstained"]:
            row.update({"target_iou": 0.0, "strict_iou": 0.0, "reference_iou": None,
                        "reference_centroid_error": None,
                        "bucket": ("NO_PROPOSALS" if not proposals else
                                   "NO_ELIGIBLE_PROPOSALS" if not eligible else
                                   "QUALITY_FILTER_ALL_REJECTED" if selection["filtered"] else
                                   "REFERENCE_NOT_COVERED_IOU50")})
            rows.append(row)
            continue
        reference_iou = iou(selection["mask"], truth_reference)
        reference_centroid = centroid_error(selection["mask"], truth_reference)
        truth_target = np.asarray(masks.mask(sample.tile_id, sample.target_source_feature_id),
                                  dtype=bool)
        target = predict_target(target_model, store, sample, selection["mask"], relation, args.device)
        target_iou = iou(target, truth_target)
        row.update({"reference_iou": reference_iou, "reference_centroid_error": reference_centroid,
                    "target_iou": target_iou,
                    "strict_iou": target_iou if row["parser_correct"] else 0.0,
                    "bucket": ("NO_PROPOSALS" if not proposals else
                               "NO_ELIGIBLE_PROPOSALS" if not eligible else
                               "REFERENCE_NOT_COVERED_IOU50" if best_eligible < 0.50 else
                               "REFERENCE_SELECTION_WRONG" if reference_iou < 0.50 else
                               "SELECTED_MASK_GEOMETRY_POOR" if reference_centroid > 0.05 else
                               "TARGET_FAIL_WITH_REFERENCE_OK" if target_iou < 0.50 else "TARGET_OK")})
        rows.append(row)

    paired_path = OUT_PAIRED
    parser_correct = sum(1 for row in rows if row["parser_correct"])
    answered = [row for row in rows if row["reference_abstained"] is False]
    buckets = Counter(row["bucket"] for row in rows)
    payload = {
        "_doc": (
            "Task 6W section 20. Hardened natural-language integration: Task 6T ProgramHead -> W-SQ "
            "reference path -> field v0.2 -> SAM2 -> B3 on the exact MiniVal240 queries. No nearest/L3."
        ),
        "task": "6W", "stage": "J-hardened-parser-integration",
        "parser_checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "target_decoder": target_info,
        "pack": {"path": str(PACK_ROOT / "mini_val_240.json"), "records": len(rows)},
        "parser": {"exact_correct": parser_correct, "records": len(rows),
                   "exact_accuracy": parser_correct / len(rows)},
        "strict_all_miou": float(np.mean([row["strict_iou"] for row in rows])),
        "answered_only_miou": float(np.mean([row["target_iou"] for row in answered]))
        if answered else None,
        "abstentions": len(rows) - len(answered),
        "reference_fail_count": sum(buckets.get(name, 0) for name in REFERENCE_BUCKETS),
        "buckets": dict(buckets),
        "paired": {key: json.loads(paired_path.read_text(encoding="utf-8"))["systems"]["W-SQ"][key]
                   for key in ("passed", "pairs", "own_cross_margin", "mean_own_iou",
                               "mean_cross_iou")}
        if paired_path.is_file() else None,
        "nearest_or_l3_execution_evaluated": False,
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_PARSER, payload)
    print(f"[6w.parser] parser {parser_correct}/{len(rows)} | strict "
          f"{payload['strict_all_miou']:.4f} answered {payload['answered_only_miou']:.4f} abstain "
          f"{payload['abstentions']} ref_fail {payload['reference_fail_count']}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("causal", "parser"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-device", default="cuda")
    args = parser.parse_args(argv)
    return run_causal(args) if args.stage == "causal" else run_parser(args)


if __name__ == "__main__":
    raise SystemExit(main())
