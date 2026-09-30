"""Task 6V Parts F-G — downstream causal evaluation and the natural-language integration regression.

`--stage causal` (sections 10-11): MiniVal240 and PairedVal20 with **canonical program ids** through the
frozen family-conditioned resolver, field v0.2, frozen SAM2 feature and frozen B3 (no parser), so the
comparison isolates the reference stage.

`--stage parser` (section 12): the exact Task 6T hardened ProgramHead over MiniVal240 natural-language
queries → family-conditioned resolver → field v0.2 → SAM2 → B3 (must stay 240/240 exact).

Writes `task6v_downstream_minival240.json`, `task6v_downstream_pairedval20.json` and
`task6v_hardened_parser_integration.json`. No model is trained; no nearest/L3 execution is evaluated.

    python scripts/task6v_evaluate_downstream.py --stage causal
    python scripts/task6v_evaluate_downstream.py --stage parser
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
from buildreasonseg_mvp.task6n_relation_decoder import Task6NSample, read_pack  # noqa: E402
from buildreasonseg_mvp.task6q_reference_resolver import is_eligible  # noqa: E402
from buildreasonseg_mvp.task6s_directional_pipeline import (  # noqa: E402
    PROGRAM_DECOMPOSITION,
    default_program_head_checkpoint,
    parse_instruction,
)
from buildreasonseg_mvp.task6u_reference_ranker import ProposalSetRanker  # noqa: E402
from buildreasonseg_mvp.task6v_family_reference_resolver import (  # noqa: E402
    FAMILIES,
    FamilyConditionedResolver,
    load_frozen_policy,
)
from scripts.task6u_common import (  # noqa: E402
    BUCKET_ORDER,
    CONFIGS,
    EVAL,
    PROPOSAL_CHECKPOINT,
    PROPOSAL_CHECKPOINT_SHA256,
    gt_masks,
    group_records_by_tile,
    iou,
    proposals_for_tile,
    record_image_path,
    sha256_file,
)
from scripts.task6u_evaluate_downstream import load_target_models, predict_target  # noqa: E402
from task6n_evaluate import target_flags  # noqa: E402
from task6n_train import MaskStore  # noqa: E402

OUT_MINI = EVAL / "task6v_downstream_minival240.json"
OUT_PAIRED = EVAL / "task6v_downstream_pairedval20.json"
OUT_PARSER = EVAL / "task6v_hardened_parser_integration.json"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6n" / "packs"
FROZEN_POLICY = EVAL / "task6v_frozen_family_policy.json"
RANKER_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6u" / "reference_ranker_v01.pt"
RELATION_ORDER = ("left_of", "right_of", "above", "below")


def build_resolver() -> FamilyConditionedResolver:
    payload = torch.load(RANKER_CHECKPOINT, map_location="cpu", weights_only=False)
    ranker = ProposalSetRanker().to("cpu")
    ranker.load_state_dict(payload["state_dict"])
    ranker.eval()
    return FamilyConditionedResolver(load_frozen_policy(FROZEN_POLICY), ranker=ranker)


def proposal_lookup(yolo, records: list, device: str):
    """Cached proposal sets per (config, tile) for every config the frozen policy can reference.

    Accepts either frozen-pack dict records or `Task6NSample` objects.
    """

    def tile_of(record) -> str:
        return str(record["tile_id"] if isinstance(record, dict) else record.tile_id)

    def image_of(record) -> Path:
        path = Path(str(record["image_path"] if isinstance(record, dict) else record.image_path))
        return path if path.is_absolute() else REPO_ROOT / path

    grouped: dict[str, list] = {}
    for record in records:
        grouped.setdefault(tile_of(record), []).append(record)
    config_ids = ("U-C0", "U-C1")
    caches: dict[str, dict] = {config_id: {} for config_id in config_ids}
    for config_id in config_ids:
        for tile_id, tile_records in grouped.items():
            proposals, _run = proposals_for_tile(yolo, CONFIGS[config_id], tile_id,
                                                 image_of(tile_records[0]), device,
                                                 use_cache=True)
            caches[config_id][tile_id] = proposals

    def lookup(config_id: str, tile_id: str, image_path: Path):
        if tile_id not in caches[config_id]:
            proposals, _run = proposals_for_tile(yolo, CONFIGS[config_id], tile_id, image_path,
                                                 device, use_cache=True)
            caches[config_id][tile_id] = proposals
        return caches[config_id][tile_id]

    return lookup


def run_causal(args) -> int:
    started = time.time()
    checkpoint_sha = sha256_file(PROPOSAL_CHECKPOINT) if PROPOSAL_CHECKPOINT.is_file() else None
    if checkpoint_sha != PROPOSAL_CHECKPOINT_SHA256 or not FROZEN_POLICY.is_file():
        write_json(OUT_MINI, {"_doc": "Task 6V section 10.", "task": "6V",
                              "verdict": "FROZEN_RESOLVER_ASSET_UNAVAILABLE"})
        return 2
    resolver = build_resolver()
    target_model, store, target_info = load_target_models(args.device)
    masks = MaskStore()
    samples = read_pack(PACK_ROOT / "mini_val_240.json")
    pairs = json.loads((PACK_ROOT / "paired_val_20.json").read_text(encoding="utf-8"))["pairs"]
    from ultralytics import YOLO

    yolo = YOLO(str(PROPOSAL_CHECKPOINT))
    lookup = proposal_lookup(yolo, samples, args.device)
    policy = resolver.policy
    print(f"[6v.down] frozen family policy {policy}", flush=True)

    rows = []
    for sample in samples:
        family, relation = PROGRAM_DECOMPOSITION[sample.program_id]
        option = resolver.option_for(family)
        proposals = lookup(option["config"], sample.tile_id, Path(sample.image_path))
        truth_reference = gt_masks(sample.tile_id)[int(sample.reference_source_feature_id)]
        eligible = [proposal for proposal in proposals if is_eligible(proposal, family)]
        best_eligible = max((iou(proposal.mask, truth_reference) for proposal in eligible),
                            default=0.0)
        selection = resolver.select(proposals, family)
        truth_target = np.asarray(masks.mask(sample.tile_id, sample.target_source_feature_id),
                                  dtype=bool)
        flags = target_flags(masks, sample)
        row = {"sample_id": sample.sample_id, "tile_id": sample.tile_id,
               "program_id": sample.program_id, "reference_family": family, "relation": relation,
               "option": selection["option"], "config": selection["config"],
               "selector": selection["selector"], "proposal_count": len(proposals),
               "eligible_count": len(eligible), "best_eligible_iou": best_eligible,
               "reference_abstained": selection["abstained"],
               "reference_abstention_reason": selection["reason"],
               "target_touches_border": flags["target_touches_border"],
               "target_tiny": flags["target_tiny"]}
        if selection["abstained"]:
            row.update({"target_iou": 0.0, "target_dice": 0.0, "strict_iou": 0.0,
                        "reference_iou": None,
                        "bucket": ("NO_PROPOSALS" if not proposals else
                                   "NO_ELIGIBLE_PROPOSALS" if not eligible else
                                   "REFERENCE_NOT_COVERED_IOU50")})
            rows.append(row)
            continue
        reference_iou = iou(selection["mask"], truth_reference)
        from scripts.task6u_common import centroid_error

        reference_centroid = centroid_error(selection["mask"], truth_reference)
        target = predict_target(target_model, store, sample, selection["mask"], relation, args.device)
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
    reference_fail = sum(buckets[name] for name in (
        "NO_PROPOSALS", "NO_ELIGIBLE_PROPOSALS", "REFERENCE_NOT_COVERED_IOU50",
        "REFERENCE_SELECTION_WRONG", "SELECTED_MASK_GEOMETRY_POOR"))
    target_ious = [row["target_iou"] for row in rows]
    payload = {
        "_doc": (
            "Task 6V section 10. Downstream causal evaluation on MiniVal240 with canonical program ids "
            "(no parser) through the frozen family-conditioned resolver, field v0.2, frozen SAM2 feature "
            "and frozen B3. GT is for evaluation only."
        ),
        "task": "6V", "stage": "F-downstream-minival240",
        "policy": policy,
        "option_usage": {family: resolver.option_for(family)["id"] for family in FAMILIES},
        "pack": {"path": str(PACK_ROOT / "mini_val_240.json"), "records": len(samples),
                 "sha256": sha256_file(PACK_ROOT / "mini_val_240.json")},
        "target_decoder": target_info,
        "strict_all_miou": float(np.mean(target_ious)),
        "strict_all_dice": float(np.mean([row["target_dice"] for row in rows])),
        "answered_records": len(answered),
        "answered_only_miou": float(np.mean([row["target_iou"] for row in answered]))
        if answered else None,
        "answered_only_dice": float(np.mean([row["target_dice"] for row in answered]))
        if answered else None,
        "precision_at_0_5": float(np.mean([value >= 0.50 for value in target_ious])),
        "abstentions": len(rows) - len(answered),
        "reference_fail_count": reference_fail,
        "target_fail_with_reference_ok_count": buckets.get("TARGET_FAIL_WITH_REFERENCE_OK", 0),
        "buckets": {name: buckets.get(name, 0) for name in
                    (*BUCKET_ORDER, "TARGET_FAIL_WITH_REFERENCE_OK", "TARGET_OK")},
        "largest": {family: {"records": sum(1 for row in rows
                                            if row["reference_family"] == family),
                             "miou": float(np.mean([row["target_iou"] for row in rows
                                                    if row["reference_family"] == family]))
                             if any(row["reference_family"] == family for row in rows) else None}
                    for family in FAMILIES},
        "per_direction": {relation: float(np.mean([row["target_iou"] for row in rows
                                                   if row["relation"] == relation]))
                          for relation in RELATION_ORDER
                          if any(row["relation"] == relation for row in rows)},
        "border_target": {"records": sum(1 for row in rows if row["target_touches_border"]),
                          "miou": float(np.mean([row["target_iou"] for row in rows
                                                 if row["target_touches_border"]]))
                          if any(row["target_touches_border"] for row in rows) else None},
        "tiny_target": {"records": sum(1 for row in rows if row["target_tiny"]),
                        "miou": float(np.mean([row["target_iou"] for row in rows
                                               if row["target_tiny"]]))
                        if any(row["target_tiny"] for row in rows) else None},
        "parser_used": False,
        "parser_fail_count": 0,
        "training_performed": False,
        "test_split_used": False,
        "records": rows,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_MINI, payload)

    pair_rows = []
    for pair in pairs:
        members = [Task6NSample(**{key: value for key, value in pair[side].items()
                                   if key in Task6NSample.__dataclass_fields__})
                   for side in ("a", "b")]
        own, cross, abstained = [], [], 0
        for index, member in enumerate(members):
            family, relation = PROGRAM_DECOMPOSITION[member.program_id]
            option = resolver.option_for(family)
            proposals = lookup(option["config"], member.tile_id, Path(member.image_path))
            selection = resolver.select(proposals, family)
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
        pair_rows.append({"tile_id": members[0].tile_id, "own": own, "cross": cross,
                          "abstained": abstained,
                          "passes": bool(own[0] > cross[0] and own[1] > cross[1])})

    own_values = [value for row in pair_rows for value in row["own"]]
    cross_values = [value for row in pair_rows for value in row["cross"]]
    paired_payload = {
        "_doc": ("Task 6V section 11. PairedVal20 through the frozen family-conditioned resolver with "
                 "canonical program ids."),
        "task": "6V", "stage": "F-downstream-pairedval20",
        "policy": policy,
        "pack": {"path": str(PACK_ROOT / "paired_val_20.json"), "pairs": len(pair_rows),
                 "sha256": sha256_file(PACK_ROOT / "paired_val_20.json")},
        "passed": sum(1 for row in pair_rows if row["passes"]),
        "pairs": len(pair_rows),
        "mean_own_iou": float(np.mean(own_values)),
        "mean_cross_iou": float(np.mean(cross_values)),
        "own_cross_margin": float(np.mean(own_values) - np.mean(cross_values)),
        "reference_abstention_pairs": sum(1 for row in pair_rows if row["abstained"]),
        "training_performed": False,
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_PAIRED, paired_payload)
    print(f"[6v.down] strict {payload['strict_all_miou']:.4f} answered "
          f"{payload['answered_only_miou']:.4f} abstain {payload['abstentions']} ref_fail "
          f"{payload['reference_fail_count']} target_fail {payload['target_fail_with_reference_ok_count']}"
          f" | paired {paired_payload['passed']}/20 margin "
          f"{paired_payload['own_cross_margin']:+.4f}", flush=True)
    return 0


def run_parser(args) -> int:
    started = time.time()
    from buildreasonseg_mvp.program_parser import build_program_parser, load_parser_checkpoint
    from buildreasonseg_mvp.runtime import load_config
    from buildreasonseg_mvp.structured_grounding import EXPECTED_QUERY_TYPES

    resolver = build_resolver()
    target_model, store, target_info = load_target_models(args.device)
    masks = MaskStore()
    samples = read_pack(PACK_ROOT / "mini_val_240.json")
    instructions = {}
    with (REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2" / "val.jsonl").open(
            encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            instructions[record["sample_id"]] = record

    checkpoint, report = default_program_head_checkpoint()
    if checkpoint is None:
        write_json(OUT_PARSER, {"_doc": "Task 6V section 12.", "task": "6V",
                                "verdict": "PARSER_CHECKPOINT_UNAVAILABLE"})
        return 2
    training = json.loads((EVAL / "task6t_training_summary.json").read_text(encoding="utf-8"))
    if sha256_file(checkpoint) != training["checkpoint"]["sha256"]:
        write_json(OUT_PARSER, {"_doc": "Task 6V section 12.", "task": "6V",
                                "verdict": "PARSER_CHECKPOINT_UNAVAILABLE"})
        return 2
    cfg = load_config(REPO_ROOT / "configs" / "mvp" / "task6j_program_parser.yaml")
    runtime = build_program_parser(cfg, device=args.device, verbose=False)
    load_parser_checkpoint(checkpoint, runtime)

    from ultralytics import YOLO

    yolo = YOLO(str(PROPOSAL_CHECKPOINT))
    lookup = proposal_lookup(yolo, samples, args.device)
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
            row.update({"target_iou": 0.0, "strict_iou": 0.0, "bucket": "PARSER_WRONG"})
            rows.append(row)
            continue
        family, relation = decomposition
        option = resolver.option_for(family)
        proposals = lookup(option["config"], sample.tile_id, Path(sample.image_path))
        truth_reference = gt_masks(sample.tile_id)[int(sample.reference_source_feature_id)]
        eligible = [proposal for proposal in proposals if is_eligible(proposal, family)]
        best_eligible = max((iou(proposal.mask, truth_reference) for proposal in eligible),
                            default=0.0)
        row["best_eligible_iou"] = best_eligible
        row["proposal_count"] = len(proposals)
        row["eligible_count"] = len(eligible)
        selection = resolver.select(proposals, family)
        row["reference_abstained"] = selection["abstained"]
        if selection["abstained"]:
            row.update({"reference_iou": None, "reference_centroid_error": None,
                        "target_iou": 0.0, "strict_iou": 0.0,
                        "bucket": ("NO_PROPOSALS" if not proposals else
                                   "NO_ELIGIBLE_PROPOSALS" if not eligible else
                                   "REFERENCE_NOT_COVERED_IOU50")})
            rows.append(row)
            continue
        from scripts.task6u_common import centroid_error as _centroid_error

        reference_iou = iou(selection["mask"], truth_reference)
        reference_centroid = _centroid_error(selection["mask"], truth_reference)
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

    pair_payload_path = OUT_PAIRED
    parser_correct = sum(1 for row in rows if row["parser_correct"])
    answered = [row for row in rows if row["reference_abstained"] is False]
    payload = {
        "_doc": (
            "Task 6V section 12. Natural-language integration regression: the exact Task 6T hardened "
            "ProgramHead over MiniVal240 queries -> frozen family-conditioned resolver -> field v0.2 -> "
            "SAM2 -> B3. No nearest/L3 execution. Used as an integration check only."
        ),
        "task": "6V", "stage": "G-hardened-parser-integration",
        "policy": resolver.policy,
        "parser_checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "target_decoder": target_info,
        "pack": {"path": str(PACK_ROOT / "mini_val_240.json"), "records": len(rows)},
        "parser": {"exact_correct": parser_correct, "records": len(rows),
                   "exact_accuracy": parser_correct / len(rows)},
        "strict_all_miou": float(np.mean([row["strict_iou"] for row in rows])),
        "answered_only_miou": float(np.mean([row["target_iou"] for row in answered]))
        if answered else None,
        "answered_records": len(answered),
        "abstentions": len(rows) - len(answered),
        "reference_fail_count": sum(1 for row in rows if row["bucket"] in (
            "NO_PROPOSALS", "NO_ELIGIBLE_PROPOSALS", "REFERENCE_NOT_COVERED_IOU50",
            "REFERENCE_SELECTION_WRONG", "SELECTED_MASK_GEOMETRY_POOR")),
        "paired": {key: json.loads(pair_payload_path.read_text(encoding="utf-8"))[key]
                   for key in ("passed", "pairs", "own_cross_margin", "mean_own_iou", "mean_cross_iou")}
        if pair_payload_path.is_file() else None,
        "nearest_or_l3_execution_evaluated": False,
        "training_performed": False,
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_PARSER, payload)
    print(f"[6v.parser] parser {parser_correct}/{len(rows)} | strict "
          f"{payload['strict_all_miou']:.4f} answered {payload['answered_only_miou']:.4f} abstain "
          f"{payload['abstentions']} ref_fail {payload['reference_fail_count']}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("causal", "parser"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    return run_causal(args) if args.stage == "causal" else run_parser(args)


if __name__ == "__main__":
    raise SystemExit(main())
