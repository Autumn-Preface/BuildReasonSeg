"""Task 6U Parts I and J — downstream causal evaluation and the final natural-language integration check.

`--stage causal` (sections 19-20): MiniVal240 and PairedVal20 through the frozen field/B3 chain for the
three reference systems, using **canonical program ids** from the frozen records (no parser), so the
comparison isolates the proposal/reference change only:

```text
canonical program id -> family/relation -> U-S0/U-S1/U-S2 reference -> field v0.2 -> frozen SAM2
feature -> frozen B3 -> target mask
```

`--stage parser` (section 21): the Task 6T hardened ProgramHead + the U-S2 reference path + frozen
field/B3 on MiniVal240 (one integration check only; never used to train or choose U-S2).

GT is used for evaluation only. Writes `task6u_downstream_minival240.json`,
`task6u_downstream_pairedval20.json` and `task6u_hardened_parser_integration.json`.

    python scripts/task6u_evaluate_downstream.py --stage causal
    python scripts/task6u_evaluate_downstream.py --stage parser
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.geometric_relation_field_v02 import geometric_relation_field_v02  # noqa: E402
from buildreasonseg_mvp.task6n_relation_decoder import (  # noqa: E402
    DecoderConfig,
    FrozenFeatureStore,
    RelationMaskDecoder,
    load_frozen_sam2_encoder,
    read_pack,
)
from buildreasonseg_mvp.task6q_reference_resolver import is_eligible, select_reference  # noqa: E402
from buildreasonseg_mvp.task6s_directional_pipeline import (  # noqa: E402
    PROGRAM_DECOMPOSITION,
    default_program_head_checkpoint,
    parse_instruction,
)
from buildreasonseg_mvp.task6u_reference_ranker import ProposalSetRanker, select_with_ranker  # noqa: E402
from scripts.task6u_common import (  # noqa: E402
    BUCKET_ORDER,
    CONFIGS,
    EVAL,
    PROPOSAL_CHECKPOINT,
    PROPOSAL_CHECKPOINT_SHA256,
    centroid_error,
    gt_masks,
    group_records_by_tile,
    iou,
    percentile,
    proposals_for_tile,
    record_image_path,
    sha256_file,
)
from task6n_evaluate import target_flags  # noqa: E402
from task6n_train import FEATURE_ROOT, MaskStore  # noqa: E402

OUT_MINI = EVAL / "task6u_downstream_minival240.json"
OUT_PAIRED = EVAL / "task6u_downstream_pairedval20.json"
OUT_PARSER = EVAL / "task6u_hardened_parser_integration.json"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6n" / "packs"
SELECTED = EVAL / "task6u_selected_proposal_config.json"
RANKER_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6u" / "reference_ranker_v01.pt"
FIELD_SIZE = (64, 64)
TARGET_SIZE = (512, 512)
RELATION_ORDER = ("left_of", "right_of", "above", "below")
SYSTEMS = {
    "U-S0": {"config": "U-C0", "selector": "deterministic"},
    "U-S1": {"config": None, "selector": "deterministic"},   # None -> the frozen selected config
    "U-S2": {"config": None, "selector": "ranker"},
}


def selected_config() -> dict:
    payload = json.loads(SELECTED.read_text(encoding="utf-8"))
    return CONFIGS[payload["selected_config"]]


def load_target_models(device: str):
    b3 = json.loads((EVAL / "task6o_mini_val.json").read_text(encoding="utf-8"))["variants"]["B3"]
    path = Path(b3["training"]["checkpoint"]["path"])
    model = RelationMaskDecoder("B3", DecoderConfig()).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["state_dict"])
    model.eval()
    encoder, _ = load_frozen_sam2_encoder(device=device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=device)
    return model, store, {"path": str(path), "sha256": sha256_file(path),
                          "expected_sha256": b3["training"]["checkpoint"]["sha256"]}


def load_ranker():
    if not RANKER_CHECKPOINT.is_file():
        return None
    model = ProposalSetRanker().to("cpu")
    payload = torch.load(RANKER_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def select_reference_mask(proposals, family: str, selector: str, ranker) -> dict:
    """Deterministic Task 6Q selector or the learned ranker over the eligible set."""

    if selector == "deterministic":
        selection = select_reference(proposals, family)
        if selection.abstained:
            return {"abstained": True, "reason": selection.reason, "mask": None}
        return {"abstained": False, "reason": None, "mask": selection.mask,
                "index": int(selection.proposal.index)}
    eligible = [proposal for proposal in proposals if is_eligible(proposal, family)]
    if not eligible:
        return {"abstained": True, "reason": "no_eligible_proposals", "mask": None}
    outcome = select_with_ranker(ranker, eligible, family, device="cpu")
    return {"abstained": False, "reason": None,
            "mask": eligible[outcome["selected_index"]].mask,
            "index": int(eligible[outcome["selected_index"]].index)}


def predict_target(target_model, store, sample, reference_mask, relation, device: str) -> np.ndarray:
    field = geometric_relation_field_v02(
        torch.as_tensor(np.asarray(reference_mask, dtype=np.float32)), relation, FIELD_SIZE)
    visual = store.get(sample.tile_id, Path(sample.image_path)).float().unsqueeze(0).to(device)
    index = torch.as_tensor([RELATION_ORDER.index(relation)], dtype=torch.long, device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                         enabled=str(device) != "cpu"):
        logits = target_model(visual, index, None, field.to(device))
    return (F.interpolate(logits.float(), size=TARGET_SIZE, mode="bilinear",
                          align_corners=False)[0, 0] > 0.0).cpu().numpy()


def run_causal(args) -> int:
    started = time.time()
    checkpoint_sha = sha256_file(PROPOSAL_CHECKPOINT) if PROPOSAL_CHECKPOINT.is_file() else None
    if checkpoint_sha != PROPOSAL_CHECKPOINT_SHA256:
        write_json(OUT_MINI, {"_doc": "Task 6U section 19.", "task": "6U",
                              "verdict": "PROPOSAL_CHECKPOINT_UNAVAILABLE"})
        return 2
    chosen = selected_config()
    configs = {"U-S0": CONFIGS["U-C0"], "U-S1": chosen, "U-S2": chosen}
    ranker = load_ranker()
    target_model, store, target_info = load_target_models(args.device)
    masks = MaskStore()
    samples = read_pack(PACK_ROOT / "mini_val_240.json")
    pairs = json.loads((PACK_ROOT / "paired_val_20.json").read_text(encoding="utf-8"))["pairs"]

    from ultralytics import YOLO

    yolo = YOLO(str(PROPOSAL_CHECKPOINT))
    proposal_cache: dict[tuple, list] = {}

    def proposals_for(tile_id: str, image_path: Path, config: dict):
        key = (config["id"], tile_id)
        if key not in proposal_cache:
            proposals, _run = proposals_for_tile(yolo, config, tile_id, image_path, args.device,
                                                 use_cache=True)
            proposal_cache[key] = proposals
        return proposal_cache[key]

    def evaluate_system(system: str, sample_list: list) -> dict:
        config = configs[system]
        selector = SYSTEMS[system]["selector"]
        rows = []
        for sample in sample_list:
            decomposition = PROGRAM_DECOMPOSITION[sample.program_id]
            family, relation = decomposition
            image_path = Path(sample.image_path)
            proposals = proposals_for(sample.tile_id, image_path, config)
            eligible = [proposal for proposal in proposals if is_eligible(proposal, family)]
            truth_reference = gt_masks(sample.tile_id)[int(sample.reference_source_feature_id)]
            best_eligible = max((iou(proposal.mask, truth_reference) for proposal in eligible),
                                default=0.0)
            selection = select_reference_mask(proposals, family, selector, ranker)
            row = {
                "sample_id": sample.sample_id, "tile_id": sample.tile_id,
                "program_id": sample.program_id, "reference_family": family, "relation": relation,
                "proposal_count": len(proposals), "eligible_count": len(eligible),
                "best_eligible_iou": best_eligible,
                "reference_abstained": selection["abstained"],
                "reference_abstention_reason": selection["reason"],
            }
            truth_target = np.asarray(masks.mask(sample.tile_id, sample.target_source_feature_id),
                                      dtype=bool)
            flags = target_flags(masks, sample)
            row.update({"target_touches_border": flags["target_touches_border"],
                        "target_tiny": flags["target_tiny"], "target_area_px": flags["target_area_px"]})
            if selection["abstained"]:
                row.update({"target_iou": 0.0, "target_dice": 0.0, "strict_iou": 0.0,
                            "reference_iou": None})
                row["bucket"] = ("NO_PROPOSALS" if not proposals else
                                 "NO_ELIGIBLE_PROPOSALS" if not eligible else
                                 "REFERENCE_NOT_COVERED_IOU50")
                if best_eligible >= 0.50:
                    row["bucket"] = "REFERENCE_NOT_COVERED_IOU50"
                rows.append(row)
                continue
            reference = selection["mask"]
            reference_iou = iou(reference, truth_reference)
            reference_centroid = centroid_error(reference, truth_reference)
            target = predict_target(target_model, store, sample, reference, relation, args.device)
            target_iou = iou(target, truth_target)
            intersection = float(np.logical_and(target, truth_target).sum())
            row.update({
                "reference_iou": reference_iou,
                "reference_centroid_error": reference_centroid,
                "target_iou": target_iou,
                "target_dice": (2.0 * intersection + 1e-6)
                / (float(target.sum()) + float(truth_target.sum()) + 1e-6),
                "strict_iou": target_iou,
                "selected_reference_index": selection["index"],
            })
            if not proposals:
                row["bucket"] = "NO_PROPOSALS"
            elif not eligible:
                row["bucket"] = "NO_ELIGIBLE_PROPOSALS"
            elif best_eligible < 0.50:
                row["bucket"] = "REFERENCE_NOT_COVERED_IOU50"
            elif reference_iou < 0.50:
                row["bucket"] = "REFERENCE_SELECTION_WRONG"
            elif reference_centroid > 0.05:
                row["bucket"] = "SELECTED_MASK_GEOMETRY_POOR"
            elif target_iou < 0.50:
                row["bucket"] = "TARGET_FAIL_WITH_REFERENCE_OK"
            else:
                row["bucket"] = "TARGET_OK"
            rows.append(row)
        return {"rows": rows}

    def summarise(rows: list[dict], label: str) -> dict:
        answered = [row for row in rows if not row["reference_abstained"]]
        target_ious = [row["target_iou"] for row in rows]
        answered_ious = [row["target_iou"] for row in answered]
        buckets = Counter(row["bucket"] for row in rows)
        reference_fail = sum(buckets[name] for name in (
            "NO_PROPOSALS", "NO_ELIGIBLE_PROPOSALS", "REFERENCE_NOT_COVERED_IOU50",
            "REFERENCE_SELECTION_WRONG", "SELECTED_MASK_GEOMETRY_POOR"))
        return {
            "label": label, "records": len(rows),
            "strict_all_miou": float(np.mean(target_ious)),
            "strict_all_dice": float(np.mean([row["target_dice"] for row in rows])),
            "answered_records": len(answered),
            "answered_only_miou": float(np.mean(answered_ious)) if answered_ious else None,
            "answered_only_dice": float(np.mean([row["target_dice"] for row in answered]))
            if answered else None,
            "precision_at_0_5": float(np.mean([value >= 0.50 for value in target_ious])),
            "abstentions": len(rows) - len(answered),
            "abstention_rate": (len(rows) - len(answered)) / len(rows),
            "reference_fail_count": reference_fail,
            "parser_fail_count": 0,
            "parser_bucket_status": "not applicable (canonical program ids)",
            "buckets": {name: buckets.get(name, 0) for name in
                        (*BUCKET_ORDER, "TARGET_FAIL_WITH_REFERENCE_OK", "TARGET_OK")},
            "largest": {
                "records": sum(1 for row in rows if row["reference_family"] == "largest"),
                "miou": float(np.mean([row["target_iou"] for row in rows
                                       if row["reference_family"] == "largest"]))
                if any(row["reference_family"] == "largest" for row in rows) else None},
            "smallest": {
                "records": sum(1 for row in rows if row["reference_family"] == "smallest"),
                "miou": float(np.mean([row["target_iou"] for row in rows
                                       if row["reference_family"] == "smallest"]))
                if any(row["reference_family"] == "smallest" for row in rows) else None},
            "per_direction": {
                relation: float(np.mean([row["target_iou"] for row in rows
                                         if row["relation"] == relation]))
                for relation in RELATION_ORDER
                if any(row["relation"] == relation for row in rows)},
            "border_target": {
                "records": sum(1 for row in rows if row["target_touches_border"]),
                "miou": float(np.mean([row["target_iou"] for row in rows
                                       if row["target_touches_border"]]))
                if any(row["target_touches_border"] for row in rows) else None},
            "tiny_target": {
                "records": sum(1 for row in rows if row["target_tiny"]),
                "miou": float(np.mean([row["target_iou"] for row in rows if row["target_tiny"]]))
                if any(row["target_tiny"] for row in rows) else None},
            "reference_miou": {
                "mean": float(np.mean([row["reference_iou"] for row in answered]))
                if answered else None,
                "precision_at_0_5": float(np.mean([row["reference_iou"] >= 0.50
                                                   for row in answered])) if answered else None,
                "centroid_median": float(np.median([row["reference_centroid_error"]
                                                    for row in answered])) if answered else None,
            },
        }

    mini_results = {name: evaluate_system(name, samples) for name in SYSTEMS}
    mini_payload = {
        "_doc": (
            "Task 6U section 19. Downstream causal evaluation on MiniVal240 with canonical program ids "
            "(no parser): canonical program -> family/relation -> U-S0/U-S1/U-S2 reference -> field v0.2 "
            "-> frozen SAM2 feature -> frozen B3 -> target mask. Only the proposal/reference stage "
            "differs between systems. GT is for evaluation only."
        ),
        "task": "6U", "stage": "I-downstream-minival240",
        "reference_source": "predicted_proposal_reference",
        "pack": {"path": str(PACK_ROOT / "mini_val_240.json"), "records": len(samples),
                 "sha256": sha256_file(PACK_ROOT / "mini_val_240.json")},
        "configs": {name: configs[name]["id"] for name in SYSTEMS},
        "selectors": {name: SYSTEMS[name]["selector"] for name in SYSTEMS},
        "target_decoder": target_info,
        "systems": {name: summarise(mini_results[name]["rows"], name) for name in SYSTEMS},
        "test_split_used": False,
        "parser_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_MINI, mini_payload)

    pair_rows = []
    for pair in pairs:
        from buildreasonseg_mvp.task6n_relation_decoder import Task6NSample

        members = [Task6NSample(**{key: value for key, value in pair[side].items()
                                   if key in Task6NSample.__dataclass_fields__})
                   for side in ("a", "b")]
        entry = {"tile_id": members[0].tile_id, "systems": {}}
        for name in SYSTEMS:
            config = configs[name]
            selector = SYSTEMS[name]["selector"]
            targets, own, cross, abstained = [], [], [], 0
            for index, member in enumerate(members):
                family, relation = PROGRAM_DECOMPOSITION[member.program_id]
                proposals = proposals_for(member.tile_id, Path(member.image_path), config)
                selection = select_reference_mask(proposals, family, selector, ranker)
                if selection["abstained"]:
                    abstained += 1
                    targets.append(None)
                    own.append(0.0)
                    cross.append(0.0)
                    continue
                target = predict_target(target_model, store, member, selection["mask"], relation,
                                        args.device)
                targets.append(target)
                own_mask = np.asarray(masks.mask(member.tile_id, member.target_source_feature_id),
                                      dtype=bool)
                own.append(float(np.logical_and(target, own_mask).sum())
                           / float(np.logical_or(target, own_mask).sum() + 1e-6))
                other = members[1 - index]
                other_mask = np.asarray(masks.mask(other.tile_id, other.target_source_feature_id),
                                        dtype=bool)
                cross.append(float(np.logical_and(target, other_mask).sum())
                             / float(np.logical_or(target, other_mask).sum() + 1e-6))
            entry["systems"][name] = {
                "own": own, "cross": cross, "abstained": abstained,
                "passes": bool(own[0] > cross[0] and own[1] > cross[1]),
            }
        pair_rows.append(entry)

    paired_payload = {
        "_doc": (
            "Task 6U section 20. PairedVal20 through the three reference systems with canonical program "
            "ids; a pair passes only if both members prefer their own GT target over the paired "
            "alternative by IoU."
        ),
        "task": "6U", "stage": "I-downstream-pairedval20",
        "pack": {"path": str(PACK_ROOT / "paired_val_20.json"), "pairs": len(pairs),
                 "sha256": sha256_file(PACK_ROOT / "paired_val_20.json")},
        "systems": {},
        "configs": {name: configs[name]["id"] for name in SYSTEMS},
        "test_split_used": False,
    }
    for name in SYSTEMS:
        rows = [entry["systems"][name] for entry in pair_rows]
        own = [value for row in rows for value in row["own"]]
        cross = [value for row in rows for value in row["cross"]]
        paired_payload["systems"][name] = {
            "passed": sum(1 for row in rows if row["passes"]),
            "pairs": len(rows),
            "mean_own_iou": float(np.mean(own)), "mean_cross_iou": float(np.mean(cross)),
            "own_cross_margin": float(np.mean(own) - np.mean(cross)),
            "reference_abstention_pairs": sum(1 for row in rows if row["abstained"]),
        }
    write_json(OUT_PAIRED, paired_payload)

    for name in SYSTEMS:
        summary = mini_payload["systems"][name]
        print(f"[6u.down] {name} ({configs[name]['id']}/{SYSTEMS[name]['selector']}): strict "
              f"{summary['strict_all_miou']:.4f} answered {summary['answered_only_miou']:.4f} "
              f"abstain {summary['abstentions']} ref_fail {summary['reference_fail_count']} | paired "
              f"{paired_payload['systems'][name]['passed']}/20 margin "
              f"{paired_payload['systems'][name]['own_cross_margin']:+.4f}", flush=True)
    return 0


def run_parser(args) -> int:
    started = time.time()
    from buildreasonseg_mvp.program_parser import build_program_parser, load_parser_checkpoint
    from buildreasonseg_mvp.runtime import load_config
    from buildreasonseg_mvp.structured_grounding import EXPECTED_QUERY_TYPES

    chosen = selected_config()
    ranker = load_ranker()
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
        write_json(OUT_PARSER, {"_doc": "Task 6U section 21.", "task": "6U",
                                "verdict": "PARSER_CHECKPOINT_UNAVAILABLE"})
        return 2
    cfg = load_config(REPO_ROOT / "configs" / "mvp" / "task6j_program_parser.yaml")
    runtime = build_program_parser(cfg, device=args.device, verbose=False)
    load_parser_checkpoint(checkpoint, runtime)

    from ultralytics import YOLO

    yolo = YOLO(str(PROPOSAL_CHECKPOINT))
    proposal_cache: dict[str, list] = {}
    rows = []
    for sample in samples:
        query = str(instructions[sample.sample_id]["instruction_en"])
        parsed = parse_instruction(runtime, query, tuple(EXPECTED_QUERY_TYPES))
        decomposition = PROGRAM_DECOMPOSITION.get(parsed["program"])
        row = {"sample_id": sample.sample_id, "expected_program": sample.program_id,
               "parsed_program": parsed["program"],
               "parser_correct": parsed["program"] == sample.program_id}
        if decomposition is None:
            row.update({"target_iou": 0.0, "strict_iou": 0.0, "reference_abstained": None,
                        "bucket": "PARSER_WRONG"})
            rows.append(row)
            continue
        family, relation = decomposition
        if sample.tile_id not in proposal_cache:
            proposals, _run = proposals_for_tile(yolo, chosen, sample.tile_id,
                                                 Path(sample.image_path), args.device,
                                                 use_cache=True)
            proposal_cache[sample.tile_id] = proposals
        proposals = proposal_cache[sample.tile_id]
        selection = select_reference_mask(proposals, family, "ranker", ranker)
        row["reference_abstained"] = selection["abstained"]
        if selection["abstained"]:
            row.update({"target_iou": 0.0, "strict_iou": 0.0, "bucket": "REFERENCE_NOT_COVERED_IOU50"})
            rows.append(row)
            continue
        truth_target = np.asarray(masks.mask(sample.tile_id, sample.target_source_feature_id),
                                  dtype=bool)
        target = predict_target(target_model, store, sample, selection["mask"], relation, args.device)
        target_iou = iou(target, truth_target)
        row.update({"target_iou": target_iou,
                    "strict_iou": target_iou if row["parser_correct"] else 0.0,
                    "bucket": "TARGET_OK" if target_iou >= 0.50 else "TARGET_FAIL_WITH_REFERENCE_OK"})
        rows.append(row)

    parser_correct = sum(1 for row in rows if row["parser_correct"])
    answered = [row for row in rows if row["reference_abstained"] is False]
    payload = {
        "_doc": (
            "Task 6U section 21. One final integration check: the Task 6T hardened ProgramHead + the "
            "U-S2 reference path + the frozen field/B3 on MiniVal240. Never used to train or choose "
            "U-S2; no nearest/L3 execution is evaluated."
        ),
        "task": "6U", "stage": "J-hardened-parser-integration",
        "parser_checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "reference_system": "U-S2",
        "reference_config": chosen["id"],
        "target_decoder": target_info,
        "pack": {"path": str(PACK_ROOT / "mini_val_240.json"), "records": len(samples)},
        "parser": {"exact_correct": parser_correct, "records": len(rows),
                   "exact_accuracy": parser_correct / len(rows)},
        "strict_all_miou": float(np.mean([row["strict_iou"] for row in rows])),
        "answered_only_miou": float(np.mean([row["target_iou"] for row in answered]))
        if answered else None,
        "abstentions": len(rows) - len(answered),
        "abstention_rate": (len(rows) - len(answered)) / len(rows),
        "buckets": dict(Counter(row["bucket"] for row in rows)),
        "reference_fail_count": sum(1 for row in rows if row["bucket"] in (
            "NO_PROPOSALS", "NO_ELIGIBLE_PROPOSALS", "REFERENCE_NOT_COVERED_IOU50",
            "REFERENCE_SELECTION_WRONG", "SELECTED_MASK_GEOMETRY_POOR")),
        "nearest_or_l3_execution_evaluated": False,
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_PARSER, payload)
    print(f"[6u.parser] parser {parser_correct}/{len(rows)} | strict "
          f"{payload['strict_all_miou']:.4f} answered {payload['answered_only_miou']:.4f} "
          f"abstain {payload['abstentions']}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("causal", "parser"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    return run_causal(args) if args.stage == "causal" else run_parser(args)


if __name__ == "__main__":
    raise SystemExit(main())
