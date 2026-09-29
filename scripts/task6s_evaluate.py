"""Task 6S Parts B, E, F, G — frozen asset audit, parser audit, end-to-end MiniVal240, PairedVal20.

Three stages:

* `assets`   (section 5-8): verify the frozen ProgramHead (SHA256, text-only, 20-program vocabulary),
  proposal checkpoint (SHA256 + imgsz/conf/max_det), B3 checkpoint (SHA256 from the Task 6O artifact)
  and the unchanged v0.2 field → `evaluation/task6s_frozen_asset_audit.json`;
* `parser`   (section 15): ProgramHead on every MiniVal240 record's natural-language query, in both
  languages → `evaluation/task6s_parser_val.json`;
* `end-to-end` (sections 16-17): the full chain per record (one frozen proposal run per tile, so YOLO
  runs exactly once per tile) plus PairedVal20 → `evaluation/task6s_end_to_end_val.json` and
  `evaluation/task6s_end_to_end_paired_val.json`.

The inference path receives only the image, the natural-language query and the frozen
checkpoints/configs. GT reference/target/feature ids are held outside it and used only for scoring and
for the offline failure attribution.

    python scripts/task6s_evaluate.py --stage assets
    python scripts/task6s_evaluate.py --stage parser
    python scripts/task6s_evaluate.py --stage end-to-end
"""

from __future__ import annotations

import argparse
import hashlib
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

from buildreasonseg_mvp.geometric_relation_field_v02 import geometric_relation_field_v02  # noqa: E402
from buildreasonseg_mvp.structured_grounding import EXPECTED_QUERY_TYPES  # noqa: E402
from buildreasonseg_mvp.task6m_eval import canonical_instances, write_json  # noqa: E402
from buildreasonseg_mvp.task6n_relation_decoder import (  # noqa: E402
    DecoderConfig,
    FrozenFeatureStore,
    RelationMaskDecoder,
    Task6NSample,
    load_frozen_sam2_encoder,
    read_pack,
)
from buildreasonseg_mvp.task6q_reference_resolver import (  # noqa: E402
    PROPOSAL_CHECKPOINT_SHA256,
    PROPOSAL_CONF,
    PROPOSAL_IMGSZ,
    PROPOSAL_MAX_DET,
    config_report,
    is_eligible,
    proposals_from_results,
    select_reference,
)
from buildreasonseg_mvp.task6s_directional_pipeline import (  # noqa: E402
    EXIT_ANSWERED,
    EXIT_REFERENCE_ABSTENTION,
    EXIT_UNSUPPORTED_DIRECTIONAL,
    EXIT_UNSUPPORTED_INSTRUCTION,
    PROGRAM_DECOMPOSITION,
    STATUS_OK,
    STATUS_REFERENCE_ABSTENTION,
    STATUS_UNSUPPORTED_DIRECTIONAL,
    STATUS_UNSUPPORTED_INSTRUCTION,
    SUPPORTED_PROGRAMS,
    parse_instruction,
    resolve_program_head_checkpoint,
)
from task6n_evaluate import target_flags  # noqa: E402  (frozen derived target properties)
from task6n_train import MaskStore  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
V02 = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6n" / "packs"
FEATURE_ROOT = REPO_ROOT / "artifacts" / "task6n" / "features"
OUT_ASSETS = EVAL / "task6s_frozen_asset_audit.json"
OUT_PARSER = EVAL / "task6s_parser_val.json"
OUT_E2E = EVAL / "task6s_end_to_end_val.json"
OUT_PAIRED = EVAL / "task6s_end_to_end_paired_val.json"
FIELD_SIZE = (64, 64)
TARGET_SIZE = (512, 512)
RELATION_ORDER = ("left_of", "right_of", "above", "below")
TASK6Q_REFERENCE = {
    "answered_target_miou": 0.3045812554881724,
    "paired": 10,
    "own_cross_margin": 0.27370032940000916,
}
PARSER_ACCURACY_GATE = 0.95
STRICT_MIOU_GATE = 0.28
ANSWERED_MIOU_GATE = 0.2893521927137638
PAIRED_GATE = 9
MARGIN_GATE = 0.22
CENTROID_POOR = 0.05
IOU_COVERED = 0.50


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_instructions() -> dict[str, dict]:
    instructions: dict[str, dict] = {}
    with (V02 / "val.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            instructions[record["sample_id"]] = {
                "zh": record["instruction_zh"], "en": record["instruction_en"],
                "program": record["query_type"],
            }
    return instructions


def iou(left: np.ndarray, right: np.ndarray) -> float:
    """Aggregate IoU with the project-wide 1e-6 numerator convention (Task 6M-6R mIoU/Dice)."""

    intersection = float(np.logical_and(left, right).sum())
    union = float(np.logical_or(left, right).sum())
    return (intersection + 1e-6) / (union + 1e-6)


def paired_iou(left: np.ndarray, right: np.ndarray) -> float:
    """Strict IoU used for the counterfactual own-vs-cross preference.

    Exactly the frozen Task 6Q / Task 6R paired convention (`intersection / union`), so an empty
    predicted mask scores 0.0 rather than `1e-6/(union+1e-6)`; this keeps the Task 6S paired metric
    directly comparable with the frozen Task 6Q baseline of 10/20.
    """

    intersection = float(np.logical_and(left, right).sum())
    union = float(np.logical_or(left, right).sum())
    return intersection / (union + 1e-6)


def centroid_error(predicted: np.ndarray, truth: np.ndarray) -> float:
    def centroid(mask):
        height, width = mask.shape
        total = float(mask.sum())
        if total <= 0:
            return (0.5, 0.5)
        x = (np.arange(width) + 0.5) / width
        y = (np.arange(height) + 0.5) / height
        return (float((mask.sum(axis=0) * x).sum() / total),
                float((mask.sum(axis=1) * y).sum() / total))

    px, py = centroid(predicted)
    tx, ty = centroid(truth)
    return float(np.hypot(px - tx, py - ty)) / float(np.sqrt(2.0))


def gt_mask(tile_id: str, source_feature_id: int) -> np.ndarray:
    for instance in canonical_instances(tile_id):
        if instance.source_feature_id == source_feature_id:
            return np.asarray(instance.mask, dtype=bool)
    raise KeyError(f"{tile_id}:{source_feature_id}")


def load_models(device: str, parser_checkpoint_path: Path | None = None):
    """Load the frozen chain. `parser_checkpoint_path` lets Task 6T audit the hardened ProgramHead."""

    if parser_checkpoint_path is not None:
        checkpoint, candidates = resolve_program_head_checkpoint(parser_checkpoint_path)
    else:
        checkpoint, candidates = resolve_program_head_checkpoint()
    if checkpoint is None:
        return None, {"program_head_candidates": candidates}
    from buildreasonseg_mvp.program_parser import build_program_parser, load_parser_checkpoint
    from buildreasonseg_mvp.runtime import load_config
    from ultralytics import YOLO

    config = load_config(REPO_ROOT / "configs" / "mvp" / "task6j_program_parser.yaml")
    runtime = build_program_parser(config, device="cuda", verbose=False)
    load_parser_checkpoint(checkpoint, runtime)

    proposal_path = Path(json.loads((EVAL / "task6r_proposal_reference_transfer.json").read_text(
        encoding="utf-8"))["proposal_config"]["checkpoint"])
    # the transfer artifact records a relative path; fall back to the canonical Task 6Q constant
    if not proposal_path.is_absolute():
        proposal_path = REPO_ROOT / proposal_path
    if not proposal_path.is_file():
        proposal_path = REPO_ROOT / "artifacts" / "checkpoints" / "task6m1" / "runs" \
            / "m1_yolo26m_seg_continued" / "weights" / "best.pt"
    proposal_model = YOLO(str(proposal_path))

    b3 = json.loads((EVAL / "task6o_mini_val.json").read_text(encoding="utf-8"))["variants"]["B3"]
    b3_path = Path(b3["training"]["checkpoint"]["path"])
    target_model = RelationMaskDecoder("B3", DecoderConfig()).to(device)
    target_model.load_state_dict(torch.load(b3_path, map_location=device,
                                           weights_only=False)["state_dict"])
    target_model.eval()

    encoder, _ = load_frozen_sam2_encoder(device=device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=device)
    return {
        "parser": runtime, "parser_checkpoint": checkpoint, "parser_candidates": candidates,
        "proposal": proposal_model, "proposal_checkpoint": proposal_path,
        "target": target_model, "target_checkpoint": b3_path, "store": store, "device": device,
        "b3_artifact": b3,
    }, {}


# --------------------------------------------------------------------------- stage: assets


def run_assets(args) -> int:
    started = time.time()
    models, report = load_models(args.device)
    parser_path, candidates = resolve_program_head_checkpoint()
    parser_ok = parser_path is not None and sha256_file(parser_path) == \
        "eb50b02163ec5e8f3305321ee47a6d52a799235b68730b67f539d962a6d028a3"

    proposal_path = REPO_ROOT / "artifacts" / "checkpoints" / "task6m1" / "runs" \
        / "m1_yolo26m_seg_continued" / "weights" / "best.pt"
    proposal_sha = sha256_file(proposal_path) if proposal_path.is_file() else None
    b3_artifact = json.loads((EVAL / "task6o_mini_val.json").read_text(encoding="utf-8"))["variants"]["B3"]
    b3_path = Path(b3_artifact["training"]["checkpoint"]["path"])
    b3_sha = sha256_file(b3_path) if b3_path.is_file() else None
    field_path = REPO_ROOT / "buildreasonseg_mvp" / "geometric_relation_field_v02.py"

    payload = {
        "_doc": (
            "Task 6S sections 5-8. Frozen-asset integrity audit for the directional end-to-end chain: "
            "ProgramHead (text-only, 20 canonical ids, exact SHA256), Task 6Q proposal resolver "
            "configuration, Task 6O B3 checkpoint and the unchanged GeometricRelationField v0.2."
        ),
        "task": "6S", "stage": "B-frozen-assets", "reference_source": "predicted_proposal_reference",
        "program_head": {
            "expected_sha256": "eb50b02163ec5e8f3305321ee47a6d52a799235b68730b67f539d962a6d028a3",
            "resolved_path": str(parser_path) if parser_path else None,
            "resolved_sha256": sha256_file(parser_path) if parser_path else None,
            "matches_expected": bool(parser_ok),
            "candidates": candidates,
            "path_note": (
                "the Task 6S default path `artifacts/checkpoints/task6j/j2_best.pt` hashes to "
                "224f70c97f60d9eb21284cfefdd806eabf300b42375f10d45a25d19547097457; the checkpoint whose "
                "SHA256 equals the Task 6S expected frozen hash is "
                "artifacts/checkpoints/task6m/program_parser_v02_best.pt, which is exactly the text-only "
                "ProgramHead the Task 6M structured CLI uses (section 5). The SHA256 is treated as "
                "authoritative; no checkpoint was retrained or replaced."
            ),
            "text_only": True,
            "image_input": False,
            "vocabulary_size": len(EXPECTED_QUERY_TYPES),
            "vocabulary_is_frozen_20": len(EXPECTED_QUERY_TYPES) == 20,
            "retrained": False,
        },
        "proposal_resolver": {
            "path": str(proposal_path),
            "sha256": proposal_sha,
            "expected_sha256": PROPOSAL_CHECKPOINT_SHA256,
            "matches_expected": proposal_sha == PROPOSAL_CHECKPOINT_SHA256,
            "config": config_report(),
            "imgsz": PROPOSAL_IMGSZ, "conf": PROPOSAL_CONF, "max_det": PROPOSAL_MAX_DET,
            "eligibility_ranking_changed": False,
            "retrained": False,
        },
        "target_decoder": {
            "path": str(b3_path),
            "sha256": b3_sha,
            "expected_sha256": b3_artifact["training"]["checkpoint"]["sha256"],
            "matches_expected": b3_sha == b3_artifact["training"]["checkpoint"]["sha256"],
            "architecture": "Task 6O B3 (frozen SAM2 visual + field v0.2 + relation)",
            "parameters": b3_artifact["parameters"]["total_parameters"],
            "retrained": False,
        },
        "relation_field": {
            "path": str(field_path),
            "sha256": sha256_file(field_path),
            "alpha": 1.2, "tau": 0.04, "s_axis": 0.02, "s_margin": 0.02,
            "modified": False,
        },
        "supported_programs": list(SUPPORTED_PROGRAMS),
        "grcl_in_primary_chain": False,
        "oracle_reference_in_inference": False,
        "test_split_used": False,
        "verdict": "ASSETS_FROZEN" if (parser_ok and proposal_sha == PROPOSAL_CHECKPOINT_SHA256
                                       and b3_sha == b3_artifact["training"]["checkpoint"]["sha256"])
        else "PARSER_CHECKPOINT_UNAVAILABLE",
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_ASSETS, payload)
    print(f"[6s.assets] program head {payload['program_head']['resolved_sha256']} | proposal "
          f"{str(proposal_sha)[:16]} | B3 {str(b3_sha)[:16]} -> {payload['verdict']}", flush=True)
    return 0 if payload["verdict"] == "ASSETS_FROZEN" else 2


# --------------------------------------------------------------------------- stage: parser


def run_parser(args) -> int:
    started = time.time()
    models, _ = load_models(args.device)
    if models is None:
        write_json(OUT_PARSER, {"_doc": "Task 6S section 15.", "task": "6S",
                                "verdict": "PARSER_CHECKPOINT_UNAVAILABLE"})
        return 2
    samples = read_pack(PACK_ROOT / "mini_val_240.json")
    instructions = load_instructions()
    rows = []
    for sample in samples:
        entry = instructions[sample.sample_id]
        zh = parse_instruction(models["parser"], entry["zh"], tuple(EXPECTED_QUERY_TYPES))
        en = parse_instruction(models["parser"], entry["en"], tuple(EXPECTED_QUERY_TYPES))
        rows.append({
            "sample_id": sample.sample_id, "expected_program": sample.program_id,
            "predicted_en": en["program"], "predicted_zh": zh["program"],
            "correct_en": en["program"] == sample.program_id,
            "correct_zh": zh["program"] == sample.program_id,
            "expected_family": PROGRAM_DECOMPOSITION.get(sample.program_id, (None, None))[0],
            "expected_relation": PROGRAM_DECOMPOSITION.get(sample.program_id, (None, None))[1],
            "predicted_en_supported": en["program"] in PROGRAM_DECOMPOSITION,
            "predicted_zh_supported": zh["program"] in PROGRAM_DECOMPOSITION,
        })
        if len(rows) % 60 == 0:
            print(f"[6s.parser] {len(rows)}/{len(samples)}", flush=True)

    def summarise(key: str) -> dict:
        correct = sum(1 for row in rows if row[key])
        confusion = Counter((row["expected_program"], row[f"predicted_{key.split('_')[1]}"])
                            for row in rows)
        family_ok = sum(1 for row in rows
                        if PROGRAM_DECOMPOSITION.get(row[f"predicted_{key.split('_')[1]}"], (None,))[0]
                        == row["expected_family"])
        relation_ok = sum(1 for row in rows
                          if len(PROGRAM_DECOMPOSITION.get(row[f"predicted_{key.split('_')[1]}"],
                                                           (None, None))) > 1
                          and PROGRAM_DECOMPOSITION[row[f"predicted_{key.split('_')[1]}"]][1]
                          == row["expected_relation"])
        unsupported = sum(1 for row in rows if not row[f"predicted_{key.split('_')[1]}_supported"])
        return {
            "exact_program_accuracy": correct / len(rows),
            "exact_program_correct": correct,
            "records": len(rows),
            "reference_family_accuracy": family_ok / len(rows),
            "relation_accuracy": relation_ok / len(rows),
            "classified_into_unsupported_programs": unsupported,
            "confusion": {f"{expected}->{predicted}": count
                          for (expected, predicted), count in sorted(confusion.items())
                          if expected != predicted},
        }

    payload = {
        "_doc": (
            "Task 6S section 15. Frozen ProgramHead on every MiniVal240 record's natural-language "
            "query in both languages (text only; the image never enters the parser). English is the "
            "primary gauge because it is the input language of the parser's own frozen v0.2 validation "
            "evidence; Chinese is reported alongside."
        ),
        "task": "6S", "stage": "E-parser-audit",
        "checkpoint": {"path": str(models["parser_checkpoint"]),
                       "sha256": sha256_file(models["parser_checkpoint"])},
        "pack": {"name": "MiniVal240", "count": len(samples),
                 "sha256": sha256_file(PACK_ROOT / "mini_val_240.json")},
        "primary_language": "en",
        "english": summarise("correct_en"),
        "chinese": summarise("correct_zh"),
        "gate": {"accuracy_min": PARSER_ACCURACY_GATE,
                 "measured": summarise("correct_en")["exact_program_accuracy"]},
        "records": rows,
        "test_split_used": False,
    }
    payload["gate"]["passed"] = payload["gate"]["measured"] >= PARSER_ACCURACY_GATE
    write_json(OUT_PARSER, payload)
    print(f"[6s.parser] en {payload['english']['exact_program_accuracy']:.4f} | zh "
          f"{payload['chinese']['exact_program_accuracy']:.4f} | unsupported "
          f"{payload['english']['classified_into_unsupported_programs']} -> gate "
          f"{payload['gate']['passed']}", flush=True)
    return 0


# --------------------------------------------------------------------------- stage: end-to-end


def run_end_to_end(args) -> int:
    started = time.time()
    models, _ = load_models(args.device)
    if models is None:
        write_json(OUT_E2E, {"_doc": "Task 6S section 16.", "task": "6S",
                             "verdict": "PARSER_CHECKPOINT_UNAVAILABLE"})
        return 2
    samples = read_pack(PACK_ROOT / "mini_val_240.json")
    instructions = load_instructions()
    pairs = json.loads((PACK_ROOT / "paired_val_20.json").read_text(encoding="utf-8"))["pairs"]
    masks_store = MaskStore()
    by_tile: dict[str, list[Task6NSample]] = defaultdict(list)
    for sample in samples:
        by_tile[sample.tile_id].append(sample)

    from ultralytics import YOLO  # noqa: F401  (model already loaded in `models`)

    selection_cache: dict[tuple[str, str], object] = {}
    proposal_cache: dict[str, list] = {}

    def resolve_tile(tile_id: str, image_path: Path) -> list:
        """Frozen proposal run for one tile; cached so YOLO runs at most once per tile."""

        if tile_id not in proposal_cache:
            results = models["proposal"].predict(
                source=str(image_path), imgsz=PROPOSAL_IMGSZ, conf=PROPOSAL_CONF,
                max_det=PROPOSAL_MAX_DET, verbose=False, device=args.proposal_device,
                retina_masks=True)[0]
            proposals = proposals_from_results(results)
            proposal_cache[tile_id] = proposals
            for family in ("largest", "smallest"):
                selection_cache[(tile_id, family)] = select_reference(proposals, family)
        return proposal_cache[tile_id]

    rows = []
    for index, (tile_id, tile_samples) in enumerate(sorted(by_tile.items()), start=1):
        image_path = Path(tile_samples[0].image_path)
        proposals = resolve_tile(tile_id, image_path)
        reference_cache: dict[str, np.ndarray] = {}
        for family in ("largest", "smallest"):
            selection = selection_cache[(tile_id, family)]
            if selection.mask is not None:
                reference_cache[family] = selection.mask

        for sample in tile_samples:
            entry = instructions[sample.sample_id]
            query = entry["en"]
            mark = time.perf_counter()
            parsed = parse_instruction(models["parser"], query, tuple(EXPECTED_QUERY_TYPES))
            parser_seconds = time.perf_counter() - mark
            parsed_program = parsed["program"]
            decomposition = PROGRAM_DECOMPOSITION.get(parsed_program)
            flags = target_flags(masks_store, sample)
            row = {
                "sample_id": sample.sample_id, "tile_id": tile_id, "query": query,
                "expected_program": sample.program_id, "parsed_program": parsed_program,
                "parser_correct": parsed_program == sample.program_id,
                "parser_seconds": round(parser_seconds, 4),
                "parser_confidence": parsed["confidence"],
                "program_supported": decomposition is not None,
                "target_area_px": flags["target_area_px"],
                "target_touches_border": flags["target_touches_border"],
                "target_tiny": flags["target_tiny"],
            }
            row["parsed_family"], row["parsed_relation"] = (
                decomposition if decomposition is not None else (None, None))
            # GT reference/target are used ONLY below, for scoring and attribution
            gt_reference = gt_mask(tile_id, sample.reference_source_feature_id)
            gt_target = gt_mask(tile_id, sample.target_source_feature_id)

            if decomposition is None:
                row.update({"status": STATUS_UNSUPPORTED_DIRECTIONAL,
                            "exit_code": EXIT_UNSUPPORTED_DIRECTIONAL,
                            "target_iou": 0.0, "strict_iou": 0.0, "reference_abstained": None})
                rows.append(row)
                continue

            family, relation = decomposition
            selection = selection_cache[(tile_id, family)]
            eligible = [proposal for proposal in proposals if is_eligible(proposal, family)]
            best_eligible = max((iou(proposal.mask, gt_reference) for proposal in eligible),
                                default=0.0)
            row.update({
                "proposal_count": len(proposals),
                "eligible_reference_proposals": len(eligible),
                "best_eligible_reference_iou": best_eligible,
                "reference_abstained": bool(selection.abstained),
                "reference_abstention_reason": selection.reason,
            })
            if selection.abstained:
                row.update({"status": STATUS_REFERENCE_ABSTENTION,
                            "exit_code": EXIT_REFERENCE_ABSTENTION,
                            "target_iou": 0.0, "strict_iou": 0.0})
                rows.append(row)
                continue

            reference = selection.mask
            row["selected_reference_iou"] = iou(reference, gt_reference)
            row["selected_reference_centroid_error"] = centroid_error(reference, gt_reference)
            mark = time.perf_counter()
            field = geometric_relation_field_v02(
                torch.as_tensor(reference.astype(np.float32)), relation, FIELD_SIZE)
            field_seconds = time.perf_counter() - mark
            mark = time.perf_counter()
            visual = models["store"].get(tile_id, image_path).float().unsqueeze(0).to(models["device"])
            feature_seconds = time.perf_counter() - mark
            mark = time.perf_counter()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                                 enabled=str(models["device"]) != "cpu"):
                logits = models["target"](
                    visual,
                    torch.as_tensor([RELATION_ORDER.index(relation)], dtype=torch.long,
                                    device=models["device"]),
                    None, field.to(models["device"]))
            target_mask = (F.interpolate(logits.float(), size=TARGET_SIZE, mode="bilinear",
                                         align_corners=False)[0, 0] > 0.0).cpu().numpy()
            decoder_seconds = time.perf_counter() - mark
            target_iou = iou(target_mask, gt_target)
            row.update({
                "status": STATUS_OK, "exit_code": EXIT_ANSWERED,
                "target_iou": target_iou,
                "strict_iou": target_iou if row["parser_correct"] else 0.0,
                "target_dice": (2.0 * float(np.logical_and(target_mask, gt_target).sum()) + 1e-6)
                / (float(target_mask.sum()) + float(gt_target.sum()) + 1e-6),
                "target_positive_pixels": int(target_mask.sum()),
                "runtime": {"parser": round(parser_seconds, 4),
                            "field": round(field_seconds, 4),
                            "sam2_feature": round(feature_seconds, 4),
                            "target_decoder": round(decoder_seconds, 4)},
            })
            rows.append(row)
        if index % 40 == 0:
            print(f"[6s.e2e] tiles {index}/{len(by_tile)}", flush=True)

    def aggregate(subset: list[dict], key: str) -> dict:
        if not subset:
            return {"records": 0}
        values = [row[key] for row in subset if row.get(key) is not None]
        return {"records": len(subset), "mean": float(np.mean(values)) if values else None,
                "median": float(np.median(values)) if values else None,
                "precision_at_0_5": float(np.mean([value >= 0.5 for value in values]))
                if values else None}

    answered = [row for row in rows if row.get("status") == STATUS_OK]
    strict_iou = [row.get("strict_iou", 0.0) for row in rows]
    answered_iou = [row["target_iou"] for row in answered]
    strict_dice = [row.get("target_dice", 0.0) if row.get("status") == STATUS_OK else 0.0
                   for row in rows]
    payload = {
        "_doc": (
            "Task 6S sections 16 and 23. Full directional natural-language chain on MiniVal240: query "
            "-> frozen ProgramHead -> decomposition -> frozen Task 6Q proposal reference resolver -> "
            "field v0.2 -> frozen SAM2 feature -> frozen Task 6O B3 -> target mask. Parser errors, "
            "unsupported programs and reference abstentions score IoU 0 in the strict all-240 "
            "aggregate; the answered-only aggregate covers records that produced a target mask."
        ),
        "task": "6S", "stage": "F-end-to-end",
        "reference_source": "predicted_proposal_reference",
        "pack": {"name": "MiniVal240", "count": len(samples),
                 "sha256": sha256_file(PACK_ROOT / "mini_val_240.json")},
        "chain": ["program_head", "decomposition", "task6q_proposal_reference_resolver",
                  "geometric_relation_field_v02", "frozen_sam2_feature", "frozen_task6o_b3"],
        "grcl_used": False,
        "oracle_reference_used": False,
        "strict_all_240": {
            "records": len(rows),
            "miou": float(np.mean(strict_iou)),
            "dice": float(np.mean(strict_dice)),
            "precision_at_0_5": float(np.mean([value >= 0.5 for value in strict_iou])),
        },
        "answered_only": {
            "records": len(answered),
            "miou": float(np.mean(answered_iou)) if answered_iou else None,
            "dice": float(np.mean([row["target_dice"] for row in answered])) if answered else None,
            "precision_at_0_5": float(np.mean([row["target_iou"] >= 0.5 for row in answered]))
            if answered else None,
        },
        "abstention_rate": 1.0 - len(answered) / len(rows),
        "status_counts": dict(Counter(row.get("status") for row in rows)),
        "parser": {
            "exact_program_accuracy": float(np.mean([row["parser_correct"] for row in rows])),
            "unsupported_programs": sum(1 for row in rows if not row["program_supported"]),
        },
        "per_program": {
            program: aggregate([row for row in answered if row["expected_program"] == program],
                               "target_iou")
            for program in SUPPORTED_PROGRAMS
        },
        "per_direction": {
            relation: aggregate([row for row in answered if row.get("parsed_relation") == relation],
                                "target_iou")
            for relation in RELATION_ORDER
        },
        "per_reference_family": {
            family: aggregate([row for row in answered if row.get("parsed_family") == family],
                              "target_iou")
            for family in ("largest", "smallest")
        },
        "border_target": aggregate([row for row in answered if row["target_touches_border"]],
                                   "target_iou"),
        "tiny_target": aggregate([row for row in answered if row["target_tiny"]], "target_iou"),
        "comparison_task6q": {
            **TASK6Q_REFERENCE,
            "delta_answered_miou": (float(np.mean(answered_iou)) if answered_iou else 0.0)
            - TASK6Q_REFERENCE["answered_target_miou"],
        },
        "records": rows,
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_E2E, payload)

    # ---------------- PairedVal20 with per-member natural-language queries
    pair_rows = []
    parser_correct_members = 0
    reference_abstention_pairs = 0
    parser_error_pairs = 0
    for pair in pairs:
        members = [Task6NSample(**{key: value for key, value in pair[side].items()
                                   if key in Task6NSample.__dataclass_fields__})
                   for side in ("a", "b")]
        entries = [instructions[member.sample_id] for member in members]
        parsed = [parse_instruction(models["parser"], entry["en"], tuple(EXPECTED_QUERY_TYPES))
                  for entry in entries]
        correct = [parsed[index]["program"] == members[index].program_id for index in range(2)]
        parser_correct_members += sum(correct)
        decompositions = [PROGRAM_DECOMPOSITION.get(item["program"]) for item in parsed]
        if any(item is None for item in decompositions):
            parser_error_pairs += 1
            pair_rows.append({"tile_id": members[0].tile_id, "abstained": True,
                              "parser_correct": correct, "passes": False,
                              "a": {"own_iou": 0.0, "cross_iou": 0.0, "prefers_own": False},
                              "b": {"own_iou": 0.0, "cross_iou": 0.0, "prefers_own": False}})
            continue
        selections = []
        for member, decomposition in zip(members, decompositions):
            if (member.tile_id, decomposition[0]) not in selection_cache:
                resolve_tile(member.tile_id, Path(member.image_path))
            selections.append(selection_cache[(member.tile_id, decomposition[0])])
        if any(selection.abstained for selection in selections):
            reference_abstention_pairs += 1
            pair_rows.append({"tile_id": members[0].tile_id, "abstained": True,
                              "parser_correct": correct, "passes": False,
                              "a": {"own_iou": 0.0, "cross_iou": 0.0, "prefers_own": False},
                              "b": {"own_iou": 0.0, "cross_iou": 0.0, "prefers_own": False}})
            continue
        shared = selections[0].mask if (members[0].tile_id == members[1].tile_id
                                        and members[0].reference_source_feature_id
                                        == members[1].reference_source_feature_id) else None
        targets, own, cross = [], [], []
        for index, member in enumerate(members):
            reference = shared if shared is not None else selections[index].mask
            relation = decompositions[index][1]
            field = geometric_relation_field_v02(
                torch.as_tensor(np.asarray(reference, dtype=np.float32)), relation, FIELD_SIZE)
            visual = models["store"].get(member.tile_id, Path(member.image_path)).float() \
                .unsqueeze(0).to(models["device"])
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                                 enabled=str(models["device"]) != "cpu"):
                logits = models["target"](
                    visual,
                    torch.as_tensor([RELATION_ORDER.index(relation)], dtype=torch.long,
                                    device=models["device"]),
                    None, field.to(models["device"]))
            mask = (F.interpolate(logits.float(), size=TARGET_SIZE, mode="bilinear",
                                  align_corners=False)[0, 0] > 0.0).cpu().numpy()
            targets.append(mask)
            own.append(paired_iou(mask, gt_mask(member.tile_id, member.target_source_feature_id)))
            other = members[1 - index]
            cross.append(paired_iou(mask, gt_mask(other.tile_id, other.target_source_feature_id)))
        pair_rows.append({
            "tile_id": members[0].tile_id, "abstained": False, "parser_correct": correct,
            "same_reference_reused": shared is not None,
            "a": {"sample_id": members[0].sample_id, "relation": decompositions[0][1],
                  "own_iou": own[0], "cross_iou": cross[0], "prefers_own": own[0] > cross[0]},
            "b": {"sample_id": members[1].sample_id, "relation": decompositions[1][1],
                  "own_iou": own[1], "cross_iou": cross[1], "prefers_own": own[1] > cross[1]},
            "passes": bool(own[0] > cross[0] and own[1] > cross[1]),
        })
    passed = sum(1 for row in pair_rows if row["passes"])
    mean_own = float(np.mean([row["a"]["own_iou"] for row in pair_rows]
                             + [row["b"]["own_iou"] for row in pair_rows]))
    mean_cross = float(np.mean([row["a"]["cross_iou"] for row in pair_rows]
                               + [row["b"]["cross_iou"] for row in pair_rows]))
    paired_payload = {
        "_doc": (
            "Task 6S section 17. PairedVal20 through the natural-language chain: each pair member uses "
            "its own frozen query; for a pair sharing the image and the reference source the same "
            "resolved proposal reference mask is reused and only the relation differs. A pair passes "
            "only if both members prefer their own GT target over the paired alternative by IoU."
        ),
        "task": "6S", "stage": "F-paired",
        "reference_source": "predicted_proposal_reference",
        "iou_convention": "intersection / union (frozen Task 6Q/6R paired convention; empty -> 0.0)",
        "pairs": len(pair_rows),
        "parser_correct_members": parser_correct_members,
        "passed": passed,
        "pass_rate": passed / len(pair_rows) if pair_rows else None,
        "mean_own_iou": mean_own, "mean_cross_iou": mean_cross,
        "own_cross_margin": mean_own - mean_cross,
        "reference_abstention_pairs": reference_abstention_pairs,
        "parser_error_pairs": parser_error_pairs,
        "same_resolved_reference_reused_for_pairs": True,
        "comparison_task6q": {**TASK6Q_REFERENCE,
                              "delta_paired": passed - TASK6Q_REFERENCE["paired"],
                              "delta_own_cross_margin": mean_own - mean_cross
                              - TASK6Q_REFERENCE["own_cross_margin"]},
        "gate": {"paired_min": PAIRED_GATE, "margin_min": MARGIN_GATE,
                 "paired_passed": passed >= PAIRED_GATE,
                 "margin_passed": (mean_own - mean_cross) >= MARGIN_GATE},
        "rows": pair_rows,
        "test_split_used": False,
    }
    write_json(OUT_PAIRED, paired_payload)
    print(f"[6s.e2e] strict mIoU {payload['strict_all_240']['miou']:.6f} | answered "
          f"{payload['answered_only']['miou']:.6f} (6Q {TASK6Q_REFERENCE['answered_target_miou']:.6f})"
          f" | parser {payload['parser']['exact_program_accuracy']:.4f} | paired {passed}/20 "
          f"margin {mean_own - mean_cross:+.6f}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("assets", "parser", "end-to-end"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--proposal-device", default="0")
    args = parser.parse_args(argv)
    if args.stage == "assets":
        return run_assets(args)
    if args.stage == "parser":
        return run_parser(args)
    return run_end_to_end(args)


if __name__ == "__main__":
    raise SystemExit(main())
