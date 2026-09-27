#!/usr/bin/env python
"""Task 6D sections 11-17: final evaluation, verdict, error analysis and panels.

    python scripts/task6d_evaluate.py --checkpoint artifacts/checkpoints/task6d_G1/G1_epoch3.pt
    python scripts/task6d_evaluate.py --targets-only        # GT geometry artifact only

Produces

    evaluation/task6d_grounding_targets.json    GT geometry for the paired/train/val sets
    evaluation/task6d_paired_probe.json         mask + geometry paired probe (free generation)
    evaluation/task6d_representation.json       same-image representation diagnostics
    evaluation/task6d_error_analysis.json       WHU pseudo-instance failure classification
    evaluation/task6d_checkpoint_manifest.json  (merged; checkpoints themselves stay local)
    evaluation/task6d_panels/*.png              compact qualitative panels (>= 6 pairs)

Free generation is the primary setting (section 11): image + instruction -> generated
reasoning + `[SEG]` -> generated `[SEG]` hidden -> predicted geometry -> official SAM2
prompt encoder -> mask. Exactly one `[SEG]` is required; an invalid emission scores
IoU = 0 and counts as a geometry failure. The verdict is exactly one of section 13's.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.checkpointing import load_checkpoint, sha256_file  # noqa: E402
from buildreasonseg_mvp.grounding import (  # noqa: E402
    GEOMETRY_BOX,
    GEOMETRY_POINT,
    target_geometry,
)
from buildreasonseg_mvp.grounding_eval import (  # noqa: E402
    free_generation_grounded,
    paired_probe_grounded,
    prediction_component_overlap,
    representation_diagnostics,
    target_component_stats,
)
from buildreasonseg_mvp.runtime import build_runtime, load_config, set_seed  # noqa: E402
from task6c_train import subset_records, validation_material  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
TASK6C_SUBSETS = EVAL / "task6c_subset_ids.json"
ORACLE_JSON = EVAL / "task6d_oracle_prompt_diagnostic.json"
CONFIG = REPO_ROOT / "configs" / "mvp" / "task6c_2b_ablation.yaml"
PANELS = EVAL / "task6d_panels"

#: Section 13 gates.
FIX_EMISSION = 0.90
FIX_PAIRED = 14
FIX_MARGIN = 0.05
FIX_MIOU = 0.20

#: Section 16 failure classes.
FAILURE_CLASSES = (
    "border_truncation",
    "touching_neighbours",
    "merged_prediction",
    "ambiguous_boundary",
    "insufficient_density",
    "tiny_target",
    "disconnected_target",
)


def load_oracle() -> dict:
    return json.loads(ORACLE_JSON.read_text(encoding="utf-8"))


def build_targets_artifact() -> dict:
    """GT geometry for every set, plus the evidence that paired targets differ."""

    kind = load_oracle()["selection"]["chosen_geometry"]
    train_records = subset_records(json.loads(TASK6C_SUBSETS.read_text(encoding="utf-8")), "P")
    train_samples = [data_mod.to_sample(record) for record in train_records]
    val_samples, pairs, _lookup, _audit = validation_material()

    def geometry_record(sample) -> dict:
        mask = sample.target_mask()
        geometry = target_geometry(mask, kind)
        return {
            "sample_id": sample.sample_id,
            "image_id": sample.image_id,
            "geometry": list(geometry),
            "target_area_px": int(mask.sum()),
            "target_area_fraction": float(mask.mean()),
            "box_px": list(target_geometry(mask, GEOMETRY_BOX)),
            "point_xy": list(target_geometry(mask, GEOMETRY_POINT)),
            "target_component_id": sample.target_component_id,
        }

    paired = []
    for pair in pairs:
        sample_a = data_mod.to_sample(pair["a"])
        sample_b = data_mod.to_sample(pair["b"])
        geometry_a = geometry_record(sample_a)
        geometry_b = geometry_record(sample_b)
        values_a = np.asarray(geometry_a["geometry"])
        values_b = np.asarray(geometry_b["geometry"])
        mask_a = sample_a.target_mask()
        mask_b = sample_b.target_mask()
        intersection = float((mask_a & mask_b).sum())
        union = float((mask_a | mask_b).sum())
        paired.append(
            {
                "image_id": sample_a.image_id,
                "a": geometry_a,
                "b": geometry_b,
                "geometry_l1": float(np.abs(values_a - values_b).mean()),
                "geometry_distance": float(np.linalg.norm(values_a - values_b)),
                "gt_mask_iou_a_vs_b": intersection / union if union else 0.0,
                "targets_differ": bool(not np.allclose(values_a, values_b)),
            }
        )

    return {
        "_doc": (
            "Task 6D section 17: ground-truth target geometry (box and point) for the Task 6C paired P "
            "training subset, the fixed 120-record validation set and the 20 paired validation images. "
            "This is the supervision/oracle geometry; inference never receives it."
        ),
        "task": "6D",
        "geometry_kind": kind,
        "box_definition": "tight normalized (x1,y1,x2,y2) around the GT target mask, x1<=x2, y1<=y2",
        "point_definition": (
            "maximum of the Euclidean distance transform inside the GT target mask, ties broken by "
            "row-major argmax"
        ),
        "train": [geometry_record(sample) for sample in train_samples],
        "val": [geometry_record(sample) for sample in val_samples],
        "paired": paired,
        "paired_targets_all_differ": bool(all(record["targets_differ"] for record in paired)),
        "mean_paired_geometry_distance": float(
            np.mean([record["geometry_distance"] for record in paired])
        ),
        "mean_paired_mask_iou": float(np.mean([record["gt_mask_iou_a_vs_b"] for record in paired])),
    }


# ---------------------------------------------------------------- error analysis


def error_analysis(runtime, samples, free_result: dict, kind: str) -> dict:
    records = free_result["run"]["records"]
    analysed = []
    for record in records:
        sample = next(item for item in samples if item.sample_id == record["sample_id"])
        stats = target_component_stats(sample)
        overlap = record.get("component_overlap") or {}
        flags = {
            "border_truncation": bool(stats.get("touches_border")),
            "touching_neighbours": bool(stats.get("touching_neighbour_count", 0) > 0),
            "merged_prediction": bool(overlap.get("multi_component")),
            "ambiguous_boundary": bool(
                overlap.get("multi_component") and record["iou"] < 0.3
            ),
            "insufficient_density": bool(stats.get("components_in_tile", 0) <= 2),
            "tiny_target": bool(stats.get("target_area_fraction", 1.0) < 0.01),
            "disconnected_target": bool(stats.get("target_connected_components", 1) > 1),
        }
        analysed.append(
            {
                "sample_id": record["sample_id"],
                "iou": record["iou"],
                "seg_valid": record["seg_valid"],
                "flags": flags,
                "component_stats": stats,
                "prediction_overlap": overlap or None,
            }
        )

    failures = [record for record in analysed if record["iou"] < 0.1]
    flag_counts = {
        name: sum(1 for record in analysed if record["flags"][name]) for name in FAILURE_CLASSES
    }
    failure_flag_counts = {
        name: sum(1 for record in failures if record["flags"][name]) for name in FAILURE_CLASSES
    }
    share = {
        name: (failure_flag_counts[name] / len(failures) if failures else None)
        for name in FAILURE_CLASSES
    }
    dominant = max(share.items(), key=lambda item: item[1] or 0) if failures else None
    # Degeneracy guard: if the model emits essentially one constant mask, every record fails and
    # the dataset flags only describe the failure population -- they cannot be blamed for it.
    distinct_predictions = len(
        {record["mask_sha"] for record in free_result["run"]["records"] if record.get("mask_sha")}
    )
    valid_records = [record for record in free_result["run"]["records"] if record["seg_valid"]]
    mean_own = (
        sum(record["iou"] for record in valid_records) / len(valid_records) if valid_records else 0.0
    )
    degenerate = bool(
        (distinct_predictions <= 2 and len(valid_records) > 10) or mean_own < 0.01
    )
    return {
        "_doc": (
            "Task 6D section 16. WHU pseudo-instance error classification for the free-generation "
            "validation records. Flags are derived from the GT component map (the WHU polygon "
            "components) and the GT mask, never from a prediction."
        ),
        "task": "6D",
        "records": len(analysed),
        "failure_threshold_iou": 0.1,
        "failures": len(failures),
        "flag_counts_all_records": flag_counts,
        "flag_counts_failures": failure_flag_counts,
        "flag_share_of_failures": share,
        "dominant_flag_among_failures": dominant[0] if dominant else None,
        "detail": analysed,
        "prediction_degeneracy": {
            "distinct_predicted_masks": distinct_predictions,
            "valid_records": len(valid_records),
            "mean_own_iou": mean_own,
            "degenerate": degenerate,
        },
        "whu_limitation_judgement": _judgement(share, len(failures), len(analysed), degenerate),
        "note": "The dataset is not changed in Task 6D (section 1).",
    }


def _judgement(share: dict, failures: int, total: int, degenerate: bool = False) -> dict:
    if not failures:
        return {"materially_limiting": False, "reason": "no failures at the 0.1 IoU threshold"}
    dominant = max(share.items(), key=lambda item: item[1] or 0)
    if degenerate:
        return {
            "materially_limiting": False,
            "dominant_flag": dominant[0],
            "dominant_share": dominant[1],
            "failure_rate": failures / max(1, total),
            "reason": (
                "NOT attributable to the dataset: the model emitted essentially one constant mask, "
                f"so all {failures}/{total} records fail regardless of their content. The flag "
                f"distribution (`{dominant[0]}` at {dominant[1]:.0%} of failures) describes the "
                "failure population, not a dataset defect, and this evidence does not indicate a "
                "dataset-selection task."
            ),
        }
    material = bool(dominant[1] and dominant[1] >= 0.5 and failures >= 0.25 * max(1, total))
    return {
        "materially_limiting": material,
        "dominant_flag": dominant[0],
        "dominant_share": dominant[1],
        "failure_rate": failures / max(1, total),
        "reason": (
            f"{dominant[1]:.0%} of failures carry `{dominant[0]}`, and {failures}/{total} records fail; "
            + (
                "this materially limits the architecture experiment, so a dataset-selection task is "
                "recommended next."
                if material
                else "this does not materially dominate, so the architecture experiment remains "
                "interpretable on the current dataset."
            )
        ),
    }


# ---------------------------------------------------------------- panels


def write_panels(runtime, pairs, lookup, kind, limit: int = 6, tile: int = 256) -> dict:
    """Compact qualitative panels: image, GT A/B, and predicted geometry + mask A/B.

    Tiles are downscaled to `tile` pixels so the committed panels stay small.
    """

    PANELS.mkdir(parents=True, exist_ok=True)
    written = []
    runtime.model.eval()
    config = runtime.cfg["inference"]
    max_new_tokens = int(config.get("validation_max_new_tokens", config["max_new_tokens"]))
    from buildreasonseg_mvp.grounding_eval import _generate_grounded  # noqa: PLC0415

    for index, pair in enumerate(pairs[:limit]):
        sample_a = data_mod.to_sample(pair["a"])
        sample_b = data_mod.to_sample(pair["b"])
        image = sample_a.image_rgb()
        gt_a = sample_a.target_mask()
        gt_b = sample_b.target_mask()
        with torch.no_grad():
            first = _generate_grounded(runtime, sample_a, max_new_tokens, kind)
            second = _generate_grounded(runtime, sample_b, max_new_tokens, kind)
        tiles = [
            _panel_tile(image, None, f"image {sample_a.image_id}", tile),
            _panel_tile(image, gt_a, f"GT A: {sample_a.instruction_zh[:26]}", tile),
            _panel_tile(image, gt_b, f"GT B: {sample_b.instruction_zh[:26]}", tile),
            _panel_tile(
                image,
                _mask_from_logits(first["logits"], gt_a.shape) if first["valid"] else None,
                _geometry_caption("pred A", first, kind),
                tile,
            ),
            _panel_tile(
                image,
                _mask_from_logits(second["logits"], gt_b.shape) if second["valid"] else None,
                _geometry_caption("pred B", second, kind),
                tile,
            ),
        ]
        width = 3 * tiles[0].width
        height = 2 * tiles[0].height
        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        for position, tile_image in enumerate(tiles):
            canvas.paste(tile_image, ((position % 3) * tile_image.width, (position // 3) * tile_image.height))
        path = PANELS / f"pair_{index + 1:02d}_{sample_a.image_id}.png"
        canvas.save(path, optimize=True)
        written.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "bytes": path.stat().st_size,
                "image_id": sample_a.image_id,
                "instruction_a": sample_a.instruction_zh,
                "instruction_b": sample_b.instruction_zh,
                "predicted_geometry_a": first["geometry"],
                "predicted_geometry_b": second["geometry"],
                "gt_geometry_a": list(target_geometry(gt_a, kind)),
                "gt_geometry_b": list(target_geometry(gt_b, kind)),
                "emission_valid_a": first["valid"],
                "emission_valid_b": second["valid"],
            }
        )
    runtime.model.train()
    return {
        "count": len(written),
        "panels": written,
        "directory": str(PANELS.relative_to(REPO_ROOT)),
        "tile_pixels": tile,
    }


def _mask_from_logits(logits, size) -> np.ndarray:
    mask = (
        torch.nn.functional.interpolate(
            logits.detach().float().reshape(1, 1, *logits.shape[-2:]),
            size=tuple(size),
            mode="bilinear",
            align_corners=False,
        )
        .reshape(tuple(size))
        > 0.0
    )
    return mask.cpu().numpy().astype(bool)


def _panel_tile(image: np.ndarray, mask, caption: str, tile: int = 256) -> Image.Image:
    base = Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB").resize((tile, tile))
    if mask is not None:
        resized = np.asarray(
            Image.fromarray(np.asarray(mask).astype(np.uint8) * 255).resize((tile, tile), Image.NEAREST)
        ) > 127
        overlay = np.asarray(base).copy()
        overlay[resized] = (0.45 * overlay[resized] + 0.55 * np.array([255, 60, 60])).astype(np.uint8)
        base = Image.fromarray(overlay)
    canvas = Image.new("RGB", (base.width, base.height + 14), (255, 255, 255))
    canvas.paste(base, (0, 14))
    draw = ImageDraw.Draw(canvas)
    draw.text((2, 2), caption[:64], fill=(0, 0, 0))
    return canvas


def _geometry_caption(prefix: str, result: dict, kind: str) -> str:
    if not result["valid"]:
        return f"{prefix}: invalid emission"
    values = ", ".join(f"{value:.2f}" for value in result["geometry"])
    return f"{prefix} {kind}: {values}"


# ---------------------------------------------------------------- verdict


def verdict(g0: dict, paired: dict, free: dict, kind: str, leakage_clean: bool) -> dict:
    checks = {
        "g0_passed": bool(g0.get("gate", {}).get("passed")),
        "emission_rate_ge_0.90": bool((free.get("emission_rate") or 0) >= FIX_EMISSION),
        "geometry_paired_ge_14": bool((paired.get("geometry_paired_pass") or 0) >= FIX_PAIRED),
        "mask_paired_ge_14": bool((paired.get("mask_paired_pass") or 0) >= FIX_PAIRED),
        "margin_gt_0.05": bool((paired.get("mean_own_minus_cross_margin") or 0) > FIX_MARGIN),
        "strict_miou_ge_0.20": bool((free.get("strict_end_to_end_miou") or 0) >= FIX_MIOU),
        "no_gt_leakage": bool(leakage_clean),
    }
    geometry_conditioned = bool(
        (paired.get("geometry_paired_pass") or 0) >= FIX_PAIRED
        and (paired.get("mean_geometry_distance") or 0) > 0
    )
    if not leakage_clean:
        choice = "INVALID_EXPERIMENT"
        reason = "GT geometry reached an inference-time prompt"
    elif checks["g0_passed"] and all(checks.values()):
        choice = "SPATIAL_GROUNDING_FIX_FOUND"
        reason = "all section 13 requirements are met"
    elif checks["g0_passed"] and geometry_conditioned:
        choice = "SPATIAL_GROUNDING_PARTIAL"
        reason = (
            "the predicted geometry is instruction-conditioned, but the mask gate is not fully solved "
            f"(mask paired {paired.get('mask_paired_pass')}/{paired.get('paired_total')}, "
            f"margin {paired.get('mean_own_minus_cross_margin')}, "
            f"strict mIoU {free.get('strict_end_to_end_miou')})"
        )
    elif not checks["g0_passed"]:
        choice = "GROUNDING_REPRESENTATION_FAILED"
        reason = "G0 did not reach its gate: the [SEG] hidden state did not learn reliable target geometry"
    else:
        choice = "SPATIAL_GROUNDING_PARTIAL"
        reason = "G0 passed but the geometry is not yet instruction-conditioned on unseen images"
    return {"checks": checks, "verdict": choice, "reason": reason}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--targets-only", action="store_true")
    parser.add_argument("--free-limit", type=int, default=None)
    parser.add_argument("--pairs-limit", type=int, default=None)
    parser.add_argument("--panel-limit", type=int, default=6)
    parser.add_argument("--panel-tile", type=int, default=256, help="panel tile size in pixels")
    parser.add_argument("--panels-only", action="store_true", help="regenerate the panels alone")
    parser.add_argument("--skip-panels", action="store_true")
    parser.add_argument(
        "--representation-only",
        action="store_true",
        help="regenerate evaluation/task6d_representation.json alone (with the 4-condition control)",
    )
    args = parser.parse_args(argv)

    targets = build_targets_artifact()
    (EVAL / "task6d_grounding_targets.json").write_text(
        json.dumps(targets, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"[task6d:eval] targets: paired geometry distance "
        f"{targets['mean_paired_geometry_distance']:.4f}, GT mask IoU between pair members "
        f"{targets['mean_paired_mask_iou']:.4f}",
        flush=True,
    )
    if args.targets_only:
        return 0

    oracle = load_oracle()
    kind = oracle["selection"]["chosen_geometry"]
    cfg = load_config(CONFIG)
    set_seed(int(cfg["seed"]))
    runtime = build_runtime(cfg, device="cuda", verbose=True)
    if bool(cfg.get("training", {}).get("visual_feature_cache", False)):
        runtime.install_visual_cache(
            enabled=True,
            max_images=int(cfg.get("training", {}).get("visual_feature_cache_max_images", 512)),
        )
    runtime.install_grounding_head(kind)
    checkpoint_report = None
    if args.checkpoint:
        checkpoint_report = load_checkpoint(Path(args.checkpoint), runtime.model)
        print(f"[task6d:eval] loaded {args.checkpoint}: {json.dumps(checkpoint_report, default=str)}", flush=True)

    val_samples, pairs, lookup, _audit = validation_material()
    val_subset = val_samples[: args.free_limit] if args.free_limit else val_samples
    pair_subset = pairs[: args.pairs_limit] if args.pairs_limit else pairs

    if args.representation_only:
        representation = representation_diagnostics(runtime, pair_subset)
        (EVAL / "task6d_representation.json").write_text(
            json.dumps(
                {
                    "_doc": (
                        "Task 6D section 14. Same-image/different-instruction representation "
                        "diagnostics, including the four-condition same/different image x "
                        "same/different template control."
                    ),
                    "task": "6D",
                    "geometry_kind": kind,
                    "checkpoint": args.checkpoint,
                    "representation": representation,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"[task6d:eval] representation-only: hidden cos {representation['mean_hidden_cosine']:.4f} "
            f"geometry L1 {representation['mean_geometry_l1']:.4f} "
            f"control {json.dumps(representation['hidden_control'].get('cosines'), ensure_ascii=False)}",
            flush=True,
        )
        print(
            f"[task6d:eval] {representation['hidden_control'].get('statement')}",
            flush=True,
        )
        del runtime
        torch.cuda.empty_cache()
        return 0

    if args.panels_only:
        panels = write_panels(runtime, pair_subset, lookup, kind, limit=args.panel_limit, tile=args.panel_tile)
        (EVAL / "task6d_checkpoint_manifest.json").write_text(
            json.dumps(
                {
                    "_doc": (
                        "Task 6D: checkpoints are local and gitignored; this manifest records what "
                        "was produced and which checkpoint was evaluated. Regenerated by "
                        "`task6d_evaluate.py --panels-only` together with the panels."
                    ),
                    "task": "6D",
                    "geometry_kind": kind,
                    "evaluated_checkpoint": (
                        {
                            "path": args.checkpoint,
                            "sha256": sha256_file(Path(args.checkpoint)),
                            "load_report": checkpoint_report,
                        }
                        if args.checkpoint and Path(args.checkpoint).is_file()
                        else None
                    ),
                    "panels": panels,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"[task6d:eval] panels-only: {panels['count']} panels at {panels['tile_pixels']} px, "
            f"{sum(panel['bytes'] for panel in panels['panels']) / 1024:.0f} KiB total; manifest updated",
            flush=True,
        )
        del runtime
        torch.cuda.empty_cache()
        return 0

    free = free_generation_grounded(runtime, val_subset, lookup, include_component_overlap=True)
    paired = paired_probe_grounded(runtime, pair_subset, lookup, free_generation=True)
    representation = representation_diagnostics(runtime, pair_subset)
    analysis = error_analysis(runtime, val_subset, free, kind)

    g0_path = EVAL / "task6d_g0.json"
    g0 = json.loads(g0_path.read_text(encoding="utf-8")) if g0_path.is_file() else {}
    result = verdict(g0, paired, free, kind, leakage_clean=True)

    (EVAL / "task6d_paired_probe.json").write_text(
        json.dumps(
            {
                "_doc": (
                    "Task 6D section 12. Same-image paired probe under free generation: masks AND "
                    "geometry must favour the own target. GT geometry is used only to score."
                ),
                "task": "6D",
                "geometry_kind": kind,
                "checkpoint": args.checkpoint,
                "paired": paired,
                "free_generation_summary": {
                    key: value
                    for key, value in free.items()
                    if key != "run"
                },
                "verdict": result,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (EVAL / "task6d_representation.json").write_text(
        json.dumps(
            {
                "_doc": "Task 6D section 14. Same-image/different-instruction representation diagnostics.",
                "task": "6D",
                "geometry_kind": kind,
                "representation": representation,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (EVAL / "task6d_error_analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not args.skip_panels:
        panels = write_panels(
            runtime, pair_subset, lookup, kind, limit=args.panel_limit, tile=args.panel_tile
        )
        (EVAL / "task6d_checkpoint_manifest.json").write_text(
            json.dumps(
                {
                    "_doc": (
                        "Task 6D: checkpoints are local and gitignored; this manifest records what "
                        "was produced and which checkpoint was evaluated."
                    ),
                    "task": "6D",
                    "geometry_kind": kind,
                    "evaluated_checkpoint": (
                        {
                            "path": args.checkpoint,
                            "sha256": sha256_file(Path(args.checkpoint)),
                            "load_report": checkpoint_report,
                        }
                        if args.checkpoint and Path(args.checkpoint).is_file()
                        else None
                    ),
                    "panels": panels,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    print(
        f"[task6d:eval] free: emission {free['emission_valid']}/{free['count']} "
        f"strict mIoU {free['strict_end_to_end_miou']:.4f} dice {free['strict_end_to_end_dice']:.4f} "
        f"box IoU {free.get('mean_box_iou')} center_inside {free.get('center_inside_rate')}",
        flush=True,
    )
    print(
        f"[task6d:eval] paired: mask {paired['mask_paired_pass']}/{paired['paired_total']} "
        f"geometry {paired['geometry_paired_pass']}/{paired['paired_total']} "
        f"own {paired['mean_own_iou']} cross {paired['mean_cross_iou']} "
        f"margin {paired['mean_own_minus_cross_margin']} "
        f"IoU(predAB) {paired['mean_iou_pred_a_vs_b']}",
        flush=True,
    )
    print(
        f"[task6d:eval] representation: hidden cos {representation['mean_hidden_cosine']:.4f} "
        f"geometry L1 {representation['mean_geometry_l1']:.4f} mask IoU(A,B) "
        f"{representation['mean_mask_iou_a_vs_b']:.4f}",
        flush=True,
    )
    print(
        f"[task6d:eval] error analysis: failures {analysis['failures']}/{analysis['records']} "
        f"dominant {analysis['dominant_flag_among_failures']} "
        f"materially_limiting {analysis['whu_limitation_judgement']['materially_limiting']}",
        flush=True,
    )
    print(f"[task6d:eval] VERDICT: {result['verdict']} — {result['reason']}", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
