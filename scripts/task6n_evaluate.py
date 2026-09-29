"""Task 6N section 16 — final evaluation of the selected N2 models.

Two jobs for each variant:

1. **MiniVal240 breakdowns** including the ones the training loop does not emit: per-relation mIoU,
   largest-ref vs smallest-ref, border-target mIoU and tiny-target mIoU. The border/tiny flags are
   derived from the canonical native instances at evaluation time (the frozen packs store the oracle
   identity, not the derived flags), so the packs stay byte-identical.
2. **PairedVal20** — both relations are run for every pair with the same image and the same oracle
   reference; a pair passes only if *both* members prefer their own GT target over the paired
   alternative by IoU.

Writes `evaluation/task6n_paired_val.json`. Peak VRAM and wall time are carried over from the N2
training artifact.

    python scripts/task6n_evaluate.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.task6m_eval import canonical_instances, write_json  # noqa: E402
from buildreasonseg_mvp.task6n_relation_decoder import (  # noqa: E402
    DecoderConfig,
    FrozenFeatureStore,
    RelationMaskDecoder,
    Task6NSample,
    VARIANTS,
    VARIANT_LABELS,
    load_frozen_sam2_encoder,
    read_pack,
    upsampled_logits,
)

EVAL = REPO_ROOT / "evaluation"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6n" / "packs"
CHECKPOINT_ROOT = REPO_ROOT / "artifacts" / "task6n" / "checkpoints"
FEATURE_ROOT = REPO_ROOT / "artifacts" / "task6n" / "features"
OUT = EVAL / "task6n_paired_val.json"
SUMMARY = EVAL / "task6n_ablation_summary.json"
FEATURE_SIZE = (64, 64)

from scripts.task6n_train import MaskStore, _build_batch, evaluate  # noqa: E402


def target_flags(masks: MaskStore, sample: Task6NSample) -> dict:
    """Derived target properties, read from the canonical native store (not stored in the pack)."""

    instances = masks.instances(sample.tile_id)
    for instance in instances:
        if instance.source_feature_id == sample.target_source_feature_id:
            return {
                "target_area_px": int(instance.area_px),
                "target_touches_border": bool(instance.touches_border),
                "target_tiny": bool(instance.tiny),
            }
    return {"target_area_px": 0, "target_touches_border": False, "target_tiny": False}


def breakdown(rows: list[dict]) -> dict:
    def mean(subset, key):
        return float(np.mean([row[key] for row in subset])) if subset else None

    return {
        "records": len(rows),
        "miou": mean(rows, "miou"),
        "dice": mean(rows, "dice"),
        "precision_at_0_5": mean(rows, "precision_at_0_5"),
        "border_target": {
            "records": sum(1 for row in rows if row["target_touches_border"]),
            "miou": mean([row for row in rows if row["target_touches_border"]], "miou"),
        },
        "tiny_target": {
            "records": sum(1 for row in rows if row["target_tiny"]),
            "miou": mean([row for row in rows if row["target_tiny"]], "miou"),
        },
    }


def per_relation(rows: list[dict]) -> dict:
    out = {}
    for relation in ("left_of", "right_of", "above", "below"):
        subset = [row for row in rows if row["program_id"].endswith(f"to_{relation}")]
        out[relation] = {
            "records": len(subset),
            "miou": float(np.mean([row["miou"] for row in subset])) if subset else None,
            "dice": float(np.mean([row["dice"] for row in subset])) if subset else None,
        }
    return out


def per_family(rows: list[dict]) -> dict:
    out = {}
    for family in ("largest", "smallest"):
        subset = [row for row in rows if row["program_id"].startswith(family)]
        out[family] = {
            "records": len(subset),
            "miou": float(np.mean([row["miou"] for row in subset])) if subset else None,
            "dice": float(np.mean([row["dice"] for row in subset])) if subset else None,
        }
    return out


@torch.no_grad()
def detailed_rows(model, samples, store, masks, device, batch_size=8) -> list[dict]:
    model.eval()
    rows = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start: start + batch_size]
        batch = _build_batch(chunk, store, masks, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            logits = model(batch.visual, batch.relation_index, batch.mask_ref_down, batch.field)
        upsampled = upsampled_logits(logits, batch.target_size) > 0.0
        target = batch.target > 0.5
        if target.dim() == 3:
            target = target.unsqueeze(1)
        for index, sample in enumerate(chunk):
            prediction = upsampled[index]
            gt = target[index]
            intersection = float((prediction & gt).sum())
            union = float((prediction | gt).sum())
            predicted = float(prediction.sum())
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "program_id": sample.program_id,
                    "relation": sample.relation,
                    "reference_family": sample.program_id.split("_", 1)[0],
                    "miou": (intersection + 1e-6) / (union + 1e-6),
                    "dice": (2.0 * intersection + 1e-6) / (predicted + float(gt.sum()) + 1e-6),
                    "precision_at_0_5": (intersection + 1e-6) / (predicted + 1e-6),
                    **target_flags(masks, sample),
                }
            )
    return rows


@torch.no_grad()
def paired_report(model, pairs_payload: dict, store, masks, device) -> dict:
    """PairedVal20: same image, same oracle reference, different direction and target."""

    model.eval()
    rows = []
    for pair in pairs_payload["pairs"]:
        members = [
            Task6NSample(**{key: value for key, value in pair["a"].items()
                            if key in Task6NSample.__dataclass_fields__}),
            Task6NSample(**{key: value for key, value in pair["b"].items()
                            if key in Task6NSample.__dataclass_fields__}),
        ]
        batch = _build_batch(members, store, masks, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            logits = model(batch.visual, batch.relation_index, batch.mask_ref_down, batch.field)
        upsampled = upsampled_logits(logits, batch.target_size) > 0.0
        own, cross = [], []
        for index in range(len(members)):
            own_target = batch.target[index] > 0.5
            own.append(float((upsampled[index] & own_target).sum())
                       / float((upsampled[index] | own_target).sum() + 1e-6))
            other = members[1 - index]
            other_mask = torch.as_tensor(
                masks.mask(other.tile_id, other.target_source_feature_id), dtype=torch.bool,
                device=upsampled.device,
            )
            cross.append(float((upsampled[index] & other_mask).sum())
                         / float((upsampled[index] | other_mask).sum() + 1e-6))
        rows.append(
            {
                "tile_id": pair["tile_id"],
                "reference_source_feature_id": pair["reference_source_feature_id"],
                "a": {"sample_id": members[0].sample_id, "relation": members[0].relation,
                      "own_iou": own[0], "cross_iou": cross[0], "prefers_own": own[0] > cross[0]},
                "b": {"sample_id": members[1].sample_id, "relation": members[1].relation,
                      "own_iou": own[1], "cross_iou": cross[1], "prefers_own": own[1] > cross[1]},
                "passes": bool(own[0] > cross[0] and own[1] > cross[1]),
            }
        )
    passed = sum(1 for row in rows if row["passes"])
    mean_own = float(np.mean([row["a"]["own_iou"] for row in rows] + [row["b"]["own_iou"] for row in rows]))
    mean_cross = float(np.mean([row["a"]["cross_iou"] for row in rows] + [row["b"]["cross_iou"] for row in rows]))
    return {
        "pairs": len(rows),
        "passed": passed,
        "pass_rate": passed / len(rows) if rows else None,
        "mean_own_iou": mean_own,
        "mean_cross_iou": mean_cross,
        "own_cross_margin": mean_own - mean_cross,
        "pair_pass_rule": "both members prefer their own GT target over the paired alternative by IoU",
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    started = time.time()
    val_samples = read_pack(PACK_ROOT / "mini_val_240.json")
    pairs_payload = json.loads((PACK_ROOT / "paired_val_20.json").read_text(encoding="utf-8"))
    mini_val_artifact = json.loads((EVAL / "task6n_mini_val.json").read_text(encoding="utf-8"))

    masks = MaskStore()
    encoder, sam_report = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)

    variant_reports = {}
    detailed = {}
    for variant in VARIANTS:
        checkpoint_path = CHECKPOINT_ROOT / f"n2_{variant}.pt"
        model = RelationMaskDecoder(variant, DecoderConfig()).to(args.device)
        payload = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        rows = detailed_rows(model, val_samples, store, masks, args.device)
        detailed[variant] = {
            "label": VARIANT_LABELS[variant],
            "overall": breakdown(rows),
            "per_relation": per_relation(rows),
            "per_reference_family": per_family(rows),
            "parameters": model.parameter_report(),
        }
        variant_reports[variant] = {
            "label": VARIANT_LABELS[variant],
            "paired_val": paired_report(model, pairs_payload, store, masks, args.device),
            "checkpoint": str(checkpoint_path),
            "best_epoch": payload.get("best_epoch"),
        }
        report = variant_reports[variant]
        print(
            f"[6n.eval] {variant}: paired {report['paired_val']['passed']}/{report['paired_val']['pairs']} "
            f"| own {report['paired_val']['mean_own_iou']:.4f} cross "
            f"{report['paired_val']['mean_cross_iou']:.4f} margin "
            f"{report['paired_val']['own_cross_margin']:.4f}",
            flush=True,
        )

    payload = {
        "_doc": (
            "Task 6N section 16. PairedVal20 evaluation for the three controlled variants: both "
            "relations of every pair are run with the same image and the same oracle reference mask "
            "(reference_source = oracle_native_gt); a pair passes only if both members prefer their "
            "own GT target over the paired alternative by IoU."
        ),
        "task": "6N",
        "stage": "N2-paired",
        "reference_source": "oracle_native_gt",
        "pair_pack": {
            "path": str(PACK_ROOT / "paired_val_20.json"),
            "pairs": len(pairs_payload["pairs"]),
            "detail": pairs_payload.get("detail"),
        },
        "variants": variant_reports,
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(Path(args.out), payload)

    # merge the detailed MiniVal breakdowns into the ablation summary (§16 completeness)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.is_file() else {
        "task": "6N",
        "reference_source": "oracle_native_gt",
        "_doc": "Task 6N sections 16-17. Ablation summary for the three controlled variants.",
    }
    summary["mini_val_detailed"] = detailed
    summary["paired_val"] = {
        variant: {key: value for key, value in report["paired_val"].items() if key != "rows"}
        for variant, report in variant_reports.items()
    }
    summary["n2_training"] = {
        variant: {
            "selected_model": mini_val_artifact["variants"][variant]["selected_model"],
            "best": mini_val_artifact["variants"][variant]["best"],
            "epochs_run": mini_val_artifact["variants"][variant]["epochs_run"],
            "parameters": mini_val_artifact["variants"][variant]["parameters"],
            "peak_vram_gb": mini_val_artifact["variants"][variant]["peak_vram_gb"],
            "wall_seconds": mini_val_artifact["variants"][variant]["wall_seconds"],
            "checkpoint": mini_val_artifact["variants"][variant]["checkpoint"],
        }
        for variant in VARIANTS
    }
    summary["visual"] = mini_val_artifact.get("visual")
    summary["comparisons"] = mini_val_artifact.get("comparisons")
    write_json(SUMMARY, summary)
    print(f"[6n.eval] wrote {Path(args.out).name} and refreshed {SUMMARY.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
