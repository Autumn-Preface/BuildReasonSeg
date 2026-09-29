"""Task 6O Part A + Part E — B2 reproduction and the B3/B4 evaluation.

**Part A (sections 6-7)**: verifies the frozen Task 6N B2 checkpoint (path + SHA256 from
`evaluation/task6n_mini_val.json`), then runs the frozen Task 6N evaluator on MiniVal240 and
PairedVal20 **exactly once** and requires the stored Task 6N values to be reproduced within `1e-6`
(mIoU, Dice, own/cross means) with an exact pair-pass count. Writes
`evaluation/task6o_b2_reproduction.json`.

**Part E (sections 12-13)**: MiniVal240 and PairedVal20 metrics for B3/B4 → `task6o_mini_val.json`
and `task6o_paired_val.json`.

    python scripts/task6o_evaluate.py --stage b2-reproduction
    python scripts/task6o_evaluate.py --stage variants
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.task6m_eval import write_json  # noqa: E402
from buildreasonseg_mvp.task6n_relation_decoder import (  # noqa: E402
    DecoderConfig,
    FrozenFeatureStore,
    RelationMaskDecoder,
    TASK6O_VARIANTS,
    VARIANT_LABELS,
    load_frozen_sam2_encoder,
    read_pack,
)
from scripts.task6n_evaluate import (  # noqa: E402
    breakdown,
    per_family,
    per_relation,
    target_flags,
)
from scripts.task6n_train import FEATURE_ROOT, PACK_ROOT, MaskStore, _build_batch  # noqa: E402
from scripts.task6o_train import variant_forward  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
TASK6N_PACKS = EVAL / "task6n_pack_manifest.json"
CHECKPOINT_ROOT = REPO_ROOT / "artifacts" / "task6o" / "checkpoints"
OUT_B2 = EVAL / "task6o_b2_reproduction.json"
OUT_MINI = EVAL / "task6o_mini_val.json"
OUT_PAIRED = EVAL / "task6o_paired_val.json"
TOLERANCE = 1e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_packs() -> tuple[dict, list[str]]:
    """Section 3.1: the four Task 6N packs must match the manifest byte-for-byte."""

    manifest = json.loads(TASK6N_PACKS.read_text(encoding="utf-8"))
    problems = []
    verified = {}
    for name, entry in manifest["packs"].items():
        path = Path(entry["path"])
        if not path.is_file():
            problems.append(f"{name}: missing {path}")
            continue
        digest = sha256_file(path)
        verified[name] = {"path": str(path), "expected": entry["sha256"], "actual": digest,
                          "matches": digest == entry["sha256"]}
        if digest != entry["sha256"]:
            problems.append(f"{name}: hash mismatch")
    return {"manifest": str(TASK6N_PACKS), "packs": verified}, problems


@torch.no_grad()
def detailed_rows(model, samples, store, masks, device, batch_size=8) -> list[dict]:
    """Per-sample metrics identical to the frozen Task 6N evaluator, variant-appropriate forward."""

    model.eval()
    rows = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start: start + batch_size]
        batch = _build_batch(chunk, store, masks, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            logits = variant_forward(model, batch)
        upsampled = torch.nn.functional.interpolate(
            logits.float(), size=(batch.target_size[0], batch.target_size[1]),
            mode="bilinear", align_corners=False,
        ) > 0.0
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


import numpy as np  # noqa: E402


@torch.no_grad()
def paired_rows(model, pairs_payload: dict, store, masks, device) -> dict:
    from buildreasonseg_mvp.task6n_relation_decoder import Task6NSample

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
            logits = variant_forward(model, batch)
        upsampled = torch.nn.functional.interpolate(
            logits.float(), size=(batch.target_size[0], batch.target_size[1]),
            mode="bilinear", align_corners=False,
        ) > 0.0
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
    mean_cross = float(np.mean([row["a"]["cross_iou"] for row in rows]
                               + [row["b"]["cross_iou"] for row in rows]))
    return {"pairs": len(rows), "passed": passed, "pass_rate": passed / len(rows) if rows else None,
            "mean_own_iou": mean_own, "mean_cross_iou": mean_cross,
            "own_cross_margin": mean_own - mean_cross, "rows": rows}


def close(left, right, tolerance=TOLERANCE) -> bool:
    return left is not None and right is not None and abs(float(left) - float(right)) <= tolerance


def run_b2_reproduction(args) -> int:
    started = time.time()
    packs, problems = verify_packs()
    frozen = json.loads((EVAL / "task6n_mini_val.json").read_text(encoding="utf-8"))
    stored_b2 = frozen["variants"]["B2"]
    checkpoint_path = Path(stored_b2["checkpoint"]["path"])
    expected_sha = stored_b2["checkpoint"]["sha256"]
    checkpoint_present = checkpoint_path.is_file()
    actual_sha = sha256_file(checkpoint_path) if checkpoint_present else None

    if problems or not checkpoint_present or actual_sha != expected_sha:
        payload = {
            "_doc": "Task 6O section 6. Frozen Task 6N B2 checkpoint verification.",
            "task": "6O",
            "stage": "A-b2-verification",
            "pack_verification": packs,
            "pack_problems": problems,
            "checkpoint": {"path": str(checkpoint_path), "present": checkpoint_present,
                           "expected_sha256": expected_sha, "actual_sha256": actual_sha,
                           "matches": actual_sha == expected_sha},
            "verdict": "TASK6N_PACK_MISMATCH" if problems else "TASK6N_B2_CHECKPOINT_UNAVAILABLE",
            "b2_retrained": False,
        }
        write_json(OUT_B2, payload)
        print(f"[6o.b2] STOP {payload['verdict']}", flush=True)
        return 2

    print("[6o.b2] checkpoint + packs verified; re-evaluating B2 exactly once", flush=True)
    masks = MaskStore()
    encoder, sam_report = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)
    val_samples = read_pack(PACK_ROOT / "mini_val_240.json")
    pairs_payload = json.loads((PACK_ROOT / "paired_val_20.json").read_text(encoding="utf-8"))

    from scripts.task6n_evaluate import detailed_rows as frozen_detailed_rows
    from scripts.task6n_evaluate import paired_report as frozen_paired_report

    model = RelationMaskDecoder("B2", DecoderConfig()).to(args.device)
    state = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    rows = frozen_detailed_rows(model, val_samples, store, masks, args.device)
    reproduced = breakdown(rows)
    paired = frozen_paired_report(model, pairs_payload, store, masks, args.device)

    stored_miou = stored_b2["selected_model"]["miou"]
    stored_dice = stored_b2["selected_model"]["dice"]
    stored_paired = json.loads((EVAL / "task6n_paired_val.json").read_text(encoding="utf-8"))
    stored_pair = stored_paired["variants"]["B2"]["paired_val"]

    comparisons = {
        "mini_val_miou": {"stored_task6n": stored_miou, "reproduced": reproduced["miou"],
                          "abs_delta": abs(reproduced["miou"] - stored_miou),
                          "within_tolerance": close(reproduced["miou"], stored_miou)},
        "mini_val_dice": {"stored_task6n": stored_dice, "reproduced": reproduced["dice"],
                          "abs_delta": abs(reproduced["dice"] - stored_dice),
                          "within_tolerance": close(reproduced["dice"], stored_dice)},
        "paired_pass": {"stored_task6n": stored_pair["passed"], "reproduced": paired["passed"],
                        "exact": paired["passed"] == stored_pair["passed"]},
        "paired_mean_own_iou": {"stored_task6n": stored_pair["mean_own_iou"],
                                "reproduced": paired["mean_own_iou"],
                                "abs_delta": abs(paired["mean_own_iou"] - stored_pair["mean_own_iou"]),
                                "within_tolerance": close(paired["mean_own_iou"],
                                                          stored_pair["mean_own_iou"])},
        "paired_mean_cross_iou": {"stored_task6n": stored_pair["mean_cross_iou"],
                                  "reproduced": paired["mean_cross_iou"],
                                  "abs_delta": abs(paired["mean_cross_iou"]
                                                   - stored_pair["mean_cross_iou"]),
                                  "within_tolerance": close(paired["mean_cross_iou"],
                                                            stored_pair["mean_cross_iou"])},
    }
    passed = all(
        entry.get("within_tolerance", entry.get("exact", False)) for entry in comparisons.values()
    )
    payload = {
        "_doc": (
            "Task 6O sections 6-7. The frozen Task 6N B2 checkpoint is re-evaluated exactly once with "
            "the frozen Task 6N evaluator on MiniVal240 and PairedVal20; the stored Task 6N values must "
            "be reproduced within 1e-6 and the pair-pass count exactly."
        ),
        "task": "6O",
        "stage": "A-b2-reproduction",
        "reference_source": "oracle_native_gt",
        "tolerance": TOLERANCE,
        "pack_verification": packs,
        "checkpoint": {"path": str(checkpoint_path), "present": True,
                       "expected_sha256": expected_sha, "actual_sha256": actual_sha, "matches": True},
        "b2_retrained": False,
        "reproduced": {"mini_val": {key: reproduced[key] for key in ("miou", "dice",
                                                                    "precision_at_0_5", "records")},
                       "paired_val": {key: value for key, value in paired.items() if key != "rows"}},
        "stored_task6n": {"mini_val": {"miou": stored_miou, "dice": stored_dice},
                          "paired_val": {key: value for key, value in stored_pair.items()
                                         if key != "rows"}},
        "comparisons": comparisons,
        "verdict": "B2_REPRODUCED" if passed else "TASK6N_B2_REPRODUCTION_FAIL",
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_B2, payload)
    print(f"[6o.b2] reproduced mIoU {reproduced['miou']:.6f} (stored {stored_miou:.6f}) | "
          f"paired {paired['passed']}/{paired['pairs']} -> {payload['verdict']}", flush=True)
    return 0 if passed else 2


def run_variants(args) -> int:
    started = time.time()
    packs, problems = verify_packs()
    if problems:
        print(f"[6o.eval] TASK6N_PACK_MISMATCH {problems}", flush=True)
        return 2
    training = json.loads((EVAL / "task6o_o2_training.json").read_text(encoding="utf-8"))
    val_samples = read_pack(PACK_ROOT / "mini_val_240.json")
    pairs_payload = json.loads((PACK_ROOT / "paired_val_20.json").read_text(encoding="utf-8"))

    masks = MaskStore()
    encoder, sam_report = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)

    mini_payload = {
        "_doc": (
            "Task 6O section 12. MiniVal240 metrics for the two causal-decomposition variants, using "
            "the byte-identical frozen Task 6N pack and the frozen evaluation protocol."
        ),
        "task": "6O",
        "stage": "O2-evaluation",
        "reference_source": "oracle_native_gt",
        "pack": {"path": str(PACK_ROOT / "mini_val_240.json"),
                 "sha256": packs["packs"]["mini_val_240"]["actual"]},
        "visual": store.provenance(),
        "variants": {},
        "test_split_used": False,
    }
    paired_payload = {
        "_doc": "Task 6O section 13. PairedVal20 for the two causal-decomposition variants.",
        "task": "6O",
        "stage": "O2-paired",
        "reference_source": "oracle_native_gt",
        "pair_pack": {"path": str(PACK_ROOT / "paired_val_20.json"),
                      "sha256": packs["packs"]["paired_val_20"]["actual"],
                      "pairs": len(pairs_payload["pairs"])},
        "variants": {},
        "test_split_used": False,
    }

    for variant in TASK6O_VARIANTS:
        checkpoint_path = Path(training["variants"][variant]["checkpoint"]["path"])
        model = RelationMaskDecoder(variant, DecoderConfig()).to(args.device)
        state = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        model.load_state_dict(state["state_dict"])
        rows = detailed_rows(model, val_samples, store, masks, args.device)
        paired = paired_rows(model, pairs_payload, store, masks, args.device)
        mini_payload["variants"][variant] = {
            "label": VARIANT_LABELS[variant],
            "overall": breakdown(rows),
            "per_relation": per_relation(rows),
            "per_reference_family": per_family(rows),
            "parameters": model.parameter_report(),
            "training": {
                "selected_epoch": training["variants"][variant]["best"]["epoch"],
                "epochs_run": training["variants"][variant]["epochs_run"],
                "peak_vram_gb": training["variants"][variant]["peak_vram_gb"],
                "wall_seconds": training["variants"][variant]["wall_seconds"],
                "checkpoint": training["variants"][variant]["checkpoint"],
            },
        }
        paired_payload["variants"][variant] = {
            "label": VARIANT_LABELS[variant],
            **{key: value for key, value in paired.items() if key != "rows"},
            "rows": paired["rows"],
        }
        print(f"[6o.eval] {variant}: mIoU {mini_payload['variants'][variant]['overall']['miou']:.6f} "
              f"Dice {mini_payload['variants'][variant]['overall']['dice']:.6f} | paired "
              f"{paired['passed']}/{paired['pairs']} margin {paired['own_cross_margin']:+.6f}", flush=True)

    mini_payload["runtime_seconds"] = round(time.time() - started, 1)
    paired_payload["runtime_seconds"] = round(time.time() - started, 1)
    write_json(OUT_MINI, mini_payload)
    write_json(OUT_PAIRED, paired_payload)
    print(f"[6o.eval] wrote {OUT_MINI.name} and {OUT_PAIRED.name}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("b2-reproduction", "variants"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    return run_b2_reproduction(args) if args.stage == "b2-reproduction" else run_variants(args)


if __name__ == "__main__":
    raise SystemExit(main())
