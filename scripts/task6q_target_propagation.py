"""Task 6Q Part E (sections 9-10) — downstream target propagation with the proposal reference.

For every frozen Task 6N **MiniVal240** record:

1. take the frozen YOLO26m-seg proposals (read from the gitignored resolver cache built by
   `task6q_reference_audit.py`; tiles missing from the cache — only possible for PairedVal20 members
   outside MiniVal240 — are resolved on demand with the same frozen configuration, once per tile);
2. derive the family from the canonical program (`largest` / `smallest`);
3. resolve the reference mask with the deterministic section-6 rule;
4. on abstention the target prediction is an explicit abstention;
5. otherwise generate `P_rel` with the frozen GeometricRelationField **v0.2**;
6. feed the frozen Task 6O **B3** (frozen SAM2 visual feature + `P_rel` + direction id);
7. predict the target mask.

The oracle reference mask, the target GT, candidate target masks and the target instance id are never
provided to the chain; the GT target is scoring only. The PairedVal20 stage reuses exactly the same
resolved reference mask for pairs that share the image and the reference source.

    python scripts/task6q_target_propagation.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.geometric_relation_field_v02 import geometric_relation_field_v02  # noqa: E402
from buildreasonseg_mvp.task6m_eval import write_json  # noqa: E402
from buildreasonseg_mvp.task6n_relation_decoder import (  # noqa: E402
    DecoderConfig,
    FrozenFeatureStore,
    RelationMaskDecoder,
    Task6NSample,
    load_frozen_sam2_encoder,
    read_pack,
)
from buildreasonseg_mvp.task6q_reference_resolver import (  # noqa: E402
    FAMILIES,
    PROPOSAL_CHECKPOINT,
    PROPOSAL_CHECKPOINT_SHA256,
    PROPOSAL_CONF,
    PROPOSAL_IMGSZ,
    PROPOSAL_MAX_DET,
    config_report,
    family_of_program,
    proposals_from_results,
    select_reference,
    unpack_mask,
)
from scripts.task6n_evaluate import breakdown, per_family, per_relation, target_flags  # noqa: E402
from scripts.task6n_train import FEATURE_ROOT, PACK_ROOT, MaskStore  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
CACHE_PATH = REPO_ROOT / "artifacts" / "task6q" / "resolved_references.npz"
OUT_TARGET = EVAL / "task6q_target_val.json"
OUT_PAIRED = EVAL / "task6q_target_paired_val.json"
FIELD_SIZE = (64, 64)
TARGET_SIZE = (512, 512)
ORACLE_B3_MIOU = 0.4299680351479113
DENSE_PREDICTED_MIOU = 0.24096754293919803
ORACLE_B3_PAIRED = 14
ORACLE_B3_MARGIN = 0.39719566349802166
DENSE_PAIRED = 0
DENSE_MARGIN = 0.0036186017082471683


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cache() -> tuple[dict[tuple[str, str], np.ndarray], dict]:
    payload = np.load(CACHE_PATH, allow_pickle=True)
    keys = [str(key) for key in payload["keys"].tolist()]
    masks = payload["masks"]
    table: dict[tuple[str, str], np.ndarray] = {}
    for index, key in enumerate(keys):
        tile_id, family = key.split("|")
        table[(tile_id, family)] = unpack_mask(masks[index])
    meta = {
        "path": str(CACHE_PATH),
        "sha256": sha256_file(CACHE_PATH),
        "entries": len(keys),
        "checkpoint_sha256": str(payload["checkpoint_sha256"]),
        "conf": float(payload["conf"]),
        "imgsz": int(payload["imgsz"]),
        "max_det": int(payload["max_det"]),
    }
    return table, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--proposal-device", default="0")
    args = parser.parse_args(argv)

    started = time.time()
    checkpoint = REPO_ROOT / PROPOSAL_CHECKPOINT
    if not checkpoint.is_file() or sha256_file(checkpoint) != PROPOSAL_CHECKPOINT_SHA256:
        write_json(OUT_TARGET, {"_doc": "Task 6Q section 9.", "task": "6Q",
                                "verdict": "PROPOSAL_CHECKPOINT_UNAVAILABLE"})
        print("[6q.chain] STOP PROPOSAL_CHECKPOINT_UNAVAILABLE", flush=True)
        return 2

    table, cache_meta = load_cache()
    if (cache_meta["checkpoint_sha256"] != PROPOSAL_CHECKPOINT_SHA256
            or cache_meta["conf"] != PROPOSAL_CONF or cache_meta["imgsz"] != PROPOSAL_IMGSZ
            or cache_meta["max_det"] != PROPOSAL_MAX_DET):
        write_json(OUT_TARGET, {"_doc": "Task 6Q section 9.", "task": "6Q",
                                "verdict": "INVALID_EXPERIMENT",
                                "reason": "resolver cache was built with a different configuration"})
        print("[6q.chain] STOP INVALID_EXPERIMENT (cache/config mismatch)", flush=True)
        return 2

    samples = read_pack(PACK_ROOT / "mini_val_240.json")
    pairs = json.loads((PACK_ROOT / "paired_val_20.json").read_text(encoding="utf-8"))

    # ---------------- resolve any missing (tile, family) keys on demand, once per tile
    needed: set[tuple[str, str]] = {(sample.tile_id, family_of_program(sample.program_id))
                                    for sample in samples}
    for pair in pairs["pairs"]:
        for side in ("a", "b"):
            record = pair[side]
            needed.add((record["tile_id"], family_of_program(record["program_id"])))
    missing = sorted(key for key in needed if key not in table)
    tiles_added = 0
    if missing:
        from ultralytics import YOLO

        model = YOLO(str(checkpoint))
        tiles = sorted({tile_id for tile_id, _ in missing})
        image_paths = {record["tile_id"]: record["image_path"]
                       for record in (json.loads((PACK_ROOT / "mini_val_240.json").read_text(
                           encoding="utf-8"))["records"])}
        paired_image_paths = {pair[side]["tile_id"]: pair[side]["image_path"]
                              for pair in pairs["pairs"] for side in ("a", "b")}
        image_paths.update(paired_image_paths)
        for tile_id in tiles:
            results = model.predict(source=str(image_paths[tile_id]), imgsz=PROPOSAL_IMGSZ,
                                    conf=PROPOSAL_CONF, max_det=PROPOSAL_MAX_DET, verbose=False,
                                    device=args.proposal_device, retina_masks=True)[0]
            proposals = proposals_from_results(results)
            for family in FAMILIES:
                selection = select_reference(proposals, family)
                if selection.mask is not None:
                    table[(tile_id, family)] = selection.mask
            tiles_added += 1
        print(f"[6q.chain] resolved {tiles_added} extra tiles on demand for PairedVal20", flush=True)

    masks = MaskStore()
    encoder, _ = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)

    b3_path = Path(json.loads((EVAL / "task6o_mini_val.json").read_text(encoding="utf-8"))
                   ["variants"]["B3"]["training"]["checkpoint"]["path"])
    b3_expected = json.loads((EVAL / "task6o_mini_val.json").read_text(encoding="utf-8"))[
        "variants"]["B3"]["training"]["checkpoint"]["sha256"]
    b3_sha = sha256_file(b3_path) if b3_path.is_file() else None
    if b3_sha != b3_expected:
        write_json(OUT_TARGET, {"_doc": "Task 6Q section 9.", "task": "6Q",
                                "verdict": "INVALID_EXPERIMENT",
                                "reason": "frozen Task 6O B3 checkpoint hash mismatch"})
        print("[6q.chain] STOP INVALID_EXPERIMENT (B3 hash)", flush=True)
        return 2
    b3 = RelationMaskDecoder("B3", DecoderConfig()).to(args.device)
    b3.load_state_dict(torch.load(b3_path, map_location=args.device, weights_only=False)["state_dict"])
    b3.eval()

    @torch.no_grad()
    def predict_target(reference_mask: np.ndarray, sample: Task6NSample) -> np.ndarray:
        """field v0.2 from the resolved reference -> frozen B3 -> 512x512 target mask."""

        tensor = torch.as_tensor(reference_mask.astype(np.float32))
        field = geometric_relation_field_v02(tensor, sample.relation, FIELD_SIZE)
        visual = store.get(sample.tile_id, Path(sample.image_path)).float().unsqueeze(0).to(args.device)
        index = torch.as_tensor([("left_of", "right_of", "above", "below").index(sample.relation)],
                                dtype=torch.long, device=args.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device != "cpu"):
            logits = b3(visual, index, None, field.to(args.device))
        upsampled = F.interpolate(logits.float(), size=TARGET_SIZE, mode="bilinear",
                                  align_corners=False)
        return (upsampled[0, 0] > 0.0).cpu().numpy()

    # ---------------- MiniVal240
    rows = []
    abstentions = 0
    for sample in samples:
        family = family_of_program(sample.program_id)
        reference = table.get((sample.tile_id, family))
        if reference is None:
            abstentions += 1
            rows.append({"sample_id": sample.sample_id, "program_id": sample.program_id,
                         "relation": sample.relation, "reference_family": family,
                         "abstained": True, "miou": 0.0, "dice": 0.0, "precision_at_0_5": 0.0,
                         "target_area_px": 0.0, "target_touches_border": False, "target_tiny": False})
            continue
        prediction = predict_target(reference, sample)
        truth = np.asarray(masks.mask(sample.tile_id, sample.target_source_feature_id), dtype=bool)
        intersection = float(np.logical_and(prediction, truth).sum())
        union = float(np.logical_or(prediction, truth).sum())
        predicted = float(prediction.sum())
        flags = target_flags(masks, sample)
        rows.append({"sample_id": sample.sample_id, "program_id": sample.program_id,
                     "relation": sample.relation, "reference_family": family, "abstained": False,
                     "miou": (intersection + 1e-6) / (union + 1e-6),
                     "dice": (2.0 * intersection + 1e-6) / (predicted + float(truth.sum()) + 1e-6),
                     "precision_at_0_5": (intersection + 1e-6) / (predicted + 1e-6),
                     **flags})

    answered = [row for row in rows if not row["abstained"]]
    target_breakdown = breakdown(answered) if answered else {"records": 0}
    gate = {
        "section_12_passed": None, "target_miou_min": 0.3009776246035379,
        "paired_pass_min": 10, "own_cross_margin_min": 0.20,
        "measured_target_miou": target_breakdown.get("miou"),
        "measured_target_abstention_rate": abstentions / len(samples) if samples else None,
    }
    target_payload = {
        "_doc": (
            "Task 6Q section 9. Downstream target propagation through the deterministic frozen-proposal "
            "reference resolver: proposals -> family eligibility -> deterministic largest/smallest "
            "selection -> GeometricRelationField v0.2 -> frozen Task 6O B3 -> target mask. The oracle "
            "reference mask, target GT, candidate target masks and target instance id are never inputs; "
            "the GT target is scoring only."
        ),
        "task": "6Q",
        "stage": "E-target-propagation",
        "reference_source": "frozen_proposal_resolver",
        "proposal_config": config_report(),
        "checkpoint": {"path": str(checkpoint), "expected_sha256": PROPOSAL_CHECKPOINT_SHA256,
                       "matches": True},
        "resolver_cache": {**cache_meta, "tiles_resolved_on_demand": tiles_added},
        "b3": {"path": str(b3_path), "sha256": b3_sha, "matches": b3_sha == b3_expected,
               "retrained": False},
        "field": "GeometricRelationField v0.2 (unchanged)",
        "pack": {"name": "MiniVal240", "count": len(samples)},
        "target": {
            "overall": target_breakdown,
            "answered_records": len(answered),
            "abstentions": abstentions,
            "abstention_rate": abstentions / len(samples) if samples else None,
            "per_relation": per_relation(answered) if answered else {},
            "per_reference_family": per_family(answered) if answered else {},
        },
        "comparison": {
            "task6o_oracle_b3_miou": ORACLE_B3_MIOU,
            "task6p_dense_predicted_reference_miou": DENSE_PREDICTED_MIOU,
            "delta_vs_oracle": (target_breakdown.get("miou") or 0.0) - ORACLE_B3_MIOU,
            "delta_vs_dense_predicted": (target_breakdown.get("miou") or 0.0) - DENSE_PREDICTED_MIOU,
        },
        "gate": gate,
        "records": rows,
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }

    # ---------------- PairedVal20 (same resolved reference for same image/reference)
    pair_rows = []
    pair_abstentions = 0
    for pair in pairs["pairs"]:
        members = [
            Task6NSample(**{key: value for key, value in pair[side].items()
                            if key in Task6NSample.__dataclass_fields__})
            for side in ("a", "b")
        ]
        families = [family_of_program(member.program_id) for member in members]
        references = [table.get((member.tile_id, family)) for member, family in zip(members, families)]
        if any(reference is None for reference in references):
            pair_abstentions += 1
            pair_rows.append({"tile_id": members[0].tile_id,
                              "reference_source_feature_id": members[0].reference_source_feature_id,
                              "abstained": True, "passes": False,
                              "a": {"sample_id": members[0].sample_id, "relation": members[0].relation,
                                    "own_iou": 0.0, "cross_iou": 0.0, "prefers_own": False},
                              "b": {"sample_id": members[1].sample_id, "relation": members[1].relation,
                                    "own_iou": 0.0, "cross_iou": 0.0, "prefers_own": False}})
            continue
        # same image and same reference source -> exactly the same resolved mask is reused
        shared = references[0] if (members[0].tile_id == members[1].tile_id
                                   and members[0].reference_source_feature_id
                                   == members[1].reference_source_feature_id) else None
        predictions = [predict_target(shared if shared is not None else reference, member)
                       for member, reference in zip(members, references)]
        own, cross = [], []
        for index, member in enumerate(members):
            own_mask = np.asarray(masks.mask(member.tile_id, member.target_source_feature_id),
                                  dtype=bool)
            own.append(float(np.logical_and(predictions[index], own_mask).sum())
                       / float(np.logical_or(predictions[index], own_mask).sum() + 1e-6))
            other = members[1 - index]
            other_mask = np.asarray(masks.mask(other.tile_id, other.target_source_feature_id),
                                    dtype=bool)
            cross.append(float(np.logical_and(predictions[index], other_mask).sum())
                         / float(np.logical_or(predictions[index], other_mask).sum() + 1e-6))
        pair_rows.append(
            {
                "tile_id": members[0].tile_id,
                "reference_source_feature_id": members[0].reference_source_feature_id,
                "abstained": False,
                "same_resolved_reference_reused": shared is not None,
                "a": {"sample_id": members[0].sample_id, "relation": members[0].relation,
                      "own_iou": own[0], "cross_iou": cross[0], "prefers_own": own[0] > cross[0]},
                "b": {"sample_id": members[1].sample_id, "relation": members[1].relation,
                      "own_iou": own[1], "cross_iou": cross[1], "prefers_own": own[1] > cross[1]},
                "passes": bool(own[0] > cross[0] and own[1] > cross[1]),
            }
        )
    passed = sum(1 for row in pair_rows if row["passes"])
    mean_own = float(np.mean([row["a"]["own_iou"] for row in pair_rows]
                             + [row["b"]["own_iou"] for row in pair_rows]))
    mean_cross = float(np.mean([row["a"]["cross_iou"] for row in pair_rows]
                               + [row["b"]["cross_iou"] for row in pair_rows]))
    margin = mean_own - mean_cross
    paired_payload = {
        "_doc": (
            "Task 6Q section 10. PairedVal20 with the frozen-proposal resolver: for a pair sharing the "
            "image and the reference source exactly the same resolved proposal reference mask is "
            "reused and only the relation changes. A pair passes only if both members prefer their own "
            "GT target over the paired alternative by IoU."
        ),
        "task": "6Q",
        "stage": "E-paired",
        "reference_source": "frozen_proposal_resolver",
        "pairs": len(pair_rows),
        "passed": passed,
        "pass_rate": passed / len(pair_rows) if pair_rows else None,
        "mean_own_iou": mean_own,
        "mean_cross_iou": mean_cross,
        "own_cross_margin": margin,
        "pairs_with_reference_abstention": pair_abstentions,
        "same_resolved_reference_reused_for_pairs": True,
        "comparison": {
            "task6o_oracle_b3_paired_pass": ORACLE_B3_PAIRED,
            "task6o_oracle_b3_margin": ORACLE_B3_MARGIN,
            "task6p_dense_predicted_paired_pass": DENSE_PAIRED,
            "task6p_dense_predicted_margin": DENSE_MARGIN,
        },
        "gate": {"paired_pass_min": 10, "own_cross_margin_min": 0.20,
                 "measured_paired_pass": passed, "measured_margin": margin},
        "rows": pair_rows,
        "test_split_used": False,
    }
    target_payload["gate"]["measured_paired_pass"] = passed
    target_payload["gate"]["measured_own_cross_margin"] = margin
    write_json(OUT_TARGET, target_payload)
    write_json(OUT_PAIRED, paired_payload)

    print(f"[6q.chain] target mIoU {target_breakdown.get('miou'):.6f} Dice "
          f"{target_breakdown.get('dice'):.6f} abstention rate "
          f"{target_payload['target']['abstention_rate']:.6f}", flush=True)
    print(f"[6q.chain] paired {passed}/{len(pair_rows)} margin {margin:+.6f} "
          f"(reference abstentions {pair_abstentions})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
