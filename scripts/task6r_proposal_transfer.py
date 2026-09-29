"""Task 6R Part I (section 19) — R1 under the frozen Task 6Q proposal reference resolver.

Diagnostic only: **no training and no gate tuning**. The frozen Task 6Q resolver cache and its exact
frozen proposal configuration are reused verbatim; the trained R1 decoder is run on the field built from
the resolved proposal reference, and the chain abstains exactly when the resolver abstains.

Writes `evaluation/task6r_proposal_reference_transfer.json`.

    python scripts/task6r_proposal_transfer.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.geometric_relation_field_v02 import geometric_relation_field_v02  # noqa: E402
from buildreasonseg_mvp.grcl_directional import hard_relation_metrics, summarise_relation_metrics  # noqa: E402
from buildreasonseg_mvp.task6m_eval import write_json  # noqa: E402
from buildreasonseg_mvp.task6n_relation_decoder import (  # noqa: E402
    FrozenFeatureStore,
    Task6NSample,
    load_frozen_sam2_encoder,
    read_pack,
)
from buildreasonseg_mvp.task6q_reference_resolver import (  # noqa: E402
    PROPOSAL_CHECKPOINT_SHA256,
    config_report,
    family_of_program,
    unpack_mask,
)
from scripts.task6n_train import FEATURE_ROOT, PACK_ROOT, MaskStore  # noqa: E402
from scripts.task6r_train import (  # noqa: E402
    FIELD_SIZE,
    TARGET_SIZE,
    model_for,
    variant_forward,
)

EVAL = REPO_ROOT / "evaluation"
CACHE_PATH = REPO_ROOT / "artifacts" / "task6q" / "resolved_references.npz"
OUT = EVAL / "task6r_proposal_reference_transfer.json"
TASK6Q_REFERENCE = {"miou": 0.3045812554881724, "paired": 10, "own_cross_margin": 0.27370032940000916}
RELATION_ORDER = ("left_of", "right_of", "above", "below")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_resolver_cache() -> dict[tuple[str, str], np.ndarray]:
    payload = np.load(CACHE_PATH, allow_pickle=True)
    keys = [str(key) for key in payload["keys"].tolist()]
    table = {}
    for index, key in enumerate(keys):
        tile_id, family = key.split("|")
        table[(tile_id, family)] = unpack_mask(payload["masks"][index])
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    started = time.time()
    if not CACHE_PATH.is_file():
        write_json(OUT, {"_doc": "Task 6R section 19.", "task": "6R",
                         "verdict": "INVALID_EXPERIMENT",
                         "reason": "frozen Task 6Q resolver cache is unavailable"})
        print("[6r.transfer] STOP: 6Q resolver cache unavailable", flush=True)
        return 2

    training = json.loads((EVAL / "task6r_training.json").read_text(encoding="utf-8"))
    r1_path = Path(training["variants"]["R1"]["checkpoint"]["path"])
    r1_sha = sha256_file(r1_path)
    model = model_for("R1", args.device)
    model.load_state_dict(torch.load(r1_path, map_location=args.device, weights_only=False)["state_dict"])
    model.eval()

    table = load_resolver_cache()
    masks = MaskStore()
    encoder, _ = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)
    samples = read_pack(PACK_ROOT / "mini_val_240.json")
    pairs = json.loads((PACK_ROOT / "paired_val_20.json").read_text(encoding="utf-8"))

    @torch.no_grad()
    def predict(sample: Task6NSample, reference: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(reference.astype(np.float32))
        field = geometric_relation_field_v02(tensor, sample.relation, FIELD_SIZE)
        visual = store.get(sample.tile_id, Path(sample.image_path)).float().unsqueeze(0).to(args.device)
        index = torch.as_tensor([RELATION_ORDER.index(sample.relation)], dtype=torch.long,
                                device=args.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device != "cpu"):
            logits = variant_forward(model, "R1", visual, index, field.to(args.device))
        probabilities = torch.sigmoid(
            F.interpolate(logits.float(), size=TARGET_SIZE, mode="bilinear", align_corners=False)
        )
        return probabilities[0, 0].cpu().numpy()

    rows = []
    abstentions = 0
    for sample in samples:
        family = family_of_program(sample.program_id)
        reference = table.get((sample.tile_id, family))
        if reference is None:
            abstentions += 1
            rows.append({"sample_id": sample.sample_id, "relation": sample.relation,
                         "reference_family": family, "abstained": True, "miou": 0.0, "dice": 0.0,
                         "empty": True, "correct": False, "margin": 0.0, "axis_violation": True})
            continue
        probability = predict(sample, reference)
        truth = np.asarray(masks.mask(sample.tile_id, sample.target_source_feature_id), dtype=bool)
        prediction = probability > 0.5
        intersection = float(np.logical_and(prediction, truth).sum())
        union = float(np.logical_or(prediction, truth).sum())
        relation = hard_relation_metrics(
            torch.as_tensor(probability)[None, None].to(args.device),
            torch.as_tensor(reference.astype(np.float32))[None, None].to(args.device),
            [sample.relation],
        )
        rows.append({
            "sample_id": sample.sample_id, "relation": sample.relation, "reference_family": family,
            "abstained": False,
            "miou": (intersection + 1e-6) / (union + 1e-6),
            "dice": (2.0 * intersection + 1e-6) / (float(prediction.sum()) + float(truth.sum()) + 1e-6),
            "empty": bool(relation["empty"][0]), "correct": bool(relation["correct"][0]),
            "margin": float(relation["margin"][0]),
            "axis_violation": bool(relation["axis_violation"][0]),
        })

    answered = [row for row in rows if not row["abstained"]]
    target = {
        "records": len(rows), "answered": len(answered), "abstentions": abstentions,
        "abstention_rate": abstentions / len(rows) if rows else None,
        "miou": float(np.mean([row["miou"] for row in answered])) if answered else None,
        "dice": float(np.mean([row["dice"] for row in answered])) if answered else None,
        "relation_accuracy": summarise_relation_metrics(answered)["relation_accuracy"] if answered else None,
        "per_relation_relation_accuracy": summarise_relation_metrics(answered)["per_relation"]
        if answered else None,
        "axis_violation_rate": summarise_relation_metrics(answered)["axis_violation_rate"]
        if answered else None,
    }

    pair_rows = []
    pair_abstentions = 0
    cache: dict[tuple[str, str], np.ndarray] = {}
    for pair in pairs["pairs"]:
        members = [Task6NSample(**{key: value for key, value in pair[side].items()
                                   if key in Task6NSample.__dataclass_fields__})
                   for side in ("a", "b")]
        families = [family_of_program(member.program_id) for member in members]
        key = (members[0].tile_id, family_of_program(members[0].program_id))
        references = [table.get((member.tile_id, family))
                      for member, family in zip(members, families)]
        if any(reference is None for reference in references):
            pair_abstentions += 1
            pair_rows.append({"abstained": True, "passes": False,
                              "a": {"own_iou": 0.0, "cross_iou": 0.0, "prefers_own": False},
                              "b": {"own_iou": 0.0, "cross_iou": 0.0, "prefers_own": False}})
            continue
        shared = references[0] if (members[0].tile_id == members[1].tile_id
                                   and members[0].reference_source_feature_id
                                   == members[1].reference_source_feature_id) else None
        own, cross = [], []
        for index, member in enumerate(members):
            reference = shared if shared is not None else references[index]
            probability = predict(member, reference)
            prediction = probability > 0.5
            own_mask = np.asarray(masks.mask(member.tile_id, member.target_source_feature_id),
                                  dtype=bool)
            own.append(float(np.logical_and(prediction, own_mask).sum())
                       / float(np.logical_or(prediction, own_mask).sum() + 1e-6))
            other = members[1 - index]
            other_mask = np.asarray(masks.mask(other.tile_id, other.target_source_feature_id),
                                    dtype=bool)
            cross.append(float(np.logical_and(prediction, other_mask).sum())
                         / float(np.logical_or(prediction, other_mask).sum() + 1e-6))
        pair_rows.append({"abstained": False, "same_resolved_reference_reused": shared is not None,
                          "a": {"own_iou": own[0], "cross_iou": cross[0],
                                "prefers_own": own[0] > cross[0]},
                          "b": {"own_iou": own[1], "cross_iou": cross[1],
                                "prefers_own": own[1] > cross[1]},
                          "passes": bool(own[0] > cross[0] and own[1] > cross[1])})
    passed = sum(1 for row in pair_rows if row["passes"])
    mean_own = float(np.mean([row["a"]["own_iou"] for row in pair_rows]
                             + [row["b"]["own_iou"] for row in pair_rows]))
    mean_cross = float(np.mean([row["a"]["cross_iou"] for row in pair_rows]
                               + [row["b"]["cross_iou"] for row in pair_rows]))
    paired = {
        "pairs": len(pair_rows), "passed": passed,
        "pass_rate": passed / len(pair_rows) if pair_rows else None,
        "mean_own_iou": mean_own, "mean_cross_iou": mean_cross,
        "own_cross_margin": mean_own - mean_cross,
        "pairs_with_reference_abstention": pair_abstentions,
        "same_resolved_reference_reused_for_pairs": True,
    }

    payload = {
        "_doc": (
            "Task 6R section 19. Diagnostic transfer of the trained R1 decoder onto the frozen Task 6Q "
            "proposal reference resolver (exact frozen proposal configuration): resolve reference -> "
            "field v0.2 -> R1 target decoder, abstaining exactly when the resolver abstains. No "
            "training and no gate tuning; this diagnostic never selects lambda."
        ),
        "task": "6R",
        "stage": "I-proposal-reference-transfer",
        "reference_source": "frozen_task6q_proposal_resolver",
        "proposal_config": config_report(),
        "proposal_checkpoint_sha256_expected": PROPOSAL_CHECKPOINT_SHA256,
        "resolver_cache": {"path": str(CACHE_PATH), "sha256": sha256_file(CACHE_PATH),
                           "entries": len(table)},
        "r1_checkpoint": {"path": str(r1_path), "sha256": r1_sha, "retrained_here": False},
        "target": target,
        "paired": paired,
        "comparison_task6q_b3_chain": {
            "miou": TASK6Q_REFERENCE["miou"], "paired": TASK6Q_REFERENCE["paired"],
            "own_cross_margin": TASK6Q_REFERENCE["own_cross_margin"],
            "delta_miou": (target["miou"] or 0.0) - TASK6Q_REFERENCE["miou"],
            "delta_paired": passed - TASK6Q_REFERENCE["paired"],
            "delta_own_cross_margin": paired["own_cross_margin"]
            - TASK6Q_REFERENCE["own_cross_margin"],
        },
        "diagnostic_only": True,
        "training_performed": False,
        "lambda_tuning": False,
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT, payload)
    print(f"[6r.transfer] R1 under 6Q resolver: mIoU {target['miou']:.6f} (6Q B3 "
          f"{TASK6Q_REFERENCE['miou']:.6f}) | paired {passed}/{len(pair_rows)} "
          f"(6Q B3 {TASK6Q_REFERENCE['paired']}/20) margin {paired['own_cross_margin']:+.6f} | "
          f"rel-acc {target['relation_accuracy']:.4f} abstentions {abstentions}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
