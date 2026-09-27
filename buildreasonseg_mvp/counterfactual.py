"""Task 6H: canonical same-image counterfactual pairs and the own-vs-cross region-ranking loss.

Task 6G's dense head localizes nothing because per-sample BCE+Dice never has to decide *which*
building an instruction selects. Task 6H keeps the architecture identical and changes only the
training semantics: one optimizer step consumes one canonical pair from the Task 6C `P` subset
(same image, different instruction, different target) and adds an explicit ranking loss

    L_cf = 0.5 * ( softplus(margin - (s_AA - s_AB)) + softplus(margin - (s_BB - s_BA)) )
    s_XY   = region_score(H_X, M_Y) = sum(H_X * M_Y) / max(sum(M_Y), eps)

so heatmap A must score target A above target B and heatmap B must score target B above target A.
Nothing else changes: no extra query token, no cross-attention, no coordinate channels, no new
vocabulary, no SAM2 training.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import torch

#: Section 8 fixes the margin; it is a constant, not a hyperparameter to sweep.
CF_MARGIN = 1.0

#: Section 9 fixes the weights; no sweep.
PAIR_REASONING_WEIGHT = 0.5
PAIR_HEATMAP_WEIGHT = 1.0
PAIR_CF_WEIGHT = 2.0


# ------------------------------------------------------------------ canonical pairs


@dataclass(frozen=True)
class CounterfactualPair:
    """One canonical same-image / different-instruction / different-target pair."""

    index: int
    image_id: str
    a: str
    b: str
    target_a: int
    target_b: int

    def key(self) -> str:
        return "|".join([self.image_id, self.a, self.b, str(self.target_a), str(self.target_b)])

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "image_id": self.image_id,
            "a": self.a,
            "b": self.b,
            "target_a": self.target_a,
            "target_b": self.target_b,
            "key": self.key(),
        }


def build_canonical_pairs(payload: dict, records_by_id: dict) -> list[tuple[CounterfactualPair, dict, dict]]:
    """Task 6H section 4: the 240 canonical pairs, with every assertion enforced.

    Order is the recorded Task 6C `P` payload order, so it is deterministic and auditable.
    Returns `(pair, record_a, record_b)` triples.
    """

    triples: list[tuple[CounterfactualPair, dict, dict]] = []
    for index, raw in enumerate(payload["P"]["pairs"]):
        record_a = records_by_id[raw["a"]]
        record_b = records_by_id[raw["b"]]
        pair = CounterfactualPair(
            index=index,
            image_id=str(raw["image_id"]),
            a=str(raw["a"]),
            b=str(raw["b"]),
            target_a=int(record_a["target_component_id"]),
            target_b=int(record_b["target_component_id"]),
        )
        if str(record_a["image_id"]) != str(record_b["image_id"]):
            raise RuntimeError(f"pair {index}: A and B do not share a source image")
        if str(record_a["image_id"]) != pair.image_id:
            raise RuntimeError(f"pair {index}: recorded image_id disagrees with the records")
        if pair.a == pair.b:
            raise RuntimeError(f"pair {index}: identical sample ids")
        if pair.target_a == pair.target_b:
            raise RuntimeError(f"pair {index}: identical target component ids")
        triples.append((pair, record_a, record_b))

    seen: set[str] = set()
    for pair, _a, _b in triples:
        for sample_id in (pair.a, pair.b):
            if sample_id in seen:
                raise RuntimeError(f"sample {sample_id} appears in more than one canonical pair")
            seen.add(sample_id)
    return triples


def mask_overlap_diagnostics(mask_a: np.ndarray, mask_b: np.ndarray, grid: int) -> dict:
    """Section 7: record (never discard) the downsampled mask overlap of a pair."""

    left = np.asarray(mask_a).astype(bool)
    right = np.asarray(mask_b).astype(bool)
    intersection = float(np.logical_and(left, right).sum())
    union = float(np.logical_or(left, right).sum())
    return {
        "same_mask": bool(np.array_equal(left, right)),
        "raw_iou": float(intersection / union) if union else 0.0,
        "raw_intersection_pixels": int(intersection),
        "area_a": int(left.sum()),
        "area_b": int(right.sum()),
        "overlapping": bool(intersection > 0),
        "grid": int(grid),
    }


def pair_manifest(triples: list[tuple[CounterfactualPair, dict, dict]], grid: int,
                  overlap_rows: list[dict] | None = None) -> dict:
    """Machine-readable pair manifest plus a canonical hash over the pair identities."""

    rows = []
    for position, (pair, record_a, record_b) in enumerate(triples):
        row = pair.as_dict()
        row.update(
            {
                "a_level": int(record_a["level"]),
                "b_level": int(record_b["level"]),
                "a_query_type": str(record_a["query_type"]),
                "b_query_type": str(record_b["query_type"]),
            }
        )
        if overlap_rows is not None:
            row["mask_overlap"] = overlap_rows[position]
        rows.append(row)

    canonical = json.dumps(
        [[row["image_id"], row["a"], row["b"], row["target_a"], row["target_b"]] for row in rows],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "_doc": (
            "Task 6H section 4. Canonical counterfactual pairs from the Task 6C `P` subset: same "
            "source image, different sample ids, different target component ids, different masks, "
            "deterministic (payload) order. This manifest is the audit trail for the pair step."
        ),
        "task": "6H",
        "pair_count": len(rows),
        "image_count": len({row["image_id"] for row in rows}),
        "grid": int(grid),
        "canonical_order": "Task 6C P payload order (evaluation/task6c_subset_ids.json)",
        "pair_identity_sha256": digest,
        "test_split_used": False,
        "assertions": {
            "same_source_image": True,
            "different_sample_ids": True,
            "different_target_component_ids": True,
            "target_masks_differ": all(
                not row["mask_overlap"]["same_mask"] for row in rows if "mask_overlap" in row
            )
            if rows and "mask_overlap" in rows[0]
            else None,
            "deterministic_order": True,
        },
        "overlap_summary": (
            None
            if not rows or "mask_overlap" not in rows[0]
            else {
                "pairs_with_overlapping_targets": sum(
                    1 for row in rows if row["mask_overlap"]["overlapping"]
                ),
                "mean_raw_iou": float(
                    sum(row["mask_overlap"]["raw_iou"] for row in rows) / len(rows)
                ),
                "max_raw_iou": float(max(row["mask_overlap"]["raw_iou"] for row in rows)),
            }
        ),
        "pairs": rows,
    }


# ------------------------------------------------------------------ region scores and loss


def region_score(logits: torch.Tensor, soft_mask: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """`sum(H * M) / max(sum(M), eps)` — the mean heatmap logit over a target region."""

    mask = soft_mask.float()
    return (logits.float() * mask).sum() / torch.clamp(mask.sum(), min=epsilon)


def probability_region_score(logits: torch.Tensor, soft_mask: torch.Tensor, epsilon: float = 1e-6) -> float:
    """Diagnostic: the same region score in probability space (bounded in [0,1])."""

    probabilities = torch.sigmoid(logits.detach().float())
    mask = soft_mask.detach().float()
    return float((probabilities * mask).sum() / torch.clamp(mask.sum(), min=epsilon))


def region_scores(logits_a: torch.Tensor, logits_b: torch.Tensor,
                  mask_a: torch.Tensor, mask_b: torch.Tensor) -> dict:
    """The four own/cross region scores of a pair (section 7)."""

    s_aa = region_score(logits_a, mask_a)
    s_ab = region_score(logits_a, mask_b)
    s_bb = region_score(logits_b, mask_b)
    s_ba = region_score(logits_b, mask_a)
    return {
        "s_aa": s_aa,
        "s_ab": s_ab,
        "s_bb": s_bb,
        "s_ba": s_ba,
        "margin_a": s_aa - s_ab,
        "margin_b": s_bb - s_ba,
        "mean_margin": 0.5 * ((s_aa - s_ab) + (s_bb - s_ba)),
        "raw": {
            "s_aa": float(s_aa.detach()),
            "s_ab": float(s_ab.detach()),
            "s_bb": float(s_bb.detach()),
            "s_ba": float(s_ba.detach()),
            "margin_a": float((s_aa - s_ab).detach()),
            "margin_b": float((s_bb - s_ba).detach()),
            "mean_margin": float((0.5 * ((s_aa - s_ab) + (s_bb - s_ba))).detach()),
            "probability_margin_a": probability_region_score(logits_a, mask_a)
            - probability_region_score(logits_a, mask_b),
            "probability_margin_b": probability_region_score(logits_b, mask_b)
            - probability_region_score(logits_b, mask_a),
        },
    }


def counterfactual_loss(scores: dict, margin: float = CF_MARGIN) -> dict:
    """Section 8: `L_cf = 0.5 * (softplus(margin - m_A) + softplus(margin - m_B))`."""

    loss_a = torch.nn.functional.softplus(torch.as_tensor(margin, dtype=torch.float32, device=scores["margin_a"].device) - scores["margin_a"])
    loss_b = torch.nn.functional.softplus(torch.as_tensor(margin, dtype=torch.float32, device=scores["margin_b"].device) - scores["margin_b"])
    loss = 0.5 * (loss_a + loss_b)
    return {
        "loss": loss,
        "loss_a": loss_a,
        "loss_b": loss_b,
        "raw": float(loss.detach()),
        "raw_a": float(loss_a.detach()),
        "raw_b": float(loss_b.detach()),
        "margin": float(margin),
        "pair_ranking_pass": bool(float(scores["margin_a"].detach()) > 0.0 and float(scores["margin_b"].detach()) > 0.0),
        "strict_margin_pass": bool(
            float(scores["margin_a"].detach()) >= margin and float(scores["margin_b"].detach()) >= margin
        ),
    }


def pair_step_loss(reasoning_a: torch.Tensor, reasoning_b: torch.Tensor,
                   heatmap_a: torch.Tensor, heatmap_b: torch.Tensor,
                   scores: dict) -> dict:
    """Section 9: the fixed pair objective and every recorded component."""

    cf = counterfactual_loss(scores)
    reasoning = 0.5 * (reasoning_a + reasoning_b)
    heatmaps = heatmap_a + heatmap_b
    total = PAIR_REASONING_WEIGHT * reasoning + PAIR_HEATMAP_WEIGHT * heatmaps + PAIR_CF_WEIGHT * cf["loss"]
    return {
        "total": total,
        "reasoning": reasoning,
        "heatmaps": heatmaps,
        "cf": cf,
        "raw": {
            "total": float(total.detach()),
            "reasoning_ce": float(reasoning.detach()),
            "heatmap_sum": float(heatmaps.detach()),
            "cf": cf["raw"],
            "cf_a": cf["raw_a"],
            "cf_b": cf["raw_b"],
        },
        "weights": {
            "reasoning": PAIR_REASONING_WEIGHT,
            "heatmap": PAIR_HEATMAP_WEIGHT,
            "cf": PAIR_CF_WEIGHT,
        },
    }


# ------------------------------------------------------------------ pair metrics


def pair_ranking_summary(rows: list[dict]) -> dict:
    """Sections 10/12: pair ranking and strict-margin rates with own-minus-cross margins."""

    ranked = [row for row in rows if row.get("pair_ranking_pass")]
    strict = [row for row in rows if row.get("strict_margin_pass")]
    margins = [row["mean_margin"] for row in rows if row.get("mean_margin") is not None]
    return {
        "pair_count": len(rows),
        "pair_ranking_pass": len(ranked),
        "pair_ranking_rate": (len(ranked) / len(rows)) if rows else None,
        "strict_margin_pass": len(strict),
        "strict_margin_rate": (len(strict) / len(rows)) if rows else None,
        "mean_own_minus_cross_margin": (float(sum(margins) / len(margins)) if margins else None),
        "min_own_minus_cross_margin": (float(min(margins)) if margins else None),
        "max_own_minus_cross_margin": (float(max(margins)) if margins else None),
    }
