"""Task 6C section 13: prompt-pathway diagnostics.

Measures the same quantities at every stage of

    [SEG] hidden (2048) -> Projection MLP -> sparse prompt actually given to SAM

so the four arms can be compared with one code path. Nothing here is derived from
ground truth except the mask overlap between two predictions, and no ground-truth
geometry enters inference.
"""

from __future__ import annotations

import itertools
import math
from typing import Sequence

import torch

from .metrics import upsample_logits


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))


def vector_stats(a: torch.Tensor, b: torch.Tensor, vectors: Sequence[torch.Tensor]) -> dict:
    """Cosine / L2 / norms for one same-image pair plus the population stats."""

    stacked = torch.stack([v.flatten().float() for v in vectors]) if vectors else None
    per_dim_std = float(stacked.std(dim=0).mean()) if stacked is not None and stacked.shape[0] > 1 else 0.0
    mean_abs = float(stacked.abs().mean()) if stacked is not None else 0.0
    return {
        "cosine": _cosine(a, b),
        "l2": float((a.flatten().float() - b.flatten().float()).norm()),
        "norm_a": float(a.flatten().float().norm()),
        "norm_b": float(b.flatten().float().norm()),
        "mean_norm": float(stacked.norm(dim=1).mean()) if stacked is not None else None,
        "norm_std": float(stacked.norm(dim=1).std()) if stacked is not None and stacked.shape[0] > 1 else 0.0,
        "per_dim_std_mean": per_dim_std,
        "mean_abs_activation": mean_abs,
        "dim": int(a.numel()),
    }


def svd_summary(vectors: Sequence[torch.Tensor]) -> dict:
    """Centred SVD of a set of vectors: spectrum, variance explained, effective rank."""

    if len(vectors) < 2:
        return {"available": False, "n": len(vectors)}
    matrix = torch.stack([v.flatten().float() for v in vectors])
    centred = matrix - matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centred)
    energy = singular.pow(2)
    total = float(energy.sum())
    if total <= 0:
        return {"available": True, "n": len(vectors), "total_energy": 0.0, "effective_rank": 0.0}
    explained = (energy / total).tolist()
    probabilities = energy / total
    entropy = float(-(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum())
    return {
        "available": True,
        "n": len(vectors),
        "top10_singular_values": [round(value, 6) for value in singular[:10].tolist()],
        "variance_explained_top1": round(sum(explained[:1]), 6),
        "variance_explained_top5": round(sum(explained[:5]), 6),
        "variance_explained_top10": round(sum(explained[:10]), 6),
        "effective_rank": round(math.exp(entropy), 4),
        "effective_rank_by_90pct": int(
            next((index + 1 for index, value in enumerate(torch.cumsum(energy / total, 0).tolist()) if value >= 0.9), len(explained))
        ),
        "total_energy": total,
    }


def _prediction_iou(logits_a, logits_b, size) -> float:
    mask_a = upsample_logits(logits_a, size) > 0.0
    mask_b = upsample_logits(logits_b, size) > 0.0
    intersection = int((mask_a & mask_b).sum().item())
    union = int((mask_a | mask_b).sum().item())
    return intersection / union if union else 0.0


@torch.no_grad()
def diagnose(runtime, pairs: Sequence[dict], n_same_image: int = 10, n_cross_image: int = 10) -> dict:
    """Teacher-forced prompt-pathway diagnosis for one arm.

    `pairs` are same-image / different-instruction records (`{"a": rec, "b": rec}`).
    Ground truth is used only to score the two decoded masks against each other.
    """

    from . import data as data_mod

    runtime.model.eval()
    selected = list(pairs)[:n_same_image]

    hidden_vectors: list[torch.Tensor] = []
    projected_vectors: list[torch.Tensor] = []
    sparse_vectors: list[torch.Tensor] = []
    entries: list[dict] = []

    for pair in selected:
        measured: dict[str, dict] = {}
        for label in ("a", "b"):
            sample = data_mod.to_sample(pair[label])
            image = sample.image_rgb()
            batch, _image = runtime.prepare(sample)
            features, _cached = runtime.features_for(sample, image)
            output = runtime.model(batch.to(runtime.device), features)
            measured[label] = {
                "sample_id": sample.sample_id,
                "query_type": sample.query_type,
                "seg_hidden": output.seg_hidden.detach().float().flatten().cpu(),
                "projected": output.projected.detach().float().flatten().cpu(),
                # The full sparse prompt exactly as SAM receives it. For the centre
                # bridge SAM's prompt encoder appends its own padding slot, so this is
                # (1, 2, 256); for the language bridge it is (1, 1, 256).
                "sparse_prompt": output.sparse_prompt.detach().float().flatten().cpu(),
                "sparse_prompt_shape": list(output.sparse_prompt.shape),
                # Slot 0 is the language-carrying slot in both bridges.
                "sparse_language_slot": output.sparse_prompt.detach().float()[0, 0, :].cpu(),
                "logits": output.mask_logits.detach().float().cpu(),
                "mask_shape": sample.target_mask().shape,
                "prompt_diagnostics": dict(output.prompt_diagnostics),
            }
            hidden_vectors.append(measured[label]["seg_hidden"])
            projected_vectors.append(measured[label]["projected"])
            sparse_vectors.append(measured[label]["sparse_prompt"])

        a, b = measured["a"], measured["b"]
        entries.append(
            {
                "image_id": pair["a"]["image_id"] if isinstance(pair["a"], dict) else None,
                "a": a["sample_id"],
                "b": b["sample_id"],
                "a_query_type": a["query_type"],
                "b_query_type": b["query_type"],
                "seg_hidden": vector_stats(a["seg_hidden"], b["seg_hidden"], hidden_vectors),
                "projected": vector_stats(a["projected"], b["projected"], projected_vectors),
                "sparse_prompt": vector_stats(a["sparse_prompt"], b["sparse_prompt"], sparse_vectors),
                "pred_iou_a_vs_b": _prediction_iou(a["logits"], b["logits"], a["mask_shape"]),
                "sparse_prompt_shape": a["sparse_prompt_shape"],
                "bridge_diagnostics_a": a["prompt_diagnostics"],
                "bridge_diagnostics_b": b["prompt_diagnostics"],
                # For the language bridge slot 0 must be the projected token itself;
                # for the centre bridge it must NOT be (it is point + language).
                "sparse_prompt_equals_projected": bool(
                    torch.allclose(a["sparse_language_slot"], a["projected"], atol=1e-6, rtol=1e-5)
                ),
            }
        )

    # Different-image reference: the first record of each selected pair against the
    # first record of a different pair.
    cross_hidden: list[float] = []
    cross_projected: list[float] = []
    cross_sparse: list[float] = []
    first_of_pair = []
    for pair in selected:
        sample = data_mod.to_sample(pair["a"])
        image = sample.image_rgb()
        batch, _image = runtime.prepare(sample)
        features, _cached = runtime.features_for(sample, image)
        output = runtime.model(batch.to(runtime.device), features)
        first_of_pair.append(
            (
                output.seg_hidden.detach().float().flatten().cpu(),
                output.projected.detach().float().flatten().cpu(),
                output.sparse_prompt.detach().float().flatten().cpu(),
            )
        )
    for (h1, p1, s1), (h2, p2, s2) in itertools.combinations(first_of_pair, 2):
        cross_hidden.append(_cosine(h1, h2))
        cross_projected.append(_cosine(p1, p2))
        cross_sparse.append(_cosine(s1, s2))

    def _mm(values: list[float]) -> dict:
        if not values:
            return {"mean": None, "min": None, "max": None, "n": 0}
        return {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "n": len(values),
        }

    bridge = runtime.bridge
    point_norms = [
        entry["bridge_diagnostics_a"]["point_embedding_norm"]
        for entry in entries
        if entry["bridge_diagnostics_a"].get("point_embedding_norm") is not None
    ]
    ratios = [
        entry["bridge_diagnostics_a"]["ratio_projected_over_point"]
        for entry in entries
        if entry["bridge_diagnostics_a"].get("ratio_projected_over_point") is not None
    ]
    language_norms = [
        entry["bridge_diagnostics_a"]["projected_norm"]
        for entry in entries
        if entry["bridge_diagnostics_a"].get("projected_norm") is not None
    ]

    runtime.model.train()
    return {
        "_doc": (
            "Task 6C section 13 prompt-pathway diagnosis. Teacher-forced, so generation quality cannot "
            "confound it. Ground truth is used only for the prediction-vs-prediction overlap."
        ),
        "bridge": bridge,
        "n_same_image_pairs": len(entries),
        "n_different_image_references": len(cross_hidden),
        "same_image_different_instruction": {
            "seg_hidden_cosine": _mm([e["seg_hidden"]["cosine"] for e in entries]),
            "seg_hidden_l2": _mm([e["seg_hidden"]["l2"] for e in entries]),
            "projected_cosine": _mm([e["projected"]["cosine"] for e in entries]),
            "projected_l2": _mm([e["projected"]["l2"] for e in entries]),
            "sparse_prompt_cosine": _mm([e["sparse_prompt"]["cosine"] for e in entries]),
            "sparse_prompt_l2": _mm([e["sparse_prompt"]["l2"] for e in entries]),
            "prediction_iou_a_vs_b": _mm([e["pred_iou_a_vs_b"] for e in entries]),
        },
        "different_image_reference": {
            "seg_hidden_cosine": _mm(cross_hidden),
            "projected_cosine": _mm(cross_projected),
            "sparse_prompt_cosine": _mm(cross_sparse),
        },
        "representation": {
            "seg_hidden": svd_summary(hidden_vectors),
            "projected": svd_summary(projected_vectors),
            "sparse_prompt": svd_summary(sparse_vectors),
            "projected_norm_mean": (
                sum(v.norm().item() for v in projected_vectors) / len(projected_vectors)
                if projected_vectors
                else None
            ),
        },
        "bridge_specific": {
            "mode": bridge,
            "sparse_prompt_shape": entries[0]["sparse_prompt_shape"] if entries else None,
            "sparse_prompt_shape_note": (
                "centre bridge: SAM's prompt encoder appends its own padding slot, so the sparse "
                "prompt is (1, 2, prompt_dim); language bridge: (1, 1, prompt_dim). The cosine/L2/norm "
                "numbers below are computed on the full tensor SAM actually receives."
            ),
            "point_embedding_norm_mean": (sum(point_norms) / len(point_norms)) if point_norms else None,
            "projected_language_norm_mean": (sum(language_norms) / len(language_norms)) if language_norms else None,
            "ratio_projected_over_point_mean": (sum(ratios) / len(ratios)) if ratios else None,
            "sparse_prompt_equals_projected_all": (
                all(e["sparse_prompt_equals_projected"] for e in entries) if entries else None
            ),
            "sparse_prompt_equals_projected_expected": bridge == "language",
        },
        "pairs": entries,
    }
