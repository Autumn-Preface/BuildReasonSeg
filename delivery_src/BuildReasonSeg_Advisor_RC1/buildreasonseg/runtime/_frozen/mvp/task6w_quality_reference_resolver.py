"""Task 6W Part G — quality-filtered deterministic reference resolver.

Inference path (section 16):

```text
U-C1 proposals
  -> Task 6Q family eligibility
  -> ProposalQualityEstimator v0.1
  -> keep quality_prob >= 0.50   (frozen threshold)
  -> largest: max area / smallest: min area
  -> Task 6Q confidence-then-index tie-break
```

If every family-eligible proposal is filtered out the resolver abstains explicitly with reason
`no_quality_eligible_proposals`; there is **no fallback** to an unfiltered proposal, and the Task 6U
ProposalSetRanker is not used anywhere in this resolver. A proposal whose 64x64 mask has no positive cell
is `feature_invalid_small`: its quality is forced to 0 (so it is filtered) and it is never scored.

GT is never an input.
"""

from __future__ import annotations

import numpy as np
import torch

from buildreasonseg.runtime._frozen.mvp.task6q_reference_resolver import is_eligible
from buildreasonseg.runtime._frozen.mvp.task6w_proposal_quality import (
    QUALITY_THRESHOLD,
    ProposalQualityEstimator,
    proposal_quality_features,
    quality_probabilities,
)

FAMILIES = ("largest", "smallest")


def deterministic_by_area(proposals: list, family: str):
    """Task 6Q largest/smallest area semantics with the confidence-then-index tie-break."""

    if not proposals:
        return None
    if family == "largest":
        return max(proposals, key=lambda proposal: (proposal.area_px, proposal.confidence,
                                                    -proposal.index))
    return min(proposals, key=lambda proposal: (proposal.area_px, -proposal.confidence,
                                                proposal.index))


class QualityReferenceResolver:
    """W-SQ: U-C1 proposals + learned quality filter + deterministic area selection."""

    def __init__(self, model: ProposalQualityEstimator, feature_store, device: str = "cpu",
                 threshold: float = QUALITY_THRESHOLD):
        self.model = model
        self.feature_store = feature_store
        self.device = device
        self.threshold = float(threshold)

    def score(self, proposals: list, feature: np.ndarray | None) -> dict:
        """Quality probabilities for the given proposals (invalid-small forced to 0)."""

        geometry, visual, valid = [], [], []
        for proposal in proposals:
            bundle = proposal_quality_features(proposal, feature)
            geometry.append(bundle.geometry)
            visual.append(bundle.visual)
            valid.append(bundle.valid)
        if not proposals:
            return {"probabilities": np.zeros((0,), dtype=np.float32), "valid": []}
        probabilities = quality_probabilities(self.model, np.asarray(visual, dtype=np.float32),
                                              np.asarray(geometry, dtype=np.float32),
                                              device=self.device)
        for index, is_valid in enumerate(valid):
            if not is_valid:
                probabilities[index] = 0.0
        return {"probabilities": probabilities, "valid": valid}

    def select(self, proposals: list, family: str, tile_id: str, image_path) -> dict:
        """Return the selected reference for one family (no GT, no ranker, no fallback)."""

        eligible = [proposal for proposal in proposals if is_eligible(proposal, family)]
        base = {"family": family, "proposal_count": len(proposals), "eligible_count": len(eligible),
                "threshold": self.threshold}
        if not proposals:
            return {**base, "abstained": True, "reason": "no_proposals", "mask": None}
        if not eligible:
            return {**base, "abstained": True, "reason": "no_eligible_proposals", "mask": None}
        feature = None
        if self.feature_store is not None:
            feature = self.feature_store.get(tile_id, image_path).float().cpu().numpy()
        scored = self.score(eligible, feature)
        probabilities = scored["probabilities"]
        kept = [(proposal, float(probability))
                for proposal, probability in zip(eligible, probabilities)
                if probability >= self.threshold]
        if not kept:
            return {**base, "abstained": True, "reason": "no_quality_eligible_proposals",
                    "mask": None,
                    "max_quality": float(probabilities.max()) if len(probabilities) else None,
                    "invalid_small": int(sum(1 for value in scored["valid"] if not value))}
        selected = deterministic_by_area([proposal for proposal, _ in kept], family)
        return {
            **base, "abstained": False, "reason": None, "mask": selected.mask,
            "selected_index": int(selected.index), "selected_area_px": int(selected.area_px),
            "selected_confidence": float(selected.confidence),
            "kept_count": len(kept),
            "filtered_count": len(eligible) - len(kept),
            "invalid_small": int(sum(1 for value in scored["valid"] if not value)),
        }

    def policy_report(self) -> dict:
        return {
            "pipeline": ["U-C1 proposals", "Task 6Q family eligibility",
                         "ProposalQualityEstimator v0.1", f"quality_prob >= {self.threshold}",
                         "largest: max area / smallest: min area", "Task 6Q tie-break"],
            "quality_threshold": self.threshold,
            "threshold_sweep": False,
            "fallback_to_unfiltered": False,
            "ranker_used": False,
            "abstention_reason_when_all_rejected": "no_quality_eligible_proposals",
            "invalid_small_quality_forced_to": 0.0,
            "gt_inputs": False,
        }


__all__ = ["FAMILIES", "QualityReferenceResolver", "deterministic_by_area"]
