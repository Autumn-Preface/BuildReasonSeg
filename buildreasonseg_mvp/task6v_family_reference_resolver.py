"""Task 6V — family-conditioned reference resolver over already-frozen resolver options.

The resolver performs **only family dispatch** onto the three predeclared frozen options; it adds no
learned logic, no thresholds and no fourth option:

```text
V-P0 = U-C0 proposals + frozen Task 6Q deterministic area selector
V-P1 = U-C1 proposals + frozen Task 6Q deterministic area selector
V-P2 = U-C1 proposals + frozen ProposalSetRanker v0.1

if family == largest:  use the frozen selected largest policy
elif family == smallest: use the frozen selected smallest policy
```

The per-family policy is chosen **only** on the train-only U-Calib200 split and frozen before any
RefValUnique evaluation. The resolver never receives GT, the relation, the target mask or the natural
language; no new model is trained in Task 6V.
"""

from __future__ import annotations

import json
from pathlib import Path

from buildreasonseg_mvp.task6q_reference_resolver import is_eligible, select_reference

FAMILIES = ("largest", "smallest")

#: Exactly three predeclared resolver options (section 4). No fourth option may be introduced.
OPTIONS: dict[str, dict] = {
    "V-P0": {"id": "V-P0", "config": "U-C0", "selector": "deterministic",
             "description": "U-C0 + deterministic Task 6Q area selector"},
    "V-P1": {"id": "V-P1", "config": "U-C1", "selector": "deterministic",
             "description": "U-C1 + deterministic Task 6Q area selector"},
    "V-P2": {"id": "V-P2", "config": "U-C1", "selector": "ranker",
             "description": "U-C1 + frozen ProposalSetRanker v0.1"},
}
OPTION_ORDER = ("V-P0", "V-P1", "V-P2")

#: Simplicity tie-break order used by the section 7 priority rule.
SIMPLICITY_ORDER = {"V-P0": 0, "V-P1": 1, "V-P2": 2}

DEFAULT_POLICY = {"largest": "V-P1", "smallest": "V-P1"}


def load_frozen_policy(path: Path) -> dict:
    """Read the frozen per-family policy artifact (must contain both families and no extra keys)."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    policy = payload.get("policy", payload)
    resolved = {}
    for family in FAMILIES:
        option = policy.get(family)
        if option not in OPTIONS:
            raise ValueError(f"frozen policy for {family!r} is not a declared option: {option!r}")
        resolved[family] = option
    return resolved


def resolve_family_policy(family: str, policy: dict) -> str:
    """Family-only dispatch (section 8) — the relation/target/GT are not inputs."""

    if family not in OPTIONS and family not in FAMILIES:
        raise ValueError(f"unknown reference family {family!r}")
    option = policy.get(family)
    if option not in OPTIONS:
        raise ValueError(f"no frozen policy for family {family!r}: {option!r}")
    return option


class FamilyConditionedResolver:
    """Dispatch onto frozen resolver options by reference family only."""

    def __init__(self, policy: dict, ranker=None):
        self.policy = {family: resolve_family_policy(family, policy) for family in FAMILIES}
        self.ranker = ranker

    def option_for(self, family: str) -> dict:
        return OPTIONS[resolve_family_policy(family, self.policy)]

    def select(self, proposals, family: str) -> dict:
        """Return the selected reference mask for one family (no GT, no relation input)."""

        option = self.option_for(family)
        if option["selector"] == "deterministic":
            selection = select_reference(proposals, family)
            if selection.abstained:
                return {"abstained": True, "reason": selection.reason, "mask": None,
                        "option": option["id"], "config": option["config"],
                        "selector": option["selector"]}
            return {"abstained": False, "reason": None, "mask": selection.mask,
                    "option": option["id"], "config": option["config"],
                    "selector": option["selector"],
                    "index": int(selection.proposal.index),
                    "area_px": int(selection.proposal.area_px)}

        eligible = [proposal for proposal in proposals if is_eligible(proposal, family)]
        if not eligible:
            return {"abstained": True, "reason": "no_eligible_proposals", "mask": None,
                    "option": option["id"], "config": option["config"],
                    "selector": option["selector"]}
        if self.ranker is None:
            raise RuntimeError("V-P2 requires the frozen ProposalSetRanker checkpoint")
        from buildreasonseg_mvp.task6u_reference_ranker import select_with_ranker

        outcome = select_with_ranker(self.ranker, eligible, family, device="cpu")
        chosen = eligible[outcome["selected_index"]]
        return {"abstained": False, "reason": None, "mask": chosen.mask,
                "option": option["id"], "config": option["config"],
                "selector": option["selector"], "index": int(chosen.index),
                "area_px": int(chosen.area_px),
                "ranker_score": outcome["scores"][outcome["selected_index"]]}

    def policy_report(self) -> dict:
        return {
            "policy": dict(self.policy),
            "options": {family: OPTIONS[self.policy[family]]["description"] for family in FAMILIES},
            "dispatch": "reference family only; no relation, target, GT or language input",
            "learned_logic_added": False,
            "fourth_option_added": False,
        }


def policy_priority_key(metrics: dict) -> tuple:
    """Section 7 priority (higher tuple sorts first).

    1. higher Pr@0.5; 2. higher selected-reference mIoU; 3. lower REFERENCE_SELECTION_WRONG;
    4. lower centroid median; 5. lower abstention rate; 6. simpler option first.
    """

    return (
        metrics["precision_at_0_5"],
        metrics["selected_reference_miou"],
        -metrics["REFERENCE_SELECTION_WRONG"],
        -metrics["centroid_median"],
        -metrics["abstention_rate"],
        -SIMPLICITY_ORDER[metrics["option"]],
    )


def choose_policy(per_family_metrics: dict) -> dict:
    """Freeze one option per family from the calibration metrics (U-Calib200 only)."""

    policy = {}
    for family in FAMILIES:
        candidates = per_family_metrics[family]
        ranked = sorted(OPTION_ORDER,
                        key=lambda option: policy_priority_key(candidates[option]), reverse=True)
        policy[family] = ranked[0]
    return policy


__all__ = [
    "DEFAULT_POLICY",
    "FAMILIES",
    "FamilyConditionedResolver",
    "OPTIONS",
    "OPTION_ORDER",
    "SIMPLICITY_ORDER",
    "choose_policy",
    "load_frozen_policy",
    "policy_priority_key",
    "resolve_family_policy",
]
