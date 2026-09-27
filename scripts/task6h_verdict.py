"""Task 6H section 20: resolve exactly one verdict from the recorded artifacts.

Reads only tracked artifacts; it never re-runs a model. Writes
`evaluation/task6h_verdict.json` with the verdict and the evidence for every section-20 condition.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from task6h_common import EVAL, write_json  # noqa: E402

OUT = EVAL / "task6h_verdict.json"

VERDICTS = (
    "COUNTERFACTUAL_GROUNDING_FIX_FOUND",
    "COUNTERFACTUAL_GROUNDING_PARTIAL",
    "COUNTERFACTUAL_DENSE_GROUNDING_FAILED",
    "COUNTERFACTUAL_QUERY_SIGNAL_FAILED",
    "INVALID_EXPERIMENT",
)


def _load(name: str) -> dict | None:
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _pairs_are_valid(manifest: dict) -> list[str]:
    problems = []
    if int(manifest.get("pair_count", 0)) != 240:
        problems.append(f"expected 240 canonical pairs, found {manifest.get('pair_count')}")
    assertions = manifest.get("assertions", {})
    for key in (
        "same_source_image",
        "different_sample_ids",
        "different_target_component_ids",
        "deterministic_order",
    ):
        if not assertions.get(key):
            problems.append(f"pair manifest assertion failed: {key}")
    if manifest.get("test_split_used"):
        problems.append("the pair manifest used the test split")
    if assertions.get("target_masks_differ") is False:
        problems.append("a canonical pair has identical target masks")
    return problems


def main() -> int:
    manifest = _load("task6h_pair_manifest.json")
    token_setup = _load("task6h_token_setup.json")
    h0 = _load("task6h_h0_overfit.json")
    h1 = _load("task6h_h1_training.json")
    segmentation = _load("task6h_segmentation_eval.json")

    invalid_reasons = []
    if manifest is None:
        invalid_reasons.append("pair manifest missing")
    else:
        invalid_reasons.extend(_pairs_are_valid(manifest))
    if token_setup is None:
        invalid_reasons.append("token setup artifact missing")
    elif not token_setup.get("passed"):
        invalid_reasons.append("token setup smoke did not pass")
    if token_setup is not None:
        if not token_setup.get("architecture", {}).get("unchanged_from_task6g"):
            invalid_reasons.append("the Task 6G architecture dimensions changed")
        if not token_setup.get("causal_placement", {}).get("box_hidden_bit_identical"):
            invalid_reasons.append("the causal-placement bit-identity test did not pass")
        if not token_setup.get("one_pair_step", {}).get("sam2_bit_identical"):
            invalid_reasons.append("SAM2 parameters moved during a pair step (freeze violation)")
        if not token_setup.get("one_pair_step", {}).get("shared_feature_bit_identical"):
            invalid_reasons.append("the shared SAM2 feature changed during a pair step")
        sign = token_setup.get("one_pair_step", {}).get("region_scores")
        if sign is None:
            invalid_reasons.append("the pair step did not record region scores")
    if segmentation is not None and not segmentation.get("sam_frozen", False):
        invalid_reasons.append("SAM2 parameters were not frozen during H2")

    evidence = {"pair_manifest": None, "token_setup": None, "H0": None, "H1": None, "H2": None}
    if manifest is not None:
        evidence["pair_manifest"] = {
            "pair_count": manifest["pair_count"],
            "image_count": manifest["image_count"],
            "pair_identity_sha256": manifest["pair_identity_sha256"],
            "overlap_summary": manifest["overlap_summary"],
            "assertions": manifest["assertions"],
        }
    if token_setup is not None:
        evidence["token_setup"] = {
            "passed": token_setup["passed"],
            "architecture": token_setup["architecture"],
            "region_scores_before": token_setup["region_scores_before"],
            "region_scores_after": token_setup["region_scores_after"],
            "one_pair_step": {
                "losses": token_setup["one_pair_step"]["losses"],
                "sam2_bit_identical": token_setup["one_pair_step"]["sam2_bit_identical"],
                "shared_feature_bit_identical": token_setup["one_pair_step"][
                    "shared_feature_bit_identical"
                ],
            },
        }
    if h0 is not None:
        evidence["H0"] = {
            "verdict": h0["verdict"],
            "pair_ranking_pass": h0["final"]["pair"]["pair_ranking_pass"],
            "pair_count": h0["final"]["pair"]["pair_count"],
            "strict_margin_pass": h0["final"]["pair"]["strict_margin_pass"],
            "point_inside_own": h0["final"]["point_inside_own"],
            "record_count": h0["final"]["record_count"],
            "paired_point_selection_pass": h0["final"]["pair"]["paired_point_selection_pass"],
            "mean_own_minus_cross_margin": h0["final"]["pair"]["mean_own_minus_cross_margin"],
            "heatmap_dice_diagnostic": h0["final"]["heatmap_dice_diagnostic"],
            "gate": h0["final"]["gate"],
        }
    if h1 is not None:
        evidence["H1"] = {
            "verdict": h1["verdict"],
            "selected_epoch": h1["selection"]["selected_epoch"],
            "paired_point_selection_pass": h1["final"]["validation_pairs"][
                "paired_point_selection_pass"
            ],
            "paired_point_total": h1["final"]["validation_pairs"]["pair_count"],
            "pair_ranking_pass": h1["final"]["validation_pairs"]["pair_ranking_pass"],
            "mean_own_minus_cross_margin": h1["final"]["validation_pairs"][
                "mean_own_minus_cross_margin"
            ],
            "point_inside_target_rate": h1["final"]["val"]["point_inside_target_rate"],
            "gate": h1["final"]["gate"],
        }
    if segmentation is not None:
        paired = segmentation["segmentation"]["paired"]
        evidence["H2"] = {
            "strict_end_to_end_miou": segmentation["segmentation"]["strict_end_to_end_miou"],
            "mean_dice": segmentation["segmentation"]["mean_dice"],
            "mask_paired_pass": paired["mask_paired_pass"],
            "paired_total": paired["paired_total"],
            "mean_own_minus_cross_margin": paired["mean_own_minus_cross_margin"],
            "fix_checks": segmentation["fix_checks"],
        }

    if invalid_reasons:
        verdict, reason = "INVALID_EXPERIMENT", "; ".join(invalid_reasons)
    elif h0 is None or h0["verdict"] != "H0_PASS":
        failed = h0["final"]["pair"] if h0 is not None else {}
        verdict, reason = (
            "COUNTERFACTUAL_QUERY_SIGNAL_FAILED",
            "H0 could not learn own-vs-cross preference after the focused audit: pair ranking "
            f"{failed.get('pair_ranking_pass')}/{failed.get('pair_count')}, mean margin "
            f"{failed.get('mean_own_minus_cross_margin')}",
        )
    elif h1 is None or not h1["final"]["gate"]["passed"]:
        verdict, reason = (
            "COUNTERFACTUAL_DENSE_GROUNDING_FAILED",
            "H0 passes, but the pair-aware objective does not generalize: the H1 gate failed on the "
            "selected epoch",
        )
    elif segmentation is not None and all(segmentation["fix_checks"].values()):
        verdict, reason = (
            "COUNTERFACTUAL_GROUNDING_FIX_FOUND",
            "H0, the H1 pair/point gate, the H2 mask gate and the own-minus-cross mask margin all "
            "pass with no ground-truth geometry in any prompt",
        )
    else:
        verdict, reason = (
            "COUNTERFACTUAL_GROUNDING_PARTIAL",
            "pair ranking/query separation improves and localization improves materially, but the "
            "full H2 mask gate is not met or H2 was not run",
        )

    if verdict not in VERDICTS:
        raise RuntimeError(f"unknown verdict {verdict!r}")
    report = {
        "_doc": (
            "Task 6H section 20. Exactly one verdict, resolved from the recorded artifacts. "
            "Section 21: same-image counterfactual supervision is not claimed as paper novelty and "
            "is not called Spatial Consistency Loss."
        ),
        "task": "6H",
        "verdict": verdict,
        "reason": reason,
        "evidence": evidence,
    }
    write_json(OUT, report)
    print(f"[task6h:verdict] {verdict}: {reason}", flush=True)
    print(f"[task6h:verdict] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
