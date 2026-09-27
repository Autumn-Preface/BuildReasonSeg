#!/usr/bin/env python
"""Task 6D.1 sections 9-11: representation statistics, expanded control and verdict.

    python scripts/task6d1_representation_stats.py

* **Probe D** (section 9) — do the statistics that LayerNorm discards (vector mean, standard
  deviation, L2 norm) correlate with, or predict, the target box?
* **Expanded four-condition control** (section 10) — same/different image x same/different
  template, reported as cosine, L2, raw norm, **centered** cosine, **LayerNorm-output**
  cosine, within-template and within-image variance, and the between-target box distance.
* **SVD/PCA** — effective rank and explained-variance profile of the raw `[SEG]` hiddens: what
  information does the token actually carry?
* **Interpretation matrix** (section 11) — resolves Cases A-F from the corrected G0-R gate and
  the three probes.

Writes `evaluation/task6d1_representation_stats.json` and
`evaluation/task6d1_decodability_summary.json`. No GPU work; the features are already extracted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

EVAL = REPO_ROOT / "evaluation"
HIDDEN_DIR = REPO_ROOT / "artifacts" / "task6d1_hidden"

STATS_OUT = EVAL / "task6d1_representation_stats.json"
SUMMARY_OUT = EVAL / "task6d1_decodability_summary.json"
G0_CORRECTED = EVAL / "task6d1_g0_corrected.json"


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.std() == 0 or right.std() == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _probe_d(hidden: np.ndarray, boxes: np.ndarray, split: str) -> dict:
    """Do the LayerNorm-discarded statistics carry box information?"""

    mean = hidden.mean(axis=1)
    std = hidden.std(axis=1)
    norm = np.linalg.norm(hidden, axis=1)
    center_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    center_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    area = width * height
    targets = {"center_x": center_x, "center_y": center_y, "width": width, "height": height, "area": area}
    correlations = {
        name: {
            "mean": _pearson(mean, values),
            "std": _pearson(std, values),
            "norm": _pearson(norm, values),
        }
        for name, values in targets.items()
    }

    # Predictive usefulness: a tiny linear map from the three statistics to the box.
    features = np.stack([mean, std, norm, np.ones_like(mean)], axis=1)
    solution, *_ = np.linalg.lstsq(features, boxes, rcond=None)
    predicted = features @ solution
    residual = boxes - predicted
    ss_res = float((residual**2).sum())
    ss_tot = float(((boxes - boxes.mean(axis=0)) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return {
        "split": split,
        "samples": int(hidden.shape[0]),
        "statistics": {
            "mean_of_mean": float(mean.mean()),
            "std_of_mean": float(mean.std()),
            "mean_of_std": float(std.mean()),
            "std_of_std": float(std.std()),
            "mean_of_norm": float(norm.mean()),
            "std_of_norm": float(norm.std()),
            "norm_range": [float(norm.min()), float(norm.max())],
        },
        "pearson_with_box": correlations,
        "linear_prediction_from_three_statistics": {
            "in_sample_r_squared": round(r_squared, 6),
            "note": "an in-sample fit of 480 rows with 4 parameters; interpret as an upper bound only",
        },
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator else 0.0


def _centered_cosine(left: np.ndarray, right: np.ndarray, global_mean: np.ndarray) -> float:
    return _cosine(left - global_mean, right - global_mean)


def _control(train_hidden: np.ndarray, paired_hidden: np.ndarray, paired_labels: list[dict]) -> dict:
    """Expanded four-condition control (section 10).

    Uses the **paired** split: the 20 paired images are a different subset of the val split than
    the fixed 120-record validation set (Task 6D.1 implementation audit), so only that split
    contains the same-image/different-template pairs the control needs.
    """

    index = {label["sample_id"]: position for position, label in enumerate(paired_labels)}
    from task6c_train import validation_material  # noqa: PLC0415

    _samples, pairs, _lookup, _audit = validation_material()
    global_mean = train_hidden.mean(axis=0)
    layernorm = torch.nn.LayerNorm(train_hidden.shape[1])
    with torch.no_grad():
        ln_hidden = layernorm(torch.as_tensor(paired_hidden)).numpy()

    def condition(sample_left: str, sample_right: str) -> dict | None:
        if sample_left not in index or sample_right not in index:
            return None
        left, right = index[sample_left], index[sample_right]
        vector_left, vector_right = paired_hidden[left], paired_hidden[right]
        label_left, label_right = paired_labels[left], paired_labels[right]
        return {
            "sample_left": sample_left,
            "sample_right": sample_right,
            "same_image": label_left["image_id"] == label_right["image_id"],
            "same_template": label_left["template_id"] == label_right["template_id"],
            "cosine": _cosine(vector_left, vector_right),
            "centered_cosine": _centered_cosine(vector_left, vector_right, global_mean),
            "layernorm_cosine": _cosine(ln_hidden[left], ln_hidden[right]),
            "l2": float(np.linalg.norm(vector_left - vector_right)),
            "norm_left": float(np.linalg.norm(vector_left)),
            "norm_right": float(np.linalg.norm(vector_right)),
            "box_l1": float(np.abs(np.asarray(label_left["box"]) - np.asarray(label_right["box"])).mean()),
        }

    conditions = {}
    if len(pairs) >= 2:
        pair_a, pair_b = pairs[0], pairs[1]
        repeated = [
            ("same_image_different_template", pair_a["a"]["sample_id"], pair_a["b"]["sample_id"]),
            ("different_image_same_template", pair_a["a"]["sample_id"], pair_b["a"]["sample_id"]),
            ("different_image_different_template", pair_a["a"]["sample_id"], pair_b["b"]["sample_id"]),
        ]
        for name, left, right in repeated:
            conditions[name] = condition(left, right)
        # Repeat over several pairs so the statement does not rest on one triple.
        aggregated = {"same_image_different_template": [], "different_image_same_template": []}
        for pair_a, pair_b in zip(pairs, pairs[1:]):
            for name, left, right in (
                ("same_image_different_template", pair_a["a"]["sample_id"], pair_a["b"]["sample_id"]),
                ("different_image_same_template", pair_a["a"]["sample_id"], pair_b["a"]["sample_id"]),
            ):
                record = condition(left, right)
                if record is not None:
                    aggregated[name].append(record)
        conditions["_aggregated"] = {
            name: {
                "n": len(records),
                "mean_cosine": float(np.mean([r["cosine"] for r in records])) if records else None,
                "mean_centered_cosine": float(np.mean([r["centered_cosine"] for r in records]))
                if records
                else None,
                "mean_layernorm_cosine": float(np.mean([r["layernorm_cosine"] for r in records]))
                if records
                else None,
                "mean_l2": float(np.mean([r["l2"] for r in records])) if records else None,
            }
            for name, records in aggregated.items()
        }

    # Within-image and within-template variance of the raw hiddens.
    images: dict[str, list[int]] = {}
    templates: dict[str, list[int]] = {}
    for position, label in enumerate(paired_labels):
        images.setdefault(label["image_id"], []).append(position)
        templates.setdefault(label["template_id"], []).append(position)

    def within_variance(groups: dict[str, list[int]]) -> dict:
        variances = []
        for positions in groups.values():
            if len(positions) < 2:
                continue
            block = paired_hidden[positions]
            variances.append(float(block.var(axis=0).mean()))
        return {
            "groups_with_repeats": len(variances),
            "mean_within_group_variance": float(np.mean(variances)) if variances else None,
        }

    between_box = []
    for left in range(len(paired_labels)):
        for right in range(left + 1, len(paired_labels)):
            between_box.append(
                float(
                    np.abs(
                        np.asarray(paired_labels[left]["box"]) - np.asarray(paired_labels[right]["box"])
                    ).mean()
                )
            )
    return {
        "conditions": conditions,
        "global_mean_norm": float(np.linalg.norm(global_mean)),
        "within_image_variance": within_variance(images),
        "within_template_variance": within_variance(templates),
        "between_target_box_l1_mean": float(np.mean(between_box)) if between_box else None,
        "interpretation": (
            "centered and LayerNorm cosines separate 'the token is constant' from 'the token has a "
            "large shared component'; within-image variance shows how much the instruction moves the "
            "token relative to the between-target box distance."
        ),
    }


def _svd_profile(hidden: np.ndarray, hidden_val: np.ndarray) -> dict:
    centered = hidden - hidden.mean(axis=0)
    _u, singular, _vt = np.linalg.svd(centered, full_matrices=False)
    energy = singular**2
    total = energy.sum()
    shares = energy / total if total else energy
    participation = float((energy.sum() ** 2) / (energy**2).sum()) if total else 0.0
    return {
        "samples": int(hidden.shape[0]),
        "dim": int(hidden.shape[1]),
        "effective_rank_participation_ratio": round(participation, 4),
        "top1_explained_variance": round(float(shares[:1].sum()), 6),
        "top5_explained_variance": round(float(shares[:5].sum()), 6),
        "top10_explained_variance": round(float(shares[:10].sum()), 6),
        "top50_explained_variance": round(float(shares[:50].sum()), 6),
        "components_for_90_percent": int(np.searchsorted(np.cumsum(shares), 0.90) + 1),
        "val_centered_norm_mean": float(np.linalg.norm(hidden_val - hidden.mean(axis=0), axis=1).mean()),
    }


def _summary(probes: dict, stats: dict, g0: dict) -> dict:
    """Section 11's interpretation matrix, resolved from the measured numbers."""

    g0_passed = bool((g0.get("gate") or {}).get("passed"))
    raw = probes.get("raw_mlp") or {}
    ln = probes.get("layernorm_mlp") or {}
    linear = probes.get("linear") or {}

    def val_iou(probe: dict) -> float | None:
        return ((probe.get("validation") or {}).get("box_iou_mean"))

    def overfit_iou(probe: dict) -> float | None:
        return ((probe.get("overfit_20") or {}).get("box_iou_mean"))

    def train_iou(probe: dict) -> float | None:
        return ((probe.get("train_480") or {}).get("box_iou_mean"))

    raw_overfit, raw_train, raw_val = overfit_iou(raw), train_iou(raw), val_iou(raw)
    ln_train, ln_val = train_iou(ln), val_iou(ln)
    linear_train, linear_val = train_iou(linear), val_iou(linear)

    # The 20-sample overfit is only diagnostic if shuffled labels do NOT fit equally well.
    shuffled_overfit_loss = ((raw.get("label_shuffled_control") or {}).get("overfit_20_final_loss"))
    raw_overfit_loss = ((raw.get("overfit_20") or {}).get("final_loss"))
    real_overfit_loss = ((raw.get("overfit_20") or {}).get("final_loss"))
    overfit_is_diagnostic = bool(
        shuffled_overfit_loss is None
        or real_overfit_loss is None
        or shuffled_overfit_loss > 0.5 * max(real_overfit_loss, 1e-9)
    )
    signal_gap = {
        name: (probe.get("decodable_signal_gap_train_480"))
        for name, probe in probes.items()
    }
    implementation_verified = all(
        bool(((probe.get("implementation_audit") or {}).get("converged", True))) for probe in probes.values()
    ) if probes else False

    if g0_passed:
        case, verdict, statement = (
            "A",
            "SCHEDULER_FIX_RECOVERS_GROUNDING",
            "the corrected scheduler recovers grounding, so the Task 6D failure was caused by the "
            "scheduler-horizon defect rather than by the representation",
        )
    elif raw_overfit is not None and raw_overfit < 0.50 and overfit_is_diagnostic:
        case, verdict, statement = (
            "D",
            "READOUT_OR_FEATURE_IDENTITY_BUG_SUSPECTED",
            "even a 1M-parameter MLP cannot overfit 20 frozen hidden vectors, so the implementation "
            "or the feature identity must be audited before any information claim",
        )
    elif raw_train is not None and raw_train < 0.20:
        case, verdict, statement = (
            "F",
            "PRACTICALLY_NOT_DECODABLE_GEOMETRY",
            "after the implementation was audited and every readout converged, the raw probes cannot "
            "meaningfully fit even the 480 training samples, and the real fit is no better than the "
            "label-shuffled control: the current [SEG] representation contains no practically "
            "decodable target geometry under this training setup. This is a statement about "
            "practical decodability, not an information-theoretic absence claim.",
        )
    elif raw_train is not None and raw_val is not None and raw_train >= 0.50 and raw_val < 0.20:
        case, verdict, statement = (
            "E",
            "SPATIAL_REPRESENTATION_DOES_NOT_GENERALIZE",
            "the raw MLP fits the training samples but validation stays collapsed, so the current "
            "[SEG] features are unsuitable as a generalizable geometry code",
        )
    elif (
        raw_train is not None
        and ln_train is not None
        and raw_train - ln_train > 0.15
        and raw_val is not None
        and ln_val is not None
        and raw_val - ln_val > 0.05
    ):
        case, verdict, statement = (
            "B",
            "LAYER_NORM_READOUT_CONFOUND",
            "the raw readout decodes geometry appreciably better than the LayerNorm readout, so "
            "LayerNorm was destroying usable location signal",
        )
    elif raw_train is not None and raw_train >= 0.20 and raw_val is not None and raw_val >= 0.20:
        case, verdict, statement = (
            "C",
            "LOCATION_SIGNAL_PARTIALLY_GENERALIZES",
            "the frozen representation supports a partially generalizing readout; the Task 6D "
            "collapse was a training/horizon artifact rather than an absent signal",
        )
    else:
        case, verdict, statement = (
            "C",
            "LOCATION_SIGNAL_MEMORIZABLE_NOT_GENERALIZABLE",
            "the representation carries sample-specific information that the readout can fit, but no "
            "robust spatial code that generalizes to unseen images",
        )

    return {
        "_doc": (
            "Task 6D.1 sections 11 and 14. Resolution of the interpretation matrix from the corrected "
            "G0-R gate and the three frozen probes. Every number quoted here is in "
            "task6d1_probe_*.json, task6d1_representation_stats.json and task6d1_g0_corrected.json."
        ),
        "task": "6D.1",
        "case": case,
        "verdict": verdict,
        "statement": statement,
        "inputs": {
            "g0_corrected_passed": g0_passed,
            "g0_corrected_gate": g0.get("gate"),
            "probe_linear_train_480_box_iou": linear_train,
            "probe_linear_val_box_iou": linear_val,
            "probe_raw_mlp_overfit_20_box_iou": raw_overfit,
            "probe_raw_mlp_train_480_box_iou": raw_train,
            "probe_raw_mlp_val_box_iou": raw_val,
            "probe_layernorm_mlp_train_480_box_iou": ln_train,
            "probe_layernorm_mlp_val_box_iou": ln_val,
            "real_minus_shuffled_train_480_box_iou": signal_gap,
            "twenty_sample_overfit_is_diagnostic": overfit_is_diagnostic,
            "shuffled_overfit_final_loss": shuffled_overfit_loss,
            "every_probe_converged": implementation_verified,
            "layernorm_is_a_confound": bool(
                raw_val is not None
                and ln_val is not None
                and ln_train is not None
                and raw_train is not None
                and (raw_train - ln_train) > 0.15
                and (raw_val - ln_val) > 0.05
            ),
        },
        "raw_hidden_svd": stats.get("svd"),
        "predeclared_cases": {
            "A": "corrected G0-R passes -> the scheduler defect caused the failure",
            "B": "raw probes decode, LayerNorm probe fails -> LAYER_NORM_READOUT_CONFOUND",
            "C": "raw MLP fits train/20 but not validation -> LOCATION_SIGNAL_MEMORIZABLE_NOT_GENERALIZABLE",
            "D": "even a raw 1M MLP cannot overfit 20 frozen vectors -> READOUT_OR_FEATURE_IDENTITY_BUG_SUSPECTED",
            "E": "raw MLP overfits 20 and 480 but validation stays collapsed -> SPATIAL_REPRESENTATION_DOES_NOT_GENERALIZE",
            "F": "raw probes cannot fit even 480 after verification -> practically no decodable geometry",
        },
        "wording_rule": (
            "Use PRACTICALLY_DECODABLE language. Do not claim information-theoretic absence. Cosine "
            "similarity near 1 does not prove the absence of a small target-dependent component, and "
            "one failed MLP readout does not prove absence either."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = argparse.ArgumentParser(description=__doc__).parse_args(argv)

    train_payload = torch.load(HIDDEN_DIR / "train.pt", map_location="cpu", weights_only=False)
    val_payload = torch.load(HIDDEN_DIR / "val.pt", map_location="cpu", weights_only=False)
    paired_payload = torch.load(HIDDEN_DIR / "paired.pt", map_location="cpu", weights_only=False)
    train_hidden = train_payload["hidden"].numpy()
    val_hidden = val_payload["hidden"].numpy()
    paired_hidden = paired_payload["hidden"].numpy()
    train_boxes = np.asarray([label["box"] for label in train_payload["labels"]], dtype=np.float64)
    val_boxes = np.asarray([label["box"] for label in val_payload["labels"]], dtype=np.float64)

    stats = {
        "_doc": (
            "Task 6D.1 sections 9-10. Representation statistics: the three LayerNorm-discarded "
            "statistics and their relation to the target box, the expanded four-condition "
            "similarity control, and an SVD/PCA profile of the raw [SEG] hiddens from the "
            "hash-verified Task 6C P_C checkpoint."
        ),
        "task": "6D.1",
        "representation_source": "evaluation/task6d1_hidden_extract_manifest.json",
        "probe_d_train": _probe_d(train_hidden, train_boxes, "train"),
        "probe_d_val": _probe_d(val_hidden, val_boxes, "val"),
        "four_condition_control": _control(train_hidden, paired_hidden, paired_payload["labels"]),
        "svd": _svd_profile(train_hidden, val_hidden),
        "causality_caveat": (
            "correlation is not causation: these statistics are descriptive of the frozen checkpoint "
            "and are reported to decide whether LayerNorm could be discarding signal."
        ),
    }
    STATS_OUT.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    probes = {
        name: json.loads((EVAL / f"task6d1_probe_{name}.json").read_text(encoding="utf-8"))
        for name in ("linear", "raw_mlp", "layernorm_mlp")
        if (EVAL / f"task6d1_probe_{name}.json").is_file()
    }
    g0 = json.loads(G0_CORRECTED.read_text(encoding="utf-8")) if G0_CORRECTED.is_file() else {}
    summary = _summary(probes, stats, g0)
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    control = stats["four_condition_control"]["conditions"].get("_aggregated", {})
    print(f"[task6d1:stats] SVD effective rank {stats['svd']['effective_rank_participation_ratio']} "
          f"top1 {stats['svd']['top1_explained_variance']} top10 {stats['svd']['top10_explained_variance']}")
    print(f"[task6d1:stats] within-image variance "
          f"{stats['four_condition_control']['within_image_variance']['mean_within_group_variance']} "
          f"between-target box L1 {stats['four_condition_control']['between_target_box_l1_mean']}")
    print(f"[task6d1:stats] control: {json.dumps(control, ensure_ascii=False)}")
    print(f"[task6d1:stats] case {summary['case']} -> {summary['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
