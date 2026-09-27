"""Task 6H.1 section 14: focused H0-R audit (run only when H0-R fails).

Inspects the corrected objective only — no architecture or loss change:

* point-CE and bounded-`L_cf` gradient norms into the query projection, the visual 1×1 projection,
  the Qwen LoRA and the `[BOX]`/`[SEG]` token rows;
* target-cell probability, maximum non-target probability and spatial-softmax entropy, before
  (clean initialization) and after H0-R;
* whether `L_point` decreases and whether the argmax point approaches the GT interior point;
* A/B query-hidden and projected-query distances;
* pair identity / target-index bookkeeping and shared-feature bit identity.

Writes `evaluation/task6h1_h0r_audit.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.box_query import build_box_query_batch, box_query_forward  # noqa: E402
from buildreasonseg_mvp.checkpointing import load_checkpoint  # noqa: E402
from buildreasonseg_mvp.dense_eval import forward_heatmap  # noqa: E402
from buildreasonseg_mvp.dense_grounding import downsample_target_mask, feature_tensor_for_grid  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, set_seed  # noqa: E402
from buildreasonseg_mvp.spatial_objective import (  # noqa: E402
    bounded_pair_loss,
    counterfactual_masses,
    max_non_target_probability,
    point_cross_entropy,
    spatial_entropy,
    spatial_probabilities,
    target_cell,
    target_cell_probability,
    topk_hit,
)

from task6h1_common import (  # noqa: E402
    EVAL,
    canonical_train_pairs,
    h0r_pairs,
    load_point_config,
    write_json,
)

H0R_JSON = EVAL / "task6h1_h0r_overfit.json"
OUT = EVAL / "task6h1_h0r_audit.json"


def _gradient_groups(runtime, loss) -> dict:
    runtime.model.zero_grad(set_to_none=True)
    loss.backward()
    groups = {
        "dense_head.query_proj": 0.0,
        "dense_head.query_norm": 0.0,
        "dense_head.visual_proj": 0.0,
        "qwen_lora": 0.0,
        "token_rows": 0.0,
        "other": 0.0,
    }
    for name, parameter in runtime.model.named_parameters():
        if parameter.grad is None:
            continue
        norm = float(parameter.grad.detach().float().norm())
        if name.startswith("dense_head.query_proj"):
            key = "dense_head.query_proj"
        elif name.startswith("dense_head.query_norm"):
            key = "dense_head.query_norm"
        elif name.startswith("dense_head.visual_proj"):
            key = "dense_head.visual_proj"
        elif "lora_" in name:
            key = "qwen_lora"
        elif "trainable_tokens" in name or "token_row" in name or "token_holder" in name:
            key = "token_rows"
        else:
            key = "other"
        groups[key] = (groups[key] ** 2 + norm**2) ** 0.5
    runtime.model.zero_grad(set_to_none=True)
    return groups


def _state(runtime, triples, grid: int) -> dict:
    """Objective/geometry state over the H0-R pairs at the current weights."""

    point_ces, cell_probs, top1, inside, entropies, non_target = [], [], [], [], [], []
    hidden_distances, projected_distances = [], []
    p_aa_list, p_ab_list, p_bb_list, p_ba_list, margins, cf_losses = [], [], [], [], [], []
    distance_errors = []
    for pair, record_a, record_b in triples:
        sample_a = data_mod.to_sample(record_a)
        sample_b = data_mod.to_sample(record_b)
        image = sample_a.image_rgb()
        features, _cached = runtime.features_for(sample_a, image)
        raw_a = forward_heatmap(runtime, sample_a, image=image, features=features)
        raw_b = forward_heatmap(runtime, sample_b, image=image, features=features)
        cell_a = target_cell(sample_a.target_mask(), grid)
        cell_b = target_cell(sample_b.target_mask(), grid)

        for raw, sample, cell in ((raw_a, sample_a, cell_a), (raw_b, sample_b, cell_b)):
            logits = raw["logits"]
            probabilities = spatial_probabilities(logits)
            point_ces.append(float(point_cross_entropy(logits, cell.index)))
            cell_probs.append(target_cell_probability(probabilities, cell.index))
            top1.append(1.0 if topk_hit(logits, cell.index, 1) else 0.0)
            entropies.append(spatial_entropy(probabilities))
            non_target.append(max_non_target_probability(probabilities, raw["soft_target"]))
            predicted = torch.argmax(logits.reshape(-1)).item()
            y, x = divmod(int(predicted), grid)
            predicted_point = ((x + 0.5) / grid, (y + 0.5) / grid)
            from buildreasonseg_mvp.dense_grounding import point_inside_mask

            inside.append(1.0 if point_inside_mask(sample.target_mask(), predicted_point) else 0.0)
            distance_errors.append(
                float(
                    (abs(predicted_point[0] - cell.point[0]) + abs(predicted_point[1] - cell.point[1]))
                    / 2.0
                )
            )

        hidden_distances.append(float((raw_a["box_hidden"] - raw_b["box_hidden"]).norm()))
        projected_distances.append(
            float((raw_a["query_projected"] - raw_b["query_projected"]).norm())
        )
        probabilities_a = spatial_probabilities(raw_a["logits"])
        probabilities_b = spatial_probabilities(raw_b["logits"])
        masses = counterfactual_masses(
            probabilities_a, probabilities_b, raw_a["soft_target"], raw_b["soft_target"]
        )
        p_aa_list.append(masses["raw"]["p_aa"])
        p_ab_list.append(masses["raw"]["p_ab"])
        p_bb_list.append(masses["raw"]["p_bb"])
        p_ba_list.append(masses["raw"]["p_ba"])
        margins.append(masses["raw"]["mean_margin"])
        cf_losses.append(
            float(
                bounded_pair_loss(
                    masses["p_aa"], masses["p_ab"], masses["p_bb"], masses["p_ba"]
                )["raw"]
            )
        )

    def mean(values):
        return float(sum(values) / len(values)) if values else None

    return {
        "pairs": len(triples),
        "mean_point_ce": mean(point_ces),
        "mean_target_cell_probability": mean(cell_probs),
        "target_cell_top1_rate": mean(top1),
        "point_inside_rate": mean(inside),
        "mean_normalized_point_error": mean(distance_errors),
        "mean_spatial_entropy": mean(entropies),
        "mean_max_non_target_probability": mean(non_target),
        "mean_p_aa": mean(p_aa_list),
        "mean_p_ab": mean(p_ab_list),
        "mean_p_bb": mean(p_bb_list),
        "mean_p_ba": mean(p_ba_list),
        "mean_own_mass": mean(
            [0.5 * (a + b) for a, b in zip(p_aa_list, p_bb_list)]
        ),
        "mean_cross_mass": mean(
            [0.5 * (a + b) for a, b in zip(p_ab_list, p_ba_list)]
        ),
        "mean_probability_margin": mean(margins),
        "mean_cf_loss": mean(cf_losses),
        "mean_ab_hidden_distance": mean(hidden_distances),
        "mean_ab_projected_query_distance": mean(projected_distances),
    }


def main() -> int:
    cfg, payload = load_point_config()
    grid = int(cfg["dense_grounding"]["grid"])
    triples = h0r_pairs(canonical_train_pairs(payload), int(cfg["h0r"]["pairs"]))
    h0r = json.loads(H0R_JSON.read_text(encoding="utf-8"))

    set_seed(int(cfg["seed"]))
    runtime = build_runtime(cfg, verbose=True)
    first = data_mod.to_sample(triples[0][1])
    image = first.image_rgb()
    features, _cached = runtime.features_for(first, image)
    runtime.install_dense_head(in_channels=int(feature_tensor_for_grid(features, grid).shape[1]))
    runtime.freeze_for_point_objective()

    before = _state(runtime, triples, grid)

    # gradient coverage of the corrected objective at clean initialization
    sample_a = data_mod.to_sample(triples[0][1])
    sample_b = data_mod.to_sample(triples[0][2])
    cell_a = target_cell(sample_a.target_mask(), grid)
    cell_b = target_cell(sample_b.target_mask(), grid)

    def _forward_losses():
        spatial = feature_tensor_for_grid(features, grid).to(runtime.device)
        batch_a = build_box_query_batch(
            runtime.processor, runtime.tokenizer, image, sample_a.instruction_zh,
            sample_a.reasoning_zh, int(runtime.box_setup.box_token_id),
            int(runtime.model.seg_token_id), append_eos=True,
        ).to(runtime.device)
        batch_b = build_box_query_batch(
            runtime.processor, runtime.tokenizer, image, sample_b.instruction_zh,
            sample_b.reasoning_zh, int(runtime.box_setup.box_token_id),
            int(runtime.model.seg_token_id), append_eos=True,
        ).to(runtime.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            _la, hidden_a, _sa = box_query_forward(runtime.model.qwen, batch_a)
            heatmap_a = runtime.model.dense_head(hidden_a, spatial)
            _lb, hidden_b, _sb = box_query_forward(runtime.model.qwen, batch_b)
            heatmap_b = runtime.model.dense_head(hidden_b, spatial)
            point = point_cross_entropy(heatmap_a[0], cell_a.index) + point_cross_entropy(
                heatmap_b[0], cell_b.index
            )
            target_a = downsample_target_mask(sample_a.target_mask(), grid).to(runtime.device)
            target_b = downsample_target_mask(sample_b.target_mask(), grid).to(runtime.device)
            masses = counterfactual_masses(
                spatial_probabilities(heatmap_a[0]), spatial_probabilities(heatmap_b[0]),
                target_a, target_b,
            )
            cf = bounded_pair_loss(masses["p_aa"], masses["p_ab"], masses["p_bb"], masses["p_ba"])
        return point, cf["loss"]

    point_loss, _cf = _forward_losses()
    point_gradients = _gradient_groups(runtime, point_loss)
    del point_loss, _cf
    _point, cf_loss = _forward_losses()
    cf_gradients = _gradient_groups(runtime, cf_loss)
    del _point, cf_loss

    features_again, _cached = runtime.features_for(first, image)
    feature_delta = float(
        (
            feature_tensor_for_grid(features, grid).detach()
            - feature_tensor_for_grid(features_again, grid).detach()
        )
        .abs()
        .max()
    )

    load_report = load_checkpoint(Path(h0r["checkpoint"]["path"]), runtime.model)
    after = _state(runtime, triples, grid)

    history = h0r["history"]
    report = {
        "_doc": (
            "Task 6H.1 section 14. Focused H0-R audit of the corrected objective: gradient coverage, "
            "target-cell probability / entropy / non-target probability evolution, point-CE and "
            "bounded-L_cf trends, argmax-vs-GT-point error, A/B query geometry, target-index "
            "bookkeeping and shared-feature bit identity. No architecture or loss change."
        ),
        "task": "6H.1",
        "h0r_verdict": h0r["verdict"],
        "h0r_final": {
            "point_inside_count": h0r["final"]["records"]["point_inside_count"],
            "record_count": h0r["final"]["records"]["count"],
            "paired_point_selection_pass": h0r["final"]["pair"]["paired_point_selection_pass"],
            "pair_ranking_pass": h0r["final"]["pair"]["pair_ranking_pass"],
            "mean_own_mass": h0r["final"]["pair"]["mean_own_mass"],
            "mean_cross_mass": h0r["final"]["pair"]["mean_cross_mass"],
            "mean_abs_logit": h0r["final"]["mean_abs_logit"],
            "gate": h0r["final"]["gate"],
        },
        "before_training": before,
        "after_training": after,
        "trends": {
            "point_ce": {"before": before["mean_point_ce"], "after": after["mean_point_ce"],
                         "decreased": bool(after["mean_point_ce"] < before["mean_point_ce"])},
            "target_cell_probability": {
                "before": before["mean_target_cell_probability"],
                "after": after["mean_target_cell_probability"],
            },
            "point_inside_rate": {
                "before": before["point_inside_rate"],
                "after": after["point_inside_rate"],
            },
            "normalized_point_error": {
                "before": before["mean_normalized_point_error"],
                "after": after["mean_normalized_point_error"],
            },
            "spatial_entropy": {
                "before": before["mean_spatial_entropy"],
                "after": after["mean_spatial_entropy"],
            },
            "max_non_target_probability": {
                "before": before["mean_max_non_target_probability"],
                "after": after["mean_max_non_target_probability"],
            },
            "probability_margin": {
                "before": before["mean_probability_margin"],
                "after": after["mean_probability_margin"],
            },
            "cf_loss": {"before": before["mean_cf_loss"], "after": after["mean_cf_loss"]},
            "history_first": history[0]["losses"] if history else None,
            "history_last": history[-1]["losses"] if history else None,
        },
        "gradients": {"point_ce": point_gradients, "bounded_cf": cf_gradients},
        "pair_identity": {
            "pairs": len(triples),
            "target_indices_a": [target_cell(data_mod.to_sample(ra).target_mask(), grid).index
                                 for _p, ra, _rb in triples],
            "target_indices_b": [target_cell(data_mod.to_sample(rb).target_mask(), grid).index
                                 for _p, _ra, rb in triples],
            "indices_differ_for_all_pairs": all(
                target_cell(data_mod.to_sample(ra).target_mask(), grid).index
                != target_cell(data_mod.to_sample(rb).target_mask(), grid).index
                for _p, ra, rb in triples
            ),
        },
        "shared_feature_max_abs_delta": feature_delta,
        "shared_feature_bit_identical": bool(feature_delta == 0.0),
        "checkpoint_load": load_report,
    }
    write_json(OUT, report)
    print(
        f"[task6h1:audit] point CE {before['mean_point_ce']:.3f} -> {after['mean_point_ce']:.3f}; "
        f"inside {before['point_inside_rate']} -> {after['point_inside_rate']}; p_target "
        f"{before['mean_target_cell_probability']:.5f} -> {after['mean_target_cell_probability']:.5f}; "
        f"entropy {before['mean_spatial_entropy']:.2f} -> {after['mean_spatial_entropy']:.2f}; "
        f"non-target max {before['mean_max_non_target_probability']:.5f} -> "
        f"{after['mean_max_non_target_probability']:.5f}; grads point/cf "
        f"{point_gradients['dense_head.query_proj']:.3f}/"
        f"{cf_gradients['dense_head.query_proj']:.3f}",
        flush=True,
    )
    print(f"[task6h1:audit] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
