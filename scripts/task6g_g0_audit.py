"""Task 6G G0 implementation audit diagnostic (section 11).

The G0 gate failed. Before declaring `DENSE_GROUNDING_IMPLEMENTATION_FAILED` this script checks the
implementation-faithfulness hypotheses on the recorded G0 checkpoint:

1. train/eval equivalence: the query hidden (and therefore the heatmap) at the [BOX] position must
   be bit-identical between the training-form sequence (prompt+[BOX]+reasoning+[SEG]+EOS) and the
   evaluation-form sequence (prompt+[BOX]) -- causal masking makes them equal, and any deviation is
   a defect;
2. heatmap logits magnitude: whether the dot-product field saturates the sigmoid (which would
   explain the stuck Dice term under the frozen recipe);
3. gradient magnitudes of the BCE and Dice terms, to document why the Dice signal is weak.

Writes `evaluation/task6g_g0_audit.json`.
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
from buildreasonseg_mvp.box_query import box_hidden_from_batch, box_query_forward, build_box_query_batch  # noqa: E402
from buildreasonseg_mvp.checkpointing import load_checkpoint  # noqa: E402
from buildreasonseg_mvp.dense_grounding import (  # noqa: E402
    downsample_target_mask,
    feature_tensor_for_grid,
    heatmap_loss,
)
from buildreasonseg_mvp.dense_eval import predict_heatmap  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, set_seed  # noqa: E402

from task6g_common import EVAL, g0_selection, load_dense_config, write_json  # noqa: E402

G0_JSON = EVAL / "task6g_g0_overfit.json"
OUT = EVAL / "task6g_g0_audit.json"


def main() -> int:
    g0 = json.loads(G0_JSON.read_text(encoding="utf-8"))
    cfg, payload = load_dense_config()
    grid = int(cfg["dense_grounding"]["grid"])
    checkpoint = g0["checkpoint"]["path"]

    records, pairs = g0_selection(payload, images=10)
    samples = [data_mod.to_sample(record) for record in records]

    set_seed(int(cfg["seed"]))
    runtime = build_runtime(cfg, verbose=True)
    probe_image = samples[0].image_rgb()
    probe_features, _cached = runtime.features_for(samples[0], probe_image)
    in_channels = int(feature_tensor_for_grid(probe_features, grid).shape[1])
    runtime.install_dense_head(in_channels=in_channels)
    load_checkpoint(Path(checkpoint), runtime.model)

    rows = []
    for sample in samples:
        image = sample.image_rgb()
        features, _cached = runtime.features_for(sample, image)
        spatial = feature_tensor_for_grid(features, grid).to(runtime.device)
        training_batch = build_box_query_batch(
            runtime.processor, runtime.tokenizer, image, sample.instruction_zh,
            sample.reasoning_zh, int(runtime.box_setup.box_token_id),
            int(runtime.model.seg_token_id), append_eos=True,
        ).to(runtime.device)
        runtime.set_visual_cache_key(str(sample.image_id))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            _logits, hidden_train, _seg = box_query_forward(runtime.model.qwen, training_batch)
            logits_train = runtime.model.dense_head(hidden_train, spatial)
        prediction = predict_heatmap(runtime, sample, with_heatmap=False)
        from buildreasonseg_mvp.dense_grounding import argmax_point_from_logits

        soft_target = downsample_target_mask(sample.target_mask(), grid)
        train_extraction = argmax_point_from_logits(logits_train[0].detach().float().cpu(), grid)
        losses = heatmap_loss(
            logits_train.unsqueeze(1).detach().float(),
            soft_target[None, None, :, :].to(logits_train.device),
        )
        rows.append(
            {
                "sample_id": str(sample.sample_id),
                "train_eval_point_identical": bool(
                    train_extraction["point"] == prediction["predicted_point"]
                ),
                "logits_min": float(logits_train.min()),
                "logits_max": float(logits_train.max()),
                "logits_mean": float(logits_train.mean()),
                "logits_std": float(logits_train.std()),
                "sigmoid_mean": float(torch.sigmoid(logits_train.float()).mean()),
                "train_bce": losses["bce_raw"],
                "train_dice_loss": losses["dice_raw"],
                "eval_point": prediction["predicted_point"],
                "gt_point": prediction["gt_point"],
                "point_inside_own": prediction["point_inside_own"],
                "error_512px": prediction["error_512px"],
                "peakiness": prediction["peakiness"],
            }
        )

    # gradient magnitudes of the two heatmap terms at this checkpoint, on the first sample
    sample = samples[0]
    image = sample.image_rgb()
    features, _cached = runtime.features_for(sample, image)
    spatial = feature_tensor_for_grid(features, grid).to(runtime.device)
    batch = build_box_query_batch(
        runtime.processor, runtime.tokenizer, image, sample.instruction_zh,
        sample.reasoning_zh, int(runtime.box_setup.box_token_id),
        int(runtime.model.seg_token_id), append_eos=True,
    ).to(runtime.device)
    runtime.set_visual_cache_key(str(sample.image_id))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        _logits, hidden, _seg = box_query_forward(runtime.model.qwen, batch)
        logits = runtime.model.dense_head(hidden, spatial).unsqueeze(1)
    soft_target = downsample_target_mask(sample.target_mask(), grid)[None, None, :, :].to(runtime.device)
    for name in ("bce", "dice"):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            _logits, hidden, _seg = box_query_forward(runtime.model.qwen, batch)
            logits = runtime.model.dense_head(hidden, spatial).unsqueeze(1)
        if name == "bce":
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), soft_target.float())
        else:
            loss = heatmap_loss(logits, soft_target)["dice"]
        runtime.model.zero_grad(set_to_none=True)
        loss.backward()
        head_norm = float(
            sum(
                parameter.grad.detach().float().norm() ** 2
                for parameter in runtime.model.dense_head.parameters()
                if parameter.grad is not None
            )
            ** 0.5
        )
        rows[0][f"{name}_grad_norm_dense_head"] = head_norm

    report = {
        "_doc": (
            "Task 6G section 11 implementation audit for the failed G0 gate. Checks train/eval "
            "equivalence of the query path, heatmap logits saturation and the two heatmap terms' "
            "gradient magnitudes on the recorded G0 checkpoint."
        ),
        "task": "6G",
        "checkpoint": checkpoint,
        "records": rows,
        "summary": {
            "train_eval_point_identical_all": all(row["train_eval_point_identical"] for row in rows),
            "mean_logits_min": float(sum(row["logits_min"] for row in rows) / len(rows)),
            "mean_logits_max": float(sum(row["logits_max"] for row in rows) / len(rows)),
            "mean_logits_std": float(sum(row["logits_std"] for row in rows) / len(rows)),
            "mean_sigmoid_mean": float(sum(row["sigmoid_mean"] for row in rows) / len(rows)),
            "mean_train_bce": float(sum(row["train_bce"] for row in rows) / len(rows)),
            "mean_train_dice_loss": float(sum(row["train_dice_loss"] for row in rows) / len(rows)),
            "bce_grad_norm_dense_head": rows[0].get("bce_grad_norm_dense_head"),
            "dice_grad_norm_dense_head": rows[0].get("dice_grad_norm_dense_head"),
        },
    }
    write_json(OUT, report)
    print(
        f"[task6g:audit] train/eval identical {report['summary']['train_eval_point_identical_all']}; "
        f"logits [{report['summary']['mean_logits_min']:.2f}, {report['summary']['mean_logits_max']:.2f}] "
        f"std {report['summary']['mean_logits_std']:.2f}; sigmoid mean "
        f"{report['summary']['mean_sigmoid_mean']:.4f}; bce grad "
        f"{report['summary']['bce_grad_norm_dense_head']:.3e} dice grad "
        f"{report['summary']['dice_grad_norm_dense_head']:.3e}",
        flush=True,
    )
    print(f"[task6g:audit] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
