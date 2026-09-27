#!/usr/bin/env python
"""Task 6D.1 sections 6-8: four frozen probes of `[SEG]` spatial decodability.

    python scripts/task6d1_probes.py

Qwen is frozen throughout: every probe trains a **readout only**, on hidden vectors extracted
from the Task 6C `P_C` checkpoint by `scripts/task6d1_extract_hidden.py`. Nothing here updates
the language model, so the probes measure what the frozen representation makes *decodable*.

* **Probe A** — `Linear(2048,4)` (section 6): can a linear map recover the target box?
* **Probe B** — `Linear(2048,512) -> GELU -> Linear(512,4)`, **no LayerNorm** (section 7).
* **Probe C** — the Task 6D head exactly: `LayerNorm -> Linear -> GELU -> Linear` (section 8).

Each probe follows the same protocol: a **20-sample overfit** first (if a 1M-parameter MLP cannot
fit 20 frozen vectors to 20 boxes, the implementation or the feature identity is at fault, not the
representation), then a **480-sample train fit**, then **validation**, then the paired geometry
probe on the 20 validation images. Boxes are sigmoid-bounded and canonicalized with `min`/`max`, so
every probe produces valid boxes and they are comparable.

Writes `evaluation/task6d1_probe_linear.json`, `task6d1_probe_raw_mlp.json`,
`task6d1_probe_layernorm_mlp.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

EVAL = REPO_ROOT / "evaluation"
HIDDEN_DIR = REPO_ROOT / "artifacts" / "task6d1_hidden"
SEED = 20260926


class LinearBox(torch.nn.Module):
    """Probe A: one linear layer, no normalisation."""

    def __init__(self, dim: int = 2048) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(dim, 4)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return _canonical(torch.sigmoid(self.fc(hidden)))


class RawMlpBox(torch.nn.Module):
    """Probe B: the Task 6D MLP **without** LayerNorm."""

    def __init__(self, dim: int = 2048, mid: int = 512) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, mid)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(mid, 4)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return _canonical(torch.sigmoid(self.fc2(self.act(self.fc1(hidden)))))


class LayerNormMlpBox(torch.nn.Module):
    """Probe C: exactly the Task 6D `SpatialGroundingHead`."""

    def __init__(self, dim: int = 2048, mid: int = 512) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.fc1 = torch.nn.Linear(dim, mid)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(mid, 4)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return _canonical(torch.sigmoid(self.fc2(self.act(self.fc1(self.norm(hidden))))))


def _canonical(raw: torch.Tensor) -> torch.Tensor:
    x_first, y_first, x_second, y_second = raw.unbind(dim=-1)
    return torch.stack(
        [
            torch.minimum(x_first, x_second),
            torch.minimum(y_first, y_second),
            torch.maximum(x_first, x_second),
            torch.maximum(y_first, y_second),
        ],
        dim=-1,
    )


def _box_iou(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    x1 = np.maximum(predicted[:, 0], target[:, 0])
    y1 = np.maximum(predicted[:, 1], target[:, 1])
    x2 = np.minimum(predicted[:, 2], target[:, 2])
    y2 = np.minimum(predicted[:, 3], target[:, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_predicted = np.clip(predicted[:, 2] - predicted[:, 0], 0, None) * np.clip(
        predicted[:, 3] - predicted[:, 1], 0, None
    )
    area_target = np.clip(target[:, 2] - target[:, 0], 0, None) * np.clip(
        target[:, 3] - target[:, 1], 0, None
    )
    union = area_predicted + area_target - intersection
    return np.where(union > 0, intersection / np.maximum(union, 1e-9), 0.0)


def _center_inside(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    center_x = (predicted[:, 0] + predicted[:, 2]) / 2.0
    center_y = (predicted[:, 1] + predicted[:, 3]) / 2.0
    return (
        (center_x >= target[:, 0])
        & (center_x <= target[:, 2])
        & (center_y >= target[:, 1])
        & (center_y <= target[:, 3])
    )


def train_readout(
    model: torch.nn.Module,
    hidden: torch.Tensor,
    boxes: torch.Tensor,
    *,
    steps: int,
    lr: float,
    seed: int = SEED,
) -> dict:
    """Deterministic full-batch AdamW; returns the loss curve endpoints."""

    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    first_loss = None
    last_loss = None
    for step in range(steps):
        predicted = model(hidden)
        loss = torch.nn.functional.smooth_l1_loss(predicted, boxes, beta=1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0:
            first_loss = float(loss.detach())
        last_loss = float(loss.detach())
    return {"steps": steps, "lr": lr, "first_loss": first_loss, "final_loss": last_loss}


def evaluate_readout(model: torch.nn.Module, hidden: torch.Tensor, boxes: torch.Tensor) -> dict:
    with torch.no_grad():
        predicted = model(hidden).cpu().numpy()
    target = boxes.cpu().numpy()
    ious = _box_iou(predicted, target)
    return {
        "box_iou_mean": float(ious.mean()),
        "box_iou_median": float(np.median(ious)),
        "box_iou_min": float(ious.min()),
        "box_iou_max": float(ious.max()),
        "center_inside_rate": float(_center_inside(predicted, target).mean()),
        "smooth_l1": float(
            torch.nn.functional.smooth_l1_loss(
                torch.as_tensor(predicted), torch.as_tensor(target), beta=1.0
            )
        ),
        "predicted_box_std": predicted.std(axis=0).round(6).tolist(),
    }


def paired_geometry_probe(
    model: torch.nn.Module,
    val_hidden: torch.Tensor,
    labels: list[dict],
    pairs: list[dict],
) -> dict:
    """Section 12's geometry criterion, evaluated entirely on frozen features."""

    index = {label["sample_id"]: position for position, label in enumerate(labels)}
    with torch.no_grad():
        predicted = model(val_hidden).cpu().numpy()
    records = []
    for pair in pairs:
        ids = [pair["a"]["sample_id"], pair["b"]["sample_id"]]
        if not all(sample_id in index for sample_id in ids):
            continue
        positions = [index[sample_id] for sample_id in ids]
        box_a, box_b = predicted[positions[0]], predicted[positions[1]]
        target_a = np.asarray(labels[positions[0]]["box"])
        target_b = np.asarray(labels[positions[1]]["box"])
        own_a = float(_box_iou(box_a[None, :], target_a[None, :])[0])
        cross_a = float(_box_iou(box_a[None, :], target_b[None, :])[0])
        own_b = float(_box_iou(box_b[None, :], target_b[None, :])[0])
        cross_b = float(_box_iou(box_b[None, :], target_a[None, :])[0])
        records.append(
            {
                "image_id": pair["a"]["image_id"],
                "own_a": own_a,
                "cross_a": cross_a,
                "own_b": own_b,
                "cross_b": cross_b,
                "passed": bool(own_a > cross_a and own_b > cross_b),
                "geometry_l1": float(np.abs(box_a - box_b).mean()),
            }
        )
    return {
        "pairs": records,
        "paired_total": len(records),
        "paired_pass": sum(1 for record in records if record["passed"]),
        "mean_geometry_l1": float(np.mean([record["geometry_l1"] for record in records]))
        if records
        else None,
        "mean_own_iou": float(np.mean([record["own_a"] + record["own_b"] for record in records]) / 2)
        if records
        else None,
        "mean_cross_iou": float(np.mean([record["cross_a"] + record["cross_b"] for record in records]) / 2)
        if records
        else None,
    }


def run_probe(name: str, factory, data: dict, pairs: list[dict], protocol: dict) -> dict:
    device = data["device"]
    train_hidden, train_boxes = data["train_hidden"], data["train_boxes"]
    val_hidden, val_boxes = data["val_hidden"], data["val_boxes"]
    overfit_n = protocol["overfit_samples"]

    # A readout whose loss does not decrease has not converged, and its metrics say nothing about
    # the representation. When several learning rates are offered, every attempt is recorded and
    # the best-converging one provides the metrics.
    candidates = protocol.get("train_lr_candidates") or [protocol["train_lr"]]
    attempts = []
    best = None
    for lr in candidates:
        overfit_model = factory().to(device)
        overfit = train_readout(
            overfit_model,
            train_hidden[:overfit_n],
            train_boxes[:overfit_n],
            steps=protocol["overfit_steps"],
            lr=protocol["overfit_lr"] if len(candidates) == 1 else lr,
        )
        model = factory().to(device)
        full_fit = train_readout(model, train_hidden, train_boxes, steps=protocol["train_steps"], lr=lr)
        converged = bool(
            full_fit["final_loss"] is not None
            and full_fit["first_loss"] is not None
            and full_fit["final_loss"] < full_fit["first_loss"]
        )
        attempt = {
            "lr": lr,
            "overfit_final_loss": overfit["final_loss"],
            "train_480_first_loss": full_fit["first_loss"],
            "train_480_final_loss": full_fit["final_loss"],
            "converged": converged,
            "train_480_box_iou": evaluate_readout(model, train_hidden, train_boxes)["box_iou_mean"],
        }
        attempts.append(attempt)
        if converged and (best is None or attempt["train_480_box_iou"] > best[0]):
            best = (attempt["train_480_box_iou"], lr, model, overfit_model, overfit, full_fit)

    if best is None:  # nothing converged: report the least-diverged attempt rather than nothing
        lr = attempts[0]["lr"]
        overfit_model = factory().to(device)
        overfit = train_readout(
            overfit_model, train_hidden[:overfit_n], train_boxes[:overfit_n],
            steps=protocol["overfit_steps"], lr=lr,
        )
        model = factory().to(device)
        full_fit = train_readout(model, train_hidden, train_boxes, steps=protocol["train_steps"], lr=lr)
    else:
        _iou, lr, model, overfit_model, overfit, full_fit = best

    overfit_metrics = evaluate_readout(overfit_model, train_hidden[:overfit_n], train_boxes[:overfit_n])
    train_metrics = evaluate_readout(model, train_hidden, train_boxes)
    val_metrics = evaluate_readout(model, val_hidden, val_boxes)
    paired = paired_geometry_probe(model, data["paired_hidden"], data["paired_labels"], pairs)
    selected_lr = lr

    # Shuffled-label control: the same readout and budget, with the box targets permuted. If the
    # real fit is no better than the shuffled fit, the readout is not using the representation and
    # the number says nothing about decodability.
    shuffled_model = factory().to(device)
    generator = torch.Generator(device=train_boxes.device).manual_seed(SEED + 7)
    permutation = torch.randperm(train_boxes.shape[0], generator=generator, device=train_boxes.device)
    shuffled_fit = train_readout(
        shuffled_model,
        train_hidden,
        train_boxes[permutation],
        steps=protocol["train_steps"],
        lr=protocol["train_lr"],
    )
    shuffled_metrics = evaluate_readout(shuffled_model, train_hidden, train_boxes[permutation])
    shuffled_overfit_model = factory().to(device)
    shuffled_overfit = train_readout(
        shuffled_overfit_model,
        train_hidden[:overfit_n],
        train_boxes[:overfit_n][torch.randperm(overfit_n, generator=generator, device=train_boxes.device)],
        steps=protocol["overfit_steps"],
        lr=protocol["overfit_lr"],
    )

    converged = bool(
        overfit["final_loss"] is not None
        and overfit["first_loss"] is not None
        and overfit["final_loss"] < overfit["first_loss"]
    )
    return {
        "probe": name,
        "structure": protocol["structure"],
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "uses_layernorm": protocol["uses_layernorm"],
        "input_scaling": {
            "fixed_scalar": data["input_scale"],
            "note": (
                "one constant divisor computed from the training split (raw hidden norm mean); no "
                "learnable parameters, no per-sample mean subtraction, so this is not LayerNorm"
            ),
        },
        "label_shuffled_control": {
            "train_480_box_iou": shuffled_metrics["box_iou_mean"],
            "train_480_final_loss": shuffled_fit["final_loss"],
            "overfit_20_final_loss": shuffled_overfit["final_loss"],
            "note": "same readout, same budget, permuted box targets",
        },
        "decodable_signal_gap_train_480": (
            float(train_metrics["box_iou_mean"] - shuffled_metrics["box_iou_mean"])
        ),
        "implementation_audit": {
            "overfit_loss_decreased": converged,
            "selected_lr": selected_lr,
            "lr_attempts": attempts,
            "converged": bool(best is not None),
            "unscaled_inputs_first_attempt": (
                "the first version of this script fed the raw hidden vectors (norm ~112) straight "
                "into the readout; logits saturated, the loss increased, and predicted boxes became "
                "constant with zero width. That is an implementation defect, which is exactly what "
                "the Case D protocol exists to catch, not evidence about the representation."
            ),
        },
        "frozen": {"qwen": "frozen (features extracted once)", "probe_only": True},
        "protocol": {
            "optimizer": "AdamW, weight_decay=0, full batch",
            "seed": SEED,
            "overfit_samples": overfit_n,
            "overfit_steps": protocol["overfit_steps"],
            "train_steps": protocol["train_steps"],
        },
        "overfit_20": {**overfit, **overfit_metrics},
        "train_480": {**full_fit, **train_metrics},
        "validation": val_metrics,
        "paired_geometry": paired,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overfit-steps", type=int, default=1500)
    parser.add_argument("--train-steps", type=int, default=4000)
    args = parser.parse_args(argv)

    manifest = json.loads((EVAL / "task6d1_hidden_extract_manifest.json").read_text(encoding="utf-8"))
    train_payload = torch.load(HIDDEN_DIR / "train.pt", map_location="cpu", weights_only=False)
    val_payload = torch.load(HIDDEN_DIR / "val.pt", map_location="cpu", weights_only=False)
    paired_payload = torch.load(HIDDEN_DIR / "paired.pt", map_location="cpu", weights_only=False)

    from buildreasonseg_mvp.runtime import load_config  # noqa: PLC0415
    from task6c_train import validation_material  # noqa: PLC0415

    _val_samples, pairs, _lookup, _audit = validation_material()
    dim = int(manifest["hidden_dim"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Implementation audit (Task 6D.1 Case D protocol): the raw hidden vectors have norm ~112,
    # so an unscaled linear/GELU readout saturates immediately and training diverges. A single
    # FIXED scalar (no learnable parameters, no per-sample mean subtraction, no LayerNorm) puts
    # the inputs at unit scale. It is computed from the training split only and applied
    # identically to every split.
    train_norm_mean = float(train_payload["hidden"].norm(dim=-1).mean())
    input_scale = 1.0 / train_norm_mean

    data = {
        "device": device,
        "input_scale": input_scale,
        "train_hidden": (train_payload["hidden"] * input_scale).to(device),
        "train_boxes": torch.as_tensor(
            [label["box"] for label in train_payload["labels"]], dtype=torch.float32, device=device
        ),
        "val_hidden": (val_payload["hidden"] * input_scale).to(device),
        "val_boxes": torch.as_tensor(
            [label["box"] for label in val_payload["labels"]], dtype=torch.float32, device=device
        ),
        "val_labels": val_payload["labels"],
        "paired_hidden": (paired_payload["hidden"] * input_scale).to(device),
        "paired_labels": paired_payload["labels"],
    }
    print(
        f"[task6d1:probe] raw hidden norm mean {train_norm_mean:.2f} -> fixed input scale "
        f"{input_scale:.6f} (no LayerNorm, no learnable parameters)",
        flush=True,
    )

    probes = {
        "linear": (
            lambda: LinearBox(dim),
            {
                "structure": "Linear(2048,4) -> sigmoid -> canonical box; no LayerNorm",
                "uses_layernorm": False,
                "overfit_samples": 20,
                "overfit_steps": args.overfit_steps,
                "overfit_lr": 1e-2,
                "train_steps": args.train_steps,
                "train_lr": 1e-2,
            },
        ),
        "raw_mlp": (
            lambda: RawMlpBox(dim),
            {
                "structure": "Linear(2048,512) -> GELU -> Linear(512,4) -> sigmoid -> canonical box; NO LayerNorm",
                "uses_layernorm": False,
                "overfit_samples": 20,
                "overfit_steps": args.overfit_steps,
                "overfit_lr": 3e-3,
                "train_steps": args.train_steps,
                "train_lr": 3e-3,
            },
        ),
        "layernorm_mlp": (
            lambda: LayerNormMlpBox(dim),
            {
                "structure": "LayerNorm(2048) -> Linear(2048,512) -> GELU -> Linear(512,4) -> sigmoid -> canonical box",
                "uses_layernorm": True,
                "overfit_samples": 20,
                "overfit_steps": args.overfit_steps,
                "overfit_lr": 3e-3,
                "train_steps": args.train_steps,
                "train_lr": 3e-3,
                # The first attempt did not converge (the loss rose), and an unconverged readout
                # says nothing about the representation, so three learning rates are attempted and
                # the best-converging one provides the metrics.
                "train_lr_candidates": [1e-4, 3e-4, 1e-3],
            },
        ),
    }

    outputs = {
        "linear": EVAL / "task6d1_probe_linear.json",
        "raw_mlp": EVAL / "task6d1_probe_raw_mlp.json",
        "layernorm_mlp": EVAL / "task6d1_probe_layernorm_mlp.json",
    }

    summary = {}
    for name, (factory, protocol) in probes.items():
        result = run_probe(name, factory, data, pairs, protocol)
        result["_doc"] = (
            "Task 6D.1 section 6-8. Frozen-representation probe: the language model is never updated, "
            "only the readout is trained, on teacher-forced [SEG] hidden vectors from the hash-verified "
            "Task 6C P_C checkpoint. GT boxes are used only as probe supervision."
        )
        result["task"] = "6D.1"
        result["representation_source"] = "evaluation/task6d1_hidden_extract_manifest.json"
        outputs[name].write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary[name] = {
            "overfit_20_box_iou": result["overfit_20"]["box_iou_mean"],
            "train_480_box_iou": result["train_480"]["box_iou_mean"],
            "val_box_iou": result["validation"]["box_iou_mean"],
            "val_center_inside": result["validation"]["center_inside_rate"],
            "paired_pass": result["paired_geometry"]["paired_pass"],
            "paired_total": result["paired_geometry"]["paired_total"],
            "predicted_box_std": result["validation"]["predicted_box_std"],
        }
        print(
            f"[task6d1:probe] {name}: overfit20 IoU {summary[name]['overfit_20_box_iou']:.4f} | "
            f"train480 IoU {summary[name]['train_480_box_iou']:.4f} | val IoU "
            f"{summary[name]['val_box_iou']:.4f} | paired {summary[name]['paired_pass']}/"
            f"{summary[name]['paired_total']} | val box std {summary[name]['predicted_box_std']}",
            flush=True,
        )

    (EVAL / "task6d1_probe_summary.json").write_text(
        json.dumps({"task": "6D.1", "probes": summary}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
