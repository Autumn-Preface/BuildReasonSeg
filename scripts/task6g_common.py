"""Task 6G shared script helpers: config, paired material, gates, checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.runtime import load_config  # noqa: E402

from task6c6_common import EVAL, write_json  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "mvp" / "task6g_dense_grounding.yaml"
TASK6C_SUBSETS = EVAL / "task6c_subset_ids.json"
GRID_ORACLE = EVAL / "task6g_grid_oracle.json"
MANIFEST = EVAL / "task6g_checkpoint_manifest.json"


def load_dense_config(grid_override: int | None = None) -> tuple[dict, dict]:
    """The Task 6G config with `dense_grounding.grid` set from the grid oracle (never twice)."""

    cfg = load_config(CONFIG)
    payload = json.loads(TASK6C_SUBSETS.read_text(encoding="utf-8"))
    oracle = json.loads(GRID_ORACLE.read_text(encoding="utf-8"))
    selected = oracle["selection"]["selected_grid"]
    if selected is None:
        raise SystemExit("grid oracle selection failed; DENSE_GRID_POINT_PATH_INADEQUATE")
    grid = int(grid_override or selected)
    cfg["dense_grounding"]["grid"] = grid
    cfg["_grid_oracle_selection"] = oracle["selection"]
    cfg["_grid_oracle_candidates"] = {
        key: {
            "strict_miou": entry["strict_miou"],
            "paired_pass": entry["paired_pass"],
            "paired_total": entry["paired_total"],
            "mean_displacement_512px": entry["mean_displacement_512px"],
        }
        for key, entry in oracle["candidates"].items()
    }
    cfg["_subset_source"] = str(TASK6C_SUBSETS.relative_to(REPO_ROOT).as_posix())
    cfg["_subset_pairs"] = len(payload["P"]["pairs"])
    return cfg, payload


def paired_records(payload: dict) -> list[dict]:
    """The 480 Task 6C `P` records as labelled pair dictionaries."""

    train = {record["sample_id"]: record for record in data_mod.read_records("train")}
    pairs = []
    for pair in payload["P"]["pairs"]:
        record_a = train[pair["a"]]
        record_b = train[pair["b"]]
        pairs.append(
            {
                "image_id": str(pair["image_id"]),
                "a": str(pair["a"]),
                "b": str(pair["b"]),
                "record_a": record_a,
                "record_b": record_b,
                "a_level": int(pair["a_level"]),
                "b_level": int(pair["b_level"]),
                "a_query_type": str(pair["a_query_type"]),
                "b_query_type": str(pair["b_query_type"]),
            }
        )
    return pairs


def flat_paired_records(payload: dict) -> list[dict]:
    records = []
    for pair in paired_records(payload):
        records.extend([pair["record_a"], pair["record_b"]])
    return records


def g0_selection(payload: dict, images: int = 10) -> tuple[list[dict], list[dict]]:
    """Section 11: the same deterministic 20-record / 10 paired-image style as Tasks 6E/6F."""

    pairs = paired_records(payload)[:images]
    records = []
    for pair in pairs:
        records.extend([pair["record_a"], pair["record_b"]])
    return records, pairs


def validation_pair_dicts(pairs) -> list[dict]:
    """The fixed 20 paired validation images as labelled pair dictionaries."""

    result = []
    for pair in pairs:
        record_a, record_b = pair["a"], pair["b"]
        result.append(
            {
                "image_id": str(record_a["image_id"]),
                "a": str(record_a["sample_id"]),
                "b": str(record_b["sample_id"]),
                "record_a": record_a,
                "record_b": record_b,
                "a_level": int(record_a["level"]),
                "b_level": int(record_b["level"]),
                "a_query_type": str(record_a["query_type"]),
                "b_query_type": str(record_b["query_type"]),
            }
        )
    return result


def optimizer_and_scheduler(runtime, *, total_steps: int, warmup_steps: int):
    """AdamW over the Task 6G trainable set + cosine schedule over the real step budget."""

    import torch

    cfg = runtime.cfg["optimizer"]
    groups = runtime.model.trainable_parameter_groups(
        lora_lr=float(cfg["lora_lr"]),
        head_lr=float(cfg["token_lr"]),
        weight_decay=float(cfg["weight_decay"]),
        decoder_lr=float(cfg["decoder_lr"]),
        token_lr=float(cfg["token_lr"]),
    )
    optimizer = torch.optim.AdamW(groups, betas=tuple(cfg["betas"]))
    scheduler = runtime.build_scheduler_for(
        optimizer,
        max_steps=max(1, int(total_steps)),
        schedule_cfg={"lr_schedule": cfg["lr_schedule"], "warmup_steps": int(warmup_steps)},
    )
    return optimizer, scheduler, groups


def optimizer_coverage(groups) -> dict:
    return [
        {
            "name": group.get("name"),
            "lr": group.get("lr"),
            "weight_decay": group.get("weight_decay"),
            "tensors": len(group.get("params", [])),
            "parameters": sum(parameter.numel() for parameter in group.get("params", [])),
        }
        for group in groups
    ]


def lr_values(optimizer) -> list[float]:
    return [float(group["lr"]) for group in optimizer.param_groups]


def checkpoint_dir(cfg: dict, stage: str) -> Path:
    directory = Path(cfg["paths"]["checkpoints"]) / stage
    if not directory.is_absolute():
        directory = REPO_ROOT / directory
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_stage_checkpoint(cfg: dict, stage: str, name: str, model, step: int, metrics: dict) -> dict:
    from buildreasonseg_mvp.checkpointing import save_checkpoint, sha256_file

    directory = checkpoint_dir(cfg, stage)
    path = directory / f"{name}.pt"
    info = save_checkpoint(path, model, step, metrics=metrics, config=cfg)
    info["sha256"] = sha256_file(path)
    info["stage"] = stage
    info["name"] = name
    info["relative_path"] = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    manifest = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest.setdefault(
        "_doc",
        (
            "Task 6G: hashes of every Task 6G checkpoint. The checkpoint files themselves stay "
            "gitignored (artifacts/); only this manifest is tracked."
        ),
    )
    manifest.setdefault("task", "6G")
    manifest.setdefault("checkpoints", {})
    manifest["checkpoints"][f"{stage}/{name}"] = info
    write_json(MANIFEST, manifest)
    return info


def gate_report(metrics: dict, gate: dict, kind: str) -> dict:
    """Section 11 (G0) and section 14 (G1) gates, expressed once."""

    if kind == "g0":
        checks = {
            "point_inside_own_ge": int(metrics["point_inside_own"]) >= int(gate["point_inside_own_min"]),
            "paired_point_selection_ge": int(metrics["paired"]["paired_point_selection_pass"])
            >= int(gate["paired_point_selection_min"]),
            "heatmap_dice_ge": float(metrics["heatmap_dice"] or 0.0) >= float(gate["heatmap_dice_min"]),
            "paired_points_distinct": int(metrics["paired"]["pairs_with_distinct_points"])
            >= int(metrics["paired"]["paired_total"]),
            "no_gt_leakage": bool(metrics.get("no_gt_leakage", False)),
        }
    elif kind == "g1":
        checks = {
            "point_inside_target_ge": float(metrics["val"]["point_inside_target_rate"] or 0.0)
            >= float(gate["point_inside_target_min"]),
            "paired_point_selection_ge": int(metrics["paired"]["paired_point_selection_pass"])
            >= int(gate["paired_point_selection_min"]),
            "heatmap_dice_ge": float(metrics["val"]["mean_heatmap_dice"] or 0.0)
            >= float(gate["heatmap_dice_min"]),
        }
    else:
        raise ValueError(f"unknown gate kind {kind!r}")
    return {
        "kind": kind,
        "checks": checks,
        "passed": all(checks.values()),
        "gate": {key: value for key, value in gate.items()},
    }
