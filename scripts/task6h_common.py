"""Task 6H shared script helpers: config, canonical pairs, gates, checkpoints."""

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
from buildreasonseg_mvp.counterfactual import build_canonical_pairs  # noqa: E402
from buildreasonseg_mvp.runtime import load_config  # noqa: E402

from task6c6_common import EVAL, write_json  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "mvp" / "task6h_counterfactual_grounding.yaml"
TASK6C_SUBSETS = EVAL / "task6c_subset_ids.json"
PAIR_MANIFEST = EVAL / "task6h_pair_manifest.json"
MANIFEST = EVAL / "task6h_checkpoint_manifest.json"


def load_pair_config() -> tuple[dict, dict]:
    """The Task 6H config plus the Task 6C `P` payload."""

    cfg = load_config(CONFIG)
    payload = json.loads(TASK6C_SUBSETS.read_text(encoding="utf-8"))
    cfg["_subset_source"] = str(TASK6C_SUBSETS.relative_to(REPO_ROOT).as_posix())
    cfg["_pair_source"] = "Task 6C P payload pairs (deterministic order)"
    return cfg, payload


def train_records_by_id() -> dict:
    return {record["sample_id"]: record for record in data_mod.read_records("train")}


def canonical_train_pairs(payload: dict) -> list:
    """The 240 canonical training pairs (assertions enforced inside the builder)."""

    return build_canonical_pairs(payload, train_records_by_id())


def pair_dicts_for_eval(triples: list) -> list[dict]:
    """Pair dictionaries in the shape the evaluator and gates expect."""

    rows = []
    for pair, record_a, record_b in triples:
        rows.append(
            {
                "image_id": pair.image_id,
                "a": pair.a,
                "b": pair.b,
                "target_a": pair.target_a,
                "target_b": pair.target_b,
                "record_a": record_a,
                "record_b": record_b,
                "a_level": int(record_a["level"]),
                "b_level": int(record_b["level"]),
                "a_query_type": str(record_a["query_type"]),
                "b_query_type": str(record_b["query_type"]),
            }
        )
    return rows


def h0_pairs(triples: list, pairs: int = 10) -> list:
    """Section 11: the same first 10 pairs as Task 6G G0 (deterministic)."""

    return triples[: int(pairs)]


def validation_pair_dicts(pairs) -> list[dict]:
    """The fixed 20 paired validation images as pair dictionaries (one pair per image)."""

    result = []
    for pair in pairs:
        record_a, record_b = pair["a"], pair["b"]
        result.append(
            {
                "image_id": str(record_a["image_id"]),
                "a": str(record_a["sample_id"]),
                "b": str(record_b["sample_id"]),
                "target_a": int(record_a["target_component_id"]),
                "target_b": int(record_b["target_component_id"]),
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
    """AdamW over the Task 6H trainable set + cosine over the real pair-step budget."""

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


def optimizer_coverage(groups) -> list[dict]:
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
            "Task 6H: hashes of every Task 6H checkpoint. The checkpoint files themselves stay "
            "gitignored (artifacts/); only this manifest is tracked."
        ),
    )
    manifest.setdefault("task", "6H")
    manifest.setdefault("checkpoints", {})
    manifest["checkpoints"][f"{stage}/{name}"] = info
    write_json(MANIFEST, manifest)
    return info


def gate_report(metrics: dict, gate: dict, kind: str) -> dict:
    """Section 12 (H0) and section 16 (H1) gates, expressed once."""

    if kind == "h0":
        checks = {
            "pair_ranking_ge": int(metrics["pair"]["pair_ranking_pass"]) >= int(gate["pair_ranking_min"]),
            "point_inside_own_ge": int(metrics["point_inside_own"]) >= int(gate["point_inside_own_min"]),
            "paired_point_selection_ge": int(metrics["pair"]["paired_point_selection_pass"])
            >= int(gate["paired_point_selection_min"]),
            "mean_margin_gt": float(metrics["pair"]["mean_own_minus_cross_margin"] or 0.0)
            > float(gate["mean_margin_min"]),
            "no_gt_leakage": bool(metrics.get("no_gt_leakage", False)),
        }
    elif kind == "h1":
        checks = {
            "paired_point_selection_ge": int(metrics["validation_pairs"]["paired_point_selection_pass"])
            >= int(gate["paired_point_selection_min"]),
            "paired_region_ranking_ge": int(metrics["validation_pairs"]["pair_ranking_pass"])
            >= int(gate["paired_region_ranking_min"]),
            "point_inside_target_ge": float(metrics["val"]["point_inside_target_rate"] or 0.0)
            >= float(gate["point_inside_target_min"]),
        }
    else:
        raise ValueError(f"unknown gate kind {kind!r}")
    return {
        "kind": kind,
        "checks": checks,
        "passed": all(checks.values()),
        "gate": {key: value for key, value in gate.items()},
    }
