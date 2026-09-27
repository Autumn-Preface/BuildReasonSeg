"""Task 6E shared script helpers: config, paired material, gates, checkpoints."""

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

CONFIG = REPO_ROOT / "configs" / "mvp" / "task6e_spatial_tokens.yaml"
ORACLE = EVAL / "task6e_quantized_oracle.json"
TASK6C_SUBSETS = EVAL / "task6c_subset_ids.json"


def load_spatial_config(bins_override: int | None = None) -> tuple[dict, dict]:
    """Config with `spatial_tokens.bins` set from the oracle audit (never hard-coded twice)."""

    cfg = load_config(CONFIG)
    payload = json.loads(TASK6C_SUBSETS.read_text(encoding="utf-8"))
    oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
    selected = oracle["selection"]["selected_bins"]
    if selected is None:
        raise SystemExit("oracle selection failed; QUANTIZED_BOX_REPRESENTATION_INADEQUATE")
    bins = int(bins_override or selected)
    cfg["spatial_tokens"]["bins"] = bins
    cfg["_oracle_selection"] = oracle["selection"]
    cfg["_subset_source"] = str(TASK6C_SUBSETS.relative_to(REPO_ROOT).as_posix())
    cfg["_subset_pairs"] = len(payload["P"]["pairs"])
    return cfg, payload


def paired_records(payload: dict) -> list[dict]:
    """The 480 Task 6C `P` records as pair dictionaries.

    Each entry is `{"image_id", "a", "b", "record_a", "record_b", "a_level", "b_level",
    "a_query_type", "b_query_type"}` where `a`/`b` are sample ids.
    """

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


def e0_selection(payload: dict, images: int = 10) -> tuple[list[dict], list[dict]]:
    """Deterministic E0 material: the first `images` same-image/different-target pairs."""

    pairs = paired_records(payload)[:images]
    records = []
    for pair in pairs:
        records.extend([pair["record_a"], pair["record_b"]])
    return records, pairs


def optimizer_and_scheduler(runtime, *, total_steps: int, warmup_steps: int):
    """AdamW over the spatial trainable set + cosine schedule over the real step budget."""

    cfg = runtime.cfg["optimizer"]
    groups = runtime.model.trainable_parameter_groups(
        lora_lr=float(cfg["lora_lr"]),
        head_lr=float(cfg["token_lr"]),
        weight_decay=float(cfg["weight_decay"]),
        decoder_lr=float(cfg["token_lr"]),
        token_lr=float(cfg["token_lr"]),
    )
    import torch

    optimizer = torch.optim.AdamW(groups, betas=tuple(cfg["betas"]))
    scheduler = runtime.build_scheduler_for(
        optimizer,
        max_steps=max(1, int(total_steps)),
        schedule_cfg={"lr_schedule": cfg["lr_schedule"], "warmup_steps": int(warmup_steps)},
    )
    return optimizer, scheduler


def lr_values(optimizer) -> list[float]:
    return [float(group["lr"]) for group in optimizer.param_groups]


def checkpoint_dir(cfg: dict, stage: str) -> Path:
    directory = Path(cfg["paths"]["checkpoints"]) / stage
    if not directory.is_absolute():
        directory = REPO_ROOT / directory
    directory.mkdir(parents=True, exist_ok=True)
    return directory


MANIFEST = EVAL / "task6e_checkpoint_manifest.json"


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
    manifest.setdefault("_doc", (
        "Task 6E: hashes of every Task 6E checkpoint. The checkpoint files themselves stay "
        "gitignored (artifacts/); only this manifest is tracked."
    ))
    manifest.setdefault("task", "6E")
    manifest.setdefault("checkpoints", {})
    manifest["checkpoints"][f"{stage}/{name}"] = info
    write_json(MANIFEST, manifest)
    return info


def same_image_instruction_divergence(paired_free: list[dict], pairs: list[dict]) -> dict:
    """Section 24: do two instructions on one image now emit different tokens/boxes?"""

    by_id = {record["sample_id"]: record for record in paired_free}
    rows = []
    for pair in pairs:
        a = by_id.get(str(pair["a"]))
        b = by_id.get(str(pair["b"]))
        if a is None or b is None:
            continue
        both_valid = bool(a["structural_valid"] and b["structural_valid"])
        token_differ = None
        box_l1 = None
        if both_valid:
            token_differ = a["predicted_codes"] != b["predicted_codes"]
            box_l1 = float(
                sum(abs(float(x) - float(y)) for x, y in zip(a["predicted_box"], b["predicted_box"])) / 4.0
            )
        rows.append(
            {
                "image_id": str(pair["image_id"]),
                "a": str(pair["a"]),
                "b": str(pair["b"]),
                "both_structural_valid": both_valid,
                "different_predicted_tokens": token_differ,
                "predicted_box_l1": box_l1,
                "a_codes": a["predicted_codes"],
                "b_codes": b["predicted_codes"],
            }
        )
    compared = [row for row in rows if row["both_structural_valid"]]
    return {
        "compared_pairs": len(compared),
        "pairs_with_different_tokens": sum(
            1 for row in compared if row["different_predicted_tokens"]
        ),
        "pairs_with_identical_boxes": sum(1 for row in compared if row["predicted_box_l1"] == 0.0),
        "mean_predicted_box_l1": (
            None
            if not compared
            else float(sum(row["predicted_box_l1"] for row in compared) / len(compared))
        ),
        "rows": rows,
    }


def gate_report(metrics: dict, gate: dict, kind: str) -> dict:
    """Section 10/13 gates, expressed once so E0 and E1 cannot drift apart."""

    if kind == "e0":
        checks = {
            "structural_valid_ge": int(metrics["structural_valid"]) >= int(gate["structural_valid_min"]),
            "exact_four_token_ge": int(metrics["exact_four_token_sequence"])
            >= int(gate["exact_four_token_min"]),
            "mean_box_iou_ge": float(metrics["mean_predicted_box_iou"] or 0.0)
            >= float(gate["mean_box_iou_min"]),
        }
    elif kind == "e1":
        checks = {
            "structural_valid_rate_ge": float(metrics["structural_valid_rate"])
            >= float(gate["structural_valid_rate_min"]),
            "geometry_paired_ge": int(metrics["paired"]["geometry_paired_pass"])
            >= int(gate["geometry_paired_min"]),
            "mean_box_iou_ge": float(metrics["mean_predicted_box_iou"] or 0.0)
            >= float(gate["mean_box_iou_min"]),
        }
    else:
        raise ValueError(f"unknown gate kind {kind!r}")
    return {
        "kind": kind,
        "checks": checks,
        "passed": all(checks.values()),
        "gate": {key: value for key, value in gate.items()},
    }
