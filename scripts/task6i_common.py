"""Task 6I shared script helpers: config, canonical pairs, gates, checkpoints."""

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
from task6h1_common import (  # noqa: E402
    canonical_train_pairs,
    h0r_pairs,
    lr_values,
    optimizer_and_scheduler,
    optimizer_coverage,
    pair_dicts_for_eval,
    validation_pair_dicts,
)

CONFIG = REPO_ROOT / "configs" / "mvp" / "task6i_visual_query_refinement.yaml"
TASK6C_SUBSETS = EVAL / "task6c_subset_ids.json"
CHECKPOINT_MANIFEST = EVAL / "task6i_checkpoint_manifest.json"

#: Section 8 I0 gate + section 10 I1 gate constants (also recorded in the config).
I0_GATE = {
    "point_inside_own_min": 18,
    "paired_point_selection_min": 9,
    "pair_ranking_min": 9,
    "max_mean_normalized_error": 0.08,
}
I1_GATE = {
    "paired_point_selection_min": 14,
    "pair_ranking_min": 14,
    "point_inside_target_min": 0.60,
}


def load_refinement_config() -> tuple[dict, dict]:
    """The Task 6I config plus the Task 6C `P` payload."""

    cfg = load_config(CONFIG)
    payload = json.loads(TASK6C_SUBSETS.read_text(encoding="utf-8"))
    cfg["_subset_source"] = str(TASK6C_SUBSETS.relative_to(REPO_ROOT).as_posix())
    cfg["_pair_source"] = "Task 6C P payload pairs (same canonical 240 pairs as Task 6H/6H.1)"
    return cfg, payload


def pair_step_budget(cfg: dict, epochs: int | None = None) -> dict:
    """Section 10: the pair-step budget and the matching scheduler horizon."""

    epochs = int(epochs or cfg["i1"]["epochs"])
    pairs = int(cfg["counterfactual"]["pairs_per_epoch"])
    return {
        "epochs": epochs,
        "pairs_per_epoch": pairs,
        "total_pair_steps": max(1, epochs * pairs),
        "max_pair_steps": int(cfg["i1"]["max_pair_steps"]),
        "scheduler_horizon": max(1, epochs * pairs),
    }


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
    if CHECKPOINT_MANIFEST.exists():
        manifest = json.loads(CHECKPOINT_MANIFEST.read_text(encoding="utf-8"))
    manifest.setdefault(
        "_doc",
        (
            "Task 6I: hashes of every Task 6I checkpoint. The checkpoint files themselves stay "
            "gitignored (artifacts/); only this manifest is tracked."
        ),
    )
    manifest.setdefault("task", "6I")
    manifest.setdefault("checkpoints", {})
    manifest["checkpoints"][f"{stage}/{name}"] = info
    write_json(CHECKPOINT_MANIFEST, manifest)
    return info


def gate_report(metrics: dict, gate: dict, kind: str) -> dict:
    """Section 8 (I0) and section 10 (I1) gates, expressed once."""

    if kind == "i0":
        checks = {
            "point_inside_own_ge": int(metrics["point_inside_own"]) >= int(gate["point_inside_own_min"]),
            "paired_point_selection_ge": int(metrics["pair"]["paired_point_selection_pass"])
            >= int(gate["paired_point_selection_min"]),
            "pair_ranking_ge": int(metrics["pair"]["pair_ranking_pass"]) >= int(gate["pair_ranking_min"]),
            "mean_normalized_error_below": float(metrics["mean_normalized_error"] or 1.0)
            < float(gate["max_mean_normalized_error"]),
        }
    elif kind == "i1":
        checks = {
            "paired_point_selection_ge": int(metrics["validation_pairs"]["paired_point_selection_pass"])
            >= int(gate["paired_point_selection_min"]),
            "pair_ranking_ge": int(metrics["validation_pairs"]["pair_ranking_pass"])
            >= int(gate["pair_ranking_min"]),
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
