"""Task 6F section 13 (recomputation): `[BOX]` representation diagnostics at the best F1 checkpoint.

Re-collects the query hiddens from the checkpoint recorded in `task6f_f1_training.json` and
rewrites `evaluation/task6f_representation.json` with the corrected global-mean centring.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.box_query_eval import box_record, representation_report  # noqa: E402
from buildreasonseg_mvp.checkpointing import load_checkpoint  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, set_seed  # noqa: E402

from task6c_train import validation_material  # noqa: E402
from task6f_common import (  # noqa: E402
    EVAL,
    load_box_config,
    validation_pair_dicts,
    write_json,
)

F1_JSON = EVAL / "task6f_f1_training.json"
OUT = EVAL / "task6f_representation.json"


def main() -> int:
    started = time.time()
    cfg, _payload = load_box_config()
    f1 = json.loads(F1_JSON.read_text(encoding="utf-8"))
    checkpoint = f1["selection"]["selected_checkpoint"]["path"]
    selected_epoch = int(f1["selection"]["selected_epoch"])

    val_samples, raw_pairs, _lookup, _audit = validation_material()
    pairs = validation_pair_dicts(raw_pairs)
    paired_samples = [
        data_mod.to_sample(pair[side]) for pair in pairs for side in ("record_a", "record_b")
    ]

    set_seed(int(cfg["seed"]))
    runtime = build_runtime(cfg, verbose=True)
    if bool(cfg.get("training", {}).get("visual_feature_cache", False)):
        runtime.install_visual_cache(
            enabled=True,
            max_images=int(cfg.get("training", {}).get("visual_feature_cache_max_images", 512)),
        )
    runtime.install_box_head(
        hidden_dim=int(cfg["box_query"]["head_hidden_dim"]),
        mid_dim=int(cfg["box_query"]["head_mid_dim"]),
    )
    load_report = load_checkpoint(Path(checkpoint), runtime.model)

    val_records = [box_record(runtime, sample, with_hidden=True) for sample in val_samples]
    paired_records = [box_record(runtime, sample, with_hidden=True) for sample in paired_samples]
    representation = representation_report(paired_records, val_records, pairs)

    report = {
        "_doc": (
            "Task 6F section 13. [BOX]-query hidden representation diagnostics at the best F1 "
            "checkpoint, with global-mean centring, compared against the frozen Task 6D.1 legacy "
            "[SEG] values. Do not infer causality from cosine alone."
        ),
        "task": "6F",
        "selected_epoch": selected_epoch,
        "checkpoint": checkpoint,
        "checkpoint_load": load_report,
        "representation": representation,
        "seconds": round(time.time() - started, 2),
    }
    write_json(OUT, report)
    print(
        f"[task6f:repr] same-image cosine {representation['same_image_different_instruction_cosine']} "
        f"centered {representation['same_image_different_instruction_centered_cosine']}; diff-image "
        f"{representation['different_image_cosine']} centered "
        f"{representation['different_image_centered_cosine']}; effective rank "
        f"{representation['effective_rank_participation_ratio']}; dist-vs-box correlation "
        f"{representation['hidden_distance_vs_gt_box_distance_correlation']}",
        flush=True,
    )
    print(f"[task6f:repr] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
