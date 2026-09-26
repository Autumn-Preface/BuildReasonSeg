#!/usr/bin/env python
"""Task 6B: re-run the fixed validation on an existing checkpoint.

    python scripts/task6b_revalidate.py --checkpoint artifacts/checkpoints/task6b/best_joint.pt \
                                        --tag original

Why this exists
---------------
`teacher_forced_validation` originally mapped a supervised label at index `i` to
`lm_logits[i - 1]`. The labels are already shifted (`labels[i]` holds token
`i + 1`) and `lm_logits[i]` is the distribution over token `i + 1`, so the two
disagreed by one position and the reported token accuracies were wrong. The bug
was fixed in `buildreasonseg_mvp/validation.py`; this script re-measures an
already-trained checkpoint with the fixed code so the recorded result is
comparable with the runs that follow, without retraining.

Writes `evaluation/task6b_revalidation_<tag>.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# The run is offline: huggingface.co does not resolve on this network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp import language_metrics as LM  # noqa: E402
from buildreasonseg_mvp import validation as V  # noqa: E402
from buildreasonseg_mvp.checkpointing import load_checkpoint, sha256_file, write_json  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, load_config, set_phase_trainables  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from task6b_train import (  # noqa: E402
    BEST_JOINT_RULE,
    SUBSETS,
    full_validation,
    samples_for,
    selection_metrics,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--val-limit", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config or (REPO_ROOT / "configs" / "mvp" / "task6b_2b_minitrain.yaml"))
    payload = json.loads(SUBSETS.read_text(encoding="utf-8"))
    val_samples = samples_for(payload, "val")
    paired_ids = payload["paired_probe"]["sample_ids"]
    by_id = {r["sample_id"]: r for r in data_mod.read_records("val")}
    paired_records = [by_id[sid] for sid in paired_ids if sid in by_id]
    pairs = [{"a": paired_records[i], "b": paired_records[i + 1]} for i in range(0, len(paired_records), 2)]

    lookup = LM.build_reasoning_lookup(list(data_mod.read_records("train")) + list(data_mod.read_records("val")))

    if args.val_limit:
        val_samples = val_samples[: args.val_limit]
        pairs = pairs[: max(1, args.val_limit // 4)]

    runtime = build_runtime(cfg, device="cuda", verbose=True)
    set_phase_trainables(runtime.model, "B")

    checkpoint = Path(args.checkpoint)
    restore = load_checkpoint(checkpoint, runtime.model)
    print(f"[revalidate] restored {checkpoint.name}: {restore}", flush=True)

    print(f"[revalidate] validating {len(val_samples)} records ...", flush=True)
    result = full_validation(runtime, val_samples, lookup)
    paired = V.paired_probe(runtime, pairs, lookup)

    report = {
        "_doc": (
            "Task 6B re-validation of an existing checkpoint with the corrected "
            "teacher-forced token-accuracy indexing."
        ),
        "task": "6B",
        "tag": args.tag,
        "config_path": str(args.config or (REPO_ROOT / "configs" / "mvp" / "task6b_2b_minitrain.yaml")),
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
            "restore": restore,
        },
        "best_joint_rule": list(BEST_JOINT_RULE),
        "selection": selection_metrics(result),
        "teacher_forced": result["teacher_forced"],
        "teacher_forced_breakdown_level": result["teacher_forced_breakdown_level"],
        "teacher_forced_breakdown_family": result["teacher_forced_breakdown_family"],
        "free_generation": result["free_generation"],
        "free_generation_breakdown_level": result["free_generation_breakdown_level"],
        "free_generation_breakdown_family": result["free_generation_breakdown_family"],
        "language": result["language"],
        "paired_probe": paired,
        "per_record_free_generation": result["_free_records"],
        "per_record_teacher_forced": result["_tf_records"],
        "free_generation_seconds": result["free_generation_seconds"],
        "vram": result["free_generation_peak_vram"],
    }

    out = REPO_ROOT / "evaluation" / f"task6b_revalidation_{args.tag}.json"
    write_json(out, report)
    print(f"[revalidate] selection: {report['selection']}", flush=True)
    print(f"[revalidate] paired: {paired['passed']}/{paired['n_pairs']}", flush=True)
    print(f"[revalidate] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
