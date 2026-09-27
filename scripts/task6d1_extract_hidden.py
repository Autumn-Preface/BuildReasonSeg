#!/usr/bin/env python
"""Task 6D.1 section 5: extract frozen `[SEG]` hidden vectors for the decodability audit.

    python scripts/task6d1_extract_hidden.py [--checkpoint <path>] [--limit N]

Representation source: the **Task 6C `P_C` checkpoint before any Task 6D grounding training**,
hash-verified against `evaluation/task6c_checkpoint_manifest.json`. The Qwen stack is frozen
during extraction (`torch.no_grad`), so the vectors are a fixed property of that checkpoint.

Extracts teacher-forced `[SEG]` hidden vectors for all 480 `P` training records and the fixed
120 validation records, and records for each: image id, query type, template id, target box,
target component id. No test split is touched.

Tensors stay local and gitignored (`artifacts/task6d1_hidden/`); the committed manifest records
shape, dtype, a sample-id hash and the SHA256 of each serialized file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.checkpointing import load_checkpoint, sha256_file  # noqa: E402
from buildreasonseg_mvp.grounding import GEOMETRY_BOX, target_geometry  # noqa: E402
from buildreasonseg_mvp.qwen_seg import forward_qwen  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, load_config, set_seed  # noqa: E402
from task6c_train import subset_records, validation_material  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
HIDDEN_DIR = REPO_ROOT / "artifacts" / "task6d1_hidden"
CONFIG = REPO_ROOT / "configs" / "mvp" / "task6c_2b_ablation.yaml"
TASK6C_SUBSETS = EVAL / "task6c_subset_ids.json"
MANIFEST = EVAL / "task6c_checkpoint_manifest.json"
OUT = EVAL / "task6d1_hidden_extract_manifest.json"
DEFAULT_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6c" / "P_C" / "last.pt"


def verify_checkpoint(path: Path) -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = (manifest.get("P_C") or {}).get("sha256")
    actual = sha256_file(path)
    return {
        "path": str(path),
        "sha256": actual,
        "manifest_sha256": expected,
        "matches_manifest": bool(expected) and expected == actual,
    }


def extract(runtime, samples, split: str) -> dict:
    hidden_vectors = []
    labels = []
    with torch.no_grad():
        for index, sample in enumerate(samples, start=1):
            batch, image = runtime.prepare(sample)
            batch = batch.to(runtime.device)
            _logits, hidden = forward_qwen(runtime.model.qwen, batch)
            hidden_vectors.append(hidden.detach().float().cpu().reshape(-1))
            mask = sample.target_mask()
            labels.append(
                {
                    "sample_id": sample.sample_id,
                    "image_id": sample.image_id,
                    "split": split,
                    "level": sample.level,
                    "query_type": sample.query_type,
                    "template_id": sample.template_id,
                    "target_component_id": int(sample.target_component_id),
                    "box": list(target_geometry(mask, GEOMETRY_BOX)),
                    "point": list(target_geometry(mask, "point")),
                    "target_area_fraction": float(mask.mean()),
                }
            )
            del batch, hidden
            if index % 100 == 0:
                print(f"[task6d1:extract] {split}: {index}/{len(samples)}", flush=True)
    return {"hidden": torch.stack(hidden_vectors), "labels": labels}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--limit", type=int, default=None, help="smoke only")
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint)
    verification = verify_checkpoint(checkpoint) if checkpoint.is_file() else {"path": str(checkpoint), "exists": False}
    if not verification.get("matches_manifest"):
        print(
            "[task6d1:extract] WARNING: the checkpoint does not match the Task 6C manifest hash; "
            "the manifest records this fact.",
            flush=True,
        )

    cfg = load_config(CONFIG)
    set_seed(int(cfg["seed"]))
    runtime = build_runtime(cfg, device="cuda", verbose=True)
    runtime.model.eval()
    load_report = load_checkpoint(checkpoint, runtime.model) if checkpoint.is_file() else None

    train_records = subset_records(json.loads(TASK6C_SUBSETS.read_text(encoding="utf-8")), "P")
    train_samples = [data_mod.to_sample(record) for record in train_records]
    val_samples, pairs, _lookup, _audit = validation_material()
    # The 20 paired images are a *different* subset of the val split than the fixed 120-record
    # validation set, so they need their own extraction (Task 6D.1 implementation audit).
    paired_ids: list[str] = []
    for pair in pairs:
        for key in ("a", "b"):
            sample_id = pair[key]["sample_id"]
            if sample_id not in paired_ids:
                paired_ids.append(sample_id)
    val_records = {record["sample_id"]: record for record in data_mod.read_records("val")}
    paired_samples = [data_mod.to_sample(val_records[sample_id]) for sample_id in paired_ids]
    overlap = {sample.sample_id for sample in paired_samples} & {sample.sample_id for sample in val_samples}
    if args.limit:
        train_samples = train_samples[: args.limit]
        val_samples = val_samples[: args.limit]
        paired_samples = paired_samples[: args.limit]

    HIDDEN_DIR.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split, samples in (
        ("train", train_samples),
        ("val", val_samples),
        ("paired", paired_samples),
    ):
        payload = extract(runtime, samples, split)
        path = HIDDEN_DIR / f"{split}.pt"
        torch.save(payload, path)
        splits[split] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "samples": len(payload["labels"]),
            "hidden_shape": list(payload["hidden"].shape),
            "dtype": str(payload["hidden"].dtype),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "sample_ids_sha256": hashlib.sha256(
                "\n".join(label["sample_id"] for label in payload["labels"]).encode("utf-8")
            ).hexdigest(),
            "image_ids": len({label["image_id"] for label in payload["labels"]}),
            "templates": len({label["template_id"] for label in payload["labels"]}),
            "test_split_used": False,
        }
        print(
            f"[task6d1:extract] {split}: {splits[split]['samples']} vectors "
            f"{splits[split]['hidden_shape']} {splits[split]['dtype']}",
            flush=True,
        )

    manifest = {
        "_doc": (
            "Task 6D.1 section 5. Frozen teacher-forced [SEG] hidden vectors extracted from the Task "
            "6C P_C checkpoint (hash-verified) before any Task 6D grounding training. Tensors are "
            "local and gitignored; this manifest is committed instead."
        ),
        "task": "6D.1",
        "representation_source": {
            "checkpoint": verification,
            "load_report": load_report,
            "extraction": "torch.no_grad; Qwen frozen; teacher-forced assistant text (reasoning + [SEG])",
            "readout_position": "final hidden layer at batch.seg_position",
            "not_used": "no Task 6D grounding checkpoint; no test split",
        },
        "geometry_kind_for_labels": GEOMETRY_BOX,
        "hidden_dim": int(next(iter(splits.values()))["hidden_shape"][-1]) if splits else None,
        "splits": splits,
        "paired_validation_images": len(pairs),
        "paired_samples_extracted": splits.get("paired", {}).get("samples"),
        "paired_overlap_with_fixed_val_set": len(overlap),
        "pairs_note": (
            "the 20 paired images are a different subset of the val split than the fixed 120-record "
            "validation set (overlap "
            f"{len(overlap)}), so they are extracted separately for the paired probe"
        ),
        "determinism": runtime.reports["determinism"],
    }
    OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[task6d1:extract] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
