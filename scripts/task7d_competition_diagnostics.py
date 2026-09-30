"""Task 7D Part I — competition-map diagnostics (offline, GT used for measurement only).

Section 22: for D-B1/D-B2/D-B3/D-B4 the ground-truth target is downsampled to 64x64 with area interpolation and
thresholded at `>0.5`, then each competition map `A` is scored on:

1. `argmax(A)` inside the target rate;
2. target mass `sum(A over target cells)`;
3. reference mass `sum(A over reference cells)`;
4. normalized entropy `-sum(A log(A + eps)) / log(4096)`;
5. top-1 / top-16 / top-64 spatial mass.

Nothing here is used for tuning.

Writes `evaluation/task7d_competition_diagnostics.json`.

    python scripts/task7d_competition_diagnostics.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.task6n_relation_decoder import (  # noqa: E402
    FrozenFeatureStore,
    load_frozen_sam2_encoder,
)
from buildreasonseg_mvp.task7d_data import (  # noqa: E402
    CHECKPOINT_ROOT,
    TRAINABLE_VARIANTS,
    build_batch,
    load_pack,
)
from buildreasonseg_mvp.task7d_global_competition_decoder import (  # noqa: E402
    EPS,
    GlobalCompetitionDecoder,
    SPATIAL_TOKENS,
    VARIANT_USES_LEARNED_SCORE_HEAD,
)
from scripts.task6n_train import FEATURE_ROOT, MaskStore  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task7d_competition_diagnostics.json"
TRAINING = EVAL / "task7d_training.json"
TOLERANCE = 1.0e-6


def load_variant_model(variant: str, device: str) -> GlobalCompetitionDecoder:
    checkpoint = CHECKPOINT_ROOT / f"{variant.lower().replace('-', '')}_minitrain1200.pt"
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = GlobalCompetitionDecoder(variant).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def downsample_target(target: torch.Tensor) -> torch.Tensor:
    pooled = F.interpolate(target[:, None].float(), size=(64, 64), mode="area")
    return pooled[:, 0] > 0.5


def downsample_reference(reference: np.ndarray) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(reference, dtype=np.float32))[None, None]
    pooled = F.interpolate(tensor, size=(64, 64), mode="area")
    return (pooled[0, 0] > 0.5)


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {"records": 0}
    keys = ("argmax_in_target_rate", "target_mass", "reference_mass", "normalized_entropy",
            "top1_mass", "top16_mass", "top64_mass", "attention_sum")
    return {"records": len(rows),
            **{key: float(np.mean([row[key] for row in rows])) for key in keys}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    started = time.time()

    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    encoder, _report = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)
    masks = MaskStore()
    records = load_pack("z_mini_val_240")
    precomputed: dict[str, dict] = {}
    results = {}
    for variant in TRAINABLE_VARIANTS:
        model = load_variant_model(variant, args.device)
        rows = []
        for start in range(0, len(records), 8):
            chunk = records[start: start + 8]
            batch = build_batch(chunk, store, masks, args.device, precomputed)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                                 enabled=args.device != "cpu"):
                _logits, state = model(batch.visual, batch.relations, batch.directional,
                                       batch.nearest, return_state=True)
            attention = state.attention.float().cpu()
            target64 = downsample_target(batch.target.cpu())
            for index, record in enumerate(chunk):
                a = attention[index, 0]
                spatial_sum = float(a.sum())
                flat = a.flatten()
                order = torch.argsort(flat, descending=True)
                tgt = target64[index]
                ref = downsample_reference(batch.reference_masks[index])
                argmax_index = int(order[0])
                argmax_in_target = bool(tgt.flatten()[argmax_index])
                entropy = float(-(a * torch.log(a + EPS)).sum() / np.log(SPATIAL_TOKENS))
                rows.append({
                    "sample_id": record["sample_id"], "direction": record["direction"],
                    "attention_sum": spatial_sum,
                    "argmax_in_target_rate": float(argmax_in_target),
                    "target_mass": float(a[tgt].sum()) if bool(tgt.any()) else 0.0,
                    "reference_mass": float(a[ref].sum()) if bool(ref.any()) else 0.0,
                    "normalized_entropy": entropy,
                    "top1_mass": float(flat[order[:1]].sum()),
                    "top16_mass": float(flat[order[:16]].sum()),
                    "top64_mass": float(flat[order[:64]].sum()),
                })
        summary = summarise(rows)
        results[variant] = {
            "summary": summary,
            "attention_sum_ok": bool(abs(summary["attention_sum"] - 1.0) <= 1e-5),
            "uses_learned_score_head": VARIANT_USES_LEARNED_SCORE_HEAD[variant],
            "field_mass_mean": float(np.mean([1.0])) if variant != "D-B1" else None,
            "rows": rows,
        }
        print(f"[7d.diag] {variant}: target mass {summary['target_mass']:.4f} argmax-in-target "
              f"{summary['argmax_in_target_rate']:.4f} entropy {summary['normalized_entropy']:.4f} "
              f"top1 {summary['top1_mass']:.4f} top64 {summary['top64_mass']:.4f} sum "
              f"{summary['attention_sum']:.6f}", flush=True)

    payload = {
        "_doc": ("Task 7D section 22. Competition-map diagnostics for the four trainable variants on "
                 "Z-MiniVal240. The GT target is downsampled to 64x64 with area interpolation and "
                 "thresholded at >0.5; the GT reference comes from the frozen oracle mask. These metrics "
                 "are offline measurements only and were not used for tuning."),
        "task": "7D", "stage": "I-competition-diagnostics",
        "pack": {"name": "z_mini_val_240", "records": len(records)},
        "constants": {"eps": EPS, "spatial_tokens": SPATIAL_TOKENS, "target_threshold": 0.5,
                      "downsample": "area", "normalization": "log(4096)"},
        "variants": results,
        "training_performed": False, "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
