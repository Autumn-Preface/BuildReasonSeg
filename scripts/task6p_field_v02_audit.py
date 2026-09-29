"""Task 6P section 3.4 — GeometricRelationField v0.2 equivalence and gradient audit.

Gates (fixed; failure means `DIFFERENTIABLE_FIELD_INVALID`):

1. 64 binary oracle reference masks deterministically selected from the frozen Task 6N MiniVal240 pack,
   balanced over the 4 directions (16 each);
2. v0.1 vs v0.2 at 64 x 64: **max abs error <= 1e-6** and **mean abs error <= 1e-7**;
3. gradient check: >= 8 deterministic non-binary soft masks with `requires_grad=True`, all four
   directions represented, a fixed nonuniform spatial weighting tensor `W`, `scalar = sum(P_rel * W)`,
   backprop, and the gradient to `M_ref` must exist, be finite and have L1 norm > 1e-8.

Writes `evaluation/task6p_field_v02_audit.json`.

    python scripts/task6p_field_v02_audit.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.geometric_relation_field import (  # noqa: E402
    geometric_relation_field as v01_field,
    PROGRAM_TO_RELATION,
)
from buildreasonseg_mvp.geometric_relation_field_v02 import (  # noqa: E402
    EPS, geometric_relation_field_v02, soft_centroid,
)
from buildreasonseg_mvp.task6m_eval import canonical_instances, write_json  # noqa: E402
from buildreasonseg_mvp.task6n_relation_decoder import read_pack  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6n" / "packs"
OUT = EVAL / "task6p_field_v02_audit.json"
SIZE = (64, 64)
MAX_ABS_TOLERANCE = 1e-6
MEAN_ABS_TOLERANCE = 1e-7
MIN_GRADIENT_L1 = 1e-8
DIRECTIONS = ("left_of", "right_of", "above", "below")
PER_DIRECTION = 16


def select_audit_masks() -> tuple[list[dict], list[str]]:
    """16 MiniVal240 references per direction, deterministic order, distinct (tile, feature) keys."""

    records = read_pack(PACK_ROOT / "mini_val_240.json")
    per_direction: dict[str, list] = {direction: [] for direction in DIRECTIONS}
    for record in sorted(records, key=lambda item: (item.relation, item.tile_id,
                                                    item.reference_source_feature_id)):
        bucket = per_direction.get(record.relation)
        if bucket is None:
            continue
        bucket.append(record)
    selected = []
    problems = []
    for direction in DIRECTIONS:
        bucket = per_direction[direction][:PER_DIRECTION]
        if len(bucket) < PER_DIRECTION:
            problems.append(f"{direction}: only {len(bucket)} records available")
        selected.extend(bucket)
    return selected, problems


def mask_for(record) -> np.ndarray:
    instances = canonical_instances(record.tile_id)
    for instance in instances:
        if instance.source_feature_id == record.reference_source_feature_id:
            return np.asarray(instance.mask, dtype=bool)
    raise KeyError(f"{record.tile_id}: reference {record.reference_source_feature_id} not found")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    started = time.time()
    records, problems = select_audit_masks()
    rows = []
    max_abs = 0.0
    total_abs = 0.0
    total_values = 0
    for record in records:
        mask = mask_for(record)
        tensor = torch.as_tensor(mask, dtype=torch.float32)
        with torch.no_grad():
            reference = v01_field(tensor, record.relation, SIZE)
            candidate = geometric_relation_field_v02(tensor, record.relation, SIZE)
        difference = (reference - candidate).abs()
        row_max = float(difference.max())
        row_mean = float(difference.mean())
        max_abs = max(max_abs, row_max)
        total_abs += float(difference.sum())
        total_values += difference.numel()
        rows.append(
            {
                "sample_id": record.sample_id,
                "tile_id": record.tile_id,
                "relation": record.relation,
                "reference_area_px": int(mask.sum()),
                "max_abs_error": row_max,
                "mean_abs_error": row_mean,
                "v01_centroid": list(_v01_centroid(mask)),
                "v02_centroid": [float(value) for value in
                                 torch.cat([item.view(-1) for item in soft_centroid(tensor)]).tolist()],
            }
        )
    mean_abs = total_abs / max(total_values, 1)
    equivalence = {
        "masks": len(rows),
        "per_direction": {direction: sum(1 for row in rows if row["relation"] == direction)
                          for direction in DIRECTIONS},
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "max_abs_tolerance": MAX_ABS_TOLERANCE,
        "mean_abs_tolerance": MEAN_ABS_TOLERANCE,
        "max_abs_ok": max_abs <= MAX_ABS_TOLERANCE,
        "mean_abs_ok": mean_abs <= MEAN_ABS_TOLERANCE,
        "worst_rows": sorted(rows, key=lambda row: -row["max_abs_error"])[:5],
    }
    equivalence["passed"] = bool(equivalence["max_abs_ok"] and equivalence["mean_abs_ok"]
                                and not problems)

    # ---------------- gradient check: determined non-binary soft masks
    gradient_rows = []
    height = width = 64
    rows_grid = (torch.arange(height, dtype=torch.float32) + 0.5) / height
    columns_grid = (torch.arange(width, dtype=torch.float32) + 0.5) / width
    yy, xx = torch.meshgrid(rows_grid, columns_grid, indexing="ij")
    weighting = 1.0 + 2.0 * xx + 3.0 * yy * yy  # fixed nonuniform spatial weighting tensor W
    for index in range(8):
        direction = DIRECTIONS[index % len(DIRECTIONS)]
        # deterministic smooth non-binary blob, different per index
        centre_x = 0.25 + 0.06 * index
        centre_y = 0.30 + 0.05 * (index % 3)
        sigma = 0.10 + 0.01 * (index % 4)
        blob = torch.exp(-(((xx - centre_x) ** 2 + (yy - centre_y) ** 2) / (2 * sigma * sigma)))
        mask = (0.90 * blob + 0.05).clone().requires_grad_(True)
        field = geometric_relation_field_v02(mask, direction, SIZE)
        scalar = (field * weighting.view(1, 1, height, width)).sum()
        scalar.backward()
        gradient = mask.grad
        finite = bool(torch.isfinite(gradient).all())
        l1 = float(gradient.abs().sum())
        gradient_rows.append(
            {
                "index": index,
                "relation": direction,
                "mask_is_non_binary": bool(((mask.detach() > 0.01) & (mask.detach() < 0.99)).any()),
                "scalar": float(scalar.detach()),
                "gradient_exists": gradient is not None,
                "gradient_finite": finite,
                "gradient_l1": l1,
                "gradient_max_abs": float(gradient.abs().max()) if gradient is not None else None,
                "passed": bool(gradient is not None and finite and l1 > MIN_GRADIENT_L1),
            }
        )
    directions_covered = sorted({row["relation"] for row in gradient_rows})
    gradient = {
        "masks": len(gradient_rows),
        "directions": directions_covered,
        "all_four_directions": set(directions_covered) == set(DIRECTIONS),
        "weighting": "W = 1 + 2x + 3y^2 (fixed nonuniform)",
        "min_gradient_l1": MIN_GRADIENT_L1,
        "rows": gradient_rows,
    }
    gradient["passed"] = bool(
        gradient["masks"] >= 8
        and gradient["all_four_directions"]
        and all(row["passed"] and row["mask_is_non_binary"] for row in gradient_rows)
    )

    eps_effect = {
        "eps": EPS,
        "note": (
            "v0.2 adds eps=1e-6 to the mask mass so it stays well defined for a vanishing soft mask; "
            "for a binary mask with area A the centroid perturbation is ~1e-6/A, which is why the "
            "measured error above stays inside the fixed v0.1-equivalence tolerances"
        ),
    }
    passed = bool(equivalence["passed"] and gradient["passed"])
    payload = {
        "_doc": (
            "Task 6P section 3.4. GeometricRelationField v0.2 autocraft audit: numerical equivalence to "
            "the frozen v0.1 on deterministic binary oracle reference masks, plus a gradient check on "
            "deterministic non-binary soft masks."
        ),
        "task": "6P",
        "stage": "B-v02-audit",
        "reference_source": "oracle_native_gt",
        "size": list(SIZE),
        "v01_file": str(REPO_ROOT / "buildreasonseg_mvp" / "geometric_relation_field.py"),
        "v01_file_sha256": hashlib.sha256(
            (REPO_ROOT / "buildreasonseg_mvp" / "geometric_relation_field.py").read_bytes()
        ).hexdigest(),
        "v02_file": str(REPO_ROOT / "buildreasonseg_mvp" / "geometric_relation_field_v02.py"),
        "v01_differentiable_wrt_reference_mask": False,
        "v01_reason": (
            "v0.1 calls .detach() on tensor masks and converts the centroid to a Python float, so it is "
            "not differentiable with respect to the input reference mask; that did not invalidate Task "
            "6N/6O because the reference mask was oracle and frozen"
        ),
        "selection_problems": problems,
        "equivalence": equivalence,
        "gradient_check": gradient,
        "eps_effect": eps_effect,
        "rows": rows,
        "verdict": "FIELD_V02_VALID" if passed else "DIFFERENTIABLE_FIELD_INVALID",
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(Path(args.out), payload)
    print(
        f"[6p.v02] equivalence max {max_abs:.3e} (<= {MAX_ABS_TOLERANCE:.0e}), mean {mean_abs:.3e} "
        f"(<= {MEAN_ABS_TOLERANCE:.0e}); gradient L1 "
        f"{min(row['gradient_l1'] for row in gradient_rows):.3e} -> {payload['verdict']}",
        flush=True,
    )
    return 0 if passed else 2


def _v01_centroid(mask: np.ndarray) -> tuple[float, float]:
    from buildreasonseg_mvp.geometric_relation_field import reference_centroid

    return reference_centroid(torch.as_tensor(mask, dtype=torch.float32))


if __name__ == "__main__":
    raise SystemExit(main())
