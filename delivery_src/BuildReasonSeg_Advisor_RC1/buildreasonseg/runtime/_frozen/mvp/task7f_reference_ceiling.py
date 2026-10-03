"""Task 7F — reference-mode construction and ceiling measurement (read-only).

Four exact reference modes over the frozen Task 7E `E-HoldoutL3` / `E-PairedHoldout20` populations, using the
frozen U-C1 proposal set (one inference pass per tile, reused by every mode), the frozen Task 7D D-B1 decoder
and the canonical GT reference **only** for the declared diagnostic ceilings:

```text
F-R0 CURRENT_SELECTED_PRED_MASK     eligible proposals -> max predicted area, tie higher confidence, lower
                                    index -> that predicted mask        (production-like baseline)
F-R1 ORACLE_SELECTED_PRED_MASK      among the same eligible proposals -> max IoU to GT reference,
                                    tie higher confidence then lower index -> that **predicted** mask
F-R2 COVERAGE_CONDITIONAL_GT_MASK   best eligible IoU >= 0.50 -> canonical GT mask, else abstain
F-R3 FULL_ORACLE_GT_MASK            canonical GT mask always            (full reference ceiling)
```

Thresholds, eligibility, the decoder and the fields are all frozen; GT enters only as the declared diagnostic
reference of F-R2/F-R3 (and to *choose* the F-R1 proposal), never as a production input.

The helper caches its per-record / per-pair rows in the gitignored `artifacts/task7f/` so the reporting scripts
can each read the same measurement instead of recomputing it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
HOLDOUT_ROOT = REPO_ROOT / "artifacts" / "task7e" / "holdout"
CACHE_ROOT = REPO_ROOT / "artifacts" / "task7f"
CACHE_PATH = CACHE_ROOT / "reference_modes_cache.json"
EVAL = REPO_ROOT / "evaluation"
TOLERANCE = 1.0e-6
COVERAGE_IOU = 0.50
COVERAGE_BANDS = (0.25, 0.50, 0.75)
MODES = ("F-R0", "F-R1", "F-R2", "F-R3")
MODE_LABELS = {
    "F-R0": "CURRENT_SELECTED_PRED_MASK",
    "F-R1": "ORACLE_SELECTED_PRED_MASK",
    "F-R2": "COVERAGE_CONDITIONAL_GT_MASK",
    "F-R3": "FULL_ORACLE_GT_MASK",
}
DIRECTIONS = ("above", "below", "left", "right")
D_B1_SHA256 = "6df31909cefdb54b9997b1e6ab76b8c5589771defa55666edbe75106221a89c0"
YOLO_SHA256 = "ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def iou_of(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    union = float((left | right).sum())
    if union == 0.0:
        return 1.0
    return float((left & right).sum()) / union


def dice_of(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    denominator = float(left.sum()) + float(right.sum())
    if denominator == 0.0:
        return 1.0
    return 2.0 * float((left & right).sum()) / denominator


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    rows, columns = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size == 0:
        return (float("nan"), float("nan"))
    return (float(rows.mean()), float(columns.mean()))


def centroid_error(left: np.ndarray, right: np.ndarray) -> float:
    left_rows, left_columns = mask_centroid(left)
    right_rows, right_columns = mask_centroid(right)
    if np.isnan(left_rows) or np.isnan(right_rows):
        return float("nan")
    return float(np.hypot(left_rows - right_rows, left_columns - right_columns))


# ---------------------------------------------------------------- mode construction


def select_current(eligible: list) -> object | None:
    """F-R0: maximum predicted mask area, tie higher confidence, then lower original index."""

    if not eligible:
        return None
    return max(eligible, key=lambda proposal: (int(np.asarray(proposal.mask).sum()),
                                               float(proposal.confidence),
                                               -int(getattr(proposal, "index", 0))))


def select_oracle(eligible: list, gt_reference: np.ndarray) -> object | None:
    """F-R1: maximum IoU with the canonical GT reference, tie higher confidence, then lower index."""

    if not eligible:
        return None
    scored = [(iou_of(np.asarray(proposal.mask, dtype=bool), gt_reference), float(proposal.confidence),
               -int(getattr(proposal, "index", 0)), proposal) for proposal in eligible]
    best = max(scored, key=lambda entry: (entry[0], entry[1], entry[2]))
    return best[3]


def best_eligible_iou(eligible: list, gt_reference: np.ndarray) -> float:
    if not eligible:
        return 0.0
    return max(iou_of(np.asarray(proposal.mask, dtype=bool), gt_reference) for proposal in eligible)


def build_mode_masks(eligible: list, gt_reference: np.ndarray) -> dict:
    """The exact four mode masks for one record (None means abstention)."""

    current = select_current(eligible)
    oracle = select_oracle(eligible, gt_reference)
    best_iou = best_eligible_iou(eligible, gt_reference)
    covered = bool(eligible) and best_iou >= COVERAGE_IOU
    masks = {
        "F-R0": (np.asarray(current.mask, dtype=bool) if current is not None else None),
        "F-R1": (np.asarray(oracle.mask, dtype=bool) if oracle is not None else None),
        "F-R2": (np.asarray(gt_reference, dtype=bool) if covered else None),
        "F-R3": np.asarray(gt_reference, dtype=bool),
    }
    detail = {
        "eligible_count": len(eligible),
        "best_eligible_iou": best_iou,
        "covered50": covered,
        "selected_confidence": (float(current.confidence) if current is not None else None),
        "selected_area": (int(np.asarray(current.mask).sum()) if current is not None else None),
        "oracle_selected_confidence": (float(oracle.confidence) if oracle is not None else None),
        "oracle_selected_area": (int(np.asarray(oracle.mask).sum()) if oracle is not None else None),
        "selected_vs_best_iou": (iou_of(np.asarray(current.mask, dtype=bool),
                                        np.asarray(oracle.mask, dtype=bool))
                                 if (current is not None and oracle is not None) else None),
        "reference_iou_gap": ((best_iou - iou_of(np.asarray(current.mask, dtype=bool), gt_reference))
                              if current is not None else None),
    }
    return {"masks": masks, "detail": detail}


# ---------------------------------------------------------------- cache


def load_holdout_rows() -> list[dict]:
    rows = []
    with (HOLDOUT_ROOT / "holdout_l3_rows.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_pairs() -> list[dict]:
    return json.loads((HOLDOUT_ROOT / "holdout_pairs.json").read_text(encoding="utf-8"))["pairs"]


def cache_exists() -> bool:
    return CACHE_PATH.is_file()


def read_cache() -> dict:
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def write_cache(payload: dict) -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


__all__ = ["CACHE_PATH", "CACHE_ROOT", "COVERAGE_BANDS", "COVERAGE_IOU", "D_B1_SHA256", "DIRECTIONS",
           "EVAL", "HOLDOUT_ROOT", "MODES", "MODE_LABELS", "TOLERANCE", "YOLO_SHA256",
           "best_eligible_iou", "build_mode_masks", "cache_exists", "centroid_error", "dice_of",
           "iou_of", "load_holdout_rows", "load_pairs", "mask_centroid", "read_cache",
           "select_current", "select_oracle", "sha256_file", "sha256_text", "write_cache"]
