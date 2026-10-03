"""Lightweight visual outputs for the Task 6A sample pack.

Panel content: source image, ground-truth mask outline, predicted mask outline,
instruction, query type and IoU. Kept small (one PNG per sample, ~256 px tall).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

THUMB_HEIGHT = 300
BANNER_HEIGHT = 76


def _outline(mask: np.ndarray) -> np.ndarray:
    """Boolean edge map of a binary mask (4-neighbour difference)."""

    mask = mask.astype(bool)
    edge = np.zeros_like(mask)
    edge[1:, :] |= mask[1:, :] & ~mask[:-1, :]
    edge[:-1, :] |= mask[:-1, :] & ~mask[1:, :]
    edge[:, 1:] |= mask[:, 1:] & ~mask[:, :-1]
    edge[:, :-1] |= mask[:, :-1] & ~mask[:, 1:]
    return edge


def render_panel(
    out_path: Path,
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    lines: list[str],
    iou: float,
) -> bool:
    import cv2

    canvas = cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2BGR).copy()
    canvas = (canvas * 0.65).astype(np.uint8)

    gt = gt_mask.astype(bool)
    pred = pred_mask.astype(bool)

    tint_gt = np.zeros_like(canvas)
    tint_gt[gt] = (0, 200, 255)          # amber = ground truth
    canvas = np.where(gt[..., None], (canvas * 0.55 + tint_gt * 0.45).astype(np.uint8), canvas)

    tint_pred = np.zeros_like(canvas)
    tint_pred[pred] = (0, 255, 0)        # green = prediction
    canvas = np.where(pred[..., None], (canvas * 0.55 + tint_pred * 0.45).astype(np.uint8), canvas)

    canvas[_outline(gt)] = (0, 200, 255)
    canvas[_outline(pred)] = (0, 255, 0)

    banner = np.zeros((BANNER_HEIGHT, canvas.shape[1], 3), np.uint8)
    header = f"IoU {iou:.3f}   GT=amber  PRED=green"
    cv2.putText(banner, header, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    for index, text in enumerate(lines[:3]):
        cv2.putText(
            banner,
            text[:120],
            (4, 38 + index * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

    combined = np.vstack([banner, canvas])
    scale = THUMB_HEIGHT / combined.shape[0]
    resized = cv2.resize(
        combined, (max(1, int(combined.shape[1] * scale)), THUMB_HEIGHT), interpolation=cv2.INTER_AREA
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), resized))


def render_task6b_panel(
    out_path: Path,
    image_rgb: np.ndarray,
    gt_mask: np.ndarray | None,
    pred_mask: np.ndarray | None,
    header: str,
    lines: list[str],
    seg_valid: bool,
    iou: float | None,
) -> bool:
    """Task 6B panel: source, GT outline, prediction outline, generated reasoning.

    A missing prediction (invalid or absent `[SEG]`) is drawn as "no mask" rather
    than silently omitted, so a format failure is visible in the pack.
    """

    import cv2

    canvas = cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2BGR).copy()
    canvas = (canvas * 0.70).astype(np.uint8)

    if gt_mask is not None:
        gt = gt_mask.astype(bool)
        tint = np.zeros_like(canvas)
        tint[gt] = (0, 200, 255)
        canvas = np.where(gt[..., None], (canvas * 0.60 + tint * 0.40).astype(np.uint8), canvas)
        canvas[_outline(gt)] = (0, 200, 255)

    if pred_mask is not None:
        pred = pred_mask.astype(bool)
        tint = np.zeros_like(canvas)
        tint[pred] = (0, 255, 0)
        canvas = np.where(pred[..., None], (canvas * 0.60 + tint * 0.40).astype(np.uint8), canvas)
        canvas[_outline(pred)] = (0, 255, 0)

    banner_height = 118
    banner = np.zeros((banner_height, canvas.shape[1], 3), np.uint8)
    colour = (0, 255, 0) if seg_valid else (0, 0, 255)
    badge = f"[SEG] OK  IoU {iou:.3f}" if (seg_valid and iou is not None) else "[SEG] FAILED (IoU 0)"
    cv2.putText(banner, badge, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, colour, 1, cv2.LINE_AA)
    cv2.putText(banner, header[:118], (4, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    for index, text in enumerate(lines[:4]):
        cv2.putText(
            banner,
            text[:118],
            (4, 58 + index * 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    combined = np.vstack([banner, canvas])
    scale = THUMB_HEIGHT / combined.shape[0]
    resized = cv2.resize(
        combined, (max(1, int(combined.shape[1] * scale)), THUMB_HEIGHT), interpolation=cv2.INTER_AREA
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), resized))


def build_contact_sheet(sample_dir: Path, out_path: Path, columns: int = 4, thumb: int = 256) -> int:
    import cv2

    paths = sorted(p for p in sample_dir.glob("*.png") if p.name != "contact_sheet.png")
    thumbs = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        tile = cv2.resize(image, (thumb, thumb), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, path.stem[:32], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(tile)
    if not thumbs:
        return 0
    rows = (len(thumbs) + columns - 1) // columns
    sheet = np.zeros((rows * thumb, columns * thumb, 3), np.uint8)
    for index, tile in enumerate(thumbs):
        row, column = divmod(index, columns)
        sheet[row * thumb : (row + 1) * thumb, column * thumb : (column + 1) * thumb] = tile
    cv2.imwrite(str(out_path), sheet)
    return len(thumbs)
