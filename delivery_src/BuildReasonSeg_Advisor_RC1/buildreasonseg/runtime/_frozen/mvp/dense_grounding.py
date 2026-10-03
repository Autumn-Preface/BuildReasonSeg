"""Task 6G: Dense Query–Visual Spatial Grounding Map v0.1.

Task 6F showed that a single global hidden vector regressed through SmoothL1 cannot localize
WHU-scale buildings on 480 records. Task 6G keeps Task 6F's causally clean pre-reasoning `[BOX]`
query but stops regressing a box: the query hidden is matched against the **frozen SAM2 dense
spatial feature map** to produce a target-specific heatmap, and the predicted point is the argmax
cell centre of that heatmap:

    image + instruction -> fixed [BOX] query -> reasoning_zh [SEG] EOS
    [BOX] hidden -> LayerNorm -> Linear(2048,128) -> q
    SAM2 spatial feature [B,C,H,W] -> Conv1x1(C,128) -> K
    heatmap_logits[y,x] = dot(q, K[:,y,x]) / sqrt(128) (+ optional scalar bias)

    L_total = 1.0 * L_reasoning + 2.0 * L_heatmap
    L_heatmap = 1.0 * BCEWithLogits(heatmap_logits, soft_target) + 1.0 * SoftDice(sigmoid, soft_target)

The soft target is the instruction-selected target building mask, area-downsampled to the selected
grid. GT is supervision only; inference = image + instruction + constant `[BOX]` + learned weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .losses import mask_bce_with_logits, mask_soft_dice

#: Section 3 candidate grids.
GRID_CANDIDATES = (64, 128, 256)

#: Section 5: the shared query/key dimension.
KEY_DIM = 128

#: Section 8 fixes these; they are constants, not hyperparameters to sweep.
HEATMAP_BCE_WEIGHT = 1.0
HEATMAP_DICE_WEIGHT = 1.0
HEATMAP_LOSS_WEIGHT = 2.0
REASONING_LOSS_WEIGHT = 1.0


# ------------------------------------------------------------------ grid geometry


def cell_centre(index: int, grid: int) -> float:
    """The centre of grid cell `index` in normalized coordinates."""

    return (int(index) + 0.5) / int(grid)


def snap_point_to_cell(point_xy, grid: int) -> tuple[int, int]:
    """Snap a normalized point to the nearest cell of a `grid x grid` grid.

    Cell i spans [i/grid, (i+1)/grid) and its centre is (i+0.5)/grid, so the nearest cell is
    simply floor(x * grid): the midpoint between two cell centres sits exactly on a cell border.
    `math.floor` makes the choice deterministic everywhere, including at exact ties.
    """

    x, y = float(point_xy[0]), float(point_xy[1])
    ix = int(min(max(math.floor(x * int(grid)), 0), int(grid) - 1))
    iy = int(min(max(math.floor(y * int(grid)), 0), int(grid) - 1))
    return ix, iy


def snapped_point(point_xy, grid: int) -> tuple[float, float]:
    """The normalized cell-centre coordinates of the snapped point."""

    ix, iy = snap_point_to_cell(point_xy, grid)
    return cell_centre(ix, grid), cell_centre(iy, grid)


# ------------------------------------------------------------------ feature levels


def feature_tensor_for_grid(features, grid: int) -> torch.Tensor:
    """The frozen SAM2 spatial feature whose spatial size is `grid x grid`.

    Verified, not assumed: the 64x64 level is the main image embedding; the 128x128 and 256x256
    levels are picked from `high_res_features` **by spatial size**, never by list order.
    """

    grid = int(grid)
    if grid == 64:
        tensor = features.image_embeddings
    elif grid in (128, 256):
        by_size = {int(level.shape[-2]): level for level in features.high_res_features}
        if grid not in by_size:
            raise RuntimeError(f"no SAM2 high-res feature at {grid}x{grid}; got {sorted(by_size)}")
        tensor = by_size[grid]
    else:
        raise ValueError(f"grid must be one of {GRID_CANDIDATES}, got {grid}")
    if int(tensor.shape[-2]) != grid or int(tensor.shape[-1]) != grid:
        raise RuntimeError(f"feature {tuple(tensor.shape)} does not match grid {grid}")
    if tensor.dim() == 3:
        tensor = tensor[None, ...]
    return tensor


def feature_level_report(features) -> dict:
    """Section 2: the exact measured shapes of every SAM2 spatial level."""

    return {
        "image_embeddings_shape": list(tuple(features.image_embeddings.shape)),
        "high_res_shapes": [list(tuple(level.shape)) for level in (features.high_res_features or [])],
        "levels_by_grid": {
            grid: list(tuple(feature_tensor_for_grid(features, grid).shape))
            for grid in GRID_CANDIDATES
        },
    }


# ------------------------------------------------------------------ head


class DenseSpatialGroundingHead(torch.nn.Module):
    """Section 5: query-hidden x frozen-visual-feature dot-product localization map.

    Trainable: the query projection (`LayerNorm -> Linear(2048,128)`) and the 1x1 visual-key
    projection (`Conv1x1(C,128)`), plus an optional scalar logit bias. Nothing else: no transformer
    block, no relation module, no large decoder.
    """

    def __init__(self, query_dim: int = 2048, key_dim: int = KEY_DIM,
                 in_channels: int | None = None, use_bias: bool = True) -> None:
        super().__init__()
        self.query_dim = int(query_dim)
        self.key_dim = int(key_dim)
        self.query_norm = torch.nn.LayerNorm(self.query_dim)
        self.query_proj = torch.nn.Linear(self.query_dim, self.key_dim)
        self.visual_proj = None if in_channels is None else torch.nn.Conv2d(int(in_channels), self.key_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(())) if use_bias else None
        self.register_buffer("logit_scale", torch.tensor(1.0 / math.sqrt(self.key_dim)), persistent=False)

    @property
    def in_channels(self) -> int | None:
        return None if self.visual_proj is None else int(self.visual_proj.in_channels)

    def forward(self, query_hidden: torch.Tensor, spatial_feature: torch.Tensor) -> torch.Tensor:
        """`(query [B,2048], feature [B,C,H,W]) -> heatmap logits [B,H,W]`."""

        if self.visual_proj is None:
            raise RuntimeError("visual_proj not initialised; call set_in_channels first")
        q = self.query_proj(self.query_norm(query_hidden.float()))          # [B, 128]
        keys = self.visual_proj(spatial_feature.float())                     # [B, 128, H, W]
        logits = torch.einsum("bc,bchw->bhw", q, keys) * self.logit_scale   # [B, H, W]
        if self.bias is not None:
            logits = logits + self.bias
        return logits

    def set_in_channels(self, in_channels: int) -> None:
        """Late-bind the 1x1 projection once the selected feature level is known."""

        if self.visual_proj is not None:
            raise RuntimeError("visual_proj already initialised")
        self.visual_proj = torch.nn.Conv2d(int(in_channels), self.key_dim, 1)
        self.visual_proj.to(next(self.parameters()).device, dtype=next(self.parameters()).dtype)

    def as_dict(self) -> dict:
        return {
            "class": "DenseSpatialGroundingHead",
            "query_dim": self.query_dim,
            "key_dim": self.key_dim,
            "in_channels": self.in_channels,
            "structure": (
                "query: LayerNorm(2048) -> Linear(2048,128); visual: Conv1x1(C,128); "
                "heatmap = dot(q, K)/sqrt(128) + scalar bias"
            ),
            "scalar_bias": self.bias is not None,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
        }


# ------------------------------------------------------------------ supervision


def downsample_target_mask(mask_bool, grid: int) -> torch.Tensor:
    """The instruction-selected target mask, area-downsampled to `grid x grid` soft occupancy.

    Returns a `[grid, grid]` tensor in [0,1] and asserts the target has mass (section 7).
    """

    binary = np.asarray(mask_bool)
    tensor = torch.as_tensor(binary, dtype=torch.float32)[None, None, :, :]
    soft = F.interpolate(tensor, size=(int(grid), int(grid)), mode="area")[0, 0]
    if float(soft.sum()) <= 0.0:
        raise RuntimeError("downsampled target heatmap has no mass")
    return soft


def heatmap_loss(logits: torch.Tensor, soft_target: torch.Tensor) -> dict:
    """Section 8: `1.0 * BCEWithLogits + 1.0 * SoftDice(sigmoid, target)`."""

    bce = mask_bce_with_logits(logits, soft_target)
    dice = mask_soft_dice(logits, soft_target)
    return {
        "bce": bce,
        "dice": dice,
        "heatmap": HEATMAP_BCE_WEIGHT * bce + HEATMAP_DICE_WEIGHT * dice,
        "bce_raw": float(bce.detach()),
        "dice_raw": float(dice.detach()),
        "heatmap_raw": float((HEATMAP_BCE_WEIGHT * bce + HEATMAP_DICE_WEIGHT * dice).detach()),
    }


def soft_dice_from_logits(logits: torch.Tensor, soft_target: torch.Tensor) -> float:
    """The section 8 soft Dice as a *quality* metric (1 - loss term)."""

    return float(1.0 - mask_soft_dice(logits.detach(), soft_target.detach()))


def binary_iou_from_logits(logits: torch.Tensor, soft_target: torch.Tensor, threshold: float = 0.5) -> float:
    """Diagnostic: IoU of the thresholded heatmap against the thresholded soft target."""

    probabilities = torch.sigmoid(logits.detach().float())
    prediction = probabilities > threshold
    target = soft_target.detach().float() > threshold
    intersection = float((prediction & target).sum())
    union = float((prediction | target).sum())
    return intersection / union if union else 0.0


# ------------------------------------------------------------------ point extraction


def argmax_point_from_logits(logits: torch.Tensor, grid: int) -> dict:
    """Section 9: `(y*,x*) = argmax(heatmap_logits)`, point = centre of the selected cell.

    `torch.argmax` returns the first maximal element in row-major order, so the point is a pure
    deterministic function of the logits. No threshold tuning and no GT-guided repair.
    """

    flat = logits.detach().float().reshape(-1)
    index = int(torch.argmax(flat))
    y, x = divmod(index, int(grid))
    point = (cell_centre(x, grid), cell_centre(y, grid))

    probabilities = torch.softmax(flat, dim=0)
    coords = torch.tensor(
        [[cell_centre(xi % int(grid), grid), cell_centre(xi // int(grid), grid)] for xi in range(flat.numel())],
        dtype=torch.float32,
    )
    soft_expectation = (probabilities[:, None] * coords).sum(dim=0).tolist()
    return {
        "point": [float(value) for value in point],
        "cell": [x, y],
        "logit_value": float(flat[index]),
        "soft_argmax_point": [float(value) for value in soft_expectation],
        "peakiness": float(probabilities[index]),
    }


def point_inside_mask(mask_bool, point_xy) -> bool:
    """Whether the (normalized) point's rounded pixel lies inside the binary target mask."""

    binary = np.asarray(mask_bool).astype(bool)
    height, width = binary.shape
    px = int(round(float(point_xy[0]) * (width - 1)))
    py = int(round(float(point_xy[1]) * (height - 1)))
    if not (0 <= px < width and 0 <= py < height):
        return False
    return bool(binary[py, px])


def normalized_point_error(pred_point, gt_point) -> dict:
    """Normalized and original-512-pixel point error to the frozen interior point."""

    dx = float(pred_point[0]) - float(gt_point[0])
    dy = float(pred_point[1]) - float(gt_point[1])
    return {
        "normalized_error": float((abs(dx) + abs(dy)) / 2.0),
        "error_512px": float((abs(dx) + abs(dy)) / 2.0 * 512.0),
        "dx": dx,
        "dy": dy,
    }
