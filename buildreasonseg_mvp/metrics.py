"""Mask metrics for the BuildReasonSeg-MVP.

All reported IoU is computed at the ORIGINAL 512x512 tile resolution, from a
fixed threshold on the upsampled logits (Task 6A section 11.4). Training may
supervise at the decoder's low resolution, but the number reported to a reader is
always the original-resolution one.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def upsample_logits(logits: torch.Tensor, size: tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    """Upsample low-resolution mask logits to the original tile size.

    Two modes are supported and both are reported:

    * ``bilinear`` -- **the default, and the one vanilla SAM2 uses** in
      ``SAM2Transforms.postprocess_masks``;
    * ``nearest``  -- block replication, a stricter view that penalises boundary
      placement on small objects much more heavily.
    """

    if logits.dim() == 3:
        logits = logits.unsqueeze(1)
    if mode == "nearest":
        return F.interpolate(logits.float(), size=size, mode="nearest")
    return F.interpolate(logits.float(), size=size, mode="bilinear", align_corners=False)


def downsample_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Nearest-neighbour resize of a GT mask to the decoder's logit resolution.

    Training choice, documented here and in the smoke report: GT is resized to
    the logit grid with nearest-neighbour so the target stays binary and
    deterministic. Evaluation upsamples the logits instead.
    """

    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return F.interpolate(mask.float(), size=size, mode="nearest")


def binary_iou(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """IoU of two boolean/0-1 tensors, per sample, returned as a 1-D tensor."""

    prediction = prediction.bool()
    target = target.bool()
    if prediction.dim() > 1:
        dims = tuple(range(1, prediction.dim()))
    else:
        dims = (0,)
    intersection = (prediction & target).sum(dim=dims).float()
    union = (prediction | target).sum(dim=dims).float()
    return (intersection + epsilon) / (union + epsilon)


def mask_iou_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    size: tuple[int, int],
    threshold: float = 0.0,
    mode: str = "bilinear",
) -> float:
    """Original-resolution IoU between thresholded upsampled logits and the GT mask.

    ``mode`` defaults to ``bilinear`` because that is what vanilla SAM2 itself
    does: ``sam2/utils/transforms.py::SAM2Transforms.postprocess_masks`` ends with
    ``F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)``.
    Evaluating with anything else would measure a different pipeline from the one
    being built. ``nearest`` is also reported, as the stricter block-replication
    view, so the sensitivity of the number to this choice is visible.

    The target is moved to the logits' device so the metric can be called with a
    CPU ground-truth mask and GPU predictions.
    """

    upsampled = upsample_logits(logits, size, mode=mode)
    prediction = upsampled > threshold
    target = target.to(prediction.device)
    if target.dim() == 2:
        target = target.unsqueeze(0).unsqueeze(0)
    return float(binary_iou(prediction, target).mean())


def collapse_flags(logits: torch.Tensor, threshold: float = 0.0) -> dict:
    """Detect empty / full-image collapse, which must never count as success."""

    prediction = logits > threshold
    fraction = float(prediction.float().mean())
    return {
        "positive_fraction": fraction,
        "empty": fraction < 1e-6,
        "full": fraction > 1.0 - 1e-6,
        "constant_logits": bool(float(logits.std()) < 1e-9),
    }
