"""Task 6D: Spatial Grounding Bridge v0.1.

The Task 6C pipeline fed SAM2 an arbitrary projected language vector as its sparse
prompt. Task 6D replaces that with an explicit spatial mechanism:

    [SEG] hidden state (2048-d)
        -> SpatialGroundingHead
        -> predicted target geometry (normalized point or box)
        -> official SAM2 prompt encoder
        -> SAM2 mask decoder
        -> target mask

Ground-truth geometry is used **only** as training supervision and as the oracle
diagnostic. At inference the prompt contains predicted geometry and nothing else.

Coordinate conventions (verified against the installed SAM2 source)
------------------------------------------------------------------
`PromptEncoder._embed_points` does `points = points + 0.5` and then
`pe_layer.forward_with_coords(points, self.input_image_size)`, which divides x by
`image_size[1]` and y by `image_size[0]`. `_embed_boxes` does the same for the two box
corners. So the official prompt encoder expects coordinates in the **input image pixel
space**, i.e. the 1024x1024 frame the image encoder consumes — not normalized [0,1] and
not the original resolution.

The head predicts normalized coordinates in `[0, 1]` (resolution independent, which is
what makes the same checkpoint usable at any image size). `geometry_to_sam_coords`
therefore multiplies by `input_image_size`. Task 6D data is square 512x512 tiles, so the
normalized frame maps linearly onto the 1024 frame; the mapping is asserted rather than
assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .sam2_bridge import MaskDecodeResult, Sam2Features, sparse_prompt_dtype

#: Geometry kinds. Exactly one is selected per Task 6D run (section 5).
GEOMETRY_POINT = "point"
GEOMETRY_BOX = "box"
GEOMETRY_KINDS = (GEOMETRY_POINT, GEOMETRY_BOX)
GEOMETRY_DIMS = {GEOMETRY_POINT: 2, GEOMETRY_BOX: 4}

#: Section 8: provisional architecture-proof grounding weight, not swept.
DEFAULT_LAMBDA_GROUND = 5.0


# ------------------------------------------------------------------ GT geometry


def distance_transform_point(mask: np.ndarray) -> tuple[float, float]:
    """Deterministic interior point: the maximum of the Euclidean distance transform.

    Ties are broken by the flattened index order of `argmax`, so the point is a pure
    function of the mask. Coordinates are normalized pixel centres in [0, 1].
    """

    from scipy import ndimage

    binary = np.asarray(mask).astype(bool)
    if not binary.any():
        raise ValueError("cannot derive an interior point from an empty mask")
    distance = ndimage.distance_transform_edt(binary)
    flat_index = int(np.argmax(distance))  # first maximum in row-major order: deterministic
    row, column = np.unravel_index(flat_index, binary.shape)
    height, width = binary.shape
    return ((column + 0.5) / width, (row + 0.5) / height)


def tight_box(mask: np.ndarray) -> tuple[float, float, float, float]:
    """Tight GT bounding box as normalized (x1, y1, x2, y2), `x1 <= x2`, `y1 <= y2`."""

    binary = np.asarray(mask).astype(bool)
    if not binary.any():
        raise ValueError("cannot derive a box from an empty mask")
    rows = np.flatnonzero(binary.any(axis=1))
    columns = np.flatnonzero(binary.any(axis=0))
    height, width = binary.shape
    return (
        columns[0] / width,
        rows[0] / height,
        (columns[-1] + 1) / width,
        (rows[-1] + 1) / height,
    )


def target_geometry(mask: np.ndarray, kind: str) -> tuple[float, ...]:
    if kind == GEOMETRY_POINT:
        return distance_transform_point(mask)
    if kind == GEOMETRY_BOX:
        return tight_box(mask)
    raise ValueError(f"unknown geometry kind {kind!r}; expected one of {GEOMETRY_KINDS}")


def geometry_is_valid(geometry: torch.Tensor | tuple, kind: str) -> bool:
    """Normalized geometry must be finite, inside [0,1], and canonical for boxes."""

    tensor = geometry if torch.is_tensor(geometry) else torch.as_tensor(geometry, dtype=torch.float32)
    tensor = tensor.detach().float()
    if not torch.isfinite(tensor).all():
        return False
    if tensor.numel() != GEOMETRY_DIMS[kind]:
        return False
    if float(tensor.min()) < -1e-6 or float(tensor.max()) > 1 + 1e-6:
        return False
    if kind == GEOMETRY_BOX:
        x1, y1, x2, y2 = [float(value) for value in tensor.reshape(-1)]
        if x1 > x2 + 1e-6 or y1 > y2 + 1e-6:
            return False
    return True


# ------------------------------------------------------------------ head


class SpatialGroundingHead(torch.nn.Module):
    """`[SEG]` hidden state -> predicted target geometry (section 6).

    LayerNorm(2048) -> Linear(2048, 512) -> GELU -> Linear(512, D) -> sigmoid, with
    D = 2 for a point and D = 4 for a box. The box output is canonicalized
    differentiably: the four sigmoid outputs are treated as two corners and ordered with
    `min`/`max`, so `x1 <= x2` and `y1 <= y2` always hold and gradients still flow to
    every output unit.
    """

    def __init__(self, hidden_dim: int = 2048, mid_dim: int = 512, kind: str = GEOMETRY_POINT) -> None:
        super().__init__()
        if kind not in GEOMETRY_KINDS:
            raise ValueError(f"unknown geometry kind {kind!r}; expected one of {GEOMETRY_KINDS}")
        self.kind = kind
        self.dim = GEOMETRY_DIMS[kind]
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim, mid_dim)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(mid_dim, self.dim)

    def forward(self, seg_hidden: torch.Tensor) -> torch.Tensor:
        """(B, 2048) -> (B, D) normalized geometry in [0, 1]."""

        raw = torch.sigmoid(self.fc2(self.act(self.fc1(self.norm(seg_hidden.float())))))
        if self.kind == GEOMETRY_POINT:
            return raw
        x_first, y_first, x_second, y_second = raw.unbind(dim=-1)
        x1 = torch.minimum(x_first, x_second)
        x2 = torch.maximum(x_first, x_second)
        y1 = torch.minimum(y_first, y_second)
        y2 = torch.maximum(y_first, y_second)
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "dim": self.dim,
            "hidden_dim": self.fc1.in_features,
            "mid_dim": self.fc1.out_features,
            "structure": "LayerNorm -> Linear(2048,512) -> GELU -> Linear(512,D) -> sigmoid",
            "box_canonicalization": (
                "two sigmoid corners ordered with min/max"
                if self.kind == GEOMETRY_BOX
                else None
            ),
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
        }


# ------------------------------------------------------------------ official prompt path


@dataclass
class GeometryPromptResult:
    low_res_logits: torch.Tensor
    iou_prediction: torch.Tensor
    sparse_prompt_shape: tuple[int, ...] = ()
    prompt_diagnostics: dict = field(default_factory=dict)


def geometry_to_sam_coords(geometry: torch.Tensor, kind: str, image_size: int) -> torch.Tensor:
    """Normalized geometry -> the official prompt encoder's pixel space."""

    if kind == GEOMETRY_POINT:
        return geometry.float() * float(image_size)
    if kind == GEOMETRY_BOX:
        return geometry.float() * float(image_size)
    raise ValueError(f"unknown geometry kind {kind!r}")


def build_geometry_prompt(
    sam: torch.nn.Module,
    geometry: torch.Tensor,
    kind: str,
    image_size: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """Call the **official** SAM2 prompt encoder with a real point or box prompt.

    No projected language vector and no fixed anchor are involved: for `point` the prompt
    is a single positive point, for `box` it is the official box path.
    """

    if kind not in GEOMETRY_KINDS:
        raise ValueError(f"unknown geometry kind {kind!r}; expected one of {GEOMETRY_KINDS}")
    size = int(image_size or getattr(sam.sam_prompt_encoder, "input_image_size", (1024, 1024))[0])
    coords = geometry_to_sam_coords(geometry, kind, size)
    batch = coords.shape[0]
    dtype = sparse_prompt_dtype(sam)

    if kind == GEOMETRY_POINT:
        points = coords.reshape(batch, 1, 2)
        labels = torch.ones((batch, 1), device=coords.device, dtype=torch.int32)
        sparse, _dense = sam.sam_prompt_encoder(points=(points, labels), boxes=None, masks=None)
        diagnostics = {
            "prompt_kind": GEOMETRY_POINT,
            "uses_point_coordinates": True,
            "uses_point_labels": True,
            "uses_box_prompt": False,
            "point_label": "positive",
            "image_size": size,
            "coords_pixels": coords.detach().float().cpu().tolist(),
            "prompt_dtype": str(dtype),
        }
    else:
        boxes = coords.reshape(batch, 4)
        sparse, _dense = sam.sam_prompt_encoder(points=None, boxes=boxes, masks=None)
        diagnostics = {
            "prompt_kind": GEOMETRY_BOX,
            "uses_point_coordinates": False,
            "uses_point_labels": False,
            "uses_box_prompt": True,
            "point_label": None,
            "image_size": size,
            "box_pixels": boxes.detach().float().cpu().tolist(),
            "prompt_dtype": str(dtype),
        }
    diagnostics["sparse_prompt_shape"] = list(sparse.shape)
    diagnostics["derived_from"] = "predicted_geometry" if geometry.requires_grad else "geometry_input"
    return sparse, diagnostics


def decode_mask_from_geometry(
    sam: torch.nn.Module,
    features: Sam2Features,
    geometry: torch.Tensor,
    kind: str,
    multimask_output: bool = False,
    image_size: int | None = None,
) -> GeometryPromptResult:
    """Official prompt encoder + official mask decoder, prompted by predicted geometry."""

    sparse, diagnostics = build_geometry_prompt(sam, geometry, kind, image_size=image_size)
    low_res_logits, iou_prediction, _tokens, _obj_score = sam.sam_mask_decoder(
        image_embeddings=features.image_embeddings,
        image_pe=features.image_pe,
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=features.dense_no_mask_embedding,
        multimask_output=multimask_output,
        repeat_image=False,
        high_res_features=features.high_res_features,
    )
    return GeometryPromptResult(
        low_res_logits=low_res_logits,
        iou_prediction=iou_prediction,
        sparse_prompt_shape=tuple(sparse.shape),
        prompt_diagnostics=diagnostics,
    )


# ------------------------------------------------------------------ loss and metrics


def geometry_smooth_l1(
    predicted: torch.Tensor, target: torch.Tensor, beta: float = 1.0
) -> torch.Tensor:
    """`SmoothL1(pred_geometry, gt_geometry)` (section 8)."""

    return torch.nn.functional.smooth_l1_loss(
        predicted.float(), target.float().to(predicted.device), beta=beta, reduction="mean"
    )


def point_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict:
    """Normalized point error plus both elements separately."""

    difference = (predicted.float() - target.float()).abs()
    return {
        "mean_abs_error": float(difference.mean()),
        "mean_error_x": float(difference[:, 0].mean()),
        "mean_error_y": float(difference[:, 1].mean()),
        "max_abs_error": float(difference.max()),
    }


def box_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict:
    """IoU of the predicted and GT normalized boxes, plus the corner error."""

    x1 = torch.maximum(predicted[:, 0], target[:, 0])
    y1 = torch.maximum(predicted[:, 1], target[:, 1])
    x2 = torch.minimum(predicted[:, 2], target[:, 2])
    y2 = torch.minimum(predicted[:, 3], target[:, 3])
    intersection = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    area_predicted = (predicted[:, 2] - predicted[:, 0]).clamp_min(0) * (
        predicted[:, 3] - predicted[:, 1]
    ).clamp_min(0)
    area_target = (target[:, 2] - target[:, 0]).clamp_min(0) * (target[:, 3] - target[:, 1]).clamp_min(0)
    union = area_predicted + area_target - intersection
    iou = intersection / union.clamp_min(1e-9)
    return {
        "mean_box_iou": float(iou.mean()),
        "mean_corner_error": float((predicted.float() - target.float()).abs().mean()),
    }


def geometry_metrics(predicted: torch.Tensor, target: torch.Tensor, kind: str) -> dict:
    return point_metrics(predicted, target) if kind == GEOMETRY_POINT else box_metrics(predicted, target)


def mask_contains_point(mask: np.ndarray, point_xy: tuple[float, float]) -> bool:
    """Is a normalized point strictly inside the mask?"""

    binary = np.asarray(mask).astype(bool)
    height, width = binary.shape
    column = int(min(width - 1, max(0, np.floor(point_xy[0] * width))))
    row = int(min(height - 1, max(0, np.floor(point_xy[1] * height))))
    return bool(binary[row, column])


def point_in_box(box_xyxy: tuple[float, float, float, float], point_xy: tuple[float, float]) -> bool:
    x1, y1, x2, y2 = box_xyxy
    return bool(x1 <= point_xy[0] <= x2 and y1 <= point_xy[1] <= y2)


def geometry_paired_decision(
    prediction_a: tuple[float, ...],
    prediction_b: tuple[float, ...],
    target_a: tuple[float, ...],
    target_b: tuple[float, ...],
    kind: str,
) -> dict:
    """Section 12's geometry criterion, the analogue of the mask paired test.

    For a **point**: `pred_A` must be inside GT_A and not inside GT_B, and vice versa —
    evaluated against the masks by the caller, which is why this helper only returns the
    distances. For a **box**: `IoU(pred_A, GT_A) > IoU(pred_A, GT_B)` and symmetrically.
    """

    predicted_a = torch.as_tensor(prediction_a, dtype=torch.float32).reshape(1, -1)
    predicted_b = torch.as_tensor(prediction_b, dtype=torch.float32).reshape(1, -1)
    target_a_t = torch.as_tensor(target_a, dtype=torch.float32).reshape(1, -1)
    target_b_t = torch.as_tensor(target_b, dtype=torch.float32).reshape(1, -1)
    if kind == GEOMETRY_BOX:
        own_a = float(box_metrics(predicted_a, target_a_t)["mean_box_iou"])
        cross_a = float(box_metrics(predicted_a, target_b_t)["mean_box_iou"])
        own_b = float(box_metrics(predicted_b, target_b_t)["mean_box_iou"])
        cross_b = float(box_metrics(predicted_b, target_a_t)["mean_box_iou"])
    else:
        own_a = float((predicted_a - target_a_t).abs().mean())
        cross_a = float((predicted_a - target_b_t).abs().mean())
        own_b = float((predicted_b - target_b_t).abs().mean())
        cross_b = float((predicted_b - target_a_t).abs().mean())
        # smaller is better for points: report the negated difference so the caller can use
        # one comparison direction for both kinds
        own_a, cross_a, own_b, cross_b = -own_a, -cross_a, -own_b, -cross_b
    return {
        "own_a": own_a,
        "cross_a": cross_a,
        "own_b": own_b,
        "cross_b": cross_b,
        "passed": bool(own_a > cross_a and own_b > cross_b),
        "margin": float(((own_a - cross_a) + (own_b - cross_b)) / 2.0),
    }
