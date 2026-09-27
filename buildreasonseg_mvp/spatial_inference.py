"""Task 6E section 19: reusable inference plumbing for the explicit spatial-token pathway.

    image + instruction -> reasoning -> [BOX] <loc_x1> <loc_y1> <loc_x2> <loc_y2> -> [SEG]

Four small functions, no GUI and no top-level `predict.py` (that waits for review):

* `generate_spatial_tokens(image, instruction)` -- free generation, token ids out;
* `parse_box_tokens(token_ids)` -- token-id parse, no string heuristics;
* `predict_box(image, instruction)` -- dequantized box for one request;
* `predict_mask_from_generated_box(image, instruction)` -- that box as the official frozen
  SAM2 box prompt, returning a mask.

Ground truth never enters any of these paths.
"""

from __future__ import annotations

import torch

from .grounding import decode_mask_from_geometry
from .metrics import upsample_logits
from .qwen_seg import generate_with_seg
from .spatial_tokens import box_iou_values, parse_generated_ids


def generate_spatial_tokens(
    runtime,
    image,
    instruction: str,
    *,
    max_new_tokens: int | None = None,
    image_id: str | None = None,
) -> dict:
    """Generate the assistant answer for one image + instruction and return its token ids."""

    runtime.set_visual_cache_key(image_id)
    limit = int(
        max_new_tokens
        if max_new_tokens is not None
        else runtime.cfg.get("inference", {}).get("validation_max_new_tokens", 96)
    )
    generation = generate_with_seg(
        runtime.model.qwen,
        runtime.processor,
        runtime.tokenizer,
        image,
        instruction,
        max_new_tokens=limit,
        seg_token_id=runtime.model.seg_token_id,
    )
    prompt_length = int(generation["prompt_length"])
    return {
        "generated_token_ids": [int(value) for value in generation["token_ids"][prompt_length:]],
        "full_token_ids": [int(value) for value in generation["token_ids"]],
        "prompt_length": prompt_length,
        "text": generation["text"],
    }


def parse_box_tokens(runtime, token_ids) -> dict:
    """Parse generated ids into `(codes, box, structural validity)`."""

    parsed = parse_generated_ids(
        token_ids, runtime.spatial_setup, runtime.spatial_codec, runtime.model.seg_token_id
    )
    return parsed.as_dict()


def predict_box(runtime, image, instruction: str, **kwargs) -> dict:
    """One request -> one normalized `(x1, y1, x2, y2)` prediction (or a structural failure)."""

    generation = generate_spatial_tokens(runtime, image, instruction, **kwargs)
    parsed = parse_box_tokens(runtime, generation["generated_token_ids"])
    return {**parsed, "generated_token_ids": generation["generated_token_ids"], "text": generation["text"]}


@torch.no_grad()
def predict_mask_from_generated_box(runtime, image, instruction: str, **kwargs) -> dict:
    """Generated box -> official frozen SAM2 box prompt -> mask (Task 6E section 14)."""

    prediction = predict_box(runtime, image, instruction, **kwargs)
    if not prediction["structural_valid"]:
        return {**prediction, "mask": None, "miou": None, "reason": prediction["failure"]}
    features, cached = runtime.features_for_image(image, image_id=kwargs.get("image_id"))
    geometry = torch.tensor(prediction["box"], dtype=torch.float32)
    decoded = decode_mask_from_geometry(runtime.model.sam, features, geometry, "box", multimask_output=False)
    logits = decoded.low_res_logits
    height, width = int(image.shape[0]), int(image.shape[1])
    logits = upsample_logits(logits, (height, width), mode="bilinear")
    mask = (logits[0, 0] > 0).detach().cpu().numpy().reshape(height, width)
    return {**prediction, "mask": mask, "cached_features": bool(cached)}


def box_iou(left, right) -> float:
    """Convenience IoU for two normalized boxes (used by the error analysis)."""

    return float(box_iou_values(left, right))
