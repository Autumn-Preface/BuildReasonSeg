"""Task 6F: Target-Aware `[BOX]` Query Grounding v0.1.

Task 6E showed that asking a 2B model to *emit* a 256-way coordinate vocabulary fails on the
480-record paired mini-train. Task 6F keeps the box an explicit supervision target but changes
**where** the representation is formed and **how** it is read:

    USER: image + instruction
    ASSISTANT PREFIX: [BOX]                  <- fixed learned query token, identical every sample
    ASSISTANT TARGET: reasoning_zh [SEG] EOS

The `[BOX]` token is an inserted input (never predicted), placed immediately after the
generation prefix and *before* any reasoning text. Its hidden state is therefore conditioned by
image + instruction only, and a deliberately simple readout head — the same LayerNorm ->
Linear -> GELU -> Linear -> sigmoid structure Task 6D used — regresses the box directly:

    [BOX] hidden (2048-d) -> TargetAwareBoxHead -> (x1, y1, x2, y2) in [0,1], canonical

    L_total = 1.0 * L_reasoning + 5.0 * L_box

with `L_reasoning` the causal CE over `reasoning_zh [SEG] EOS` only (the `[BOX]` token itself is
never a prediction target) and `L_box` SmoothL1 against the frozen Task 6D tight-box GT.

Nothing from Task 6E's coordinate-token path is reused: no `<loc_*>` tokens are added, no
location-token CE, no location-run parsing, no Task 6E checkpoint initialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .grounding import GEOMETRY_BOX, SpatialGroundingHead, geometry_smooth_l1, target_geometry
from .losses import lm_cross_entropy
from .qwen_seg import TeacherForcedBatch

BOX_QUERY_TOKEN = "[BOX]"

#: Section 7 fixes these; they are constants, not hyperparameters to sweep.
REASONING_LOSS_WEIGHT = 1.0
BOX_LOSS_WEIGHT = 5.0


# ------------------------------------------------------------------ token setup


@dataclass
class BoxQuerySetup:
    """What adding the `[BOX]` query token did to the tokenizer and the embedding table."""

    box_token_id: int
    vocab_before: int
    vocab_after: int
    added_tokens: int
    embedding_shape_before: tuple
    embedding_shape_after: tuple
    tied_embeddings: bool
    single_token_roundtrip: bool

    def as_dict(self) -> dict:
        return {
            "box_token": BOX_QUERY_TOKEN,
            "box_token_id": int(self.box_token_id),
            "vocab_before": int(self.vocab_before),
            "vocab_after": int(self.vocab_after),
            "added_tokens": int(self.added_tokens),
            "embedding_shape_before": list(self.embedding_shape_before),
            "embedding_shape_after": list(self.embedding_shape_after),
            "tied_embeddings": bool(self.tied_embeddings),
            "single_token_roundtrip": bool(self.single_token_roundtrip),
        }


def add_box_query_token(model, tokenizer, token: str = BOX_QUERY_TOKEN) -> BoxQuerySetup:
    """Add exactly one `[BOX]` query token and resize the embedding table once."""

    vocab_before = len(tokenizer)
    embedding_before = tuple(model.get_input_embeddings().weight.shape)

    tokenizer.add_special_tokens({"additional_special_tokens": [token]})
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"{token} did not become a single token: {ids}")
    box_token_id = int(ids[0])

    model.resize_token_embeddings(len(tokenizer))

    decoded = tokenizer.decode([box_token_id], skip_special_tokens=False)
    return BoxQuerySetup(
        box_token_id=box_token_id,
        vocab_before=vocab_before,
        vocab_after=len(tokenizer),
        added_tokens=1,
        embedding_shape_before=embedding_before,
        embedding_shape_after=tuple(model.get_input_embeddings().weight.shape),
        tied_embeddings=bool(
            model.get_input_embeddings().weight.data_ptr()
            == model.get_output_embeddings().weight.data_ptr()
        ),
        single_token_roundtrip=bool(token in decoded),
    )


# ------------------------------------------------------------------ head


class TargetAwareBoxHead(SpatialGroundingHead):
    """Section 5: the `[BOX]` hidden state regressed to a canonical normalized box.

    Deliberately the **same readout** as Task 6D's `SpatialGroundingHead` with `kind="box"`
    (LayerNorm(2048) -> Linear(2048, 512) -> GELU -> Linear(512, 4) -> sigmoid -> min/max
    canonicalization), so the only experimental change is *where the supervised representation
    is formed*: the pre-reasoning `[BOX]` query instead of the end-of-reasoning `[SEG]` token.
    """

    def __init__(self, hidden_dim: int = 2048, mid_dim: int = 512) -> None:
        super().__init__(hidden_dim=hidden_dim, mid_dim=mid_dim, kind=GEOMETRY_BOX)

    def as_dict(self) -> dict:
        report = super().as_dict()
        report["class"] = "TargetAwareBoxHead"
        report["input"] = "[BOX] hidden (2048-d), pre-reasoning query position"
        return report


# ------------------------------------------------------------------ batch


def build_box_query_batch(
    processor,
    tokenizer,
    image,
    instruction: str,
    reasoning: str,
    box_token_id: int,
    seg_token_id: int,
    append_eos: bool = True,
) -> TeacherForcedBatch:
    """`prompt + [BOX] + reasoning + [SEG] + EOS` with the query excluded from the labels.

    The supervised span starts AT the `[BOX]` position: the label there is the first reasoning
    token, and the label at the last prompt position stays `-100`, so the model never learns to
    predict `[BOX]` — it is an inserted query (section 4).
    """

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_ids = prompt["input_ids"]
    prompt_length = int(prompt_ids.shape[1])

    reasoning_ids = tokenizer(reasoning, add_special_tokens=False, return_tensors="pt")["input_ids"]
    target_pieces = [reasoning_ids]
    target_pieces.append(torch.tensor([[int(seg_token_id)]], dtype=reasoning_ids.dtype))
    if append_eos and tokenizer.eos_token_id is not None:
        target_pieces.append(torch.tensor([[int(tokenizer.eos_token_id)]], dtype=reasoning_ids.dtype))
    target_ids = torch.cat(target_pieces, dim=1)

    input_ids = torch.cat([prompt_ids, torch.tensor([[int(box_token_id)]], dtype=prompt_ids.dtype), target_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)
    total_length = int(input_ids.shape[1])
    box_position = prompt_length

    labels = torch.full_like(input_ids, -100)
    # logits at position i predict token i+1; the span starts at the query position, so the
    # position that would predict [BOX] (prompt_length - 1) stays -100.
    labels[0, box_position : total_length - 1] = input_ids[0, box_position + 1 : total_length]

    seg_positions = (input_ids[0] == seg_token_id).nonzero(as_tuple=False).flatten()
    if seg_positions.numel() != 1:
        raise RuntimeError(f"expected exactly one [SEG] token, found {seg_positions.numel()}")
    seg_position = int(seg_positions[0])
    if int((input_ids[0] == box_token_id).sum()) != 1:
        raise RuntimeError("the [BOX] query token must appear exactly once")
    if seg_position <= box_position:
        raise RuntimeError("the [BOX] query must precede the reasoning target")

    grid = prompt.get("image_grid_thw")
    visual_tokens = 0
    if grid is not None:
        merge = getattr(processor.image_processor, "merge_size", 2) or 2
        visual_tokens = int(grid.prod(dim=-1).sum().item()) // (int(merge) ** 2)

    extra = {
        key: value
        for key, value in prompt.items()
        if key not in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
    }
    extended_extra: dict = {}
    for key, value in extra.items():
        if torch.is_tensor(value) and value.dim() >= 1 and value.shape[-1] == prompt_length:
            pad_shape = list(value.shape)
            pad_shape[-1] = total_length - prompt_length
            pad = torch.zeros(pad_shape, dtype=value.dtype, device=value.device)
            extended_extra[key] = torch.cat([value, pad], dim=-1)
        else:
            extended_extra[key] = value

    return TeacherForcedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        pixel_values=prompt.get("pixel_values"),
        image_grid_thw=grid,
        prompt_length=prompt_length,
        total_length=total_length,
        seg_position=seg_position,
        visual_tokens=visual_tokens,
        box_position=box_position,
        extra_inputs=extended_extra,
    )


# ------------------------------------------------------------------ forward / loss


def box_query_forward(model, batch: TeacherForcedBatch):
    """One pass returning LM logits, the `[BOX]` hidden and the `[SEG]` hidden."""

    outputs = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        pixel_values=batch.pixel_values,
        image_grid_thw=batch.image_grid_thw,
        output_hidden_states=True,
        use_cache=False,
        **batch.extra_inputs,
    )
    box_hidden = outputs.hidden_states[-1][:, int(batch.box_position), :]
    seg_hidden = outputs.hidden_states[-1][:, int(batch.seg_position), :]
    return outputs.logits, box_hidden, seg_hidden


def box_query_loss(lm_logits: torch.Tensor, batch: TeacherForcedBatch,
                   predicted_box: torch.Tensor, gt_box: torch.Tensor) -> dict:
    """`1.0 * L_reasoning + 5.0 * L_box` (section 7); both raw losses recorded."""

    reasoning_ce = lm_cross_entropy(lm_logits, batch.labels)
    box_loss = geometry_smooth_l1(predicted_box, gt_box)
    total = REASONING_LOSS_WEIGHT * reasoning_ce + BOX_LOSS_WEIGHT * box_loss
    with torch.no_grad():
        box_iou = box_iou_tensor(predicted_box.detach(), gt_box.detach())
    return {
        "total": total,
        "reasoning_ce": reasoning_ce,
        "box_loss": box_loss,
        "reasoning_ce_raw": float(reasoning_ce.detach()),
        "box_loss_raw": float(box_loss.detach()),
        "total_raw": float(total.detach()),
        "box_iou": float(box_iou),
        "reasoning_weight": REASONING_LOSS_WEIGHT,
        "box_weight": BOX_LOSS_WEIGHT,
    }


def box_iou_tensor(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Elementwise IoU of two canonical boxes in [0,1]."""

    x1 = torch.maximum(left[..., 0], right[..., 0])
    y1 = torch.maximum(left[..., 1], right[..., 1])
    x2 = torch.minimum(left[..., 2], right[..., 2])
    y2 = torch.minimum(left[..., 3], right[..., 3])
    intersection = torch.clamp(x2 - x1, min=0.0) * torch.clamp(y2 - y1, min=0.0)
    area_left = torch.clamp(left[..., 2] - left[..., 0], min=0.0) * torch.clamp(
        left[..., 3] - left[..., 1], min=0.0
    )
    area_right = torch.clamp(right[..., 2] - right[..., 0], min=0.0) * torch.clamp(
        right[..., 3] - right[..., 1], min=0.0
    )
    union = area_left + area_right - intersection
    return torch.where(union > 0, intersection / torch.clamp(union, min=1e-12), torch.zeros_like(intersection))


def box_values_iou(left, right) -> float:
    """Numpy-friendly IoU of two 4-value boxes."""

    return float(box_iou_tensor(torch.as_tensor(left, dtype=torch.float32),
                                torch.as_tensor(right, dtype=torch.float32)))


def center_inside_mask(mask, box) -> bool:
    """Whether the box centre pixel lies inside the binary target mask."""

    import numpy as np

    binary = np.asarray(mask).astype(bool)
    height, width = binary.shape
    cx = int(round(0.5 * (float(box[0]) + float(box[2])) * (width - 1)))
    cy = int(round(0.5 * (float(box[1]) + float(box[3])) * (height - 1)))
    if not (0 <= cx < width and 0 <= cy < height):
        return False
    return bool(binary[cy, cx])


def box_l1(left, right) -> float:
    """Mean absolute coordinate distance between two boxes."""

    return float(sum(abs(float(a) - float(b)) for a, b in zip(left, right)) / 4.0)


# ------------------------------------------------------------------ inference-form query path


@torch.no_grad()
def box_hidden_from_batch(model, batch: TeacherForcedBatch) -> torch.Tensor:
    """The `[BOX]` hidden state only (the evaluation path never reads `[SEG]`)."""

    outputs = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        pixel_values=batch.pixel_values,
        image_grid_thw=batch.image_grid_thw,
        output_hidden_states=True,
        use_cache=False,
        **batch.extra_inputs,
    )
    return outputs.hidden_states[-1][:, int(batch.box_position), :]


@torch.no_grad()
def predict_box_from_batch(runtime, batch: TeacherForcedBatch) -> dict:
    """The section 9 inference-form path: prompt + constant `[BOX]` -> head -> box.

    The batch here must contain ONLY the prompt and the query token (no future reasoning),
    which is exactly what the causal attention mask guarantees in training as well.
    """

    batch = batch.to(runtime.device)
    with torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=bool(runtime.cfg.get("training", {}).get("bf16_autocast", True)),
    ):
        box_hidden = box_hidden_from_batch(runtime.model.qwen, batch)
        predicted = runtime.model.box_head(box_hidden)
    return {
        "predicted_box": [float(value) for value in predicted[0].detach().float().cpu().tolist()],
        "box_hidden": box_hidden[0].detach().float().cpu().clone(),
        "canonical": bool(float(predicted[0, 0]) <= float(predicted[0, 2]) and float(predicted[0, 1]) <= float(predicted[0, 3])),
        "in_range": bool(float(predicted.min()) >= 0.0 and float(predicted.max()) <= 1.0),
    }


def build_query_batch(runtime, image, instruction: str) -> TeacherForcedBatch:
    """`prompt + [BOX]` only — the evaluation/generation input, with no future tokens.

    There is deliberately no `[SEG]` and no target in this batch: the query hidden must be
    produced by exactly the tokens it may attend to (image + instruction + prefix + itself).
    """

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    prompt = runtime.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_ids = prompt["input_ids"]
    prompt_length = int(prompt_ids.shape[1])
    input_ids = torch.cat(
        [prompt_ids, torch.tensor([[int(runtime.box_setup.box_token_id)]], dtype=prompt_ids.dtype)], dim=1
    )
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    total_length = int(input_ids.shape[1])

    grid = prompt.get("image_grid_thw")
    visual_tokens = 0
    if grid is not None:
        merge = getattr(runtime.processor.image_processor, "merge_size", 2) or 2
        visual_tokens = int(grid.prod(dim=-1).sum().item()) // (int(merge) ** 2)

    extra = {
        key: value
        for key, value in prompt.items()
        if key not in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
    }
    extended_extra: dict = {}
    for key, value in extra.items():
        if torch.is_tensor(value) and value.dim() >= 1 and value.shape[-1] == prompt_length:
            pad_shape = list(value.shape)
            pad_shape[-1] = total_length - prompt_length
            pad = torch.zeros(pad_shape, dtype=value.dtype, device=value.device)
            extended_extra[key] = torch.cat([value, pad], dim=-1)
        else:
            extended_extra[key] = value

    return TeacherForcedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        pixel_values=prompt.get("pixel_values"),
        image_grid_thw=grid,
        prompt_length=prompt_length,
        total_length=total_length,
        seg_position=-1,  # no [SEG] in the query-only batch
        visual_tokens=visual_tokens,
        box_position=prompt_length,
        extra_inputs=extended_extra,
    )


def predict_box_for_sample(runtime, sample, image=None) -> dict:
    """One validation record -> predicted box through the inference-form query path."""

    if image is None:
        image = sample.image_rgb()
    runtime.set_visual_cache_key(str(sample.image_id))
    batch = build_query_batch(runtime, image, sample.instruction_zh)
    return predict_box_from_batch(runtime, batch)


@torch.no_grad()
def generate_with_box_prefix(model, processor, tokenizer, image, instruction: str,
                             box_token_id: int, seg_token_id: int,
                             max_new_tokens: int = 128) -> dict:
    """Section 14: continue generation from `image + instruction + fixed [BOX]`.

    The query is inserted exactly as in training; the generated suffix is reasoning text,
    which the caller inspects for `[SEG]` emission and EOS termination.
    """

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    prompt_length = int(inputs["input_ids"].shape[1])
    box_ids = torch.tensor([[int(box_token_id)]], dtype=inputs["input_ids"].dtype)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], box_ids], dim=1)
    if "attention_mask" in inputs:
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones_like(box_ids)], dim=1
        )
    for key, value in list(inputs.items()):
        if torch.is_tensor(value) and value.dim() >= 1 and value.shape[-1] == prompt_length:
            pad_shape = list(value.shape)
            pad_shape[-1] = 1
            pad = torch.zeros(pad_shape, dtype=value.dtype)
            inputs[key] = torch.cat([value, pad], dim=-1)

    device = next(model.parameters()).device
    model_inputs = {
        key: (value.to(device) if torch.is_tensor(value) else value) for key, value in inputs.items()
    }
    generated = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
    query_length = int(model_inputs["input_ids"].shape[1])
    new_tokens = generated[0, query_length:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    seg_positions = (generated[0] == seg_token_id).nonzero(as_tuple=False).flatten()
    return {
        "text": text,
        "token_ids": generated[0].tolist(),
        "query_length": query_length,
        "seg_positions": seg_positions.tolist(),
        "seg_count": int(seg_positions.numel()),
        "terminated_with_eos": bool(
            new_tokens.numel() > 0
            and int(new_tokens[-1]) == int(tokenizer.eos_token_id or -1)
        ),
        "generated_token_count": int(new_tokens.numel()),
    }
