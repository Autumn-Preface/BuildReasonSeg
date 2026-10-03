"""Task 6E sections 8-9: token-aware teacher forcing, position masks and the objective.

The Task 6E assistant target is built from **explicit token ids**, never by decoding a
string:

    reasoning  [BOX]  <loc_x1> <loc_y1> <loc_x2> <loc_y2>  [SEG]  (EOS)

so the position of every span is known exactly. This module exposes those positions as
*label positions* in the full sequence under the project's causal rule
(`logits at position i predict token i + 1`), and scores:

    L_total = 1.0 * L_assistant + 5.0 * L_location

with `L_assistant` the assistant-span causal CE and `L_location` the causal CE at exactly
the four location-token prediction positions. There is no mask loss, no SmoothL1 on
hidden states, and no loss-weight sweep (section 9).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .losses import lm_cross_entropy
from .spatial_tokens import (
    QuantizedBoxCodec,
    SpatialTokenSetup,
    build_target_ids,
)

#: Section 9 fixes these; they are constants, not hyperparameters to sweep.
ASSISTANT_LOSS_WEIGHT = 1.0
LOCATION_LOSS_WEIGHT = 5.0


def target_label_positions(prompt_length: int, spans: dict[str, list[int]]) -> dict[str, list[int]]:
    """Flat label positions for every assistant span.

    The assistant target starts at index `prompt_length` of the full sequence, and the
    assistant token at target offset `k` is predicted by the logits at position
    `prompt_length + k - 1`. Offset 0 is predicted by the last prompt token, which is
    exactly the project's causal-label rule.
    """

    positions: dict[str, list[int]] = {}
    for name, offsets in spans.items():
        positions[name] = [int(prompt_length) + int(offset) - 1 for offset in offsets]
    return positions


@dataclass
class SpatialExample:
    """One supervised spatial-token example (GT geometry is supervision only)."""

    sample_id: str
    image_id: str
    level: int
    query_family: str | None
    reasoning: str
    codes: list[int]
    box: list[float]
    gt_box: list[float]
    target_ids: list[int]
    spans: dict[str, list[int]]
    label_positions: dict[str, list[int]] = field(default_factory=dict)
    prompt_length: int = 0
    total_length: int = 0

    def as_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "image_id": self.image_id,
            "level": self.level,
            "query_family": self.query_family,
            "codes": list(self.codes),
            "box": list(self.box),
            "gt_box": list(self.gt_box),
            "target_length": len(self.target_ids),
            "spans": {key: list(value) for key, value in self.spans.items()},
            "label_positions": {key: list(value) for key, value in self.label_positions.items()},
            "prompt_length": self.prompt_length,
            "total_length": self.total_length,
        }


def build_spatial_example(
    sample,
    gt_mask,
    setup: SpatialTokenSetup,
    codec: QuantizedBoxCodec,
    tokenizer,
    seg_token_id: int,
    *,
    append_eos: bool = True,
) -> SpatialExample:
    """Quantize the GT box and build the explicit assistant target ids."""

    from .grounding import target_geometry
    from .validation import family_of

    gt_box = [float(value) for value in target_geometry(gt_mask, "box")]
    codes = codec.encode(gt_box)
    target_ids, spans = build_target_ids(
        tokenizer,
        sample.reasoning_zh,
        codes,
        setup,
        seg_token_id,
        append_eos=append_eos,
    )
    return SpatialExample(
        sample_id=str(sample.sample_id),
        image_id=str(sample.image_id),
        level=int(getattr(sample, "level", 0) or 0),
        query_family=family_of(str(sample.query_type)),
        reasoning=str(sample.reasoning_zh),
        codes=[int(value) for value in codes],
        box=[float(value) for value in codec.decode(codes)],
        gt_box=gt_box,
        target_ids=[int(value) for value in target_ids[0].tolist()],
        spans={key: [int(v) for v in value] for key, value in spans.items()},
    )


def spatial_batch(runtime, sample, image=None, *, append_eos: bool = True):
    """`(TeacherForcedBatch, SpatialExample)` for one sample on a spatial runtime."""

    from .qwen_seg import build_teacher_forcing_batch

    setup = runtime.spatial_setup
    if setup is None:
        raise RuntimeError("runtime has no spatial token setup; enable cfg['spatial_tokens']['bins']")
    codec = runtime.spatial_codec
    gt_mask = sample.target_mask()
    if image is None:
        image = sample.image_rgb()
    example = build_spatial_example(
        sample,
        gt_mask,
        setup,
        codec,
        runtime.tokenizer,
        runtime.model.seg_token_id,
        append_eos=append_eos,
    )
    batch = build_teacher_forcing_batch(
        runtime.processor,
        runtime.tokenizer,
        image,
        sample.instruction_zh,
        "",
        runtime.model.seg_token_id,
        target_ids=torch.tensor([example.target_ids], dtype=torch.long),
    )
    example.prompt_length = int(batch.prompt_length)
    example.total_length = int(batch.total_length)
    example.label_positions = target_label_positions(batch.prompt_length, example.spans)
    verify_label_positions(batch, example, setup)
    return batch, example


def verify_label_positions(batch, example: SpatialExample, setup: SpatialTokenSetup) -> None:
    """Off-by-one audit: every recorded label position must hold the expected token id.

    `labels[i]` is the token predicted by `logits[i]`, so it must equal
    `input_ids[i + 1]` and the token id that the span says belongs there.
    """

    input_ids = batch.input_ids[0]
    labels = batch.labels[0]
    if int(input_ids.shape[0]) != int(example.total_length):
        raise RuntimeError("example length disagrees with the batch")
    for name, offsets in example.spans.items():
        expected = [example.target_ids[offset] for offset in offsets]
        positions = example.label_positions[name]
        if len(positions) != len(expected):
            raise RuntimeError(f"span {name} position count mismatch")
        for position, token_id in zip(positions, expected):
            if int(labels[position]) != int(token_id):
                raise RuntimeError(
                    f"span {name}: label at {position} is {int(labels[position])}, expected {token_id}"
                )
            if int(input_ids[position + 1]) != int(token_id):
                raise RuntimeError(
                    f"span {name}: input at {position + 1} is {int(input_ids[position + 1])}, "
                    f"expected {token_id}"
                )
    if len(example.label_positions["loc"]) != 4:
        raise RuntimeError("the location span must contain exactly four positions")


def spatial_loss(lm_logits: torch.Tensor, batch, example: SpatialExample, setup: SpatialTokenSetup) -> dict:
    """`1.0 * L_assistant + 5.0 * L_location`, both means over their own positions."""

    labels = batch.labels
    assistant_ce = lm_cross_entropy(lm_logits, labels)

    loc_positions = example.label_positions["loc"]
    if len(loc_positions) != 4:
        raise RuntimeError("the location span must contain exactly four positions")
    loc_labels = labels[0, loc_positions]
    expected = torch.tensor(
        [example.target_ids[offset] for offset in example.spans["loc"]],
        dtype=loc_labels.dtype,
        device=loc_labels.device,
    )
    if not bool(torch.equal(loc_labels, expected)):
        raise RuntimeError("location label positions do not hold the expected loc token ids")
    loc_logits = lm_logits[0, loc_positions, :].float()
    location_ce = F.cross_entropy(loc_logits, loc_labels)

    with torch.no_grad():
        predicted = loc_logits.argmax(dim=-1)
        token_accuracy = float((predicted == loc_labels).float().mean())
        bin_error = float((predicted - loc_labels).abs().float().mean())

    total = ASSISTANT_LOSS_WEIGHT * assistant_ce + LOCATION_LOSS_WEIGHT * location_ce
    return {
        "total": total,
        "assistant_ce": assistant_ce,
        "location_ce": location_ce,
        "assistant_ce_raw": float(assistant_ce.detach()),
        "location_ce_raw": float(location_ce.detach()),
        "total_raw": float(total.detach()),
        "location_token_accuracy": token_accuracy,
        "location_abs_bin_error": bin_error,
        "assistant_weight": ASSISTANT_LOSS_WEIGHT,
        "location_weight": LOCATION_LOSS_WEIGHT,
    }


def spatial_teacher_forced_metrics(lm_logits: torch.Tensor, batch, example: SpatialExample) -> dict:
    """Diagnostic-only teacher-forced token accuracy (section 13: free generation is primary)."""

    labels = batch.labels[0]
    logits = lm_logits[0].float()
    result: dict = {}
    for name in ("reasoning", "box", "loc", "seg", "eos"):
        positions = example.label_positions.get(name) or []
        if not positions:
            result[f"{name}_token_accuracy"] = None
            result[f"{name}_positions"] = 0
            continue
        predicted = logits[positions, :].argmax(dim=-1)
        target = labels[positions]
        result[f"{name}_token_accuracy"] = float((predicted == target).float().mean())
        result[f"{name}_positions"] = len(positions)
    return result
