"""Task 6E: explicit spatial tokens — quantized box codec, token vocabulary, sequence targets.

The Task 6D.1 audit established that the frozen `[SEG]` representation contains no
practically decodable target geometry. Task 6E therefore stops asking a hidden state to
*contain* the box and makes the box an explicit autoregressive target:

    image + instruction -> reasoning -> [BOX] <loc_x1> <loc_y1> <loc_x2> <loc_y2> -> [SEG]

One shared location vocabulary `<loc_000>..<loc_{B-1}>` is used for all four coordinates;
the *position* of a token decides whether it is x1, y1, x2 or y2 (section 3).

Quantization (section 4) is a deterministic **enclosing** box:

* min edges: ``q1 = floor(c1 * (B - 1))``
* max edges: ``q2 = ceil (c2 * (B - 1))``
* clipped to ``[0, B-1]``, dequantized with ``c_hat = q / (B - 1)``
* a non-empty target keeps non-zero width/height: if the quantized span collapses, the
  upper edge is expanded by one bin within bounds (deterministically), or the lower edge
  is pulled back when the span is already at the top.

Nothing here touches GT at inference: `decode` takes token ids, and the only consumer of
`encode` is supervision/the oracle diagnostic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch

#: Section 3 candidates.
BIN_CANDIDATES = (32, 64, 128, 256)
BOX_TOKEN = "[BOX]"
LOC_TEMPLATE = "<loc_{index:0{width}d}>"
LOC_PATTERN = re.compile(r"^<loc_(\d+)>$")


def loc_width(bins: int) -> int:
    return max(3, len(str(int(bins) - 1)))


def loc_name(index: int, bins: int) -> str:
    return LOC_TEMPLATE.format(index=int(index), width=loc_width(bins))


def spatial_token_names(bins: int) -> list[str]:
    """`[BOX]` followed by the shared location vocabulary, in id order."""

    return [BOX_TOKEN] + [loc_name(index, bins) for index in range(int(bins))]


@dataclass
class QuantizedBoxCodec:
    """Deterministic enclosing-box quantization for one bin count."""

    bins: int

    def __post_init__(self) -> None:
        if self.bins < 2:
            raise ValueError(f"bins must be >= 2, got {self.bins}")

    @property
    def scale(self) -> float:
        return float(self.bins - 1)

    def quantize_values(self, box) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = [float(value) for value in box]
        low = [x1, y1]
        high = [x2, y2]
        quantized = [
            int(min(max(0, int(torch.floor(torch.tensor(c * self.scale)).item())), self.bins - 1))
            for c in low
        ] + [
            int(min(max(0, int(torch.ceil(torch.tensor(c * self.scale)).item())), self.bins - 1))
            for c in high
        ]
        qx1, qy1, qx2, qy2 = quantized
        # Enclosing quantization must not produce a degenerate box for a non-empty target.
        if qx2 <= qx1:
            if qx1 + 1 <= self.bins - 1:
                qx2 = qx1 + 1
            else:
                qx1 = max(0, qx2 - 1)
        if qy2 <= qy1:
            if qy1 + 1 <= self.bins - 1:
                qy2 = qy1 + 1
            else:
                qy1 = max(0, qy2 - 1)
        return qx1, qy1, qx2, qy2

    def dequantize_values(self, codes) -> tuple[float, float, float, float]:
        qx1, qy1, qx2, qy2 = [int(value) for value in codes]
        return (
            qx1 / self.scale,
            qy1 / self.scale,
            qx2 / self.scale,
            qy2 / self.scale,
        )

    def encode(self, box) -> list[int]:
        return list(self.quantize_values(box))

    def decode(self, codes) -> list[float]:
        return list(self.dequantize_values(codes))

    def is_valid(self, codes) -> bool:
        if codes is None or len(codes) != 4:
            return False
        if any(not isinstance(value, int) for value in codes):
            return False
        if any(value < 0 or value >= self.bins for value in codes):
            return False
        qx1, qy1, qx2, qy2 = codes
        return qx1 < qx2 and qy1 < qy2

    def as_dict(self) -> dict:
        return {
            "bins": self.bins,
            "scale": self.scale,
            "min_edge_rule": "floor(c * (B-1)), clipped to [0, B-1]",
            "max_edge_rule": "ceil(c * (B-1)), clipped to [0, B-1]",
            "dequantize_rule": "q / (B-1)",
            "degenerate_guard": "upper edge expanded by one bin, or lower edge pulled back at the top",
            "max_quantization_error": 1.0 / self.scale,
        }


def box_iou_values(left, right) -> float:
    x1, y1 = max(float(left[0]), float(right[0])), max(float(left[1]), float(right[1]))
    x2, y2 = min(float(left[2]), float(right[2])), min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    area_right = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    union = area_left + area_right - intersection
    return intersection / union if union > 0 else 0.0


# ------------------------------------------------------------------ token vocabulary


@dataclass
class SpatialTokenSetup:
    """What adding `[BOX]` + `<loc_*>` did to the tokenizer and the embedding table."""

    bins: int
    box_token_id: int
    loc_token_ids: list[int]
    vocab_before: int
    vocab_after: int
    added_tokens: int
    embedding_shape_before: tuple
    embedding_shape_after: tuple
    tied_embeddings: bool
    single_token_roundtrip: bool
    label_names: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "bins": self.bins,
            "box_token_id": self.box_token_id,
            "loc_token_ids_first": self.loc_token_ids[:4],
            "loc_token_ids_last": self.loc_token_ids[-2:],
            "loc_token_count": len(self.loc_token_ids),
            "vocab_before": self.vocab_before,
            "vocab_after": self.vocab_after,
            "added_tokens": self.added_tokens,
            "embedding_shape_before": list(self.embedding_shape_before),
            "embedding_shape_after": list(self.embedding_shape_after),
            "tied_embeddings": self.tied_embeddings,
            "single_token_roundtrip": self.single_token_roundtrip,
        }


def add_spatial_tokens(model, tokenizer, bins: int) -> SpatialTokenSetup:
    """Add `[BOX]` and the shared location vocabulary, resizing the table once."""

    names = spatial_token_names(bins)
    vocab_before = len(tokenizer)
    embedding_before = tuple(model.get_input_embeddings().weight.shape)

    tokenizer.add_special_tokens({"additional_special_tokens": names})
    token_ids = tokenizer.convert_tokens_to_ids(names)
    if any(not isinstance(value, int) or value < 0 for value in token_ids):
        raise RuntimeError(f"added spatial tokens did not resolve to ids: {token_ids[:4]}")
    if len(set(token_ids)) != len(names):
        raise RuntimeError("added spatial tokens collided on ids")

    model.resize_token_embeddings(len(tokenizer))

    # every added token must be a single id and round-trip through decode
    decoded = [tokenizer.decode([int(token_id)], skip_special_tokens=False) for token_id in token_ids]
    roundtrip = all(name in text for name, text in zip(names, decoded))

    ids = [int(value) for value in token_ids]
    return SpatialTokenSetup(
        bins=int(bins),
        box_token_id=ids[0],
        loc_token_ids=ids[1:],
        vocab_before=vocab_before,
        vocab_after=len(tokenizer),
        added_tokens=len(names),
        embedding_shape_before=embedding_before,
        embedding_shape_after=tuple(model.get_input_embeddings().weight.shape),
        tied_embeddings=bool(_is_tied(model)),
        single_token_roundtrip=bool(roundtrip),
        label_names=names,
    )


def _is_tied(model) -> bool:
    try:
        return bool(model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr())
    except Exception:  # noqa: BLE001
        return bool(getattr(model.config, "tie_word_embeddings", False))


# ------------------------------------------------------------------ sequence target


@dataclass
class SpatialTargetLayout:
    """Absolute positions of every assistant span inside the full sequence."""

    prompt_length: int
    reasoning_positions: list[int]
    box_position: int
    loc_positions: list[int]
    seg_position: int
    eos_position: int
    total_length: int

    def as_dict(self) -> dict:
        return {
            "prompt_length": self.prompt_length,
            "reasoning_positions": self.reasoning_positions,
            "box_position": self.box_position,
            "loc_positions": self.loc_positions,
            "seg_position": self.seg_position,
            "eos_position": self.eos_position,
            "total_length": self.total_length,
        }


def build_target_ids(
    tokenizer,
    reasoning: str,
    codes: list[int],
    setup: SpatialTokenSetup,
    seg_token_id: int,
    *,
    append_eos: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Explicit assistant target ids: reasoning + `[BOX]` + 4 loc + `[SEG]` (+ EOS)."""

    reasoning_ids = tokenizer(reasoning, add_special_tokens=False, return_tensors="pt")["input_ids"]
    pieces = [reasoning_ids]
    spans: dict[str, list[int]] = {"reasoning": list(range(int(reasoning_ids.shape[1])))}
    offset = int(reasoning_ids.shape[1])

    spans["box"] = [offset]
    pieces.append(torch.tensor([[setup.box_token_id]], dtype=reasoning_ids.dtype))
    offset += 1

    if len(codes) != 4:
        raise ValueError(f"expected four location codes, got {codes}")
    spans["loc"] = []
    for code in codes:
        if not (0 <= int(code) < setup.bins):
            raise ValueError(f"location code {code} outside [0, {setup.bins - 1}]")
        spans["loc"].append(offset)
        pieces.append(torch.tensor([[setup.loc_token_ids[int(code)]]], dtype=reasoning_ids.dtype))
        offset += 1

    spans["seg"] = [offset]
    pieces.append(torch.tensor([[int(seg_token_id)]], dtype=reasoning_ids.dtype))
    offset += 1

    if append_eos and tokenizer.eos_token_id is not None:
        spans["eos"] = [offset]
        pieces.append(torch.tensor([[int(tokenizer.eos_token_id)]], dtype=reasoning_ids.dtype))
        offset += 1
    else:
        spans["eos"] = []

    target_ids = torch.cat(pieces, dim=1)
    if int(target_ids.shape[1]) != offset:
        raise RuntimeError("target construction and span bookkeeping disagree")
    return target_ids, spans


def location_token_ids(setup: SpatialTokenSetup) -> list[int]:
    """The tokens scored by the weighted location term: `[BOX]` plus the four loc slots."""

    return [int(setup.box_token_id), *[int(value) for value in setup.loc_token_ids]]


# ------------------------------------------------------------------ parsing


@dataclass
class ParsedSpatial:
    """Token-id parse of one generated sequence (section 12)."""

    structural_valid: bool
    box_token_count: int
    loc_token_count: int
    consecutive_loc_run: int
    exact_four_token_sequence: bool
    seg_count: int
    seg_after_box: bool
    codes: list[int] | None
    box: list[float] | None
    failure: str | None = None

    def as_dict(self) -> dict:
        return {
            "structural_valid": self.structural_valid,
            "box_token_count": self.box_token_count,
            "loc_token_count": self.loc_token_count,
            "consecutive_loc_run": self.consecutive_loc_run,
            "exact_four_token_sequence": self.exact_four_token_sequence,
            "seg_count": self.seg_count,
            "seg_after_box": self.seg_after_box,
            "codes": self.codes,
            "box": self.box,
            "failure": self.failure,
        }


def parse_generated_ids(
    token_ids, setup: SpatialTokenSetup, codec: QuantizedBoxCodec, seg_token_id: int
) -> ParsedSpatial:
    """Parse token ids: one `[BOX]`, then exactly four consecutive loc tokens, then `[SEG]`.

    Nothing is repaired from GT or prose; an invalid structure is a geometry failure.
    """

    ids = [int(value) for value in token_ids]
    box_id = int(setup.box_token_id)
    loc_ids = [int(value) for value in setup.loc_token_ids]
    loc_lookup = {token_id: index for index, token_id in enumerate(loc_ids)}
    seg_id = int(seg_token_id)

    box_positions = [index for index, value in enumerate(ids) if value == box_id]
    loc_positions = [index for index, value in enumerate(ids) if value in loc_lookup]
    seg_positions = [index for index, value in enumerate(ids) if value == seg_id]

    parsed = ParsedSpatial(
        structural_valid=False,
        box_token_count=len(box_positions),
        loc_token_count=len(loc_positions),
        consecutive_loc_run=0,
        exact_four_token_sequence=False,
        seg_count=len(seg_positions),
        seg_after_box=False,
        codes=None,
        box=None,
    )

    longest = 0
    current = 0
    for index in range(len(ids)):
        if ids[index] in loc_lookup:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    parsed.consecutive_loc_run = longest

    if len(box_positions) != 1:
        parsed.failure = f"expected exactly one [BOX], found {len(box_positions)}"
        return parsed
    if len(seg_positions) != 1:
        parsed.failure = f"expected exactly one [SEG], found {len(seg_positions)}"
        return parsed

    box_position = box_positions[0]
    seg_position = seg_positions[0]
    parsed.seg_after_box = seg_position > box_position
    if not parsed.seg_after_box:
        parsed.failure = "[SEG] does not follow [BOX]"
        return parsed

    run = [index for index in range(box_position + 1, min(len(ids), box_position + 1 + 4))]
    if len(run) != 4 or any(ids[index] not in loc_lookup for index in run):
        parsed.failure = "the four tokens after [BOX] are not four consecutive location tokens"
        return parsed
    # Section 6/12: the required span is exactly `[BOX] <loc_x1> <loc_y1> <loc_x2> <loc_y2> [SEG]`,
    # so a fifth location token or any filler between the box and [SEG] is malformed output.
    if seg_position != box_position + 5:
        parsed.failure = (
            f"[SEG] is at {seg_position}, expected immediately after the four location tokens "
            f"({box_position + 5})"
        )
        return parsed
    codes = [loc_lookup[ids[index]] for index in run]
    parsed.codes = codes
    parsed.exact_four_token_sequence = True

    if not codec.is_valid(codes):
        parsed.failure = f"non-canonical quantized box {codes}"
        return parsed

    parsed.box = codec.decode(codes)
    parsed.structural_valid = True
    return parsed
