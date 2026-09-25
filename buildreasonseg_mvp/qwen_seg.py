"""Qwen3-VL side of the MVP: tokenizer, `[SEG]` token, LoRA scope, hidden states.

Task 6A uses exactly one new architecture token: `[SEG]`. There is no `[REF]`.

Design notes that matter for correctness:

* **Exactly one `[SEG]` token.** It is added as an additional special token and
  asserted to round-trip as a single id.
* **Embeddings are resized exactly once**, and the tied-weight state is verified
  afterwards (Qwen3-VL-2B has `tie_word_embeddings = True`).
* **Only the `[SEG]` row is trainable.** Two mechanisms are supported, chosen at
  runtime: PEFT's `trainable_token_indices=[seg_id]` when the installed PEFT
  exposes it, otherwise a project-owned token-row adapter. In neither case is the
  full vocabulary embedding matrix handed to the optimizer.
* **LoRA attaches to language-model projections only.** The visual tower is
  excluded by full module name, and the count of LoRA modules under the visual
  tower must be zero.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from .data import SEG_TOKEN

#: Projection suffixes LoRA is allowed to touch, text side only.
LORA_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

#: Substrings that mark a module as belonging to the visual tower.
VISUAL_MARKERS = ("visual", "vision_tower", "vision_model")


# --------------------------------------------------------------------------
# Tokenizer / processor
# --------------------------------------------------------------------------


@dataclass
class SegTokenSetup:
    seg_token_id: int
    vocab_before: int
    vocab_after: int
    single_token_roundtrip: bool
    tied_embeddings: bool
    tied_after_resize: bool
    embedding_shape_before: tuple[int, ...]
    embedding_shape_after: tuple[int, ...]

    def as_dict(self) -> dict:
        return {
            "seg_token_id": self.seg_token_id,
            "vocab_before": self.vocab_before,
            "vocab_after": self.vocab_after,
            "single_token_roundtrip": self.single_token_roundtrip,
            "tied_embeddings": self.tied_embeddings,
            "tied_after_resize": self.tied_after_resize,
            "embedding_shape_before": list(self.embedding_shape_before),
            "embedding_shape_after": list(self.embedding_shape_after),
        }


def _is_tied(model: nn.Module) -> bool:
    input_embedding = model.get_input_embeddings()
    output_embedding = model.get_output_embeddings()
    if input_embedding is None or output_embedding is None:
        return False
    return input_embedding.weight is getattr(output_embedding, "weight", None)


def setup_seg_token(model, tokenizer) -> SegTokenSetup:
    """Add `[SEG]` as one special token and resize the embedding table once."""

    vocab_before = len(tokenizer)
    embedding_before = tuple(model.get_input_embeddings().weight.shape)
    tied_before = _is_tied(model)

    added = tokenizer.add_special_tokens({"additional_special_tokens": [SEG_TOKEN]})
    seg_ids = tokenizer.encode(SEG_TOKEN, add_special_tokens=False)
    if len(seg_ids) != 1:
        raise RuntimeError(f"{SEG_TOKEN} did not become a single token: {seg_ids}")
    seg_token_id = int(seg_ids[0])

    model.resize_token_embeddings(len(tokenizer))

    embedding_after = tuple(model.get_input_embeddings().weight.shape)
    tied_after = _is_tied(model)

    # round-trip: decode(encode(text)) must reproduce the literal marker
    decoded = tokenizer.decode([seg_token_id], skip_special_tokens=False)
    roundtrip = SEG_TOKEN in decoded

    return SegTokenSetup(
        seg_token_id=seg_token_id,
        vocab_before=vocab_before,
        vocab_after=len(tokenizer),
        single_token_roundtrip=bool(roundtrip),
        tied_embeddings=bool(tied_before),
        tied_after_resize=bool(tied_after),
        embedding_shape_before=embedding_before,
        embedding_shape_after=embedding_after,
    )


# --------------------------------------------------------------------------
# Trainable token row
# --------------------------------------------------------------------------


class TrainableTokenRow(nn.Module):
    """A single trainable embedding row, injected without unfreezing the table.

    `nn.Parameter` holder only; the injection itself is a forward hook so the
    base embedding module keeps its identity and every call site is covered.
    """

    def __init__(self, token_id: int, initial_row: torch.Tensor) -> None:
        super().__init__()
        self.token_id = int(token_id)
        self.row = nn.Parameter(initial_row.detach().clone())

    def as_dict(self) -> dict:
        return {"token_id": self.token_id, "shape": list(self.row.shape), "numel": int(self.row.numel())}


def install_trainable_token_row(model, token_id: int) -> TrainableTokenRow:
    """Attach a forward hook that substitutes one embedding row.

    With tied input/output embeddings the *output* classifier row still comes
    from the frozen base table. That is acceptable for Task 6A because the
    signal that trains this row is the mask loss flowing through the hidden
    state, and it is recorded as a known limitation rather than hidden.
    """

    embedding = model.get_input_embeddings()
    holder = TrainableTokenRow(token_id, embedding.weight.data[token_id])

    def hook(module, args, output):  # noqa: ANN001
        if not torch.is_tensor(output):
            return output
        input_ids = args[0] if args else None
        if not torch.is_tensor(input_ids):
            return output
        mask = input_ids == holder.token_id
        if not bool(mask.any()):
            return output
        row = holder.row.to(dtype=output.dtype, device=output.device)
        return torch.where(mask.unsqueeze(-1), row, output)

    holder._handle = embedding.register_forward_hook(hook)  # type: ignore[attr-defined]
    return holder


def uninstall_trainable_token_row(holder: TrainableTokenRow) -> None:
    handle = getattr(holder, "_handle", None)
    if handle is not None:
        handle.remove()


# --------------------------------------------------------------------------
# LoRA scope
# --------------------------------------------------------------------------


def text_lora_target_names(model: nn.Module, suffixes: Sequence[str] = LORA_SUFFIXES) -> list[str]:
    """Full module names of language-model projections eligible for LoRA."""

    names: list[str] = []
    for name, module in model.named_modules():
        if not name or not isinstance(module, nn.Linear):
            continue
        if any(marker in name for marker in VISUAL_MARKERS):
            continue
        if name.endswith(tuple(suffixes)):
            names.append(name)
    return sorted(names)


def lora_module_names(model: nn.Module) -> list[str]:
    """Names of modules already carrying a LoRA adapter (peft naming)."""

    return sorted(
        name for name, module in model.named_modules() if isinstance(module, nn.Module) and "lora_" in name
    )


@dataclass
class LoraAttachReport:
    target_modules: list[str] = field(default_factory=list)
    visual_lora_modules: list[str] = field(default_factory=list)
    lora_module_count: int = 0
    total_params: int = 0
    trainable_params: int = 0
    lora_params: int = 0
    token_params: int = 0
    token_mechanism: str = ""
    peft_trainable_token_indices_supported: bool = False
    peft_wraps_output_head: bool = False
    output_row_installed: bool = False
    output_row_params: int = 0

    def as_dict(self) -> dict:
        return {
            "target_modules": self.target_modules,
            "n_target_modules": len(self.target_modules),
            "visual_lora_modules": self.visual_lora_modules,
            "n_visual_lora_modules": len(self.visual_lora_modules),
            "lora_module_count": self.lora_module_count,
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "lora_params": self.lora_params,
            "token_params": self.token_params,
            "token_mechanism": self.token_mechanism,
            "peft_trainable_token_indices_supported": self.peft_trainable_token_indices_supported,
            "peft_wraps_output_head": self.peft_wraps_output_head,
            "output_row_installed": self.output_row_installed,
            "output_row_params": self.output_row_params,
        }


def attach_lora(
    model,
    seg_token_id: int,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    prefer_peft_token_indices: bool = True,
):
    """Attach LoRA to text projections and make only the `[SEG]` row trainable."""

    from peft import LoraConfig, get_peft_model

    targets = text_lora_target_names(model)
    if not targets:
        raise RuntimeError("no language-model projection modules found for LoRA")

    supports_token_indices = "trainable_token_indices" in inspect.signature(LoraConfig.__init__).parameters

    kwargs: dict = {}
    token_mechanism = "project_forward_hook"
    if prefer_peft_token_indices and supports_token_indices:
        kwargs["trainable_token_indices"] = [int(seg_token_id)]
        token_mechanism = "peft_trainable_token_indices"

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=targets,
        task_type=None,
        **kwargs,
    )
    peft_model = get_peft_model(model, config)

    token_holder = None
    if token_mechanism == "project_forward_hook":
        token_holder = install_trainable_token_row(peft_model, seg_token_id)

    # With peft >= 0.21 the token adapter also wraps the tied OUTPUT head, so the
    # [SEG] row is trainable on both sides. Its delta is created lazily on the
    # first forward, which is why the parameter count grows after one step.
    # `SegOutputRowLinear` below is retained only as a fallback for peft builds
    # that do not wrap the output head; it is installed only when needed.
    output_head = peft_model.get_output_embeddings()
    peft_wraps_output_head = type(output_head).__name__ == "TrainableTokensWrapper"
    output_wrapper = None
    if not peft_wraps_output_head:
        output_wrapper = install_trainable_output_row(peft_model, seg_token_id)

    visual_lora = [
        name
        for name in lora_module_names(peft_model)
        if any(marker in name for marker in VISUAL_MARKERS)
    ]
    if visual_lora:
        raise RuntimeError(f"LoRA leaked into the visual tower: {visual_lora[:5]}")

    report = LoraAttachReport(
        target_modules=targets,
        visual_lora_modules=visual_lora,
        lora_module_count=len(lora_module_names(peft_model)),
        token_mechanism=token_mechanism,
        peft_trainable_token_indices_supported=bool(supports_token_indices),
        peft_wraps_output_head=bool(peft_wraps_output_head),
        output_row_installed=output_wrapper is not None,
        output_row_params=int(output_wrapper.delta.numel()) if output_wrapper is not None else 0,
    )
    return peft_model, token_holder, report


#: Parameter-name fragments that identify the trainable `[SEG]` representation.
#: PEFT's `trainable_token_indices` names it `...trainable_tokens_delta.<id>`;
#: the project fallback names it `token_row` / `trainable_tokens`.
TOKEN_PARAM_MARKERS = ("trainable_tokens", "token_row")


def is_token_parameter(name: str) -> bool:
    return any(marker in name for marker in TOKEN_PARAM_MARKERS)


class SegOutputRowLinear(nn.Module):
    """`lm_head` with exactly one trainable output row for `[SEG]`.

    Why this exists: PEFT's `trainable_token_indices` wraps the **input** embedding,
    so the `[SEG]` input row becomes trainable while the tied output classifier row
    stays frozen. A model cannot then learn to *emit* `[SEG]` during free
    generation, which Task 6A section 16.5 measures. This wrapper keeps the base
    output matrix frozen and adds a trainable residual to the single `[SEG]`
    column, so the vocabulary matrix is never handed to the optimizer.
    """

    def __init__(self, base: nn.Linear, token_id: int) -> None:
        super().__init__()
        self.base = base
        self.token_id = int(token_id)
        self.delta = nn.Parameter(torch.zeros(base.weight.shape[1], dtype=base.weight.dtype))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def out_features(self) -> int:
        return int(self.base.weight.shape[0])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.base(hidden)
        extra = torch.nn.functional.linear(
            hidden.to(self.delta.dtype), self.delta.view(1, -1)
        ).squeeze(-1)
        index = torch.tensor([self.token_id], device=logits.device)
        return logits.index_add(-1, index, extra.to(logits.dtype))


def install_trainable_output_row(model, token_id: int) -> SegOutputRowLinear | None:
    """Wrap the output head so the `[SEG]` column has one trainable residual row."""

    output = model.get_output_embeddings()
    if output is None or not isinstance(output, nn.Linear):
        return None

    parent = None
    attribute = None
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if child is output:
                parent, attribute = module, child_name
                break
        if parent is not None:
            break
    if parent is None or attribute is None:
        return None

    wrapper = SegOutputRowLinear(output, token_id)
    setattr(parent, attribute, wrapper)
    return wrapper


def output_row_param_ids(model) -> set[int]:
    """Parameter ids of every `SegOutputRowLinear.delta` under `model`."""

    ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, SegOutputRowLinear):
            ids.add(id(module.delta))
    return ids


def _embedding_of(model):
    """Input embedding module, whether given the language model or a wrapper."""

    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings()
    language_model = getattr(model, "qwen", None)
    if language_model is not None and hasattr(language_model, "get_input_embeddings"):
        return language_model.get_input_embeddings()
    return None


def parameter_report(model: nn.Module, seg_token_id: int, token_holder: TrainableTokenRow | None) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    output_row_ids = output_row_param_ids(model)
    lora = 0
    token = 0
    output_row = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in output_row_ids:
            output_row += parameter.numel()
        elif is_token_parameter(name):
            token += parameter.numel()
        elif "lora_" in name:
            lora += parameter.numel()
    if token_holder is not None:
        token += int(token_holder.row.numel())

    embedding = _embedding_of(model)
    base_layer = getattr(getattr(embedding, "token_adapter", None), "base_layer", None)
    base_weight_frozen = None
    base_weight_numel = 0
    if base_layer is not None and hasattr(base_layer, "weight"):
        base_weight_frozen = not bool(base_layer.weight.requires_grad)
        base_weight_numel = int(base_layer.weight.numel())
    elif embedding is not None and hasattr(embedding, "weight"):
        base_weight_frozen = not bool(embedding.weight.requires_grad)
        base_weight_numel = int(embedding.weight.numel())

    return {
        "total_params": total,
        "trainable_params": trainable,
        "lora_params": lora,
        "token_params": token,
        "output_row_params": output_row,
        "trainable_fraction": trainable / max(total, 1),
        "seg_token_id": seg_token_id,
        "embedding_base_frozen": base_weight_frozen,
        "embedding_base_numel": base_weight_numel,
    }


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------


def load_qwen(
    model_id: str,
    cache_dir: str | os.PathLike | None = None,
    dtype: torch.dtype | None = None,
    device: str = "cuda",
    attn_implementation: str = "sdpa",
):
    """Load the Qwen3-VL processor and model. No download policy is applied here."""

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = dtype or torch.bfloat16
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    model.to(device)
    return processor, model


# --------------------------------------------------------------------------
# Batch construction and hidden-state extraction
# --------------------------------------------------------------------------


@dataclass
class TeacherForcedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    prompt_length: int
    total_length: int
    seg_position: int
    visual_tokens: int
    extra_inputs: dict = field(default_factory=dict)

    def to(self, device: str) -> "TeacherForcedBatch":
        moved = {
            "input_ids": self.input_ids.to(device),
            "attention_mask": self.attention_mask.to(device),
            "labels": self.labels.to(device),
        }
        if self.pixel_values is not None:
            moved["pixel_values"] = self.pixel_values.to(device)
        if self.image_grid_thw is not None:
            moved["image_grid_thw"] = self.image_grid_thw.to(device)
        moved["extra_inputs"] = {
            key: (value.to(device) if torch.is_tensor(value) else value)
            for key, value in self.extra_inputs.items()
        }
        return TeacherForcedBatch(
            prompt_length=self.prompt_length,
            total_length=self.total_length,
            seg_position=self.seg_position,
            visual_tokens=self.visual_tokens,
            **moved,
        )


def build_teacher_forcing_batch(
    processor,
    tokenizer,
    image,
    instruction: str,
    assistant_text: str,
    seg_token_id: int,
    append_eos: bool = True,
) -> TeacherForcedBatch:
    """Chat input from image + instruction, then the assistant target appended.

    LM labels are `-100` everywhere except the assistant target span, so the
    user/image/prompt tokens contribute nothing to the cross-entropy.

    `append_eos` appends the tokenizer's end-of-sequence token after `[SEG]`.
    Without it the model is never taught to STOP, so free generation runs to the
    token cap and repeats `[SEG]` -- which is exactly what the first recorded
    Stage 2 run showed (`seg_count` between 4 and 13).
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

    target_ids = tokenizer(assistant_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if append_eos and tokenizer.eos_token_id is not None:
        eos = torch.tensor([[int(tokenizer.eos_token_id)]], dtype=target_ids.dtype)
        target_ids = torch.cat([target_ids, eos], dim=1)

    input_ids = torch.cat([prompt_ids, target_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)

    total_length = int(input_ids.shape[1])
    labels = torch.full_like(input_ids, -100)
    # logits at position i predict token i+1
    labels[0, prompt_length - 1 : total_length - 1] = input_ids[0, prompt_length:total_length]

    seg_positions = (input_ids[0] == seg_token_id).nonzero(as_tuple=False).flatten()
    if seg_positions.numel() != 1:
        raise RuntimeError(f"expected exactly one [SEG] token, found {seg_positions.numel()}")
    seg_position = int(seg_positions[0])

    grid = prompt.get("image_grid_thw")
    visual_tokens = 0
    if grid is not None:
        merge = getattr(processor.image_processor, "merge_size", 2) or 2
        visual_tokens = int(grid.prod(dim=-1).sum().item()) // (int(merge) ** 2)

    extra = {
        k: v for k, v in prompt.items() if k not in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
    }

    # The Qwen3-VL processor returns per-token multi-modal side inputs (notably
    # `mm_token_type_ids`, consumed by `get_rope_index`) whose length equals the
    # PROMPT length. Appending the assistant target tokens lengthens `input_ids`,
    # so every such tensor must be extended to match or the forward pass raises a
    # shape error. Text tokens use type 0, which is the correct neutral value.
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
        extra_inputs=extended_extra,
    )


def forward_qwen(model, batch: TeacherForcedBatch):
    """One forward pass returning both LM logits and the `[SEG]` hidden state.

    Doing this in a single pass is what makes the 2-sample smoke test cheap: the
    language loss and the mask pathway share the same forward.
    """

    outputs = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        pixel_values=batch.pixel_values,
        image_grid_thw=batch.image_grid_thw,
        output_hidden_states=True,
        use_cache=False,
        **batch.extra_inputs,
    )
    hidden = outputs.hidden_states[-1][:, batch.seg_position, :]
    return outputs.logits, hidden


def extract_seg_hidden(model, batch: TeacherForcedBatch) -> torch.Tensor:
    """Final-layer hidden state at the `[SEG]` position. Shape (B, hidden)."""

    _logits, hidden = forward_qwen(model, batch)
    return hidden


@torch.no_grad()
def generate_with_seg(model, processor, tokenizer, image, instruction: str, max_new_tokens: int = 256,
                      seg_token_id: int | None = None):
    """Free generation from image + instruction only, then locate `[SEG]`.

    Returns the generated text, the full token sequence, the prompt length and the
    index/count of `[SEG]`. No ground-truth reasoning or target is used.
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
    device = next(model.parameters()).device
    model_inputs = {
        key: (value.to(device) if torch.is_tensor(value) else value) for key, value in inputs.items()
    }
    generated = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_length = int(model_inputs["input_ids"].shape[1])
    new_tokens = generated[0, prompt_length:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    if seg_token_id is None:
        ids = tokenizer.encode(SEG_TOKEN, add_special_tokens=False)
        seg_token_id = int(ids[0])
    positions = (generated[0] == seg_token_id).nonzero(as_tuple=False).flatten()
    return {
        "text": text,
        "token_ids": generated[0].tolist(),
        "prompt_length": prompt_length,
        "seg_positions": positions.tolist(),
        "seg_count": int(positions.numel()),
        "generated_token_count": int(new_tokens.numel()),
    }


def seg_hidden_from_full_sequence(model, processor, image, full_token_ids: list[int],
                                  seg_position: int, instruction: str = "") -> torch.Tensor:
    """Re-forward a complete generated sequence and read the `[SEG]` hidden state.

    The image is supplied again because the sequence still contains its vision
    tokens; nothing is taken from the ground truth.

    Qwen3-VL requires `mm_token_type_ids` whenever `image_grid_thw` is passed: it
    is what lets `get_rope_index` apply multimodal RoPE. The processor returns it
    for a chat prompt, but the generated sequence is longer, so the prompt's
    per-token types are reused for the prefix (the image span is identical, since
    the image always comes first) and the generated tail is marked as text (0).
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
    prompt_inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )

    device = next(model.parameters()).device
    input_ids = torch.tensor([full_token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    pixel_values = prompt_inputs["pixel_values"].to(device)
    image_grid_thw = prompt_inputs["image_grid_thw"].to(device)

    mm_token_type_ids = prompt_inputs.get("mm_token_type_ids")
    if mm_token_type_ids is not None:
        prompt_types = mm_token_type_ids.to(device)
        total = input_ids.shape[1]
        prompt_length = int(prompt_types.shape[1])
        if prompt_length < total:
            pad = torch.zeros(
                (1, total - prompt_length), dtype=prompt_types.dtype, device=device
            )
            prompt_types = torch.cat([prompt_types, pad], dim=1)
        else:
            prompt_types = prompt_types[:, :total]
    else:
        prompt_types = None

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        mm_token_type_ids=prompt_types,
        output_hidden_states=True,
        use_cache=False,
    )
    return outputs.hidden_states[-1][:, seg_position, :]

