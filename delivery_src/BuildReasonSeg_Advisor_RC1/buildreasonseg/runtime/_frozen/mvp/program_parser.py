"""Task 6J Part E: ProgramHead — instruction -> canonical program (text-only).

This branch answers exactly one question: *what spatial program does the instruction request?*
It never localizes pixels and never sees GT geometry. The backbone is Qwen3-VL-2B but the input
is the instruction TEXT ONLY (normal tokenizer/chat formatting, no image tokens), which
deliberately avoids the image-dominated `[BOX]` problem measured in Tasks 6F-6I:

    instruction text
        -> Qwen3-VL (text-only, text-only LoRA)
        -> hidden state of the last prompt position (the assistant-prefix representation)
        -> LayerNorm
        -> Linear(hidden_dim, num_programs)      # 20 canonical programs, frozen vocabulary
        -> program id

Trainable: text-only LoRA + the ProgramHead. Frozen: Qwen base, the visual tower, and every
prior grounding head (none are part of this model). The canonical program vocabulary is the
frozen BuildSpatialReason v0.1.1 query-type set (1:1; `evaluation/task6j_program_spec.json`).
No free-form program generation is used in Task 6J.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from .qwen_seg import attach_lora, load_qwen, setup_seg_token

#: Program ids come from the frozen query-type vocabulary (spec builder enforces 1:1).
EXPECTED_PROGRAM_IDS = (
    "leftmost",
    "rightmost",
    "topmost",
    "bottommost",
    "largest",
    "smallest",
    "largest_to_nearest",
    "smallest_to_nearest",
    "largest_to_above",
    "largest_to_below",
    "largest_to_left_of",
    "largest_to_right_of",
    "smallest_to_above",
    "smallest_to_below",
    "smallest_to_left_of",
    "smallest_to_right_of",
    "largest_to_above_to_nearest",
    "largest_to_below_to_nearest",
    "largest_to_left_of_to_nearest",
    "largest_to_right_of_to_nearest",
)

PROGRAM_ID_TO_INDEX = {program_id: index for index, program_id in enumerate(EXPECTED_PROGRAM_IDS)}
NUM_PROGRAMS = len(EXPECTED_PROGRAM_IDS)


class ProgramHead(nn.Module):
    """Section 11: minimal classifier over a documented text representation.

    `LayerNorm -> Linear(hidden_dim, num_programs)` on the last prompt position's hidden state
    (the assistant-prefix representation). Nothing else.
    """

    def __init__(self, hidden_dim: int, num_programs: int = NUM_PROGRAMS) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.classifier = nn.Linear(int(hidden_dim), int(num_programs))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.norm(hidden.float()))

    def as_dict(self) -> dict:
        return {
            "class": "ProgramHead",
            "hidden_dim": self.norm.normalized_shape[0],
            "num_programs": int(self.classifier.out_features),
            "input": "last prompt position hidden (assistant-prefix representation), text-only",
            "structure": "LayerNorm -> Linear(hidden, num_programs)",
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
        }


@dataclass
class ProgramBatch:
    """One text-only instruction batch (no image tokens, no pixel_values)."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor  # program index (CE target)

    def to(self, device):
        return ProgramBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device),
        )


@dataclass
class ProgramParserRuntime:
    """The J2 parser stack: Qwen (text-only LoRA) + ProgramHead + tokenizer/processor."""

    cfg: dict
    processor: object
    tokenizer: object
    qwen: nn.Module
    head: ProgramHead
    device: str
    reports: dict = field(default_factory=dict)

    def build_batch(self, instructions: list[str], program_ids: list[str],
                    max_length: int = 256) -> ProgramBatch:
        """Chat-template the instruction texts only (no image) and index the program labels."""

        input_ids = []
        attention_masks = []
        for instruction in instructions:
            messages = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            ids = prompt["input_ids"][0]
            input_ids.append(ids[:max_length])
            attention_masks.append(torch.ones_like(input_ids[-1]))
        padded_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id))
        padded_mask = nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
        labels = torch.tensor(
            [PROGRAM_ID_TO_INDEX[str(program_id)] for program_id in program_ids], dtype=torch.long
        )
        return ProgramBatch(input_ids=padded_ids, attention_mask=padded_mask, labels=labels)

    def forward(self, batch: ProgramBatch):
        """`batch -> (program_logits [B,20], assistant_prefix_hidden [B,2048])`."""

        outputs = self.qwen(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        # Last REAL prompt position per sample (padding-aware) = the assistant-prefix
        # representation (documented). Padding positions are excluded by the attention mask.
        last_positions = batch.attention_mask.sum(dim=1) - 1
        index = torch.arange(batch.input_ids.shape[0], device=batch.input_ids.device)
        last_hidden = outputs.hidden_states[-1][index, last_positions, :]
        logits = self.head(last_hidden)
        return logits, last_hidden

    @torch.no_grad()
    def predict(self, instruction: str) -> str:
        """One instruction -> predicted canonical program id (inference only)."""

        self.qwen.eval()
        batch = self.build_batch([instruction], [EXPECTED_PROGRAM_IDS[0]]).to(self.device)
        logits, _hidden = self.forward(batch)
        index = int(torch.argmax(logits[0]))
        return EXPECTED_PROGRAM_IDS[index]

    def train_step(self, batch: ProgramBatch, optimizer, grad_clip_norm: float) -> dict:
        use_autocast = bool(self.cfg["training"].get("bf16_autocast", True))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
            logits, _hidden = self.forward(batch)
            loss = nn.functional.cross_entropy(logits, batch.labels)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clipped = None
        if optimizer is not None:
            clipped = float(
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.parameters() if p.requires_grad], float(grad_clip_norm)
                )
            )
            optimizer.step()
        return {
            "loss": float(loss.detach()),
            "grad_clip_total_norm": clipped,
        }

    def parameters(self, recurse: bool = True):
        return [*self.qwen.parameters(recurse=recurse), *self.head.parameters(recurse=recurse)]

    def trainable_parameter_groups(self, lora_lr: float, head_lr: float, weight_decay: float) -> list:
        adapters, head_params = [], []
        for name, parameter in self.qwen.named_parameters():
            if not parameter.requires_grad:
                continue
            adapters.append(parameter)
        for parameter in self.head.parameters():
            parameter.requires_grad_(True)
            head_params.append(parameter)
        groups = []
        if adapters:
            groups.append({"params": adapters, "lr": lora_lr, "weight_decay": weight_decay, "name": "lora"})
        if head_params:
            groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay, "name": "program_head"})
        return groups


def build_program_parser(cfg: dict, device: str = "cuda", verbose: bool = True) -> ProgramParserRuntime:
    """Assemble the J2 parser stack (no SAM2, no visual grounding heads, no image input)."""

    from pathlib import Path as _Path

    repo_root = _Path(__file__).resolve().parents[1]
    cache_root = repo_root / cfg["paths"]["hf_cache"]
    if verbose:
        print("[j2] loading Qwen (text-only branch) ...", flush=True)
    processor, qwen = load_qwen(
        cfg["models"]["qwen_model_id"],
        cache_dir=str(cache_root / "hub"),
        dtype=torch.bfloat16,
        device=device,
        attn_implementation=cfg["models"]["qwen_attn_implementation"],
    )
    tokenizer = processor.tokenizer
    token_setup = setup_seg_token(qwen, tokenizer)  # stack convention; [SEG] is never used by J2
    qwen, token_holder, lora_report = attach_lora(
        qwen,
        token_setup.seg_token_id,
        rank=int(cfg["lora"]["rank"]),
        alpha=int(cfg["lora"]["alpha"]),
        dropout=float(cfg["lora"]["dropout"]),
        extra_token_ids=[],
    )
    hidden_size = (
        int(qwen.config.text_config.hidden_size)
        if hasattr(qwen.config, "text_config")
        else int(qwen.config.hidden_size)
    )
    head = ProgramHead(hidden_dim=hidden_size, num_programs=NUM_PROGRAMS).to(device)
    reports = {
        "token": token_setup.as_dict(),
        "lora": lora_report.as_dict(),
        "head": head.as_dict(),
        "num_programs": NUM_PROGRAMS,
        "program_ids": list(EXPECTED_PROGRAM_IDS),
        "input": "instruction text only; no image tokens; last prompt position hidden",
    }
    return ProgramParserRuntime(
        cfg=cfg, processor=processor, tokenizer=tokenizer, qwen=qwen, head=head,
        device=device, reports=reports,
    )


# ------------------------------------------------------------------ checkpointing


def save_parser_checkpoint(path, runtime: ProgramParserRuntime, step: int,
                           metrics: dict | None = None, optimizer=None) -> dict:
    """LoRA + token rows + ProgramHead + optimizer + RNG state. Never base weights."""

    import hashlib
    import random

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lora_and_token_state = {}
    for key, value in runtime.qwen.state_dict().items():
        if "lora_" in key or "trainable_tokens" in key or "token_row" in key:
            lora_and_token_state[key] = value.detach().cpu()
    payload = {
        "format": "buildreasonseg-program-parser-v1",
        "step": int(step),
        "metrics": metrics or {},
        "lora_and_token_state": lora_and_token_state,
        "program_head": {k: v.detach().cpu() for k, v in runtime.head.state_dict().items()},
        "rng_state": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "step": int(step),
    }


def load_parser_checkpoint(path, runtime: ProgramParserRuntime, optimizer=None) -> dict:
    """Load a `save_parser_checkpoint` payload back into a parser runtime."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    missing, unexpected = runtime.qwen.load_state_dict(payload["lora_and_token_state"], strict=False)
    runtime.head.load_state_dict(payload["program_head"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return {
        "step": payload.get("step", 0),
        "metrics": payload.get("metrics", {}),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "program_head_loaded": True,
    }
