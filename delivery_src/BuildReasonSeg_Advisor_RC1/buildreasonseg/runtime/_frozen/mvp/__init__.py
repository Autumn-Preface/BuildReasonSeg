"""BuildReasonSeg-MVP: the minimal Qwen3-VL -> [SEG] -> SAM2.1 segmentation pipeline.

Task 6A scope (deliberately narrow):

    image + instruction
        -> Qwen3-VL-2B-Instruct
        -> reasoning_zh + " [SEG]"
        -> hidden state of the [SEG] token
        -> projection MLP
        -> one sparse prompt embedding
        -> vanilla SAM2.1 Hiera Base+ mask decoder
        -> target mask

Explicitly **not** in this package: `[REF]`, Spatial Relation Encoder, Spatial
Consistency Loss, multi-dataset training, Qwen3-VL-4B.

Module responsibilities are kept separate so a later Spatial Relation Encoder can
be inserted between the projection and the mask decoder without touching the
language-model path:

``data``            dataset records -> tensors, masks, instruction strings
``qwen_seg``        tokenizer/processor, `[SEG]` token, LoRA scope, hidden states
``sam2_bridge``     frozen SAM2.1 features, projection MLP, sparse prompt decode
``model``           the assembled module and its forward/loss
``losses``          LM cross-entropy, BCE-with-logits, soft Dice
``metrics``         IoU at original resolution
``checkpointing``   adapter-only save/load and the checkpoint manifest
"""

from __future__ import annotations

__version__ = "0.1.0-task6a"

__all__ = [
    "__version__",
    "data",
    "qwen_seg",
    "sam2_bridge",
    "model",
    "losses",
    "metrics",
    "checkpointing",
]
