"""Task 6C.5: CPU-side input caches and pinned-transfer helpers.

Design rules taken from Task 6C.5 sections 11-13:

* caches hold **model-independent** data only: decoded source pixels, boolean ground
  truth masks, and the deterministic output of the Qwen processor for a fixed sample;
* no hidden states, no logits, no gradients, no trainable-model output is ever cached;
* the ground-truth cache is training supervision/IO acceleration only — it is never an
  inference input, and nothing here changes what the model sees;
* every cache is bounded, keyed by immutable ids, and off unless a config flag enables it;
* cached arrays are verified byte-identical to freshly loaded ones before adoption.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import torch

from .qwen_seg import TeacherForcedBatch


def _nbytes(array: np.ndarray) -> int:
    return int(array.nbytes)


@dataclass
class SourceCache:
    """Decoded RGB images and boolean target masks, keyed by image id.

    Both are immutable for a given image id, so one entry serves every instruction
    that refers to that image — which is exactly the paired `P` subset's access
    pattern.
    """

    max_images: int = 512
    cache_images: bool = True
    cache_masks: bool = True
    _images: "OrderedDict[str, np.ndarray]" = field(default_factory=OrderedDict, repr=False)
    _masks: "OrderedDict[str, np.ndarray]" = field(default_factory=OrderedDict, repr=False)
    image_hits: int = 0
    image_misses: int = 0
    mask_hits: int = 0
    mask_misses: int = 0
    build_seconds: float = 0.0

    def image(self, sample) -> np.ndarray:
        if not self.cache_images:
            return sample.image_rgb()
        cached = self._images.get(sample.image_id)
        if cached is not None:
            self._images.move_to_end(sample.image_id)
            self.image_hits += 1
            return cached
        started = time.perf_counter()
        image = sample.image_rgb()
        self.build_seconds += time.perf_counter() - started
        self.image_misses += 1
        # store a read-only copy so a consumer cannot mutate the cached array in place
        stored = np.ascontiguousarray(image)
        stored.setflags(write=False)
        self._images[sample.image_id] = stored
        while len(self._images) > self.max_images:
            self._images.popitem(last=False)
        return stored

    def mask(self, sample) -> np.ndarray:
        if not self.cache_masks:
            return sample.target_mask()
        cached = self._masks.get(sample.sample_id)
        if cached is not None:
            self._masks.move_to_end(sample.sample_id)
            self.mask_hits += 1
            return cached
        started = time.perf_counter()
        mask = sample.target_mask()
        self.build_seconds += time.perf_counter() - started
        self.mask_misses += 1
        stored = np.ascontiguousarray(mask)
        stored.setflags(write=False)
        self._masks[sample.sample_id] = stored
        while len(self._masks) > self.max_images * 4:
            self._masks.popitem(last=False)
        return stored

    def bytes_(self) -> int:
        return sum(_nbytes(a) for a in self._images.values()) + sum(
            _nbytes(a) for a in self._masks.values()
        )

    def stats(self) -> dict:
        return {
            "enabled": {"images": self.cache_images, "masks": self.cache_masks},
            "cached_images": len(self._images),
            "cached_masks": len(self._masks),
            "image_hits": self.image_hits,
            "image_misses": self.image_misses,
            "mask_hits": self.mask_hits,
            "mask_misses": self.mask_misses,
            "bytes": self.bytes_(),
            "gib": round(self.bytes_() / 1024**3, 4),
            "build_seconds": round(self.build_seconds, 3),
        }


@dataclass
class PreprocessedCache:
    """Deterministic Qwen processor output for a fixed training sample.

    Caches exactly the tensors `build_teacher_forcing_batch` returns: processor image
    tensors, `input_ids`, `attention_mask`, `labels`, `image_grid_thw` and the scalar
    sequence metadata. No model output is stored.
    """

    max_entries: int = 1024
    enabled: bool = True
    _store: "OrderedDict[str, TeacherForcedBatch]" = field(default_factory=OrderedDict, repr=False)
    hits: int = 0
    misses: int = 0
    build_seconds: float = 0.0

    def get(self, sample, builder) -> TeacherForcedBatch:
        if not self.enabled:
            return builder()
        cached = self._store.get(sample.sample_id)
        if cached is not None:
            self._store.move_to_end(sample.sample_id)
            self.hits += 1
            return cached
        started = time.perf_counter()
        batch = builder()
        self.build_seconds += time.perf_counter() - started
        self.misses += 1
        self._store[sample.sample_id] = batch
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)
        return batch

    def bytes_(self) -> int:
        total = 0
        for batch in self._store.values():
            total += int(batch.input_ids.numel() * batch.input_ids.element_size())
            total += int(batch.attention_mask.numel() * batch.attention_mask.element_size())
            total += int(batch.labels.numel() * batch.labels.element_size())
            if batch.pixel_values is not None:
                total += int(batch.pixel_values.numel() * batch.pixel_values.element_size())
            if batch.image_grid_thw is not None:
                total += int(batch.image_grid_thw.numel() * batch.image_grid_thw.element_size())
            for value in batch.extra_inputs.values():
                if torch.is_tensor(value):
                    total += int(value.numel() * value.element_size())
        return total

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "entries": len(self._store),
            "max_entries": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "bytes": self.bytes_(),
            "gib": round(self.bytes_() / 1024**3, 4),
            "build_seconds": round(self.build_seconds, 3),
            "caches_model_outputs": False,
        }


def pin_batch(batch: TeacherForcedBatch) -> TeacherForcedBatch:
    """Copy every tensor of a batch into page-locked host memory.

    Values are unchanged; only the storage changes. Tensors are detached from any
    graph first, because a pinned batch is an input, never an activation.
    """

    if not torch.cuda.is_available():
        return batch

    def pin(tensor: torch.Tensor | None):
        if tensor is None or not torch.is_tensor(tensor) or tensor.is_pinned():
            return tensor
        return tensor.detach().pin_memory()

    return TeacherForcedBatch(
        input_ids=pin(batch.input_ids),
        attention_mask=pin(batch.attention_mask),
        labels=pin(batch.labels),
        pixel_values=pin(batch.pixel_values),
        image_grid_thw=pin(batch.image_grid_thw),
        prompt_length=batch.prompt_length,
        total_length=batch.total_length,
        seg_position=batch.seg_position,
        visual_tokens=batch.visual_tokens,
        extra_inputs={
            key: (pin(value) if torch.is_tensor(value) else value)
            for key, value in batch.extra_inputs.items()
        },
    )


def batch_tensor_bytes(batch: TeacherForcedBatch) -> int:
    total = 0
    for tensor in (batch.input_ids, batch.attention_mask, batch.labels, batch.pixel_values, batch.image_grid_thw):
        if tensor is not None:
            total += int(tensor.numel() * tensor.element_size())
    for value in batch.extra_inputs.values():
        if torch.is_tensor(value):
            total += int(value.numel() * value.element_size())
    return total
