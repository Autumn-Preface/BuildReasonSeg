"""Task 6C.7: bounded CPU cache for the frozen Qwen3-VL visual tower output.

Why this is cacheable
---------------------
The Qwen3-VL visual tower is frozen in this project (no visual LoRA, ADR-013), and its
input is the deterministic processor output for one source image. Its output is therefore
a pure function of the image, while every trainable parameter lives in the language-model
LoRA adapters, the `[SEG]` token and the SAM decoder. `task6c7_visual_cache.py` proves that
eligibility before anything here is enabled.

The cached boundary
-------------------
`Qwen3VLModel.forward` uses exactly two things from the visual tower:

    image_outputs = self.get_image_features(pixel_values, image_grid_thw)
    image_embeds  = image_outputs.pooler_output        # merger output, split per image
    deepstack      = image_outputs.deepstack_features  # deepstack merger outputs

so the cache stores those two and nothing else. In particular it never stores
`last_hidden_state` (the pre-merger tower state, which the language model does not read),
language-model hidden states, the `[SEG]` state or logits.

Cache key
---------
The key is a content hash of the processed `pixel_values` plus `image_grid_thw`, or an
explicit source-image identity when the caller supplies one. Both are functions of the
source image alone, which is what section 8 requires: two instructions on one image must
hit one entry, and no sample-id plumbing mistake can return another image's features.
Entries hold detached CPU copies, so a hit cannot keep an autograd graph alive.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import torch

#: Task 6C.7 section 8: keep this cache inside ~8 GiB unless justification is recorded.
DEFAULT_MAX_IMAGES = 512
BYTES_PER_GIB = 1024**3


def _tensor_bytes(tensors) -> int:
    total = 0
    for tensor in tensors:
        if torch.is_tensor(tensor):
            total += tensor.numel() * tensor.element_size()
    return total


@dataclass
class CachedVisual:
    """One cached visual-tower output: CPU, detached, contiguous."""

    pooler: tuple[torch.Tensor, ...]
    deepstack: tuple[torch.Tensor, ...]
    bytes_: int
    key_kind: str

    def as_dict(self) -> dict:
        return {
            "pooler_shapes": [list(tensor.shape) for tensor in self.pooler],
            "deepstack_shapes": [list(tensor.shape) for tensor in self.deepstack],
            "dtype": str(self.pooler[0].dtype) if self.pooler else None,
            "bytes": self.bytes_,
            "key_kind": self.key_kind,
        }


@dataclass
class VisualCacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    stored: int = 0
    bytes_stored: int = 0
    build_seconds: float = 0.0
    hit_seconds: float = 0.0
    miss_seconds: float = 0.0
    rebuilt_for_determinism_checks: int = 0
    keys_by_kind: dict = field(default_factory=dict)


class VisualFeatureCache:
    """Bounded LRU cache of frozen visual-tower outputs, held on the CPU."""

    def __init__(self, max_images: int = DEFAULT_MAX_IMAGES, enabled: bool = True) -> None:
        self.max_images = int(max_images)
        self.enabled = bool(enabled)
        self._entries: "OrderedDict[str, CachedVisual]" = OrderedDict()
        self._lock = threading.Lock()
        self.stats_ = VisualCacheStats()

    # -- keys ------------------------------------------------------------

    @staticmethod
    def content_key(pixel_values: torch.Tensor, image_grid_thw: torch.Tensor | None) -> str:
        """Content hash of the processed image input.

        A hash rather than a sample id: several instructions share one image, and a hash
        cannot be silently mis-plumbed into another image's features.
        """

        digest = hashlib.blake2b(digest_size=16)
        if torch.is_tensor(pixel_values):
            digest.update(pixel_values.detach().to("cpu").contiguous().numpy().tobytes())
        if torch.is_tensor(image_grid_thw):
            digest.update(image_grid_thw.detach().to("cpu").contiguous().numpy().tobytes())
        return digest.hexdigest()

    def key_for(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
        explicit_key: str | None = None,
    ) -> tuple[str, str]:
        """(key, key_kind). An explicit image identity avoids hashing the pixels."""

        if explicit_key:
            return f"image:{explicit_key}", "image_identity"
        return f"content:{self.content_key(pixel_values, image_grid_thw)}", "content_hash"

    # -- storage ---------------------------------------------------------

    def get(self, key: str) -> CachedVisual | None:
        if not self.enabled:
            return None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
        return entry

    def put(self, key: str, pooler, deepstack, key_kind: str) -> CachedVisual:
        """Store detached CPU copies; never stores anything else."""

        stored_pooler = tuple(
            tensor.detach().to("cpu").contiguous() for tensor in pooler if torch.is_tensor(tensor)
        )
        stored_deepstack = tuple(
            tensor.detach().to("cpu").contiguous()
            for tensor in deepstack
            if torch.is_tensor(tensor)
        )
        entry = CachedVisual(
            pooler=stored_pooler,
            deepstack=stored_deepstack,
            bytes_=_tensor_bytes(stored_pooler) + _tensor_bytes(stored_deepstack),
            key_kind=key_kind,
        )
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
            self._entries[key] = entry
            self.stats_.stored += 1
            self.stats_.bytes_stored += entry.bytes_
            self.stats_.keys_by_kind[key_kind] = self.stats_.keys_by_kind.get(key_kind, 0) + 1
            while len(self._entries) > self.max_images:
                _evicted_key, evicted = self._entries.popitem(last=False)
                self.stats_.bytes_stored -= evicted.bytes_
                self.stats_.evictions += 1
        return entry

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.stats_.bytes_stored = 0

    # -- reporting -------------------------------------------------------

    def entries(self) -> int:
        with self._lock:
            return len(self._entries)

    def resident_bytes(self) -> int:
        with self._lock:
            return sum(entry.bytes_ for entry in self._entries.values())

    def bytes_per_image(self) -> float:
        count = self.entries()
        return (self.resident_bytes() / count) if count else 0.0

    def estimate_gib(self, images: int) -> float:
        per_image = self.bytes_per_image()
        if not per_image:
            return 0.0
        return round(per_image * images / BYTES_PER_GIB, 3)

    def stats(self) -> dict:
        stats = self.stats_
        return {
            "enabled": self.enabled,
            "max_images": self.max_images,
            "entries": self.entries(),
            "bytes": self.resident_bytes(),
            "gib": round(self.resident_bytes() / BYTES_PER_GIB, 4),
            "bytes_per_image": round(self.bytes_per_image(), 1),
            "hits": stats.hits,
            "misses": stats.misses,
            "hit_rate": round(stats.hits / max(1, stats.hits + stats.misses), 4),
            "evictions": stats.evictions,
            "build_seconds": round(stats.build_seconds, 4),
            "hit_seconds_total": round(stats.hit_seconds, 4),
            "miss_seconds_total": round(stats.miss_seconds, 4),
            "mean_hit_seconds": round(stats.hit_seconds / max(1, stats.hits), 6),
            "mean_miss_seconds": round(stats.miss_seconds / max(1, stats.misses), 6),
            "keys_by_kind": dict(stats.keys_by_kind),
            "stores_model_outputs": False,
            "stores": ["pooler_output", "deepstack_features"],
        }


# ------------------------------------------------------------------ install


def _rebuild_output(template, pooler, deepstack, device):
    """A `BaseModelOutputWithDeepstackFeatures` carrying the cached tensors.

    `last_hidden_state` is intentionally None: the language model reads only
    `pooler_output` and `deepstack_features`, and storing the pre-merger tower state
    would roughly double the cache for a tensor nothing consumes.
    """

    from transformers.models.qwen3_vl.modeling_qwen3_vl import (  # noqa: PLC0415
        BaseModelOutputWithDeepstackFeatures,
    )

    moved_pooler = tuple(tensor.to(device=device) for tensor in pooler)
    moved_deepstack = [tensor.to(device=device) for tensor in deepstack]
    return BaseModelOutputWithDeepstackFeatures(
        last_hidden_state=None,
        pooler_output=moved_pooler,
        deepstack_features=moved_deepstack,
    )


def resolve_visual_host(qwen):
    """The `Qwen3VLModel` that owns `.visual` and `get_image_features`.

    Accepts the `Qwen3VLForConditionalGeneration` (the project's `model.qwen`), the
    `Qwen3VLModel` itself, or the project's `BuildReasonSegMvp` container.
    """

    candidate = qwen
    for _ in range(3):
        if hasattr(candidate, "visual") and hasattr(candidate, "get_image_features"):
            return candidate
        inner = getattr(candidate, "model", None)
        if inner is None or inner is candidate:
            break
        candidate = inner
    raise ValueError(f"cannot locate a Qwen3-VL visual host from {type(qwen).__name__}")


def install_visual_feature_cache(model, cache: VisualFeatureCache, key_provider=None) -> dict:
    """Wrap `get_image_features` on our own model instance with a bounded cache.

    This is an instance-level wrapper on the project's own module, not a patch of library
    source: the original bound method is kept and called unchanged on a miss, so the miss
    path is byte-identical to the current behaviour. The wrapper is reversible through
    `remove_visual_feature_cache`.

    `key_provider` returns the current source-image identity (or None); when it returns
    None the wrapper falls back to the content hash, which is always correct.
    """

    target = resolve_visual_host(model)

    if getattr(target, "_task6c7_original_get_image_features", None) is None:
        target._task6c7_original_get_image_features = target.get_image_features
    original = target._task6c7_original_get_image_features

    import time  # noqa: PLC0415

    def cached_get_image_features(pixel_values, image_grid_thw=None, **kwargs):
        if not cache.enabled:
            return original(pixel_values, image_grid_thw, **kwargs)
        explicit = None
        if key_provider is not None:
            try:
                explicit = key_provider()
            except Exception:  # noqa: BLE001
                explicit = None
        key, key_kind = cache.key_for(pixel_values, image_grid_thw, explicit)
        hit = cache.get(key)
        if hit is not None:
            started = time.perf_counter()
            result = _rebuild_output(None, hit.pooler, hit.deepstack, pixel_values.device)
            cache.stats_.hits += 1
            cache.stats_.hit_seconds += time.perf_counter() - started
            return result
        started = time.perf_counter()
        output = original(pixel_values, image_grid_thw, **kwargs)
        cache.stats_.miss_seconds += time.perf_counter() - started
        cache.stats_.misses += 1
        build_started = time.perf_counter()
        cache.put(key, output.pooler_output, output.deepstack_features or (), key_kind)
        cache.stats_.build_seconds += time.perf_counter() - build_started
        return output

    target.get_image_features = cached_get_image_features
    return {
        "installed": True,
        "target": type(target).__name__,
        "enabled": cache.enabled,
        "max_images": cache.max_images,
        "wrapped": "get_image_features",
        "stores": ["pooler_output", "deepstack_features"],
        "miss_path": "original bound method, unchanged",
    }


def remove_visual_feature_cache(model) -> bool:
    target = resolve_visual_host(model)
    original = getattr(target, "_task6c7_original_get_image_features", None)
    if original is None:
        return False
    target.get_image_features = original
    return True


def visual_tower_trainable_report(model) -> dict:
    """Section 5 checks 1-4: frozen parameters, no visual LoRA, no training-time state."""

    target = resolve_visual_host(model)
    visual = target.visual

    trainable = []
    total = 0
    dtypes = {}
    for name, parameter in visual.named_parameters():
        total += 1
        dtypes[str(parameter.dtype)] = dtypes.get(str(parameter.dtype), 0) + 1
        if parameter.requires_grad:
            trainable.append(name)

    from peft.tuners.lora.layer import LoraLayer  # noqa: PLC0415

    lora_modules = [name for name, module in visual.named_modules() if isinstance(module, LoraLayer)]

    dropout_modules = []
    buffer_modules = []
    for name, module in visual.named_modules():
        if isinstance(module, torch.nn.Dropout) and float(getattr(module, "p", 0.0)) > 0.0:
            dropout_modules.append(f"{name}:p={module.p}")
        buffers = [buffer_name for buffer_name, buffer in module.named_buffers(recurse=False) if buffer is not None]
        if buffers:
            buffer_modules.append(f"{name}:{buffers}")

    return {
        "visual_module_class": type(visual).__name__,
        "parameter_count": total,
        "trainable_parameter_names": trainable,
        "all_parameters_frozen": not trainable,
        "parameter_dtypes": dtypes,
        "lora_module_count": len(lora_modules),
        "lora_modules": lora_modules[:10],
        "zero_visual_lora": not lora_modules,
        "dropout_modules_with_p_gt_0": dropout_modules,
        # Registered buffers (e.g. a rotary `inv_freq` table) are constants, not state.
        # Whether any of them actually *changes* during a training step is checked
        # empirically in the eligibility script, which is the condition that matters.
        "modules_with_buffers": buffer_modules,
        "registered_buffer_count": sum(len(module) for module in [buffer_modules]),
        "training_mode": bool(visual.training),
    }
