"""Task 6C.5: a modular, switchable batch-1 training pipeline.

Every switch here is a *pipeline* change: which CPU work is cached, whether host
memory is pinned, how the copy is issued, and whether preparation overlaps with the
GPU. None of them changes what the model computes, and Task 6C.5 section 18 requires
bit-identical prepared tensors, losses, gradients and post-step parameters before any
of them can become the default.

With `PipelineFlags()` (all switches off) this reproduces the unmodified Task 6C path:
`sample.image_rgb()` -> `processor` -> `sample.target_mask()` -> `.to(device)` ->
`runtime.train_step`.
"""

from __future__ import annotations

import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass

from .input_cache import PreprocessedCache, SourceCache, pin_batch
from .perf import StageProfile


@dataclass
class PipelineFlags:
    """Pipeline switches. All default to the Task 6C behaviour."""

    source_cache: bool = False
    preprocessed_cache: bool = False
    pin_memory: bool = False
    non_blocking: bool = False
    prefetch_threads: int = 0
    skip_grad_norm_instrumentation: bool = False

    def as_dict(self) -> dict:
        return {
            "source_cache": self.source_cache,
            "preprocessed_cache": self.preprocessed_cache,
            "pin_memory": self.pin_memory,
            "non_blocking": self.non_blocking,
            "prefetch_threads": int(self.prefetch_threads),
            "skip_grad_norm_instrumentation": self.skip_grad_norm_instrumentation,
        }

    @property
    def is_baseline(self) -> bool:
        return not any(
            (
                self.source_cache,
                self.preprocessed_cache,
                self.pin_memory,
                self.non_blocking,
                self.prefetch_threads,
                self.skip_grad_norm_instrumentation,
            )
        )


class TrainingPipeline:
    """One-sample-per-optimizer-step loop with optional CPU-side optimizations."""

    def __init__(self, runtime, flags: PipelineFlags | None = None) -> None:
        self.runtime = runtime
        self.flags = flags or PipelineFlags()
        self.source = SourceCache()
        self.preprocessed = PreprocessedCache(enabled=self.flags.preprocessed_cache)
        self._prepared: dict[str, tuple] = {}
        self._condition: threading.Condition | None = None
        self._max_ready = 0
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self.prepare_seconds = 0.0

    # -- preparation ------------------------------------------------------

    def _stage(self, timer: StageProfile | None, name: str, enabled: bool = True):
        if timer is None or not enabled:
            return nullcontext()
        return timer.stage(name)

    def prepare(self, sample, timer: StageProfile | None = None):
        """Return `(cpu_batch, image, gt_mask)` for one sample using the caches."""

        started = time.perf_counter()
        with self._stage(timer, "image_io", self.flags.source_cache or not self.flags.preprocessed_cache):
            image = self.source.image(sample) if self.flags.source_cache else sample.image_rgb()

        def _build():
            with self._stage(timer, "qwen_prepare_total"):
                batch, _image = self.runtime.prepare(sample, image=image)
                return batch

        batch = self.preprocessed.get(sample, _build)

        with self._stage(timer, "target_mask_io", self.flags.source_cache or not self.flags.preprocessed_cache):
            gt_mask = self.source.mask(sample) if self.flags.source_cache else sample.target_mask()

        if self.flags.pin_memory:
            batch = pin_batch(batch)
        self.prepare_seconds += time.perf_counter() - started
        return batch, image, gt_mask

    # -- optional bounded prefetch ---------------------------------------

    def start_prefetch(self, sequence) -> None:
        """Start a bounded background preparation pool for an exact step sequence.

        `sequence` is the ordered list of samples the caller is about to request,
        including warmup steps. The pool is keyed by **request position**, not by
        sample id and not by completion order:

        * the workers produce positions `0 .. len(sequence)-1` in order;
        * the consumer asks for its own position, so step *i* always receives the batch
          prepared for the sample step *i* is supposed to train on — Task 6C.5 section
          18 requires the optimized pipeline to consume the same samples in the same
          order as the baseline;
        * the buffer is bounded, and because positions are monotonic the producer can
          never block on a position the consumer will not ask for. (A pool keyed by
          sample id deadlocks as soon as the consumer restarts the sequence while the
          buffer is full, which is exactly what a warmup pass followed by a measured
          pass does.)
        """

        if self.flags.prefetch_threads <= 0:
            return
        positions = len(sequence)
        self._prepared: dict[int, tuple] = {}
        self._condition = threading.Condition()
        self._stop.clear()
        self._max_ready = max(2, self.flags.prefetch_threads * 2)
        cursor = {"index": 0, "lock": threading.Lock()}

        def worker():
            while not self._stop.is_set():
                with cursor["lock"]:
                    if self._stop.is_set() or cursor["index"] >= positions:
                        return
                    index = cursor["index"]
                    cursor["index"] = index + 1
                sample = sequence[index]
                try:
                    item = self.prepare(sample)
                except Exception:  # noqa: BLE001 - a failed prefetch must not deadlock the loop
                    return
                with self._condition:
                    while len(self._prepared) >= self._max_ready and not self._stop.is_set():
                        self._condition.wait(timeout=0.2)
                    if self._stop.is_set():
                        return
                    self._prepared[index] = item
                    self._condition.notify_all()

        self._threads = [
            threading.Thread(target=worker, daemon=True) for _ in range(self.flags.prefetch_threads)
        ]
        for thread in self._threads:
            thread.start()

    def stop_prefetch(self) -> None:
        self._stop.set()
        if self._condition is not None:
            with self._condition:
                self._condition.notify_all()
        for thread in self._threads:
            thread.join(timeout=5)
        self._threads = []
        self._prepared = {}

    def next_prepared(self, sample, position: int):
        """The prepared batch for `position`: from the pool, else prepared inline."""

        if not self._threads:
            return self.prepare(sample)
        with self._condition:
            while position not in self._prepared and not self._stop.is_set():
                self._condition.wait(timeout=0.2)
            item = self._prepared.pop(position, None)
            self._condition.notify_all()
        return item if item is not None else self.prepare(sample)

    # -- one optimization step -------------------------------------------

    def step(self, sample, prepared, optimizer, timer: StageProfile | None = None) -> dict:
        batch, image, gt_mask = prepared
        with self._stage(timer, "cpu_to_gpu"):
            moved = batch.to(self.runtime.device, non_blocking=self.flags.non_blocking)
        with self._stage(timer, "sam_feature_lookup"):
            features, cached = self.runtime.features_for(sample, image)
        result = self.runtime.train_step(
            moved,
            gt_mask,
            features,
            optimizer=optimizer,
            timer=timer,
            collect_grad_norms=not self.flags.skip_grad_norm_instrumentation,
        )
        result["feature_cache_hit"] = bool(cached)
        return result

    def stats(self) -> dict:
        return {
            "flags": self.flags.as_dict(),
            "source_cache": self.source.stats(),
            "preprocessed_cache": self.preprocessed.stats(),
            "prepare_seconds": round(self.prepare_seconds, 3),
            "prefetch_threads": len(self._threads),
        }


def warm_caches(pipeline: TrainingPipeline, samples, timer: StageProfile | None = None) -> dict:
    """Populate the enabled caches and the SAM feature cache without any optimizer step."""

    started = time.perf_counter()
    encoded = 0
    for sample in samples:
        pipeline.prepare(sample, timer=timer)
        image = pipeline.source.image(sample) if pipeline.flags.source_cache else None
        if image is None:
            image = sample.image_rgb()
        _features, cached = pipeline.runtime.features_for(sample, image)
        encoded += 0 if cached else 1
    return {
        "seconds": round(time.perf_counter() - started, 3),
        "samples": len(samples),
        "sam_features_encoded": encoded,
        "source_cache": pipeline.source.stats(),
        "preprocessed_cache": pipeline.preprocessed.stats(),
    }
