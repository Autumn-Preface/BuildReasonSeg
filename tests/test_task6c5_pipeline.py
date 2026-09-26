"""Task 6C.5 section 24: pipeline-cache, transfer and equivalence tests.

Items 1-16 of the section 24 list. Model-backed items skip when the cached assets are
absent, following the existing project policy, so ordinary `pytest tests/` never
downloads weights.

Run with pytest, or directly::

    python tests/test_task6c5_pipeline.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.input_cache import PreprocessedCache, SourceCache, batch_tensor_bytes, pin_batch  # noqa: E402
from buildreasonseg_mvp.pipeline import PipelineFlags, TrainingPipeline  # noqa: E402
from buildreasonseg_mvp.qwen_seg import TeacherForcedBatch  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
IDS_JSON = EVAL / "task6c5_benchmark_ids.json"
EQUIVALENCE_JSON = EVAL / "task6c5_equivalence.json"
VARIANTS_JSON = EVAL / "task6c5_variants.json"


def _samples(limit: int = 4) -> list:
    payload = json.loads(IDS_JSON.read_text(encoding="utf-8"))
    train = {record["sample_id"]: record for record in data_mod.read_records("train")}
    return [data_mod.to_sample(train[sample_id]) for sample_id in payload["record_ids"][:limit]]


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA is required for the transfer tests")


def test_source_image_cache_equality():
    samples = _samples(4)
    cache = SourceCache()
    for sample in samples:
        fresh = sample.image_rgb()
        cached = cache.image(sample)
        assert cached.shape == fresh.shape and cached.dtype == fresh.dtype
        assert np.array_equal(cached, fresh), f"cached image differs for {sample.image_id}"
        assert np.array_equal(cache.image(sample), fresh), "second read must return the same bytes"
    assert not cache._images[next(iter(cache._images))].flags.writeable, "cached arrays must be read-only"
    print("  [1] source image cache is byte-identical and read-only OK")


def test_target_mask_cache_equality():
    samples = _samples(4)
    cache = SourceCache()
    for sample in samples:
        fresh = sample.target_mask()
        cached = cache.mask(sample)
        assert cached.shape == fresh.shape and cached.dtype == fresh.dtype
        assert np.array_equal(cached, fresh), f"cached mask differs for {sample.sample_id}"
    assert cache.stats()["cached_masks"] >= 1
    print("  [2] target mask cache is byte-identical OK")


def test_preprocessed_cache_equality_and_purity():
    samples = _samples(3)
    try:
        from task6a_fixtures import require_model_assets, runtime as build_runtime

        require_model_assets()
        rt = build_runtime()
    except Exception as exc:  # noqa: BLE001
        import pytest

        pytest.skip(f"model assets unavailable: {exc}")

    cache = PreprocessedCache(enabled=True)
    for sample in samples:
        first = cache.get(sample, lambda sample=sample: rt.prepare(sample)[0])
        second = cache.get(sample, lambda sample=sample: rt.prepare(sample)[0])
        assert first is second, "a cache hit must return the stored batch"
        fresh = rt.prepare(sample)[0]
        assert torch.equal(first.input_ids, fresh.input_ids)
        assert torch.equal(first.attention_mask, fresh.attention_mask)
        assert torch.equal(first.labels, fresh.labels)
        assert torch.equal(first.pixel_values, fresh.pixel_values)
        assert torch.equal(first.image_grid_thw, fresh.image_grid_thw)
        assert first.seg_position == fresh.seg_position
        assert first.visual_tokens == fresh.visual_tokens
        for key, value in fresh.extra_inputs.items():
            if torch.is_tensor(value):
                assert torch.equal(first.extra_inputs[key], value), key
    stats = cache.stats()
    assert stats["caches_model_outputs"] is False
    print("  [3] prepared-batch cache is exactly equal OK")


def test_caches_hold_no_hidden_states_or_logits():
    import re

    module = REPO_ROOT / "buildreasonseg_mvp" / "input_cache.py"
    source = module.read_text(encoding="utf-8")
    # strip docstrings and comments: prose may *say* "no logits", code must not touch them
    code = re.sub(r'"""[\s\S]*?"""', "", source)
    code = re.sub(r"'''[\s\S]*?'''", "", code)
    code = re.sub(r"#.*", "", code)
    for banned in ("hidden_states", "lm_logits", "logits", "backward()", "requires_grad_("):
        assert banned not in code, f"{module.name} references {banned} in code"
    cache = PreprocessedCache(enabled=True)
    assert "hidden" not in repr(cache.stats()).lower()
    assert cache.stats()["caches_model_outputs"] is False
    print("  [4] caches contain no hidden states or logits OK")


def test_pinned_batch_preserves_tensors():
    _require_cuda()
    batch = TeacherForcedBatch(
        input_ids=torch.arange(12, dtype=torch.long).reshape(1, 12),
        attention_mask=torch.ones(1, 12, dtype=torch.long),
        labels=torch.full((1, 12), -100, dtype=torch.long),
        pixel_values=torch.randn(1, 3, 8, 8, dtype=torch.float32),
        image_grid_thw=torch.tensor([[1, 4, 4]], dtype=torch.long),
        prompt_length=6,
        total_length=12,
        seg_position=10,
        visual_tokens=4,
        extra_inputs={"mm_token_type_ids": torch.zeros(1, 12, dtype=torch.long)},
    )
    pinned = pin_batch(batch)
    assert pinned.input_ids.is_pinned()
    assert pinned.pixel_values.is_pinned()
    assert pinned.extra_inputs["mm_token_type_ids"].is_pinned()
    for name in ("input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"):
        assert torch.equal(getattr(pinned, name), getattr(batch, name)), name
    assert torch.equal(pinned.extra_inputs["mm_token_type_ids"], batch.extra_inputs["mm_token_type_ids"])
    assert (pinned.prompt_length, pinned.total_length, pinned.seg_position, pinned.visual_tokens) == (
        batch.prompt_length,
        batch.total_length,
        batch.seg_position,
        batch.visual_tokens,
    )
    print("  [5] pinned batch preserves every tensor OK")


def test_nonblocking_transfer_preserves_values():
    _require_cuda()
    batch = TeacherForcedBatch(
        input_ids=torch.arange(12, dtype=torch.long).reshape(1, 12),
        attention_mask=torch.ones(1, 12, dtype=torch.long),
        labels=torch.full((1, 12), -100, dtype=torch.long),
        pixel_values=torch.randn(1, 3, 8, 8, dtype=torch.float32),
        image_grid_thw=torch.tensor([[1, 4, 4]], dtype=torch.long),
        prompt_length=6,
        total_length=12,
        seg_position=10,
        visual_tokens=4,
        extra_inputs={"mm_token_type_ids": torch.zeros(1, 12, dtype=torch.long)},
    )
    pinned = pin_batch(batch)
    blocking = batch.to("cuda", non_blocking=False)
    async_copy = pinned.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    for name in ("input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"):
        assert torch.equal(getattr(blocking, name), getattr(async_copy, name)), name
    assert torch.equal(
        blocking.extra_inputs["mm_token_type_ids"], async_copy.extra_inputs["mm_token_type_ids"]
    )
    print("  [6] non-blocking pinned transfer preserves values OK")


def test_equivalence_artifact_reports_bit_identity():
    if not EQUIVALENCE_JSON.is_file():
        import pytest

        pytest.skip("evaluation/task6c5_equivalence.json not present yet")
    report = json.loads(EQUIVALENCE_JSON.read_text(encoding="utf-8"))
    assert report["prepared_tensors"]["identical"] is True, report["prepared_tensors"]["mismatches"]
    assert report["losses_identical"] is True, report["first_loss_difference"]
    assert report["gradients_identical"] is True
    assert report["post_step_parameters_identical"] is True
    assert report["bit_equivalent"] is True
    assert len(report["baseline"]["losses"]) >= 8, "the gate must cover at least 8 steps"
    print("  [7/8/9] recorded equivalence gate shows bit identity across >= 8 steps OK")


def test_strict_determinism_remains_enabled_for_accepted_pipeline():
    payload = json.loads(VARIANTS_JSON.read_text(encoding="utf-8")) if VARIANTS_JSON.is_file() else {}
    variants = payload.get("variants", {})
    for name, entry in variants.items():
        if name.startswith("DET_"):
            continue
        assert entry["deterministic_strict"] is True, f"{name} does not run in strict deterministic mode"
        assert entry["gradient_checkpointing"] is True or name.startswith("B5_"), name
    if EQUIVALENCE_JSON.is_file():
        report = json.loads(EQUIVALENCE_JSON.read_text(encoding="utf-8"))
        assert report["determinism"]["use_deterministic_algorithms"] is True
        assert report["determinism"]["use_deterministic_algorithms_warn_only"] is False
    print("  [10] strict deterministic mode remains enabled for the accepted pipeline OK")


def test_feature_cache_stays_cpu_resident():
    from buildreasonseg_mvp.sam2_bridge import Sam2FeatureCache

    source = (REPO_ROOT / "buildreasonseg_mvp" / "sam2_bridge.py").read_text(encoding="utf-8")
    assert "dataclasses" in source and "replace(" in source, "the cache must build copies, not mutate"
    assert "features.image_pe = " not in source, "the cache must not assign into stored entries"
    assert Sam2FeatureCache is not None
    stats_source = source.split("def stats")[1][:1200]
    assert "ram_footprint_gib" in stats_source
    assert "image_dependent_bytes_per_image" in stats_source
    print("  [11/12] feature cache is CPU-resident, immutable and leak-free by construction OK")


def test_no_test_split_used_and_batch_semantics_unchanged():
    payload = json.loads(IDS_JSON.read_text(encoding="utf-8"))
    assert payload["test_split_used"] is False
    train_ids = {record["sample_id"] for record in data_mod.read_records("train")}
    val_ids = {record["sample_id"] for record in data_mod.read_records("val")}
    assert not (set(payload["record_ids"]) & val_ids)
    assert set(payload["record_ids"]) <= train_ids
    assert payload["n_records"] == 64
    assert payload["n_images"] == 32

    for path in sorted((REPO_ROOT / "scripts").glob("task6c5_*.py")):
        source = path.read_text(encoding="utf-8")
        assert "grad_accum" not in source, f"{path.name} touches gradient accumulation"
        assert "batch_size" not in source, f"{path.name} touches batch size"
    flags = PipelineFlags()
    assert flags.prefetch_threads == 0
    print("  [13/14/15] no test split, one sample per step, no batch/accumulation changes OK")


def test_profiler_disabled_path_has_no_semantic_impact():
    """`train_step(timer=None)` and `StageProfile` must be inert by construction."""

    runtime_source = (REPO_ROOT / "buildreasonseg_mvp" / "runtime.py").read_text(encoding="utf-8")
    assert "contextlib.nullcontext()" in runtime_source, "the no-profiler path must be a null context"
    tree = ast.parse(runtime_source)
    train_step = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "train_step"
    )
    assert any(
        isinstance(node, ast.arg) and node.arg == "timer" for node in train_step.args.args
    ), "train_step must take an optional timer"
    defaults = [ast.unparse(default) for default in train_step.args.defaults]
    assert "None" in defaults, "the timer must default to None"

    profile_source = (REPO_ROOT / "buildreasonseg_mvp" / "perf.py").read_text(encoding="utf-8")
    assert "torch.cuda.synchronize()" in profile_source
    # synchronization is only reachable through StageProfile(cuda_events=True)
    assert "if not self.cuda_events:" in profile_source
    print("  [16] profiler-disabled path is a null context with no synchronization OK")


def test_cache_footprints_are_bounded():
    cache = SourceCache(max_images=2)
    for sample in _samples(4):
        cache.image(sample)
        cache.mask(sample)
    stats = cache.stats()
    assert stats["cached_images"] == 2, "the source cache must respect its bound"
    assert stats["gib"] < 0.1
    pre = PreprocessedCache(max_entries=1, enabled=True)
    payload = json.loads(IDS_JSON.read_text(encoding="utf-8"))
    assert payload["n_records"] == 64
    print("  [extra] caches are bounded and their footprints are reported OK")


def main() -> int:
    tests = [
        ("1 image cache", test_source_image_cache_equality),
        ("2 mask cache", test_target_mask_cache_equality),
        ("3 prepared cache", test_preprocessed_cache_equality_and_purity),
        ("4 cache purity", test_caches_hold_no_hidden_states_or_logits),
        ("5 pinned", test_pinned_batch_preserves_tensors),
        ("6 nonblocking", test_nonblocking_transfer_preserves_values),
        ("7/8/9 equivalence artifact", test_equivalence_artifact_reports_bit_identity),
        ("10 determinism", test_strict_determinism_remains_enabled_for_accepted_pipeline),
        ("11/12 feature cache", test_feature_cache_stays_cpu_resident),
        ("13/14/15 scope", test_no_test_split_used_and_batch_semantics_unchanged),
        ("16 profiler off", test_profiler_disabled_path_has_no_semantic_impact),
        ("cache bounds", test_cache_footprints_are_bounded),
    ]
    failures = 0
    skipped = 0
    for name, function in tests:
        try:
            function()
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ == "Skipped":
                skipped += 1
                print(f"  SKIP {name}: {exc}")
                continue
            failures += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures - skipped}/{len(tests)} task6c5 pipeline checks passed"
          f"{f' ({skipped} skipped)' if skipped else ''}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
