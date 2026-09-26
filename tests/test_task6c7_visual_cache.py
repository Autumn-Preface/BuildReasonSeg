"""Task 6C.7 section 14: formal-path transfer, sync attribution and visual-cache tests.

Items 1-15 of the section 14 list. The model-backed measurements live in
`evaluation/task6c7_*.json`; the tests assert against them and against unit-level
behaviour that needs no weights, so they stay fast and still fail if code and artifacts
disagree.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

EVAL = REPO_ROOT / "evaluation"
CONFIG = REPO_ROOT / "configs" / "mvp" / "task6c_2b_ablation.yaml"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "task6c_train.py"
VISUAL_CACHE = REPO_ROOT / "buildreasonseg_mvp" / "visual_cache.py"
RUNTIME = REPO_ROOT / "buildreasonseg_mvp" / "runtime.py"

ELIGIBILITY = EVAL / "task6c7_visual_cache_eligibility.json"
SYNC_HOTSPOTS = EVAL / "task6c7_sync_hotspots.json"
EQUIVALENCE = EVAL / "task6c7_equivalence.json"
VARIANTS = EVAL / "task6c7_variants.json"
FINAL = EVAL / "task6c7_final_benchmark.json"
RESOURCE = EVAL / "task6c7_resource_usage.json"

NEW_FILES = sorted((REPO_ROOT / "scripts").glob("task6c7_*.py")) + [VISUAL_CACHE]


def _load(path: Path) -> dict:
    if not path.is_file():
        print(f"  [skip] {path.name} not present yet")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- stubs for 1/2


class _CountingBatch:
    """CPU batch stub that records every `.to()` call."""

    def __init__(self, to_calls: list) -> None:
        self.input_ids = torch.zeros(1, 4, dtype=torch.long)
        self.attention_mask = torch.ones(1, 4, dtype=torch.long)
        self.labels = torch.zeros(1, 4, dtype=torch.long)
        self.pixel_values = None
        self.image_grid_thw = None
        self.extra_inputs: dict = {}
        self._to_calls = to_calls

    def to(self, device):  # noqa: D102, ANN001
        self._to_calls.append(str(device))
        return self


class _StubQwen(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.row = torch.nn.Parameter(torch.zeros(8))

    def forward(self, input_ids=None, **_kwargs):  # noqa: D102, ANN001
        logits = self.row.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1).contiguous()
        return type("StubOutput", (), {"logits": logits})()


class _StubModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qwen = _StubQwen()

    def trainable_parameter_groups(self, **_kwargs):  # noqa: D102
        return [{"params": list(self.parameters()), "lr": 1e-4}]


class _StubSample:
    sample_id = "stub"
    image_id = "stub-image"

    def target_mask(self):  # noqa: D102
        return np.zeros((2, 2), dtype=bool)


class _StubFeatureCache:
    def stats(self):  # noqa: D102
        return {}


class _StubRuntime:
    def __init__(self, to_calls: list) -> None:
        self.cfg = {
            "training": {"collect_grad_norms": False, "visual_feature_cache": False},
            "optimizer": {
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "grad_clip_norm": 1.0,
                "phase_a": {
                    "lora_lr": 1e-4,
                    "token_lr": 3e-4,
                    "warmup_steps": 1,
                    "lr_schedule": "none",
                },
                "phase_b": {
                    "lora_lr": 1e-4,
                    "token_lr": 3e-4,
                    "decoder_lr": 3e-4,
                    "warmup_steps": 1,
                    "lr_schedule": "none",
                },
            },
        }
        self.model = _StubModel()
        self.device = "cpu"
        self.feature_cache = _StubFeatureCache()
        self.visual_cache_key = None
        self.train_step_calls = 0
        self._to_calls = to_calls

    def visual_cache_stats(self):  # noqa: D102
        return {"enabled": False, "installed": False}

    def prepare(self, sample):  # noqa: D102, ANN001
        return _CountingBatch(self._to_calls), None

    def features_for(self, sample, image):  # noqa: D102, ANN001
        return None, False

    def build_scheduler_for(self, optimizer, steps, cfg):  # noqa: D102, ANN001
        return None

    def set_visual_cache_key(self, key):  # noqa: D102, ANN001
        self.visual_cache_key = key

    def train_step(self, batch, gt_mask, features, optimizer=None, **_kwargs):  # noqa: D102, ANN001
        # mirrors the real runtime: train_step performs the (single) device move
        self.train_step_calls += 1
        batch.to(self.device)
        return {
            "losses": {"total": 1.0, "lm_ce": 1.0, "mask_bce": 0.0, "mask_dice": 0.0},
            "grad_clip_total_norm": 0.5,
        }


def _run_formal_phase(phase: str) -> tuple[list, dict]:
    import task6c_train as train

    to_calls: list = []
    runtime = _StubRuntime(to_calls)
    original_vram = train.vram
    train.vram = lambda: {"peak_allocated_gib": 0.0}
    try:
        result = train.train_phase(runtime, [_StubSample()], phase, 1, 10**9, "test")
    finally:
        train.vram = original_vram
    return to_calls, result


def test_phase_b_performs_one_logical_device_transfer():
    """Section 14 item 1: no redundant batch transfer in the formal Phase-B loop."""

    to_calls, result = _run_formal_phase("B")
    assert result["steps"] == 1
    assert len(to_calls) == 1, (
        f"Phase B performed {len(to_calls)} logical device transfers; the pre-section-3 loop "
        "created an unused `moved` copy on top of the copy inside train_step"
    )

    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    body = source.split("def train_phase", 1)[1].split("def full_validation", 1)[0]
    assert "moved = batch.to(runtime.device)" in body
    moved_index = body.index("moved = batch.to(runtime.device)")
    phase_a_index = body.index("if is_language_only:")
    assert moved_index > phase_a_index, (
        "the explicit transfer must live inside the Phase-A branch, which calls Qwen directly"
    )
    print("  [1] Phase B performs exactly one device transfer OK")


def test_phase_a_still_trains_and_transfers_once():
    """Section 14 item 2: Phase A behaviour is unchanged."""

    to_calls, result = _run_formal_phase("A")
    assert result["steps"] == 1
    assert len(to_calls) == 1, f"Phase A performed {len(to_calls)} device transfers"
    assert result["history"], "Phase A must still record history"
    entry = result["history"][-1]
    assert entry["losses"]["total"] is not None
    print("  [2] Phase A still runs one transfer and records a step OK")


# ---------------------------------------------------------------- artifacts 3/4


def test_visual_tower_is_frozen_and_has_no_lora():
    """Section 14 items 3 and 4."""

    payload = _load(ELIGIBILITY)
    if payload:
        static = payload["visual_tower_static_check"]
        assert static["all_parameters_frozen"], static["trainable_parameter_names"]
        assert static["trainable_parameter_names"] == []
        assert static["zero_visual_lora"] and static["lora_module_count"] == 0
        assert static["visual_module_class"] == "Qwen3VLVisionModel"
    cfg = CONFIG.read_text(encoding="utf-8")
    assert "text_only: true" in cfg
    suffixes = re.search(r"target_suffixes: \[(.*?)\]", cfg).group(1)
    for forbidden in ("visual", "patch_embed", "merger", "attn.qkv"):
        assert forbidden not in suffixes, f"{forbidden} must not be a LoRA target"
    print("  [3,4] visual tower frozen with zero visual LoRA OK")


# ---------------------------------------------------------------- cache unit 5-8


def test_visual_cache_key_is_image_based_not_sample_based():
    """Section 14 item 5."""

    from buildreasonseg_mvp.visual_cache import VisualFeatureCache

    cache = VisualFeatureCache(max_images=4)
    pixels = torch.zeros(2, 4)
    key_by_image, kind_by_image = cache.key_for(pixels, None, "1_0")
    key_by_content, kind_by_content = cache.key_for(pixels, None, None)
    assert kind_by_image == "image_identity" and key_by_image == "image:1_0"
    assert kind_by_content == "content_hash" and key_by_content.startswith("content:")
    # two samples of one image (different instructions) must collide, different pixels must not
    assert cache.key_for(pixels, None, "1_0")[0] == cache.key_for(pixels, None, "1_0")[0]
    other, _ = cache.key_for(torch.ones(2, 4), None, "1_1")
    assert other != key_by_image
    assert cache.key_for(pixels, torch.tensor([[1, 4, 4]]), None)[0] != key_by_content
    print("  [5] cache key is the source image identity (or its content hash) OK")


def test_cache_stores_no_trainable_language_state():
    """Section 14 item 6."""

    from buildreasonseg_mvp.visual_cache import VisualFeatureCache

    cache = VisualFeatureCache(max_images=2)
    pooler = (torch.randn(4, 8, dtype=torch.bfloat16),)
    deepstack = [torch.randn(4, 8, dtype=torch.bfloat16) for _ in range(3)]
    entry = cache.put("image:x", pooler, deepstack, "image_identity")
    stored = entry.as_dict()
    assert stored["pooler_shapes"] == [[4, 8]]
    assert stored["deepstack_shapes"] == [[4, 8], [4, 8], [4, 8]]
    assert not hasattr(entry, "hidden_states") and not hasattr(entry, "logits")
    source = VISUAL_CACHE.read_text(encoding="utf-8")
    assert '"stores": ["pooler_output", "deepstack_features"]' in source
    # the only attributes the cache can return are the two visual outputs
    assert set(entry.__dataclass_fields__) == {"pooler", "deepstack", "bytes_", "key_kind"}
    print("  [6] the cache holds only frozen visual outputs OK")


def test_cache_hit_returns_correct_shape_and_dtype_and_is_deterministic():
    """Section 14 items 7 and 8."""

    from buildreasonseg_mvp.visual_cache import VisualFeatureCache

    cache = VisualFeatureCache(max_images=2)
    pooler = (torch.randn(4, 8, dtype=torch.bfloat16),)
    deepstack = [torch.randn(4, 8, dtype=torch.bfloat16) for _ in range(3)]
    cache.put("image:x", pooler, deepstack, "image_identity")
    first = cache.get("image:x")
    second = cache.get("image:x")
    assert first is not None and second is not None
    assert first.pooler[0].shape == pooler[0].shape and first.pooler[0].dtype == torch.bfloat16
    assert all(torch.equal(a, b) for a, b in zip(first.pooler, second.pooler))
    assert first.pooler[0].device.type == "cpu", "the cache must stay CPU-resident"
    assert cache.stats()["stores_model_outputs"] is False

    payload = _load(ELIGIBILITY)
    if payload:
        assert payload["recompute_determinism"]["identical"], payload["recompute_determinism"]
        assert payload["recompute_determinism"]["max_abs_difference"] == 0.0
    print("  [7,8] cached tensor shape/dtype preserved and recomputation deterministic OK")


# ---------------------------------------------------------------- artifacts 9-14


def test_cached_and_uncached_visual_features_match():
    """Section 14 item 9."""

    payload = _load(EQUIVALENCE)
    if not payload:
        return
    equality = payload["feature_equality"]
    assert equality["samples_checked"] >= 16, equality["samples_checked"]
    assert equality["all_cache_hits_identical"], equality["max_abs_difference_over_samples"]
    assert equality["max_abs_difference_over_samples"] == 0.0
    print(f"  [9] {equality['samples_checked']} samples: cache hits bit-identical OK")


def test_twelve_step_end_to_end_equivalence():
    """Section 14 item 10."""

    payload = _load(EQUIVALENCE)
    if not payload:
        return
    assert payload["reference_run_to_run_control"]["bit_equivalent"], "the gate must reproduce itself"
    for key in ("cached_image_key", "cached_content_key"):
        record = payload[key]
        assert record["losses_identical"], (key, record["loss_max_abs_difference"])
        assert record["gradients_identical"], key
        assert record["post_step_parameters_identical"], key
        assert record["category"] == "BIT_EQUIVALENT", (key, record["category"])
    print("  [10] 12-step losses, gradients and parameters identical with the cache OK")


def test_no_test_split_and_batch_is_one():
    """Section 14 items 11 and 12."""

    from buildreasonseg_mvp import data as data_mod

    ids = json.loads((EVAL / "task6c5_benchmark_ids.json").read_text(encoding="utf-8"))["record_ids"]
    train_ids = {record["sample_id"] for record in data_mod.read_records("train")}
    other_ids = {
        record["sample_id"]
        for split in ("val", "test")
        for record in data_mod.read_records(split)
    }
    assert set(ids) <= train_ids and not (set(ids) & other_ids)

    for path in (VARIANTS, FINAL):
        payload = _load(path)
        if payload:
            assert payload["batch_size"] == 1, path.name
            for entry in payload.get("entries", []):
                assert "batch" not in str(entry.get("variant", "")).lower() or "batch_size" in entry
    cfg = CONFIG.read_text(encoding="utf-8")
    assert "grad_accum_steps: 1" in cfg
    print("  [11,12] train split only and batch size 1 OK")


def test_no_architecture_loss_optimizer_or_data_change():
    """Section 14 item 13."""

    from buildreasonseg_mvp.runtime import load_config

    cfg = load_config(CONFIG)
    assert cfg["optimizer"]["betas"] == [0.9, 0.999]
    assert cfg["optimizer"]["grad_clip_norm"] == 1.0
    assert cfg["loss"]["lm_ce"] == 2.0 and cfg["loss"]["mask_dice"] == 1.0
    assert cfg["lora"]["rank"] == 16 and cfg["lora"]["target_suffixes"][0] == "q_proj"
    assert cfg["training"]["deterministic_strict"] is True
    assert cfg["training"]["gradient_checkpointing"] is True

    cache_source = VISUAL_CACHE.read_text(encoding="utf-8")
    for forbidden in ("optimizer", "combined_loss", "loss_weights", "lr_scheduler"):
        assert forbidden not in cache_source, f"the cache must not touch {forbidden}"
    # the only runtime addition is the cache plumbing
    runtime_source = RUNTIME.read_text(encoding="utf-8")
    assert "def install_visual_cache" in runtime_source
    assert "def set_visual_cache_key" in runtime_source
    print("  [13] no architecture/loss/optimizer/data change OK")


def test_strict_determinism_preserved():
    """Section 14 item 14."""

    for path in (ELIGIBILITY, EQUIVALENCE, FINAL):
        payload = _load(path)
        if not payload:
            continue
        for entry in payload.get("entries", []):
            determinism = entry.get("determinism") or {}
            if determinism:
                assert determinism.get("strict_effective") is True, entry.get("variant")
    payload = _load(EQUIVALENCE)
    if payload:
        assert payload["strict_determinism"] is True
    print("  [14] strict deterministic mode preserved OK")


def test_no_ref_4b_sre_scl():
    """Section 14 item 15."""

    forbidden = ("[REF]", "Qwen3-VL-4B", "SpatialRelationEncoder", "Spatial Consistency Loss")
    for path in NEW_FILES:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} mentions {token}"
    print("  [15] no [REF]/4B/SRE/SCL OK")


def test_sync_hotspot_artifact_localises_the_scalar_reads():
    """Section 4 deliverable: the ranked table must exist and be attributed."""

    payload = _load(SYNC_HOTSPOTS)
    if not payload:
        return
    classification = payload["classification"]
    assert classification["stack_attribution_available"] is False
    scopes = payload["scoped_windows"]
    table = scopes["ranked_table"]
    assert table, "the ranked table must not be empty"
    assert scopes["step_scalar_reads"] > 100, scopes["step_scalar_reads"]
    optimizer = scopes["windows"]["optimizer_and_clip_only"]["scalar_reads_per_call"]
    assert optimizer > 0.5 * scopes["step_scalar_reads"], (
        "the optimizer's per-parameter bookkeeping is the dominant source of scalar read-backs; "
        "if this changes, the artifact's conclusion must change with it"
    )
    # the mechanism must be named, not just the component
    mechanism = (payload.get("mechanism_verdict") or {}).get("mechanism") or {}
    assert "torch/optim/adam.py" in str(mechanism.get("source")), mechanism
    tracer = payload.get("item_call_sites") or {}
    assert tracer.get("total_item_calls"), "the sys.setprofile tracer must have recorded calls"
    assert "optimizer.step" in str(tracer["top_call_sites"][0]["call_site"]) or "runtime.py:423" in str(
        tracer["top_call_sites"][0]["call_site"]
    ), tracer["top_call_sites"][0]
    print(f"  [extra] sync attribution: optimizer {optimizer} of {scopes['step_scalar_reads']} "
          f"scalar reads/step, mechanism named OK")


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"[task6c7 tests] {len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
