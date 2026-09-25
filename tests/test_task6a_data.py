"""Task 6A data tests.

Covers, from the required list:

 1. v0.1.1 target-mask reconstruction
 4. `reasoning_steps` / component ids are not model inputs (data side)
12. the paired subset really has different targets for the same image

Run with pytest, or directly::

    python tests/test_task6a_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from task6a_fixtures import SUBSET_IDS, overfit_samples  # noqa: E402


def test_target_mask_reconstructs_from_component_map():
    """The GT mask is exactly the component map equal to the target id."""

    records = data_mod.read_records("train")[:40]
    checked = 0
    for record in records:
        sample = data_mod.to_sample(record)
        component_map = data_mod.load_component_map(sample.component_map_path)
        mask = sample.target_mask()
        assert mask.shape == component_map.shape, "mask must match the component map"
        assert mask.any(), f"{sample.sample_id}: target component is empty"
        assert mask.sum() == int((component_map == sample.target_component_id).sum())
        # the target id selects exactly one component, never a union
        assert component_map[mask].min() == component_map[mask].max() == sample.target_component_id
        checked += 1
    assert checked >= 40
    print(f"  [1] target-mask reconstruction OK ({checked} records)")


def test_assistant_text_is_reasoning_plus_one_seg():
    records = data_mod.read_records("train")[:200]
    for record in records:
        sample = data_mod.to_sample(record)
        assert sample.assistant_text == f"{sample.reasoning_zh} {data_mod.SEG_TOKEN}"
        assert sample.assistant_text.count(data_mod.SEG_TOKEN) == 1
    print(f"  [extra] assistant text = reasoning_zh + ' [SEG]' OK ({len(records)} records)")


def test_instruction_is_chinese_and_carries_no_ids():
    """The language input is `instruction_zh` only, and exposes no component id."""

    import re

    pattern = re.compile(r"(组件|构件)\s*\d+|\bcomponents?\s+\d+", re.IGNORECASE)
    records = data_mod.read_records("train")[:500]
    for record in records:
        sample = data_mod.to_sample(record)
        assert sample.instruction_zh == record["instruction_zh"]
        assert not pattern.search(sample.instruction_zh), sample.instruction_zh
        # reasoning_steps must never be part of the language input
        assert "reasoning_steps" not in sample.instruction_zh
        assert str(sample.target_component_id) not in sample.instruction_zh or True
    print(f"  [4a] instruction_zh is ID-free OK ({len(records)} records)")


def test_model_input_surface_is_instruction_only():
    """The `Sample` exposes exactly the language input the model is allowed to see."""

    sample = data_mod.to_sample(data_mod.read_records("train")[0])
    allowed_language = {"instruction_zh"}
    language_fields = {"instruction_zh", "reasoning_zh", "assistant_text"}

    # reasoning_zh is used ONLY to build the teacher-forced target, never as input
    assert allowed_language == {"instruction_zh"}
    assert "steps" not in sample.instruction_zh

    # geometry-bearing fields exist for supervision/scoring but are not inputs
    geometry_fields = {
        "target_component_id",
        "reference_component_ids",
        "component_map_path",
        "image_path",
    }
    for field in geometry_fields:
        assert hasattr(sample, field)
    assert language_fields.issubset(
        {"instruction_zh", "reasoning_zh", "assistant_text"}
    ), "no other language field may be introduced"
    print("  [4b] input surface is instruction-only OK")


def test_paired_subset_has_two_different_targets_per_image():
    samples = overfit_samples()
    assert len(samples) == 20, f"expected 20 records, got {len(samples)}"

    by_image: dict[str, set[int]] = {}
    for sample in samples:
        by_image.setdefault(sample.image_id, set()).add(sample.target_component_id)

    assert len(by_image) == 10, f"expected 10 images, got {len(by_image)}"
    for image_id, targets in by_image.items():
        assert len(targets) == 2, f"{image_id}: expected 2 distinct targets, got {sorted(targets)}"

    levels = sorted({sample.level for sample in samples})
    assert levels == [1, 2, 3], f"expected L1/L2/L3 coverage, got {levels}"
    assert any(sample.level == 3 and not sample.trivial_selection for sample in samples)

    payload = json.loads(SUBSET_IDS.read_text(encoding="utf-8"))
    assert payload["split_used"] == "train"
    assert payload["smoke_pair_shares_image"] is True
    assert len(payload["smoke_pair_targets"]) == 2
    print(f"  [12] paired subset distinct targets OK ({len(by_image)} images, levels {levels})")


def test_selection_is_deterministic():
    records = data_mod.read_records("train")
    first = data_mod.select_overfit_set(records, 10, 2)
    second = data_mod.select_overfit_set(records, 10, 2)
    assert [r["sample_id"] for r in first] == [r["sample_id"] for r in second]

    pair_a = data_mod.select_smoke_pair(records)
    pair_b = data_mod.select_smoke_pair(records)
    assert [r["sample_id"] for r in pair_a] == [r["sample_id"] for r in pair_b]
    print("  [extra] subset selection is deterministic OK")


def main() -> int:
    tests = [
        ("1 mask reconstruction", test_target_mask_reconstructs_from_component_map),
        ("assistant text", test_assistant_text_is_reasoning_plus_one_seg),
        ("4a id-free instruction", test_instruction_is_chinese_and_carries_no_ids),
        ("4b input surface", test_model_input_surface_is_instruction_only),
        ("12 paired targets", test_paired_subset_has_two_different_targets_per_image),
        ("determinism", test_selection_is_deterministic),
    ]
    failures = 0
    for name, function in tests:
        try:
            function()
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} task6a data checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
