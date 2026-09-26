"""Task 6B subset and language-target tests.

Covers, from the Task 6B section 21 list:

 3. subset determinism
 4. exactly 480 train / 120 val
 5. val comes from val only
 6. no test data in tuning
 7. the assistant target ends `[SEG]` then EOS

Run with pytest, or directly::

    python tests/test_task6b_subsets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp import language_metrics as LM  # noqa: E402
from buildreasonseg_mvp import subsets  # noqa: E402

SUBSET_JSON = REPO_ROOT / "evaluation" / "task6b_subset_ids.json"


def _payload() -> dict:
    return json.loads(SUBSET_JSON.read_text(encoding="utf-8"))


def test_subsets_have_the_required_sizes():
    payload = _payload()
    assert payload["sizes"]["train"] == 480, payload["sizes"]
    assert payload["sizes"]["val"] == 120, payload["sizes"]
    assert payload["sizes"]["paired"] == 40
    assert payload["train"]["by_level"] == {"1": 160, "2": 160, "3": 160}
    assert payload["val"]["by_level"] == {"1": 40, "2": 40, "3": 40}
    assert payload["paired_probe"]["n_pairs"] == 20
    print(f"  [4] subset sizes OK: {payload['sizes']}")


def test_no_trivial_level3_and_no_test_data():
    payload = _payload()
    for key in ("train", "val", "paired_probe"):
        assert payload[key]["trivial_l3"] == 0, f"{key} contains trivial L3 records"
    assert payload["test_split_used"] is False
    assert payload["train"]["source_split"] == "train"
    assert payload["val"]["source_split"] == "val"
    assert payload["paired_probe"]["source_split"] == "val"
    for key in ("train", "val", "paired_probe"):
        assert "test" not in payload[key]["source_split"]
    print("  [5/6] val-only, no trivial L3, no test data OK")


def test_val_ids_really_come_from_the_val_split():
    payload = _payload()
    val_ids = {r["sample_id"] for r in data_mod.read_records("val")}
    train_ids = {r["sample_id"] for r in data_mod.read_records("train")}
    for key in ("val", "paired_probe"):
        for sample_id in payload[key]["sample_ids"]:
            assert sample_id in val_ids, f"{sample_id} is not a val record"
            assert sample_id not in train_ids
    for sample_id in payload["train"]["sample_ids"]:
        assert sample_id in train_ids, f"{sample_id} is not a train record"
        assert sample_id not in val_ids
    print("  [5] train/val provenance OK")


def test_selection_is_deterministic():
    first = subsets.build_all()
    second = subsets.build_all()
    for key in ("train", "val", "paired_probe"):
        assert first[key]["sample_ids"] == second[key]["sample_ids"], key
    print("  [3] subset selection deterministic OK")


def test_paired_probe_pairs_are_distinct_targets_and_query_types():
    payload = _payload()
    records = {r["sample_id"]: r for r in data_mod.read_records("val")}
    different_types = 0
    for pair in payload["paired_probe"]["pairs"]:
        a, b = records[pair["a"]], records[pair["b"]]
        assert a["image_id"] == b["image_id"]
        assert int(a["target_component_id"]) != int(b["target_component_id"])
        different_types += int(a["query_type"] != b["query_type"])
    assert different_types >= 15, f"only {different_types}/20 pairs have different query types"
    print(f"  [extra] paired probe OK ({different_types}/20 with different query types)")


def test_assistant_target_ends_with_seg_then_eos():
    """Build a real teacher-forcing batch and inspect the tail of the target."""

    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from task6a_fixtures import require_model_assets, runtime

    require_model_assets()
    rt = runtime()
    payload = _payload()
    records = {r["sample_id"]: r for r in data_mod.read_records("val")}
    sample = data_mod.to_sample(records[payload["val"]["sample_ids"][0]])
    batch = rt.prepare(sample)[0]
    ids = batch.input_ids[0].tolist()
    eos = rt.tokenizer.eos_token_id
    seg = rt.model.seg_token_id

    assert ids[-1] == eos, "the assistant target must end with EOS"
    assert ids.count(seg) == 1, "exactly one [SEG] must be present"
    assert ids.index(seg) == len(ids) - 2, "[SEG] must immediately precede EOS"
    # `labels` is PRE-SHIFTED: labels[i] holds input_ids[i + 1], because
    # lm_logits[i] is the distribution over token i + 1. The supervised span is
    # therefore [prompt_length - 1, total_length - 2] and the final position is
    # not supervised at all.
    labels = batch.labels[0].tolist()
    assert labels[-1] == -100, "the final label position is not supervised (pre-shifted labels)"
    assert labels[-2] == eos, "the label that predicts EOS must be supervised"
    assert labels[-3] == seg, "the label that predicts [SEG] must be supervised"
    assert [value for value in labels if value != -100][-2:] == [seg, eos]
    print("  [7] assistant target ends [SEG] then EOS OK")


def test_language_metric_definitions():
    """The metrics must be deterministic template comparisons, no LLM judge."""

    record = data_mod.read_records("val")[0]
    assert LM.exact_match(record["reasoning_zh"] + " [SEG]", record["reasoning_zh"]) is True
    assert LM.exact_match("完全不同的文本 [SEG]", record["reasoning_zh"]) is False
    assert LM.reasoning_part("abc [SEG] def") == "abc "
    assert LM.count_seg("[SEG] x [SEG]") == 2
    assert 0.0 <= LM.char_similarity("abc", "abd") <= 1.0

    lookup = LM.build_reasoning_lookup([record])
    assert LM.operation_chain_correct(
        record["reasoning_zh"] + " [SEG]", record["query_type"], lookup
    )
    assert not LM.operation_chain_correct("无关文本 [SEG]", record["query_type"], lookup)
    print("  [extra] language metric definitions OK")


def main() -> int:
    tests = [
        ("4 sizes", test_subsets_have_the_required_sizes),
        ("5/6 provenance", test_no_trivial_level3_and_no_test_data),
        ("5 val ids", test_val_ids_really_come_from_the_val_split),
        ("3 determinism", test_selection_is_deterministic),
        ("paired probe", test_paired_probe_pairs_are_distinct_targets_and_query_types),
        ("7 seg then eos", test_assistant_target_ends_with_seg_then_eos),
        ("language metrics", test_language_metric_definitions),
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
    print(f"\n{len(tests) - failures}/{len(tests)} task6b subset checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
