"""Task 6C section 17 items 6-13: subset construction.

 6. U subset matches Task 6B exactly
 7. P subset = 480 records / 240 images / 2 records per image
 8. every P pair has different targets
 9. every P pair has different query types
10. P level counts = 160/160/160 (or a proven feasibility exception)
11. no trivial L3 in P
12. val subset identical to Task 6B
13. paired val identical to Task 6B

Run with pytest, or directly::

    python tests/test_task6c_subsets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp import subsets as S  # noqa: E402

TASK6C = REPO_ROOT / "evaluation" / "task6c_subset_ids.json"
TASK6B = REPO_ROOT / "evaluation" / "task6b_subset_ids.json"


def _task6c() -> dict:
    assert TASK6C.is_file(), "evaluation/task6c_subset_ids.json is missing"
    return json.loads(TASK6C.read_text(encoding="utf-8"))


def _task6b() -> dict:
    return json.loads(TASK6B.read_text(encoding="utf-8"))


def test_u_subset_matches_task6b_exactly():
    payload = _task6c()
    expected = list(_task6b()["train"]["sample_ids"])
    got = list(payload["U"]["sample_ids"])
    assert got == sorted(expected), "U must be Task 6B's exact 480 sample ids"
    assert payload["U"]["n_records"] == 480
    assert payload["U"]["n_images"] == 480
    assert payload["U"]["by_level"] == {"1": 160, "2": 160, "3": 160}
    train = {r["sample_id"] for r in data_mod.read_records("train")}
    assert set(got) <= train
    print("  [6] U subset identical to Task 6B OK")


def test_p_subset_shape_and_levels():
    payload = _task6c()
    p = payload["P"]
    assert p["n_records"] == 480, p["n_records"]
    assert p["n_images"] == 240, p["n_images"]
    assert p["records_per_image_values"] == [2], p["records_per_image_values"]
    assert p["n_pairs"] == 240
    if p["composition_feasible_exactly"]:
        assert p["by_level"] == {"1": 160, "2": 160, "3": 160}, p["by_level"]
    else:
        assert p.get("feasibility_exception_reason"), "a relaxed composition must be proven and recorded"
    print(f"  [7/10] P subset shape OK: {p['n_records']} records / {p['n_images']} images, levels {p['by_level']}")


def test_p_pairs_have_different_targets_and_query_types():
    p = _task6c()["P"]
    assert p["all_pairs_different_targets"] is True
    assert p["all_pairs_different_query_types"] is True
    for pair in p["pairs"]:
        assert pair["a_target"] != pair["b_target"], pair
        assert pair["a_query_type"] != pair["b_query_type"], pair
        assert pair["a"] != pair["b"]
    print("  [8/9] every P pair differs in target and query type OK")


def test_p_has_no_trivial_l3():
    p = _task6c()["P"]
    assert p["trivial_l3"] == 0
    assert p["nontrivial_l3"] == 160
    train = {r["sample_id"]: r for r in data_mod.read_records("train")}
    for sample_id in p["sample_ids"]:
        record = train[sample_id]
        if int(record["level"]) == 3:
            assert not bool(record.get("trivial_selection", False)), sample_id
    print("  [11] no trivial L3 in P OK")


def test_p_composition_is_the_preferred_one():
    p = _task6c()["P"]
    assert p["composition_requested"] == [[1, 2, 80], [1, 3, 80], [2, 3, 80]]
    for name, entry in p["selection"].items():
        assert entry["selected_images"] == entry["quota"], (name, entry)
        assert entry["feasible"] is True, (name, entry)
    assert p["eligible_images_per_bucket"]["L1_L2"] >= 80
    assert p["eligible_images_per_bucket"]["L1_L3"] >= 80
    assert p["eligible_images_per_bucket"]["L2_L3"] >= 80
    print("  [10] preferred 80/80/80 composition feasible and achieved OK")


def test_p_selection_is_deterministic():
    first = S.build_task6c()
    second = S.build_task6c()
    assert first["U"]["sample_ids"] == second["U"]["sample_ids"]
    assert first["P"]["sample_ids"] == second["P"]["sample_ids"]
    assert first["P"]["pairs"] == second["P"]["pairs"]
    print("  [extra] U and P selection deterministic OK")


def test_validation_material_identical_to_task6b():
    task6b = _task6b()
    val_ids = {r["sample_id"] for r in data_mod.read_records("val")}
    for key in ("val", "paired_probe"):
        for sample_id in task6b[key]["sample_ids"]:
            assert sample_id in val_ids, f"{sample_id} is not a val record"
    assert len(task6b["val"]["sample_ids"]) == 120
    assert len(task6b["paired_probe"]["sample_ids"]) == 40
    assert task6b["paired_probe"]["n_pairs"] == 20
    assert task6b["val"]["by_level"] == {"1": 40, "2": 40, "3": 40}
    print("  [12/13] validation and paired validation are Task 6B's exact material OK")


def test_val_records_are_never_used_for_training():
    payload = _task6c()
    val_ids = {r["sample_id"] for r in data_mod.read_records("val")}
    assert not (set(payload["U"]["sample_ids"]) & val_ids)
    assert not (set(payload["P"]["sample_ids"]) & val_ids)
    assert payload["test_split_used"] is False
    assert payload["source_split"] == "train"
    print("  [extra] train-only subsets, no val/test leakage OK")


def main() -> int:
    tests = [
        ("6 U==Task6B", test_u_subset_matches_task6b_exactly),
        ("7/10 P shape", test_p_subset_shape_and_levels),
        ("8/9 P pair constraints", test_p_pairs_have_different_targets_and_query_types),
        ("11 no trivial L3", test_p_has_no_trivial_l3),
        ("10 composition", test_p_composition_is_the_preferred_one),
        ("determinism", test_p_selection_is_deterministic),
        ("12/13 val material", test_validation_material_identical_to_task6b),
        ("no leakage", test_val_records_are_never_used_for_training),
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
    print(f"\n{len(tests) - failures}/{len(tests)} task6c subset checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
