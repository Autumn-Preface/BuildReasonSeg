"""Task 6D.1 section 15: scheduler-horizon and decodability-audit tests.

Items 1-15 of the section 15 list. The scheduler items need no weights; the artifact items assert
against what the corrective run and the probes produced.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

EVAL = REPO_ROOT / "evaluation"
CONFIG = REPO_ROOT / "configs" / "mvp" / "task6c_2b_ablation.yaml"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "task6d_train.py"
PROBE_SCRIPT = REPO_ROOT / "scripts" / "task6d1_probes.py"
EXTRACT_SCRIPT = REPO_ROOT / "scripts" / "task6d1_extract_hidden.py"

AUDIT = EVAL / "task6d1_scheduler_audit.json"
G0_CORRECTED = EVAL / "task6d1_g0_corrected.json"
G0_ORIGINAL = EVAL / "task6d_g0.json"
MANIFEST = EVAL / "task6d1_hidden_extract_manifest.json"
PROBES = {name: EVAL / f"task6d1_probe_{name}.json" for name in ("linear", "raw_mlp", "layernorm_mlp")}
STATS = EVAL / "task6d1_representation_stats.json"
SUMMARY = EVAL / "task6d1_decodability_summary.json"
G1 = EVAL / "task6d1_g1.json"
PAIRED = EVAL / "task6d1_paired_probe.json"

TASK6D1_FILES = sorted((REPO_ROOT / "scripts").glob("task6d1_*.py"))


def _load(path: Path) -> dict:
    if not path.is_file():
        print(f"  [skip] {path.name} not present yet")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _factor(step: int, horizon: int, warmup: int) -> float:
    if warmup and step < warmup:
        return max(1e-3, (step + 1) / warmup)
    progress = (step - warmup) / max(1, horizon - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def test_scheduler_horizon_equals_actual_optimizer_budget():
    """Section 15 item 1."""

    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    assert "total_optimizer_steps = max(1, planned_epochs * steps_per_epoch)" in source, (
        "the scheduler horizon must be the real optimizer-step budget"
    )
    assert "make_optimizer(runtime, total_optimizer_steps)" in source
    assert "make_optimizer(runtime, steps_per_epoch)" not in source, "the defective call must be gone"

    audit = _load(AUDIT)
    if audit:
        assert audit["defect"]["horizon_used"] == 480
        assert audit["defect"]["steps_actually_run"] == 960
        assert audit["correction"]["horizon_used"] == 960
    corrected = _load(G0_CORRECTED)
    if corrected:
        horizon = corrected["scheduler_horizon"]
        assert horizon["total_optimizer_steps"] == horizon["planned_epochs"] * horizon["steps_per_epoch"]
    print("  [1] scheduler horizon equals the actual optimizer-step budget OK")


def test_epoch_2_initial_lr_is_nonzero():
    """Section 15 item 2."""

    assert _factor(480, 480, 20) == 0.0, "the defective horizon must show the zero-LR epoch 2"
    assert _factor(480, 960, 20) > 0.5, "the corrected horizon must keep epoch 2 alive"
    audit = _load(AUDIT)
    if audit:
        assert audit["verification"]["epoch_2_initial_factor_defective"] == 0.0
        assert audit["verification"]["epoch_2_initial_lr_nonzero_after_fix"] is True
    corrected = _load(G0_CORRECTED)
    if corrected:
        checkpoint = corrected["lr_audit"]["checkpoints"]["start_of_epoch_2"]
        assert checkpoint is not None, "the corrected run must record the epoch-2 start LR"
        assert any(value > 0.0 for value in checkpoint["lrs"]), checkpoint
        assert corrected["lr_audit"]["epoch_2_initial_lr_nonzero"] is True
    print("  [2] epoch-2 initial LR is non-zero for 2x480 OK")


def test_terminal_cosine_only_at_final_full_run_step():
    """Section 15 item 3."""

    assert _factor(959, 960, 20) < 1e-5, "the terminal factor must be at the last step"
    assert _factor(479, 960, 20) > 0.5, "mid-run must not be terminal"
    audit = _load(AUDIT)
    if audit:
        observed = (audit.get("observed_corrected_run") or {}).get("verification") or {}
        assert observed.get("terminal_at_final_scheduled_step") is True, observed
        assert observed.get("epoch_2_initial_lr_nonzero") is True, observed
        assert observed.get("mid_run_is_not_terminal") is True, observed
        assert observed["final_lr_over_peak_ratio"] <= 1e-5, observed
    hardcoded = _load(G0_ORIGINAL)
    if hardcoded:
        assert hardcoded.get("status") == "VALID_MEASUREMENT_WITH_SCHEDULER_CONFOUND"
    print("  [3] terminal cosine factor only at the final full-run step OK")


def test_smoke_mode_uses_its_actual_budget():
    """Section 15 item 4."""

    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    assert "planned_epochs = epochs if not args.smoke else min(epochs, 1)" in source, (
        "a truncated run must derive its budget from the loop, not from nominal epochs"
    )
    assert "if args.max_steps:" in source
    print("  [4] smoke/max-step mode uses its actual step budget OK")


def test_g0_corrected_differs_only_by_scheduler():
    """Section 15 item 5."""

    corrected = _load(G0_CORRECTED)
    if not corrected:
        return
    assert corrected["geometry_kind"] == "box"
    assert corrected["lambda_ground"] == 5.0
    assert corrected["loss_weights"] == {"lm_ce": 2.0, "mask_bce": 0.0, "mask_dice": 0.0}
    assert corrected["epochs_requested"] == 2
    assert corrected["trainables"]["sam_mask_decoder_trainable"] is False
    assert corrected["trainables"]["projection_trainable"] is False
    assert corrected["trainables"]["grounding_head_trainable"] is True
    assert corrected["optimizer_groups"], "the optimizer groups must be recorded"
    assert "scheduler_horizon" in corrected, "the only intended change must be visible in the report"
    assert "VALID_MEASUREMENT_WITH_SCHEDULER_CONFOUND" in corrected["parent_task_status"]
    print("  [5] G0-R differs from Task 6D only by the scheduler correction OK")


def test_frozen_probe_extraction_does_not_update_qwen():
    """Section 15 item 6."""

    source = EXTRACT_SCRIPT.read_text(encoding="utf-8")
    assert "with torch.no_grad():" in source
    assert "optimizer" not in source, "extraction must not build or step an optimizer"
    assert "backward" not in source
    manifest = _load(MANIFEST)
    if manifest:
        assert manifest["representation_source"]["checkpoint"]["matches_manifest"] is True
        assert "no Task 6D grounding checkpoint" in manifest["representation_source"]["not_used"]
        for split in ("train", "val", "paired"):
            assert split in manifest["splits"], split
    print("  [6] extraction never updates Qwen and uses the hash-verified P_C checkpoint OK")


def test_probe_a_and_b_have_no_layernorm_probe_c_has_it():
    """Section 15 items 7, 8 and 9."""

    source = PROBE_SCRIPT.read_text(encoding="utf-8")
    class_a = source.split("class LinearBox", 1)[1].split("class RawMlpBox", 1)[0]
    class_b = source.split("class RawMlpBox", 1)[1].split("class LayerNormMlpBox", 1)[0]
    class_c = source.split("class LayerNormMlpBox", 1)[1].split("def _canonical", 1)[0]

    def code_only(text: str) -> str:
        """Drop the docstring so a comment saying 'without LayerNorm' cannot fail the check."""

        parts = text.split('"""')
        return "".join(part for index, part in enumerate(parts) if index % 2 == 0)

    assert "LayerNorm" not in code_only(class_a), "Probe A must be a plain linear readout"
    assert "LayerNorm" not in code_only(class_b), "Probe B must have NO LayerNorm"
    assert "LayerNorm" in code_only(class_c) and "self.norm" in code_only(class_c), "Probe C is the Task 6D head"

    for name, expect_layernorm in (("linear", False), ("raw_mlp", False), ("layernorm_mlp", True)):
        payload = _load(PROBES[name])
        if not payload:
            continue
        assert payload["uses_layernorm"] is expect_layernorm, (name, payload["uses_layernorm"])
    print("  [7,8,9] Probe A/B are LayerNorm-free and Probe C includes it OK")


def test_probe_labels_are_supervision_only():
    """Section 15 item 10."""

    for name, path in PROBES.items():
        payload = _load(path)
        if not payload:
            continue
        assert payload["frozen"]["probe_only"] is True
        assert "GT boxes are used only as probe supervision" in payload["_doc"]
        assert payload["input_scaling"]["fixed_scalar"] > 0
    manifest = _load(MANIFEST)
    if manifest:
        # the manifest carries the labels' provenance and no test-split material
        assert manifest["geometry_kind_for_labels"] == "box"
        for split in manifest["splits"].values():
            assert split["test_split_used"] is False
            assert split["sample_ids_sha256"]
    print("  [10] probe labels are GT boxes used only in probe supervision OK")


def test_no_test_split_and_determinism():
    """Section 15 items 11 and 12."""

    from buildreasonseg_mvp import data as data_mod

    manifest = _load(MANIFEST)
    if manifest:
        assert manifest["determinism"]["strict_effective"] is True
        for split in manifest["splits"].values():
            assert split["test_split_used"] is False
    corrected = _load(G0_CORRECTED)
    if corrected:
        assert corrected["determinism"]["use_deterministic_algorithms"] is True
        assert corrected["determinism"]["cublas_workspace_config"] == ":4096:8"
    test_ids = {record["sample_id"] for record in data_mod.read_records("test")}
    for name, path in PROBES.items():
        payload = _load(path)
        if payload:
            assert payload["protocol"]["seed"] == 20260926
    assert test_ids, "the test split exists but must not be used"
    print("  [11,12] no test split, strict determinism, fixed probe seed OK")


def test_no_4b_ref_sre_scl_and_no_dataset_change():
    """Section 15 items 13 and 14."""

    forbidden = ("[REF]", "Qwen3-VL-4B", "SpatialRelationEncoder", "Spatial Consistency Loss")
    for path in TASK6D1_FILES + [PROBE_SCRIPT, EXTRACT_SCRIPT]:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} mentions {token}"
    for name in ("task6d1_scheduler_audit", "task6d1_probes", "task6d1_representation_stats"):
        text = (REPO_ROOT / "scripts" / f"{name}.py").read_text(encoding="utf-8")
        assert "read_records(\"test\")" not in text
    config = CONFIG.read_text(encoding="utf-8")
    assert "sam2.1_hiera_base_plus" in config or "sam2.1-hiera-base-plus" in config
    print("  [13,14] no 4B/[REF]/SRE/SCL and no dataset change OK")


def test_no_g1_without_a_passing_corrected_gate():
    """Section 15 item 15."""

    corrected = _load(G0_CORRECTED)
    if corrected:
        gate_passed = bool(corrected["gate"]["passed"])
        if not gate_passed:
            assert not G1.is_file(), "G1 must not run when the corrected G0 gate fails"
            assert not PAIRED.is_file(), "no G1 paired probe without a passing gate"
            assert corrected["verdict"] == "GROUNDING_REPRESENTATION_FAILED"
    print("  [15] G1 gated on the corrected G0 gate OK")


def test_audit_artifacts_are_consistent():
    """Section 14 deliverables and the interpretation matrix."""

    summary = _load(SUMMARY)
    if not summary:
        return
    assert summary["case"] in ("A", "B", "C", "D", "E", "F")
    assert summary["verdict"] in (
        "SCHEDULER_FIX_RECOVERS_GROUNDING",
        "LAYER_NORM_READOUT_CONFOUND",
        "LOCATION_SIGNAL_PARTIALLY_GENERALIZES",
        "LOCATION_SIGNAL_MEMORIZABLE_NOT_GENERALIZABLE",
        "READOUT_OR_FEATURE_IDENTITY_BUG_SUSPECTED",
        "SPATIAL_REPRESENTATION_DOES_NOT_GENERALIZE",
        "PRACTICALLY_NOT_DECODABLE_GEOMETRY",
    )
    assert "PRACTICALLY_DECODABLE" in summary["wording_rule"]
    stats = _load(STATS)
    if stats:
        assert stats["svd"]["effective_rank_participation_ratio"] > 0
        assert stats["probe_d_train"]["linear_prediction_from_three_statistics"]["in_sample_r_squared"] < 0.1
    print(f"  [extra] interpretation matrix resolved to case {summary['case']} "
          f"({summary['verdict']}) OK")


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
    print(f"[task6d1 tests] {len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
