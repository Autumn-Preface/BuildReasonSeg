"""Task 6O tests (section 17): 30 checks on the causal decomposition and frozen-asset integrity."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "spatial_reasoning"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

EVAL = REPO_ROOT / "evaluation"
SCRIPTS = REPO_ROOT / "scripts"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6n" / "packs"

TASK6O_SOURCES = ("task6o_train.py", "task6o_evaluate.py", "task6o_report.py")
TASK6N_SOURCES = ("task6n_freeze_packs.py", "task6n_field_sanity.py", "task6n_train.py",
                  "task6n_evaluate.py", "task6n_report.py")
ALLOWED_PROGRAMS = {
    "largest_to_left_of", "largest_to_right_of", "largest_to_above", "largest_to_below",
    "smallest_to_left_of", "smallest_to_right_of", "smallest_to_above", "smallest_to_below",
}
BASE_COMMIT = "90f3735cf57115cd67e15b1b3457752e1fb049a6"


def _artifact(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_only(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    for quote in ('"""', "'''"):
        if stripped.startswith(quote):
            end = stripped.find(quote, len(quote))
            if end != -1:
                stripped = stripped[end + len(quote):]
            break
    return "\n".join(line.split("#", 1)[0] for line in stripped.splitlines())


def _random_batch(variant_config=None):
    from buildreasonseg_mvp.task6n_relation_decoder import DecoderConfig

    config = variant_config or DecoderConfig()
    return (
        torch.randn(1, config.visual_channels, 8, 8),
        torch.zeros(1, dtype=torch.long),
        torch.rand(1, 1, 8, 8),
        torch.rand(1, 1, 8, 8),
    )


# ---------------------------------------------------------------- 1-5 frozen assets and protocol


def test_task6n_artifacts_unchanged():
    for name in ("task6n_verdict.json", "task6n_ablation_summary.json", "task6n_mini_val.json",
                 "task6n_paired_val.json", "task6n_overfit20.json", "task6n_field_sanity.json",
                 "task6n_pack_manifest.json"):
        assert (EVAL / name).is_file(), name
    verdict = _artifact("task6n_verdict.json")
    assert verdict["verdict"] == "GEOMETRIC_RELATION_FIELD_FEASIBLE"
    changed = subprocess.run(
        ["git", "diff", "--name-only", BASE_COMMIT, "--", "evaluation/task6n_",
         "buildreasonseg_mvp/geometric_relation_field.py", "configs/spatial_relations_v1.yaml"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    # only the two added Task 6O files inside evaluation/ may appear from the Task 6N prefix
    violations = [line for line in changed.splitlines() if line and "task6o" not in line]
    assert not violations, f"frozen Task 6N assets changed: {violations}"


def test_task6n_pack_hashes_exact():
    manifest = _artifact("task6n_pack_manifest.json")
    if manifest is None:
        pytest.skip("Task 6N pack manifest missing")
    for name, entry in manifest["packs"].items():
        path = Path(entry["path"])
        if not path.is_file():
            pytest.skip("packs are regenerable artifacts under gitignored artifacts/")
        assert _sha256(path) == entry["sha256"], f"{name} hash mismatch"
    reproduction = _artifact("task6o_b2_reproduction.json")
    if reproduction is not None:
        assert all(entry["matches"] for entry in reproduction["pack_verification"]["packs"].values())
        assert reproduction["pack_verification"]["packs"]["overfit20"]["matches"] is True


def test_task6n_b2_checkpoint_hash_matches():
    frozen = _artifact("task6n_mini_val.json")
    if frozen is None:
        pytest.skip("Task 6N training artifact missing")
    checkpoint = frozen["variants"]["B2"]["checkpoint"]
    path = Path(checkpoint["path"])
    if not path.is_file():
        pytest.skip("checkpoints live under gitignored artifacts/")
    assert _sha256(path) == checkpoint["sha256"]
    reproduction = _artifact("task6o_b2_reproduction.json")
    if reproduction is not None:
        assert reproduction["checkpoint"]["matches"] is True
        assert reproduction["checkpoint"]["actual_sha256"] == checkpoint["sha256"]
        assert reproduction["b2_retrained"] is False


def test_b2_reproduction_within_tolerance():
    reproduction = _artifact("task6o_b2_reproduction.json")
    if reproduction is None:
        pytest.skip("B2 reproduction not run yet")
    assert reproduction["verdict"] == "B2_REPRODUCED"
    assert reproduction["tolerance"] == 1e-6
    for name, entry in reproduction["comparisons"].items():
        if "within_tolerance" in entry:
            assert entry["within_tolerance"], name
            assert entry["abs_delta"] <= 1e-6, name
        else:
            assert entry["exact"], name
    frozen = _artifact("task6n_mini_val.json")
    assert abs(reproduction["reproduced"]["mini_val"]["miou"]
               - frozen["variants"]["B2"]["selected_model"]["miou"]) <= 1e-6


def test_no_test_split_access():
    paths = [SCRIPTS / name for name in TASK6O_SOURCES] + [SCRIPTS / name for name in TASK6N_SOURCES]
    paths += [REPO_ROOT / "buildreasonseg_mvp" / "task6n_relation_decoder.py"]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for marker in ('"test.jsonl"', "'test.jsonl'", 'split="test"', "split('test')",
                       "task6o_test", "test_fixed120", "paired_test"):
            assert marker not in text, f"{path.name} must not touch the test split ({marker})"
    for name in ("task6o_b2_reproduction.json", "task6o_overfit20.json", "task6o_mini_val.json",
                 "task6o_paired_val.json", "task6o_causal_summary.json", "task6o_verdict.json"):
        payload = _artifact(name)
        if payload is not None:
            assert payload.get("test_split_used", False) is False


# ---------------------------------------------------------------- 6-7 scope


def test_only_eight_directional_programs():
    from buildreasonseg_mvp.geometric_relation_field import DIRECTIONAL_PROGRAMS

    assert set(DIRECTIONAL_PROGRAMS) == ALLOWED_PROGRAMS
    for name in TASK6O_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        assert "nearest" not in text.split('"""')[2] if text.count('"""') >= 2 else True
    summary = _artifact("task6o_causal_summary.json")
    if summary is not None:
        assert set(summary["criteria"]) == {"14.1", "14.2", "14.3"}


def test_target_never_model_input():
    decoder = _code_only(REPO_ROOT / "buildreasonseg_mvp" / "task6n_relation_decoder.py")
    forward = decoder.split("def forward(")[1].split("def parameter_report")[0]
    assert "target" not in forward
    for name in TASK6O_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        assert "task6n_loss(upsampled_logits(logits, batch.target_size), batch.target)" in text \
            or "task6n_loss" not in text or "variant_forward" in text, name
    trainer = (SCRIPTS / "task6o_train.py").read_text(encoding="utf-8")
    assert "model(None, batch.relation_index, None, batch.field)" in trainer
    assert "logits = variant_forward(model, batch)" in trainer


# ---------------------------------------------------------------- 8-13 B3/B4 input contracts


def test_b3_receives_visual_field_and_relation_only():
    from buildreasonseg_mvp.task6n_relation_decoder import (
        DecoderConfig, RelationMaskDecoder, VARIANT_USES_FIELD, VARIANT_USES_REFERENCE,
        VARIANT_USES_VISUAL, first_conv_in_channels,
    )

    assert VARIANT_USES_VISUAL["B3"] is True
    assert VARIANT_USES_FIELD["B3"] is True
    assert VARIANT_USES_REFERENCE["B3"] is False
    config = DecoderConfig()
    assert first_conv_in_channels("B3", config) == 145
    torch.manual_seed(0)
    model = RelationMaskDecoder("B3", config).eval()
    visual, relation, reference, field = _random_batch(config)
    with pytest.raises(ValueError):
        model(visual, relation)  # the field is required
    with torch.no_grad():
        logits = model(visual, relation, None, field)
    assert logits.shape == (1, 1, 8, 8)


def test_b3_does_not_receive_direct_reference_mask():
    from buildreasonseg_mvp.task6n_relation_decoder import DecoderConfig, RelationMaskDecoder

    config = DecoderConfig()
    torch.manual_seed(0)
    model = RelationMaskDecoder("B3", config).eval()
    visual, relation, reference, field = _random_batch(config)
    with torch.no_grad():
        without_reference = model(visual, relation, None, field)
        with_reference = model(visual, relation, reference, field)
    # the reference mask must be ignored entirely by B3
    assert torch.allclose(without_reference, with_reference, atol=1e-6)
    summary = _artifact("task6o_causal_summary.json")
    if summary is not None:
        assert summary["variants"]["B3"]["first_conv_in_channels"] == 145


def test_b4_receives_field_and_relation_only():
    from buildreasonseg_mvp.task6n_relation_decoder import (
        DecoderConfig, RelationMaskDecoder, VARIANT_USES_FIELD, VARIANT_USES_REFERENCE,
        VARIANT_USES_VISUAL, first_conv_in_channels,
    )

    assert VARIANT_USES_VISUAL["B4"] is False
    assert VARIANT_USES_FIELD["B4"] is False
    assert VARIANT_USES_REFERENCE["B4"] is False
    config = DecoderConfig()
    assert first_conv_in_channels("B4", config) == 144
    torch.manual_seed(0)
    model = RelationMaskDecoder("B4", config).eval()
    visual, relation, reference, field = _random_batch(config)
    with pytest.raises(ValueError):
        model(None, relation)  # the field is required
    with torch.no_grad():
        logits = model(None, relation, None, field)
    assert logits.shape == (1, 1, 8, 8)
    summary = _artifact("task6o_causal_summary.json")
    if summary is not None:
        assert summary["variants"]["B4"]["first_conv_in_channels"] == 144


def test_b4_does_not_receive_visual():
    from buildreasonseg_mvp.task6n_relation_decoder import DecoderConfig, RelationMaskDecoder

    config = DecoderConfig()
    torch.manual_seed(0)
    model = RelationMaskDecoder("B4", config).eval()
    visual, relation, reference, field = _random_batch(config)
    with torch.no_grad():
        without_visual = model(None, relation, None, field)
        with_visual = model(visual, relation, None, field)
    assert torch.allclose(without_visual, with_visual, atol=1e-6), "B4 must ignore the RGB feature"
    assert not hasattr(model, "project"), "B4 must not even own a visual projection"
    trainer = (SCRIPTS / "task6o_train.py").read_text(encoding="utf-8")
    assert "model(None, batch.relation_index, None, batch.field)" in trainer


def test_b4_does_not_receive_direct_reference_mask():
    from buildreasonseg_mvp.task6n_relation_decoder import DecoderConfig, RelationMaskDecoder

    config = DecoderConfig()
    torch.manual_seed(0)
    model = RelationMaskDecoder("B4", config).eval()
    visual, relation, reference, field = _random_batch(config)
    with torch.no_grad():
        without_reference = model(None, relation, None, field)
        with_reference = model(None, relation, reference, field)
    assert torch.allclose(without_reference, with_reference, atol=1e-6)


def test_b4_field_projection_is_one_to_128():
    from buildreasonseg_mvp.task6n_relation_decoder import DecoderConfig, RelationMaskDecoder

    config = DecoderConfig()
    model = RelationMaskDecoder("B4", config)
    first = model.field_project[0]
    assert isinstance(first, torch.nn.Conv2d)
    assert first.in_channels == 1, "B4 section 9 requires Conv1x1(1 -> 128)"
    assert first.out_channels == 128
    assert isinstance(model.field_project[1], torch.nn.GroupNorm)
    assert model.field_project[1].num_channels == 128
    trunk_first = model.trunk[0]
    assert trunk_first.in_channels == 144
    report = model.parameter_report()
    assert report["total_parameters"] == 240833


# ---------------------------------------------------------------- 14-20 frozen definitions/protocol


def test_geometric_relation_field_unchanged():
    field_path = REPO_ROOT / "buildreasonseg_mvp" / "geometric_relation_field.py"
    diff = subprocess.run(
        ["git", "diff", "--name-only", BASE_COMMIT, "--", "buildreasonseg_mvp/geometric_relation_field.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert diff == "", "the GeometricRelationField formula must not change in Task 6O"
    # the session-13 N0 clarification is documented, not "fixed"
    from buildreasonseg_mvp.geometric_relation_field import PROGRAM_TO_RELATION

    assert set(PROGRAM_TO_RELATION) == ALLOWED_PROGRAMS


def test_alpha_tau_softness_unchanged():
    from buildreasonseg_mvp.geometric_relation_field import DEFAULT_FIELD_CONFIG

    config = DEFAULT_FIELD_CONFIG
    assert config.alpha == 1.2
    assert config.tau == 0.04
    assert config.s_axis == 0.02 and config.s_margin == 0.02
    assert config.softness == pytest.approx(config.tau / 2)
    yaml_text = (REPO_ROOT / "configs" / "task6n_oracle_relation_field.yaml").read_text(encoding="utf-8")
    assert "alpha: 1.2" in yaml_text and "tau: 0.04" in yaml_text


def test_frozen_sam2_path_unchanged():
    from buildreasonseg_mvp.task6n_relation_decoder import (
        SAM2_CHECKPOINT, SAM2_CONFIG_NAME, VARIANT_USES_VISUAL,
    )

    assert SAM2_CHECKPOINT.name == "sam2.1_hiera_base_plus.pt"
    assert SAM2_CONFIG_NAME == "configs/sam2.1/sam2.1_hiera_b+.yaml"
    assert VARIANT_USES_VISUAL["B3"] is True and VARIANT_USES_VISUAL["B4"] is False
    reproduction = _artifact("task6o_b2_reproduction.json")
    if reproduction is not None:
        assert reproduction["checkpoint"]["matches"] is True
    for name in TASK6O_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        for marker in ("ultralytics", "yolo26", "resnet", "swin", "YOLO("):
            assert marker not in text, f"{name} must not change the visual backbone ({marker})"


def test_same_packs_across_b2_b3_b4():
    reproduction = _artifact("task6o_b2_reproduction.json")
    mini = _artifact("task6o_mini_val.json")
    overfit = _artifact("task6o_overfit20.json")
    paired = _artifact("task6o_paired_val.json")
    if not all([reproduction, mini, overfit, paired]):
        pytest.skip("Task 6O artifacts not all generated yet")
    manifest = _artifact("task6n_pack_manifest.json")
    for name in ("overfit20", "mini_train_1000", "mini_val_240", "paired_val_20"):
        assert reproduction["pack_verification"]["packs"][name]["actual"] == manifest["packs"][name]["sha256"]
    assert mini["pack"]["sha256"] == manifest["packs"]["mini_val_240"]["sha256"]
    assert paired["pair_pack"]["sha256"] == manifest["packs"]["paired_val_20"]["sha256"]
    assert overfit["pack"]["sha256"] == manifest["packs"]["overfit20"]["sha256"]


def test_o1_settings_identical_to_task6n_n1():
    overfit = _artifact("task6o_overfit20.json")
    frozen = _artifact("task6n_overfit20.json")
    if overfit is None or frozen is None:
        pytest.skip("O1 artifacts not generated yet")
    config = overfit["config"]
    assert config["optimizer"] == "AdamW"
    assert config["lr"] == 1.0e-3
    assert config["weight_decay"] == 1.0e-4
    assert config["max_steps"] == 1200
    assert config["batch_size"] == 4
    assert config["scheduler"] == "none"
    assert config["augmentation"] == "none"
    assert config["seed"] == frozen["config"]["seed"] == 20260929
    assert config["evaluate_every_steps"] == frozen["config"]["evaluate_every_steps"] == 100
    for key in ("optimizer", "lr", "weight_decay", "max_steps", "batch_size", "scheduler",
                "augmentation", "seed", "evaluate_every_steps"):
        assert config[key] == frozen["config"][key], key
    for variant in ("B3", "B4"):
        assert overfit["variants"][variant]["history"][-1]["step"] == 1200
    assert overfit["gate"]["miou_min"] == frozen["gate"]["train_miou_min"] == 0.85
    assert overfit["gate"]["dice_min"] == frozen["gate"]["train_dice_min"] == 0.90
    assert overfit["gate"]["applies_to"] == "B3"


def test_o2_settings_identical_to_task6n_n2():
    training = _artifact("task6o_o2_training.json")
    frozen = _artifact("task6n_mini_val.json")
    if training is None or frozen is None:
        pytest.skip("O2 artifacts not generated yet")
    config = training["config"]
    for key in ("optimizer", "lr", "weight_decay", "batch_size", "max_epochs",
                "early_stopping_patience", "early_stopping_monitor", "model_selection", "seed",
                "scheduler", "augmentation", "fresh_initialisation"):
        assert config[key] == frozen["config"][key], key
    assert config["lr"] == 3.0e-4
    assert config["batch_size"] == 8
    assert config["max_epochs"] == 25
    assert config["early_stopping_patience"] == 5
    for variant in ("B3", "B4"):
        assert training["variants"][variant]["epochs_run"] <= 25


def test_deterministic_seed_exact():
    for name in ("task6o_overfit20.json", "task6o_o2_training.json"):
        payload = _artifact(name)
        if payload is not None:
            assert payload["config"]["seed"] == 20260929, name
    trainer = (SCRIPTS / "task6o_train.py").read_text(encoding="utf-8")
    assert "SEED" in trainer
    from scripts.task6n_train import SEED

    assert SEED == 20260929


# ---------------------------------------------------------------- 21-30 no scope creep


def test_no_new_relation():
    from buildreasonseg_mvp.geometric_relation_field import (
        PROGRAM_TO_RELATION, RELATIONS, RELATION_TO_INDEX,
    )
    from buildreasonseg_mvp.task6n_relation_decoder import (
        ALL_VARIANTS, TASK6O_VARIANTS, VARIANTS,
    )

    assert set(RELATIONS) == {"left_of", "right_of", "above", "below"}
    assert set(RELATION_TO_INDEX) == set(RELATIONS)
    # exactly the 8 Task 6N directional programs, no new relation registered
    assert set(PROGRAM_TO_RELATION) == ALLOWED_PROGRAMS
    assert set(PROGRAM_TO_RELATION.values()) == set(RELATIONS)
    # no new variant family beyond the frozen Task 6N trio and the two Task 6O variants
    assert VARIANTS == ("B0", "B1", "B2")
    assert TASK6O_VARIANTS == ("B3", "B4")
    assert ALL_VARIANTS == ("B0", "B1", "B2", "B3", "B4")


def test_no_nearest_or_l3():
    summary = _artifact("task6o_causal_summary.json")
    if summary is not None:
        assert summary["reference_source"] == "oracle_native_gt"
    # a compositional/nearest program id must never appear anywhere in Task 6O or Task 6N code
    for name in TASK6O_SOURCES + TASK6N_SOURCES:
        code = _code_only(SCRIPTS / name)
        assert "to_nearest" not in code, name
        assert "to_above_to_" not in code, name
    from buildreasonseg_mvp.geometric_relation_field import DIRECTIONAL_PROGRAMS

    assert not any("nearest" in program for program in DIRECTIONAL_PROGRAMS)
    assert not any(program.count("_to_") > 1 for program in DIRECTIONAL_PROGRAMS)


def test_no_ref_token():
    for name in TASK6O_SOURCES:
        assert "[REF]" not in (SCRIPTS / name).read_text(encoding="utf-8"), name
    assert "[REF]" not in (REPO_ROOT / "buildreasonseg_mvp" / "task6n_relation_decoder.py").read_text(
        encoding="utf-8"
    )


def test_no_grcl_or_scl():
    for name in TASK6O_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("GRCL", "SCL", "counterfactual_loss", "relation_loss", "focal_loss"):
            assert marker not in code, f"{name} must not add {marker}"


def test_no_graph_transformer():
    decoder = _code_only(REPO_ROOT / "buildreasonseg_mvp" / "task6n_relation_decoder.py")
    for marker in ("MultiheadAttention", "TransformerEncoder", "GraphConv", "MessagePassing",
                   "nn.Transformer"):
        assert marker not in decoder, marker
    for name in TASK6O_SOURCES:
        assert "MultiheadAttention" not in _code_only(SCRIPTS / name), name


def test_no_proposal_training():
    for name in TASK6O_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("ultralytics", "yolo26", "task6m_train", "YOLO("):
            assert marker not in code, f"{name} must not train a proposal model ({marker})"


def test_no_4b():
    for name in TASK6O_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        assert "Qwen3-VL-4B" not in text, name
        assert "4B" not in text.replace("4B)", ""), name


def test_no_download_or_install():
    for name in TASK6O_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("urlretrieve", "requests.get", "Invoke-WebRequest", "hf_hub_download",
                       "snapshot_download", "pip install", "subprocess.run([\"pip"):
            assert marker not in code, f"{name} must not download/install ({marker})"


def test_no_gui():
    for name in TASK6O_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("tkinter", "PyQt", "gradio", "streamlit", "cv2.imshow", "flask", "fastapi"):
            assert marker not in code, f"{name} must not build a GUI ({marker})"


def test_previous_suite_preserved():
    for name in ("test_task6n_oracle_relation_field.py", "test_task6m1_convergence_demo_fix.py",
                 "test_task6m_native_vector_proposal.py", "test_task6h_counterfactual_grounding.py"):
        assert (REPO_ROOT / "tests" / name).is_file(), name
    removed = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=D", BASE_COMMIT, "--", "tests/"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert removed == "", f"no test file may be removed: {removed}"
