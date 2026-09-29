"""Task 6N tests (section 21): 30 checks on the oracle-reference geometric relation field.

Checks that need a generated Task 6N artifact skip (never fail) while the corresponding stage has not
run yet.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "spatial_reasoning"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

EVAL = REPO_ROOT / "evaluation"
SCRIPTS = REPO_ROOT / "scripts"
V02 = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6n" / "packs"

TASK6N_SOURCES = (
    "task6n_freeze_packs.py", "task6n_field_sanity.py", "task6n_train.py",
    "task6n_evaluate.py", "task6n_report.py",
)
TASK6N_MODULES = (
    REPO_ROOT / "buildreasonseg_mvp" / "geometric_relation_field.py",
    REPO_ROOT / "buildreasonseg_mvp" / "task6n_relation_decoder.py",
)
ALLOWED_PROGRAMS = {
    "largest_to_left_of", "largest_to_right_of", "largest_to_above", "largest_to_below",
    "smallest_to_left_of", "smallest_to_right_of", "smallest_to_above", "smallest_to_below",
}


def _artifact(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference_mask(size=(64, 64), box=(24, 28, 40, 44)) -> torch.Tensor:
    mask = torch.zeros(size[0], size[1])
    mask[box[0]:box[2], box[1]:box[3]] = 1.0
    return mask


# ---------------------------------------------------------------- 1-4 frozen assets


def test_task6m1_artifacts_unchanged():
    for name in (
        "task6m1_verdict.json", "task6m1_training_summary.json", "task6m1_j1v2_val.json",
        "task6m1_proposal_val.json", "task6m1_inference_config_frozen.json",
        "task6m1_source_checkpoint_audit.json", "task6m1_demo_cli_audit.json",
    ):
        assert (EVAL / name).is_file(), name
    verdict = _artifact("task6m1_verdict.json")
    assert verdict["verdict"] == "PROPOSAL_MODEL_NEEDS_IMPROVEMENT_AFTER_CONVERGENCE"
    assert verdict["task6m_evidence_preserved"]["recorded_hashes"]["best.pt"].startswith(
        "fd407db634a8a7ef"
    )


def test_v02_unchanged():
    index = _artifact("task6l_artifact_index.json")
    if index is None:
        pytest.skip("Task 6L artifact index missing")
    for entry in index["artifacts"]:
        if not str(entry["path"]).startswith("datasets/build_spatial_reason/v0.2/"):
            continue
        path = REPO_ROOT / entry["path"]
        if path.is_file():
            assert _sha256(path) == entry["sha256"], f"{entry['path']} changed"
    statistics = json.loads((V02 / "statistics.json").read_text(encoding="utf-8"))
    for split in ("train", "val", "test"):
        lines = sum(
            1 for line in (V02 / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
        )
        assert lines == statistics["by_split"][split]


def test_spatial_config_unchanged():
    index = _artifact("task6l_artifact_index.json")
    config = REPO_ROOT / "configs" / "spatial_relations_v1.yaml"
    if index is not None:
        for entry in index["artifacts"]:
            if str(entry["path"]).endswith("spatial_relations_v1.yaml"):
                assert _sha256(config) == entry["sha256"]
    text = config.read_text(encoding="utf-8")
    assert "alpha: 1.2" in text
    assert "tau: 0.04" in text


def test_no_test_access():
    for path in [SCRIPTS / name for name in TASK6N_SOURCES] + list(TASK6N_MODULES):
        text = path.read_text(encoding="utf-8")
        for marker in ('"test.jsonl"', "'test.jsonl'", 'split="test"', "split('test')",
                       "task6n_test", "test_fixed120", "paired_test"):
            assert marker not in text, f"{path.name} must not touch the test split ({marker})"
    for name in ("task6n_pack_manifest.json", "task6n_field_sanity.json", "task6n_overfit20.json",
                 "task6n_mini_val.json", "task6n_ablation_summary.json", "task6n_paired_val.json",
                 "task6n_verdict.json"):
        payload = _artifact(name)
        if payload is not None:
            assert payload.get("test_split_used", False) is False or "test_split_used" not in payload


# ---------------------------------------------------------------- 5-7 scope and protocol


def test_only_eight_allowed_programs():
    from buildreasonseg_mvp.geometric_relation_field import DIRECTIONAL_PROGRAMS, PROGRAM_TO_RELATION

    assert set(DIRECTIONAL_PROGRAMS) == ALLOWED_PROGRAMS
    assert len(DIRECTIONAL_PROGRAMS) == 8
    assert set(PROGRAM_TO_RELATION.values()) == {"left_of", "right_of", "above", "below"}
    manifest = _artifact("task6n_pack_manifest.json")
    if manifest is not None:
        assert set(manifest["scope"]["programs"] if "scope" in manifest else ALLOWED_PROGRAMS) <= ALLOWED_PROGRAMS
        pack_files = {name: PACK_ROOT / name for name in (
            "mini_val_240.json", "overfit20.json", "mini_train_1000.json", "paired_val_20.json",
        )}
        if not all(path.is_file() for path in pack_files.values()):
            pytest.skip("packs are regenerable artifacts under gitignored artifacts/ (run task6n_freeze_packs.py)")
        records = json.loads(pack_files["mini_val_240.json"].read_text(encoding="utf-8"))["records"]
        assert {record["program_id"] for record in records} == ALLOWED_PROGRAMS
        for name in ("overfit20.json", "mini_train_1000.json", "paired_val_20.json"):
            payload = json.loads(pack_files[name].read_text(encoding="utf-8"))
            programs = {record["program_id"] for record in payload.get("records", [])}
            if name == "paired_val_20.json":
                programs = {pair["a"]["program_id"] for pair in payload["pairs"]} | {
                    pair["b"]["program_id"] for pair in payload["pairs"]
                }
            assert programs <= ALLOWED_PROGRAMS, name


def test_target_never_input():
    decoder = (REPO_ROOT / "buildreasonseg_mvp" / "task6n_relation_decoder.py").read_text(encoding="utf-8")
    forward = decoder.split("def forward(")[1].split("def parameter_report")[0]
    assert "target" not in forward, "the decoder forward must not receive the GT target"
    for name in ("visual", "relation_index", "mask_ref_down", "field"):
        assert name in forward
    trainer = (SCRIPTS / "task6n_train.py").read_text(encoding="utf-8")
    # the target is only ever used for the loss / metrics
    assert "task6n_loss(upsampled_logits(logits, batch.target_size), batch.target)" in trainer
    assert "model(batch.visual, batch.relation_index, batch.mask_ref_down, batch.field)" in trainer


def test_oracle_reference_explicitly_marked():
    for name in ("task6n_pack_manifest.json", "task6n_overfit20.json", "task6n_mini_val.json",
                 "task6n_ablation_summary.json", "task6n_verdict.json", "task6n_field_sanity.json"):
        payload = _artifact(name)
        if payload is not None:
            assert payload.get("reference_source") == "oracle_native_gt", name
    module = (REPO_ROOT / "buildreasonseg_mvp" / "task6n_relation_decoder.py").read_text(encoding="utf-8")
    assert '"reference_source": "oracle_native_gt"' in module
    if PACK_ROOT.is_dir():
        for pack in PACK_ROOT.glob("*.json"):
            payload = json.loads(pack.read_text(encoding="utf-8"))
            assert payload["reference_source"] == "oracle_native_gt", pack.name


# ---------------------------------------------------------------- 8-15 field geometry


def test_direction_signs_correct():
    from buildreasonseg_mvp.geometric_relation_field import geometric_relation_field, reference_centroid

    mask = _reference_mask()
    cx, cy = reference_centroid(mask)
    left = geometric_relation_field(mask, "left_of", (64, 64))
    right = geometric_relation_field(mask, "right_of", (64, 64))
    col_left = int(cx * 64) - 10
    col_right = int(cx * 64) + 10
    row = int(cy * 64)
    assert float(left[0, 0, row, col_left]) > float(left[0, 0, row, col_right])
    assert float(right[0, 0, row, col_right]) > float(right[0, 0, row, col_left])
    # canonical convention: left_of(T, R) means cx_T < cx_R, so positions with x - cx_ref < 0 score
    x = (torch.arange(64, dtype=torch.float32) + 0.5) / 64
    assert float(x[col_left]) < cx < float(x[col_right])


def test_y_axis_convention_correct():
    from buildreasonseg_mvp.geometric_relation_field import geometric_relation_field, reference_centroid

    mask = _reference_mask()
    _, cy = reference_centroid(mask)
    above = geometric_relation_field(mask, "above", (64, 64))
    below = geometric_relation_field(mask, "below", (64, 64))
    row_above = max(int(cy * 64) - 10, 0)
    row_below = min(int(cy * 64) + 10, 63)
    column = 32
    # image y increases downward, so `above` (cy_T < cy_R) must peak at smaller row indices
    assert float(above[0, 0, row_above, column]) > float(above[0, 0, row_below, column])
    assert float(below[0, 0, row_below, column]) > float(below[0, 0, row_above, column])
    y = (torch.arange(64, dtype=torch.float32) + 0.5) / 64
    assert float(y[row_above]) < cy < float(y[row_below])


def test_field_constants_alpha_tau_softness():
    from buildreasonseg_mvp.geometric_relation_field import DEFAULT_FIELD_CONFIG, FieldConfig

    config = DEFAULT_FIELD_CONFIG
    assert config.alpha == 1.2
    assert config.tau == 0.04
    assert config.s_axis == 0.02
    assert config.s_margin == 0.02
    assert config.softness == pytest.approx(0.02)
    assert config.softness == pytest.approx(config.tau / 2)
    assert FieldConfig().alpha == 1.2
    yaml_text = (REPO_ROOT / "configs" / "task6n_oracle_relation_field.yaml").read_text(encoding="utf-8")
    assert "alpha: 1.2" in yaml_text and "tau: 0.04" in yaml_text
    assert "s_axis: 0.02" in yaml_text and "s_margin: 0.02" in yaml_text


def test_field_bounded_and_parameter_free():
    from buildreasonseg_mvp.geometric_relation_field import geometric_relation_field

    mask = _reference_mask()
    for relation in ("left_of", "right_of", "above", "below"):
        field = geometric_relation_field(mask, relation, (64, 64))
        assert field.shape == (1, 1, 64, 64)
        assert float(field.min()) >= 0.0 and float(field.max()) <= 1.0
        assert torch.isfinite(field).all()
    reference_centroid_module = (REPO_ROOT / "buildreasonseg_mvp" / "geometric_relation_field.py").read_text(
        encoding="utf-8"
    )
    assert "nn.Module" not in reference_centroid_module
    assert "Parameter" not in reference_centroid_module
    assert "requires_grad" not in reference_centroid_module


def _centered_reference_mask(size=(64, 64)) -> torch.Tensor:
    """Reference mask centred on the tile centre so tile-centre mirroring equals centroid mirroring."""

    mask = torch.zeros(size[0], size[1])
    mask[24:40, 24:40] = 1.0
    return mask


def test_left_right_mirror_sanity():
    from buildreasonseg_mvp.geometric_relation_field import geometric_relation_field, reference_centroid

    mask = _centered_reference_mask()
    cx, cy = reference_centroid(mask)
    assert cx == pytest.approx(0.5, abs=1e-6) and cy == pytest.approx(0.5, abs=1e-6)
    left = geometric_relation_field(mask, "left_of", (64, 64))
    right = geometric_relation_field(mask, "right_of", (64, 64))
    mirrored = torch.flip(right, dims=[-1])
    assert torch.allclose(left, mirrored, atol=1e-6), "left_of must be the mirror of right_of"
    assert not torch.allclose(left, right, atol=1e-3), "left_of and right_of must differ"


def test_above_below_mirror_sanity():
    from buildreasonseg_mvp.geometric_relation_field import geometric_relation_field, reference_centroid

    mask = _centered_reference_mask()
    cx, cy = reference_centroid(mask)
    assert cx == pytest.approx(0.5, abs=1e-6) and cy == pytest.approx(0.5, abs=1e-6)
    above = geometric_relation_field(mask, "above", (64, 64))
    below = geometric_relation_field(mask, "below", (64, 64))
    mirrored = torch.flip(below, dims=[-2])
    assert torch.allclose(above, mirrored, atol=1e-6), "above must be the mirror of below"
    assert not torch.allclose(above, below, atol=1e-3), "above and below must differ"


# ---------------------------------------------------------------- 16-18 variant wiring


def test_b0_has_no_reference_and_no_field():
    from buildreasonseg_mvp.task6n_relation_decoder import (
        DecoderConfig,
        RelationMaskDecoder,
        VARIANT_USES_FIELD,
        VARIANT_USES_REFERENCE,
        first_conv_in_channels,
    )

    assert VARIANT_USES_REFERENCE["B0"] is False
    assert VARIANT_USES_FIELD["B0"] is False
    config = DecoderConfig()
    assert first_conv_in_channels("B0", config) == 144
    torch.manual_seed(0)
    model = RelationMaskDecoder("B0", config).eval()
    visual = torch.randn(1, config.visual_channels, 8, 8)
    relation = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        bare = model(visual, relation)
        with_extras = model(visual, relation, torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8))
    assert bare.shape == (1, 1, 8, 8)
    # B0 must ignore the reference mask and the field entirely
    assert torch.allclose(bare, with_extras, atol=1e-6)


def test_b1_has_reference_but_no_field():
    from buildreasonseg_mvp.task6n_relation_decoder import (
        DecoderConfig,
        RelationMaskDecoder,
        VARIANT_USES_FIELD,
        VARIANT_USES_REFERENCE,
        first_conv_in_channels,
    )

    assert VARIANT_USES_REFERENCE["B1"] is True
    assert VARIANT_USES_FIELD["B1"] is False
    config = DecoderConfig()
    assert first_conv_in_channels("B1", config) == 145
    torch.manual_seed(0)
    model = RelationMaskDecoder("B1", config).eval()
    visual = torch.randn(1, config.visual_channels, 8, 8)
    relation = torch.zeros(1, dtype=torch.long)
    reference = torch.rand(1, 1, 8, 8)
    with pytest.raises(ValueError):
        model(visual, relation)  # B1 requires the reference mask
    with torch.no_grad():
        without_field = model(visual, relation, reference)
        with_field = model(visual, relation, reference, torch.rand(1, 1, 8, 8))
    assert without_field.shape == (1, 1, 8, 8)
    # B1 must ignore the field entirely
    assert torch.allclose(without_field, with_field, atol=1e-6)


def test_b2_has_reference_and_field():
    from buildreasonseg_mvp.task6n_relation_decoder import (
        DecoderConfig,
        RelationMaskDecoder,
        VARIANT_USES_FIELD,
        VARIANT_USES_REFERENCE,
        first_conv_in_channels,
    )

    assert VARIANT_USES_REFERENCE["B2"] is True
    assert VARIANT_USES_FIELD["B2"] is True
    config = DecoderConfig()
    assert first_conv_in_channels("B2", config) == 146
    torch.manual_seed(0)
    model = RelationMaskDecoder("B2", config).eval()
    visual = torch.randn(1, config.visual_channels, 8, 8)
    relation = torch.zeros(1, dtype=torch.long)
    reference = torch.rand(1, 1, 8, 8)
    field = torch.rand(1, 1, 8, 8)
    with pytest.raises(ValueError):
        model(visual, relation, reference)  # B2 requires the field
    with torch.no_grad():
        with_field = model(visual, relation, reference, field)
        other_field = model(visual, relation, reference, torch.rand(1, 1, 8, 8))
    assert with_field.shape == (1, 1, 8, 8)
    # B2 must actually consume the field
    assert not torch.allclose(with_field, other_field, atol=1e-4)
    summary = _artifact("task6n_ablation_summary.json")
    if summary is not None:
        assert summary["variants"]["B2"]["architecture"]["first_conv_in_channels"] == 146
        assert summary["variants"]["B0"]["architecture"]["first_conv_in_channels"] == 144
        assert summary["variants"]["B1"]["architecture"]["first_conv_in_channels"] == 145
        # no padding to equalise parameters
        totals = {summary["variants"][v]["architecture"]["total_parameters"] for v in ("B0", "B1", "B2")}
        assert len(totals) == 3


# ---------------------------------------------------------------- 19-24 controlled protocol


def test_same_visual_backbone_and_cache():
    from buildreasonseg_mvp.task6n_relation_decoder import SAM2_CHECKPOINT, SAM2_CONFIG_NAME

    assert SAM2_CHECKPOINT.name == "sam2.1_hiera_base_plus.pt"
    assert SAM2_CONFIG_NAME == "configs/sam2.1/sam2.1_hiera_b+.yaml"
    for name in ("task6n_overfit20.json", "task6n_mini_val.json", "task6n_ablation_summary.json"):
        payload = _artifact(name)
        if payload is not None:
            visual = payload.get("visual") or {}
            assert "SAM2.1 Hiera Base+" in visual.get("encoder", ""), name
            assert visual.get("reused_from", "").startswith("Task 6C.7")
    for name in TASK6N_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        for marker in ("YOLO(", "ultralytics", "yolo26", "resnet", "swin"):
            assert marker not in text, f"{name} must not switch the visual backbone ({marker})"


def test_same_packs_across_variants():
    manifest = _artifact("task6n_pack_manifest.json")
    if manifest is None:
        pytest.skip("Task 6N packs not frozen yet")
    for name, entry in manifest["packs"].items():
        path = Path(entry["path"])
        assert path.is_file(), name
        assert _sha256(path) == entry["sha256"], f"{name} changed after freezing"
    for artifact_name in ("task6n_overfit20.json", "task6n_mini_val.json"):
        payload = _artifact(artifact_name)
        if payload is None:
            continue
        packs = payload.get("packs") or ({"overfit20": payload.get("pack")} if payload.get("pack") else {})
        for entry in packs.values():
            if entry and entry.get("sha256"):
                assert _sha256(Path(entry["path"])) == entry["sha256"], artifact_name


def test_same_n1_optimizer_and_steps():
    payload = _artifact("task6n_overfit20.json")
    if payload is None:
        pytest.skip("Stage N1 not run yet")
    config = payload["config"]
    assert config["optimizer"] == "AdamW"
    assert config["lr"] == 1.0e-3
    assert config["weight_decay"] == 1.0e-4
    assert config["max_steps"] == 1200
    assert config["batch_size"] == 4
    assert config["scheduler"] == "none"
    assert config["augmentation"] == "none"
    assert config["seed"] == 20260929
    assert config["evaluate_every_steps"] == 100
    for variant in ("B0", "B1", "B2"):
        entry = payload["variants"][variant]
        assert entry["history"][-1]["step"] == 1200, variant
        assert len(entry["history"]) == 12, variant
    assert payload["gate"]["train_miou_min"] == 0.85
    assert payload["gate"]["train_dice_min"] == 0.90


def test_same_n2_optimizer_and_epochs():
    payload = _artifact("task6n_mini_val.json")
    if payload is None:
        pytest.skip("Stage N2 not run yet")
    config = payload["config"]
    assert config["optimizer"] == "AdamW"
    assert config["lr"] == 3.0e-4
    assert config["weight_decay"] == 1.0e-4
    assert config["batch_size"] == 8
    assert config["max_epochs"] == 25
    assert config["early_stopping_patience"] == 5
    assert config["early_stopping_monitor"] == "val_miou"
    assert config["model_selection"] == "highest MiniVal240 mIoU"
    assert config["seed"] == 20260929
    assert config["scheduler"] == "none"
    assert config["augmentation"] == "none"
    assert config["fresh_initialisation"] is True
    for variant in ("B0", "B1", "B2"):
        assert payload["variants"][variant]["epochs_run"] <= 25


def test_deterministic_pack_hashes():
    manifest = _artifact("task6n_pack_manifest.json")
    if manifest is None:
        pytest.skip("Task 6N packs not frozen yet")
    recorded = {name: entry["sha256"] for name, entry in manifest["packs"].items()}
    assert len(recorded) == 4
    for name, digest in recorded.items():
        assert len(digest) == 64 and digest == _sha256(Path(manifest["packs"][name]["path"])), name
    assert manifest["packs"]["overfit20"]["detail"]["pair_count"] >= 4
    assert manifest["packs"]["mini_val_240"]["detail"]["all_eight_programs"] is True
    assert manifest["packs"]["mini_train_1000"]["count"] == 1000


def test_paired_val_same_tile_reference_different_target():
    manifest = _artifact("task6n_pack_manifest.json")
    paired_path = PACK_ROOT / "paired_val_20.json"
    if manifest is None or not paired_path.is_file():
        pytest.skip("packs are regenerable artifacts under gitignored artifacts/ (run task6n_freeze_packs.py)")
    payload = json.loads(paired_path.read_text(encoding="utf-8"))
    assert payload["count"] >= 20
    for pair in payload["pairs"]:
        a, b = pair["a"], pair["b"]
        assert a["tile_id"] == b["tile_id"] == pair["tile_id"]
        assert a["reference_source_feature_id"] == b["reference_source_feature_id"]
        assert a["relation"] != b["relation"]
        assert a["target_source_feature_id"] != b["target_source_feature_id"]
        assert a["reference_source"] == b["reference_source"] == "oracle_native_gt"
    report = _artifact("task6n_paired_val.json")
    if report is not None:
        for variant in ("B0", "B1", "B2"):
            entry = report["variants"][variant]["paired_val"]
            assert entry["pairs"] == len(payload["pairs"])
            for row in entry["rows"]:
                assert row["passes"] == (row["a"]["prefers_own"] and row["b"]["prefers_own"])


# ---------------------------------------------------------------- 25-30 no scope creep


def test_no_ref_token():
    for path in [SCRIPTS / name for name in TASK6N_SOURCES] + list(TASK6N_MODULES):
        assert "[REF]" not in path.read_text(encoding="utf-8"), path.name


def test_no_grcl_or_scl():
    for path in [SCRIPTS / name for name in TASK6N_SOURCES] + list(TASK6N_MODULES):
        text = path.read_text(encoding="utf-8")
        for marker in ("GRCL", "SCL", "counterfactual_loss", "relation_loss", "focal_loss",
                       "auxiliary_field_loss"):
            assert marker not in text, f"{path.name} must not add {marker}"


def _code_only(path: Path) -> str:
    """Module source with the leading docstring and `#` comments removed (prose is not architecture)."""

    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    for quote in ('"""', "'''"):
        if stripped.startswith(quote):
            end = stripped.find(quote, len(quote))
            if end != -1:
                stripped = stripped[end + len(quote):]
            break
    lines = [line.split("#", 1)[0] for line in stripped.splitlines()]
    return "\n".join(lines)


def test_no_graph_transformer():
    decoder = _code_only(REPO_ROOT / "buildreasonseg_mvp" / "task6n_relation_decoder.py")
    for marker in ("MultiheadAttention", "TransformerEncoder", "GraphConv", "GAT", "MessagePassing",
                   "nn.Transformer", "attention"):
        assert marker not in decoder, f"the decoder must not add {marker}"
    trunk = decoder.split("self.trunk = nn.Sequential(")[1].split("self.trunk")[0]
    assert trunk.count("Conv2d") == 3
    assert trunk.count("GroupNorm") == 2
    field_module = _code_only(REPO_ROOT / "buildreasonseg_mvp" / "geometric_relation_field.py")
    for marker in ("MultiheadAttention", "TransformerEncoder", "GraphConv", "MessagePassing"):
        assert marker not in field_module


def test_no_proposal_training():
    for path in [SCRIPTS / name for name in TASK6N_SOURCES] + list(TASK6N_MODULES):
        text = path.read_text(encoding="utf-8")
        for marker in ("YOLO(", "ultralytics", "yolo26m", "yolo26s", "task6m_train"):
            assert marker not in text, f"{path.name} must not train a proposal model ({marker})"


def test_no_4b():
    for path in [SCRIPTS / name for name in TASK6N_SOURCES] + list(TASK6N_MODULES):
        text = path.read_text(encoding="utf-8")
        assert "Qwen3-VL-4B" not in text, path.name
        assert "4B" not in text.replace("4B)", ""), path.name


def test_no_gui_download_or_install():
    for path in [SCRIPTS / name for name in TASK6N_SOURCES] + list(TASK6N_MODULES):
        text = path.read_text(encoding="utf-8")
        for marker in ("tkinter", "PyQt", "gradio", "streamlit", "cv2.imshow", "flask", "fastapi",
                       "urlretrieve", "requests.get", "Invoke-WebRequest", "hf_hub_download",
                       "snapshot_download", "pip install", "subprocess.run([\"pip"):
            assert marker not in text, f"{path.name} must not use {marker}"
