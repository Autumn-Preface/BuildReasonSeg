"""Task 6Z tests (section O): 49 checks on the oracle-reference L3 direction x nearest composition."""

from __future__ import annotations

import hashlib
import json
import subprocess
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
COMPOSITION_MODULE = REPO_ROOT / "buildreasonseg_mvp" / "task6z_field_composition.py"
DECODER_MODULE = REPO_ROOT / "buildreasonseg_mvp" / "task6z_l3_decoder.py"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6z" / "packs"
CONFIG = REPO_ROOT / "configs" / "spatial_relations_v1.yaml"
BASE_COMMIT = "24be95494a36e1a21d1f3566590c06ef118c7f28"
TASK6Z_SOURCES = ("task6z_freeze_packs.py", "task6z_composition_sanity.py", "task6z_train.py",
                  "task6z_evaluate.py", "task6z_report.py", "task6z_fields.py")
L3_PROGRAMS = ("largest_to_left_of_to_nearest", "largest_to_right_of_to_nearest",
               "largest_to_above_to_nearest", "largest_to_below_to_nearest")


def _artifact(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _pack(name: str):
    path = PACK_ROOT / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _git_changed(prefix: str) -> str:
    return subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "--", prefix],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _code_only(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    for prefix in ("r", "f", "b", "u", "rf", "fr"):
        if stripped.startswith(prefix + '"""') or stripped.startswith(prefix + "'''"):
            stripped = stripped[len(prefix):]
            break
    for quote in ('"""', "'''"):
        if stripped.startswith(quote):
            end = stripped.find(quote, len(quote))
            if end != -1:
                stripped = stripped[end + len(quote):]
            break
    return "\n".join(line.split("#", 1)[0] for line in stripped.splitlines())


def _reference_mask() -> np.ndarray:
    mask = np.zeros((512, 512), dtype=bool)
    mask[200:260, 180:240] = True
    return mask


# ---------------------------------------------------------------- 1-5 freeze and scope


def test_task6y_artifacts_unchanged():
    assert _git_changed("evaluation/task6y_") == ""
    assert _git_changed("buildreasonseg_mvp/nearest_boundary_field.py") == ""
    assert _git_changed("buildreasonseg_mvp/task6y_nearest_decoder.py") == ""
    assert _artifact("task6y_verdict.json")["verdict"] == "NEAREST_FIELD_NO_MEANINGFUL_GAIN"


def test_exactly_four_l3_program_ids():
    from buildreasonseg_mvp.task6z_field_composition import L3_PROGRAMS as programs

    assert tuple(programs) == L3_PROGRAMS
    manifest = _artifact("task6z_pack_manifest.json")
    assert manifest is not None
    assert manifest["programs"] == list(L3_PROGRAMS)
    assert manifest["integrity"]["programs_present"] == sorted(L3_PROGRAMS)


def test_no_l1_l2_or_test_record_in_packs():
    for name in ("z_overfit20", "z_mini_train_1200", "z_mini_val_240", "z_paired_val20"):
        pack = _pack(name)
        if pack is None:
            pytest.skip("packs not frozen yet")
        for record in pack["records"]:
            assert record["program_id"] in L3_PROGRAMS, record["program_id"]
            assert record["level"] == 3, (name, record["sample_id"])
            assert record["split"] in ("train", "val")
    manifest = _artifact("task6z_pack_manifest.json")
    assert manifest["integrity"]["l1_records"] == 0
    assert manifest["integrity"]["l2_records"] == 0
    assert manifest["integrity"]["test_records"] == 0


def test_no_nonexistent_smallest_l3_program():
    for name in TASK6Z_SOURCES:
        code = _code_only(SCRIPTS / name)
        assert "smallest_to_left_of_to_nearest" not in code
        assert "smallest_to_right_of_to_nearest" not in code
        assert "smallest_to_above_to_nearest" not in code
        assert "smallest_to_below_to_nearest" not in code
    manifest = _artifact("task6z_pack_manifest.json")
    assert manifest["integrity"]["smallest_l3_programs"] == 0


def test_canonical_l3_operation_order():
    from buildreasonseg_mvp.task6z_field_composition import field_report

    report = field_report()
    assert report["canonical_operation_order"] == ["argmax_area (largest eligible reference)",
                                                   "filter_relation",
                                                   "argmin_boundary_distance"]
    manifest = _artifact("task6z_pack_manifest.json")
    assert manifest["canonical_semantics"]["operation_order"] == ["argmax_area", "filter_relation",
                                                                  "argmin_boundary_distance"]
    assert manifest["canonical_semantics"]["labels_regenerated"] is False
    verdict = _artifact("task6z_verdict.json")
    if verdict is not None:
        assert verdict["protocol"]["canonical_operation_order"] == ["argmax_area", "filter_relation",
                                                                    "argmin_boundary_distance"]


# ---------------------------------------------------------------- 6-10 frozen constants and fields


def test_direction_alpha_tau_unchanged():
    from buildreasonseg_mvp.geometric_relation_field_v02 import ALPHA, TAU

    assert ALPHA == 1.2 and TAU == 0.04
    assert _git_changed("buildreasonseg_mvp/geometric_relation_field_v02.py") == ""
    manifest = _artifact("task6z_pack_manifest.json")
    assert manifest["canonical_semantics"]["directional_alpha"] == 1.2
    assert manifest["canonical_semantics"]["directional_tau"] == 0.04


def test_nearest_boundary_distance_semantics_unchanged():
    section = CONFIG.read_text(encoding="utf-8").split("nearest:")[1].split("size_rank:")[0]
    assert "distance_metric: boundary_distance" in section
    assert "margin_px_floor: 2.0" in section
    assert "margin_diag_fraction: 0.005" in section
    assert _git_changed("configs/spatial_relations_v1.yaml") == ""
    assert _git_changed("spatial_reasoning/relations.py") == ""
    manifest = _artifact("task6z_pack_manifest.json")
    assert manifest["canonical_semantics"]["distance_metric"] == "boundary_distance"


def test_nearest_sigma_unchanged():
    from buildreasonseg_mvp.nearest_boundary_field import SIGMA_DIAG

    assert SIGMA_DIAG == 0.05
    sanity = _artifact("task6z_composition_sanity.json")
    assert sanity["sigma_diag"] == 0.05
    assert sanity["constants_tuned"] is False
    for name in TASK6Z_SOURCES:
        assert "SIGMA_DIAG = " not in _code_only(SCRIPTS / name)


def test_geometric_relation_field_v02_unchanged():
    assert _git_changed("buildreasonseg_mvp/geometric_relation_field_v02.py") == ""
    assert _git_changed("buildreasonseg_mvp/geometric_relation_field.py") == ""
    sanity = _artifact("task6z_composition_sanity.json")
    assert sanity["fields"]["directional_field"]["modified"] is False


def test_nearest_boundary_field_v01_unchanged():
    assert _git_changed("buildreasonseg_mvp/nearest_boundary_field.py") == ""
    assert _git_changed("buildreasonseg_mvp/task6y_nearest_decoder.py") == ""
    sanity = _artifact("task6z_composition_sanity.json")
    assert sanity["fields"]["nearest_field"]["modified"] is False


# ---------------------------------------------------------------- 11-15 the product field


def test_p_prod_source_formula_exact_multiplication():
    from buildreasonseg_mvp.task6z_field_composition import product_field

    directional = torch.tensor([[[[0.5, 0.25], [0.0, 1.0]]]])
    nearest = torch.tensor([[[[0.4, 0.8], [0.9, 1.0]]]])
    expected = torch.tensor([[[[0.20, 0.20], [0.0, 1.0]]]])
    assert torch.allclose(product_field(directional, nearest), expected)
    report = _composition_report()
    assert report["product_field"]["source"] == "clamp(P_dir_512 * P_near_512, 0, 1)"


def test_p_prod_decoder_formula_exact_multiplication():
    from buildreasonseg_mvp.task6z_field_composition import record_fields

    fields = record_fields(_reference_mask(), "largest_to_left_of_to_nearest")
    directional, nearest = fields["P_dir_64"], fields["P_near_64"]
    assert tuple(directional.shape) == (64, 64)
    assert torch.allclose(fields["P_prod_64"], (directional * nearest).clamp(0.0, 1.0), atol=1e-6)
    report = _composition_report()
    assert report["product_field"]["decoder"] == "clamp(P_dir_64 * P_near_64, 0, 1)"


def test_no_renormalization_temperature_or_power():
    report = _composition_report()
    for key in ("renormalization", "learned_scalar", "temperature", "exponent", "threshold"):
        assert report["product_field"][key] is False, key
    code = _code_only(COMPOSITION_MODULE)
    for marker in ("normalize", "softmax", "**", "pow(", "temperature", "exp("):
        assert marker not in code.split("def product_field")[1].split("def record_fields")[0], marker


def test_composition_sanity_score_is_max_field_in_candidate_mask():
    from buildreasonseg_mvp.nearest_boundary_field import candidate_field_score

    field = np.zeros((8, 8), dtype=np.float32)
    field[1, 1] = 0.9
    field[4, 4] = 0.3
    candidate = np.zeros((8, 8), dtype=bool)
    candidate[1, 1] = True
    candidate[4, 4] = True
    assert candidate_field_score(field, candidate) == pytest.approx(0.9, abs=1e-6)
    code = _code_only(SCRIPTS / "task6z_composition_sanity.py")
    assert "candidate_field_score(field, candidate.mask)" in code


def test_sanity_direction_valid_subset_uses_frozen_relation_engine():
    code = _code_only(SCRIPTS / "task6z_composition_sanity.py")
    assert "R.evaluate_direction(relation, image, subject, reference_component, config," in code
    sanity = _artifact("task6z_composition_sanity.json")
    assert sanity["dynamic_relation_engine"]["module"] == \
        "spatial_reasoning/relations.py::evaluate_direction"
    assert sanity["dynamic_relation_engine"]["frozen"] is True
    assert sanity["sanity_passed"] is True
    assert sanity["gate"]["top1"]["measured"] >= 0.95
    assert sanity["gate"]["top3"]["measured"] >= 0.99
    assert sanity["gate"]["spearman"]["measured"] >= 0.90
    # the engine recomputation must never be a superset of the canonical step-2 list
    assert sanity["dynamic_relation_engine"]["engine_superset_records"] == 0


# ---------------------------------------------------------------- 16-24 packs


def test_overfit20_exact_5_5_5_5():
    manifest = _artifact("task6z_pack_manifest.json")
    entry = manifest["packs"]["z_overfit20"]
    assert entry["records"] == 20
    assert entry["by_direction"] == {"above": 5, "below": 5, "left": 5, "right": 5}
    assert entry["unique_sample_ids"] == 20
    assert entry["unique_tiles"] >= 15


def test_mini_train_1200_exact_300_each():
    manifest = _artifact("task6z_pack_manifest.json")
    entry = manifest["packs"]["z_mini_train_1200"]
    assert entry["records"] == 1200
    assert entry["by_direction"] == {"above": 300, "below": 300, "left": 300, "right": 300}
    assert entry["unique_sample_ids"] == 1200


def test_mini_val_240_exact_60_each():
    manifest = _artifact("task6z_pack_manifest.json")
    entry = manifest["packs"]["z_mini_val_240"]
    assert entry["records"] == 240
    assert entry["by_direction"] == {"above": 60, "below": 60, "left": 60, "right": 60}
    assert entry["unique_sample_ids"] == 240


def test_paired_val_same_tile():
    pack = _pack("z_paired_val20")
    if pack is None:
        pytest.skip("packs not frozen yet")
    records = pack["records"]
    for index in range(0, len(records), 2):
        assert records[index]["tile_id"] == records[index + 1]["tile_id"]


def test_paired_val_same_reference_id():
    pack = _pack("z_paired_val20")
    if pack is None:
        pytest.skip("packs not frozen yet")
    records = pack["records"]
    for index in range(0, len(records), 2):
        assert records[index]["reference_source_feature_id"] == \
            records[index + 1]["reference_source_feature_id"]


def test_paired_val_different_direction():
    pack = _pack("z_paired_val20")
    if pack is None:
        pytest.skip("packs not frozen yet")
    records = pack["records"]
    for index in range(0, len(records), 2):
        assert records[index]["program_id"] != records[index + 1]["program_id"]
        assert records[index]["relation"] != records[index + 1]["relation"]


def test_paired_val_different_target_id():
    pack = _pack("z_paired_val20")
    if pack is None:
        pytest.skip("packs not frozen yet")
    records = pack["records"]
    for index in range(0, len(records), 2):
        assert records[index]["target_source_feature_id"] != \
            records[index + 1]["target_source_feature_id"]
    manifest = _artifact("task6z_pack_manifest.json")
    assert manifest["paired"]["n_pair"] * 2 == len(records)
    assert manifest["paired"]["n_pair"] >= manifest["paired"]["minimum"]


def test_pack_hashes_recorded():
    manifest = _artifact("task6z_pack_manifest.json")
    for name in ("z_overfit20", "z_mini_train_1200", "z_mini_val_240", "z_paired_val20"):
        entry = manifest["packs"][name]
        assert len(entry["sha256"]) == 64
        path = PACK_ROOT / f"{name}.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], name
        assert entry["bytes"] == path.stat().st_size


def test_no_test_split():
    for name in TASK6Z_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        for marker in ('"test.jsonl"', "'test.jsonl'", 'split="test"', "scene_disjoint_v1/test"):
            assert marker not in text, f"{name}: {marker}"
    for artifact in ("task6z_pack_manifest.json", "task6z_composition_sanity.json",
                     "task6z_overfit20.json", "task6z_training.json", "task6z_mini_val.json",
                     "task6z_paired_val.json", "task6z_verdict.json"):
        payload = _artifact(artifact)
        if payload is not None:
            assert payload.get("test_split_used", False) is False, artifact


# ---------------------------------------------------------------- 25-36 frozen representation and variants


def test_frozen_sam2_visual_path_unchanged():
    assert _git_changed("buildreasonseg_mvp/task6n_relation_decoder.py") == ""
    from buildreasonseg_mvp.task6n_relation_decoder import SAM2_CHECKPOINT, SAM2_CONFIG_NAME

    assert Path(SAM2_CHECKPOINT).is_file()
    assert SAM2_CONFIG_NAME == "configs/sam2.1/sam2.1_hiera_b+.yaml"


def test_z_b0_exact_inputs():
    from buildreasonseg_mvp.task6z_l3_decoder import (
        VARIANT_FUSION_CHANNELS,
        VARIANT_USES_DIRECTIONAL_FIELD,
        VARIANT_USES_NEAREST_FIELD,
        VARIANT_USES_PRODUCT_FIELD,
        L3TargetDecoder,
    )

    assert VARIANT_USES_DIRECTIONAL_FIELD["Z-B0"] is False
    assert VARIANT_USES_NEAREST_FIELD["Z-B0"] is False
    assert VARIANT_USES_PRODUCT_FIELD["Z-B0"] is False
    assert VARIANT_FUSION_CHANNELS["Z-B0"] == 144
    model = L3TargetDecoder("Z-B0")
    assert int(model.trunk.conv1.in_channels) == 144
    assert model(_visual(2), ["left_of", "above"], None, None, None).shape == (2, 1, 64, 64)


def test_z_b1_exact_inputs():
    from buildreasonseg_mvp.task6z_l3_decoder import (
        VARIANT_FUSION_CHANNELS,
        VARIANT_USES_DIRECTIONAL_FIELD,
        VARIANT_USES_NEAREST_FIELD,
        L3TargetDecoder,
    )

    assert VARIANT_USES_DIRECTIONAL_FIELD["Z-B1"] is True
    assert VARIANT_USES_NEAREST_FIELD["Z-B1"] is False
    assert VARIANT_FUSION_CHANNELS["Z-B1"] == 145
    model = L3TargetDecoder("Z-B1")
    assert model(_visual(2), ["left_of", "below"], _field(2), None, None).shape == (2, 1, 64, 64)
    with pytest.raises(ValueError):
        model(_visual(2), ["left_of"], None, _field(2), None)


def test_z_b2_exact_inputs():
    from buildreasonseg_mvp.task6z_l3_decoder import (
        VARIANT_FUSION_CHANNELS,
        VARIANT_USES_DIRECTIONAL_FIELD,
        VARIANT_USES_NEAREST_FIELD,
        L3TargetDecoder,
    )

    assert VARIANT_USES_NEAREST_FIELD["Z-B2"] is True
    assert VARIANT_USES_DIRECTIONAL_FIELD["Z-B2"] is False
    assert VARIANT_FUSION_CHANNELS["Z-B2"] == 145
    model = L3TargetDecoder("Z-B2")
    assert model(_visual(2), ["left_of", "below"], None, _field(2), None).shape == (2, 1, 64, 64)


def test_z_b3_exact_inputs():
    from buildreasonseg_mvp.task6z_l3_decoder import (
        VARIANT_FUSION_CHANNELS,
        VARIANT_USES_DIRECTIONAL_FIELD,
        VARIANT_USES_NEAREST_FIELD,
        VARIANT_USES_PRODUCT_FIELD,
        L3TargetDecoder,
    )

    assert VARIANT_USES_DIRECTIONAL_FIELD["Z-B3"] is True
    assert VARIANT_USES_NEAREST_FIELD["Z-B3"] is True
    assert VARIANT_USES_PRODUCT_FIELD["Z-B3"] is False
    assert VARIANT_FUSION_CHANNELS["Z-B3"] == 146
    model = L3TargetDecoder("Z-B3")
    assert int(model.trunk.conv1.in_channels) == 146
    assert model(_visual(2), ["left_of", "below"], _field(2), _field(2), None).shape == (2, 1, 64, 64)
    report = _variant_report()
    assert report["deterministic_multiplication_before_decoder_z_b3"] is False


def test_z_b4_exact_inputs():
    from buildreasonseg_mvp.task6z_l3_decoder import (
        VARIANT_FUSION_CHANNELS,
        VARIANT_USES_DIRECTIONAL_FIELD,
        VARIANT_USES_NEAREST_FIELD,
        VARIANT_USES_PRODUCT_FIELD,
        L3TargetDecoder,
    )

    assert VARIANT_USES_PRODUCT_FIELD["Z-B4"] is True
    assert VARIANT_USES_DIRECTIONAL_FIELD["Z-B4"] is False
    assert VARIANT_USES_NEAREST_FIELD["Z-B4"] is False
    assert VARIANT_FUSION_CHANNELS["Z-B4"] == 145
    model = L3TargetDecoder("Z-B4")
    assert model(_visual(2), ["left_of", "below"], None, None, _field(2)).shape == (2, 1, 64, 64)


def test_z_b5_no_visual_input():
    from buildreasonseg_mvp.task6z_l3_decoder import (
        VARIANT_FUSION_CHANNELS,
        VARIANT_USES_VISUAL,
        L3TargetDecoder,
    )

    assert VARIANT_USES_VISUAL["Z-B5"] is False
    assert VARIANT_FUSION_CHANNELS["Z-B5"] == 144
    model = L3TargetDecoder("Z-B5")
    assert not hasattr(model, "visual_projection")
    assert model(None, ["left_of", "below"], None, None, _field(2)).shape == (2, 1, 64, 64)
    with pytest.raises(ValueError):
        model(None, ["left_of"], None, None, None)


def test_z_b5_projection_exactly_1_to_128():
    from buildreasonseg_mvp.task6z_l3_decoder import L3TargetDecoder

    model = L3TargetDecoder("Z-B5")
    first = model.field_projection[0]
    assert isinstance(first, torch.nn.Conv2d)
    assert (first.in_channels, first.out_channels) == (1, 128)
    assert isinstance(model.field_projection[1], torch.nn.GroupNorm)
    assert model.field_projection[1].num_groups == 8
    assert model.field_projection[1].num_channels == 128
    assert isinstance(model.field_projection[2], torch.nn.GELU)


def test_exactly_six_variants():
    from buildreasonseg_mvp.task6z_l3_decoder import ALL_VARIANTS

    assert ALL_VARIANTS == ("Z-B0", "Z-B1", "Z-B2", "Z-B3", "Z-B4", "Z-B5")
    report = _variant_report()
    assert report["variant_count"] == 6
    assert set(report["variants"]) == set(ALL_VARIANTS)
    verdict = _artifact("task6z_verdict.json")
    if verdict is not None:
        assert verdict["protocol"]["variant_count"] == 6


def test_direction_embedding_vocab_exactly_4():
    from buildreasonseg_mvp.task6z_l3_decoder import DirectionEmbedding

    embedding = DirectionEmbedding()
    assert embedding.embedding.num_embeddings == 4
    assert embedding.ids == ("left_of", "right_of", "above", "below")


def test_direction_embedding_dim_16():
    from buildreasonseg_mvp.task6z_l3_decoder import DIRECTION_EMBED_DIM, DirectionEmbedding

    assert DIRECTION_EMBED_DIM == 16
    embedding = DirectionEmbedding()
    assert embedding.embedding.embedding_dim == 16
    broadcast = embedding(["left_of", "above"], 64, 64, "cpu")
    assert broadcast.shape == (2, 16, 64, 64)
    assert torch.allclose(broadcast[0, :, 0, 0], broadcast[1, :, 63, 63]) is False


def test_no_separate_learned_nearest_embedding():
    from buildreasonseg_mvp.task6z_l3_decoder import ALL_VARIANTS, L3TargetDecoder

    report = _variant_report()
    assert report["separate_nearest_embedding"] is False
    for variant in ALL_VARIANTS:
        model = L3TargetDecoder(variant)
        assert not hasattr(model, "nearest_embedding"), variant
        assert hasattr(model, "direction_embedding"), variant
    assert all("nearest_embed" not in entry["inputs"]
               for entry in report["variants"].values())


# ---------------------------------------------------------------- 37-41 loss and schedule


def test_common_target_trunk_exact():
    from buildreasonseg_mvp.task6z_l3_decoder import TargetTrunk

    trunk = TargetTrunk(146)
    assert (trunk.conv1.in_channels, trunk.conv1.out_channels, trunk.conv1.kernel_size) == (146, 128,
                                                                                           (3, 3))
    assert trunk.conv1.padding == (1, 1)
    assert isinstance(trunk.norm1, torch.nn.GroupNorm) and trunk.norm1.num_groups == 8
    assert (trunk.conv2.in_channels, trunk.conv2.out_channels) == (128, 64)
    assert isinstance(trunk.norm2, torch.nn.GroupNorm) and trunk.norm2.num_groups == 8
    assert (trunk.head.in_channels, trunk.head.out_channels) == (64, 1)
    assert trunk(torch.zeros(1, 146, 64, 64)).shape == (1, 1, 64, 64)


def test_bce_plus_dice_only():
    from buildreasonseg_mvp.task6n_relation_decoder import task6n_loss

    report = _variant_report()
    assert report["loss"] == "BCEWithLogitsLoss + DiceLoss"
    assert report["extra_losses"] == []
    losses = task6n_loss(torch.zeros(1, 1, 16, 16), torch.zeros(1, 1, 16, 16))
    assert set(losses) == {"loss", "bce", "dice"}


def test_no_grcl():
    report = _variant_report()
    assert report["grcl"] is False
    for name in TASK6Z_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("grcl_directional", "LAMBDA_GRCL", "grcl.loss"):
            assert marker not in code, f"{name}: {marker}"


def test_no_auxiliary_or_ranking_loss():
    report = _variant_report()
    assert report["extra_losses"] == []
    code = _code_only(SCRIPTS / "task6z_train.py")
    for marker in ("ranking", "margin_loss", "aux_loss", "candidate_loss", "triplet",
                   "field_supervision"):
        assert marker not in code, marker


def test_same_schedule_across_variants():
    code = _code_only(SCRIPTS / "task6z_train.py")
    assert "for variant in ALL_VARIANTS" in code
    training = _artifact("task6z_training.json")
    if training is not None:
        assert training["training"]["lr"] == 3.0e-4
        assert training["training"]["weight_decay"] == 1.0e-4
        assert training["training"]["batch"] == 8
        assert training["training"]["max_epochs"] == 25
        assert training["training"]["patience"] == 5
        for variant in ("Z-B0", "Z-B1", "Z-B2", "Z-B3", "Z-B4", "Z-B5"):
            assert training["results"][variant]["steps"] > 0
    overfit = _artifact("task6z_overfit20.json")
    if overfit is not None:
        assert overfit["training"]["lr"] == 1.0e-3
        assert overfit["training"]["batch"] == 4
        assert overfit["training"]["max_steps"] == 1200
        assert overfit["training"]["eval_every"] == 100


# ---------------------------------------------------------------- 42-49 protocol guards


def test_oracle_reference_explicitly_recorded():
    manifest = _artifact("task6z_pack_manifest.json")
    assert manifest["canonical_semantics"]["labels_regenerated"] is False
    for name in ("z_overfit20", "z_mini_train_1200", "z_mini_val_240", "z_paired_val20"):
        pack = _pack(name)
        assert pack["reference_source"] == "oracle_native_gt"
        for record in pack["records"]:
            assert record["metadata"]["reference_source"] == "oracle_native_gt"
    verdict = _artifact("task6z_verdict.json")
    assert verdict["protocol"]["reference_source"] == "oracle_native_gt"


def test_gt_target_only_label_or_evaluation():
    code = _code_only(SCRIPTS / "task6z_train.py")
    assert "masks.mask(record[\"tile_id\"], record[\"target_source_feature_id\"])" in code
    assert "forward_variant(model, batch)" in code


def test_no_candidate_mask_model_input():
    code = _code_only(SCRIPTS / "task6z_train.py")
    for marker in ("candidate.mask", "instance.mask", "candidate_ids", "eligible_ids"):
        assert marker not in code, marker
    verdict = _artifact("task6z_verdict.json")
    assert verdict["protocol"]["candidate_masks_as_input"] is False


def test_no_predicted_or_proposal_reference_in_main_experiment():
    for name in TASK6Z_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("proposals_for_tile(", "QualityReferenceResolver(", "ProposalSetRanker(",
                       "predict_reference(", "import task6q_reference_resolver",
                       "from buildreasonseg_mvp.task6q_reference_resolver import"):
            assert marker not in code, f"{name}: {marker}"
    verdict = _artifact("task6z_verdict.json")
    assert verdict["protocol"]["predicted_reference_used"] is False
    assert verdict["interpretation_boundary"]["predicted_reference_used"] is False


def test_no_programhead_training():
    for name in TASK6Z_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("parse_instruction", "build_program_parser", "load_parser_checkpoint"):
            assert marker not in code, f"{name}: {marker}"
    assert _git_changed("buildreasonseg_mvp/program_parser.py") == ""


def test_no_attention_transformer_or_gnn():
    report = _variant_report()
    for key in ("attention", "transformer", "gnn"):
        assert report[key] is False, key
    code = _code_only(DECODER_MODULE)
    for marker in ("MultiheadAttention", "TransformerEncoder", "nn.Transformer",
                   "MessagePassing", "Attention("):
        assert marker not in code, marker


def test_no_new_dataset_download_install_or_gui():
    for name in TASK6Z_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("urlretrieve", "requests.get", "hf_hub_download", "snapshot_download",
                       "pip install", "load_dataset", "tkinter", "gradio", "streamlit", "flask"):
            assert marker not in code, f"{name}: {marker}"


def test_required_modules_and_artifacts_exist():
    for path in (COMPOSITION_MODULE, DECODER_MODULE,
                 REPO_ROOT / "docs" / "task6z_oracle_l3_composition.md"):
        assert path.is_file(), path
    for name in ("task6z_pack_manifest.json", "task6z_composition_sanity.json",
                 "task6z_overfit20.json", "task6z_training.json", "task6z_mini_val.json",
                 "task6z_paired_val.json", "task6z_verdict.json"):
        assert (EVAL / name).is_file(), name


def test_previous_suite_preserved():
    for name in ("test_task6y_nearest_boundary_field.py", "test_task6x_sam2_refinement.py",
                 "test_task6w_proposal_quality.py", "test_task6v_family_reference_resolver.py"):
        assert (REPO_ROOT / "tests" / name).is_file(), name
    removed = subprocess.run(["git", "diff", "--name-only", "--diff-filter=D", BASE_COMMIT, "--",
                              "tests/"], cwd=REPO_ROOT, capture_output=True, text=True,
                             check=True).stdout
    assert removed.strip() == "", f"no test file may be removed: {removed}"


def test_program_to_relation_mapping_exact():
    from buildreasonseg_mvp.task6z_field_composition import (
        DIRECTION_IDS,
        PROGRAM_TO_RELATION,
        direction_index,
        relation_of,
    )

    assert PROGRAM_TO_RELATION == {
        "largest_to_left_of_to_nearest": "left_of",
        "largest_to_right_of_to_nearest": "right_of",
        "largest_to_above_to_nearest": "above",
        "largest_to_below_to_nearest": "below",
    }
    assert direction_index("largest_to_above_to_nearest") == 2
    assert relation_of("largest_to_below_to_nearest") == "below"
    assert DIRECTION_IDS == ("left_of", "right_of", "above", "below")
    with pytest.raises(ValueError):
        relation_of("smallest_to_left_of_to_nearest")


def _visual(batch: int) -> torch.Tensor:
    return torch.zeros(batch, 256, 64, 64)


def _field(batch: int) -> torch.Tensor:
    return torch.zeros(batch, 1, 64, 64)


def _variant_report() -> dict:
    from buildreasonseg_mvp.task6z_l3_decoder import variant_report

    return variant_report()


def _composition_report() -> dict:
    from buildreasonseg_mvp.task6z_field_composition import field_report

    return field_report()
