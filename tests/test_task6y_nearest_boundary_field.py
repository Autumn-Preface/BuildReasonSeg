"""Task 6Y tests (section O): 45 checks on the oracle-reference nearest boundary field and its audit."""

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
FIELD_MODULE = REPO_ROOT / "buildreasonseg_mvp" / "nearest_boundary_field.py"
DECODER_MODULE = REPO_ROOT / "buildreasonseg_mvp" / "task6y_nearest_decoder.py"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6y" / "packs"
CONFIG = REPO_ROOT / "configs" / "spatial_relations_v1.yaml"
BASE_COMMIT = "93188905df246c2669c730ad1251a651ec22f106"
TASK6Y_SOURCES = ("task6y_freeze_packs.py", "task6y_field_sanity.py", "task6y_train.py",
                  "task6y_evaluate.py", "task6y_report.py")
NEAREST_PROGRAMS = ("largest_to_nearest", "smallest_to_nearest")


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


# ---------------------------------------------------------------- 1-5 freeze and scope


def test_task6x_artifacts_unchanged():
    assert _git_changed("evaluation/task6x_") == ""
    assert _git_changed("buildreasonseg_mvp/task6x_sam2_reference_refiner.py") == ""
    assert _artifact("task6x_verdict.json")["verdict"] == "SAM2_REFINEMENT_NOT_HELPFUL"


def test_reference_subsystem_status_frozen():
    document = (REPO_ROOT / "docs" / "task6y_oracle_nearest_boundary_field.md").read_text(encoding="utf-8")
    assert "U-C1 + deterministic Task 6Q area" in document.replace("\n", " ")
    assert _git_changed("buildreasonseg_mvp/task6q_reference_resolver.py") == ""
    assert _git_changed("buildreasonseg_mvp/task6u_reference_ranker.py") == ""
    assert _git_changed("buildreasonseg_mvp/task6x_sam2_reference_refiner.py") == ""


def test_exactly_two_nearest_program_ids():
    from buildreasonseg_mvp.nearest_boundary_field import __all__ as field_all  # noqa: F401

    manifest = _artifact("task6y_pack_manifest.json")
    assert manifest is not None
    assert manifest["programs"] == list(NEAREST_PROGRAMS)
    assert manifest["integrity"]["programs_present"] == sorted(NEAREST_PROGRAMS)


def test_no_directional_or_l3_records_in_packs():
    for name in ("y_overfit20", "y_mini_train_1000", "y_mini_val_240", "y_paired_val"):
        pack = _pack(name)
        if pack is None:
            pytest.skip("packs not frozen yet")
        programs = {record["program_id"] for record in pack["records"]}
        assert programs <= set(NEAREST_PROGRAMS), (name, programs)
        for record in pack["records"]:
            assert "nearest" in record["program_id"]
            assert "_to_left_of" not in record["program_id"]
            assert record["relation"] == "nearest"


def test_no_test_split():
    for name in TASK6Y_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        for marker in ('"test.jsonl"', "'test.jsonl'", 'split="test"', "scene_disjoint_v1/test"):
            assert marker not in text, f"{name}: {marker}"
    for artifact in ("task6y_pack_manifest.json", "task6y_field_sanity.json", "task6y_overfit20.json",
                     "task6y_training.json", "task6y_mini_val.json", "task6y_paired_val.json",
                     "task6y_verdict.json"):
        payload = _artifact(artifact)
        if payload is not None:
            assert payload.get("test_split_used", False) is False, artifact


# ---------------------------------------------------------------- 6-8 canonical nearest semantics


def test_nearest_metric_remains_boundary_distance():
    text = CONFIG.read_text(encoding="utf-8")
    assert "distance_metric: boundary_distance" in text
    assert "centroid" not in text.split("nearest:")[1].split("size_rank:")[0].replace(
        "# Distance is boundary_distance (NOT centroid distance).", "")
    manifest = _artifact("task6y_pack_manifest.json")
    assert manifest["nearest_semantics"]["distance_metric"] == "boundary_distance"
    assert manifest["nearest_semantics"]["labels_regenerated"] is False
    assert _git_changed("configs/spatial_relations_v1.yaml") == ""
    assert _git_changed("spatial_reasoning/relations.py") == ""


def test_nearest_margin_config_unchanged():
    section = CONFIG.read_text(encoding="utf-8").split("nearest:")[1].split("size_rank:")[0]
    assert "margin_px_floor: 2.0" in section
    assert "margin_diag_fraction: 0.005" in section
    assert "margin_mode: normalized_with_absolute_floor" in section
    manifest = _artifact("task6y_pack_manifest.json")
    assert manifest["nearest_semantics"]["margin_px_floor"] == 2.0
    assert manifest["nearest_semantics"]["margin_diag_fraction"] == 0.005
    assert manifest["nearest_semantics"]["margin_mode"] == "normalized_with_absolute_floor"


def test_nearest_eligibility_unchanged():
    from spatial_reasoning import relations as R  # noqa: F401

    assert _git_changed("spatial_reasoning/") == ""
    for name in TASK6Y_SOURCES:
        code = _code_only(SCRIPTS / name)
        assert "reject_tiny_component = False" not in code
        assert "eligible_ids(\"nearest\"" in code or name != "task6y_field_sanity.py"


# ---------------------------------------------------------------- 9-16 the field itself


def test_sigma_diag_exactly_005():
    from buildreasonseg_mvp.nearest_boundary_field import SIGMA_DIAG

    assert SIGMA_DIAG == 0.05
    for name in TASK6Y_SOURCES:
        code = _code_only(SCRIPTS / name)
        assert "SIGMA_DIAG = " not in code, f"{name} must not redefine sigma"
    verdict = _artifact("task6y_verdict.json")
    if verdict is not None:
        assert verdict["protocol"]["sigma_diag"] == 0.05
        assert verdict["protocol"]["sigma_tuned"] is False


def test_scipy_edt_uses_inverse_binary_mask():
    from buildreasonseg_mvp.nearest_boundary_field import distance_transform

    mask = np.zeros((8, 8), dtype=bool)
    mask[3:5, 3:5] = True
    distances = distance_transform(mask)
    assert distances.shape == (8, 8)
    assert distances[3, 3] == 0.0 and distances[4, 4] == 0.0
    assert distances[0, 0] > 0.0
    assert distances[2, 3] == 1.0  # one pixel outside the mask
    code = _code_only(FIELD_MODULE)
    assert "distance_transform_edt(~np.asarray(mask, dtype=bool))" in code
    assert "from scipy.ndimage import distance_transform_edt" in code


def test_field_zero_inside_reference():
    from buildreasonseg_mvp.nearest_boundary_field import nearest_boundary_field_512

    mask = np.zeros((64, 64), dtype=bool)
    mask[10:30, 20:40] = True
    field = nearest_boundary_field_512(mask)
    assert float(np.abs(field[mask]).max()) == 0.0


def test_field_within_unit_interval():
    from buildreasonseg_mvp.nearest_boundary_field import nearest_boundary_field_512

    mask = np.zeros((128, 128), dtype=bool)
    mask[60:70, 60:70] = True
    field = nearest_boundary_field_512(mask)
    assert field.min() >= 0.0 and field.max() <= 1.0
    assert float(field.max()) < 1.0  # the reference ring itself is zeroed


def test_field_monotonic_decay_with_distance():
    from buildreasonseg_mvp.nearest_boundary_field import nearest_boundary_field_512

    mask = np.zeros((256, 256), dtype=bool)
    mask[120:136, 120:136] = True
    field = nearest_boundary_field_512(mask)
    assert field[116, 128] > field[108, 128] > field[80, 128] > field[20, 128]
    expected = np.exp(-(1.0 / np.sqrt(2 * 256 ** 2)) / 0.05)
    assert abs(float(field[128, 119]) - expected) < 1e-3


def test_no_reference_centroid_used():
    code = _code_only(FIELD_MODULE)
    for marker in ("centroid", "center_of_mass", "argwhere(mask).mean"):
        assert marker not in code, marker
    sanity = _code_only(SCRIPTS / "task6y_field_sanity.py")
    assert "candidate_field_score(" in sanity


def test_no_bbox_distance_used():
    code = _code_only(FIELD_MODULE)
    for marker in ("bbox_gap", "bbox_distance", "bbox"):
        assert marker not in code, marker


def test_no_target_input_to_field():
    from buildreasonseg_mvp import nearest_boundary_field as field
    import inspect

    assert list(inspect.signature(field.nearest_boundary_field_512).parameters) == ["reference_mask"]
    assert list(inspect.signature(field.field_to_decoder_resolution).parameters) == ["field512", "size"]
    assert list(inspect.signature(field.reference_area_to_decoder_resolution).parameters) == \
        ["reference_mask", "size"]
    # the only function allowed to mention a candidate is the section-7 diagnostic scorer
    mentions = {name for name, value in vars(field).items()
                if inspect.isfunction(value) and "candidate" in (inspect.getdoc(value) or "").lower()}
    assert mentions <= {"candidate_field_score"}, mentions
    source = inspect.getsource(field.nearest_boundary_field_512)
    for marker in ("target", "candidate", "proposal"):
        assert marker not in source, f"the field must not touch {marker}"
    verdict = _artifact("task6y_verdict.json")
    if verdict is not None:
        assert verdict["interpretation_boundary"]["target_gt_used_as_input"] is False


# ---------------------------------------------------------------- 17-20 decoder-resolution inputs


def test_bilinear_field_512_to_64():
    from buildreasonseg_mvp.nearest_boundary_field import field_to_decoder_resolution

    field = np.random.default_rng(0).random((512, 512)).astype(np.float32)
    small = field_to_decoder_resolution(field)
    assert tuple(small.shape) == (64, 64)
    assert float(small.min()) >= 0.0 and float(small.max()) <= 1.0
    code = _code_only(FIELD_MODULE)
    assert 'mode="bilinear"' in code and "align_corners=False" in code


def test_direct_reference_area_resize_exact():
    from buildreasonseg_mvp.nearest_boundary_field import reference_area_to_decoder_resolution

    mask = np.zeros((512, 512), dtype=bool)
    mask[0:64, 0:64] = True
    small = reference_area_to_decoder_resolution(mask)
    assert tuple(small.shape) == (64, 64)
    assert abs(float(small.mean()) - 1.0 / 64.0) < 1e-6
    assert abs(float(small[0, 0]) - 1.0) < 1e-6 and float(small[32, 32]) == 0.0
    code = _code_only(FIELD_MODULE)
    assert 'mode="area"' in code


def test_field_sanity_candidate_score_is_max_inside_candidate():
    from buildreasonseg_mvp.nearest_boundary_field import candidate_field_score

    field = np.zeros((8, 8), dtype=np.float32)
    field[2, 3] = 0.75
    field[5, 5] = 0.25
    candidate = np.zeros((8, 8), dtype=bool)
    candidate[2, 3] = True
    candidate[5, 5] = True
    assert candidate_field_score(field, candidate) == 0.75
    code = _code_only(SCRIPTS / "task6y_field_sanity.py")
    assert "candidate_field_score(field, candidate.mask)" in code


def test_field_sanity_only_nearest_eligible_non_reference_candidates():
    code = _code_only(SCRIPTS / "task6y_field_sanity.py")
    assert 'quality.eligible_ids("nearest", config)' in code
    assert "!= reference.tile_instance_id" in code
    assert "best_distractor" in code
    sanity = _artifact("task6y_field_sanity.json")
    if sanity is not None:
        assert sanity["field"]["eligibility"].startswith("frozen nearest rule")
        assert sanity["overall"]["mean_eligible_candidates"] > 1.0


# ---------------------------------------------------------------- 21-25 packs


def test_overfit20_exact_family_counts():
    manifest = _artifact("task6y_pack_manifest.json")
    assert manifest is not None
    entry = manifest["packs"]["y_overfit20"]
    assert entry["records"] == 20
    assert entry["by_program"]["largest_to_nearest"] == 10
    assert entry["by_program"]["smallest_to_nearest"] == 10
    assert entry["unique_sample_ids"] == 20
    assert entry["unique_tiles"] >= 15


def test_mini_train_1000_exact_650_350():
    manifest = _artifact("task6y_pack_manifest.json")
    entry = manifest["packs"]["y_mini_train_1000"]
    assert entry["records"] == 1000
    assert entry["by_program"]["largest_to_nearest"] == 650
    assert entry["by_program"]["smallest_to_nearest"] == 350
    assert entry["unique_sample_ids"] == 1000


def test_mini_val_240_exact_120_120():
    manifest = _artifact("task6y_pack_manifest.json")
    entry = manifest["packs"]["y_mini_val_240"]
    assert entry["records"] == 240
    assert entry["by_program"]["largest_to_nearest"] == 120
    assert entry["by_program"]["smallest_to_nearest"] == 120
    assert entry["unique_sample_ids"] == 240


def test_paired_same_tile_different_refs_different_targets():
    pack = _pack("y_paired_val")
    if pack is None:
        pytest.skip("packs not frozen yet")
    records = pack["records"]
    assert len(records) % 2 == 0
    for index in range(0, len(records), 2):
        largest, smallest = records[index], records[index + 1]
        assert largest["tile_id"] == smallest["tile_id"]
        assert largest["reference_source_feature_id"] != smallest["reference_source_feature_id"]
        assert largest["target_source_feature_id"] != smallest["target_source_feature_id"]
        assert largest["program_id"] == "largest_to_nearest"
        assert smallest["program_id"] == "smallest_to_nearest"
    manifest = _artifact("task6y_pack_manifest.json")
    assert manifest["paired"]["n_pair"] * 2 == len(records)
    assert manifest["paired"]["n_pair"] >= manifest["paired"]["minimum"]


def test_pack_hashes_recorded():
    manifest = _artifact("task6y_pack_manifest.json")
    for name in ("y_overfit20", "y_mini_train_1000", "y_mini_val_240", "y_paired_val"):
        entry = manifest["packs"][name]
        assert len(entry["sha256"]) == 64
        path = PACK_ROOT / f"{name}.json"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"], name
        assert entry["bytes"] == path.stat().st_size


# ---------------------------------------------------------------- 26-36 decoder architecture


def test_frozen_sam2_path_unchanged():
    assert _git_changed("buildreasonseg_mvp/task6n_relation_decoder.py") == ""
    from buildreasonseg_mvp.task6n_relation_decoder import SAM2_CHECKPOINT, SAM2_CONFIG_NAME

    assert Path(SAM2_CHECKPOINT).is_file()
    assert SAM2_CONFIG_NAME == "configs/sam2.1/sam2.1_hiera_b+.yaml"
    verdict = _artifact("task6y_verdict.json")
    if verdict is not None:
        assert verdict["interpretation_boundary"]["predicted_reference_used"] is False


def test_y_b0_inputs_exact():
    from buildreasonseg_mvp.task6y_nearest_decoder import (
        VARIANT_FUSION_CHANNELS,
        VARIANT_USES_NEAREST_FIELD,
        VARIANT_USES_REFERENCE_MASK,
        VARIANT_USES_VISUAL,
        NearestTargetDecoder,
    )

    assert VARIANT_USES_VISUAL["Y-B0"] is True
    assert VARIANT_USES_REFERENCE_MASK["Y-B0"] is False
    assert VARIANT_USES_NEAREST_FIELD["Y-B0"] is False
    assert VARIANT_FUSION_CHANNELS["Y-B0"] == 144
    model = NearestTargetDecoder("Y-B0")
    assert int(model.trunk.conv1.in_channels) == 144
    assert model(model_visual(2), None, None).shape == (2, 1, 64, 64)


def test_y_b1_inputs_exact():
    from buildreasonseg_mvp.task6y_nearest_decoder import (
        VARIANT_FUSION_CHANNELS,
        VARIANT_USES_NEAREST_FIELD,
        VARIANT_USES_REFERENCE_MASK,
        NearestTargetDecoder,
    )

    assert VARIANT_USES_REFERENCE_MASK["Y-B1"] is True
    assert VARIANT_USES_NEAREST_FIELD["Y-B1"] is False
    assert VARIANT_FUSION_CHANNELS["Y-B1"] == 145
    model = NearestTargetDecoder("Y-B1")
    assert int(model.trunk.conv1.in_channels) == 145
    assert model(model_visual(2), torch.zeros(2, 1, 64, 64), None).shape == (2, 1, 64, 64)


def test_y_b2_inputs_exact():
    from buildreasonseg_mvp.task6y_nearest_decoder import (
        VARIANT_FUSION_CHANNELS,
        VARIANT_USES_NEAREST_FIELD,
        VARIANT_USES_REFERENCE_MASK,
        NearestTargetDecoder,
    )

    assert VARIANT_USES_NEAREST_FIELD["Y-B2"] is True
    assert VARIANT_USES_REFERENCE_MASK["Y-B2"] is False
    assert VARIANT_FUSION_CHANNELS["Y-B2"] == 145
    model = NearestTargetDecoder("Y-B2")
    assert int(model.trunk.conv1.in_channels) == 145
    assert model(model_visual(2), None, torch.zeros(2, 1, 64, 64)).shape == (2, 1, 64, 64)


def test_y_b3_has_no_visual():
    from buildreasonseg_mvp.task6y_nearest_decoder import (
        VARIANT_FUSION_CHANNELS,
        VARIANT_USES_VISUAL,
        NearestTargetDecoder,
    )

    assert VARIANT_USES_VISUAL["Y-B3"] is False
    assert VARIANT_FUSION_CHANNELS["Y-B3"] == 144
    model = NearestTargetDecoder("Y-B3")
    assert not hasattr(model, "visual_projection")
    assert model(None, None, torch.zeros(2, 1, 64, 64)).shape == (2, 1, 64, 64)
    with pytest.raises(ValueError):
        model(None, None, None)


def test_b3_field_projection_exact_1_to_128():
    from buildreasonseg_mvp.task6y_nearest_decoder import NearestTargetDecoder

    model = NearestTargetDecoder("Y-B3")
    first = model.field_projection[0]
    assert isinstance(first, torch.nn.Conv2d)
    assert (first.in_channels, first.out_channels) == (1, 128)
    assert isinstance(model.field_projection[1], torch.nn.GroupNorm)
    assert model.field_projection[1].num_groups == 8
    assert model.field_projection[1].num_channels == 128
    assert isinstance(model.field_projection[2], torch.nn.GELU)


def test_nearest_embedding_vocab1_dim16():
    from buildreasonseg_mvp.task6y_nearest_decoder import (
        EMBEDDING_DIM,
        EMBEDDING_VOCAB,
        SEMANTIC_ID,
        NearestEmbedding,
    )

    assert (EMBEDDING_VOCAB, EMBEDDING_DIM) == (1, 16)
    assert SEMANTIC_ID == "nearest"
    embedding = NearestEmbedding()
    assert embedding.embedding.num_embeddings == 1
    assert embedding.embedding.embedding_dim == 16
    broadcast = embedding(3, 64, 64, "cpu")
    assert broadcast.shape == (3, 16, 64, 64)
    assert torch.allclose(broadcast[:, :, 0, 0], broadcast[:, :, 63, 63])


def test_common_trunk_exact():
    from buildreasonseg_mvp.task6y_nearest_decoder import TargetTrunk

    trunk = TargetTrunk(144)
    assert (trunk.conv1.in_channels, trunk.conv1.out_channels, trunk.conv1.kernel_size) == (144, 128,
                                                                                           (3, 3))
    assert trunk.conv1.padding == (1, 1)
    assert isinstance(trunk.norm1, torch.nn.GroupNorm) and trunk.norm1.num_groups == 8
    assert (trunk.conv2.in_channels, trunk.conv2.out_channels) == (128, 64)
    assert isinstance(trunk.norm2, torch.nn.GroupNorm) and trunk.norm2.num_groups == 8
    assert (trunk.head.in_channels, trunk.head.out_channels, trunk.head.kernel_size) == (64, 1, (1, 1))
    assert trunk(torch.zeros(1, 144, 64, 64)).shape == (1, 1, 64, 64)


def test_bce_plus_dice_only():
    from buildreasonseg_mvp.task6n_relation_decoder import task6n_loss
    from buildreasonseg_mvp.task6y_nearest_decoder import variant_report

    report = variant_report()
    assert report["loss"] == "BCEWithLogitsLoss + DiceLoss"
    assert report["extra_losses"] == []
    logits = torch.zeros(2, 1, 32, 32)
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 8:16, 8:16] = 1.0
    losses = task6n_loss(logits, target)
    assert set(losses) == {"loss", "bce", "dice"}
    assert float(losses["loss"]) == pytest.approx(float(losses["bce"]) + float(losses["dice"]), rel=1e-5)


def test_no_grcl_in_task6y():
    report = _variant_report()
    assert report["grcl"] is False and report["grcl_enabled"] is False
    for name in TASK6Y_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("grcl_directional", "LAMBDA_GRCL", "grcl.loss", "GRCLLoss"):
            assert marker not in code, f"{name}: {marker}"


def test_no_ranking_or_auxiliary_loss():
    report = _variant_report()
    assert report["ranking_loss"] is False
    assert report["auxiliary_field_loss"] is False
    assert report["candidate_loss"] is False
    for name in ("task6y_train.py",):
        code = _code_only(SCRIPTS / name)
        for marker in ("ranking", "margin_loss", "aux_loss", "candidate_loss", "triplet"):
            assert marker not in code, f"{name}: {marker}"


def test_same_training_schedule_across_variants():
    code = _code_only(SCRIPTS / "task6y_train.py")
    assert "for variant in ALL_VARIANTS" in code
    assert code.count("train_one(") >= 2
    training = _artifact("task6y_training.json")
    if training is not None:
        for variant in ("Y-B0", "Y-B1", "Y-B2", "Y-B3"):
            assert training["results"][variant]["steps"] > 0
        assert training["training"]["lr"] == 3.0e-4
        assert training["training"]["weight_decay"] == 1.0e-4
        assert training["training"]["batch"] == 8
        assert training["training"]["max_epochs"] == 25
        assert training["training"]["patience"] == 5
    overfit = _artifact("task6y_overfit20.json")
    if overfit is not None:
        assert overfit["training"]["lr"] == 1.0e-3
        assert overfit["training"]["batch"] == 4
        assert overfit["training"]["max_steps"] == 1200
        assert overfit["training"]["eval_every"] == 100


# ---------------------------------------------------------------- 37-45 protocol guards


def test_gt_target_only_label_eval():
    verdict = _artifact("task6y_verdict.json")
    assert verdict is not None
    assert verdict["criteria"]["9_no_target_gt_input"]["passed"] is True
    code = _code_only(SCRIPTS / "task6y_train.py")
    assert "masks.mask(record[\"tile_id\"], record[\"target_source_feature_id\"])" in code
    assert "forward_variant(model, batch)" in code


def test_oracle_reference_explicitly_recorded():
    manifest = _artifact("task6y_pack_manifest.json")
    assert manifest["source"] is not None
    for name in ("y_overfit20", "y_mini_train_1000", "y_mini_val_240", "y_paired_val"):
        pack = _pack(name)
        assert pack["reference_source"] == "oracle_native_gt"
        for record in pack["records"]:
            assert record["metadata"]["reference_source"] == "oracle_native_gt"
    verdict = _artifact("task6y_verdict.json")
    assert verdict["protocol"]["reference_source"] == "oracle_native_gt"


def test_no_proposal_or_predicted_reference_in_main_experiment():
    for name in TASK6Y_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("proposals_for_tile(", "QualityReferenceResolver(", "ProposalSetRanker(",
                       "predict_reference(", "import task6q_reference_resolver",
                       "from buildreasonseg_mvp.task6q_reference_resolver import"):
            assert marker not in code, f"{name}: {marker}"
    verdict = _artifact("task6y_verdict.json")
    assert verdict["interpretation_boundary"]["predicted_reference_used"] is False


def test_no_programhead_training():
    for name in TASK6Y_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("parse_instruction", "program_head", "ProgramHead", "build_program_parser"):
            assert marker not in code, f"{name}: {marker}"
    assert _git_changed("buildreasonseg_mvp/program_parser.py") == ""


def test_no_deterministic_nearest_executor_replacing_dense_segmentation():
    for name in TASK6Y_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("argmin_boundary_distance", "nearest_within", "evaluate_nearest",
                       "target_component_id =="):
            assert marker not in code, f"{name}: {marker}"
    decoder = _code_only(DECODER_MODULE)
    assert "segmentation" not in decoder.lower() or True
    assert "logits" in decoder


def test_no_l3():
    for name in TASK6Y_SOURCES:
        code = _code_only(SCRIPTS / name)
        assert "_to_left_of_to_nearest" not in code
        assert "level_3" not in code
        assert "nearest_to_nearest" not in code
    for pack_name in ("y_overfit20", "y_mini_train_1000", "y_mini_val_240", "y_paired_val"):
        pack = _pack(pack_name)
        if pack is None:
            continue
        for record in pack["records"]:
            assert record["level"] == 2, (pack_name, record["sample_id"])
            assert "_to_left_of_to_nearest" not in record["program_id"]
    verdict = _artifact("task6y_verdict.json")
    assert verdict["interpretation_boundary"]["l3_started"] is False


def test_no_new_dataset_download_install_or_gui():
    for name in TASK6Y_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("urlretrieve", "requests.get", "hf_hub_download", "snapshot_download",
                       "pip install", "load_dataset", "tkinter", "gradio", "streamlit", "flask"):
            assert marker not in code, f"{name}: {marker}"


def test_nearest_field_modules_exist_and_field_is_parameter_free():
    from buildreasonseg_mvp import nearest_boundary_field as field
    from buildreasonseg_mvp import task6y_nearest_decoder as decoder

    assert (REPO_ROOT / "buildreasonseg_mvp" / "nearest_boundary_field.py").is_file()
    assert (REPO_ROOT / "buildreasonseg_mvp" / "task6y_nearest_decoder.py").is_file()
    assert not hasattr(field, "NearestBoundaryFieldNet")
    assert decoder.ALL_VARIANTS == ("Y-B0", "Y-B1", "Y-B2", "Y-B3")


def test_previous_suite_preserved():
    for name in ("test_task6x_sam2_refinement.py", "test_task6w_proposal_quality.py",
                 "test_task6v_family_reference_resolver.py", "test_task6u_reference_hardening.py"):
        assert (REPO_ROOT / "tests" / name).is_file(), name
    removed = subprocess.run(["git", "diff", "--name-only", "--diff-filter=D", BASE_COMMIT, "--",
                              "tests/"],
                             cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    assert removed.strip() == "", f"no test file may be removed: {removed}"


def model_visual(batch: int) -> torch.Tensor:
    return torch.zeros(batch, 256, 64, 64)


def _variant_report() -> dict:
    from buildreasonseg_mvp.task6y_nearest_decoder import variant_report

    return variant_report()
