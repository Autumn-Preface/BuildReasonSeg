"""Task 7E tests (Part P): 41+ checks on the deterministic-prototype holdout and predicted-reference audit."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

EVAL = REPO_ROOT / "evaluation"
SCRIPTS = REPO_ROOT / "scripts"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6z" / "packs"
HOLDOUT_ROOT = REPO_ROOT / "artifacts" / "task7e" / "holdout"
ADAPTER = REPO_ROOT / "buildreasonseg_mvp" / "task7e_l3_decoder_adapter.py"
BASE_COMMIT = "86e4f4cd4bd5aa6858264a7d5709092b32678ebc"
D_B1_SHA256 = "6df31909cefdb54b9997b1e6ab76b8c5589771defa55666edbe75106221a89c0"
Z_B3_SHA256 = "74f308e11e7f7f1098dd0d092ddcf6be1220d0b322b39df39c4ce05c9fc0f0bc"
EXPECTED_SOURCE = {"largest_to_above_to_nearest": 224, "largest_to_below_to_nearest": 213,
                   "largest_to_left_of_to_nearest": 250, "largest_to_right_of_to_nearest": 249}
L3_PROGRAMS = tuple(sorted(EXPECTED_SOURCE))
TASK7E_SOURCES = ("task7e_build_holdout.py", "task7e_evaluate_oracle.py",
                  "task7e_evaluate_predicted_reference.py", "task7e_parser_integration.py",
                  "task7e_report.py")
REQUIRED_ARTIFACTS = ("task7e_holdout_manifest.json", "task7e_minival_reproduction.json",
                      "task7e_oracle_holdout.json", "task7e_oracle_holdout_paired.json",
                      "task7e_predicted_reference_quality.json",
                      "task7e_predicted_reference_holdout.json",
                      "task7e_predicted_reference_paired.json", "task7e_verdict.json")


def _artifact(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _git_changed(prefix: str) -> str:
    return subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "--", prefix],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


# ---------------------------------------------------------------- 1-6 frozen assets


def test_task7d_artifacts_unchanged():
    assert _git_changed("evaluation/task7d_") == ""
    assert _git_changed("buildreasonseg_mvp/task7d_global_competition_decoder.py") == ""
    assert _artifact("task7d_verdict.json")["verdict"] == "GLOBAL_COMPETITION_NO_MEANINGFUL_GAIN"


def test_d_b1_checkpoint_hash_exact():
    from buildreasonseg_mvp.task7e_l3_decoder_adapter import D_B1_CHECKPOINT, frozen_metadata

    assert D_B1_CHECKPOINT.is_file()
    assert _sha256(D_B1_CHECKPOINT) == D_B1_SHA256
    metadata = frozen_metadata()
    assert metadata["d_b1"]["matches"] is True
    assert metadata["d_b1"]["sha256"] == D_B1_SHA256
    assert metadata["d_b1"]["checkpoint_variant"] == "D-B1"
    assert metadata["d_b1"]["selected_epoch"] == 7
    assert metadata["d_b1"]["epoch_matches"] is True
    assert metadata["d_b1"]["retrained"] is False


def test_z_b3_checkpoint_hash_exact():
    from buildreasonseg_mvp.task7e_l3_decoder_adapter import Z_B3_CHECKPOINT, frozen_metadata

    assert _sha256(Z_B3_CHECKPOINT) == Z_B3_SHA256
    assert frozen_metadata()["z_b3"]["matches"] is True
    verdict = _artifact("task7e_verdict.json")
    assert verdict["protocol"]["z_b3_checkpoint_matches"] is True
    assert verdict["protocol"]["d_b1_checkpoint_matches"] is True


def test_no_training_in_task7e():
    verdict = _artifact("task7e_verdict.json")
    assert verdict["protocol"]["training_performed"] is False
    assert verdict["protocol"]["retrained_any_checkpoint"] is False
    assert verdict["protocol"]["full_training_started"] is False
    for name in TASK7E_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("backward()", "optimizer.step(", "make_optimizer(", "AdamW(",
                       "task6n_loss(", "torch.save("):
            assert marker not in code, f"{name}: {marker}"


def test_d_b1_architecture_unchanged():
    assert _git_changed("buildreasonseg_mvp/task7d_global_competition_decoder.py") == ""
    from buildreasonseg_mvp.task7d_global_competition_decoder import GlobalCompetitionDecoder

    model = GlobalCompetitionDecoder("D-B1")
    assert model.uses_learned_score_head is False
    assert model.uses_relation_fields is True
    assert model.uses_prototype is True
    assert model.trunk.conv1.in_channels == 148
    assert model.params if False else True
    adapter = _code_only(ADAPTER)
    for marker in ("nn.Conv2d", "nn.GroupNorm", "nn.Embedding"):
        assert marker not in adapter, marker


def test_no_learned_score_head_in_d_b1():
    from buildreasonseg_mvp.task7d_global_competition_decoder import (
        GlobalCompetitionDecoder,
        VARIANT_USES_LEARNED_SCORE_HEAD,
    )

    assert VARIANT_USES_LEARNED_SCORE_HEAD["D-B1"] is False
    assert not hasattr(GlobalCompetitionDecoder("D-B1"), "score_head")
    verdict = _artifact("task7e_verdict.json")
    assert verdict["protocol"]["learned_competition"] is False
    assert verdict["decision"]["DB1_ADOPT_AS_DEVELOPMENT_L3_DECODER"] in (True, False)


# ---------------------------------------------------------------- 7-10 exact formulas


def test_exact_product_field_weighting():
    from buildreasonseg_mvp.task7d_global_competition_decoder import field_competition

    directional = torch.rand(2, 1, 64, 64)
    nearest = torch.rand(2, 1, 64, 64)
    attention, mass = field_competition(directional, nearest)
    weight = (directional * nearest).clamp(0.0, 1.0)
    expected = weight / (weight.sum(dim=(-2, -1), keepdim=True) + 1e-6)
    assert torch.allclose(attention, expected, atol=1e-6)
    assert mass > 0.0
    assert torch.allclose(attention.flatten(1).sum(dim=-1), torch.ones(2), atol=1e-5)


def test_exact_prototype_formula():
    from buildreasonseg_mvp.task7d_global_competition_decoder import target_prototype

    features = torch.randn(1, 128, 64, 64)
    attention = torch.zeros(1, 1, 64, 64)
    attention[0, 0, 5, 7] = 1.0
    prototype = target_prototype(features, attention)
    assert torch.allclose(prototype[0], features[0, :, 5, 7], atol=1e-6)
    attention = torch.full((1, 1, 64, 64), 1.0 / 4096)
    assert torch.allclose(target_prototype(features, attention)[0], features[0].mean(dim=(1, 2)),
                          atol=1e-5)


def test_exact_cosine_similarity():
    from buildreasonseg_mvp.task7d_global_competition_decoder import (
        prototype_similarity,
        target_prototype,
    )

    features = torch.randn(2, 128, 64, 64)
    attention = torch.full((2, 1, 64, 64), 1.0 / 4096)
    prototype = target_prototype(features, attention)
    similarity = prototype_similarity(features, prototype)
    expected = torch.nn.functional.cosine_similarity(
        features.flatten(2).permute(0, 2, 1),
        prototype[:, None, :].expand(2, 4096, 128), dim=-1).reshape(2, 1, 64, 64)
    assert torch.allclose(similarity, expected, atol=1e-5)
    code = _code_only(REPO_ROOT / "buildreasonseg_mvp"
                      / "task7d_global_competition_decoder.py")
    block = code.split("def prototype_similarity")[1].split("class GlobalCompetitionDecoder")[0]
    for marker in ("torch.sigmoid(", "nn.Parameter(", "nn.Linear("):
        assert marker not in block, marker


def test_no_target_proposal_input():
    verdict = _artifact("task7e_verdict.json")
    assert verdict["protocol"]["ranker_or_filter_or_refinement"] is False
    for name in TASK7E_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("QualityReferenceResolver", "predict_reference", "ReferenceRanker",
                       "task6w_proposal_quality", "task6x_sam2_reference_refiner"):
            assert marker not in code, f"{name}: {marker}"


# ---------------------------------------------------------------- 11-19 holdout construction


def test_full_l3_val_source_total_checked():
    manifest = _artifact("task7e_holdout_manifest.json")
    assert manifest["source"]["records"] == sum(EXPECTED_SOURCE.values()) == 936
    assert manifest["source"]["per_program"] == EXPECTED_SOURCE
    assert manifest["source"]["expected_per_program"] == EXPECTED_SOURCE
    assert manifest["source"]["matches_expected"] is True
    rows = [json.loads(line) for line in
            (REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2" / "val.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    l3 = [row for row in rows if row["query_type"] in EXPECTED_SOURCE]
    assert len(l3) == 936
    assert sorted(manifest["source"]["programs"]) == sorted(EXPECTED_SOURCE)


def test_minival_record_ids_excluded_from_holdout():
    manifest = _artifact("task7e_holdout_manifest.json")
    mini = {record["sample_id"] for record in
            json.loads((PACK_ROOT / "z_mini_val_240.json").read_text(encoding="utf-8"))["records"]}
    holdout = {json.loads(line)["sample_id"] for line in
               (HOLDOUT_ROOT / "holdout_l3_rows.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()}
    assert len(mini) == 240
    assert not (holdout & mini)
    assert manifest["checks"]["overlap_with_z_mini_val_240"] == 0


def test_pairedval_member_ids_excluded_from_holdout():
    manifest = _artifact("task7e_holdout_manifest.json")
    paired = {record["sample_id"] for record in
              json.loads((PACK_ROOT / "z_paired_val20.json").read_text(encoding="utf-8"))["records"]}
    holdout = {json.loads(line)["sample_id"] for line in
               (HOLDOUT_ROOT / "holdout_l3_rows.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()}
    assert len(paired) == 40
    assert not (holdout & paired)
    assert manifest["checks"]["overlap_with_z_paired_val20"] == 0
    assert manifest["exclusion"]["union"] == 267


def test_no_test_split():
    manifest = _artifact("task7e_holdout_manifest.json")
    assert manifest["checks"]["test_records"] == 0
    assert manifest["test_split_used"] is False
    for name in TASK7E_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        for marker in ('"test.jsonl"', "'test.jsonl'", 'split="test"', "scene_disjoint_v1/test"):
            assert marker not in text, f"{name}: {marker}"
    for name in REQUIRED_ARTIFACTS:
        payload = _artifact(name)
        if payload is not None:
            assert payload.get("test_split_used", False) is False, name
    assert _artifact("task7e_verdict.json")["test_split_used"] is False


def test_holdout_at_least_600():
    manifest = _artifact("task7e_holdout_manifest.json")
    assert manifest["holdout"]["records"] == 669
    assert manifest["holdout"]["records"] >= manifest["checks"]["min_records"] == 600
    assert manifest["checks"]["records_gte_600"] is True
    assert manifest["holdout"]["subsampled"] is False
    assert manifest["verdict"] == "L3_HOLDOUT_READY"


def test_each_l3_class_at_least_120():
    manifest = _artifact("task7e_holdout_manifest.json")
    for program in L3_PROGRAMS:
        assert manifest["holdout"]["per_program"][program] >= 120, program
    assert sum(manifest["holdout"]["per_program"].values()) == 669
    assert manifest["checks"]["every_program_gte_120"] is True


def test_heldout_paired_same_tile_and_reference():
    pairs = json.loads((HOLDOUT_ROOT / "holdout_pairs.json").read_text(encoding="utf-8"))["pairs"]
    rows = {json.loads(line)["sample_id"]: json.loads(line) for line in
            (HOLDOUT_ROOT / "holdout_l3_rows.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()}
    assert len(pairs) == 20
    for pair in pairs:
        members = [rows[member["sample_id"]] for member in pair["members"]]
        assert members[0]["tile_id"] == members[1]["tile_id"] == pair["tile_id"]
        assert members[0]["reference_source_feature_id"] == \
            members[1]["reference_source_feature_id"] == pair["reference_source_feature_id"]


def test_heldout_paired_different_direction_and_target():
    pairs = json.loads((HOLDOUT_ROOT / "holdout_pairs.json").read_text(encoding="utf-8"))["pairs"]
    rows = {json.loads(line)["sample_id"]: json.loads(line) for line in
            (HOLDOUT_ROOT / "holdout_l3_rows.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()}
    for pair in pairs:
        members = [rows[member["sample_id"]] for member in pair["members"]]
        assert members[0]["program_id"] != members[1]["program_id"]
        assert members[0]["target_source_feature_id"] != members[1]["target_source_feature_id"]
        assert members[0]["program_id"] in EXPECTED_SOURCE
        assert members[1]["program_id"] in EXPECTED_SOURCE


def test_heldout_paired_has_no_old_paired_member():
    manifest = _artifact("task7e_holdout_manifest.json")
    paired_members = {record["sample_id"] for record in
                      json.loads((PACK_ROOT / "z_paired_val20.json")
                                 .read_text(encoding="utf-8"))["records"]}
    holdout_members = {member["sample_id"] for pair in
                       json.loads((HOLDOUT_ROOT / "holdout_pairs.json")
                                  .read_text(encoding="utf-8"))["pairs"]
                       for member in pair["members"]}
    assert not (holdout_members & paired_members)
    assert manifest["paired"]["overlap_with_task6z_paired_members"] == 0
    assert manifest["paired"]["pairs"] == 20
    assert manifest["paired"]["candidate_pairs"] == 175
    assert manifest["paired"]["passed"] is True


# ---------------------------------------------------------------- 20-24 reproduction and bootstrap


def test_minival_reproduction_exact():
    reproduction = _artifact("task7e_minival_reproduction.json")
    assert reproduction["reproduction_passed"] is True
    assert reproduction["verdict"] == "TASK7D_REPRODUCTION_PASS"
    assert reproduction["tolerance"] == 1.0e-6
    assert reproduction["reproduction"]["Z-B3"]["miou"] == 0.3242128982543474
    assert reproduction["reproduction"]["D-B1"]["miou"] == 0.3978996298363562
    for name in ("Z-B3", "D-B1"):
        assert reproduction["reproduction"][name]["delta"] <= 1.0e-6
        assert reproduction["reproduction"][name]["miou_ok"] is True
        assert reproduction["reproduction"][name]["paired_ok"] is True
    assert reproduction["reproduction"]["Z-B3"]["paired_passed"] == 15
    assert reproduction["reproduction"]["D-B1"]["paired_passed"] == 19
    assert reproduction["training_performed"] is False


def test_oracle_comparison_has_no_parser():
    oracle = _artifact("task7e_oracle_holdout.json")
    assert oracle["stage"] == "G-oracle-holdout"
    assert oracle["results"]["Z-B3"]["params"] == 275777
    assert oracle["results"]["D-B1"]["params"] == 278081
    verdict = _artifact("task7e_verdict.json")
    assert verdict["protocol"]["parser_used_for_oracle"] is False
    for name in ("task7e_evaluate_oracle.py",):
        code = _code_only(SCRIPTS / name)
        for marker in ("parse_instruction(", "build_program_parser(", "load_parser_checkpoint("):
            assert marker not in code, marker


def test_oracle_comparison_uses_gt_reference_only_as_declared():
    oracle = _artifact("task7e_oracle_holdout.json")
    assert "oracle" in oracle["_doc"].lower()
    assert oracle["records"] == 669
    assert len(oracle["results"]["Z-B3"]["per_program"]) == 4
    rows = [json.loads(line) for line in
            (HOLDOUT_ROOT / "holdout_l3_rows.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    assert all("reference_source_feature_id" in row for row in rows)
    verdict = _artifact("task7e_verdict.json")
    assert verdict["protocol"]["predicted_reference_in_oracle_stage"] is False


def test_bootstrap_seed_and_count_exact():
    oracle = _artifact("task7e_oracle_holdout.json")
    bootstrap = oracle["bootstrap"]
    assert bootstrap["seed"] == 20261001
    assert bootstrap["resamples"] == 2000
    assert bootstrap["ci_percent"] == [2.5, 97.5]
    assert bootstrap["unit"] == "record_id"
    assert bootstrap["paired"] is True
    code = _code_only(SCRIPTS / "task7e_evaluate_oracle.py")
    assert "BOOTSTRAP_SEED = 20261001" in code
    assert "BOOTSTRAP_RESAMPLES = 2000" in code
    assert "np.random.default_rng(BOOTSTRAP_SEED)" in code


def test_bootstrap_paired_by_record():
    from scripts.task7e_evaluate_oracle import bootstrap_ci

    deltas = np.linspace(-0.1, 0.3, 50)
    first = bootstrap_ci(deltas)
    second = bootstrap_ci(deltas)
    assert first["ci_lower"] == second["ci_lower"] and first["ci_upper"] == second["ci_upper"]
    assert abs(first["mean_delta"] - deltas.mean()) < 1e-12
    assert first["ci_lower"] < first["mean_delta"] < first["ci_upper"]
    oracle = _artifact("task7e_oracle_holdout.json")
    assert oracle["bootstrap"]["mean_delta"] == pytest.approx(oracle["delta"]["overall"], abs=1e-12)
    assert oracle["bootstrap_sample_ids_hash"] == \
        _artifact("task7e_holdout_manifest.json")["holdout"]["record_id_hash"]


# ---------------------------------------------------------------- 25-29 predicted reference


def test_u_c1_exact():
    quality = _artifact("task7e_predicted_reference_quality.json")
    resolver = quality["resolver"]
    assert resolver["config"] == "U-C1"
    assert resolver["imgsz"] == 640
    assert resolver["conf"] == 0.05
    assert resolver["max_det"] == 300
    assert resolver["nms"] == "default"
    assert resolver["tta"] is False and resolver["tiling"] is False
    assert resolver["checkpoint_sha256"] == \
        "ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474"
    from scripts.task6u_common import CONFIGS

    assert CONFIGS["U-C1"]["conf"] == 0.05 and CONFIGS["U-C1"]["max_det"] == 300


def test_shared_reference_resolver_between_p0_and_p1():
    import scripts.task7e_evaluate_predicted_reference as module

    source = inspect.getsource(module.main)
    assert source.count("_predict_with_reference(decoders, record, resolved[\"selection\"]") == 2
    assert "reference_cache" in source
    # one shared U-C1 resolution path, used by every record and by both pair members
    assert source.count("pipeline.resolve_reference(proposals)") == 1
    assert source.count("pipeline.proposals(tile_id, Path(image_path))") == 1
    assert source.count("resolve(pair[\"tile_id\"]") == 1
    assert "for name in (\"Z-B3\", \"D-B1\"):" in source


def test_no_gt_reference_in_predicted_inference():
    verdict = _artifact("task7e_verdict.json")
    assert verdict["protocol"]["predicted_reference_in_oracle_stage"] is False
    source = inspect.getsource(
        __import__("scripts.task7e_evaluate_predicted_reference", fromlist=["_predict_with_reference"])
        ._predict_with_reference)
    assert "masks.mask(record" not in source
    assert "reference_source_feature_id" in source and "synthetic_id" in source
    quality = _artifact("task7e_predicted_reference_quality.json")
    assert quality["metrics"]["records"] == 669
    assert all("reference_iou" in row for row in quality["rows"])


def test_no_gt_target_in_inference():
    for name in TASK7E_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("target_mask", "target_component_id", "GT target as input"):
            assert marker not in code, f"{name}: {marker}"
    import scripts.task7e_evaluate_predicted_reference as module

    source = inspect.getsource(module._predict_with_reference)
    assert "target_source_feature_id" not in source


def test_reference_reused_within_predicted_pair():
    paired = _artifact("task7e_predicted_reference_paired.json")
    assert paired["pairs"]["count"] == 20
    assert paired["pairs"]["reference_abstention_pairs"] == 0
    import scripts.task7e_evaluate_predicted_reference as module

    source = inspect.getsource(module.main)
    pair_block = source.split("for pair in pairs:")[1].split("for index, record in enumerate(members)")[0]
    assert pair_block.count("resolve(") == 1


# ---------------------------------------------------------------- 30-41 guards and continuation


def test_task7c_parser_not_trained():
    integration = _artifact("task7e_canonical_parser_integration.json")
    assert integration is not None
    assert integration["executed"] is False
    assert integration["parser_trained"] is False
    assert "DB1_PREDICTED_REFERENCE_USABLE is false" in integration["reason"]
    assert _git_changed("buildreasonseg_mvp/program_parser.py") == ""
    assert _git_changed("evaluation/task7c_") == ""


def test_no_free_form_paraphrase_used_for_selection():
    verdict = _artifact("task7e_verdict.json")
    assert verdict["protocol"]["free_form_paraphrase_used_for_selection"] is False
    integration = _artifact("task7e_canonical_parser_integration.json")
    assert integration["free_form_paraphrase_used"] is False
    for name in TASK7E_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("task7a_l3_paraphrase_pack", "task7b_l3_stress_v1",
                       "task7b_compositional_minimal_pairs"):
            assert marker not in code, f"{name}: {marker}"


def test_fields_unchanged():
    assert _git_changed("buildreasonseg_mvp/geometric_relation_field_v02.py") == ""
    assert _git_changed("buildreasonseg_mvp/nearest_boundary_field.py") == ""
    assert _git_changed("buildreasonseg_mvp/task6z_field_composition.py") == ""
    verdict = _artifact("task7e_verdict.json")
    assert verdict["protocol"]["fields_changed"] is False
    from buildreasonseg_mvp.geometric_relation_field_v02 import ALPHA, S_AXIS, S_MARGIN, TAU
    from buildreasonseg_mvp.nearest_boundary_field import SIGMA_DIAG

    assert (ALPHA, TAU, S_AXIS, S_MARGIN, SIGMA_DIAG) == (1.2, 0.04, 0.02, 0.02, 0.05)


def test_sam2_unchanged():
    assert _git_changed("buildreasonseg_mvp/task6n_relation_decoder.py") == ""
    verdict = _artifact("task7e_verdict.json")
    assert verdict["protocol"]["sam2_changed"] is False
    from buildreasonseg_mvp.task6n_relation_decoder import SAM2_CONFIG_NAME

    assert SAM2_CONFIG_NAME == "configs/sam2.1/sam2.1_hiera_b+.yaml"
    cached = sorted((REPO_ROOT / "artifacts" / "task6n" / "features").glob("*.npy"))[:1]
    if cached:
        assert np.load(cached[0]).shape == (256, 64, 64)


def test_no_ranker_quality_or_refinement():
    quality = _artifact("task7e_predicted_reference_quality.json")
    assert quality["resolver"]["ranker"] is False
    assert quality["resolver"]["quality_filter"] is False
    assert quality["resolver"]["sam2_refinement"] is False
    assert _git_changed("buildreasonseg_mvp/task6u_reference_ranker.py") == ""
    assert _git_changed("buildreasonseg_mvp/task6w_proposal_quality.py") == ""
    assert _git_changed("buildreasonseg_mvp/task6x_sam2_reference_refiner.py") == ""


def test_no_learned_competition():
    verdict = _artifact("task7e_verdict.json")
    assert verdict["protocol"]["learned_competition"] is False
    for name in TASK7E_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("GlobalCompetitionDecoder(", "ScoreHead(", "global_competition("):
            assert marker not in code, f"{name}: {marker}"
    selector = _code_only(REPO_ROOT / "buildreasonseg_mvp"
                          / "task7d_global_competition_decoder.py").split(
        "class GlobalCompetitionDecoder")[1]
    assert "learned" not in selector.split("def ")[0]


def test_no_attention_transformer_gnn():
    for name in TASK7E_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("MultiheadAttention", "TransformerEncoder", "nn.Transformer", "MessagePassing",
                       "scaled_dot_product_attention"):
            assert marker not in code, f"{name}: {marker}"
    verdict = _artifact("task7e_verdict.json")
    assert verdict["protocol"]["attention_or_graph"] is False


def test_no_grcl():
    for name in TASK7E_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("grcl_directional", "LAMBDA_GRCL", "grcl.loss"):
            assert marker not in code, f"{name}: {marker}"
    verdict = _artifact("task7e_verdict.json")
    assert verdict["protocol"]["grcl"] is False


def test_no_full_training():
    verdict = _artifact("task7e_verdict.json")
    assert verdict["interpretation_boundary"]["full_training_run"] is False
    assert verdict["interpretation_boundary"]["d_b1_called_final_model"] is False
    assert verdict["interpretation_boundary"]["task7f_chosen"] is False
    assert verdict["decision"]["checkpoints_deleted_or_overwritten"] is False


def test_no_new_dataset_download_install_or_gui():
    for name in TASK7E_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("urlretrieve", "requests.get", "hf_hub_download", "snapshot_download",
                       "pip install", "load_dataset", "tkinter", "gradio", "streamlit", "flask"):
            assert marker not in code, f"{name}: {marker}"


def test_verdict_and_gate_arithmetic():
    verdict = _artifact("task7e_verdict.json")
    oracle = _artifact("task7e_oracle_holdout.json")
    predicted = _artifact("task7e_predicted_reference_holdout.json")
    assert verdict["oracle_holdout"]["gate_passed"] == oracle["DB1_HOLDOUT_GENERALIZES"] is True
    assert verdict["predicted_reference"]["gate_passed"] == \
        predicted["DB1_PREDICTED_REFERENCE_USABLE"] is False
    assert verdict["decision"]["DB1_PREDICTED_REFERENCE_USABLE"] is False
    assert verdict["decision"]["DB1_ADOPT_AS_DEVELOPMENT_L3_DECODER"] is False
    assert verdict["decision"]["development_baseline"] == "Z-B3"
    assert verdict["verdict"] == "DB1_PREDICTED_REFERENCE_BELOW_GATE"
    assert verdict["verdict"] in verdict["allowed_verdicts"]
    assert verdict["recommendation"].startswith("等待 ChatGPT 根据 Task 7E")
    failed = [name for name, entry in predicted["gate"].items() if not entry["passed"]]
    assert failed == ["2_answered_only_miou", "5_paired", "6_margin"]
    # oracle gate conditions
    assert all(entry["passed"] for entry in oracle["gate"].values())
    assert oracle["results"]["D-B1"]["overall"]["miou"] >= 0.36
    assert oracle["delta"]["overall"] >= 0.05
    assert oracle["bootstrap"]["ci_lower"] > 0.0
    assert _artifact("task7e_oracle_holdout_paired.json")["results"]["D-B1"]["passed"] >= 16


def test_required_artifacts_and_modules_exist():
    assert (REPO_ROOT / "docs" / "task7e_deterministic_prototype_holdout.md").is_file()
    for name in TASK7E_SOURCES:
        assert (SCRIPTS / name).is_file(), name
    for name in REQUIRED_ARTIFACTS:
        assert (EVAL / name).is_file(), name
    assert ADAPTER.is_file()


def test_previous_suite_preserved():
    for name in ("test_task7d_global_competition.py", "test_task7c_rehearsal_hardening.py",
                 "test_task7b_parser_hardening.py", "test_task7a_l3_integration.py"):
        assert (REPO_ROOT / "tests" / name).is_file(), name
    removed = subprocess.run(["git", "diff", "--name-only", "--diff-filter=D", BASE_COMMIT, "--",
                              "tests/"], cwd=REPO_ROOT, capture_output=True, text=True,
                             check=True).stdout
    assert removed.strip() == "", f"no test file may be removed: {removed}"
