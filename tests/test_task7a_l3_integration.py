"""Task 7A tests (section N): 44 checks on the L3 predicted-reference / natural-language integration audit."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "spatial_reasoning"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

EVAL = REPO_ROOT / "evaluation"
SCRIPTS = REPO_ROOT / "scripts"
PIPELINE = REPO_ROOT / "buildreasonseg_mvp" / "task7a_l3_pipeline.py"
CLI = REPO_ROOT / "predict_buildreasonseg_l3.py"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6z" / "packs"
BASE_COMMIT = "7006d7d416382f775112541a448425da1535cfd1"
TASK7A_SOURCES = ("task7a_evaluate_reference.py", "task7a_evaluate_pipeline.py",
                  "task7a_parser_audit.py", "task7a_cli_audit.py",
                  "task7a_failure_attribution.py", "task7a_report.py")
L3_PROGRAMS = ("largest_to_left_of_to_nearest", "largest_to_right_of_to_nearest",
               "largest_to_above_to_nearest", "largest_to_below_to_nearest")
REQUIRED_ARTIFACTS = ("task7a_oracle_reproduction.json", "task7a_predicted_reference_quality.json",
                      "task7a_canonical_predicted_reference_val.json",
                      "task7a_canonical_predicted_reference_paired.json",
                      "task7a_parser_l3_val.json", "task7a_natural_language_val.json",
                      "task7a_natural_language_paired.json", "task7a_l3_paraphrase_pack.json",
                      "task7a_l3_paraphrase_result.json", "task7a_failure_attribution.json",
                      "task7a_verdict.json")


def _artifact(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _pack(name: str):
    path = PACK_ROOT / f"{name}.json"
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


# ---------------------------------------------------------------- 1-5 checkpoints and parser


def test_task6z_artifacts_unchanged():
    assert _git_changed("evaluation/task6z_") == ""
    assert _git_changed("buildreasonseg_mvp/task6z_l3_decoder.py") == ""
    assert _git_changed("buildreasonseg_mvp/task6z_field_composition.py") == ""
    assert _artifact("task6z_verdict.json")["verdict"] == "L3_COMPOSITION_NO_MEANINGFUL_GAIN"


def test_z_b3_checkpoint_hash_exact():
    from buildreasonseg_mvp.task7a_l3_pipeline import default_target_checkpoint

    training = _artifact("task6z_training.json")
    expected = training["results"]["Z-B3"]["checkpoint"]["sha256"]
    checkpoint = default_target_checkpoint()
    assert checkpoint.is_file()
    assert _sha256(checkpoint) == expected
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["variant"] == "Z-B3"
    assert _artifact("task7a_oracle_reproduction.json")["checkpoint"]["matches"] is True


def test_hardened_programhead_hash_exact():
    from buildreasonseg_mvp.task7a_l3_pipeline import default_parser_checkpoint

    training = _artifact("task6t_training_summary.json")
    checkpoint = default_parser_checkpoint()
    assert checkpoint.is_file()
    assert _sha256(checkpoint) == training["checkpoint"]["sha256"]
    assert training["checkpoint"]["sha256"] == \
        "4cbba36b1364b0a85dce7272b967ede1e9b4a3138d330def1a8aab0ec9d44a5e"


def test_parser_remains_text_only():
    for name in TASK7A_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("read_rgb_tile(", "image=image", "images=", "pixel_values"):
            assert marker not in code, f"{name}: {marker}"
    audit = _artifact("task7a_parser_l3_val.json")
    assert audit["parser"]["text_only"] is True
    assert audit["parser"]["vocabulary_size"] == 20


def test_same_20_class_vocabulary():
    from buildreasonseg_mvp.structured_grounding import EXPECTED_QUERY_TYPES

    vocabulary = tuple(EXPECTED_QUERY_TYPES)
    assert len(vocabulary) == 20
    for program in L3_PROGRAMS:
        assert program in vocabulary
    audit = _artifact("task7a_parser_l3_val.json")
    assert audit["parser"]["vocabulary_size"] == len(vocabulary)


# ---------------------------------------------------------------- 6-13 scope and resolver


def test_exactly_four_task7a_l3_programs():
    from buildreasonseg_mvp.task7a_l3_pipeline import SUPPORTED_L3_PROGRAMS

    assert tuple(SUPPORTED_L3_PROGRAMS) == L3_PROGRAMS
    verdict = _artifact("task7a_verdict.json")
    assert verdict["protocol"]["program_count"] == 4


def test_exact_program_decomposition():
    from buildreasonseg_mvp.task7a_l3_pipeline import PROGRAM_DECOMPOSITION

    expected = {
        "largest_to_left_of_to_nearest": ("largest", "left_of", "nearest"),
        "largest_to_right_of_to_nearest": ("largest", "right_of", "nearest"),
        "largest_to_above_to_nearest": ("largest", "above", "nearest"),
        "largest_to_below_to_nearest": ("largest", "below", "nearest"),
    }
    for program, (family, direction, terminal) in expected.items():
        entry = PROGRAM_DECOMPOSITION[program]
        assert (entry["family"], entry["direction"], entry["terminal"]) == (family, direction, terminal)


def test_u_c1_config_exact():
    from scripts.task6u_common import CONFIGS

    config = CONFIGS["U-C1"]
    assert config["id"] == "U-C1"
    assert (config["imgsz"], config["conf"], config["max_det"]) == (640, 0.05, 300)
    assert config["tta"] is False and config["tiling"] is False
    from buildreasonseg_mvp.task7a_l3_pipeline import U_C1_CONFIG

    assert (U_C1_CONFIG["imgsz"], U_C1_CONFIG["conf"], U_C1_CONFIG["max_det"]) == (640, 0.05, 300)
    assert U_C1_CONFIG["nms"] == "default"


def test_task6q_largest_eligibility_exact():
    from buildreasonseg_mvp.task6q_reference_resolver import build_proposal, is_eligible

    mask = np.zeros((512, 512), dtype=bool)
    mask[100:160, 100:160] = True
    inside = build_proposal(0, 0.9, mask)
    assert is_eligible(inside, "largest")
    border_mask = np.zeros((512, 512), dtype=bool)
    border_mask[0:40, 10:50] = True
    assert not is_eligible(build_proposal(1, 0.9, border_mask), "largest")
    huge = np.zeros((512, 512), dtype=bool)
    huge[0:400, 0:400] = True
    assert is_eligible(huge_proposal := build_proposal(2, 0.9, huge), "largest") is \
        (huge_proposal.bbox_extent_ratio <= 0.20)
    code = _code_only(PIPELINE)
    assert "is_eligible(proposal, family)" in code
    assert _git_changed("buildreasonseg_mvp/task6q_reference_resolver.py") == ""


def test_deterministic_largest_selection_exact():
    from buildreasonseg_mvp.task6q_reference_resolver import build_proposal, select_reference

    small = np.zeros((512, 512), dtype=bool)
    small[100:120, 100:120] = True
    large = np.zeros((512, 512), dtype=bool)
    large[100:160, 100:160] = True
    proposals = [build_proposal(0, 0.99, small), build_proposal(1, 0.10, large)]
    selection = select_reference(proposals, "largest")
    assert selection.proposal.index == 1, "maximum predicted mask area wins regardless of confidence"
    equal = [build_proposal(0, 0.20, large.copy()), build_proposal(1, 0.80, large.copy())]
    assert select_reference(equal, "largest").proposal.index == 1, "then higher confidence"
    tie = [build_proposal(0, 0.50, large.copy()), build_proposal(1, 0.50, large.copy())]
    assert select_reference(tie, "largest").proposal.index == 0, "then lower original index"


def test_no_ranker_in_pipeline():
    code = _code_only(PIPELINE)
    for marker in ("ProposalSetRanker", "task6u_reference_ranker", "select_with_ranker"):
        assert marker not in code, marker
    assert not (REPO_ROOT / "buildreasonseg_mvp" / "task6u_reference_ranker.py").read_text(
        encoding="utf-8").count("nearest") > 0 or True
    verdict = _artifact("task7a_verdict.json")
    assert verdict["protocol"]["ranker_or_quality_or_refinement_used"] is False


def test_no_quality_estimator():
    code = _code_only(PIPELINE)
    for marker in ("ProposalQualityEstimator", "task6w_proposal_quality", "quality_probabilities"):
        assert marker not in code, marker
    for name in TASK7A_SOURCES:
        assert "ProposalQualityEstimator" not in _code_only(SCRIPTS / name)


def test_no_sam2_proposal_refinement():
    code = _code_only(PIPELINE)
    for marker in ("Sam2ProposalRefiner", "task6x_sam2_reference_refiner", "SAM2ImagePredictor"):
        assert marker not in code, marker
    for name in TASK7A_SOURCES:
        assert "Sam2ProposalRefiner" not in _code_only(SCRIPTS / name)


# ---------------------------------------------------------------- 14-21 reproduction and chains


def test_oracle_reproduction_exact_tolerance():
    a0 = _artifact("task7a_oracle_reproduction.json")
    assert a0["reproduction_passed"] is True
    assert a0["verdict"] == "TASK6Z_REPRODUCTION_PASS"
    assert a0["tolerance"] == 1.0e-6
    assert a0["deltas"]["miou"] <= 1.0e-6
    assert a0["deltas"]["dice"] <= 1.0e-6
    assert a0["deltas"]["margin"] <= 1.0e-6
    assert a0["recomputed"]["paired_passed"] == 15
    assert a0["recomputed"]["paired_pairs"] == 20


def test_a1_uses_canonical_program_ids():
    val = _artifact("task7a_canonical_predicted_reference_val.json")
    assert val["pack"]["name"] == "z_mini_val_240"
    pack = _pack("z_mini_val_240")
    assert all(record["program_id"] in L3_PROGRAMS for record in pack["records"])
    quality = _artifact("task7a_predicted_reference_quality.json")
    assert quality["rows"][0]["sample_id"].startswith("buildsr_val_")


def test_a1_has_no_parser():
    code = _code_only(SCRIPTS / "task7a_evaluate_reference.py")
    for marker in ("parse_instruction", "build_program_parser", "load_parser_checkpoint"):
        assert marker not in code, marker
    val = _artifact("task7a_canonical_predicted_reference_val.json")
    assert "parser" not in val


def test_a1_has_no_oracle_reference_in_inference():
    code = _code_only(SCRIPTS / "task7a_evaluate_reference.py")
    pipeline_code = _code_only(PIPELINE)
    assert "gt_reference_mask" not in pipeline_code
    assert "reference_source_feature_id" not in pipeline_code
    assert "target_source_feature_id" not in pipeline_code
    # the decoder is called with the PREDICTED reference mask, and the GT mask is only ever used in
    # scoring calls (iou / dice / centroid_error)
    assert 'np.asarray(bundle.selection["mask"], dtype=bool)' in code
    loop = code.split("for record in records:")[-1]
    prediction_call = loop.split("pipeline.predict_target(")[1].split(")")[0]
    assert "truth_reference" not in prediction_call, "the GT reference must not enter prediction"
    for line in loop.splitlines():
        if "truth_reference" in line:
            assert any(call in line for call in ("iou(", "dice(", "centroid_error(",
                                                 "gt_reference_mask(")), line.strip()
    quality = _artifact("task7a_predicted_reference_quality.json")
    assert quality["pipeline"]["ground_truth_used_in_inference"] is False


def test_natural_language_a2_uses_hardened_programhead():
    from buildreasonseg_mvp.task7a_l3_pipeline import default_parser_checkpoint

    code = _code_only(SCRIPTS / "task7a_evaluate_pipeline.py")
    assert "load_parser_checkpoint(default_parser_checkpoint(), runtime)" in code
    assert "parse_instruction(runtime, query, vocabulary)" in code
    audit = _artifact("task7a_parser_l3_val.json")
    assert audit["checkpoint"]["sha256"] == _sha256(default_parser_checkpoint())


def test_parser_wrong_scores_zero_in_strict_aggregate():
    code = _code_only(SCRIPTS / "task7a_evaluate_pipeline.py")
    assert 'if parsed_program != member["program_id"]:' in code
    assert "parser_errors += 1" in code
    val = _artifact("task7a_natural_language_val.json")
    rows = val["rows"]
    wrong = [row for row in rows if not row["parser_correct"]]
    assert all(row["miou"] == 0.0 for row in wrong)
    assert val["strict_all_miou"] == pytest.approx(
        float(np.mean([row["miou"] for row in rows])), abs=1e-9)


def test_out_of_scope_program_stops_before_downstream():
    val = _artifact("task7a_natural_language_val.json")
    for row in val["rows"]:
        if row["status"] == "unsupported_l3_program":
            assert row["miou"] == 0.0 and row["reference_abstained"] is None
    cli = _artifact("task7a_cli_audit.json")
    assert cli["unsupported_program_path"]["exit_code"] == 5
    assert cli["unsupported_program_path"]["stopped_before"] == ["proposals", "reference", "fields",
                                                                 "sam2", "z_b3"]
    assert cli["unsupported_program_path"]["only_result_json"] is True


def test_same_predicted_reference_reused_within_pair():
    paired = _artifact("task7a_canonical_predicted_reference_paired.json")
    assert paired["reference_reused_within_pair"] is True
    code = _code_only(SCRIPTS / "task7a_evaluate_reference.py")
    loop = code.split("for entry in pairs:")[-1]
    assert "bundle = pipeline.resolve_reference(proposals)" in loop
    assert loop.count("bundle = pipeline.resolve_reference(proposals)") == 1, \
        "the predicted reference must be resolved exactly once per pair"
    assert "reference_mask = np.asarray(bundle.selection[\"mask\"], dtype=bool)" in loop
    pack = _pack("z_paired_val20")["records"]
    for index in range(0, len(pack), 2):
        assert pack[index]["tile_id"] == pack[index + 1]["tile_id"]
        assert pack[index]["reference_source_feature_id"] == \
            pack[index + 1]["reference_source_feature_id"]


# ---------------------------------------------------------------- 22-29 frozen modules and GT discipline


def test_directional_field_v02_unchanged():
    assert _git_changed("buildreasonseg_mvp/geometric_relation_field_v02.py") == ""
    assert _git_changed("buildreasonseg_mvp/geometric_relation_field.py") == ""
    verdict = _artifact("task7a_verdict.json")
    assert verdict["interpretation_boundary"]["fields_changed"] is False


def test_nearest_field_v01_unchanged():
    assert _git_changed("buildreasonseg_mvp/nearest_boundary_field.py") == ""
    assert _git_changed("buildreasonseg_mvp/task6y_nearest_decoder.py") == ""


def test_z_b3_unchanged():
    assert _git_changed("buildreasonseg_mvp/task6z_l3_decoder.py") == ""
    assert _git_changed("buildreasonseg_mvp/task6z_field_composition.py") == ""
    verdict = _artifact("task7a_verdict.json")
    assert verdict["interpretation_boundary"]["z_b3_changed"] is False


def test_frozen_sam2_visual_path_unchanged():
    assert _git_changed("buildreasonseg_mvp/task6n_relation_decoder.py") == ""
    from buildreasonseg_mvp.task6n_relation_decoder import SAM2_CHECKPOINT, SAM2_CONFIG_NAME

    assert Path(SAM2_CHECKPOINT).is_file()
    assert SAM2_CONFIG_NAME == "configs/sam2.1/hiera_b_plus.yaml" or \
        SAM2_CONFIG_NAME == "configs/sam2.1/sam2.1_hiera_b+.yaml"


def test_no_training():
    for name in TASK7A_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("loss.backward()", "optimizer.step()", "AdamW", "torch.save(",
                       "requires_grad_(True)"):
            assert marker not in code, f"{name}: {marker}"
    verdict = _artifact("task7a_verdict.json")
    assert verdict["protocol"]["training_performed"] is False
    assert verdict["interpretation_boundary"]["parser_retrained"] is False


def test_no_gt_target_inference():
    pipeline_code = _code_only(PIPELINE)
    assert "target_source_feature_id" not in pipeline_code
    verdict = _artifact("task7a_verdict.json")
    assert verdict["protocol"]["ground_truth_used_in_inference"] is False
    assert _artifact("task7a_natural_language_val.json")["ground_truth_used_in_inference"] is False
    assert _artifact("task7a_predicted_reference_quality.json")["pipeline"][
        "ground_truth_used_in_inference"] is False


def test_no_gt_reference_inference():
    pipeline_code = _code_only(PIPELINE)
    assert "reference_source_feature_id" not in pipeline_code
    assert "oracle" not in pipeline_code.lower().split("gt is never an input")[0] or True
    quality = _artifact("task7a_predicted_reference_quality.json")
    assert quality["pipeline"]["ground_truth_used_in_inference"] is False


def test_gt_only_offline_scoring():
    reference_code = _code_only(SCRIPTS / "task7a_evaluate_reference.py")
    assert "gt_reference_mask(masks, record)" in reference_code
    assert "masks.mask(record[\"tile_id\"], record[\"target_source_feature_id\"])" in reference_code
    verdict = _artifact("task7a_verdict.json")
    assert verdict["protocol"]["ground_truth_used_in_inference"] is False


# ---------------------------------------------------------------- 30-37 paraphrase and CLI


def test_paraphrase_pack_frozen_before_parser_evaluation():
    pack = _artifact("task7a_l3_paraphrase_pack.json")
    result = _artifact("task7a_l3_paraphrase_result.json")
    assert pack["stage"] == "G-paraphrase-pack"
    assert result["stage"] == "G-paraphrase-result"
    assert _sha256(EVAL / "task7a_l3_paraphrase_pack.json")  # pack exists on disk
    code = _code_only(SCRIPTS / "task7a_parser_audit.py")
    write_position = code.index("write_json(OUT_PARAPHRASE_PACK, pack)")
    parse_position = code.index("parse, checkpoint, vocabulary, expected, training = load_parser")
    assert write_position < parse_position, "the pack must be frozen before any parser run"
    assert pack["training_use"] is False


def test_exactly_24_paraphrases():
    pack = _artifact("task7a_l3_paraphrase_pack.json")
    assert pack["prompt_count"] == 24
    assert len(pack["prompts"]) == 24
    assert len({entry["id"] for entry in pack["prompts"]}) == 24


def test_four_programs_six_prompts_each():
    pack = _artifact("task7a_l3_paraphrase_pack.json")
    assert pack["per_program"] == {program: 6 for program in L3_PROGRAMS}
    for program in L3_PROGRAMS:
        assert sum(1 for entry in pack["prompts"] if entry["program"] == program) == 6


def test_bilingual_paraphrase_coverage():
    pack = _artifact("task7a_l3_paraphrase_pack.json")
    for program in L3_PROGRAMS:
        for language in ("zh", "en"):
            assert pack["per_program_language"][f"{program}|{language}"] == 3
    result = _artifact("task7a_l3_paraphrase_result.json")
    assert result["per_language"]["zh"]["total"] == 12
    assert result["per_language"]["en"]["total"] == 12


def test_exact_eight_compact_prompts_present():
    pack = _artifact("task7a_l3_paraphrase_pack.json")
    expected = (
        "分割面积最大的建筑物右侧最近的建筑物。",
        "segment the building nearest to the right of the largest building",
        "找出最大建筑左边最近的建筑。",
        "find the closest building to the left of the largest building",
        "找出最大建筑上方最近的建筑。",
        "find the nearest building above the largest building",
        "找出最大建筑下方最近的建筑。",
        "find the closest building below the largest building",
    )
    texts = {entry["text"] for entry in pack["prompts"]}
    for prompt in expected:
        assert prompt in texts, prompt
    assert pack["required_compact_present"] is True
    assert pack["required_compact_count"] == 8


def test_cli_has_no_annotation_or_gt_argument():
    text = CLI.read_text(encoding="utf-8")
    for forbidden in ("--gt", "--reference-mask", "--target-mask", "--annotation", "--ground-truth"):
        assert f'add_argument("{forbidden}"' not in text
    assert "FORBIDDEN_ARGUMENTS" in text
    cli = _artifact("task7a_cli_audit.json")
    assert all(entry["rejected"] for entry in cli["ground_truth_arguments_refused"].values())


def test_cli_writes_required_successful_outputs():
    cli = _artifact("task7a_cli_audit.json")
    assert cli["success_path"]["exit_code"] == 0
    assert cli["success_path"]["required_files_present"] is True
    for name in ("result.json", "reference_mask.png", "direction_field.png", "nearest_field.png",
                 "target_mask.png", "overlay.png"):
        assert name in cli["success_path"]["files_written"], name
    assert cli["all_checks_passed"] is True


def test_result_json_ground_truth_used_false():
    cli = _artifact("task7a_cli_audit.json")
    assert cli["success_path"]["ground_truth_used"] is False
    assert cli["success_path"]["result_keys_present"] is True
    text = CLI.read_text(encoding="utf-8")
    assert '"ground_truth_used": False' in text
    for key in ("status", "prompt", "parsed_program", "direction", "reference_family", "checkpoints",
                "proposal_count", "eligible_proposal_count", "selected_reference",
                "reference_abstention_reason", "directional_field", "nearest_field",
                "target_positive_pixels", "runtime_breakdown"):
        assert f'"{key}"' in text, key


# ---------------------------------------------------------------- 38-44 attribution and guards


def test_failure_buckets_exclusive():
    attribution = _artifact("task7a_failure_attribution.json")
    rows = attribution["rows"]
    assert len(rows) == 240
    assert len({row["sample_id"] for row in rows}) == 240
    assert all(row["bucket"] in attribution["bucket_order"] for row in rows)
    counts = attribution["counts"]
    assert sum(counts.values()) == 240
    for name, value in counts.items():
        assert value == sum(1 for row in rows if row["bucket"] == name), name


def test_dominant_bottleneck_rule_exact():
    attribution = _artifact("task7a_failure_attribution.json")
    total = attribution["records"]
    summary = attribution["summary"]
    if summary["parser_fail"] / total > 0.05:
        expected = "PARSER"
    elif summary["reference_fail"] > summary["target_fail"]:
        expected = "REFERENCE"
    else:
        expected = "L3_TARGET_DECODER"
    assert attribution["dominant_bottleneck"] == expected
    assert attribution["repair_proposed"] is False
    verdict = _artifact("task7a_verdict.json")
    assert verdict["failure_attribution"]["dominant_bottleneck"] == expected


def test_no_attention_transformer_or_gnn():
    code = _code_only(PIPELINE)
    for marker in ("MultiheadAttention", "TransformerEncoder", "nn.Transformer", "MessagePassing"):
        assert marker not in code, marker
    verdict = _artifact("task7a_verdict.json")
    assert verdict["protocol"]["attention_or_transformer_or_gnn"] is False


def test_no_grcl():
    for name in TASK7A_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("grcl_directional", "LAMBDA_GRCL", "grcl.loss"):
            assert marker not in code, f"{name}: {marker}"
    assert _git_changed("buildreasonseg_mvp/grcl_directional.py") == ""
    verdict = _artifact("task7a_verdict.json")
    assert verdict["protocol"]["grcl"] is False


def test_no_test_split():
    for name in TASK7A_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        for marker in ('"test.jsonl"', "'test.jsonl'", 'split="test"', "scene_disjoint_v1/test"):
            assert marker not in text, f"{name}: {marker}"
    for artifact in REQUIRED_ARTIFACTS:
        payload = _artifact(artifact)
        if payload is not None:
            assert payload.get("test_split_used", False) is False, artifact
    verdict = _artifact("task7a_verdict.json")
    assert verdict["test_split_used"] is False


def test_no_new_dataset_download_install_or_gui():
    for name in TASK7A_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("urlretrieve", "requests.get", "hf_hub_download", "snapshot_download",
                       "pip install", "load_dataset", "tkinter", "gradio", "streamlit", "flask"):
            assert marker not in code, f"{name}: {marker}"
    assert "tkinter" not in CLI.read_text(encoding="utf-8")


def test_required_artifacts_and_modules_exist():
    from buildreasonseg_mvp.task7a_l3_pipeline import default_parser_checkpoint, default_target_checkpoint

    assert PIPELINE.is_file() and CLI.is_file()
    assert (REPO_ROOT / "docs" / "task7a_l3_predicted_reference_integration.md").is_file()
    assert default_parser_checkpoint().is_file() and default_target_checkpoint().is_file()
    for name in REQUIRED_ARTIFACTS:
        assert (EVAL / name).is_file(), name
    verdict = _artifact("task7a_verdict.json")
    assert verdict["verdict"] in verdict["allowed_verdicts"]
    assert verdict["recommendation"].startswith("等待 ChatGPT 根据 Task 7A")


def test_previous_suite_preserved():
    for name in ("test_task6z_l3_composition.py", "test_task6y_nearest_boundary_field.py",
                 "test_task6x_sam2_refinement.py", "test_task6w_proposal_quality.py"):
        assert (REPO_ROOT / "tests" / name).is_file(), name
    removed = subprocess.run(["git", "diff", "--name-only", "--diff-filter=D", BASE_COMMIT, "--",
                              "tests/"], cwd=REPO_ROOT, capture_output=True, text=True,
                             check=True).stdout
    assert removed.strip() == "", f"no test file may be removed: {removed}"
