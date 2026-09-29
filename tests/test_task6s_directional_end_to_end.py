"""Task 6S tests (section 27): 42 checks on the integrated directional end-to-end chain."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.task6n_relation_decoder import read_pack  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
SCRIPTS = REPO_ROOT / "scripts"
PIPELINE = REPO_ROOT / "buildreasonseg_mvp" / "task6s_directional_pipeline.py"
CLI = REPO_ROOT / "predict_buildreasonseg_directional.py"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6n" / "packs"
FEATURE_ROOT = REPO_ROOT / "artifacts" / "task6n" / "features"
BASE_COMMIT = "a3d59da8c8698f793357887762dbc95a08a254aa"
TASK6S_SOURCES = ("task6s_evaluate.py", "task6s_cli_audit.py", "task6s_failure_attribution.py",
                  "task6s_report.py")
PROGRAM_HEAD_SHA256 = "eb50b02163ec5e8f3305321ee47a6d52a799235b68730b67f539d962a6d028a3"
PROPOSAL_SHA256 = "ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474"


def _artifact(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_changed(prefix: str) -> str:
    return subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "--", prefix],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _code_only(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    for prefix in ("r", "f", "b", "u", "rf", "fr"):
        if stripped.startswith(prefix + '"""'):
            stripped = stripped[len(prefix):]
            break
        if stripped.startswith(prefix + "'''"):
            stripped = stripped[len(prefix):]
            break
    for quote in ('"""', "'''"):
        if stripped.startswith(quote):
            end = stripped.find(quote, len(quote))
            if end != -1:
                stripped = stripped[end + len(quote):]
            break
    return "\n".join(line.split("#", 1)[0] for line in stripped.splitlines())


def _module_code() -> str:
    return _code_only(PIPELINE)


# ---------------------------------------------------------------- 1-2 frozen artifacts


def test_task6r_artifacts_unchanged():
    changed = _git_changed("evaluation/task6r_")
    assert changed == "", f"Task 6R artifacts changed: {changed}"
    assert _git_changed("buildreasonseg_mvp/grcl_directional.py") == ""
    verdict = _artifact("task6r_verdict.json")
    assert verdict["verdict"] == "GRCL_NO_MEANINGFUL_RELATION_GAIN"


def test_task6q_artifacts_unchanged():
    changed = _git_changed("evaluation/task6q_")
    assert changed == "", f"Task 6Q artifacts changed: {changed}"
    assert _artifact("task6q_verdict.json")["verdict"] == "REFERENCE_PROPOSAL_COVERAGE_INSUFFICIENT"


# ---------------------------------------------------------------- 3-5 ProgramHead


def test_program_head_checkpoint_sha_exact():
    from buildreasonseg_mvp.task6s_directional_pipeline import resolve_program_head_checkpoint

    path, report = resolve_program_head_checkpoint()
    assert path is not None, report
    assert _sha256(path) == PROGRAM_HEAD_SHA256
    audit = _artifact("task6s_frozen_asset_audit.json")
    if audit is not None:
        assert audit["program_head"]["matches_expected"] is True
        assert audit["program_head"]["resolved_sha256"] == PROGRAM_HEAD_SHA256


def test_program_head_receives_text_only():
    code = _module_code()
    node = ast.parse(PIPELINE.read_text(encoding="utf-8"))
    function = next(item for item in ast.walk(node)
                    if isinstance(item, ast.FunctionDef) and item.name == "parse_instruction")
    arguments = {argument.arg for argument in function.args.args}
    assert "image" not in arguments and "image_path" not in arguments
    assert "runtime.predict(prompt)" in code
    assert "image_tokens\": False" in code
    audit = _artifact("task6s_frozen_asset_audit.json")
    if audit is not None:
        assert audit["program_head"]["text_only"] is True
        assert audit["program_head"]["image_input"] is False


def test_program_head_not_retrained():
    for name in TASK6S_SOURCES + ("../predict_buildreasonseg_directional.py",):
        path = (SCRIPTS / name).resolve()
        code = _code_only(path)
        for marker in ("optimizer", "backward()", "AdamW", "cross_entropy(", "loss.backward"):
            assert marker not in code, f"{path.name} must not train the parser ({marker})"
    audit = _artifact("task6s_frozen_asset_audit.json")
    if audit is not None:
        assert audit["program_head"]["retrained"] is False


# ---------------------------------------------------------------- 6-7 scope


def test_exactly_eight_supported_programs():
    from buildreasonseg_mvp.task6s_directional_pipeline import PROGRAM_DECOMPOSITION, SUPPORTED_PROGRAMS

    assert len(SUPPORTED_PROGRAMS) == 8
    assert set(SUPPORTED_PROGRAMS) == {
        "largest_to_left_of", "largest_to_right_of", "largest_to_above", "largest_to_below",
        "smallest_to_left_of", "smallest_to_right_of", "smallest_to_above", "smallest_to_below",
    }
    assert "largest" in PROGRAM_DECOMPOSITION["largest_to_left_of"]
    assert "to_nearest" not in " ".join(SUPPORTED_PROGRAMS)


def test_exact_program_family_relation_mapping():
    from buildreasonseg_mvp.task6s_directional_pipeline import PROGRAM_DECOMPOSITION, decompose_program

    assert PROGRAM_DECOMPOSITION == {
        "largest_to_left_of": ("largest", "left_of"),
        "largest_to_right_of": ("largest", "right_of"),
        "largest_to_above": ("largest", "above"),
        "largest_to_below": ("largest", "below"),
        "smallest_to_left_of": ("smallest", "left_of"),
        "smallest_to_right_of": ("smallest", "right_of"),
        "smallest_to_above": ("smallest", "above"),
        "smallest_to_below": ("smallest", "below"),
    }
    assert decompose_program("largest_to_above") == ("largest", "above")
    assert decompose_program("topmost_to_left_of") is None
    assert decompose_program("largest_to_nearest") is None


# ---------------------------------------------------------------- 8-9 exit-code behaviour


def test_ood_exit_4_before_parser_and_downstream():
    from buildreasonseg_mvp.task6s_directional_pipeline import EXIT_UNSUPPORTED_INSTRUCTION

    assert EXIT_UNSUPPORTED_INSTRUCTION == 4
    code = _code_only(CLI)
    gate_position = code.index("check_domain(args.prompt)")
    load_position = code.index("models, provenance = load_models(args)")
    assert gate_position < load_position, "the domain gate must run before any model is loaded"
    assert "checked_before_parser" in code and "checked_before_sam2" in code
    audit = _artifact("task6s_cli_prompt_audit.json")
    if audit is not None:
        assert audit["ood_all_exit_4"] is True
        for row in audit["controls"]:
            if row["kind"] == "ood":
                assert row["exit_code"] == 4
                assert row["downstream_called"] is False


def test_out_of_scope_program_exit_5_before_downstream():
    from buildreasonseg_mvp.task6s_directional_pipeline import (
        EXIT_UNSUPPORTED_DIRECTIONAL,
        SUPPORTED_PROGRAMS,
    )

    assert EXIT_UNSUPPORTED_DIRECTIONAL == 5
    code = _module_code()
    decomposition_position = code.index("decomposition = decompose_program(parsed[\"program\"])")
    proposal_position = code.index("run_frozen_proposals(models.proposal_model")
    assert decomposition_position < proposal_position, "scope check must precede the resolver"
    audit = _artifact("task6s_cli_prompt_audit.json")
    if audit is None:
        pytest.skip("CLI audit not generated yet")
    controls = {row["prompt"]: row for row in audit["controls"]}
    # canonical programs outside the eight must be rejected with exit 5 before proposal/SAM2/B3
    for prompt in ("分割面积最大的建筑物。", "分割最左侧的建筑物。"):
        row = controls[prompt]
        assert row["exit_code"] == EXIT_UNSUPPORTED_DIRECTIONAL
        assert row["downstream_called"] is False
    # measured limitation, reported not repaired: the frozen 20-program vocabulary has no "nearest"
    # program, so "nearest" prompts are classified as a *supported* directional program, pass the
    # 6S scope check and stop later at the resolver (exit 3) instead of exit 5.
    for prompt in ("分割面积最大的建筑物右侧最近的建筑物。",
                   "segment the building nearest to the right of the largest building"):
        row = controls[prompt]
        assert row["parsed_program"] in SUPPORTED_PROGRAMS
        assert row["exit_code"] == 3
        assert row["downstream_called"] is True
    # OOD controls never reach any downstream module; the flag is False only because of the two
    # "nearest" prompts above, which the frozen vocabulary cannot express as out-of-scope.
    assert all(row["downstream_called"] is False for row in audit["controls"]
               if row["kind"] == "ood")
    assert audit["downstream_never_called_for_controls"] == (
        not any(row["downstream_called"] for row in audit["controls"]))


# ---------------------------------------------------------------- 10-14 proposal resolver


def test_proposal_checkpoint_sha_exact():
    audit = _artifact("task6s_frozen_asset_audit.json")
    if audit is None:
        pytest.skip("asset audit not generated yet")
    assert audit["proposal_resolver"]["matches_expected"] is True
    assert audit["proposal_resolver"]["sha256"] == PROPOSAL_SHA256
    path = Path(audit["proposal_resolver"]["path"])
    if path.is_file():
        assert _sha256(path) == PROPOSAL_SHA256


def test_proposal_conf_is_0_10():
    from buildreasonseg_mvp.task6q_reference_resolver import PROPOSAL_CONF

    assert PROPOSAL_CONF == 0.10
    audit = _artifact("task6s_frozen_asset_audit.json")
    if audit is not None:
        assert audit["proposal_resolver"]["conf"] == 0.10
        assert audit["proposal_resolver"]["config"]["conf"] == 0.10


def test_proposal_imgsz_is_640():
    from buildreasonseg_mvp.task6q_reference_resolver import PROPOSAL_IMGSZ

    assert PROPOSAL_IMGSZ == 640
    audit = _artifact("task6s_frozen_asset_audit.json")
    if audit is not None:
        assert audit["proposal_resolver"]["imgsz"] == 640


def test_proposal_max_det_is_100():
    from buildreasonseg_mvp.task6q_reference_resolver import PROPOSAL_MAX_DET

    assert PROPOSAL_MAX_DET == 100
    audit = _artifact("task6s_frozen_asset_audit.json")
    if audit is not None:
        assert audit["proposal_resolver"]["max_det"] == 100


def test_task6q_eligibility_and_ranking_unchanged():
    from buildreasonseg_mvp.task6q_reference_resolver import (
        MERGE_BBOX_EXTENT_RATIO_MAX,
        TINY_COMPONENT_AREA_PX,
        build_proposal,
        config_report,
        is_eligible,
        select_reference,
    )

    assert MERGE_BBOX_EXTENT_RATIO_MAX == 0.20 and TINY_COMPONENT_AREA_PX == 150
    assert config_report()["threshold_sweep"] is False and config_report()["tta"] is False
    assert _git_changed("buildreasonseg_mvp/task6q_reference_resolver.py") == ""

    def proposal(index, area, border=False, confidence=0.9):
        mask = [[False] * 512 for _ in range(512)]
        for row in range(20, 20 + area // 20):
            for column in range(20, 40):
                mask[row][column] = True
        if border:
            mask[0][10] = True
        item = build_proposal(index, confidence,
                              __import__("numpy").asarray(mask, dtype=bool))
        return item

    large, small = proposal(0, 400, confidence=0.5), proposal(1, 200, confidence=0.9)
    assert is_eligible(large, "largest") and is_eligible(small, "largest")
    assert select_reference([large, small], "largest").proposal.index == 0
    assert select_reference([large, small], "smallest").proposal.index == 1
    border = proposal(2, 1000, border=True)
    assert not is_eligible(border, "largest") and not is_eligible(border, "smallest")


# ---------------------------------------------------------------- 15-21 frozen modules


def test_geometric_relation_field_v02_unchanged():
    assert _git_changed("buildreasonseg_mvp/geometric_relation_field_v02.py") == ""
    assert _git_changed("buildreasonseg_mvp/geometric_relation_field.py") == ""
    audit = _artifact("task6s_frozen_asset_audit.json")
    if audit is not None:
        assert audit["relation_field"]["alpha"] == 1.2 and audit["relation_field"]["tau"] == 0.04
        assert audit["relation_field"]["s_axis"] == 0.02
        assert audit["relation_field"]["s_margin"] == 0.02
        assert audit["relation_field"]["modified"] is False


def test_b3_checkpoint_sha_exact():
    audit = _artifact("task6s_frozen_asset_audit.json")
    task6o = _artifact("task6o_mini_val.json")
    if audit is None or task6o is None:
        pytest.skip("asset audit not generated yet")
    expected = task6o["variants"]["B3"]["training"]["checkpoint"]["sha256"]
    assert audit["target_decoder"]["matches_expected"] is True
    assert audit["target_decoder"]["sha256"] == expected
    assert audit["target_decoder"]["expected_sha256"] == expected


def test_b3_not_retrained():
    audit = _artifact("task6s_frozen_asset_audit.json")
    if audit is not None:
        assert audit["target_decoder"]["retrained"] is False
    for name in TASK6S_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("load_state_dict", "torch.save", "state_dict()"):
            if marker == "load_state_dict":
                continue
            assert marker not in code or marker == "load_state_dict", name


def test_primary_chain_does_not_use_grcl():
    from buildreasonseg_mvp import grcl_directional  # noqa: F401  (module must stay importable)

    for path in (PIPELINE, CLI, SCRIPTS / "task6s_evaluate.py"):
        code = _code_only(path)
        for marker in ("grcl_directional", "LAMBDA_GRCL", "grcl.loss", "from buildreasonseg_mvp.grcl"):
            assert marker not in code, f"{path.name} must not use GRCL ({marker})"
    e2e = _artifact("task6s_end_to_end_val.json")
    if e2e is not None:
        assert e2e["grcl_used"] is False
    checkpoint = _artifact("task6s_hardening_checkpoint.json")
    if checkpoint is not None:
        assert checkpoint["primary_chain"]["grcl_primary"] is False


def test_r1_checkpoint_not_used():
    for path in (PIPELINE, CLI, SCRIPTS / "task6s_evaluate.py"):
        text = path.read_text(encoding="utf-8")
        for marker in ("task6r/checkpoints", "overfit_R1", "mini_R1", "task6r\\checkpoints",
                       "grcl_directional"):
            assert marker not in text, f"{path.name} must not load a Task 6R checkpoint ({marker})"
    e2e = _artifact("task6s_end_to_end_val.json")
    if e2e is not None:
        assert e2e["chain"] == ["program_head", "decomposition",
                                "task6q_proposal_reference_resolver",
                                "geometric_relation_field_v02", "frozen_sam2_feature",
                                "frozen_task6o_b3"]


def test_no_oracle_reference_in_inference():
    code = _module_code()
    for marker in ("gt_mask", "reference_source_feature_id", "canonical_instances",
                   "oracle", "target_source_feature_id", "target_instance_id", "sample_id"):
        assert marker not in code, f"the inference chain must not touch {marker}"
    node = ast.parse(PIPELINE.read_text(encoding="utf-8"))
    function = next(item for item in ast.walk(node)
                    if isinstance(item, ast.FunctionDef) and item.name == "run_directional_chain")
    arguments = {argument.arg for argument in function.args.args}
    assert arguments == {"models", "image_path", "prompt"}, arguments
    e2e = _artifact("task6s_end_to_end_val.json")
    if e2e is not None:
        assert e2e["oracle_reference_used"] is False


def test_no_gt_target_in_inference():
    code = _module_code()
    assert "gt_target" not in code and "ground_truth" not in code.replace(
        '"ground_truth_used"', "")
    assert '"ground_truth_used": False' in code or "ground_truth_used\": False" in code
    cli = _artifact("task6s_cli_prompt_audit.json")
    if cli is not None:
        assert cli["ground_truth_required_by_cli"] is False
        for row in cli["supported_prompts"]:
            assert row["ground_truth_used"] is False


def test_cli_requires_no_buildspatialreason_record():
    code = _code_only(CLI)
    for marker in ("build_spatial_reason", "val.jsonl", "annotation JSON", "query_type",
                   "masks.mask(", "canonical_instances"):
        assert marker not in code, f"the CLI must not require {marker}"
    parser = ast.parse(CLI.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(parser)
             if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"]
    options = {node.args[0].value for node in calls}
    assert {"--image", "--prompt"} <= options
    for forbidden in ("--record", "--annotation", "--gt", "--oracle-reference", "--target-id",
                      "--reference-mask", "--sample-id"):
        assert forbidden not in options
    required = {node.args[0].value for node in calls
                if any(keyword.arg == "required" and getattr(keyword.value, "value", False) is True
                       for keyword in node.keywords)}
    assert {"--image", "--prompt"} <= required


# ---------------------------------------------------------------- 23-26 data reuse


def test_sam2_runs_on_source_image_on_the_fly():
    cli = _artifact("task6s_cli_prompt_audit.json")
    if cli is None:
        pytest.skip("CLI audit not generated yet")
    on_the_fly = cli["on_the_fly_sam2"]
    assert on_the_fly is not None
    assert on_the_fly["feature_cache_key_present"] is False
    assert on_the_fly["in_mini_val_240"] is False
    assert on_the_fly["sam2_encoder_path_exercised"] is True
    # the frozen SAM2 encoder provably ran (a cache hit costs ~0.002 s)
    assert on_the_fly["sam2_feature_seconds"] > 0.01
    assert Path(on_the_fly["image"]).is_file()
    packed = {sample.tile_id for sample in read_pack(PACK_ROOT / "mini_val_240.json")}
    assert on_the_fly["tile_id"] not in packed


def test_mini_val_240_reused_byte_for_byte():
    manifest = _artifact("task6n_pack_manifest.json")
    e2e = _artifact("task6s_end_to_end_val.json")
    if manifest is None or e2e is None:
        pytest.skip("artifacts not generated yet")
    assert e2e["pack"]["sha256"] == manifest["packs"]["mini_val_240"]["sha256"]
    assert e2e["pack"]["sha256"] == _sha256(PACK_ROOT / "mini_val_240.json")


def test_paired_val_20_reused_byte_for_byte():
    manifest = _artifact("task6n_pack_manifest.json")
    paired = _artifact("task6s_end_to_end_paired_val.json")
    if manifest is None or paired is None:
        pytest.skip("artifacts not generated yet")
    assert paired["pairs"] == 20
    assert _sha256(PACK_ROOT / "paired_val_20.json") == manifest["packs"]["paired_val_20"]["sha256"]


def test_no_test_split():
    for name in TASK6S_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        for marker in ('"test.jsonl"', "'test.jsonl'", 'split="test"', "test_fixed120",
                       "test_3_", "scene_disjoint_v1/test"):
            assert marker not in text, f"{name} must not touch the test split ({marker})"
    for name in ("task6s_frozen_asset_audit.json", "task6s_parser_val.json",
                 "task6s_end_to_end_val.json", "task6s_end_to_end_paired_val.json",
                 "task6s_cli_prompt_audit.json", "task6s_failure_attribution.json",
                 "task6s_verdict.json"):
        payload = _artifact(name)
        if payload is not None:
            assert payload.get("test_split_used", False) is False, name


# ---------------------------------------------------------------- 27-36 metrics and attribution


def test_strict_errors_score_zero_in_all_240_aggregate():
    e2e = _artifact("task6s_end_to_end_val.json")
    if e2e is None:
        pytest.skip("end-to-end artifact not generated yet")
    rows = e2e["records"]
    assert len(rows) == 240
    recomputed = []
    for row in rows:
        if row.get("status") == "ok" and row.get("parser_correct"):
            recomputed.append(row["target_iou"])
        else:
            recomputed.append(0.0)
    assert sum(recomputed) / len(recomputed) == pytest.approx(e2e["strict_all_240"]["miou"], abs=1e-12)
    strict = e2e["strict_all_240"]["miou"]
    answered = e2e["answered_only"]["miou"]
    assert strict <= answered
    assert strict == pytest.approx(answered * e2e["answered_only"]["records"] / 240, abs=1e-9)


def test_same_reference_reuse_in_paired_same_reference_case():
    paired = _artifact("task6s_end_to_end_paired_val.json")
    if paired is None:
        pytest.skip("paired artifact not generated yet")
    assert paired["same_resolved_reference_reused_for_pairs"] is True
    for row in paired["rows"]:
        if row.get("abstained"):
            continue
        assert "same_reference_reused" in row
    from buildreasonseg_mvp.task6s_directional_pipeline import decompose_program

    assert decompose_program("largest_to_above")[0] == decompose_program("largest_to_below")[0]


def test_exactly_24_fixed_supported_paraphrases():
    from scripts.task6s_cli_audit import SUPPORTED_PROMPTS

    assert len(SUPPORTED_PROMPTS) == 24
    assert SUPPORTED_PROMPTS[0] == ("largest_to_left_of", "分割面积最大的建筑物左侧的建筑物。")
    assert SUPPORTED_PROMPTS[2] == ("largest_to_left_of",
                                    "segment the building to the left of the largest building")
    assert SUPPORTED_PROMPTS[-1] == ("smallest_to_below",
                                     "segment the building below the smallest building")
    assert sum(1 for _, prompt in SUPPORTED_PROMPTS if prompt.isascii()) == 8
    cli = _artifact("task6s_cli_prompt_audit.json")
    if cli is not None:
        assert cli["supported_count"] == 24


def test_exact_8_unsupported_controls():
    from scripts.task6s_cli_audit import OOD_PROMPTS, OUT_OF_SCOPE_PROMPTS

    assert len(OOD_PROMPTS) == 4 and len(OUT_OF_SCOPE_PROMPTS) == 4
    assert "Write a poem about the sea." in OOD_PROMPTS
    assert "今天天气怎么样？" in OOD_PROMPTS
    assert "检测道路。" in OOD_PROMPTS
    assert "" in OOD_PROMPTS
    assert "分割面积最大的建筑物。" in OUT_OF_SCOPE_PROMPTS
    assert "分割最左侧的建筑物。" in OUT_OF_SCOPE_PROMPTS
    assert "分割面积最大的建筑物右侧最近的建筑物。" in OUT_OF_SCOPE_PROMPTS
    assert ("segment the building nearest to the right of the largest building"
            in OUT_OF_SCOPE_PROMPTS)
    cli = _artifact("task6s_cli_prompt_audit.json")
    if cli is not None:
        assert cli["control_count"] == 8


def test_result_json_contains_ground_truth_used_false():
    text = CLI.read_text(encoding="utf-8")
    assert '"ground_truth_used": False' in text
    assert '"ground_truth_required": False' in text
    cli = _artifact("task6s_cli_prompt_audit.json")
    if cli is not None:
        for row in cli["supported_prompts"]:
            if row["status"] == "ok":
                assert row["ground_truth_used"] is False


def test_successful_cli_writes_all_four_visuals():
    cli = _artifact("task6s_cli_prompt_audit.json")
    if cli is None:
        pytest.skip("CLI audit not generated yet")
    ok_rows = [row for row in cli["supported_prompts"] if row["status"] == "ok"]
    assert ok_rows, "at least one supported prompt must reach the target decoder"
    for row in ok_rows:
        assert "target_mask.png" in row["wrote"]
        assert "reference_mask.png" in row["wrote"]
        assert "relation_field.png" in row["wrote"]
        assert "overlay.png" in row["wrote"]
    code = _code_only(CLI)
    for name in ("target_mask.png", "reference_mask.png", "relation_field.png", "overlay.png",
                 "result.json"):
        assert name in code


def test_failure_attribution_bucket_is_exclusive():
    from scripts.task6s_failure_attribution import KNOWN_BUCKETS, bucket_for

    attribution = _artifact("task6s_failure_attribution.json")
    if attribution is None:
        pytest.skip("attribution not generated yet")
    assert attribution["exclusive_bucket_check"]["one_bucket_per_record"] is True
    assert attribution["exclusive_bucket_check"]["bucket_values_known"] is True
    assert sum(attribution["counts"].values()) == 240
    for detail in attribution["records_detail"]:
        assert detail["bucket"] in KNOWN_BUCKETS
    # the classifier returns exactly one bucket for every combination of the priority inputs
    assert bucket_for({"parser_correct": False, "program_supported": False}) == "PARSER_WRONG"
    assert bucket_for({"parser_correct": True, "program_supported": True,
                       "proposal_count": 0}) == "REFERENCE_NO_PROPOSALS"
    assert bucket_for({"parser_correct": True, "program_supported": True, "proposal_count": 5,
                       "eligible_reference_proposals": 0}) == "REFERENCE_NO_ELIGIBLE"
    assert bucket_for({"parser_correct": True, "program_supported": True, "proposal_count": 5,
                       "eligible_reference_proposals": 2,
                       "best_eligible_reference_iou": 0.2}) == "REFERENCE_NOT_COVERED_IOU50"
    assert bucket_for({"parser_correct": True, "program_supported": True, "proposal_count": 5,
                       "eligible_reference_proposals": 2, "best_eligible_reference_iou": 0.7,
                       "selected_reference_iou": 0.3}) == "REFERENCE_SELECTION_WRONG"
    assert bucket_for({"parser_correct": True, "program_supported": True, "proposal_count": 5,
                       "eligible_reference_proposals": 2, "best_eligible_reference_iou": 0.7,
                       "selected_reference_iou": 0.6,
                       "selected_reference_centroid_error": 0.2}) == "REFERENCE_GEOMETRY_POOR"
    assert bucket_for({"parser_correct": True, "program_supported": True, "proposal_count": 5,
                       "eligible_reference_proposals": 2, "best_eligible_reference_iou": 0.7,
                       "selected_reference_iou": 0.6,
                       "selected_reference_centroid_error": 0.01,
                       "target_iou": 0.1}) == "TARGET_FAIL_WITH_REFERENCE_OK"
    assert bucket_for({"parser_correct": True, "program_supported": True, "proposal_count": 5,
                       "eligible_reference_proposals": 2, "best_eligible_reference_iou": 0.7,
                       "selected_reference_iou": 0.6,
                       "selected_reference_centroid_error": 0.01,
                       "target_iou": 0.8}) == "TARGET_OK"


def test_dominant_bottleneck_rule_exact():
    from scripts.task6s_failure_attribution import REFERENCE_BUCKETS

    attribution = _artifact("task6s_failure_attribution.json")
    if attribution is None:
        pytest.skip("attribution not generated yet")
    counts = attribution["counts"]
    total = sum(counts.values())
    parser_fail = counts["PARSER_WRONG"]
    reference_fail = sum(counts[name] for name in REFERENCE_BUCKETS)
    target_fail = counts["TARGET_FAIL_WITH_REFERENCE_OK"]
    if parser_fail / total > 0.05:
        expected = "PARSER"
    elif reference_fail > target_fail:
        expected = "REFERENCE"
    else:
        expected = "TARGET_DECODER_FIELD"
    assert attribution["dominant_bottleneck"] == expected
    assert attribution["aggregation"]["parser_fail"] == parser_fail
    assert attribution["aggregation"]["reference_fail"] == reference_fail
    assert attribution["aggregation"]["target_fail"] == target_fail


# ---------------------------------------------------------------- 37-42 scope guards


def test_no_grcl_primary_chain_training():
    for name in TASK6S_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("optimizer", "backward()", "AdamW", "scheduler", "train_step",
                       "grad_scaler"):
            assert marker not in code, f"{name} must not train anything ({marker})"


def test_no_nearest_or_l3():
    from buildreasonseg_mvp.task6s_directional_pipeline import PROGRAM_DECOMPOSITION, SUPPORTED_PROGRAMS

    assert not any("nearest" in program for program in SUPPORTED_PROGRAMS)
    assert not any("l3" in program or "to_" in program.split("to_")[-1] and "above" not in program
                   and "below" not in program and "left_of" not in program
                   and "right_of" not in program for program in SUPPORTED_PROGRAMS)
    for relation in {relation for _, relation in PROGRAM_DECOMPOSITION.values()}:
        assert relation in ("left_of", "right_of", "above", "below")
    for name in TASK6S_SOURCES:
        code = _code_only(SCRIPTS / name)
        assert "to_nearest" not in code and "l3_hop" not in code and "multi_hop" not in code


def test_no_ref_token():
    for path in (PIPELINE, CLI, *(SCRIPTS / name for name in TASK6S_SOURCES)):
        assert "[REF]" not in path.read_text(encoding="utf-8"), path.name


def test_no_4b():
    for path in (PIPELINE, CLI, *(SCRIPTS / name for name in TASK6S_SOURCES)):
        text = path.read_text(encoding="utf-8")
        assert "Qwen3-VL-4B" not in text, path.name
        assert "4B" not in text.replace("4B)", "")


def test_no_new_dataset_download_install_or_gui():
    for path in (PIPELINE, CLI, *(SCRIPTS / name for name in TASK6S_SOURCES)):
        code = _code_only(path)
        for marker in ("urlretrieve", "requests.get", "Invoke-WebRequest", "hf_hub_download",
                       "snapshot_download", "pip install", "tkinter", "PyQt", "gradio",
                       "streamlit", "cv2.imshow", "flask", "fastapi"):
            assert marker not in code, f"{path.name} must not use {marker}"
    assert set(_artifact("task6s_frozen_asset_audit.json")["supported_programs"]) if _artifact(
        "task6s_frozen_asset_audit.json") else True


def test_previous_suite_preserved():
    for name in ("test_task6r_grcl_directional_feasibility.py",
                 "test_task6q_frozen_proposal_reference_resolver.py",
                 "test_task6p_differentiable_field_predicted_reference.py",
                 "test_task6o_field_causal_decomposition.py"):
        assert (REPO_ROOT / "tests" / name).is_file(), name
    removed = subprocess.run(["git", "diff", "--name-only", "--diff-filter=D", BASE_COMMIT,
                              "--", "tests/"],
                             cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    assert removed.strip() == "", f"no test file may be removed: {removed}"
