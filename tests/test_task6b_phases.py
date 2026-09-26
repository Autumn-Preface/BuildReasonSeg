"""Task 6B phase / validation / inference-validity tests.

Covers, from the Task 6B section 21 list:

 8. the Phase A optimiser excludes the projection and the SAM2 mask decoder
 9. the Phase B optimiser includes them
10. the Qwen vision tower stays frozen in both phases
11. the teacher-forced and free-generation paths are separate
12. the strict metric scores a generation failure as zero
13. the paired probe compares each prediction against both ground-truth masks
14. no ground-truth geometry enters inference
15. no `[REF]` in Task 6B

Run with pytest, or directly::

    python tests/test_task6b_phases.py
"""

from __future__ import annotations

import inspect
import re
import sys
import tokenize
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from buildreasonseg_mvp import validation as V  # noqa: E402

MVP_PACKAGE = REPO_ROOT / "buildreasonseg_mvp"
TASK6B_SCRIPTS = sorted((REPO_ROOT / "scripts").glob("task6b_*.py"))
VISUAL_MARKERS = ("visual", "vision_tower", "vision_model")


def test_phase_trainables_are_correct():
    from task6a_fixtures import require_model_assets, runtime
    from buildreasonseg_mvp.runtime import set_phase_trainables

    require_model_assets()
    rt = runtime()

    phase_a = set_phase_trainables(rt.model, "A")
    assert phase_a["projection_trainable"] is False
    assert phase_a["sam_mask_decoder_trainable"] is False
    assert phase_a["qwen_visual_trainable"] == []

    optimiser_a = rt.model.trainable_parameter_groups(1e-4, 3e-4, 0.01, 3e-4)
    names_a = [group.get("name") for group in optimiser_a]
    assert "decoder" not in names_a, f"Phase A optimiser must exclude the decoder group: {names_a}"
    a_ids = {id(p) for group in optimiser_a for p in group["params"]}
    assert not (a_ids & {id(p) for p in rt.model.projection.parameters()})
    assert not (a_ids & {id(p) for p in rt.model.sam.sam_mask_decoder.parameters()})

    phase_b = set_phase_trainables(rt.model, "B")
    assert phase_b["projection_trainable"] is True
    assert phase_b["sam_mask_decoder_trainable"] is True
    assert phase_b["qwen_visual_trainable"] == []

    optimiser_b = rt.model.trainable_parameter_groups(1e-4, 3e-4, 0.01, 3e-4)
    b_ids = {id(p) for group in optimiser_b for p in group["params"]}
    assert b_ids & {id(p) for p in rt.model.projection.parameters()}
    assert b_ids & {id(p) for p in rt.model.sam.sam_mask_decoder.parameters()}

    # in both phases the vision tower and the frozen SAM2 encoder stay out
    for ids in (a_ids, b_ids):
        assert not (ids & {id(p) for p in rt.model.sam.image_encoder.parameters()})
        assert not (ids & {id(p) for p in rt.model.sam.sam_prompt_encoder.parameters()})
    print("  [8/9/10] phase trainable sets OK")


def test_qwen_vision_tower_frozen_in_both_phases():
    from task6a_fixtures import require_model_assets, runtime
    from buildreasonseg_mvp.runtime import set_phase_trainables

    require_model_assets()
    rt = runtime()
    for phase in ("A", "B"):
        set_phase_trainables(rt.model, phase)
        for name, parameter in rt.model.named_parameters():
            if any(marker in name for marker in VISUAL_MARKERS):
                assert not parameter.requires_grad, f"{phase}: {name} is trainable"
    set_phase_trainables(rt.model, "B")
    print("  [10] Qwen vision tower frozen in both phases OK")


def test_strict_metric_scores_generation_failure_as_zero():
    records = [
        {"seg_valid": True, "iou": 0.8, "dice": 0.9, "iou_conditional": 0.8, "dice_conditional": 0.9,
         "collapse": None, "reasoning_exact_match": True, "reasoning_char_similarity": 1.0,
         "operation_chain_correct": True, "generated_token_count": 20},
        {"seg_valid": False, "iou": 0.0, "dice": 0.0, "iou_conditional": None, "dice_conditional": None,
         "collapse": None, "reasoning_exact_match": False, "reasoning_char_similarity": 0.1,
         "operation_chain_correct": False, "generated_token_count": 5},
    ]
    aggregate = V.aggregate(records)
    assert aggregate["valid_seg_emission_rate"] == 0.5
    assert abs(aggregate["strict_end_to_end_miou"] - 0.4) < 1e-9, aggregate
    assert abs(aggregate["conditional_miou"] - 0.8) < 1e-9, aggregate
    print("  [12] strict metric zeros an invalid emission, conditional does not OK")


def test_paired_pass_rule_compares_each_prediction_against_both_targets():
    # A matches GT-A better than GT-B, B matches GT-B better than GT-A -> pass
    assert V.pair_pass_decision(0.9, 0.0, 0.0, 0.9) is True
    # A and B both prefer GT-A -> fail, even though one of them is "right"
    assert V.pair_pass_decision(0.9, 0.1, 0.8, 0.2) is False
    # swapped case
    assert V.pair_pass_decision(0.2, 0.8, 0.8, 0.2) is False
    # ties fail
    assert V.pair_pass_decision(0.5, 0.5, 0.5, 0.5) is False
    print("  [13] paired pass rule compares each prediction against both GTs OK")


def _code_without_comments_or_strings(path: Path) -> str:
    pieces: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for token in tokenize.generate_tokens(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            pieces.append(token.string)
    return " ".join(pieces)


def test_no_ref_in_task6b():
    for path in list(MVP_PACKAGE.glob("*.py")) + TASK6B_SCRIPTS:
        code = _code_without_comments_or_strings(path)
        assert "[REF]" not in code, f"{path.name} uses [REF] in code"
    print("  [15] no [REF] in Task 6B OK")


def test_inference_path_takes_no_ground_truth_geometry():
    forbidden = ("gt_", "ground_truth", "centroid", "bbox", "target_id", "reference_mask")
    for function in (V.free_generation_validation, V.paired_probe):
        for name in inspect.signature(function).parameters:
            assert not any(token in name.lower() for token in forbidden), (
                f"{function.__qualname__} exposes {name!r}"
            )
    # sampling helpers take only the sample list and the reasoning lookup
    assert list(inspect.signature(V.free_generation_validation).parameters)[:3] == [
        "runtime",
        "samples",
        "lookup",
    ]
    print("  [14] inference path takes no GT geometry OK")


def test_paths_are_separate_functions():
    """Teacher-forced and free-generation validation must be distinct entry points."""

    source = inspect.getsource(V)
    assert "def teacher_forced_validation" in source
    assert "def free_generation_validation" in source
    assert "generate_with_seg" in inspect.getsource(V.free_generation_validation)
    assert "generate_with_seg" not in inspect.getsource(V.teacher_forced_validation)
    print("  [11] teacher-forced and free-generation paths are separate OK")


def test_no_forbidden_components_in_task6b():
    forbidden = {
        "spatial relation encoder": re.compile(r"class\s+\w*RelationEncoder", re.IGNORECASE),
        "spatial consistency loss": re.compile(r"class\s+\w*ConsistencyLoss", re.IGNORECASE),
    }
    for path in TASK6B_SCRIPTS:
        code = _code_without_comments_or_strings(path)
        for label, pattern in forbidden.items():
            assert not pattern.search(code), f"{path.name} implements {label}"

    import yaml

    config = yaml.safe_load((REPO_ROOT / "configs" / "mvp" / "task6b_2b_minitrain.yaml").read_text(encoding="utf-8"))
    assert "4B" not in config["models"]["qwen_model_id"]
    print("  [extra] no forbidden Task 6B components OK")


def test_best_joint_rule_is_declared_before_phase_b():
    """The selection rule must be fixed in the config, not tuned after results."""

    import yaml

    config = yaml.safe_load((REPO_ROOT / "configs" / "mvp" / "task6b_2b_minitrain.yaml").read_text(encoding="utf-8"))
    rule = config["validation"]["best_joint_rule"]
    assert rule == ["valid_seg_emission_rate", "strict_end_to_end_miou", "operation_chain_accuracy", "lm_ce"]

    source = (REPO_ROOT / "scripts" / "task6b_train.py").read_text(encoding="utf-8")
    assert 'BEST_JOINT_RULE = ("valid_seg_emission_rate", "strict_end_to_end_miou", "operation_chain_accuracy", "lm_ce")' in source
    print("  [extra] best_joint rule declared before Phase B OK")


def main() -> int:
    tests = [
        ("8/9 phase trainables", test_phase_trainables_are_correct),
        ("10 vision frozen", test_qwen_vision_tower_frozen_in_both_phases),
        ("12 strict metric", test_strict_metric_scores_generation_failure_as_zero),
        ("13 paired rule", test_paired_pass_rule_compares_each_prediction_against_both_targets),
        ("15 no [REF]", test_no_ref_in_task6b),
        ("14 no GT geometry", test_inference_path_takes_no_ground_truth_geometry),
        ("11 separate paths", test_paths_are_separate_functions),
        ("no forbidden components", test_no_forbidden_components_in_task6b),
        ("best_joint rule", test_best_joint_rule_is_declared_before_phase_b),
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
    print(f"\n{len(tests) - failures}/{len(tests)} task6b phase checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
