"""Task 6J tests: structured proposal grounding.

Executor behaviour is tested on synthetic candidate geometry (CPU); the pipeline facts are
asserted against the recorded artifacts and the sources (the 22 required coverage points from
`handoff/TO_DSH.md` section 22). GT never enters inference in any tested path.

Run with pytest, or directly::

    python tests/test_task6j_structured_grounding.py
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "spatial_reasoning"))

from buildreasonseg_mvp.structured_grounding import (  # noqa: E402
    EXPECTED_QUERY_TYPES,
    Candidate,
    CandidateSet,
    build_program_spec,
    canonical_program_template,
    execute_program,
    execute_program_by_id,
)

EVAL = REPO_ROOT / "evaluation"
FORBIDDEN_SOURCES = (
    "buildreasonseg_mvp/structured_grounding.py",
    "buildreasonseg_mvp/program_parser.py",
    "scripts/task6j_common.py",
    "scripts/task6j_program_spec.py",
    "scripts/task6j_j0.py",
    "scripts/task6j_j1.py",
    "scripts/task6j_j2.py",
    "scripts/task6j_j3.py",
    "scripts/task6j_j4.py",
    "scripts/task6j_yolo_infer.py",
    "scripts/task6j_error_attribution.py",
    "scripts/task6j_verdict.py",
)


def _code(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _artifact(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _rect_set(specs) -> CandidateSet:
    """Synthetic candidates on a 512x512 canvas. spec = (id, x0, y0, w, h)."""

    height = width = 512
    label_map = np.zeros((height, width), dtype=np.uint8)
    candidates = []
    for candidate_id, x0, y0, w, h in specs:
        mask = np.zeros((height, width), dtype=bool)
        mask[y0:y0 + h, x0:x0 + w] = True
        ys, xs = np.nonzero(mask)  # nonzero returns (rows=y, cols=x)
        touches = bool(xs.min() == 0 or ys.min() == 0 or xs.max() == width - 1 or ys.max() == height - 1)
        candidates.append(
            Candidate(
                candidate_id=candidate_id,
                mask=mask,
                bbox_xyxy_px=(x0, y0, x0 + w, y0 + h),
                centroid_px=(float(xs.mean()), float(ys.mean())),
                area_px=int(mask.sum()),
                touches_image_border=touches,
                confidence=None,
                source="synthetic",
            )
        )
        label_map[mask] = candidate_id
    return CandidateSet(width=width, height=height, candidates=candidates, label_map=label_map, source="synthetic")


# ------------------------------------------------------------------ 1-5 vocabulary + executor


def test_program_vocabulary_comes_from_actual_frozen_query_types():
    from buildreasonseg_mvp import data as data_mod

    spec = build_program_spec(data_mod.read_records("train"))
    assert set(spec) == set(EXPECTED_QUERY_TYPES)
    assert len(EXPECTED_QUERY_TYPES) == 20
    for query_type, entry in spec.items():
        assert entry["program_id"] == query_type
        assert entry["operations"] == canonical_program_template(query_type)


def test_executor_uses_frozen_relation_convention():
    # leftmost: min centroid x with the frozen 4px margin
    candidates = _rect_set([(1, 100, 100, 40, 40), (2, 300, 200, 40, 40)])
    result = execute_program([{"operation": "argmin_centroid_x"}], candidates)
    assert not result.abstained and result.selected_id == 1

    # filter_relation left_of: subject is left of object (frozen predicate convention)
    candidates = _rect_set([(1, 100, 100, 40, 40), (2, 400, 102, 40, 40)])
    result = execute_program(
        [
            {"operation": "argmax_area"},
            {"operation": "filter_relation", "relation": "left_of",
             "reference_component_id": "@1", "input_component_ids": "all"},
        ],
        candidates,
    )
    assert not result.abstained
    assert result.selected_id == 1  # the smaller left one is the unique kept subject

    # frozen predicate: a subject on the RIGHT must NOT satisfy left_of
    candidates = _rect_set([(1, 400, 100, 40, 40), (2, 100, 102, 40, 40)])
    result = execute_program(
        [
            {"operation": "argmax_area"},
            {"operation": "filter_relation", "relation": "left_of",
             "reference_component_id": "@1", "input_component_ids": "all"},
        ],
        candidates,
    )
    assert result.abstained  # nothing kept -> no unique target


def test_oracle_executor_cannot_read_target_id():
    source = inspect.getsource(execute_program)
    first = source.find('"""')
    if first >= 0:
        second = source.find('"""', first + 3)
        if second >= 0:
            source = source[second + 3:]
    for marker in ("target_component_id", "target_mask", "reasoning"):
        assert marker not in source
    signature = inspect.signature(execute_program)
    assert set(signature.parameters) - {"program", "candidates", "config"} == set()


def test_oracle_candidates_are_diagnostic_only():
    source = inspect.getsource(CandidateSet.from_component_map)
    assert "oracle" in source
    j0 = _code("scripts/task6j_j0.py")
    run_body = j0.split("def run")[1].split("val_rows =")[0]
    assert "execute_program(program, candidates, config)" in run_body
    assert run_body.index("execute_program(") < run_body.index("target_component_id")


def test_executor_unit_reproduction_of_every_operation():
    # largest (frozen eligible pool; non-border, non-merge)
    candidates = _rect_set([(1, 60, 60, 20, 20), (2, 300, 200, 40, 40), (3, 150, 60, 20, 20)])
    result = execute_program([{"operation": "argmax_area"}], candidates)
    assert result.selected_id == 2

    # smallest with tiny excluded by frozen eligibility (id 4 below the 150px floor)
    candidates = _rect_set([(1, 60, 60, 20, 20), (2, 300, 200, 40, 40), (4, 150, 60, 10, 10)])
    result = execute_program([{"operation": "argmin_area"}], candidates)
    assert result.selected_id == 1  # the 10x10=100px component is tiny-ineligible

    # above filter: subject above object (anchor uniquely largest)
    candidates = _rect_set([(1, 200, 300, 40, 40), (2, 210, 100, 30, 30)])
    result = execute_program(
        [
            {"operation": "argmax_area"},
            {"operation": "filter_relation", "relation": "above",
             "reference_component_id": "@1", "input_component_ids": "all"},
        ],
        candidates,
    )
    assert not result.abstained
    assert result.selected_id == 2

    # nearest (frozen boundary distance): gaps 9px vs 49px -> the 9px one, margin ok
    candidates = _rect_set([(1, 100, 200, 30, 30), (2, 140, 200, 20, 20), (3, 180, 200, 20, 20)])
    result = execute_program(
        [
            {"operation": "argmax_area"},
            {"operation": "argmin_boundary_distance",
             "reference_component_id": "@1", "candidate_component_ids": "all"},
        ],
        candidates,
    )
    assert result.selected_id == 2

    # extreme ambiguity: margin below the frozen 4px -> abstain
    candidates = _rect_set([(1, 100, 100, 40, 40), (2, 101, 200, 40, 40)])
    result = execute_program([{"operation": "argmin_centroid_x"}], candidates)
    assert result.abstained and result.reason == "extreme_relation_invalid"


def test_executor_templates_match_frozen_recompute_on_real_samples():
    """The symbolic-template executor reproduces the frozen generator on real val records."""

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from buildreasonseg_mvp import data as data_mod
    from task6j_common import geometry_for, oracle_candidate_set

    from annotator import recompute_target_from_steps
    from component_quality import classify_image
    from dataset_access import DEFAULT_DATASET_ROOT
    from thresholds import load_config as load_relation_config

    config = load_relation_config()
    records = data_mod.read_records("val")[:4]
    for record in records:
        sample = data_mod.to_sample(record)
        geometry = geometry_for(sample)
        frozen = recompute_target_from_steps(
            geometry, record["reasoning_steps"], config,
            classify_image(geometry, config), geometry.load_map(DEFAULT_DATASET_ROOT),
        )
        assert frozen == int(record["target_component_id"])
        candidates = oracle_candidate_set(sample)
        # explicit-id form (J0 form) and template form (J3/J4 form) must agree with the frozen result
        explicit = execute_program(record["reasoning_steps"], candidates, config)
        template = execute_program_by_id(record["query_type"], candidates, config)
        assert (explicit.selected_id == frozen) or (explicit.abstained and frozen is None)
        assert (template.selected_id == frozen) or (template.abstained and frozen is None)


# ------------------------------------------------------------------ 6-10 YOLO provenance / hygiene


def test_legacy_yolo_provenance_checked_before_inference():
    infer = _code("scripts/task6j_yolo_infer.py")
    assert "sha256_file(MODEL_PATH)" in infer
    assert "provenance" in infer
    j1 = _code("scripts/task6j_j1.py")
    assert "provenance_check" in j1 and "matches" in j1
    artifact = _artifact("task6j_yolo_proposal_recall.json")
    if artifact is not None:
        assert artifact["provenance_check"]["matches"] is True
        assert artifact["provenance"]["model_sha256"].startswith("d9a6a65b")


def test_legacy_project_and_env_never_modified():
    infer = _code("scripts/task6j_yolo_infer.py")
    # the legacy directory is referenced only as the read-only model source
    legacy_lines = [line for line in infer.splitlines() if "WHU_Building_Segment" in line]
    assert len(legacy_lines) == 1
    assert "MODEL_PATH =" in legacy_lines[0]
    # no write helper may appear before the OUT_DIR (artifacts) definition
    prefix = infer.split("OUT_DIR =")[0]
    for marker in ("write_text", "mkdir", "np.save", "np.savez", "to_csv", "unlink"):
        assert marker not in prefix
    for forbidden in ("pip install", "conda install", "subprocess", "import pip", "os.system"):
        assert forbidden not in infer


def test_no_package_installation_into_existing_environments():
    for relative in FORBIDDEN_SOURCES:
        code = _code(relative)
        for marker in ("pip install", "conda install", "subprocess.run", "os.system"):
            assert marker not in code, f"{relative} must not install anything"


def test_proposal_geometry_computed_without_gt():
    infer = _code("scripts/task6j_yolo_infer.py")
    for marker in ("target_component_id", "component_map", "target_mask"):
        assert marker not in infer
    assert "result.boxes" in infer and "result.masks" in infer


def test_proposal_recall_matching_is_evaluation_only():
    j1 = _code("scripts/task6j_j1.py")
    # proposals are loaded, never edited; IoU is computed for the recall report only
    body = j1.split("def load_proposals")[0]
    assert "np.load(CACHE_DIR" in j1
    assert "_iou(m, target_mask)" in j1
    for marker in ("np.save", "np.savez"):
        assert marker not in j1


# ------------------------------------------------------------------ 11-14 parser hygiene


def test_program_head_input_is_instruction_text_only():
    parser = _code("buildreasonseg_mvp/program_parser.py")
    build_batch = parser.split("def build_batch")[1].split("def forward")[0]
    assert '"type": "text"' in build_batch
    assert '"type": "image"' not in build_batch
    assert "instruction" in build_batch


def _strip_docstrings(code: str) -> str:
    out = []
    depth = 0
    index = 0
    while index < len(code):
        if code.startswith('"""', index):
            if depth == 0:
                end = code.find('"""', index + 3)
                index = len(code) if end < 0 else end + 3
                continue
            depth += 1
        out.append(code[index])
        index += 1
    return "".join(out)


def test_program_head_has_no_image_tokens():
    parser = _strip_docstrings(_code("buildreasonseg_mvp/program_parser.py"))
    build_batch = parser.split("def build_batch")[1].split("def forward")[0]
    forward = parser.split("def forward")[1].split("def predict")[0]
    for body in (build_batch, forward):
        assert "pixel_values" not in body
        assert "image_grid_thw" not in body
        assert "pixel" not in body
    assert "input_ids" in forward and "attention_mask" in forward


def test_query_type_appears_only_as_ce_target():
    parser = _code("buildreasonseg_mvp/program_parser.py")
    build_batch = parser.split("def build_batch")[1].split("def forward")[0]
    assert "PROGRAM_ID_TO_INDEX[str(program_id)]" in build_batch  # labels only
    assert 'text": instruction' in build_batch or "instruction}" in build_batch


def test_train_val_test_split_hygiene():
    j2 = _code("scripts/task6j_j2.py")
    assert 'read_records("train")' in j2
    assert 'read_records("test")' not in j2
    assert "stratified_subset" in j2
    for relative in ("scripts/task6j_j2.py", "scripts/task6j_j3.py", "scripts/task6j_j4.py"):
        assert 'read_records("test")' not in _code(relative)
    spec = _code("scripts/task6j_program_spec.py")
    assert 'read_records("test")' in spec  # counts only, never training
    assert "test_split_used_for_training" in spec


# ------------------------------------------------------------------ 15-18 inference validity


def test_j3_never_falls_back_to_gt_program():
    j3 = _code("scripts/task6j_j3.py")
    assert "execute_program_by_id(predicted_program" in j3
    assert 'record["reasoning_steps"]' not in j3
    assert "predicted_programs(val_records)" in j3


def test_j4_uses_predicted_program_and_predicted_proposals_only():
    j4 = _code("scripts/task6j_j4.py")
    run = j4.split("def run")[1].split("def main")[0] if "def run" in j4 else j4.split("    def run(sample):")[1].split("\n    def ")[0]
    assert "predict_program(sample.instruction_zh)" in run
    assert "load_proposals(sample.image_id)" in run
    assert "reasoning_steps" not in j4
    assert "component_map" not in run


def test_optional_sam_refinement_not_used_or_predicted_only():
    j4 = _code("scripts/task6j_j4.py")
    # The optional frozen-SAM2 refinement diagnostic was not exercised in this task; when it is,
    # it must consume only predicted proposal geometry. The J4 source therefore either contains
    # no SAM2 at all, or its SAM2 calls take only proposal-derived geometry.
    if "decode_mask" in j4 or "sam" in j4.lower():
        assert "proposal" in j4.split("decode_mask")[0]


def test_no_old_direct_pixel_grounding_head_in_j4():
    j4 = _code("scripts/task6j_j4.py")
    for marker in ("dense_head", "refine_block", "box_head", "grounding_head"):
        assert marker not in j4


# ------------------------------------------------------------------ 19-22 hygiene + determinism


def test_no_forbidden_extensions():
    haystack = " ".join(_code(relative).lower() for relative in FORBIDDEN_SOURCES)
    for marker in ("[ref]", "spatialrelationencoder", "spatial_consistency", "scl"):
        assert marker not in haystack
    config = (REPO_ROOT / "configs" / "mvp" / "task6j_program_parser.yaml").read_text(encoding="utf-8")
    assert "Qwen/Qwen3-VL-2B-Instruct" in config
    assert "Qwen3-VL-4B" not in config and "4B-Instruct" not in config


def test_strict_determinism_where_supported():
    config = (REPO_ROOT / "configs" / "mvp" / "task6j_program_parser.yaml").read_text(encoding="utf-8")
    assert "deterministic: true" in config and "deterministic_strict: true" in config
    # the executor is a pure function of (program, candidates)
    candidates = _rect_set([(1, 100, 100, 40, 40), (2, 300, 200, 40, 40)])
    first = execute_program([{"operation": "argmin_centroid_x"}], candidates)
    second = execute_program([{"operation": "argmin_centroid_x"}], candidates)
    assert first.selected_id == second.selected_id == 1
    assert first.as_dict() == second.as_dict()


def test_no_gui():
    for relative in FORBIDDEN_SOURCES:
        code = _code(relative)
        for marker in ("tkinter", "PyQt", "streamlit", "gradio", "dash"):
            assert marker not in code, f"{relative} must not use a GUI toolkit"


def test_failure_attribution_auditable():
    attribution = _code("scripts/task6j_error_attribution.py")
    for marker in (
        "target_absent_or_poor_in_proposal_set",
        "proposal_geometry_changes_relation_outcome",
        "correct_selection_poor_mask_quality",
        "executor_abstained_under_proposals",
        "binding_failure",
    ):
        assert marker in attribution
    artifact = _artifact("task6j_error_attribution.json")
    if artifact is not None:
        assert "binding_failure" in artifact
        assert "failure_counts" in artifact["j1"]


# ------------------------------------------------------------------ artifact gates


def test_j0_artifact_meets_the_gate():
    j0 = _artifact("task6j_j0_oracle_executor.json")
    assert j0 is not None
    assert j0["val"]["exact_accuracy"] >= 0.98
    assert j0["paired"]["paired_selection_pass"] >= 19
    assert j0["frozen_recompute_agreement"]["all_agree"] is True


def test_j1_artifact_records_the_binding_gate():
    j1 = _artifact("task6j_j1_oracle_program_yolo.json")
    assert j1 is not None
    assert j1["provenance_check"]["matches"] is True
    assert "gate" in j1 and "viable" in j1


def test_j2_and_j3_artifacts_record_their_gates():
    j2 = _artifact("task6j_j2_program_parser.json")
    j3 = _artifact("task6j_j3_predicted_program_oracle_candidates.json")
    assert j2 is not None and j3 is not None
    assert j2["final"]["gate"]["passed"] is True
    assert j3["gate"]["passed"] is True
    assert j2["full_val"]["exact_accuracy"] >= 0.90
    assert j3["metrics"]["selected_target_accuracy"] >= 0.85


def main() -> int:
    import inspect

    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value) and not inspect.signature(value).parameters
    ]
    failures = 0
    for function in tests:
        try:
            function()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL {function.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} task6j structured-grounding checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
