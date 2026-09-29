"""Task 6P tests (section 20): 31 checks on the differentiable field v0.2, the reference head and the
predicted-reference chain."""

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
REF_ROOT = REPO_ROOT / "artifacts" / "task6p" / "reference_packs"
BASE_COMMIT = "595e7bb4867a6a91dd6f3cd8e0671e7862b4c6b8"

TASK6P_SOURCES = ("task6p_freeze_reference_packs.py", "task6p_field_v02_audit.py",
                  "task6p_train_reference.py", "task6p_evaluate_chain.py", "task6p_report.py")
TASK6P_MODULES = (REPO_ROOT / "buildreasonseg_mvp" / "geometric_relation_field_v02.py",
                  REPO_ROOT / "buildreasonseg_mvp" / "task6p_reference_head.py")
ALLOWED_PROGRAMS = {
    "largest_to_left_of", "largest_to_right_of", "largest_to_above", "largest_to_below",
    "smallest_to_left_of", "smallest_to_right_of", "smallest_to_above", "smallest_to_below",
}


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


def _git_changed(prefix: str) -> str:
    return subprocess.run(
        ["git", "diff", "--name-only", BASE_COMMIT, "--", prefix],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()


# ---------------------------------------------------------------- 1-3 frozen evidence


def test_task6n_artifacts_unchanged():
    changed = _git_changed("evaluation/task6n_")
    assert changed == "", f"Task 6N artifacts changed: {changed}"
    assert _artifact("task6n_verdict.json")["verdict"] == "GEOMETRIC_RELATION_FIELD_FEASIBLE"


def test_task6o_artifacts_unchanged():
    changed = _git_changed("evaluation/task6o_")
    assert changed == "", f"Task 6O artifacts changed: {changed}"
    artifact = _artifact("task6o_mini_val.json")
    assert artifact["variants"]["B3"]["overall"]["miou"] == pytest.approx(0.4299680351479113, abs=1e-12)
    assert _artifact("task6o_verdict.json")["verdict"] == "FIELD_GUIDED_VISUAL_SEGMENTATION_SUPPORTED"


def test_v01_field_file_unchanged():
    changed = _git_changed("buildreasonseg_mvp/geometric_relation_field.py")
    assert changed == "", "Task 6P must not modify GeometricRelationField v0.1"
    audit = _artifact("task6p_field_v02_audit.json")
    if audit is not None:
        assert audit["v01_file_sha256"] == _sha256(
            REPO_ROOT / "buildreasonseg_mvp" / "geometric_relation_field.py"
        )
        assert audit["v01_differentiable_wrt_reference_mask"] is False


# ---------------------------------------------------------------- 4-7 v0.2 field


def test_v02_binary_numerical_equivalence():
    from buildreasonseg_mvp.geometric_relation_field import geometric_relation_field as v01
    from buildreasonseg_mvp.geometric_relation_field_v02 import geometric_relation_field_v02 as v02

    torch.manual_seed(0)
    for relation in ("left_of", "right_of", "above", "below"):
        mask = torch.zeros(512, 512)
        mask[100:180, 200:290] = 1.0
        reference = v01(mask, relation, (64, 64))
        candidate = v02(mask, relation, (64, 64))
        assert reference.shape == candidate.shape
        assert float((reference - candidate).abs().max()) <= 1e-6
        assert float((reference - candidate).abs().mean()) <= 1e-7
    audit = _artifact("task6p_field_v02_audit.json")
    if audit is not None:
        assert audit["verdict"] == "FIELD_V02_VALID"
        assert audit["equivalence"]["max_abs_error"] <= 1e-6
        assert audit["equivalence"]["mean_abs_error"] <= 1e-7
        assert audit["equivalence"]["masks"] == 64
        assert audit["equivalence"]["per_direction"] == {"left_of": 16, "right_of": 16,
                                                         "above": 16, "below": 16}


def test_v02_finite_nonzero_gradient_to_soft_mask():
    from buildreasonseg_mvp.geometric_relation_field_v02 import geometric_relation_field_v02

    torch.manual_seed(1)
    height = width = 32
    rows = (torch.arange(height, dtype=torch.float32) + 0.5) / height
    columns = (torch.arange(width, dtype=torch.float32) + 0.5) / width
    yy, xx = torch.meshgrid(rows, columns, indexing="ij")
    blob = torch.exp(-(((xx - 0.4) ** 2 + (yy - 0.6) ** 2) / (2 * 0.1 ** 2)))
    mask = (0.8 * blob + 0.1).clone().requires_grad_(True)
    field = geometric_relation_field_v02(mask, "right_of", (16, 16))
    scalar = (field * (1.0 + xx[:16, :16] + 2 * yy[:16, :16]).view(1, 1, 16, 16)).sum()
    scalar.backward()
    assert mask.grad is not None
    assert torch.isfinite(mask.grad).all()
    assert float(mask.grad.abs().sum()) > 1e-8
    audit = _artifact("task6p_field_v02_audit.json")
    if audit is not None:
        assert audit["gradient_check"]["passed"] is True
        assert audit["gradient_check"]["masks"] >= 8
        assert audit["gradient_check"]["all_four_directions"] is True
        for row in audit["gradient_check"]["rows"]:
            assert row["gradient_exists"] and row["gradient_finite"]
            assert row["gradient_l1"] > 1e-8
            assert row["mask_is_non_binary"] is True


def _attribute_calls(source: str, attribute: str, function: str | None = None) -> list:
    """All real `x.<attribute>(...)` calls, optionally restricted to one function (AST-based)."""

    import ast

    tree = ast.parse(source)
    scope = tree
    if function is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function:
                scope = node
                break
        else:  # pragma: no cover - the function must exist
            raise AssertionError(f"{function} not found")
    return [
        node for node in ast.walk(scope)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == attribute
    ]


def test_v02_no_detach_in_field_forward_path():
    source = (REPO_ROOT / "buildreasonseg_mvp" / "geometric_relation_field_v02.py").read_text(
        encoding="utf-8"
    )
    # the module docstring must *document* v0.1's detach; no real detach call may exist anywhere
    assert ".detach()" in source, "the v0.2 docstring must record the v0.1 limitation"
    assert not _attribute_calls(source, "detach"), "the v0.2 code must never call .detach()"
    assert not _attribute_calls(source, "numpy")
    for function in ("geometric_relation_field_v02", "soft_centroid", "as_batch_mask"):
        assert not _attribute_calls(source, "detach", function), function
        assert not _attribute_calls(source, "detach_", function), function
    # no no_grad context inside the module either
    assert "no_grad" not in _code_only(
        REPO_ROOT / "buildreasonseg_mvp" / "geometric_relation_field_v02.py"
    )


def test_v02_no_python_float_centroid_in_field_forward_path():
    source = (REPO_ROOT / "buildreasonseg_mvp" / "geometric_relation_field_v02.py").read_text(
        encoding="utf-8"
    )
    import ast

    for function in ("soft_centroid", "geometric_relation_field_v02"):
        calls = _attribute_calls(source, "item", function)
        assert not calls, f"{function} must not call .item()"
        tree = ast.parse(source)
        target = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef) and node.name == function)
        builtin_float = [
            node for node in ast.walk(target)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "float"
        ]
        assert not builtin_float, f"{function} must not convert the centroid to a Python float"
    # the v0.1 behaviour being replaced is documented in the v0.2 module docstring
    assert "Python `float`" in source


# ---------------------------------------------------------------- 8-9 scope


def test_no_test_split_access():
    for name in TASK6P_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        for marker in ('"test.jsonl"', "'test.jsonl'", 'split="test"', "split('test')",
                       "task6p_test", "test_fixed120"):
            assert marker not in text, f"{name} must not touch the test split ({marker})"
    for name in ("task6p_field_v02_audit.json", "task6p_b3_reproduction.json",
                 "task6p_reference_pack_manifest.json", "task6p_reference_overfit20.json",
                 "task6p_reference_val.json", "task6p_predicted_reference_target_val.json",
                 "task6p_predicted_reference_paired_val.json",
                 "task6p_field_propagation_diagnostics.json", "task6p_verdict.json"):
        payload = _artifact(name)
        if payload is not None:
            assert payload.get("test_split_used", False) is False, name


def test_only_eight_directional_programs():
    from buildreasonseg_mvp.geometric_relation_field import DIRECTIONAL_PROGRAMS

    assert set(DIRECTIONAL_PROGRAMS) == ALLOWED_PROGRAMS
    manifest = _artifact("task6p_reference_pack_manifest.json")
    if manifest is not None:
        assert set(manifest["scope_programs"]) == ALLOWED_PROGRAMS
    for name in TASK6P_SOURCES:
        code = _code_only(SCRIPTS / name)
        assert "to_nearest" not in code, name
        assert "to_above_to_" not in code, name


# ---------------------------------------------------------------- 10-14 reference packs and head


def test_reference_packs_deduplicated_by_exact_key():
    manifest = _artifact("task6p_reference_pack_manifest.json")
    if manifest is None:
        pytest.skip("reference packs not frozen yet")
    assert manifest["unique_key"] == "(split, tile_id, reference_source_feature_id, reference_family)"
    assert manifest["target_included_in_reference_packs"] is False
    assert manifest["relation_id_stored_in_reference_packs"] is False
    for pack_name, expected_key in (("ref_train_unique", "ref_train_unique"),
                                    ("ref_val_unique", "ref_val_unique")):
        path = Path(manifest["packs"][pack_name]["path"])
        if not path.is_file():
            pytest.skip("packs live under gitignored artifacts/")
        payload = json.loads(path.read_text(encoding="utf-8"))
        keys = [(record["split"], record["tile_id"], record["reference_source_feature_id"],
                 record["reference_family"]) for record in payload["records"]]
        assert len(keys) == len(set(keys)), f"{pack_name} contains duplicate reference keys"
        assert len(keys) == manifest["packs"][pack_name]["count"]
        assert all(record["reference_source"] == "oracle_native_gt" for record in payload["records"])
        assert expected_key in manifest["packs"]


def test_target_source_id_never_reference_head_input():
    for name in ("reference_overfit20", "ref_train_unique", "ref_val_unique"):
        path = REF_ROOT / f"{name}.json"
        if not path.is_file():
            pytest.skip("packs live under gitignored artifacts/")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload["records"]:
            assert "target_source_feature_id" not in record, f"{name} must not carry target identity"
            assert "target_instance_id" not in record
    head = _code_only(REPO_ROOT / "buildreasonseg_mvp" / "task6p_reference_head.py")
    forward = head.split("def forward(")[1].split("def reference_logits_512")[0]
    assert "target" not in forward


def test_relation_id_never_reference_head_input():
    head = _code_only(REPO_ROOT / "buildreasonseg_mvp" / "task6p_reference_head.py")
    forward = head.split("def forward(")[1].split("def reference_logits_512")[0]
    for marker in ("relation", "program", "direction"):
        assert marker not in forward, f"the reference head must not receive {marker}"
    parameters = _artifact("task6p_reference_val.json")
    if parameters is not None:
        assert "target_relation_id" in parameters["parameters"]["not_inputs"]


def test_reference_head_input_is_visual_and_family_only():
    from buildreasonseg_mvp.task6p_reference_head import FAMILY_TO_INDEX, ReferenceMaskHead

    model = ReferenceMaskHead()
    assert model.fusion_channels == 144
    visual = torch.randn(2, 256, 64, 64)
    family = torch.as_tensor([0, 1], dtype=torch.long)
    logits = model(visual, family)
    assert logits.shape == (2, 1, 64, 64)
    assert model.project[0].in_channels == 256 and model.project[0].out_channels == 128
    assert model.trunk[0].in_channels == 144
    assert list(FAMILY_TO_INDEX) == ["largest", "smallest"]
    report = model.parameter_report()
    assert report["inputs"] == ["frozen_visual_feature_256x64x64", "reference_family_id"]


def test_family_vocab_exactly_largest_smallest():
    from buildreasonseg_mvp.task6p_reference_head import (
        FAMILIES, FAMILY_TO_INDEX, family_of_program,
    )

    assert FAMILIES == ("largest", "smallest")
    assert set(FAMILY_TO_INDEX) == {"largest", "smallest"}
    assert family_of_program("largest_to_left_of") == "largest"
    assert family_of_program("smallest_to_below") == "smallest"
    with pytest.raises(ValueError):
        family_of_program("nearest")
    packs = _artifact("task6p_reference_pack_manifest.json")
    if packs is not None:
        for counts in packs["family_counts"].values():
            assert set(counts) <= {"largest", "smallest"}


# ---------------------------------------------------------------- 15-18 frozen visual and B3


def test_frozen_sam2_unchanged():
    from buildreasonseg_mvp.task6n_relation_decoder import (
        SAM2_CHECKPOINT, SAM2_CONFIG_NAME, sam2_checkpoint_sha256,
    )

    assert SAM2_CHECKPOINT.name == "sam2.1_hiera_base_plus.pt"
    assert SAM2_CONFIG_NAME == "configs/sam2.1/sam2.1_hiera_b+.yaml"
    if SAM2_CHECKPOINT.is_file():
        assert sam2_checkpoint_sha256().startswith("a2345aede8715ab1")
    for name in TASK6P_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("ultralytics", "yolo26", "resnet", "swin", "YOLO("):
            assert marker not in code, f"{name} must not change the visual backbone ({marker})"


def test_b3_checkpoint_hash_verified():
    reproduction = _artifact("task6p_b3_reproduction.json")
    if reproduction is None:
        pytest.skip("B3 reproduction not run yet")
    assert reproduction["checkpoint"]["matches"] is True
    assert reproduction["checkpoint"]["retrained"] is False
    task6o = _artifact("task6o_mini_val.json")
    assert reproduction["checkpoint"]["expected_sha256"] == \
        task6o["variants"]["B3"]["training"]["checkpoint"]["sha256"]
    path = Path(reproduction["checkpoint"]["path"])
    if path.is_file():
        assert _sha256(path) == reproduction["checkpoint"]["expected_sha256"]


def test_b3_oracle_reproduction_tolerance():
    reproduction = _artifact("task6p_b3_reproduction.json")
    if reproduction is None:
        pytest.skip("B3 reproduction not run yet")
    assert reproduction["verdict"] == "B3_REPRODUCED"
    assert reproduction["tolerance"] == 1e-6
    for name, entry in reproduction["comparisons"].items():
        assert entry["within_tolerance"], name
        assert entry["abs_delta"] <= 1e-6, name
    assert reproduction["field_used"].startswith("geometric_relation_field_v02")


def test_b3_not_retrained():
    chain = _artifact("task6p_predicted_reference_target_val.json")
    if chain is None:
        pytest.skip("chain evaluation not run yet")
    assert chain["b3"]["retrained"] is False
    assert chain["reference_head"]["joint_training_with_b3"] is False
    assert chain["reference_head"]["frozen"] is True
    for name in TASK6P_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("B3\").train", "b3.train(", "optimizer.step()" if "train_reference" not in name
                       else "B3"):
            if marker == "B3":
                continue
            assert marker not in code, f"{name} must not train B3 ({marker})"


# ---------------------------------------------------------------- 19-23 chain semantics


def test_predicted_chain_uses_v02():
    code = (SCRIPTS / "task6p_evaluate_chain.py").read_text(encoding="utf-8")
    assert "geometric_relation_field_v02" in code
    assert "geometric_relation_field_v02(soft_reference" in code
    for name in TASK6P_SOURCES:
        assert "geometric_relation_field_v01" not in (SCRIPTS / name).read_text(encoding="utf-8")


def test_predicted_chain_does_not_use_oracle_reference_mask():
    code = (SCRIPTS / "task6p_evaluate_chain.py").read_text(encoding="utf-8")
    predicted = code.split("def predicted_field_for(")[1].split("def run_chain")[0]
    assert "reference_source_feature_id" not in predicted
    assert "masks.mask" not in predicted
    assert "soft_reference" in predicted
    target = _artifact("task6p_predicted_reference_target_val.json")
    if target is not None:
        assert target["reference_source"] == "predicted_reference_mask"
        assert "frozen_reference_mask_head(largest/smallest)" in target["chain"]


def test_gt_reference_only_for_reference_evaluation():
    code = _code_only(SCRIPTS / "task6p_evaluate_chain.py")
    # the GT reference mask appears only in the diagnostics comparison and in the GT centroid
    predicted = code.split("def predicted_field_for(")[1].split("def run_chain")[0]
    assert "reference_source_feature_id" not in predicted
    training = _code_only(SCRIPTS / "task6p_train_reference.py")
    assert "reference_source_feature_id" in training  # the reference label is the GT reference
    assert "target_source_feature_id" not in training


def test_gt_target_only_for_target_scoring():
    code = _code_only(SCRIPTS / "task6p_evaluate_chain.py")
    assert "target_source_feature_id" in code  # used to fetch the scoring GT
    training = _code_only(SCRIPTS / "task6p_train_reference.py")
    assert "target_source_feature_id" not in training
    for name in TASK6P_SOURCES:
        body = _code_only(SCRIPTS / name)
        for marker in ("loss(upsampled, target_mask_from_prediction", "target = target_field"):
            assert marker not in body, name


def test_same_predicted_reference_reused_for_pairs():
    paired = _artifact("task6p_predicted_reference_paired_val.json")
    if paired is None:
        pytest.skip("paired evaluation not run yet")
    assert paired["same_predicted_reference_reused_for_pairs"] is True
    for row in paired["rows"]:
        assert row["same_predicted_reference_reused"] is True
        assert row["passes"] == (row["a"]["prefers_own"] and row["b"]["prefers_own"])
    assert paired["pairs"] == 20


# ---------------------------------------------------------------- 24-31 no scope creep


def test_no_ref_token():
    for path in [SCRIPTS / name for name in TASK6P_SOURCES] + list(TASK6P_MODULES):
        assert "[REF]" not in path.read_text(encoding="utf-8"), path.name


def test_no_grcl_or_scl():
    for path in [SCRIPTS / name for name in TASK6P_SOURCES] + list(TASK6P_MODULES):
        code = _code_only(path)
        for marker in ("GRCL", "SCL", "counterfactual_loss", "relation_loss", "focal_loss"):
            assert marker not in code, f"{path.name} must not add {marker}"


def test_no_nearest_or_l3():
    for path in [SCRIPTS / name for name in TASK6P_SOURCES] + list(TASK6P_MODULES):
        code = _code_only(path)
        assert "to_nearest" not in code, path.name
        assert "to_above_to_" not in code, path.name
    from buildreasonseg_mvp.geometric_relation_field_v02 import RELATIONS

    assert set(RELATIONS) == {"left_of", "right_of", "above", "below"}


def test_no_graph_transformer():
    for path in [SCRIPTS / name for name in TASK6P_SOURCES] + list(TASK6P_MODULES):
        code = _code_only(path)
        for marker in ("MultiheadAttention", "TransformerEncoder", "GraphConv", "MessagePassing",
                       "nn.Transformer"):
            assert marker not in code, f"{path.name} must not add {marker}"


def test_no_proposal_training():
    for path in [SCRIPTS / name for name in TASK6P_SOURCES] + list(TASK6P_MODULES):
        code = _code_only(path)
        for marker in ("ultralytics", "yolo26", "task6m_train", "YOLO("):
            assert marker not in code, f"{path.name} must not train a proposal model ({marker})"


def test_no_4b():
    for path in [SCRIPTS / name for name in TASK6P_SOURCES] + list(TASK6P_MODULES):
        text = path.read_text(encoding="utf-8")
        assert "Qwen3-VL-4B" not in text, path.name
        assert "4B" not in text.replace("4B)", ""), path.name


def test_no_download_install_or_gui():
    for path in [SCRIPTS / name for name in TASK6P_SOURCES] + list(TASK6P_MODULES):
        code = _code_only(path)
        for marker in ("urlretrieve", "requests.get", "Invoke-WebRequest", "hf_hub_download",
                       "snapshot_download", "pip install", "tkinter", "PyQt", "gradio",
                       "streamlit", "cv2.imshow", "flask", "fastapi"):
            assert marker not in code, f"{path.name} must not use {marker}"


def test_previous_suite_preserved():
    for name in ("test_task6o_field_causal_decomposition.py", "test_task6n_oracle_relation_field.py",
                 "test_task6m1_convergence_demo_fix.py", "test_task6m_native_vector_proposal.py"):
        assert (REPO_ROOT / "tests" / name).is_file(), name
    removed = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=D", BASE_COMMIT, "--", "tests/"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert removed == "", f"no test file may be removed: {removed}"
