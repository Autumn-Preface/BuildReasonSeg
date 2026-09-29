"""Task 6R tests (section 23): 37 checks on GRCL v0.1, the R1/R2 variants and the transfer diagnostic."""

from __future__ import annotations

import ast
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
GRCL_MODULE = REPO_ROOT / "buildreasonseg_mvp" / "grcl_directional.py"
BASE_COMMIT = "7c19bec577282029d0eb0eac7187940262c1a63a"
TASK6R_SOURCES = ("task6r_grcl_audit.py", "task6r_train.py", "task6r_evaluate.py",
                  "task6r_proposal_transfer.py", "task6r_report.py")


def _artifact(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_changed(prefix: str) -> str:
    return subprocess.run(
        ["git", "diff", "--name-only", BASE_COMMIT, "--", prefix],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()


def _function_node(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path.name}")


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


def _mask(y0, y1, x0, x1, size=64) -> torch.Tensor:
    mask = torch.zeros(size, size)
    mask[y0:y1, x0:x1] = 1.0
    return mask


# ---------------------------------------------------------------- 1-3 frozen evidence


def test_task6q_artifacts_unchanged():
    changed = _git_changed("evaluation/task6q_")
    assert changed == "", f"Task 6Q artifacts changed: {changed}"
    assert _artifact("task6q_verdict.json")["verdict"] == "REFERENCE_PROPOSAL_COVERAGE_INSUFFICIENT"


def test_task6o_b3_hash_exact():
    baseline = _artifact("task6r_b3_relation_baseline.json")
    task6o = _artifact("task6o_mini_val.json")
    expected = task6o["variants"]["B3"]["training"]["checkpoint"]["sha256"]
    if baseline is not None:
        assert baseline["checkpoint"]["matches"] is True
        assert baseline["checkpoint"]["actual_sha256"] == expected
        assert baseline["checkpoint"]["retrained"] is False
    path = Path(task6o["variants"]["B3"]["training"]["checkpoint"]["path"])
    if path.is_file():
        assert _sha256(path) == expected


def test_geometric_relation_field_v02_unchanged():
    changed = _git_changed("buildreasonseg_mvp/geometric_relation_field_v02.py")
    assert changed == "", "Task 6R must not modify GeometricRelationField v0.2"
    assert _git_changed("buildreasonseg_mvp/geometric_relation_field.py") == ""


# ---------------------------------------------------------------- 4-16 GRCL definition


def test_alpha_exactly_1_2():
    from buildreasonseg_mvp.grcl_directional import ALPHA

    assert ALPHA == 1.2
    audit = _artifact("task6r_grcl_audit.json")
    if audit is not None:
        assert audit["constants"]["alpha"] == 1.2


def test_tau_exactly_0_04():
    from buildreasonseg_mvp.grcl_directional import TAU

    assert TAU == 0.04
    audit = _artifact("task6r_grcl_audit.json")
    if audit is not None:
        assert audit["constants"]["tau"] == 0.04


def test_lambda_grcl_exactly_0_5():
    from buildreasonseg_mvp.grcl_directional import LAMBDA_GRCL

    assert LAMBDA_GRCL == 0.5
    audit = _artifact("task6r_grcl_audit.json")
    if audit is not None:
        assert audit["constants"]["lambda_grcl"] == 0.5
    mini = _artifact("task6r_training.json")
    if mini is not None:
        assert mini["config"]["lambda_grcl"] == 0.5
        assert mini["config"]["loss"] == "BCE + Dice + 0.5 * GRCL"
    trainer = _code_only(SCRIPTS / "task6r_train.py")
    assert "LAMBDA_GRCL * report.loss" in trainer


def test_target_centroid_is_differentiable():
    from buildreasonseg_mvp.grcl_directional import soft_centroids

    logits = torch.randn(1, 1, 32, 32, requires_grad=True)
    reference = _mask(4, 12, 4, 12, size=32)
    centroids = soft_centroids(logits, reference)
    assert centroids["cx_t"].requires_grad and centroids["cy_t"].requires_grad
    (centroids["cx_t"].sum() + centroids["cy_t"].sum()).backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0.0


def test_target_logits_receive_finite_nonzero_grcl_gradient():
    from buildreasonseg_mvp.grcl_directional import grcl_directional

    torch.manual_seed(0)
    logits = (torch.randn(2, 1, 32, 32) * 2).requires_grad_(True)
    reference = torch.stack([_mask(4, 12, 4, 12, size=32), _mask(20, 28, 20, 28, size=32)])
    report = grcl_directional(logits, reference, ["right_of", "left_of"])
    assert bool(torch.isfinite(report.loss))
    report.loss.backward()
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())
    assert float(logits.grad.abs().sum()) > 1e-8
    audit = _artifact("task6r_grcl_audit.json")
    if audit is not None:
        assert audit["verdict"] == "GRCL_VALID"
        assert audit["directional_sanity_passed"] is True
        for row in audit["gradient_rows"]:
            assert row["gradient_l1"] > 1e-8 and row["gradient_finite"] and row["hinge_active"]


def _is_structural(node) -> bool:
    """True for shape/dimension expressions (`.shape`, `.dim()`, `.size()`) and constants."""

    if isinstance(node, ast.Call):
        return isinstance(node.func, ast.Attribute) and node.func.attr in {"dim", "size", "ndim"}
    if isinstance(node, ast.Attribute):
        return node.attr in {"shape", "ndim"}
    if isinstance(node, ast.Subscript):
        return _is_structural(node.value)
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp):
        return _is_structural(node.operand)
    if isinstance(node, ast.BinOp):
        return _is_structural(node.left) and _is_structural(node.right)
    return False


def test_no_thresholding_inside_grcl():
    """Only structural (shape/dim) and categorical (relation-id) comparisons are allowed.

    No probability, mask, logit or displacement value may ever be thresholded inside the loss path.
    """

    categorical = {"indices", "left", "right", "above", "below", "relation"}
    for function in ("grcl_directional", "soft_centroids", "signed_orthogonal"):
        node = _function_node(GRCL_MODULE, function)
        for child in ast.walk(node):
            if not isinstance(child, ast.Compare):
                continue
            for operand in [child.left, *child.comparators]:
                if _is_structural(operand):
                    continue
                names = {sub.id for sub in ast.walk(operand) if isinstance(sub, ast.Name)}
                assert names and names <= categorical, (
                    f"{function} must not threshold or compare tensor values (saw {names})"
                )
    # and the loss itself is a pure hinge combination
    source = _code_only(GRCL_MODULE)
    assert source.count("torch.relu(") == 2, "exactly the margin and axis hinge terms"


def test_hard_relation_metric_threshold_is_evaluation_only():
    import inspect

    from buildreasonseg_mvp.grcl_directional import hard_relation_metrics

    signature = inspect.signature(hard_relation_metrics)
    assert signature.parameters["threshold"].default == 0.5
    assert signature.parameters["alpha"].default == 1.2
    assert signature.parameters["tau"].default == 0.04
    # the metric lives in the evaluation path, never in the training loss
    for function in ("grcl_directional", "soft_centroids"):
        node = _function_node(GRCL_MODULE, function)
        names = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
        assert "hard_relation_metrics" not in names, function
    evaluator = _code_only(SCRIPTS / "task6r_evaluate.py")
    assert "hard_relation_metrics(" in evaluator


def test_left_sign_exact():
    from buildreasonseg_mvp.grcl_directional import signed_orthogonal

    cx_t = torch.tensor([0.2])
    cy_t = torch.tensor([0.5])
    cx_r = torch.tensor([0.7])
    cy_r = torch.tensor([0.4])
    signed, orth = signed_orthogonal(cx_t, cy_t, cx_r, cy_r, torch.tensor([0]))
    assert float(signed) == pytest.approx(0.7 - 0.2)
    assert float(orth) == pytest.approx(abs(0.5 - 0.4))


def test_right_sign_exact():
    from buildreasonseg_mvp.grcl_directional import signed_orthogonal

    signed, orth = signed_orthogonal(torch.tensor([0.8]), torch.tensor([0.3]), torch.tensor([0.2]),
                                     torch.tensor([0.6]), torch.tensor([1]))
    assert float(signed) == pytest.approx(0.8 - 0.2)
    assert float(orth) == pytest.approx(abs(0.3 - 0.6))


def test_above_sign_exact():
    from buildreasonseg_mvp.grcl_directional import signed_orthogonal

    signed, orth = signed_orthogonal(torch.tensor([0.4]), torch.tensor([0.1]), torch.tensor([0.6]),
                                     torch.tensor([0.7]), torch.tensor([2]))
    assert float(signed) == pytest.approx(0.7 - 0.1)
    assert float(orth) == pytest.approx(abs(0.4 - 0.6))


def test_below_sign_exact():
    from buildreasonseg_mvp.grcl_directional import signed_orthogonal

    signed, orth = signed_orthogonal(torch.tensor([0.9]), torch.tensor([0.8]), torch.tensor([0.2]),
                                     torch.tensor([0.3]), torch.tensor([3]))
    assert float(signed) == pytest.approx(0.8 - 0.3)
    assert float(orth) == pytest.approx(abs(0.9 - 0.2))


def test_axis_term_exact():
    from buildreasonseg_mvp.grcl_directional import ALPHA, grcl_directional

    logits = torch.full((1, 1, 32, 32), -10.0)
    logits[0, 0, 12:20, 12:20] = 10.0
    reference = _mask(2, 6, 2, 6, size=32)
    report = grcl_directional(logits, reference, "right_of")
    expected = torch.relu(ALPHA * report.orth - report.signed)
    assert torch.allclose(report.axis, expected, atol=1e-6)
    assert float(report.axis.mean()) > 0.0


def test_margin_term_exact():
    from buildreasonseg_mvp.grcl_directional import TAU, grcl_directional

    logits = torch.full((1, 1, 32, 32), -10.0)
    logits[0, 0, 12:20, 12:20] = 10.0
    reference = _mask(2, 6, 2, 6, size=32)
    report = grcl_directional(logits, reference, "right_of")
    expected = torch.relu(TAU - report.signed)
    assert torch.allclose(report.margin, expected, atol=1e-6)
    # L_GRCL is exactly the mean of the two terms
    assert float(report.loss) == pytest.approx(float((report.margin + report.axis).mean()), abs=1e-9)


# ---------------------------------------------------------------- 17-23 variant contracts


def test_r1_architecture_equals_b3():
    from buildreasonseg_mvp.task6n_relation_decoder import VARIANTS

    training = _artifact("task6r_training.json")
    assert VARIANTS == ("B0", "B1", "B2")
    if training is not None:
        assert training["variants"]["R1"]["decoder"] == "B3"
        assert training["variants"]["R1"]["parameters"]["first_conv_in_channels"] == 145
        assert training["variants"]["R1"]["parameters"]["total_parameters"] == 274625
    trainer = _code_only(SCRIPTS / "task6r_train.py")
    assert '"R1": {"decoder": "B3", "uses_field": True' in trainer


def test_r1_has_field():
    from buildreasonseg_mvp.task6n_relation_decoder import VARIANT_USES_FIELD

    assert VARIANT_USES_FIELD["B3"] is True
    training = _artifact("task6r_training.json")
    if training is not None:
        assert training["variants"]["R1"]["uses_field"] is True


def test_r1_no_direct_reference_channel():
    from buildreasonseg_mvp.task6n_relation_decoder import VARIANT_USES_REFERENCE

    assert VARIANT_USES_REFERENCE["B3"] is False
    trainer = _code_only(SCRIPTS / "task6r_train.py")
    assert "model(visual, relation_index, None, field)" in trainer
    mini = _artifact("task6r_mini_val.json")
    if mini is not None:
        assert mini["variants"]["R1"]["decoder"] == "B3"
        assert mini["variants"]["R1"]["uses_field"] is True


def test_r2_architecture_equals_b0():
    from buildreasonseg_mvp.task6n_relation_decoder import DecoderConfig, RelationMaskDecoder

    model = RelationMaskDecoder("B0", DecoderConfig())
    report = model.parameter_report()
    assert report["first_conv_in_channels"] == 144
    assert report["total_parameters"] == 273473
    training = _artifact("task6r_training.json")
    if training is not None:
        assert training["variants"]["R2"]["decoder"] == "B0"
        assert training["variants"]["R2"]["parameters"]["total_parameters"] == 273473
        assert training["variants"]["R2"]["uses_field"] is False


def test_r2_has_no_field_input():
    from buildreasonseg_mvp.task6n_relation_decoder import VARIANT_USES_FIELD

    assert VARIANT_USES_FIELD["B0"] is False
    trainer = _code_only(SCRIPTS / "task6r_train.py")
    assert "return model(visual, relation_index, None, None)" in trainer


def test_r2_has_no_reference_mask_model_input():
    from buildreasonseg_mvp.task6n_relation_decoder import VARIANT_USES_REFERENCE

    assert VARIANT_USES_REFERENCE["B0"] is False
    # the R2 path passes no reference tensor to the decoder
    trainer = _code_only(SCRIPTS / "task6r_train.py")
    r2_branch = trainer.split("if VARIANTS[variant][\"uses_field\"]:")[1].split("def build_batch")[0]
    assert "reference" not in r2_branch


def test_oracle_reference_used_by_r2_only_in_loss_and_evaluation():
    trainer = (SCRIPTS / "task6r_train.py").read_text(encoding="utf-8")
    # the reference reaches the loss and the relation metric, and never the R2 model call
    assert "batch_loss(logits, target, reference, chunk" in trainer
    assert "hard_relation_metrics(probabilities, reference" in trainer
    assert "model(visual, relation_index, None, None)" in trainer
    mini = _artifact("task6r_mini_val.json")
    if mini is not None:
        assert mini["variants"]["R2"]["uses_field"] is False


# ---------------------------------------------------------------- 24-29 protocol


def test_exact_frozen_packs_reused():
    manifest = _artifact("task6n_pack_manifest.json")
    training = _artifact("task6r_training.json")
    overfit = _artifact("task6r_overfit20.json")
    mini = _artifact("task6r_mini_val.json")
    if not all([manifest, training, overfit, mini]):
        pytest.skip("Task 6R artifacts not all generated yet")
    for name, entry in (("mini_train_1000", training["packs"]["train"]),
                        ("mini_val_240", training["packs"]["val"]),
                        ("overfit20", overfit["pack"]),
                        ("mini_val_240", mini["pack"])):
        assert entry["sha256"] == manifest["packs"][name]["sha256"], name
    paired = _artifact("task6r_paired_val.json")
    if paired is not None:
        assert paired["pack"]["sha256"] == manifest["packs"]["paired_val_20"]["sha256"]


def test_same_training_settings_r1_and_r2():
    training = _artifact("task6r_training.json")
    overfit = _artifact("task6r_overfit20.json")
    if training is None or overfit is None:
        pytest.skip("Task 6R training artifacts not generated yet")
    config = training["config"]
    assert config["variants"] == ["R1", "R2"]
    assert config["optimizer"] == "AdamW"
    assert config["lr"] == 3.0e-4 and config["weight_decay"] == 1.0e-4
    assert config["batch_size"] == 8 and config["max_epochs"] == 25
    assert config["early_stopping_patience"] == 5
    assert config["seed"] == 20260929 and config["scheduler"] == "none"
    assert config["augmentation"] == "none"
    assert overfit["config"]["lr"] == 1.0e-3 and overfit["config"]["max_steps"] == 1200
    assert overfit["config"]["batch_size"] == 4 and overfit["config"]["seed"] == 20260929
    for variant in ("R1", "R2"):
        assert training["variants"][variant]["epochs_run"] <= 25
    # a single shared config drives both variants
    trainer = _code_only(SCRIPTS / "task6r_train.py")
    assert trainer.count("optimizer = make_optimizer(model, config[\"lr\"], config[\"weight_decay\"])") == 1


def test_no_test_split():
    for name in TASK6R_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        for marker in ('"test.jsonl"', "'test.jsonl'", 'split="test"', "split('test')",
                       "task6r_test", "test_fixed120"):
            assert marker not in text, f"{name} must not touch the test split ({marker})"
    for name in ("task6r_grcl_audit.json", "task6r_b3_relation_baseline.json",
                 "task6r_overfit20.json", "task6r_training.json", "task6r_mini_val.json",
                 "task6r_paired_val.json", "task6r_proposal_reference_transfer.json",
                 "task6r_verdict.json"):
        payload = _artifact(name)
        if payload is not None:
            assert payload.get("test_split_used", False) is False, name


def test_gt_target_never_model_input():
    from buildreasonseg_mvp.task6n_relation_decoder import RelationMaskDecoder

    node = _function_node(REPO_ROOT / "buildreasonseg_mvp" / "task6n_relation_decoder.py", "forward")
    names = {child.arg for child in ast.walk(node) if isinstance(child, ast.arg)}
    assert "target" not in names
    trainer = _code_only(SCRIPTS / "task6r_train.py")
    assert "batch_loss(logits, target, reference, chunk" in trainer  # target is a label argument only


def test_proposal_transfer_uses_frozen_task6q_resolver_and_config():
    transfer = _artifact("task6r_proposal_reference_transfer.json")
    if transfer is None:
        pytest.skip("transfer diagnostic not run yet")
    assert transfer["reference_source"] == "frozen_task6q_proposal_resolver"
    config = transfer["proposal_config"]
    assert config["conf"] == 0.10 and config["imgsz"] == 640 and config["max_det"] == 100
    assert config["threshold_sweep"] is False and config["tta"] is False
    assert config["checkpoint_sha256_expected"] == transfer["proposal_checkpoint_sha256_expected"]
    code = (SCRIPTS / "task6r_proposal_transfer.py").read_text(encoding="utf-8")
    assert "load_resolver_cache" in code
    assert "run_frozen_proposals" not in code  # it must reuse the cache, not re-run the resolver


def test_proposal_transfer_does_not_tune():
    transfer = _artifact("task6r_proposal_reference_transfer.json")
    if transfer is None:
        pytest.skip("transfer diagnostic not run yet")
    assert transfer["diagnostic_only"] is True
    assert transfer["training_performed"] is False
    assert transfer["lambda_tuning"] is False
    code = _code_only(SCRIPTS / "task6r_proposal_transfer.py")
    for marker in ("backward()", "optimizer", "AdamW", "model.train()", "for lambda"):
        assert marker not in code, f"the diagnostic must not train or tune ({marker})"


# ---------------------------------------------------------------- 30-37 no scope creep


def test_no_yolo_retraining():
    # `model.train()` in task6r_train.py is the decoder's train/eval mode switch, not YOLO training
    for name in TASK6R_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("YOLO(", "ultralytics", "resume=True", "task6q_reference_audit",
                       "run_frozen_proposals", "PROPOSAL_TRAIN"):
            assert marker not in code, f"{name} must not retrain the proposal model ({marker})"
    transfer = _artifact("task6r_proposal_reference_transfer.json")
    if transfer is not None:
        assert transfer["proposal_config"]["conf"] == 0.10
        assert transfer["training_performed"] is False


def test_no_ref_token():
    for path in [SCRIPTS / name for name in TASK6R_SOURCES] + [GRCL_MODULE]:
        assert "[REF]" not in path.read_text(encoding="utf-8"), path.name


def test_no_nearest_or_l3():
    from buildreasonseg_mvp.grcl_directional import RELATIONS

    assert set(RELATIONS) == {"left_of", "right_of", "above", "below"}
    for path in [SCRIPTS / name for name in TASK6R_SOURCES] + [GRCL_MODULE]:
        code = _code_only(path)
        assert "to_nearest" not in code, path.name
        assert "to_above_to_" not in code, path.name


def test_no_mllm_hidden_state_fusion():
    from buildreasonseg_mvp.task6n_relation_decoder import DecoderConfig

    config = DecoderConfig()
    assert config.visual_channels == 256  # frozen SAM2 visual feature only
    for path in [SCRIPTS / name for name in TASK6R_SOURCES] + [GRCL_MODULE]:
        code = _code_only(path)
        for marker in ("hidden_state", "hidden_states", "qwen", "Qwen", "transformers", "MLLM"):
            assert marker not in code, f"{path.name} must not add MLLM hidden-state fusion ({marker})"


def test_no_graph_transformer():
    for path in [SCRIPTS / name for name in TASK6R_SOURCES] + [GRCL_MODULE]:
        code = _code_only(path)
        for marker in ("MultiheadAttention", "TransformerEncoder", "GraphConv", "MessagePassing",
                       "nn.Transformer"):
            assert marker not in code, f"{path.name} must not add {marker}"


def test_no_4b():
    for path in [SCRIPTS / name for name in TASK6R_SOURCES] + [GRCL_MODULE]:
        text = path.read_text(encoding="utf-8")
        assert "Qwen3-VL-4B" not in text, path.name
        assert "4B" not in text.replace("4B)", ""), path.name


def test_no_new_dataset_download_install_or_gui():
    for path in [SCRIPTS / name for name in TASK6R_SOURCES] + [GRCL_MODULE]:
        code = _code_only(path)
        for marker in ("urlretrieve", "requests.get", "Invoke-WebRequest", "hf_hub_download",
                       "snapshot_download", "pip install", "tkinter", "PyQt", "gradio",
                       "streamlit", "cv2.imshow", "flask", "fastapi"):
            assert marker not in code, f"{path.name} must not use {marker}"


def test_previous_suite_preserved():
    for name in ("test_task6q_frozen_proposal_reference_resolver.py",
                 "test_task6p_differentiable_field_predicted_reference.py",
                 "test_task6o_field_causal_decomposition.py",
                 "test_task6n_oracle_relation_field.py"):
        assert (REPO_ROOT / "tests" / name).is_file(), name
    removed = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=D", BASE_COMMIT, "--", "tests/"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert removed == "", f"no test file may be removed: {removed}"
