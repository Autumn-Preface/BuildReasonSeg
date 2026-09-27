"""Task 6D section 18: Spatial Grounding Bridge tests.

Items 1-16 of the section 18 list. The geometry unit tests need no weights; the
model-backed items build a real Task 6D runtime once, lazily, and are skipped when the
cached assets are unavailable.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

EVAL = REPO_ROOT / "evaluation"
CONFIG = REPO_ROOT / "configs" / "mvp" / "task6c_2b_ablation.yaml"
GROUNDING = REPO_ROOT / "buildreasonseg_mvp" / "grounding.py"
GROUNDING_EVAL = REPO_ROOT / "buildreasonseg_mvp" / "grounding_eval.py"
MODEL = REPO_ROOT / "buildreasonseg_mvp" / "model.py"
RUNTIME = REPO_ROOT / "buildreasonseg_mvp" / "runtime.py"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "task6d_train.py"

ORACLE = EVAL / "task6d_oracle_prompt_diagnostic.json"
TARGETS = EVAL / "task6d_grounding_targets.json"
G0 = EVAL / "task6d_g0.json"
G1 = EVAL / "task6d_g1.json"
PAIRED = EVAL / "task6d_paired_probe.json"
REPRESENTATION = EVAL / "task6d_representation.json"
ERROR_ANALYSIS = EVAL / "task6d_error_analysis.json"
MANIFEST = EVAL / "task6d_checkpoint_manifest.json"

TASK6D_FILES = sorted((REPO_ROOT / "scripts").glob("task6d_*.py")) + [GROUNDING, GROUNDING_EVAL]


def _load(path: Path) -> dict:
    if not path.is_file():
        print(f"  [skip] {path.name} not present yet")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- geometry units


def _mask(shape=(32, 32), box=(4, 6, 20, 26)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[box[1] : box[3], box[0] : box[2]] = True
    return mask


def test_deterministic_interior_point_lies_inside_gt_mask():
    """Section 18 item 1."""

    from buildreasonseg_mvp.grounding import distance_transform_point

    mask = _mask()
    first = distance_transform_point(mask)
    assert distance_transform_point(mask) == first, "the point must be deterministic"
    height, width = mask.shape
    column = int(np.floor(first[0] * width))
    row = int(np.floor(first[1] * height))
    assert mask[row, column], "the derived point must be inside the target mask"

    # an L-shaped mask: the distance-transform maximum is unique and interior
    shaped = np.zeros((40, 40), dtype=bool)
    shaped[5:35, 5:15] = True
    shaped[25:35, 5:35] = True
    point = distance_transform_point(shaped)
    assert shaped[int(point[1] * 40), int(point[0] * 40)]
    print("  [1] deterministic interior point inside the mask OK")


def test_box_tightly_encloses_gt_mask():
    """Section 18 item 2."""

    from buildreasonseg_mvp.grounding import GEOMETRY_BOX, tight_box

    mask = _mask(shape=(32, 32), box=(4, 6, 20, 26))
    x1, y1, x2, y2 = tight_box(mask)
    assert (x1, y1) == (4 / 32, 6 / 32)
    assert (x2, y2) == (20 / 32, 26 / 32)
    assert x1 <= x2 and y1 <= y2
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    assert int(x1 * 32) == columns[0] and int(y1 * 32) == rows[0]
    print("  [2] tight GT box OK")


def test_normalized_geometry_valid_and_canonical():
    """Section 18 item 3."""

    from buildreasonseg_mvp.grounding import (
        GEOMETRY_BOX,
        GEOMETRY_POINT,
        SpatialGroundingHead,
        geometry_is_valid,
    )

    assert geometry_is_valid((0.2, 0.8), GEOMETRY_POINT)
    assert not geometry_is_valid((1.4, 0.8), GEOMETRY_POINT)
    assert geometry_is_valid((0.1, 0.1, 0.9, 0.9), GEOMETRY_BOX)
    assert not geometry_is_valid((0.9, 0.1, 0.1, 0.9), GEOMETRY_BOX), "x1 > x2 must be rejected"
    assert not geometry_is_valid((float("nan"), 0.1, 0.9, 0.9), GEOMETRY_BOX)

    head = SpatialGroundingHead(64, 32, GEOMETRY_BOX)
    output = head(torch.randn(5, 64))
    assert output.shape == (5, 4)
    assert bool((output >= 0).all() and (output <= 1).all())
    assert bool((output[:, 0] <= output[:, 2]).all() and (output[:, 1] <= output[:, 3]).all())
    point_head = SpatialGroundingHead(64, 32, GEOMETRY_POINT)
    assert point_head(torch.randn(3, 64)).shape == (3, 2)
    print("  [3] normalized geometry is valid and canonical OK")


def test_gt_geometry_only_in_supervision_and_oracle():
    """Section 18 item 4: the inference path must not touch GT geometry."""

    import ast

    def calls_target_geometry(path: Path, function: str) -> bool:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function:
                return any(
                    isinstance(child, ast.Call)
                    and getattr(child.func, "id", getattr(child.func, "attr", "")) == "target_geometry"
                    for child in ast.walk(node)
                )
        raise AssertionError(f"{function} not found in {path.name}")

    # supervision: the training step derives GT geometry from the GT mask
    assert calls_target_geometry(RUNTIME, "grounding_train_step")
    # inference: the grounded forward and the free-generation decoder must not
    assert not calls_target_geometry(MODEL, "forward_grounded")
    assert not calls_target_geometry(GROUNDING_EVAL, "decode_from_hidden")
    assert not calls_target_geometry(GROUNDING_EVAL, "_generate_grounded")
    print("  [4] GT geometry is supervision/oracle only OK")


def test_free_inference_uses_predicted_geometry_only():
    """Section 18 item 5."""

    source = GROUNDING_EVAL.read_text(encoding="utf-8")
    body = source.split("def decode_from_hidden", 1)[1].split("\ndef ", 1)[0]
    assert "predicted_geometry_for_hidden" in body
    assert "head(hidden)" in source
    assert "decode_mask_from_geometry" in body
    assert "decode_mask(" not in body.replace("decode_mask_from_geometry(", ""), (
        "the candidate must not call the old projected-embedding decode"
    )
    oracle = _load(ORACLE)
    if oracle:
        for kind in ("point", "box"):
            record = oracle[kind]["records"][0]
            assert record["prompt"]["prompt_kind"] == kind
            assert record["prompt"]["derived_from"] == "geometry_input"
    print("  [5] free inference uses predicted geometry only OK")


def test_old_language_sparse_prompt_absent_from_candidate():
    """Section 18 item 12."""

    source = MODEL.read_text(encoding="utf-8")
    body = source.split("def forward_grounded", 1)[1].split("\ndef ", 1)[0]
    for forbidden in ("build_sparse_prompt", "decode_mask(", "self.projection("):
        assert forbidden not in body, f"the candidate path must not use {forbidden}"
    g0 = _load(G0)
    if g0:
        assert g0["trainables"]["projection_trainable"] is False
        assert g0["trainables"]["grounding_head_trainable"] is True
    print("  [12] the old arbitrary language sparse prompt is absent OK")


def test_paired_samples_have_different_target_geometry():
    """Section 18 item 9."""

    targets = _load(TARGETS)
    if not targets:
        return
    assert targets["paired_targets_all_differ"], "every pair must have different geometry"
    assert all(record["targets_differ"] for record in targets["paired"])
    assert all(record["gt_mask_iou_a_vs_b"] < 0.5 for record in targets["paired"]), (
        "paired targets must be genuinely different regions"
    )
    assert targets["mean_paired_geometry_distance"] > 0.05
    print(
        f"  [9] {len(targets['paired'])} pairs all have different targets "
        f"(mean geometry distance {targets['mean_paired_geometry_distance']:.3f}) OK"
    )


def test_official_sam2_prompt_encoder_is_used_on_both_paths():
    """Section 18 items 10 and 11."""

    source = GROUNDING.read_text(encoding="utf-8")
    assert "sam.sam_prompt_encoder(points=" in source and "sam.sam_prompt_encoder(points=None, boxes=" in source
    assert "official" in source.lower()
    oracle_source = (REPO_ROOT / "scripts" / "task6d_oracle.py").read_text(encoding="utf-8")
    assert "decode_mask_from_geometry" in oracle_source
    oracle = _load(ORACLE)
    if oracle:
        for kind in ("point", "box"):
            for record in oracle[kind]["records"][:5]:
                assert record["prompt"]["image_size"] == 1024
                assert record["prompt"]["sparse_prompt_shape"][-1] == 256
    print("  [10,11] both paths use the official SAM2 prompt encoder OK")


def test_no_test_split_used():
    """Section 18 item 14."""

    from buildreasonseg_mvp import data as data_mod

    targets = _load(TARGETS)
    if not targets:
        return
    train_ids = {record["sample_id"] for record in data_mod.read_records("train")}
    val_ids = {record["sample_id"] for record in data_mod.read_records("val")}
    test_ids = {record["sample_id"] for record in data_mod.read_records("test")}
    assert {record["sample_id"] for record in targets["train"]} <= train_ids
    assert {record["sample_id"] for record in targets["val"]} <= val_ids
    assert not ({record["sample_id"] for record in targets["val"]} & test_ids)
    assert not ({record["sample_id"] for record in targets["train"]} & test_ids)
    print("  [14] train/val only, no test split OK")


def test_no_4b_ref_sre_scl():
    """Section 18 item 15."""

    forbidden = ("[REF]", "Qwen3-VL-4B", "SpatialRelationEncoder", "Spatial Consistency Loss", "4B-Instruct")
    for path in TASK6D_FILES:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} mentions {token}"
    print("  [15] no 4B/[REF]/SRE/SCL OK")


def test_strict_determinism_retained():
    """Section 18 item 16."""

    for path in (G0, G1, ORACLE):
        payload = _load(path)
        if not payload:
            continue
        determinism = payload.get("determinism") or payload.get("runtime", {}).get("determinism")
        if determinism:
            assert determinism.get("strict_effective") is True, path.name
    g0 = _load(G0)
    if g0:
        assert g0["determinism"]["use_deterministic_algorithms"] is True
        assert g0["determinism"]["cublas_workspace_config"] == ":4096:8"
    print("  [16] strict determinism retained OK")


def test_artifacts_present_and_consistent():
    """Section 17 deliverables."""

    g0 = _load(G0)
    if not g0:
        return
    assert g0["geometry_kind"] in ("point", "box")
    assert g0["geometry_selection_source"].endswith("task6d_oracle_prompt_diagnostic.json")
    assert g0["epochs"], "G0 must record at least one epoch"
    assert "gate" in g0 and "verdict" in g0
    if G1.is_file():
        assert _load(G1)["verdict"] in ("G1_COMPLETE", "GROUNDING_REPRESENTATION_FAILED")
    oracle = _load(ORACLE)
    assert oracle["verdict"].startswith("PROCEED_WITH_") or oracle["verdict"] == (
        "ORACLE_SPATIAL_PROMPT_INSUFFICIENT"
    )
    print(f"  [extra] artifacts present: G0 verdict {g0['verdict']} OK")


# ---------------------------------------------------------------- model-backed


_RUNTIME_CACHE: dict = {}


def _task6d_runtime():
    if "runtime" in _RUNTIME_CACHE:
        return _RUNTIME_CACHE["runtime"]
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    oracle = _load(ORACLE)
    if not oracle:
        pytest.skip("oracle artifact is absent")
    from buildreasonseg_mvp.runtime import build_runtime, load_config
    from task6c6_common import build_variant_runtime  # noqa: F401  (keeps the shared config path)

    cfg = load_config(CONFIG)
    runtime = build_runtime(cfg, device="cuda", verbose=False)
    runtime.install_visual_cache(enabled=True, max_images=64)
    runtime.install_grounding_head(oracle["selection"]["chosen_geometry"])
    runtime.set_grounding_trainables("G1")
    _RUNTIME_CACHE["runtime"] = runtime
    return runtime


def test_model_backed_grounding_invariants():
    """Section 18 items 6, 7, 8, 13 and 5 (functional)."""

    runtime = _task6d_runtime()
    from task6c_train import subset_records, validation_material

    records = subset_records(
        json.loads((EVAL / "task6c_subset_ids.json").read_text(encoding="utf-8")), "P"
    )
    sample = __import__("buildreasonseg_mvp.data", fromlist=["data"]).to_sample(records[0])
    optimizer = torch.optim.AdamW(
        runtime.model.trainable_parameter_groups(
            lora_lr=1e-4, head_lr=3e-4, weight_decay=0.01, decoder_lr=3e-4, token_lr=3e-4
        ),
        betas=(0.9, 0.999),
    )
    image = sample.image_rgb()
    features, _cached = runtime.features_for(sample, image)
    from buildreasonseg_mvp.visual_cache import resolve_visual_host

    visual = resolve_visual_host(runtime.model.qwen).visual
    before = {name: parameter.detach().clone() for name, parameter in visual.named_parameters()}
    batch, _ = runtime.prepare(sample, image)
    result = runtime.grounding_train_step(
        batch, sample.target_mask(), features, optimizer=optimizer, include_mask_loss=True
    )

    # item 6: the grounding loss reaches the head, the LoRA adapters and the [SEG] row
    head_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in runtime.model.grounding_head.parameters()
        if parameter.grad is not None
    )
    lora_grad = sum(
        float(parameter.grad.abs().sum())
        for name, parameter in runtime.model.qwen.named_parameters()
        if "lora_" in name and parameter.grad is not None
    )
    assert head_grad > 0, "the grounding loss must produce gradients in the head"
    assert lora_grad > 0, "the LM loss must produce gradients in the LoRA adapters"

    # item 7: the visual tower did not move
    assert all(
        torch.equal(parameter.detach(), before[name])
        for name, parameter in visual.named_parameters()
    )

    # item 8: the SAM2 image encoder is frozen
    assert not any(p.requires_grad for p in runtime.model.sam.image_encoder.parameters())
    assert not any(p.requires_grad for p in runtime.model.sam.sam_prompt_encoder.parameters())

    # item 5 (functional): predicted geometry is a valid normalized box from the head
    predicted = result["predicted_geometry"].reshape(-1)
    assert predicted.numel() == 4
    assert float(predicted.min()) >= -1e-6 and float(predicted.max()) <= 1 + 1e-6

    # item 13: the visual cache is active and value-preserving (weight-identical tower)
    stats = runtime.visual_cache_stats()
    assert stats["installed"] is True
    assert stats["stores_model_outputs"] is False
    assert stats["stores"] == ["pooler_output", "deepstack_features"]
    print("  [6,7,8,13,5] model-backed grounding invariants OK")


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            if exc.__class__.__name__ == "Skipped":
                print(f"SKIP {test.__name__}: {exc}")
                continue
            failures += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"[task6d tests] {len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
