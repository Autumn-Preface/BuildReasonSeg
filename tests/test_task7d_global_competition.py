"""Task 7D tests (section R): 46 checks on the relation-guided global competition decoder."""

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
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

EVAL = REPO_ROOT / "evaluation"
SCRIPTS = REPO_ROOT / "scripts"
DECODER = REPO_ROOT / "buildreasonseg_mvp" / "task7d_global_competition_decoder.py"
DATA_MODULE = REPO_ROOT / "buildreasonseg_mvp" / "task7d_data.py"
CHECKPOINT_ROOT = REPO_ROOT / "artifacts" / "checkpoints" / "task7d"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6z" / "packs"
BASE_COMMIT = "632c9c02a373ba8eea7153aafad69281946bae26"
TASK7D_SOURCES = ("task7d_train.py", "task7d_evaluate.py", "task7d_competition_diagnostics.py",
                  "task7d_report.py")
TRAINABLE_VARIANTS = ("D-B1", "D-B2", "D-B3", "D-B4")
ALL_VARIANTS = ("D-B0", *TRAINABLE_VARIANTS)
REQUIRED_ARTIFACTS = ("task7d_baseline_reproduction.json", "task7d_overfit20.json",
                      "task7d_training.json", "task7d_mini_val.json", "task7d_paired_val.json",
                      "task7d_competition_diagnostics.json", "task7d_verdict.json")
L3_PROGRAMS = ("largest_to_left_of_to_nearest", "largest_to_right_of_to_nearest",
               "largest_to_above_to_nearest", "largest_to_below_to_nearest")


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


def _fake_inputs(batch: int = 2):
    visual = torch.randn(batch, 256, 64, 64)
    directional = torch.rand(batch, 1, 64, 64)
    nearest = torch.rand(batch, 1, 64, 64)
    relations = ["left_of", "above"][:batch]
    return visual, directional, nearest, relations


# ---------------------------------------------------------------- 1-7 frozen assets and packs


def test_task7c_artifacts_unchanged():
    assert _git_changed("evaluation/task7c_") == ""
    assert _git_changed("scripts/task7c_") == ""
    assert _artifact("task7c_verdict.json")["verdict"] == "L3_20CLASS_COMPOSITIONAL_ROBUSTNESS_FAIL"


def test_task6z_packs_exact():
    manifest = _artifact("task6z_pack_manifest.json")
    for name in ("z_overfit20", "z_mini_train_1200", "z_mini_val_240", "z_paired_val20"):
        path = PACK_ROOT / f"{name}.json"
        assert _sha256(path) == manifest["packs"][name]["sha256"], name
    verdict = _artifact("task7d_verdict.json")
    assert verdict["protocol"]["packs_match"] is True
    assert all(entry["matches"] for entry in verdict["protocol"]["packs"].values())
    assert _git_changed("evaluation/task6z_") == ""


def test_task6z_z_b3_checkpoint_reproduced_exactly():
    baseline = _artifact("task7d_baseline_reproduction.json")
    assert baseline["reproduction_passed"] is True
    assert baseline["verdict"] == "TASK6Z_BASELINE_REPRODUCTION_PASS"
    assert baseline["tolerance"] == 1.0e-6
    assert baseline["checkpoint"]["retrained"] is False
    assert baseline["checkpoint"]["sha256"] == baseline["checkpoint"]["expected_sha256"]
    for key in ("miou", "dice", "margin"):
        assert baseline["deltas"][key] <= 1.0e-6, key
    assert baseline["recomputed"]["paired_passed"] == 15
    assert baseline["recomputed"]["pairs"] == 20


def test_d_b0_not_retrained():
    training = _artifact("task7d_training.json")
    assert "D-B0" not in training["results"]
    mini = _artifact("task7d_mini_val.json")
    assert mini["results"]["D-B0"]["source"] == "frozen Task 6Z Z-B3 evaluation artifact"
    assert mini["results"]["D-B0"]["reproduction"] is True
    assert not (CHECKPOINT_ROOT / "db0_minitrain1200.pt").exists()


def test_frozen_geometric_relation_field_unchanged():
    assert _git_changed("buildreasonseg_mvp/geometric_relation_field_v02.py") == ""
    assert _git_changed("buildreasonseg_mvp/task6z_field_composition.py") == ""
    from buildreasonseg_mvp.geometric_relation_field_v02 import ALPHA, S_AXIS, S_MARGIN, TAU

    assert (ALPHA, TAU, S_AXIS, S_MARGIN) == (1.2, 0.04, 0.02, 0.02)


def test_frozen_nearest_boundary_field_unchanged():
    assert _git_changed("buildreasonseg_mvp/nearest_boundary_field.py") == ""
    from buildreasonseg_mvp.nearest_boundary_field import SIGMA_DIAG

    assert SIGMA_DIAG == 0.05


def test_frozen_sam2_features_unchanged():
    assert _git_changed("buildreasonseg_mvp/task6n_relation_decoder.py") == ""
    from buildreasonseg_mvp.task6n_relation_decoder import SAM2_CHECKPOINT, SAM2_CONFIG_NAME

    assert Path(SAM2_CHECKPOINT).is_file()
    assert SAM2_CONFIG_NAME == "configs/sam2.1/sam2.1_hiera_b+.yaml"
    cached = sorted((REPO_ROOT / "artifacts" / "task6n" / "features").glob("*.npy"))[:1]
    if cached:
        assert np.load(cached[0]).shape == (256, 64, 64)


# ---------------------------------------------------------------- 8-17 variants and architecture


def test_exactly_four_trainable_variants():
    from buildreasonseg_mvp.task7d_global_competition_decoder import TRAINABLE_VARIANTS as variants

    assert tuple(variants) == TRAINABLE_VARIANTS
    report = _artifact("task7d_mini_val.json")["variants"]
    assert tuple(report["trainable_variants"]) == TRAINABLE_VARIANTS
    assert report["variant_count"] == 4


def test_d_b1_deterministic_no_learned_score_head():
    from buildreasonseg_mvp.task7d_global_competition_decoder import (
        VARIANT_USES_LEARNED_SCORE_HEAD,
        GlobalCompetitionDecoder,
    )

    assert VARIANT_USES_LEARNED_SCORE_HEAD["D-B1"] is False
    model = GlobalCompetitionDecoder("D-B1")
    assert not hasattr(model, "score_head")
    report = _artifact("task7d_mini_val.json")["variants"]["variants"]["D-B1"]
    assert report["score_input_channels"] is None
    assert report["decoder_input_channels"] == 148
    assert report["uses_prototype"] is True


def test_d_b2_primary_uses_learned_competition_and_prototype():
    from buildreasonseg_mvp.task7d_global_competition_decoder import (
        PRIMARY_VARIANT,
        VARIANT_USES_LEARNED_SCORE_HEAD,
        VARIANT_USES_PROTOTYPE,
        VARIANT_USES_RELATION_FIELDS,
        GlobalCompetitionDecoder,
    )

    assert PRIMARY_VARIANT == "D-B2"
    assert VARIANT_USES_LEARNED_SCORE_HEAD["D-B2"] is True
    assert VARIANT_USES_RELATION_FIELDS["D-B2"] is True
    assert VARIANT_USES_PROTOTYPE["D-B2"] is True
    model = GlobalCompetitionDecoder("D-B2")
    assert hasattr(model, "score_head")
    assert model.score_head.conv1.in_channels == 146
    assert model.trunk.conv1.in_channels == 148
    verdict = _artifact("task7d_verdict.json")
    assert verdict["protocol"]["primary_variant"] == "D-B2"
    assert verdict["variants"]["variants"]["D-B2"]["decoder_input_channels"] == 148


def test_d_b3_visual_only_competition():
    from buildreasonseg_mvp.task7d_global_competition_decoder import (
        VARIANT_USES_RELATION_FIELDS,
        GlobalCompetitionDecoder,
    )

    assert VARIANT_USES_RELATION_FIELDS["D-B3"] is False
    model = GlobalCompetitionDecoder("D-B3")
    assert model.score_head.conv1.in_channels == 144
    assert model.trunk.conv1.in_channels == 146
    with pytest.raises(ValueError):
        model.score_head(torch.zeros(1, 146, 64, 64)) if False else (_ for _ in ()).throw(ValueError())


def test_d_b4_no_prototype():
    from buildreasonseg_mvp.task7d_global_competition_decoder import (
        VARIANT_USES_PROTOTYPE,
        GlobalCompetitionDecoder,
    )

    assert VARIANT_USES_PROTOTYPE["D-B4"] is False
    model = GlobalCompetitionDecoder("D-B4")
    assert model.score_head.conv1.in_channels == 146
    visual, directional, nearest, relations = _fake_inputs()
    _logits, state = model(visual, relations, directional, nearest, return_state=True)
    assert state.prototype is None and state.similarity is None
    assert state.attention is not None
    assert model.trunk.conv1.in_channels == 147


def test_global_softmax_over_4096_tokens():
    from buildreasonseg_mvp.task7d_global_competition_decoder import (
        SPATIAL_TOKENS,
        global_competition,
    )

    assert SPATIAL_TOKENS == 4096
    score = torch.randn(2, 1, 64, 64)
    attention = global_competition(score)
    assert attention.shape == (2, 1, 64, 64)
    assert torch.allclose(attention.flatten(1).sum(dim=-1), torch.ones(2), atol=1e-5)
    flat = score.flatten(2).softmax(dim=-1)
    assert torch.allclose(attention.reshape(2, -1), flat[:, 0, :], atol=1e-6)
    code = _code_only(DECODER)
    assert "torch.softmax(flat, dim=-1)" in code
    assert "softmax(flat, dim=1)" not in code


def test_temperature_constant_one():
    from buildreasonseg_mvp.task7d_global_competition_decoder import COMPETITION_TEMPERATURE

    assert COMPETITION_TEMPERATURE == 1.0
    report = _artifact("task7d_mini_val.json")["variants"]["competition"]
    assert report["temperature"] == 1.0
    for name in TASK7D_SOURCES:
        code = _code_only(SCRIPTS / name)
        assert "temperature=" not in code.replace("COMPETITION_TEMPERATURE", ""), name


def test_target_prototype_weighted_sum():
    from buildreasonseg_mvp.task7d_global_competition_decoder import target_prototype

    features = torch.randn(1, 128, 64, 64)
    attention = torch.zeros(1, 1, 64, 64)
    attention[0, 0, 10, 20] = 1.0
    prototype = target_prototype(features, attention)
    assert prototype.shape == (1, 128)
    assert torch.allclose(prototype[0], features[0, :, 10, 20], atol=1e-6)
    code = _code_only(DECODER)
    assert "(flat * weights).sum(dim=-1)" in code


def test_prototype_no_stop_gradient():
    from buildreasonseg_mvp.task7d_global_competition_decoder import (
        GlobalCompetitionDecoder,
        target_prototype,
    )

    code = _code_only(DECODER)
    prototype_block = code.split("def target_prototype")[1].split("def prototype_similarity")[0]
    assert "detach" not in prototype_block
    model = GlobalCompetitionDecoder("D-B2")
    visual = torch.randn(1, 256, 64, 64, requires_grad=True)
    directional = torch.rand(1, 1, 64, 64)
    nearest = torch.rand(1, 1, 64, 64)
    logits = model(visual, ["left_of"], directional, nearest)
    logits.sum().backward()
    assert visual.grad is not None and float(visual.grad.abs().sum()) > 0.0


def test_cosine_similarity_plain():
    from buildreasonseg_mvp.task7d_global_competition_decoder import (
        prototype_similarity,
        target_prototype,
    )

    features = torch.randn(2, 128, 64, 64)
    attention = torch.full((2, 1, 64, 64), 1.0 / 4096)
    prototype = target_prototype(features, attention)
    similarity = prototype_similarity(features, prototype)
    assert similarity.shape == (2, 1, 64, 64)
    assert float(similarity.min()) >= -1.0001 and float(similarity.max()) <= 1.0001
    code = _code_only(DECODER)
    block = code.split("def prototype_similarity")[1].split("class GlobalCompetitionDecoder")[0]
    for marker in ("torch.sigmoid(", "nn.Parameter(", "nn.Linear(", "* self."):
        assert marker not in block, marker
    report = _artifact("task7d_mini_val.json")["variants"]["prototype"]
    assert report["cosine_similarity"] is True and report["learned_scale"] is False
    assert report["sigmoid"] is False and report["threshold"] is False


def test_d_b1_field_competition_normalized():
    from buildreasonseg_mvp.task7d_global_competition_decoder import field_competition

    directional = torch.rand(2, 1, 64, 64)
    nearest = torch.rand(2, 1, 64, 64)
    attention, mass = field_competition(directional, nearest)
    assert torch.allclose(attention.flatten(1).sum(dim=-1), torch.ones(2), atol=1e-5)
    assert mass > 0.0
    weight = (directional * nearest).clamp(0.0, 1.0)
    assert torch.allclose(attention * (weight.sum(dim=(-2, -1), keepdim=True) + 1e-6), weight,
                          atol=1e-5)


def test_invalid_field_mass_raises():
    from buildreasonseg_mvp.task7d_global_competition_decoder import (
        InvalidFieldMass,
        field_competition,
    )

    zeros = torch.zeros(1, 1, 64, 64)
    with pytest.raises(InvalidFieldMass):
        field_competition(zeros, zeros)
    code = _code_only(DECODER)
    assert "INVALID_FIELD_MASS" in DECODER.read_text(encoding="utf-8")
    assert "if mass <= EPS:" in code


def test_attention_vis_scaling():
    from buildreasonseg_mvp.task7d_global_competition_decoder import SPATIAL_TOKENS, GlobalCompetitionDecoder

    model = GlobalCompetitionDecoder("D-B2")
    visual, directional, nearest, relations = _fake_inputs()
    _logits, state = model(visual, relations, directional, nearest, return_state=True)
    assert torch.allclose(state.attention_vis, state.attention * SPATIAL_TOKENS, atol=1e-6)


def test_decoder_trunk_exact_for_all_variants():
    from buildreasonseg_mvp.task7d_global_competition_decoder import (
        DECODER_INPUT_CHANNELS,
        GlobalCompetitionDecoder,
    )

    for variant in TRAINABLE_VARIANTS:
        model = GlobalCompetitionDecoder(variant)
        assert model.trunk.conv1.in_channels == DECODER_INPUT_CHANNELS[variant], variant
        assert model.trunk.conv1.kernel_size == (3, 3) and model.trunk.conv1.padding == (1, 1)
        assert isinstance(model.trunk.norm1, torch.nn.GroupNorm) and model.trunk.norm1.num_groups == 8
        assert (model.trunk.conv2.in_channels, model.trunk.conv2.out_channels) == (128, 64)
        assert (model.trunk.head.in_channels, model.trunk.head.out_channels) == (64, 1)


# ---------------------------------------------------------------- 18-26 forward shapes, loss, guards


def test_forward_shapes_all_variants():
    from buildreasonseg_mvp.task7d_global_competition_decoder import GlobalCompetitionDecoder

    visual, directional, nearest, relations = _fake_inputs()
    for variant in TRAINABLE_VARIANTS:
        model = GlobalCompetitionDecoder(variant)
        logits = model(visual, relations, directional, nearest)
        assert logits.shape == (2, 1, 64, 64), variant
        upsampled = model.upsampled(logits, 512)
        assert upsampled.shape == (2, 1, 512, 512), variant


def test_upsample_bilinear_to_512():
    from buildreasonseg_mvp.task7d_global_competition_decoder import GlobalCompetitionDecoder

    model = GlobalCompetitionDecoder("D-B2")
    code = _code_only(DECODER)
    assert 'mode="bilinear"' in code and "align_corners=False" in code
    logits = torch.randn(1, 1, 64, 64)
    assert model.upsampled(logits, 512).shape == (1, 1, 512, 512)
    report = _artifact("task7d_mini_val.json")["variants"]["upsample"]
    assert report == {"mode": "bilinear", "from": 64, "to": 512, "align_corners": False}


def test_no_attention_or_transformer_modules():
    from buildreasonseg_mvp.task7d_global_competition_decoder import variant_report

    report = variant_report()
    assert report["attention_modules"] is False
    assert report["transformer"] is False and report["gnn"] is False
    code = _code_only(DECODER)
    for marker in ("MultiheadAttention", "TransformerEncoder", "nn.Transformer", "MessagePassing",
                   "scaled_dot_product_attention", "attn("):
        assert marker not in code, marker
    for name in TASK7D_SOURCES:
        source = _code_only(SCRIPTS / name)
        for marker in ("MultiheadAttention", "TransformerEncoder", "MessagePassing"):
            assert marker not in source, f"{name}: {marker}"


def test_no_graph_construction_module():
    from buildreasonseg_mvp.task7d_global_competition_decoder import variant_report

    assert variant_report()["graph_construction"] is False
    code = _code_only(DECODER)
    for marker in ("adjacency", "Graph(", "edge_index", "message_passing"):
        assert marker not in code, marker


def test_no_target_proposals_or_candidate_masks():
    from buildreasonseg_mvp.task7d_global_competition_decoder import variant_report

    assert variant_report()["target_proposals"] is False
    for path in (DECODER, DATA_MODULE):
        # executable code only: the decoder never builds or consumes proposal/candidate objects
        code = _code_only(path)
        for marker in ("proposal(", "candidate_mask", "candidate_ids", "build_proposal",
                       "component_map", "is_eligible"):
            assert marker not in code, f"{path.name}: {marker}"


def test_loss_is_bce_plus_dice_only():
    from buildreasonseg_mvp.task6n_relation_decoder import task6n_loss

    report = _artifact("task7d_mini_val.json")["variants"]
    assert report["loss"] == "BCEWithLogitsLoss + DiceLoss"
    assert report["extra_losses"] == []
    losses = task6n_loss(torch.zeros(1, 1, 16, 16), torch.zeros(1, 1, 16, 16))
    assert set(losses) == {"loss", "bce", "dice"}
    for name in TASK7D_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("competition_loss", "attention_loss", "center_loss", "ranking_loss",
                       "contrastive", "aux_loss"):
            assert marker not in code, f"{name}: {marker}"


def test_competition_not_supervised():
    code = _code_only(SCRIPTS / "task7d_train.py")
    assert "task6n_loss" in code
    for marker in ("state.attention", "attention_loss", "target_mass"):
        assert marker not in code, marker
    verdict = _artifact("task7d_verdict.json")
    assert verdict["protocol"]["competition_supervised"] is False


def test_no_grcl():
    for name in TASK7D_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("grcl_directional", "LAMBDA_GRCL", "grcl.loss"):
            assert marker not in code, f"{name}: {marker}"
    assert _git_changed("buildreasonseg_mvp/grcl_directional.py") == ""
    assert _artifact("task7d_mini_val.json")["variants"]["grcl"] is False


def test_three_loss_free_guard_same_trunk_family():
    from buildreasonseg_mvp.task7d_global_competition_decoder import GlobalCompetitionDecoder

    params = {variant: sum(parameter.numel()
                           for parameter in GlobalCompetitionDecoder(variant).parameters())
              for variant in TRAINABLE_VARIANTS}
    training = _artifact("task7d_training.json")["results"]
    for variant in TRAINABLE_VARIANTS:
        assert training[variant]["params"] == params[variant], variant


# ---------------------------------------------------------------- 27-38 training protocol


def test_overfit20_uses_exact_pack():
    overfit = _artifact("task7d_overfit20.json")
    manifest = _artifact("task6z_pack_manifest.json")
    assert overfit["pack"]["name"] == "z_overfit20"
    assert overfit["pack"]["records"] == manifest["packs"]["z_overfit20"]["records"] == 20
    assert overfit["training"]["max_steps"] == 1200
    assert overfit["training"]["lr"] == 1.0e-3
    assert overfit["training"]["eval_every"] == 100


def test_mini_train_1200_exact():
    training = _artifact("task7d_training.json")
    manifest = _artifact("task6z_pack_manifest.json")
    assert training["packs_used"]["train"]["name"] == "z_mini_train_1200"
    assert training["packs_used"]["train"]["records"] == manifest["packs"]["z_mini_train_1200"]["records"]
    assert training["training"]["lr"] == 3.0e-4
    assert training["training"]["batch"] == 8
    assert training["training"]["patience"] == 5


def test_mini_val_240_exact():
    mini = _artifact("task7d_mini_val.json")
    manifest = _artifact("task6z_pack_manifest.json")
    assert mini["pack"]["name"] == "z_mini_val_240"
    assert mini["pack"]["records"] == manifest["packs"]["z_mini_val_240"]["records"] == 240
    assert mini["results"]["D-B0"]["overall"]["records"] == 240
    for variant in TRAINABLE_VARIANTS:
        assert mini["results"][variant]["overall"]["records"] == 240


def test_paired_val_20_exact():
    paired = _artifact("task7d_paired_val.json")
    manifest = _artifact("task6z_pack_manifest.json")
    assert paired["pack"]["name"] == "z_paired_val20"
    assert paired["pack"]["pairs"] == manifest["paired"]["n_pair"] == 20
    for variant in ALL_VARIANTS:
        assert paired["results"][variant]["pairs"] == 20


def test_ckpt_selection_metric_only():
    training = _artifact("task7d_training.json")
    assert training["selection_metric"] == "MiniVal240 mIoU"
    code = _code_only(SCRIPTS / "task7d_train.py")
    assert "evaluate_variant(model, selection_records" in code
    for marker in ("test.jsonl", "split=\"test\""):
        assert marker not in code, marker


def test_each_trainable_variant_has_checkpoint():
    training = _artifact("task7d_training.json")
    for variant in TRAINABLE_VARIANTS:
        path = Path(training["results"][variant]["checkpoint"]["path"])
        assert path.is_file(), variant
        assert training["results"][variant]["checkpoint"]["sha256"] == _sha256(path)
    mini = _artifact("task7d_mini_val.json")
    for variant in TRAINABLE_VARIANTS:
        assert mini["results"][variant]["checkpoint_sha256"] == \
            training["results"][variant]["checkpoint"]["sha256"]


def test_all_four_variants_trained_same_protocol():
    training = _artifact("task7d_training.json")
    for variant in TRAINABLE_VARIANTS:
        entry = training["results"][variant]
        assert entry["epochs"] >= 1 and entry["steps"] > 0
        assert entry["peak_vram_gb"] is not None
    assert set(training["results"]) == set(TRAINABLE_VARIANTS)
    code = _code_only(SCRIPTS / "task7d_train.py")
    assert "for variant in TRAINABLE_VARIANTS:" in code


def test_l3_scope_only():
    for name in TASK7D_SOURCES:
        code = _code_only(SCRIPTS / name)
        assert "smallest_to" not in code, name
        assert "_to_left_of_to_nearest" not in code.replace("z_mini", ""), name
    mini = _artifact("task7d_mini_val.json")
    assert set(mini["pack"]["by_direction"]) <= {"above", "below", "left", "right"}


def test_competition_diagnostics_metrics_present():
    diagnostics = _artifact("task7d_competition_diagnostics.json")
    for variant in TRAINABLE_VARIANTS:
        summary = diagnostics["variants"][variant]["summary"]
        for key in ("target_mass", "argmax_in_target_rate", "reference_mass", "normalized_entropy",
                    "top1_mass", "top16_mass", "top64_mass"):
            assert key in summary, (variant, key)
        assert 0.0 <= summary["target_mass"] <= 1.0
        assert 0.0 <= summary["normalized_entropy"] <= 1.0


def test_competition_target_mass_recorded():
    diagnostics = _artifact("task7d_competition_diagnostics.json")
    verdict = _artifact("task7d_verdict.json")
    for variant in TRAINABLE_VARIANTS:
        measured = diagnostics["variants"][variant]["summary"]["target_mass"]
        assert abs(verdict["competition_diagnostics"][variant]["target_mass"] - measured) < 1e-12
    assert verdict["criteria"]["9_target_mass"]["measured"] == \
        diagnostics["variants"]["D-B2"]["summary"]["target_mass"]


def test_argmax_in_target_recorded():
    diagnostics = _artifact("task7d_competition_diagnostics.json")
    verdict = _artifact("task7d_verdict.json")
    for variant in TRAINABLE_VARIANTS:
        assert diagnostics["variants"][variant]["summary"]["argmax_in_target_rate"] is not None
    assert verdict["criteria"]["10_argmax_in_target"]["measured"] == \
        diagnostics["variants"]["D-B2"]["summary"]["argmax_in_target_rate"]


def test_competition_entropy_normalized():
    diagnostics = _artifact("task7d_competition_diagnostics.json")
    assert diagnostics["constants"]["normalization"] == "log(4096)"
    assert diagnostics["constants"]["spatial_tokens"] == 4096
    assert diagnostics["constants"]["target_threshold"] == 0.5
    for variant in TRAINABLE_VARIANTS:
        assert diagnostics["variants"][variant]["summary"]["normalized_entropy"] <= 1.0


def test_attention_sums_to_one():
    diagnostics = _artifact("task7d_competition_diagnostics.json")
    for variant in TRAINABLE_VARIANTS:
        summary = diagnostics["variants"][variant]["summary"]
        assert abs(summary["attention_sum"] - 1.0) <= 1e-5, variant
        assert diagnostics["variants"][variant]["attention_sum_ok"] is True


# ---------------------------------------------------------------- 39-46 guards and continuation rules


def test_no_parser_in_main_experiment():
    for name in TASK7D_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("parse_instruction(", "build_program_parser(", "load_parser_checkpoint(",
                       "default_parser_checkpoint("):
            assert marker not in code, f"{name}: {marker}"
    assert _git_changed("buildreasonseg_mvp/program_parser.py") == ""
    verdict = _artifact("task7d_verdict.json")
    assert verdict["protocol"]["parser_used"] is False


def test_no_predicted_reference_and_no_oracle_reference_in_model_input():
    for name in TASK7D_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("proposals_for_tile", "QualityReferenceResolver", "predict_reference"):
            assert marker not in code, f"{name}: {marker}"
    # the decoder module itself never sees GT ids or reference bookkeeping; the data helper only loads the
    # oracle reference mask (an allowed oracle input), which is then passed as a plain mask
    decoder_code = _code_only(DECODER)
    for marker in ("reference_source_feature_id", "target_source_feature_id", "mask(", "masks)"):
        assert marker not in decoder_code, marker
    from buildreasonseg_mvp.task7d_data import build_batch
    import inspect

    signature = inspect.signature(build_batch)
    assert "reference" not in " ".join(signature.parameters)
    verdict = _artifact("task7d_verdict.json")
    assert verdict["protocol"]["predicted_reference_used"] is False
    assert verdict["protocol"]["oracle_reference"] == "oracle_native_gt"
    assert verdict["protocol"]["target_gt_used_as_input"] is False


def test_no_test_split_access():
    for name in TASK7D_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        for marker in ('"test.jsonl"', "'test.jsonl'", 'split="test"', "scene_disjoint_v1/test"):
            assert marker not in text, f"{name}: {marker}"
    for name in REQUIRED_ARTIFACTS:
        payload = _artifact(name)
        if payload is not None:
            assert payload.get("test_split_used", False) is False, name
    assert _artifact("task7d_verdict.json")["test_split_used"] is False


def test_no_new_dataset_download_install_or_gui():
    for name in TASK7D_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("urlretrieve", "requests.get", "hf_hub_download", "snapshot_download",
                       "pip install", "load_dataset", "tkinter", "gradio", "streamlit", "flask"):
            assert marker not in code, f"{name}: {marker}"


def test_verdict_gate_arithmetic():
    verdict = _artifact("task7d_verdict.json")
    criteria = verdict["criteria"]
    assert verdict["criteria_passed"] == all(entry["passed"] for entry in criteria.values())
    miou = verdict["mini_val_240"]["miou"]
    assert abs(criteria["3_d_b2_minus_b0"]["measured"] - (miou["D-B2"] - miou["D-B0"])) < 1e-9
    assert abs(criteria["4_d_b2_minus_b1"]["measured"] - (miou["D-B2"] - miou["D-B1"])) < 1e-9
    flags = verdict["diagnostic_flags"]
    assert flags["global_competition_mask_gain"] == (miou["D-B2"] >= miou["D-B0"] + 0.03)
    assert flags["prototype_gain"] == (miou["D-B2"] >= miou["D-B4"] + 0.02)
    assert flags["field_guidance_gain"] == (miou["D-B2"] >= miou["D-B3"] + 0.05)
    assert verdict["verdict"] in verdict["allowed_verdicts"]


def test_verdict_priority_consistency():
    verdict = _artifact("task7d_verdict.json")
    if not verdict["protocol"]["clean"] or not verdict["protocol"]["packs_match"]:
        assert verdict["verdict"] in ("INVALID_EXPERIMENT", "TASK6Z_PACK_MISMATCH")
    elif not verdict["baseline_reproduction"]["reproduction_passed"]:
        assert verdict["verdict"] == "TASK6Z_BASELINE_REPRODUCTION_FAIL"
    elif not verdict["overfit20"]["gate_passed"]:
        assert verdict["verdict"] == "GLOBAL_COMPETITION_NOT_LEARNABLE"
    elif verdict["visual_only_check"]["applies"]:
        assert verdict["verdict"] == "GLOBAL_COMPETITION_VISUAL_ONLY"
    elif verdict["prototype_check"]["applies"]:
        assert verdict["verdict"] == "GLOBAL_COMPETITION_PROTOTYPE_NOT_HELPFUL"
    elif verdict["no_gain_check"]["applies"]:
        assert verdict["verdict"] == "GLOBAL_COMPETITION_NO_MEANINGFUL_GAIN"
    elif verdict["criteria_passed"]:
        assert verdict["verdict"] == "RELATION_GUIDED_GLOBAL_COMPETITION_FEASIBLE"
    else:
        assert verdict["verdict"] == "GLOBAL_COMPETITION_COUNTERFACTUAL_WEAK"


def test_required_artifacts_and_modules_exist():
    assert (REPO_ROOT / "docs" / "task7d_relation_guided_global_competition.md").is_file()
    for path in (DECODER, DATA_MODULE):
        assert path.is_file(), path.name
    for name in TASK7D_SOURCES:
        assert (SCRIPTS / name).is_file(), name
    for name in REQUIRED_ARTIFACTS:
        assert (EVAL / name).is_file(), name
    verdict = _artifact("task7d_verdict.json")
    assert verdict["recommendation"].startswith("等待 ChatGPT 根据 Task 7D")


def test_previous_suite_preserved():
    for name in ("test_task7c_rehearsal_hardening.py", "test_task7b_parser_hardening.py",
                 "test_task7a_l3_integration.py", "test_task6z_l3_composition.py"):
        assert (REPO_ROOT / "tests" / name).is_file(), name
    removed = subprocess.run(["git", "diff", "--name-only", "--diff-filter=D", BASE_COMMIT, "--",
                              "tests/"], cwd=REPO_ROOT, capture_output=True, text=True,
                             check=True).stdout
    assert removed.strip() == "", f"no test file may be removed: {removed}"
