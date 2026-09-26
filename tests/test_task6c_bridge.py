"""Task 6C section 17 items 14-20: evaluation lookup, the two bridges, and the
inference-validity / scope checks.

14. operation lookup built without val text
15. centre bridge remains Task 6B behaviour
16. language-only bridge contains no point/box/mask prompt
17. language-only sparse tensor shape correct
18. no GT geometry in inference
19. no `[REF]`
20. no 4B

Run with pytest, or directly::

    python tests/test_task6c_bridge.py
"""

from __future__ import annotations

import ast
import inspect
import json
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from buildreasonseg_mvp import language_metrics as LM  # noqa: E402
from buildreasonseg_mvp import sam2_bridge as SB  # noqa: E402

SCRIPTS = sorted((REPO_ROOT / "scripts").glob("task6c_*.py"))


class _StubPromptEncoder(nn.Module):
    """Minimal stand-in for SAM2's prompt encoder, enough to observe the call."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.point_embeddings = nn.ModuleList([nn.Embedding(2, dim)])
        self.calls: list[dict] = []

    def forward(self, points=None, boxes=None, masks=None):
        self.calls.append({"points": points, "boxes": boxes, "masks": masks})
        batch = 1
        sparse = torch.zeros(batch, 1, self.point_embeddings[0].embedding_dim)
        if points is not None:
            coordinates, labels = points
            sparse = torch.ones(batch, 1, self.point_embeddings[0].embedding_dim) * 3.0
            # emulate SAM appending its own padding slot
            sparse = torch.cat([sparse, torch.full_like(sparse, 0.5)], dim=1)
        return sparse, torch.zeros(1, self.point_embeddings[0].embedding_dim, 4, 4)


class _StubSam(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.sam_prompt_encoder = _StubPromptEncoder(dim)


def test_centre_bridge_is_task6b_behaviour():
    sam = _StubSam()
    projected = torch.randn(1, 8)
    sparse, diagnostics = SB.build_sparse_prompt(sam, projected, bridge="centre")
    encoder = sam.sam_prompt_encoder
    assert len(encoder.calls) == 1, "the centre bridge must consult the prompt encoder once"
    call = encoder.calls[0]
    assert call["points"] is not None, "the centre bridge uses a real point prompt"
    coordinates, labels = call["points"]
    assert tuple(coordinates.shape) == (1, 1, 2)
    assert coordinates[0, 0].tolist() == list(SB.PROMPT_ANCHOR_XY)
    assert labels[0, 0].item() == 1, "the point label must be positive"
    assert call["boxes"] is None and call["masks"] is None
    # slot 0 = point embedding + projected language vector; slot 1 = SAM padding
    assert sparse.shape == (1, 2, 8)
    assert torch.allclose(sparse[:, 0, :], torch.full((1, 8), 3.0) + projected, atol=1e-6)
    assert diagnostics["point_prompt_used"] is True
    assert diagnostics["uses_point_coordinates"] is True
    assert diagnostics["ratio_projected_over_point"] is not None
    print("  [15] centre bridge reproduces the Task 6B point + language behaviour OK")


def test_language_bridge_has_no_point_box_or_mask_prompt():
    sam = _StubSam()
    projected = torch.randn(1, 8)
    sparse, diagnostics = SB.build_sparse_prompt(sam, projected, bridge="language")
    assert sam.sam_prompt_encoder.calls == [], "the language bridge must not call the prompt encoder"
    assert diagnostics["point_prompt_used"] is False
    assert diagnostics["uses_point_coordinates"] is False
    assert diagnostics["uses_point_labels"] is False
    assert diagnostics["uses_box_prompt"] is False
    assert diagnostics["uses_mask_prompt"] is False
    print("  [16] language bridge creates no point, box or mask prompt OK")


def test_language_bridge_sparse_shape_and_values():
    sam = _StubSam(dim=16)
    projected = torch.randn(3, 16)
    sparse, _diagnostics = SB.build_sparse_prompt(sam, projected, bridge="language")
    assert tuple(sparse.shape) == (3, 1, 16), sparse.shape
    assert sparse.dtype == SB.sparse_prompt_dtype(sam)
    assert torch.allclose(sparse[:, 0, :].float(), projected.float(), atol=1e-6), (
        "the language bridge's sparse prompt must be the projected token itself"
    )
    print("  [17] language-bridge sparse tensor shape/dtype/values OK")


def test_unknown_bridge_is_rejected():
    sam = _StubSam()
    try:
        SB.build_sparse_prompt(sam, torch.randn(1, 8), bridge="magic")
    except ValueError:
        print("  [extra] unknown bridge rejected OK")
        return
    raise AssertionError("an unknown bridge must raise ValueError")


def test_operation_lookup_is_built_without_val_text():
    train_record = {"reasoning_zh": "TRAIN ONLY REASONING", "query_type": "largest"}
    val_record = {"reasoning_zh": "VAL ONLY REASONING", "query_type": "smallest"}
    lookup = LM.build_reasoning_lookup([train_record])
    assert LM.normalise("VAL ONLY REASONING") not in lookup
    assert LM.normalise("TRAIN ONLY REASONING") in lookup
    assert not LM.operation_chain_correct("VAL ONLY REASONING [SEG]", "smallest", lookup)

    source = (REPO_ROOT / "scripts" / "task6c_train.py").read_text(encoding="utf-8")
    assert "build_reasoning_lookup(train_records)" in source, (
        "Task 6C must build the lookup from the train split only"
    )
    assert "build_reasoning_lookup(list(data_mod.read_records(\"train\")) + list(data_mod.read_records(\"val\")))" not in source
    print("  [14] operation lookup built from train text only OK")


def _strip_comments(text: str) -> str:
    text = re.sub(r"#.*", "", text)
    text = re.sub(r'"""[\s\S]*?"""', "", text)
    text = re.sub(r"'''[\s\S]*?'''", "", text)
    return text


def _string_literals(tree: ast.AST) -> list[str]:
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def test_no_ref_token_in_task6c_code():
    """No `[REF]` pathway is implemented (prose mentions in longer strings are allowed)."""

    import ast

    for path in SCRIPTS:
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        literals = _string_literals(tree)
        assert "[REF]" not in literals, f"{path.name} defines a literal '[REF]'"
        identifiers = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        } | {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        } | {
            node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)
        }
        offenders = {name for name in identifiers if "ref_token" in name or "ref_embedding" in name}
        assert not offenders, f"{path.name} declares {offenders}"

    config_text = (REPO_ROOT / "configs" / "mvp" / "task6c_2b_ablation.yaml").read_text(encoding="utf-8")
    import yaml

    cfg = yaml.safe_load(config_text)
    token_keys = set(cfg.get("token", {}))
    ref_keys = {
        key for key in token_keys if key.lower() in {"ref", "ref_token"} or key.lower().startswith("ref_")
    }
    assert not ref_keys, ref_keys
    assert cfg["token"]["seg_token"] == "[SEG]"
    assert not any(
        isinstance(value, str) and "[REF]" in value for value in cfg.get("token", {}).values()
    )
    setup_source = inspect.getsource(
        __import__("buildreasonseg_mvp.qwen_seg", fromlist=["setup_seg_token"]).setup_seg_token
    )
    assert "[REF]" not in setup_source
    print("  [19] no [REF] pathway in Task 6C code/config OK")


def test_no_4b_model_in_task6c_code():
    for path in SCRIPTS + [REPO_ROOT / "configs" / "mvp" / "task6c_2b_ablation.yaml"]:
        code = _strip_comments(path.read_text(encoding="utf-8"))
        assert not re.search(r"Qwen3-VL-4B", code), f"{path.name} references 4B"
    print("  [20] no 4B reference in Task 6C code/config OK")


def test_no_ground_truth_geometry_in_inference():
    """Inference takes image + instruction only; ground truth is used to score.

    The per-record output legitimately *records* `target_component_id` for scoring and
    reporting (Task 6B does the same), so the check is on the inputs of the model
    calls, not on the presence of the field.
    """

    import ast

    from buildreasonseg_mvp import validation as V

    banned = ("target_mask", "reasoning_zh", "target_component_id", "candidate_component_ids",
              "reference_component_ids", "distractor_component_ids")
    model_calls = ("generate_with_seg", "seg_hidden_from_full_sequence", "decode_mask")

    for function in (V.free_generation_validation, V.paired_probe):
        tree = ast.parse(inspect.getsource(function).replace("\\\n", "").lstrip())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name not in model_calls:
                continue
            source = ast.unparse(node)
            for token in banned:
                assert token not in source, f"{function.__name__} passes {token} into {name}: {source[:200]}"

    # the generator itself must not accept geometry at all
    from buildreasonseg_mvp.qwen_seg import generate_with_seg

    parameters = set(inspect.signature(generate_with_seg).parameters)
    assert not (parameters & set(banned)), parameters
    print("  [18] no ground-truth geometry on the inference path OK")


def main() -> int:
    tests = [
        ("15 centre bridge", test_centre_bridge_is_task6b_behaviour),
        ("16 language bridge", test_language_bridge_has_no_point_box_or_mask_prompt),
        ("17 language shape", test_language_bridge_sparse_shape_and_values),
        ("bridge validation", test_unknown_bridge_is_rejected),
        ("14 lookup", test_operation_lookup_is_built_without_val_text),
        ("19 no [REF]", test_no_ref_token_in_task6c_code),
        ("20 no 4B", test_no_4b_model_in_task6c_code),
        ("18 no GT geometry", test_no_ground_truth_geometry_in_inference),
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
    print(f"\n{len(tests) - failures}/{len(tests)} task6c bridge checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
