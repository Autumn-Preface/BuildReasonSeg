"""Task 6A trainable-surface tests.

Covers, from the required list:

 6. LoRA attaches to no visual-tower module
10. there is no `[REF]` path in Task 6A

plus a source-level audit that Task 6A introduces no forbidden component.

Run with pytest, or directly::

    python tests/test_task6a_trainables.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from task6a_fixtures import require_model_assets, runtime  # noqa: E402

MVP_PACKAGE = REPO_ROOT / "buildreasonseg_mvp"
MVP_SCRIPTS = [
    REPO_ROOT / "scripts" / "task6a_probe.py",
    REPO_ROOT / "scripts" / "task6a_smoke2.py",
    REPO_ROOT / "scripts" / "task6a_overfit20.py",
    REPO_ROOT / "scripts" / "task6a_infer.py",
]

VISUAL_MARKERS = ("visual", "vision_tower", "vision_model")


def test_lora_targets_no_visual_module():
    require_model_assets()
    rt = runtime()

    visual_trainable = [
        name
        for name, parameter in rt.model.named_parameters()
        if parameter.requires_grad and any(marker in name for marker in VISUAL_MARKERS)
    ]
    assert visual_trainable == [], f"trainable visual-tower parameters: {visual_trainable[:5]}"

    lora_names = [
        name for name, module in rt.model.named_modules() if "lora_" in name
    ]
    visual_lora = [name for name in lora_names if any(marker in name for marker in VISUAL_MARKERS)]
    assert visual_lora == [], f"LoRA leaked into the visual tower: {visual_lora[:5]}"

    # every LoRA target is a language-model projection
    targets = rt.reports["lora"]["target_modules"]
    assert targets, "no LoRA target modules recorded"
    for name in targets:
        assert "language_model" in name, f"non-text LoRA target: {name}"
        assert name.endswith(("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))
    assert rt.reports["lora"]["n_visual_lora_modules"] == 0
    print(f"  [6] LoRA targets {len(targets)} text projections, 0 visual modules OK")


def _code_without_comments_or_strings(path: Path) -> str:
    """Source text with comments and string literals removed.

    Needed because the docstrings legitimately *say* that `[REF]` is absent, and a
    raw substring scan would flag the explanation rather than the code.
    """

    import io
    import tokenize

    pieces: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for token in tokenize.generate_tokens(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            pieces.append(token.string)
    return " ".join(pieces)


def test_no_ref_path_in_task6a():
    """Task 6A has exactly one new token; `[REF]` must not appear in any code path."""

    forbidden_token = "[REF]"
    for path in sorted(MVP_PACKAGE.glob("*.py")) + [p for p in MVP_SCRIPTS if p.is_file()]:
        code = _code_without_comments_or_strings(path)
        assert forbidden_token not in code, f"{path.name} uses {forbidden_token} in code"

    # the token is only ever named in prose, so confirm the prose is present
    joined = "\n".join(p.read_text(encoding="utf-8") for p in MVP_PACKAGE.glob("*.py"))
    assert forbidden_token in joined, "expected an explicit note that Task 6A has no [REF]"

    from buildreasonseg_mvp import data as data_mod

    assert data_mod.SEG_TOKEN == "[SEG]"
    print("  [10] no [REF] path in Task 6A OK")


def test_no_forbidden_components_implemented():
    """No Spatial Relation Encoder / Spatial Consistency Loss / 4B model is implemented."""

    forbidden_patterns = {
        "spatial relation encoder": re.compile(r"class\s+\w*RelationEncoder", re.IGNORECASE),
        "spatial consistency loss": re.compile(r"class\s+\w*ConsistencyLoss", re.IGNORECASE),
        "qwen3-vl-4b": re.compile(r"Qwen3-VL-4B", re.IGNORECASE),
    }
    for path in sorted(MVP_PACKAGE.glob("*.py")):
        code = _code_without_comments_or_strings(path)
        for label, pattern in forbidden_patterns.items():
            assert not pattern.search(code), f"{path.name} implements forbidden {label}"

    # 4B must not be configurable either: the only model id in the Task 6A config
    # is the 2B one.
    import yaml

    config = yaml.safe_load((REPO_ROOT / "configs" / "mvp" / "task6a_2b_seg.yaml").read_text(encoding="utf-8"))
    model_ids = {value for key, value in config["models"].items() if key.endswith("_model_id") or key.endswith("_repo_id")}
    assert not any("4B" in str(value) or "4b" in str(value) for value in model_ids), model_ids
    print("  [extra] no forbidden Task 6A components OK")


def test_trainable_surface_is_exactly_the_four_allowed_groups():
    require_model_assets()
    rt = runtime()

    # The output-row delta is created lazily on the first forward pass, so run one
    # training step before enumerating the trainable surface.
    from task6a_fixtures import smoke_samples

    sample = smoke_samples()[0]
    batch, image = rt.prepare(sample)
    features, _ = rt.features_for(sample, image)
    rt.train_step(batch, sample.target_mask(), features, optimizer=None)

    groups = {"lora": 0, "token": 0, "output_row": 0, "projection": 0, "sam_mask_decoder": 0, "other": 0}
    others: list[str] = []
    from buildreasonseg_mvp.qwen_seg import output_row_param_ids

    embedding = rt.qwen.get_input_embeddings()
    input_adapter = getattr(embedding, "token_adapter", None)
    assert input_adapter is not None
    input_token_delta = input_adapter.trainable_tokens_delta[
        next(iter(input_adapter.trainable_tokens_delta))
    ]

    output_row_ids = output_row_param_ids(rt.model)
    for name, parameter in rt.model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name:
            groups["lora"] += 1
        elif id(parameter) in output_row_ids:
            groups["output_row"] += 1
        elif "trainable_tokens" in name or "token_row" in name:
            groups["token"] += 1
        elif name.startswith("projection."):
            groups["projection"] += 1
        elif name.startswith("sam.sam_mask_decoder."):
            groups["sam_mask_decoder"] += 1
        else:
            groups["other"] += 1
            others.append(name)

    assert groups["other"] == 0, f"unexpected trainable tensors: {others[:6]}"
    assert all(groups[key] > 0 for key in ("lora", "token", "projection", "sam_mask_decoder"))

    # PEFT preserves the weight tying semantically for the trainable token: the
    # output head's token adapter shares the SAME delta parameter object as the
    # input embedding's adapter. There is therefore exactly ONE trainable `[SEG]`
    # row, used on both sides -- which is what "tied embeddings" should mean.
    assert rt.reports["lora"]["peft_wraps_output_head"] is True
    output_head = rt.qwen.get_output_embeddings()
    output_adapter = getattr(output_head, "token_adapter", None)
    assert output_adapter is not None, "the output head must carry a PEFT token adapter"
    output_deltas = list(output_adapter.trainable_tokens_delta.values())
    assert len(output_deltas) == 1
    assert output_deltas[0] is input_token_delta, (
        "with tied embeddings the input and output adapters must share one delta"
    )
    assert output_deltas[0].numel() == rt.reports["qwen_hidden_size"]
    assert output_deltas[0].requires_grad

    assert groups["token"] == 1, "expected exactly one shared trainable [SEG] row"

    # every trainable parameter must be covered by the optimizer
    optimizer = rt.build_optimizer()
    covered = {id(p) for group in optimizer.param_groups for p in group["params"]}
    trainable_ids = {id(p) for p in rt.model.parameters() if p.requires_grad}
    assert trainable_ids - covered == set(), "some trainable tensors are not optimised"

    params = rt.reports["params"]
    assert params["trainable_params"] < params["total_params"] * 0.02
    assert params["token_params"] >= rt.reports["qwen_hidden_size"]
    print(f"  [extra] trainable surface OK: {groups}, optimizer covers {len(covered)} tensors")
    return None


def main() -> int:
    tests = [
        ("6 LoRA scope", test_lora_targets_no_visual_module),
        ("10 no [REF]", test_no_ref_path_in_task6a),
        ("no forbidden components", test_no_forbidden_components_implemented),
        ("trainable surface", test_trainable_surface_is_exactly_the_four_allowed_groups),
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
    print(f"\n{len(tests) - failures}/{len(tests)} task6a trainable checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
