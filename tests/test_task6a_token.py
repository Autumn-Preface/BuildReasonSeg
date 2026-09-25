"""Task 6A `[SEG]` token tests.

Covers, from the required list:

 2. `[SEG]` is exactly one tokenizer token
 3. only the assistant target is labelled for the LM cross-entropy
 5. selective `[SEG]` training leaves ordinary vocabulary rows unchanged

These need the downloaded assets; they skip cleanly otherwise.

Run with pytest, or directly::

    python tests/test_task6a_token.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.qwen_seg import build_teacher_forcing_batch  # noqa: E402
from task6a_fixtures import require_model_assets, runtime, smoke_samples  # noqa: E402


def test_seg_is_exactly_one_token():
    require_model_assets()
    rt = runtime()
    tokenizer = rt.tokenizer
    seg_id = rt.model.seg_token_id

    ids = tokenizer.encode(data_mod.SEG_TOKEN, add_special_tokens=False)
    assert len(ids) == 1, f"[SEG] must be one token, got {ids}"
    assert ids[0] == seg_id

    assert tokenizer.decode([seg_id], skip_special_tokens=False).strip() == data_mod.SEG_TOKEN
    assert rt.reports["token"]["single_token_roundtrip"] is True
    assert rt.reports["token"]["vocab_after"] == rt.reports["token"]["vocab_before"] + 1
    print(f"  [2] [SEG] is one token (id={seg_id}) OK")


def test_embedding_resize_happened_once_and_ties_are_intact():
    require_model_assets()
    rt = runtime()
    token_report = rt.reports["token"]

    # Qwen ships an embedding table PADDED beyond its tokenizer (151,936 rows for
    # a 151,669-token tokenizer). `resize_token_embeddings(len(tokenizer))`
    # therefore normalises the table to the tokenizer length plus the new token
    # rather than only appending a row. Those padding rows were unreachable from
    # the tokenizer, so the shrink removes dead weight; it is asserted explicitly
    # so the behaviour is visible rather than assumed.
    assert token_report["embedding_shape_after"][0] == token_report["vocab_after"]
    assert token_report["vocab_after"] == token_report["vocab_before"] + 1
    assert token_report["seg_token_id"] == token_report["vocab_after"] - 1
    assert token_report["embedding_shape_before"][1] == token_report["embedding_shape_after"][1]
    assert token_report["tied_embeddings"] is True
    assert token_report["tied_after_resize"] is True

    # and the resize must not have happened twice
    current_rows = int(rt.qwen.get_input_embeddings().weight.shape[0])
    assert current_rows == token_report["embedding_shape_after"][0]
    print(
        "  [2b] embedding resized exactly once, weight tying preserved OK "
        f"(tokenizer {token_report['vocab_after']}, table "
        f"{token_report['embedding_shape_before'][0]} -> {current_rows})"
    )


def test_only_assistant_target_is_labelled():
    require_model_assets()
    rt = runtime()
    samples = smoke_samples()

    for sample in samples:
        image = sample.image_rgb()
        batch = build_teacher_forcing_batch(
            rt.processor,
            rt.tokenizer,
            image,
            sample.instruction_zh,
            sample.assistant_text,
            rt.model.seg_token_id,
        )
        labels = batch.labels[0]
        supervised = (labels != -100).nonzero(as_tuple=False).flatten().tolist()

        # every supervised position is inside the assistant span
        assert min(supervised) >= batch.prompt_length - 1, "prompt tokens must not be supervised"
        assert max(supervised) <= batch.total_length - 2

        # the label at position i is the token at i+1 (standard shifted targets)
        for position in supervised:
            assert int(labels[position]) == int(batch.input_ids[0, position + 1])

        # the [SEG] token itself is part of the supervised target
        assert batch.seg_position in supervised or batch.seg_position - 1 in supervised

        # the prompt (image + instruction) contributes zero loss terms
        assert int((labels[: batch.prompt_length - 1] != -100).sum()) == 0
    print(f"  [3] LM labels cover only the assistant target OK ({len(samples)} samples)")


def test_selective_token_training_leaves_vocab_rows_unchanged():
    require_model_assets()
    rt = runtime()

    embedding = rt.qwen.get_input_embeddings()
    adapter = getattr(embedding, "token_adapter", None)
    assert adapter is not None, "expected PEFT TrainableTokensWrapper around the embedding"
    assert adapter.base_layer.weight.requires_grad is False, "the base embedding table must be frozen"

    watched = [0, 1, 100, 5000, 100000, 151000]
    before = {token: adapter.base_layer.weight[token].detach().float().clone() for token in watched}
    seg_before = adapter.trainable_tokens_delta[
        next(iter(adapter.trainable_tokens_delta))
    ].detach().float().clone()

    optimizer = rt.build_optimizer()
    sample = smoke_samples()[0]
    batch, image = rt.prepare(sample)
    features, _ = rt.features_for(sample, image)
    rt.train_step(batch, sample.target_mask(), features, optimizer=optimizer)

    seg_after = adapter.trainable_tokens_delta[next(iter(adapter.trainable_tokens_delta))].detach().float()
    assert float((seg_after - seg_before).abs().max()) > 0, "the [SEG] row must change"

    for token in watched:
        delta = float((adapter.base_layer.weight[token].detach().float() - before[token]).abs().max())
        assert delta == 0.0, f"ordinary vocabulary row {token} changed by {delta}"

    trainable_token_params = [
        name for name, parameter in rt.qwen.named_parameters() if parameter.requires_grad and "trainable_tokens" in name
    ]
    assert len(trainable_token_params) == 1
    assert rt.reports["params"]["token_params"] == rt.qwen.config.text_config.hidden_size
    assert rt.reports["params"]["embedding_base_frozen"] is True
    print("  [5] selective [SEG] training leaves normal vocab rows unchanged OK")


def main() -> int:
    tests = [
        ("2 one token", test_seg_is_exactly_one_token),
        ("2b resize/tying", test_embedding_resize_happened_once_and_ties_are_intact),
        ("3 assistant-only labels", test_only_assistant_target_is_labelled),
        ("5 selective token training", test_selective_token_training_leaves_vocab_rows_unchanged),
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
    print(f"\n{len(tests) - failures}/{len(tests)} task6a token checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
