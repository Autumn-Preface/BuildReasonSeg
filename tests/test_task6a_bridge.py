"""Task 6A SAM2 bridge tests.

Covers, from the required list:

 7. SAM bridge dimensions line up end to end
 8. the SAM2 image encoder is frozen and receives no gradient
 9. the projection MLP and the SAM2 mask decoder are trainable
11. the inference path consumes no ground-truth geometry

Run with pytest, or directly::

    python tests/test_task6a_bridge.py
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from buildreasonseg_mvp.sam2_bridge import (  # noqa: E402
    PROMPT_ANCHOR_XY,
    ProjectionMLP,
    build_sparse_prompt,
    decode_mask,
)
from task6a_fixtures import require_model_assets, runtime, smoke_samples  # noqa: E402


def test_bridge_dimensions():
    require_model_assets()
    rt = runtime()
    sample = smoke_samples()[0]
    image = sample.image_rgb()
    features = rt.sam_encoder.encode(image)

    assert features.image_embeddings.shape[1] == rt.reports["sam_prompt_embed_dim"] == 256
    assert features.image_embeddings.shape[-2:] == (64, 64)
    assert features.transformed_shape == (1, 3, 1024, 1024)
    assert features.source_shape == (512, 512, 3)
    assert features.high_res_shapes == [(1, 32, 256, 256), (1, 64, 128, 128)]

    hidden_size = rt.reports["qwen_hidden_size"]
    device = features.image_embeddings.device
    projection = ProjectionMLP(in_dim=hidden_size, out_dim=256).to(device)
    projected = projection(torch.zeros(1, hidden_size, device=device))
    assert projected.shape == (1, 256)

    sparse = build_sparse_prompt(rt.sam, projected)
    assert sparse.shape == (1, 2, 256), "one real prompt slot plus prompt-encoder padding"
    assert tuple(PROMPT_ANCHOR_XY) == (0.5, 0.5)

    decoded = decode_mask(rt.sam, features, projected.detach(), multimask_output=False)
    assert decoded.low_res_logits.shape == (1, 1, 256, 256)
    assert decoded.iou_prediction.shape[0] == 1
    print("  [7] SAM bridge dimensions OK")


def test_sam_image_encoder_frozen_and_no_grad():
    require_model_assets()
    rt = runtime()
    for name, parameter in rt.sam.named_parameters():
        if "image_encoder" in name or "memory" in name or "sam_prompt_encoder" in name:
            assert not parameter.requires_grad, f"{name} must be frozen"
    assert all(not p.requires_grad for p in rt.sam.sam_prompt_encoder.parameters())
    assert all(not p.requires_grad for p in rt.sam.image_encoder.parameters())

    # top-level parameters that are not child modules must also be frozen
    top_level = {"no_mem_embed", "no_mem_pos_enc", "maskmem_tpos_enc", "no_obj_ptr", "no_obj_embed_spatial"}
    for name in top_level:
        parameter = getattr(rt.sam, name, None)
        if parameter is not None:
            assert not parameter.requires_grad, f"top-level SAM2 parameter {name} must be frozen"

    sample = smoke_samples()[0]
    batch, image = rt.prepare(sample)
    features, _ = rt.features_for(sample, image)
    step = rt.train_step(batch, sample.target_mask(), features, optimizer=None)
    for name, parameter in rt.sam.named_parameters():
        if "image_encoder" in name:
            assert parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0
    assert step["grad_norms"]["sam_image_encoder"]["norm"] == 0.0
    print("  [8] SAM2 image encoder frozen with no gradient OK")


def test_projection_and_mask_decoder_trainable():
    require_model_assets()
    rt = runtime()
    assert all(p.requires_grad for p in rt.projection.parameters())
    assert all(p.requires_grad for p in rt.sam.sam_mask_decoder.parameters())
    assert rt.reports["projection"]["params"] > 0
    assert rt.reports["sam2"]["mask_decoder_params"] > 0
    print("  [9] projection and SAM2 mask decoder are trainable OK")


def test_inference_path_takes_no_ground_truth():
    """Structural check: no inference entry point accepts GT geometry."""

    from buildreasonseg_mvp import qwen_seg, sam2_bridge

    forbidden = ("gt", "ground_truth", "target_id", "target_component", "centroid", "bbox", "reference")

    for function in (sam2_bridge.decode_mask, sam2_bridge.build_sparse_prompt, sam2_bridge.Sam2Encoder.encode):
        signature = inspect.signature(function)
        for name in signature.parameters:
            assert not any(token in name.lower() for token in forbidden), (
                f"{function.__qualname__} exposes a suspicious parameter {name!r}"
            )

    generate_parameters = inspect.signature(qwen_seg.generate_with_seg).parameters
    for name in generate_parameters:
        assert not any(token in name.lower() for token in forbidden), name

    # the decode path consumes only the projected language embedding + frozen features
    assert list(inspect.signature(sam2_bridge.decode_mask).parameters)[:3] == [
        "sam",
        "features",
        "projected",
    ]
    print("  [11] inference path consumes no GT geometry OK")


def main() -> int:
    tests = [
        ("7 bridge dimensions", test_bridge_dimensions),
        ("8 frozen encoder", test_sam_image_encoder_frozen_and_no_grad),
        ("9 trainable projection/decoder", test_projection_and_mask_decoder_trainable),
        ("11 no GT geometry at inference", test_inference_path_takes_no_ground_truth),
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
    print(f"\n{len(tests) - failures}/{len(tests)} task6a bridge checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
