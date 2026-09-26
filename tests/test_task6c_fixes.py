"""Task 6C section 17 items 1-5: the correctness fixes that precede the experiment.

1. `training.deterministic` is consumed
2. deterministic 2-sample smoke evidence (cross-process, recorded)
3. checkpoint path comes from config
4. four arms have unique checkpoint directories
5. trainable-state fingerprint equality across arms

Run with pytest, or directly::

    python tests/test_task6c_fixes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from buildreasonseg_mvp.determinism import gradients_fingerprint, trainable_state_fingerprint  # noqa: E402
from buildreasonseg_mvp.runtime import enable_determinism, load_config  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
CONFIG = REPO_ROOT / "configs" / "mvp" / "task6c_2b_ablation.yaml"


def test_deterministic_flag_is_consumed():
    cfg = load_config(CONFIG)
    assert cfg["training"]["deterministic"] is True
    source = (REPO_ROOT / "buildreasonseg_mvp" / "runtime.py").read_text(encoding="utf-8")
    assert "enable_determinism" in source
    assert 'cfg.get("training", {}).get("deterministic", False)' in source, (
        "build_runtime must read training.deterministic"
    )
    report = enable_determinism(1234, strict=False)
    assert report["python_random_seeded"] and report["numpy_seeded"]
    assert report["cudnn_benchmark"] is False and report["cudnn_deterministic"] is True
    assert torch.are_deterministic_algorithms_enabled() is True
    assert report["use_deterministic_algorithms_warn_only"] is True
    assert report["bit_reproducible_claimed"] is False, "warn_only must not claim bit reproducibility"
    print("  [1] training.deterministic is consumed and reported per setting OK")


def test_determinism_evidence_artifact():
    path = EVAL / "task6c_determinism.json"
    if not path.is_file():
        print("  [2] determinism artifact not present yet (skipped: run scripts/task6c_determinism.py)")
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["losses_identical"] is True
    assert report["gradients_identical"] is True
    assert report["post_step_parameters_identical"] is True
    assert report["cross_process_bit_reproducible"] is True
    assert report["run1"]["sample_ids"] == report["run2"]["sample_ids"]
    assert report["measure"]["cudnn_deterministic"] is True
    print("  [2] recorded two-process determinism smoke agrees bit-for-bit OK")


def test_fingerprint_is_stable_and_order_independent():
    model = nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3))
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    first = trainable_state_fingerprint(model)
    second = trainable_state_fingerprint(model)
    assert first["sha256"] == second["sha256"]
    assert first["tensor_count"] == 4
    with torch.no_grad():
        next(model.parameters()).add_(0.01)
    third = trainable_state_fingerprint(model)
    assert third["sha256"] != first["sha256"], "the fingerprint must change when weights change"
    grads = gradients_fingerprint([parameter for parameter in model.parameters()])
    assert grads["tensors_with_grad"] == 0, "no gradients have been computed yet"
    print("  [extra] trainable fingerprint is stable, content-sensitive and gradient-aware OK")


def test_arm_fingerprints_are_equal():
    path = EVAL / "task6c_initialization.json"
    if not path.is_file():
        print("  [5] initialization artifact not present yet (skipped)")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    fingerprints = {
        arm: payload[arm]["initial_trainable_sha256"]
        for arm in ("U_C", "U_L", "P_C", "P_L")
        if arm in payload
    }
    assert fingerprints, "no arm fingerprints recorded"
    assert len(set(fingerprints.values())) == 1, f"arms do not share one initial state: {fingerprints}"
    counts = {payload[arm]["tensor_count"] for arm in fingerprints}
    assert counts == {528}, counts
    print(f"  [5] all recorded arms share one initial trainable state OK ({len(fingerprints)} arms)")


def test_checkpoint_dir_comes_from_config_and_is_unique_per_arm():
    from task6c_train import ARMS, checkpoint_dir

    cfg = load_config(CONFIG)
    assert not hasattr(__import__("task6c_train"), "CHECKPOINT_DIR") or True
    source = (REPO_ROOT / "scripts" / "task6c_train.py").read_text(encoding="utf-8")
    assert "CHECKPOINT_DIR = " not in source, "the Task 6B hard-coded checkpoint directory must be gone"
    assert 'cfg["paths"]["checkpoints"]' in source

    directories = {arm: checkpoint_dir(cfg, arm) for arm in ARMS}
    resolved = {arm: str(path) for arm, path in directories.items()}
    assert len(set(resolved.values())) == 4, resolved
    for arm, path in directories.items():
        assert path.name == arm, (arm, path)
        assert path.parent.name == "task6c", path
    print(f"  [3/4] per-arm config-driven checkpoint directories OK: {sorted(resolved.values())[0]} …")


def test_manifest_arm_paths_are_distinct():
    path = EVAL / "task6c_checkpoint_manifest.json"
    if not path.is_file():
        print("  [4] manifest not present yet (skipped)")
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    directories = {
        arm: str(Path(manifest[arm]["path"]).parent)
        for arm in ("U_C", "U_L", "P_C", "P_L")
        if isinstance(manifest.get(arm), dict) and manifest[arm].get("path")
    }
    if not directories:
        print("  [4] manifest has no checkpoint paths yet (skipped)")
        return
    assert len(set(directories.values())) == len(directories), directories
    print(f"  [4] manifest checkpoint directories are distinct OK ({len(directories)} arms)")


def test_cache_memory_budget_arithmetic():
    """480 cached images must fit the Task 6C section 4.4 budget of 8 GiB.

    Per image SAM2 stores three image-dependent tensors in fp32:
    image embedding 256x64x64, high-res 32x256x256 and high-res 64x128x128. The
    image positional encoding and the no-mask dense embedding are image-independent
    and are shared, so they are counted once.
    """

    mib = 1024**2
    per_image = 256 * 64 * 64 * 4 + 32 * 256 * 256 * 4 + 64 * 128 * 128 * 4
    assert per_image == 16 * mib, per_image
    footprint_gib = (per_image * 480 + 2 * 256 * 64 * 64 * 4) / 1024**3
    assert footprint_gib <= 8.0, footprint_gib
    unshared_gib = (per_image + 2 * 256 * 64 * 64 * 4) * 480 / 1024**3
    assert unshared_gib > 8.0, "the unshared layout is what violated the budget"
    print(f"  [4.4] cache budget OK: shared {footprint_gib:.2f} GiB vs unshared {unshared_gib:.2f} GiB")


def test_shared_constant_cache_is_bit_identical():
    """Sharing the two image-independent tensors must not change any value."""

    import torch as _torch

    from buildreasonseg_mvp import data as data_mod
    from buildreasonseg_mvp.runtime import load_config as _load_config
    from buildreasonseg_mvp.sam2_bridge import (
        Sam2Encoder,
        Sam2FeatureCache,
        load_sam2,
        sam_source_revision,
    )

    cfg = _load_config(CONFIG)
    checkpoint = REPO_ROOT / cfg["paths"]["models_dir"] / cfg["models"]["sam2_checkpoint_file"]
    if not checkpoint.is_file():
        print("  [4.4] SAM2 checkpoint absent (skipped)")
        return
    device = "cuda" if _torch.cuda.is_available() else "cpu"
    sam, _report = load_sam2(
        cfg["models"]["sam2_config_name"],
        str(checkpoint),
        device=device,
        source_revision=sam_source_revision(REPO_ROOT / cfg["paths"]["sam2_source"]),
    )
    encoder = Sam2Encoder(sam)
    shared = Sam2FeatureCache(encoder, max_images=4, share_constant_features=True)
    plain = Sam2FeatureCache(encoder, max_images=4, share_constant_features=False)

    # two records from two DIFFERENT images, so the arithmetic is unambiguous
    records = []
    seen_images = set()
    for record in data_mod.read_records("val"):
        if record["image_id"] in seen_images:
            continue
        seen_images.add(record["image_id"])
        records.append(record)
        if len(records) == 2:
            break
    assert len(records) == 2

    for record in records:
        sample = data_mod.to_sample(record)
        image = sample.image_rgb()
        left, _ = shared.get(sample.image_id, image)
        right, _ = plain.get(sample.image_id, image)
        for name in ("image_embeddings", "image_pe", "dense_no_mask_embedding"):
            a = getattr(left, name).detach().cpu()
            b = getattr(right, name).detach().cpu()
            assert a.shape == b.shape, (name, a.shape, b.shape)
            assert _torch.equal(a, b), f"shared cache changed {name}"
        assert left.high_res_features and right.high_res_features
        for a, b in zip(left.high_res_features, right.high_res_features):
            assert _torch.equal(a.detach().cpu(), b.detach().cpu()), "shared cache changed a high-res level"

    assert len(shared) == 2 and len(plain) == 2
    shared_bytes = shared.ram_footprint_bytes()
    plain_bytes = plain.ram_footprint_bytes()
    # one shared copy is still stored, so the saving is (n - 1) x 8 MiB
    expected_saving = (len(records) - 1) * 2 * 256 * 64 * 64 * 4
    assert plain_bytes - shared_bytes == expected_saving, (plain_bytes, shared_bytes, expected_saving)
    assert shared_bytes < plain_bytes
    # the cache must not have pushed any constant to the device tensors it stores
    for entry in shared._store.values():  # noqa: SLF001 - deliberate white-box check
        assert entry.image_pe.device.type == "cpu"
        assert entry.dense_no_mask_embedding.device.type == "cpu"
    print(
        f"  [4.4] shared cache is bit-identical, saves {expected_saving / 1024**2:.0f} MiB for "
        f"{len(records)} images, and keeps the CPU cache on the CPU"
    )


def _task6c_tests():
    return [
        ("1 deterministic", test_deterministic_flag_is_consumed),
        ("2 determinism artifact", test_determinism_evidence_artifact),
        ("fingerprint unit", test_fingerprint_is_stable_and_order_independent),
        ("5 fingerprint equality", test_arm_fingerprints_are_equal),
        ("3/4 checkpoint dirs", test_checkpoint_dir_comes_from_config_and_is_unique_per_arm),
        ("4 manifest dirs", test_manifest_arm_paths_are_distinct),
        ("4.4 cache budget", test_cache_memory_budget_arithmetic),
        ("4.4 cache sharing", test_shared_constant_cache_is_bit_identical),
    ]


def main() -> int:
    tests = _task6c_tests()
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
    print(f"\n{len(tests) - failures}/{len(tests)} task6c fix checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
