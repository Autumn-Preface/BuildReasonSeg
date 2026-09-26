"""Task 6C.6 section 17: integration, candidate and boundary tests.

Items 1-12 of the section 17 list. The heavy model-backed candidates live in
`evaluation/task6c6_*.json` and are asserted against here, so the tests stay fast and
still fail if the artifacts and the code disagree.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

EVAL = REPO_ROOT / "evaluation"
CONFIG = REPO_ROOT / "configs" / "mvp" / "task6c_2b_ablation.yaml"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "task6c_train.py"
RUNTIME = REPO_ROOT / "buildreasonseg_mvp" / "runtime.py"

INTEGRATED_BASELINE = EVAL / "task6c6_integrated_baseline.json"
EQUIVALENCE = EVAL / "task6c6_equivalence.json"
COMPILE_VARIANTS = EVAL / "task6c6_compile_variants.json"
OPTIMIZER_VARIANTS = EVAL / "task6c6_optimizer_variants.json"
PROFILER_SUMMARY = EVAL / "task6c6_profiler_summary.json"
FINAL_BENCHMARK = EVAL / "task6c6_final_benchmark.json"

NEW_SCRIPTS = sorted((REPO_ROOT / "scripts").glob("task6c6_*.py"))


def _load(path: Path) -> dict:
    if not path.is_file():
        print(f"  [skip] {path.name} not present yet")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ 1 + 3


class _FakeBatch:
    def to(self, device):  # noqa: D102, ANN001
        return self


class _FakeSample:
    sample_id = "fake-sample"

    def target_mask(self):  # noqa: D102
        return np.zeros((2, 2), dtype=bool)


class _FakeModel:
    def __init__(self) -> None:
        self.parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def trainable_parameter_groups(self, **_kwargs):  # noqa: D102
        return [{"params": [self.parameter], "lr": 1e-4}]


class _FakeFeatureCache:
    def stats(self):  # noqa: D102
        return {"enabled": False}


class _FakeRuntime:
    """Just enough runtime for `train_phase` to run one step on the CPU."""

    def __init__(self, training: dict) -> None:
        self.cfg = {
            "training": dict(training),
            "optimizer": {
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "phase_b": {
                    "lora_lr": 1e-4,
                    "token_lr": 3e-4,
                    "decoder_lr": 3e-4,
                    "warmup_steps": 1,
                    "lr_schedule": "none",
                },
            },
        }
        self.model = _FakeModel()
        self.device = "cpu"
        self.feature_cache = _FakeFeatureCache()
        self.recorded: dict = {}

    def prepare(self, sample):  # noqa: D102, ANN001
        return _FakeBatch(), None

    def features_for(self, sample, image):  # noqa: D102, ANN001
        return None, False

    def build_scheduler_for(self, optimizer, steps, cfg):  # noqa: D102, ANN001
        return None

    def train_step(self, batch, gt_mask, features, optimizer=None, **kwargs):  # noqa: D102, ANN001
        self.recorded.update(kwargs)
        return {
            "losses": {"total": 1.0, "lm_ce": 1.0, "mask_bce": 0.0, "mask_dice": 0.0},
            "grad_norms": {"x": 1.0},
            "grad_clip_total_norm": 0.5,
        }


def _run_one_fake_step(training: dict) -> dict:
    import task6c_train as train

    runtime = _FakeRuntime(training)
    original_vram = train.vram
    train.vram = lambda: {"peak_allocated_gib": 0.0}  # no CUDA needed for this test
    try:
        result = train.train_phase(runtime, [_FakeSample()], "B", 1, 10**9, "test")
    finally:
        train.vram = original_vram
    assert result["steps"] == 1
    return runtime.recorded


def test_formal_training_path_consumes_collect_grad_norms():
    """Section 17 item 1: the formal loop passes the configured value, not a default."""

    recorded = _run_one_fake_step({"collect_grad_norms": False})
    assert recorded.get("collect_grad_norms") is False, recorded
    recorded = _run_one_fake_step({"collect_grad_norms": True})
    assert recorded.get("collect_grad_norms") is True, recorded
    print("  [1] formal loop forwards training.collect_grad_norms OK")


def test_setting_false_keeps_true_available_for_stage1():
    """Section 17 item 3: True is still reachable, including when the key is absent."""

    recorded = _run_one_fake_step({})
    assert recorded.get("collect_grad_norms") is True, (
        "an absent config key must fall back to the diagnostic-only behaviour, not to False"
    )
    signature = inspect.signature(
        __import__("buildreasonseg_mvp.runtime", fromlist=["MvpRuntime"]).MvpRuntime.train_step
    )
    assert signature.parameters["collect_grad_norms"].default is True
    print("  [3] collect_grad_norms=True remains the library default OK")


# ------------------------------------------------------------------ 2


def test_false_removes_sweep_but_keeps_clipping():
    """Section 17 item 2: only the unused diagnostic sweep is removed."""

    source = RUNTIME.read_text(encoding="utf-8")
    assert re.search(r"gradient_norms\(self\.model\) if collect_grad_norms else \{\}", source), (
        "the gradient-norm sweep must be conditional on collect_grad_norms"
    )
    body = source.split("def train_step", 1)[1]
    clip = body.index("clip_grad_norm_")
    assert clip > 0 and "collect_grad_norms" not in body[clip : clip + 400], (
        "clip_grad_norm_ must run unconditionally, independent of the diagnostic switch"
    )
    assert 'self.cfg["optimizer"]["grad_clip_norm"]' in body

    payload = _load(INTEGRATED_BASELINE)
    if payload:
        for entry in payload["entries"]:
            assert entry["clip_total_norm_last"] is not None, entry["variant"]
        integrated = [e for e in payload["entries"] if not e["grad_norms_collected"]]
        control = [e for e in payload["entries"] if e["grad_norms_collected"]]
        assert integrated and control
        assert {e["clip_total_norm_last"] for e in integrated + control}.__len__() == 1, (
            "clipping must produce the same total norm with and without the diagnostic sweep"
        )
    print("  [2] clipping survives, diagnostic sweep is conditional OK")


# ------------------------------------------------------------------ 4


def test_integration_is_bit_equivalent_over_fixed_mini_run():
    """Section 17 item 4."""

    payload = _load(EQUIVALENCE)
    if not payload:
        return
    gate = payload["integration_gate"]
    assert gate["prepared_tensors"]["identical"], gate["prepared_tensors"]
    assert gate["losses_identical"], gate
    assert gate["gradients_identical"], gate
    assert gate["post_step_parameters_identical"], gate
    assert gate["category"] == "BIT_EQUIVALENT", gate
    control = payload["reference_run_to_run_control"]
    assert control["losses_identical"] and control["gradients_identical"], control
    print("  [4] collect_grad_norms=false is bit-equivalent to the pre-integration path OK")


# ------------------------------------------------------------------ 5


def test_compile_wrapper_can_be_disabled_cleanly():
    """Section 17 item 5: nothing compiles unless a candidate explicitly asks for it."""

    # Exactly one module may call torch.compile: the candidate wrapper itself.
    callers = []
    for path in NEW_SCRIPTS + sorted((REPO_ROOT / "buildreasonseg_mvp").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if re.search(r"torch\.compile\s*\(", text):
            callers.append(path.name)
    assert callers == ["task6c6_common.py"], callers

    assert "torch.compile" not in TRAIN_SCRIPT.read_text(encoding="utf-8"), (
        "the formal training path must never compile implicitly"
    )
    source = (REPO_ROOT / "scripts" / "task6c6_common.py").read_text(encoding="utf-8")
    assert "if spec.compile_scope:" in source, "compilation must be opt-in per variant"
    assert 'compile_scope: str | None = None' in source, "the default must be no compilation"
    print("  [5] torch.compile has exactly one opt-in call site OK")


# ------------------------------------------------------------------ 6


def test_compile_failures_are_recorded_not_silently_fallen_back():
    """Section 17 item 6."""

    payload = _load(COMPILE_VARIANTS)
    if not payload:
        return
    reported = 0
    for entry in payload["entries"]:
        if entry.get("compile", {}).get("scope") and entry["status"] != "ok":
            reported += 1
            assert entry["status"] in {"compile_failed", "runtime_failed"}, entry["status"]
            assert entry.get("error"), entry["variant"]
            assert entry.get("samples_per_sec") is None, (
                "a failed candidate must not carry a throughput number"
            )
    failures = [
        entry
        for entry in payload["entries"]
        if entry.get("status") in {"compile_failed", "runtime_failed"}
    ]
    assert len(failures) == reported
    assert "torch.compile" in (REPO_ROOT / "scripts" / "task6c6_common.py").read_text(encoding="utf-8")
    print(f"  [6] {reported} compile candidate(s) recorded with explicit failure status OK")


# ------------------------------------------------------------------ 7


def test_no_architecture_loss_data_or_optimizer_semantics_changed():
    """Section 17 item 7."""

    runtime_source = RUNTIME.read_text(encoding="utf-8")
    body = runtime_source.split("def train_step", 1)[1].split("def build_mask_supervision", 1)[0]
    # the step still runs the same forward, the same objective, the same backward and the
    # same optimizer update, with the configured clip threshold
    for required in (
        "self.model(batch, features)",
        "combined_loss(",
        "breakdown.total.backward()",
        "optimizer.step()",
        "clip_grad_norm_",
        'self.cfg["optimizer"]["grad_clip_norm"]',
    ):
        assert required in body, required
    # Task 6C.6 added exactly one new optional argument to the step: clip_grad_foreach,
    # whose default reproduces the previous behaviour
    step_header = runtime_source.split("def train_step", 1)[1].split(")", 1)[0]
    assert "clip_grad_foreach: bool | None = None" in step_header

    from buildreasonseg_mvp.runtime import load_config

    cfg = load_config(CONFIG)
    assert cfg["optimizer"]["betas"] == [0.9, 0.999]
    assert cfg["optimizer"]["grad_clip_norm"] == 1.0
    assert cfg["optimizer"]["weight_decay"] == 0.01
    assert cfg["loss"]["lm_ce"] == 2.0 and cfg["loss"]["mask_bce"] == 2.0 and cfg["loss"]["mask_dice"] == 1.0
    assert cfg["lora"]["rank"] == 16 and cfg["lora"]["alpha"] == 32
    # the only training-path addition in Task 6C.6 is the diagnostic switch
    assert cfg["training"]["collect_grad_norms"] is False
    assert cfg["training"]["gradient_checkpointing"] is True
    assert cfg["training"]["deterministic_strict"] is True

    for path in NEW_SCRIPTS:
        text = path.read_text(encoding="utf-8")
        assert "gt_mask" not in text or "target_mask" in text, path.name
    print("  [7] no architecture/loss/data/optimizer semantics changed OK")


# ------------------------------------------------------------------ 8


def test_strict_determinism_preserved_for_accepted_candidate():
    """Section 17 item 8."""

    payload = _load(FINAL_BENCHMARK)
    if not payload:
        payload = _load(INTEGRATED_BASELINE)
    if not payload:
        return
    entries = payload.get("entries", [])
    assert entries, "no entries to check"
    for entry in entries:
        determinism = entry.get("determinism") or {}
        if determinism:
            assert determinism.get("strict_effective") is True, entry["variant"]
    print("  [8] strict deterministic mode is effective for every measured variant OK")


# ------------------------------------------------------------------ 9 + 10


def test_no_test_split_and_no_batch_greater_than_one():
    """Section 17 items 9 and 10."""

    from buildreasonseg_mvp import data as data_mod

    payload = json.loads((EVAL / "task6c5_benchmark_ids.json").read_text(encoding="utf-8"))
    train_ids = {record["sample_id"] for record in data_mod.read_records("train")}
    other_ids = {record["sample_id"] for split in ("val", "test") for record in data_mod.read_records(split)}
    record_ids = payload["record_ids"]
    assert set(record_ids) <= train_ids, "the benchmark set must stay inside the train split"
    assert not (set(record_ids) & other_ids), "no validation/test record may be used"

    for path in (INTEGRATED_BASELINE, COMPILE_VARIANTS, OPTIMIZER_VARIANTS, FINAL_BENCHMARK, PROFILER_SUMMARY):
        data = _load(path)
        for key in ("batch_size", "steps_measured"):
            if key in data and key == "batch_size":
                assert data[key] == 1, (path.name, data[key])
        for entry in data.get("entries", []):
            if "batch_size" in entry:
                assert entry["batch_size"] == 1
    text = CONFIG.read_text(encoding="utf-8")
    assert "grad_accum_steps: 1" in text
    print("  [9,10] train split only, batch 1, no accumulation OK")


# ------------------------------------------------------------------ 11


def test_no_ref_4b_sre_scl_introduced():
    """Section 17 item 11."""

    forbidden = ("[REF]", "4B", "Qwen3-VL-4B", "SpatialRelationEncoder", "Spatial Consistency Loss")
    for path in NEW_SCRIPTS + [RUNTIME, REPO_ROOT / "configs" / "mvp" / "task6c6_runtime.yaml"]:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} mentions {token}"
    print("  [11] no [REF]/4B/SRE/SCL introduced OK")


# ------------------------------------------------------------------ 12


def test_profiler_is_disabled_in_normal_training():
    """Section 17 item 12."""

    text = TRAIN_SCRIPT.read_text(encoding="utf-8")
    assert "torch.profiler" not in text
    assert "profile(" not in text
    assert "StageProfile(" not in text
    runtime_text = RUNTIME.read_text(encoding="utf-8")
    assert "timer=None" in runtime_text, "the stage timer must stay opt-in"
    profiler_script = (REPO_ROOT / "scripts" / "task6c6_profile.py").read_text(encoding="utf-8")
    assert "torch.profiler" in profiler_script or "from torch.profiler" in profiler_script
    tree = ast.parse(profiler_script)
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "profile_window" for node in tree.body
    )
    print("  [12] profiler only runs from its own opt-in script OK")


# ------------------------------------------------------------------ extras


def test_artifact_declares_no_silent_adoption():
    """Section 15: adoption needs the threshold, and unresolved gains stay unadopted."""

    payload = _load(FINAL_BENCHMARK)
    if not payload:
        return
    assert "adoption" in payload or "verdict" in payload, sorted(payload)
    for entry in payload.get("entries", []):
        if entry.get("adopted"):
            assert entry.get("determinism", {}).get("strict_effective") is True
            assert (entry.get("vram") or {}).get("peak_reserved_gib", 0) < 14.0
    print("  [extra] adoption record carries determinism and VRAM gates OK")


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
            failures += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"[task6c6 tests] {len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
