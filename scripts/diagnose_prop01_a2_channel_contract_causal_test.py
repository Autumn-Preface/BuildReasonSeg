"""Supervisor-authorized paired A2 RGB/BGR test, with no product/source edits.

Use established .conda/buildreasonseg-mvp/python.exe -B:
--phase prepare: verify identities, persist experiment design; zero inference.
--phase run: exactly one paired experiment, 9 baseline + 9 corrected predicts.
--phase finalize: validate saved evidence; zero inference.
Only the new task JSON and disposable temporary library settings may be written.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile

# Read-only reuse of the accepted forensic milestone's identity/integrity helpers.
import diagnose_prop01_a2_zero_proposals as prior

ROOT = prior.ROOT
EXTERNAL = prior.EXTERNAL
START = "59296d25195a656e6a75e303e3cbb43fa9df8b8b"
BRANCH = "audit/task8b3-prop01-a2-channel-contract-causal-test-v1"
TASK = "PROP01_A2_CHANNEL_CONTRACT_CAUSAL_TEST_V1"
EVIDENCE = ROOT / "evaluation/task8b3_prop01_a2_channel_contract_causal_test_v1.json"
PARAMS = dict(imgsz=640, conf=0.05, max_det=300, verbose=False, retina_masks=False, device="cuda")
ALLOWED = {"handoff/CURRENT_TASK.md", "handoff/EXECUTOR_STATE.yaml",
           "docs/task8b3_prop01_a2_channel_contract_causal_test_v1.md",
           "evaluation/task8b3_prop01_a2_channel_contract_causal_test_v1.json",
           "scripts/diagnose_prop01_a2_channel_contract_causal_test.py"}


def write(data):
    EVIDENCE.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def inventory_hash(inventory):
    return prior.digest(json.dumps(inventory, sort_keys=True).encode())


class ExperimentGuard(Exception):
    def __init__(self, classification, detail):
        self.classification = classification
        self.detail = detail


def prepare(before, protected):
    import numpy as np
    from PIL import Image
    from buildreasonseg.runtime import detector as d
    from buildreasonseg.runtime.imageio import load_image

    assert prior.git("branch", "--show-current").decode().strip() == BRANCH
    assert prior.git("merge-base", START, "HEAD").decode().strip() == START
    accepted = prior.read_json(prior.EVIDENCE)
    assert accepted["status"] == "READY_FOR_SUPERVISOR_AUDIT"
    assert accepted["baseline_reproduction"] == "REPRODUCED"
    assert protected == accepted["external_integrity"]["protected_content_hashes"]
    assert all(getattr(d, name) == value for name, value in prior.POLICY.items())
    assert accepted["frozen_parameters"] == prior.POLICY
    libraries = {name: importlib.metadata.version(name) for name in accepted["environment"]["libraries"]}
    assert libraries == accepted["environment"]["libraries"]
    for source in accepted["library_input_contract"].values():
        assert prior.file_hash(source["path"]) == source["sha256"]
    image = Path(accepted["A2_provenance"]["image_path"])
    assert image == EXTERNAL / "inference/input/A2.png"
    assert prior.file_hash(image) == prior.IMAGE_HASHES["A2"]
    loaded = load_image(image)
    with Image.open(image) as pil:
        assert pil.mode == "RGB" and np.array_equal(loaded.rgb, np.asarray(pil))
    assert loaded.rgb.shape == (1024, 1024, 3) and loaded.rgb.dtype == np.uint8
    planned = []
    for window, recorded in zip(d.plan_tiles(loaded.width, loaded.height), accepted["per_tile"]):
        rgb, padding = d.extract_tile(loaded.rgb, window)
        assert not padding["applied"]
        sha = prior.digest(rgb.tobytes())
        assert sha == recorded["tile_sha256"]
        assert window.source_tile_id == recorded["tile_id"]
        planned.append(dict(tile_id=window.source_tile_id, top=window.top, left=window.left,
                            tile_rgb_sha256=sha, shape=list(rgb.shape), dtype=str(rgb.dtype), padding=padding))
    assert len(planned) == 9
    return dict(task_id=TASK, status="IN_PROGRESS", stage="PREPARED",
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                starting_branch="audit/task8b3-prop01-a2-zero-proposals-forensics-v1", starting_head=START,
                starting_working_tree="clean", task_branch=BRANCH,
                authorization_attachment_sha256="1b99bc499174ba10db15f5cbadf226576a2e0f4983356b11c5afc261225d4c4a",
                accepted_forensic_evidence=dict(path=str(prior.EVIDENCE), sha256=prior.file_hash(prior.EVIDENCE)),
                A2_identity=accepted["A2_provenance"], detector_identity=accepted["detector_model"],
                frozen_parameters=prior.POLICY, invocation_args=PARAMS,
                environment=dict(executable=sys.executable, python=sys.version, libraries=libraries),
                installed_library_input_contract=accepted["library_input_contract"],
                planned_tiles=planned, per_tile=[], planned_case_predict_calls=18, case_predict_calls=0,
                baseline_total_raw=None, corrected_total_raw=None,
                baseline_tensor_contract="NOT_RUN", corrected_tensor_contract="NOT_RUN",
                primary_classification=None, causal_conclusion=None,
                independent_variable="API-facing NumPy channel representation only: RGB -> contiguous BGR",
                corrected_conversion="np.ascontiguousarray(RGB_tile[..., ::-1])",
                threshold_sweep_run=False, parameter_tuning_run=False, product_source_changed=False,
                external_write=False, product_repair_run=False, full_pipeline_run=False,
                positive_control_run=False, external_sync_run=False,
                next_gate="CHATGPT_PROP01_CHANNEL_CAUSAL_TEST_REMOTE_AUDIT",
                external_integrity=dict(inventory_before_sha256=inventory_hash(before),
                                        inventory_file_count=len(before), protected_hash_count=len(protected),
                                        protected_content_hashes=protected))


def experiment(data):
    import numpy as np
    import torch
    from buildreasonseg.runtime import detector as d
    from buildreasonseg.runtime.imageio import load_image
    from ultralytics.engine.predictor import BasePredictor

    if not torch.cuda.is_available():
        raise RuntimeError("Historical CUDA device unavailable: no device substitution is authorized")
    loaded = load_image(data["A2_identity"]["image_path"])
    accepted = prior.read_json(prior.EVIDENCE)
    data["environment"].update(device="cuda", cuda=torch.version.cuda,
                                cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0))
    assert all(data["environment"][key] == accepted["environment"][key]
               for key in ("device", "cuda", "cudnn", "gpu"))
    runtime = d.DetectorRuntime(checkpoint=Path(data["detector_identity"]["path"]), device="cuda")
    model = runtime.load()
    data["detector_identity"]["model_task"] = model.task
    assert model.task == "segment"
    original = BasePredictor.preprocess
    current = {}

    def observer(predictor, images):
        tensor = original(predictor, images)
        expected_array = np.stack(predictor.pre_transform([current["rgb"]]))
        intended = torch.from_numpy(np.ascontiguousarray(expected_array.transpose(0, 3, 1, 2)))
        intended = intended.to(tensor.device).to(tensor.dtype) / 255
        reversed_rgb = intended.flip(1).contiguous()
        actual = tensor.detach().cpu().contiguous().numpy()
        record = dict(network_tensor_sha256=prior.digest(actual.tobytes()), shape=list(actual.shape),
                      dtype=str(actual.dtype), device=str(tensor.device),
                      intended_rgb_tensor_sha256=prior.digest(intended.cpu().contiguous().numpy().tobytes()),
                      reversed_rgb_tensor_sha256=prior.digest(reversed_rgb.cpu().contiguous().numpy().tobytes()),
                      equals_intended_rgb=bool(torch.equal(tensor, intended)),
                      equals_reversed_rgb=bool(torch.equal(tensor, reversed_rgb)))
        current["result"]["tensor"] = record
        # Assertions happen before returning the original tensor to inference.
        expected = reversed_rgb if current["arm"] == "baseline" else intended
        distinct = not bool(torch.equal(intended, reversed_rgb))
        valid = bool(torch.equal(tensor, expected)) and distinct
        current["result"]["tensor_contract_valid"] = valid
        if not valid:
            raise ExperimentGuard("C4_COUNTERFACTUAL_INVALID", f"{current['arm']} tensor contract failed")
        return tensor

    def predict(rgb, source, arm):
        current.clear()
        result = dict(api_source_sha256=prior.digest(source.tobytes()), api_shape=list(source.shape),
                      api_dtype=str(source.dtype), invocation_args=dict(PARAMS),
                      raw_box_count=None, masks_present=None, confidences=None)
        current.update(rgb=rgb, arm=arm, result=result)
        data["case_predict_calls"] += 1
        with torch.no_grad():
            results = model.predict(source=source, **PARAMS)
        actual = results[0] if results else None
        boxes = actual.boxes if actual is not None else None
        result.update(results_count=len(results), raw_box_count=len(boxes) if boxes is not None else 0,
                      masks_present=bool(actual is not None and actual.masks is not None),
                      confidences=boxes.conf.detach().cpu().tolist() if boxes is not None else [],
                      boxes_xyxy=boxes.xyxy.detach().cpu().tolist() if boxes is not None else [],
                      mask_count=int(actual.masks.data.shape[0]) if actual is not None and actual.masks is not None else 0)
        relevant = {key: vars(model.predictor.args).get(key) for key in accepted["effective_library_args"]}
        assert relevant == accepted["effective_library_args"], "Frozen library arguments changed"
        result["effective_library_args"] = relevant
        result["backend_fp16"] = bool(model.predictor.model.fp16)
        assert not result["backend_fp16"] and result["tensor"]["dtype"] == "float32"
        return result

    BasePredictor.preprocess = observer
    try:
        for window, plan in zip(d.plan_tiles(loaded.width, loaded.height), data["planned_tiles"]):
            rgb, padding = d.extract_tile(loaded.rgb, window)
            assert prior.digest(rgb.tobytes()) == plan["tile_rgb_sha256"] and padding == plan["padding"]
            bgr = np.ascontiguousarray(rgb[..., ::-1])
            assert bgr.dtype == rgb.dtype and bgr.shape == rgb.shape and bgr.flags.c_contiguous
            assert np.array_equal(bgr[..., ::-1], rgb)
            row = dict(plan)
            data["per_tile"].append(row)
            try:
                row["baseline"] = predict(rgb, rgb, "baseline")
                if row["baseline"]["raw_box_count"] != 0:
                    raise ExperimentGuard("C3_BASELINE_NOT_REPRODUCIBLE", "Baseline tile returned nonzero boxes")
                predictor = model.predictor
                backend = predictor.model
                row["corrected"] = predict(rgb, bgr, "corrected")
                assert model.predictor is predictor and model.predictor.model is backend
                assert row["baseline"]["invocation_args"] == row["corrected"]["invocation_args"]
                assert row["baseline"]["effective_library_args"] == row["corrected"]["effective_library_args"]
                assert row["baseline"]["tensor"]["intended_rgb_tensor_sha256"] == row["corrected"]["tensor"]["intended_rgb_tensor_sha256"]
                assert row["baseline"]["tensor"]["reversed_rgb_tensor_sha256"] == row["corrected"]["tensor"]["reversed_rgb_tensor_sha256"]
                row["same_model_predictor_and_backend"] = True
                print(f"{row['tile_id']}: baseline={row['baseline']['raw_box_count']} corrected={row['corrected']['raw_box_count']}", flush=True)
                write(data)  # Persist each completed pair; partial results survive interruption.
            except ExperimentGuard:
                row.setdefault(current["arm"], current["result"])
                raise
    finally:
        BasePredictor.preprocess = original
    assert len(data["per_tile"]) == 9 and data["case_predict_calls"] == 18
    data["baseline_total_raw"] = sum(row["baseline"]["raw_box_count"] for row in data["per_tile"])
    data["corrected_total_raw"] = sum(row["corrected"]["raw_box_count"] for row in data["per_tile"])
    data["baseline_tensor_contract"] = "CONFIRMED_REVERSED_SOURCE_RGB"
    data["corrected_tensor_contract"] = "CONFIRMED_INTENDED_SOURCE_RGB"
    if data["corrected_total_raw"] > 0:
        data["primary_classification"] = "C1_CHANNEL_CORRECTION_RECOVERS_PROPOSALS"
        data["causal_conclusion"] = "Channel-contract defect has experimentally established causal contribution to A2 zero-proposal failure under the frozen setup; sole scientific root cause and full pipeline correctness are not established."
    else:
        data["primary_classification"] = "C2_CHANNEL_CORRECTION_STILL_ZERO"
        data["causal_conclusion"] = "Channel-contract defect is real, but correcting it is not sufficient to recover A2 proposals; valid corrected A2 detector tensors still yield zero raw proposals under the frozen setup."
    data["stage"] = "PAIRED_EXPERIMENT_COMPLETE"


def validate_saved(data):
    assert len(data["per_tile"]) == 9 and data["case_predict_calls"] == 18
    assert data["baseline_total_raw"] == 0
    assert data["baseline_tensor_contract"] == "CONFIRMED_REVERSED_SOURCE_RGB"
    assert data["corrected_tensor_contract"] == "CONFIRMED_INTENDED_SOURCE_RGB"
    for row in data["per_tile"]:
        baseline, corrected = row["baseline"], row["corrected"]
        assert row["same_model_predictor_and_backend"]
        assert baseline["tensor_contract_valid"] and corrected["tensor_contract_valid"]
        assert baseline["raw_box_count"] == 0
        assert baseline["tensor"]["network_tensor_sha256"] == baseline["tensor"]["reversed_rgb_tensor_sha256"]
        assert corrected["tensor"]["network_tensor_sha256"] == corrected["tensor"]["intended_rgb_tensor_sha256"]
        assert baseline["invocation_args"] == corrected["invocation_args"] == PARAMS
        assert baseline["effective_library_args"] == corrected["effective_library_args"]
        assert len(corrected["confidences"]) == corrected["raw_box_count"]
    assert data["corrected_total_raw"] == sum(row["corrected"]["raw_box_count"] for row in data["per_tile"])
    expected = "C1_CHANNEL_CORRECTION_RECOVERS_PROPOSALS" if data["corrected_total_raw"] > 0 else "C2_CHANNEL_CORRECTION_STILL_ZERO"
    assert data["primary_classification"] == expected
    assert prior.protected_hashes() == data["external_integrity"]["protected_content_hashes"]
    assert inventory_hash(prior.snapshot()) == data["external_integrity"]["inventory_after_sha256"]
    assert set(prior.git("diff", START, "--name-only").decode().splitlines()) <= ALLOWED


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "run", "finalize"), required=True)
    phase = parser.parse_args().phase
    sys.dont_write_bytecode = True
    if phase == "finalize":
        data = prior.read_json(EVIDENCE)
        validate_saved(data)
        data.update(status="READY_FOR_SUPERVISOR_AUDIT", stage="FINAL_EVIDENCE",
                    repair_candidate_only="API-compatible RGB→BGR conversion before passing NumPy tile to Ultralytics",
                    validation=dict(saved_evidence_assertions="PASS", external_integrity="UNCHANGED",
                                    scope="FIVE_AUTHORIZED_PATHS_ONLY", finalize_inference_calls=0))
        write(data)
        print(json.dumps(dict(status=data["status"], classification=data["primary_classification"],
                              baseline=data["baseline_total_raw"], corrected=data["corrected_total_raw"])))
        return
    before = prior.snapshot()
    protected = prior.protected_hashes()
    with tempfile.TemporaryDirectory(prefix="prop01_channel_causal_") as temporary:
        os.environ.update(YOLO_CONFIG_DIR=temporary, MPLCONFIGDIR=temporary,
                          PYTHONDONTWRITEBYTECODE="1", YOLO_AUTOINSTALL="false")
        sys.path.insert(0, str(EXTERNAL))
        data = prepare(before, protected)
        write(data)
        try:
            if phase == "run":
                experiment(data)
        except ExperimentGuard as guard:
            data.update(status="STOPPED", stage="GUARD_STOP", primary_classification=guard.classification,
                        causal_conclusion="Counterfactual interpretation prohibited: experiment guard failed.",
                        stop_reason=guard.detail,
                        guard_condition="BASELINE_NOT_REPRODUCIBLE" if guard.classification.startswith("C3") else "INPUT_CONTRACT_COUNTERFACTUAL_INVALID")
            raise
        except Exception as error:
            data.update(status="STOPPED", stage="ERROR_STOP", stop_reason=f"{type(error).__name__}: {error}")
            raise
        finally:
            after = prior.snapshot()
            after_protected = prior.protected_hashes()
            changed = [key for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)]
            hash_changes = [key for key in protected if protected[key] != after_protected[key]]
            data["external_integrity"].update(inventory_after_sha256=inventory_hash(after),
                                              changed_inventory_paths=changed, changed_protected_hash_paths=hash_changes)
            if changed or hash_changes:
                data.update(status="STOPPED", stage="EXTERNAL_INTEGRITY_STOP", external_write=True,
                            stop_reason="External integrity discrepancy requires investigation; no repair authorized.")
            write(data)
            assert not changed and not hash_changes, "External integrity discrepancy: STOP"
    print(json.dumps(dict(stage=data["stage"], baseline=data["baseline_total_raw"],
                          corrected=data["corrected_total_raw"], calls=data["case_predict_calls"])))


if __name__ == "__main__":
    main()
