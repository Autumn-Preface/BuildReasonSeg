"""Frozen PROP-01 forensics; observes existing RC1 functions without editing them.

Run with the established .conda/buildreasonseg-mvp/python.exe and -B.
--phase provenance persists identities without loading a model.
--phase inference performs one detect_global call each on locked A2 and A1.
All library writes/caches are redirected to a disposable temporary directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path(r"C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1")
CANONICAL = "delivery_src/BuildReasonSeg_Advisor_RC1"
EVIDENCE = ROOT / "evaluation/task8b3_prop01_a2_zero_proposals_forensics_v1.json"
TASK = "PROP01_A2_ZERO_PROPOSALS_ROOT_CAUSE_FORENSICS_V1"
START = "504268128b7a9b058580b7702c9429b27705c053"
BRANCH = "audit/task8b3-prop01-a2-zero-proposals-forensics-v1"
IMAGE_HASHES = {
    "A2": "10286b1e76db9e38c474635a465c9e677dbcf58375c1d39f7b742eeb991f434f",
    "A1": "8a4b459d65773a7dfb0ffcc509c26b5d3a7cea23cd759cdadf94cd46be84c227",
}
MODEL_HASH = "ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474"
POLICY = dict(TILE_SIZE=512, TILE_OVERLAP=128, TILE_STRIDE=384, IMGSZ=640,
              CONF=0.05, MAX_DET=300, FROZEN_THRESHOLD=0.5, DUPLICATE_IOU=0.50)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def write(data):
    EVIDENCE.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def snapshot():
    # Whole external tree inventory; content hashes additionally cover all manifest
    # entries, locked inputs/results/logs, detector weights and model configuration.
    return {str(p.relative_to(EXTERNAL)).replace("\\", "/"):
            [p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(EXTERNAL.rglob("*")) if p.is_file()}


def protected_hashes():
    manifest = read_json(ROOT / CANONICAL / "source_manifest.json")
    relatives = {entry["path"] for entry in manifest["files"]}
    relatives.update({"source_manifest.json", "model/buildreasonseg_advisor/detector.pt",
                      "model/buildreasonseg_advisor/model.yaml", "logs/task8b3_suite_results.json",
                      "logs/task8b3_p1d2_probe.json", "logs/task8b3_transcripts/A2.txt"})
    for case in IMAGE_HASHES:
        relatives.add(f"inference/input/{case}.png")
        relatives.add(f"inference/output/diagnostics/{case}/result.json")
    return {relative: file_hash(EXTERNAL / relative) for relative in sorted(relatives)}


def provenance():
    import numpy as np
    from PIL import Image
    from buildreasonseg.runtime.imageio import load_image
    from buildreasonseg.runtime import detector as d

    assert git("branch", "--show-current").decode().strip() == BRANCH
    assert git("merge-base", START, "HEAD").decode().strip() == START
    assert all(getattr(d, name) == value for name, value in POLICY.items())
    suite_path = EXTERNAL / "logs/task8b3_suite_results.json"
    suite = read_json(suite_path)
    cases = {}
    for case, expected in IMAGE_HASHES.items():
        historical = next(row for row in suite if row["id"] == case)
        lock = next(row for row in historical["inputs"] if row["id"] == case)
        image = EXTERNAL / historical["file"]
        result_path = EXTERNAL / f"inference/output/diagnostics/{case}/result.json"
        result = read_json(result_path)
        assert historical["file"] == f"inference/input/{case}.png"
        assert lock["sha256"] == expected == file_hash(image)
        loaded = load_image(image)
        with Image.open(image) as pil:
            independently_decoded = np.asarray(pil).copy()
            image_mode = pil.mode
        assert np.array_equal(loaded.rgb, independently_decoded)
        assert image_mode == "RGB" and loaded.rgb.dtype == np.uint8
        assert list(loaded.rgb.shape) == [1024, 1024, 3]
        assert image.stat().st_size == lock["bytes"]
        assert result["parsed"]["program"] == historical["expected_program"]
        assert result["raw_proposal_count"] == historical["raw_proposal_count"]
        assert result["merged_proposal_count"] == historical["merged_proposal_count"]
        if case == "A2":
            assert result["error_code"] == "E401" and historical["raw_proposal_count"] == 0
        else:
            assert historical["raw_proposal_count"] > 0
        cases[case] = dict(case_id=case, image_path=str(image), filename=image.name,
                           file_size=image.stat().st_size, sha256=expected,
                           dimensions=[loaded.width, loaded.height], mode=image_mode,
                           dtype=str(loaded.rgb.dtype), decoded_rgb_sha256=digest(loaded.rgb.tobytes()),
                           runtime_decode_equals_independent_Pillow=True,
                           program=historical["expected_program"], relation="left_of" if case == "A2" else "right_of",
                           historical_raw_proposal_count=historical["raw_proposal_count"],
                           historical_merged_proposal_count=historical["merged_proposal_count"],
                           historical_status=result["status"], historical_device=result["device"],
                           result_path=str(result_path), result_sha256=file_hash(result_path),
                           historical_suite_path=str(suite_path), suite_sha256=file_hash(suite_path),
                           suite_locked_input=lock, suite_diagnostics_path=historical["diagnostics_path"])
    checkpoint = EXTERNAL / "model/buildreasonseg_advisor/detector.pt"
    assert file_hash(checkpoint) == MODEL_HASH
    config_path = checkpoint.with_name("model.yaml")
    assert MODEL_HASH in config_path.read_text(encoding="utf-8")
    manifest = read_json(ROOT / CANONICAL / "source_manifest.json")
    identity = []
    for entry in manifest["files"]:
        blob = git("show", f"HEAD:{CANONICAL}/{entry['path']}")
        external_sha = file_hash(EXTERNAL / entry["path"])
        assert digest(blob) == external_sha == entry["sha256"], entry["path"]
        identity.append(dict(path=entry["path"], git_blob_sha256=digest(blob), external_sha256=external_sha))
    external_manifest = read_json(EXTERNAL / "source_manifest.json")
    external_entries = {entry["path"]: entry for entry in external_manifest["files"]}
    manifest_drift = [dict(canonical=entry, external_manifest=external_entries.get(entry["path"]))
                      for entry in manifest["files"] if entry != external_entries.get(entry["path"])]
    return dict(task_id=TASK, status="IN_PROGRESS", stage="A2_PROVENANCE_ESTABLISHED",
                starting_branch="docs/governance-v1-1-idle-state-semantics", starting_head=START,
                starting_working_tree="clean", task_branch=BRANCH,
                authorization_attachment_sha256="cd270805140da288a30d8ae67953241763b9de51e089c82a2e5f6ad9b8103a80",
                timestamp_utc=datetime.now(timezone.utc).isoformat(), A2_provenance=cases["A2"],
                positive_control=dict(status="PROVENANCE_ESTABLISHED", provenance=cases["A1"]),
                detector_model=dict(path=str(checkpoint), sha256=MODEL_HASH, bytes=checkpoint.stat().st_size,
                                    family="YOLO26m-seg", model_yaml_path=str(config_path), model_yaml_sha256=file_hash(config_path)),
                frozen_parameters=POLICY, source_identity=identity, manifest_entries_verified=len(identity),
                external_manifest_drift=manifest_drift,
                tile_count=None, per_tile=[], pipeline_counts=None,
                primary_root_cause_classification=None, root_cause_summary=None,
                threshold_sweep_run=False, parameter_tuning_run=False, product_source_changed=False,
                external_write=False, next_gate="CHATGPT_PROP01_FORENSICS_V1_REMOTE_AUDIT")


def observe_case(data, case, runtime):
    import numpy as np
    import torch
    from buildreasonseg.runtime import detector as d
    from buildreasonseg.runtime.imageio import load_image
    from ultralytics.engine.predictor import BasePredictor

    loaded = load_image(EXTERNAL / f"inference/input/{case}.png")
    windows = d.plan_tiles(loaded.width, loaded.height)
    records = []
    tensor_records = []
    raw_predict = runtime.load().predict
    original_tile = runtime.detect_tile
    original_compact = d._compact_mask
    original_merge = d.merge_proposals
    original_preprocess = BasePredictor.preprocess
    counts = dict(raw_ultralytics=0, detect_tile=0, accumulated_before_compact=0,
                  compact_valid=0, accumulated=0, merge_input=0, merged=0)
    compact_events = []
    active = {}

    def preprocess_observer(predictor, im):
        tensor = original_preprocess(predictor, im)
        # Diagnostic comparison only: build independent expected channel orders from
        # the library's own unchanged pre_transform (letterboxing). Never feed these
        # comparison tensors to the model or alter the returned production tensor.
        transformed = np.stack(predictor.pre_transform(im))
        native = torch.from_numpy(np.ascontiguousarray(transformed.transpose(0, 3, 1, 2)))
        native = native.to(tensor.device).to(tensor.dtype) / 255
        swapped = native.flip(1).contiguous()
        cpu = tensor.detach().cpu().contiguous().numpy()
        record = dict(shape=list(cpu.shape), dtype=str(cpu.dtype), sha256=digest(cpu.tobytes()),
                      min=float(cpu.min()), max=float(cpu.max()),
                      equals_letterboxed_source_RGB=bool(torch.equal(tensor, native)),
                      equals_letterboxed_source_channel_reversed=bool(torch.equal(tensor, swapped)),
                      source_channels_are_distinct=not bool(torch.equal(native, swapped)),
                      expected_source_RGB_tensor_sha256=digest(native.cpu().contiguous().numpy().tobytes()))
        tensor_records.append(record)
        return tensor

    def predict_observer(*args, **kwargs):
        tile = kwargs["source"]
        expected, padding = d.extract_tile(loaded.rgb, windows[len(records)])
        assert np.array_equal(tile, expected)
        window = windows[len(records)]
        assert not padding["applied"]
        assert np.array_equal(tile, loaded.rgb[window.top:window.top + 512, window.left:window.left + 512])
        assert kwargs == dict(source=tile, imgsz=640, conf=0.05, max_det=300,
                             verbose=False, retina_masks=False, device="cuda")
        results = raw_predict(*args, **kwargs)
        result = results[0] if results else None
        box_count = len(result.boxes) if result is not None and result.boxes is not None else 0
        confidences = result.boxes.conf.detach().cpu().tolist() if box_count else []
        mask_count = int(result.masks.data.shape[0]) if result is not None and result.masks is not None else 0
        active.clear()
        active.update(dict(tile_id=window.source_tile_id, tile_index=len(records), top=window.top, left=window.left,
                           shape=list(tile.shape), dtype=str(tile.dtype), padding=padding,
                           tile_sha256=digest(tile.tobytes()), equals_exact_source_slice=True,
                           detector_numpy_source_sha256=digest(kwargs["source"].tobytes()),
                           results_count=len(results), raw_ultralytics_count=box_count,
                           masks_present=bool(result is not None and result.masks is not None),
                           masks_count=mask_count, raw_confidences=confidences,
                           detector_tensor=tensor_records[-1]))
        counts["raw_ultralytics"] += box_count
        return results

    def tile_observer(tile):
        proposals = original_tile(tile)
        active["runtime_detect_tile_count"] = len(proposals)
        active["adapter_early_return"] = (
            "result.masks is None OR result.boxes is None OR len(result.boxes) == 0"
            if active["results_count"] and (not active["masks_present"] or active["raw_ultralytics_count"] == 0)
            else "not results" if not active["results_count"] else None)
        counts["detect_tile"] += len(proposals)
        records.append(dict(active))
        print(f"{case} {active['tile_id']}: raw={active['raw_ultralytics_count']} adapter={len(proposals)}", flush=True)
        return proposals

    def compact_observer(mask, *, top, left):
        counts["accumulated_before_compact"] += 1
        compact = original_compact(mask, top=top, left=left)
        counts["compact_valid"] += int(compact is not None)
        compact_events.append(dict(top=top, left=left, input_mask_shape=list(mask.shape),
                                   nonzero_pixels=int(np.count_nonzero(mask)), retained=compact is not None))
        return compact

    def merge_observer(entries):
        counts["accumulated"] = len(entries)
        counts["merge_input"] = len(entries)
        result = original_merge(entries)
        counts["merged"] = len(result[0])
        return result

    runtime._model.predict = predict_observer
    runtime.detect_tile = tile_observer
    d._compact_mask = compact_observer
    d.merge_proposals = merge_observer
    BasePredictor.preprocess = preprocess_observer
    try:
        output = runtime.detect_global(loaded.rgb)
    finally:
        runtime._model.predict = raw_predict
        runtime.detect_tile = original_tile
        d._compact_mask = original_compact
        d.merge_proposals = original_merge
        BasePredictor.preprocess = original_preprocess
    assert len(records) == len(windows) == len(tensor_records) == 9
    assert counts["detect_tile"] == output["raw_count"]
    args = vars(runtime._model.predictor.args)
    relevant = {k: args.get(k) for k in ("imgsz", "conf", "max_det", "iou", "agnostic_nms", "classes",
                                        "augment", "retina_masks", "half", "rect", "save", "save_txt",
                                        "save_crop", "device", "batch", "end2end")}
    assert not relevant["augment"] and not relevant["save"] and not relevant["save_txt"] and not relevant["save_crop"]
    return dict(tile_count=len(windows), per_tile=records, pipeline_counts=counts,
                compact_events=compact_events, raw_proposal_count=output["raw_count"],
                merged_proposal_count=len(output["merged"]), detector_seconds=output["detector_seconds"],
                effective_library_args=relevant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("provenance", "inference"), required=True)
    phase = parser.parse_args().phase
    sys.dont_write_bytecode = True
    before = snapshot()
    before_hashes = protected_hashes()
    with tempfile.TemporaryDirectory(prefix="prop01_forensics_") as temporary:
        os.environ.update(YOLO_CONFIG_DIR=temporary, MPLCONFIGDIR=temporary,
                          PYTHONDONTWRITEBYTECODE="1", YOLO_AUTOINSTALL="false")
        sys.path.insert(0, str(EXTERNAL))
        data = provenance()
        data["environment"] = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                                    libraries={name: importlib.metadata.version(name) for name in
                                               ("torch", "torchvision", "ultralytics", "numpy", "Pillow", "opencv-python")})
        if phase == "inference":
            import torch
            from buildreasonseg.runtime.detector import DetectorRuntime
            from ultralytics.engine.predictor import BasePredictor
            from ultralytics.data.loaders import LoadPilAndNumpy
            assert torch.cuda.is_available(), "Historical CUDA device unavailable; do not silently substitute"
            data["environment"].update(device="cuda", cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                                        gpu=torch.cuda.get_device_name(0))
            data["library_input_contract"] = {}
            for name, function in (("BasePredictor.preprocess", BasePredictor.preprocess),
                                   ("LoadPilAndNumpy._single_check", LoadPilAndNumpy._single_check)):
                path = Path(inspect.getsourcefile(function))
                lines, start = inspect.getsourcelines(function)
                data["library_input_contract"][name] = dict(path=str(path), sha256=file_hash(path),
                                                            first_line=start, source="".join(lines))
            runtime = DetectorRuntime(checkpoint=Path(data["detector_model"]["path"]), device="cuda")
            a2 = observe_case(data, "A2", runtime)
            data.update(a2)
            data["baseline_reproduction"] = "REPRODUCED" if a2["raw_proposal_count"] == a2["merged_proposal_count"] == 0 else "NON_REPRODUCIBLE_BASELINE"
            data["stage"] = "A2_FROZEN_BASELINE_AND_PER_TILE_CAPTURED"
            write(data)  # Recoverable before the single positive control.
            if data["baseline_reproduction"] == "REPRODUCED":
                control = observe_case(data, "A1", runtime)
                data["positive_control"].update(status="PROPOSAL_POSITIVE" if control["raw_proposal_count"] > 0 else "PROPOSAL_ZERO", result=control)
            data["detector_model"]["model_task"] = runtime._model.task
            data["stage"] = "INFERENCE_COMPLETE_PENDING_CLASSIFICATION"
            data["inference_calls"] = len(data["per_tile"]) + len(data["positive_control"].get("result", {}).get("per_tile", []))
        after = snapshot()
        after_hashes = protected_hashes()
        changed = [key for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)]
        hash_changes = [key for key in before_hashes if before_hashes[key] != after_hashes[key]]
        data["external_integrity"] = dict(inventory_file_count=len(before),
                                           inventory_before_sha256=digest(json.dumps(before, sort_keys=True).encode()),
                                           inventory_after_sha256=digest(json.dumps(after, sort_keys=True).encode()),
                                           changed_inventory_paths=changed, changed_protected_hash_paths=hash_changes,
                                           protected_content_hashes=before_hashes,
                                           hash_scope="135 manifest entries plus manifest, locked inputs/results/logs, model weights/config")
        write(data)
        assert not changed and not hash_changes, "External integrity changed: STOP"
        print(json.dumps(dict(stage=data["stage"], A2_sha256=data["A2_provenance"]["sha256"],
                              pipeline_counts=data["pipeline_counts"],
                              positive_control=data["positive_control"]["status"], external_changes=changed)))


if __name__ == "__main__":
    main()
