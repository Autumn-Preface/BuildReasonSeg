"""V2 trace/UI tests with fixed fake stages; no weights or scientific model inference."""
from __future__ import annotations

import importlib.util
import json
import hashlib
import queue
import sys
import threading
import time
import subprocess
import random
import gc
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "demo/user_demo_v2"
CANONICAL = REPO / "delivery_src/BuildReasonSeg_Advisor_RC1"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(APP))
sys.path.insert(0, str(CANONICAL))
from _ui import backend as b, trace as t, render as r
from _ui.trace_ui import TracePage


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def engine_pair(tmp_path_factory):
    patcher = load_module("task8f_patch", APP / "engine_observation_patch.py")
    blobs = {p: subprocess.run(["git", "show", "93f66bc9a4fcb7b7ce5699f1c1afe412d18c43d4:delivery_src/BuildReasonSeg_Advisor_RC1/"+p], capture_output=True, check=True).stdout for p in patcher.FILES}
    patched, records = patcher.patched_sources(blobs)
    directory = tmp_path_factory.mktemp("derived-engine")
    observation = load_module("buildreasonseg.runtime.observation", APP / "engine_observation/observation.py")
    old, new = {}, {}
    for i, relative in enumerate(patcher.FILES):
        a, z = directory / f"old_{i}.py", directory / f"new_{i}.py"
        a.write_bytes(blobs[relative]); z.write_bytes(patched[relative])
        key = Path(relative).stem
        old[key] = load_module("task8f_old_"+key, a)
        new[key] = load_module("task8f_new_"+key, z)
    return NS(old=old, new=new, observation=observation, patcher=patcher, blobs=blobs, patched=patched, records=records)


class FixedFakeDb1:
    """No nn.Module or weights. Executes original competition/forward with fixed tensor fixtures."""
    variant = "D-B1"
    uses_learned_score_head = False
    uses_relation_fields = True
    uses_prototype = True
    def __init__(self, competition, mask):
        import torch
        self.module, self.mask = competition, mask
        self.calls = {"projection": 0, "competition": 0, "forward": 0}
        self.states = []
    def direction_embedding(self, relations, h, w, device):
        import torch
        return torch.zeros((1, 16, h, w), device=device)
    def visual_projection(self, visual):
        import torch
        self.calls["projection"] += 1
        # Deliberately distinguish diagnostic and actual forward states without any randomness.
        v = torch.arange(128*64*64, dtype=torch.float32).reshape(1,128,64,64)
        return torch.sin(v * .00013 * self.calls["projection"])
    def competition(self, *args):
        self.calls["competition"] += 1
        state = self.module.GlobalCompetitionDecoder.competition(self, *args)
        self.states.append(state)
        return state
    def trunk(self, features):
        import torch
        return torch.as_tensor(self.mask.astype(np.float32)*4-2)[None,None]
    def __call__(self, *args):
        self.calls["forward"] += 1
        return self.module.GlobalCompetitionDecoder.forward(self, *args)
    def upsampled(self, logits, size):
        return self.module.GlobalCompetitionDecoder.upsampled(self, logits, size)


def fake_core(pair, which, monkeypatch, mask=None, crash=None):
    import torch
    modules = getattr(pair, which)
    core, frozen = modules["core"], modules["task7d_global_competition_decoder"]
    mask = mask if mask is not None else np.pad(np.ones((32,32),bool), ((32,0),(16,16)))
    model = FixedFakeDb1(frozen, mask)
    db = NS(device="cpu", load=lambda: model)
    db.forward = lambda *args, **kw: core.Db1Runtime.forward(db, *args, **kw)
    sam_calls = []
    def encode(rgb):
        sam_calls.append(rgb.copy())
        if crash == "sam":
            raise RuntimeError("fixed fake SAM failure")
        return torch.zeros((1,256,64,64))
    def fields(ref, program):
        if crash == "fields":
            raise RuntimeError("fixed fake fields failure")
        return {"P_dir_64": np.full((64,64), .75, np.float32), "P_near_64": np.linspace(.1,.9,4096,dtype=np.float32).reshape(64,64)}
    monkeypatch.setattr(core, "record_fields", fields)
    if crash == "db1":
        db.forward = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fixed fake D-B1 failure"))
    return core, NS(encode=encode), db, model, sam_calls


def test_exact_scientific_ast_both_observer_paths_and_math_change_rejected(engine_pair):
    pair = engine_pair
    assert all(row.get("scientific_ast_audit", {}).get("paths") == {"enabled":True,"disabled":True} for row in pair.records[:3])
    relative = pair.patcher.FILES[-1]
    changed = pair.patched[relative].replace(b"attention = weight /", b"attention = 2 * weight /")
    with pytest.raises(ValueError, match="Scientific AST differs"):
        pair.patcher.audit_scientific_ast(pair.blobs[relative], changed)
    relative = pair.patcher.FILES[1]
    changed = pair.patched[relative].replace(b"with _trace_scope(observer):", b"with _trace_scope(observer):\n                        visual_batch = visual_batch * 2")
    with pytest.raises(ValueError, match="Scientific AST differs"):
        pair.patcher.audit_scientific_ast(pair.blobs[relative], changed)


@pytest.mark.parametrize("direction", ["left","right","above","below"])
def test_core_outputs_counts_rng_identical_and_actual_forward_maps(engine_pair, monkeypatch, direction):
    import torch
    pair = engine_pair
    records, values, counts = [], [], []
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    for which, enabled in (("old",False),("new",False),("new",True)):
        core, sam, db, model, sam_calls = fake_core(pair, which, monkeypatch)
        kw = {"observer": records.append} if enabled else {}
        output = core.run_core_chain(np.zeros((512,512,3),np.uint8), np.ones((512,512),bool), "largest_to_"+direction+"_to_nearest" if direction in ("above","below") else "largest_to_"+direction+"_of_to_nearest", sam2=sam, db1=db, **kw)
        values.append(output)
        counts.append((dict(model.calls),len(sam_calls)))
        if enabled:
            captured = next(e for e in records if e["stage_id"] == 7 and e["state"] == "COMPLETED")
            assert captured["metadata"]["q_dimension"] == 128
            np.testing.assert_array_equal(captured["arrays"]["C"], model.states[1].similarity.numpy())
            assert not np.array_equal(captured["arrays"]["C"], model.states[0].similarity.numpy())
    for value in values[1:]:
        for field in ("logits","mask_context","probability_context","direction_field","nearest_field","relation_weight","prototype_similarity","attention"):
            np.testing.assert_array_equal(getattr(values[0], field), getattr(value,field))
        assert value.field_mass == values[0].field_mass and value.notes == values[0].notes
    assert counts == [({"projection":2,"competition":2,"forward":1},1)]*3
    assert random.getstate() == states[0]
    current_np = np.random.get_state()
    assert current_np[0] == states[1][0] and np.array_equal(current_np[1],states[1][1]) and current_np[2:] == states[1][2:]
    assert torch.equal(torch.get_rng_state(), states[2])
    assert [(e["stage_id"],e["state"]) for e in records] == [(5,"RUNNING"),(5,"COMPLETED"),(6,"RUNNING"),(6,"RUNNING"),(6,"COMPLETED"),(7,"RUNNING"),(7,"COMPLETED"),(8,"RUNNING"),(8,"COMPLETED")]
    assert pair.observation.active_observer() is None


def test_observer_failure_and_immutable_snapshot_do_not_change_core(engine_pair, monkeypatch):
    pair = engine_pair
    core, sam, db, model, _ = fake_core(pair, "new", monkeypatch)
    failures=[]
    class Failing:
        def __call__(self, event):
            for array in event["arrays"].values():
                if isinstance(array,np.ndarray):
                    with pytest.raises(ValueError):
                        array.setflags(write=True)
            raise RuntimeError("renderer failed")
        def observation_failed(self, error, stage): failures.append(stage)
    observer = Failing()
    output = core.run_core_chain(np.zeros((512,512,3),np.uint8), np.ones((512,512),bool), "largest_to_above_to_nearest", sam2=sam,db1=db,observer=observer)
    assert output.mask_context.any() and failures
    assert len(pair.observation.observation_errors(observer)) == len(failures)
    assert model.calls == {"projection":2,"competition":2,"forward":1}
    assert pair.observation.active_observer() is None


@pytest.mark.parametrize("case", ["success","zero","no_reference","context","empty","wrong_direction","sam","fields","db1"])
def test_pipeline_scientific_output_error_counts_and_retained_stages(engine_pair, monkeypatch, tmp_path, case):
    from buildreasonseg.runtime.detector import GlobalProposal
    from buildreasonseg.runtime import outputs
    from buildreasonseg.errors import BuildReasonSegError
    pair = engine_pair
    image = tmp_path / "fake.png"; Image.new("RGB",(512,512),(110,140,170)).save(image)
    records, results, calls = [], [], []
    for i,(which,enabled) in enumerate((("old",False),("new",False),("new",True))):
        pipeline = getattr(pair,which)["pipeline"]
        core,sam,db,model,sam_calls = fake_core(pair,which,monkeypatch,mask=np.zeros((64,64),bool) if case=="empty" else np.pad(np.ones((8,8),bool),((48,8),(16,40))) if case=="wrong_direction" else np.pad(np.ones((8,8),bool),((24,32),(20,36))),crash=case)
        monkeypatch.setattr(pipeline,"run_core_chain",core.run_core_chain)
        prop = GlobalProposal(proposal_id=0,source_tile_id="fake",tile_index=0,raw_index=0,global_bbox=(240,240,270,270),mask_crop=np.ones((31,31),bool),mask_area=961,confidence=.9,centroid=(255,255),touches_image_border=False,border_clearance=240,image_size=(512,512))
        target = GlobalProposal(proposal_id=1,source_tile_id="fake",tile_index=0,raw_index=1,global_bbox=(60,240,80,260),mask_crop=np.ones((21,21),bool),mask_area=441,confidence=.8,centroid=(70,250),touches_image_border=False,border_clearance=60,image_size=(512,512))
        detected=[]
        def detect(rgb):
            detected.append(rgb.copy())
            return {"merged": [] if case=="zero" else [prop,target],"raw_count":0 if case=="zero" else 2,"detector_seconds":.01,"tile_size":512,"overlap":128,"windows":[1]}
        runtime = NS(device="cpu", detector=NS(detect_global=detect),sam2=sam,db1=db)
        if case=="no_reference": monkeypatch.setattr(pipeline,"select_reference",lambda *a,**k:None)
        if case=="context": monkeypatch.setattr(pipeline,"guard_directional_candidates",lambda *a: (_ for _ in ()).throw(BuildReasonSegError("E404",context={"reason":"no_directional_candidate"})))
        directory = tmp_path / str(i); directory.mkdir()
        for name in ("diagnostics","masks","overlays"): (directory/name).mkdir()
        output = outputs.SampleOutputs(stem="fake",suffix_index=0,diagnostics_dir=str(directory/"diagnostics"),mask_path=str(directory/"masks/fake_mask.png"),overlay_path=str(directory/"overlays/fake_overlay.png"))
        monkeypatch.setattr(outputs,"allocate_outputs",lambda p:output)
        request = pipeline.PipelineRequest(image=image,parsed=ParsedProgram(program="largest_to_above_to_nearest",source="qwen_program_head"),parsed_info={"original_prompt":"FAKE only"})
        result = pipeline.predict_one(runtime,request,**({"observer":records.append} if enabled else {}))
        results.append(result); calls.append((len(detected),len(sam_calls),dict(model.calls)))
    assert calls[0] == calls[1] == calls[2]
    assert [x.status for x in results] == [results[0].status]*3
    assert [x.error_code for x in results] == [results[0].error_code]*3
    assert [x.error_reason for x in results] == [results[0].error_reason]*3
    if case=="success":
        assert results[0].ok
        for x in results[1:]: np.testing.assert_array_equal(x.mask,results[0].mask)
        for name in ("masks/fake_r000.png","overlays/fake_r000.png"):
            # Save names are taken from returned paths, never inferred from expected suffixes.
            key = "mask" if name.startswith("masks") else "overlay"
            data = [Path(x.result_payload["output_paths"][key]).read_bytes() for x in results]
            assert data[0] == data[1] == data[2]
        assert results[2].result_payload["semantic_status"] == "NOT_EVALUATED"
    else:
        expected = {"zero":"E401","no_reference":"E402","context":"E404","empty":"E404","wrong_direction":"E404","sam":"E502","fields":"E502","db1":"E502"}[case]
        assert results[0].error_code == expected
        assert not list(tmp_path.rglob("masks/*.png")) and not list(tmp_path.rglob("overlays/*.png"))
    completed = [e["stage_id"] for e in records if e["state"]=="COMPLETED"]
    if case in ("empty","wrong_direction"): assert completed == [2,3,4,5,6,7,8] and records[-1]["state"]=="FAILED" and records[-1]["metadata"]["guard_executed"]
    if case=="zero": assert completed==[2] and records[-1]["metadata"]["error_code"]=="E401"
    if case=="no_reference": assert completed==[2] and records[-1]["metadata"]["error_code"]=="E402"
    if case=="context": assert completed==[2,3] and any(e["stage_id"]==4 and e["metadata"].get("context") for e in records)
    if case=="sam": assert completed==[2,3,4]
    if case in ("fields","db1"): assert completed==[2,3,4,5]
    # Real hook packets, including torch 4-D W/A/C, must survive the actual renderer/storage bridge.
    trace=session(tmp_path,"bridge")
    trace.emit(1,"RUNNING"); trace.emit(1,"COMPLETED",{"original_prompt":"FIXED FAKE / no model"})
    for event in records: trace(event)
    assert trace.finish(result=results[2],engine_run_root="FAKE_PRIVATE_TMP_ENGINE_RUN")
    assert not trace.failures
    assert (trace.trace/"stage_events.jsonl").is_file()
    assert trace.stages[9] == ("COMPLETED" if case=="success" else "FAILED")
    if case in ("empty","wrong_direction"):
        assert trace.stages[8]=="COMPLETED" and trace.latest[8]["previews"]


def test_v2_builder_plan_provenance_and_rejects_math_drift():
    from scripts import build_task8f_user_demo as builder
    blobs=builder.accepted_blobs()
    plan,derived=builder.plan(blobs)
    assert len(plan["source_paths"])==54 and len(plan["assets"])==17
    assert plan["expected_manifest_entries"]==84
    assert len(plan["observation_source_differences"])==4 and len(derived)==4
    assert builder.FINAL.name=="BuildReasonSeg_Demo_V2" and builder.STAGING.parent==builder.FINAL.parent
    for row in plan["observation_source_differences"][:3]:
        assert row["base_sha256"] == hashlib.sha256(blobs[row["path"]]).hexdigest()
        assert row["v2_sha256"] == hashlib.sha256(derived[row["path"]]).hexdigest()
    assert not any("weights" in r or "torch.load" in r for r in derived["buildreasonseg/runtime/observation.py"].decode().splitlines())


def test_existing_result_controls_and_mask_view_retained(tmp_path,monkeypatch):
    root,app=make_app(tmp_path)
    try:
        directory=tmp_path/"results/success";directory.mkdir(parents=True)
        Image.new("RGB",(512,512)).save(directory/"overlay.png")
        Image.new("L",(512,512),255).save(directory/"mask.png")
        app.present_result({"directory":str(directory),"summary":{"runtime_status":"SUCCESS","overlay_file":"overlay.png","mask_file":"mask.png","prompt":"FAKE","interpreted_program":"largest_to_above_to_nearest","elapsed_seconds":.1}})
        assert app.mask_button.instate(["!disabled"]) and app.results_button.instate(["!disabled"])
        opened=[];monkeypatch.setattr(gui.os,"startfile",lambda path:opened.append(str(path)),raising=False)
        app.open_results();assert opened==[str(directory)]
        app.view_mask();root.update()
        assert any(isinstance(child,app.tk.Toplevel) for child in root.winfo_children())
        for child in root.winfo_children():
            if isinstance(child,app.tk.Toplevel): child.destroy()
        assert "语义正确性：未自动验证" in app.summary.get()
    finally: app.close()


gui = load_module("task8f_gui", APP / "BuildReasonSeg_Demo.py")
helpers = load_module("task8f_cli", CANONICAL / "predict.py")
from buildreasonseg.language.frontend import DeterministicFallbackFrontend
from buildreasonseg.language.registry import ParsedProgram
from buildreasonseg.language.validator import validate


@pytest.fixture(autouse=True)
def no_models(monkeypatch, tmp_path):
    import torch
    from buildreasonseg.runtime import pipeline
    def forbidden(*args, **kwargs):
        pytest.fail("Real weights/model inference prohibited in Task8F fake tests")
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.nn.Module, "__init__", forbidden)
    for cls, methods in ((pipeline.DetectorRuntime, ("load", "detect_global")),
                         (pipeline.Sam2Runtime, ("load", "encode")),
                         (pipeline.Db1Runtime, ("load", "forward")),
                         (pipeline.ProgramHeadRuntime, ("load", "parse", "generate_suggestion"))):
        for method in methods:
            monkeypatch.setattr(cls, method, forbidden)
    from _ui.environment import configure_environment
    configure_environment(tmp_path / "private-cache-owner")
    yield
    # Multiple real Tk interpreters in one test process must be disposed by their owner thread.
    gc.collect()


def fixture_arrays():
    rgb = np.full((640, 700, 3), (90, 130, 160), np.uint8)
    props = [{"proposal_id": i, "bbox": [100, 100+i*80, 130, 130+i*80],
              "mask_crop": np.ones((31, 31), bool)} for i in (0, 1, 2)]
    context = np.full((512, 512, 3), (110, 140, 170), np.uint8)
    ref = np.zeros((512, 512), bool); ref[180:215, 190:225] = True
    field = np.linspace(0, 1, 4096).reshape(64, 64).astype(np.float32)
    mask = np.zeros((512, 512), bool); mask[40:80, 190:230] = True
    return rgb, props, context, ref, field, mask


def fake_events(session, until=8, zero=False):
    rgb, props, context, ref, field, mask = fixture_arrays()
    data = {
        1: ({"original_prompt": "FAKE_TEST_ONLY 最大建筑上方最近的建筑", "initial_program": "largest_to_above_to_nearest", "initial_supported": True, "final_program": "largest_to_above_to_nearest"}, {}),
        2: ({"raw_count": len(props), "merged_count": 0 if zero else len(props), "detector_seconds": .1, "proposal_ids": [] if zero else [0, 1, 2]}, {"rgb": rgb, "proposals": [] if zero else props}),
        3: ({"selected_id": 1, "mode": "automatic", "mask_area": 961, "confidence": .85, "bbox": props[1]["bbox"]}, {"reference_mask_crop": props[1]["mask_crop"]}),
        4: ({"context": {"origin": [25, 70], "size": 512, "padding": {"applied": False}}, "direction": "above", "reference_pixels": int(ref.sum())}, {"context_rgb": context, "reference_mask": ref}),
        5: ({"encoding_complete": True, "feature_shape": [1, 256, 64, 64], "raw_feature_tensor_saved": False, "elapsed_seconds": .1}, {}),
        6: ({"source": "FAKE_FIXED_MAPS"}, {"P_dir": field, "P_near": 1-field, "W": field*(1-field), "A": np.full((64, 64), 1/4096, np.float32)}),
        7: ({"q_formed": True, "q_shape": [1, 128], "q_dimension": 128, "C_shape": [1, 1, 64, 64], "similarity_scope": "visual cosine, not target correctness probability"}, {"C": field*2-1}),
        8: ({"candidate_only": True, "validation_state": "NOT_YET_GUARDED", "mask_pixels": int(mask.sum())}, {"mask": mask, "probability": mask.astype(np.float32)*.8, "logits": mask.astype(np.float32)*3-1}),
    }
    for i in range(1, until+1):
        session.emit(i, "RUNNING")
        metadata, arrays = data[i]
        session.emit(i, "COMPLETED", metadata, arrays)


def session(tmp_path, name="run", renderer=None):
    directory = tmp_path / "results" / name
    directory.mkdir(parents=True)
    return t.TraceSession(directory, queue.Queue(), renderer=renderer)


def result(status="FAILED", code="E404"):
    return NS(status=status, ok=status=="SUCCESS", error_code=None if status=="SUCCESS" else code,
              result_payload={"status": status, "output_paths": {}, "validity_scope": "RUNTIME_STRUCTURAL_ONLY", "semantic_status": "NOT_EVALUATED"})


def test_snapshot_is_irreversibly_read_only_and_detached():
    array = np.arange(12).reshape(3, 4)
    packet = t.immutable({"array": array, "nested": [{"x": 3}]})
    array[0, 0] = 99
    assert packet["array"][0, 0] == 0
    with pytest.raises(ValueError):
        packet["array"][0, 0] = 2
    with pytest.raises(ValueError):
        packet["array"].setflags(write=True)
    with pytest.raises(TypeError):
        packet["nested"][0]["x"] = 4


@pytest.mark.parametrize("selected", [None, 0, 1, 2])
def test_all_proposals_shown_and_original_masks_unchanged(selected):
    rgb, props, *_ = fixture_arrays()
    before = [p["mask_crop"].tobytes() for p in props]
    _, ids = r.proposal_preview(rgb, props, selected)
    assert ids == [0, 1, 2] and before == [p["mask_crop"].tobytes() for p in props]


@pytest.mark.parametrize("name,fixed", [("P_dir", (0,1)), ("P_near", (0,1)), ("W", (0,1)), ("A", None), ("C", (-1,1))])
def test_display_normalization_does_not_mutate_raw_map(name, fixed):
    a = np.linspace(-.2, .8, 4096, dtype=np.float32).reshape(64,64)
    before = a.tobytes()
    _, info = r.heatmap(a, name, fixed_range=fixed)
    assert a.tobytes() == before and info["sha256"] == hashlib.sha256(before).hexdigest()
    assert info["color_is_not_raw_probability"] and info["raw_min"] == float(a.min())


@pytest.mark.parametrize("code,until,failed_stage", [("E401",2,None),("E402",2,3),("E404",8,None),("E502",5,6)])
def test_failures_preserve_completed_stage_snapshots_and_no_final_fake_images(tmp_path, code, until, failed_stage):
    s = session(tmp_path)
    fake_events(s,until,zero=code=="E401")
    if failed_stage:
        s.emit(failed_stage,"RUNNING")
    s.emit(9,"FAILED",{"runtime_status":"FAILED","error_code":code,"guard_executed":code=="E404","final_output_valid":False})
    assert s.finish(result=result(code=code))
    assert all(s.stages[i]=="COMPLETED" for i in range(1,until+1))
    if failed_stage:
        assert s.stages[failed_stage]=="FAILED"
    assert all(s.stages[i]=="NOT_EXECUTED" for i in range(until+1,9) if i!=failed_stage)
    assert not (s.directory/"mask.png").exists() and not (s.directory/"overlay.png").exists()
    if code=="E401":
        assert s.latest[2]["metadata"]["merged_count"]==0 and s.latest[2]["previews"]
    if code=="E404":
        assert s.latest[8]["previews"] and s.latest[8]["metadata"]["candidate_only"]


def test_context_preview_uses_supplied_true_origin_rgb_padding(tmp_path):
    s=session(tmp_path);fake_events(s,4);s.finish(result=result())
    row=s.latest[4]
    assert row["metadata"]["context"]["origin"]==[25,70]
    assert row["metadata"]["context"]["size"]==512
    assert row["metadata"]["array_snapshot_identities"]["context_rgb"]["sha256"]==hashlib.sha256(fixture_arrays()[2].tobytes()).hexdigest()
    assert len(row["previews"])==2


def test_no_feature_tensor_saved_and_small_previews(tmp_path):
    s=session(tmp_path);fake_events(s);s.emit(9,"FAILED",{"runtime_status":"FAILED","error_code":"E404"});s.finish(result=result())
    assert not list(s.trace.glob('*.npz')) and not list(s.trace.glob('*.pt'))
    assert s.latest[5]["metadata"]["feature_shape"]==[1,256,64,64]
    for p in s.trace.glob('*.png'):
        with Image.open(p) as image:
            assert image.width<=960 and image.height<=720


def test_real_stage_required_for_completion_order(tmp_path):
    s=session(tmp_path);s.emit(6,"COMPLETED",{}, {"W":np.zeros((64,64))})
    assert not s.finish(result=result(code="E502"))
    assert s.failures and s.stages[6]!="COMPLETED"


def test_user_cancel_has_no_later_stage_completion(tmp_path):
    s=session(tmp_path);s.emit(1,"RUNNING",{"initial_program":"largest_to_above","initial_supported":False})
    s.finish(error=b.DemoFailure("E102"),cancelled=True)
    assert s.stages[1]=="CANCELLED" and all(s.stages[i]=="NOT_EXECUTED" for i in range(2,10))
    assert not list(s.trace.glob('*.png'))


def test_persistence_reopen_complete_and_failure_runs(tmp_path):
    s=session(tmp_path);fake_events(s,2,zero=True);s.emit(9,"FAILED",{"runtime_status":"FAILED","error_code":"E401"});s.finish(result=result(code="E401"))
    events=queue.Queue();manifest=t.reopen_trace(s.trace/"trace_manifest.json",tmp_path,events)
    assert manifest["event_count"]==len(s.records)
    received=[]
    while not events.empty():received.append(events.get())
    assert received[0][0]=="trace_start" and received[-1][0]=="trace_frozen"
    assert all(item[1]["event"]["run_id"]==s.run_id for item in received if item[0]=="stage_event")


def test_reopen_rejects_cross_run_or_modified_snapshot(tmp_path):
    s=session(tmp_path);fake_events(s,2);s.finish(result=result())
    png=s.trace/s.latest[2]["previews"][0]["path"];png.write_bytes(b"tampered-test")
    events=queue.Queue()
    with pytest.raises(ValueError):t.reopen_trace(s.trace/"trace_manifest.json",tmp_path,events)
    assert events.empty()


def test_observation_failure_explicit_and_scientific_result_untouched(tmp_path):
    class BadRenderer:
        def render(self,*args):raise OSError('synthetic storage failure')
    s=session(tmp_path,renderer=BadRenderer());scientific=result("SUCCESS")
    s.emit(1,"RUNNING");assert not s.finish(result=scientific)
    assert scientific.status=="SUCCESS" and s.failures
    out=b.publish_result(tmp_path,Path('fake.png'),'text',None,{},scientific,observation_failed=True,directory=s.directory)
    assert out["summary"]["runtime_status"]=="FAILED" and out["summary"]["underlying_runtime_status"]=="SUCCESS"
    assert out["summary"]["failure_scope"]=="OBSERVATION_ONLY"
    assert not (s.directory/'mask.png').exists()


class FakeLanguage:
    def __init__(self, program='largest_to_above',available=True,fails=False):
        self.program,self.is_available,self.fails=program,available,fails
        self.parse_calls=self.suggestion_calls=0
    def available(self):return self.is_available,'fake only'
    def parse(self,prompt):
        self.parse_calls+=1
        if self.fails:raise RuntimeError('fake language failure')
        return NS(program=self.program,confidence=.7,score_source='FAKE_ONLY',to_dict=lambda:{'top5':[]})
    def generate_suggestion(self,prompt,program):
        self.suggestion_calls+=1
        return json.dumps({'status':'SUGGESTION','program':'largest_to_above_to_nearest','display_command':'最大建筑上方最近的建筑'})


def language_adapter(language,trace=None):
    adapter=b.EngineAdapter.__new__(b.EngineAdapter)
    adapter.runtime=NS(language_runtime=lambda:language)
    adapter.Parsed,adapter.validate=ParsedProgram,validate
    adapter.Fallback,adapter.helpers=DeterministicFallbackFrontend,helpers
    adapter.observer=trace;adapter.language_trace={}
    return adapter


@pytest.mark.parametrize('accepted',[True,False])
def test_language_missing_nearest_and_suggestion_confirmation_truth(tmp_path,accepted):
    s=session(tmp_path);s.emit(1,'RUNNING')
    lang=FakeLanguage();adapter=language_adapter(lang,s);confirmations=[]
    confirm=lambda k,m:confirmations.append(k) or accepted
    if accepted:
        parsed,info,display=adapter.parse('FAKE input missing nearest',confirm)
        assert parsed.program=='largest_to_above_to_nearest' and info['suggestion_used']
    else:
        with pytest.raises(b.DemoFailure):adapter.parse('FAKE input missing nearest',confirm)
    s.finish(error=b.DemoFailure('E102'),cancelled=True)
    initial=[row for row in s.records if row['metadata'].get('initial_program')=='largest_to_above']
    assert initial and initial[0]['metadata']['initial_has_nearest'] is False
    assert initial[0]['metadata']['initial_supported'] is False
    assert confirmations==['suggestion'] and lang.parse_calls==lang.suggestion_calls==1


@pytest.mark.parametrize('available,fails',[(False,False),(True,True)])
def test_fallback_still_needs_consent_and_final_confirmation(available,fails):
    adapter=language_adapter(FakeLanguage(available=available,fails=fails));calls=[]
    parsed,info,_=adapter.parse(b.EXAMPLES[2],lambda k,m:calls.append(k) or True)
    assert calls==['fallback','direct'] and info['language_mode']=='fallback' and validate(parsed).supported


@pytest.mark.parametrize('program',list(b.DIRECTIONS))
def test_direct_supported_always_requires_confirmation(program):
    adapter=language_adapter(FakeLanguage(program));calls=[]
    parsed,info,_=adapter.parse('FAKE_TEXT',lambda k,m:calls.append(k) or True)
    assert calls==['direct'] and parsed.program==program and info['user_confirmation']=='Y'


def make_app(tmp_path, **kwargs):
    import tkinter as tk
    root=tk.Tk();root.withdraw()
    app=gui.DemoApp(root,demo_root=tmp_path,start_check=False,**kwargs)
    root.update()
    return root,app


def test_two_peer_tabs_switch_without_worker_or_inference(tmp_path,monkeypatch):
    root,app=make_app(tmp_path)
    try:
        assert [app.notebook.tab(i,'text') for i in range(2)]==['结果','推理过程']
        assert app.result_tab.master is app.notebook and app.trace_tab.master is app.notebook
        calls=[];monkeypatch.setattr(app.worker,'start',lambda action:calls.append(action))
        for _ in range(6):
            app.notebook.select(app.trace_tab);root.update();app.notebook.select(app.result_tab);root.update()
        assert not calls
    finally:app.close()


def test_quickfill_text_only_and_busy_duplicate_run_disabled(tmp_path):
    root,app=make_app(tmp_path)
    try:
        for i,button in enumerate(app.example_buttons):
            button.invoke();assert app.prompt.get('1.0','end').strip()==b.EXAMPLES[i]
        app.worker.busy=True;app.set_busy(True)
        assert app.run_button.instate(['disabled']) and app.trace_page.open_button.instate(['disabled'])
        assert app.notebook.instate(['!disabled'])
        app.run();app.run();assert app.worker.thread is None
        app.worker.busy=False
    finally:app.close()


def test_trace_ui_rejects_other_run_and_worker_thread_updates(tmp_path):
    root,app=make_app(tmp_path)
    try:
        page=app.trace_page;page.begin({'run_id':'new','directory':'new'})
        packet={'event':{'run_id':'old','stage_id':2,'state':'COMPLETED','metadata':{}},'images':[]}
        assert page.stage(packet) is False and page.cards[2]['state'].get()=='未执行'
        errors=[]
        def background():
            try:page.stage(packet)
            except RuntimeError as error:errors.append(str(error))
        thread=threading.Thread(target=background);thread.start();thread.join()
        assert errors
        assert '尚未执行' in page.cards[8]['note'].get()
    finally:app.close()


def test_worker_serial_and_rendering_does_not_block_main_thread(tmp_path):
    entered,release=threading.Event(),threading.Event()
    class SlowRenderer(r.PreviewRenderer):
        def render(self,stage,*args):
            if stage==2:entered.set();assert release.wait(8)
            return super().render(stage,*args)
    root,app=make_app(tmp_path)
    s=t.TraceSession(b.reserve_result(tmp_path,Path('fake.png')),app.worker.events,renderer=SlowRenderer())
    def action():fake_events(s,2);s.emit(9,'FAILED',{'error_code':'E401','runtime_status':'FAILED'});s.finish(result=result(code='E401'))
    try:
        assert app.worker.start(action) and not app.worker.start(action)
        assert entered.wait(5)
        beats=[];root.after(20,lambda:beats.append('responsive'))
        deadline=time.monotonic()+1
        while time.monotonic()<deadline and not beats:
            app.notebook.select(app.trace_tab);root.update();time.sleep(.02)
        assert beats and app.worker.busy
        release.set();app.worker.thread.join(8);assert not app.worker.busy
        deadline=time.monotonic()+.3
        while time.monotonic()<deadline:root.update();time.sleep(.01)
        assert app.trace_page.cards[2]['state'].get()=='已完成'
    finally:
        release.set()
        if app.worker.thread:app.worker.thread.join(8)
        app.close()


def test_no_diagnostic_controls_and_success_uncertainty():
    source=(APP/'BuildReasonSeg_Demo.py').read_text(encoding='utf-8')
    assert '--reference-id' not in source and '--inspect-proposals' not in source
    assert 'threshold' not in source and 'ranking' not in source
    assert '人工核验目标是否正确' in b.SUCCESS_TEXT


def test_missing_observations_cannot_be_silent_success(tmp_path):
    s=session(tmp_path);s.emit(1,'RUNNING');s.emit(1,'COMPLETED')
    scientific=result('SUCCESS')
    assert not s.finish(result=scientific)
    assert scientific.status=='SUCCESS' and s.stages[2]=='NOT_EXECUTED'
    assert json.loads((s.trace/'trace_manifest.json').read_text(encoding='utf-8'))['status']=='OBSERVATION_FAILED'


def test_stage_display_updates_before_worker_execution_ends(tmp_path):
    release=threading.Event()
    root,app=make_app(tmp_path)
    s=t.TraceSession(b.reserve_result(tmp_path,Path('fake.png')),app.worker.events)
    def action():
        fake_events(s,2)
        assert release.wait(8)
        s.emit(9,'FAILED',{'runtime_status':'FAILED','error_code':'E401'})
        s.finish(result=result(code='E401'))
    try:
        assert app.worker.start(action)
        deadline=time.monotonic()+5
        while time.monotonic()<deadline and app.trace_page.cards[2]['state'].get()!='已完成':
            app.notebook.select(app.trace_tab);root.update();time.sleep(.02)
        assert app.trace_page.cards[2]['state'].get()=='已完成' and app.worker.busy
        assert app.trace_page.cards[3]['state'].get()=='未执行'
        release.set();app.worker.thread.join(8)
    finally:
        release.set()
        if app.worker.thread:app.worker.thread.join(8)
        app.close()
