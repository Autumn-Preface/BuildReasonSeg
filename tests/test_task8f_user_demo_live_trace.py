"""V2 trace/UI tests with fixed fake stages; no weights or scientific model inference."""
from __future__ import annotations

import importlib.util
import json
import hashlib
import queue
import sys
import threading
import time
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
