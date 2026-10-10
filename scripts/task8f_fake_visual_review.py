"""Fixed fake visual acceptance only. Never creates user-package results or loads weights."""
from __future__ import annotations
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.task8f_validate_demo import snapshot_window

def main():
    spec=importlib.util.spec_from_file_location("task8f_fake_visual_tests",REPO/"tests/test_task8f_user_demo_live_trace.py")
    tests=importlib.util.module_from_spec(spec);sys.modules[spec.name]=tests;spec.loader.exec_module(tests)
    import torch
    def forbidden(*args,**kw):raise RuntimeError("No weights/models in fake visual review")
    torch.load=forbidden
    torch.nn.Module.__init__=forbidden
    root_path=Path(tempfile.gettempdir())/"task8f-fake-visual-v1";root_path.mkdir(exist_ok=False)
    import tkinter as tk
    window=tk.Tk();app=tests.gui.DemoApp(window,demo_root=root_path,start_check=False)
    window.geometry("1160x850+25+10")
    session=tests.t.TraceSession(tests.b.reserve_result(root_path,Path("FIXED_FAKE.png")),app.worker.events)
    def action():
        tests.fake_events(session,8)
        session.emit(9,"RUNNING",{"guard_executed":True})
        session.emit(9,"FAILED",{"runtime_status":"FAILED","error_code":"E404","reason":"FAKE_direction_constraint_violated","guard_executed":True,"final_output_valid":False,"semantic_status":"NOT_EVALUATED"})
        session.finish(result=tests.result(code="E404"),engine_run_root="FAKE_TEST_ONLY")
    app.worker.start(action)
    deadline=time.monotonic()+15
    while time.monotonic()<deadline and (app.worker.busy or app.trace_page.cards[9]["state"].get()!="失败"):
        window.update();time.sleep(.02)
    if app.worker.busy or app.trace_page.cards[8]["state"].get()!="已完成" or app.trace_page.cards[9]["state"].get()!="失败":
        raise RuntimeError("Fake visual state did not converge")
    app.notebook.select(app.trace_tab)
    app.trace_page.banner.set("FIXED FAKE TEST ONLY / 非真实模型：E404 时保留 D-B1 候选，最终结果未通过。")
    window.update()
    pictures=[]
    for offset,name in ((0.,"proposals_context"),(.48,"fields_similarity"),(1.,"candidate_failure")):
        app.trace_page.canvas.yview_moveto(offset);window.update();time.sleep(.1)
        pictures.append(snapshot_window(window,REPO/f"docs/task8f_fake_{name}.png"))
    records={"pass":True,"model_calls":0,"root":str(root_path),"fake_only":True,"screenshots":pictures,
             "states":{str(i):session.stages[i] for i in range(1,10)},"stage8_candidate_retained":True,"stage9_failed":True}
    path=REPO/"evaluation/task8f_user_demo_live_trace_v1.json";e=json.loads(path.read_text(encoding="utf-8"));e["fake_visual_review"]=records;path.write_text(json.dumps(e,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    app.close();del app,window
    import gc;gc.collect()
    print(json.dumps({"pass":True,"fake_only":True,"model_calls":0}))
if __name__=="__main__":main()
