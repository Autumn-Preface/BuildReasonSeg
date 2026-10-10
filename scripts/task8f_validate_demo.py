"""Native no-model validation of the authorized new V2 folder; private caches only."""
from __future__ import annotations
import argparse
import ctypes
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]

def snapshot_window(window,path):
    from PIL import ImageGrab
    window.update()
    hwnd=ctypes.windll.user32.GetParent(window.winfo_id())
    bounds=(ctypes.c_long*4)()
    if ctypes.windll.dwmapi.DwmGetWindowAttribute(hwnd,9,ctypes.byref(bounds),ctypes.sizeof(bounds))!=0:
        raise RuntimeError("Cannot obtain this test window's bounds")
    # PrintWindow captures only this owned test window, even if another app is foreground.
    ImageGrab.grab(window=hwnd).save(path)
    return str(path.relative_to(REPO))

def validate(root,phase):
    sys.path.insert(0,str(root))
    from _ui.environment import configure_environment
    configure_environment(root)
    spec=importlib.util.spec_from_file_location("task8f_native_gui",root/"BuildReasonSeg_Demo.py")
    gui=importlib.util.module_from_spec(spec);spec.loader.exec_module(gui)
    checked=gui.self_check(root,full=True)
    if not checked["ok"]: raise RuntimeError("New V2 full self-check failed: "+str(checked))
    from _ui.backend import bind_engine,EngineAdapter
    bind_engine(root)
    import torch
    from buildreasonseg.runtime import pipeline
    calls=[]
    def forbidden(*args,**kwargs):
        calls.append("prohibited scientific call")
        raise RuntimeError("Model loading/inference forbidden during native no-model validation")
    torch.load=forbidden
    for cls,methods in ((pipeline.DetectorRuntime,("load","detect_global")),(pipeline.Sam2Runtime,("load","encode")),(pipeline.Db1Runtime,("load","forward")),(pipeline.ProgramHeadRuntime,("load","parse","generate_suggestion"))):
        for method in methods: setattr(cls,method,forbidden)
    adapter=EngineAdapter(root)
    import tkinter as tk
    window=tk.Tk(); app=gui.DemoApp(window,demo_root=root)
    window.geometry("1160x850+25+10")
    deadline=time.monotonic()+90
    while time.monotonic()<deadline and (not app.ready or app.worker.busy):
        window.update();time.sleep(.02)
    if not app.ready or app.worker.busy: raise RuntimeError("Native GUI environment readiness timed out")
    tabs=[app.notebook.tab(i,"text") for i in range(2)]
    if tabs!=["结果","推理过程"]: raise RuntimeError("Peer-page contract failed")
    before_thread=app.worker.thread
    pictures=[]
    for page,name in ((app.result_tab,"results"),(app.trace_tab,"trace")):
        app.notebook.select(page);window.update();time.sleep(.25);window.update()
        pictures.append(snapshot_window(window,REPO/f"docs/task8f_{phase}_{name}.png"))
    if app.worker.thread is not before_thread or calls: raise RuntimeError("Tab change/model-call contract failed")
    state={"pass":True,"phase":phase,"root":str(root),"cwd":str(Path.cwd()),"self_check":checked,"tabs":tabs,
           "gui_ready":True,"same_worker_on_tab_switch":True,"screenshots":pictures,"engine_adapter_initializes_without_weight_load":True,
           "scientific_calls":len(calls),"real_inference":False}
    app.close()
    import gc;del app,window;gc.collect()
    return state

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--phase",choices=("staging","final"),required=True);args=p.parse_args()
    root=Path(args.root).resolve()
    allowed={"staging":Path(r"C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Demo_V2_task8f_staging_v1"),"final":Path(r"C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Demo_V2")}
    if root != allowed[args.phase]: raise ValueError("Only explicitly authorized new V2 root may be validated")
    state=validate(root,args.phase)
    path=REPO/"evaluation/task8f_user_demo_live_trace_v1.json";e=json.loads(path.read_text(encoding="utf-8"));e[args.phase+"_validation"]=state;path.write_text(json.dumps(e,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"pass":state["pass"],"root":str(root),"files":state["self_check"]["package"]["checked"],"model_calls":0},ensure_ascii=False))
if __name__=="__main__":main()
