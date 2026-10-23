"""Test-only V3 evidence window: replay real records or emit labelled fake failure.

Never shipped in the runtime package. No model loading or inference is allowed.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'demo/user_demo_v3'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['replay', 'failure'])
    p.add_argument('--size', default='1280x800')
    a = p.parse_args()
    # Importing torch is unnecessary here. These windows use persisted images only.
    import tkinter as tk
    from BuildReasonSeg_Demo import DemoApp
    from _ui.trace import reopen_trace
    root = tk.Tk()
    package = Path('C:/D/DeepSeekHarness/delivery/BuildReasonSeg_Demo_V3')
    if a.mode == 'replay':
        app = DemoApp(root, demo_root=package, start_check=False)
        trace = package / 'results/20261023_230642_1034/trace/trace_manifest.json'
        app.worker.start(lambda: reopen_trace(trace, package, app.worker.events))
        title = 'DEMO-R1 真实记录只读回放（不运行模型）'
    else:
        spec = importlib.util.spec_from_file_location('demo_r1_fixtures', REPO / 'tests/test_demo_r1.py')
        fixtures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixtures)
        owner = REPO / '.tmp/demo_r1_fake_evidence'
        owner.mkdir(parents=True, exist_ok=True)
        app = DemoApp(root, demo_root=owner, start_check=False)
        from _ui.backend import reserve_result
        from _ui.trace import TraceSession
        s = TraceSession(reserve_result(owner, Path('SIMULATION_ONLY.png')), app.worker.events)
        def fake():
            fixtures.fake_events(s, 8)
            s.emit(9, 'FAILED', {'runtime_status': 'FAILED', 'error_code': 'E404',
                'guard_executed': True, 'final_output_valid': False, 'reason': 'SIMULATION / 固定模拟事件，仅测试失败展示'})
            s.finish(result=fixtures.result())
        app.worker.start(fake)
        title = 'DEMO-R1 模拟事件 / E404 失败展示 / 非模型结果'
    root.title(title)
    root.geometry(a.size)
    app.notebook.select(app.trace_tab)
    root.after(1500, lambda: app.trace_page.jump(9 if a.mode == 'failure' else 6))
    root.mainloop()


if __name__ == '__main__':
    main()
