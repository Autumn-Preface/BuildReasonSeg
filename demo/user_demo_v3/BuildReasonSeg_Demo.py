"""BuildReasonSeg local user Demo. Paths follow this file, independent of the working directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
from pathlib import Path

from _ui.environment import configure_environment, probe_runtime
from _ui.backend import (DIRECTIONS, EXAMPLES, ERROR_MESSAGES, SUCCESS_TEXT, Worker,
                         interpretation, preview_image, run_request)

DEMO_ROOT = Path(__file__).resolve().parent
TITLE = "BuildReasonSeg 建筑空间推理分割 Demo V3"


def package_check(root: Path, full=False) -> dict:
    root = Path(root).resolve()
    path = root / "_engine/package_manifest.json"
    if not path.is_file():
        return {"ok": False, "message": "Demo 文件不完整：缺少文件清单。"}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    for item in manifest["files"]:
        target = (root / item["path"]).resolve()
        if not target.is_relative_to(root) or not target.is_file() or target.stat().st_size != item["bytes"]:
            failures.append(item["path"])
            continue
        if full:
            digest = hashlib.sha256()
            with target.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            if digest.hexdigest() != item["sha256"]:
                failures.append(item["path"])
    return {"ok": not failures, "message": "文件检查：通过" if not failures else "Demo 文件校验未通过，请恢复完整的 Demo 文件夹。",
            "checked": len(manifest["files"]), "full_hash": full, "failures": failures}


def self_check(root: Path, full=True) -> dict:
    environment = probe_runtime(root)
    package = package_check(root, full=full)
    return {"ok": environment["ok"] and package["ok"],
            "message": "环境检查：通过" if environment["ok"] and package["ok"] else environment["message"] + "\n" + package["message"],
            "runtime": environment, "package": package,
            "paths": {name: str(Path(root).resolve() / name) for name in ("_engine", "results", "_runtime_cache")},
            "model_calls": 0}


class DemoApp:
    def __init__(self, root, demo_root=DEMO_ROOT, adapter_factory=None, start_check=True):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk, self.window = tk, ttk, root
        self.root = Path(demo_root).resolve()
        configure_environment(self.root)
        self.worker = Worker()
        self.adapter_factory = adapter_factory
        self.image_path = None
        self.published = None
        self.ready = False
        self.photos = {}
        root.title(TITLE)
        root.geometry("1440x900")
        root.minsize(1180, 760)
        from _ui.theme import apply_theme
        apply_theme(root)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=16, pady=12)
        self.result_tab = ttk.Frame(self.notebook)
        self.trace_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.result_tab, text="结果")
        self.notebook.add(self.trace_tab, text="推理过程")
        shell = ttk.Frame(self.result_tab, padding=20)
        shell.pack(fill="both", expand=True)
        ttk.Label(shell, text="BUILDREASONSEG  /  空间推理实验室", style="Hint.TLabel").pack(anchor="w")
        ttk.Label(shell, text="从一句指令，到可追溯的建筑分割", style="Title.TLabel").pack(anchor="w", pady=(8,18))
        body=ttk.Frame(shell);body.pack(fill="both",expand=True)
        sidebar_host=ttk.Frame(body,width=340);sidebar_host.pack(side="left",fill="y",padx=(0,16));sidebar_host.pack_propagate(False)
        sidebar_canvas=tk.Canvas(sidebar_host,bg="#0b1423",highlightthickness=0)
        sidebar_scroll=ttk.Scrollbar(sidebar_host,orient="vertical",command=sidebar_canvas.yview);sidebar_scroll.pack(side="right",fill="y")
        sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set);sidebar_canvas.pack(side="left",fill="both",expand=True)
        sidebar=ttk.Frame(sidebar_canvas);sidebar_id=sidebar_canvas.create_window((0,0),window=sidebar,anchor="nw")
        sidebar.bind("<Configure>",lambda e:sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all")))
        sidebar_canvas.bind("<Configure>",lambda e:sidebar_canvas.itemconfigure(sidebar_id,width=e.width))
        ttk.Label(sidebar,text="01  选择影像",style="Heading.TLabel").pack(anchor="w")
        self.path_text=tk.StringVar(value="尚未选择图片")
        ttk.Label(sidebar,textvariable=self.path_text,wraplength=300).pack(fill="x",pady=10)
        self.select_button=ttk.Button(sidebar,text="选择 RGB 影像",command=self.select_image);self.select_button.pack(fill="x")
        ttk.Label(sidebar,text="02  描述目标",style="Heading.TLabel").pack(anchor="w",pady=(24,10))
        self.prompt=tk.Text(sidebar,height=4,wrap="word",font=("Microsoft YaHei UI",12),bg="#16243a",fg="#e4ecfa",insertbackground="#67ddce",relief="flat",padx=12,pady=12)
        self.prompt.pack(fill="x")
        self.example_buttons=[]
        for i,text in enumerate(("左侧最近建筑","右侧最近建筑","上方最近建筑","下方最近建筑")):
            b=ttk.Button(sidebar,text=text,command=lambda i=i:self.fill_example(i));b.pack(fill="x",pady=(6,0));self.example_buttons.append(b)
        self.run_button=ttk.Button(sidebar,text="开始分析  →",style="Run.TButton",command=self.run);self.run_button.pack(fill="x",pady=(20,10))
        self.check_button=ttk.Button(sidebar,text="环境检查",command=self.check_environment);self.check_button.pack(fill="x")
        ttk.Button(sidebar,text="使用说明",command=self.open_help).pack(fill="x",pady=6)
        self.status=tk.StringVar(value="正在检查本机运行环境……")
        ttk.Label(sidebar,textvariable=self.status,wraplength=300,style="Hint.TLabel").pack(fill="x",pady=10)
        self.progress=ttk.Progressbar(sidebar,mode="indeterminate");self.progress.pack(fill="x")
        right=ttk.Frame(body);right.pack(side="left",fill="both",expand=True)
        self.preview_tabs=ttk.Notebook(right);self.preview_tabs.pack(fill="both",expand=True)
        for attr,title in (("original","原始影像"),("overlay","最终 Overlay"),("mask_preview","最终 Mask")):
            pane=ttk.Frame(self.preview_tabs);self.preview_tabs.add(pane,text=title)
            label=tk.Label(pane,text="尚无可展示内容",bg="#101c2e",fg="#9aacc7",font=("Microsoft YaHei UI",13),cursor="hand2")
            label.pack(fill="both",expand=True);setattr(self,attr,label)
            label.bind("<Button-1>",lambda e,key=attr:self.zoom_result(key))
        ttk.Label(right,text="点击图片放大 · 候选分割请到推理过程查看 · 最终结果须人工核验",style="Hint.TLabel").pack(anchor="w",pady=10)
        self.summary=tk.StringVar(value="支持：最大建筑 → 方向关系 → 最近建筑。语义正确性未自动验证。")
        ttk.Label(right,textvariable=self.summary,wraplength=760).pack(anchor="w",fill="x",pady=10)
        footer=ttk.Frame(right);footer.pack(fill="x")
        self.mask_button=ttk.Button(footer,text="查看 Mask",command=self.view_mask);self.mask_button.pack(side="left")
        self.results_button=ttk.Button(footer,text="打开结果文件夹",command=self.open_results);self.results_button.pack(side="left",padx=8)
        self.new_button=ttk.Button(footer,text="重新选图",command=self.select_image);self.new_button.pack(side="right")
        self.detail_button=ttk.Button(footer,text="错误详情",command=self.show_detail);self.error_detail=""
        self.controls=[self.select_button,self.new_button,self.run_button,self.check_button,*self.example_buttons]
        self.mask_button.state(["disabled"]);self.results_button.state(["disabled"])
        from _ui.trace_ui import TracePage
        self.trace_page = TracePage(self.trace_tab, self.open_trace, lambda: self.worker.busy)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.report_callback_exception = self.callback_error
        self.poll_id = root.after(80, self.poll)
        if start_check:
            self.check_environment(full=False)
        else:
            self.ready = True
            self.status.set("请选择图片并输入指令。")

    def open_trace(self):
        if self.worker.busy:
            return
        from tkinter import filedialog, messagebox
        name = filedialog.askopenfilename(parent=self.window, title="打开已保存推理过程",
            initialdir=str(self.root / "results"), filetypes=[("过程记录", "trace_manifest.json")])
        if not name:
            return
        self.set_busy(True)
        def action():
            from _ui.trace import reopen_trace
            try:
                reopen_trace(Path(name), self.root, self.worker.events)
            except Exception:
                self.worker.events.put(("trace_open_failed", None))
        self.worker.start(action)

    def set_busy(self, busy):
        self.trace_page.set_busy(busy)
        for b in self.controls:
            b.state(["disabled"] if busy else ["!disabled"])
        self.prompt.configure(state="disabled" if busy else "normal")
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()
            if not self.ready:
                self.run_button.state(["disabled"])

    def fill_example(self, index):
        if self.worker.busy:
            return
        self.prompt.delete("1.0", "end")
        self.prompt.insert("1.0", EXAMPLES[index])  # text only; no program binding or inference

    def prefill_prompt(self, text):
        previous = self.prompt.cget("state")
        self.prompt.configure(state="normal")
        self.prompt.delete("1.0", "end")
        self.prompt.insert("1.0", text)
        self.prompt.configure(state=previous)

    def show_preview(self, label, key, image):
        from PIL import ImageTk
        photo = ImageTk.PhotoImage(image, master=self.window)
        self.photos[key] = photo
        label.configure(image=photo, text="")

    def select_image(self):
        if self.worker.busy:
            return
        from tkinter import filedialog, messagebox
        name = filedialog.askopenfilename(parent=self.window, title="选择 RGB 遥感图片",
            filetypes=[("RGB 光学图片", "*.png *.jpg *.jpeg *.tif *.tiff"), ("PNG", "*.png"),
                       ("JPG / JPEG", "*.jpg *.jpeg"), ("TIF / TIFF", "*.tif *.tiff")])
        if not name:
            return
        try:
            image = preview_image(self.root, Path(name))
        except Exception as e:
            code = getattr(e, "code", "E202")
            messagebox.showerror("图片无法使用", ERROR_MESSAGES.get(code, ERROR_MESSAGES["E202"]), parent=self.window)
            return
        self.load_image_path(Path(name), image)

    def load_image_path(self, name, image=None):
        if image is None:
            image=preview_image(self.root,Path(name))
        self.image_path = Path(name).resolve()
        self.path_text.set(str(self.image_path))
        self.show_preview(self.original, "original", image)
        self.overlay.configure(image="", text="结果将在这里显示")
        self.photos.pop("overlay", None)
        self.mask_preview.configure(image="", text="尚无最终 Mask")
        self.photos.pop("mask_preview", None)
        self.published = None
        self.mask_button.state(["disabled"])
        self.results_button.state(["disabled"])
        self.summary.set("图片已选择。输入指令，或点击示例填入文字。")
        self.status.set("准备就绪。")

    def check_environment(self, full=True):
        if self.worker.busy:
            return
        self.ready = False
        self.set_busy(True)
        self.status.set("正在检查运行环境与 Demo 文件，请稍候……")
        def action():
            self.worker.events.put(("check", self_check(self.root, full=full)))
        if not self.worker.start(action):
            self.set_busy(self.worker.busy)

    def run(self):
        if self.worker.busy or not self.ready:
            return
        from tkinter import messagebox
        prompt = self.prompt.get("1.0", "end").strip()
        if self.image_path is None:
            messagebox.showinfo("请选择图片", "请先选择一张 RGB 遥感图片。", parent=self.window)
            return
        if not prompt:
            messagebox.showinfo("请输入指令", "请描述最大建筑某个方向上最近的建筑，或点击示例。", parent=self.window)
            return
        image = self.image_path
        self.published = None
        self.mask_preview.configure(image="", text="等待最终结构检查")
        self.photos.pop("mask_preview", None)
        self.overlay.configure(image="", text="正在分析，请稍候……")
        self.mask_button.state(["disabled"])
        self.results_button.state(["disabled"])
        self.detail_button.pack_forget()
        self.set_busy(True)
        action = lambda: run_request(self.root, image, prompt, self.worker, **(
            {"adapter_factory": self.adapter_factory} if self.adapter_factory else {}))
        if not self.worker.start(action):
            self.set_busy(self.worker.busy)

    def poll(self):
        from tkinter import messagebox
        try:
            while True:
                kind, value = self.worker.events.get_nowait()
                if kind == "confirmation":
                    value.accepted = messagebox.askyesno("请确认系统理解", value.message, parent=self.window, default="no")
                    value.event.set()
                elif kind == "progress":
                    text, program = value
                    self.status.set(text)
                    if program in DIRECTIONS:
                        self.summary.set(interpretation(program).replace("\n", "  ·  "))
                elif kind == "check":
                    self.ready = value["ok"]
                    self.status.set(value["message"])
                    if not self.ready:
                        self.error_detail = value["message"]
                        self.detail_button.pack(side="left")
                elif kind == "trace_start":
                    self.trace_page.begin(value)
                elif kind == "stage_event":
                    self.trace_page.stage(value)
                elif kind == "trace_error":
                    self.trace_page.error(value)
                elif kind == "trace_frozen":
                    self.trace_page.frozen(value)
                elif kind == "trace_open_failed":
                    self.status.set("无法打开过程记录，请检查记录是否完整；不会重新推理。")
                elif kind == "result":
                    self.present_result(value)
                elif kind == "worker_error":
                    self.status.set(ERROR_MESSAGES.get(value, ERROR_MESSAGES["E502"]))
                    self.error_detail = "错误代码：" + value + "\n" + self.status.get()
                    self.detail_button.pack(side="left")
                elif kind == "idle":
                    self.set_busy(False)
        except queue.Empty:
            pass
        self.poll_id = self.window.after(80, self.poll)

    def present_result(self, published):
        self.published = published
        s = published["summary"]
        self.results_button.state(["!disabled"])
        if s["runtime_status"] == "SUCCESS":
            self.status.set(SUCCESS_TEXT)
            from PIL import Image
            with Image.open(Path(published["directory"]) / s["overlay_file"]) as im:
                image = im.convert("RGB")
                image.thumbnail((530, 360), Image.Resampling.LANCZOS)
            self.show_preview(self.overlay, "overlay", image)
            with Image.open(Path(published["directory"]) / s["mask_file"]) as im:
                mask_image=im.convert("RGB");mask_image.thumbnail((700,470),Image.Resampling.NEAREST)
            self.show_preview(self.mask_preview,"mask_preview",mask_image)
            self.preview_tabs.select(1)
            self.mask_button.state(["!disabled"])
            task = interpretation(s["interpreted_program"]).replace("\n", "  ·  ")
            self.summary.set("指令：" + s["prompt"] + "\n" + task + "\n运行状态：流程完成  ·  语义正确性：未自动验证  ·  耗时：" + str(s["elapsed_seconds"]) + " 秒")
        else:
            self.status.set(s["error_message"])
            self.overlay.configure(image="", text="本次未生成可展示的分割结果")
            self.summary.set("运行状态：未完成。错误代码：" + s["error_code"] + "。未自动重试。")
            self.error_detail = "错误代码：" + s["error_code"] + "\n" + s["error_message"]
            self.detail_button.pack(side="left")

    def zoom_result(self, key):
        from PIL import Image
        path=self.image_path if key=="original" else (Path(self.published["directory"])/("mask.png" if key=="mask_preview" else "overlay.png") if self.published and self.published["summary"].get("runtime_status")=="SUCCESS" else None)
        if path and Path(path).is_file():
            with Image.open(path) as im:
                image=im.convert("RGB");image.thumbnail((1400,950))
            self.trace_page.zoom(image)

    def view_mask(self):
        if not self.published or not self.published["summary"].get("mask_file"):
            return
        from PIL import Image, ImageTk
        top = self.tk.Toplevel(self.window)
        top.title("Mask：白色为模型生成的目标区域")
        with Image.open(Path(self.published["directory"]) / "mask.png") as im:
            image = im.convert("RGB")
            image.thumbnail((850, 650), Image.Resampling.NEAREST)
        top.photo = ImageTk.PhotoImage(image, master=top)
        self.tk.Label(top, image=top.photo, bg="#101c2e").pack(padx=16, pady=16)

    def open_results(self):
        if self.published:
            os.startfile(self.published["directory"])

    def open_help(self):
        os.startfile(str(self.root / "使用说明.md"))

    def show_detail(self):
        from tkinter import messagebox
        messagebox.showinfo("错误详情", self.error_detail, parent=self.window)

    def close(self):
        if self.worker.busy:
            from tkinter import messagebox
            messagebox.showinfo("当前操作尚未结束", "请等待当前操作完成后再关闭。程序不会强行终止正在进行的分析。", parent=self.window)
            return
        self.progress.stop()
        self.window.after_cancel(self.poll_id)
        # Release Tk-owned references on their owner thread before later worker-side GC.
        self.photos.clear()
        self.trace_page.photos.clear()
        self.trace_page.cards.clear()
        self.trace_page.banner = None
        self.trace_page.busy = None
        self.path_text = self.status = self.summary = None
        self.window.destroy()
        self.window.report_callback_exception = None

    def callback_error(self, kind, error, trace):
        import traceback
        log_dir = self.root / "_engine/logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / "demo_runtime.log").open("a", encoding="utf-8") as handle:
            traceback.print_exception(kind, error, trace, file=handle)
        self.status.set(ERROR_MESSAGES["E502"])
        self.error_detail = "错误代码：E502\n" + ERROR_MESSAGES["E502"]
        self.detail_button.pack(side="left")


def main():
    parser = argparse.ArgumentParser(description="BuildReasonSeg 本地 Demo 环境检查", allow_abbrev=False)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--image", type=Path, help="仅预填影像，不启动推理")
    parser.add_argument("--prompt", default="", help="仅预填文字，仍需用户确认")
    args = parser.parse_args()
    configure_environment(DEMO_ROOT)
    if args.self_check:
        result = self_check(DEMO_ROOT, full=True)
        print(result["message"])
        raise SystemExit(0 if result["ok"] else 1)
    import tkinter as tk
    root = tk.Tk()
    app=DemoApp(root)
    if args.image:
        app.load_image_path(args.image)
    if args.prompt:
        app.prefill_prompt(args.prompt)
    root.mainloop()


if __name__ == "__main__":
    main()
