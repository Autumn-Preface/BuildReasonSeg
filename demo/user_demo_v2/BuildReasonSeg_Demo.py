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
TITLE = "BuildReasonSeg 建筑空间推理分割 Demo V2"


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
        root.geometry("1160x850")
        root.minsize(940, 760)
        root.configure(bg="#f3f5f7")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f3f5f7")
        style.configure("TLabel", background="#f3f5f7", font=("Microsoft YaHei UI", 10))
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=(12, 7))
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 20, "bold"), foreground="#17324d")
        style.configure("Hint.TLabel", foreground="#536777")
        style.configure("Run.TButton", font=("Microsoft YaHei UI", 12, "bold"), padding=(18, 9))
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)
        self.result_tab = ttk.Frame(self.notebook)
        self.trace_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.result_tab, text="结果")
        self.notebook.add(self.trace_tab, text="推理过程")
        body = ttk.Frame(self.result_tab, padding=(24, 18))
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=TITLE, style="Title.TLabel").pack(anchor="w")
        ttk.Label(body, text="选择遥感图片，用一句空间指令生成建筑 Mask。结果需要您结合原图核验。", style="Hint.TLabel").pack(anchor="w", pady=(6, 16))
        row = ttk.Frame(body)
        row.pack(fill="x")
        self.path_text = tk.StringVar(value="尚未选择图片")
        ttk.Label(row, textvariable=self.path_text, width=88).pack(side="left", fill="x", expand=True)
        self.select_button = ttk.Button(row, text="选择图片", command=self.select_image)
        self.select_button.pack(side="right")
        ttk.Label(body, text="空间指令").pack(anchor="w", pady=(16, 5))
        self.prompt = tk.Text(body, height=2, wrap="word", font=("Microsoft YaHei UI", 12),
                              relief="flat", borderwidth=1, padx=10, pady=8)
        self.prompt.pack(fill="x")
        examples = ttk.Frame(body)
        examples.pack(fill="x", pady=(8, 12))
        self.example_buttons = []
        for i, text in enumerate(("左侧最近建筑", "右侧最近建筑", "上方最近建筑", "下方最近建筑")):
            b = ttk.Button(examples, text=text, command=lambda i=i: self.fill_example(i))
            b.pack(side="left", padx=(0, 8))
            self.example_buttons.append(b)
        toolbar = ttk.Frame(body)
        toolbar.pack(fill="x")
        self.run_button = ttk.Button(toolbar, text="开始分析", style="Run.TButton", command=self.run)
        self.run_button.pack(side="left")
        self.check_button = ttk.Button(toolbar, text="环境检查", command=self.check_environment)
        self.check_button.pack(side="right")
        ttk.Button(toolbar, text="查看使用说明", command=self.open_help).pack(side="right", padx=8)
        self.status = tk.StringVar(value="正在检查本机运行环境……")
        ttk.Label(body, textvariable=self.status, style="Hint.TLabel").pack(anchor="w", pady=(10, 4))
        self.progress = ttk.Progressbar(body, mode="indeterminate")
        self.progress.pack(fill="x", pady=(0, 12))
        previews = ttk.Frame(body)
        previews.pack(fill="both", expand=True)
        previews.columnconfigure(0, weight=1)
        previews.columnconfigure(1, weight=1)
        previews.rowconfigure(1, weight=1)
        ttk.Label(previews, text="原始图片").grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Label(previews, text="分割结果 Overlay").grid(row=0, column=1, sticky="w", pady=(0, 6))
        self.original = tk.Label(previews, text="请先选择图片", bg="white", fg="#778893", font=("Microsoft YaHei UI", 12))
        self.overlay = tk.Label(previews, text="结果将在这里显示", bg="white", fg="#778893", font=("Microsoft YaHei UI", 12))
        self.original.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self.overlay.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        self.summary = tk.StringVar(value="当前支持：最大建筑 → 左侧 / 右侧 / 上方 / 下方 → 最近建筑")
        ttk.Label(body, textvariable=self.summary, wraplength=1090).pack(anchor="w", pady=(12, 7))
        footer = ttk.Frame(body)
        footer.pack(fill="x")
        self.mask_button = ttk.Button(footer, text="查看 Mask", command=self.view_mask)
        self.mask_button.pack(side="left")
        self.results_button = ttk.Button(footer, text="打开结果文件夹", command=self.open_results)
        self.results_button.pack(side="left", padx=8)
        self.new_button = ttk.Button(footer, text="重新选择图片", command=self.select_image)
        self.new_button.pack(side="right")
        self.detail_button = ttk.Button(footer, text="错误详情", command=self.show_detail)
        self.error_detail = ""
        self.controls = [self.select_button, self.new_button, self.run_button, self.check_button, *self.example_buttons]
        self.mask_button.state(["disabled"])
        self.results_button.state(["disabled"])
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
        self.image_path = Path(name).resolve()
        self.path_text.set(str(self.image_path))
        self.show_preview(self.original, "original", image)
        self.overlay.configure(image="", text="结果将在这里显示")
        self.photos.pop("overlay", None)
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
            self.mask_button.state(["!disabled"])
            task = interpretation(s["interpreted_program"]).replace("\n", "  ·  ")
            self.summary.set("指令：" + s["prompt"] + "\n" + task + "\n运行状态：流程完成  ·  语义正确性：未自动验证  ·  耗时：" + str(s["elapsed_seconds"]) + " 秒")
        else:
            self.status.set(s["error_message"])
            self.overlay.configure(image="", text="本次未生成可展示的分割结果")
            self.summary.set("运行状态：未完成。错误代码：" + s["error_code"] + "。未自动重试。")
            self.error_detail = "错误代码：" + s["error_code"] + "\n" + s["error_message"]
            self.detail_button.pack(side="left")

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
        self.tk.Label(top, image=top.photo, bg="white").pack(padx=16, pady=16)

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
    args = parser.parse_args()
    configure_environment(DEMO_ROOT)
    if args.self_check:
        result = self_check(DEMO_ROOT, full=True)
        print(result["message"])
        raise SystemExit(0 if result["ok"] else 1)
    import tkinter as tk
    root = tk.Tk()
    DemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
