"""Main-thread-only, two-column scrolling stage cards shared with the result worker."""
from __future__ import annotations

import json
import threading

from .trace import STAGES, STATE_TEXT


def display_metadata(stage, metadata):
    keys = {
        1: ("original_prompt", "initial_program", "initial_supported", "initial_has_nearest", "confidence",
            "confidence_scope", "suggestion", "suggested_program", "suggestion_validator_pass", "user_confirmation", "final_program", "language_mode", "error_code"),
        2: ("raw_count", "merged_count", "detector_seconds", "proposal_ids", "error_code"),
        3: ("mode", "selected_id", "mask_area", "confidence", "bbox", "error_code"),
        4: ("context", "direction", "reference_pixels", "error_code", "reason"),
        5: ("encoding_complete", "feature_shape", "elapsed_seconds", "raw_feature_tensor_saved", "error_code"),
        6: ("source", "field_shapes", "error_code", "reason"),
        7: ("q_formed", "q_shape", "q_dimension", "C_shape", "similarity_scope", "source", "error_code"),
        8: ("candidate_only", "validation_state", "mask_pixels", "elapsed_seconds", "error_code"),
        9: ("runtime_status", "error_code", "reason", "guard_executed", "final_output_valid", "validity_scope", "semantic_status", "elapsed_seconds", "failure_scope"),
    }[stage]
    labels = {"original_prompt": "原始输入", "initial_program": "初始 program", "initial_supported": "初始受支持",
              "initial_has_nearest": "初始含 nearest", "confidence": "模型置信度（非正确率）", "confidence_scope": "置信度含义",
              "suggestion": "真实建议", "suggested_program": "建议 program", "suggestion_validator_pass": "建议通过 Validator",
              "user_confirmation": "用户确认", "final_program": "确认后 program", "language_mode": "语言模式",
              "raw_count": "Raw proposals", "merged_count": "Merged proposals", "detector_seconds": "检测耗时/秒",
              "proposal_ids": "全部 merged IDs", "mode": "模式", "selected_id": "实际 Reference ID", "mask_area": "Mask 面积",
              "bbox": "BBox [top,left,bottom,right]", "context": "实际 Context", "direction": "方向", "reference_pixels": "Context 参考像素",
              "encoding_complete": "视觉编码完成", "feature_shape": "真实特征形状", "elapsed_seconds": "耗时/秒",
              "raw_feature_tensor_saved": "保存原始特征张量", "source": "观测来源", "field_shapes": "原始场形状",
              "q_formed": "q 已形成", "q_shape": "q 形状", "q_dimension": "q 维度", "C_shape": "C 形状",
              "similarity_scope": "C 含义", "candidate_only": "仅中间候选", "validation_state": "结构验证状态",
              "mask_pixels": "候选 Mask 像素", "runtime_status": "运行状态", "error_code": "错误代码", "reason": "原因",
              "guard_executed": "最终结构检查已执行", "final_output_valid": "最终结构输出有效", "validity_scope": "有效性范围",
              "semantic_status": "语义正确性", "failure_scope": "失败范围"}
    lines = []
    for key in keys:
        if key in metadata:
            value = metadata[key]
            text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
            lines.append(labels.get(key, key) + "：" + text)
    if stage == 6 and metadata.get("display"):
        lines.append("图例列出原始数值范围和显示归一化；颜色不是原始概率。")
    return "\n".join(lines)


class TracePage:
    def __init__(self, parent, open_saved, busy):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.main_thread = threading.get_ident()
        self.run_id = None
        self.directory = None
        self.cards = {}
        self.photos = {}
        self.busy = busy
        toolbar = ttk.Frame(parent, padding=(18, 12))
        toolbar.pack(fill="x")
        self.banner = tk.StringVar(value="尚未运行。各阶段只展示同一次实际执行的状态，不代表 GT 验证。")
        ttk.Label(toolbar, textvariable=self.banner, wraplength=950).pack(side="left", fill="x", expand=True)
        self.open_button = ttk.Button(toolbar, text="打开已保存过程", command=open_saved)
        self.open_button.pack(side="right")
        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(container, bg="#f3f5f7", highlightthickness=0)
        scroll = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = ttk.Frame(self.canvas, padding=(16, 5, 16, 18))
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self.resize)
        self.inner.columnconfigure(0, weight=1, uniform="stage")
        self.inner.columnconfigure(1, weight=1, uniform="stage")
        for i, name in enumerate(STAGES, 1):
            card = ttk.LabelFrame(self.inner, text=f"{i}. {name}", padding=12)
            card.grid(row=(i-1)//2, column=(i-1)%2, sticky="new", padx=6, pady=8)
            state = tk.StringVar(value="未执行")
            text = tk.StringVar(value="本次运行尚未到达此阶段。")
            label = ttk.Label(card, textvariable=state, foreground="#536777")
            label.pack(anchor="w")
            details = ttk.Label(card, textvariable=text, wraplength=485, justify="left")
            details.pack(anchor="w", fill="x", pady=(6, 8))
            images = ttk.Frame(card)
            images.pack(fill="x")
            note = ""
            if i == 7:
                note = "C 只表示视觉相似度，不是目标正确概率。"
            elif i == 8:
                note = "尚未执行 D-B1，不展示候选 Mask。"
            elif i == 9:
                note = "SUCCESS 仅表示当前结构流程完成；目标身份与最近关系未自动验证。"
            note_value = tk.StringVar(value=note)
            ttk.Label(card, textvariable=note_value, wraplength=485, foreground="#925c30").pack(anchor="w", pady=(6, 0))
            self.cards[i] = {"frame": card, "state": state, "details": text, "label": label,
                             "images": images, "detail_widget": details, "snapshot_labels": [], "note": note_value}
            for widget in (card, label, details, images):
                widget.bind("<MouseWheel>", self.wheel)
        self.canvas.bind("<MouseWheel>", self.wheel)

    def _main(self):
        if threading.get_ident() != self.main_thread:
            raise RuntimeError("Tk updates are only allowed on the main thread")

    def resize(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width)
        wrap = max(280, (event.width-110)//2)
        for i, card in self.cards.items():
            card["detail_widget"].configure(wraplength=wrap)

    def wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta/120), "units")

    def set_busy(self, busy):
        self._main()
        self.open_button.state(["disabled"] if busy else ["!disabled"])

    def begin(self, data):
        self._main()
        self.run_id, self.directory = data["run_id"], data["directory"]
        prefix = "查看已保存过程" if data.get("reopened") else "本次运行"
        self.banner.set(prefix + "：" + self.run_id + "。过程记录不是 GT 验证。")
        self.photos.clear()
        for i, card in self.cards.items():
            card["state"].set("未执行")
            card["details"].set("本次运行尚未到达此阶段。")
            card["label"].configure(foreground="#536777")
            for label in card["snapshot_labels"]:
                label.destroy()
            card["snapshot_labels"].clear()
            if i == 8:
                card["note"].set("尚未执行 D-B1，不展示候选 Mask。")
        self.canvas.yview_moveto(0)

    def stage(self, data):
        self._main()
        event = data["event"]
        if event["run_id"] != self.run_id:
            return False                # never mix a delayed packet from another run
        stage = int(event["stage_id"])
        card = self.cards[stage]
        card["state"].set(STATE_TEXT[event["state"]])
        card["label"].configure(foreground={"FAILED": "#b83544", "COMPLETED": "#257254", "CANCELLED": "#925c30"}.get(event["state"], "#536777"))
        card["details"].set(display_metadata(stage, event["metadata"]) or "阶段状态已记录。")
        if stage == 8:
            card["note"].set("D-B1 已生成候选 Mask，但尚未通过最终结构检查。" if event["state"] == "COMPLETED" else "D-B1 尚未完成，不展示虚构候选 Mask。")
        if data.get("images"):
            from PIL import ImageTk
            for label in card["snapshot_labels"]:
                label.destroy()
            card["snapshot_labels"].clear()
            self.photos[stage] = []
            for image in data["images"]:
                # Worker has already decoded/rendered and bounded the display image.
                photo = ImageTk.PhotoImage(image, master=self.canvas)
                label = self.ttk.Label(card["images"], image=photo)
                label.pack(anchor="w", pady=4)
                label.bind("<MouseWheel>", self.wheel)
                label.bind("<Double-Button-1>", lambda event, image=image: self.zoom(image))
                self.photos[stage].append(photo)
                card["snapshot_labels"].append(label)
        return True

    def zoom(self, image):
        self._main()
        from PIL import ImageTk
        top = self.tk.Toplevel(self.canvas)
        top.title("只读阶段预览 / 不改变推理")
        top.photo = ImageTk.PhotoImage(image, master=top)
        self.ttk.Label(top, image=top.photo).pack(padx=12, pady=12)

    def error(self, data):
        self._main()
        if data["run_id"] == self.run_id:
            self.banner.set("过程观测失败：本次记录不完整，未自动重试。请保留当前结果文件夹。")

    def frozen(self, data):
        self._main()
        if data["run_id"] == self.run_id:
            self.banner.set(("过程已保存：" if data["observation_ok"] else "过程记录失败：") + self.run_id + "。已完成阶段继续保留；未执行阶段不补造。")
