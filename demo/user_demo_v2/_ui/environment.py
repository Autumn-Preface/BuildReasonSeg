"""Offline environment check, private caches and deterministic Python discovery."""
from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

REQUIRED_ULTRALYTICS = "8.4.164"
DEPENDENCIES = ("torch", "ultralytics", "transformers", "numpy", "PIL", "cv2", "scipy",
                "yaml", "torchvision", "accelerate", "safetensors", "peft", "sam2", "hydra", "omegaconf", "tkinter")
MINIMUMS = {"torch": (2, 4), "transformers": (4, 49), "numpy": (1, 26),
            "PIL": (10, 0), "cv2": (4, 9), "scipy": (1, 11)}


def configure_environment(root: Path) -> dict:
    root = Path(root).resolve()
    cache = root / "_runtime_cache"
    # Create first, then configure; imports can now only use existing writable cache locations.
    for name in ("yolo", "matplotlib"):
        directory = cache / name
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / (".probe_" + str(os.getpid()))
        with probe.open("x", encoding="utf-8") as f:
            f.write("ok")
        probe.unlink()
    settings = {"YOLO_CONFIG_DIR": str(cache / "yolo"), "MPLCONFIGDIR": str(cache / "matplotlib"),
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    os.environ.update(settings)
    sys.dont_write_bytecode = True
    return settings


def version_pair(text: str) -> tuple:
    return tuple(int(n) for n in re.findall(r"\d+", text)[:2])


def probe_runtime(root: Path, importer=importlib.import_module) -> dict:
    settings = configure_environment(root)
    errors, versions = [], {"Python": ".".join(map(str, sys.version_info[:3]))}
    if sys.version_info < (3, 10):
        errors.append("需要 Python 3.10 或更新版本。")
    for name in DEPENDENCIES:
        try:
            module = importer(name)
            version = str(getattr(module, "__version__", getattr(module, "TkVersion", "available")))
            versions[name] = version
            if name == "ultralytics" and version != REQUIRED_ULTRALYTICS:
                errors.append("Ultralytics 版本不兼容，需要 " + REQUIRED_ULTRALYTICS + "。")
            if name in MINIMUMS and version_pair(version) < MINIMUMS[name]:
                errors.append(name + " 版本不兼容。")
            if name == "transformers" and not hasattr(module, "Qwen3VLForConditionalGeneration"):
                errors.append("Transformers 不支持所需的本地语言模型。")
        except Exception:
            errors.append("无法使用 " + name + "，请使用已配置好的运行环境。")
    return {"ok": not errors, "message": "环境检查：通过" if not errors else "环境检查未通过：" + "；".join(errors),
            "errors": errors, "versions": versions, "python": sys.executable, "cache_environment": settings,
            "model_calls": 0}


def python_candidates(root: Path, env=None, which=shutil.which) -> list[tuple[str, str]]:
    """The configured Python path is a machine environment locator, never a scientific source path."""
    root = Path(root).resolve()
    env = os.environ if env is None else env
    config = root / "_ui/runtime.json"
    configured = json.loads(config.read_text(encoding="utf-8")).get("validated_python", "") if config.is_file() else ""
    possibilities = [("BUILDREASONSEG_PYTHON", env.get("BUILDREASONSEG_PYTHON", "")),
                     ("bundled_future", str(root / "runtime/python.exe")),
                     ("validated_machine_environment", configured), ("PATH", which("python") or "")]
    seen, result = set(), []
    for source, value in possibilities:
        if value and value.lower() not in seen:
            seen.add(value.lower())
            result.append((source, value))
    return result


def discover_python(root: Path, checker, env=None, which=shutil.which) -> dict:
    attempts = []
    for source, path in python_candidates(root, env=env, which=which):
        if not Path(path).is_file():
            attempts.append({"source": source, "python": path, "ok": False, "reason": "not_found"})
            continue
        try:
            ok = bool(checker(path))
        except Exception:
            ok = False
        attempts.append({"source": source, "python": path, "ok": ok})
        if ok:
            return {"ok": True, "python": path, "source": source, "attempts": attempts}
    return {"ok": False, "attempts": attempts,
            "message": "未找到兼容的 Python 运行环境。请指定已配置好的 BUILDREASONSEG_PYTHON 后重新启动；本程序不会安装依赖。"}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    # Imported library output is internal; the probe emits just its own structured result.
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = probe_runtime(root)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["ok"] else 1)
