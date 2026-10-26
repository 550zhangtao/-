# -*- coding: utf-8 -*-
"""
check_env.py —— 环境自检（训练/编译/串口）
=========================================

    python3 tools/check_env.py           # PC 环境 + 串口
    python3 tools/check_env.py --board   # 额外读板端版本/模型
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def check(name, fn):
    try:
        ok, detail = fn()
    except Exception as exc:
        ok, detail = False, "%s: %s" % (type(exc).__name__, exc)
    print(" %s %-20s %s" % ("✓" if ok else "✗", name, detail))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", action="store_true", help="读板端版本/模型")
    args = ap.parse_args()

    print("=" * 64)
    print(" K230 工具链环境自检")
    print("=" * 64)
    print("[PC]")
    check("Python", lambda: (sys.version_info >= (3, 7),
                             platform.python_version() + " " + platform.system()))
    for mod in ("serial", "numpy", "cv2"):
        def _m(m=mod):
            x = importlib.import_module(m)
            return True, getattr(x, "__version__", "?")
        check(mod, _m)

    print("\n[训练/转换]（不训练可忽略）")
    for mod in ("ultralytics", "torch", "onnx", "onnxruntime", "onnxsim", "onnxslim"):
        def _m(m=mod):
            x = importlib.import_module(m)
            return True, getattr(x, "__version__", "?")
        if mod == "onnxsim":
            try:
                _m()
            except Exception:
                print(" ! %-20s 未安装（可选；onnxslim 可替代）" % mod)
                continue
        check(mod, _m)

    def _dotnet():
        for c in (os.environ.get("DOTNET_ROOT"), os.path.expanduser("~/.dotnet"),
                  "/usr/share/dotnet"):
            if c and os.path.isdir(os.path.join(c, "host")):
                return True, c
        exe = shutil.which("dotnet")
        if exe:
            return True, exe
        return False, "未找到；安装 .NET 7 并 export DOTNET_ROOT"
    check("dotnet", _dotnet)

    def _nncase():
        # 必须在 import 前设置 DOTNET_ROOT / PLUGIN_PATH
        for c in (os.environ.get("DOTNET_ROOT"), os.path.expanduser("~/.dotnet"),
                  "/usr/share/dotnet"):
            if c and os.path.isdir(os.path.join(c, "host")):
                os.environ["DOTNET_ROOT"] = c
                break
        import importlib.util
        spec = importlib.util.find_spec("nncase")
        if spec and spec.origin:
            mod = Path(spec.origin).parent / "modules"
            if mod.is_dir():
                os.environ.setdefault("NNCASE_PLUGIN_PATH", str(mod))
        import nncase  # noqa: F401
        import _nncase
        return True, getattr(_nncase, "__version__", "?")
    check("nncase", _nncase)

    print("\n[串口]")
    try:
        from k230tool.link import describe_ports, find_port
        print(describe_ports())
        port = find_port()
        print(" %s %-20s %s" % ("✓" if port else "✗", "K230 端口", port or "未找到"))
    except Exception as exc:
        print(" ✗ 串口检查：%s" % exc)
        port = None

    print("\n[本地模型]")
    models = sorted((ROOT / "models").glob("*.kmodel"))
    if not models:
        print("  (models/ 为空，可从板子 pull 或训练后放进来)")
    for p in models:
        print("  %-40s %.2f MB" % (p.name, p.stat().st_size / 1048576.0))

    if args.board and port:
        print("\n[板端]")
        from k230tool.link import K230Link
        link = K230Link(port=port).open()
        try:
            out, err, ok = link.exec_and_collect(
                "import sys, os, gc\n"
                "print('VER', sys.version)\n"
                "print('BOARD', os.uname()[-1])\n"
                "print('MEM', gc.mem_free())\n"
                "print('MODELS', os.listdir('/sdcard/models'))\n", timeout=15)
            print(out.strip() or err.strip())
        finally:
            link.close()

    print("\n" + "=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
