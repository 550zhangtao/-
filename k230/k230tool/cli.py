# -*- coding: utf-8 -*-
"""
k230tool —— 统一命令行入口
==========================

    python3 -m k230tool probe                 # 体检
    python3 -m k230tool live                  # 实时桥接（写 live/latest.jpg）
    python3 -m k230tool live --no-ai          # 只看画面
    python3 -m k230tool snapshot -o a.jpg     # 取一张图（可复用 live）
    python3 -m k230tool status                # 看 live 状态
    python3 -m k230tool monitor               # OpenCV 窗口
    python3 -m k230tool grab -n 20            # 批量抓帧
    python3 -m k230tool serve --port 8080     # 浏览器 MJPEG
    python3 -m k230tool push 文件 --dest ...  # PC -> 板
    python3 -m k230tool pull /sdcard/... .    # 板 -> PC
    python3 -m k230tool models                # 列板端模型
    python3 -m k230tool mode debug            # 切 IDE 调试（REPL 空闲）
    python3 -m k230tool mode work             # 切板端工作（自动跑 main.py）
    python3 -m k230tool mode status           # 只读查看当前模式
    python3 -m k230tool reset                 # 真软复位（不拔插 USB）
"""

from __future__ import annotations

import argparse
import sys

from . import live as live_mod
from .link import ROOT

PASSTHROUGH = {
    "probe": ("probe", "全链路体检（连不上先跑这个）"),
    "monitor": ("monitor", "OpenCV 实时窗口"),
    "grab": ("grab", "批量抓帧到目录"),
    "serve": ("serve_mjpeg", "HTTP MJPEG 网络摄像头"),
    "push": ("push_file", "PC -> 板端文件"),
    "pull": ("pull_file", "板端 -> PC 文件"),
    "mode": ("mode", "调试/工作模式切换（debug/work/status/reset）"),
}


def _run_module(modname: str, argv) -> int:
    import importlib
    mod = importlib.import_module("." + modname, __package__)
    old = sys.argv
    sys.argv = ["k230tool " + modname] + list(argv)
    try:
        return int(mod.main() or 0)
    finally:
        sys.argv = old


def _add_snapshot(sub):
    p = sub.add_parser("snapshot", help="取一张图（优先复用 live，秒回）")
    p.add_argument("-o", "--out", default=str(ROOT / "live" / "snapshot.jpg"))
    p.add_argument("--live-dir", default=str(ROOT / "live"))
    p.add_argument("--max-age", type=float, default=5.0,
                   help="live/latest.jpg 多久内视为新鲜")
    p.add_argument("--no-reuse", action="store_true", help="强制新开流抓图")
    p.add_argument("--port", default=None)
    p.add_argument("--w", "--width", dest="width", type=int, default=640)
    p.add_argument("--h", "--height", dest="height", type=int, default=480)
    p.add_argument("--aw", type=int, default=0)
    p.add_argument("--ah", type=int, default=0)
    p.add_argument("--q", "--quality", dest="quality", type=int, default=70)
    p.add_argument("--fps", type=int, default=0)
    p.add_argument("--model", default=None)
    p.add_argument("--labels", default=None)
    p.add_argument("--labels-file", default=None)
    p.add_argument("--input-size", type=int, default=320)
    p.add_argument("--conf", type=float, default=0.66)
    p.add_argument("--nms", type=float, default=0.45)
    p.add_argument("--jpeg-every", type=int, default=1)
    p.add_argument("--full-every", type=int, default=15)
    p.add_argument("--roi-pad", type=float, default=0.25)
    p.add_argument("--roi-min", type=int, default=160)
    p.add_argument("--roi-max", type=int, default=256)
    p.add_argument("--no-ai", action="store_true")
    p.add_argument("--no-overlay", action="store_true")
    p.add_argument("--timeout", type=float, default=40.0)
    return p


def _add_status(sub):
    p = sub.add_parser("status", help="查看 live 状态（不占用串口）")
    p.add_argument("--live-dir", default=str(ROOT / "live"))
    return p


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="k230tool",
        description="K230 USB 工具链：摄像头回传 / YOLOv8 标靶识别",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--version", action="version", version="k230tool 1.0.0")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, (_mod, helptext) in PASSTHROUGH.items():
        p = sub.add_parser(name, help=helptext, add_help=False)
        p.add_argument("args", nargs=argparse.REMAINDER)

    # live（参数交给 live.py 自己的 parser）
    p = sub.add_parser("live", help="实时桥接：写 live/latest.jpg + latest.json",
                       add_help=False)
    p.add_argument("args", nargs=argparse.REMAINDER)

    _add_snapshot(sub)
    _add_status(sub)

    p = sub.add_parser("models", help="列出板端 /sdcard/models 与本地 models/")
    p.add_argument("--port", default=None)

    p = sub.add_parser(
        "reset",
        help="真软复位（machine.reset + 等待 USB 重连，不拔插）")
    p.add_argument("--port", default=None)
    return ap


def cmd_reset(port=None) -> int:
    # 复用 mode reset：先抢回 REPL，读当前模式，再 machine.reset() 并等 USB 恢复。
    # 旧实现只发 Ctrl-D（VM 软重启），不会清相机/KPU，工作模式下 main.py
    # 会立刻重新占住 REPL，看起来就像“软复位没反应、只能拔插 USB”。
    from . import mode as mode_mod
    argv = ["reset"] + (["--port", port] if port else [])
    return mode_mod.main(argv)


def cmd_models(port=None) -> int:
    from .link import K230Link, describe_ports, find_port
    print("本地 models/：")
    local = sorted((ROOT / "models").glob("*")) if (ROOT / "models").exists() else []
    if not local:
        print("  (空)")
    for p in local:
        if p.is_file():
            print("  %-48s %8.1f KB" % (p.name, p.stat().st_size / 1024.0))
    dev = port or find_port()
    if not dev:
        print("\n板端：未连接（%s）" % describe_ports().strip().replace("\n", " | "))
        return 0
    link = K230Link(port=dev).open()
    try:
        out, err, ok = link.exec_and_collect(
            "import os\n"
            "print('MODELS', os.listdir('/sdcard/models'))\n"
            "print('LIBS', os.listdir('/sdcard/libs'))\n", timeout=15)
        print("\n板端 /sdcard（%s）：" % dev)
        print(out.strip() or err.strip())
        return 0 if ok else 1
    finally:
        link.close()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        build_parser().print_help()
        return 0

    cmd, rest = argv[0], argv[1:]

    # 这几个命令的参数原样透传，避免 argparse 吃掉未知选项
    if cmd in PASSTHROUGH:
        return _run_module(PASSTHROUGH[cmd][0], rest)
    if cmd == "live":
        return live_mod.main(["live"] + rest)

    # snapshot / status / models 由 argparse 正常解析
    ap = build_parser()
    try:
        ns = ap.parse_args([cmd] + rest)
    except SystemExit as exc:
        return int(exc.code or 0)
    if ns.cmd == "snapshot":
        return live_mod.snapshot(ns)
    if ns.cmd == "status":
        return live_mod.status(ns)
    if ns.cmd == "models":
        return cmd_models(ns.port)
    if ns.cmd == "reset":
        return cmd_reset(ns.port)
    build_parser().print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
