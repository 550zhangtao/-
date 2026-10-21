#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_laser_deploy.py —— 只读检查板端激光脚本版本/部署是否一致

用法（K230 必须在调试模式，REPL 空闲）：
    bash k230/scripts/mode.sh debug
    python3 k230/scripts/check_laser_deploy.py

输出：
    * PC 侧 laser_spot.py / gimbal_agent.py / yolo_preview_ide.py 的 md5；
    * 板端 /sdcard、/sdcard/board、/ 下所有 laser_spot.py 副本的 md5/大小；
    * 板端实际 import 的 laser_spot 版本/API/配置哈希/路径；
    * sys.path、laser_calib.json 内容。

这是 A（版本统一）的诊断入口；不改板端任何文件。
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from k230tool.link import BoardError, K230Link, find_port  # noqa: E402

PC_FILES = [
    ROOT / "board" / "laser_spot.py",
    ROOT / "board" / "gimbal_agent.py",
    ROOT / "board" / "yolo_preview_ide.py",
]
BOARD_FILES = [
    "/sdcard/laser_spot.py",
    "/sdcard/main.py",
    "/sdcard/yolo_preview_ide.py",
    "/sdcard/board/laser_spot.py",
    "/laser_spot.py",
]


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    args = ap.parse_args(argv)

    print("== PC ==")
    for p in PC_FILES:
        if p.is_file():
            print("  %-42s %8d  %s" % (p.name, p.stat().st_size, _md5(p)))
        else:
            print("  %-42s MISSING" % p.name)

    port = args.port or find_port(verbose=False)
    if not port:
        print("\n[!] 没有找到 K230 串口；检查 USB0 和 fuser -v /dev/ttyACM*")
        return 2
    link = K230Link(port=port, verbose=False)
    try:
        link.open()
    except BoardError as exc:
        print("\n[!] 连接失败：%s" % exc)
        print("    如果板子在“工作模式”，先：bash k230/scripts/mode.sh debug")
        return 2

    try:
        print("\n== BOARD /sdcard ==")
        for remote in BOARD_FILES:
            md5 = link._board_md5(remote)
            size = "?"
            try:
                out, _err, _ok = link.exec_and_collect(
                    "import os\nprint('SZ', os.stat(%r)[6])" % remote,
                    timeout=8)
                for line in out.replace("\r", "\n").split("\n"):
                    if line.startswith("SZ "):
                        size = line[3:].strip()
            except Exception:
                pass
            print("  %-34s %8s  %s" % (remote, size, md5 or "MISSING"))

        print("\n== import 实际版本 ==")
        out, _err, _ok = link.exec_and_collect(
            "import sys\n"
            "print('SYSPATH', sys.path)\n"
            "try:\n"
            "    import laser_spot\n"
            "    print('VERSION', getattr(laser_spot,'VERSION','?'))\n"
            "    print('API', getattr(laser_spot,'API_VERSION',0))\n"
            "    print('CFG', getattr(laser_spot,'CFG_HASH',0))\n"
            "    print('FILE', getattr(laser_spot,'__file__','?'))\n"
            "except Exception as e:\n"
            "    print('IMPORT_ERR', e)\n",
            timeout=15)
        print(out.strip() or "(no output)")

        print("\n== laser_calib.json ==")
        out, _err, _ok = link.exec_and_collect(
            "try:\n"
            "    f=open('/sdcard/laser_calib.json')\n"
            "    print(f.read())\n"
            "    f.close()\n"
            "except Exception as e:\n"
            "    print('CALIB_ERR', e)\n",
            timeout=10)
        print(out.strip() or "(no output)")

        print("\n[提示] 只读检查完成；工作模式切换：bash k230/scripts/mode.sh work")
        return 0
    finally:
        link.close()


if __name__ == "__main__":
    sys.exit(main())
