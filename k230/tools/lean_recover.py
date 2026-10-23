# -*- coding: utf-8 -*-
"""lean_recover.py —— 在 /sdcard 只读、REPL 又脆弱时的最小恢复路径

背景（2026-10-23）：
  * `/sdcard` 变只读（OSError 30），官方 `mode work` 写不进 `/sdcard/main.py`；
  * 往 `/data/main.py` 写会直接触发固件 watcher，把整块板卡死（已复现两次）；
  * `/data` 可写，但大文件分块追加不可靠（实测只落了第一块 2048 字节）。

所以这里**不传大文件**：板上 `/sdcard/gimbal_agent.py` 本来就在（只读也能读），
只在 `/data` 写一个 50 字节的引导文件，然后 `exec` 它：

    /data/boot_agent.py:  exec(open('/sdcard/gimbal_agent.py').read())

用法：
    python3 -m tools.lean_recover --wait 900      # 等板子回来 -> 引导 -> 启动
    python3 -m tools.lean_recover --status        # 只读看板子/代理状态
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from k230tool.link import K230Link, find_port  # noqa: E402

BOOT = "exec(open('/sdcard/gimbal_agent.py').read())\n"
BOOT_PATH = "/data/boot_agent.py"


def wait_repl(seconds: float):
    t0 = time.time()
    last = ""
    while time.time() - t0 < seconds:
        port = find_port()
        if not port:
            time.sleep(1.0)
            continue
        try:
            link = K230Link(port=port)
            link.open()
            link.enter_raw(timeout=3.0)
            print("[lean] REPL 就绪:", port)
            return link
        except Exception as e:
            last = str(e)[:80]
            try:
                link.close()
            except Exception:
                pass
            time.sleep(2.0)
    print("[lean] %.0fs 内没等到 REPL（%s）" % (seconds, last))
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", type=float, default=300.0)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args(argv)

    if args.status:
        port = find_port()
        if not port:
            print("[lean] 没有 K230 串口")
            return 2
        link = K230Link(port=port)
        link.open()
        try:
            out, err, ok = link.exec_and_collect(
                "import os\n"
                "print('SD', [(f, os.stat('/sdcard/'+f)[6]) for f in "
                "('gimbal_agent.py','main.py') if f in os.listdir('/sdcard')])\n"
                "print('DATA', os.listdir('/data'))\n", timeout=15)
            print(out or err)
        finally:
            link.exit_raw()
            link.close()
        return 0

    link = wait_repl(args.wait)
    if link is None:
        return 2
    try:
        # 1) 写 50 字节引导文件（小文件在本固件上是可靠的）
        code = ("f=open(%r,'w')\nf.write(%r)\nf.close()\n"
                "g=open(%r)\nprint('READBACK', g.read())\ng.close()\n"
                % (BOOT_PATH, BOOT, BOOT_PATH))
        out, err, ok = link.exec_and_collect(code, timeout=15)
        good = "READBACK" in out and "gimbal_agent.py" in out
        print("[lean] 引导文件:", "OK" if good else "失败 %s %s" % (out[:100], err[:200]))
        if not good:
            return 1

        # 2) 普通 REPL 里启动（raw REPL 会被脚本占死，退不出来）
        link.exit_raw()
        time.sleep(0.3)
        link.ser.write(("exec(open(%r).read())\r\n" % BOOT_PATH).encode())
        print("[lean] 已启动，等待代理打印…")

        # 3) 被动读输出，确认代理起来
        t0 = time.time()
        buf = b""
        while time.time() - t0 < 8.0:
            chunk = link.ser.read(8192)
            if chunk:
                buf += chunk
        text = buf.decode("utf-8", "replace")
        for line in text.replace("\r", "\n").splitlines():
            if line.strip():
                print("   ", line.strip()[:120])
        ok_agent = ("常驻就绪" in text) or ("agent:" in text)
        print("[lean] 代理状态:", "已运行" if ok_agent else "未确认（可能仍在启动）")
        return 0 if ok_agent else 1
    finally:
        try:
            link.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
