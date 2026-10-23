# -*- coding: utf-8 -*-
"""board_watch.py —— PC 侧“看门狗”：K230 每次上电/复位自动把代理拉起来

为什么需要：`/sdcard` 只读后，官方 `mode.sh work`（写 /sdcard/main.py）不可用；
板上电只会停在 REPL。既然 USB 供电本来就要连 PC，就让 PC 在板子重新出现时
自动执行一次最小引导（`tools/lean_recover.py`），等效于“上电自启工作模式”。

行为：
  * 监听 /sys 里 1209:abd1 的 devnum；板子重新枚举 -> 等 10s 开机 -> 探测 REPL；
  * 只在 REPL **空闲**（能打印 >>>）时才引导，避免打断已在跑的代理；
  * 板子不在/卡死时只记录，不反复重试大块写（防止再拖挂 REPL）。

用法：
    python3 -m tools.board_watch                 # 前台
    nohup python3 -m tools.board_watch > live/watch.log 2>&1 &   # 后台

日志里能看到 `[watch] launch ok/失败` 与代理启动打印。
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from k230tool.link import find_port  # noqa: E402

SYS_USB = "/sys/bus/usb/devices"


def k230_devnum():
    for vendor in glob.glob(SYS_USB + "/*/idVendor"):
        try:
            if open(vendor).read().strip() != "1209":
                continue
            base = os.path.dirname(vendor)
            return (open(base + "/busnum").read().strip(),
                    open(base + "/devnum").read().strip())
        except Exception:
            continue
    return None


def _esp_frames():
    """读 ESP32 的 frames 计数；读不到返回 None（ESP32 可能没插）。"""
    try:
        import serial
        s = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.1)
    except Exception:
        return None
    try:
        s.reset_input_buffer()
        s.write(b"status\n")
        t0 = time.time()
        buf = b""
        while time.time() - t0 < 0.6:
            buf += s.read(4096)
        for line in buf.decode("utf-8", "replace").splitlines():
            if line.startswith("PAN="):
                for tok in line.split():
                    if tok.startswith("frames="):
                        return int(tok.split("=", 1)[1])
    except Exception:
        pass
    finally:
        try:
            s.close()
        except Exception:
            pass
    return None


def _esp_cmd(cmd):
    try:
        import serial
        s = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.1)
        s.write((cmd + "\n").encode())
        s.flush()
        s.close()
    except Exception:
        pass


def k230_streaming(seconds: float = 3.0) -> str:
    """用 ESP32 的 frames 计数判断 K230 是否在推 $KT（不会打断 K230）。

    返回 'streaming' / 'idle' / 'unknown'（ESP32 不在）。
    """
    f1 = _esp_frames()
    if f1 is None:
        return "unknown"
    _esp_cmd("gtrack")
    t0 = time.time()
    f2 = f1
    while time.time() - t0 < seconds:
        time.sleep(0.5)
        f = _esp_frames()
        if f is not None:
            f2 = f
    if f2 > f1:
        return "streaming"            # 已在跑，别动它（保持 TRACK）
    _esp_cmd("gnone")                  # 没在跑，恢复空闲
    return "idle"


def launch(timeout: float = 120.0) -> int:
    return subprocess.run(
        [sys.executable, "-m", "tools.lean_recover", "--wait", str(timeout)],
        cwd=str(ROOT)).returncode


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--settle", type=float, default=10.0, help="上电后等待秒数")
    ap.add_argument("--probe-every", type=float, default=2.0)
    ap.add_argument("--once", action="store_true", help="只检查一次（启动时用）")
    args = ap.parse_args(argv)

    last = None
    print("[watch] 启动，等待 K230（1209:abd1）…", flush=True)
    while True:
        cur = k230_devnum()
        if cur is None:
            if last is not None:
                print("[watch] K230 掉线", flush=True)
            last = None
        elif cur != last:
            print("[watch] K230 出现/重新枚举 %s，等 %.0fs 开机…" % (cur, args.settle),
                  flush=True)
            time.sleep(args.settle)
            st = k230_streaming()
            print("[watch] K230 推流探测: %s" % st, flush=True)
            if st in ("idle", "unknown"):
                rc = launch()
                print("[watch] 引导返回 %d" % rc, flush=True)
            else:
                print("[watch] 代理已在推流，跳过", flush=True)
            last = cur
            if args.once:
                return 0
        time.sleep(args.probe_every)


if __name__ == "__main__":
    raise SystemExit(main())
