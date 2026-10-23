# -*- coding: utf-8 -*-
"""fix_autostart.py —— 一次搞定「上电自启工作模式」

背景（2026-10-23 事故链）：
  1. `/sdcard` 根目录被 232 个文件占满 -> 板上写文件全部 OSError 30（已清理修复）；
  2. 修卡后上了个 main.py，把 `sys.stdout` 重定向到 `/data/agent.log`，
     结果这个固件对 **已存在的 /data 文件 open(...,'w') 会卡死整板**
     （`/data/main.py` 更是必卡），于是自启失败、板子假死；
  3. 代理每 60 帧 `print` 到 USB，没人读时写缓冲满了也会把 VM 卡死。

最终方案：main.py 不做任何文件 IO，只把 stdout/stderr 换成内存 sink（丢弃），
再 exec 板上 `gimbal_agent.py`。这样：
  * 上电自启稳定（无 /data 写入）；
  * USB 上没有任何周期性输出，不会被写缓冲堵死；
  * 需要日志时，手动 `exec(open('/sdcard/gimbal_agent.py').read())` 即可。

用法：
    python3 -m tools.fix_autostart --wait 900     # 等板子 -> 写 main.py -> 复位 -> 验证
    python3 -m tools.fix_autostart --verify       # 只做“复位后验证”
"""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from k230tool.link import find_port  # noqa: E402

MAIN_SRC = (
    "import sys\n"
    "class Sink:\n"
    "    def write(self, *a):\n"
    "        return 0\n"
    "    def flush(self):\n"
    "        pass\n"
    "try:\n"
    "    sys.stdout = Sink()\n"
    "    sys.stderr = Sink()\n"
    "except Exception:\n"
    "    pass\n"
    "exec(open('/sdcard/gimbal_agent.py').read())\n"
)


def ports():
    return sorted(glob.glob("/dev/ttyACM*"))


def open_acm(timeout=60.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        p = ports()
        if p:
            try:
                import serial
                return serial.Serial(p[0], 115200, timeout=0.4)
            except Exception:
                pass
        time.sleep(0.5)
    return None


def repl_reply(s, timeout=1.0):
    t0 = time.time()
    buf = b""
    while time.time() - t0 < timeout:
        buf += s.read(8192)
    return buf.decode("utf-8", "replace")


def wait_repl(timeout=600.0):
    """等到能打印 >>> 的 REPL。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        remain = max(1.0, min(15.0, timeout - (time.time() - t0)))
        s = open_acm(remain)
        if s is None:
            continue
        try:
            s.reset_input_buffer()
            s.write(b"\r\n")
            if ">>>" in repl_reply(s, 0.8):
                return s
            s.write(b"\x02\x03")          # Ctrl-B 退出 raw / 打断
            if ">>>" in repl_reply(s, 1.0):
                return s
            s.close()
        except Exception:
            try:
                s.close()
            except Exception:
                pass
        time.sleep(1.0)
    return None


def write_main(s):
    esc = MAIN_SRC.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    cmd = 'open("/sdcard/main.py","w").write("%s")' % esc
    s.write(cmd.encode() + b"\r\n")
    repl_reply(s, 1.5)
    s.write(b"import os;print('SZ', os.stat('/sdcard/main.py')[6])\r\n")
    out = repl_reply(s, 1.5)
    print("[fix] main.py:", "OK" if "SZ" in out else "写入未确认", out.strip()[-80:])


def esp(cmd):
    try:
        import serial
        s = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.15)
        s.write((cmd + "\n").encode())
        s.flush()
        s.close()
    except Exception:
        pass


def esp_frames():
    try:
        import serial
        s = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.15)
        s.reset_input_buffer()
        s.write(b"status\n")
        t0 = time.time()
        buf = b""
        while time.time() - t0 < 0.6:
            buf += s.read(8192)
        s.close()
        for line in buf.decode("utf-8", "replace").splitlines():
            if line.startswith("PAN="):
                for tok in line.split():
                    if tok.startswith("frames="):
                        return int(tok.split("=", 1)[1])
    except Exception:
        pass
    return None


def verify(seconds=14.0):
    esp("gnone")
    time.sleep(0.5)
    f1 = esp_frames()
    if f1 is None:
        print("[fix] ESP32 不可读，跳过验证")
        return False
    esp("gtrack")
    f2 = f1
    t0 = time.time()
    while time.time() - t0 < seconds:
        time.sleep(1.0)
        f = esp_frames()
        if f is not None:
            f2 = f
    esp("gnone")
    print("[fix] ESP32 frames: %d -> %d（%d 帧 / %.0fs）" % (f1, f2, f2 - f1, seconds))
    return f2 > f1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", type=float, default=900.0)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)

    if args.verify:
        return 0 if verify() else 1

    s = wait_repl(args.wait)
    if s is None:
        print("[fix] 等不到 REPL，请给板子断电再上电（拔插 USB）后重跑")
        return 2
    try:
        print("[fix] REPL 就绪，写 main.py（无文件 IO 版本）…")
        write_main(s)
        s.write(b"import machine;machine.reset()\r\n")
        time.sleep(0.5)
    finally:
        try:
            s.close()
        except Exception:
            pass
    time.sleep(6.0)
    ok = verify()
    print("[fix] 自启验证:", "通过" if ok else "未通过（看 ESP32 frames 是否增长）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
