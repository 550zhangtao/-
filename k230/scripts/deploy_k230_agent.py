#!/usr/bin/env python3
# --- PC 端脚本守卫：误在 K230 上运行时给出明确提示 ---
try:
    import sys as _sys
    if getattr(_sys, "implementation", None) is not None and _sys.implementation.name == "micropython":
        print("[PC 脚本] 这是电脑端工具，请在 PC 上运行；板端请运行 k230/board/*.py")
        raise SystemExit
except SystemExit:
    raise
except Exception:
    pass
# -*- coding: utf-8 -*-
"""deploy_k230_agent.py —— 断电重上电后：抢 REPL -> 部署 agent -> 先交互验证 -> 再设自启动

阶段：
  1. 反复尝试抢 REPL（板子卡死时只能靠断电重上电）
  2. 删掉 /sdcard/main.py（坏掉的自动运行脚本）
  3. 用 k230tool 把 board/gimbal_agent.py 推到 /sdcard/gimbal_agent.py
  4. **先在 REPL 里交互跑一遍**，同时看 ESP32 有没有收到 $KT（不通过就不设自启动）
  5. 验证通过后复制成 /sdcard/main.py，复位，确认上电自动运行

用法：python3 -u scratch/deploy_k230_agent.py [等待秒数]
"""
import glob
import subprocess
import sys
import time

import serial

import os as _os
# 本脚本现在位于 k230/scripts/，工具包根就是它的上一级
TOOLKIT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
SRC_LOCAL = _os.path.join(TOOLKIT, "board", "gimbal_agent.py")
PK_LOCAL = _os.path.join(TOOLKIT, "board", "point_kalman.py")
SRC = open(SRC_LOCAL, encoding="utf-8").read()
WAIT_S = float(sys.argv[1]) if len(sys.argv) > 1 else 900.0


def log(*a):
    print(*a, flush=True)


def esp_read(seconds):
    try:
        e = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.2)
    except Exception as exc:
        return "(打不开 ESP32 串口: %s)" % exc
    try:
        e.dtr = False
        e.rts = False
    except Exception:
        pass
    e.reset_input_buffer()
    t0 = time.time()
    buf = b""
    while time.time() - t0 < seconds:
        buf += e.read(4096)
    e.close()
    return buf.decode("utf-8", "replace")


def frames_seen():
    out = esp_read(2.5)
    for line in out.splitlines():
        if "[GIMBAL]" in line and "frames=" in line:
            try:
                return int(line.split("frames=")[1].split()[0]), line.strip()
            except Exception:
                pass
    return None, ""


def try_catch_repl():
    ports = sorted(glob.glob("/dev/ttyACM*"))
    if not ports:
        return None
    try:
        s = serial.Serial(ports[0], 115200, timeout=0.1)
    except Exception:
        return None
    try:
        for _ in range(30):
            s.write(b"\x03")
            time.sleep(0.01)
        s.write(b"\r\n")
        time.sleep(0.2)
        if b">>>" in s.read(3000):
            return s, ports[0]
    except Exception:
        pass
    try:
        s.close()
    except Exception:
        pass
    return None


def wait_repl():
    log("等待 K230 可交互（卡死时请断电重上电）...")
    deadline = time.time() + WAIT_S
    n = 0
    while time.time() < deadline:
        got = try_catch_repl()
        if got:
            return got
        n += 1
        if n % 15 == 0:
            log("  ...重试中(%d)" % n)
        time.sleep(0.2)
    return None


def last_lines(txt, n=2):
    ls = [l.strip() for l in (txt or "").splitlines() if l.strip()]
    return ls[-n:] if ls else ["(无输出)"]


def repl(s, code, wait=0.5, limit=4000):
    try:
        s.write(code.encode() + b"\r\n")
        time.sleep(wait)
        return s.read(limit).decode("utf-8", "replace")
    except Exception as e:
        return "<repl err %s>" % e


def main():
    got = wait_repl()
    if not got:
        log("超时没抢到 REPL")
        return 1
    s, port = got
    log("[1] REPL OK:", port)

    log("[2] 删除 /sdcard/main.py ...")
    # 注意：REPL 里一次发多行会被吞/串行错乱，必须一行一条（或用 ; 拼成单行）
    log("   ", last_lines(repl(s, "import os", 0.3), 1))
    log("   ", last_lines(repl(s, "try: os.remove('/sdcard/main.py')", 0.4)
                            + repl(s, "except Exception as e: print('rm:', e)", 0.4), 1))
    log("   ", last_lines(repl(s, "print('RM_DONE')", 0.4), 1))

    log("[3] push point_kalman.py + gimbal_agent.py ...")
    s.close()
    r = subprocess.run([sys.executable, "-m", "k230tool", "push", PK_LOCAL,
                        "--dest", "/sdcard/point_kalman.py"],
                       cwd=TOOLKIT, capture_output=True, text=True)
    log("   pk:", (r.stdout or r.returncode).strip()
        if r.stdout else r.stderr.strip()[-300:])
    r = subprocess.run([sys.executable, "-m", "k230tool", "push", SRC_LOCAL,
                        "--dest", "/sdcard/gimbal_agent.py"],
                       cwd=TOOLKIT, capture_output=True, text=True)
    log("   ", (r.stdout or r.returncode).strip() if r.stdout else r.stderr.strip()[-300:])
    if not glob.glob("/dev/ttyACM*"):
        log("   端口消失了，等回来")
        time.sleep(5)

    got = wait_repl()
    if not got:
        log("push 之后没抢到 REPL")
        return 1
    s, port = got
    log("[4] 交互验证（REPL 里跑 agent，同时看 ESP32 收帧）...")
    s.write(b"exec(open('/sdcard/gimbal_agent.py').read())\r\n")
    time.sleep(1.0)
    out = s.read(4000).decode("utf-8", "replace")
    log("   K230:", " / ".join([l for l in out.splitlines() if l.strip()][:4]))

    n_frames = None
    for i in range(4):
        time.sleep(4.0)
        n_frames, line = frames_seen()
        log("   ESP32 %s" % (line or "(没解析到 [GIMBAL] 行)"))
        if n_frames and n_frames > 0:
            break

    out = s.read(6000).decode("utf-8", "replace")
    log("   K230 后续输出:", " / ".join([l for l in out.splitlines() if l.strip()][-5:]))

    if not n_frames:
        s.write(b"\x03\x03")
        time.sleep(0.8)
        log("   !! 没收到 $KT，暂不设自启动（板子保持 REPL 可控）")
        s.close()
        return 1

    log("[5] 验证通过，设为自启动 /sdcard/main.py ...")
    s.write(b"\x03\x03")
    time.sleep(0.8)
    log("   ", last_lines(repl(s, "src=open('/sdcard/gimbal_agent.py').read()", 0.5), 1))
    log("   ", last_lines(repl(s, "f=open('/sdcard/main.py','w'); n=f.write(src); f.close(); print('WROTE', n)", 1.5), 1))
    chk = repl(s, "import os; print('CHK', os.stat('/sdcard/main.py')[6])", 0.8)
    log("   ", last_lines(chk, 2))
    if str(len(SRC.encode())) not in chk:
        log("   !! 大小对不上（期望 %d），不设自启动" % len(SRC.encode()))
        s.close()
        return 1

    log("[6] 复位，确认上电自动运行 ...")
    try:
        s.write(b"import machine\r\nmachine.reset()\r\n")
        time.sleep(0.5)
        s.close()
    except Exception:
        pass
    time.sleep(14.0)
    n_frames, line = frames_seen()
    log("   ESP32:", line or "(无)")
    for p in sorted(glob.glob("/dev/ttyACM*")):
        try:
            k = serial.Serial(p, 115200, timeout=0.2)
            k.reset_input_buffer()
            t0 = time.time()
            buf = b""
            while time.time() - t0 < 6.0:
                buf += k.read(4096)
            k.close()
            keep = [l for l in buf.decode("utf-8", "replace").splitlines()
                    if "agent" in l or "ERROR" in l or "Traceback" in l]
            log("   K230 自启动日志:", " | ".join(keep[-6:]) or "(无)")
        except Exception as e:
            log("   读失败", p, e)

    if n_frames:
        log("完成：离线自启动可用（拔掉 USB、5V 独立供电即可）")
        return 0
    log("自启动后没收到帧，需要再查")
    return 1


if __name__ == "__main__":
    sys.exit(main())
