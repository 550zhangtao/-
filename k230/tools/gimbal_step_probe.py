# -*- coding: utf-8 -*-
"""gimbal_step_probe.py —— 云台控制环阶跃响应测量（PC 模拟 K230 帧）

ESP32 的 USB 口支持直接喂 `$KT,...!` 帧（固件里就是这么自测的），
所以不需要 K230 在跑，就能单独测「控制环 + 舵机」这一段的真实响应：

  1. 切 TRACK，用 (320,240) 把云台拉到中心；
  2. 持续 10Hz 喂 `$KT`，在 t=1s 把靶心阶跃到 (500,300)；
  3. 20ms 轮询 status，记录 pan/tilt/ctrlErr；
  4. 输出：阶跃后 90% 到位耗时、超调量、稳态误差。

用法：
    python3 -m tools.gimbal_step_probe --esp32 /dev/ttyUSB0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def open_esp(dev):
    import serial
    s = serial.Serial(dev, 115200, timeout=0.05)
    time.sleep(0.2)
    s.reset_input_buffer()
    return s


def send(s, cmd):
    s.write((cmd + "\n").encode())


def read_status(s, wait=0.06):
    """发一次 status 并解析；返回 dict 或 None。"""
    s.write(b"status\n")
    t0 = time.monotonic()
    buf = b""
    while time.monotonic() - t0 < wait:
        buf += s.read(4096)
    for line in buf.decode("utf-8", "replace").replace("\r", "\n").splitlines():
        line = line.strip()
        if line.startswith("PAN="):
            d = {}
            for tok in line.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    d[k] = v
            try:
                return {
                    "pan": int(d["PAN"]), "tilt": int(d["TILT"]),
                    "state": int(d.get("state", -1)),
                    "frames": int(d.get("frames", -1)),
                    "age": int(d.get("age", -1)),
                    "err": d.get("ctrlErr", ""),
                }
            except Exception:
                return None
    return None


def feed(s, seq, bull, run=1):
    send(s, "$KT,%d,%d,320,240,%d,%d,900!" % (seq, run, bull[0], bull[1]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--esp32", default="/dev/ttyUSB0")
    ap.add_argument("--period", type=float, default=0.1, help="喂帧周期 s")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--from", dest="p0", default="320,240")
    ap.add_argument("--to", dest="p1", default="500,300")
    ap.add_argument("--out", default=str(ROOT / "live" / "gimbal_step.json"))
    args = ap.parse_args(argv)

    p0 = tuple(int(v) for v in args.p0.split(","))
    p1 = tuple(int(v) for v in args.p1.split(","))
    s = open_esp(args.esp32)
    send(s, "gtrack")
    time.sleep(0.4)

    seq = 0
    samples = []
    t0 = time.monotonic()
    next_feed = t0
    last_ctrl = 0.0
    try:
        while time.monotonic() - t0 < args.seconds:
            now = time.monotonic()
            elapsed = now - t0
            bull = p0 if elapsed < 1.0 else p1
            if now >= next_feed:
                seq += 1
                feed(s, seq, bull)
                next_feed += args.period
            if now - last_ctrl >= 0.02:
                st = read_status(s, wait=0.02)
                if st:
                    samples.append({"t": round(elapsed, 3), "bull": list(bull), **st})
                last_ctrl = now
            else:
                time.sleep(0.005)
    finally:
        s.close()

    # ---- 分析 ----
    pre = [x for x in samples if x["t"] < 1.0 and x["pan"] > 0]
    post = [x for x in samples if x["t"] >= 1.0 and x["pan"] > 0]
    print("\n=== 阶跃响应 %s -> %s ===" % (p0, p1))
    if pre:
        print("  阶跃前 pan=%d/%d tilt=%d/%d（平均）"
              % (sum(x["pan"] for x in pre) // len(pre),
                 sum(x["tilt"] for x in pre) // len(pre),
                 sum(x["tilt"] for x in pre) // len(pre),
                 sum(x["tilt"] for x in pre) // len(pre)))
    if post:
        p0v = sum(x["pan"] for x in pre) // max(1, len(pre)) if pre else 0
        goal = post[-1]["pan"]
        span = goal - p0v
        reach = None
        for x in post:
            if abs(x["pan"] - p0v) >= abs(span) * 0.9:
                reach = x["t"] - 1.0
                break
        peak = max(post, key=lambda x: abs(x["pan"] - p0v))
        overshoot = (abs(peak["pan"] - goal) / max(1.0, abs(span))) * 100.0
        print("  90%% 到位耗时    : %s"
              % ("%.3fs" % reach if reach is not None else "未在窗口内到位"))
        print("  最大超调        : %.1f%%（peak pan=%d, 终值=%d）"
              % (overshoot, peak["pan"], goal))
        print("  终值稳态误差    : %s" % (post[-1].get("err", "?")))
        print("\n  时间线（阶跃后，每 2 个采样一行）:")
        print("   t(s)   pan  tilt  err        age")
        for x in post[::2][:40]:
            print("  %5.2f  %4d  %4d  %-10s %5d"
                  % (x["t"], x["pan"], x["tilt"], x.get("err", ""), x["age"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
    print("\n[probe] 原始数据 -> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
