#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gimbal_sweep.py —— 舵机多角度起测（靶子固定，不动靶纸）
=========================================================

思路
----
靶纸固定在激光点上；用 ESP32 的 `gmanual` + `s1/s2` 把云台偏到不同
pan/tilt 起点（相当于相机从不同方向看靶），再 `gtrack` 让 PID 收敛；
稳定后采 4s，记录真实瞄准 `bull-laser`、`ctrlErr`、`$LS src`、激光模式、
pan/tilt。每轮结束 `gmanual`，避免下一轮起点被上一轮影响。

这样测的是：
  * 靶纸在相机里的**透视/角度**不同（左/右/上/下/对角看靶）；
  * 云台从不同方向起调时 PID 的收敛能力；
  * 不需要用户手动转靶纸。

用法
----
    # 靶纸固定、front 摆好，激光点在靶心
    python3 scripts/gimbal_sweep.py --sample 4 --settle 2
    python3 scripts/gimbal_sweep.py --sets "center:1500,1500;left:1300,1500;right:1700,1500"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "live"
LATEST = OUT / "latest.json"

# 云台起点：pan/tilt 都是 us；tilt 限位 1200~1800，pan 软限位更宽
DEFAULT_SETS = [
    ("center", 1500, 1500),
    ("left",   1300, 1500),
    ("right",  1700, 1500),
    ("down",   1500, 1350),
    ("up",     1500, 1650),
    ("dl",     1300, 1350),
    ("ur",     1700, 1650),
]

RE_CTRL = re.compile(r"ctrlErr=(-?[\d.]+),(-?[\d.]+)")
RE_PANTILT = re.compile(r"PAN=(\d+) TILT=(\d+)")
RE_AIM = re.compile(r"laser=(-?[\d.]+),(-?[\d.]+) src=(-?\d+)")


def _log(msg):
    print("[sweep] %s" % msg, flush=True)


def _pct(vals, q):
    if not vals:
        return None
    vals = sorted(vals)
    i = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return vals[i]


def start_live(duration, log_path):
    cmd = [sys.executable, "-m", "k230tool", "live",
           "--duration", str(duration), "--full-every", "9999"]
    _log("起 live：%s" % " ".join(cmd))
    f = open(log_path, "w")
    return subprocess.Popen(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)


def read_latest():
    try:
        d = json.loads(LATEST.read_text())
        if time.time() - float(d.get("ts", 0)) > 2.0:
            return None
        return d
    except Exception:
        return None


def esp(ser, cmd):
    ser.reset_input_buffer()
    ser.write((cmd + "\n").encode())
    ser.flush()
    time.sleep(0.12)
    return ser.read(2048).decode("utf-8", "replace")


def sample(ser):
    row = {"t": time.time()}
    d = read_latest()
    if d is not None:
        lz = d.get("laser") or {}
        bull = d.get("bull") or {}
        row.update({
            "laser_x": lz.get("x"), "laser_y": lz.get("y"),
            "laser_src": lz.get("src"), "laser_mode": lz.get("mode"),
            "calib_dev": lz.get("calib_dev"), "laser_ms": d.get("laser_ms"),
            "bull_x": bull.get("x"), "bull_y": bull.get("y"),
            "bull_status": bull.get("status"), "bull_conf": bull.get("conf"),
        })
        if (bull.get("status") == "ok"
                and lz.get("x") is not None and lz.get("y") is not None):
            row["aim_x"] = float(bull["x"]) - float(lz["x"])
            row["aim_y"] = float(bull["y"]) - float(lz["y"])
    txt = esp(ser, "aim")
    m = RE_AIM.search(txt)
    if m:
        row["ls_src"] = int(m.group(3))
    txt = esp(ser, "status")
    m = RE_CTRL.search(txt)
    if m:
        row["ctrl_x"] = float(m.group(1))
        row["ctrl_y"] = float(m.group(2))
    m = RE_PANTILT.search(txt)
    if m:
        row["pan"] = int(m.group(1))
        row["tilt"] = int(m.group(2))
    return row


def main():
    ap = argparse.ArgumentParser(description="舵机多角度起测")
    ap.add_argument("--sample", type=float, default=4.0, help="每角度采样秒数")
    ap.add_argument("--settle", type=float, default=2.0, help="gtrack 后稳定秒数")
    ap.add_argument("--sets", default="", help="label:pan,tilt;...")
    ap.add_argument("--pid", default=None, help="PAN PID 'kp ki kd'")
    ap.add_argument("--pid2", default=None, help="TILT PID 'kp ki kd'")
    ap.add_argument("--smooth", type=float, default=None, help="速度低通 tau")
    ap.add_argument("--trk", default=None, help="目标 alpha/lead 'a lead'")
    ap.add_argument("--label", default="sweep")
    args = ap.parse_args()

    if args.sets:
        sets = []
        for part in args.sets.split(";"):
            part = part.strip()
            if not part:
                continue
            lab, pt = part.split(":")
            p, t = pt.split(",")
            sets.append((lab, int(p), int(t)))
    else:
        sets = list(DEFAULT_SETS)

    try:
        import serial
    except ImportError:
        sys.exit("需要 pyserial")

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    live = start_live(int(len(sets) * (args.settle + args.sample) + 25),
                      OUT / ("gimbal_sweep_live_%s.log" % stamp))
    _log("等 live 出图（最多 45s）…")
    t0 = time.time()
    while time.time() - t0 < 45:
        if read_latest() is not None:
            break
        time.sleep(0.5)
    if read_latest() is None:
        _log("live 没起来；看 log")
        live.terminate()
        return 2

    ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.3)
    time.sleep(0.2)
    ser.reset_input_buffer()
    _log("ESP32 已打开；gtrack 后云台会动，靶纸保持不动")
    esp(ser, "gtrack")
    time.sleep(1.0)
    for cmd, val in (("pid", args.pid), ("pid2", args.pid2), ("trk", args.trk)):
        if val:
            _log("ESP32 %s %s -> %s" % (cmd, val, esp(ser, "%s %s" % (cmd, val)).strip()))
    if args.smooth is not None:
        _log("ESP32 smooth -> %s" % esp(ser, "smooth %.4f" % args.smooth).strip())

    # baseline：中心保持
    results = []
    for label, pan, tilt in sets:
        _log("=== %s pan=%d tilt=%d ===" % (label, pan, tilt))
        esp(ser, "gmanual")
        esp(ser, "s1 %d" % pan)
        esp(ser, "s2 %d" % tilt)
        time.sleep(0.4)
        esp(ser, "gtrack")
        time.sleep(args.settle)
        rows = []
        t1 = time.time()
        while time.time() - t1 < args.sample:
            rows.append(sample(ser))
            time.sleep(0.2)
        esp(ser, "gmanual")

        ok = [r for r in rows if r.get("aim_x") is not None]
        ax = [abs(r["aim_x"]) for r in ok]
        ay = [abs(r["aim_y"]) for r in ok]
        cx = [abs(r["ctrl_x"]) for r in ok if r.get("ctrl_x") is not None]
        cy = [abs(r["ctrl_y"]) for r in ok if r.get("ctrl_y") is not None]
        src = [r.get("ls_src") for r in ok if r.get("ls_src") is not None]
        modes = [r.get("laser_mode") for r in ok if r.get("laser_mode")]
        rec = {
            "label": label, "pan": pan, "tilt": tilt,
            "n": len(rows), "ok": len(ok),
            "aim_x_p50": _pct(ax, .5), "aim_y_p50": _pct(ay, .5),
            "aim_x_p95": _pct(ax, .95), "aim_y_p95": _pct(ay, .95),
            "ctrl_x_p50": _pct(cx, .5), "ctrl_y_p50": _pct(cy, .5),
            "src1": (100.0 * src.count(1) / len(src)) if src else 0.0,
            "roi": modes.count("roi"), "coast": modes.count("coast"),
            "calib": modes.count("calib"),
        }
        results.append(rec)
        _log("%-7s n=%d ok=%d aim p50=(%s,%s) p95=(%s,%s) ctrl=(%s,%s) src1=%s roi=%d"
             % (label, len(rows), len(ok),
                _fmt(rec["aim_x_p50"]), _fmt(rec["aim_y_p50"]),
                _fmt(rec["aim_x_p95"]), _fmt(rec["aim_y_p95"]),
                _fmt(rec["ctrl_x_p50"]), _fmt(rec["ctrl_y_p50"]),
                _fmt(rec["src1"], 0), rec["roi"]))

    esp(ser, "gmanual")
    esp(ser, "s1 1500")
    esp(ser, "s2 1500")
    ser.close()

    csv_path = OUT / ("gimbal_sweep_%s_%s.csv" % (stamp, args.label))
    keys = list(results[0].keys()) if results else []
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow(r)

    print("\n=== 云台多角度起测（靶子固定） ===")
    print("%-7s %5s %5s %4s %4s  %-15s %-15s %-13s %6s" %
          ("label", "pan", "tilt", "n", "ok", "aim p50", "aim p95", "ctrl p50", "src1"))
    for r in results:
        print("%-7s %5d %5d %4d %4d  (%5s,%5s)  (%5s,%5s)  (%5s,%5s) %5s%%" %
              (r["label"], r["pan"], r["tilt"], r["n"], r["ok"],
               _fmt(r["aim_x_p50"]), _fmt(r["aim_y_p50"]),
               _fmt(r["aim_x_p95"]), _fmt(r["aim_y_p95"]),
               _fmt(r["ctrl_x_p50"]), _fmt(r["ctrl_y_p50"]),
               _fmt(r["src1"], 0)))
    print("CSV: %s" % csv_path)
    try:
        live.wait(timeout=15)
    except subprocess.TimeoutExpired:
        live.terminate()
    return 0


def _fmt(v, nd=1):
    if v is None:
        return "-"
    return ("%." + str(nd) + "f") % v


if __name__ == "__main__":
    raise SystemExit(main())
