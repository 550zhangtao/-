#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
motion_test_laser_aim.py —— 运动靶基线 / 抗干扰采样（新激光模块）
=================================================================

做什么
------
1. 起一个 `k230tool live`（Pose + `laser_aim` + `$KT`/`$LS` 到 ESP32）；
2. 通过 ESP32 USB 采样 `aim` / `status`（`$LS src/conf`、`ctrlErr`、`frames`）；
3. 同时每 0.2s 读 `k230/live/latest.json`（`laser`、`laser_ms`、`calib_dev`、
   `bull`、`detections`）；
4. 落 CSV + 打印汇总：`src=1` 比例、`calib_dev`、`laser_ms`、`|ctrlErr|`、
   激光坐标抖动、`$KT` 帧率、`bull` 检出比例。

用法
----
    # 推荐：脚本自己起 live，自动 gtrack/gmanual
    python3 scripts/motion_test_laser_aim.py --duration 45

    # live 已经手动在跑（另一个终端）：
    python3 scripts/motion_test_laser_aim.py --duration 45 --no-live

    # 只要抗干扰（不让云台动）：
    python3 scripts/motion_test_laser_aim.py --duration 30 --no-gtrack

注意
----
- 会占 K230 串口（起 live 时）；不要同时开 CanMV IDE；
- ESP32 串口默认自动找 /dev/ttyUSB0；
- 测试时靶纸要保持在画面内；否则 ESP32 三态 A 会在 ~0.9s 后复位云台。
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

ROOT = Path(__file__).resolve().parent.parent      # k230/
OUT_DIR = ROOT / "live"
LATEST = OUT_DIR / "latest.json"

RE_AIM_LASER = re.compile(r"laser=(-?[\d.]+),(-?[\d.]+) src=(-?\d+) conf=(-?\d+)")
RE_AIM_VALID = re.compile(r"laserValid=(\d+)")
RE_AIM_STATE = re.compile(r"AIM-IFACE state=(\d+)")
RE_CTRL = re.compile(r"ctrlErr=(-?[\d.]+),(-?[\d.]+)")
RE_FRAMES = re.compile(r"frames=(\d+)")
RE_BAD = re.compile(r"bad=(\d+)")
RE_GMODE = re.compile(r"state=(\d+)")
RE_BULLLOST = re.compile(r"bullLost=(\d+)")
RE_PANTILT = re.compile(r"PAN=(\d+) TILT=(\d+)")


def _log(msg: str) -> None:
    print("[motion] %s" % msg, flush=True)


def _send(ser, cmd: str) -> str:
    ser.reset_input_buffer()
    ser.write((cmd + "\n").encode())
    ser.flush()
    time.sleep(0.12)
    return ser.read(2048).decode("utf-8", "replace")


def _pct(vals, q):
    if not vals:
        return None
    vals = sorted(vals)
    i = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return vals[i]


def _read_latest():
    try:
        d = json.loads(LATEST.read_text())
        if time.time() - float(d.get("ts", 0)) > 3.0:
            return None
        return d
    except Exception:
        return None


def start_live(duration: int, log_path: Path) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "k230tool", "live",
           "--duration", str(duration),
           "--full-every", "9999"]
    _log("起 live：%s" % " ".join(cmd))
    f = open(log_path, "w")
    return subprocess.Popen(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)


def main() -> int:
    ap = argparse.ArgumentParser(description="运动靶/抗干扰采样（新激光模块）")
    ap.add_argument("--duration", type=int, default=45, help="采样秒数")
    ap.add_argument("--esp-port", default="/dev/ttyUSB0")
    ap.add_argument("--no-live", action="store_true", help="live 已在跑，不要重复起")
    ap.add_argument("--no-gtrack", action="store_true", help="不自动进 R2 跟踪")
    ap.add_argument("--out", default=None, help="CSV 路径")
    ap.add_argument("--pid", default=None, help="PAN PID 'kp ki kd'（gtrack 前下发）")
    ap.add_argument("--pid2", default=None, help="TILT PID 'kp ki kd'")
    ap.add_argument("--smooth", type=float, default=None, help="速度低通 tau")
    ap.add_argument("--trk", default=None, help="目标 alpha/lead 'a lead'")
    ap.add_argument("--label", default="", help="本次运行标签（写入 CSV 名）")
    args = ap.parse_args()

    try:
        import serial
    except ImportError:
        sys.exit("需要 pyserial")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = ("_" + args.label) if args.label else ""
    csv_path = Path(args.out) if args.out else (OUT_DIR / ("motion_%s%s.csv" % (stamp, tag)))
    live_log = OUT_DIR / ("motion_live_%s%s.log" % (stamp, tag))

    live = None
    if not args.no_live:
        live = start_live(args.duration + 15, live_log)
        _log("等 live 的 latest.json 变新鲜（最多 40s）…")
        t0 = time.time()
        while time.time() - t0 < 40:
            d = _read_latest()
            if d is not None:
                break
            time.sleep(0.5)
        if _read_latest() is None:
            _log("live 没起来；看 %s" % live_log)
            if live:
                live.terminate()
            return 2
        _log("live 已出图")

    ser = None
    try:
        ser = serial.Serial(args.esp_port, 115200, timeout=0.3)
        time.sleep(0.3)
        ser.reset_input_buffer()
        _log("ESP32 %s 已打开" % args.esp_port)
    except Exception as e:
        _log("ESP32 打不开（%s）-> 只采 latest.json，不采 aim/status" % e)

    if ser is not None:
        for cmd, val in (("pid", args.pid), ("pid2", args.pid2),
                         ("trk", args.trk)):
            if val:
                _log("ESP32 %s %s -> %s" % (cmd, val, _send(ser, "%s %s" % (cmd, val)).strip()))
        if args.smooth is not None:
            _log("ESP32 smooth -> %s" % _send(ser, "smooth %.4f" % args.smooth).strip())
        _log("ESP32 pidshow: %s" % _send(ser, "pidshow").strip())

    if ser is not None and not args.no_gtrack:
        _log("发 gtrack（R2）；请开始移动靶纸…")
        _send(ser, "gtrack")
        time.sleep(0.8)
    elif ser is None:
        _log("没有 ESP32：只测激光自身的 src/mode/耗时/抖动")
    else:
        _log("不进跟踪；只测激光/抗干扰")

    rows = []
    t_start = time.time()
    last_print = t_start
    frames_prev = None
    t_frames_prev = None
    try:
        while time.time() - t_start < args.duration:
            now = time.time()
            row = {"t": round(now - t_start, 3)}

            d = _read_latest()
            if d is not None:
                lz = d.get("laser") or {}
                bull = d.get("bull") or {}
                det = (d.get("detections") or [None])[0] or {}
                box = det.get("box") or [None] * 4
                row.update({
                    "laser_x": lz.get("x"), "laser_y": lz.get("y"),
                    "laser_src": lz.get("src"), "laser_status": lz.get("status"),
                    "laser_conf": lz.get("conf"), "calib_dev": lz.get("calib_dev"),
                    "laser_ms": d.get("laser_ms"),
                    "laser_mode": lz.get("mode"),
                    "laser_vx": lz.get("vx"), "laser_vy": lz.get("vy"),
                    "bull_x": bull.get("x"), "bull_y": bull.get("y"),
                    "bull_status": bull.get("status"),
                    "align_x": (float(bull["x"]) - float(lz["x"]))
                               if (bull.get("x") is not None and lz.get("x") is not None) else None,
                    "align_y": (float(bull["y"]) - float(lz["y"]))
                               if (bull.get("y") is not None and lz.get("y") is not None) else None,
                    "box_w": box[2] if len(box) > 2 else None,
                    "box_h": box[3] if len(box) > 3 else None,
                    "fps_pc": d.get("fps"),
                })
            else:
                row["live"] = 0

            txt = _send(ser, "aim") if ser is not None else ""
            m = RE_AIM_LASER.search(txt)
            if m:
                row["ls_x"] = float(m.group(1))
                row["ls_y"] = float(m.group(2))
                row["ls_src"] = int(m.group(3))
                row["ls_conf"] = int(m.group(4))
            mv = RE_AIM_VALID.search(txt)
            if mv:
                row["ls_valid"] = int(mv.group(1))
            ms = RE_AIM_STATE.search(txt)
            if ms:
                row["aim_state"] = int(ms.group(1))

            txt = _send(ser, "status") if ser is not None else ""
            m = RE_CTRL.search(txt)
            if m:
                row["ctrl_ex"] = float(m.group(1))
                row["ctrl_ey"] = float(m.group(2))
            m = RE_FRAMES.search(txt)
            if m:
                row["frames"] = int(m.group(1))
                if frames_prev is not None and t_frames_prev is not None:
                    dt = max(1e-3, now - t_frames_prev)
                    row["kt_fps"] = round((row["frames"] - frames_prev) / dt, 1)
                frames_prev = row["frames"]
                t_frames_prev = now
            m = RE_BAD.search(txt)
            if m:
                row["bad"] = int(m.group(1))
            m = RE_BULLLOST.search(txt)
            if m:
                row["bull_lost"] = int(m.group(1))
            m = RE_PANTILT.search(txt)
            if m:
                row["pan"] = int(m.group(1))
                row["tilt"] = int(m.group(2))
            m = RE_GMODE.search(txt)
            if m:
                row["track_state"] = int(m.group(1))
            rows.append(row)

            if now - last_print >= 5.0:
                _log("t=%4.1fs live=%s src=%s laser=(%s,%s) calib_dev=%s ctrl=(%s,%s) kt=%s"
                     % (now - t_start, row.get("laser_src"), row.get("ls_src"),
                        row.get("laser_x"), row.get("laser_y"),
                        row.get("calib_dev"), row.get("ctrl_ex"),
                        row.get("ctrl_ey"), row.get("kt_fps")))
                last_print = now
            time.sleep(0.2)
    finally:
        if ser is not None and not args.no_gtrack:
            _log("发 gmanual（停止跟踪）")
            _send(ser, "gmanual")
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    # ---- CSV
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    _log("CSV: %s（%d 行）" % (csv_path, len(rows)))

    # ---- 汇总
    def col(k):
        return [r[k] for r in rows if r.get(k) is not None]

    src1 = [r.get("ls_src") for r in rows if r.get("ls_src") is not None]
    lm_src = [r.get("laser_src") for r in rows if r.get("laser_src") is not None]
    dev = [abs(float(v)) for v in col("calib_dev")]
    lms = [float(v) for v in col("laser_ms")]
    ex = [abs(float(v)) for v in col("ctrl_ex")]
    ey = [abs(float(v)) for v in col("ctrl_ey")]
    lx = [float(v) for v in col("laser_x")]
    ly = [float(v) for v in col("laser_y")]
    kt = [float(v) for v in col("kt_fps")]
    bull_ok = [1 for r in rows if (r.get("bull_status") == "ok")]

    print("\n=== 运动靶基线汇总 ===")
    print("采样 %d 点 / %.1fs，CSV=%s" % (len(rows), args.duration, csv_path))
    print("$LS src=1 比例 : %s" % (
        "%.1f%% (%d/%d)" % (100.0 * src1.count(1) / len(src1), src1.count(1), len(src1))
        if src1 else "-"))
    print("K230 laser src=measured : %s" % (
        "%.1f%%" % (100.0 * lm_src.count("measured") / len(lm_src)) if lm_src else "-"))
    print("calib_dev px   : p50=%s p95=%s max=%s" % (
        _pct(dev, .5), _pct(dev, .95), max(dev) if dev else None))
    print("laser_ms       : p50=%s p95=%s max=%s" % (
        _pct(lms, .5), _pct(lms, .95), max(lms) if lms else None))
    print("|ctrlErrX| px  : p50=%s p95=%s" % (_pct(ex, .5), _pct(ex, .95)))
    print("|ctrlErrY| px  : p50=%s p95=%s" % (_pct(ey, .5), _pct(ey, .95)))
    if lx:
        print("laser x 抖动   : std=%.2f range=%.1f" % (st.pstdev(lx), max(lx) - min(lx)))
    if ly:
        print("laser y 抖动   : std=%.2f range=%.1f" % (st.pstdev(ly), max(ly) - min(ly)))
    print("$KT 帧率       : p50=%s" % _pct(kt, .5))
    print("bull 检出比例  : %.1f%%" % (100.0 * len(bull_ok) / max(1, len(rows))))
    ax = [float(r["align_x"]) for r in rows if r.get("align_x") is not None]
    ay = [float(r["align_y"]) for r in rows if r.get("align_y") is not None]
    if ax and ay:
        print("瞄准误差(靶心-激光): p50=(%.1f,%.1f) p95=(%.1f,%.1f) mean=(%.1f,%.1f) std=(%.1f,%.1f)"
              % (_pct(sorted(abs(v) for v in ax), .5),
                 _pct(sorted(abs(v) for v in ay), .5),
                 _pct(sorted(abs(v) for v in ax), .95),
                 _pct(sorted(abs(v) for v in ay), .95),
                 sum(ax) / len(ax), sum(ay) / len(ay),
                 st.pstdev(ax) if len(ax) > 1 else 0.0,
                 st.pstdev(ay) if len(ay) > 1 else 0.0))
    modes = [r.get("laser_mode") for r in rows if r.get("laser_mode")]
    if modes:
        print("laser mode     : roi=%d coast=%d calib=%d lost=%d"
              % (modes.count("roi"), modes.count("coast"),
                 modes.count("calib"), modes.count("lost")))

    def seg(name, sub):
        if not sub:
            return
        s1 = [r.get("ls_src") for r in sub if r.get("ls_src") is not None]
        dv = [abs(float(r["calib_dev"])) for r in sub if r.get("calib_dev") is not None]
        lm = [float(r["laser_ms"]) for r in sub if r.get("laser_ms") is not None]
        cx = [abs(float(r["ctrl_ex"])) for r in sub if r.get("ctrl_ex") is not None]
        cy = [abs(float(r["ctrl_ey"])) for r in sub if r.get("ctrl_ey") is not None]
        print("  [%s] n=%d  src1=%s  calib_dev p50=%s p95=%s  "
              "laser_ms p50=%s p95=%s  |ctrlErr| p50=(%s,%s)"
              % (name, len(sub),
                 ("%.0f%%" % (100.0 * s1.count(1) / len(s1))) if s1 else "-",
                 _pct(dv, .5), _pct(dv, .95), _pct(lm, .5), _pct(lm, .95),
                 _pct(cx, .5), _pct(cy, .5)))

    print("\n--- 分段（靶纸是否在视野） ---")
    seg("bull ok", [r for r in rows if r.get("bull_status") == "ok"])
    seg("bull lost/low", [r for r in rows if r.get("bull_status") != "ok"])

    if live is not None:
        _log("等 live 自己结束（或直接 Ctrl-C；不要 kill -9）…")
        try:
            live.wait(timeout=args.duration + 25)
        except subprocess.TimeoutExpired:
            _log("live 超时，terminate（不是 kill -9）")
            live.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
