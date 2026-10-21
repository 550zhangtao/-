#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pid_autotune.py —— 在线 PID 自动调参（K230 live + ESP32 USB）
=============================================================

做什么
------
1. 起 `k230tool live`（Pose + `laser_aim` + `$KT`/`$LS`），ESP32 发 `gtrack`；
2. 逐组下发 `pid` / `pid2` / `smooth` / `trk`，每组先 settle、再采 N 秒；
3. 评分 = 真实瞄准误差 `bull-laser`（同一帧 `latest.json`）的
   `mean + 0.5·p95 + 0.5·std + 0.02·(pan/tilt 抖动)`，激光丢失/靶心丢失加罚；
4. 坐标下降：PAN kp → TILT kp → ki → kd → smooth → lead，逐轮取最优；
5. 全程安全阀：连续 6 个采样无有效瞄准（激光/靶心丢失或误差 >120px）
   → 立即 `gmanual` + 舵机回中 + 恢复当前最优，退出并提示重新摆靶；
6. 结束把最优值写回 ESP32，并打印完整对比表 + CSV。

用法
----
    # 靶纸放正、激光红点在靶上、靶心画面中间偏上，保持不动
    python3 scripts/pid_autotune.py --sample-sec 4 --settle 1.2

    # 从某个已知值开始（比如上一轮最优，复跑某角度）
    python3 scripts/pid_autotune.py --start "3.0 0.3 0.15 3.0 0.2 0.1 0.02 0.35 0.20"

注意
----
- 会占 K230 串口；不要同时开 CanMV IDE；
- `gtrack` 之后云台会动，靶纸必须保持在激光束里、画面内；
- 安全阀触发就说明该组增益在该姿态下发散/丢靶，不是让你反复试同一组。
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

RE_CTRL = re.compile(r"ctrlErr=(-?[\d.]+),(-?[\d.]+)")
RE_PANTILT = re.compile(r"PAN=(\d+) TILT=(\d+)")
RE_STATE = re.compile(r"auto=(\d+) state=(\d+)")
RE_AIM_LASER = re.compile(r"laser=(-?[\d.]+),(-?[\d.]+) src=(-?\d+)")

DEFAULT_START = {
    "kp": 2.5, "ki": 0.3, "kd": 0.15,
    "kp2": 2.5, "ki2": 0.2, "kd2": 0.10,
    "smooth": 0.02, "trk": (0.35, 0.20),
}


def _log(msg: str) -> None:
    print("[tune] %s" % msg, flush=True)


def _pct(vals, q):
    if not vals:
        return None
    vals = sorted(vals)
    i = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return vals[i]


def start_live(duration: int, log_path: Path) -> subprocess.Popen:
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


def esp(ser, cmd: str) -> str:
    ser.reset_input_buffer()
    ser.write((cmd + "\n").encode())
    ser.flush()
    time.sleep(0.1)
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
            "laser_conf": lz.get("conf"), "calib_dev": lz.get("calib_dev"),
            "laser_ms": d.get("laser_ms"), "fps": d.get("fps"),
            "bull_x": bull.get("x"), "bull_y": bull.get("y"),
            "bull_status": bull.get("status"), "bull_conf": bull.get("conf"),
        })
        if (bull.get("status") == "ok"
                and lz.get("x") is not None and lz.get("y") is not None):
            row["align_x"] = float(bull["x"]) - float(lz["x"])
            row["align_y"] = float(bull["y"]) - float(lz["y"])
        else:
            row["align_x"] = None
            row["align_y"] = None
    txt = esp(ser, "aim")
    m = RE_AIM_LASER.search(txt)
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
    m = RE_STATE.search(txt)
    if m:
        row["auto"] = int(m.group(1))
        row["state"] = int(m.group(2))
    return row


def valid(r) -> bool:
    if r.get("align_x") is None or r.get("align_y") is None:
        return False
    if r.get("laser_src") != "measured":
        return False
    if r.get("bull_status") != "ok":
        return False
    bx = r.get("bull_x") or -1
    by = r.get("bull_y") or -1
    return 0 <= bx <= 640 and 0 <= by <= 480


def score(rows):
    ok = [r for r in rows if valid(r)]
    if len(ok) < max(2, int(0.6 * len(rows))):
        return 1e9, {"ok": len(ok), "n": len(rows)}
    ax = [abs(r["align_x"]) for r in ok]
    ay = [abs(r["align_y"]) for r in ok]
    mx = st.mean(ax)
    my = st.mean(ay)
    px = _pct(ax, 0.95)
    py = _pct(ay, 0.95)
    jx = st.pstdev(ax) if len(ax) > 1 else 0.0
    jy = st.pstdev(ay) if len(ay) > 1 else 0.0
    pan = [r["pan"] for r in ok if r.get("pan") is not None]
    tilt = [r["tilt"] for r in ok if r.get("tilt") is not None]
    pj = st.pstdev(pan) if len(pan) > 1 else 0.0
    tj = st.pstdev(tilt) if len(tilt) > 1 else 0.0
    s = (mx + my) + 0.5 * (px + py) + 0.5 * (jx + jy) + 0.02 * (pj + tj)
    s += 20.0 * (len(rows) - len(ok)) / max(1, len(rows))
    return s, {"ok": len(ok), "n": len(rows), "mx": mx, "my": my,
               "px": px, "py": py, "jx": jx, "jy": jy,
               "pj": pj, "tj": tj}


def apply_set(ser, p):
    esp(ser, "pid %.3f %.3f %.3f" % (p["kp"], p["ki"], p["kd"]))
    esp(ser, "pid2 %.3f %.3f %.3f" % (p["kp2"], p["ki2"], p["kd2"]))
    if p.get("smooth") is not None:
        esp(ser, "smooth %.4f" % p["smooth"])
    if p.get("trk") is not None:
        esp(ser, "trk %.2f %.3f" % (p["trk"][0], p["trk"][1]))


def run_set(ser, p, settle, sec):
    apply_set(ser, p)
    time.sleep(settle)
    rows = []
    bad = 0
    t0 = time.time()
    while time.time() - t0 < sec:
        r = sample(ser)
        rows.append(r)
        if not valid(r):
            bad += 1
        elif abs(r["align_x"]) > 120 or abs(r["align_y"]) > 120:
            bad += 2
        else:
            bad = 0
        if bad >= 6:
            return rows, "lost"
        time.sleep(0.2)
    return rows, "ok"


def fmt_set(p) -> str:
    return ("kp=%.2f ki=%.2f kd=%.2f | kp2=%.2f ki2=%.2f kd2=%.2f | "
            "smooth=%.3f lead=%.2f" % (p["kp"], p["ki"], p["kd"],
                                       p["kp2"], p["ki2"], p["kd2"],
                                       p["smooth"], p["trk"][1]))


def main() -> int:
    ap = argparse.ArgumentParser(description="在线 PID 自动调参")
    ap.add_argument("--esp-port", default="/dev/ttyUSB0")
    ap.add_argument("--sample-sec", type=float, default=4.0)
    ap.add_argument("--settle", type=float, default=1.2)
    ap.add_argument("--start", default=None,
                    help="9 个数：kp ki kd kp2 ki2 kd2 smooth alpha lead")
    ap.add_argument("--label", default="autotune")
    ap.add_argument("--dry-sets", action="store_true", help="只打印参数组，不连板")
    args = ap.parse_args()

    start = dict(DEFAULT_START)
    if args.start:
        v = [float(x) for x in args.start.replace(",", " ").split()]
        if len(v) != 9:
            sys.exit("--start 需要 9 个数")
        start.update({"kp": v[0], "ki": v[1], "kd": v[2],
                      "kp2": v[3], "ki2": v[4], "kd2": v[5],
                      "smooth": v[6], "trk": (v[7], v[8])})

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    # 预估组数：baseline + PAN kp 4 + TILT kp 4 + ki 4 + kd 4 + smooth 3 + lead 3
    n_sets = 1 + 4 + 4 + 4 + 4 + 3 + 3
    total_sec = n_sets * (args.settle + args.sample_sec) + 90
    if args.dry_sets:
        _log("预计 %d 组，约 %.0fs；start=%s" % (n_sets, total_sec, fmt_set(start)))
        return 0

    try:
        import serial
    except ImportError:
        sys.exit("需要 pyserial")

    live = start_live(int(total_sec) + 40, OUT / ("pid_autotune_live_%s.log" % stamp))
    _log("等 live 出图（最多 45s）…")
    t0 = time.time()
    while time.time() - t0 < 45:
        if read_latest() is not None:
            break
        time.sleep(0.5)
    if read_latest() is None:
        _log("live 没起来；看 %s" % (OUT / ("pid_autotune_live_%s.log" % stamp)))
        live.terminate()
        return 2

    ser = serial.Serial(args.esp_port, 115200, timeout=0.3)
    time.sleep(0.2)
    ser.reset_input_buffer()
    _log("ESP32 %s 已打开；发 gtrack，请保持靶纸不动" % args.esp_port)
    esp(ser, "pidshow")
    esp(ser, "gtrack")
    time.sleep(1.0)

    results = []
    best = dict(start)
    best_score = 1e9

    def evaluate(label, p):
        nonlocal best, best_score
        rows, status = run_set(ser, p, args.settle, args.sample_sec)
        sc, stat = score(rows)
        rec = dict(p)
        rec.update({"label": label, "status": status, "score": sc,
                    "trk_a": p["trk"][0], "trk_l": p["trk"][1]})
        rec.update(stat)
        results.append(rec)
        _log("%-10s %s -> score=%s %s"
             % (label, fmt_set(p), ("%.1f" % sc) if sc < 1e8 else "INF", status))
        if status == "lost":
            return sc, stat, True
        if sc < best_score:
            best = dict(p)
            best_score = sc
            _log("        ^ new best")
        return sc, stat, False

    def phase(name, variants):
        """variants: [(label, dict)]；返回是否 lost。"""
        nonlocal best
        for label, p in variants:
            p = dict(p)
            p["kp"] = p.get("kp", best["kp"])
            sc, stat, lost = evaluate(label, p)
            if lost:
                return True
        return False

    # 0) baseline
    sc, stat, lost = evaluate("A", best)
    if lost:
        _log("baseline 就丢了：靶纸不在激光束/画面里，先摆好再跑")
        esp(ser, "gmanual")
        ser.close()
        live.terminate()
        return 3
    if sc < best_score:
        best_score = sc

    # 每个 phase 的候选必须基于**该 phase 开始时的 best** 生成（坐标下降）
    phases = [
        ("pan-kp", lambda b: [("kp%.1f" % k, dict(b, kp=k))
                              for k in (2.0, 2.5, 3.0, 3.5)]),
        ("tilt-kp", lambda b: [("kp2%.1f" % k, dict(b, kp2=k))
                               for k in (2.5, 3.0, 3.5, 4.0)]),
        ("ki", lambda b: [("ki%.1f" % k, dict(b, ki=k, ki2=k))
                          for k in (0.0, 0.1, 0.2, 0.3)]),
        ("kd", lambda b: [("kd%g/%g" % (a, c), dict(b, kd=a, kd2=c))
                          for a, c in ((0.05, 0.05), (0.10, 0.10),
                                       (0.20, 0.15), (0.30, 0.20))]),
        ("smooth", lambda b: [("sm%.2f" % v, dict(b, smooth=v))
                              for v in (0.01, 0.02, 0.03)]),
        ("lead", lambda b: [("ld%.2f" % v, dict(b, trk=(b["trk"][0], v)))
                            for v in (0.15, 0.20, 0.25)]),
    ]

    for name, maker in phases:
        variants = maker(best)
        _log("=== phase %s（当前 best score=%.1f）===" % (name, best_score))
        lost = phase(name, variants)
        if lost:
            _log("安全阀：该组丢靶/发散，恢复当前最优并退出；请重新摆靶后复跑")
            esp(ser, "gmanual")
            esp(ser, "s1 1500")
            esp(ser, "s2 1500")
            apply_set(ser, best)
            break

    # 结束：设回最优，采 5s 终评
    _log("=== 设回最优：%s ===" % fmt_set(best))
    apply_set(ser, best)
    time.sleep(1.0)
    final_rows, final_status = run_set(ser, best, 1.0, 5.0)
    fsc, fstat = score(final_rows)
    _log("终评 score=%.1f %s" % (fsc, fstat))
    esp(ser, "gmanual")
    ser.close()

    # CSV
    csv_path = OUT / ("pid_autotune_%s_%s.csv" % (stamp, args.label))
    keys = []
    for r in results:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow(r)

    print("\n=== PID 自动调参汇总 ===")
    print("%-10s %-58s %8s %5s %s" % ("label", "params", "score", "ok", "status"))
    for r in sorted(results, key=lambda x: x["score"]):
        print("%-10s %-58s %8.1f %2d/%-2d %s"
              % (r["label"], fmt_set(r), r["score"] if r["score"] < 1e8 else 1e9,
                 r.get("ok", 0), r.get("n", 0), r.get("status")))
    print("\nBEST: %s" % fmt_set(best))
    print("还原命令：pid %.3f %.3f %.3f / pid2 %.3f %.3f %.3f / smooth %.4f / trk %.2f %.3f"
          % (best["kp"], best["ki"], best["kd"], best["kp2"], best["ki2"],
             best["kd2"], best["smooth"], best["trk"][0], best["trk"][1]))
    print("CSV: %s" % csv_path)

    try:
        live.wait(timeout=20)
    except subprocess.TimeoutExpired:
        live.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
