#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""motion_test_laser.py —— 激光光斑运动/遮挡/预测实测（需板子串口空闲）

把 `k230tool live` 跑起来，同时按时间轴提示你移动/遮挡激光，逐帧记录
`latest.json` 里的 `laser`（x/y/pred/coast/budget/src/status）、`laser_ms`
和板端 fps，结束后输出：

  * 实测帧 / 标定帧 / 无位置帧
  * 预测误差 |pred - measured|（只在 status=ok 的实测帧统计）
  * 最大 coast 帧数、最大 laser_ms、预算命中次数
  * 最长丢失段与恢复耗时
  * fps（板端自报优先）

准备：
  1. VSCode CanMV 插件点 **Disconnect Board**，释放 `/dev/ttyACM1`；
  2. K230 在调试模式（`/sdcard/main.py` 不占 REPL；`mode.sh debug`）；
  3. **ESP32 切到 R1（`gmanual`）让激光亮，但不要 R2 跟踪**，
     否则云台会跟着动，干扰运动测试。

用法：
    python3 scripts/motion_test_laser.py --duration 28

时间轴（脚本会打印）：
    0-8s   慢速左右/上下移动光斑
    8-14s  快速来回/画圈
    14-20s 用手遮住激光或关掉激光 3~4s，再移开
    20-28s 恢复慢速移动
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path


def read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=28.0)
    ap.add_argument("--out", default="/tmp/motion_test")
    ap.add_argument("--poll", type=float, default=0.10)
    ap.add_argument("--full-every", type=int, default=9999,
                    help="测试期间禁用/降低全图 JPEG，避免 370ms 卡顿")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    latest = out / "latest.json"
    if latest.exists():
        latest.unlink()

    k230 = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, "-m", "k230tool", "live",
           "--duration", str(args.duration + 8), "--out", str(out),
           "--full-every", str(args.full_every), "--quiet"]
    print("[*] " + " ".join(cmd))
    print("[!] 先确认：ESP32 在 R1（gmanual，激光亮、云台不跟踪），K230 在调试模式")
    proc = subprocess.Popen(cmd, cwd=k230)

    # 等 live 出第一帧（板端脚本启动约 3~5s）
    t0 = time.time()
    first = None
    while time.time() - t0 < 25:
        rec = read_json(latest)
        if rec and rec.get("seq"):
            first = rec
            break
        time.sleep(0.2)
    if first is None:
        print("[x] 没等到 latest.json：检查串口是否被 CanMV IDE 占用、K230 是否在工作模式")
        proc.terminate()
        return 2
    print("[✓] 收流开始，按时间轴操作（从现在开始计时）")

    phases = [
        (0.0, "【0-8s】慢速左右/上下移动光斑"),
        (8.0, "【8-14s】快速来回 / 画圈"),
        (14.0, "【14-20s】遮住激光 3~4s，再移开"),
        (20.0, "【20-28s】恢复慢速移动"),
    ]
    recs = {}
    t_start = time.time()
    last_phase = -1
    try:
        while time.time() - t_start < args.duration:
            el = time.time() - t_start
            pi = 0
            for i, (ts, _) in enumerate(phases):
                if el >= ts:
                    pi = i
            if pi != last_phase:
                print("  >>> " + phases[pi][1])
                last_phase = pi
            rec = read_json(latest)
            if rec and rec.get("seq") is not None:
                recs[int(rec["seq"])] = rec
                # 实时打印一行，便于观察
                laser = rec.get("laser") or {}
                print("  t=%4.1fs seq=%4s laser=%s src=%s pred=%s coast=%s lms=%sms fps=%s"
                      % (el, rec.get("seq"), laser.get("status"), laser.get("src"),
                         laser.get("pred"), laser.get("coast"),
                         rec.get("laser_ms"), (rec.get("board") or {}).get("fps", rec.get("fps"))))
            time.sleep(args.poll)
    finally:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.terminate()

    rows = [recs[k] for k in sorted(recs)]
    print("\n[*] 收集 %d 帧（去重）" % len(rows))
    if not rows:
        return 2

    measured = [r for r in rows if (r.get("laser") or {}).get("status") == "ok"]
    track = [r for r in rows if (r.get("laser") or {}).get("status") == "track"]
    calib = [r for r in rows if (r.get("laser") or {}).get("status") == "calib"]
    none = [r for r in rows if (r.get("laser") or {}).get("status") in (None, "none")]

    errs = []
    for r in measured:
        l = r["laser"]
        pred = l.get("pred")
        if pred and len(pred) == 2:
            errs.append(math.hypot(float(pred[0]) - float(l["x"]),
                                   float(pred[1]) - float(l["y"])))

    coasts = [int((r.get("laser") or {}).get("coast", 0)) for r in rows]
    lms = [int(r.get("laser_ms") or 0) for r in rows]
    fps = [float((r.get("board") or {}).get("fps") or r.get("fps") or 0) for r in rows]
    fps = [f for f in fps if f > 0]
    budget_hits = sum(1 for r in rows if (r.get("laser") or {}).get("budget"))

    # 最长连续丢失（status != ok）
    best_run = cur = 0
    best_start = cur_start = 0
    for i, r in enumerate(rows):
        st = (r.get("laser") or {}).get("status")
        if st not in ("ok", "track"):
            if cur == 0:
                cur_start = i
            cur += 1
            if cur > best_run:
                best_run, best_start = cur, cur_start
        else:
            cur = 0
    recover_s = None
    for r in rows[best_start + best_run:]:
        if (r.get("laser") or {}).get("status") == "ok":
            recover_s = round((r.get("ts", 0) - rows[best_start].get("ts", 0)), 2)
            break

    def agg(a):
        if not a:
            return {"n": 0}
        return {"n": len(a), "mean": round(statistics.mean(a), 2),
                "median": round(statistics.median(a), 2),
                "p95": round(sorted(a)[max(0, int(len(a) * 0.95) - 1)], 2),
                "max": round(max(a), 2)}

    summary = {
        "frames": len(rows),
        "measured": len(measured),
        "track": len(track),
        "calib": len(calib),
        "none": len(none),
        "pred_err_px": agg(errs),
        "coast_frames": agg(coasts),
        "laser_ms": agg(lms),
        "fps": agg(fps),
        "budget_hits": budget_hits,
        "longest_lost_frames": best_run,
        "recover_after_lost_s": recover_s,
        "duration_s": args.duration,
    }
    (out / "motion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "motion_frames.csv").open("w", encoding="utf-8") as f:
        f.write("seq,ts,status,src,x,y,pred_x,pred_y,coast,conf,budget,laser_ms,fps\n")
        for r in rows:
            l = r.get("laser") or {}
            pred = l.get("pred") or [None, None]
            f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (
                r.get("seq"), r.get("ts"), l.get("status"), l.get("src"),
                l.get("x"), l.get("y"), pred[0] if len(pred) > 0 else None,
                pred[1] if len(pred) > 1 else None, l.get("coast"),
                l.get("conf"), l.get("budget"), r.get("laser_ms"),
                (r.get("board") or {}).get("fps", r.get("fps"))))

    print("\n===== 运动/遮挡测试汇总 =====")
    print("  帧数 %d  实测 %d  跟踪预测 %d  标定/丢失 %d  无位置 %d"
          % (len(rows), len(measured), len(track), len(calib), len(none)))
    print("  预测误差 |pred-measured|: %s" % summary["pred_err_px"])
    print("  coast 最大 %s 帧 / laser_ms %s / fps %s"
          % (summary["coast_frames"].get("max"), summary["laser_ms"],
             summary["fps"]))
    print("  预算命中 %d 次；最长丢失 %d 帧，恢复 %.2fs"
          % (budget_hits, best_run, recover_s if recover_s is not None else -1))
    print("[✓] %s/motion_summary.json, motion_frames.csv" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
