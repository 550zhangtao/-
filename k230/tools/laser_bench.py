#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
激光识别效果验证 —— 一键采集 + 评测。

流程：
    1) 逐场景采集（每个场景 N 秒，全程 USB 推流）
    2) 自动跑 board/laser_spot.py 同源判据做离线评测
    3) 输出 markdown 报告

用法（按提示逐个场景操作）：
    python3 tools/laser_bench.py --scenes wall_on,wall_off,paper_on,paper_off
    python3 tools/laser_bench.py --scenes paper_on --duration 12

注意：一次只能占用一个串口；IDE 连接时不要跑。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "tools" / "laser_samples"

SCENE_DESC = {
    "wall_on": "激光打在墙上（保持不动）",
    "wall_off": "激光关闭（同一墙面背景，测��检）",
    "paper_on": "激光打在靶纸/白纸中心附近（保持不动）",
    "paper_off": "激光关闭，靶纸/白纸仍在画面中（测误检）",
    "paper_edge": "激光打在靶纸边缘/靶环上",
    "far_on": "远距离（约 2~3 米）激光打在靶纸上",
    "far_off": "远距离、激光关闭",
}


def run_capture(label: str, duration: float) -> Path:
    print("\n" + "=" * 70)
    print("场景：%s —— %s" % (label, SCENE_DESC.get(label, label)))
    print("准备就绪后按回车开始采集（%.0f 秒）…" % duration, flush=True)
    try:
        input()
    except EOFError:
        print("(非交互模式，直接开始)")
    cmd = [sys.executable, str(ROOT / "tools" / "capture_laser.py"),
           "--label", label, "--duration", str(duration)]
    subprocess.run(cmd, check=True)
    dirs = sorted(SAMPLES.glob("%s_*" % label))
    return dirs[-1] if dirs else SAMPLES / label


def run_eval(dirs) -> list:
    out = []
    for d in dirs:
        cmd = [sys.executable, str(ROOT / "tools" / "eval_laser.py"), str(d)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        for line in r.stdout.strip().splitlines():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def write_report(results, path: Path) -> None:
    lines = ["# 激光光斑识别效果报告", "",
             "生成时间：%s" % time.strftime("%Y-%m-%d %H:%M:%S"), "",
             "| 场景 | 帧数 | 检出 | 检出率 | x 均值±std | y 均值±std |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in results:
        name = Path(r["dir"]).name.rsplit("_", 2)[0]
        lines.append("| %s | %d | %d | %.1f%% | %s | %s |" % (
            name, r["frames"], r["hits"], r.get("hit_rate", 0),
            ("%.1f±%.2f" % (r["x_mean"], r["x_std"])) if "x_mean" in r else "-",
            ("%.1f±%.2f" % (r["y_mean"], r["y_std"])) if "y_mean" in r else "-"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n报告 ->", path)
    print("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="wall_on,wall_off,paper_on,paper_off")
    ap.add_argument("--duration", type=float, default=10.0)
    args = ap.parse_args()
    dirs = [run_capture(s.strip(), args.duration)
            for s in args.scenes.split(",") if s.strip()]
    results = run_eval(dirs)
    write_report(results, SAMPLES / "report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
