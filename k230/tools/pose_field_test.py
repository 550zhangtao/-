# -*- coding: utf-8 -*-
"""
pose_field_test.py —— K230 Pose 现场测试（逐帧记录 + 分阶段统计）
================================================================

时间轴（从第一帧开始，脚本自动按窗口统计）：
    0~5s    热身：靶纸居中
    5~17s   背景/无靶：镜头对桌面、墙、杂物
    17~29s  靶纸居中
    29~49s  移动/倾斜：左右移动、±30~45° 倾斜、远近变化
    49~61s  移出/遮挡：靶纸移出画面或用手遮住

用法：
    python3 tools/pose_field_test.py --duration 65 --conf 0.66 \
        --out ../k230_target_project/field_test.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from k230tool.link import K230Stream  # noqa: E402

PHASES = [
    ("warmup", 0, 5),
    ("background", 5, 17),
    ("target_center", 17, 29),
    ("move_tilt", 29, 49),
    ("removed_occluded", 49, 61),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=65)
    ap.add_argument("--conf", type=float, default=0.66)
    ap.add_argument("--q", type=int, default=60)
    ap.add_argument("--roi-max", type=int, default=160)
    ap.add_argument("--jpeg-every", type=int, default=1)
    ap.add_argument("--out", default="field_test.json")
    args = ap.parse_args(argv)

    st = K230Stream(
        cfg={"conf": args.conf, "q": args.q, "roi_max": args.roi_max,
             "roi_min": 128, "jpeg_every": args.jpeg_every,
             "stats": 5, "reset_on_exit": 1},
        ai=True, auto_restart=False, log=lambda m: print("[st]", m))
    st.start(timeout=40)
    t0 = time.time()
    rows = []
    last = 0
    last_jpg = None
    print("[*] 开始记录 %.0fs" % args.duration)
    while True:
        elapsed = time.time() - t0
        if elapsed >= args.duration:
            break
        jpg, idx = st.wait_frame(last, timeout=3)
        if jpg is None:
            continue
        last = idx
        last_jpg = jpg
        target = st.detections(idx) or {}
        det = target.get("det") or []
        box = det[0][:4] if det else None
        score = float(det[0][4]) if len(det) > 0 and len(det[0]) > 4 else None
        bull = target.get("bull") or {}
        kp = target.get("kp") or []
        rows.append({
            "t": round(time.time() - t0, 3),
            "seq": idx,
            "det": 1 if det else 0,
            "box": box,
            "score": score,
            "bull_status": bull.get("status"),
            "bull_x": bull.get("x"),
            "bull_y": bull.get("y"),
            "bull_conf": bull.get("conf"),
            "kp_ok": 1 if len(kp) >= 15 and float(kp[14]) >= 0.3 else 0,
        })

    st.stop()
    # 分阶段统计
    phases = {}
    for name, a, b in PHASES:
        seg = [r for r in rows if a <= r["t"] < b]
        n = len(seg)
        det_n = sum(r["det"] for r in seg)
        bull_n = sum(1 for r in seg if r["bull_status"] == "ok")
        scores = [r["score"] for r in seg if r["score"] is not None]
        phases[name] = {
            "window": [a, b],
            "frames": n,
            "detect_rate": round(det_n / n, 3) if n else 0,
            "bull_ok_rate": round(bull_n / n, 3) if n else 0,
            "score_mean": round(sum(scores) / len(scores), 3) if scores else None,
            "score_max": round(max(scores), 3) if scores else None,
            "status_counts": {},
        }
        for r in seg:
            s = r["bull_status"] or "none"
            phases[name]["status_counts"][s] = phases[name]["status_counts"].get(s, 0) + 1

    out = {"conf": args.conf, "duration": args.duration,
           "frames": len(rows), "phases": phases, "rows": rows}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    csv_path = Path(args.out).with_suffix(".csv")
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    if last_jpg:
        Path(args.out).with_suffix(".jpg").write_bytes(last_jpg)
    print(json.dumps(phases, ensure_ascii=False, indent=1))
    print("[✓] %s（%d 帧）" % (args.out, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
