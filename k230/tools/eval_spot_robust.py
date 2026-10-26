#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评估"靶心被激光光斑盖住"时的检出能力，并可对比多个模型。

从**未参与训练**的 test 集生成光斑变体：
  * 靶心正中（最关键）
  * 靶心偏移
  * 靶纸其他环
然后用每个模型评估检出率 + 靶心误差。

用法：
    python3 tools/eval_spot_robust.py --data <dataset_pose_v2_B> \
        --model old.pt --model new.pt [--tmp /tmp/spot_eval]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def add_laser(img, x, y, rad, rng):
    h, w = img.shape[:2]
    x, y = int(round(x)), int(round(y))
    if not (0 <= x < w and 0 <= y < h):
        return
    for r, col in ((rad * 2.4, (40, 25, 130)), (rad * 1.7, (55, 40, 190)),
                   (rad * 1.15, (75, 55, 235)), (rad * 0.75, (110, 85, 250))):
        cv2.circle(img, (x, y), int(max(1, r)), col, -1, cv2.LINE_AA)
    core = max(1, int(round(rad * rng.uniform(0.3, 0.6))))
    cv2.circle(img, (x, y), core, (250, 252, 255), -1, cv2.LINE_AA)
    k = int(max(3, rad * 1.2))
    if k % 2 == 0:
        k += 1
    x0, y0 = max(0, x - k), max(0, y - k)
    x1, y1 = min(w, x + k), min(h, y + k)
    if x1 - x0 > 2 and y1 - y0 > 2:
        img[y0:y1, x0:x1] = cv2.GaussianBlur(img[y0:y1, x0:x1], (3, 3), 0)


def build_cases(data: Path, tmp: Path, seed=3):
    rng = np.random.RandomState(seed)
    cases = []          # (image_path, gx, gy, kind)
    src = data / "images" / "test"
    lab = data / "labels" / "test"
    out = tmp / "images"
    if out.exists():
        shutil.rmtree(tmp)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src.glob("*.jpg")):
        lp = lab / (f.stem + ".txt")
        if not lp.exists():
            continue
        p = lp.read_text().split()
        if len(p) < 20:
            continue
        im = cv2.imread(str(f))
        if im is None:
            continue
        h, w = im.shape[:2]
        bx, by = float(p[17]) * w, float(p[18]) * h
        cx, cy = float(p[1]) * w, float(p[2]) * h
        bw, bh = float(p[3]) * w, float(p[4]) * h
        ref = max(24.0, min(bw, bh))
        for kind, (ox, oy) in (
                ("center", (0, 0)),
                ("center_l", (-int(bw * 0.06), 0)),
                ("center_r", (int(bw * 0.06), 0)),
                ("ring_mid", (int(bw * 0.18), int(bh * 0.10))),
                ("ring_out", (int(bw * 0.30), -int(bh * 0.22)))):
            for k in range(2):
                canvas = im.copy()
                lx = bx + ox + rng.uniform(-2, 2)
                ly = by + oy + rng.uniform(-2, 2)
                rad = float(rng.uniform(0.10, 0.22) * ref)
                rad = max(6.0, min(38.0, rad))
                add_laser(canvas, lx, ly, rad, rng)
                name = "%s_%s%d.jpg" % (f.stem, kind, k)
                cv2.imwrite(str(out / name), canvas,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                cases.append((str(out / name), bx, by, kind))
                n += 1
    return cases, n


def evaluate(models, cases, groups):
    from ultralytics import YOLO
    results = {}
    for mp in models:
        m = YOLO(mp)
        # 第一次预测会建 warmup，单独跑一次
        m.predict(cases[0][0], imgsz=320, conf=0.25, verbose=False)
        stat = {g: [0, 0, []] for g in groups}
        for path, gx, gy, kind in cases:
            res = m.predict(path, imgsz=320, conf=0.25, verbose=False)
            grp = "靶心正中" if kind.startswith("center") else "靶环/边缘"
            stat[grp][1] += 1
            if not res or res[0].keypoints is None or len(res[0].keypoints) == 0:
                continue
            kp = res[0].keypoints.data[0]
            if kp.shape[0] < 5:
                continue
            d = float(((float(kp[4][0]) - gx) ** 2 +
                       (float(kp[4][1]) - gy) ** 2) ** 0.5)
            if d < 25:
                stat[grp][0] += 1
                stat[grp][2].append(d)
        results[mp] = stat
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--tmp", default="/tmp/spot_eval")
    args = ap.parse_args()
    data = Path(args.data)
    cases, n = build_cases(data, Path(args.tmp))
    print("构造测试样本 %d 张（从 test 集 %d 张原图派生，均未参与训练）" %
          (n, len({c[0].rsplit('_', 2)[0] for c in cases})))
    groups = ["靶心正中", "靶环/边缘"]
    res = evaluate(args.model, cases, groups)
    for mp, stat in res.items():
        print("\n=== %s ===" % Path(mp).name)
        for g in groups:
            hit, tot, errs = stat[g]
            if tot == 0:
                continue
            e = np.median(errs) if errs else float("nan")
            print("  %-8s 检出 %3d/%3d = %5.1f%%   靶心中位误差 %.1fpx" % (
                g, hit, tot, 100.0 * hit / tot, e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
