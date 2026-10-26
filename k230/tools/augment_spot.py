#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给靶纸 Pose 数据集叠加"激光光斑"，解决"光斑盖住靶心导致漏检"。

背景：原训练集里靶纸上**没有激光光斑**，模型从没见过"靶心被一团白光+
红晕盖住"的样子，所以真打靶时光斑一落到靶心就识别不到靶纸。

做法：在原数据集基础上，按 5 个关键点里的靶心位置合成激光光斑，生成
"同一张靶纸 + 不同光斑位置/大小"的新样本。标注**保持不变**（靶纸没动），
所以不需要重新标点。

光斑参数按真机实测调：
  * 白核：直径 2~6px 过曝白（min(R,G,B) ≈ 240+）
  * 红晕：直径 8~30px，RGB≈(255,80,120) 向外衰减到 (140,40,70)
  * 位置：60% 压靶心、25% 靶纸其他位置、15% 靶纸外

用法：
    python3 tools/augment_spot.py \
        --src /path/dataset_pose_v2_B --out /path/dataset_pose_spot \
        --copies 2
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def add_laser(img, x, y, rad, rng):
    """叠加激光光斑（BGR）。rad 为白核外径的一半，整体直径约 4*rad。"""
    h, w = img.shape[:2]
    x, y = int(round(x)), int(round(y))
    if not (0 <= x < w and 0 <= y < h):
        return
    # 红晕：由外向内加亮（BGR 中 R 最高）
    rings = [
        (rad * 2.4, (40, 25, 130)),
        (rad * 1.7, (55, 40, 190)),
        (rad * 1.15, (75, 55, 235)),
        (rad * 0.75, (110, 85, 250)),
    ]
    for r, col in rings:
        cv2.circle(img, (x, y), int(max(1, r)), col, -1, cv2.LINE_AA)
    # 过曝白核（近距离大、远距离小甚至没有）
    core = max(1, int(round(rad * rng.uniform(0.30, 0.60))))
    cv2.circle(img, (x, y), core, (250, 252, 255), -1, cv2.LINE_AA)
    # 轻微柔化，避免硬边
    k = int(max(3, rad * 1.2))
    if k % 2 == 0:
        k += 1
    x0, y0 = max(0, x - k), max(0, y - k)
    x1, y1 = min(w, x + k), min(h, y + k)
    if x1 - x0 > 2 and y1 - y0 > 2:
        img[y0:y1, x0:x1] = cv2.GaussianBlur(img[y0:y1, x0:x1], (3, 3), 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--copies", type=int, default=2,
                    help="每张原图额外生成几份带光斑样本")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--center-ratio", type=float, default=0.60)
    ap.add_argument("--on-target-ratio", type=float, default=0.25)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    src, out = Path(args.src), Path(args.out)

    for sub in ("images/train", "images/val", "images/test",
                "labels/train", "labels/val", "labels/test"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    total = 0
    spot_n = 0
    for split in ("train", "val", "test"):
        # 1) 原图+原标注原样复制（保证"无光斑"样本不丢）
        for img_p in sorted((src / "images" / split).glob("*.jpg")):
            shutil.copy2(img_p, out / "images" / split / img_p.name)
            lab = src / "labels" / split / (img_p.stem + ".txt")
            if lab.exists():
                shutil.copy2(lab, out / "labels" / split / lab.name)
            total += 1
        # 2) 生成带光斑的变体（只对 train 做，val/test 保持原样便于对比）
        if split != "train":
            continue
        for img_p in sorted((src / "images" / split).glob("*.jpg")):
            lab = src / "labels" / split / (img_p.stem + ".txt")
            if not lab.exists():
                continue
            line = lab.read_text(encoding="utf-8").strip().split()
            if len(line) < 5 + 5 * 3:
                continue
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            h, w = img.shape[:2]
            # 关键点 5 = 靶心（归一化）
            bx = float(line[5 + 4 * 3]) * w
            by = float(line[5 + 4 * 3 + 1]) * h
            # 靶纸 bbox（归一化）
            cx, cy = float(line[1]) * w, float(line[2]) * h
            bw, bh = float(line[3]) * w, float(line[4]) * h
            for k in range(args.copies):
                canvas = img.copy()
                r = rng.random()
                if r < args.center_ratio:
                    # ★ 核心场景：光斑压在靶心上
                    lx = bx + rng.uniform(-bw * 0.07, bw * 0.07)
                    ly = by + rng.uniform(-bh * 0.07, bh * 0.07)
                elif r < args.center_ratio + args.on_target_ratio:
                    # 靶纸上的其他位置（各环）
                    lx = cx + rng.uniform(-bw / 2, bw / 2)
                    ly = cy + rng.uniform(-bh / 2, bh / 2)
                else:
                    # 靶纸外
                    lx = rng.uniform(0, w - 1)
                    ly = rng.uniform(0, h - 1)
                # 光斑大小：按靶纸在画面里的尺寸换算，模拟不同距离
                ref = max(24.0, min(bw, bh))
                rad = rng.uniform(0.020, 0.075) * ref
                rad = max(1.5, min(14.0, rad))
                add_laser(canvas, lx, ly, rad, rng)
                name = "%s_spot%d" % (img_p.stem, k)
                cv2.imwrite(str(out / "images" / split / (name + ".jpg")),
                            canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
                shutil.copy2(lab, out / "labels" / split / (name + ".txt"))
                total += 1
                spot_n += 1

    (out / "data.yaml").write_text(
        "path: %s\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names: ['target']\n"
        "kpt_shape: [5, 3]\n"
        "flip_idx: [1, 0, 3, 2, 4]\n" % out.resolve(), encoding="utf-8")
    print("总样本 %d（其中带光斑 %d）-> %s" % (total, spot_n, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
