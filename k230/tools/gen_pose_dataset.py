#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成"靶纸 + 激光光斑"合成训练集（YOLO-Pose 格式）。

目的：原训练集里没有"激光打在靶纸上"的样本，导致光斑落在靶心时光斑把
白色靶心盖住，模型认不出。这里用程序合成大量此类样本，覆盖：
  * 光斑位置：靶心正中 / 各环 / 靶纸边缘 / 靶纸外
  * 光斑大小：2~14px（对应远距离~近距离）
  * 视角：透视/旋转/缩放/倾斜
  * 光照：亮/暗/偏色/高光
  * 背景：真实采集帧（有红色杂物、灭火器等）

关键点顺序与训练约定一致：TL, TR, BR, BL, 靶心。

用法：
    python3 tools/gen_pose_dataset.py --out dataset_pose --n 600 \
        --bg tools/laser_samples/wall_on_20261021_203803/raw
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np


def draw_target(size=420):
    """画一张 10 环靶纸（白底 + 红环 + 黑心），返回 (img, 靶心xy)。"""
    S = size
    img = np.full((S, S, 3), 246, np.uint8)
    c = S // 2
    # 10 个环：从外到内，红白相间
    max_r = S * 0.47
    for i in range(10, 0, -1):
        r = int(max_r * i / 10.0)
        if i % 2 == 0:
            cv2.circle(img, (c, c), r, (58, 58, 200), -1, cv2.LINE_AA)   # BGR 红
        else:
            cv2.circle(img, (c, c), r, (245, 245, 245), -1, cv2.LINE_AA)
    # 靶心黑点（10 号内的一小点）
    cv2.circle(img, (c, c), max(2, S // 90), (30, 30, 30), -1, cv2.LINE_AA)
    # 轻微纸纹/噪声
    noise = np.random.normal(0, 3.2, (S, S, 3))
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img, (c, c)


def rand_perspective(img, kpts, rng):
    """随机透视 + 旋转 + 缩放，关键点同步变换。"""
    h, w = img.shape[:2]
    ang = rng.uniform(-28, 28)
    sc = rng.uniform(0.55, 1.0)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    # 透视扰动
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jit = rng.uniform(-w * 0.10, w * 0.10, (4, 2)).astype(np.float32)
    dst = src + jit
    P = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, P, (w, h), flags=cv2.INTER_LINEAR,
                              borderValue=(240, 240, 240))
    pts = np.array([kpts], np.float32)
    pts = cv2.transform(pts, P)[0]
    pts = cv2.transform(np.array([pts], np.float32), M)[0]
    return out, pts


def paste(bg, fg, cx, cy, alpha_mask=None):
    """把 fg 贴到 bg 的 (cx,cy) 中心，返回 (合成图, 偏移)。"""
    fh, fw = fg.shape[:2]
    bh, bw = bg.shape[:2]
    x0 = int(cx - fw / 2)
    y0 = int(cy - fh / 2)
    x1, y1 = x0 + fw, y0 + fh
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1 = fw - max(0, x1 - bw)
    sy1 = fh - max(0, y1 - bh)
    if sx1 <= sx0 or sy1 <= sy0:
        return bg, (0, 0)
    dx0, dy0 = max(0, x0), max(0, y0)
    region = bg[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)]
    patch = fg[sy0:sy1, sx0:sx1]
    bg[dy0:dy0 + patch.shape[0], dx0:dx0 + patch.shape[1]] = patch
    return bg, (dx0 - (x0 + sx0), dy0 - (y0 + sy0))


def add_laser(img, x, y, rad, rng):
    """在图上叠加激光光斑：红晕 + 过曝白核。"""
    h, w = img.shape[:2]
    x, y = int(round(x)), int(round(y))
    if not (0 <= x < w and 0 <= y < h):
        return
    # 红晕（逐层衰减）
    for r, col in ((rad * 2 + 4, (40, 20, 120)), (rad + 3, (70, 40, 210)),
                   (max(2, rad), (110, 70, 250))):
        cv2.circle(img, (x, y), int(r), col, -1, cv2.LINE_AA)
    # 白核（近距离过曝；远距离时白核很小甚至没有）
    core = max(1, int(rad * rng.uniform(0.25, 0.55)))
    cv2.circle(img, (x, y), core, (250, 252, 255), -1, cv2.LINE_AA)
    # 轻微光晕模糊
    k = max(3, int(rad) * 2 + 1)
    roi = img[max(0, y - k):y + k, max(0, x - k):x + k]
    if roi.size:
        img[max(0, y - k):y + k, max(0, x - k):x + k] = cv2.GaussianBlur(roi, (3, 3), 0)


def photo_effects(img, rng):
    """光照/色彩/模糊/噪声。"""
    alpha = rng.uniform(0.65, 1.25)
    beta = rng.uniform(-40, 30)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    if rng.random() < 0.45:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if rng.random() < 0.3:
        img = cv2.convertScaleAbs(img, alpha=1.0, beta=0)
        b, g, r = cv2.split(img)
        r = np.clip(r.astype(np.int16) + rng.randint(-18, 18), 0, 255).astype(np.uint8)
        b = np.clip(b.astype(np.int16) + rng.randint(-18, 18), 0, 255).astype(np.uint8)
        img = cv2.merge([b, g, r])
    noise = np.random.normal(0, rng.uniform(1.5, 6.0), img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_pose")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--bg", default="tools/laser_samples/wall_on_20261021_203803/raw",
                    help="背景图目录（用真实采集帧）")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--laser-ratio", type=float, default=0.7,
                    help="有多大比例的样本带激光光斑")
    ap.add_argument("--center-ratio", type=float, default=0.45,
                    help="光斑打中靶心附近的比例（这是要解决的核心场景）")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    nprng = np.random.RandomState(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    bgs = []
    for p in sorted(Path(args.bg).glob("*.jpg"))[:40]:
        im = cv2.imread(str(p))
        if im is not None:
            bgs.append(cv2.resize(im, (args.size, args.size),
                                  interpolation=cv2.INTER_LINEAR))
    if not bgs:
        bgs = [np.full((args.size, args.size, 3), 110, np.uint8)]

    target_src, _ = draw_target(420)
    made = 0
    stats = {"with_laser": 0, "center_laser": 0}
    for i in range(args.n):
        bg = bgs[rng.randrange(len(bgs))].copy()
        # 靶纸随机缩放
        tsc = rng.uniform(0.45, 0.95)
        tsz = max(90, int(420 * tsc))
        tgt = cv2.resize(target_src, (tsz, tsz), interpolation=cv2.INTER_AREA)
        c = tsz // 2.0
        kp = [(0, 0), (tsz - 1, 0), (tsz - 1, tsz - 1), (0, tsz - 1), (c, c)]
        tgt, kp = rand_perspective(tgt, kp, rng)
        # 贴到随机位置
        px = rng.uniform(tsz * 0.45, args.size - tsz * 0.45)
        py = rng.uniform(tsz * 0.45, args.size - tsz * 0.45)
        bg, off = paste(bg, tgt, px, py)
        kp = [(x + off[0], y + off[1]) for (x, y) in kp]
        # 激光光斑
        if rng.random() < args.laser_ratio:
            if rng.random() < args.center_ratio:
                # 打在靶心附近（核心难点：盖住白色靶心）
                lx = kp[4][0] + rng.uniform(-tsz * 0.08, tsz * 0.08)
                ly = kp[4][1] + rng.uniform(-tsz * 0.08, tsz * 0.08)
            else:
                # 打在靶纸任意位置或靶纸外
                lx = rng.uniform(0, args.size)
                ly = rng.uniform(0, args.size)
            # 光斑半径随距离变化
            rad = rng.choice([2, 3, 4, 5, 6, 8, 10, 13])
            add_laser(bg, lx, ly, rad, rng)
            stats["with_laser"] += 1
            if abs(lx - kp[4][0]) < tsz * 0.1 and abs(ly - kp[4][1]) < tsz * 0.1:
                stats["center_laser"] += 1
        bg = photo_effects(bg, rng)

        # 标注：box 覆盖靶纸，关键点归一化
        xs = [p[0] for p in kp]
        ys = [p[1] for p in kp]
        bx0, bx1 = min(xs), max(xs)
        by0, by1 = min(ys), max(ys)
        cx = (bx0 + bx1) / 2 / args.size
        cy = (by0 + by1) / 2 / args.size
        bw = (bx1 - bx0) / args.size
        bh = (by1 - by0) / args.size
        if bw < 0.08 or bh < 0.08:
            continue
        parts = ["0", "%.6f" % cx, "%.6f" % cy, "%.6f" % bw, "%.6f" % bh]
        ok = True
        for (x, y) in kp:
            if not (0 <= x < args.size and 0 <= y < args.size):
                ok = False
                break
            parts += ["%.6f" % (x / args.size), "%.6f" % (y / args.size), "2"]
        if not ok:
            continue
        split = "val" if (i % 10 == 0) else "train"
        name = "syn_%04d" % i
        cv2.imwrite(str(out / "images" / split / (name + ".jpg")), bg,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        (out / "labels" / split / (name + ".txt")).write_text(
            " ".join(parts) + "\n", encoding="utf-8")
        made += 1

    (out / "data.yaml").write_text(
        "path: %s\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: target\n"
        "kpt_shape: [5, 3]\n"
        "flip_idx: [1, 0, 3, 2, 4]\n" % out.resolve(), encoding="utf-8")
    print("生成 %d 张 -> %s" % (made, out))
    print("其中带激光 %d 张，光斑盖住靶心附近 %d 张" %
          (stats["with_laser"], stats["center_laser"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
