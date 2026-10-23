#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_laser_heatmap.py —— 用 nncase Simulator 在 PC 上评估激光热图 kmodel。

测试三类：
  1) 真实 snapshot（激光在 (317,261)）裁 128x128 -> 看峰值位置
  2) 真实负样本（grab 帧任意位置）-> 峰值应低
  3) 合成正样本（add_laser 于随机位置）-> 峰值位置误差

用法：
    python3 tools/eval_laser_heatmap.py --kmodel models/laser_heatmap.kmodel \
        --pos /tmp/k230_now.jpg --neg-dir /tmp/k230_laser_grab
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_laser_heatmap import add_laser  # noqa: E402

nncase = None
os.environ.setdefault("DOTNET_ROOT", os.path.expanduser("~/.dotnet"))


def load_sim(kmodel):
    global nncase
    import nncase
    sim = nncase.Simulator()
    sim.load_model(Path(kmodel).read_bytes())
    return sim


def infer(sim, crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    x = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None].astype(np.uint8))
    sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(x))
    sim.run()
    return sim.get_output_tensor(0).to_numpy()


def peak(y):
    # kmodel 输出是 logits，先 sigmoid
    a = np.asarray(y).reshape(-1)
    n = int(round(len(a) ** 0.5))
    hm = a.reshape(n, n)
    iy, ix = np.unravel_index(int(hm.argmax()), hm.shape)
    v = float(hm.max())
    return 1.0 / (1.0 + math.exp(-v)), int(ix), int(iy), n


def crop_at(img, cx, cy, size=128):
    h, w = img.shape[:2]
    x0 = max(0, min(w - size, int(cx - size // 2)))
    y0 = max(0, min(h - size, int(cy - size // 2)))
    return img[y0:y0 + size, x0:x0 + size], x0, y0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmodel", default="models/laser_heatmap.kmodel")
    ap.add_argument("--pos", default="/tmp/k230_now.jpg")
    ap.add_argument("--neg-dir", default="/tmp/k230_laser_grab")
    ap.add_argument("--n-neg", type=int, default=20)
    args = ap.parse_args()

    sim = load_sim(args.kmodel)
    print("[*] kmodel loaded:", args.kmodel)

    # 1) 真实正样本
    if Path(args.pos).exists():
        im = cv2.imread(args.pos)
        c, x0, y0 = crop_at(im, 317, 261)
        y = infer(sim, c)
        v, ix, iy, n = peak(y)
        print("[real+] out shape %s  peak %.3f at cell(%d,%d)/%d -> global(%d,%d)" % (
            np.asarray(y).shape, v, ix, iy, n, x0 + ix * 4 + 2, y0 + iy * 4 + 2))

    # 2) 真实负样本
    negs = sorted(Path(args.neg_dir).glob("*.jpg"))[:args.n_neg]
    if negs:
        maxv = []
        for p in negs:
            im = cv2.imread(str(p))
            c, _, _ = crop_at(im, 320, 240)
            y = infer(sim, c)
            maxv.append(peak(y)[0])
        print("[real-] n=%d  peak mean %.3f  max %.3f  >0.5: %d" % (
            len(maxv), float(np.mean(maxv)), float(np.max(maxv)),
            sum(v >= 0.5 for v in maxv)))

    # 3) 合成正样本（同一张图，5 个位置）
    if Path(args.pos).exists():
        im = cv2.imread(args.pos)
        rng = np.random.RandomState(0)
        errs = []
        for _ in range(8):
            cx = int(rng.randint(80, 560)); cy = int(rng.randint(80, 400))
            canvas = im.copy()
            add_laser(canvas, cx, cy, float(rng.uniform(4, 10)), rng)
            c, x0, y0 = crop_at(canvas, cx, cy)
            y = infer(sim, c)
            v, ix, iy, n = peak(y)
            gx = x0 + ix * 4 + 2; gy = y0 + iy * 4 + 2
            errs.append((v, ((gx - cx) ** 2 + (gy - cy) ** 2) ** 0.5))
        print("[synth+] n=%d  peak mean %.2f  loc err median %.1fpx" % (
            len(errs), float(np.mean([e[0] for e in errs])),
            float(np.median([e[1] for e in errs]))))


if __name__ == "__main__":
    main()
