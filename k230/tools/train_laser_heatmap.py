#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_laser_heatmap.py —— 激光点 ROI 热图小模型（v2：真实正样本 + 多形态）

v2 相比 v1：
  * 修正 BGR 颜色（v1 把红晕写成了蓝晕，模型学错）
  * 模拟两种真实形态：白核+淡粉晕（白墙）、强红团（深色面）
  * 注入板上实测的真实激光帧（on.png / lt_on_* / laser_on_static）
  * 负样本 50% 对准"最红/最亮"像素，专门学红杂物/高光

用法：
    python3 tools/train_laser_heatmap.py --epochs 10 --steps 300 --out models/laser_heatmap
"""
from __future__ import annotations
# --- PC 端脚本守卫：误在 K230 上运行时给出明确提示 ---
try:
    import sys as _sys
    if getattr(_sys, "implementation", None) is not None and _sys.implementation.name == "micropython":
        print("[PC 脚本] 这是电脑端工具，请在 PC 上运行；板端请运行 k230/board/*.py")
        raise SystemExit
except SystemExit:
    raise
except Exception:
    pass

import argparse
import glob
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent


# ---------------------------------------------------------------- 激光 PSF
def add_laser(img, x, y, rad, rng, style="red"):
    """style='red'：强红团（深色面）；'white'：白核+淡粉晕（白墙）。"""
    h, w = img.shape[:2]
    x, y = int(round(x)), int(round(y))
    if not (0 <= x < w and 0 <= y < h):
        return
    if style == "blob":
        # 近距离：白黄强核 + 红晕（2026-10-24 按缺陷截图补）
        rings = [(rad * 1.9, (30, 25, 110)), (rad * 1.45, (50, 45, 175)),
                 (rad * 1.05, (85, 90, 232)), (rad * 0.72, (150, 170, 250))]
        core = (240, 250, 255)
    elif style == "red":
        rings = [(rad * 2.4, (40, 25, 130)), (rad * 1.7, (55, 40, 190)),
                 (rad * 1.15, (75, 55, 235)), (rad * 0.75, (110, 85, 250))]
        core = (250, 252, 255)
    else:
        rings = [(rad * 2.0, (215, 195, 225)), (rad * 1.4, (200, 180, 218)),
                 (rad * 0.9, (230, 205, 238))]
        core = (255, 250, 255)
    for r, col in rings:
        cv2.circle(img, (x, y), int(max(1, r)), col, -1, cv2.LINE_AA)
    cr = max(1, int(round(rad * rng.uniform(0.30, 0.60))))
    cv2.circle(img, (x, y), cr, core, -1, cv2.LINE_AA)
    k = int(max(3, rad * 1.2))
    if k % 2 == 0:
        k += 1
    x0, y0 = max(0, x - k), max(0, y - k)
    x1, y1 = min(w, x + k), min(h, y + k)
    if x1 - x0 > 2 and y1 - y0 > 2:
        img[y0:y1, x0:x1] = cv2.GaussianBlur(img[y0:y1, x0:x1], (3, 3), 0)


def _scale_to_640(im):
    if im is None:
        return None
    if im.shape[1] == 640 and im.shape[0] == 480:
        return im
    return cv2.resize(im, (640, 480), interpolation=cv2.INTER_LINEAR)


def load_real_positives(paths, size=128):
    """从真实 on 帧里定位激光并裁 128x128。

    文件名形如 `xxx_<x>_<y>.jpg` 时用给定坐标（近距离大光斑自动定位会偏，
    手工标中心更可靠）；否则退回"最红+最亮"自动定位。
    """
    import re as _re
    crops = []
    for p in paths:
        im = _scale_to_640(cv2.imread(p))
        if im is None:
            continue
        m = _re.search(r"_(\d+)_(\d+)\.(?:jpg|jpeg|png)$", str(p))
        if m:
            xx, yy = int(m.group(1)), int(m.group(2))
        else:
            b, g, r = [im[:, :, i].astype(np.int16) for i in range(3)]
            pink = r - (g + b) // 2
            bright = (r + g + b) // 3
            score = pink + bright // 4
            yy, xx = np.unravel_index(score.argmax(), score.shape)
        x0 = max(0, min(im.shape[1] - size, int(xx - size // 2)))
        y0 = max(0, min(im.shape[0] - size, int(yy - size // 2)))
        crops.append(im[y0:y0 + size, x0:x0 + size].copy())
    return crops


def load_sweep_positives(d, csv_path, size=128):
    """按 coords.csv 的模型坐标裁剪扫动帧（x<0 的漏检跳过）。"""
    import csv as _csv
    crops = []
    try:
        rows = list(_csv.DictReader(open(csv_path)))
    except Exception:
        return crops
    for r in rows:
        try:
            x = float(r["x"]); y = float(r["y"])
        except Exception:
            continue
        if x < 0:
            continue
        p = Path(d) / ("s%03d.jpg" % int(r["frame"]))
        if not p.exists():
            continue
        im = cv2.imread(str(p))
        if im is None:
            continue
        x0 = max(0, min(im.shape[1] - size, int(x - size // 2)))
        y0 = max(0, min(im.shape[0] - size, int(y - size // 2)))
        crops.append(im[y0:y0 + size, x0:x0 + size].copy())
    return crops


def gather_backgrounds(globs, cap=320):
    paths = []
    for g in globs:
        paths.extend(sorted(glob.glob(g)))
    paths = [p for p in paths if "_spot" not in Path(p).stem]
    rng = random.Random(7)
    rng.shuffle(paths)
    paths = paths[:cap]
    bgs = []
    for p in paths:
        im = cv2.imread(p)
        if im is None or im.shape[0] < 160 or im.shape[1] < 160:
            continue
        bgs.append(im)
    return bgs, len(paths)


def gaussian_heat(n, cx, cy, sigma):
    yy, xx = np.mgrid[0:n, 0:n]
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma)).astype(np.float32)


def _hard_neg_crop(bg, size, rng):
    """负样本 50% 对准最红/最亮像素（红杂物、高光）。"""
    h, w = bg.shape[:2]
    if rng.random() < 0.5:
        b, g, r = [bg[:, :, i].astype(np.int16) for i in range(3)]
        score = (r - (g + b) // 2) + ((r + g + b) // 3) // 4
        flat = score.ravel()
        thr = np.partition(flat, -20)[-20]
        idx = np.where(flat >= thr)[0]
        k = int(rng.choice(idx))
        yy, xx = divmod(k, w)
    else:
        yy, xx = rng.randint(0, h - 1), rng.randint(0, w - 1)
    x0 = max(0, min(w - size, int(xx - size // 2)))
    y0 = max(0, min(h - size, int(yy - size // 2)))
    return bg[y0:y0 + size, x0:x0 + size]


def make_sample(bg, size, rng, positive):
    half = size // 2
    if positive:
        h, w = bg.shape[:2]
        if rng.random() < 0.6:
            cx = rng.uniform(w * 0.25, w * 0.75)
            cy = rng.uniform(h * 0.25, h * 0.75)
        else:
            cx = rng.uniform(half + 2, w - half - 2)
            cy = rng.uniform(half + 2, h - half - 2)
        canvas = bg.copy()
        r = rng.random()
        if r < 0.22:
            rad = rng.uniform(1.2, 3.0)      # 远距离：小光斑
        elif r < 0.55:
            rad = rng.uniform(3.0, 6.5)      # 中距离
        elif r < 0.78:
            rad = rng.uniform(6.5, 11.0)     # 近一点
        elif r < 0.90:
            rad = rng.uniform(11.0, 20.0)    # 近距离：大光斑/泛光（2026-10-24 加）
        elif r < 0.97:
            rad = rng.uniform(20.0, 34.0)    # 极近：饱和白核 + 红晕
        else:
            rad = rng.uniform(34.0, 55.0)    # 贴脸：整片泛白（按手背截图）
        if rad >= 11.0:
            style = "blob" if rng.random() < 0.65 else ("white" if rng.random() < 0.5 else "red")
        else:
            style = "red" if rng.random() < 0.5 else "white"
        add_laser(canvas, cx, cy, rad, rng, style)
        px = cx + rng.uniform(-half * 0.35, half * 0.35)
        py = cy + rng.uniform(-half * 0.35, half * 0.35)
        x0 = int(round(px - half)); y0 = int(round(py - half))
        x0 = max(0, min(w - size, x0)); y0 = max(0, min(h - size, y0))
        crop = canvas[y0:y0 + size, x0:x0 + size]
        heat = gaussian_heat(size // 4, (cx - x0) / 4.0, (cy - y0) / 4.0, 2.0)
    else:
        crop = _hard_neg_crop(bg, size, rng)
        heat = np.zeros((size // 4, size // 4), np.float32)
    if rng.random() < 0.6:
        crop = cv2.convertScaleAbs(crop, alpha=rng.uniform(0.5, 1.15),
                                   beta=rng.uniform(-25, 20))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), heat


# ---------------------------------------------------------------- 模型
def conv(cin, cout, k=3, s=1):
    return nn.Conv2d(cin, cout, k, s, k // 2, bias=False)


class TinyHeat(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            conv(3, 16, 3, 2), nn.ReLU(inplace=True), conv(16, 16), nn.ReLU(inplace=True),
            conv(16, 32, 3, 2), nn.ReLU(inplace=True), conv(32, 32), nn.ReLU(inplace=True),
            conv(32, 64, 3, 2), nn.ReLU(inplace=True), conv(64, 64), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv(64, 32), nn.ReLU(inplace=True), conv(32, 1),
        )

    def forward(self, x):
        return self.head(self.enc(x))


# ---------------------------------------------------------------- 训练
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg-globs", nargs="*", default=[
        str(REPO / "recovered/dataset_pose_v3/images/train/*.jpg"),
        "/tmp/k230_laser_grab/*.jpg",
        str(REPO / "recovered/k230_target_project/captures/neg/*"),
        str(REPO / "scratch/hardset/red/*"),
    ])
    ap.add_argument("--real-on", nargs="*", default=[
        "/tmp/laser_real/on.png", "/tmp/laser_real/on_syn.jpg",
        "/tmp/laser_real/lt_on_202320.jpg", "/tmp/laser_real/lt_on_202554.jpg",
        "/tmp/laser_real/lt_on_203504.jpg", "/tmp/laser_real/lt_on1_640.jpg",
        "/tmp/laser_real/laser_on_static.jpg", "/tmp/k230_now.jpg",
    ])
    ap.add_argument("--real-sweep-dir", default="/tmp/sweep_real")
    ap.add_argument("--real-sweep-csv", default="/tmp/sweep/coords.csv")
    ap.add_argument("--out", default=str(ROOT / "models" / "laser_heatmap"))
    ap.add_argument("--imgsz", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cap", type=int, default=320)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    bgs, n = gather_backgrounds(args.bg_globs, args.cap)
    real_pos = load_real_positives(args.real_on, args.imgsz)
    if args.real_sweep_csv and args.real_sweep_dir:
        sp = load_sweep_positives(args.real_sweep_dir, args.real_sweep_csv, args.imgsz)
        if sp:
            real_pos += sp
            print("[*] 扫动真实正样本 %d 张" % len(sp))
    print("[*] 背景 %d 张，真实正样本 %d 张" % (len(bgs), len(real_pos)))
    if len(bgs) < 10 or not real_pos:
        raise SystemExit("数据不足")
    val_bgs = bgs[-max(20, len(bgs) // 5):]
    train_bgs = bgs[:len(bgs) - len(val_bgs)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyHeat().to(device)
    print("[*] 参数 %.1fK device=%s" % (sum(p.numel() for p in model.parameters()) / 1e3, device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * args.steps)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.0, device=device))

    def batch(bgs_, bs):
        xs, ys = [], []
        for _ in range(bs):
            if rng.random() < 0.5:
                if rng.random() < 0.5:
                    crop, heat = make_sample(rng.choice(bgs_), args.imgsz, rng, True)
                else:
                    crop = cv2.cvtColor(rng.choice(real_pos), cv2.COLOR_BGR2RGB)
                    if rng.random() < 0.5:
                        crop = cv2.flip(crop, 1)
                    heat = gaussian_heat(args.imgsz // 4, args.imgsz // 8.0,
                                         args.imgsz // 8.0, 2.0)
            else:
                crop, heat = make_sample(rng.choice(bgs_), args.imgsz, rng, False)
            xs.append(crop.transpose(2, 0, 1))
            ys.append(heat[None])
        x = torch.from_numpy(np.stack(xs)).float().to(device) / 255.0
        y = torch.from_numpy(np.stack(ys)).float().to(device)
        return x, y

    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for it in range(args.steps):
            x, y = batch(train_bgs, args.batch)
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss)
        print("[ep %d] loss %.4f" % (ep, tot / args.steps))

    model.eval()
    tp = fp = fn = 0; errs = []
    with torch.no_grad():
        for _ in range(200):
            if rng.random() < 0.5:
                crop, heat = make_sample(rng.choice(val_bgs), args.imgsz, rng, True)
                pos = True
            else:
                crop, heat = make_sample(rng.choice(val_bgs), args.imgsz, rng, False)
                pos = False
            x = torch.from_numpy(crop.transpose(2, 0, 1)[None]).float().to(device) / 255.0
            out = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            got = out.max() >= 0.5
            if pos:
                if got:
                    tp += 1
                    gy, gx = np.unravel_index(heat.argmax(), heat.shape)
                    py, px = np.unravel_index(out.argmax(), out.shape)
                    errs.append(math.hypot(px - gx, py - gy) * 4)
                else:
                    fn += 1
            elif got:
                fp += 1
    print("[val] recall %.1f%% precision %.1f%% err %.1fpx" % (
        100 * tp / max(1, tp + fn), 100 * tp / max(1, tp + fp),
        float(np.median(errs)) if errs else -1))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(Path(args.out).with_suffix(".pt")))
    model.eval().cpu()
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
    onnx_path = str(Path(args.out).with_suffix(".onnx"))
    torch.onnx.export(model, dummy, onnx_path, opset_version=12,
                      input_names=["input"], output_names=["heat"], dynamic_axes=None)
    print("[OK] %s / %s" % (Path(args.out).with_suffix(".pt"), onnx_path))


if __name__ == "__main__":
    main()
