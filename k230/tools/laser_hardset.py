#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从真实采集帧构造"困难场景"测试集（几何/光照/颜色退化），用于泛化验证。

三类难点：
  far   远距离：把光斑及其周围缩小，模拟单位面积能量下降（光斑变小、变暗）
  dark  黑背景：整体压暗，模拟夜间/黑绒布背景
  red   红色杂物：在光斑附近粘贴红块，模拟红色干扰物

用法：
    python3 tools/laser_hardset.py <src_dir> <out_dir>
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def find_spot(img, gt):
    """用采集时的真值坐标定位光斑（650x480 坐标系换算到图像）。"""
    h, w = img.shape[:2]
    return int(round(gt[0] * w / 640.0)), int(round(gt[1] * h / 480.0))


def make_far(img, cx, cy, scale=0.45):
    """模拟远距离：把以光斑为中心的 48x48 区域缩小再放大回去。"""
    h, w = img.shape[:2]
    r = 24
    x0, y0 = max(0, cx - r), max(0, cy - r)
    x1, y1 = min(w, cx + r), min(h, cy + r)
    patch = img[y0:y1, x0:x1]
    if patch.size == 0:
        return img.copy()
    ph, pw = patch.shape[:2]
    small = cv2.resize(patch, (max(3, int(pw * scale)), max(3, int(ph * scale))),
                       interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (pw, ph), interpolation=cv2.INTER_LINEAR)
    out = img.copy()
    out[y0:y1, x0:x1] = back
    # 再整体压一点亮度，模拟距离衰减
    return cv2.convertScaleAbs(out, alpha=0.92, beta=-6)


def make_dark(img, cx, cy):
    """黑背景：整体压暗（模拟夜间/黑绒布）。"""
    return cv2.convertScaleAbs(img, alpha=0.45, beta=-8)


def make_red(img, cx, cy):
    """红色杂物：在光斑附近 60~120px 处粘贴红色块（模拟红绳、红纸、红壳）。"""
    out = img.copy()
    h, w = img.shape[:2]
    rng = np.random.RandomState(cx * 7 + cy)
    for k in range(3):
        ang = rng.uniform(0, 6.28)
        dist = rng.uniform(50, 110)
        rx = int(cx + dist * np.cos(ang))
        ry = int(cy + dist * np.sin(ang))
        bw = int(rng.uniform(10, 26))
        bh = int(rng.uniform(10, 40))
        x0 = max(0, min(w - bw - 1, rx - bw // 2))
        y0 = max(0, min(h - bh - 1, ry - bh // 2))
        col = (int(rng.uniform(20, 70)), int(rng.uniform(20, 70)),
               int(rng.uniform(150, 230)))  # BGR 红
        cv2.rectangle(out, (x0, y0), (x0 + bw, y0 + bh), col, -1)
    return out


def main():
    if len(sys.argv) < 3:
        print("用法: python3 tools/laser_hardset.py <src_dir> <out_dir>")
        return 2
    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    lines = [json.loads(l) for l in open(src / "result.jsonl")]
    for name in ("far", "dark", "red"):
        (out / name).mkdir(parents=True, exist_ok=True)
    meta = {"far": [], "dark": [], "red": []}
    n = 0
    for dd in lines:
        L = dd.get("laser") or {}
        if L.get("status") != "ok":
            continue
        img = cv2.imread(str(src / dd["file"]))
        if img is None:
            continue
        cx, cy = find_spot(img, (L["x"], L["y"]))
        n += 1
        for name, fn in (("far", make_far), ("dark", make_dark), ("red", make_red)):
            o = fn(img, cx, cy)
            f = "%s/%04d.jpg" % (name, n)
            cv2.imwrite(str(out / f), o)
            meta[name].append({"file": f, "x": L["x"], "y": L["y"]})
            if n >= 20:
                break
        if n >= 20:
            break
    (out / "gt.json").write_text(json.dumps(meta, ensure_ascii=False),
                                 encoding="utf-8")
    print("生成 %d 组 -> %s（每组 far/dark/red 各一份）" % (n, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
