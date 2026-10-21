#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测激光光斑识别效果：跑板端同源判据（numpy 复刻 board/laser_spot.py），
对采集到的原始帧给出检出率、坐标误差、误报率、性���耗时。

用法：
    python3 tools/eval_laser.py tools/laser_samples/wall_on_xxx
    python3 tools/eval_laser.py tools/laser_samples/* --report report.json

有标注（annotation.json，人工标记光斑真值 x,y）时计算误差；
无标注时只统计检出率和稳定性（连续帧坐标抖动）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("需要 opencv-python")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "board"))
import laser_spot as LS  # noqa: E402  板端同源参数

CFG = dict(LS.CFG)
# LAB / 判据与板端保持一致的近似：LAB_WHITE L>=82 -> luma>=209
WHITE_LUMA_MIN = 209
RED_MIN_R = 160  # 兼容旧配置名


def scan(img, white=True):
    """numpy 复刻板端 _scan()：返回候选列表（按 score 降序）。"""
    b, g, r = cv2.split(img.astype(np.int16))
    luma = ((r.astype(np.int32) * 77 + g.astype(np.int32) * 150 +
             b.astype(np.int32) * 29) >> 8).astype(np.int16)
    warm = r - g
    mn = np.minimum(np.minimum(r, g), b)
    m = (luma >= WHITE_LUMA_MIN).astype(np.uint8) if white else \
        (r > g + CFG["red_min_warm"]).astype(np.uint8)
    n, lab, st, ce = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < CFG["min_area"] or a > CFG["max_area"]:
            continue
        if w > CFG["max_side"] or h > CFG["max_side"]:
            continue
        if max(w, h) > min(w, h) * CFG["max_aspect"]:
            continue
        bw_ = warm[y:y + h, x:x + w]
        bmn = mn[y:y + h, x:x + w]
        if white:
            if bmn.max() < CFG["min_sat"] or bw_.max() < CFG["min_warm"]:
                continue
        else:
            if bw_.max() < CFG["red_min_warm"]:
                continue
        peak = int(luma[y:y + h, x:x + w].max())
        ring = luma[max(0, y - 12):y + h + 12, max(0, x - 12):x + w + 12]
        contrast = peak - float(np.median(ring))
        if contrast < CFG["min_contrast"]:
            continue
        out.append({"x": float(ce[i][0]), "y": float(ce[i][1]),
                    "contrast": contrast, "warm": int(bw_.max()),
                    "sat": int(bmn.max()), "area": int(a)})
    out.sort(key=lambda c: -(c["contrast"] + 0.55 * c["warm"] + 0.12 * c["sat"]))
    return out


def scan_combined(img):
    """numpy 复刻板端 laser_spot.LaserSpot._scan：红区先验 + 白点种子 + 红点兜底。"""
    b, g, r = cv2.split(img.astype(np.int16))
    luma = ((r.astype(np.int32)*77 + g.astype(np.int32)*150 + b.astype(np.int32)*29) >> 8).astype(np.int16)
    warm = r - g
    mn = np.minimum(np.minimum(r, g), b)
    H, W = img.shape[:2]
    # --- 0) 红区先验（LAB_RED 近似）
    # 实测板端 LAB_RED 会漏掉激光较亮的红晕，但能稳定命中的是"较红的像素"；
    # 这里用 warm>=60 作为红区种子（真激光 warm 160~224；
    # 红壳/红缆 warm 80~130 但仍会因无白核被拒）。
    mr = ((warm >= 35) & (luma >= 40)).astype(np.uint8)
    n2, _, st2, _ = cv2.connectedComponentsWithStats(mr, 8)
    zones = []
    small_red = []
    for i in range(1, n2):
        x, y, w, h, a = st2[i]
        if a < 2 or w <= 0 or h <= 0 or w > CFG['max_side'] or h > CFG['max_side']:
            continue
        if a > CFG['max_zone_area']:
            continue
        zw = warm[y:y+h, x:x+w]
        if zw.max() < CFG['min_warm']:
            continue
        # 红区必须含三通道全饱和的白核
        if mn[y:y+h, x:x+w].max() < CFG['min_sat']:
            continue
        if a <= 4 and w <= 4 and h <= 4:
            small_red.append((x, y, w, h))
            continue
        fill = a / float(max(1, w*h))
        if fill < CFG['min_fill']:
            continue
        if max(w, h) > min(w, h) * CFG['max_zone_aspect']:
            continue
        zones.append((x, y, w, h))
        if len(zones) >= CFG['max_zones']:
            break
    out = []

    def _redc(x0, y0, bw, bh):
        xa, ya = max(0, x0), max(0, y0)
        xb, yb = min(W, x0+bw), min(H, y0+bh)
        return warm[ya:yb, xa:xb]

    base_luma = int(np.median(luma[8::16, 8::16]))
    # --- 1) 白点种子必须落在红区内
    mw = (luma >= WHITE_LUMA_MIN).astype(np.uint8)
    n, _, st, ce = cv2.connectedComponentsWithStats(mw, 8)
    whites = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w <= 0 or h <= 0 or w > CFG['max_side'] or h > CFG['max_side']:
            continue
        whites.append((x, y, w, h))
        inside = any(not (x > zx+zw or zx > x+w or y > zy+zh or zy > y+h) for (zx, zy, zw, zh) in zones)
        if not inside:
            continue
        peak = int(luma[y:y+h, x:x+w].max())
        max_warm = int(warm[y:y+h, x:x+w].max())
        max_min = int(mn[y:y+h, x:x+w].max())
        if max_min < CFG['min_sat'] or max_warm >= CFG['min_warm']:
            continue
        if peak < CFG['min_contrast'] + 30 or peak < base_luma + CFG['abs_contrast']:
            continue
        out.append(_mk(img, luma, ce[i][0], ce[i][1], w, h, peak, max_warm, max_min, (x, y, w, h), 'white', 1))
    # --- 2) 红点兜底
    for i in range(1, n2):
        x, y, w, h, a = st2[i]
        if a < CFG['min_area'] or a > CFG['max_area']:
            continue
        if w <= 0 or h <= 0 or w > CFG['max_side'] or h > CFG['max_side']:
            continue
        big, sml = max(w, h), max(1, min(w, h))
        if big > sml * CFG['max_aspect']:
            continue
        if any(not (x > wx+ww or wx > x+w or y > wy+wh or wy > y+h) for (wx, wy, ww, wh) in whites):
            continue
        if not any(not (x > zx+zw or zx > x+w or y > zy+zh or zy > y+h) for (zx, zy, zw, zh) in zones):
            continue
        peak = int(luma[y:y+h, x:x+w].max())
        max_warm = int(warm[y:y+h, x:x+w].max())
        max_min = int(mn[y:y+h, x:x+w].max())
        if max_warm < CFG['red_min_warm'] or peak < CFG['min_contrast'] + 30:
            continue
        if peak < base_luma + CFG['abs_contrast']:
            continue
        if max_min < CFG['red_min_sat']:
            continue

        out.append(_mk(img, luma, x+w/2, y+h/2, w, h, peak, max_warm, max_min, (x, y, w, h), 'red', 0))
    # --- 3) 无白核光斑：红区/小红光点本身够亮，且没有被白点覆盖
    for (zx, zy, zw, zh) in zones + small_red:
        if any(not (zx > wx+ww or wx > zx+zw or zy > wy+wh or wy > zy+zh) for (wx, wy, ww, wh) in whites):
            continue
        a_ = int((mr[zy:zy+zh, zx:zx+zw] > 0).sum())
        if a_ < CFG['min_area'] or a_ > CFG['max_core_area']:
            continue
        big, sml = max(zw, zh), max(1, min(zw, zh))
        if big > sml * CFG['max_aspect']:
            continue
        peak = int(luma[zy:zy+zh, zx:zx+zw].max())
        max_warm = int(warm[zy:zy+zh, zx:zx+zw].max())
        max_min = int(mn[zy:zy+zh, zx:zx+zw].max())
        if max_warm < CFG['red_min_warm'] or peak < base_luma + CFG['abs_contrast']:
            continue
        out.append(_mk(img, luma, zx + zw/2, zy + zh/2, zw, zh, peak, max_warm, max_min, (zx, zy, zw, zh), 'redcore', 1))
    out.sort(key=lambda c: -(c['contrast'] + 0.7 * c['warm'] + 0.12 * c['sat'] + (1.5 * c['halo'] if c['mode'] in ('white', 'redcore') else 0)))
    # 形态校验只作用于最终候选（与板端 locate 一致）
    keep = []
    for c in out:
        bx0, by0, bw, bh = c['bbox']
        body, frac = _body(mn, luma, bx0, by0, bw, bh)
        need = CFG['red_sat_frac'] if c['mode'] == 'red' else CFG['min_sat_frac']
        if body > CFG['max_body'] or frac < need:
            continue
        c['body'] = body
        c['satfrac'] = frac
        keep.append(c)
    return keep


def _body(mn, luma, x, y, w, h):
    """(亮度>=200 像素数, 三通道饱和占比 0..100)，对应板端 _body_stats。"""
    lm = luma[y:y+h, x:x+w]; mm = mn[y:y+h, x:x+w]
    body = int((lm >= 200).sum())
    frac = int((mm >= 190).sum() * 100 / max(1, w*h))
    return body, frac


def _mk(img, luma, cx, cy, w, h, peak, max_warm, max_min, bbox, mode, halo=0, mn=None):
    H, W = img.shape[:2]
    R = max(w, h) + 12
    x0, y0 = max(0, int(cx)-R), max(0, int(cy)-R)
    x1, y1 = min(W, int(cx)+R), min(H, int(cy)+R)
    bx0, by0, bw, bh = bbox
    ring = luma[y0:y1:2, x0:x1:2]
    # 排除 bbox 内
    mask = np.ones(ring.shape, bool)
    ry = np.arange(y0, y1, 2); rx = np.arange(x0, x1, 2)
    inside_y = (ry >= by0) & (ry < by0+bh); inside_x = (rx >= bx0) & (rx < bx0+bw)
    mask[np.ix_(inside_y, inside_x)] = False
    vals = ring[mask]
    med = float(np.median(vals)) if vals.size else 0.0
    contrast = peak - med
    return {'x': float(cx), 'y': float(cy), 'contrast': contrast,
            'warm': max_warm, 'sat': max_min, 'area': bw*bh, 'mode': mode,
            'halo': halo, 'bbox': bbox}


def detect(img):
    got = scan_combined(img)
    if got:
        return got[0], got[0]['mode']
    return None, None


def eval_dir(d: Path) -> dict:
    ann_path = d / "annotation.json"
    ann = {}
    if ann_path.exists():
        raw = json.loads(ann_path.read_text(encoding="utf-8"))
        for k, v in raw.items():
            ann[k] = v  # {"file": "raw/frame_0001.jpg", "x": 335, "y": 280, "has": true}
    files = sorted((d / "raw").glob("frame_*.jpg"))
    rows = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        c, mode = detect(img)
        rows.append({"file": str(f.relative_to(d)),
                     "w": w, "h": h,
                     "found": c is not None,
                     "x": None if c is None else round(c["x"] * 640.0 / w, 2),
                     "y": None if c is None else round(c["y"] * 480.0 / h, 2),
                     "contrast": None if c is None else round(c["contrast"], 1),
                     "mode": mode})
    n = len(rows)
    hit = sum(1 for r in rows if r["found"])
    res = {"dir": str(d), "frames": n, "hits": hit,
           "hit_rate": round(100.0 * hit / max(1, n), 2)}
    xs = [r["x"] for r in rows if r["found"]]
    ys = [r["y"] for r in rows if r["found"]]
    if xs:
        # 主簇（模态）：激光场景是"同一位置",静态物体误检会被分到别的簇
        pts = np.array(list(zip(xs, ys)))
        best_cluster = [0] * len(pts)
        for i in range(len(pts)):
            d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
            cl = list(np.where(d <= 24)[0])
            if len(cl) > len(best_cluster):
                best_cluster = cl
        cx_ = pts[best_cluster, 0]
        cy_ = pts[best_cluster, 1]
        res.update({"x_mean": round(float(cx_.mean()), 2),
                    "y_mean": round(float(cy_.mean()), 2),
                    "x_std": round(float(cx_.std()), 2),
                    "y_std": round(float(cy_.std()), 2),
                    "cluster_n": len(best_cluster),
                    "cluster_ratio": round(100.0 * len(best_cluster) / len(pts), 1),
                    "off_cluster": len(pts) - len(best_cluster)})
    # 有标注：误差
    if ann:
        errs = []
        fp = 0
        fn = 0
        for k in sorted(ann):
            a = ann[k]
            m = [r for r in rows if r["file"] == a["file"]]
            if not m:
                continue
            r = m[0]
            if a.get("has", True):
                if r["found"]:
                    errs.append(float(np.hypot(r["x"] - a["x"], r["y"] - a["y"])))
                else:
                    fn += 1
            elif r["found"]:
                fp += 1
        if errs:
            errs = np.array(errs)
            res.update({"labeled": len(errs) + fn + fp,
                        "err_mean": round(float(errs.mean()), 2),
                        "err_p95": round(float(np.percentile(errs, 95)), 2),
                        "err_max": round(float(errs.max()), 2),
                        "fn": fn, "fp": fp})
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    all_res = [eval_dir(Path(d)) for d in args.dirs]
    for r in all_res:
        print(json.dumps(r, ensure_ascii=False))
    if args.report:
        Path(args.report).write_text(
            json.dumps(all_res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
