#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_pose.py —— Pose 模型统一评测（靶纸检出 / 靶心精度 / 误检率）

和 `pose_eval.py` 的区别：这里直接对比**多个权重**在同一批图上，
同时给出"干净图"和"合成激光光斑图"两组指标，用来判断一次训练
到底是变好还是变差；负样本单独统计误检率。

指标
----
* det_*    靶纸检出：IoU>=0.5 匹配后的 P / R / F1（含置信度阈值扫描）
* bull_*   靶心：匹配上的样本里第 5 关键点与原图标注的像素误差
           mean / median / P95 / <=15px 命中率
* corner_* 四角像素误差（mean / P95）
* neg_fp   负样本图（空标注）里出现任一 >=conf 检测的比例

用法
----
    python3 tools/bench_pose.py \
        --weights models/pose_v3.pt \
        --data recovered/dataset_pose_v3 \
        --neg-data recovered/k230_target_project/dataset_pose_farneg_B \
        --splits test val --confs 0.1 0.25 0.4 --spot-copies 2 --out bench.json
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
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

KPT = 5


# ---------------------------------------------------------------- 数据
def read_label(path: Path) -> Optional[Dict]:
    """读取 YOLO-Pose 标注；空文件 -> 负样本（返回 None）。"""
    if not path.exists():
        return None
    lines = [l.split() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        return None
    out = []
    for t in lines:
        if len(t) < 5 + KPT * 3:
            continue
        box = np.array([float(v) for v in t[1:5]], np.float32)
        pts = np.array([[float(t[5 + i * 3]), float(t[6 + i * 3]),
                         float(t[7 + i * 3])] for i in range(KPT)], np.float32)
        out.append({"box": box, "pts": pts})
    return {"boxes": out}


def to_pixels(box_n: np.ndarray, pts_n: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    cx, cy, bw, bh = box_n
    box = np.array([(cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h], np.float32)
    pts = pts_n.copy()
    pts[:, 0] *= w
    pts[:, 1] *= h
    return box, pts


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return 0.0 if ua <= 0 else inter / ua


# ---------------------------------------------------------------- 光斑合成（与 augment_spot 一致）
def add_laser(img: np.ndarray, x: float, y: float, rad: float, rng: random.Random) -> None:
    h, w = img.shape[:2]
    x, y = int(round(x)), int(round(y))
    if not (0 <= x < w and 0 <= y < h):
        return
    rings = [
        (rad * 2.4, (40, 25, 130)),
        (rad * 1.7, (55, 40, 190)),
        (rad * 1.15, (75, 55, 235)),
        (rad * 0.75, (110, 85, 250)),
    ]
    for r, col in rings:
        cv2.circle(img, (x, y), int(max(1, r)), col, -1, cv2.LINE_AA)
    core = max(1, int(round(rad * rng.uniform(0.30, 0.60))))
    cv2.circle(img, (x, y), core, (250, 252, 255), -1, cv2.LINE_AA)
    k = int(max(3, rad * 1.2))
    if k % 2 == 0:
        k += 1
    x0, y0 = max(0, x - k), max(0, y - k)
    x1, y1 = min(w, x + k), min(h, y + k)
    if x1 - x0 > 2 and y1 - y0 > 2:
        img[y0:y1, x0:x1] = cv2.GaussianBlur(img[y0:y1, x0:x1], (3, 3), 0)


def make_spot_variant(img: np.ndarray, box: np.ndarray, pts: np.ndarray,
                      rng: random.Random) -> np.ndarray:
    """在靶心附近叠加一个激光光斑（保持标注不变）。"""
    out = img.copy()
    bw, bh = box[2] - box[0], box[3] - box[1]
    bx, by = pts[4, 0], pts[4, 1]
    lx = bx + rng.uniform(-bw * 0.07, bw * 0.07)
    ly = by + rng.uniform(-bh * 0.07, bh * 0.07)
    ref = max(24.0, min(bw, bh))
    rad = max(1.5, min(14.0, rng.uniform(0.020, 0.075) * ref))
    add_laser(out, lx, ly, rad, rng)
    return out


def make_far_variant(img: np.ndarray, box: np.ndarray, pts: np.ndarray,
                     rng: random.Random,
                     spot: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """整图缩小到 0.30~0.65 倍后贴回 640x480，模拟远距离小靶。"""
    h, w = img.shape[:2]
    s = rng.uniform(0.30, 0.65)
    nw, nh = max(8, int(round(w * s))), max(8, int(round(h * s)))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    ox = rng.randint(0, max(0, w - nw))
    oy = rng.randint(0, max(0, h - nh))
    out = cv2.copyMakeBorder(small, oy, h - nh - oy, ox, w - nw - ox,
                             cv2.BORDER_REPLICATE)
    b = box.copy()
    b[[0, 2]] = b[[0, 2]] * s + ox
    b[[1, 3]] = b[[1, 3]] * s + oy
    p = pts.copy()
    p[:, 0] = p[:, 0] * s + ox
    p[:, 1] = p[:, 1] * s + oy
    if spot:
        bw, bh = b[2] - b[0], b[3] - b[1]
        lx = p[4, 0] + rng.uniform(-bw * 0.07, bw * 0.07)
        ly = p[4, 1] + rng.uniform(-bh * 0.07, bh * 0.07)
        add_laser(out, lx, ly, max(1.2, min(10.0, 0.05 * max(bw, bh))), rng)
    return out, b, p


# ---------------------------------------------------------------- 评估
def evaluate(weights: Path, items: List[Dict], confs: List[float], imgsz: int,
             device: str, tag: str) -> Dict:
    from ultralytics import YOLO
    model = YOLO(str(weights))
    stats = {c: {"tp": 0, "fp": 0, "fn": 0, "top1_ok": 0, "top1_wrong": 0,
                 "top1_miss": 0} for c in confs}
    bull_err: List[float] = []
    corner_err: List[List[float]] = [[] for _ in range(4)]
    bull_vis = 0
    n_det_matched = 0
    per_conf_bull: Dict[float, List[float]] = {c: [] for c in confs}

    for i, it in enumerate(items):
        img = it["img"]
        res = model.predict(img, imgsz=imgsz, conf=min(confs), max_det=20,
                            device=device, verbose=False)[0]
        pboxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4), np.float32)
        pconf = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros((0,), np.float32)
        pkpts = res.keypoints.xy.cpu().numpy() if (res.keypoints is not None and res.keypoints.xy.numel()) else np.zeros((0, KPT, 2), np.float32)
        if len(pboxes) and len(pkpts) != len(pboxes):
            pkpts = np.zeros((len(pboxes), KPT, 2), np.float32)

        gts = it["gts"]
        matched = set()
        pairs = []  # (det_idx, gt_idx)
        if len(gts) and len(pboxes):
            ious = np.array([[iou(pb, g["box"]) for g in gts] for pb in pboxes])
            order = np.dstack(np.unravel_index(np.argsort(-ious, axis=None), ious.shape))[0]
            for di, gi in order:
                if ious[di, gi] < 0.5:
                    break
                if di in matched or gi in [p[1] for p in pairs]:
                    continue
                pairs.append((int(di), int(gi)))
        matched_di = {p[0] for p in pairs}

        for c in confs:
            tp = sum(1 for di, _ in pairs if pconf[di] >= c)
            fp = int(((pconf >= c) & ~np.isin(np.arange(len(pconf)), list(matched_di))).sum())
            fn = max(0, len(gts) - tp)
            stats[c]["tp"] += tp
            stats[c]["fp"] += fp
            stats[c]["fn"] += fn
            # 板端真实行为：按分数取 top-1 再判对错
            if len(pconf):
                top = int(np.argmax(pconf))
                if pconf[top] < c:
                    stats[c]["top1_miss"] += 1
                elif top in matched_di:
                    stats[c]["top1_ok"] += 1
                else:
                    stats[c]["top1_wrong"] += 1
            else:
                stats[c]["top1_miss"] += 1
            # 该置信度下的靶心误差
            for di, gi in pairs:
                if pconf[di] >= c and gts[gi]["pts"][4, 2] > 0:
                    d = float(np.linalg.norm(pkpts[di, 4] - gts[gi]["pts"][4, :2]))
                    per_conf_bull[c].append(d)

        # 全量误差（只统计 conf>=0.25 的匹配）
        for di, gi in pairs:
            if pconf[di] < 0.25:
                continue
            gt = gts[gi]
            n_det_matched += 1
            if gt["pts"][4, 2] > 0:
                bull_vis += 1
                bull_err.append(float(np.linalg.norm(pkpts[di, 4] - gt["pts"][4, :2])))
            for k in range(4):
                if gt["pts"][k, 2] > 0:
                    corner_err[k].append(float(np.linalg.norm(pkpts[di, k] - gt["pts"][k, :2])))

    def agg(v: List[float]) -> Dict[str, float]:
        if not v:
            return {"n": 0}
        a = np.array(v)
        return {"n": int(len(a)), "mean": round(float(a.mean()), 2),
                "median": round(float(np.median(a)), 2),
                "p95": round(float(np.percentile(a, 95)), 2),
                "max": round(float(a.max()), 2),
                "hit15": round(float((a <= 15).mean() * 100), 1)}

    out = {"weights": str(weights), "tag": tag, "n_images": len(items)}
    out["det"] = {}
    for c in confs:
        s = stats[c]
        p = s["tp"] / max(1, s["tp"] + s["fp"])
        r = s["tp"] / max(1, s["tp"] + s["fn"])
        out["det"]["%.2f" % c] = {
            "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
            "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / max(1e-9, p + r), 4),
            "top1": {"ok": s["top1_ok"], "wrong": s["top1_wrong"],
                     "miss": s["top1_miss"],
                     "acc_pct": round(100.0 * s["top1_ok"] / max(1, len(items)), 1)},
            "bull_err": agg(per_conf_bull[c]),
        }
    out["matched"] = n_det_matched
    out["bull"] = agg(bull_err)
    out["corner"] = {k: agg(corner_err[k]) for k in range(4)}
    out["corner_all"] = agg([v for l in corner_err for v in l])
    return out


def eval_negatives(weights: Path, paths: List[Path], confs: List[float],
                   imgsz: int, device: str) -> Dict:
    from ultralytics import YOLO
    model = YOLO(str(weights))
    hits = {c: 0 for c in confs}
    detail = []
    for p in paths:
        res = model.predict(str(p), imgsz=imgsz, conf=min(confs), max_det=20,
                            device=device, verbose=False)[0]
        conf = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros((0,), np.float32)
        for c in confs:
            if (conf >= c).any():
                hits[c] += 1
        if len(conf):
            detail.append({"img": p.name, "max_conf": round(float(conf.max()), 3)})
    return {
        "n_images": len(paths),
        "fp_rate": {"%.2f" % c: {"hits": hits[c],
                                 "rate_pct": round(100.0 * hits[c] / max(1, len(paths)), 1)}
                    for c in confs},
        "detail": sorted(detail, key=lambda d: -d["max_conf"])[:20],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pose 模型统一评测")
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--data", required=True, help="含 images/<split> 的目录")
    ap.add_argument("--neg-data", default=None,
                    help="负样本目录（labels 为空的图会自动收集）")
    ap.add_argument("--splits", nargs="+", default=["test", "val"])
    ap.add_argument("--confs", nargs="+", type=float, default=[0.1, 0.25, 0.4])
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--spot-copies", type=int, default=2,
                    help="每张正样本再生成几张合成光斑图（0=不做）")
    ap.add_argument("--far-copies", type=int, default=0,
                    help="每张正样本再生成几张缩小图（模拟远距离小靶）")
    ap.add_argument("--far-spot", action="store_true",
                    help="缩小图里再叠一个靶心光斑（最难的组合）")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    root = Path(args.data)
    rng = random.Random(args.seed)
    items = []
    neg_items = []
    for split in args.splits:
        img_dir = root / "images" / split
        if not img_dir.is_dir():
            continue
        for p in sorted(img_dir.glob("*.jpg")):
            lab = read_label(root / "labels" / split / (p.stem + ".txt"))
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            if lab is None or not lab["boxes"]:
                neg_items.append(p)
                continue
            gts = []
            for g in lab["boxes"]:
                box, pts = to_pixels(g["box"], g["pts"], w, h)
                gts.append({"box": box, "pts": pts})
            items.append({"img": img, "gts": gts, "name": p.name, "variant": "clean"})
            for k in range(args.spot_copies):
                g = lab["boxes"][0]
                box, pts = to_pixels(g["box"], g["pts"], w, h)
                items.append({"img": make_spot_variant(img, box, pts, rng),
                              "gts": gts, "name": "%s_spot%d" % (p.name, k),
                              "variant": "spot"})
            for k in range(args.far_copies):
                g = lab["boxes"][0]
                box, pts = to_pixels(g["box"], g["pts"], w, h)
                far_img, fb, fp = make_far_variant(img, box, pts, rng, spot=args.far_spot)
                fg = [{"box": fb, "pts": fp}]
                items.append({"img": far_img, "gts": fg,
                              "name": "%s_far%d" % (p.name, k),
                              "variant": "far"})

    if args.neg_data:
        neg_root = Path(args.neg_data)
        for split in ("train", "val", "test"):
            img_dir = neg_root / "images" / split
            ldir = neg_root / "labels" / split
            if not img_dir.is_dir():
                continue
            for p in sorted(img_dir.glob("*.jpg")):
                lab = ldir / (p.stem + ".txt")
                if (not lab.exists()) or (not lab.read_text(encoding="utf-8").strip()):
                    if p not in neg_items:
                        neg_items.append(p)
    print("[*] 正样本 %d 张（含合成光斑），负样本 %d 张" % (len(items), len(neg_items)))

    results = []
    for wt in args.weights:
        wt = Path(wt)
        print("\n=== %s ===" % wt)
        by_var = {}
        for v in ("clean", "spot", "far"):
            sub = [it for it in items if it.get("variant") == v]
            if sub:
                by_var[v] = evaluate(wt, sub, args.confs, args.imgsz, args.device, v)
        r_neg = eval_negatives(wt, neg_items, args.confs, args.imgsz, args.device) if neg_items else None
        for v, r in by_var.items():
            print("  [%s] imgs=%d" % (r["tag"], r["n_images"]))
            for c, d in r["det"].items():
                t1 = d["top1"]
                print("    conf=%s  P=%.3f R=%.3f F1=%.3f  top1=%.1f%% (ok/wrong/miss=%d/%d/%d)  bull n=%d med=%.1fpx p95=%.1fpx <=15px=%.0f%%"
                      % (c, d["precision"], d["recall"], d["f1"], t1["acc_pct"],
                         t1["ok"], t1["wrong"], t1["miss"],
                         d["bull_err"].get("n", 0), d["bull_err"].get("median", -1),
                         d["bull_err"].get("p95", -1), d["bull_err"].get("hit15", -1)))
            print("    bull all: %s" % r["bull"])
            print("    corner: mean=%.2f p95=%.2f" % (r["corner_all"].get("mean", -1),
                                                       r["corner_all"].get("p95", -1)))
        if r_neg:
            print("  [neg] fp@0.25=%.1f%%  fp@0.4=%.1f%%" % (
                r_neg["fp_rate"]["0.25"]["rate_pct"], r_neg["fp_rate"]["0.40"]["rate_pct"]))
        results.append({"weights": str(wt), "variants": by_var, "neg": r_neg})

    out = {"data": str(args.data), "splits": args.splits, "imgsz": args.imgsz,
           "n_items": len(items), "n_neg": len(neg_items), "results": results}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\n[✓] -> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
