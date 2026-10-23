# -*- coding: utf-8 -*-
"""
pose_eval.py —— A/B Pose 模型在同一批 K230 实拍测试图上的评估
============================================================

A：4 角点 -> 单应 -> 标准版面固定靶心 -> 反投影
B：4 角点 + 第 5 关键点（靶心）

指标：
    * 靶心中心像素误差 mean / P95 / max
    * 角点误差 mean / P95（分 TL/TR/BR/BL）
    * 角点出框率
    * 靶心关键点可见率 / 回归成功率
    * 失败态：误报 fail / 漏报 fail
    * pose_ms（PyTorch/ONNX 端，仅参考；板端见 live 统计）

用法：
    python3 tools/pose_eval.py --weights runs/poseB/weights/best.pt \
        --data ../k230_target_project/dataset_pose_B --split test \
        --kpt 5 --out eval_B.json

    python3 tools/pose_eval.py --weights runs/poseA/weights/best.pt \
        --data ../k230_target_project/dataset_pose_A --split test \
        --kpt 4 --out eval_A.json
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
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

CANON_W, CANON_H = 120, 160
DST_Q = np.float32([[0, 0], [CANON_W, 0], [CANON_W, CANON_H], [0, CANON_H]])


def load_label(path: Path, kpt: int) -> Optional[Dict]:
    if not path.exists():
        return None
    lines = [l.split() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        return None
    t = lines[0]
    box = [float(v) for v in t[1:5]]
    pts = []
    for i in range(kpt):
        o = 5 + i * 3
        pts.append([float(t[o]), float(t[o + 1]), int(float(t[o + 2]))])
    return {"box": box, "pts": pts}


def order(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s = pts.sum(1); d = np.diff(pts, axis=1).ravel()
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)],
                       pts[np.argmax(s)], pts[np.argmax(d)]])


def convex_ok(pts: np.ndarray) -> bool:
    pts = order(pts)
    signs = []
    for i in range(4):
        a, b, c = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(1 if cr > 0 else -1)
    return len(set(signs)) == 1


def side_ratio_ok(pts: np.ndarray, min_side: float = 8.0, max_ratio: float = 2.5) -> bool:
    p = order(pts)
    sides = [float(np.linalg.norm(p[(i + 1) % 4] - p[i])) for i in range(4)]
    if min(sides) < min_side:
        return False
    r1 = max(sides[0], sides[2]) / max(1e-6, min(sides[0], sides[2]))
    r2 = max(sides[1], sides[3]) / max(1e-6, min(sides[1], sides[3]))
    return r1 <= max_ratio and r2 <= max_ratio


def map_bull_A(corners: np.ndarray, canon_bull: Tuple[float, float]) -> np.ndarray:
    Hm = cv2.getPerspectiveTransform(order(corners), DST_Q)
    pt = cv2.perspectiveTransform(np.float32([[canon_bull]]).reshape(1, 1, 2),
                                  np.linalg.inv(Hm))[0, 0]
    return pt


def confidence_stats(errs: List[float]) -> Dict:
    if not errs:
        return {"n": 0}
    a = np.array(errs, np.float64)
    return {"n": len(a), "mean": round(float(a.mean()), 3),
            "p95": round(float(np.percentile(a, 95)), 3),
            "max": round(float(a.max()), 3)}


def eval_model(weights: Path, data: Path, split: str, kpt: int,
               conf: float, v_min: float, imgsz: int,
               canon_bull: Optional[Tuple[float, float]],
               gt_data: Optional[Path] = None) -> Dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    img_dir = data / "images" / split
    gt_root = gt_data or data
    lbl_dir = gt_root / "labels" / split
    files = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not files:
        raise SystemExit("测试集为空：%s" % img_dir)

    # 预热，避免首帧加载时间污染 pose_ms
    try:
        model.predict(str(files[0]), imgsz=imgsz, conf=conf, verbose=False)
    except Exception:
        pass

    # A 版：从训练集标注统计标准版面靶心位置
    if kpt == 4 and canon_bull is None:
        vals = []
        for lp in sorted((gt_root / "labels" / "train").glob("*.txt")):
            rec = load_label(lp, 5)
            if not rec or rec["pts"][4][2] < 1:
                continue
            w, h = 640.0, 480.0
            gt = np.float32([[p[0] * w, p[1] * h] for p in rec["pts"][:4]])
            Hm = cv2.getPerspectiveTransform(order(gt), DST_Q)
            b = np.float32([[[rec["pts"][4][0] * w, rec["pts"][4][1] * h]]])
            c = cv2.perspectiveTransform(b, Hm)[0, 0]
            vals.append(c)
        canon_bull = tuple(np.mean(vals, axis=0)) if vals else (58.0, 118.8)
        print("[*] A 版标准版面靶心：%.1f,%.1f（来自 %d 张训练标注）"
              % (canon_bull[0], canon_bull[1], len(vals)))

    bull_err, corner_err = [], []
    corner_err_pts = [[], [], [], []]
    out_of_frame = 0; corner_total = 0
    bull_visible = 0; bull_success = 0
    fails = {"none": 0}
    false_fail = 0; missed_fail = 0; valid_gt = 0
    times = []
    per_image = []

    for p in files:
        stem = p.stem
        rec = load_label(lbl_dir / (stem + ".txt"), 5)
        if rec is None:
            continue
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        ih, iw = frame.shape[:2]
        gt_bull = None
        gt_valid = all(q[2] >= 1 for q in rec["pts"][:4])
        if len(rec["pts"]) >= 5 and rec["pts"][4][2] >= 1:
            gt_bull = np.array([rec["pts"][4][0] * iw, rec["pts"][4][1] * ih])
        else:
            gt_valid = False
        if gt_valid:
            valid_gt += 1
        stage = stem.split("_frame")[0] if "_frame" in stem else "unknown"

        t0 = time.time()
        r = model.predict(str(p), imgsz=imgsz, conf=conf, verbose=False)[0]
        times.append((time.time() - t0) * 1000.0)

        status = "ok"
        if r.boxes is None or len(r.boxes) == 0:
            status = "no_target"
        else:
            i = int(np.argmax(r.boxes.conf.cpu().numpy()))
            bconf = float(r.boxes.conf.cpu().numpy()[i])
            if bconf < conf:
                status = "low_box_conf"
            else:
                kp_xy = r.keypoints.xy.cpu().numpy()[i]
                kp_cf = (r.keypoints.conf.cpu().numpy()[i]
                         if r.keypoints.conf is not None else np.ones(len(kp_xy)))
                if len(kp_xy) < 4:
                    status = "no_corners"
                elif np.any(kp_cf[:4] < v_min):
                    status = "corner_low_conf"
                elif not convex_ok(kp_xy[:4]):
                    status = "not_convex"
                elif not side_ratio_ok(kp_xy[:4]):
                    status = "bad_side_ratio"
                else:
                    oof = int(np.sum((kp_xy[:4, 0] < 0) | (kp_xy[:4, 0] >= iw) |
                                     (kp_xy[:4, 1] < 0) | (kp_xy[:4, 1] >= ih)))
                    corner_total += 4; out_of_frame += oof
                    if oof > 1:
                        status = "corners_out"
                    else:
                        if kpt == 5:
                            if len(kp_xy) < 5 or kp_cf[4] < v_min:
                                status = "bull_low_conf"
                            else:
                                bull_visible += 1
                                bull = kp_xy[4]
                                # 凸包外 -> fail（用四角包围盒近似）
                                x1, y1 = kp_xy[:4].min(0); x2, y2 = kp_xy[:4].max(0)
                                if not (x1 - 4 <= bull[0] <= x2 + 4 and y1 - 4 <= bull[1] <= y2 + 4):
                                    status = "bull_outside"
                                else:
                                    pred_bull = bull
                        else:
                            pred_bull = map_bull_A(kp_xy[:4], canon_bull)

                        if status == "ok":
                            for k in range(4):
                                e = float(np.linalg.norm(kp_xy[k] - np.array(
                                    [rec["pts"][k][0] * iw, rec["pts"][k][1] * ih])))
                                corner_err_pts[k].append(e); corner_err.append(e)
                            if gt_bull is not None:
                                e = float(np.linalg.norm(pred_bull - gt_bull))
                                bull_err.append(e); bull_success += 1
                            if not gt_valid:
                                missed_fail += 1
                        elif gt_valid:
                            false_fail += 1

        fails[status] = fails.get(status, 0) + 1
        per_image.append({
            "key": stem, "stage": stage, "status": status,
            "bull_err": (round(float(np.linalg.norm(pred_bull - gt_bull)), 3)
                         if status == "ok" and gt_bull is not None else None),
            "bull_conf": (round(float(kp_cf[4]), 3)
                          if status == "ok" and kpt == 5 else None),
        })

    # 按场景分组
    per_stage = {}
    for rec in per_image:
        g = per_stage.setdefault(rec["stage"], {"n": 0, "ok": 0, "bull_err": [], "fails": {}})
        g["n"] += 1
        if rec["status"] == "ok":
            g["ok"] += 1
            if rec["bull_err"] is not None:
                g["bull_err"].append(rec["bull_err"])
        else:
            g["fails"][rec["status"]] = g["fails"].get(rec["status"], 0) + 1
    for g in per_stage.values():
        e = g.pop("bull_err")
        g["bull_err"] = confidence_stats(e)
        g["ok_rate"] = round(g["ok"] / max(1, g["n"]), 3)

    return {
        "weights": str(weights), "data": str(data), "split": split, "kpt": kpt,
        "images": len(files),
        "bull_err": confidence_stats(bull_err),
        "corner_err": confidence_stats(corner_err),
        "corner_err_per_point": {n: confidence_stats(corner_err_pts[i])
                                 for i, n in enumerate(["TL", "TR", "BR", "BL"])},
        "out_of_frame_rate": round(out_of_frame / max(1, corner_total), 4),
        "bull_visible_rate": round(bull_visible / max(1, valid_gt), 4),
        "bull_success_rate": round(bull_success / max(1, valid_gt), 4),
        "false_fail": false_fail, "missed_fail": missed_fail, "valid_gt": valid_gt,
        "fails": fails,
        "pose_ms_mean": round(float(np.mean(times)), 2) if times else None,
        "per_stage": per_stage,
        "per_image": per_image,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A/B Pose 评估")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--kpt", type=int, choices=[4, 5], required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--v-min", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--canon-bull", default=None, help="A 版标准版面靶心 x,y；默认从训练标注统计")
    ap.add_argument("--gt-data", default=None, help="A 版评估时用 B 数据集提供 5 点 GT")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cb = tuple(float(v) for v in args.canon_bull.split(",")) if args.canon_bull else None
    res = eval_model(Path(args.weights), Path(args.data), args.split, args.kpt,
                     args.conf, args.v_min, args.imgsz, cb,
                     Path(args.gt_data) if args.gt_data else None)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
