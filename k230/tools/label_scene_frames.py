#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""label_scene_frames.py —— 给静止场景连拍帧生成统一的 Pose 标注

现场调试时经常是「相机固定、靶纸固定」，连拍几十帧其实只有一个目标。
逐帧人工标注没有意义，用模型多帧预测取**中值**得到一份稳定标注，
再让所有帧共用它；同时输出一张 4x 放大叠加图供人工核对。

只做辅助，不替代人工：产物落盘前请务必看一眼 --sheet 图。

用法：
    python3 tools/label_scene_frames.py \
        --frames /tmp/scene_frames --weights models/pose_v3.pt \
        --imgsz 416 --conf 0.25 --out scene_labels.json --sheet scene_check.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

KPT = 5
COLORS = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 220, 220), (255, 0, 255)]
NAMES = ["TL", "TR", "BR", "BL", "BULL"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="连拍帧目录")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", required=True, help="输出 JSON（含归一化标注）")
    ap.add_argument("--sheet", default=None, help="人工核对图")
    ap.add_argument("--crop", default=None, help="x0,y0,x1,y1 裁剪放大区域")
    ap.add_argument("--zoom", type=float, default=4.0)
    args = ap.parse_args(argv)

    from ultralytics import YOLO
    model = YOLO(args.weights)
    frames = sorted(Path(args.frames).glob("*.jpg"))
    if not frames:
        print("[x] 没有图片：%s" % args.frames)
        return 2

    boxes, kpts, confs, used = [], [], [], []
    for p in frames:
        r = model.predict(str(p), imgsz=args.imgsz, conf=args.conf,
                          max_det=5, device="cpu", verbose=False)[0]
        if r.boxes is None or not len(r.boxes):
            continue
        i = int(np.argmax(r.boxes.conf.numpy()))
        boxes.append(r.boxes.xyxy.numpy()[i])
        kpts.append(r.keypoints.xy.numpy()[i])
        confs.append(float(r.boxes.conf.numpy()[i]))
        used.append(p)
    if not boxes:
        print("[x] 所有帧都没检出，无法生成标注")
        return 3

    box = np.median(np.stack(boxes), axis=0)
    kp = np.median(np.stack(kpts), axis=0)
    img = cv2.imread(str(frames[0]))
    h, w = img.shape[:2]
    print("[*] 参与中值 %d/%d 帧  conf min/med/max = %.3f/%.3f/%.3f"
          % (len(boxes), len(frames), min(confs), float(np.median(confs)), max(confs)))
    print("    box  [%.1f %.1f %.1f %.1f]  std=%s"
          % (*box, np.std(np.stack(boxes), axis=0).round(2)))
    print("    kpts %s" % np.round(kp, 1).tolist())
    print("    逐帧 kpt std max=%.2f px" % float(np.std(np.stack(kpts), axis=0).max()))

    rec = {
        "frames": str(Path(args.frames).resolve()),
        "weights": str(args.weights),
        "imgsz": args.imgsz,
        "n_used": len(boxes),
        "n_total": len(frames),
        "size": [w, h],
        "box_px": [round(float(v), 2) for v in box],
        "kpts_px": [[round(float(x), 2), round(float(y), 2)] for x, y in kp],
        "conf_median": round(float(np.median(confs)), 4),
        "conf_min": round(float(min(confs)), 4),
        "per_frame_std_px": np.std(np.stack(kpts), axis=0).round(3).tolist(),
        # YOLO-Pose 归一化格式：cls cx cy w h + 5*(x y v)
        "label": [0] + [round(float(v), 6) for v in (
            ((box[0] + box[2]) / 2) / w, ((box[1] + box[3]) / 2) / h,
            (box[2] - box[0]) / w, (box[3] - box[1]) / h)] +
            [round(float(v), 6) for p in kp for v in (p[0] / w, p[1] / h, 2)],
        "frames_labeled": [p.name for p in frames],
    }
    Path(args.out).write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[✓] 标注 -> %s" % args.out)

    if args.sheet:
        vis = img.copy()
        cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
        for i, (x, y) in enumerate(kp):
            cv2.circle(vis, (int(round(x)), int(round(y))), 2, COLORS[i], -1)
            cv2.putText(vis, NAMES[i], (int(x) + 3, int(y) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS[i], 1, cv2.LINE_AA)
        if args.crop:
            x0, y0, x1, y1 = [int(v) for v in args.crop.split(",")]
            vis = vis[y0:y1, x0:x1]
        vis = cv2.resize(vis, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(args.sheet, vis)
        print("[✓] 核对图 -> %s" % args.sheet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
