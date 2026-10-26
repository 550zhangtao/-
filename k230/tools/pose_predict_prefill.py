# -*- coding: utf-8 -*-
"""
pose_predict_prefill.py —— 用已训练 Pose 模型给未标注图片自动预标
=================================================================

输出一个新的 pose_manifest_auto.json：
    images[key] = {url, box, points:[[x,y,v],...], conf}

标注工具加载后会直接用这些点作为起点，用户只修正错误点。

用法：
    python3 tools/pose_predict_prefill.py \
        --weights ../k230_target_project/runs_target/poseB_pilot/weights/best.pt \
        --manifest tools/pose_manifest.json \
        --captures ../k230_target_project/captures \
        --out tools/pose_manifest_auto.json \
        --sheet ../scratch_bullseye/auto_prefill.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--captures", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    from ultralytics import YOLO
    model = YOLO(args.weights)
    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    caps = Path(args.captures)
    out = {"root": man.get("root", "../../"), "images": {}}
    tiles = []

    keys = sorted(man["images"].keys())
    if args.limit:
        keys = keys[:args.limit]
    for n, key in enumerate(keys, 1):
        rec = man["images"][key]
        src = caps / key
        if not src.exists():
            continue
        im = cv2.imread(str(src))
        if im is None:
            continue
        ih, iw = im.shape[:2]
        box = rec.get("box")
        pts = None
        conf = 0.0
        try:
            r = model.predict(str(src), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            if r.boxes is not None and len(r.boxes):
                i = int(np.argmax(r.boxes.conf.cpu().numpy()))
                conf = float(r.boxes.conf.cpu().numpy()[i])
                xy = r.keypoints.xy.cpu().numpy()[i]
                cf = (r.keypoints.conf.cpu().numpy()[i]
                      if r.keypoints.conf is not None else np.ones(len(xy)))
                if len(xy) >= 5:
                    pts = []
                    for k in range(5):
                        v = 2 if cf[k] >= 0.6 else (1 if cf[k] >= 0.3 else 0)
                        pts.append([round(float(xy[k][0]), 1), round(float(xy[k][1]), 1), v])
                    b = r.boxes.xywh.cpu().numpy()[i]
                    box = [round(float(b[0] - b[2] / 2), 1), round(float(b[1] - b[3] / 2), 1),
                           round(float(b[2]), 1), round(float(b[3]), 1)]
        except Exception as e:
            print("  [warn] %s: %s" % (key, e))
        if pts is None:
            x, y, w, h = box or [0, 0, iw, ih]
            pts = [[x, y, 2], [x + w, y, 2], [x + w, y + h, 2], [x, y + h, 2],
                   [x + w / 2, y + h / 2, 2]]
        out["images"][key] = {"url": rec.get("url"), "box": box,
                              "points": pts, "conf": round(conf, 3)}
        if len(tiles) < 16:
            vis = im.copy()
            for k, p in enumerate(pts):
                col = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 220, 220), (255, 0, 255)][k]
                cv2.circle(vis, (int(p[0]), int(p[1])), 4, col, -1)
            vis = cv2.resize(vis, (320, 240))
            cv2.putText(vis, "%s c%.2f" % (key.split("/")[-1][:22], conf), (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            tiles.append(vis)
        if n % 20 == 0:
            print("  %d/%d" % (n, len(keys)))

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[✓] %s（%d 张）" % (args.out, len(out["images"])))
    if args.sheet and tiles:
        cols = 4
        rows = int(np.ceil(len(tiles) / cols))
        sheet = np.zeros((rows * 240, cols * 320, 3), np.uint8)
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            sheet[r * 240:(r + 1) * 240, c * 320:(c + 1) * 320] = t
        Path(args.sheet).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.sheet, sheet)
        print("[✓] 抽检图：%s" % args.sheet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
