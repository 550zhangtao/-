# -*- coding: utf-8 -*-
"""
kmodel_eval.py —— 直接在 PC 上用 nncase Simulator 评估 kmodel（含 PTQ 档位）
=========================================================================

用途：
    * 对比不同 PTQ（uint8/Kld、NoClip、MSE 权重等）对 5 关键点精度的影响；
    * 不占串口、不需要推板。

输入预处理与板端一致：
    640x480 -> letterbox 320x320（pad 114）-> RGB -> NCHW uint8

用法：
    python3 tools/kmodel_eval.py --kmodel poseB.kmodel \
        --data ../k230_target_project/dataset_pose_B --split test \
        --kpt 5 --conf 0.25 --out eval_kmodel.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


def setup_nncase():
    os.environ.setdefault("DOTNET_ROOT", os.path.expanduser("~/.dotnet"))
    sp = Path.home() / ".local" / "lib" / "python3.10" / "site-packages"
    os.environ["PATH"] = str(sp) + ":" + os.environ.get("PATH", "")
    import nncase
    return nncase


def load_gt(path: Path, kpt: int):
    if not path.exists():
        return None
    lines = [l.split() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        return None
    t = lines[0]
    pts = []
    for i in range(kpt):
        o = 5 + i * 3
        if o + 2 >= len(t):
            return None
        pts.append([float(t[o]), float(t[o + 1]), int(float(t[o + 2]))])
    return pts


def letterbox(img, size=320):
    h, w = img.shape[:2]
    s = min(size / w, size / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh))
    # BGR -> RGB, NHWC -> NCHW
    x = np.transpose(canvas[:, :, ::-1], (2, 0, 1))[None]
    return np.ascontiguousarray(x), s, px, py


def geom_ok(pts) -> bool:
    if len(pts) < 5:
        return False
    P = [(p[0], p[1]) for p in pts[:4]]
    signs = []
    for i in range(4):
        ax, ay = P[i]; bx, by = P[(i + 1) % 4]; cx, cy = P[(i + 2) % 4]
        cr = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if abs(cr) < 1e-3:
            return False
        signs.append(1 if cr > 0 else -1)
    if len(set(signs)) != 1:
        return False
    sides = [math.hypot(P[(i + 1) % 4][0] - P[i][0], P[(i + 1) % 4][1] - P[i][1]) for i in range(4)]
    if min(sides) < 8:
        return False
    if max(sides[0], sides[2]) / min(sides[0], sides[2]) > 2.5:
        return False
    if max(sides[1], sides[3]) / min(sides[1], sides[3]) > 2.5:
        return False
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmodel", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--kpt", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--gt-data", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    nncase = setup_nncase()
    sim = nncase.Simulator()
    sim.load_model(Path(args.kmodel).read_bytes())

    data = Path(args.data)
    img_dir = data / "images" / args.split
    lbl_dir = (Path(args.gt_data) if args.gt_data else data) / "labels" / args.split
    files = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not files:
        raise SystemExit("测试集为空：%s" % img_dir)

    errs = []; corner_errs = []; ok = 0; fails = {}
    per_stage = {}
    for p in files:
        gt = load_gt(lbl_dir / (p.stem + ".txt"), 5)
        if gt is None:
            continue
        im = cv2.imread(str(p))
        if im is None:
            continue
        ih, iw = im.shape[:2]
        x, s, px, py = letterbox(im, args.imgsz)
        sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(x))
        sim.run()
        y = sim.get_output_tensor(0).to_numpy()[0]
        idx = int(np.argmax(y[4]))
        score = float(y[4][idx])
        st = "ok"
        if score < args.conf:
            st = "low_conf"
        else:
            row = y[:, idx]
            pts = []
            for k in range(5):
                kx = (float(row[5 + k * 3]) - px) / s
                ky = (float(row[5 + k * 3 + 1]) - py) / s
                kv = float(row[5 + k * 3 + 2])
                pts.append([kx, ky, 1 if kv >= 0.3 else 0])
            if not geom_ok(pts):
                st = "bad_geom"
            else:
                pred = np.array([pts[4][0], pts[4][1]])
                g = np.array([gt[4][0] * iw, gt[4][1] * ih])
                errs.append(float(np.linalg.norm(pred - g)))
                ce = []
                for k in range(4):
                    ge = np.array([gt[k][0] * iw, gt[k][1] * ih])
                    ce.append(float(np.linalg.norm(np.array(pts[k][:2]) - ge)))
                corner_errs.extend(ce)
                ok += 1
        fails[st] = fails.get(st, 0) + 1
        stage = p.stem.split("_frame")[0] if "_frame" in p.stem else "unknown"
        g = per_stage.setdefault(stage, {"n": 0, "ok": 0, "bull_err": [], "fails": {}})
        g["n"] += 1
        if st == "ok":
            g["ok"] += 1
            g["bull_err"].append(errs[-1])
        else:
            g["fails"][st] = g["fails"].get(st, 0) + 1

    def stat(a):
        if not a:
            return {"n": 0}
        a = np.array(a)
        return {"n": len(a), "mean": round(float(a.mean()), 3),
                "p95": round(float(np.percentile(a, 95)), 3),
                "max": round(float(a.max()), 3)}

    for g in per_stage.values():
        e = g.pop("bull_err"); g["bull_err"] = stat(e)
        g["ok_rate"] = round(g["ok"] / max(1, g["n"]), 3)

    res = {"kmodel": args.kmodel, "split": args.split, "images": len(files),
           "ok": ok, "bull_err": stat(errs), "corner_err": stat(corner_errs),
           "fails": fails, "per_stage": per_stage}
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
