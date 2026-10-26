# -*- coding: utf-8 -*-
"""
check_pose_labels.py —— 检查 pose_labels.json + 生成抽检图
=========================================================

用法：
    python3 tools/check_pose_labels.py --json ../pose_labels.json \
        --captures ../k230_target_project/captures \
        --out ../scratch_bullseye/pose_check.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

COLORS = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 220, 220), (255, 0, 255)]
NAMES = ["TL", "TR", "BR", "BL", "BULL"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--captures", required=True)
    ap.add_argument("--out", default="pose_check.jpg")
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args(argv)

    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    imgs = data.get("images", {})
    caps = Path(args.captures)
    problems = []
    vcnt = Counter(); stages = Counter()
    tiles = []
    done = 0

    for key, rec in sorted(imgs.items()):
        pts = rec.get("points") or []
        w, h = int(rec.get("w") or 640), int(rec.get("h") or 480)
        if len(pts) != 5:
            problems.append("%s: 点数 %d != 5" % (key, len(pts)))
            continue
        for i, p in enumerate(pts):
            if len(p) < 3:
                problems.append("%s: 点 %s 缺 v" % (key, NAMES[i]))
                continue
            vcnt[int(p[2])] += 1
            if not (0 <= p[0] <= w and 0 <= p[1] <= h):
                problems.append("%s: %s 越界 (%.0f,%.0f)" % (key, NAMES[i], p[0], p[1]))
        if rec.get("done"):
            done += 1
        stages[key.split("/")[0]] += 1

        if len(tiles) < 16:
            src = caps / key
            if not src.exists():
                src = caps / key.split("/", 1)[-1]
            im = cv2.imread(str(src))
            if im is None:
                continue
            for i, p in enumerate(pts):
                col = COLORS[i] if int(p[2]) > 0 else (90, 90, 90)
                cv2.circle(im, (int(p[0]), int(p[1])), 5, col, -1)
                cv2.putText(im, NAMES[i], (int(p[0]) + 6, int(p[1]) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            if all(int(p[2]) > 0 for p in pts[:4]):
                q = np.array([[int(p[0]), int(p[1])] for p in pts[:4]], np.int32)
                cv2.polylines(im, [q], True, (0, 255, 0), 1)
            im = cv2.resize(im, (320, 240))
            cv2.putText(im, key.split("/")[-1][:28], (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 255), 1)
            tiles.append(im)

    print("[*] 完成 %d / %d 张；阶段分布：%s" % (done, len(imgs), dict(stages)))
    print("[*] 可见性分布：v0=%d v1=%d v2=%d" % (vcnt[0], vcnt[1], vcnt[2]))
    if problems:
        print("[!] 发现 %d 个问题：" % len(problems))
        for p in problems[:30]:
            print("   ", p)
    else:
        print("[✓] 格式检查通过")

    if tiles:
        cols = args.cols
        rows = int(np.ceil(len(tiles) / cols))
        sheet = np.zeros((rows * 240, cols * 320, 3), np.uint8)
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            sheet[r * 240:(r + 1) * 240, c * 320:(c + 1) * 320] = t
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.out, sheet)
        print("[✓] 抽检图：%s" % args.out)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
