#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在板端对"单张图片"跑激光检测 —— 用于快速回归验证，不用反复摆相机。

原理：把图片推到 /sdcard，用 `image.Image(path).to_rgb565()` 加载，
再调用与预览/推流完全相同的 `board/laser_spot.py`。

注意两个坑（都踩过）：
  1. 必须 `.to_rgb565()`，否则 bytearray 是 3 字节/像素，`_rgb565()` 解析全乱；
  2. 采集下来的"全图"帧通常是 320x240，坐标要按 真值*宽/640 换算。

用法：
    python3 tools/test_laser_frame.py a.jpg [b.jpg ...] --expect 376,249 --no-expect c.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from k230tool.link import K230Link  # noqa: E402

BOARD_SCRIPT = r'''
import sys, time, gc
sys.path.append('/sdcard')
try:
    sys.modules.pop('laser_spot', None)
except Exception:
    pass
import laser_spot, image

CASES = %(cases)s

def run(path, tag, gt):
    im = image.Image(path)
    try:
        im = im.to_rgb565()
    except Exception as e:
        print(tag, "to_rgb565 fail:", e)
        return
    w, h = im.width(), im.height()
    lf = laser_spot.LaserSpot()
    t0 = time.ticks_us()
    r = lf.locate(im)
    dt = time.ticks_diff(time.ticks_us(), t0) // 1000
    if gt is None:
        print("%%-10s %%dx%%d  %%-7s %%3dms cand=%%d%%s" %% (
            tag, w, h, "HIT" if r else "MISS", dt, lf.last_cands,
            ("  (%%0.1f,%%0.1f) %%s" %% (r["x"], r["y"], r["mode"])) if r else ""))
    else:
        gx, gy = gt[0] * w / 640.0, gt[1] * h / 480.0
        if r:
            d = ((r["x"]-gx)**2 + (r["y"]-gy)**2) ** 0.5
            print("%%-10s %%dx%%d  HIT   (%%0.1f,%%0.1f) d=%%.1fpx  %%s  %%dms" %% (
                tag, w, h, r["x"], r["y"], d, r["mode"], dt))
        else:
            print("%%-10s %%dx%%d  MISS  真值(%%0.0f,%%0.0f)  %%dms" %% (tag, w, h, gx, gy, dt))
    gc.collect()

for path, tag, gt in CASES:
    run(path, tag, gt)
print("done")
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--laser", nargs="*", default=[],
                    help="有激光的图，格式 路径:x,y（真值用 640x480 坐标系）")
    ap.add_argument("--nolaser", nargs="*", default=[],
                    help="无激光的图（应 0 命中）")
    ap.add_argument("--port", default=None)
    args = ap.parse_args()

    cases = []
    link = K230Link(port=args.port)
    link.open()
    try:
        from k230tool.push_file import push_one
        idx = 0
        for item in args.laser:
            path, gt = item.rsplit(":", 1)
            gx, gy = [float(v) for v in gt.split(",")]
            idx += 1
            remote = "/sdcard/_t%d.jpg" % idx
            push_one(link, str(Path(path)), remote, verbose=False)
            cases.append((remote, "laser%d" % idx, (gx, gy)))
        for item in args.nolaser:
            idx += 1
            remote = "/sdcard/_t%d.jpg" % idx
            push_one(link, str(Path(item)), remote, verbose=False)
            cases.append((remote, "nolser%d" % idx, None))
        src = BOARD_SCRIPT % {"cases": repr(cases)}
        out, err, ok = link.exec_and_collect(src, timeout=120)
        print(out)
        if err.strip():
            print("ERR:", err[:600])
        return 0
    finally:
        link.close()


if __name__ == "__main__":
    sys.exit(main())
