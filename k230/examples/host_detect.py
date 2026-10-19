#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：直接消费 K230 视频流，做自己的处理
=======================================

适合把 K230 当成普通摄像头：

    python3 examples/host_detect.py --no-ai          # 12 fps 原始流
    python3 examples/host_detect.py --ai             # YOLO 流 + 检测结果

这里只做两件事：
    1. JPEG -> BGR ndarray（和 cv2.VideoCapture 一样）
    2. 打印/保存；你自己的算法写在 process() 里
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from k230tool.link import K230Stream, find_port  # noqa: E402

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("需要 opencv-python / numpy")


def process(frame, seq, target=None):
    """你的算法入口。frame: BGR (H,W,3)，target: 板端检测 JSON。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 例子：中心区域平均亮度 + 检测框数量
    h, w = gray.shape
    roi = gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    return {
        "seq": seq,
        "brightness": float(np.mean(roi)),
        "targets": len((target or {}).get("det") or []),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai", action="store_true", help="跑板端 YOLO（默认关）")
    ap.add_argument("--model", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--save", default=None, help="把每帧存到这个目录")
    ap.add_argument("--seconds", type=float, default=0, help="跑多久，0=一直")
    args = ap.parse_args()

    cfg = {"w": args.w, "h": args.h, "q": 70, "stats": 10, "fps": 0}
    if args.ai:
        cfg.update({"model": args.model or "/sdcard/models/v2_std1.kmodel",
                    "labels": ["target"], "conf": args.conf})
    st = K230Stream(cfg=cfg, ai=args.ai, auto_restart=True,
                    log=lambda m: print("[k230]", m))
    st.start(timeout=40)
    if not st.hello:
        print("板端无响应：", st.last_error)
        st.stop()
        return 1

    out = Path(args.save) if args.save else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    last = 0
    n = 0
    t0 = time.time()
    try:
        while args.seconds <= 0 or time.time() - t0 < args.seconds:
            jpeg, idx = st.wait_frame(last, timeout=2.0)
            if jpeg is None:
                continue
            last = idx
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            target = st.detections(idx) if args.ai else None
            info = process(frame, idx, target)
            n += 1
            if n % 10 == 1:
                fps = n / max(1e-6, time.time() - t0)
                print("seq=%d %.1ffps %s" % (idx, fps, info))
            if out:
                cv2.imwrite(str(out / ("f_%06d.jpg" % idx)), frame)
    except KeyboardInterrupt:
        pass
    finally:
        st.stop()
    print("共 %d 帧" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
