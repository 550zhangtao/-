#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：不碰串口，只读 live 的输出（Agent / 其它进程推荐这样用）
=============================================================

    python3 -m k230tool live &            # 另开一个终端/后台
    python3 examples/watch_latest.py       # 打印最新检测
    python3 examples/watch_latest.py --show  # 顺便用 OpenCV 显示
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "live"


def read_json():
    p = LIVE / "latest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="OpenCV 显示 latest.jpg")
    ap.add_argument("--interval", type=float, default=0.5)
    args = ap.parse_args()

    last_seq = -1
    try:
        while True:
            d = read_json()
            if d is None:
                print("还没有 live/latest.json，先启动：python3 -m k230tool live")
            elif d.get("seq") != last_seq:
                last_seq = d.get("seq")
                dets = d.get("detections") or []
                print("seq=%s fps=%.1f age=%.0fms targets=%d %s" % (
                    d.get("seq"), d.get("fps", 0), d.get("age_ms") or 0,
                    len(dets),
                    " | ".join("%s %.2f %s" % (x["label"], x["conf"], x["box"])
                               for x in dets)))
            if args.show:
                import cv2
                img = cv2.imread(str(LIVE / "latest.jpg"))
                if img is not None:
                    cv2.imshow("K230 live", img)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
