# -*- coding: utf-8 -*-
"""
eval_live.py —— 板端实时检出率统计
==================================

直接跑板端 YOLO 流，统计一段时间内的：
    * 有检出帧比例（recall proxy）
    * 多检帧比例（误检 proxy）
    * 平均置信度、平均帧率、平均推理耗时

用法：
    python3 tools/eval_live.py --model /sdcard/models/pose_v3_256.kmodel --seconds 30
    python3 tools/eval_live.py --model ... --conf 0.4 --json out.json
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
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from k230tool.link import K230Stream, find_port  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/sdcard/models/pose_v3_256.kmodel")
    ap.add_argument("--labels", default="target")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--input-size", type=int, default=320)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--port", default=None)
    args = ap.parse_args(argv)

    if not find_port():
        print("没有找到 K230")
        return 2
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    st = K230Stream(cfg={
        "model": args.model, "labels": labels,
        "in_w": args.input_size, "in_h": args.input_size,
        "conf": args.conf, "nms": args.nms,
        "w": 640, "h": 480, "q": 70, "stats": 10,
    }, ai=True, auto_restart=False, log=lambda m: print("[k230]", m))
    st.start(timeout=40)
    if not st.hello:
        print("板端无响应：", st.last_error)
        st.stop()
        return 3

    last = 0
    n = 0
    hit = 0
    multi = 0
    confs = []
    infer = []
    t0 = time.time()
    try:
        while time.time() - t0 < args.seconds:
            jpeg, idx = st.wait_frame(last, timeout=2.0)
            if jpeg is None:
                continue
            last = idx
            d = st.detections(idx) or {}
            dets = d.get("det") or []
            n += 1
            if dets:
                hit += 1
                confs.append(max(float(x[4]) for x in dets))
            if len(dets) >= 2:
                multi += 1
            if d.get("ms"):
                infer.append(float(d["ms"]))
            if n % 25 == 0:
                print("  %3d 帧  有检出 %3d  多检 %3d  平均conf %.2f" %
                      (n, hit, multi, (sum(confs) / len(confs)) if confs else 0))
    except KeyboardInterrupt:
        pass
    finally:
        st.stop()

    dt = time.time() - t0
    res = {
        "frames": n,
        "fps": round(n / dt, 2) if dt else 0,
        "frames_with_det": hit,
        "hit_rate": round(hit / n, 3) if n else 0,
        "multi_det_frames": multi,
        "mean_conf": round(sum(confs) / len(confs), 3) if confs else 0,
        "mean_infer_ms": round(sum(infer) / len(infer), 1) if infer else 0,
        "conf_thresh": args.conf,
        "model": args.model,
    }
    print("\n结果：", json.dumps(res, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
