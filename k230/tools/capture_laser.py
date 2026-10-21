#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
采集激光光斑样本：USB 推流，保存原始帧 + 逐帧检测结果。

用法：
    python3 tools/capture_laser.py --label wall_on --duration 10
    python3 tools/capture_laser.py --label paper_on --duration 10

输出到 k230/tools/laser_samples/<label>_<时间戳>/：
    raw/frame_0001.jpg ...   原始帧（全图 640x480）
    result.jsonl             每帧 {"seq","w","h","roi","laser","lms","ms","bull"}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from k230tool.link import K230Stream, POSE_DEFAULT_CFG  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--out", default=str(ROOT / "tools" / "laser_samples"))
    ap.add_argument("--laser", type=int, default=1, help="1=开激光检测 0=关")
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) / ("%s_%s" % (args.label, stamp))
    (out / "raw").mkdir(parents=True, exist_ok=True)

    cfg = dict(POSE_DEFAULT_CFG)
    cfg.update({"jpeg_every": 1, "full_every": 1,   # 每帧都要全图，便于后处理
                "laser_enable": args.laser, "stats": 5})
    st = K230Stream(cfg=cfg, ai=True, auto_restart=False,
                    log=lambda m: print("[cap]", m))
    st.start(timeout=40)

    meta = open(out / "result.jsonl", "w", encoding="utf-8")
    n = 0
    got = 0
    t0 = time.time()
    hits = 0
    try:
        while time.time() - t0 < args.duration:
            jpg, idx = st.wait_frame(got, timeout=1.0)
            if jpg is None or idx == got:
                continue
            got = idx
            tgt = st.detections(idx) or {}
            # 只保留全图帧（避免激光锁定后的 ROI 帧覆盖全图）
            roi = tgt.get("roi") or [0, 0, 640, 480]
            if roi[2] < 600:
                continue
            n += 1
            (out / "raw" / ("frame_%04d.jpg" % n)).write_bytes(jpg)
            laser = tgt.get("laser") or {}
            if laser.get("status") == "ok":
                hits += 1
            meta.write(json.dumps({
                "file": "raw/frame_%04d.jpg" % n, "seq": idx,
                "roi": roi, "laser": laser, "laser_ms": tgt.get("lms"),
                "infer_ms": tgt.get("ms"), "bull": tgt.get("bull"),
                "kpts": tgt.get("kp"),
            }, ensure_ascii=False) + "\n")
            if n % 10 == 0:
                meta.flush()
    finally:
        meta.close()
        st.stop()

    print("[cap] 完成：%d 张全图 -> %s" % (n, out))
    print("[cap] 有光斑 %d / %d = %.1f%%" % (hits, n, 100.0 * hits / max(1, n)))
    (out / "summary.json").write_text(json.dumps({
        "label": args.label, "frames": n, "hits": hits,
        "hit_rate": round(100.0 * hits / max(1, n), 2),
        "laser_enable": args.laser,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
