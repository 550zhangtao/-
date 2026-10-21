#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_live.py —— 用板端真实 kmodel 跑一段 live，统计检出/误检/靶心误差

为什么需要它：PC 上评的是 PyTorch 权重，板端跑的是 **量化后的 kmodel**，
而且 `yolo_pose.py` 还有候选筛选/校验逻辑。只有真板子跑出来的数字才算数
（AGENTS.md 的硬性要求）。

做法：
    * 起一个 `python3 -m k230tool live --duration N` 子进程；
    * 每 0.15 s 读一次它的 `latest.json`，按 seq 去重，收集逐帧结果；
    * 若给了 `--gt`（label_scene_frames.py 的产物），按 IoU≥0.5 匹配 GT，
      统计 top-1 命中率、靶心像素误差、误检数；不给则只打印检测分布。

用法：
    python3 scripts/verify_live.py --duration 20 --conf 0.25 \
        --gt /tmp/scene_ds/scene_labels.json --out /tmp/verify_v3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BOX_KEYS = ("x", "y", "w", "h")


def iou_xywh(box: list, gt: list) -> float:
    bx, by, bw, bh = [float(v) for v in box]
    b = [bx, by, bx + bw, by + bh]
    x1, y1 = max(b[0], gt[0]), max(b[1], gt[1])
    x2, y2 = min(b[2], gt[2]), min(b[3], gt[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ua = (b[2] - b[0]) * (b[3] - b[1]) + (gt[2] - gt[0]) * (gt[3] - gt[1]) - inter
    return 0.0 if ua <= 0 else inter / ua


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=20)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--model", default=None, help="板端 kmodel 路径（默认用板端配置）")
    ap.add_argument("--gt", default=None, help="label_scene_frames.py 的 JSON")
    ap.add_argument("--out", default="/tmp/verify_live")
    ap.add_argument("--json-every", type=float, default=0.15)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    latest = out / "latest.json"
    if latest.exists():
        latest.unlink()

    gt = None
    if args.gt:
        man = json.loads(Path(args.gt).read_text(encoding="utf-8"))
        gt = {"box": man["box_px"], "kpts": man["kpts_px"]}

    cmd = [sys.executable, "-m", "k230tool", "live", "--duration", str(args.duration),
           "--out", str(out), "--conf", str(args.conf), "--quiet"]
    if args.model:
        cmd += ["--model", args.model]
    print("[*] " + " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=Path(__file__).resolve().parent.parent)

    seen = {}
    t0 = time.time()
    try:
        while time.time() - t0 < args.duration + 30:
            if proc.poll() is not None and time.time() - t0 > 3:
                break
            rec = read_json(latest)
            if rec and rec.get("seq") is not None:
                seen[int(rec["seq"])] = rec
            time.sleep(args.json_every)
    finally:
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.terminate()

    recs = [seen[k] for k in sorted(seen)]
    print("[*] 收集到 %d 帧（去重后）" % len(recs))
    if not recs:
        print("[x] 没收到结果，检查串口是否被 IDE/其他进程占用")
        return 2

    rows = []
    for r in recs:
        dets = r.get("detections") or []
        bull = r.get("bull") or {}
        laser = r.get("laser") or {}
        top = None
        if dets:
            top = max(dets, key=lambda d: float(d.get("conf") or 0))
        rows.append({
            "seq": r["seq"], "fps": r.get("fps"), "infer_ms": r.get("infer_ms"),
            "n_det": len(dets), "top_conf": None if top is None else top.get("conf"),
            "top_box": None if top is None else top.get("box"),
            "bull": bull, "laser": laser,
            "iou_gt": None if (top is None or gt is None) else iou_xywh(top["box"], gt["box"]),
            "bull_err": None if (bull.get("status") != "ok" or gt is None)
                        else float(np.hypot(float(bull["x"]) - gt["kpts"][4][0],
                                            float(bull["y"]) - gt["kpts"][4][1])),
        })

    fps = [x["fps"] for x in rows if x.get("fps")]
    infer = [x["infer_ms"] for x in rows if x.get("infer_ms")]
    print("    帧率 中位 %.1f fps（%d 帧）  推理 中位 %.0f ms"
          % (np.median(fps) if fps else -1, len(rows),
             np.median(infer) if infer else -1))

    if gt is not None:
        ok = [x for x in rows if x["iou_gt"] is not None and x["iou_gt"] >= 0.5]
        wrong = [x for x in rows if x["iou_gt"] is not None and x["iou_gt"] < 0.5
                 and float(x["top_conf"] or 0) >= args.conf]
        miss = [x for x in rows if x["iou_gt"] is None or float(x["top_conf"] or 0) < args.conf]
        errs = [x["bull_err"] for x in rows if x.get("bull_err") is not None]
        print("    conf>=%.2f: top-1 命中 %d/%d = %.1f%%  误检 %d  漏检 %d"
              % (args.conf, len(ok), len(rows), 100.0 * len(ok) / len(rows), len(wrong), len(miss)))
        if errs:
            a = np.array(errs)
            print("    靶心误差 mean %.1f  median %.1f  P95 %.1f px（%d 帧）"
                  % (a.mean(), np.median(a), np.percentile(a, 95), len(a)))
        laser_ok = sum(1 for x in rows if (x["laser"] or {}).get("status") == "ok")
        print("    激光光斑 ok %d/%d" % (laser_ok, len(rows)))
    else:
        confs = [float(x["top_conf"]) for x in rows if x["top_conf"] is not None]
        if confs:
            a = np.array(confs)
            print("    top-1 conf: min %.3f  median %.3f  P95 %.3f"
                  % (a.min(), np.median(a), np.percentile(a, 95)))

    (out / "verify.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[✓] -> %s/verify.json" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
