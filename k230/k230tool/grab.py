#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grab.py —— 无窗口批量抓帧（给后续项目/数据集用）
================================================

直接保存板端回传的原始 JPEG 字节（不重新编码，零画质损失），
并写一份 meta.json 记录参数与时间戳。

用法：
    python3 -m k230tool grab                       # 抓 10 张到 frames/
    python3 -m k230tool grab -n 200 -o dataset     # 抓 200 张
    python3 -m k230tool grab -n 0 --every 0.2      # 无限抓，每 0.2s 一张，Ctrl-C 停
    python3 -m k230tool grab -n 1 --stdout | ...   # 抓 1 张并把 JPEG 写到标准输出
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

from .link import DEFAULT_CFG, K230Stream, describe_ports, find_port  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="K230 批量抓帧")
    ap.add_argument("--port", default=None)
    ap.add_argument("-n", "--count", type=int, default=10, help="抓多少张，0=无限")
    ap.add_argument("-o", "--out", default="frames", help="输出目录")
    ap.add_argument("-w", "--width", type=int, default=DEFAULT_CFG["w"])
    ap.add_argument("-H", "--height", type=int, default=DEFAULT_CFG["h"])
    ap.add_argument("-q", "--quality", type=int, default=DEFAULT_CFG["q"])
    ap.add_argument("--fmt", choices=["rgb565", "grayscale"], default="rgb565")
    ap.add_argument("--fps", type=int, default=0, help="板端限速，0=跑满")
    ap.add_argument("--every", type=float, default=0.0,
                    help="每张之间间隔秒数（默认跟板端帧率）")
    ap.add_argument("--stdout", action="store_true",
                    help="把 JPEG 直接写到标准输出（不落盘，便于管道）")
    args = ap.parse_args()

    if not find_port():
        print("没有找到 K230。当前串口：\n%s" % describe_ports(), file=sys.stderr)
        return 2

    cfg = {"w": args.width, "h": args.height, "q": args.quality,
           "fmt": args.fmt, "fps": args.fps, "stats": 0}
    st = K230Stream(cfg=cfg, port=args.port, log=lambda m: None)
    st.start(timeout=25)
    if not st.hello:
        print("板端无响应", file=sys.stderr)
        st.stop()
        return 3

    out_dir = Path(args.out)
    if not args.stdout:
        out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    idx = 0
    saved = 0
    t_start = time.time()
    try:
        while args.count == 0 or saved < args.count:
            jpeg, new_idx = st.wait_frame(idx, timeout=3.0)
            if jpeg is None:
                print("[!] 3 秒没收到新帧，跳过", file=sys.stderr)
                continue
            idx = new_idx
            if args.every > 0:
                time.sleep(args.every)

            if args.stdout:
                sys.stdout.buffer.write(jpeg)
                sys.stdout.buffer.flush()
            else:
                path = out_dir / ("frame_%06d.jpg" % saved)
                path.write_bytes(jpeg)
                records.append({"file": path.name,
                                "bytes": len(jpeg),
                                "ts": round(time.time(), 3)})
            saved += 1
            if saved % 10 == 0 or saved == 1:
                dt = time.time() - t_start
                print("[*] %d 张 / %.1fs = %.1f 张每秒，最新 %.1f KB"
                      % (saved, dt, saved / dt if dt else 0, len(jpeg) / 1024.0),
                      file=sys.stderr)
    except KeyboardInterrupt:
        print("\n[*] Ctrl-C 停止", file=sys.stderr)
    finally:
        st.stop()

    if not args.stdout:
        meta = {
            "board": st.hello,
            "cfg": cfg,
            "count": saved,
            "avg_bytes": round(sum(r["bytes"] for r in records) / saved, 1) if saved else 0,
            "frames": records,
        }
        (out_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[*] 已保存 %d 张到 %s（附 meta.json）" % (saved, out_dir), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
