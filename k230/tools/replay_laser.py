#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""replay_laser.py —— 激光固定分支的主机回放/合成评测（无需板子）

用途：
  1) 合成场景 A/B：calib 过期，真实光斑跳到 200+px / 画面底部；检查会不会
     锁到背景里的旧光斑/反射（回放前会把旧光斑和地面反射一起擦掉）。
  2) 合成场景 C：手挡住 calib 6~8s 再移开；检查手/高光会不会更新 fixed。
  3) 真实样本 k230/tools/laser_samples：固定分支命中率/误检/耗时。

用法：
    python3 k230/tools/replay_laser.py --all
    python3 k230/tools/replay_laser.py --synthetic
    python3 k230/tools/replay_laser.py --samples
    python3 k230/tools/replay_laser.py --old /tmp/old_laser_spot.py --all

注意：这是主机回放，耗时是 CPython 的参考值；板端 laser_ms 以 live/代理日志为准。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
W, H = 640, 480
SAMPLES = ROOT / "tools" / "laser_samples"
BASE = SAMPLES / "wall_on_20261021_203504" / "raw" / "frame_0001.jpg"

if not hasattr(time, "ticks_ms"):
    time.ticks_ms = lambda: int(time.monotonic() * 1000)
if not hasattr(time, "ticks_diff"):
    time.ticks_diff = lambda a, b: a - b


# ---------------------------------------------------------------- 模块
def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def old_module() -> Path:
    out = Path("/tmp/laser_spot_old.py")
    src = subprocess.check_output(
        ["git", "show", "HEAD:k230/board/laser_spot.py"], cwd=str(ROOT.parent))
    out.write_bytes(src)
    return out


# ---------------------------------------------------------------- 图像
def load_rgb(path: Path):
    from PIL import Image
    im = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)
    return [list(p) for p in im.getdata()]


def pack(px) -> bytearray:
    ba = bytearray(W * H * 2)
    i = 0
    for r, g, b in px:
        v = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
        ba[i] = v & 0xFF
        ba[i + 1] = (v >> 8) & 0xFF
        i += 2
    return ba


class Img:
    def __init__(self, px):
        self._ba = pack(px)

    def bytearray(self):
        return self._ba

    def width(self):
        return W

    def height(self):
        return H


def _put(px, x, y, c):
    if 0 <= x < W and 0 <= y < H:
        px[y * W + x] = list(c)


def erase_spot(px, cx, cy, r=14):
    """用环平均把旧光斑/反射擦掉，避免合成场景里残留第二个光斑。"""
    acc = [0, 0, 0]
    n = 0
    for y in range(cy - r, cy + r + 1):
        for x in range(cx - r, cx + r + 1):
            if not (0 <= x < W and 0 <= y < H):
                continue
            d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if r - 4 <= d <= r:
                c = px[y * W + x]
                acc[0] += c[0]
                acc[1] += c[1]
                acc[2] += c[2]
                n += 1
    c = [v // max(1, n) for v in acc]
    for y in range(max(0, cy - r), min(H, cy + r + 1)):
        for x in range(max(0, cx - r), min(W, cx + r + 1)):
            px[y * W + x] = list(c)


def draw_spot(px, cx, cy):
    """红色晕圈 + 过曝白核，模拟墙上的激光。"""
    for y in range(cy - 8, cy + 9):
        for x in range(cx - 8, cx + 9):
            if not (0 <= x < W and 0 <= y < H):
                continue
            d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if d <= 7:
                k = max(0.0, 1.0 - d / 8.0)
                r, g, b = px[y * W + x]
                _put(px, x, y, (min(255, int(r * (1 - k) + 255 * k)),
                                min(255, int(g * (1 - k) + 60 * k)),
                                min(255, int(b * (1 - k) + 70 * k))))
    for y in range(cy - 2, cy + 3):
        for x in range(cx - 2, cx + 3):
            _put(px, x, y, (255, 255, 255))


def draw_hand(px, x0, y0, x1, y1):
    import random
    for y in range(y0, y1):
        for x in range(x0, x1):
            base = 205 + int(18 * (x - x0) / max(1, x1 - x0)) + random.randint(-4, 4)
            _put(px, x, y, (base, 158 + random.randint(-4, 4),
                            138 + random.randint(-4, 4)))


def base_scene():
    """擦掉样本里的旧墙光斑和地面反射，作为合成场景背景。"""
    px = load_rgb(BASE)
    erase_spot(px, 408, 278, 14)
    erase_spot(px, 248, 273, 16)
    return px


# ---------------------------------------------------------------- 场景
def run_scene(mod, calib, spot, frames=120, hand=None):
    base = base_scene()
    ls = mod.LaserSpot({"fixed_verify_every": 5,
                        "fixed_allow_reacquire": 1, "budget_ms": 12})
    ls.set_fixed(*calib)
    first = None
    false = 0
    ms = []
    last = None
    for i in range(frames):
        px = [p[:] for p in base]
        if hand and hand[0] <= i < hand[1]:
            draw_hand(px, *hand[2])
        elif spot is not None:
            draw_spot(px, spot[0], spot[1])
        img = Img(px)
        t = time.ticks_ms()
        r = ls.locate(img)
        ms.append(time.ticks_diff(time.ticks_ms(), t))
        last = r
        if r is not None and r.get("src") == "measured":
            if first is None:
                first = i
            if spot is not None:
                d = ((r["x"] - spot[0]) ** 2 + (r["y"] - spot[1]) ** 2) ** 0.5
                if d > 80:
                    false += 1
    return {"first": first, "false": false, "last": last,
            "need_recalib": bool(getattr(ls, "need_recalib", False)),
            "p50": statistics.median(ms) if ms else 0,
            "p95": sorted(ms)[max(0, int(len(ms) * 0.95) - 1)] if ms else 0}


def run_synthetic(mod, tag):
    print("\n== 合成场景：%s ==" % tag)
    scenes = [
        ("A 旧calib/光斑跳到(500,400)", (312, 230), (500, 400), 120, None),
        ("B 旧calib/光斑到底部(313,464)", (312, 230), (313, 464), 120, None),
        ("C 手挡calib 60帧后移开", (408, 278), (408, 278), 140,
         (20, 80, (300, 190, 520, 380))),
    ]
    for name, calib, spot, n, hand in scenes:
        r = run_scene(mod, calib, spot, n, hand)
        last = r["last"]
        print("  %-30s first_measured=%-4s false>80px=%-3d recalib=%-5s p50=%2dms p95=%2dms last=%s"
              % (name, r["first"], r["false"], r["need_recalib"], r["p50"], r["p95"],
                 None if last is None else (round(last["x"], 1), round(last["y"], 1),
                                            last.get("src"), last.get("conf"))))


# ---------------------------------------------------------------- 样本
SAMPLE_SETS = {
    "wall_on_20261021_202320": ((382, 259), True),
    "wall_on_20261021_202449": ((382, 269), True),
    "wall_on_20261021_202554": ((381, 268), True),
    "wall_on_20261021_203315": ((409, 282), True),
    "wall_on_20261021_203504": ((408, 278), True),
    "wall_off_20261021_202001": ((320, 240), False),
    "wall_off_20261021_202629": ((320, 240), False),
    "wall_off_20261021_203939": ((320, 240), False),
}


def run_samples(mod, tag):
    print("\n== 真实样本：%s ==" % tag)
    for name, (calib, is_on) in SAMPLE_SETS.items():
        files = sorted((SAMPLES / name / "raw").glob("*.jpg"))
        if not files:
            continue
        ls = mod.LaserSpot({"fixed_verify_every": 5,
                            "fixed_allow_reacquire": 1, "budget_ms": 12})
        ls.set_fixed(*calib)
        meas = 0
        ms = []
        last = None
        for f in files:
            img = Img(load_rgb(f))
            t = time.ticks_ms()
            r = ls.locate(img)
            ms.append(time.ticks_diff(time.ticks_ms(), t))
            last = r
            if r is not None and r.get("src") == "measured":
                meas += 1
        print("  %-28s on=%-5s frames=%2d measured=%2d p95=%2dms last=%s"
              % (name, is_on, len(files), meas,
                 sorted(ms)[max(0, int(len(ms) * 0.95) - 1)] if ms else 0,
                 None if last is None else (round(last["x"], 1), round(last["y"], 1),
                                            last.get("src"))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default=str(ROOT / "board" / "laser_spot.py"))
    ap.add_argument("--old", default=None, help="旧版 laser_spot.py；不给则用 git HEAD")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args(argv)
    do_all = args.all or (not args.synthetic and not args.samples)
    new = load_module(Path(args.module), "laser_new")
    old_path = Path(args.old) if args.old else old_module()
    old = load_module(old_path, "laser_old")
    if do_all or args.synthetic:
        run_synthetic(old, "OLD")
        run_synthetic(new, "NEW")
    if do_all or args.samples:
        run_samples(old, "OLD")
        run_samples(new, "NEW")
    return 0


if __name__ == "__main__":
    sys.exit(main())
