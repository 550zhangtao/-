# -*- coding: utf-8 -*-
"""
capture_dataset.py —— 分阶段自动采集标靶数据集
==============================================

设计成「脚本管流程，你管摆场景」：
    每个阶段开始前，脚本先用同心圆检测看板端画面里有没有标靶、够不够大；
    合格才开抓，不合格就一直提示你调整（可以按 s 跳过本阶段 / q 退出）。

用法（在 k230 根目录）：
    python3 tools/capture_dataset.py --out captures
    python3 tools/capture_dataset.py --out captures --stage near --count 40
    python3 tools/capture_dataset.py --out captures --stages pos_mid,pos_near,neg

阶段默认值：
    pos_far   60 张  标靶占画面 1/10 ~ 1/6
    pos_mid   70 张  标靶占画面 1/5 ~ 1/3
    pos_near  70 张  标靶占画面 1/3 ~ 1/2
    pos_angle 60 张  左右/上下倾斜 15~30°
    pos_edge  50 张  标靶放在四角/边缘
    pos_light 40 张  顺光/侧光/逆光/阴影
    pos_occl  30 张  纸片/手遮住 10~30%
    pos_blur  30 张  轻微晃动
    neg       80 张  画面里不能有标靶

每个阶段抓完会：
    * 写 captures/<stage>/frame_XXXXXX.jpg
    * 写 captures/<stage>/meta.json
    * 抽 3 张画上检测圆，存 captures/<stage>/_preview/ 方便你核对
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
from typing import Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from k230tool.link import K230Stream, find_port  # noqa: E402


STAGES = {
    "pos_far":   dict(count=60, every=0.12, need_target=True,
                      hint="标靶放【远】一点：占画面约 1/10 ~ 1/6，完整可见，别遮挡"),
    "pos_mid":   dict(count=70, every=0.12, need_target=True,
                      hint="标靶放【中】距离：占画面约 1/5 ~ 1/3"),
    "pos_near":  dict(count=70, every=0.12, need_target=True,
                      hint="标靶放【近】：占画面约 1/3 ~ 1/2，完整可见"),
    "pos_angle": dict(count=60, every=0.12, need_target=True,
                      hint="标靶左右/上下【倾斜 15~30°】，边倾斜边慢慢移动"),
    "pos_edge":  dict(count=50, every=0.12, need_target=True,
                      hint="把标靶放到画面【四角/边缘】，每个位置停 2 秒"),
    "pos_light": dict(count=40, every=0.15, need_target=True,
                      hint="改变【光照】：顺光/侧光/背光/开灯关灯/有阴影"),
    "pos_occl":  dict(count=30, every=0.15, need_target=True,
                      hint="用纸片或手【遮住标靶 10~30%】，换不同方位遮"),
    "pos_blur":  dict(count=30, every=0.10, need_target=True,
                      hint="手拿标靶【轻微晃动】时抓拍，制造轻微运动模糊"),
    "pos_robot": dict(count=70, every=0.05, need_target=True,
                      hint="相机由机械臂/小车带着【整体平移/旋转/俯仰】，标靶保持可见"),
    "static":    dict(count=20, every=0.2, need_target=True,
                      hint="标靶静止放在相机正前方，不要动"),
    "neg":       dict(count=80, every=0.15, need_target=False,
                      hint="画面里【不要有标靶】：拍桌面/线缆/键盘/墙/人/其他圆形物"),
    # ---- 补充场景（v2 数据集，独立保存，不覆盖原 captures） ----
    "sup_lowlight": dict(count=30, every=0.15, need_target=True,
                      hint="弱光：关主灯，只留台灯/屏幕侧光，靶纸可辨但偏暗"),
    "sup_backlight": dict(count=30, every=0.15, need_target=True,
                      hint="逆光/过曝：强光在靶后/侧后，白卡过曝、黑环发灰"),
    "sup_tilt":   dict(count=40, every=0.12, need_target=True,
                      hint="大倾角：水平 ±45~60°，俯仰 ±30°，滚转 90/180°"),
    "sup_far":    dict(count=30, every=0.12, need_target=True,
                      hint="远距离：3~5m，靶纸只占画面 30~60px"),
    "sup_occl":   dict(count=30, every=0.15, need_target=True,
                      hint="遮挡：手/物遮 1~2 个角、遮靶心、遮半张"),
    "sup_near":   dict(count=20, every=0.12, need_target=True,
                      hint="极近/出画：0.3~0.5m，靶纸占满或部分出画"),
    "sup_blur":   dict(count=30, every=0.10, need_target=True,
                      hint="运动模糊：手持快速平移/转动，或相机运动"),
    "sup_clutter": dict(count=40, every=0.15, need_target=False,
                      hint="杂乱背景/负样本：相似圆形物、屏幕显示靶、反光、无靶"),
    "sup_green":  dict(count=30, every=0.12, need_target=True,
                      hint="绿色底板版本：换板拍不同角度/光照"),
    "sup_robot":  dict(count=30, every=0.05, need_target=True,
                      hint="机器人/振动：靶固定，相机相对运动"),
}

ORDER = ["pos_far", "pos_mid", "pos_near", "pos_angle", "pos_edge",
         "pos_light", "pos_occl", "pos_blur", "pos_robot", "static", "neg",
         "sup_lowlight", "sup_backlight", "sup_tilt", "sup_far", "sup_occl",
         "sup_near", "sup_blur", "sup_clutter", "sup_green", "sup_robot"]


# ------------------------------------------------------------------ 检测
def find_bullseye(frame: np.ndarray, min_r: int = 12, max_r: int = 180,
                  param2: int = 22) -> Optional[Tuple[float, float, float]]:
    """返回最像同心圆标靶的 (cx, cy, r)；没有返回 None。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
        param1=110, param2=param2, minRadius=min_r, maxRadius=max_r)
    if circles is None:
        return None
    circles = np.round(circles[0]).astype(float)
    # 按半径从大到小聚成同心组
    groups = []
    for x, y, r in sorted(circles, key=lambda c: -c[2]):
        for g in groups:
            if np.hypot(x - g["cx"], y - g["cy"]) <= max(10.0, 0.25 * max(r, g["r"])):
                g["xs"].append(x)
                g["ys"].append(y)
                g["rs"].append(r)
                g["cx"] = float(np.mean(g["xs"]))
                g["cy"] = float(np.mean(g["ys"]))
                g["r"] = max(g["r"], r)
                break
        else:
            groups.append({"xs": [x], "ys": [y], "rs": [r],
                           "cx": float(x), "cy": float(y), "r": float(r)})
    # 至少两个半径有区分度的圆才算“同心靶”
    good = []
    for g in groups:
        rs = sorted(g["rs"], reverse=True)
        if len(rs) >= 2 and rs[0] >= 1.25 * rs[-1]:
            good.append((g["cx"], g["cy"], rs[0]))
    if not good:
        return None
    return max(good, key=lambda t: t[2])


def ratio_of_frame(r: float, w: int, h: int) -> float:
    """标靶直径占画面短边的比例。"""
    return 2.0 * r / float(min(w, h))


# ------------------------------------------------------------------ 采集
class Collector:
    def __init__(self, args):
        self.args = args
        self.out_root = Path(args.out)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.stream = K230Stream(
            cfg={"w": args.width, "h": args.height, "q": args.q,
                 "fmt": "rgb565", "fps": 0, "stats": 0},
            auto_restart=True, log=lambda m: None)
        self.stats = {}
        self.motion = args.motion

    # ---- 基础
    def start(self):
        print("[*] 连接 K230 并启动摄像头…")
        self.stream.start(timeout=30)
        if not self.stream.hello:
            raise SystemExit("板端无响应：%s" % (self.stream.last_error,))
        print("[*] 已连接：%sx%s" %
              (self.stream.hello.get("w"), self.stream.hello.get("h")))

    def stop(self):
        try:
            self.stream.stop()
        except Exception:
            pass

    def grab_one(self, last_id: int, timeout: float = 3.0):
        jpeg, idx = self.stream.wait_frame(last_id, timeout=timeout)
        if jpeg is None:
            return None, idx, None
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        return jpeg, idx, frame

    # ---- 阶段
    def check_view(self, need_target: bool, min_ratio: float, max_ratio: float,
                   timeout: float = 12.0) -> Tuple[bool, str]:
        """等画面满足条件；返回 (ok, 描述)。"""
        last = 0
        t0 = time.time()
        while time.time() - t0 < timeout:
            jpeg, idx, frame = self.grab_one(last, timeout=2.0)
            if frame is None:
                continue
            last = idx
            h, w = frame.shape[:2]
            hit = find_bullseye(frame)
            if not need_target:
                if hit is None or ratio_of_frame(hit[2], w, h) < 0.03:
                    return True, "画面里没有明显标靶，可以开抓"
                return False, "检测到标靶（r=%.0f），负样本阶段请移走标靶" % hit[2]
            if hit is None:
                msg = "没检测到标靶：请让同心圆完整出现在画面里"
            else:
                ratio = ratio_of_frame(hit[2], w, h)
                if ratio < min_ratio:
                    msg = "标靶太小（占画面 %.0f%%）：请靠近一些" % (ratio * 100)
                elif ratio > max_ratio:
                    msg = "标靶太大（占画面 %.0f%%）：请离远一些" % (ratio * 100)
                else:
                    return True, "标靶 OK：中心 (%.0f,%.0f) 半径 %.0f，占画面 %.0f%%" % (
                        hit[0], hit[1], hit[2], ratio * 100)
            print("    %s" % msg)
            time.sleep(0.6)
        return False, "等待超时"

    def run_stage(self, name: str, cfg: dict, count: int, every: float):
        out = self.out_root / name
        out.mkdir(parents=True, exist_ok=True)
        prev = out / "_preview"
        prev.mkdir(exist_ok=True)

        print("\n" + "=" * 70)
        print(" 阶段 %-10s  %d 张" % (name, count))
        print(" 请这样摆：%s" % cfg["hint"])
        print(" (按 s 跳过本阶段，q 退出全部；倒计时 5 秒后开始检查画面)")
        print("=" * 70)
        for i in range(5, 0, -1):
            print("   %d..." % i, end="", flush=True)
            time.sleep(0.6)
        print()

        # 画面检查只做提示，不阻塞（Hough 在杂乱场景不可靠，最终由人/Agent 看预览确认）
        if cfg["need_target"]:
            lo, hi = (0.06, 0.20) if name == "pos_far" else (
                (0.15, 0.40) if name == "pos_mid" else (0.28, 0.60))
            ok, desc = self.check_view(True, lo, hi, timeout=6.0)
            print("   -> %s（仅提示，照常采集）" % desc)
        else:
            ok, desc = self.check_view(False, 0.0, 1.0, timeout=6.0)
            print("   -> %s（仅提示，照常采集）" % desc)

        saved = 0
        last = 0
        t0 = time.time()
        preview_saved = 0
        prev_small = None
        skipped = 0
        try:
            while saved < count:
                jpeg, idx, frame = self.grab_one(last, timeout=3.0)
                if jpeg is None:
                    print("    [!] 3 秒没收到新帧，重试")
                    continue
                last = idx

                # 只保存“画面确实在变”的帧，避免几十张完全重复
                small = cv2.resize(frame, (160, 120))
                if self.motion > 0 and prev_small is not None:
                    diff = float(np.mean(cv2.absdiff(
                        cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY))))
                    if diff < self.motion:
                        skipped += 1
                        if skipped % 20 == 0:
                            print("    (画面几乎没动，跳过 %d 帧；请移动标靶/相机)" % skipped)
                        continue
                prev_small = small

                path = out / ("frame_%06d.jpg" % saved)
                path.write_bytes(jpeg)
                saved += 1

                hit = find_bullseye(frame)
                if preview_saved < 3 and (saved % max(1, count // 3) == 0):
                    vis = frame.copy()
                    if hit:
                        cv2.circle(vis, (int(hit[0]), int(hit[1])), int(hit[2]),
                                   (0, 255, 0), 2)
                    cv2.imwrite(str(prev / ("check_%02d.jpg" % preview_saved)), vis,
                                [cv2.IMWRITE_JPEG_QUALITY, 85])
                    preview_saved += 1

                if saved % 10 == 0 or saved == count:
                    dt = time.time() - t0
                    extra = "" if hit is None else "  r=%.0f" % hit[2]
                    print("    %3d/%3d  %.1f 张/s%s" %
                          (saved, count, saved / max(dt, 1e-6), extra))
                if every > 0:
                    time.sleep(every)
        except KeyboardInterrupt:
            print("\n    [*] Ctrl-C，本阶段已保存 %d 张" % saved)

        meta = {
            "stage": name,
            "count": saved,
            "cfg": {"w": self.args.width, "h": self.args.height, "q": self.args.q},
            "hint": cfg["hint"],
            "started": round(t0, 3),
            "seconds": round(time.time() - t0, 1),
        }
        (out / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self.stats[name] = saved
        print("    [✓] %s：%d 张 -> %s" % (name, saved, out))
        return True

    def _ask_skip(self):
        """非交互环境返回 s；有终端时读一个键。"""
        if not sys.stdin.isatty():
            return "s"
        print("    [s] 跳过本阶段  [q] 退出  [回车] 重新检查")
        try:
            ans = input("    > ").strip().lower()
        except EOFError:
            return "s"
        if ans == "q":
            return "q"
        if ans == "s":
            return "s"
        return "r"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="K230 标靶数据集分阶段自动采集")
    ap.add_argument("--out", default="captures")
    ap.add_argument("--stage", default=None, help="只跑某一个阶段")
    ap.add_argument("--stages", default=None, help="逗号分隔，按顺序跑")
    ap.add_argument("--count", type=int, default=0, help="覆盖本阶段张数；0=用默认")
    ap.add_argument("--every", type=float, default=-1, help="覆盖帧间隔秒数")
    ap.add_argument("--w", "--width", dest="width", type=int, default=640)
    ap.add_argument("--h", "--height", dest="height", type=int, default=480)
    ap.add_argument("--q", type=int, default=70)
    ap.add_argument("--motion", type=float, default=6.0,
                    help="只在画面变化超过该阈值(平均绝对差)时才保存；0=关闭")
    args = ap.parse_args(argv)

    if args.stage:
        stages = [args.stage]
    elif args.stages:
        stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    else:
        stages = list(ORDER)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        print("未知阶段：%s\n可用：%s" % (unknown, ", ".join(ORDER)))
        return 2
    if not find_port():
        print("没有找到 K230 串口")
        return 2

    col = Collector(args)
    try:
        col.start()
        for name in stages:
            cfg = STAGES[name]
            count = args.count or cfg["count"]
            every = cfg["every"] if args.every < 0 else args.every
            cont = col.run_stage(name, cfg, count, every)
            if not cont:
                break
    finally:
        col.stop()

    print("\n" + "=" * 70)
    print(" 采集汇总")
    for k in stages:
        if k in col.stats:
            print("   %-10s %4d 张" % (k, col.stats[k]))
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
