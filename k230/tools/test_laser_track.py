#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_laser_track.py —— 用仿真验证"延迟外推"相对"直接喂原始坐标"的收益。

场景与真机一致：
  * 目标在像面里匀速运动，每 2s 左右换向（模拟靶标晃动 / 云台自身运动）；
  * 观测周期 60ms、噪声 3px；
  * 感知-执行链路固定延迟 80ms：t 时刻拿到的是 t-80ms 的位置，
    t 时刻发出的指令要到 t+80ms 才生效。

对比两种喂给 PID 的坐标：
  raw   —— 直接把观测值送进去（现状）
  pred  —— LaserTracker 外推到"指令生效时刻"（本层新增）

用法：
    python3 tools/test_laser_track.py
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

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "board"))
from laser_track import LaserTracker  # noqa: E402

DELAY_MS = 80.0        # 感知-执行固定延迟（待实测标定）
PERIOD_MS = 60.0       # 识别周期（约 16fps）
NOISE_PX = 3.0         # 观测噪声
DURATION_S = 20.0
SEED = 7

CX, CY = 320.0, 240.0


def target_pos(t):
    """三角波：匀速段 + 瞬时换向，位置连续。"""
    ux = ((t + 0.0) % 2.0) / 2.0
    uy = ((t + 0.7) % 2.6) / 2.6
    x = CX + 80.0 * (4.0 * abs(ux - 0.5) - 1.0)   # |v| = 160 px/s
    y = CY + 45.0 * (4.0 * abs(uy - 0.5) - 1.0)   # |v| ≈ 69 px/s
    return x, y


def run():
    rng = random.Random(SEED)
    tr = LaserTracker(delay_ms=DELAY_MS, q=120.0, r=NOISE_PX ** 2)
    n = int(DURATION_S * 1000.0 / PERIOD_MS)
    raw_err, pred_err = [], []
    for i in range(n):
        t_obs = i * PERIOD_MS                       # 这帧的曝光时刻
        tx, ty = target_pos(t_obs / 1000.0)
        mx = tx + rng.gauss(0.0, NOISE_PX)
        my = ty + rng.gauss(0.0, NOISE_PX)
        tr.update(mx, my, int(t_obs))               # 观测时间戳 = 曝光时刻
        t_eff = t_obs + DELAY_MS                    # 这条指令真正生效的时刻
        x_eff, y_eff = target_pos(t_eff / 1000.0)
        raw_err.append(math.hypot(mx - x_eff, my - y_eff))
        p = tr.predict(int(t_obs))                  # 从曝光时刻外推 delay
        if p is not None:
            pred_err.append(math.hypot(p[0] - x_eff, p[1] - y_eff))

    def stats(name, errs):
        errs = sorted(errs)
        rms = math.sqrt(sum(e * e for e in errs) / len(errs))
        print("%-6s n=%4d  RMS %6.2f px   P95 %6.2f px   max %6.2f px"
              % (name, len(errs), rms, errs[int(len(errs) * 0.95)], errs[-1]))
        return rms

    print("延迟 %.0fms / 周期 %.0fms / 噪声 %.1fpx / %.0fs" % (
        DELAY_MS, PERIOD_MS, NOISE_PX, DURATION_S))
    r_raw = stats("raw", raw_err)
    r_pred = stats("pred", pred_err)
    print("RMS 降低 %.1f%%（%.2f -> %.2f px）" % (
        100.0 * (1.0 - r_pred / r_raw), r_raw, r_pred))
    v = tr.velocity(int((n - 1) * PERIOD_MS - DELAY_MS))
    if v:
        print("末帧速度估计 (%.0f, %.0f) px/s" % (v[0], v[1]))


if __name__ == "__main__":
    run()
