#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_spot_tracker.py —— 光斑 Kalman 预测器/门控单元测试（PC，不连板）

覆盖：
  1. 匀速运动：收敛后预测误差 < 门控半径；
  2. 连续丢帧 5 帧：coast 预测误差可控，门控不无限放大；
  3. 离群点：100px 跳变被 gate_ok 拒绝；
  4. 静止光斑：速度收敛到 ~0，门控半径保持小。

用法：
    python3 tools/test_spot_tracker.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "board"))

import laser_spot  # noqa: E402

STEP_MS = 50


def test_constant_velocity() -> bool:
    k = laser_spot.SpotKalman(q=40.0, r=4.0, max_gate=48.0)
    k.reset(100.0, 100.0, 0)
    for i in range(1, 21):
        k.predict(i * STEP_MS)
        k.update(100.0 + 3.0 * i, 100.0 + 1.0 * i)
    # 再预测 5 帧不更新（coast）
    px = py = 0.0
    for i in range(21, 26):
        px, py = k.predict(i * STEP_MS)
    truth_x = 100.0 + 3.0 * 25
    truth_y = 100.0 + 1.0 * 25
    err = ((px - truth_x) ** 2 + (py - truth_y) ** 2) ** 0.5
    print("  constant-v: pred=(%.1f,%.1f) truth=(%.1f,%.1f) err=%.1f gate=%.1f v=(%.1f,%.1f)"
          % (px, py, truth_x, truth_y, err, k.gate_radius(), k.vx, k.vy))
    return err < k.gate_radius() * 2.0


def test_dropout_coast() -> bool:
    k = laser_spot.SpotKalman(q=40.0, r=4.0, max_gate=48.0)
    k.reset(320.0, 240.0, 0)
    for i in range(1, 11):
        k.predict(i * STEP_MS)
        k.update(320.0 + 2.0 * i, 240.0 - 1.0 * i)
    px, py = k.predict(11 * STEP_MS)
    for i in range(12, 17):
        px, py = k.predict(i * STEP_MS)
    truth_x = 320.0 + 2.0 * 16
    truth_y = 240.0 - 1.0 * 16
    err = ((px - truth_x) ** 2 + (py - truth_y) ** 2) ** 0.5
    print("  5-frame coast: err=%.1f gate=%.1f (should stay bounded)"
          % (err, k.gate_radius()))
    return err < 20.0 and k.gate_radius() <= 48.0


def test_outlier_gate() -> bool:
    k = laser_spot.SpotKalman(q=40.0, r=4.0, max_gate=48.0)
    k.reset(100.0, 100.0, 0)
    for i in range(1, 6):
        k.predict(i * STEP_MS)
        k.update(100.0, 100.0)
    # 100px 跳变应被拒绝
    ok = k.gate_ok(200.0, 100.0)
    print("  outlier gate: accepted=%s (expect False) gate=%.1f" % (ok, k.gate_radius()))
    return not ok


def test_static() -> bool:
    k = laser_spot.SpotKalman(q=40.0, r=4.0, max_gate=48.0)
    k.reset(350.0, 290.0, 0)
    for i in range(1, 16):
        k.predict(i * STEP_MS)
        # 模拟 ±1.5px 噪声
        dx = 1.5 if i % 2 else -1.5
        k.update(350.0 + dx, 290.0 - dx)
    speed = (k.vx ** 2 + k.vy ** 2) ** 0.5
    print("  static: v=(%.2f,%.2f) speed=%.2f gate=%.1f"
          % (k.vx, k.vy, speed, k.gate_radius()))
    return speed < 20.0 and k.gate_radius() < 20.0


def main() -> int:
    tests = [("constant velocity", test_constant_velocity),
             ("dropout coast", test_dropout_coast),
             ("outlier gate", test_outlier_gate),
             ("static spot", test_static)]
    ok = True
    for name, fn in tests:
        print("[*] %s" % name)
        try:
            r = fn()
        except Exception as e:
            print("  EXCEPTION:", e)
            r = False
        print("    -> %s" % ("PASS" if r else "FAIL"))
        ok = ok and r
    print("\n%s" % ("ALL PASS" if ok else "SOME FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
