#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_laser_aim.py —— `board/laser_aim.py` 的 PC 单测（不需要板子）
=================================================================

用法：
    cd k230 && python3 tools/test_laser_aim.py

覆盖：
    1. RGB565 微窗质心能找到真实光斑（旧样本 frame_0010.jpg，真值 ~408,278）；
    2. 标定偏 60px 时只回退到标定位置，不误锁到别处；
    3. 没有激光的帧（inpaint 掉光斑）不会报 measured；
    4. `c + k·s` 距离补偿的符号/线性；
    5. 标定文件读写（旧两字段格式也能读）。

注意：这里用 numpy 把 BGR 转成 RGB565 再喂给 LaserAim，和板端
`img.bytearray()` 的字节序一致（小端），所以才叫"同源"测试。
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "board"))
import laser_aim as LA  # noqa: E402

SAMPLE = ROOT / "tools" / "laser_samples" / "wall_on_20261021_203504" / "raw" / "frame_0010.jpg"
SPOT_320 = (206.0, 139.0)          # 320x240 图上的人工核对位置
SPOT_640 = (SPOT_320[0] * 2.0, SPOT_320[1] * 2.0)


class FakeImg:
    """最小 image 接口：width/height/bytearray，RGB565 小端。"""

    def __init__(self, bgr: np.ndarray):
        self.h, self.w = bgr.shape[:2]
        b = bgr[:, :, 0].astype(np.uint16) >> 3
        g = bgr[:, :, 1].astype(np.uint16) >> 2
        r = bgr[:, :, 2].astype(np.uint16) >> 3
        v = (r << 11) | (g << 5) | b
        self.ba = bytearray(v.astype("<u2").tobytes())

    def width(self):
        return self.w

    def height(self):
        return self.h

    def bytearray(self):
        return self.ba


def load_big(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit("读不到样本：%s" % path)
    return cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)


def run(name: str, fn) -> bool:
    try:
        fn()
        print("PASS  %s" % name)
        return True
    except AssertionError as e:
        print("FAIL  %s: %s" % (name, e))
        return False


def test_hit():
    img = FakeImg(load_big(SAMPLE))
    la = LA.LaserAim({"rad": 12, "budget_ms": 20})
    la.set_calib(SPOT_640[0], SPOT_640[1])
    r = la.sample(img)
    assert r is not None, "sample 返回 None"
    assert r["src"] == "measured", "没找到光斑：%s" % r
    assert abs(r["x"] - SPOT_640[0]) < 4.0, "x 偏了：%.1f" % (r["x"] - SPOT_640[0])
    assert abs(r["y"] - SPOT_640[1]) < 4.0, "y 偏了：%.1f" % (r["y"] - SPOT_640[1])
    print("      measured=(%.1f,%.1f) n=%d conf=%.2f %dms"
          % (r["x"], r["y"], r["n"], r["conf"], r["ms"]))


def test_wrong_calib_no_lock():
    img = FakeImg(load_big(SAMPLE))
    la = LA.LaserAim({"rad": 12, "budget_ms": 20})
    la.set_calib(SPOT_640[0] - 60, SPOT_640[1])     # 偏 60px，光斑在窗外
    r = la.sample(img)
    assert r is not None and r["src"] == "calib", "偏 60px 时不应报 measured：%s" % r
    for _ in range(12):
        r = la.sample(img)
    assert r["status"] == "lost", "连续丢失后应 status=lost：%s" % r
    print("      fallback=(%.1f,%.1f) status=%s" % (r["x"], r["y"], r["status"]))


def test_off_frame_no_false_lock():
    img = load_big(SAMPLE)
    off = img.copy()
    mask = np.zeros(off.shape[:2], np.uint8)
    cv2.circle(mask, (int(SPOT_640[0]), int(SPOT_640[1])), 34, 255, -1)
    off = cv2.inpaint(off, mask, 7, cv2.INPAINT_TELEA)
    la = LA.LaserAim({"rad": 12, "budget_ms": 20})
    la.set_calib(SPOT_640[0], SPOT_640[1])
    for _ in range(3):
        r = la.sample(FakeImg(off))
    assert r["src"] == "calib", "关激光的帧误报 measured：%s" % r
    print("      off frame -> src=%s status=%s" % (r["src"], r["status"]))


def test_distance_model():
    la = LA.LaserAim()
    la.set_calib(400.0, 240.0, kx=0.5, ky=-0.25, ref=40.0)
    x, y = la.predict(80.0)         # (s-ref)=40 -> x+20, y-10
    assert abs(x - 420.0) < 1e-6 and abs(y - 230.0) < 1e-6, (x, y)
    x, y = la.predict(40.0)
    assert abs(x - 400.0) < 1e-6 and abs(y - 240.0) < 1e-6, (x, y)
    print("      predict(80)=(%.1f,%.1f) predict(40)=(%.1f,%.1f)" % (x, y, x, y))


def test_tracker_cv():
    """α-β 常速滤波：匀速 3px/帧（66ms）下，50ms 前瞻误差应 < 2px。"""
    tr = LA._Tracker(0.45, 0.10, 2000.0)
    for i in range(20):
        tr.update(100.0 + 3.0 * i, 200.0 + 1.5 * i, i * 66)
    t_pred = 19 * 66 + 116            # 最后一帧 + 50ms 前瞻（外加一帧）
    px, py = tr.predict(t_pred)
    tx = 100.0 + 3.0 * (t_pred / 66.0)
    ty = 200.0 + 1.5 * (t_pred / 66.0)
    assert abs(px - tx) < 2.0, "x 预测误差 %.2f" % (px - tx)
    assert abs(py - ty) < 2.0, "y 预测误差 %.2f" % (py - ty)
    print("      cv predict=(%.1f,%.1f) truth=(%.1f,%.1f) v=(%.1f,%.1f)"
          % (px, py, tx, ty, tr.vx, tr.vy))


def test_coast():
    """命中 2 帧后光斑消失：应先 coast 外推，长时间丢失才回标定/lost。"""
    on = load_big(SAMPLE)
    off = on.copy()
    mask = np.zeros(off.shape[:2], np.uint8)
    cv2.circle(mask, (int(SPOT_640[0]), int(SPOT_640[1])), 34, 255, -1)
    off = cv2.inpaint(off, mask, 7, cv2.INPAINT_TELEA)
    la = LA.LaserAim({"rad": 12, "budget_ms": 20})
    la.set_calib(SPOT_640[0], SPOT_640[1])
    for _ in range(2):
        r = la.sample(FakeImg(on))
        assert r["mode"] == "roi", r
    r = la.sample(FakeImg(off))
    assert r["mode"] == "coast", "第一帧丢失应 coast：%s" % r
    for _ in range(12):
        r = la.sample(FakeImg(off))
    assert r["mode"] != "roi", "长时间丢失不应还报 roi：%s" % r
    assert r["status"] in ("calib", "lost"), r
    print("      coast -> %s/%s (conf %.2f)" % (r["mode"], r["status"], r["conf"]))


def test_rezero():
    """光斑稳定时，标定先验应慢慢被拉向实测（在线 re-zero）。"""
    img = FakeImg(load_big(SAMPLE))
    # 测试里把 rezero 速度阈值放宽、速率加快，专门验证"机制"能收敛；
    # 板端默认是 rezero_speed=6 / rate=0.03（更慢、更保守）。
    la = LA.LaserAim({"rad": 12, "budget_ms": 20,
                      "rezero_speed": 50.0, "rezero_rate": 0.05})
    la.set_calib(SPOT_640[0] + 5.0, SPOT_640[1])     # 偏 5px，仍在微窗内
    d0 = None
    for i in range(30):
        time.sleep(0.02)          # 模拟板端 ~50fps 的帧间隔，避免 dt 下限放大速度噪声
        r = la.sample(img)
        assert r is not None and r["mode"] == "roi", r
        if i == 0:
            d0 = abs(la.calib[0] - r["x"]) + abs(la.calib[1] - r["y"])
    d1 = abs(la.calib[0] - r["x"]) + abs(la.calib[1] - r["y"])
    assert d1 < d0 * 0.7, "re-zero 没把标定拉近：%.2f -> %.2f" % (d0, d1)
    print("      calib->measured dist %.2f -> %.2f, calib=(%.1f,%.1f)"
          % (d0, d1, la.calib[0], la.calib[1]))


def test_calib_io():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "laser_calib.json"
        p.write_text('{"x": 312.5, "y": 230.1}')      # 旧两字段格式
        la = LA.LaserAim()
        assert la.load(str(p)), "旧格式读取失败"
        assert la.calib[0] == 312.5 and la.calib[1] == 230.1
        assert la.save(str(p))
        d = json.loads(p.read_text())
        assert abs(d["x"] - 312.5) < 1e-6 and abs(d["y"] - 230.1) < 1e-6
        print("      load/save ok: %s" % d)


def main() -> int:
    if not SAMPLE.exists():
        print("缺样本：%s" % SAMPLE)
        return 2
    print("样本真值（640x480）≈ (%.0f, %.0f)" % SPOT_640)
    ok = True
    for name, fn in (
        ("hit", test_hit),
        ("wrong-calib", test_wrong_calib_no_lock),
        ("off-frame", test_off_frame_no_false_lock),
        ("distance-model", test_distance_model),
        ("tracker-cv", test_tracker_cv),
        ("coast", test_coast),
        ("rezero", test_rezero),
        ("calib-io", test_calib_io),
    ):
        ok = run(name, fn) and ok
    print("\n%s" % ("ALL PASS" if ok else "SOME FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
