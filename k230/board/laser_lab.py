# -*- coding: utf-8 -*-
"""
laser_lab.py —— 移植 zhouzhou-code/elec_vision_k230 的激光检测核心
=================================================================

原仓库的激光部分是：
    RED = (9,100, 15,98, -13,77)          # LAB: Lmin,Lmax,Amin,Amax,Bmin,Bmax
    blobs = img.find_blobs([RED], False, roi=roi, x_stride=2, y_stride=2,
                           pixels_threshold=1, merge=True, margin=True)
    max_red = max(blobs, key=lambda b: (b.area(), -b.y()))

我们实测后在此基础上补了三处（否则在本场景会锁红袋子）：
  1. 尺寸上限（max_side / max_area）——原版没有，会选中 81x118 的红袋子；
  2. ROI 跟随上一帧（原版是固定居中 ROI，激光一移动就出窗）；
  3. 跳变抑制 + 每轴卡尔曼（降低坐标抖动）。

阈值保持原仓库默认值，可通过 cfg 覆盖。
"""

import time

VERSION = "2026-10-23.1-lab"

CFG = {
    # 原仓库的红激光 LAB 阈值
    "lab": (9, 100, 15, 98, -13, 77),
    "x_stride": 2,
    "y_stride": 2,
    "pixels_threshold": 1,     # 原版就是 1（很宽松）
    "merge": 1,                # 原版 merge=True
    # 我们补的约束
    "max_side": 40,
    "max_area": 600,
    "min_area": 20,
    "roi_size": 120,
    "track": 1,
    "lost_reset": 30,
    "max_step": 20.0,
    "step_confirm": 3,
    "kf_on": 1,
    "kf_dt": 0.020,
    "kf_r": 4.0,
    "kf_qp": 0.5,
    "kf_qv": 200.0,
    "start_x": 320.0,
    "start_y": 240.0,
}


def _now_ms():
    try:
        return time.ticks_ms()
    except AttributeError:
        return int(time.monotonic() * 1000)


class LaserLab:
    """用 CanMV 原生 find_blobs(LAB) 做检测，外层套 ROI 跟随 + 抗跳变 + 卡尔曼。"""

    def __init__(self, cfg=None):
        self.cfg = dict(CFG)
        if cfg:
            self.cfg.update(cfg)
        self.roi = (0, 0, int(self.cfg["roi_size"]), int(self.cfg["roi_size"]))
        self.last = None
        self.last_x = float(self.cfg.get("start_x", 320.0))
        self.last_y = float(self.cfg.get("start_y", 240.0))
        self.roi_c = (self.last_x, self.last_y)
        self.lost_run = 0
        self.jump_n = 0
        self.kf_x = None
        self.kf_y = None
        self.last_ms = 0
        self.raw_blobs = 0
        self.blobs = []
        self.frame = 0
        self.sel = "-"
        self.mode = "lab"

    def _kf_step(self, cx, cy):
        c = self.cfg
        dt = float(c.get("kf_dt", 0.02))
        r = float(c.get("kf_r", 4.0))
        qp = float(c.get("kf_qp", 0.5))
        qv = float(c.get("kf_qv", 200.0))
        if self.kf_x is None:
            self.kf_x = [float(cx), 0.0, r, 0.0, qv]
            self.kf_y = [float(cy), 0.0, r, 0.0, qv]
            return cx, cy
        for st in (self.kf_x, self.kf_y):
            st[0] += st[1] * dt
            p00 = st[2] + 2.0 * dt * st[3] + dt * dt * st[4] + qp * dt
            p01 = st[3] + dt * st[4]
            st[4] = st[4] + qv * dt
            st[2] = p00
            st[3] = p01
        for st, z in ((self.kf_x, cx), (self.kf_y, cy)):
            S = st[2] + r
            if S <= 1e-9:
                continue
            k0 = st[2] / S
            k1 = st[3] / S
            y = float(z) - st[0]
            st[0] += k0 * y
            st[1] += k1 * y
            st[2] = (1.0 - k0) * st[2]
            st[3] = (1.0 - k0) * st[3]
            st[4] = st[4] - k1 * st[3]
        return self.kf_x[0], self.kf_y[0]

    def sample(self, img):
        """img: RGB565/RGB888 的 Image（显示通道）。返回 dict 或 None。"""
        self.frame += 1
        c = self.cfg
        # ---- ROI 跟随 ----
        if int(c.get("track", 1)):
            sz = int(c.get("roi_size", 120))
            x0 = int(self.last_x - sz * 0.5)
            y0 = int(self.last_y - sz * 0.5)
            if x0 < 0:
                x0 = 0
            if y0 < 0:
                y0 = 0
            if x0 + sz > 640:
                x0 = 640 - sz
            if y0 + sz > 480:
                y0 = 480 - sz
            self.roi = (x0, y0, sz, sz)
            self.roi_c = (x0 + sz * 0.5, y0 + sz * 0.5)
        roi = self.roi
        x0, y0, rw, rh = roi

        t0 = _now_ms()
        try:
            blobs = img.find_blobs([tuple(int(v) for v in c["lab"])], False,
                                   roi=roi,
                                   x_stride=int(c["x_stride"]),
                                   y_stride=int(c["y_stride"]),
                                   pixels_threshold=int(c["pixels_threshold"]),
                                   merge=bool(int(c["merge"])), margin=True)
        except Exception:
            blobs = None
        self.last_ms = time.ticks_diff(_now_ms(), t0) if hasattr(time, "ticks_diff") else 0
        self.blobs = blobs or []
        self.raw_blobs = len(self.blobs)

        mx_side = int(c.get("max_side", 40))
        mx_area = int(c.get("max_area", 600))
        mn_area = int(c.get("min_area", 20))
        best = None
        for b in self.blobs:
            w = int(b.w())
            h = int(b.h())
            a = int(b.pixels())
            if w > mx_side or h > mx_side:
                continue
            if a < mn_area or a > mx_area:
                continue
            if best is None or int(b.area()) > int(best.area()):
                best = b
        if best is None:
            self.lost_run += 1
            if self.lost_run > int(c.get("lost_reset", 30)):
                self.last_x = float(c.get("start_x", 320.0))
                self.last_y = float(c.get("start_y", 240.0))
                self.lost_run = 0
            self.last = None
            self.sel = "miss"
            return None

        cx = float(best.cx())
        cy = float(best.cy())
        self.sel = "lab"

        # ---- 跳变抑制 ----
        if self.last is not None:
            ms_px = float(c.get("max_step", 20.0))
            ddx = cx - self.last[0]
            ddy = cy - self.last[1]
            if ddx * ddx + ddy * ddy > ms_px * ms_px:
                self.jump_n += 1
                if self.jump_n < int(c.get("step_confirm", 3)):
                    cx = self.last[0]
                    cy = self.last[1]
                    self.sel = "hold"
                else:
                    self.jump_n = 0
            else:
                self.jump_n = 0

        # ---- 卡尔曼 ----
        if int(c.get("kf_on", 0)):
            try:
                cx, cy = self._kf_step(cx, cy)
            except Exception:
                pass

        self.last = (cx, cy)
        self.last_x = cx
        self.last_y = cy
        self.lost_run = 0
        return {"x": cx, "y": cy, "n": int(best.pixels()),
                "w": int(best.w()), "h": int(best.h()),
                "raw": self.raw_blobs, "ms": self.last_ms, "sel": self.sel}
