# -*- coding: utf-8 -*-
"""
laser_repo.py —— 照搬 zhouzhou-code/elec_vision_k230 的激光检测，
                 并补上可配置的尺寸/面积门限。

来源：https://github.com/zhouzhou-code/elec_vision_k230  (main.py)

原版逻辑（判据未改）：

    red_blobs = img.find_blobs([default_red_thres_list], False, roi=...,
                               x_stride=2, y_stride=2, pixels_threshold=1,
                               merge=True, margin=True)
    if red_blobs:
        max_red = max(red_blobs, key=lambda b: (b.area(), -b.y()))

原版的问题：直接"取面积最大"，红袋子 / 红电线 / 橙色绳 / 手 都比
8×10 的激光点大，于是必然锁到杂物上；而且 `merge=True` 还会把
"红杂物 + 激光"并成一个大块。

本版（VERSION = repo-port-2-gated）只做两件事：
  1) 阈值 / 门限 / ROI 全部做成可传参（`LASER_CFG` 不再失效）；
  2) 在"取最大"之前先过滤掉明显不可能是激光的块
     （最小/最大面积、单边上限、长宽比、可选填充率）。

判据本身仍是原版"红色 + 面积最大"，**没有**引入颜色掩码 / 多帧 / 卡尔曼。
所有门限默认关闭（0），因此 `LaserRepo()` 无参调用 = 原版行为，
scripts/accept_laser_static.py 等旧调用不受影响。
"""

import time

VERSION = "repo-port-2-gated"

# 原仓库常量
RED_LAB = (9, 100, 15, 98, -13, 77)
LCD_W, LCD_H = 640, 480

# 默认参数；全部可通过 LaserRepo(**kw) / yolo_preview_ide.py 的 LASER_CFG 覆盖。
# 门限为 0 表示关闭（保持原版"取面积最大"的行为）。
DEFAULTS = {
    "roi_scale": 1.0,
    "roi": None,             # (x, y, w, h)；None -> 以画面中心 * roi_scale
    "threshold": None,       # None -> RED_LAB；也可传 [(lo...), (lo...)] 多区间
    "x_stride": 2,
    "y_stride": 2,
    "pixels_threshold": 1,
    "merge": True,
    "margin": 1,
    # ---- 门限（0 / 0.0 = 关闭）----
    "min_area": 0,           # bbox 面积下限（w*h）
    "max_area": 0,           # bbox 面积上限
    "max_side": 0,           # 单边上限
    "max_aspect": 0.0,       # 长宽比上限（max(w,h)/min(w,h)）
    "min_fill": 0.0,         # 填充率下限（pixels/(w*h)），排细长/稀疏块
}


def _as_threshold_list(thr):
    """find_blobs 要的是 [(lo..hi), ...]；兼容用户传单个 tuple。"""
    if thr is None:
        thr = RED_LAB
    if isinstance(thr, (list, tuple)) and thr and isinstance(thr[0], (list, tuple)):
        return list(thr)
    return [thr]


class LaserRepo:
    def __init__(self, **kw):
        cfg = dict(DEFAULTS)
        cfg.update(kw or {})
        self.cfg = cfg
        self.thresholds = _as_threshold_list(cfg.get("threshold"))
        self.x_stride = int(cfg["x_stride"])
        self.y_stride = int(cfg["y_stride"])
        self.pixels_threshold = int(cfg["pixels_threshold"])
        self.merge = bool(cfg["merge"])
        self.margin = int(cfg["margin"])
        self.min_area = int(cfg["min_area"] or 0)
        self.max_area = int(cfg["max_area"] or 0)
        self.max_side = int(cfg["max_side"] or 0)
        self.max_aspect = float(cfg["max_aspect"] or 0.0)
        self.min_fill = float(cfg["min_fill"] or 0.0)
        self.roi = self._make_roi(cfg)
        self.last = None
        self.last_ms = 0
        self.nblob = 0          # 通过门限的候选数
        self.raw_blobs = 0      # find_blobs 原始候选数
        self.blobs = []

    def _make_roi(self, cfg):
        r = cfg.get("roi")
        if r:
            return tuple(int(v) for v in r)
        sc = float(cfg.get("roi_scale", 1.0) or 1.0)
        c_x, c_y = LCD_W // 2, LCD_H // 2
        w = int(LCD_W // 2 * sc)
        h = int(LCD_H // 2 * sc)
        return (c_x - w // 2, c_y - h // 2, w, h)

    def _passes(self, b):
        """尺寸/面积/形状门限；返回 True 表示可能是激光。"""
        try:
            w, h = int(b.w()), int(b.h())
        except Exception:
            return False
        area = w * h
        if self.min_area and area < self.min_area:
            return False
        if self.max_area and area > self.max_area:
            return False
        if self.max_side and max(w, h) > self.max_side:
            return False
        if self.max_aspect:
            short = min(w, h)
            if short <= 0 or (max(w, h) / float(short)) > self.max_aspect:
                return False
        if self.min_fill:
            try:
                fill = float(b.pixels()) / float(area) if area else 0.0
            except Exception:
                fill = 0.0
            if fill < self.min_fill:
                return False
        return True

    def sample(self, img):
        t0 = time.ticks_ms()
        try:
            blobs = img.find_blobs(self.thresholds, False, roi=self.roi,
                                   x_stride=self.x_stride,
                                   y_stride=self.y_stride,
                                   pixels_threshold=self.pixels_threshold,
                                   merge=self.merge, margin=self.margin)
        except Exception:
            blobs = None
        self.last_ms = time.ticks_diff(time.ticks_ms(), t0)
        self.blobs = blobs or []
        self.raw_blobs = len(self.blobs)

        kept = [b for b in self.blobs if self._passes(b)]
        self.nblob = len(kept)
        if not kept:
            self.last = None
            return None

        b = max(kept, key=lambda z: (z.area(), -z.y()))
        self.last = (b.cx(), b.cy())
        return {"x": b.cx(), "y": b.cy(), "area": b.area(),
                "w": b.w(), "h": b.h(), "n": b.pixels(),
                "raw": self.raw_blobs, "kept": self.nblob,
                "ms": self.last_ms}
