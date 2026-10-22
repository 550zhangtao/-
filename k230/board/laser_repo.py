# -*- coding: utf-8 -*-
"""
laser_repo.py —— 完全照搬 zhouzhou-code/elec_vision_k230 的激光检测逻辑
=====================================================================

来源：https://github.com/zhouzhou-code/elec_vision_k230  (main.py)

原仓库相关代码（未做任何修改，逐行对应）：

    threshold_list = [30000, 80,240, 5000, 1, 9,100, 15,98, -13,77, 1]
    default_red_thres_list = (9, 100,    # Lmin, Lmax
                              15, 98,    # Amin, Amax
                              -13, 77)   # Bmin, Bmax 红色激光
    roi_w = 640 // 2 ; roi_h = 480 // 2
    roi_x = (640 - roi_w)//2 ; roi_y = (480 - roi_h)//2
    new_roi = (roi_x, roi_y, roi_w, roi_h)

    red_blobs = img.find_blobs([default_red_thres_list], False, roi=new_roi,
                               x_stride=2, y_stride=2, pixels_threshold=1,
                               merge=True, margin=True)
    if red_blobs:
        max_red = max(red_blobs, key=lambda b: (b.area(), -b.y()))
        red_coords = (max_red.cx(), max_red.cy())

本文件只把上述逻辑包成一个可调用对象，**没有加任何尺寸门限 / ROI 跟随 /
卡尔曼 / 跳变抑制**——目的就是看原版效果。
"""

import time

VERSION = "repo-port-1"

# 原仓库常量
RED_LAB = (9, 100, 15, 98, -13, 77)
LCD_W, LCD_H = 640, 480


class LaserRepo:
    def __init__(self, roi_scale=1.0):
        c_x, c_y = LCD_W // 2, LCD_H // 2
        w = int(LCD_W // 2 * roi_scale)
        h = int(LCD_H // 2 * roi_scale)
        self.roi = (c_x - w // 2, c_y - h // 2, w, h)
        self.last = None
        self.last_ms = 0
        self.nblob = 0
        self.blobs = []

    def sample(self, img):
        t0 = time.ticks_ms()
        try:
            blobs = img.find_blobs([RED_LAB], False, roi=self.roi,
                                   x_stride=2, y_stride=2,
                                   pixels_threshold=1, merge=True, margin=True)
        except Exception:
            blobs = None
        self.last_ms = time.ticks_diff(time.ticks_ms(), t0)
        self.blobs = blobs or []
        self.nblob = len(self.blobs)
        if not self.blobs:
            self.last = None
            return None
        b = max(self.blobs, key=lambda z: (z.area(), -z.y()))
        self.last = (b.cx(), b.cy())
        return {"x": b.cx(), "y": b.cy(), "area": b.area(),
                "w": b.w(), "h": b.h(), "n": b.pixels(),
                "raw": self.nblob, "ms": self.last_ms}
