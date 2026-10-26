# -*- coding: utf-8 -*-
"""
laser_repo.py —— 移植 zhouzhou-code/elec_vision_k230 的激光检测逻辑
====================================================================

来源：https://github.com/zhouzhou-code/elec_vision_k230  (main.py)

原仓库相关代码（逐行对应）：

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

本文件在原逻辑上加了：

1. **形状自适应（方案 2）**：倾斜反光面把光斑拉成亮带时，整条带的面积质心
   不是入射点。先用 blob.elongation() / 外接框长宽比判断“是不是亮带”；是的话
   沿长轴采样找最亮端，再在最亮端 ±r 小窗做 V² 加权质心。
2. **PointKalman 平滑**：有测量 update、短暂丢失 predict 兜底、MAX_JUMP 限幅。

安全加固：所有形状/像素 API（elongation / major_axis_line / get_pixel）都做了
try/except；任一不可用就**回退到 b.cx()/b.cy()**，不会因为某个方法缺失而漏检。

关于 LAB 取值范围（CanMV image.find_blobs，RGB565 图；官方示例
examples/11-Feature-Detection/find_blobs.py 与
examples/09-Color-Tracking/single_color_code_tracking.py 均如此）：
    L: 0~100        （不是 0~255）
    A: -128~127     （+A = 红，-A = 绿）
    B: -128~127     （+B = 黄，-B = 蓝）
"""

import time

try:
    from point_kalman import PointKalman
except Exception:
    PointKalman = None

VERSION = "repo-port-5"

# 原仓库阈值（红色激光）：L 9~100 / A 15~98 / B -13~77
RED_LAB = (9, 100, 15, 98, -13, 77)

# 亮红激光（用户实测是红色激光）：L 下限抬到 60，把靶纸上的暗红十字/阴影排除掉；
# A 收窄到强红；B 允许略偏黄（倾斜反光面会把光斑往黄色拉一点）。
# 关键：暗红十字的 L 很低，会被 Lmin=60 直接挡掉，不再和激光抢 blob。
RED_LASER_LAB = (60, 100, 45, 127, -20, 80)

# ---- 形状自适应（方案 2）----
LONG_ELONG = 2.0        # blob.elongation() >= 此值 -> 视为亮带
LONG_AR = 2.5           # 外接框 长/宽 >= 此值 -> 视为亮带
AXIS_STEP = 6.0         # 沿长轴每 ~6px 采样一次
AXIS_MIN = 7            # 最少采样点
AXIS_MAX = 25           # 最多采样点
CORE_REFINE_R = 2       # 最亮端 ±r 小窗做 V² 加权质心（r=2 -> 5x5）
CORE_TIE_RATIO = 0.90    # 亮度 >= 90% 最亮的候选里，取离上一帧最近的

# 激光点更小、更容易被单个噪点带偏，MAX_JUMP 收得比靶心紧一点
LASER_MAX_JUMP = 30.0

# 原仓库常量
LCD_W, LCD_H = 640, 480


def _pix_bright(img, x, y):
    """取像素亮度（0~255）。失败返回 -1（调用方会回退到 cx/cy）。"""
    try:
        v = img.get_pixel(x, y, rgbtuple=True)
    except Exception:
        try:
            v = img.get_pixel(x, y)
        except Exception:
            return -1
    if v is None:
        return -1
    try:
        r, g, b = v
        return (r * 299 + g * 587 + b * 114) // 1000
    except Exception:
        try:
            return int(v)
        except Exception:
            return -1


def _weighted_centroid(img, cx, cy, r=CORE_REFINE_R):
    """在 (cx,cy) ±r 窗口内按亮度平方加权求质心。"""
    x0 = cx - r
    if x0 < 0:
        x0 = 0
    y0 = cy - r
    if y0 < 0:
        y0 = 0
    x1 = cx + r
    if x1 > img.width() - 1:
        x1 = img.width() - 1
    y1 = cy + r
    if y1 > img.height() - 1:
        y1 = img.height() - 1
    sx = sy = sw = 0.0
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            v = _pix_bright(img, x, y)
            if v <= 0:
                continue
            w = float(v) * float(v)
            sx += x * w
            sy += y * w
            sw += w
    if sw <= 0.0:
        return float(cx), float(cy)
    return sx / sw, sy / sw


def _axis_line(blob):
    """blob 长轴；major_axis_line 不可用时用外接框长边兜底。"""
    try:
        return blob.major_axis_line()
    except Exception:
        x, y, w, h = int(blob.x()), int(blob.y()), int(blob.w()), int(blob.h())
        cx = x + w // 2
        cy = y + h // 2
        if h >= w:
            return (cx, y, cx, y + h)
        return (x, cy, x + w, cy)


def _core_on_axis(img, blob, last):
    """亮带：沿长轴找最亮端，返回 (x, y) 或 None（None 表示调用方回退）。"""
    try:
        x1, y1, x2, y2 = _axis_line(blob)
    except Exception:
        return None
    n = int(((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) ** 0.5 / AXIS_STEP) + 3
    if n < AXIS_MIN:
        n = AXIS_MIN
    if n > AXIS_MAX:
        n = AXIS_MAX
    wmax = img.width() - 1
    hmax = img.height() - 1
    cands = []
    max_v = -1
    for i in range(n):
        t = float(i) / float(n - 1)
        x = int(round(x1 + (x2 - x1) * t))
        y = int(round(y1 + (y2 - y1) * t))
        if x < 0:
            x = 0
        elif x > wmax:
            x = wmax
        if y < 0:
            y = 0
        elif y > hmax:
            y = hmax
        v = _pix_bright(img, x, y)
        cands.append((x, y, v))
        if v > max_v:
            max_v = v
    if max_v <= 0:
        return None
    # 亮度接近最亮时，优先离上一帧最近的（整条都饱和时抑制乱跳）
    if last is None:
        best = None
        for x, y, v in cands:
            if v == max_v:
                best = (x, y)
                break
        if best is None:
            best = (cands[0][0], cands[0][1])
    else:
        lx, ly = last
        thr = max_v * CORE_TIE_RATIO
        best = None
        bd = 1e18
        for x, y, v in cands:
            if v < thr:
                continue
            d = (x - lx) * (x - lx) + (y - ly) * (y - ly)
            if d < bd:
                bd = d
                best = (x, y)
        if best is None:
            best = (cands[0][0], cands[0][1])
    return _weighted_centroid(img, best[0], best[1], CORE_REFINE_R)


class LaserRepo:
    def __init__(self, threshold=None, roi=None, roi_scale=1.0,
                 x_stride=1, y_stride=1, pixels_threshold=1,
                 merge=True, margin=1,
                 min_area=0, max_area=0, max_side=0, max_aspect=0.0,
                 min_fill=0.0):
        # 阈值：默认用"亮红激光"（排除暗红十字）；也可传老 RED_LAB。
        self.lab = tuple(int(v) for v in threshold) if threshold else RED_LASER_LAB
        if roi is not None:
            self.roi = tuple(int(v) for v in roi)
        else:
            c_x, c_y = LCD_W // 2, LCD_H // 2
            w = int(LCD_W // 2 * roi_scale)
            h = int(LCD_H // 2 * roi_scale)
            self.roi = (c_x - w // 2, c_y - h // 2, w, h)
        self.x_stride = int(x_stride)
        self.y_stride = int(y_stride)
        self.pixels_threshold = int(pixels_threshold)
        self.merge = bool(merge)
        self.margin = int(margin)
        self.min_area = int(min_area)
        self.max_area = int(max_area)
        self.max_side = int(max_side)
        self.max_aspect = float(max_aspect)
        self.min_fill = float(min_fill)
        # 用户要求：不做卡尔曼滤波，直接用原始质心（更跟手、不引入额外延迟）
        self.kf = None
        self.last = None
        self.last_ms = 0
        self.nblob = 0
        self.raw_blobs = 0
        self.blobs = []
        self.core = 0

    def _passes(self, b):
        """形状/面积/填充率门控：挡掉暗红十字、大片反光、细长亮带。"""
        try:
            bw, bh = int(b.w()), int(b.h())
            area = int(b.area())
        except Exception:
            return True
        if self.min_area and area < self.min_area:
            return False
        if self.max_area and area > self.max_area:
            return False
        side = max(bw, bh)
        if self.max_side and side > self.max_side:
            return False
        if self.max_aspect > 0.0:
            ar = side / float(max(1, min(bw, bh)))
            if ar > self.max_aspect:
                return False
        if self.min_fill > 0.0:
            fill = area / float(max(1, bw * bh))
            if fill < self.min_fill:
                return False
        return True

    def sample(self, img):
        t0 = time.ticks_ms()
        try:
            blobs = img.find_blobs([self.lab], False, roi=self.roi,
                                   x_stride=self.x_stride, y_stride=self.y_stride,
                                   pixels_threshold=self.pixels_threshold,
                                   merge=self.merge, margin=self.margin)
        except Exception:
            blobs = None
        self.last_ms = time.ticks_diff(time.ticks_ms(), t0)
        self.blobs = blobs or []
        self.nblob = len(self.blobs)
        self.raw_blobs = self.nblob
        now = time.ticks_ms()
        # 形状/面积/填充率门控：把暗红十字、大片反光、细长亮带挡在门外
        cands = [b for b in self.blobs if self._passes(b)]
        if not cands:
            self.last = None
            return None
        b = max(cands, key=lambda z: (z.area(), -z.y()))
        # 默认就用原始质心（所有 API 失败时也安全）
        rx, ry = b.cx(), b.cy()
        # ---- 方案 2：形状自适应（整段可回退，绝不漏检）----
        self.core = 0
        try:
            elong = float(b.elongation())
        except Exception:
            elong = 1.0
        bw, bh = int(b.w()), int(b.h())
        ar = float(max(bw, bh)) / float(max(1, min(bw, bh)))
        if elong >= LONG_ELONG or ar >= LONG_AR:
            core = _core_on_axis(img, b, self.last)
            if core is not None:
                rx, ry = core
                self.core = 1
        self.last = (rx, ry)
        return {"x": round(rx, 1), "y": round(ry, 1), "area": b.area(),
                "w": bw, "h": bh, "n": b.pixels(),
                "bx": int(b.x()), "by": int(b.y()),
                "raw": self.nblob, "core": self.core, "ms": self.last_ms}
