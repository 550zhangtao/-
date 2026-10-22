# -*- coding: utf-8 -*-
"""
laser_openmv.py —— 极简激光点检测（2026-10-22 重写，VERSION 2026-10-22.8-roi）
================================================================================

**本轮只有一件事：给定 ROI，用 cv_lite 灰度阈值找最大 blob。**

刻意**没有**：先验 / 评分 / 候选过滤 / pending 多帧确认 / 大块取亮核 /
自适应阈值 / 状态机。上一版这些全删了——它们把 lms 拖到 110~209ms。

为什么之前会 110~209ms
----------------------
旧实现用 OpenMV 原生 `img.find_blobs()`（全图 11~24ms）**再对每个候选做 Python
逐像素扫描**（`_blob_pixels`，几千次取像素）→ 单帧上百毫秒。
本实现只用 cv_lite：C 实现的连通域，**ROI 裁剪后只要 0.2ms**（实测）。

实测（CanMV v1.4.3 / GC2093 / 640×480 灰度）
-------------------------------------------
| 操作 | 耗时 |
| --- | --- |
| `gray.crop(roi=120x120)` + `to_numpy_ref()` | 0.30 ms |
| `cv_lite.grayscale_find_blobs` 120×120 | **0.20 ms** |
| 同上，640×480 全图（对照） | 4.10 ms |

接口
----
    lf = LaserSpot()                 # 可选 cfg 覆盖 ROI/THRESHOLD
    r = lf.sample(gray_img)          # gray_img = chn0 的 GRAYSCALE Image
    # r = {"x","y","conf","n","raw","ms",...} 或 None
    lf.last_ms                       # 本次 裁剪+cv_lite 耗时
    lf.raw_blobs                     # ROI 内原始 blob 数量

坐标是 **640×480 原图坐标**（ROI 内坐标已加偏移）。
"""

import time

try:
    import cv_lite
except Exception:                      # 固件没有 cv_lite 时置 None，sample 返回 None
    cv_lite = None

VERSION = "2026-10-22.8-roi"

# ---- 诊断模式：True 时逐 blob 打印（含真实像素数），用于确认阈值是否合适 ----
DIAG_MODE = False

CFG = {
    # ROI：画面中心 120x120（640x480 中心是 (320,240)，左上角 = 中心 - 60）
    "roi": (260, 220, 120, 120),
    # 灰度阈值。实测扫描（本机现场）：200~220 选中的是激光；230~240 时激光会
    # 被拆成两片、反光反而面积最大 -> 选错；245 只剩激光碎片。取 220 留余量。
    "threshold": (220, 255),
    "min_area": 6,        # cv_lite 的 min_area（外接框面积下限）
    "kernel_size": 1,     # 1 = 不做形态学（3 会抹掉 3~4px 的小光斑核心）
    "diag_mode": 0,
    "diag_max_side": 40,  # 诊断时只对这么大的 blob 数真实像素，避免开销
    # 选最大 blob 时，离 ROI 中心越远扣得越多（单帧、无状态）。
    # 0 = 纯按面积；0.35 表示最远处面积打 65 折。实测能压掉 ROI 角落的
    # 反光点（激光 273px@中心附近  vs  反光 156~300px@偏 35px）。
    # cv_lite 没有 merge 参数：把质心相距 <= merge_px 的块当成同一光斑合并
    # （激光常被阈值拆成 2~3 片相邻亮块，不合并会在几片之间来回跳）
    "merge_px": 20.0,
    "center_bias": 0.35,
    "center_ref_px": 60.0,   # 距离归一化尺度
}


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def _now_ms():
    try:
        return time.ticks_ms()
    except AttributeError:
        return int(time.monotonic() * 1000)


def _diff_ms(a, b):
    try:
        return time.ticks_diff(a, b)
    except AttributeError:
        return a - b


class LaserSpot:
    """ROI + cv_lite 灰度阈值，取面积最大的 blob。没有任何先验/评分。"""

    def __init__(self, cfg=None):
        self.cfg = dict(CFG)
        if cfg:
            self.cfg.update(cfg)
        self.roi = tuple(int(v) for v in self.cfg["roi"])
        self.threshold = (int(self.cfg["threshold"][0]),
                          int(self.cfg["threshold"][1]))
        self.last = None
        self.last_ms = 0
        self.raw_blobs = 0
        self.last_px = 0            # 兼容旧调用方（= 上次 blob 的框面积）
        self.blobs = []             # 本次所有 blob（诊断/外部查看）
        # 兼容旧调用方：本轮没有先验
        self.prior = None
        self.calib_dev = 0.0
        self.need_calib = False
        self.need_recalib = False
        self.candidates = 0
        self.mode = "none"
        self.src = "none"
        self.status = "none"

    # ---- 兼容旧接口：本轮不用先验 ----
    def load(self, path=None):
        return False

    def set_calib(self, x, y, kx=0.0, ky=0.0, ref=0.0):
        return None

    def reset(self):
        self.last = None
        self.last_ms = 0
        self.raw_blobs = 0
        self.blobs = []
        self.mode = "none"
        self.src = "none"
        self.status = "none"

    def _diag_pixels(self, arr, x, y, w, h):
        """诊断用：数 blob 内真实像素数（只对 <= diag_max_side 的块做）。"""
        lim = int(self.cfg.get("diag_max_side", 40))
        if w > lim or h > lim:
            return -1
        lo = self.threshold[0]
        n = 0
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                if arr[yy, xx] >= lo:
                    n += 1
        return n

    def sample(self, gray_img):
        """gray_img: GRAYSCALE Image（建议 chn0）。返回 dict 或 None。"""
        self.last_ms = 0
        self.raw_blobs = 0
        self.blobs = []
        self.candidates = 0
        if cv_lite is None:
            self.mode = "none"
            self.status = "no_cv_lite"
            return None

        t0 = _now_ms()
        x0, y0, rw, rh = self.roi
        # ---- ROI 裁剪（0.3ms）----
        try:
            sub = gray_img.crop(roi=(x0, y0, rw, rh))
            arr = sub.to_numpy_ref()
        except Exception as e:
            self.status = "crop_err"
            self.last_ms = _diff_ms(_now_ms(), t0)
            if self.cfg.get("diag_mode", 0):
                print("laser: crop 失败 %s" % e)
            return None

        # ---- cv_lite（120x120 实测 0.2ms）----
        lo, hi = self.threshold
        try:
            raw = cv_lite.grayscale_find_blobs(
                [rh, rw], arr, lo, hi,
                int(self.cfg["min_area"]), int(self.cfg["kernel_size"]))
        except Exception as e:
            self.status = "cv_err"
            self.last_ms = _diff_ms(_now_ms(), t0)
            if self.cfg.get("diag_mode", 0):
                print("laser: cv_lite 失败 %s" % e)
            return None

        nblob = len(raw) // 4
        self.raw_blobs = nblob
        self.last_ms = _diff_ms(_now_ms(), t0)      # 只算裁剪 + cv_lite

        # ---- 解析 raw -> self.blobs（cv_lite 只给外接框，用 w*h 当面积）----
        for i in range(0, len(raw) - 3, 4):
            bx, by, bw, bh = raw[i], raw[i + 1], raw[i + 2], raw[i + 3]
            self.blobs.append({"x": bx, "y": by, "w": bw, "h": bh,
                               "area": bw * bh,
                               "cx": bx + bw * 0.5 + x0,
                               "cy": by + bh * 0.5 + y0})

        # ---- 近距离合并（模拟 OpenMV find_blobs 的 merge=True）----
        mg = float(self.cfg.get("merge_px", 0.0) or 0.0)
        groups = []
        for it in self.blobs:
            placed = False
            if mg > 0.0:
                for g in groups:
                    dx = it["cx"] - g["cx"]
                    dy = it["cy"] - g["cy"]
                    if dx * dx + dy * dy <= mg * mg:
                        a0 = g["area"]
                        a1 = it["area"]
                        tot = a0 + a1
                        g["cx"] = (g["cx"] * a0 + it["cx"] * a1) / tot
                        g["cy"] = (g["cy"] * a0 + it["cy"] * a1) / tot
                        g["area"] = tot
                        g["w"] = max(g["w"], it["w"])
                        g["h"] = max(g["h"], it["h"])
                        g["parts"] += 1
                        placed = True
                        break
            if not placed:
                groups.append({"cx": it["cx"], "cy": it["cy"], "w": it["w"],
                               "h": it["h"], "area": it["area"], "parts": 1})
        self.groups = groups

        bias = float(self.cfg.get("center_bias", 0.0) or 0.0)
        cref = float(self.cfg.get("center_ref_px", 60.0) or 60.0)
        ccx, ccy = x0 + rw * 0.5, y0 + rh * 0.5      # ROI 中心（原图坐标）
        best = None
        best_key = 0.0
        for item in groups:
            area = item["area"]
            cx = item["cx"]
            cy = item["cy"]
            key = float(area)
            if bias > 0.0:
                dx = cx - ccx
                dy = cy - ccy
                d = (dx * dx + dy * dy) ** 0.5
                f = d / cref
                if f > 1.0:
                    f = 1.0
                key = float(area) * (1.0 - bias * f)
            item["key"] = key
            if best is None or key > best_key:
                best_key = key
                best = item

        if self.cfg.get("diag_mode", 0):
            print("[DIAG] roi=%s thr=%s raw=%d lms=%.1fms"
                  % (self.roi, self.threshold, nblob, self.last_ms))
            for it in self.blobs:
                px = self._diag_pixels(arr, it["x"], it["y"], it["w"], it["h"])
                print("  [DIAG] blob cx=%.0f cy=%.0f w=%d h=%d area=%d pixels=%s"
                      % (it["cx"], it["cy"], it["w"], it["h"], it["area"],
                         "n/a" if px < 0 else px))

        if best is None:
            self.last = None
            self.mode = "none"
            self.src = "none"
            self.status = "none"
            return None

        cx, cy = best["cx"], best["cy"]
        area = best["area"]
        # conf 只是"相对面积"的可读指标，不参与任何逻辑
        conf = _clamp(area / 60.0, 0.10, 1.0)
        self.last = (cx, cy)
        self.last_px = area
        self.candidates = 1
        self.mode = "roi"
        self.src = "measured"
        self.status = "ok"
        return {"x": cx, "y": cy, "conf": conf, "n": area,
                "w": best["w"], "h": best["h"],
                "mode": "roi", "src": "measured", "status": "ok",
                "path": "roi", "raw": nblob, "ms": self.last_ms}


# 旧名字兼容
LaserAim = LaserSpot
