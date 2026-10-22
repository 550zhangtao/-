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

try:
    import ulab.numpy as _np           # 只用来把红通道比较成 0/255 掩码
except Exception:
    _np = None

VERSION = "2026-10-22.8-roi"

# ---- 诊断模式：True 时逐 blob 打印（含真实像素数），用于确认阈值是否合适 ----
DIAG_MODE = False

CFG = {
    # ROI：画面中心 120x120（640x480 中心是 (320,240)，左上角 = 中心 - 60）
    "roi": (295, 207, 100, 100),
    # 灰度阈值。实测扫描（本机现场）：200~220 选中的是激光；230~240 时激光会
    # 被拆成两片、反光反而面积最大 -> 选错；245 只剩激光碎片。取 220 留余量。
    "threshold": (220, 255),
    "min_area": 6,        # cv_lite 的 min_area（外接框面积下限）
    "kernel_size": 1,     # 1 = 不做形态学（3 会抹掉 3~4px 的小光斑核心）
    "diag_mode": 0,
    # 白色物体/白光/白墙 -> 大片均匀高亮。实测：笔电白屏 18x80=1440px，
    # 白墙整窗可到 120x120。激光只有 8x10≈80px，所以用**上限**把它们挡掉。
    # （白色圆柱边缘 7x9=63px 与激光同尺寸，尺寸上无法区分，见文档说明）
    "max_side": 40,
    "max_area": 600,
    # ---- 方案 B：红掩码作主检测（推荐）----
    # 激光是"红"而不是"亮"：实测红晕 R−G≈150，而白纸/白瓶 R−G≈−10，
    # 但激光灰度只有 120~160（红通道饱和时 G/B 仍低），白物体灰度 229+。
    # 所以用 (R>red_rmin 且 G<red_gmax) 建 0/255 掩码，再交给 cv_lite 找连通域。
    # 注意必须用 uint8 比较，不能先做 R-G（uint8 相减会溢出回绕）。
    "red_primary": 1,      # 1=红掩码主检测；0=退回灰度
    "red_dg": 60,          # R-G 阈值（相对判据，实测激光≈150，白物≈-10）
    # 比例判据（抗曝光漂移）：R/(R+G+B) > red_ratio 才算法是"饱和红"。
    # 与 red_rmin（绝对亮度）相比，亮度整体升降不影响判定——
    # 木墙 L 从 160 变到 230，R/sum 都稳定在 0.37 < 0.45。
    # 实测：激光红晕 0.545 | 白物 0.324 | 木墙 0.372 | 红袋 0.547
    "red_ratio": 0.45,
    # 孤立性：激光是孤立的点，红袋子是大片红。检查候选 bbox 外圈 band 像素
    # 内仍然"红"的比例，超过 iso_max 就判定它嵌在大片红色里，丢弃。
    "iso_band": 2,
    "iso_max": 0.35,
    "red_rmin": 240,       # 红色亮度下限。木纹墙 R≈160，激光 R≈255 ->
                           # 调到 200 可把木纹/棕墙排除（实测关键）
    "red_gmax": 110,       # 已不用；保留配置兼容
    "red_min_area": 40,    # 掩码连通域最小面积（像素）
    # 自动获取 ROI：连续 miss 到阈值后，把画面切成 4x3 块逐块建红掩码，
    # 找到"尺寸像激光"的红块就把 ROI 挪过去（单帧行为，不引入状态机）。
    # 只在丢失时跑，一次约 25ms；固定 ROI 够用时设 0 关闭。
    "auto_roi": 0,
    "auto_roi_miss": 2,      # 连续丢失多少帧后触发
    "auto_roi_every": 5,     # 最多每 N 帧触发一次
    "auto_roi_min_area": 60, # 候选红块的最小 bbox 面积
    "auto_roi_max_area": 800,
    "auto_roi_max_side": 45,
    # ---- 红占优颜色门控（灰度模式下的可选白物判据）----
    # 白色物体/白墙在灰度上与激光同样"小而亮"，灰度无解；但激光有**红色光晕**，
    # 白物体没有。这里在同一个 ROI 上再跑一次 rgb888_find_blobs（红阈值），
    # 只有"灰度候选点附近存在红色像素团"才接受，否则丢弃。
    # 实测 120x120 的 rgb888_find_blobs 只要 ~1ms。
    # 0 = 关闭（只传灰度图时自动关闭）
    "red_gate": 1,
    # 判据：候选 blob **自己的 bbox 内**，满足 (R-G>red_dg 且 R>red_rmin) 的
    # 像素占比 >= red_frac。实测（真实帧）：
    #   激光 100% | 白色圆柱 1% | 笔电白屏 0% | 白瓶 0%
    # 取 0.35 留余量（激光光晕弱时比例会下降，白物很难超过 0.2）。
    # 比"按红色团距离/bbox 包含"稳得多——红色背景会把这些几何判据全污染。
    "red_dg": 60,
    "red_rmin": 180,
    "red_frac": 0.35,
    "red_need_px": 6,        # bbox 太小(<6px)时不做判定
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
        self.frame = 0
        self.miss = 0
        self.auto_moves = 0

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

    def _auto_roi(self, rgb_img, t0):
        """扫 4x3 块，找尺寸像激光的红块，把 ROI 挪过去。返回 True 表示挪了。"""
        c = self.cfg
        tw, th = 160, 120
        best = None
        best_area = 0
        for ty in range(0, 480, th):
            for tx in range(0, 640, tw):
                try:
                    a = rgb_img.crop(roi=(tx, ty, tw, th)).to_numpy_ref()
                    dg2 = int(c.get("red_dg", 60))
                    rmin2 = int(c.get("red_rmin", 150))
                    m = _np.where((a[0] > rmin2) & (a[1] <= 255 - dg2) &
                                  (a[0] > a[1] + dg2), 255, 0)
                    raw = cv_lite.grayscale_find_blobs(
                        [th, tw], m, 128, 255,
                        int(c.get("auto_roi_min_area", 60)), 1)
                except Exception:
                    continue
                for i in range(0, len(raw) - 3, 4):
                    bw = raw[i + 2]
                    bh = raw[i + 3]
                    if bw > int(c.get("auto_roi_max_side", 45)):
                        continue
                    if bh > int(c.get("auto_roi_max_side", 45)):
                        continue
                    area = bw * bh
                    if area > int(c.get("auto_roi_max_area", 800)):
                        continue
                    if area > best_area:
                        best_area = area
                        best = (tx + raw[i] + bw * 0.5,
                                ty + raw[i + 1] + bh * 0.5)
        if best is None:
            return False
        rw, rh = self.roi[2], self.roi[3]
        nx = int(best[0] - rw * 0.5)
        ny = int(best[1] - rh * 0.5)
        if nx < 0:
            nx = 0
        if ny < 0:
            ny = 0
        if nx + rw > 640:
            nx = 640 - rw
        if ny + rh > 480:
            ny = 480 - rh
        old = self.roi
        self.roi = (nx, ny, rw, rh)
        self.auto_moves = getattr(self, "auto_moves", 0) + 1
        self.miss = 0
        if c.get("diag_mode", 0) or self.auto_moves <= 3:
            print("laser: 自动 ROI %s -> %s (红块中心 %.0f,%.0f area=%d)"
                  % (old, self.roi, best[0], best[1], best_area))
        return True

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

    def sample(self, gray_img, rgb_img=None):
        """对外入口：维护 frame/miss 计数，供自动 ROI 使用。"""
        self.frame += 1
        r = self._sample_impl(gray_img, rgb_img)
        if r is None:
            self.miss += 1
        else:
            self.miss = 0
        return r

    def _sample_impl(self, gray_img, rgb_img=None):
        """gray_img: GRAYSCALE Image（建议 chn0）。
        rgb_img: 可选 RGBP888/RGB888 Image（同一个 ROI 用于红占优门控）。
                 不传或 red_gate=0 时不做颜色判定。
        返回 dict 或 None。
        """
        self.last_ms = 0
        self.raw_blobs = 0
        self.blobs = []
        self.rejected = 0
        self.last_red = None
        self.candidates = 0
        if cv_lite is None:
            self.mode = "none"
            self.status = "no_cv_lite"
            return None

        t0 = _now_ms()
        x0, y0, rw, rh = self.roi
        use_red = (int(self.cfg.get("red_primary", 1)) and rgb_img is not None
                   and _np is not None and cv_lite is not None)
        if use_red:
            self.mode = "red"
        else:
            self.mode = "gray"
        # ---- 自动获取 ROI（只在持续丢失时触发）----
        self.miss = getattr(self, "miss", 0)
        if (use_red and int(self.cfg.get("auto_roi", 1))
                and self.miss >= int(self.cfg.get("auto_roi_miss", 2))
                and self.frame % int(self.cfg.get("auto_roi_every", 5)) == 0):
            self._auto_roi(rgb_img, t0)

        arr = None
        try:
            if use_red:
                # ---- 红掩码：uint8 比较，避免 R-G 溢出回绕 ----
                # 注意：这里必须"整帧取数组 + 切片"，不能用 img.crop()。
                # 实测 crop() 出来的 RGBP888 数组内容与切片不一致（对不上画面），
                # 而整帧 to_numpy_ref() 再切片是可以精确定位的。
                full = rgb_img.to_numpy_ref()
                arr = full[:, y0:y0 + rh, x0:x0 + rw]
                # 相对判据 R-G>red_dg，用 uint8 技巧绕开溢出：
                # G<=195 时 G+60 不会回绕；G>195 时 R-G>60 不可能成立，
                # 用 (G<=195) 这一项排掉。
                dg = int(self.cfg.get("red_dg", 60))
                R = arr[0]
                G = arr[1]
                # 注意 ulab 没有 int32：整数用 int16 且换算成不溢出的比例式，
                # 比例判据用 float（无溢出）。
                rmin = int(self.cfg.get("red_rmin", 200))
                rat = float(self.cfg.get("red_ratio", 0.0))
                cond = (R > rmin) & (G <= 255 - dg) & (R > G + dg)
                if rat > 0.0:
                    # R/(R+G+B) > rat  <=>  R*20 > 9*(R+G+B)（rat=0.45 的等价式，
                    # 两端都 < 6900，int16 安全）；其他 rat 用 float 兜底。
                    if abs(rat - 0.45) < 1e-6:
                        R16 = _np.array(R, dtype=_np.int16)
                        S16 = (R16 + _np.array(G, dtype=_np.int16)
                               + _np.array(arr[2], dtype=_np.int16))
                        cond = cond & (R16 * 20 > S16 * 9)
                    else:
                        Rf = _np.array(R, dtype=_np.float)
                        Sf = (Rf + _np.array(G, dtype=_np.float)
                              + _np.array(arr[2], dtype=_np.float))
                        cond = cond & (Rf > rat * Sf)
                mask = _np.where(cond, 255, 0)
                self._cond = cond
                raw = cv_lite.grayscale_find_blobs(
                    [rh, rw], mask, 128, 255,
                    int(self.cfg.get("red_min_area", 30)), 1)
            else:
                sub = gray_img.crop(roi=(x0, y0, rw, rh))
                arr = sub.to_numpy_ref()
                lo, hi = self.threshold
                raw = cv_lite.grayscale_find_blobs(
                    [rh, rw], arr, lo, hi,
                    int(self.cfg["min_area"]), int(self.cfg["kernel_size"]))
        except Exception as e:
            self.status = "crop_err" if arr is None else "cv_err"
            self.last_ms = _diff_ms(_now_ms(), t0)
            if self.cfg.get("diag_mode", 0):
                print("laser: %s 失败 %s" % (self.mode, e))
            return None

        nblob = len(raw) // 4
        self.raw_blobs = nblob
        self.last_ms = _diff_ms(_now_ms(), t0)      # 只算裁剪 + cv_lite

        # ---- 解析 raw -> self.blobs（cv_lite 只给外接框，用 w*h 当面积）----
        max_side = int(self.cfg.get("max_side", 0) or 0)
        max_area = int(self.cfg.get("max_area", 0) or 0)
        for i in range(0, len(raw) - 3, 4):
            bx, by, bw, bh = raw[i], raw[i + 1], raw[i + 2], raw[i + 3]
            area = bw * bh
            if max_side and (bw > max_side or bh > max_side):
                self.rejected += 1
                continue                      # 大片白（屏幕/墙/灯）
            if max_area and area > max_area:
                self.rejected += 1
                continue
            self.blobs.append({"x": bx, "y": by, "w": bw, "h": bh,
                               "area": area,
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

        def _key(g):
            k = float(g["area"])
            if bias > 0.0:
                dx = g["cx"] - ccx
                dy = g["cy"] - ccy
                d = (dx * dx + dy * dy) ** 0.5
                f = d / cref
                if f > 1.0:
                    f = 1.0
                k = float(g["area"]) * (1.0 - bias * f)
            return k

        # 按分数从大到小试；红占优判据不通过就试下一个候选。
        # （旧写法只试"最大那个"，一旦它被白物判据拒掉就整帧返回 None。）
        # ---- 孤立性过滤：去掉嵌在大片红色里的碎片（红袋子）----
        iso_band = int(self.cfg.get("iso_band", 0) or 0)
        iso_max = float(self.cfg.get("iso_max", 1.0))
        cond_arr = getattr(self, "_cond", None)
        if iso_band > 0 and cond_arr is not None and groups:
            kept = []
            for g in groups:
                gw = int(g["w"])
                gh = int(g["h"])
                gx = int(g["cx"] - x0 - gw * 0.5)
                gy = int(g["cy"] - y0 - gh * 0.5)
                tot_b = 0
                red_b = 0
                for yy in range(gy - iso_band, gy + gh + iso_band):
                    if yy < 0 or yy >= rh:
                        continue
                    for xx in range(gx - iso_band, gx + gw + iso_band):
                        if xx < 0 or xx >= rw:
                            continue
                        # 只统计 bbox 之外的 ring
                        if gx <= xx < gx + gw and gy <= yy < gy + gh:
                            continue
                        tot_b += 1
                        if cond_arr[yy, xx]:
                            red_b += 1
                if tot_b and float(red_b) / float(tot_b) > iso_max:
                    self.rejected += 1
                    continue
                kept.append(g)
            groups = kept

        ordered = sorted(groups, key=lambda g: -_key(g))
        # 红掩码模式下本身就是红色，不再重复做颜色门控
        gate = ((not use_red) and bool(int(self.cfg.get("red_gate", 0)))
                and rgb_img is not None)
        rarr = None
        if gate:
            try:
                rarr = rgb_img.crop(roi=(x0, y0, rw, rh)).to_numpy_ref()
            except Exception:
                rarr = None          # 取不到颜色就退化成纯灰度
                gate = False
        best = None
        for g in ordered:
            if not gate:
                best = g
                break
            bw = int(g["w"])
            bh = int(g["h"])
            bx = int(g["cx"] - x0 - bw * 0.5)
            by = int(g["cy"] - y0 - bh * 0.5)
            if bx < 0:
                bx = 0
            if by < 0:
                by = 0
            if bx + bw > rw:
                bw = rw - bx
            if by + bh > rh:
                bh = rh - by
            tot = bw * bh
            if tot < int(self.cfg.get("red_need_px", 6)) or bw <= 0 or bh <= 0:
                best = g            # 太小，无法判定 -> 直接接受
                break
            dg = int(self.cfg.get("red_dg", 60))
            rmin = int(self.cfg.get("red_rmin", 180))
            red_n = 0
            for yy in range(by, by + bh):
                for xx in range(bx, bx + bw):
                    v = rarr[0, yy, xx]
                    if v > rmin and v - rarr[1, yy, xx] > dg:
                        red_n += 1
            frac = float(red_n) / float(tot)
            self.last_red = (red_n, tot, round(frac, 2))
            if frac >= float(self.cfg.get("red_frac", 0.5)):
                best = g
                break
            self.rejected += 1      # 白物判据拒掉，继续看下一个候选

        if best is None:
            self.status = "red_reject"
            self.mode = "none"
            if self.cfg.get("diag_mode", 0):
                print("  [DIAG] 全部候选都被白物判据拒掉 (red=%s)"
                      % (getattr(self, "last_red", None),))
            self.last = None
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
        return {"x": cx, "y": cy, "conf": conf, "n": area, "red": getattr(self, "last_red", None),
                "w": best["w"], "h": best["h"],
                "mode": "roi", "src": "measured", "status": "ok",
                "path": "roi", "raw": nblob, "ms": self.last_ms}


# 旧名字兼容
LaserAim = LaserSpot
