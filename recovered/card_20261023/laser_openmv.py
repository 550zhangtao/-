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
    "roi": (315, 135, 100, 100),
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
    "red_primary": 1,      # 本步验证 ROI 跟随：只走灰度      # 1=红掩码主检测；0=退回灰度
    "red_dg": 60,          # R-G 阈值（相对判据，实测激光≈150，白物≈-10）
    # 比例判据（抗曝光漂移）：R/(R+G+B) > red_ratio 才算法是"饱和红"。
    # 与 red_rmin（绝对亮度）相比，亮度整体升降不影响判定——
    # 木墙 L 从 160 变到 230，R/sum 都稳定在 0.37 < 0.45。
    # 实测：激光红晕 0.545 | 白物 0.324 | 木墙 0.372 | 红袋 0.547
    "red_ratio": 0.0,   # 比例判据（float 较慢，实测 R>200 已够）
    # 孤立性：激光是孤立的点，红袋子是大片红。检查候选 bbox 外圈 band 像素
    # 内仍然"红"的比例，超过 iso_max 就判定它嵌在大片红色里，丢弃。
    # ---- 跳变抑制（2~3 帧的轻量关联，不是卡尔曼/状态机）----
    "max_step": 20.0,
    "step_confirm": 3,
    "jump_guard": 1,
    "acquire": 1,   # 启动全图捕获一次
    # 直方图自适应阈值：用 ROI 的 R 通道直方图取"最亮的 1%"作为红色下限，
    # 替代写死的 R>200（光照变化时自动跟随）。
    "hist_thr": 1,
    "hist_pct": 0.01,
    "rmin_floor": 150,
    "rmin_ceil": 245,
    "iso_band": 2,
    "iso_max": 0.35,
    "red_rmin": 200,       # 红色亮度下限。木纹墙 R≈160，激光 R≈255 ->
                           # 调到 200 可把木纹/棕墙排除（实测关键）
    "red_gmax": 110,       # 已不用；保留配置兼容
    "red_min_area": 40,    # 掩码连通域最小面积（像素）
    # ---- ROI 跟随上一帧（本步核心）----
    # 每帧 ROI 中心 = 上一帧检测坐标，大小 roi_size；检测失败保持不动，
    # 连续失败超过 lost_reset 帧才回到画面中心。没有卡尔曼、没有舵机映射。
    "track": 1,
    "roi_size": 120,
    "lost_reset": 30,
    # 分级恢复的 ROI 尺寸（按连续丢失帧数）
    "roi_lv2": 160,        # 丢失 6~15 帧
    "roi_lv3": 200,        # 丢失 16~30 帧（改用纯面积选择）
    # 选择规则：跟随模式下"离 ROI 中心最近"比"面积最大"更稳——
    # 实测同一帧常有 3~4 个尺寸相近的候选，按面积选会在它们之间跳。
    # 1 = 按距离选（推荐）；0 = 按面积（旧行为）
    # 形状门控（单帧几何判据）：激光是近似圆形、实心但不过饱和的小块。
    # cv_lite 只给外接框，所以 fill 用"bbox 内灰度>=阈值的像素数 / bbox 面积"实测。
    "shape_gate": 0,
    "aspect_max2": 2.0,    # max(w,h)/min(w,h) <= 2.0
    "fill_min": 0.40,
    "fill_max": 0.85,
    # 起始 ROI 中心（默认画面中心）。现场激光不在中心时，先指到它所在处，
    # 让跟随从正确的位置起步；之后完全由 last_x/last_y 接管。
    "start_x": 320.0,
    "start_y": 240.0,
    # 自动获取 ROI：连续 miss 到阈值后，把画面切成 4x3 块逐块建红掩码，
    # 找到"尺寸像激光"的红块就把 ROI 挪过去（单帧行为，不引入状态机）。
    # 只在丢失时跑，一次约 25ms；固定 ROI 够用时设 0 关闭。
    "auto_roi": 0,         # 本步关闭（由 track 接管）
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
    "merge_px": 12.0,        # 碎片合并的中心距阈值（文档推荐 12）
    "merge_ratio": 3.0,      # 面积比 1:3 ~ 3:1 才合并，防止吞掉大块
    "close_k": 0,            # >0 时对掩码做闭运算（弥合碎片），如 3
    "subpix": 1,             # 1=加权亚像素质心
    # ---- 轻量卡尔曼：每轴 2 状态 [p, v]，预测周期 20ms ----
    # 对应文档 [x,y,vx,vy] 状态向量；此处按轴解耦，等效且更省算力。
    "kf_on": 1,
    "kf_dt": 0.020,          # 预测周期 20ms
    "kf_r": 4.0,             # 观测噪声方差（约 ±2px）
    "kf_qp": 0.5,            # 位置过程噪声
    "kf_qv": 200.0,          # 速度过程噪声
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
        # ROI 跟随状态：初始在画面中心
        self.last_x = float(self.cfg.get("start_x", 320.0))
        self.last_y = float(self.cfg.get("start_y", 240.0))
        self.roi_c = (self.last_x, self.last_y)
        self.acquired = 0
        self.kf_x = None
        self.kf_y = None
        self.kf_t = 0
        self.lost_run = 0
        self.jump_n = 0

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

    def _acquire(self, gray_img):
        """全图扫一次，挑最像激光的块（小、方、孤立）作为跟随起点。"""
        c = self.cfg
        full = gray_img.to_numpy_ref()
        rh, rw = 480, 640
        raw = cv_lite.grayscale_find_blobs(
            [rh, rw], full, int(c["threshold"][0]), int(c["threshold"][1]),
            max(6, int(c.get("min_area", 6))), int(c.get("kernel_size", 1)))
        mx_side = int(c.get("max_side", 40))
        mx_area = int(c.get("max_area", 600))
        best = None
        best_score = -1e9
        for i in range(0, len(raw) - 3, 4):
            bx, by, bw, bh = raw[i], raw[i + 1], raw[i + 2], raw[i + 3]
            if bw <= 0 or bh <= 0 or bw > mx_side or bh > mx_side:
                continue
            area = bw * bh
            if area > mx_area or area < 60:
                continue
            big = bw if bw > bh else bh
            sml = bh if bw > bh else bw
            asp = float(big) / float(sml) if sml else 99.0
            if asp > 2.0:
                continue
            # 越接近方形、面积适中越像激光
            score = 100.0 - abs(asp - 1.2) * 20.0 - abs(area - 260) * 0.05
            if score > best_score:
                best_score = score
                best = (bx + bw * 0.5, by + bh * 0.5)
        if best is not None:
            self.last_x = float(best[0])
            self.last_y = float(best[1])
            self.roi_c = (self.last_x, self.last_y)
            print("laser: 启动捕获 (%.0f,%.0f)" % best)

    def _kf_step(self, cx, cy, now):
        """每轴 2 状态卡尔曼 [p,v]；返回滤波后的 (x,y)。"""
        c = self.cfg
        dt = float(c.get("kf_dt", 0.02))
        r = float(c.get("kf_r", 4.0))
        qp = float(c.get("kf_qp", 0.5))
        qv = float(c.get("kf_qv", 200.0))
        if self.kf_x is None:
            # 初始化：位置=观测，速度=0，协方差给一个中等初值
            self.kf_x = [float(cx), 0.0, r, 0.0, qv]
            self.kf_y = [float(cy), 0.0, r, 0.0, qv]
            self.kf_t = now
            return cx, cy
        # 预测（用固定 20ms 周期，避免帧间隔抖动被放大）
        for st in (self.kf_x, self.kf_y):
            st[0] += st[1] * dt
            p00 = st[2] + dt * (st[3] + st[3]) + dt * dt * st[4] + qp * dt
            p01 = st[3] + dt * st[4]
            p11 = st[4] + qv * dt
            st[2] = p00
            st[3] = p01
            st[4] = p11
        # 更新
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
        self.kf_t = now
        return self.kf_x[0], self.kf_y[0]

    def _acquire_color(self, rgb_img):
        """全图一次 R−G 扫描，挑最像激光的红色小块作为跟随起点。"""
        c = self.cfg
        full = rgb_img.to_numpy_ref()
        H, W = 480, 640
        if len(full.shape) == 3 and full.shape[0] == 3:
            Rf, Gf, Bf = full[0], full[1], full[2]
        else:
            Rf = full[:, :, 0]
            Gf = full[:, :, 1]
            Bf = full[:, :, 2]
        dg = int(c.get("red_dg", 60))
        rmin = int(c.get("red_rmin", 200))
        cond = (Rf > rmin) & (Gf <= 255 - dg) & (Rf > Gf + dg)
        mask = _np.where(cond, 255, 0)
        raw = cv_lite.grayscale_find_blobs([H, W], mask, 128, 255, 40, 1)
        mx_side = int(c.get("max_side", 40))
        mx_area = int(c.get("max_area", 600))
        best = None
        best_score = -1e9
        for i in range(0, len(raw) - 3, 4):
            bx, by, bw, bh = raw[i], raw[i + 1], raw[i + 2], raw[i + 3]
            if bw <= 0 or bh <= 0 or bw > mx_side or bh > mx_side:
                continue
            area = bw * bh
            if area > mx_area or area < 100:
                continue
            big = bw if bw > bh else bh
            sml = bh if bw > bh else bw
            asp = float(big) / float(sml) if sml else 99.0
            if asp > 2.2:
                continue
            score = 100.0 - abs(asp - 1.2) * 20.0 - abs(area - 300) * 0.05
            if score > best_score:
                best_score = score
                best = (bx + bw * 0.5, by + bh * 0.5)
        if best is not None:
            self.last_x = float(best[0])
            self.last_y = float(best[1])
            self.roi_c = (self.last_x, self.last_y)
            print("laser: 颜色捕获 (%.0f,%.0f)" % best)
        else:
            print("laser: 颜色捕获失败，沿用起点")

    def _auto_roi(self, rgb_img, t0):
        """扫 4x3 块，找尺寸像激光的红块，把 ROI 挪过去。返回 True 表示挪了。"""
        c = self.cfg
        tw, th = 160, 120
        best = None
        best_area = 0
        try:
            _full = rgb_img.to_numpy_ref()      # 必须整帧取再切片：
        except Exception:                        # crop() 对 RGBP888 不可靠
            return False
        _pl = (len(_full.shape) == 3 and _full.shape[0] == 3)
        for ty in range(0, 480, th):
            for tx in range(0, 640, tw):
                try:
                    if _pl:
                        a = _full[:, ty:ty + th, tx:tx + tw]
                    else:
                        a = _full[ty:ty + th, tx:tx + tw, :].transpose(2, 0, 1)
                    dg2 = int(c.get("red_dg", 60))
                    rmin2 = int(c.get("red_rmin", 150))
                    rat2 = float(c.get("red_ratio", 0.0))
                    Rr = a[0]
                    Gg = a[1]
                    cond2 = ((Rr > rmin2) & (Gg <= 255 - dg2) &
                             (Rr > Gg + dg2))
                    if rat2 > 0.0:
                        R2 = _np.array(Rr, dtype=_np.float)
                        S2 = (R2 + _np.array(Gg, dtype=_np.float)
                              + _np.array(a[2], dtype=_np.float))
                        cond2 = cond2 & (R2 > rat2 * S2)
                    m = _np.where(cond2, 255, 0)
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
            self.lost_run += 1
            # 连续丢失超过 lost_reset 帧 -> 回到画面中心重新找
            if self.lost_run > int(self.cfg.get("lost_reset", 30)):
                self.last_x = float(self.cfg.get("start_x", 320.0))
                self.last_y = float(self.cfg.get("start_y", 240.0))
                self.lost_run = 0
        else:
            self.miss = 0
            self.lost_run = 0
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
        # ---- 启动自动捕获（只做一次）----
        if (not self.acquired) and int(self.cfg.get("acquire", 1)):
            self.acquired = 1
            try:
                if int(self.cfg.get("red_primary", 0)) and rgb_img is not None:
                    self._acquire_color(rgb_img)
                elif gray_img is not None:
                    self._acquire(gray_img)
            except Exception:
                pass
        # ---- ROI 跟随：中心 = 上一帧检测坐标 ----
        if int(self.cfg.get("track", 1)):
            lr = int(getattr(self, "lost_run", 0))
            if lr <= 5:
                sz = int(self.cfg.get("roi_size", 120))
            elif lr <= 15:
                sz = int(self.cfg.get("roi_lv2", 160))
            else:
                sz = int(self.cfg.get("roi_lv3", 200))
            ccx = float(self.last_x)
            ccy = float(self.last_y)
            x0 = int(ccx - sz * 0.5)
            y0 = int(ccy - sz * 0.5)
            if x0 < 0:
                x0 = 0
            if y0 < 0:
                y0 = 0
            if x0 + sz > 640:
                x0 = 640 - sz
            if y0 + sz > 480:
                y0 = 480 - sz
            rw = rh = sz
            self.roi = (x0, y0, rw, rh)
            self.roi_c = (x0 + rw * 0.5, y0 + rh * 0.5)
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
        self._wimg = None
        try:
            if use_red:
                # ---- 红掩码：uint8 比较，避免 R-G 溢出回绕 ----
                # 注意：这里必须"整帧取数组 + 切片"，不能用 img.crop()。
                # 实测 crop() 出来的 RGBP888 数组内容与切片不一致（对不上画面），
                # 而整帧 to_numpy_ref() 再切片是可以精确定位的。
                full = rgb_img.to_numpy_ref()
                # to_numpy_ref() 的布局随来源不同：
                #   原生 RGBP888 -> (3,H,W) 平面；to_rgb888() 之类 -> (H,W,3) 交错。
                # 两种都支持，避免 IndexError。
                _sh = full.shape
                if len(_sh) == 3 and _sh[0] == 3:
                    _plane = True
                    arr = full[:, y0:y0 + rh, x0:x0 + rw]
                    R = arr[0]
                    G = arr[1]
                    B = arr[2]
                else:
                    _plane = False
                    arr = full[y0:y0 + rh, x0:x0 + rw, :]
                    R = arr[:, :, 0]
                    G = arr[:, :, 1]
                    B = arr[:, :, 2]
                self._wimg = R            # 亚像素质心的权重图（红强度）
                self._wlo = rmin if False else 0
                dg = int(self.cfg.get("red_dg", 60))
                # 注意 ulab 没有 int32：整数用 int16 且换算成不溢出的比例式，
                # 比例判据用 float（无溢出）。
                rmin = int(self.cfg.get("red_rmin", 200))
                if int(self.cfg.get("hist_thr", 0)):
                    try:
                        _h = cv_lite.rgb888_calc_histogram(
                            [rh, rw], _np.array(R, dtype=_np.uint8))
                        _hr = _h[0]
                        _tot = 0
                        for _v in _hr:
                            _tot += _v
                        _tgt = _tot * float(self.cfg.get("hist_pct", 0.01))
                        _acc = 0
                        _b = 255
                        while _b > 0:
                            _acc += _hr[_b]
                            if _acc >= _tgt:
                                break
                            _b -= 1
                        _lo = int(self.cfg.get("rmin_floor", 150))
                        _hi = int(self.cfg.get("rmin_ceil", 245))
                        rmin = _b
                        if rmin < _lo:
                            rmin = _lo
                        if rmin > _hi:
                            rmin = _hi
                        self.auto_rmin = rmin
                    except Exception:
                        pass
                rat = float(self.cfg.get("red_ratio", 0.0))
                cond = (R > rmin) & (G <= 255 - dg) & (R > G + dg)
                if rat > 0.0:
                    # R/(R+G+B) > rat  <=>  R*20 > 9*(R+G+B)（rat=0.45 的等价式，
                    # 两端都 < 6900，int16 安全）；其他 rat 用 float 兜底。
                    if abs(rat - 0.45) < 1e-6:
                        R16 = _np.array(R, dtype=_np.int16)
                        S16 = (R16 + _np.array(G, dtype=_np.int16)
                               + _np.array(B, dtype=_np.int16))
                        cond = cond & (R16 * 20 > S16 * 9)
                    else:
                        Rf = _np.array(R, dtype=_np.float)
                        Sf = (Rf + _np.array(G, dtype=_np.float)
                              + _np.array(B, dtype=_np.float))
                        cond = cond & (Rf > rat * Sf)
                mask = _np.where(cond, 255, 0)
                ck = int(self.cfg.get("close_k", 0) or 0)
                if ck > 0:
                    try:
                        mask = cv_lite.rgb888_close(
                            [rh, rw], _np.array(mask, dtype=_np.uint8), ck, 1)
                    except Exception:
                        try:
                            # 退化方案：对掩码做一次 3x3 膨胀再腐蚀
                            _m = _np.array(mask, dtype=_np.uint8)
                            mask = cv_lite.rgb888_dilate([rh, rw], _m, ck, 1)
                            mask = cv_lite.rgb888_erode([rh, rw], mask, ck, 1)
                        except Exception:
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

        if not use_red:
            self._wimg = arr
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
            # 实测 fill：bbox 内满足灰度阈值的像素数 / bbox 面积
            # 只在灰度模式 + 开启形状门控时计算（红掩码模式下 arr 是三通道，
            # 按二维索引会越界）。
            cnt_px = area
            if (not use_red) and int(self.cfg.get("shape_gate", 0)):
                lo_t = int(self.cfg["threshold"][0])
                cnt_px = 0
                for yy in range(by, by + bh):
                    for xx in range(bx, bx + bw):
                        if arr[yy, xx] >= lo_t:
                            cnt_px += 1
            big_s = bw if bw > bh else bh
            sml_s = bh if bw > bh else bw
            self.blobs.append({"x": bx, "y": by, "w": bw, "h": bh,
                               "area": area,
                               "cx": bx + bw * 0.5 + x0,
                               "cy": by + bh * 0.5 + y0,
                               "fill": (float(cnt_px) / float(area)) if area else 0.0,
                               "aspect": (float(big_s) / float(sml_s)) if sml_s else 99.0,
                               "px": cnt_px})

        # ---- 近距离合并（模拟 OpenMV find_blobs 的 merge=True）----
        mg = float(self.cfg.get("merge_px", 0.0) or 0.0)
        groups = []
        for it in self.blobs:
            placed = False
            if mg > 0.0:
                for g in groups:
                    dx = it["cx"] - g["cx"]
                    dy = it["cy"] - g["cy"]
                    # 面积比必须在 1:ratio ~ ratio:1 之间才允许合并
                    ra = float(it["area"])
                    rb = float(g["area"])
                    rr_ok = True
                    if rb > 0 and ra > 0:
                        q = ra / rb
                        mr = float(self.cfg.get("merge_ratio", 3.0) or 0.0)
                        if mr > 0.0 and (q > mr or q < (1.0 / mr)):
                            rr_ok = False
                    if rr_ok and dx * dx + dy * dy <= mg * mg:
                        a0 = g["area"]
                        a1 = it["area"]
                        tot = a0 + a1
                        g["cx"] = (g["cx"] * a0 + it["cx"] * a1) / tot
                        g["cy"] = (g["cy"] * a0 + it["cy"] * a1) / tot
                        g["area"] = tot
                        g["w"] = max(g["w"], it["w"])
                        g["h"] = max(g["h"], it["h"])
                        g["px"] = g.get("px", 0) + int(it.get("px", 0))
                        g["parts"] += 1
                        placed = True
                        break
            if not placed:
                groups.append({"cx": it["cx"], "cy": it["cy"], "w": it["w"],
                               "h": it["h"], "area": it["area"], "parts": 1,
                               "px": int(it.get("px", 0))})
        # 合并后的 group 必须重新算 aspect/fill（否则形状门控拿不到字段）
        for g in groups:
            gw = int(g["w"])
            gh = int(g["h"])
            bs = gw if gw > gh else gh
            ss = gh if gw > gh else gw
            g["aspect"] = (float(bs) / float(ss)) if ss else 99.0
            box = gw * gh
            g["fill"] = (min(1.0, float(g.get("px", 0)) / float(box))) if box else 0.0
        self.groups = groups

        bias = float(self.cfg.get("center_bias", 0.0) or 0.0)
        cref = float(self.cfg.get("center_ref_px", 60.0) or 60.0)
        ccx, ccy = x0 + rw * 0.5, y0 + rh * 0.5      # ROI 中心（原图坐标）

        def _key(g):
            if int(self.cfg.get("select_near", 0)):
                dx = g["cx"] - ccx
                dy = g["cy"] - ccy
                d = (dx * dx + dy * dy) ** 0.5
                # 距离为主、面积为次（面积只用来在距离相同时分胜负）
                return -d + min(float(g["area"]), 2000.0) / 100000.0
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

        # ---- 选择规则：先形状门控，通过者取面积最大；全被拒则退回纯面积 ----
        # 单帧几何判据，不做多帧确认。
        sel = None
        best = None
        if groups:
            gate_on = int(self.cfg.get("shape_gate", 0))
            amax = float(self.cfg.get("aspect_max2", 2.0))
            fmin = float(self.cfg.get("fill_min", 0.40))
            fmax = float(self.cfg.get("fill_max", 0.85))
            passed = []
            if gate_on:
                for g in groups:
                    if (float(g.get("aspect", 99.0)) <= amax
                            and fmin <= float(g.get("fill", 0.0)) <= fmax):
                        passed.append(g)
                    else:
                        self.rejected += 1
            if passed:
                best = max(passed, key=lambda g: float(g["area"]))
                sel = "shape"
            else:
                # 保底：形状门控没放行任何候选时，仍按面积选，避免丢锁
                best = max(groups, key=lambda g: float(g["area"]))
                sel = "fallback" if gate_on else "area"
        self.sel = sel
        if best is None:
            self.status = "no_blob"
            self.mode = "none"
            self.last = None
            return None
        cx = best["cx"]
        cy = best["cy"]
        # ---- 灰度加权亚像素质心（减少 bbox 整数中心的量化误差）----
        _wimg = getattr(self, "_wimg", None)
        if int(self.cfg.get("subpix", 1)) and _wimg is not None:
            try:
                bw0 = int(best["w"])
                bh0 = int(best["h"])
                bx0 = int(best["cx"] - 640.0 - bw0 * 0.5) if False else int(best["cx"] - x0 - bw0 * 0.5)
                by0 = int(best["cy"] - y0 - bh0 * 0.5)
                sx = 0.0
                sy = 0.0
                sw = 0.0
                lo_w = int(self.cfg["threshold"][0]) if not use_red else int(rmin)
                for yy in range(by0, by0 + bh0):
                    if yy < 0 or yy >= rh:
                        continue
                    for xx in range(bx0, bx0 + bw0):
                        if xx < 0 or xx >= rw:
                            continue
                        if yy >= len(_wimg) or xx >= len(_wimg[0]):
                            continue
                        w_px = float(_wimg[yy][xx]) - lo_w
                        if w_px > 0:
                            sx += xx * w_px
                            sy += yy * w_px
                            sw += w_px
                if sw > 0:
                    cx = sx / sw + x0
                    cy = sy / sw + y0
            except Exception:
                pass

        area = best["area"]
        # conf 只是"相对面积"的可读指标，不参与任何逻辑
        conf = _clamp(area / 60.0, 0.10, 1.0)

        # ---- 跳变抑制：位移过大先按住，连续 N 帧才接受 ----
        if int(self.cfg.get("jump_guard", 1)) and self.last is not None:
            ms_px = float(self.cfg.get("max_step", 20.0))
            ddx = cx - self.last[0]
            ddy = cy - self.last[1]
            if ddx * ddx + ddy * ddy > ms_px * ms_px:
                self.jump_n += 1
                if self.jump_n < int(self.cfg.get("step_confirm", 3)):
                    cx = self.last[0]
                    cy = self.last[1]
                    sel = "hold"
                else:
                    self.jump_n = 0
            else:
                self.jump_n = 0

        # ---- 卡尔曼滤波（在跳变抑制之后、更新跟随点之前）----
        if int(self.cfg.get("kf_on", 0)):
            try:
                cx, cy = self._kf_step(cx, cy, _now_ms())
            except Exception:
                pass

        # ---- 跟随状态更新 ----
        self.last_x = float(cx)
        self.last_y = float(cy)
        self.lost_run = 0
        self.last = (cx, cy)
        self.last_px = area
        self.candidates = 1
        self.mode = "roi"
        self.src = "measured"
        self.status = "ok"
        return {"x": cx, "y": cy, "conf": conf, "n": area, "red": getattr(self, "last_red", None),
                "w": best["w"], "h": best["h"],
                "mode": "roi", "src": "measured", "status": "ok",
                "path": "roi", "raw": nblob, "ms": self.last_ms,
                "sel": sel}


# 旧名字兼容
LaserAim = LaserSpot
