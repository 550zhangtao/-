# -*- coding: utf-8 -*-
"""
laser_spot.py —— K230 板端激光光斑检测（RGB565，纯板端，零 PC 依赖）
======================================================================

由 `yolo_pose.py`（USB 推流）和 `yolo_preview_ide.py`（CanMV IDE 预览）
共同 import，保证两处看到的是**同一份**检测结果。

真机判据（2026-10-21 现场实测校准）
------------------------------------
1. **白色种子**：光斑核心过曝成 1~6px 白点（LAB L 高、a/b 中性）。
   板端 `find_blobs(LAB_WHITE)` 只命中最核心的几个像素，
   **红色晕圈要另外用 `find_blobs(LAB_RED)` 找**（两路合并）。
2. **红色晕圈**：每个白点种子在 ±3px 内必须存在 max(R-G) >= min_warm。
   实测：真激光 warm 119~220；墙上白色插座/日光灯高光 warm 2~33。
3. **三通道饱和**：min(R,G,B) >= min_sat（真激光 237~253）。
4. **局部对比**：峰值亮度 - 外圈(24x24)中位数 >= min_contrast（自适应光照）。
   实测真激光 83~98（墙上）。
5. **形状**：面积/bbox/长宽比限制，剔除窗框、灯带。
6. **红色光晕**（2026-10-21 补）：白色种子 ±5px 内有 >= min_halo_warm 个红像素
   （R-G >= halo_warm_min）。消防栓/红色物品的小反光点是"孤立白点"，
   周围一圈**没有**红晕；真激光的晕圈连成片。
   实测：真激光 halo 30~60+，误检 0~14。

两段搜索（速度）
----------------
* 已锁定：上一帧附近 70px 小 ROI（`find_blobs` 是 C 实现，几 ms）；
* 丢失：140px → 以靶心为锚 240/320px → 全图；
* 白点先找，红点兜底（激光弱、没过曝时）。

注意：`get_pixel()` 是 Python 级调用（~6.7µs/px），绝不能用它扫全图；
全部用 `img.bytearray()`（零拷贝视图）+ 下标读字节，且只在很小的 bbox 内。

坐标：全部是**原图 640x480 坐标系**（img 必须是 chn1 RGB565 那一路）。
"""

import time

# ----------------------------------------------------------- 配置
# 会被 PC 端按正则替换，务必保持**单行** dict 字面量
CFG = {'enabled': 1, 'auto_calib': 1, 'calib_tries': 6, 'white_enable': 1, 'min_sat': 190, 'core_sat': 200, 'strict_sat': 190, 'core_ratio_pct': 6, 'core_min_px': 1, 'core_max_px': 40, 'min_warm': 60, 'abs_contrast': 35, 'rel_contrast': 0.30, 'dim_peak': 45, 'min_contrast': 55, 'min_fill': 0.30, 'strict_fill': 0.40, 'max_side': 48, 'min_area_big': 2, 'wide_enable': 0, 'wide_peak': 110, 'wide_warm': 40, 'wide_area': 60, 'wide_contrast': 60, 'dim_conf_scale': 0.40, 'dim_radius': 50, 'anchor_rad': 120, 'confirm_rad': 30, 'win_rad': 12, 'win_grow': 12, 'win_rad_max': 60, 'win_luma': 110, 'win_warm': 40, 'win_contrast': 40, 'win_red_warm': 120, 'win_red_luma': 50, 'win_warm_hot': 70, 'win_luma_low': 150, 'win_hot_luma': 245, 'win_hot_min': 220, 'slow_every': 1,  'subpix_rad': 5, 'search_rad': 70, 'roi_stride': 2, 'reacquire_rad': 140, 'conf_k': 120, 'budget_ms': 12, 'gate_k': 3.0, 'kf_q': 40.0, 'kf_r': 4.0, 'kf_max_gate': 48.0, 'fixed_coast_max': 10, 'fixed_verify_every': 5, 'fixed_allow_reacquire': 0, 'fixed_reacquire_after': 10, 'fixed_reacquire_every': 30, 'fixed_reacquire_rad': 180, 'fixed_reacquire_stride': 6}

# 亮白阈值（板端 LAB：L 0~100）。光斑核心 L 接近 100。
LAB_WHITE = [(82, 100, -30, 30, -30, 30)]
# 红点兜底 / 红晕：激光功率弱、核心没过曝时，以及给白点找红晕。
# L 下限 40（实测 L=20~30 时会命中消防栓/绿瓶的暗红晕，造成大量候选）。
LAB_RED = [(40, 100, 35, 127, -128, 127)]
# 宽亮阈值：原本想让"远处不过曝的小光斑"也能被 find_blobs 捞到。
# **实测不可用**：它会把墙上大片亮区连成一个 110x128 的大 blob
# （merge=True 合并相邻亮区），激光白核被吞进去后因 max_side 被整体丢弃，
# 反而造成"有锚点时必漏"。代码保留但默认关闭（wide_enable=0）。
LAB_WIDE = [(70, 100, -80, 80, -60, 60)]


# ----------------------------------------------------------- 像素
def _rgb565(ba, w, x, y):
    i = (int(y) * w + int(x)) * 2
    if i + 1 >= len(ba):
        return (0, 0, 0)
    v = ba[i] | (ba[i + 1] << 8)
    return (((v >> 11) & 0x1F) * 8, ((v >> 5) & 0x3F) * 4, (v & 0x1F) * 8)


def _luma(r, g, b):
    return (r * 77 + g * 150 + b * 29) >> 8


def _dist(ax, ay, bx, by):
    dx = ax - bx
    dy = ay - by
    return (dx * dx + dy * dy) ** 0.5


def _roi(cx, cy, rad, w, h):
    x = int(cx) - rad
    y = int(cy) - rad
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    rw = rad * 2
    rh = rad * 2
    if x + rw > w:
        rw = w - x
    if y + rh > h:
        rh = h - y
    if rw < 8:
        rw = 8
    if rh < 8:
        rh = 8
    return (x, y, rw, rh)


def _overlap(a, b, pad=3):
    """两个 bbox (x,y,w,h) 是否近距离重合。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if ax > bx + bw + pad or bx > ax + aw + pad:
        return False
    if ay > by + bh + pad or by > ay + ah + pad:
        return False
    return True


# ----------------------------------------------------------- 时间（板端/PC 通用）
def _now_ms():
    try:
        return time.ticks_ms()
    except AttributeError:
        return int(time.monotonic() * 1000)


def _diff_ms(a, b):
    try:
        return time.ticks_diff(a, b)
    except AttributeError:
        return int(a) - int(b)


# ----------------------------------------------------------- 光斑预测器
class SpotKalman:
    """2D 匀速 Kalman（x/y 各一组 p,v），SORT 运动模型的轻量版。

    为什么不用 OpenCV KCF/CSRT：K230 板端没有 OpenCV，CSRT 内存/算力也
    撑不住；光斑是单点、运动近似匀速，2 状态 Kalman + 门控就够。
    参考：
      * SORT（Kalman + Mahalanobis 门控）https://github.com/abewley/sort
      * ByteTrack（track/coast/lost 生命周期）
        https://github.com/ifzhang/ByteTrack
      * OpenCV Kalman 教程
        https://docs.opencv.org/4.x/dd/d6a/tutorial_track_kalman.html

    q: 过程噪声（px^2/s^3），r: 测量噪声（px^2）
    max_gate: 门控最大半径（px），防止预测发散后窗口无限大
    """
    def __init__(self, q=40.0, r=4.0, max_gate=48.0):
        self.q = float(q)
        self.r = float(r)
        self.max_gate = float(max_gate)
        self.init = False
        self.t = 0
        self.px = 0.0; self.vx = 0.0
        self.Px0 = 0.0; self.Px1 = 0.0; self.Px2 = 0.0
        self.py = 0.0; self.vy = 0.0
        self.Py0 = 0.0; self.Py1 = 0.0; self.Py2 = 0.0

    def reset(self, x, y, t):
        self.px = float(x); self.py = float(y)
        self.vx = 0.0; self.vy = 0.0
        self.Px0 = self.r * 64.0; self.Px1 = 0.0; self.Px2 = 25.0
        self.Py0 = self.r * 64.0; self.Py1 = 0.0; self.Py2 = 25.0
        self.t = int(t)
        self.init = True

    def _predict_axis(self, p, v, P0, P1, P2, dt):
        p += v * dt
        P0 += dt * (2.0 * P1 + dt * P2) + self.q * dt * dt * dt / 3.0
        P1 += dt * P2 + self.q * dt * dt / 2.0
        P2 += self.q * dt
        return p, P0, P1, P2

    def predict(self, t):
        if not self.init:
            return (self.px, self.py)
        dt = _diff_ms(t, self.t) / 1000.0
        if dt < 0.001:
            dt = 0.001
        if dt > 0.20:
            dt = 0.20
        self.t = int(t)
        self.px, self.Px0, self.Px1, self.Px2 = self._predict_axis(
            self.px, self.vx, self.Px0, self.Px1, self.Px2, dt)
        self.py, self.Py0, self.Py1, self.Py2 = self._predict_axis(
            self.py, self.vy, self.Py0, self.Py1, self.Py2, dt)
        return (self.px, self.py)

    def _update_axis(self, z, p, v, P0, P1, P2):
        S = P0 + self.r
        if S <= 1e-6:
            return z, v, P0, P1, P2
        K0 = P0 / S
        K1 = P1 / S
        innov = float(z) - p
        p += K0 * innov
        v += K1 * innov
        P0n = (1.0 - K0) * P0
        P1n = (1.0 - K0) * P1
        P2n = P2 - K1 * P1
        return p, v, P0n, P1n, P2n

    def update(self, zx, zy):
        if not self.init:
            self.reset(zx, zy, self.t)
            return
        self.px, self.vx, self.Px0, self.Px1, self.Px2 = self._update_axis(
            zx, self.px, self.vx, self.Px0, self.Px1, self.Px2)
        self.py, self.vy, self.Py0, self.Py1, self.Py2 = self._update_axis(
            zy, self.py, self.vy, self.Py0, self.Py1, self.Py2)

    def gate_radius(self, k=3.0):
        if not self.init:
            return self.max_gate
        rx = k * ((self.Px0 + self.r) ** 0.5)
        ry = k * ((self.Py0 + self.r) ** 0.5)
        r = rx if rx > ry else ry
        if r < 8.0:
            r = 8.0
        if r > self.max_gate:
            r = self.max_gate
        return r

    def gate_ok(self, zx, zy, k=3.0):
        if not self.init:
            return True
        dx = float(zx) - self.px
        dy = float(zy) - self.py
        if abs(dx) > self.max_gate or abs(dy) > self.max_gate:
            return False
        return (dx * dx <= k * k * (self.Px0 + self.r)
                and dy * dy <= k * k * (self.Py0 + self.r))


# ----------------------------------------------------------- 单次检测
class LaserSpot:
    """有状态的光斑检测器：记住上一帧位置，下一帧先在小 ROI 里找。"""

    def __init__(self, cfg=None):
        self.cfg = dict(CFG)
        if cfg:
            self.cfg.update(cfg)
        self.last = None        # 上一帧 (x, y)，用于小 ROI 优先
        self.miss = 0
        self._ba = None
        self._base_luma = None  # 背景亮度基线（低频更新，省 8ms/帧）
        self._frame_n = 0
        self.last_cands = 0     # 上次候选数（诊断用）
        self.last_ms = 0        # 上次耗时（无论成功与否）
        self._slow_n = 0        # 慢速(全流程)检测计数，用于降频
        self.fixed = None       # 标定位置 (x, y)：相机与激光笔固连，光斑位置近似常量
        self.just_calibrated = False   # 刚完成一次视觉标定（供上层保存到文件）
        self._verify_fail = 0   # 复核连续失败次数
        self._verify_ok = True
        self._conf_n = 0        # 弱特征候选连续命中计数
        self._conf_xy = None    # 上次数时序位置
        self.mode_hist = []     # 最近几次命中模式（诊断）
        self.kf = SpotKalman(self.cfg["kf_q"], self.cfg["kf_r"],
                             self.cfg["kf_max_gate"])
        self.coast = 0          # 连续无测量帧数
        self._grid_i = 0        # 无先验捕获的分格索引
        self._budget_hit = False

    def set_fixed(self, x, y):
        """写入标定位置：之后 locate() 直接返回它，几乎零开销。

        依据：相机与激光笔都固定在云台上（AGENTS.md 实测 pan 50µs 只移动
        5~9px），所以光斑在画面里的位置是常量，没必要每帧去"找"它。
        """
        self.fixed = (float(x), float(y))
        self._verify_fail = 0
        self._verify_ok = True
        self.just_calibrated = False
        self.last = (float(x), float(y))
        self.kf.reset(float(x), float(y), _now_ms())
        self.coast = 0

    def reset(self):
        # 注意：fixed 不清（标定是物理量），只清跟踪状态
        self.last = (self.fixed[0], self.fixed[1]) if self.fixed else None
        self.miss = 0
        self._base_luma = None
        self.last_cands = 0
        self._conf_n = 0
        self._conf_xy = None
        self.mode_hist = []
        self.coast = 0
        self._grid_i = 0
        self._budget_hit = False
        if self.fixed:
            self.kf.reset(self.fixed[0], self.fixed[1], _now_ms())

    # ------------------------------------------------------ 单 ROI 检测
    def _scan(self, img, roi, base_luma=0, stride=1, allow_wide=False,
              strict=False):
        """单次 find_blobs（白+红阈值合并），一次调用拿完所有候选。

        为什么合并：find_blobs 约有 11ms 固定开销（与 ROI 面积关系不大），
        调两次就是 22ms 以上，是预览掉帧的主因。
        实测 merge=True 会把"过曝白核"与其外围"红色晕圈"合并成一块完整
        光斑（bbox 覆盖两者），所以一次调用即可同时判断两个特征。

        判定：bbox 内必须同时出现
          * 白核：某像素 min(R,G,B) >= min_sat
          * 红晕：某像素 R-G >= min_warm
        两者不可能同一像素，分开累计并早退（见 _bbox_stats）。
        """
        c = self.cfg
        w = img.width()
        h = img.height()
        ba = self._ba
        try:
            ths = [LAB_WHITE[0], LAB_RED[0]]
            # 宽亮阈值只在"有位置先验"时加：它会让 blob 数翻倍、每次多 ~13ms，
            # 换来的是"远处不过曝的激光"能被找到。没有先验时纯靠颜色区分不了
            # 远处激光和红色小标签，加了也只是带来误检，不值。
            if c["wide_enable"] and allow_wide:
                ths.append(LAB_WIDE[0])
            blobs = img.find_blobs(ths, roi=roi,
                                   x_stride=stride, y_stride=stride,
                                   pixels_threshold=1, area_threshold=1,
                                   merge=True)
        except Exception:
            return []
        out = []
        for b in blobs:
            bw = b.w()
            bh = b.h()
            if bw <= 0 or bh <= 0 or bw > c["max_side"] or bh > c["max_side"]:
                continue
            if b.pixels() < c["min_area_big"]:
                continue
            x0 = b.x()
            y0 = b.y()
            if x0 + bw > w or y0 + bh > h:
                continue
            peak, mwarm, mmin = self._bbox_stats(
                ba, w, x0, y0, bw, bh, 1,
                need_sat=c["min_sat"], need_warm=c["min_warm"])
            # --- 强特征：白核 + 红晕同时存在（正常光照/近距离）
            # 注意：不再用"绝对白核阈值"区分红杂物——实测激光的白核
            # minRGB 只有 196（远处/弱光），而门框红标签也能到 199，
            # 绝对阈值无法分开。改用**白核像素占比**：
            #   真激光 core(>=200) 占 bbox 约 20%，红电缆/红标签只有 3%。
            if mmin >= c["min_sat"] and mwarm >= c["min_warm"]:
                if not self._contrast_ok(peak, base_luma):
                    continue
                # 白核必须是"真过曝"的一小团：像素数 >= core_min_px，
                # 且不超过 core_max_px（太大是灯罩/大面积高光）。
                # 实测红标签最亮处只有 1 个 ~199 的伪影像素，会被拒。
                ncore = self._core_pixels(ba, w, x0, y0, bw, bh,
                                          c["core_sat"], c["core_max_px"])
                if ncore < c["core_min_px"] or ncore > c["core_max_px"]:
                    continue
                # 白核占比：真激光是"过曝的一小团"，红色杂物只是零星高光
                if ncore * 100 < c["core_ratio_pct"] * bw * bh:
                    continue
                # 填充率：激光是"实心点扩散"（0.5+），红电缆那种"线状反光"
                # 在 bbox 里只占 0.1~0.2（实测 0.19）。
                if b.pixels() < c["min_fill"] * bw * bh:
                    continue
                out.append(self._score(img, ba, w, h, b.cx(), b.cy(),
                                       max(bw, bh), peak, mwarm, mmin,
                                       (x0, y0, bw, bh), (b.cx(), b.cy()),
                                       "white", 1))
                continue
            # --- 弱特征：远距离/暗光的小光斑
            # 实测缩小到 19px 时 RGB=(248,204,224)：luma 219、warm 44、min 204，
            # 白核不再严格饱和、红晕也变淡，但仍是"小、亮、微红、强对比"。
            if not c["wide_enable"] or strict:
                continue
            if b.pixels() > c["wide_area"] or peak < c["wide_peak"]:
                continue
            if mwarm < c["wide_warm"]:
                continue
            if peak < base_luma + c["wide_contrast"]:
                continue
            cd = self._score(img, ba, w, h, b.cx(), b.cy(), max(bw, bh),
                             peak, mwarm, mmin, (x0, y0, bw, bh),
                             (b.cx(), b.cy()), "dim", 1)
            cd["score"] *= c["dim_conf_scale"]
            out.append(cd)
        return out

    def _contrast_ok(self, peak, base):
        """光斑是否"明显比背景亮"。

        两套判据取或：
          * 绝对差：peak >= base + abs_contrast（亮场景）
          * 相对比：peak >= base * (1 + rel_contrast)（黑底：base 很小时
            绝对差天然很小，必须换比例来判，否则黑背景上的光斑会被漏掉）
        再加一个绝对亮度下限 dim_peak，压掉全暗场景的传感器噪点。
        """
        c = self.cfg
        if peak < c["dim_peak"]:
            return False
        if peak >= base + c["abs_contrast"]:
            return True
        return peak >= base * (1.0 + c["rel_contrast"])

    @staticmethod
    def _core_pixels(ba, w, x, y, bw, bh, thr, limit=0):
        """bbox 内 min(R,G,B) >= thr 的像素数（白核像素数）。

        这是区分"激光过曝白核"与"红色印刷品/贴纸"最硬的指标：
        激光中心是真的三通道全饱和，红色物体最多在 JPEG 锐化边缘出现
        个别接近饱和的像素（且往往只有 1 个）。
        """
        n = 0
        cnt = 0
        for yy in range(y, y + bh):
            for xx in range(x, x + bw):
                cnt += 1
                if cnt > 600:      # 扫描预算：小光斑 bbox 远小于此
                    return n
                r, g, bl = _rgb565(ba, w, xx, yy)
                mn = r if r < g else g
                if bl < mn:
                    mn = bl
                if mn >= thr:
                    n += 1
                    if limit and n >= limit:
                        return n
        return n

    @staticmethod
    def _estimate_base(ba, w, h, step=16):
        """背景亮度基线：稀疏采样 16px 网格取中位数（约 1200 点）。"""
        vals = []
        for yy in range(8, h, step):
            for xx in range(8, w, step):
                r, g, bl = _rgb565(ba, w, xx, yy)
                vals.append(_luma(r, g, bl))
        if not vals:
            return 0
        vals.sort()
        return vals[len(vals) // 2]

    @staticmethod
    def _count_red(ba, w, h, x, y, bw, bh, c, step=1):
        """bbox 内 R-G >= min_warm 的像素数（即"红晕像素"计数）。"""
        n = 0
        thr = c["min_warm"]
        x0 = x if x > 0 else 0
        y0 = y if y > 0 else 0
        x1 = x + bw if x + bw < w else w
        y1 = y + bh if y + bh < h else h
        cnt = 0
        for yy in range(y0, y1, step):
            for xx in range(x0, x1, step):
                cnt += 1
                r, g, bl = _rgb565(ba, w, xx, yy)
                if r - g >= thr:
                    n += 1
        return n

    @staticmethod
    def _bbox_stats(ba, w, x, y, bw, bh, step=1, need_sat=0, need_warm=0):
        """小 bbox 内 (最亮, 最大 R-G, 最大 minRGB)；step>1 时稀疏采样。

        need_sat/need_warm > 0 时启用早退：一旦发现满足条件的像素就返回，
        用于"判断这块是不是激光晕圈"（不需要精确最大值），
        真激光中心几 px 就命中，比扫完整个 bbox 快一个数量级。
        """
        peak = -1
        mwarm = -999
        mmin = -1
        want = bool(need_sat)
        has_core = False
        has_warm = False
        for yy in range(y, y + bh, step):
            for xx in range(x, x + bw, step):
                r, g, bl = _rgb565(ba, w, xx, yy)
                lu = _luma(r, g, bl)
                if lu > peak:
                    peak = lu
                warm = r - g
                if warm > mwarm:
                    mwarm = warm
                mn = r if r < g else g
                if bl < mn:
                    mn = bl
                if mn > mmin:
                    mmin = mn
                if want and not (has_core and has_warm):
                    # 白核（三通道饱和）与红晕不可能在同一像素：
                    # 白核 R≈G≈B → warm≈0；红晕三通道不饱和。分开累计。
                    if mn >= need_sat:
                        has_core = True
                    elif warm >= need_warm:
                        has_warm = True
                    if has_core and has_warm:
                        return peak, mwarm, mmin
        return peak, mwarm, mmin

    def _score(self, img, ba, w, h, cx, cy, big, peak, max_warm, max_min,
               bbox, core, mode, halo=0):
        """局部对比 + 打分。contrast 用外圈中位数（手工采样，兼容 islice）。"""
        R = int(big) + 12
        x0 = int(cx) - R
        y0 = int(cy) - R
        x1 = int(cx) + R
        y1 = int(cy) + R
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 > w:
            x1 = w
        if y1 > h:
            y1 = h
        bx0, by0, bw, bh = bbox
        vals = []
        for yy in range(y0, y1, 2):
            for xx in range(x0, x1, 2):
                if bx0 <= xx < bx0 + bw and by0 <= yy < by0 + bh:
                    continue

                r, g, bl = _rgb565(ba, w, xx, yy)
                vals.append(_luma(r, g, bl))
        if vals:
            vals.sort()
            ring = vals[len(vals) // 2]
        else:
            ring = 0
        contrast = peak - ring
        cd = {"x": float(cx), "y": float(cy), "area": bw * bh,
              "peak": peak, "contrast": contrast, "warm": max_warm,
              "sat": max_min, "mode": mode, "bbox": bbox, "core": core,
              "ring": ring}
        # 暖色权重高：墙上白高光 warm 只有 2~33，激光 >= 119
        cd["score"] = contrast + 0.7 * max_warm + 0.12 * max_min
        # 光晕面积：真激光晕圈成片，红色物体孤立小反光没有
        cd["halo"] = halo
        cd["score"] += 1.5 * halo if mode == "white" else 0.0
        if mode == "white":
            cd["score"] += 8.0
        return cd

    # ------------------------------------------------------ 亚像素
    def _subpixel(self, img, cd):
        w = img.width()
        h = img.height()
        ba = self._ba
        px, py = cd["core"]
        R = self.cfg["subpix_rad"]
        x0 = px - R if px - R > 0 else 0
        y0 = py - R if py - R > 0 else 0
        x1 = px + R + 1 if px + R + 1 < w else w
        y1 = py + R + 1 if py + R + 1 < h else h
        sx = 0.0
        sy = 0.0
        sw = 0.0
        # 优先用红色晕圈加权：核心是白的、靶纸白底没有红晕，不会被拉偏
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                r, g, bl = _rgb565(ba, w, xx, yy)
                warm = r - g
                if warm > 4:
                    sx += xx * warm
                    sy += yy * warm
                    sw += warm
        if sw > 0:
            return (sx / sw, sy / sw)
        # 整窗都不红：退回亮度加权
        base = cd["peak"] - 80
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                r, g, bl = _rgb565(ba, w, xx, yy)
                wt = _luma(r, g, bl) - base
                if wt > 0:
                    sx += xx * wt
                    sy += yy * wt
                    sw += wt
        if sw <= 0:
            return (float(px), float(py))
        return (sx / sw, sy / sw)

    # ------------------------------------------------------ 对外
    def _scan_window(self, img, cx, cy, rad, base_luma, stride=1, t0=0, budget=0):
        """只在 (cx,cy)±rad 的小窗口里逐像素找光斑，不调用 find_blobs。

        为什么可行：相机与激光笔固连，云台小幅转动时光斑只移动几~几十 px
        （遗留 aim_core 实测 pan 50µs ≈ 5~9px）。所以绝大多数帧只需在上一帧
        附近几十 px 内确认一下即可。
        开销：窗口 (2*rad)^2 / stride^2 个像素 × ~7µs。rad=14,stride=1 →
        784 px ≈ 5ms；受限重捕 rad=180,stride=6 → 3600 px ≈ 25ms（每 30 帧一次）。
        """
        c = self.cfg
        w = img.width()
        h = img.height()
        ba = self._ba
        stride = int(stride)
        if stride < 1:
            stride = 1
        rad = int(rad)          # gate_radius() 是 float；range() 必须 int
        x0 = int(cx) - rad
        y0 = int(cy) - rad
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        x1 = int(cx) + rad
        y1 = int(cy) + rad
        if x1 > w:
            x1 = w
        if y1 > h:
            y1 = h
        need_l = base_luma + c["win_contrast"]
        if need_l < c["win_luma"]:
            need_l = c["win_luma"]
        best = -1
        bx = -1
        by = -1
        bmk = -1
        for yy in range(y0, y1, stride):
            # 硬时间预算：小窗逐像素是 Python 级循环，必须能中途放弃。
            if budget and _diff_ms(_now_ms(), t0) > budget:
                self._budget_hit = True
                break
            for xx in range(x0, x1, stride):
                r, g, bl = _rgb565(ba, w, xx, yy)
                lu = _luma(r, g, bl)
                warm = r - g
                # 过曝白核：live 曝光可能把红点打成纯白，warm≈0 但 luma 很高
                hot = (lu >= c.get("win_hot_luma", 245)
                       and r >= c.get("win_hot_min", 220)
                       and g >= c.get("win_hot_min", 220)
                       and bl >= c.get("win_hot_min", 220))
                if warm < c["win_warm"] and not hot:
                    continue
                if lu < need_l and not hot:
                    # 墙很亮时，激光点 luma 只比墙高一点（实测 RGB≈255,174,197,
                    # luma≈200 vs 墙≈190），但 r-g≈81 远高于墙；允许这种
                    # “高暖色 + 中亮”的点。纯暗红点走 win_red_* 通道。
                    if warm >= c.get("win_red_warm", 120) and lu >= c.get("win_red_luma", 50):
                        pass
                    elif (warm >= c.get("win_warm_hot", 70)
                          and lu >= c.get("win_luma_low", 150)):
                        pass
                    else:
                        continue
                # 亮的 + 偏红的优先（激光是"最亮又带红晕"的那一点）
                mk = lu + warm
                if mk > best:
                    best = mk
                    bx = xx
                    by = yy
                    bmk = warm
        if best < 0:
            return None
        return {"x": float(bx), "y": float(by),
                "conf": min(1.0, (best - need_l) / 120.0 + 0.4),
                "contrast": 0, "warm": bmk, "sat": 0, "mode": "win",
                "area": 1, "ms": 0}

    def locate(self, img, anchor=None):
        """返回 dict(x, y, conf, contrast, warm, sat, mode, ms) 或 None。

        anchor: 粗搜索锚点（一般是 Pose 靶心/框中心）。
        """
        if not self.cfg["enabled"]:
            return None
        t0 = _now_ms()
        now = t0
        self._budget_hit = False
        try:
            self._ba = img.bytearray()
        except Exception:
            return None
        ba = self._ba
        w = img.width()
        h = img.height()
        c = self.cfg
        budget = int(c.get("budget_ms", 12))

        # ===== 固定标定模式：预测点门控 + 每 N 帧复核，绝不回退全图搜索 =====
        # 设计要点（踩过两个坑后定的）：
        #  1) 相机与激光笔固连，光斑在画面里的位置基本是常量：只在每
        #     fixed_verify_every 帧复核一次（12~60px 小窗，~5ms），其余帧
        #     直接返回标定位置，**零搜索开销**。
        #  2) 复核连续失败 fixed_reacquire_after 次后，每 fixed_reacquire_every
        #     帧做一次 ±fixed_reacquire_rad 的**受限**重捕（stride=6，~25ms），
        #     绝不回退全图 find_blobs（复杂画面单帧 0.5~2.3s，会把流拖到 0.5fps）。
        #  3) fixed_allow_reacquire 只由 R2/START 的离线代理置 1；IDE 预览 /
        #     live 置 0，保证「只有 R2 才进入（受限）全局搜索」。
        if self.fixed is not None:
            self._frame_n += 1
            if not self.kf.init:
                self.kf.reset(self.fixed[0], self.fixed[1], now)
            # 先预测下一帧位置，再只在预测点附近的门控窗里找
            px, py = self.kf.predict(now)
            pred_x, pred_y = px, py
            if self._base_luma is None:
                self._base_luma = self._estimate_base(ba, w, h)
            ve = int(c.get("fixed_verify_every", 5))
            src = "fixed"
            mode = "fixed"
            conf = 1.0
            if ve <= 1 or (self._frame_n % ve) == 0:
                # 门控半径来自预测协方差，不再"逐级放大搜索范围"
                rad = self.kf.gate_radius(float(c.get("gate_k", 3.0))) + 4.0
                grow = c["win_rad"] + self._verify_fail * c["win_grow"]
                if grow > rad:
                    rad = float(grow)
                if rad < c["win_rad"]:
                    rad = float(c["win_rad"])
                if rad > c["win_rad_max"]:
                    rad = float(c["win_rad_max"])
                # 两段搜索：先在预测点 ±16px 中心小窗找（光斑通常就在预测点），
                # 找不到再扩到门控大窗。中心先扫很关键：_scan_window 从左上角
                # 开始，而预算包含前面的 _estimate_base(8ms)，大窗可能还没扫到
                # 中心的点就超时了（实测一直 calib 的根因）。
                vr = self._scan_window(img, int(px), int(py), 16,
                                       self._base_luma, stride=2,
                                       t0=t0, budget=budget)
                if vr is None:
                    vr = self._scan_window(img, int(px), int(py), int(rad),
                                           self._base_luma, stride=2,
                                           t0=t0, budget=budget)
                if vr is not None and (self.kf.gate_ok(
                        vr["x"], vr["y"], float(c.get("gate_k", 3.0)))
                        or self._verify_fail >= 2):
                    px, py = vr["x"], vr["y"]
                    self.kf.update(px, py)
                    if (not self.just_calibrated
                            and _dist(px, py, self.fixed[0], self.fixed[1]) > 15):
                        self.just_calibrated = True   # 上层写回 laser_calib.json
                    self.fixed = (px, py)
                    self._verify_fail = 0
                    self._verify_ok = True
                    self.coast = 0
                    conf = vr["conf"]
                    src = "measured"
                else:
                    self._verify_fail += 1
                    self._verify_ok = False
                    self.coast += 1
                    conf = 0.35
                    # 连续失败后的受限重捕（±180px、stride6、仍受预算约束）
                    if (c.get("fixed_allow_reacquire", 0)
                            and self._verify_fail >= int(c.get("fixed_reacquire_after", 10))
                            and (self._frame_n % int(c.get("fixed_reacquire_every", 30))) == 0):
                        vr2 = self._scan_window(
                            img, int(px), int(py),
                            int(c.get("fixed_reacquire_rad", 180)),
                            self._base_luma,
                            stride=int(c.get("fixed_reacquire_stride", 6)),
                            t0=t0, budget=budget)
                        if vr2 is not None:
                            px, py = vr2["x"], vr2["y"]
                            self.kf.reset(px, py, now)   # 重捕 = 重新锁定
                            self.fixed = (px, py)
                            self._verify_fail = 0
                            self._verify_ok = True
                            self.coast = 0
                            conf = vr2["conf"]
                            src = "measured"
                            mode = "fixed+reacquire"
                    if self.coast > int(c.get("fixed_coast_max", 10)):
                        # 长期丢失：回到标定位置，避免预测漂移出画面
                        self.kf.reset(self.fixed[0], self.fixed[1], now)
                        self.coast = 0
                        px, py = self.fixed
            self.last = (px, py)
            self.last_ms = _diff_ms(_now_ms(), t0)
            self.last_cands = 1
            return {"x": px, "y": py, "conf": conf, "contrast": 0,
                    "warm": 0, "sat": 0, "mode": mode, "area": 0,
                    "src": src, "ms": self.last_ms,
                    "pred": [pred_x, pred_y], "coast": self.coast,
                    "ok": 1 if self._verify_ok else 0,
                    "budget": self._budget_hit}

        # 非 fixed 路径（首帧捕获/无标定）：预算已用完就先返回，下一帧继续
        if _diff_ms(_now_ms(), t0) > budget:
            self._budget_hit = True
            self.last_ms = _diff_ms(_now_ms(), t0)
            return None

        # 背景基线低频更新：全图稀疏采样约 8ms，每 10 帧算一次即可
        self._frame_n += 1
        if self._base_luma is None or (self._frame_n % 20) == 1:
            self._base_luma = self._estimate_base(ba, w, h)
        base = self._base_luma

        # 分级搜索（性能关键）：
        #   已锁定 -> 上一帧 ±search_rad 小窗（面积 1/20，find_blobs 只要 1~2ms）
        #   丢失   -> 扩大到 reacquire_rad
        #   还不行 -> 全图（约 40ms，罕见）
        # 注：锁定在小窗内搜索天生抑制"两个候选来回跳"。
        # 1) 已锁定：先在小窗口里逐像素确认（~5ms，不调 find_blobs）
        if self.last is not None:
            rad = c["win_rad"] + self.miss * c["win_grow"]
            if rad > c["win_rad_max"]:
                rad = c["win_rad_max"]
            wr = self._scan_window(img, self.last[0], self.last[1], rad, base)
            if wr is not None:
                self.last = (wr["x"], wr["y"])
                self.miss = 0
                self.last_cands = 1
                self.last_ms = time.ticks_diff(time.ticks_ms(), t0)
                self.mode_hist.append("win")
                if len(self.mode_hist) > 10:
                    self.mode_hist.pop(0)
                return wr

        # 2) 小窗口失败：走完整流程（find_blobs），但降频避免每帧都慢。
        #    例外：还没标定成功时，前 calib_tries 次**连续**尝试，
        #    因为上电头几帧相机曝光/白平衡还没稳定，很快就能捕获到；
        #    连续失败后再退回降频（每 slow_every 帧试一次），保证空闲开销≈0。
        self._slow_n += 1
        if self.fixed is None and self._slow_n <= c["calib_tries"]:
            pass
        elif self._slow_n % c["slow_every"] != 0 and self.miss > 0:
            self.miss += 1
            self.last_ms = time.ticks_diff(time.ticks_ms(), t0)
            return None

        # 有位置先验（跟踪中 / 有靶心锚点）才允许宽阈值路径
        wide_ok = c["wide_enable"] and (self.last is not None
                                        or anchor is not None)
        cands = []
        over = lambda: _diff_ms(_now_ms(), t0) > budget
        # 搜索中心 = Kalman 预测点（有的话），否则上一帧位置
        cx = cy = None
        if self.kf.init:
            cx, cy = self.kf.px, self.kf.py
        elif self.last is not None:
            cx, cy = self.last
        if cx is not None:
            rad = int(c["search_rad"])
            if self.kf.init:
                rad = int(self.kf.gate_radius(float(c.get("gate_k", 3.0))) + 8.0)
                if rad < c["search_rad"]:
                    rad = int(c["search_rad"])
                if rad > c["reacquire_rad"]:
                    rad = int(c["reacquire_rad"])
            cands = self._scan(img, _roi(cx, cy, rad, w, h), base,
                               c["roi_stride"], wide_ok)
            # 门控失败才扩大到 reacquire_rad，且必须先看预算
            if not cands and not over():
                cands = self._scan(img, _roi(cx, cy, c["reacquire_rad"], w, h),
                                   base, c["roi_stride"], wide_ok)
        if not cands and not over():
            # 有靶心锚点：激光必定在靶纸附近，只在锚点窗口内搜。
            # 首次捕获且没有靶心锚点：分格捕获，每帧只扫一格 160x120，
            # **绝不扫全图**（复杂画面全图 find_blobs 单帧 0.5~2.3s，
            # 会把整条流拖到 0.5fps，这是"板子像死锁"的根因）。
            strict = (self.last is None and anchor is None)
            if anchor is not None:
                ar = c["anchor_rad"]
                cands = self._scan(img, _roi(anchor[0], anchor[1], ar, w, h),
                                   base, 1, wide_ok, strict)
            elif self.last is None and not self.kf.init:
                gw, gh = 160, 120
                cols = max(1, (w + gw - 1) // gw)
                rows = max(1, (h + gh - 1) // gh)
                gi = self._grid_i % (cols * rows)
                self._grid_i += 1
                gx = (gi % cols) * gw
                gy = (gi // cols) * gh
                cands = self._scan(img, (gx, gy, min(gw, w - gx),
                                         min(gh, h - gy)), base, 1, False, strict)
        self.last_cands = len(cands)
        if not cands:
            self.miss += 1
            if self.miss > 8:
                self.last = None
            self.last_ms = time.ticks_diff(time.ticks_ms(), t0)
            return None

        # 3) 选最佳：得分 + 离上一帧/锚点近的加成
        best = None
        for cd in cands:
            # 弱特征（无白核）必须有位置先验才敢用：
            # 实测远处激光 luma 124/warm 64，门框红标签 luma 100/warm 107，
            # 单帧颜色几乎无法区分，只能靠"激光在靶心/上一帧附近"约束。
            if cd["mode"] == "dim":
                if self.last is not None:
                    if _dist(cd["x"], cd["y"], self.last[0],
                             self.last[1]) > c["dim_radius"]:
                        continue
                elif anchor is not None:
                    if _dist(cd["x"], cd["y"], anchor[0],
                             anchor[1]) > c["anchor_rad"]:
                        continue
                else:
                    continue
            sc = cd["score"]
            if self.last is not None:
                d = _dist(cd["x"], cd["y"], self.last[0], self.last[1])
                if d > c["reacquire_rad"]:
                    continue

                sc += max(0.0, 30.0 - d * 0.5)
            elif anchor is not None:
                d = _dist(cd["x"], cd["y"], anchor[0], anchor[1])
                if d > c["anchor_rad"] * 2:
                    continue

                sc += max(0.0, 24.0 - d * 0.08)
            if best is None or sc > best[0]:
                best = (sc, cd)
        if best is None:
            self.miss += 1
            self.last_ms = time.ticks_diff(time.ticks_ms(), t0)
            return None

        cd = best[1]

        # 时序确认：白核是强特征，直接输出；"dim"（无白核、靠红团+对比）
        # 是弱特征，容易被红色杂物骗，必须连续 confirm_frames 次出现在
        # 同一位置附近才采纳。
        # 弱特征不阻塞输出，只压低置信度：首帧 conf 约 0.35，连续命中逐步升到 1。
        # 这样"能看到"和"敢信任"分开，上层（ESP32/PC）用 conf 阈值决定。
        weak_conf = 1.0
        if cd["mode"] != "white":
            if (self._conf_xy is not None
                    and _dist(cd["x"], cd["y"],
                              self._conf_xy[0], self._conf_xy[1])
                    <= c["confirm_rad"]):
                self._conf_n += 1
            else:
                self._conf_n = 1
            self._conf_xy = (cd["x"], cd["y"])
            weak_conf = 0.35 + 0.25 * self._conf_n
            if weak_conf > 1.0:
                weak_conf = 1.0
        else:
            self._conf_n = 0
            self._conf_xy = None

        # 形态校验已在 _scan（白核饱和 + 局部红晕）里完成，这里不再重复统计
        x, y = self._subpixel(img, cd)
        self.last = (x, y)
        self.miss = 0
        if not self.kf.init:
            self.kf.reset(x, y, _now_ms())
        else:
            self.kf.update(x, y)
        self.coast = 0
        conf = cd["contrast"] / float(c["conf_k"])
        if conf > 1.0:
            conf = 1.0
        self.last_ms = time.ticks_diff(time.ticks_ms(), t0)
        # ===== 上电首次视觉标定 =====
        # 完整检测一旦成功（白核+红晕都在，说明是真激光），就把它锁成标定值。
        # 之后每帧只在小窗口里跟踪，开销从几十毫秒降到 4~8ms。
        # 只对强特征（white）做，弱特征（dim）容易锁到红杂物上，不采信。
        if (self.fixed is None and c["auto_calib"] and cd["mode"] == "white"
                and cd["contrast"] > 0):
            self.fixed = (x, y)
            self._verify_fail = 0
            self.just_calibrated = True
        self.mode_hist.append(cd["mode"])
        if len(self.mode_hist) > 10:
            self.mode_hist.pop(0)
        return {"x": x, "y": y, "conf": conf, "contrast": cd["contrast"],
                "warm": cd["warm"], "sat": cd["sat"], "mode": cd["mode"],
                "area": cd["area"], "ms": self.last_ms,
                "pred": [self.kf.px, self.kf.py], "coast": self.coast,
                "budget": self._budget_hit}
