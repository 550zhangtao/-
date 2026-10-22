# laser_openmv.py -- K230 laser spot, article-style full-frame blob
# + online adaptive LAB / size / core thresholds.
#
# Reference: CSDN "K230视觉检测模块识别极小激光优化方案" (2025-07-14)
#   https://blog.csdn.net/m0_74800726/article/details/149333221
#   raw idea: tuned LAB + merge=True + pixels_threshold/area_threshold=10
#             + choose the largest small blob
# OpenMV find_blobs API (MIT): openmv/openmv color-tracking examples.
#
# Adaptive part added 2026-10-22:
#   * every few stable frames, crop the blob and read LAB mean/stdev from
#     img.get_statistics(); slowly adapt L lower bound and b bounds
#   * adapt min_px / max_px from an EMA of the blob area
#   * adapt min white-core from an EMA of the core saturation
#   * if the blob is lost, try a wider threshold every few frames (bounded)
#   * if lost for a long time, reset back to the seed threshold
# All adaptation is clamped; a is kept red-positive, so the threshold cannot
# drift onto white walls or purple cables.

import time

# ============================================================
# 诊断模式：True 时
#   * 关闭先验门控、关闭所有候选过滤（面积/边长/长宽比/fill/peak/core/contrast）
#   * 每帧打印 find_blobs 的原始 blob 数量 + 每个 blob 的 (cx, cy, pixels, area, w, h)
#     + 它会被哪一条规则拒掉
# 用法：把下面改成 True 跑一轮，看清画面里到底有几个亮点，再改回 False。
# ============================================================
DIAG_MODE = False

VERSION = "2026-10-22.7-diag"
API_VERSION = 1
CALIB_FILE = "/sdcard/laser_calib.json"

CFG = {
    "enabled": 1,
    # seed threshold: red laser on a wall
    "lab": (5, 100, 25, 127, -60, 80),
    "min_px": 20,
    "max_px": 320,
    "max_side": 60,
    "aspect_max": 5.0,
    "min_core": 180,
    "x_stride": 2,
    "y_stride": 2,
    "gamma": 0.0,
    "lost_after": 8,
    "calib_warn": 0,
    # ---- adaptive color threshold ----
    "adapt_enable": 0,
    "adapt_after": 3,       # consecutive detections before learning
    "adapt_every": 4,       # learn at most every N frames
    "adapt_alpha": 0.22,    # slow EMA
    "lab_l_min": 5,
    "lab_l_max": 100,
    "lab_a_min": 25,        # keep red-positive; never learn neutral/white
    "lab_b_min": -60,
    "lab_b_max": 80,
    "l_margin": 5,
    "b_margin": 6,
    "max_l_step": 5,        # max L0 change per update
    "max_b_step": 6,
    "min_b_width": 90,
    # ---- adaptive size / core ----
    "area_alpha": 0.25,
    "area_min_px": 20,
    "area_max_px": 50,
    "core_alpha": 0.20,
    "core_floor": 160,
    "core_ceil": 200,
    # ---- lost recovery (bounded, no avalanche) ----
    "relax_enable": 1,
    "relax_after": 2,
    "relax_every": 4,
    "relax_slow": 10,
    "relax_slow_after": 12,
    "relax_margin_l": 12,
    "relax_margin_a": 10,
    "relax_margin_b": 15,
    "relax_core_drop": 35,
    "relax_min_px_drop": 15,
    "reset_after": 20,
    # ---- output stabiliser ----
    # ---- spatial gate: laser is mounted / hand moves continuously ----
    "start_gate": 280,      # first lock around prior (or last)
    "track_gate": 0,        # 0 = 关闭（默认）；开启后只在上一帧位置附近搜
    "loose_after_lock": 0,  # 0 = 锁定后仍用严格门限（推荐）
    "gate_speed_gain": 1.2,
    "gate_grow": 12,
    "gate_max": 220,
    "jump_gate": 110,        # larger jump -> require motion confirmation
    "jump_move_min": 2.0,
    "jump_move_max": 120.0,
    "jump_keep": 2,
    # 本轮只求单帧稳定：默认**不做**多帧跳变确认（pending 那套）
    "jump_confirm": 0,
    # 先验自动降级：检测到的光斑离先验超过这么多像素，就认为先验失效，直接弃用
    "prior_grace": 120,

    # ---- shape / local-contrast gates (OpenCV SimpleBlobDetector idea) ----
    # fill = area/(w*h), aspect = max(w,h)/min(w,h)
    # peak = max luma, core = max min(R,G,B), contrast = peak - ring_mean
    "fg_startup": {"fill": 0.50, "aspect": 2.8, "peak": 200,
                   "core": 210, "warm": 160, "contrast": 60},
    "fg_track": {"fill": 0.42, "aspect": 6.0, "peak": 155,
                 "core": 158, "warm": 110, "contrast": 22},
    "fg_relax": {"fill": 0.40, "aspect": 6.5, "peak": 145,
                 "core": 145, "warm": 90, "contrast": 18},
    "pos_alpha": 0.35,      # only for tiny (<6 px) frame-to-frame motion
    "pos_smooth_gate": 6.0,
    # ---- spatial gate / reacquire ----
    "gate_px": 80,          # base gate around last measured spot
    "gate_speed_gain": 0.8,
    "gate_max": 180,
    "reacquire_after": 3,   # misses before full-frame reacquire
    "reacquire_core": 230,
    "reacquire_area_lo": 0.45,
    "reacquire_area_hi": 2.5,
    "diag_mode": 0,
    # ---- 大块内取亮核（激光打在红色/反光物体上时会和背景连成一大块）----
    "core_fallback": 1,
    "core_lab": (75, 100, -20, 20, -20, 20),   # 近白亮核
    "core_peak_min": 225,      # 大块峰值够亮才值得找核
    "core_min_px": 20,
    "core_max_px": 220,
    "core_max_side": 30,
    "core_topk": 2,            # 最多检查几个大块
    "core_aspect_max": 1.8,    # 亮核必须是近似圆形（实测激光 1.4，反光 2.2，线缆 4.2）
    "core_warm_min": 60,       # 红晕强度下限（激光 96~120，白色反光只有 40~56）
    # 亮核自身的亮度下限（激光 248/250；暗反光只有 148/160，必须拒掉）
    "core_core_min": 200,
    "core_peak_min_sub": 200,
}


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


def _clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


class LaserSpot:
    """Full-frame red/white-core blob detector with online threshold adaptation."""

    def __init__(self, cfg=None):
        self.cfg = dict(CFG)
        if cfg:
            self.cfg.update(cfg)
        c = self.cfg
        self.lab = tuple(int(v) for v in c["lab"])
        self.lab_seed = tuple(int(v) for v in c["lab"])
        self.prior = None
        self.last = None
        self.last_meas = None
        self.last_conf = 0.0
        self.last_px = 0
        self.last_ms = 0
        self.last_budget = False
        self.last_lead = 0
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.mode = "none"
        self.src = "none"
        self.status = "none"
        self.miss = 0
        self.frame = 0
        self.raw_blobs = 0
        self.candidates = 0
        self.last_core = 0
        self.calib_dev = 0.0
        self.need_calib = True
        self.need_recalib = False
        # adaptation state
        self.adapt_n = 0
        self.adapt_skip = 0
        self.area_ema = None
        self.core_ema = None
        self.min_px_eff = float(c["min_px"])
        self.max_px_eff = float(c["max_px"])
        self.core_min_eff = float(c["min_core"])
        self.stable_n = 0
        self.had_lock = False
        self.relaxed = 0
        self.relax_n = 0
        self.last_speed = 0.0
        # ---- 以下 4 个属性曾经只在 reset() 里初始化，导致第一次运行就
        # AttributeError：self.last_step（第 2 帧必读）、self.pending /
        # self.pending_t（出现大跳变时读）、self.rej（_find 里累加）。
        self.last_step = 0.0
        self.pending = None
        self.pending_t = 0
        self.rej = {}
        self._miss_counted = False
        self.prior_degraded = False
        self.core_fallback_used = 0
        self.diag_blobs = []
        # compatibility with old callers
        self.prior_hold = False
        self.prior_anchor = None
        self.last_blob = None

    def set_calib(self, x, y, kx=0.0, ky=0.0, ref=0.0):
        self.prior = (float(x), float(y))
        self.need_calib = False

    def load(self, path=CALIB_FILE):
        try:
            try:
                import ujson as _json
            except Exception:
                import json as _json
            f = open(path)
            try:
                d = _json.loads(f.read())
            finally:
                f.close()
            x = float(d.get("x", -1))
            y = float(d.get("y", -1))
            if x < 0 or y < 0:
                return False
            self.set_calib(x, y)
            return True
        except Exception:
            return False

    def predict(self, s=None):
        if self.last is not None:
            return self.last
        if self.prior is not None:
            return self.prior
        return None

    def lab_i(self):
        return (int(self.lab[0]), int(self.lab[1]), int(self.lab[2]),
                int(self.lab[3]), int(self.lab[4]), int(self.lab[5]))

    def _blob_pixels(self, img, x0, y0, bw, bh, pad=5):
        """只裁剪候选 bbox+外圈，计算 core/peak/ring，避免每帧 614KB 全图 bytearray。"""
        w = img.width()
        h = img.height()
        xa = x0 - pad
        if xa < 0:
            xa = 0
        ya = y0 - pad
        if ya < 0:
            ya = 0
        xb = x0 + bw + pad
        if xb > w:
            xb = w
        yb = y0 + bh + pad
        if yb > h:
            yb = h
        cw = xb - xa
        ch = yb - ya
        if cw <= 0 or ch <= 0:
            return -1, -1, 0
        crop = img.crop(roi=(xa, ya, cw, ch))
        ba = crop.bytearray()
        core = -1
        peak = -1
        warm = -1
        rsum = 0
        rn = 0
        for yy in range(ch):
            gy = ya + yy
            row = yy * cw
            in_y = gy >= y0 and gy < y0 + bh
            for xx in range(cw):
                i = (row + xx) * 2
                v = ba[i] | (ba[i + 1] << 8)
                r = ((v >> 11) & 0x1F) * 8
                g = ((v >> 5) & 0x3F) * 4
                b = (v & 0x1F) * 8
                lu = (r * 77 + g * 150 + b * 29) >> 8
                if in_y:
                    gx = xa + xx
                    if gx >= x0 and gx < x0 + bw:
                        mn = r if r < g else g
                        if b < mn:
                            mn = b
                        if mn > core:
                            core = mn
                        if lu > peak:
                            peak = lu
                        wm = r - (g if g > b else b)
                        if wm > warm:
                            warm = wm
                        continue
                rsum += lu
                rn += 1
        ring = rsum // rn if rn else 0
        return core, peak, ring, warm

    def _blob_core(self, img, x0, y0, bw, bh):
        return self._blob_pixels(img, x0, y0, bw, bh, 0)[0]


    def _diag_reason(self, a, bw, bh, min_px, max_px, max_side, aspect, fg,
                     fill, asp, peak, core, con, warm, ring,
                     bcx, bcy, ref, gate):
        """诊断用：返回这个 blob 会被哪一条规则拒掉（OK = 能通过）。"""
        if a < min_px:
            return "AREA_LOW(a<min_px)"
        if a > max_px:
            return "AREA_HIGH(a>max_px)"
        if bw <= 0 or bh <= 0 or bw > max_side or bh > max_side:
            return "SIDE(%dx%d>max_side)" % (bw, bh)
        big = bw if bw > bh else bh
        sml = bh if bw > bh else bw
        if sml and big > sml * aspect:
            return "ASPECT(%.1f>%.1f)" % (big / float(sml), aspect)
        if ref is not None and gate > 0:
            ddx = bcx - ref[0]
            ddy = bcy - ref[1]
            if ddx * ddx + ddy * ddy > gate * gate:
                return "PRIOR_GATE(%.0fpx>%d)" % ((ddx * ddx + ddy * ddy) ** 0.5, int(gate))
        if fg is not None:
            if fill < float(fg["fill"]):
                return "FILL(%.2f<%.2f)" % (fill, float(fg["fill"]))
            if asp > float(fg["aspect"]):
                return "ASPECT(%.1f>%.1f)" % (asp, float(fg["aspect"]))
            if peak < int(fg["peak"]):
                return "PEAK(%d<%d)" % (peak, int(fg["peak"]))
            white_ok = core >= int(fg["core"])
            dark_ok = (ring < 110 and warm >= int(fg["warm"])
                       and peak >= int(fg["peak"]) and con >= int(fg["contrast"]))
            if not (white_ok or dark_ok):
                return "CORE(white %d<%d, dark warm=%d ring=%d)" % (
                    core, int(fg["core"]), warm, ring)
        if fg is not None and con < int(fg["contrast"]):
            return "CONTRAST(%d<%d)" % (con, int(fg["contrast"]))
        return "OK"

    def _core_fallback(self, img, use_gate=True):
        """正常路径找不到时，在"太大的红块"内部用严格亮核阈值找光斑。

        场景：激光打在红色物体/反光面上，find_blobs(merge=True) 会把激光和
        背景连成一个几千像素的大块，于是被 AREA_HIGH 整块拒掉。实测这种大块
        内部仍能看到一个 40~90 px 的近白亮核（就是激光），用 roi 限定在这个
        bbox 里再扫一次即可把它单独取出来。

        只在失败帧调用，一次 11~14ms；成功帧不付这个代价。
        """
        c = self.cfg
        try:
            blobs = img.find_blobs([self.lab_i()], merge=True,
                                   pixels_threshold=max(6, int(c["min_px"]) // 2),
                                   area_threshold=max(6, int(c["min_px"]) // 2),
                                   x_stride=int(c["x_stride"]),
                                   y_stride=int(c["y_stride"]))
        except Exception:
            return 0, 0, None
        if not blobs:
            return 0, 0, None
        blobs = sorted(blobs, key=lambda z: -z.pixels())
        best = None
        # 位置门控：锁定后只在上一帧位置附近取核。否则两个静止亮点会各占一半
        # （实测激光 279,280 与暗反光 200,350 相距 141px，全图各 50%）。
        rp = (self.last_meas if self.last_meas is not None else self.prior) \
            if use_gate else None
        rg = 0.0
        if rp is not None:
            if self.had_lock:
                rg = float(c.get("track_gate", 0))
            elif self.prior is not None:
                rg = float(c.get("start_gate", 0))
        checked = 0
        for b in blobs:
            if checked >= int(c["core_topk"]):
                break
            a0 = int(b.pixels())
            if a0 <= int(c["max_px"]):
                continue                      # 正常路径已经能处理，不重复
            x0, y0, bw, bh = int(b.x()), int(b.y()), int(b.w()), int(b.h())
            core, peak, ring, warm = self._blob_pixels(img, x0, y0, bw, bh, 5)
            if peak < int(c["core_peak_min"]):
                continue                      # 这块不够亮，里面不可能有激光
            checked += 1
            try:
                subs = img.find_blobs([self.lab_core()], merge=True,
                                      pixels_threshold=int(c["core_min_px"]),
                                      area_threshold=int(c["core_min_px"]),
                                      x_stride=1, y_stride=1,
                                      roi=(x0, y0, bw, bh))
            except Exception:
                subs = []
            for sb in subs:
                sa = int(sb.pixels())
                sw, sh = int(sb.w()), int(sb.h())
                if sa < int(c["core_min_px"]) or sa > int(c["core_max_px"]):
                    continue
                if sw > int(c["core_max_side"]) or sh > int(c["core_max_side"]):
                    continue
                sc1, spe1, sri1, swa1 = self._blob_pixels(img, sb.x(), sb.y(),
                                                          sw, sh, 3)
                big_s = sw if sw > sh else sh
                sml_s = sh if sw > sh else sw
                asp1 = float(big_s) / float(sml_s) if sml_s else 99.0
                # 近似圆形：激光点 1.2~1.6；反光条/线缆 2.2~4.2（实测）
                if asp1 > float(c.get("core_aspect_max", 2.0)):
                    continue
                if swa1 < int(c.get("core_warm_min", 60)):
                    continue
                if int(sc1) < int(c.get("core_core_min", 200)):
                    continue
                if int(spe1) < int(c.get("core_peak_min_sub", 200)):
                    continue
                if rp is not None and rg > 0:
                    gdx = float(sb.cx()) - rp[0]
                    gdy = float(sb.cy()) - rp[1]
                    if gdx * gdx + gdy * gdy > rg * rg:
                        continue          # 超出跟踪门控，不取这个核
                con1 = int(spe1) - int(sri1)
                fil1 = float(sa) / float(sw * sh) if sw * sh else 0.0
                sc = (2.0 * (sc1 / 255.0) + 1.4 * (spe1 / 255.0)
                      + 1.3 * (min(con1, 150) / 150.0)
                      + 1.2 * fil1 + 0.7 / asp1
                      + 0.8 * (min(max(swa1, 0), 200) / 200.0))
                # 面积越大越像激光（点状光斑 40~90px）
                sc += 0.5 * min(float(sa), 120.0) / 120.0
                # 只做**排序先验**：离上一帧位置近的加分（不做多帧确认）
                rp = self.last_meas if self.last_meas is not None else self.prior
                if rp is not None:
                    ddx = float(sb.cx()) - rp[0]
                    ddy = float(sb.cy()) - rp[1]
                    dd = (ddx * ddx + ddy * ddy) ** 0.5
                    sc += 1.2 * max(0.0, 1.0 - dd / 80.0)
                    sc -= 0.5 * min(dd, 200.0) / 200.0
                if best is None or sc > best[0]:
                    best = (sc, sb, sc1, spe1, con1, fil1, asp1, sa)
        return len(blobs), (1 if best else 0), best

    def lab_core(self):
        lc = self.cfg["core_lab"]
        return (int(lc[0]), int(lc[1]), int(lc[2]), int(lc[3]),
                int(lc[4]), int(lc[5]))

    def _find(self, img, lab, min_px, max_px, min_core,
              max_side, aspect, fg=None, ref=None, gate=0.0):
        c = self.cfg
        diag = bool(c.get("diag_mode", 0))
        # 诊断模式下把 find_blobs 的门限也放到最低，先把"原始 blob"看清楚
        pth = 4 if diag else 10
        ath = 4 if diag else 10
        try:
            blobs = img.find_blobs([lab], merge=True,
                                   pixels_threshold=pth, area_threshold=ath,
                                   x_stride=int(c["x_stride"]),
                                   y_stride=int(c["y_stride"]))
        except Exception:
            blobs = []
        raw = len(blobs)
        self.rej["raw"] += raw
        cands = 0
        best = None
        if diag:
            self.diag_blobs = []
            print("[DIAG] lab=%s min_px=%s max_px=%s max_side=%s aspect=%s "
                  "core=%s prior=%s gate=%s raw=%d"
                  % (lab, int(min_px), int(max_px), int(max_side),
                     round(float(aspect), 2), int(min_core), ref,
                     int(gate), raw))
        w = img.width()
        h = img.height()
        for b in blobs:
            try:
                a = int(b.pixels())
                bw = int(b.w())
                bh = int(b.h())
                x0 = int(b.x())
                y0 = int(b.y())
                bcx = float(b.cx())
                bcy = float(b.cy())
            except Exception:
                continue
            # ---- 诊断模式：不做任何过滤，先算指标再打印 ----
            if diag:
                core_d, peak_d, ring_d, warm_d = self._blob_pixels(
                    img, x0, y0, bw, bh, 5)
                fill_d = float(a) / float(bw * bh) if bw * bh else 0.0
                big_d = bw if bw > bh else bh
                sml_d = bh if bh > bw else bw
                asp_d = float(big_d) / float(sml_d) if sml_d else 0.0
                con_d = int(peak_d) - int(ring_d)
                why = self._diag_reason(a, bw, bh, min_px, max_px, max_side,
                                        aspect, fg, fill_d, asp_d, peak_d,
                                        core_d, con_d, warm_d, ring_d,
                                        bcx, bcy, ref, gate)
                self.diag_blobs.append(
                    {"cx": bcx, "cy": bcy, "px": a, "area": bw * bh,
                     "w": bw, "h": bh, "fill": fill_d, "asp": asp_d,
                     "peak": peak_d, "core": core_d, "con": con_d,
                     "warm": warm_d, "why": why})
                print("  [DIAG] blob cx=%.1f cy=%.1f px=%d area=%d w=%d h=%d "
                      "fill=%.2f asp=%.2f peak=%d core=%d con=%d warm=%d -> %s"
                      % (bcx, bcy, a, bw * bh, bw, bh, fill_d, asp_d,
                         peak_d, core_d, con_d, warm_d, why))
                if why == "OK":
                    cands += 1
                    self.rej["ok"] += 1
                else:
                    key = why.lower()
                    if key in self.rej:
                        self.rej[key] += 1
                # 诊断模式下过滤已关闭，但**仍然要选一个 best**，否则永远出不了
                # 结果（旧实现这里直接 continue，导致 cand>0 却一直 laser -）。
                sc = (2.0 * (core_d / 255.0) + 1.4 * (peak_d / 255.0)
                      + 1.3 * (min(con_d, 150) / 150.0)
                      + 1.2 * fill_d + 0.7 / (asp_d if asp_d else 1.0)
                      + 0.8 * (min(max(warm_d, 0), 200) / 200.0))
                if self.last_meas is not None:
                    ddx = bcx - self.last_meas[0]
                    ddy = bcy - self.last_meas[1]
                    dd = (ddx * ddx + ddy * ddy) ** 0.5
                    sc += 1.2 * max(0.0, 1.0 - dd / 80.0) - 0.5 * min(dd, 200.0) / 200.0
                if best is None or sc > best[0]:
                    best = (sc, b, core_d, peak_d, con_d, fill_d, asp_d, a)
                if c.get("diag_draw", 0):
                    try:
                        img.draw_rectangle(x0, y0, bw, bh,
                                           color=(0, 255, 0), thickness=1)
                    except Exception:
                        pass
                continue

            if a < min_px or a > max_px:
                self.rej["area"] += 1
                continue
            if bw <= 0 or bh <= 0 or bw > max_side or bh > max_side:
                self.rej["side"] += 1
                continue
            big = bw if bw > bh else bh
            sml = bh if bh > bw else bw
            if sml <= 0 or big > sml * aspect:
                self.rej["aspect"] += 1
                continue
            if ref is not None and gate > 0.0:
                ddx = bcx - ref[0]
                ddy = bcy - ref[1]
                if ddx * ddx + ddy * ddy > gate * gate:
                    self.rej["gate"] += 1
                    continue
            core, peak, ring, warm = self._blob_pixels(
                img, x0, y0, bw, bh, 5)
            fill = float(a) / float(bw * bh) if bw * bh else 0.0
            asp = float(big) / float(sml)
            contrast = int(peak) - int(ring)
            if fg is not None:
                if fill < float(fg["fill"]):
                    self.rej["fill"] += 1
                    continue
                if asp > float(fg["aspect"]):
                    self.rej["aspect"] += 1
                    continue
                if peak < int(fg["peak"]):
                    self.rej["peak"] += 1
                    continue
                # 白墙：白核；黑背景：允许纯红点（core 低但 warm/contrast 高）
                white_ok = core >= int(fg["core"])
                dark_ok = (ring < 110 and warm >= int(fg["warm"])
                           and peak >= int(fg["peak"])
                           and contrast >= int(fg["contrast"]))
                if not (white_ok or dark_ok):
                    self.rej["core"] += 1
                    continue
            if contrast < int(fg["contrast"]):
                self.rej["contrast"] += 1
                continue
            # OpenCV-SimpleBlobDetector-like score: bright compact high-contrast
            sc = (2.0 * (core / 255.0) + 1.4 * (peak / 255.0)
                  + 1.3 * (min(contrast, 150) / 150.0)
                  + 1.2 * fill + 0.7 / asp
                  + 0.8 * (min(max(warm, 0), 200) / 200.0))
            if ref is not None:
                dist = ((bcx - ref[0]) ** 2 + (bcy - ref[1]) ** 2) ** 0.5
                sc += 0.9 * max(0.0, 1.0 - dist / 120.0)
                sc -= 0.4 * min(dist, 220.0) / 220.0
            if self.area_ema:
                sc -= 0.6 * abs(a - self.area_ema) / max(1.0, self.area_ema)
            cands += 1
            self.rej["ok"] += 1
            if best is None or sc > best[0]:
                best = (sc, b, core, peak, contrast, fill, asp, a)
        return raw, cands, best

    def _adapt_color(self, img, b):
        """用高置信 blob 的裁剪 LAB 统计，缓慢更新 L0 / b0 / b1。"""
        c = self.cfg
        try:
            st = img.crop(roi=(int(b.x()), int(b.y()),
                               int(b.w()), int(b.h()))).get_statistics()
            lm = float(st.l_mean())
            ls = float(st.l_stdev())
            bm = float(st.b_mean())
            bs = float(st.b_stdev())
        except Exception:
            self.adapt_skip += 1
            return False
        cur_l0 = float(self.lab[0])
        # exposure / lighting lower bound: cover the red halo, not just core
        want_l0 = lm - 1.8 * ls - float(c["l_margin"])
        want_l0 = _clamp(want_l0, float(c["lab_l_min"]), 40.0)
        step = _clamp(want_l0 - cur_l0,
                      -float(c["max_l_step"]), float(c["max_l_step"]))
        new_l0 = cur_l0 + float(c["adapt_alpha"]) * step
        # b: color temperature / white balance
        want_b0 = bm - 1.8 * bs - float(c["b_margin"])
        want_b1 = bm + 1.8 * bs + float(c["b_margin"])
        want_b0 = _clamp(want_b0, float(c["lab_b_min"]), -5.0)
        want_b1 = _clamp(want_b1, 5.0, float(c["lab_b_max"]))
        if want_b1 - want_b0 < float(c["min_b_width"]):
            mid = 0.5 * (want_b0 + want_b1)
            half = 0.5 * float(c["min_b_width"])
            want_b0 = mid - half
            want_b1 = mid + half
        cur_b0 = float(self.lab[4])
        cur_b1 = float(self.lab[5])
        step_b0 = _clamp(want_b0 - cur_b0,
                         -float(c["max_b_step"]), float(c["max_b_step"]))
        step_b1 = _clamp(want_b1 - cur_b1,
                         -float(c["max_b_step"]), float(c["max_b_step"]))
        new_b0 = cur_b0 + float(c["adapt_alpha"]) * step_b0
        new_b1 = cur_b1 + float(c["adapt_alpha"]) * step_b1
        self.lab = (int(round(new_l0)), int(c["lab_l_max"]),
                    int(c["lab_a_min"]), 127,
                    int(round(new_b0)), int(round(new_b1)))
        self.adapt_n += 1
        return True

    def _adapt_size_core(self, area, core):
        c = self.cfg
        aa = float(c["area_alpha"])
        ca = float(c["core_alpha"])
        if self.area_ema is None:
            self.area_ema = float(area)
            self.core_ema = float(core)
        else:
            self.area_ema = (1.0 - aa) * self.area_ema + aa * float(area)
            self.core_ema = (1.0 - ca) * self.core_ema + ca * float(core)
        # size floor only moves DOWN (helps small/dim laser), never above 50
        if self.area_ema < 100.0:
            self.min_px_eff = _clamp(self.area_ema * 0.60,
                                     float(c["area_min_px"]), 50.0)
        else:
            self.min_px_eff = 50.0
        self.max_px_eff = _clamp(self.area_ema * 2.5, 250.0, 320.0)
        # core floor only moves DOWN (helps dim core), never above 200
        self.core_min_eff = _clamp(self.core_ema * 0.85,
                                   float(c["core_floor"]), float(c["core_ceil"]))

    def _relaxed_lab(self):
        c = self.cfg
        l0 = _clamp(float(self.lab[0]) - float(c["relax_margin_l"]),
                    14.0, 70.0)
        a0 = _clamp(float(self.lab[2]) - float(c["relax_margin_a"]),
                    12.0, 127.0)
        b0 = _clamp(float(self.lab[4]) - float(c["relax_margin_b"]),
                    -55.0, 60.0)
        b1 = _clamp(float(self.lab[5]) + float(c["relax_margin_b"]),
                    -55.0, 75.0)
        return (int(l0), 100, int(a0), 127, int(b0), int(b1))

    def sample(self, img, s=None):
        c = self.cfg
        t0 = _now_ms()
        self.frame += 1
        self.last_ms = 0
        self.last_budget = False
        self.raw_blobs = 0
        self.candidates = 0
        self.relaxed = 0
        self.rej = {"raw": 0, "gate": 0, "area": 0, "side": 0,
                    "aspect": 0, "core": 0, "fill": 0, "peak": 0,
                    "contrast": 0, "ok": 0, "jump": 0}
        if not c.get("enabled", 1):
            self.mode = "none"
            self.src = "none"
            self.status = "none"
            return None
        if c.get("gamma", 0.0) and float(c["gamma"]) > 1.0:
            try:
                img.gamma_corr(float(c["gamma"]))
            except Exception:
                pass
        # ---- normal threshold (article selection, no spatial gate) ----
        scan_min_px = self.min_px_eff
        scan_min_core = self.core_min_eff
        scan_max_side = int(c["max_side"])
        scan_aspect = float(c["aspect_max"])
        scan_max_px = self.max_px_eff
        # 始终使用严格门限（原来"锁定后切 fg_track 放宽"是抖动的根源之一：
        # peak 200->155 会让远处暗反光(peak≈160)也通过，和激光抢位置）。
        # 真正难检的情况现在由 _core_fallback 兜底，不需要靠放宽门限提召回。
        if not c.get("loose_after_lock", 0):
            scan_min_px = max(scan_min_px, 20.0)
            scan_min_core = max(scan_min_core, 220.0)
            scan_max_side = 40
            scan_aspect = 3.0
            scan_max_px = min(scan_max_px, 320.0)
        fg = c["fg_track"] if (c.get("loose_after_lock", 0) and self.had_lock) \
            else c["fg_startup"]
        # Cold start: only the prior area, to reject the static red LED.
        # After the first lock: full-frame, but every big jump must pass the
        # pending/motion confirmation below.
        # 冷启动：先验 220px 内（覆盖当前 (321,78)，但拒掉底左 (198,434)）；
        # 锁定后全图 + 大跳变确认。
        ref = None
        gate = 0.0
        if not self.had_lock and self.prior is not None:
            ref = self.prior
            gate = float(c["start_gate"])
        # ---- 跟踪门控（用已有的 track_gate，之前一直没生效）----
        # 锁定后先只在上一帧位置 track_gate 内找，避免在两个静止亮点之间来回跳
        # （实测激光 279,280 与反光 200,350 相距 141px，全图搜索时各占一半）。
        # 门控内找不到会立刻回退全图，所以不会因为移动快而丢目标。
        if self.had_lock and self.last_meas is not None and \
                int(c.get("track_gate", 0)) > 0:
            tg = float(c["track_gate"])
            raw, cand, best = self._find(
                img, self.lab_i(), scan_min_px, scan_max_px,
                scan_min_core, scan_max_side, scan_aspect, fg,
                self.last_meas, tg)
            self.raw_blobs = raw
            self.candidates = cand
            self._path = "gate"
            if best is None:
                raw, cand, best = self._find(
                    img, self.lab_i(), scan_min_px, scan_max_px,
                    scan_min_core, scan_max_side, scan_aspect, fg,
                    ref if gate > 0 else None, gate)
                self.raw_blobs += raw
                self.candidates += cand
                self.gated_miss = getattr(self, "gated_miss", 0) + 1
                self._path = "full"
            else:
                self.gated_miss = 0
        else:
            raw, cand, best = self._find(
                img, self.lab_i(), scan_min_px, scan_max_px,
                scan_min_core, scan_max_side, scan_aspect, fg,
                ref if gate > 0 else None, gate)
            self.raw_blobs = raw
            self.candidates = cand
            self._path = "start"

        # ---- 先验自动降级（本轮新增）----
        # 旧逻辑：冷启动只在先验 start_gate 内找，先验偏了就把真光斑永久拒掉。
        # 新逻辑：门控内没找到 -> 立刻无门控再找一遍；找到了就说明先验已失效，
        #         直接把先验丢掉（prior=None），后面完全按阈值检测走。
        if (best is None and not self.had_lock and self.prior is not None
                and gate > 0):
            r2, c2, b2 = self._find(
                img, self.lab_i(), scan_min_px, scan_max_px,
                scan_min_core, scan_max_side, scan_aspect, fg, None, 0.0)
            self.raw_blobs += r2
            self.candidates += c2
            if b2 is not None:
                best = b2
                self.prior_degraded = True
                bx2 = float(b2[1].cx())
                by2 = float(b2[1].cy())
                print("laser: 先验 %s 附近没找到光斑，已在全图找到 (%.0f,%.0f)，"
                      "自动弃用先验" % (self.prior, bx2, by2))
                self.prior = None
                self.need_calib = True
                self.calib_dev = 0.0

        # ---- 大块内取亮核（正常路径失败时的一次单帧补救）----
        if best is None and c.get("core_fallback", 1):
            r3, c3, b3 = self._core_fallback(img, use_gate=True)
            if b3 is None:
                # 门控内没取到 -> 全图再试一次，保证不丢帧
                r3b, c3b, b3b = self._core_fallback(img, use_gate=False)
                r3 += r3b
                c3 += c3b
                b3 = b3b
            self.raw_blobs += r3
            self.candidates += c3
            if b3 is not None:
                best = b3
                self.core_fallback_used = 1
                self.core_fb_n = getattr(self, "core_fb_n", 0) + 1
                if self.core_fb_n <= 3 or (c.get("diag_mode", 0)):
                    print("laser: 大块内取到亮核 (%.0f,%.0f) px=%d"
                          % (b3[1].cx(), b3[1].cy(), b3[7]))
            else:
                self.core_fallback_used = 0
        else:
            self.core_fallback_used = 0

        # ---- lost recovery: bounded wider threshold ----
        if best is None and c.get("relax_enable", 1) and self.had_lock:
            self.miss += 1
            if self.miss >= int(c["relax_after"]):
                every = int(c["relax_every"])
                if self.miss >= int(c["relax_slow_after"]):
                    every = int(c["relax_slow"])
                if self.frame % every == 0:
                    rlab = self._relaxed_lab()
                    rmin = _clamp(self.min_px_eff - float(c["relax_min_px_drop"]),
                                  float(c["area_min_px"]), 120.0)
                    rcore = _clamp(self.core_min_eff - float(c["relax_core_drop"]),
                                   150.0, 220.0)
                    r2, c2, b2 = self._find(
                        img, rlab, rmin, self.max_px_eff + 50.0,
                        rcore, int(c["max_side"]) + 8,
                        float(c["aspect_max"]) + 0.2, c["fg_relax"])
                    self.raw_blobs += r2
                    self.candidates += c2
                    if b2 is not None:
                        best = b2
                        self.relaxed = 1
                        self.relax_n += 1
            # above already counted one miss; normal path must not count again
            self._miss_counted = True
        else:
            self._miss_counted = False

        self.last_ms = _diff_ms(_now_ms(), t0)

        if best is None:
            if not getattr(self, "_miss_counted", False):
                self.miss += 1
            self._miss_counted = False
            self.stable_n = 0
            if self.miss >= int(c["lost_after"]):
                self.need_recalib = True
            if self.miss >= int(c["reset_after"]):
                # long loss: go back to the safe seed, keep had_lock for reacquire
                self.lab = self.lab_seed
                self.area_ema = None
                self.core_ema = None
                self.min_px_eff = float(c["min_px"])
                self.max_px_eff = float(c["max_px"])
                self.core_min_eff = float(c["min_core"])
            self.last = self.prior
            self.mode = "prior" if self.prior is not None else "none"
            self.src = self.mode
            self.status = "lost" if self.miss >= int(c["lost_after"]) else self.mode
            return None
        self._miss_counted = False

        score, b, core, peak, contrast, fill, asp, a = best
        try:
            raw_x = float(b.cx())
            raw_y = float(b.cy())
        except Exception:
            return None

        # ---- 先验不一致 -> 直接弃用（不再拿错误先验参与评分）----
        if self.prior is not None:
            dpx = raw_x - self.prior[0]
            dpy = raw_y - self.prior[1]
            dprior = (dpx * dpx + dpy * dpy) ** 0.5
            if dprior > float(c.get("prior_grace", 120)):
                print("laser: 实测 (%.0f,%.0f) 离先验 %s 有 %.0fpx > %d，"
                      "弃用先验" % (raw_x, raw_y, self.prior, dprior,
                                    int(c.get("prior_grace", 120))))
                self.calib_dev = dprior
                self.prior = None
                self.need_calib = True
                self.prior_degraded = True

        # 大跳变的多帧确认（pending 那套）默认**关闭**（CFG["jump_confirm"]=0）：
        # 它会吃掉"单帧稳定"，而且跳一下就要连续 2 帧代价，表现为大量 laser -。
        # 需要时把 jump_confirm 设 1 可恢复原行为。
        if self.last_meas is not None and int(c.get("jump_confirm", 0)):

            jdx = raw_x - self.last_meas[0]
            jdy = raw_y - self.last_meas[1]
            jdist = (jdx * jdx + jdy * jdy) ** 0.5
            if jdist > float(c["jump_gate"]):
                if (self.pending is not None
                        and (self.frame - self.pending_t) <= int(c["jump_keep"])):
                    pdx = raw_x - self.pending[0]
                    pdy = raw_y - self.pending[1]
                    pd = (pdx * pdx + pdy * pdy) ** 0.5
                    if float(c["jump_move_min"]) < pd <= float(c["jump_move_max"]):
                        self.pending = None
                    else:
                        self.rej["jump"] += 1
                        self.pending = (raw_x, raw_y)
                        self.pending_t = self.frame
                        self.miss += 1
                        self.stable_n = 0
                        self.mode = "none"
                        self.src = "none"
                        self.status = "none"
                        return None
                else:
                    self.rej["jump"] += 1
                    self.pending = (raw_x, raw_y)
                    self.pending_t = self.frame
                    self.miss += 1
                    self.stable_n = 0
                    self.mode = "none"
                    self.src = "none"
                    self.status = "none"
                    return None
            else:
                self.pending = None

        # tiny-motion smoothing only; fast movement passes through
        cx, cy = raw_x, raw_y
        if self.last_meas is not None:
            dx = raw_x - self.last_meas[0]
            dy = raw_y - self.last_meas[1]
            if dx * dx + dy * dy < float(c["pos_smooth_gate"]) ** 2:
                k = float(c["pos_alpha"])
                cx = self.last_meas[0] + k * dx
                cy = self.last_meas[1] + k * dy

        if self.last_meas is not None:
            stx = raw_x - self.last_meas[0]
            sty = raw_y - self.last_meas[1]
            self.last_step = 0.7 * self.last_step + 0.3 * ((stx * stx + sty * sty) ** 0.5)
        else:
            self.last_step = 0.0
        area_ref = self.area_ema if self.area_ema is not None else float(a)
        self.last_meas = (raw_x, raw_y)
        self.last = (cx, cy)
        self.last_conf = _clamp(score / 6.0, 0.2, 1.0)
        self.last_px = a
        self.last_core = core
        self.last_blob = {"x": cx, "y": cy, "score": score, "area": a,
                          "fill": fill, "peak": peak, "contrast": contrast}
        self.miss = 0
        self.need_recalib = False
        self.had_lock = True
        self.stable_n += 1
        self._adapt_size_core(a, core)
        grow_ok = (a <= area_ref * 1.25 and a >= area_ref * 0.65)
        if (c.get("adapt_enable", 1) and self.stable_n >= int(c["adapt_after"])
                and self.frame % int(c["adapt_every"]) == 0
                and core >= 225 and a >= 45 and grow_ok
                and cand == 1 and score >= 4.5):
            self._adapt_color(img, b)
        if self.prior is not None:
            dxp = cx - self.prior[0]
            dyp = cy - self.prior[1]
            self.calib_dev = (dxp * dxp + dyp * dyp) ** 0.5
        self.mode = "blob"
        self.src = "measured"
        self.status = "ok"
        _path = ("core" if getattr(self, "core_fallback_used", 0)
                 else getattr(self, "_path", "?"))
        return {"x": cx, "y": cy, "conf": self.last_conf, "mode": "blob",
                "path": _path,
                "src": "measured", "status": "ok", "ms": self.last_ms,
                "n": a, "budget": 1 if self.last_ms > 35 else 0,
                "vx": 0.0, "vy": 0.0, "core": core, "lead": 0,
                "bg": 0.0, "lab": self.lab, "adapt": self.adapt_n,
                "relaxed": self.relaxed, "min_px": int(self.min_px_eff),
                "core_min": int(self.core_min_eff), "score": score,
                "fill": fill, "peak": peak, "contrast": contrast,
                "aspect": asp, "diag": self.rej}

    def reset(self):
        c = self.cfg
        self.lab = self.lab_seed
        self.area_ema = None
        self.core_ema = None
        self.min_px_eff = float(c["min_px"])
        self.max_px_eff = float(c["max_px"])
        self.core_min_eff = float(c["min_core"])
        self.adapt_n = 0
        self.stable_n = 0
        self.had_lock = False
        self.last_speed = 0.0
        self.last_step = 0.0
        self.pending = None
        self.pending_t = 0
        self.rej = {}
        self.last = self.prior
        self.last_meas = None
        self.last_conf = 0.0
        self.last_px = 0
        self.last_ms = 0
        self.miss = 0
        self.last_core = 0
        self.mode = "prior" if self.prior is not None else "none"
        self.src = self.mode
        self.status = self.mode


LaserAim = LaserSpot


class AimFilter:
    """激光-靶心相对误差的 α-β + 前瞻（与原模块兼容）。"""

    def __init__(self, alpha=0.5, beta=0.08, lead_ms=50, v_max=2000.0):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lead_ms = float(lead_ms)
        self.v_max = float(v_max)
        self.x = None
        self.y = None
        self.vx = 0.0
        self.vy = 0.0
        self.t = 0

    def reset(self):
        self.x = None
        self.y = None
        self.vx = 0.0
        self.vy = 0.0
        self.t = 0

    def update(self, bx, by, lx, ly, t):
        ax = float(bx) - float(lx)
        ay = float(by) - float(ly)
        if self.x is None:
            self.x = ax
            self.y = ay
            self.vx = 0.0
            self.vy = 0.0
            self.t = t
            return (ax, ay, ax, ay)
        dt = (float(t) - self.t) / 1000.0
        if dt < 0.02:
            dt = 0.02
        elif dt > 0.5:
            dt = 0.5
        px = self.x + self.vx * dt
        py = self.y + self.vy * dt
        rx = ax - px
        ry = ay - py
        self.x = px + self.alpha * rx
        self.y = py + self.alpha * ry
        self.vx += self.beta * rx / dt
        self.vy += self.beta * ry / dt
        sp = (self.vx * self.vx + self.vy * self.vy) ** 0.5
        if sp > self.v_max:
            k = self.v_max / sp
            self.vx *= k
            self.vy *= k
        self.t = float(t)
        lead = self.lead_ms / 1000.0
        return (self.x, self.y, self.x + self.vx * lead,
                self.y + self.vy * lead)

    def get(self):
        return (self.x, self.y, self.vx, self.vy)
