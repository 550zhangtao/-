# -*- coding: utf-8 -*-
"""
laser_aim.py —— 激光光斑：标定先验 + 25×25 微窗亚像素质心（K230 板端）
=====================================================================

设计（2026-10-22 重构，替换 board/laser_spot.py）
------------------------------------------------
旧实现是"每帧/每 5 帧在画面里找光斑"：LAB 阈值 + find_blobs + 多套候选打分 +
全图分格 + 重捕 + 看门狗，1200 行状态机，仍然会被红色杂物/光照骗。
新实现把问题拆成两半：

1. **标定**（PC 端做一次，见 `k230/scripts/calib_laser.py`）
   - 相机与激光笔刚性固定在云台上 → 光斑在图像里的位置是**可标定的物理量**；
   - **当前硬件：激光常亮、GPIO5 已弃用（2026-10-22 用户确认）**，所以 PC 端
     默认走"抓一帧 → LAB 红点候选 → 鼠标点一下光斑"；
   - 如果以后激光接到可控开关，可用 `--pair` 做 `|ON - OFF|` 主动差分标定
     （背景/红物/环境光全部抵消，最稳）；
   - 多个距离标定后拟合 `spot = c + k·s`（s=靶纸表观边长，pose 每帧给），
     写 `/sdcard/laser_calib.json`。

2. **运行时**（本文件）
   - 由标定值预测光斑位置，只在其 ±rad 像素的微窗内做一次**加权质心**；
   - 25×25 微窗 ≈ **1 ms**，49×49 ≈ 4 ms（本机 K230 实测，见下）；
   - 找不到就退回标定/预测位置（`src=calib`），**绝不扫全图、绝不重捕**；
   - 连续丢失时只把窗口放大一次（rad_max），仍找不到就报 `lost`。

为什么不用 find_blobs
---------------------
本机 CanMV v1.4.3（K230）实测：
    find_blobs(48×48 ROI)   ≈ 11~14 ms   （固定开销，和 ROI 大小关系不大）
    find_blobs(160×120)     ≈ 12~14 ms
    find_blobs(640×480)     ≈ 22 ms
    img.difference() 全图   ≈ 7 ms
    img.binary() 全图       ≈ 16 ms
    img.mean_pool(4,4)      ≈ 3 ms
    纯 Python bytearray 扫描 25×25 ≈ 1 ms、49×49 ≈ 4 ms
所以"每帧识别"用 find_blobs 是 12 ms 起步，而微窗扫描只要 1 ms；标定用差分
放在 PC 上做（OpenCV，一次几十 ms 无所谓）。旧文档说"全图 find_blobs 0.5~2.3s"
是误记，实际 22 ms，但 12 ms 固定开销仍然不适合每帧。

坐标与协议
----------
- 输入 `img` 必须是 chn1 的 RGB565 原图（640×480），坐标即原图坐标；
- 输出 dict 字段与旧 `$LS` 完全兼容：
      x, y, conf, mode, src("measured"/"calib"), status("ok"/"calib"/"lost"), ms
- `src=1` 仅表示**本帧微窗实测成功**；`src=2` 是标定/预测位置。

标定文件
--------
    {"x": 408.2, "y": 277.9, "kx": 0.0, "ky": 0.0, "ref": 0.0, "ts": ...}
`x,y` 是标定时的光斑像素坐标；`kx,ky` 是每像素靶纸表观尺寸变化对应的
光斑位移（`spot = c + k·(s-ref)`），默认 0（等价于旧 fixed 模式）。

注意
----
- 没有标定文件时 `sample()` 返回 None，上层应关闭 `$LS`（不要临场上电自动标定）；
- 相机/激光支架松动后必须重标（`calib_dev > calib_warn` 会置 `need_recalib`）；
- 本文件只依赖 `time` 和 `img.bytearray()`，板端/PC 都可 import（PC 只用于读
  标定文件/做单测，`sample()` 需要一个有 `bytearray()/width()/height()` 的图对象）。
"""

import time

VERSION = "2026-10-22.2"
API_VERSION = 1
CALIB_FILE = "/sdcard/laser_calib.json"

# 参数全部可在实例化时覆盖；单位：像素 / 毫秒
CFG = {
    "enabled": 1,
    "rad": 12,              # 初始微窗半径：±12 -> 25×25，实测 ~1ms
    "rad_max": 24,          # 连续丢失后放大的上限：±24 -> 49×49，实测 ~4ms
    "budget_ms": 10,        # 单帧硬预算；超时按"没找到"处理，绝不拖住主循环
    "warm_min": 55,         # R-G 阈值（本机实测激光 119~220，白墙 2~33）
    "luma_min": 30,         # 压掉暗红色噪点/暗红物体
    "min_px": 2,            # 至少几个满足阈值的像素才算光斑
    "max_px": 800,          # 微窗里红像素太多 -> 是红色物体，不是光斑
    "ema": 0.5,             # 偏移量平滑（0=完全信标定，1=完全信实测）
    "lost_widen_after": 3,  # 连续 3 次微窗失败 -> 放大到 rad_max 再试
    "lost_after": 10,       # 连续 10 次失败 -> status=lost（ESP32 侧自行降级）
    "max_step": 24,         # 实测偏离预测超过此值视为离群，不更新
    "calib_warn": 40,       # 实测相对标定偏移超过此值 -> need_recalib
    "warm_k": 80.0,         # conf = (max_warm - warm_min)/warm_k + conf_floor
    "conf_floor": 0.2,
    # ---- 运动滤波/预测（α-β 常速 + 前瞻 + 短时 coast）----
    "ab_alpha": 0.45,       # 位置残差增益（越小越平滑、越滞后）
    "ab_beta": 0.10,        # 速度残差增益（越小速度越稳）
    "lead_ms": 50,          # 输出前瞻：补偿相机->USB/UART->ESP32 的链路延迟
    "v_max": 2000.0,        # 速度上限 px/s，防离群把速度带飞
    "coast_max": 8,         # 微窗连续失败时最多外推几帧
    "rad_extra": 8,         # 微窗随速度最多放大多少 px（49x49 预算内）
    "coast_conf": 0.35,     # coast 输出置信度（逐帧轻微衰减）
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


class _Tracker:
    """α-β 常速滤波器（位置 + 速度），几行浮点，适合 MicroPython。

    predict(t)  : x + v*dt
    update(z,t) : 对预测残差按 α 修位置、按 β/dt 修速度
    比一阶低通多一个速度状态，能补匀速运动滞后；比 Kalman 轻，参数直观。
    """

    def __init__(self, alpha=0.45, beta=0.10, v_max=2000.0):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.v_max = float(v_max)
        self.x = None
        self.y = None
        self.vx = 0.0
        self.vy = 0.0
        self.t = None
        self.dt_last = 0.0

    def ready(self):
        return self.x is not None

    def reset(self):
        self.x = None
        self.y = None
        self.vx = 0.0
        self.vy = 0.0
        self.t = None
        self.dt_last = 0.0

    def init(self, x, y, t):
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.t = t
        self.dt_last = 0.0

    def predict(self, t):
        if self.x is None:
            return None
        dt = (t - self.t) / 1000.0
        if dt < 0:
            dt = 0.0
        elif dt > 0.5:
            dt = 0.5
        return (self.x + self.vx * dt, self.y + self.vy * dt)

    def update(self, zx, zy, t):
        if self.x is None:
            self.init(zx, zy, t)
            return
        dt = (t - self.t) / 1000.0
        if dt < 0.02:
            # 最小 20ms：快速循环调用时 beta/dt 不会把速度噪声放大
            dt = 0.02
        elif dt > 0.5:
            dt = 0.5
        px = self.x + self.vx * dt
        py = self.y + self.vy * dt
        rx = float(zx) - px
        ry = float(zy) - py
        self.x = px + self.alpha * rx
        self.y = py + self.alpha * ry
        self.vx += self.beta * rx / dt
        self.vy += self.beta * ry / dt
        sp = (self.vx * self.vx + self.vy * self.vy) ** 0.5
        if sp > self.v_max:
            k = self.v_max / sp
            self.vx *= k
            self.vy *= k
        self.t = t
        self.dt_last = dt

    def speed(self):
        return (self.vx * self.vx + self.vy * self.vy) ** 0.5

    def decay(self, f=0.5):
        self.vx *= f
        self.vy *= f


class LaserAim:
    """标定先验 + 微窗质心的有状态光斑定位器。"""

    def __init__(self, cfg=None):
        self.cfg = dict(CFG)
        if cfg:
            self.cfg.update(cfg)
        self.calib = None       # (x, y, kx, ky, ref)
        self.off = [0.0, 0.0]   # 实测 - 预测的平滑偏移
        self._s = None          # 靶纸表观边长 EMA（抑制框/关键点抖动）
        self.tr = _Tracker(float(self.cfg["ab_alpha"]),
                           float(self.cfg["ab_beta"]),
                           float(self.cfg["v_max"]))
        self._reject_n = 0      # max_step 连续拒绝次数
        self.reject_dev = 0.0   # 最近一次被拒绝的偏移（诊断）
        self.last = None
        self.last_ms = 0
        self.last_conf = 0.0
        self.last_px = 0
        self.last_budget = False
        self.miss = 0
        self.need_calib = True
        self.need_recalib = False
        self.calib_dev = 0.0
        self.mode = "none"
        self.src = "none"
        self.status = "none"

    # ------------------------------------------------------------ 标定
    def set_calib(self, x, y, kx=0.0, ky=0.0, ref=0.0):
        """写入标定：之后 sample() 只在这个位置附近找。"""
        self.calib = (float(x), float(y), float(kx), float(ky), float(ref))
        self.off = [0.0, 0.0]
        self._s = None
        self.tr.reset()
        self._reject_n = 0
        self.reject_dev = 0.0
        self.last = (float(x), float(y))
        self.miss = 0
        self.need_calib = False
        self.need_recalib = False
        self.calib_dev = 0.0
        self.last_conf = 0.0
        self.last_px = 0
        self.last_budget = False
        self.mode = "calib"
        self.src = "calib"
        self.status = "calib"

    def load(self, path=CALIB_FILE):
        """读标定文件；兼容旧的 {"x","y"} 两字段格式。返回 True/False。"""
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
            self.set_calib(x, y, float(d.get("kx", 0.0)),
                           float(d.get("ky", 0.0)), float(d.get("ref", 0.0)))
            return True
        except Exception:
            return False

    def save(self, path=CALIB_FILE):
        """板端一般不需要写（标定在 PC 做）；保留给手动/调试。"""
        if self.calib is None:
            return False
        x, y, kx, ky, ref = self.calib
        try:
            f = open(path, "w")
            try:
                f.write('{"x":%.2f,"y":%.2f,"kx":%.5f,"ky":%.5f,"ref":%.2f}'
                        % (x, y, kx, ky, ref))
            finally:
                f.close()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------ 预测
    def _base(self, s=None):
        x, y, kx, ky, ref = self.calib
        # s = 靶纸表观边长（px），与距离成反比；k 由多距离标定得到
        if s is not None and ref > 0 and s > 0:
            d = float(s) - ref
            x += kx * d
            y += ky * d
        return (x, y)

    def predict(self, s=None):
        if self.calib is None:
            return None
        if self.tr.ready():
            p = self.tr.predict(_now_ms() + int(self.cfg["lead_ms"]))
            if p is not None:
                return p
        return self._base(s)

    # ------------------------------------------------------------ 微窗
    def _scan(self, img, cx, cy, rad, t0):
        """±rad 微窗内 R-G/亮度加权质心。返回 (x,y,n,max_warm) 或 None。

        只做一次稠密扫描（1.6us/px 量级），没有候选列表、没有 find_blobs。
        光斑过曝成白核时，红晕仍然满足 R-G 阈值，质心仍然在光斑中心。
        """
        c = self.cfg
        try:
            ba = img.bytearray()
        except Exception:
            return None
        w = img.width()
        h = img.height()
        rad = int(rad)
        x0 = int(cx) - rad
        y0 = int(cy) - rad
        x1 = int(cx) + rad + 1
        y1 = int(cy) + rad + 1
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 > w:
            x1 = w
        if y1 > h:
            y1 = h
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None
        warm_min = int(c["warm_min"])
        luma_min = int(c["luma_min"])
        budget = int(c["budget_ms"])
        sw = 0.0
        sx = 0.0
        sy = 0.0
        n = 0
        mw = 0
        for yy in range(y0, y1):
            if budget and _diff_ms(_now_ms(), t0) > budget:
                self.last_budget = True
                return None
            row = yy * w
            for xx in range(x0, x1):
                i = (row + xx) * 2
                v = ba[i] | (ba[i + 1] << 8)
                r = ((v >> 11) & 0x1F) * 8
                g = ((v >> 5) & 0x3F) * 4
                b = (v & 0x1F) * 8
                warm = r - g
                if warm < warm_min:
                    continue
                lu = (r * 77 + g * 150 + b * 29) >> 8
                if lu < luma_min:
                    continue
                n += 1
                if warm > mw:
                    mw = warm
                wt = warm - warm_min + ((lu - luma_min) >> 2)
                if wt <= 0:
                    wt = 1
                sx += xx * wt
                sy += yy * wt
                sw += wt
        if n < int(c["min_px"]) or n > int(c["max_px"]) or sw <= 0:
            return None
        return (sx / sw, sy / sw, n, mw)

    # ------------------------------------------------------------ 对外
    def _ret(self, px, py, conf, mode, src, status, n=0, vx=0.0, vy=0.0):
        return {"x": px, "y": py, "conf": conf, "mode": mode, "src": src,
                "status": status, "ms": self.last_ms, "n": n,
                "budget": 1 if self.last_budget else 0,
                "vx": vx, "vy": vy}

    def _coast(self, t0, bx, by):
        """短时外推：微窗暂时没命中，用 α-β 速度继续预测（不是全图搜索）。"""
        c = self.cfg
        self.tr.decay(0.9)
        out = self.tr.predict(t0 + int(c["lead_ms"]))
        if out is None:
            out = (bx, by)
        px, py = out
        self.mode = "coast"
        self.src = "coast"
        self.status = "coast"
        conf = float(c["coast_conf"]) * (1.0 - 0.05 * self.miss)
        if conf < 0.1:
            conf = 0.1
        self.last_conf = conf
        self.last = (px, py)
        self.last_px = 0
        return self._ret(px, py, conf, "coast", "coast", "coast",
                         vx=self.tr.vx, vy=self.tr.vy)

    def sample(self, img, s=None):
        """每帧调用：返回 dict 或 None（无标定）。

        s: 靶纸表观边长（px），用于 `c + k·s` 距离补偿；没有就传 None。
        输出位置是 **α-β 滤波 + lead_ms 前瞻** 的结果；微窗中心用无前瞻的
        预测位置，保证测量窗口对得上本帧图像。
        """
        c = self.cfg
        t0 = _now_ms()
        self.last_ms = 0
        self.last_budget = False
        if not c.get("enabled", 1) or self.calib is None:
            self.need_calib = True
            self.mode = "none"
            self.src = "none"
            self.status = "none"
            return None

        # s 做 EMA：检测框/关键点单帧抖动不该直接乘进 k·s
        if s is not None and float(s) > 0:
            if self._s is None:
                self._s = float(s)
            else:
                self._s += 0.3 * (float(s) - self._s)
            s_use = self._s
        else:
            s_use = None
        bx, by = self._base(s_use)

        # 微窗中心：有航迹用 α-β 预测，否则用标定
        if self.tr.ready():
            p = self.tr.predict(t0)
            if p is None:
                px, py = bx, by
            else:
                px, py = p
        else:
            px, py = bx, by

        rad = int(c["rad"])
        if self.miss >= int(c["lost_widen_after"]):
            rad = int(c["rad_max"])
        else:
            # 窗口随速度放大：快速运动时预测点附近仍能覆盖光斑
            extra = int(self.tr.speed() * self.tr.dt_last / 1000.0)
            if extra > int(c["rad_extra"]):
                extra = int(c["rad_extra"])
            if extra > 0:
                rad += extra

        r = self._scan(img, px, py, rad, t0)
        self.last_ms = _diff_ms(_now_ms(), t0)
        lead_ms = int(c["lead_ms"])

        if r is None:
            self.miss += 1
            if self.tr.ready() and self.miss <= int(c["coast_max"]):
                return self._coast(t0, bx, by)
            self.tr.decay(0.5)
            if self.miss >= int(c["lost_after"]):
                self.status = "lost"
                self.last_conf = 0.0
                # 连续丢失不是"暂时遮挡"就是"支架被撞/标定过期"：
                # 标记出来让 PC/`latest.json`/agent 日志能看到，别静默沿用。
                self.need_recalib = True
            else:
                self.status = "calib"
                self.last_conf = 0.25
            self.mode = "calib"
            self.src = "calib"
            self.last = (bx, by)
            self.last_px = 0
            return self._ret(bx, by, self.last_conf, "calib", "calib",
                             self.status)

        mx, my, n, mw = r
        dx = mx - px
        dy = my - py
        max_step = float(c["max_step"])
        if dx * dx + dy * dy > max_step * max_step:
            # 窗口里最像光斑的东西离预测太远：当离群，不跟。
            # 但连续发生说明标定/支架可能变了（突然偏移不会走 calib_dev），
            # 记下来并在连续 N 次后置 need_recalib。
            self.miss += 1
            self._reject_n += 1
            self.reject_dev = (dx * dx + dy * dy) ** 0.5
            if self._reject_n >= int(c["lost_after"]):
                self.need_recalib = True
            if self.tr.ready() and self.miss <= int(c["coast_max"]):
                return self._coast(t0, bx, by)
            if self.miss >= int(c["lost_after"]):
                self.status = "lost"
                self.last_conf = 0.0
            else:
                self.status = "calib"
                self.last_conf = 0.25
            self.mode = "calib"
            self.src = "calib"
            self.last = (bx, by)
            self.last_px = 0
            return self._ret(bx, by, self.last_conf, "calib", "calib",
                             self.status)

        # 命中：α-β 更新，输出带 lead_ms 的前瞻位置
        self.tr.update(mx, my, t0)
        self.miss = 0
        self._reject_n = 0
        self.reject_dev = 0.0
        out = self.tr.predict(t0 + lead_ms)
        if out is None:
            out = (mx, my)
        px, py = out
        self.off = [self.tr.x - bx, self.tr.y - by]
        self.calib_dev = (self.off[0] ** 2 + self.off[1] ** 2) ** 0.5
        self.need_recalib = self.calib_dev > float(c["calib_warn"])
        conf = (mw - int(c["warm_min"])) / float(c["warm_k"]) + float(c["conf_floor"])
        if conf > 1.0:
            conf = 1.0
        if conf < 0.0:
            conf = 0.0
        self.last_conf = conf
        self.last = (px, py)
        self.last_px = n
        self.mode = "roi"
        self.src = "measured"
        self.status = "ok"
        return self._ret(px, py, conf, "roi", "measured", "ok", n=n,
                         vx=self.tr.vx, vy=self.tr.vy)

    def reset(self):
        """清跟踪状态，回到标定位置（标定本身不丢）。"""
        self.off = [0.0, 0.0]
        self._s = None
        self.tr.reset()
        self._reject_n = 0
        self.reject_dev = 0.0
        self.last_budget = False
        self.miss = 0
        self.last_conf = 0.0
        self.last_px = 0
        self.last_budget = False
        self.mode = "calib"
        self.src = "calib"
        self.status = "calib" if self.calib is not None else "none"
        self.last = self.predict()
