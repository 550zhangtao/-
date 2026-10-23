# -*- coding: utf-8 -*-
"""laser_track.py —— 激光/靶心追踪的"运动状态估计 + 延迟外推"层（第一段）

背景：感知-执行链路（曝光 → 推理 → 串口 → 舵机启动）有几十毫秒固定延迟，
纯反馈 PID 只能等误差出现再动作，追移动目标必然滞后。这一层把
"现在测到的坐标"换算成"指令真正生效时目标所在的坐标"，让云台提前转向。

结构（两轴独立，状态 [位置, 速度]，匀速模型）：

    tracker = LaserTracker(delay_ms=80.0)
    tracker.update(x, y, t_ms)          # 每帧识别成功后调用
    px, py, vx, vy = tracker.predict(t_ms)
    #   px,py = 延迟补偿后的预测位置（送 $KT / PID）
    #   vx,vy = 速度（px/s，前馈用）

参数含义：
    delay_ms   端到端固定延迟（曝光→舵机开始响应），第一段先给经验值 80，
               用测量脚本标定后填真实值；这是预测补偿的基准。
    q          加速度噪声强度（px^2/s^3），越大越"信观测"、机动响应快但抖。
    r          观测噪声方差（px^2），热图模型定位抖动约 2~3px → r≈9。
    max_gap_ms 超过这么久没更新就认为目标丢失，predict 返回 None。

设计约束：
    * 纯 Python，无 numpy，可在 K230 MicroPython 上每帧调用（<0.1ms）。
    * update/predict 不做"跳变拒绝"（那是识别层的事），只做状态估计。
    * predict 不推进滤波器状态（匀速假设下 p + v*dt 即可），
      这样同一帧多次调用、或下一帧 update 都不会互相污染。
"""

import time

__version__ = "1.0"

# MicroPython 有 ticks_ms/ticks_diff；PC 上做仿真时用单调时钟兜底
try:
    _ticks_ms = time.ticks_ms
    _ticks_diff = time.ticks_diff
except AttributeError:  # CPython
    def _ticks_ms():
        return int(time.monotonic() * 1000.0)

    def _ticks_diff(a, b):
        return a - b


class _Axis:
    """单轴 2 状态（位置、速度）匀速卡尔曼。"""

    def __init__(self, q, r):
        self.q = float(q)
        self.r = float(r)
        self.reset()

    def reset(self):
        self.p = 0.0
        self.v = 0.0
        # 初始协方差放大：第一次观测基本直接采纳
        self.p00 = 1.0e4
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = 1.0e4
        self.seen = False

    def predict(self, dt):
        """状态推进 dt 秒（dt<=0 直接返回）。"""
        if dt <= 0.0:
            return
        self.p += self.v * dt
        q = self.q
        dt2 = dt * dt
        dt3 = dt2 * dt
        p00 = self.p00 + dt * (self.p01 + self.p10) + dt2 * self.p11 + q * dt3 / 3.0
        p01 = self.p01 + dt * self.p11 + q * dt2 / 2.0
        p10 = self.p10 + dt * self.p11 + q * dt2 / 2.0
        p11 = self.p11 + q * dt
        self.p00, self.p01, self.p10, self.p11 = p00, p01, p10, p11

    def update(self, z):
        """用观测 z 更新状态。"""
        y = z - self.p
        s = self.p00 + self.r
        if s <= 0.0:
            s = 1.0
        k0 = self.p00 / s
        k1 = self.p10 / s
        self.p += k0 * y
        self.v += k1 * y
        # P = (I - K H) P
        p00 = (1.0 - k0) * self.p00
        p01 = (1.0 - k0) * self.p01
        p10 = self.p10 - k1 * self.p00
        p11 = self.p11 - k1 * self.p01
        self.p00, self.p01, self.p10, self.p11 = p00, p01, p10, p11
        self.seen = True


class LaserTracker:
    """两轴匀速卡尔曼 + 固定延迟外推（预测锚点用最新观测，速度用近几帧斜率）。

    为什么锚点不用滤波后的位置：滤波位置有滞后，外推时滞后会原样叠加；
    实测（tools/test_laser_track.py）锚点用最新观测比用滤波位置误差更小。

    速度 = PRED_SLOPE_A * 最近 N 帧最小二乘斜率 + (1-A) * 卡尔曼速度：
    斜率对换向响应快，卡尔曼速度平滑；两者混合兼顾。
    """

    def __init__(self, delay_ms=80.0, q=4000.0, r=9.0, max_gap_ms=250.0,
                 slope_n=3, slope_a=0.75, clamp_px=60.0):
        self.delay_ms = float(delay_ms)
        self.max_gap_ms = float(max_gap_ms)
        self.slope_n = int(slope_n)
        self.slope_a = float(slope_a)
        self.clamp_px = float(clamp_px)
        self.ax = _Axis(q, r)
        self.ay = _Axis(q, r)
        self.t = 0
        self.ok = False
        self.hist = []          # [(t, x, y), ...] 最近 slope_n 帧观测

    # ------------------------------------------------------------ 基础
    def reset(self):
        self.ax.reset()
        self.ay.reset()
        self.ok = False
        self.hist = []

    def update(self, x, y, t_ms=None):
        """喂一帧观测。返回 True 表示已被滤波器采纳。"""
        if t_ms is None:
            t_ms = _ticks_ms()
        x = float(x)
        y = float(y)
        if not self.ok:
            self.ax.reset()
            self.ay.reset()
            self.ax.p = x
            self.ay.p = y
            self.ax.seen = True
            self.ay.seen = True
            self.t = t_ms
            self.ok = True
            self.hist = [(t_ms, x, y)]
            return True
        dt = _ticks_diff(t_ms, self.t) / 1000.0
        if dt <= 0.0:
            return False
        if dt * 1000.0 > self.max_gap_ms:
            self.reset()
            return self.update(x, y, t_ms)
        self.ax.predict(dt)
        self.ay.predict(dt)
        self.ax.update(x)
        self.ay.update(y)
        self.t = t_ms
        self.hist.append((t_ms, x, y))
        while len(self.hist) > self.slope_n:
            self.hist.pop(0)
        return True

    # ------------------------------------------------------------ 输出
    def valid(self, t_ms=None):
        if not self.ok or not self.ax.seen:
            return False
        if t_ms is None:
            t_ms = _ticks_ms()
        gap = _ticks_diff(t_ms, self.t)
        return 0 <= gap <= self.max_gap_ms

    def _slope(self):
        """最近 N 帧的最小二乘速度（px/s）；点数不足 2 时返回 (0,0)。"""
        n = len(self.hist)
        if n < 2:
            return (0.0, 0.0)
        st = sx = sy = stt = stx = sty = 0.0
        for (tt, xx, yy) in self.hist:
            st += tt
            sx += xx
            sy += yy
            stt += tt * tt
            stx += tt * xx
            sty += tt * yy
        den = n * stt - st * st
        if den == 0.0:
            return (0.0, 0.0)
        return ((n * stx - st * sx) / den * 1000.0,
                (n * sty - st * sy) / den * 1000.0)

    def velocity(self, t_ms=None):
        """混合速度 (vx, vy) px/s：斜率 + 卡尔曼速度。"""
        if not self.valid(t_ms):
            return None
        svx, svy = self._slope()
        return (self.slope_a * svx + (1.0 - self.slope_a) * self.ax.v,
                self.slope_a * svy + (1.0 - self.slope_a) * self.ay.v)

    def predict(self, t_ms=None, delay_ms=None):
        """返回 (px, py, vx, vy)：预测"指令生效时刻"的目标位置与速度。

        锚点 = 最新观测（无滤波滞后）；外推量 = v * (delay + 距上次观测的时间)。
        外推量按 clamp_px 限幅，避免换向瞬间的过大速度把目标推飞。
        """
        v = self.velocity(t_ms)
        if v is None:
            return None
        if delay_ms is None:
            delay_ms = self.delay_ms
        if t_ms is None:
            t_ms = _ticks_ms()
        dt = (_ticks_diff(t_ms, self.t) + delay_ms) / 1000.0
        if dt < 0.0:
            dt = 0.0
        ex = v[0] * dt
        ey = v[1] * dt
        if ex > self.clamp_px:
            ex = self.clamp_px
        elif ex < -self.clamp_px:
            ex = -self.clamp_px
        if ey > self.clamp_px:
            ey = self.clamp_px
        elif ey < -self.clamp_px:
            ey = -self.clamp_px
        x_m, y_m = self.hist[-1][1], self.hist[-1][2]
        return (x_m + ex, y_m + ey, v[0], v[1])

    def position(self):
        """滤波后的当前位置（不外推，画图/日志用）。"""
        return (self.ax.p, self.ay.p)


class IncPID:
    """增量式 PID + 死区 + 前馈限幅（给舵机侧移植用的参考实现）。

    step(err, dt) 返回 PWM 增量；ff(vel) 返回限幅后的前馈偏置。
    死区：|err| < deadband 时增量为 0，避免舵机在靶心附近来回抖。
    前馈限幅：|ff| <= ff_max，防止换向瞬间的过大速度估计造成过冲。
    """

    def __init__(self, kp=2.5, ki=0.3, kd=0.15, deadband=6.0, ff_max=120.0,
                 out_max=120.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.deadband = float(deadband)
        self.ff_max = float(ff_max)
        self.out_max = float(out_max)
        self.reset()

    def reset(self):
        self.e1 = 0.0
        self.e2 = 0.0

    def step(self, err, dt=0.06):
        if dt <= 0.0:
            return 0.0
        if abs(err) < self.deadband:
            d = 0.0
        else:
            d = (self.kp * (err - self.e1)
                 + self.ki * err * dt
                 + self.kd * (err - 2.0 * self.e1 + self.e2) / dt)
        self.e2 = self.e1
        self.e1 = err
        if d > self.out_max:
            d = self.out_max
        elif d < -self.out_max:
            d = -self.out_max
        return d

    def ff(self, vx, vy=0.0, gain=1.0):
        v = vx if abs(vx) >= abs(vy) else vy
        f = gain * v
        if f > self.ff_max:
            f = self.ff_max
        elif f < -self.ff_max:
            f = -self.ff_max
        return f
