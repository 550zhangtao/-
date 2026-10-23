# -*- coding: utf-8 -*-
"""
point_kalman.py —— 二维点 [x, y, vx, vy] 常速卡尔曼平滑（板端共享）
====================================================================

同一套滤波同时用于：
    * Pose 第 5 关键点（靶心）：board/yolo_pose.py / yolo_preview_ide.py /
      gimbal_agent.py 三处；
    * 激光点：board/laser_repo.py（IDE 预览里的激光十字）。

模型：每轴 位置+速度（协方差 2x2），等价于 4 状态 [x,y,vx,vy] 的块对角
卡尔曼（两轴独立、无互相关）。

用法：
    from point_kalman import PointKalman
    kf = PointKalman()
    s = kf.update(x, y, time.ticks_ms())   # 有测量 -> (x,y,vx,vy)
    s = kf.predict(time.ticks_ms())        # 没测量（短暂丢失）-> 预测，超时 None

参数（构造时可覆盖）：
    q         过程噪声（加速度方差, px^2/s^3）：越大越跟手，越小越平滑
    r         观测噪声（抖动方差, px^2）：默认 9 ≈ 3px 标准差
    max_jump  单次观测偏差限幅（px）：偏差超过就截断，抑制异常跳变
    max_coast 连续丢失多少帧后放弃预测并 reset
"""

# 默认参数
Q_ACC = 150.0      # px^2/s^3
R_MEAS = 9.0       # px^2（≈3px 标准差）
MAX_JUMP = 40.0    # px
MAX_COAST = 6      # 帧


class PointKalman:
    def __init__(self, q=Q_ACC, r=R_MEAS, max_jump=MAX_JUMP, max_coast=MAX_COAST):
        self.q = float(q)
        self.r = float(r)
        self.mj = float(max_jump)
        self.mc = int(max_coast)
        self.reset()

    def reset(self):
        self.ok = False
        self.lost = 0
        self.t = None
        self.px = self.py = self.vx = self.vy = 0.0
        # 每轴协方差 [p00, p01, p11]
        self.cx = [0.0, 0.0, 0.0]
        self.cy = [0.0, 0.0, 0.0]

    def _dt(self, t):
        if self.t is None:
            return 0.0
        d = (float(t) - float(self.t)) / 1000.0
        if d <= 0.0:
            d = 0.001
        elif d > 0.5:
            d = 0.5
        return d

    def _pred(self, p, v, c, d):
        np_ = p + v * d
        a00 = c[0] + d * c[1]
        a01 = c[1] + d * c[2]
        return np_, v, [a00 + self.q * d * d * d / 3.0,
                        a00 * d + a01 + self.q * d * d * 0.5,
                        c[1] * d + c[2] + self.q * d]

    def _upd(self, p, v, c, z):
        y = z - p
        s = c[0] + self.r
        if s < 1e-9:
            s = 1e-9
        k0 = c[0] / s
        k1 = c[1] / s
        # Joseph 形式：P=(I-KH)P(I-KH)^T+K R K^T，保证协方差正定，
        # 避免长跑后简单形式 (I-KH)P 因舍入变负、再滚成 inf/NaN。
        p00 = (1.0 - k0) * (1.0 - k0) * c[0] + k0 * k0 * self.r
        p01 = (1.0 - k0) * (c[1] - k1 * c[0]) + k0 * k1 * self.r
        p11 = c[2] - 2.0 * k1 * c[1] + k1 * k1 * (c[0] + self.r)
        if p00 != p00 or p01 != p01 or p11 != p11:
            # 数值异常：重建协方差
            p00 = self.r
            p01 = 0.0
            p11 = 400.0
        else:
            if p00 < 1e-6:
                p00 = 1e-6
            if p11 < 1e-6:
                p11 = 1e-6
            lim = (p00 * p11) ** 0.5
            if p01 > lim:
                p01 = lim
            elif p01 < -lim:
                p01 = -lim
        return p + k0 * y, v + k1 * y, [p00, p01, p11]

    def update(self, x, y, t):
        """喂一个有效测量；返回 (x, y, vx, vy) 或 None。"""
        if x != x or y != y:
            return None
        if self.ok and (self.px != self.px or self.py != self.py
                        or self.vx != self.vx or self.vy != self.vy):
            self.reset()
        if not self.ok:
            self.ok = True
            self.px = float(x)
            self.py = float(y)
            self.vx = self.vy = 0.0
            self.cx = [self.r, 0.0, 400.0]
            self.cy = [self.r, 0.0, 400.0]
            self.t = t
            self.lost = 0
            return (self.px, self.py, 0.0, 0.0)
        d = self._dt(t)
        self.t = t
        self.px, self.vx, self.cx = self._pred(self.px, self.vx, self.cx, d)
        self.py, self.vy, self.cy = self._pred(self.py, self.vy, self.cy, d)
        # MAX_JUMP：把观测偏差截断到 ±max_jump，异常跳变不会把状态带飞
        ex = float(x) - self.px
        ey = float(y) - self.py
        if ex > self.mj:
            ex = self.mj
        elif ex < -self.mj:
            ex = -self.mj
        if ey > self.mj:
            ey = self.mj
        elif ey < -self.mj:
            ey = -self.mj
        self.px, self.vx, self.cx = self._upd(self.px, self.vx, self.cx, self.px + ex)
        self.py, self.vy, self.cy = self._upd(self.py, self.vy, self.cy, self.py + ey)
        self.lost = 0
        return (self.px, self.py, self.vx, self.vy)

    def predict(self, t):
        """没测量时外推；连续丢失超过 max_coast 帧则 reset 并返回 None。"""
        if not self.ok:
            return None
        if self.px != self.px or self.py != self.py:
            self.reset()
            return None
        d = self._dt(t)
        self.t = t
        self.px, self.vx, self.cx = self._pred(self.px, self.vx, self.cx, d)
        self.py, self.vy, self.cy = self._pred(self.py, self.vy, self.cy, d)
        self.lost += 1
        if self.lost > self.mc:
            self.reset()
            return None
        return (self.px, self.py, self.vx, self.vy)

    def state(self):
        if not self.ok:
            return None
        return (self.px, self.py, self.vx, self.vy)
