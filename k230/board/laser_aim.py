# -*- coding: utf-8 -*-
"""laser_aim.py —— 把激光点引导到靶心（闭环 aim 点生成）

思路（复用工程已有链路）：
  * 输入：激光点 (lx,ly) 与靶心 (bx,by)，都来自现有检测/热图；
  * 误差：e = 靶心 - 激光；
  * 输出：给 $KT 的 aim 点 aim = center + sign*gain*e；
  * ESP32 的 PID 会把 aim 点往画面中心拉，等效于以速度 ∝ e 驱动云台；
  * K230 每帧重算 e，形成闭环（视觉伺服）。

符号 SIGN 取决于机械安装：
  * 相机+激光共装（转云台光斑几乎不动）→ sign=-1（默认），
    云台把靶心"拉"到激光所在像素；
  * 激光单独装在云台上（转云台光斑会动）→ 先试 sign=+1。
调参：先 gain 小（0.4~0.6），看方向对不对；方向反了就把对应 sign 取反。

用法：
    aim = LaserAim(gain=0.6, sign_x=-1, sign_y=-1)
    ax, ay, ex, ey = aim.aim((lx, ly), (bx, by))
    # 把 ax, ay 作为 $KT 的 bx,by 发出去
"""
import math


class LaserAim:
    def __init__(self, gain=0.6, sign_x=-1, sign_y=-1,
                 center=(320.0, 240.0), max_off=180.0, deadband=2.0):
        self.gain = float(gain)
        self.sign_x = float(sign_x)
        self.sign_y = float(sign_y)
        self.cx0 = float(center[0])
        self.cy0 = float(center[1])
        self.max_off = float(max_off)
        self.deadband = float(deadband)

    def error(self, laser, bull):
        return (float(bull[0]) - float(laser[0]),
                float(bull[1]) - float(laser[1]))

    def aim(self, laser, bull):
        """返回 (aim_x, aim_y, ex, ey)；ex/ey 为靶心-激光的原始像素误差。"""
        ex, ey = self.error(laser, bull)

        # 死区内不动（避免抖动）
        if abs(ex) < self.deadband and abs(ey) < self.deadband:
            return int(round(self.cx0)), int(round(self.cy0)), ex, ey

        ax = self.cx0 + self.sign_x * self.gain * ex
        ay = self.cy0 + self.sign_y * self.gain * ey

        # 限幅：防止误差过大时云台速度爆掉
        ax = max(self.cx0 - self.max_off, min(self.cx0 + self.max_off, ax))
        ay = max(self.cy0 - self.max_off, min(self.cy0 + self.max_off, ay))
        return int(round(ax)), int(round(ay)), ex, ey

    def aligned(self, laser, bull, tol=6.0):
        """激光是否已落到靶心上（用于状态提示）。"""
        ex, ey = self.error(laser, bull)
        return math.sqrt(ex * ex + ey * ey) <= float(tol)
