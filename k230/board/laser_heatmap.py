# -*- coding: utf-8 -*-
"""laser_heatmap.py —— 激光点 ROI 热图推理（K230）

模型：/sdcard/models/laser_heatmap.kmodel
      输入 128x128x3 uint8（RGB packed）-> 输出 1x1x32x32 logits

流程：AI2D 做 RGB_packed -> NCHW，KPU 推理，取热图峰值 + 3x3 加权质心。
调用方给 128x128 的 RGB 裁剪（原生像素），返回全局坐标。
"""
import math

import nncase_runtime as nn
import ulab.numpy as np
from libs.AIBase import AIBase
from libs.AI2D import Ai2d


class LaserHeat(AIBase):
    def __init__(self, kmodel_path, roi=128, debug=0):
        super().__init__(kmodel_path, model_input_size=[roi, roi],
                         rgb888p_size=[roi, roi], debug_mode=debug)
        self.roi = roi
        self.ai2d = Ai2d(debug)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.RGB_packed,
                                 nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
        self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
        self.ai2d.build([1, roi, roi, 3], [1, 3, roi, roi])
        self.last = None

    @staticmethod
    def _sig(v):
        try:
            v = float(v)
            if v > 30:
                return 1.0
            if v < -30:
                return 0.0
            return 1.0 / (1.0 + math.exp(-v))
        except Exception:
            return 0.0

    def detect(self, crop_rgb, x0=0, y0=0, thr=0.5):
        """crop_rgb: (roi,roi,3) uint8。返回 (gx,gy,conf) 或 None。"""
        tensors = self.preprocess(crop_rgb)
        res = self.inference(tensors)
        h = res[0]
        try:
            while len(h.shape) > 2:
                h = h[0]
        except Exception:
            pass
        n = h.shape[1]
        idx = int(np.argmax(h))
        iy, ix = divmod(idx, n)
        peak = self._sig(h[iy, ix])

        sx = sy = sw = 0.0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy = iy + dy
                xx = ix + dx
                if 0 <= yy < n and 0 <= xx < n:
                    w = self._sig(h[yy, xx])
                    sx += xx * w
                    sy += yy * w
                    sw += w
        if sw > 0:
            cx = sx / sw
            cy = sy / sw
        else:
            cx = float(ix)
            cy = float(iy)

        stride = self.roi / float(n)
        gx = x0 + cx * stride + stride * 0.5
        gy = y0 + cy * stride + stride * 0.5
        self.last = (gx, gy, peak)
        if peak < thr:
            return None
        return (gx, gy, peak)
