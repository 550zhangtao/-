# -*- coding: utf-8 -*-
"""laser_heatmap_static.py —— 不碰相机的静态图验证

从 /sdcard 读一张真实激光帧，裁 128x128，跑 kmodel，打印峰值/坐标。
用于先验证 nncase_runtime + Ai2d + laser_heatmap 这条链路，避免相机 hang。
"""
MODEL = "/sdcard/models/laser_heatmap.kmodel"
IMG = "/sdcard/laser_on_static.jpg"
ROI = 128
CX, CY = 152, 300

import sys
sys.path.insert(0, "/sdcard")

import image
import gc

sys.modules.pop("laser_heatmap", None)
from laser_heatmap import LaserHeat

img = None
try:
    img = image.Image(IMG)
except Exception as e:
    print("Image(path) failed:", e)
    try:
        img = image.Image()
        img.load(IMG)
    except Exception as e2:
        print("Image.load failed:", e2)
        raise

print("IMG", img.width(), img.height())
x0 = max(0, min(img.width() - ROI, CX - ROI // 2))
y0 = max(0, min(img.height() - ROI, CY - ROI // 2))
crop = img.crop(roi=(x0, y0, ROI, ROI)).to_rgb888().to_numpy_ref()
print("CROP", crop.shape, "at", x0, y0)

lh = LaserHeat(MODEL)
r = lh.detect(crop, x0, y0)
print("RESULT", r)

# keepalive，避免 KPU 析构
keep = sys.modules.get("_k230_ai_keepalive")
if keep is None:
    keep = {}
    sys.modules["_k230_ai_keepalive"] = keep
keep["lh"] = lh
gc.collect()
print("OK")
