# -*- coding: utf-8 -*-
"""laser_capture_s1.py —— 云台 s1 左右扫动期间的激光采集/留帧（K230 侧）

不发送 $KT；云台由 PC 侧 /dev/ttyUSB0 的 `s1` 指令慢速左右驱动。
每 SAVE_EVERY 帧存 /sdcard/sweep3/，并写 coords.csv。
"""
MODEL = "/sdcard/models/laser_heatmap.kmodel"
ROI = 128
PRIOR = (317, 261)
DURATION_S = 40
SAVE_EVERY = 6
OUT_DIR = "/sdcard/tmp/sweep3"

from media.sensor import *
from media.media import *

import sys
import time
import gc
import os

sys.path.insert(0, "/sdcard")
sys.modules.pop("laser_heatmap", None)
from laser_heatmap import LaserHeat

_sensor = None
_lh = None


def main():
    global _sensor, _lh
    try:
        os.mkdir(OUT_DIR)
    except Exception:
        pass
    _sensor = Sensor(id=2)
    _sensor.reset()
    _sensor.set_framesize(w=640, h=480, chn=CAM_CHN_ID_1)
    _sensor.set_pixformat(Sensor.RGB565, chn=CAM_CHN_ID_1)
    MediaManager.init()
    _sensor.run()
    time.sleep_ms(300)
    _lh = LaserHeat(MODEL)

    prior = (float(PRIOR[0]), float(PRIOR[1]))
    t0 = time.ticks_ms()
    n = 0
    rows = []
    hits = 0
    while True:
        el = time.ticks_diff(time.ticks_ms(), t0) / 1000.0
        if el > DURATION_S:
            break
        img = _sensor.snapshot(chn=CAM_CHN_ID_1)
        x0 = max(0, min(640 - ROI, int(prior[0] - ROI // 2)))
        y0 = max(0, min(480 - ROI, int(prior[1] - ROI // 2)))
        r = None
        try:
            crop = img.crop(roi=(x0, y0, ROI, ROI)).to_rgb888().to_numpy_ref()
            r = _lh.detect(crop, x0, y0)
        except Exception:
            r = None
        if r is not None:
            prior = (r[0], r[1])
            hits += 1
        if (n % SAVE_EVERY) == 0:
            try:
                img.save(OUT_DIR + "/s%03d.jpg" % n)
                rows.append("%d,%.1f,%.1f,%.4f" % (
                    n, r[0] if r else -1, r[1] if r else -1,
                    r[2] if r else 0.0))
            except Exception:
                pass
        n += 1
        if (n & 0xF) == 0:
            gc.collect()

    try:
        with open(OUT_DIR + "/coords.csv", "w") as f:
            f.write("frame,x,y,conf\n")
            for row in rows:
                f.write(row + "\n")
    except Exception:
        pass
    print("CAPTURE DONE frames=%d hits=%d saved=%d" % (n, hits, len(rows)))


try:
    main()
except Exception as e:
    print("CAPTURE ERR", type(e).__name__, e)
finally:
    keep = sys.modules.get("_k230_ai_keepalive")
    if keep is None:
        keep = {}
        sys.modules["_k230_ai_keepalive"] = keep
    keep["lh"] = _lh
    _lh = None
    try:
        if _sensor:
            _sensor.stop()
    except Exception:
        pass
    try:
        MediaManager.deinit()
    except Exception:
        pass
    gc.collect()
    print("stopped")
