# -*- coding: utf-8 -*-
"""laser_heatmap_test.py —— 激光热图板端测试（健壮版：异常不 exit，留帧留坐标）

HEADLESS=1：不开 IDE，按 PREVIEW_W/H 做 JPEG 编码（模拟预览），打印 FPS。
HEADLESS=0：CanMV IDE 预览（compress_for_ide）。
每 SAVE_EVERY 帧存一张 /sdcard/laser_dbg/f_xxx.jpg，并写 coords.csv。
"""
MODEL = "/sdcard/models/laser_heatmap.kmodel"
ROI_CX, ROI_CY = 317, 261
ROI = 128
PREVIEW_W, PREVIEW_H = 512, 384
QUALITY = 90
HEADLESS = 1
LIMIT = 90
PRINT_EVERY = 15
SAVE_EVERY = 15
SAVE_DIR = "/sdcard/laser_dbg"

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
        os.mkdir(SAVE_DIR)
    except Exception:
        pass
    _sensor = Sensor(id=2)
    _sensor.reset()
    _sensor.set_framesize(w=640, h=480, chn=CAM_CHN_ID_1)
    _sensor.set_pixformat(Sensor.RGB565, chn=CAM_CHN_ID_1)
    MediaManager.init()
    _sensor.run()
    time.sleep_ms(300)

    _lh = LaserHeat(MODEL, debug=0)
    print("laser_heatmap: %dx%d q%d headless=%d roi=%d" % (
        PREVIEW_W, PREVIEW_H, QUALITY, HEADLESS, ROI))

    x0 = max(0, min(640 - ROI, ROI_CX - ROI // 2))
    y0 = max(0, min(480 - ROI, ROI_CY - ROI // 2))
    n = 0
    win_t = time.ticks_ms()
    win_n = 0
    hits = 0
    rows = []
    errs = 0
    while True:
        t0 = time.ticks_ms()
        img = _sensor.snapshot(chn=CAM_CHN_ID_1)
        r = None
        try:
            crop = img.crop(roi=(x0, y0, ROI, ROI)).to_rgb888().to_numpy_ref()
            r = _lh.detect(crop, x0, y0)
        except Exception as e:
            errs += 1
            if errs <= 3:
                print("detect err:", e)
        if r is not None:
            hits += 1
            img.draw_cross(int(r[0]), int(r[1]), color=(255, 0, 0),
                           size=10, thickness=2)
        if HEADLESS:
            _ = img.compress(quality=QUALITY, x_size=PREVIEW_W, y_size=PREVIEW_H)
        else:
            img.compress_for_ide(x_size=PREVIEW_W, y_size=PREVIEW_H,
                                 quality=QUALITY)
        if (n % SAVE_EVERY) == 0:
            try:
                img.save(SAVE_DIR + "/f_%03d.jpg" % n)
                rows.append("%d,%.1f,%.1f,%.4f" % (
                    n, r[0] if r else -1, r[1] if r else -1,
                    r[2] if r else 0.0))
            except Exception as e:
                print("save err:", e)
        n += 1
        win_n += 1
        if n % PRINT_EVERY == 0:
            now = time.ticks_ms()
            dt = time.ticks_diff(now, win_t)
            fps = win_n * 1000.0 / dt if dt else 0.0
            print("[%d] %.1ffps laser=%s frame_ms=%d hits=%d errs=%d" % (
                n, fps, ("(%.0f,%.0f,%.2f)" % r if r else "-"),
                time.ticks_diff(now, t0), hits, errs))
            win_t = now
            win_n = 0
        if (n & 0xF) == 0:
            gc.collect()
        if LIMIT and n >= LIMIT:
            break

    try:
        with open(SAVE_DIR + "/coords.csv", "w") as f:
            f.write("frame,x,y,conf\n")
            for row in rows:
                f.write(row + "\n")
    except Exception as e:
        print("csv err:", e)
    print("DONE frames=%d hits=%d errs=%d" % (n, hits, errs))


try:
    main()
except KeyboardInterrupt:
    print("interrupted")
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
    except Exception as e:
        print("stop err:", e)
    try:
        MediaManager.deinit()
    except Exception as e:
        print("deinit err:", e)
    gc.collect()
    print("stopped")
