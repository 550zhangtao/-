# -*- coding: utf-8 -*-
"""laser_sweep_capture.py —— 云台慢速扫动 + 激光采集/留帧

边发 $KT（慢速正弦目标）让 ESP32 云台跟着摆，边用热图模型检测激光，
每 SAVE_EVERY 帧存一张 /sdcard/sweep/，并写 coords.csv。

前提：ESP32 已在 GIMBAL_TRACK（PC 侧 /dev/ttyUSB0 发 `gtrack`）。
"""
MODEL = "/sdcard/models/laser_heatmap.kmodel"
ROI = 128
PRIOR = (317, 261)
DURATION_S = 30
KT_HZ = 20
SAVE_EVERY = 8
OUT_DIR = "/sdcard/sweep"

from media.sensor import *
from media.media import *

import sys
import time
import gc
import os
import math

sys.path.insert(0, "/sdcard")
sys.modules.pop("laser_heatmap", None)
from laser_heatmap import LaserHeat

import machine

_sensor = None
_lh = None
_uart = None


def _init_uart():
    global _uart
    fpioa = machine.FPIOA()
    fpioa.set_function(32, fpioa.UART3_TXD, ie=0, oe=1)
    fpioa.set_function(33, fpioa.UART3_RXD, ie=1, oe=0)
    _uart = machine.UART(machine.UART.UART3, baudrate=115200,
                         bits=machine.UART.EIGHTBITS,
                         parity=machine.UART.PARITY_NONE,
                         stop=machine.UART.STOPBITS_ONE)


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
    _init_uart()

    prior = (float(PRIOR[0]), float(PRIOR[1]))
    t0 = time.ticks_ms()
    n = 0
    seq = 0
    rows = []
    hits = 0
    period = 1000 // KT_HZ
    while True:
        el = time.ticks_diff(time.ticks_ms(), t0) / 1000.0
        if el > DURATION_S:
            break
        # 慢速正弦目标：pan ~8s 周期，tilt ~11s 周期
        cx = 320 + 90 * math.sin(2 * math.pi * el / 8.0)
        cy = 240 + 55 * math.sin(2 * math.pi * el / 11.0)
        if _uart is not None:
            try:
                _uart.write("$KT,%d,1,%d,%d,%d,%d,900!\n" % (
                    seq, int(cx), int(cy), int(cx), int(cy)))
                _uart.flush()
            except Exception:
                pass
        seq += 1

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
        # 限速到 KT_HZ
        dt = time.ticks_diff(time.ticks_ms(), t0) - n * period
        if dt < 0:
            time.sleep_ms(min(20, -dt))

    # 回中
    for _ in range(12):
        try:
            _uart.write("$KT,%d,1,320,240,320,240,900!\n" % seq)
            _uart.flush()
        except Exception:
            pass
        seq += 1
        time.sleep_ms(50)
    try:
        with open(OUT_DIR + "/coords.csv", "w") as f:
            f.write("frame,x,y,conf\n")
            for row in rows:
                f.write(row + "\n")
    except Exception:
        pass
    print("SWEEP DONE frames=%d hits=%d saved=%d" % (n, hits, len(rows)))


try:
    main()
except Exception as e:
    print("SWEEP ERR", type(e).__name__, e)
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
