# -*- coding: utf-8 -*-
"""
yolo_preview_ide.py —— 在 CanMV IDE 里看 YOLOv8n-Pose 标靶效果
=============================================================

画面内容：
    * 检测框（只作为特征区域，不要求四角精确）
    * 4 个角点连成的靶纸轮廓
    * 第 5 个关键点：白色 10 号靶心（红十字）

配置：
    MODEL      训练好的 pose kmodel（5 关键点）
    VIEW       "full"=固定全图 640×480（默认，画面稳定不跳动）；"center"=固定中心裁剪；"roi"=旧动态ROI

说明：
    * 不依赖 libs/PipeLine（板端 libs 可能比固件新）；
    * IDE 预览用 img.compress_for_ide()，不初始化 Display；
    * 退出自动复位，避免 KPU 析构死锁。
"""

# ============================ 用户配置 ============================
MODEL = "/sdcard/models/target_pose.kmodel"
LABELS = ["target"]
MODEL_IN = [320, 320]
CAPTURE_SIZE = [640, 480]
DISPLAY_MODE = "ide"          # ide=CanMV IDE；hdmi/lcd=本地屏
VIEW = "full"                 # full=全图固定缩放；center=固定中心裁剪；roi=旧动态ROI
VIEW_SCALE = 0.55            # full 缩放：0.55 -> 352x264（追求 15~20fps）；1.0 -> 640x480
IDE_QUALITY = 80              # IDE 预览 JPEG 质量（85 平衡清晰度/帧率；95 更清晰但更慢）
CENTER_SIZE = 320             # center 模式的固定裁剪边长
CONF = 0.25
NMS = 0.45
MAX_BOXES = 20
KP_NUM = 5                    # 4 角点 + 靶心
KP_DIM = 3                    # x,y,v
VIEW_PAD = 0.25               # ROI 外扩比例
AI_EVERY = 3                 # 每 N 帧跑一次 pose（其余帧复用上次结果）
# K230 -> ESP32 云台 UART（UART1: TX=GPIO3, RX=GPIO4）
UART_ENABLE = 1
UART_ID = 3
UART_TX = 32
UART_RX = 33
UART_BAUD = 115200
PRINT_EVERY = 30
RESET_ON_EXIT = 1
# =================================================================

from media.sensor import *
from media.media import *
import machine
from media.display import Display
import libs.YOLO as _YL
from libs.YOLO import YOLOv8
from ulab import numpy as _np

import sys
import gc
import time

W, H = CAPTURE_SIZE
_sensor = None
_yolo = None


def _display_init():
    if DISPLAY_MODE == "ide":
        print("display: IDE direct (compress_for_ide)")
        return
    if DISPLAY_MODE in ("hdmi", "lt9611"):
        Display.init(Display.LT9611, width=W, height=H, to_ide=False)
    elif DISPLAY_MODE in ("lcd", "st7701"):
        Display.init(Display.ST7701, width=W, height=H, to_ide=False)
    else:
        Display.init(Display.VIRT, width=W, height=H, to_ide=True)


def _pose_postprocess(new_result, active_size, model_size, out_size,
                     nc, kp_num, kp_dim, conf, nms, max_boxes):
    """单目标 top-1 Pose 解码（板端 aidemo 缺 pose 后处理时使用）。"""
    scores = new_result[:, 4]
    idx = int(_np.argmax(scores))
    score = float(scores[idx])
    if score < conf:
        return ([], [], [], [])
    row = new_result[idx]
    ah, aw = out_size
    mh, mw = model_size
    scale = min(mw / float(aw), mh / float(ah))
    nw = int(round(aw * scale)); nh = int(round(ah * scale))
    # AI2D 左上角对齐
    px = 0.0; py = 0.0
    x = (float(row[0]) - px) / scale
    y = (float(row[1]) - py) / scale
    w = float(row[2]) / scale
    h = float(row[3]) / scale
    kp = []
    for k in range(kp_num):
        kp.append((float(row[5 + k * kp_dim]) - px) / scale)
        kp.append((float(row[5 + k * kp_dim + 1]) - py) / scale)
        kp.append(float(row[5 + k * kp_dim + 2]))
    return ([_np.array([x, y, w, h])], [0], [score], [_np.array(kp)])


class _PoseAidemo:
    def yolov8_pose_postprocess(self, new_result, active_size, model_size,
                                out_size, nc, kp_num, kp_dim, conf, nms,
                                max_boxes):
        return _pose_postprocess(new_result, active_size, model_size, out_size,
                                 nc, kp_num, kp_dim, conf, nms, max_boxes)


_YL.aidemo = _PoseAidemo()

_uart = None
_uart_warned = False


def _init_uart():
    global _uart
    if not UART_ENABLE:
        return
    try:
        fpioa = machine.FPIOA()
        if UART_ID == 3:
            fpioa.set_function(UART_TX, fpioa.UART3_TXD, ie=0, oe=1)
            fpioa.set_function(UART_RX, fpioa.UART3_RXD, ie=1, oe=0)
            unum = machine.UART.UART3
        else:
            fpioa.set_function(UART_TX, fpioa.UART1_TXD, ie=0, oe=1)
            fpioa.set_function(UART_RX, fpioa.UART1_RXD, ie=1, oe=0)
            unum = machine.UART.UART1
        _uart = machine.UART(unum, baudrate=UART_BAUD,
                     bits=machine.UART.EIGHTBITS, parity=machine.UART.PARITY_NONE,
                     stop=machine.UART.STOPBITS_ONE)
        print("uart%d tx=%d rx=%d baud=%d" % (UART_ID, UART_TX, UART_RX, UART_BAUD))
    except Exception as e:
        _uart = None
        print("uart init failed:", e)


def _send_uart(seq, res):
    global _uart_warned
    if _uart is None:
        if not _uart_warned:
            _uart_warned = True
            print("uart not ready: $KT 未发送（检查 UART_ENABLE / UART3 引脚）")
        return
    try:
        run = 1 if (res and res[0]) else 0
        if run:
            b = res[0][0]
            # b = [x, y, w, h]（左上角+宽高）→ 协议要求框中心
            cx = int(b[0] + b[2] * 0.5)
            cy = int(b[1] + b[3] * 0.5)
            kp = res[3][0]
            if len(kp) >= 15 and float(kp[14]) >= 0.3:
                bx, by = int(kp[12]), int(kp[13])
                conf = int(float(kp[14]) * 1000)
            else:
                bx = by = -1
                conf = 0
        else:
            cx = cy = bx = by = -1
            conf = 0
        _uart.write("$KT,%d,%d,%d,%d,%d,%d,%d!\n" % (seq, run, cx, cy, bx, by, conf))
        _uart.flush()
    except Exception as e:
        print("uart tx:", e)


def _draw(img, res, s=1.0):
    """画框、四角轮廓、靶心。"""
    if not res or not res[0]:
        return None
    boxes, ids, scores, kpts = res[0], res[1], res[2], res[3]
    for i in range(len(boxes)):
        b = boxes[i]
        x, y, w, h = [int(round(float(v) * s)) for v in b[:4]]
        img.draw_rectangle(x, y, w, h, color=(0, 200, 0), thickness=max(1, int(2 * s)))
        kp = kpts[i]
        pts = []
        for k in range(KP_NUM):
            px = float(kp[k * KP_DIM + 0])
            py = float(kp[k * KP_DIM + 1])
            pv = float(kp[k * KP_DIM + 2])
            if pv >= 0.3:
                pts.append((int(round(px * s)), int(round(py * s))))
            else:
                pts.append(None)
        quad = [p for p in pts[:4] if p is not None]
        n = len(quad)
        if n >= 2:
            for k in range(n):
                x1, y1 = quad[k]
                x2, y2 = quad[(k + 1) % n]
                img.draw_line(x1, y1, x2, y2, color=(0, 255, 0), thickness=max(1, int(2 * s)))
        for p in quad:
            img.draw_circle(p[0], p[1], max(2, int(3 * s)), color=(0, 255, 0), thickness=max(1, int(2 * s)))
        if len(pts) >= 5 and pts[4] is not None:
            bx, by = pts[4]
            img.draw_circle(bx, by, max(3, int(7 * s)), color=(255, 0, 0), thickness=max(1, int(2 * s)))
            img.draw_line(bx - int(12 * s), by, bx + int(12 * s), by, color=(255, 0, 0), thickness=max(1, int(2 * s)))
            img.draw_line(bx, by - int(12 * s), bx, by + int(12 * s), color=(255, 0, 0), thickness=max(1, int(2 * s)))
    return boxes[0]


def main():
    global _sensor, _yolo
    _sensor = Sensor(id=2)
    _sensor.reset()
    _sensor.set_framesize(w=W, h=H, chn=CAM_CHN_ID_1)
    _sensor.set_pixformat(Sensor.RGB565, chn=CAM_CHN_ID_1)
    _sensor.set_framesize(w=W, h=H, chn=CAM_CHN_ID_2)
    _sensor.set_pixformat(Sensor.RGBP888, chn=CAM_CHN_ID_2)
    MediaManager.init()
    _sensor.run()
    try:
        gc.threshold(96 * 1024)
    except Exception:
        pass
    time.sleep_ms(300)

    _display_init()
    if DISPLAY_MODE != "ide":
        print("display", Display.width(), Display.height())

    _yolo = YOLOv8(task_type="pose", mode="image", kmodel_path=MODEL,
                   labels=LABELS, rgb888p_size=[W, H],
                   model_input_size=MODEL_IN, display_size=[W, H],
                   conf_thresh=CONF, nms_thresh=NMS,
                   kp_num=KP_NUM, kp_dim=KP_DIM,
                   max_boxes_num=MAX_BOXES, debug_mode=0)
    _yolo.config_preprocess()

    # 云台链路：必须在主循环前初始化，否则 _send_uart() 静默不发帧
    _init_uart()

    n = 0
    t_last = time.ticks_ms()
    fps = 0.0
    last_res = None
    while True:
        # 先拍显示帧，再拍 AI 帧（仅在 AI 帧执行）
        img = _sensor.snapshot(chn=CAM_CHN_ID_1)
        if last_res is None or (n % AI_EVERY) == 0:
            arr = _sensor.snapshot(chn=CAM_CHN_ID_2).to_numpy_ref()
            last_res = _yolo.run(arr)
        res = last_res
        _send_uart(n, res)

        if DISPLAY_MODE == "ide":
            if VIEW == "full":
                # 先缩放到固定尺寸，再画覆盖层：绘制成本降数倍，帧率更稳
                disp = img.copy(x_scale=VIEW_SCALE, y_scale=VIEW_SCALE)
                box = _draw(disp, res, VIEW_SCALE)
                disp.compress_for_ide(quality=IDE_QUALITY)
                roi = None
            else:
                box = _draw(img, res)
                roi = None
                if box is not None:
                    bx, by, bw, bh = [int(v) for v in box[:4]]
                    pad = int(max(bw, bh) * VIEW_PAD)
                    rx = max(0, bx - pad)
                    ry = max(0, by - pad)
                    rw = min(W - rx, bw + pad * 2)
                    rh = min(H - ry, bh + pad * 2)
                    roi = (rx, ry, rw, rh)
                if VIEW == "center":
                    size = min(CENTER_SIZE, W, H)
                    vx = (W - size) // 2
                    vy = (H - size) // 2
                    img.compress_for_ide(roi=(vx, vy, size, size), quality=IDE_QUALITY)
                elif VIEW == "roi" and roi is not None:
                    img.compress_for_ide(roi=roi, quality=IDE_QUALITY)
                else:
                    img.compress_for_ide(x_scale=VIEW_SCALE, y_scale=VIEW_SCALE, quality=IDE_QUALITY)
        else:
            Display.show_image(img)

        n += 1
        dt = time.ticks_diff(time.ticks_ms(), t_last)
        t_last = time.ticks_ms()
        if dt > 0:
            fps = 0.8 * fps + 0.2 * (1000.0 / dt) if fps else 1000.0 / dt
        if PRINT_EVERY and (n % PRINT_EVERY == 0):
            if res and res[0]:
                kp = res[3][0]
                bx, by, bv = float(kp[12]), float(kp[13]), float(kp[14])
                print("[%d] %.1ffps target %.2f  kp4(%.0f,%.0f,v%.2f)  roi=%s" % (
                    n, fps, float(res[2][0]), bx, by, bv, roi))
            else:
                print("[%d] %.1ffps 无目标" % (n, fps))
        if (n & 0x07) == 0:
            gc.collect()


def _cleanup():
    global _yolo, _sensor
    keep = sys.modules.get("_k230_ai_keepalive")
    if keep is None:
        keep = {}
        sys.modules["_k230_ai_keepalive"] = keep
    keep["yolo"] = _yolo
    keep["kpu"] = getattr(_yolo, "kpu", None)
    keep["ai2d"] = getattr(_yolo, "ai2d", None)
    keep["tensors"] = getattr(_yolo, "tensors", None)
    _yolo = None
    if DISPLAY_MODE != "ide":
        try:
            Display.deinit()
        except Exception as e:
            print("display.deinit:", e)
    try:
        if _sensor:
            _sensor.stop()
    except Exception as e:
        print("sensor.stop:", e)
    _sensor = None
    try:
        MediaManager.deinit()
    except Exception as e:
        print("media.deinit:", e)
    gc.collect()


try:
    main()
except KeyboardInterrupt:
    print("interrupted")
except Exception as e:
    print("ERROR", type(e).__name__, e)
    try:
        sys.print_exception(e)
    except Exception:
        pass
finally:
    _cleanup()
    print("stopped")
    if RESET_ON_EXIT:
        time.sleep_ms(100)
        import machine
        machine.reset()
