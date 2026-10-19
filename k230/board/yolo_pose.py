# -*- coding: utf-8 -*-
"""
K230 板端 Pose 标靶识别脚本（关键点 + 20 FPS）
========================================

流程（每帧）：
    1. chn2 快照 -> YOLOv8 检测（320 输入，约 28 ms）
    2. chn1 快照 -> 在检测框 ROI 内定位白色 10 号靶心（find_circles + 径向对称兜底）
    3. 隔帧编码 ROI JPEG（原生分辨率，不降采样）并回传
    4. 每 N 帧回传一张全图（缩小到 320x240）用于上下文

回传 JSON 额外字段：
    roi    本次 JPEG 对应的原图区域 [x,y,w,h]
    jpeg   本次 JPEG 的实际尺寸 [w,h]
    bull   靶心 {x,y,r,conf,status}（坐标为原图 640x480 坐标系）
    ms     推理耗时；bs 靶心耗时

CFG 必须保持单行，PC 端会用正则整行替换后下发。
"""

CFG = {'w': 640, 'h': 480, 'q': 70, 'model': '/sdcard/models/target_pose.kmodel', 'labels': ['target'], 'in_w': 320, 'in_h': 320, 'conf': 0.25, 'cand_conf': 0.03, 'far_refine': 0, 'far_min': 100, 'far_topk': 3, 'far_max': 200, 'nms': 0.45, 'kp_num': 5, 'kp_dim': 3, 'max_boxes': 20, 'jpeg_every': 1, 'full_every': 15, 'roi_pad': 0.25, 'roi_min': 160, 'roi_max': 256, 'fps': 0, 'stats': 10, 'limit': 0, 'sensor_id': 2, 'reset_on_exit': 1, 'uart_enable': 1, 'uart_id': 3, 'uart_tx': 32, 'uart_rx': 33, 'uart_baud': 115200}

from media.sensor import *
from media.media import *
import libs.YOLO as _YL
from libs.YOLO import YOLOv8
from ulab import numpy as _np
import machine

import sys
import time
import gc
import binascii
import ujson
import math

MAGIC = b"\xAA\x55K2"
HDR_LEN = 38
T_FRAME = 1
T_LOG = 2
T_STATS = 3
T_HELLO = 4
T_ERR = 5
T_TARGET = 6
TYPE_CHAR = {1: b"F", 2: b"L", 3: b"S", 4: b"H", 5: b"E", 6: b"T"}

_RAW = sys.stdout.buffer if hasattr(sys.stdout, "buffer") else sys.stdout


def _write_all(data):
    if not isinstance(data, (bytes, bytearray)):
        data = bytes(data)
    total = 0
    size = len(data)
    while total < size:
        chunk = data if total == 0 else data[total:]
        try:
            n = _RAW.write(chunk)
        except Exception:
            return False
        if n is None:
            return True
        if n <= 0:
            return False
        total += n
    return True


def emit(typename, seq, payload):
    if payload is None:
        payload = b""
    if not isinstance(payload, (bytes, bytearray)):
        payload = bytes(payload)
    payload = bytes(payload)
    esc = payload.replace(b"\r", b"\r\r")
    wlen = len(esc) + esc.count(b"\n")
    meta = "%08X%08X%08X%08X" % (seq & 0xFFFFFFFF, len(payload), wlen,
                                 binascii.crc32(payload) & 0xFFFFFFFF)
    hdr = MAGIC + TYPE_CHAR[typename] + meta.encode() + b"|"
    return _write_all(hdr + esc)


def log(msg):
    emit(T_LOG, 0, ("[k230-pose] " + msg).encode())


# ---------------------------------------------------------------- 传感器
_sensor = None
_yolo = None


def _pose_postprocess(new_result, active_size, model_size, out_size,
                     nc, kp_num, kp_dim, conf, nms, max_boxes):
    """板端 Pose 解码：低阈值返回 top-K 候选，供后续裁剪放大复筛。"""
    scores = new_result[:, 4]
    ah, aw = out_size
    mh, mw = model_size
    scale = min(mw / float(aw), mh / float(ah))
    px = 0.0
    py = 0.0

    boxes = []
    ids = []
    out_scores = []
    kpts = []
    topk = int(CFG.get("far_topk", 3))
    for _ in range(topk):
        idx = int(_np.argmax(scores))
        score = float(scores[idx])
        if score < conf:
            break
        row = new_result[idx]
        x = (float(row[0]) - px) / scale
        y = (float(row[1]) - py) / scale
        w = float(row[2]) / scale
        h = float(row[3]) / scale
        kp = []
        for k in range(kp_num):
            kp.append((float(row[5 + k * kp_dim]) - px) / scale)
            kp.append((float(row[5 + k * kp_dim + 1]) - py) / scale)
            kp.append(float(row[5 + k * kp_dim + 2]))
        boxes.append(_np.array([x, y, w, h]))
        ids.append(0)
        out_scores.append(score)
        kpts.append(_np.array(kp))
        scores[idx] = -1.0
    return (boxes, ids, out_scores, kpts)


class _PoseAidemo:
    """替换 libs.YOLO 里缺失的 aidemo.yolov8_pose_postprocess。"""
    def yolov8_pose_postprocess(self, new_result, active_size, model_size,
                                out_size, nc, kp_num, kp_dim, conf, nms,
                                max_boxes):
        return _pose_postprocess(new_result, active_size, model_size, out_size,
                                 nc, kp_num, kp_dim, conf, nms, max_boxes)


_YL.aidemo = _PoseAidemo()


def _check_pose(box, kp, kp_dim=3):
    """几何/置信度校验；返回 ok 或失败原因。"""
    if len(kp) < 4 * kp_dim:
        return "no_corners"
    pts = []
    for k in range(4):
        x = float(kp[k * kp_dim])
        y = float(kp[k * kp_dim + 1])
        v = float(kp[k * kp_dim + 2])
        if v < 0.3:
            return "corner_low_conf"
        pts.append((x, y))
    # 凸性
    signs = []
    for i in range(4):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % 4]
        cx, cy = pts[(i + 2) % 4]
        cr = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if abs(cr) < 1e-3:
            return "not_convex"
        signs.append(1 if cr > 0 else -1)
    if len(set(signs)) != 1:
        return "not_convex"
    # 边长
    sides = []
    for i in range(4):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % 4]
        sides.append(math.sqrt((bx - ax) ** 2 + (by - ay) ** 2))
    if min(sides) < 8:
        return "bad_geometry"
    r1 = max(sides[0], sides[2]) / max(1e-6, min(sides[0], sides[2]))
    r2 = max(sides[1], sides[3]) / max(1e-6, min(sides[1], sides[3]))
    if r1 > 2.5 or r2 > 2.5:
        return "bad_geometry"
    # 出框
    out = 0
    for x, y in pts:
        if x < 0 or x >= CFG["w"] or y < 0 or y >= CFG["h"]:
            out += 1
    if out > 1:
        return "corners_out"
    # 靶心
    if len(kp) < 5 * kp_dim:
        return "no_bullseye"
    bx, by, bv = float(kp[12]), float(kp[13]), float(kp[14])
    if bv < 0.3:
        return "bull_low_conf"
    x1 = min(p[0] for p in pts); x2 = max(p[0] for p in pts)
    y1 = min(p[1] for p in pts); y2 = max(p[1] for p in pts)
    mx = (x2 - x1) * 0.05 + 4
    my = (y2 - y1) * 0.05 + 4
    if not (x1 - mx <= bx <= x2 + mx and y1 - my <= by <= y2 + my):
        return "bull_outside"
    return "ok"


def _open_sensor():
    global _sensor
    _sensor = Sensor(id=CFG["sensor_id"])
    _sensor.reset()
    _sensor.set_framesize(w=CFG["w"], h=CFG["h"], chn=CAM_CHN_ID_1)
    _sensor.set_pixformat(Sensor.RGB565, chn=CAM_CHN_ID_1)
    _sensor.set_framesize(w=CFG["w"], h=CFG["h"], chn=CAM_CHN_ID_2)
    _sensor.set_pixformat(Sensor.RGBP888, chn=CAM_CHN_ID_2)
    MediaManager.init()
    _sensor.run()
    return _sensor


def _close():
    global _yolo, _sensor
    mod = sys.modules.get("_k230_ai_keepalive")
    if mod is None:
        mod = {}
        sys.modules["_k230_ai_keepalive"] = mod
    mod["yolo"] = _yolo
    mod["kpu"] = getattr(_yolo, "kpu", None)
    mod["ai2d"] = getattr(_yolo, "ai2d", None)
    mod["tensors"] = getattr(_yolo, "tensors", None)
    _yolo = None
    try:
        if _sensor:
            _sensor.stop()
    except Exception as e:
        log("sensor.stop: %s" % e)
    _sensor = None
    try:
        MediaManager.deinit()
    except Exception as e:
        log("media.deinit: %s" % e)
    gc.collect()


# ---------------------------------------------------------------- 靶心
def _clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _encode_roi(img, roi, q):
    """编码 ROI（原生分辨率），返回 (jpeg_bytes, [w,h])。"""
    x, y, w, h = roi
    j = img.compress(quality=q, roi=(x, y, w, h))
    if not isinstance(j, bytes):
        j = bytes(j)
    return j, [w, h]


def _encode_full(img, q):
    """全图缩小一半（320x240）编码，返回 (jpeg_bytes, [w,h], roi)。"""
    j = img.compress(quality=q, x_scale=0.5, y_scale=0.5)
    if not isinstance(j, bytes):
        j = bytes(j)
    return j, [CFG["w"] // 2, CFG["h"] // 2], [0, 0, CFG["w"], CFG["h"]]


def det_from(boxes, ids, scores):
    det = []
    for i in range(len(boxes)):
        b = boxes[i]
        det.append([int(round(b[0])), int(round(b[1])),
                    int(round(b[2])), int(round(b[3])),
                    round(float(scores[i]), 3), int(ids[i])])
    return det


def _roi_around(box):
    """根据检测框算 ROI（带 margin、clamp、最小/最大尺寸）。"""
    W = CFG["w"]
    H = CFG["h"]
    bx, by, bw, bh = [int(v) for v in box[:4]]
    pad = int(max(bw, bh) * CFG["roi_pad"])
    x = bx - pad
    y = by - pad
    w = bw + pad * 2
    h = bh + pad * 2
    # 至少 roi_min，最大 roi_max
    if w < CFG["roi_min"]:
        d = (CFG["roi_min"] - w) // 2
        x -= d
        w = CFG["roi_min"]
    if h < CFG["roi_min"]:
        d = (CFG["roi_min"] - h) // 2
        y -= d
        h = CFG["roi_min"]
    if w > CFG["roi_max"]:
        d = (w - CFG["roi_max"]) // 2
        x += d
        w = CFG["roi_max"]
    if h > CFG["roi_max"]:
        d = (h - CFG["roi_max"]) // 2
        y += d
        h = CFG["roi_max"]
    x = _clamp(x, 0, W - 8)
    y = _clamp(y, 0, H - 8)
    w = _clamp(w, 8, W - x)
    h = _clamp(h, 8, H - y)
    return [x, y, w, h]


def _refine(img, box):
    """对候选框裁剪放大到 640x480 再跑一次，返回 (score, box, kp, status)。"""
    bx, by, bw, bh = [float(v) for v in box[:4]]
    side = max(bw, bh) * 3.5
    if side < 64:
        side = 64
    cw = int(side)
    ch = int(side * 0.75)
    cx = bx + bw / 2.0
    cy = by + bh / 2.0
    x = int(_clamp(cx - cw / 2.0, 0, CFG["w"] - 8))
    y = int(_clamp(cy - ch / 2.0, 0, CFG["h"] - 8))
    cw = int(_clamp(cw, 16, CFG["w"] - x))
    ch = int(_clamp(ch, 16, CFG["h"] - y))
    try:
        crop = img.crop(roi=(x, y, cw, ch))
        zoom = crop.copy(x_scale=CFG["w"] / float(cw),
                         y_scale=CFG["h"] / float(ch))
        arr = zoom.to_rgb888().to_numpy_ref()
        res = _yolo.run(arr)
        if not res or not res[0]:
            return None
        b = res[0][0]
        kp = res[3][0]
        score = float(res[2][0])
        sx = cw / float(CFG["w"])
        sy = ch / float(CFG["h"])
        nb = [x + float(b[0]) * sx, y + float(b[1]) * sy,
              float(b[2]) * sx, float(b[3]) * sy]
        nkp = []
        for i in range(0, len(kp), 3):
            nkp.append(x + float(kp[i]) * sx)
            nkp.append(y + float(kp[i + 1]) * sy)
            nkp.append(float(kp[i + 2]))
        st = _check_pose(nb, nkp, CFG["kp_dim"])
        return (score, nb, nkp, st)
    except Exception as e:
        log("refine: %s" % e)
        return None


# ---------------------------------------------------------------- K230 -> ESP32 UART
_uart = None
_uart_failed = False


def _init_uart():
    """UART1: TX=GPIO3, RX=GPIO4（与 ESP32 UART2 交叉接线）。"""
    global _uart, _uart_failed
    if not CFG.get("uart_enable", 0):
        log("uart disabled")
        return
    try:
        fpioa = machine.FPIOA()
        if CFG.get("uart_id", 1) == 3:
            fpioa.set_function(CFG["uart_tx"], fpioa.UART3_TXD, ie=0, oe=1)
            fpioa.set_function(CFG["uart_rx"], fpioa.UART3_RXD, ie=1, oe=0)
            unum = machine.UART.UART3
        else:
            fpioa.set_function(CFG["uart_tx"], fpioa.UART1_TXD, ie=0, oe=1)
            fpioa.set_function(CFG["uart_rx"], fpioa.UART1_RXD, ie=1, oe=0)
            unum = machine.UART.UART1
        _uart = machine.UART(unum, baudrate=CFG["uart_baud"],
                     bits=machine.UART.EIGHTBITS, parity=machine.UART.PARITY_NONE,
                     stop=machine.UART.STOPBITS_ONE)
        log("uart%d tx=%d rx=%d baud=%d" % (CFG.get("uart_id", 1), CFG["uart_tx"], CFG["uart_rx"], CFG["uart_baud"]))
    except Exception as e:
        _uart = None
        _uart_failed = True
        log("uart init failed: %s" % e)


def _send_uart(seq, det, bull, has_det):
    """$KT,<seq>,<run>,<cx>,<cy>,<bx>,<by>,<conf>!"""
    if _uart is None:
        return
    try:
        if has_det and det:
            # det[0] = [x, y, w, h, score, id]（左上角+宽高）→ 协议要框中心
            cx = int(det[0][0] + det[0][2] * 0.5)
            cy = int(det[0][1] + det[0][3] * 0.5)
            run = 1
        else:
            cx = cy = -1
            run = 0
        conf = 0
        if bull and bull.get("status") == "ok":
            bx = int(bull.get("x", -1))
            by = int(bull.get("y", -1))
            conf = int(float(bull.get("conf", 0.0)) * 1000)
        else:
            bx = by = -1
        line = "$KT,%d,%d,%d,%d,%d,%d,%d!\n" % (seq, run, cx, cy, bx, by, conf)
        _uart.write(line)
        _uart.flush()
    except Exception as e:
        log("uart tx: %s" % e)


# ---------------------------------------------------------------- 主流程
def main():
    global _yolo
    sen = _open_sensor()
    time.sleep_ms(300)
    _yolo = YOLOv8(task_type="pose", mode="image",
                   kmodel_path=CFG["model"], labels=CFG["labels"],
                   rgb888p_size=[CFG["w"], CFG["h"]],
                   model_input_size=[CFG["in_w"], CFG["in_h"]],
                   display_size=[CFG["w"], CFG["h"]],
                   conf_thresh=CFG["cand_conf"], nms_thresh=CFG["nms"],
                   kp_num=CFG["kp_num"], kp_dim=CFG["kp_dim"],
                   max_boxes_num=CFG["max_boxes"], debug_mode=0)
    _yolo.config_preprocess()
    _init_uart()

    try:
        stype = str(sen.get_type())
    except Exception:
        stype = "?"
    emit(T_HELLO, 0, ("pose w=%d h=%d q=%d model=%s mem=%d sensor=%s version=%s"
                      % (CFG["w"], CFG["h"], CFG["q"], CFG["model"],
                         gc.mem_free(), stype, sys.version)).encode())
    log("pose streaming %dx%d q=%d model=%s" %
        (CFG["w"], CFG["h"], CFG["q"], CFG["model"]))
    try:
        gc.threshold(96 * 1024)
    except Exception:
        pass

    seq = 0
    win_t = time.ticks_ms()
    win_n = 0
    every = CFG["stats"]
    target_fps = CFG["fps"]

    while True:
        t0 = time.ticks_ms()
        seq += 1

        # ---- 先拍显示帧，再拍 AI 帧（避免 ROI/检测框与画面错位）
        img = sen.snapshot(chn=CAM_CHN_ID_1)
        arr = sen.snapshot(chn=CAM_CHN_ID_2).to_numpy_ref()
        t_ai = time.ticks_ms()
        res = _yolo.run(arr)
        infer_ms = time.ticks_diff(time.ticks_ms(), t_ai)
        if res and len(res) >= 4:
            boxes, ids, scores, kpts = res[0], res[1], res[2], res[3]
        else:
            boxes, ids, scores, kpts = [], [], [], []
        has_det = len(boxes) > 0

        # ---- 候选选择 + 远距离裁剪放大复筛
        kp_flat = []
        bull = {"status": "no_target"}
        det = []
        refine_ms = 0
        best = None
        if has_det:
            for i in range(len(boxes)):
                st = _check_pose(boxes[i], kpts[i], CFG["kp_dim"])
                if st == "ok":
                    if best is None or float(scores[i]) > best[0]:
                        best = (float(scores[i]), boxes[i], kpts[i], st)
            top_score = max(float(v) for v in scores) if scores else 0.0
            small = best is not None and min(float(best[1][2]), float(best[1][3])) < CFG["far_min"]
            if CFG["far_refine"] and (top_score < CFG["conf"] or small):
                t_r = time.ticks_ms()
                rb = None
                for i in range(min(len(boxes), int(CFG["far_topk"]))):
                    if max(float(boxes[i][2]), float(boxes[i][3])) > CFG["far_max"]:
                        continue
                    r = _refine(img, boxes[i])
                    if r is not None and r[3] == "ok":
                        if rb is None or r[0] > rb[0]:
                            rb = r
                refine_ms = time.ticks_diff(time.ticks_ms(), t_r)
                if rb is not None and (best is None or rb[0] > best[0] + 0.03):
                    best = rb

            if best is not None and best[0] >= CFG["conf"]:
                kp = best[2]
                box = best[1]
                for i in range(len(kp)):
                    kp_flat.append(round(float(kp[i]), 2))
                bull = {"x": round(float(kp[12]), 1), "y": round(float(kp[13]), 1),
                        "conf": round(float(kp[14]), 3), "status": "ok",
                        "src": "pose+refine" if refine_ms else "pose"}
                det = [[int(round(box[0])), int(round(box[1])),
                        int(round(box[2])), int(round(box[3])),
                        round(float(best[0]), 3), 0]]
                has_det = True
            else:
                bull = {"status": "low_conf"}
                has_det = False

        # ---- 图像回传：ROI 原生分辨率；全图每 full_every 帧或丢失目标
        want_full = (seq % CFG["full_every"] == 0) or (not has_det)
        want_jpeg = (seq % CFG["jpeg_every"] == 0) or want_full
        jpg = None
        roi = None
        jsz = None
        jpg_ms = 0
        if want_jpeg:
            t_j = time.ticks_ms()
            if want_full:
                jpg, jsz, roi = _encode_full(img, CFG["q"])
            else:
                roi = _roi_around(boxes[0])
                jpg, jsz = _encode_roi(img, roi, CFG["q"])
            jpg_ms = time.ticks_diff(time.ticks_ms(), t_j)

        out = {"seq": seq, "w": CFG["w"], "h": CFG["h"],
               "ms": infer_ms, "ref_ms": refine_ms, "det": det, "kp": kp_flat,
               "labels": CFG["labels"], "task": "pose",
               "roi": roi, "jpeg": jsz, "bull": bull}
        emit(T_TARGET, seq, ujson.dumps(out).encode())
        _send_uart(seq, det, bull, has_det)
        if jpg is not None:
            emit(T_FRAME, seq, jpg)

        win_n += 1
        if every and (seq % every == 0):
            now = time.ticks_ms()
            dt = time.ticks_diff(now, win_t)
            fps = 0.0 if dt <= 0 else (win_n * 1000.0 / dt)
            emit(T_STATS, seq, ("seq=%d fps=%.1f infer_ms=%d jpg_ms=%d bytes=%d mem=%d bull=%s"
                                % (seq, fps, infer_ms, jpg_ms,
                                   len(jpg) if jpg else 0, gc.mem_free(),
                                   bull.get("status"))).encode())
            win_t = now
            win_n = 0

        if CFG["limit"] and seq >= CFG["limit"]:
            log("limit %d reached" % CFG["limit"])
            break
        if (seq & 0x07) == 0:
            gc.collect()
        if target_fps:
            due = time.ticks_add(time.ticks_ms(), 1000 // target_fps)
            delay = time.ticks_diff(due, time.ticks_ms())
            if delay > 0:
                time.sleep_ms(delay)

def _safe_exit():
    try:
        _close()
    except Exception as e:
        log("close: %s" % e)
    log("stopped, camera released")
    if CFG["reset_on_exit"]:
        try:
            time.sleep_ms(150)
            import machine
            machine.reset()
        except Exception as e:
            log("reset: %s" % e)


try:
    main()
except KeyboardInterrupt:
    log("interrupted by Ctrl-C")
except Exception as e:
    emit(T_ERR, 0, ("%s: %s" % (type(e).__name__, e)).encode())
    try:
        sys.print_exception(e)
    except Exception:
        print("ERROR", type(e).__name__, e)
finally:
    _safe_exit()
