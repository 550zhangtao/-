# -*- coding: utf-8 -*-
"""
K230 板端 Pose 标靶识别脚本（关键点 + 高帧率）
========================================

流程（每帧）：
    1. chn2 快照 -> YOLOv8-Pose 推理（384 输入，约 38 ms）
    2. 隔帧编码 ROI JPEG（原生分辨率，不降采样）并回传
    3. 每 N 帧回传一张全图（缩小到 320x240）用于上下文

回传 JSON 字段：
    roi    本次 JPEG 对应的原图区域 [x,y,w,h]
    jpeg   本次 JPEG 的实际尺寸 [w,h]
    bull   靶心 {x,y,conf,status}（坐标为原图 640x480 坐标系）
    kp     5 个关键点扁平数组 [x,y,v] * 5（TL,TR,BR,BL,靶心）
    ms     推理耗时；ref_ms 复筛耗时

CFG 必须保持单行，PC 端会用正则整行替换后下发。

目标另外走 UART（给 ESP32 云台）：

    $KT,<seq>,<run>,<cx>,<cy>,<bx>,<by>,<conf>!
        cx,cy = 检测框中心；bx,by = 靶心；run=0 表示本帧无目标。
"""

# 2026-10-23 延迟实测后的默认值：ai_every=1（每帧都发 $KT）+ jpeg_every=3 +
# enc_scale=0.5（把 65ms 的全图编码摊到 1/3 帧，降到 ~32ms）。
# 只改 ai_every=1 不降 enc_scale 反而更慢，见 docs/TRACKING_LATENCY_PLAN.md §8。
CFG = {'w': 640, 'h': 480, 'q': 70, 'model': '/sdcard/models/pose_v3_384.kmodel', 'labels': ['target'], 'in_w': 384, 'in_h': 384, 'conf': 0.25, 'cand_conf': 0.03, 'far_refine': 0, 'far_min': 100, 'far_topk': 3, 'far_max': 200, 'low_refine': 1, 'low_refine_topk': 2, 'low_refine_mult': 7.0, 'low_refine_max': 120, 'low_refine_cooldown': 3, 'nms': 0.45, 'kp_num': 5, 'kp_dim': 3, 'max_boxes': 20, 'ai_every': 1, 'jpeg_every': 3, 'full_every': 9999, 'roi_scale': 0.5, 'roi_pad': 0.25, 'roi_min': 192, 'roi_max': 288, 'perf_mode': 'performance', 'perf_fb': 3, 'perf_fps': 0, 'jpeg_q': 92, 'enc_scale': 0.5, 'fps': 0, 'stats': 15, 'limit': 0, 'sensor_id': 2, 'reset_on_exit': 1, 'uart_enable': 1, 'uart_id': 3, 'uart_tx': 32, 'uart_rx': 33, 'uart_baud': 115200}

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

for _p in ("/sdcard", "/sdcard/board", "/"):
    try:
        if _p in sys.path:
            sys.path.remove(_p)
        sys.path.insert(0, _p)
    except Exception:
        pass

try:
    import k230_perf as perf
except Exception as _e:
    perf = None
    print("k230_perf 导入失败:", _e)

try:
    from point_kalman import PointKalman
except Exception as _e:
    PointKalman = None
    print("point_kalman 导入失败，靶心/激光不平滑:", _e)

_lref_skip = 0        # 低置信复筛冷却（剩下几帧内不再复筛）

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
        # YOLO 输出是 [cx, cy, w, h]（中心点+宽高），不是左上角！
        # 旧代码把 cx,cy 当 x,y，框整体右下偏 w/2,h/2（实测 26x36 的靶纸
        # 偏 11~18px），$KT 的框中心、ROI、复筛裁剪全都跟着偏。
        w = float(row[2]) / scale
        h = float(row[3]) / scale
        cx = (float(row[0]) - px) / scale
        cy = (float(row[1]) - py) / scale
        x = cx - w * 0.5
        y = cy - h * 0.5
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

# 靶心平滑（缺失时 _bull_kf=None，脚本行为与原来一致）
_bull_kf = None   # 用户要求：靶心不再做卡尔曼平滑


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
    if perf is not None:
        fb = CFG.get("perf_fb", 3)
        fps = CFG.get("perf_fps", 0) or None
        _sensor = perf.open_sensor(CFG["sensor_id"], CFG["w"], CFG["h"],
                                   framebuffers=fb, fps=fps)
    else:
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


# ---- JPEG 回传：质量/缩放统一走 k230_perf（实测编码是瓶颈，见该文件顶部表）----
PERF_ENABLE = perf is not None


def _jq():
    return int(CFG.get("jpeg_q", 92))


def _escale():
    return CFG.get("enc_scale", None)


def _encode_roi(img, roi, q):
    """编码 ROI；roi_scale<1 时在编码器内缩放（信息量 = 回传像素数）。"""
    sc = float(CFG.get("roi_scale", 1.0) or 1.0)
    if PERF_ENABLE:
        if sc < 0.999:
            # 连同缩放一起交给编码器：一次压缩，不额外复制
            w = max(2, int(round(int(roi[2]) * sc / 2.0)) * 2)
            h = max(2, int(round(int(roi[3]) * sc / 2.0)) * 2)
            j = img.compress(quality=q, roi=(int(roi[0]), int(roi[1]),
                                            int(roi[2]), int(roi[3])),
                             x_size=w, y_size=h)
            return (j if isinstance(j, bytes) else bytes(j)), [w, h]
        j, sz, _ = perf.encode(img, roi=roi, quality=q)
        return j, sz
    j = img.compress(quality=q, roi=(roi[0], roi[1], roi[2], roi[3]))
    return (j if isinstance(j, bytes) else bytes(j)), [int(roi[2]), int(roi[3])]


def _encode_full(img, q):
    """编码整幅：按 enc_scale 缩放（默认 0.80 -> 512x384）。"""
    if PERF_ENABLE:
        return perf.encode(img, scale=_escale(), quality=q)
    j = img.compress(quality=q, x_scale=0.5, y_scale=0.5)
    return (j if isinstance(j, bytes) else bytes(j)), [CFG["w"] // 2, CFG["h"] // 2], \
           [0, 0, CFG["w"], CFG["h"]]


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


def _refine(img, box, mult=3.5):
    """对候选框裁剪放大到 640x480 再跑一次，返回 (score, box, kp, status)。

    mult：裁剪边长 = 框长边 * mult。远处 30px 的小靶用 7 倍：
    裁 260x195 -> 放大 2.4 倍喂 320 输入，等效分辨率翻倍。
    """
    bx, by, bw, bh = [float(v) for v in box[:4]]
    side = max(bw, bh) * float(mult)
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
    global _yolo, _lref_skip
    if perf is not None:
        perf.apply_cpu(CFG.get("perf_mode"))
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
    _pstat = perf.status() if perf is not None else \
        "perf=OFF(需要把 board/k230_perf.py 推到 /sdcard，并重跑 live)"
    log("perf 模块: %s" % _pstat)
    emit(T_HELLO, 0, ("pose w=%d h=%d q=%d model=%s mem=%d sensor=%s version=%s %s"
                      % (CFG["w"], CFG["h"], _jq(), CFG["model"],
                         gc.mem_free(), stype, sys.version, _pstat)).encode())
    log("pose streaming %dx%d q=%d model=%s" %
        (CFG["w"], CFG["h"], _jq(), CFG["model"]))
    try:
        gc.threshold(96 * 1024)
    except Exception:
        pass

    seq = 0
    last_res = None
    last_bull = {"status": "no_target"}
    last_det = []
    last_kp_flat = []
    last_has_det = False
    win_t = time.ticks_ms()
    win_n = 0
    every = CFG["stats"]
    target_fps = CFG["fps"]
    prev_loop_ms = 0
    uart_ms = 0
    post_ms = 0

    while True:
        t0 = time.ticks_ms()
        seq += 1
        if _lref_skip > 0:
            _lref_skip -= 1

        # ---- 先拍显示帧，再拍 AI 帧（避免 ROI/检测框与画面错位）
        t_s1 = time.ticks_ms()
        img = sen.snapshot(chn=CAM_CHN_ID_1)
        s1_ms = time.ticks_diff(time.ticks_ms(), t_s1)

        ai_every = int(CFG.get("ai_every", 1) or 1)
        ai_run = (last_res is None) or ai_every <= 1 or (seq % ai_every) == 0
        if ai_run:
            t_s2 = time.ticks_ms()
            arr = sen.snapshot(chn=CAM_CHN_ID_2).to_numpy_ref()
            s2_ms = time.ticks_diff(time.ticks_ms(), t_s2)
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

                # ---- 低置信兜底复筛：小靶/逆光/轻微运动模糊时首轮分数会掉到
                # 0.25 以下（实测 26x38 靶纸有 4/20 帧 <0.25）。此时对分数最高的
                # 2 个候选各做一次 7 倍裁剪放大复筛，分数能回到 0.5~0.8；
                # 负样本实测误检率不变（80 张：1/80 -> 1/80）。
                if (CFG.get("low_refine", 1) and _lref_skip <= 0
                        and (best is None or best[0] < CFG["conf"])):
                    t_l = time.ticks_ms()
                    rb2 = None
                    for i in range(min(len(boxes), int(CFG.get("low_refine_topk", 2)))):
                        # 只复筛小目标（远处靶纸 ≤120px）；大框多半是人/纸箱，
                        # 放大没意义，还会在小堆内存板上多分配两张 640x480 图。
                        if max(float(boxes[i][2]), float(boxes[i][3])) > float(CFG.get("low_refine_max", 120)):
                            continue
                        r = _refine(img, boxes[i], float(CFG.get("low_refine_mult", 7.0)))
                        if r is not None and r[3] == "ok":
                            if rb2 is None or r[0] > rb2[0]:
                                rb2 = r
                    refine_ms += time.ticks_diff(time.ticks_ms(), t_l)
                    if rb2 is not None and rb2[0] >= CFG["conf"]:
                        _lref_skip = 0
                        if best is None or rb2[0] > best[0]:
                            best = rb2
                    else:
                        # 没救回来（多半是负样本/空画面）：冷却几帧，避免每帧
                        # 两次复筛把 13fps 拖到 6fps。
                        _lref_skip = int(CFG.get("low_refine_cooldown", 3))

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

            # ---- 靶心卡尔曼：有测量 update；短暂丢失 predict 兜底 ----
            if _bull_kf is not None:
                _now = time.ticks_ms()
                if bull.get("status") == "ok" and bull.get("x") is not None:
                    _s = _bull_kf.update(float(bull["x"]), float(bull["y"]), _now)
                    if _s is not None:
                        bull["x"] = round(_s[0], 1)
                        bull["y"] = round(_s[1], 1)
                        bull["vx"] = round(_s[2], 1)
                        bull["vy"] = round(_s[3], 1)
                else:
                    _s = _bull_kf.predict(_now)
                    if _s is not None:
                        bull = {"x": round(_s[0], 1), "y": round(_s[1], 1),
                                "conf": round(float(last_bull.get("conf", 0.0) or 0.0), 3),
                                "status": "ok", "src": "kf_coast",
                                "vx": round(_s[2], 1), "vy": round(_s[3], 1)}

        else:
            res = last_res
            infer_ms = 0
            s2_ms = 0
            bull = last_bull
            det = last_det
            kp_flat = last_kp_flat
            has_det = last_has_det
            refine_ms = 0
        last_res = res
        last_bull = bull
        last_det = det
        last_kp_flat = kp_flat
        last_has_det = has_det
        # ---- 板端绘制自检（AGENTS 要求）：每 dbg_save 帧把画好的整帧存到
        # /sdcard/pose_dbg.jpg，PC 端 `k230tool pull` 拉回检查框/关键点。
        if CFG.get("dbg_save", 0) and (seq % int(CFG["dbg_save"]) == 0):
            try:
                if det and len(kp_flat) >= 15:
                    d = det[0]
                    img.draw_rectangle(d[0], d[1], d[2], d[3],
                                       color=(0, 255, 0), thickness=2)
                    for _k in range(5):
                        _x = int(kp_flat[_k * 3])
                        _y = int(kp_flat[_k * 3 + 1])
                        img.draw_circle(_x, _y, 6 if _k == 4 else 4,
                                        color=(255, 0, 0) if _k == 4 else (0, 255, 0),
                                        thickness=2)
                img.save("/sdcard/pose_dbg.jpg")
            except Exception as e:
                log("dbg save: %s" % e)

        # ---- 图像回传：有目标发检测 ROI（原生分辨率），
        # 没目标 / full_every 到点发缩小全图（让 PC 看到完整上下文）。
        want_full = (seq % CFG["full_every"] == 0) or (not has_det)
        want_jpeg = (seq % CFG["jpeg_every"] == 0) or want_full
        jpg = None
        roi = None
        jsz = None
        jpg_ms = 0
        if want_jpeg:
            t_j = time.ticks_ms()
            if want_full:
                # 全图优先（full_every 到点 / 没目标）：保证 PC 能看到完整上下文
                jpg, jsz, roi = _encode_full(img, _jq())
            elif has_det:
                # 用经过复筛/校验后的 det[0]（boxes[0] 可能是低分候选/负样本框）
                roi = _roi_around(det[0] if det else boxes[0])
                jpg, jsz = _encode_roi(img, roi, _jq())
            else:
                jpg, jsz, roi = _encode_full(img, _jq())
            jpg_ms = time.ticks_diff(time.ticks_ms(), t_j)

        out = {"seq": seq, "w": CFG["w"], "h": CFG["h"],
               "ai": 1 if ai_run else 0,
               "ms": infer_ms, "ref_ms": refine_ms,
               "s1": s1_ms, "s2": s2_ms, "jpg": jpg_ms,
               "loop": prev_loop_ms, "uart": uart_ms, "post": post_ms,
               "det": det, "kp": kp_flat,
               "labels": CFG["labels"], "task": "pose",
               "roi": roi, "jpeg": jsz, "bull": bull}
        emit(T_TARGET, seq, ujson.dumps(out).encode())
        if ai_run:
            t_u = time.ticks_ms()
            _send_uart(seq, det, bull, has_det)
            uart_ms = time.ticks_diff(time.ticks_ms(), t_u)
        else:
            uart_ms = 0
        if jpg is not None:
            emit(T_FRAME, seq, jpg)

        win_n += 1
        if every and (seq % every == 0):
            now = time.ticks_ms()
            dt = time.ticks_diff(now, win_t)
            fps = 0.0 if dt <= 0 else (win_n * 1000.0 / dt)
            emit(T_STATS, seq, ("seq=%d fps=%.1f infer_ms=%d jpg_ms=%d s1=%d s2=%d bytes=%d mem=%d bull=%s"
                                % (seq, fps, infer_ms, jpg_ms, s1_ms, s2_ms,
                                   len(jpg) if jpg else 0, gc.mem_free(),
                                   bull.get("status"))).encode())
            win_t = now
            win_n = 0

        prev_loop_ms = time.ticks_diff(time.ticks_ms(), t0)
        post_ms = prev_loop_ms - s1_ms - s2_ms - infer_ms - jpg_ms - refine_ms - uart_ms
        if post_ms < 0:
            post_ms = 0

        if CFG["limit"] and seq >= CFG["limit"]:
            log("limit %d reached" % CFG["limit"])
            break
        if (seq & 0x0F) == 0:
            # 每 16 帧回收一次：jpeg_every=2 后临时 bytes 减半，8 帧一次太勤
            # （每次 gc 在板端约 1~2ms）。
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
