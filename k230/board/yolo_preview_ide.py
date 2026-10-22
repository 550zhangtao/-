# -*- coding: utf-8 -*-
"""
yolo_preview_ide.py —— 在 CanMV IDE 里看 YOLOv8n-Pose 标靶
==========================================================

画面内容：
    * 检测框（只作为特征区域，不要求四角精确）
    * 4 个角点连成的靶纸轮廓
    * 第 5 个关键点：白色 10 号靶心（红十字）

配置：
    MODEL      训练好的 pose kmodel（5 关键点）
    VIEW       "full"=固定全图 640×480（默认，画面稳定不跳动）；"center"=固定中心裁剪；"roi"=旧动态ROI
    DBG_SAVE   每 N 帧把"画好的帧"存到 /sdcard/pose_preview_dbg.jpg（0=关）

说明：
    * 不依赖 libs/PipeLine（板端 libs 可能比固件新）；
    * IDE 预览用 img.compress_for_ide()，不初始化 Display；
    * 退出自动复位，避免 KPU 析构死锁。
"""

# ============================ 用户配置 ============================
MODEL = "/sdcard/models/pose_v3_384.kmodel"   # 384 输入（当前唯一模型，远小靶更准）
LABELS = ["target"]
MODEL_IN = [384, 384]
CAPTURE_SIZE = [640, 480]
DISPLAY_MODE = "ide"          # ide=CanMV IDE；hdmi/lcd=本地屏
VIEW = "full"                 # full=全图固定缩放；center=固定中心裁剪；roi=旧动态ROI
VIEW_SCALE = 0.45            # full 缩放：0.45 -> 288x216（帧率优先）；0.55 -> 352x264
IDE_QUALITY = 80              # IDE 预览 JPEG 质量（70 帧率优先；85~90 更清晰）
CENTER_SIZE = 320             # center 模式的固定裁剪边长
CONF = 0.25
# 低置信兜底复筛：远处 30px 小靶/逆光时首轮分数会掉到 0.25 以下。
# 对分数最高的 2 个候选各做一次 7 倍裁剪放大复筛（实测 0.02~0.20 -> 0.5~0.8，
# 负样本误检率不变），只对候选各多跑一次 320 推理。
LOW_REFINE = 1
LOW_REFINE_TOPK = 2
LOW_REFINE_MULT = 7.0
NMS = 0.45
MAX_BOXES = 20
KP_NUM = 5                    # 4 角点 + 靶心
KP_DIM = 3                    # x,y,v
VIEW_PAD = 0.25               # ROI 外扩比例
AI_EVERY = 2                 # 每 N 帧跑一次 pose（其余帧复用上次结果）

# ---- 板端性能（见 board/k230_perf.py；实测采集 57fps，编码/绘图才是瓶颈）----
PERF_MODE = "performance"     # auto / balanced / performance
SENSOR_FPS = 60               # 采集帧率（GC2093 原生 1080p60，降下来不会更快）
SENSOR_FB = 3                 # 固件不支持时自动跳过
ENC_SCALE = 0.80              # 预览图缩放：0.80 -> 512x384 更清晰，还比 640x480 快

# ---- 激光点：极简 ROI + cv_lite（见 board/laser_openmv.py）----
# 只用一路灰度通道 chn0 的「画面中心 120x120」窗口，cv_lite 灰度阈值找最大 blob。
# 没有先验/评分/多帧确认。实测 ROI 内 cv_lite 0.2ms。
LASER_ENABLE = 1
LASER_CFG = {}          # 覆盖 laser_openmv.CFG，例如 {"diag_mode": 1}
LASER_EVERY = 1         # 每 N 帧检测一次（其余帧复用上次结果）
LASER_LABEL = 1         # 1=画坐标文字（用内置 3x5 点阵，不调用 deprecated API）
LASER_LABEL_EVERY = 1   # 每 N 帧重画一次文字
LASER_CROSS_COLOR = (0, 0, 0)   # 黑色压在亮光斑上最清楚
LASER_LABEL_COLOR = (255, 255, 255)

# ---- 板端绘制自检（AGENTS 要求）----
# 每 N 帧把"叠加画好"的帧存到 /sdcard，PC 端 `k230tool pull` 拉回核对框/关键点。
DBG_SAVE = 0                  # 0=关；调试时可改 120

# K230 -> ESP32 云台 UART（UART3: TX=GPIO32, RX=GPIO33）
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

try:
    import k230_perf as perf
except Exception as _e:
    perf = None
    print("k230_perf import failed:", _e)

try:
    import cv_lite
except Exception as _e:
    cv_lite = None
    print("cv_lite 不可用:", _e)

try:
    import laser_openmv as laser_spot_mod
except Exception as _e:
    laser_spot_mod = None
    print("laser_openmv 不可用，激光检测关闭:", _e)

# 固定从 /sdcard 加载（版本统一；避免旧副本抢先）。
for _p in ("/sdcard", "/sdcard/board", "/"):
    try:
        if _p in sys.path:
            sys.path.remove(_p)
        sys.path.insert(0, _p)
    except Exception:
        pass
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
    # AI2D 左上角对齐：pad 恒为 0
    # YOLO 输出 [cx,cy,w,h] 是**中心点**，旧代码当左上角用，框整体右下偏
    # w/2,h/2（26x36 的靶纸偏 11~18px），ROI/绘制/复筛全跟着偏。
    w = float(row[2]) / scale
    h = float(row[3]) / scale
    cx = float(row[0]) / scale
    cy = float(row[1]) / scale
    x = cx - w * 0.5
    y = cy - h * 0.5
    kp = []
    for k in range(kp_num):
        kp.append(float(row[5 + k * kp_dim]) / scale)
        kp.append(float(row[5 + k * kp_dim + 1]) / scale)
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


def _bull_of(res):
    """取靶心 (x, y) 或 None。"""
    if not res or not res[0] or not res[3]:
        return None
    kp = res[3][0]
    if len(kp) >= 15 and float(kp[14]) >= 0.3:
        return (float(kp[12]), float(kp[13]))
    return None


def _refine_pose(img, box, mult=7.0):
    """裁剪候选框并放大到 640x480 再跑一次，返回 (box, kp, score) 或 None。

    `box` 是 [x,y,w,h]（原图坐标）。放大倍数 mult 越大，远处小靶在 320
    输入里占的像素越多；7 倍对 26x38 的靶纸效果最好（等效分辨率翻倍）。
    """
    bx, by, bw, bh = [float(v) for v in box[:4]]
    side = max(max(bw, bh) * float(mult), 64.0)
    cw = int(side)
    ch = int(side * 0.75)
    cx = bx + bw * 0.5
    cy = by + bh * 0.5
    x = int(max(0.0, min(W - cw, cx - cw * 0.5)))
    y = int(max(0.0, min(H - ch, cy - ch * 0.5)))
    cw = int(min(cw, W - x))
    ch = int(min(ch, H - y))
    if cw < 16 or ch < 16:
        return None
    try:
        crop = img.crop(roi=(x, y, cw, ch))
        zoom = crop.copy(x_scale=W / float(cw), y_scale=H / float(ch))
        arr = zoom.to_rgb888().to_numpy_ref()
        res = _yolo.run(arr)
        if not res or not res[0]:
            return None
        b = res[0][0]
        kp = res[3][0]
        score = float(res[2][0])
        sx = cw / float(W)
        sy = ch / float(H)
        nb = [x + float(b[0]) * sx, y + float(b[1]) * sy,
              float(b[2]) * sx, float(b[3]) * sy]
        nkp = []
        for i in range(0, len(kp), KP_DIM):
            nkp.append(x + float(kp[i]) * sx)
            nkp.append(y + float(kp[i + 1]) * sy)
            nkp.append(float(kp[i + 2]))
        return (nb, nkp, score)
    except Exception as e:
        print("refine:", e)
        return None


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


# ---------------------------------------------------------------- 激光点
_laser = None            # LaserSpot 实例
_laser_ms = 0
_laser_spot = None       # 最近一次结果 {"cx","cy","n","x","y","w","h"}
_laser_lock = None       # 锁定的 (cx, cy)
_laser_miss = 0
_laser_n = 0


def _init_laser():
    """建 LaserSpot（极简版：ROI + cv_lite 灰度阈值，无先验/评分）。"""
    global _laser
    _laser = None
    if not LASER_ENABLE:
        return
    if laser_spot_mod is None:
        print("激光: laser_openmv.py 缺失，跳过")
        return
    try:
        _laser = laser_spot_mod.LaserSpot(LASER_CFG or None)
        print("激光: ROI=%s THR=%s lms<10ms 版本 %s"
              % (_laser.roi, _laser.threshold,
                 getattr(laser_spot_mod, "VERSION", "?")))
    except Exception as e:
        _laser = None
        print("激光初始化失败:", e)


# 3x5 点阵字体（只用 draw_rectangle 画，避免 draw_string 的 Deprecated 警告）
_FONT = {
    "0": (0b111, 0b101, 0b101, 0b101, 0b111),
    "1": (0b010, 0b110, 0b010, 0b010, 0b111),
    "2": (0b111, 0b001, 0b111, 0b100, 0b111),
    "3": (0b111, 0b001, 0b111, 0b001, 0b111),
    "4": (0b101, 0b101, 0b111, 0b001, 0b001),
    "5": (0b111, 0b100, 0b111, 0b001, 0b111),
    "6": (0b111, 0b100, 0b111, 0b101, 0b111),
    "7": (0b111, 0b001, 0b010, 0b010, 0b010),
    "8": (0b111, 0b101, 0b111, 0b101, 0b111),
    "9": (0b111, 0b101, 0b111, 0b001, 0b111),
    "(": (0b011, 0b100, 0b100, 0b100, 0b011),
    ")": (0b110, 0b001, 0b001, 0b001, 0b110),
    ",": (0b000, 0b000, 0b000, 0b010, 0b100),
    "-": (0b000, 0b000, 0b111, 0b000, 0b000),
    " ": (0, 0, 0, 0, 0),
}


def _text_width(txt, sc):
    return len(txt) * 4 * sc


def _draw_text(dst, x, y, txt, sc=2, color=(255, 255, 255),
               bg=(0, 0, 0)):
    """3x5 点阵写字：先铺黑底，再逐像素画方块。不调用任何 deprecated API。"""
    w = _text_width(txt, sc)
    h = 5 * sc
    if bg is not None:
        dst.draw_rectangle(x - sc, y - sc, w + 2 * sc, h + 2 * sc,
                           color=bg, thickness=1, fill=True)
    cur = x
    for ch in txt:
        glyph = _FONT.get(ch)
        if glyph is not None:
            for row in range(5):
                bits = glyph[row]
                for col in range(3):
                    if bits & (4 >> col):
                        dst.draw_rectangle(cur + col * sc, y + row * sc,
                                           sc, sc, color=color, thickness=1,
                                           fill=True)
        cur += 4 * sc
    return w, h


def _draw_laser(dst, spot, sc=1.0, label=True):
    """十字 + 圆环 + 坐标文字。坐标是 640x480 原图真实值。"""
    if spot is None:
        return
    cx = int(round(spot["cx"] * sc))
    cy = int(round(spot["cy"] * sc))
    dst.draw_cross(cx, cy, color=LASER_CROSS_COLOR,
                   size=max(8, int(14 * sc)), thickness=2)
    dst.draw_circle(cx, cy, max(6, int(12 * sc)), color=LASER_CROSS_COLOR,
                    thickness=2)
    if not (label and LASER_LABEL):
        return
    txt = "(%d,%d)" % (int(spot["cx"]), int(spot["cy"]))
    tsc = 2 if sc > 0.5 else 1
    tx = cx + int(18 * sc)
    ty = cy - 5 * tsc - int(6 * sc)
    if tx + _text_width(txt, tsc) > dst.width() - 2:
        tx = max(2, dst.width() - _text_width(txt, tsc) - 2)
    if ty < 2:
        ty = min(dst.height() - 5 * tsc - 2, cy + int(18 * sc))
    _draw_text(dst, tx, ty, txt, tsc, LASER_LABEL_COLOR)


def main():
    global _sensor, _yolo
    if perf is not None:
        perf.apply_cpu(PERF_MODE)
        perf.ENC_SCALE = ENC_SCALE
        # gray=True：额外开 chn0 灰度通道专供 cv_lite（实测三通道采集仍 ~57fps）
        _sensor = perf.open_sensor(2, W, H, framebuffers=SENSOR_FB, fps=SENSOR_FPS,
                                   gray=(LASER_ENABLE and laser_spot_mod is not None))
    else:
        _sensor = Sensor(id=2)
        _sensor.reset()
        _sensor.set_framesize(w=W, h=H, chn=CAM_CHN_ID_1)
        _sensor.set_pixformat(Sensor.RGB565, chn=CAM_CHN_ID_1)
        _sensor.set_framesize(w=W, h=H, chn=CAM_CHN_ID_2)
        _sensor.set_pixformat(Sensor.RGBP888, chn=CAM_CHN_ID_2)
        if LASER_ENABLE and laser_spot_mod is not None:
            _sensor.set_framesize(w=W, h=H, chn=CAM_CHN_ID_0)
            _sensor.set_pixformat(Sensor.GRAYSCALE, chn=CAM_CHN_ID_0)
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
    _init_laser()

    n = 0
    t_last = time.ticks_ms()
    fps = 0.0
    last_res = None
    while True:
        # 先拍显示帧，再拍 AI 帧（仅在 AI 帧执行）
        img = _sensor.snapshot(chn=CAM_CHN_ID_1)
        # ---- 激光：chn0 灰度 + ROI + cv_lite（每帧都要取帧，否则堵管线）
        global _laser_spot, _laser_ms
        if LASER_ENABLE and _laser is not None:
            gimg = _sensor.snapshot(chn=CAM_CHN_ID_0)
            if (n % LASER_EVERY) == 0:
                try:
                    _ls = _laser.sample(gimg)
                    _laser_ms = int(getattr(_laser, "last_ms", 0))
                    if _ls is not None:
                        _laser_spot = {"cx": _ls["x"], "cy": _ls["y"],
                                       "n": _ls.get("n", 0),
                                       "conf": _ls.get("conf", 0.0)}
                    else:
                        _laser_spot = None
                except Exception as e:
                    print("laser:", e)
                    _laser_spot = None
        if last_res is None or (n % AI_EVERY) == 0:
            arr = _sensor.snapshot(chn=CAM_CHN_ID_2).to_numpy_ref()
            last_res = _yolo.run(arr)
            # ---- 低置信兜底复筛（与 board/yolo_pose.py 的策略一致）
            if LOW_REFINE and last_res and last_res[0]:
                top = max(float(v) for v in last_res[2])
                if top < CONF:
                    rb = None
                    _boxes = last_res[0]
                    for _i in range(min(len(_boxes), LOW_REFINE_TOPK)):
                        r = _refine_pose(img, _boxes[_i], LOW_REFINE_MULT)
                        if r is not None and (rb is None or r[2] > rb[2]):
                            rb = r
                    if rb is not None and rb[2] >= CONF:
                        last_res = ([rb[0]], [0], [rb[2]], [rb[1]])
        res = last_res

        _send_uart(n, res)
        if DISPLAY_MODE == "ide":
            if VIEW == "full":
                # 先缩放到固定尺寸，再画覆盖层：绘制成本降数倍，帧率更稳
                d = perf.enc_dims(W, H, VIEW_SCALE) if perf is not None else None
                sc = (d[0] / float(W)) if d else VIEW_SCALE
                disp = img.copy(x_scale=sc, y_scale=sc)
                _draw(disp, res, sc)
                _draw_laser(disp, _laser_spot, sc,
                            label=(n % LASER_LABEL_EVERY) == 0)
                if d:
                    disp.compress_for_ide(x_size=d[0], y_size=d[1], quality=IDE_QUALITY)
                else:
                    disp.compress_for_ide(quality=IDE_QUALITY)
            else:
                _draw(img, res)
                _draw_laser(img, _laser_spot, 1.0,
                            label=(n % LASER_LABEL_EVERY) == 0)
                roi = None
                if res and res[0]:
                    b = res[0][0]
                    bx, by, bw, bh = [int(v) for v in b[:4]]
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

        # 板端绘制路径自检：把"画好"的帧存到 /sdcard 供 k230tool pull 检查
        if DBG_SAVE and n and (n % DBG_SAVE) == 0:
            try:
                # VIEW=full 时叠加画在 disp 上；保存 img 会得到一张没有框的
                # 原图，起不到核对绘制路径的作用。
                if VIEW == "full" and DISPLAY_MODE == "ide":
                    disp.save("/sdcard/pose_preview_dbg.jpg")
                else:
                    img.save("/sdcard/pose_preview_dbg.jpg")
            except Exception as e:
                print("save dbg:", e)

        n += 1
        dt = time.ticks_diff(time.ticks_ms(), t_last)
        t_last = time.ticks_ms()
        if dt > 0:
            fps = 0.8 * fps + 0.2 * (1000.0 / dt) if fps else 1000.0 / dt
        if PRINT_EVERY and (n % PRINT_EVERY == 0):
            tg = "target -"
            bx = by = -1
            if res and res[0]:
                tg = "target %.2f" % float(res[2][0])
                try:
                    kp = res[3][0]
                    if KP_NUM >= 5 and float(kp[14]) >= 0.3:
                        bx = int(kp[12])
                        by = int(kp[13])
                except Exception:
                    pass
            _nraw = 0 if _laser is None else int(getattr(_laser, "raw_blobs", 0))
            if _laser_spot is not None:
                lz = "laser (%d,%d) conf %.2f" % (
                    int(_laser_spot["cx"]), int(_laser_spot["cy"]),
                    float(_laser_spot.get("conf", 0.0)))
            else:
                lz = "laser -"
            lz += " | lms %dms raw %d" % (_laser_ms, _nraw)
            print("[%d] %.1ffps | %s | %s" % (n, fps, lz, tg))
        if (n & 0x0F) == 0:
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
