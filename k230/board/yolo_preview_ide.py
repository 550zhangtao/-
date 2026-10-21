# -*- coding: utf-8 -*-
"""
yolo_preview_ide.py —— 在 CanMV IDE 里看 YOLOv8n-Pose 标靶 + 激光光斑
=====================================================================

画面内容：
    * 检测框（只作为特征区域，不要求四角精确）
    * 4 个角点连成的靶纸轮廓
    * 第 5 个关键点：白色 10 号靶心（红十字）
    * **激光光斑**：青色十字 + 圆环 + 坐标标注（`board/laser_spot.py`）
    * 靶心与光斑之间的黄色连线（有靶心时）

配置：
    MODEL      训练好的 pose kmodel（5 关键点）
    LASER_*    激光检测开关与参数（见 board/laser_spot.py）
    VIEW       "full"=固定全图 640×480（默认，画面稳定不跳动）；"center"=固定中心裁剪；"roi"=旧动态ROI

说明：
    * 不依赖 libs/PipeLine（板端 libs 可能比固件新）；
    * IDE 预览用 img.compress_for_ide()，不初始化 Display；
    * 退出自动复位，避免 KPU 析构死锁；
    * 激光检测在 chn1 的 RGB565 帧上做（坐标即原图坐标，不用反变换），
      每 LASER_EVERY 帧跑一次。
"""

# ============================ 用户配置 ============================
MODEL = "/sdcard/models/pose_v3.kmodel"   # 若 IDE 找不到，改成 /models/pose_v3.kmodel
LABELS = ["target"]
MODEL_IN = [320, 320]
CAPTURE_SIZE = [640, 480]
DISPLAY_MODE = "ide"          # ide=CanMV IDE；hdmi/lcd=本地屏
VIEW = "full"                 # full=全图固定缩放；center=固定中心裁剪；roi=旧动态ROI
VIEW_SCALE = 0.55            # full 缩放：0.55 -> 352x264（追求 15~20fps）；1.0 -> 640x480
IDE_QUALITY = 80              # IDE 预览 JPEG 质量（85 平衡清晰度/帧率；95 更清晰但更慢）
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

# ---- 激光光斑 ----
LASER_ENABLE = 1              # 1=检测并绘制激光光斑
# 固定标定下每 N 帧才复核一次，其余帧零搜索。
# 每帧都调 locate()：非复核帧只返回缓存，开销≈0；这样复核/重捕节奏与代理一致。
LASER_EVERY = 1
LASER_VERIFY_EVERY = 5        # 每 5 帧复核一次（固定标定基本不变）
LASER_GLOBAL = 1              # 与代理/live 一致：允许现有受限分格重捕（不是全图）
LASER_BUDGET_MS = 35          # 单帧激光超过此值才跳；只跳 1 帧，避免人为卡顿
LASER_COOLDOWN = 1
LASER_CFG = {}                # 覆盖 board/laser_spot.py 的 CFG，如 {"min_contrast": 45}
LASER_LABEL = 0               # 1=在光斑旁画坐标文字（会让每帧多一条 deprecation 警告，略拖慢）
# 固定标定：相机与激光笔固连 → 光斑在画面里是常量，标定一次就不用每帧找。
# 优先用这个值；为 None 时读 LASER_CALIB_FILE；文件也没有就自动标定并写回。
LASER_FIXED = None            # 例如 (375, 275)
LASER_CALIB_FILE = "/sdcard/laser_calib.json"
LASER_DBG_SAVE = 120          # 每 N 帧把"画好叠加"的帧存 /sdcard/laser_dbg.jpg（0=关）

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

# 激光检测模块：固定从 /sdcard 加载（版本统一；避免旧副本抢先）。
for _p in ("/sdcard", "/sdcard/board", "/"):
    try:
        if _p in sys.path:
            sys.path.remove(_p)
        sys.path.insert(0, _p)
    except Exception:
        pass
try:
    import laser_spot
except Exception as _e:
    laser_spot = None
    print("laser_spot import failed:", _e)

W, H = CAPTURE_SIZE
_sensor = None
_yolo = None
_laser = None
_laser_skip = 0
_laser_calib_warned = False
_laser_recalib_warned = False


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


def _send_uart_laser(seq, laser):
    """`$LS,<seq>,<run>,<x*10>,<y*10>,<conf*1000>,<src>!`（三态接口，TODO 控制）"""
    if _uart is None:
        return
    try:
        if laser is not None and float(laser.get("x", -1)) >= 0:
            # F：src=1 只给本帧实测；fixed/calib/confirm/reacquire 都是 2
            src = 1 if laser.get("src") == "measured" else 2
            _uart.write("$LS,%d,1,%d,%d,%d,%d!\n" % (
                seq, int(laser["x"] * 10), int(laser["y"] * 10),
                int(float(laser.get("conf", 0)) * 1000), src))
        else:
            _uart.write("$LS,%d,0,-1,-1,0,0!\n" % seq)
        _uart.flush()
    except Exception as e:
        print("uart ls:", e)


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


def _draw_laser(img, laser, s=1.0, bull=None):
    """画激光光斑：青色十字 + 圆环 + 坐标文字；有靶心时画偏差连线。"""
    if laser is None:
        return
    lx = int(round(laser["x"] * s))
    ly = int(round(laser["y"] * s))
    cross = max(8, int(14 * s))
    th = max(1, int(2 * s))
    # 画框用**蓝色**：原画面里的红光条/红色物体不会被误认为"光斑"，
    # 也更容易和青色/黄色的其他叠加区分
    try:
        img.draw_cross(lx, ly, color=(0, 120, 255), size=cross,
                       thickness=th)
    except Exception:
        img.draw_line(lx - cross, ly, lx + cross, ly, color=(0, 120, 255),
                      thickness=th)
        img.draw_line(lx, ly - cross, lx, ly + cross, color=(0, 120, 255),
                      thickness=th)
    img.draw_circle(lx, ly, max(6, int(18 * s)), color=(0, 120, 255),
                    thickness=max(1, int(1 * s)))
    if bull is not None:
        bx = int(round(bull[0] * s))
        by = int(round(bull[1] * s))
        img.draw_line(bx, by, lx, ly, color=(255, 255, 0),
                      thickness=max(1, int(1 * s)))
    # 坐标文字默认关闭：本机固件只有 draw_string 可用（draw_string_advanced
    # 会报 "FreeType init failed, font (null)"），而 draw_string 每次调用都会
    # 打印一条 "Deprecated function" 警告，走 USB 到 IDE 很慢，会拖低帧率。
    # 需要看数字时把 LASER_LABEL 改 1。
    if LASER_LABEL:
        try:
            img.draw_string(lx + cross + 2, ly - cross - 12,
                            "(%.0f,%.0f)" % (laser["x"], laser["y"]),
                            color=(0, 120, 255), scale=max(1, int(2 * s)))
        except Exception:
            pass


def _load_calib():
    """读标定文件，返回 (x, y) 或 None。"""
    try:
        f = open(LASER_CALIB_FILE)
        s = f.read()
        f.close()
        i = s.find('"x"')
        j = s.find('"y"')
        if i >= 0 and j >= 0:
            x = float(s[s.find(":", i) + 1:s.find(",", i)])
            y = float(s[s.find(":", j) + 1:s.find("}", j)])
            return (x, y)
    except Exception:
        pass
    return None


def _save_calib(x, y):
    try:
        f = open(LASER_CALIB_FILE, "w")
        f.write('{"x":%.1f,"y":%.1f}' % (x, y))
        f.close()
        print("laser calib saved: (%.1f, %.1f) -> %s" % (x, y, LASER_CALIB_FILE))
    except Exception as e:
        print("laser calib save failed:", e)


def _init_laser():
    global _laser
    if not LASER_ENABLE:
        return None
    if laser_spot is None:
        print("激光检测不可用：laser_spot.py 未找到")
        return None
    ver = getattr(laser_spot, "VERSION", "?")
    api = int(getattr(laser_spot, "API_VERSION", 0))
    ch = int(getattr(laser_spot, "CFG_HASH", 0))
    md5 = ""
    try:
        md5 = laser_spot.file_md5("/sdcard/laser_spot.py")
    except Exception:
        pass
    print("laser: laser_spot v%s api=%d cfg=0x%08x md5=%s" %
          (ver, api, ch, md5 or "?"))
    if api < 2:
        print("laser: laser_spot 版本过旧（需要 API>=2），已关闭；"
              "先推 /sdcard/laser_spot.py")
        return None
    cfg = {
        "fixed_verify_every": LASER_VERIFY_EVERY,
        "fixed_allow_reacquire": LASER_GLOBAL,
    }
    if LASER_CFG:
        cfg.update(LASER_CFG)
    _laser = laser_spot.LaserSpot(cfg)
    fixed = LASER_FIXED
    src = "config"
    if fixed is None:
        fixed = _load_calib()
        src = "file"
    if (fixed is not None and 0 < fixed[0] < W and 0 < fixed[1] < H):
        _laser.set_fixed(fixed[0], fixed[1])
        print("laser: 固定标定 (%.1f, %.1f) 来自%s → 每 %d 帧复核一次，其余帧零搜索" %
              (fixed[0], fixed[1], src, LASER_VERIFY_EVERY))
    else:
        print("laser: 未标定 → 上电后自动做一次视觉标定（成功即锁定并写回 %s），"
              "之后每帧只在小窗口跟踪" % LASER_CALIB_FILE)
    return _laser


def main():
    global _sensor, _yolo, _laser_skip, _laser_calib_warned, _laser_recalib_warned
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
    _init_laser()

    n = 0
    t_last = time.ticks_ms()
    fps = 0.0
    last_res = None
    laser = None
    laser_ms = 0
    while True:
        # 先拍显示帧，再拍 AI 帧（仅在 AI 帧执行）
        img = _sensor.snapshot(chn=CAM_CHN_ID_1)
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

        # 激光光斑：直接在 chn1 RGB565 原图上找（坐标即原图坐标）
        if _laser is not None and (n % LASER_EVERY) == 0:
            if _laser_skip > 0:
                _laser_skip -= 1
                # 看门狗跳帧：保留坐标，用 cached() 标成回退（src=2）
                try:
                    laser = _laser.cached()
                except Exception:
                    laser = None
            else:
                t_l = time.ticks_ms()
                try:
                    laser = _laser.locate(img, anchor=_bull_of(res))
                except Exception as e:
                    print("laser:", e)
                    laser = None
                laser_ms = time.ticks_diff(time.ticks_ms(), t_l)
                if laser_ms > LASER_BUDGET_MS:
                    _laser_skip = LASER_COOLDOWN
                    print("laser %dms > %dms, skip %d"
                          % (laser_ms, LASER_BUDGET_MS, LASER_COOLDOWN))
                if (getattr(_laser, "need_recalib", False)
                        and not _laser_recalib_warned):
                    _laser_recalib_warned = True
                    print("laser: NEED_RECALIB (local verify failed; "
                          "check laser/calib, no full-screen search)")
                elif laser is not None and laser.get("src") == "measured":
                    _laser_recalib_warned = False
                # calib 偏差告警：只打印一次，不写文件
                if (not _laser_calib_warned
                        and getattr(_laser, "calib_stale", False)):
                    _laser_calib_warned = True
                    print("laser: CALIB_STALE dev=%.0fpx (file not modified)"
                          % getattr(_laser, "_calib_dev", 0.0))
                # 上电首次视觉标定成功后存盘，之后走窄窗跟踪
                if laser is not None and _laser.just_calibrated:
                    _laser.just_calibrated = False
                    _save_calib(laser["x"], laser["y"])
                    print("laser: 首次视觉标定完成 (%.1f, %.1f)，之后窄窗跟踪"
                          % (laser["x"], laser["y"]))
        # 三态接口：没有实拍光斑时用固定标定位置上报 $LS（src=2）。
        # 这里只上报/绘制，不做云台控制（控制算法后续再接）。
        if laser is None and _laser is not None:
            _fx = getattr(_laser, "fixed", None)
            if _fx is not None and _fx[0] > 0:
                laser = {"x": _fx[0], "y": _fx[1], "conf": 0,
                         "mode": "calib", "src": "fixed"}
        _send_uart_laser(n, laser)
        bull = _bull_of(res)

        if DISPLAY_MODE == "ide":
            if VIEW == "full":
                # 先缩放到固定尺寸，再画覆盖层：绘制成本降数倍，帧率更稳
                disp = img.copy(x_scale=VIEW_SCALE, y_scale=VIEW_SCALE)
                _draw(disp, res, VIEW_SCALE)
                _draw_laser(disp, laser, VIEW_SCALE, bull)
                disp.compress_for_ide(quality=IDE_QUALITY)
            else:
                _draw(img, res)
                _draw_laser(img, laser, 1.0, bull)
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
        if LASER_DBG_SAVE and n and (n % LASER_DBG_SAVE) == 0:
            try:
                # VIEW=full 时叠加画在 disp 上；保存 img 会得到一张没有框的
                # 原图，起不到核对绘制路径的作用。
                if VIEW == "full" and DISPLAY_MODE == "ide":
                    disp.save("/sdcard/laser_dbg.jpg")
                else:
                    img.save("/sdcard/laser_dbg.jpg")
            except Exception as e:
                print("save dbg:", e)

        n += 1
        dt = time.ticks_diff(time.ticks_ms(), t_last)
        t_last = time.ticks_ms()
        if dt > 0:
            fps = 0.8 * fps + 0.2 * (1000.0 / dt) if fps else 1000.0 / dt
        if PRINT_EVERY and (n % PRINT_EVERY == 0):
            ls = "laser -"
            if laser is not None:
                ls = "laser (%.1f,%.1f) conf %.2f %s" % (
                    laser["x"], laser["y"], laser["conf"], laser["mode"])
                if bull is not None:
                    ls += " dx %.0f dy %.0f" % (laser["x"] - bull[0],
                                                laser["y"] - bull[1])
            tg = "target -"
            if res and res[0]:
                tg = "target %.2f" % float(res[2][0])
            nc = 0 if _laser is None else _laser.last_cands
            print("[%d] %.1ffps %s | %s | lms %d cand %d" % (
                n, fps, tg, ls, laser_ms, nc))
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
