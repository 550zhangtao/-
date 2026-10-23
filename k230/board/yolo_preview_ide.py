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
MODEL = "/sdcard/models/pose_v3_256.kmodel"   # 256 输入（更快；精度与 384 相当）
LABELS = ["target"]
MODEL_IN = [256, 256]
CAPTURE_SIZE = [640, 480]
DISPLAY_MODE = "ide"          # ide=CanMV IDE；hdmi/lcd=本地屏
VIEW = "full"                 # full=全图固定缩放；center=固定中心裁剪；roi=旧动态ROI
VIEW_SCALE = 0.60            # 0.60 -> 384x288（Pose/2 实测 17.8-18.1fps）
IDE_QUALITY = 85              # 384x288 q85：编码约 35ms
CENTER_SIZE = 320             # center 模式的固定裁剪边长
CONF = 0.25
BULL_MIN_KP = 0.20           # 靶心关键点置信度门限（2026-10-24 从 0.3 放宽，提高可用率）
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
AI_EVERY = 2                 # 每 2 帧跑 pose（384x288 q85 实测 17.8-18.1fps）

# ---- 板端性能（见 board/k230_perf.py；实测采集 57fps，编码/绘图才是瓶颈）----
PERF_MODE = "performance"     # auto / balanced / performance
SENSOR_FPS = 60               # 采集帧率（GC2093 原生 1080p60，降下来不会更快）
SENSOR_FB = 3                 # 固件不支持时自动跳过
ENC_SCALE = 0.80              # 预览图缩放：0.80 -> 512x384 更清晰，还比 640x480 快

# ---- 激光检测：移植 zhouzhou-code/elec_vision_k230（见 board/laser_repo.py）----
# 原版逻辑：固定居中 ROI(320x240) + LAB 阈值(9,100,15,98,-13,77)
#          + find_blobs(pixels_threshold=1, merge=True) + 取面积最大。
# 2026-10-23 修：原版"直接取最大"会锁红袋子/红电线/手。laser_repo.py 现在
# 支持门限，这里的 LASER_CFG 会被真正传进 LaserRepo（以前是空操作）。
# 门限含义：min_area/max_area=bbox 面积，max_side=单边，max_aspect=长宽比，
#          min_fill=pixels/(w*h)。置 0 即关闭。
LASER_ENABLE = 1
# ---- E 路线：学习式 ROI 热图小模型（优先；失败回退 laser_repo）----
LASER_MODEL = "/sdcard/models/laser_heatmap.kmodel"
LASER_ROI = 128
# ---- ROI/阈值「调节模式」（2026-10-23 优化）----
# 进入用高阈值防误锁；锁定后用低阈值防丢失；远处跳变必须高置信度才算数；
# 丢失时先做便宜的局部螺旋搜索，仍找不到才全域粗扫（且降低扫描频率）。
LASER_THR_ACQ = 0.70          # 未锁定/重捕：进入阈值
LASER_THR_TRK = 0.45          # 已锁定：保持阈值
LASER_LOCK_N = 2              # 连续命中 N 帧视为锁定
LASER_JUMP_PX = 40            # 相邻帧位移门限（像素）
LASER_PRIOR_A = 0.35          # 先验低通系数（只挪 ROI 中心，不改变画出的点）
LASER_LOST_SCAN = 1           # 丢失后搜索重捕
LASER_LOST_MAX = 5            # 连续丢 N 帧后开始搜
LASER_SCAN_EVERY = 6          # 每 N 帧搜一次（省算力，避免掉帧）
LASER_SCAN_THR = 0.45          # 重捕复核阈值（比进入阈值低）
LASER_GRAY_EVERY = 3           # 灰度全帧搜索降频（to_grayscale 每帧分配内存，易拖垮堆）
LASER_PRIOR = (290, 190)      # 当前相机朝向下的激光先验
LASER_CFG = {
    "min_area": 6,        # 小于 6px² 的噪点不要
    "max_area": 400,      # 红袋子/大片红 >400px² 直接拒（激光约 80px²）
    "max_side": 24,       # 手/白屏/大块红 单边都 >24
    "max_aspect": 2.5,    # 红电线/橙绳细长，长宽比 >2.5 拒
}
LASER_EVERY = 1         # 每 N 帧检测一次
LASER_COARSE = 1        # 丢失时做全帧 LAB 粗搜（参考 elec_vision_k230）
LASER_LABEL = 1         # 1=画坐标文字
LASER_CROSS_COLOR = (0, 0, 255)
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
BULL_PRED_ON = 1               # 靶心延迟外推（运动状态估计）开关
BULL_PRED_MS = 80.0            # 感知-执行延迟估计（ms），待实测标定
AIM_LASER_ON_BULL = 1          # 1=把"靶心"平移成激光落点，让激光压在靶心上
AIM_LASER_MIN_LOCK = 6         # 激光至少连续锁定这么多帧才参与瞄准（防误锁跳点）
BULL_HOLD_ON = 1               # 靶纸短暂丢失时保持瞄准（不立刻放弃）
BULL_HOLD_FRAMES = 6           # 最多保持多少帧（18fps 约 0.33s）
LASER_AIM_JUMP = 40            # 激光瞄准点跳变门限（px），超过要连续确认
LASER_AIM_HOLD_MS = 3000       # 激光丢失后仍沿用记忆瞄准点的时间
DBG_RAW = 1                    # 1=DBG_SAVE 时保存原始帧（不带叠加），给训练用
DBG_RAW_DIR = "/sdcard/rot"    # 原始帧目录
DBG_RAW_MAX = 240              # 最多存多少张
PRINT_EVERY = 30
RESET_ON_EXIT = 1
LIMIT = 0                     # >0：跑 N 帧后退出（自动化测试用）
HEADLESS = 0                  # 1：不开 IDE，用 img.compress 模拟预览（量 FPS）
# ---- 稳定性（2026-10-23）----
# 1) 硬件看门狗：VM 卡死（相机/KPU 析构死锁）时自动复位，不再需要拔电
# 2) SAFE_DEINIT=0：退出时不调 sensor.stop()/MediaManager.deinit()
#    实测这两步可能死锁，导致 machine.reset() 永远执行不到 -> 整板假死
WDT_TIMEOUT_S = 20
SAFE_DEINIT = 0
# =================================================================

from media.sensor import *
from media.media import *
import machine
from media.display import Display
import libs.YOLO as _YL
from libs.YOLO import YOLOv8
from ulab import numpy as _np

import sys
import os
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
    import laser_repo as laser_spot_mod
except Exception as _e:
    laser_spot_mod = None
    print("laser_repo 不可用，激光检测关闭:", _e)

try:
    import laser_openmv as _laser_gray_mod
except Exception as _e:
    _laser_gray_mod = None
    print("laser_openmv 不可用:", _e)


class _GrayLaser:
    """把 d71fd47 版 LaserSpot（灰度 cv_lite + 全图捕获 + ROI 跟随）
    包装成 sample(img) -> dict 接口，供预览原有逻辑直接使用。"""

    def __init__(self, cfg=None):
        self.spot = _laser_gray_mod.LaserSpot(cfg)
        self.last_ms = 0

    def sample(self, img):
        try:
            g = img.to_grayscale()
        except Exception:
            g = img
        r = self.spot.sample(g, None)
        self.last_ms = int(getattr(self.spot, "last_ms", 0) or 0)
        if r is None:
            return None
        return {"x": r["x"], "y": r["y"], "n": r.get("n", 0),
                "w": 0, "h": 0, "area": 0, "raw": r.get("raw", 0),
                "conf": r.get("conf", 0.0)}

try:
    from laser_heatmap import LaserHeat as _LaserHeat
except Exception as _e:
    _LaserHeat = None
    print("laser_heatmap 不可用，回退 laser_repo:", _e)

try:
    from point_kalman import PointKalman
except Exception as _e:
    PointKalman = None
    print("point_kalman 不可用，靶心平滑关闭:", _e)

try:
    from laser_track import LaserTracker
except Exception as _e:
    LaserTracker = None
    print("laser_track 不可用，靶心预测关闭:", _e)

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
_WDT = None


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

# 靶心平滑（PointKalman 缺失时为 None = 不平滑）
_bull_kf = None   # 用户要求：靶心不做卡尔曼平滑（2026-10-24 改回）
_bull_sm = None
_bull_trk = None   # 靶心运动状态估计（laser_track.LaserTracker）
_last_box = None   # 最近一次有效框中心 (cx, cy)，保持瞄准用
_laser_aim = None  # 锁定的激光瞄准点 (x, y)，抗激光检测抖动
_laser_aim_ms = 0
_laser_aim_bad = 0
_dbg_raw_n = 0
_miss_frames = 0   # 连续未检出框的帧数

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
        cx = cy = -1
        bx = by = -1
        conf = 0
        if run:
            b = res[0][0]
            # b = [x, y, w, h]（左上角+宽高）→ 协议要求框中心
            cx = int(b[0] + b[2] * 0.5)
            cy = int(b[1] + b[3] * 0.5)
            if res[3] and len(res[3][0]) >= 15 \
                    and float(res[3][0][14]) >= BULL_MIN_KP:
                conf = int(float(res[3][0][14]) * 1000)
                if conf < 300:
                    # ESP32 AIM_BULL_MIN_CONF=300，保底防止被判靶心丢失
                    conf = 300
            if _bull_sm is not None:
                bx = int(round(_bull_sm[0]))
                by = int(round(_bull_sm[1]))
            elif conf:
                bx = int(res[3][0][12])
                by = int(res[3][0][13])
            if _bull_trk is not None:
                # 运动状态估计：外推到"指令生效时刻"
                _pp = _bull_trk.predict()
                if _pp is not None:
                    bx = int(max(0, min(W - 1, _pp[0])))
                    by = int(max(0, min(H - 1, _pp[1])))
        elif (BULL_HOLD_ON and _miss_frames <= BULL_HOLD_FRAMES
                and _last_box is not None and _bull_trk is not None):
            # 靶纸短暂丢失：用最近框中心 + 外推继续发 run=1，
            # 避免 ESP32 退回框中心或判定靶心丢失而停摆。
            _pp = _bull_trk.predict()
            if _pp is not None:
                run = 1
                cx = int(_last_box[0])
                cy = int(_last_box[1])
                conf = 300
                bx = int(max(0, min(W - 1, _pp[0])))
                by = int(max(0, min(H - 1, _pp[1])))
        # 激光压靶心：把上报点平移 (画面中心 - 激光点)，
        # ESP32 收紧后靶心正好落在激光落点（保持阶段同样生效）。
        _aim_ok = False
        if _laser_aim is not None:
            _aim_ok = time.ticks_diff(time.ticks_ms(), _laser_aim_ms) <= LASER_AIM_HOLD_MS
        if AIM_LASER_ON_BULL and run and bx >= 0 and by >= 0 and _aim_ok:
            _lx = int(_laser_aim[0])
            _ly = int(_laser_aim[1])
            bx = max(-1000, min(2000, bx + (W // 2 - _lx)))
            by = max(-1000, min(2000, by + (H // 2 - _ly)))
        _uart.write("$KT,%d,%d,%d,%d,%d,%d,%d!\n" % (seq, run, cx, cy, bx, by, conf))
        _uart.flush()
    except Exception as e:
        print("uart tx:", e)


def _bull_of(res):
    """取靶心 (x, y) 或 None。"""
    if not res or not res[0] or not res[3]:
        return None
    kp = res[3][0]
    if len(kp) >= 15 and float(kp[14]) >= BULL_MIN_KP:
        return (float(kp[12]), float(kp[13]))
    return None


def _update_bull_filter(res, now):
    """有靶心测量就 update，否则 predict 兜底；结果存 _bull_sm。"""
    global _bull_sm
    if _bull_kf is None:
        _bull_sm = None
        return
    bx = by = None
    if res and len(res) > 3 and res[3] and len(res[3][0]) >= 15:
        kp = res[3][0]
        if float(kp[14]) >= BULL_MIN_KP:
            bx, by = float(kp[12]), float(kp[13])
    if bx is not None:
        s = _bull_kf.update(bx, by, now)
    else:
        s = _bull_kf.predict(now)
    _bull_sm = None if s is None else (s[0], s[1])


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


def _draw(img, res, s=1.0, bull_xy=None):
    """画框、四角轮廓、靶心。bull_xy 给定时用平滑后的靶心坐标覆盖。"""
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
        if i == 0 and bull_xy is not None and len(pts) >= 5:
            bxv = bull_xy[0]
            byv = bull_xy[1]
            if (-1e6 < bxv < 1e6) and (-1e6 < byv < 1e6):
                pts[4] = (int(round(float(bxv) * s)),
                          int(round(float(byv) * s)))
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
_laser = None            # LaserRepo 实例（回退）
_heat = None             # LaserHeat 实例（优先）
# 2026-10-24：按用户要求，激光坐标重新加卡尔曼（见 _laser_kf）
_laser_lock = 0          # 连续命中计数（>=LASER_LOCK_N 视为锁定）
_laser_thr = 0.0         # 本帧实际使用的阈值（日志/调参用）
_laser_peak = 0.0        # 本帧峰值置信度（日志/调参用）
_laser_kf = None       # 激光坐标卡尔曼（2026-10-24：按用户要求加回）
_laser_coarse = None   # 全帧 LAB 粗搜器（LaserRepo，roi=整幅）
_laser_gray = None     # 全帧灰度搜索（d71fd47 laser_openmv）
_laser_prior = LASER_PRIOR
_laser_ms = 0
_ai_ms = 0
_laser_spot = None       # 最近一次结果 {"cx","cy","n","x","y","w","h"}
_laser_lock = None       # 锁定的 (cx, cy)
_laser_miss = 0
_laser_n = 0


def _init_laser():
    """优先建学习式热图（LaserHeat），失败回退 LaserRepo。"""
    global _laser, _heat, _laser_prior, _laser_lock, _laser_thr, _laser_peak, _laser_kf
    global _laser_coarse
    _laser = None
    _heat = None
    _laser_coarse = None
    _laser_prior = (float(LASER_PRIOR[0]), float(LASER_PRIOR[1]))
    # 激光坐标卡尔曼：q/r 与旧版一致；max_jump 放宽到 60（跳变门控已经先过滤了）
    _laser_kf = (PointKalman(q=120.0, r=9.0, max_jump=60.0, max_coast=4)
                 if PointKalman is not None else None)
    _laser_lock = 0
    _laser_thr = 0.0
    _laser_peak = 0.0
    if not LASER_ENABLE:
        return
    # 全帧 LAB 粗搜器：无论热图是否可用都建（用于丢失后的全画面重捕）
    if laser_spot_mod is not None and LASER_COARSE:
        try:
            _cfg2 = dict(LASER_CFG or {})
            _cfg2["roi"] = (0, 0, W, H)
            _laser_coarse = laser_spot_mod.LaserRepo(**_cfg2)
            print("激光全帧粗搜: roi=%s thr=%s" % (_laser_coarse.roi, _laser_coarse.thresholds))
        except Exception as e:
            _laser_coarse = None
            print("全帧粗搜初始化失败:", e)
    # 2026-10-24：热图做主检测，d71fd47 灰度全帧（laser_openmv）做扩展搜索：
    # 热图丢失时先用它全图找候选，再用热图复核。
    global _laser_gray
    if _laser_gray_mod is not None and LASER_ENABLE:
        try:
            _cfg2 = dict(LASER_CFG or {})
            _cfg2["red_primary"] = 0          # 灰度主检测
            _cfg2["acquire"] = 1              # 启动全图捕获，搜索范围=整幅
            _laser_gray = _GrayLaser(_cfg2)
            print("激光全帧灰度: laser_openmv %s thr=%s"
                  % (getattr(_laser_gray_mod, "VERSION", "?"),
                     _laser_gray.spot.cfg.get("threshold")))
        except Exception as _e:
            _laser_gray = None
            print("laser_openmv 初始化失败:", _e)
    if _LaserHeat is not None:
        try:
            _heat = _LaserHeat(LASER_MODEL)
            print("激光热图: model=%s roi=%d thr=%.2f/%.2f(acq/trk) prior=(%d,%d)"
                  % (LASER_MODEL, LASER_ROI, LASER_THR_ACQ, LASER_THR_TRK,
                     LASER_PRIOR[0], LASER_PRIOR[1]))
            return
        except Exception as e:
            _heat = None
            print("激光热图初始化失败，回退 laser_repo:", e)
    if laser_spot_mod is None:
        print("激光: laser_repo.py 缺失，跳过")
        return
    try:
        _laser = laser_spot_mod.LaserRepo(**(LASER_CFG or {}))
        print("激光: thr=%s ROI=%s 门限(area %s~%s, side<=%s, aspect<=%s) 版本 %s"
              % (_laser.thresholds, _laser.roi, _laser.min_area, _laser.max_area,
                 _laser.max_side, _laser.max_aspect,
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
    """画原版 ROI 框 + 选中的 blob 框 + 十字 + 坐标（640x480 原图坐标）。"""
    if spot is None:
        return
    # 选中的 blob 外接框
    if spot.get("w") and spot.get("h"):
        try:
            bw = int(spot["w"] * sc)
            bh = int(spot["h"] * sc)
            dst.draw_rectangle(int(round(spot["cx"] * sc)) - bw // 2,
                               int(round(spot["cy"] * sc)) - bh // 2,
                               bw, bh, color=(0, 255, 0), thickness=1)
        except Exception:
            pass
    cx = int(round(spot["cx"] * sc))
    cy = int(round(spot["cy"] * sc))
    dst.draw_cross(cx, cy, color=LASER_CROSS_COLOR,
                   size=max(8, int(14 * sc)), thickness=2)
    dst.draw_circle(cx, cy, max(6, int(12 * sc)), color=LASER_CROSS_COLOR,
                    thickness=2)
    if not (label and LASER_LABEL):
        return
    # 标签带上调节状态：T=已锁定(低阈值) / A=进入中(高阈值)，后面是峰值置信度
    _tag = "T" if _laser_lock >= LASER_LOCK_N else "A"
    txt = "(%d,%d)%s%.2f" % (int(spot["cx"]), int(spot["cy"]), _tag,
                             float(spot.get("conf", 0.0) or 0.0))
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
        # 激光直接在 chn1(RGB565) 上做 LAB find_blobs，不需要额外灰度通道
        _sensor = perf.open_sensor(2, W, H, framebuffers=SENSOR_FB, fps=SENSOR_FPS)
    else:
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
    # 硬件看门狗：卡死自动复位（相机初始化完成后再启动，避免误触发）
    global _WDT
    try:
        _WDT = machine.WDT(1, WDT_TIMEOUT_S)
        print("wdt: %ds 看门狗已启用" % WDT_TIMEOUT_S)
    except Exception as _e:
        _WDT = None
        print("wdt 不可用:", _e)
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
    global _bull_trk
    if BULL_PRED_ON and LaserTracker is not None:
        try:
            _bull_trk = LaserTracker(delay_ms=BULL_PRED_MS, q=4000.0, r=9.0)
            print("靶心预测: delay=%.0fms slope_n=%d" % (BULL_PRED_MS, _bull_trk.slope_n))
        except Exception as _e:
            _bull_trk = None
            print("靶心预测初始化失败:", _e)

    n = 0
    t_last = time.ticks_ms()
    fps = 0.0
    last_res = None
    while True:
        if _WDT is not None:
            try:
                _WDT.feed()
            except Exception:
                pass
        # 先拍显示帧，再拍 AI 帧（仅在 AI 帧执行）
        img = _sensor.snapshot(chn=CAM_CHN_ID_1)
        # ---- 激光：学习式热图（ROI/阈值调节模式，2026-10-23 优化）
        global _laser_spot, _laser_ms, _laser_prior, _laser_miss, _laser_lock
        global _laser_thr, _laser_peak
        if LASER_ENABLE and _heat is not None and (n % LASER_EVERY) == 0:
            try:
                locked = _laser_lock >= LASER_LOCK_N
                thr = LASER_THR_TRK if locked else LASER_THR_ACQ
                _laser_thr = thr
                _laser_peak = 0.0
                lx, ly = _laser_prior
                hx0 = max(0, min(W - LASER_ROI, int(lx - LASER_ROI // 2)))
                hy0 = max(0, min(H - LASER_ROI, int(ly - LASER_ROI // 2)))
                crop = img.crop(roi=(hx0, hy0, LASER_ROI, LASER_ROI)).to_rgb888().to_numpy_ref()
                r = _heat.detect(crop, hx0, hy0, thr)
                _laser_ms = 1
                if r is not None:
                    _laser_peak = float(r[2])
                    _ddx = r[0] - lx
                    _ddy = r[1] - ly
                    # 跳变门控：离先验太远的新点，必须达到"进入阈值"才算数
                    if (_ddx * _ddx + _ddy * _ddy) > (LASER_JUMP_PX * LASER_JUMP_PX) \
                            and float(r[2]) < LASER_THR_ACQ:
                        r = None
                if r is not None:
                    gx, gy, conf = r
                    # 激光坐标卡尔曼：画出来的坐标用平滑值
                    if _laser_kf is not None:
                        _st = _laser_kf.update(float(gx), float(gy), time.ticks_ms())
                        if _st is not None:
                            gx, gy = _st[0], _st[1]
                    if _laser_lock < 1000:
                        _laser_lock += 1
                    # 先验低通：只影响下一帧 ROI 中心
                    _laser_prior = (lx + (gx - lx) * LASER_PRIOR_A,
                                    ly + (gy - ly) * LASER_PRIOR_A)
                    _laser_miss = 0
                    _laser_spot = {"cx": gx, "cy": gy, "w": 0, "h": 0,
                                   "n": 0, "area": 0, "raw": 0, "conf": conf}
                else:
                    _laser_miss += 1
                    _laser_lock = 0
                    _laser_spot = None
                    # 短暂丢失：卡尔曼外推兜底（最多 max_coast 帧）
                    if _laser_kf is not None:
                        _sp = _laser_kf.predict(time.ticks_ms())
                        if _sp is not None:
                            _laser_spot = {"cx": _sp[0], "cy": _sp[1], "w": 0, "h": 0,
                                           "n": 0, "area": 0, "raw": 0, "conf": 0.0}
                    # 丢失：先做全帧 LAB 粗搜（参考 elec_vision_k230：一次 find_blobs
                    # 覆盖整幅 640x480），命中后用热图在候选处复核；
                    # 仍失败才每 12 帧跑一次热图全域网格兜底。
                    best = None
                    # 1) 灰度全帧候选（d71fd47 逻辑，1~2ms，覆盖整幅）→ 热图复核
                    if (LASER_LOST_SCAN and _laser_gray is not None
                            and (_laser_miss % LASER_GRAY_EVERY) == 0):
                        _ls2 = None
                        try:
                            _ls2 = _laser_gray.sample(img)
                        except Exception:
                            _ls2 = None
                        if _ls2 is not None:
                            _gx3 = int(_ls2["x"])
                            _gy3 = int(_ls2["y"])
                            _hx3 = max(0, min(W - LASER_ROI, _gx3 - LASER_ROI // 2))
                            _hy3 = max(0, min(H - LASER_ROI, _gy3 - LASER_ROI // 2))
                            try:
                                _c3 = img.crop(roi=(_hx3, _hy3, LASER_ROI, LASER_ROI)).to_rgb888().to_numpy_ref()
                                _r3 = _heat.detect(_c3, _hx3, _hy3, LASER_SCAN_THR)
                            except Exception:
                                _r3 = None
                            if _r3 is not None:
                                best = _r3
                    # 2) 灰度没找到时才用 LAB 粗搜
                    if best is None and LASER_LOST_SCAN and LASER_COARSE and _laser_coarse is not None:
                        _lc = None
                        try:
                            _lc = _laser_coarse.sample(img)
                        except Exception:
                            _lc = None
                        if _lc is not None:
                            _gx2 = int(_lc["x"])
                            _gy2 = int(_lc["y"])
                            _hx2 = max(0, min(W - LASER_ROI, _gx2 - LASER_ROI // 2))
                            _hy2 = max(0, min(H - LASER_ROI, _gy2 - LASER_ROI // 2))
                            try:
                                _c2 = img.crop(roi=(_hx2, _hy2, LASER_ROI, LASER_ROI)).to_rgb888().to_numpy_ref()
                                _r2 = _heat.detect(_c2, _hx2, _hy2, LASER_THR_ACQ)
                            except Exception:
                                _r2 = None
                            if _r2 is not None:
                                best = _r2
                    if (best is None and LASER_LOST_SCAN and _laser_miss >= LASER_LOST_MAX
                            and (_laser_miss % (LASER_SCAN_EVERY * 2)) == 0):
                        for _sy in range(0, H - LASER_ROI + 1, LASER_ROI):
                            for _sx in range(0, W - LASER_ROI + 1, LASER_ROI):
                                try:
                                    _c2 = img.crop(roi=(_sx, _sy, LASER_ROI, LASER_ROI)).to_rgb888().to_numpy_ref()
                                    _r2 = _heat.detect(_c2, _sx, _sy, LASER_THR_ACQ)
                                except Exception:
                                    _r2 = None
                                if _r2 is not None and (best is None or _r2[2] > best[2]):
                                    best = _r2
                    if best is not None:
                        _laser_prior = (best[0], best[1])
                        _laser_miss = 0
                        if _laser_kf is not None:
                            # 重捕是新位置：先 reset，避免被 max_jump 拦住
                            _laser_kf.reset()
                            _st = _laser_kf.update(float(best[0]), float(best[1]),
                                                   time.ticks_ms())
                            if _st is not None:
                                best = (_st[0], _st[1], best[2])
                        _laser_spot = {"cx": best[0], "cy": best[1], "w": 0, "h": 0,
                                       "n": 0, "area": 0, "raw": 0, "conf": best[2]}
            except Exception as e:
                print("heat:", e)
                _laser_spot = None
        elif LASER_ENABLE and _laser is not None and (n % LASER_EVERY) == 0:
            try:
                _ls = _laser.sample(img)
                _laser_ms = int(getattr(_laser, "last_ms", 0))
                if _ls is not None:
                    _laser_peak = float(_ls.get("conf", 0.0) or 0.0)
                    if _laser_lock < 1000:
                        _laser_lock += 1
                    _lxk = float(_ls["x"])
                    _lyk = float(_ls["y"])
                    if _laser_kf is not None:
                        _st = _laser_kf.update(_lxk, _lyk, time.ticks_ms())
                        if _st is not None:
                            _lxk, _lyk = _st[0], _st[1]
                    _laser_spot = {"cx": _lxk, "cy": _lyk,
                                   "n": _ls.get("n", 0),
                                   "w": _ls.get("w", 0), "h": _ls.get("h", 0),
                                   "area": _ls.get("area", 0),
                                   "raw": _ls.get("raw", 0)}
                else:
                    _laser_lock = 0
                    _laser_spot = None
                    if _laser_kf is not None:
                        _sp = _laser_kf.predict(time.ticks_ms())
                        if _sp is not None:
                            _laser_spot = {"cx": _sp[0], "cy": _sp[1], "n": 0,
                                           "w": 0, "h": 0, "area": 0, "raw": 0}
            except Exception as e:
                print("laser:", e)
                _laser_spot = None
        if last_res is None or (n % AI_EVERY) == 0:
            _t_ai = time.ticks_ms()
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
            _update_bull_filter(last_res, time.ticks_ms())
            if _bull_trk is not None:
                _b2 = _bull_of(last_res)
                if _b2 is not None:
                    _bull_trk.update(_b2[0], _b2[1], time.ticks_ms())
            _ai_ms = time.ticks_diff(time.ticks_ms(), _t_ai)
        res = last_res

        global _last_box, _miss_frames
        if res and res[0]:
            _miss_frames = 0
            _b0 = res[0][0]
            _last_box = (int(_b0[0] + _b0[2] * 0.5), int(_b0[1] + _b0[3] * 0.5))
        else:
            _miss_frames += 1

        # 激光瞄准点锁定：位置基本固定，用 EMA 缓慢跟随 + 大跳变确认，
        # 这样激光偶尔丢帧也不会让瞄准点在"激光/画面中心"之间来回跳。
        global _laser_aim, _laser_aim_ms, _laser_aim_bad
        if (_laser_spot is not None
                and _laser_lock >= AIM_LASER_MIN_LOCK):
            _ax = int(_laser_spot["cx"])
            _ay = int(_laser_spot["cy"])
            _now2 = time.ticks_ms()
            if _laser_aim is None:
                _laser_aim = (float(_ax), float(_ay))
                _laser_aim_ms = _now2
                _laser_aim_bad = 0
            else:
                _dx3 = _ax - _laser_aim[0]
                _dy3 = _ay - _laser_aim[1]
                if _dx3 * _dx3 + _dy3 * _dy3 <= LASER_AIM_JUMP * LASER_AIM_JUMP:
                    _laser_aim = (_laser_aim[0] + _dx3 * 0.2,
                                  _laser_aim[1] + _dy3 * 0.2)
                    _laser_aim_ms = _now2
                    _laser_aim_bad = 0
                else:
                    _laser_aim_bad += 1
                    if _laser_aim_bad >= 3:
                        _laser_aim = (float(_ax), float(_ay))
                        _laser_aim_ms = _now2
                        _laser_aim_bad = 0

        _send_uart(n, res)
        if DISPLAY_MODE == "ide":
            if HEADLESS:
                d = perf.enc_dims(W, H, VIEW_SCALE) if perf is not None else None
                sc = (d[0] / float(W)) if d else VIEW_SCALE
                disp = img.copy(x_scale=sc, y_scale=sc)
                _draw(disp, res, sc, _bull_sm)
                _draw_laser(disp, _laser_spot, sc, label=True)
                if d:
                    disp.compress(quality=IDE_QUALITY, x_size=d[0], y_size=d[1])
                else:
                    disp.compress(quality=IDE_QUALITY)
            elif VIEW == "full":
                # 先缩放到固定尺寸，再画覆盖层：绘制成本降数倍，帧率更稳
                d = perf.enc_dims(W, H, VIEW_SCALE) if perf is not None else None
                sc = (d[0] / float(W)) if d else VIEW_SCALE
                disp = img.copy(x_scale=sc, y_scale=sc)
                _draw(disp, res, sc, _bull_sm)
                _draw_laser(disp, _laser_spot, sc,
                            label=True)
                if d:
                    disp.compress_for_ide(x_size=d[0], y_size=d[1], quality=IDE_QUALITY)
                else:
                    disp.compress_for_ide(quality=IDE_QUALITY)
            else:
                _draw(img, res, 1.0, _bull_sm)
                _draw_laser(img, _laser_spot, 1.0,
                            label=True)
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
        if DBG_SAVE and DBG_RAW and n and (n % DBG_SAVE) == 0:
            try:
                global _dbg_raw_n
                if _dbg_raw_n < DBG_RAW_MAX:
                    try:
                        os.mkdir(DBG_RAW_DIR)
                    except Exception:
                        pass
                    img.save(DBG_RAW_DIR + "/rot_%04d.jpg" % _dbg_raw_n)
                    _dbg_raw_n += 1
            except Exception as e:
                print("save raw:", e)
        elif DBG_SAVE and n and (n % DBG_SAVE) == 0:
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
        if LIMIT and n >= LIMIT:
            break
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
                    if KP_NUM >= 5 and float(kp[14]) >= BULL_MIN_KP:
                        bx = int(kp[12])
                        by = int(kp[13])
                except Exception:
                    pass
            _nraw = 0 if _laser is None else int(getattr(_laser, "raw_blobs", 0))
            _nkept = 0 if _laser is None else int(getattr(_laser, "nblob", 0))
            if _laser_spot is not None:
                lz = "laser (%d,%d) area %d %dx%d" % (
                    int(_laser_spot["cx"]), int(_laser_spot["cy"]),
                    int(_laser_spot.get("area", 0)),
                    int(_laser_spot.get("w", 0)), int(_laser_spot.get("h", 0)))
            else:
                lz = "laser -"
            lz += " | lms %dms ai %dms raw %d kept %d thr %.2f lock %d peak %.2f" % (
                _laser_ms, _ai_ms, _nraw, _nkept, _laser_thr, _laser_lock, _laser_peak)
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
    if SAFE_DEINIT:
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
    else:
        # 实测 stop/deinit 会触发 KPU/相机析构死锁（之后 reset 都执行不到），
        # 所以直接跳过，交给 machine.reset() 整体重启（keepalive 防 GC 析构）。
        _sensor = None
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
