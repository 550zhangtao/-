# -*- coding: utf-8 -*-
"""
gimbal_agent.py —— K230 常驻代理：ESP32 说开始才开始跑识别
===========================================================

设计（2026-10-20 实机踩坑后定稿）
--------------------------------
* 上电常驻，**先不开相机**（开相机却不取帧会把 VICAP 管线堵死，实测能卡到
  REPL 都没反应、只能断电）；
* 相机/模型在**第一次收到 `$E,START!`** 时才打开（~1.5s，之后一直复用，不再重开）；
* 空闲时（还没 START，或已经 STOP）**持续排空两路帧**，保证管线不堵，
  这样 R2 一按立刻出帧；
* `$E,STOP!` 只是停止推理（不再占 KPU），相机继续排空，随时可以再 START；
* 相机配置、AI2D 反变换、pose 后处理全部沿用已验证的
  `yolo_preview_ide.py` / `yolo_pose.py`（双通道、RGBP888 通道送推理）。

与 ESP32 的协议
---------------
    ESP32 -> K230 : `$E,START!` / `$E,STOP!`
    K230 -> ESP32 : `$KT,<seq>,<run>,<cx>,<cy>,<bx>,<by>,<conf>!`（每帧）
    状态回执      : `$E,K230,READY!` / `START_OK!` / `STOP_OK!` / `ERR,INIT!`

免 REPL 切调试
--------------
    1.5 秒内连续收到 3 条 `$E,STOP!`（PC 可让 ESP32 连发 `gnone`）：
    写 `/sdcard/offline_off` 后立刻 `machine.reset()`。复位发生在 Python
    对象析构之前，不会触发 KPU/相机死锁；复位后本脚本看到 marker 直接退出，
    REPL 交还给 CanMV IDE。PC 侧入口：`bash scripts/mode.sh debug`。

部署
----
    python3 -m k230tool mode work --agent board/gimbal_agent.py
上电（不需要 USB，5V 独立供电即可）自动运行。想用 CanMV IDE 调试：
`bash scripts/mode.sh debug`（不要再手工 Ctrl-C / 拔插）。
"""

import gc
import os
import sys
import time

import machine
from media.sensor import *
from media.media import *
from libs.YOLO import YOLOv8
import libs.YOLO as _YL
from ulab import numpy as _np

try:
    import laser_spot
except Exception as _e:
    laser_spot = None
    print("agent: laser_spot import failed:", _e)

# ============================ 配置 ============================
MODEL = "/sdcard/models/pose_v3.kmodel"
LABELS = ["target"]
MODEL_IN = [320, 320]
W, H = 640, 480
SENSOR_ID = 2
CONF = 0.25
NMS = 0.45
MAX_BOXES = 20
KP_NUM = 5
KP_DIM = 3
BULL_MIN_V = 0.30

# ---- 激光上报（$LS，三态追踪接口；本轮不接控制）----
# 代理只在 R2/START 后跑，所以允许 fixed 模式的受限重捕（±180px，每 30 帧最多一次）。
# 没有 laser_calib.json 时直接关闭激光上报（不在瞄准启动时做全图标定，避免卡 2s）。
LASER_ENABLE = 1
LASER_VERIFY_EVERY = 5
LASER_GLOBAL = 1
LASER_CALIB_FILE = "/sdcard/laser_calib.json"

UART_ENABLE = 1
UART_ID = 3                     # 12Pin UART3: TX=GPIO32, RX=GPIO33
UART_TX = 32
UART_RX = 33
UART_BAUD = 115200

PRINT_EVERY = 60
DISABLE_FILE = "offline_off"
IDLE_MS = 30
# 严格"只由 ESP32 的 R2/START 触发"：默认 0 = 不自动开跑。
# 上电后没人按 R2 -> K230 一直空转等命令（上电默认无模式）。
# 想让它上电 N 毫秒后自己开跑，把这里改成 5000 之类即可。
AUTO_START_MS = 0
# =============================================================

_uart = None
_sensor = None
_yolo = None
_cam_open = False
_ready = False
_running = False
_rxbuf = ""
_fail = 0
_laser = None

# 工作模式下 REPL 被本脚本占用，不能再靠 Ctrl-C 切模式（实测会把板子
# 打进 USB 不枚举状态）。ESP32 的 gnone 命令会发 `$E,STOP!`，连续 3 条
# STOP（1.5s 内）作为“切调试模式”的暗号：PC 不需要 REPL，也不需要改
# ESP32 固件。
_stop_marks = []
_DEBUG_STOP_COUNT = 3
_DEBUG_STOP_WINDOW_MS = 1500

_keep = {}
try:
    _m = sys.modules
    _k = _m.get("_k230_ai_keepalive")
    if _k is None:
        _m["_k230_ai_keepalive"] = _keep
    else:
        _keep = _k
except Exception:
    pass


def _pose_postprocess(new_result, active_size, model_size, out_size,
                      nc, kp_num, kp_dim, conf, nms, max_boxes):
    """单目标 top-1 Pose 解码（板端 aidemo 没有 pose 后处理，运行时替换）。"""
    scores = new_result[:, 4]
    idx = int(_np.argmax(scores))
    score = float(scores[idx])
    if score < conf:
        return ([], [], [], [])
    row = new_result[idx]
    ah, aw = out_size
    mh, mw = model_size
    scale = min(mw / float(aw), mh / float(ah))
    # AI2D 左上角对齐, pad 恒为 0
    # YOLO 输出 [cx,cy,w,h] 是中心点：旧代码当左上角，$KT 的框中心会偏
    # w/2,h/2（26x36 靶纸偏 11~18px），云台瞄准也跟着偏。
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


# ---------------- UART ----------------
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
                             bits=machine.UART.EIGHTBITS,
                             parity=machine.UART.PARITY_NONE,
                             stop=machine.UART.STOPBITS_ONE)
        print("agent: uart%d tx=%d rx=%d %d" % (UART_ID, UART_TX, UART_RX, UART_BAUD))
    except Exception as e:
        _uart = None
        print("agent: uart init failed:", e)


def _reply(msg):
    if _uart is None:
        return
    try:
        _uart.write("$E,K230,%s!\n" % msg)
    except Exception:
        pass


def _record_stop():
    """3 条 STOP 在 1.5s 内 -> True。用于免 REPL 的“切调试”暗号。"""
    global _stop_marks
    try:
        now = time.ticks_ms()
        _stop_marks.append(now)
        while (_stop_marks and
               time.ticks_diff(now, _stop_marks[0]) > _DEBUG_STOP_WINDOW_MS):
            _stop_marks.pop(0)
        if len(_stop_marks) >= _DEBUG_STOP_COUNT:
            _stop_marks = []
            return True
    except Exception:
        _stop_marks = []
    return False


def _poll_cmd():
    global _rxbuf
    if _uart is None:
        return None
    try:
        if not _uart.any():
            return None
        data = _uart.read()
    except Exception:
        return None
    if not data:
        return None
    try:
        _rxbuf += data.decode()
    except Exception:
        return None
    if len(_rxbuf) > 600:
        _rxbuf = _rxbuf[-300:]
    cmd = None
    while True:
        i = _rxbuf.find("!")
        j = _rxbuf.find("\n")
        if i < 0 and j < 0:
            break
        if i < 0:
            idx = j
        elif j < 0:
            idx = i
        else:
            idx = i if i < j else j
        line = _rxbuf[:idx].strip()
        _rxbuf = _rxbuf[idx + 1:]
        if "START" in line:
            cmd = "START"
        elif "STOP" in line:
            cmd = "DEBUG" if _record_stop() else "STOP"
            if cmd == "DEBUG":
                break
    return cmd


# ---------------- 免 REPL 切调试 ----------------
def _enter_debug_mode():
    """写 offline_off 后立即整机复位：不析构 KPU/相机，避免原生对象死锁。"""
    global _running
    _running = False
    try:
        f = open("/sdcard/offline_off", "w")
        f.write("1")
        f.close()
        size = os.stat("/sdcard/offline_off")[6]
        if size <= 0:
            raise OSError("marker size 0")
    except Exception as e:
        print("agent: DEBUG marker failed:", e)
        _reply("ERR,DEBUG")
        return False
    try:
        print("agent: DEBUG requested -> offline_off written, reset to REPL")
        sys.stdout.flush()
    except Exception:
        pass
    _reply("DEBUG_OK")
    time.sleep_ms(200)
    try:
        import machine
        machine.reset()
    except Exception as e:
        print("agent: machine.reset failed:", e)
        return False
    return True


# ---------------- 相机 / 模型（按需打开） ----------------
def _open_camera():
    """双通道配置，与 yolo_preview_ide.py / yolo_pose.py 完全一致（已验证）。"""
    global _sensor, _cam_open
    if _cam_open:
        return True
    try:
        _sensor = Sensor(id=SENSOR_ID)
        _sensor.reset()
        _sensor.set_framesize(w=W, h=H, chn=CAM_CHN_ID_1)
        _sensor.set_pixformat(Sensor.RGB565, chn=CAM_CHN_ID_1)
        _sensor.set_framesize(w=W, h=H, chn=CAM_CHN_ID_2)
        _sensor.set_pixformat(Sensor.RGBP888, chn=CAM_CHN_ID_2)
        MediaManager.init()
        _sensor.run()
        time.sleep_ms(300)
        _cam_open = True
        print("agent: camera open")
        return True
    except Exception as e:
        _cam_open = False
        print("agent: camera open failed:", e)
        try:
            sys.print_exception(e)
        except Exception:
            pass
        return False


def _open_model():
    global _yolo, _ready
    if _ready:
        return True
    try:
        _yolo = YOLOv8(task_type="pose", mode="image", kmodel_path=MODEL,
                       labels=LABELS, rgb888p_size=[W, H],
                       model_input_size=MODEL_IN, display_size=[W, H],
                       conf_thresh=CONF, nms_thresh=NMS,
                       kp_num=KP_NUM, kp_dim=KP_DIM,
                       max_boxes_num=MAX_BOXES, debug_mode=0)
        _yolo.config_preprocess()
        try:
            gc.threshold(96 * 1024)
        except Exception:
            pass
        _ready = True
        print("agent: model ready (%s)" % MODEL)
        return True
    except Exception as e:
        _ready = False
        print("agent: model open failed:", e)
        try:
            sys.print_exception(e)
        except Exception:
            pass
        return False


def _drain_frames():
    """空闲时排空两路帧：不取帧会把 VICAP 管线堵死（实测会卡死整块板）。"""
    if not _cam_open:
        return
    try:
        _sensor.snapshot(chn=CAM_CHN_ID_1)
        _sensor.snapshot(chn=CAM_CHN_ID_2)
    except Exception as e:
        print("agent: drain failed:", e)


# ---------------- 推理 + 回传 ----------------
def _send_frame(seq, boxes, scores, kpts):
    if _uart is None:
        return
    run = 1 if boxes else 0
    conf = 0
    if run:
        b = boxes[0]
        cx = int(b[0] + b[2] * 0.5)
        cy = int(b[1] + b[3] * 0.5)
        kp = kpts[0]
        if len(kp) >= 15 and float(kp[14]) >= BULL_MIN_V:
            bx, by = int(kp[12]), int(kp[13])
            # $KT 的 conf 协议上是**靶心置信度**（yolo_pose.py 也这么发）；
            # 旧代理发的是框分数，ESP32 三态接口按 bull conf 判有效会误判。
            conf = int(float(kp[14]) * 1000)
        else:
            bx = by = -1
    else:
        cx = cy = bx = by = -1
    _uart.write("$KT,%d,%d,%d,%d,%d,%d,%d!\n" %
                (seq, run, cx, cy, bx, by, conf))
    try:
        _uart.flush()
    except Exception:
        pass


def _load_laser_calib():
    """读 /sdcard/laser_calib.json（与 IDE 预览 / yolo_pose 共用）。"""
    try:
        f = open(LASER_CALIB_FILE)
        s = f.read()
        f.close()
        i = s.find('"x"')
        j = s.find('"y"')
        if i >= 0 and j >= 0:
            return (float(s[s.find(":", i) + 1:s.find(",", i)]),
                    float(s[s.find(":", j) + 1:s.find("}", j)]))
    except Exception:
        pass
    return None


def _open_laser():
    """初始化激光检测（固定标定 + 受限重捕）；失败不影响 $KT 推流。"""
    global _laser
    if not LASER_ENABLE or laser_spot is None:
        _laser = None
        return False
    fx = _load_laser_calib()
    if fx is None or fx[0] <= 0:
        _laser = None
        print("agent: 无 %s -> 激光上报关闭（先在 IDE 预览里标定一次）"
              % LASER_CALIB_FILE)
        return False
    try:
        _laser = laser_spot.LaserSpot({
            "fixed_verify_every": LASER_VERIFY_EVERY,
            "fixed_allow_reacquire": LASER_GLOBAL,
        })
        _laser.set_fixed(fx[0], fx[1])
        print("agent: laser fixed (%.1f,%.1f), verify every %d, reacquire=%d"
              % (fx[0], fx[1], LASER_VERIFY_EVERY, LASER_GLOBAL))
        return True
    except Exception as e:
        _laser = None
        print("agent: laser init failed:", e)
        return False


def _send_laser(seq, laser):
    """`$LS,<seq>,<run>,<x*10>,<y*10>,<conf*1000>,<src>!` —— 三态接口数据源。"""
    if _uart is None:
        return
    try:
        if laser is not None and float(laser.get("x", -1)) >= 0:
            src = 2 if laser.get("src") == "fixed" else 1
            _uart.write("$LS,%d,1,%d,%d,%d,%d!\n" % (
                seq, int(laser["x"] * 10), int(laser["y"] * 10),
                int(float(laser.get("conf", 0)) * 1000), src))
        else:
            fx = getattr(_laser, "fixed", None) if _laser is not None else None
            if fx is not None and fx[0] > 0:
                _uart.write("$LS,%d,1,%d,%d,0,2!\n" %
                            (seq, int(fx[0] * 10), int(fx[1] * 10)))
            else:
                _uart.write("$LS,%d,0,-1,-1,0,0!\n" % seq)
        _uart.flush()
    except Exception as e:
        print("agent: laser tx:", e)


def _stream_frame(seq):
    img1 = _sensor.snapshot(chn=CAM_CHN_ID_1)           # 显示通道：激光检测用
    laser = None
    if _laser is not None:
        try:
            laser = _laser.locate(img1)
        except Exception as e:
            print("agent: laser:", e)
    arr = _sensor.snapshot(chn=CAM_CHN_ID_2).to_numpy_ref()
    res = _yolo.run(arr)
    if res and len(res) >= 4:
        boxes, scores, kpts = res[0], res[2], res[3]
    else:
        boxes, scores, kpts = [], [], []
    _send_frame(seq, boxes, scores, kpts)
    _send_laser(seq, laser)
    return len(boxes)


def _drop_pipeline():
    """连续出错时丢相机，等下一次 START 重开（模型保留，避免 KPU 反复构造）。"""
    global _cam_open, _running, _fail
    _fail = 0
    _running = False
    _cam_open = False
    try:
        _sensor.stop()
    except Exception:
        pass
    try:
        MediaManager.deinit()
    except Exception:
        pass
    print("agent: camera dropped, 等下一次 $E,START!")


def main():
    global _running, _fail

    try:
        if DISABLE_FILE in os.listdir("/sdcard"):
            print("agent: /sdcard/%s 存在, 跳过自动运行" % DISABLE_FILE)
            return
    except Exception:
        pass

    _init_uart()
    _reply("READY")
    print("agent: 常驻就绪（相机按需打开），等 $E,START! / $E,STOP!")

    seq = 0
    fps = 0.0
    t_last = time.ticks_ms()
    spent = 0
    boot_ms = time.ticks_ms()
    heard_cmd = False
    auto_done = False

    while True:
        cmd = _poll_cmd()
        if cmd == "DEBUG":
            # 写 offline_off 后立刻复位；不要先 _drop_pipeline()/gc，
            # 否则会在 KPU/相机原生对象析构时死锁（实测会把整块板卡死）。
            _enter_debug_mode()
            return
        if cmd:
            heard_cmd = True
        # 冷启动自愈：上电后一直没人喊 START，就自己开跑（只触发一次；AUTO_START_MS=0 关闭）
        if (cmd is None and not heard_cmd and not _running and not auto_done
                and AUTO_START_MS and time.ticks_diff(time.ticks_ms(), boot_ms) >= AUTO_START_MS):
            auto_done = True
            cmd = "START"
            print("agent: %dms 没收到 START, 自动开跑" % AUTO_START_MS)
        if cmd == "START":
            ok = _open_camera() and _open_model()
            _open_laser()          # 失败只关闭 $LS，不影响 $KT 推流
            _running = ok
            _reply("START_OK" if ok else "ERR,INIT")
            print("agent: START -> %s" % ("streaming" if ok else "init failed"))
        elif cmd == "STOP":
            _running = False
            _reply("STOP_OK")
            print("agent: STOP -> idle")

        if _running and _ready and _cam_open:
            try:
                _stream_frame(seq)
                seq += 1
                _fail = 0
                spent += 1
                dt = time.ticks_diff(time.ticks_ms(), t_last)
                t_last = time.ticks_ms()
                if dt > 0:
                    inst = 1000.0 / dt
                    fps = inst if fps == 0 else (0.8 * fps + 0.2 * inst)
                if PRINT_EVERY and (spent % PRINT_EVERY) == 0:
                    print("agent: %d frames, %.1ffps" % (spent, fps))
                if (spent & 0x07) == 0:
                    gc.collect()
            except Exception as e:
                _fail += 1
                print("agent: stream error(%d):" % _fail, e)
                if _fail >= 3:
                    _drop_pipeline()
                time.sleep_ms(50)
        else:
            _drain_frames()
            time.sleep_ms(IDLE_MS)


if __name__ == "__main__":
    while True:
        try:
            main()
            break
        except KeyboardInterrupt:
            print("agent: interrupted -> 退出，REPL 交还给 IDE")
            break
        except Exception as e:
            msg = str(e)
            low = msg.lower()
            # CanMV IDE / VSCode 插件按"停止"时抛的是 Exception("IDE interrupt")，
            # 不是 KeyboardInterrupt。这种情况必须**退出**把 REPL 交还；否则代理会
            # 自我重启、REPL 一直被占，IDE 就报"当前固件不支持 REPL 输入"。
            if "interrupt" in low or "ide" in low:
                print("agent: 被 IDE 中断(%s) -> 退出，REPL 交还；重新上电或"
                      "用 IDE 运行 /sdcard/main.py 可恢复" % msg)
                break
            print("agent ERROR", type(e).__name__, e)
            try:
                sys.print_exception(e)
            except Exception:
                pass
            _drop_pipeline()
            time.sleep_ms(2000)     # 真异常才重来；不复位整机
