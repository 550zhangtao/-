# -*- coding: utf-8 -*-
"""
k230_perf.py —— K230 板端性能配置（单点入口，三处脚本共用）
=========================================================

`yolo_pose.py`（PC 推流）、`yolo_preview_ide.py`（IDE 预览）、`gimbal_agent.py`
（离线云台代理）都调用这里，避免"改一处忘两处"。

PC 下发时**只替换 `CFG = {...}` 单行，不替换本文件**，所以这里的默认值必须
"离线代理也能用"。PC 端要覆盖时下发 `perf_*` 键（见 `k230tool/link.py` 的
`PERF_KEYS`），优先级：传入参数 > `CFG["perf_*"]` > 本文件默认值。

2026-10-22 实测（CanMV v1.4.3 / GC2093 / 640×480 双通道）
--------------------------------------------------------
| 项目 | 实测 | 结论 |
| --- | --- | --- |
| 双通道采集 | **57 fps**（17.5 ms/帧对） | 采集不是瓶颈 |
| `img.compress` 640×480 q70（RGB565） | **65~70 ms** | **编码才是瓶颈** |
| 同上（RGB888 通道） | 77 ms | 预览/编码一律用 RGB565 通道 |
| 512×384 q70 / q92 | 52 / 58 ms | 同样帧率下 q92 更清晰，文件也只大一点 |
| 448×336 q92 | 48 ms | 约 21 fps 回传 |
| 384×288 q92 | 39 ms | 约 25 fps 回传 |
| 640×480 q92 | 78 ms | 约 13 fps，**不推荐** |
| Pose 384 推理 | 40~43 ms | 第二瓶颈（`ai_every=2` 时摊薄一半） |
| 端到端 `live`（384 模型 + ROI 回传） | **约 11 fps**（PC 侧） | 板端约 20~26 fps |

> 结论：**ROI 模式下不要指望"提高实测 fps"**——每帧一块原生像素，编码天然
> 50~70 ms。想更快只能降回传像素数或隔帧。之前文档里的"25~28 fps"是
> 隔帧/小 ROI 的旧口径，不是同一件事。

因此：

1. **CPU 频率在本固件上调不了**（没有 `os.set_cpu_freq`，`os` 只有 `cpu_usage`），
   代码保留探测，日志会写 `cpu=auto`，不要指望这个开关；
2. **`set_framebuffers` 也不支持**（固件直接打印 not support），已 try 包住；
   采集缓冲由固件按 `Sensor(fps=...)` 自动分配（实测 buffer_num=4）；
3. 提升帧率的正确顺序：**降编码像素数**（`ENC_SCALE`）> 隔帧 `jpeg_every`
   > 降 `ENC_QUALITY`。信息量看像素，不看 JPEG 质量档位；
   同样的帧率下 `0.80×q92` 比 `1.0×q70` **更清晰**（文件反而小一半）。

`ENC_SCALE` 是唯一旋钮（只影响**无目标时的全图**回传，有目标时走
`yolo_pose.CFG["roi_scale"]`）：0.80 → 512×384；1.0 → 640×480。
PC 端可用 `live --enc-scale` 覆盖。
"""

from media.media import MediaManager

# 注意：模块有自己的命名空间 —— 调用方脚本里 `from media.sensor import *`
# 产生的 Sensor / CAM_CHN_ID_* **不会**出现在这里，必须自己 import。
# （2026-10-22 实测：漏了这行 -> NameError: name 'Sensor' isn't defined）
try:
    from media.sensor import Sensor, CAM_CHN_ID_1, CAM_CHN_ID_2
except Exception:            # 个别固件常量名不同，回落到数字
    from media.sensor import Sensor
    CAM_CHN_ID_1 = 1
    CAM_CHN_ID_2 = 2

# ---- CPU 频率 ----
# "auto" / "balanced" / "performance"
# 注意：本固件没有调频接口，三种取值实际都落在 "auto"（不会报错，只打日志）。
PERF_MODE = "performance"

# ---- 相机 ----
SENSOR_FPS = 60          # GC2093 原生 1920x1080@60；请求 >采集能力 的值不会更快
SENSOR_FRAMEBUFFERS = 3  # 固件不支持 set_framebuffers 时自动跳过

# ---- JPEG 回传（性能主旋钮）----
ENC_SCALE = 0.80         # 回传图缩放：0.80 -> 512x384；1.0 = 原生 640x480
ENC_QUALITY = 92         # 0.80 缩放下 q92 的编码耗时仍低于满幅 q70

_FREQ = {
    "balanced": (800000000, 800000000),
    "performance": (1600000000, 800000000),
}

_warned = {}
_applied = None


def _log(msg):
    # PC 下发的脚本自带 log()；IDE 预览没有，就用 print
    try:
        log(msg)          # noqa: F821  （板端脚本的全局函数）
    except Exception:
        print(msg)


def _try(name, fn, arg):
    try:
        fn(arg)
        return True
    except Exception as e:
        if name not in _warned:
            _warned[name] = True
            _log("perf: %s 不可用（%s）" % (name, e))
        return False


def set_cpu_frequency(mode=None):
    """尝试拉高大小核频率；本固件没有接口时返回 "auto"。"""
    global _applied
    mode = (mode or PERF_MODE or "auto").lower()
    if mode not in _FREQ:
        _applied = "auto"
        return _applied
    big, little = _FREQ[mode]
    try:
        import os
    except Exception:
        os = None
    ok = False
    if os is not None and hasattr(os, "set_cpu_freq"):
        ok = _try("os.set_cpu_freq(big,little)", os.set_cpu_freq, (big, little))
        if not ok:
            ok = _try("os.set_cpu_freq(big)", os.set_cpu_freq, big)
    _applied = mode if ok else "auto"
    return _applied


def apply_cpu(mode=None):
    """脚本启动时调用一次，打印结果，方便从日志确认。"""
    mode = set_cpu_frequency(mode)
    if mode != "auto":
        big, little = _FREQ[mode]
        _log("perf: CPU 大核 %dMHz / 小核 %dMHz" % (big // 1000000, little // 1000000))
    else:
        _log("perf: CPU 频率沿用固件默认（本固件无调频接口）")
    return mode


def open_sensor(sensor_id=2, w=640, h=480, framebuffers=None, fps=None):
    """统一口径打开双通道相机，返回已 run() 的 Sensor。

    通道 1 = RGB565（预览/编码，比 RGB888 编码快 12 ms）；
    通道 2 = RGBP888（送 AI）。
    需要调用方已经 `from media.sensor import *`（提供 Sensor / MediaManager）。
    """
    f = SENSOR_FPS if fps is None else fps
    try:
        sen = Sensor(id=sensor_id, fps=f) if f else Sensor(id=sensor_id)
    except TypeError:
        # 老固件 Sensor() 不吃 fps 关键字
        sen = Sensor(id=sensor_id)
    sen.reset()
    n = SENSOR_FRAMEBUFFERS if framebuffers is None else int(framebuffers)
    if n and n > 0:
        _try("set_framebuffers(%d)" % n, sen.set_framebuffers, n)
    # 通道 1：预览 / JPEG 编码
    sen.set_framesize(w=w, h=h, chn=CAM_CHN_ID_1)
    sen.set_pixformat(Sensor.RGB565, chn=CAM_CHN_ID_1)
    # 通道 2：AI
    sen.set_framesize(w=w, h=h, chn=CAM_CHN_ID_2)
    sen.set_pixformat(Sensor.RGBP888, chn=CAM_CHN_ID_2)
    MediaManager.init()
    sen.run()
    return sen


def enc_dims(w, h, scale=None):
    """回传图尺寸；None 表示原生尺寸（不缩放）。取偶数避免 JPEG 边缘补齐。"""
    sc = ENC_SCALE if scale is None else scale
    if sc is None or float(sc) >= 0.999:
        return None
    sc = float(sc)
    return (max(2, int(round(w * sc / 2.0)) * 2),
            max(2, int(round(h * sc / 2.0)) * 2))


def encode(img, roi=None, scale=None, quality=None):
    """统一编码口径，返回 (jpeg_bytes, [w,h], roi)。

    roi 给定时编码该区域（原生像素，不缩放）；否则按 scale/ENC_SCALE 缩放整幅。
    """
    q = ENC_QUALITY if quality is None else int(quality)
    if roi is not None:
        x, y, w, h = [int(v) for v in roi]
        return _bytes(img.compress(quality=q, roi=(x, y, w, h))), [w, h], [x, y, w, h]
    w0, h0 = img.width(), img.height()
    dims = enc_dims(w0, h0, scale)
    if dims is None:
        return _bytes(img.compress(quality=q)), [w0, h0], [0, 0, w0, h0]
    return (_bytes(img.compress(quality=q, x_size=dims[0], y_size=dims[1])),
            [dims[0], dims[1]], [0, 0, w0, h0])


def _bytes(j):
    return j if isinstance(j, bytes) else bytes(j)


def status():
    """给 hello / stats 日志用的一行摘要。"""
    d = enc_dims(640, 480)
    return "cpu=%s fb=%s enc=%s/q%d%s" % (
        _applied or "?", SENSOR_FRAMEBUFFERS, ENC_SCALE, ENC_QUALITY,
        "" if d is None else "(%dx%d)" % d)
