# -*- coding: utf-8 -*-
"""JPEG 解码、检测框绘制、状态叠加（主机端，OpenCV）。"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

# BGR 调色板（类别多时循环使用）
_PALETTE = [
    (60, 220, 60), (60, 180, 255), (255, 160, 60), (255, 80, 220),
    (80, 255, 240), (160, 120, 255), (240, 240, 80), (255, 120, 120),
]


def require_cv2():
    if cv2 is None:
        raise RuntimeError("需要 opencv-python：pip3 install opencv-python")
    return cv2


def decode_jpeg(jpeg: bytes):
    """JPEG bytes -> BGR ndarray；失败返回 None。"""
    if not jpeg:
        return None
    cv = require_cv2()
    arr = np.frombuffer(jpeg, np.uint8)
    return cv.imdecode(arr, cv.IMREAD_COLOR)


def encode_jpeg(frame, quality: int = 85) -> bytes:
    cv = require_cv2()
    ok, buf = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


def _scale_box(box: Sequence[float], src_w: int, src_h: int,
               dst_w: int, dst_h: int) -> Tuple[int, int, int, int]:
    x, y, w, h = box[:4]
    sx = dst_w / float(src_w or dst_w)
    sy = dst_h / float(src_h or dst_h)
    x1 = max(0, min(dst_w - 1, int(round(x * sx))))
    y1 = max(0, min(dst_h - 1, int(round(y * sy))))
    x2 = max(0, min(dst_w, int(round((x + w) * sx))))
    y2 = max(0, min(dst_h, int(round((y + h) * sy))))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def draw_detections(frame, target: Optional[Dict], labels: Optional[List[str]] = None):
    """画检测框、4 角点轮廓和靶心（Pose 输出），支持 ROI 裁剪/缩放。"""
    if not target:
        return frame
    cv = require_cv2()
    h, w = frame.shape[:2]
    src_w = int(target.get("w") or w)
    src_h = int(target.get("h") or h)
    roi = target.get("roi") or [0, 0, src_w, src_h]
    sx = w / float(roi[2] or w)
    sy = h / float(roi[3] or h)
    ox, oy = float(roi[0]), float(roi[1])

    def mp(x, y):
        return int(round((x - ox) * sx)), int(round((y - oy) * sy))

    det = target.get("det") or []
    kp = target.get("kp") or []
    for idx, item in enumerate(det):
        try:
            box, conf, cls = item[:4], float(item[4]), int(item[5])
        except (TypeError, ValueError, IndexError):
            continue
        color = _PALETTE[cls % len(_PALETTE)]
        name = (labels or target.get("labels") or ["cls%d" % cls])
        try:
            name = name[cls]
        except Exception:
            name = "cls%d" % cls
        x1, y1 = mp(box[0], box[1])
        x2, y2 = mp(box[0] + box[2], box[1] + box[3])
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        # 5 关键点：前 4 个是角点，第 5 个是白色 10 号靶心
        pts = []
        if len(kp) >= 15:
            for k in range(5):
                px, py, pv = kp[k * 3], kp[k * 3 + 1], kp[k * 3 + 2]
                if pv is not None and float(pv) >= 0.3:
                    pts.append((mp(px, py), float(pv)))
                else:
                    pts.append(None)
            quad = [p for p in pts[:4] if p]
            if len(quad) >= 4:
                cv.polylines(frame, [np.array([p[0] for p in quad], np.int32)],
                             True, color, 2, cv.LINE_AA)
            for i, p in enumerate(pts[:4]):
                if p:
                    cv.circle(frame, p[0], 3, color, -1)

        text = "%s %.2f" % (name, conf)
        (tw, th), base = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(14, y1)
        cv.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 6, ty + base - 2), color, -1)
        cv.putText(frame, text, (x1 + 3, ty - 3), cv.FONT_HERSHEY_SIMPLEX,
                   0.55, (0, 0, 0), 1, cv.LINE_AA)

    # 靶心：只有校验通过才画，失败状态不输出错误坐标
    bull = target.get("bull") or {}
    bcx = bcy = None
    if bull.get("status") == "ok" and bull.get("x") is not None:
        bcx, bcy = bull["x"], bull["y"]
    if bcx is not None:
        bx, by = mp(bcx, bcy)
        cv.drawMarker(frame, (bx, by), (0, 0, 255), cv.MARKER_CROSS, 14, 2)
        cv.circle(frame, (bx, by), 6, (0, 0, 255), 1, cv.LINE_AA)
        txt = "10"
        cv.putText(frame, txt, (bx + 8, by - 8), cv.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 0, 255), 1, cv.LINE_AA)

    # 激光光斑：青色准星 + 坐标；有靶心时画黄色偏差连线
    laser = target.get("laser") or {}
    if laser.get("status") == "ok" and laser.get("x") is not None and laser["x"] >= 0:
        lx, ly = mp(laser["x"], laser["y"])
        cv.drawMarker(frame, (lx, ly), (255, 128, 0), cv.MARKER_CROSS, 18, 2)
        cv.circle(frame, (lx, ly), 9, (255, 128, 0), 1, cv.LINE_AA)
        if bcx is not None:
            cv.line(frame, (bx, by), (lx, ly), (0, 255, 255), 1, cv.LINE_AA)
        cv.putText(frame, "(%.0f,%.0f)" % (laser["x"], laser["y"]),
                   (lx + 11, ly - 10), cv.FONT_HERSHEY_SIMPLEX,
                   0.48, (255, 128, 0), 1, cv.LINE_AA)
    return frame


def draw_hud(frame, lines: Sequence[Tuple[str, Tuple[int, int, int]]]):
    """左上角状态板：lines = [(text, BGR), ...]。"""
    cv = require_cv2()
    h, w = frame.shape[:2]
    scale = max(0.38, min(0.62, w / 1100.0))
    lh = max(16, int(round(34 * scale)))
    margin = max(6, int(round(10 * scale)))
    box_h = margin * 2 + lh * len(lines)
    shade = frame.copy()
    cv.rectangle(shade, (0, 0), (w, box_h), (0, 0, 0), -1)
    cv.addWeighted(shade, 0.5, frame, 0.5, 0, frame)
    for i, (text, color) in enumerate(lines):
        y = margin + lh * (i + 1) - 4
        cv.putText(frame, text, (margin + 1, y + 1), cv.FONT_HERSHEY_SIMPLEX,
                   scale, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(frame, text, (margin, y), cv.FONT_HERSHEY_SIMPLEX,
                   scale, color, 1, cv.LINE_AA)
    return frame


def annotate(jpeg: bytes, target: Optional[Dict], labels: Optional[List[str]] = None,
             hud: bool = True, fps: float = 0.0, seq: int = 0,
             extra: Optional[str] = None, quality: int = 82) -> bytes:
    """原始 JPEG -> 带框/关键点/靶心/HUD 的 JPEG。"""
    frame = decode_jpeg(jpeg)
    if frame is None:
        return jpeg
    if target:
        frame = draw_detections(frame, target, labels)
    if hud:
        t = target or {}
        n = len(t.get("det") or [])
        infer = t.get("ms")
        src_w = int(t.get("w") or frame.shape[1])
        roi = t.get("roi") or [0, 0, src_w, 1]
        view = "ROI %dx%d" % (roi[2], roi[3]) if roi[2] < src_w * 0.99 else "FULL"
        bull = t.get("bull") or {}
        if bull.get("status") == "ok":
            bt = "bull10  ok  %.2f  (%.0f,%.0f)" % (
                float(bull.get("conf") or 0), float(bull.get("x") or 0),
                float(bull.get("y") or 0))
            bcol = (60, 180, 255)
        elif bull:
            bt = "bull10  %s" % bull.get("status", "?")
            bcol = (80, 80, 255)
        else:
            bt = "bull10  -"
            bcol = (150, 150, 150)
        lz = t.get("laser") or {}
        if lz.get("status") == "ok" and float(lz.get("x") or -1) >= 0:
            lt = "laser  ok  %.0f,%.0f  conf %.2f" % (
                float(lz["x"]), float(lz["y"]), float(lz.get("conf") or 0))
            if bull.get("status") == "ok":
                lt += "  d(%+.0f,%+.0f)" % (float(lz["x"]) - float(bull["x"]),
                                            float(lz["y"]) - float(bull["y"]))
            lcol = (255, 255, 0)
        else:
            lt = "laser  -"
            lcol = (120, 120, 120)
        lines = [
            ("K230 Pose  %s  %.1f fps  seq %d" % (view, fps, seq),
             (120, 255, 120)),
            (time.strftime("%Y-%m-%d %H:%M:%S"), (210, 210, 210)),
            ("targets %d%s" % (n, "" if infer is None else "   infer %sms" % infer),
             (80, 255, 240) if n else (160, 160, 160)),
            (bt, bcol),
            (lt, lcol),
        ]
        if extra:
            lines.append((extra, (200, 200, 200)))
        frame = draw_hud(frame, lines)
    return encode_jpeg(frame, quality)
