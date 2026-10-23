#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monitor.py —— K230 摄像头实时监视窗口
=====================================

把 K230 当成普通 USB 摄像头用：开一个 OpenCV 窗口实时显示，并叠加帧率/延迟信息。

用法：
    python3 -m k230tool monitor                  # 默认 640x480 q70
    python3 -m k230tool monitor -w 320 -h 240 -q 60 --fps 30
    python3 -m k230tool monitor --record         # 启动即开始录像
    python3 -m k230tool monitor --port /dev/ttyACM1

窗口内快捷键：
    q / ESC   退出
    s         保存当前帧为 PNG（存到 captures/）
    r         开始 / 停止录像
    m         切换画面镜像（仅显示层）
    Space     暂停 / 继续
"""

from __future__ import annotations
# --- PC 端脚本守卫：误在 K230 上运行时给出明确提示 ---
try:
    import sys as _sys
    if getattr(_sys, "implementation", None) is not None and _sys.implementation.name == "micropython":
        print("[PC 脚本] 这是电脑端工具，请在 PC 上运行；板端请运行 k230/board/*.py")
        raise SystemExit
except SystemExit:
    raise
except Exception:
    pass

import argparse
import signal
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .link import DEFAULT_CFG, K230Stream, find_port, describe_ports  # noqa: E402


def decode(jpeg: bytes):
    """JPEG bytes -> BGR ndarray；失败返回 None。"""
    arr = np.frombuffer(jpeg, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


class Recorder:
    """录像小封装，自动挑一个可用的编码器。"""

    def __init__(self, out_dir: Path, size):
        self.out_dir = out_dir
        self.size = size
        self.writer = None
        self.path = None
        self.frames = 0

    def start(self, fps: float = 15.0):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        candidates = [
            (self.out_dir / ("k230_%s.mp4" % ts), "mp4v"),
            (self.out_dir / ("k230_%s.avi" % ts), "XVID"),
        ]
        for path, codec in candidates:
            w = cv2.VideoWriter(str(path),
                                cv2.VideoWriter_fourcc(*codec),
                                max(1.0, fps), self.size)
            if w.isOpened():
                self.writer, self.path, self.frames = w, path, 0
                return path
            w.release()
        raise RuntimeError("没有可用的视频编码器（mp4v / XVID 都打不开）")

    def write(self, frame):
        if self.writer is not None:
            self.writer.write(frame)
            self.frames += 1

    def stop(self):
        if self.writer is not None:
            self.writer.release()
            path, n = self.path, self.frames
            self.writer, self.path = None, None
            return path, n
        return None, 0


def draw_overlay(frame, info, paused, recording, recording_secs):
    """在画面上写状态文字（只用 ASCII，避免中文字体问题）。"""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 叠加层随画面宽度自适应，窗口缩放后排版也不会挤
    scale = max(0.32, min(0.70, w / 1200.0))
    lh = max(14, int(round(42 * scale)))     # 行高
    margin = max(5, int(round(12 * scale)))
    lines = [
        ("K230 %dx%d   %.1f fps" % (info["w"], info["h"], info["fps"]), (120, 255, 120)),
        ("frame %d   %.1f KB   age %d ms" % (info["seq"], info["kb"], info["lag_ms"]), (200, 220, 200)),
        ("board %.1f fps   enc %s ms   heap %.0f KB"
         % (info["board_fps"], info["enc"], info["heap_kb"]), (200, 220, 200)),
        ("rx %.1f MB   bad %d" % (info["rx_mb"], info["bad"]),
         (200, 220, 200) if info["bad"] == 0 else (80, 80, 255)),
    ]

    # 半透明底板，保证任何画面下都看得清
    box_h = margin * 2 + lh * len(lines)
    shade = frame.copy()
    cv2.rectangle(shade, (0, 0), (w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(shade, 0.45, frame, 0.55, 0, frame)

    for i, (text, color) in enumerate(lines):
        y = margin + lh * (i + 1) - 5
        cv2.putText(frame, text, (margin + 2, y + 1), font, scale,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (margin, y), font, scale, color, 1, cv2.LINE_AA)

    # 状态角标
    tag = None
    if recording:
        tag = ("REC %02d:%02d" % (recording_secs // 60, recording_secs % 60), (60, 60, 255))
    elif paused:
        tag = ("PAUSED", (60, 200, 255))
    if tag:
        (tw, _), _ = cv2.getTextSize(tag[0], font, scale, 1)
        cv2.rectangle(frame, (w - tw - 20, 6), (w - 6, 6 + lh),
                      (0, 0, 0), -1)
        cv2.putText(frame, tag[0], (w - tw - 12, 6 + lh - 6), font, scale,
                    tag[1], 2, cv2.LINE_AA)

    # 底部快捷键
    hint = "q:quit  s:shot  r:rec  m:mirror  SPACE:pause"
    (hw, _), _ = cv2.getTextSize(hint, font, scale * 0.85, 1)
    cv2.rectangle(frame, (0, h - lh - 10), (hw + 18, h), (0, 0, 0), -1)
    cv2.putText(frame, hint, (margin, h - 8), font, scale * 0.85,
                (170, 170, 170), 1, cv2.LINE_AA)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser(description="K230 实时监视窗口")
    ap.add_argument("--port", default=None, help="串口，默认自动发现")
    ap.add_argument("-w", "--width", type=int, default=DEFAULT_CFG["w"])
    ap.add_argument("-H", "--height", type=int, default=DEFAULT_CFG["h"])
    ap.add_argument("-q", "--quality", type=int, default=DEFAULT_CFG["q"],
                    help="JPEG 质量 30~90")
    ap.add_argument("--fmt", choices=["rgb565", "grayscale"], default="rgb565")
    ap.add_argument("--fps", type=int, default=0, help="板端限速，0=跑满")
    ap.add_argument("--record", action="store_true", help="启动即录像")
    ap.add_argument("--save-dir", default="captures", help="截图/录像目录")
    ap.add_argument("--no-window", action="store_true", help="无窗口，仅打印统计")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not find_port():
        print("没有找到 K230。当前串口：")
        print(describe_ports())
        print("\n请确认开发板 USB 已连接（VID:PID 应为 1209:abd1），"
              "并断开 VS Code 里的 CanMV 连接（串口是独占的）。")
        return 2

    # 让 kill / 关窗口也能走到 finally，把板子上的脚本停掉
    def _on_term(signum, frame):
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, _on_term)
    except Exception:
        pass

    cfg = {"w": args.width, "h": args.height, "q": args.quality,
           "fmt": args.fmt, "fps": args.fps,
           "stats": 10, "limit": 0}
    save_dir = Path(args.save_dir)

    def log(msg):
        print("[k230]", msg, file=sys.stderr)

    st = K230Stream(cfg=cfg, port=args.port, verbose=args.verbose, log=log)
    print("[*] 正在启动摄像头…")
    st.start(timeout=25)
    if not st.hello:
        print("[!] 板端没有回应，请检查供电/固件。")
        st.stop()
        return 3

    print("[*] 已连接：%s" % st.hello.get("version", "?"))
    print("[*] 分辨率 %sx%s，质量 %s" % (st.hello.get("w"), st.hello.get("h"), st.hello.get("q")))

    recorder = Recorder(save_dir, (args.width, args.height))
    rec_start = 0.0
    if args.record:
        p = recorder.start()
        print("[*] 开始录像 ->", p)
        rec_start = time.time()

    window = "K230 Camera"
    if not args.no_window:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, args.width, args.height)   # 初始 1:1，可再拖拽

    paused = False
    mirror = False
    last_idx = 0
    last_frame = None
    last_ts = 0.0
    stamps = deque(maxlen=30)
    fps_host = 0.0
    last_report = time.time()
    frame_count = 0
    shots = 0

    try:
        while True:
            jpeg, idx = st.wait_frame(last_idx, timeout=0.5)
            if jpeg is not None and idx != last_idx:
                last_idx = idx
                frame = decode(jpeg)
                if frame is None:
                    log("收到一帧但解码失败（长度 %d）" % len(jpeg))
                    continue
                now = time.time()
                if last_ts:
                    stamps.append(now - last_ts)
                last_ts = now
                last_frame = frame
                frame_count += 1
                if stamps:
                    avg = sum(stamps) / len(stamps)
                    fps_host = 1.0 / avg if avg > 0 else 0.0

                if recorder.writer is not None:
                    recorder.write(frame)

            if last_frame is None:
                if not args.no_window:
                    blank = np.zeros((args.height, args.width, 3), np.uint8)
                    cv2.putText(blank, "waiting for K230 ...", (30, args.height // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
                    cv2.imshow(window, blank)
                    if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                        break
                continue

            display = last_frame.copy()
            if args.fmt == "grayscale" and display.ndim == 3:
                display = cv2.cvtColor(cv2.cvtColor(display, cv2.COLOR_BGR2GRAY),
                                       cv2.COLOR_GRAY2BGR)
            if mirror:
                display = cv2.flip(display, 1)

            if not paused:
                info = {
                    "w": display.shape[1], "h": display.shape[0],
                    "fps": fps_host,
                    "seq": last_idx,
                    "kb": len(jpeg) / 1024.0 if jpeg else 0.0,
                    "lag_ms": st.frame_age() * 1000.0,
                    "board_fps": float(st.stats.get("fps", 0) or 0),
                    "enc": st.stats.get("enc_ms", "?"),
                    "heap_kb": float(st.stats.get("mem", 0) or 0) / 1024.0,
                    "rx_mb": (st.link.rx_bytes / 1048576.0) if st.link else 0.0,
                    "bad": (st.link.bad_packets if st.link else 0),
                }
                rec_secs = int(time.time() - rec_start) if rec_start else 0
                display = draw_overlay(display, info, paused, recorder.writer is not None, rec_secs)

            if args.no_window:
                now = time.time()
                if now - last_report >= 1.0:
                    last_report = now
                    print("[stat] fps=%.1f board=%s size=%.1fKB lag=%.0fms rx=%.1fMB bad=%d"
                          % (fps_host, st.stats.get("fps", "?"),
                             (len(jpeg) / 1024.0) if jpeg else 0.0,
                             st.frame_age() * 1000.0,
                             (st.link.rx_bytes / 1048576.0) if st.link else 0.0,
                             st.link.bad_packets if st.link else 0))
                time.sleep(0.005)
                continue

            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                save_dir.mkdir(parents=True, exist_ok=True)
                path = save_dir / time.strftime("shot_%Y%m%d_%H%M%S.png")
                cv2.imwrite(str(path), display)
                shots += 1
                print("[*] 截图 ->", path)
            elif key == ord("r"):
                if recorder.writer is None:
                    p = recorder.start(fps_host or 15.0)
                    rec_start = time.time()
                    print("[*] 开始录像 ->", p)
                else:
                    p, n = recorder.stop()
                    rec_start = 0.0
                    print("[*] 录像结束 -> %s（%d 帧）" % (p, n))
            elif key == ord("m"):
                mirror = not mirror
            elif key == ord(" "):
                paused = not paused
            elif not st.connected and not st.auto_restart:
                break
    except KeyboardInterrupt:
        print("\n[*] 收到 Ctrl-C，正在停止…")
    finally:
        if recorder.writer is not None:
            p, n = recorder.stop()
            print("[*] 录像已保存 -> %s（%d 帧）" % (p, n))
        st.stop()
        cv2.destroyAllWindows()
        print("[*] 共接收 %d 帧，截图 %d 张。" % (frame_count, shots))
        if st.logs:
            tail = [x for x in list(st.logs)[-4:]]
            if tail:
                print("[*] 板端最后日志：", " | ".join(tail))
    return 0


if __name__ == "__main__":
    sys.exit(main())
