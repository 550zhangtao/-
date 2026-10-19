#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
serve_mjpeg.py —— 把 K230 变成网络摄像头（HTTP MJPEG）
=====================================================

这样任何程序/浏览器都能拿到「摄像头」画面，后续做检测、推流、ROS 桥接都方便。

用法：
    python3 -m k230tool serve                 # http://127.0.0.1:8080
    python3 -m k230tool serve --port 9000 --host 0.0.0.0   # 局域网可访问

接口：
    /               网页预览
    /stream         MJPEG 流（multipart/x-mixed-replace）
    /snapshot.jpg   当前帧（单张 JPEG）
    /stats          JSON 状态（帧率、分辨率、丢包）

其它程序读取示例：
    cap = cv2.VideoCapture("http://127.0.0.1:8080/stream")
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .link import DEFAULT_CFG, K230Stream, describe_ports, find_port  # noqa: E402

PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>K230 摄像头</title>
<style>
  body{margin:0;background:#111;color:#ddd;font-family:system-ui,sans-serif;
       display:flex;flex-direction:column;align-items:center;justify-content:center;
       min-height:100vh}
  h1{font-size:16px;font-weight:500;letter-spacing:.1em;color:#8f8}
  img{max-width:96vw;max-height:82vh;border:1px solid #333;border-radius:6px;
      background:#000}
  .bar{margin-top:10px;font-size:12px;color:#777}
  a{color:#6cf}
</style></head>
<body>
  <h1>K230 CAMERA</h1>
  <img src="/stream" alt="stream">
  <div class="bar">/snapshot.jpg &nbsp;·&nbsp; /stats &nbsp;·&nbsp;
     <a href="/snapshot.jpg">单张</a></div>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "K230MJPEG/1.0"

    def log_message(self, fmt, *args):   # 静音访问日志
        pass

    # ------------------------------------------------------------ 路由
    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send_bytes(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/snapshot.jpg":
            jpeg, _ = self.server.stream.latest()
            if jpeg is None:
                self.send_error(503, "no frame yet")
                return
            self._send_bytes(jpeg, "image/jpeg",
                             extra={"Cache-Control": "no-store"})
        elif path == "/stats":
            body = json.dumps(self.server.stats_snapshot(),
                              ensure_ascii=False).encode("utf-8")
            self._send_bytes(body, "application/json; charset=utf-8")
        elif path == "/stream":
            self._stream_mjpeg()
        else:
            self.send_error(404, "not found")

    # ------------------------------------------------------------ 工具
    def _send_bytes(self, body: bytes, ctype: str, extra=None):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _stream_mjpeg(self):
        st = self.server.stream
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        idx = 0
        try:
            while not self.server.stopping.is_set():
                jpeg, idx = st.wait_frame(idx, timeout=2.0)
                if jpeg is None:
                    if not st.connected:
                        time.sleep(0.2)
                    continue
                head = (b"--frame\r\nContent-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n")
                self.wfile.write(head)
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.server.frames_sent += 1
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.close_connection = True


def main() -> int:
    ap = argparse.ArgumentParser(description="K230 -> HTTP MJPEG 网络摄像头")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址，0.0.0.0 为全网")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--serial", default=None, help="K230 串口，默认自动发现")
    ap.add_argument("-w", "--width", type=int, default=DEFAULT_CFG["w"])
    ap.add_argument("-H", "--height", type=int, default=DEFAULT_CFG["h"])
    ap.add_argument("-q", "--quality", type=int, default=DEFAULT_CFG["q"])
    ap.add_argument("--fmt", choices=["rgb565", "grayscale"], default="rgb565")
    ap.add_argument("--fps", type=int, default=0)
    args = ap.parse_args()

    if not find_port():
        print("没有找到 K230。当前串口：\n%s" % describe_ports(), file=sys.stderr)
        return 2

    cfg = {"w": args.width, "h": args.height, "q": args.quality,
           "fmt": args.fmt, "fps": args.fps, "stats": 10}
    print("[*] 启动 K230 摄像头…")
    stream = K230Stream(cfg=cfg, port=args.serial,
                        log=lambda m: print("[k230]", m, file=sys.stderr))
    stream.start(timeout=25)
    if not stream.hello:
        print("[!] 板端无响应", file=sys.stderr)
        stream.stop()
        return 3

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    httpd.stream = stream
    httpd.stopping = threading.Event()
    httpd.frames_sent = 0

    def stats_snapshot():
        jpeg, idx = stream.latest()
        return {
            "connected": stream.connected,
            "board": stream.hello,
            "board_stats": stream.stats,
            "frames_received": stream.frames_received,
            "latest_frame_bytes": len(jpeg) if jpeg else 0,
            "bad_packets": stream.link.bad_packets if stream.link else 0,
            "rx_bytes": stream.link.rx_bytes if stream.link else 0,
            "stream_frames_sent": httpd.frames_sent,
        }

    httpd.stats_snapshot = stats_snapshot

    def _on_term(signum, frame):
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, _on_term)
    except Exception:
        pass

    print("[*] 已就绪： http://%s:%d" % (args.host, args.port))
    print("    实时流: http://%s:%d/stream" % (args.host, args.port))
    print("    单张图: http://%s:%d/snapshot.jpg" % (args.host, args.port))
    print("    Ctrl-C 退出")
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\n[*] 正在关闭…")
    finally:
        httpd.stopping.set()
        httpd.shutdown()
        stream.stop()
        httpd.server_close()
        print("[*] 已停止，共转发 %d 帧给客户端" % httpd.frames_sent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
