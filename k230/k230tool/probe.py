#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe.py —— K230 全链路体检（连不上时先跑这个）
===============================================

按顺序检查：环境依赖 -> 串口发现 -> REPL 握手 -> 板端信息 -> 摄像头 ->
            串口带宽 -> 实际收流，每一步都给结论和修复建议。

用法：
    python3 -m k230tool probe            # 完整体检
    python3 -m k230tool probe --quick    # 跳过带宽/收流测试
    python3 -m k230tool probe --frames 60 --json out.json
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from pathlib import Path

from .link import (  # noqa: E402
    DEFAULT_CFG, K230Link, K230Stream, describe_ports, find_port, list_canmv_ports,
)

OK = "✓"
NG = "✗"
WARN = "!"


class Report:
    def __init__(self):
        self.items = []

    def add(self, name: str, ok: bool, detail: str = "", hint: str = ""):
        self.items.append({"name": name, "ok": ok,
                           "detail": detail, "hint": hint})
        mark = OK if ok else NG
        print(" %s %-14s %s" % (mark, name, detail))
        if not ok and hint:
            for line in hint.splitlines():
                print("      → %s" % line)
        return ok

    @property
    def all_ok(self):
        return all(i["ok"] for i in self.items)


BOARD_PROBE = """
import gc, sys
print("VERSION", sys.version)
print("MEM", gc.mem_free())
print("PLATFORM", sys.platform)
"""

SENSOR_PROBE = """
import time
s = None
try:
    from media.sensor import Sensor
    from media.media import MediaManager
    s = Sensor(id=2)
    s.reset()
    s.set_framesize(width=640, height=480)
    s.set_pixformat(Sensor.RGB565)
    MediaManager.init()
    s.run()
    time.sleep_ms(300)
    img = s.snapshot()
    print("SENSOR", s.get_type())
    print("FRAME", img.width(), img.height())
    try:
        s.stop()
    except Exception:
        pass
    try:
        MediaManager.deinit()
    except Exception:
        pass
    print("SENSOR_OK")
except Exception as e:
    print("SENSOR_ERR", type(e).__name__, e)
    try:
        s.stop()
    except Exception:
        pass
    try:
        MediaManager.deinit()
    except Exception:
        pass
"""


def parse_kv(text: str):
    out = {}
    for line in text.replace("\r", "\n").split("\n"):
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            out[parts[0]] = parts[1]
        elif len(parts) == 1:
            out[parts[0]] = ""
    return out


def probe_camera(link: K230Link):
    """跑一遍完整采集流程（init->run->snapshot->stop），返回 (ok, detail, chip)。"""
    out, _err, _ok = link.exec_and_collect(SENSOR_PROBE, timeout=25)
    sinfo = parse_kv(out)
    sensor_ok = ("SENSOR_OK" in out) or ("FRAME" in sinfo)
    chip = ""
    m = re.search(r"find sensor (\w+)", out)
    if m:
        chip = m.group(1)
    if "SENSOR_ERR" in sinfo:
        detail = sinfo["SENSOR_ERR"]
    elif "FRAME" in sinfo:
        detail = "%s 采集正常" % sinfo["FRAME"].replace(" ", "x")
    else:
        detail = "无响应"
    return sensor_ok, detail, chip


def measure_throughput(link: K230Link, nbytes: int = 262144) -> float:
    """让板子连发 nbytes 字节，返回 KiB/s（返回 0 表示失败）。"""
    link.enter_raw()
    src = ("import sys, time\n"
           "_R = sys.stdout.buffer\n"
           "_b = b'Q' * %d\n"
           "_R.write(_b)\n"
           "print()\n" % nbytes)
    link._upload(src)
    t0 = time.monotonic()
    got = 0
    while got < nbytes and time.monotonic() - t0 < 20:
        chunk = link.ser.read(65536)
        got += chunk.count(b"Q")
    dt = time.monotonic() - t0
    link.exit_raw()
    return (got / dt / 1024.0) if dt > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="K230 全链路体检")
    ap.add_argument("--port", default=None)
    ap.add_argument("--quick", action="store_true", help="跳过带宽/收流测试")
    ap.add_argument("--frames", type=int, default=30, help="收流测试帧数")
    ap.add_argument("--json", default=None, help="把结果写到 JSON 文件")
    args = ap.parse_args()

    rep = Report()
    print("=" * 62)
    print(" K230 摄像头链路体检   %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 62)

    # ---------------------------------------------------------- 1 环境
    print("\n[1/6] 运行环境")
    rep.add("Python", sys.version_info >= (3, 7),
            "%d.%d.%d (%s)" % (sys.version_info[0], sys.version_info[1],
                               sys.version_info[2], platform.system()))
    try:
        import serial
        rep.add("pyserial", True, "v%s" % serial.__version__)
    except ImportError:
        rep.add("pyserial", False, "未安装", "pip3 install pyserial")
    try:
        import cv2
        rep.add("opencv", True, "v%s" % cv2.__version__)
    except ImportError:
        rep.add("opencv", False, "未安装", "pip3 install opencv-python")
    try:
        import numpy
        rep.add("numpy", True, "v%s" % numpy.__version__)
    except ImportError:
        rep.add("numpy", False, "未安装", "pip3 install numpy")

    # ---------------------------------------------------------- 2 串口
    print("\n[2/6] 串口发现")
    ports = list_canmv_ports()
    print(describe_ports())
    port = args.port or find_port()
    if not rep.add("端口", bool(port), port or "没有可用串口",
                   "1) 确认 USB 插在板子的 USB0/调试口\n"
                   "2) lsusb 应能看到 1209:abd1\n"
                   "3) VS Code 的 CanMV 插件会独占串口，先 Disconnect Board"):
        _finish(rep, args)
        return 1
    if ports:
        rep.add("VID:PID", True, "1209:abd1 已识别")

    # ---------------------------------------------------------- 3 握手
    print("\n[3/6] REPL 握手 / 板端信息")
    link = K230Link(port=port)
    try:
        link.open()
    except Exception as exc:
        rep.add("打开串口", False, str(exc), "串口被占用时请断开 CanMV 插件")
        _finish(rep, args)
        return 1
    rep.add("打开串口", True, "%s @%d" % (link.port, link.baudrate))

    out, err, ok = link.exec_and_collect(BOARD_PROBE, timeout=15)
    info = parse_kv(out)
    rep.add("raw REPL", ok, (info.get("VERSION", "") or err.strip()[:80]))
    rep.add("固件", "CanMV" in info.get("VERSION", ""),
            info.get("VERSION", "?"))
    def _mem_ok(value):
        try:
            return int(value) > 1_000_000
        except (TypeError, ValueError):
            return False

    mem = info.get("MEM", "?")
    if not _mem_ok(mem):
        # 上一轮脚本没干净退出会吃掉堆，这里先自愈再看结果
        print("      (板端堆偏小，疑似脚本残留，先软复位…)")
        try:
            link.hard_reset(on_wait=lambda m: print("      [reset]", m))
            out, err, ok = link.exec_and_collect(BOARD_PROBE, timeout=15)
            info = parse_kv(out)
            mem = info.get("MEM", "?")
        except Exception as exc:
            print("      [reset] 失败：%s" % exc)
    rep.add("板端堆内存", _mem_ok(mem), "%s 字节" % mem,
            "堆太小说明有脚本残留，程序会自动软复位；也可手动拔插 USB")

    # ---------------------------------------------------------- 4 摄像头
    print("\n[4/6] 摄像头")
    sensor_ok, detail, chip = probe_camera(link)
    if not sensor_ok and "already inited" in detail:
        # 上一次脚本没干净退出时会残留占用，deinit() 清不掉，只能软复位
        print("      (检测到传感器残留占用，自动软复位后重试)")
        try:
            link.hard_reset(on_wait=lambda m: print("      [reset]", m))
            sensor_ok, detail, chip = probe_camera(link)
        except Exception as exc:
            detail = "软复位失败：%s" % exc
    rep.add("CSI2 传感器", sensor_ok,
            ("%s · %s" % (chip, detail)) if chip else detail,
            "亚博 K230 的摄像头接在 CSI2(id=2)；检查排线是否插紧")

    # ---------------------------------------------------------- 5 带宽
    print("\n[5/6] 串口带宽")
    if args.quick:
        print("  (--quick，跳过)")
    else:
        kbps = measure_throughput(link, 262144)
        rep.add("吞吐", kbps > 500, "%.0f KiB/s" % kbps,
                "带宽过低通常是 USB 线材/口的问题（劣质线会掉到几 KB/s）")
    link.close()

    # ---------------------------------------------------------- 6 收流
    print("\n[6/6] 实际收流")
    result = {}
    if args.quick:
        print("  (--quick，跳过)")
    else:
        try:
            import cv2
            import numpy as np
        except ImportError:
            cv2 = None

        st = K230Stream(cfg={"stats": 10, "limit": 0}, port=port,
                        log=lambda m: print("      [k230]", m))
        st.start(timeout=45)
        t0 = time.monotonic()          # 启动/复位不算进帧率
        if not st.hello:
            rep.add("收流", False, "板端无响应")
        else:
            n = 0
            last = None
            idx = 0
            while n < args.frames and time.monotonic() - t0 < 30:
                jpeg, idx = st.wait_frame(idx, timeout=3.0)
                if jpeg is not None:
                    n += 1
                    last = jpeg
            dt = time.monotonic() - t0
            fps = n / dt if dt else 0
            decoded = None
            if last and cv2 is not None:
                decoded = cv2.imdecode(np.frombuffer(last, np.uint8),
                                       cv2.IMREAD_COLOR)
            rep.add("收到帧", n > 0,
                    "%d 帧 / %.1fs = %.1f fps" % (n, dt, fps))
            rep.add("JPEG 解码", decoded is not None,
                    ("%dx%d" % (decoded.shape[1], decoded.shape[0]))
                    if decoded is not None else "无法解码")
            if last:
                rep.add("帧大小", True, "%.1f KB" % (len(last) / 1024.0))
            bad = st.link.bad_packets if st.link else 0
            rep.add("协议校验", bad == 0, "坏包 %d" % bad,
                    "坏包持续增长说明串口有干扰或固件输出被改写")
            result = {"fps": round(fps, 2),
                      "frames": n,
                      "frame_bytes": len(last) if last else 0,
                      "board_stats": st.stats,
                      "board": st.hello}
        st.stop()

    # ---------------------------------------------------------- 汇总
    print("\n" + "=" * 62)
    if rep.all_ok:
        print(" 结论：链路正常，可以开始用 k230tool monitor / serve / live")
        if result:
            print(" 实测：%.1f fps，每帧 %.1f KB，分辨率 %sx%s"
                  % (result.get("fps", 0),
                     result.get("frame_bytes", 0) / 1024.0,
                     result.get("board", {}).get("w", "?"),
                     result.get("board", {}).get("h", "?")))
    else:
        bad = [i["name"] for i in rep.items if not i["ok"]]
        print(" 结论：有问题 -> %s（详见上面的 → 提示）" % ", ".join(bad))
    print("=" * 62)

    _finish(rep, args, result)
    return 0 if rep.all_ok else 1


def _finish(rep: Report, args, extra=None):
    if not args.json:
        return
    payload = {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
               "ok": rep.all_ok, "checks": rep.items}
    if extra:
        payload["stream"] = extra
    Path(args.json).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[*] 结果已写入 %s" % args.json)


if __name__ == "__main__":
    sys.exit(main())
