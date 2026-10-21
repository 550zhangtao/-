#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
link.py —— 亚博智能 K230(CanMV) 与 PC 之间的串口链路库
================================================================

功能
----
* 自动发现板子：按 USB VID:PID `1209:abd1` 识别，找不到则探测 /dev/ttyACM*
* 通过 MicroPython raw REPL 下发脚本（**分块节流**，实测一次性灌入会丢字节）
* 自定义二进制帧协议收发：JPEG 帧 / 日志 / 统计 / 错误，带 CRC32 校验与重同步
* 断线自动恢复：USB 重枚举后波特率/端口号变化都能重新连上
* 板子状态异常（sensor 残留占用）时自动软复位

帧协议（板端 -> PC，v2）
------------------------
    包头 38 字节，全部 ASCII（固件会把 0x0A 改写成 0x0D 0x0A，二进制头会被打坏）
        magic(4)   = AA 55 4B 32
        type(1)    = F/JPEG  L/日志  S/统计  H/握手  E/错误
        seq(8)     = 8 位十六进制序号
        plen(8)    = 8 位十六进制，原始载荷长度
        wlen(8)    = 8 位十六进制，线上（转义+换行展开后）的载荷字节数
        crc(8)     = CRC32(原始载荷)
        '|'
    载荷：原始字节做 CR 双写转义（0x0D -> 0x0D 0x0D）后再写，
          PC 端 replace 两次即可还原（见 _try_parse）。

命令行自测：  python3 k230tool/link.py --info
"""

from __future__ import annotations

import argparse
import binascii
import os
import glob
import json
import re
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterator, Optional, Tuple

try:
    import serial
    import serial.tools.list_ports as list_ports
except ImportError:  # pragma: no cover
    sys.exit("缺少 pyserial，请先执行：pip3 install pyserial")

# ----------------------------------------------------------------- 常量
MAGIC = b"\xAA\x55K2"
# 包头 38 字节，全是 ASCII（固件会把 0x0A 改写成 0x0D 0x0A，纯 ASCII 头才安全）：
#   magic(4) | type(1) | seq(hex8) | plen(hex8) | wlen(hex8) | crc(hex8) | '|'
HDR_LEN = 38
MAX_PAYLOAD = 8 << 20

VID_PID = (0x1209, 0xABD1)  # Kendryte CanMV

T_FRAME, T_LOG, T_STATS, T_HELLO, T_ERR, T_TARGET = 1, 2, 3, 4, 5, 6
TYPE_NAME = {T_FRAME: "JPEG", T_LOG: "LOG", T_STATS: "STATS",
             T_HELLO: "HELLO", T_ERR: "ERROR", T_TARGET: "TARGET"}
TYPE_FROM_CHAR = {b"F": T_FRAME, b"L": T_LOG, b"S": T_STATS,
                  b"H": T_HELLO, b"E": T_ERR, b"T": T_TARGET}

ROOT = Path(__file__).resolve().parent.parent
BOARD_DIR = ROOT / "board"
BOARD_SCRIPT = BOARD_DIR / "stream_camera.py"          # 板端纯推流脚本
POSE_BOARD_SCRIPT = BOARD_DIR / "yolo_pose.py"         # 板端 YOLOv8-Pose 关键点脚本
DEFAULT_BAUD = 115200

# 板端默认参数。改这里即可改变整条链路的默认行为。
DEFAULT_CFG: Dict[str, object] = {
    "w": 640, "h": 480,        # 分辨率
    "q": 70,                   # JPEG 质量
    "fmt": "rgb565",           # rgb565 | grayscale
    "mirror": 0, "flip": 0,    # 镜像/翻转
    "fps": 0,                  # 限速上限，0 = 跑满
    "stats": 30,               # 每多少帧上报一次统计，0 = 关闭
    "limit": 0,                # 发完多少帧自动停，0 = 无限
    "sensor_id": 2,
    "uart_enable": 1, "uart_id": 3, "uart_tx": 32, "uart_rx": 33, "uart_baud": 115200,            # 亚博 K230 摄像头接在 CSI2
}

# YOLOv8-Pose 板端默认参数（会被写进 yolo_pose.py 顶部 CFG 行）
POSE_DEFAULT_CFG: Dict[str, object] = {
    "w": 640, "h": 480, "q": 70,
    "model": "/sdcard/models/pose_v3.kmodel",
    "labels": ["target"],
    "in_w": 320, "in_h": 320,
    "conf": 0.25, "cand_conf": 0.03,
    "far_refine": 0, "far_min": 100, "far_topk": 3, "far_max": 200,
    "low_refine": 1, "low_refine_topk": 2, "low_refine_mult": 7.0,
    "low_refine_max": 120, "low_refine_cooldown": 3, "dbg_save": 0,
    "nms": 0.45,
    "kp_num": 5, "kp_dim": 3,     # 4 角点 + 靶心
    "max_boxes": 20,
    "jpeg_every": 2, "full_every": 9999,
    "roi_pad": 0.25, "roi_min": 160, "roi_max": 256,
    "laser_enable": 1, "laser_every": 1,   # 激光光斑：每帧检测（board/laser_spot.py）
    # 固定标定下每 5 帧才复核一次；受限重捕默认关（仅 R2 代理打开）
    "laser_verify_every": 5, "laser_global": 0,
    "laser_budget_ms": 35, "laser_cooldown": 1,
    "fps": 0, "stats": 15, "limit": 0,
    "sensor_id": 2,
    "uart_enable": 1, "uart_id": 3, "uart_tx": 32, "uart_rx": 33, "uart_baud": 115200,
    "uart_laser": 1,
    "reset_on_exit": 1,
}

_CFG_LINE_RE = re.compile(r"^CFG = \{.*\}$", re.M)


class BoardError(RuntimeError):
    """板端通信/状态错误。"""


# ----------------------------------------------------------------- 数据包
@dataclass
class Packet:
    type: int
    seq: int
    payload: bytes

    @property
    def name(self) -> str:
        return TYPE_NAME.get(self.type, "UNKNOWN(%d)" % self.type)

    @property
    def text(self) -> str:
        return self.payload.decode("utf-8", "replace")

    def kv(self) -> Dict[str, str]:
        """把 "a=1 b=2" 形式的文本解析成字典。"""
        out: Dict[str, str] = {}
        for tok in self.text.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                out[k] = v
        return out


# ----------------------------------------------------------------- 端口发现
def list_canmv_ports() -> list:
    """按 VID:PID 精确识别 CanMV 设备。"""
    return [p.device for p in list_ports.comports()
            if (p.vid, p.pid) == VID_PID]


def list_acm_ports() -> list:
    return sorted(glob.glob("/dev/ttyACM*"))


def open_serial(dev: str, baudrate: int = DEFAULT_BAUD,
                timeout: float = 0.02, write_timeout: float = 3.0):
    """独占方式打开串口。

    exclusive=True 走 Linux 的 TIOCEXCL：同一时刻只允许一个进程持有，
    避免「两个监视程序同时读一条流」导致画面撕裂 / device disconnected。
    """
    try:
        return serial.Serial(dev, baudrate, timeout=timeout,
                             write_timeout=write_timeout, exclusive=True)
    except TypeError:  # 老版本 pyserial 没有 exclusive 参数
        return serial.Serial(dev, baudrate, timeout=timeout,
                             write_timeout=write_timeout)


def probe_repl(dev: str, baudrate: int = DEFAULT_BAUD,
               timeout: float = 0.3) -> bool:
    """打开串口并尝试进入 raw REPL，判断是不是一块活着的 MicroPython 板子。"""
    try:
        with open_serial(dev, baudrate, timeout=timeout,
                         write_timeout=1.0) as s:
            s.write(b"\x03")
            time.sleep(0.15)
            s.reset_input_buffer()
            s.write(b"\x01")
            t0 = time.monotonic()
            acc = b""
            while time.monotonic() - t0 < 1.0:
                chunk = s.read(256)
                if chunk:
                    acc += chunk
                    if b"raw REPL" in acc:
                        s.write(b"\x02")
                        return True
            try:
                s.write(b"\x02")
            except Exception:
                pass
    except Exception:
        return False
    return False



def port_holders(dev: str):
    """返回占用该串口的 [(pid, comm)]，用于识别 CanMV IDE 占用。"""
    try:
        out = subprocess.run(["fuser", dev], capture_output=True, text=True, timeout=2)
        pids = [t for t in out.stdout.split() if t.isdigit()]
    except Exception:
        return []
    holders = []
    for pid in pids:
        try:
            r = subprocess.run(["ps", "-o", "comm=", "-p", pid],
                               capture_output=True, text=True, timeout=2)
            holders.append((int(pid), r.stdout.strip()))
        except Exception:
            holders.append((int(pid), "?"))
    return holders


def canmv_backend_holding(dev: str) -> bool:
    return any("canmv" in comm.lower() for _pid, comm in port_holders(dev))


# ----------------------------------------------------------------- USB 拓扑
# K230 的 USB 是原生 CDC：machine.reset() 会让设备真正从总线上消失再出现。
# 下面这些函数不打开串口，因此可以在板端脚本运行、REPL 被占用时安全使用。
USB_RESET_IOCTL = 0x5514          # linux/usbdevice_fs.h: USBDEVFS_RESET


def _read_sysfs(path: str) -> Optional[str]:
    try:
        with open(path, "r") as fp:
            return fp.read().strip()
    except Exception:
        return None


def _usb_sysfs_from_tty(dev: str) -> Optional[str]:
    """由 /dev/ttyACM0 找到它所属的 USB 设备 sysfs 目录（如 .../1-9）。"""
    real = os.path.realpath("/sys/class/tty/%s/device" % os.path.basename(dev))
    p = Path(real)
    for parent in [p] + list(p.parents):
        if (parent / "idVendor").exists():
            return str(parent)
    return None


def list_canmv_usb_infos():
    """列出当前枚举到的 K230 USB 设备（VID:PID = 1209:abd1）。"""
    infos = []
    for d in sorted(glob.glob("/sys/bus/usb/devices/*")):
        if _read_sysfs(os.path.join(d, "idVendor")) != "1209":
            continue
        if _read_sysfs(os.path.join(d, "idProduct")) != "abd1":
            continue
        try:
            busnum = int(_read_sysfs(os.path.join(d, "busnum")) or 0)
            devnum = int(_read_sysfs(os.path.join(d, "devnum")) or 0)
        except ValueError:
            busnum = devnum = 0
        ttys = sorted(glob.glob(d + ":*/tty/ttyACM*"))
        infos.append({
            "sysfs": d,
            "bus": busnum,
            "dev": devnum,
            "vid": "1209",
            "pid": "abd1",
            "tty": ("/dev/" + os.path.basename(ttys[0])) if ttys else None,
        })
    return infos


def usb_device_info(dev: Optional[str] = None) -> Optional[dict]:
    """返回 K230 的 USB 拓扑信息；dev 给定时优先匹配该 tty。"""
    infos = list_canmv_usb_infos()
    if dev:
        d = _usb_sysfs_from_tty(dev)
        if d:
            for info in infos:
                if info["sysfs"] == d:
                    return info
    return infos[0] if infos else None


def usb_present(dev: Optional[str] = None) -> bool:
    return usb_device_info(dev) is not None


def usb_reset(dev: Optional[str] = None) -> dict:
    """对 K230 做一次 USB 端口复位（等价于“软件拔插”）。

    只复位 USB 外设/重新枚举，不会复位 SoC，也不碰 BOOT 引脚；
    板端固件若已经彻底卡死，仍然需要断电。需要 /dev/bus/usb/... 的写权限。
    """
    info = usb_device_info(dev)
    if not info:
        raise BoardError("没有找到 K230 USB 设备（1209:abd1）")
    if not info.get("bus") or not info.get("dev"):
        raise BoardError("读不到 USB busnum/devnum：%s" % info.get("sysfs"))
    node = "/dev/bus/usb/%03d/%03d" % (info["bus"], info["dev"])
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows/macOS 不支持
        raise BoardError("当前系统没有 fcntl，无法做 USB 软件拔插")
    try:
        fd = os.open(node, os.O_RDWR)
    except OSError as exc:
        raise BoardError(
            "打不开 %s（%s）。USB 软件拔插需要该设备节点的写权限；"
            "可安装 scripts/99-k230.rules 或临时用 sudo 运行。"
            % (node, exc))
    try:
        fcntl.ioctl(fd, USB_RESET_IOCTL)
    except OSError as exc:
        raise BoardError("USBDEVFS_RESET 失败（%s）：%s" % (node, exc))
    finally:
        try:
            os.close(fd)
        except Exception:
            pass
    return info


def find_port(verbose: bool = False) -> Optional[str]:
    """找到一块可用的 K230 串口，找不到返回 None。"""
    for dev in list_canmv_ports():
        try:
            open_serial(dev, DEFAULT_BAUD, timeout=0.1).close()
            if verbose:
                print("[link] 按 VID:PID 命中 %s" % dev)
            return dev
        except Exception as exc:
            if verbose:
                print("[link] %s 打不开：%s" % (dev, exc))
    for dev in list_acm_ports():
        if probe_repl(dev):
            if verbose:
                print("[link] 探测命中 %s" % dev)
            return dev
    return None


def describe_ports() -> str:
    lines = []
    for p in list_ports.comports():
        is_usb = bool(p.vid) or p.device.startswith(("/dev/ttyACM", "/dev/ttyUSB"))
        if not is_usb:          # 过滤掉一片没用的 /dev/ttyS*
            continue
        mark = "  <== K230" if (p.vid, p.pid) == VID_PID else ""
        lines.append("  %-18s VID:PID=%s:%s %s%s"
                     % (p.device,
                        "%04x" % (p.vid or 0), "%04x" % (p.pid or 0),
                        (p.product or ""), mark))
    return "\n".join(lines) if lines else "  (pyserial 没有枚举到任何串口)"


# ----------------------------------------------------------------- 脚本模板
def render_board_script(cfg: Optional[Dict[str, object]] = None,
                        template: Optional[Path] = None,
                        defaults: Optional[Dict[str, object]] = None) -> str:
    """读取板端模板并替换顶部的 CFG 行。"""
    path = Path(template) if template else BOARD_SCRIPT
    merged = dict(defaults if defaults is not None else DEFAULT_CFG)
    merged.update(cfg or {})
    src = path.read_text(encoding="utf-8")
    new_src, n = _CFG_LINE_RE.subn("CFG = " + repr(merged), src, count=1)
    if n != 1:
        raise BoardError("模板里找不到可替换的 CFG 行：%s" % path)
    return new_src


class K230Link:
    """一条到 K230 的串口链路（独占使用）。"""

    def __init__(self, port: Optional[str] = None,
                 baudrate: int = DEFAULT_BAUD,
                 read_timeout: float = 0.02,
                 verbose: bool = False):
        self.port = port
        self.baudrate = baudrate
        self.read_timeout = read_timeout
        self.verbose = verbose
        self.ser: Optional[serial.Serial] = None
        self._buf = bytearray()
        self.rx_bytes = 0
        self.bad_packets = 0

    # ---------------------------------------------------------- 生命周期
    def open(self, retries: int = 3) -> "K230Link":
        last: Optional[Exception] = None
        for attempt in range(retries):
            dev = self.port or find_port(verbose=self.verbose)
            if not dev:
                last = BoardError(
                    "没有找到 K230 串口。请检查：\n"
                    "  1) USB 线是否插在板子的 USB0(调试口) 上\n"
                    "  2) `lsusb` 里是否有 1209:abd1\n"
                    "  3) 是否被 CanMV 插件/IDE 占用了串口（需先断开）")
                time.sleep(0.6)
                continue
            # 串口独占：CanMV IDE 正在占用时直接报错，避免两边抢串口后
            # IDE 的 runScript 10s 超时、板端脚本被顶死
            if os.environ.get("K230_ALLOW_CONFLICT") != "1" and canmv_backend_holding(dev):
                raise BoardError(
                    "串口 %s 正被 CanMV IDE（canmv-backend）占用。\n"
                    "  请先在 VS Code 里执行 `CanMV: Disconnect Board`（或关闭 IDE 串口连接），\n"
                    "  再运行 k230tool；两者不能同时占用同一个串口。\n"
                    "  确需强制使用：K230_ALLOW_CONFLICT=1 ..." % dev)
            try:
                self.ser = open_serial(dev, self.baudrate,
                                       timeout=self.read_timeout)
            except Exception as exc:
                last = BoardError(
                    "打开 %s 失败：%s\n"
                    "如果提示 busy，说明串口被 CanMV 插件/IDE 占用，"
                    "请在 VS Code 里执行 `CanMV: Disconnect Board`。" % (dev, exc))
                time.sleep(0.6)
                continue
            self.port = dev
            self._buf = bytearray()
            if self.verbose:
                print("[link] 已连接 %s @%d" % (dev, self.baudrate))
            return self
        raise last or BoardError("连接 K230 失败")

    def close(self) -> None:
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
        self.ser = None

    def __enter__(self) -> "K230Link":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.stop_script()
        self.close()

    # ---------------------------------------------------------- 读写底层
    def _read_for(self, duration: float) -> bytes:
        assert self.ser is not None
        acc = b""
        t0 = time.monotonic()
        while time.monotonic() - t0 < duration:
            chunk = self.ser.read(8192)
            if chunk:
                acc += chunk
                self.rx_bytes += len(chunk)
        return acc

    def sync(self) -> None:
        """打断板上可能在跑的脚本并清空缓冲。"""
        assert self.ser is not None
        self.ser.reset_input_buffer()
        self.ser.write(b"\x03")
        time.sleep(0.25)
        self._read_for(0.15)
        self.ser.reset_input_buffer()

    def enter_raw(self, timeout: float = 3.0) -> None:
        assert self.ser is not None
        self.sync()
        self.ser.write(b"\x01")
        t0 = time.monotonic()
        acc = b""
        while time.monotonic() - t0 < timeout:
            chunk = self.ser.read(256)
            if chunk:
                acc += chunk
                if b"raw REPL" in acc:
                    return
        raise BoardError("进入 raw REPL 失败，收到：%r" % acc[:120])

    def exit_raw(self) -> None:
        if self.ser:
            try:
                self.ser.write(b"\x02")
                time.sleep(0.05)
            except Exception:
                pass

    # ---------------------------------------------------------- 下发脚本
    def _upload(self, src: str, chunk_size: int = 128,
                chunk_delay: float = 0.002) -> None:
        """分块发送源码。

        实测教训：一次性 write 几 KB 会让板端 UART RX 溢出、丢字节，
        表现为 "IndentationError: unexpected indent"。128B/2ms 稳定。
        """
        assert self.ser is not None
        data = src.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8")
        for i in range(0, len(data), chunk_size):
            self.ser.write(data[i:i + chunk_size])
            time.sleep(chunk_delay)
        self.ser.write(b"\x04")

    def exec_and_collect(self, src: str, timeout: float = 20.0
                         ) -> Tuple[str, str, bool]:
        """执行一段会正常结束的脚本，返回 (stdout, stderr, 是否成功)。"""
        assert self.ser is not None
        self.enter_raw()
        self._upload(src)
        buf = bytearray()
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            chunk = self.ser.read(8192)
            if chunk:
                buf += chunk
                self.rx_bytes += len(chunk)
                if bytes(buf[-8:]).find(b"\x04>") >= 0:
                    break
            elif buf.endswith(b">"):
                break
        raw = bytes(buf)
        self.exit_raw()
        if raw.startswith(b"OK"):
            parts = raw[2:].split(b"\x04", 2)
            out = parts[0].decode("utf-8", "replace")
            err = parts[1].decode("utf-8", "replace") if len(parts) > 1 else ""
            return out, err, True
        return "", raw.decode("utf-8", "replace"), False

    def mem_free(self, timeout: float = 8.0) -> Optional[int]:
        """读取板端 MicroPython 堆剩余字节数。"""
        out, _err, ok = self.exec_and_collect(
            "import gc\nprint(gc.mem_free())\n", timeout=timeout)
        if not ok:
            return None
        for tok in out.replace("\r", " ").replace("\n", " ").split():
            if tok.isdigit():
                return int(tok)
        return None

    def _ensure_board_deps(self) -> None:
        """把板端脚本依赖的模块推到 /sdcard（字节数不同才推，避免每次重写）。"""
        deps = [BOARD_DIR / "laser_spot.py"]
        for local in deps:
            if not local.exists():
                continue
            remote = "/sdcard/" + local.name
            try:
                self.enter_raw()
                out, _err, ok = self.exec_and_collect(
                    "import os\nprint(os.stat('%s')[6])" % remote, timeout=8)
                cur = -1
                for tok in out.replace("\r", " ").replace("\n", " ").split():
                    if tok.isdigit():
                        cur = int(tok)
                        break
            except Exception:
                cur = -1
            if cur == local.stat().st_size:
                continue
            from .push_file import push_one
            self.exit_raw()
            if not push_one(self, str(local), remote, verbose=False):
                raise BoardError("推送依赖失败：%s" % local)

    def start_stream(self, cfg: Optional[Dict[str, object]] = None,
                     timeout: float = 15.0,
                     template: Optional[Path] = None,
                     defaults: Optional[Dict[str, object]] = None) -> None:
        """下发板端脚本并确认已经开始工作（不阻塞后续读取）。"""
        assert self.ser is not None
        src = render_board_script(cfg, template, defaults)
        self._ensure_board_deps()
        self.enter_raw()
        self._upload(src)
        self._buf = bytearray()
        t0 = time.monotonic()
        acc = bytearray()
        while time.monotonic() - t0 < timeout:
            chunk = self.ser.read(4096)
            if not chunk:
                continue
            acc += chunk
            self.rx_bytes += len(chunk)
            head = bytes(acc[:64])
            if head.startswith(b"OK"):
                # OK 之后可能还有板端初始化日志，剩下的交给 read_packet
                self._buf += acc[2:]
                return
            if b"Traceback" in acc or b"Error" in acc:
                self.exit_raw()
                raise BoardError("板端脚本启动失败：\n%s"
                                 % bytes(acc).decode("utf-8", "replace")[:600])
        self.exit_raw()
        raise BoardError("下发推流脚本超时（%ds）" % timeout)

    # ---------------------------------------------------------- 收包
    def _try_parse(self) -> Optional[Packet]:
        """从缓冲里切出一个完整包；数据不够时返回 None。

        载荷在线上是被固件「cooked」过的（0x0A -> 0x0D 0x0A），板端已把原始
        0x0D 双写成 0x0D 0x0D，所以这里用两次 replace 就能无歧义还原：
            0x0D 0x0A  -> 0x0A   (固件插入的回车)
            0x0D 0x0D  -> 0x0D   (原始回车)
        """
        while True:
            j = self._buf.find(MAGIC)
            if j < 0:
                if len(self._buf) > 3:
                    del self._buf[:-3]
                return None
            if j > 0:
                del self._buf[:j]
            if len(self._buf) < HDR_LEN:
                return None

            ptype = TYPE_FROM_CHAR.get(bytes(self._buf[4:5]))
            if ptype is None:
                del self._buf[:4]
                self.bad_packets += 1
                continue
            if self._buf[37:38] != b"|":
                del self._buf[:4]
                self.bad_packets += 1
                continue
            try:
                seq = int(bytes(self._buf[5:13]), 16)
                plen = int(bytes(self._buf[13:21]), 16)
                wlen = int(bytes(self._buf[21:29]), 16)
                crc = int(bytes(self._buf[29:37]), 16)
            except ValueError:
                del self._buf[:4]
                self.bad_packets += 1
                continue

            if plen > MAX_PAYLOAD or wlen > MAX_PAYLOAD:
                del self._buf[:4]
                self.bad_packets += 1
                continue
            total = HDR_LEN + wlen
            if len(self._buf) < total:
                return None

            wire = bytes(self._buf[HDR_LEN:total])
            payload = wire.replace(b"\r\n", b"\n").replace(b"\r\r", b"\r")
            if len(payload) == plen and (binascii.crc32(payload) & 0xFFFFFFFF) == crc:
                del self._buf[:total]
                return Packet(ptype, seq, payload)

            # 假同步或数据损坏：跳过这个 magic 继续找
            del self._buf[:4]
            self.bad_packets += 1

    def read_packet(self, timeout: Optional[float] = 0.5) -> Optional[Packet]:
        """读取一个完整包；超时返回 None。timeout=None 表示一直等。"""
        assert self.ser is not None
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            pkt = self._try_parse()
            if pkt is not None:
                return pkt
            if deadline is not None and time.monotonic() >= deadline:
                return None
            chunk = self.ser.read(65536)
            if chunk:
                self._buf += chunk
                self.rx_bytes += len(chunk)

    # ---------------------------------------------------------- 停止 / 复位
    def stop_script(self, wait: float = 1.5) -> None:
        """Ctrl-C 打断板端脚本，让它自己跑 finally 释放摄像头。"""
        if not self.ser:
            return
        try:
            self.ser.write(b"\x03")
        except Exception:
            return
        t0 = time.monotonic()
        buf = bytearray()
        while time.monotonic() - t0 < wait:
            chunk = self.ser.read(4096)
            if chunk:
                buf += chunk
                if bytes(buf[-8:]).find(b"\x04>") >= 0:
                    break
        self.exit_raw()

    def interrupt_to_repl(self, timeout: float = 12.0, on_wait=None) -> None:
        """反复 Ctrl-C + Ctrl-A，把板端脚本打断并切回 raw REPL。

        与 stop_script() 的区别：
        * 不会只发一次 Ctrl-C 就放弃；
        * 板端脚本如果自己调了 machine.reset()（例如 CanMV 预览的 finally），
          会等 USB 重新枚举后重新进入 REPL；
        * 只有确实回不来才抛错，错误信息会区分“脚本卡死”和“USB 卡死”。
        """
        deadline = time.monotonic() + timeout
        last_exc: Optional[Exception] = None
        while time.monotonic() < deadline:
            if not self.ser or not self.ser.is_open:
                try:
                    self.open()
                except Exception as exc:
                    last_exc = exc
                    time.sleep(0.4)
                    continue
            try:
                self.ser.reset_input_buffer()
                for _ in range(4):
                    self.ser.write(b"\x03")
                    time.sleep(0.06)
                # Ctrl-A -> raw REPL
                self.ser.write(b"\x01")
                t0 = time.monotonic()
                acc = bytearray()
                while time.monotonic() - t0 < 1.2:
                    chunk = self.ser.read(4096)
                    if chunk:
                        acc += chunk
                        if b"raw REPL" in acc:
                            self._buf = bytearray()
                            return
                # 可能还在 friendly REPL / 刚打印完 traceback，再试一次
                self.ser.write(b"\x03")
                time.sleep(0.1)
                self.ser.write(b"\x01")
                t0 = time.monotonic()
                while time.monotonic() - t0 < 0.8:
                    chunk = self.ser.read(4096)
                    if chunk and b"raw REPL" in (bytes(acc) + chunk):
                        self._buf = bytearray()
                        return
            except (serial.SerialException, OSError, TypeError) as exc:
                last_exc = exc
                self.close()
                time.sleep(0.4)
                continue
            time.sleep(0.15)

        # 脚本可能刚刚自己复位：给它一次重新枚举的机会
        try:
            if on_wait:
                on_wait("Ctrl-C 没有立刻让出 REPL，等待板子重新枚举…")
            self.wait_for_repl(timeout=min(timeout, 12.0), settle=0.8,
                               on_wait=on_wait)
            return
        except Exception as exc:
            raise BoardError(
                "板端脚本 %.0f 秒内没有让出 REPL（%s）。\n"
                "  先再执行一次 `python3 -m k230tool reset`；\n"
                "  若 USB 还在但 Ctrl-C 完全无响应，说明脚本卡在原生调用里，\n"
                "  只能按板子 RESET 或断电重上电。"
                % (timeout, last_exc or exc))

    def wait_for_usb(self, timeout: float = 30.0, settle: float = 1.0,
                     on_wait=None, gone_timeout: float = 4.0) -> str:
        """等待 K230 的 USB 设备重新枚举并稳定下来（不读写串口）。

        关键点：复位后先等设备真正消失/换 devnum，再等 tty 出现并稳定
        `settle` 秒，最后才返回。不要在枚举过程中反复 probe_repl()：
        刚启动时发 Ctrl-C 会打断 boot.py/main.py，反而把板子留在半启动状态。
        """
        info0 = usb_device_info(self.port)
        old_dev = info0.get("dev") if info0 else None
        deadline = time.monotonic() + timeout

        # 阶段 1：等旧设备消失或 devnum 变化（最多 gone_timeout 秒）
        t0 = time.monotonic()
        while time.monotonic() - t0 < min(gone_timeout, timeout):
            infos = list_canmv_usb_infos()
            if not infos:
                break
            if old_dev and all(i.get("dev") != old_dev for i in infos):
                break
            time.sleep(0.1)

        # 阶段 2：等新 tty 出现并稳定
        stable_since: Optional[float] = None
        while time.monotonic() < deadline:
            ttys = [i["tty"] for i in list_canmv_usb_infos() if i.get("tty")]
            if ttys:
                if stable_since is None:
                    stable_since = time.monotonic()
                    if on_wait:
                        on_wait("USB 已重新枚举：%s" % ttys[-1])
                if time.monotonic() - stable_since >= settle:
                    self.port = ttys[-1]
                    return self.port
            else:
                stable_since = None
            time.sleep(0.1)

        raise BoardError(
            "复位后 %.0f 秒内 USB 没有重新枚举。\n"
            "  先看 dmesg / lsusb 是否有 1209:abd1；若 USB 设备在但 tty 没了，\n"
            "  可执行 `python3 -m k230tool reset`（内部会自动尝试 USB 软件拔插）。"
            % timeout)

    def wait_for_repl(self, timeout: float = 30.0, settle: float = 1.0,
                      on_wait=None) -> str:
        """等待 USB 重新枚举并确认 REPL 可用；成功时串口保持打开。"""
        self.close()
        deadline = time.monotonic() + timeout
        try:
            dev = self.wait_for_usb(timeout=timeout, settle=settle,
                                    on_wait=on_wait)
        except BoardError:
            dev = self.port
        while time.monotonic() < deadline:
            ports = list_acm_ports()
            if dev:
                ports = [dev] + [p for p in ports if p != dev]
            for port in ports:
                if probe_repl(port, timeout=0.25):
                    self.port = port
                    self.open()
                    if on_wait:
                        on_wait("REPL 已恢复：%s" % port)
                    return port
            time.sleep(0.3)
        raise BoardError(
            "复位后 USB 已枚举，但 REPL 在 %.0f 秒内没有应答。\n"
            "  如果 /sdcard/offline_off 不存在，可能是 /sdcard/main.py 又自动跑起来\n"
            "  并占用了 REPL；用 `python3 -m k230tool mode debug` 切回调试模式。\n"
            "  如果连 Ctrl-C 都完全无响应，只能按 RESET 或断电。"
            % timeout)

    def soft_reset(self, wait_repl: bool = True, timeout: float = 30.0,
                   settle: float = 1.0, on_wait=None) -> str:
        """真正的软复位：raw REPL 里执行 machine.reset()，等待 USB 重新枚举。

        wait_repl=True  : 调试模式；等 REPL 回来并保持串口打开。
        wait_repl=False : 工作模式；/sdcard/main.py 会占用 REPL，只等 USB/tty 稳定。
        """
        if not self.ser or not self.ser.is_open:
            self.open()
        try:
            self.enter_raw(timeout=2.0)
        except BoardError:
            self.interrupt_to_repl(timeout=min(8.0, timeout), on_wait=on_wait)

        # 先打印一个标记再复位：主机能确认命令确实被执行了。250ms 足够把
        # USB 缓冲刷出去，避免复位瞬间 USB 还在写导致枚举异常。
        src = ("import sys, time\n"
               "print('K230_RESET')\n"
               "sys.stdout.flush()\n"
               "time.sleep_ms(250)\n"
               "import machine\n"
               "machine.reset()\n")
        self._upload(src)
        t0 = time.monotonic()
        acc = bytearray()
        while time.monotonic() - t0 < 1.5:
            try:
                chunk = self.ser.read(4096)
            except Exception:
                break
            if chunk:
                acc += chunk
                if b"K230_RESET" in acc:
                    break
        # 不要 exit_raw：板子马上/已经复位，再写 \x02 只会污染新会话
        self.close()
        if on_wait:
            on_wait("已发送 machine.reset()，等待 USB 重新枚举…")
        if wait_repl:
            return self.wait_for_repl(timeout=timeout, settle=settle,
                                      on_wait=on_wait)
        return self.wait_for_usb(timeout=timeout, settle=settle,
                                 on_wait=on_wait)

    def hard_reset(self, timeout: float = 30.0, on_wait=None,
                   wait_repl: bool = True) -> str:
        """兼容旧接口的软复位；失败时尝试 USB 软件拔插兜底。

        wait_repl=False 用于工作模式（main.py 会自动运行并占用 REPL）。
        """
        try:
            if self.ser and self.ser.is_open:
                try:
                    self.interrupt_to_repl(timeout=8.0, on_wait=on_wait)
                except Exception:
                    pass
            return self.soft_reset(wait_repl=wait_repl, timeout=timeout,
                                   settle=1.0, on_wait=on_wait)
        except BoardError as first:
            # 枚举层卡死时，USBDEVFS_RESET 可以免物理拔插恢复 tty。
            # 注意：它不会复位 SoC；固件本身卡死仍然要断电。
            if not usb_present(self.port or None):
                raise
            if on_wait:
                on_wait("复位超时，尝试 USB 软件拔插…")
            try:
                usb_reset(self.port or None)
                if wait_repl:
                    return self.wait_for_repl(timeout=timeout, settle=1.0,
                                              on_wait=on_wait)
                return self.wait_for_usb(timeout=timeout, settle=1.0,
                                         on_wait=on_wait)
            except Exception as exc:
                raise BoardError("%s\n  USB 软件拔插也失败：%s" % (first, exc))


# ----------------------------------------------------------------- 线程化数据流
class K230Stream:
    """后台线程持续拉流，调用方随时取「最新一帧」。

    用法::

        with K230Stream(cfg={"w": 640, "h": 480}) as st:
            jpeg, idx = st.latest()
    """

    def __init__(self, cfg: Optional[Dict[str, object]] = None,
                 port: Optional[str] = None,
                 auto_restart: bool = True,
                 verbose: bool = False,
                 log=print,
                 fresh_start: bool = True,
                 min_heap: int = 1_500_000,
                 board_script: Optional[Path] = None,
                 defaults: Optional[Dict[str, object]] = None,
                 ai: bool = False):
        self.ai = bool(ai)
        if board_script:
            self.board_script = Path(board_script)
        else:
            self.board_script = POSE_BOARD_SCRIPT if self.ai else BOARD_SCRIPT
        if defaults is not None:
            self.defaults = dict(defaults)
        elif self.ai:
            self.defaults = dict(POSE_DEFAULT_CFG)
        else:
            self.defaults = dict(DEFAULT_CFG)
        self.cfg = dict(self.defaults)
        self.cfg.update(cfg or {})
        self.port = port
        self.auto_restart = auto_restart
        self.verbose = verbose
        self._log_fn = log
        # 板端堆低于 min_heap 时先软复位，否则推流会卡死在 snapshot()。
        # 实测：反复 exec 会把 MicroPython 堆吃到几十 KB，软复位后回到 ~4MB。
        self.fresh_start = fresh_start
        # AI 流要加载 kmodel + AI2D，堆低于 2.5 MB 先软复位更稳
        if self.ai:
            min_heap = max(min_heap, 2_500_000)
        self.min_heap = min_heap
        # 板端进入 "sensor already inited" 这类脏状态时只能软复位，
        # 单独 deinit() 清不掉（实测），这里记一个待复位标记。
        self._need_reset = False

        self.link: Optional[K230Link] = None
        self.connected = False
        self.hello: Dict[str, str] = {}
        self.stats: Dict[str, str] = {}
        self.target: Optional[dict] = None       # 最新一帧的检测结果
        self.target_seq = -1
        self.target_by_seq: Dict[int, dict] = {}
        self.frames_received = 0
        self.last_error: Optional[str] = None
        self.started_at = 0.0

        self._jpeg: Optional[bytes] = None
        self._frame_id = 0
        self._frame_ts = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.logs: Deque[str] = deque(maxlen=200)

    # ---------------------------------------------------------- 控制
    def start(self, timeout: float = 20.0) -> "K230Stream":
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._run, name="k230-stream",
                                        daemon=True)
        self._thread.start()
        # 等到真出第一帧（或超时）
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            with self._lock:
                if self._jpeg is not None:
                    return self
            time.sleep(0.05)
        self._log("警告：%ds 内没有收到图像帧：%s"
                  % (timeout, self.last_error or "板端未响应"))
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6.0)
        self._thread = None

    def __enter__(self) -> "K230Stream":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---------------------------------------------------------- 取帧
    def latest(self) -> Tuple[Optional[bytes], int]:
        with self._lock:
            return self._jpeg, self._frame_id

    def wait_frame(self, last_id: int, timeout: float = 2.0
                   ) -> Tuple[Optional[bytes], int]:
        """等到出现比 last_id 更新的帧。"""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            with self._lock:
                if self._frame_id != last_id:
                    return self._jpeg, self._frame_id
            time.sleep(0.002)
        return None, last_id

    def frame_age(self) -> float:
        with self._lock:
            return time.time() - self._frame_ts if self._frame_ts else 0.0

    def detections(self, frame_id: Optional[int] = None) -> Optional[dict]:
        """取某帧的检测结果。frame_id 为 None 时返回最新一份。"""
        with self._lock:
            if frame_id is None:
                return self.target
            return self.target_by_seq.get(frame_id) or self.target

    # ---------------------------------------------------------- 内部
    def _log(self, msg: str) -> None:
        self.logs.append(msg)
        if self._log_fn:
            self._log_fn(msg)

    def _start_board(self, link: "K230Link") -> None:
        """在板子上启动脚本（ai=True 时用 YOLOv8 推流脚本）。"""
        link.start_stream(self.cfg, template=self.board_script,
                          defaults=self.defaults)

    def _handle(self, pkt: Packet) -> None:
        if pkt.type == T_FRAME:
            with self._lock:
                self._jpeg = pkt.payload
                self._frame_id += 1
                self._frame_ts = time.time()
            self.frames_received += 1
        elif pkt.type == T_STATS:
            self.stats = pkt.kv()
        elif pkt.type == T_TARGET:
            text = pkt.text
            try:
                data = json.loads(text)
            except Exception:
                data = {"raw": text}
            if not isinstance(data, dict):
                data = {"det": data}
            seq = data.get("seq", pkt.seq)
            try:
                seq = int(seq)
            except Exception:
                seq = pkt.seq
            with self._lock:
                self.target = data
                self.target_seq = seq
                self.target_by_seq[seq] = data
                if len(self.target_by_seq) > 120:
                    for key in sorted(self.target_by_seq)[:-120]:
                        del self.target_by_seq[key]
        elif pkt.type == T_HELLO:
            self.hello = pkt.kv()
        elif pkt.type == T_LOG:
            self.logs.append(pkt.text)
            if self.verbose:
                print("[board]", pkt.text)
        elif pkt.type == T_ERR:
            raise BoardError("板端报错：%s" % pkt.text)

    def _session(self) -> None:
        link = K230Link(port=self.port, verbose=self.verbose)
        link.open()
        self.link = link
        try:
            if self._need_reset:
                self._log("板端状态需要清理，软复位…")
                link.hard_reset(on_wait=self._log)
                self.port = link.port
                self._need_reset = False
            if self.fresh_start:
                free = link.mem_free()
                if free is None or free < self.min_heap:
                    self._log("板端堆仅 %s 字节（阈值 %d），先软复位取干净环境…"
                              % (free, self.min_heap))
                    link.hard_reset(on_wait=self._log)
                    self.port = link.port
            try:
                self._start_board(link)
            except BoardError as exc:
                # sensor 残留占用等脏状态：软复位一次再试
                self._log("启动失败（%s），执行软复位后重试…" % exc)
                link.hard_reset(on_wait=self._log)
                self._start_board(link)
            self.connected = True
            self._log("已开始收流：%s" % (self.hello or ""))
            while not self._stop.is_set():
                pkt = link.read_packet(timeout=0.5)
                if pkt is None:
                    continue
                self._handle(pkt)
        except BoardError as exc:
            msg = str(exc)
            if "already inited" in msg or "should call reset" in msg:
                self._need_reset = True
            raise
        finally:
            self.connected = False
            try:
                link.stop_script()
            except Exception:
                pass
            link.close()
            self.link = None
            self.port = link.port

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._session()
            except Exception as exc:
                self.last_error = str(exc)
                self.connected = False
                if not self.auto_restart:
                    self._log("连接中断：%s" % exc)
                    return
                self._log("连接中断：%s；2 秒后重连…" % exc)
                self._stop.wait(2.0)
