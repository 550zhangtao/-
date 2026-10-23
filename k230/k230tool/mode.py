#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mode.py —— K230 调试模式 / 工作模式自由切换
==========================================

两种状态
--------
调试模式（debug）
    /sdcard/offline_off 存在；/sdcard/main.py 启动后立即退出，
    相机/KPU 不初始化，REPL 空闲。CanMV IDE 可以连板子、直接运行
    board/yolo_preview_ide.py 看识别效果。

工作模式（work）
    删除 /sdcard/offline_off；上电自动运行 /sdcard/main.py（常驻代理），
    ESP32 发 `$E,START!` 后开始轮询识别。此时 REPL 被脚本占用，
    **CanMV IDE 不能连接**，这是正常现象。

用法
----
    python3 -m k230tool mode debug      # 切到 IDE 调试（等 REPL 回来）
    python3 -m k230tool mode work       # 切到板端工作（等 USB 启动完成）
    python3 -m k230tool mode status     # 只读查看，不打断工作模式
    python3 -m k230tool mode reset      # 保持当前模式做一次真软复位
    python3 -m k230tool mode work --agent board/gimbal_agent.py

为什么不会“软复位后要拔插 USB”
--------------------------------
旧做法靠 Ctrl-C 打断工作脚本，再在 REPL 里 machine.reset()。实测：
* 工作脚本已经打开相机/KPU 时，Ctrl-C 会让 Python 退出并析构原生对象，
  可能直接死锁；狂发 Ctrl-C 更会把 K230 打进 USB 不枚举状态；
* 就算没死锁，工作脚本也会在复位后重新占住 REPL，看起来像“复位没反应”。

新版 board/gimbal_agent.py 复用 ESP32 已有的 `gnone` 命令：
PC 让 ESP32 连发 3 条 `$E,STOP!`（1.5s 内），K230 代理把它识别为调试暗号：
    写 /sdcard/offline_off -> machine.reset()
复位发生在 Python 对象析构之前，因此不会触发 KPU 死锁；复位后 main.py
看到 offline_off 直接退出，REPL 自然回来。全程不拔插、不按 Ctrl-C、
不需要改 ESP32 固件。

旧版代理没有这个暗号。首次升级需要一次性进入调试模式：
断电 -> 取出 TF 卡 -> 在根目录建空文件 `offline_off` -> 装回上电；
然后执行 `mode work --agent board/gimbal_agent.py` 部署新版。
（也可以试 `mode debug --force-ctrl-c`，但可能死锁，不推荐。）
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
import hashlib
import os
import sys
import time
from pathlib import Path

import serial

from .link import (
    BoardError,
    K230Link,
    describe_ports,
    find_port,
    list_acm_ports,
    list_canmv_usb_infos,
    open_serial,
    port_holders,
)

MARKER = "/sdcard/offline_off"
MAIN = "/sdcard/main.py"
MAIN_TMP = "/sdcard/main.py.new"

_STATE_SRC = """import os
try:
    print('OFFLINE', 1 if 'offline_off' in os.listdir('/sdcard') else 0)
except Exception as e:
    print('OFFLINE', -1, e)
try:
    print('MAIN', os.stat('/sdcard/main.py')[6])
except Exception:
    print('MAIN', -1)
try:
    print('GIMBAL', os.stat('/sdcard/gimbal_agent.py')[6])
except Exception:
    print('GIMBAL', -1)
"""


# ------------------------------------------------------------------ 小工具
def _log(msg: str) -> None:
    print("[mode] %s" % msg, flush=True)


def _open(port=None, verbose=False) -> K230Link:
    link = K230Link(port=port, verbose=verbose)
    link.open()
    return link


def _parse_state(text: str) -> dict:
    """把板端 _STATE_SRC 的输出解析成 dict。"""
    state = {"offline": None, "main": -1, "gimbal": -1}
    for line in (text or "").replace("\r", "\n").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        key = parts[0]
        try:
            value = int(parts[1])
        except ValueError:
            continue
        if key == "OFFLINE":
            state["offline"] = bool(value) if value >= 0 else None
        elif key == "MAIN":
            state["main"] = value
        elif key == "GIMBAL":
            state["gimbal"] = value
    return state


def read_state(link: K230Link) -> dict:
    out, err, ok = link.exec_and_collect(_STATE_SRC, timeout=10)
    if not ok:
        raise BoardError("读取板端状态失败：%s" % (err or out)[:300])
    return _parse_state(out)


def _write_marker(link: K230Link) -> None:
    out, err, ok = link.exec_and_collect(
        "f=open(%r,'w')\nf.write('1')\nf.close()\n"
        "import os\nprint('MARKER', os.stat(%r)[6])\n" % (MARKER, MARKER),
        timeout=10)
    if not ok or "MARKER 1" not in out:
        raise BoardError("写入 %s 失败：%s" % (MARKER, (err or out)[:300]))


def _remove_marker(link: K230Link) -> None:
    out, err, ok = link.exec_and_collect(
        "import os\n"
        "try:\n    os.remove(%r)\nexcept Exception as e:\n    print('RMERR', e)\n"
        "print('HAS', 'offline_off' in os.listdir('/sdcard'))\n" % MARKER,
        timeout=10)
    if not ok or "HAS False" not in out:
        raise BoardError("删除 %s 失败：%s" % (MARKER, (err or out)[:300]))


def _push_replace(link: K230Link, local: Path, remote: str, tmp: str,
                  label: str) -> None:
    """推 tmp -> md5 校验 -> 删旧 -> rename；任何一步失败都不动旧文件。"""
    from .push_file import push_one

    want = hashlib.md5(local.read_bytes()).hexdigest()
    _log("部署 %s（%d 字节, md5 %s）…" % (label, local.stat().st_size, want))
    if not push_one(link, str(local), tmp, verbose=True):
        raise BoardError("推送到 %s 失败" % tmp)
    got = link._board_md5(tmp)
    if got != want:
        raise BoardError("%s md5 校验失败：期望 %s，实际 %s"
                         % (tmp, want, got or "读取失败"))
    # 本机固件 os.rename 覆盖已存在文件会 EEXIST，必须先删再改名
    out, err, ok = link.exec_and_collect(
        "import os\n"
        "try:\n    os.remove(%r)\nexcept Exception as e:\n    print('RMERR', e)\n"
        "try:\n    os.rename(%r, %r)\nexcept Exception as e:\n    print('MVERR', e)\n"
        "try:\n    print('SZ', os.stat(%r)[6])\n"
        "except Exception as e:\n    print('SZ', -1)\n"
        % (remote, tmp, remote, remote),
        timeout=15)
    if not ok or ("SZ %d" % local.stat().st_size) not in out:
        # 假失败：切换工作模式时板子可能在替换后复位，REPL 没吐 SZ。
        # 重连后按 md5 再核对一次；文件对就继续，别吓人。
        verified = False
        for _ in range(3):
            time.sleep(1.5)
            try:
                link.close()
                link.open()
                if link._board_md5(remote) == want:
                    verified = True
                    break
            except Exception as e:
                _log("替换后复核失败：%s" % (e or type(e).__name__))
        if not verified:
            raise BoardError(
                "替换 %s 失败（期望 %d 字节）：%s\n"
                "  可先切 debug 后重试，或手工检查 %s"
                % (remote, local.stat().st_size, (err or out)[:500], tmp))
        _log("替换后 REPL 无输出，但 md5 一致 -> 视为成功（板子可能在切换中复位）")
    _log("已部署 %s（%d 字节, md5 %s）"
         % (remote, local.stat().st_size, got))


def _deploy_agent(link: K230Link, local_path: str) -> None:
    """把本地代理推到 /sdcard/main.py，md5 校验后替换。"""
    src = Path(local_path).expanduser().resolve()
    if not src.is_file():
        raise BoardError("--agent 文件不存在：%s" % src)
    _push_replace(link, src, MAIN, MAIN_TMP, "main.py")


def _read_boot_log(port: str, seconds: float = 2.5) -> str:
    """只读串口日志，不发任何字节（工作模式下不能 Ctrl-C）。"""
    if not port:
        return ""
    try:
        ser = open_serial(port, timeout=0.2)
    except Exception:
        try:
            ser = serial.Serial(port, 115200, timeout=0.2, exclusive=False)
        except Exception:
            return ""
    try:
        try:
            ser.dtr = False
            ser.rts = False
        except Exception:
            pass
        ser.reset_input_buffer()
        deadline = time.monotonic() + seconds
        buf = bytearray()
        while time.monotonic() < deadline:
            chunk = ser.read(4096)
            if chunk:
                buf += chunk
        return bytes(buf).decode("utf-8", "replace")
    finally:
        try:
            ser.close()
        except Exception:
            pass


def _repl_ready(link: K230Link, timeout: float = 1.0) -> bool:
    """不发 Ctrl-C，判断板子是否在 REPL（friendly 或 raw）。

    工作脚本运行时，这些控制字节只会进入它的输入缓冲（新代理会轮询并忽略），
    不会触发 KeyboardInterrupt，因此是安全的探测。
    """
    if not link.ser or not link.ser.is_open:
        return False
    try:
        link.ser.reset_input_buffer()
        # Ctrl-B: 若在 raw REPL 则退回 friendly；Ctrl-A: 进入 raw REPL
        link.ser.write(b"\x02\x01")
        t0 = time.monotonic()
        acc = bytearray()
        while time.monotonic() - t0 < timeout:
            chunk = link.ser.read(4096)
            if chunk:
                acc += chunk
                if b"raw REPL" in acc:
                    link._buf = bytearray()
                    return True
                if b">>>" in acc:
                    link.ser.write(b"\x01")
                    t0 = time.monotonic()
                    acc = bytearray()
                    continue
            time.sleep(0.05)
        return False
    except Exception:
        return False


def _gentle_ctrl_c(link: K230Link, wait: float = 2.0) -> bool:
    """旧代理的兜底：只发一次 Ctrl-C，等 2 秒；绝不连发。"""
    if not link.ser or not link.ser.is_open:
        return False
    try:
        link.ser.write(b"\x03")
    except Exception:
        return False
    time.sleep(wait)
    try:
        link.ser.read(8192)
    except Exception:
        pass
    return _repl_ready(link, timeout=2.0)


def _find_esp32_port() -> str:
    """找 ESP32 的 USB 串口（CH340/CH343: 1a86:7522）。"""
    try:
        from serial.tools import list_ports
        for p in list_ports.comports():
            if (p.vid, p.pid) == (0x1A86, 0x7522):
                return p.device
    except Exception:
        pass
    for dev in ("/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM1"):
        if os.path.exists(dev):
            return dev
    return ""


def _esp32_stop_burst(count: int = 3, wait_ready: float = 2.0) -> bool:
    """通过 ESP32 的 USB 命令 `gnone` 连发 STOP 给 K230。

    新 main.py 会把“1.5s 内 3 条 STOP”识别为切调试模式：写 offline_off
    后 machine.reset()。不需要 K230 REPL，也不需要改 ESP32 固件。
    """
    dev = _find_esp32_port()
    if not dev:
        _log("没有找到 ESP32 串口，无法用 UART 暗号")
        return False
    try:
        ser = serial.Serial(dev, 115200, timeout=0.1, write_timeout=1.0)
    except Exception as exc:
        _log("打不开 ESP32 %s：%s" % (dev, exc))
        return False
    try:
        try:
            ser.dtr = False
            ser.rts = False
        except Exception:
            pass
        # 打开 ESP32 串口通常会触发它复位；等它启动完再发命令
        ser.reset_input_buffer()
        t0 = time.monotonic()
        boot = bytearray()
        while time.monotonic() - t0 < wait_ready:
            boot += ser.read(4096)
            if b"ESP32_Gimbal_K230 ready" in boot:
                break
        _log("ESP32 %s 已就绪，发送 %d 条 STOP 暗号 …" % (dev, count))
        for _ in range(count):
            ser.write(b"gnone\n")
            time.sleep(0.12)
        time.sleep(0.4)
        out = bytes(boot + ser.read(16384)).decode("utf-8", "replace")
        sent = out.count("-> K230 $E,STOP!")
        _log("ESP32 已发出 STOP ×%d" % sent)
        return sent >= count
    except Exception as exc:
        _log("ESP32 暗号发送失败：%s" % exc)
        return False
    finally:
        try:
            ser.close()
        except Exception:
            pass


def _print_recovery() -> None:
    print("""
如果这次切换失败，按顺序试（都不需要拔插 USB）：
  1) python3 -m k230tool mode status
  2) 断电 -> 取出 TF 卡 -> 在根目录建空文件 offline_off -> 装回上电
     （一次性操作；进调试模式后执行 mode work --agent board/gimbal_agent.py
       部署带 3×STOP 暗号的新代理，以后切换就不需要再这样）
  3) 仍无响应：按板子 RESET 或断电重上电
""", flush=True)


# ------------------------------------------------------------------ 命令
def _finish_debug(link: K230Link, dev: str) -> int:
    state = read_state(link)
    if not state["offline"]:
        raise BoardError("复位后 %s 不见了，无法保证 IDE 模式" % MARKER)
    _log("调试模式就绪：%s" % dev)
    print("\n现在可以：打开 CanMV IDE -> 连接板子 -> 运行 "
          "k230/board/yolo_preview_ide.py\n"
          "（串口已空闲；IDE 运行时不要同时跑 k230tool live/monitor）",
          flush=True)
    return 0


def cmd_debug(args) -> int:
    port = args.port or find_port(verbose=args.verbose)
    if not port:
        _log("没有找到 K230 串口\n%s" % describe_ports())
        return 2
    link = _open(port, args.verbose)
    try:
        # 1) REPL 空闲：老路径，直接写标记 + 真软复位
        if _repl_ready(link):
            _log("REPL 空闲，直接切调试模式 …")
            _write_marker(link)
            state = read_state(link)
            _log("当前：offline=%s main.py=%s 字节"
                 % ("是" if state["offline"] else "否", state["main"]))
            _log("软复位并等待 USB 重新枚举（不拔插）…")
            dev = link.soft_reset(wait_repl=True, timeout=args.timeout,
                                  settle=1.0, on_wait=_log)
            return _finish_debug(link, dev)

        # 2) REPL 被工作脚本占用：走 ESP32 UART 暗号，不碰 Ctrl-C
        #    新版代理把 1.5s 内的 3 条 STOP 当成 DEBUG。不需要 REPL，
        #    也不需要改 ESP32 固件；gnone 本来就存在。
        _log("REPL 被工作脚本占用；尝试 ESP32 UART 暗号（gnone ×3 -> STOP ×3）…")
        if _esp32_stop_burst(3):
            _log("已发 3 条 STOP；等待 K230 复位（不探测、不发 Ctrl-C）…")
            try:
                link.wait_for_usb(timeout=min(12.0, args.timeout),
                                  settle=0.5, on_wait=_log)
            except BoardError as exc:
                raise BoardError(
                    "ESP32 已发 STOP ×3，但 K230 没有复位。\n"
                    "  说明当前 main.py 还是旧版（没有 _record_stop() 暗号）。\n"
                    "  一次性处理：断电 -> 取 TF 卡 -> 根目录建空文件 offline_off -> 上电；\n"
                    "  然后：python3 -m k230tool mode work --agent board/gimbal_agent.py\n"
                    "  原始错误：%s" % exc)
            dev = link.wait_for_repl(timeout=args.timeout, settle=1.0,
                                     on_wait=_log)
            return _finish_debug(link, dev)

        # 3) 旧代理兜底（显式 --force-ctrl-c，只发一次）
        if args.force_ctrl_c:
            _log("ESP32 暗号不可用，尝试一次 Ctrl-C（旧代理兜底）…")
            if _gentle_ctrl_c(link, wait=2.5):
                _write_marker(link)
                dev = link.soft_reset(wait_repl=True, timeout=args.timeout,
                                      settle=1.0, on_wait=_log)
                return _finish_debug(link, dev)

        raise BoardError(
            "工作脚本没有识别 ESP32 的 3×STOP 暗号。\n"
            "  说明当前 /sdcard/main.py 还是旧版。\n"
            "  一次性处理：断电 -> 取 TF 卡 -> 根目录建空文件 offline_off -> 上电；\n"
            "  然后运行：python3 -m k230tool mode work --agent board/gimbal_agent.py\n"
            "  （之后工作<->调试切换全程不需要拔插/按 Ctrl-C）")
    except BoardError as exc:
        _log("切换失败：%s" % exc)
        _print_recovery()
        return 1
    finally:
        link.close()


def cmd_work(args) -> int:
    port = args.port or find_port(verbose=args.verbose)
    if not port:
        _log("没有找到 K230 串口\n%s" % describe_ports())
        return 2
    link = _open(port, args.verbose)
    try:
        if not _repl_ready(link):
            # 已经处于工作模式：不要再复位一次
            if args.agent:
                raise BoardError(
                    "板子在工作模式，REPL 被占用，无法直接部署新 main.py。\n"
                    "  先运行 python3 -m k230tool mode debug（新版代理会识别 3×STOP），\n"
                    "  部署成功后再切回 work。")
            _log("已经在工作模式（REPL 被工作脚本占用）")
            print("CanMV IDE 不能连接，这是正常的。"
                  "切调试：python3 -m k230tool mode debug", flush=True)
            return 0

        # 调试模式 -> 工作模式
        if args.agent:
            _deploy_agent(link, args.agent)
        state = read_state(link)
        if state["main"] <= 0:
            raise BoardError(
                "/sdcard/main.py 不存在或为空；先部署代理：\n"
                "  python3 -m k230tool mode work --agent board/gimbal_agent.py")
        _remove_marker(link)
        _log("已删除 %s；main.py=%d 字节" % (MARKER, state["main"]))
        _log("软复位并等待 USB 重新枚举（不拔插）…")
        dev = link.soft_reset(wait_repl=False, timeout=args.timeout,
                              settle=1.2, on_wait=_log)
        log = _read_boot_log(dev, 3.0)
        if log.strip():
            _log("板端启动日志：%s"
                 % " / ".join(l.strip() for l in log.splitlines()
                                if l.strip())[-300:])
        else:
            _log("启动日志未捕获（正常：main.py 在 USB 重新枚举前已打印；"
                 "用 mode status 或 ESP32 status 确认）")
        _log("工作模式已启动（%s）：/sdcard/main.py 常驻，等 ESP32 的 START" % dev)
        print("\n此时 CanMV IDE 无法连接，这是正常的：REPL 被工作脚本占用。\n"
              "要回 IDE 预览：python3 -m k230tool mode debug",
              flush=True)
        return 0
    except BoardError as exc:
        _log("切换失败：%s" % exc)
        _print_recovery()
        return 1
    finally:
        link.close()


def cmd_reset(args) -> int:
    port = args.port or find_port(verbose=args.verbose)
    if not port:
        _log("没有找到 K230 串口\n%s" % describe_ports())
        return 2
    link = _open(port, args.verbose)
    try:
        if _repl_ready(link):
            state = read_state(link)
            debug = bool(state["offline"])
            _log("复位后保持%s模式" % ("调试" if debug else "工作"))
            dev = link.soft_reset(wait_repl=debug, timeout=args.timeout,
                                  settle=1.0, on_wait=_log)
            if debug:
                _log("软复位完成：%s，REPL 可用" % dev)
            else:
                _log("软复位完成：%s，/sdcard/main.py 已重新启动"
                     "（REPL 被占用，属正常）" % dev)
            return 0

        # 工作模式没有单独的“只复位不切模式”通道（3×STOP 已定义为切调试）。
        # 自动走 debug -> work 两步，效果等于“保持工作模式重启一次”。
        link.close()
        _log("工作模式；自动执行 debug -> work 两步复位（不拔插、不发 Ctrl-C）…")
        # 复位后端口号可能从 ACM0 变 ACM1；让下一步重新按 VID:PID 发现。
        # reset 是“保持原 main.py”，不要把 --agent 再部署一遍。
        args.port = None
        args.agent = None
        rc = cmd_debug(args)
        if rc != 0:
            return rc
        args.port = None
        return cmd_work(args)
    except BoardError as exc:
        _log("复位失败：%s" % exc)
        _print_recovery()
        return 1
    finally:
        link.close()


def _fmt_usb(infos) -> str:
    if not infos:
        return "未枚举到 1209:abd1"
    parts = []
    for i in infos:
        parts.append("%s bus=%s dev=%s tty=%s"
                     % (i.get("sysfs", "?"), i.get("bus", "?"),
                        i.get("dev", "?"), i.get("tty") or "(无)"))
    return " | ".join(parts)


def cmd_status(args) -> int:
    port = args.port or find_port(verbose=args.verbose)
    infos = list_canmv_usb_infos()
    print("USB：%s" % _fmt_usb(infos))
    print("串口：%s" % (sorted(list_acm_ports()) or "(无)"))
    if not port:
        if infos:
            print("状态：USB 在，但 tty 还没出现（可能正在启动/卡在枚举）")
        else:
            print("状态：板子未连接")
        return 2
    holders = port_holders(port)
    if holders:
        print("占用：%s" % ", ".join("%d(%s)" % h for h in holders))
    if holders and not args.probe:
        print("状态：串口被占用（大概率是 CanMV IDE，调试中）")
        return 0

    try:
        link = _open(port, args.verbose)
    except Exception as exc:
        print("状态：串口打不开（%s）" % exc)
        return 1
    try:
        if _repl_ready(link):
            state = read_state(link)
            print("模式：%s   main.py=%s 字节"
                  % ("调试" if state["offline"] else "工作",
                     state["main"]))
            return 0
        print("状态：工作模式（REPL 被工作脚本占用，IDE 不能连）")
        print("切调试：python3 -m k230tool mode debug")
        return 0
    finally:
        link.close()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="k230tool mode",
        description="K230 调试模式（CanMV IDE 可预览）/ 工作模式（板端轮询识别）切换")
    ap.add_argument("state", nargs="?", default="status",
                    choices=["debug", "work", "reset", "status"],
                    help="debug=IDE 调试；work=板端工作；reset=保持模式软复位；status=只读状态")
    ap.add_argument("--port", default=None)
    ap.add_argument("--agent", default=None,
                    help="work 时顺带部署这个本地文件到 /sdcard/main.py")
    ap.add_argument("--timeout", type=float, default=35.0,
                    help="等待 USB/REPL 恢复的秒数（默认 35）")
    ap.add_argument("--force-ctrl-c", action="store_true",
                    help="旧代理兜底：只发一次 Ctrl-C（可能死锁，默认不用）")
    ap.add_argument("--probe", action="store_true",
                    help="status 时兼容旧行为（保留参数，现在默认就会安全探测）")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.state == "debug":
        return cmd_debug(args)
    if args.state == "work":
        return cmd_work(args)
    if args.state == "reset":
        return cmd_reset(args)
    return cmd_status(args)


if __name__ == "__main__":
    sys.exit(main())
