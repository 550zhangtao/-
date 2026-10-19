# -*- coding: utf-8 -*-
"""
K230 / CanMV 板端摄像头推流脚本
================================

作用：从 GC2093 摄像头取流 -> JPEG 编码 -> 按二进制帧协议写到 USB CDC(REPL)，
      由 PC 端 k230tool/link.py 接收、解包、解码、显示或转发。

本文件由 PC 端作为「模板」读取：会就地替换下面这一行 CFG = {...} 后再下发执行，
所以 CFG 必须保持单行。也可以在 CanMV IDE 里直接运行（用默认参数）。

=========================================================================
固件行为记录（都是实测出来的坑，改代码前务必看一遍）
=========================================================================
1. `from media.* import *` 会把 sys 覆盖成固件内部的 mpp.sys（没有 stdout），
   所以标准库 import 必须放在 media 之后。
2. 传感器生命周期：reset() -> set_framesize/set_pixformat -> MediaManager.init()
   -> run()；stop() 之后要再 reset() 才能 run()。
3. `img.compress()` 返回的是 **Image 对象不是 bytes**，必须 `bytes()` 转换，
   否则 len()/crc32/切片写入全都不对，载荷会静默丢失。
4. **REPL 输出会把 0x0A 改写成 0x0D 0x0A**（cooked 输出，所有写通路都一样），
   所以二进制载荷必须先转义（本脚本把 0x0D 双写成 0x0D 0x0D），
   PC 端再用 replace 还原。否则 JPEG 每帧都被插入若干 0x0D 而 CRC 失败。
5. 包头必须是纯 ASCII：seq/plen 等字段一旦含 0x0A 就会被改写，头就解析不了。
6. 大块 write 不能被「先写头再写载荷」拆成两次，一次写完整个包最稳。
7. Ctrl-C 会中断主循环，finally 里务必 stop + deinit，保证下次能干净启动。
"""

# 会被 k230tool/link.py 用正则整行替换，务必保持单行
CFG = {'w': 640, 'h': 480, 'q': 70, 'fmt': 'rgb565', 'mirror': 0, 'flip': 0, 'fps': 0, 'stats': 30, 'limit': 0, 'sensor_id': 2}

# ！！！导入顺序不能改 ！！！（见文件头第 1 条）
from media.sensor import *
from media.media import *

import sys
import time
import gc
import binascii

# ---------------------------------------------------------------- 帧协议
# 包头 38 字节，全部 ASCII，保证不会被 0x0A -> 0x0D 0x0A 改写：
#   magic(4) | type(1) | seq(hex8) | plen(hex8) | wlen(hex8) | crc(hex8) | '|'
MAGIC = b"\xAA\x55K2"
HDR_LEN = 38
T_FRAME = 1
T_LOG = 2
T_STATS = 3
T_HELLO = 4
T_ERR = 5
TYPE_CHAR = {1: b"F", 2: b"L", 3: b"S", 4: b"H", 5: b"E"}

_RAW = sys.stdout.buffer if hasattr(sys.stdout, "buffer") else sys.stdout


def _write_all(data):
    """把 data 完整写出去（UART write 可能只写一部分，需要循环）。"""
    if not isinstance(data, (bytes, bytearray)):
        data = bytes(data)
    total = 0
    size = len(data)
    while total < size:
        chunk = data if total == 0 else data[total:]
        try:
            n = _RAW.write(chunk)
        except Exception:
            return False
        if n is None:
            return True
        if n <= 0:
            return False
        total += n
    return True


def emit(typename, seq, payload):
    """发一个数据包。

    载荷做 CR 转义后再写：固件会把每个 0x0A 变成 0x0D 0x0A，
    先把原来的 0x0D 双写成 0x0D 0x0D，PC 端就能无歧义还原。
    """
    if payload is None:
        payload = b""
    if not isinstance(payload, (bytes, bytearray)):
        payload = bytes(payload)
    payload = bytes(payload)

    esc = payload.replace(b"\r", b"\r\r")
    wlen = len(esc) + esc.count(b"\n")      # 线上实际字节数
    meta = "%08X%08X%08X%08X" % (seq & 0xFFFFFFFF, len(payload), wlen,
                                 binascii.crc32(payload) & 0xFFFFFFFF)
    hdr = MAGIC + TYPE_CHAR[typename] + meta.encode() + b"|"
    return _write_all(hdr + esc)


def log(msg):
    emit(T_LOG, 0, ("[k230] " + msg).encode())


# ---------------------------------------------------------------- 主流程
_sensor = None


def _open_sensor():
    """按固件要求完成 sensor 初始化，必要时清理上一次残留。"""
    global _sensor
    try:
        _sensor = Sensor(id=CFG["sensor_id"])
    except OSError:
        try:
            MediaManager.deinit()
        except Exception:
            pass
        time.sleep_ms(100)
        _sensor = Sensor(id=CFG["sensor_id"])

    _sensor.reset()
    _sensor.set_framesize(width=CFG["w"], height=CFG["h"])
    if CFG["fmt"] == "rgb565":
        _sensor.set_pixformat(Sensor.RGB565)
    else:
        _sensor.set_pixformat(Sensor.GRAYSCALE)
    if CFG["mirror"]:
        _sensor.set_hmirror(1)
    if CFG["flip"]:
        _sensor.set_vflip(1)
    MediaManager.init()
    _sensor.run()
    return _sensor


def _close_sensor():
    try:
        if _sensor:
            _sensor.stop()
    except Exception:
        pass
    try:
        MediaManager.deinit()
    except Exception:
        pass


def main():
    sen = _open_sensor()
    time.sleep_ms(200)

    try:
        stype = str(sen.get_type())
    except Exception:
        stype = "?"

    emit(T_HELLO, 0, ("w=%d h=%d fmt=%s q=%d fps=%d sensor=%s mem=%d version=%s"
                      % (CFG["w"], CFG["h"], CFG["fmt"], CFG["q"], CFG["fps"],
                         stype, gc.mem_free(), sys.version)).encode())
    log("streaming start %dx%d q=%d" % (CFG["w"], CFG["h"], CFG["q"]))

    seq = 0
    win_t = time.ticks_ms()
    win_n = 0
    target = CFG["fps"]
    every = CFG["stats"]

    while True:
        t0 = time.ticks_ms()
        img = sen.snapshot()
        t1 = time.ticks_ms()
        jpg = img.compress(quality=CFG["q"])
        if not isinstance(jpg, bytes):
            jpg = bytes(jpg)
        t2 = time.ticks_ms()

        seq += 1
        emit(T_FRAME, seq, jpg)

        win_n += 1
        snap_ms = time.ticks_diff(t1, t0)
        enc_ms = time.ticks_diff(t2, t1)

        if every and (seq % every == 0):
            now = time.ticks_ms()
            dt = time.ticks_diff(now, win_t)
            fps = 0.0 if dt <= 0 else (win_n * 1000.0 / dt)
            emit(T_STATS, seq, ("seq=%d fps=%.1f snap_ms=%d enc_ms=%d bytes=%d mem=%d"
                                % (seq, fps, snap_ms, enc_ms, len(jpg),
                                   gc.mem_free())).encode())
            win_t = now
            win_n = 0

        if CFG["limit"] and seq >= CFG["limit"]:
            log("limit %d reached" % CFG["limit"])
            break

        if (seq & 0x7F) == 0:
            gc.collect()

        if target:
            due = time.ticks_add(time.ticks_ms(), 1000 // target)
            delay = time.ticks_diff(due, time.ticks_ms())
            if delay > 0:
                time.sleep_ms(delay)


try:
    main()
except KeyboardInterrupt:
    log("interrupted by Ctrl-C")
except Exception as e:
    emit(T_ERR, 0, ("%s: %s" % (type(e).__name__, e)).encode())
    try:
        sys.print_exception(e)
    except Exception:
        print("ERROR", type(e).__name__, e)
finally:
    _close_sensor()
    log("stopped, sensor released")
