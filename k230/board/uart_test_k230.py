# -*- coding: utf-8 -*-
"""
K230 UART1 通信测试（与 ESP32 云台主控）
=====================================
接线（任选其一）：
  A. 接 ESP32：K230 4Pin通讯口 第3脚 GPIO9(TX) -> ESP32 GPIO16(RX)
                K230 4Pin通讯口 第4脚 GPIO10(RX) <- ESP32 GPIO17(TX)
                GND 共地
  B. 仅测 K230 串口：用杜邦线把 GPIO3 与 GPIO4 短接（loopback）

运行：
  CanMV IDE 打开本文件直接运行；或
  python3 -m k230tool run board/uart_test_k230.py

预期：
  - 接 ESP32：每 200ms 发送一帧 $KT，收到 $E,ACK,<seq>,...!
  - loopback：收到自己发出的 $KT 原文
"""

import machine
import time

UART_ID = 3
TX_PIN = 32
RX_PIN = 33
BAUD = 115200

fpioa = machine.FPIOA()
if UART_ID == 3:
    fpioa.set_function(TX_PIN, fpioa.UART3_TXD, ie=0, oe=1)
    fpioa.set_function(RX_PIN, fpioa.UART3_RXD, ie=1, oe=0)
    unum = machine.UART.UART3
else:
    fpioa.set_function(TX_PIN, fpioa.UART1_TXD, ie=0, oe=1)
    fpioa.set_function(RX_PIN, fpioa.UART1_RXD, ie=1, oe=0)
    unum = machine.UART.UART1
uart = machine.UART(unum, baudrate=BAUD, bits=machine.UART.EIGHTBITS,
            parity=machine.UART.PARITY_NONE, stop=machine.UART.STOPBITS_ONE)

print("UART%d ready: TX=GPIO%d RX=GPIO%d %d" % (UART_ID, TX_PIN, RX_PIN, BAUD))
print("等待 ESP32 的 $E,ACK 或 loopback 回显...")

seq = 0
while True:
    seq += 1
    # 模拟目标：画面中心偏右 80px、偏下 40px，靶心同位置
    line = "$KT,%d,1,%d,%d,%d,%d,%d!\n" % (
        seq, 320 + 80, 240 + 40, 320 + 80, 240 + 40, 900)
    uart.write(line)
    try:
        uart.flush()
    except Exception:
        pass
    print("TX", line.strip())

    time.sleep_ms(50)
    if uart.any():
        data = uart.read()
        try:
            print("RX", data.decode("utf-8", "replace").strip())
        except Exception:
            print("RX", data)

    time.sleep_ms(150)
