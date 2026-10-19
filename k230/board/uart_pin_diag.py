# -*- coding: utf-8 -*-
"""
K230 UART1 引脚诊断（YAHBOOM 4Pin: 3=GPIO9/TX, 4=GPIO10/RX, 2=GND）
=====================================================================
用万用表测 4Pin 第3脚对 GND：
  [GPIO LOW]  应 0V
  [GPIO HIGH] 应 3.3V
  [UART 0x55] 平均值应约 1.5~2.0V（不是恒定 3.3V）

如果 GPIO LOW 阶段仍是 3.3V：
  - 说明第3脚不是 GPIO9，或 4Pin 不是这个 UART 口，或线没接上。
如果 GPIO 能 0/3.3 切换，但 UART 阶段仍恒定 3.3V：
  - 说明 UART1 外设没有真正输出，需要检查 UART 初始化。
"""

import machine
import time

fp = machine.FPIOA()

# ---- 阶段 1：GPIO9 普通输出 ----
fp.set_function(9, fp.GPIO9, ie=0, oe=1)
out = machine.Pin(9, machine.Pin.OUT)
print("PHASE GPIO: 0V 2s -> 3.3V 2s -> 0V 2s ...")
for i in range(3):
    out.value(0)
    print("GPIO9 = 0  (measure ~0V)")
    time.sleep(2)
    out.value(1)
    print("GPIO9 = 1  (measure ~3.3V)")
    time.sleep(2)

# ---- 阶段 2：UART1 TX 连续发送 0x55 ----
fp.set_function(9, fp.UART1_TXD, ie=0, oe=1)
fp.set_function(10, fp.UART1_RXD, ie=1, oe=0)
uart = machine.UART(machine.UART.UART1, baudrate=115200,
                    bits=machine.UART.EIGHTBITS,
                    parity=machine.UART.PARITY_NONE,
                    stop=machine.UART.STOPBITS_ONE)
print("PHASE UART: send 0x55 x 300, measure pin3 average (~1.5-2V)")
for i in range(300):
    uart.write(b"\x55" * 32)
    uart.flush()
    time.sleep_ms(5)
print("PHASE UART done; TX should return to idle 3.3V")
print("DONE")
