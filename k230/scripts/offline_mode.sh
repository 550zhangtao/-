#!/usr/bin/env bash
# offline_mode.sh —— 恢复"离线代理模式"（K230 上电自动跑 gimbal_agent）
#   删掉 /sdcard/offline_off 后复位，/sdcard/main.py 会重新自动运行。
set -e
cd "$(dirname "$0")/.."
python3 - <<'PYINNER'
import sys
sys.path.insert(0, ".")
from k230tool.link import K230Link
link = K230Link()
link.open()
link.stop_script(wait=1.0)
out, err, ok = link.exec_and_collect(
    "import os\ntry:\n    os.remove('/sdcard/offline_off')\nexcept Exception:\n    pass\nprint('OFFLINE_MODE_ON')",
    timeout=10)
print((out or err).strip()[-80:])
try:
    link.exec_and_collect("import machine; machine.reset()", timeout=1)
except Exception:
    pass
link.close()
print("已恢复离线代理模式：上电自动跑 /sdcard/main.py，等 ESP32 的 START")
PYINNER
