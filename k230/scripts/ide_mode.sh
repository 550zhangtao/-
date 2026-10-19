#!/usr/bin/env bash
# ide_mode.sh —— K230 进入"IDE 预览模式"
#   板端建 /sdcard/offline_off 后复位：/sdcard/main.py（离线代理）启动即退出，
#   相机完全空闲，CanMV IDE / VSCode 插件可直接运行 yolo_preview_ide.py。
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
    "f=open('/sdcard/offline_off','w'); f.close(); print('IDE_MODE_ON')", timeout=10)
print((out or err).strip()[-80:])
try:
    link.exec_and_collect("import machine; machine.reset()", timeout=1)
except Exception:
    pass
link.close()
print("已切到 IDE 预览模式：在 CanMV IDE 里运行 /sdcard/yolo_preview_ide.py")
PYINNER
