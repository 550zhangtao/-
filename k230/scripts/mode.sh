#!/usr/bin/env bash
# mode.sh —— K230 调试 / 工作模式一键切换（不需要拔插 USB）
#
#   bash scripts/mode.sh debug    # IDE 调试：main.py 跳过，REPL 空闲
#   bash scripts/mode.sh work     # 板端工作：自动跑 /sdcard/main.py
#   bash scripts/mode.sh status   # 只读查看（不打断工作）
#   bash scripts/mode.sh reset    # 保持当前模式真软复位
#
# 可选：切工作时顺带部署代理
#   bash scripts/mode.sh work --agent board/gimbal_agent.py
set -e
cd "$(dirname "$0")/.."
exec python3 -m k230tool mode "$@"
