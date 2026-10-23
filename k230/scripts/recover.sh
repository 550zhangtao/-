#!/usr/bin/env bash
# recover.sh —— /sdcard 只读、REPL 卡死后的最小恢复（不需要写 sdcard）
#
#   bash scripts/recover.sh           # 等板子回来 -> 写 /data/boot_agent.py -> 启动代理
#   bash scripts/recover.sh --status  # 只看板子/代理状态
#
# 原理：板上 /sdcard/gimbal_agent.py 本来就在（只读也能读），只需要在 /data
# 写一个 50 字节引导文件，然后 exec 它。避免大文件推送把 raw REPL 拖挂。
set -e
cd "$(dirname "$0")/.."
exec python3 -m tools.lean_recover "$@"
