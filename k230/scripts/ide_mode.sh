#!/usr/bin/env bash
# ide_mode.sh —— 兼容旧用法：切到 CanMV IDE 调试模式
#   等价于 scripts/mode.sh debug
#   等 REPL 真正恢复后才返回；不需要拔插 USB。
set -e
cd "$(dirname "$0")/.."
exec python3 -m k230tool mode debug "$@"
