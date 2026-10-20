#!/usr/bin/env bash
# offline_mode.sh —— 兼容旧用法：切到板端工作模式
#   等价于 scripts/mode.sh work
#   /sdcard/main.py 自动运行并占用 REPL，此时 CanMV IDE 不能连接。
set -e
cd "$(dirname "$0")/.."
exec python3 -m k230tool mode work "$@"
