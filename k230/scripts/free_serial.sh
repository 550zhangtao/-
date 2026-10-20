#!/usr/bin/env bash
# 释放 K230 串口：杀掉残留的 k230tool/live 进程，并列出占用者
#   bash scripts/free_serial.sh          # 只查看
#   bash scripts/free_serial.sh --kill   # 杀掉残留 python live 进程
set -u
PORT="${1:-/dev/ttyACM0}"
[ "${1:-}" = "--kill" ] && PORT="${2:-/dev/ttyACM0}"

echo "== $PORT 占用 =="
fuser -v "$PORT" 2>&1 || echo "(无人占用)"

echo
echo "== 残留 k230tool 进程 =="
pgrep -af "k230tool (live|monitor|serve|grab)" || echo "(无)"

if [ "${1:-}" = "--kill" ]; then
  echo
  echo "== 结束残留进程 =="
  pkill -f "k230tool (live|monitor|serve|grab)" && echo "已结束" || echo "无需结束"
  sleep 1
  echo
  echo "== 结束后占用 =="
  fuser -v "$PORT" 2>&1 || echo "(已释放)"
  echo
  echo "提示：板端脚本可能还在跑，执行  bash scripts/mode.sh debug  切回调试（会真软复位）"
fi
