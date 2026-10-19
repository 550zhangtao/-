#!/usr/bin/env bash
# 后台启动实时桥接，日志写 live/live.log
#
#   bash scripts/run_live.sh                 # 默认模型（v2_std1.kmodel）
#   bash scripts/run_live.sh --conf 0.05     # 额外参数透传
#
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"
mkdir -p live

if [ -f live/live.pid ] && kill -0 "$(cat live/live.pid)" 2>/dev/null; then
  echo "live 已在运行（pid $(cat live/live.pid)）。先 kill 它或删 live/live.pid"
  exit 1
fi

nohup ./scripts/k230tool live --duration 0 "$@" >> live/live.log 2>&1 &
echo $! > live/live.pid
echo "live 已启动（pid $(cat live/live.pid)），日志 live/live.log"
echo "查看： ./scripts/k230tool status"
echo "停止： kill \$(cat live/live.pid)   或   touch live/STOP"
