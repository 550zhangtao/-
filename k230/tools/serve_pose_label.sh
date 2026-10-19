#!/usr/bin/env bash
# 启动本地标注服务（在项目根目录）
set -e
cd "$(dirname "$0")/../.."
echo "打开： http://127.0.0.1:8765/k230/tools/pose_label.html"
exec python3 -m http.server 8765 --bind 127.0.0.1
