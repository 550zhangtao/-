#!/usr/bin/env bash
# 仓库根目录快捷入口：转发到 k230/scripts/ide_mode.sh（= mode debug）
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$DIR/k230/scripts/ide_mode.sh" "$@"
