#!/usr/bin/env bash
# 安装/打包辅助：装依赖 + 生成 ~/.local/bin/k230tool 包装脚本
#
#   bash scripts/install.sh            # 只装 Python 依赖 + 包装脚本
#   bash scripts/install.sh --udev     # 额外安装 udev 规则（需要 sudo）
#
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

echo "==> 目录：$DIR"
echo "==> 安装基础依赖（--user）"
python3 -m pip install --user -r requirements.txt

BIN="$HOME/.local/bin"
mkdir -p "$BIN"
cat > "$BIN/k230tool" <<EOF
#!/usr/bin/env bash
export PYTHONPATH="$DIR\${PYTHONPATH:+:\$PYTHONPATH}"
exec python3 -m k230tool "\$@"
EOF
chmod +x "$BIN/k230tool"
echo "==> 已生成 $BIN/k230tool"
case ":$PATH:" in
  *":$BIN:"*) echo "    PATH 已包含 $BIN" ;;
  *) echo "    提示：把 $BIN 加到 PATH，例如：echo 'export PATH=\$HOME/.local/bin:\$PATH' >> ~/.bashrc" ;;
esac

if [[ "${1:-}" == "--udev" ]]; then
  echo "==> 安装 udev 规则（需要 sudo）"
  sudo cp scripts/99-k230.rules /etc/udev/rules.d/
  sudo udevadm control --reload-rules
  sudo udevadm trigger
  echo "    重新插拔 USB 后可用 --port /dev/k230"
fi

echo
echo "完成。试试："
echo "  k230tool probe"
echo "  k230tool live"
