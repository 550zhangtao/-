#!/usr/bin/env bash
# 清理调试/基准测试时留在板端 /sdcard 的临时脚本。
#   bash scripts/clean_board_tmp.sh
# 需要串口空闲（CanMV IDE 先 Disconnect Board）。
set -e
cd "$(dirname "$0")/.."
for f in probe_perf.py probe_api.py probe_fb.py bench_api.py bench_jpg.py \
         bench_enc2.py bench_enc3.py bench_final.py dbg_roi.py; do
  python3 - <<PY
from k230tool.link import K230Link
link = K230Link(); link.open()
try:
    out, err, ok = link.exec_and_collect(
        "try:\n    import os; os.remove('/sdcard/$f'); print('removed $f')\nexcept Exception as e:\n    print('skip $f', e)", timeout=10)
    print((out or '').strip().splitlines()[-1] if out else ('skip $f'))
finally:
    link.close()
PY
done
