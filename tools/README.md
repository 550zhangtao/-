# tools/openmv_link.py — 不依赖 OpenMV IDE 的板端调试通道

`openmv_link.py` 通过 USB CDC 直接和板子的 MicroPython **raw REPL** 对话，
可以在不打开 OpenMV IDE 的情况下：

- 中断正在运行的 `main.py`、进入 REPL；
- 执行任意 Python 代码、读取变量；
- 读取/写入板载文件（例如部署 `main.py`）；
- 抓一帧 QVGA/VGA 图像回传到电脑（JPEG）。

## 依赖

```bash
pip install pyserial
```

## 使用前提

**先关闭 OpenMV IDE**（或断开其对串口的占用），否则会报
`could not open port /dev/ttyACM0: Device or resource busy`。

Linux 下串口通常是 `/dev/ttyACM0`，可用
`ls /dev/serial/by-id/` 确认。

## 示例

抓一帧并保存：

```python
import sys
sys.path.insert(0, "tools")
from openmv_link import OpenMV

omv = OpenMV("/dev/ttyACM0")
jpg = omv.capture_jpeg(quality=88, settle_ms=800)
open("frame.jpg", "wb").write(jpg)
omv.close()
```

部署脚本（会覆盖板上的 `main.py`）：

```python
omv = OpenMV()
omv.raw_repl()
omv.write_file("/main.py", open("firmware/1014_by_LinPeng/main.py", "rb").read())
omv.close()
```

执行一段代码并拿输出：

```python
out, err = omv.exec("import os\nprint(os.listdir('/'))\n")
print(out.decode())
```

## 注意

- `main.py` 只有在**硬复位/上电**时才会自动运行；REPL 里 `Ctrl-D` 软复位不会。
  想手动跑一遍：在 friendly REPL 里输入 `exec(open('/main.py').read())`，
  之后 `Ctrl-C` 可中断回 REPL（本工具的 `interrupt()`/`raw_repl()` 会做这件事）。
- 板子上的旧版本备份为 `/main_v101.py`，需要回退时直接复制回 `/main.py`。
- 抓图会短暂占用摄像头，抓完后板子停在 REPL；重新运行主程序即可恢复。
