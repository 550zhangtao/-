# 排错手册

## 1. 连不上板子

| 现象 | 检查 | 处理 |
| --- | --- | --- |
| `没有找到 K230 串口` | `lsusb \| grep 1209` | 确认 USB 插在板子的 **USB0（调试口）**，换一根数据线 |
| `lsusb` 有但 `/dev/ttyACM*` 没有 | `dmesg \| tail`（可能需要 sudo） | 重新插拔；装 `scripts/99-k230.rules`；确认 cdc_acm 驱动 |
| `打开 /dev/ttyACM0 失败: busy` | `fuser -v /dev/ttyACM0` | 串口被 CanMV IDE/VSCode 插件/另一个 k230tool 占用，先全部断开 |
| `Permission denied` | `ls -l /dev/ttyACM0` | 装 udev 规则，或 `sudo usermod -aG dialout $USER` 后重新登录 |
| 端口号在 ACM0/ACM1 之间跳 | — | 工具按 VID:PID `1209:abd1` 自动发现，**不要硬编码端口** |

## 2. 板端状态异常

### `sensor(2) is already inited`

上一次脚本没有干净释放摄像头。`MediaManager.deinit()` 清不掉，只能复位。

- 本工具：`K230Stream` 检测到后会自动 `hard_reset()` 重试；
- 手动：按板子 RESET，或执行

```python
import machine
machine.reset()
```

### 软复位后要拔插 USB / 工作模式切不回调试

**根因**：旧 `k230tool reset` 只发 Ctrl-C + Ctrl-D。Ctrl-D 只是 MicroPython
VM 软重启，不会清相机/KPU；工作模式下 `/sdcard/main.py` 会立刻重新占住
REPL，看起来就像“复位没反应”。如果直接 Ctrl-C 杀正在跑识别的脚本，
KPU/相机原生对象析构还可能死锁，把整块板卡到 USB 不枚举，只能拔插。

**现在的处理**：

```bash
bash scripts/mode.sh debug    # 工作 -> 调试（不用 Ctrl-C，不用拔插）
bash scripts/mode.sh work     # 调试 -> 工作
bash scripts/mode.sh status   # 只读查看
python3 -m k230tool reset     # 真 machine.reset + 等 USB 重新枚举
```

`mode debug` 让 ESP32 连发 3 条 `$E,STOP!`（`gnone`），新版
`board/gimbal_agent.py` 识别为调试暗号：写 `offline_off` 后立即
`machine.reset()`；复位发生在对象析构之前，不会死锁。
旧版 `main.py` 没有暗号时，会提示一次性用 TF 卡建 `offline_off` 升级。

**不要**再对正在跑识别的工作脚本连发 Ctrl-C。

### 跑完 AI 脚本后 REPL 无响应 / 写串口超时

K230 的 KPU 原生对象析构死锁（已知固件问题）。表现：脚本已经打印完，
但 PC 再发 `Ctrl-C`/`\x01` 没有任何回应，USB 设备还在。

- 本包 `board/yolo_stream.py`：退出时保活 KPU 引用并 `machine.reset()`，
  默认 `reset_on_exit=1`；
- 如果仍然卡住：先 `bash scripts/mode.sh debug`；仍无响应再按 RESET / 断电；
- 不要尝试在脚本里直接 `del kpu` / `kpu.release()` / `gc.collect()` 清理。

### 堆内存不足 / `snapshot()` 卡死

反复 exec 会吃 MicroPython 堆。工具在开流前检查 `gc.mem_free()`，
低于 1.5 MB 会自动软复位。也可以：

```python
import gc
gc.collect()
```

或直接 RESET。

## 3. 收不到画面 / 帧率低

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 0 帧 | 传感器残留、镜头排线、供电 | 先 `probe`；看板端日志；必要时复位 |
| 画面全黑 | 镜头盖/排线/曝光 | 摘镜头盖，检查 CSI 排线；`probe` 会报传感器型号 |
| 只有 2~3 fps | USB Hub/慢口、线材差 | 直插电脑 USB 口，换线 |
| 12 fps 左右 | 正常（640×480 JPEG 瓶颈） | 想快用 `-w 320 -h 240` 或 `--no-ai` |
| AI 只有 ~8 fps | JPEG 73 ms + 推理 28 ms | 降分辨率/质量，或只传检测结果不回传大图（需改板端脚本） |
| 坏包增长 | 干扰/线材/固件改写输出 | 换线；确认固件版本；`probe` 的协议校验 |

## 4. YOLOv8 相关

### 板端报 `OSError: sensor(2) is already inited`

见 2.1；另外确认没有两个 k230tool 同时跑。

### 模型加载失败 / 输出乱码

- kmodel 与固件/nncase 版本不匹配：用与固件同版本的 nncase 重新编译；
- 输出通道数不对：`libs/YOLO` 期望 `[1, 4+nc, 2100]`（320 输入）；
- 类别数/标签顺序不一致：`data.yaml`、`--labels`、kmodel 训练时一致。

### 检测不到标靶

1. 先看 `live/latest.jpg` 确认画面里真的有标靶、没被遮挡；
2. 降阈值：`--conf 0.05`；
3. 检查模型：`python3 -m k230tool models`，确认板端有 `pose_v3_256.kmodel`；
4. 标靶太小：靠近，或用更大输入（416/480）重训重编；
5. 训练数据与现场差异大：重新采集、训练。

### 框位置偏 / 误检

- 模型训练数据不足或场景差异大；
- kmodel 预处理不对：必须按 `input_range=[0,1]`、`mean=0`、`std=1` 编译，
  板端不要重复归一化；
- 坐标映射：`board/yolo_stream.py` 用 `mode="image"`，检测框和 JPEG 同为
  `aw×ah`；如果改了 `--aw/--ah` 与 `-w/-h` 不一致，PC 会按 `det` 里的
  `w/h` 缩放，仍然正确；
- 用 `tools/autolabel_target.py --preview` 看标注是否准确。

## 5. 训练 / 编译

| 报错 | 处理 |
| --- | --- |
| `ModuleNotFoundError: ultralytics` | `pip3 install ultralytics` |
| `Dataset 'data.yaml' images not found` | `data.yaml` 里 `path` 改成绝对路径；用 `dataset_from_captures.py` 生成 |
| `CUDA out of memory` | 降 `--batch` / `--imgsz`，或 `--device cpu` |
| `Failed to get hostfxr path` | 装 .NET 7，`export DOTNET_ROOT=$HOME/.dotnet` |
| `NNCASE_PLUGIN_PATH is not set` | 只影响 KPU 插件；`compile_kmodel.py` 会在 import 前自动设置 |
| `Feed Value Is Invalid, need u8[1,3,320,320]` | 校准数据形状应为 `[N,1,3,H,W]`；用本包工具 |
| `Unsupported operator` | ONNX 里有 K230 不支持的算子（NMS/后处理等）；用原始输出重新导出 |
| 编译成功但板端加载失败 | nncase 与固件版本不匹配；换同版本；或先跑官方示例 kmodel 验证固件 |
| kmodel 比预期大很多 | 没开 PTQ 或权重是 float；本工具默认 uint8 PTQ |

### 确认 ONNX 输出正确

```bash
python3 tools/export_onnx.py --weights best.pt --imgsz 320 --out best.onnx
# 应打印：output0 [1, 4+nc, 2100]
```

如果是 `[1,300,6]`，说明导出带了 NMS/end2end，板端不认。

## 6. live / Agent

| 现象 | 处理 |
| --- | --- |
| `latest.jpg` 不更新 | `python3 -m k230tool status` 看 `age_ms`/`connected`；看后台日志 |
| `status` 说 `port_ok:false` | 板子正在复位（AI 停止后约 8 秒）或串口被占用 |
| `snapshot` 很慢 | 没有 live 在跑，临时开流+复位约 15 秒；先启动 live 再 snapshot |
| `live` 退出后板子要等一会 | 正常：AI 脚本退出会 `machine.reset()` |
| 两个程序抢串口 | 串口独占；停掉一个 |

## 7. CanMV IDE 相关

- IDE 连不上：先停 `k230tool live/monitor`；如果板子在跑工作代理，
  执行 `bash scripts/mode.sh debug`（不要直接拔插）；
- IDE 运行 `yolo_preview_ide.py` 后断开：停止脚本会 `machine.reset()`，
  等 USB 重新枚举（约 2~3 秒）再重连即可；
- 工作模式下 IDE 连不上是正常的：REPL 被 `/sdcard/main.py` 占用，
  先 `mode.sh debug`；
- IDE 预览黑屏：确认 `DISPLAY_MODE="virt"` 且 `to_ide=True`；换 `"hdmi"` 试本地屏；
- 不要在 IDE 里同时 import 本包的 PC 端模块（那是给 PC 用的）。

### `AttributeError: type object 'Display' has no attribute 'JD9852'`

板端 `/sdcard/libs/PipeLine.py` / `DisplayConfig.py` 比固件新（固件 v1.4.3
没有 JD9852 这个屏幕驱动），官方 YOLO 例程用 `PipeLine` 会在这里报错。

处理：
- 用本包的 `board/yolo_preview_ide.py`（**不依赖 PipeLine**，直接
  Sensor + Display），已实测兼容 v1.4.3；
- 或者把 `/sdcard/libs` 换成与固件同版本的；
- 或者升级固件到与 libs 匹配的版本。

## 8. 一键收集诊断信息

```bash
python3 -m k230tool status
python3 -m k230tool probe --quick
python3 -m k230tool models
python3 tools/check_env.py
```

把这些输出贴出来，基本能定位 90% 的问题。

---

## 2026-10-23 事故：`/sdcard` 只读 + raw REPL 卡死（已复现三条）

表现：`mode.sh work` 部署失败（`OSError 30`）；之后板子 REPL 完全不响应
（115200 / 921600 / 1500000 都无输出），Ctrl-C/Ctrl-D 无效，只能硬件复位。

三个独立根因（按踩坑顺序）：

1. **`push_file` 12 KB 分块推 18 KB 文件会把 raw REPL 拖挂**。
   AGENTS 里记过">32KB 有概率写坏"，实测 12 KB 也会，且卡死后板子只能复位。
   → 现在用 `tools/recover_deploy.py` 的 **2 KB 分块 + 板上 md5 校验**；
2. **`/data/main.py` 这个文件名会把整块板子卡死**（`open(...,'wb')` 一执行就无响应，
   已复现两次，疑似固件 watcher 在 exec 半截文件）。**不要往 `/data` 写 `main.py`**；
3. **`/sdcard` 变只读**（`OSError 30`），`os.umount` 报 EINVAL、`os.mount` 报 ENODEV，
   软件层无法恢复。断电重上电仍是只读 → 需要重插 TF 卡；仍不行就备份后格 FAT32。

### 最小恢复（`/sdcard` 只读时也能用）

```bash
cd k230 && bash scripts/recover.sh          # 等板子 -> 写 /data/boot_agent.py -> 启动代理
bash scripts/recover.sh --status            # 只看状态
```

原理：板上 `/sdcard/gimbal_agent.py` 本来就在（只读也能读），只在 `/data` 写一个
50 字节引导文件 `exec(open('/sdcard/gimbal_agent.py').read())` 再执行。
**不写 `/sdcard`、不传大文件**，所以不会再拖挂 REPL。
实测代理起来后 `agent: 60 frames, 25.0fps` / `120 frames, 25.7fps`。

局限：这种方式**不持久**（复位后要再跑一次 `recover.sh`）。要让板上电自启，
必须修好 `/sdcard` 再执行 `bash scripts/mode.sh work --agent board/gimbal_agent.py`。
