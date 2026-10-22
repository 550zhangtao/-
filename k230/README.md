# K230 USB 工具链 + YOLOv8-Pose 标靶识别（可移植包）

把亚博智能（YAHBOOM）K230 / CanMV 开发板当成一台 **USB 摄像头 + AI 相机**：
一根 USB 线，PC 上实时看画面、抓帧、跑 YOLOv8-Pose、把结果落成文件；
AI Agent 也能通过文件「看到」板子的实时画面。整个工具链都在本文件夹内，
拷到别的机器上只要装 Python 依赖即可。

> 纯 Python + pyserial + OpenCV，**不依赖 CanMV IDE 插件**。
> 板端脚本由工具自动下发，也可单独放进 CanMV IDE 运行。

---

## 1. 目录结构

```
k230/
├── k230tool/                  PC 端 Python 包（python3 -m k230tool …）
│   ├── link.py                核心：串口发现 / raw REPL / 帧协议 / 多线程收流 / 自动重连
│   ├── live.py                ★ 实时桥接：写 live/latest.jpg + latest.json
│   ├── decode.py              JPEG 解码、检测框、HUD
│   ├── monitor.py             OpenCV 窗口（原始画面）
│   ├── grab.py                批量抓帧（做数据集）
│   ├── serve_mjpeg.py         浏览器 MJPEG 网络摄像头
│   ├── probe.py               全链路体检
│   ├── mode.py                ★ 调试/工作模式切换（免拔插）
│   ├── push_file.py / pull_file.py
│   └── cli.py                 统一命令行
├── board/                     板端 MicroPython（会被自动下发）
│   ├── yolo_pose.py           ★ PC 推流：Pose 推理 + ROI JPEG + `$KT`
│   ├── yolo_preview_ide.py    ★ CanMV IDE 预览：框/四角/靶心
│   ├── gimbal_agent.py        离线常驻代理（ESP32 START/STOP 指挥）
│   ├── k230_perf.py           板端性能单点入口（CPU 频率 / 相机帧缓冲）
│   ├── stream_camera.py       采图 -> JPEG -> USB
│   └── uart_test_k230.py / uart_pin_diag.py   接线自检
├── tools/                     Pose 标注/训练/转换/评估
│   ├── pose_label.html        ★ 5 关键点浏览器标注工具
│   ├── build_pose_dataset_v3.py  小目标 + 硬负样本数据集生成
│   ├── train_pose_v3.py       Pose 训练（对 Pose 安全的增强配方）
│   ├── export_onnx.py         best.pt -> ONNX（nms=False）
│   ├── compile_kmodel.py      ONNX -> K230 kmodel（nncase + PTQ）
│   ├── kmodel_eval.py / bench_pose.py   精度评估
│   └── check_env.py           环境自检
├── models/                    当前 kmodel / pt（见 models/README.md）
├── scripts/                   mode.sh、udev 规则、离线部署
├── docs/                      部署/架构/排错/训练文档
├── live/                      运行时生成（latest.jpg / latest.json）
├── requirements.txt           基础依赖
└── requirements-ai.txt        训练/编译依赖
```

---

## 2. 移植 / 安装

```bash
# 1) 整个文件夹拷到目标机器
# 2) 基础依赖（只看画面/抓图）
pip3 install -r requirements.txt

# 3) 串口权限（可选；本机 /dev/ttyACM0 已是 0666 就不需要）
sudo cp scripts/99-k230.rules /etc/udev/rules.d/
sudo udevadm control --reload && sudo udevadm trigger
```

插上板子（USB0 调试口）后应看到：

```bash
lsusb | grep 1209        # 1209:abd1 Generic OpenMV Cam
ls /dev/ttyACM*          # /dev/ttyACM0
```

所有命令在 **本文件夹根目录** 执行：

```bash
python3 -m k230tool --help
```

> 不想每次 `cd` 进来？执行 `bash scripts/install.sh` 会建一个
> `~/.local/bin/k230tool` 包装脚本（加 `--udev` 才会用 sudo 装 udev 规则）。

---

## 3. 快速开始

```bash
# 体检：串口/REPL/传感器/带宽/收流
python3 -m k230tool probe

# 实时桥接（最常用）：持续写 live/latest.jpg + live/latest.json
python3 -m k230tool live

# 只取一张图（live 在跑时秒回；否则临时开一次流）
python3 -m k230tool snapshot -o live/snap.jpg

# OpenCV 窗口（q 退出，s 截图，r 录像）
python3 -m k230tool monitor

# 浏览器看流：http://127.0.0.1:8080
python3 -m k230tool serve --port 8080

# 抓 200 张图做数据集
python3 -m k230tool grab -n 200 -o captures

# 往板子推 kmodel / 脚本
python3 -m k230tool push models/pose_v3_384.kmodel --dest /sdcard/models/pose_v3_384.kmodel

# 从板子拉文件
python3 -m k230tool pull /sdcard/models/pose_v3_384.kmodel /tmp/x.kmodel
```

`live` 的常用参数：

```bash
# 图像更清晰（ROI 回传，按原生像素编码；90~92 是清晰/带宽的折中）
python3 -m k230tool live --jpeg-q 90

# 提高帧率：每 3 帧做一次推理 / 回传，省一半以上开销
python3 -m k230tool live --ai-every 3 --jpeg-every 3

# 只回传画面，不跑 AI
python3 -m k230tool live --no-ai

# 跑 60 秒后自动停（板子会自动复位）
python3 -m k230tool live --duration 60

# 板端 CPU 频率档位（默认 performance；不插 USB 供电时尤其重要）
python3 -m k230tool live --perf performance --perf-fb 3
```

---

## 4. 让 AI Agent 实时监测画面

`live` 在后台常驻，把最新一帧落盘：

| 文件 | 内容 |
| --- | --- |
| `live/latest.jpg` | 最新一帧，默认已画好检测框、时间戳、帧率、目标数 |
| `live/latest.json` | 序号、帧率、帧龄、检测框（类别/置信度/坐标）、靶心、关键点、板端统计 |
| `live/status.txt` | 一行文本：`seq=… fps=… targets=… age=…ms connected=…` |
| `live/history/` | 可选历史帧（`--history N`） |
| `live/STOP` | 创建这个文件即可让 live 优雅退出 |

Agent 侧只需要：

```python
# 看到画面
read_image("k230/live/latest.jpg")

# 拿到结构化检测结果
json.load(open("k230/live/latest.json"))
```

`latest.json` 示例：

```json
{
  "seq": 133,
  "fps": 18.4,
  "age_ms": 40.2,
  "connected": true,
  "size": [640, 480],
  "detections": [
    {"label": "target", "cls": 0, "conf": 0.71, "box": [160, 330, 167, 152]}
  ],
  "bull": {"x": 243.1, "y": 405.2, "conf": 0.92, "status": "ok"},
  "kpts": [180.2, 332.5, 0.96, ...],
  "infer_ms": 22
}
```

---

## 5. CanMV IDE 预览 + 调试/工作模式

### 5.1 调试模式 / 工作模式一键切换（不需要拔插 USB）

```bash
bash scripts/mode.sh debug    # 调试：/sdcard/main.py 跳过，REPL 空闲 -> IDE 可连
bash scripts/mode.sh work     # 工作：自动跑 /sdcard/main.py，REPL 被占 -> IDE 不能连
bash scripts/mode.sh status   # 只读查看当前模式（不会打断识别）
bash scripts/mode.sh reset    # 保持当前模式真软复位（machine.reset + 等 USB 回来）
```

- `debug`：写 `/sdcard/offline_off`，真软复位，等 REPL 回来后才返回；
- `work`：删 `offline_off`，真软复位，等 USB 启动完成；ESP32 按 R2/`gtrack`
  后开始轮询识别；
- **工作 → 调试** 不用 Ctrl-C 杀脚本（会把 K230 打进 USB 不枚举/死锁）：
  PC 让 ESP32 连发 3 条 `$E,STOP!`（`gnone`），代理识别为调试暗号：
  写 `offline_off` → `machine.reset()`。全程不拔插、不需要改 ESP32 固件；
- 首次升级 / 旧版 `main.py` 没有暗号时，需要一次性进入调试：
  断电 → 取 TF 卡 → 根目录建空文件 `offline_off` → 上电，然后执行
  `python3 -m k230tool mode work --agent board/gimbal_agent.py`；
- 别名：`scripts/ide_mode.sh` = `mode debug`，`scripts/offline_mode.sh` = `mode work`。

串口是 **独占** 的：`k230tool live`/`monitor` 在跑时，CanMV IDE 连不上；
反之亦然。两条路径二选一。

### 5.2 在 IDE 里看

1. `bash scripts/mode.sh debug`（等它打印“调试模式就绪”）
2. 打开 CanMV IDE，连接板子
3. 打开 `board/yolo_preview_ide.py`，按运行

画面：**检测框 + 四个角点轮廓 + 靶心红十字**。默认配置：

```python
MODEL = "/sdcard/models/pose_v3_384.kmodel"   # 384 输入，远小靶更准
VIEW = "full"                                 # 固定全图，画面稳定不跳动
VIEW_SCALE = 0.45                             # 288×216；0.55 -> 352×264
IDE_QUALITY = 70                              # 预览 JPEG 质量
CONF = 0.25
PERF_MODE = "performance"                     # 板端 CPU 频率（见 k230_perf.py）
SENSOR_FB = 3                                 # 相机帧缓冲
```

实现要点（踩过的坑见 `AGENTS.md`）：

- 板端 `aidemo` 没有 `yolov8_pose_postprocess`，预览脚本里用
  `libs.YOLO.aidemo = _PoseAidemo()` 注入 top-1 解码；
- IDE 预览**不初始化 Display**，直接用 `img.compress_for_ide()` 推帧；
- 采集顺序是**先拍显示帧、再拍 AI 帧**，避免框/关键点比画面晚一帧；
- 板端没有 FreeType 字体，默认不写字；结果打印在 IDE 终端：
  `[30] 19.0fps target 0.71 bull (243,405)`；
- 每 N 帧可把"画好的帧"存到 `/sdcard/pose_preview_dbg.jpg`
  （`DBG_SAVE`，默认 0），用 `k230tool pull` 拉回核对绘制是否正确。

> 停止脚本后板子会自动复位（KPU 析构死锁，复位最干净）。

---

## 6. YOLOv8-Pose 标靶关键点（当前路线）

### 6.1 路线

模型输出 **检测框 + 5 个关键点**：

| 序号 | 关键点 | 用途 |
| --- | --- | --- |
| 1 | TL 左上角 | 单应矫正 / 角点校验 |
| 2 | TR 右上角 | 同上 |
| 3 | BR 右下角 | 同上 |
| 4 | BL 左下角 | 同上 |
| 5 | 白色 10 号靶心 | 直接输出靶心坐标 |

- 训练：`tools/train_pose_v3.py`
- 标注：`docs/POSE_ANNOTATION.md`，工具 `tools/pose_label.html`
- 数据集：`tools/build_pose_dataset_v3.py`
- 评估：`tools/bench_pose.py` / `tools/kmodel_eval.py`
- 板端推流：`board/yolo_pose.py`
- CanMV 预览：`board/yolo_preview_ide.py`

### 6.2 板端模型

| 模型 | 说明 |
| --- | --- |
| `pose_v3_384.kmodel` | **当前唯一部署**：5 关键点 Pose（384 输入，Q1） |
| `pose_v3.pt` | 对应的 PyTorch 权重（重训 / 重导出起点） |

Pose kmodel 输出 `[1, 20, 3024]`，即 4+nc+3×5（nc=1）。

### 6.3 跑 Pose

```bash
# PC 端看结果（框 + 四角轮廓 + 靶心）
python3 -m k230tool live --conf 0.25

# 图像更清晰
python3 -m k230tool live --jpeg-q 90
```

`latest.json` 里：`detections`（框）、`kpts`（5 点扁平数组）、
`bull`（靶心 `{x,y,conf,status}`）。

### 6.4 性能（板端实测，2026-10-22，CanMV v1.4.3 / GC2093）

| 项目 | 数值 | 说明 |
| --- | --- | --- |
| 双通道采集 | 57 fps | 不是瓶颈 |
| JPEG 编码 640×480 q70 | 65~70 ms | **真正的瓶颈** |
| JPEG 编码 512×384 q92 | 58 ms | 同样帧率下比 640×480 q70 清晰 |
| Pose 384 推理 | 40~43 ms | `ai_every=2` 时摊薄一半 |
| 端到端 `live`（384 + ROI） | 约 11 fps（PC 侧） | 板端 20~26 fps |
| 纯推流（`live --no-ai`） | 11.6 fps | 也是编码卡住（`enc_ms=69`） |

**怎么提速（按性价比排序）**：

1. **降回传像素数**：`ROI_*`（有目标时）或 `--enc-scale`（无目标全图）。
   信息量 = 回传像素数，比 JPEG 质量档位更关键；
2. `--ai-every` / `--jpeg-every` 隔帧：推理和编码各减一半（`$KT` 仍按 AI 帧发）；
3. `--jpeg-q`：清晰度与带宽的小幅折中（512×384 下 88~92 差别不大）。

**别做的事**：

- 指望"把 CPU 拉满"——本固件没有调频接口（`os` 只有 `cpu_usage`），
  `board/k230_perf.py` 会打印 `cpu=auto`；
- 指望 `set_framebuffers`——固件直接打印 not support，代码已 try 包住；
- **不插 USB 就没有回传**：离线代理只走 UART 给 ESP32。

### 6.4.1 首次部署 / 换模型后要做的事

PC 下发的 `yolo_pose.py` 会带 `CFG`，但它依赖板端 `/sdcard/k230_perf.py`
（统一编码口径）。**没有这个文件时识别照常跑，只是少了缩放/质量优化**，
`hello` 日志里会出现 `perf=OFF(需要把 board/k230_perf.py 推到 /sdcard…)`。

```bash
cd k230
# 1) 推编码模块（一次性）
python3 -m k230tool push board/k230_perf.py --dest /sdcard/k230_perf.py
# 2) 推模型（换模型时）
python3 -m k230tool push models/pose_v3_384.kmodel --dest /sdcard/models/pose_v3_384.kmodel
# 3) 复位后跑一次，确认 hello 里有 enc=0.8/q92
python3 -m k230tool reset
python3 -m k230tool live --duration 10
```

### 6.5 离线云台代理（不插 USB，R2 触发识别）

`board/gimbal_agent.py` 是给 **ESP32 云台主控**用的常驻代理：

```bash
python3 -m k230tool mode work --agent board/gimbal_agent.py
```

- CanMV 上电自动执行 `/sdcard/main.py`，**不需要 USB 数据线**；
- 上电先常驻等命令；ESP32 进跟踪（手柄 R2 / USB `gtrack`）→ `$E,START!`
  → 代理开始逐帧推理并回传 `$KT,...!`；离开跟踪 → `$E,STOP!` 停止推理；
- 跟踪中 ESP32 2.5s 收不到 `$KT` → 每 3s 重发 START；K230 重启会发
  `$E,K230,READY!`，ESP32 收到且当前在 TRACK 就补发 START；
- 回执：`$E,K230,READY! / START_OK! / STOP_OK! / ERR,INIT!`；
- **回 IDE 调试**：`bash scripts/mode.sh debug`。

---

## 7. 协议速览

板端 -> PC，每包 38 字节 ASCII 包头 + 载荷：

```
MAGIC(AA 55 4B 32) | 类型(1B) | seq(hex8) | plen(hex8) | wlen(hex8) | crc(hex8) | '|' | 载荷
类型：F=JPEG帧  T=检测JSON  L=日志  S=统计  H=握手  E=错误
```

K230 -> ESP32（UART3，115200）：

```
$KT,<seq>,<run>,<cx>,<cy>,<bx>,<by>,<conf>!
    cx,cy = 检测框中心；bx,by = 靶心；run=0 = 本帧无目标
$E,START! / $E,STOP! / $E,K230,READY!
```

- 串口会把 `0x0A` 改写成 `0x0D 0x0A`，板端把原始 `0x0D` 双写、包头纯 ASCII，
  PC 端两次 `replace` 还原，二进制零损伤。
- `T` 包在 `F` 包之前发送，PC 端按 `seq` 配对，画框时不会错帧。

详细设计见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)、
[`docs/K230_ESP32_UART.md`](docs/K230_ESP32_UART.md)。

---

## 8. 排错

| 现象 | 处理 |
| --- | --- |
| 找不到串口 | USB 插板子 **USB0 调试口**；`lsusb` 应有 `1209:abd1` |
| `busy` / 打开失败 | 串口被 CanMV IDE/插件占用 → 先断开/关闭 IDE |
| `sensor(2) is already inited` | 上次脚本没干净退出 → 本工具会自动软复位；也可手动 RESET |
| 跑完 AI 脚本后板子无响应 | K230 KPU 析构死锁（已知）→ 按 RESET |
| 画面全黑 | 镜头盖/排线；`probe` 会报传感器型号 |
| 帧率低 | 先看 `latest.json` 的 `ms/jpg/loop` 是哪一段慢；再调 `--ai-every/--jpeg-every` |
| 检测不到标靶 | `--conf 0.20` 试；确认模型/类别；看 `latest.jpg` 的画面 |
| 框位置偏/误检 | 场景与训练数据差异大；补采 + 重训（见 `docs/POSE_V3_TRAINING.md`） |
| 编译报 hostfxr | 安装 .NET 7 并 `export DOTNET_ROOT=~/.dotnet` |

更多见 [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)。

---

## 9. 已知限制

- **串口独占**：同一时刻只能有一个程序用板子（详见第 5 节）。
- **AI 脚本退出会复位板子**：约 2~4 秒 USB 重新枚举后重连；这是为了绕开
  KPU 析构死锁。工作模式切调试用 `scripts/mode.sh debug`，不要 Ctrl-C。
- **kmodel 与 nncase 版本相关**：换固件后若加载失败，用同版本 nncase 重新编译。
- **板端单目标 top-1**：`yolo_pose.py` 只取最高分目标，不做 NMS；
  多目标场景需要 PC 端评估或补 NMS。
- **NMS 版 ONNX 不能转换**：nncase 导入失败，必须用原始输出 + 板端解码。
- **不插 USB 就没有回传**：离线代理只走 UART 给 ESP32，`live/latest.*` 不会更新。
