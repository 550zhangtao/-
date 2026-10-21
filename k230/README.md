# K230 USB 工具链 + YOLOv8 标靶识别（可移植包）

把亚博智能（YAHBOOM）K230 / CanMV 开发板当成一台 **USB 摄像头 + AI 相机**：
一根 USB 线，PC 上实时看画面、抓帧、跑 YOLOv8、把检测结果落成文件；
AI Agent 也能通过文件「看到」板子的实时画面。整个工具链都在本文件夹内，
拷到别的机器上只要装 Python 依赖即可。

> 本包由早期分散的插件/板端脚本重构而来，**不依赖 CanMV IDE 插件**，
> 纯 Python + pyserial + OpenCV。板端脚本会被自动下发，也可单独放进 CanMV IDE 运行。

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
│   ├── push_file.py           PC -> 板端文件
│   ├── pull_file.py           板端 -> PC 文件
│   └── cli.py                 统一命令行
├── board/                     板端 MicroPython（会被自动下发）
│   ├── stream_camera.py       采图 -> JPEG -> USB
│   ├── yolo_pose.py           YOLOv8-Pose 推理 -> 关键点 JSON + ROI JPEG
│   └── yolo_preview_ide.py    ★ 在 CanMV IDE 里直接看框/四角/靶心
├── tools/                     Pose 训练/标注/转换工具
│   ├── pose_label.html        ★ 5 关键点浏览器标注工具
│   ├── pose_prefill.js        约 180 张推荐图的预标注框
│   ├── pose_json_to_yolo.py   标注 JSON -> A/B 两版数据集
│   ├── train_pose_target.py   训练 4/5 关键点 YOLOv8n-Pose
│   ├── pose_eval.py           A/B 靶心/角点误差评估
│   ├── export_onnx.py         best.pt -> ONNX（nms=False）
│   ├── compile_kmodel.py      ONNX -> K230 kmodel（nncase + PTQ）
│   ├── dataset_from_captures.py  抓图 -> 数据集
│   └── check_env.py           环境自检
├── models/                    本地 kmodel / labels 放置处
├── scripts/                   udev 规则、启动脚本
├── examples/                  二次开发示例
├── docs/                      部署/架构/排错文档
├── live/                      运行时生成（latest.jpg / latest.json / history）
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
python3 -m k230tool push models/best.kmodel --dest /sdcard/models/best.kmodel

# 从板子拉文件/备份模型
python3 -m k230tool pull /sdcard/models/v2_std1.kmodel models/v2_std1.kmodel
```

`live` 的常用参数：

```bash
# 板端现有标靶模型（v2_std1 有响应；是否匹配你的标靶以现场画面为准）
python3 -m k230tool live --model /sdcard/models/v2_std1.kmodel --conf 0.05

# 提高帧率：降到 320x240
python3 -m k230tool live -w 320 -h 240 -q 60

# 只回传画面，不跑 AI（12 fps）
python3 -m k230tool live --no-ai

# 跑 60 秒后自动停（板子会自动复位）
python3 -m k230tool live --duration 60

# 保留最近 50 张历史帧（每 5 秒一张）
python3 -m k230tool live --history 50
```

---

## 4. 让 AI Agent 实时监测画面

这是本包的核心用法。`live` 在后台常驻，把最新一帧落盘：

| 文件 | 内容 |
| --- | --- |
| `live/latest.jpg` | 最新一帧，默认已画好检测框、时间戳、帧率、目标数 |
| `live/latest.json` | 序号、帧率、帧龄、检测框（类别/置信度/坐标）、板端统计 |
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

或者用命令行：

```bash
python3 -m k230tool status          # 不占用串口，查看 live 是否新鲜
python3 -m k230tool snapshot -o a.jpg   # live 在跑时直接复用最新帧
```

建议后台启动：

```bash
# 终端 A（或 nohup / 后台任务）
python3 -m k230tool live --duration 0 --history 200

# Agent / 你自己随时看
python3 -m k230tool status
```

`latest.json` 示例：

```json
{
  "seq": 133,
  "fps": 8.47,
  "age_ms": 40.2,
  "connected": true,
  "size": [640, 480],
  "detections": [
    {"label": "target", "cls": 0, "conf": 0.199, "box": [160, 330, 167, 152]}
  ],
  "infer_ms": 29,
  "board": {"fps": "8.4", "infer_ms": "28", "jpg_ms": "73", "mem": "3521856"}
}
```

---

## 5. 你自己在 CanMV IDE 里看画面

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
- **工作 → 调试** 不再用 Ctrl-C 杀脚本（会把 K230 打进 USB 不枚举/死锁）。
  PC 让 ESP32 连发 3 条 `$E,STOP!`（`gnone`），新版代理识别为调试暗号：
  写 `offline_off` → `machine.reset()`。全程不拔插、不需要改 ESP32 固件；
- 首次升级 / 旧版 `main.py` 没有暗号时，需要一次性进入调试：
  断电 → 取 TF 卡 → 根目录建空文件 `offline_off` → 上电，然后执行
  `python3 -m k230tool mode work --agent board/gimbal_agent.py`；
- 别名：`scripts/ide_mode.sh` = `mode debug`，`scripts/offline_mode.sh` = `mode work`。

串口是 **独占** 的：`k230tool live`/`monitor` 在跑时，CanMV IDE 连不上；
反之亦然。两条路径二选一：

**A. Agent 看（USB 工具）**：`python3 -m k230tool live`（先 `mode.sh debug`）

**B. 人看（CanMV IDE）**：

1. `bash scripts/mode.sh debug`（等它打印“调试模式就绪”）
2. 打开 CanMV IDE，连接板子
3. 打开 `board/yolo_preview_ide.py`，按运行

**默认已经是 Pose 模式**（模型 `/sdcard/models/pose_v3.kmodel`，
阈值 0.25），画面里画的是 **检测框 + 四个角点轮廓 + 靶心红十字**：

```python
MODEL = "/sdcard/models/pose_v3.kmodel"       # 5 关键点 Pose（小目标版）
VIEW = "full"                                 # 固定视窗；VIEW_SCALE=0.55 -> 352×264
CONF = 0.25
KP_NUM = 5                                    # TL,TR,BR,BL,靶心
DISPLAY_MODE = "ide"
```

实现要点（踩过的坑，见 `AGENTS.md`）：

- 板端 `aidemo` 没有 `yolov8_pose_postprocess`，预览脚本里用
  `libs.YOLO.aidemo = _PoseAidemo()` 注入 top-1 解码；
- IDE 预览**不初始化 Display**，直接用 `img.compress_for_ide()` 推帧；
- 默认 `VIEW="full"`，固定 320×240 全图，不跟随目标裁剪，画面尺寸/位置稳定；
  `VIEW="center"` 是固定中心 320×320 裁剪；`VIEW_SCALE=0.75` 可提高细节但保持尺寸恒定；
- 采集顺序改为**先拍显示帧、再拍 AI 帧**，避免框/关键点比画面晚约一帧；
- 板端没有 FreeType 字体，默认不写字；结果打印在 IDE 终端：
  `[30] 15.0fps target 0.35 kp4(320,240,v0.98) roi=(...)`。

> 停止脚本后板子会自动复位（KPU 析构死锁，复位最干净）。

**需要「人看 IDE + Agent 同时看」怎么办？**
单根 USB 调试口做不到同时被两个进程读。可选方案：
- 本包的 `live` 由一个人看，CanMV IDE 停用；
- 或者用板子的网络能力另开一路 MJPEG（板端 `libs/Network.py` 支持，
  但需要你提供 WiFi 账号并自行验证），USB 留给 IDE。

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
| 5 | 白色 10 号靶心 | 直接输出靶心坐标（B 版） |

框只负责提供特征区域，角点由关键点分支回归；B 版直接用第 5 点，
不再用 Hough/模板/分割。

- 训练：`tools/train_pose_target.py`（A=4 点，B=5 点）
- 标注：`docs/POSE_ANNOTATION.md`，工具 `tools/pose_label.html`
- 数据集：`tools/pose_json_to_yolo.py`（同一份标注生成 A/B 两版）
- 评估：`tools/pose_eval.py`
- 板端推流：`board/yolo_pose.py`
- CanMV 预览：`board/yolo_preview_ide.py`

### 6.2 板端模型

| 模型 | 说明 |
| --- | --- |
| `target_v2.kmodel` | 旧检测模型，仅作对照 |
| `pose_v3.kmodel` | **当前部署：v3 小目标 5 关键点 Pose（320，Q1）** |
| `target_pose.kmodel` | 旧 poseB_farhead_merged（板端保留回退） |

Pose kmodel 输出 `[1, 20, 2100]`，即 4+nc+3×5；A 版 `[1,17,2100]`。

### 6.3 跑 Pose

```bash
# PC 端看结果（框 + 四角轮廓 + 靶心）
python3 -m k230tool live --model /sdcard/models/pose_v3.kmodel --conf 0.25

# 无模型时先验证链路：用冒烟模型
python3 -m k230tool live --model /sdcard/models/pose5.kmodel --conf 0.05
```

`latest.json` 新增：

```json
{"det": [...], "kp": [x1,y1,v1, ... x5,y5,v5], "bull": {"x":..,"y":..,"conf":..,"status":"ok"}}
```

### 6.4 20 FPS

- 每帧 Pose 推理（320，约 31~33 ms）；
- 只编码检测框 ROI（原生分辨率），每 15 帧回传一张缩小全图；
- 板端 top-1 解码，不做 Python NMS（单目标场景）；
- 实测（B v1 Q1）：**pose_ms 30~36 ms，ROI 128 + 隔帧 JPEG 时识别循环 20.4~21.1 fps**（图像 ~10fps）；
- 正式模型若超过 50 ms，再评估隔帧 + 跟踪（跟踪态靶心漂移上限需实测）。

### 6.5 标注 / A/B 实验

> **注意**：训练工程（`k230_target_project/`：数据集、captures、runs）和历史标注
> `pose_labels*.json` 已随 2026-10-20 仓库清理删除，归档在工程外
> 未随本仓库发布；下面命令里的路径按你自己的数据目录替换。

1. 启动本地服务：`python3 -m http.server 8765 --bind 127.0.0.1`（项目根目录），
   打开 `http://127.0.0.1:8765/k230/tools/pose_label.html`；
2. 点“加载试点 20 张”，按 `docs/POSE_ANNOTATION.md` 标 5 个点；
3. 导出 `pose_labels.json` 到项目根目录；
4. 生成 A/B 数据集：

```bash
python3 tools/pose_json_to_yolo.py --json ../pose_labels.json \
    --captures ../k230_target_project/captures \
    --out-root ../k230_target_project
```

5. 训练：

```bash
python3 tools/train_pose_target.py --data ../k230_target_project/dataset_pose_A/data.yaml \
    --model ../k230_target_project/yolov8n.pt --epochs 100 --imgsz 320 --batch 8 --name poseA
python3 tools/train_pose_target.py --data ../k230_target_project/dataset_pose_B/data.yaml \
    --model ../k230_target_project/yolov8n.pt --epochs 100 --imgsz 320 --batch 8 --name poseB
```

6. 评估：

```bash
python3 tools/pose_eval.py --weights runs_target/poseB/weights/best.pt \
    --data ../k230_target_project/dataset_pose_B --kpt 5 --split test
python3 tools/pose_eval.py --weights runs_target/poseA/weights/best.pt \
    --data ../k230_target_project/dataset_pose_A --gt-data ../k230_target_project/dataset_pose_B \
    --kpt 4 --split test
```

### 6.6 CanMV 预览

`board/yolo_preview_ide.py` 默认：

```python
MODEL = "/sdcard/models/pose_v3.kmodel"
VIEW = "full"       # 固定视窗；VIEW_SCALE=0.55 -> 352×264
CONF = 0.25
```

画面：检测框 + 四角轮廓 + 第 5 点红十字；`VIEW="full"` 看全图。

### 6.7 离线云台代理（不插 USB，R2 触发识别）

`board/gimbal_agent.py` 是给 **ESP32 云台主控**用的常驻代理：

```bash
python3 -m k230tool push board/gimbal_agent.py --dest /sdcard/main.py
```

- CanMV 上电自动执行 `/sdcard/main.py`，**不需要 USB 数据线**（5V 独立供电即可）；
- 上电先初始化相机 + 模型，然后**空转等命令**：
  - ESP32 进跟踪（手柄 R2 / USB `gtrack` / 上电默认 TRACK）→ 发 `$E,START!`
    → 代理开始逐帧推理并回传 `$KT,...!`；
  - 离开跟踪（R1 / `gnone`）→ 发 `$E,STOP!` → 代理停止推理（不再占 KPU）；
  - 跟踪中 ESP32 2.5s 收不到 `$KT` → 每 3s 重发 `$E,START!`；
    K230 重启后会主动发 `$E,K230,READY!`，ESP32 收到就补一条 START；
- 回执：`$E,K230,READY! / START_OK! / STOP_OK! / ERR,INIT!`（ESP32 串口会打印）；
- **想回 CanMV IDE 调试**：`bash scripts/mode.sh debug`。
  PC 会让 ESP32 连发 3 条 STOP（`gnone`），代理写 `offline_off` 后立即
  `machine.reset()`，等 REPL 回来即可；不需要拔插 USB、不需要按 Ctrl-C。
  手工兜底：在 `/sdcard` 建空文件 `offline_off` 再上电。

> 自动运行的脚本如果写坏/卡死，板子会连 REPL 都没反应，只能**断电重上电**；
> `scripts/deploy_k230_agent.py` 会在重新上电的第一时间抢 REPL、删掉坏 `main.py`
> 并写入新代理（实测可用）。新版代理带“3×STOP = 切调试”暗号，
> 部署一次后工作↔调试切换全程免拔插、免 Ctrl-C。

## 7. 实测性能（本机，CanMV v1.4.3 / GC2093 / USB2.0 HS）

| 项目 | 帧率 | 说明 |
| --- | --- | --- |
| 纯推流 640×480 q70 | 11.8 fps | JPEG 编码 69 ms 是瓶颈 |
| 旧检测 320 | 8~8.5 fps | 整幅 JPEG |
| **Pose 5 点 320（B v1 Q1）** | **20.4~21.1 fps 识别循环** | pose 30~36 ms，ROI 128 + 隔帧 JPEG |
| Pose 图像流 | ~18 fps | 每帧 ROI，2~4 KB |
| USB 吞吐 | ~6 MB/s | 不是瓶颈 |

> 正式模型的帧率取决于 pose 推理耗时；超过 50 ms 时再考虑隔帧+跟踪。

## 8. 协议速览

板端 -> PC，每包 38 字节 ASCII 包头 + 载荷：

```
MAGIC(AA 55 4B 32) | 类型(1B) | seq(hex8) | plen(hex8) | wlen(hex8) | crc(hex8) | '|' | 载荷
类型：F=JPEG帧  T=检测JSON  L=日志  S=统计  H=握手  E=错误
```

- 串口会把 `0x0A` 改写成 `0x0D 0x0A`，板端把原始 `0x0D` 双写、包头纯 ASCII，
  PC 端两次 `replace` 还原，二进制零损伤。
- `T` 包在 `F` 包之前发送，PC 端按 `seq` 配对，画框时不会错帧。
- 详细设计见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

---

## 9. 排错

| 现象 | 处理 |
| --- | --- |
| 找不到串口 | USB 插板子 **USB0 调试口**；`lsusb` 应有 `1209:abd1` |
| `busy` / 打开失败 | 串口被 CanMV IDE/插件占用 → 先断开/关闭 IDE |
| `sensor(2) is already inited` | 上次脚本没干净退出 → 本工具会自动软复位；也可手动 RESET |
| 跑完 AI 脚本后板子无响应 | K230 KPU 析构死锁（已知）→ 按 RESET；本包脚本退出会自动 `machine.reset()` |
| 画面全黑 | 镜头盖/排线；`probe` 会报传感器型号 |
| 帧率低 | 降低分辨率/质量，或 `--no-ai`；检查 USB 是否插在 Hub/慢口 |
| 检测不到标靶 | `--conf 0.05` 试；模型/类别是否匹配；看 `latest.jpg` 确认画面 |
| 框位置偏/误检 | 模型训练数据与场景差异大；重新采集+训练；检查 kmodel 是否按官方预处理编译 |
| 编译报 hostfxr | 安装 .NET 7 并 `export DOTNET_ROOT=~/.dotnet` |
| ONNX 输出 84×2100 而非 4+nc | 导出时开了 NMS/end2end；用本包 `export_onnx.py` |

更多见 [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)。

---

## 10. 已知限制

- **串口独占**：同一时刻只能有一个程序用板子（详见第 5 节）。
- **AI 脚本退出会复位板子**：约 2~4 秒 USB 重新枚举后重连；这是为了绕开
  KPU 析构死锁。工作模式切调试用 `scripts/mode.sh debug`，不要 Ctrl-C。
- **K230 的 kmodel 与 nncase 版本相关**：训练/编译工具按官方 K230 文档配置；
  换固件后若加载失败，用同版本 nncase 重新编译。
- **Pose 模型待训练**：本包不附带 5 关键点数据集；当前 `pose5.kmodel`
  只是 1 张图的冒烟模型，输出无意义，仅用于验证链路。
- **板端单目标 top-1**：`yolo_pose.py` 只取最高分目标，不做 NMS；
  多目标场景需要 PC 端评估或补 NMS。
- **NMS 版 ONNX 不能转换**：nncase 导入失败，必须用原始输出 + 板端解码。
