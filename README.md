# 打靶小车（ESP32 主控 + K230 视觉）

用于打靶小车 srdp 组内成员使用：包含板卡资料、原理图、ESP32 主控固件、
K230 视觉端（YOLOv8-Pose 靶纸/靶心检测）、USB 调试工具与模型权重。

> **视觉方案：K230**（OpenMV 路线 V12/V13 已弃用，代码从仓库移除）。
> K230 用 Pose 模型一次给出「靶纸四角 + 靶心」，比 OpenMV 的阈值/几何法更稳，
> 而且可以离线独立供电运行。

## 系统分工

- **K230**：只负责"看"。YOLOv8n-Pose @320 找靶纸四角 + 靶心，
  通过 UART3（TX=GPIO32 / RX=GPIO33，115200）以
  `$KT,<seq>,<run>,<cx>,<cy>,<bx>,<by>,<conf>!` 上报目标框中心、靶心、置信度。
  板端常驻一个代理（`/sdcard/main.py`），由 ESP32 指挥：
  `$E,START!` 才开相机跑识别，`$E,STOP!` 停。
- **ESP32**：独占云台（pan GPIO27 / tilt GPIO14）与激光笔（GPIO5，低电平点亮，
  仅 R1/R2 模式开），把像素误差换算成舵机微秒，
  跑 PID + 速度自适应前视；同时负责电机（TB6612）、8 路循迹、PS2 手柄与模式切换。
- **上电默认无模式**：电机不使能、云台不动、K230 不跑识别；
  手柄按 **R2** 进跟踪（同时给 K230 发 START），**R1** 回手动（发 STOP）。

## 目录

```
.
├── README.md                       本文件
├── AGENTS.md                       开发约定 + 踩坑记录（改代码前先读）
├── firmware/
│   └── ESP32_Gimbal_K230/
│       └── ESP32_Gimbal_K230.ino   ★ ESP32 主控（云台 PID / 电机 / 循迹 / PS2 / $KT 解析）
├── k230/                           ★ K230 视觉端（自包含，命令都在这里面跑）
│   ├── board/                      板端脚本：gimbal_agent（离线代理）、yolo_pose、
│   │                               yolo_preview_ide（CanMV IDE 预览）、stream_camera、
│   │                               uart_test_k230、uart_pin_diag
│   ├── k230tool/                   PC 端 USB 插件（python3 -m k230tool …）
│   ├── tools/                      Pose 标注 / 训练 / 导出 / 编译 / 评估
│   ├── models/                     kmodel / pt / onnx 权重（见 models/README.md）
│   ├── docs/                       UART 协议、部署、标注、排错、架构
│   ├── scripts/                    udev 规则、离线部署、模式切换（mode.sh）
│   └── live/                       运行时输出（latest.jpg / latest.json，git 忽略）
├── scripts/                        根目录快捷入口：mode.sh / ide_mode.sh / offline_mode.sh
└── hardware/                       板卡资料 / 引脚图 / 原理图 / 电机与发射器手册
```

## 快速开始

### 1) K230 侧（视觉 + USB 调试）

```bash
cd k230
pip3 install -r requirements.txt          # pyserial / numpy / opencv

python3 -m k230tool probe                 # 全链路体检（连不上先跑这个）
python3 -m k230tool live                  # 实时识别，写 live/latest.jpg + latest.json
python3 -m k230tool snapshot -o a.jpg     # 取一张（可复用 live 的最新帧）
python3 -m k230tool status                # live 状态（不占串口）
python3 -m k230tool push 文件 --dest /sdcard/...   # 推文件
```

`live/latest.json` 里有检测框、5 个关键点、靶心、置信度、板端帧率与推理耗时。

### 2) 离线部署（不插 USB，5V 独立供电）

```bash
cd k230
python3 -m k230tool push board/gimbal_agent.py --dest /sdcard/main.py
```

CanMV 上电自动执行 `/sdcard/main.py`：常驻等命令 → 收到 `$E,START!` 才开相机跑识别。

### 3) 调试 / 工作模式切换 + CanMV IDE 预览

板端**常驻代理会占着 REPL**，用 IDE 预览前先切到调试模式
（`/sdcard/main.py` 不启动、相机空闲）：

```bash
# 在仓库根目录（推荐；根目录 scripts/ 会转发到 k230/scripts/）
bash scripts/mode.sh debug     # 工作 -> 调试：等 REPL 回来
bash scripts/mode.sh status    # 只读查看当前模式
bash scripts/mode.sh work      # 调试 -> 工作：代理重新自启

# 也可以 cd k230 后用同样的命令（真实脚本在 k230/scripts/）
cd k230 && bash scripts/mode.sh debug
```

然后打开 CanMV IDE / VSCode CanMV 插件，运行 `k230/board/yolo_preview_ide.py`。
调试完执行 `bash scripts/mode.sh work` 恢复；**工作模式下 IDE 连不上是正常的**。

> 工作 → 调试不需要拔插 USB、也不需要 Ctrl-C：PC 会让 ESP32 连发 3 条
> `$E,STOP!`（`gnone`），新版代理写 `offline_off` 后立即 `machine.reset()`。
> 首次升级代理（在 `k230/` 下）：
> `python3 -m k230tool mode work --agent board/gimbal_agent.py`。

预览脚本默认已配好（`/sdcard/models/pose_v3.kmodel`、`VIEW="full"`、`VIEW_SCALE=0.55`、
`CONF=0.25`），点运行即可看到检测框 + 四角轮廓 + 红色靶心，同时仍会向 ESP32 发 `$KT`。

### 4) 重训 / 评测 Pose 模型（小目标版）

```bash
cd k230
python3 tools/build_pose_dataset_v3.py ...          # 小目标 + 激光 + 硬负样本
python3 tools/train_pose_v3.py --data ../recovered/dataset_pose_v3/data.yaml \
    --weights models/spot_v1.pt --epochs 40 --imgsz 320 --device cpu
python3 tools/bench_pose.py --weights models/pose_v3.pt \
    --data ../recovered/k230_target_project/dataset_pose_farneg_B \
    --splits test --far-copies 2 --neg-data ../recovered/k230_target_project/dataset_pose_farneg_B
```

排查「远靶/靶心识别率下降、误检上升」的完整过程和数据见
[k230/docs/POSE_V3_TRAINING.md](k230/docs/POSE_V3_TRAINING.md)。

### 5) ESP32 侧

Arduino-ESP32 core **3.3.10**，板型 `ESP32 Dev Module`：用 Arduino IDE 打开
`firmware/ESP32_Gimbal_K230/` 直接上传；或

```bash
arduino-cli compile --fqbn esp32:esp32:esp32 firmware/ESP32_Gimbal_K230
arduino-cli upload  -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32 firmware/ESP32_Gimbal_K230
```

接线：ESP32 UART2 RX=16 / TX=17 ↔ K230 12Pin **UART3 TX=GPIO32 / RX=GPIO33**，共地。
舵机 PAN=27 / TILT=14；电机 PWM 33/32 + IN1/IN2 2/4、13/26，STBY=25。

USB 命令：`status` / `mode` / `kt` / `pad` / `s1 <us>` / `s2 <us>` / `sh` /
`gtrack` / `gmanual` / `gnone` / `pid <kp> <ki> <kd>` / `sweep` / `ledcdump`（`help` 看全部）。

## 模型

当前部署：`k230/models/pose_v3.kmodel`（2026-10-21 小目标微调，5 关键点，
板端路径 `/sdcard/models/pose_v3.kmodel`）。相比旧 `poseB` / `spot_v1`：

- `farneg_B` test 远靶召回 **50~56.5% → 100%**，靶心 **4.2~4.7 px → 1.48 px**；
- 全新视角 26×38 px 小靶：旧模型 0%，v3 80%（开低置信复筛后 95%）；
- 负样本误检 1.2%@0.25、0%@0.4；板端 13.6 → 18.2 fps（激光搜索优化后）/ 推理 31 ms。

配套的 `.pt` / `.onnx`、旧 poseB 与 v2 历史模型都在 `k230/models/`，
说明见 [k230/models/README.md](k230/models/README.md)。
旧的 `/sdcard/models/target_pose.kmodel`（poseB）保留在板端作回退。

## 文档

- [k230/docs/POSE_V3_TRAINING.md](k230/docs/POSE_V3_TRAINING.md) —— **小目标/误检问题的诊断、数据集、训练与评测**
- [k230/docs/AIM_LASER_SEARCH.md](k230/docs/AIM_LASER_SEARCH.md) —— 激光 IO / 搜索优化 / 三态追踪
- [k230/docs/LASER_PREDICTION.md](k230/docs/LASER_PREDICTION.md) —— **Kalman 预测 + 门控搜索 + 预算/看门狗（防卡死）**
- [k230/docs/K230_ESP32_UART.md](k230/docs/K230_ESP32_UART.md) —— `$KT` / `$E` 协议与调参记录
- [k230/docs/YOLOV8_DEPLOY.md](k230/docs/YOLOV8_DEPLOY.md) —— 训练 → ONNX → kmodel → 板端
- [k230/docs/POSE_ANNOTATION.md](k230/docs/POSE_ANNOTATION.md) —— 5 关键点标注规范
- [k230/docs/TROUBLESHOOTING.md](k230/docs/TROUBLESHOOTING.md) —— 排错
- [k230/README.md](k230/README.md) —— K230 USB 工具链细节

## 注意

- **串口独占**：`k230tool live` 与 CanMV IDE 不能同时占用 `/dev/ttyACM0`。
- 板端 REPL 一次只能发一行命令；`/sdcard/main.py` 被占用时先 `machine.reset()` 再写。
- 板端脚本不要"开相机却不取帧"（会堵死 VICAP 管线，只能断电恢复）。
- PS2 手柄若 `pad=0` 说明接收器没被识别（查供电与 DI/DO 接线），此时 R2/R1 无效。
