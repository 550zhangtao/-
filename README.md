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
- **ESP32**：独占云台（pan GPIO27 / tilt GPIO14），把像素误差换算成舵机微秒，
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

### 4) 调试 / 工作模式切换 + CanMV IDE 预览

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

预览脚本默认已配好（`/sdcard/models/target_pose.kmodel`、`VIEW="full"`、`VIEW_SCALE=0.55`、
`CONF=0.25`），点运行即可看到检测框 + 四角轮廓 + 红色靶心，同时仍会向 ESP32 发 `$KT`。

### 3) ESP32 侧

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

当前部署：`k230/models/target_pose.kmodel`（poseB_farhead_merged Q1，farneg 检测头 +
B_final 关键点分支），板端路径 `/sdcard/models/target_pose.kmodel`。
配套的 `.pt` / `.onnx`、A 版最终权重和 v2 历史模型都在 `k230/models/`，
说明见 [k230/models/README.md](k230/models/README.md)。

## 文档

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
