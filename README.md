# K230 视觉工程（USB 工具链 + Pose 模型 + 训练数据集）

把 K230 当「USB 摄像头 + AI 相机」用：一根 USB 线，PC 上实时看画面、抓帧、
跑 YOLOv8n-Pose（靶纸四角 + 靶心），结果落成 `live/latest.jpg` /
`live/latest.json`；训练、标注、导出、编译 kmodel 全在 `k230/tools/`。
板端脚本也可以放进 CanMV IDE 直接跑。

> **2026-10-22 工程精简**：激光识别（`laser_openmv.py`、激光标定、`$LS` 协议、
> 固件三态激光瞄准）、历史模型、历史训练 run、硬件资料均已删除；只保留下面列的。
> 被删内容有 tar 包留底（见文末「回滚」）。

## 系统分工

- **K230**：只负责"看"。YOLOv8n-Pose（`pose_v3`，5 关键点 = 四角 + 靶心）
  走 UART3（TX=GPIO32 / RX=GPIO33，115200）以
  `$KT,<seq>,<run>,<cx>,<cy>,<bx>,<by>,<conf>!` 上报**框中心 + 靶心 + 置信度**。
  板端常驻代理 `/sdcard/main.py` 由 ESP32 指挥：收到 `$E,START!` 才开相机跑识别，
  `$E,STOP!` 停。
- **ESP32**（固件单独维护，不在本仓库）：收到 `$KT` 后跑 PID + 速度自适应前视，
  让**靶心居中**；靶心连续 10 帧 / 900ms 丢失即判定打靶完成，云台回 NONE。
- **上电默认无模式**：K230 不跑识别，直到 ESP32 发 START（手柄 R2 / USB `gtrack`）。

## 目录

```
.
├── README.md                     本文件
├── AGENTS.md                     开发约定 + 踩坑记录（改代码前先读）
├── k230/                         ★ K230 视觉端（自包含，命令都在这里面跑）
│   ├── board/                    板端 MicroPython
│   │   ├── gimbal_agent.py       ★ 离线常驻代理（放 /sdcard/main.py）
│   │   ├── yolo_preview_ide.py   ★ CanMV IDE 预览（框 + 四角 + 靶心）
│   │   ├── yolo_pose.py          PC 推流用（Pose + ROI JPEG + $KT）
│   │   ├── k230_perf.py          板端性能单点入口（CPU 频率 / 帧缓冲）
│   │   ├── stream_camera.py      纯推流
│   │   └── uart_*.py             UART 接线自检
│   ├── k230tool/                 PC 端 USB 工具（python3 -m k230tool …）
│   ├── tools/                    标注 / 数据集 / 训练 / 导出 / 编译 / 评估
│   ├── models/                   当前模型（见 models/README.md）
│   ├── docs/                     架构、UART 协议、训练、标注、部署、排错
│   ├── scripts/                  mode.sh（调试/工作切换）、部署、udev
│   └── requirements*.txt
├── recovered/                    训练数据集
│   ├── dataset_pose_v3/          最终模型用的数据集（stats.json 里有划分统计）
│   └── k230_target_project/      两个父数据集（pose_v2_B + farneg_B）
└── scripts/                      mode.sh / ide_mode.sh / offline_mode.sh 根目录快捷入口
```

## 快速开始

### 1) K230 侧（视觉 + USB 调试）

```bash
cd k230
pip3 install -r requirements.txt          # pyserial / numpy / opencv

python3 -m k230tool probe                 # 全链路体检（连不上先跑这个）
python3 -m k230tool live                  # 实时识别，写 live/latest.jpg + latest.json
python3 -m k230tool live --jpeg-q 90      # 图像更清晰（ROI 模式）
python3 -m k230tool snapshot -o a.jpg     # 取一张（可复用 live 的最新帧）
python3 -m k230tool status                # live 状态（不占串口）
python3 -m k230tool push 文件 --dest /sdcard/...   # 推文件
```

`live/latest.json` 有检测框、5 个关键点、靶心、置信度、板端帧率和各段耗时
（`ms` 推理 / `jpg` 编码 / `loop` 整帧）。

### 2) 离线部署（不插 USB，5V 独立供电）

```bash
cd k230
python3 -m k230tool mode work --agent board/gimbal_agent.py
```

CanMV 上电自动执行 `/sdcard/main.py`：常驻等命令 → 收到 `$E,START!` 才开相机跑识别。
**不插 USB 时板子没有数据链路**，只有插着 USB 才谈得上"回传"。

### 3) 调试 / 工作模式切换 + CanMV IDE 预览

板端常驻代理会占着 REPL，用 IDE 预览前先切到调试模式：

```bash
# 仓库根目录（推荐；根目录 scripts/ 会转发到 k230/scripts/）
bash scripts/mode.sh debug     # 工作 -> 调试：等 REPL 回来
bash scripts/mode.sh status    # 只读查看当前模式
bash scripts/mode.sh work      # 调试 -> 工作：代理重新自启

cd k230 && bash scripts/mode.sh debug      # 也可以这样
```

然后打开 CanMV IDE，运行 `k230/board/yolo_preview_ide.py`（默认已配好
`/sdcard/models/pose_v3_384.kmodel`、`VIEW="full"`、`VIEW_SCALE=0.45`、`CONF=0.25`）。

> 工作 → 调试不用拔插 USB、也不用 Ctrl-C：PC 让 ESP32 连发 3 条 `$E,STOP!`
> （`gnone`），代理写 `offline_off` 后立即 `machine.reset()`。工作模式下 IDE
> 连不上是正常现象。

### 4) 重训 / 评测 Pose 模型

```bash
cd k230
# 数据集已生成好；要重建就运行 build_pose_dataset_v3.py
python3 tools/build_pose_dataset_v3.py \
    --sources ../recovered/k230_target_project/dataset_pose_v2_B \
              ../recovered/k230_target_project/dataset_pose_farneg_B \
    --val-src ../recovered/k230_target_project/dataset_pose_farneg_B \
    --test-src ../recovered/k230_target_project/dataset_pose_farneg_B \
    --out ../recovered/dataset_pose_v3

python3 tools/train_pose_v3.py --data ../recovered/dataset_pose_v3/data.yaml \
    --weights models/pose_v3.pt --epochs 40 --imgsz 320 --device cpu
python3 tools/export_onnx.py --weights models/pose_v3.pt --imgsz 384 --out models/pose_v3_384.onnx
python3 tools/compile_kmodel.py --onnx models/pose_v3_384.onnx --out models/pose_v3_384.kmodel \
    --input-size 384 --calib-dir ../recovered/dataset_pose_v3/images/train --samples 20
```

`models/pose_v3.pt` 既是在线模型，也是重训 / 重导出的起点权重。

## 模型（当前唯一）

`k230/models/pose_v3_384.kmodel`：5 关键点 Pose，输入 `uint8 [1,3,384,384]`，
输出 `[1,20,3024]`，3.48 MB（板端推理耗时待实测；256 版本约 22 ms）。

- 训练/评测过程见 [k230/docs/POSE_V3_TRAINING.md](k230/docs/POSE_V3_TRAINING.md)；
- 数据集 `recovered/dataset_pose_v3`；
- 远靶（farneg_B test）召回 50~56.5% → 100%，靶心 4.2~4.7 px → 1.48 px。

## 文档

- [k230/docs/POSE_V3_TRAINING.md](k230/docs/POSE_V3_TRAINING.md) —— 小目标/误检诊断、数据集、训练与评测
- [k230/docs/K230_ESP32_UART.md](k230/docs/K230_ESP32_UART.md) —— `$KT` / `$E` 协议
- [k230/docs/YOLOV8_DEPLOY.md](k230/docs/YOLOV8_DEPLOY.md) —— 训练 → ONNX → kmodel → 板端
- [k230/docs/POSE_ANNOTATION.md](k230/docs/POSE_ANNOTATION.md) —— 5 关键点标注规范
- [k230/docs/LASER_DETECT_TEST.md](k230/docs/LASER_DETECT_TEST.md) —— 激光点检测测试（cv_lite 灰度，独立脚本）
- [k230/docs/ARCHITECTURE.md](k230/docs/ARCHITECTURE.md) / [TROUBLESHOOTING.md](k230/docs/TROUBLESHOOTING.md)
- [k230/README.md](k230/README.md) —— K230 USB 工具链细节

## 回滚

清理前的完整快照：`scratch/删除归档_20261022/pre_cleanup_part1.tar.gz`
（被删的文件）、`pre_cleanup_part2_scratch.tar.gz`（旧 scratch/）、
`working_tree_changes.patch`（清理前的未提交改动）。`scratch/` 被 gitignore 忽略。

## 注意

- **串口独占**：`k230tool live` 与 CanMV IDE 不能同时占用 `/dev/ttyACM0`。
- 板端 REPL 一次只能发一行命令；`/sdcard/main.py` 被占用时先 `machine.reset()` 再写。
- 板端脚本不要"开相机却不取帧"（会堵死 VICAP 管线，只能断电恢复）。
