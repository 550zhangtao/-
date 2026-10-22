# 项目约定（给后续开发/AI 会话）

本文件记录本项目的硬性约定。修改代码前先读。

---

## 0. 现在的工程长什么样（2026-10-22 精简后）

只保留四件事：

1. **USB 工具插件**：`k230/k230tool/`（`python3 -m k230tool …`）；
2. **训练模型**：`k230/models/pose_v3_384.kmodel`（部署）+ `k230/models/pose_v3.pt`（重训起点）；
3. **训练数据集**：`recovered/dataset_pose_v3/`（最终模型用）+ `recovered/k230_target_project/`（两个父数据集）；
4. **调试/工作模式切换脚本**：`k230/scripts/mode.sh`（根目录 `scripts/mode.sh` 转发）。

> **激光识别已整体删除**（2026-10-22）：`board/laser_openmv.py`、PC 激光标定、
> `$LS` 协议、固件里的激光三态瞄准与激光 IO **都不在了**。除非用户明确要求，
> 不要再加回来。被删内容的快照在 `scratch/删除归档_20261022/`（被 gitignore）。
>
> 训练数据集里的"激光光斑"只是**图像增强**（让模型不把白光斑当成靶心），
> 与板端激光识别无关。

---

## 1. CanMV IDE 预览是一等交付物

**用户主要通过 CanMV IDE 查看模型效果。** 任何新算法能力（检测 / Pose 关键点 /
跟踪 / 新的后处理）都必须同时落地三处，缺一不可：

1. PC 端：`k230/k230tool/`（`k230tool live` 能看到）；
2. 板端推流：`k230/board/yolo_pose.py`；
3. **CanMV 预览：`k230/board/yolo_preview_ide.py`**（IDE 里直接看）。

并且：

- 新能力要设为预览脚本的**默认可运行状态**（模型路径、阈值都配好），
  用户打开 `yolo_preview_ide.py` 点运行就能看到，不需要改代码；
- 交付前必须验证**板端绘制路径**：把画好的帧 `img.save()` 到 `/sdcard`
  （预览脚本里 `DBG_SAVE=N`），用 `k230tool pull` 拉回 PC 检查轮廓/框是否画对；
- 不能只验证 PC 端就宣布完成。

---

## 2. 串口独占

- `k230tool live/monitor/serve/probe/push/pull` 与 CanMV IDE **不能同时**占用串口；
- 自动化测试前先 `fuser -v /dev/ttyACM0` 确认没有 `canmv-backend`；
- 用户正在用 IDE 时不要抢串口。

### 2.1 事故复盘：`CanMV: Request 'runScript' timed out after 10000ms`

真凶是**同一个串口被两个进程同时打开**：后台 `k230tool live`（还在推流）
+ `canmv-backend`。IDE 抢不到板子，`runScript` 等 10s 就报超时。
而且强杀 `k230tool`（kill -9）后，板端脚本仍在跑、USB 写阻塞，REPL 会彻底卡死，
此时 Ctrl-C / Ctrl-D 都没用，必须**硬件复位**；复位时若碰了 BOOT 会进
`29f1:0230 K230 USB Boot Device`（USB 启动模式），拔插 USB 重上电即可恢复。

已做的加固：

1. `k230tool/link.py`：`K230Link.open()` 会先 `fuser` 检查端口占用，
   若占用者是 `canmv-backend` 直接报错并提示 `CanMV: Disconnect Board`，
   不再抢串口（强制绕过：`K230_ALLOW_CONFLICT=1`）；
2. `k230tool reset`：**真软复位**（raw REPL 里 `machine.reset()` + 等 USB 重新枚举），
   不再用 Ctrl-D（只重启 VM，工作模式下 `main.py` 会立刻重新占住 REPL）；
   枚举层卡住时会尝试 `USBDEVFS_RESET` 软件拔插兜底；
3. `k230/scripts/free_serial.sh [--kill]`：查看/清理残留 `k230tool` 进程；
4. 别再 `kill -9` 后台 live：优先 `k230tool reset` 或让它自己到 `--duration`。

### 2.2 调试/工作模式切换

```bash
bash k230/scripts/mode.sh debug    # 调试：/sdcard/main.py 跳过，REPL 空闲，IDE 可连
bash k230/scripts/mode.sh work     # 工作：自动跑 /sdcard/main.py，REPL 被占，IDE 不能连
bash k230/scripts/mode.sh status   # 只读；不会打断识别
bash k230/scripts/mode.sh reset    # 保持模式真软复位（工作模式自动 debug->work）
```

- **工作 → 调试绝不再连发 Ctrl-C**：实测会触发 KPU/相机析构死锁，把 K230
  打进 USB 不枚举（内核 `descriptor error -71`），只能断电。
- `board/gimbal_agent.py` 把 **1.5s 内 3 条 `$E,STOP!`** 识别为调试暗号：
  写 `/sdcard/offline_off` → `machine.reset()`。PC 侧让 ESP32 连发 3 次
  USB `gnone` 即可，全程免拔插。
- 调试 → 工作：`mode work` 删 `offline_off`，`machine.reset(wait_repl=False)`，
  等 USB/tty 稳定；`/sdcard/main.py` 自动运行。
- `yolo_preview_ide.py` 退出仍会 `machine.reset()`；IDE 需等 2~3 秒 USB
  重新枚举后再连。工作模式下 IDE 连不上是**正常现象**。

---

## 3. 板端固件限制（CanMV v1.4.3 / k230_canmv_yahboom）

| 限制 | 处理 |
| --- | --- |
| `aidemo` 没有 `yolov8_pose_postprocess` | 运行时替换 `libs.YOLO.aidemo`，用脚本里的 top-1 解码；不要改固件 lib |
| NMS 版 ONNX（`[1,300,21]`）nncase 无法导入 | kmodel 保持原始输出 `[1,20,N]`，板端只取 top-1（单目标场景） |
| `ulab.numpy.array(0)` 不支持标量 | id/score 用 Python 标量，坐标用 `np.array([...])` |
| `compress_for_ide` 在 IDE 中是唯一可靠预览方式 | 不初始化 Display；ROI 视图用 `compress_for_ide(roi=...)` |
| KPU 原生对象析构死锁 | 退出时把 yolo/kpu/ai2d 存进 keepalive 容器，默认 `machine.reset()`；`sys.modules` 访问必须 try 包住 |
| `emit()` 临时 bytes 多 | `gc.threshold(96*1024)` + 每 16 帧 `gc.collect()` |
| AI2D 是**左上角对齐**，不是居中 letterbox | 板端反变换必须 `pad_x=0, pad_y=0`，否则输出整体偏移半个 pad（640×480→320×320 会偏 80px） |
| IDE 预览画面/位置跳动 | `VIEW="full"` 固定视窗；`VIEW_SCALE=0.45`（288×216）、`IDE_QUALITY=70`、`AI_EVERY=2` |
| 阈值与远距离 | 合并模型 @0.4：80 张负样本 0 误检；@0.25：为 20px 极小靶保留召回，误检 1/80 |
| 串口独占 | 自动化前 `fuser -v /dev/ttyACM0` |
| `from media.sensor import *` 会把 `sys` 覆盖成 `mpp.sys` | `import sys` 必须放在星号导入**之后** |
| 板端文件 >32KB 的 `push_file` 有概率写坏 | 用 `tools/push_verified.py --chunk 8192`；先批量推完再 import |
| 一次发多行 REPL 命令会被吞 | 一行一条，或用 `;` 拼成单行 |
| `/sdcard/main.py` 被占用时 `open(...,'w')` 报 EPERM、`os.remove` 报 EBUSY | 先 `machine.reset()` 释放句柄再写，写完校验字节数 |
| 相机**开了却不取帧**会堵死 VICAP，整板连 REPL 都没反应 | 空闲时每 30ms `snapshot()` 排空通道；绝不 `machine.reset()` 做错误重试 |

---

## 4. 模型约定（Pose 路线）

- 关键点：**5 个**，顺序 TL, TR, BR, BL, 靶心；
- 训练：`tools/train_pose_v3.py`，`kpt_shape: [5,3]`，`flip_idx: [1,0,3,2,4]`；
- **不要**用 `flipud>0`（垂直翻转不会重排关键点）和 `erasing>0`（会擦掉靶心）；
- ONNX 输出：`[1, 4+nc+3*kpt, N]`；nc=1 时 384 输入是 `[1,20,3024]`、320 输入是 `[1,20,2100]`、256 是 `[1,20,1344]`；
- 输入 384×384、uint8、`input_range=[0,1]`、mean=0、std=1、`nms=False`；
- 框与关键点坐标已在模型输入像素系；板端按 letterbox 反变换到 640×480；
- 导出/编译：`tools/export_onnx.py` → `tools/compile_kmodel.py`；
  精度核对 `tools/kmodel_eval.py`（nncase Simulator，不需要板子）；
- 标注：`docs/POSE_ANNOTATION.md`，工具 `tools/pose_label.html`。

---

## 5. 当前状态

- **当前唯一模型**：`k230/models/pose_v3_384.kmodel`（384 输入，3.48 MB），
  板端路径 `/sdcard/models/pose_v3_384.kmodel`。由**同一个** `pose_v3.pt` 重新导出
  + 编译得到（不是新训练）；PC nncase Simulator 在 test 上 57/57，靶心
  3.45 px（256 版本 3.20 px，差异在噪声内；**远小靶 `sup_far` 2.28 vs 3.01 px**，
  384 明显更好，所以选 384）。
  `link.py` / `yolo_pose.py` / `yolo_preview_ide.py` / `gimbal_agent.py`
  四处默认值一致；
- **数据集**：`recovered/dataset_pose_v3`（train 1507 / val 28 / test 57）。
  生成脚本 `tools/build_pose_dataset_v3.py`；父数据集在
  `recovered/k230_target_project/`（pose_v2_B + farneg_B），现场帧在
  `k230/tools/scene_20261021*/`；
- **精度**：`farneg_B` test 远靶召回 50~56.5% → 100%，靶心 1.48 px；
  80 张负样本 @0.25 误检 1/80、@0.4 误检 0；PC kmodel 评测 57/57，靶心 3.2 px；
- **板端链路**：`$KT` 每个 AI 帧发一次（`ai_every=2`）；图像回传是
  **检测 ROI 原生像素**，没目标才发缩小全图；
- **性能（2026-10-22 实测）**：采集 57 fps，**JPEG 编码 640×480 要 65~70 ms，
  是唯一真瓶颈**；Pose 384 推理 40~43 ms。端到端 `live` 约 11 fps（ROI 回传）。
  提速只能降**回传像素数**或隔帧：`CFG["roi_scale"]` / `live --enc-scale` /
  `--ai-every` / `--jpeg-every`。
  本固件**没有 CPU 调频接口**（`os` 只有 `cpu_usage`），`set_framebuffers`
  也不支持；`board/k230_perf.py` 会打印 `cpu=auto`、`set_framebuffers 不可用`，
  这是正常降级，不是 bug。**不插 USB 时板子没有数据链路**，`live/latest.*` 不会更新；
- **ESP32 固件不在本仓库**；`$KT` 协议见 `docs/K230_ESP32_UART.md`。

### 历史问题的结论（代码已删，教训仍有效）

- **Pose 退化（2026-10-21）**：外场小靶识别率下降、误检上升。根因与修法（训练集尺度、
  硬负样本、flipud/erasing、框中心坐标 bug）见 `docs/POSE_V3_TRAINING.md`；
- **激光识别（2026-10-21 ~ 10-22）**：find_blobs → 标定微窗 → 文章式全图 →
  自适应阈值 → 形状门控，试了多轮，最后**整体删除**。可复用的教训：
  `find_blobs` 起步 11~14 ms，不适合每帧调用；`img.bytearray()` 全图 614 KB
  是掉帧/卡死主因；快速移动 + 复杂背景的传统 CV 方案很难同时做到"高检出 + 低误检"。
  以后再要"每帧全图扫描"的类似能力，先量单次 API 耗时再设计；
- **云台 PID（2026-10-22）**：旧记录里 kp=9 那套是配 **40px 死区**整定的，
  在 **8px 死区**（`WBAND_PX=8`）上会振荡发散，把云台推到限位。
  当前稳定值是 PAN `2.5/0.3/0.15`、TILT `2.5/0.2/0.10`、`smooth=0.02`；
  舵机多角度扫实测单轴 0.8~8.7 px、对角线 X 20~28 px（视角相关，不是单纯速度）。

### 下一步

- 20px 以下的极小靶：需要更多该尺度的人工标注；
- 检测抖动已接近模型上限，再提升应做"多候选 + 时序关联"（当前只取 top-1，
  相似物之间会跳）；
- 帧率要再上一个台阶，优先降**延迟**（推理 / 编码 / USB），不要改阈值换召回。
