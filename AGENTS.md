# 项目约定（给后续开发/AI 会话）

本文件记录本项目的硬性约定。修改代码前先读。

## 1. CanMV IDE 预览是一等交付物

**用户主要通过 CanMV IDE 查看模型效果。** 任何新算法能力（检测 / Pose 关键点 / 跟踪 / 新的后处理）都必须同时落地三处，缺一不可：

1. PC 端：`k230/k230tool/`（`k230tool live` 能看到）；
2. 板端推流：`k230/board/yolo_pose.py`（USB 回传）；
3. **CanMV 预览：`k230/board/yolo_preview_ide.py`**（IDE 里直接看）。

并且：

- 新能力要设为预览脚本的**默认可运行状态**（模型路径、TASK、阈值都配好），
  用户打开 `yolo_preview_ide.py` 点运行就能看到，不需要改代码；
- 交付前必须验证**板端绘制路径**：把画好的帧 `img.save()` 到 `/sdcard`，
  用 `k230tool pull` 拉回 PC 检查轮廓/框是否画对；
- 不能只验证 PC 端就宣布完成。

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
3. `k230tool/scripts/free_serial.sh [--kill]`：查看/清理残留 `k230tool` 进程；
4. 别再 `kill -9` 后台 live：优先 `k230tool reset` 或让它自己到 `--duration`。

### 2.2 调试/工作模式切换（2026-10-21）

```bash
bash k230/scripts/mode.sh debug    # 调试：/sdcard/main.py 跳过，REPL 空闲，IDE 可连
bash k230/scripts/mode.sh work     # 工作：自动跑 /sdcard/main.py，REPL 被占，IDE 不能连
bash k230/scripts/mode.sh status   # 只读；不会打断识别
bash k230/scripts/mode.sh reset    # 保持模式真软复位（工作模式自动 debug->work）
```

- **工作 → 调试绝不再连发 Ctrl-C**：实测会触发 KPU/相机析构死锁，把 K230
  打进 USB 不枚举（内核 `descriptor error -71`），只能断电。
- 新 `board/gimbal_agent.py` 把 **1.5s 内 3 条 `$E,STOP!`** 识别为调试暗号：
  写 `/sdcard/offline_off` → `machine.reset()`。PC 侧让 ESP32 连发 3 次
  USB `gnone` 即可，不需要改 ESP32 固件；全程免拔插。
- 调试 → 工作：`mode work` 删 `offline_off`，`machine.reset(wait_repl=False)`，
  等 USB/tty 稳定；`/sdcard/main.py` 自动运行。
- 旧版 `main.py` 没有暗号时，一次性断电取 TF 卡建 `offline_off`，上电后
  `mode work --agent board/gimbal_agent.py` 升级代理。
- `yolo_preview_ide.py` 退出仍会 `machine.reset()`；IDE 需等 2~3 秒 USB
  重新枚举后再连。工作模式下 IDE 连不上是**正常现象**。

## 3. 板端固件限制（CanMV v1.4.3 / k230_canmv_yahboom）

| 限制 | 处理 |
| --- | --- |
| `aidemo` 没有 `yolov8_pose_postprocess` | 运行时替换 `libs.YOLO.aidemo`，用脚本里的 top-1 解码；不要改固件 lib |
| NMS 版 ONNX（`[1,300,21]`）nncase 无法导入 | kmodel 保持原始输出 `[1,20,2100]`，板端只取 top-1（单目标场景） |
| `ulab.numpy.array(0)` 不支持标量 | id/score 用 Python 标量，坐标用 `np.array([...])` |
| `compress_for_ide` 在 IDE 中是唯一可靠预览方式 | 不初始化 Display；ROI 视图用 `compress_for_ide(roi=...)` |
| KPU 原生对象析构死锁 | 退出时把 yolo/kpu/ai2d 存进 keepalive 容器，默认 `machine.reset()`；**本机固件 `sys` 没有 `modules`**，`sys.modules[...]` 必须 try 包住（实测会 AttributeError） |
| `emit()` 临时 bytes 多 | `gc.threshold(96*1024)` + 每 8 帧 `gc.collect()` |
| AI2D 是**左上角对齐**，不是居中 letterbox | 板端反变换必须 `pad_x=0, pad_y=0`，否则输出整体偏移半个 pad（640×480→320×320 会偏 80px） |
| IDE 预览画面/位置跳动 | `VIEW="full"` 固定视窗；性能模式 `VIEW_SCALE=0.55`（352×264）、q80、`AI_EVERY=3` |
| 阈值与远距离 | 合并模型 @0.4：80 张负样本 0 误检；@0.25：为 20px 极小靶保留召回，误检 1/80 |
| 串口独占 | 自动化前 `fuser -v /dev/ttyACM0` |

## 4. 模型约定（Pose 路线）

- 关键点：**5 个**，顺序 TL, TR, BR, BL, 靶心；
- 训练：`tools/train_pose_target.py`，`kpt_shape: [5,3]`，`flip_idx: [1,0,3,2,4]`；
- ONNX 输出：`[1, 4+nc+3*kpt, 2100]`，nc=1 时 `[1,20,2100]`；A 版 `[1,17,2100]`；
- 输入 320×320、uint8、`input_range=[0,1]`、mean=0、std=1、`nms=False`；
- 框与关键点坐标已在模型输入像素系；板端按 letterbox 反变换到 640×480；
- 导出/编译：`tools/export_onnx.py` → `tools/compile_kmodel.py`；
- 标注：`docs/POSE_ANNOTATION.md`，工具 `tools/pose_label.html`。

## 5. 当前状态

- **仓库已清理（2026-10-20）**：只保留 `k230/`（USB 插件 + 板端脚本 +
  工具 + 权重）、`ESP32_Gimbal_K230/`（固件）、`OpenMV 路线与旧版小车固件均已移除。
  旧路线、训练工程（数据集/captures/runs）、历史标注 JSON、`backups/`、`scratch/`
  全部删除，归档在工程外 本地归档（未入库）（873 MB）；
- 旧路线（segment / Hough / 模板 / 手工角点单应）已删除（归档在
  本地归档（未入库）），工具包内不再引用；
- **板端 Pose 链路已验证**：5 关键点 kmodel `[1,20,2100]` 可转换/运行；
  B v1 Q1 实测 `pose_ms≈30~36ms`，ROI 128 + `jpeg_every 2` 时识别循环 **20.4 fps**（图像 ~10fps）；
  AI2D 坐标反变换已修正（左上对齐，pad=0），预览四角/靶心与 PyTorch 一致；
- 冒烟模型只训练 1 张，输出无意义，仅用于验证链路；
- **A/B 对照结果（13 张人工 held-out）**：
  A v1 11.07 px / B v1 **6.04 px**；A v2 8.03 px / B v2 10.39 px。
  新数据明显改善 A，但自动伪标签让 B 负迁移；最终用 farneg 检测头 + B_final 关键点分支合并部署；
- 补充采集 340 张（10 场景），v2 数据集 279 张（252 train/10 val/16 test）；
  `sup_occl` 因自动预标落在手上已剔除，`sup_far` 唯一帧标错已剔除；
- 数据集生成：`tools/pose_json_to_yolo.py` 从同一 JSON 生成 A/B 两版；
- CanMV 预览：`board/yolo_preview_ide.py` 已接 5 关键点绘制 + top-1 解码，
  默认 `MODEL=/sdcard/models/pose_v3.kmodel`、`VIEW="full"`、`VIEW_SCALE=0.55`（352×264）；
- PC：`k230tool live` 默认走 Pose，`latest.json` 输出 `kpts/bull`；
- **当前部署模型（2026-10-21 起）**：`pose_v3`（小目标微调），板端
  `/sdcard/models/pose_v3.kmodel`；旧 `poseB_farhead_merged` 保留为
  `/sdcard/models/target_pose.kmodel` 回退；
- 远距离 10 张：3.05 px / P95 4.92；原 13 张：6.47 px / P95 9.57；
  80 张负样本 @0.4 误检 0；@0.25 误检 1/80（为 20px 极小靶保留召回）；
- IDE 预览：性能模式 352×264、`IDE_QUALITY=80`、每 3 帧跑一次 pose；
  需要 640×480 时设 `VIEW_SCALE=1.0`（彩色约 8~10fps）。
- **现场测试**（`tools/pose_field_test.py`）：无靶 199 帧误检 **0**；
  靶纸可见段检出率 74~83%、靶心成功率 71~75%（移动/倾斜时下降）；
  IDE 预览固定视窗、先拍显示帧再拍 AI 帧，画面/叠加不再跳动或错位；
- **ESP32 云台精简工程**：`firmware/ESP32_Gimbal_K230/ESP32_Gimbal_K230.ino`，
  只保留 UART 通信 + PAN/TILT 舵机；K230 发送 `$KT`，ESP32 回 `$E,ACK`；
  - ESP32 USB 命令/协议解析/舵机 PWM 已实测；
  - K230 UART1 自环通过（收到自己的 `$KT`），ESP32 UART2 自环通过；
  - **链路已连通（2026-10-19 复测）**：K230 **12Pin UART3（TX=GPIO32，RX=GPIO33）**
    ↔ ESP32 UART2（RX=16，TX=17）全双工 20/20 帧、`bad=0`，ACK 正常回传，
    云台随目标 x 自动修正（PAN 1500→1158）。之前 0 帧是 K230 侧接线插错，
    不是 GPIO16/固件问题；**4Pin 口 UART1（GPIO9/10）实测仍不通**，不要再走该口；
  - 复测方法：`k230tool` 的 raw REPL 下发 `uart_test_k230.py`（`UART_ID=3`），
    ESP32 用 USB `status` 看 `frames/bad`，K230 打印 `ACK $E,ACK,...`；
  - **云台跟踪已联调可用（2026-10-19）**，踩到并修掉三个真 bug：
    1. `yolo_preview_ide.py` 里 `_init_uart()` **从未被调用** → IDE 预览下
       `_send_uart()` 一直静默 return，**一帧 `$KT` 都没发**（这就是"舵机不动"的根因）；
       已在 `main()` 的 `_config_preprocess()` 后补上调用，并加了一次性告警；
    2. 协议要求 `cx,cy` 是**框中心**，`yolo_pose.py` / `yolo_preview_ide.py`
       发的是框左上角 `x,y`；已改为 `x+w/2, y+h/2`；
    3. core 3.3.10 的 **`ledcWrite()` 每次阻塞约 40ms**（等 PWM 周期），
       两个舵机=80ms，控制环被迫降到 25Hz 且积分按 10ms 算 → 实际转速只有设定的 1/4。
       已改为**直接写 LEDC 寄存器**（`LEDC.channel_group[0].channel[ch].duty.duty =
       duty<<4; ...conf1.duty_start = 1`，约 0.6µs），控制环 66kHz、舵机 100Hz 更新；
       通道固定 CH_PAN=0 / CH_TILT=1（`ledcAttachChannel`）；
  - **控制器 = 图像系 PID + 速度自适应前视 + 速度一阶低通**（可在线调参）：
    **PAN `kp=2.5 ki=0.3 kd=0.15` / TILT `kp=2.5 ki=0.2 kd=0.1`**；
    速度指令低通 `smooth=0.02s`（丝滑感的主要来源），死区 8px；
    限速 2600/1000us/s、加速度 40k/20k；输出角速度（us/s），100Hz 更新；
    丢目标清积分；抗积分饱和；微分低通 0.35；
  - **重要教训：真实对象增益比仿真假设大 3 倍以上**。仿真按 k=0.6px/µs 整定出
    kp=7，真机上 kp≥4 就会把云台怼到限位并丢失目标（pan 一路走到 2260），
    **真机稳定值 kp=2**。以后重调务必先在真机上扫 kp，不要直接套仿真值；
  - 目标预处理链：**3 点中值滤波** → 一阶低通 α=0.25 → 离群点抑制
    （单帧跳变>80px 先忽略，连续 4 帧才接受）；
  - 前视预测：`err = x + vx·leadEff`，`leadEff = 0.2s × clamp(|vx|/500, 0.7, 1.2)`；
    目标平滑 α=0.35（跟手优先）；
  - **丝滑调参（2026-10-20）**：速度指令加一阶低通 `smooth <tau>`，
    静止时 tau=0.10 可以让 pan 完全不动（0µs 漂移、误差 6px），
    但 tau=0.10 移动时会明显变粘（|ex| 90px）；tau=0.05 折中，
    最终手感取 **0.02**（用户实测"不粘也不抖"）；kd 从 0.4 降到 0.15、
    ki 从 1 降到 0.3 也明显减少抖动（微分/积分都在放大检测噪声）；
  - 真机实测（开发脚本 `track_eval.py` 已随清理删除，20s/轮，靶纸在画面内、kp=2）：
    | 场景 | \|ex\| 中位 | \|ex\| P95 | \|ey\| 中位 | 死区占比 | 结论 |
    | --- | --- | --- | --- | --- | --- |
    | 静止 | **9px** | 31px | **7px** | 45% | state=2 有 16/44 帧，pan 速度中位 0 |
    | 移动 | **69px** | 186px | 36px | 2% | 滞后明显，受帧率/延迟限制 |
  - **机械限位坑**：靶纸平放在桌面、相机朝下时，tilt 会顶在 1200（下限），
    纵向永远差 40~100px。放靶纸要让它落在画面中间偏上，或调整相机安装角；
  - 在线调参：`pid <kp> <ki> <kd>` / `pid2 ...` / `pidshow` / `trk <alpha> <lead>`；
  - 试过并放弃：世界系（角度域）速度前馈（依赖拍摄时间戳，capdel 差 30ms 就发散）、
    纯逐帧比例增量（k=0.6 静态就震荡）、更大前视（更差）；
  - 结论：静止/慢速已可用（个位数 px）；再提速只能降延迟
    （提高 K230 帧率/降推理耗时），控制器侧已到真机稳定极限；
  - 自检命令：`status` / `s1` / `s2` / `sh` / `sweep` / `pulse <pin>`（脉宽回读）/
    `ledcdump`（LEDC 通道+定时器计数）/ `s1api <us>`（走老 ledcWrite 路径对照）；
  - 注意：相机装在云台上，验证舵机是否真动可对比 pan 两端的全图
    （`--full-every 1` 抓帧做相位相关，或直接看画面是否完全不同）；
- **V19 合并（2026-10-20）**：旧版小车固件（V19，已并入 K230 云台跟踪，代码不再单独保留）：
  - 模式拆成两套**并行**状态：`carMode`（L1 手动 / L2 循迹）+ `gimbalMode`
    （R1 手动 / R2 跟踪），小车行驶与云台跟踪互不影响；
  - **上电状态（2026-10-20 定稿）**：小车 `CAR_NONE`、云台 `GIMBAL_NONE`，
    **上电默认无模式**：电机不使能、云台不动、K230 不跑识别（ESP32 还会发一次
    `$E,STOP!`）。**必须按手柄 R2（MODE/SELECT 不必须）才进跟踪**；
    掉电重启同理，不会自己跑。`BOOT_GIMBAL_MODE` 可改回 `GIMBAL_TRACK` 改成上电即跟踪；
  - **云台跟踪只依赖 `$KT`，绝不依赖手柄**（2026-10-20 修）：以前手柄掉线 2s 会把
    `autoTrack` 置 0，表现就是"明明在 R2 模式，云台一动不动"；现在掉线只断电机。
    注意：手柄没识别时（`pad=0`）按 R2 没有任何反应，这是硬件问题，不是模式逻辑；
  - 新增 `$KT` 解析；控制器就是上面那套 PID+前视+低通，代码在 `updateTrack()`，
    参数在线可调：`pid/pid2/smooth/trk`（旧的 `$G2,TGT`/OpenMV 瞄准链路已废弃）；
  - **LEDC 通道划分**：舵机固定 **0/1**（`applyServos()` 直接写寄存器，V19 的
    `ledcWrite()` 在 50Hz 下每次阻塞约 40ms，云台每帧要卡 80ms），电机固定
    **2/3**（`ledcAttachChannel` 显式指定）；group0 内分属不同定时器，互不影响；
  - **实测云台图像增益（2026-10-20，逐帧标定）**：pan ≈ **0.97 px/µs**、
    tilt ≈ **0.20 px/µs**。V19 注释里写的 "pan≈3.0px/µs" 是错的，
    这导致之前按错误增益整定，tilt 实际比 pan 慢 5 倍；
  - **两个真正的"抖"根因（都已修）**：
    1. **整数截断**：`panUs = int(panUs + 速度*dt)` 在小速度时每周期增量 <1µs
       被截成 0，表现为"不动→攒够→跳一步"。改成内部浮点累积 `ktPosUs[]`，
       输出时再 `lroundf`（外部改动 >8µs 时自动重同步）；
    2. **积分对抗静摩擦**：舵机有 ~40µs 不响应区，ki>0 时积分慢慢绕动，
       攒够后突然跳一下。**ki 必须为 0**，死区取 40px（≈40µs 分辨率）；
  - 云台参数最终值（2026-10-20，用户实测"快、稳、丝滑"）：
    `kp=9.0 **ki=0** kd=0.5`（PAN）/`kp=9.0 ki=0 kd=0.6`（TILT）、
    `db=40px`、`accel=15000 us/s²`、`smooth=0.006`、`alpha=0.40`、
    `lead=0.25`（自适应 0.7~1.2×）、boost=0；
  - 调试工具（新增）：
    - `ktlog`：导出最近 120 帧逐帧数据（dt,bx,by,cx,cy），用于判断抖动来自
      检测还是控制。实测静止靶纸**检测抖动只有 3~7px（P90）**，不是抖动源；
    - `gmanual`/`gm`、`gnone`：云台进手动/无模式（配合 `s1/s2` 标定增益）；
  - 实测（静止，tracking on）：pan/tilt 范围可到 **0µs**、|ex| 3px；
    移动：kp 3→9 时 |ex| 中位 62→**31px**，kp=11 静止开始发散（9 是上限）；
  - **调参顺序（以后照这个来）**：先把 ki 归零、死区设到舵机分辨率级
    （40px），再用 `s1/s2 + ktlog` 标定真实 px/µs，最后按"环增益 ≈ kp*k ≈ 9/s"
    选 kp，用 accel 压抖动；不要再用仿真里假定的 k=0.6。
  - 物理上限：K230 目前 ~14fps + ~100ms 链路延迟，快靶误差主要来自这里，
    再提速需要先降延迟（提高 K230 帧率/降推理耗时），控制器侧已到真机稳定极限；
  - **补齐交付物（2026-10-20）**：`firmware/ESP32_Gimbal_K230/ESP32_Gimbal_K230.ino` 已用
    V19 的模块补齐 PS2 手柄 / TB6612 电机 / 8 路循迹 / 双模式（约 +1000 行），
    **原有云台 + `$KT` + 舵机直写代码逐字节保留**（只换了 loop 里一个回调：
    `handleTargetLine` → `onK230Line`，后者内部仍原样转给 `handleTargetLine`，
    并额外认 `DONE`/`$E,` 帧）。合并/yaml 生成脚本与纯净版备份已随清理删除，
    `.ino` 现在就是唯一源文件，直接改它即可；
  - 该文件编译验证：`arduino-cli compile --fqbn esp32:esp32:esp32 ESP32_Gimbal_K230`
    （core 3.3.10-cn，`--warnings all` 时 0 warning）；arduino-cli 已随清理删除，
    用 Arduino IDE 上传，或从归档里取回 `scratch/bin/arduino-cli`；
  - 链路自诊断（`linkDiag()`，1s 一次，只报问题）：TRACK 但 1s 没 `$KT` 会打印
    "查 K230 UART3 TX=GPIO32→ESP32 GPIO16 / RX=GPIO33←GPIO17"；非 TRACK 却收到
    `run=1` 帧会提示按 R2 / 输入 `gtrack`；5s 还没识别到手柄会提示 `pad=0` 排查点；
  - PS2 时钟从 5µs 放宽到 **10µs（50kHz）**：V19 的 5µs≈100kHz 已贴着 PS2 上限，
    克隆接收器容易一个字节都读不出来（`pad_bad` 一直涨、`pad=0`）；
  - 回归仿真（宿主桩 + 真实 .ino，覆盖上电即跟踪/无手柄/R1保持/R2恢复/掉线/提示 6 项）
    已随清理删除，源码在归档 tar 的 `scratch/sim/`；
  - 没接 PS2 手柄时不会误判"手柄失联"（`padEverSeen` 门控），
    USB/`$KT` 链路可单独工作；旧的 `manual/auto` 仍只是裸开关（直接改
    `autoTrack`），选模式请用 `cmanual/cline/cnone/gmanual/gtrack/gnone`；
- **K230 卡死教训（2026-10-20）**：`/sdcard/main.py` 里 **开了相机却不取帧**（比如
  "初始化相机后空转等命令"）会把 VICAP 管线堵死，表现是**整块板连 REPL 都没反应**，
  Ctrl-C / `k230tool reset` 全部无效，只能断电重上电。正确做法：**首次 START 才开相机**，
  之后空闲时每 30ms `snapshot()` 排空两路帧；且绝不 `machine.reset()` 做错误重试。
- **R2 触发 K230 识别（2026-10-20）**：ESP32 是"指挥方"，K230 不再自己乱跑：
  - ESP32 进 `GIMBAL_TRACK`（手柄 R2 / USB `gtrack`）-> 发 `$E,START!`；
    离开跟踪（R1 / `gnone`）-> 发 `$E,STOP!`（`k230Cmd()`）；
    **上电默认 `GIMBAL_NONE`，只发一次 STOP，不会自己开跑**；
  - **IDE 停止 ≠ 致命错误（2026-10-20 修）**：CanMV IDE / VSCode 插件按"停止"抛的是
    `Exception("IDE interrupt")`（不是 `KeyboardInterrupt`）。旧版代理把它当致命错误
    → `_drop_pipeline()` + 2s 后重启，REPL 一直被占，IDE 就报"当前固件不支持 REPL 输入"。
    现在识别到 `interrupt`/`ide` 关键字就打印提示并**退出把 REPL 交还**；
    要恢复代理：断电重上电，或在 IDE 里运行 `/sdcard/main.py`。
  - 跟踪中 2.5s 没 `$KT` -> 每 3s 重发 START；K230 重启会发 `$E,K230,READY!`，
    ESP32 收到且当前在 TRACK 就补发 START（两边先后启动都能自愈）；
  - K230 侧 `board/gimbal_agent.py`：放 `/sdcard/main.py`，上电常驻；
    **首次 START 才开相机+模型**（约 1.5s），START 之后逐帧推理，STOP 只停推理；
    空闲时每 30ms 排空两路帧（不排空会堵死管线）；
    **`AUTO_START_MS = 0`（默认，不自己开跑）**：上电后没人按 R2 就一直空转等命令；
    回执 `$E,K230,READY/START_OK/STOP_OK/ERR,INIT!`；
  - **离线运行**：CanMV 上电自动执行 `/sdcard/main.py`，**不插 USB、5V 独立供电**
    即可（实测 main.py 自动运行有效）；想用 CanMV IDE 调试用
    `bash k230/scripts/mode.sh debug`（不再手工建 `offline_off`），代理会退出停在 REPL；
  - 部署/救援：`python3 -m k230tool mode work --agent board/gimbal_agent.py`
    （内部走 push + 校验 + 真软复位）；旧式 `k230tool push` 仍可用；
    自动运行脚本写坏会把板子卡到 **REPL 都没反应**（本次踩过），只能断电重上电，
    `k230/scripts/deploy_k230_agent.py` 负责在重上电第一秒抢 REPL、
    删坏文件、写新代理（一行一条 REPL 命令 + 校验字节数 + 交互验证后才设自启动）；
  - 本机 CanMV 固件的 `sys` **没有 `modules` 属性**（`sys.modules` 会 AttributeError），
    写板端脚本别依赖它；
  - **板端 REPL 两个坑（2026-10-20 踩过）**：
    1. **一次发多行命令会被吞/串行错乱**（`src=...\nf=open(...)\nf.write(...)` 只执行了前
       两行，留下 0 字节 `main.py`）——必须**一行一条**，或用 `;` 拼成单行；
    2. `/sdcard/main.py` 若被固件占用，`open(...,'w')` 报 **EPERM(1)**、
       `os.remove` 报 **EBUSY(16)**、`os.rename` 撞 **EEXIST(17)** ——
       **先 `machine.reset()` 释放句柄**再写，写完必须校验字节数；
  - **实测（2026-10-20，本机）**：`/sdcard/main.py` = 11654 字节，断电/复位后
    自动跑 `agent: 240 frames, 24.6fps ... 360 frames`；ESP32 侧
    `-> K230 $E,START!` / `K230: START_OK` / `[GIMBAL] state=1 frames=24→48→73→97
    bad=0`，`err 62,33 → 12,11` 收敛；代理离线自启后 `agent: 600 frames, 24.6fps`
    （无 JPEG/绘图，比 IDE 预览高）；ESP32 `[GIMBAL] state=1 frames=25 bad=0`，
    pan/tilt 随目标变化；`gmanual`（=R1）后 frames 停在 143 不再涨，
    `gtrack`（=R2）后 143→223→297 —— START/STOP 触发确认可用；
- 下一步：若还要提升 20px 以下的极小靶，需要更多该尺度人工标注；
  40~60px 远距离当前已可用；检测抖动已接近模型上限，再提升应做
  K230 侧"多候选 + 时序关联"（当前只取 top-1，相似物之间会跳）。

### 2026-10-21 Pose 退化修复（`pose_v3`）

用户反馈「接入激光后靶纸/靶心识别率下降、误检上升」。现场用
`k230tool grab` 抓 60 帧实测：部署的 poseB 对 33×49 px 靶纸 **0/60**、
把笔电 **60/60** 误检；spot_v1 56/60 但笔电仍误检 22/60。根因与修法：

1. **训练集尺度错位**：v2_B 正样本 min-side 中位数 100 px，<60 px 只占 7%。
   → `tools/build_pose_dataset_v3.py` 增加整图缩小 0.30~0.65 倍的 zoom-out
   增强（可再叠靶心光斑），<60 px 样本占比升到 43%。
2. **spot_v1 丢了硬负样本**：只继承 v2_B 的 20 张空标注，丢了 farneg 的 80 张。
   → v3 用 `v2_B ∪ farneg_B`（哈希去重，248 正 / 80 负）+ 每张负样本随机裁 2 块
   + 现场帧避开靶纸的硬负裁剪，共 257 张负样本。
3. **`flipud=0.5` 对 Pose 是错的**：ultralytics 垂直翻转后仍套 `flip_idx`
   （只描述左右镜像），关键点语义被打乱；`erasing=0.25` 会擦掉靶心。
   → 训练配方固定 `flipud=0.0, erasing=0.0`（`tools/train_pose_v3.py`，
   `tools/train_pose_target.py` 也已修正默认值）。
4. **板端把 YOLO 的框中心当左上角**：`_pose_postprocess` 里 `x=row[0]/scale`，
   但输出是 `[cx,cy,w,h]` → 框/ROI/`$KT` 框中心整体右下偏 `w/2,h/2`
   （26×36 靶纸偏 11~18 px）。→ 改为 `x=cx-w/2, y=cy-h/2`
   （`yolo_pose.py`、`yolo_preview_ide.py`、离线代理 `gimbal_agent.py` 三处都改）。

结果（`bench_pose.py`，farneg_B test + 合成光斑/缩小变体 + 80 负样本）：

| 模型 | 干净 top1 | 光斑 top1 | 缩小远靶 top1 | 靶心（干净/远靶） | 负样本误检 |
| --- | --- | --- | --- | --- | --- |
| poseB | 100% | 82.6% | 50.0% | 4.18 / 4.73 px | 1.2% |
| spot_v1 | 100% | 100% | 56.5% | 4.03 / 2.48 px | 1.2% |
| **pose_v3** | **100%** | **100%** | **100%** | **1.99 / 1.48 px** | **1.2%** |

- 全新视角（26×38 px，未参与训练）：旧模型 0%，v3 80%（conf 0.25），
  开 `low_refine` 后 95%；板端当前场景 `verify_live.py` **48/48 命中、
  0 误检、靶心 1.5 px、13.3 fps / 推理 31 ms**；
- 新增 `tools/label_scene_frames.py`（多帧中值预标 + 核对图）、
  `tools/bench_pose.py`（多模型/多组指标/负样本）、
  `scripts/verify_live.py`（板端真实 kmodel 统计）；
- **板端模型路径改为 `/sdcard/models/pose_v3.kmodel`**（旧 `target_pose.kmodel`
  被 KPU 占用时写入会 EBUSY，故用新文件名，旧文件保留回退）；
  `link.py` / `yolo_pose.py` / `yolo_preview_ide.py` / `gimbal_agent.py` 默认值已同步；
- **激光标定必须共用**：`board/yolo_pose.py` 也读 `/sdcard/laser_calib.json`
  （2026-10-22 起由 PC `scripts/calib_laser.py` 生成，板端不再自动全图搜索；
  详见文末「2026-10-22 激光识别重构」）；
- 板端低置信复筛：`low_refine=1`，仅对 ≤120 px 的候选、最多 2 个、
  7 倍裁剪放大，失败后冷却 3 帧；PC 实测 80 张负样本误检不变（1/80）。

> **历史章节（2026-10-21 / 10-22）**：以下三节描述的 `board/laser_spot.py`
> 已于 2026-10-22 删除（快照 `0c5ff1b`），当前方案见文末
> 「2026-10-22 激光识别重构」。`$LS` 协议、ESP32 三态 A/B/C、激光 IO 代码保留。

### 2026-10-21 激光 IO / 搜索优化 / 三态追踪

- **（2026-10-22 更新：GPIO5 已弃用，激光常亮；以下 IO 代码保留但当前不驱动激光）**
- **激光笔 IO（ESP32 GPIO5，低电平点亮）**：02 板其余脚全被电机/舵机/PS2/
  K230/循迹占用，GPIO5 是唯一空出的可输出脚；它是 strapping pin（上电必须为高），
  所以按 active-low 接线（NPN+N-MOS），保证 bootloader 阶段不亮。
  `applyLaserMode()` 只在 `GIMBAL_MANUAL`(R1)/`GIMBAL_TRACK`(R2) 点亮，
  `GIMBAL_NONE`/上电熄灭；`carMode`(L1/L2) 完全不影响。实测 USB
  `gnone/gmanual/gtrack/gnone` → `laser on=0/1/1/0`。改线只改
  `PIN_LASER` / `LASER_ACTIVE_LOW`。
- **激光搜索优化**：`laser_spot.py` fixed 模式改为每 `fixed_verify_every=5` 帧
  复核一次，其余帧零搜索；连续 10 次复核失败后每 30 帧在 ±180px 内做 stride=6
  的**受限重捕**，**永不回退全图**。`fixed_allow_reacquire` 只在 R2 代理置 1，
  live/IDE 预览为 0 → 只有 R2 才进入（受限）全局搜索。没有
  `/sdcard/laser_calib.json` 时代理直接关闭 `$LS`，不启动全图标定。
- **帧率**：`jpeg_every=2`、`full_every=9999`（全图 JPEG 单帧约 370ms，默认关；需要全图用 `k230tool grab`）、`stats=15`、gc 每 16 帧、
  ROI 160×120 保持质量；K230 live 实测 **13.6 → 18.2 fps**，激光 11 → 2ms；
  代理约 20fps（含 `$LS`）。
- **三态追踪（已接入控制）**：`$LS` 增加第 6 段 `src`（0 无 / 1 实拍 / 2 标定）。
  A：`aimEverBull` 后连续 10 帧无 bull 或 900ms 无 bull → `setGimbalMode(NONE)`，
  激光高阻关断、K230 STOP；**只复位云台，不改 carMode**。启动给 3s grace，
  从未见 bull 时 3.9s 后也复位。B/C：控制参考点从 `CENTER_X/Y` 改为
  `$LS` 激光位置（src=2 标定 / src=1 实拍），PID/前视/低通/死区结构不变。
  `status` 增加 `ctrlErr=x,y`；`aim` 打印 state/everBull/bullLost/laser/src/conf；
  `kstop` 调试命令用于在 TRACK 中单独停 K230 验证 A。
- **激光关断 = 高阻**：`setLaserOutput(false)` 用 `pinMode(INPUT)`，避免 GPIO5
  误接蜂鸣器时上电/NONE/R1 关断持续鸣叫；`laserpin 0|1|z` 可硬件排查。
- **N 实测**（pose_v3，conf 0.25，bull conf≥0.3）：现场 60 帧 60/60（最大连续
  漏检 0）；新视角 20 帧 16/20（最大连续 2）；80 负样本误检 1/80 →
  **N=10 帧 + 900ms 兜底**，已实现。
- **实测**：`kstop` 后 **902ms** 触发 `AIM: bull lost -> DONE`，云台 NONE、
  激光 OFF、K230 STOP；B/C 在 `aim` 中分别显示 state=2（src=2）/ state=3（src=1）。
- **硬件观察（重要）**：`ctrlErr≈(-27,-13)` 时 pan 1562→1798 持续增大，但靶心
  图像坐标几乎不变（326,281）——**云台/相机可能机械卡住、舵机未动或限位**，
  不是控制律不收敛。先检查 pan 舵机臂/相机支架/供电，再评估收敛。
- **端到端**：ESP32 新固件烧录通过；K230 部署新代理后 USB `gtrack`：
  `START_OK`、`$KT` 202 帧、`bad=2`（启动边界）、`$LS src=2`、
  `laserValid=1`、激光 ON；`gnone` 后 STOP/激光灭。
- 新增文档：`k230/docs/AIM_LASER_SEARCH.md`。

### 2026-10-22 激光预测跟踪 / 防卡死

用户指出“分级搜索只是缩小范围”，要求改成**预测下一帧光斑位置 + 门控搜索**，并解决
找不到激光时的板子卡死。调研/采用 SORT（Kalman+门控）、ByteTrack（track/coast/lost
生命周期）的思路，在 `board/laser_spot.py` 实现 `SpotKalman`：

- 2D 匀速 Kalman（x/y 各 p,v），预测点 ±16px 中心小窗优先，找不到再扩到 ±48px
  门控窗；`fixed_verify_every=5`，其余帧零搜索；
- **全图 `find_blobs` 彻底移除**：无标定/首帧用 160×120 分格捕获，每帧一格；
  丢失只做有界重捕/分格；
- 单帧 `budget_ms=12`（`_scan_window` 逐行检查）+ 调用侧 `laser_budget_ms=25`/
  `laser_cooldown=8` 看门狗；超时跳帧，AI/图传继续；
- 红点/粉点/过曝白核三种阈值；首次锁定与旧标定偏差 >15px 自动回写
  `/sdcard/laser_calib.json`（实测从 375,275 修正到 303.6,233.3）；
- `full_every` 默认 **9999**：640×480 全图 JPEG 单帧约 370ms，会让 fps 周期性
  从 19 掉到 9；需要全图用 `k230tool grab`。

**实测**（`scripts/motion_test_laser.py`，K230 debug + ESP32 R1 激光 + `--full-every 9999`）：
30s 轮 97 帧/16 实测/**19.21fps**/最大 coast 5/最大激光 28ms/遮挡 2.75s 恢复；
24s 轮 77 帧/12 实测/19.24fps/coast 7/遮挡 2.81s 恢复；两轮均**无全图搜索、无卡死**。
`tools/test_spot_tracker.py` 单元测试全过（匀速误差 0.2px、5 帧 coast 7.7px、离群点
拒绝、静止速度 0.47px/s）。

**统计/标定修复（2026-10-22 晚）**：

- 固定分支原先在偏移>15px 时自动写 `laser_calib.json`，移动测试把 calib 写成
  `(307,215)→(0,26)` 的漂移值；现在**只有首次无标定捕获**才回写，固定分支只更新
  内存 `self.fixed`，24s 测试后 calib 仍为 (312.5,230.1)。
- `pred` 改为保留更新前预测；固定分支区分 `status=track`（复核成功）/`calib`
  （coast/丢失）。24s 稳定标定轮：78 帧/13 实测/**54 预测跟踪**/11 丢失；
  预测误差 **mean 2.24px / median 2.02 / p95 3.71 / max 5.73**；19.03fps；
  最大激光 28ms；coast 9；无全图。
- **CanMV “当前固件不支持 REPL 输入”**：板子在工作模式自动跑 main.py/代理或旧预览
  脚本占着 REPL，不是固件问题。先 `mode.sh debug`/3×`gnone` 写 offline_off；仍不行
  按 RESET（offline_off 保留，仍进 debug）；debug 下 IDE 可正常连。

文档：`k230/docs/LASER_PREDICTION.md`；测试脚本：`k230/tools/test_spot_tracker.py`、
`k230/scripts/motion_test_laser.py`。

**2026-10-22 晚回退**：用户要求“不用预测激光，用回原来的算法”，已移除
`SpotKalman`/门控/coast/`pred`/`track`，恢复 `fixed_verify_every=5` + `win_rad/
win_grow` + 受限重捕的原算法；**保留**无全图/160×120 分格捕获/12ms 预算/25ms
看门狗/暗红点阈值/`full_every=9999`。`$LS src` 仍为 1/2，三态 A/B/C 不变。
`tools/test_spot_tracker.py` 随预测层删除；回退后需重新跑基线/运动测试确认
fps 与遮挡（原 24s 预测版的 2.24px/54 track 不再适用）。

### 2026-10-22 激光识别重构（当前方案）

用户明确要求删掉旧激光识别、换新方案。旧 `board/laser_spot.py`（1200 行、
fixed/verify/reacquire/confirm 状态机）已删除并有快照 `0c5ff1b`；
新方案是 **"标定即识别"**：标定一次光斑位置，运行时只在 ±12px 微窗做质心，
不再"找"光斑，也没有全图/分格/重捕。

**硬件前提（用户 2026-10-22 确认）**：
- **激光常亮，GPIO5 已弃用**。ESP32 `setLaserOutput()`/`applyLaserMode()`/
  `laserpin` 代码保留（以后接可控开关可用），但当前对激光没有实际作用；
- 所以标定默认走 **单帧 LAB + PC 人工点选**；只有以后能可靠关激光时才用
  `--pair` 做 ON/OFF 主动差分。

**板端 API 实测（CanMV v1.4.3 / K230，2026-10-22）**：
| API | 耗时 |
| --- | --- |
| `find_blobs(48×48)` | 11~14 ms（固定开销，和 ROI 大小基本无关） |
| `find_blobs(160×120)` | 12~14 ms |
| `find_blobs(640×480)` | **22 ms**（旧文档"0.5~2.3s"是误记） |
| `img.difference()` 全图 | 7 ms |
| `img.binary()` 全图 | 16 ms |
| `img.mean_pool(4,4)` | 3 ms |
| `img.get_statistics(roi)` | ~0 ms |
| 纯 Python bytearray 25×25 / 49×49 | **1 ms / 4 ms** |

**关键结论：不要再用 `find_blobs` 做每帧激光识别**（12ms 起步）；
25×25 的 `img.bytearray()` 直扫只要 1ms，这才是每帧可用的。

**新文件 / 改动**：
- `board/laser_aim.py`：`LaserAim` 类；`predict = calib + k·(s-ref) + off`，
  `sample()` 只扫微窗；状态只有 `ok / calib / lost`；无全图路径；
- `scripts/calib_laser.py`：PC 标定（单帧 LAB + 鼠标点选；可选 ON/OFF 差分；
  `--fit` 多点拟合 `c + k·s`；`--push` 推到板端）；
- `tools/test_laser_aim.py`：PC 单测（命中/偏标定/关激光/距离模型/标定读写）；
- `board/yolo_pose.py`、`board/yolo_preview_ide.py`、`board/gimbal_agent.py`
  三处已切到 `laser_aim`，`$LS` 协议不变；
- `k230tool/link.py` / `mode.py` 的依赖文件名改为 `laser_aim.py`；
- 删除旧算法专用工具：`capture_laser/eval_laser/laser_bench/laser_hardset/
  laser_label.html/make_label_page/replay_laser/test_laser_frame/
  motion_test_laser/check_laser_deploy`；
- 保留 `tools/laser_samples/`（回归数据）、`augment_spot.py` / `eval_spot_robust.py`
  （训练用，让 pose 抗光斑遮挡）、ESP32 `$LS`/AIM/激光 IO；
- 文档：新增 `k230/docs/LASER_AIM.md`，删除 `AIM_LASER_SEARCH.md` /
  `LASER_PREDICTION.md`。

**板端实测（2026-10-22，现场墙 + 真实激光）**：
- 标定 `(306.28, 229.32)`；`LaserAim.sample()` 20 次 **1~2ms**，
  实测 `(306.7,229.7)`、`n=25`、`src=measured`；
- `img.save('/sdcard/laser_aim_dbg.jpg')` + `k230tool pull` 核对：
  十字/圆环正好压在激光点上（板端绘制路径已验证）；
- `k230tool live` 端到端：`latest.json` 的 `laser=(306.7,229.4) src=measured
  calib_dev=0.4px laser_ms=2`；ESP32 `aim` 显示 `laser=306.6,229.1 src=1 conf=356`；
  `loop=67ms`、`infer=32ms`、`uart=7ms`。
- **运动/静止靶基线**（`scripts/motion_test_laser_aim.py`，`latest.json` + ESP32
  `aim/status` 同时采样）：

  | 场景 | 采样 | `$LS src=1` | `calib_dev` p95 | `laser_ms` p95 | 抖动 std | `$KT` fps |
  | --- | --- | --- | --- | --- | --- | --- |
  | 静止（10ms） | 20 | **100%** | 0.4px | 2ms | x 0.13 / y 0.08 | 18.2 |
  | 运动（6ms，旧） | 44 | 56.8% | 2.7px | 6ms（顶预算） | x 1.02 / y 0.88 | 15.4 |
  | 运动（10ms） | 29 | **86.2%** | 16.9px\* | 5ms | x 5.59 / y 2.00 | 17.3 |
  | 运动（靶纸在光斑上 + α-β） | 29 | **96.6%** | 36~48px\* | 9ms（当时预算 10；现 12） | x 4.46 / y 4.35 | 17.3 |

  \* 16.9px 是靶纸移动/倾斜让光斑图像位置真的移动（打白墙时 2.7px），不是噪声；
  `off` EMA / α-β 跟着实测走，窗口中心随之平移，所以仍是 measured。
- **α-β 滤波后的控制误差**（靶纸举在光斑上、手举移动 30s）：
  `|ctrlErrX|` p50/p95 **13.9/88.7 → 10.4/35.2**，`|ctrlErrY|` **22.6/67.3 → 9.5/42.2**；
  `$LS src=1` **86.2% → 96.6%**，`mode=roi` 29/29、coast 0。
  `calib_dev` 36~48px 是手举靶纸让光斑移动，不是标定过期；
  **`need_recalib` 已改为只由丢失/连续拒识触发，不再按 offset 触发**
  （否则手举靶会一直误报）；重新命中会清掉。
- 预算从 10ms 放到 **12ms**：运动靶 `laser_ms` p95=9ms，10ms 偏紧。
- **6ms 预算是主要瓶颈**：运动轮 `laser_ms` p95 正好 6ms，预算截断直接变成
  `src=2`；改 10ms 后静止 100%、运动 86.2%；10ms 在 α-β 运动轮 p95=9ms 偏紧，
  最终 `budget_ms` 默认 **12**。
- `need_recalib` 现在在**连续丢失**或 **`max_step` 连续拒绝**时也会置 1
  （之前只在缓慢漂移超过 `calib_warn` 时置），agent/preview 会打印告警；
  `$LS` 仍发 `src=2`（三态 B 的标定参考），`conf=0`，`latest.json` 有 `need_recalib`。
- 靶纸表观边长 `s` 改为 **4 角点平均边长 + 内部 EMA(0.3)**，不再用检测框
  `sqrt(w*h)`；角点不可信时才退回框。`k·s` 的点标定不会被框抖污染。
- IDE 预览的激光标记改成**中间留 6px 的十字 + 圆环**（不再用 `draw_cross`
  压住光斑）；板端 `img.save('/sdcard/laser_aim_dbg2.jpg')` + pull 已核对，
  圆环套住光斑、中心可见。
- **`mode work` 假失败修复**：替换 `/sdcard/main.py` 后板子可能复位、REPL 不吐
  `SZ`；`mode._push_replace` 现在会重连并按 md5 复核一次，文件对就继续，
  不再误报（这次曾骗我去按 RESET）。
- **运动滤波/预测（2026-10-22 加）**：`laser_aim.py` v2026-10-22.2 在**微窗测量值**
  上接 α-β 常速滤波（不是旧 `laser_spot.py` 的 Kalman 搜索层）：`update`
  用 α=0.45/β=0.10，输出 `pos + v·lead_ms`（默认 50ms）补偿相机→UART→ESP32
  延迟；微窗中心用无前瞻预测；窗口随速度最多 +8px；丢失先 coast 8 帧再回标定；
  `dt` 下限 20ms 防 `β/dt` 放大噪声。`latest.json` 的 `laser` 多了 `vx/vy` 和
  `mode=roi/coast/calib`；`$LS` 的 coast 仍映射 `src=2`。单测见
  `tools/test_laser_aim.py` 的 `tracker-cv`（50ms 前瞻误差 <2px）与 `coast`。
  **待实测**：静止/运动靶复测（需先断开 CanMV IDE 占用串口）。

**标定文件**：
```json
{"x":306.28,"y":229.32,"kx":0.0,"ky":0.0,"ref":0.0}
```
`x,y` 是 640×480 原图坐标；`kx,ky` 是 `c + k·s` 的距离项（`s`=靶纸表观边长，
pose 每帧给），单距离标定时为 0。没有这份文件时三处都关闭 `$LS`（run=0），
不做任何自动全图搜索。

**踩坑 / 更正**：
- **`from media.sensor import *` 会把 `sys` 覆盖成 `mpp.sys`**（没有 `.path` /
  `.modules`）。之前 AGENTS 里"本机固件 sys 没有 modules"其实是这个原因：
  真正的 `sys` 有 `modules`（实测 `hasattr(sys,"modules")=True`）。
  写板端脚本时：`import sys` 要放在 `from media.sensor import *` **之后**，
  或者先 `_sys = sys` 存引用；`gimbal_agent.py` 的 `sys.path` 块一直静默失效
  （`/sdcard` 本来就在默认 path 里，所以没出问题）。
- 真实小光斑（5~10px）的 LAB 阈值环只有 2~3px 宽，**3×3 `MORPH_OPEN` 会把它
  直接抹掉**（实测 19px → 0px）。标定工具的形态学只做 `close`，不做 `open`；
  5×5 close 才能把被 JPEG 压断的 2~3 块重新连成一个连通域。
- `/sdcard/laser_calib.json` 被占着时 `open(...,'w')` 报 **EPERM**、
  `os.remove` 也 EPERM（不是 EBUSY）；`k230tool mode reset` 后即可写。

**待办 / 注意**：
- 工作模式下 `/sdcard/main.py` 仍是旧代理时，新标定不会被旧 `laser_spot.py`
  使用；切工作模式前先 `python3 -m k230tool mode work --agent board/gimbal_agent.py`
  重新部署（本文件已改好，但板端是否重部署取决于现场）；
- 激光笔如果以后换可控开关，优先用 `calib_laser.py --pair` 做 ON/OFF 差分，
  比单帧点选更稳；
- 多距离 `c + k·s` 的第二点需要在板上同时拿到靶纸表观边长 `s`（`latest.json`
  的 `detections[0].box`）与光斑坐标；当前只有单距离标定（k=0）。

### 2026-10-22 云台闭环瞄准 / PID（当前结论）

K230 只给 `$KT`（靶心）和 `$LS`（激光），ESP32 `updateTrack()` 用
`err = (bull + 前视) - laser` 跑 PID。真实瞄准误差用 PC 侧
`bull - laser`（同一帧 `latest.json`）统计，`scripts/motion_test_laser_aim.py`
的 `aimErr`；`ctrlErr` 是前瞻修正后的误差，两者差几 px 正常。

**A/B 实测（静止靶，15s/轮，详见 `k230/docs/GIMBAL_AIM.md`）**：

| 组 | PID | 真实 `aimErr` | `ctrlErr` | 结果 |
| --- | --- | --- | --- | --- |
| A 当前默认 | PAN 2.5/0.3/0.15，TILT 2.5/0.2/0.10 | 稳定后约 **(-5.6,-8.4)px**；含抬升过渡 p50 (10.2,13.1) | p50 **(6.3,8.4)** | **稳定** |
| B AGENTS 最终 | PAN 9/0/0.5，TILT 9/0/0.6，smooth .006，trk .40/.25 | 发散：`ctrlErrY≈-219`、bull 出画、激光 0% | — | **不可用** |

- **根因：`WBAND_PX` 是 8px，不是 AGENTS 旧记录的 40px**。kp=9 那套是配
  40px 死区整定的；8px + kp=9 + ~100ms 延迟会振荡，把云台推到限位。
  **不要在 8px 死区上直接用 kp=9**。要试 kp>4 必须先改 `WBAND_PX` 再重新整定，
  但 40px 死区会让“激光到靶心中央”变差。
- 已恢复 A 组 PID（`pid/pid2/smooth/trk` 命令只改 RAM；上电默认就是 A）。
  B 组的运行值已用 `pid 2.5 0.3 0.15` / `pid2 2.5 0.2 0.1` /
  `smooth 0.02` / `trk 0.35 0.20` 还原，并用 `s1 1500`/`s2 1500` 回中。
- **机械限位**：tilt 1200/1800；靶纸太低时 tilt 顶 1200，纵向固定差 40~100px。
  靶纸必须放在激光束里、画面中间偏上；靶纸出画时 `$KT run=0`，三态 A 会复位。
- **运动**：α-β 后 `$LS src=1` 96.6%、`ctrlErr` p95 88.7/67.3 → 35.2/42.2；
  剩余 p95 来自 K230 ~15fps + ~100ms 链路延迟和手举靶的物理运动，不是 PID 单方面。
- **激光在线 re-zero**：`laser_aim.py` 的 `rezero_*`，光斑稳定 8 帧后以 3%/帧
  把标定先验拉向实测，适配靶纸举高/换姿态；但**不能救“冷启动偏 >24px”**
  （微窗看不到），那要在 PC 上重标。
