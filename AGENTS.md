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
- **激光标定必须共用**：`board/yolo_pose.py` 现在也读 `/sdcard/laser_calib.json`
  （首次自动标定后写回）。没有标定文件时，复杂画面下每次上电要做 12 次
  全图搜索（单次 0.5~2.3 s），整条流会掉到 0.5~3 fps；
- 板端低置信复筛：`low_refine=1`，仅对 ≤120 px 的候选、最多 2 个、
  7 倍裁剪放大，失败后冷却 3 帧；PC 实测 80 张负样本误检不变（1/80）。
