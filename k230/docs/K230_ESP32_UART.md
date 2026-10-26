# K230 ↔ ESP32 云台 UART 协议

## 1. 接线

| K230（CanMV） | ESP32 云台主控 | 说明 |
| --- | --- | --- |
| 12Pin 第5脚 / GPIO32 / UART3_TX | GPIO16 / UART2_RX | K230 发目标数据 |
| 12Pin 第3脚 / GPIO33 / UART3_RX | GPIO17 / UART2_TX | ESP32 回应答 |
| 12Pin 第4脚 / GND | GND | 必须共地；5V 不要接 |
| — | 舵机 5V / GND | 舵机独立 5V 电源，不要从 ESP32 3.3V 取电 |

- 电平 3.3V，115200 8N1。
- K230 用 `machine.UART(3)` + FPIOA 映射 **GPIO32/GPIO33**（12Pin 口），
  这是**唯一实测可用**的接线；USB 旁 4Pin 口的 UART1（GPIO9/GPIO10）实测收不到。
- ESP32 使用 `HardwareSerial(2)`，RX=16、TX=17。

## 2. 协议

### K230 → ESP32（每帧一条，约 10~50Hz）

```text
$KT,<seq>,<run>,<cx>,<cy>,<bx>,<by>,<conf>,<laser>!
```

| 字段 | 含义 |
| --- | --- |
| `seq` | 递增序号 |
| `run` | 1=有目标可跟踪；0=无目标 |
| `cx,cy` | 目标框中心（640×480 像素） |
| `bx,by` | 靶心中心（或"压靶心"后的瞄准点）；无效时 `-1` |
| `conf` | 置信度 0..1000 |
| `laser` | 1=本帧检测到红色激光落点；0=未检测到（自动射击的触发前提） |

共 **8 个数字字段**（2026-10-24 起；`laser` 为新增，旧 7 字段的 K230 帧 `laser` 自动按 0 处理）。`!` 结束，后跟 `\n`。

### ESP32 → K230

```text
$E,ACK,<seq>,<state>,<pan_us>,<tilt_us>!
```

`state`：0=空闲，1=跟踪，2=已对准，3=数据超时。

## 3. ESP32 固件

源码：`ESP32_Gimbal_K230/ESP32_Gimbal_K230.ino`

只保留：
- UART2 接收 `$KT`、发送 `$E,ACK`；
- PAN=GPIO27、TILT=GPIO14 的 50Hz 舵机 PWM；
- USB 串口调试命令。

删除：PS2、TB6612 电机、循迹 I2C、模式切换、急停电机等。

### 编译/上传

```bash
arduino-cli compile --fqbn esp32:esp32:esp32 ESP32_Gimbal_K230
arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32 ESP32_Gimbal_K230
```

### USB 调试命令（115200）

| 命令 | 作用 |
| --- | --- |
| `s1 <us>` | 直接设置 PAN（740~2260） |
| `s2 <us>` | 直接设置 TILT（1200~1800） |
| `pan <±us>` / `tilt <±us>` | 相对微调 |
| `sh` | 回中 1500/1500 |
| `auto` / `manual` | 自动跟踪 / 手动 |
| `sweep` | 扫动测试（约 4 秒） |
| `status` | 查看状态 |
| `fire` | 开启自动射击：识别到激光每 5s 一发（波箱 PWM GPIO25） |
| `cease` | 停止自动射击 |
| `$KT,...` | PC 模拟一帧 K230 数据，测试解析与云台 |

## 4. K230 端

- `board/yolo_pose.py`：USB 推流时同时通过 UART3（GPIO32/33）发送 `$KT`。
- `board/yolo_preview_ide.py`：CanMV IDE 预览时同时发送 `$KT`。
- `board/uart_test_k230.py`：独立 UART 测试（默认接 ESP32；把 TX/RX 短接可 loopback）。

打开 CanMV IDE 运行 `uart_test_k230.py`：
- 接 ESP32：串口终端应看到 `RX $E,ACK,...`；
- loopback：应看到 `RX $KT,...` 回显。

### 为什么不用 4Pin 口

YAHBOOM K230 的 4Pin UART1（GPIO9/GPIO10）在转接板/丝印层面实测无法连通
（K230 侧拉低时 ESP32 `probe16` 仍为 1，收发两个方向都是 0 帧），改用 12Pin
明确引出的 UART3 后一次通过。

## 5. 云台方向/参数

参数集中在 `ESP32_Gimbal_K230.ino` 顶部，**当前生效值与调参建议见第 7 节**。

- 追反了：把对应 `PAN_SIGN`/`TILT_SIGN` 改为 `-1`；
- 抖动：增大死区 / 降低 `GAIN` / 加大阻尼；
- 反应慢：提高 `GAIN` / `MAX_RATE`，但先确认机械行程。

## 6. 当前实测状态（2026-10-19）

- K230 UART1 自环（GPIO9↔GPIO10）：通过，能收到自己发的 `$KT`；
- ESP32 UART2 自环（GPIO16↔GPIO17）：通过；
- **已连通（2026-10-19 复测）**：走 **12Pin UART3**，
  K230 GPIO32(TX) → ESP32 GPIO16(RX)、K230 GPIO33(RX) ← ESP32 GPIO17(TX)、GND 共地，
  115200 8N1 全双工：20 帧 `$KT` 全部被 ESP32 解析（`frames=20 bad=0`），
  20 条 `$E,ACK` 全部回到 K230，云台随目标偏移自动修正（PAN 1500→1158）。
- 之前 0 帧的原因：**K230 侧接线插错**（不是 GPIO16 损坏、不是固件 RX 引脚）。
  诊断时 `probe16` 在 K230 GPIO32/GPIO33 拉低时始终读 1，即为该症状。
- **4Pin 口 UART1（GPIO9/10）实测仍不通**，不要再走该口；K230 端保持
  `UART_ID=3, TX=32, RX=33`（`uart_test_k230.py`、`yolo_pose.py`、
  `yolo_preview_ide.py` 的默认值均已如此）。
- 注意：K230 GPIO33 只能做 UART3_RXD / GPIO，`set_function(33, UART3_TXD)`
  会报 `set pin func failed`，所以 TX 必须用 GPIO32。

## 7. 云台控制链路（2026-10-19 联调结论）

### 7.1 三个必须知道的坑

1. **IDE 预览曾经完全不发帧**：`yolo_preview_ide.py` 里 `_init_uart()` 定义了却
   从未调用，`_uart` 恒为 `None`，`_send_uart()` 静默 return。表现是"检测正常、
   舵机不动"。已在 `main()` 里补上调用，并在未初始化时打印一次性告警。
2. **`cx,cy` 曾经是框左上角**：协议要求框中心，两版板端脚本原来直接发 `x,y`。
   已统一改成 `x + w/2, y + h/2`。
3. **`ledcWrite()` 每次阻塞约 40ms**（core 3.3.10，实测 20 次=799940µs）。
   两个舵机一次 `applyServos()` 就是 80ms，控制环被迫 25Hz，而 `dt` 仍按 10ms
   积分 → 实际舵速只有设定值的 1/4。改成直接写寄存器（约 0.6µs/次）：

   ```cpp
   constexpr uint8_t CH_PAN = 0, CH_TILT = 1;   // 配合 ledcAttachChannel()
   LEDC.channel_group[0].channel[ch].duty.duty = duty << 4;  // 20bit，低 4 位小数
   LEDC.channel_group[0].channel[ch].conf1.duty_start = 1;
   ```

   改完控制环 66kHz、舵机 100Hz 更新。

### 7.2 当前控制律（图像系 PID + 速度自适应前视）

```cpp
PAN  : kp=2.0  ki=1.0  kd=0.4      // 输出 us/s，输入像素误差
TILT : kp=2.0  ki=0.5  kd=0.4
AIM_DEADBAND = 8 px
PAN_MAX_RATE 2600 us/s / TILT_MAX_RATE 1000 us/s
PAN_ACCEL 40000 / TILT_ACCEL 20000 us/s^2
PID_D_FILTER 0.35
AIM_CTRL_MS 10                     // 100Hz，dt 用真实经过时间
```

目标预处理：3 点中值 → 一阶低通 α=0.25 → 离群点抑制（单帧跳变 >80px 先忽略，
连续 4 帧才接受）。

**前视预测**：`err = (x + vx·leadEff) − 320`，其中
`leadEff = trkLead(0.2s) × clamp(|vx|/500, 0.7, 1.2)`。
慢速时前视小（不抖），高速时前视大（补延迟）。实测高速跟踪 RMS 从
99.6px（无前视）降到 37~49px。

### 7.3 整定结果与实测

整定台（开发脚本已随清理删除，源码在归档 tar 里）：`scratch/pid_tune.py`（阶跃/正弦/二维）、`scratch/track_sim2.py`
（带 N 帧视觉延迟，更接近真实链路）。带 135ms 延迟的仿真结果：

| 速度 | pan RMS | tilt RMS |
| --- | --- | --- |
| 慢速 0.1Hz | 8.4px | ~16px |
| 中速 0.25Hz | 15.6px | ~18px |
| 高速 0.5Hz | 48.9px | ~20px |

关键结论：
- **仿真与真机差别很大**：仿真按 k=0.6px/µs 整定出 kp=7~9，真机上 kp≥4 就会
  把云台怼到限位并丢失目标，**真机稳定值是 kp=2**（静止 |ex| 9px / |ey| 7px，
  死区占比 45%，pan 速度中位 0）；
- 真机上加大前视（α=0.35 / lead=0.35）反而更差（84px vs 69px），保持 0.25/0.20；
- 移动目标 |ex| 中位 69px，滞后主要来自 ~20fps 检测 + ~100ms 链路延迟；
- 世界系（角度域）速度前馈方案已试过并放弃：依赖"拍摄时刻的云台角度"，
  capdel 差 30ms 就发散，而真实链路拿不到精确时间戳。

真机验收（`scratch/track_eval.py`，已删除，20s/轮）：

| 场景 | \|ex\| 中位 | \|ex\| P95 | \|ey\| 中位 |
| --- | --- | --- | --- |
| 静止靶纸 | **6px** | 49px | 14px |
| 缓慢移动 | **21px** | 56px | 14px |
| 快速任意方向 | **53px** | ~160px | 57~73px |

### 7.3.1 在线调参

| 命令 | 作用 |
| --- | --- |
| `pid <kp> <ki> <kd>` | PAN 增益 |
| `pid2 <kp> <ki> <kd>` | TILT 增益 |
| `trk <alpha> <lead>` | 目标平滑系数 / 最大前视（秒） |
| `pidshow` | 打印当前增益 |

### 7.4 自检命令（ESP32 USB 115200）

| 命令 | 作用 |
| --- | --- |
| `ledcdump` | 打印 LEDC 通道寄存器 + 定时器计数（`moving=1` 表示 PWM 在跑） |
| `pulse <pin>` | 把舵机信号跳线到该脚，回读 PWM 高电平宽度（应为 ~500~2500µs） |
| `s1api <us>` | 用老的 `ledcWrite` 路径设 pan，用于对照 |
| `sweep` | 双轴扫动，验证机械/供电 |


## 8. 测试顺序

1. ESP32 只接 USB，发 `sh`、`s1 1200`、`s2 1600`，确认两个舵机各自动作；
2. 发 `sweep`，确认云台扫动无卡顿、无堵转；
3. K230 跑 `uart_test_k230.py`，确认 K230 TX 有输出；
4. 接好三根线（TX/RX/GND），再看 K230 RX 是否收到 `$E,ACK`；
5. 运行 `yolo_preview_ide.py`，把靶纸放到画面中，观察云台是否朝靶心方向修正；
6. 方向不对改 `PAN_SIGN`/`TILT_SIGN`，抖动调死区/增益。
