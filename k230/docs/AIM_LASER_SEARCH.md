# 激光 IO / 搜索优化 / 三态追踪接口（2026-10-21）

> 本轮范围：**① 激光笔 IO 与 R1/R2 门控；② 激光搜索优化；③ 三态追踪；④ 帧率优化。**
> 三态 A/B/C 已接入 `updateTrack()`；PID/前视/低通/死区结构不变，
> 只把瞄准参考点从图像中心改成 `$LS` 激光位置。

## 1. 激光笔 IO（ESP32，GPIO5）

### 1.1 引脚选择

02 板（`hardware/esp32-02/底板引脚图.png`）上其余脚都已被占用：

| 用途 | GPIO |
| --- | --- |
| 电机 PWM/IN | 33/32、2/4、13/26、STBY 25 |
| 舵机 | PAN 27 / TILT 14 |
| PS2 | DAT 19 / CMD 18 / ATT 15 / CLK 23 |
| K230 UART2 | RX 16 / TX 17 |
| 8 路循迹 I2C | SDA 21 / SCL 22 |

30Pin 排针上唯一空出的可输出脚是 **GPIO5**。它是 strapping pin（上电必须为高），
所以固件默认按**低电平点亮**，保证 bootloader 阶段（GPIO5=高）激光不亮：

```
GPIO5 --1k--> NPN 基极
NPN 集电极 --> N-MOSFET 栅极（10k 上拉到 3.3V）
N-MOSFET 源极 GND、漏极 --> 激光笔负极；激光笔正极 --> 5V/3.3V（按模块）
GPIO5 高 = NPN 导通 = 栅极低 = 灭；GPIO5 低 = 栅极高 = 亮
```

- 模块本身是 active-low 使能时，可直接接 GPIO5；
- active-high 模块必须加反相/上拉，否则上电会短暂点亮；
- 改线后只改 `PIN_LASER` / `LASER_ACTIVE_LOW` 两个常量。

### 1.2 门控逻辑

```cpp
void applyLaserMode() {
  setLaserOutput(gimbalMode == GIMBAL_MANUAL || gimbalMode == GIMBAL_TRACK);
}
```

| 模式 | 按键 | 激光 | 说明 |
| --- | --- | --- | --- |
| `GIMBAL_NONE` | 上电 / `gnone` | **灭** | 上电默认，`setup()` 里先拉灭 |
| `GIMBAL_MANUAL` | R1 / `gmanual` | **亮** | 云台手动 |
| `GIMBAL_TRACK` | R2 / `gtrack` | **亮** | K230 识别 + 云台跟踪 |
| `CAR_MANUAL` / `CAR_LINE` | L1 / L2 | 不改变 | 与 R1/R2 并行，互不影响 |

`setGimbalMode()` 里调用 `applyLaserMode()`，所以 USB `gnone/gmanual/gtrack`
和手柄 R1/R2 走的是同一条路径。

### 1.3 实测（2026-10-21，`/dev/ttyUSB0`）

```
> laser      LASER pin=GPIO5 on=0 active_low=1 gimbalMode=0
> gmanual    GIMBAL: MANUAL ...
> laser      LASER pin=GPIO5 on=1 active_low=1 gimbalMode=1
> gtrack     GIMBAL: TRACK ...
> laser      LASER pin=GPIO5 on=1 active_low=1 gimbalMode=2
> gnone      GIMBAL: NONE ...
> laser      LASER pin=GPIO5 on=0 active_low=1 gimbalMode=0
```

> 本次测试时 PS2 接收器 `pad=0`（未识别），所以用 USB 命令验证门控；
> 插好接收器后 R1/R2 走同样的 `setGimbalMode()`，行为一致。

## 2. 激光搜索优化

### 2.1 旧问题

`laser_spot.py` 的 fixed 模式原本**每帧**都在 ±12~60px 窗口逐像素扫（约 5~11ms）；
没有标定时每次上电最多 12 次全图 `find_blobs`（单次 0.5~2.3s），复杂画面帧率
掉到 0.5~3fps。

### 2.2 现在的策略

1. **固定标定 + 每 5 帧复核**：相机与激光笔固连，光斑在画面里基本是常量。
   `fixed_verify_every=5`，4/5 帧直接返回标定位置（**零搜索**），第 5 帧才在
   12~60px 小窗复核（~5ms）。
2. **受限重捕（不是全图）**：连续 `fixed_reacquire_after=10` 次复核失败后，
   每 `fixed_reacquire_every=30` 帧在 ±`fixed_reacquire_rad=180` px 内做一次
   **stride=6** 的稀疏重捕（~20ms），找到就更新标定；找不到继续用旧标定。
3. **只有 R2 打开重捕**：`fixed_allow_reacquire` 只在 R2/START 的离线代理里
   置 1；`k230tool live` 和 IDE 预览默认 0。K230 本身只在收到 `$E,START!`
   （ESP32 的 R2）后才跑代理，所以**全局/受限搜索只发生在 R2**。
4. **没有标定就不搜**：代理启动时读 `/sdcard/laser_calib.json`；没有就关闭
   `$LS` 上报（不做事后全图搜索）。标定应在 IDE 预览里做一次，文件两处共用。

### 2.3 配置

`board/laser_spot.py`：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `fixed_verify_every` | 5 | 每 N 帧复核一次 |
| `fixed_allow_reacquire` | 0 | 仅 R2 代理置 1 |
| `fixed_reacquire_after` | 10 | 连续失败 N 次才重捕 |
| `fixed_reacquire_every` | 30 | 重捕帧间隔 |
| `fixed_reacquire_rad` | 180 | 重捕半径（不是全图） |
| `fixed_reacquire_stride` | 6 | 稀疏步长 |
| `calib_tries` | 6 | 首次自动标定最多 6 次（仅无标定文件的调试场景） |

## 3. 三态追踪（已接入控制）

### 3.1 协议

`$KT` 不变。`$LS` 增加第 6 段 `src`：

```
$LS,<seq>,<run>,<x*10>,<y*10>,<conf*1000>,<src>!
    src: 0=无位置  1=本帧实拍  2=laser_calib.json 标定位置
```

旧 ESP32 的 5 字段 `sscanf` 会忽略第 6 段，向后兼容。

### 3.2 状态机（`updateTrack()` 顶部）

```cpp
constexpr uint32_t AIM_LOST_FRAMES = 10;
constexpr uint32_t AIM_LOST_MS = 900;
constexpr uint32_t AIM_START_GRACE_MS = 3000;
constexpr uint32_t AIM_LASER_TIMEOUT_MS = 500;
constexpr int AIM_BULL_MIN_CONF = 300;
enum { AIM_IDLE, AIM_BULL_LOST, AIM_NO_LASER, AIM_READY };
```

| 状态 | 进入条件 | 行为 |
| --- | --- | --- |
| A 靶心丢失 | `aimEverBull && (aimBullLostFrames >= 10 \|\| now-aimLastBullMs > 900ms)`，或从未见 bull 且已过 3.9s | `setGimbalMode(GIMBAL_NONE)`：激光关（GPIO5 高阻）、K230 STOP、云台保持；**不动 carMode** |
| B 有靶心、无实拍激光 | 有 bull，`$LS src==2`（标定）或 laser 无效/过期 | 用 `aimLaserX10/Y10`（`laser_calib.json`）作为瞄准参考 |
| C 两者都有 | 有 bull，`$LS src==1`（实拍） | 用实拍激光位置作为瞄准参考；K230 做局部复核 + 受限重捕 |

- **A 的 `aimEverBull` 门控**：上电/R2 刚进入时 K230 要 1.5s 才开相机，
  不能立刻判完成；先给 `AIM_START_GRACE_MS=3000`，且只有“见过靶心再丢”
  才按 900ms 立即复位。从未见靶心时 3.9s 后也会复位。
- **“上电复位模式”只复位云台/激光/K230**；`carMode` 保持，L1/L2 并行不受影响。
- **B/C 的控制参考点**从 `CENTER_X/Y` 改成激光位置：
  ```cpp
  laserFresh = aimLaserValid && (now - aimLaserMs < 500ms);
  refX = laserFresh ? aimLaserX10/10.0 : CENTER_X;
  errX = (tx + vxF*lead) - refX;   // tx 是靶心（无靶心时退回框中心）
  ```
  PID / 前视 / 低通 / 死区结构不变。`status` 新增 `ctrlErr=x,y`；
  `aim` 打印 `state / everBull / bullLostFrames / laser= / src= / conf=`。
- **激光关断 = 高阻**：`setLaserOutput(false)` 执行 `pinMode(PIN_LASER, INPUT)`。
  这样即使 GPIO5 上误接了蜂鸣器，上电/NONE/R1 关断时也不会驱动它。
- 调试命令 `kstop`：TRACK 中只给 K230 发 STOP、不改 `gimbalMode`，
  用于验证状态 A 的 900ms 自动复位。

### 3.3 N 的实测依据

`pose_v3`，conf 0.25，bull kpt conf≥0.3：

| 序列 | 命中 | 最大连续漏检 |
| --- | --- | --- |
| 现场 60 帧（静态靶） | 60/60 | 0 |
| 新视角 20 帧（26×38px） | 16/20 | 2（帧 0、1，相机未稳定） |
| 80 张负样本 | 误检 1/80（box 0.294） | — |

取 **N=10 帧 + 900ms**（13~20fps 下 0.5~0.75s），对最大连续漏检 2 有 5 倍余量；
`aimLastBullMs` 的 900ms 兜底覆盖 K230 停止发帧/掉线。

## 4. 帧率优化

| 参数 | 旧 | 新 | 作用 |
| --- | --- | --- | --- |
| `fixed_verify_every` | 每帧扫 | 5 | 激光 11ms → ~2ms/帧 |
| `jpeg_every` | 1 | 2 | JPEG 18ms → ~9ms/帧 |
| `full_every` | 15 | **9999（默认关）** | 全图 JPEG 单帧 ~370ms，会周期性卡顿；需要全图用 `k230tool grab` |
| `stats` | 10 | 15 | 少一次格式化 |
| `gc` | 每 8 帧 | 每 16 帧 | 少一次 1~2ms |
| `laser_global` | — | 0（live/预览） | 非 R2 不做重捕 |

**实测（K230 USB live，640×480 q70，pose_v3，无靶场景）**：

| 版本 | 板端 fps | infer | laser |
| --- | --- | --- | --- |
| 优化前 | 13.6 | 31ms | 11ms |
| **优化后** | **18.2** | 31~36ms | **2ms** |

图像质量：目标 ROI 仍为 **160×120 @ q70**（`roi_min=160/roi_max=256`），
PC 端 `latest.jpg` 约 9~10fps；训练/抓图用 `k230tool grab` 直接取 640×480 原图，
IDE 预览仍是 640×480 全图缩放。若只追求 PC 推流帧率，可把 `--roi-min 128`
或 `--jpeg-every 3`，但会牺牲调试清晰度。

## 5. 端到端验证（2026-10-21）

ESP32 烧录新固件后，K230 部署新代理，USB 发 `gtrack`（=R2）：

```
K230: START_OK
[GIMBAL] frames=1 → 20 → 37 → ... → 202  bad=2
bull=392,284 conf=998   laser=ON laserValid=1 lsrc=2
AIM-IFACE state=0 (TODO not wired) bullLostFrames=0 laserValid=1
          laser=348.0,229.0 src=2 conf=1000
```

- `$KT`、`$LS` 均到达；`$LS src=2` 证明代理在无实拍光斑时用了标定位置；
- 激光随 R2 点亮、`gnone` 熄灭；
- 2 个 `bad` 出现在启动边界（部分行），稳定后 `bad` 不增长；
- 三态 A/B/C 已接入；A 用 `kstop` 验证 902ms 自动复位；B/C 的 `src=2/1` 与
  `ctrlErr` 可在 `status/aim` 观察。
- **硬件观察**：测试中 `ctrlErr≈(-27,-13)` 时 pan 从 1562 持续加到 1798，但
  靶心图像坐标几乎不变（326,281）——说明**云台/相机可能机械卡住、舵机没动或
  限位**，不是控制律不收敛。需要检查 pan 舵机臂/相机支架/供电后再评估收敛。

## 6. 相关文件

| 文件 | 改动 |
| --- | --- |
| `firmware/ESP32_Gimbal_K230/ESP32_Gimbal_K230.ino` | GPIO5 激光、`$LS` 解析、`aim` 接口 |
| `k230/board/laser_spot.py` | `verify_every`、受限重捕（stride）、`src` 字段 |
| `k230/board/yolo_pose.py` | `$LS` 6 段、标定回退、fps 参数 |
| `k230/board/yolo_preview_ide.py` | `LASER_GLOBAL=0`、`AI_EVERY=2`、`$LS` |
| `k230/board/gimbal_agent.py` | R2 代理 `$LS`（fixed）、`$KT conf` 改为靶心置信度 |
| `k230/k230tool/link.py` / `live.py` | 默认 fps/激光参数 |
