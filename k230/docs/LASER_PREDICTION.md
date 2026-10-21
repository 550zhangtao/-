> **2026-10-22 晚：按用户要求已回退。** 本文件保留为 Kalman 预测的实验记录；
> 当前 `laser_spot.py` 已恢复 fixed verify 原算法（保留无全图/分格/预算/看门狗/
> 暗红点阈值；`SpotKalman`、pred/coast/track 已移除）。下面的预测内容不再对应当前代码。

# 激光光斑预测跟踪与防卡死（2026-10-22）

## 1. 为什么不再“逐级扩大搜索”

原实现是分级搜索：小窗 → 大窗 → 全图 `find_blobs`。在复杂画面里全图一次要
0.5~2.3s，主循环被拖住，表现就是“找不到激光 → 板子像死锁”。另外 `full_every=30`
的全图 JPEG 单帧约 370ms，也会周期性掉帧。

更成熟的做法是 **运动预测 + 门控 + 丢失/重捕生命周期**：

1. 用匀速 Kalman 预测下一帧光斑位置；
2. 只在预测点周围的门控窗里找；
3. 没找到就 coast（沿用预测）若干帧；
4. 连续丢失才做**有界**重捕/分格捕获；
5. 所有搜索都有硬时间预算，超时就返回预测/空，绝不让单帧拖死主循环。

## 2. 开源参考

| 方案 | 可借鉴点 | 链接 |
| --- | --- | --- |
| SORT | 匀速 Kalman + Mahalanobis 门控 + 匈牙利关联 | https://github.com/abewley/sort |
| ByteTrack | track / coast / lost 生命周期，低分检测二次关联 | https://github.com/ifzhang/ByteTrack |
| DeepSORT | 级联匹配 + 轨迹确认 | https://github.com/nwojke/deep_sort |
| OpenCV Kalman | 标准 predict/update 公式 | https://docs.opencv.org/4.x/dd/d6a/tutorial_track_kalman.html |
| OpenCV Tracking API | KCF/CSRT/MOSSE 等相关滤波跟踪器 | https://docs.opencv.org/4.x/d9/df8/group__tracking.html |
| Laser spot + optical flow + Kalman | 光斑检测/跟踪的论文方案 | https://global-sci.org/index.php/jics/article/download/15374/30502/31732 |
| filterpy | Python Kalman 参考实现 | https://github.com/rlabbe/filterpy |

**为什么不用 KCF/CSRT**：K230 板端没有 OpenCV，CSRT 的内存/算力也撑不住；光斑是
单点、运动近似匀速，2 状态/轴的 Kalman + 小 ROI 就够，SORT 的运动模型 + ByteTrack
的生命周期已经覆盖主要问题。

## 3. 实现（`board/laser_spot.py`）

### 3.1 SpotKalman

- 状态：`(px, vx)` / `(py, vy)`，两轴解耦；
- `predict(t)`：`p += v*dt`，协方差按 `F P F^T + Q` 传播；
- `update(z)`：标量 Kalman，`H=[1,0]`；
- `gate_ok(z)`：`|innov| <= k*sqrt(P00+r)` 且 `<= max_gate`；
- `gate_radius()`：`k*sqrt(P00+r)`，限制在 8~48px；
- 参数：`kf_q=40`、`kf_r=4`、`max_gate=48`；初始 `P00=r*64`（标定不确定度
  ~48px，覆盖标定漂移）。

### 3.2 搜索生命周期

| 状态 | 行为 |
| --- | --- |
| 已锁定 / 复核帧 | 在预测点 **±16px 中心小窗** 先扫；没找到再扩到 ±48px 门控大窗；stride=2 |
| 非复核帧 | 直接返回 Kalman 预测，零搜索（`fixed_verify_every=5`） |
| 连续复核失败 | coast；`coast` 达到 10 后回到标定位置；R2 代理可做 ±180px 受限重捕 |
| 无标定 / 首帧 | 160×120 **分格捕获**，每帧只扫一格，绝不扫全图 |
| 长时间丢失 | 只做有界重捕/分格，不进入全图 `find_blobs` |

**中心小窗很关键**：`_scan_window` 从窗口左上角开始扫，而 `t0` 包含前面的
`_estimate_base`（约 8ms）；12ms 预算下，大窗可能还没扫到中心的点就超时了。
实测这是“标定明明对、却一直 calib”的根因。

### 3.3 阈值（红激光）

实测光斑在相机里有三种形态，`_scan_window` 都接受：

- 红点/暗红点：`r-g>=120` 且 `luma>=50`；
- 亮墙上的粉点：`r-g>=70` 且 `luma>=150`（墙面 luma≈190、点≈200）；
- 过曝白核：`luma>=245` 且 RGB 都>=220（live 曝光可能把红点打成白色）。

首次锁定与旧标定偏差 >15px 时，`just_calibrated=True`，上层写回
`/sdcard/laser_calib.json`（本次实测从 375,275 修正到 303.6,233.3）。

### 3.4 时间预算 / 看门狗

- `laser_spot.budget_ms=12`：`_scan_window` **逐行**检查，超时立即返回；
- `yolo_pose/agent/preview` 的 `laser_budget_ms=25`：单帧超过 25ms 就
  `laser_cooldown=8` 帧跳过激光，AI/图传继续；
- `fixed_allow_reacquire` 只在 R2 代理置 1；live/IDE 默认 0；
- 全图 `find_blobs` 已从所有路径移除。

### 3.5 全图 JPEG

`full_every` 默认改为 **9999**（不周期性发全图）：640×480 全图 JPEG 单帧约
370ms，会让 fps 周期性从 19 掉到 9。PC 需要全图时用：

```bash
python3 -m k230tool grab -n 1 -o /tmp/grab
```

## 4. 测试

### 4.1 预测器单元测试

`python3 tools/test_spot_tracker.py`（PC，不连板）：

| 用例 | 结果 |
| --- | --- |
| 匀速运动（3px/帧，20 帧后 coast 5 帧） | 预测误差 0.2px |
| 连续丢 5 帧 coast | 误差 7.7px，门控 ≤48px |
| 100px 离群点 | 被 gate 拒绝 |
| 静止 + ±1.5px 噪声 | 速度 0.47px/s，门控 8px |

### 4.2 板端运动/遮挡

`python3 scripts/motion_test_laser.py --duration 24`（K230 debug、ESP32 R1 激光亮、
云台不跟踪、`--full-every 9999`）：

| 指标 | 30s 轮 | 24s 轮 |
| --- | --- | --- |
| 帧数 | 97 | 77 |
| 实测帧（status=ok） | 16 | 12 |
| fps（板端自报） | 19.21 | 19.24 |
| 最大 coast | 5 | 7 |
| 最大 laser_ms | 28 | 28 |
| 遮挡后恢复 | 2.75s | 2.81s |
| 全图搜索 | 无 | 无 |

稳定标定后的 24s 轮（calib 312.5,230.1）：78 帧 / 13 实测 / **54 预测跟踪** /
11 标定丢失；预测误差 **2.24px mean / 3.71 p95 / 5.73 max**；19.03fps；
最大激光 28ms；coast 9；无全图。`pred` 统计已修，不再是 0。

- 遮挡期间只 coast/返回标定，无全图、无卡死；
- 激光耗时中位数 0、P95 3ms；首帧 28ms 触发一次看门狗跳帧；
- 预测在跳过帧上连续变化（如 299.1,229.8 → 300.3,231.9），跟踪实测帧正常。

### 4.3 预测/状态统计（已修）

之前 PC 汇总 `track=0`、`|pred-measured|=0` 的根因有两个：

1. 固定分支更新前没有保留 `pred`，返回时被实测值覆盖；
2. 看门狗跳帧时走 `laser=None` 的标定回退，状态标成 calib。

现在保留更新前 `pred_x/pred_y`；固定分支区分 `status=track`（上一帧复核成功）
与 `calib`（coast/丢失）；看门狗跳帧仍走标定回退，但不再冒充 track。

**24s 稳定标定轮**（calib 312.5,230.1）：78 帧 / 13 实测 / **54 预测跟踪** /
11 标定丢失；预测误差 **mean 2.24px、median 2.02、p95 3.71、max 5.73**；
19.03fps；最大激光 28ms；coast 9；无全图。

### 4.4 标定自动回写（已修）

固定分支原先在“实测与旧标定偏差>15px”时写 `just_calibrated`，上层立刻
`laser_calib.json`。移动测试会把它一路写成 `(307,215)→(276,204)→…→(0,26)`
的漂移值（CanMV 预览日志实测）。现在：

- **只有首次无标定捕获**（非 fixed 路径）才允许自动回写；
- 固定分支只更新内存 `self.fixed` 用于跟踪，**不写文件**；
- 24s 运动测试后 calib 仍为 (312.5,230.1)，不再漂移。

### 4.5 CanMV IDE “当前固件不支持 REPL 输入”

这不是固件能力问题，而是板子在工作模式自动跑 `/sdcard/main.py`（代理）或旧
预览脚本占着 REPL。处理：

1. CanMV/VSCode 里 **Disconnect Board**；
2. `bash k230/scripts/mode.sh debug`，或让 ESP32 连发 3 次 `gnone`（代理暗号，
   写 `/sdcard/offline_off` 并复位）；
3. 仍不行按板子 **RESET**（offline_off 保留，板子仍进 debug；不会自动跑代理）；
4. 工作模式下 IDE 连不上是正常的；debug 下 REPL 空闲，IDE 可正常连。

## 5. 使用

```bash
# 调试模式 + R1 激光
python3 -m k230tool live --duration 20 --full-every 9999
python3 scripts/motion_test_laser.py --duration 24

# 全图抓帧（替代 full_every=30）
python3 -m k230tool grab -n 1 -o /tmp/grab
```

参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `fixed_verify_every` | 5 | 每 5 帧复核一次，其余帧零搜索 |
| `budget_ms` | 12 | 单次光斑搜索硬预算 |
| `laser_budget_ms` | 25 | 调用侧看门狗阈值 |
| `laser_cooldown` | 8 | 超时后跳过的帧数 |
| `kf_q` / `kf_r` | 40 / 4 | Kalman 过程/测量噪声 |
| `kf_max_gate` | 48 | 门控最大半径 |
| `fixed_coast_max` | 10 | 长期丢失后回到标定 |
| `full_every` | 9999 | 全图 JPEG 周期（默认关） |

## 6. 下一步

1. 修 PC 汇总的 `track`/pred 诊断（不影响控制）；
2. 把受限重捕从“单次 ±180 大窗”升级为多帧分格（避免 stride 漏掉 5px 光斑）；
3. R2 实车运动测试：把预测接入 ESP32 的 B/C 状态，观察 `ctrlErr` 是否收敛；
4. 之前记录的云台机械卡住问题仍需检查 pan 舵机臂/相机支架。
