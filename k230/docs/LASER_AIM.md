# 激光瞄准参考：标定先验 + 微窗质心（2026-10-22 重构）

> 本文描述**当前**方案。旧 `board/laser_spot.py`（LAB 阈值 + find_blobs +
> 多套候选打分 + 全图分格 + 重捕 + 1200 行状态机）已删除，历史记录见
> `git show 0c5ff1b` 与 AGENTS.md 旧章节。

## 1. 一句话

**标定一次光斑位置，运行时只在它 ±12px 的 25×25 微窗里做加权质心；
找不到就退回标定位置，绝不扫全图、绝不重捕。** 板端实测 1~2ms/帧。

## 2. 硬件前提（重要）

- **激光常亮，GPIO5 已弃用**（2026-10-22 用户确认）。所以不能再靠
  ESP32 `laserpin 0/1` 关/开激光做 ON/OFF 差分标定；
- 相机与激光笔仍刚性固定在云台上 → 光斑在图像里的位置是**可标定的物理量**；
- ESP32 侧的 `setLaserOutput()` / `applyLaserMode()` / `laserpin` 代码保留
  （以后接可控开关可直接用），但当前对激光没有实际作用；
- `$LS` / 三态追踪（A 靶心丢失 / B 标定参考 / C 实测参考）协议不变。

## 3. 为什么换掉旧实现

| 问题 | 旧实现 | 现在 |
| --- | --- | --- |
| 每帧成本 | `find_blobs` 固定 ~12ms（与 ROI 大小基本无关） | 25×25 纯 bytearray 扫描 **1~2ms** |
| 误检 | 红色杂物/暖光靠多套阈值 + 打分硬扛 | 先标定再微窗，窗口外的东西根本看不到 |
| 状态机 | fixed/verify/reacquire/confirm/lost，1200 行 | `ok / calib / lost` 三个状态，~350 行 |
| 标定 | 板端自动全图搜索（慢、会锁到红物） | PC 端单帧 LAB + **人工点选**（或可选 ON/OFF 差分） |
| 卡死风险 | 全图搜索单帧 0.5~2.3s（误记）/ 实测 22ms | 无全图路径；微窗预算 6ms，超时直接回退 |

> 旧文档里“全图 find_blobs 0.5~2.3s”是误记；本机实测 640×480 约 **22ms**。
> 12ms 的固定开销仍然不适合每帧，所以不再用 `find_blobs`。

## 4. 板端实测（CanMV v1.4.3 / K230，2026-10-22）

| API | 耗时 |
| --- | --- |
| `find_blobs(48×48 ROI)` | **11~14ms**（固定开销，和 ROI 大小关系不大） |
| `find_blobs(160×120)` | 12~14ms |
| `find_blobs(640×480)` | 22ms |
| `img.difference()` 全图 | 7ms |
| `img.binary()` 全图 | 16ms |
| `img.mean_pool(4,4)` | 3ms |
| `img.get_statistics(roi)` | ~0ms |
| 纯 Python bytearray 扫描 25×25 | **1ms** |
| 纯 Python bytearray 扫描 49×49 | 4ms |

端到端（`k230tool live`，pose_v3，现场墙+激光）：

```
calib = (306.28, 229.32)
laser = (306.7, 229.4)  src=measured  status=ok  calib_dev=0.4px  laser_ms=2ms
$LS   = 306.6,229.1     src=1          conf=356
loop=67ms infer=32ms uart=7ms
```

`img.save('/sdcard/laser_aim_dbg.jpg')` 后 `k230tool pull` 回来核对：
十字/圆环正好压在墙上的激光点上（AGENTS.md 要求的板端绘制路径验证）。

## 5. 标定步骤（PC）

### 5.1 默认：单帧 + 点选（激光常亮）

```bash
cd k230
bash scripts/mode.sh debug          # 工作模式下 REPL 被代理占用，先切调试
python3 scripts/calib_laser.py --grab --push
```

窗口里：

- 绿色 `#1..#5` 是 LAB 候选，红色十字是当前点；
- **鼠标左键点一下真正的光斑**；
- `Enter` / `Space` 接受，`r` 重跑，`q` 取消。

没有显示器时用 `--x 306 --y 229` 直接给坐标（640×480 原图）。

### 5.2 可选：ON/OFF 主动差分（激光可关时）

```bash
python3 scripts/calib_laser.py --grab --pair --esp-port /dev/ttyUSB0 --push
# 或手动开关：--grab --pair --manual
```

`|ON-OFF|` 差分 + 5×5 close + 连通域 + 亚像素质心。背景/红物/环境光全部抵消，
是最稳的标定方式；缺点是要求能可靠关掉激光。

### 5.3 多距离（可选）

`spot = c + k·s`（s = 靶纸表观边长，pose 每帧给）。先在两个距离各标一次，
把坐标和当时的 `s` 记成 JSON：

```bash
python3 scripts/calib_laser.py --fit points.json
# points.json = [{"x":408,"y":278,"s":42}, {"x":392,"y":271,"s":78}]
```

单距离标定时 `k=0`，等价于旧的 fixed 模式；靶纸倾斜时 `s` 会混入姿态，
k·s 只是近似，必要时用 `--pair` 重标或缩小 `rad`。

## 6. 运行时逻辑（`board/laser_aim.py`）

```
predict = calib + k·(s - ref) + off_smooth
scan ±rad（默认 12）→ R-G>=warm_min 且 luma>=luma_min 的像素加权质心
  命中：off_smooth += ema·(measured - predict)；src=measured / status=ok
  命中但偏离 > max_step：当离群，不跟
  连续失败 >= lost_widen_after(3)：rad 放大到 rad_max(24) 再试
  连续失败 >= lost_after(10)：status=lost
  任何失败：返回 predict，src=calib，绝不扩大搜索
```

输出字段与旧 `$LS` 完全兼容：
`x, y, conf, mode, src("measured"/"calib"), status("ok"/"calib"/"lost"), ms`。

`status=lost` 不改变输出：仍返回标定/预测位置（`src=2`）、`conf=0`、
`need_recalib=1`——这是三态 B 的设计（激光被遮挡时继续用几何标定）。
只有支架被撞/标定过期才需要人工重标；`need_recalib` 只告警，不会自动改文件。

`$LS` 语义不变：

```
$LS,<seq>,<run>,<x*10>,<y*10>,<conf*1000>,<src>!
    src: 0=无位置  1=本帧微窗实测  2=标定/预测
```

ESP32 侧 `laserFresh = aimLaserValid && (now-aimLaserMs < 500ms)` 逻辑不变；
`aim` 命令可直接看 `laser=..., src=1/2, conf=...`。

## 7. 参数（`board/laser_aim.py` 的 `CFG`）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `rad` | 12 | 微窗半径，25×25 ≈ 1ms |
| `rad_max` | 24 | 连续丢失后放大上限，49×49 ≈ 4ms |
| `budget_ms` | 10 | 单帧硬预算；超时按未命中处理（6ms 在运动/JPEG 竞争下会截断，实测 p95 正好顶 6ms） |
| `warm_min` | 55 | R-G 阈值（实测激光 84~220，白墙 -20~33） |
| `luma_min` | 30 | 压掉暗红噪点 |
| `min_px` / `max_px` | 2 / 800 | 太少=噪点，太多=红色物体 |
| `ema` | 0.5 | 实测偏移的平滑系数 |
| `lost_widen_after` / `lost_after` | 3 / 10 | 放大/判丢失的连续失败次数 |
| `max_step` | 24 | 偏离预测超过此值视为离群 |
| `calib_warn` | 40 | 实测相对标定偏移超过此值 → `need_recalib` |
| `warm_k` | 80 | `conf=(max_warm-warm_min)/warm_k+0.2` |
| `ema`（s） | 0.3 内部固定 | 靶纸表观边长的 EMA；调用方优先传 4 角点平均边长 |

## 8. 运动靶 / 静止靶实测（2026-10-22）

场地：室内白墙 + 真实激光（USB `laserpin` 对 GPIO5 无效，激光常亮）。
指标由 `scripts/motion_test_laser_aim.py` 采集（`latest.json` + ESP32 `aim/status`）。

| 场景 | 采样 | `$LS src=1` | `calib_dev` p95 | `laser_ms` p95 | 激光抖动 std | `$KT` fps |
| --- | --- | --- | --- | --- | --- | --- |
| 静止（预算 10ms） | 20 点 | **100%** | **0.4px** | 2ms | x 0.13 / y 0.08 | 18.2 |
| 运动（预算 **6ms**，旧） | 44 点 | 56.8% | 2.7px | 6ms（顶预算） | x 1.02 / y 0.88 | 15.4 |
| 运动（预算 **10ms**，靶纸在动/倾斜） | 29 点 | **86.2%** | 16.9px\* | 5ms | x 5.59 / y 2.00 | 17.3 |

- 运动轮 57% 的根因是 **6ms 预算截断**：`laser_ms` p95 正好 6ms；改 10ms 后
  静止轮 100%、p95 2ms。运动轮已复查（见下）。
- 靶纸在视野时（`bull_status=ok`）6ms 运动轮 `src=1` 64%、`calib_dev` p95 2.3px；
  靶纸丢失/低置信时 44%，但坐标仍用标定值，不跳。
- \* 10ms 运动轮的 `calib_dev` p50=9.7 / p95=16.9，是**靶纸移动/倾斜让光斑图像
  位置真的移动**（上一轮打白墙时只有 2.7px），不是检测噪声；`off` EMA 跟着实测走，
  窗口中心随之平移，所以 86% 仍是 `measured`。`k=0` 时标定值只当先验，
  距离/姿态补偿要靠 `--fit` 的 `k·s`（需要第二个距离）。
- 关键结论：**激光和相机同云台，靶子动时光斑在图像里仍近似不动**，
  所以标定值在运动靶下依然成立；运动靶主要考验 `$LS` 的稳定性和 ESP32 控制。

> 运动轮复测（10ms）：`$LS src=1` **86.2%**、`laser_ms` p50=2/p95=5ms、
> `$KT` 17.3fps、bull 96.6%；`calib_dev` 10~17px 是靶纸运动/倾斜导致的真实位移。

## 8. 排错

| 现象 | 处理 |
| --- | --- |
| `$LS` 一直 `src=2` | 光斑不在微窗内：重标；或临时把 `rad` 调到 20 看是否命中 |
| `laser_ms` 经常 >6 | 检查是否有人把 `rad` 调大；正常 25×25 应 1~2ms |
| `calib_dev` 持续 >40 | 相机/激光支架松动，或靶纸距离/角度变化很大 → 重标 |
| `need_recalib=1` | 只是告警，不会自动改文件；重标后写新 `laser_calib.json` |
| 点选器没有窗口 | 用 `--x/--y` 手动给坐标，或 `--no-gui` 自动接受最佳候选（不保证对） |
| 单帧自动选到红物 | 关掉 `--no-gui` 用鼠标点；红物/红灯没有“白核+红晕”但 LAB 仍可能入选 |
| 标定文件写不进去（EPERM） | 板端有脚本占着句柄；`python3 -m k230tool mode reset` 后再写 |

## 9. 文件

| 文件 | 作用 |
| --- | --- |
| `board/laser_aim.py` | 板端标定先验 + 微窗质心（本文件） |
| `scripts/calib_laser.py` | PC 标定：单帧点选 / ON/OFF 差分 / 多点拟合 / push |
| `tools/test_laser_aim.py` | PC 单测（命中/偏标定/关激光/距离模型/标定读写） |
| `board/yolo_pose.py` | USB 推流：`laser` JSON + `$LS` |
| `board/yolo_preview_ide.py` | CanMV IDE 预览：蓝十字/圆环 + `$LS` |
| `board/gimbal_agent.py` | 离线代理：R2/START 后 `$LS` |
| `firmware/ESP32_Gimbal_K230/*.ino` | `$LS` 解析 + 三态 A/B/C（未改协议） |
