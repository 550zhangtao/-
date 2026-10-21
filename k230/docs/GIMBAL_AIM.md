# 云台闭环瞄准：PID / 激光参考 / 运动标定（2026-10-22）

> 目标：让**激光点落在靶心中央**。
> 控制律在 ESP32（`firmware/ESP32_Gimbal_K230/ESP32_Gimbal_K230.ino` 的
> `updateTrack()`）；K230 只提供靶心 `$KT` 和激光 `$LS` 两个图像坐标。
> 本文记录 2026-10-22 的 A/B 实测、参数和机械限制。

## 1. 控制律

每一帧（100Hz 控制环，`AIM_CTRL_MS`）：

```
tx, ty   = 靶心（$KT 的 bx,by；3 点中值 + α 低通）
vxF,vyF  = 靶心图像速度（低通）
refX,refY= 激光位置（$LS；K230 侧已做 α-β + 50ms 前瞻）
errX = (tx + vxF·trkLead·leadF) - refX
errY = (ty + vyF·trkLead·leadT) - refY
PID：desiredPan  = -PAN_SIGN  · (kp·err + ki·∫err + kd·d(err)/dt)
     desiredTilt = -TILT_SIGN · (...)
速度一阶低通 smooth → 舵机 us（100Hz 更新）
```

- 死区 `WBAND_PX = 8`（`err` 绝对值小于 8px 时 PID 输出 0、不积分）；
- 瞄准误差的**真实值**（PC 侧）= `bull_x/y - laser_x/y`（同一帧
  `latest.json` 的 `bull` 与 `laser`），由 `scripts/motion_test_laser_aim.py`
  的 `aimErr` 统计；
- `ctrlErr`（ESP32 `status`）是**前瞻修正后**的误差，和 `aimErr` 可能差
  几 px（滤波/前瞻/采样时刻），看趋势用 `ctrlErr`，看“是否对准”用 `aimErr`。

## 2. 参数（可在线调）

USB 命令（不需要重烧）：

```
pid  <kp> <ki> <kd>      # PAN
pid2 <kp> <ki> <kd>      # TILT
smooth <tau_s>           # 速度指令低通
trk  <alpha> <lead_s>    # 靶心低通 / 前视时间
pidshow                  # 查看当前值
aim / status             # 看 ctrlErr / laser / src
```

当前固件**默认值**（稳定，A 组）：

| 参数 | PAN | TILT |
| --- | --- | --- |
| kp | 2.5 | 2.5 |
| ki | 0.3 | 0.2 |
| kd | 0.15 | 0.10 |
| 死区 | 8px（`WBAND_PX`，需改固件） | 同 |
| smooth | 0.02s | |
| alpha / lead | 0.35 / 0.20s | |

## 3. A/B 实测（静止靶，15s/轮）

| 组 | PID | 真实瞄准误差 `aimErr` | `ctrlErr` | 结果 |
| --- | --- | --- | --- | --- |
| **A 当前默认** | PAN 2.5/0.3/0.15，TILT 2.5/0.2/0.10 | 稳定后约 **(-5.6, -8.4)px**；15s 含抬升过渡 p50 (10.2,13.1) | 稳定后 p50 **(6.3,8.4)** | **稳定**，靶心和激光都在视野 |
| **B AGENTS 最终** | PAN 9/0/0.5，TILT 9/0/0.6，smooth 0.006，trk 0.40/0.25 | 发散：`ctrlErrY≈-219`、bull 被送出画面、激光 `src=1` 0% | — | **不可用** |

**为什么 B 会发散**：AGENTS 里 kp=9 那套是配 **`WBAND_PX=40`** 的；本机固件
是 **8px** 死区。8px + kp=9 + ~100ms 链路延迟 → 过冲/振荡，把云台推到限位、
靶纸出画、激光丢失。**不要在 8px 死区上直接用 kp=9**；要么回到 A 组，
要么先改 `WBAND_PX` 再重新整定（但 40px 死区会让“激光到靶心中央”变差）。

## 4. 激光识别在运动下的标定/优化

- **α-β 常速滤波 + 50ms 前瞻**：`$LS` 输出更平滑，运动轮
  `$LS src=1` **86.2% → 96.6%**，`ctrlErr` p95（X/Y）
  **88.7/67.3 → 35.2/42.2**（见 `LASER_AIM.md`）；
- **在线慢速 re-zero**（`laser_aim.py`，`rezero_*`）：光斑稳定（速度<6px/s）
  连续 8 帧后，把标定先验慢慢拉向实测（3%/帧）。靶纸举高/换姿态后，
  下一次丢失的回退位置不会差几十像素；
- 但 re-zero **不能救“冷启动时标定偏 >24px”**：微窗（±12/±24）里根本看不到
  光斑。这种情况要在 PC 上重标（`scripts/calib_laser.py --grab`/`--image`）；
- 靶纸在光斑上的手举运动，光斑会在图像里真的移动 36~48px（白墙只有 0.4px），
  这是物理量，不是检测噪声；α-β 跟着它走，`calib_dev` 大是正常的，
  `need_recalib` 不再按 offset 触发。

## 5. 机械/几何限制（比 PID 更常见）

- **tilt 限位 1200/1800**：靶纸平放桌面/太低时，tilt 顶在 1200，
  纵向误差会固定 40~100px，PID 再大也没用。把靶纸放到画面中间偏上；
- **靶纸必须在激光束里**：激光打在白墙、靶纸在下/上 80px 外时，
  光斑不随靶纸动，闭环只是在“把靶心拉到墙上的光点”，不是打靶；
- **靶心要在画面里**：靶纸出画/被手挡，`$KT run=0`，ESP32 会 stale/三态 A 复位；
- 死区 8px + 舵机静摩擦：静止残差通常在 5~15px；要更小需要改 `WBAND_PX`
  并接受抖动，或者换更高分辨率的执行机构。

## 6. 测试流程

```bash
# 1) 工作/调试：K230 要能推流（live 或 IDE 预览），ESP32 USB 插着
cd k230
python3 scripts/motion_test_laser_aim.py --duration 15 --label A
# 2) 换 PID（可选；不要用 kp=9 + db=8）
python3 scripts/motion_test_laser_aim.py --duration 15 \
    --pid "2.5 0.3 0.15" --pid2 "2.5 0.2 0.10" \
    --smooth 0.02 --trk "0.35 0.20" --label A_retry
# 3) 运动：--duration 30，手举/移动靶纸
```

输出：`live/motion_*.csv` + 汇总里的
`瞄准误差(靶心-激光): p50/p95/mean/std`、`|ctrlErr|`、`pan/tilt` 等。

## 7. 结论

- **能对准**：A 组静止稳定后真实误差约 **8~10px**（在靶心范围内）；
- **B 组不可用**：kp=9 配 8px 死区会发散，已恢复 A；
- **运动**：α-β 把 `ctrlErr` p95 从 88/67px 降到 35/42px，`$LS src=1` 96.6%；
  剩余的 p95 主要来自 K230 ~15fps + ~100ms 链路延迟和手举靶的物理运动，
  不是激光滤波或 PID 单方面能解决的；
- **下一步**若要更小误差：先降延迟（提高 K230 帧率/降推理），再考虑
  改 `WBAND_PX` 后重整整定，而不是继续加 kp。
