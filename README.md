# 打靶小车（ESP32 + OpenMV）

用于打靶小车 srdp 组内成员使用，包含板型资料、引脚信息、原理图、固件与视觉调试记录。

## 目录

```
firmware/1014_by_LinPeng/
  ESP32_UART.ino   ESP32 主控（舵机/电机/循迹/PS2/视觉串口）
  main.py          OpenMV 视觉端（当前 V12）
hardware/          板卡资料、引脚图、手册
tools/openmv_link.py   不依赖 OpenMV IDE 的串口调试/抓图工具
docs/V12_调试报告.md    本轮调试的根因、改动、实测数据与现场调参
docs/img/          靶面参考图、检测结果
```

## 系统分工

- **OpenMV**：只负责"看"。识别 10 环靶心，通过 UART3（P4/P5，115200）以
  `$G2,TGT,<cx>,<cy>,<vx>,<vy>,<lx>,<ly>,<aim>,<run>!` 上报坐标、速度、
  激光点和跟踪状态。
- **ESP32**：独占云台（pan GPIO27 / tilt GPIO14），把像素误差换算成舵机微秒，
  运行瞄准 PID + 速度前馈；同时负责电机、循迹、PS2 手柄和模式切换。

`$G2,TGT` 的坐标是 **VGA 像素**（320,240 中心）。OpenMV 内部以 QVGA 运行
（为了 46 fps），上报前统一 ×2 还原，因此 ESP32 的 PID/死区参数不需要改。

## 视觉端要点（V12）

- QVGA 320×240，**46.6 fps**（传感器上限）；
- 曝光 57 ms → **10.4 ms** + 36 dB 增益，显著抑制高速运动模糊；
- 白斑种子阈值自适应（不再写死 `L* ≥ 60`）；
- 靶心身份判据：归一化对比度 ≥0.50、白环覆盖 ≥70%、中心暗字比例 ≥0.15；
- 搜索窗带速度×延迟裕量，帧时间戳取在 `snapshot()` 之后；
- 激光颜色阈值修正（旧阈值在 LAB 空间无解）。

详见 [docs/V12_调试报告.md](docs/V12_调试报告.md)。

## 快速验证

1. OpenMV IDE 打开 `firmware/1014_by_LinPeng/main.py`，另存到板子为 `main.py`；
2. 断电重上电，日志应出现 `V12 BULLSEYE READY ...` 与 `FPS 46.x`；
3. 把镜头对焦到实际射击距离，靶心放在画面中央；
4. 串口（ESP32）应持续收到 `$G2,TGT,...`，`run=1` 后云台开始跟踪。

> 没有 IDE 时可用 `tools/openmv_link.py` 通过串口直接部署/抓图，
> 用法见 [tools/README.md](tools/README.md)。
