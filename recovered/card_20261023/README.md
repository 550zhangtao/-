# 板载 SD 卡快照（2026-10-23 22:31，回退前）

来源：K230 `/sdcard` 的完整备份（`k230/live/sd_backup/`，74MB，被 gitignore）。
这里只挑出**与仓库版本不同**的板端脚本，防止回退把用户未入库的改动弄丢。

| 文件 | 大小 | 说明 |
| --- | --- | --- |
| `gimbal_agent.py` | 22291 | 卡上实际在跑的常驻代理，比仓库 18327 版新（含 `laser_openmv` 集成） |
| `laser_openmv.py` | 38906 | 卡上专用激光模块，仓库里没有（AGENTS 记载"旧激光已删"的另一种路线） |
| `yolo_pose.py` | 27281 | 卡上版本（仓库 708c5a8 为 28582） |
| `yolo_preview_ide.py` | 30513 | 与仓库 30517 基本一致 |
| `laser_aim.py` | 2469 | 激光-靶心闭环模块 |

完整备份（含 models/libs/所有调试图）在本地 `k230/live/sd_backup/`。
