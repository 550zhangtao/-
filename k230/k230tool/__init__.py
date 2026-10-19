# -*- coding: utf-8 -*-
"""K230 USB 工具链（主机端 Python 包）。

模块：
    link         串口链路、帧协议、板端脚本下发、多线程收流
    live         实时桥接：live/latest.jpg + latest.json
    decode       JPEG 解码 / 检测框 / HUD
    monitor      OpenCV 窗口
    grab         批量抓帧
    serve_mjpeg  HTTP MJPEG
    probe        全链路体检
    push_file    PC -> 板
    pull_file    板 -> PC
"""

__version__ = "1.0.0"

from .link import (  # noqa: F401
    BOARD_SCRIPT, POSE_BOARD_SCRIPT, DEFAULT_CFG, POSE_DEFAULT_CFG,
    K230Link, K230Stream, BoardError, Packet,
    find_port, list_canmv_ports, describe_ports, open_serial,
)
