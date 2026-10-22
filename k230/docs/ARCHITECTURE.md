# 架构与设计说明

## 1. 总体结构

```
┌──────────────────────── PC（Python 3） ────────────────────────┐
│  k230tool.cli                                                   │
│    ├── probe / monitor / grab / serve / push / pull            │
│    └── live ──> K230Stream（后台线程）                          │
│                     │  串口 /dev/ttyACM0 @115200                │
│                     │  二进制帧协议（JPEG / JSON / 日志 / 统计） │
│                     ▼                                           │
│            live/latest.jpg  live/latest.json  live/status.txt   │
└─────────────────────────────────────────────────────────────────┘
                              │ USB CDC
┌──────────────────────── K230（MicroPython） ───────────────────┐
│  raw REPL 接收 PC 下发的脚本（stream_camera.py / yolo_stream.py）│
│                                                                 │
│  Sensor(id=2)                                                   │
│    ├── chn1 RGB565 640×480 ── img.compress(q) ── JPEG ──┐      │
│    └── chn2 RGBP888 640×480 ── AI2D letterbox ── KPU ───┤      │
│                                        │                │      │
│                                  aidemo.yolov8_det_pp    │      │
│                                        │                │      │
│                                   T_TARGET(JSON) ───────┴──> USB│
└─────────────────────────────────────────────────────────────────┘
```

PC 端只依赖 `pyserial` / `numpy` / `opencv-python`；板端脚本由 PC 自动下发，
不需要 CanMV IDE 插件。

## 2. 串口帧协议

板端 -> PC，每包：

```
偏移  长度  内容
0     4     MAGIC = AA 55 4B 32
4     1     类型字符：F JPEG / T 检测 / L 日志 / S 统计 / H 握手 / E 错误
5     8     seq        （8 位十六进制，ASCII）
13    8     plen       （原始载荷字节数）
21    8     wlen       （线上字节数，转义后）
29    8     crc32      （原始载荷）
37    1     '|'
38    wlen  载荷
```

### 为什么包头全是 ASCII、载荷要转义

K230 的 REPL 输出是 cooked 的：固件会把每个 `0x0A` 改写成 `0x0D 0x0A`。
如果包头里出现 `0x0A`，长度字段会被改写，解析必失败。因此：

- 包头只用 ASCII 十六进制字符，绝不含 `0x0A`；
- 载荷先把原始 `0x0D` 双写成 `0x0D 0x0D`，再写入；
- PC 端：

```python
payload = wire.replace(b"\r\n", b"\n").replace(b"\r\r", b"\r")
```

两次 replace 无歧义还原，CRC32 校验通过才收下。实测坏包 0。

### 为什么 T 包在 F 包之前

板端每帧先发 `T_TARGET`（几十字节 JSON），再发 `F`（几十 KB JPEG）。
PC 端收到 `F` 时，`self.target` 已经是同一 `seq` 的检测结果，画框不会错帧。
`K230Stream.detections(frame_id)` 也按 `seq` 缓存了最近 120 帧。

## 3. 板端脚本下发

1. `render_board_script(cfg)` 读取 `board/*.py`，用正则替换顶部单行 `CFG = {...}`；
2. `K230Link.enter_raw()` 进入 raw REPL；
3. 源码按 **128 B / 2 ms** 分块写入（一次性灌入会丢字节）；
4. `\x04` 结束粘贴，板端开始执行；
5. PC 端读 `OK` 后转为帧协议收包；
6. 停止时发 `\x03`（Ctrl-C），板端 `finally` 释放摄像头，然后（AI 脚本）`machine.reset()`。

### 板端 CFG

`stream_camera.py`：

```python
CFG = {'w':640,'h':480,'q':70,'fmt':'rgb565','mirror':0,'flip':0,
       'fps':0,'stats':30,'limit':0,'sensor_id':2}
```

`yolo_stream.py`：

```python
CFG = {'w':640,'h':480,'q':70,'aw':640,'ah':480,
       'model':'/sdcard/models/pose_v3_256.kmodel','labels':['target'],
       'in_w':256,'in_h':256,'conf':0.25,'nms':0.45,'max_boxes':20,
       'fps':0,'stats':10,'limit':0,'sensor_id':2,'reset_on_exit':1}
```

## 4. 线程与恢复

`K230Stream`：

- 后台线程 `_run()` 循环调用 `_session()`；
- `_session()` 打开串口 → 检查堆（`min_heap=1.5 MB`）→ 必要时 `hard_reset`
  → 下发脚本 → 收包；
- 任何异常都会记 `last_error`，`auto_restart=True` 时 2 秒后重连；
- `latest()` / `wait_frame()` 用锁访问最新 JPEG 和帧号，主线程不阻塞收流。

### 堆与传感器残留

MicroPython 堆被反复 exec 吃掉后 `snapshot()` 会卡死。开流前检查
`gc.mem_free()`，低于阈值就软复位（回到 ~4 MB）。

`OSError: sensor(2) is already inited` 是上次脚本没干净退出留下的，
`MediaManager.deinit()` 清不掉，只能 `machine.reset()`；工具检测到后自动做。

## 5. AI 数据流

### 5.1 双通道

- `chn1` RGB565：只用于 JPEG 编码/回传，画质与帧率和纯推流一致；
- `chn2` RGBP888：RGB 平面格式，`to_numpy_ref()` 得到 `(3,H,W) uint8`，
  直接喂给 `AI2D`，不需要 PC 端转换。

两个通道同分辨率（640×480），所以检测框坐标与 JPEG 一一对应。

### 5.2 预处理

`libs.YOLO.YOLOv8(config_preprocess)` 用 `Ai2d` 做 letterbox：

- pad 到正方形（填充值 128）；
- `tf_bilinear` + `half_pixel` 缩放到 320×320；
- 模型输入 uint8，kmodel 内部按 `input_range=[0,1]` 反量化到 0~1。

### 5.3 后处理

`aidemo.yolov8_det_postprocess(...)` 输出 `([boxes], [ids], [scores])`，
`mode="image"` 时 boxes 在 `rgb888p_size` 坐标系。板端把结果压成：

```json
{"seq": 12, "w": 640, "h": 480, "ms": 29,
 "det": [[172,345,162,142,0.083,0]], "labels": ["target"]}
```

PC 端按 `det` 的 `(w,h)` 缩放到 JPEG 尺寸画框。

### 5.5 长跑内存

`emit()` 每帧要为 JPEG 做 CR 转义并拼接包头，会产生几十 KB 临时 bytes；
如果只在固定间隔（如每 32 帧）调用 `gc.collect()`，AI 帧较大时会在两次回收之间
把 MicroPython 堆吃光（实测 91 帧后只剩 202 KB）。板端脚本因此：

- 启动时 `gc.threshold(96 * 1024)`，让分配器自动触发 GC；
- 另外每 8 帧显式 `gc.collect()`。

实测 30 秒 / 216 帧后堆稳定在 ~3.7 MB，帧率不受影响（~8.3 fps）。

### 5.4 为什么 AI 脚本退出要复位

K230 的 `nn.kpu()` 原生对象在 Python 命名空间销毁时，析构会死锁
（实测：`del yolo`/GC/`kpu.release()` 都可能卡住，REPL 随后无响应）。
`AIBase.deinit()` 还调用了不存在的 `kpu.release()`，更不稳。

本包的处理：

1. 退出时把 `yolo/kpu/ai2d/tensors` 存进 `sys.modules["_k230_ai_keepalive"]`，
   让它们**不被析构**；
2. 停止 sensor、`MediaManager.deinit()`；
3. 默认 `reset_on_exit=1`，主动 `machine.reset()`，彻底清干净。

代价是每次停止 AI 流约 8 秒重连；换来的是下一次一定能用。
把 `reset_on_exit` 设为 0 可以跳过复位，但反复开关可能积累 nncase 内存池。

## 6. live 桥接

`live.py` 主循环：

```
wait_frame → 解码 → 画检测框/HUD → 原子写 latest.jpg
           → 每 0.5 s 原子写 latest.json / status.txt
           → 可选：每 N 秒写 history/frame_XXXXXX.jpg，滚动删除
```

- 原子写：先写 `.tmp`，`os.replace()`，Agent 不会读到半张图；
- `connected` 由帧龄判断：超过 `--stale-after` 或板端断开就写 false；
- 停止：SIGINT/SIGTERM 或创建 `live/STOP` 文件；
- `snapshot` 优先复用新鲜的 `live/latest.jpg`（0 延迟、不占串口），
  否则临时开一次流。

## 7. 端口独占与工作流

USB CDC 只有一个 REPL，任何时刻只能一个进程读。因此：

- Agent 看 → `k230tool live`，CanMV IDE 必须断开；
- 人看 IDE → 停掉 `live`（板子复位），在 IDE 里跑 `board/yolo_preview_ide.py`。

如果项目要求「IDE 调试 + Agent 同时看」，需要第二条链路（WiFi MJPEG /
第二路串口）。板端 `libs/Network.py` 支持网络，但需要额外配置和验证，
本包暂不默认启用。

## 8. 扩展点

- 自定义板端程序：继承 `K230Stream` 覆盖 `_start_board()`，或直接把
  `board_script=` 指向自己的脚本；
- 自定义检测后处理：改 `board/yolo_stream.py` 的 `_pack_target()`；
- 自己的 PC 算法：`from k230tool.link import K230Stream`，取 JPEG 解码成
  `numpy.ndarray`，和普通摄像头完全一致；
- 自己的 UI：读 `live/latest.jpg` 或起 `serve_mjpeg.py` 当网络摄像头。
