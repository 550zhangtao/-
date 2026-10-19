# YOLOv8 标靶识别部署指南（K230 / CanMV）

本文按「采集 → 标注 → 训练 → 导出 → 编译 → 推板 → 实时验证」走一遍，
所有命令都在 `k230/` 根目录执行。

---

## 0. 先确认板端环境

```bash
python3 -m k230tool probe          # 链路体检
python3 -m k230tool models         # 看 /sdcard/models 里已有的 kmodel
```

CanMV v1.4.3 的 YOLOv8 运行时在 `/sdcard/libs/YOLO.py`，接口：

```python
from libs.YOLO import YOLOv8
yolo = YOLOv8(task_type="detect", mode="image",
              kmodel_path="/sdcard/models/x.kmodel",
              labels=["target"], rgb888p_size=[640, 480],
              model_input_size=[320, 320], display_size=[640, 480],
              conf_thresh=0.25, nms_thresh=0.45, max_boxes_num=20)
yolo.config_preprocess()
res = yolo.run(arr)     # arr: (3,H,W) uint8，RGB 平面
# res = ([boxes], [class_ids], [scores])，box = x,y,w,h
```

`mode="image"` 时返回的坐标是 `rgb888p_size` 坐标系；`mode="video"` 时
映射到 `display_size`。本包的 `board/yolo_stream.py` 用 `mode="image"`，
让检测框和回传 JPEG 使用同一坐标系。

---

## 1. 采集标靶图片

```bash
# 连续抓 300 张（按板端帧率）
python3 -m k230tool grab -n 300 -o captures

# 或每 0.5 秒一张，Ctrl-C 停
python3 -m k230tool grab -n 0 --every 0.5 -o captures
```

采集建议：

- 覆盖不同**距离**（远/中/近）、**角度**（±30°）、**光照**（顺光/逆光/阴影）；
- 标靶在画面中的占比至少 1/8，太小的目标在 320 输入下只有几十像素；
- 包含一些**没有标靶**的背景帧（负样本），降低误检；
- 类别只有一类时，`--names target`。

---

## 2. 标注

### 2.1 自动粗标（同心圆标靶）

```bash
python3 tools/autolabel_target.py --images captures --out labels \
    --preview preview --roi 200,150,450,400 \
    --param2 20 --min-radius 10 --max-radius 100
```

- `--roi` 很重要：杂乱背景下全图 Hough 会被线缆/支架淹没；
  先用 `python3 -m k230tool snapshot -o frame.jpg` 看一眼，量出标靶区域。
- 输出 `labels/*.txt`（YOLO 格式）和 `preview/*.jpg`。
- **必须人工复核**：删除误检、补漏检。也可以用 Label Studio / labelImg /
  Roboflow 重新标；K230 抓的图可以直接导入。

### 2.2 手动标注

用任意 YOLO 标注工具，导出为：

```
captures/000001.jpg  ->  labels/000001.txt
captures/000002.jpg  ->  labels/000002.txt
```

每行 `class cx cy w h`（归一化到 0~1）。

---

## 3. 整理数据集

```bash
python3 tools/dataset_from_captures.py \
    --images captures --labels labels \
    --out dataset --names target --val-ratio 0.2
```

生成：

```
dataset/
├── images/train, images/val
├── labels/train, labels/val
└── data.yaml
```

`data.yaml` 里 `nc: 1`、`names: ['target']`。多类别用
`--names target,target_red`，顺序必须与训练/板端 `labels` 一致。

---

## 4. 训练

```bash
# 有 CUDA：--device 0
python3 tools/train_target.py --data dataset/data.yaml \
    --model yolov8n.pt --epochs 100 --imgsz 320 --batch 16 --device 0

# 纯 CPU 也能跑，建议减少轮数先验证流程
python3 tools/train_target.py --data dataset/data.yaml \
    --epochs 30 --imgsz 320 --batch 8 --device cpu
```

产出：`runs/target/exp/weights/best.pt`。

标靶类目标单一、特征强，通常 30~100 轮就够。mAP 看
`runs/target/exp/results.png` / `val_batch*_pred.jpg`。

---

## 5. 导出 ONNX（关键：不能带 NMS）

```bash
python3 tools/export_onnx.py \
    --weights runs/target/exp/weights/best.pt \
    --imgsz 320 --out best.onnx
```

正确输出：

```
输出张量： output0 [1, 4+nc, 2100]      # 320 输入：40²+20²+10² = 2100
```

如果看到 `[1, 300, 6]` 或 `[1, 84, 2100, ...]`，说明开了 NMS/end2end，
板端 `libs/YOLO.YOLOv8` 不认。本工具已强制 `nms=False`。

> 输入尺寸必须是 32 的倍数（320/384/416/480/512/640）。

---

## 6. 编译 kmodel

### 6.1 环境

```bash
pip3 install nncase nncase-kpu onnx onnxsim onnxruntime
# 需要 .NET 7 runtime；脚本会自动找 ~/.dotnet 或 /usr/share/dotnet
export DOTNET_ROOT=$HOME/.dotnet
```

验证：

```bash
python3 -c "import nncase; print('ok')"
```

### 6.2 编译

```bash
python3 tools/compile_kmodel.py \
    --onnx best.onnx --out best.kmodel \
    --input-size 320 \
    --calib-dir captures --samples 30 \
    --quant uint8 --w-quant uint8 --calibrate Kld
```

脚本内部按嘉楠官方 K230 YOLOv8 示例设置：

| 选项 | 值 |
| --- | --- |
| `target` | `k230` |
| `preprocess` | `True` |
| `input_type` | `uint8` |
| `input_shape` | `[1,3,320,320]` |
| `input_range` | `[0,1]`（uint8 反量化到 0~1） |
| `mean` / `std` | `[0,0,0]` / `[1,1,1]` |
| `input_layout` | `NCHW` |
| `swapRB` | `False`（板端给的是 RGB） |
| PTQ | `samples_count=30`, `Kld`, `uint8/uint8` |

校准集：

- 用 `captures/` 里的实拍图最稳，30~100 张；
- 没有校准集时可加 `--allow-synthetic-calib` 先跑通，但精度会差；
- 图片会被缩放到 320×320（与官方示例一致），不是 letterbox。

**常见报错**：

| 报错 | 原因/处理 |
| --- | --- |
| `Failed to get hostfxr path` | 没装/没设 .NET 7：`export DOTNET_ROOT=$HOME/.dotnet` |
| `Feed Value Is Invalid, need u8[1,3,320,320]` | 校准数据形状应为 `[1,3,H,W]`；本工具已处理 |
| `NNCASE_PLUGIN_PATH is not set` | 只影响 KPU 算子插件；本工具在 import 前自动设置 |
| `Unsupported operator` | 模型含 K230 不支持的算子（如后处理/NMS）；用原始 YOLO 输出重新导出 |
| kmodel 加载失败/输出乱 | nncase 版本与固件不匹配；换同版本 nncase 或固件 |

---

## 7. 推到板子并验证

```bash
python3 -m k230tool push best.kmodel --dest /sdcard/models/best.kmodel

# 实时看：live/latest.jpg + latest.json
python3 -m k230tool live --model /sdcard/models/best.kmodel --labels target --conf 0.25
```

也可以把 `board/yolo_preview_ide.py` 顶部的 `MODEL` 改成新模型，在 CanMV IDE 里看。

### 7.1 调参

| 参数 | 位置 | 说明 |
| --- | --- | --- |
| `--conf` | `live` | 置信度阈值；弱目标降到 0.05~0.15 |
| `--nms` | `live` | NMS IoU，密集目标可降到 0.3 |
| `--input-size` | `live` | 必须与 kmodel 输入一致（320） |
| `-w/-h` | `live` | 采集/回传分辨率；AI 用 `--aw/--ah` 单独指定 |
| `--q` | `live` | JPEG 质量；降低可提速 |
| `--fps` | `live` | 板端限速 |

`board/yolo_stream.py` 顶部的 `CFG` 行也可以手动改（PC 端下发时会整行替换）：

```python
CFG = {'w': 640, 'h': 480, 'q': 70, 'aw': 640, 'ah': 480,
       'model': '/sdcard/models/best.kmodel', 'labels': ['target'],
       'in_w': 320, 'in_h': 320, 'conf': 0.25, 'nms': 0.45,
       'max_boxes': 20, 'fps': 0, 'stats': 10, 'limit': 0,
       'sensor_id': 2, 'reset_on_exit': 1}
```

---

## 8. 精度不好怎么办

按影响从大到小排查：

1. **数据**：标靶占比太小、样本太单一、没有负样本 → 多距离/角度采集；
2. **标注**：框是否贴紧外圈、有没有漏标/错标（看 `preview/`）；
3. **输入尺寸**：320 对远距离小目标不够 → 换 416/480 重新训练+编译（帧率会降）；
4. **量化**：PTQ 掉点 → 增加校准集、换 `--calibrate NoClip`、或 `w-quant int16`；
5. **预处理**：kmodel 必须按本文 `input_range=[0,1]` 编译，板端不要重复归一化；
6. **类别/标签顺序**：`data.yaml`、`labels`、板端 `--labels` 必须一致。

---

## 9. 性能参考

| 模型/输入 | 板端推理 | 整链路（640×480 JPEG） |
| --- | --- | --- |
| YOLOv8n 320 | 28~31 ms | ~8.5 fps |
| YOLOv8n 416 | ~45 ms（估） | ~6 fps |
| 纯推流（无 AI） | — | ~12 fps |

瓶颈是 JPEG 编码（~73 ms）；想只做检测不回传大图，可以把 `--q` 降到 50、
分辨率降到 320×240，帧率能到 ~15 fps。
