# Pose 精度退化：诊断与 v3 训练方案

> 2026-10-21 现场实测。适用模型：`k230/models/pose_v3.*`（5 关键点：TL/TR/BR/BL + 靶心）。

## 1. 现象

用户反馈：外场（靶纸上带激光光斑）YOLO 的靶纸/靶心**识别率下降、误检率上升**。
> 注：数据集里的"激光光斑"是**训练用图像增强**（把白光斑合成到靶心上，
> 让模型不把光斑当成靶心），与板端激光识别无关 —— 后者已在 2026-10-22 删除。

用 USB 工具（`k230tool grab` / `k230tool live`）抓现场 60 帧实测：

| 模型 | 靶纸检出 | 笔电误检 | 备注 |
| --- | --- | --- | --- |
| `poseB_farhead_merged`（板端当时在跑） | **0 / 60** | **60 / 60** | 框到键盘，靶心 0.74 落在键帽上 |
| `spot_v1`（新训的光斑增强模型） | 56 / 60 | 22 / 60 | 靶纸置信度只有 0.21~0.35 |

现场靶纸只有 **33×49 px**（640×480 画面），而训练集 GT 的 min-side
中位数是 100 px、<60 px 的样本只占 7%、**没有 <50 px 的样本**。

## 2. 根因（四条，均已修）

### 2.1 训练集尺度分布错位

`dataset_pose_v2_B` 的 232 个正样本：

| 指标 | 值 |
| --- | --- |
| min-side 中位数 | 100.4 px |
| min-side 5% 分位 | 51.8 px |
| min-side <60 px | 17 / 232 = 7% |
| min-side <50 px | 0 |

32×48 的靶纸在 320 输入下只剩 16×24 px，模型基本没见过这个尺度。

**修法**：新增「整图缩小 0.30~0.65 倍后贴回 640×480」的 zoom-out 增强，
把 min-side <60 px 的样本占比提高到 43%（<40 px 占 23%）。

### 2.2 spot_v1 丢了硬负样本

`poseB` 用的 `dataset_pose_farneg_B` 含 **80 张空标注负样本**；
`spot_v1` 只用 `dataset_pose_v2_B`（只有 20 张空标注）再叠光斑。
笔电/键盘这类硬负样本从 80 降到 20，误检自然回来。

**修法**：以 `v2_B ∪ farneg_B`（内容哈希去重，328 张，正 248 / 负 80）
为底，再给每张负样本随机裁 2 块，并从现场帧里裁避开靶纸的硬负样本。

### 2.3 `flipud=0.5` 对 Pose 是错的

Ultralytics 的 `RandomFlip.apply_instances()`：

```python
if params["direction"] == "vertical":
    instances.flipud(params["h"])          # 上下翻 y
...
if params["flip_idx"] is not None:
    instances.keypoints = keypoints[:, flip_idx]   # 这里仍用左右镜像映射
```

`data.yaml` 的 `flip_idx: [1,0,3,2,4]` 只描述**左右**镜像（TL↔TR、BR↔BL）。
上下翻转后仍套这个映射，关键点语义直接被打乱——等于给一半样本灌了错标签。

**修法**：`flipud=0.0`。左右翻转保留（有 flip_idx 保护）。

### 2.4 `erasing=0.25` 会擦掉关键点

随机擦除 1/4 图像，经常正好盖住靶心/角点，标签却仍在原处，
逼模型在「靶心消失」时凭空回归。**修法**：`erasing=0.0`。

### 2.5 板端把「框中心」当成「左上角」

修完模型后做板端实测，发现框比关键点整体右下偏 11~18 px：

```
kpts 框:  [284.8, 196.1] - [307.0, 227.8]      ← 关键点围出来的框
det  框:  [295, 211, 26, 36] → [295,211]-[321,247]
```

YOLO 的原始输出是 `[cx, cy, w, h]`（中心点 + 宽高），
`board/yolo_pose.py` 和 `board/yolo_preview_ide.py` 的 `_pose_postprocess()`
都把 `cx,cy` 直接当成了左上角。后果：

- 画出来的框右下偏 `w/2, h/2`；
- ROI 跟着偏；
- `$KT` 的 `cx,cy` 是用 `box.x + box.w/2` 算的 → 再偏 `w/2`，
  **云台瞄准点整体偏半个框**（26×36 的靶纸约偏 13~18 px）。

修法：`x = cx - w/2; y = cy - h/2`（`yolo_pose.py` / `yolo_preview_ide.py` /
离线代理 `gimbal_agent.py` 三处都改）。修完板端 top-1 命中从 0/48 变 48/48
（之前 verify 脚本按框 IoU 匹配，框偏了自然全判错），靶心误差 1.5 px。

## 3. v3 数据集

构建脚本：`tools/build_pose_dataset_v3.py`

| 类别 | 数量 | 说明 |
| --- | --- | --- |
| 正样本原图 | 248 | `v2_B ∪ farneg_B` train 去重 |
| + 合成激光光斑 | 496 | 60% 压靶心 / 25% 靶纸其他位置 / 15% 靶外 |
| + zoom-out（远距离小靶） | 248 | 0.30~0.65 倍，replicate padding |
| + zoom-out + 靶心光斑 | 248 | 最难的组合：远处小靶被激光压住 |
| + 现场帧 | 10 | 2026-10-21 实拍（33×49 px），多帧中值标注 |
| 负样本 | 257 | 80 原图 + 160 随机裁块 + 17 现场硬负样本 |
| **train 合计** | **1507** | 正 1250 / 负 257（负样本 17%） |

- val：`farneg_B` val 12 张 + 现场帧 16 张 = 28 张；
- test：`farneg_B` test 23 张 + 现场帧 34 张 = 57 张；
- 现场帧标注工具：`tools/label_scene_frames.py`（多帧预测取中值 + 4x 核对图）。

## 4. 训练配方

脚本：`tools/train_pose_v3.py`（最初从 `spot_v1.pt` 微调；
该权重已随精简删除，现在重训直接以 `models/pose_v3.pt` 为起点）

| 项目 | 值 | 理由 |
| --- | --- | --- |
| imgsz / batch | 320 / 16 | 与板端 kmodel 一致 |
| epochs | 40 | CPU 约 40 s/轮 |
| optimizer / lr | AdamW / 0.001，cosine，warmup 3 | 小数据微调 |
| mosaic / close_mosaic | 1.0 / 8 | 小目标多尺度 |
| degrees / translate / scale | 12° / 0.12 / 0.55 | 轻量几何增强 |
| fliplr / **flipud** | 0.5 / **0.0** | Pose 不能上下翻 |
| **erasing** | **0.0** | 不能擦关键点 |
| hsv | 0.015 / 0.6 / 0.5 | 光照鲁棒 |
| freeze | 0 | 全量微调 |

### 5.1 多样化测试集（`farneg_B` test 23 张 + 合成光斑/缩小变体 + 80 张负样本）

`conf=0.25`，top-1 指板端「取最高分候选」后的命中率：

| 模型 | 干净 top1 | 合成光斑 top1 | 缩小远靶 top1 | 靶心误差（干净 / 远靶） | 负样本误检 |
| --- | --- | --- | --- | --- | --- |
| `poseB_farhead_merged` | 100% | 82.6% | **50.0%** | 4.18 / 4.73 px | 1.2% |
| `spot_v1` | 100% | 100% | **56.5%** | 4.03 / 2.48 px | 1.2% |
| **`pose_v3`** | **100%** | **100%** | **100%** | **1.99 / 1.48 px** | **1.2%** |

- v3 的远靶靶心误差 **1.48 px / P95 3.73 px**，比 spot_v1 的 2.48 / 6.30 明显更好；
- v3 在 `conf=0.4` 下远靶仍是 100% 召回，负样本误检 0%；
- v3 test（57 张，含现场帧）：top1 100%，靶心 1.6 px；spot_v1 只有 77.2% / 4.4 px。

### 5.2 全新视角（`scene_20261021_fresh`，靶纸 26×38 px，未参与训练）

| 模型 | conf=0.25 top1 | 靶心中位 | 说明 |
| --- | --- | --- | --- |
| `poseB_farhead_merged` | 0% | — | 完全检不到 |
| `spot_v1` | 0% | — | 完全检不到 |
| **`pose_v3`** | **80%** | **1.83 px** | 4 帧置信度 0.08~0.20（模糊/逆光） |
| **`pose_v3` + 低置信复筛** | **95%** | **~1.8 px** | 复筛把 0.02~0.20 拉回 0.5~0.8，负样本误检不变 |

残留的 2 帧是连拍开头相机未稳定时的运动模糊；实时 13fps 下由控制器
的中值/低通滤波兜住。

### 5.3 板端 kmodel 实测（2026-10-21 夜，`verify_live.py`；该脚本已随精简删除）

```
帧率 13.3 fps   推理 31 ms
conf>=0.25: top-1 命中 48/48 = 100.0%   误检 0   漏检 0
靶心误差 mean 1.5  median 1.5  P95 1.7 px
```

kmodel（nncase Simulator 量化后）：`farneg_B` test 23/23 命中、靶心 3.6 px；
全新视角 14/20（低置信 6 帧，开复筛后与 PyTorch 一致）。

## 6. 部署

```bash
cd k230
python3 tools/export_onnx.py --weights models/pose_v3.pt --imgsz 256 --out models/pose_v3_256.onnx
python3 tools/compile_kmodel.py --onnx models/pose_v3_256.onnx --out models/pose_v3_256.kmodel \
    --input-size 256 --calib-dir ../recovered/dataset_pose_v3/images/train --samples 20
python3 -m k230tool push models/pose_v3_256.kmodel --dest /sdcard/models/pose_v3_256.kmodel
python3 -m k230tool live --duration 20 --conf 0.25      # 默认已经是 pose_v3_256.kmodel
```

**路径说明**：板端只保留这一份模型 `/sdcard/models/pose_v3_256.kmodel`。
`k230tool/link.py`、`board/yolo_pose.py`、`board/yolo_preview_ide.py`、
`board/gimbal_agent.py` 四处默认 MODEL 一致。模型被 KPU 占用时直接覆盖会报
EBUSY，先 `bash scripts/mode.sh debug` 停跑再推。

板端默认配置：

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `in_w/in_h` | 320/320 | 与 kmodel 一致；换 416 要三处一起改并重新编译 |
| `conf` | 0.25 | v3 在 0.4 下远靶仍 100%，可按现场再提高 |
| `low_refine` | 1 | 低置信小靶（≤120px）裁 7 倍放大复筛，冷却 3 帧 |
| `perf_mode` | `performance` | 板端大核 1600MHz（见 `board/k230_perf.py`，不插 USB 时尤其重要） |
| `jpeg_q` | 同 `q` | 回传 JPEG 质量，ROI 建议 88~92 |
| `low_refine_topk` | 2 | 只对前 2 个候选复筛 |

板端回传的 JPEG 是**原生像素的检测 ROI**（不是缩小图），所以小靶在
`latest.jpg` 里也看得清；没有目标时才发一张缩小全图给出上下文。

## 7. 复现

```bash
# 1) 现场帧标注（相机静止时）
python3 tools/label_scene_frames.py --frames /tmp/scene_frames \
    --weights models/pose_v3.pt --imgsz 416 --conf 0.25 \
    --out /tmp/scene_ds/scene_labels.json --sheet /tmp/scene_ds/check.png \
    --crop 295,180,400,270 --zoom 5

# 2) 构建数据集
python3 tools/build_pose_dataset_v3.py \
    --sources ../recovered/k230_target_project/dataset_pose_v2_B \
              ../recovered/k230_target_project/dataset_pose_farneg_B \
    --val-src ../recovered/k230_target_project/dataset_pose_farneg_B \
    --test-src ../recovered/k230_target_project/dataset_pose_farneg_B \
    --scene-frames /tmp/scene_frames --scene-labels /tmp/scene_ds/scene_labels.json \
    --out ../recovered/dataset_pose_v3

# 3) 训练
python3 tools/train_pose_v3.py --data ../recovered/dataset_pose_v3/data.yaml \
    --weights models/pose_v3.pt --epochs 40 --imgsz 320 --batch 16 --device cpu \
    --project ../runs --name pose_v3

# 4) 评测（--weights 支持多个，历史权重已删除，这里以当年的 pose_v3.pt 为基线）
python3 tools/bench_pose.py --weights models/pose_v3.pt \
    --data ../recovered/k230_target_project/dataset_pose_farneg_B --splits test \
    --spot-copies 1 --far-copies 2 --far-spot \
    --neg-data ../recovered/k230_target_project/dataset_pose_farneg_B \
    --confs 0.1 0.25 0.4 --out eval_v3.json
```

## 8. 后续建议

1. **继续补小目标实拍**：30~50 px 的靶纸是最缺的，尤其是斜视/低照度；
2. **现场硬负样本**：笔电键盘、白色插座、纸箱、人的手，纳入训练集；
3. 若还要提升钉住的靶心精度，板端可开 `far_refine=1`（裁剪放大复筛，
   每帧多一次推理，约 +35 ms）；
4. 更小的目标（<20 px）再靠模型收益有限，应该改用「ROI 跟踪推理」：
   锁定后只对上一帧位置周围的 320×320 窗口推理，等效分辨率翻倍，
   代价与全图 320 相同。
