# 激光点学习式 ROI 热图模型（E 路线）

> 2026-10-23。替代/兜底 `laser_repo.py` 的 find_blobs：用 128×128 小热图模型
> 直接学"激光长什么样"，对红杂物/手/反光更鲁棒。

## 1. 模型

| 项 | 值 |
| --- | --- |
| 输入 | 128×128×3 uint8（RGB packed，原生像素裁剪） |
| 输出 | 32×32 单通道 logits（stride 4，峰值即激光） |
| 结构 | 5 层 tiny CNN + 1 次上采样，90.6K 参数 |
| kmodel | `models/laser_heatmap.kmodel`，**0.11 MB** |
| 板端推理 | **1.00 ms**（实测） |

## 2. 训练（`tools/train_laser_heatmap.py`）

- 正样本：真实背景上合成 PSF（**强红团** / **白核+淡粉晕** 两种形态），
  另注入板上实测真实 on 帧（`on.png`、`lt_on_*`、`laser_on_static`、`k230_now`）。
- 负样本：同一批真实背景避开光斑裁剪；50% 对准"最红/最亮"像素，
  专学红钳子/红电线/靶纸白心/白墙高光。
- 合成图排除 `*_spot*`（自带光斑，做负样本会灌错标签）。
- 训练 10 epoch × 300 step，batch 32，CUDA 约 1 分钟。

## 3. 精度

| 测试 | 结果 |
| --- | --- |
| PC 模拟器：真实白核激光 | 峰值 **0.98**，定位 **~2px** |
| PC 模拟器：20 张真实负样本 | 最大 0.006，**0 误检** |
| 板端静态图（`laser_on_static`） | (153.9,302.1)，真值 (152,300)，**2.4px**，conf 0.86 |
| 板端实时 90 帧 | **90/90 命中**，(323,259)，conf 0.90~0.94 |
| PC 内容位移回放（±40px） | 21 点，**平均 3.5px / 最大 10.2px** |

## 4. 部署

```bash
cd k230
python3 -m k230tool push models/laser_heatmap.kmodel --dest /sdcard/models/laser_heatmap.kmodel
python3 -m k230tool push board/laser_heatmap.py     --dest /sdcard/laser_heatmap.py
```

`board/yolo_preview_ide.py` 已默认启用：

```python
LASER_MODEL = "/sdcard/models/laser_heatmap.kmodel"
LASER_ROI = 128
LASER_THR = 0.55
LASER_PRIOR = (317, 261)     # 当前相机朝向下的激光先验
```

初始化顺序：优先 `LaserHeat`；导入/加载失败自动回退 `laser_repo.py`。
ROI 跟随上一帧检测结果；丢失时保持先验。

## 5. FPS / 画质（`yolo_preview_ide.py`）

| 配置 | 实测 |
| --- | --- |
| **256 Pose/2 + 384×288 q85** | **17.8~18.1 fps**（当前默认，Pose 约 0.13s） |
| 256 Pose/1 + 320×240 q85 | 15.4~15.5 fps（Pose 每帧） |
| 256 Pose/1 + 384×288 q85 | 14.0~14.1 fps |
| 256 Pose/2 + 448×336 q85 | 15.5~15.7 fps |
| 256 Pose/5 + 512×384 q83 | 15.1~15.3 fps（画质优先） |
| 384 Pose/6 + 512×384 q85 | 14.9~15.1 fps |
| 仅激光 + 512×384 q90 | 15.2 fps |

> 线程并行实测无效：ENC10=293ms + POSE10=198ms，BOTH=492ms ≈ 串行（`img.compress` 持 GIL）。
> 硬件 VENC 与 Sensor 的 VB 池冲突（`vb config failed(3)`），需重启，暂不可用。

> Pose 256 裸 KPU **17.2ms**；`YOLOv8` 封装（ai2d+后处理）后 **23ms**（关掉激光仍是 23ms，
> 说明激光模型不拖慢 Pose）。Pose 更新延迟 = `AI_EVERY × 帧间隔`：/5≈0.33s，/2≈0.13s。
> 编码仍是唯一真瓶颈：640×480 q70 单编码 68ms，本固件做不到 640×480@15fps。

> 640×480 q70 单编码就 68ms（14.6fps 封顶），本固件**做不到** 640×480@15fps。

## 6. 调参

- `LASER_THR`：默认 0.55；误检多就提到 0.65~0.75，漏检多就降到 0.45。
- `LASER_PRIOR`：换机位/换场景后，把激光点的大致像素填进来。
- `LASER_ROI`：默认 128；激光会跑出画面就加大到 160/192（推理仍 < 2ms）。
- 若预览里激光明显变暗（开双通道后曝光下降），用
  `tools/train_laser_heatmap.py` 把亮度增强改到 `alpha 0.5~1.15` 再训一轮。

## 7. 已知限制

- 单帧、单目标；画面里同时有多个"激光级"亮点时仍可能选错（需时序关联）。
- 激光关闭时返回 `None`（正确拒检），不是 bug。
- 训练集里的红杂物/手主要来自现场帧，换场景建议补采硬负样本。
