# 本地模型目录

把 kmodel / 标签放在这里，作为一个 **可移植的备份/来源**。推荐布局：

```
models/
├── best.kmodel           你训练并编译好的模型
├── labels_target.txt     类别名（每行一个，顺序与训练一致）
└── README.md
```

推到板子：

```bash
python3 -m k230tool push models/best.kmodel --dest /sdcard/models/best.kmodel
python3 -m k230tool models          # 确认板端能看到
```

## ★ 本项目训练出的模型：`target_v2.kmodel`

用 `k230_target_project/` 里的数据集训练（YOLOv8n @320，1 类 `target`），
板端 30 秒 / 251 帧实测：

| 指标 | 结果 |
| --- | --- |
| 有检出帧 | **251 / 251 = 100%** |
| 多检（误检）帧 | 1 / 251 |
| 平均置信度 | 0.668 |
| 平均推理 | 27.9 ms |
| 帧率 | 8.0 fps |

推板：

```bash
python3 -m k230tool push models/target_v2.kmodel --dest /sdcard/models/target_v2.kmodel
python3 -m k230tool live            # 默认就用 target_v2.kmodel
```

训练数据/权重/日志在包外的 `k230_target_project/`，见那边的 README。

## 分割验证模型：`target_seg_template.kmodel`

用「检测框转矩形 mask」训的 **YOLOv8n-seg（30 轮）**，只用于验证板端分割链路：

- 输入 `uint8 [1,3,320,320]`，双输出（检测头 + mask 原型 32×80×80）；
- 板端实测 5.3 fps（含 24 点轮廓提取），推理 56 ms；
- **mask 是矩形**，轮廓只能近似矩形，不能真正贴边；
- 真实贴边效果需要用多边形标注重新训练，见 `docs/ANNOTATION.md`。

用法：

```bash
python3 -m k230tool live --task segment \
    --model /sdcard/models/target_seg.kmodel --conf 0.05
```

> 本目录还备份了 `v2_std1.kmodel`（约 3.34 MB，旧模型），换机器/换板子时可直接推回去：
> `python3 -m k230tool push models/v2_std1.kmodel --dest /sdcard/models/v2_std1.kmodel`
> 另外两份 target 模型仍在板端，用 `python3 -m k230tool pull ...` 自行备份。

## 板端现有的模型（本机实测）

`/sdcard/models/` 里已有三份 1 类目标模型，结构均为：

| 项目 | 值 |
| --- | --- |
| 输入 | `uint8 [1,3,320,320]` |
| 输出 | `float32 [1,5,2100]`（4 box + 1 class） |
| 大小 | ~3.49 MB |

| 文件名 | 备注 |
| --- | --- |
| `target_yolov8n_320.kmodel` | 当前桌面场景低阈值下未检出 |
| `target_yolov8n_320_v2.kmodel` | 同上 |
| `v2_std1.kmodel` | 能给出目标框，但当前桌面场景框到了线缆/支架（与现场标靶不匹配，置信度 0.08~0.35） |
| `yolov8n_320.kmodel` | 本包用 COCO `yolov8n.pt` 重新编译的验证模型（80 类，证明工具链可用） |

> 这三份模型在「桌面 + 同心圆靶纸」场景下没有框准标靶；用本包流程重新编译的
> COCO yolov8n 能正确检出 person，说明链路/编译正常，问题在训练数据与现场目标的匹配。
> 建议按 `docs/YOLOV8_DEPLOY.md` 用现场标靶重新采集训练。

## 标签

- `labels_target.txt`：标靶模型用
- `coco_labels.txt`：验证用 COCO 模型（`yolov8n_320.kmodel`）的 80 类标签

`labels_target.txt`：

```
target
```

多类别时每行一个，顺序必须与 `data.yaml` 的 `names`、板端 `--labels` 一致。

板端运行：

```bash
python3 -m k230tool live --model /sdcard/models/best.kmodel \
    --labels-file models/labels_target.txt --conf 0.25

# 验证 COCO 模型（应能检出 person / cup / keyboard 等）
python3 -m k230tool live --model /sdcard/models/yolov8n_320.kmodel \
    --labels-file models/coco_labels.txt --conf 0.25
```

## 本仓库保留的权重（2026-10-20 清理后）

| 文件 | 说明 |
| --- | --- |
| `target_pose.kmodel` | **当前部署模型**：`poseB_farhead_merged` Q1（farneg 检测头 + B_final 关键点分支），板端路径 `/sdcard/models/target_pose.kmodel` |
| `poseB_farhead_merged.pt` | 上面模型的 PyTorch 权重（重新导出 ONNX / 再训练起点） |
| `poseB_farhead_merged.onnx` | 导出的 ONNX（`nms=False`，输入 320×320），`tools/compile_kmodel.py` 的输入 |
| `poseA_final_best.pt` | A 版最终权重（4 角点，held-out 靶心 6.04 px，留档对照） |
| `target_v2.kmodel` / `target_v2.pt` | v2 数据集训练的历史版本（100% 检出，置信度 0.668） |
| `v2_std1.kmodel` | v2 的另一组预处理标定（std=1）版本 |

推模型到板子：

```bash
python3 -m k230tool push models/target_pose.kmodel --dest /sdcard/models/target_pose.kmodel
python3 -m k230tool models        # 确认板端能看到
```

> 训练工程（数据集 / captures / runs）已随 2026-10-20 清理删除，归档在
> 需要重训时用你自己的数据集与标注。
