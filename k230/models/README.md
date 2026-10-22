# 本地模型目录

只保留**当前部署**的这一份模型。历史模型（poseA / poseB / spot_v1 / target_v2 /
v2_std1）已在 2026-10-22 工程精简时删除，快照在
`scratch/删除归档_20261022/pre_cleanup_part1.tar.gz`。

## 当前唯一模型：`pose_v3_384.kmodel`

| 文件 | 说明 |
| --- | --- |
| `pose_v3_384.kmodel` | **部署到板端**：5 关键点 Pose，384 输入，nncase Q1，3.48 MB |
| `pose_v3.pt` | 对应的 PyTorch 权重：重训起点 / 重新导出 ONNX 用 |
| `labels_target.txt` | 类别名（1 类 `target`） |

- 输入 `uint8 [1,3,384,384]`，输出 `float32 [1,20,3024]`（4 box + 1 cls + 5×3 关键点）
- 板端默认路径：`/sdcard/models/pose_v3_384.kmodel`
  （`k230tool/link.py`、`board/yolo_pose.py`、`board/yolo_preview_ide.py`、
  `board/gimbal_agent.py` 四处一致）
- 板端推理耗时待实测（256 输入版本约 22 ms，320 约 34 ms）
- 训练数据：`recovered/dataset_pose_v3/`，方法见
  [`../docs/POSE_V3_TRAINING.md`](../docs/POSE_V3_TRAINING.md)

## 推到板子

```bash
python3 -m k230tool push models/pose_v3_384.kmodel --dest /sdcard/models/pose_v3_384.kmodel
python3 -m k230tool models          # 确认板端能看到
```

> 模型被 KPU 占用时直接覆盖会报 EBUSY；先 `bash scripts/mode.sh debug` 让板端停跑，
> 再推。

## 重新导出 / 重新编译

```bash
cd k230
# 1) pt -> ONNX（384 输入，nms=False）
python3 tools/export_onnx.py --weights models/pose_v3.pt --imgsz 384 --out models/pose_v3_384.onnx

# 2) ONNX -> kmodel（nncase PTQ，用训练集图像做校准）
python3 tools/compile_kmodel.py --onnx models/pose_v3_384.onnx --out models/pose_v3_384.kmodel \
    --input-size 384 --calib-dir ../recovered/dataset_pose_v3/images/train --samples 20

# 3) 精度核对（nncase Simulator，不需要板子）
python3 tools/kmodel_eval.py --kmodel models/pose_v3_384.kmodel \
    --data ../recovered/dataset_pose_v3 --split test --input-size 384
```

中间产物（`.onnx`、`tmp_nncase/`、`gmodel_dump_dir/`）都在 `.gitignore` 里，
需要时按上面的命令重新生成。

## 标签文件

`labels_target.txt` —— 1 类 `target`。多类别时每行一个，顺序必须与
`data.yaml` 的 `names`、板端 `--labels-file` 一致。
