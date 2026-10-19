# 5 关键点标注规范（A/B 对照实验）

> **注意（2026-10-20 清理）**：历史标注 `pose_labels*.json` 与训练工程
> `k230_target_project/`（数据集 / captures / runs）已删除，归档在工程外
> 未随本仓库发布；下文命令里的路径仅作流程示例。

> 目标：训练 **A=4 角点** 与 **B=4 角点+靶心** 两个 Pose 版本，在同一批 K230 实拍图上比较靶心精度、稳定性与帧率。

---

## 1. 关键点定义（顺序严格固定）

| 序号 | 名称 | 定义 |
| --- | --- | --- |
| 1 | TL | 靶纸/底板外缘的**左上物理角点** |
| 2 | TR | 右上物理角点 |
| 3 | BR | 右下物理角点 |
| 4 | BL | 左下物理角点 |
| 5 | BULL | 白色 10 号靶心区域（白色圆盘）的**中心** |

注意：

- 角点是**物理外缘**，不是检测框、不是黑色印刷区、不是同心圆外接框；
- 圆角卡片：取两条外边缘延长线的交点（理论尖角）；
- 靶心取白色 10 号圆盘的几何中心；不要取黑色数字“10”的中心；
- 检测框只提供特征区域，**框不需要卡得极准**，但必须完整覆盖靶纸。

---

## 2. 可见性 v

| v | 含义 | 训练/评估处理 |
| --- | --- | --- |
| 2 | 清晰可见 | 正常监督 |
| 1 | 被遮挡/出画，但可由边缘、对称性推断 | 保留坐标；v=1 仍参与损失 |
| 0 | 完全不可辨，无法合理推断 | 不参与关键点损失；该图不进入靶心误差统计 |

规则：

- 角点被手/物体挡住但相邻边可见：标 **v=1**，坐标放在推断位置；
- 角点完全出画：标 **v=1**，坐标可放在图像边缘（工具会夹到边缘，可接受）；
- 靶心被手完全遮住：标 **v=0**；该帧仍可用于 A 版角点训练，但不参与 B 版靶心指标；
- 一张图最多允许 2 个角点 v=0；否则跳过该图。

---

## 3. 标注工具（已提供）

启动本地服务（**不要双击 HTML**，浏览器对本地文件有限制）：

```bash
cd /home/khalil/personal_prj/K230
python3 -m http.server 8765 --bind 127.0.0.1
```

浏览器打开：

```
http://127.0.0.1:8765/k230/tools/pose_label.html
```

步骤：

1. 点 **“加载试点 20 张（推荐）”**（不用选文件夹）；
2. 左侧列表自动出现 20 张；
3. 每张图拖动 5 个点到正确位置：
   - `1-5` 选点；
   - 鼠标拖动；
   - `V` 切换可见性 2→1→0；
   - `R` 恢复预标注；
   - `N`/`P` 下一张/上一张；
4. 点 **“完成并下一张”**；
5. 全部完成后点 **“导出 pose_labels.json”**；
6. 把下载的文件移动到：

```
/home/khalil/personal_prj/K230/pose_labels.json
```

> 预标注来自 `target_v2` 检测框的四角 + 框中心，**只是起点，必须逐个拖到真实角点/靶心**。
> 进度存在浏览器 localStorage，刷新不丢；也可“导入 JSON 继续”。
> 如果按钮无反应，先看页面顶部红色错误条；多数是页面用 `file://` 打开导致图片加载失败。

### 先标 20 张试点

试点 20 张已按阶段均衡选好（覆盖 black_move / pos_mid / pos_far / pos_robot / pos_occl / pos_blur / pos_light / pos_near / pos_angle）。导出后我先检查格式与一致性，
再决定是否需要调整定义，避免整批返工。

---

## 4. 数据划分与 A/B 数据集生成

```bash
cd k230

python3 tools/pose_json_to_yolo.py \
    --json ../pose_labels.json \
    --captures ../k230_target_project/captures \
    --out-root ../k230_target_project
```

输出：

```
k230_target_project/
├── dataset_pose_A/     # 4 角点（kpt_shape [4,3]）
└── dataset_pose_B/     # 4 角点 + 靶心（kpt_shape [5,3]）
```

- 两版使用**完全相同**的图片与四角标注，只差第 5 点；
- 按帧号 `%10` 划分 train/val/test，避免相邻帧泄漏；
- 文件名带阶段前缀，避免不同阶段同名覆盖；
- 每版自带 `data.yaml`。

---

## 5. 训练 A/B（同条件）

```bash
# A：4 角点
python3 tools/train_pose_target.py \
    --data ../k230_target_project/dataset_pose_A/data.yaml \
    --model ../k230_target_project/yolov8n.pt \
    --epochs 100 --imgsz 320 --batch 8 --device cpu \
    --project ../k230_target_project/runs_target --name poseA

# B：5 关键点（4 角点 + 靶心）
python3 tools/train_pose_target.py \
    --data ../k230_target_project/dataset_pose_B/data.yaml \
    --model ../k230_target_project/yolov8n.pt \
    --epochs 100 --imgsz 320 --batch 8 --device cpu \
    --project ../k230_target_project/runs_target --name poseB
```

要求：

- 相同 `imgsz/epochs/batch/增强/初始化权重`；
- 日志记录 `box_loss / pose_loss / kpt_loss`，确认关键点损失没有被框损失淹没；
- 若 B 的 `kpt_loss` 明显偏小或没下降，检查第 5 点标注与 v 值。

---

## 6. 评估指标（同一批 K230 实拍测试图）

按 `dataset_pose_*/images/test` 评估：

- 靶心中心像素误差：均值、P95、最大值；
- 角点误差：均值/P95，按 TL/TR/BR/BL 分列；
- 角点出框率（预测点落在图像外的比例）；
- 靶心关键点可见率（v≥0.5）与回归成功率；
- 失败态：误报 fail / 漏报 fail；
- 端到端：识别循环 fps、pose_ms、图像流 fps。

评估命令：

```bash
python3 tools/pose_eval.py --weights runs_target/poseB/weights/best.pt \
    --data ../k230_target_project/dataset_pose_B --kpt 5 --split test
python3 tools/pose_eval.py --weights runs_target/poseA/weights/best.pt \
    --data ../k230_target_project/dataset_pose_A \
    --gt-data ../k230_target_project/dataset_pose_B --kpt 4 --split test
```

---

## 7. 失败判定规则（两版一致）

A 版：

1. 框置信度 < `conf` → fail；
2. 四角任一 v < `v_min`（建议 0.3）→ fail；
3. 凸性校验失败（叉积符号不一致）→ fail；
4. 边长比例异常（对边比 > 2.5，或最短边 < 8px）→ fail；
5. 角点出框率 > 1 个 → fail；
6. 单应矩阵退化（行列式接近 0）→ fail。

B 版：

1. 同 A 的 1~6；
2. 第 5 点 v < `v_min` → fail（不输出靶心坐标）；
3. 第 5 点落在四角凸包外 → fail；
4. 可选一致性校验：用四角单应映射标准靶心，与第 5 点距离 > 阈值 → fail（建议阈值 8px @ 320）。

所有 fail 帧输出 `status` 与原因，不输出坐标。
