# 激光检测修复：`LASER_CFG` 生效 + 尺寸/面积门限

> 2026-10-23。针对"蓝十字锁到红杂物 / 手 / 反光"的问题，先做**不改判据**的
> 最小修复。涉及 `board/laser_repo.py`（VERSION `repo-port-2-gated`）和
> `board/yolo_preview_ide.py` 的 `LASER_CFG`。

## 1. 原来的问题（代码级）

`laser_repo.py` 是 `zhouzhou-code/elec_vision_k230` 的裸移植：

```python
img.find_blobs([RED_LAB], False, roi=center_320x240,
               merge=True, pixels_threshold=1)
max(blobs, key=lambda b: (b.area(), -b.y()))
```

- 直接"取面积最大"：红袋子 / 红电线 / 橙色绳 / 手都比 8×10 的激光点大；
- `merge=True` 会把"红杂物 + 激光"并成一个大块；
- `yolo_preview_ide.py` 的 `LASER_CFG` 只是传给 `LaserRepo`，但构造函数只收
  `roi_scale`，所以**调 `LASER_CFG` 完全无效**（注释里"阈值已收紧"并没有生效）。

## 2. 现在的行为

`LaserRepo(**kw)` 接受全部参数；先过滤、再取最大：

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `min_area` | 0 | bbox 面积下限（`w*h`） |
| `max_area` | 0 | bbox 面积上限；拒红袋子/大片红 |
| `max_side` | 0 | 单边上限；拒手/白屏/大块 |
| `max_aspect` | 0.0 | 长宽比上限；拒红电线/橙绳 |
| `min_fill` | 0.0 | 填充率下限；拒细长/稀疏块 |
| `threshold` | `RED_LAB` | LAB 阈值，可传单 tuple 或多区间 list |
| `merge` / `margin` | `True` / `1` | 原版合并策略 |
| `roi` / `roi_scale` | 居中 320×240 | 搜索窗口 |

门限为 0 = 关闭，因此 `LaserRepo()` 无参调用仍等于原版行为，
`scripts/accept_laser_static.py` 不受影响。

预览脚本默认（`yolo_preview_ide.py` 顶部）：

```python
LASER_CFG = {"min_area": 6, "max_area": 400, "max_side": 24, "max_aspect": 2.5}
```

## 3. 现场怎么调

看串口/IDE 打印的 `laser ... | lms Xms raw N kept M`：

- `raw` 很大、`kept=0`：门限太紧（把激光也拒了），先放宽 `max_area/max_side`；
- `kept` 有多个且坐标跳：说明还有同尺寸候选，把 `max_aspect` 收到 2.0、
  或加 `min_fill`；
- 红杂物仍被选中：确认它的 bbox 面积/单边是否 < 门限，必要时把
  `max_area` 收到 300、`max_side` 收到 20。

> 注意：这是单帧门限，**挡得住"大片/细长"的红杂物和手，挡不住"与激光同尺寸
> 的反光/散射点"**（`LASER_ROI_MINIMAL.md` 已证明单帧几何无法区分）。
> 反射光要靠下一步的多帧关联或窄带滤光片。

## 4. 上板验证

```bash
cd k230
python3 -m k230tool push board/laser_repo.py       --dest /sdcard/laser_repo.py
python3 -m k230tool push board/yolo_preview_ide.py --dest /sdcard/yolo_preview_ide.py
# IDE 里打开 yolo_preview_ide.py 运行，观察蓝十字是否还锁红杂物/手
```

板端绘制自检：把 `DBG_SAVE` 改 120，跑一会儿后

```bash
python3 -m k230tool pull /sdcard/pose_preview_dbg.jpg .
```
