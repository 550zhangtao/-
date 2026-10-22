# 激光点检测测试脚本（`board/laser_detect_test.py`）

独立测试脚本：用 **cv_lite 灰度 `find_blobs`** 找激光点，画十字并打印坐标/面积。
不依赖 YOLO、不依赖舵机/UART。

---

## 1. 怎么跑

**CanMV IDE**：打开 `board/laser_detect_test.py`，点运行（画面在 IDE 里看）。

**无 IDE（自动化/批量测试）**：

```bash
cd k230
python3 -m k230tool mode debug                      # 让 REPL 空闲
python3 -m k230tool push board/laser_detect_test.py --dest /sdcard/laser_detect_test.py
python3 - <<'PY'
from k230tool.link import K230Link
link = K230Link(); link.open()
out, err, ok = link.exec_and_collect(
    "exec(open('/sdcard/laser_detect_test.py').read())", timeout=120)
print(out[-3000:])
link.close()
PY
python3 -m k230tool pull /sdcard/laser_detect_log.csv .   # 拉逐帧记录
```

跑批时把文件顶部 `ROUNDS = 1`、`SECONDS = 15`、`DRAW_ON_IDE = False` 改一下，
跑完自动退出（省得手动停）。

---

## 2. 参数（都在文件顶部）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `THRESHOLD` | `(210, 255)` | **唯一需要手调的**：灰度二值区间 |
| `AREA_THRESHOLD` | `6` | 外接**框**面积下限（OpenMV 的 area 语义） |
| `MIN_PIXELS` | `6` | 真实**亮像素数**下限 |
| `MERGE` / `MERGE_GAP` | `True` / `3` | 相近小块合并 |
| `MAX_PIXELS` / `MAX_SIDE` | `150` / `24` | 光斑**上限**，用来排除灯/墙/手电筒 |
| `MIN_FILL` | `0.45` | 填充率；激光点圆而实，灯带/反光条细长 |
| `KERNEL_SIZE` | `1` | 形态学核；**保持 1**（激光核心可能只有 3~4px，开运算会抹掉） |
| `LOCK_R` / `LOCK_LOST` | `40` / `12` | 时序锁定：在上一帧位置附近找，丢 12 帧才重新全局搜 |
| `CORE_X/Y`、`GATE_R` | 中心 / `0` | 位置先验；`GATE_R>0` 才启用门限 |

### 阈值怎么调

| 场景 | 建议 `THRESHOLD[0]` |
| --- | --- |
| 关灯 / 暗背景 | 170 ~ 200 |
| 室内常光（白墙 + 台灯） | **200 ~ 225（默认 210）** |
| 开灯直射白墙 / 手电筒照靶 | 230 ~ 250 |

调法：从低往高加，直到日志里只剩一个候选、且 `cx/cy` 不再跳。

**实测教训（很重要）**：这台机器上激光核心只有 **3×4 px**、峰值 255。阈值设到
240 时核心会被开运算抹掉，检测器转而锁定天花板灯/噪点（`cx` 标准差 121px）；
降到 210 后整帧只剩激光一个候选，`cx` 标准差 **0.00**。所以：

> 阈值太高不是"更干净"，而是**把激光自己过滤掉了**，然后去锁别的东西。

---

## 3. cv_lite 和 OpenMV `find_blobs` 的差别（实测本机固件）

```python
cv_lite.grayscale_find_blobs(image_shape, data, thr_lo, thr_hi, min_area, kernel_size)
#   -> [x, y, w, h, ...]   （只有外接矩形，没有面积、没有 merge、没有 pixels_threshold）
```

所以脚本自己补了三件事：

1. **`merge=True`** → `_merge_close()`：质心距离 ≤ `MERGE_GAP` 的小块合成一个；
2. **`pixels_threshold`** → 亮像素数。小框直接用 `w*h`（实心，误差小），
   大框才用 8 连通逐像素拆（`_split_by_gray`）；
3. **上下限过滤** → 只认「小 + 亮 + 圆」的块。

> 踩坑记录：一开始用 tuple 当 visited 集合，一块几百像素的亮斑就让单帧后处理
> 从 5ms 涨到 **64ms**（掉到 13fps）；换成一维 `bytearray` 标记后回到 **5ms**。

---

## 4. 实测（CanMV v1.4.3 / GC2093 / 640×480 灰度）

| 项目 | 结果 |
| --- | --- |
| 帧率（`DRAW_ON_IDE=True`，IDE 预览） | **56.8 fps**（455 帧 / 8s） |
| 帧率（`DRAW_ON_IDE=False`） | 56.3 ~ 57.1 fps |
| `cv_lite` 单次 | 3 ~ 5 ms |
| 后处理 | 约 1 ms（快路径）／最坏 45ms（旧的逐像素泛洪，已弃用） |
| 采集 | 57 fps（17.5 ms/帧，就是上限） |

> **帧率天花板就是采集 57fps**，检测本身只占 4~5ms。再快只能降分辨率或隔帧。

**每帧存图会掉帧**：`SAVE_HIT` 每帧 `img.save()` 会把 57fps 拖到 **9fps**，
所以默认 `SAVE_HIT=""`，调试时用 `SAVE_HIT_MAX=1` 只存一张。

### 性能关键：不要逐像素拆分

一开始为了补 `merge`/`area` 语义，对每个"外接框较大的块"做 8 连通泛洪：
一块 900px 的亮斑要 **7ms**，场景有 20+ 亮块时后处理飙到 **45ms（掉到 16fps）**。
现在改成：**超过 `MAX_PIXELS*2` 的框直接拒掉**（既然它不可能是激光），
后处理降到 ~1ms，帧率回到 57fps。

---

## 5. 检出准确性（数据集核对）

用 `recovered/dataset_pose_v3` 的 300 张（含 128 张无激光的负样本）离线跑同一套逻辑：

| `THRESHOLD` | 正样本检出率 | 负样本误检率 |
| --- | --- | --- |
| 230 | **99%** (170/172) | 48% |
| 240（默认） | 92% (159/172) | 45% |
| 248 | 94% (161/172) | 45% |

**怎么读这个表**：灰度只能靠「亮度 + 大小」区分，数据集里没有激光的负样本
本身就是「台灯 / 白纸 / 键盘」的近景裁块，其中约一半存在和激光点一样
「小 + 亮 + 圆」的饱和小块，所以离线误检率看起来高。

现场实测反而干净：连续 451 帧，命中 450（99.8%），`cx` 标准差 **0.00**、
`cy` 标准差 0.39（锁定在 271,282）；IDE 预览轮 455 帧 **100%** 命中。

**准确性核对方法**（每次改完参数都该做一次）：脚本会同时存
`/sdcard/laser_hit_raw.jpg`（未绘制）和 `/sdcard/laser_hit.jpg`（画了十字），
拉回 PC 做差分：

```python
diff = cv2.absdiff(raw, drawn); ys, xs = np.where(diff > 20)
print(xs.mean(), ys.mean())     # 十字质心
```

实测十字质心 (271.4, 282.8) vs 检测点 (271, 283) → 偏差 **0.4px**，
且逐像素核对该点峰值 **255**、周围背景 160~190，确认压在激光上。

要再压误检，按现场加位置门限：激光笔与相机相对固定，把 `CORE_X/CORE_Y`
调到实测的那个点，`GATE_R` 设 0.2~0.35。

---

## 6. 已知限制

- **灰度无法区分「小亮斑」和「小反光」**：真实现场靠「激光位置固定」+
  `MAX_*` 组合解决；换场景（换背景/换灯位）后建议重新确认 `THRESHOLD`。
- 手电筒直射时整幅画面可能饱和，此时 `THRESHOLD[0]` 要提到 248~250，
  或把光斑调小/远离直射。
- 每帧 `img.save()` 会显著掉帧，只在调试时用，且限制张数。

---

## 7. 在 CanMV IDE 预览里看激光（`yolo_preview_ide.py`）

同一套检测逻辑已经接进 IDE 预览，**彩色画面 + 激光十字 + 坐标**可以同时看到：

| 图层 | 内容 |
| --- | --- |
| Pose | 绿色检测框 + 四角连线 + 靶心红十字 |
| 激光 | 黑色十字 + 圆圈 + 黑底白字 `L(cx,cy) A=<面积>` |

实测（2026-10-22，288×216 预览）：**18 fps**，激光稳定在 (275,282)/(277,277)，
Pose 靶心同时在画。加激光图层后预览从 22 fps 掉到 18 fps
（多一路取帧 + `cv_lite` ~4ms + 文字）。

### 实现要点

- **单独一路灰度通道 chn0**：`perf.open_sensor(..., gray=True)` 会在
  `MediaManager.init()`/`run()` **之前**配好 `chn0=GRAYSCALE`；
  彩色 chn1 继续预览、chn2 送 AI。实测三通道采集仍是 **56.7 fps**，
  不影响采集速度。
- 参数在 `yolo_preview_ide.py` 顶部：`LASER_ENABLE` / `LASER_THRESHOLD` /
  `LASER_MIN_PIXELS` / `LASER_AREA_THRESHOLD` / `LASER_MAX_PIXELS` /
  `LASER_MAX_SIDE` / `LASER_LOCK_R` / `LASER_LABEL` / `LASER_LABEL_EVERY`。
- 文字用 `draw_string`：`draw_string_advanced` 在本机固件缺字体
  （`FreeType init failed, font (null)`）。`draw_string` 每次调用会打一条
  `Deprecated function` 日志，所以每 `LASER_LABEL_EVERY`（默认 5）帧才重画一次；
  嫌刷屏设 `LASER_LABEL = 0`，十字和圆圈仍然会画。
- 坐标是 **640×480 原图坐标系**的真实值，和独立脚本打印的一致。

### 已知现象

- 刚启动的**前 ~20 帧**可能先锁到别的小亮点（自动曝光还没稳定、天花板灯边缘更亮），
  之后会自己跳到激光并稳定下来。想避免这段抖动：把 `LASER_LOCK_R` 调小
  （如 20）后再启动，或先让画面里只有激光可见。
