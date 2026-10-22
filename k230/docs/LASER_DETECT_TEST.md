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
| `THRESHOLD` | `(240, 255)` | **唯一需要手调的**：灰度二值区间 |
| `AREA_THRESHOLD` | `6` | 外接**框**面积下限（OpenMV 的 area 语义） |
| `MIN_PIXELS` | `6` | 真实**亮像素数**下限 |
| `MERGE` / `MERGE_GAP` | `True` / `3` | 相近小块合并 |
| `MAX_PIXELS` / `MAX_SIDE` | `400` / `48` | 光斑**上限**，用来排除灯/墙/手电筒 |
| `MIN_FILL` | `0.45` | 填充率；激光点圆而实，灯带/反光条细长 |
| `KERNEL_SIZE` | `3` | 形态学开运算核，抹掉 1px 级传感器散点 |
| `CORE_X/Y`、`GATE_R` | 中心 / `0` | 位置先验；`GATE_R>0` 才启用门限 |

### 阈值怎么调

| 场景 | 建议 `THRESHOLD[0]` |
| --- | --- |
| 关灯 / 暗背景 | 180 ~ 210 |
| 室内常光（白墙+台灯） | **230 ~ 245**（默认 240） |
| 开灯直射白墙 / 手电筒照靶 | 245 ~ 250 |

调完看日志：命中时打印 `cx/cy/area`，没命中打印 `none`。

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
| 帧率（`DRAW_ON_IDE=False`） | **33.4 fps**（334 帧 / 10s） |
| 帧率（IDE 预览开） | 约 20~25 fps（`Display.show_image` + JPEG 回传占时间） |
| `cv_lite` 单次 | 3 ~ 6 ms |
| 后处理（合并+过滤） | 约 5 ms |
| 采集 | 57 fps（不是瓶颈） |

**每帧存图会掉帧**：`SAVE_HIT` 每帧 `img.save()` 会把 33fps 拖到 **9fps**，
所以默认 `SAVE_HIT=""`，调试时用 `SAVE_HIT_MAX=1` 只存一张。

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

现场实测反而干净：激光点在 (269,283)，逐像素核对灰度矩阵——
中心峰值 **255**，向外一圈 235~254，9×9 邻域均值 234，
而周围背景只有 160~190。台灯/白纸是大片连续亮区（数百~数千像素），
被 `MAX_SIDE=48` / `MAX_PIXELS=400` 直接排除。

要再压误检，按现场加位置门限：激光笔与相机相对固定，把 `CORE_X/CORE_Y`
调到实测的那个点，`GATE_R` 设 0.2~0.35。

---

## 6. 已知限制

- **灰度无法区分「小亮斑」和「小反光」**：真实现场靠「激光位置固定」+
  `MAX_*` 组合解决；换场景（换背景/换灯位）后建议重新确认 `THRESHOLD`。
- 手电筒直射时整幅画面可能饱和，此时 `THRESHOLD[0]` 要提到 248~250，
  或把光斑调小/远离直射。
- 每帧 `img.save()` 会显著掉帧，只在调试时用，且限制张数。
