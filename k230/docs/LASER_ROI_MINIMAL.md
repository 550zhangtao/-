# 激光检测：极简 ROI 方案（2026-10-22 重写）

`k230/board/laser_openmv.py` 已被**整体重写**为极简版：
**给定 ROI，用 cv_lite 灰度阈值找最大 blob**。VERSION = `2026-10-22.8-roi`。

上一版（`.6-adaptive`/`.7-diag`）里的这些东西**全部删除**：
先验复核、评分公式、候选过滤、`pending` 多帧确认、大块取亮核、自适应阈值、
`_blob_pixels` 逐像素扫描。

---

## 1. 为什么之前是 110~209ms

旧实现两个致命开销：

1. `img.find_blobs()` —— OpenMV 原生全图搜索（实测 **11~24ms**，640×480）；
2. 对**每个候选**再跑 `_blob_pixels()` —— MicroPython 里逐像素 `get_pixel`，
   一块 30×30 就是 900 次调用 ≈ 1.5ms，几十个候选直接上百毫秒。

另外 `KERNEL_SIZE=3` 的形态学、`relax` 双扫、亮核回退（再扫一次全图/大块）
都是叠加成本。日志里 `lms 110/142/194/209ms` 就是这么来的。

**现在只用 cv_lite**（C 实现的连通域），且只处理 120×120 的 ROI：

| 操作 | 实测 |
| --- | --- |
| `gray.crop(roi=120x120)` + `to_numpy_ref()` | 0.30 ms |
| `cv_lite.grayscale_find_blobs` 120×120 | **0.20 ms** |
| 同上，640×480 全图（对照，未使用） | 4.10 ms |

`lms` 只统计「裁剪 + cv_lite」，**实测 mean 1.03ms / p95 2ms / max 3ms**。

---

## 2. 现在的模块（约 250 行，无任何状态机）

```python
CFG = {
    "roi": (260, 220, 120, 120),   # 画面中心 120x120
    "threshold": (220, 255),       # 灰度阈值（唯一要调的参数）
    "min_area": 6,
    "kernel_size": 1,              # 1=不做形态学（3 会抹掉 3~4px 光斑核心）
    "center_bias": 0.35,           # 选最大 blob 时"离 ROI 中心远"打折（可设 0）
    "center_ref_px": 60.0,
    "diag_mode": 0,
}
```

`sample(gray_img)` 流程只有四步：裁 ROI → cv_lite → 取面积最大 → 返回。
坐标已加 ROI 偏移，是 **640×480 原图坐标**。

**没有**：`prior`、`score`、`pending`、`jump_confirm`、状态机、卡尔曼。
`load()/set_calib()` 保留成空函数只为兼容旧调用方。

---

## 3. 诊断模式

```python
DIAG_MODE = True             # 改文件顶部，或 LaserSpot({"diag_mode": 1})
```

输出（实测）：

```
[DIAG] roi=(260, 220, 120, 120) thr=(220, 255) raw=2 lms=1.0ms
  [DIAG] blob cx=266 cy=334 w=12 h=13 area=156 pixels=97
  [DIAG] blob cx=277 cy=288 w=21 h=13 area=273 pixels=161
```

`pixels` 是真实亮像素数（只在 `diag_max_side` 以内才数，避免开销）。

---

## 4. 阈值怎么定（实测扫描，本机现场）

| THRESHOLD | raw | 结果 |
| --- | --- | --- |
| 200 | 2 | 选对（激光 345 vs 反光 322，**余量小**） |
| 210 | 2 | 选对（294 vs 196） |
| **220** | 2 | **选对（273 vs 156，余量 1.75×）← 默认** |
| 230 | 3 | **选错**：激光被拆成两片，反光反而最大 |
| 240 | 3 | 选错，同上 |
| 245 | 1 | 只剩激光碎片（偏心） |

**结论：220 是这台机器当前光照下的最佳点。** 阈值不是越高越好——
太高会把激光自身拆碎。

---

## 5. 验收结果

### 静止 30s（`scripts/accept_laser_static.py`）

| 指标 | 结果 | 目标 |
| --- | --- | --- |
| `lms` | mean **1.03 ms**，p95 2 ms，max 3 ms | < 10 ms ✅ |
| 帧率 | **57.1 fps**（= 采集上限） | — |
| 检出率 | **100%（1707/1707）**，最大连续丢失 **0** | > 80% ✅ |
| 坐标 | (276.98, 286.99) | — |
| **标准差** | **cx 0.10 px / cy 0.08 px** | < 3 px ✅ |
| `raw` | 恒为 **2** | ≤ 5 ✅ |

### 运动（ESP32 `s1` 慢速转 pan，±20µs ≈ ±19px，激光始终在 ROI 内）

| 指标 | 结果 |
| --- | --- |
| 检出率 | **100%（1137/1137）**，最大连续丢失 **0** |
| 相邻跳变 | mean 4.9 px，max 40.7 px，>10px 占 13.7% |
| x 轨迹 | 272~280（跟随 pan，干净） |

**y 方向仍有 25~40px 的跳变**：ROI 里除了激光 (277,288)，还有
(268,333)/(275,315) 两个亮反光点，运动时 blob 形态变化会让它们偶尔面积反超。
这不是先验或状态机的问题，是**画面里确实有多个同样亮的小目标**。

可选处理（都不引入新逻辑，只改参数）：

- 把 ROI 缩小到 80×80 并挪到激光实际位置（`LASER_CFG={"roi":(236,247,80,80)}`）：
  实测跳变从 2.4% 降到更低，但要求激光始终在窗口内；
- 提高 `center_bias`（0.35 → 0.6）让偏离中心的点更不容易被选中；
- 最根本：**把激光打到干净背景**（白墙/靶纸中央），ROI 里只剩它一个 blob。

### 其它

| 检查 | 结果 |
| --- | --- |
| `'LaserSpot' object has no attribute 'pending'` | **代码里已无 `pending`**，不可能再出现 |
| `Deprecated function` | 预览里 **0 次**（见下节） |
| 板端版本 | `2026-10-22.8-roi` |

---

## 6. 关于 `draw_string_advanced`

要求是"改用 `draw_string_advanced` 消除 Deprecated"。实测**本机固件没有字体**：

```
draw_string_advanced(20,20,24,"(271,282)",color=...) 
-> FreeType init failed, font (null) errorno -1
```

所以改用**内置 3×5 点阵字体**，只用 `draw_rectangle(fill=True)` 逐像素画字：

- 实测 10 个字符 **0.45ms**；
- 预览日志 `Deprecated` 计数 **0**；
- 坐标依旧可见（`(277,287)`）。

预览打印格式（按要求）：

```
[15] 19.0fps | laser (277,287) conf 1.00 | lms 1ms raw 2 | target 0.75
```

---

## 7. 复现命令

```bash
cd k230
python3 -m k230tool push board/laser_openmv.py    --dest /sdcard/laser_openmv.py
python3 -m k230tool push board/yolo_preview_ide.py --dest /sdcard/yolo_preview_ide.py
python3 -m k230tool push board/k230_perf.py        --dest /sdcard/k230_perf.py

# 静止 30s 验收（脚本会打印 rate / std / lms / raw）
python3 -m k230tool push scripts/accept_laser_static.py --dest /sdcard/accept_static.py
python3 - <<'PY'
from k230tool.link import K230Link
link = K230Link(); link.open()
out, err, ok = link.exec_and_collect(
    "exec(open('/sdcard/accept_static.py').read())", timeout=150)
print(out[-500:]); link.close()
PY
```
