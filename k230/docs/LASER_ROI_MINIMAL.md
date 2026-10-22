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

### 运动测试（ESP32 `s1` 慢速转 pan，±60µs，3 个来回，34s）

| 指标 | ROI 120×120 | ROI 80×80 |
| --- | --- | --- |
| 检出率 | **100%（1935/1935）** | **100%（1934/1934）** |
| 最大连续丢失 | **0 帧** | **0 帧** |
| 相邻位移 mean / p95 | 2.17 / 6.32 px | **1.88 / 4.00 px** |
| 相邻位移 max | 66.4 px | **48.2 px** |
| >10px 占比 | 3.7% | **3.2%** |
| x / y 范围 | 266~281 / 274~344 | 266~281 / **276~326** |

**结论：跟随是连续的（p50 位移 0.5px、p95 4~6px），没有长时间 `laser -`（0 丢失）。**
剩余的大跳变来自"切换到 ROI 内另一个反光亮块"，把 ROI 收到 80×80 后
最大跳变从 66px 降到 48px、>10px 占比 3.7%→3.2%。

### ⚠ 重要发现：转 pan 几乎不会让光斑在画面里移动

实测"pan 指令 vs 光斑 meanX"：

| pan | 光斑 meanX |
| --- | --- |
| 1500 | 278.5 |
| 1580 | 275.2 |
| 1420 | 255.6 |

160µs 转角只让光斑移动约 **23px（≈0.14 px/µs）**；如果激光固定在房间、
只有相机转动，按 FOV 估算应该是 0.4~1 px/µs（约 60~160px）。
说明**激光与相机是共装的**（这正是"激光笔和相机相对位置固定"的物理含义），
所以转云台时**两者一起动、光斑在画面里几乎不动，变的是背景**。

因此：

- 上面的"运动测试"严格说是**背景运动**测试——算法在背景变化时仍 100% 锁定；
  偶发的 y 跳变是"背景里某个反光点瞬间变大、抢过激光"；
- **要真正测激光运动，必须移动激光笔本身**（手拿着移动，或用独立的激光支架），
  转云台是测不出来的。

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
