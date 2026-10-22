# `LaserSpot` 修复与诊断记录（2026-10-22）

对象：`k230/board/laser_openmv.py`（`class LaserSpot`，`VERSION = 2026-10-22.7-diag`）
现象：`laser: 'LaserSpot' object has no attribute 'pending'` 连续刷屏、大量 `laser -`、
坐标大跳、`cand 0 npx 269`。

---

## 1. 根因：两个属性只在 `reset()` 里初始化

用 AST 对 `LaserSpot` 做了"使用 vs 赋值"审计，发现 **`__init__` 缺少 4 个属性**，
其中两个会在运行时抛异常：

| 属性 | 首次被读 | 后果 |
| --- | --- | --- |
| `self.last_step` | `sample()` 第 2 次成功检测 | **第 2 帧必崩**（`0.7*self.last_step`） |
| `self.pending` | 出现 `jdist > jump_gate` 时 | 大跳变那帧崩（就是日志里刷屏那个） |
| `self.pending_t` | 同上 | 同上 |
| `self.rej` | `_find()` 累加 | 仅在 `sample()` 之前调用 `_find` 才崩 |

调用方 `except Exception as e: print("laser:", e)` 把异常吞掉当成"本帧没检测到"，
所以表现为 **检测率骤降 + 日志刷屏 + 坐标乱跳**（每帧结果取决于有没有走到崩的分支）。

**修复**：在 `__init__` 里补齐
`self.last_step = 0.0`、`self.pending = None`、`self.pending_t = 0`、`self.rej = {}`，
并加上 `self._miss_counted = False`、`self.prior_degraded = False`、
`self.core_fallback_used = 0`、`self.diag_blobs = []`。

> 修完再跑同一套 AST 审计：除方法名外无遗漏。

---

## 2. 诊断模式 `DIAG_MODE`

模块顶部：

```python
DIAG_MODE = False     # 改成 True 开诊断
```

也可以从调用方覆盖：`LaserSpot({"diag_mode": 1})`（预览里是 `LASER_CFG`）。

开启后：

- `find_blobs` 的 `pixels_threshold/area_threshold` 从 10 降到 4，先看原始 blob；
- **关闭**先验门控与全部候选过滤（面积/边长/长宽比/fill/peak/core/contrast）；
- 每帧打印一行头 + 每个 blob 一行：

```
[DIAG] lab=(5,100,25,127,-60,80) min_px=20 max_px=320 max_side=40 aspect=3.0 core=220 prior=(300.0,150.0) gate=280 raw=14
  [DIAG] blob cx=294.0 cy=316.0 px=35 area=64 w=8 h=8 fill=0.55 asp=1.00 peak=237 core=232 con=110 warm=104 -> OK
  [DIAG] blob cx=216.0 cy=277.0 px=2665 area=3268 w=43 h=76 fill=0.82 asp=1.00 peak=247 core=240 con=90 warm=144 -> AREA_HIGH(a>max_px)
  [DIAG] blob cx=463.0 cy=422.0 px=240 area=512 w=32 h=16 fill=0.47 asp=1.00 peak=137 core=136 con=69 warm=176 -> PRIOR_GATE(317px>280)
```

`-> ` 后面就是"被哪一条规则拒掉"，`OK` 表示能通过。

> 旧实现里诊断分支打印完直接 `continue`，**跳过了 best 选择**，所以会出现
> `cand 2` 却仍然 `laser -`。现在诊断模式也会选 best（过滤全关，等权打分）。

### 实测（本机现场）

`raw = 11~20` 个原始 blob。按用户的判据（5~10 个 = 有反光干扰）属于偏多，实际是：

| 位置 | 尺寸/像素 | 判定 | 是什么 |
| --- | --- | --- | --- |
| (216,277) | 43×76, 2665px | AREA_HIGH | 红色布/纸箱 |
| (280,281) | 52×64, 1307px | AREA_HIGH | **激光 + 红布被 merge 成一块** |
| (195,251) | 5×21, 56px | 能过阈值但长宽比 4.2 | 红色线缆 |
| (302,297) | 5×11, 51px | 长宽比 2.2 / warm 40 | 白色反光（瓶身/靶纸高光） |
| **(279,280)** | **8×11, 68px** | 通过 | **激光本体** |

---

## 3. 候选过滤逻辑审查

### 3.1 `cand` 与 `npx`

- `_find()` 里 `cands += 1` 只在 blob **通过全部门限**时执行，最后一个门限之后才算；
  `self.candidates = cand` 每帧重置。
- `self.last_px = a` 记录的是**上一个被接受**的 blob 的像素数。
- 所以 `cand 0 npx 269` = **上一帧成功接受了 269px 的光斑，这一帧没有任何候选通过**。
  两者统计口径不同帧，并不矛盾；它说明的是"这一帧全军覆没"。
  典型原因：激光和红色物体 merge 成 >`max_px` 的大块（AREA_HIGH），或先验门控把真光斑挡住。

### 3.2 评分公式

```python
sc = 2.0*(core/255) + 1.4*(peak/255) + 1.3*(min(contrast,150)/150)
     + 1.2*fill + 0.7/aspect + 0.8*(min(max(warm,0),200)/200)
if ref is not None:                       # 有先验/参考时
    sc += 0.9*max(0, 1 - dist/120)        # 离参考近加分
    sc -= 0.4*min(dist,220)/220           # 离参考远扣分
if area_ema: sc -= 0.6*abs(a-area_ema)/area_ema   # 面积突变扣分
```

| 项 | 物理含义 | 现场量级 |
| --- | --- | --- |
| `core` | bbox 内 `max(min(R,G,B))`，白核/饱和程度 | 激光 248；暗反光 148 |
| `peak` | bbox 内最大亮度 | 激光 250；暗反光 160 |
| `contrast` | `peak - 外圈 5px 均值`，点光源的局部对比 | 激光 90~110；反光有时为负 |
| `fill` | `pixels/(w*h)`，实心程度 | 激光 0.7~0.8；细长物 0.5 |
| `aspect` | `max(w,h)/min(w,h)`，圆度 | 激光 1.4；反光条 2.2；线缆 4.2 |
| `warm` | `max(R - max(G,B))`，红晕强度 | 激光 96~120；白色反光 40~56 |

门限（`fg_startup`）：`fill≥0.50, aspect≤2.8, peak≥200, core≥210, contrast≥60`。

### 3.3 先验复核为什么"拒掉真光斑"

原逻辑只有一处硬门控：`_find(..., ref=prior, gate=start_gate=280)`，
`dist > gate` 时 `rej["gate"] += 1` 并跳过。**偏离 195px < 280 并不会被硬拒**，
真正让真光斑落选的是**评分里的先验项**：

- 离错误先验 195px 的真光斑：`+0.9*max(0,1-195/120)=0`，`-0.4*195/220 = -0.35`
- 任何恰好落在先验附近的杂散 blob：`+0.9*(1-0) = +0.9`

一进一出 **1.25 分**，足以压过"更亮更圆"的差异，于是**每个静止/低分杂点都能击败真光斑**，
而且先验一旦过期就会一直错下去（原来没有任何"先验失效"判断）。

**修复**：两处自动降级
1. 冷启动门控内没找到 → 立刻无门控全图重找，找到就 `prior = None` 并打印提示；
2. 找到的光斑离先验 > `prior_grace`(120px) → 同样直接弃用先验（不再让过期先验参与评分）。

---

## 4. 本轮做的其它改动（都只影响激光，不动 YOLO/舵机）

1. **关闭多帧跳变确认**：`CFG["jump_confirm"] = 0`，`pending` 那套整块跳过
   （它会连续吃 2 帧代价，是大量 `laser -` 的来源之一）。需要时可设 1 恢复。
2. **大块内取亮核** `_core_fallback()`：激光打在红色物体上时会被 `merge` 吞成
   1000~2700px 的大块（AREA_HIGH 整块拒掉）。失败时对最大的 2 个"过亮大块"
   （`peak ≥ 225`）在其 bbox 内用近白阈值再扫一次，取出 40~90px 的亮核。
   只在失败帧跑，一次 11~14ms。
3. **亮核自身的门限**：`core_aspect_max=1.8`（圆度）、`core_warm_min=60`（红晕）、
   `core_core_min=200` / `core_peak_min_sub=200`（亮度）。
   实测这三项把"白色反光(2.2/warm 40)"和"暗反光(148/160)"全部排除。
4. **锁定后不再放宽门限**：`loose_after_lock = 0`（原来切 `fg_track` 会把
   `peak` 从 200 放到 155，暗反光立刻混进来抢位置）。难检情况交给亮核回退。
5. 日志降噪：亮核回退只在前 3 次打印（诊断模式除外）。

---

## 5. 实测结果（本机现场，激光静止）

| 指标 | 修复前 | 修复后 |
| --- | --- | --- |
| 是否能跑 | 第 2 帧就 `AttributeError` | 无任何 `has no attribute` |
| 30s 检出率 | 13~17fps 下大量 `laser -` | **100%（312/312）** |
| 坐标 | (35,257)→(111,439)→(308,306) | **(278.8, 280.3)** |
| 标准差 | 几十~上百 px | **cx 2.80 / cy 4.20 px** |
| 最大连续丢失 | — | **0** |
| 帧率 | 13~17 fps | 10.2 fps（本场景每帧都走亮核回退，+12ms） |

**帧率说明**：当前现场激光打在红色布上，正常路径必然失败、每帧都触发回退，
所以掉到 10 fps。激光打在**中性背景**（白墙/靶纸）时走正常路径，预期回到 15~18 fps；
想要更高就把 `core_fallback=0`（只保留旧路径）或 `core_topk=1`。

**运动测试**（ESP32 `s1` 慢速转 pan，18s）：命中 **90.3%**，最大连续丢失 7 帧，
但出现 21 次 >80px 跳变——轨迹里的 `(199,279)`、`(276,417)` 就是另外两个亮点。
说明运动时评分仍会在竞争亮点间切换；这一轮按要求**没有引入多帧确认**，
所以单帧层面已到极限。

---

## 6. 建议的验收步骤

```bash
cd k230
python3 -m k230tool push board/laser_openmv.py --dest /sdcard/laser_openmv.py

# 1) 先开诊断，确认画面里有几个亮点
#    改 /sdcard/laser_openmv.py 顶部 DIAG_MODE = True（或调用时传 {"diag_mode":1}）
#    激光静止在画面中央，看 raw 数量和每个 blob 的过滤原因

# 2) 关诊断，静止 30s 看检出率与 std（<5px 为合格）
python3 - <<'PY'
from k230tool.link import K230Link
link = K230Link(); link.open()
out, err, ok = link.exec_and_collect(
    "exec(open('/sdcard/accept_static.py').read())", timeout=150)
print(out[-600:]); link.close()
PY

# 3) 慢速移动激光（或用 ESP32 `s1` 慢转 pan），看轨迹是否连续
```

判定：`RESULT ... rate=100.0%` 且 `std` 两项都 < 5 即通过。若 std 仍大，
说明还有第二个"同样亮"的点，办法是把激光指到干净背景，或调 `core_aspect_max`
/ `core_warm_min` 把干扰点排掉。
