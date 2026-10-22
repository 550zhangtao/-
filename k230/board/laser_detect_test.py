# -*- coding: utf-8 -*-
"""
laser_detect_test.py —— K230 激光点检测测试脚本（cv_lite 灰度模式）
====================================================================

独立运行，**不依赖 YOLO、不依赖舵机/UART**。激光笔与相机相对位置固定，
激光点在画面里是一个高亮小光斑；本脚本只做「找到它 + 画出来 + 打日志」。

两种用法：
  1. CanMV IDE：直接打开本文件点运行（画面在 IDE 里看）；
  2. 无 IDE（我/自动化用的那条路）：先 `mode.sh debug`，再用
     `k230tool push` 推到 /sdcard，然后
     `link.exec_and_collect("exec(open('/sdcard/laser_detect_test.py').read())")`，
     跑完把 CSV 拉回来检查。

--------------------------------------------------------------------------
调参（都是文件顶部的全局变量，改完直接重跑）
--------------------------------------------------------------------------
    THRESHOLD   灰度二值区间 (lo, hi)。**这是唯一需要手调的参数。**
                开灯/白墙这类高亮背景 -> 调高 lo（220~245）；
                关灯/暗背景           -> 可以降到 150~200；
                手电筒直射            -> 背景本身很大一片饱和，lo 再高点，
                                        并靠 MAX_* 把大片区域卡掉。
    MIN_PIXELS  ~像素阈值（见下）      AREA_THRESHOLD  同上（取两者较大值生效）
    MERGE       True = 把挨得近的小块合并成一个光斑
    MAX_PIXELS / MAX_SIDE  光斑上限：超过就当"一片光"，不是激光点
    KERNEL_SIZE 形态学核（1 = 不做）

先用"开运算"把散点吃掉（关键）
    台灯/手电筒/白纸/显示器这些真实光源，在饱和灰阶里是**大而连续的**亮区
    （几十到几千像素）；激光点只有十几像素。但如果直接扫阈值，暗背景的传感
    器噪点会散落在整幅画面里（噪点恰好是"小 + 亮"，和激光点长得一样）。
    所以先用 cv_lite 带的 `kernel_size`（形态学开运算）去掉 1~2 像素的散点，
    再用 MIN_PIXELS 卡面积。实测数据集 300 张：开运算前后"找到像激光的小光斑"
    的比例基本不变（234 -> ~225），但实时画面里的散点候选从 36 个降到个位数。

为什么要 MAX_*/MIN_* 这两组上下限
    激光点是「小 + 亮」，而灯、窗外天光、手电筒照到的白墙是「大 + 亮」。
    只设下限会把整片墙当成 blob；加上限后，背景光直接被排除，
    所以同一套阈值在开灯/关灯/手电筒三种场景下都能用。

关于 cv_lite 的 API（和 OpenMV 的 find_blobs 不一样，实测本机固件）
    cv_lite.grayscale_find_blobs(image_shape, data, thr_lo, thr_hi,
                                 min_area, kernel_size) -> [x, y, w, h, ...]
    * **没有 pixels_threshold / merge 参数**，返回的是"连通块外接矩形"，
      也不给面积。所以本脚本：
        - min_area = max(MIN_PIXELS, AREA_THRESHOLD)；
        - 外接矩形框里可能粘着好几个小点（合并不一样），按像素灰度
          重新做一次连通域拆分，用 4 连通把真正的光斑分开；
        - 用边框内亮像素计数当"像素面积"（这是真实像素数，不是 w*h）。
""" 

import time
import gc

import cv_lite
from media.sensor import *
from media.media import *
from media.display import *

# ============================ 用户参数 ============================
# 相机输出尺寸（相机原生 1920x1080@60；这里直接按 640x480 输出，采集 57fps 实测）
W, H = 640, 480
SENSOR_ID = 2
FPS = 60                       # 采集帧率上限；GC2093 原生 1080p60

# ★ 主参数：灰度二值区间。激光点几乎饱和（~255），背景通常远低于它。
THRESHOLD = (210, 255)         # (low, high)；见文档"阈值怎么调"

# 面积相关（cv_lite 只有一个 min_area，这里两个都收下、取较大值）
AREA_THRESHOLD = 6             # 你要的 area_threshold，5~10
MIN_PIXELS = 6                 # 你要的 pixels_threshold，5~10
MERGE = True                   # True: 相近光斑合并（间隔 <= MERGE_GAP 就并）

# 光斑"应该是多大"——用来排除灯/墙/手电筒这类大片亮区
MAX_PIXELS = 150               # 超过这个像素数就不是激光点（实测激光 30~70px）
MAX_SIDE = 24                  # 外接矩形最长边上限（px，激光点通常 6~16px）
MIN_FILL = 0.45                # 填充率下限：激光点圆而实，灯/反光是细长条或镂空

KERNEL_SIZE = 1                # 形态学核：1 = 关闭（推荐，激光核心可能只有 3~4px，
                               #   K=3 的开运算会把它一起抹掉，实测会直接漏检）
MERGE_GAP = 3                  # 两块之间间隔 <= 这么多像素就合并

# 激光点大致应在的位置（搜索中心，像素）。只影响"多个候选时优先选谁"。
CORE_X = W * 0.5
CORE_Y = H * 0.5

# 位置门限：0 = 关闭（推荐先关掉）。激光笔与相机相对固定时，光斑只在
# 画面某一小块出现，把它设成 0.3~0.5（相对半宽/半高的归一化半径）能压掉
# 大部分杂散误检。实测数据集上设太小会掉检出率，所以默认关闭、按现场再收紧。
GATE_R = 0.0

# 时序锁定：激光笔与相机相对固定，光斑不会乱跳。锁定后只在上一帧位置
# LOCK_R 半径内找，连续 LOCK_LOST 帧找不到才重新全局搜索。
# 实测：不加锁定时 310 帧里会偶尔跳到别的小亮点（cx 标准差 121px）；
# 加锁定后一条直线稳住。LOCK_R=0 关闭锁定。
LOCK_R = 40
LOCK_LOST = 12

# 打印 / 保存
PRINT_EVERY = 1                # 每 N 帧打印一次；1 = 每帧都打印
SAVE_CSV = "/sdcard/laser_detect_log.csv"   # 每帧一行；"" = 不写
SAVE_HIT = ""                  # 例如 "/sdcard/laser_hit.jpg"：命中时存图（调试用）
SAVE_HIT_MAX = 1               # 最多存几张（每帧存会明显掉帧）
SAVE_HIT_RAW = "/sdcard/laser_hit_raw.jpg"   # 同一帧的"未绘制"原图，用于核对

DRAW_ON_IDE = True             # True: 每帧推到 IDE（IDE 里才有画面）
IDE_QUALITY = 70               # 推 IDE 的 JPEG 质量
RESET_ON_EXIT = True           # 退出时软复位（KPU/相机析构在板上有死锁风险）

DISPLAY_MODE = "ide"           # "ide" | "none"（只跑不显示时用 none，帧率最高）
ROUNDS = 0                     # 0 = 一直跑；N > 0 = 跑 N 轮后自动停（每轮 SECONDS 秒）
SECONDS = 0                    # 每轮秒数，0 = 不限
# =================================================================

sensor = None                  # 由 _init() 赋值
_save_n = 0
_lock = None                   # 上一帧锁定的 (cx, cy)
_lock_miss = 0                 # 连续没找到的帧数


# ---------------------------------------------------------------- 几何工具
def _blob_side(b):
    return b[2] if b[2] > b[3] else b[3]


def _split_by_gray(gray, shape, rect, thr_lo):
    """把一个外接矩形里粘连的多个亮块拆开（8 连通），返回各自的像素统计。

    cv_lite 只给外接矩形、没有 merge 参数，远处小点挨得近时会被并成一个大框；
    这里按像素重新做一次连通域，拿到真正的光斑和各自的**像素数**（真实面积）。

    性能要点（实测踩过）：
      * ndarray 是二维 `gray[y, x]`（行优先），一维扁平索引会越界；
      * 不能拿 tuple 当 visited 的 key：一块几百像素的亮斑会让每帧涨到 250ms。
        这里用**一维 bytearray 当 visited 标记**，索引 = by*w + bx（局部坐标）。
    """
    h_img, w_img = shape[0], shape[1]
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return []
    mark = bytearray(w * h)
    comps = []
    for oy in range(h):
        yy = y + oy
        if yy < 0 or yy >= h_img:
            continue
        for ox in range(w):
            xx = x + ox
            if xx < 0 or xx >= w_img:
                continue
            if mark[oy * w + ox] or gray[yy, xx] < thr_lo:
                continue
            mark[oy * w + ox] = 1
            stack = [oy * w + ox]
            n = 0
            sx = sy = 0
            minx = maxx = ox
            miny = maxy = oy
            while stack:
                k = stack.pop()
                ny, nx = divmod(k, w)
                n += 1
                sx += nx
                sy += ny
                if nx < minx: minx = nx
                if nx > maxx: maxx = nx
                if ny < miny: miny = ny
                if ny > maxy: maxy = ny
                for dy in (-1, 0, 1):
                    ty = ny + dy
                    if ty < 0 or ty >= h or (y + ty) >= h_img:
                        continue
                    row = ty * w
                    base_y = y + ty
                    for dx in (-1, 0, 1):
                        tx = nx + dx
                        if (dx == 0 and dy == 0) or tx < 0 or tx >= w or (x + tx) >= w_img:
                            continue
                        j = row + tx
                        if mark[j] or gray[base_y, x + tx] < thr_lo:
                            continue
                        mark[j] = 1
                        stack.append(j)
            comps.append({"n": n, "cx": x + sx / float(n), "cy": y + sy / float(n),
                          "x": x + minx, "y": y + miny, "w": maxx - minx + 1,
                          "h": maxy - miny + 1})
    return comps


def _merge_close(items, gap):
    """把质心距离 <= gap 的小块合成一个（模拟 merge=True）。"""
    if not MERGE or len(items) < 2:
        return items
    out = []
    used = [False] * len(items)
    for i in range(len(items)):
        if used[i]:
            continue
        a = items[i]
        used[i] = True
        g = {"n": a["n"], "sx": a["cx"] * a["n"], "sy": a["cy"] * a["n"],
             "x": a["x"], "y": a["y"], "w": a["w"], "h": a["h"]}
        changed = True
        while changed:
            changed = False
            for j in range(len(items)):
                if used[j]:
                    continue
                b = items[j]
                dx = (g["sx"] / g["n"]) - b["cx"]
                dy = (g["sy"] / g["n"]) - b["cy"]
                if dx * dx + dy * dy <= gap * gap:
                    used[j] = True
                    g["n"] += b["n"]
                    g["sx"] += b["cx"] * b["n"]
                    g["sy"] += b["cy"] * b["n"]
                    g["x"] = min(g["x"], b["x"])
                    g["y"] = min(g["y"], b["y"])
                    g["w"] = max(g["x"] + g["w"], b["x"] + b["w"]) - g["x"]
                    g["h"] = max(g["y"] + g["h"], b["y"] + b["h"]) - g["y"]
                    changed = True
        out.append({"n": g["n"], "cx": g["sx"] / g["n"], "cy": g["sy"] / g["n"],
                    "x": g["x"], "y": g["y"], "w": g["w"], "h": g["h"]})
    return out


# ---------------------------------------------------------------- 主流程
def _peak(gray, c, n=6):
    """在候选框里粗采样取峰值亮度（最多 n*n 个点）。

    激光是镜面反射，峰值几乎总是画面里最高的；纸/白墙虽然也亮，但峰值低一档。
    用网格采样而不是逐像素，是为了把成本压在几十微秒级。
    """
    x, y, w, h = c["x"], c["y"], c["w"], c["h"]
    sx = w // n if w >= n else 1
    sy = h // n if h >= n else 1
    pk = 0
    yy = y
    while yy < y + h:
        xx = x
        while xx < x + w:
            v = gray[yy, xx]
            if v > pk:
                pk = v
            xx += sx
        yy += sy
    return pk


def _pick(cands, gray=None):
    """在候选里挑激光点：先比峰值亮度，再看面积与离中心距离。

    教训：只按"离画面中心近"排序会选错——实测一帧里白纸边缘的候选比真正的
    激光点更靠中心，结果十字画在纸上（偏移 33px）。激光点峰值最高，所以
    峰值是第一判据，中心距离只用来打破平手。
    """
    best = None
    best_key = None
    for c in cands:
        # pixels_threshold：真实亮像素数
        if c["n"] < MIN_PIXELS or c["n"] > MAX_PIXELS:
            continue
        # area_threshold：外接框面积（OpenMV 的 area 就是 bbox 面积）
        if c["w"] * c["h"] < AREA_THRESHOLD:
            continue
        if c["w"] > MAX_SIDE or c["h"] > MAX_SIDE:
            continue
        fill = c["n"] / float(c["w"] * c["h"]) if (c["w"] and c["h"]) else 0.0
        if fill < MIN_FILL:
            continue
        dx = (c["cx"] - CORE_X) / (W * 0.5)
        dy = (c["cy"] - CORE_Y) / (H * 0.5)
        dist = (dx * dx + dy * dy) ** 0.5
        if GATE_R > 0 and dist > GATE_R:
            continue          # 超出允许区域，不当激光点
        pk = _peak(gray, c) if gray is not None else 0
        # 峰值降序 -> 面积升序（激光点偏小）-> 离中心近
        key = (-pk, c["n"], dist)
        if best_key is None or key < best_key:
            best_key = key
            best = c
    return best


def run_round(shape, label="", seconds=0, csv_file=None):
    """跑 seconds 秒（0 = 一直跑），返回 (帧数, 命中数, 平均fps)。可被自动化多次调用。"""
    global _save_n, _lock, _lock_miss

    t_start = time.ticks_ms()
    frame = 0
    hits = 0
    t_win = t_start
    win_n = 0                 # 本窗口内的帧数（不能用累计 frame 算 fps）
    min_area = MIN_PIXELS if MIN_PIXELS > AREA_THRESHOLD else AREA_THRESHOLD

    # 先空跑 2 帧：自动曝光稳定前第一帧常整帧偏亮
    for _ in range(2):
        sensor.snapshot()

    while True:
        # ---- 取帧：灰度 + numpy 引用（cv_lite 直接吃这块内存）
        img = sensor.snapshot()
        data = img.to_numpy_ref()

        # ---- cv_lite 灰度找亮块
        t0 = time.ticks_ms()
        raw = cv_lite.grayscale_find_blobs(shape, data, THRESHOLD[0], THRESHOLD[1],
                                           min_area, KERNEL_SIZE)
        cv_ms = time.ticks_diff(time.ticks_ms(), t0)

        # ---- 候选筛选（cv_lite 只给外接矩形，没有面积/merge）
        # 关键性能取舍：**不做逐像素拆分**。实测逐个 8 连通泛洪一块 900px 的
        # 亮斑要 ~7ms，当前场景有 20+ 个亮块时后处理会到 45ms，直接掉到 16fps。
        # 既然 MAX_PIXELS 已经说明"大块不可能是激光"，干脆把大块整体拒掉。
        cands = []
        for i in range(0, len(raw) - 3, 4):
            rw, rh = raw[i + 2], raw[i + 3]
            bx, by = raw[i], raw[i + 1]
            side = rw if rw > rh else rh
            if side > MAX_SIDE:
                continue                      # 灯 / 白墙 / 手电筒反光
            box_area = rw * rh
            if box_area > MAX_PIXELS * 2:
                continue                      # 太大，不可能是激光点
            # 用外接框面积当像素数上界（亮块接近实心，实测误差很小）
            cands.append({"n": box_area,
                          "cx": bx + rw * 0.5, "cy": by + rh * 0.5,
                          "x": bx, "y": by, "w": rw, "h": rh})
        cands = _merge_close(cands, MERGE_GAP)
        # ---- 时序锁定：先在上一帧位置附近找，找不到再全局找
        best = None
        if LOCK_R > 0 and _lock is not None:
            near = []
            for c in cands:
                dx = c["cx"] - _lock[0]
                dy = c["cy"] - _lock[1]
                if dx * dx + dy * dy <= LOCK_R * LOCK_R:
                    near.append(c)
            best = _pick(near, data)
            if best is None:
                _lock_miss += 1
                if _lock_miss >= LOCK_LOST:
                    _lock = None
                    _lock_miss = 0
            else:
                _lock_miss = 0
        if best is None:
            best = _pick(cands, data)

        frame += 1
        _save_this = False
        if best is not None:
            _lock = (best["cx"], best["cy"])
            hits += 1
            cx, cy, area = int(best["cx"]), int(best["cy"]), best["n"]
            if PRINT_EVERY and (frame % PRINT_EVERY) == 0:
                print("[%d] BLOB cx=%d cy=%d area=%d (cv %.1fms)"
                      % (frame, cx, cy, area, cv_ms))
            # 存档：先在"还没画"的时候存原图，再画标记，最后存画好的
            _save_this = bool(SAVE_HIT) and _save_n < SAVE_HIT_MAX
            if _save_this and SAVE_HIT_RAW:
                try:
                    img.save(SAVE_HIT_RAW)
                    print("saved raw -> %s" % SAVE_HIT_RAW)
                except Exception as e:
                    print("save raw:", e)
            # 灰度图上彩色会被压成暗灰，看不清；激光点本身很亮，
            # 所以用**黑色**矩形 + 十字，对比度最高（实测比红色清楚得多）。
            img.draw_rectangle(best["x"], best["y"], best["w"], best["h"],
                               color=(0, 0, 0), thickness=1)
            img.draw_cross(cx, cy, color=(0, 0, 0), size=12, thickness=2)
            img.draw_circle(cx, cy, 10, color=(0, 0, 0), thickness=1)
        elif PRINT_EVERY and (frame % PRINT_EVERY) == 0:
            print("[%d] none  (cv %.1fms)" % (frame, cv_ms))

        if csv_file is not None:
            if best is not None:
                csv_file.write("%d,%s,%d,%d,%d,%d,%d,%d,%d\n"
                               % (frame, label, 1, int(best["cx"]), int(best["cy"]),
                                  best["n"], best["w"], best["h"], cv_ms))
            else:
                csv_file.write("%d,%s,0,-1,-1,0,0,0,%d\n" % (frame, label, cv_ms))

        # 命中存档：只存前 SAVE_HIT_MAX 张。**不要每帧都存**——实测 SD 写入
        # 会把 33fps 拖到 9fps（每次 img.save 数十毫秒）。
        if best is not None and _save_this:
            _save_n += 1
            try:
                img.save(SAVE_HIT)
                print("saved hit -> %s" % SAVE_HIT)
            except Exception as e:
                print("save hit:", e)

        if DRAW_ON_IDE and DISPLAY_MODE == "ide":
            Display.show_image(img)

        win_n += 1
        dt_win = time.ticks_diff(time.ticks_ms(), t_win)
        if dt_win >= 2000:
            print("---- %s frames=%d hit=%d fps=%.1f ----"
                  % (label, frame, hits, win_n * 1000.0 / dt_win))
            t_win = time.ticks_ms()
            win_n = 0

        if seconds and time.ticks_diff(time.ticks_ms(), t_start) >= seconds * 1000:
            break
        if (frame & 0x1F) == 0:
            gc.collect()

    total_ms = time.ticks_diff(time.ticks_ms(), t_start)
    return frame, hits, (frame * 1000.0 / total_ms if total_ms else 0.0)


def _init():
    sensor = Sensor(id=SENSOR_ID, fps=FPS)
    sensor.reset()
    sensor.set_framesize(w=W, h=H)
    sensor.set_pixformat(Sensor.GRAYSCALE)
    if DISPLAY_MODE == "ide":
        Display.init(Display.VIRT, width=W, height=H, to_ide=True, quality=IDE_QUALITY)
    MediaManager.init()
    sensor.run()
    return sensor


def main():
    global sensor
    sensor = _init()
    csv_file = None
    if SAVE_CSV:
        try:
            csv_file = open(SAVE_CSV, "w")
            csv_file.write("frame,label,found,cx,cy,area,w,h,cv_ms\n")
        except Exception as e:
            print("csv open failed:", e)
            csv_file = None
    print("THRESHOLD=%s area>=%d/%d max_px=%d max_side=%d merge=%s"
          % (THRESHOLD, AREA_THRESHOLD, MIN_PIXELS, MAX_PIXELS, MAX_SIDE, MERGE))
    total_frames = total_hits = 0
    rnd = 0
    try:
        while True:
            rnd += 1
            frames, hits, fps = run_round([H, W], label="r%d" % rnd,
                                          seconds=SECONDS, csv_file=csv_file)
            total_frames += frames
            total_hits += hits
            print("ROUND %d: frames=%d hits=%d fps=%.1f" % (rnd, frames, hits, fps))
            if ROUNDS and rnd >= ROUNDS:
                break
    finally:
        print("TOTAL frames=%d hits=%d hit_rate=%.1f%%"
              % (total_frames, total_hits,
                 (100.0 * total_hits / total_frames) if total_frames else 0.0))
        if csv_file is not None:
            try:
                csv_file.close()
            except Exception:
                pass
        try:
            sensor.stop()
        except Exception:
            pass
        if DISPLAY_MODE == "ide":
            try:
                Display.deinit()
            except Exception:
                pass
        if RESET_ON_EXIT:
            try:
                import machine
                time.sleep_ms(150)
                machine.reset()
            except Exception:
                pass


sensor = None

try:
    main()
except KeyboardInterrupt:
    print("stopped by user")
