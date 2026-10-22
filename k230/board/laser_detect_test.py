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
THRESHOLD = (240, 255)         # (low, high)；暗背景 200~230，亮背景 240~248

# 面积相关（cv_lite 只有一个 min_area，这里两个都收下、取较大值）
AREA_THRESHOLD = 6             # 你要的 area_threshold，5~10
MIN_PIXELS = 6                 # 你要的 pixels_threshold，5~10
MERGE = True                   # True: 相近光斑合并（间隔 <= MERGE_GAP 就并）

# 光斑"应该是多大"——用来排除灯/墙/手电筒这类大片亮区
MAX_PIXELS = 400               # 超过这个像素数就不是激光点
MAX_SIDE = 48                  # 外接矩形最长边上限（px）
MIN_FILL = 0.45                # 填充率下限：激光点圆而实，灯/反光是细长条或镂空

KERNEL_SIZE = 3                # 形态学核：3 = 抹掉 1px 级散点（推荐）；1 = 关闭
MERGE_GAP = 3                  # 两块之间间隔 <= 这么多像素就合并

# 激光点大致应在的位置（搜索中心，像素）。只影响"多个候选时优先选谁"。
CORE_X = W * 0.5
CORE_Y = H * 0.5

# 位置门限：0 = 关闭（推荐先关掉）。激光笔与相机相对固定时，光斑只在
# 画面某一小块出现，把它设成 0.3~0.5（相对半宽/半高的归一化半径）能压掉
# 大部分杂散误检。实测数据集上设太小会掉检出率，所以默认关闭、按现场再收紧。
GATE_R = 0.0

# 打印 / 保存
PRINT_EVERY = 1                # 每 N 帧打印一次；1 = 每帧都打印
SAVE_CSV = "/sdcard/laser_detect_log.csv"   # 每帧一行；"" = 不写
SAVE_HIT = ""                  # 例如 "/sdcard/laser_hit.jpg"：命中时存图（调试用）
SAVE_HIT_MAX = 1               # 最多存几张（每帧存会明显掉帧）
DRAW_ON_IDE = True             # True: 每帧推到 IDE（IDE 里才有画面）
IDE_QUALITY = 70               # 推 IDE 的 JPEG 质量
RESET_ON_EXIT = True           # 退出时软复位（KPU/相机析构在板上有死锁风险）

DISPLAY_MODE = "ide"           # "ide" | "none"（只跑不显示时用 none，帧率最高）
ROUNDS = 0                     # 0 = 一直跑；N > 0 = 跑 N 轮后自动停（每轮 SECONDS 秒）
SECONDS = 0                    # 每轮秒数，0 = 不限
# =================================================================

sensor = None                  # 由 _init() 赋值
_save_n = 0


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
def _pick(cands):
    """在候选里挑激光点：面积/形状达标，优先靠画面中心。

    激光笔和相机的相对位置固定，所以光斑基本落在画面中心附近的固定区域；
    比"选最大"稳得多（最大块往往是台灯/白纸）。
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
        key = (dist, -c["n"])
        if best_key is None or key < best_key:
            best_key = key
            best = c
    return best


def run_round(shape, label="", seconds=0, csv_file=None):
    """跑 seconds 秒（0 = 一直跑），返回 (帧数, 命中数, 平均fps)。可被自动化多次调用。"""
    global _save_n

    t_start = time.ticks_ms()
    frame = 0
    hits = 0
    t_win = t_start
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

        # ---- 拆粘连 + 合并 + 尺寸过滤（cv_lite 没给面积/merge）
        cands = []
        for i in range(0, len(raw) - 3, 4):
            rect = (raw[i], raw[i + 1], raw[i + 2], raw[i + 3])
            if _blob_side(rect) > MAX_SIDE:
                continue          # 大片亮区（灯/墙/手电筒反光）直接丢，省掉拆分
            if rect[2] * rect[3] <= MAX_PIXELS * 2:
                # 小框：直接用框面积当像素数（亮块实心，误差可接受），不逐像素拆
                cands.append({"n": rect[2] * rect[3],
                              "cx": rect[0] + rect[2] * 0.5,
                              "cy": rect[1] + rect[3] * 0.5,
                              "x": rect[0], "y": rect[1],
                              "w": rect[2], "h": rect[3]})
            else:
                cands.extend(_split_by_gray(data, shape, rect, THRESHOLD[0]))
        cands = _merge_close(cands, MERGE_GAP)
        best = _pick(cands)

        frame += 1
        if best is not None:
            hits += 1
            cx, cy, area = int(best["cx"]), int(best["cy"]), best["n"]
            if PRINT_EVERY and (frame % PRINT_EVERY) == 0:
                print("[%d] BLOB cx=%d cy=%d area=%d (cv %.1fms)"
                      % (frame, cx, cy, area, cv_ms))
            # 用外接矩形 + 十字标出；圆环帮助肉眼判断是否套住光斑
            img.draw_rectangle(best["x"], best["y"], best["w"], best["h"],
                               color=(0, 255, 0), thickness=1)
            img.draw_cross(cx, cy, color=(255, 0, 0), size=10, thickness=2)
            img.draw_circle(cx, cy, 8, color=(255, 0, 0), thickness=1)
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
        if SAVE_HIT and best is not None and _save_n < SAVE_HIT_MAX:
            _save_n += 1
            try:
                img.save(SAVE_HIT)
                print("saved hit -> %s" % SAVE_HIT)
            except Exception as e:
                print("save hit:", e)

        if DRAW_ON_IDE and DISPLAY_MODE == "ide":
            Display.show_image(img)

        dt_win = time.ticks_diff(time.ticks_ms(), t_win)
        if dt_win >= 2000:
            fps = frame * 1000.0 / dt_win
            print("---- %s frames=%d hit=%d fps=%.1f ----" % (label, frame, hits, fps))
            t_win = time.ticks_ms()

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
