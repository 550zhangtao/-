# --- PC 端脚本守卫：误在 K230 上运行时给出明确提示 ---
try:
    import sys as _sys
    if getattr(_sys, "implementation", None) is not None and _sys.implementation.name == "micropython":
        print("[PC 脚本] 这是电脑端工具，请在 PC 上运行；板端请运行 k230/board/*.py")
        raise SystemExit
except SystemExit:
    raise
except Exception:
    pass
# 移植版（laser_repo.py）验收：静止 N 秒，打印检出率/坐标/std/lms
import time, math
from media.sensor import *
from media.media import *
import laser_repo as LR
W, H = 640, 480
SECS = 30
s = Sensor(id=2, fps=60); s.reset()
s.set_framesize(w=W, h=H, chn=CAM_CHN_ID_1); s.set_pixformat(Sensor.RGB565, chn=CAM_CHN_ID_1)
MediaManager.init(); s.run(); time.sleep(1.2)
try:
    for _ in range(10): s.snapshot(chn=CAM_CHN_ID_1)
    lf = LR.LaserRepo()
    n = 0; hit = 0; xs = []; ys = []; lms = []; areas = []
    t0 = time.ticks_ms()
    while time.ticks_diff(time.ticks_ms(), t0) < SECS * 1000:
        r = lf.sample(s.snapshot(chn=CAM_CHN_ID_1))
        n += 1; lms.append(lf.last_ms)
        if r:
            hit += 1; xs.append(r["x"]); ys.append(r["y"]); areas.append(r["area"])
    def std(v):
        if len(v) < 2: return 0.0
        m = sum(v) / len(v)
        return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
    print("RESULT frames=%d hits=%d rate=%.1f%%" % (n, hit, 100.0 * hit / n))
    print("RESULT x=%.2f std=%.2f | y=%.2f std=%.2f" % (
        sum(xs) / len(xs) if xs else -1, std(xs),
        sum(ys) / len(ys) if ys else -1, std(ys)))
    print("RESULT area mean=%.0f min=%.0f max=%.0f | lms mean=%.1f max=%d" % (
        sum(areas) / len(areas) if areas else 0,
        min(areas) if areas else 0, max(areas) if areas else 0,
        sum(lms) / len(lms), max(lms)))
finally:
    try: s.stop()
    except Exception: pass
    try: MediaManager.deinit()
    except Exception: pass
