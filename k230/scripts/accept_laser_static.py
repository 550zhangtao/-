import time, math
from media.sensor import *
from media.media import *
import laser_openmv as L
W,H=640,480
SECS = 30
s = Sensor(id=2, fps=60); s.reset()
s.set_framesize(w=W,h=H,chn=CAM_CHN_ID_1); s.set_pixformat(Sensor.RGB565, chn=CAM_CHN_ID_1)
MediaManager.init(); s.run()
for _ in range(8): s.snapshot(chn=CAM_CHN_ID_1)
lf = L.LaserSpot({"diag_mode":0})
lf.load("/sdcard/laser_calib.json")
n=0; hit=0; xs=[]; ys=[]; misses=0; maxmiss=0
t0=time.ticks_ms(); tlast=t0
fps_n=0
while time.ticks_diff(time.ticks_ms(), t0) < SECS*1000:
    img = s.snapshot(chn=CAM_CHN_ID_1)
    r = lf.sample(img)
    n += 1
    if r is not None:
        hit += 1; misses=0
        xs.append(float(r["x"])); ys.append(float(r["y"]))
    else:
        misses += 1
        if misses>maxmiss: maxmiss=misses
    if n % 100 == 0:
        dt = time.ticks_diff(time.ticks_ms(), tlast); tlast = time.ticks_ms()
        print("  ..%d frames, %.1ffps, hit %d/%d" % (n, 100*1000.0/dt, hit, n))
s.stop(); MediaManager.deinit()
def mean(v): return sum(v)/len(v) if v else 0.0
def std(v):
    if len(v)<2: return 0.0
    m=mean(v); return math.sqrt(sum((x-m)**2 for x in v)/(len(v)-1))
print("RESULT frames=%d hits=%d rate=%.1f%% maxmiss_run=%d fps=%.1f" % (
    n, hit, 100.0*hit/n if n else 0, maxmiss, n*1000.0/max(1,time.ticks_diff(time.ticks_ms(),t0))))
print("RESULT cx mean=%.1f std=%.2f  cy mean=%.1f std=%.2f  n=%d" % (
    mean(xs), std(xs), mean(ys), std(ys), len(xs)))
