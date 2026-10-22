# 极简版验收：ROI + cv_lite，静止 30s
import time, math
from media.sensor import *
from media.media import *
import laser_openmv as L
W,H=640,480
SECS = 30
s = Sensor(id=2, fps=60); s.reset()
s.set_framesize(w=W,h=H,chn=CAM_CHN_ID_0); s.set_pixformat(Sensor.GRAYSCALE, chn=CAM_CHN_ID_0)
MediaManager.init(); s.run()
for _ in range(8): s.snapshot(chn=CAM_CHN_ID_0)
lf = L.LaserSpot({})
n=0; hit=0; xs=[]; ys=[]; misses=0; maxmiss=0; lms=[]; raws=[]
t0=time.ticks_ms(); tlast=t0; win=0
while time.ticks_diff(time.ticks_ms(), t0) < SECS*1000:
    r = lf.sample(s.snapshot(chn=CAM_CHN_ID_0))
    n += 1; win += 1
    lms.append(lf.last_ms); raws.append(lf.raw_blobs)
    if r:
        hit += 1; misses = 0
        xs.append(r["x"]); ys.append(r["y"])
    else:
        misses += 1
        if misses > maxmiss: maxmiss = misses
    if n % 100 == 0:
        dt = time.ticks_diff(time.ticks_ms(), tlast); tlast = time.ticks_ms()
        print("  ..%d frames %.1ffps hit %d/%d lms=%.1fms raw=%d"
              % (n, 100*1000.0/dt, hit, n, lf.last_ms, lf.raw_blobs))
s.stop(); MediaManager.deinit()
def mean(v): return sum(v)/len(v) if v else 0.0
def std(v):
    if len(v)<2: return 0.0
    m=mean(v); return math.sqrt(sum((x-m)**2 for x in v)/(len(v)-1))
lr = sorted(lms)
print("RESULT frames=%d hits=%d rate=%.1f%% maxmiss=%d" % (n, hit, 100.0*hit/n if n else 0, maxmiss))
print("RESULT cx=%.2f std=%.2f | cy=%.2f std=%.2f" % (mean(xs), std(xs), mean(ys), std(ys)))
print("RESULT lms mean=%.2f p95=%.1f max=%.1f | raw mean=%.2f max=%d" % (
    mean(lms), lr[int(len(lr)*0.95)] if lr else 0, max(lms) if lms else 0,
    mean(raws), max(raws) if raws else 0))
